"""fnFL2 H36: per-request acceptance profile (SPEC-ACCEPT-PROFILE).

Pins (a) the head/rest split of the correct-drafts histogram and the
accept_length arithmetic (bonus included), (b) the wiring into the batch
result processor -- the head histogram is fed from the same verify-round loop
that feeds ``Req.spec_correct_drafts_histogram``, and the line is written when
a request finishes -- and (c) the off switch and the draft-cold mark.
"""

import logging
import unittest
from types import SimpleNamespace
from unittest import mock

import torch

from sglang.srt.environ import envs
from sglang.srt.managers.schedule_batch import FINISH_LENGTH, Req
from sglang.srt.managers.scheduler_components import (
    batch_result_processor as brp,
)
from sglang.srt.managers.scheduler_components.batch_result_processor import (
    SchedulerBatchResultProcessor,
)
from sglang.srt.sampling.sampling_params import SamplingParams
from sglang.srt.speculative import accept_profile as ap
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class _FakeSpecAlgorithm:
    def is_none(self) -> bool:
        return False

    def is_dflash(self) -> bool:
        return False


class _FakeBatch:
    def __init__(self, reqs):
        self.reqs = reqs
        self.spec_algorithm = _FakeSpecAlgorithm()


def _make_processor() -> SchedulerBatchResultProcessor:
    return SchedulerBatchResultProcessor(
        is_generation=True,
        disaggregation_mode=None,
        enable_overlap=False,
        enable_overlap_mlx=False,
        server_args=SimpleNamespace(
            enable_metrics=False,
            disaggregation_decode_enable_offload_kvcache=False,
            enable_hisparse=False,
        ),
        model_config=SimpleNamespace(think_end_id=None),
        token_to_kv_pool_allocator=None,
        tree_cache=None,
        hisparse_coordinator=None,
        req_to_token_pool=None,
        decode_offload_manager=None,
        metrics_collector=None,
        metrics_reporter=SimpleNamespace(),
        draft_worker=None,
        model_worker=SimpleNamespace(on_verify_complete_cpu=lambda *a, **k: None),
        logprob_result_processor=None,
        output_streamer=SimpleNamespace(),
        abort_request=lambda *a, **k: None,
    )


def _make_req(rid: str = "r0") -> Req:
    sp = SamplingParams(max_new_tokens=4096, temperature=0)
    sp.normalize(None)
    req = Req(
        rid=rid,
        origin_input_text="",
        origin_input_ids=list(range(100)),
        sampling_params=sp,
    )
    req.kv_committed_len = 0
    return req


def _round_result(num_correct_drafts: int, stride: int = 4):
    accept_len = num_correct_drafts + 1
    flat = list(range(500, 500 + accept_len)) + [0] * (stride - accept_len)
    return SimpleNamespace(
        next_token_ids=torch.tensor(flat, dtype=torch.long),
        accept_lens=torch.tensor([accept_len], dtype=torch.long),
        speculative_num_draft_tokens=stride,
        num_correct_drafts=None,
        num_correct_drafts_per_req_cpu=[num_correct_drafts],
        block_accept_lens=None,
        cap_lens=None,
    )


def _run_rounds(proc, req, seq):
    for k in seq:
        proc._resolve_spec_v2_tokens(_round_result(k), _FakeBatch([req]))


class TestArithmetic(CustomTestCase):
    def test_accept_length_includes_bonus(self):
        # 1 round with 0 correct drafts (1 token), 3 rounds with 3 (4 tokens).
        self.assertAlmostEqual(ap.accept_length([1, 0, 0, 3]), 13 / 4)
        self.assertIsNone(ap.accept_length([0, 0]))

    def test_rest_is_total_minus_head(self):
        self.assertEqual(ap.rest_histogram([5, 2, 7, 9], [1, 1]), [4, 1, 7, 9])
        self.assertEqual(ap.rest_histogram([2], [1, 0, 0, 1]), [1, 0, 0, -1])

    def test_head_window_stops_at_n(self):
        req = SimpleNamespace(spec_verify_ct=0)
        for _ in range(10):
            req.spec_verify_ct += 1
            ap.note_round(req, 2, head=4)
        self.assertEqual(getattr(req, ap.HEAD_HISTOGRAM_ATTR), [0, 0, 4])

    def test_cold_attr_is_the_bootstrap_constant(self):
        from sglang.srt.managers import phase_flip_draft_bootstrap as pfdb

        self.assertEqual(ap.DRAFT_COLD_ATTR, pfdb.COLD_ARMED_ATTR)


class TestProcessorWiring(CustomTestCase):
    def test_head_histogram_fed_by_verify_loop(self):
        proc = _make_processor()
        req = _make_req()
        seq = [0, 1, 3, 3] + [3, 2] * 3
        with envs.SGLANG_LOG_SPEC_ACCEPT_PROFILE_HEAD_ROUNDS.override(4):
            _run_rounds(proc, req, seq)
        self.assertEqual(req.spec_verify_ct, len(seq))
        self.assertEqual(req.spec_correct_drafts_histogram, [1, 1, 3, 5])
        self.assertEqual(getattr(req, ap.HEAD_HISTOGRAM_ATTR), [1, 1, 0, 2])

        line = ap.format_profile(req, 4)
        self.assertIn("rounds=10 ", line)
        self.assertIn("head4 rounds=4 accept_length=2.75 hist=[1, 1, 0, 2]", line)
        # rest: 3x3 + 3x2 correct drafts -> (3*4 + 3*3) / 6 = 3.50
        self.assertIn("rest rounds=6 accept_length=3.50 hist=[0, 0, 3, 3]", line)
        self.assertIn("share=0:0.100/1:0.100/2:0.300/3:0.500", line)
        self.assertIn("draft_cold=no", line)

    def test_off_switch_keeps_no_head(self):
        proc = _make_processor()
        req = _make_req()
        with envs.SGLANG_LOG_SPEC_ACCEPT_PROFILE_HEAD_ROUNDS.override(0):
            _run_rounds(proc, req, [3, 3])
            self.assertIsNone(ap.log_finished(req))
        self.assertFalse(hasattr(req, ap.HEAD_HISTOGRAM_ATTR))
        # the stock histogram is untouched by the switch
        self.assertEqual(req.spec_correct_drafts_histogram, [0, 0, 0, 2])

    def _finish(self, proc, req):
        req.finished_reason = FINISH_LENGTH(length=1)
        with mock.patch.object(
            SchedulerBatchResultProcessor,
            "_mamba_prefix_cache_update",
            lambda *a, **k: None,
        ), mock.patch.object(brp, "release_kv_cache", lambda *a, **k: None), mock.patch.object(
            brp,
            "get_server_args",
            lambda: SimpleNamespace(enable_mamba_extra_buffer_lazy=lambda: False),
        ), mock.patch.object(
            brp, "get_global_indexer_capturer", lambda: None
        ):
            proc._handle_finish_state_updated_req(req, None, None, 0, None)

    def test_finish_writes_the_line(self):
        proc = _make_processor()
        req = _make_req("weg2-2-6")
        setattr(req, ap.DRAFT_COLD_ATTR, True)
        with envs.SGLANG_LOG_SPEC_ACCEPT_PROFILE_HEAD_ROUNDS.override(2):
            _run_rounds(proc, req, [0, 3, 3, 1])
            with self.assertLogs(ap.logger, level=logging.INFO) as logs:
                self._finish(proc, req)
        text = "\n".join(logs.output)
        self.assertIn("SPEC-ACCEPT-PROFILE rid=weg2-2-6 draft_cold=yes prompt=100", text)
        self.assertIn("head2 rounds=2 accept_length=2.50 hist=[1, 0, 0, 1]", text)
        self.assertIn("rest rounds=2 accept_length=3.00 hist=[0, 1, 0, 1]", text)

    def test_no_line_without_verify_rounds(self):
        proc = _make_processor()
        req = _make_req()
        with envs.SGLANG_LOG_SPEC_ACCEPT_PROFILE_HEAD_ROUNDS.override(64):
            with mock.patch.object(ap, "log_finished") as spy:
                self._finish(proc, req)
        spy.assert_not_called()


if __name__ == "__main__":
    unittest.main()
