"""fnFL2 H72: a Form A expert worker must skip every logprob dereference.

x171 (24.09., 23:13:22Z) died on D the moment the ReplaySSM precision gate
sent its greedy probe with ``logprobs=True, top_logprobs=1``: TP1 and TP2 --
the Form A expert workers, whose forward is the MoE route only -- raised

    AttributeError: 'NoneType' object has no attribute 'next_token_logprobs'

in ``move_logprobs_to_cpu``. tp_worker and the spec verify already treat a
Form A worker like a weightless worker (``is_weightless_worker or
is_form_a_worker``: no logits, the host samples and broadcasts); the batch
result processor asked only ``_is_weightless_worker()`` and so walked into
``result.logits_output is None`` on any request that asked for logprobs. Every
client that sets ``logprobs`` would have taken NF-D down the same way.

Hermetic (no CUDA, no scheduler boot): drives the REAL
``process_batch_result_prefill`` with a Form A worker's model runner, a
logprob request and ``logits_output=None`` -- exactly the x171 inputs.
"""

import ast
import pathlib
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

from sglang.srt.managers.schedule_batch import Req
from sglang.srt.managers.scheduler_components.batch_result_processor import (
    SchedulerBatchResultProcessor,
)
from sglang.srt.sampling.sampling_params import SamplingParams
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=3, suite="base-a-test-cpu")

_PROCESSOR = (
    pathlib.Path(__file__).resolve().parents[4]
    / "python"
    / "sglang"
    / "srt"
    / "managers"
    / "scheduler_components"
    / "batch_result_processor.py"
)


def _make_processor(model_worker) -> SchedulerBatchResultProcessor:
    return SchedulerBatchResultProcessor(
        is_generation=True,
        disaggregation_mode=None,
        enable_overlap=False,
        enable_overlap_mlx=False,
        server_args=SimpleNamespace(enable_metrics=False, enable_hisparse=False),
        model_config=SimpleNamespace(think_end_id=None),
        token_to_kv_pool_allocator=None,
        tree_cache=None,
        hisparse_coordinator=None,
        req_to_token_pool=None,
        decode_offload_manager=None,
        metrics_collector=None,
        metrics_reporter=SimpleNamespace(report_prefill_stats=lambda **k: None),
        draft_worker=None,
        model_worker=model_worker,
        logprob_result_processor=None,
        output_streamer=SimpleNamespace(stream_output=lambda *a, **k: None),
        abort_request=lambda *a, **k: None,
        record_first_token_progress=lambda: None,
    )


def _make_logprob_req(rid: str) -> Req:
    sp = SamplingParams(max_new_tokens=256, temperature=0)
    sp.normalize(None)
    return Req(
        rid=rid,
        origin_input_text="",
        origin_input_ids=[1, 2, 3],
        sampling_params=sp,
        vocab_size=32000,
        return_logprob=True,
        top_logprobs_num=1,
    )


class _LogprobBatch:
    def __init__(self, reqs):
        self.reqs = reqs
        self.return_logprob = True
        self.decoding_reqs = []
        self.prefill_stats = None
        self.dp_cooperation_info = None


class _WorkerResult:
    """What a Form A worker's forward hands the processor: sampled ids from
    the host broadcast, no logits at all."""

    def __init__(self, next_token_ids):
        self.copy_done = None
        self.routed_experts_output = None
        self.indexer_topk_output = None
        self.logits_output = None
        self.next_token_ids = torch.tensor(next_token_ids, dtype=torch.long)
        self.extend_input_len_per_req = None
        self.extend_logprob_start_len_per_req = None
        self.can_run_cuda_graph = False


def _form_a_worker(spec: bool):
    runner = SimpleNamespace(is_weightless_worker=False, is_form_a_worker=True)
    if spec:
        # Under MTP the scheduler's model_worker is the spec worker; the
        # target runner sits behind target_worker (no model_runner of its own).
        return SimpleNamespace(target_worker=SimpleNamespace(model_runner=runner))
    return SimpleNamespace(model_runner=runner)


@patch(
    "sglang.srt.managers.scheduler_components.batch_result_processor."
    "maybe_cache_unfinished_req"
)
class TestFormAWorkerPrefillWithLogprobs(CustomTestCase):
    def _drive(self, model_worker):
        req = _make_logprob_req("weg2-gate")
        proc = _make_processor(model_worker)
        proc.process_batch_result_prefill(
            _LogprobBatch([req]), _WorkerResult(next_token_ids=[999])
        )
        return req

    def test_x171_inputs_no_longer_raise_on_the_spec_worker(self, _cache):
        req = self._drive(_form_a_worker(spec=True))
        self.assertEqual(list(req.output_ids), [999])

    def test_x171_inputs_no_longer_raise_on_a_plain_tp_worker(self, _cache):
        req = self._drive(_form_a_worker(spec=False))
        self.assertEqual(list(req.output_ids), [999])

    def test_the_host_still_dereferences_logits(self, _cache):
        """CAN-FAIL guard: the skip must stay worker-only. The host (TP0) has
        is_form_a_worker False; handed logits_output=None it must still walk
        into the logprob path -- a predicate that skipped logprobs everywhere
        would silently drop them from every response."""
        host = SimpleNamespace(
            model_runner=SimpleNamespace(
                is_weightless_worker=False, is_form_a_worker=False
            )
        )
        with self.assertRaises(AttributeError):
            self._drive(host)


class TestFormAWorkerPredicate(CustomTestCase):
    @staticmethod
    def _predicate(model_worker):
        stub = _make_processor(model_worker)
        return stub._is_form_a_worker()

    def test_plain_and_spec_worker(self):
        self.assertTrue(self._predicate(_form_a_worker(spec=False)))
        self.assertTrue(self._predicate(_form_a_worker(spec=True)))

    def test_default_path_without_the_attribute_is_false(self):
        self.assertFalse(self._predicate(SimpleNamespace()))
        self.assertFalse(self._predicate(SimpleNamespace(target_worker=None)))


class TestBothResultPathsAskForTheFormARole(CustomTestCase):
    """Decode has the same guard (``_normalize_decode_outputs`` and the
    per-request logprob / hidden-state blocks read ``wl_worker``); pin that
    both assignments include the Form A role, so a later edit of one path
    cannot reopen the x171 crash on the other."""

    def test_prefill_and_decode_assign_wl_worker_with_the_form_a_role(self):
        tree = ast.parse(_PROCESSOR.read_text())
        found = {}
        for cls in (n for n in tree.body if isinstance(n, ast.ClassDef)):
            for fn in (n for n in cls.body if isinstance(n, ast.FunctionDef)):
                for node in ast.walk(fn):
                    if (
                        isinstance(node, ast.Assign)
                        and len(node.targets) == 1
                        and isinstance(node.targets[0], ast.Name)
                        and node.targets[0].id == "wl_worker"
                    ):
                        found[fn.name] = ast.unparse(node.value)
        for name in ("process_batch_result_prefill", "process_batch_result_decode"):
            self.assertIn(name, found, f"{name} lost its wl_worker assignment")
            self.assertIn("_is_form_a_worker()", found[name], found[name])
            self.assertIn("logits_output is None", found[name], found[name])


if __name__ == "__main__":
    unittest.main()
