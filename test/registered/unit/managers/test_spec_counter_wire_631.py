# SPDX-License-Identifier: Apache-2.0
"""#631: the spec counters survive a request that FINISHES in the PP phase.

THE DEFECT, measured on metal 2026-08-09. A phase-flip instance rests in
the PP prefill layout and speculates only in the TP decode layout. Under
POLICY=auto a request routinely verifies in TP and then finishes after the
return flip, in PP -- observed directly, `phase_before tp phase_after pp`.

The accumulator gated its spec-counter appends on the LIVE phase
(`spec_algorithm.is_none()`), so those requests had their counters
dropped on the way out: the numbers were sitting on the Req, and the
phase that happened to be active at completion decided they would not be
shipped. `meta_info` therefore carried no `spec_accept_length` while the
scheduler log printed `accept len: 3.20` for the very same traffic.

Two properties are pinned here, and the second is the one that makes the
first safe:

  * a request that speculated ships its counters no matter which phase it
    finishes in, and
  * the lists stay BATCH-ALIGNED, because tokenizer_manager indexes them
    by request position. Appending for some requests and not others is
    what made the defensive `len(...) > i` guard necessary there.
"""

import unittest
from types import SimpleNamespace

from sglang.srt.disaggregation.utils import DisaggregationMode
from sglang.srt.managers.scheduler_components.output_streamer import (
    _GenerationStreamAccumulator,
)
from sglang.srt.speculative.spec_info import SpeculativeAlgorithm
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=1, suite="base-a-test-cpu")


class _FakeReq:
    def __init__(self, rid, output_ids, verify_ct=0, correct_drafts=0):
        self.rid = rid
        self.http_worker_ipc = None
        self.finished_reason = None
        self.finished_output = False
        self.finished_len = None
        self.stream = False
        self.sampling_params = SimpleNamespace(
            stream_interval=None,
            skip_special_tokens=True,
            spaces_between_special_tokens=True,
            no_stop_trim=False,
        )
        self.output_ids = output_ids
        self.output_ids_through_stop = output_ids
        self.send_token_offset = 0
        self.send_output_token_logprobs_offset = 0
        self.send_decode_id_offset = 0
        self.decoded_text = ""
        self.origin_input_ids = []
        self.reasoning_tokens = 0
        self.cached_tokens = 0
        self.retraction_count = 0
        self.time_stats = None
        self.mm_image_tokens = 0
        self.mm_audio_tokens = 0
        self.mm_video_tokens = 0
        self.multimodal_inputs = None
        self.customized_info = None
        # What the TP phase accumulated before the return flip.
        self.spec_verify_ct = verify_ct
        self.spec_num_correct_drafts = correct_drafts
        self.spec_num_block_accept_tokens = 0
        self.spec_num_cap_tokens = 0
        self.spec_correct_drafts_histogram = None
        self.spec_cap_lens_histogram = None

    def finished(self):
        return False

    def init_incremental_detokenize(self):
        return self.output_ids_through_stop, 0

    def check_match_stop_str_prefix(self):
        return False


def _acc(*, spec_configured, spec_algorithm):
    return _GenerationStreamAccumulator(
        return_logprob=False,
        return_hidden_states=False,
        return_routed_experts=False,
        return_indexer_topk=False,
        spec_algorithm=spec_algorithm,
        spec_configured=spec_configured,
        disaggregation_mode=DisaggregationMode.NULL,
        default_stream_interval=1,
        default_force_stream_interval=1,
        get_cached_tokens_details=lambda req: None,
    )


class TestSpecCounterWireSurvivesThePpFinish(unittest.TestCase):
    def test_counters_ship_when_the_finishing_phase_carries_no_speculation(self):
        """The measured case: verified in TP, finished in PP."""
        acc = _acc(
            spec_configured=True,
            # PP phase: the live algorithm is NONE at completion time.
            spec_algorithm=SpeculativeAlgorithm.NONE,
        )
        acc.accept(req=_FakeReq("r0", [1, 2, 3], verify_ct=4, correct_drafts=7))
        payload = acc.to_payload(dp_rank=0, is_idle_batch=False)
        self.assertEqual(payload.spec_verify_ct, [4])
        self.assertEqual(payload.spec_num_correct_drafts, [7])

    def test_lists_stay_batch_aligned_across_a_mixed_batch(self):
        """Alignment is what lets tokenizer_manager index by position.

        A batch holding one request that speculated and two that did not
        must still produce one entry PER REQUEST, with 0 for the ones that
        never verified -- that zero is what the `> 0` check downstream
        keys on.
        """
        acc = _acc(spec_configured=True, spec_algorithm=SpeculativeAlgorithm.NONE)
        acc.accept(req=_FakeReq("r0", [1], verify_ct=0))
        acc.accept(req=_FakeReq("r1", [2, 3], verify_ct=5, correct_drafts=9))
        acc.accept(req=_FakeReq("r2", [4], verify_ct=0))
        payload = acc.to_payload(dp_rank=0, is_idle_batch=False)
        self.assertEqual(len(payload.spec_verify_ct), 3)
        self.assertEqual(payload.spec_verify_ct, [0, 5, 0])
        self.assertEqual(
            len(payload.rids),
            len(payload.spec_verify_ct),
            "one counter entry per request, or downstream indexing is wrong",
        )

    def test_can_fail_a_phase_gated_accumulator_drops_them(self):
        """The old behaviour, reproduced by turning the predicate off.

        This is what the live-phase gate did on every PP-finishing
        request: silent, answer-preserving, evidence-destroying.
        """
        acc = _acc(spec_configured=False, spec_algorithm=SpeculativeAlgorithm.NONE)
        acc.accept(req=_FakeReq("r0", [1, 2, 3], verify_ct=4, correct_drafts=7))
        payload = acc.to_payload(dp_rank=0, is_idle_batch=False)
        self.assertEqual(
            payload.spec_verify_ct,
            [],
            "with the predicate off the counters are dropped, which is the "
            "regression this test exists to catch",
        )

    def test_non_speculating_server_is_byte_identical(self):
        """A server without speculation must not start shipping lists.

        The predicate is keyed on server config precisely so this path is
        unchanged: no speculative_algorithm, no counters, exactly as
        before.
        """
        acc = _acc(spec_configured=False, spec_algorithm=SpeculativeAlgorithm.NONE)
        acc.accept(req=_FakeReq("r0", [1]))
        acc.accept(req=_FakeReq("r1", [2]))
        payload = acc.to_payload(dp_rank=0, is_idle_batch=False)
        self.assertEqual(payload.spec_verify_ct, [])
        self.assertEqual(payload.spec_num_correct_drafts, [])


if __name__ == "__main__":
    unittest.main()
