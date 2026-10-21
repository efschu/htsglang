"""BP-ANCHOR review (42fcfed1a3): the host's spec track boundary must count the run the round RETAINED.

The device commit (``spec_utils._verify_commit_step_indices`` / ``commit_mamba_states_after_verify``) decides a
track crossing from ``seq_lens`` (= kv_committed_len before the round, P) and the FULL ``accept_lens`` (a):
crossing iff ``P // I != (P + a) // I``. The host (``_mamba_check_track_boundary``) reconstructs P as
``cur - (num_correct_drafts + 1)`` with ``cur = req.seqlen - 1``. When a grammar terminates inside the accepted
run, ``_resolve_spec_v2_tokens`` keeps only k < a tokens (``_accept_grammar_tokens``) -- ``output_ids`` and
``kv_committed_len`` grow by k, so ``cur = P + k`` and the host's P is ``P + k - a`` < P.

If the previous round ended exactly on a multiple m (or within a - k - 1 above it), the host then claims the
crossing at m a SECOND time: it swaps the ping-pong slot again while the device wrote nothing, so the finish
donates the OTHER slot (an older state) labelled m -- a mamba anchor whose state is not the state at m.
With the page grid (64 instead of 256) the exposure per grammar-finished request is 4x higher.

Fix: the host uses the retained count k of this round (``GenerationBatchResult.num_retained_per_req_cpu``):
P = cur - k, crossing iff a multiple lies in (P, P + k] -- where the device crossed too (a < I), and a device
crossing above the retained end (P + k, P + a] is not adopted (its key would run past output_ids).
"""

import unittest
from types import SimpleNamespace
from unittest import mock

I = 64


def _check(cur, correct, retained=None):
    from flliper.srt.managers.scheduler_components.batch_result_processor import (
        SchedulerBatchResultProcessor,
    )

    sa = SimpleNamespace(mamba_track_interval=I)
    batch = SimpleNamespace(spec_algorithm=SimpleNamespace(is_none=lambda: False))
    req = SimpleNamespace(seqlen=cur + 1, kv_committed_len=cur)
    res = SimpleNamespace(
        num_correct_drafts_per_req_cpu=[correct],
        num_retained_per_req_cpu=None if retained is None else [retained],
    )
    with mock.patch(
        "flliper.srt.managers.scheduler_components.batch_result_processor.get_server_args",
        return_value=sa,
    ):
        return SchedulerBatchResultProcessor._mamba_check_track_boundary(None, req, batch, res, 0)


def _device_crosses(pre, accept):
    return pre // I != (pre + accept) // I, (pre + accept) // I * I


class TestGrammarTrimmedRoundTrackBoundary(unittest.TestCase):
    def test_untrimmed_round_unchanged(self):
        # P = 1278, a = 4 -> device crosses 1280; host (k = a) claims 1280
        self.assertEqual(_check(1282, 3, retained=4), (True, 1280))
        self.assertEqual(_check(1282, 3), (True, 1280))   # no retained list: the old rule
        self.assertEqual(_device_crosses(1278, 4), (True, 1280))

    def test_trimmed_round_after_exact_boundary_claims_nothing(self):
        # previous round ended at P = 1280 (crossed, swapped); this round accepts a = 4, grammar keeps k = 1
        pre, a, k = 1280, 4, 1
        self.assertEqual(_device_crosses(pre, a), (False, 1280))     # device writes no track slot
        hit, _ = _check(pre + k, a - 1, retained=k)
        self.assertFalse(hit, "a second swap at 1280 would donate the stale ping-pong slot as the 1280 anchor")

    def test_trimmed_round_crossing_inside_the_retained_run_is_kept(self):
        # P = 1279, a = 4, k = 2 -> 1280 lies in (1279, 1281]: both sides cross at 1280
        self.assertEqual(_device_crosses(1279, 4), (True, 1280))
        self.assertEqual(_check(1281, 3, retained=2), (True, 1280))

    def test_trimmed_round_crossing_above_the_retained_end_is_not_adopted(self):
        # P = 1277, a = 4, k = 1 -> device crosses 1280 > P + k = 1278: the key would run past output_ids
        self.assertEqual(_device_crosses(1277, 4), (True, 1280))
        hit, _ = _check(1278, 3, retained=1)
        self.assertFalse(hit)

    def test_resolve_records_the_retained_run_and_the_host_follows_the_device(self):
        """END TO END over the real result path: ``_resolve_spec_v2_tokens`` with a grammar that terminates on
        the first accepted token, then the real boundary check. The previous round ended exactly on 1280."""
        import torch

        from flliper.srt.managers.scheduler_components.batch_result_processor import (
            SchedulerBatchResultProcessor,
        )

        class _Grammar:
            def __init__(self):
                self.n = 0

            def accept_token(self, t):
                self.n += 1

            def is_terminated(self):
                return self.n >= 1

        pre, a = 1280, 4
        req = SimpleNamespace(
            is_retracted=False, grammar=_Grammar(), kv_committed_len=pre, spec_verify_ct=0,
            spec_num_correct_drafts=0, update_spec_correct_drafts_histogram=lambda n: None,
            spec_num_block_accept_tokens=0, spec_num_cap_tokens=0, origin_input_ids=list(range(pre)),
            output_ids=[7], rid="g",
        )
        req.finished = lambda: False
        result = SimpleNamespace(
            next_token_ids=torch.tensor([11, 12, 13, 14]), accept_lens=torch.tensor([a]),
            block_accept_lens=None, cap_lens=None, speculative_num_draft_tokens=a,
        )
        proc = SimpleNamespace(model_worker=SimpleNamespace(on_verify_complete_cpu=lambda *x, **k: None))
        proc._accept_grammar_tokens = lambda r, t: SchedulerBatchResultProcessor._accept_grammar_tokens(proc, r, t)
        batch = SimpleNamespace(reqs=[req], spec_algorithm=SimpleNamespace(is_none=lambda: False))
        with mock.patch(
            "flliper.srt.managers.scheduler_components.batch_result_processor.accept_profile"
        ):
            toks = SchedulerBatchResultProcessor._resolve_spec_v2_tokens(proc, result, batch)
        self.assertEqual(toks, [[11]])
        req.output_ids.extend(toks[0])
        req.seqlen = len(req.origin_input_ids) + len(req.output_ids)
        self.assertEqual(req.seqlen - 1, pre + 1)
        self.assertEqual(_device_crosses(pre, a), (False, 1280))
        sa = SimpleNamespace(mamba_track_interval=I)
        with mock.patch(
            "flliper.srt.managers.scheduler_components.batch_result_processor.get_server_args",
            return_value=sa,
        ):
            hit, _ = SchedulerBatchResultProcessor._mamba_check_track_boundary(None, req, batch, result, 0)
        self.assertFalse(hit, "host claimed a crossing the device never wrote: the finish would donate a stale slot")
        self.assertEqual(getattr(result, "num_retained_per_req_cpu", None), [1])

if __name__ == "__main__":
    unittest.main()
