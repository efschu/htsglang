"""20.09.: the per-round chain policy (adaptive_chain) activates a runtime
state on every rank locally. In offload mode ensure_active runs the
SGLANG_ADAPTIVE_ALIAS_VERIFY_RANK_SYNC check after the swap; in resident
mode it returned before the check, so a rank-divergent activation would
have deadlocked in the next collective instead of failing loudly. The
controller now calls note_resident_activation on every activation."""
import unittest
from unittest.mock import patch

from sglang.srt.speculative.adaptive_graph_memory import AdaptiveGraphMemoryManager
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=2, suite="stage-a-cpu")


class TestResidentRankSync(unittest.TestCase):
    def test_resident_activation_runs_the_check_and_counts(self):
        mgr = AdaptiveGraphMemoryManager(mode="resident")
        calls = []
        with patch.object(
            mgr, "_maybe_verify_rank_sync", side_effect=lambda s: calls.append(s)
        ):
            mgr.note_resident_activation(3)
            mgr.note_resident_activation(1)
        self.assertEqual(calls, [3, 1])
        self.assertEqual(mgr.swap_count, 2)

    def test_offload_mode_leaves_it_to_ensure_active(self):
        # Constructing an offload manager needs the preload hook; flip the
        # mode on a resident one instead (offload_enabled reads self.mode).
        mgr = AdaptiveGraphMemoryManager(mode="resident")
        mgr.mode = "offload"
        calls = []
        with patch.object(
            mgr, "_maybe_verify_rank_sync", side_effect=lambda s: calls.append(s)
        ):
            mgr.note_resident_activation(3)
        self.assertEqual(calls, [])
        self.assertEqual(mgr.swap_count, 0)

    def test_check_is_a_noop_without_the_env(self):
        # No torch.distributed init here: the check must return before any
        # collective when the env is unset.
        mgr = AdaptiveGraphMemoryManager(mode="resident")
        with patch.dict("os.environ", {}, clear=False):
            import os

            os.environ.pop("SGLANG_ADAPTIVE_ALIAS_VERIFY_RANK_SYNC", None)
            mgr.note_resident_activation(2)
        self.assertEqual(mgr.swap_count, 1)


if __name__ == "__main__":
    unittest.main()
