# SPDX-License-Identifier: Apache-2.0
"""RANK-DEATH (27B port of the rank part of NF 62357f2ba1, test_pdflip_y4u_rw_eager_rank_death_0930).

NF y4u, D TP1 15:18:30Z: the rank died, state.json lifecycle stayed "serving" until DEADMAN_CRASH at
15:19:48 (78 s). The scheduler's exception handler now writes lifecycle ``dead`` (origin rank, code
RANK_EXCEPTION) through the one state writer (state_file.note_rank_death, writer "rank" that owns only
its death; first cause wins) -- after the #1223 hold, before the census and the SIGQUIT.

Only the rank half: NF's RW-eager half (expert_pool_device / expert offload) is NF-specific. The 27B
state_file.py is byte-identical to the host copy /spinning/gpu-arb/docker/acc_state.py, the union of
27B 7a000bbfc4 (PROGRESS-STALL) and NF 62357f2ba1 (rank) -- pinned by
test_pdflip_state_file_0928::TestHostCopy::test_host_copy_is_byte_identical.
"""
from __future__ import annotations

import inspect
import os
import tempfile
import unittest

from flliper.srt.pdflip import state_file as sf


class RankDeathReachesTheLifecycle(unittest.TestCase):
    def _serving(self, root):
        d = sf.init(root, "b1", "boot", {})
        sf.transition(d, "launching")
        sf.transition(d, "serving")
        return d

    def test_the_dying_rank_writes_dead_at_once(self):
        with tempfile.TemporaryDirectory() as root:
            d = self._serving(root)
            exc = IndexError("index 2147483647 is out of bounds for dimension 0 with size 128")
            self.assertTrue(sf.note_rank_death(d, group="D", tp_rank=1, pp_rank=0, exc=exc))
            st = sf.read(d)
            self.assertEqual(st["lifecycle"]["state"], "dead")
            c = st["cause"]
            self.assertEqual((c["code"], c["origin"], c["group"], c["rank"], c["exception_type"], c["rc"]),
                             ("RANK_EXCEPTION", "rank", "D", "tp1pp0", "IndexError", sf.RC_DEAD_AFTER_SERVING))
            self.assertEqual(sf.health(d)[0], sf.HEALTH_DEAD)
            self.assertTrue(any(e.get("type") == "lifecycle" and (e.get("data") or {}).get("state") == "dead"
                                for e in sf.events(d)))
            # the first cause wins: a second rank's death changes nothing
            self.assertFalse(sf.note_rank_death(d, group="D", tp_rank=2, pp_rank=0, exc=RuntimeError("x")))
            self.assertEqual(sf.read(d)["cause"]["rank"], "tp1pp0")

    def test_no_state_dir_or_state_nothing_and_never_raises(self):
        self.assertFalse(sf.note_rank_death(None, group="D", tp_rank=0, pp_rank=0, exc=RuntimeError()))
        with tempfile.TemporaryDirectory() as root:
            self.assertFalse(sf.note_rank_death(root, group="D", tp_rank=0, pp_rank=0, exc=RuntimeError()))

    def test_the_rank_writer_owns_only_its_death(self):
        with tempfile.TemporaryDirectory() as root:
            d = self._serving(root)
            with self.assertRaises(sf.StateFileError):
                sf.transition(d, "stopping", writer="rank")
            with self.assertRaises(sf.StateFileError):
                sf.transition(d, "dead", cause=sf.make_cause("X", "front"), writer="rank")

    def test_the_scheduler_writes_it_after_the_hold_before_the_sigquit(self):
        from flliper.srt.managers import scheduler

        src = inspect.getsource(scheduler.run_scheduler_process)
        i = src.index("Scheduler hit an exception")
        hold = src.index("debug_hold.maybe_hold(", i)
        death = src.index("note_rank_death(", i)
        sigquit = src.index("SIGQUIT", death)
        self.assertLess(hold, death)
        self.assertLess(death, sigquit)


if __name__ == "__main__":
    unittest.main()
