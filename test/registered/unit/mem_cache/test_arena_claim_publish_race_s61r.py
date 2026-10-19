# SPDX-License-Identifier: Apache-2.0
"""S61r (review of S61 a0766cc02d, NF rc15 K21 09:24:27/28): the join is committed under the slot lock only on a
slot that is CLAIMED for the joiner's key. Two orders S61 left open, both from the same rc15 interleaving
(TP0's #231 reap frees the orphan, then a rank claims the SAME key fresh -- with a full arena on exactly the
slot the reap freed):

  1. claim_slot made the slot CLAIMED (CAS FREE->CLAIMED) BEFORE it wrote the header. A joiner that found the
     orphan before the reap and is preempted until then sees CLAIMED + its own key (the new claimant wrote the
     key words) while generation and writer word are still the old ones: it joins with a generation the new
     claim bumps right after, and the writer word is reset to 0 under it -> completion 3 'recycled under the
     writer', A2 orphan on that rank only: the K21 class S61 closes, by another door.
  2. A fresh claim that loses the index publication to another claimant of the same key yields its slot
     (state FREE, no generation move). A joiner that committed on it in between (CLAIMED, key, generation all
     current) writes into a FREE slot -> completion 4.

Fix: claim_slot takes the slot privately (S_REAPING, which no freer, finder or joiner takes) and makes it CLAIMED
only together with its index publication, under the index lock. A joiner sees either the finished, published
claim or a slot that is not CLAIMED (then it looks the key up again).

Plus the freer side S61 could not drive deterministically ("Nachpruefung der Freigeber"): a freer's CAS lands
while the joiner holds the slot lock and has decided; the freer's re-check must wait for the lock and see the
joiner's writer (reap_partial and release_claims).

Real C arena on a temp file; every interleaving driven through test hooks (NULL in production).
"""

import ctypes
import os
import threading
import time
import unittest

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

from test_arena_join_reap_race_k21_s61 import HOOK_T, S_CLAIMED, S_COMPLETE, _ArenaCase  # noqa: E402

CLAIM_HOOK_T = ctypes.CFUNCTYPE(None, ctypes.c_int64, ctypes.c_int64)
S_EVICTING = 3


class _RaceCase(_ArenaCase):
    SLOTS = 5

    def setUp(self):
        super().setUp()
        self.lib.arena_set_claim_hook.restype = None
        self.lib.arena_set_claim_hook.argtypes = [ctypes.c_void_p]
        self.lib.arena_set_join_locked_hook.restype = None
        self.lib.arena_set_join_locked_hook.argtypes = [ctypes.c_void_p]
        self._refs = []

    def tearDown(self):
        self.lib.arena_set_claim_hook(None)
        self.lib.arena_set_join_locked_hook(None)
        super().tearDown()

    def _layout(self):
        """slot 0 = the orphan K (CLAIMED, open 0), slots 1..3 busy (open writers, never reaped), slot 4 free with
        the clock hand on 0 -- the claim after the reap takes the reaped slot, the next one slot 4."""
        a, _g = self._orphan("K")
        busy = [self.arena.claim_slots([f"busy{i}"], [self.SLOT])[0] for i in range(3)]
        (f, st, g), = self.arena.claim_slots(["spare"], [self.SLOT])
        assert st == 0 and self.arena.release_claims([f], [g]) == [0]
        assert a == 0 and [b[0] for b in busy] == [1, 2, 3] and f == 4, (a, busy, f)
        return a, f

    def _publish_race(self, phase, third_claims_first):
        a, f = self._layout()
        ev_paused, ev_go = threading.Event(), threading.Event()
        paused, res = [], {}

        def _claim_cb(slot, ph):
            if threading.current_thread().name != "B" or paused or int(ph) != phase:
                return
            paused.append(int(slot))
            ev_paused.set()
            ev_go.wait(10)

        cref = CLAIM_HOOK_T(_claim_cb)
        self._refs.append(cref)
        self.lib.arena_set_claim_hook(ctypes.cast(cref, ctypes.c_void_p))

        def _rank_b():   # TP0: its reap makes room, then it claims the key fresh (the arena's only free slot)
            res["freed"] = self.arena.reap_partial(0.0)
            res["b"] = self.arena.claim_slots(["K"], [self.SLOT])[0]

        tb = threading.Thread(target=_rank_b, name="B", daemon=True)

        def _at_join(slot):   # TP1 found the orphan, is preempted: TP0 reaps and claims, paused mid-claim
            tb.start()
            assert ev_paused.wait(10), "B never reached the claim point"
            if third_claims_first:   # TP2 claims the key fresh meanwhile and publishes first
                res["c"] = self.arena.claim_slots(["K"], [self.SLOT])[0]

        self._hook(_at_join)
        res["a"] = self._bounded(lambda: self.arena.claim_slots(["K"], [self.SLOT])[0], "the join hung")
        ev_go.set()
        tb.join(10)
        self.assertFalse(tb.is_alive(), "B's claim hung")
        self.assertEqual(res["freed"], [a], "the reap freed the orphan (no open writer)")
        self.assertEqual(paused, [a], "B's fresh claim took the reaped slot -- the rc15 arena-full case")
        return a, f, res

    def _all_land(self, claims):
        """every rank holds the one slot the key names, and every completion lands (never 3/4)"""
        (slot, state), = self.arena.find_slots(["K"])
        self.assertEqual(state, S_CLAIMED)
        for name, (s, st, _g) in claims.items():
            self.assertEqual(s, slot, f"{name} is on the key's slot, not on a recycled/yielded one")
            self.assertIn(st, (0, 1), name)
        self.assertEqual(sorted(st for _s, st, _g in claims.values()).count(0), 1, "exactly one fresh claimant")
        self.assertEqual(self._open(slot), len(claims), "every rank's writer is open on it")
        n = len(claims)
        part = self.SLOT // n
        outs = []
        for j, (name, (s, _st, g)) in enumerate(sorted(claims.items())):
            lo = j * part
            hi = self.SLOT if j == n - 1 else lo + part
            outs.append(self.arena.complete_slots([s], [g], [(lo, hi - lo)])[0])
        self.assertEqual(outs, [0] * (n - 1) + [1], f"completions {outs}: 3 = recycled, 4 = written into FREE")
        self.assertEqual(self._state(slot), S_COMPLETE)
        self.assertEqual(self.arena.stats()["claimed"], 3, "only the busy claims are left: nothing leaks")


class ClaimPublishRaceTest(_RaceCase):
    def test_a_join_never_commits_on_a_claim_whose_header_is_half_written(self):
        """phase 0: B's claim of K on the reaped slot has written the key, not yet generation/writers."""
        a, _f, res = self._publish_race(phase=0, third_claims_first=False)
        self._all_land({"A": res["a"], "B": res["b"]})
        self.assertEqual(self._state(a), 0, "B's yielded slot is FREE again, not held in the private state")

    def test_a_join_never_commits_on_a_claim_that_yields_its_slot(self):
        """phase 1: B's header is set up, a third rank publishes K first -- B yields its slot; A must not be on it."""
        a, _f, res = self._publish_race(phase=1, third_claims_first=True)
        self._all_land({"A": res["a"], "B": res["b"], "C": res["c"]})
        self.assertEqual(self._state(a), 0, "B's yielded slot is FREE again, not held in the private state")


class FreerRecheckUnderTheLockTest(_RaceCase):
    """the freer's CAS lands while the joiner holds the slot lock and has decided to join"""

    def _freer_in_the_join(self, freer):
        out = {}

        def _locked(slot):
            tb = threading.Thread(target=lambda: out.setdefault("r", freer(int(slot))), daemon=True)
            out["t"] = tb
            tb.start()
            t0 = time.monotonic()
            while self._state(int(slot)) == S_CLAIMED and time.monotonic() - t0 < 5:
                time.sleep(0.001)
            out["seen"] = self._state(int(slot))

        once = []

        def _cb(slot):
            if once:
                return
            once.append(slot)
            _locked(slot)

        ref = HOOK_T(_cb)
        self._refs.append(ref)
        self.lib.arena_set_join_locked_hook(ctypes.cast(ref, ctypes.c_void_p))
        got = self._bounded(lambda: self.arena.claim_slots(["K"], [self.SLOT])[0], "the join hung")
        out["t"].join(10)
        self.assertFalse(out["t"].is_alive(), "the freer hung on the slot lock")
        return got, out

    def test_reap_partial_backs_off_from_a_join_it_raced(self):
        s, g = self._orphan("K")
        (s2, st2, g2), out = self._freer_in_the_join(lambda slot: self.arena.reap_partial(0.0))
        self.assertNotEqual(out["seen"], S_CLAIMED, "the reap's CAS landed inside the join")
        self.assertEqual((s2, st2, g2), (s, 1, g), "the join committed")
        self.assertEqual(out["r"], [], "the reap saw the joiner's writer and kept the slot")
        self.assertEqual(self._state(s), S_CLAIMED)
        self.assertEqual(self._open(s), 1)
        self.assertEqual(self.arena.complete_slots([s], [g], [(self.SLOT // 4, 3 * self.SLOT // 4)]), [1])

    def test_release_claims_backs_off_from_a_join_it_raced(self):
        (s, st, g), = self.arena.claim_slots(["K"], [self.SLOT])   # TP0's fresh claim, its writer open
        assert st == 0
        (s2, st2, g2), out = self._freer_in_the_join(lambda slot: self.arena.release_claims([slot], [g]))
        self.assertNotEqual(out["seen"], S_CLAIMED, "the release's CAS landed inside the join")
        self.assertEqual((s2, st2, g2), (s, 1, g), "the join committed")
        self.assertEqual(out["r"], [1], "joined by another writer: kept for it (#1427r), never freed")
        self.assertEqual(self._state(s), S_CLAIMED)
        self.assertEqual(self._open(s), 1, "TP0's writer resolved, the joiner's open")
        self.assertEqual(self.arena.complete_slots([s], [g], [(0, self.SLOT)]), [1])

    def test_the_last_writer_out_backs_off_from_a_join_it_raced(self):
        """OS free_if_abandoned: TP0 claimed fresh, TP2 joined and gave up; TP0 gives up too (the last open writer
        of a slot nobody wrote) while TP1 joins -- the slot stays TP1's."""
        (s, st, g), = self.arena.claim_slots(["K"], [self.SLOT])
        assert st == 0
        (s1, st1, _g1), = self.arena.claim_slots(["K"], [self.SLOT])
        assert (s1, st1) == (s, 1) and self.arena.unclaim([s], [g]) == 1
        (s2, st2, g2), out = self._freer_in_the_join(lambda slot: self.arena.release_claims([slot], [g]))
        self.assertNotEqual(out["seen"], S_CLAIMED, "the give-up's CAS landed inside the join")
        self.assertEqual((s2, st2, g2), (s, 1, g), "the join committed")
        self.assertEqual(out["r"], [1], "the last-out free saw the joiner's writer: kept")
        self.assertEqual(self._state(s), S_CLAIMED)
        self.assertEqual(self._open(s), 1)
        self.assertEqual(self.arena.complete_slots([s], [g], [(0, self.SLOT)]), [1])


if __name__ == "__main__":
    unittest.main()
