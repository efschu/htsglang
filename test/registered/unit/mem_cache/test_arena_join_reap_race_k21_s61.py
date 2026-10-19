# SPDX-License-Identifier: Apache-2.0
"""S61 (NF rc15 489909d138, boot ..._1019_071429, D group TP3): a JOIN of an arena claim raced the free of
that claim by another process -- the joiner held a claim on a slot that was already freed.

MEASURED (D log, every line of the chain; TP0/TP1/TP2 = three processes on ONE shared /dev/shm arena):

  09:24:27 TP1/TP2  #924D station=backup rid=node3230 (depth 104192)      <- the write-back of the same node
  09:24:27 TP0      #231 ARENA-REAP-PARTIAL slots=[5725] age_s=30          <- TP0's claim makes room: reaps an
                                                                              orphan claim (no open writer)
  09:24:28 TP1      #1427 ARENA-COMPLETE LOST site=complete_write recycled=1 slots=[5725]
  09:24:28 TP1      A2 LOST-ROWS ... orphaned=1 / K12 A2-ORPHAN DIRECT-ACK node=3230 / K21 EVICT-SKIP node=3230
  09:24:30 TP1      K21 RANK-DIVERGENCE this worker's ack orphaned node hash=f40780df6b97b66b depth=104192 but
                    TP0's ack of the same write was clean -- group stop

``arena_reap_partial`` frees only a claim with NO open writer, and a joiner opens its writer at the join -- yet
TP1's completion found the slot recycled (status 3 = the generation moved under it). The only way: arena.c
``arena_claim`` joined in three unguarded steps (find the slot, read its generation, add the open writer); the
reap's guard (CAS to EVICTING, re-check the writer word, free) ran entirely between the first two and the third.
TP1 kept a claim (old generation, its open writer added on a FREED slot) that no completion could land on;
TP0 and TP2 claimed the key after the reap, fresh -- their acks were clean. One rank's tree kept the node on the
device (A2 orphan), the others did not: K21 (raenge-nie-uneins-crash-stop) stopped the group.

The abort that rode the same pass (#801 'deferred=False') is NOT a rank-local event on D: AbortReqs reach every
TP rank through TP0's broadcast of the pass (request_receiver.recv_requests -> _broadcast_reqs_across_ranks),
and TP1 raised inside ``consume`` of exactly that broadcast, before it got to log them.

Real C arena on a temp file; the race point is driven deterministically through the S61 test hook (called for a
join candidate between "found" and "committed"). RED on 489909d138 (+ hook only), GREEN on the fix.
"""

import ctypes
import os
import shutil
import struct
import tempfile
import threading
import unittest

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

S_CLAIMED, S_COMPLETE, S_EVICTING = 1, 2, 3
HOOK_T = ctypes.CFUNCTYPE(None, ctypes.c_int64)


class _ArenaCase(unittest.TestCase):
    SLOT = 4096
    SLOTS = 4

    def setUp(self):
        if shutil.which("gcc") is None:
            self.skipTest("needs gcc for the C arena")
        from flliper.srt.mem_cache.storage.file.hicache_arena import ShmArena

        self.dir = tempfile.mkdtemp()
        self.arena = ShmArena(os.path.join(self.dir, "arena-kv.bin"), self.SLOT, self.SLOTS)
        self.lib = self.arena._lib
        self.lib.arena_set_join_hook.restype = None
        self.lib.arena_set_join_hook.argtypes = [ctypes.c_void_p]
        self._hook_ref = None

    def tearDown(self):
        self.lib.arena_set_join_hook(None)
        self.arena.close()
        shutil.rmtree(self.dir, ignore_errors=True)

    # -- helpers ---------------------------------------------------------------------------------------------
    def _hook(self, fn):
        """arm the S61 join hook ONCE: ``fn(slot)`` runs at the join's race point, then the hook disarms."""
        fired = []

        def _cb(slot):
            if fired:
                return
            fired.append(int(slot))
            fn(int(slot))

        self._hook_ref = HOOK_T(_cb)   # keep the trampoline alive
        self.lib.arena_set_join_hook(ctypes.cast(self._hook_ref, ctypes.c_void_p))
        return fired

    def _orphan(self, stem):
        """the rc15 orphan: a claim whose writer merged a part and resolved -- CLAIMED, open 0, never COMPLETE."""
        (s, st, g), = self.arena.claim_slots([stem], [self.SLOT])
        assert st == 0
        assert self.arena.complete_slots([s], [g], [(0, self.SLOT // 4)]) == [0]
        return s, g

    def _open(self, slot):
        return self.arena.claim_info([slot])[0][5]

    def _state(self, slot):
        return self.arena.claim_info([slot])[0][6]

    def _set_state(self, slot, state):
        out = (ctypes.c_int64 * 6)()
        self.lib.arena_layout(self.SLOTS, self.SLOT, out)
        header_bytes, headers_off = int(out[0]), int(out[3])
        struct.pack_into("<I", self.arena._mm, headers_off + slot * header_bytes, int(state))


class JoinReapRaceTest(_ArenaCase):
    def test_rc15_join_preempted_while_the_reap_frees_never_completes_lost(self):
        """09:24:27/28: TP0's reap frees the orphan while TP1 joins it -- TP1's completion must land."""
        s, _g = self._orphan("K")
        freed = []
        fired = self._hook(lambda slot: freed.extend(self.arena.reap_partial(0.0)))
        (s2, st2, g2), = self.arena.claim_slots(["K"], [self.SLOT])
        self.assertEqual(fired, [s], "the join candidate is the orphan")
        self.assertEqual(freed, [s], "the reap (no open writer yet) frees it -- the rc15 interleaving")
        self.assertEqual(st2, 0, "the orphan is gone: the joiner claims the key fresh, never a join of a freed slot")
        self.assertEqual(self.arena.find_slots(["K"]), [(s2, S_CLAIMED)], "the key names the joiner's slot")
        self.assertEqual(self._open(s2), 1, "exactly the joiner's open writer")
        # TP1 merges its extents (the token-owner half): never 'recycled under the writer'
        self.assertEqual(self.arena.complete_slots([s2], [g2], [(self.SLOT // 2, self.SLOT // 2)]), [0])
        self.assertEqual(self._open(s2), 0)
        self.assertEqual(self.arena.stats()["claimed"], 1, "the reaped orphan is not counted twice, nothing leaks")

    def test_aba_the_freed_slot_reclaimed_for_another_key_is_never_joined(self):
        """Between the find and the commit the slot is reaped AND re-claimed for another key (the only free slot):
        the joiner must not write its page into that key's slot."""
        a, _g = self._orphan("K")
        others = [self.arena.claim_slots([f"busy{i}"], [self.SLOT])[0] for i in range(self.SLOTS - 1)]
        assert all(st == 0 for _s, st, _g in others)          # open writers: never reaped
        got = {}

        def _reap_and_reclaim(slot):
            got["freed"] = self.arena.reap_partial(0.0)
            got["k2"] = self.arena.claim_slots(["K2"], [self.SLOT])[0]

        self._hook(_reap_and_reclaim)
        (s, st, _g2), = self.arena.claim_slots(["K"], [self.SLOT])
        self.assertEqual(got["freed"], [a])
        k2_slot, k2_st, _ = got["k2"]
        self.assertEqual((k2_slot, k2_st), (a, 0), "K2 took the freed slot fresh")
        self.assertEqual(st, 4, "arena full for K: refused by name, never a join of K2's slot")
        self.assertNotEqual(s, a)
        self.assertEqual(self._open(a), 1, "K2's slot carries K2's writer only")

    def test_a_join_waiting_on_a_freer_that_never_finishes_is_bounded(self):
        """A freer died between its CAS (EVICTING) and its free: the join waits a bounded time, then claims the key
        fresh -- it never hangs and never joins the dead slot."""
        s, _g = self._orphan("K")
        self._hook(lambda slot: self._set_state(slot, S_EVICTING))
        res = {}
        t = threading.Thread(target=lambda: res.setdefault("r", self.arena.claim_slots(["K"], [self.SLOT])[0]),
                             daemon=True)
        t.start()
        t.join(10.0)
        self.assertFalse(t.is_alive(), "the join hung on a slot that never leaves EVICTING")
        s2, st2, g2 = res["r"]
        self.assertEqual(st2, 0, "fresh claim beside the dead slot")
        self.assertNotEqual(s2, s)
        self.assertEqual(self.arena.complete_slots([s2], [g2], [(0, self.SLOT)]), [1])

    def test_a_plain_join_is_unchanged(self):
        """No race: the join of an orphan is a join (same slot, same generation), and an open joiner is never
        reaped."""
        s, g = self._orphan("K")
        self._hook(lambda slot: None)
        (s2, st2, g2), = self.arena.claim_slots(["K"], [self.SLOT])
        self.assertEqual((s2, st2, g2), (s, 1, g))
        self.assertEqual(self._open(s), 1)
        self.assertEqual(self.arena.reap_partial(0.0), [], "an open writer keeps its claim")
        self.assertEqual(self.arena.complete_slots([s], [g], [(self.SLOT // 4, 3 * self.SLOT // 4)]), [1])
        self.assertEqual(self._state(s), S_COMPLETE)

    def test_a_join_of_a_complete_page_stays_status_2(self):
        (s, st, g), = self.arena.claim_slots(["C"], [self.SLOT])
        assert st == 0 and self.arena.complete_slots([s], [g], [(0, self.SLOT)]) == [1]
        self._hook(lambda slot: None)
        self.assertEqual(self.arena.claim_slots(["C"], [self.SLOT]), [(s, 2, g)])
        self.assertEqual(self._open(s), 0, "a found-COMPLETE page opens no writer")


if __name__ == "__main__":
    unittest.main()
