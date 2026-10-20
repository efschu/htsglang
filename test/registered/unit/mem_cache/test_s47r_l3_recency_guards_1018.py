"""S47r (review of S47, NF 18.10.): the danger directions of L3-READ-RECENCY-1014 the S47 tests left open.

(d) no false index entry: a ``T`` line for a page the OWNER has already evicted (``E``, file gone) must not put it back
    into the owner's LRU at the wake -- the index would name a missing file (S47 test set: mutant "apply T resurrects"
    survived).
(d) no growth without bound in RAM: the per-writer dedupe map of ``_journal_touch`` stays at its 65536 cap (mutant
    "dedupe cache unbounded" survived).
"""

import os

from test_pdflip_l3_read_recency_1014 import SHARED, _Env, _ev, _fill


class TestS47rGuards(_Env):
    def test_a_t_line_for_a_page_the_owner_evicted_does_not_resurrect_it_at_the_wake(self):
        me = _ev(self.d)
        sib = _ev(self.d, group="P")
        stems = [f"e{i:03d}{SHARED}" for i in range(4)]
        _fill(me, self.d, stems)
        me.rescan()
        victim = stems[0]
        sib.touch(victim, os.path.join(self.d, victim + ".bin"))   # the sibling's T line
        with me._lock:
            me._evict_one_lru_locked()                             # the owner evicts it before its next wake
        self.assertNotIn(victim, me._lru)
        total = me._total_bytes
        me.rescan()                                                # the wake applies the sibling's T line
        self.assertNotIn(victim, me._lru, "a T line resurrected an evicted page in the owner's index")
        self.assertEqual(me._total_bytes, total)
        self.assertFalse(os.path.exists(os.path.join(self.d, victim + ".bin")))

    def test_the_touch_dedupe_map_is_bounded(self):
        sib = _ev(self.d, group="P")
        for i in range(65536 + 300):
            sib._journal_touch(f"k{i:06d}{SHARED}")
        self.assertLessEqual(len(sib.__dict__["_journal_touched"]), 65536)
        # the newest keys are the ones kept (LRU of the map, not a freeze)
        self.assertIn(f"k{65536 + 299:06d}{SHARED}", sib.__dict__["_journal_touched"])
