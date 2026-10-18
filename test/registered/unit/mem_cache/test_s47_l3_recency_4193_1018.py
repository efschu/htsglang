"""S47 (NF rc12, boot ..._6b72947257_1018_153151): pdflip-4-193 read 0 of 562 store pages.

16:40:47 front 'L3-INDEX-PRESENCE rid=pdflip-4-193 tier=l2_arena depth=35328 pages=552 l3_pages=532' (the session
head is in L3). 16:41:43 PP0 -- the storage OWNER of group P, at its wake -- 'HiCacheFile eviction index updated
from the journal at wake: 20026 line(s) ... +4300/-8244 files', then 'L3-EVICT-FILE stem=d3bf71baccd0bf4728293193
9130b2eb550caec262b6439b60503734372b91e8_... comp=kv bytes=786432 age_s=1 why=cap_bg run=2952 n=1'. 16:41:44/45 every
P rank '#1439 ARENA-PRESENT keys=562 leading_complete=0 ... first_stem=d3bf71ba...' and '#1472 READ-TRACE asked=512
readable=511 arena_hits=345 first_missing=d3bf71baccd0bf47... why=no-file': page 0 -- and only page 0 -- was gone,
so 'PDFLIP-CLAIM-SHORT rid=pdflip-4-193 asked=562 kv=0' on all three ranks and P prefilled 36000 tokens instead of
672 ('P-REUSE-GAP ... read_short=35328').

age_s=1: the file's mtime was one second old (the only processes that utime a page are the eviction owners, so the
sibling group's owner had just touched it), yet it stood at the OLD end of PP0's LRU: the owner takes its order from
journal lines (C/E) and a touch never became one -- the L3-READ-RECENCY-1014 fix (38c6d40239) that journals a read
hit as a ``T`` line was lost at the h17/h18 fallback. Ported (port of 38c6d40239); these tests pin the 4-193 shape:
P's owner, D's owner as the toucher, and P's followers (non-owners: no utime at all, the journal is their only
voice).
"""

import os
import tempfile
import unittest
from unittest import mock

from flliper.srt.mem_cache.storage.file import store_journal as SJ
from flliper.srt.mem_cache.storage.file.lru_file_evictor import LRUFileEvictor

SHARED = "_K"
PAGE = 4096


def _ev(d, group, tp=0, pp=0, cap=10 ** 9):
    def _iter_existing():
        for name in sorted(os.listdir(d)):
            if name.endswith(".bin"):
                yield name[:-4], os.stat(os.path.join(d, name))

    with mock.patch.dict(os.environ, {"FLLIPER_PDFLIP_GROUP": group}):
        return LRUFileEvictor(d, SHARED, tp_rank=tp, pp_rank=pp, attn_cp_rank=0,
                              writes_shared_keys=True, scan_suffixes=(SHARED,),
                              extra_config={"max_size": str(cap), "max_size_scope": "shared"},
                              writer_count=1, iter_existing=_iter_existing)


def _fill(ev, d, stems):
    for s in stems:
        assert ev.reserve(s, PAGE, key=s, owner_writes_whole_file=False)
        with open(os.path.join(d, s + ".bin"), "wb") as f:
            f.write(b"\x07" * PAGE)
        ev.commit(s)


class _Env(unittest.TestCase):
    KEYS = (SJ.ENV, "FLLIPER_PDFLIP_L3_EPOCH", "FLLIPER_PDFLIP_GROUP", "FLLIPER_HICACHE_L3_JOURNAL_TOUCH",
            "FLLIPER_HICACHE_L3_JOURNAL_TOUCH_S")

    def setUp(self):
        self._saved = {k: os.environ.pop(k, None) for k in self.KEYS}
        self._td = tempfile.TemporaryDirectory()
        self.d = self._td.name
        os.environ["FLLIPER_PDFLIP_L3_EPOCH"] = "100"

    def tearDown(self):
        self._td.cleanup()
        for k, v in self._saved.items():
            os.environ.pop(k, None)
            if v is not None:
                os.environ[k] = v

    def _p_owner_with_session(self):
        """P's owner (pp0) wrote the session head first (page 0 = the 4-193 stem), other pages after it."""
        p0 = _ev(self.d, "P")
        head = [f"h{i:03d}{SHARED}" for i in range(4)]
        rest = [f"r{i:03d}{SHARED}" for i in range(8)]
        _fill(p0, self.d, head + rest)
        p0.rescan()  # P's own lines are marked; what follows comes from the other writers
        return p0, head, rest


class TestS47Recency4193(_Env):
    def test_d_owner_touch_of_page0_survives_p_owners_wake_eviction(self):
        """The 4-193 sequence: D's owner touches page 0 (D's park write of an existing page is a touch), P's owner
        wakes (journal apply) and its eviction run takes the oldest pages -- page 0 must not be among them."""
        p0, head, rest = self._p_owner_with_session()
        d0 = _ev(self.d, "D")
        d0.touch(head[0], os.path.join(self.d, head[0] + ".bin"))
        p0.rescan()  # the wake
        with p0._lock:
            for _ in range(len(head) - 1 + 3):
                p0._evict_one_lru_locked()
        self.assertIn(head[0], p0._lru, "page 0, touched by the sibling before the wake, is MRU")
        self.assertTrue(os.path.exists(os.path.join(self.d, head[0] + ".bin")))

    def test_a_follower_of_ps_group_is_heard_through_the_journal(self):
        """P's followers are not eviction owners: their touch never reaches the inode (no utime), the T line is
        the only way their read reaches PP0's order."""
        p0, head, rest = self._p_owner_with_session()
        pp1 = _ev(self.d, "P", pp=1)
        self.assertFalse(pp1._eviction_enabled, "a P follower is not the store owner")
        pp1.touch(head[1], os.path.join(self.d, head[1] + ".bin"))
        with open(pp1._journal.path, "rb") as f:
            recs, _ = SJ.parse_lines(f.read())
        self.assertEqual([r[3] for r in recs if r[1] == "T"], [head[1]])
        p0.rescan()
        self.assertEqual(list(p0._lru)[-1], head[1])

    def test_without_the_t_line_page0_is_the_first_victim(self):
        """The rc12 behaviour (the switch off = the h16 tree): page 0 leaves first, as at 16:41:43."""
        os.environ["FLLIPER_HICACHE_L3_JOURNAL_TOUCH"] = "0"
        p0, head, _rest = self._p_owner_with_session()
        _ev(self.d, "D").touch(head[0], os.path.join(self.d, head[0] + ".bin"))
        p0.rescan()
        with p0._lock:
            p0._evict_one_lru_locked()
        self.assertNotIn(head[0], p0._lru)
        self.assertFalse(os.path.exists(os.path.join(self.d, head[0] + ".bin")))

    def test_the_next_attach_keeps_the_read_order(self):
        """A boot later (attach replay): the page the sibling read last is the newest, not its write time."""
        p0, head, rest = self._p_owner_with_session()
        _ev(self.d, "D").touch(head[0], os.path.join(self.d, head[0] + ".bin"))
        items = {}
        for p in SJ.journal_paths(self.d):
            with open(p, "rb") as f:
                recs, _ = SJ.parse_lines(f.read())
            for r in recs:
                if r[1] in ("C", "T"):
                    items.setdefault("_recs", []).append(r)
        recs = sorted(items["_recs"], key=lambda r: r[0])
        idx = {}
        SJ.replay(idx, recs)
        self.assertEqual(list(idx)[-1], head[0])


if __name__ == "__main__":
    unittest.main()
