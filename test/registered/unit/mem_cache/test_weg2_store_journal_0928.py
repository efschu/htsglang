"""The L3 store's write journal (28.09.): the wake reads the delta, not the directory.

27B 18:51: the wake rescan walked 821,711 files (4.7 s at 700k, ~18 s projected
at the 150 GB cap) on every wake. Graded here: the journal read reaches the same
index as a walk, walks nothing, falls back to a walk where it cannot be trusted,
compacts at attach, and stays inside its own store.
"""

import os
import tempfile
import unittest

from sglang.srt.mem_cache.storage.file import store_journal as SJ
from sglang.srt.mem_cache.storage.file.lru_file_evictor import LRUFileEvictor

SHARED = "_K"
PAGE = 4096


def _owner(d, cap, walks=None):
    def _iter_existing():
        if walks is not None:
            walks.append(1)
        for name in sorted(os.listdir(d)):
            if name.endswith(".bin"):
                yield name[:-4], os.stat(os.path.join(d, name))

    return LRUFileEvictor(d, SHARED, tp_rank=0, pp_rank=0, attn_cp_rank=0,
                          writes_shared_keys=True, scan_suffixes=(SHARED,),
                          extra_config={"max_size": str(cap), "max_size_scope": "shared"},
                          writer_count=1, iter_existing=_iter_existing)


def _write(d, stem, n=PAGE):
    with open(os.path.join(d, stem + ".bin"), "wb") as f:
        f.write(b"\x07" * n)


def _fill(ev, d, stems):
    for s in stems:
        assert ev.reserve(s, PAGE, key=s)
        _write(d, s)
        ev.commit(s)


class _Env(unittest.TestCase):
    KEYS = (SJ.ENV, SJ.FULL_WALK_EVERY_ENV, "SGLANG_WEG2_L3_EPOCH")

    def setUp(self):
        self._saved = {k: os.environ.pop(k, None) for k in self.KEYS}
        self._td = tempfile.TemporaryDirectory()
        self.d = self._td.name

    def tearDown(self):
        self._td.cleanup()
        for k, v in self._saved.items():
            os.environ.pop(k, None)
            if v is not None:
                os.environ[k] = v


class TestDelta(_Env):
    def test_the_wake_reads_the_siblings_lines_and_walks_nothing(self):
        walks = []
        me = _owner(self.d, 10 ** 9, walks)
        sib = _owner(self.d, 10 ** 9)
        _fill(sib, self.d, [f"s{i:04d}{SHARED}" for i in range(30)])
        n = len(walks)
        census = me.rescan()
        self.assertEqual(len(walks), n, "the wake walked the directory")
        self.assertEqual(census.get("mode"), "journal")
        self.assertEqual(census["indexed_entries"], 30)
        self.assertEqual(me.stats()["foreign_indexed_bytes"], 30 * PAGE)

    def test_the_journal_index_equals_a_walk(self):
        me = _owner(self.d, 10 ** 9)
        sib = _owner(self.d, 10 ** 9)
        _fill(me, self.d, [f"m{i:04d}{SHARED}" for i in range(7)])
        _fill(sib, self.d, [f"s{i:04d}{SHARED}" for i in range(11)])
        _write(self.d, "priv0000_B", 3 * PAGE)          # a suffix this group does not scan
        sib._journal.write("C", 0.0, 3 * PAGE, "priv0000_B")
        me.rescan()
        os.environ[SJ.ENV] = "0"
        ref = _owner(self.d, 10 ** 9)
        self.assertEqual(me.index_coverage()["indexed_bytes"], ref.index_coverage()["indexed_bytes"])
        self.assertEqual(me.stats()["directory_bytes"] - me.stats()["staging_bytes"],
                         ref.stats()["directory_bytes"] - ref.stats()["staging_bytes"])
        self.assertEqual(me.index_coverage()["seen_bytes"], ref.index_coverage()["seen_bytes"])
        self.assertEqual(me.index_coverage()["indexed_entries"], ref.index_coverage()["indexed_entries"])

    def test_an_unlink_line_removes_the_page(self):
        me = _owner(self.d, 10 ** 9)
        sib = _owner(self.d, 10 * PAGE)               # small cap: it evicts its own
        _fill(sib, self.d, [f"s{i:04d}{SHARED}" for i in range(10)])
        me.rescan()
        _fill(sib, self.d, [f"t{i:04d}{SHARED}" for i in range(5)])   # evicts 5 of s*
        me.rescan()
        on_disk = sum(1 for n in os.listdir(self.d) if n.endswith(".bin"))
        self.assertEqual(me.index_coverage()["indexed_entries"], on_disk)


class TestCrashGapAndFallback(_Env):
    def test_a_page_without_a_line_is_found_by_the_periodic_walk(self):
        os.environ[SJ.FULL_WALK_EVERY_ENV] = "2"
        walks = []
        me = _owner(self.d, 10 ** 9, walks)
        _write(self.d, f"crash0000{SHARED}")            # landed, writer died before its line
        n = len(walks)
        me.rescan()                                     # wake 1: journal, blind to it
        self.assertEqual(len(walks), n)
        self.assertEqual(me.index_coverage()["indexed_entries"], 0)
        me.rescan()                                     # wake 2: the named full walk
        self.assertEqual(len(walks), n + 1)
        self.assertEqual(me.index_coverage()["indexed_entries"], 1)

    def test_a_truncated_journal_forces_a_walk(self):
        walks = []
        me = _owner(self.d, 10 ** 9, walks)
        sib = _owner(self.d, 10 ** 9)
        _fill(sib, self.d, [f"s{i:04d}{SHARED}" for i in range(3)])
        me.rescan()
        os.truncate(sib._journal.path, 0)
        n = len(walks)
        me.rescan()
        self.assertEqual(len(walks), n + 1)

    def test_a_line_whose_file_is_gone_is_a_drop_not_an_error(self):
        me = _owner(self.d, 3 * PAGE)
        sib = _owner(self.d, 10 ** 9)
        _fill(sib, self.d, [f"s{i:04d}{SHARED}" for i in range(2)])
        me.rescan()
        os.unlink(os.path.join(self.d, f"s0000{SHARED}.bin"))   # journal says there, file is not
        # foreign pages are never unlinked here; this owner's own writes evict
        # its own and the stale foreign entry stays counted until a walk -- no raise
        _fill(me, self.d, [f"m{i:04d}{SHARED}" for i in range(1)])

    def test_switch_zero_walks_every_wake(self):
        os.environ[SJ.ENV] = "0"
        walks = []
        me = _owner(self.d, 10 ** 9, walks)
        n = len(walks)
        me.rescan()
        self.assertEqual(len(walks), n + 1)
        self.assertFalse([p for p in os.listdir(self.d) if p.endswith(SJ.SUFFIX)])


class TestCompaction(_Env):
    def test_an_attach_removes_earlier_boots_journals_only(self):
        old = os.path.join(self.d, f"{SJ.PREFIX}100.g-t0-p0-c0.aaaa{SJ.SUFFIX}")
        same = os.path.join(self.d, f"{SJ.PREFIX}200.g-t1-p0-c0.bbbb{SJ.SUFFIX}")
        for p in (old, same):
            with open(p, "w") as f:
                f.write("C 1.0 4096 x_K\n")
        os.environ["SGLANG_WEG2_L3_EPOCH"] = "200"
        _owner(self.d, 10 ** 9)
        self.assertFalse(os.path.exists(old))
        self.assertTrue(os.path.exists(same))


class TestTwoStoresOnOneFilesystem(_Env):
    def test_a_neighbour_stores_writes_never_reach_this_index(self):
        with tempfile.TemporaryDirectory() as other:
            me = _owner(self.d, 10 ** 9)
            neighbour = _owner(other, 10 ** 9)
            _fill(neighbour, other, [f"n{i:04d}{SHARED}" for i in range(20)])
            census = me.rescan()
            self.assertEqual(census["indexed_entries"], 0)
            self.assertEqual(census["seen_bytes"], 0)


if __name__ == "__main__":
    unittest.main()
