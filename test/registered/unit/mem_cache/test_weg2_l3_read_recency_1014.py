"""L3-READ-RECENCY-1014 (NF h13, boot ..._84e787ec96_1014_135753).

A vision session (134k tokens, 5 images) is served on D from its DEVICE across every flip and read by P from the
shared L3 store at each P leg: 14:21:32 PP0 'loaded=124928 ... hit_tokens=124928' (weg2-18-98). Its text prefix
had been written by an EARLIER boot (the first P leg of this boot, weg2-4-16, already hit 86272 of 87427).
14:22:55 D TP0 'HiCacheFile EVICTION-BG reclaimed 7546953728 B ... directory 142536695808 -> 134989742080 B toward
cap 150000000000 B x ratio 0.90' -- oldest-first in D's owner index. 14:25:49 the front probe found 88 leading
KV pages and no mamba anchor (14:20:53: 196 pages, anchor depth 8320); P prefilled 134240 tokens in 39.6 s.

D's owner index is rebuilt from snapshot + journal lines (R/C/A/E: write times) and the wake applies only C/E --
P's read hit was a utime on the inode (``_touch_mtime``, whose docstring still says the sibling "rebuilds its
order from st_mtime when it wakes"; it has not walked since the 28.09. journal index). So a page only the
SIBLING reads stays at its write time and is the first victim. Fix: a read hit journals a ``T`` line; the
owner's wake and the attach replay move the page to MRU.
"""

import os
import tempfile
import unittest
from unittest import mock

from sglang.srt.mem_cache.storage.file import store_journal as SJ
from sglang.srt.mem_cache.storage.file.lru_file_evictor import LRUFileEvictor

SHARED = "_K"
PAGE = 4096


def _ev(d, cap=10 ** 9, tp=0, group="D"):
    def _iter_existing():
        for name in sorted(os.listdir(d)):
            if name.endswith(".bin"):
                yield name[:-4], os.stat(os.path.join(d, name))

    with mock.patch.dict(os.environ, {"SGLANG_WEG2_GROUP": group}):
        return LRUFileEvictor(d, SHARED, tp_rank=tp, pp_rank=0, attn_cp_rank=0,
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
    KEYS = (SJ.ENV, "SGLANG_WEG2_L3_EPOCH", "SGLANG_WEG2_GROUP", "SGLANG_HICACHE_L3_JOURNAL_TOUCH",
            "SGLANG_HICACHE_L3_JOURNAL_TOUCH_S")

    def setUp(self):
        self._saved = {k: os.environ.pop(k, None) for k in self.KEYS}
        self._td = tempfile.TemporaryDirectory()
        self.d = self._td.name
        os.environ["SGLANG_WEG2_L3_EPOCH"] = "100"

    def tearDown(self):
        self._td.cleanup()
        for k, v in self._saved.items():
            os.environ.pop(k, None)
            if v is not None:
                os.environ[k] = v

    def _scene(self):
        """D (owner) wrote the session prefix first, other pages later; P only READS the prefix."""
        me = _ev(self.d)
        sib = _ev(self.d, group="P")
        prefix = [f"a{i:03d}{SHARED}" for i in range(10)]
        later = [f"b{i:03d}{SHARED}" for i in range(10)]
        _fill(me, self.d, prefix + later)
        me.rescan()  # the reader's mark: the lines above are this owner's own
        for s in prefix:
            sib.touch(s, os.path.join(self.d, s + ".bin"))
        return me, sib, prefix, later


class TestReadRecency(_Env):
    def test_default_is_on(self):
        from sglang.srt.environ import envs
        self.assertTrue(envs.SGLANG_HICACHE_L3_JOURNAL_TOUCH.get())

    def test_the_siblings_read_moves_the_prefix_to_mru_at_the_owners_wake(self):
        me, _sib, prefix, later = self._scene()
        self.assertEqual(list(me._lru)[:10], prefix, "before the wake: the prefix is the oldest")
        me.rescan()
        self.assertEqual(list(me._lru)[:10], later, "the pages nobody read are the oldest now")
        self.assertEqual(list(me._lru)[10:], prefix)

    def test_the_owner_evicts_the_unread_pages_first(self):
        me, _sib, prefix, later = self._scene()
        me.rescan()
        with me._lock:
            for _ in range(5):
                me._evict_one_lru_locked()
        left = set(me._lru)
        self.assertTrue(set(prefix) <= left, "the prefix the sibling read survives")
        self.assertEqual(len(set(later) - left), 5)

    def test_off_is_the_old_order(self):
        os.environ["SGLANG_HICACHE_L3_JOURNAL_TOUCH"] = "0"
        me, _sib, prefix, _later = self._scene()
        me.rescan()
        self.assertEqual(list(me._lru)[:10], prefix)
        for p in SJ.journal_paths(self.d):
            with open(p, "rb") as f:
                recs, _ = SJ.parse_lines(f.read())
            self.assertFalse([r for r in recs if r[1] == "T"], "no T line with the switch off")

    def test_at_most_one_line_per_page_and_window(self):
        sib = _ev(self.d, group="P")
        _fill(sib, self.d, [f"c000{SHARED}"])
        for _ in range(50):
            sib.touch(f"c000{SHARED}", os.path.join(self.d, f"c000{SHARED}.bin"))
        with open(sib._journal.path, "rb") as f:
            recs, _ = SJ.parse_lines(f.read())
        self.assertEqual(sum(1 for r in recs if r[1] == "T"), 1)

    def test_the_attach_replay_takes_the_read_time(self):
        items = {"x": (10.0, PAGE), "y": (20.0, PAGE)}
        SJ.replay(items, [(30.0, "T", 0, "x"), (31.0, "T", 0, "nope")])
        self.assertEqual(list(items), ["y", "x"])
        self.assertEqual(items["x"], (30.0, PAGE))
        self.assertNotIn("nope", items)

    def test_an_old_reader_skips_t_lines(self):
        recs, _ = SJ.parse_lines(b"C 1.000 4096 a_K\nT 2.000 0 a_K\nZ 3.000 0 a_K\n")
        self.assertEqual([r[1] for r in recs], ["C", "T"])


if __name__ == "__main__":
    unittest.main()
