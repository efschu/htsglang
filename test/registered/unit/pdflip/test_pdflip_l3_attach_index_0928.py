"""The launcher's L3 attach from the persistent index (28.09., operator order).

``l3_persist_attach`` walked the whole store with a stat per page on every boot.
It now derives pages, orphans, staging files and revoked pages from snapshot +
journals, and walks only where the ranks walk too (snapshot missing/broken,
another kernel boot). Every page it removes gets an ``E`` line the ranks' load
replays, so no index keeps an entry the launcher unlinked.
"""

import hashlib
import inspect
import json
import os
import shutil
import tempfile
import time
import unittest
from unittest import mock

from flliper.srt.mem_cache.storage.file import store_journal as SJ
from flliper.srt.pdflip import launcher

K = "_K"


def _stem(i, suffix=K):
    return hashlib.sha256(str(i).encode()).hexdigest() + suffix


class _Log(list):
    def __call__(self, line):
        self.append(line)


class _Store(unittest.TestCase):
    KEYS = (SJ.ENV, "FLLIPER_PDFLIP_L3_EPOCH", "FLLIPER_PDFLIP_L3_INHERITED_SUFFIXES")

    def setUp(self):
        self._saved = {k: os.environ.pop(k, None) for k in self.KEYS}
        self._td = tempfile.TemporaryDirectory()
        self.d = self._td.name
        with open(os.path.join(self.d, "L3_SUFFIXES.D.json"), "w") as f:
            json.dump({"suffixes": [K]}, f)

    def tearDown(self):
        shutil.rmtree(self.d.rstrip("/") + launcher.L3_REVOKED_SUFFIX, ignore_errors=True)
        self._td.cleanup()
        for k, v in self._saved.items():
            os.environ.pop(k, None)
            if v is not None:
                os.environ[k] = v

    def page(self, stem, n=4096):
        p = os.path.join(self.d, stem[:2], stem + ".bin")
        os.makedirs(os.path.dirname(p), exist_ok=True)
        with open(p, "wb") as f:
            f.write(b"\x01" * n)
        return p

    def snapshot(self):
        items = []
        for root, _dirs, names in os.walk(self.d):
            for n in names:
                if n.endswith(".bin"):
                    st = os.lstat(os.path.join(root, n))
                    items.append((st.st_mtime, n[:-4], st.st_blocks * 512))
        items.sort()
        SJ.write_snapshot(self.d, items, 100)

    def attach(self, walk_forbidden=True):
        log = _Log()
        if walk_forbidden:
            with mock.patch.object(launcher.os, "walk",
                                   side_effect=AssertionError("the attach walked")):
                got = launcher.l3_persist_attach(log, self.d, dry=False)
        else:
            got = launcher.l3_persist_attach(log, self.d, dry=False)
        return got, log


class TestAttachFromIndex(_Store):
    def test_the_index_answers_what_the_walk_answered(self):
        for i in range(20):
            self.page(_stem(i))
        self.snapshot()
        os.environ[SJ.ENV] = "0"
        (w_files, w_bytes, _), _ = self.attach(walk_forbidden=False)
        os.environ.pop(SJ.ENV)
        (files, nbytes, _), log = self.attach()
        self.assertEqual((files, nbytes), (w_files, w_bytes))
        self.assertIn("source=index", log[-1])

    def test_an_orphan_is_removed_and_the_ranks_load_sees_it_gone(self):
        self.page(_stem(1))
        orphan = _stem(2, "_OLDHASH")
        path = self.page(orphan)
        self.snapshot()
        (files, _b, removed), _ = self.attach()
        self.assertEqual((files, removed), (1, 1))
        self.assertFalse(os.path.exists(path))
        items, why = SJ.load_index(self.d)
        self.assertEqual(why.split(" ")[0], "snapshot")
        self.assertNotIn(orphan, items, "the ranks' index keeps the page the launcher unlinked")

    def test_a_dead_writers_staging_file_goes_by_its_open_intent(self):
        live, dead = _stem(1), _stem(2)
        self.page(live)
        self.snapshot()
        jn = SJ.JournalWriter(self.d, "D-t1-p0-c0", epoch=100)
        jn.write("R", time.time(), 4096, dead)              # the writer died after this line
        tmp = os.path.join(self.d, dead[:2], dead + ".bin.tmp.0123abcd")
        os.makedirs(os.path.dirname(tmp), exist_ok=True)
        open(tmp, "wb").write(b"half")
        (files, _b, removed), _ = self.attach()
        self.assertEqual((files, removed), (1, 1))
        self.assertFalse(os.path.exists(tmp))

    def test_a_revoked_page_is_moved_by_its_indexed_mtime(self):
        keep, bad = _stem(1), _stem(2)
        os.utime(self.page(keep), (1000.0, 1000.0))      # written long before the window
        self.snapshot()
        t = time.time()
        self.page(bad)
        jn = SJ.JournalWriter(self.d, "P-t0-p0-c0", epoch=100)
        jn.write("R", t, 4096, bad)
        jn.write("C", t, 4096, bad)
        with mock.patch.object(launcher, "l3_revoked_windows",
                               return_value=[(t - 0.5, t + 0.5, "test")]):
            (files, _b, _r), _ = self.attach()
        self.assertEqual(files, 1)
        self.assertTrue(os.path.exists(
            os.path.join(self.d.rstrip("/") + launcher.L3_REVOKED_SUFFIX, bad[:2], bad + ".bin")))
        self.assertNotIn(bad, SJ.load_index(self.d)[0])

    def test_a_missing_or_broken_snapshot_walks_once(self):
        self.page(_stem(1))
        (files, _b, _r), log = self.attach(walk_forbidden=False)
        self.assertEqual(files, 1)
        self.assertIn("index unusable (no snapshot)", " ".join(log))
        self.snapshot()
        with open(os.path.join(self.d, SJ.SNAP), "ab") as f:
            f.write(b"x")
        with self.assertRaises(AssertionError):
            self.attach()                                    # it must walk, and walking is forbidden


class TestShardRule(unittest.TestCase):
    def test_page_path_follows_the_backends_shard_rule(self):
        from flliper.srt.mem_cache.hicache_storage import page_shard

        with tempfile.TemporaryDirectory() as d:
            for stem in (_stem(7), "zz-synthetic", "A1upper"):
                self.assertEqual(os.path.basename(os.path.dirname(SJ.page_path(d, stem))),
                                 page_shard(stem))


class TestOneWalkPerStore(_Store):
    """NF metal rc12z30c: launcher, PP0-2 and D TP0 each walked one store."""

    def _rank(self, walks, group):
        from flliper.srt.mem_cache.storage.file.lru_file_evictor import LRUFileEvictor

        def _iter():
            walks.append(group)
            for root, _dirs, names in os.walk(self.d):
                for n in names:
                    if n.endswith(".bin"):
                        yield n[:-4], os.stat(os.path.join(root, n))

        with mock.patch.dict(os.environ, {"FLLIPER_PDFLIP_GROUP": group}):
            return LRUFileEvictor(self.d, K, tp_rank=0, pp_rank=0, attn_cp_rank=0,
                                  writes_shared_keys=True, scan_suffixes=(K,),
                                  extra_config={"max_size": str(10 ** 9),
                                                "max_size_scope": "shared"},
                                  writer_count=1, iter_existing=_iter)

    def test_the_launcher_walk_is_the_only_walk_of_the_boot(self):
        for i in range(12):
            self.page(_stem(i))
        (files, _b, _r), log = self.attach(walk_forbidden=False)   # first boot: no snapshot
        self.assertEqual(files, 12)
        self.assertIn("SNAPSHOT written by the one walk", " ".join(log))
        walks = []
        ranks = [self._rank(walks, g) for g in ("P", "P", "P", "D")]
        self.assertEqual(walks, [], "a rank walked after the launcher's walk")
        self.assertEqual(ranks[-1].index_coverage()["indexed_entries"], 12)
        stems, why = SJ.index_stems(self.d)                   # the #1459 seed, no walk
        self.assertEqual(len(stems), 12, why)
        # second boot: the launcher reads and rewrites, the ranks load
        (files, _b, _r), log = self.attach()
        self.assertIn("snapshot rewritten", log[-1])
        walks = []
        for g in ("P", "P", "P", "D"):
            self._rank(walks, g)
        self.assertEqual(walks, [])

    def test_the_seed_of_the_shared_stem_index_reads_the_index(self):
        from flliper.srt.mem_cache import hicache_storage as HS

        src = inspect.getsource(HS.HiCacheFile._l3p_seed_index)
        self.assertIn("_sj.index_stems(self.file_path", src)


class TestLedgerCountsShards(_Store):
    def test_the_first_boot_count_sees_the_shard_directories(self):
        for i in range(5):
            self.page(_stem(i))
        self.assertEqual(SJ.index_entries(self.d), (5, "file count, no snapshot yet"))


if __name__ == "__main__":
    unittest.main()
