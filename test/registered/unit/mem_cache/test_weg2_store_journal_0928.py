"""The L3 store's persistent index (28.09., user order): snapshot + write-ahead journals.

27B 18:51: the wake rescan walked 821,711 files (4.7 s at 700k, ~18 s projected
at the 150 GB cap) on every wake. Now: an attach LOADS snapshot + journals, a
wake reads the other writers' new lines, and the directory is walked ONCE only
when the snapshot is missing/broken or the kernel boot id changed.
"""

import os
import tempfile
import unittest
from unittest import mock

from sglang.srt.mem_cache.storage.file import store_journal as SJ
from sglang.srt.mem_cache.storage.file.lru_file_evictor import LRUFileEvictor

SHARED = "_K"
PAGE = 4096


def _ev(d, cap=10 ** 9, walks=None, tp=0, group="D"):
    def _iter_existing():
        if walks is not None:
            walks.append(1)
        for name in sorted(os.listdir(d)):
            if name.endswith(".bin"):
                yield name[:-4], os.stat(os.path.join(d, name))

    with mock.patch.dict(os.environ, {"SGLANG_WEG2_GROUP": group}):
        return LRUFileEvictor(d, SHARED, tp_rank=tp, pp_rank=0, attn_cp_rank=0,
                              writes_shared_keys=True, scan_suffixes=(SHARED,),
                              extra_config={"max_size": str(cap), "max_size_scope": "shared"},
                              writer_count=1, iter_existing=_iter_existing)


def _write(d, stem, n=PAGE, append=False):
    with open(os.path.join(d, stem + ".bin"), "ab" if append else "wb") as f:
        f.write(b"\x07" * n)


def _fill(ev, d, stems):
    for s in stems:
        assert ev.reserve(s, PAGE, key=s, owner_writes_whole_file=False)
        _write(d, s)
        ev.commit(s)


def _cov(ev):
    c = ev.index_coverage()
    return c["indexed_entries"], c["indexed_bytes"], c["seen_entries"], c["seen_bytes"]


class _Env(unittest.TestCase):
    KEYS = (SJ.ENV, "SGLANG_WEG2_L3_EPOCH", "SGLANG_WEG2_GROUP")

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

    def reboot(self, epoch):
        os.environ["SGLANG_WEG2_L3_EPOCH"] = str(epoch)


class TestWakeDelta(_Env):
    def test_the_wake_reads_the_siblings_lines_and_walks_nothing(self):
        walks = []
        me = _ev(self.d, walks=walks)
        sib = _ev(self.d, group="P")
        _fill(sib, self.d, [f"s{i:04d}{SHARED}" for i in range(30)])
        n = len(walks)
        census = me.rescan()
        self.assertEqual(len(walks), n, "the wake walked the directory")
        self.assertEqual(census.get("mode"), "journal")
        self.assertEqual(census["indexed_entries"], 30)

    def test_an_unlink_line_removes_the_page(self):
        me = _ev(self.d)
        sib = _ev(self.d, cap=10 * PAGE, group="P")    # small cap: it evicts its own
        _fill(sib, self.d, [f"s{i:04d}{SHARED}" for i in range(10)])
        me.rescan()
        _fill(sib, self.d, [f"t{i:04d}{SHARED}" for i in range(5)])
        me.rescan()
        on_disk = sum(1 for x in os.listdir(self.d) if x.endswith(".bin"))
        self.assertEqual(me.index_coverage()["indexed_entries"], on_disk)

    def test_a_shrunk_journal_is_named_never_walked(self):
        walks = []
        me = _ev(self.d, walks=walks)
        sib = _ev(self.d, group="P")
        _fill(sib, self.d, [f"s{i:04d}{SHARED}" for i in range(3)])
        me.rescan()
        os.truncate(sib._journal.path, 0)
        n = len(walks)
        me.rescan()
        self.assertEqual(len(walks), n)


class TestPersistence(_Env):
    def test_first_attach_walks_once_and_writes_the_snapshot(self):
        _write(self.d, f"old0000{SHARED}")
        walks = []
        _ev(self.d, walks=walks)
        self.assertEqual(len(walks), 1)
        self.assertTrue(os.path.exists(os.path.join(self.d, SJ.SNAP)))

    def test_a_reboot_loads_and_equals_a_walk(self):
        a = _ev(self.d)
        sib = _ev(self.d, group="P")
        _fill(a, self.d, [f"a{i:04d}{SHARED}" for i in range(7)])
        _fill(sib, self.d, [f"s{i:04d}{SHARED}" for i in range(11)])
        _write(self.d, "priv0000_B", 3 * PAGE)
        sib._journal.write("R", 0.0, 3 * PAGE, "priv0000_B")
        sib._journal.write("C", 0.0, 3 * PAGE, "priv0000_B")
        self.reboot(200)
        walks = []
        b = _ev(self.d, walks=walks)
        self.assertEqual(len(walks), 0, "the reboot walked although the snapshot holds")
        os.environ[SJ.ENV] = "0"
        ref = _ev(self.d)
        self.assertEqual(_cov(b), _cov(ref))

    def test_a_broken_snapshot_walks_once(self):
        _ev(self.d)
        with open(os.path.join(self.d, SJ.SNAP), "a") as f:
            f.write("garbage\n")
        self.reboot(200)
        walks = []
        _ev(self.d, walks=walks)
        self.assertEqual(len(walks), 1)

    def test_a_changed_kernel_boot_id_walks_once(self):
        _ev(self.d)
        self.reboot(200)
        walks = []
        with mock.patch.object(SJ, "kernel_boot_id", lambda: "another-boot"):
            _ev(self.d, walks=walks)
        self.assertEqual(len(walks), 1)

    def test_a_writer_without_a_journal_walks_once(self):
        # rc12z30c review: the bridge images of 28.09. (rc12z29b/d) write the
        # same L3 without journals; their pages are invisible to snapshot +
        # journals (never evicted, never moved by a revoked window)
        _ev(self.d)
        shard = os.path.join(self.d, "ab")
        os.makedirs(shard)
        with open(os.path.join(shard, f"ab01{SHARED}.bin"), "wb") as f:
            f.write(b"\x07" * PAGE)
        newest = max(os.stat(p).st_mtime for p in
                     [os.path.join(self.d, SJ.SNAP)] + SJ.journal_paths(self.d))
        os.utime(shard, (newest + 60, newest + 60))
        self.reboot(200)
        items, why = SJ.load_index(self.d)
        self.assertIsNone(items)
        self.assertIn("without a journal", why)
        walks = []
        _ev(self.d, walks=walks)
        self.assertEqual(len(walks), 1)

    def test_a_journaled_shard_change_still_loads(self):
        _ev(self.d)
        shard = os.path.join(self.d, "cd")
        os.makedirs(shard)
        newest = max(os.stat(p).st_mtime for p in
                     [os.path.join(self.d, SJ.SNAP)] + SJ.journal_paths(self.d))
        os.utime(shard, (newest - 1, newest - 1))
        self.reboot(200)
        items, why = SJ.load_index(self.d)
        self.assertIsNotNone(items, why)

    def test_compaction_by_the_d_owner_only(self):
        _ev(self.d)
        p = _ev(self.d, group="P")
        _fill(p, self.d, [f"p{i:04d}{SHARED}" for i in range(2)])
        old = p._journal.path
        self.reboot(200)
        _ev(self.d, group="P")                          # P loads, never compacts
        self.assertTrue(os.path.exists(old))
        _ev(self.d, group="D")                          # D writes the snapshot, then compacts
        self.assertFalse(os.path.exists(old))
        self.reboot(300)
        walks = []
        c = _ev(self.d, walks=walks)
        self.assertEqual(len(walks), 0)
        self.assertEqual(c.index_coverage()["indexed_entries"], 2)


class TestCrashOrder(_Env):
    def test_an_intent_without_its_commit_is_resolved_by_one_stat(self):
        a = _ev(self.d)
        w = _ev(self.d, tp=1, group="D")
        for stem, lands in ((f"x0000{SHARED}", True), (f"y0000{SHARED}", False)):
            w._journal.write("R", 1.0, PAGE, stem)       # the process died after this line
            if lands:
                _write(self.d, stem)
        self.reboot(200)
        b = _ev(self.d)
        stems = set(b._lru)
        self.assertIn(f"x0000{SHARED}", stems, "a landed page without its C line was lost")
        self.assertNotIn(f"y0000{SHARED}", stems, "an intent whose file never landed was indexed")

    def test_enoent_strikes_the_entry_in_both_indexes(self):
        me = _ev(self.d)
        sib = _ev(self.d, group="P")
        _fill(me, self.d, [f"m0000{SHARED}"])
        sib.rescan()
        os.unlink(os.path.join(self.d, f"m0000{SHARED}.bin"))   # unlinked, E line lost
        self.assertEqual(me.forget([f"m0000{SHARED}"]), 1)
        sib.rescan()
        self.assertNotIn(f"m0000{SHARED}", sib._lru)


class TestF14TwoOwnersOfOnePage(_Env):
    def test_both_workers_extents_land_in_the_index(self):
        _ev(self.d)                                      # the D owner (TP0)
        w1 = _ev(self.d, tp=1)
        w2 = _ev(self.d, tp=2)
        stem = f"f14page{SHARED}"
        assert w1.reserve(stem, PAGE, key=stem, owner_writes_whole_file=False)
        _write(self.d, stem, PAGE)
        w1.commit(stem)
        assert w2.reserve(stem, PAGE, key=stem, owner_writes_whole_file=False)
        _write(self.d, stem, PAGE, append=True)
        w2.commit(stem)
        self.reboot(200)
        walks = []
        b = _ev(self.d, walks=walks)
        self.assertEqual(len(walks), 0)
        self.assertIn(stem, b._lru)
        self.assertEqual(b._lru[stem], os.stat(os.path.join(self.d, stem + ".bin")).st_size)


class TestSwitchAndNeighbours(_Env):
    def test_switch_zero_walks_and_writes_no_journal(self):
        os.environ[SJ.ENV] = "0"
        walks = []
        me = _ev(self.d, walks=walks)
        me.rescan()
        self.assertEqual(len(walks), 2)
        self.assertFalse([x for x in os.listdir(self.d) if x.endswith(SJ.SUFFIX)])

    def test_a_neighbour_stores_writes_never_reach_this_index(self):
        with tempfile.TemporaryDirectory() as other:
            me = _ev(self.d)
            nb = _ev(other, group="P")
            _fill(nb, other, [f"n{i:04d}{SHARED}" for i in range(20)])
            census = me.rescan()
            self.assertEqual(census["indexed_entries"], 0)
            self.assertEqual(census["seen_bytes"], 0)


class TestSnapshotFormatAndLedger(_Env):
    def test_pickled_snapshot_roundtrip_and_a_torn_tmp_is_removed(self):
        items = [(1.0 + i, f"s{i:04d}{SHARED}", PAGE * (i + 1)) for i in range(5)]
        torn = os.path.join(self.d, SJ.SNAP + ".w1.dead00.tmp")
        open(torn, "wb").write(b"half")
        self.assertEqual(SJ.remove_snapshot_tmps(self.d), 1)
        self.assertFalse(os.path.exists(torn))
        SJ.write_snapshot(self.d, items, 100)
        snap, why = SJ.load_snapshot(self.d)
        self.assertEqual(why, "")
        self.assertEqual(snap.epoch, 100)
        self.assertEqual(list(snap.items), [s for _, s, _ in items])   # LRU order kept
        self.assertEqual(snap.items[f"s0002{SHARED}"], (3.0, 3 * PAGE))
        with open(os.path.join(self.d, SJ.SNAP), "r+b") as f:  # one flipped payload byte
            f.seek(-3, os.SEEK_END)
            b = f.read(1)
            f.seek(-3, os.SEEK_END)
            f.write(bytes([b[0] ^ 1]))
        self.assertIsNone(SJ.load_snapshot(self.d)[0])

    def test_the_ledger_counts_entries_from_the_header_else_the_files(self):
        _write(self.d, f"a0000{SHARED}")
        _write(self.d, f"a0001{SHARED}")
        self.assertEqual(SJ.index_entries(self.d), (2, "file count, no snapshot yet"))
        SJ.write_snapshot(self.d, [(1.0, "x", 1)] * 7, 100)
        self.assertEqual(SJ.index_entries(self.d), (7, "snapshot header"))

    def test_the_ledger_post_is_charged_at_both_moments(self):
        from sglang.srt.weg2 import host_ledger as HL
        t = {k: 0.0 for k in ("heaps_gib", "anchors_gib", "rings_gib", "overhead_gib",
                              "draft_host_p_gib", "draft_host_d_gib", "xchg_bounce_gib")}
        t["flip_ratchet_gib"] = None
        base = HL._boot_charges_gib(dict(t))
        t["l3_index_gib"] = 0.5
        self.assertAlmostEqual(HL._boot_charges_gib(t) - base, 0.5)
        self.assertAlmostEqual(HL._run_moment_charges_gib(t) - base, 0.5)


if __name__ == "__main__":
    unittest.main()
