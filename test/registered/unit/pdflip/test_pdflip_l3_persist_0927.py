# SPDX-License-Identifier: Apache-2.0
"""L3P (user 2026-09-27): the disk L3 store survives boots, one store per model.

User, verbatim: "der l3 hicache wird jedesmal verworfen, oder? der soll
natürlich persistent sein" and "aber natürlich zwischen 27b und nf verschiedene
L3 caches. sonst knallts".

THE DEFECT: ``plan_store`` named the store directory after the BOOT TAG and
``sweep_store_residue`` rmtree'd every other directory under the root at the
next boot, so every boot started with an empty L3 (NF rc12r's mount held only
``dkrnfh91dprbar1dauer09271632``). A second, hidden half: the shared L3 stem
index (#1459) answers "not on disk" for any stem it does not know, and a freshly
created index knows none -- so even a kept directory would read as all misses.

Hermetic: no GPU, no server, no boot; every disk fact is a temp directory or a
fake statvfs.
"""

import inspect
import json
import os
import tempfile
import time
import unittest
from unittest import mock

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

from flliper.srt.pdflip import host_ledger, launcher
from flliper.test.test_utils import CustomTestCase

GIB = host_ledger.GIB
POOL = 304950
ATTN = (11, 2, 3)
KV_MIB = 2048 / float(1024 * 1024)


class _FakeStatvfs:
    def __init__(self, free, total):
        self.f_frsize = 1
        self.f_bavail = free
        self.f_blocks = total
        self.f_bfree = free


class _Log:
    def __init__(self):
        self.lines = []

    def __call__(self, msg):
        self.lines.append(str(msg))

    @property
    def text(self):
        return "\n".join(self.lines)


def _model(root, name, cfg):
    d = os.path.join(root, name)
    os.makedirs(d, exist_ok=True)
    with open(os.path.join(d, "config.json"), "w") as f:
        json.dump(cfg, f)
    return d


def _write(path, n):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as f:
        f.write(b"\0" * n)


class TestSwitch(CustomTestCase):
    def test_default_on_and_zero_is_the_old_per_boot_store(self):
        self.assertTrue(launcher.l3_persist_enabled({}))
        for off in ("0", "false", "off", "no"):
            self.assertFalse(launcher.l3_persist_enabled({"FLLIPER_PDFLIP_L3_PERSIST": off}))


class TestOneStorePerModel(CustomTestCase):
    def test_27b_and_nf_get_two_named_directories(self):
        with tempfile.TemporaryDirectory() as m:
            a = _model(m, "Qwen3.8-27B-INT8", {"model_type": "qwen3_5"})
            b = _model(m, "Qwen3.8-NextFlash-INT4", {"model_type": "qwen4_exp"})
            ia = launcher.l3_persist_identity(a, "qwen27b")
            ib = launcher.l3_persist_identity(b, "nextflash")
            na, nb = launcher.l3_persist_dir_name(ia), launcher.l3_persist_dir_name(ib)
            self.assertNotEqual(na, nb)
            self.assertTrue(na.startswith("l3-qwen27b-Qwen3.8-27B-INT8-"), na)
            self.assertTrue(nb.startswith("l3-nextflash-Qwen3.8-NextFlash-INT4-"), nb)
            # stable across boots: same identity, same name
            self.assertEqual(na, launcher.l3_persist_dir_name(
                launcher.l3_persist_identity(a, "qwen27b")))

    def test_a_different_kv_format_or_generation_is_a_different_directory(self):
        with tempfile.TemporaryDirectory() as m:
            a = _model(m, "M", {"x": 1})
            base = launcher.l3_persist_dir_name(launcher.l3_persist_identity(a, "p"))
            self.assertNotEqual(base, launcher.l3_persist_dir_name(
                launcher.l3_persist_identity(a, "p", form_kv="int8")))
            self.assertNotEqual(base, launcher.l3_persist_dir_name(
                launcher.l3_persist_identity(a, "p", kv_cache_dtype="bf16")))
            with mock.patch.object(launcher, "L3_PERSIST_GENERATION", "999"):
                self.assertNotEqual(base, launcher.l3_persist_dir_name(
                    launcher.l3_persist_identity(a, "p")))
            # a re-quantised checkpoint at the same path: the config moved
            _model(m, "M", {"x": 2})
            self.assertNotEqual(base, launcher.l3_persist_dir_name(
                launcher.l3_persist_identity(a, "p")))

    def test_the_identity_file_is_written_then_checked_and_a_foreign_one_refused(self):
        with tempfile.TemporaryDirectory() as root, tempfile.TemporaryDirectory() as m:
            ia = launcher.l3_persist_identity(_model(m, "A", {"a": 1}), "qwen27b")
            ib = launcher.l3_persist_identity(_model(m, "B", {"b": 1}), "nextflash")
            d = os.path.join(root, launcher.l3_persist_dir_name(ia))
            self.assertEqual(launcher.l3_persist_check_identity(d, ia, dry=False), "new")
            self.assertEqual(launcher.l3_persist_check_identity(d, ia, dry=False), "match")
            with self.assertRaises(launcher.PdFlipStoreDiskRefused) as cm:
                launcher.l3_persist_check_identity(d, ib, dry=False)
            self.assertIn("belongs to another identity", str(cm.exception))
            self.assertIn("model_path", str(cm.exception))

    def test_dry_run_writes_no_identity_file(self):
        with tempfile.TemporaryDirectory() as root, tempfile.TemporaryDirectory() as m:
            ia = launcher.l3_persist_identity(_model(m, "A", {}), "p")
            d = os.path.join(root, "l3-x")
            self.assertEqual(launcher.l3_persist_check_identity(d, ia, dry=True), "new")
            self.assertFalse(os.path.exists(d))


class TestTheSweepSparesEveryPersistentStore(CustomTestCase):
    def test_tag_dirs_go_both_models_l3_dirs_stay(self):
        with tempfile.TemporaryDirectory() as root:
            keep = os.path.join(root, "l3-qwen27b-A-0123456789")
            other = os.path.join(root, "l3-nextflash-B-9876543210")
            old_tag = os.path.join(root, "dkrnfh91dprbar1dauer09271632")
            for d in (keep, other, old_tag):
                _write(os.path.join(d, "ab", "x.bin"), 4096)
            log = _Log()
            launcher.sweep_store_residue(log, keep, dry=False, root=root)
            self.assertTrue(os.path.isdir(keep))
            self.assertTrue(os.path.isdir(other), "the other model's L3 was swept")
            self.assertFalse(os.path.exists(old_tag))


class TestAttach(CustomTestCase):
    def test_reuse_counts_pages_and_drops_crash_staging(self):
        with tempfile.TemporaryDirectory() as root:
            d = os.path.join(root, "l3-p-m-0000000000")
            _write(os.path.join(d, "ab", "k1_s.bin"), 8192)
            _write(os.path.join(d, "cd", "k2_s.bin"), 8192)
            _write(os.path.join(d, "cd", "k3_s.bin.tmp.deadbeef"), 8192)
            log = _Log()
            files, nbytes, removed = launcher.l3_persist_attach(log, d, dry=False)
            self.assertEqual((files, removed), (2, 1))
            self.assertGreater(nbytes, 0)
            self.assertFalse(os.path.exists(os.path.join(d, "cd", "k3_s.bin.tmp.deadbeef")))
            self.assertIn("L3-PERSIST reuse dir=", log.text)
            self.assertIn("files=2", log.text)
            self.assertIn("staging+orphans_removed=1", log.text)

    def test_dry_run_removes_nothing(self):
        with tempfile.TemporaryDirectory() as root:
            d = os.path.join(root, "l3-x")
            t = os.path.join(d, "ab", "k.bin.tmp.1")
            _write(t, 10)
            launcher.l3_persist_attach(_Log(), d, dry=True)
            self.assertTrue(os.path.exists(t))

    def test_an_absent_directory_is_fresh(self):
        log = _Log()
        self.assertEqual(launcher.l3_persist_attach(log, "/nonexistent/l3-x", dry=False), (0, 0, 0))
        self.assertIn("L3-PERSIST fresh", log.text)


class TestTheDiskCheckCreditsTheStoresOwnBytes(CustomTestCase):
    def test_a_full_persistent_store_does_not_refuse_its_own_reboot(self):
        with tempfile.TemporaryDirectory() as root:
            name = "l3-p-m-0000000000"
            _write(os.path.join(root, name, "ab", "k.bin"), 4 * 1024 * 1024)
            used = launcher._tree_bytes(os.path.join(root, name))[0]
            cap_gb = 1.0
            min_free = 0.001
            need = int(cap_gb * 1e9) + int(min_free * GIB)
            free = need - used // 2          # short by less than what the store holds
            kw = dict(root=root, store_max_gb=cap_gb, min_free_gib=min_free)
            with mock.patch.object(os, "statvfs", return_value=_FakeStatvfs(free, 10 * need)):
                with self.assertRaises(launcher.PdFlipStoreDiskRefused):
                    launcher.plan_store("tag", 1, (1,), 1e-6, **kw)   # per-boot: no credit
                plan = launcher.plan_store("tag", 1, (1,), 1e-6, directory_name=name, **kw)
            self.assertEqual(plan.directory, os.path.join(root, name))
            self.assertEqual(plan.reused_bytes, used)

    def test_no_walk_when_the_disk_is_roomy(self):
        with tempfile.TemporaryDirectory() as root:
            with mock.patch.object(os, "statvfs", return_value=_FakeStatvfs(10 ** 15, 10 ** 15)), \
                    mock.patch.object(launcher, "_tree_bytes") as tb:
                plan = launcher.plan_store("tag", POOL, ATTN, KV_MIB, root=root,
                                           directory_name="l3-p-m-1")
            tb.assert_not_called()
            self.assertEqual(plan.reused_bytes, 0)
            self.assertTrue(plan.directory.endswith("/l3-p-m-1"))

    def test_without_a_name_the_directory_is_still_the_tag(self):
        with tempfile.TemporaryDirectory() as root:
            with mock.patch.object(os, "statvfs", return_value=_FakeStatvfs(10 ** 15, 10 ** 15)):
                plan = launcher.plan_store("tag", POOL, ATTN, KV_MIB, root=root)
            self.assertTrue(plan.directory.endswith("/tag"))


class TestIndexSeed(CustomTestCase):
    def _store(self, stems):
        from flliper.srt.mem_cache.hicache_storage import HiCacheFile

        fake = mock.MagicMock()
        fake.file_path = "/x"
        fake._iter_existing_files = lambda: iter((s, None) for s in stems)
        return HiCacheFile._l3p_seed_index, fake

    def test_a_created_index_is_seeded_from_the_disk(self):
        fn, fake = self._store([f"k{i}_s" for i in range(5000)])
        added = []
        idx = mock.MagicMock()
        idx.add.side_effect = lambda b: added.extend(b)
        with mock.patch.dict(os.environ, {"FLLIPER_PDFLIP_L3_PERSIST": "1"}):
            n = fn(fake, idx)
        self.assertEqual(n, 5000)
        self.assertEqual(len(added), 5000)
        self.assertEqual(idx.add.call_count, 2)          # batches of 4096

    def test_switch_off_seeds_nothing(self):
        fn, fake = self._store(["k_s"])
        idx = mock.MagicMock()
        with mock.patch.dict(os.environ, {"FLLIPER_PDFLIP_L3_PERSIST": "0"}):
            self.assertEqual(fn(fake, idx), 0)
        idx.add.assert_not_called()

    def test_only_the_creator_seeds(self):
        from flliper.srt.mem_cache.hicache_storage import HiCacheFile

        src = inspect.getsource(HiCacheFile._l3_index)
        self.assertIn("if idx.created and self._l3p_on():", src)
        self.assertIn("threading.Thread(target=self._l3p_seed_index", src)

    def test_the_index_is_opened_at_backend_init_not_lazily_in_a_collective(self):
        from flliper.srt.mem_cache.hicache_storage import HiCacheFile

        src = inspect.getsource(HiCacheFile.__init__)
        self.assertIn("self._l3p_open_index_eagerly()", src)
        self.assertLess(src.index("self._l3p_check_rank_identity(storage_config)"),
                        src.index("self._evictor = LRUFileEvictor("))


class TestWiring(CustomTestCase):
    def test_identity_before_attach_before_prepare_and_the_name_reaches_plan_store(self):
        src = inspect.getsource(launcher.main) if hasattr(launcher, "main") else ""
        if "l3_persist_check_identity" not in src:
            src = open(launcher.__file__).read()
        i_id = src.index("l3_persist_check_identity(store_plan.directory")
        i_at = src.index("l3_persist_attach(log, store_plan.directory")
        i_pr = src.index("store_dir = prepare_store(log, store_plan, dry)")
        i_sw = src.index("sweep_store_residue(log, store_plan.directory, dry)")
        self.assertLess(i_sw, i_id)
        self.assertLess(i_id, i_at)
        self.assertLess(i_at, i_pr)
        self.assertIn("directory_name=(l3_persist_dir_name(_l3_ident) if _l3_ident else None)", src)


class TestW8bOnAWarmPersistentStore(CustomTestCase):
    """The warm-store W8b refusal pinned by test_pdflip_store_cap_1295 is the
    day the retention tier is reused -- it is today. Inherited = unindexed,
    older than the launcher's GROUP-WIDE epoch (N2), and under a suffix the
    previous boot's groups recorded (N1, published by the launcher). A blind
    own filter and orphan suffixes are NOT inherited."""

    PAGE = 4096

    def _dir(self, root, sibling="_B", identity=True):
        if identity:
            with open(os.path.join(root, "L3_IDENTITY.json"), "w") as f:
                f.write("{}")
        for i in range(4):
            _write(os.path.join(root, f"{i:04d}_A.bin"), self.PAGE)
        for i in range(6):
            _write(os.path.join(root, f"{i:04d}{sibling}.bin"), self.PAGE)
        past = time.time() - 3600
        for n in os.listdir(root):
            os.utime(os.path.join(root, n), (past, past))

    def _env(self, suffixes=("_A", "_B"), epoch=None, persist="1"):
        return mock.patch.dict(os.environ, {
            "FLLIPER_PDFLIP_L3_PERSIST": persist,
            "FLLIPER_PDFLIP_L3_EPOCH": f"{time.time() if epoch is None else epoch:.6f}",
            "FLLIPER_PDFLIP_L3_INHERITED_SUFFIXES": json.dumps(list(suffixes)),
        })

    def _owner(self, root):
        from flliper.srt.mem_cache.storage.file.lru_file_evictor import LRUFileEvictor

        return LRUFileEvictor(
            root, "_A", tp_rank=0, pp_rank=0, attn_cp_rank=0,
            writes_shared_keys=True, scan_suffixes=("_A",),
            extra_config={"max_size": str(100 * self.PAGE), "max_size_scope": "shared"},
            writer_count=1)

    def _blind(self):
        from flliper.srt.mem_cache.pdflip_store_gates import PdFlipStoreIndexBlind

        return self.assertRaises(PdFlipStoreIndexBlind)

    def test_60_percent_sibling_private_pages_are_inherited_not_blind(self):
        with tempfile.TemporaryDirectory() as root, self._env():
            self._dir(root)
            ev = self._owner(root)                       # must not raise W8b
            cov = ev.index_coverage()
            self.assertEqual(cov["inherited_entries"], 6)
            self.assertEqual(cov["indexed_entries"], 4)
            self.assertEqual(cov["seen_entries"], 10)
            self.assertIn("walk_s", cov)

    def test_an_orphan_suffix_no_group_recorded_is_not_inherited(self):
        with tempfile.TemporaryDirectory() as root, self._env(suffixes=("_A", "_B")):
            self._dir(root, sibling="_ORPHAN")
            with self._blind():
                self._owner(root)

    def test_a_blind_own_filter_across_boots_is_still_refused(self):
        # this group wrote its pages under _X last boot while scanning _A (the
        # 83.7 % defect shape): _X is in NO record, so it is not inherited
        with tempfile.TemporaryDirectory() as root, self._env(suffixes=("_A",)):
            self._dir(root, sibling="_X")
            with self._blind():
                self._owner(root)

    def test_without_the_identity_file_the_old_refusal_stands(self):
        with tempfile.TemporaryDirectory() as root, self._env():
            self._dir(root, identity=False)
            with self._blind():
                self._owner(root)

    def test_switch_off_keeps_the_old_refusal(self):
        with tempfile.TemporaryDirectory() as root, self._env(persist="0"):
            self._dir(root)
            with self._blind():
                self._owner(root)

    def test_files_newer_than_the_group_epoch_are_not_inherited(self):
        with tempfile.TemporaryDirectory() as root, self._env(epoch=time.time() - 7200):
            self._dir(root)                           # files 1 h old, epoch 2 h old
            with self._blind():
                self._owner(root)

    def test_no_published_epoch_inherits_nothing(self):
        with tempfile.TemporaryDirectory() as root, \
                mock.patch.dict(os.environ, {"FLLIPER_PDFLIP_L3_PERSIST": "1"}):
            os.environ.pop("FLLIPER_PDFLIP_L3_EPOCH", None)
            os.environ.pop("FLLIPER_PDFLIP_L3_INHERITED_SUFFIXES", None)
            self._dir(root)
            with self._blind():
                self._owner(root)


class TestOrphanReapAndPublication(CustomTestCase):
    def test_orphans_reaped_only_with_a_record_and_the_inherited_set_published(self):
        with tempfile.TemporaryDirectory() as root:
            d = os.path.join(root, "l3-x")
            _write(os.path.join(d, "ab", "k1_A.bin"), 4096)
            _write(os.path.join(d, "ab", "k2_B.bin"), 4096)
            _write(os.path.join(d, "ab", "k3_OLDHASH.bin"), 4096)
            with mock.patch.dict(os.environ, {}, clear=False):
                log = _Log()
                files, _b, removed = launcher.l3_persist_attach(log, d, dry=False)
                self.assertEqual((files, removed), (3, 0))       # no record: counted, kept
                self.assertIn("orphans=0", log.text)
                self.assertNotIn("FLLIPER_PDFLIP_L3_INHERITED_SUFFIXES", os.environ)
                self.assertIn("FLLIPER_PDFLIP_L3_EPOCH", os.environ)
                for g, sfx in (("P", ["_A"]), ("D", ["_B"])):
                    with open(os.path.join(d, f"L3_SUFFIXES.{g}.json"), "w") as f:
                        json.dump({"group": g, "suffixes": sfx}, f)
                log = _Log()
                files, _b, removed = launcher.l3_persist_attach(log, d, dry=False)
                self.assertEqual((files, removed), (2, 1))
                self.assertFalse(os.path.exists(os.path.join(d, "ab", "k3_OLDHASH.bin")))
                self.assertIn("orphans=1", log.text)
                # the first attach's one walk wrote the snapshot: this one reads it
                self.assertIn("source=index", log.text)
                self.assertEqual(sorted(json.loads(os.environ["FLLIPER_PDFLIP_L3_INHERITED_SUFFIXES"])),
                                 ["_A", "_B"])


class TestRankSide(CustomTestCase):
    def _fake(self, root, ident, group="P"):
        from flliper.srt.mem_cache.hicache_storage import HiCacheFile

        fake = mock.MagicMock()
        fake.file_path = root
        fake._l3p_on = HiCacheFile._l3p_on
        fake._l3p_group = HiCacheFile._l3p_group
        fake._l3p_persistent_dir = lambda: HiCacheFile._l3p_persistent_dir(fake)
        cfg = mock.MagicMock()
        cfg.l3_rank_identity = ident
        return HiCacheFile._l3p_check_rank_identity, fake, cfg

    def test_rank_identity_recorded_then_matched_then_a_drift_refused(self):
        from flliper.srt.mem_cache.pdflip_store_gates import PdFlipL3IdentityMismatch

        with tempfile.TemporaryDirectory() as root, mock.patch.dict(
                os.environ, {"FLLIPER_PDFLIP_L3_PERSIST": "1", "FLLIPER_PDFLIP_GROUP": "D"}):
            with open(os.path.join(root, "L3_IDENTITY.json"), "w") as f:
                f.write("{}")
            ident = {"model_identity": "a", "override_sha": "", "weights_fp": "w"}
            fn, fake, cfg = self._fake(root, ident)
            fn(fake, cfg)
            self.assertTrue(os.path.isfile(os.path.join(root, "L3_RANK_IDENTITY.D.json")))
            fn(fake, cfg)                                  # match
            fn2, fake2, cfg2 = self._fake(root, dict(ident, override_sha="yarn2"))
            with self.assertRaises(PdFlipL3IdentityMismatch) as cm:
                fn2(fake2, cfg2)
            self.assertIn("W165 PdFlipL3IdentityMismatch", str(cm.exception))

    def test_not_a_persistent_dir_checks_nothing(self):
        with tempfile.TemporaryDirectory() as root, mock.patch.dict(
                os.environ, {"FLLIPER_PDFLIP_L3_PERSIST": "1", "FLLIPER_PDFLIP_GROUP": "D"}):
            fn, fake, cfg = self._fake(root, {"x": 1})
            fn(fake, cfg)
            self.assertEqual(os.listdir(root), [])

    def test_rank_identity_moves_with_override_and_weights_not_with_ratio(self):
        from flliper.srt.mem_cache.hicache_storage import l3_rank_identity

        with tempfile.TemporaryDirectory() as m:
            _write(os.path.join(m, "model-00001.safetensors"), 16)
            sa = mock.MagicMock(model_path=m, revision=None, dtype="bfloat16", quantization="",
                                kv_cache_dtype="fp8_e4m3", json_model_override_args="{}",
                                rank_tp_ratio="13,6,6", rank_kv_ratio=None)
            base = l3_rank_identity(sa)
            sa.rank_tp_ratio = "12,7,7"
            self.assertEqual(base, l3_rank_identity(sa))
            sa.json_model_override_args = '{"rope_scaling": {"factor": 2}}'
            self.assertNotEqual(base["override_sha"], l3_rank_identity(sa)["override_sha"])
            sa.json_model_override_args = "{}"
            os.utime(os.path.join(m, "model-00001.safetensors"), (1, 1))
            self.assertNotEqual(base["weights_fp"], l3_rank_identity(sa)["weights_fp"])

    def test_suffix_record_written_for_the_group(self):
        from flliper.srt.mem_cache.hicache_storage import HiCacheFile

        with tempfile.TemporaryDirectory() as root, mock.patch.dict(
                os.environ, {"FLLIPER_PDFLIP_L3_PERSIST": "1", "FLLIPER_PDFLIP_GROUP": "P"}):
            with open(os.path.join(root, "L3_IDENTITY.json"), "w") as f:
                f.write("{}")
            fake = mock.MagicMock()
            fake.file_path = root
            fake._l3p_group = HiCacheFile._l3p_group
            fake._l3p_persistent_dir = lambda: HiCacheFile._l3p_persistent_dir(fake)
            fake._group_scan_suffixes = lambda: ("_A", "_A2")
            HiCacheFile._l3p_register_suffixes(fake, True)
            with open(os.path.join(root, "L3_SUFFIXES.P.json")) as f:
                self.assertEqual(json.load(f)["suffixes"], ["_A", "_A2"])


class TestLauncherIdentityN4(CustomTestCase):
    def test_yarn_override_kv_extra_vision_and_swapped_weights_move_the_directory(self):
        with tempfile.TemporaryDirectory() as m:
            a = _model(m, "NF", {"x": 1})
            _write(os.path.join(a, "w.safetensors"), 8)
            name = lambda **kw: launcher.l3_persist_dir_name(launcher.l3_persist_identity(a, "nextflash", **kw))
            base = name()
            self.assertNotEqual(base, name(extra_p='--json-model-override-args \'{"rope_scaling":{"factor":2}}\''))
            self.assertNotEqual(base, name(extra_d="--kv-cache-dtype bf16"))
            self.assertNotEqual(base, name(vision="transient"))
            self.assertEqual(base, name(extra_p="--some-other-flag 1"))
            os.utime(os.path.join(a, "w.safetensors"), (1, 1))
            self.assertNotEqual(base, name())


class TestDiskSum(CustomTestCase):
    """User 2026-09-27 "L3, jeder 150gb": two 150 GB stores share one disk."""

    def _mk(self, root, name, cap, have):
        d = os.path.join(root, name)
        os.makedirs(d, exist_ok=True)
        with open(os.path.join(d, launcher.L3_CAP_FILE), "w") as f:
            json.dump({"max_size_bytes": cap, "bytes_at_attach": have, "utc": "t"}, f)
        return d

    def test_two_150_gb_stores_fit_the_nvme(self):
        with tempfile.TemporaryDirectory() as root:
            self._mk(root, "l3-nextflash-B-1", 150 * 10 ** 9, 10 * 10 ** 9)
            own = os.path.join(root, "l3-qwen27b-A-1")
            os.makedirs(own)
            log = _Log()
            with mock.patch.object(os, "statvfs", return_value=_FakeStatvfs(1400 * 10 ** 9, 1800 * 10 ** 9)):
                short = launcher.l3_persist_disk_sum(log, root, own, 150 * 10 ** 9, 32 * GIB, 0, dry=False)
            self.assertEqual(short, 0)
            self.assertIn("L3-PERSIST DISK-SUM ok", log.text)
            self.assertIn("l3-nextflash-B-1 grow=140.0 GB", log.text)
            with open(os.path.join(own, launcher.L3_CAP_FILE)) as f:
                self.assertEqual(json.load(f)["max_size_bytes"], 150 * 10 ** 9)

    def test_a_disk_too_small_for_both_caps_is_a_named_warning_with_numbers(self):
        with tempfile.TemporaryDirectory() as root:
            self._mk(root, "l3-nextflash-B-1", 150 * 10 ** 9, 0)
            own = os.path.join(root, "l3-qwen27b-A-1")
            os.makedirs(own)
            log = _Log()
            with mock.patch.object(os, "statvfs", return_value=_FakeStatvfs(200 * 10 ** 9, 1800 * 10 ** 9)):
                short = launcher.l3_persist_disk_sum(log, root, own, 150 * 10 ** 9, 32 * GIB, 0, dry=False)
            self.assertGreater(short, 0)
            self.assertIn("L3-PERSIST DISK-SUM WARN", log.text)
            self.assertIn("SHORT by", log.text)
            self.assertIn("nothing is capped silently", log.text)

    def test_a_tag_dir_and_a_store_without_record_are_named_not_counted(self):
        with tempfile.TemporaryDirectory() as root:
            os.makedirs(os.path.join(root, "l3-old-x"))
            os.makedirs(os.path.join(root, "sometag"))
            own = os.path.join(root, "l3-a")
            os.makedirs(own)
            log = _Log()
            with mock.patch.object(os, "statvfs", return_value=_FakeStatvfs(10 ** 15, 10 ** 15)):
                launcher.l3_persist_disk_sum(log, root, own, 10, 0, 0, dry=True)
            self.assertIn("l3-old-x=no cap record", log.text)
            self.assertNotIn("sometag", log.text)
            self.assertFalse(os.path.exists(os.path.join(own, launcher.L3_CAP_FILE)))


if __name__ == "__main__":
    unittest.main()
