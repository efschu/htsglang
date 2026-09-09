# SPDX-License-Identifier: Apache-2.0
"""Weg 2 (#1236): the page store is a DIRECTORY ON DISK, sized from the P pool.

THE DEFECT, measured on boot ``weg2sb5g`` (2026-09-09, front/P logs in
``/spinning/evidence-665-f1``):

* the host ledger sized a tmpfs out of its RAM leftover and got **6 GiB**
  (``WEG2-HOST-LEDGER CHOSEN ... store=6 GiB tmpfs ... bound=reap``), of which
  ``max_size 5G`` + ``min_free_space 1G`` left **5 GiB usable**;
* group P's KV pool on that boot was **304,950 tokens** at **32,768 B/token**
  (``PP-POOL-JOIN ... pool_tokens=304950 ... cell_bytes=22528,4096,6144``,
  summing to 32,768) = **9.30 GiB**;
* so the carrier every prefix must travel through was **1.86x smaller than
  what one prefill leg produces**, and group P logged
  ``HiCacheFile ... would fall below min_free`` **1,324 times** (counted in
  ``boot_weg2_weg2sb5g_*.P.log``). The "#1236 Store >= P pool" law was violated
  before the first flip.

THE CUT UNDER TEST, and what it deliberately is NOT. It is not a new tier, not
an overflow mechanism and not a second bookkeeping -- user ruling 2026-09-09,
the store is "normales hicaching mit lvl2 und lvl3". The mechanism is the
UPSTREAM HiCache file backend that already exists (``HiCacheFile`` +
``LRUFileEvictor`` with ``max_size`` / ``min_free_space`` /
``max_size_scope``). The only things this branch changes are **where the
launcher points that backend** (a plain directory on the ZFS dataset under
``/spinning`` instead of a per-boot tmpfs) and **how it sizes it** (the P KV
pool's own bytes times a named sidecar factor, checked against the disk).

Everything downstream follows from those two: the host RAM ledger stops
charging a store term at all, ``--store-min-gib`` and ``size_store_gib`` are
deleted rather than re-pointed, and the one host-RAM risk that IS new -- the
ZFS ARC -- gets a preflight line and a narrow refusal.

Hermetic: ``CUDA_VISIBLE_DEVICES=""``, no server, no NVML, no GPU, no boot, no
real store. Every filesystem reading is a fake handed to the seam.
"""

import inspect
import json
import os
import unittest
from unittest import mock

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

from sglang.srt.weg2 import host_ledger, launcher
from sglang.test.test_utils import CustomTestCase

GIB = host_ledger.GIB

# --------------------------------------------------------- MEASURED: weg2sb5g
#: ``PP-POOL-JOIN`` of boot weg2sb5g, 2026-09-09T07:04:59Z.
SB5G_POOL_TOKENS = 304950
SB5G_ATTN = (11, 2, 3)                      # sums to 16 attention layers
#: 2048 B per token per attention layer -> 16 x 2048 = 32,768 B/token, which is
#: the ``cell_bytes=22528,4096,6144`` of that same line, summed.
KV_MIB_PER_TOKEN_PER_ATTN_LAYER = 2048 / float(1024 * 1024)
SB5G_CELL_BYTES = 32768
SB5G_POOL_BYTES = SB5G_POOL_TOKENS * SB5G_CELL_BYTES        # 9.30 GiB
#: What the tmpfs form gave it, and what it cost: 6 GiB mounted, 5 GiB usable.
SB5G_TMPFS_GIB = 6
SB5G_MIN_FREE_REFUSALS = 1324

#: A disk with room, and one without. ``statvfs`` fields the seam reads.
BIG_DISK_FREE = int(600 * GIB)
BIG_DISK_TOTAL = int(2300 * GIB)
SMALL_DISK_FREE = int(20 * GIB)


class _FakeStatvfs:
    def __init__(self, free_bytes, total_bytes):
        self.f_frsize = 4096
        self.f_bavail = free_bytes // 4096
        self.f_blocks = total_bytes // 4096
        self.f_bfree = self.f_bavail


def _plan(tmp, free=BIG_DISK_FREE, total=BIG_DISK_TOTAL, **over):
    """``plan_store`` against a fake disk -- no real statvfs, no real mkdir."""
    kw = dict(
        p_pool_tokens=SB5G_POOL_TOKENS,
        attn_counts=SB5G_ATTN,
        kv_mib_per_token_per_attn_layer=KV_MIB_PER_TOKEN_PER_ATTN_LAYER,
        root=tmp,
    )
    kw.update(over)
    with mock.patch.object(os, "statvfs", return_value=_FakeStatvfs(free, total)):
        return launcher.plan_store("testboot", **kw)


class _Log:
    def __init__(self):
        self.lines = []

    def __call__(self, msg):
        self.lines.append(str(msg))

    @property
    def text(self):
        return "\n".join(self.lines)


# =====================================================================
# 1. THE STORE IS SIZED FROM THE P POOL, NOT FROM HOST RAM
# =====================================================================


class TestSizedFromThePPool(CustomTestCase):
    def test_the_cell_comes_from_the_pool_model_not_a_typed_32_kib(self):
        # #1286 F1's rule, applied one layer up: the join key is the pool
        # model's own cell. A literal 32768 here would keep printing a
        # plausible number on any config with a different KV dtype or head
        # geometry while the pool itself had moved.
        with mock.patch.object(
            os, "statvfs", return_value=_FakeStatvfs(BIG_DISK_FREE, BIG_DISK_TOTAL)
        ):
            plan = launcher.plan_store(
                "t", SB5G_POOL_TOKENS, SB5G_ATTN,
                KV_MIB_PER_TOKEN_PER_ATTN_LAYER, root="/tmp",
            )
        self.assertEqual(plan.cell_bytes, SB5G_CELL_BYTES)
        self.assertEqual(plan.p_pool_bytes, SB5G_POOL_BYTES)
        # and a DIFFERENT geometry gives a different cell, which is the point.
        with mock.patch.object(
            os, "statvfs", return_value=_FakeStatvfs(BIG_DISK_FREE, BIG_DISK_TOTAL)
        ):
            other = launcher.plan_store(
                "t", SB5G_POOL_TOKENS, (8, 8, 8),
                KV_MIB_PER_TOKEN_PER_ATTN_LAYER, root="/tmp",
            )
        self.assertEqual(other.cell_bytes, 24 * 2048)
        self.assertNotEqual(other.cell_bytes, plan.cell_bytes)

    def test_the_law_holds_where_weg2sb5g_violated_it(self):
        # THE RED-FIRST FACT: sb5g's store was 0.54x its P pool. The same pool
        # through this seam is >= 1.0x, and by the sidecar factor above that.
        plan = _plan("/tmp")
        self.assertGreaterEqual(plan.max_size_bytes, plan.p_pool_bytes)
        self.assertGreater(
            plan.max_size_bytes / plan.p_pool_bytes, 1.0,
            "the store must exceed the P pool, not merely equal it",
        )
        sb5g_usable = (SB5G_TMPFS_GIB - 1) * GIB
        self.assertLess(sb5g_usable / SB5G_POOL_BYTES, 0.6)   # 0.54x, the defect
        self.assertGreater(plan.max_size_bytes / SB5G_POOL_BYTES, 1.0)

    def test_the_sidecar_factor_is_derived_from_a_named_census(self):
        # DERIVED, not picked: (kv + mamba + draft) / kv at the weg2sb5g W9
        # census, with the byte weights measured on /spinning/hicache-l3.
        kv = launcher.STORE_CENSUS_KV_PAGES * launcher.STORE_CENSUS_KV_PAGE_BYTES
        mamba = launcher.STORE_CENSUS_MAMBA_BLOBS * launcher.STORE_MAMBA_BLOB_BYTES
        draft = launcher.STORE_CENSUS_DRAFT_PAGES * host_ledger.DRAFT_PAGE_BYTES
        self.assertAlmostEqual(
            launcher.STORE_SIDECAR_FACTOR, (kv + mamba + draft) / kv, delta=1e-9
        )
        # The census's own arithmetic explains sb5g's refusals without any
        # further hypothesis: its content was already at the 5 GiB wall.
        self.assertAlmostEqual((kv + mamba + draft) / GIB, 4.66, delta=0.05)
        self.assertGreater(SB5G_MIN_FREE_REFUSALS, 0)

    def test_the_factor_is_a_flag_not_a_law(self):
        small = _plan("/tmp", sidecar_factor=1.0)
        big = _plan("/tmp", sidecar_factor=4.0)
        self.assertEqual(small.max_size_bytes, small.p_pool_bytes)
        self.assertGreater(big.max_size_bytes, small.max_size_bytes)


# =====================================================================
# 2. min_free IS AGAINST THE DISK, AND THE DISK MUST FUND THE WHOLE THING
# =====================================================================


class TestTheDiskCheck(CustomTestCase):
    def test_min_free_is_the_disk_floor_not_the_old_one_gib(self):
        plan = _plan("/tmp")
        self.assertEqual(
            plan.min_free_bytes, int(launcher.STORE_DISK_MIN_FREE_GIB * GIB)
        )
        # The old value is what latched HiCacheFile's write stop on a 6 GiB
        # tmpfs. It must not be what a DISK floor is set to.
        self.assertGreater(plan.min_free_bytes, 1 * GIB)

    def test_the_check_reads_the_filesystem_the_store_lives_on(self):
        # MUTANT-SHAPED: the whole point is WHICH filesystem is consulted. The
        # seam must statvfs the store root it was given, not a fixed path.
        seen = []

        def _spy(path):
            seen.append(path)
            return _FakeStatvfs(BIG_DISK_FREE, BIG_DISK_TOTAL)

        with mock.patch.object(os, "statvfs", _spy), \
                mock.patch.object(os, "makedirs"):
            launcher.plan_store(
                "t", 1000, (16,), KV_MIB_PER_TOKEN_PER_ATTN_LAYER,
                root="/some/other/root",
            )
        self.assertEqual(seen, ["/some/other/root"])

    def test_a_disk_that_cannot_fund_it_is_W52_by_name(self):
        with self.assertRaises(launcher.Weg2StoreDiskRefused) as cm:
            _plan("/tmp", free=SMALL_DISK_FREE)
        msg = str(cm.exception)
        self.assertIn("W52 Weg2StoreDiskRefused", msg)
        for needle in ("NEEDED", "AVAILABLE", "max_size", "min_free",
                       "P pool", "sidecar factor",
                       "--store-sidecar-factor", "--store-disk-min-free-gib"):
            self.assertIn(needle, msg)
        # and it says what is NOT a lever, so nobody goes looking for RAM.
        self.assertIn("NOT a lever: host RAM", msg)

    def test_needed_is_max_size_plus_min_free_not_either_alone(self):
        plan = _plan("/tmp")
        self.assertEqual(
            plan.needed_bytes, plan.max_size_bytes + plan.min_free_bytes
        )
        # a disk that fits max_size but not max_size + min_free must refuse
        with self.assertRaises(launcher.Weg2StoreDiskRefused):
            _plan("/tmp", free=plan.max_size_bytes + plan.min_free_bytes // 2)

    def test_the_extra_config_is_bytes_and_shared_scope(self):
        cfg = json.loads(_plan("/tmp").extra_config())
        self.assertEqual(cfg["max_size_scope"], "shared")
        self.assertEqual(int(cfg["max_size"]), _plan("/tmp").max_size_bytes)
        self.assertEqual(int(cfg["min_free_space"]), _plan("/tmp").min_free_bytes)
        # BYTES, not "5G": the old form rounded a GiB float to a whole G and
        # lost up to a GiB of a 6 GiB store on the way into the backend.
        self.assertNotIn("G", cfg["max_size"])


# =====================================================================
# 3. THE HOST LEDGER NO LONGER CHARGES A STORE
# =====================================================================


class TestTheLedgerDoesNotChargeRam(CustomTestCase):
    def test_the_run_peak_takes_no_store_argument_at_all(self):
        # THE MUTANT THIS PINS: re-adding `+ store_gib` to the run peak. It
        # cannot even be expressed -- the parameter is gone -- and a caller
        # that tries fails loudly instead of quietly re-charging RAM.
        params = [
            n for n in inspect.signature(
                host_ledger.Arm.predicted_run_peak_gib
            ).parameters if n != "self"
        ]
        self.assertEqual(params, [])

    def test_choose_takes_no_store_floor_and_returns_the_headroom(self):
        params = inspect.signature(host_ledger.choose).parameters
        self.assertNotIn("store_min_gib", params)
        self.assertFalse(hasattr(host_ledger, "size_store_gib"))
        self.assertFalse(hasattr(host_ledger, "StoreSizing"))

    def test_the_arm_and_chosen_lines_name_the_disk_and_never_a_ram_post(self):
        arm, headroom, lines = host_ledger.choose(
            int(118.05 * GIB), int(103.56 * GIB),
            ring_bytes=32964 * 1024 * 1024,
            ring_span1_bytes=29912 * 1024 * 1024,
            ring_provenance="test",
            cg_current_bytes=int(21.16 * GIB),
            cg_ceiling_bytes=int(118.05 * GIB),
            cg_ceiling_source="test",
            cg_oom_kill=0,
        )
        chosen = [ln for ln in lines if "WEG2-HOST-LEDGER CHOSEN" in ln][0]
        self.assertIn("store=ON DISK, NOT A RAM POST", chosen)
        self.assertIn("page_cache=reclaimable", chosen)
        self.assertIn("reap_headroom=", chosen)
        self.assertIsNotNone(headroom)
        # THE DANGER DIRECTION, pinned as text: no ARM or CHOSEN line may
        # print a store as a GiB RAM quantity again. Scoped to those lines on
        # purpose -- the surrounding prose legitimately REFERS to the tmpfs
        # form as history ("6 GiB on boot weg2sb5g, whose store was a 6 GiB
        # tmpfs"), and a blanket ban on the word would forbid the explanation
        # instead of the charge.
        priced = [ln for ln in lines
                  if "WEG2-HOST-LEDGER ARM" in ln or "WEG2-HOST-LEDGER CHOSEN" in ln]
        self.assertEqual(len(priced), 4)      # three arms plus the choice
        for ln in priced:
            self.assertNotIn("GiB tmpfs", ln)
        for ln in lines:
            if "WEG2-HOST-LEDGER ARM" in ln:
                self.assertIn("store=NOT CHARGED HERE", ln)

    def test_the_dormant_sample_no_longer_subtracts_the_store(self):
        params = inspect.signature(host_ledger.dormant_image_sample).parameters
        self.assertNotIn("store_used_bytes", params)
        # and the residual is derivable WITHOUT one, which the old gate refused
        rec = host_ledger.dormant_image_sample(
            group="P", shmem_before_bytes=0, shmem_after_bytes=0, pids=(),
            weight_tags_gib=28.83, interleaved=False, boot_tag="t", commit="c",
            cg_current_bytes=int(90.10 * GIB), reclaimable_bytes=0,
            arm={"s_gb": 1, "m_mib": 1200},
        )
        self.assertIsNotNone(rec["run_residual_gib"])
        self.assertNotIn("store_used_bytes", rec)

    def test_the_front_no_longer_reads_a_store_used_size(self):
        from sglang.srt.weg2 import front

        # statvfs of a directory on a SHARED dataset answers with the root
        # filesystem's usage, which is not a store. The reading is deleted.
        self.assertFalse(hasattr(front.Front, "_store_used_bytes"))


# =====================================================================
# 4. THE ZFS ARC PREFLIGHT
# =====================================================================


class TestTheArcPreflight(CustomTestCase):
    def _plan(self):
        return _plan("/tmp")

    def test_a_capped_arc_prints_and_never_refuses(self):
        arc = {"arc_max": 5 * GIB, "c_max": 5 * GIB, "size": int(4.92 * GIB)}
        line = launcher.arc_preflight_line(arc, self._plan(), margin_gib=0.5)
        self.assertIn("WEG2-STORE ARC:", line)
        self.assertIn("zfs_arc_max=", line)
        self.assertIn("VERDICT: capped, no refusal.", line)

    def test_an_uncapped_arc_with_a_thin_margin_is_W53_by_name(self):
        arc = {"arc_max": 0, "c_max": None, "size": int(40 * GIB)}
        with self.assertRaises(launcher.Weg2StoreArcRefused) as cm:
            launcher.arc_preflight_line(arc, self._plan(), margin_gib=0.5)
        self.assertIn("W53 Weg2StoreArcRefused", str(cm.exception))
        self.assertIn("NO explicit cap", str(cm.exception))

    def test_an_uncapped_arc_with_room_is_stated_not_refused(self):
        arc = {"arc_max": 0, "c_max": None, "size": 0}
        line = launcher.arc_preflight_line(arc, self._plan(), margin_gib=10_000.0)
        self.assertIn("uncapped", line)
        self.assertIn("stated, not refused", line)

    def test_an_unreadable_arc_never_prints_as_zero(self):
        arc = {"arc_max": None, "c_max": None, "size": None}
        line = launcher.arc_preflight_line(arc, self._plan(), margin_gib=None)
        self.assertIn("unreadable", line)
        self.assertNotIn("0.00 GiB", line)

    def test_the_reader_returns_none_for_missing_files_never_zero(self):
        arc = launcher.read_arc_state(
            param_path="/nonexistent/zfs_arc_max",
            kstat_path="/nonexistent/arcstats",
        )
        self.assertIsNone(arc["arc_max"])
        self.assertIsNone(arc["c_max"])
        self.assertIsNone(arc["size"])

    def test_the_reader_parses_the_real_kstat_shape(self):
        import tempfile

        with tempfile.TemporaryDirectory() as tmp:
            par = os.path.join(tmp, "p")
            kst = os.path.join(tmp, "k")
            with open(par, "w") as f:
                f.write("5368709120\n")
            with open(kst, "w") as f:
                f.write("name                            type data\n")
                f.write("c_min                           4    1073741824\n")
                f.write("c_max                           4    5368709120\n")
                f.write("size                            4    5288088320\n")
            arc = launcher.read_arc_state(param_path=par, kstat_path=kst)
        self.assertEqual(arc["arc_max"], 5368709120)
        self.assertEqual(arc["c_max"], 5368709120)
        self.assertEqual(arc["size"], 5288088320)


# =====================================================================
# 5. THE READ-SIDE COST LINE
# =====================================================================


class TestTheReadCostLine(CustomTestCase):
    def test_the_priced_read_is_covered_by_the_deferral_bound(self):
        plan = _plan("/tmp")
        line = launcher.store_read_cost_line(plan, cap_tokens=262144)
        self.assertIn("WEG2-STORE READ COST:", line)
        self.assertIn("262144 tokens", line)
        # the two figures that must be there, and the bound they are checked
        # against (#1238 fix 7).
        self.assertIn("_deferred_prefetch_bound_s", line)
        self.assertIn("P-POOL-SIZED PREFIX", line)
        # arithmetic, not prose: 262144 x 32768 B at 2754.9 MB/s ~= 3.12 s,
        # against base 2.0 + 262144/1024 = 258.0 s.
        cap_s = 262144 * plan.cell_bytes / (launcher.STORE_DISK_READ_MBPS * 1e6)
        self.assertAlmostEqual(cap_s, 3.12, delta=0.05)
        self.assertLess(cap_s, 2.0 + 262144 / 1024.0)
        self.assertLess(cap_s, 60.0)      # even the configured max clip

    def test_only_the_direct_figures_are_cited(self):
        line = launcher.store_read_cost_line(_plan("/tmp"), cap_tokens=262144)
        self.assertIn("2754.9", line)
        self.assertIn("1346.9", line)
        # the buffered write is an ARC artefact and must be named as excluded,
        # never used as the price.
        self.assertIn("3016.5 MB/s buffered write", line)
        self.assertIn("NOT cited", line)


# =====================================================================
# 6. THE DIRECTORY: PLAIN, FRESH PER BOOT, NO MOUNT ANYWHERE
# =====================================================================


class TestTheDirectory(CustomTestCase):
    def test_no_tmpfs_verb_survives_in_the_launcher(self):
        # THE MUTANT: keeping the RAM form behind a flag. There is no flag --
        # the mount/umount pair is gone from the source. Checked structurally
        # (AST) rather than by import, because `main` cannot be called here.
        import ast

        with open(launcher.__file__) as f:
            tree = ast.parse(f.read())
        literals = {
            n.value for n in ast.walk(tree)
            if isinstance(n, ast.Constant) and isinstance(n.value, str)
        }
        self.assertNotIn("umount", literals)
        self.assertIn("/spinning/hicache-weg2", literals)
        self.assertFalse(hasattr(launcher, "mount_store"))
        self.assertFalse(hasattr(launcher, "STORE_MOUNT"))

    def test_the_directory_is_per_boot_under_the_root(self):
        plan = _plan("/tmp")
        self.assertTrue(plan.directory.startswith("/tmp/"))
        self.assertTrue(plan.directory.endswith("/testboot"))

    def test_a_previous_boots_store_is_swept_and_its_bytes_reported(self):
        import tempfile

        with tempfile.TemporaryDirectory() as root:
            keep = os.path.join(root, "thisboot")
            stale = os.path.join(root, "oldboot")
            os.makedirs(keep)
            os.makedirs(stale)
            with open(os.path.join(stale, "page.bin"), "wb") as f:
                f.write(b"x" * 8192)
            log = _Log()
            freed = launcher.sweep_store_residue(log, keep, dry=False, root=root)
            # INSIDE the TemporaryDirectory: outside it the tree is gone and
            # `isdir(stale)` is False for a reason that has nothing to do with
            # the sweep -- the assertion would pass whatever the sweep did.
            self.assertGreater(freed, 0)
            self.assertFalse(os.path.isdir(stale))
            self.assertTrue(os.path.isdir(keep))
        self.assertIn("oldboot", log.text)
        self.assertIn("DISK, not host RAM", log.text)

    def test_the_sweep_keeps_this_boots_directory(self):
        import tempfile

        with tempfile.TemporaryDirectory() as root:
            keep = os.path.join(root, "thisboot")
            os.makedirs(keep)
            log = _Log()
            launcher.sweep_store_residue(log, keep, dry=False, root=root)
            self.assertTrue(os.path.isdir(keep))
        self.assertIn("none under", log.text)

    def test_a_dry_run_frees_nothing_and_says_so(self):
        import tempfile

        with tempfile.TemporaryDirectory() as root:
            stale = os.path.join(root, "oldboot")
            os.makedirs(stale)
            with open(os.path.join(stale, "page.bin"), "wb") as f:
                f.write(b"x" * 8192)
            log = _Log()
            freed = launcher.sweep_store_residue(
                log, os.path.join(root, "thisboot"), dry=True, root=root
            )
            # INSIDE the TemporaryDirectory, for the same reason as above.
            self.assertEqual(freed, 0)
            self.assertTrue(os.path.isdir(stale))
        self.assertIn("DRY-RUN", log.text)

    def test_prepare_store_makes_a_plain_directory(self):
        import tempfile

        with tempfile.TemporaryDirectory() as root:
            plan = _plan(root)
            got = launcher.prepare_store(_Log(), plan, dry=False)
            self.assertEqual(got, plan.directory)
            self.assertTrue(os.path.isdir(got))
            # and it is a plain directory, not a mount point
            with open("/proc/mounts") as f:
                self.assertNotIn(f" {got} ", f.read())


# =====================================================================
# 7. THE ARGV CARRIES THE CONFIG, NOT A GiB NUMBER
# =====================================================================


class TestTheArgvSeam(CustomTestCase):
    def test_common_flags_ships_the_rendered_config_verbatim(self):
        cfg = launcher.store_extra_config(30 * GIB, 32 * GIB)
        flags = launcher.common_flags("m", 1, 600, cfg, 262144)
        i = flags.index("--hicache-storage-backend-extra-config")
        self.assertEqual(flags[i + 1], cfg)
        # THE MUTANT: a float slipping back into this position would render as
        # a repr, not JSON. The value must parse as the backend's own config.
        parsed = json.loads(flags[i + 1])
        self.assertEqual(set(parsed), {"max_size", "min_free_space", "max_size_scope"})

    def test_the_signature_no_longer_accepts_a_store_size(self):
        for fn in (launcher.common_flags, launcher.argv_p, launcher.argv_d):
            self.assertNotIn("store_gib", inspect.signature(fn).parameters, fn.__name__)
            self.assertIn("store_cfg", inspect.signature(fn).parameters, fn.__name__)


# =====================================================================
# 8. THE W-CODES ARE FREE
# =====================================================================


class TestTheWCodes(CustomTestCase):
    def test_w52_and_w53_are_not_already_taken(self):
        import re
        import subprocess

        root = os.path.dirname(os.path.dirname(os.path.dirname(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))
        out = subprocess.run(
            ["grep", "-rhoE", r"\bW5[23] Weg2[A-Za-z0-9_]+",
             os.path.join(root, "python", "sglang", "srt", "weg2")],
            capture_output=True, text=True,
        ).stdout
        names = {}
        for line in out.split("\n"):
            m = re.match(r"(W5[23]) (Weg2\w+)", line.strip())
            if m:
                names.setdefault(m.group(1), set()).add(m.group(2))
        self.assertEqual(names.get("W52"), {"Weg2StoreDiskRefused"})
        self.assertEqual(names.get("W53"), {"Weg2StoreArcRefused"})


if __name__ == "__main__":
    unittest.main()
