"""#1386 (BOOT7 xsn31/4-6 finding): the ZIELFRAGE MINIMALFORM (two manual
flips + shadow compare, zero requests served) pays 8.14 GiB of the
57.378-GiB host-shmem baseline for a HiCache/Mamba-anchor host tier that a
0-request boot cannot use -- HiCache exists to reuse KV pages ACROSS
requests, and the Minimalform serves none (``served: {"P": 0, "D": 0}`` on
every attempt measured, xsn31/4/5/6).

``--weg2-disable-hicache`` is the switch, and the whole point of this file
is the ONE-SWITCH-ONE-TRUTH rule the order stated verbatim: the exact
"priced one number, allocated another" shape that cost #1385/#1386 three
boots (a lane cap priced at 6.75 GiB while the allocator ran 33.81, twice)
would be repeated here if the ledger term and the argv gate ever read two
different bools. So every test below either (a) proves the two sides agree
for the SAME input, or (b) is one half of the pair a manual mutant run
against (documented at the bottom, not re-run automatically: reverting
either gate alone was confirmed to fail the paired test, then restored).

Read the docstring before adding a sixth call site that threads this bool:
``launcher.py`` `common_flags` -> `argv_p`/`argv_d` and `host_ledger.py`
`charge_terms` -> `price` -> `choose` are the two chains; `launcher.main`
resolves `ns.weg2_disable_hicache` into ONE local exactly once (beside
`draft_kv_on_p`) and hands it, unread again, to both.
"""

from __future__ import annotations

import glob
import io
import json
import os
import pathlib
import tempfile
import unittest
from contextlib import redirect_stdout
from unittest import mock

import pytest

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

try:
    from sglang.srt.registry import nvml as nvml_registry
    from sglang.srt.weg2 import host_ledger, launcher
except Exception as exc:  # pragma: no cover - no weg2 launcher in this build
    pytest.skip(f"weg2 launcher unavailable: {exc}", allow_module_level=True)

GIB = host_ledger.GIB
GB = host_ledger.GB

MODEL = "/spinning/llm_stuff/club-3090/models-cache/Qwen3.8-27B-INT8-gdncov-vocabembed"
BUDGETS = [28904, 17704, 17672]

# The same #1378 fixture test_weg2_dry_run_order_probe_1378.py uses, for the
# same reason: this is the ONE place in this file that drives `launcher.main`
# itself, at the CLI/argv boundary, rather than the arithmetic layer directly
# -- because the falls-with-it grouping this class tests (P_DRAFT_KV_FLAGS
# leaving as a whole when `hicache_disabled`) is `main`-local logic, not a
# `common_flags`/`argv_p` parameter.
_FIXTURES = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                         "fixtures", "xchg_launch_replay_0911")
_RING_EVIDENCE = os.path.join(_FIXTURES, "ring_evidence")
_CENSUS = os.path.join(_FIXTURES, "census_weg2sn5b_48f55fb393.json")
_NVML_REPLAY_JSON = os.path.join(_FIXTURES, "nvml_devices_1378.json")
_RING_TABLE_BOOT_STEM_SUBSTR = "d8ea6261f7"
_TREE_ROOT = str(pathlib.Path(launcher.__file__).resolve().parents[4])

with open(os.path.join(_FIXTURES, "recorded.json"), encoding="utf-8") as _fh:
    _RECORDED = json.load(_fh)

_MODEL_PATH = _RECORDED["checkpoint_dependency"]["model_path"]


def _checkpoint_present() -> bool:
    return bool(glob.glob(os.path.join(_MODEL_PATH, "*.safetensors")))


_NEEDS_CHECKPOINT = unittest.skipUnless(
    _checkpoint_present(),
    "the checkpoint the fixture's ring evidence names is not on this box",
)


def _main_argv(tag: str, *extra: str) -> list:
    return [
        "--tree", _TREE_ROOT,
        "--tag", tag,
        "--dry-run",
        "--model", _MODEL_PATH,
        "--weg2-weight-source", "exchange",
        "--weg2-xchg-census", _CENSUS,
        "--evidence-dir", _RING_EVIDENCE,
        "--ring-table-boot", _RING_TABLE_BOOT_STEM_SUBSTR,
        # #1451: THE BOOT FORM since the arena entered the ledger (#1432): the
        # host ring is off and the exchange injects authoritatively (every arm
        # since weg2xsn2xx).  Without it the launcher's `auto` armed the
        # fixture's ring-era evidence (host weights 42.96 GiB) beside the
        # 33.6 GiB arena and no rung was fundable on the pinned quiet box.
        "--weg2-weights-cpu-backup", "off",
        "--weg2-xchg-inject", "authoritative",
        *extra,
    ]


# #1390: a QUIET reading of THIS box, 2026-09-14T02:2xZ, via `cat
# /proc/meminfo` / `cat /sys/fs/cgroup/memory.{current,max,stat,events}` --
# the SAME numbers `TheDraftKvProducerFallsWithIt`'s five tests were already
# reading live off the real box whenever no other boot happened to be
# running, pinned once here instead of re-read every test run. GENAU DIESE
# GROESSE, not an arbitrarily larger one: an over-generous fixture would
# fund an arm the real box's own margins might not, silently widening what
# these tests cover instead of preserving it.
_QUIET_MEMTOTAL_KB = 123_781_120       # 118.05 GiB
_QUIET_MEMAVAIL_KB = 118_833_964       # 113.32 GiB
_QUIET_CG_CURRENT_B = 14_064_181_248   # 13.10 GiB
_QUIET_CG_ANON_B = 5_015_404_544
_QUIET_CG_FILE_B = 8_809_852_928
_QUIET_CG_SHMEM_B = 577_536
_QUIET_CG_UNEVICTABLE_B = 36_864
_QUIET_CG_SLAB_RECLAIM_B = 188_900_232


def _fake_quiet_host(tmp: str) -> tuple:
    """Write the pinned quiet-box reading above as real files; return
    (meminfo_path, cgroup_root). Verified empirically (2026-09-14) that
    WITHOUT this fixture these five tests read the REAL box and refuse
    (W20 Weg2HostLedgerRefused) under a simulated busy box (MemAvailable
    forced to 5 GiB, cgroup current to 100 GiB) -- a live dependency
    entirely separate from the #1217 SHM/pgrep guard `SHM_DIR` isolates.
    """
    meminfo = os.path.join(tmp, "meminfo")
    with open(meminfo, "w") as f:
        f.write(
            f"MemTotal:       {_QUIET_MEMTOTAL_KB} kB\n"
            f"MemFree:        110046680 kB\n"
            f"MemAvailable:   {_QUIET_MEMAVAIL_KB} kB\n"
            f"Shmem:          {_QUIET_CG_SHMEM_B // 1024} kB\n"
            f"SwapTotal:      0 kB\n"
        )
    cg = os.path.join(tmp, "cgroup")
    os.makedirs(cg, exist_ok=True)
    with open(os.path.join(cg, "memory.current"), "w") as f:
        f.write(f"{_QUIET_CG_CURRENT_B}\n")
    with open(os.path.join(cg, "memory.peak"), "w") as f:
        f.write(f"{_QUIET_CG_CURRENT_B}\n")
    with open(os.path.join(cg, "memory.max"), "w") as f:
        f.write("max\n")
    with open(os.path.join(cg, "memory.events"), "w") as f:
        f.write("low 0\nhigh 0\nmax 0\noom 0\noom_kill 0\n")
    with open(os.path.join(cg, "memory.stat"), "w") as f:
        f.write(
            f"anon {_QUIET_CG_ANON_B}\n"
            f"file {_QUIET_CG_FILE_B}\n"
            f"shmem {_QUIET_CG_SHMEM_B}\n"
            f"unevictable {_QUIET_CG_UNEVICTABLE_B}\n"
            f"slab_reclaimable {_QUIET_CG_SLAB_RECLAIM_B}\n"
        )
    return meminfo, cg


class _HermeticMainBase(unittest.TestCase):
    """Same isolation as test_weg2_dry_run_order_probe_1378.py's base class:
    replayed NVML cards, a private guaranteed-empty SHM_DIR -- PLUS (#1390)
    a pinned quiet-box /proc/meminfo + cgroup2 tree, reusing the exact
    fake-host construction test_weg2_cushion_headroom_autoresolve_1378_stage2b.py
    built for `choose_host_ledger`'s own `meminfo_path`/`cgroup_root` seam,
    rather than a second fixture path for the same two files."""

    def setUp(self):
        super().setUp()
        self._old_replay_env = os.environ.get(nvml_registry.ENV_NVML_REPLAY)
        os.environ[nvml_registry.ENV_NVML_REPLAY] = _NVML_REPLAY_JSON
        self._shm_tmp = tempfile.mkdtemp(prefix="weg2-1386-empty-shm-")
        self._shm_patch = mock.patch.object(launcher, "SHM_DIR", self._shm_tmp)
        self._shm_patch.start()
        self._host_tmp = tempfile.mkdtemp(prefix="weg2-1386-quiet-host-")
        meminfo, cg = _fake_quiet_host(self._host_tmp)
        self._meminfo_patch = mock.patch.object(launcher, "MEMINFO_PATH", meminfo)
        self._cgroup_patch = mock.patch.object(launcher, "CGROUP_ROOT", cg)
        self._meminfo_patch.start()
        self._cgroup_patch.start()

    def tearDown(self):
        self._cgroup_patch.stop()
        self._meminfo_patch.stop()
        self._shm_patch.stop()
        if self._old_replay_env is None:
            os.environ.pop(nvml_registry.ENV_NVML_REPLAY, None)
        else:
            os.environ[nvml_registry.ENV_NVML_REPLAY] = self._old_replay_env
        super().tearDown()

    def run_main(self, argv):
        buf = io.StringIO()
        rc = None
        exc = None
        try:
            with redirect_stdout(buf):
                rc = launcher.main(argv)
        except Exception as e:  # the refusal path raises rather than returning
            exc = e
        return rc, buf.getvalue(), exc


def _flag_value(argv, flag):
    return argv[argv.index(flag) + 1] if flag in argv else None


def _images():
    return host_ledger.ImageTerms(
        p_gib=1.0, d_gib=1.0, p_source="test", d_source="test",
        p_measured=True, d_measured=True, extra_p_gib=0.0, extra_d_gib=0.0,
    )


HICACHE_FLAGS = (
    "--enable-hierarchical-cache",
    "--hicache-host-role",
    "--hicache-size",
    "--hicache-mamba-host-mib",
    "--hicache-write-policy",
    "--hicache-storage-backend",
    "--hicache-mem-layout",
    "--hicache-io-backend",
    "--hicache-storage-backend-extra-config",
    "--hicache-canonical-kv-page",
)


class TheLedgerHalf(unittest.TestCase):
    """`host_ledger.charge_terms` -> `price` -> `choose`, all three rungs."""

    def test_default_is_byte_identical_to_every_pre_1386_caller(self):
        # Byte-identical: an existing caller that never heard of the switch
        # must get the exact numbers it always got.
        implicit = host_ledger.charge_terms(1, 1200, 3, _images())
        explicit = host_ledger.charge_terms(1, 1200, 3, _images(), hicache_disabled=False)
        self.assertEqual(implicit, explicit)
        self.assertGreater(implicit["anchors_gib"], 0.0)
        self.assertGreater(implicit["rings_gib"], 0.0)
        self.assertGreater(implicit["overhead_gib"], 0.0)
        self.assertFalse(implicit["hicache_disabled"])

    def test_hicache_disabled_zeroes_exactly_anchors_rings_overhead(self):
        # OFF MEANS OFF: not a smaller S/M routed through the same formula --
        # charge_terms(s_gb=0, ...) would still be a positive number for any
        # s_gb_d that defaults from it, and a caller passing 0 M would still
        # walk the anchors formula to 0.0 by division, not by a named switch.
        # The distinction matters because a future edit that changes what
        # "0" means upstream (e.g. `s_gb_d` no longer defaulting from `s_gb`)
        # would silently change what an `s_gb=0` trick prices; this bool
        # cannot be affected by that, because it never touches s_gb/m_mib.
        on = host_ledger.charge_terms(1, 1200, 3, _images(), hicache_disabled=True)
        off = host_ledger.charge_terms(1, 1200, 3, _images(), hicache_disabled=False)
        self.assertEqual(on["anchors_gib"], 0.0)
        self.assertEqual(on["rings_gib"], 0.0)
        self.assertEqual(on["overhead_gib"], 0.0)
        self.assertTrue(on["hicache_disabled"])
        # heaps/draft/image terms are UNTOUCHED -- this switch owns three
        # keys and no others.
        for key in ("heaps_gib", "draft_host_p_gib", "draft_host_d_gib",
                    "image_p_gib", "image_d_gib", "s_gb_d"):
            self.assertEqual(on[key], off[key], key)

    def test_the_s_gb_0_trick_is_a_DIFFERENT_and_smaller_number(self):
        # Pins the order's own "kein S=0-Trick" line: routing 0 through the
        # EXISTING formula and using the NEW switch must not collide on the
        # same result, or a future reader could mistake one for the other.
        zero_s = host_ledger.charge_terms(0, 0, 3, _images())
        switched_off = host_ledger.charge_terms(1, 1200, 3, _images(), hicache_disabled=True)
        self.assertEqual(switched_off["anchors_gib"], 0.0)
        self.assertEqual(switched_off["rings_gib"], 0.0)
        # anchors_gib at m_mib=0 is ALSO 0.0 (0/2400 * const), but rings_gib
        # at s_gb=0 depends on s_gb_d's own default -- s_gb_d defaults to
        # s_gb, so rings_gib is 0 too. Both land on 0.0 here BY COINCIDENCE
        # of the inputs chosen, which is exactly why the switch, not the
        # trick, is the real fix: the trick's 0 is a property of s_gb/m_mib
        # this call happened to pick, not a declared intent, and any caller
        # that still needs `s_gb`/`m_mib` for something else (they do: the
        # ARM line prints S=/M=, and a 0 there reads as a broken boot, not a
        # disabled cache).
        self.assertEqual(zero_s["anchors_gib"], 0.0)
        self.assertEqual(zero_s["rings_gib"], 0.0)

    def test_price_forwards_the_switch_to_charge_terms(self):
        arm_on = host_ledger.price(
            int(200 * GIB), int(150 * GIB), 1, 1200,
            ring_bytes=int(40 * GIB), ring_span1_bytes=int(35 * GIB),
            hicache_disabled=True,
        )
        arm_off = host_ledger.price(
            int(200 * GIB), int(150 * GIB), 1, 1200,
            ring_bytes=int(40 * GIB), ring_span1_bytes=int(35 * GIB),
            hicache_disabled=False,
        )
        self.assertEqual(arm_on.terms["rings_gib"], 0.0)
        self.assertEqual(arm_on.terms["anchors_gib"], 0.0)
        self.assertTrue(arm_on.terms["hicache_disabled"])
        self.assertGreater(arm_off.terms["rings_gib"], 0.0)
        self.assertFalse(arm_off.terms["hicache_disabled"])
        # THE BOOT CHARGE ACTUALLY FALLS -- the whole point, stated as a
        # number rather than "the fields are zero and I trust that matters".
        # `_boot_charges_gib` is the same sum `predicted_run_peak_gib` adds
        # to the run origin (host_ledger.py:2426); comparing it directly
        # avoids needing a full cgroup reading just to get a run origin.
        saved = arm_off.terms["anchors_gib"] + arm_off.terms["rings_gib"] + arm_off.terms["overhead_gib"]
        self.assertGreater(saved, 0.0)
        self.assertAlmostEqual(
            host_ledger._boot_charges_gib(arm_off.terms)
            - host_ledger._boot_charges_gib(arm_on.terms),
            saved, places=6,
        )

    def test_choose_forwards_the_switch_to_every_rung_of_the_ladder(self):
        arm, _headroom, lines = host_ledger.choose(
            int(200 * GIB), int(150 * GIB),
            ring_bytes=int(40 * GIB), ring_span1_bytes=int(35 * GIB),
            hicache_disabled=True,
        )
        self.assertEqual(arm.terms["rings_gib"], 0.0)
        self.assertEqual(arm.terms["anchors_gib"], 0.0)
        joined = "\n".join(lines)
        self.assertIn("WEG2-HICACHE-DISABLED hicache_disabled=True", joined)

    def test_the_declared_line_always_prints_on_or_off(self):
        # NEVER SILENT: an absent line must not be confused with "was never
        # checked" -- the same rule the #1385 lanes-concurrent line follows.
        _arm, _h, lines_off = host_ledger.choose(
            int(200 * GIB), int(150 * GIB),
            ring_bytes=int(40 * GIB), ring_span1_bytes=int(35 * GIB),
        )
        self.assertIn("WEG2-HICACHE-DISABLED hicache_disabled=False", "\n".join(lines_off))


class TheArgvHalf(unittest.TestCase):
    """`launcher.common_flags` -> `argv_p`/`argv_d`."""

    def _common(self, **kw):
        return launcher.common_flags(
            MODEL, 1, 1200, "{}", 69632, "write_through", "both",
            **kw,
        )

    def test_default_argv_is_byte_identical_to_every_pre_1386_caller(self):
        implicit = self._common()
        explicit = self._common(hicache_disabled=False)
        self.assertEqual(implicit, explicit)
        for flag in HICACHE_FLAGS:
            self.assertIn(flag, implicit, flag)

    def test_disabled_omits_every_hicache_flag_and_nothing_else(self):
        on = self._common(hicache_disabled=True)
        off = self._common(hicache_disabled=False)
        for flag in HICACHE_FLAGS:
            self.assertNotIn(flag, on, flag)
            self.assertIn(flag, off, flag)
        # NOT A BLANKET CUT: --enable-cache-report/--enable-metrics are a
        # DIFFERENT feature (request-level cache reporting / prometheus),
        # never gated by this switch, and must still be there.
        self.assertIn("--enable-cache-report", on)
        self.assertIn("--enable-metrics", on)
        self.assertIn("--host", on)
        self.assertIn("--page-size", on)
        # Every non-hicache token count matches: the block removed is EXACTLY
        # 18 tokens (9 --flag/value pairs plus the one bare
        # --enable-hierarchical-cache), never a byte more or less.
        self.assertEqual(len(off) - len(on), 18)

    def test_argv_p_carries_the_switch_through_common_flags(self):
        argv_on = launcher.argv_p(
            "py", MODEL, BUDGETS, 1, 1200, "{}", [], hicache_disabled=True,
        )
        argv_off = launcher.argv_p(
            "py", MODEL, BUDGETS, 1, 1200, "{}", [], hicache_disabled=False,
        )
        for flag in HICACHE_FLAGS:
            self.assertNotIn(flag, argv_on, flag)
            self.assertIn(flag, argv_off, flag)

    def test_argv_d_carries_the_switch_through_common_flags(self):
        argv_on = launcher.argv_d(
            "py", MODEL, BUDGETS, 1, 1200, "{}", [], hicache_disabled=True,
        )
        argv_off = launcher.argv_d(
            "py", MODEL, BUDGETS, 1, 1200, "{}", [], hicache_disabled=False,
        )
        for flag in HICACHE_FLAGS:
            self.assertNotIn(flag, argv_on, flag)
            self.assertIn(flag, argv_off, flag)

    def test_argv_d_default_is_byte_identical(self):
        # argv_d has NO keyword-only marker (#1356's own lesson): the switch
        # was appended LAST so no existing positional caller shifts.
        implicit = launcher.argv_d("py", MODEL, BUDGETS, 1, 1200, "{}", [])
        explicit = launcher.argv_d(
            "py", MODEL, BUDGETS, 1, 1200, "{}", [], hicache_disabled=False,
        )
        self.assertEqual(implicit, explicit)


class TheCliFlag(unittest.TestCase):
    def test_defaults_to_off(self):
        ns = launcher.build_parser().parse_args(["--tree", "/t", "--tag", "x"])
        self.assertFalse(bool(getattr(ns, "weg2_disable_hicache", None)))

    def test_parses_when_given(self):
        ns = launcher.build_parser().parse_args(
            ["--tree", "/t", "--tag", "x", "--weg2-disable-hicache"]
        )
        self.assertTrue(ns.weg2_disable_hicache)


class OneSwitchOneTruth(unittest.TestCase):
    """The paired invariant: for the SAME bool, the ledger term and the argv
    gate must agree. This is the automatic half of the danger-direction
    guard; the MANUAL half (reverting one side alone and watching this class
    go red) is documented at the bottom of the file, not re-run here.
    """

    def test_ledger_and_argv_agree_for_both_booleans(self):
        for disabled in (True, False):
            with self.subTest(hicache_disabled=disabled):
                terms = host_ledger.charge_terms(1, 1200, 3, _images(),
                                                 hicache_disabled=disabled)
                argv = launcher.common_flags(
                    MODEL, 1, 1200, "{}", 69632, "write_through", "both",
                    hicache_disabled=disabled,
                )
                priced_zero = (terms["rings_gib"] == 0.0
                               and terms["anchors_gib"] == 0.0)
                argv_absent = "--enable-hierarchical-cache" not in argv
                self.assertEqual(
                    priced_zero, disabled,
                    "charge_terms did not honour hicache_disabled",
                )
                self.assertEqual(
                    argv_absent, disabled,
                    "common_flags did not honour hicache_disabled",
                )
                # THE DANGER DIRECTION, NAMED: priced_zero True while
                # argv_absent False (or the reverse) is exactly "priced one
                # number, allocated another" -- the #1385/#1386 shape. Both
                # must move together.
                self.assertEqual(priced_zero, argv_absent)


@_NEEDS_CHECKPOINT
class TheDraftKvProducerFallsWithIt(_HermeticMainBase):
    """xsn31/7 wall follow-up: ``--speculative-draft-kv-only`` belongs in the
    SAME falls-together group as the ten ``--hicache-*`` flags, because its
    canonical-page-store target (``--hicache-canonical-kv-page``) has no
    existence once HiCache is disabled -- ``server_args.py:8432`` crashed
    group P 9 seconds after launch on exactly this before this fix. MTP
    itself is untouched: group D keeps its own, independent NEXTN head in
    both forms (``ring_table.p_carries_drafter`` reads P's SHIPPED argv, not
    ``ns.draft_kv_on_p``, so this class exercises ``launcher.main`` at the
    CLI boundary rather than the arithmetic layer -- the falls-with-it
    grouping is ``main``-local logic).
    """

    def test_default_draft_kv_on_p_falls_silently_when_hicache_disabled(self):
        # No explicit --draft-kv-on-p: the operator stated no intent, so the
        # auto-off fallback applies with no refusal and no crash.
        rc, out, exc = self.run_main(_main_argv("t1386a", "--weg2-disable-hicache"))
        self.assertIsNone(exc, f"unexpected exception: {exc}")
        self.assertEqual(rc, 0)
        p_line = next(line for line in out.splitlines() if "WEG2-LAUNCH group P argv:" in line)
        self.assertNotIn("--speculative-draft-kv-only", p_line)
        self.assertNotIn("--speculative-algorithm", p_line)
        self.assertNotIn("--hicache-canonical-kv-page", p_line)
        # group D keeps its own, independent NEXTN head -- MTP itself is
        # untouched, only P's co-production into the (now nonexistent) store
        # drops.
        d_line = next(line for line in out.splitlines() if "WEG2-LAUNCH group D argv:" in line)
        self.assertIn("--speculative-algorithm", d_line)
        # the pre-existing #1264 off-form declaration fires -- this class
        # adds NO second, competing "off" mechanism, it reuses the one that
        # already exists for --draft-kv-on-p off. (The W10/W11-skip line
        # itself lives further down main(), past the point --dry-run returns
        # -- ring_table.p_carries_drafter(shipped_argv_p), which that skip
        # reads, is exercised by the argv assertions above instead.)
        self.assertIn("WEG2 DRAFT-KV-ON-P: off", out)

    def test_explicit_draft_kv_on_p_on_refuses_by_name(self):
        # An EXPLICIT --draft-kv-on-p on is a stated intent this boot cannot
        # honor -- silently overriding it would repeat the exact "priced one
        # number, allocated another" shape this whole switch exists to
        # prevent, so this is a named refusal, not a silent downgrade.
        rc, out, exc = self.run_main(_main_argv(
            "t1386b", "--weg2-disable-hicache", "--draft-kv-on-p", "on",
        ))
        self.assertIsNotNone(exc, "expected Weg2LaunchRefused, got none")
        self.assertIsInstance(exc, launcher.Weg2LaunchRefused)
        self.assertIn("W104 Weg2HicacheDraftKvProducerConflict", str(exc))

    def test_explicit_draft_kv_on_p_off_is_fine_no_refusal(self):
        # Explicit `off` agrees with the auto-fallback -- no conflict, no
        # refusal, same as the default case.  H25: the env names the same
        # fact (W127 refuses an explicit env that contradicts an explicit
        # flag), so this case states it the same way on both inputs.
        os.environ["SGLANG_WEG2_DRAFT_ON_P"] = "0"
        rc, out, exc = self.run_main(_main_argv(
            "t1386c", "--weg2-disable-hicache", "--draft-kv-on-p", "off",
        ))
        self.assertIsNone(exc, f"unexpected exception: {exc}")
        self.assertEqual(rc, 0)

    def test_hicache_enabled_default_leaves_the_producer_untouched(self):
        # Regression control: hicache ENABLED (the byte-identical default)
        # must not be touched by this fix at all.
        rc, out, exc = self.run_main(_main_argv("t1386d"))
        self.assertIsNone(exc, f"unexpected exception: {exc}")
        self.assertEqual(rc, 0)
        p_line = next(line for line in out.splitlines() if "WEG2-LAUNCH group P argv:" in line)
        self.assertIn("--speculative-draft-kv-only", p_line)
        self.assertIn("--hicache-canonical-kv-page", p_line)

    def test_hicache_enabled_explicit_draft_kv_on_p_on_no_refusal(self):
        # hicache enabled + explicit --draft-kv-on-p on: no conflict at all,
        # must never trip the new refusal (it is scoped to hicache_disabled).
        rc, out, exc = self.run_main(_main_argv(
            "t1386e", "--draft-kv-on-p", "on",
        ))
        self.assertIsNone(exc, f"unexpected exception: {exc}")
        self.assertEqual(rc, 0)


class TheFiveTestsAboveAreActuallyHermeticNow(_HermeticMainBase):
    """#1390: `_HermeticMainBase` LOOKED isolated (a replayed NVML card set,
    a private SHM_DIR) but two of its three seams were decorative.

    `shm_residue_sweep(shm_dir: str = SHM_DIR)` binds that default ONCE, at
    module-import time -- `mock.patch.object(launcher, "SHM_DIR", tmp)`
    changes the ATTRIBUTE, but `main`'s own call
    (`shm_residue_sweep(log, ns.tag, stamp, dry)`, no `shm_dir=` keyword)
    kept reading the literal `/dev/shm` the function was DEFINED against,
    regardless of the mock. `choose_host_ledger`'s `meminfo_path`/
    `cgroup_root` were never even threaded through `main`'s call at all --
    every dry run in this file read the REAL `/proc/meminfo` and the REAL
    cgroup tree. Verified empirically (2026-09-14) with `host_ledger.
    read_meminfo`/`read_cgroup` monkeypatched to a busy-box reading
    (MemAvailable forced to 5 GiB, cgroup current to 100 GiB): the SAME
    five tests' own model/arm combination refuses with W20
    Weg2HostLedgerRefused under that reading -- a live dependency entirely
    separate from, and in addition to, the SHM/pgrep one.

    Both are fixed at the ONE call site (`main`, #1390): `shm_dir=SHM_DIR`,
    `meminfo_path=MEMINFO_PATH`, `cgroup_root=CGROUP_ROOT`, forcing a fresh
    read of the (now genuinely mockable) module attribute at call time
    instead of the frozen default. The two tests below reproduce each half
    of the live dependency HERMETICALLY -- a self-held file descriptor for
    the SHM live-holder path (no subprocess, no real pgrep target needed:
    `shm_holder_pids` scans the real `/proc`, and this test process is
    itself a real, live holder of a file it just opened) and a
    deliberately too-small fake host for the memory path -- and each is
    paired with the manual mutant that proves the fix is load-bearing, not
    decorative like the seam it replaces.
    """

    def test_a_mocked_shm_residue_with_a_live_holder_is_actually_seen(self):
        """Proves `main` reads the MOCKED `SHM_DIR`, not the real one: a
        residue file this test itself holds open (a live holder, found via
        the real `/proc`, no subprocess needed) must refuse the boot."""
        marker = os.path.join(self._shm_tmp, "sglang-phase-flip-presence-1390test")
        fh = open(marker, "wb")
        try:
            fh.write(b"x" * 16)
            fh.flush()
            rc, out, exc = self.run_main(_main_argv("t1390shm"))
            self.assertIsNotNone(
                exc, "a live-held residue file in the MOCKED SHM_DIR must "
                "refuse the boot -- if it does not, main() is reading the "
                "real /dev/shm instead of the mock")
            self.assertIsInstance(exc, launcher.Weg2LaunchRefused)
            self.assertIn("LIVE HOLDER", str(exc))
        finally:
            fh.close()

    def test_a_too_small_mocked_host_actually_refuses_the_ladder(self):
        """Proves `main` reads the MOCKED `MEMINFO_PATH`/`CGROUP_ROOT`, not
        the real box: a deliberately too-small fake host must refuse to
        fund even the frugal end of this file's own arm ladder."""
        tiny = tempfile.mkdtemp(prefix="weg2-1390-tiny-host-")
        meminfo = os.path.join(tiny, "meminfo")
        with open(meminfo, "w") as f:
            f.write("MemTotal:       2097152 kB\nMemFree:        1048576 kB\n"
                     "MemAvailable:   1048576 kB\nShmem:          1024 kB\n"
                     "SwapTotal:      0 kB\n")
        cg = os.path.join(tiny, "cgroup")
        os.makedirs(cg, exist_ok=True)
        with open(os.path.join(cg, "memory.current"), "w") as f:
            f.write("1073741824\n")
        with open(os.path.join(cg, "memory.peak"), "w") as f:
            f.write("1073741824\n")
        with open(os.path.join(cg, "memory.max"), "w") as f:
            f.write("2147483648\n")
        with open(os.path.join(cg, "memory.events"), "w") as f:
            f.write("oom_kill 0\n")
        with open(os.path.join(cg, "memory.stat"), "w") as f:
            f.write("anon 536870912\nfile 0\nshmem 0\nunevictable 0\n"
                     "slab_reclaimable 0\n")
        with mock.patch.object(launcher, "MEMINFO_PATH", meminfo), \
             mock.patch.object(launcher, "CGROUP_ROOT", cg):
            rc, out, exc = self.run_main(_main_argv("t1390mem"))
        self.assertIsNotNone(
            exc, "a 2 GiB fake host must refuse this file's own arm ladder "
            "-- if it does not, main() is reading the real host instead of "
            "the mock")
        self.assertIsInstance(
            exc, (host_ledger.Weg2HostLedgerRefused,
                  host_ledger.Weg2HostRunPeakRefused))

    def test_main_names_both_seams_at_their_call_site(self):
        """Structural companion to the two behavioural tests above: the
        call site must NAME the module attribute, not rely on a bare call
        (see the class docstring for why a bare call is silently inert)."""
        import inspect

        src = inspect.getsource(launcher.main)
        self.assertIn("shm_dir=SHM_DIR", src)
        self.assertIn("meminfo_path=MEMINFO_PATH", src)
        self.assertIn("cgroup_root=CGROUP_ROOT", src)


if __name__ == "__main__":
    unittest.main()

# MANUALLY VERIFIED (danger direction, not re-run automatically -- restored
# after, diff empty):
#
# (a) `host_ledger.charge_terms`: reverted the `0.0 if hicache_disabled else
#     (...)` guards on `anchors_gib`/`rings_gib` back to the unconditional
#     formula (switch computed but never read). Result: every test in
#     `TheLedgerHalf` that asserts a 0.0 term failed immediately, and
#     `OneSwitchOneTruth.test_ledger_and_argv_agree_for_both_booleans` failed
#     on `hicache_disabled=True` with `priced_zero=False, argv_absent=True`
#     -- the exact "ledger prices a number, argv allocates a different
#     reality" shape named in the order. Restored; suite green again.
#
# (b) `launcher.common_flags`: reverted the `+ ([] if hicache_disabled else
#     [...]) +` gate to an unconditional list (switch computed but never
#     read). Result: every test in `TheArgvHalf` that asserts a flag's
#     absence failed immediately, and `OneSwitchOneTruth`'s same test failed
#     on `hicache_disabled=True` with `priced_zero=True, argv_absent=False`
#     -- the mirror direction: the ledger charges 0 while the ranks still
#     allocate the buffer, which is EXACTLY how #1385's lane cap under-priced
#     a real boot's allocator twice. Restored; suite green again.
#
# xsn31/7 wall follow-up (`TheDraftKvProducerFallsWithIt`), two more manual
# mutants, same discipline (revert, watch red, restore, diff empty):
#
# (c) `launcher.main`: reverted `draft_kv_on_p = _draft_kv_on_p_requested and
#     not hicache_disabled` back to `draft_kv_on_p = _draft_kv_on_p_requested`
#     (the switch computed but the producer flag not folded in). Result:
#     `test_default_draft_kv_on_p_falls_silently_when_hicache_disabled` failed
#     immediately with `'--speculative-draft-kv-only' unexpectedly found` in
#     group P's shipped argv -- reproducing the exact xsn31/7 crash condition
#     (server_args.py:8432) this class exists to prevent. Restored; suite
#     green again.
#
# (d) `launcher.main`: replaced the `if hicache_disabled and
#     _draft_kv_on_p_requested and (bs_source(...) == "flag"):` guard with
#     `if False:` (the named refusal disarmed). Result:
#     `test_explicit_draft_kv_on_p_on_refuses_by_name` failed with
#     `unexpectedly None : expected Weg2LaunchRefused, got none` -- an
#     explicit, contradicted `--draft-kv-on-p on` would have been silently
#     downgraded instead of refused, the exact swallow class this switch's
#     "EIN Schalter, EINE Wahrheit" doctrine exists to prevent. Restored;
#     suite green again.
