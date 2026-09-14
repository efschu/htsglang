# SPDX-License-Identifier: Apache-2.0
"""#1392 -- the dry run says WHICH box state it funded against.

`choose()` already prices whatever the box holds RIGHT NOW into every arm
(`cg_current_bytes` -> the origin -> the predicted peak), and a parallel
suite or boot TIGHTENS that bound -- the safe direction. What was missing is
the ARM line ever SAYING so: a verdict measured on a quiet box and one
measured under ~2.5 GiB of parallel load printed IDENTICALLY (the exact
incident the coordinator named, dry-run-misst-die-live-box, 2026-09-13: the
same serving argv worked to ARM 95.18 on a quiet box and refused W97 97.67
under a parallel pytest suite). The asymmetry that matters: foreign load
tightens the bound (a false refusal, attributable and safe), but the
dangerous case is the reverse -- measured quiet, funded, then booted later
once the box has picked up load the dry run never saw.

THE DECISION THIS FILE PINS: PROVISIONAL, not a second refusal gate.
`foreign_load_now_gib` crossing `FOREIGN_LOAD_PROVISIONAL_THRESHOLD_GIB`
never changes `ok`/which arm is chosen -- it only changes the TEXT. Reasons,
both already established in this codebase before this ticket:

1. A hard refusal keyed on instantaneous box noise would misattribute the
   box's transient state to the FORM -- exactly what the
   dry-run-misst-die-live-box memory note already warns against ("ein
   Refusal aus einem Dry-Run neben Fremdlast ist kein Befund ueber die
   Form, sondern ueber die Box").
2. Any GENUINE danger from current foreign load is already caught by the
   EXISTING checks (peak_ok / cushion_ok / headroom_ok), since foreign load
   raises cg_current -> raises the origin -> raises the predicted peak --
   the same asymmetric-safe coupling the coordinator named. A second gate
   answering the same underlying signal would be the #1358 double-
   bookkeeping shape one layer up (two places deciding the same fact, one
   day disagreeing).
3. PROVISIONAL preserves every fact for the reader without inventing new
   binding semantics -- the same "indicator, not Freibrief" shape
   `cushion_headroom_gib` (#1378 Stage 2) already established for exactly
   this kind of honest-but-non-binding evidence.

NO NEW MEASUREMENT APPARATUS: `cg_current_bytes`/`cg_anon_bytes`/
`cg_shmem_bytes` are read once already (`host_ledger.read_cgroup`); this
only prints them and compares the first against a named, dated baseline
(`QUIET_BASELINE_CG_CURRENT_GIB`, this box's own quiet reading, 2026-09-14).
"""

import os

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

import inspect
import sys
import tempfile
import unittest
from unittest import mock

from sglang.srt.registry import nvml as nvml_registry
from sglang.srt.weg2 import host_ledger as hl
from sglang.srt.weg2 import launcher
from sglang.test.test_utils import CustomTestCase

# Coordinator order 2026-09-14: prior-art gate first, reuse rather than a
# second hermetic harness for the identical seam. `_HermeticMainBase` (#1390)
# already isolates SHM_DIR/MEMINFO_PATH/CGROUP_ROOT and runs `launcher.main`
# for real; `_fake_quiet_host` is the pinned quiet-box reading. Both live in
# test_weg2_hicache_disabled_1386.py, one file over -- imported, not
# reimplemented, the same convention test_weg2_band_credit_1397.py already
# uses for a sibling test module's fixture (its own explicit sys.path.insert,
# needed because this directory carries an __init__.py and a bare top-level
# import would otherwise only resolve when some earlier-collected test
# happened to import the sibling module first).
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from test_weg2_hicache_disabled_1386 import (  # noqa: E402
    _HermeticMainBase,
    _NEEDS_CHECKPOINT,
    _NVML_REPLAY_JSON,
    _QUIET_CG_ANON_B,
    _QUIET_CG_FILE_B,
    _QUIET_CG_SHMEM_B,
    _QUIET_CG_SLAB_RECLAIM_B,
    _QUIET_CG_UNEVICTABLE_B,
    _QUIET_MEMAVAIL_KB,
    _QUIET_MEMTOTAL_KB,
    _main_argv,
)

GIB = hl.GIB

_KW = dict(
    arms=[(1, 150)],
    ring_bytes=int(12.0 * GIB), ring_span1_bytes=int(4.0 * GIB),
    cg_ceiling_bytes=int(123.78 * GIB), s_gb_d=4,
)
MEMTOTAL = int(123.78 * GIB)
MEMAVAIL = int(110.0 * GIB)


def _choose(**kw):
    return hl.choose(
        MEMTOTAL, MEMAVAIL,
        flip_ratchet=hl.resolve_flip_ratchet_gib(),
        **_KW, **kw,
    )


def _box_state_line(lines):
    hit = [l for l in lines if l.startswith("WEG2-DRY-RUN-BOX-STATE")]
    assert hit, "no box-state line printed"
    return hit[0]


class TheThreeNumbersAreAlwaysPrinted(CustomTestCase):
    """Constraint: on or off, funded or refused, the line exists and carries
    the raw reading -- never a summary, never omitted."""

    def test_memavail_anon_shmem_are_all_on_the_line_when_measured(self):
        arm, _, lines = _choose(
            cg_current_bytes=int(13.10 * GIB), reclaimable_bytes=int(3.0 * GIB),
            box_state_at="2026-09-14T03:00:00Z",
            cg_anon_bytes=int(5.0 * GIB), cg_shmem_bytes=int(0.5 * GIB),
        )
        self.assertIsNotNone(arm)
        line = _box_state_line(lines)
        self.assertIn("at=2026-09-14T03:00:00Z", line)
        self.assertIn("memavail=110.00 GiB", line)
        self.assertIn("anon=5.00 GiB", line)
        self.assertIn("shmem=0.50 GiB", line)

    def test_an_unmeasured_reading_prints_unreadable_never_a_blank_or_zero(self):
        arm, _, lines = _choose(
            cg_current_bytes=int(13.10 * GIB), reclaimable_bytes=int(3.0 * GIB),
        )
        line = _box_state_line(lines)
        self.assertIn("anon=unreadable", line)
        self.assertIn("shmem=unreadable", line)
        self.assertIn("at=not stamped by the caller", line)

    def test_the_line_prints_even_when_every_arm_refuses(self):
        with self.assertRaises(
                (hl.Weg2HostLedgerRefused, hl.Weg2HostRunPeakRefused)) as cm:
            _choose(
                cg_current_bytes=int(120.0 * GIB), reclaimable_bytes=0,
                box_state_at="2026-09-14T03:00:00Z",
                cg_anon_bytes=int(100.0 * GIB), cg_shmem_bytes=int(1.0 * GIB),
            )
        msg = str(cm.exception)
        self.assertIn("WEG2-DRY-RUN-BOX-STATE", msg)
        self.assertIn("anon=100.00 GiB", msg)


class ForeignLoadNowGibIsCorrect(CustomTestCase):
    def test_at_the_baseline_it_is_zero(self):
        self.assertEqual(
            hl.foreign_load_now_gib(int(hl.QUIET_BASELINE_CG_CURRENT_GIB * GIB)), 0.0)

    def test_below_the_baseline_it_floors_at_zero_not_negative(self):
        self.assertEqual(hl.foreign_load_now_gib(int(1.0 * GIB)), 0.0)

    def test_above_the_baseline_it_is_the_exact_delta(self):
        got = hl.foreign_load_now_gib(
            int((hl.QUIET_BASELINE_CG_CURRENT_GIB + 4.2) * GIB))
        self.assertAlmostEqual(got, 4.2, places=2)

    def test_unmeasured_is_none_not_zero(self):
        self.assertIsNone(hl.foreign_load_now_gib(None))


class ProvisionalNeverChangesTheVerdict(CustomTestCase):
    """THE DECISION: crossing the threshold changes the TEXT, never `ok`.
    Same specimen, only `cg_anon_bytes`/`box_state_at` differ (cosmetic to
    the ladder) -- both must reach the identical funded arm."""

    def test_a_fundable_arm_stays_fundable_under_heavy_foreign_load(self):
        quiet_arm, quiet_headroom, quiet_lines = _choose(
            cg_current_bytes=int(13.10 * GIB), reclaimable_bytes=int(3.0 * GIB),
        )
        noisy_arm, noisy_headroom, noisy_lines = _choose(
            cg_current_bytes=int(13.10 * GIB), reclaimable_bytes=int(3.0 * GIB),
            cg_anon_bytes=int(50.0 * GIB), cg_shmem_bytes=int(50.0 * GIB),
        )
        self.assertIsNotNone(quiet_arm)
        self.assertIsNotNone(noisy_arm)
        self.assertEqual(quiet_arm.s_gb, noisy_arm.s_gb)
        self.assertEqual(quiet_arm.m_mib, noisy_arm.m_mib)
        self.assertAlmostEqual(quiet_headroom, noisy_headroom, places=6)

    def test_the_provisional_text_fires_only_above_the_named_threshold(self):
        # cg_current itself (not anon/shmem, which are print-only here) is
        # what foreign_load_now_gib compares -- push it 3 GiB over baseline,
        # comfortably past the 2.5 GiB threshold.
        arm, _, lines = _choose(
            cg_current_bytes=int((hl.QUIET_BASELINE_CG_CURRENT_GIB + 3.0) * GIB),
            reclaimable_bytes=int(3.0 * GIB),
        )
        self.assertIsNotNone(arm, "a fundable specimen; PROVISIONAL is a "
                              "text change, not a refusal")
        line = _box_state_line(lines)
        self.assertIn("PROVISIONAL", line)
        self.assertIn(
            f"{hl.FOREIGN_LOAD_PROVISIONAL_THRESHOLD_GIB:.2f} GiB threshold",
            line)
        self.assertIn("dry-run-misst-die-live-box", line)
        self.assertIn("RE-VERIFIED", line)

    def test_no_provisional_text_at_or_below_the_threshold(self):
        arm, _, lines = _choose(
            cg_current_bytes=int(
                (hl.QUIET_BASELINE_CG_CURRENT_GIB
                 + hl.FOREIGN_LOAD_PROVISIONAL_THRESHOLD_GIB) * GIB),
            reclaimable_bytes=int(3.0 * GIB),
        )
        line = _box_state_line(lines)
        self.assertNotIn("PROVISIONAL", line)
        self.assertIn("not stale-suspect", line)

    def test_a_refused_arm_under_heavy_load_names_the_box_not_the_form(self):
        """The other honesty direction: a REFUSAL under high foreign load
        must say the box may be the cause, not just print the refusal."""
        with self.assertRaises(
                (hl.Weg2HostLedgerRefused, hl.Weg2HostRunPeakRefused)) as cm:
            _choose(
                cg_current_bytes=int(120.0 * GIB), reclaimable_bytes=0,
            )
        msg = str(cm.exception)
        self.assertIn("PROVISIONAL", msg)
        self.assertIn("a refusal below may be about THIS box's transient "
                      "state and not the form", msg)


class TheDangerDirectionMutantGuard(CustomTestCase):
    """Structural companion to the manually-run mutant (the order's own
    wording: "Fremdlast-Term entfernt -> die Zeile behauptet weiter ein
    Verdikt und der Test muss sterben"). Pins that the box-state line's
    computation is a REAL call, not a hardcoded string, so a future edit
    that quietly drops `foreign_load_now_gib(...)` from the line breaks
    this test rather than silently shipping a permanently-quiet-looking
    line."""

    def test_the_line_is_built_from_a_real_foreign_load_now_gib_call(self):
        src = inspect.getsource(hl.choose)
        i = src.index("WEG2-DRY-RUN-BOX-STATE")
        window = src[max(0, i - 400):i + 1400]
        self.assertIn("foreign_load_now_gib(cg_current_bytes)", window,
                      "the line must be computed FROM the live reading, "
                      "not a constant or a cached string")
        self.assertIn("QUIET_BASELINE_CG_CURRENT_GIB", window)
        self.assertIn("FOREIGN_LOAD_PROVISIONAL_THRESHOLD_GIB", window)

    def test_choose_never_folds_foreign_load_into_ok(self):
        """The decision, pinned structurally: `_fl_now`/foreign load must
        never appear in the `ok = ...` conjunction -- see
        ProvisionalNeverChangesTheVerdict for the behavioural half."""
        src = inspect.getsource(hl.choose)
        i = src.index("ok = moments_ok and peak_ok and cushion_ok")
        line = src[i:i + 120].splitlines()[0]
        self.assertNotIn("_fl_now", line)
        self.assertNotIn("foreign_load", line)


class TheLauncherThreadsTheThreeReadingsThrough(CustomTestCase):
    """Structural: `choose_host_ledger` is the ONE call site that already
    reads the cgroup (`cg = host_ledger.read_cgroup(cgroup_root)`) for every
    other term on the ARM line -- #1392 rides the same reading rather than
    taking a second one, and captures the timestamp at that read, not
    reconstructed later from a log line."""

    def test_the_timestamp_is_captured_at_the_cgroup_read_not_after(self):
        from sglang.srt.weg2 import launcher

        src = inspect.getsource(launcher.choose_host_ledger)
        i = src.index("_box_state_at = time.strftime(")
        j = src.index("cg = host_ledger.read_cgroup(cgroup_root)")
        self.assertLess(i, j, "the timestamp must be captured BEFORE (or at) "
                        "the read it is stamping, not reconstructed later")

    def test_ledger_kw_forwards_all_three_from_the_pricing_cgroup_reading(self):
        from sglang.srt.weg2 import launcher

        src = inspect.getsource(launcher.choose_host_ledger)
        self.assertIn("box_state_at=_box_state_at", src)
        # `cg.get(...)`, not `_cg0.get(...)`: `cg` is the SAME reading
        # `cg_ceiling`/`reclaimable_bytes`/`cg_oom_kill` already price on
        # this ladder -- `_cg0` is a separate, earlier PEAK-BASELINE
        # snapshot for a different line, and forwarding from it would make
        # the box-state line lie about which instant the rest of the ARM
        # line was actually priced against.
        self.assertIn('cg_anon_bytes=cg.get("anon")', src)
        self.assertIn('cg_shmem_bytes=cg.get("shmem")', src)
        self.assertNotIn('cg_anon_bytes=_cg0', src)
        self.assertNotIn('cg_shmem_bytes=_cg0', src)

    def test_host_ledger_choose_accepts_the_three_new_kwargs(self):
        sig = inspect.signature(hl.choose)
        for name in ("box_state_at", "cg_anon_bytes", "cg_shmem_bytes"):
            self.assertIn(name, sig.parameters)
            self.assertIsNone(sig.parameters[name].default) if name != "box_state_at" \
                else self.assertEqual(sig.parameters[name].default, "")


# ---------------------------------------------------------------------------
# END-TO-END, per the coordinator's 2026-09-14 order: not `host_ledger.choose`
# called directly, but `launcher.main(['--dry-run', ...])` through the REAL
# wiring, using #1390's own `_HermeticMainBase` / quiet-box fixture rather
# than a second hermetic harness for the identical seam.
# ---------------------------------------------------------------------------


def _fake_loaded_host(tmp: str) -> tuple:
    """The IDENTICAL quiet-box reading `_fake_quiet_host` writes, except
    `memory.current` (and `memory.peak`, read the same way) pushed to
    QUIET_BASELINE_CG_CURRENT_GIB + FOREIGN_LOAD_PROVISIONAL_THRESHOLD_GIB
    + 1 GiB -- one GiB PAST the threshold #1392's PROVISIONAL text keys on,
    never AT it (the ARM-line's own comparison is a strict ``>``). Every
    other reading (MemAvailable, anon/file/shmem/unevictable/slab) is
    UNCHANGED from the quiet fixture: this isolates the ONE axis under test.
    """
    loaded_current_b = int(
        (hl.QUIET_BASELINE_CG_CURRENT_GIB
         + hl.FOREIGN_LOAD_PROVISIONAL_THRESHOLD_GIB + 1.0) * GIB
    )
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
        f.write(f"{loaded_current_b}\n")
    with open(os.path.join(cg, "memory.peak"), "w") as f:
        f.write(f"{loaded_current_b}\n")
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


class _HermeticMainBaseLoaded(unittest.TestCase):
    """SAME isolation as `_HermeticMainBase` (#1390: replayed NVML cards, a
    private guaranteed-empty SHM_DIR) -- the ONE difference is the host
    reading: `_fake_loaded_host` instead of `_fake_quiet_host`, which is the
    whole point (the OTHER half of the asymmetry #1392 exists to make
    visible). Not a subclass of `_HermeticMainBase` overriding `setUp`,
    because that would run the quiet fixture first and the loaded one
    second, leaving two live mock.patch layers active at once."""

    def setUp(self):
        super().setUp()
        self._old_replay_env = os.environ.get(nvml_registry.ENV_NVML_REPLAY)
        os.environ[nvml_registry.ENV_NVML_REPLAY] = _NVML_REPLAY_JSON
        self._shm_tmp = tempfile.mkdtemp(prefix="weg2-1392-empty-shm-")
        self._shm_patch = mock.patch.object(launcher, "SHM_DIR", self._shm_tmp)
        self._shm_patch.start()
        self._host_tmp = tempfile.mkdtemp(prefix="weg2-1392-loaded-host-")
        meminfo, cg = _fake_loaded_host(self._host_tmp)
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
        import io
        from contextlib import redirect_stdout

        buf = io.StringIO()
        rc = None
        exc = None
        try:
            with redirect_stdout(buf):
                rc = launcher.main(argv)
        except Exception as e:  # the refusal path raises rather than returning
            exc = e
        return rc, buf.getvalue(), exc


def _box_state_line_from_stdout(out: str) -> str:
    # NOT startswith: `main()`'s own logger prefixes every line with
    # "[<timestamp>] WEG2-LAUNCH " before the marker -- the SAME reason
    # `_box_state_line` (the direct host_ledger.choose() helper above) does
    # not need this, because `choose()`'s own returned `lines` carry no such
    # prefix at all.
    hit = [l for l in out.splitlines() if "WEG2-DRY-RUN-BOX-STATE" in l]
    assert len(hit) == 1, (
        f"expected exactly one WEG2-DRY-RUN-BOX-STATE line, found {len(hit)} "
        f"-- either the line stopped printing or something duplicated it"
    )
    return hit[0]


@_NEEDS_CHECKPOINT
class TheRealDryRunLineOnAQuietBox(_HermeticMainBase):
    """`launcher.main(['--dry-run', ...])`, the REAL wiring end to end, on
    the quiet-box fixture -- not `host_ledger.choose()` called directly."""

    def test_the_real_printed_line_says_not_stale_suspect(self):
        rc, out, exc = self.run_main(_main_argv("t1392quiet"))
        self.assertIsNone(exc, f"main() raised: {exc}\n{out}")
        line = _box_state_line_from_stdout(out)
        self.assertIn("not stale-suspect", line)
        self.assertNotIn("PROVISIONAL", line)
        self.assertIn("memavail=", line)
        self.assertIn("anon=", line)
        self.assertIn("shmem=", line)


@_NEEDS_CHECKPOINT
class TheRealDryRunLineOnALoadedBox(_HermeticMainBaseLoaded):
    """The other half: the SAME argv, the SAME quiet identity/checkpoint,
    ONLY the host reading changed -- `main()`'s printed line must say so."""

    def test_the_real_printed_line_says_PROVISIONAL(self):
        rc, out, exc = self.run_main(_main_argv("t1392loaded"))
        self.assertIsNone(exc, f"main() raised: {exc}\n{out}")
        line = _box_state_line_from_stdout(out)
        self.assertIn("PROVISIONAL", line)
        self.assertIn("foreign_load_now=", line)

    def test_pflicht_mutant_removing_the_foreign_load_term_still_claims_a_verdict(self):
        """THE COORDINATOR'S OWN DANGER DIRECTION, verbatim: "Fremdlast-Term
        aus der Zeile entfernen -> die Zeile behauptet weiter ein Verdikt ->
        der Test muss sterben." Run against the REAL file
        (`host_ledger.FOREIGN_LOAD_PROVISIONAL_THRESHOLD_GIB`), not a
        planted fixture: NOT "the reading is unmeasured" (`None` is an
        already-supported, honestly-printed state -- see
        `ProvisionalNeverChangesTheVerdict`'s own tests above) but "the
        line's own EFFECT of a real, measured, genuinely-loaded box was
        silently removed" -- the threshold patched to infinity so
        `foreign_load_now` still prints its real, honest number (the box
        genuinely IS ~1 GiB over the real threshold) while the PROVISIONAL
        clause can never fire again, regardless of how loaded the box is.
        Applied via `mock.patch` against the live module constant `main()`
        actually reads through `host_ledger.choose` -- not a copy, not a
        subclass override -- so this exercises exactly the code path the
        real boot runs.
        """
        with mock.patch.object(
            hl, "FOREIGN_LOAD_PROVISIONAL_THRESHOLD_GIB", float("inf"),
        ):
            rc, out, exc = self.run_main(_main_argv("t1392mutant"))
        self.assertIsNone(exc, f"main() raised: {exc}\n{out}")
        line = _box_state_line_from_stdout(out)
        # The mutant's own failure mode, stated as an assertion rather than
        # only prose: the line still exists, still names a real
        # foreign_load_now figure (the box genuinely IS loaded, and this
        # number proves the code still MEASURED it) -- it merely lost the
        # ability to ever ACT on that measurement in the text.
        self.assertIn("foreign_load_now=", line)
        self.assertNotIn("not measured", line)
        self.assertNotIn(
            "PROVISIONAL", line,
            "with the threshold neutralised, the REAL loaded box's dry run "
            "no longer flags itself as provisional even though "
            "foreign_load_now proves the box IS loaded -- this is exactly "
            "the mutant the coordinator named: a verdict that keeps "
            "claiming FUNDABLE (or a refusal) while going silent about the "
            "one signal that would have told a reader to re-verify",
        )


if __name__ == "__main__":
    unittest.main()
