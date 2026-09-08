# SPDX-License-Identifier: Apache-2.0
"""Weg 2 (#1233, fix 7): the fix-6 wires, pinned where a mutation can reach them.

The fix-6 review left three surviving mutants, all of the same class: the code
that DECIDES was pinned, and the code that CARRIES the decision into a boot was
not.  Measured by that review, each against the full 152-test slice:

* ``launcher``'s ledger call site stops passing ``cg["reclaimable"]`` (pass
  ``None`` instead) -- the whole boot reverts to fix 5's denominator, at
  weg2dk5's own launch readings the entire ladder refuses, and **152 tests stay
  green**.  This is the ONLY wire that carries fix 6 into a boot.
* the new ``W11b`` refusal is gated off -- **152 green**.
* the W11b tolerance is made one-sided (``abs()`` dropped) -- **152 green**,
  and note which side: the single calibration sample is NEGATIVE (-109.9 MiB),
  while the shape the gate hunts is POSITIVE.

The cause is structural, not an oversight: both call sites sat inside
``launcher.main``, behind NVML enumeration, a tmpfs mount and two launched
servers, so nothing hermetic could execute them.  Fix 7 lifts them into two
named functions -- :func:`launcher.choose_host_ledger` and
:func:`launcher.gate_w11` -- which ``main`` now calls, and drives those.  The
tests below are the reason those functions exist; a gate nothing can reach is
a gate nothing can pin.

Hermetic: ``CUDA_VISIBLE_DEVICES=""``, no server, no NVML, no checkpoint.  The
host is a fake ``/proc/meminfo`` and a fake cgroup tree written per test; group
P's log is a fake log carrying one real ``armed`` line shape.
"""

import inspect
import os
import tempfile
import unittest

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

from sglang.srt.weg2 import host_ledger, launcher
from sglang.test.test_utils import CustomTestCase

GIB = host_ledger.GIB

# ------------------------------------------------------- MEASURED, boot weg2dk5
# The launch readings of the last boot this line actually launched from, so the
# arm this seam chooses is the arm the record names.  Front log 21:05:49Z
# ``WEG2-HOST-LEDGER TERMS memtotal=118.05`` and the memts row of the same
# second (/spinning/gpu-arb/memts_weg2_weg2dk5.csv):
# ``cg_current_b=22719148032, cached_kb=9859520, shmem_kb=3393080,
# pagecache_ex_shmem_kb=6466440, oom_kill=18``.
DK5_MEMTOTAL_KB = 123_781_120          # = 126,751,866,880 B = 118.05 GiB
DK5_MEMAVAIL_KB = 108_589_919          # = 111,196,077,056 B = 103.56 GiB
DK5_CG_CURRENT_B = 22_719_148_032      # 21.16 GiB
DK5_SHMEM_B = 3_393_080 * 1024
DK5_PAGECACHE_EX_SHMEM_B = 6_466_440 * 1024   # 6.17 GiB, the reclaimable share
#: C19: the host weights term is the measured per-card ring table, not a chunk
#: count.  Same figures the s3s4 refusal suite pins (Sigma H / Sigma image_P).
DK5_RING_BYTES = 32964 * 1024 * 1024
DK5_RING_SPAN1_BYTES = 29912 * 1024 * 1024
DK5_STORE_MIN_GIB = 8.0


def _source_of(module: str) -> str:
    """The module's own source text -- for facts that live in a COMMENT."""
    return inspect.getsource({"launcher": launcher, "host_ledger": host_ledger}[module])


def _fake_host(tmp: str, reclaimable_b: int = DK5_PAGECACHE_EX_SHMEM_B) -> tuple:
    """Write a fake /proc/meminfo and a fake cgroup tree; return their paths.

    ``slab_reclaimable`` is 0 exactly as the memts sampler read it -- the row
    has no such column, so the record's arm is priced with that term charged as
    spent.  The fake reproduces the reading, not an idealised box.
    """
    meminfo = os.path.join(tmp, "meminfo")
    with open(meminfo, "w") as f:
        f.write(
            f"MemTotal:       {DK5_MEMTOTAL_KB} kB\n"
            f"MemFree:        1000000 kB\n"
            f"MemAvailable:   {DK5_MEMAVAIL_KB} kB\n"
            f"Shmem:          {DK5_SHMEM_B // 1024} kB\n"
            f"SwapTotal:      0 kB\n"
        )
    cg = os.path.join(tmp, "cgroup")
    os.makedirs(cg, exist_ok=True)
    with open(os.path.join(cg, "memory.current"), "w") as f:
        f.write(f"{DK5_CG_CURRENT_B}\n")
    with open(os.path.join(cg, "memory.peak"), "w") as f:
        f.write(f"{DK5_CG_CURRENT_B}\n")
    with open(os.path.join(cg, "memory.max"), "w") as f:
        f.write("max\n")          # this LXC container publishes no finite ceiling
    with open(os.path.join(cg, "memory.events"), "w") as f:
        f.write("low 0\nhigh 0\nmax 0\noom 0\noom_kill 18\n")
    with open(os.path.join(cg, "memory.stat"), "w") as f:
        f.write(
            f"anon 12000000000\n"
            f"file {DK5_SHMEM_B + reclaimable_b}\n"   # v2: `file` INCLUDES `shmem`
            f"shmem {DK5_SHMEM_B}\n"
            f"unevictable 0\n"
            f"slab_reclaimable 0\n"
        )
    return meminfo, cg


def _seam_lines():
    """The seam's printed lines, whether it funds an arm or refuses.

    FIX 8: at weg2dk5's readings the measured image refuses every arm, and both
    the TERMS and the RUN-PEAK lines are printed with the refusal exactly as
    they are with a choice -- so this evidence sits in one place either way.
    """
    with tempfile.TemporaryDirectory() as tmp:
        meminfo, cg = _fake_host(tmp)
        try:
            _arm, _store, lines, _r = launcher.choose_host_ledger(
                DK5_STORE_MIN_GIB, DK5_RING_BYTES, DK5_RING_SPAN1_BYTES, meminfo_path=meminfo, cgroup_root=cg,
                record_path=os.path.join(tmp, "no-such-record.json"),
            )
            return lines
        except (host_ledger.Weg2HostLedgerRefused,
                host_ledger.Weg2HostRunPeakRefused) as e:
            return str(e).splitlines()


class TestTheLedgerWireIsLoadBearing(CustomTestCase):
    """MU-E: ``cg_reclaimable_bytes=cg["reclaimable"]`` -> ``None``.

    Driven through the real function ``main`` calls, from files, so the mutant
    changes what this seam RETURNS -- not merely what a hand-built kwarg says.
    """

    def test_the_seam_charges_only_the_non_reclaimable_share_of_the_reading(self):
        # UPDATED BY FIX 8: this test used to assert that weg2dk5's readings
        # choose S=1 M=1200 store 8 through this seam.  The dormant image is now
        # MEASURED (38.63 GiB, boot weg2dk7) instead of summed from the #809
        # weight-tag census (28.83 GiB), and at the true price these readings
        # fund no arm -- so what the seam is pinned on is the DENOMINATOR it
        # exists to carry, and the refusal that denominator now produces.
        with tempfile.TemporaryDirectory() as tmp:
            meminfo, cg = _fake_host(tmp)
            with self.assertRaises(host_ledger.Weg2HostLedgerRefused) as cm:
                launcher.choose_host_ledger(
                    DK5_STORE_MIN_GIB, DK5_RING_BYTES, DK5_RING_SPAN1_BYTES, meminfo_path=meminfo, cgroup_root=cg,
                    record_path=os.path.join(tmp, "no-such-record.json"),
                )
        terms = [ln for ln in str(cm.exception).splitlines()
                 if "WEG2-HOST-LEDGER TERMS" in ln][0]
        # base = 118.05 (ceiling) - 14.99 (non-reclaimable) - 10 (CLI) = 93.05.
        self.assertIn("of which reclaimable=6.17 GiB", terms)
        self.assertIn("non-reclaimable=14.99 GiB charged", terms)
        self.assertIn("base=93.05 GiB", terms)
        # And the image the refusal is priced with is the measured one, labelled.
        self.assertIn("image_P=38.63 GiB", terms)

    def test_the_seam_hands_the_reading_and_the_arm_back_when_the_box_can_fund_one(self):
        # The control: the same seam, same files, a store floor the box can
        # meet -- the reading reaches the caller (main stores it in the boot
        # state) and the reclaimable share is what it was measured to be.
        with tempfile.TemporaryDirectory() as tmp:
            meminfo, cg = _fake_host(tmp)
            try:
                _arm, _store, _lines, reading = launcher.choose_host_ledger(
                    0.0, DK5_RING_BYTES, DK5_RING_SPAN1_BYTES, meminfo_path=meminfo, cgroup_root=cg,
                    record_path=os.path.join(tmp, "no-such-record.json"),
                )
            except (host_ledger.Weg2HostLedgerRefused,
                    host_ledger.Weg2HostRunPeakRefused):
                reading = host_ledger.read_cgroup(cg)
        self.assertEqual(reading["current"], DK5_CG_CURRENT_B)
        self.assertEqual(reading["reclaimable"], DK5_PAGECACHE_EX_SHMEM_B)

    def test_the_same_box_without_that_wire_cannot_launch_at_all(self):
        # THE POINT OF THE PIN, stated as behaviour rather than as a kwarg:
        # fix 5's denominator (the whole memory.current charged) is not a
        # different arm at these readings, it is NO arm -- so a wire that stops
        # carrying the reclaimable share turns a bootable box into a refusal,
        # and the test above turns from green to an exception.
        with tempfile.TemporaryDirectory() as tmp:
            meminfo, cg = _fake_host(tmp)
            os.remove(os.path.join(cg, "memory.stat"))
            with self.assertRaises(host_ledger.Weg2HostLedgerRefused):
                launcher.choose_host_ledger(
                    DK5_STORE_MIN_GIB, DK5_RING_BYTES, DK5_RING_SPAN1_BYTES, meminfo_path=meminfo, cgroup_root=cg
                )

    def test_a_bigger_reclaimable_share_moves_the_budget_by_exactly_that_share(self):
        # Not only refuse-vs-boot: the term is monotone and load-bearing on the
        # quantity R3's pricing rests on.  UPDATED BY FIX 8: doubling the page
        # cache used to open the ladder's top arm (M=2400); at the measured
        # image price it moves the leftovers by exactly the extra 6.17 GiB
        # without funding an arm, and the size is the pin.
        def _leftover(reclaimable_b):
            with tempfile.TemporaryDirectory() as tmp:
                meminfo, cg = _fake_host(tmp, reclaimable_b=reclaimable_b)
                reading = host_ledger.read_cgroup(cg)
                mi = host_ledger.read_meminfo(meminfo)
            return host_ledger.price(
                mi["MemTotal"], mi["MemAvailable"], 1, 1200,
                ring_bytes=DK5_RING_BYTES, ring_span1_bytes=DK5_RING_SPAN1_BYTES,
                cg_current_bytes=reading["current"],
                reclaimable_bytes=reading["reclaimable"],
                cg_ceiling_bytes=mi["MemTotal"],
            ).run_leftover_gib

        self.assertAlmostEqual(
            _leftover(2 * DK5_PAGECACHE_EX_SHMEM_B) - _leftover(DK5_PAGECACHE_EX_SHMEM_B),
            DK5_PAGECACHE_EX_SHMEM_B / GIB,
            delta=0.01,
        )


# --------------------------------------------------------------------- W11 gate

_ARMED = (
    "[2026-09-07 21:07:11] WEG2 DRAFT-KV-PRODUCER armed pp_rank=2 "
    "resident_mib={resident} head_released_mib={released} nvml_delta_mib={delta} "
    "budget_mib=1800.0\n"
)


def _p_log(tmp: str, resident: float, released: float, delta: float) -> str:
    path = os.path.join(tmp, "P.log")
    with open(path, "w") as f:
        f.write("=== WEG2 group P ===\n")
        f.write(_ARMED.format(resident=resident, released=released, delta=delta))
    return path


class _SilentLog:
    def __init__(self):
        self.lines = []

    def __call__(self, msg):
        self.lines.append(msg)


class TestTheW11bRefusalFiresAtItsCallSite(CustomTestCase):
    """MU-D: the ``if not w11["accounted"]: raise`` gated off."""

    def test_an_unaccounted_build_refuses_the_boot_by_name(self):
        # The fix-2 shape: the lm_head leaves the graph (resident falls to the
        # budget) but nothing is freed, so the BUILD is 2425 MiB larger than
        # residue + release explain.  weg2dk5's own numbers with the release
        # term zeroed -- exactly the failure resident_mib is blind to.
        with tempfile.TemporaryDirectory() as tmp:
            log_p = _p_log(tmp, resident=1682.9, released=0.0, delta=3998.0)
            logger = _SilentLog()
            with self.assertRaises(launcher.Weg2LaunchRefused) as cm:
                launcher.gate_w11(log_p, logger)
        self.assertIn("W11b Weg2DraftBuildUnaccounted", str(cm.exception))
        self.assertIn("2315.1", str(cm.exception))
        # The gate LOGS its reading whichever way it goes -- the refusal is not
        # the only evidence that it ran.
        self.assertTrue(any("W11b BUILD-ACCOUNTING" in ln for ln in logger.lines))

    def test_dk5s_own_accounted_build_passes_the_same_gate(self):
        # The control that makes the refusal above a finding and not a
        # tautology: the real boot's three instruments (3998.0 against 1682.9 +
        # 2425.0 = -109.9 MiB) pass, so the gate does not simply always fire.
        with tempfile.TemporaryDirectory() as tmp:
            log_p = _p_log(tmp, resident=1682.9, released=2425.0, delta=3998.0)
            w11 = launcher.gate_w11(log_p, _SilentLog())
        self.assertTrue(w11["accounted"])
        self.assertTrue(w11["ok"])
        self.assertAlmostEqual(w11["unaccounted_mib"], -109.9, delta=0.05)

    def test_an_unmeasured_instrument_is_a_refusal_not_a_pass(self):
        # Silence is not a pass: a log with no armed line at all refuses.
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "P.log")
            with open(path, "w") as f:
                f.write("=== WEG2 group P ===\nnothing was measured here\n")
            with self.assertRaises(launcher.Weg2LaunchRefused):
                launcher.gate_w11(path, _SilentLog())


class TestTheAccountingToleranceIsTwoSided(CustomTestCase):
    """MU-B: ``abs(out["unaccounted_mib"]) <= TOL`` -> the bare comparison.

    Both directions are refusals and both are measured here, because the ONE
    calibration sample sits on the negative side while the shape the gate hunts
    is on the positive side.
    """

    def _accounting(self, resident, released, delta):
        with tempfile.TemporaryDirectory() as tmp:
            return launcher.check_draft_resident(
                _p_log(tmp, resident=resident, released=released, delta=delta)
            )

    def test_positive_beyond_the_tolerance_is_unaccounted(self):
        # VRAM held that nobody accounts for -- the fix-2 class.  400 MiB over.
        out = self._accounting(1682.9, 2425.0, 4507.9)
        self.assertAlmostEqual(out["unaccounted_mib"], 400.0, delta=0.05)
        self.assertFalse(out["accounted"])
        self.assertFalse(out["ok"])

    def test_negative_beyond_the_tolerance_is_unaccounted_too(self):
        # The accounting OVER-claims: the two explaining terms are 400 MiB
        # larger than the build the driver reports, so at least one of them is
        # measuring something that did not happen.  Without the abs() this
        # reads as a pass -- a "-400 <= 256" that grades a broken instrument as
        # a healthy build.
        out = self._accounting(1682.9, 2425.0, 3707.9)
        self.assertAlmostEqual(out["unaccounted_mib"], -400.0, delta=0.05)
        self.assertFalse(out["accounted"])
        self.assertFalse(out["ok"])

    def test_both_sides_inside_the_tolerance_are_accounted(self):
        for delta, expect in ((4307.9, 200.0), (3907.9, -200.0)):
            with self.subTest(delta=delta):
                out = self._accounting(1682.9, 2425.0, delta)
                self.assertAlmostEqual(out["unaccounted_mib"], expect, delta=0.05)
                self.assertTrue(out["accounted"])

    def test_the_constant_says_which_side_was_measured(self):
        # The comment used to name the POSITIVE residual as the shape while the
        # only sample was NEGATIVE.  A bound calibrated on one side must say so
        # where the bound lives, or the next reader takes it for measured.
        src = _source_of("launcher")
        i = src.index("P_DRAFT_BUILD_ACCOUNTING_TOL_MIB = ")
        block = src[max(0, i - 2200):i]
        self.assertIn("NEGATIVE", block)
        self.assertIn("POSITIVE", block)
        self.assertIn("NEVER been measured", block)


class TestTheWatermarkStatesItsDirection(CustomTestCase):
    """nb5: the reap watermark drops a reclaimable term it never sampled."""

    def test_the_advisory_line_names_the_bound_and_its_direction(self):
        advisory = [ln for ln in _seam_lines() if "RUN-PEAK ADVISORY" in ln][0]
        self.assertIn("UPPER bound", advisory)
        self.assertIn("UNDER-warns", advisory)
        self.assertIn("slab_reclaimable", advisory)

    def test_the_constant_does_not_claim_the_omission_makes_it_tighter(self):
        # Fix 6 wrote "can only make the watermark tighter": the sign is
        # backwards.  NOT subtracting a reclaimable term leaves the watermark
        # HIGHER, and a higher warn threshold warns LESS often.
        src = _source_of("host_ledger")
        i = src.index("OBSERVED_REAP_NONRECLAIM_BYTES = ")
        block = src[max(0, i - 2600):i]
        self.assertNotIn("make the watermark tighter", block)
        self.assertIn("UPPER BOUND", block)
        self.assertIn("UNDER-warns", block)

    def test_both_residuals_of_the_line_stay_stated_side_by_side(self):
        advisory = [ln for ln in _seam_lines() if "RUN-PEAK ADVISORY" in ln][0]
        # What the under-warn is WORTH, so the statement is a size and not a
        # disclaimer: the new bound (0.53 GiB live, 0.73 GiB at the fix-6
        # reading) is printed on the SAME line as the 3.01 GiB by which this
        # advisory under-predicted weg2dk5's own reap -- an order of magnitude
        # apart, and neither one allowed to quietly stand in for the other.
        self.assertIn("0.53 GiB", advisory)
        # The residual's SIZE is a per-boot measurement and will move (dk5
        # 3.01, dk6 4.83/7.67), so what is pinned is that the line still
        # carries one -- not the digits of the newest boot.
        self.assertIn("UNDER-prediction", advisory)
        # And the constant is still exactly the reap row minus that row's own
        # reclaimable share -- the bound is a STATEMENT, not a correction.
        self.assertEqual(
            host_ledger.OBSERVED_REAP_NONRECLAIM_BYTES,
            host_ledger.OBSERVED_REAP_CURRENT_BYTES - 28_916 * 1024,
        )


class TestMainStillCallsTheSeams(CustomTestCase):
    """The seams are only worth their tests while ``main`` is their caller.

    ``main`` resolves NVML, mounts a tmpfs and spawns two servers, so it cannot
    be executed here -- but a call site that stops calling the pinned function
    is exactly the mutation these tests would otherwise miss, and that is a
    source fact, checkable without running anything.
    """

    def test_main_delegates_the_ledger_and_the_w11_gate(self):
        src = _source_of("launcher")
        main_src = src[src.index("\ndef main("):]
        self.assertIn("choose_host_ledger(", main_src)
        self.assertIn("ring_plan.host_weights_bytes", main_src)
        self.assertIn("gate_w11(spec_p.log, log)", main_src)
        # And the four-line inline form is gone from main, not merely shadowed.
        self.assertNotIn("host_ledger.choose(", main_src)
        self.assertNotIn("check_draft_resident(spec_p.log)", main_src)


if __name__ == "__main__":
    unittest.main()
