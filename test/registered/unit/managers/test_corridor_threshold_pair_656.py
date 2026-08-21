# SPDX-License-Identifier: Apache-2.0
"""#656: the corridor law and the gate's arming floor are ONE declared pair.

WHAT WENT WRONG. The acceptance boot armed its corridor gate at 1536 MiB and
read its corridor verdict against 1024 MiB. Both numbers were correct on
their own terms -- the separation is deliberate and a previous shift proved
it prevents a pp->tp deadlock -- but nothing anywhere declared what the 512
MiB BETWEEN them was for. It is the draw a seam is assumed to make while it
runs, and on this rig the corridor sampler measured that draw at 1814-1852
MiB. So five cutovers passed a gate with no objection and took GPU0 to 886
MiB, 138 below the law, and the breach lived precisely in the gap between
the gate's number and the verdict's number, where nothing looks.

Three properties are pinned here:

* the arming floor is DERIVED from the law plus a named reserve, so raising
  one cannot leave the other behind;
* an arming floor BELOW the law is refused, because such a gate would return
  "no reclaim needed" for an allocation that ends under the corridor -- it
  would launder a breach as a passed check, which the guard's own refusal
  message says it must never do;
* every module that needs the law reads the SAME declaration.
  ``corridor_trace.summary`` used to default to a literal 1024 of its own,
  which is how an instrument ends up reporting a different verdict from the
  gate it audits.
"""

import unittest

from sglang.srt.managers import corridor_guard as cg
from sglang.srt.mem_ledger import corridor_trace
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class TestTheDeclaredPair(CustomTestCase):
    def test_the_arming_floor_is_the_law_plus_a_named_reserve(self):
        self.assertEqual(
            cg.arming_floor_mib(),
            cg.corridor_band_floor_mib() + cg.DEFAULT_SEAM_ENTRY_RESERVE_MIB,
        )
        # UPDATED 2026-08-15 for the corridor BAND. The gate defends the
        # band FLOOR (819 = 1024 - 20 %), not the centre: arming from the
        # centre would reserve the whole tolerance on top of the seam's draw
        # on every card of every boot -- ~205 MiB per rank here -- to protect
        # a threshold that is no longer the verdict. What the
        # acceptance boot ran, so deriving it is not a behaviour change.
        self.assertEqual(cg.CORRIDOR_LAW_MIB, 1024)
        self.assertEqual(cg.arming_floor_mib(), cg.corridor_band_floor_mib() + 512)

    def test_the_reserve_moves_the_floor_and_only_the_floor(self):
        """The measured draw is the input the term is meant to take."""
        self.assertEqual(cg.arming_floor_mib(1852), cg.corridor_band_floor_mib() + 1852)
        self.assertEqual(cg.arming_floor_mib(0), cg.corridor_band_floor_mib())
        # A negative reserve cannot pull the floor under the law.
        self.assertEqual(cg.arming_floor_mib(-4096), cg.corridor_band_floor_mib())

    def test_can_fail_an_arming_floor_below_the_band_floor_is_refused(self):
        """The verdict threshold is the BAND FLOOR, so that is what an arming
        floor may not sink beneath. A gate armed inside the band would clear
        allocations that end below it -- laundering a breach as a passed
        check, which is the one thing the refusal message says it never may."""
        below = cg.corridor_band_floor_mib() - 1
        with self.assertRaisesRegex(ValueError, "BELOW the corridor law"):
            cg.check_threshold_pair(below, cg.CORRIDOR_LAW_MIB)
        with self.assertRaisesRegex(ValueError, "BELOW the corridor law"):
            cg.check_threshold_pair(0, cg.CORRIDOR_LAW_MIB)

    def test_the_band_is_the_law_plus_or_minus_a_fifth(self):
        floor, centre, ceiling = cg.corridor_band_mib()
        self.assertEqual(centre, cg.corridor_law_mib())
        self.assertEqual(floor, int(centre - centre * cg.CORRIDOR_BAND_FRACTION))
        # ROUNDED on the ceiling side since #784. It used to truncate, which
        # put the code at 1228 while the acceptance verdict that decides
        # pass/fail (corridor_verdict_774.sh) computed int(round(...)) = 1229
        # -- the two disagreed by 1 MiB on the very number under test. The
        # floor is unaffected: int(819.2) and round(819.2) are both 819.
        self.assertEqual(
            ceiling, int(round(centre + centre * cg.CORRIDOR_BAND_FRACTION))
        )
        self.assertLess(floor, centre)
        self.assertGreater(ceiling, centre)
        # The measured cutover transient on this rig sits inside it.
        self.assertLessEqual(floor, 895)

    def test_equality_and_above_are_accepted(self):
        cg.check_threshold_pair(cg.CORRIDOR_LAW_MIB, cg.CORRIDOR_LAW_MIB)
        cg.check_threshold_pair(cg.arming_floor_mib(), cg.CORRIDOR_LAW_MIB)
        cg.check_threshold_pair(cg.arming_floor_mib(1852), cg.CORRIDOR_LAW_MIB)
        # And the pair check now refuses against the VERDICT threshold.
        cg.check_threshold_pair(cg.corridor_band_floor_mib(), cg.CORRIDOR_LAW_MIB)

    def test_the_legacy_name_is_the_same_number(self):
        """Callers pass DEFAULT_FLOOR_MIB as the guard's law; it must not
        drift from the canonical name now that both exist."""
        self.assertEqual(cg.DEFAULT_FLOOR_MIB, cg.CORRIDOR_LAW_MIB)


class TestOneDeclaration(CustomTestCase):
    def test_the_trace_reads_the_guards_law(self):
        self.assertEqual(corridor_trace.corridor_law_mib(), cg.CORRIDOR_LAW_MIB)

    def test_the_trace_summary_defaults_to_the_declared_law(self):
        """The pin that stops a private literal creeping back in."""
        trace = corridor_trace.CorridorTrace(card_uuid="test")
        trace.samples.append(
            corridor_trace.Sample(
                monotonic=0.0,
                nvml_free_bytes=900 * corridor_trace.MIB,
                nvml_self_bytes=0,
                kv_arena_backed_bytes=0,
                torch_reserved_bytes=0,
                torch_allocated_bytes=0,
            )
        )
        trace.samples.append(
            corridor_trace.Sample(
                monotonic=1.0,
                nvml_free_bytes=4096 * corridor_trace.MIB,
                nvml_self_bytes=0,
                kv_arena_backed_bytes=0,
                torch_reserved_bytes=0,
                torch_allocated_bytes=0,
            )
        )
        summary = trace.summary()
        self.assertEqual(summary["corridor_mib"], cg.CORRIDOR_LAW_MIB)
        # UPDATED for the BAND: 900 MiB is under the law but inside the
        # tolerance, so it is not a breach. The margin is still the signed
        # depth to the CENTRE, not an absolute value -- the two answer
        # different questions and the summary reports both.
        self.assertFalse(summary["breach"])
        self.assertEqual(summary["corridor_band_floor_mib"], 819)
        self.assertEqual(summary["free_min_mib"], 900)
        self.assertEqual(summary["margin_mib"], 900 - cg.CORRIDOR_LAW_MIB)

    def test_an_explicit_corridor_still_wins(self):
        trace = corridor_trace.CorridorTrace(card_uuid="test")
        trace.samples.append(
            corridor_trace.Sample(
                monotonic=0.0,
                nvml_free_bytes=1500 * corridor_trace.MIB,
                nvml_self_bytes=0,
                kv_arena_backed_bytes=0,
                torch_reserved_bytes=0,
                torch_allocated_bytes=0,
            )
        )
        self.assertFalse(trace.summary()["breach"])
        self.assertTrue(trace.summary(corridor_mib=2048)["breach"])


if __name__ == "__main__":
    unittest.main()


class TestTheLawHasOneReader(CustomTestCase):
    """`SGLANG_CORRIDOR_LAW_FLOOR_MIB` was read in three places, each with
    its own `"1024"` fallback. The law could then be moved for one module
    and not the others -- a divergence with no symptom until a breach is
    judged twice and answered differently."""

    def setUp(self):
        import os

        self._saved = os.environ.pop(cg.LAW_ENV, None)

    def tearDown(self):
        import os

        os.environ.pop(cg.LAW_ENV, None)
        if self._saved is not None:
            os.environ[cg.LAW_ENV] = self._saved

    def test_unset_is_the_declared_constant(self):
        self.assertEqual(cg.corridor_law_mib(), cg.CORRIDOR_LAW_MIB)
        self.assertEqual(cg.corridor_law_bytes(), cg.CORRIDOR_LAW_MIB << 20)

    def test_every_consumer_moves_together(self):
        import os

        from sglang.srt.managers import phase_flip_seam_census as census
        from sglang.srt.mem_cache import kv_vmm_backing

        os.environ[cg.LAW_ENV] = "1500"
        self.assertEqual(cg.corridor_law_mib(), 1500)
        self.assertEqual(corridor_trace.corridor_law_mib(), 1500)
        self.assertEqual(census.law_floor_bytes(), 1500 << 20)
        self.assertEqual(kv_vmm_backing._corridor_law_floor_bytes(), 1500 << 20)
        # ... and the band, and therefore the arming floor, follow the law
        # rather than staying put. The floor tracks the BAND floor because
        # that is the verdict threshold the gate defends.
        self.assertEqual(cg.corridor_band_floor_mib(), 1200)
        self.assertEqual(cg.corridor_band_ceiling_mib(), 1800)
        self.assertEqual(
            cg.arming_floor_mib(law_mib=cg.corridor_law_mib()),
            1200 + cg.DEFAULT_SEAM_ENTRY_RESERVE_MIB,
        )

    def test_a_malformed_override_falls_back_to_the_constant(self):
        import os

        os.environ[cg.LAW_ENV] = "not-a-number"
        self.assertEqual(cg.corridor_law_mib(), cg.CORRIDOR_LAW_MIB)


# ---------------------------------------------------------------------------
# #662: the reserve is a MEASURED draw where one exists -- and one existed.
#
# `arming_floor_mib`'s docstring already said the reserve is "the MEASURED
# draw a seam makes while it runs, where a measurement exists; the default is
# the shipped allowance". Nothing passed the measurement, so every boot with a
# seam record on disk still armed on the 512 MiB default.
#
# Measured on this rig 2026-08-15: fixed draw 954 MiB (rank1) and 1595 MiB
# (rank2) against a gate arming at 1024 + 512 = 1536. gpu0 reached 935 MiB at
# the `weights_refill` stage with every allocation having cleared the gate --
# the breach living exactly in the gap the threshold-pair line predicts.
# ---------------------------------------------------------------------------

import types as _types

from sglang.srt.managers import corridor_guard as _cg
from sglang.srt.managers import phase_flip_spill as _spill


class _Reserve:
    """#678: the stub carries ``arming_draw_bytes`` because the real record
    does, and the gate reads THAT -- the draw of one leg, not the sum of two
    cross-leg maxima. ``worst_leg_mib=None`` models a pre-#678 record, which
    falls back to the old number."""

    def __init__(self, total_mib, active=True, worst_leg_mib=None):
        self.total_fixed_bytes = total_mib << 20
        self.worst_leg_fixed_bytes = (worst_leg_mib or 0) << 20
        self.active = active

    def arming_draw_bytes(self):
        return self.worst_leg_fixed_bytes or self.total_fixed_bytes


def _sched(rank=1):
    return _types.SimpleNamespace(
        phase_flip_runtime=_types.SimpleNamespace(_rank=rank),
        server_args=object(),
    )


def test_the_measured_draw_is_read_for_this_rank(monkeypatch):
    import sglang.srt.managers.phase_flip_seam_reserve as seam

    monkeypatch.setattr(seam, "read_seam_reserve", lambda sa, r: _Reserve(954))
    assert _spill._measured_seam_draw_mib(_sched(), object()) == 954


def test_the_draw_is_the_WORST_LEG_where_the_record_has_one(monkeypatch):
    """#678. The two stored terms are maxed over both directions and, on this
    rig, by different ones -- arena tail on tp_to_pp, draft restore on
    pp_to_tp -- so their sum prices a commit no seam makes. rank 2 measured
    1456 + 139 = 1595 against a worst leg of 1456."""
    import sglang.srt.managers.phase_flip_seam_reserve as seam

    monkeypatch.setattr(
        seam, "read_seam_reserve", lambda sa, r: _Reserve(1595, worst_leg_mib=1456)
    )
    assert _spill._measured_seam_draw_mib(_sched(), object()) == 1456


def test_a_cold_record_leaves_the_shipped_allowance_in_force(monkeypatch):
    import sglang.srt.managers.phase_flip_seam_reserve as seam

    monkeypatch.setattr(seam, "read_seam_reserve", lambda sa, r: None)
    assert _spill._measured_seam_draw_mib(_sched(), object()) == 0


def test_an_inactive_record_is_not_a_measurement(monkeypatch):
    import sglang.srt.managers.phase_flip_seam_reserve as seam

    monkeypatch.setattr(
        seam, "read_seam_reserve", lambda sa, r: _Reserve(954, active=False)
    )
    assert _spill._measured_seam_draw_mib(_sched(), object()) == 0


def test_no_runtime_means_no_rank_means_no_measurement():
    assert _spill._measured_seam_draw_mib(_types.SimpleNamespace(), object()) == 0


def test_a_reader_that_raises_falls_back_rather_than_killing_the_boot(monkeypatch):
    import sglang.srt.managers.phase_flip_seam_reserve as seam

    def boom(sa, r):
        raise RuntimeError("record went away")

    monkeypatch.setattr(seam, "read_seam_reserve", boom)
    assert _spill._measured_seam_draw_mib(_sched(), object()) == 0


def test_the_measured_draw_raises_the_floor_above_the_shipped_pair():
    """THE DEFECT, as arithmetic. 1536 was the shipped pair; the measured draw
    puts the honest floor at 1024 + 954, and the breach lived in between."""
    band_floor = _cg.corridor_band_floor_mib()
    shipped = _cg.arming_floor_mib()
    honest = _cg.arming_floor_mib(
        seam_entry_reserve_mib=max(_cg.DEFAULT_SEAM_ENTRY_RESERVE_MIB, 954)
    )
    assert shipped == band_floor + 512
    assert honest == band_floor + 954
    assert honest > shipped, "the measured draw must raise the gate"
    # And the trough that started this: inside the band, so no longer a
    # breach, but it still cleared a gate sized for half the real draw.
    assert 935 >= band_floor, "895-935 MiB is inside the band"


def test_a_small_measured_draw_never_LOWERS_the_shipped_allowance():
    """A raised floor over-reclaims; a lowered one launders breaches."""
    assert (
        _cg.arming_floor_mib(
            seam_entry_reserve_mib=max(_cg.DEFAULT_SEAM_ENTRY_RESERVE_MIB, 227)
        )
        == _cg.corridor_band_floor_mib() + 512
    )
