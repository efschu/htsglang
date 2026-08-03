"""#363 slice 1 — the regime controller's DECISION layer (DESIGN_363 §20).

What is under test: the machine that DECIDES whether a layout switch pays,
plans the pair that makes it cheap, and prices the residency rung it would
run at. Nothing here executes a switch: the pointer flip, the diff-spill
executor and the pre-capture are #363 slices 2+ (`ROADMAP_456` WAVE 4).
The remaining classes of this suite land in the commits that follow this
one, alongside the code they cover.

FIXTURE PROVENANCE. The two canon shapes come from #424's phase-layout table
as quoted in `DESIGN_363_regime_controller.md` §20.1 (the battery itself is
`/spinning/gpu-battery-results/2026-08-02_424_phase_record_bench/`,
`comparison_table.md` / `RESULTS.md` §2):

  * INT8-27B — decode layout 1890.6 tok/s vs prefill layout 1847.2 tok/s at
    s=1 ON PREFILL, i.e. the concentrating layout loses -2.3 % on its own
    phase.
  * FP8-27B  — +24.1 % prefill-layout gain (1231.7 -> 1528.9 tok/s at s=1)
    against a -32.8 % decode-layout cost (125.1 -> 84.1 tok/s at bs=1).

Following the #434 generality pattern, the tests are built on SYNTHETIC
tables SHAPED like those numbers rather than on a rig fixture: the decision
must follow the table, not the rig. Where a real number appears it is
labelled with the battery it came from and is used as a fixture, never as a
constant the code may read. §20.1 itself flags that the INT8 "one layout"
canon rests on a re-pinned prefill vector (`NOTE_433_int8_prefill_vector.md`,
`/root/addendum_435.md`) — which is exactly why the autocheck decides from
the table it is handed and knows nothing about the canon.

The geometry fixture is §20.3's own example: the INT8-27B ``10,1,1`` <->
even-split pair, where the big card's decode shard is a prefix of its prefill
shard. It is pinned on a readable 12-unit toy grid in
:class:`TestOverlapMath`.
"""

import unittest

from sglang.srt.planner.cost_model import Provenance, Rate
from sglang.srt.planner.regime_switch import (
    DEFAULT_PAIR_TOLERANCE_PCT,
    GIB,
    LayoutVector,
    PhaseCandidate,
    PhaseTable,
    WorkloadShape,
    layout_overlap,
    residency_rung,
    solve_layout_pair,
    unit_ranges,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=10, suite="base-a-test-cpu")

_B424 = (
    "#424 phase_record_bench comparison_table.md, quoted in DESIGN_363 §20.1"
)
_SYNTH = "synthetic fixture (#434 pattern): shaped like " + _B424

PREFILL = LayoutVector("prefill", (10, 1, 1), kv_tokens=(2, 11, 10))
DECODE = LayoutVector("decode", (1, 1, 1), kv_tokens=(1, 1, 1))


def _table(triple, p_on_p, p_on_d, d_on_p, d_on_d, *, absent=(), floor=3.0):
    """A 2x2 phase table; names in ``absent`` are dropped instead of priced."""
    raw = {
        ("prefill", "prefill"): p_on_p,
        ("prefill", "decode"): p_on_d,
        ("decode", "prefill"): d_on_p,
        ("decode", "decode"): d_on_d,
    }
    cells = {}
    for key, value in raw.items():
        if key in absent:
            cells[key] = Rate.absent(
                "this arm was never run in the window", unit="tok/s"
            )
            continue
        cells[key] = Rate.measured(value, _SYNTH, unit="tok/s")
    return PhaseTable(
        triple=triple,
        layouts=(PREFILL, DECODE),
        cells=cells,
        noise_floor_pct=floor,
        noise_floor_source="synthetic A-vs-A floor supplied with the fixture",
    )


#: INT8-shaped: the decode layout wins BOTH phases (§20.1's "one layout").
INT8_SHAPED = dict(p_on_p=1847.2, p_on_d=120.0, d_on_p=1890.6, d_on_d=125.1)
#: FP8-shaped: +24.1 % on prefill, -32.8 % on decode. A real divergence.
FP8_SHAPED = dict(p_on_p=1528.9, p_on_d=84.1, d_on_p=1231.7, d_on_d=125.1)
#: The round the FP8 fixture is priced against. Long prompt, 2048-token
#: completion: benefit 3.157 s/round (9.7 % of the 32.609 s best single-layout
#: round), which two RUNG 0 switches (2.0 s) clear and two RUNG 1 switches
#: (6.05 s) do not.
FP8_ROUND = WorkloadShape(prefill_tokens=20000, decode_tokens=2048)


def _rung0_ledger():
    """A ledger with room for both layouts and both graph families."""
    overlap = layout_overlap(
        PREFILL, DECODE, units=12, bytes_per_unit=1.6 * GIB, active="decode"
    )
    return overlap, residency_rung(
        overlap,
        card_total_bytes=[48 * GIB, 40 * GIB, 40 * GIB],
        committed_bytes=[18 * GIB, 12 * GIB, 12 * GIB],
        graph_state_bytes=[3 * GIB, 3 * GIB, 3 * GIB],
    )


class TestOverlapMath(CustomTestCase):
    """§20.3's shard-overlap arithmetic, pinned on the 27B example geometry."""

    def test_unit_ranges_are_contiguous_prefix_sums(self):
        # The property the whole overlap objective rests on: an uneven-TP
        # shard is a contiguous range in rank order (tp_loaded_shard_start
        # takes sum(sizes[:rank])), so overlap is interval intersection.
        self.assertEqual(unit_ranges(12, (10, 1, 1)), [(0, 10), (10, 11), (11, 12)])
        self.assertEqual(unit_ranges(12, (1, 1, 1)), [(0, 4), (4, 8), (8, 12)])
        for vec in [(10, 1, 1), (1, 1, 1), (3, 2, 2), (5, 4, 3)]:
            ranges = unit_ranges(12, vec)
            self.assertEqual(ranges[0][0], 0)
            self.assertEqual(ranges[-1][1], 12)
            for (_, end), (start, _) in zip(ranges, ranges[1:]):
                self.assertEqual(end, start, f"gap/overlap in {vec}")

    def test_27b_big_card_prefix_costs_zero_extra(self):
        """§20.3: 'the big card's decode shard is a PREFIX of its prefill
        shard, so the 5090 carries zero extra bytes.'"""
        rep = layout_overlap(
            PREFILL, DECODE, units=12, bytes_per_unit=1.6 * GIB, active="decode"
        )
        big = rep.per_rank[0]
        self.assertEqual(big.a_range, (0, 10))
        self.assertEqual(big.b_range, (0, 4))
        self.assertTrue(big.nested, "decode must nest inside prefill on rank 0")
        # 'Zero extra bytes' is true against the LARGER of the two layouts:
        # nothing has to be allocated that the prefill layout did not already
        # need. Against the ACTIVE (decode) layout the same rank owes the six
        # units prefill holds and decode does not -- both numbers are reported
        # because only the second is the ledger item.
        self.assertEqual(big.extra_units_vs_larger, 0)
        self.assertEqual(big.extra_bytes_vs_larger, 0.0)
        self.assertEqual(big.extra_units_vs_active, 6)

    def test_27b_small_cards_union_five_twelfths(self):
        """§20.3: 'union ~5/12 against ~4/12 of the MLP'."""
        rep = layout_overlap(
            PREFILL, DECODE, units=12, bytes_per_unit=1.6 * GIB, active="decode"
        )
        mid = rep.per_rank[1]
        self.assertEqual(mid.a_range, (10, 11))
        self.assertEqual(mid.b_range, (4, 8))
        self.assertEqual(mid.intersect_units, 0, "rank 1's ranges are disjoint")
        self.assertEqual(mid.union_units, 5)  # 5/12
        self.assertEqual(mid.extra_units_vs_active, 1)  # 5/12 against 4/12
        self.assertFalse(mid.nested)

        # The third card, which §20.3's "each" implies is symmetric with the
        # second, is NOT: its prefill range [11,12) is a SUFFIX nested inside
        # its decode range [8,12), so it holds no extra bytes at all. On this
        # exact geometry the pair's extra-vs-larger cost is ONE unit, not two.
        last = rep.per_rank[2]
        self.assertEqual(last.a_range, (11, 12))
        self.assertEqual(last.b_range, (8, 12))
        self.assertTrue(last.nested)
        self.assertEqual(last.extra_units_vs_larger, 0)
        self.assertEqual(rep.extra_bytes_vs_larger, 1 * 1.6 * GIB)

    def test_totals_and_diff_are_the_non_overlapping_remainder(self):
        rep = layout_overlap(
            PREFILL, DECODE, units=12, bytes_per_unit=1.0, active="decode"
        )
        self.assertEqual(rep.total_union_units, 19)  # 10 + 5 + 4
        self.assertEqual(rep.total_intersect_units, 5)  # 4 + 0 + 1
        self.assertAlmostEqual(rep.overlap_fraction, 5 / 19)
        # §20.3: 'a diff is exactly the non-overlapping remainder'.
        self.assertEqual(rep.diff_units, 19 - 5)

    def test_identical_layouts_overlap_completely(self):
        rep = layout_overlap(
            PREFILL,
            LayoutVector("copy", (10, 1, 1)),
            units=12,
            bytes_per_unit=1.0,
            active="copy",
        )
        self.assertEqual(rep.overlap_fraction, 1.0)
        self.assertEqual(rep.extra_bytes_vs_active, 0.0)
        self.assertEqual(rep.diff_units, 0)

    def test_mismatched_rank_counts_refuse(self):
        with self.assertRaises(ValueError):
            layout_overlap(
                PREFILL,
                LayoutVector("four", (1, 1, 1, 1)),
                units=12,
                bytes_per_unit=1.0,
            )

    def test_too_few_units_for_the_ranks_refuse(self):
        with self.assertRaises(ValueError):
            layout_overlap(PREFILL, DECODE, units=2, bytes_per_unit=1.0)


class TestResidencyLadder(CustomTestCase):
    """§20.3's rungs, as ledger arithmetic on the 27B geometry."""

    def _overlap(self):
        # 12 units over an MLP mass of ~19.2 GiB -> 1.6 GiB per unit.
        return layout_overlap(
            PREFILL, DECODE, units=12, bytes_per_unit=1.6 * GIB, active="decode"
        )

    def test_rung0_when_everything_fits(self):
        rep = residency_rung(
            self._overlap(),
            card_total_bytes=[64 * GIB, 40 * GIB, 40 * GIB],
            committed_bytes=[20 * GIB, 12 * GIB, 12 * GIB],
            graph_state_bytes=[3 * GIB, 3 * GIB, 3 * GIB],
        )
        self.assertEqual(rep.rung, 0)
        self.assertTrue(rep.rung0_feasible)
        self.assertIn("both layouts and both graph families fit", rep.reason)
        # Graph-state sizes are an ESTIMATE until #286/#102 measures them.
        self.assertIs(rep.provenance, Provenance.ESTIMATE)
        self.assertIn("ESTIMATE", rep.reason)

    def test_rung1_when_the_dual_extra_does_not_fit(self):
        # The big card owes 6 units = 9.6 GiB of dual-residency extra; give
        # it only 4 GiB of headroom past the active layout and its graphs.
        rep = residency_rung(
            self._overlap(),
            card_total_bytes=[32 * GIB, 40 * GIB, 40 * GIB],
            committed_bytes=[25 * GIB, 12 * GIB, 12 * GIB],
            graph_state_bytes=[3 * GIB, 3 * GIB, 3 * GIB],
        )
        self.assertEqual(rep.rung, 1)
        self.assertEqual(rep.per_rank[0].rung, 1)
        self.assertIn("does not fit", rep.reason)
        self.assertIn("rank 0 governs", rep.reason)
        # The other two cards would have served RUNG 0 on their own; the
        # worst rank governs, because everyone waits for its diff reload.
        self.assertEqual(rep.per_rank[1].rung, 0)
        self.assertEqual(rep.per_rank[2].rung, 0)

    def test_rung2_when_not_even_the_graph_families_fit(self):
        rep = residency_rung(
            self._overlap(),
            card_total_bytes=[32 * GIB, 40 * GIB, 40 * GIB],
            committed_bytes=[30 * GIB, 12 * GIB, 12 * GIB],
            graph_state_bytes=[3 * GIB, 3 * GIB, 3 * GIB],
        )
        self.assertEqual(rep.rung, 2)
        self.assertIn("cannot be pre-captured here at all", rep.reason)

    def test_rung2_when_the_family_was_never_pre_captured(self):
        rep = residency_rung(
            self._overlap(),
            card_total_bytes=[64 * GIB, 40 * GIB, 40 * GIB],
            committed_bytes=[20 * GIB, 12 * GIB, 12 * GIB],
            pre_captured=False,
        )
        self.assertEqual(rep.rung, 2)
        self.assertIn("lazy recapture", rep.reason)

    def test_corridor_is_charged(self):
        overlap = self._overlap()
        common = dict(
            card_total_bytes=[43.2 * GIB, 40 * GIB, 40 * GIB],
            committed_bytes=[30.0 * GIB, 12 * GIB, 12 * GIB],
            graph_state_bytes=[3 * GIB, 3 * GIB, 3 * GIB],
        )
        # 30 + 3 + 9.6 = 42.6 GiB against 43.2 GiB -> fits with 0.6 GiB to
        # spare, but not once a 1 GiB corridor must stay free.
        self.assertEqual(residency_rung(overlap, **common).rung, 0)
        self.assertEqual(
            residency_rung(overlap, corridor_bytes=1 * GIB, **common).rung, 1
        )

    def test_measured_graph_state_upgrades_the_provenance(self):
        rep = residency_rung(
            self._overlap(),
            card_total_bytes=[64 * GIB, 40 * GIB, 40 * GIB],
            committed_bytes=[20 * GIB, 12 * GIB, 12 * GIB],
            graph_state_bytes=[3 * GIB, 3 * GIB, 3 * GIB],
            graph_state_provenance=Provenance.MEASURED,
        )
        self.assertIs(rep.provenance, Provenance.MEASURED)
        self.assertNotIn("ESTIMATE", rep.reason)

    def test_rank_count_mismatch_refuses(self):
        with self.assertRaises(ValueError):
            residency_rung(
                self._overlap(),
                card_total_bytes=[64 * GIB, 40 * GIB],
                committed_bytes=[20 * GIB, 12 * GIB],
            )


class TestPairObjective(CustomTestCase):
    """§20.3's planner consequence: overlap as a bounded SECONDARY objective."""

    def _cands(self):
        # Prefill: 10,1,1 is the optimum; 8,1,1 is 1 % behind but its range
        # nests better against the decode candidates. Decode: the even split
        # is the optimum, 5,4,3 is 1 % behind.
        cp = [
            PhaseCandidate(
                "prefill",
                LayoutVector("p-10", (10, 1, 1)),
                Rate.measured(1000.0, _SYNTH, unit="tok/s"),
            ),
            PhaseCandidate(
                "prefill",
                LayoutVector("p-8", (8, 2, 2)),
                Rate.measured(990.0, _SYNTH, unit="tok/s"),
            ),
        ]
        cd = [
            PhaseCandidate(
                "decode",
                LayoutVector("d-even", (1, 1, 1)),
                Rate.measured(100.0, _SYNTH, unit="tok/s"),
            ),
            PhaseCandidate(
                "decode",
                LayoutVector("d-543", (5, 4, 3)),
                Rate.measured(99.0, _SYNTH, unit="tok/s"),
            ),
        ]
        return cp, cd

    def test_prefers_the_more_overlapping_pair_within_tolerance(self):
        cp, cd = self._cands()
        sol = solve_layout_pair(cp, cd, units=12, bytes_per_unit=1.0, tolerance_pct=2.0)
        self.assertGreaterEqual(
            sol.overlap.overlap_fraction, sol.baseline_overlap_fraction
        )
        self.assertEqual(sol.considered_pairs, 4)
        self.assertLessEqual(sol.max_concession_pct, 2.0)

    def test_never_exceeds_the_stated_tolerance(self):
        """The hard contract: a secondary objective may not buy overlap with
        more than ``tolerance_pct`` of a phase's own optimum."""
        cp, cd = self._cands()
        # A perfectly-overlapping pair that is 30 % slower must NOT be taken.
        cp.append(
            PhaseCandidate(
                "prefill",
                LayoutVector("p-slow", (1, 1, 1)),
                Rate.measured(700.0, _SYNTH, unit="tok/s"),
            )
        )
        for tol in (0.0, 0.5, 1.0, 2.0, 5.0):
            sol = solve_layout_pair(
                cp, cd, units=12, bytes_per_unit=1.0, tolerance_pct=tol
            )
            self.assertLessEqual(
                sol.max_concession_pct,
                tol + 1e-9,
                f"tolerance {tol} was exceeded: {sol.concessions}",
            )
            self.assertNotEqual(sol.a.name, "p-slow")

    def test_zero_tolerance_pins_the_pure_optima(self):
        cp, cd = self._cands()
        sol = solve_layout_pair(cp, cd, units=12, bytes_per_unit=1.0, tolerance_pct=0.0)
        self.assertEqual(sol.a.name, "p-10")
        self.assertEqual(sol.b.name, "d-even")
        self.assertEqual(sol.max_concession_pct, 0.0)
        self.assertAlmostEqual(
            sol.overlap.overlap_fraction, sol.baseline_overlap_fraction
        )

    def test_a_wide_tolerance_actually_buys_overlap(self):
        """A pair that overlaps strictly better exists and is taken when the
        tolerance admits it — otherwise the objective would be inert."""
        cp = [
            PhaseCandidate(
                "prefill",
                LayoutVector("p-10", (10, 1, 1)),
                Rate.measured(1000.0, _SYNTH, unit="tok/s"),
            ),
            PhaseCandidate(
                "prefill",
                LayoutVector("p-even", (1, 1, 1)),
                Rate.measured(950.0, _SYNTH, unit="tok/s"),
            ),
        ]
        cd = [
            PhaseCandidate(
                "decode",
                LayoutVector("d-even", (1, 1, 1)),
                Rate.measured(100.0, _SYNTH, unit="tok/s"),
            )
        ]
        tight = solve_layout_pair(cp, cd, units=12, bytes_per_unit=1.0, tolerance_pct=1.0)
        loose = solve_layout_pair(cp, cd, units=12, bytes_per_unit=1.0, tolerance_pct=6.0)
        self.assertEqual(tight.a.name, "p-10")
        self.assertEqual(loose.a.name, "p-even")
        self.assertEqual(loose.overlap.overlap_fraction, 1.0)
        self.assertGreater(
            loose.overlap.overlap_fraction, tight.overlap.overlap_fraction
        )
        self.assertLessEqual(loose.max_concession_pct, 6.0)
        self.assertIn("overlap", loose.reason)

    def test_default_tolerance_is_below_the_measured_noise_floor(self):
        """Documented invariant: the secondary objective may only break ties,
        never knowingly trade measurable performance."""
        from sglang.srt.planner import key_solver

        self.assertLess(DEFAULT_PAIR_TOLERANCE_PCT, key_solver.NOISE_FLOOR_PCT)

    def test_unpriced_candidates_refuse_rather_than_rank(self):
        cp = [
            PhaseCandidate(
                "prefill",
                LayoutVector("p-10", (10, 1, 1)),
                Rate.absent("never run", unit="tok/s"),
            )
        ]
        cd = [
            PhaseCandidate(
                "decode",
                LayoutVector("d-even", (1, 1, 1)),
                Rate.measured(100.0, _SYNTH, unit="tok/s"),
            )
        ]
        with self.assertRaises(ValueError):
            solve_layout_pair(cp, cd, units=12, bytes_per_unit=1.0)

    def test_empty_candidate_list_refuses(self):
        _, cd = self._cands()
        with self.assertRaises(ValueError):
            solve_layout_pair([], cd, units=12, bytes_per_unit=1.0)


if __name__ == "__main__":
    unittest.main()
