"""#363 slice 1 — the regime controller's DECISION layer (DESIGN_363 §20).

What is under test: the machine that DECIDES whether a layout switch pays,
plans the pair that makes it cheap, and prices the residency rung it would
run at. Nothing here executes a switch, and the tests assert that too: the
verdict object carries ``executes: False``, because the pointer flip, the
diff-spill executor and the pre-capture are #363 slices 2+ (`ROADMAP_456`
WAVE 4).

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
shard. It appears twice — on a readable 12-unit toy grid in
:class:`TestOverlapMath`, and on Qwen3.6-27B's REAL 544-unit quant-group grid
in :meth:`TestPlanOutputWiring.test_real_27b_grid_reproduces_the_20_3_geometry`,
which is where the section's "~5/12 against ~4/12" is checked against the
partitioner rather than against a hand-built example.
"""

import unittest

from sglang.srt.planner.cost_model import Provenance, Rate
from sglang.srt.planner.regime_switch import (
    DEFAULT_PAIR_TOLERANCE_PCT,
    GIB,
    MIB,
    AutocheckResult,
    LayoutVector,
    PhaseCandidate,
    PhaseTable,
    Verdict,
    WorkloadShape,
    autocheck,
    layout_overlap,
    phase_table_from_json,
    price_switch,
    render_autocheck_text,
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


class TestAutocheckDecisionTable(CustomTestCase):
    """§20.1: the verdict, from the TABLE, with the reason and the numbers."""

    def test_int8_shaped_yields_no_switch_by_dominance(self):
        t = _table("INT8-shaped / synthetic", **INT8_SHAPED)
        r = autocheck(t, prefill_layout="prefill", decode_layout="decode")
        self.assertIs(r.verdict, Verdict.NO_SWITCH)
        self.assertEqual(r.numbers["dominant_layout"], "decode")
        # -2.3 %: the concentrating layout loses on its OWN phase.
        self.assertAlmostEqual(r.numbers["prefill_gain_pct"], -2.295, places=2)
        self.assertIn("one layout, checked", r.reason)
        self.assertFalse(r.acts)
        # A no-op verdict is STATED, never silence.
        self.assertTrue(r.reason.strip())
        self.assertEqual(r.missing, ())

    def test_without_a_ledger_the_cheap_rung_is_not_assumed(self):
        """The same divergence, no ledger: priced at RUNG 1 and refused, with
        the reason naming both numbers. Assuming RUNG 0 would be assuming the
        answer the caller did not supply the evidence for."""
        t = _table("FP8-shaped / synthetic", **FP8_SHAPED)
        r = autocheck(
            t, prefill_layout="prefill", decode_layout="decode", workload=FP8_ROUND
        )
        self.assertEqual(r.numbers["rung"], 1)
        self.assertIn("no residency ledger supplied", r.numbers["rung_note"])
        self.assertIs(r.verdict, Verdict.NO_SWITCH)
        self.assertIn("does not clear", r.reason)

    def test_amortised_switching_pays_where_per_round_switching_does_not(self):
        """A REGIME controller flips when the regime changes, not per request.
        The rate is a parameter, and it is the parameter that decides here."""
        t = _table("FP8-shaped / synthetic", **FP8_SHAPED)
        per_round = autocheck(
            t,
            prefill_layout="prefill",
            decode_layout="decode",
            workload=WorkloadShape(20000, 2048, switches_per_round=2.0),
        )
        amortised = autocheck(
            t,
            prefill_layout="prefill",
            decode_layout="decode",
            workload=WorkloadShape(20000, 2048, switches_per_round=0.02),
        )
        self.assertIs(per_round.verdict, Verdict.NO_SWITCH)
        self.assertIs(amortised.verdict, Verdict.SWITCH_FULL)

    def test_absent_cell_yields_unpriceable_not_a_guess(self):
        t = _table(
            "FP8-shaped, decode arm never run",
            **FP8_SHAPED,
            absent=(("prefill", "decode"),),
        )
        r = autocheck(t, prefill_layout="prefill", decode_layout="decode")
        self.assertIs(r.verdict, Verdict.UNPRICEABLE)
        self.assertIs(r.provenance, Provenance.ABSENT)
        self.assertEqual(len(r.missing), 1)
        self.assertIn("(prefill, decode)", r.missing[0])
        self.assertIn("UNPRICED", r.reason)
        self.assertFalse(r.acts)
        self.assertIsNone(r.switch_cost)

    def test_all_four_cells_absent_are_all_named(self):
        t = _table(
            "nothing measured",
            **FP8_SHAPED,
            absent=(
                ("prefill", "prefill"),
                ("prefill", "decode"),
                ("decode", "prefill"),
                ("decode", "decode"),
            ),
        )
        r = autocheck(t, prefill_layout="prefill", decode_layout="decode")
        self.assertIs(r.verdict, Verdict.UNPRICEABLE)
        self.assertEqual(len(r.missing), 4)

    def test_same_weight_vector_is_kv_only(self):
        """Two layouts that differ only in the #435-coupled KV vector."""
        a = LayoutVector("prefill", (2, 2, 2), kv_tokens=(2, 11, 10))
        b = LayoutVector("decode", (1, 1, 1), kv_tokens=(1, 1, 1))
        t = PhaseTable(
            triple="same weights, different KV vector",
            layouts=(a, b),
            cells={
                ("prefill", "prefill"): Rate.measured(1528.9, _SYNTH, unit="tok/s"),
                ("prefill", "decode"): Rate.measured(84.1, _SYNTH, unit="tok/s"),
                ("decode", "prefill"): Rate.measured(1231.7, _SYNTH, unit="tok/s"),
                ("decode", "decode"): Rate.measured(125.1, _SYNTH, unit="tok/s"),
            },
            noise_floor_pct=3.0,
            noise_floor_source="synthetic",
        )
        r = autocheck(
            t, prefill_layout="prefill", decode_layout="decode", workload=FP8_ROUND
        )
        self.assertIs(r.verdict, Verdict.SWITCH_KV_ONLY)
        self.assertTrue(r.numbers["kv_only"])
        # #297's actuator is the only mover, and its cost is MEASURED.
        self.assertEqual(list(r.switch_cost.components), ["kv_delta_move"])
        self.assertIs(r.switch_cost.provenance, Provenance.MEASURED)

    def test_divergence_below_the_floor_is_not_a_divergence(self):
        """Neither layout dominates, but the round-level benefit is under the
        table's own A-vs-A floor: NO_SWITCH, and the floor is the reason."""
        t = _table("marginal", p_on_p=1290.0, p_on_d=120.0, d_on_p=1231.7,
                   d_on_d=125.1, floor=3.0)
        r = autocheck(
            t,
            prefill_layout="prefill",
            decode_layout="decode",
            workload=FP8_ROUND,
        )
        self.assertIs(r.verdict, Verdict.NO_SWITCH)
        self.assertNotIn("dominant_layout", r.numbers)  # a genuine divergence
        self.assertLess(r.numbers["benefit_pct_of_round"], 3.0)
        self.assertIn("below the 3.0 % A-vs-A floor", r.reason)

    def test_render_states_a_no_op_verdict_too(self):
        t = _table("INT8-shaped / synthetic", **INT8_SHAPED)
        lines = render_autocheck_text(
            autocheck(t, prefill_layout="prefill", decode_layout="decode")
        )
        blob = "\n".join(lines)
        self.assertIn("NO_SWITCH", blob)
        self.assertIn("one layout, checked", blob)
        self.assertIn("nothing in this build executes", blob)


class TestAutocheckWithLedger(CustomTestCase):
    """The same autocheck, given the §20.3 residency ledger it prices at.

    Split from :class:`TestAutocheckDecisionTable` because these cases
    need the rung arithmetic; the ones there need only the phase table.
    """

    def test_fp8_shaped_yields_switch_full_at_rung0(self):
        t = _table("FP8-shaped / synthetic", **FP8_SHAPED)
        overlap, residency = _rung0_ledger()
        r = autocheck(
            t,
            prefill_layout="prefill",
            decode_layout="decode",
            workload=FP8_ROUND,
            overlap=overlap,
            residency=residency,
        )
        self.assertIs(r.verdict, Verdict.SWITCH_FULL)
        self.assertTrue(r.acts)
        self.assertAlmostEqual(r.numbers["prefill_gain_pct"], 24.13, places=1)
        self.assertAlmostEqual(r.numbers["decode_cost_pct"], -32.77, places=1)
        self.assertEqual(r.numbers["rung"], 0)
        self.assertAlmostEqual(r.numbers["benefit_s_per_round"], 3.157, places=2)
        self.assertAlmostEqual(r.numbers["switch_cost_s_per_round"], 2.0, places=6)
        self.assertGreater(r.numbers["margin_s_per_round"], 0.0)
        self.assertIn("real divergence", r.reason)

    def test_switch_cost_can_out_price_a_real_divergence(self):
        """Same FP8 divergence at the same RUNG 0, 200x more switches."""
        t = _table("FP8-shaped / synthetic", **FP8_SHAPED)
        overlap, residency = _rung0_ledger()

        def verdict(rate):
            return autocheck(
                t,
                prefill_layout="prefill",
                decode_layout="decode",
                workload=WorkloadShape(20000, 2048, switches_per_round=rate),
                overlap=overlap,
                residency=residency,
            )

        self.assertIs(verdict(2.0).verdict, Verdict.SWITCH_FULL)
        dear = verdict(400.0)
        self.assertIs(dear.verdict, Verdict.NO_SWITCH)
        self.assertIn("does not clear", dear.reason)

    def test_short_prompt_kills_a_prefill_layout_gain(self):
        """The workload shape is a real lever, not decoration."""
        t = _table("FP8-shaped / synthetic", **FP8_SHAPED)
        overlap, residency = _rung0_ledger()
        long_prompt = autocheck(
            t, prefill_layout="prefill", decode_layout="decode",
            workload=WorkloadShape(20000, 2048),
            overlap=overlap, residency=residency,
        )
        short_prompt = autocheck(
            t, prefill_layout="prefill", decode_layout="decode",
            workload=WorkloadShape(200, 2048),
            overlap=overlap, residency=residency,
        )
        self.assertIs(long_prompt.verdict, Verdict.SWITCH_FULL)
        self.assertIs(short_prompt.verdict, Verdict.NO_SWITCH)

    def test_verdict_object_says_it_does_not_execute(self):
        t = _table("FP8-shaped / synthetic", **FP8_SHAPED)
        overlap, residency = _rung0_ledger()
        js = autocheck(
            t, prefill_layout="prefill", decode_layout="decode",
            workload=FP8_ROUND, overlap=overlap, residency=residency,
        ).to_json()
        self.assertFalse(js["executes"])
        self.assertIn("decision", js["executes_note"].lower())
        self.assertEqual(js["verdict"], "SWITCH_FULL")


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


class TestRungSelectionUnderPressure(CustomTestCase):
    """Falsifier (b) part 2: the same FP8 divergence, rising KV pressure.

    Nothing about the phase table changes across these three points; only the
    committed KV bytes do. The rung the autocheck prices at must follow, and
    the switch cost with it.
    """

    def _verdict(self, kv_gib_on_big_card, pre_captured=True):
        overlap = layout_overlap(
            PREFILL, DECODE, units=12, bytes_per_unit=1.6 * GIB, active="decode"
        )
        residency = residency_rung(
            overlap,
            card_total_bytes=[48 * GIB, 40 * GIB, 40 * GIB],
            committed_bytes=[
                8 * GIB + kv_gib_on_big_card * GIB,
                12 * GIB,
                12 * GIB,
            ],
            graph_state_bytes=[3 * GIB, 3 * GIB, 3 * GIB],
            pre_captured=pre_captured,
        )
        return residency, autocheck(
            _table("FP8-shaped / synthetic", **FP8_SHAPED),
            prefill_layout="prefill",
            decode_layout="decode",
            workload=WorkloadShape(20000, 512),
            overlap=overlap,
            residency=residency,
        )

    def test_rung_flips_as_kv_pressure_grows(self):
        low, low_v = self._verdict(10)  # 8+10+3+9.6 = 30.6 of 48 GiB
        mid, mid_v = self._verdict(30)  # 8+30+3+9.6 = 50.6 -> no dual
        high, high_v = self._verdict(38)  # 8+38+3 = 49 -> not even that
        self.assertEqual([low.rung, mid.rung, high.rung], [0, 1, 2])

        # And the priced switch cost rises monotonically with the rung.
        costs = [v.numbers["switch_cost_s_per_round"] for v in (low_v, mid_v, high_v)]
        self.assertEqual(costs, sorted(costs))
        self.assertLess(costs[0], costs[-1])

        # RUNG 0 is a pointer flip plus the #297 KV delta and nothing else.
        self.assertEqual(
            sorted(low_v.switch_cost.components),
            ["kv_delta_move", "pointer_flip"],
        )
        # RUNG 1 reloads the diff and the graph state.
        self.assertIn("weight_diff_spill", mid_v.switch_cost.components)
        self.assertIn("graph_state_reload", mid_v.switch_cost.components)
        # RUNG 2 pays the un-amortised cold recapture.
        self.assertIn("lazy_recapture", high_v.switch_cost.components)

    def test_the_verdict_can_flip_with_the_rung(self):
        """Enough switches per round and RUNG 2's cost buries the gain while
        RUNG 0's does not — same table, same divergence, different rung."""
        overlap = layout_overlap(
            PREFILL, DECODE, units=12, bytes_per_unit=1.6 * GIB, active="decode"
        )
        table = _table("FP8-shaped / synthetic", **FP8_SHAPED)
        shape = WorkloadShape(20000, 2048, switches_per_round=2.0)

        def verdict(rung_ledger):
            return autocheck(
                table,
                prefill_layout="prefill",
                decode_layout="decode",
                workload=shape,
                overlap=overlap,
                residency=rung_ledger,
            ).verdict

        rung0 = residency_rung(
            overlap,
            card_total_bytes=[48 * GIB, 40 * GIB, 40 * GIB],
            committed_bytes=[18 * GIB, 12 * GIB, 12 * GIB],
            graph_state_bytes=[3 * GIB, 3 * GIB, 3 * GIB],
        )
        rung2 = residency_rung(
            overlap,
            card_total_bytes=[48 * GIB, 40 * GIB, 40 * GIB],
            committed_bytes=[18 * GIB, 12 * GIB, 12 * GIB],
            graph_state_bytes=[3 * GIB, 3 * GIB, 3 * GIB],
            pre_captured=False,
        )
        self.assertEqual(rung0.rung, 0)
        self.assertEqual(rung2.rung, 2)
        self.assertIs(verdict(rung0), Verdict.SWITCH_FULL)
        # The benefit here is 3.157 s/round; two RUNG 2 switches cost 18 s.
        self.assertIs(verdict(rung2), Verdict.NO_SWITCH)


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


class TestSwitchCostProvenance(CustomTestCase):
    """§20.3's measurement duty: the decomposition is reported separately and
    every component says whether it was measured."""

    def test_components_are_kept_apart(self):
        c0 = price_switch(0, kv_only=False)
        c1 = price_switch(1, kv_only=False)
        c2 = price_switch(2, kv_only=False)
        self.assertEqual(c0.seconds, sum(c0.components.values()))
        self.assertLess(c0.seconds, c1.seconds)
        self.assertLess(c1.seconds, c2.seconds)

    def test_only_the_kv_move_is_measured(self):
        # #297's actuator is reused as-is, so its target is inherited; every
        # other component is the §20.2 physics estimate or the #102 analogy.
        self.assertIs(price_switch(0, kv_only=True).provenance, Provenance.MEASURED)
        for rung in (0, 1, 2):
            self.assertIs(
                price_switch(rung, kv_only=False).provenance, Provenance.ESTIMATE
            )


class TestPhaseTableJson(CustomTestCase):
    """A number without provenance never enters the decision."""

    def _payload(self, cell_overrides=None):
        cell_overrides = cell_overrides or {}
        cells = [
            {"layout": "prefill", "phase": "prefill", "tok_s": 1528.9,
             "provenance": "measured", "source": _B424},
            {"layout": "prefill", "phase": "decode", "tok_s": 84.1,
             "provenance": "measured", "source": _B424},
            {"layout": "decode", "phase": "prefill", "tok_s": 1231.7,
             "provenance": "measured", "source": _B424},
            {"layout": "decode", "phase": "decode", "tok_s": 125.1,
             "provenance": "measured", "source": _B424},
        ]
        for c in cells:
            c.update(cell_overrides.get((c["layout"], c["phase"]), {}))
        return {
            "triple": "FP8-27B / Qwen3.6-27B / 5090+2x3080",
            "layouts": [
                {"name": "prefill", "weights": [10, 1, 1]},
                {"name": "decode", "weights": [1, 1, 1]},
            ],
            "cells": cells,
        }

    def test_round_trip(self):
        t = phase_table_from_json(self._payload())
        r = autocheck(
            t,
            prefill_layout="prefill",
            decode_layout="decode",
            workload=WorkloadShape(20000, 2048, switches_per_round=0.02),
        )
        self.assertIs(r.verdict, Verdict.SWITCH_FULL)
        self.assertIn("FP8-27B", r.triple)
        # No explicit floor -> the key solver's own measured floor, sourced.
        from sglang.srt.planner import key_solver

        self.assertEqual(r.numbers["noise_floor_pct"], key_solver.NOISE_FLOOR_PCT)
        self.assertEqual(
            r.numbers["noise_floor_source"], key_solver.NOISE_FLOOR_SOURCE
        )

    def test_a_cell_without_a_source_is_refused(self):
        with self.assertRaises(ValueError) as ctx:
            phase_table_from_json(
                self._payload({("prefill", "prefill"): {"source": ""}})
            )
        self.assertIn("must name its source", str(ctx.exception))

    def test_an_absent_cell_with_a_value_is_refused(self):
        with self.assertRaises(ValueError):
            phase_table_from_json(
                self._payload({("prefill", "prefill"): {"provenance": "absent"}})
            )

    def test_an_absent_cell_without_a_value_is_accepted_and_unpriceable(self):
        t = phase_table_from_json(
            self._payload(
                {("prefill", "prefill"): {"provenance": "absent", "tok_s": None}}
            )
        )
        r = autocheck(t, prefill_layout="prefill", decode_layout="decode")
        self.assertIs(r.verdict, Verdict.UNPRICEABLE)

    def test_an_unknown_provenance_is_refused(self):
        with self.assertRaises(ValueError):
            phase_table_from_json(
                self._payload({("prefill", "prefill"): {"provenance": "probably"}})
            )

    def test_a_cell_for_an_undeclared_layout_is_refused(self):
        payload = self._payload()
        payload["cells"].append(
            {"layout": "ghost", "phase": "decode", "tok_s": 1.0,
             "provenance": "measured", "source": _B424}
        )
        with self.assertRaises(ValueError):
            phase_table_from_json(payload)

    def test_a_one_layout_table_is_refused(self):
        payload = self._payload()
        payload["layouts"] = payload["layouts"][:1]
        payload["cells"] = [c for c in payload["cells"] if c["layout"] == "prefill"]
        with self.assertRaises(ValueError):
            phase_table_from_json(payload)


class TestTypeInvariants(CustomTestCase):
    def test_layout_rejects_a_non_positive_weight(self):
        with self.assertRaises(ValueError):
            LayoutVector("bad", (10, 0, 1))

    def test_layout_rejects_a_mismatched_kv_vector(self):
        with self.assertRaises(ValueError):
            LayoutVector("bad", (10, 1, 1), kv_tokens=(1, 1))

    def test_same_weights_compares_the_reduced_vector(self):
        self.assertTrue(
            LayoutVector("a", (2, 2, 2)).same_weights_as(LayoutVector("b", (1, 1, 1)))
        )
        self.assertFalse(
            LayoutVector("a", (2, 1, 1)).same_weights_as(LayoutVector("b", (1, 1, 1)))
        )

    def test_phase_table_rejects_mixed_tp_sizes(self):
        with self.assertRaises(ValueError):
            PhaseTable(
                triple="mixed",
                layouts=(LayoutVector("a", (1, 1, 1)), LayoutVector("b", (1, 1))),
                cells={},
            )

    def test_phase_table_rejects_duplicate_layout_names(self):
        with self.assertRaises(ValueError):
            PhaseTable(
                triple="dup",
                layouts=(LayoutVector("a", (1, 1, 1)), LayoutVector("a", (2, 1, 1))),
                cells={},
            )

    def test_workload_rejects_negative_tokens(self):
        with self.assertRaises(ValueError):
            WorkloadShape(prefill_tokens=-1)

    def test_missing_cell_reads_as_absent_not_as_a_crash(self):
        t = PhaseTable(
            triple="sparse",
            layouts=(PREFILL, DECODE),
            cells={},
        )
        cell = t.cell("prefill", "decode")
        self.assertIs(cell.provenance, Provenance.ABSENT)
        self.assertIn("never run", cell.source)
        self.assertIsInstance(
            autocheck(t, prefill_layout="prefill", decode_layout="decode"),
            AutocheckResult,
        )

    def test_mib_and_gib_are_binary(self):
        self.assertEqual(MIB, 1024**2)
        self.assertEqual(GIB, 1024**3)


class TestSolverApiSurface(CustomTestCase):
    """``POST /api/regime_switch`` — the planner API surface."""

    def _table_payload(self):
        return {
            "triple": "FP8-27B / Qwen3.6-27B / 5090+2x3080",
            "layouts": [
                {"name": "prefill", "weights": [10, 1, 1]},
                {"name": "decode", "weights": [1, 1, 1]},
            ],
            "cells": [
                {"layout": "prefill", "phase": "prefill", "tok_s": 1528.9,
                 "provenance": "measured", "source": _B424},
                {"layout": "prefill", "phase": "decode", "tok_s": 84.1,
                 "provenance": "measured", "source": _B424},
                {"layout": "decode", "phase": "prefill", "tok_s": 1231.7,
                 "provenance": "measured", "source": _B424},
                {"layout": "decode", "phase": "decode", "tok_s": 125.1,
                 "provenance": "measured", "source": _B424},
            ],
        }

    def test_no_table_is_refused_with_a_reason(self):
        from sglang.srt.planner.solver_api import regime_switch_payload

        out = regime_switch_payload({})
        self.assertFalse(out["ok"])
        self.assertIn("no phase_table given", out["reasons"][0])

    def test_full_payload_with_geometry_and_ledger(self):
        from sglang.srt.planner.solver_api import regime_switch_payload

        out = regime_switch_payload(
            {
                "phase_table": self._table_payload(),
                "workload": {"prefill_tokens": 20000, "decode_tokens": 2048},
                "geometry": {"units": 12, "bytes_per_unit": 1.6 * GIB},
                "ledger": {
                    "card_total_bytes": [48 * GIB, 40 * GIB, 40 * GIB],
                    "committed_bytes": [18 * GIB, 12 * GIB, 12 * GIB],
                    "graph_state_bytes": [3 * GIB, 3 * GIB, 3 * GIB],
                },
            }
        )
        self.assertTrue(out["ok"])
        self.assertEqual(out["verdict"], "SWITCH_FULL")
        self.assertEqual(out["residency"]["rung"], 0)
        self.assertFalse(out["executes"])
        self.assertEqual(out["overlap"]["per_rank"][0]["extra_units_vs_larger"], 0)

    def test_a_ledger_without_a_geometry_is_refused(self):
        from sglang.srt.planner.solver_api import regime_switch_payload

        out = regime_switch_payload(
            {
                "phase_table": self._table_payload(),
                "ledger": {
                    "card_total_bytes": [1, 2, 3],
                    "committed_bytes": [1, 2, 3],
                },
            }
        )
        self.assertFalse(out["ok"])
        self.assertIn("without a geometry", out["reasons"][0])

    def test_pair_mode_rides_along(self):
        from sglang.srt.planner.solver_api import regime_switch_payload

        out = regime_switch_payload(
            {
                "phase_table": self._table_payload(),
                "geometry": {"units": 12, "bytes_per_unit": 1.0},
                "pair_mode": {
                    "tolerance_pct": 2.0,
                    "candidates_prefill": [
                        {"layout": {"name": "p-10", "weights": [10, 1, 1]},
                         "tok_s": 1000.0, "provenance": "measured",
                         "source": _SYNTH},
                        {"layout": {"name": "p-8", "weights": [8, 2, 2]},
                         "tok_s": 990.0, "provenance": "measured",
                         "source": _SYNTH},
                    ],
                    "candidates_decode": [
                        {"layout": {"name": "d-even", "weights": [1, 1, 1]},
                         "tok_s": 100.0, "provenance": "measured",
                         "source": _SYNTH},
                    ],
                },
            }
        )
        self.assertTrue(out["ok"])
        self.assertLessEqual(out["pair"]["max_concession_pct"], 2.0)
        self.assertIn("overlap", out["pair"])

    def test_an_unsourced_pair_candidate_is_refused(self):
        from sglang.srt.planner.solver_api import regime_switch_payload

        out = regime_switch_payload(
            {
                "phase_table": self._table_payload(),
                "geometry": {"units": 12, "bytes_per_unit": 1.0},
                "pair_mode": {
                    "candidates_prefill": [
                        {"layout": {"name": "p", "weights": [10, 1, 1]},
                         "tok_s": 1.0, "provenance": "measured", "source": ""}
                    ],
                    "candidates_decode": [
                        {"layout": {"name": "d", "weights": [1, 1, 1]},
                         "tok_s": 1.0, "provenance": "measured",
                         "source": _SYNTH}
                    ],
                },
            }
        )
        self.assertFalse(out["ok"])
        self.assertIn("no source", out["reasons"][0])

    def test_an_unknown_layout_name_is_refused(self):
        from sglang.srt.planner.solver_api import regime_switch_payload

        out = regime_switch_payload(
            {"phase_table": self._table_payload(), "decode_layout": "ghost"}
        )
        self.assertFalse(out["ok"])
        self.assertIn("ghost", out["reasons"][0])


class TestPlanOutputWiring(CustomTestCase):
    """The verdict in the PLAN's own output, on the plan's own ledger.

    Same synthetic Qwen3.6-27B-shaped checkpoint the planner core tests use:
    no GPU, no NVML, no weight load.
    """

    @classmethod
    def setUpClass(cls):
        import json
        import os
        import tempfile

        from sglang.srt.planner.hardware import hardware_from_manual

        cls._tmp = tempfile.TemporaryDirectory()
        path = os.path.join(cls._tmp.name, "model")
        os.makedirs(path, exist_ok=True)
        config = {
            "architectures": ["Qwen3NextForCausalLM"],
            "hidden_size": 5120,
            "intermediate_size": 17408,
            "num_hidden_layers": 48,
            "num_attention_heads": 24,
            "num_key_value_heads": 4,
            "head_dim": 256,
            "vocab_size": 151936,
            "linear_num_key_heads": 16,
            "linear_num_value_heads": 32,
            "linear_key_head_dim": 128,
            "linear_value_head_dim": 128,
            "linear_conv_kernel_dim": 4,
            "layer_types": (["linear_attention"] * 3 + ["full_attention"]) * 12,
            "quantization_config": {"group_size": 32},
        }
        with open(os.path.join(path, "config.json"), "w") as f:
            json.dump(config, f)
        with open(os.path.join(path, "model-00001.safetensors"), "wb") as f:
            f.truncate(int(14.0 * 2**30))
        cls.model = path
        cls.hw = hardware_from_manual(
            ("RTX 5090:32607", "RTX 3080:20480", "RTX 3080:20480")
        )

    @classmethod
    def tearDownClass(cls):
        cls._tmp.cleanup()

    def _table_payload(self, **over):
        cells = [
            {"layout": "prefill", "phase": "prefill", "tok_s": 1528.9,
             "provenance": "measured", "source": _B424},
            {"layout": "prefill", "phase": "decode", "tok_s": 84.1,
             "provenance": "measured", "source": _B424},
            {"layout": "decode", "phase": "prefill", "tok_s": 1231.7,
             "provenance": "measured", "source": _B424},
            {"layout": "decode", "phase": "decode", "tok_s": 125.1,
             "provenance": "measured", "source": _B424},
        ]
        out = {
            "triple": "FP8-27B / Qwen3.6-27B / 5090+2x3080",
            "layouts": [
                {"name": "prefill", "weights": [10, 1, 1]},
                {"name": "decode", "weights": [1, 1, 1]},
            ],
            "cells": cells,
        }
        out.update(over)
        return out

    def test_default_plan_carries_no_verdict(self):
        """Opt-in: an existing caller's answer is unchanged."""
        from sglang.srt.planner.feasibility import plan

        result = plan(self.model, self.hw, tp_size=3)
        self.assertIsNone(result.regime)

    def test_plan_computes_the_verdict_on_its_own_ledger(self):
        from sglang.srt.planner.feasibility import plan
        from sglang.srt.planner.regime_switch import AutocheckResult

        result = plan(
            self.model,
            self.hw,
            tp_size=3,
            regime_phase_table=self._table_payload(),
        )
        self.assertIsInstance(result.regime, AutocheckResult)
        # The rung came out of the plan's OWN capacity report, not a constant.
        self.assertIsNotNone(result.regime.residency)
        self.assertEqual(len(result.regime.residency.per_rank), 3)
        self.assertIn(result.regime.residency.rung, (0, 1, 2))
        # The overlap used the checkpoint's real MLP unit grid.
        self.assertGreaterEqual(result.regime.overlap.units, 3)
        js = result.regime.to_json()
        self.assertFalse(js["executes"])

    def test_real_27b_grid_reproduces_the_20_3_geometry(self):
        """§20.3's example on the checkpoint's OWN grid, not a 12-unit toy.

        Qwen3.6-27B's FFN intermediate is 17408 elements = 544 quant-group
        units, so the ``10,1,1`` prefill vector partitions to 453/46/45 and
        the even decode split to 182/181/181. Three facts §20.3 asserts, all
        reproduced here:

          * the big card's decode shard [0,182) is a strict PREFIX of its
            prefill shard [0,453) -> zero extra against the larger layout;
          * the middle card is disjoint, union 227/544 = 0.4173 against
            the decode layout's 181/544 = 0.3327 -- the section's "union ~5/12
            (0.4167) against ~4/12 (0.3333)";
          * the THIRD card is NOT symmetric with the second: its prefill
            range [499,544) is a SUFFIX nested inside its decode range
            [363,544), so it holds no extra bytes either. §20.3's "the two
            smaller cards hold disjoint ranges ... each" over-counts by one
            card.
        """
        from sglang.srt.planner.feasibility import plan

        result = plan(
            self.model,
            self.hw,
            tp_size=3,
            regime_phase_table=self._table_payload(),
        )
        ov = result.regime.overlap
        self.assertEqual(ov.units, 544)
        self.assertEqual(
            [r.a_range for r in ov.per_rank], [(0, 453), (453, 499), (499, 544)]
        )
        self.assertEqual(
            [r.b_range for r in ov.per_rank], [(0, 182), (182, 363), (363, 544)]
        )
        self.assertEqual([r.nested for r in ov.per_rank], [True, False, True])
        self.assertEqual([r.extra_units_vs_larger for r in ov.per_rank], [0, 46, 0])
        self.assertEqual([r.extra_units_vs_active for r in ov.per_rank], [271, 46, 0])
        self.assertAlmostEqual(ov.per_rank[1].union_units / 544, 5 / 12, places=2)
        self.assertAlmostEqual(
            (ov.per_rank[1].b_range[1] - ov.per_rank[1].b_range[0]) / 544,
            4 / 12,
            places=2,
        )
        # The two baselines differ by ~7x on this geometry (317 vs 46 units),
        # which is why both are reported and neither is called "the" cost.
        self.assertAlmostEqual(
            ov.extra_bytes_vs_active / ov.extra_bytes_vs_larger, 317 / 46, places=6
        )

    def test_a_malformed_table_is_named_not_swallowed(self):
        from sglang.srt.planner.feasibility import plan

        bad = self._table_payload()
        bad["cells"][0]["source"] = ""
        with self.assertRaises(ValueError):
            plan(self.model, self.hw, tp_size=3, regime_phase_table=bad)

    def test_not_pre_captured_pins_rung_two(self):
        from sglang.srt.planner.feasibility import plan

        result = plan(
            self.model,
            self.hw,
            tp_size=3,
            regime_phase_table=self._table_payload(),
            regime_pre_captured=False,
        )
        self.assertEqual(result.regime.residency.rung, 2)

    def test_cli_prints_the_verdict(self):
        import contextlib
        import io

        from sglang.srt.planner.cli import _print_regime
        from sglang.srt.planner.feasibility import plan

        result = plan(
            self.model,
            self.hw,
            tp_size=3,
            regime_phase_table=self._table_payload(),
        )
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            _print_regime(result)
        text = buf.getvalue()
        self.assertIn("REGIME AUTOCHECK (#363 §20.1)", text)
        self.assertIn("verdict :", text)
        self.assertIn("nothing in this build executes", text)

    def test_cli_prints_nothing_without_a_table(self):
        import contextlib
        import io

        from sglang.srt.planner.cli import _print_regime
        from sglang.srt.planner.feasibility import plan

        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            _print_regime(plan(self.model, self.hw, tp_size=3))
        self.assertEqual(buf.getvalue(), "")


if __name__ == "__main__":
    unittest.main()
