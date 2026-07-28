"""Guided wizard v2 (#270 backlog) -- the four blocks and the lane structure.

What is pinned here is not "the code runs". It is the four properties that
make these blocks worth showing at all, and each one is a property that would
be easy to lose in a later edit without anybody noticing.

**A number that is not a measurement of this rig says so, and says what would
replace it.** Tipping points, link rates, offload depths and island families
all go through the same three words, and the "measure it now" action exists
exactly where the number is not measured.

**An intra-rig figure never fills a cross-rig cell.** The card-to-card matrix
prices a path a network handover does not cross. This is the substitution
that would be easiest to make and hardest to see, so it has its own test.

**The offload axis computes what a depth frees and refuses to state what it
costs.** The buy side is arithmetic over the plan's own pool; the price side
for this model has never been measured, and the three reference boots must
not leak into it as a value.

**Islands are estimated, never refused.** We own no NVLink, and the rule is
that this rig is a lower bound rather than a verdict about hardware other
people have. So the families exist with modelled figures and an origin that
says they were modelled -- and the model is the discount ladder that is
already in the tree, not a constant invented here.

Plus the structural requirement of DESIGN_201 PRIO-Nachtrag 8: the lane model
is a LIST from the first line. A test that a one-lane configuration is one
entry in a list rather than a special case is the only thing that keeps the
next author from adding ``lane_a``/``lane_b``.
"""

import json
import os
import tempfile

from sglang.srt.planner import webui
from sglang.srt.planner import wizard as wz
from sglang.srt.planner import wizard_islands as isl
from sglang.srt.planner import wizard_lanes as lanes
from sglang.srt.planner import wizard_links as links
from sglang.srt.planner import wizard_offload as off
from sglang.srt.planner import wizard_tipping as tip
from sglang.srt.planner.bench_factors import ABSENT, ESTIMATE, MEASURED
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=30, suite="base-a-test-cpu")


# ---------------------------------------------------------------------------
# 1 -- tipping points: origin, and the button that follows from it
# ---------------------------------------------------------------------------


def _sources(**kw) -> tip.TippingSources:
    base = dict(
        model_path="/models/Qwen3.6-27B-FP8",
        tp_size=3,
        gpus=[
            {"index": 0, "name": "RTX 5090", "uuid": "GPU-a"},
            {"index": 1, "name": "RTX 3080", "uuid": "GPU-b"},
        ],
        split_table={
            "ladder": ["auto", "6,1,1"],
            "baseline": "auto",
            "rows": [
                {"candidate": "auto", "measured": True, "is_baseline": True,
                 "decode_tok_s": 41.9, "prefill_tok_s": 1240.0,
                 "provenance": "measured", "source": "boot 2026-07-28"},
                {"candidate": "6,1,1", "measured": False,
                 "missing_reason": "not measured on this rig."},
            ],
        },
        prefill_tok_s=1240.0,
        loaded_fraction=0.427,
        context_tokens=8192,
        handshake_s=0.136,
        anchor_study="the #212 study",
    )
    base.update(kw)
    return tip.TippingSources(**base)


class TippingPoints(CustomTestCase):
    def test_every_point_says_what_it_tips_and_where_it_is_from(self):
        out = tip.build_tipping_points(_sources())
        self.assertTrue(out["points"])
        for p in out["points"]:
            self.assertTrue(p["tips"], p["key"])
            self.assertTrue(p["question"], p["key"])
            self.assertIn(p["value"]["provenance"], (MEASURED, ESTIMATE, ABSENT))
            # The origin is a citation, not a mood: a source AND a detail.
            self.assertTrue(p["origin"]["source"] or p["origin"]["detail"], p["key"])
            self.assertTrue(p["value"]["basis"], p["key"])

    def test_absent_cells_carry_no_value(self):
        out = tip.build_tipping_points(_sources(prefill_tok_s=None))
        for p in out["points"]:
            if p["value"]["provenance"] == ABSENT:
                self.assertIsNone(p["value"]["value"], p["key"])
                self.assertFalse(p["value"]["available"], p["key"])

    def test_unmeasured_thresholds_offer_the_study(self):
        """Absent-is-always-measurable: a threshold that is not measured here
        carries a runnable action, or a stated reason why not."""
        out = tip.build_tipping_points(_sources())
        for p in out["points"]:
            if p["value"]["provenance"] == MEASURED:
                continue
            m = p["measure"]
            self.assertTrue(
                m.get("offered") or m.get("reason"),
                f"{p['key']}: not measured and no way offered and no reason",
            )

    def test_the_split_button_targets_the_existing_probe_endpoint(self):
        out = tip.build_tipping_points(_sources())
        p = [x for x in out["points"] if x["key"] == "mlp_split_crossover"][0]
        a = p["measure"]["action"]
        self.assertEqual(a["path"], "/api/split_probe")
        self.assertEqual(a["status_path"], "/api/split_probe/status")
        # It measures the first ladder candidate with no row yet, so pressing
        # it twice measures two different things.
        self.assertEqual(p["measure"]["body"]["mlp_vector"], "6,1,1")
        self.assertEqual(p["measure"]["body"]["model_path"], "/models/Qwen3.6-27B-FP8")

    def test_the_preview_names_duration_cards_and_interruption(self):
        out = tip.build_tipping_points(
            _sources(busy_cards={"GPU-b": "4711 4712"})
        )
        p = [x for x in out["points"] if x["key"] == "mlp_split_crossover"][0]
        pv = p["measure"]["preview"]
        self.assertEqual(pv["duration_minutes"], [6, 8])
        self.assertTrue(pv["interruption"])
        self.assertEqual(len(pv["cards"]), 2)
        # A busy card blocks the action and names who holds it, rather than
        # letting the probe start and fail.
        self.assertTrue(pv["blocked"])
        self.assertIn("4711", pv["blocked_reason"])

    def test_a_card_without_a_uuid_is_unknown_not_free(self):
        out = tip.build_tipping_points(
            _sources(gpus=[{"index": 0, "name": "RTX 5090"}])
        )
        p = [x for x in out["points"] if x["key"] == "mlp_split_crossover"][0]
        self.assertIsNone(p["measure"]["preview"]["cards"][0]["busy"])

    def test_measured_ladder_rows_are_not_offered_a_re_measure_by_default(self):
        out = tip.build_tipping_points(_sources())
        p = [x for x in out["points"] if x["key"] == "mlp_split_crossover"][0]
        rows = {r["candidate"]: r for r in p["candidates"]}
        self.assertEqual(rows["auto"]["provenance"], MEASURED)
        self.assertFalse(rows["auto"]["measure"]["offered"])
        # ...but the action is still there, so refreshing a stale row is one
        # click rather than a hunt for the page that owns the study.
        self.assertTrue(rows["auto"]["measure"]["available"])
        self.assertEqual(rows["6,1,1"]["provenance"], ABSENT)
        self.assertTrue(rows["6,1,1"]["measure"]["offered"])

    def test_satellite_threshold_is_the_loaded_rate_not_the_idle_one(self):
        """The whole point of the pair: an idle serving card is not the case
        anybody moves prefill away from."""
        out = tip.build_tipping_points(_sources(kv_bytes_per_token=None))
        p = [x for x in out["points"] if x["key"] == "satellite_break_even"][0]
        self.assertEqual(p["value"]["provenance"], ESTIMATE)
        ctx, loaded = 8192, 1240.0 * 0.427
        expect = ctx / (ctx / loaded - 0.136)
        self.assertAlmostEqual(p["value"]["value"], expect, places=3)

    def test_the_side_is_stated_only_when_it_is_known(self):
        known = tip.build_tipping_points(
            _sources(satellite_prefill_tok_s=99999.0, kv_bytes_per_token=None)
        )
        p = [x for x in known["points"] if x["key"] == "satellite_break_even"][0]
        self.assertTrue(p["side"]["available"])
        self.assertEqual(p["side"]["which"], "above")

        unknown = tip.build_tipping_points(_sources(kv_bytes_per_token=None))
        p = [x for x in unknown["points"] if x["key"] == "satellite_break_even"][0]
        self.assertFalse(p["side"]["available"])

    def test_coverage_counts_the_ladder_too(self):
        out = tip.build_tipping_points(_sources())
        c = out["coverage"]
        self.assertEqual(
            c["total"], c["measured"] + c["estimate"] + c["absent"]
        )
        self.assertGreater(c["total"], len(out["points"]))


# ---------------------------------------------------------------------------
# 2 -- link and satellite rates: read paths, not constants
# ---------------------------------------------------------------------------


_ANCHORS = dict(wz.ANCHORS)


def _link_sources(**kw) -> links.LinkSources:
    base = dict(
        card_probe_pairs=[
            {"src_uuid": "A", "dst_uuid": "B", "bandwidth_gbs": 6.88,
             "transport": "host staging (pinned)", "peer_access": False},
            {"src_uuid": "B", "dst_uuid": "A", "bandwidth_gbs": 4.52,
             "transport": "host staging (pinned)", "peer_access": False},
        ],
        card_names={"A": "RTX 5090 #0", "B": "RTX 3080 #2"},
        card_probe_created=1785110400.0,
        anchors=_ANCHORS,
        anchor_study=wz.ANCHOR_STUDY,
    )
    base.update(kw)
    return links.LinkSources(**base)


class LinkRates(CustomTestCase):
    def test_intra_rig_narrowest_comes_off_the_measured_matrix(self):
        r = links.read_rates(_link_sources())["rates"]["intra_rig_narrowest_gbs"]
        self.assertEqual(r["provenance"], MEASURED)
        self.assertAlmostEqual(r["value"], 4.52)
        self.assertIn("RTX 3080 #2", r["basis"])

    def test_an_intra_rig_figure_never_fills_the_cross_rig_cell(self):
        """The substitution that would be easiest to make and hardest to see.

        A rig with a fully measured card matrix and no wire to anywhere must
        report the wire as absent -- and must say that the matrix was looked
        at and deliberately not used.
        """
        out = links.read_rates(_link_sources(remote_targets=["rig2:31000"]))
        cross = out["rates"]["cross_rig_gbs"]
        self.assertEqual(cross["provenance"], ABSENT)
        self.assertIsNone(cross["value"])
        self.assertTrue(cross["way"])
        rungs = {c["rung"]: c["verdict"] for c in cross["considered"]}
        self.assertIn("card probe pair matrix", rungs)
        self.assertIn("not consulted", rungs["card probe pair matrix"])

    def test_the_comm_suite_artifact_fills_the_cross_rig_cell(self):
        out = links.read_rates(
            _link_sources(
                artifact_measurements=[
                    {"id": "comm/cross_rig/all_reduce/256KiB/rate",
                     "label": "Cross-rig link all_reduce/256KiB rate",
                     "unit": "Gbit/s", "value": 15.44, "taken_at": "2026-07-28"},
                    {"id": "comm/collective_nccl/all_reduce/20KiB/rate",
                     "unit": "Gbit/s", "value": 900.0, "taken_at": "2026-07-28"},
                ]
            )
        )
        cross = out["rates"]["cross_rig_gbs"]
        self.assertEqual(cross["provenance"], MEASURED)
        self.assertAlmostEqual(cross["value"], 15.44 / 8.0)
        # An intra-rig NCCL row must not be mistaken for the wire.
        self.assertIn("cross_rig", cross["source"])

    def test_the_form_wins_over_everything(self):
        out = links.read_rates(
            _link_sources(form_link_gbs=1.93, form_link_source="iperf3, my rack")
        )
        self.assertAlmostEqual(out["rates"]["cross_rig_gbs"]["value"], 1.93)

    def test_reference_anchors_are_estimates_here_not_measurements(self):
        """They were measured -- on other hardware and another checkpoint.
        Labelling them ``measured`` would make this rig claim a study it never
        ran."""
        out = links.read_rates(_link_sources())["rates"]
        for key in ("loaded_prefill_fraction", "pd_handshake_s",
                    "decode_spike_ms", "decode_spike_offloaded_ms"):
            self.assertEqual(out[key]["provenance"], ESTIMATE, key)
            self.assertEqual(out[key]["study"], wz.ANCHOR_STUDY, key)

    def test_the_satellite_rate_is_absent_rather_than_borrowed(self):
        r = links.read_rates(_link_sources())["rates"]["satellite_prefill_tok_s"]
        self.assertEqual(r["provenance"], ABSENT)
        self.assertIsNone(r["value"])
        # The reference figure appears as a rung that was considered, never
        # as the answer.
        self.assertTrue(
            any("reference anchor" in c["rung"] for c in r["considered"])
        )

    def test_the_matrix_prices_with_the_resolved_rates(self):
        """The rate report is the authority, not the module constants."""
        ctx = wz.MatrixContext(
            rates={"rates": {"pd_handshake_s": {"value": 9.0}}}
        )
        self.assertEqual(ctx.rate("pd_handshake_s", 0.136), 9.0)
        # ...and with no report the constant is the last rung, unchanged.
        self.assertEqual(wz.MatrixContext().rate("pd_handshake_s", 0.136), 0.136)


# ---------------------------------------------------------------------------
# 3 -- offload depth
# ---------------------------------------------------------------------------


class OffloadDepth(CustomTestCase):
    def _src(self, **kw) -> off.OffloadSources:
        base = dict(
            is_moe=True, tp_size=3, offloadable_mib=20000.0,
            fits_without_offload=False, shortfall_mib=6000.0,
            kv_bytes_per_token=98304.0, host_ram_free_mib=100000.0,
        )
        base.update(kw)
        return off.OffloadSources(**base)

    def test_dense_checkpoints_have_no_depth_axis(self):
        out = off.offload_dimension(off.OffloadSources(is_moe=False))
        self.assertFalse(out["applies"])
        self.assertIn("no routed experts", out["reason"])

    def test_every_step_carries_gain_price_and_who_it_is_for(self):
        out = off.offload_dimension(self._src())
        self.assertEqual(len(out["steps"]), len(off.OFFLOAD_STEPS))
        for s in out["steps"]:
            self.assertTrue(s["tradeoff"]["gain"], s["fraction"])
            self.assertTrue(s["tradeoff"]["price"], s["fraction"])
            self.assertTrue(s["tradeoff"]["worth_for"], s["fraction"])

    def test_freed_vram_is_derived_from_the_plans_own_pool(self):
        out = off.offload_dimension(self._src())
        by_frac = {s["fraction"]: s for s in out["steps"]}
        self.assertEqual(by_frac[1.0]["vram_freed_mib"]["value"], 0.0)
        self.assertEqual(by_frac[0.5]["vram_freed_mib"]["value"], 10000.0)
        self.assertEqual(by_frac[0.0]["vram_freed_mib"]["value"], 20000.0)
        self.assertEqual(by_frac[0.5]["vram_freed_mib"]["provenance"], ESTIMATE)
        # The per-rank ceil rounding is DISCLOSED rather than folded in.
        self.assertIn("ceil", by_frac[0.5]["vram_freed_mib"]["basis"])

    def test_the_decode_price_is_absent_for_this_model_at_every_depth(self):
        """The honest half. Three boots of other models exist; none of them is
        this model's decode rate, and none may become one."""
        out = off.offload_dimension(self._src())
        for s in out["steps"]:
            self.assertEqual(s["decode_price"]["provenance"], ABSENT)
            self.assertIsNone(s["decode_price"]["value"])
        self.assertIn("not computed and not borrowed",
                      out["counter_reckoning"]["price_side"])

    def test_the_advice_changes_when_the_plan_already_fits(self):
        needed = off.offload_dimension(self._src())
        elective = off.offload_dimension(
            self._src(fits_without_offload=True, shortfall_mib=None)
        )
        by = {s["fraction"]: s for s in needed["steps"]}
        self.assertIn("closes the", by[0.5]["tradeoff"]["worth_for"])
        by2 = {s["fraction"]: s for s in elective["steps"]}
        self.assertIn("elective", by2[0.5]["tradeoff"]["worth_for"])

    def test_a_step_too_shallow_to_close_the_gap_says_so(self):
        out = off.offload_dimension(self._src(shortfall_mib=6000.0))
        by = {s["fraction"]: s for s in out["steps"]}
        self.assertIn("not enough on its own", by[0.75]["tradeoff"]["worth_for"])

    def test_a_host_pool_that_does_not_fit_is_a_wall_not_a_footnote(self):
        out = off.offload_dimension(self._src(host_ram_free_mib=8000.0))
        by = {s["fraction"]: s for s in out["steps"]}
        self.assertTrue(
            any("pinned" in n for n in by[0.0]["notes"]), by[0.0]["notes"]
        )

    def test_every_evidence_row_states_what_it_does_not_say(self):
        for e in off.EVIDENCE:
            self.assertTrue(e.showed)
            self.assertTrue(e.does_not_transfer)

    def test_a_blocked_checkpoint_kind_reports_the_register_verdict(self):
        out = off.offload_dimension(
            self._src(is_gguf=True, blocked_verdict="GGUF MoE has no offload path",
                      blocked_evidence="#123 finding")
        )
        self.assertTrue(out["applies"])
        self.assertFalse(out["available"])
        self.assertEqual(out["steps"], [])
        self.assertIn("#123", out["source"])


# ---------------------------------------------------------------------------
# 4 -- island families
# ---------------------------------------------------------------------------


class IslandFamilies(CustomTestCase):
    def test_this_rig_has_no_islands_and_says_why(self):
        cards = [{"uuid": u} for u in ("A", "B", "C")]
        pairs = [
            {"src_uuid": "A", "dst_uuid": "B", "bandwidth_gbs": 6.0,
             "transport": "host staging (pinned)", "peer_access": False},
        ]
        topo = isl.islands_from_pairs(cards, pairs)
        self.assertEqual(topo.island_count, 3)
        self.assertFalse(topo.has_islands)
        out = isl.island_families(topo)
        self.assertFalse(out["applies"])
        self.assertIn("no pair has peer access", out["reason"])
        # Not a dead end: it says how to look at hardware that does have them.
        self.assertTrue(out["how_to_explore"])

    def test_fast_edges_group_transitively(self):
        cards = [{"uuid": u} for u in ("A", "B", "C", "D")]
        pairs = [
            {"src_uuid": "A", "dst_uuid": "B", "peer_access": True,
             "bandwidth_gbs": 40.0, "transport": "cuda p2p"},
            {"src_uuid": "B", "dst_uuid": "C", "peer_access": True,
             "bandwidth_gbs": 40.0, "transport": "cuda p2p"},
            {"src_uuid": "C", "dst_uuid": "D", "peer_access": False,
             "bandwidth_gbs": 6.0, "transport": "host staging (pinned)"},
        ]
        topo = isl.islands_from_pairs(cards, pairs)
        self.assertEqual([len(i) for i in topo.islands], [3, 1])
        self.assertTrue(topo.has_islands)

    def test_described_hardware_is_estimated_never_refused(self):
        topo = isl.described_topology([4, 4], intra_tier="nvlink")
        out = isl.island_families(topo, model_fits_in_island=True)
        self.assertTrue(out["applies"])
        self.assertTrue(out["families"])
        for f in out["families"]:
            o = f["origin"]
            self.assertEqual(o["provenance"], ESTIMATE)
            self.assertIn("roofline", o["source"])
            self.assertTrue(o["caveat"])
            # The rule that makes this legitimate is stated with the number.
            self.assertIn("lower bound", o["rule"])

    def test_the_advantage_is_a_ratio_of_two_rungs_of_the_existing_ladder(self):
        """Not a predicted rate, and not a constant invented here."""
        topo = isl.described_topology([4, 4], intra_tier="nvlink")
        out = isl.island_families(topo, model_fits_in_island=True)
        local = [f for f in out["families"] if f["key"] == "island_local_tp"][0]
        expected = isl.collective_discount("nvlink", 4) / isl.collective_discount(
            "pcie-host-staging", 8
        )
        self.assertAlmostEqual(local["collective_advantage"]["value"], expected)
        self.assertEqual(local["collective_advantage"]["provenance"], ESTIMATE)

    def test_the_ladder_is_read_from_roofline_not_copied(self):
        from sglang.srt.planner import roofline

        self.assertEqual(
            isl.collective_discount("nvlink", 2), roofline._NVLINK_DISCOUNT
        )
        self.assertEqual(
            isl.collective_discount("pcie-p2p", 2), roofline._PCIE_P2P_DISCOUNT
        )
        self.assertEqual(isl.collective_discount("pcie-host-staging", 9),
                         roofline._PCIE_NOP2P_MANY)
        self.assertEqual(isl.collective_discount("nvlink", 1), 1.0)

    def test_a_family_that_does_not_fit_one_island_is_marked_out(self):
        topo = isl.described_topology([2, 2])
        out = isl.island_families(topo, model_fits_in_island=False)
        local = [f for f in out["families"] if f["key"] == "island_local_tp"][0]
        self.assertFalse(local["feasible"])
        self.assertIn("does NOT", local["requires"])

    def test_island_families_produce_as_many_lanes_as_the_topology(self):
        topo = isl.described_topology([2, 2, 2])
        out = isl.island_families(topo, model_fits_in_island=True)
        local = [f for f in out["families"] if f["key"] == "island_local_tp"][0]
        self.assertEqual(local["lanes"]["count"], 3)


# ---------------------------------------------------------------------------
# Lanes -- the structural requirement (DESIGN_201 PRIO-Nachtrag 8)
# ---------------------------------------------------------------------------


class RejectedRegister(CustomTestCase):
    """The register has to teach, or it is a wall of settled opinions."""

    def test_every_row_answers_gain_cost_and_why(self):
        from sglang.srt.planner import rejected as rej

        for e in rej.REGISTER:
            self.assertTrue(e.gain, e.key)
            self.assertTrue(e.cost, e.key)
            self.assertTrue(e.why, e.key)

    def test_the_three_lines_stay_short(self):
        """Short and precise was the requirement. A paragraph in the gain
        field is how this becomes the wall it replaced."""
        from sglang.srt.planner import rejected as rej

        for e in rej.REGISTER:
            self.assertLessEqual(len(e.gain), 120, e.key)
            self.assertLessEqual(len(e.cost), 120, e.key)
            self.assertLessEqual(len(e.why), 400, e.key)

    def test_only_a_not_default_row_can_be_unlocked(self):
        from sglang.srt.planner import rejected as rej

        for e in rej.REGISTER:
            row = e.to_json()
            if e.level == rej.BLOCKED:
                self.assertFalse(row["unlockable"], e.key)
                self.assertEqual(e.unlock, "", e.key)
            if row["unlockable"]:
                self.assertTrue(e.unlock.startswith("--"), e.key)

    def test_at_least_one_row_is_genuinely_offerable(self):
        """"Available on request" is only true if the request exists."""
        from sglang.srt.planner import rejected as rej

        offerable = [e for e in rej.REGISTER if e.to_json()["unlockable"]]
        self.assertTrue(offerable)


class Lanes(CustomTestCase):
    def test_a_single_lane_is_a_list_of_one(self):
        s = lanes.single_lane([0, 1, 2], goal="max_kv")
        self.assertEqual(len(s), 1)
        self.assertEqual(s.to_json()["count"], 1)
        self.assertFalse(s.to_json()["multi_lane"])
        self.assertEqual(s[0].lane_id, "main")

    def test_no_signature_in_the_lane_model_takes_a_pair(self):
        """Nachtrag 8 (a), enforced rather than remembered: a function taking
        two lanes is how a multi-group runtime silently becomes a dual-group
        one."""
        import inspect

        for name, fn in vars(lanes).items():
            if not callable(fn) or not getattr(fn, "__module__", "") == lanes.__name__:
                continue
            try:
                params = list(inspect.signature(fn).parameters)
            except (TypeError, ValueError):
                continue
            for bad in ("lane_a", "lane_b", "first_lane", "second_lane",
                        "pd_lane", "main_lane"):
                self.assertNotIn(bad, params, f"{name}({bad}) hardcodes a pair")

    def test_priority_is_a_class_with_an_order(self):
        fg = lanes.Lane("a", "A", (0,), priority_class=lanes.FOREGROUND)
        sc = lanes.Lane("b", "B", (0,), priority_class=lanes.SCAVENGER)
        self.assertLess(fg.priority_rank, sc.priority_rank)
        with self.assertRaises(ValueError):
            lanes.Lane("c", "C", (0,), priority_class="urgent")

    def test_duplicate_ids_are_refused(self):
        a = lanes.Lane("x", "A", (0,))
        b = lanes.Lane("x", "B", (1,))
        with self.assertRaises(ValueError):
            lanes.LaneSet((a, b))

    def test_co_residence_is_found_pairwise_over_the_set(self):
        s = lanes.LaneSet(
            (
                lanes.Lane("a", "A", (0, 1)),
                lanes.Lane("b", "B", (1,), priority_class=lanes.SCAVENGER),
                lanes.Lane("c", "C", (1,), priority_class=lanes.SCAVENGER),
            )
        )
        self.assertEqual(s.co_residence(), {1: ["a", "b", "c"]})
        # Three lanes on one card yield all three pairs, not the first two.
        self.assertEqual(len(s.sharing_pairs()), 3)

    def test_an_empty_lane_set_is_not_a_configuration(self):
        with self.assertRaises(ValueError):
            lanes.LaneSet(())

    def test_round_trips_through_json(self):
        s = lanes.lanes_from_card_groups([[0, 1], [2]], goals=["max_kv", "max_decode"])
        again = lanes.LaneSet.from_json(s.to_json())
        self.assertEqual(again.to_json(), s.to_json())


# ---------------------------------------------------------------------------
# The wiring: what /api/wizard/* actually answers
# ---------------------------------------------------------------------------


_TEXT = dict(
    model_type="qwen3_5", hidden_size=5120, num_hidden_layers=64,
    num_attention_heads=24, num_key_value_heads=4, head_dim=256,
    intermediate_size=17408, vocab_size=248064, linear_num_key_heads=16,
    linear_num_value_heads=48, linear_key_head_dim=128,
    linear_value_head_dim=128, linear_conv_kernel_dim=4,
    full_attention_interval=4, max_position_embeddings=262144,
    layer_types=[
        "full_attention" if (i + 1) % 4 == 0 else "linear_attention"
        for i in range(64)
    ],
)


class Endpoints(CustomTestCase):
    @classmethod
    def setUpClass(cls):
        cls._tmp = tempfile.TemporaryDirectory()
        with open(os.path.join(cls._tmp.name, "config.json"), "w") as f:
            json.dump(
                dict(
                    architectures=["Qwen3_5ForConditionalGeneration"],
                    model_type="qwen3_5",
                    text_config=dict(_TEXT),
                    quantization_config=dict(
                        quant_method="fp8", fmt="e4m3",
                        activation_scheme="dynamic",
                    ),
                ),
                f,
            )
        cls.payload = {
            "model": cls._tmp.name,
            "hardware": {
                "gpus": ["RTX 5090:32760", "RTX 3080:20480", "RTX 3080:20480"]
            },
            "gpus": [
                {"index": 0, "name": "NVIDIA GeForce RTX 5090", "total_mib": 32760},
                {"index": 1, "name": "NVIDIA GeForce RTX 3080", "total_mib": 20480},
                {"index": 2, "name": "NVIDIA GeForce RTX 3080", "total_mib": 20480},
            ],
            "remotes": [],
            "tp_size": 3,
        }

    @classmethod
    def tearDownClass(cls):
        cls._tmp.cleanup()

    def test_the_matrix_carries_all_four_blocks(self):
        d = webui.wizard_families_payload(dict(self.payload))
        self.assertTrue(d["ok"], d.get("reasons"))
        for key in ("rates", "tipping_points", "offload_depth", "islands"):
            self.assertIsNotNone(d[key], key)
        self.assertTrue(d["provenance_coverage"]["summary"])

    def test_every_family_declares_its_lanes(self):
        d = webui.wizard_families_payload(dict(self.payload))
        for f in d["families"]:
            self.assertGreaterEqual(f["lanes"]["count"], 1, f["key"])
        by = {f["key"]: f for f in d["families"]}
        # A family with two arms has two lanes -- and, because which cards
        # each arm keeps IS the split control (#258), the card sets are
        # deliberately empty and say so instead of being guessed.
        pd = by["pd_disjoint"]["lanes"]
        self.assertEqual(pd["count"], 2)
        self.assertEqual([ln["cards"] for ln in pd["lanes"]], [[], []])
        self.assertTrue(all("#258" in ln["note"] for ln in pd["lanes"]))
        # The rank-reuse family's lanes SHARE a card, and the sharing is data.
        self.assertTrue(by["pd_rank_reuse"]["lanes"]["co_residence"])

    def test_the_tipping_endpoint_answers_on_its_own(self):
        d = webui.wizard_tipping_payload(dict(self.payload))
        self.assertTrue(d["ok"])
        self.assertTrue(d["points"])
        self.assertIn("coverage", d)
        self.assertIsNotNone(d["rates"])

    def test_the_page_draws_the_blocks_and_the_endpoint(self):
        for token in (
            "wzTipping", "wzRates", "wzOffload", "wzIslands", "wzLanes",
            "wizardMeasureRun", "/api/wizard/tipping",
        ):
            self.assertIn(token, webui.INDEX_HTML, token)

    def test_the_loaded_model_bar_is_above_the_tabs(self):
        """It qualifies every tab, so it may not live inside one of them."""
        html = webui.INDEX_HTML
        self.assertIn('id="loadbar"', html)
        self.assertLess(html.index('id="loadbar"'), html.index('id="tab_landing"'))
        for token in ("renderLoadbar", "loadbarKvTokens", "loadbarPoll"):
            self.assertIn(token, html, token)

    def test_a_step_with_no_answer_is_never_marked_stale(self):
        """The first model pick used to grey the whole page and tell the
        reader its numbers were from the previous input. There were none."""
        self.assertIn("if(stale && !window._wizAnswered[step]) return;",
                      webui.INDEX_HTML)
        self.assertIn("wizardFamiliesDebounced", webui.INDEX_HTML)

    def test_a_scan_in_progress_has_its_own_colour(self):
        html = webui.INDEX_HTML
        self.assertIn(".scanning {", html)
        self.assertIn('<span class="scanning">scanning the model roots', html)

    def test_the_register_is_cards_not_a_wide_table(self):
        """The four-column table with two prose columns pushed the page past
        its own width. A card wraps; a table cell does not."""
        html = webui.INDEX_HTML
        self.assertIn("renderRejected", html)
        self.assertIn("rejectedUnlock", html)
        self.assertIn(".rj-b { display: grid;", html)
        # The old header row is gone, along with the layout that came with it.
        self.assertNotIn("<th>combination</th>", html)

    def test_the_pages_script_block_parses(self):
        """The dashboard is one inline script; a syntax error anywhere in it
        blanks every tab at once, and no Python test would notice. Cheap to
        check, so it is checked."""
        import re

        try:
            import esprima
        except ImportError:
            self.skipTest("esprima not installed; JS syntax not checked")
        blocks = re.findall(
            r"<script[^>]*>(.*?)</script>", webui.INDEX_HTML, re.S
        )
        self.assertEqual(len(blocks), 1)
        # esprima 4 predates ES2020; `??` and `?.` are used in the page and
        # are valid. Substituting them keeps the parser useful for everything
        # else rather than making the check unrunnable.
        probe = blocks[0].replace("??", "||").replace("?.", ".")
        esprima.parseScript(probe)
