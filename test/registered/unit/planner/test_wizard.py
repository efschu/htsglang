"""Guided configuration wizard (#270) -- the family and feasibility logic.

Three properties are pinned here, and they are the ones that make the page
trustworthy rather than merely present.

**A rejected combination is never proposed.** The register
(:mod:`sglang.srt.planner.rejected`) is the blocklist, and every ``blocked``
row it holds is a combination this project measured and settled. If the
matrix ever offers one, the wizard is inviting a repeat of work that was
already done and thrown away.

**A family that cannot boot here is shown WITH the reason.** An empty square
teaches nothing. Every infeasible cell carries a sentence naming what stops
it and where that sentence comes from -- an engine guard, a hardware count,
a design status, or a register row.

**Every number carries its provenance, and an absent one carries no value.**
The vocabulary is ``bench_factors``' three words and no fourth. A cell
labelled ``absent`` never has a number attached to it, because the whole
point of the third label is telling "nobody measured this" apart from "this
is zero".

The rig and the geometry are the reference ones -- Qwen3.6-27B FP8 on the
5090 + 2x 3080 box -- so the figures here can be read against the working
points ``test_lever_profiles`` records for the same configuration.
"""

import json
import os
import tempfile

from sglang.srt.planner import rejected as rejmod
from sglang.srt.planner import webui
from sglang.srt.planner import wizard as wz
from sglang.srt.planner.bench_factors import ABSENT, ESTIMATE, MEASURED
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=30, suite="base-a-test-cpu")

#: The reference rig: 5090 32 GiB + 2x 3080 20 GiB.
_RIG3 = ["RTX 5090:32760", "RTX 3080:20480", "RTX 3080:20480"]
_RIG3_CARDS = [
    {"index": 0, "name": "NVIDIA GeForce RTX 5090", "total_mib": 32760},
    {"index": 1, "name": "NVIDIA GeForce RTX 3080", "total_mib": 20480},
    {"index": 2, "name": "NVIDIA GeForce RTX 3080", "total_mib": 20480},
]

#: Qwen3.6-27B geometry, verbatim from the checkpoint's ``text_config``.
_QWEN36_27B_TEXT = dict(
    model_type="qwen3_5",
    hidden_size=5120,
    num_hidden_layers=64,
    num_attention_heads=24,
    num_key_value_heads=4,
    head_dim=256,
    intermediate_size=17408,
    vocab_size=248064,
    linear_num_key_heads=16,
    linear_num_value_heads=48,
    linear_key_head_dim=128,
    linear_value_head_dim=128,
    linear_conv_kernel_dim=4,
    full_attention_interval=4,
    max_position_embeddings=262144,
    layer_types=[
        "full_attention" if (i + 1) % 4 == 0 else "linear_attention" for i in range(64)
    ],
)


def _write_27b_config(tmpdir: str) -> str:
    cfg = dict(
        architectures=["Qwen3_5ForConditionalGeneration"],
        model_type="qwen3_5",
        text_config=dict(_QWEN36_27B_TEXT),
        quantization_config=dict(
            quant_method="fp8", fmt="e4m3", activation_scheme="dynamic"
        ),
    )
    with open(os.path.join(tmpdir, "config.json"), "w") as f:
        json.dump(cfg, f)
    return tmpdir


class WizardFixture(CustomTestCase):
    @classmethod
    def setUpClass(cls):
        cls._tmp = tempfile.TemporaryDirectory()
        cls.model = _write_27b_config(cls._tmp.name)

    @classmethod
    def tearDownClass(cls):
        cls._tmp.cleanup()

    def _payload(self, **kw):
        p = {
            "model": self.model,
            "hardware": {"source": "manual", "gpus": list(_RIG3)},
            "gpus": list(_RIG3_CARDS),
            "tp_size": 3,
            "kv_cache_dtype": "fp8_e4m3",
            "max_running_requests": 16,
            "usage_pattern": "fresh",
            "wizard_context_tokens": 8192,
            # Explicit: this fixture describes a rig with no coupled remote,
            # so the answer does not depend on whatever pairing session
            # another test happened to leave in the store.
            "remotes": [],
        }
        p.update(kw)
        return p

    def _matrix(self, **kw):
        d = webui.wizard_families_payload(self._payload(**kw))
        self.assertTrue(d.get("ok"), d.get("reasons"))
        return d

    @staticmethod
    def _fam(d, key):
        for f in d["families"]:
            if f["key"] == key:
                return f
        raise AssertionError(f"family {key} not in the matrix")

    @staticmethod
    def _variant(fam, spill="off", locality="local"):
        for v in fam["variants"]:
            if v["spill"] == spill and v["locality"] == locality:
                return v
        raise AssertionError(f"variant {spill}/{locality} missing")


# ---------------------------------------------------------------------------
# The register is a blocklist, and it is consulted
# ---------------------------------------------------------------------------


class TestRejectedRegister(CustomTestCase):
    def test_every_entry_carries_its_evidence(self):
        """A rejection without a number is an opinion, and an opinion cannot
        bind a later attempt."""
        for e in rejmod.REGISTER:
            self.assertTrue(e.evidence.strip(), f"{e.key} has no evidence")
            self.assertTrue(e.verdict.strip(), f"{e.key} has no verdict")
            self.assertIn(e.level, rejmod.LEVELS, e.key)
            self.assertIn(e.scope, ("rig", "general"), e.key)

    def test_keys_are_unique(self):
        keys = [e.key for e in rejmod.REGISTER]
        self.assertEqual(len(keys), len(set(keys)))

    def test_a_partial_tag_set_does_not_fire_a_combination_row(self):
        """Tree speculation on its own is not the rejected thing -- tree
        speculation UNDER uneven DCP is. Matching on the incomplete set would
        block a configuration the register never refused."""
        hits = {e.key for e in rejmod.check_combination(["tree-spec"])}
        self.assertNotIn("tree_spec_uneven_dcp", hits)
        hits = {e.key for e in rejmod.check_combination(["tree-spec", "uneven-dcp"])}
        self.assertIn("tree_spec_uneven_dcp", hits)

    def test_an_untagged_entry_never_matches_automatically(self):
        every_tag = sorted({t for e in rejmod.REGISTER for t in e.tags})
        for e in rejmod.check_combination(every_tag):
            self.assertTrue(e.tags, f"{e.key} matched with no tags")

    def test_the_two_levels_stay_apart(self):
        """Blocked is a wall; not-default is a measured loss that is still
        available. Collapsing them would either hide a working option or
        offer a broken one."""
        self.assertIn("tree_spec_uneven_dcp", rejmod.blocked_keys())
        self.assertNotIn("tree_spec_tp1", rejmod.blocked_keys())
        self.assertEqual(
            rejmod.by_key("tree_spec_tp1").level, rejmod.NOT_DEFAULT
        )

    def test_the_endpoint_serves_it_and_filters_by_level(self):
        allrows = webui.wizard_rejected_payload({})
        self.assertTrue(allrows["ok"])
        self.assertEqual(allrows["count"], len(rejmod.REGISTER))
        blocked = webui.wizard_rejected_payload({"level": "blocked"})
        self.assertEqual(
            blocked["count"], len([e for e in rejmod.REGISTER if e.level == "blocked"])
        )
        self.assertFalse(webui.wizard_rejected_payload({"level": "nonsense"})["ok"])


# ---------------------------------------------------------------------------
# Feasibility: refused families are visible, and they say why
# ---------------------------------------------------------------------------


class TestFamilyFeasibility(WizardFixture):
    def test_every_family_appears_whether_or_not_it_fits(self):
        d = self._matrix()
        self.assertEqual(
            [f["key"] for f in d["families"]], list(wz.FAMILY_KEYS)
        )

    def test_every_infeasible_cell_carries_a_reason(self):
        d = self._matrix()
        for f in d["families"]:
            for v in f["variants"]:
                if v["feasible"]:
                    self.assertEqual(v["reasons"], [], f'{f["key"]} {v["spill"]}')
                else:
                    self.assertTrue(
                        v["reasons"], f'{f["key"]} {v["spill"]}/{v["locality"]}'
                    )
                    for r in v["reasons"]:
                        self.assertTrue(r["reason"].strip())

    def test_spill_and_pd_are_refused_with_the_engine_guard(self):
        """The engine rejects the pair at arg parse. The wizard says so before
        a boot has to."""
        v = self._variant(self._fam(self._matrix(), "pd_disjoint"), spill="on")
        self.assertFalse(v["feasible"])
        keys = [r.get("register_key") for r in v["reasons"]]
        self.assertIn("spill_with_pd", keys)

    def test_spill_and_pipeline_parallelism_are_refused_too(self):
        v = self._variant(
            self._fam(self._matrix(), "pp_cross_rig"), spill="on", locality="network"
        )
        self.assertFalse(v["feasible"])
        self.assertIn("spill_with_pp_dp", [r.get("register_key") for r in v["reasons"]])

    def test_a_design_only_family_is_shown_but_never_offered(self):
        f = self._fam(self._matrix(), "pd_rank_reuse")
        self.assertFalse(f["feasible"])
        self.assertEqual(f["status"], "design")
        v = self._variant(f)
        self.assertIn("geometry re-sharder", " ".join(r["reason"] for r in v["reasons"]))
        # And the explanation still says which rank lives twice, because a
        # refusal the reader cannot understand is not a refusal they can act on.
        self.assertIn("tp1", f["explain"])
        self.assertIn("GDN state", f["explain"])

    def test_the_cross_rig_pipeline_slice_is_named_as_the_missing_part(self):
        f = self._fam(self._matrix(), "pp_cross_rig")
        self.assertEqual(f["status"], "partial")
        self.assertIn("slice 2", f["status_reason"])

    def test_network_families_need_a_remote_host_and_say_so(self):
        d = self._matrix(remotes=[])
        v = self._variant(self._fam(d, "satellite_prefill"), locality="network")
        self.assertFalse(v["feasible"])
        self.assertIn(
            "no remote host is known", " ".join(r["reason"] for r in v["reasons"])
        )

    def test_a_single_card_rig_refuses_the_families_that_need_more(self):
        d = self._matrix(
            hardware={"source": "manual", "gpus": ["RTX 5090:32760"]},
            gpus=[_RIG3_CARDS[0]],
            tp_size=1,
        )
        v = self._variant(self._fam(d, "uneven_tp_dcp"))
        self.assertFalse(v["feasible"])
        self.assertIn("at least 2 local cards", " ".join(r["reason"] for r in v["reasons"]))

    def test_the_blocklist_travels_with_the_answer(self):
        """The page shows what is never proposed, so the absence of an option
        is readable rather than mysterious."""
        d = self._matrix()
        self.assertEqual(d["blocked"]["count"], len(rejmod.blocked_keys()))

    def test_no_feasible_cell_is_a_blocked_register_row(self):
        """The property this whole module exists for: nothing the register
        blocked can come back offered."""
        d = self._matrix()
        blocked = set(rejmod.blocked_keys())
        for f in d["families"]:
            for v in f["variants"]:
                if not v["feasible"]:
                    continue
                tags = set(f["tags"]) | ({"spill"} if v["spill"] == "on" else set())
                for e in rejmod.check_combination(sorted(tags)):
                    self.assertNotIn(
                        e.key,
                        blocked,
                        f'{f["key"]} {v["spill"]}/{v["locality"]} offers {e.key}',
                    )


# ---------------------------------------------------------------------------
# Labels
# ---------------------------------------------------------------------------


class TestProvenanceLabels(WizardFixture):
    def test_only_the_three_words_are_used(self):
        d = self._matrix()
        allowed = {MEASURED, ESTIMATE, ABSENT}
        for f in d["families"]:
            for v in f["variants"]:
                for key, c in (v["targets"] or {}).items():
                    cells = (
                        [c["idle"], c["loaded"]] if c.get("pair") else [c]
                    )
                    for x in cells:
                        self.assertIn(x["provenance"], allowed, f'{f["key"]}/{key}')
                self.assertIn(v["undisturbedness"]["provenance"], allowed)

    def test_an_absent_cell_never_carries_a_value(self):
        d = self._matrix()
        for f in d["families"]:
            for v in f["variants"]:
                for key, c in (v["targets"] or {}).items():
                    cells = [c["idle"], c["loaded"]] if c.get("pair") else [c]
                    for x in cells:
                        if x["provenance"] == ABSENT:
                            self.assertIsNone(x["value"], f'{f["key"]}/{key}')
                            self.assertFalse(x["available"])
                            self.assertTrue(x["basis"].strip(), f'{f["key"]}/{key}')

    def test_the_helper_drops_a_value_handed_to_an_absent_label(self):
        """The invariant is enforced where the cell is built, not at every
        call site."""
        c = wz.cell(42.0, ABSENT, "no study", unit="tok/s")
        self.assertIsNone(c["value"])
        self.assertFalse(c["available"])

    def test_planner_arithmetic_is_an_estimate_not_a_measurement(self):
        d = self._matrix()
        for key in ("max_kv", "max_decode", "max_prefill", "max_parallel"):
            self.assertEqual(d["base_targets"][key]["provenance"], ESTIMATE)

    def test_a_split_probe_row_turns_the_estimate_into_a_measurement(self):
        """The one promotion path, and it applies only to the topology the
        probe actually boots."""
        row = {
            "candidate": "auto",
            "timestamp": "2026-07-28T00:00:00Z",
            "unbootable": "",
            "decode_tok_s": 77.5,
            "prefill_tok_s": 4321.0,
            "max_total_num_tokens": 123456,
        }
        base = {
            k: wz.cell(1.0, ESTIMATE, "planner arithmetic", unit="x")
            for k in ("max_kv", "max_decode", "max_prefill", "max_parallel")
        }
        ctx = wz.MatrixContext(gpus=_RIG3_CARDS, measured=row)
        fam = {f.key: f for f in wz.FAMILIES}["uneven_tp_dcp"]
        v = wz._family_variant(fam, "off", "local", ctx, base)
        self.assertEqual(v["targets"]["max_decode"]["provenance"], MEASURED)
        self.assertEqual(v["targets"]["max_decode"]["value"], 77.5)
        self.assertIn("split_probe", v["targets"]["max_decode"]["basis"])
        # The same row does not colour a family the probe never booted.
        solo = {f.key: f for f in wz.FAMILIES}["solo_tp"]
        v2 = wz._family_variant(solo, "off", "local", ctx, base)
        self.assertEqual(v2["targets"]["max_decode"]["provenance"], ESTIMATE)

    def test_an_unbootable_row_refuses_the_family_it_describes(self):
        """A split that was tried and did not boot is a finding, and the
        finding is what the cell says."""
        row = {
            "candidate": "auto",
            "timestamp": "2026-07-28T00:00:00Z",
            "unbootable": "UNBOOTABLE (rank 1 residual free 120 MiB < 2700 MiB)",
        }
        ctx = wz.MatrixContext(gpus=_RIG3_CARDS, measured=row)
        fam = {f.key: f for f in wz.FAMILIES}["uneven_tp_dcp"]
        v = wz._family_variant(fam, "off", "local", ctx, {})
        self.assertFalse(v["feasible"])
        self.assertIn("UNBOOTABLE", " ".join(r["reason"] for r in v["reasons"]))


# ---------------------------------------------------------------------------
# TTFT, undisturbedness and the link gate
# ---------------------------------------------------------------------------


class TestTtftAndUndisturbedness(WizardFixture):
    def test_ttft_is_always_a_pair(self):
        d = self._matrix()
        for f in d["families"]:
            for v in f["variants"]:
                if not v["feasible"]:
                    continue
                tt = v["targets"]["ttft"]
                self.assertTrue(tt["pair"])
                self.assertIn("idle", tt)
                self.assertIn("loaded", tt)

    def test_load_is_the_adversary_so_loaded_is_never_the_faster_half(self):
        pair = wz.ttft_pair(10000.0, 8192)
        self.assertGreater(pair["loaded"]["value"], pair["idle"]["value"])

    def test_without_a_prefill_rate_both_halves_are_absent(self):
        pair = wz.ttft_pair(None, 8192)
        for half in ("idle", "loaded"):
            self.assertEqual(pair[half]["provenance"], ABSENT)
            self.assertIsNone(pair[half]["value"])
            self.assertIn("split_probe", pair[half]["basis"])

    def test_undisturbedness_is_its_own_quantity_not_folded_into_ttft(self):
        """Moving prefill off the decode cards removes the spike and costs
        TTFT. Reporting one number would hide one of the two."""
        d = self._matrix()
        mono = self._variant(self._fam(d, "uneven_tp_dcp"))
        pd = self._variant(self._fam(d, "pd_disjoint"))
        self.assertEqual(mono["undisturbedness"]["value"], wz.ANCHOR_DECODE_SPIKE_MS)
        self.assertEqual(
            pd["undisturbedness"]["value"], wz.ANCHOR_DECODE_SPIKE_OFFLOADED_MS
        )
        self.assertLess(
            pd["undisturbedness"]["value"], mono["undisturbedness"]["value"]
        )

    def test_a_satellite_without_a_measured_rate_states_no_ttft(self):
        """The satellite's own prefill rate is 93.5 % of the answer, so
        without it there is nothing to say."""
        pair = wz._satellite_ttft(8192, None, 0.05)
        self.assertEqual(pair["idle"]["provenance"], ABSENT)
        self.assertIn("93.5", pair["idle"]["basis"])

    def test_a_satellite_ttft_counts_compute_plus_wire_plus_handover(self):
        pair = wz._satellite_ttft(8192, 2385.0, 0.053)
        expected = 8192 / 2385.0 + 0.053 + wz.ANCHOR_PD_HANDSHAKE_S
        self.assertAlmostEqual(pair["idle"]["value"], expected, places=6)
        self.assertEqual(pair["idle"]["provenance"], ESTIMATE)

    def test_pd_families_carry_the_no_spec_malus_visibly(self):
        """A PD decode figure is a no-spec figure. Showing the speculative
        plan's number under a PD label would compare two different servers."""
        d = self._matrix()
        for key in ("pd_disjoint", "satellite_prefill", "pp_cross_rig"):
            f = self._fam(d, key)
            for v in f["variants"]:
                self.assertEqual(
                    v["targets"]["max_decode"]["provenance"],
                    ABSENT,
                    key,
                )
                self.assertIn("without speculation", v["targets"]["max_decode"]["basis"])

    def test_the_link_gate_is_absent_without_a_measured_link(self):
        d = self._matrix()
        self.assertFalse(d["link_gate"]["available"])
        self.assertEqual(d["link_gate"]["provenance"], ABSENT)

    def test_the_link_gate_prices_transport_per_context_length(self):
        g = wz.link_gate(12288.0, 19 * 2**20, 1.93, [4096, 8192])
        self.assertTrue(g["available"])
        self.assertEqual(len(g["rows"]), 2)
        self.assertLess(g["rows"][0]["seconds"], g["rows"][1]["seconds"])
        # The fixed state block does not grow with the prompt, so doubling the
        # context does not double the bytes.
        self.assertLess(g["rows"][1]["bytes"], 2 * g["rows"][0]["bytes"])

    def test_a_slower_line_is_priced_from_the_same_geometry(self):
        fast = wz.link_gate(12288.0, 0.0, wz.ANCHOR_LINK_GBS_40G, [8192])
        slow = wz.link_gate(12288.0, 0.0, wz.ANCHOR_LINK_GBS_1GBE, [8192])
        self.assertGreater(
            slow["rows"][0]["seconds"] / fast["rows"][0]["seconds"], 10.0
        )


# ---------------------------------------------------------------------------
# The usage pattern, and the hybrid caveat that rides with it
# ---------------------------------------------------------------------------


class TestUsagePattern(WizardFixture):
    def test_fresh_prompts_add_no_warm_prefix_note(self):
        d = self._matrix(usage_pattern="fresh")
        v = self._variant(self._fam(d, "uneven_tp_dcp"))
        self.assertNotIn("Recurring prefixes", " ".join(v["notes"]))

    def test_recurring_prefixes_on_a_hybrid_model_name_the_store_truncation(self):
        """The reference checkpoint is hybrid GDN, and there the cross-server
        store route matches zero tokens -- only PD carries the handover."""
        d = self._matrix(usage_pattern="recurring")
        v = self._variant(self._fam(d, "uneven_tp_dcp"))
        joined = " ".join(v["notes"])
        self.assertIn("Recurring prefixes", joined)
        self.assertIn("Mamba", joined)
        pd = self._variant(self._fam(d, "pd_disjoint"))
        self.assertIn("PD ships the", " ".join(pd["notes"]))


# ---------------------------------------------------------------------------
# Command generation and the expert diff
# ---------------------------------------------------------------------------


class TestCommandGeneration(WizardFixture):
    def _cmd(self, family, **kw):
        d = webui.wizard_command_payload(
            self._payload(family=family, **kw)
        )
        self.assertTrue(d.get("ok"), d.get("error"))
        return d

    def test_an_unknown_family_is_refused_rather_than_guessed(self):
        d = webui.wizard_command_payload(self._payload(family="not-a-family"))
        self.assertFalse(d["ok"])

    def test_metrics_are_on_every_generated_command(self):
        """Mandatory on this rig with no topology exception: without it the
        dashboard has nothing to read."""
        for fam in ("solo_tp", "uneven_tp_dcp", "pd_disjoint"):
            self.assertIn("--enable-metrics", self._cmd(fam)["argv"], fam)

    def test_a_pd_command_does_not_carry_speculation_flags(self):
        """The engine turns speculation off with a warning, not an error, so a
        command that still names it launches a server that quietly differs."""
        argv = self._cmd("pd_disjoint")["argv"]
        for flag in wz._PD_DROP_FLAGS:
            self.assertNotIn(flag, argv)
        self.assertIn("--disaggregation-mode", argv)

    def test_spill_adds_exactly_the_flags_the_lane_requires(self):
        argv = self._cmd("uneven_tp_dcp", spill="on")["argv"]
        self.assertIn("--enable-kv-session-offload", argv)
        self.assertEqual(argv[argv.index("--attention-backend") + 1], "flashinfer")
        self.assertEqual(argv[argv.index("--page-size") + 1], "1")

    def test_every_added_flag_says_why_it_is_there(self):
        d = self._cmd("pd_disjoint", spill="off")
        for p in d["provenance"]:
            self.assertTrue(p["why"].strip(), p["flag"])

    def test_a_two_arm_family_returns_both_arms(self):
        """Launching one arm of a pair serves nothing."""
        self.assertTrue(self._cmd("pd_disjoint")["arms"])
        roles = [a["role"] for a in self._cmd("satellite_prefill")["arms"]]
        self.assertEqual(len(roles), 2)

    def test_an_edit_reads_as_a_difference_not_a_second_recommendation(self):
        d = self._cmd("uneven_tp_dcp", overrides={"--context-length": 16384})
        self.assertTrue(d["diff"])
        row = [x for x in d["diff"] if x["flag"] == "--context-length"][0]
        self.assertEqual(row["yours"], 16384)
        self.assertNotEqual(row["guided"], row["yours"])
        self.assertIn("--context-length", d["edited_argv"])
        self.assertEqual(
            d["edited_argv"][d["edited_argv"].index("--context-length") + 1], "16384"
        )
        # The guided command is still there, unedited, to compare against.
        self.assertNotEqual(d["command"], d["edited_command"])

    def test_no_edit_means_no_diff(self):
        d = self._cmd("uneven_tp_dcp")
        self.assertEqual(d["diff"], [])
        self.assertEqual(d["command"], d["edited_command"])

    def test_spill_drops_speculation_because_the_lane_refuses_it(self):
        """The spill lane rejects a speculative algorithm at arg parse unless
        a bring-up gate is set, so a command that keeps the flags is a
        command that does not boot."""
        prof = {
            "name": "test",
            "argv": ["--model-path", "/m", "--speculative-algorithm", "NEXTN",
                     "--speculative-num-steps", "3"],
            "launch_env": {},
        }
        d = wz.build_command("uneven_tp_dcp", profile=prof, spill="on")
        for flag in wz._PD_DROP_FLAGS:
            self.assertNotIn(flag, d["argv"])
        self.assertIn(
            "KVSO_ALLOW_SPEC", " ".join(p["why"] for p in d["provenance"])
        )
        # The same profile keeps them when spill is off -- the removal is the
        # lane's constraint, not a blanket rule.
        keep = wz.build_command("uneven_tp_dcp", profile=prof, spill="off")
        self.assertIn("--speculative-algorithm", keep["argv"])

    def test_every_generated_command_parses_against_the_deployed_arg_schema(self):
        """The check that matters without a boot: argparse is the same gate
        the server hits first, so a flag this build does not declare, a
        misspelled name and a value of the wrong type all fail here."""
        import argparse

        from sglang.srt.server_args import ServerArgs

        cases = [
            ("solo_tp", "off"),
            ("uneven_tp_dcp", "off"),
            ("uneven_tp_dcp", "on"),
            ("pd_disjoint", "off"),
            ("satellite_prefill", "off"),
            ("extra_solo_session", "off"),
            ("extra_solo_session", "on"),
            ("pp_cross_rig", "off"),
        ]
        for family, spill in cases:
            argv = self._cmd(family, spill=spill)["argv"]
            parser = argparse.ArgumentParser()
            ServerArgs.add_cli_args(parser)
            try:
                ns = parser.parse_args(argv)
            except SystemExit:  # argparse exits on an unknown or bad flag
                self.fail(f"{family}/{spill} produced an argv argparse rejects")
            self.assertTrue(ns.enable_metrics, f"{family}/{spill}")
            if family in ("pd_disjoint", "satellite_prefill") or spill == "on":
                self.assertIsNone(ns.speculative_algorithm, f"{family}/{spill}")

    def test_the_argv_comes_from_the_shared_generator(self):
        """The wizard and the runner tab must launch the same server, so the
        argv is the profile generator's, not one assembled here."""
        d = self._cmd("uneven_tp_dcp")
        self.assertEqual(d["profile_kind_wanted"], "uneven-max-tokens")
        self.assertIn("--model-path", d["argv"])


# ---------------------------------------------------------------------------
# Modifiers
# ---------------------------------------------------------------------------


class TestModifiers(WizardFixture):
    def test_expert_offload_is_a_modifier_not_a_family(self):
        self.assertNotIn("moe_expert_offload", wz.FAMILY_KEYS)
        d = self._matrix()
        keys = [m["key"] for m in d["modifiers"]]
        self.assertIn("moe_expert_offload", keys)

    def test_a_dense_checkpoint_says_there_is_nothing_to_offload(self):
        d = self._matrix()
        m = [x for x in d["modifiers"] if x["key"] == "moe_expert_offload"][0]
        self.assertFalse(m["applies"])
        self.assertIn("no experts", m["reason"])

    def test_gguf_moe_offload_is_refused_because_the_engine_will_not_refuse_it(self):
        """The one combination that fails late and confusingly rather than at
        arg parse -- so the wizard is the place that has to say no."""
        ctx = wz.MatrixContext(config_tags=("gguf", "moe"))
        m = [x for x in wz._modifiers(ctx) if x["key"] == "moe_expert_offload"][0]
        self.assertTrue(m["applies"])
        self.assertFalse(m["available"])
        self.assertIn("#268", m["reason"])


# ---------------------------------------------------------------------------
# The page is wired to the endpoints and holds no logic of its own
# ---------------------------------------------------------------------------


class TestWizardInIndex(CustomTestCase):
    def test_the_tab_and_the_view_exist(self):
        h = webui.INDEX_HTML
        for token in ("tab_wizard", "view_wizard", "wz_families", "wz_cmd",
                      "wz_overrides", "wz_rejected", "wz_hw"):
            self.assertIn(token, h, token)

    def test_the_view_is_a_direct_child_of_body_so_it_keeps_the_page_gutter(self):
        h = webui.INDEX_HTML
        self.assertIn('<div id="view_wizard" style="display:none">', h)
        # Round 5: #loadbar joined this selector too (it was previously
        # missing and rendered full-bleed, inconsistent with every other
        # top-level block) -- the wizard view's own membership is unchanged.
        self.assertIn(
            'body > .hdr, body > .tabs, body > #loadbar, body > div[id^="view_"]',
            h,
        )

    def test_the_tab_is_in_the_switch(self):
        h = webui.INDEX_HTML
        self.assertIn("'landing','wizard'", h)
        # #1: the Planner is no longer a tab. Its markup is nested INSIDE the
        # wizard as the expert step, so it must not appear in the tab switch
        # (it would be hidden and shown independently of its own container),
        # and no button may navigate to it.
        self.assertNotIn("'runner'", h.split("<script>")[1].split("showTab")[1][:400])
        self.assertNotIn('id="tab_runner"', h)
        self.assertNotIn("showTab('runner')", h)
        wiz = h.index('<div id="view_wizard"')
        run = h.index('<div id="view_runner"')
        self.assertLess(wiz, run, "view_runner must be nested inside view_wizard")

    def test_the_page_calls_the_endpoints_and_derives_nothing(self):
        h = webui.INDEX_HTML
        for path in ("/api/wizard/hardware", "/api/wizard/families",
                     "/api/wizard/command", "/api/wizard/rejected"):
            self.assertIn("'" + path + "'", h, path)
        # No family logic in the browser: the family list, the reasons and the
        # argv all arrive already decided.
        for leaked in ("FAMILY_KEYS", "spill_with_pd", "--enable-kv-session-offload"):
            self.assertNotIn(leaked, h.split("<script>")[1], leaked)

    def test_every_new_control_has_a_trade_off_written(self):
        from sglang.srt.planner import tooltips as tipsmod

        for key in ("wizard.target", "wizard.usage", "wizard.families",
                    "wizard.command", "wizard.expert"):
            self.assertIn(key, tipsmod.TRADEOFFS, key)
            self.assertIn(key, webui.tooltips_payload()["tooltips"])


class TestWizardHardwareEndpoint(CustomTestCase):
    def test_it_answers_without_probing_anything(self):
        d = webui.wizard_hardware_payload()
        self.assertTrue(d["ok"])
        self.assertIn("cards", d)
        self.assertIn("capabilities", d)
        self.assertIn("remotes", d)
        for c in d["cards"]:
            self.assertIn(c["rate_provenance"], (MEASURED, ABSENT))
            self.assertTrue(c["rate_basis"].strip())
