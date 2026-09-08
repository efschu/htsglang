"""#1235 / #1017 / #1254 / #1032: the Weg-2 launcher's argv stops carrying hand numbers.

Five things this file pins, all launcher-level, none needing a GPU or a launch:

  * group D's WEIGHT objective is a stated choice with a default of maxkv, not
    the literal ``--rank-tp-ratio auto`` that every boot shipped without saying
    which objective it had taken (#1017);
  * group D no longer seeds the RETRACTED #602 token vector, and a retracted
    vector offered on the launcher's own flag is refused BY TICKET at the desk
    rather than warned about after the window is spent (#1032/#797);
  * the decode-steps provenance is ONE sentence quoted by the constant, the
    ``--help`` and the boot line, so the confound the measuring boot recorded
    cannot be dropped in one of the three copies (#1030);
  * group P's cut objective defaults to the kv-floor and the provenance line
    prices BOTH objectives' cuts on pool AND ms (#1254);
  * every remaining literal and env gate in the two argv lists is either a flag
    with provenance or a declared early-read fact with one writer (#1235).
"""

import io
import os
import unittest

import pytest

try:
    from sglang.srt.weg2.launcher import (
        D_TP_OBJECTIVE_CHOICES,
        D_TP_OBJECTIVE_DEFAULT,
        Card,
        Weg2LaunchRefused,
        argv_d,
        build_parser,
        d_tp_ratio_decision,
    )
except Exception as exc:  # pragma: no cover
    pytest.skip(f"weg2 launcher unavailable: {exc}", allow_module_level=True)


MODEL = "/spinning/llm_stuff/club-3090/models-cache/Qwen3.8-27B-INT8-gdncov-vocabembed"
BUDGETS = [28904, 17704, 17672]
CARDS = [
    Card(0, "GPU-0", "NVIDIA GeForce RTX 5090", 32607),
    Card(1, "GPU-1", "NVIDIA GeForce RTX 3080", 20480),
    Card(2, "GPU-2", "NVIDIA GeForce RTX 3080", 20480),
]


def _flag_value(argv, flag):
    return argv[argv.index(flag) + 1] if flag in argv else None


class TestDWeightObjective1017(unittest.TestCase):
    """#1017: the capacity-first split is a CHOICE with a name and a price."""

    def test_default_objective_is_maxkv(self):
        self.assertEqual(D_TP_OBJECTIVE_DEFAULT, "maxkv")
        self.assertEqual(tuple(D_TP_OBJECTIVE_CHOICES), ("maxkv", "speed"))
        ns = build_parser().parse_args(["--tree", "/t", "--tag", "x"])
        self.assertEqual(ns.d_tp_objective, "maxkv")

    def test_maxkv_emits_auto_and_speed_emits_auto_performance(self):
        maxkv = d_tp_ratio_decision("maxkv", "both", CARDS, BUDGETS, MODEL, 8)
        self.assertEqual(maxkv.flags, ("--rank-tp-ratio", "auto"))
        speed = d_tp_ratio_decision("speed", "dec", CARDS, BUDGETS, MODEL, 8)
        self.assertEqual(
            speed.flags,
            ("--rank-tp-ratio", "auto-performance", "--rank-perf-tune", "dec"),
        )

    def test_line_names_objective_weights_and_heads(self):
        line = d_tp_ratio_decision("maxkv", "both", CARDS, BUDGETS, MODEL, 8).line
        self.assertIn("objective=maxkv", line)
        # the weight vector the RUNTIME will derive from the same budgets
        self.assertIn("[3613, 2213, 2209]", line)
        # the per-rank head partition, from the runtime's own cost model
        self.assertIn("attention [12, 6, 6] of 24 q-heads", line)

    def test_line_prices_the_slowest_rank_and_refuses_to_oversell_the_speed_arm(self):
        line = d_tp_ratio_decision("maxkv", "both", CARDS, BUDGETS, MODEL, 8).line
        self.assertIn("slowest-rank law", line)
        # the cost of the capacity split at the attention barrier, and the
        # honest limit: auto-performance does not move the attention split.
        self.assertIn("auto-performance never moves the attention split", line)
        self.assertIn("--rank-kv-ratio speed", line)

    def test_unknown_objective_is_refused_by_name(self):
        with self.assertRaises(Weg2LaunchRefused) as ctx:
            d_tp_ratio_decision("fast", "both", CARDS, BUDGETS, MODEL, 8)
        self.assertIn("W47", str(ctx.exception))

    def test_argv_d_carries_the_chosen_flags_not_a_literal(self):
        speed = d_tp_ratio_decision("speed", "enc", CARDS, BUDGETS, MODEL, 8)
        argv = argv_d(
            "py", MODEL, BUDGETS, 8, 1024, 8.0, [], tp_ratio_flags=speed.flags
        )
        self.assertEqual(_flag_value(argv, "--rank-tp-ratio"), "auto-performance")
        self.assertEqual(_flag_value(argv, "--rank-perf-tune"), "enc")

    def test_auto_stays_reachable_as_an_explicit_choice(self):
        argv = argv_d("py", MODEL, BUDGETS, 8, 1024, 8.0, [])
        self.assertEqual(_flag_value(argv, "--rank-tp-ratio"), "auto")
        self.assertNotIn("--rank-perf-tune", argv)


if __name__ == "__main__":
    unittest.main()


class TestTokenVector1032(unittest.TestCase):
    """#1032/#797: the retracted #602 seed leaves the argv, and stays out."""

    def test_default_argv_ships_no_token_vector_at_all(self):
        argv = argv_d("py", MODEL, BUDGETS, 8, 1024, 8.0, [])
        self.assertNotIn("--uneven-token-vector", argv)
        self.assertNotIn("--uneven-token-vector-role", argv)
        self.assertNotIn("29,19,16", argv)
        # the uneven-DCP path itself is unchanged: only the vector left.
        self.assertIn("--uneven-dcp", argv)
        self.assertIn("--uneven-dcp-weighted", argv)

    def test_default_line_names_the_branch_and_what_it_gives_up(self):
        line = d_token_vector_decision(None, "pin", None).line
        self.assertIn("none shipped", line)
        self.assertIn("assert_seed_superseded", line)
        self.assertIn("17,7,8", line)

    def test_retracted_vector_is_refused_by_ticket_in_either_role(self):
        for role in ("seed", "pin"):
            with self.assertRaises(Weg2LaunchRefused) as ctx:
                d_token_vector_decision("29,19,16", role, None)
            text = str(ctx.exception)
            self.assertIn("W46", text)
            self.assertIn("#602", text)

    def test_retraction_is_matched_after_gcd_and_despite_a_clean_lineage(self):
        # 58,38,32 reduces to 29,19,16; a declared non-retracted provenance
        # must not acquit it (#900).
        with self.assertRaises(Weg2LaunchRefused) as ctx:
            d_token_vector_decision("58,38,32", "pin", "planner")
        self.assertIn("#602", str(ctx.exception))

    def test_a_clean_vector_passes_and_carries_its_provenance(self):
        got = d_token_vector_decision("17,7,8", "seed", "measured")
        self.assertEqual(
            got.flags,
            (
                "--uneven-token-vector",
                "17,7,8",
                "--uneven-token-vector-role",
                "seed",
                "--uneven-token-vector-provenance",
                "measured",
            ),
        )

    def test_operator_role_defaults_to_pin_not_seed(self):
        ns = build_parser().parse_args(["--tree", "/t", "--tag", "x"])
        self.assertIsNone(ns.d_uneven_token_vector)
        self.assertEqual(ns.d_uneven_token_vector_role, "pin")


class TestDecodeStepsProvenance1030(unittest.TestCase):
    """#1030: one provenance sentence, three readers, and it keeps the confound.

    Group D's overlap + decode-steps defaults are CLOSED on the train (overlap
    is P-only by justification and D gets it by absence; the steps default is
    measured). What was open is the CITATION: the constant's comment carried
    the confound the measuring boot recorded about its own headline number and
    the other two copies -- ``--help`` and the boot's SCHEDULER: line -- had
    dropped it. A provenance sentence that degrades between copies reads as
    evidence while being less than the record supports.
    """

    def test_the_sentence_states_the_pair_and_refuses_the_overlap_alone_reading(self):
        from sglang.srt.weg2.launcher import D_DECODE_STEPS_PROVENANCE as prov

        self.assertIn("+14.0 %", prov)  # the pair, D2 vs D0
        self.assertIn("CONFOUNDED", prov)  # the record's own word
        self.assertIn("-7.35 %", prov)  # overlap+buffer measured NEGATIVE
        self.assertIn("+23.1 %", prov)  # the unconfounded steps knob, D2 vs D1
        self.assertIn("73.6", prov)  # D3 regresses -- an optimum, not a direction

    def test_help_quotes_the_same_sentence_and_still_renders(self):
        from sglang.srt.weg2.launcher import D_DECODE_STEPS_PROVENANCE as prov

        # argparse evaluates help as `help % params`; one bare '%' would make
        # --help raise for every flag, so the escape is part of the mechanism.
        text = build_parser().format_help()
        for fragment in ("CONFOUNDED", "+23.1 %", "-7.35 %"):
            self.assertIn(fragment, text)
        self.assertNotIn("%%", prov)

    def test_overlap_stays_group_p_s_flag_and_d_gets_it_by_absence(self):
        argv = argv_d("py", MODEL, BUDGETS, 8, 1024, 8.0, [])
        self.assertNotIn("--disable-overlap-schedule", argv)
        self.assertEqual(_flag_value(argv, "--mamba-radix-cache-strategy"), "extra_buffer")
        back = argv_d("py", MODEL, BUDGETS, 8, 1024, 8.0, [], disable_overlap=True)
        self.assertIn("--disable-overlap-schedule", back)
        # the strategy FOLLOWS the overlap choice; it is not a second knob.
        self.assertEqual(_flag_value(back, "--mamba-radix-cache-strategy"), "no_buffer")


def _pool_model_and_families():
    import json as _json

    from sglang.srt.planner import pp_cut as C

    cfg = _json.load(io.open(os.path.join(MODEL, "config.json")))
    text = cfg.get("text_config") or cfg
    kinds = text["layer_types"]
    families = tuple(
        C.LAYER_FAMILY_ATTENTION if k == "full_attention" else C.LAYER_FAMILY_LINEAR
        for k in kinds
    )
    kv = C.kv_mib_per_token_per_attn_layer_from_config(cfg, "fp8_e4m3", len(kinds))
    terms = C.checkpoint_weight_terms(MODEL)
    n_attn = len(terms.attention_layer_indices)
    mean = (
        terms.attn_layer_weight_bytes * n_attn
        + terms.linear_layer_weight_bytes * (terms.n_layers - n_attn)
    ) / max(1, terms.n_layers) / C.MIB
    budgets = (28904.0, 17704.0, 17672.0)
    model = C.PhasePoolModel(
        free_mib=budgets,
        weight_mib_per_layer=mean,
        kv_mib_per_token_per_attn_layer=kv,
        arming_floor_mib=tuple(1229.0 for _ in budgets),
        mamba_mib_per_linear_layer_per_slot=0.0,
        mamba_slots=8,
    )
    return families, model


@pytest.mark.skipif(
    not os.path.isdir(MODEL), reason="checkpoint headers not on this box"
)
class TestPCutObjective1254(unittest.TestCase):
    """#1254: group P's cut defaults to the kv-floor, and both cuts are priced."""

    def _solve(self, objective):
        from sglang.srt.planner.pp_cut_launch import solve_launch_cut

        families, pool = _pool_model_and_families()
        return solve_launch_cut(
            layer_families=families,
            incumbent_layers=[32, 18, 14],
            measured_ms_per_layer=[8.10, 35.16, 33.59],
            measured_provenance="test",
            card_names=[c.name for c in CARDS],
            pool_model=pool,
            cap_tokens=262144,
            objective=objective,
        )

    def test_default_objective_is_maxkv_at_the_flag_and_in_the_solver(self):
        ns = build_parser().parse_args(["--tree", "/t", "--tag", "x"])
        self.assertEqual(ns.pp_solve_objective, "maxkv")
        self.assertEqual(self._solve("maxkv").objective, "maxkv")

    def test_maxkv_takes_the_pool_maximal_feasible_cut(self):
        maxkv = self._solve("maxkv")
        makespan = self._solve("makespan")
        self.assertGreater(maxkv.chosen.pool_tokens, makespan.chosen.pool_tokens)
        self.assertLess(makespan.chosen.total_ms, maxkv.chosen.total_ms)
        # the two arms name each other, identically, from either side
        self.assertEqual(maxkv.chosen.layers, makespan.kv_floor.layers)
        self.assertEqual(makespan.chosen.layers, maxkv.makespan.layers)

    def test_provenance_prints_pool_AND_ms_of_BOTH_objectives_either_way(self):
        for objective in ("maxkv", "makespan"):
            line = self._solve(objective).provenance_line()
            self.assertIn("objective=%s" % objective, line)
            self.assertIn("BOTH objectives priced:", line)
            self.assertIn("maxkv cut", line)
            self.assertIn("makespan cut", line)
            self.assertEqual(line.count("ms/chunk"), 2)
            self.assertEqual(line.count("pool"), 3)  # constraint + two rows

    def test_trade_line_does_the_division_in_both_currencies(self):
        trade = self._solve("maxkv").trade_line()
        self.assertIn("objective=maxkv", trade)
        self.assertIn("pool", trade)
        self.assertIn("time", trade)
        self.assertIn("--pp-solve-objective", trade)

    def test_the_makespan_row_is_in_the_table_even_when_not_chosen(self):
        rows = self._solve("maxkv").table_lines()
        self.assertTrue(any(r.endswith("makespan") for r in rows))

    def test_gapped_is_not_a_default_and_the_753_gate_still_stands(self):
        # boot weg2gp1 (2026-09-08) probed the gapped layout on metal: 3 of 6
        # determined answers diverge, and the map loses 45.2 % of the pool.
        # The verdict is STAYS, so no objective may choose a gapped map here.
        for objective in ("maxkv", "makespan"):
            self.assertEqual(self._solve(objective).chosen.kind, "contiguous")


class TestLiteralsAndEnvGates1235(unittest.TestCase):
    """#1235: no unowned literal, no env gate whose value can be inherited."""

    def test_the_doubly_stated_facts_come_from_one_table(self):
        from sglang.srt.weg2.launcher import (
            EARLY_READ_FACTS,
            build_env,
            early_read_flags,
            early_read_provenance,
        )

        env = build_env("/t", "/v", "cvd", "/s", False, "tag")
        for fact in EARLY_READ_FACTS:
            # both currencies, one row: the env half...
            self.assertEqual(env[fact.env_key], fact.env_value)
            # ...and the argv half, for the groups the row names.
            self.assertTrue(set(fact.flag) <= set(early_read_flags("D")))
            if fact.groups == "both":
                self.assertTrue(set(fact.flag) <= set(early_read_flags("P")))
            # and the reader that makes the env half necessary is NAMED.
            self.assertIn("server_args.py:", fact.reader + early_read_provenance())

    def test_uneven_dcp_reaches_group_d_and_the_ssm_dtype_reaches_both(self):
        p = argv_p("py", MODEL, BUDGETS, 8, 1024, 8.0, [])
        d = argv_d("py", MODEL, BUDGETS, 8, 1024, 8.0, [])
        self.assertIn("--uneven-dcp", d)
        self.assertIn("--uneven-dcp-weighted", d)
        self.assertNotIn("--uneven-dcp", p)  # unchanged: P is tp_size=1
        self.assertEqual(_flag_value(p, "--mamba-ssm-dtype"), "bfloat16")
        self.assertEqual(_flag_value(d, "--mamba-ssm-dtype"), "bfloat16")

    def test_the_four_timeouts_are_output_and_an_inherited_export_cannot_win(self):
        from sglang.srt.weg2.launcher import build_env

        keys = {
            "SGLANG_BARLINK_BUILD_WINDOW_CAP_S": "60",
            "SGLANG_PP_CHAIN_RECV_STALL_S": "60",
            "SGLANG_PP_OCCUPANT_HORIZON_S": "90",
            "SGLANG_MATCH_REFUSAL_CENSUS_EVERY": "64",
        }
        saved = {k: os.environ.get(k) for k in keys}
        try:
            for k in keys:
                os.environ[k] = "999999"
            env = build_env("/t", "/v", "cvd", "/s", False, "tag")
            for k, want in keys.items():
                self.assertEqual(env[k], want, k)
        finally:
            for k, v in saved.items():
                if v is None:
                    os.environ.pop(k, None)
                else:
                    os.environ[k] = v

    def test_the_boolean_gates_have_flags_and_default_on(self):
        from sglang.srt.weg2.launcher import build_env

        ns = build_parser().parse_args(["--tree", "/t", "--tag", "x"])
        self.assertTrue(ns.arming_floor_solved)
        self.assertTrue(ns.hicache_bigram_keys)
        self.assertTrue(ns.hicache_flush_publish_sweep)
        on = build_env("/t", "/v", "cvd", "/s", False, "tag")
        self.assertEqual(on["SGLANG_HICACHE_BIGRAM_KEYS"], "1")
        off = build_env(
            "/t", "/v", "cvd", "/s", False, "tag", hicache_bigram_keys=False
        )
        self.assertNotIn("SGLANG_HICACHE_BIGRAM_KEYS", off)

    def test_the_argv_literals_became_flags_with_the_same_values(self):
        ns = build_parser().parse_args(["--tree", "/t", "--tag", "x"])
        self.assertEqual(ns.random_seed, 785500001)
        self.assertEqual(ns.barlink_bar1_cap_cycles, 300000000000)
        self.assertEqual(ns.collective_census_interval, 50)
        self.assertEqual(ns.p_barlink_bar1_window_mib, "24,PP_0=96")
        p = argv_p("py", MODEL, BUDGETS, 8, 1024, 8.0, [], random_seed=7)
        self.assertEqual(_flag_value(p, "--random-seed"), "7")
        self.assertEqual(_flag_value(p, "--barlink-bar1-window-mib"), "24,PP_0=96")
        d = argv_d("py", MODEL, BUDGETS, 8, 1024, 8.0, [], census_interval=3)
        self.assertEqual(_flag_value(d, "--collective-census-interval"), "3")

    def test_group_ds_measured_window_is_untouched(self):
        # #1234 C1 is the best-documented number in the file and this slice
        # must not have moved it while routing its neighbours.
        d = argv_d("py", MODEL, BUDGETS, 8, 1024, 8.0, [])
        self.assertEqual(
            _flag_value(d, "--barlink-bar1-window-mib"), "16,TP_0=32,DCP_0=40"
        )
        self.assertEqual(_flag_value(d, "--barlink-uncovered-class"), "refuse")
