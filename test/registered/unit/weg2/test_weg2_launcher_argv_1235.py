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

# THE SKIP GUARD IS NARROW ON PURPOSE. The sibling weg2 files wrap their whole
# import in try/except and skip the module, which is right for "this build has
# no weg2 launcher" and WRONG here: every symbol below is one this slice adds,
# so an ImportError on it is exactly the regression these tests exist to catch.
# Measured: with the wide guard, running this file against the parent commit
# reported "1 skipped" -- a red-first proof that could never go red.
try:
    from sglang.srt.weg2 import launcher as _launcher  # noqa: F401
except Exception as exc:  # pragma: no cover - no weg2 launcher in this build
    pytest.skip(f"weg2 launcher unavailable: {exc}", allow_module_level=True)

from sglang.srt.weg2.launcher import (  # noqa: E402
    D_TP_OBJECTIVE_CHOICES,
    D_TP_OBJECTIVE_DEFAULT,
    Card,
    Weg2LaunchRefused,
    argv_d,
    argv_p,
    build_parser,
    d_token_vector_decision,
    d_tp_ratio_decision,
)

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

    def test_default_line_names_source_objective_and_the_pool_it_moves(self):
        """#1270: the line follows the rule every vector line follows -- SOURCE,
        OBJECTIVE, and the pool figure the decision moves.

        It no longer cites ``assert_seed_superseded``: that arming belonged to
        the #1032 seed, and with the seed gone the install is armed by the ROLE
        instead (an undeclared vector is an ``estimate``, not a ``pin``). The
        line has to say which mechanism carries it, because the previous
        wording promised an install that boot weg2sb1 did not get.
        """
        line = d_token_vector_decision(None, "pin", None).line
        self.assertIn("none shipped", line)
        # SOURCE and OBJECTIVE, named as such.
        self.assertIn("SOURCE = estimate(budget)", line)
        self.assertIn("OBJECTIVE = the profiled per-rank capacity optimum", line)
        # The pool the decision moves, from the boot that measured both sides.
        self.assertIn("574,336", line)
        self.assertIn("681,856", line)
        self.assertIn("17,7,8", line)
        # ... and the mechanism that actually arms it now.
        self.assertIn("#1270", line)
        self.assertIn("role='estimate'", line)

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
    """#1254: ONE objective knob, all arms priced, and the DEFAULT is the
    measured max-KV cut rather than the solved one.

    TEST DEBT OF THE TRAIN 2 ONE-KNOB DECISION, paid here. This class was
    written when ``--pp-solve-objective`` defaulted to ``maxkv``; train 2
    reconciled the train's ``--p-cut-objective`` into this same flag and moved
    the default to ``incumbent``, and this file's guard was not moved with it.

    THE MAX-KV INTENT IS KEPT -- what is not the default is the UNPROVEN SOLVE.
    ``incumbent`` (32,18,14 / 8,4,4) is the cut boot weg2rg6 MEASURED at a
    714,788-token pool. The solver's ``maxkv`` row 31,17,16 is priced by a pool
    model that carries no per-stage FIXED POSTS -- lm_head, the MTP/draft head,
    the draft pools -- and boot weg2tr1 priced PP2 at 966,544 tokens against a
    PROFILED 155,164 (410,857 on the stage above). Defaulting to a cut chosen
    to maximise a number the model overstates ~6x on the binding stage is the
    law applied to the wrong quantity, so the default is the cut with a
    measurement behind it until #1259 puts those posts in the model. Both
    solved arms stay SELECTABLE and both stay PRICED on every boot.
    """

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

    def test_the_flag_defaults_to_makespan_by_user_order(self):
        """USER ORDER 2026-09-08, verbatim: 'nimm als default ab jetzt makespan'.

        This test asserted `incumbent` until that order. The name changed with
        the value on purpose -- a test called
        `..._defaults_to_the_measured_incumbent_...` that asserts `makespan`
        is how a suite starts lying about what it pins.
        """
        ns = build_parser().parse_args(["--tree", "/t", "--tag", "x"])
        self.assertEqual(ns.pp_solve_objective, "makespan")
        # The COST of the default must be findable at the flag, not only here:
        # makespan buys TTFT with KV and the help text has to say so.
        help_text = build_parser().format_help()
        self.assertIn("nimm als default ab jetzt makespan", help_text)
        self.assertIn("BUYS TTFT", help_text)
        # and the incumbent's expiry reason stays reachable on its own arm
        self.assertIn("#1259", help_text)

    def test_both_solved_arms_stay_selectable_and_the_solver_honours_them(self):
        # The max-KV intent is kept: the arm still exists, still solves, and
        # still names itself. Only its status as the DEFAULT moved.
        for objective in ("maxkv", "makespan"):
            ns = build_parser().parse_args(
                ["--tree", "/t", "--tag", "x", "--pp-solve-objective", objective]
            )
            self.assertEqual(ns.pp_solve_objective, objective)
            self.assertEqual(self._solve(objective).objective, objective)

    def test_incumbent_is_an_arm_of_this_one_flag_and_not_a_second_flag(self):
        # Train 2 reconciled two knobs into one; a re-appearing --p-cut-objective
        # is the second-bookkeeping shape this assertion refuses.
        act = next(
            a for a in build_parser()._actions if a.dest == "pp_solve_objective"
        )
        self.assertEqual(set(act.choices), {"maxkv", "makespan", "incumbent"})
        self.assertNotIn(
            "p_cut_objective", {a.dest for a in build_parser()._actions}
        )

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
            # both rows' POOLS by value, not a magic substring count: the
            # point is that a reader sees each objective's capacity as well
            # as its time, from either side of the choice.
            decision = self._solve(objective)
            self.assertIn(" pool %d " % int(decision.kv_floor.pool_tokens), line)
            self.assertIn(" pool %d " % int(decision.makespan.pool_tokens), line)

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


class TestMaxRunningRequestsReader(unittest.TestCase):
    """A pre-existing boot killer on the merge-train tip, found by --dry-run.

    ``_max_running_requests`` read ``--max-running-requests`` out of
    ``common_flags``. The flag LEFT common_flags when C1/R-12 made P's and D's
    bs independent, and a fifth required parameter was later added ahead of the
    ones it passed -- so the reader raised TypeError, two frames below
    ``solve_p_cut`` and ``d_overlap_cost_line``, i.e. on every boot from that
    tip before group P ever launched. The class is: a flag MOVED and one
    consumer did not move with it.
    """

    def test_it_answers_per_group_off_that_group_s_own_argv(self):
        from sglang.srt.weg2.launcher import _max_running_requests

        self.assertEqual(_max_running_requests(MODEL, "P", 8), 8)
        self.assertEqual(_max_running_requests(MODEL, "D", 6), 6)
        # independent by construction -- sharing one value is the coupling
        # C1/R-12 removed, so the two groups must be able to disagree.
        self.assertNotEqual(
            _max_running_requests(MODEL, "P", 8), _max_running_requests(MODEL, "D", 6)
        )

    def test_the_price_line_that_used_to_raise_now_renders(self):
        from sglang.srt.weg2.launcher import d_overlap_cost_line

        line = d_overlap_cost_line(MODEL, False, 6)
        self.assertIn("extra_buffer", line)
        self.assertIn("--max-running-requests 6", line)

    def test_no_consumer_still_indexes_a_common_flags_list(self):
        # the sibling sweep, as a check rather than as prose: common_flags is
        # read for --chunked-prefill-size (still there) and for nothing else.
        import inspect

        from sglang.srt.weg2 import launcher as L

        src = inspect.getsource(L)
        for line in src.splitlines():
            if "common_flags(" in line and "def common_flags" not in line:
                self.assertNotIn("--max-running-requests", line)
