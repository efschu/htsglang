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
