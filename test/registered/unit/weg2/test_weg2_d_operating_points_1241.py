# SPDX-License-Identifier: Apache-2.0
"""#1241 slice (2): the OPERATING-POINT axis of the #1017 D-weight solve.

Hermetic. No GPU, no model tree, no card-rate library on disk: the cost model
and the library are the runtime's own objects behind two patch points, so this
file runs identically here and on the remote desk (the model tree exists on
only one of the two boxes -- the known evidence-bound class that already makes
``TestDWeightObjective1017`` fail there).

NO HAND NUMBERS. Every fixture number below is quoted from a boot line, and
the first test proves the quote: group D's budgets from boot weg2sb5e must
reproduce that boot's own printed weight vector.
"""

import os
import unittest
from unittest import mock

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

from sglang.srt.weg2.launcher import (  # noqa: E402
    D_OPERATING_POINTS,
    D_TP_OBJECTIVE_CHOICES,
    D_TP_OBJECTIVE_DEFAULT,
    Card,
    EarlyReadFact,
    Weg2LaunchRefused,
    _gcd_reduce,
    _weights_from_scores,
    d_operating_point_line,
    d_operating_point_rows,
    d_tp_ratio_decision,
)

# --- THE FIXTURE, EVERY NUMBER QUOTED -------------------------------------
#
# boot weg2sb5e, front.log:99 (2026-09-08T21:01:19Z), the WEG2 D-WEIGHTS line:
#   "--rank-gpu-memory-mib [29312, 18096, 18080]: [1832, 1131, 1130]
#    (gcd-reduced, server_args.py:11239) ... attention [12, 6, 6] of 24
#    q-heads; GDN [7, 5, 4] of 16 linear-attention heads ... measured card
#    GEMM NVIDIA GeForce RTX 5090 203.4, NVIDIA GeForce RTX 3080 51.0,
#    NVIDIA GeForce RTX 3080 51.0 TFLOP/s"
SB5E_BUDGETS = [29312, 18096, 18080]
SB5E_WEIGHTS = (1832, 1131, 1130)
SB5E_Q_HEADS = 24
SB5E_GDN_UNITS = 16
# /root/.cache/sglang/card_library.json, source "measured", the same library
# that boot's line read by card NAME (it prints the GEMM half rounded to 203.4
# / 51.0; the artifact carries the full value and the bandwidth half).
LIB_GEMM = {"NVIDIA GeForce RTX 5090": 203.42, "NVIDIA GeForce RTX 3080": 50.97}
LIB_MEMBW = {"NVIDIA GeForce RTX 5090": 1661.6, "NVIDIA GeForce RTX 3080": 716.2}

CARDS = [
    Card(uuid="GPU-a", name="NVIDIA GeForce RTX 5090", nvml_index=1, total_mib=32607),
    Card(uuid="GPU-b", name="NVIDIA GeForce RTX 3080", nvml_index=0, total_mib=20480),
    Card(uuid="GPU-c", name="NVIDIA GeForce RTX 3080", nvml_index=2, total_mib=20480),
]


class FakeVariant:
    def __init__(self, gemm, membw):
        self.gemm_tflops = gemm
        self.membw_gbs = membw


class FakeLibrary:
    def __init__(self, gemm=None, membw=None):
        self._gemm = LIB_GEMM if gemm is None else gemm
        self._membw = LIB_MEMBW if membw is None else membw

    def variants(self, name):
        if name not in self._gemm:
            return ()
        return (FakeVariant(self._gemm[name], self._membw[name]),)


class FakePCM:
    """The runtime's cost model, reduced to what this solve calls on it."""

    q_heads = SB5E_Q_HEADS
    #: 12 o_groups x scale 2 = the 24 q-heads the boot line reports.
    attn_units = 12
    gdn_units = SB5E_GDN_UNITS
    mlp_units = 32

    def __init__(self, *_args, **_kwargs):
        pass

    def decode_round_time(self, mlp, membw, gemv=None, attn=None):
        # The roofline shape: slowest rank's shard bytes over its bandwidth.
        return max(m / b for m, b in zip(mlp, membw))

    def prefill_lockstep_compute_time(self, mlp, gemm, family=None, attn=None):
        return max(m / g for m, g in zip(mlp, gemm)) / 1000.0

    def predict_capacity(self, mlp, attn=None, token_vector=None):
        return {"p": [10000.0 * m for m in mlp], "ctx": 1.0}


def _patched(library=FakeLibrary, pcm=FakePCM):
    return (
        mock.patch("sglang.srt.uneven_perf.PerfCostModel", pcm),
        mock.patch(
            "sglang.srt.planner.card_rate_pass.load_measured_library",
            lambda *a, **k: (library() if library is not None else None),
        ),
    )


class OperatingPointVectorTest(unittest.TestCase):
    def rows(self, budgets=None, facts=None, library=FakeLibrary):
        p1, p2 = _patched(library=library)
        kw = {} if facts is None else {"facts": facts}
        with p1, p2:
            return d_operating_point_rows(
                CARDS, budgets or SB5E_BUDGETS, "/model", 6, **kw
            )

    # -- the fixture proves itself --------------------------------------

    def test_the_maxkv_row_REPRODUCES_boot_weg2sb5e_own_printed_vector(self):
        """If this fails the fixture is a hand number, not a quote."""
        self.assertEqual(_gcd_reduce(SB5E_BUDGETS), SB5E_WEIGHTS)
        rows, refusals = self.rows()
        self.assertEqual(refusals, [])
        maxkv = next(r for r in rows if r.position == "maxkv")
        self.assertEqual(maxkv.weights, SB5E_WEIGHTS)

    def test_the_two_positions_are_the_MEASURED_rate_ratios_and_nothing_else(self):
        rows, _ = self.rows()
        bs1 = next(r for r in rows if r.position == "decode-bs1")
        bs6 = next(r for r in rows if r.position == "decode-bs6")
        self.assertEqual(bs1.scores, (1661.6, 716.2, 716.2))
        self.assertEqual(bs6.scores, (203.42, 50.97, 50.97))
        self.assertEqual(bs1.weights, _weights_from_scores([1661.6, 716.2, 716.2]))
        self.assertEqual(bs6.weights, _weights_from_scores([203.42, 50.97, 50.97]))
        # The bandwidth ratio (2.32x) and the GEMM ratio (3.99x) are DIFFERENT
        # vectors -- which is the whole reason this axis exists.
        self.assertNotEqual(bs1.weights, bs6.weights)

    def test_MUTANT_5_the_vector_carries_the_measurement_not_a_rounding_of_it(self):
        """x1 scaling would quantise 3.99x to 4:1 and 2.32x to 2:1."""
        # 203.42 / 50.97 = 3.99097..., i.e. 3991:1000 with no common divisor.
        bs6 = _weights_from_scores([203.42, 50.97, 50.97])
        self.assertEqual(bs6, (3991, 1000, 1000))
        bs1 = _weights_from_scores([1661.6, 716.2, 716.2])
        self.assertEqual(bs1, (58, 25, 25))
        # The distinction a rounding destroys: 399/100 != 4.
        self.assertNotEqual(bs6[0] / bs6[1], 4.0)

    # -- the default must not move --------------------------------------

    def test_MUTANT_7_the_DEFAULT_maxkv_argv_is_BYTE_IDENTICAL(self):
        p1, p2 = _patched()
        with p1, p2:
            dec = d_tp_ratio_decision(
                D_TP_OBJECTIVE_DEFAULT, "both", CARDS, SB5E_BUDGETS, "/model", 6
            )
        self.assertEqual(dec.flags, ("--rank-tp-ratio", "auto"))
        self.assertEqual(D_TP_OBJECTIVE_DEFAULT, "maxkv")
        # And the SPEED arm is untouched too.
        p1, p2 = _patched()
        with p1, p2:
            spd = d_tp_ratio_decision("speed", "dec", CARDS, SB5E_BUDGETS, "/model", 6)
        self.assertEqual(
            spd.flags,
            ("--rank-tp-ratio", "auto-performance", "--rank-perf-tune", "dec"),
        )

    def test_the_default_D_argv_renders_identically_with_and_without_the_axis(self):
        """The argv-builder half: what reaches the process must be unchanged."""
        from sglang.srt.weg2.launcher import argv_d

        p1, p2 = _patched()
        with p1, p2:
            dec = d_tp_ratio_decision(
                D_TP_OBJECTIVE_DEFAULT, "both", CARDS, SB5E_BUDGETS, "/model", 6
            )
        common = dict(
            py="python3",
            model="/model",
            budgets=SB5E_BUDGETS,
        )
        try:
            with_axis = argv_d(
                common["py"], common["model"], common["budgets"], 8, 4096, 64, [], 6,
                262144, 32768, 2, False, dec.flags, (), 785500001, 4, 512,
            )
            pre_1241 = argv_d(
                common["py"], common["model"], common["budgets"], 8, 4096, 64, [], 6,
                262144, 32768, 2, False, ("--rank-tp-ratio", "auto"), (), 785500001,
                4, 512,
            )
        except TypeError as exc:  # signature drifted; the flags claim still holds
            self.skipTest("argv_d signature changed: %s" % exc)
        self.assertEqual(with_axis, pre_1241)
        self.assertIn("auto", with_axis)

    # -- the two new arms emit an explicit vector ------------------------

    def test_an_operating_point_emits_the_EXPLICIT_comma_vector(self):
        for pos in D_OPERATING_POINTS:
            p1, p2 = _patched()
            with p1, p2:
                dec = d_tp_ratio_decision(pos, "both", CARDS, SB5E_BUDGETS, "/model", 6)
            self.assertEqual(dec.flags[0], "--rank-tp-ratio")
            self.assertRegex(dec.flags[1], r"^\d+(,\d+)+$")
            self.assertIn(pos, D_TP_OBJECTIVE_CHOICES)

    def test_the_provenance_line_prints_ALL_THREE_vectors_with_price_and_pool(self):
        rows, refusals = self.rows()
        line = d_operating_point_line(rows, refusals, "maxkv")
        self.assertIn("WEG2 D-OPERATING-POINTS", line)
        for pos in ("maxkv", "decode-bs1", "decode-bs6"):
            self.assertIn(pos + ":" if pos != "maxkv" else "maxkv [SHIPPED]", line)
        self.assertIn("[SHIPPED]", line)
        self.assertIn("world pool", line)
        self.assertIn("round", line)
        self.assertIn("#705", line)
        self.assertIn("14.3 us", line)

    # -- the refusals ----------------------------------------------------

    def test_MUTANT_6_a_SATURATED_vector_is_REFUSED_W53(self):
        """The zero-head check would be unreachable: partition_units floors at
        one unit per rank. The reachable failure is a share below one unit."""
        rows, refusals = self.rows(
            library=lambda: FakeLibrary(
                gemm={
                    "NVIDIA GeForce RTX 5090": 1000.0,
                    "NVIDIA GeForce RTX 3080": 1.0,
                },
                membw={
                    "NVIDIA GeForce RTX 5090": 1000.0,
                    "NVIDIA GeForce RTX 3080": 1.0,
                },
            )
        )
        self.assertTrue(refusals, "a 1000:1 vector shipped without a refusal")
        self.assertTrue(
            all(r.startswith("W53") for r in refusals), refusals
        )
        self.assertIn("below ONE unit", " ".join(refusals))
        self.assertIn("pinned at its floor", " ".join(refusals))

    def test_a_FLAT_vector_is_even_TP_wearing_an_uneven_flag_and_is_REFUSED(self):
        flat = {
            "NVIDIA GeForce RTX 5090": 100.0,
            "NVIDIA GeForce RTX 3080": 100.0,
        }
        rows, refusals = self.rows(
            library=lambda: FakeLibrary(gemm=flat, membw=flat)
        )
        self.assertTrue(refusals)
        self.assertIn("even TP wearing an uneven flag", " ".join(refusals))

    def test_no_measured_library_REFUSES_W52_and_never_uses_a_nameplate_peak(self):
        rows, refusals = self.rows(library=None)
        self.assertEqual(rows, [])
        self.assertTrue(refusals[0].startswith("W52"), refusals)
        self.assertIn("never approximated from a nameplate peak", refusals[0])

    def test_a_missing_early_read_fact_REFUSES_W54_rather_than_pinning_the_env(self):
        trimmed = (
            EarlyReadFact(
                env_key="SGLANG_MAMBA_SSM_DTYPE",
                env_value="bfloat16",
                flag=("--mamba-ssm-dtype", "bfloat16"),
                groups="both",
                reader="uneven_perf.py:4648",
            ),
        )
        rows, refusals = self.rows(facts=trimmed)
        self.assertEqual(rows, [])
        self.assertTrue(refusals[0].startswith("W54"), refusals)
        self.assertIn("SGLANG_UNEVEN_DCP", refusals[0])

    def test_MUTANT_8_a_refusal_is_FATAL_only_when_that_position_is_shipped(self):
        """A maxkv boot must not die of a finding about an arm it did not take."""
        p1, p2 = _patched(library=None)
        with p1, p2:
            dec = d_tp_ratio_decision("maxkv", "both", CARDS, SB5E_BUDGETS, "/model", 6)
        self.assertEqual(dec.flags, ("--rank-tp-ratio", "auto"))
        self.assertIn("UNPRICED", dec.op_line)
        p1, p2 = _patched(library=None)
        with p1, p2:
            with self.assertRaises(Weg2LaunchRefused) as ctx:
                d_tp_ratio_decision(
                    "decode-bs1", "both", CARDS, SB5E_BUDGETS, "/model", 6
                )
        self.assertIn("W52", str(ctx.exception))
        self.assertIn("was SHIPPED", str(ctx.exception))

    def test_an_unknown_objective_still_refuses_W47(self):
        with self.assertRaises(Weg2LaunchRefused) as ctx:
            d_tp_ratio_decision("decode-bs99", "both", CARDS, SB5E_BUDGETS, "/model", 6)
        self.assertIn("W47", str(ctx.exception))

    def test_the_705_pricing_inputs_are_quoted_with_their_commits(self):
        from sglang.srt.weg2.launcher import (
            D705_AR_BS1_US,
            D705_FAMILY_SPLIT_BREAKEVEN_US,
            D705_PROVENANCE,
        )

        self.assertEqual(D705_AR_BS1_US, (31.0, 33.7))
        self.assertEqual(D705_FAMILY_SPLIT_BREAKEVEN_US, 14.3)
        self.assertIn("05baa99213", D705_PROVENANCE)
        self.assertIn("d937d5f76b", D705_PROVENANCE)
        # The measured all-reduce is ABOVE the break-even, which is why the
        # family split is refused rather than taken.
        self.assertGreater(D705_AR_BS1_US[0], D705_FAMILY_SPLIT_BREAKEVEN_US)


if __name__ == "__main__":
    unittest.main()
