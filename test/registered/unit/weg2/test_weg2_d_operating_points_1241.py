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

import dataclasses
import os
import unittest
from unittest import mock

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

from sglang.srt.weg2.launcher import (  # noqa: E402
    RING_FORM_SENTINEL_STORE_CFG,
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
    """The runtime's cost model, reduced to what this solve calls on it.

    ``gdn_unit_partition`` and the full ``predict_capacity`` dict are here
    because the SOLVE calls them, not because a test wants them: the review's
    R2 (a diagnostic that raises on the default boot) and the refuter's MF-4
    (``feasible`` and ``ctx`` read and discarded) are both defects of calling
    the runtime's objects through a narrower fake than the runtime's own
    surface. A fake narrower than the real object hides exactly this class.
    """

    q_heads = SB5E_Q_HEADS
    #: 12 o_groups x scale 2 = the 24 q-heads the boot line reports.
    attn_units = 12
    gdn_units = SB5E_GDN_UNITS
    mlp_units = 32
    tp_size = 3
    #: What `predict_capacity` reports. Overridden by the infeasible fake.
    feasible = True

    def __init__(self, inputs=None, *_args, **_kwargs):
        #: Kept so a test can assert WHICH configuration was priced.
        self.inputs = inputs

    def gdn_unit_partition(self, attn_vector=None):
        """The runtime's own: `[0] * tp_size` when it will not shard."""
        from sglang.srt.distributed.utils import partition_units

        n = len(attn_vector) if attn_vector else self.tp_size
        if int(self.gdn_units) >= n:
            return partition_units(int(self.gdn_units), list(attn_vector))
        return [0] * n

    def decode_round_time(self, mlp, membw, gemv=None, attn=None):
        # The roofline shape: slowest rank's shard bytes over its bandwidth.
        return max(m / b for m, b in zip(mlp, membw))

    def prefill_lockstep_compute_time(self, mlp, gemm, family=None, attn=None):
        return max(m / g for m, g in zip(mlp, gemm)) / 1000.0

    def predict_capacity(self, mlp, attn=None, token_vector=None):
        p = [10000.0 * m for m in mlp]
        # The runtime's own funded-context formula (uneven_perf.py:4881):
        # bounded by the SMALLEST rank, which is why a vector that collapses
        # one rank barely moves sum(p) and can halve this.
        ctx = min(sum(p), 64 * min(p)) if self.feasible else 0.0
        return {
            "p": p,
            "ctx": ctx,
            "token_vector": None,
            "feasible": self.feasible,
            "weights_gib": [0.0 for _ in mlp],
        }


class SmallGdnPCM(FakePCM):
    """A checkpoint whose GDN family is BELOW the world size.

    ``partition_units`` RAISES for it (`units < len(weights)`), which is the
    review's R2: the first version guarded with `gdn_units > 1`, true here,
    and the call sat outside every try/except on a path the DEFAULT boot
    walks.
    """

    gdn_units = 2


class InfeasiblePCM(FakePCM):
    feasible = False


class SmallestRankBoundPCM(FakePCM):
    """A rig where the funded context is bound by the SMALLEST rank.

    ``ctx = min(sum_r P_r, 64 * min_r P_r)``. Whenever the second term wins,
    ``sum(p)`` and the funded context are numerically different quantities --
    which is the whole reason the row carries both. A fake in which they can
    never diverge could not tell a correct row from one that prints
    ``sum(p)`` under both labels.
    """

    def predict_capacity(self, mlp, attn=None, token_vector=None):
        # One rank collapsed: 64 x its P is far below the world sum.
        p = [100000.0, 100000.0, 10.0]
        return {
            "p": p,
            "ctx": min(sum(p), 64 * min(p)),
            "token_vector": None,
            "feasible": True,
            "weights_gib": [0.0 for _ in mlp],
        }


def _patched(library=FakeLibrary, pcm=FakePCM):
    return (
        mock.patch("sglang.srt.uneven_perf.PerfCostModel", pcm),
        mock.patch(
            "sglang.srt.planner.card_rate_pass.load_measured_library",
            lambda *a, **k: (library() if library is not None else None),
        ),
    )


class OperatingPointVectorTest(unittest.TestCase):
    def rows(self, budgets=None, facts=None, library=FakeLibrary, pcm=FakePCM):
        p1, p2 = _patched(library=library, pcm=pcm)
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

    def test_MUTANT_7_the_maxkv_argv_is_BYTE_IDENTICAL_when_asked_for(self):
        # The user order of 2026-09-09 moved the DEFAULT to 'decode-bs6'. What
        # this pin protects is unchanged and is now stated where it belongs:
        # asking for 'maxkv' still produces the pre-#1241 argv, byte for byte.
        # It is asserted against the LITERAL and not against the default
        # constant, so that a future move of the default cannot silently take
        # the byte-identity proof with it.
        p1, p2 = _patched()
        with p1, p2:
            dec = d_tp_ratio_decision(
                "maxkv", "both", CARDS, SB5E_BUDGETS, "/model", 6
            )
        self.assertEqual(dec.flags, ("--rank-tp-ratio", "auto"))
        self.assertEqual(D_TP_OBJECTIVE_DEFAULT, "decode-bs6")
        # And the SPEED arm is untouched too.
        p1, p2 = _patched()
        with p1, p2:
            spd = d_tp_ratio_decision("speed", "dec", CARDS, SB5E_BUDGETS, "/model", 6)
        self.assertEqual(
            spd.flags,
            ("--rank-tp-ratio", "auto-performance", "--rank-perf-tune", "dec"),
        )

    def test_the_maxkv_D_argv_renders_identically_with_and_without_the_axis(self):
        """The argv-builder half: what reaches the process must be unchanged.

        Pinned against the LITERAL 'maxkv' since the user order of 2026-09-09
        moved the default to 'decode-bs6'. The guarantee this test exists for
        is about the maxkv ARM, not about whichever arm happens to be the
        default, and binding it to the default constant would have quietly
        re-pointed the proof at the new arm the day the default moved.
        """
        from sglang.srt.weg2.launcher import argv_d

        p1, p2 = _patched()
        with p1, p2:
            dec = d_tp_ratio_decision(
                "maxkv", "both", CARDS, SB5E_BUDGETS, "/model", 6
            )
        common = dict(
            py="python3",
            model="/model",
            budgets=SB5E_BUDGETS,
        )
        # NO skipTest HERE (review R7). A `except TypeError: skipTest` made
        # the byte-identity proof disappear silently on the next signature
        # change instead of going red -- and this is the whole argv-level
        # guarantee that the default path did not move. Call by KEYWORD so a
        # reordering cannot break it, and let a genuine signature change fail
        # loudly, which is the only way anyone learns the proof needs redoing.
        kw = dict(
            py=common["py"],
            model=common["model"],
            budgets=common["budgets"],
            s_gb=8,
            m_mib=4096,
            store_cfg=RING_FORM_SENTINEL_STORE_CFG,
            extra=[],
            d_bs=6,
            max_kv_per_request=262144,
            x_tokens=32768,
            num_continuous_decode_steps=2,
            disable_overlap=False,
            token_vector_flags=(),
            random_seed=785500001,
            barlink_cap_cycles=4,
            census_interval=512,
        )
        with_axis = argv_d(tp_ratio_flags=dec.flags, **kw)
        pre_1241 = argv_d(tp_ratio_flags=("--rank-tp-ratio", "auto"), **kw)
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
            all(r.startswith("W62") for r in refusals), refusals
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
        self.assertTrue(refusals[0].startswith("W61"), refusals)
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
        self.assertTrue(refusals[0].startswith("W63"), refusals)
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
        self.assertIn("W61", str(ctx.exception))
        self.assertIn("was SHIPPED", str(ctx.exception))

    # -- R2: a diagnostic on the default path must not RAISE ---------------

    def test_MUTANT_9_a_GDN_family_below_the_world_size_does_not_KILL_the_boot(self):
        """`partition_units` raises when `units < len(weights)`. The first
        version guarded GDN with `> 1` -- true for a 2-unit family on 3 ranks
        -- and the call sat outside every try/except in a function the
        DEFAULT maxkv boot calls unconditionally. A line that prints a trade
        must not be able to stop a launch that took neither side of it."""
        rows, refusals = self.rows(pcm=SmallGdnPCM)
        self.assertEqual(
            [r.position for r in rows], ["maxkv", "decode-bs1", "decode-bs6"]
        )
        for row in rows:
            self.assertEqual(
                row.gdn_heads,
                tuple(),
                "a family the runtime answers [0,0,0] for was reported as sharded",
            )
            # AND THE ROW IS STILL PRICED. Not raising is only half the fix:
            # a bare try/except around the raising call would also survive
            # this, by swallowing the geometry and reporting every arm
            # UNPRICED -- the diagnostic silently gone on exactly the models
            # that need it. The predicate has to keep the OTHER families
            # working, which is why it is the runtime's predicate and not a
            # catch. (This assertion is what mutant M15 -- restore
            # `gdn_units > 1` + a raw partition_units -- fails on.)
            self.assertTrue(row.attn_heads, "%s lost its attention geometry" % row.position)
            self.assertNotIn("geometry:", row.round_unit)
            self.assertIsNotNone(row.round_ms, row.position)
            self.assertIsNotNone(row.world_pool_tokens, row.position)
        # And the whole decision, which is what a boot actually calls. Pinned
        # on 'maxkv' by literal: this assertion is about the auto-emitting
        # arm, and the default is 'decode-bs6' since the 2026-09-09 order.
        p1, p2 = _patched(pcm=SmallGdnPCM)
        with p1, p2:
            dec = d_tp_ratio_decision(
                "maxkv", "both", CARDS, SB5E_BUDGETS, "/model", 6
            )
        self.assertEqual(dec.flags, ("--rank-tp-ratio", "auto"))
        self.assertIn("WEG2 D-OPERATING-POINTS", dec.op_line)

    def test_a_GDN_family_the_runtime_will_not_shard_earns_NO_W53(self):
        """The saturation check used the weaker `<= 1` predicate, so it could
        refuse a position over an axis that does not exist on this
        checkpoint -- a false refusal, which for a SHIPPED position is a
        boot blocker."""
        _rows, refusals = self.rows(pcm=SmallGdnPCM)
        self.assertEqual(
            [r for r in refusals if "GDN" in r and r.startswith("W62")],
            [],
            refusals,
        )

    # -- MF-4: feasibility and the FUNDED context ---------------------------

    def test_MUTANT_10_an_INFEASIBLE_vector_is_REFUSED_not_shipped(self):
        """`predict_capacity` returns `feasible`; the first version read only
        `p`. An infeasible vector was printed with a world-pool number and
        shipped."""
        _rows, refusals = self.rows(pcm=InfeasiblePCM)
        w64 = [r for r in refusals if r.startswith("W64")]
        self.assertEqual(len(w64), 2, refusals)
        for r in w64:
            self.assertIn("feasible=False", r)
        # ...and it is FATAL for the position that is shipped, and only then.
        p1, p2 = _patched(pcm=InfeasiblePCM)
        with p1, p2:
            with self.assertRaises(Weg2LaunchRefused) as ctx:
                d_tp_ratio_decision(
                    "decode-bs6", "both", CARDS, SB5E_BUDGETS, "/model", 6
                )
        self.assertIn("W64", str(ctx.exception))
        p1, p2 = _patched(pcm=InfeasiblePCM)
        with p1, p2:
            dec = d_tp_ratio_decision(
                "maxkv", "both", CARDS, SB5E_BUDGETS, "/model", 6
            )
        self.assertEqual(dec.flags, ("--rank-tp-ratio", "auto"))

    def test_MUTANT_11_the_line_carries_the_FUNDED_context_not_only_sum_p(self):
        """`sum(p)` is the wrong invariant on its own: the funded context is
        `min(sum P, 64 min P)`, bounded by the SMALLEST rank, so a vector
        that collapses one rank barely moves the sum and can halve the
        quantity a carrier bound is actually about."""
        rows, refusals = self.rows()
        line = d_operating_point_line(rows, refusals, "maxkv")
        self.assertIn("funded ctx", line)
        self.assertIn("feasible", line)
        for r in rows:
            self.assertIsNotNone(r.funded_ctx_tokens, r.position)
            self.assertIs(r.feasible, True, r.position)
        # The two must be able to DISAGREE, or carrying both is decoration:
        # on a rig where one rank collapses, the world pool barely moves and
        # the funded context is 64 x the smallest rank.
        rows2, _ = self.rows(pcm=SmallestRankBoundPCM)
        for r in rows2:
            self.assertEqual(r.world_pool_tokens, 200010)
            self.assertEqual(r.funded_ctx_tokens, 640)
        line2 = d_operating_point_line(rows2, [], "maxkv")
        self.assertIn("world pool 200010, funded ctx 640", line2)

    # -- MF-3 / MF-5: the claims the numbers cannot carry -------------------

    def test_the_bs6_price_does_NOT_claim_per_rank_family_GEMM_scores(self):
        """`family_tflops` is passed as None, so #475's `sum_fam max_rank`
        degenerates to `max_rank sum_fam`. Claiming the per-(rank, family)
        scores while passing None is a claim the arithmetic does not carry."""
        import inspect

        from sglang.srt.weg2 import launcher as lz

        rows, _ = self.rows()
        unit = next(r for r in rows if r.position == "decode-bs6").round_unit
        self.assertIn("PER-CARD scalar", unit)
        self.assertIn("family_tflops=None", unit)
        src = inspect.getsource(lz.d_operating_point_rows)
        self.assertNotIn("#324 per-(rank, family)", src)

    def test_the_bs6_regime_premise_is_marked_UNPROVEN_on_the_row_and_in_help(self):
        """"a bs=6 round is compute-bound" is the very quantity slice (1)
        measures, and slice (1) has produced no measurement. An argv-affecting
        arm may not present a hypothesis as a finding.

        SINCE THE USER ORDER OF 2026-09-09 THIS ARM IS THE DEFAULT, which
        makes the marking matter more, not less: the premise behind the arm a
        boot gets by saying nothing must be the loudest of the three. The
        assertion is therefore on the PROPERTY ("regime premises", "UNPROVEN"
        and the fact that it is now the default) rather than on one exact
        sentence -- a pin on the sentence goes red for a wording change that
        strengthens the warning, which is the wrong direction to defend.
        """
        import argparse
        import inspect

        from sglang.srt.weg2 import launcher as lz

        rows, _ = self.rows()
        unit = next(r for r in rows if r.position == "decode-bs6").round_unit
        self.assertIn("BOOT-UNPROVEN", unit)
        src = inspect.getsource(lz)
        i = src.index('"--d-tp-objective"')
        help_text = src[i : i + 4000]
        self.assertIn("REGIME PREMISES ARE", help_text)
        self.assertIn("UNPROVEN", help_text)
        # And the reader is told that one of these unproven premises is what
        # they get by default -- the half that only became true with the
        # 2026-09-09 order, and the half a reader most needs.
        self.assertIn("IS NOW THE DEFAULT", help_text)
        del argparse

    def test_the_help_no_longer_promises_the_ZERO_HEADS_refusal(self):
        """W62 is a SATURATION check. The zero-head check it replaced could
        never fire -- `partition_units` guarantees >= 1 unit per rank -- so a
        help text promising it documents a guard that does not exist."""
        import inspect

        from sglang.srt.weg2 import launcher as lz

        src = inspect.getsource(lz)
        i = src.index('"--d-tp-objective"')
        help_text = src[i : i + 4000]
        self.assertNotIn("would give any rank ZERO heads of a family", help_text)
        self.assertIn("SATURATES", help_text)
        self.assertIn("W64", help_text)

    # -- MF-6: one writer for the boot's own spec/KV facts ------------------

    def test_MUTANT_12_the_priced_config_is_the_SHIPPED_config_not_a_copy(self):
        """Four literals (`fp8_e4m3`, `NEXTN`, 3 draft tokens) stood in the
        argv builder AND in both PlanInputs blocks. They agreed by typing,
        not by construction, so moving the shipped draft-token count -- which
        #1242 is already doing -- would have gone on pricing the OLD
        configuration on every boot, silently."""
        from sglang.srt.weg2.launcher import (
            KV_CACHE_DTYPE,
            SPEC_ALGORITHM,
            SPEC_NUM_DRAFT_TOKENS,
            argv_d,
            d_plan_inputs,
        )

        argv = argv_d(
            py="python3", model="/model", budgets=SB5E_BUDGETS, s_gb=8, m_mib=4096,
            store_cfg=RING_FORM_SENTINEL_STORE_CFG, extra=[], d_bs=6,
        )
        i = argv.index("--speculative-num-draft-tokens")
        self.assertEqual(argv[i + 1], str(SPEC_NUM_DRAFT_TOKENS))
        self.assertEqual(argv[argv.index("--speculative-algorithm") + 1],
                         SPEC_ALGORITHM)
        self.assertEqual(argv[argv.index("--kv-cache-dtype") + 1], KV_CACHE_DTYPE)
        inputs = d_plan_inputs("/model", 3, 6)
        self.assertEqual(inputs.speculative_num_draft_tokens, SPEC_NUM_DRAFT_TOKENS)
        self.assertEqual(inputs.speculative_algorithm, SPEC_ALGORITHM)
        self.assertEqual(inputs.kv_cache_dtype, KV_CACHE_DTYPE)
        # And the rows really price THAT object, not a second literal block.
        p1, p2 = _patched()
        with p1, p2:
            rows, _ = d_operating_point_rows(CARDS, SB5E_BUDGETS, "/model", 6)
        self.assertTrue(rows)

    def test_the_spec_and_KV_literals_have_exactly_ONE_writer_each(self):
        import inspect

        from sglang.srt.weg2 import launcher as lz

        src = inspect.getsource(lz)
        self.assertEqual(src.count('"NEXTN"'), 1, "NEXTN typed more than once")
        self.assertEqual(
            src.count('speculative_num_draft_tokens=3'), 0,
            "a PlanInputs block still types the draft-token count",
        )

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


# --- #1293: THE ATTENTION AXIS OF THE OPERATING-POINT SOLVE ----------------
#
# THE FIXTURE IS BOOT weg2dec1 (BOOT_weg2dec1_0909.md, arms 2/3, 2026-09-09),
# EVERY NUMBER QUOTED. Both operating-point positions were REFUSED at launch
# with W62 on this rig:
#
#   arm 2 (decode-bs1): weights [58, 25, 25] from measured membw_gbs
#     [1661.6, 716.2, 716.2]; "rank 1's share of the 4 attention units is
#     below ONE unit ... the shipped partition [12, 6, 6] no longer
#     represents the measured ratio" -> REFUSED
#   arm 3 (decode-bs6): weights [3991, 1000, 1000] from measured gemm_tflops
#     [203.4, 51.0, 51.0] -> same floor, same refusal.
#
# ROOT CLASS (#492 correction, #503's enumerator fix at placement.py:813):
# under replicated-KV uneven DCP the attention COMPUTE follows the TOKEN
# vector (continuous, 64 largest-remainder units), not the 4-kv-head-unit
# grid -- the grid only carries the projections. Judging a measured rate
# ratio against a grid its compute does not ride is the defect these tests
# pin: today -> refused (proven on metal by that boot, and reproduced here by
# MUTANT A's head-grid arm); after -> feasible priced vectors whose token
# shares are proportional to the measurement.

#: weg2dec1's own maxkv line: "maxkv weights [3669, 2267, 2265]". Budgets
#: are 8x (gcd-reduced by the solve; gcd(3669, 2267, 2265) == 1).
DEC1_BUDGETS = [29352, 18136, 18120]
DEC1_MAXKV_WEIGHTS = (3669, 2267, 2265)
#: The refusal's own derived weights, reproduced from LIB_* by
#: test_the_two_positions... above: membw -> [58, 25, 25], gemm ->
#: [3991, 1000, 1000] (the library carries 203.42/50.97; the boot line
#: prints them rounded to 203.4/51.0).
DEC1_BS1_WEIGHTS = (58, 25, 25)
DEC1_BS6_WEIGHTS = (3991, 1000, 1000)
#: partition_units(64, weights) -- largest-remainder over the SAME 64 token
#: units the runtime's "Uneven DCP ... installed" vector uses
#: (_CP_TOKEN_UNITS, distributed/utils.py).
DEC1_BS1_TOKEN_UNITS = (34, 15, 15)
DEC1_BS6_TOKEN_UNITS = (42, 11, 11)


class Dec1PCM(FakePCM):
    """The weg2dec1 checkpoint geometry: 4 attention units over 3 ranks.

    ``predict_capacity`` honours ``token_vector`` with the runtime's OWN
    pinned-vector formula (``cp_token_context_budget``), because the #1293
    seam must be provably feeding the pin to the gate -- a fake that ignores
    the kwarg could not tell "token vector computed" from "token vector
    priced", which is exactly the #492 lesson (the price must reach the
    gate, not be rounded away).
    """

    attn_units = 4
    q_heads = 24
    gdn_units = 16

    def predict_capacity(self, mlp, attn=None, token_vector=None):
        from sglang.srt.distributed.utils import cp_token_context_budget

        p = [10000.0 * m for m in mlp]
        if token_vector is not None:
            ctx = float(
                cp_token_context_budget(
                    [max(int(v), 1) for v in token_vector],
                    [max(int(x), 1) for x in p],
                )
            )
        else:
            ctx = min(sum(p), 64 * min(p)) if self.feasible else 0.0
        return {
            "p": p,
            "ctx": ctx,
            "token_vector": None,
            "feasible": self.feasible,
            "weights_gib": [0.0 for _ in mlp],
        }


class AttentionTokenAxis1293Test(unittest.TestCase):
    def rows(self, budgets=None, library=FakeLibrary, pcm=Dec1PCM):
        p1, p2 = _patched(library=library, pcm=pcm)
        with p1, p2:
            return d_operating_point_rows(
                CARDS, budgets or DEC1_BUDGETS, "/model", 6
            )

    # -- the red half, statically: the head grid FLOORS this rig ----------

    def test_the_head_grid_floors_ranks_1_and_2_exactly_as_weg2dec1_said(self):
        """The defect's mechanism, held as a fact: on 4 attention units both
        measured vectors give ranks 1 and 2 a sub-unit share. This is the
        arithmetic behind both metal refusals, and it must STAY true -- the
        fix moves the verdict to the axis the compute rides, it does not
        bend the head grid."""
        from sglang.srt.weg2.launcher import _axis_floor_ranks

        self.assertEqual(_axis_floor_ranks(4, DEC1_BS1_WEIGHTS), (1, 2))
        self.assertEqual(_axis_floor_ranks(4, DEC1_BS6_WEIGHTS), (1, 2))
        # And the token grid does NOT floor them.
        self.assertEqual(_axis_floor_ranks(64, DEC1_BS1_WEIGHTS), ())
        self.assertEqual(_axis_floor_ranks(64, DEC1_BS6_WEIGHTS), ())

    # -- the fix: both positions feasible on the token axis ----------------

    def test_1293_both_weg2dec1_positions_are_FEASIBLE_priced_vectors(self):
        """weg2dec1 arms 2/3 in fixture form: today's tree refuses both with
        W62 ("pool-feasible, axis-infeasible"); after #1293 both must yield
        a feasible priced vector, because on this rig's serving form
        (replicated-KV uneven DCP) the attention compute rides the token
        axis, where [58, 25, 25] and [3991, 1000, 1000] are representable."""
        rows, refusals = self.rows()
        self.assertEqual(refusals, [], refusals)
        maxkv = next(r for r in rows if r.position == "maxkv")
        bs1 = next(r for r in rows if r.position == "decode-bs1")
        bs6 = next(r for r in rows if r.position == "decode-bs6")
        self.assertEqual(maxkv.weights, DEC1_MAXKV_WEIGHTS)
        self.assertEqual(bs1.weights, DEC1_BS1_WEIGHTS)
        self.assertEqual(bs6.weights, DEC1_BS6_WEIGHTS)
        for row in (bs1, bs6):
            self.assertEqual(row.attn_axis, "token", row.position)
            self.assertIs(row.feasible, True, row.position)
            self.assertIsNotNone(row.round_ms, row.position)
            self.assertIsNotNone(row.world_pool_tokens, row.position)
            self.assertIsNotNone(row.funded_ctx_tokens, row.position)
        # The projection split is still the head grid's [12, 6, 6] -- that
        # split is REAL (the q/o projections do shard that way); what moved
        # is which axis the family's COMPUTE is judged and priced on.
        self.assertEqual(bs1.attn_heads, (12, 6, 6))

    def test_1293_token_shares_are_proportional_to_the_measurement(self):
        """Largest-remainder over the 64 token units, exactly the runtime's
        own integerisation -- and the shares must SUM TO THE WORLD (the
        owner rule hands out every one of the 64 units exactly once)."""
        rows, _ = self.rows()
        bs1 = next(r for r in rows if r.position == "decode-bs1")
        bs6 = next(r for r in rows if r.position == "decode-bs6")
        self.assertEqual(bs1.attn_token_units, DEC1_BS1_TOKEN_UNITS)
        self.assertEqual(bs6.attn_token_units, DEC1_BS6_TOKEN_UNITS)
        for row in (bs1, bs6):
            units = row.attn_token_units
            self.assertEqual(sum(units), 64, row.position)
            total_w = sum(row.weights)
            for r, (u, w) in enumerate(zip(units, row.weights)):
                self.assertLess(
                    abs(u - 64.0 * w / total_w),
                    1.0,
                    "rank %d of %s drifted a full unit off the measured "
                    "ratio" % (r, row.position),
                )

    def test_1293_the_pinned_token_vector_REACHES_the_capacity_gate(self):
        """#492's discipline: the pinned vector's funded context is
        cp_token_context_budget(pin, P) -- strictly the weaker of the two --
        and it must reach the gate, not be rounded away. mlp for
        [58, 25, 25] on 32 units is [17, 8, 7] -> P = [170000, 80000, 70000];
        matched ctx would be 320000, the pin funds
        min(170000//34, 80000//15, 70000//15) * 64 = 4666 * 64 = 298624."""
        rows, _ = self.rows()
        bs1 = next(r for r in rows if r.position == "decode-bs1")
        self.assertEqual(bs1.funded_ctx_tokens, 298624)
        self.assertNotEqual(bs1.funded_ctx_tokens, 320000)
        # maxkv keeps the derived-matched call (byte-identical default):
        # mlp = partition_units(32, [3669, 2267, 2265]) = [14, 9, 9] ->
        # P = [140000, 90000, 90000] -> min(320000, 64 * 90000) = 320000.
        maxkv = next(r for r in rows if r.position == "maxkv")
        self.assertEqual(maxkv.funded_ctx_tokens, 320000)

    # -- MUTANT A: attention back on the head grid -------------------------

    def test_1293_MUTANT_A_the_head_grid_arm_reproduces_the_metal_refusal(self):
        """Forcing the axis back to "head" must reproduce weg2dec1's W62 for
        both positions -- this is the red half of the slice, executable, and
        the detector for the mutant that reverts the axis."""
        p1, p2 = _patched(pcm=Dec1PCM)
        with p1, p2, mock.patch(
            "sglang.srt.weg2.launcher._attn_axis_for",
            lambda weights, plan_flags: "head",
        ):
            _rows, refusals = d_operating_point_rows(
                CARDS, DEC1_BUDGETS, "/model", 6
            )
        w62 = [r for r in refusals if r.startswith("W62")]
        self.assertEqual(len(w62), 2, refusals)
        for r in w62:
            self.assertIn("attention", r)
            self.assertIn("axis=head", r)
            self.assertIn("below ONE unit", r)
        self.assertIn("[58, 25, 25]", w62[0])
        self.assertIn("[3991, 1000, 1000]", w62[1])

    # -- MUTANT C: a non-DCP form must never take the token axis -----------

    def test_1293_MUTANT_C_no_DCP_evidence_means_the_head_axis(self):
        """The axis decision is #503's shared plan-time predicate
        (plan_uneven_dcp_kv_replicated), never a third spelling: with no DCP
        evidence on the flags (dcp_size unset, no KV token vector) even the
        steepest weights ride the head grid, because there the token vector
        does not exist."""
        from sglang.srt.uneven_perf import PlanInputs
        from sglang.srt.weg2.launcher import _attn_axis_for

        bare = PlanInputs(tp_size=3, model_path="/m")
        self.assertEqual(_attn_axis_for(DEC1_BS1_WEIGHTS, bare), "head")
        armed = dataclasses.replace(bare, dcp_size=3)
        self.assertEqual(_attn_axis_for(DEC1_BS1_WEIGHTS, armed), "token")
        # A uniform plan is even DCP's fast path -- head axis even when armed.
        self.assertEqual(_attn_axis_for((25, 25, 25), armed), "head")
        # And a KV token vector alone is also DCP evidence (the predicate's
        # own second arm), so the spelling here really is the shared one.
        vec = dataclasses.replace(bare, kv_token_vector=[2, 1, 1])
        self.assertEqual(_attn_axis_for(DEC1_BS1_WEIGHTS, vec), "token")

    def test_1293_a_FLAT_vector_is_still_refused_whatever_the_axis(self):
        flat = {
            "NVIDIA GeForce RTX 5090": 100.0,
            "NVIDIA GeForce RTX 3080": 100.0,
        }
        _rows, refusals = self.rows(
            library=lambda: FakeLibrary(gemm=flat, membw=flat)
        )
        self.assertTrue(refusals)
        self.assertIn("even TP wearing an uneven flag", " ".join(refusals))

    # -- the refusal STAYS for a genuinely unrepresentable vector ----------

    def test_1293_a_vector_the_TOKEN_grid_cannot_represent_is_still_W53(self):
        """64 units floor a rank whose share is below 1/64 -- a ~1000:1 rate
        ratio. The fix moves the verdict to the riding axis; it does not
        delete the verdict."""
        _rows, refusals = self.rows(
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
        self.assertTrue(refusals)
        for r in refusals:
            self.assertTrue(r.startswith("W62"), r)
            self.assertIn("axis=token", r)
            self.assertIn("64", r)
            self.assertIn("below ONE unit", r)

    # -- MLP/GDN keep their own grids --------------------------------------

    def test_1293_GDN_stays_on_its_own_unit_grid(self):
        """partition_units(16, [58,25,25]) == [8,4,4] and
        partition_units(16, [3991,1000,1000]) == [10,3,3]: the token axis
        carries ONLY the attention compute; GDN (and MLP, asserted through
        the funded-ctx number above, which prices mlp [17,8,7] of 32) keep
        the grids the runtime shards them on."""
        rows, _ = self.rows()
        bs1 = next(r for r in rows if r.position == "decode-bs1")
        bs6 = next(r for r in rows if r.position == "decode-bs6")
        self.assertEqual(bs1.gdn_heads, (8, 4, 4))
        self.assertEqual(bs6.gdn_heads, (10, 3, 3))

    # -- the line names the axis, per row ----------------------------------

    def test_1293_the_line_prints_attn_axis_and_token_units_per_row(self):
        rows, refusals = self.rows()
        line = d_operating_point_line(rows, refusals, "maxkv")
        self.assertIn("attn_axis=token", line)
        self.assertIn("token_units [34, 15, 15]", line)
        self.assertIn("token_units [42, 11, 11]", line)
        # The head grid still BINDS for the projections on this rig (4 units,
        # ranks 1/2 floored) -- stated on the row instead of either silent or
        # fatal.
        self.assertIn("attention-projections/head grid BINDS", line)
        # A head-axis form still says so.
        from sglang.srt.uneven_perf import PlanInputs
        from sglang.srt.weg2.launcher import _attn_axis_for

        self.assertEqual(
            _attn_axis_for((2, 1, 1), PlanInputs(tp_size=3, model_path="/m")),
            "head",
        )

    def test_1293_the_maxkv_argv_is_STILL_byte_identical(self):
        """The #1241 golden, re-run under the dec1 geometry: the axis work
        must not move a single argv byte of the maxkv arm.

        Bound to the LITERAL since the 2026-09-09 order made 'decode-bs6' the
        default -- the golden is about that arm, not about whichever arm is
        currently default.
        """
        p1, p2 = _patched(pcm=Dec1PCM)
        with p1, p2:
            dec = d_tp_ratio_decision(
                "maxkv", "both", CARDS, DEC1_BUDGETS, "/model", 6
            )
        self.assertEqual(dec.flags, ("--rank-tp-ratio", "auto"))
        self.assertIn("WEG2 D-OPERATING-POINTS", dec.op_line)
        # And the previously-refused positions now DECIDE (the launch-level
        # consequence: an operating-point boot on this rig gets a vector).
        for pos, want in (
            ("decode-bs1", "58,25,25"),
            ("decode-bs6", "3991,1000,1000"),
        ):
            p1, p2 = _patched(pcm=Dec1PCM)
            with p1, p2:
                d = d_tp_ratio_decision(pos, "both", CARDS, DEC1_BUDGETS, "/model", 6)
            self.assertEqual(d.flags, ("--rank-tp-ratio", want))


if __name__ == "__main__":
    unittest.main()
