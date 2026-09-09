# SPDX-License-Identifier: Apache-2.0
"""#1286: every candidate, every objective, priced by the BOOT-SIZING formula.

THE DEFECT, measured on boot weg2sb5f (2026-09-09 04:38Z, tip 11df059318,
record /spinning/gpu-arb/weg2/BOOT_weg2sb5f_PREFLIGHT_0909.md).  The launcher
shipped the makespan cut and printed::

    PP-CUT SHIPPED: layers=44,10,10 attn=11,2,3 pool_tokens=499967 ...

and group P then sized itself::

    [PP0] KV pool sizing: available_bytes=6863273984 (6.392 GiB),
          cell_size=22528, page_size=1 -> max_total_num_tokens=304655
    [PP0] KV token sizing: rank 0 local capacity 304655 tokens, min-reduced
          across ranks to 304655 (THIS RANK BINDS)

**+64.1 % over-priced on the binding stage.**  Three halves of the formula are
RIGHT and must not be blamed: the per-rank attention count (PP0's ``cell_size``
22528 = 11 x 2048 exactly), the 2048-byte fp8 cell, and the min-over-ranks
reduction.  What is missing is the SUBTRACTION half -- the boot's own budget
posts, which the runtime emits verbatim on the success path
(``model_runner_kv_cache_mixin.py:1174``)::

    [world_rank 0] KV budget posts (GiB): weights + runtime state=17.908,
      gapped corridor holdback=1.000, mamba state pool=1.005,
      prefill activation reserve=1.000, GGUF dequant scratch=0.000
      | rest=6.392

The pool model charged only ``mean_layer_mib * n_layers`` plus a 1229 MiB
arming floor, so it did not see (a) the per-stage FIXED weight posts -- the
embedding on stage 0, lm_head + the MTP/draft head on the last stage, and the
replicated payload every stage carries -- (b) the mamba state pool, passed as
``0.0 = UNFUNDED``, and (c) the prefill activation reserve, which had no field
at all.  An unpriced term does not read as "unknown", it reads as "free"
(#1009), and free is the OVER-pricing direction on every one of them.

#1019 IS THE PRECEDENT AND THE REASON THIS FILE EXISTS: a pool metric that is
not the boot's own arithmetic ranked three cuts in the exact reverse of the
serving world.  ``pp_cut.world_kv_floor``'s docstring closes with "Use the boot
sizing formula for anything the boot will then size."  This is that formula,
and it is ONE function -- ``pp_cut.pp_phase_pool`` -- used by every candidate of
every objective, so there is no second bookkeeping to drift.

TWO METAL POINTS, TWO DIFFERENT CUTS, ONE MODEL.  The fixtures below are the
emitted numbers of two boots that ran DIFFERENT cuts, which is what makes the
per-stage fixed post falsifiable rather than fitted: it is recovered
independently from each boot and the two answers agree to <= 12.7 MiB.

Hermetic: no GPU, no NVML, no checkpoint, no server.
"""

import math
import os
import unittest

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

from sglang.srt.planner.pp_cut import (
    LAYER_FAMILY_ATTENTION,
    LAYER_FAMILY_LINEAR,
    PhasePoolModel,
    attention_counts,
    gapped_phase_pool,
    pp_phase_pool,
    stage_pp_capacities,
)
from sglang.test.test_utils import CustomTestCase

GIB = 1024.0

# ---------------------------------------------------------------------------
# THE TWO BOOTS, verbatim from their own logs.
# ---------------------------------------------------------------------------

#: Group P's per-rank budgets, from the launcher's own PP-CUT inputs line of
#: boot weg2sb5f: ``free=[27960, 17064, 16552] MiB``.  Identical on weg2rg6.
BUDGETS_MIB = (27960.0, 17064.0, 16552.0)

#: ``weights attn 355.1 / linear 366.2 MiB per layer -> mean 363.4 used``
#: (same line).  The mean is the model's own deliberate choice; keeping it
#: makes this test about the POSTS and not about the weight split.
#:
#: THE LOG ROUNDS IT TO ONE DECIMAL and the launcher does not, so reproducing
#: the shipped pool from this fixture is exact to about 5 tokens on the binding
#: stage (0.086 MiB of weight over 44 layers, 4 tokens at an 11-layer cell) --
#: which is why the two "reproduce the shipped number" assertions below carry
#: ``REPRO_TOKENS`` and not ``assertEqual``.  Asserting the last digit here
#: would be asserting precision the printed input does not have.
MEAN_LAYER_MIB = 363.4

#: Slack for reproducing a SHIPPED pool figure from the rounded log inputs.
REPRO_TOKENS = 25

#: ``kv=2048 B/token/attn-layer (from config, fp8_e4m3)`` -> MiB.
KV_MIB_PER_TOKEN_PER_ATTN_LAYER = 2048.0 / (1024.0 * 1024.0)

#: The two constants the runtime charges per rank, read off its own posts
#: line: ``gapped corridor holdback=1.000`` and ``prefill activation
#: reserve=1.000``, both exactly 1 GiB on both boots.
CORRIDOR_HOLDBACK_MIB = 1024.0
ACTIVATION_RESERVE_MIB = 1024.0

#: ``[auto-mamba] ... max_mamba_cache_size=20 slots`` on every rank of both
#: boots (``ceil(target_concurrency 8 * ratio 2 * safety 1.25)``,
#: model_runner_kv_cache_mixin.py:2088).
MAMBA_SLOTS = 20

#: Recovered from ``mamba state pool`` / (linear layers * slots) on six rank
#: readings across the two boots: 31.185 / 31.23 / 31.16 and 31.19 / 31.16 /
#: 31.13 MiB per linear layer, i.e. 1.5588 MiB per linear layer per slot.
MAMBA_MIB_PER_LINEAR_LAYER_PER_SLOT = 1.5588

#: Per-stage FIXED post: everything in ``weights + runtime state`` that is not
#: ``mean_layer_mib * n_layers`` -- the embedding on stage 0, lm_head plus the
#: MTP/draft head on the last stage, the replicated payload every stage
#: carries, and the per-rank runtime state.  Recovered SEPARATELY from each
#: boot (see ``test_the_fixed_post_is_cut_invariant``); these are the means.
STAGE_FIXED_MIB = (2342.0, 1105.5, 3518.0)

#: weg2sb5f: the shipped makespan cut and the world pool it actually sized.
SB5F_COUNTS = (44, 10, 10)
SB5F_ATTN = (11, 2, 3)
SB5F_WEIGHTS_RUNTIME_GIB = (17.908, 4.623, 6.982)
SB5F_MAMBA_GIB = (1.005, 0.244, 0.213)
SB5F_REST_GIB = (6.392, 9.797, 6.969)
SB5F_AVAILABLE_BYTES = (6863273984, 10519969792, 7482425344)
SB5F_CELL_BYTES = (22528, 4096, 6144)
SB5F_LOCAL_TOKENS = (304655, 2568352, 1217842)
SB5F_WORLD_TOKENS = 304655
#: What the launcher printed for the same cut on the same boot.
SB5F_PRICED_BEFORE_THE_FIX = 499967

#: weg2rg6 / weg2sb4: the incumbent cut, a genuinely different layout.
RG6_COUNTS = (32, 18, 14)
RG6_ATTN = (8, 4, 4)
RG6_WEIGHTS_RUNTIME_GIB = (13.637, 7.473, 8.406)
RG6_MAMBA_GIB = (0.731, 0.426, 0.304)
RG6_REST_GIB = (10.937, 6.765, 5.453)
RG6_CELL_BYTES = (16384, 8192, 8192)
RG6_LOCAL_TOKENS = (716792, 886732, 714788)
RG6_WORLD_TOKENS = 714788
#: The incumbent row of weg2sb5f's own PP-CUT SHIPPED line.
RG6_PRICED_BEFORE_THE_FIX = 966544

#: 64 layers, every 4th a full-attention layer -- the checkpoint's
#: ``layer_types``, and the pattern both cuts' attention counts fall out of.
FAMILIES = tuple(
    LAYER_FAMILY_ATTENTION if (i % 4) == 3 else LAYER_FAMILY_LINEAR
    for i in range(64)
)


def funded_model(budgets=BUDGETS_MIB) -> PhasePoolModel:
    """The pool model with EVERY post the boot charges, funded."""
    return PhasePoolModel(
        free_mib=tuple(budgets),
        weight_mib_per_layer=MEAN_LAYER_MIB,
        kv_mib_per_token_per_attn_layer=KV_MIB_PER_TOKEN_PER_ATTN_LAYER,
        arming_floor_mib=tuple(1229.0 for _ in budgets),
        mamba_mib_per_linear_layer_per_slot=MAMBA_MIB_PER_LINEAR_LAYER_PER_SLOT,
        mamba_slots=MAMBA_SLOTS,
        stage_fixed_mib=STAGE_FIXED_MIB,
        activation_reserve_mib=ACTIVATION_RESERVE_MIB,
        corridor_holdback_mib=CORRIDOR_HOLDBACK_MIB,
    )


def unfunded_model(budgets=BUDGETS_MIB) -> PhasePoolModel:
    """The model AS THE LAUNCHER BUILT IT on weg2sb5f -- the defect itself."""
    return PhasePoolModel(
        free_mib=tuple(budgets),
        weight_mib_per_layer=MEAN_LAYER_MIB,
        kv_mib_per_token_per_attn_layer=KV_MIB_PER_TOKEN_PER_ATTN_LAYER,
        arming_floor_mib=tuple(1229.0 for _ in budgets),
        mamba_mib_per_linear_layer_per_slot=0.0,
        mamba_slots=8,
    )


def _hand_capacity(model: PhasePoolModel, counts, attn, r: int) -> float:
    """One rank's capacity, written out here rather than called out to.

    A SECOND SOURCE on purpose: if this and ``stage_pp_capacities`` were the
    same code the test would only prove the code equals itself.  This is the
    boot's post list transcribed as arithmetic, in the order the runtime
    subtracts it.
    """
    linear = int(counts[r]) - int(attn[r])
    available_mib = (
        float(model.free_mib[r])
        - float(model.weight_mib_per_layer) * int(counts[r])
        - (float(model.stage_fixed_mib[r]) if model.stage_fixed_mib else 0.0)
        - (
            float(model.corridor_holdback_mib)
            if model.corridor_holdback_mib is not None
            else float(model.arming_floor_mib[r])
        )
        - float(model.mamba_mib_per_linear_layer_per_slot)
        * linear
        * int(model.mamba_slots)
        - float(model.activation_reserve_mib)
    )
    cell_bytes = int(attn[r]) * 2048
    return float(math.floor(available_mib * 1024.0 * 1024.0 / cell_bytes))


class TestTheFixtureIsTheBootsOwnArithmetic(CustomTestCase):
    """Before trusting the model, prove the fixture reproduces the metal.

    Every number below is emitted; nothing here exercises the model.  If this
    class fails, the fixture was transcribed wrong and no verdict from the
    rest of the file means anything.
    """

    def test_the_cell_is_the_attention_count_times_2048(self):
        for cell, attn in zip(SB5F_CELL_BYTES, SB5F_ATTN):
            self.assertEqual(cell, attn * 2048)
        for cell, attn in zip(RG6_CELL_BYTES, RG6_ATTN):
            self.assertEqual(cell, attn * 2048)

    def test_available_bytes_over_the_cell_is_the_local_capacity(self):
        for avail, cell, tokens in zip(
            SB5F_AVAILABLE_BYTES, SB5F_CELL_BYTES, SB5F_LOCAL_TOKENS
        ):
            self.assertEqual(avail // cell, tokens)

    def test_the_world_is_the_min_over_ranks(self):
        self.assertEqual(min(SB5F_LOCAL_TOKENS), SB5F_WORLD_TOKENS)
        self.assertEqual(min(RG6_LOCAL_TOKENS), RG6_WORLD_TOKENS)

    def test_the_posts_reconcile_budget_to_rest(self):
        """``budget - sum(posts) == rest``, the sizer's own identity."""
        for boot, counts, w, mamba, rest in (
            ("sb5f", SB5F_COUNTS, SB5F_WEIGHTS_RUNTIME_GIB, SB5F_MAMBA_GIB,
             SB5F_REST_GIB),
            ("rg6", RG6_COUNTS, RG6_WEIGHTS_RUNTIME_GIB, RG6_MAMBA_GIB,
             RG6_REST_GIB),
        ):
            for r in range(3):
                expected = (
                    BUDGETS_MIB[r] / GIB
                    - w[r]
                    - CORRIDOR_HOLDBACK_MIB / GIB
                    - mamba[r]
                    - ACTIVATION_RESERVE_MIB / GIB
                )
                self.assertAlmostEqual(
                    expected, rest[r], delta=0.002,
                    msg=f"{boot} rank{r}: posts do not reconcile to rest",
                )

    def test_the_attention_counts_fall_out_of_the_layer_pattern(self):
        self.assertEqual(attention_counts(FAMILIES, SB5F_COUNTS), SB5F_ATTN)
        self.assertEqual(attention_counts(FAMILIES, RG6_COUNTS), RG6_ATTN)

    def test_the_fixed_post_is_cut_invariant(self):
        """Recovered from EACH boot separately; the two must agree.

        This is what makes ``STAGE_FIXED_MIB`` a measurement and not a fit: the
        two boots ran 44,10,10 and 32,18,14, so a term that tracked layer count
        would move by thousands of MiB between them.
        """
        for r in range(3):
            from_sb5f = SB5F_WEIGHTS_RUNTIME_GIB[r] * GIB - MEAN_LAYER_MIB * SB5F_COUNTS[r]
            from_rg6 = RG6_WEIGHTS_RUNTIME_GIB[r] * GIB - MEAN_LAYER_MIB * RG6_COUNTS[r]
            self.assertLess(
                abs(from_sb5f - from_rg6), 13.0,
                msg=f"rank{r}: fixed post {from_sb5f:.1f} vs {from_rg6:.1f} MiB",
            )
            self.assertAlmostEqual(
                STAGE_FIXED_MIB[r], 0.5 * (from_sb5f + from_rg6), delta=1.0
            )


class TestBootSizingFormula(CustomTestCase):
    """The model, against the two metal points."""

    #: 0.5 % -- the fixed post's own two-boot spread is 12.7 MiB, about 0.2 %
    #: of the binding stage's available bytes, so a tighter bound would be
    #: asserting precision the input does not have.
    TOLERANCE = 0.005

    def _assert_prices(self, counts, attn, metal):
        priced = pp_phase_pool(counts, attn, funded_model())
        rel = abs(priced - metal) / float(metal)
        self.assertLess(
            rel, self.TOLERANCE,
            msg=f"cut {counts}/{attn}: priced {priced:.0f}, metal {metal} "
                f"({100.0 * rel:+.2f} %)",
        )

    def test_makespan_cut_is_priced_at_the_boot_number(self):
        self._assert_prices(SB5F_COUNTS, SB5F_ATTN, SB5F_WORLD_TOKENS)

    def test_incumbent_cut_is_priced_at_the_boot_number(self):
        self._assert_prices(RG6_COUNTS, RG6_ATTN, RG6_WORLD_TOKENS)

    def test_per_rank_capacities_match_the_boots_per_rank_capacities(self):
        """Not only the min: every rank, so a right answer for a wrong reason
        (two errors cancelling on the binder) cannot pass."""
        model = funded_model()
        caps = stage_pp_capacities(SB5F_COUNTS, SB5F_ATTN, model)
        for r, (got, metal) in enumerate(zip(caps, SB5F_LOCAL_TOKENS)):
            rel = abs(got - metal) / float(metal)
            self.assertLess(
                rel, self.TOLERANCE,
                msg=f"rank{r}: priced {got:.0f} vs metal {metal}",
            )
            # EXACT against the second source, so a rounding that goes the
            # wrong way (ceil, round-half-up) cannot hide inside the 0.5 %
            # band the fixed post's own spread earns.
            self.assertEqual(
                got, _hand_capacity(model, SB5F_COUNTS, SB5F_ATTN, r),
                msg=f"rank{r}: the model and the transcribed post list differ",
            )

    def test_the_unfunded_model_is_the_measured_over_pricing(self):
        """THE RED HALF: reproduce the defect exactly as it shipped."""
        self.assertAlmostEqual(
            pp_phase_pool(SB5F_COUNTS, SB5F_ATTN, unfunded_model()),
            SB5F_PRICED_BEFORE_THE_FIX,
            delta=REPRO_TOKENS,
        )
        self.assertAlmostEqual(
            pp_phase_pool(RG6_COUNTS, RG6_ATTN, unfunded_model()),
            RG6_PRICED_BEFORE_THE_FIX,
            delta=REPRO_TOKENS,
        )
        # ... and it is over-pricing, the dangerous direction, on both cuts.
        self.assertGreater(SB5F_PRICED_BEFORE_THE_FIX, 1.6 * SB5F_WORLD_TOKENS)
        self.assertGreater(RG6_PRICED_BEFORE_THE_FIX, 1.3 * RG6_WORLD_TOKENS)

    def test_the_pool_is_floored_like_the_boot(self):
        """The boot does ``int(available_bytes) // cell``; so does this."""
        got = pp_phase_pool(SB5F_COUNTS, SB5F_ATTN, funded_model())
        self.assertEqual(got, math.floor(got))


class TestEveryPostIsLoadBearing(CustomTestCase):
    """Drop ONE post and the price must rise -- the mutant direction.

    Over-pricing is the dangerous half: it lets a cut through the
    ``pool >= --max-kv-per-request`` floor that the boot cannot hold, and it
    understates what an objective is trading away.  Each subtest is the
    hand-written twin of one mutant.
    """

    def _price(self, **overrides):
        import dataclasses

        return pp_phase_pool(
            SB5F_COUNTS, SB5F_ATTN, dataclasses.replace(funded_model(), **overrides)
        )

    def setUp(self):
        self.baseline = pp_phase_pool(SB5F_COUNTS, SB5F_ATTN, funded_model())

    def test_dropping_the_stage_fixed_post_over_prices(self):
        self.assertGreater(self._price(stage_fixed_mib=()), self.baseline)

    def test_dropping_the_activation_reserve_over_prices(self):
        self.assertGreater(self._price(activation_reserve_mib=0.0), self.baseline)

    def test_dropping_the_mamba_pool_over_prices(self):
        self.assertGreater(
            self._price(mamba_mib_per_linear_layer_per_slot=0.0), self.baseline
        )

    def test_falling_back_to_the_arming_floor_under_prices_by_205_mib(self):
        """The one term that errs the SAFE way, so it is asserted by sign.

        The boot charges a 1024 MiB corridor holdback; the model's stand-in was
        the 1229 MiB arming floor.  Using the arming floor therefore costs 205
        MiB per rank the boot does not spend -- conservative, and named rather
        than left as a coincidence.
        """
        self.assertLess(self._price(corridor_holdback_mib=None), self.baseline)

    def test_every_post_together_is_the_shipped_defect(self):
        got = self._price(
            stage_fixed_mib=(),
            activation_reserve_mib=0.0,
            mamba_mib_per_linear_layer_per_slot=0.0,
            corridor_holdback_mib=None,
        )
        self.assertAlmostEqual(got, SB5F_PRICED_BEFORE_THE_FIX, delta=REPRO_TOKENS)


class TestUnfundedPostsAreNamedNotSilent(CustomTestCase):
    """#1009: an unpriced term reads as free, so it must say its own name."""

    def test_the_defect_model_names_all_three(self):
        names = unfunded_model().unfunded_posts
        self.assertIn("stage_fixed_mib", names)
        self.assertIn("activation_reserve_mib", names)
        self.assertIn("corridor_holdback_mib", names)

    def test_the_funded_model_names_none(self):
        self.assertEqual(funded_model().unfunded_posts, ())

    def test_the_launcher_seam_refuses_an_unfunded_model(self):
        from sglang.srt.planner.pp_cut_launch import PPCutRefused, refuse_unfunded_posts

        refuse_unfunded_posts(funded_model())  # must not raise
        with self.assertRaises(PPCutRefused) as cm:
            refuse_unfunded_posts(unfunded_model())
        message = str(cm.exception)
        self.assertIn("W40", message)
        self.assertIn("stage_fixed_mib", message)


class TestKvLessRankIsRefusedNotPriced(CustomTestCase):
    """#1255: a rank with no full-attention layer must never carry a number.

    The runtime's own artifact is the reason: with ``cell_size=0`` the boot
    prints ``max_total_num_tokens=1048576``, and a solver that priced such a
    stage would rank a division that never happened (measured on the gapped
    probe, /spinning/gpu-arb/weg2 gapped refutation, #1240).
    """

    def test_contiguous_refuses(self):
        with self.assertRaises(ValueError) as cm:
            pp_phase_pool((44, 10, 10), (11, 0, 5), funded_model())
        self.assertIn("no full-attention layer", str(cm.exception))

    def test_gapped_excludes_the_kv_less_stage_from_the_min(self):
        """Excluded from the reduction, not priced high and then out-ranked."""
        model = funded_model()
        counts, attn = (44, 10, 10), (11, 0, 5)
        got = gapped_phase_pool(counts, attn, model)
        self.assertEqual(got, min(_hand_capacity(model, counts, attn, r)
                                  for r in (0, 2)))
        self.assertNotEqual(int(got), 1048576)

    def test_gapped_refuses_a_map_with_no_attention_anywhere(self):
        with self.assertRaises(ValueError):
            gapped_phase_pool((44, 10, 10), (0, 0, 0), funded_model())


class TestOneFunctionPricesEveryObjective(CustomTestCase):
    """No objective may carry a pool number a different function produced.

    The #1019 lesson in one assertion: the ranking and the boot must be the
    same arithmetic.  Every row the decision publishes -- the chosen cut, the
    kv-floor row, the makespan row and every ranked alternative -- is checked
    against ``pp_phase_pool`` for its OWN counts.
    """

    def _decide(self, objective: str):
        from sglang.srt.planner.pp_cut_launch import solve_launch_cut

        return solve_launch_cut(
            layer_families=FAMILIES,
            incumbent_layers=RG6_COUNTS,
            measured_ms_per_layer=(8.10, 35.16, 33.59),
            measured_provenance="test fixture",
            card_names=["NVIDIA GeForce RTX 5090", "NVIDIA GeForce RTX 3080",
                        "NVIDIA GeForce RTX 3080"],
            pool_model=funded_model(),
            cap_tokens=262144,
            enumerate_gapped=False,
            objective=objective,
        )

    def test_every_published_row_matches_the_one_function(self):
        for objective in ("maxkv", "makespan"):
            decision = self._decide(objective)
            rows = [decision.chosen, decision.kv_floor, decision.makespan]
            rows += list(decision.ranked[:20])
            for row in rows:
                if row is None or row.kind != "contiguous":
                    continue
                self.assertEqual(
                    float(row.pool_tokens),
                    pp_phase_pool(row.layers, row.attn, funded_model()),
                    msg=f"objective={objective} row {row.layers}/{row.attn} "
                        f"carries a pool the boot-sizing formula did not "
                        f"produce",
                )

    def test_the_shipped_cut_is_ranked_at_the_metal_number(self):
        """The whole point of #1286, end to end through the decision object.

        The row is looked up by its COUNTS rather than asserted to be the
        winner: which cut wins the time axis depends on the card-rate library
        and the depth model, neither of which #1286 touches.
        """
        decision = self._decide("makespan")
        row = next(
            (c for c in decision.ranked
             if c.kind == "contiguous" and c.layers == SB5F_COUNTS),
            None,
        )
        self.assertIsNotNone(row, "the shipped makespan cut was not ranked")
        self.assertEqual(row.attn, SB5F_ATTN)
        rel = abs(row.pool_tokens - SB5F_WORLD_TOKENS) / float(SB5F_WORLD_TOKENS)
        self.assertLess(rel, 0.005, msg=f"{100.0 * rel:+.2f} % off metal")


if __name__ == "__main__":
    unittest.main()
