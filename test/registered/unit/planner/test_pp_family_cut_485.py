"""Hermetic tests for the planner-solved per-family PP cut (#485).

No GPU, no checkpoint, no probe run: the rig is the pinned card-probe
artifact and the geometry is the reference checkpoint's config arithmetic,
both inlined so a re-probe or a model-cache change cannot silently move a
regression number.

The falsifier is ``test_anti_proportional_cut_is_strictly_worse``: an
attention split skewed AWAY from the fast card must score strictly worse
than the solved cut. Without it, "the solver produced a skewed cut" is not
evidence that the skew is the right one.

PROVENANCE, #910 (checked 2026-08-26, no code change owed). The reference
checkpoint named throughout this file --
``/spinning/llm_stuff/club-3090/models-cache/Qwen3.6-27B-INT8-W8A8-yarn1.5``
-- is GONE from this box. That is deliberately not a skip here, and this
file is the one site of the four in that drift where it is not: the first
line above is literal, and it was verified rather than trusted. There is no
path, no ``model_path`` and no filesystem touch anywhere in this module; the
geometry constants below and ``TestCheckpointConservation``'s measured
safetensors constants are INLINED, which is precisely why "a model-cache
change cannot silently move a regression number" -- and the model cache has
now changed. 64 passed hermetically with the checkpoint absent.

What the absence DOES cost is re-derivation: the constants in
``TestCheckpointConservation`` (measured 2026-08-12) can no longer be
re-measured from the headers they came from, so this file is now their only
record. A future edit to those numbers therefore cannot be checked against
the checkpoint and must be treated as a new specimen with its own
measurement, not as a correction of these.
"""

import dataclasses
import itertools
import unittest

from sglang.srt.distributed.utils import derive_pp_layer_split
from sglang.srt.planner import pp_cut
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


# ---------------------------------------------------------------------------
# Reference geometry: Qwen3.6-27B, from its config.json
# ---------------------------------------------------------------------------

N_LAYERS = 64
FULL_ATTENTION_INTERVAL = 4
HIDDEN = 5120
Q_HEADS = 24
KV_HEADS = 4
HEAD_DIM = 256
INTERMEDIATE = 17408
GDN_K_HEADS, GDN_K_DIM = 16, 128
GDN_V_HEADS, GDN_V_DIM = 48, 128
CONV_KERNEL = 4

#: fp8_e4m3 KV: 4 kv heads x 256 head_dim x 2 (K and V) x 1 byte = 2048 B.
#: x 16 full-attention layers = the 32 KiB/token the 631 bench log validates
#: against two independent boots (PROD_BRINGUP_BENCH.md sec. 2).
KV_BYTES_PER_TOKEN_PER_ATTN_LAYER = KV_HEADS * HEAD_DIM * 2 * 1

#: Weight params of one layer of each family, from the same formulas the
#: cost model uses (uneven_perf.PerfCostModel._build_families:4011-4044).
_ATTN_PROJ = (
    HIDDEN * (Q_HEADS * HEAD_DIM + 2 * KV_HEADS * HEAD_DIM)
    + (Q_HEADS * HEAD_DIM) * HIDDEN
)
_MLP = 3 * HIDDEN * INTERMEDIATE
_K_SZ = GDN_K_HEADS * GDN_K_DIM
_V_SZ = GDN_V_HEADS * GDN_V_DIM
_GDN = (
    HIDDEN * (2 * _K_SZ + 2 * _V_SZ)
    + HIDDEN * 2 * GDN_V_HEADS
    + _V_SZ * HIDDEN
    + (2 * _K_SZ + _V_SZ) * CONV_KERNEL
)

#: fp8 checkpoint -> 1 byte per param.
BYTES_PER_PARAM = 1.0
ATTN_LAYER_BYTES = (_ATTN_PROJ + _MLP) * BYTES_PER_PARAM
LINEAR_LAYER_BYTES = (_GDN + _MLP) * BYTES_PER_PARAM

#: 2 FLOPs per param per token for the dense projections and MLP.
ATTN_LAYER_FLOPS_PER_TOKEN = 2.0 * (_ATTN_PROJ + _MLP)
LINEAR_LAYER_FLOPS_PER_TOKEN = 2.0 * (_GDN + _MLP)
#: QK^T and A@V, per query token per KV depth token, per attention layer.
ATTN_CORE_FLOPS_PER_TOKEN_PAIR = 4.0 * Q_HEADS * HEAD_DIM


# ---------------------------------------------------------------------------
# Reference rig: the pinned card probe (test_key_solver._PROBE)
# ---------------------------------------------------------------------------
#
# IdentityMap for this rig: PP rank 0 is the 5090; nvidia-smi index 0 is a
# 3080. Stage order below is rank order, so stage 0 is the 5090.

RANK0_5090 = dict(gemm_tflops=231.97, attn_bw_gbs=1533.8, total_mib=32607)
RANK1_3080 = dict(gemm_tflops=65.57, attn_bw_gbs=717.4, total_mib=20480)
RANK2_3080 = dict(gemm_tflops=65.59, attn_bw_gbs=717.8, total_mib=20480)

#: Measured per-rank transients at the 111405-token trigger, in RANK order
#: (PROD_BRINGUP_BENCH.md sec. 1f reports them in nvidia-smi order
#: rank1/rank0/rank2 = 1120/1346/982).
TRANSIENT_MIB = (1346.0, 1120.0, 982.0)

#: The ship config's per-rank budgets.
SHIP_BUDGET_MIB = (31800.0, 17400.0, 17450.0)


def _families():
    return pp_cut.layer_families_from_config(
        {
            "num_hidden_layers": N_LAYERS,
            "full_attention_interval": FULL_ATTENTION_INTERVAL,
        }
    )


def _inputs(
    *,
    depth=179000,
    chunk=2048,
    budgets=SHIP_BUDGET_MIB,
    transients=TRANSIENT_MIB,
    corridor=1024.0,
    bw=None,
    tflops=None,
    pool=0,
    tp_token_shares=None,
    overheads=(0.0, 0.0, 0.0),
    seam_staging=(0.0, 0.0, 0.0),
    draft_residency=(None, None, None),
    draft_runner_present=False,
):
    cards = (RANK0_5090, RANK1_3080, RANK2_3080)
    bw = bw or tuple(c["attn_bw_gbs"] for c in cards)
    tflops = tflops or tuple(c["gemm_tflops"] for c in cards)
    ranks = tuple(
        pp_cut.RankResources(
            label=label,
            attn_bw_gbs=bw[i],
            gemm_tflops=tflops[i],
            budget_mib=budgets[i],
            transient_mib=transients[i],
            fixed_overhead_mib=overheads[i],
            seam_staging_mib=seam_staging[i],
            draft_residency_mib=draft_residency[i],
        )
        for i, label in enumerate(("rank0-5090", "rank1-3080", "rank2-3080"))
    )
    return pp_cut.PPCutInputs(
        layer_families=_families(),
        attn_layer_weight_bytes=ATTN_LAYER_BYTES,
        linear_layer_weight_bytes=LINEAR_LAYER_BYTES,
        attn_layer_flops_per_token=ATTN_LAYER_FLOPS_PER_TOKEN,
        linear_layer_flops_per_token=LINEAR_LAYER_FLOPS_PER_TOKEN,
        attn_core_flops_per_token_pair=ATTN_CORE_FLOPS_PER_TOKEN_PAIR,
        kv_bytes_per_token_per_attn_layer=KV_BYTES_PER_TOKEN_PER_ATTN_LAYER,
        kv_depth_tokens=depth,
        prefill_chunk_tokens=chunk,
        ranks=ranks,
        kv_pool_tokens=pool,
        tp_token_shares=tp_token_shares,
        corridor_mib=corridor,
        draft_runner_present=draft_runner_present,
    )


def _brute_force_best(inputs, require_attention_per_stage=True):
    """Every contiguous split, priced. The solver must match this exactly."""
    n, k = inputs.n_layers, inputs.pp_size
    hybrid = 0 < inputs.n_full_attention < n
    best = None
    for cuts in itertools.combinations(range(1, n), k - 1):
        bounds = list(cuts) + [n]
        counts = [bounds[0]] + [bounds[i] - bounds[i - 1] for i in range(1, k)]
        sol, violations = pp_cut.validate_pp_cut(
            counts, inputs, require_attention_per_stage=require_attention_per_stage
        )
        if violations:
            continue
        if require_attention_per_stage and hybrid and 0 in sol.attention_counts:
            continue
        if best is None or sol.makespan_seconds < best.makespan_seconds:
            best = sol
    return best


class TestReferenceGeometry(CustomTestCase):
    """Pin the geometry model against the numbers measured on metal."""

    def test_family_map_matches_checkpoint_rule(self):
        fams = _families()
        self.assertEqual(len(fams), N_LAYERS)
        attn_idx = [i for i, f in enumerate(fams) if f == pp_cut.LAYER_FAMILY_ATTENTION]
        self.assertEqual(len(attn_idx), 16)
        self.assertEqual(attn_idx[:3], [3, 7, 11])
        self.assertEqual(attn_idx[-1], 63)

    def test_kv_per_token_matches_measured_32_kib(self):
        node_wide = 16 * KV_BYTES_PER_TOKEN_PER_ATTN_LAYER
        self.assertEqual(node_wide, 32 * 1024)

    def test_ship_split_attention_census(self):
        """[28,20,16] holds 7/5/4 full-attention layers, as the flip plan
        records (layers/dcp/phase_flip_plan.py:93-97)."""
        self.assertEqual(pp_cut.attention_counts(_families(), [28, 20, 16]), (7, 5, 4))
        self.assertEqual(pp_cut.attention_counts(_families(), [32, 16, 16]), (8, 4, 4))

    def test_explicit_layer_types_are_honoured(self):
        types = ["linear_attention"] * 3 + ["full_attention"]
        fams = pp_cut.layer_families_from_config(
            {"num_hidden_layers": 4, "layer_types": types}
        )
        self.assertEqual(fams[3], pp_cut.LAYER_FAMILY_ATTENTION)
        self.assertEqual(fams[0], pp_cut.LAYER_FAMILY_LINEAR)

    def test_layer_types_length_mismatch_refuses(self):
        with self.assertRaises(ValueError):
            pp_cut.layer_families_from_config(
                {"num_hidden_layers": 8, "layer_types": ["full_attention"] * 4}
            )


class TestAttentionRoofline(CustomTestCase):
    """The attention core is compute-bound in prefill, not bandwidth-bound."""

    def test_prefill_attention_is_compute_bound(self):
        sol, violations = pp_cut.validate_pp_cut([28, 20, 16], _inputs())
        self.assertEqual(violations, ())
        for stage in sol.stages:
            self.assertEqual(
                stage.attn_bound_by,
                "compute",
                f"{stage.rank} attention core should be compute-bound at a "
                f"2048-token prefill chunk",
            )

    def test_decode_shaped_chunk_flips_to_bandwidth(self):
        """The roofline is real: a chunk of 1 lands on the other side."""
        sol, _ = pp_cut.validate_pp_cut([28, 20, 16], _inputs(chunk=1))
        for stage in sol.stages:
            self.assertEqual(stage.attn_bound_by, "bandwidth")

    def test_intensity_is_depth_independent(self):
        """Depth scales both sides of the roofline equally, so the binding
        side cannot change with depth -- only with chunk."""
        for depth in (4096, 179000, 393216):
            sol, _ = pp_cut.validate_pp_cut([28, 20, 16], _inputs(depth=depth))
            self.assertEqual(
                {s.attn_bound_by for s in sol.stages}, {"compute"}, f"depth={depth}"
            )


class TestSolver(CustomTestCase):
    def test_matches_brute_force(self):
        """Exactness, not a heuristic: with the headroom slack switched off
        the DP equals exhaustive search."""
        inputs = _inputs()
        solved = pp_cut.solve_pp_cut(inputs, makespan_slack=1.0)
        brute = _brute_force_best(inputs)
        self.assertTrue(solved.feasible)
        self.assertIsNotNone(brute)
        self.assertAlmostEqual(
            solved.makespan_seconds, brute.makespan_seconds, places=12
        )

    def test_default_slack_stays_within_its_budget(self):
        """The default may trade a little speed for headroom, but never
        more than it advertises, and never less headroom."""
        inputs = _inputs()
        exact = pp_cut.solve_pp_cut(inputs, makespan_slack=1.0)
        default = pp_cut.solve_pp_cut(inputs)
        self.assertLessEqual(
            default.makespan_seconds,
            exact.makespan_seconds * pp_cut._MAKESPAN_SLACK + 1e-12,
        )
        self.assertGreaterEqual(default.min_headroom_mib, exact.min_headroom_mib - 1e-9)

    def test_slack_below_one_is_refused(self):
        with self.assertRaises(ValueError):
            pp_cut.solve_pp_cut(_inputs(), makespan_slack=0.9)

    def test_matches_brute_force_under_tight_memory(self):
        inputs = _inputs(budgets=(31800.0, 15000.0, 15000.0))
        solved = pp_cut.solve_pp_cut(inputs)
        brute = _brute_force_best(inputs)
        if brute is None:
            self.assertFalse(solved.feasible)
            return
        self.assertTrue(solved.feasible)
        self.assertAlmostEqual(
            solved.makespan_seconds, brute.makespan_seconds, places=12
        )

    def test_cut_is_skewed_toward_the_fast_card(self):
        """The solved cut must give the 5090 strictly more than a uniform
        share of BOTH families."""
        sol = pp_cut.solve_pp_cut(_inputs())
        self.assertTrue(sol.feasible, sol.refusals)
        uniform_layers = N_LAYERS / 3.0
        uniform_attn = 16 / 3.0
        self.assertGreater(sol.counts[0], uniform_layers)
        self.assertGreater(sol.attention_counts[0], uniform_attn)

    def test_anti_proportional_cut_is_strictly_worse(self):
        """FALSIFIER. Mirror the solved attention split onto the slow cards.
        If that scored as well, the objective would not be measuring
        anything."""
        inputs = _inputs()
        sol = pp_cut.solve_pp_cut(inputs)
        self.assertTrue(sol.feasible, sol.refusals)

        anti_counts = list(reversed(sol.counts))
        anti, violations = pp_cut.validate_pp_cut(anti_counts, inputs)
        # The mirrored cut may itself be infeasible on memory; that is a
        # strictly worse outcome too, and the test accepts it as such.
        if violations:
            self.assertFalse(anti.feasible)
            return
        self.assertGreater(
            anti.makespan_seconds,
            sol.makespan_seconds,
            "the anti-proportional cut must be strictly slower than the solved cut",
        )

    def test_every_alternative_cut_is_at_least_as_slow(self):
        """Optimality over the whole representable space, stated directly."""
        inputs = _inputs()
        sol = pp_cut.solve_pp_cut(inputs)
        worse = 0
        for cuts in itertools.combinations(range(1, N_LAYERS), 2):
            bounds = list(cuts) + [N_LAYERS]
            counts = [bounds[0], bounds[1] - bounds[0], bounds[2] - bounds[1]]
            other, violations = pp_cut.validate_pp_cut(counts, inputs)
            if violations:
                continue
            self.assertGreaterEqual(
                other.makespan_seconds, sol.makespan_seconds - 1e-15
            )
            if other.makespan_seconds > sol.makespan_seconds:
                worse += 1
        self.assertGreater(worse, 0, "no strictly worse cut exists to compare against")

    def test_determinism(self):
        a = pp_cut.solve_pp_cut(_inputs())
        b = pp_cut.solve_pp_cut(_inputs())
        self.assertEqual(a.counts, b.counts)

    def test_bottleneck_is_reported(self):
        sol = pp_cut.solve_pp_cut(_inputs())
        times = [s.total_seconds for s in sol.stages]
        self.assertEqual(sol.bottleneck_stage, times.index(max(times)))
        self.assertAlmostEqual(sol.makespan_seconds, max(times), places=12)

    def test_depth_moves_the_cut_toward_the_fast_card(self):
        """The attention term grows with depth and the 5090 is the faster
        card on it, so more depth must not move attention AWAY from rank 0."""
        shallow = pp_cut.solve_pp_cut(_inputs(depth=2048))
        deep = pp_cut.solve_pp_cut(_inputs(depth=393216))
        self.assertGreaterEqual(deep.attention_counts[0], shallow.attention_counts[0])


class TestRefusals(CustomTestCase):
    """Never a silent even split (the #202 lesson)."""

    def test_impossible_budget_refuses_with_named_rank(self):
        sol = pp_cut.solve_pp_cut(_inputs(budgets=(1200.0, 1200.0, 1200.0)))
        self.assertFalse(sol.feasible)
        self.assertTrue(sol.refusals)
        self.assertEqual(sol.counts, ())
        joined = " ".join(sol.refusals)
        self.assertIn("rank0-5090", joined)
        self.assertIn("corridor", joined)

    def test_refusal_names_the_numbers(self):
        sol = pp_cut.solve_pp_cut(
            _inputs(budgets=(9000.0, 9000.0, 9000.0), depth=393216)
        )
        self.assertFalse(sol.feasible)
        joined = " ".join(sol.refusals)
        self.assertIn("MiB", joined)

    def test_more_stages_than_attention_layers_refuses(self):
        fams = tuple([pp_cut.LAYER_FAMILY_LINEAR] * 7 + [pp_cut.LAYER_FAMILY_ATTENTION])
        ranks = tuple(
            pp_cut.RankResources(
                label=f"r{i}",
                attn_bw_gbs=700.0,
                gemm_tflops=60.0,
                budget_mib=1e9,
            )
            for i in range(4)
        )
        inputs = pp_cut.PPCutInputs(
            layer_families=fams,
            attn_layer_weight_bytes=1.0,
            linear_layer_weight_bytes=1.0,
            attn_layer_flops_per_token=1.0,
            linear_layer_flops_per_token=1.0,
            attn_core_flops_per_token_pair=1.0,
            kv_bytes_per_token_per_attn_layer=1.0,
            kv_depth_tokens=16,
            prefill_chunk_tokens=8,
            ranks=ranks,
            corridor_mib=0.0,
        )
        sol = pp_cut.solve_pp_cut(inputs)
        self.assertFalse(sol.feasible)
        self.assertTrue(any("full-attention" in r for r in sol.refusals))

    def test_zero_rate_is_refused_not_defaulted(self):
        with self.assertRaises(ValueError):
            pp_cut.RankResources(
                label="x", attn_bw_gbs=0.0, gemm_tflops=1.0, budget_mib=1.0
            )
        with self.assertRaises(ValueError):
            pp_cut.RankResources(
                label="x", attn_bw_gbs=1.0, gemm_tflops=0.0, budget_mib=1.0
            )


class TestMemoryModel(CustomTestCase):
    """Arena sizing, and the honesty of the feasibility verdict.

    NOTE: ``fixed_overhead_mib`` defaults to 0, which makes the verdict a
    LOWER BOUND on real occupancy. These tests validate the MECHANISM; the
    constant itself is uncalibrated on this rig (see HANDOFF_485_PPCUT.md --
    the residual measured off the 631 at-rest boot is 10171/4982/7582 MiB,
    which is not cut-invariant, so one scalar per rank may not suffice).
    """

    def test_pool_tokens_drive_memory_not_depth(self):
        """A 600k-token arena must not be priced as a 179k request."""
        small = pp_cut.validate_pp_cut([28, 20, 16], _inputs(depth=179000))[0]
        big = pp_cut.validate_pp_cut([28, 20, 16], _inputs(depth=179000, pool=600000))[
            0
        ]
        self.assertGreater(big.stages[0].kv_mib, small.stages[0].kv_mib)
        # ... while the timing term is untouched, because depth is the same.
        self.assertAlmostEqual(
            big.stages[0].attn_seconds, small.stages[0].attn_seconds, places=12
        )

    def test_kv_arena_matches_the_validated_formula(self):
        """``T x 32 KiB x layer_share``, checked against the bench log's own
        measured rows (PROD_BRINGUP_BENCH.md sec. 2). Those rows were taken
        on the ship config ``[32,16,16]`` = attention ``8,4,4`` (sec. 3), and
        they report the K arena alone, which is half of the K+V arena this
        model prices."""
        sol = pp_cut.validate_pp_cut([32, 16, 16], _inputs(depth=179000, pool=540000))[
            0
        ]
        self.assertEqual(sol.attention_counts, (8, 4, 4))
        expected_rank0_gib = 540000 * 32 * 1024 * (8 / 16) / (1024**3)
        self.assertAlmostEqual(
            sol.stages[0].kv_mib / 1024.0, expected_rank0_gib, places=6
        )
        # measured: T=540000 PP rank0 K = 4.12 GiB, rank1/rank2 K = 2.06 GiB.
        self.assertAlmostEqual(sol.stages[0].kv_mib / 1024.0 / 2.0, 4.12, places=2)
        self.assertAlmostEqual(sol.stages[1].kv_mib / 1024.0 / 2.0, 2.06, places=2)
        # measured: T=460000 and T=380000 rank0 K = 3.51 and 2.90 GiB.
        for pool, k_gib in ((460000, 3.51), (380000, 2.90)):
            row = pp_cut.validate_pp_cut([32, 16, 16], _inputs(pool=pool))[0]
            self.assertAlmostEqual(row.stages[0].kv_mib / 1024.0 / 2.0, k_gib, places=2)

    def test_shared_arena_takes_the_max_of_both_layouts(self):
        """With a TP token vector the arena is max(PP share, TP share)."""
        pp_only = pp_cut.validate_pp_cut([28, 20, 16], _inputs(pool=600000))[0]
        shared = pp_cut.validate_pp_cut(
            [28, 20, 16],
            _inputs(pool=600000, tp_token_shares=(0.378, 0.351, 0.270)),
        )[0]
        # rank0's PP share (7/16 = 0.4375) already exceeds its TP share.
        self.assertAlmostEqual(
            shared.stages[0].kv_mib, pp_only.stages[0].kv_mib, places=9
        )
        # rank1's TP share (0.351) exceeds its PP share (5/16 = 0.3125).
        self.assertGreater(shared.stages[1].kv_mib, pp_only.stages[1].kv_mib)

    def test_fixed_overhead_binds_the_constraint(self):
        """The gate is inert only because the constant is zero; supply one
        and it bites."""
        loose = pp_cut.validate_pp_cut([28, 20, 16], _inputs(pool=600000))
        self.assertEqual(loose[1], ())
        tight = pp_cut.validate_pp_cut(
            [28, 20, 16], _inputs(pool=600000, overheads=(10171.0, 4982.0, 7582.0))
        )
        self.assertTrue(tight[1], "measured residuals must make rank1 infeasible")
        self.assertFalse(tight[0].feasible)

    def test_overhead_changes_the_solved_cut(self):
        """A constraint that never changes the answer is not a constraint."""
        free = pp_cut.solve_pp_cut(_inputs(pool=400000))
        loaded = pp_cut.solve_pp_cut(
            _inputs(pool=400000, overheads=(9000.0, 4500.0, 4500.0))
        )
        self.assertTrue(free.feasible, free.refusals)
        if loaded.feasible:
            self.assertNotEqual(free.counts, loaded.counts)
        else:
            self.assertTrue(loaded.refusals)

    def test_tp_token_shares_are_validated(self):
        with self.assertRaises(ValueError):
            _inputs(tp_token_shares=(0.5, 0.5))
        with self.assertRaises(ValueError):
            _inputs(tp_token_shares=(0.5, -0.1, 0.6))


class TestOverrideValidation(CustomTestCase):
    """--pp-layer-ratio stops being a planner bypass."""

    def test_ship_split_validates_clean(self):
        sol, violations = pp_cut.validate_pp_cut([28, 20, 16], _inputs())
        self.assertEqual(violations, ())
        self.assertTrue(sol.feasible)
        self.assertEqual(sol.attention_counts, (7, 5, 4))

    def test_memory_overflow_is_named_with_the_overage(self):
        # Force rank 1 far too small for the 20 layers the ship split gives it.
        sol, violations = pp_cut.validate_pp_cut(
            [28, 20, 16], _inputs(budgets=(31800.0, 6000.0, 17450.0))
        )
        self.assertTrue(violations)
        self.assertFalse(sol.feasible)
        joined = " ".join(violations)
        self.assertIn("rank1-3080", joined)
        self.assertIn("over by", joined)

    def test_zero_attention_stage_is_refused(self):
        # Layers 0..2 carry no full-attention layer (the first is index 3).
        sol, violations = pp_cut.validate_pp_cut([3, 45, 16], _inputs())
        self.assertTrue(any("zero" in v for v in violations))
        self.assertEqual(sol.attention_counts[0], 0)

    def test_wrong_length_and_bad_sum_raise(self):
        with self.assertRaises(ValueError):
            pp_cut.validate_pp_cut([32, 32], _inputs())
        with self.assertRaises(ValueError):
            pp_cut.validate_pp_cut([28, 20, 15], _inputs())
        with self.assertRaises(ValueError):
            pp_cut.validate_pp_cut([28, 0, 36], _inputs())


class TestDecoupling(CustomTestCase):
    """The point of the ticket: attention mass and linear mass move
    independently."""

    def test_linear_layers_move_at_zero_kv_cost(self):
        """[28,20,16] -> [31,17,16] shifts three GDN layers off rank 1 while
        the attention split stays exactly [7,5,4]. No single score vector can
        express this, and the 631 bench log concluded from that that the rig
        could not be levelled from the PP side at all
        (PROD_BRINGUP_BENCH.md sec. 1g)."""
        fams = _families()
        base = pp_cut.attention_counts(fams, [28, 20, 16])
        for counts in ([29, 19, 16], [30, 18, 16], [31, 17, 16]):
            self.assertEqual(
                pp_cut.attention_counts(fams, counts),
                base,
                f"{counts} must not disturb the attention split",
            )

        inputs = _inputs()
        ship, _ = pp_cut.validate_pp_cut([28, 20, 16], inputs)
        moved, _ = pp_cut.validate_pp_cut([31, 17, 16], inputs)
        # KV bytes are untouched on every stage ...
        self.assertEqual(
            [s.kv_mib for s in ship.stages], [s.kv_mib for s in moved.stages]
        )
        # ... while the binding card sheds real weight bytes.
        self.assertGreater(
            ship.stages[1].weight_mib - moved.stages[1].weight_mib, 1000.0
        )

    def test_four_layer_quantisation_is_not_a_hardware_limit(self):
        """Sixteen distinct layer splits hold the attention split at
        [7,5,4]. The 4-layer step is an artifact of deriving both targets
        from one score vector, not a property of the model."""
        fams = _families()
        holding = [
            (b1, b2 - b1, N_LAYERS - b2)
            for b1 in range(1, N_LAYERS)
            for b2 in range(b1 + 1, N_LAYERS)
            if pp_cut.attention_counts(fams, [b1, b2 - b1, N_LAYERS - b2]) == (7, 5, 4)
        ]
        self.assertEqual(len(holding), 16)
        self.assertIn((28, 20, 16), holding)
        self.assertIn((31, 17, 16), holding)


class TestDerivePPLayerSplitDecoupling(CustomTestCase):
    """``derive_pp_layer_split`` gains an independent attention vector."""

    ISFA = [((idx + 1) % FULL_ATTENTION_INTERVAL == 0) for idx in range(N_LAYERS)]

    def test_behaviour_neutral_without_attn_scores(self):
        """PIN: every legacy call is byte-identical. These are the two rows
        the 631 bench log recorded on metal
        (PROD_BRINGUP_BENCH.md sec. 1e)."""
        self.assertEqual(
            derive_pp_layer_split([14, 10, 8], self.ISFA, N_LAYERS), [28, 20, 16]
        )
        self.assertEqual(
            derive_pp_layer_split([15, 9, 8], self.ISFA, N_LAYERS), [32, 16, 16]
        )
        self.assertEqual(
            derive_pp_layer_split([15, 10, 7], self.ISFA, N_LAYERS), [32, 18, 14]
        )

    def test_explicit_none_is_identical_to_omitting(self):
        self.assertEqual(
            derive_pp_layer_split([14, 10, 8], self.ISFA, N_LAYERS, None),
            derive_pp_layer_split([14, 10, 8], self.ISFA, N_LAYERS),
        )

    def test_homogeneous_model_unaffected(self):
        isfa = [True] * 12
        self.assertEqual(derive_pp_layer_split([1, 1, 1], isfa, 12), [4, 4, 4])
        self.assertEqual(
            derive_pp_layer_split([1, 1, 1], isfa, 12, [3, 1, 1]),
            [4, 4, 4],
            "a non-hybrid stack has no attention axis to decouple",
        )

    def test_attn_vector_reaches_the_previously_unreachable_split(self):
        """The whole ticket in one assertion: hold attention at [7,5,4]
        while the layer vector asks for more mass on stage 0."""
        counts = derive_pp_layer_split(
            [31, 17, 16], self.ISFA, N_LAYERS, attn_scores=[7, 5, 4]
        )
        self.assertEqual(counts, [31, 17, 16])
        self.assertEqual(pp_cut.attention_counts(_families(), counts), (7, 5, 4))

    def test_attn_vector_controls_attention_independently(self):
        """Same layer vector, two attention vectors, two different KV
        splits."""
        a = derive_pp_layer_split(
            [30, 18, 16], self.ISFA, N_LAYERS, attn_scores=[7, 5, 4]
        )
        b = derive_pp_layer_split(
            [30, 18, 16], self.ISFA, N_LAYERS, attn_scores=[8, 4, 4]
        )
        fams = _families()
        self.assertEqual(pp_cut.attention_counts(fams, a), (7, 5, 4))
        self.assertEqual(pp_cut.attention_counts(fams, b), (8, 4, 4))
        self.assertNotEqual(a, b)

    def test_attn_scores_are_validated(self):
        with self.assertRaises(ValueError):
            derive_pp_layer_split([14, 10, 8], self.ISFA, N_LAYERS, [7, 5])
        with self.assertRaises(ValueError):
            derive_pp_layer_split([14, 10, 8], self.ISFA, N_LAYERS, [7, 0, 9])

    def test_zero_attention_stage_still_refused(self):
        with self.assertRaises(ValueError) as ctx:
            derive_pp_layer_split(
                [1, 1, 62], self.ISFA, N_LAYERS, attn_scores=[1, 1, 400]
            )
        self.assertIn("full-attention", str(ctx.exception))

    def test_solver_output_round_trips_through_the_resolver(self):
        """The solved cut must be expressible as (layer, attention) score
        vectors, which is how it reaches the existing boot path."""
        sol = pp_cut.solve_pp_cut(_inputs())
        self.assertTrue(sol.feasible, sol.refusals)
        counts = derive_pp_layer_split(
            list(sol.counts),
            self.ISFA,
            N_LAYERS,
            attn_scores=list(sol.attention_counts),
        )
        self.assertEqual(counts, list(sol.counts))


class TestSeamStagingIsFundedNotAssumed(CustomTestCase):
    """Law 23 / C34: a residency gate cannot certify runnability.

    The measured failure this pins: the model priced weights + KV + a
    calibrated fixed overhead, called the #485 planner cut feasible with
    2617 MiB to spare, and was RIGHT about residency -- the configuration
    does fit at rest. It wedged anyway, because the phase flip needed
    4881 MiB of TRANSIENT staging on that rank at a cutover and nothing in
    this module had a term for peak demand at all.
    """

    def test_a_stage_that_fits_at_rest_can_still_be_refused(self):
        base = _inputs(pool=280000)
        counts = [28, 20, 16]
        rest, _ = pp_cut.validate_pp_cut(counts, base)
        self.assertTrue(rest.feasible, rest.refusals)
        spare = min(s.headroom_mib for s in rest.stages)

        # Ask for more transient than the tightest stage has spare.
        staged = _inputs(
            pool=280000, seam_staging=(spare + 256.0, spare + 256.0, spare + 256.0)
        )
        sol, violations = pp_cut.validate_pp_cut(counts, staged)
        self.assertFalse(sol.feasible)
        self.assertTrue(violations)
        # The refusal must SAY that it fits at rest, or the reader will go
        # looking for a residency problem that does not exist.
        self.assertIn("FITS AT REST", " ".join(violations))
        self.assertIn("seam staging", " ".join(violations))

    def test_zero_staging_reproduces_the_old_verdict_exactly(self):
        # The off switch is a VALUE of the same term, not a second path.
        a = pp_cut.solve_pp_cut(_inputs(pool=280000))
        b = pp_cut.solve_pp_cut(_inputs(pool=280000, seam_staging=(0.0, 0.0, 0.0)))
        self.assertEqual(a.counts, b.counts)
        self.assertEqual(a.min_headroom_mib, b.min_headroom_mib)

    def test_the_reported_headroom_is_the_spendable_one(self):
        # min_headroom_mib is what a caller gates on, so it must be the
        # headroom left AFTER the peak is funded -- reporting residency
        # headroom is exactly how the planner cut got certified.
        staging = 700.0
        plain = pp_cut.solve_pp_cut(_inputs(pool=280000))
        with_staging = pp_cut.solve_pp_cut(
            _inputs(pool=280000, seam_staging=(staging, staging, staging))
        )
        self.assertLess(with_staging.min_headroom_mib, plain.min_headroom_mib)

    def test_the_solver_prefers_a_cut_whose_peak_is_fundable(self):
        # The second pass must rank on runnable headroom. A cut is only
        # better for leaving room the rank can actually spend.
        staging = 400.0
        sol = pp_cut.solve_pp_cut(
            _inputs(pool=280000, seam_staging=(staging, staging, staging))
        )
        self.assertTrue(sol.feasible, sol.refusals)
        for st in sol.stages:
            self.assertGreaterEqual(
                st.runnable_headroom_mib,
                0.0,
                f"{st.rank} was emitted with an unfundable seam",
            )


class TestCheckpointConservation(CustomTestCase):
    """The gate must price the CHECKPOINT, not just its transformer layers.

    Every constant here is MEASURED from the shipping checkpoint's safetensors
    headers (dtype x shape per tensor, grouped by owner) -- not derived from
    the config's parameter formulas, because those formulas are what hid two
    of these three errors. The reference model is a VL checkpoint quantized
    INT8-W8A8: ``lm_head`` and the whole visual tower sit in the quantizer's
    explicit ``ignore`` list, and the input embedding is never a candidate at
    all (compressed-tensors only targets Linear modules), so all three load in
    bf16 and all three were invisible to a per-layer census.

    The falsifier for the WHOLE ledger is
    ``test_priced_weights_equal_the_checkpoint``: if the sum over stages of
    what the gate charges does not equal what the checkpoint contains, the
    gate is wrong by the difference no matter how well any single boot was
    calibrated. It fails by 5729 MiB on the pre-C38 model.
    """

    # Measured 2026-08-12 from
    # Qwen3.6-27B-INT8-W8A8-yarn1.5/*.safetensors headers.
    CKPT_ATTN_LAYER_MIB = 355.13
    CKPT_LINEAR_LAYER_MIB = 366.15
    CKPT_EMBED_MIB = 2425.0
    CKPT_LM_HEAD_MIB = 2425.0
    CKPT_VISUAL_MIB = 878.8
    CKPT_LANGUAGE_LAYERS_MIB = 23257.5

    def _measured_inputs(self, **kw):
        """Rig inputs whose weight terms are the MEASURED checkpoint bytes."""
        params = dict(
            attn_layer_weight_bytes=self.CKPT_ATTN_LAYER_MIB * pp_cut.MIB,
            linear_layer_weight_bytes=self.CKPT_LINEAR_LAYER_MIB * pp_cut.MIB,
            embedding_weight_bytes=self.CKPT_EMBED_MIB * pp_cut.MIB,
            lm_head_weight_bytes=self.CKPT_LM_HEAD_MIB * pp_cut.MIB,
            replicated_weight_bytes=self.CKPT_VISUAL_MIB * pp_cut.MIB,
        )
        params.update(kw)
        base = _inputs(pool=280000)
        return dataclasses.replace(base, **params)

    def test_measured_layer_bytes_are_not_the_formula_bytes(self):
        # Pins WHY the measured constants are used: the config-derived
        # formula omits attn_output_gate's second q-sized projection. If a
        # future checkpoint makes these agree, this test says so out loud
        # rather than letting the formula quietly become right by accident.
        formula_mib = ATTN_LAYER_BYTES / pp_cut.MIB
        self.assertAlmostEqual(formula_mib, 325.0, delta=0.5)
        self.assertAlmostEqual(self.CKPT_ATTN_LAYER_MIB, 355.13, delta=0.05)
        gate_mib = HIDDEN * Q_HEADS * HEAD_DIM * BYTES_PER_PARAM / pp_cut.MIB
        self.assertAlmostEqual(
            self.CKPT_ATTN_LAYER_MIB - formula_mib, gate_mib, delta=0.5
        )

    def test_the_checkpoint_census_is_self_consistent(self):
        # The per-family constants must add up to the measured language-model
        # total, or one of the four numbers above is a typo.
        total = 16 * self.CKPT_ATTN_LAYER_MIB + 48 * self.CKPT_LINEAR_LAYER_MIB
        self.assertAlmostEqual(total, self.CKPT_LANGUAGE_LAYERS_MIB, delta=1.0)

    def test_priced_weights_equal_the_checkpoint(self):
        # THE LEDGER LAW. Sum what the gate charges every stage; it must equal
        # what the checkpoint puts on the rig -- the language layers once,
        # embedding once, lm_head once, and the replicated payload once per
        # stage. Pre-C38 this is short by 5729 MiB.
        inputs = self._measured_inputs()
        sol, violations = pp_cut.validate_pp_cut([28, 20, 16], inputs)
        self.assertEqual(violations, ())
        priced = sum(s.weight_mib + s.nonlayer_weight_mib for s in sol.stages)
        expected = (
            self.CKPT_LANGUAGE_LAYERS_MIB
            + self.CKPT_EMBED_MIB
            + self.CKPT_LM_HEAD_MIB
            + 3 * self.CKPT_VISUAL_MIB
        )
        self.assertAlmostEqual(priced, expected, delta=2.0)

    def test_non_layer_weights_follow_the_stage_ROLE(self):
        # Role-scoped, not per-rank: the embedding is on the FIRST stage and
        # lm_head on the LAST, so a three-stage cut charges them once each and
        # the middle stage carries only the replicated payload. A per-rank
        # scalar cannot express this, which is why the field is not one.
        sol, _ = pp_cut.validate_pp_cut([28, 20, 16], self._measured_inputs())
        first, middle, last = sol.stages
        self.assertAlmostEqual(
            first.nonlayer_weight_mib,
            self.CKPT_EMBED_MIB + self.CKPT_VISUAL_MIB,
            delta=1.0,
        )
        self.assertAlmostEqual(
            middle.nonlayer_weight_mib, self.CKPT_VISUAL_MIB, delta=1.0
        )
        self.assertAlmostEqual(
            last.nonlayer_weight_mib,
            self.CKPT_LM_HEAD_MIB + self.CKPT_VISUAL_MIB,
            delta=1.0,
        )

    def test_the_unpriced_payload_is_worth_3300_MiB_on_rank0(self):
        # The size of the error this closes, on the rank it was measured on.
        # rank0 holds the embedding and its copy of the vision tower, and the
        # old model charged for neither.
        priced = self._measured_inputs()
        unpriced = dataclasses.replace(
            priced,
            embedding_weight_bytes=0.0,
            lm_head_weight_bytes=0.0,
            replicated_weight_bytes=0.0,
        )
        cut = [42, 11, 11]
        a, _ = pp_cut.validate_pp_cut(cut, priced)
        b, _ = pp_cut.validate_pp_cut(cut, unpriced)
        gap = a.stages[0].resident_mib - b.stages[0].resident_mib
        self.assertAlmostEqual(gap, 3303.8, delta=5.0)

    def test_recurrent_state_is_cut_shaped_not_a_rank_constant(self):
        # Moving eleven linear layers onto rank0 must move their state pool
        # with them. Folding this into fixed_overhead_mib is what made that
        # overhead look cut-invariant.
        per_layer = 51.2 * pp_cut.MIB
        inputs = self._measured_inputs(state_bytes_per_linear_layer=per_layer)
        ship, _ = pp_cut.validate_pp_cut([28, 20, 16], inputs)
        planner, _ = pp_cut.validate_pp_cut([42, 11, 11], inputs)
        # ship rank0: 21 linear layers; planner rank0: 32.
        self.assertAlmostEqual(ship.stages[0].state_mib, 21 * 51.2, delta=0.5)
        self.assertAlmostEqual(planner.stages[0].state_mib, 32 * 51.2, delta=0.5)
        self.assertAlmostEqual(
            planner.stages[0].state_mib - ship.stages[0].state_mib,
            563.2,
            delta=1.0,
        )


class TestTokenShareContract(CustomTestCase):
    """The arena follows the TOKEN vector. Half of C38 was feeding it the
    other one -- the flip's WEIGHT vector -- which is a different ratio for a
    different resource."""

    SHIP_TOKEN_VECTOR = (14, 10, 8)
    FLIP_WEIGHT_VECTOR = (32, 16, 16)

    def test_gcd_reduced_vectors_are_the_same_vector(self):
        # 14,10,8 and 7,5,4 are one configuration, not two:
        # resolve_cp_token_split gcd-reduces. A shift that "fixed" one to
        # match the other would be changing nothing and reporting a change.
        self.assertEqual(
            pp_cut.token_shares_from_vector((14, 10, 8)),
            pp_cut.token_shares_from_vector((7, 5, 4)),
        )

    def test_the_two_vectors_are_not_interchangeable(self):
        tokens = pp_cut.token_shares_from_vector(self.SHIP_TOKEN_VECTOR)
        weights = pp_cut.token_shares_from_vector(self.FLIP_WEIGHT_VECTOR)
        self.assertAlmostEqual(tokens[0], 0.4375, places=4)
        self.assertAlmostEqual(weights[0], 0.5, places=4)
        self.assertNotAlmostEqual(tokens[0], weights[0], places=3)

    def test_the_wrong_vector_overcharges_rank0_KV_on_the_ship_cut(self):
        # Both vectors normalize to 1.0, so no sum check can catch the
        # substitution -- only using one resolver can. On the ship cut the
        # weight vector rounds rank0's arena up from 7 layers to 8.
        pool = 280000
        right = _inputs(
            pool=pool, tp_token_shares=pp_cut.token_shares_from_vector((14, 10, 8))
        )
        wrong = _inputs(
            pool=pool, tp_token_shares=pp_cut.token_shares_from_vector((32, 16, 16))
        )
        a, _ = pp_cut.validate_pp_cut([28, 20, 16], right)
        b, _ = pp_cut.validate_pp_cut([28, 20, 16], wrong)
        overcharge = b.stages[0].kv_mib - a.stages[0].kv_mib
        self.assertAlmostEqual(
            overcharge,
            1 * KV_BYTES_PER_TOKEN_PER_ATTN_LAYER * pool / pp_cut.MIB,
            delta=1.0,
        )
        self.assertAlmostEqual(overcharge, 546.9, delta=1.0)

    def test_on_the_planner_cut_the_wrong_vector_is_invisible(self):
        # And this is why the substitution survived: where the PP layer share
        # dominates (rank0 owns 10 attention layers, above both token shares)
        # the max() hides it completely. An error that is invisible on the cut
        # you are testing is still there on the cut you ship.
        pool = 280000
        right = _inputs(
            pool=pool, tp_token_shares=pp_cut.token_shares_from_vector((14, 10, 8))
        )
        wrong = _inputs(
            pool=pool, tp_token_shares=pp_cut.token_shares_from_vector((32, 16, 16))
        )
        a, _ = pp_cut.validate_pp_cut([42, 11, 11], right)
        b, _ = pp_cut.validate_pp_cut([42, 11, 11], wrong)
        self.assertAlmostEqual(a.stages[0].kv_mib, b.stages[0].kv_mib, delta=0.01)

    def test_a_zero_share_is_refused(self):
        with self.assertRaises(ValueError):
            pp_cut.token_shares_from_vector((14, 0, 8))
        with self.assertRaises(ValueError):
            pp_cut.token_shares_from_vector(())


class TestTheTransientIsPerLoadState(CustomTestCase):
    """#485/law 31: a transient measured in one load state is not the transient.

    The gate charged one scalar per rank for three shifts. The scalar was
    measured at a prefill trigger (956 MiB drawn on rank0); the shipping mixed
    soak drew 1989 MiB on the same rank on one cut and 3148 on another. Every
    verdict the gate issued was optimistic by that gap, and two cuts it
    admitted broke the corridor on metal. A cut is admitted only when the
    WORST load state it will serve is funded.
    """

    def _rank(self, **kw):
        base = dict(
            label="rank0-5090",
            attn_bw_gbs=1533.8,
            gemm_tflops=231.97,
            budget_mib=31800.0,
        )
        base.update(kw)
        return pp_cut.RankResources(**base)

    def test_the_worst_state_is_charged_not_the_mean_or_the_last(self):
        rank = self._rank(
            transient_by_load_state={
                "EXTEND": 1989.0,
                "DECODE": 1204.0,
                "IDLE": 12.0,
            }
        )
        self.assertEqual(rank.worst_transient_mib, 1989.0)
        self.assertEqual(rank.governing_load_state, "EXTEND")

    def test_a_scalar_still_works_and_names_no_state(self):
        rank = self._rank(transient_mib=1346.0)
        self.assertEqual(rank.worst_transient_mib, 1346.0)
        self.assertIsNone(rank.governing_load_state)

    def test_two_sources_for_one_term_are_refused(self):
        with self.assertRaises(ValueError) as cm:
            self._rank(
                transient_mib=1346.0,
                transient_by_load_state={"EXTEND": 1989.0},
            )
        self.assertIn("exactly one", str(cm.exception))

    def test_an_empty_table_is_refused_not_read_as_zero(self):
        with self.assertRaises(ValueError) as cm:
            self._rank(transient_by_load_state={})
        self.assertIn("unmeasured", str(cm.exception))

    def test_a_negative_transient_is_refused(self):
        with self.assertRaises(ValueError):
            self._rank(transient_by_load_state={"DECODE": -1.0})

    @staticmethod
    def _with_transients(inputs, tables):
        return dataclasses.replace(
            inputs,
            ranks=tuple(
                dataclasses.replace(r, transient_mib=0.0, transient_by_load_state=t)
                for r, t in zip(inputs.ranks, tables)
            ),
        )

    def _budget_that_binds(self, cut, gentle_mib, spare=200.0):
        """A rank0 budget where the GENTLE transient fits with `spare` left.

        Derived from the model's own pricing rather than hard-coded, so this
        test keeps testing the load-state axis and not a stale constant.
        """
        probe = _inputs(transients=(0.0, 0.0, 0.0), pool=280000)
        sol, _ = pp_cut.validate_pp_cut(cut, probe)
        stage0 = sol.stages[0]
        # budget_mib on the stage is already net of the corridor.
        return stage0.resident_mib + gentle_mib + spare + 1024.0

    def test_the_gate_refuses_a_cut_the_gentle_state_would_admit(self):
        # THE AXIS THAT MATTERS, isolated: identical inputs, identical cut,
        # and the ONLY difference is which load state's transient is charged.
        cut = [40, 12, 12]
        gentle, worst = 1346.0, 1989.0
        budget0 = self._budget_that_binds(cut, gentle)
        budgets = (budget0, 20054.9, 20054.9)

        gentle_inputs = self._with_transients(
            _inputs(transients=(0.0, 0.0, 0.0), budgets=budgets, pool=280000),
            (
                {"PREFILL_TRIGGER": gentle},
                {"PREFILL_TRIGGER": 1120.0},
                {"PREFILL_TRIGGER": 982.0},
            ),
        )
        worst_inputs = self._with_transients(
            _inputs(transients=(0.0, 0.0, 0.0), budgets=budgets, pool=280000),
            (
                {"PREFILL_TRIGGER": gentle, "MIXED_SOAK": worst},
                {"PREFILL_TRIGGER": 1120.0},
                {"PREFILL_TRIGGER": 982.0},
            ),
        )

        _s1, gentle_violations = pp_cut.validate_pp_cut(cut, gentle_inputs)
        _s2, worst_violations = pp_cut.validate_pp_cut(cut, worst_inputs)

        self.assertEqual(list(gentle_violations), [])
        self.assertTrue(
            worst_violations,
            "adding a WORSE measured load state to the same rank must be "
            "able to refuse a cut the gentler state admitted",
        )

    def test_the_refusal_names_the_load_state_that_binds(self):
        cut = [40, 12, 12]
        budget0 = self._budget_that_binds(cut, 1346.0)
        inputs = self._with_transients(
            _inputs(
                transients=(0.0, 0.0, 0.0),
                budgets=(budget0, 20054.9, 20054.9),
                pool=280000,
            ),
            (
                {"MIXED_SOAK": 1989.0, "IDLE": 1.0},
                {"MIXED_SOAK": 1120.0},
                {"MIXED_SOAK": 982.0},
            ),
        )
        _sol, violations = pp_cut.validate_pp_cut(cut, inputs)
        self.assertTrue(violations)
        self.assertIn("MIXED_SOAK", violations[0])
        self.assertNotIn("IDLE", violations[0])


if __name__ == "__main__":
    unittest.main()
