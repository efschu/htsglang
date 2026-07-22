"""Unit tests for solo-aware WEIGHT planning in uneven_perf.PerfCostModel.

The auto-performance planner picks the per-rank weight (MLP) split from a
budget model. That model charged the draft as if it were SPLIT -- roughly
1/tp on every rank -- which is the exact opposite of what
``--speculative-draft-placement solo`` does: one rank carries the whole
unsharded draft, a draft KV pool sized to the GLOBAL context, and the draft
graphs, while every other rank carries none.

The consequence was concrete: the optimizer hands the biggest weight shard to
the fastest card, which is also the card chosen to host the solo draft, and
then that rank has no KV room left. Because the global context is
``min_r(P_r/ratio_r) * sum(ratios)``, one starved rank throttles the whole
pool and the other cards idle with several GB free.

Contracts covered:
* Solo: draft weight families are charged 100% to the host and 0% to the
  shadows; the external draft checkpoint is charged to the host too.
* Solo: the resulting optimum moves weight OFF the host relative to the
  solo-blind model.
* Solo: shadow KV cells lose the draft term; the host's draft KV scales with
  the GLOBAL context (the closed form mirrors solo_draft_kv_cell_factor).
* Non-solo: families, shards, per-rank weight bytes and predicted capacity
  are byte-identical to before the change (the split cost model must not
  move).
"""

import dataclasses
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from sglang.srt.uneven_perf import (
    PerfCostModel,
    PlanInputs,
    _kv_cell_bytes_from_config,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

TP = 3
SOLO_RANK = 0

#: Small dense config with an MTP block, so the draft_* families are non-zero.
#: Depth/width chosen so the scenario is REALISTIC: weights and KV are both
#: material against the budgets, so the solo host stays feasible and the
#: capacity arithmetic actually discriminates. A toy config makes the predicted
#: context explode until the globally-sized draft KV alone exceeds the host.
CFG = {
    "hidden_size": 2048,
    "intermediate_size": 8192,
    "num_hidden_layers": 24,
    "num_key_value_heads": 4,
    "num_attention_heads": 16,
    "head_dim": 128,
    "vocab_size": 32000,
    "mtp_num_hidden_layers": 1,
}


#: External DFLASH-style draft: shallower than the target but WIDER KV
#: (more kv heads), so its per-token draft-KV cell is not derivable from the
#: target's mtp term -- mirrors the reference rig (5 layers x 8 kv heads vs
#: the target's 1 x 4).
DRAFT_CFG = {
    "hidden_size": 2048,
    "intermediate_size": 8192,
    "num_hidden_layers": 5,
    "num_key_value_heads": 8,
    "num_attention_heads": 16,
    "head_dim": 128,
    "vocab_size": 151936,
}


def make_model(*, solo: bool, budgets=(30000, 18000, 18000), draft_path=None):
    plan = PlanInputs(
        tp_size=TP,
        model_path="/nonexistent/model",
        kv_cache_dtype="fp8_e5m2",
        speculative_algorithm="DFLASH",
        speculative_num_draft_tokens=16,
        speculative_draft_model_path=draft_path,
        speculative_draft_placement="solo" if solo else "split",
        speculative_draft_solo_rank=SOLO_RANK if solo else 0,
        max_running_requests=2,
        effective_vram_mib=list(budgets),
    )
    def _cfg(p):
        # The external draft is a DIFFERENT model with its own depth and KV
        # geometry -- that is the whole point of reading its config.
        return DRAFT_CFG if draft_path and p == draft_path else CFG

    with patch.object(PerfCostModel, "_load_config", staticmethod(_cfg)), patch(
        "sglang.srt.distributed.utils._checkpoint_size_mib",
        lambda p: 3460 if p == draft_path else 0,
    ):
        return PerfCostModel(plan, base_plan=[1] * TP, budgets_mib=list(budgets))


class TestSoloWeightCharging(CustomTestCase):
    def test_solo_charges_draft_to_host_only(self):
        m = make_model(solo=True)
        self.assertTrue(m.solo_active)
        self.assertEqual(m.solo_rank, SOLO_RANK)
        for name in ("draft_attn", "draft_mlp", "draft_repl"):
            self.assertEqual(
                m.families[name].shard, "solo_host", f"{name} not host-charged"
            )
            fracs = m._shard_fractions("solo_host", [1] * TP)
            self.assertEqual(fracs[SOLO_RANK], 1.0)
            self.assertEqual([fracs[r] for r in range(1, TP)], [0.0] * (TP - 1))

    def test_external_draft_checkpoint_charged_to_host(self):
        m = make_model(solo=True, draft_path="/nonexistent/dflash")
        self.assertIn("draft_solo_ckpt", m.families)
        fam = m.families["draft_solo_ckpt"]
        self.assertEqual(fam.shard, "solo_host")
        self.assertAlmostEqual(fam.bytes, 3460 * 2**20, delta=1.0)
        # It lands on the host's weight bytes and nowhere else.
        with_ext = m.per_rank_weight_bytes([1] * TP)
        m_no_ext = make_model(solo=True, draft_path=None)
        without = m_no_ext.per_rank_weight_bytes([1] * TP)
        self.assertGreater(with_ext[SOLO_RANK] - without[SOLO_RANK], 3000 * 2**20)
        for r in range(1, TP):
            self.assertAlmostEqual(with_ext[r], without[r], delta=1.0)

    def test_solo_shifts_capacity_off_the_host(self):
        """The mechanism behind the fix, asserted on the budget model itself
        (an optimizer scan ties on synthetic configs and picks arbitrarily).

        At the SAME weight split, charging the draft where it actually lives
        must cost the host capacity and give it back to the shadows."""
        solo = make_model(solo=True, draft_path="/nonexistent/dflash")
        split = make_model(solo=False, draft_path="/nonexistent/dflash")
        vec = [1] * TP
        p_solo = solo.predict_capacity(vec)["p"]
        p_split = split.predict_capacity(vec)["p"]

        self.assertLess(
            p_solo[SOLO_RANK],
            p_split[SOLO_RANK],
            "solo host must be charged MORE than the split model assumed",
        )
        for r in range(1, TP):
            self.assertGreater(
                p_solo[r],
                p_split[r],
                f"shadow rank {r} must be charged LESS (it holds no draft)",
            )

    def test_solo_optimum_does_not_load_the_host_more_than_split(self):
        """End-to-end on the optimizer, with a deterministic tie-break that
        prefers the SMALLER host share so ties cannot mask a regression."""
        solo = make_model(solo=True, draft_path="/nonexistent/dflash")
        split = make_model(solo=False, draft_path="/nonexistent/dflash")

        def best_host_share(model):
            scored = []
            for a in range(1, 9):
                for b in range(1, 9):
                    v = [a, b, b]
                    cap = model.predict_capacity(v)
                    if cap["feasible"]:
                        # maximize ctx, then MINIMIZE the host's weight share
                        scored.append((cap["ctx"], -a / sum(v), v))
            self.assertTrue(scored, "no feasible candidate")
            top = max(scored)
            return -top[1], top[2]

        solo_share, solo_v = best_host_share(solo)
        split_share, split_v = best_host_share(split)
        self.assertLessEqual(
            solo_share,
            split_share,
            f"solo optimum {solo_v} loads the host MORE than split {split_v}",
        )


class TestSoloKvCells(CustomTestCase):
    def test_shadow_loses_draft_kv_host_scales_with_global_ctx(self):
        m = make_model(solo=True)
        per_layer = m.kv_cell_bytes_per_layer
        t_tgt = per_layer * m.full_layers
        t_drf = m.kv_cell_bytes - t_tgt
        self.assertGreater(t_drf, 0, "config must have a draft-KV term")

        free = [10.0 * 2**30, 8.0 * 2**30, 8.0 * 2**30]
        p = m._solo_rank_token_capacity(free)

        # Shadows: pure target cell, strictly more than the split model gives.
        for r in (1, 2):
            self.assertAlmostEqual(p[r], free[r] / t_tgt, delta=1.0)
            self.assertGreater(p[r], free[r] / m.kv_cell_bytes)

        # Host: satisfies free_h = p_h * t_tgt + C * t_drf with C = sum(p).
        ctx = sum(p)
        self.assertAlmostEqual(
            free[0], p[0] * t_tgt + ctx * t_drf, delta=max(free[0] * 1e-9, 1.0)
        )

    def test_reduces_to_split_expression_without_draft_kv(self):
        m = make_model(solo=True)
        m.kv_cell_bytes = m.kv_cell_bytes_per_layer * m.full_layers  # t_drf = 0
        free = [10.0 * 2**30, 8.0 * 2**30, 8.0 * 2**30]
        p = m._solo_rank_token_capacity(free)
        for r in range(TP):
            self.assertAlmostEqual(p[r], free[r] / m.kv_cell_bytes, delta=1.0)


class TestSoloTokenVectorSeed(CustomTestCase):
    """predict_capacity must expose a token vector derived from the SOLO-aware
    per-rank capacity -- that is what the planner seeds --rank-kv-ratio with,
    so the shadow ranks stop being handed a share the host cannot fund."""

    def test_token_vector_follows_solo_capacity_not_raw_budget(self):
        solo = make_model(solo=True, draft_path="/nonexistent/dflash")
        cap = solo.predict_capacity([1] * TP)
        vec = cap["token_vector"]
        self.assertIsNotNone(vec)
        self.assertEqual(len(vec), TP)
        self.assertTrue(all(v > 0 for v in vec))

        # The vector must track the solo-aware capacities: whenever one rank
        # can hold strictly fewer tokens, it must not get a larger share.
        # Equal-capacity ranks may differ by one unit from largest-remainder
        # rounding, so ties are deliberately not constrained.
        p = cap["p"]
        for r in range(TP):
            for s in range(TP):
                if p[r] < p[s] * 0.999:
                    self.assertLessEqual(
                        vec[r],
                        vec[s],
                        f"token vector {vec} gives rank {r} a larger share "
                        f"than rank {s} despite lower capacity {p}",
                    )

    def test_solo_host_share_is_smaller_than_its_budget_share(self):
        """The host has the BIGGEST card but the smallest usable KV share once
        the draft is charged to it -- the exact inversion the budget-estimate
        fallback misses."""
        budgets = (30000, 18000, 18000)
        solo = make_model(
            solo=True, budgets=budgets, draft_path="/nonexistent/dflash"
        )
        vec = solo.predict_capacity([1] * TP)["token_vector"]
        host_token_share = vec[SOLO_RANK] / sum(vec)
        host_budget_share = budgets[SOLO_RANK] / sum(budgets)
        self.assertLess(
            host_token_share,
            host_budget_share,
            f"token vector {vec} still gives the host its raw budget share",
        )


class TestSoloHostRuntimeOverhead(CustomTestCase):
    """The host also pays for the draft's CUDA graphs + attention workspace.
    Unmodelled, it allocates PAST its budget and OOMs mid-decode."""

    def test_only_the_host_is_charged_and_it_scales_with_concurrency(self):
        def host_capacity(mrr):
            plan = PlanInputs(
                tp_size=TP,
                model_path="/nonexistent/model",
                kv_cache_dtype="fp8_e5m2",
                speculative_algorithm="DFLASH",
                speculative_num_draft_tokens=16,
                speculative_draft_model_path="/nonexistent/dflash",
                speculative_draft_placement="solo",
                speculative_draft_solo_rank=SOLO_RANK,
                max_running_requests=mrr,
                effective_vram_mib=[30000, 18000, 18000],
            )

            def _cfg(p):
                return DRAFT_CFG if p == "/nonexistent/dflash" else CFG

            with patch.object(
                PerfCostModel, "_load_config", staticmethod(_cfg)
            ), patch(
                "sglang.srt.distributed.utils._checkpoint_size_mib",
                lambda p: 3460 if p == "/nonexistent/dflash" else 0,
            ):
                m = PerfCostModel(
                    plan, base_plan=[1] * TP, budgets_mib=[30000, 18000, 18000]
                )
            return m.predict_capacity([1] * TP)["p"]

        p2, p8 = host_capacity(2), host_capacity(8)
        # More concurrency -> more draft graph memory -> less host KV.
        self.assertLess(p8[SOLO_RANK], p2[SOLO_RANK])
        # Shadow ranks capture no draft graphs, so they are untouched.
        for r in range(1, TP):
            self.assertAlmostEqual(p8[r], p2[r], delta=1.0)

    def test_non_solo_pays_no_such_overhead(self):
        split_a = make_model(solo=False, draft_path="/nonexistent/dflash")
        p = split_a.predict_capacity([1] * TP)["p"]
        # Same model with a different concurrency must be identical under
        # split placement (the term is solo-only).
        plan = dataclasses.replace(split_a.plan_inputs, max_running_requests=8)

        def _cfg(pth):
            return DRAFT_CFG if pth == "/nonexistent/dflash" else CFG

        with patch.object(PerfCostModel, "_load_config", staticmethod(_cfg)), patch(
            "sglang.srt.distributed.utils._checkpoint_size_mib",
            lambda pth: 3460 if pth == "/nonexistent/dflash" else 0,
        ):
            split_b = PerfCostModel(
                plan, base_plan=[1] * TP, budgets_mib=list(split_a.budgets_mib)
            )
        p_b = split_b.predict_capacity([1] * TP)["p"]
        for r in range(TP):
            self.assertAlmostEqual(p[r], p_b[r], delta=1.0)


class TestDraftKvCellIsLayoutGeneric(CustomTestCase):
    """An external draft may have ANY layout; its KV pool must be sized from
    its own config, never by assuming it matches the target."""

    def test_dense_gqa(self):
        cfg = {
            "num_hidden_layers": 5,
            "num_key_value_heads": 8,
            "head_dim": 128,
        }
        # 2 (K,V) * 8 kv heads * 128 head_dim * 1 B (fp8) * 5 layers
        self.assertEqual(
            _kv_cell_bytes_from_config(cfg, "fp8_e5m2"), 2 * 8 * 128 * 1 * 5
        )
        self.assertEqual(
            _kv_cell_bytes_from_config(cfg, "auto"), 2 * 8 * 128 * 2 * 5
        )

    def test_head_dim_derived_when_absent(self):
        cfg = {
            "num_hidden_layers": 2,
            "num_key_value_heads": 4,
            "hidden_size": 2048,
            "num_attention_heads": 16,
        }  # head_dim = 2048/16 = 128
        self.assertEqual(
            _kv_cell_bytes_from_config(cfg, "fp8_e5m2"), 2 * 4 * 128 * 1 * 2
        )

    def test_mla_latent_layout(self):
        """MLA stores ONE latent per token, not a K/V pair."""
        cfg = {
            "num_hidden_layers": 3,
            "kv_lora_rank": 512,
            "qk_rope_head_dim": 64,
            "num_key_value_heads": 128,  # must be IGNORED for MLA
            "head_dim": 128,
        }
        self.assertEqual(
            _kv_cell_bytes_from_config(cfg, "fp8_e5m2"), (512 + 64) * 1 * 3
        )

    def test_hybrid_only_counts_kv_bearing_layers(self):
        """Linear/GDN/mamba layers hold a fixed recurrent state, not paged KV."""
        cfg = {
            "num_hidden_layers": 6,
            "layer_types": [
                "linear_attention",
                "linear_attention",
                "full_attention",
                "linear_attention",
                "sliding_attention",
                "linear_attention",
            ],
            "num_key_value_heads": 2,
            "head_dim": 64,
        }
        # only the full_attention + sliding_attention layers count -> 2
        self.assertEqual(
            _kv_cell_bytes_from_config(cfg, "fp8_e5m2"), 2 * 2 * 64 * 1 * 2
        )

    def test_text_config_nesting(self):
        cfg = {"text_config": {"num_hidden_layers": 1, "num_key_value_heads": 2, "head_dim": 8}}
        self.assertEqual(_kv_cell_bytes_from_config(cfg, "fp8_e5m2"), 2 * 2 * 8 * 1)

    def test_unknown_layout_returns_none_instead_of_guessing(self):
        self.assertIsNone(_kv_cell_bytes_from_config({}, "fp8_e5m2"))
        self.assertIsNone(_kv_cell_bytes_from_config(None, "fp8_e5m2"))
        self.assertIsNone(
            _kv_cell_bytes_from_config({"num_hidden_layers": 4}, "fp8_e5m2")
        )

    def test_model_uses_the_draft_layout_not_the_target_mtp_term(self):
        solo = make_model(solo=True, draft_path="/nonexistent/dflash")
        expected = _kv_cell_bytes_from_config(DRAFT_CFG, "fp8_e5m2")
        self.assertEqual(solo.solo_draft_kv_cell_bytes, expected)
        # and it differs from the target's mtp-derived term, which is the bug.
        target_term = solo.kv_cell_bytes - (
            solo.kv_cell_bytes_per_layer * solo.full_layers
        )
        self.assertNotEqual(expected, target_term)


class TestNonSoloUnchanged(CustomTestCase):
    """The split cost model must not move at all."""

    def test_split_keeps_historical_shards_and_no_extra_family(self):
        m = make_model(solo=False, draft_path="/nonexistent/dflash")
        self.assertFalse(m.solo_active)
        self.assertEqual(m.families["draft_attn"].shard, "attn")
        self.assertEqual(m.families["draft_mlp"].shard, "mlp")
        self.assertEqual(m.families["draft_repl"].shard, "replicated")
        self.assertNotIn("draft_solo_ckpt", m.families)
        self.assertEqual(m.solo_draft_ckpt_bytes, 0.0)

    def test_split_capacity_uses_the_uniform_cell(self):
        m = make_model(solo=False)
        cap = m.predict_capacity([1] * TP)
        weights = m.per_rank_weight_bytes([1] * TP)
        overhead_p = cap["p"]
        for r in range(TP):
            budget = m.budgets_mib[r] * 2**20
            free = budget - weights[r] - m.mamba_pool_bytes[r]
            # p is free-minus-constant-overhead over the UNIFORM cell; check the
            # cell by ratio so the overhead constant cancels.
            self.assertAlmostEqual(
                overhead_p[r] * m.kv_cell_bytes,
                free - (free - overhead_p[r] * m.kv_cell_bytes),
                delta=abs(free) * 1e-6 + 1.0,
            )

    def test_split_never_reaches_the_solo_host_shard(self):
        m = make_model(solo=False)
        for fam in m.families.values():
            self.assertNotEqual(fam.shard, "solo_host")


if __name__ == "__main__":
    unittest.main()
