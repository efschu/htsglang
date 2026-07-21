"""Unit tests for the granular per-card placement engine (placement.py).

Pure compute: every config is an in-memory ``_gguf_config_and_families``-shaped
dict injected into the cost model, so nothing here reads a checkpoint or touches
a GPU. The load-bearing invariants:

  * head index ranges (even TP + uneven rank_tp_ratio) sum to the head totals;
  * TP > num_kv_heads replicates KV heads and shards the Q heads;
  * uneven-DCP token ranges partition the full context with no gaps/overlaps;
  * MoE expert index ranges partition all experts;
  * duplicate rank_gpu_id (co-location) aggregates onto one card;
  * the per-card MiB breakdown reconciles byte-for-byte with the cost model's
    per-rank ``per_rank_weight_bytes``.
"""

import unittest

from sglang.srt.planner.placement import (
    PlacementFlags,
    _build_cost_model,
    compute_placement,
    compute_placement_struct,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=20, suite="base-a-test-cpu")

_GGUF_BPP = {
    "attn": 0.6,
    "mlp": 0.55,
    "gdn": 2.0,
    "vocab": 1.0625,
    "draft": 1.0625,
}


def _cfg(
    *,
    hidden=2048,
    intermediate=6144,
    layers=24,
    q_heads=16,
    kv_heads=8,
    head_dim=128,
    vocab=100000,
    num_experts=0,
    moe_intermediate=768,
    shared_intermediate=0,
    mtp_layers=0,
    gdn_k_heads=0,
    gdn_v_heads=0,
    layer_types=None,
    has_draft_body=False,
):
    """Build an in-memory GGUF-shaped config dict for the cost model."""
    return {
        "text_config": {
            "hidden_size": hidden,
            "intermediate_size": intermediate,
            "num_hidden_layers": layers,
            "num_attention_heads": q_heads,
            "num_key_value_heads": kv_heads,
            "head_dim": head_dim,
            "attn_output_gate": False,
            "vocab_size": vocab,
            "num_experts": num_experts,
            "num_experts_per_tok": 8 if num_experts else 0,
            "moe_intermediate_size": moe_intermediate if num_experts else 0,
            "shared_expert_intermediate_size": shared_intermediate,
            "mtp_num_hidden_layers": mtp_layers,
            "linear_num_key_heads": gdn_k_heads,
            "linear_num_value_heads": gdn_v_heads,
            "linear_key_head_dim": 128,
            "linear_value_head_dim": 128,
            "linear_conv_kernel_dim": 4,
            "layer_types": layer_types,
        },
        "quantization_config": {"group_size": 128},
        "__gguf_family_bpp__": dict(_GGUF_BPP),
        "__gguf_has_draft_body__": has_draft_body,
    }


class TestPlacementHeads(CustomTestCase):
    def test_even_tp_baseline_head_split(self):
        # GQA, kv_heads=8 >= tp=4, even ratio: KV heads split evenly, Q heads
        # follow at the group size (16/8 = 2), contiguous and covering [0, 16).
        cfg = _cfg(q_heads=16, kv_heads=8)
        p = compute_placement_struct(
            cfg, PlacementFlags(tp_size=4, rank_gpu_memory_mib=[16000] * 4)
        )
        heads = p.attn_heads
        self.assertEqual(len(heads), 4)
        self.assertFalse(any(h.kv_replicated for h in heads))
        # Q heads: contiguous partition of the 16-head axis, sums to 16.
        self.assertEqual(sum(h.q_heads for h in heads), 16)
        self.assertEqual([h.q_head_start for h in heads], [0, 4, 8, 12])
        self.assertEqual([h.q_head_end for h in heads], [4, 8, 12, 16])
        # K and V heads: the 8-head KV axis, sums to 8.
        self.assertEqual(sum(h.k_heads for h in heads), 8)
        self.assertEqual(sum(h.v_heads for h in heads), 8)
        self.assertEqual([h.k_head_start for h in heads], [0, 2, 4, 6])

    def test_uneven_rank_tp_ratio_head_ranges_sum_to_totals(self):
        # kv_heads=8, tp=3, ratio 4:2:2 -> KV units 4:2:2, Q at group 2.
        cfg = _cfg(q_heads=16, kv_heads=8)
        p = compute_placement_struct(
            cfg,
            PlacementFlags(
                tp_size=3, rank_tp_ratio=[4, 2, 2], rank_gpu_memory_mib=[16000] * 3
            ),
        )
        heads = p.attn_heads
        self.assertEqual(sum(h.q_heads for h in heads), 16)
        self.assertEqual(sum(h.k_heads for h in heads), 8)
        self.assertEqual(sum(h.v_heads for h in heads), 8)
        # Ranges are contiguous with no gaps: each start == previous end.
        for prev, cur in zip(heads, heads[1:]):
            self.assertEqual(cur.q_head_start, prev.q_head_end)
            self.assertEqual(cur.k_head_start, prev.k_head_end)
        # The heavy rank gets the largest share.
        self.assertGreaterEqual(heads[0].q_heads, heads[1].q_heads)

    def test_tp_greater_than_kv_heads_replicates(self):
        # tp=4 > kv_heads=2: KV heads replicated on every rank, Q heads sharded.
        cfg = _cfg(q_heads=16, kv_heads=2)
        p = compute_placement_struct(
            cfg, PlacementFlags(tp_size=4, rank_gpu_memory_mib=[10000] * 4)
        )
        heads = p.attn_heads
        self.assertTrue(all(h.kv_replicated for h in heads))
        # Every rank holds the FULL [0, 2) KV head range (replicated).
        for h in heads:
            self.assertEqual((h.k_head_start, h.k_head_end), (0, 2))
            self.assertEqual((h.v_head_start, h.v_head_end), (0, 2))
            self.assertEqual(h.k_heads, 2)
        # Q heads still shard and cover the full 16-head axis.
        self.assertEqual(sum(h.q_heads for h in heads), 16)
        self.assertEqual(heads[0].q_head_start, 0)
        self.assertEqual(heads[-1].q_head_end, 16)
        # A note flags the replication asymmetry.
        self.assertTrue(any("REPLICATED" in n for n in p.notes))

    def test_gdn_key_and_value_heads_reported_separately(self):
        # Hybrid model: 16 key / 48 value linear heads (the user's geometry).
        cfg = _cfg(
            kv_heads=8,
            gdn_k_heads=16,
            gdn_v_heads=48,
            layer_types=[
                "linear_attention" if (i + 1) % 4 else "full_attention"
                for i in range(24)
            ],
        )
        p = compute_placement_struct(
            cfg,
            PlacementFlags(
                tp_size=4, rank_tp_ratio=[1, 1, 1, 1], rank_gpu_memory_mib=[16000] * 4
            ),
        )
        self.assertIsNotNone(p.gdn_heads)
        # 16 key heads split over 4 ranks; value heads at the 48/16 = 3 group.
        self.assertEqual(sum(g.k_heads for g in p.gdn_heads), 16)
        self.assertEqual(sum(g.v_heads for g in p.gdn_heads), 48)
        for g in p.gdn_heads:
            self.assertEqual(g.v_heads, 3 * g.k_heads)


class TestPlacementTokensAndExperts(CustomTestCase):
    def test_dcp_token_ranges_partition_context(self):
        cfg = _cfg(kv_heads=8)
        context = 40960
        # Explicit token vector -> deterministic ownership direction (the
        # capacity-estimate vector instead tracks each rank's FREE VRAM, so the
        # heavy-weight rank would own FEWER tokens; that path is exercised via
        # the no-explicit-vector partition invariants below).
        p = compute_placement_struct(
            cfg,
            PlacementFlags(
                tp_size=3,
                rank_tp_ratio=[3, 1, 1],
                dcp_size=3,
                kv_token_vector=[3, 1, 1],
                context_length=context,
                rank_gpu_memory_mib=[16000] * 3,
            ),
        )
        toks = p.kv_tokens
        # Exhaustive, disjoint cover of [0, context): start_0 = 0, end_n = ctx,
        # each start == previous end, total owned == context.
        self.assertEqual(toks[0].pos_start, 0)
        self.assertEqual(toks[-1].pos_end, context)
        for prev, cur in zip(toks, toks[1:]):
            self.assertEqual(cur.pos_start, prev.pos_end)
        self.assertEqual(sum(t.tokens_owned for t in toks), context)
        # 3:1:1 ownership -> rank 0 owns strictly more positions.
        self.assertGreater(toks[0].tokens_owned, toks[1].tokens_owned)

    def test_capacity_estimate_token_partition_no_gaps(self):
        # No explicit vector: the parse-time capacity estimate drives the split;
        # it must still partition the full context exactly.
        cfg = _cfg(kv_heads=8)
        context = 32768
        p = compute_placement_struct(
            cfg,
            PlacementFlags(
                tp_size=3,
                rank_tp_ratio=[2, 1, 1],
                dcp_size=3,
                context_length=context,
                rank_gpu_memory_mib=[16000] * 3,
            ),
        )
        toks = p.kv_tokens
        self.assertEqual(toks[0].pos_start, 0)
        self.assertEqual(toks[-1].pos_end, context)
        for prev, cur in zip(toks, toks[1:]):
            self.assertEqual(cur.pos_start, prev.pos_end)
        self.assertEqual(sum(t.tokens_owned for t in toks), context)

    def test_explicit_token_vector_wins(self):
        cfg = _cfg(kv_heads=8)
        p = compute_placement_struct(
            cfg,
            PlacementFlags(
                tp_size=3,
                dcp_size=3,
                kv_token_vector=[5, 3, 2],
                context_length=10000,
                rank_gpu_memory_mib=[16000] * 3,
            ),
        )
        self.assertEqual(p.token_vector, [5, 3, 2])
        self.assertEqual(p.token_vector_source, "explicit_flag")
        self.assertEqual(sum(t.tokens_owned for t in p.kv_tokens), 10000)

    def test_moe_expert_ranges_partition_all_experts(self):
        cfg = _cfg(num_experts=128, moe_intermediate=768, intermediate=0)
        p = compute_placement_struct(
            cfg,
            PlacementFlags(
                tp_size=3,
                rank_tp_ratio=[2, 1, 1],
                rank_moe_ratio=[2, 1, 1],
                rank_gpu_memory_mib=[16000] * 3,
            ),
        )
        exp = p.experts
        self.assertIsNotNone(exp)
        # Contiguous, exhaustive expert-index cover of [0, 128).
        self.assertEqual(exp[0].expert_start, 0)
        self.assertEqual(exp[-1].expert_end, 128)
        for prev, cur in zip(exp, exp[1:]):
            self.assertEqual(cur.expert_start, prev.expert_end)
        self.assertEqual(sum(e.num_experts for e in exp), 128)
        # The 2:1:1 ratio puts the most experts on rank 0.
        self.assertGreater(exp[0].num_experts, exp[1].num_experts)

    def test_shared_expert_sharded_and_present(self):
        cfg = _cfg(num_experts=128, moe_intermediate=768, shared_intermediate=2048)
        p = compute_placement_struct(
            cfg, PlacementFlags(tp_size=2, rank_gpu_memory_mib=[16000] * 2)
        )
        self.assertIsNotNone(p.shared_expert)
        self.assertEqual(len(p.shared_expert["per_rank_mib"]), 2)
        self.assertGreater(p.shared_expert["total_mib"], 0.0)

    def test_dense_model_has_no_experts_or_offload(self):
        cfg = _cfg(num_experts=0)
        p = compute_placement_struct(
            cfg, PlacementFlags(tp_size=2, rank_gpu_memory_mib=[16000] * 2)
        )
        self.assertIsNone(p.experts)
        self.assertIsNone(p.shared_expert)
        self.assertIsNone(p.offload)  # only MoE routed experts tier to host


class TestPlacementCards(CustomTestCase):
    def test_colocation_aggregates_onto_one_card(self):
        # tp=4, ranks 1 and 2 share physical GPU 1 (co-location).
        cfg = _cfg(num_experts=128, moe_intermediate=768)
        p = compute_placement_struct(
            cfg,
            PlacementFlags(
                tp_size=4,
                rank_gpu_id=[0, 1, 1, 2],
                rank_gpu_memory_mib=[15000, 15000, 15000, 15000],
                card_total_mib={0: 20480, 1: 32760, 2: 20480},
                card_name={0: "RTX 3080", 1: "RTX 5090", 2: "RTX 3080"},
            ),
        )
        self.assertEqual(p.ranks_per_gpu, {0: [0], 1: [1, 2], 2: [3]})
        cards = {c.gpu_index: c for c in p.cards}
        self.assertEqual(len(cards), 3)
        shared_card = cards[1]
        self.assertEqual(shared_card.ranks, [1, 2])
        # The co-located card's aggregate is the SUM of both ranks.
        r1, r2 = p.ranks[1], p.ranks[2]
        self.assertAlmostEqual(
            shared_card.weight_mib, r1.weight_mib + r2.weight_mib, places=3
        )
        self.assertEqual(shared_card.budget_mib, 30000)
        # 2 x 15000 = 30000 <= 32760 -> no physical overcommit on the 5090.
        self.assertFalse(shared_card.physical_overcommit)

    def test_physical_overcommit_flagged(self):
        cfg = _cfg()
        p = compute_placement_struct(
            cfg,
            PlacementFlags(
                tp_size=2,
                rank_gpu_id=[0, 0],  # both ranks on one card
                rank_gpu_memory_mib=[15000, 15000],
                card_total_mib={0: 20480},  # 30000 > 20480 -> overcommit
            ),
        )
        card = p.cards[0]
        self.assertEqual(card.ranks, [0, 1])
        self.assertTrue(card.physical_overcommit)

    def test_per_card_breakdown_reconciles_with_cost_model(self):
        # The sum of the per-card weight buckets must equal the cost model's
        # own per_rank_weight_bytes (byte-identical, even-vocab case).
        cfg = _cfg(num_experts=128, moe_intermediate=768, mtp_layers=1,
                   has_draft_body=True)
        flags = PlacementFlags(
            tp_size=3,
            rank_gpu_id=[0, 1, 1],
            rank_tp_ratio=[2, 1, 1],
            rank_moe_ratio=[2, 1, 1],
            rank_gpu_memory_mib=[16000, 16000, 16000],
            speculative_algorithm="eagle",
            speculative_num_draft_tokens=3,
        )
        p = compute_placement_struct(cfg, flags)

        model = _build_cost_model(cfg, flags, [16000] * 3, [2, 1, 1])
        prw = model.per_rank_weight_bytes([2, 1, 1])
        for r, rb in enumerate(p.ranks):
            self.assertAlmostEqual(
                rb.weight_mib, prw[r] / 2**20, places=4,
                msg=f"rank {r} weight breakdown does not reconcile",
            )
            # The bucket sum equals the reported weight_mib.
            bucket_sum = (
                rb.attn_mib
                + rb.ffn_experts_mib
                + rb.vocab_mib
                + rb.gdn_mib
                + rb.vision_mib
                + rb.mtp_mib
            )
            self.assertAlmostEqual(bucket_sum, rb.weight_mib, places=4)

        # Per-card weight == sum of the co-located ranks' weights.
        card1 = {c.gpu_index: c for c in p.cards}[1]
        self.assertAlmostEqual(
            card1.weight_mib,
            p.ranks[1].weight_mib + p.ranks[2].weight_mib,
            places=3,
        )
        # Total budget usage per card equals weights + KV + mamba + overhead.
        for c in p.cards:
            expect = c.weight_mib + c.kv_mib + c.mamba_mib + c.overhead_mib
            self.assertAlmostEqual(c.total_mib, expect, places=4)


class TestPlacementMisc(CustomTestCase):
    def test_offload_rule_enumerates_host_experts(self):
        cfg = _cfg(num_experts=128, moe_intermediate=768)
        p = compute_placement_struct(
            cfg,
            PlacementFlags(
                tp_size=2,
                rank_gpu_memory_mib=[16000] * 2,
                moe_resident_expert_fraction=0.5,
            ),
        )
        self.assertIsNotNone(p.offload)
        self.assertEqual(p.offload.offloadable_class, "moe_routed_experts")
        self.assertEqual(p.offload.resident_fraction, 0.5)
        for rule, exp in zip(p.offload.per_rank, p.experts):
            # Resident + host = the rank's whole expert count; host is the tail.
            self.assertEqual(
                rule.resident_expert_count + rule.host_expert_count,
                exp.num_experts,
            )
            if rule.host_expert_count:
                self.assertEqual(rule.host_expert_end, exp.expert_end)
            # resident_mib + host_mib == the rank's full offloadable routed pool.
            self.assertAlmostEqual(
                rule.resident_mib + rule.host_mib, rule.offloadable_mib, places=3
            )

    def test_offload_pool_only_without_fraction(self):
        cfg = _cfg(num_experts=128, moe_intermediate=768)
        p = compute_placement_struct(
            cfg, PlacementFlags(tp_size=2, rank_gpu_memory_mib=[16000] * 2)
        )
        self.assertIsNotNone(p.offload)
        self.assertIsNone(p.offload.resident_fraction)
        self.assertEqual(p.offload.total_host_mib, 0.0)
        self.assertGreater(p.offload.total_offloadable_mib, 0.0)

    def test_mtp_layer_placement(self):
        cfg = _cfg(num_experts=0, mtp_layers=1, layers=24, has_draft_body=True)
        p = compute_placement_struct(
            cfg,
            PlacementFlags(
                tp_size=2,
                rank_gpu_memory_mib=[16000] * 2,
                speculative_algorithm="eagle",
                speculative_num_draft_tokens=3,
            ),
        )
        self.assertTrue(p.mtp.present)
        self.assertEqual(p.mtp.num_layers, 1)
        # nextn block sits after the base layers.
        self.assertEqual((p.mtp.layer_start, p.mtp.layer_end), (24, 25))
        self.assertEqual(len(p.mtp.per_rank_mib), 2)
        self.assertTrue(all(m > 0 for m in p.mtp.per_rank_mib))

    def test_top_level_is_json_able(self):
        import json

        cfg = _cfg(num_experts=128, moe_intermediate=768, shared_intermediate=512)
        d = compute_placement(
            cfg,
            dict(
                tp_size=2,
                rank_gpu_id=[0, 1],
                rank_gpu_memory_mib=15000,  # scalar broadcast
                context_length=8192,
            ),
        )
        # Round-trips through JSON unchanged.
        self.assertEqual(json.loads(json.dumps(d)).keys(), d.keys())
        self.assertEqual(d["tp_size"], 2)
        # Scalar budget broadcast to both ranks.
        self.assertEqual([r["budget_mib"] for r in d["ranks"]], [15000, 15000])

    def test_scalar_budget_broadcast(self):
        cfg = _cfg()
        f = PlacementFlags.from_dict(dict(tp_size=3, rank_gpu_memory_mib=12000))
        self.assertEqual(f.rank_gpu_memory_mib, [12000, 12000, 12000])


if __name__ == "__main__":
    unittest.main()
