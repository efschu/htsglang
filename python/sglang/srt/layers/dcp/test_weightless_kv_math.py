"""CPU reference proof for the weightless-KV fast lane core math (Variant C
Stage 1). No GPU, no flashinfer, no torch.distributed -- this isolates and
proves the two things the fast lane's correctness rests on:

  1. TOKEN-SHARDED ATTENTION + LSE-MERGE == full attention. Splitting the KV
     across DCP ranks by token (uneven token vector), computing a partial
     attention output + log-sum-exp per shard, and merging with the exact
     formula used by cp_lse_ag_out_ar_mha_uneven (comm.py) reproduces a single
     full-attention pass to fp tolerance. This is the DCP #99 invariant, re-
     proven for the weightless head geometry. The ONLY expected divergence is
     the merge fp order (partials-then-logsumexp vs one softmax) -- bounded, not
     a bug.

  2. THE [H,0,0] HEAD VECTOR delivers the merged output to the head rank only
     and an empty shard to every weightless rank, and the anti-hang guard's
     signature logic catches a collective-sequence divergence.

Run: python -m sglang.srt.layers.dcp.test_weightless_kv_math
"""

from __future__ import annotations

import math

import torch


def _ref_full_attention(q, k, v, scale):
    """Single full-attention pass. q:[T,H,D] k,v:[N,H,D] (heads already GQA-
    expanded). Non-causal (decode: one query row attends all N context)."""
    scores = torch.einsum("thd,nhd->thn", q, k) * scale  # [T,H,N]
    probs = torch.softmax(scores, dim=-1)
    return torch.einsum("thn,nhd->thd", probs, v)  # [T,H,D]


def _partial_attention_with_lse(q, k_shard, v_shard, scale):
    """One DCP token-shard's partial attention output + LSE, mirroring what a
    flashinfer forward_return_lse produces on a rank that owns k_shard/v_shard.
    Returns o:[T,H,D], lse:[T,H] (natural log)."""
    scores = torch.einsum("thd,nhd->thn", q, k_shard) * scale  # [T,H,N_i]
    lse = torch.logsumexp(scores, dim=-1)  # [T,H]
    probs = torch.softmax(scores, dim=-1)
    o = torch.einsum("thn,nhd->thd", probs, v_shard)  # [T,H,D]
    return o, lse


def _merge_lse(partials, head_counts, head_rank):
    """Replicates cp_lse_ag_out_ar_mha_uneven's math (comm.py:163):
    all-gather LSE across ranks, global logsumexp, scale each rank's partial by
    exp(lse_r - global_lse), sum (the 'all_reduce'), then slice to this rank's
    head band [sum(counts[:rank]), +counts[rank]). partials: list of (o, lse)
    per rank. Returns the head rank's merged output."""
    world = len(partials)
    lses = torch.stack([lse for (_, lse) in partials], dim=0)  # [world,T,H]
    global_lse = torch.logsumexp(lses, dim=0)  # [T,H]
    merged = torch.zeros_like(partials[0][0])  # [T,H,D]
    for r, (o, lse) in enumerate(partials):
        scale = torch.exp(lse - global_lse).unsqueeze(-1)  # [T,H,1]
        scale = torch.nan_to_num(scale, nan=0.0, posinf=0.0, neginf=0.0)
        merged = merged + torch.nan_to_num(o) * scale
    # [H,0,0] head slicing: head rank -> [0:H], weightless ranks -> empty.
    out_by_rank = []
    start = 0
    for r in range(world):
        stop = start + head_counts[r]
        out_by_rank.append(merged[:, start:stop, :])
        start = stop
    return merged, out_by_rank


def test_token_sharded_attention_matches_full():
    torch.manual_seed(0)
    T, H, Hkv, D, N = 1, 24, 4, 256, 97  # decode: 1 query, 97 context tokens
    world = 3
    scale = 1.0 / math.sqrt(D)

    q = torch.randn(T, H, D, dtype=torch.float32)
    k_kv = torch.randn(N, Hkv, D, dtype=torch.float32)
    v_kv = torch.randn(N, Hkv, D, dtype=torch.float32)
    # GQA expand kv-heads to q-heads (each kv head serves H//Hkv q heads).
    rep = H // Hkv
    k = k_kv.repeat_interleave(rep, dim=1)  # [N,H,D]
    v = v_kv.repeat_interleave(rep, dim=1)

    o_ref = _ref_full_attention(q, k, v, scale)

    # Uneven DCP token vector (weightless: big shards on the KV cards). The
    # owner rule is the #99 weighted prefix-range; here we just contiguously
    # split N into 3 uneven pieces to mirror per-rank token ownership.
    token_ratio = [1, 3, 3]  # head rank gets a small KV share, 3080s big
    bounds = [0]
    for r in token_ratio:
        bounds.append(bounds[-1] + r)
    S = bounds[-1]
    # map each of N context slots to an owner rank by (slot % S) band
    owners = [[] for _ in range(world)]
    for slot in range(N):
        band = slot % S
        for r in range(world):
            if bounds[r] <= band < bounds[r + 1]:
                owners[r].append(slot)
                break

    partials = []
    for r in range(world):
        idx = torch.tensor(owners[r], dtype=torch.long)
        if idx.numel() == 0:
            # empty shard -> lse = -inf, o = 0 (the merge's nan_to_num handles it)
            o = torch.zeros(T, H, D)
            lse = torch.full((T, H), float("-inf"))
        else:
            o, lse = _partial_attention_with_lse(q, k[idx], v[idx], scale)
        partials.append((o, lse))

    head_rank = 0
    head_counts = [H if r == head_rank else 0 for r in range(world)]
    merged, out_by_rank = _merge_lse(partials, head_counts, head_rank)

    max_err = (merged - o_ref).abs().max().item()
    print(f"[core-math] token-sharded+LSE-merge vs full attention: "
          f"max|Δ| = {max_err:.3e}")
    assert max_err < 1e-4, f"token-shard/LSE-merge diverged: {max_err}"

    # [H,0,0]: head rank gets all H heads, weightless ranks get empty.
    assert out_by_rank[0].shape == (T, H, D)
    assert out_by_rank[1].shape == (T, 0, D)
    assert out_by_rank[2].shape == (T, 0, D)
    print("[H,0,0] slicing: head rank -> [T,24,D], weightless ranks -> [T,0,D] OK")


def test_guard_signature_detects_divergence():
    """The anti-hang guard's per-step signature must DIFFER when the op-tag or
    step index differs, so a collective-order/count divergence is caught."""
    from sglang.srt.layers.dcp.collective_guard import _op_signature

    # same (tag, step) -> identical signature (ranks agree, no false positive)
    assert _op_signature("ag_heads:24", 5) == _op_signature("ag_heads:24", 5)
    # different op at the same step (order/type divergence) -> different sig
    assert _op_signature("ag_heads:24", 5) != _op_signature("lse_merge", 5)
    # same op at a different step (count divergence) -> different sig
    assert _op_signature("lse_merge", 5) != _op_signature("lse_merge", 6)
    # a Q-gather vs a KV-gather differ by the head-count sum baked into the tag
    assert _op_signature("ag_heads:24", 5) != _op_signature("ag_heads:4", 5)
    print("[guard] signature distinguishes op-type, step-index and gather-size OK")


if __name__ == "__main__":
    test_token_sharded_attention_matches_full()
    test_guard_signature_detects_divergence()
    print("\nALL CORE-MATH + GUARD-LOGIC PROOFS GREEN")
