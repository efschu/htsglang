"""Standalone 3-rank integration proof for the weightless-KV fast lane
(Variant C Stage 1), Option A.

Drives the REAL weightless attention-worker collectives over REAL NCCL on 3
ranks -- no ModelRunner, no scheduler, no full model. It upgrades the CPU
core-math proof (layers/dcp/test_weightless_kv_math.py) to the real
distributed collective path and proves the anti-hang guard fires on hardware.

WHAT IT PROVES
  (1) BYTE-IDENTITY on real collectives: the head rank is the SOLE Q/K/V
      producer; each of the 3 ranks holds a DCP TOKEN-SHARD of the same KV.
      The head rank's Q is broadcast via the real cp_all_gather_heads_uneven
      with the [H,0,0] head vector; each rank runs attention over its local
      token-shard; cp_lse_ag_out_ar_mha_uneven with [H,0,0] merges the partials
      and delivers the full output to the head rank only. That merged output
      equals a single full-attention pass over the whole KV, to fp tolerance
      (the only divergence is the merge fp order -- the #99 benign band).
  (2) GUARD FIRES LOUD, NO HANG:
        - REORDER divergence: one rank swaps the collective order -> the
          fixed-shape per-step handshake catches the signature mismatch and
          raises IMMEDIATELY (no timeout, no hang).
        - SKIP divergence: one rank omits a dispatch -> the other ranks fail
          via a BOUNDED timeout (handshake / monitored barrier), not an
          infinite NCCL hang.

REQUIRES 3 visible GPUs. Run (only inside an explicitly granted GPU window):
    /spinning/htsglang-gpu/.venv/bin/python -m pytest -s \
        test/registered/layers/test_weightless_kv_dcp.py
  or directly:
    /spinning/htsglang-gpu/.venv/bin/python \
        test/registered/layers/test_weightless_kv_dcp.py
"""

from __future__ import annotations

import math
import os
import traceback
from typing import Optional

import torch
import torch.multiprocessing as mp

WORLD = 3
HEAD_RANK = 0
T = 1        # decode: one query row
H = 24       # q heads (Qwen3.6-27B)
HKV = 4      # kv heads
D = 256      # head_dim
N = 128      # context tokens
TOKEN_RATIO = [1, 3, 3]   # head rank small KV share, the two 3080s big
# Bounded dist timeout so the SKIP divergence surfaces fast (fail-loud proof),
# not an infinite hang. Generous enough for cold NCCL init.
DIST_TIMEOUT_S = 25


# ---------------------------------------------------------------------------
# Reference + per-shard attention (fp32 for a clean byte-identity band)
# ---------------------------------------------------------------------------
def _gqa_expand(x_kv):  # [N,HKV,D] -> [N,H,D]
    return x_kv.repeat_interleave(H // HKV, dim=1)


def _full_attention(q, k, v, scale):  # q:[T,H,D] k,v:[N,H,D]
    scores = torch.einsum("thd,nhd->thn", q, k) * scale
    probs = torch.softmax(scores, dim=-1)
    return torch.einsum("thn,nhd->thd", probs, v)


def _partial_attention(q, k_shard, v_shard, scale):
    """One token-shard's partial output + LSE, mirroring a
    flashinfer forward_return_lse on a rank owning k_shard/v_shard."""
    if k_shard.shape[0] == 0:
        o = torch.zeros(T, H, D, device=q.device, dtype=q.dtype)
        lse = torch.full((T, H), float("-inf"), device=q.device, dtype=q.dtype)
        return o, lse
    scores = torch.einsum("thd,nhd->thn", q, k_shard) * scale
    lse = torch.logsumexp(scores, dim=-1)
    probs = torch.softmax(scores, dim=-1)
    o = torch.einsum("thn,nhd->thd", probs, v_shard)
    return o, lse


def _owned_token_slots(rank):
    """This rank's owned context slots under the weighted prefix-range owner
    rule (the #99 rule): slot t owned iff (t % S) in [prefix[r], prefix[r+1])."""
    from sglang.srt.distributed.utils import cp_token_prefix

    prefix = cp_token_prefix(WORLD)
    S = prefix[-1]
    lo, hi = prefix[rank], prefix[rank + 1]
    return [t for t in range(N) if lo <= (t % S) < hi]


# ---------------------------------------------------------------------------
# Distributed setup
# ---------------------------------------------------------------------------
def _init_rank(rank, port):
    torch.cuda.set_device(rank)
    from sglang.srt.distributed.parallel_state import (
        init_distributed_environment,
        initialize_model_parallel,
    )
    from sglang.srt.distributed.utils import (
        set_cp_token_ratios,
        set_weightless_kv_head_rank,
    )

    init_distributed_environment(
        world_size=WORLD,
        rank=rank,
        local_rank=rank,
        distributed_init_method=f"tcp://127.0.0.1:{port}",
        backend="nccl",
        timeout=DIST_TIMEOUT_S,
    )
    initialize_model_parallel(
        tensor_model_parallel_size=WORLD,
        decode_context_parallel_size=WORLD,
        backend="nccl",
    )
    # Install the weightless fast lane + the DCP token vector.
    set_weightless_kv_head_rank(HEAD_RANK)
    set_cp_token_ratios(TOKEN_RATIO)


def _dcp_group():
    from sglang.srt.runtime_context import get_parallel

    return get_parallel().dcp_group


def _weightless_attention_step(dcp_group, q_local, k_full, v_full, scale, *,
                               reorder=False, skip_merge=False):
    """One weightless full-attention layer's dispatch on this rank, using the
    REAL collectives.

    q_local: [T,H,D] on the head rank, [T,0,D] on weightless ranks.
    k_full/v_full: this rank's LOCAL token-shard KV, [n_owned,H,D].
    Returns the head rank's merged [T,H,D]; empty [T,0,D] on weightless ranks.

    reorder/skip_merge inject a deliberate collective-sequence divergence to
    exercise the anti-hang guard."""
    from sglang.srt.layers.dcp.comm import (
        cp_all_gather_heads_uneven,
        cp_lse_ag_out_ar_mha_uneven,
    )
    from sglang.srt.distributed.utils import weightless_head_counts

    q_counts = weightless_head_counts(H, WORLD)  # [H,0,0]

    if reorder and dcp_group.rank_in_group == WORLD - 1:
        # DIVERGENCE (reorder): this rank does the merge BEFORE the Q gather.
        # The fixed-shape per-step handshake runs before each real collective,
        # so the swapped op-tag is caught at step 0 -> immediate raise, no hang.
        o, lse = _partial_attention(
            torch.zeros(T, H, D, device=q_local.device, dtype=q_local.dtype),
            k_full, v_full, scale,
        )
        cp_lse_ag_out_ar_mha_uneven(o, lse, dcp_group, q_counts)
        cp_all_gather_heads_uneven(q_local, dcp_group, q_counts)
        return None

    # 1) Q broadcast: head rank's [T,H,D] -> all ranks; workers contribute
    #    [T,0,D]. Real cp_all_gather_heads_uneven with [H,0,0].
    q_full = cp_all_gather_heads_uneven(q_local, dcp_group, q_counts)

    # 2) local attention over this rank's KV token-shard -> partial o + lse
    o, lse = _partial_attention(q_full, k_full, v_full, scale)

    if skip_merge and dcp_group.rank_in_group == WORLD - 1:
        # DIVERGENCE (skip): this rank omits the merge dispatch entirely. The
        # other ranks block on the merge handshake -> BOUNDED timeout raise.
        return None

    # 3) LSE merge: real cp_lse_ag_out_ar_mha_uneven with [H,0,0] -> merged
    #    output on the head rank, empty on the weightless ranks.
    return cp_lse_ag_out_ar_mha_uneven(o, lse, dcp_group, q_counts)


# ---------------------------------------------------------------------------
# Worker scenarios
# ---------------------------------------------------------------------------
def _worker(rank, port, scenario, result_q):
    try:
        _init_rank(rank, port)
        from sglang.srt.layers.dcp.collective_guard import dcp_forward_guard

        dev = torch.device(f"cuda:{rank}")
        scale = 1.0 / math.sqrt(D)

        # Identical seed on every rank -> identical full K,V; each rank slices
        # its owned token-shard. Head rank alone holds Q.
        torch.manual_seed(1234)
        k_all = _gqa_expand(torch.randn(N, HKV, D, device=dev))
        v_all = _gqa_expand(torch.randn(N, HKV, D, device=dev))
        q_all = torch.randn(T, H, D, device=dev)

        owned = _owned_token_slots(rank)
        idx = torch.tensor(owned, dtype=torch.long, device=dev)
        k_shard = k_all[idx] if idx.numel() else k_all[:0]
        v_shard = v_all[idx] if idx.numel() else v_all[:0]
        q_local = q_all if rank == HEAD_RANK else q_all[:, :0, :].contiguous()

        dcp_group = _dcp_group()

        if scenario == "byte_identity":
            with dcp_forward_guard(dcp_group):
                merged = _weightless_attention_step(
                    dcp_group, q_local, k_shard, v_shard, scale
                )
            if rank == HEAD_RANK:
                ref = _full_attention(q_all, k_all, v_all, scale)
                max_err = (merged - ref).abs().max().item()
                result_q.put(("ok", rank, f"max|Δ|={max_err:.3e}", max_err))
            else:
                # weightless ranks receive an empty [T,0,D] slice
                assert merged.shape == (T, 0, D), merged.shape
                result_q.put(("ok", rank, "empty-shard", 0.0))

        elif scenario in ("guard_reorder", "guard_skip"):
            raised = False
            msg = ""
            try:
                with dcp_forward_guard(dcp_group):
                    _weightless_attention_step(
                        dcp_group, q_local, k_shard, v_shard, scale,
                        reorder=(scenario == "guard_reorder"),
                        skip_merge=(scenario == "guard_skip"),
                    )
            except Exception as e:  # guard raise OR bounded-timeout raise
                raised = True
                msg = f"{type(e).__name__}: {str(e)[:80]}"
            result_q.put(("guard", rank, msg, 1.0 if raised else 0.0))
        else:
            result_q.put(("err", rank, f"unknown scenario {scenario}", 0.0))
    except Exception as e:
        result_q.put(("err", rank, f"{type(e).__name__}: {e}\n{traceback.format_exc()[:400]}", 0.0))
    finally:
        try:
            import torch.distributed as dist

            if dist.is_initialized():
                dist.destroy_process_group()
        except Exception:
            pass


def _run(scenario, port):
    ctx = mp.get_context("spawn")
    result_q = ctx.Queue()
    procs = [ctx.Process(target=_worker, args=(r, port, scenario, result_q))
             for r in range(WORLD)]
    for p in procs:
        p.start()
    # Bounded join so a genuine hang can't wedge the harness.
    results = []
    for _ in range(WORLD):
        try:
            results.append(result_q.get(timeout=DIST_TIMEOUT_S + 40))
        except Exception:
            results.append(("timeout", -1, "no result (possible hang)", 0.0))
    for p in procs:
        p.join(timeout=10)
        if p.is_alive():
            p.terminate()
    return results


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------
def _require_gpus():
    if not torch.cuda.is_available() or torch.cuda.device_count() < WORLD:
        raise RuntimeError(
            f"need >= {WORLD} visible GPUs, have "
            f"{torch.cuda.device_count() if torch.cuda.is_available() else 0}"
        )


def test_byte_identity_real_collectives():
    _require_gpus()
    results = _run("byte_identity", port=29591)
    print("\n[byte-identity]", results)
    head = [r for r in results if r[1] == HEAD_RANK]
    assert head and head[0][0] == "ok", f"head rank failed: {results}"
    max_err = head[0][3]
    assert max_err < 1e-3, f"weightless merged output diverged from full: {max_err}"
    assert all(r[0] == "ok" for r in results), results


def test_guard_reorder_fires_loud():
    _require_gpus()
    results = _run("guard_reorder", port=29592)
    print("\n[guard-reorder]", results)
    # every participating rank must RAISE (fires loud), none silently proceed
    assert all(r[0] == "guard" and r[3] == 1.0 for r in results), results


def test_guard_skip_bounded_timeout():
    _require_gpus()
    results = _run("guard_skip", port=29593)
    print("\n[guard-skip]", results)
    # ranks that did NOT skip must raise via a BOUNDED timeout (no infinite hang)
    non_skip = [r for r in results if r[1] != WORLD - 1]
    assert non_skip and all(r[0] == "guard" and r[3] == 1.0 for r in non_skip), results


if __name__ == "__main__":
    test_byte_identity_real_collectives()
    test_guard_reorder_fires_loud()
    test_guard_skip_bounded_timeout()
    print("\nALL OPTION-A GPU PROOFS GREEN")
