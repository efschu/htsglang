"""DFLASH draft window: exclude the hole rows from the draft attention.

SGLANG_DFLASH_WINDOW_HOLE_MASK (default off; 27b-draftwin 26.09.).

PROBLEM. After a phase flip or a HiCache loadback the draft has no KV for the
committed prefix: P computes no draft (user order 24.09.: no draft on P, no
draft tier in HiCache), so the window pool's per-round rebuild
(``dflash_solo_pool.rebuild_window_rows_sync_free``) maps every such prefix
row of the W=2048 window to the hole slot 0. The draft attention then reads
up to ~2046 copies of slot 0. With a zero slot each copy has logit 0 and adds
``exp(0)`` to the softmax denominator -- it is NOT attention-neutral, it
dilutes the weight of every real row.

WHAT THIS DOES (variant (b), the form is unchanged). The hole rows are removed
from the softmax as if they were masked with -inf, without a mask kernel,
without changing any length the host planned with (the draft plan stays
host-exact under SGLANG_DFLASH_PLAN_SYNC_FREE), without a host read and without
any collective:

* the rebuild already knows, on the device, which window rows are holes (the
  translated slot is 0 -- mapped draft slots are >= 1);
  ``window_hole_counts_per_token`` turns that into the number of hole rows
  each draft query token SEES (FlashInfer's sliding-window rule
  ``kv_idx + qo_len + window_left >= kv_len + qo_idx`` applied to the hole
  positions), fixed-shape;
* the draft attention runs FlashInfer's paged prefill with ``return_lse``
  (same kernel, same plan, same output bytes) and
  ``correct_window_hole_attention`` removes the holes' share exactly. All
  holes read the SAME slot 0, so they share one logit ``s0 = q.k0 * scale``
  and one value ``v0``; with ``Z = 2**lse`` (FlashInfer's LSE is base 2) the
  holes hold the fraction ``a = H * exp(s0) / Z`` of the softmax mass and

      o_masked = (o - a * v0) / (1 - a).

  That is exact for ANY slot-0 content (zero or not), so it does not rely on
  slot 0 being zero (a padded graph row or a memory-saver re-backing may
  write it).

The per-token counts live in a static device buffer the FlashInfer draft
backend owns (allocated at backend init, so CUDA-graph replays read it); the
worker writes it before the draft forward and zeroes it afterwards. A zero
count is the identity (bit-exact: the uncorrected output is returned where
the count is zero).

Ranks stay in agreement: the count is derived from the same map state on every
rank that runs the draft, the correction is per query head, local, and needs
no communication.
"""

from __future__ import annotations

import math
from typing import Optional

import torch

LOG2E = 1.0 / math.log(2.0)
# Floor for the denominator 1 - a. a <= 1 holds by construction (the holes are
# part of the softmax set); the floor only guards the rounding of a row whose
# mass sits almost entirely on the holes.
_MIN_REAL_MASS = 1e-6


def window_hole_counts_per_token(
    holes: Optional[torch.Tensor],
    lengths: torch.Tensor,
    block_size: int,
    window_left: int,
    bs: int,
) -> torch.Tensor:
    """Hole rows each draft query token sees, flat ``[bs * block_size]`` fp32.

    ``holes``: bool ``[bs, max_len]``, True where window row ``c`` of request
    ``b`` (``c < lengths[b]``) reads the hole slot; None = no holes this round.
    ``lengths``: the compact draft prefix length ``L_b`` per request (device).
    The draft block of ``block_size`` query tokens attends over
    ``kv_len = L_b + block_size`` rows (``qo_len = block_size``); with
    FlashInfer's sliding window, query ``i`` sees row ``c`` iff
    ``c + qo_len + window_left >= kv_len + i``, i.e. ``c >= L_b + i -
    window_left``. The block rows ``[L_b, L_b + block)`` are never holes.
    """
    device = lengths.device
    if holes is None or holes.numel() == 0 or bs == 0:
        return torch.zeros(bs * block_size, dtype=torch.float32, device=device)
    max_len = int(holes.shape[1])
    h = holes.to(device=device, dtype=torch.int32)
    # sfx[b, c] = holes in [c, max_len); sfx[b, max_len] = 0.
    sfx = torch.zeros((bs, max_len + 1), dtype=torch.int32, device=device)
    sfx[:, :max_len] = torch.flip(torch.cumsum(torch.flip(h, [1]), 1), [1])
    if window_left is None or int(window_left) < 0:
        lo = torch.zeros((bs, block_size), dtype=torch.int64, device=device)
    else:
        i = torch.arange(block_size, device=device, dtype=torch.int64).unsqueeze(0)
        lo = lengths.to(device=device, dtype=torch.int64).unsqueeze(1) + i
        lo = torch.clamp(lo - int(window_left), min=0, max=max_len)
    return torch.gather(sfx, 1, lo).reshape(-1).to(torch.float32)


def correct_window_hole_attention(
    o: torch.Tensor,
    lse2: torch.Tensor,
    q: torch.Tensor,
    k0: torch.Tensor,
    v0: torch.Tensor,
    hole_tok: torch.Tensor,
    sm_scale: float,
    k_scale: Optional[float] = None,
    v_scale: Optional[float] = None,
) -> torch.Tensor:
    """Attention output with the hole rows removed from the softmax.

    ``o`` ``[T, Hq, D]``: FlashInfer's output over the full window (incl.
    ``v_scale``); ``lse2`` ``[T, Hq]``: its base-2 log-sum-exp of the scaled
    logits; ``q`` ``[T, Hq, D]``: the queries it ran on; ``k0``/``v0``
    ``[Hkv, D]``: the hole slot's K/V of this layer (storage dtype);
    ``hole_tok`` ``[>= T]``: visible hole rows per query token. Query head
    ``h`` reads KV head ``h // (Hq // Hkv)`` (FlashInfer's GQA layout). Rows
    whose count is zero are returned unchanged (bit-exact).
    """
    T, Hq, D = q.shape
    Hkv = int(k0.shape[0])
    group = Hq // Hkv
    ks = 1.0 if k_scale is None else float(k_scale)
    vs = 1.0 if v_scale is None else float(v_scale)
    k0f = k0.reshape(Hkv, D).to(torch.float32).repeat_interleave(group, dim=0)
    v0f = v0.reshape(Hkv, D).to(torch.float32).repeat_interleave(group, dim=0)
    # Hole logit in FlashInfer's base-2 units: q.k0 * sm_scale * k_scale * log2e.
    s0_2 = (q.to(torch.float32) * k0f.unsqueeze(0)).sum(-1) * (
        float(sm_scale) * ks * LOG2E
    )
    h = hole_tok[:T].to(device=o.device, dtype=torch.float32).unsqueeze(1)
    has = h > 0
    a = h * torch.exp2(torch.where(has, s0_2 - lse2.to(torch.float32), 0.0))
    a = torch.clamp(a, max=1.0 - _MIN_REAL_MASS)
    a = torch.where(has, a, torch.zeros_like(a)).unsqueeze(-1)
    of = o.to(torch.float32)
    fixed = (of - a * (v0f * vs).unsqueeze(0)) / (1.0 - a)
    return torch.where(has.unsqueeze(-1), fixed, of).to(o.dtype)


def draft_hole_buffer_rows(max_bs: int, server_args) -> int:
    """Rows of the per-token count buffer: every draft query row a (padded)
    graph bucket can carry -- max batch x draft tokens per request."""
    per_req = 0
    for name in ("speculative_num_draft_tokens", "speculative_dflash_block_size"):
        v = getattr(server_args, name, None) if server_args is not None else None
        if isinstance(v, int) and v > per_req:
            per_req = v
    per_req = max(per_req, 16)
    return int(max(1, max_bs)) * per_req
