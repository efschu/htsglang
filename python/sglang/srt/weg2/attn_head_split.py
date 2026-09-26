"""AH -- attention split BY HEADS across the P stages (27B, release table row 22).

Design and model: /spinning/gpu-arb/docs/ATTN_HEAD_SPLIT.md (sec. 2, 5, 6, 8).
Micro measurement 26.09. (ah_micro_09261050.json): share 0.913, eta 1.10,
ov 0.026 ms -> the sec. 8.1 phase-2 condition is met; this module is phase 2.

WHAT IT DOES (V1, one direction)
================================
An OWNER stage (a 3080 stage, PP1/PP2) hands the attention of the LAST
G KV groups (``G * gqa`` query heads, a :class:`HeadRange`) of its LAST
``n_layers`` full-attention layers to a HELPER stage upstream of it (the 5090,
PP0). Per split chunk and delegated layer:

  owner  : pack [hdr | q_d | k_d | v_d] (bf16, after norm+RoPE) -> send on a side
           stream; write ALL kv groups into its own pool (set_kv_buffer, as
           today); attention of its own heads over its own pool; recv O_d;
           concat -> gate -> o_proj (unchanged).
  helper : a host thread sees the payload's publish flag in the pinned barlink
           segment, checks the header, then enqueues on ONE side stream:
           recv -> k_d/v_d quantised exactly like set_kv_buffer into a dense,
           POSITION-indexed mirror [cap, G, D] -> flashinfer paged causal over
           mirror[0:p+w] -> send [hdr | O_d].

Nothing else moves: the owner's pool, req_to_token, radix, HiCache publish,
the #791 rows and the D hand-off see exactly the KV they see without the split
(the owner still computes and stores K/V of all four groups). The helper holds
no weights. The mirror is P-phase scratch in the KV_CACHE memory-saver region:
it pauses/resumes with the pool across the flip, and the split state is reset
at every KV release, so no request ever reads a mirror across a flip.

WHY UPSTREAM ONLY (V1)
======================
The helper must enqueue work for chunk i before the owner needs it. PP0 is the
leader and runs chunk i FIRST, so it knows every chunk before any owner does:
no pre-announcement is needed and no deadlock is possible. The opposite
direction (5090 owner, 3080 helpers -- NVFP4 long) needs the split row of
chunk i on the helper BEFORE it blocks on chunk i's proxy (ATTN_HEAD_SPLIT.md
sec. 6); that is V2 and refused here by name.

WHERE V2 STARTS (NVFP4 long: 5090 owner -> both 3080s, half groups)
==================================================================
Kept open on purpose, nothing below assumes "whole group" or "upstream":
  * spec     -- ``Delegation.part`` already parses ``hA-B`` (query heads);
               :func:`validate` refuses it (and helper >= owner) by name.
  * header   -- carries the query-head range and the kv-group range, not a
               group count; message sizes, mirror shape and the helper plan
               are functions of :class:`HeadRange`.
  * owner    -- V2 adds the owner's own heads as up to two ranges (the second
               flashinfer call over the shared group, ``ov_partial``) in
               ``forward_extend_head_subset`` and the concat order here.
  * helper   -- :meth:`HelperThread.push` is the seam: V1 pushes from the
               helper's own rule (it runs chunk i first); V2 pushes from the
               OWNER's announcement (the owner decides alone, the helper only
               serves), enqueues the whole chunk after its first payload
               (GIL), and gets its own barlink ring for head payloads plus an
               owner-side deadline and a per-request A/B switch
               (/spinning/gpu-arb/docs/P_MICROBATCH_AH2.md).

THE SPLIT RULE (no second bookkeeping)
======================================
One pure state machine, run on EVERY P rank from the same forward batches
(:class:`SplitRule`): a forward is split iff it is a plain EXTEND of exactly one
request with ``w >= min_w`` (``min_w`` above every prefill-graph bucket, so a
split chunk is always eager) and ``p + w <= cap``, and either ``p == 0`` (the
request starts: the mirror is written from position 0) or it continues the
active request contiguously (``rid`` same, ``p == next_pos``). Anything else
ends the split for the rest of that request -- the owner then simply computes
all heads itself from its always-complete pool. Every payload carries a header
(magic, layer, rid hash, p, w, chunk no.); a mismatch is a NAMED crash-stop
(raenge-nie-uneins), never a silent wrong answer.
"""

from __future__ import annotations

import contextlib
import json
import logging
import math
import os
import threading
import time
import zlib
from collections import deque
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import torch

logger = logging.getLogger(__name__)

#: Group P's environment (launcher -> ranks). Unset/empty = off: no runtime,
#: no post, no hook does anything.
ENV = "SGLANG_P_ATTN_HEAD_SPLIT"
#: The runtime post name. The planner mirrors it
#: (PhasePoolModel.RUNTIME_BUDGET_POSTS / attn_head_split_mib).
POST_NAME = "attention head-split mirror"

MAGIC = 0x41485350  # "AHSP"
HDR_I64 = 8
HDR_BF16 = HDR_I64 * 4  # 64 bytes in a bf16 payload
#: helper flashinfer float workspace. The helper plans with
#: disable_split_kv=True (see HelperEngine), so this is not the split-kv tmp.
HELPER_FLOAT_WS_MIB = 64.0
#: flashinfer's own int workspace per wrapper.
INT_WS_MIB = 8.0
#: header ring slots on the owner (pinned host -> device).
HDR_RING = 64
#: a helper waits at most this long for one payload before it names a stall.
HELPER_STALL_S = float(os.environ.get("SGLANG_P_ATTN_HEAD_SPLIT_STALL_S", "120"))
HELPER_POLL_S = 0.0002


class AHSpecError(ValueError):
    """An --p-attn-head-split value that cannot be run, named."""


# ---------------------------------------------------------------------------
# spec
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class HeadRange:
    """The delegated part of ONE layer, in heads: query heads [q0, q1) and the
    kv groups [g0, g1) they read. Everything below the spec (payload header,
    message sizes, mirror shape, helper plan) is written against this, never
    against "whole groups" -- V2's half groups (NVFP4: heads 18-20 -> PP1,
    21-23 -> PP2) are a spec and an owner-side change, not a rewrite."""

    q0: int
    q1: int
    g0: int
    g1: int

    @property
    def n_q(self) -> int:
        return self.q1 - self.q0

    @property
    def n_kv(self) -> int:
        return self.g1 - self.g0

    def whole_groups(self, gqa: int) -> bool:
        return self.q0 == self.g0 * gqa and self.q1 == self.g1 * gqa

    def packed(self) -> Tuple[int, int]:
        return (self.q0 << 16) | self.q1, (self.g0 << 16) | self.g1


@dataclass(frozen=True)
class Delegation:
    """``part``: ``"G"`` = the last G whole kv groups (V1), or ``"hA-B"`` =
    query heads A..B inclusive (the V2 head granularity; parsed, refused by
    :func:`validate` until the owner's partial-group call exists)."""

    owner: int
    helper: int
    n_layers: int
    part: str

    def text(self) -> str:
        return f"{self.owner}:{self.helper}:{self.n_layers}:{self.part}"

    def heads(self, num_kv_heads: int, gqa: int) -> HeadRange:
        num_q = num_kv_heads * gqa
        t = self.part.strip().lower()
        if t.startswith("h"):
            a, _, b = t[1:].partition("-")
            q0, q1 = int(a), int(b or a) + 1
        else:
            g = int(t)
            q0, q1 = (num_kv_heads - g) * gqa, num_q
        if not (0 <= q0 < q1 <= num_q):
            raise AHSpecError(f"--p-attn-head-split {self.text()}: heads [{q0},{q1}) outside 0..{num_q}")
        return HeadRange(q0, q1, q0 // gqa, -(-q1 // gqa))


@dataclass(frozen=True)
class AHConfig:
    delegations: Tuple[Delegation, ...]
    min_w: int
    cap_tokens: int
    max_w: int
    head_dim: int = 256
    gqa: int = 6
    num_kv_heads: int = 4

    # -- wire format (launcher -> ranks) ---------------------------------
    def to_env(self) -> str:
        return json.dumps(
            {
                "spec": ",".join(d.text() for d in self.delegations),
                "min_w": self.min_w,
                "cap_tokens": self.cap_tokens,
                "max_w": self.max_w,
                "head_dim": self.head_dim,
                "gqa": self.gqa,
                "num_kv_heads": self.num_kv_heads,
            },
            sort_keys=True,
        )

    @classmethod
    def from_env(cls, text: str) -> "AHConfig":
        try:
            raw = json.loads(text)
        except ValueError as exc:
            raise AHSpecError(f"{ENV}: not JSON ({exc}): {text[:200]!r}")
        cfg = cls(
            delegations=parse_spec(str(raw.get("spec", ""))),
            min_w=int(raw["min_w"]),
            cap_tokens=int(raw["cap_tokens"]),
            max_w=int(raw["max_w"]),
            head_dim=int(raw.get("head_dim", 256)),
            gqa=int(raw.get("gqa", 6)),
            num_kv_heads=int(raw.get("num_kv_heads", 4)),
        )
        return cfg

    # -- derived ---------------------------------------------------------
    def helpers(self) -> Tuple[int, ...]:
        return tuple(sorted({d.helper for d in self.delegations}))

    def owners(self) -> Tuple[int, ...]:
        return tuple(sorted({d.owner for d in self.delegations}))

    def of_owner(self, stage: int) -> Optional[Delegation]:
        for d in self.delegations:
            if d.owner == stage:
                return d
        return None

    def of_helper(self, stage: int) -> Tuple[Delegation, ...]:
        return tuple(d for d in self.delegations if d.helper == stage)

    def heads_of(self, d: Delegation) -> HeadRange:
        return d.heads(self.num_kv_heads, self.gqa)

    def p2p_bytes(self) -> int:
        """Per-pair barlink p2p buffer: one message of the largest delegation
        at ``max_w`` fits in one piece (header included), 4 KiB aligned."""
        need = 0
        for d in self.delegations:
            hr = self.heads_of(d)
            need = max(need, *message_bytes(hr.n_q, hr.n_kv, self.max_w, self.head_dim))
        return int(-(-need // 4096) * 4096)


def parse_spec(text: str) -> Tuple[Delegation, ...]:
    """``"owner:helper:n_layers:G|hA-B[,...]"``; ``''``/``off`` -> ()."""
    t = (text or "").strip().lower()
    if t in ("", "off", "0", "none"):
        return ()
    out: List[Delegation] = []
    for part in t.split(","):
        bits = part.strip().split(":")
        if len(bits) != 4:
            raise AHSpecError(
                f"--p-attn-head-split entry {part!r}: expected "
                "owner:helper:n_layers:G (e.g. 2:0:2:1) or owner:helper:n_layers:hA-B"
            )
        try:
            o, h, n = (int(x) for x in bits[:3])
            d = Delegation(o, h, n, bits[3].strip())
            d.heads(4, 6)  # syntax only; the geometry check is validate()'s
        except AHSpecError:
            raise
        except ValueError:
            raise AHSpecError(f"--p-attn-head-split entry {part!r}: not integers")
        out.append(d)
    return tuple(out)


def validate(
    delegations: Sequence[Delegation],
    *,
    pp_size: int,
    num_kv_heads: int,
    gqa: int = 6,
    owner_attn_layers: Optional[Dict[int, int]] = None,
) -> None:
    """Hard refusals, each named. ``owner_attn_layers``: stage -> number of
    full-attention layers it owns (checked when known)."""
    if not delegations:
        raise AHSpecError("--p-attn-head-split: empty spec")
    owners = [d.owner for d in delegations]
    if len(set(owners)) != len(owners):
        raise AHSpecError(f"--p-attn-head-split: an owner stage appears twice ({owners})")
    helpers = {d.helper for d in delegations}
    for d in delegations:
        if not (0 <= d.owner < pp_size and 0 <= d.helper < pp_size):
            raise AHSpecError(f"--p-attn-head-split {d.text()}: stage outside PP{pp_size}")
        if d.helper >= d.owner:
            raise AHSpecError(
                f"--p-attn-head-split {d.text()}: REFUSED -- the helper must be "
                "UPSTREAM of the owner (helper < owner). A downstream helper "
                "must know chunk i's split row before it blocks on chunk i's "
                "proxy; that pre-announcement is V2 (ATTN_HEAD_SPLIT.md sec. 6)."
            )
        if d.owner in helpers:
            raise AHSpecError(
                f"--p-attn-head-split {d.text()}: stage {d.owner} would be owner "
                "AND helper; one barlink instance serves one stream per role"
            )
        hr = d.heads(num_kv_heads, gqa)
        if hr.q0 == 0:
            raise AHSpecError(f"--p-attn-head-split {d.text()}: the owner must keep at least one head")
        if not hr.whole_groups(gqa) or hr.q1 != num_kv_heads * gqa:
            raise AHSpecError(
                f"--p-attn-head-split {d.text()}: heads [{hr.q0},{hr.q1}) -- V1 delegates the "
                "LAST whole kv groups only (one flashinfer call on the owner). Part groups and "
                "non-suffix ranges are V2: the owner needs its second call over the shared "
                "group (ATTN_HEAD_SPLIT.md sec. 6, ov_partial)"
            )
        if d.n_layers < 1:
            raise AHSpecError(f"--p-attn-head-split {d.text()}: n_layers must be >= 1")
        if owner_attn_layers is not None:
            have = int(owner_attn_layers.get(d.owner, 0))
            if d.n_layers > have:
                raise AHSpecError(
                    f"--p-attn-head-split {d.text()}: stage {d.owner} owns only "
                    f"{have} full-attention layers"
                )


def config_from_env(environ=None) -> Optional[AHConfig]:
    env = os.environ if environ is None else environ
    text = (env.get(ENV) or "").strip()
    if not text or text.lower() == "off":
        return None
    return AHConfig.from_env(text)


# ---------------------------------------------------------------------------
# geometry and the post (ONE function for runtime and planner)
# ---------------------------------------------------------------------------


def owner_layers(owned_attn_layer_ids: Sequence[int], n_layers: int) -> Tuple[int, ...]:
    """The owner's delegated layers: its LAST ``n_layers`` full-attention ones."""
    ids = sorted(int(x) for x in owned_attn_layer_ids)
    return tuple(ids[len(ids) - n_layers:]) if n_layers > 0 else ()


def own_counts(hr: HeadRange) -> Tuple[int, int]:
    """(own q heads, own kv groups) of a V1 owner: the heads before the
    delegated suffix, i.e. [0, q0) over groups [0, g0)."""
    return hr.q0, hr.g0


def message_bytes(n_q: int, n_kv: int, w: int, head_dim: int) -> Tuple[int, int]:
    """(payload bytes owner->helper, output bytes helper->owner), bf16:
    [hdr | q (n_q heads) | k, v (n_kv groups)] and [hdr | O (n_q heads)]."""
    pay = (HDR_BF16 + w * (n_q + 2 * n_kv) * head_dim) * 2
    out = (HDR_BF16 + w * n_q * head_dim) * 2
    return pay, out


def stage_post_mib(cfg: AHConfig, stage: int, *, kv_elem_bytes: int = 1) -> float:
    """MiB this stage books for the split -- the runtime post
    ``attention head-split mirror`` AND the planner's
    ``PhasePoolModel.attn_head_split_mib[stage]``.

    helper: mirror (2 x cap x G x D x fp8 per delegated layer) + float/int
            workspace + one in-flight payload and output per owner.
    owner : its subset wrapper's int workspace + payload, output, own-head q
            copy of one layer at max_w.
    """
    mib = 0.0
    D = cfg.head_dim
    helped = cfg.of_helper(stage)
    if helped:
        cells = sum(d.n_layers * cfg.heads_of(d).n_kv for d in helped)
        mib += cells * 2.0 * D * kv_elem_bytes * cfg.cap_tokens / 2**20
        mib += HELPER_FLOAT_WS_MIB + INT_WS_MIB * len(helped)
        for d in helped:
            hr = cfg.heads_of(d)
            pay, out = message_bytes(hr.n_q, hr.n_kv, cfg.max_w, D)
            mib += (pay + out) / 2**20
    own = cfg.of_owner(stage)
    if own is not None:
        hr = cfg.heads_of(own)
        pay, out = message_bytes(hr.n_q, hr.n_kv, cfg.max_w, D)
        q_own, _ = own_counts(hr)
        mib += INT_WS_MIB + (pay + out + cfg.max_w * q_own * D * 2) / 2**20
    return mib


def stage_post_vector(cfg: Optional[AHConfig], pp_size: int) -> Tuple[float, ...]:
    if cfg is None:
        return ()
    return tuple(round(stage_post_mib(cfg, s), 1) for s in range(pp_size))


def host_pinned_mib(cfg: AHConfig, pp_size: int) -> float:
    """The barlink segment's pinned host bytes (all ordered pairs, x2)."""
    npairs = pp_size * (pp_size - 1)
    return (4096 + pp_size * 2 * 4096 + npairs * 2 * cfg.p2p_bytes()) / 2**20


# ---------------------------------------------------------------------------
# the rule
# ---------------------------------------------------------------------------


def rid_hash(rid: str) -> int:
    return zlib.crc32(str(rid).encode()) & 0x7FFFFFFF


@dataclass(frozen=True)
class ChunkRow:
    rid: str
    rid_hash: int
    p: int
    w: int
    chunk_no: int


class SplitRule:
    """The per-forward split decision, a pure function of the forward
    sequence. Identical on every P rank because every rank sees the same
    batches (PP) -- and it must see ALL of them: it runs in
    ``ModelRunner.forward`` (graph replays included), not in the model's
    forward. A rank blind to graph forwards diverges once a request rewinds
    (retract + re-prefill) to exactly the old ``next_pos``; the CPU test
    ``test_a_rank_blind_to_graph_forwards_diverges`` pins that."""

    def __init__(self, min_w: int, cap_tokens: int):
        self.min_w = int(min_w)
        self.cap = int(cap_tokens)
        self.active: Optional[str] = None
        self.next_pos = 0
        self.chunk_no = 0

    def reset(self) -> None:
        self.active = None
        self.next_pos = 0
        self.chunk_no = 0

    def decide(
        self,
        plain_extend: bool,
        rids: Optional[Sequence[str]],
        prefix_lens: Optional[Sequence[int]],
        extend_lens: Optional[Sequence[int]],
    ) -> Optional[ChunkRow]:
        if rids is None or prefix_lens is None or extend_lens is None:
            return None  # capture / warmup / dummy forwards: not a request
        rids = list(rids)
        if self.active is not None and self.active in rids:
            touched = True
        else:
            touched = False
        if not plain_extend or len(rids) != 1:
            if touched:
                self.reset()
            return None
        rid, p, w = rids[0], int(prefix_lens[0]), int(extend_lens[0])
        eligible = w >= self.min_w and p + w <= self.cap
        if eligible and p == 0:
            self.active, self.next_pos, self.chunk_no = rid, w, 0
            return ChunkRow(rid, rid_hash(rid), 0, w, 0)
        if eligible and rid == self.active and p == self.next_pos:
            self.chunk_no += 1
            self.next_pos = p + w
            return ChunkRow(rid, rid_hash(rid), p, w, self.chunk_no)
        if touched:
            self.reset()
        return None


def header_values(row: ChunkRow, layer_id: int, hr: HeadRange) -> List[int]:
    qs, gs = hr.packed()
    return [MAGIC, int(layer_id), row.rid_hash, row.p, row.w, row.chunk_no, qs, gs]


def header_mismatch(got: Sequence[int], want: Sequence[int]) -> Optional[str]:
    names = ("magic", "layer", "rid_hash", "p", "w", "chunk_no", "q_range", "kv_range")
    bad = [f"{n}: got {int(g)} want {int(w)}" for n, g, w in zip(names, got, want) if int(g) != int(w)]
    return "; ".join(bad) if bad else None


# ---------------------------------------------------------------------------
# payload layout and the math every side must agree on
# ---------------------------------------------------------------------------


def pack_payload(hdr_i64: torch.Tensor, q_d: torch.Tensor, k_d: torch.Tensor, v_d: torch.Tensor) -> torch.Tensor:
    """[hdr(64 B) | q_d | k_d | v_d], block-major so every unpacked view is
    contiguous. q_d [w, gs*G, D], k_d/v_d [w, G, D], all bf16."""
    nq, nk = q_d.numel(), k_d.numel()
    buf = torch.empty(HDR_BF16 + nq + 2 * nk, dtype=q_d.dtype, device=q_d.device)
    buf[:HDR_BF16].copy_(hdr_i64.view(q_d.dtype))
    buf[HDR_BF16:HDR_BF16 + nq].view(q_d.shape).copy_(q_d)
    buf[HDR_BF16 + nq:HDR_BF16 + nq + nk].view(k_d.shape).copy_(k_d)
    buf[HDR_BF16 + nq + nk:].view(v_d.shape).copy_(v_d)
    return buf


def unpack_payload(buf: torch.Tensor, w: int, n_q: int, n_kv: int, head_dim: int):
    nq = w * n_q * head_dim
    nk = w * n_kv * head_dim
    hdr = buf[:HDR_BF16].view(torch.int64)
    q_d = buf[HDR_BF16:HDR_BF16 + nq].view(w, n_q, head_dim)
    k_d = buf[HDR_BF16 + nq:HDR_BF16 + nq + nk].view(w, n_kv, head_dim)
    v_d = buf[HDR_BF16 + nq + nk:HDR_BF16 + nq + 2 * nk].view(w, n_kv, head_dim)
    return hdr, q_d, k_d, v_d


def quantize_like_pool(x: torch.Tensor, scale: Optional[float], dtype: torch.dtype) -> torch.Tensor:
    """What ``MHATokenToKVPool.set_kv_buffer`` stores for ``x``: an in-dtype
    division by the scale (``cache_k.div_(k_scale)``), then ``.to(dtype)``.
    Out of place here, same rounding."""
    if x.dtype == dtype:
        return x
    y = x.div(scale) if scale is not None else x
    return y.to(dtype)


def reference_attention(
    q: torch.Tensor,
    k_all: torch.Tensor,
    v_all: torch.Tensor,
    p: int,
    sm_scale: float,
    k_scale: Optional[float] = None,
    v_scale: Optional[float] = None,
) -> torch.Tensor:
    """The chunk attention the paged kernel computes, as plain math: queries
    at positions p..p+w-1 over keys 0..p+w-1 (causal, bottom-right aligned),
    stored K/V dequantised by the scales. q [w, Hq, D], k/v [>= p+w, Hk, D].
    One GQA group per iteration, so a head's result never depends on how many
    other heads share the call (CPU bit-equality of the split)."""
    w, hq, d = q.shape
    hk = k_all.shape[1]
    gs = hq // hk
    L = p + w
    kf = k_all[:L].float() * (1.0 if k_scale is None else float(k_scale))
    vf = v_all[:L].float() * (1.0 if v_scale is None else float(v_scale))
    qf = q.float()
    pos_q = torch.arange(p, p + w).unsqueeze(1)
    pos_k = torch.arange(L).unsqueeze(0)
    mask = pos_k <= pos_q
    out = torch.empty(w, hq, d, dtype=torch.float32)
    for g in range(hk):
        # contiguous operands: the result must not depend on the caller's
        # layout (pool view, mirror, full tensor)
        qs = qf[:, g * gs:(g + 1) * gs, :].contiguous()  # [w, gs, D]
        kg, vg = kf[:, g, :].contiguous(), vf[:, g, :].contiguous()
        s = torch.einsum("thd,ld->htl", qs, kg) * sm_scale
        s = s.masked_fill(~mask.unsqueeze(0), float("-inf"))
        a = torch.softmax(s, dim=-1)
        out[:, g * gs:(g + 1) * gs, :] = torch.einsum("htl,ld->thd", a, vg)
    return out.to(q.dtype)


# ---------------------------------------------------------------------------
# device shims (CPU tests run the same code without streams)
# ---------------------------------------------------------------------------


def _is_cuda(dev) -> bool:
    return torch.device(dev).type == "cuda"


def _stream_ctx(stream):
    return torch.cuda.stream(stream) if stream is not None else contextlib.nullcontext()


def _new_event(dev):
    return torch.cuda.Event() if _is_cuda(dev) else None


# ---------------------------------------------------------------------------
# transport adapter (barlink host p2p; a fake in tests)
# ---------------------------------------------------------------------------


class BarlinkP2P:
    """The four things the split needs from ``BarlinkHostTransport``: send and
    recv (stream-ordered, GPU-driven) and two HOST reads of the pinned segment
    -- the publish counter of a pair and the first 64 bytes of a published
    message -- so the helper never parks a spin kernel on its card while it
    waits for an owner."""

    def __init__(self, transport):
        import numpy as np

        self._np = np
        self.tr = transport
        self.rank = transport.rank
        self.p2p_bytes = int(transport.p2p_bytes)

    def send(self, t: torch.Tensor, dst: int) -> None:
        self.tr.send(t, dst)

    def recv(self, t: torch.Tensor, src: int) -> None:
        self.tr.recv(t, src)

    def pieces(self, nbytes: int) -> int:
        return max(1, -(-int(nbytes) // self.p2p_bytes))

    def published(self, src: int) -> int:
        # row src, column 2 + dst: "p2p publish, src -> dst" (barlink_host.py)
        return int(self.tr._flags_np[src, 2 + self.rank])

    def peek_header(self, src: int, seq: int) -> List[int]:
        tr = self.tr
        off = tr._p2p_base_off + (tr._pair_index(src, self.rank) * 2 + (seq & 1)) * tr.p2p_bytes
        return [int(x) for x in self._np.frombuffer(tr._shm.buf, dtype=self._np.int64, count=HDR_I64, offset=off)]


# ---------------------------------------------------------------------------
# helper engine: one layer step (math) + the host thread (scheduling)
# ---------------------------------------------------------------------------


@dataclass
class LayerInfo:
    layer_id: int
    sm_scale: float
    logit_cap: float
    k_scale: Optional[float]
    v_scale: Optional[float]


@dataclass
class HelperJob:
    owner: int
    row: ChunkRow
    layers: Tuple[int, ...]
    next_layer: int = 0


class HelperEngine:
    """Everything the helper does for one owner layer of one chunk, given a
    received payload. ``attend(owner, layer, q, mirror_k, mirror_v, p, w,
    info)`` is the kernel (flashinfer on the card, :func:`reference_attention`
    in tests)."""

    def __init__(self, cfg: AHConfig, device, kv_dtype: torch.dtype, attend: Callable, alloc_ctx=None):
        self.cfg = cfg
        self.device = device
        self.kv_dtype = kv_dtype
        self.attend = attend
        self.mirror: Dict[Tuple[int, int], Tuple[torch.Tensor, torch.Tensor]] = {}
        self._alloc_ctx = alloc_ctx or contextlib.nullcontext

    def allocate(self, owner: int, layers: Sequence[int], n_kv: int) -> None:
        with self._alloc_ctx():
            for lid in layers:
                shape = (self.cfg.cap_tokens, n_kv, self.cfg.head_dim)
                self.mirror[(owner, int(lid))] = (
                    torch.zeros(shape, dtype=self.kv_dtype, device=self.device),
                    torch.zeros(shape, dtype=self.kv_dtype, device=self.device),
                )

    def mirror_mib(self) -> float:
        return sum(k.numel() * k.element_size() * 2 for k, _ in self.mirror.values()) / 2**20

    def step(self, owner: int, info: LayerInfo, payload: torch.Tensor, row: ChunkRow, hr: HeadRange) -> torch.Tensor:
        """Mirror write + attention for one received payload; returns the
        output message [hdr | O_d]."""
        hdr, q_d, k_d, v_d = unpack_payload(payload, row.w, hr.n_q, hr.n_kv, self.cfg.head_dim)
        mk, mv = self.mirror[(owner, info.layer_id)]
        p, w = row.p, row.w
        mk[p:p + w].copy_(quantize_like_pool(k_d, info.k_scale, self.kv_dtype))
        mv[p:p + w].copy_(quantize_like_pool(v_d, info.v_scale, self.kv_dtype))
        o = self.attend(owner, info, q_d, mk, mv, p, w)
        out = torch.empty(HDR_BF16 + o.numel(), dtype=payload.dtype, device=payload.device)
        out[:HDR_BF16].copy_(payload[:HDR_BF16])
        out[HDR_BF16:].copy_(o.reshape(-1))
        return out


class HelperThread:
    """Drives :class:`HelperEngine` from the owners' publish flags.

    Jobs are pushed by the helper's own forward hook (PP0 runs chunk i before
    any owner does). The thread polls the pinned publish counter of each
    owner -> helper pair; when the next payload is there it checks the header
    on the host and only THEN enqueues recv -> step -> send on the side stream.
    No spin kernel waits on the helper card, and flashinfer's plan() is never
    re-run while a previous plan of the same wrapper is still queued."""

    def __init__(self, cfg: AHConfig, wire: BarlinkP2P, engine: HelperEngine, infos: Dict[Tuple[int, int], LayerInfo],
                 stream=None, plan: Optional[Callable] = None, device=None):
        self.cfg = cfg
        self.wire = wire
        self.engine = engine
        self.infos = infos
        self.stream = stream
        self.plan = plan
        self.device = device
        self.jobs: Dict[int, deque] = {d.owner: deque() for d in cfg.of_helper(wire.rank)}
        self.rx_seq: Dict[int, int] = {o: 1 for o in self.jobs}
        self.last_ev: Dict[int, object] = {o: None for o in self.jobs}
        self.failure: Optional[str] = None
        self.stats = {"layers": 0, "bytes_in": 0, "bytes_out": 0}
        self._stop = threading.Event()
        self._lock = threading.Lock()
        self._thread: Optional[threading.Thread] = None
        self._wait_since: Dict[int, Optional[float]] = {o: None for o in self.jobs}

    def push(self, row: ChunkRow, layers_of: Dict[int, Tuple[int, ...]]) -> None:
        with self._lock:
            for owner, q in self.jobs.items():
                q.append(HelperJob(owner, row, layers_of[owner]))

    def pending(self) -> int:
        with self._lock:
            return sum(len(q) for q in self.jobs.values())

    def drop_all(self) -> int:
        with self._lock:
            n = sum(len(q) for q in self.jobs.values())
            for q in self.jobs.values():
                q.clear()
            return n

    def start(self) -> None:
        self._thread = threading.Thread(target=self._run, name="ah-helper", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()

    def poll_once(self) -> int:
        """One pass over the owners; returns how many layer steps it issued.
        Public so the CPU tests drive the thread's logic without a thread."""
        issued = 0
        for owner, q in self.jobs.items():
            with self._lock:
                job = q[0] if q else None
            if job is None:
                continue
            d = self.cfg.of_owner(owner)
            seq = self.rx_seq[owner]
            if self.wire.published(owner) < seq:
                t0 = self._wait_since[owner]
                now = time.monotonic()
                if t0 is None:
                    self._wait_since[owner] = now
                elif now - t0 > HELPER_STALL_S:
                    raise RuntimeError(
                        f"AH helper stall: waited {now - t0:.0f} s for owner PP{owner} "
                        f"layer {job.layers[job.next_layer]} of chunk {job.row.chunk_no} "
                        f"(p={job.row.p} w={job.row.w}); the owner never sent -- the split "
                        "rule diverged between ranks or the owner died"
                    )
                continue
            self._wait_since[owner] = None
            lid = job.layers[job.next_layer]
            want = header_values(job.row, lid, self.cfg.heads_of(d))
            got = self.wire.peek_header(owner, seq)
            bad = header_mismatch(got, want)
            if bad:
                raise RuntimeError(
                    f"AH RANKS DISAGREE (raenge-nie-uneins): owner PP{owner} sent "
                    f"a payload the helper did not expect for layer {lid}: {bad}"
                )
            self._issue(owner, d, job, lid)
            issued += 1
            job.next_layer += 1
            if job.next_layer == len(job.layers):
                with self._lock:
                    q.popleft()
        return issued

    def _issue(self, owner: int, d: Delegation, job: HelperJob, lid: int) -> None:
        hr = self.cfg.heads_of(d)
        pay_b, out_b = message_bytes(hr.n_q, hr.n_kv, job.row.w, self.cfg.head_dim)
        if job.next_layer == 0 and self.plan is not None:
            ev = self.last_ev[owner]
            if ev is not None:
                ev.synchronize()  # the wrapper's previous plan has been consumed
            with _stream_ctx(self.stream):
                self.plan(owner, job.row.p, job.row.w, hr.n_q, hr.n_kv)
        with _stream_ctx(self.stream):
            buf = torch.empty(pay_b // 2, dtype=torch.bfloat16, device=self.device)
            self.wire.recv(buf, owner)
            out = self.engine.step(owner, self.infos[(owner, lid)], buf, job.row, hr)
            self.wire.send(out, owner)
            if self.stream is not None:
                ev = torch.cuda.Event()
                ev.record(self.stream)
                self.last_ev[owner] = ev
        self.rx_seq[owner] += self.wire.pieces(pay_b)
        self.stats["layers"] += 1
        self.stats["bytes_in"] += pay_b
        self.stats["bytes_out"] += out_b

    def _run(self) -> None:
        if self.device is not None and _is_cuda(self.device):
            torch.cuda.set_device(self.device)
        try:
            while not self._stop.is_set():
                if self.poll_once() == 0:
                    time.sleep(HELPER_POLL_S)
        except BaseException as exc:  # noqa: BLE001 -- named, surfaced on the forward thread
            self.failure = f"{type(exc).__name__}: {exc}"
            logger.error("AH helper thread died: %s", self.failure)


# ---------------------------------------------------------------------------
# owner side
# ---------------------------------------------------------------------------


class OwnerSide:
    """One delegated layer on the owner. ``own_attention(q_own, k, v)`` stores
    ALL kv groups into the pool and returns the own heads' output
    (``FlashInferAttnBackend.forward_extend_head_subset`` on the card)."""

    def __init__(self, cfg: AHConfig, d: Delegation, wire, device, stream=None):
        self.cfg = cfg
        self.d = d
        self.wire = wire
        self.device = device
        self.stream = stream
        self.hr = cfg.heads_of(d)
        self.q_own, self.kv_own = own_counts(self.hr)
        self._ring = None
        self._ring_ev: List[object] = []
        self._ring_i = 0
        if _is_cuda(device):
            self._ring = torch.empty(HDR_RING, HDR_I64, dtype=torch.int64).pin_memory()
            self._ring_ev = [None] * HDR_RING
        self.stats = {"layers": 0}

    def _hdr_dev(self, vals: Sequence[int]) -> torch.Tensor:
        if self._ring is None:
            return torch.tensor(vals, dtype=torch.int64, device=self.device)
        i = self._ring_i
        self._ring_i = (i + 1) % HDR_RING
        ev = self._ring_ev[i]
        if ev is not None:
            ev.synchronize()
        self._ring[i].copy_(torch.tensor(vals, dtype=torch.int64))
        dev = self._ring[i].to(self.device, non_blocking=True)
        ev = torch.cuda.Event()
        ev.record()
        self._ring_ev[i] = ev
        return dev

    def attention(self, row: ChunkRow, layer_id: int, q, k, v, own_attention: Callable) -> torch.Tensor:
        cfg, hr = self.cfg, self.hr
        D, H, Hk = cfg.head_dim, cfg.num_kv_heads * cfg.gqa, cfg.num_kv_heads
        w = q.shape[0]
        qv = q.view(w, H, D)
        kv3 = k.view(w, Hk, D)
        vv3 = v.view(w, Hk, D)
        hdr = self._hdr_dev(header_values(row, layer_id, hr))
        # Pack BEFORE the pool write: set_kv_buffer may divide k/v in place.
        payload = pack_payload(hdr, qv[:, hr.q0:hr.q1], kv3[:, hr.g0:hr.g1], vv3[:, hr.g0:hr.g1])
        sent = None
        if self.stream is not None:
            ready = torch.cuda.Event()
            ready.record()
            self.stream.wait_event(ready)
            with torch.cuda.stream(self.stream):
                self.wire.send(payload, self.d.helper)
                sent = torch.cuda.Event()
                sent.record(self.stream)
            payload.record_stream(self.stream)
        else:
            self.wire.send(payload, self.d.helper)
        o_own = own_attention(qv[:, :self.q_own].contiguous(), kv3, vv3, self.kv_own)
        _, out_b = message_bytes(hr.n_q, hr.n_kv, w, D)
        obuf = torch.empty(out_b // 2, dtype=q.dtype, device=q.device)
        if sent is not None:
            torch.cuda.current_stream().wait_event(sent)  # one stream per barlink op at a time
        self.wire.recv(obuf, self.d.helper)
        if _is_cuda(q.device):
            torch._assert_async(torch.all(obuf[:HDR_BF16].view(torch.int64) == hdr))
        elif not torch.equal(obuf[:HDR_BF16].view(torch.int64), hdr):
            raise RuntimeError("AH RANKS DISAGREE: the helper's output header does not echo the payload's")
        self.stats["layers"] += 1
        # V1: own heads [0, q0) + delegated suffix [q0, H) = the stock order.
        return torch.cat([o_own.reshape(w, -1), obuf[HDR_BF16:].view(w, hr.n_q * D)], dim=1)


# ---------------------------------------------------------------------------
# runtime (one per P rank process)
# ---------------------------------------------------------------------------


_RUNTIME: Optional["AHRuntime"] = None


def runtime() -> Optional["AHRuntime"]:
    return _RUNTIME


def owner_split_now(layer_id: int) -> bool:
    """The model's per-layer question: does THIS forward split THIS layer?"""
    rt = _RUNTIME
    return rt is not None and rt.owner_split_active(layer_id)


class AHRuntime:
    def __init__(self, cfg: AHConfig, pp_rank: int, pp_size: int):
        self.cfg = cfg
        self.pp_rank = pp_rank
        self.pp_size = pp_size
        self.rule = SplitRule(cfg.min_w, cfg.cap_tokens)
        self.row: Optional[ChunkRow] = None
        self.owner: Optional[OwnerSide] = None
        self.owner_layer_ids: frozenset = frozenset()
        self.helper: Optional[HelperThread] = None
        self.layers_of: Dict[int, Tuple[int, ...]] = {}
        self.n_split = 0
        self.n_forwards = 0
        self.role = (
            "owner" if cfg.of_owner(pp_rank) is not None
            else "helper" if cfg.of_helper(pp_rank) else "none"
        )

    # -- hooks --------------------------------------------------------------
    def on_forward(self, forward_batch) -> None:
        if self.helper is not None and self.helper.failure:
            raise RuntimeError(f"AH helper failed: {self.helper.failure}")
        plain = _plain_extend(forward_batch)
        self.row = self.rule.decide(
            plain,
            getattr(forward_batch, "rids", None),
            getattr(forward_batch, "extend_prefix_lens_cpu", None),
            getattr(forward_batch, "extend_seq_lens_cpu", None),
        )
        self.n_forwards += 1
        if self.row is None:
            return
        self.n_split += 1
        if self.helper is not None:
            self.helper.push(self.row, self.layers_of)
        if self.n_split in (1, 2, 4, 8, 16, 64, 256, 1024) or self.n_split % 4096 == 0:
            logger.info(
                "AH-SPLIT %s PP%d chunk rid=%s p=%d w=%d no=%d (split forwards %d of %d)",
                self.role, self.pp_rank, self.row.rid, self.row.p, self.row.w,
                self.row.chunk_no, self.n_split, self.n_forwards,
            )

    def owner_split_active(self, layer_id: int) -> bool:
        return self.row is not None and self.owner is not None and layer_id in self.owner_layer_ids

    def owner_attention(self, radix_attn, q, k, v, forward_batch) -> torch.Tensor:
        from sglang.srt.model_executor.forward_context import get_attn_backend

        backend = get_attn_backend()
        fn = getattr(backend, "forward_extend_head_subset", None)
        if fn is None:
            raise RuntimeError(
                f"AH owner: attention backend {type(backend).__name__} has no "
                "forward_extend_head_subset (flashinfer only)"
            )

        def own(q_own, k_all, v_all, kv_own):
            return fn(q_own, k_all, v_all, radix_attn, forward_batch, kv_own)

        return self.owner.attention(self.row, radix_attn.layer_id, q, k, v, own)

    def on_kv_release(self) -> None:
        """The KV region (and the mirror in it) is about to be unmapped."""
        self.rule.reset()
        self.row = None
        if self.helper is not None:
            dropped = self.helper.drop_all()
            if self.helper.stream is not None:
                self.helper.stream.synchronize()
            if dropped:
                logger.warning("AH: %d helper job(s) dropped at KV release", dropped)


def _plain_extend(forward_batch) -> bool:
    try:
        from sglang.srt.model_executor.forward_batch_info import ForwardMode
    except Exception:  # noqa: BLE001 -- CPU tests pass a stand-in
        ForwardMode = None
    fm = getattr(forward_batch, "forward_mode", None)
    if fm is None:
        return False
    if ForwardMode is not None and fm != ForwardMode.EXTEND:
        return False
    if ForwardMode is None and str(fm) not in ("EXTEND", "ForwardMode.EXTEND"):
        return False
    return getattr(forward_batch, "spec_info", None) is None


# ---------------------------------------------------------------------------
# install (model runner, once per P rank process, after the attention backend)
# ---------------------------------------------------------------------------


def _find_language_model(model):
    from sglang.srt.models.qwen3_5 import Qwen3_5AttentionDecoderLayer, Qwen3_5ForCausalLM

    for m in model.modules():
        if isinstance(m, Qwen3_5ForCausalLM):
            return m, Qwen3_5AttentionDecoderLayer
    raise AHSpecError("--p-attn-head-split: no Qwen3_5ForCausalLM in this model (27B only)")


def install(runner) -> Optional[AHRuntime]:
    """Build the split for this rank. Idempotent; a no-op when the env is
    unset (the default form is untouched). Every P rank calls it at the same
    point (collective rendezvous)."""
    global _RUNTIME
    cfg = config_from_env()
    if cfg is None or getattr(runner, "is_draft_worker", False):
        return None
    if _RUNTIME is not None:
        return _RUNTIME
    import torch.distributed as dist

    from sglang.srt.distributed.parallel_state import get_pp_group
    from sglang.srt.models.qwen3_5 import owned_layer_ids

    pp = get_pp_group()
    pp_rank, pp_size = pp.rank_in_group, pp.world_size
    lm, attn_cls = _find_language_model(runner.model)
    backend = getattr(runner, "attn_backend", None)
    _check_backend(backend, runner)
    mc = runner.model_config
    num_q = int(mc.hf_text_config.num_attention_heads)
    num_kv = int(mc.hf_text_config.num_key_value_heads)
    if num_q != cfg.gqa * num_kv or int(mc.head_dim) != cfg.head_dim or num_kv != cfg.num_kv_heads:
        raise AHSpecError(
            f"--p-attn-head-split: geometry mismatch (model q={num_q} kv={num_kv} "
            f"head_dim={mc.head_dim}; spec gqa={cfg.gqa} kv={cfg.num_kv_heads} D={cfg.head_dim})"
        )
    my_attn = [i for i in owned_layer_ids(lm.layers, lm.start_layer, lm.end_layer)
               if isinstance(lm.layers[i], attn_cls)]
    gathered: List[dict] = [None] * pp_size  # type: ignore[list-item]
    mine: Dict[str, object] = {"rank": pp_rank, "n_attn": len(my_attn), "layers": {}}
    d_own = cfg.of_owner(pp_rank)
    if d_own is not None:
        for lid in owner_layers(my_attn, d_own.n_layers):
            a = lm.layers[lid].attn
            mine["layers"][lid] = (float(a.scaling), float(a.logit_cap or 0.0),
                                   None if a.k_scale_float is None else float(a.k_scale_float),
                                   None if a.v_scale_float is None else float(a.v_scale_float))
    dist.all_gather_object(gathered, mine, group=pp.cpu_group)
    validate(cfg.delegations, pp_size=pp_size, num_kv_heads=num_kv, gqa=cfg.gqa,
             owner_attn_layers={int(g["rank"]): int(g["n_attn"]) for g in gathered})

    from sglang.srt.distributed.device_communicators.barlink_host import BarlinkHostTransport

    device = torch.device("cuda", torch.cuda.current_device())
    transport = BarlinkHostTransport(pp.cpu_group, device, slot_bytes=4096, p2p_bytes=cfg.p2p_bytes())
    wire = BarlinkP2P(transport)
    rt = AHRuntime(cfg, pp_rank, pp_size)
    for g in gathered:
        if cfg.of_owner(int(g["rank"])) is not None:
            rt.layers_of[int(g["rank"])] = tuple(sorted(int(x) for x in g["layers"]))
    if d_own is not None:
        rt.owner = OwnerSide(cfg, d_own, wire, device, stream=torch.cuda.Stream(device))
        rt.owner_layer_ids = frozenset(rt.layers_of[pp_rank])
        for lid in rt.owner_layer_ids:
            lm.layers[lid]._ah_split = True
    helped = cfg.of_helper(pp_rank)
    if helped:
        from sglang.srt.constants import GPU_MEMORY_TYPE_KV_CACHE

        adapter = runner.memory_saver_adapter
        engine_kv_dtype = runner.kv_cache_dtype
        infos: Dict[Tuple[int, int], LayerInfo] = {}
        for g in gathered:
            for lid, (sc, cap, ks, vs) in g["layers"].items():
                infos[(int(g["rank"]), int(lid))] = LayerInfo(int(lid), sc, cap, ks, vs)
        kernel = FlashinferHelperKernel(cfg, device, engine_kv_dtype,
                                        alloc_ctx=lambda: adapter.region(GPU_MEMORY_TYPE_KV_CACHE))
        engine = HelperEngine(cfg, device, engine_kv_dtype, kernel.attend,
                              alloc_ctx=lambda: adapter.region(GPU_MEMORY_TYPE_KV_CACHE))
        for d in helped:
            engine.allocate(d.owner, rt.layers_of[d.owner], cfg.heads_of(d).n_kv)
            kernel.add_owner(d.owner)
        stream = torch.cuda.Stream(device, priority=-1)
        rt.helper = HelperThread(cfg, wire, engine, infos, stream=stream, plan=kernel.plan, device=device)
        rt.helper.start()
    runner._ah_runtime = rt
    _RUNTIME = rt
    logger.warning(
        "AH-SPLIT armed PP%d/%d role=%s spec=%s min_w=%d cap=%d post=%.1f MiB "
        "(ATTN_HEAD_SPLIT.md; %s) barlink p2p %.2f MiB/pair, pinned host %.1f MiB",
        pp_rank, pp_size, rt.role, ",".join(d.text() for d in cfg.delegations), cfg.min_w,
        cfg.cap_tokens, stage_post_mib(cfg, pp_rank),
        ("owner layers %s -> PP%d, q heads %s" % (
            sorted(rt.owner_layer_ids), d_own.helper, cfg.heads_of(d_own))) if d_own else
        ("helper for %s, mirror %.1f MiB" % (
            {d.owner: rt.layers_of[d.owner] for d in helped}, rt.helper.engine.mirror_mib())) if helped else "idle",
        cfg.p2p_bytes() / 2**20, host_pinned_mib(cfg, pp_size),
    )
    return rt


def _check_backend(backend, runner) -> None:
    name = type(backend).__name__
    if name != "FlashInferAttnBackend":
        raise AHSpecError(f"--p-attn-head-split: attention backend {name}, flashinfer only")
    if getattr(backend, "uneven_dcp", False):
        raise AHSpecError("--p-attn-head-split: not with uneven DCP (the P stages are not a DCP group)")
    paged_only = bool(
        getattr(backend, "is_multimodal", False)
        or getattr(backend, "enable_mis", False)
        or getattr(backend, "use_paged", False)
        or getattr(backend, "enable_deterministic", False)
    )
    if not paged_only:
        raise AHSpecError(
            "--p-attn-head-split: this backend runs extend as ragged+paged; V1 "
            "splits the paged-only form (the 27B is multimodal -> paged-only)"
        )
    sa = getattr(runner, "server_args", None)
    if int(getattr(sa, "tp_size", 1) or 1) != 1:
        raise AHSpecError("--p-attn-head-split: pure PP stages only (tp_size 1)")
    active = getattr(sa, "uneven_memory_budgets_active", None)
    if active is None or not active():
        raise AHSpecError(
            "--p-attn-head-split: its post is booked in the ABSOLUTE-budget sizing "
            "branch (--rank-gpu-memory-mib); without it the mirror would be unbooked"
        )


class FlashinferHelperKernel:
    """The helper's flashinfer paged wrappers, one per owner (they run on the
    one side stream, so they may share one float workspace). Planned once per
    chunk with host indptrs and ``disable_split_kv=True``: no split-kv tmp
    buffer (the post stays small) and more SMs left to the helper's own stage,
    which keeps running beside it."""

    def __init__(self, cfg: AHConfig, device, kv_dtype, alloc_ctx):
        self.cfg = cfg
        self.device = device
        self.kv_dtype = kv_dtype
        self._alloc_ctx = alloc_ctx
        with alloc_ctx():
            self.float_ws = torch.empty(int(HELPER_FLOAT_WS_MIB * 2**20), dtype=torch.uint8, device=device)
            self.idx = torch.arange(cfg.cap_tokens, dtype=torch.int32, device=device)
        self.wrappers: Dict[int, object] = {}
        self._last_page = torch.ones(1, dtype=torch.int32)

    def add_owner(self, owner: int) -> None:
        from flashinfer import BatchPrefillWithPagedKVCacheWrapper

        # its 8 MiB int workspace too: P-phase scratch, paused with the KV
        # region, never resident through D (the post books it on P only)
        with self._alloc_ctx():
            self.wrappers[owner] = BatchPrefillWithPagedKVCacheWrapper(self.float_ws, "NHD")

    def plan(self, owner: int, p: int, w: int, n_q: int, n_kv: int) -> None:
        wr = self.wrappers[owner]
        qo = torch.tensor([0, w], dtype=torch.int32)
        kv = torch.tensor([0, p + w], dtype=torch.int32)
        wr.plan(qo, kv, self.idx[:p + w], self._last_page, n_q, n_kv,
                self.cfg.head_dim, 1, causal=True, q_data_type=torch.bfloat16,
                kv_data_type=self.kv_dtype, non_blocking=True, disable_split_kv=True)

    def attend(self, owner, info: LayerInfo, q_d, mk, mv, p, w):
        return self.wrappers[owner].forward(
            q_d, (mk, mv), causal=True, sm_scale=info.sm_scale, window_left=-1,
            logits_soft_cap=info.logit_cap, k_scale=info.k_scale, v_scale=info.v_scale,
        )


def reset_for_tests() -> None:
    global _RUNTIME
    if _RUNTIME is not None and _RUNTIME.helper is not None:
        _RUNTIME.helper.stop()
    _RUNTIME = None
