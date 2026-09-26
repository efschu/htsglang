"""AH -- attention split BY HEADS across the P stages (27B, release table row 22).

Design and model: /spinning/gpu-arb/docs/ATTN_HEAD_SPLIT.md (sec. 2, 5, 6, 8).
Micro measurement 26.09. (ah_micro_09261050.json): share 0.913, eta 1.10,
ov 0.026 ms -> the sec. 8.1 phase-2 condition is met; this module is phase 2.

WHAT IT DOES (V1 form; V2 below)
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

V2: DOWNSTREAM HELPERS AS SERVERS (NVFP4 long, 5090 owner -> both 3080s)
=====================================================================
Spec ``0:1:4:h18-20,0:2:4:h21-23`` (P_MICROBATCH_AH2.md sec. 4): PP0 keeps
heads 0-17 (groups 0-2, one flashinfer call) and hands the two halves of
group 3 to PP1 and PP2, on its last 4 full-attention layers. Any head range
works that is whole groups or lies inside ONE group; the owner's own heads
are the complement, cut into :func:`own_pieces` (whole-group runs and partial
groups, one flashinfer call each, concatenated back in head order).

  * The OWNER DECIDES ALONE. Only its :class:`SplitRule` counts; a downstream
    helper never sees chunk i's batch before PP0 needs the answer, so it does
    not decide, it SERVES (:class:`ServerHelper`): the first payload header of
    a chunk (layer = the owner's first delegated layer) announces the chunk;
    the helper checks continuity itself (p == 0 starts a request, otherwise
    same rid hash, p == next_pos, chunk_no + 1 -- anything else is a named
    crash-stop) and enqueues the WHOLE chunk (recv -> mirror -> attend -> send
    for every delegated layer) on its side stream at once. Later headers are
    checked on the device. One Python wake-up per chunk, not per layer (GIL).
    An announced chunk is a contract: the owner sends every layer's payload,
    whatever it does with the answers.
  * NO DEADLOCK, by construction. PP1/PP2 sit downstream: while PP0 waits for
    O of chunk i, PP1's main thread is blocked on PP0's frame of chunk i (or
    still in chunk i-1). The helper work therefore never runs on the main
    thread or main stream: its own host thread polls pinned publish counters
    and its own CUDA stream (priority -1) does the work; nothing on it waits
    for the helper's main stream, and nothing on the owner's main stream waits
    for the owner's side stream except the one output event the host has
    already SEEN published. Pinned by ``test_weg2_attn_head_split_v2_0926``:
    a simulated 3-stage pipeline with blocking frames completes; the naive
    variant (helper served between the stage's own forwards) hangs.
  * DEADLINE AND FALLBACK on the owner (``deadline_ms``): the owner's host
    waits for a helper's output to be PUBLISHED (pinned counter, no spin
    kernel on the 5090) at most ``deadline_ms`` after its own heads finished.
    Past it the owner computes that range itself from its always-complete pool
    (bit-identical input: the pool holds all groups), keeps the late output as
    owed (drained later on its side stream, so barlink's two slots never fill
    up), stops waiting for the rest of the chunk, and ENDS the split for the
    rest of the request. The helper is never told: its mirror just stops being
    read. ``deadline_ms <= 0`` = no fallback, a named stall after
    HELPER_STALL_S.
  * OWN RING: the split's :class:`BarlinkHostTransport` is its own pinned
    segment with its own per-pair sequences, never the PP proxy transport's,
    so a proxy frame and a head payload can never share a FIFO.
  * A/B PER REQUEST (``ab="alt"``): every second new request is not split
    (pure function of the request sequence, logged ``AH-AB ... arm=``), so one
    boot measures both arms.

V1 (UPSTREAM HELPER, INT8: 3080 owner -> 5090 helper) is unchanged: the
helper runs chunk i first, pushes its own jobs from the same rule, the owner
waits on the device (no deadline -- the helper is never late on purpose).

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
#: V2 owner: how long past its own heads the owner waits for a downstream
#: helper's output before it computes the range itself (AHConfig.deadline_ms).
DEADLINE_MS_DEFAULT = 20.0
#: V2 server helper ranks: GIL switch interval (default 5 ms would put up to
#: one interval of latency on every chunk announcement, P_MICROBATCH_AH2 4.1-2).
SERVER_SWITCH_INTERVAL_S = 5e-4


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
    """``part``: ``"G"`` = the last G whole kv groups, or ``"hA-B"`` = query
    heads A..B inclusive (whole groups or inside ONE group). ``helper <
    owner`` = V1 (upstream helper runs its own rule), ``helper > owner`` = V2
    (downstream helper serves the owner's announcements)."""

    owner: int
    helper: int
    n_layers: int
    part: str

    def text(self) -> str:
        return f"{self.owner}:{self.helper}:{self.n_layers}:{self.part}"

    @property
    def downstream(self) -> bool:
        """V2: the helper sits behind the owner in the pipeline -> it serves."""
        return self.helper > self.owner

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
    #: V2 owner deadline past its own heads; <= 0 = wait (named stall).
    deadline_ms: float = DEADLINE_MS_DEFAULT
    #: "" = split every eligible request; "alt" = every second new request
    #: runs unsplit (A/B inside one boot).
    ab: str = ""

    # -- wire format (launcher -> ranks) ---------------------------------
    def to_env(self) -> str:
        raw = {
            "spec": ",".join(d.text() for d in self.delegations),
            "min_w": self.min_w,
            "cap_tokens": self.cap_tokens,
            "max_w": self.max_w,
            "head_dim": self.head_dim,
            "gqa": self.gqa,
            "num_kv_heads": self.num_kv_heads,
        }
        # V2 knobs only when set: a V1 config serialises byte-identically.
        if self.deadline_ms != DEADLINE_MS_DEFAULT:
            raw["deadline_ms"] = self.deadline_ms
        if self.ab:
            raw["ab"] = self.ab
        return json.dumps(raw, sort_keys=True)

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
            deadline_ms=float(raw.get("deadline_ms", DEADLINE_MS_DEFAULT)),
            ab=str(raw.get("ab", "")),
        )
        if cfg.ab not in ("", "alt"):
            raise AHSpecError(f"{ENV}: ab={cfg.ab!r}, expected '' or 'alt'")
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

    def of_owner_all(self, stage: int) -> Tuple[Delegation, ...]:
        """All delegations of one owner, in head order (V2: one per helper)."""
        ds = [d for d in self.delegations if d.owner == stage]
        return tuple(sorted(ds, key=lambda d: self.heads_of(d).q0))

    def delegation(self, owner: int, helper: int) -> Delegation:
        for d in self.delegations:
            if d.owner == owner and d.helper == helper:
                return d
        raise KeyError((owner, helper))

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
    pairs = [(d.owner, d.helper) for d in delegations]
    if len(set(pairs)) != len(pairs):
        raise AHSpecError(f"--p-attn-head-split: an owner:helper pair appears twice ({pairs})")
    helpers = {d.helper for d in delegations}
    num_q = num_kv_heads * gqa
    by_owner: Dict[int, List[Delegation]] = {}
    for d in delegations:
        if not (0 <= d.owner < pp_size and 0 <= d.helper < pp_size):
            raise AHSpecError(f"--p-attn-head-split {d.text()}: stage outside PP{pp_size}")
        if d.helper == d.owner:
            raise AHSpecError(f"--p-attn-head-split {d.text()}: a stage cannot be its own helper")
        if d.owner in helpers:
            raise AHSpecError(
                f"--p-attn-head-split {d.text()}: stage {d.owner} would be owner "
                "AND helper; one barlink instance serves one stream per role"
            )
        if d.n_layers < 1:
            raise AHSpecError(f"--p-attn-head-split {d.text()}: n_layers must be >= 1")
        hr = d.heads(num_kv_heads, gqa)
        if not hr.whole_groups(gqa) and hr.n_kv != 1:
            raise AHSpecError(
                f"--p-attn-head-split {d.text()}: heads [{hr.q0},{hr.q1}) -- a delegated range is "
                "whole kv groups or lies inside ONE group (the helper runs one flashinfer call "
                "with one GQA ratio)"
            )
        by_owner.setdefault(d.owner, []).append(d)
    for h in helpers:
        dirs = {d.downstream for d in delegations if d.helper == h}
        if len(dirs) != 1:
            raise AHSpecError(
                f"--p-attn-head-split: helper stage {h} serves an upstream AND a downstream "
                "owner; one helper role per stage (V1 rule-driven or V2 server)"
            )
    for o, ds in by_owner.items():
        text = ",".join(d.text() for d in ds)
        if len({d.n_layers for d in ds}) != 1:
            raise AHSpecError(
                f"--p-attn-head-split {text}: owner {o} delegates the same layers to every "
                "helper (one n_layers per owner)"
            )
        if len({d.downstream for d in ds}) != 1:
            raise AHSpecError(
                f"--p-attn-head-split {text}: owner {o} mixes upstream and downstream helpers; "
                "all helpers of an owner are upstream (V1) or all downstream (V2)"
            )
        ranges = sorted((d.heads(num_kv_heads, gqa) for d in ds), key=lambda r: r.q0)
        for x, y in zip(ranges, ranges[1:]):
            if y.q0 < x.q1:
                raise AHSpecError(f"--p-attn-head-split {text}: head ranges of owner {o} overlap")
        if sum(r.n_q for r in ranges) >= num_q:
            raise AHSpecError(f"--p-attn-head-split {text}: the owner must keep at least one head")
        if owner_attn_layers is not None:
            have = int(owner_attn_layers.get(o, 0))
            if ds[0].n_layers > have:
                raise AHSpecError(
                    f"--p-attn-head-split {text}: stage {o} owns only "
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


def own_pieces(delegated: Sequence[HeadRange], num_q: int, gqa: int) -> Tuple[HeadRange, ...]:
    """The owner's own heads = the complement of its delegated ranges, cut
    into pieces one flashinfer call each can run: runs of whole groups, and
    partial groups (a group shared with a helper). Head order."""
    cover = sorted((r.q0, r.q1) for r in delegated)
    gaps, at = [], 0
    for q0, q1 in cover:
        if q0 > at:
            gaps.append((at, q0))
        at = max(at, q1)
    if at < num_q:
        gaps.append((at, num_q))
    out: List[HeadRange] = []
    for a, b in gaps:
        while a < b:
            if a % gqa:
                e = min(b, (a // gqa + 1) * gqa)
            else:
                e = (b // gqa) * gqa
                if e <= a:
                    e = b
            out.append(HeadRange(a, e, a // gqa, -(-e // gqa)))
            a = e
    return tuple(out)


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
            workspace + one in-flight payload and output per owner (a server
            helper's whole-chunk queue reuses them stream-ordered).
    owner : one int workspace per subset-wrapper shape (own pieces; V2 also
            the fallback shapes of its delegated ranges) + every helper's
            payload and output of one layer at max_w + the q copies of the own
            pieces (V2: of all heads, the fallback copies too) + V2 one owed
            late output per helper.
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
    own = cfg.of_owner_all(stage)
    if own:
        ranges = [cfg.heads_of(d) for d in own]
        server = own[0].downstream
        pieces = own_pieces(ranges, cfg.num_kv_heads * cfg.gqa, cfg.gqa)
        shapes = {(p.n_q, p.n_kv) for p in pieces}
        q_heads = sum(p.n_q for p in pieces)
        nbytes = 0
        for hr in ranges:
            pay, out = message_bytes(hr.n_q, hr.n_kv, cfg.max_w, D)
            nbytes += pay + out
            if server:
                shapes.add((hr.n_q, hr.n_kv))
                q_heads += hr.n_q
                nbytes += out
        mib += INT_WS_MIB * len(shapes) + (nbytes + cfg.max_w * q_heads * D * 2) / 2**20
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

    def __init__(self, min_w: int, cap_tokens: int, ab: str = ""):
        self.min_w = int(min_w)
        self.cap = int(cap_tokens)
        self.ab = ab
        self.active: Optional[str] = None
        self.next_pos = 0
        self.chunk_no = 0
        #: new eligible requests seen (A/B counter; survives reset())
        self.n_req = 0
        #: (rid, arm, n_req) of a request that started in the last decide()
        self.last_start: Optional[Tuple[str, str, int]] = None

    def reset(self) -> None:
        self.active = None
        self.next_pos = 0
        self.chunk_no = 0

    def end_request(self) -> None:
        """V2 owner fallback: the active request is not split any further
        (its helpers' mirrors stop being read; they are never told)."""
        self.reset()

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
        self.last_start = None
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
            self.n_req += 1
            if self.ab == "alt" and self.n_req % 2 == 0:
                self.last_start = (rid, "base", self.n_req)
                self.reset()
                return None
            self.last_start = (rid, "split", self.n_req)
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
    One query head per iteration (head h reads kv group ``h // (Hq // Hk)``),
    so a head's result never depends on how many other heads share the call
    -- whole groups, half groups, the owner's fallback (CPU bit-equality of
    V1 and V2 splits)."""
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
    for h in range(hq):
        g = h // gs
        # contiguous operands: the result must not depend on the caller's
        # layout (pool view, mirror, full tensor)
        qh = qf[:, h, :].contiguous()  # [w, D]
        kg, vg = kf[:, g, :].contiguous(), vf[:, g, :].contiguous()
        s = torch.matmul(qh, kg.t()) * sm_scale  # [w, L]
        s = s.masked_fill(~mask, float("-inf"))
        a = torch.softmax(s, dim=-1)
        out[:, h, :] = torch.matmul(a, vg)
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


class _DoneEvent:
    """An event that was complete when it was born (inline execution)."""

    def query(self) -> bool:
        return True

    def synchronize(self) -> None:
        return None


class InlineExec:
    """No side stream (CPU, V1 tests): everything runs now, in order."""

    failure: Optional[str] = None

    def submit(self, fn: Callable[[], None]) -> None:
        fn()

    def record(self):
        return _DoneEvent()

    def main_record(self):
        return _DoneEvent()

    def side_waits(self, ev) -> None:
        return None

    def main_waits(self, ev) -> None:
        return None

    def keep(self, t: torch.Tensor) -> None:
        return None

    def synchronize(self) -> None:
        return None


class CudaExec:
    """One CUDA side stream. ``submit`` enqueues (never blocks the host on
    the device); ``record``/``main_record`` return torch events whose
    ``query()`` is a host poll, never a sync."""

    failure: Optional[str] = None

    def __init__(self, stream):
        self.stream = stream

    def submit(self, fn: Callable[[], None]) -> None:
        with torch.cuda.stream(self.stream):
            fn()

    def record(self):
        ev = torch.cuda.Event()
        ev.record(self.stream)
        return ev

    def main_record(self):
        ev = torch.cuda.Event()
        ev.record()
        return ev

    def side_waits(self, ev) -> None:
        self.stream.wait_event(ev)

    def main_waits(self, ev) -> None:
        torch.cuda.current_stream().wait_event(ev)

    def keep(self, t: torch.Tensor) -> None:
        t.record_stream(self.stream)

    def synchronize(self) -> None:
        self.stream.synchronize()


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
            d = self.cfg.delegation(owner, self.wire.rank)
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


class ServerHelper:
    """V2: a DOWNSTREAM helper serves the owner's announcements.

    It runs no split rule. Its host thread polls the pinned publish counter of
    every owner it serves; a new message at ``rx_seq`` must be the FIRST
    delegated layer of a chunk (the announcement). The header is checked on
    the host (magic, layer, head range, and the request continuity the helper
    keeps itself), then the WHOLE chunk goes onto the side stream at once:
    for every delegated layer recv -> device header check -> mirror write +
    attention -> send. The host thread wakes once per chunk, not per layer
    (GIL); the device waits for the later payloads (barlink recv), and that
    wait sits on the side stream only -- the helper stage's own forward on
    its main stream never waits for it and it never waits for that forward,
    which is why a downstream helper cannot deadlock the pipeline."""

    serves_announced = True

    def __init__(self, cfg: AHConfig, wire, engine: HelperEngine, infos: Dict[Tuple[int, int], LayerInfo],
                 layers_of: Dict[int, Tuple[int, ...]], exec_=None, plan: Optional[Callable] = None, device=None):
        self.cfg = cfg
        self.wire = wire
        self.engine = engine
        self.infos = infos
        self.exec = exec_ if exec_ is not None else InlineExec()
        self.plan = plan
        self.device = device
        self.dels = {d.owner: d for d in cfg.of_helper(wire.rank)}
        if not all(d.downstream for d in self.dels.values()):
            raise AHSpecError(f"ServerHelper PP{wire.rank}: serves downstream (V2) delegations only")
        self.layers_of = {o: tuple(layers_of[o]) for o in self.dels}
        self.rx_seq: Dict[int, int] = {o: 1 for o in self.dels}
        self.last_ev: Dict[int, object] = {o: None for o in self.dels}
        #: per owner: (rid_hash, next_pos, chunk_no) of the request being mirrored
        self.req: Dict[int, Optional[Tuple[int, int, int]]] = {o: None for o in self.dels}
        self.failure: Optional[str] = None
        self.stats = {"chunks": 0, "layers": 0, "bytes_in": 0, "bytes_out": 0}
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None

    # the V1 HelperThread surface the runtime uses
    def push(self, row, layers_of) -> None:  # noqa: D401 -- V2 never pushes
        return None

    def pending(self) -> int:
        return 0

    def drop_all(self) -> int:
        """KV release: the mirror is about to be unmapped. Every announced
        chunk was completed by the owner before the flip; forget the request
        (the next announcement must start at p == 0)."""
        for o in self.req:
            self.req[o] = None
        return 0

    def start(self) -> None:
        self._thread = threading.Thread(target=self._run, name="ah-server", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()

    def _accept(self, owner: int, got: Sequence[int]) -> ChunkRow:
        d = self.dels[owner]
        hr = self.cfg.heads_of(d)
        magic, layer, rh, p, w, no, qs, gs = (int(x) for x in got[:HDR_I64])
        bad = []
        if magic != MAGIC:
            bad.append(f"magic {magic:#x}")
        if layer != self.layers_of[owner][0]:
            bad.append(f"layer {layer}, a chunk announcement is layer {self.layers_of[owner][0]}")
        if (qs, gs) != hr.packed():
            bad.append(f"head range {qs:#x}/{gs:#x}, mine {hr}")
        if not (1 <= w <= self.cfg.max_w) or p < 0 or p + w > self.cfg.cap_tokens:
            bad.append(f"p={p} w={w} outside max_w {self.cfg.max_w} / mirror cap {self.cfg.cap_tokens}")
        cur = self.req[owner]
        if p == 0:
            if no != 0:
                bad.append(f"p=0 with chunk_no {no}")
        elif cur is None:
            bad.append(f"p={p} continues a request this helper never started (no p=0 chunk)")
        elif (rh, p, no) != (cur[0], cur[1], cur[2] + 1):
            bad.append(f"rid_hash/p/chunk_no {rh}/{p}/{no}, expected {cur[0]}/{cur[1]}/{cur[2] + 1}")
        if bad:
            raise RuntimeError(
                f"AH RANKS DISAGREE (raenge-nie-uneins): owner PP{owner} announced a chunk the "
                f"server helper PP{self.wire.rank} cannot serve: " + "; ".join(bad)
            )
        return ChunkRow(f"#{rh}", rh, p, w, no)

    def poll_once(self) -> int:
        """One pass over the owners; returns how many CHUNKS it enqueued."""
        if self.exec.failure:
            raise RuntimeError(f"AH server helper side stream failed: {self.exec.failure}")
        issued = 0
        for owner in self.dels:
            seq = self.rx_seq[owner]
            if self.wire.published(owner) < seq:
                continue  # idle is legitimate: the owner announces when it splits
            row = self._accept(owner, self.wire.peek_header(owner, seq))
            self._issue_chunk(owner, row)
            issued += 1
        return issued

    def _issue_chunk(self, owner: int, row: ChunkRow) -> None:
        d = self.dels[owner]
        hr = self.cfg.heads_of(d)
        layers = self.layers_of[owner]
        pay_b, out_b = message_bytes(hr.n_q, hr.n_kv, row.w, self.cfg.head_dim)
        if self.plan is not None:
            ev = self.last_ev[owner]
            if ev is not None:
                ev.synchronize()  # the wrapper's previous plan has been consumed
            self.exec.submit(lambda: self.plan(owner, row.p, row.w, hr.n_q, hr.n_kv))
        wants = [header_values(row, lid, hr) for lid in layers]
        refs: Dict[str, torch.Tensor] = {}
        if self.device is not None and _is_cuda(self.device):
            # the later layers' expected headers, pinned -> device without a
            # host sync (a pageable .to() would block this thread on the
            # side stream, i.e. on the owner's later payloads)
            host = torch.tensor(wants, dtype=torch.int64).pin_memory()
            self.exec.submit(lambda: refs.__setitem__("dev", host.to(self.device, non_blocking=True)))
        for i, lid in enumerate(layers):
            self.exec.submit(self._layer_fn(owner, hr, row, lid, wants[i], pay_b, check=i > 0, refs=refs, i=i))
            self.rx_seq[owner] += self.wire.pieces(pay_b)
        self.last_ev[owner] = self.exec.record()
        self.req[owner] = (row.rid_hash, row.p + row.w, row.chunk_no)
        self.stats["chunks"] += 1
        self.stats["layers"] += len(layers)
        self.stats["bytes_in"] += pay_b * len(layers)
        self.stats["bytes_out"] += out_b * len(layers)

    def _layer_fn(self, owner, hr, row, lid, want, pay_b, check, refs, i):
        def run():
            buf = torch.empty(pay_b // 2, dtype=torch.bfloat16, device=self.device)
            self.wire.recv(buf, owner)
            if check:
                got = buf[:HDR_BF16].view(torch.int64)
                if "dev" in refs:
                    torch._assert_async(torch.all(got == refs["dev"][i]))
                else:
                    bad = header_mismatch(got.tolist(), want)
                    if bad:
                        raise RuntimeError(
                            f"AH RANKS DISAGREE (raenge-nie-uneins): owner PP{owner} layer {lid} "
                            f"payload inside an announced chunk: {bad}"
                        )
            out = self.engine.step(owner, self.infos[(owner, lid)], buf, row, hr)
            self.wire.send(out, owner)
        return run

    def _run(self) -> None:
        if self.device is not None and _is_cuda(self.device):
            torch.cuda.set_device(self.device)
        try:
            while not self._stop.is_set():
                if self.poll_once() == 0:
                    time.sleep(HELPER_POLL_S)
        except BaseException as exc:  # noqa: BLE001 -- named, surfaced on the forward thread
            self.failure = f"{type(exc).__name__}: {exc}"
            logger.error("AH server helper thread died: %s", self.failure)


# ---------------------------------------------------------------------------
# owner side
# ---------------------------------------------------------------------------


class OwnerSide:
    """The delegated layers on the owner, for all of its helpers.

    ``own_attention(q_sub, k, v, g0, g1, store)`` returns the output of the
    query heads ``q_sub`` over kv groups [g0, g1) of the pool; ``store=True``
    (first call of a layer only) first writes ALL kv groups of the chunk into
    the pool, as the stock call does
    (``FlashInferAttnBackend.forward_extend_head_subset`` on the card).

    V1 (upstream helpers): payloads out, own pieces, outputs in, all barlink
    ops on one side stream, the main stream waits for the outputs' event.
    V2 (downstream helpers, ``server``): the same, except that the host looks
    at the helper's publish counter before it enqueues a recv, with the
    deadline and the self-computing fallback described in the module
    docstring."""

    def __init__(self, cfg: AHConfig, dels, wire, device, stream=None, *, exec_=None,
                 layers: Sequence[int] = (), on_veto: Optional[Callable[[], None]] = None,
                 clock: Callable[[], float] = time.monotonic):
        if isinstance(dels, Delegation):
            dels = (dels,)
        self.cfg = cfg
        self.dels = tuple(sorted(dels, key=lambda d: cfg.heads_of(d).q0))
        self.d = self.dels[0]
        self.wire = wire
        self.device = device
        self.stream = stream
        if exec_ is None:
            exec_ = CudaExec(stream) if stream is not None else InlineExec()
        self.exec = exec_
        self.hr = cfg.heads_of(self.d)
        self.ranges = tuple(cfg.heads_of(d) for d in self.dels)
        self.pieces = own_pieces(self.ranges, cfg.num_kv_heads * cfg.gqa, cfg.gqa)
        self.q_own, self.kv_own = own_counts(self.hr)
        self.server = self.d.downstream
        self.deadline_s = float(cfg.deadline_ms) / 1000.0
        self.layers = tuple(sorted(int(x) for x in layers))
        self.on_veto = on_veto
        self.clock = clock
        #: V2: pieces each helper has published in total once this layer's
        #: output is out (the owner's expectation of the helper's counter)
        self.need: Dict[int, int] = {d.helper: 0 for d in self.dels}
        #: V2: outputs the owner stopped waiting for, owed in FIFO order
        self.late: Dict[int, List[Tuple[int, List[int]]]] = {d.helper: [] for d in self.dels}
        self.degraded = False
        self._keep: List[Tuple[object, torch.Tensor]] = []
        self._ring = None
        self._ring_ev: List[object] = []
        self._ring_i = 0
        if _is_cuda(device):
            self._ring = torch.empty(HDR_RING, HDR_I64, dtype=torch.int64).pin_memory()
            self._ring_ev = [None] * HDR_RING
        self.stats = {"layers": 0, "fallback": 0, "late_drained": 0, "wait_s": 0.0, "vetoes": 0}

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

    def _echo_check(self, obuf: torch.Tensor, hdr_vals: Sequence[int], hdr_dev: Optional[torch.Tensor]) -> None:
        got = obuf[:HDR_BF16].view(torch.int64)
        if hdr_dev is not None and _is_cuda(obuf.device):
            torch._assert_async(torch.all(got == hdr_dev))
            return
        bad = header_mismatch(got.tolist(), hdr_vals)
        if bad:
            raise RuntimeError(f"AH RANKS DISAGREE: the helper's output header does not echo the payload's: {bad}")

    def _recv_fn(self, helper: int, obuf: torch.Tensor, hdr_vals, hdr_dev):
        def run():
            self.wire.recv(obuf, helper)
            self._echo_check(obuf, hdr_vals, hdr_dev)
        return run

    def _drain(self, helper: int, only_published: bool = False) -> None:
        """Enqueue the recvs of owed late outputs on the side stream (FIFO,
        before any newer output of that helper); ``only_published`` = only
        those the helper has already published (no device wait at all)."""
        owed = self.late[helper]
        n = 0
        pub = self.wire.published(helper) if only_published else None
        for out_b, hv in owed:
            if pub is not None and pub < upto_seq(self, helper, n):
                break
            scratch = torch.empty(out_b // 2, dtype=torch.bfloat16, device=self.device)
            self.exec.submit(self._recv_fn(helper, scratch, hv, None))
            self._keep.append((self.exec.record(), scratch))
            n += 1
        if n:
            del owed[:n]
            self.stats["late_drained"] += n

    def _gc(self) -> None:
        self._keep = [(e, t) for e, t in self._keep if not e.query()]

    def _await_published(self, helper: int, own_done, sent, t_own: List[Optional[float]]) -> bool:
        """V2: True = the output is published (recv it), False = fall back.
        The deadline runs from the moment BOTH the own heads are done and the
        payloads have left (the side stream may still be draining owed
        outputs of an earlier chunk -- that backlog is the owner's, not the
        helper's lateness)."""
        need = self.need[helper]
        t0 = self.clock()
        try:
            while True:
                if self.wire.published(helper) >= need:
                    return True
                if self.degraded:
                    return False
                if self.deadline_s > 0:
                    if t_own[0] is None:
                        if own_done.query() and sent.query():
                            t_own[0] = self.clock()
                    elif self.clock() - t_own[0] > self.deadline_s:
                        return False
                elif self.clock() - t0 > HELPER_STALL_S:
                    raise RuntimeError(
                        f"AH owner stall: waited {self.clock() - t0:.0f} s for helper PP{helper}'s "
                        "output with no deadline -- the helper died or never served the chunk"
                    )
                if self.exec.failure:
                    raise RuntimeError(f"AH owner side stream failed: {self.exec.failure}")
                time.sleep(HELPER_POLL_S)
        finally:
            self.stats["wait_s"] += self.clock() - t0

    def attention(self, row: ChunkRow, layer_id: int, q, k, v, own_attention: Callable) -> torch.Tensor:
        cfg = self.cfg
        D, H, Hk = cfg.head_dim, cfg.num_kv_heads * cfg.gqa, cfg.num_kv_heads
        w = q.shape[0]
        qv = q.view(w, H, D)
        kv3 = k.view(w, Hk, D)
        vv3 = v.view(w, Hk, D)
        if self.layers and layer_id == self.layers[0]:
            self.degraded = False
            self._gc()
        # Pack BEFORE the pool write: set_kv_buffer may divide k/v in place.
        sends = []
        for d, hr in zip(self.dels, self.ranges):
            hv = header_values(row, layer_id, hr)
            hdr = self._hdr_dev(hv)
            sends.append((d, hr, hv, hdr, pack_payload(hdr, qv[:, hr.q0:hr.q1], kv3[:, hr.g0:hr.g1], vv3[:, hr.g0:hr.g1])))
        self.exec.side_waits(self.exec.main_record())
        for d, _, _, _, payload in sends:
            self.exec.submit(lambda payload=payload, dst=d.helper: self.wire.send(payload, dst))
            self.exec.keep(payload)
        sent = self.exec.record()
        # own pieces (the first call stores all kv groups into the pool)
        outs: List[Tuple[int, torch.Tensor]] = []
        for i, pc in enumerate(self.pieces):
            o = own_attention(qv[:, pc.q0:pc.q1].contiguous(), kv3, vv3, pc.g0, pc.g1, i == 0)
            outs.append((pc.q0, o.reshape(w, pc.n_q * D)))
        own_done = self.exec.main_record()
        if not self.server:
            # V1: the outputs' recv (a device wait) starts after the own heads,
            # as V1's main-stream recv did -- no spin beside them
            self.exec.side_waits(own_done)
        t_own: List[Optional[float]] = [None]
        got_ev = None
        for d, hr, hv, hdr, _ in sends:
            _, out_b = message_bytes(hr.n_q, hr.n_kv, w, D)
            h = d.helper
            self.need[h] += self.wire.pieces(out_b)
            if self.server and not self._await_published(h, own_done, sent, t_own):
                # fallback: the owner's pool is complete -- compute the range here
                o = own_attention(qv[:, hr.q0:hr.q1].contiguous(), kv3, vv3, hr.g0, hr.g1, False)
                outs.append((hr.q0, o.reshape(w, hr.n_q * D)))
                self.late[h].append((out_b, hv))
                self._drain(h, only_published=True)  # free barlink's two slots early
                self.stats["fallback"] += 1
                if not self.degraded:
                    self.degraded = True
                    self.stats["vetoes"] += 1
                    if self.on_veto is not None:
                        self.on_veto()
                continue
            if self.server:
                self._drain(h)  # all owed outputs are published: they precede this one
            obuf = torch.empty(out_b // 2, dtype=q.dtype, device=q.device)
            self.exec.submit(self._recv_fn(h, obuf, hv, hdr))
            self.exec.keep(hdr)  # the echo check reads it on the side stream
            got_ev = self.exec.record()
            outs.append((hr.q0, obuf[HDR_BF16:].view(w, hr.n_q * D)))
        if got_ev is not None:
            self.exec.main_waits(got_ev)
        if self.server and self.layers and layer_id == self.layers[-1]:
            for h in self.late:
                self._drain(h)  # chunk end: owed outputs leave the two barlink slots
        self.stats["layers"] += 1
        outs.sort(key=lambda t: t[0])
        return torch.cat([o for _, o in outs], dim=1)

    def on_kv_release(self) -> None:
        for h in self.late:
            self._drain(h)
        self.exec.synchronize()
        self._keep.clear()


def upto_seq(side: "OwnerSide", helper: int, n: int) -> int:
    """The publish count at which owed output ``n`` (0-based, FIFO) of
    ``helper`` is out: the current expectation minus the pieces of every
    later owed output."""
    owed = side.late[helper]
    later = sum(side.wire.pieces(ob) for ob, _ in owed[n + 1:])
    return side.need[helper] - later


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
        self.rule = SplitRule(cfg.min_w, cfg.cap_tokens, cfg.ab)
        self.row: Optional[ChunkRow] = None
        self.owner: Optional[OwnerSide] = None
        self.owner_layer_ids: frozenset = frozenset()
        self.helper = None  # HelperThread (V1) or ServerHelper (V2)
        self.layers_of: Dict[int, Tuple[int, ...]] = {}
        self.n_split = 0
        self.n_forwards = 0
        helped = cfg.of_helper(pp_rank)
        self.role = (
            "owner" if cfg.of_owner(pp_rank) is not None
            else ("server" if helped[0].downstream else "helper") if helped else "none"
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
        st = self.rule.last_start
        if st is not None and self.role == "owner" and self.cfg.ab:
            logger.info("AH-AB PP%d rid=%s arm=%s req=%d", self.pp_rank, st[0], st[1], st[2])
        if self.row is None:
            return
        self.n_split += 1
        if self.helper is not None and not getattr(self.helper, "serves_announced", False):
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

        def own(q_sub, k_all, v_all, g0, g1, store):
            return fn(q_sub, k_all, v_all, radix_attn, forward_batch, (g0, g1), store)

        return self.owner.attention(self.row, radix_attn.layer_id, q, k, v, own)

    def on_kv_release(self) -> None:
        """The KV region (and the mirror in it) is about to be unmapped."""
        self.rule.reset()
        self.row = None
        if self.owner is not None:
            self.owner.on_kv_release()
            if self.owner.stats["fallback"]:
                logger.warning("AH owner PP%d: %s", self.pp_rank, self.owner.stats)
        if self.helper is not None and getattr(self.helper, "serves_announced", False):
            self.helper.drop_all()
            self.helper.exec.synchronize()
        elif self.helper is not None:
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
    dels_own = cfg.of_owner_all(pp_rank)
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
    # OWN RING: a separate pinned segment with its own per-pair sequences --
    # never the PP proxy transport, so a head payload and a proxy frame can
    # never share one FIFO (P_MICROBATCH_AH2.md 4.1-3).
    transport = BarlinkHostTransport(pp.cpu_group, device, slot_bytes=4096, p2p_bytes=cfg.p2p_bytes())
    wire = BarlinkP2P(transport)
    rt = AHRuntime(cfg, pp_rank, pp_size)
    for g in gathered:
        if cfg.of_owner(int(g["rank"])) is not None:
            rt.layers_of[int(g["rank"])] = tuple(sorted(int(x) for x in g["layers"]))
    if d_own is not None:
        side = torch.cuda.Stream(device)
        rt.owner = OwnerSide(cfg, dels_own, wire, device, stream=side, exec_=CudaExec(side),
                             layers=rt.layers_of[pp_rank], on_veto=rt.rule.end_request)
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
        if helped[0].downstream:
            import sys

            # one wake-up per chunk announcement; keep the GIL hand-over short
            sys.setswitchinterval(SERVER_SWITCH_INTERVAL_S)
            rt.helper = ServerHelper(cfg, wire, engine, infos, rt.layers_of, exec_=CudaExec(stream),
                                     plan=kernel.plan, device=device)
        else:
            rt.helper = HelperThread(cfg, wire, engine, infos, stream=stream, plan=kernel.plan, device=device)
        rt.helper.start()
    runner._ah_runtime = rt
    _RUNTIME = rt
    logger.warning(
        "AH-SPLIT armed PP%d/%d role=%s spec=%s min_w=%d cap=%d post=%.1f MiB "
        "(ATTN_HEAD_SPLIT.md; %s) barlink p2p %.2f MiB/pair, pinned host %.1f MiB",
        pp_rank, pp_size, rt.role, ",".join(d.text() for d in cfg.delegations), cfg.min_w,
        cfg.cap_tokens, stage_post_mib(cfg, pp_rank),
        ("owner layers %s -> %s, own pieces %s, deadline %.1f ms, ab=%s" % (
            sorted(rt.owner_layer_ids),
            ", ".join("PP%d q heads %s" % (d.helper, cfg.heads_of(d)) for d in dels_own),
            rt.owner.pieces, cfg.deadline_ms if dels_own[0].downstream else 0.0, cfg.ab or "off")) if d_own else
        ("%s for %s, mirror %.1f MiB" % (
            rt.role, {d.owner: rt.layers_of[d.owner] for d in helped},
            rt.helper.engine.mirror_mib())) if helped else "idle",
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
