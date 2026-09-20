# SPDX-License-Identifier: Apache-2.0
"""The Next-Flash P/D flip -- its three hard seams as one pure, testable solve.

THE FLIP, in one sentence: Qwen3.8 Next Flash prefills in the PP3 layout
(``--tp-size 1 --pp-size 3``, layer split 29/11/8, 2729 tok/s on the binding
stage) and decodes in Form A (the 5090 as attention host with all dense
weights, the whole KV and a SOLO draft; the two 3080s as pure expert
workers, 92.9/58.6 tok/s), and between the two there is a FLIP -- a
drain, a hand-over, a re-admission.

Why this module exists: the flip is not one problem but THREE, and each of
them is arithmetic before it is code. A boot that gets any of the three
wrong does not fail loudly -- it OOMs on the third request, or it reaps the
host, or it decodes from a recurrent state that belongs to nobody. So each
seam is solved here, at the desk, with the measured numbers injected and a
NAMED refusal when the arithmetic says no:

  (a) KV RE-LAY -- W113. PP3 splits the KV across three cards by
      full-attention layer count [7, 3, 2], cells 7616/3264/2176 B per
      token. Form A holds ALL of it on one card at 14143 B per token. The
      question is not "can it be copied" but "does the destination hold
      262144 tokens at all".

  (b) SHARED HOST EXPERT MEMORY -- W114. Both layouts spill their MoE
      experts into a PAGE-LOCKED host pool. Measured: PP3 pins 31.88 GiB,
      Form A pins 38.86 GiB. Six processes each holding their own copy is
      70.74 GiB of pinned host memory before a single anonymous byte of
      runtime -- against a host mark of 88 GiB. Sharing is therefore not an
      optimisation of this design, it is its PRECONDITION, and this ledger
      is what says so with a number instead of a hope.

  (c) STATE CARRY -- W115. GDN/Mamba recurrent state and MTP draft state
      are not KV: they are per-REQUEST, they are not addressable by token,
      and the Form-A draft is SOLO -- it exists on the host only, so it has
      no PP-side partner whose state could be forwarded. Anything the flip
      cannot carry must be REBUILT, and rebuilding has a price in tokens
      that belongs in the flip-time budget, not in a footnote.

What this module deliberately does NOT do, for the same reasons as
``form_a_plan``: it reads no NVML, no torch and no device; every figure is
injected and every default carries the boot it was measured on by name; and
it never rounds a refusal into a smaller plan.

Provenance of the defaults, both boots under ``/spinning/evidence-665-f1``:

  PP3     ``boot_fn_fn7s_20260919T183945Z.server.log``
          line 21     ``--pp-layer-ratio 29,11,8``, full-attention [7,3,2]
          lines 263-265 ``KV pool sizing``  cells 7616 / 3264 / 2176
          lines 242-244 ``[offload-kv-regain]`` 22.73 / 5.30 / 3.85 GiB pinned

  Form A  ``boot_fn_fnFA19_20260920T152302Z.server.log``
          lines 1000-1002 ``KV pool sizing``  cells 14143 / 768 / 768
          lines 989-991   ``[offload-kv-regain]`` 20.62 / 7.36 / 10.88 GiB pinned
          line 1027       ``Hybrid mamba/attention KV cap ... -> 262151``
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple

__all__ = [
    "FlipInfeasible",
    "Weg2FlipKvRelayInfeasible",
    "Weg2FlipHostPoolDoubled",
    "Weg2FlipDraftStateOrphaned",
    "KvLayout",
    "KvRelayPlan",
    "solve_kv_relay",
    "HostPoolPost",
    "HostPoolLedger",
    "solve_host_pool",
    "CarriedState",
    "StateCarryPlan",
    "solve_state_carry",
    "PP3_KV",
    "FORM_A_KV",
    "PP3_PINNED_GIB",
    "FORM_A_PINNED_GIB",
    "HOST_MARK_GIB",
]

_GIB = 1024.0**3

# Host reap mark, Memory ``host-schwelle-nie-uebertreten`` / the operator's
# 88 GiB brief for this design. SOFT: exceeding it is a refusal that must be
# deviated from with a number and a runtime latch, never silently.
HOST_MARK_GIB = 88.0


# --------------------------------------------------------------------------
# Refusals -- one class per seam, named with its W-code so a log line and an
# exception type carry the SAME name. W113-W115 are the first free codes
# (W0..W112 are taken in python/sglang as of 1e42021848).
# --------------------------------------------------------------------------
class FlipInfeasible(ValueError):
    """Base: this P/D flip cannot be booted as described."""


class Weg2FlipKvRelayInfeasible(FlipInfeasible):
    """W113 -- the PP3 KV cannot be re-laid into the Form-A host KV.

    Raised when the destination (single-host) KV pool cannot hold the
    context the source layout was allowed to prefill. The source is allowed
    a LARGER context than the destination precisely because it splits the
    per-token cell across three cards; the flip is where that asymmetry
    becomes an OOM, so it is checked here instead.
    """


class Weg2FlipHostPoolDoubled(FlipInfeasible):
    """W114 -- both layouts hold their own page-locked expert pool.

    Raised when the sum of the pinned host pools of all live processes
    exceeds the host mark. The fix is never a smaller mark: it is one
    pool with one holder and two readers.
    """


class Weg2FlipDraftStateOrphaned(FlipInfeasible):
    """W115 -- a D-side state has no P-side partner and no rebuild rule.

    Raised when a state the decode layout needs per request is neither
    CARRIED across the flip nor declared REBUILDABLE with a token cost.
    The Form-A draft is the motivating case: it is solo on the host, the
    PP3 side shards its draft across the stages, so there is no matching
    tensor to forward -- which is fine, but only if it is DECLARED.
    """


# ==========================================================================
# Seam (a): KV re-lay
# ==========================================================================
@dataclass(frozen=True)
class KvLayout:
    """One layout's KV geometry, as its boot log reports it.

    `cells` is the per-token cell size in BYTES on each rank, taken from the
    ``KV pool sizing: ... cell_size=N`` line -- not derived. Under PP the
    cells DIFFER per rank because a hybrid's KV mass follows the
    full-attention layer count, not the layer count.

    `avail_bytes` is the ``available_bytes=`` on the same line: what the
    profiler actually left for KV on that rank, after weights and after the
    expert offload gave its VRAM back.
    """

    name: str
    boot: str
    cells: Tuple[int, ...]
    avail_bytes: Tuple[int, ...]
    full_attn_per_rank: Tuple[int, ...]

    def __post_init__(self) -> None:
        n = len(self.cells)
        if n == 0:
            raise Weg2FlipKvRelayInfeasible(
                f"W113 Weg2FlipKvRelayInfeasible -- layout {self.name!r} has no ranks"
            )
        if len(self.avail_bytes) != n or len(self.full_attn_per_rank) != n:
            raise Weg2FlipKvRelayInfeasible(
                f"W113 Weg2FlipKvRelayInfeasible -- layout {self.name!r} is ragged: "
                f"{n} cells, {len(self.avail_bytes)} avail, "
                f"{len(self.full_attn_per_rank)} full-attention counts"
            )
        if any(c <= 0 for c in self.cells):
            raise Weg2FlipKvRelayInfeasible(
                f"W113 Weg2FlipKvRelayInfeasible -- layout {self.name!r} has a "
                f"non-positive cell: {self.cells}"
            )

    @property
    def capacity_tokens(self) -> Tuple[int, ...]:
        """Tokens each rank could hold, ignoring the min-reduce and the cap."""
        return tuple(a // c for a, c in zip(self.avail_bytes, self.cells))

    @property
    def binding_rank(self) -> int:
        """The rank whose capacity binds the whole layout (the min)."""
        caps = self.capacity_tokens
        return min(range(len(caps)), key=lambda i: caps[i])

    @property
    def binding_tokens(self) -> int:
        return min(self.capacity_tokens)

    @property
    def carrying_ranks(self) -> Tuple[int, ...]:
        """Ranks that actually hold full-attention KV (count > 0)."""
        return tuple(i for i, f in enumerate(self.full_attn_per_rank) if f > 0)

    def bytes_at(self, tokens: int) -> Tuple[int, ...]:
        return tuple(c * tokens for c in self.cells)


@dataclass(frozen=True)
class KvRelayPlan:
    """The answer to "does the destination hold what the source prefilled"."""

    src: KvLayout
    dst: KvLayout
    context_tokens: int
    dst_binding_rank: int
    dst_binding_tokens: int
    dst_bytes: Tuple[int, ...]
    src_bytes: Tuple[int, ...]
    headroom_tokens: int
    headroom_ratio: float
    moved_bytes: int
    carrier_bytes_per_token: int

    def report(self) -> str:
        lines = [
            f"KV RE-LAY  {self.src.name} -> {self.dst.name}  @ {self.context_tokens} tokens",
            f"  source  cells {self.src.cells}  full-attn {self.src.full_attn_per_rank}"
            f"  -> {tuple(round(b / _GIB, 3) for b in self.src_bytes)} GiB per rank",
            f"  dest    cells {self.dst.cells}  full-attn {self.dst.full_attn_per_rank}"
            f"  -> {tuple(round(b / _GIB, 3) for b in self.dst_bytes)} GiB per rank",
            f"  dest binds on rank {self.dst_binding_rank} at "
            f"{self.dst_binding_tokens} tokens "
            f"({self.headroom_ratio:.2f}x the demanded {self.context_tokens})",
            f"  carrier must move {self.moved_bytes / _GIB:.3f} GiB "
            f"({self.carrier_bytes_per_token} B/token summed over carrying source ranks)",
        ]
        return "\n".join(lines)


def solve_kv_relay(
    src: KvLayout,
    dst: KvLayout,
    context_tokens: int,
) -> KvRelayPlan:
    """Can `dst` hold the context `src` was allowed to prefill?

    The check is on the DESTINATION's binding rank, because that is the one
    that OOMs. It is deliberately NOT a comparison of the summed cells:
    14143 is not 7616+3264+2176 and never will be -- the destination cell
    carries terms (the solo draft's KV, the undivided mamba amortisation)
    the source splits differently or does not have at all.
    """
    if context_tokens <= 0:
        raise Weg2FlipKvRelayInfeasible(
            f"W113 Weg2FlipKvRelayInfeasible -- context_tokens={context_tokens} "
            f"is not a context"
        )

    dst_binding = dst.binding_tokens
    dst_rank = dst.binding_rank
    if dst_binding < context_tokens:
        raise Weg2FlipKvRelayInfeasible(
            f"W113 Weg2FlipKvRelayInfeasible -- destination {dst.name!r} "
            f"(boot {dst.boot}) binds at {dst_binding} tokens on rank {dst_rank} "
            f"(cell {dst.cells[dst_rank]} B, available "
            f"{dst.avail_bytes[dst_rank] / _GIB:.3f} GiB) but the flip demands "
            f"{context_tokens}. Source {src.name!r} binds at "
            f"{src.binding_tokens}. The source is allowed the larger context "
            f"because it SPLITS the cell across "
            f"{len(src.carrying_ranks)} carrying rank(s); the destination does "
            f"not. Raise the destination's KV budget by at least "
            f"{(context_tokens - dst_binding) * dst.cells[dst_rank] / _GIB:.3f} GiB "
            f"or lower the context."
        )

    carrier_bpt = sum(src.cells[i] for i in src.carrying_ranks)
    moved = carrier_bpt * context_tokens
    return KvRelayPlan(
        src=src,
        dst=dst,
        context_tokens=context_tokens,
        dst_binding_rank=dst_rank,
        dst_binding_tokens=dst_binding,
        dst_bytes=dst.bytes_at(context_tokens),
        src_bytes=src.bytes_at(context_tokens),
        headroom_tokens=dst_binding - context_tokens,
        headroom_ratio=dst_binding / context_tokens,
        moved_bytes=moved,
        carrier_bytes_per_token=carrier_bpt,
    )


# ==========================================================================
# Seam (b): the shared page-locked host expert pool
# ==========================================================================
@dataclass(frozen=True)
class HostPoolPost:
    """One process's page-locked host expert pool, as its boot log reports.

    `pinned_gib` is the ``[offload-kv-regain] ... N GiB moved to the pinned
    host pool, N GiB page-locked exactly`` figure. "page-locked EXACTLY" is
    load-bearing: Memory ``pinned-exakt-drei-fallen`` -- ``pin_memory``
    rounds to a power of two, so a pool that is not allocated via
    mmap + cudaHostRegister silently costs MORE than this number.
    """

    layout: str
    rank: int
    pinned_gib: float
    anon_gib: float

    @property
    def total_gib(self) -> float:
        return self.pinned_gib + self.anon_gib


@dataclass(frozen=True)
class HostPoolLedger:
    posts: Tuple[HostPoolPost, ...]
    shared: bool
    mark_gib: float
    pinned_gib: float
    anon_gib: float
    total_gib: float
    saved_gib: float

    def report(self) -> str:
        mode = "SHARED (one holder, two readers)" if self.shared else "PER-PROCESS"
        lines = [
            f"HOST EXPERT POOL  {mode}   mark {self.mark_gib:.1f} GiB",
            f"  pinned {self.pinned_gib:.2f} GiB + anon {self.anon_gib:.2f} GiB"
            f" = {self.total_gib:.2f} GiB"
            f"  ({self.total_gib / self.mark_gib * 100:.0f}% of the mark)",
        ]
        if self.shared:
            lines.append(
                f"  sharing saves {self.saved_gib:.2f} GiB against the per-process form"
            )
        by_layout: Dict[str, float] = {}
        for p in self.posts:
            by_layout[p.layout] = by_layout.get(p.layout, 0.0) + p.pinned_gib
        for name, gib in sorted(by_layout.items()):
            lines.append(f"    {name}: {gib:.2f} GiB pinned")
        return "\n".join(lines)


def solve_host_pool(
    posts: Sequence[HostPoolPost],
    shared: bool,
    mark_gib: float = HOST_MARK_GIB,
) -> HostPoolLedger:
    """Does the host survive both layouts being alive at once?

    `shared=False` is the naive form: every process page-locks its own
    expert pool, so the pinned term is the SUM over all processes.

    `shared=True` is the design: the pinned pool is held ONCE per physical
    expert set and both layouts read it, so the pinned term is the MAXIMUM
    per rank across layouts, not the sum. The anonymous term never shares --
    runtime, CUDA context and activations are per process either way, which
    is why this ledger keeps the two apart instead of quoting one total.
    """
    if not posts:
        raise Weg2FlipHostPoolDoubled(
            "W114 Weg2FlipHostPoolDoubled -- an empty ledger cannot prove a fit"
        )

    anon = sum(p.anon_gib for p in posts)
    naive_pinned = sum(p.pinned_gib for p in posts)
    if shared:
        per_rank: Dict[int, float] = {}
        for p in posts:
            per_rank[p.rank] = max(per_rank.get(p.rank, 0.0), p.pinned_gib)
        pinned = sum(per_rank.values())
    else:
        pinned = naive_pinned

    total = pinned + anon
    if total > mark_gib:
        layouts = sorted({p.layout for p in posts})
        raise Weg2FlipHostPoolDoubled(
            f"W114 Weg2FlipHostPoolDoubled -- {len(posts)} live process(es) across "
            f"layout(s) {layouts} hold {pinned:.2f} GiB page-locked "
            f"+ {anon:.2f} GiB anonymous = {total:.2f} GiB against a host mark of "
            f"{mark_gib:.1f} GiB (over by {total - mark_gib:.2f} GiB). "
            + (
                "The pool is ALREADY shared, so the mark cannot be met by sharing "
                "more -- the remaining levers are fewer resident-equivalent "
                "experts per layout or a smaller anonymous footprint."
                if shared
                else f"The pool is PER-PROCESS. Sharing it would take the pinned "
                f"term from {naive_pinned:.2f} to "
                f"{sum(max(q.pinned_gib for q in posts if q.rank == r) for r in sorted({p.rank for p in posts})):.2f} GiB."
            )
        )

    return HostPoolLedger(
        posts=tuple(posts),
        shared=shared,
        mark_gib=mark_gib,
        pinned_gib=pinned,
        anon_gib=anon,
        total_gib=total,
        saved_gib=naive_pinned - pinned,
    )


# ==========================================================================
# Seam (c): GDN/Mamba state and draft state across the flip
# ==========================================================================
@dataclass(frozen=True)
class CarriedState:
    """One per-request state the decode layout needs after the flip.

    `disposition` is exactly one of:
      "carried"    -- a P-side tensor exists and the flip forwards it;
      "rebuilt"    -- no P-side partner (or forwarding is more expensive
                      than recomputing); the D side rebuilds it, and
                      `rebuild_tokens` says at what cost per request;
      "absent"     -- the D side does not need it at all.

    Anything else, or "rebuilt" without a declared cost, is W115: a state
    that is neither carried nor priced is a state that will be silently
    wrong on the first request after the first flip.
    """

    name: str
    owner_p: str
    owner_d: str
    bytes_per_request: int
    disposition: str
    rebuild_tokens: int = 0


@dataclass(frozen=True)
class StateCarryPlan:
    states: Tuple[CarriedState, ...]
    carried_bytes: int
    rebuild_tokens: int
    prefill_tok_s: float
    rebuild_seconds: float

    def report(self) -> str:
        lines = ["STATE CARRY across the flip"]
        for s in self.states:
            extra = (
                f", {s.rebuild_tokens} tok to rebuild"
                if s.disposition == "rebuilt"
                else ""
            )
            lines.append(
                f"  {s.name:<24} P={s.owner_p:<10} D={s.owner_d:<10} "
                f"{s.bytes_per_request / 1024.0 / 1024.0:8.2f} MiB/req  "
                f"{s.disposition}{extra}"
            )
        lines.append(
            f"  carried {self.carried_bytes / 1024.0 / 1024.0:.2f} MiB/req; "
            f"rebuild {self.rebuild_tokens} tok/req "
            f"= {self.rebuild_seconds * 1000:.0f} ms at {self.prefill_tok_s:.0f} tok/s"
        )
        return "\n".join(lines)


_VALID_DISPOSITIONS = ("carried", "rebuilt", "absent")


def solve_state_carry(
    states: Sequence[CarriedState],
    prefill_tok_s: float,
) -> StateCarryPlan:
    if prefill_tok_s <= 0:
        raise Weg2FlipDraftStateOrphaned(
            f"W115 Weg2FlipDraftStateOrphaned -- prefill_tok_s={prefill_tok_s} "
            f"cannot price a rebuild"
        )
    for s in states:
        if s.disposition not in _VALID_DISPOSITIONS:
            raise Weg2FlipDraftStateOrphaned(
                f"W115 Weg2FlipDraftStateOrphaned -- state {s.name!r} has "
                f"disposition {s.disposition!r}, not one of {_VALID_DISPOSITIONS}"
            )
        if s.disposition == "carried" and not s.owner_p:
            raise Weg2FlipDraftStateOrphaned(
                f"W115 Weg2FlipDraftStateOrphaned -- state {s.name!r} is declared "
                f"CARRIED but has no P-side owner: there is no tensor to forward. "
                f"D-side owner is {s.owner_d!r}. If this is the Form-A solo draft, "
                f"the honest disposition is 'rebuilt' with a token cost, not "
                f"'carried' with an empty source."
            )
        if s.disposition == "rebuilt" and s.rebuild_tokens <= 0:
            raise Weg2FlipDraftStateOrphaned(
                f"W115 Weg2FlipDraftStateOrphaned -- state {s.name!r} is declared "
                f"REBUILT but priced at {s.rebuild_tokens} tokens. A rebuild with "
                f"no cost is an undeclared cost; it belongs in the flip-time "
                f"budget, not in a footnote."
            )

    carried = sum(s.bytes_per_request for s in states if s.disposition == "carried")
    rebuild = sum(s.rebuild_tokens for s in states if s.disposition == "rebuilt")
    return StateCarryPlan(
        states=tuple(states),
        carried_bytes=carried,
        rebuild_tokens=rebuild,
        prefill_tok_s=prefill_tok_s,
        rebuild_seconds=rebuild / prefill_tok_s,
    )


# ==========================================================================
# The measured defaults. Every one of these carries its boot by name.
# ==========================================================================
PP3_KV = KvLayout(
    name="PP3 prefill (29/11/8)",
    boot="boot_fn_fn7s_20260919T183945Z.server.log:263-265",
    # KV pool sizing lines, in PP-stage order 0,1,2.
    cells=(7616, 3264, 2176),
    avail_bytes=(8298127360, 5305303040, 7537844224),
    # line 21: "full-attention per stage: [7, 3, 2] of 12 total"
    full_attn_per_rank=(7, 3, 2),
)

FORM_A_KV = KvLayout(
    name="Form A decode (host + 2 workers)",
    boot="boot_fn_fnFA19_20260920T152302Z.server.log:1000-1002",
    # TP0 is the attention host; TP1/TP2 are pure expert workers and hold a
    # 768 B stub cell only (their "KV Cache is allocated" lines read
    # K size: 0.00 GB, V size: 0.00 GB).
    cells=(14143, 768, 768),
    avail_bytes=(7171801088, 2420113408, 2208301056),
    full_attn_per_rank=(12, 0, 0),
)

# [offload-kv-regain] "N GiB moved to the pinned host pool, N GiB page-locked
# exactly" -- fn7s lines 242-244, fnFA19 lines 989-991, in rank order.
PP3_PINNED_GIB = (22.73, 5.30, 3.85)
FORM_A_PINNED_GIB = (20.62, 7.36, 10.88)
