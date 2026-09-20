# SPDX-License-Identifier: Apache-2.0
"""Slice 5 -- seam C: every per-request state is CARRIED or PRICED, or W115.

``flip_nextflash_plan.solve_state_carry`` is the rule: a state declared
``carried`` needs a P-side owner, a state declared ``rebuilt`` needs a token
cost, and anything else is W115. What that rule could not do is know WHICH
states exist -- it was fed a hand list, and a hand list is exactly how the
QSA raw-key ring went missing (design risk R5).

So this module supplies the states, from the measured boots, AND the check
that the list is complete: :func:`solve_flip_state_carry` takes the runner's
ACTUAL pool inventory and refuses a family that is not declared. A state that
is neither carried nor priced is not a gap in a plan -- it is a request that
decodes from a recurrent state belonging to nobody, on the first request
after the first flip, silently.

THE THREE KINDS, and why they are three and not one:

  KV (token-addressable)        -- seam A's business, not this module's. It
                                   can be read "up to token N" and re-laid.
  RECURRENT (not addressable)   -- GDN/mamba state, the QSA raw-key ring.
                                   There is no "up to token N": the state IS
                                   the result of every token so far. It must
                                   travel whole or be recomputed, and
                                   recomputing GDN at 262k means running the
                                   whole prefill again. So: CARRIED.
  SPECULATIVE (cheap to rebuild) -- the MTP draft. Form A runs
                                   ``--speculative-draft-placement solo``, so
                                   the draft lives on the host alone; PP3
                                   SHARDS its draft across the stages
                                   (Memory ``DRAFT-ZUORDNUNG``: every draft is
                                   sharded). There is therefore NO P-side
                                   tensor to forward. That is fine -- it
                                   rebuilds in ~4 tokens -- but it must be
                                   DECLARED, and the declaration is what W115
                                   enforces.

THE ASYMMETRY IN THE GDN STATE, measured. fnFA19:1029-1031 on TP0:
``conv_state 0.01 GB, ssm_state 0.11 GB, intermediate_ssm_state_cache
0.32 GB, intermediate_conv_window_cache 0.01 GB``; on TP1/TP2 every one of
them is ``0.00`` -- the workers hold no GDN state at all. fn7s:
``ssm_state 0.06 GB`` on PP0 and ``0.02`` on PP1/PP2, so P spreads it over
three cards. P DISTRIBUTES, D CONCENTRATES: the carry is a gather, not a
copy, and the plan says which rank each piece comes from.

THE CARRIER ALREADY EXISTS AND IS NOT BUILT HERE. ``UPSTREAM-MINIMAL``:
``mem_cache/hi_mamba_radix_cache.py`` has the device->host write path
(``mamba_backup_transfers`` / ``mamba_backup_commit``, :2506-2552) and the
host->device read path (``mamba_prefetch_alloc`` / ``mamba_restore_transfers``
/ ``mamba_restore_commit``, :2565-2636), plus
``mem_cache/mamba_checkpoint_pool.py`` (``store_from_active`` :245,
``load_to_active`` :253). Today it is a PREFIX-CACHE path (eviction), not a
layout-change path -- the only touch point in ``phase_flip_runtime.py`` is a
comment (:1698). Slice 5's work is RE-PURPOSING, not building, and this plan
is what names the pieces that path must move.

Pure: no torch, no device. Every default carries its boot line.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

from sglang.srt.flip_nextflash_plan import (
    CarriedState,
    StateCarryPlan,
    Weg2FlipDraftStateOrphaned,
    solve_state_carry,
)

__all__ = [
    "KIND_RECURRENT",
    "KIND_SPECULATIVE",
    "KIND_KV_FOLLOWING",
    "StateFamily",
    "NEXT_FLASH_STATES",
    "PP3_PREFILL_TOK_S",
    "DRAFT_REBUILD_TOKENS",
    "solve_flip_state_carry",
    "FlipStateCarry",
]

KIND_RECURRENT = "recurrent"
KIND_SPECULATIVE = "speculative"
KIND_KV_FOLLOWING = "kv_following"

#: fn7s, compute-honest: 3.0 s per 8192-token chunk on the binding stage.
PP3_PREFILL_TOK_S = 2729.0

#: The MTP draft is a few tokens deep and rebuilds from the end of the
#: prefill. At 2729 tok/s that is under 2 ms -- far below the ~2 s physics
#: floor, which is WHY it is affordable, and not why it may go unpriced.
DRAFT_REBUILD_TOKENS = 4


@dataclass(frozen=True)
class StateFamily:
    """One per-request state family, with the evidence for its disposition."""

    name: str
    kind: str
    owner_p: str
    owner_d: str
    bytes_per_request: int
    disposition: str
    rebuild_tokens: int = 0
    why: str = ""

    def as_carried_state(self) -> CarriedState:
        return CarriedState(
            name=self.name,
            owner_p=self.owner_p,
            owner_d=self.owner_d,
            bytes_per_request=self.bytes_per_request,
            disposition=self.disposition,
            rebuild_tokens=self.rebuild_tokens,
        )


_MB = 1024 * 1024

#: The declared families. Anything the runner holds that is NOT here is W115.
NEXT_FLASH_STATES: Tuple[StateFamily, ...] = (
    StateFamily(
        name="mamba_conv_state",
        kind=KIND_RECURRENT,
        owner_p="PP0/1/2 (split)",
        owner_d="host TP0",
        bytes_per_request=10 * _MB,
        disposition="carried",
        why="fnFA19:1029 conv_state 0.01 GB on TP0, 0.00 on TP1/TP2",
    ),
    StateFamily(
        name="mamba_ssm_state",
        kind=KIND_RECURRENT,
        owner_p="PP0/1/2 (split)",
        owner_d="host TP0",
        bytes_per_request=110 * _MB,
        disposition="carried",
        why=(
            "fnFA19:1030 ssm_state 0.11 GB on TP0; fn7s 0.06 on PP0 and 0.02 "
            "on PP1/PP2 -- P distributes, D concentrates, so the carry is a "
            "GATHER and not a copy"
        ),
    ),
    StateFamily(
        name="intermediate_ssm_state_cache",
        kind=KIND_RECURRENT,
        owner_p="PP0/1/2 (split)",
        owner_d="host TP0",
        bytes_per_request=320 * _MB,
        disposition="carried",
        why="fnFA19:1030 intermediate_ssm_state_cache 0.32 GB on TP0",
    ),
    StateFamily(
        name="intermediate_conv_window_cache",
        kind=KIND_RECURRENT,
        owner_p="PP0/1/2 (split)",
        owner_d="host TP0",
        bytes_per_request=10 * _MB,
        disposition="carried",
        why="fnFA19:1031 intermediate_conv_window_cache 0.01 GB on TP0",
    ),
    StateFamily(
        name="qsa_pending_raw_key_ring",
        kind=KIND_RECURRENT,
        owner_p="PP stage owning the QSA layer",
        owner_d="host TP0",
        bytes_per_request=8 * _MB,
        disposition="carried",
        why=(
            "qsa_indexer.py:256-276 _pending_ring_slots/_group_ring_slots -- a "
            "ring of not-yet-compressed RAW keys. Recurrent like GDN, NOT "
            "token-addressable, and at prefill end up to one group deep. This "
            "is design risk R5: it was MISSING from the hand list, which is "
            "why the list is no longer a hand list"
        ),
    ),
    StateFamily(
        name="qsa_compressed_key_state",
        kind=KIND_KV_FOLLOWING,
        owner_p="PP stage owning the QSA layer",
        owner_d="host TP0",
        bytes_per_request=0,
        disposition="absent",
        why=(
            "qsa_indexer.py:296 set_qsa_key_state_buffer -- allocated IN the "
            "KV pool, so it follows the KV through seam A and is not a "
            "separate carry (bytes 0 here on purpose: double-counting it "
            "would inflate seam C with seam A's mass)"
        ),
    ),
    StateFamily(
        name="qsa_rope_position_buffer",
        kind=KIND_KV_FOLLOWING,
        owner_p="PP stage owning the QSA layer",
        owner_d="host TP0",
        bytes_per_request=0,
        disposition="absent",
        why="qsa_indexer.py:299 set_qsa_rope_position_buffer -- also in the KV pool",
    ),
    StateFamily(
        name="mtp_solo_draft_state",
        kind=KIND_SPECULATIVE,
        owner_p="",  # deliberately empty -- there IS no P-side partner
        owner_d="host TP0 (solo)",
        bytes_per_request=0,
        disposition="rebuilt",
        rebuild_tokens=DRAFT_REBUILD_TOKENS,
        why=(
            "Form A runs --speculative-draft-placement solo; fnFA19:998 "
            "'Draft-solo KV planning: rank 1/2 draft-KV cell term 64 -> 0 "
            "B/token'. PP3 SHARDS its draft (Memory DRAFT-ZUORDNUNG), so no "
            "P-side tensor exists to forward. Rebuilt, and PRICED -- the point "
            "of W115 is not the cost, it is the silence"
        ),
    ),
)


@dataclass(frozen=True)
class FlipStateCarry:
    plan: StateCarryPlan
    families: Tuple[StateFamily, ...]
    carried_bytes: int
    rebuild_tokens: int
    rebuild_seconds: float
    gathered_from: Dict[str, Tuple[str, ...]]

    def report(self) -> str:
        lines = [
            f"STATE CARRY (seam C) -- {len(self.families)} declared families",
            f"  carried {self.carried_bytes / _MB:.0f} MiB/req, "
            f"rebuild {self.rebuild_tokens} tok/req "
            f"= {self.rebuild_seconds * 1000:.1f} ms at {PP3_PREFILL_TOK_S:.0f} tok/s",
        ]
        for f in self.families:
            extra = (
                f" ({f.rebuild_tokens} tok)" if f.disposition == "rebuilt" else ""
            )
            lines.append(
                f"    {f.name:<32} {f.kind:<13} {f.disposition}{extra}"
            )
        return "\n".join(lines)


def solve_flip_state_carry(
    live_families: Optional[Iterable[str]] = None,
    families: Sequence[StateFamily] = NEXT_FLASH_STATES,
    prefill_tok_s: float = PP3_PREFILL_TOK_S,
) -> FlipStateCarry:
    """Every live per-request state is carried or priced -- or W115.

    ``live_families`` is the runner's ACTUAL inventory. Passing ``None`` plans
    the declared set alone, which is useful at the desk and is NOT what the
    cutover should do: feeding the real inventory is the entire mechanism by
    which an undeclared family (the QSA ring, a hyper-connection mixer buffer)
    falls out as W115 instead of being silently absent.
    """
    declared = {f.name: f for f in families}
    if live_families is not None:
        live = [str(n) for n in live_families]
        unknown = sorted({n for n in live if n not in declared})
        if unknown:
            raise Weg2FlipDraftStateOrphaned(
                f"W115 Weg2FlipDraftStateOrphaned -- the runner holds "
                f"per-request state families {unknown} that seam C does not "
                f"declare. Each is either CARRIED (name its P-side owner), "
                f"REBUILT (name its token cost) or ABSENT under Form A (say "
                f"why) -- but it may not be undeclared: after the first flip "
                f"the D side would decode from a recurrent state belonging to "
                f"nobody, and an unwritten state reads as a valid one. This is "
                f"design risk R5, which is exactly how "
                f"qsa_pending_raw_key_ring went missing from the first hand "
                f"list."
            )
        missing = sorted(
            n for n, f in declared.items()
            if n not in live and f.disposition != "absent"
        )
        if missing:
            raise Weg2FlipDraftStateOrphaned(
                f"W115 Weg2FlipDraftStateOrphaned -- seam C declares "
                f"{missing} as state the flip must handle, but the runner does "
                f"not hold them. Either the layout changed and the declaration "
                f"is stale, or the inventory is being read from the wrong "
                f"place; both make the carry a claim about something that is "
                f"not there."
            )
        plan_families = tuple(f for n, f in declared.items() if n in live)
    else:
        plan_families = tuple(families)

    # The rule itself: P-side owner for a carry, a price for a rebuild.
    plan = solve_state_carry(
        [f.as_carried_state() for f in plan_families], prefill_tok_s
    )

    gathered: Dict[str, Tuple[str, ...]] = {}
    for f in plan_families:
        if f.disposition == "carried":
            gathered[f.name] = tuple(
                p.strip() for p in f.owner_p.replace("(split)", "").split("/") if p.strip()
            )

    return FlipStateCarry(
        plan=plan,
        families=plan_families,
        carried_bytes=plan.carried_bytes,
        rebuild_tokens=plan.rebuild_tokens,
        rebuild_seconds=plan.rebuild_seconds,
        gathered_from=gathered,
    )
