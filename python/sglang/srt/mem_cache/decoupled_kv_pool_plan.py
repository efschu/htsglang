"""#704b R6: size the KV pool by TOKEN share, not by layer ownership.

Today a PP stage's attention pool is dimensioned by which layers it OWNS.
``model_runner_kv_cache_mixin.py:2496-2500`` filters the model's attention
layer ids to ``[start_layer, end_layer)`` before the pool is built, so a rank
has no row-space for a layer whose weights it does not hold:

    stage-local:  own_attn_layers  x  ALL tokens

B1 inverts that. With the decoupled-KV group armed, every rank holds **all**
attention layers but only its **share of tokens**:

    decoupled:    ALL attn_layers  x  own token share

**The world total is unchanged, and that is the invariant worth holding onto.**
Summed over ranks, stage-local gives ``sum(own_attn_i) x T = 16 x T`` and
decoupled gives ``16 x sum(share_i) x T = 16 x T``. Decoupling REDISTRIBUTES
capacity; it does not create or destroy any. A plan whose world total moved is
a bug, not a bigger pool, and :func:`validate_world_conservation` says so.

**Seam to #706, stated precisely because two nearby facts point opposite ways.**
A canonical page is one token x all attention layers. Under this plan a rank's
pool holds all attention layers for its own tokens, so **residence is
whole-page per rank** — a page never straddles ranks, which is exactly what the
canonical store wants. That does NOT remove the completeness marker: PRODUCTION
is still layer-sharded (stage 0 computes layers 0-27, stage 1 the next span),
so a page's 16 slots still arrive from three writers. Residence is whole;
authorship is not.

Scope: this module PLANS and VALIDATES. It does not arm anything and does not
build pools — caller wiring stays out until this lands.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Sequence

STAGE_LOCAL = "stage_local"
DECOUPLED = "decoupled"


class KvPoolPlanError(ValueError):
    """A pool plan that cannot be honoured. Never downgraded to a warning."""


@dataclasses.dataclass(frozen=True)
class KvPoolPlan:
    """What one rank's attention pool must hold."""

    mode: str
    layer_ids: tuple[int, ...]
    tokens: int
    cell_bytes_per_layer: int

    @property
    def bytes_total(self) -> int:
        return len(self.layer_ids) * int(self.tokens) * int(self.cell_bytes_per_layer)

    @property
    def holds_all_layers(self) -> bool:
        return self.mode == DECOUPLED


def plan_stage_local(
    all_attn_layer_ids: Sequence[int],
    start_layer: int,
    end_layer: int,
    max_total_tokens: int,
    cell_bytes_per_layer: int,
) -> KvPoolPlan:
    """Today's shape, reproduced exactly.

    Mirrors the filter at ``model_runner_kv_cache_mixin.py:2496-2500`` so the
    unarmed path can be pinned byte-identical rather than merely "similar".
    """
    ids = tuple(i for i in all_attn_layer_ids if int(start_layer) <= i < int(end_layer))
    return KvPoolPlan(STAGE_LOCAL, ids, int(max_total_tokens), int(cell_bytes_per_layer))


def plan_decoupled(
    all_attn_layer_ids: Sequence[int],
    share: float,
    max_total_tokens: int,
    cell_bytes_per_layer: int,
) -> KvPoolPlan:
    """B1-armed shape: every attention layer, this rank's token share.

    ``share`` is this rank's fraction of the world's tokens, from the free-byte
    vector -- NOT from its layer count. Deriving it from layers would rebuild
    the ownership coupling this slice exists to remove.
    """
    if not 0.0 < float(share) <= 1.0:
        raise KvPoolPlanError(
            f"token share {share!r} is not in (0, 1]. A rank holding no tokens "
            "cannot serve a decoupled read, and a share above 1 would size the "
            "pool past the world."
        )
    ids = tuple(int(i) for i in all_attn_layer_ids)
    if not ids:
        raise KvPoolPlanError("no attention layers: there is no KV pool to plan.")
    tokens = round(float(share) * int(max_total_tokens))
    if tokens <= 0:
        raise KvPoolPlanError(
            f"share {share} of {max_total_tokens} tokens rounds to zero rows; "
            "refusing to build a pool that can hold nothing."
        )
    return KvPoolPlan(DECOUPLED, ids, tokens, int(cell_bytes_per_layer))


def plan_for_rank(
    all_attn_layer_ids: Sequence[int],
    start_layer: int,
    end_layer: int,
    max_total_tokens: int,
    cell_bytes_per_layer: int,
    armed: bool = False,
    share: float | None = None,
) -> KvPoolPlan:
    """The single entry point. Unarmed is today's plan, byte-for-byte."""
    if not armed:
        if share is not None:
            raise KvPoolPlanError(
                "a token share was supplied but B1 is not armed. Refusing: a "
                "share that silently does nothing is how a decoupled pool ends "
                "up sized like a stage-local one."
            )
        return plan_stage_local(
            all_attn_layer_ids,
            start_layer,
            end_layer,
            max_total_tokens,
            cell_bytes_per_layer,
        )
    if share is None:
        raise KvPoolPlanError(
            "B1 is armed but no token share was supplied. Under decoupling the "
            "pool is sized by token share, and defaulting it to the layer "
            "fraction would rebuild the ownership coupling this removes."
        )
    return plan_decoupled(
        all_attn_layer_ids, share, max_total_tokens, cell_bytes_per_layer
    )


def validate_plan(plan: KvPoolPlan, all_attn_layer_ids: Sequence[int]) -> None:
    """Fail loudly on a pool that cannot serve the mode it claims.

    The decoupled case is the one that matters: a rank that holds only its own
    layers cannot answer a read for a layer another stage owns, and the failure
    would surface as wrong output rather than as a missing row.
    """
    expected = tuple(int(i) for i in all_attn_layer_ids)
    if plan.mode == DECOUPLED and plan.layer_ids != expected:
        missing = [i for i in expected if i not in plan.layer_ids]
        raise KvPoolPlanError(
            f"decoupled pool is missing attention layer(s) {missing}: under B1 "
            "every rank must hold ALL attention layers, because a read may "
            "target any layer regardless of which stage owns its weights. A "
            "pool sized by ownership would answer that read from rows it does "
            "not have."
        )
    if plan.tokens <= 0 or not plan.layer_ids:
        raise KvPoolPlanError("a pool with no rows or no layers cannot serve.")


def validate_world_conservation(
    plans: Sequence[KvPoolPlan], all_attn_layer_ids: Sequence[int], max_total_tokens: int
) -> None:
    """The world total must not move between modes.

    Decoupling redistributes capacity; it does not create it. A plan set whose
    total differs from ``len(attn_layers) x max_total_tokens x cell`` is a
    sizing bug that would present as a surprisingly large -- or small -- pool.
    """
    if not plans:
        raise KvPoolPlanError("no plans to check.")
    cell = plans[0].cell_bytes_per_layer
    if any(p.cell_bytes_per_layer != cell for p in plans):
        raise KvPoolPlanError(
            "ranks disagree about the KV cell width; the world total is not "
            "comparable and one of them is mis-configured."
        )
    want = len(tuple(all_attn_layer_ids)) * int(max_total_tokens) * int(cell)
    got = sum(p.bytes_total for p in plans)
    # One row per rank of rounding slack on the share split.
    slack = len(plans) * len(tuple(all_attn_layer_ids)) * int(cell)
    if abs(got - want) > slack:
        raise KvPoolPlanError(
            f"world KV total is {got:,} bytes against {want:,} expected "
            f"(slack {slack:,}). Decoupling redistributes capacity, it does not "
            "change it; a total that moved means the shares do not sum to one "
            "or a rank was sized from the wrong token count."
        )


def layer_extents(plan: KvPoolPlan, all_attn_layer_ids: Sequence[int]) -> tuple:
    """The plan in the canonical store's terms, for the #706 seam.

    A canonical page is one token x ALL attention layers, laid out layer-major.
    Returns ``(slot_index, byte_offset, byte_length)`` per layer this rank
    holds, where ``slot_index`` is the GLOBAL attention index -- the same index
    the canonical page uses, never a rank-local one.

    Under ``DECOUPLED`` this covers every slot, so a rank's residence is a
    whole page. Under ``STAGE_LOCAL`` it covers only the owned slots, which is
    precisely the partial-page shape the canonical store has to reassemble.
    """
    order = tuple(int(i) for i in all_attn_layer_ids)
    cell = int(plan.cell_bytes_per_layer)
    out = []
    for layer in plan.layer_ids:
        if layer not in order:
            raise KvPoolPlanError(
                f"layer {layer} is not a full-attention layer; it has no "
                "canonical page slot."
            )
        slot = order.index(layer)
        out.append((slot, slot * cell, cell))
    return tuple(out)
