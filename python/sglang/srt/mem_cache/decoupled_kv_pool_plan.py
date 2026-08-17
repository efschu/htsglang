"""#704b R6: size the KV pool by TOKEN share, not by layer ownership.

Today a PP stage's attention pool is dimensioned by which layers it OWNS.
``model_runner_kv_cache_mixin.py:3929-3934`` filters the model's attention
layer ids to ``[start_layer, end_layer)`` at the ``HybridLinearKVPool``
construction (``:3913``) before the pool is built, so a rank has no row-space
for a layer whose weights it does not hold:

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

REVIEW (reconciliation with the neighbour commits; corrects this module's
first commit message, which reported them as unreachable — that report was
wrong, and the miss was mine: I searched the decorated forms ``[#646]`` and
``[#635]`` instead of the bare numbers the subjects actually use).

``7fd2b566b2`` "R6 verdict (#635)" is a DIFFERENT R6. It is a risk-register
item in ``docs/dev/DESIGN_625.md`` about the PD **handover** — moving KV
between two engines — and it was already superseded there ("New R7 replaces
R6"). This module's R6 is a #704b slice about pool **allocation** inside one
engine. Two registers, one letter-number. Not a duplicate.

They are COMPLEMENTARY and they meet on one object: the token-sharded pool.
#635 established that the receive-side owner-rule scatter is already built
(``BaseKVSender.send_metadata`` owned_ordinals, ``base/conn.py:186-204``);
that says how bytes ARRIVE into such a pool. This module says how one is
SIZED. The meeting point is the share — and that is where the reconciliation
found a defect rather than a tidy fit. See ``period`` on :func:`plan_decoupled`.
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
    # Owner-rule period this plan's row count was ceiled against, DECOUPLED
    # only. Carried so world conservation can derive the ceil's slack instead
    # of being told it -- a slack passed in by the caller is a slack that gets
    # widened until the check passes.
    period: int | None = None

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

    Mirrors the filter at ``model_runner_kv_cache_mixin.py:3929-3934`` so the
    unarmed path can be pinned byte-identical rather than merely "similar".

    CORRECTION (arming slice): this docstring first cited ``:2496-2500``. That
    site is real but sits in ``_init_unified_mamba_pools`` (``:2467``), which
    ``_init_pools`` only reaches under ``--enable-unified-memory``
    (``:3194-3200``) -- a path THIS deployment does not take. The filter that
    governs our hybrid-mamba config is the classic-path one cited above. The
    logic is the same shape in both, so the model was right; the citation
    pointed at a path the byte-identity claim was never tested against.
    """
    ids = tuple(i for i in all_attn_layer_ids if int(start_layer) <= i < int(end_layer))
    return KvPoolPlan(STAGE_LOCAL, ids, int(max_total_tokens), int(cell_bytes_per_layer))


def _validate_target_section(all_attn_layer_ids: Sequence[int]) -> None:
    """Refuse a layer list that is not one clean target section (#646).

    ``b851df7626`` fixed a defect in a NEIGHBOURING flat list: the PD MHA
    registration recovered its K/V boundary by halving the list, which was
    exact until ``prefill.py``/``decode.py`` appended the DRAFT pool to the
    same list, after which the half-split cut through V_main and paired source
    V buffers with destination K buffers — silently, with no exception on the
    mispaired path. The fix was to record the section size up front and split
    on the declared boundary, with the draft section owned WHOLE by the last PP
    stage rather than split.

    This module consumes a different list (the model's attention layer ids from
    ``model_runner_kv_cache_mixin.py:2496-2500``, which the draft pool does not
    enter), so it does not inherit that defect. It adopts the RULE anyway,
    because the armed path shards by token share whatever it is handed: were a
    draft section ever appended here, those layers would be sized as target
    layers and split across ranks, contradicting whole-section ownership. A
    strictly increasing, duplicate-free list is what one clean section looks
    like; anything else is refused by name rather than sharded on trust.

    Armed path only. The unarmed path reproduces today's filter byte-for-byte
    and must not acquire a new way to fail.
    """
    ids = tuple(int(i) for i in all_attn_layer_ids)
    if len(set(ids)) != len(ids):
        dupes = sorted({i for i in ids if ids.count(i) > 1})
        raise KvPoolPlanError(
            f"attention layer ids repeat {dupes}: this is not one target "
            "section. A second section appended to the same list (cf. #646, "
            "b851df7626) would be sized as target layers and split by token "
            "share, when such a section is owned whole."
        )
    if list(ids) != sorted(ids):
        raise KvPoolPlanError(
            f"attention layer ids {ids} are not in increasing order; a "
            "concatenated list is not a target section (#646)."
        )


def realized_share(period: int, lo: int, hi: int) -> float:
    """The share the owner rule ACTUALLY hands a rank owning band ``[lo, hi)``.

    ``layers/dcp/owner.py:406-436`` assigns slot ``L`` to this rank iff
    ``(L % period)`` is in ``[lo, hi)``, so the realized share is exactly
    ``(hi - lo) / period`` — a rational with denominator ``period``, never a
    free real. This function exists so the pool planner and the actuator read
    the share off the SAME expression instead of two copies that can drift.
    """
    period, lo, hi = int(period), int(lo), int(hi)
    if period <= 0:
        raise KvPoolPlanError(f"owner-rule period {period} must be positive.")
    if not 0 <= lo < hi <= period:
        raise KvPoolPlanError(
            f"owner band [{lo}, {hi}) is not a sub-range of [0, {period})."
        )
    return (hi - lo) / period


def validate_share_realizable(share: float, period: int) -> None:
    """Refuse a share the built owner rule cannot express.

    This is the #635 reconciliation turned into a check. The handover door
    that #635 found already built scatters rows by the weighted owner rule,
    whose ownership granularity is ``1 / period`` (``owner.py:406-436``, the
    same expression on the read and write sides). A pool sized from a share
    OFF that grid is sized for a token count the rank will never be given —
    and the mismatch is silent: ``loc = block * cp_ratio + (off - cp_lo)``
    indexes an allocation that was never made that long, so an under-sized
    pool writes past its end rather than raising.

    ``planner/decoupled_kv.py:270-273`` already stated this constraint; the
    first cut of this module accepted a bare float and never enforced it. Use
    :func:`planner.decoupled_kv.quantize_shares` to land on the grid FIRST —
    this function refuses, it does not round, because rounding here would hide
    that the caller skipped the quantiser.
    """
    period = int(period)
    if period <= 0:
        raise KvPoolPlanError(f"owner-rule period {period} must be positive.")
    slots = float(share) * period
    nearest = round(slots)
    if abs(slots - nearest) > 1e-9:
        raise KvPoolPlanError(
            f"token share {share!r} is not realizable at owner-rule period "
            f"{period}: it asks for {slots:.6f} of {period} residue slots, and "
            f"the rule can only hand out whole ones (nearest {nearest} = "
            f"{nearest / period:.6f}). Sizing a pool off the grid gives the "
            "rank rows it will never be handed, or fewer than it will — "
            "quantize the share first, do not round it here."
        )


def plan_decoupled(
    all_attn_layer_ids: Sequence[int],
    share: float,
    max_total_tokens: int,
    cell_bytes_per_layer: int,
    period: int,
) -> KvPoolPlan:
    """B1-armed shape: every attention layer, this rank's token share.

    ``share`` is this rank's fraction of the world's tokens, from the free-byte
    vector -- NOT from its layer count. Deriving it from layers would rebuild
    the ownership coupling this slice exists to remove.

    ``period`` is the owner rule's residue period (``cp_S``), and it is
    REQUIRED rather than optional on purpose: without it this function cannot
    tell a realizable share from an unrealizable one, and an optional check is
    one the arming path can forget. This mirrors #646's rule for a neighbouring
    list — declare the structure, never infer it.
    """
    validate_share_realizable(share, period)
    _validate_target_section(all_attn_layer_ids)
    if not 0.0 < float(share) <= 1.0:
        raise KvPoolPlanError(
            f"token share {share!r} is not in (0, 1]. A rank holding no tokens "
            "cannot serve a decoupled read, and a share above 1 would size the "
            "pool past the world."
        )
    ids = tuple(int(i) for i in all_attn_layer_ids)
    if not ids:
        raise KvPoolPlanError("no attention layers: there is no KV pool to plan.")
    # Row count comes from the SHIPPED rule, not from a second one here.
    #
    # ``layers/dcp/owner.py:155-181`` dcp_compact_pool_rows is already the
    # sizing actuator on this path: the classic hybrid-mamba build calls it at
    # ``model_runner_kv_cache_mixin.py:3765-3767``, and its own docstring says
    # the rule is stated once because "a sizing rule whose off-by-one has
    # already cost a debugging round must not exist twice".
    #
    # This module had made it exist twice, and with the WRONG rounding:
    # ``round(share * C)`` can round DOWN, while the shipped rule ceils to a
    # whole owner block -- ``(C // cp_S + 1) * cp_ratio``. That ceil is not
    # cosmetic. Flooring let slots in a trailing partial block scatter out of
    # bounds (async illegal memory access, found by the kv-session-offload S1
    # test at --max-total-tokens 3000 on cp_S=64). A planner that hands the
    # build a floored row count re-creates exactly that bug at the seam.
    #
    # ``ratio_r`` is the rank's whole-slot band width; validate_share_realizable
    # above has already guaranteed ``share * period`` is integral, so this is a
    # conversion, not a rounding decision.
    from sglang.srt.layers.dcp.owner import dcp_compact_pool_rows

    ratio_r = round(float(share) * int(period))
    tokens = dcp_compact_pool_rows(int(max_total_tokens), int(period), ratio_r)
    if tokens <= 0:
        raise KvPoolPlanError(
            f"share {share} of {max_total_tokens} tokens rounds to zero rows; "
            "refusing to build a pool that can hold nothing."
        )
    return KvPoolPlan(DECOUPLED, ids, tokens, int(cell_bytes_per_layer), int(period))


def plan_for_rank(
    all_attn_layer_ids: Sequence[int],
    start_layer: int,
    end_layer: int,
    max_total_tokens: int,
    cell_bytes_per_layer: int,
    armed: bool = False,
    share: float | None = None,
    period: int | None = None,
) -> KvPoolPlan:
    """The single entry point. Unarmed is today's plan, byte-for-byte."""
    if not armed:
        if period is not None:
            raise KvPoolPlanError(
                "an owner-rule period was supplied but B1 is not armed. The "
                "unarmed pool is not token-sharded, so a period here describes "
                "a rule nothing applies."
            )
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
    if period is None:
        raise KvPoolPlanError(
            "B1 is armed but no owner-rule period was supplied. The period is "
            "what makes a share checkable against the rule that will actually "
            "hand out rows (owner.py:406-436); without it this call would size "
            "a pool from a share the actuator may be unable to express."
        )
    return plan_decoupled(
        all_attn_layer_ids, share, max_total_tokens, cell_bytes_per_layer, period
    )


def pool_build_args(plan: KvPoolPlan) -> tuple:
    """The two arguments the build site actually takes: ``(layer_ids, size)``.

    ``model_runner_kv_cache_mixin.py:3913`` constructs ``HybridLinearKVPool``
    with ``full_attention_layer_ids=`` (the filtered list, ``:3929-3934``) and
    ``size=`` (``_hybrid_pool_size``, ``:3765-3767``). Those are the only two
    dimensions decoupling moves, so this returns exactly them and nothing
    else: a build helper that returned a whole kwargs blob would invite
    carrying unrelated arguments through a path that has no business owning
    them.
    """
    return tuple(plan.layer_ids), int(plan.tokens)


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
    layers = len(tuple(all_attn_layer_ids))
    want = layers * int(max_total_tokens) * int(cell)
    got = sum(p.bytes_total for p in plans)
    # One row per rank of rounding slack on the share split, PLUS the ceil.
    #
    # dcp_compact_pool_rows ceils to a whole owner block, so a rank can exceed
    # its exact share by up to ratio_r rows: (C // S + 1) * ratio_r - C *
    # ratio_r / S = ratio_r * (1 - (C mod S) / S), which is at most ratio_r.
    # Summed over ranks that is at most sum(ratio_r) = S = period rows. The
    # bound is DERIVED from the period the plans were built against, so it
    # cannot be quietly widened to make a real discrepancy pass.
    slack = len(plans) * layers * int(cell)
    periods = {p.period for p in plans if p.period is not None}
    if len(periods) > 1:
        raise KvPoolPlanError(
            f"ranks disagree about the owner-rule period {sorted(periods)}; "
            "they are not sharding the same token space and the world total "
            "is not comparable."
        )
    if periods:
        slack += periods.pop() * layers * int(cell)
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
