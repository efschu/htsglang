"""#875: the layer axis of a layout-neutral seam carry.

WHAT THIS IS FOR. Across a phase flip the seam copy is taken in one geometry and
restored in another, and `restore_seam_state` currently REFUSES that case. The
refusal is correct -- it is the only thing between the tree and either the W40
IndexError or a silent wrong-layer write -- but it is not the answer: its own
counter comment says a layout refusal means "every flip loses its prefixes".
This module is the first of the three axes that answer has to cross.

#875 -- THE HEAD AXIS, SETTLED AFTER TWO WRONG JUDGEMENTS OF MINE. I deleted the
refusal's head leg, restored it, and deleted it again. Final measurement: BOTH
phases hold 4 kv-heads per layer, so the head axis needs NO remap.

The restoration was wrong because `uneven_dcp_kv_replicated`'s docstring names
`--rank-tp-ratio` while the process state it reads has a SECOND installer: the
flip installs its own vector (`phase_flip_tp_vector`, '32,16,16' here) at
phase_flip_boot.py:1428, before the TP worker is built at :1473. So at TP-pool
build time the predicate is TRUE, the #345 exception is taken, and the pool holds
the full 4 heads -- while PP, built earlier with ratios still None, holds
`max(1, 4 // 1)` = 4 as well. Same width, interchangeable entries.

THREE AXES, AND ONLY ONE OF THEM IS HERE:

  1. LAYER   PP is stage-sharded (8/4/4 attention layers on this rig, from
             `pp_attn_stage_ratio`), TP is complete (16). This module. The
             remap is exact once entries are labelled by GLOBAL layer id, which
             is what `CpuCopyLayout.start_layer` already records.
  2. HEAD    4 kv-heads per layer in BOTH phases. NO remap needed -- see the
             measurement above, and note it took three judgements to settle.
  3. TOKEN   PP holds every token at allocator slots; TP holds an owner-rule
             SUBSET at compacted rows, `(L // cp_S) * cp_ratio + (L % cp_S -
             cp_lo)` (layers/dcp/owner.py:159). A second remap, also not
             available rank-locally, and NOT solved here.

So this module is deliberately not a carry. It is the layer axis of one, pure
and desk-provable, with the collective injected by the caller. Nothing here
issues one: a collective on the per-request restore path would be the #630
wedge shape, and the place a real carry belongs is the flip's own
`pre_cutover_fns`, beside `gdn_state` and `weights_refill`, where the group is
already synchronised.

#875d -- THE HALF OF THIS THAT IS NOW WIRED, AND THE HALF THAT IS NOT.

The paragraph above used to end "NOTHING IN THIS MODULE IS WIRED INTO THE
RESTORE PATH YET", and the reason given was that a carry needs a collective. It
needs one for ONE of the two directions. The two are not symmetric:

  PP copy -> TP pool   the source is MISSING layers (8 of 16). They exist on a
                       peer and only a collective can fetch them. Still refused,
                       and refused BY NAME. `ec1717491f`'s DO-NOT-BUILD verdict
                       is about exactly this exchange and is not overturned:
                       nothing below issues, schedules or prepares a collective.

  TP copy -> PP pool   the source COVERS every layer the destination needs. The
                       carry is a rank-local SLICE -- no peer, no exchange, no
                       cutover cost. And this is the SILENT direction: the pool
                       loop simply runs fewer iterations and writes global
                       layers 0..7 into global 8..15, no crash, no log.

So the direction with nothing to make anyone look was always the one that needed
no collective at all, and the refusal was paying a recompute for it. That half is
wired (`plan_rank_local_carry` / `carry_payload`, used by `restore_seam_state`).

WHAT A CARRY MAY NEVER DO IS GUESS. The functions below act on layouts they can
IDENTIFY BY TYPE and payload shapes they can NAME. Anything else raises, and the
caller's pre-existing refusal is what runs -- so a pool family this module has
never seen keeps exactly the behaviour it has today rather than being handed a
confident slice of a structure nobody checked.
"""

from __future__ import annotations

from typing import Dict, Iterable, List, NamedTuple, Sequence, Tuple


class SeamCarryError(RuntimeError):
    """A carry that cannot be completed. Never downgraded to a partial restore.

    A PARTIAL RESTORE IS THE ONE OUTCOME THIS MODULE MAY NOT PRODUCE. Filling 8
    of 16 layers leaves the other 8 holding whatever the destination rows
    carried before, under a prefix the tree reports as restored -- a wrong
    ANSWER, which is strictly worse than the recompute a refusal costs. Same
    rule the extent contract states for its own axis (schedule_batch.py: "NOT A
    CLAMP").
    """


def global_layer_ids(layer_num: int, start_layer: int) -> range:
    """The GLOBAL layer ids a pool with this geometry holds, in local order.

    Local slot i of a pool is global layer ``start_layer + i`` -- the same
    arithmetic `local_slot` uses in reverse (memory_pool.py:2288). Stated as a
    function so the two directions cannot drift apart.
    """
    if int(layer_num) < 0:
        raise SeamCarryError(f"layer_num must be non-negative, got {layer_num}")
    return range(int(start_layer), int(start_layer) + int(layer_num))


def label_contributions(
    layer_num: int, start_layer: int, entries: Sequence
) -> Dict[int, object]:
    """Label a copy's positional entries with the GLOBAL layers they hold.

    This is the whole trick, and it is why the layer axis is the easy one: the
    copy is a positional list whose index means "local slot", and the layout
    identity already records `start_layer`, so the global id is recoverable
    exactly. No inference, no heuristic.
    """
    ids = global_layer_ids(layer_num, start_layer)
    if len(entries) != len(ids):
        raise SeamCarryError(
            f"copy holds {len(entries)} entries but its layout claims "
            f"{len(ids)} layer(s) ({start_layer}..{start_layer + layer_num - 1}). "
            f"The label and the payload disagree, so nothing here can be trusted "
            f"to name a global layer."
        )
    return {int(g): entries[i] for i, g in enumerate(ids)}


def assemble_for(
    layer_num: int, start_layer: int, contributions: Dict[int, object]
) -> List[object]:
    """Build the DESTINATION's positional entry list from labelled contributions.

    ``contributions`` is the union of every rank's labelled copy, i.e. what a
    collective would produce. This function performs no communication and takes
    no view on how the union was formed.

    REFUSES ON ANY MISSING LAYER, by name and by number. It does not fill, does
    not skip and does not shorten the list -- see `SeamCarryError`.
    """
    want = global_layer_ids(layer_num, start_layer)
    missing = [g for g in want if int(g) not in contributions]
    if missing:
        shown = ", ".join(str(g) for g in missing[:8])
        more = "" if len(missing) <= 8 else f" (+{len(missing) - 8} more)"
        raise SeamCarryError(
            f"carry cannot complete: the destination needs global layers "
            f"{want.start}..{want.stop - 1} and {len(missing)} of them were not "
            f"contributed by any rank -- {shown}{more}. Refusing rather than "
            f"restoring a prefix that is right in some layers and stale in the "
            f"rest, which is a wrong answer with no crash."
        )
    return [contributions[int(g)] for g in want]


def carry_across(
    src_layer_num: int,
    src_start_layer: int,
    src_entries: Sequence,
    dst_layer_num: int,
    dst_start_layer: int,
    peer_contributions: Iterable[Tuple[int, int, Sequence]] = (),
) -> List[object]:
    """One rank's whole layer-axis carry, collective already performed.

    ``peer_contributions`` is ``(layer_num, start_layer, entries)`` per peer --
    what an all-gather over the seam copies would hand back. This rank's own
    copy is included automatically, so a caller that forgets to add itself to
    the gathered set still gets a correct answer rather than a refusal.

    Same-layout is NOT special-cased: a PP->PP restore lands here with one
    contribution covering exactly the destination's range and comes out
    positionally identical to the input. One path, so the cross-layout case
    cannot drift away from the case that already works.
    """
    merged: Dict[int, object] = {}
    for layer_num, start_layer, entries in peer_contributions:
        merged.update(label_contributions(layer_num, start_layer, entries))
    # This rank's own copy LAST, so it wins any overlap with a peer's. Overlap
    # is not expected between PP stages, but if two contributions ever disagree
    # the local one is the copy this request was actually taken from.
    merged.update(label_contributions(src_layer_num, src_start_layer, src_entries))
    return assemble_for(dst_layer_num, dst_start_layer, merged)


# ---------------------------------------------------------------------------
# #875d: the rank-local half, and the wiring it needs.
#
# Everything above takes integers. Everything below takes LAYOUTS -- the
# identities the pools already declare (`CpuCopyLayout`) -- because that is what
# `restore_seam_state` holds at the moment it has to decide, and re-deriving the
# geometry from the payload at that point is the mechanism that produced the
# silent version of this defect in the first place.
# ---------------------------------------------------------------------------


class LayerCarryPlan(NamedTuple):
    """WHICH source entries the destination needs, as POSITIONS in the source.

    ``take[i]`` is the index in the source's positional entry list that belongs
    in the destination's local slot i. Positions rather than ids, because the
    payload is a positional list and a plan that still spoke in global ids would
    make every payload handler re-do the lookup -- and re-doing a lookup is how
    two answers to one question appear.

    A plan EXISTS only when it is complete. There is no partial plan: the
    incomplete case raises in `plan_rank_local_carry` and never becomes an
    object anyone can act on halfway.
    """

    kind: str
    take: Tuple[int, ...]
    src_ids: Tuple[int, ...]
    dst_ids: Tuple[int, ...]


def _layout_type():
    """`CpuCopyLayout`, imported LAZILY.

    Module-level purity is worth keeping: everything above this line is
    arithmetic over plain integers and can be exercised without importing a
    5000-line pool module. The import is real and by TYPE -- duck-typing on
    field names would let any three-tuple through, and #861c's contract test
    hands this path opaque tuples on purpose.
    """
    from sglang.srt.mem_cache.memory_pool import CpuCopyLayout

    return CpuCopyLayout


def layout_global_ids(layout) -> Tuple[int, ...]:
    """The GLOBAL layer ids a declared layout holds, in local slot order.

    REFUSES anything that is not a `CpuCopyLayout`. A pool that declares its
    geometry in some other shape has not been read by this module, and the
    caller's existing refusal is the correct outcome for it -- see the header's
    "WHAT A CARRY MAY NEVER DO IS GUESS".
    """
    if not isinstance(layout, _layout_type()):
        raise SeamCarryError(
            f"this layout is a {type(layout).__name__} and not a CpuCopyLayout, "
            f"so which global layers it holds is not something this module "
            f"knows. Refusing rather than assuming a shape: the restore's "
            f"existing layout refusal is the right answer for a pool family "
            f"whose identity nobody here has read."
        )
    return layout.global_layers()


def plan_rank_local_carry(src_layout, dst_layout) -> LayerCarryPlan:
    """The carry available to ONE rank with no collective, or a refusal.

    AVAILABLE EXACTLY WHEN THE SOURCE COVERS THE DESTINATION. That is the whole
    condition, and it is why this is buildable at all: the TP->PP direction has
    a superset in hand, so the answer is a selection and not an exchange.

    THE REFUSAL IS THE OTHER DIRECTION AND IT STAYS. A PP stage's copy is
    missing layers the TP pool needs; they exist, on a peer, and only an
    all-to-all inside the cutover's no-return region would fetch them --
    `ec1717491f` measured that against the recompute it would save and returned
    DO NOT BUILD. Nothing here weakens that: this function has no group, no
    handle and no peer argument.

    NEVER PARTIAL, on the rule `SeamCarryError` states.
    """
    src_ids = layout_global_ids(src_layout)
    dst_ids = layout_global_ids(dst_layout)
    if src_layout.kind != dst_layout.kind:
        raise SeamCarryError(
            f"cannot carry a {src_layout.kind!r} copy into a "
            f"{dst_layout.kind!r} pool. `kind` separates the axes precisely so "
            f"that a KV copy can never be reconciled with a mamba one, whatever "
            f"their counts happen to be."
        )
    position: Dict[int, int] = {}
    for i, g in enumerate(src_ids):
        if g in position:
            raise SeamCarryError(
                f"the source layout names global layer {g} twice "
                f"({src_layout.describe()}). An identity that cannot say which "
                f"entry holds a layer cannot be sliced by it."
            )
        position[int(g)] = i
    missing = [int(g) for g in dst_ids if int(g) not in position]
    if missing:
        shown = ", ".join(str(g) for g in missing[:8])
        more = "" if len(missing) <= 8 else f" (+{len(missing) - 8} more)"
        raise SeamCarryError(
            f"no rank-local carry: the destination {dst_layout.describe()} needs "
            f"{len(missing)} global layer(s) the copy {src_layout.describe()} "
            f"does not hold -- {shown}{more}. Under PP those layers are on a "
            f"PEER, so completing this would need a collective inside the "
            f"cutover's no-return region, which #875 measured and refused (DO "
            f"NOT BUILD). Refusing rather than restoring a prefix that is right "
            f"in some layers and stale in the rest."
        )
    return LayerCarryPlan(
        kind=str(src_layout.kind),
        take=tuple(position[int(g)] for g in dst_ids),
        src_ids=tuple(int(g) for g in src_ids),
        dst_ids=tuple(int(g) for g in dst_ids),
    )


def _select(seq, plan: LayerCarryPlan):
    """``seq`` reduced to the destination's layers, along its FIRST axis.

    Works for a Python list (the KV and conv payloads) and for a tensor (the
    mamba temporal state, whose dim 0 is the layer axis). A contiguous run is
    taken as a SLICE so a tensor stays a view and no bytes move; the general
    case falls back to fancy indexing, which mamba's non-contiguous ids need.
    """
    take = plan.take
    if not take:
        return seq[0:0]
    first = take[0]
    if list(take) == list(range(first, first + len(take))):
        return seq[first : first + len(take)]
    if isinstance(seq, list):
        return [seq[i] for i in take]
    return seq[list(take)]


def _check_span(found: int, plan: LayerCarryPlan, what: str) -> None:
    if int(found) != len(plan.src_ids):
        raise SeamCarryError(
            f"the copy's {what} holds {int(found)} entr(ies) but the layout it "
            f"was stamped with claims {len(plan.src_ids)}. The label and the "
            f"payload disagree, so nothing here can be trusted to name a global "
            f"layer -- and slicing on a label the payload contradicts is the "
            f"wrong-layer write this carry exists to remove."
        )


def carry_payload(src_layout, dst_layout, payload):
    """Rewrite a seam copy taken under ``src_layout`` to fit ``dst_layout``.

    SHAPE-EXPLICIT, DISPATCHED ON `kind`, NEVER ON WHAT THE PAYLOAD LOOKS LIKE.
    `Req` deliberately never inspects the copy payload, and the pools own its
    shape; this is the ONE place in the tree that knows both, and it is allowed
    to know only the shapes it names below:

      "kv"      a positional per-layer `list` (`MHATokenToKVPool`,
                `MLATokenToKVPool`, `PageMajor...`, the NPU pools). DSA's
                `{"kv": ..., "index_k": ...}` dict is NOT among them and is
                refused by name -- it is not built on this rig, and a guessed
                slice of a structure nobody checked is the failure this ticket
                is about.
      "mamba"   `(conv_list, temporal)`, the pair `MambaPool.get_cpu_copy`
                returns. Both are cut on the layer axis: the conv list
                positionally, `temporal` on dim 0, which is what
                `load_cpu_copy` compares against `temporal.shape[0]`.
      hybrid    `("hybrid", kv_layout, mamba_layout)` with payload
                `(kv_cpu, mamba_cpu)`. BOTH halves are carried or the whole
                thing refuses -- a KV-only carry under a hybrid pool would
                restore attention state against stale GDN state, which is a
                wrong answer that no guard below would notice.

    THE SIBLING SWEEP, RECORDED RATHER THAN LEFT TO THE NEXT READER. Three other
    pool families declare a composite identity: `SWAKVPool` tags it `"swa"`
    (swa_memory_pool.py:286) and `UnifiedSWAKVPool` `"unified-swa"`
    (unified_memory_pool.py:1222), and both carry a DICT payload
    (`{"full", "swa", "swa_mask"}`); `DSATokenToKVPool` declares a scalar "kv"
    layout over a dict as well. None of them is carried here. That is a
    decision, not an oversight: their payload shape has not been read by this
    module, none of them is constructed on this rig (verified for the unified
    family in #879), and the refusal they keep is the behaviour they have
    today. A future carry for them belongs in this function, beside the shapes
    it already names, and will red the "refused and named" tests when it lands.

    The row axis is not touched. It is the extent contract's, settled before
    this is ever called, and a second mechanism for one question is how the two
    ends of a check drift apart.
    """
    src_hybrid = _is_hybrid(src_layout)
    dst_hybrid = _is_hybrid(dst_layout)
    if src_hybrid != dst_hybrid:
        raise SeamCarryError(
            "one side of this restore declares a composite (hybrid) layout and "
            "the other does not. That is a pool CLASS change across the seam, "
            "not a geometry change, and nothing here reconciles it."
        )
    if src_hybrid:
        if not isinstance(payload, (tuple, list)) or len(payload) != 2:
            raise SeamCarryError(
                f"a hybrid layout's copy is the pair (kv_cpu, mamba_cpu); this "
                f"payload is a {type(payload).__name__} of "
                f"{len(payload) if isinstance(payload, (tuple, list)) else 'unknown'} "
                f"part(s). Refusing rather than splitting it on a guess."
            )
        kv_cpu, mamba_cpu = payload
        carried_kv = carry_payload(src_layout[1], dst_layout[1], kv_cpu)
        if src_layout[2] is None and dst_layout[2] is None:
            if mamba_cpu is not None:
                raise SeamCarryError(
                    "the copy carries a mamba half but neither layout declares "
                    "one, so there is no identity to place it by."
                )
            return carried_kv, None
        if mamba_cpu is None:
            raise SeamCarryError(
                "both layouts declare a mamba half and the copy has none. A "
                "KV-only restore under a hybrid pool leaves the GDN state stale "
                "beneath a prefix reported as restored."
            )
        return carried_kv, carry_payload(src_layout[2], dst_layout[2], mamba_cpu)

    plan = plan_rank_local_carry(src_layout, dst_layout)
    if plan.kind == "kv":
        if not isinstance(payload, list):
            raise SeamCarryError(
                f"a kv copy is a positional per-layer list; this one is a "
                f"{type(payload).__name__}. DSA's dict payload lands here, and "
                f"is refused rather than sliced: this module has not read that "
                f"shape, and the restore's existing refusal is correct for it."
            )
        _check_span(len(payload), plan, "layer list")
        return _select(payload, plan)
    if plan.kind == "mamba":
        if not isinstance(payload, (tuple, list)) or len(payload) != 2:
            raise SeamCarryError(
                f"a mamba copy is the pair (conv_list, temporal); this payload "
                f"is a {type(payload).__name__}."
            )
        conv, temporal = payload
        _check_span(len(conv), plan, "conv list")
        _check_span(int(temporal.shape[0]), plan, "temporal state")
        return _select(conv, plan), _select(temporal, plan)
    raise SeamCarryError(
        f"this module knows no payload shape for a {plan.kind!r} layout, so it "
        f"refuses rather than assuming one."
    )


def _is_hybrid(layout) -> bool:
    """`HybridLinearKVPool`'s composite identity, recognised by its own tag.

    A plain tuple whose first element is not the literal the pool writes is NOT
    a hybrid layout -- #861c's contract test passes `("kv", 18, 32)` through
    this path deliberately, and it must keep landing on the refusal.
    """
    return (
        isinstance(layout, tuple)
        and not isinstance(layout, _layout_type())
        and len(layout) == 3
        and layout[0] == "hybrid"
    )
