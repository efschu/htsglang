"""#875: the layer axis of a layout-neutral seam carry.

WHAT THIS IS FOR. Across a phase flip the seam copy is taken in one geometry and
restored in another, and `restore_seam_state` currently REFUSES that case. The
refusal is correct -- it is the only thing between the tree and either the W40
IndexError or a silent wrong-layer write -- but it is not the answer: its own
counter comment says a layout refusal means "every flip loses its prefixes".
This module is the first of the three axes that answer has to cross.

THE REFUSAL'S STATED REASON WAS WRONG ON ONE LEG, and that is why this module
exists at all. `check_cpu_copy_layers` said:

    "A remap would be correct only if the copy covered every layer the
     destination needs, and rank-locally under PP it cannot [...] The KV head
     sharding also differs between the phases (PP holds all heads of its stage,
     TP a head shard of every layer), so even the overlapping entries are not
     interchangeable."

The second sentence is FALSE on this rig, checked in the code rather than
assumed. Under `#345` uneven-DCP KV replication the full-attention cache is
TOKEN-sharded and "every rank stores the FULL, replicated kv-heads"
(model_runner_kv_cache_mixin.py:3155-3160), and the boot line at :3228 prints
`get_total_num_kv_heads()` verbatim. The PP-phase pool takes
`get_num_kv_heads(attn_tp_size)` with `attn_tp_size == 1`, i.e. the same total.
So a (layer, token) KV row has IDENTICAL width in both phases and the entries
ARE interchangeable.

The first sentence is true only RANK-LOCALLY, which is not the same as true. PP
rank r holds its stage's layers for ALL tokens; the union over the three ranks
is every layer. The data the destination needs exists -- on a peer. So the
correct reading is not "a remap is impossible" but "a rank-local remap is
impossible, and nobody asked whether a collective one was available".

THREE AXES, AND ONLY ONE OF THEM IS HERE:

  1. LAYER   PP is stage-sharded (8/4/4 attention layers on this rig, from
             `pp_attn_stage_ratio`), TP is complete (16). This module. The
             remap is exact once entries are labelled by GLOBAL layer id, which
             is what `CpuCopyLayout.start_layer` already records.
  2. HEAD    replicated in both phases. NO remap needed -- see above.
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

NOTHING IN THIS MODULE IS WIRED INTO THE RESTORE PATH YET. The refusal stands
until axis 3 is answered and the exchange has a home; wiring a layer-correct,
token-wrong carry would produce exactly the "matching row ids and mismatched
widths" corruption #719 already walked into once.
"""

from __future__ import annotations

from typing import Dict, Iterable, List, Sequence, Tuple


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
