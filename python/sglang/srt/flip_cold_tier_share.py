# SPDX-License-Identifier: Apache-2.0
"""Turning the shared cold tier ON for a two-GROUP flip -- and the two things
that stop it being one pool even when the switch is on.

PRIOR-ART GATE FIRST (Memory ``PRIOR-ART-GATE``). The shared page-locked
expert store is NOT a build task. It exists:

  * ``layers/moe/shared_pinned.py`` -- ``shared_pinned_empty(path, shape,
    dtype)`` (:71) maps a tmpfs file ``MAP_SHARED``, sizes it exactly and
    page-locks it per reader with ``cudaHostRegister`` (:91). Its module
    docstring names Task #47 and the 88 GiB mark verbatim.
  * ``layers/moe/cold_tier_shm.py`` -- ``ColdTierLayout``,
    ``create_owned_segment``, ``publish_manifest``, ``read_peer_manifest``,
    ``attach_peer_segment``, ``register_for_dma``: the whole peer protocol.
  * ``layers/moe/cold_tier_fetch.py`` -- ``ColdTierOwner.allocate_spill_pool``
    (:358), switch ``cold_tier_enabled()`` on ``SGLANG_MOE_COLD_TIER_SHM``
    (:135-146).
  * ``layers/moe/expert_offload.py:1539-1565`` -- with a ``cold_tier``, "the
    storage IS the shared segment, not a copy into it"; without one it falls
    back to the private ``pinned_exact_empty`` (:2614).

So slice 3 is an ENABLE-and-COUNT task. What this module adds is the two
things that make ``SGLANG_MOE_COLD_TIER_SHM=1`` in both arms NOT sufficient:

GAP 1 -- THE INSTANCE ID IS MINTED PER SERVER PROCESS, SO TWO GROUPS GET TWO
POOLS. ``publish_cold_tier_instance()`` (``cold_tier_fetch.py:149-165``) is
called from ``_launch_subprocesses`` (``entrypoints/engine.py:683``) and mints
``uuid4().hex[:16]`` unless the variable is already set. A flip starts TWO
``launch_server`` processes, so each mints its own id. Segment names are
``sgl-cold-<instance>-r<rank>-...`` (``cold_tier_shm.py:213-217``) and the
header validator REFUSES a segment belonging to another instance (:320-326).
Two groups would therefore not share -- they would each build a full private
pool, the ledger of slice 2 would print ``shared=true`` on both, and the
88 GiB mark would be blown by exactly the 29.77 GiB the sharing was supposed
to save. The fix is one line of launcher discipline and needs no code change
in ``cold_tier_fetch``: that function already honours a hand-set value on
purpose. :func:`build_cold_tier_group_env` is that discipline, made testable.

GAP 2 -- THE TWO LAYOUTS PARTITION THE EXPERT SET DIFFERENTLY, SO "MAX PER
RANK SLOT" IS NOT AUTOMATICALLY THE RIGHT ARITHMETIC. PP3 cuts by LAYER
(stages hold 29/11/8 MoE layers, every expert of their own layers); Form A
cuts by EXPERT INDEX (``--rank-moe-ratio 183,149,180`` across all 48 layers).
Rank 0 of P and rank 0 of D therefore do NOT hold the same rows. The design's
40.97 GiB = ``max(22.73;20.62) + max(5.30;7.36) + max(3.85;10.88)`` is only
true if a rank slot's two shards are the SAME bytes; with orthogonal cuts the
honest figure is the union of the two partitions, which is bounded below by
the max and above by the sum. :func:`check_shard_alignment` refuses the
unexamined case by name instead of letting 40.97 stand as if measured.

TEARDOWN, Memory ``SHM-RESIDUE-NUR-PER-HALTER``: segments are never removed
by name pattern. ``shared_pinned.unlink_store(dir)`` deletes every file of a
directory and is therefore correct ONLY for the process that owns the epoch.
:class:`ColdTierHolder` and :func:`may_unlink` make that owner explicit, so a
sweep by a process that is not the holder is a refusal rather than a
data race with a live peer.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Dict, FrozenSet, Iterable, Mapping, Optional, Sequence, Tuple

from sglang.srt.flip_nextflash_plan import FlipInfeasible

__all__ = [
    "COLD_TIER_SWITCH_ENV",
    "COLD_TIER_INSTANCE_ENV",
    "Weg2FlipColdTierSplit",
    "build_cold_tier_group_env",
    "assert_one_instance",
    "ShardMap",
    "check_shard_alignment",
    "ColdTierHolder",
    "may_unlink",
]

COLD_TIER_SWITCH_ENV = "SGLANG_MOE_COLD_TIER_SHM"
#: Kept in step with ``cold_tier_fetch.COLD_TIER_INSTANCE_ENV`` by
#: :func:`test_the_instance_env_name_matches_cold_tier_fetch`; NOT imported,
#: so this module stays desk-pure (``cold_tier_fetch`` pulls ``environ``).
COLD_TIER_INSTANCE_ENV = "SGLANG_MOE_COLD_TIER_INSTANCE"


class Weg2FlipColdTierSplit(FlipInfeasible):
    """W117 -- the shared cold tier is switched on but is not ONE pool.

    Three ways for that to happen, all of which read as "shared" from
    inside a single process and only show up as a blown host mark:
    the two groups carry different instance ids; one group carries the
    switch and the other does not; the two groups' rank shards are
    different expert sets, so a rank slot holds two pools, not one.
    """


# ==========================================================================
# Gap 1: ONE instance id, minted once, in BOTH groups
# ==========================================================================
def build_cold_tier_group_env(
    instance_id: str,
    groups: Sequence[str] = ("P", "D"),
    enabled: bool = True,
) -> Dict[str, Dict[str, str]]:
    """The cold-tier env fragment for EVERY flip group, from one id.

    The launcher mints ``instance_id`` ONCE, before either group is spawned,
    and hands the same value to both. ``publish_cold_tier_instance()`` then
    finds it already set and leaves it alone -- which is precisely the
    behaviour its docstring promises ("A hand-set value is never overwritten,
    which is what lets an operator re-attach to a segment set on purpose").

    ``enabled=False`` returns the switch explicitly OFF for every group, not
    absent: a flip that runs private pools on purpose must say so, so the
    slice-2 ledger prints ``shared=false`` on all six ranks rather than on
    the ones that happened to miss an export.
    """
    ident = (instance_id or "").strip()
    if enabled and not ident:
        raise Weg2FlipColdTierSplit(
            f"W117 Weg2FlipColdTierSplit -- {COLD_TIER_SWITCH_ENV}=1 with no "
            f"instance id. Left empty, every server process mints its own "
            f"uuid4 in publish_cold_tier_instance() "
            f"(cold_tier_fetch.py:149-165, called from engine.py:683), so the "
            f"two groups name different segments "
            f"(sgl-cold-<instance>-r<rank>-..., cold_tier_shm.py:213-217) and "
            f"build two full private pools while both report shared=true."
        )
    if not groups:
        raise Weg2FlipColdTierSplit(
            "W117 Weg2FlipColdTierSplit -- no groups to share between"
        )
    value = "1" if enabled else "0"
    out: Dict[str, Dict[str, str]] = {}
    for g in groups:
        frag = {COLD_TIER_SWITCH_ENV: value}
        if enabled:
            frag[COLD_TIER_INSTANCE_ENV] = ident
        out[g] = frag
    return out


def assert_one_instance(group_envs: Mapping[str, Mapping[str, str]]) -> str:
    """Read the groups' environments back and prove they name ONE pool.

    This is the gate a launcher runs on the environments it is about to
    spawn with -- after every other env builder has had its turn, because
    the bug this catches is an env that was correct when it was built and
    was overwritten afterwards.
    """
    if not group_envs:
        raise Weg2FlipColdTierSplit(
            "W117 Weg2FlipColdTierSplit -- no group environments to check"
        )
    on = {
        g: e.get(COLD_TIER_SWITCH_ENV, "") in ("1", "true", "True")
        for g, e in group_envs.items()
    }
    if len(set(on.values())) > 1:
        yes = sorted(g for g, v in on.items() if v)
        no = sorted(g for g, v in on.items() if not v)
        raise Weg2FlipColdTierSplit(
            f"W117 Weg2FlipColdTierSplit -- group(s) {yes} have "
            f"{COLD_TIER_SWITCH_ENV} on and {no} do not. A half-shared rig "
            f"double-counts exactly the group that did not attach, and its "
            f"host total is not interpretable "
            f"(flip_host_ledger.ledger_from_lines refuses it for the same "
            f"reason)."
        )
    if not any(on.values()):
        return ""

    ids = {g: (e.get(COLD_TIER_INSTANCE_ENV, "") or "").strip() for g, e in group_envs.items()}
    missing = sorted(g for g, v in ids.items() if not v)
    if missing:
        raise Weg2FlipColdTierSplit(
            f"W117 Weg2FlipColdTierSplit -- group(s) {missing} carry "
            f"{COLD_TIER_SWITCH_ENV}=1 with no {COLD_TIER_INSTANCE_ENV}. Each "
            f"will mint its own id at engine.py:683 and name segments no peer "
            f"can find."
        )
    distinct = sorted(set(ids.values()))
    if len(distinct) > 1:
        detail = ", ".join(f"{g}={v!r}" for g, v in sorted(ids.items()))
        raise Weg2FlipColdTierSplit(
            f"W117 Weg2FlipColdTierSplit -- the groups name {len(distinct)} "
            f"different cold-tier instances ({detail}). Segment names are keyed "
            f"by the instance (cold_tier_shm.py:213-217) and the header "
            f"validator REFUSES a segment of another instance (:320-326), so "
            f"these are two pools, not one -- the exact 29.77 GiB the sharing "
            f"was supposed to save. Mint the id ONCE in the launcher, before "
            f"either group is spawned, and hand the same value to both."
        )
    return distinct[0]


# ==========================================================================
# Gap 2: do the two layouts' rank shards hold the SAME rows?
# ==========================================================================
@dataclass(frozen=True)
class ShardMap:
    """Which cold rows one rank of one layout owns.

    ``rows`` is the set of ``(layer_id, expert_id)`` pairs whose cold bytes
    this rank spills to the host pool. A layout is a mapping rank -> ShardMap.
    """

    layout: str
    rank: int
    rows: FrozenSet[Tuple[int, int]]

    @staticmethod
    def of(layout: str, rank: int, rows: Iterable[Tuple[int, int]]) -> "ShardMap":
        return ShardMap(layout=layout, rank=rank, rows=frozenset(map(tuple, rows)))


def check_shard_alignment(
    left: Mapping[int, ShardMap],
    right: Mapping[int, ShardMap],
) -> None:
    """Refuse the case where "one pool, two readers" is not what happens.

    The design's saving -- ``max(22.73;20.62) + max(5.30;7.36) +
    max(3.85;10.88) = 40.97 GiB`` -- is the MAXIMUM per rank slot. That is
    the right arithmetic only when rank r of the two layouts owns the SAME
    rows: then one segment serves both and its size is the larger of the two.

    PP3 cuts the MoE mass by LAYER (29/11/8 stages), Form A cuts it by
    EXPERT INDEX (``--rank-moe-ratio 183,149,180`` over all 48 layers). Under
    orthogonal cuts a rank slot holds two DIFFERENT row sets; the honest
    figure is then the union, bounded below by the max and above by the sum,
    and 40.97 GiB is a number nobody measured.

    This refuses rather than computes the union, because the union's SIZE
    needs the per-row byte count of both layouts -- a metal measurement
    (design §Scheibenplan, slice 3 probe), not a desk one.
    """
    ranks = sorted(set(left) | set(right))
    if not ranks:
        raise Weg2FlipColdTierSplit(
            "W117 Weg2FlipColdTierSplit -- no shards to align"
        )
    missing = [r for r in ranks if r not in left or r not in right]
    if missing:
        raise Weg2FlipColdTierSplit(
            f"W117 Weg2FlipColdTierSplit -- rank slot(s) {missing} exist in one "
            f"layout only. A rank slot with one reader is not shared; its pool "
            f"is that reader's alone and belongs in the ledger as a SUM term, "
            f"not a MAX term."
        )
    bad = []
    for r in ranks:
        lrows, rrows = left[r].rows, right[r].rows
        if lrows != rrows:
            bad.append(
                (r, left[r].layout, len(lrows), right[r].layout, len(rrows),
                 len(lrows & rrows), len(lrows | rrows))
            )
    if bad:
        detail = "; ".join(
            f"rank {r}: {ln} owns {lc} rows, {rn} owns {rc}, "
            f"{inter} shared, union {uni}"
            for r, ln, lc, rn, rc, inter, uni in bad
        )
        raise Weg2FlipColdTierSplit(
            f"W117 Weg2FlipColdTierSplit -- {len(bad)} rank slot(s) hold "
            f"DIFFERENT cold rows in the two layouts ({detail}). Sharing a "
            f"segment per rank slot then stores the UNION, not the larger of "
            f"the two: PP3 cuts the MoE mass by LAYER (29/11/8 stages), Form A "
            f"by EXPERT INDEX (--rank-moe-ratio), and the two cuts are "
            f"orthogonal. The design's 40.97 GiB is max-per-slot arithmetic "
            f"and is valid only under identical shards -- under these it is a "
            f"LOWER bound, with the 70.74 GiB sum as the upper one. Measure the "
            f"union on metal before quoting either."
        )


# ==========================================================================
# Teardown by holder, never by pattern (Memory SHM-RESIDUE-NUR-PER-HALTER)
# ==========================================================================
@dataclass(frozen=True)
class ColdTierHolder:
    """Who may remove a cold-tier segment directory.

    ``instance_id`` is the epoch; ``pid`` is the process that minted it. Both
    are required: a pid alone is reused by the OS, an epoch alone does not say
    who is responsible for it.
    """

    instance_id: str
    pid: int


def may_unlink(
    holder: ColdTierHolder,
    caller_pid: int,
    caller_instance: str,
    live_peer_pids: Sequence[int] = (),
) -> bool:
    """May THIS caller run ``shared_pinned.unlink_store`` on the epoch?

    Three conditions, all of them the same rule from a different side:
    the caller is the holder's pid; the caller is in the holder's epoch; no
    peer is still alive on the segment. Anything else is refused by name,
    because ``unlink_store`` removes EVERY file of the directory it is given
    and a live peer's mapping would keep reading unlinked pages that no new
    reader can find -- a residue bug that looks like a fetch miss.

    Never call this with a name PATTERN. The epoch is the key.
    """
    if not holder.instance_id:
        raise Weg2FlipColdTierSplit(
            "W117 Weg2FlipColdTierSplit -- an unlink without an epoch is a "
            "pattern sweep by another name (Memory SHM-RESIDUE-NUR-PER-HALTER). "
            "Name the instance id whose segments are being removed."
        )
    if (caller_instance or "").strip() != holder.instance_id:
        raise Weg2FlipColdTierSplit(
            f"W117 Weg2FlipColdTierSplit -- pid {caller_pid} is in epoch "
            f"{caller_instance!r} and tried to remove epoch "
            f"{holder.instance_id!r}. A segment is removed by ITS holder or by "
            f"nobody; the other epoch's live readers would lose their backing "
            f"pages and report a fetch miss instead of a teardown."
        )
    if caller_pid != holder.pid:
        raise Weg2FlipColdTierSplit(
            f"W117 Weg2FlipColdTierSplit -- pid {caller_pid} is not the holder "
            f"of epoch {holder.instance_id!r} (pid {holder.pid}). The holder "
            f"tears its own epoch down; a peer that does it for the holder is "
            f"racing the holder's own readers."
        )
    alive = [p for p in live_peer_pids if p != caller_pid]
    if alive:
        raise Weg2FlipColdTierSplit(
            f"W117 Weg2FlipColdTierSplit -- epoch {holder.instance_id!r} still "
            f"has live peer(s) {sorted(alive)}. Unlinking now leaves them "
            f"reading pages no new reader can find."
        )
    return True
