"""HY: a park backup the full L2 refuses takes the L3-copied pages of a held request.

Metal (NF, D group, Form A token cut):

* y3w e033a931db 01:39:13: ``park_running`` retracted weg2-1-4 (121920 tokens)
  with a forced host write-through; its claim of 1905 pages came back
  ``#1427 ARENA-CLAIM REFUSED`` -- 18 slots short (``ARENA-DROP need=18``).
  The arena was full of the pages of weg2-14-25 (a 246k-token request held on
  D, not running: ``handoff_kept=3504/3841``, its read had just completed and
  pinned its span, #1417). ``#1421 BACKUP-REFUSED why=arena_claim``; the sleep
  dropped the device KV, the wake could not resume the park (W50
  x_refusal_midstream) and P re-prefilled 125145 tokens with cached=0: 30.6 s.
* y3u 5bedac26f1 00:36:46: the same shape, need=469, the hold of weg2-14-26
  pinned 3844 pages; P re-prefilled 130764 tokens: 32.0 s.

The held request is not running before the sleep -- its pages are read again
at the wake anyway. Where every page of its held span has an L3 copy, giving
the span back costs a store read at the wake (fractions of a second) instead
of a 30-second prefill of the parked one: read, don't recompute.

The rule, per park (every TP rank of group D runs ``park_running`` in the same
pass, over the same replicated lists):

1. The tree records which retained nodes' backups the arena refused
   (``#1421 arena_claim``) while the park retracts.
2. ONE group vote (MIN over the attention TP group, #59b's replicated-list
   pattern): refused anywhere, the pages still missing (``need``, group MAX),
   and per held candidate (the park list, minus what this park retracted,
   in its replicated order) the held pages with an L3 copy and whether its
   WHOLE held span has one (group MIN).
3. ``need`` <= the on-disk pages of the eligible candidates: every rank gives
   back the same candidates' spans (unpin, drop the records, evict the host
   leaves -- the tree edit is the same on every rank because the span and the
   verdict are), a barrier, then the refused backups again. Otherwise nothing
   moves -- the refusal stands as before, named.

Marker: ``PARK-BACKUP REFUSED need= held_by_hold= held_on_disk= verdict=``.
Switch ``SGLANG_WEG2_ENABLE_PARK_HOLD_YIELD`` (default on).
"""
from __future__ import annotations

import logging
from typing import Dict, List, Optional, Sequence, Tuple

import torch

from sglang.srt.environ import envs

logger = logging.getLogger(__name__)

MARK = "PARK-BACKUP REFUSED"


def begin(tree) -> None:
    """Start recording the arena-refused backups of this park's retraction."""
    if tree is not None and hasattr(tree, "_weg2_park_track"):
        tree._weg2_park_track = {}


def _end(tree) -> Dict[int, object]:
    track = tree._weg2_park_track or {}
    tree._weg2_park_track = None
    return track


def pack_vote(*, refused: bool, need: int, on_disk: Sequence[int], whole: Sequence[bool]) -> List[int]:
    """One MIN-reducible vector: -refused, -need, then (on_disk, whole) per
    candidate. After MIN: refused = any rank refused, need = the group MAX,
    on_disk / whole = the group MIN."""
    out = [-int(bool(refused)), -int(need)]
    for n, w in zip(on_disk, whole):
        out += [int(n), int(bool(w))]
    return out


def unpack_vote(vec: Sequence[int]) -> Tuple[bool, int, List[int], List[bool]]:
    refused = -int(vec[0]) > 0
    need = -int(vec[1])
    rest = list(vec[2:])
    return refused, need, [int(x) for x in rest[0::2]], [bool(int(x)) for x in rest[1::2]]


def choose(*, need: int, on_disk: Sequence[int], whole: Sequence[bool]) -> Optional[List[int]]:
    """Indices of the candidates to give back, in the replicated order, until
    their on-disk pages cover ``need``; only candidates whose whole held span
    has an L3 copy qualify. None = they do not cover it (nothing moves)."""
    picked, got = [], 0
    for i, (n, w) in enumerate(zip(on_disk, whole)):
        if got >= need:
            break
        if w and n > 0:
            picked.append(i)
            got += int(n)
    return picked if got >= need and need > 0 else None


# ---------------------------------------------------------------------------
# rank-local facts
# ---------------------------------------------------------------------------

def _kv_pool(tree):
    from sglang.srt.mem_cache.unified_cache_components.tree_component import ComponentType

    return tree._weg2_arena_pools().get(ComponentType.FULL)


def _span_nodes(tree, rid: str) -> list:
    """The host nodes a held request's completed read pinned (#1417), deepest first."""
    pins = getattr(tree, "_prefetch_span_pins", None) or {}
    return [node for node, _params in pins.get(str(rid), ())]


def _node_slots(pool, node) -> List[int]:
    from sglang.srt.mem_cache.pool_host.arena_pool import arena_ref_slots
    from sglang.srt.mem_cache.unified_cache_components.tree_component import ComponentType

    hv = node.component_data[ComponentType.FULL].host_value
    return arena_ref_slots(pool, hv).tolist() if hv is not None else []


def held_facts(tree, rid: str) -> Tuple[int, int]:
    """(held pages, of them with an L3 copy) of a held request's pinned span."""
    pool = _kv_pool(tree)
    backend = getattr(getattr(tree, "cache_controller", None), "storage_backend", None)
    if pool is None or backend is None:
        return 0, 0
    slots = [s for node in _span_nodes(tree, rid) for s in _node_slots(pool, node)]
    if not slots:
        return 0, 0
    stems = [pool.arena.slot_stem(s) for s in slots]
    named = [st for st in stems if st]
    on_disk = backend._stat_stems(named) if named else {}
    return len(slots), sum(1 for st in stems if st and st in on_disk)


def need_pages(tree, refused: Dict[int, object]) -> int:
    """Pages the refused backups still lack: not COMPLETE in the arena, minus
    the free slots (at least 1 when something was refused)."""
    pool = _kv_pool(tree)
    if not refused or pool is None:
        return 0
    arena = pool.arena
    from sglang.srt.managers.cache_controller import weg2_suffixed_stems

    missing = 0
    for node in refused.values():
        hv = getattr(node, "hash_value", None) or []
        if not hv:
            continue
        stems = weg2_suffixed_stems(tree.cache_controller.storage_backend, list(hv))
        missing += sum(1 for s in arena.find_states(stems) if int(s) != 2)
    st = arena.stats()
    free = max(0, int(st["slots"]) - int(st["complete"]) - int(st["claimed"]))
    return max(1, missing - free)


def give_back(tree, rid: str) -> int:
    """Unpin a held request's span, drop its read records and evict the host
    leaves of the span (deepest first). Returns the pages given back."""
    from sglang.srt.mem_cache.unified_cache_components.tree_component import ComponentType

    nodes = _span_nodes(tree, rid)
    pool = _kv_pool(tree)
    pages = sum(len(_node_slots(pool, n)) for n in nodes) if pool is not None else 0
    tree._unpin_prefetched_span(str(rid))
    tree.prefetch_loaded_tokens_by_reqid.pop(str(rid), None)
    tree._prefetch_completed_tokens.pop(str(rid), None)
    (getattr(tree, "_weg2_dormant_done", None) or {}).pop(str(rid), None)
    tracker = {ct: 0 for ct in ComponentType}
    for node in nodes:  # deepest first: each eviction makes its parent the leaf
        if tree._is_host_leaf(node):
            tree._evict_host_leaf(node, tracker)
    return pages


# ---------------------------------------------------------------------------
# the park step
# ---------------------------------------------------------------------------

def settle(sched, *, retracted: Sequence, parked: Sequence) -> Optional[str]:
    """After the park's retraction: the group vote, and -- when the L3-copied
    held pages cover what the refused backups lack -- the give-back and the
    backups again. Returns the verdict (for the caller's line), None when this
    tree has no arena."""
    tree = getattr(sched, "tree_cache", None)
    if tree is None or not hasattr(tree, "_weg2_park_track") or _kv_pool(tree) is None:
        return None
    refused = _end(tree)
    if not envs.SGLANG_WEG2_ENABLE_PARK_HOLD_YIELD.get():
        return None
    gone = {str(r.rid) for r in retracted}
    cands = [str(r.rid) for r in parked if str(r.rid) not in gone]
    facts = [held_facts(tree, rid) for rid in cands]
    vec = pack_vote(refused=bool(refused), need=need_pages(tree, refused),
                    on_disk=[d for _h, d in facts], whole=[h > 0 and d == h for h, d in facts])
    t = torch.tensor(vec, dtype=torch.int64)
    tree._all_reduce_attn_groups(t, torch.distributed.ReduceOp.MIN, label="park_hold_yield")
    any_refused, need, on_disk, whole = unpack_vote(t.tolist())
    if not any_refused:
        return "clean"
    held = sum(h for h, _d in facts)
    pick = choose(need=need, on_disk=on_disk, whole=whole)
    verdict = "stands" if pick is None else "yield"
    given, retried = 0, 0
    if pick is not None:
        for i in pick:
            given += give_back(tree, cands[i])
            setattr(next(r for r in parked if str(r.rid) == cands[i]), "_weg2_hold_yielded", True)
        tree._barrier_attn_groups(label="park_hold_yield")
        for node in sorted(refused.values(), key=lambda n: tree.weg2_node_depth(n)):
            if not node.backuped:
                retried += int(tree.write_backup(node) > 0)
    logger.warning(
        "%s need=%d held_by_hold=%d held_on_disk=%d refused_nodes=%d verdict=%s given_back=%s "
        "pages=%d backups_retried=%d/%d (%s)",
        MARK, need, held, sum(on_disk), len(refused), verdict,
        [cands[i] for i in pick] if pick else [], given, retried, len(refused),
        "the held requests' L3-copied pages made room; the wake reads them from L3"
        if pick is not None else
        "the L3-copied held pages do not cover the need -- the refusal stands (P recomputes the park)")
    return verdict
