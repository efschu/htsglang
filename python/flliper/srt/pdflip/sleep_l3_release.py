"""S86 SLEEP-L3-RELEASE: before the sleep flush, D's tree gives back the arena references of pages L3 holds.

THE METAL (rc19 boot ..._10e9b2b492_1019_230333, D group TP=3 Form A, sleep 05:37:22-23). Five running requests
were parked (``PDFLIP-D-PARK park_running ... parked=[pdflip-112-2715, -2708, -2712, -2717, -2718]``); the last
census before the sleep read ``ARENA-REF-CENSUS ... complete=6467 pinned=6409 refs=18944`` with
``ARENA-REF-HOLDERS ... tree=6409 queue=0`` on TP1 (TP0 tree=6126 queue=0): no queue, no prefetch, no carrier --
the TREES' references alone (S83's queue class is 0 here). Every claim of the park-end chains' write-throughs found
no unreferenced slot (``#1427 ARENA-DROP ... freed=0 stages=i:0,ii:0,iii:0``, dozens per rank), was refused
(``#1421 BACKUP-REFUSED why=arena_claim``, then ``parent_unbacked`` down the chains), and
``#1470 FLUSH-PUBLISH ARENA-FULL-STOP round=1 unbacked=19 arena_refused=19`` ended TP0's loop. The reset then
dropped them: ``#1470 FLUSH-UNBACKED site=flush nodes=17 tokens=13632``, among them the park-end nodes 6498
(pdflip-112-2708, depth 40960), 6495 (-2717, depth 93952) and 6497 (-2712, depth 56576).

Those 6409 slots were NOT the only copy of their pages: the whole boot dropped arena slots with an L3 copy
(``l3=on_disk``) and wrote 164 pages in total (``#1427 ... total written=164 dropped_without_l3=0``) -- the store
writer keeps L2 and L3 in step. And the reset gives every one of those references back a few milliseconds later
(05:37:26 ``complete=6436 pinned=531``, the parks' own references). The arena was full of references the sleep
was about to drop anyway, while the one chain that had no copy anywhere was refused room.

WHY THE EXISTING SPILL DID NOT HELP. W3-ARENA (``UnifiedRadixCache._w3_arena_spill``) is built for a full arena of
tree references, but on Form A only TP0 decides: a worker's refused claim leaves its need for TP0's NEXT
broadcast (``R12 W3-SPILL-REQUEST``), TP0 releases its own reference and records SPILL for the workers. Inside the
sleep's flush there is no broadcast -- the #1470 loop posts no collective -- and the reset clears the R12 ledger
(``form_a_host_shadow.on_tree_reset``): a slot is free only when EVERY rank let go, so the workers' references
kept every slot until the reset (``W3-ARENA FORMA-REQUEST need=53/64 released_pages=0`` at 05:37:22-23).

THE RULE (S86). At the sleep leg, after the group's HiCache drain and before the sleep flush
(``WeightUpdater._pdflip_sleep_flush`` on group D), every rank of the group runs this once, at the same point:

1. veto: each rank marks every arena slot it must keep -- a slot of a node that is not device-resident (a host-only
   node's host copy is its last tier but L3 -- not this step), has a write in flight, a host-locked load, an aux
   state only on the host, staging rows, no page hashes, or a page without a proven L3 copy
   (``secure_rows_to_l3`` + ``read_room.l3_complete``: every page COMPLETE in the shared header and on disk or
   written now; a CLAIMED slot, a lost or unpaired page keeps the node). A page without an L3 copy is NEVER
   released (#257).
2. agreement (group MAX of the veto vector = MIN of the consents, ``hicache_group_max``): a slot is released only
   when NO rank vetoes it -- a slot frees only once every rank let go, so every rank lets go of the same slots.
3. release: each rank gives back the host rows of its device-resident nodes all of whose slots were agreed -- the
   W3 transit release (``_evict_component_and_detach_lru(..., HOST)``: the node stays on the device and in the
   tree, ``l3_present``); TP0 records SPILL so its shadow of the workers stays true should the sleep not reset.
4. digest (group MIN/MAX of the released slot count and its crc32): the barrier -- no rank's #1470 sweep claims
   before every rank let go -- and the unity check: ``DIVERGED`` (or a rank's error after the first collective,
   carried in the digest) stops the group by name, ``W86 SleepL3ReleaseDivergence`` (review A1, RAENGE-NIE-UNEINS).

The #1470 loop's claims then find the slots unreferenced, the clock takes them (stage i, ``on_disk``: no write)
and the park-end chain is written before the reset. No reserve, nothing kept back: what is released is what the
reset would release a moment later, minus every page that has no other copy.

Exactly three collectives per call once the gate passes, their number independent of every value (the third and
second are skipped together only when the group-reduced slot count is 0). The gate (switch, group D, the kv tag of
the release RPC) is rank-uniform; a veto error, a rank without pool/arena or with another slot count vetoes
everything (fail closed, review A3b) and still posts every collective; unpaired pages are never secured here,
whatever ``FLLIPER_PDFLIP_L3_PRESENT_STRICT`` says (A3a).

Switch ``FLLIPER_PDFLIP_SLEEP_L3_RELEASE`` (default on).
"""
from __future__ import annotations

import logging
import time
import zlib
from typing import Any, Callable, Dict, List, Optional

import torch

logger = logging.getLogger(__name__)

MARK = "S86 SLEEP-L3-RELEASE"


class SleepL3ReleaseDivergence(RuntimeError):
    """W86 (review S86 A1, RAENGE-NIE-UNEINS, user 29.08.: "ränge dürfen sich niemals uneins sein. wenn uneins crash
    stop."): the ranks let go of different slots, or a rank failed after the first collective -- the group stops by
    name in the same pass (every rank reads the same reduced digest)."""

_N = {"calls": 0}


def enabled() -> bool:
    try:
        from flliper.srt.environ import envs

        return bool(envs.FLLIPER_PDFLIP_SLEEP_L3_RELEASE.get())
    except Exception:  # noqa: BLE001 - a hermetic tree without the field: the default (on)
        return True


def _arena_pool(pool: Any) -> Any:
    """S86b (rc22 boot ..._9c40fe40a2_1020_094747, D 10:11:49Z, 6x): D's direct pool is the hybrid
    ``HostPoolGroup`` (KV + Mamba). It forwards the arena calls (alloc_write, complete_write, ...) to its anchor
    pool, the KV ``ArenaMHAHostPool``, but not ``secure_rows_to_l3`` -- ``__getattr__`` raised, every veto failed
    closed (``vetoed_by_group=6485/6485 group_released=0``): S86 never released anything on D. The pool this step
    works on is therefore the one that HOLDS the arena: the pool itself when it carries the call, else the group's
    anchor pool (the FULL host rows are in its id space -- the group forwards resolve_rows/is_arena_id/staging_rows
    to it). None = no such pool (the full veto, fail closed). Not added to the group's forward list: W3 reads
    ``hasattr(pool, "secure_rows_to_l3")`` and would change behind its own switch
    (FLLIPER_PDFLIP_ENABLE_W3_SPILL_ANCHOR_POOL)."""
    if pool is None:
        return None
    try:
        if callable(getattr(pool, "secure_rows_to_l3", None)):
            return pool
    except Exception:  # noqa: BLE001 - a forwarding pool that raises on the name: try its anchor
        pass
    host = getattr(getattr(pool, "anchor_entry", None), "host_pool", None)
    if host is not None and callable(getattr(host, "secure_rows_to_l3", None)):
        return host
    return None


def _slots(pool: Any, host_value: Any) -> List[int]:
    from flliper.srt.mem_cache.pool_host.arena_pool import arena_ref_slots

    return [int(s) for s in arena_ref_slots(pool, host_value).tolist()]


def _reason(tree: Any, node: Any, pool: Any) -> Optional[str]:
    """None = a device-resident node whose host copy may leave L2 if its pages have an L3 copy (checked by the
    caller); else the name of the first check that keeps it. Same guards as the W3 device round, without the
    in-phase ones that protect a read still to come (claimer chain, told head, keep-payable): at the sleep the
    reset drops those references too."""
    from flliper.srt.mem_cache.unified_radix_cache import BASE_COMPONENT_TYPE, _aux_components

    cd = node.component_data[BASE_COMPONENT_TYPE]
    if cd.value is None:
        return "host_only"
    if node.id in (getattr(tree, "ongoing_write_through", None) or {}):
        return "write_pending"
    if getattr(node, "write_through_pending_id", None) is not None:
        return "write_pending"
    if any(int(getattr(c, "host_lock_ref", 0) or 0) > 0 for c in node.component_data):
        return "host_locked"
    aux = _aux_components(tree)
    if any(node.component_data[c.component_type].host_value is not None
           and node.component_data[c.component_type].value is None for c in aux):
        return "aux_host_only"
    hv = cd.host_value
    if int(hv.min()) < int(getattr(pool, "staging_rows", 0)):
        return "staging"
    if not getattr(node, "hash_value", None):
        return "no_hash"
    return None


def _l3_proven(pool: Any, node: Any) -> bool:
    from flliper.srt.mem_cache.unified_radix_cache import BASE_COMPONENT_TYPE
    from flliper.srt.pdflip.read_room import l3_complete

    sec = pool.secure_rows_to_l3(node.component_data[BASE_COMPONENT_TYPE].host_value)
    if int(sec.get("unpaired", 0) or 0):
        # review S86 A3a: an unpaired page (KV held back for want of its QSA index) is NOT on disk -- never secured
        # here, whatever FLLIPER_PDFLIP_L3_PRESENT_STRICT says (data safety does not hang on a switch)
        return False
    return bool(l3_complete(node, sec))


def local_veto(tree: Any, pool: Any, n_slots: int):
    """This rank's veto vector (int64, 1 = keep) over ``n_slots`` arena slots, the nodes it may release, the
    reasons counted and the pages proven on L3. An error vetoes every slot (fail closed)."""
    from flliper.srt.mem_cache.unified_radix_cache import BASE_COMPONENT_TYPE

    veto = torch.zeros(int(n_slots), dtype=torch.int64)
    cands: List[tuple] = []
    why: Dict[str, int] = {}
    if not n_slots:
        return veto, cands, why
    if pool is None or getattr(pool, "arena", None) is None:
        # review S86 A3b: a rank that cannot read its pool names nothing it may release -- it vetoes every slot
        # (fail closed); before, it agreed with a zero veto
        veto.fill_(1)
        why["no_pool"] = 1
        return veto, cands, why
    try:
        for node in tree._collect_all_nodes():
            if node is tree.root_node:
                continue
            hv = node.component_data[BASE_COMPONENT_TYPE].host_value
            if hv is None or int(hv.numel()) == 0:
                continue
            sl = _slots(pool, hv)
            if not sl:
                continue
            r = _reason(tree, node, pool)
            if r is None and not _l3_proven(pool, node):
                r = "not_on_l3"
            if r is not None:
                why[r] = why.get(r, 0) + 1
                veto[torch.tensor(sl, dtype=torch.int64)] = 1
            else:
                cands.append((node, sl))
    except Exception:  # noqa: BLE001 - fail closed: nothing released on this rank's word
        logger.warning("%s veto raised: every slot kept", MARK, exc_info=True)
        veto.fill_(1)
        cands = []
        why["error"] = why.get("error", 0) + 1
    return veto, cands, why


def apply(tree: Any, pool: Any, cands: List[tuple], agreed_veto: torch.Tensor,
          released: Optional[List[int]] = None) -> List[int]:
    """Release the host rows of every candidate none of whose slots a rank vetoed; returns the slots let go
    (``released``, when given, is filled as it goes -- a release that raises half-way still names what it did)."""
    from flliper.srt.mem_cache import form_a_host_shadow as _r12
    from flliper.srt.mem_cache.unified_cache_components.tree_component import EvictLayer
    from flliper.srt.mem_cache.unified_radix_cache import BASE_COMPONENT_TYPE

    base = tree.components.get(BASE_COMPONENT_TYPE)
    if released is None:
        released = []
    if base is None:
        return released
    host = _r12.role() == "host"
    for node, sl in cands:
        if any(int(agreed_veto[s]) for s in sl):
            continue
        tree._evict_component_and_detach_lru(node, base, target=EvictLayer.HOST, tracker=None)
        try:
            tree.evictable_host_leaves.discard(node)
        except AttributeError:
            pass
        node.l3_present = True
        if host:
            _r12.record_spill(tree, node)   # the workers released the same node themselves; a SPILL is a noop there
        released.extend(sl)
    return released


def _digest(slots: List[int]) -> int:
    arr = torch.tensor(sorted(set(int(s) for s in slots)), dtype=torch.int64).numpy().tobytes()
    return int(zlib.crc32(arr)) & 0x7FFFFFFF


def run(tree: Any, group_max: Optional[Callable[..., List[int]]] = None, pool: Any = None) -> Dict[str, Any]:
    """The whole step on THIS rank (see the module head). ``group_max`` defaults to ``tree.hicache_group_max``
    (a rank's own value when the tree has none). Returns the counts; ``group_released`` is group-reduced (MAX)."""
    out: Dict[str, Any] = {"ran": False, "nodes": 0, "pages": 0, "group_released": 0, "agreed": True}
    if not enabled():
        return out
    t0 = time.monotonic()
    if group_max is None:
        group_max = getattr(tree, "hicache_group_max", None)
    if not callable(group_max):
        def group_max(values, label=""):  # noqa: E306 - a cache without HiCache collectives: this rank alone
            return list(values)
    if pool is None:
        fn = getattr(tree, "_pdflip_direct_pool", None)
        try:
            pool = fn() if callable(fn) else None
        except Exception:  # noqa: BLE001 - no pool: no slot to name, still every collective
            pool = None
    pool = _arena_pool(pool)
    arena = getattr(pool, "arena", None)
    mine = int(getattr(arena, "slots", 0) or 0) if arena is not None else 0
    (n_slots,) = group_max([mine], label="s86_sleep_l3_release/slots")
    n_slots = int(n_slots)
    out["ran"] = True
    if n_slots <= 0:
        return out
    try:
        return _after_first_collective(tree, pool, group_max, mine, n_slots, out, t0)
    except SleepL3ReleaseDivergence:
        raise
    except Exception as exc:  # noqa: BLE001 - review S86 A1: after the first collective no rank goes on alone
        logger.error("%s raised after the first collective on this rank: the group stops by name", MARK,
                     exc_info=True)
        raise SleepL3ReleaseDivergence(
            "W86 SleepL3ReleaseDivergence: %s raised on this rank after the first collective (%r) -- the ranks may "
            "have let go of different slots; stopped by name instead of flushing on (RAENGE-NIE-UNEINS)"
            % (MARK, exc)) from exc


def _after_first_collective(tree, pool, group_max, mine: int, n_slots: int, out: Dict[str, Any], t0: float):
    arena = getattr(pool, "arena", None)
    if mine and mine != n_slots:
        # one arena file per group: a rank that reads another slot count names nothing (vetoes everything)
        logger.warning("%s slot count %d differs from the group's %d: this rank keeps every slot", MARK, mine,
                       n_slots)
        veto, cands, why = torch.ones(n_slots, dtype=torch.int64), [], {"slot_count": 1}
    else:
        try:
            veto, cands, why = local_veto(tree, pool if arena is not None else None, n_slots)   # A3b: no arena = all
        except Exception:  # noqa: BLE001 - fail closed and keep posting: the peers wait at the next collective
            logger.warning("%s veto raised: every slot kept", MARK, exc_info=True)
            veto, cands, why = torch.ones(n_slots, dtype=torch.int64), [], {"error": 1}
    agreed = torch.tensor(group_max(veto.tolist(), label="s86_sleep_l3_release/veto"), dtype=torch.int64)
    released: List[int] = []
    err = 0
    try:
        apply(tree, pool, cands, agreed, released)
    except Exception:  # noqa: BLE001 - carried to the digest: the whole group stops by name (A1)
        logger.error("%s release raised on this rank after %d slots", MARK, len(released), exc_info=True)
        err = 1
    try:
        cnt, dig = len(set(released)), _digest(released)
    except Exception:  # noqa: BLE001
        cnt, dig, err = len(released), 0, 1
    mx = group_max([cnt, -cnt, dig, -dig, err], label="s86_sleep_l3_release/digest")
    g_max, g_min, d_max, d_min, g_err = int(mx[0]), -int(mx[1]), int(mx[2]), -int(mx[3]), int(mx[4])
    out.update(nodes=sum(1 for n, sl in cands if not any(int(agreed[s]) for s in sl)), pages=cnt,
               group_released=g_max, agreed=(g_max == g_min and d_max == d_min),
               vetoed=int((agreed > 0).sum()), why=why, ms=(time.monotonic() - t0) * 1000.0, group_error=g_err)
    _N["calls"] += 1
    kept_line = ",".join("%s:%d" % kv for kv in sorted(why.items(), key=lambda kv: -kv[1])[:6]) or "-"
    msg = ("%s n=%d nodes=%d slots=%d of %d vetoed_by_group=%d group_released=%d..%d digest=%s kept=%s ms=%.1f "
           "(the sleep's reset drops every tree reference; the pages L3 holds go first, agreed over the group, so "
           "the #1470 flush finds room for the nodes with no copy -- the park-end chains; a page without an L3 "
           "copy is never released)") % (
        MARK, _N["calls"], out["nodes"], cnt, n_slots, out["vetoed"], g_min, g_max,
        "agreed" if out["agreed"] and not g_err else ("ERROR" if g_err else "DIVERGED"), kept_line, out["ms"])
    if out["agreed"] and not g_err:
        logger.info(msg)
        return out
    logger.error(msg)
    raise SleepL3ReleaseDivergence(
        "W86 SleepL3ReleaseDivergence: %s %s -- released %d..%d slots, digest %d..%d, rank error %d: the ranks are not "
        "one (RAENGE-NIE-UNEINS); every rank reads this reduced digest and stops in the same pass"
        % (MARK, "DIVERGED" if not out["agreed"] else "ERROR", g_min, g_max, d_min, d_max, g_err))
