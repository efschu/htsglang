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

S86e PARK-L3 (rc27 boot ..._893d09fa3f_1021_023225, D group TP=3 Form A). The veto above kept every HOST-ONLY node
("not this step") and every node with an aux state on the host only. At both sleeps of that boot those two reasons
held the whole arena: 02:44:15 ``S86 SLEEP-L3-RELEASE ... vetoed_by_group=6251 group_released=16..183
kept=host_only:19,aux_host_only:7`` next to ``S83 VICTIM-CENSUS ... pinned=6418 tree_device=2237
tree_host_only=4014``; ``#1470 FLUSH-UNBACKED nodes=4 tokens=10112`` and ``S85 PARK-END-UNBACKED rid=pdflip-2-30
node=155 depth=139008 store_end=133632 lost_tokens=5376`` -- P recomputed 5376 tokens (P ``PDFLIP-CLAIM-SHORT
rid=pdflip-2-30 ... by=kv_prefix tokens=133632``). 03:05:42 the same shape: ``kept=host_only:14,aux_host_only:8``,
``tree_host_only=4480`` of 6485, ``#1470 FLUSH-PUBLISH ARENA-FULL-STOP unbacked=14 arena_refused=14 issued=0``, the
anchor of pdflip-26-148 at 31488 lost (P ``CLAIM-SHORT ... lost=28 by=mamba tokens=29440``). rc29's S109/S109c (the
host-leaf and device-round AUX stages) act in the AWAKE W3 spill only; inside the sleep's flush the workers'
references stay until a broadcast that never comes -- this step is the only release there, and it kept them.

The reset a moment later drops every one of those references anyway. So, with the same proof as above:

* a HOST-ONLY node whose KV pages all have their L3 copy leaves the tree (``_evict_host_leaf``, children first) --
  but only when its WHOLE host-only subtree leaves with it: a node with a kept descendant (or a device child) is
  kept, and that dependency joins the S86c fixed point (``_mixed_block``: the subtree's slots, not only the node's);
* a node whose only host-only aux state is its MAMBA anchor (device-resident or host-only) is a candidate too; on the
  rank that owns the anchor pool its anchor must be published as on L3 by the write-behind (the S39/S109b helper
  ``forma_l3_subtree._secure_child_anchors``: memory read, a missing copy is QUEUED and the node is kept,
  ``anchor_pending``). A rank without an anchor pool (a Form A worker) has no anchor of its own to judge -- the
  owner's veto keeps the node's KV slots on every rank through the group MAX. Any other host-only aux type keeps the
  old veto.
* every other guard stays: a host lock, a write in flight (by id, by pending id, a #1427 direct write's mamba rows),
  an END anchor not read yet, no page hashes, staging rows, a live aux device value on a host-only node.

TP0 sends a STATE for every node it changed (exists=0 for a removed host-only node, the S109c/K23 form) so its shadow
of the workers stays true should the sleep not reset. No reserve, no timer, no switch of its own, no file I/O on the
scheduler thread (S109b: the copies are the write-behind's; this step only reads its record).
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


def _aux_kinds(tree: Any, node: Any) -> list:
    """The aux component types whose state lives on the host only on ``node``."""
    from flliper.srt.mem_cache.unified_radix_cache import _aux_components

    return [c.component_type for c in _aux_components(tree)
            if node.component_data[c.component_type].host_value is not None
            and node.component_data[c.component_type].value is None]


def _mamba_ct():
    from flliper.srt.mem_cache.unified_cache_components.tree_component import ComponentType

    return getattr(ComponentType, "MAMBA", None)


def _reason(tree: Any, node: Any, pool: Any) -> Optional[str]:
    """None = a node whose host copy may leave L2 if its pages have an L3 copy (checked by the caller; S86e: and, for
    a mamba-aux node, its anchor on the owner rank; and, for a host-only node, its whole host-only subtree); else the
    name of the first check that keeps it. Same guards as the W3 device round / the host-leaf round, without the
    in-phase ones that protect a read still to come (claimer chain, told head, keep-payable): at the sleep the reset
    drops those references too."""
    from flliper.srt.mem_cache.unified_radix_cache import BASE_COMPONENT_TYPE, _aux_components

    cd = node.component_data[BASE_COMPONENT_TYPE]
    host_only = cd.value is None
    if host_only and not (getattr(node, "evicted", True) and getattr(node, "backuped", True)):
        return "host_only_state"          # S86e: not a host leaf by the tree's own predicate
    if node.id in (getattr(tree, "ongoing_write_through", None) or {}):
        return "write_pending"
    if getattr(node, "write_through_pending_id", None) is not None:
        return "write_pending"
    if any(int(getattr(c, "host_lock_ref", 0) or 0) > 0 for c in node.component_data):
        return "host_locked"
    if host_only:
        if node.id in (getattr(tree, "_pdflip_direct_mamba_rows", None) or {}):
            return "write_pending"        # a #1427 direct write's mamba rows in flight (host_only_spill V1 guard)
        if getattr(node, "_pdflip_end_anchor", False):
            mct = _mamba_ct()
            if mct is not None and len(node.component_data) > int(mct) \
                    and node.component_data[mct].host_value is not None:
                return "end_anchor"       # a host-backed END anchor D may not have read (host_only_spill V1 guard)
    kinds = _aux_kinds(tree, node)
    if kinds and not all(k == _mamba_ct() for k in kinds):
        return "aux_host_only"            # S86e: only a MAMBA anchor is judged (below); any other aux keeps the node
    if host_only and any(node.component_data[c.component_type].value is not None for c in _aux_components(tree)):
        return "aux_device_value"         # K15b A4: removing the node would free a live device state
    hv = cd.host_value
    if int(hv.min()) < int(getattr(pool, "staging_rows", 0)):
        return "staging"
    if not getattr(node, "hash_value", None):
        return "no_hash"
    return None


def _anchor_reason(tree: Any, node: Any) -> Optional[str]:
    """S86e: None when ``node`` has no host-only MAMBA anchor, when this rank owns no anchor pool (a Form A worker:
    the owner judges it, its veto reaches this rank through the group MAX), or when the write-behind's record says
    the anchor has its L3 copy; else the refusal (the anchor is queued for the write-behind; the node is kept).
    Memory reads only (S109b)."""
    mct = _mamba_ct()
    if mct is None or len(node.component_data) <= int(mct) or node.component_data[mct].host_value is None:
        return None
    fn = getattr(tree, "_pdflip_mamba_pool", None)
    try:
        mp = fn() if callable(fn) else None
    except Exception:  # noqa: BLE001 - an unreadable pool on the owner keeps the node (fail closed)
        return "anchor_pool_error"
    if getattr(mp, "arena", None) is None:
        return None
    from flliper.srt.pdflip import forma_l3_subtree as _fls

    r = _fls._secure_child_anchors(tree, [node], [_fls.ANCHOR_SECURE_PER_FLUSH])
    if r is None:
        return None
    return "anchor_pending" if str(r[0]) == "child_anchor_pending" else str(r[0])


def _l3_proven(pool: Any, node: Any) -> bool:
    from flliper.srt.mem_cache.unified_radix_cache import BASE_COMPONENT_TYPE
    from flliper.srt.pdflip.read_room import l3_complete

    sec = pool.secure_rows_to_l3(node.component_data[BASE_COMPONENT_TYPE].host_value)
    if int(sec.get("unpaired", 0) or 0):
        # review S86 A3a: an unpaired page (KV held back for want of its QSA index) is NOT on disk -- never secured
        # here, whatever FLLIPER_PDFLIP_L3_PRESENT_STRICT says (data safety does not hang on a switch)
        return False
    return bool(l3_complete(node, sec))


#: candidate kinds (third tuple element): the node stays on the device / the host-only node leaves the tree
KIND_DEVICE = "device"
KIND_HOST_ONLY = "host_only"


def _depth(node: Any) -> int:
    d, n = 0, node
    while getattr(n, "parent", None) is not None:
        d, n = d + 1, n.parent
    return d


def local_veto(tree: Any, pool: Any, n_slots: int):
    """This rank's veto vector (int64, 1 = keep) over ``n_slots`` arena slots, the nodes it may release, the
    reasons counted and the pages proven on L3. An error vetoes every slot (fail closed).

    A candidate is ``(node, slots, kind, subtree_slots)``: ``kind`` KIND_DEVICE (the host copy leaves, the node stays
    on the device) or KIND_HOST_ONLY (S86e: the node leaves the tree, after every host-only descendant -- all of them
    are candidates, ``subtree_slots`` names their slots and the node's own)."""
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

    def _keep(r, sl):
        why[r] = why.get(r, 0) + 1
        if sl:
            veto[torch.tensor(sl, dtype=torch.int64)] = 1

    try:
        host_ok: Dict[int, tuple] = {}     # id(host-only node) -> (node, own slots), own checks passed
        host_kept: Dict[int, tuple] = {}   # id(host-only node) -> (node, own slots), kept by its own checks
        for node in tree._collect_all_nodes():
            if node is tree.root_node:
                continue
            cd = node.component_data[BASE_COMPONENT_TYPE]
            hv = cd.host_value
            if hv is None or int(hv.numel()) == 0:
                if cd.value is None:
                    host_kept[id(node)] = (node, [])   # no host copy, not on the device: never removed here
                continue
            sl = _slots(pool, hv)
            host_only = cd.value is None
            if not sl and not host_only:
                continue
            r = _reason(tree, node, pool)
            if r is None and host_only and not sl:
                r = "no_arena_slot"           # its host rows name no arena page: no L3 proof possible, it stays
            if r is None and sl and not _l3_proven(pool, node):
                r = "not_on_l3"
            if r is None:
                r = _anchor_reason(tree, node)
            if host_only:
                (host_kept if r is not None else host_ok)[id(node)] = (node, sl)
                if r is not None:
                    _keep("host_only:" + r, sl)
                continue
            if r is not None:
                _keep(r, sl)
            else:
                cands.append((node, sl, KIND_DEVICE, sl))
        # S86e: a host-only node leaves only with its whole host-only subtree (children first). A child that is on
        # the device, kept by its own checks, or itself blocked keeps it -- deepest first, so a child is settled
        # before its parent.
        blocked: Dict[int, bool] = {}
        sub: Dict[int, List[int]] = {}
        for nid, (node, sl) in sorted(host_ok.items(), key=lambda kv: -_depth(kv[1][0])):
            acc = list(sl)
            bad = False
            for ch in list(node.children.values()):
                cid = id(ch)
                if cid in host_ok and not blocked.get(cid, True):
                    acc.extend(sub[cid])
                else:
                    bad = True
            blocked[nid] = bad
            sub[nid] = acc
            if bad:
                _keep("host_only:child_kept", sl)
            else:
                cands.append((node, sl, KIND_HOST_ONLY, acc))
    except Exception:  # noqa: BLE001 - fail closed: nothing released on this rank's word
        logger.warning("%s veto raised: every slot kept", MARK, exc_info=True)
        veto.fill_(1)
        cands = []
        why["error"] = why.get("error", 0) + 1
    return veto, cands, why


def _cand(c) -> tuple:
    """``(node, slots, kind, subtree_slots)`` of a candidate (a 2-tuple is a device candidate, the S86 form)."""
    if len(c) >= 4:
        return c[0], c[1], c[2], c[3]
    return c[0], c[1], KIND_DEVICE, c[1]


def apply(tree: Any, pool: Any, cands: List[tuple], agreed_veto: torch.Tensor,
          released: Optional[List[int]] = None) -> List[int]:
    """Release the host rows of every candidate none of whose slots a rank vetoed (S86e: a host-only node none of
    whose SUBTREE slots a rank vetoed leaves the tree, deepest first); returns the slots let go (``released``, when
    given, is filled as it goes -- a release that raises half-way still names what it did)."""
    from flliper.srt.mem_cache import form_a_host_shadow as _r12
    from flliper.srt.mem_cache.unified_cache_components.tree_component import EvictLayer
    from flliper.srt.mem_cache.unified_radix_cache import BASE_COMPONENT_TYPE

    base = tree.components.get(BASE_COMPONENT_TYPE)
    if released is None:
        released = []
    if base is None:
        return released
    host = _r12.role() == "host"
    dev = [c for c in map(_cand, cands) if c[2] == KIND_DEVICE]
    hon = sorted((c for c in map(_cand, cands) if c[2] == KIND_HOST_ONLY), key=lambda c: -_depth(c[0]))
    for node, sl, _k, _sub in dev:
        if any(int(agreed_veto[s]) for s in sl):
            continue
        aux = bool(_aux_kinds(tree, node))
        tree._evict_component_and_detach_lru(node, base, target=EvictLayer.HOST, tracker=None)
        try:
            tree.evictable_host_leaves.discard(node)
        except AttributeError:
            pass
        node.l3_present = True
        if host:
            if aux:
                # S86e: the anchor's host state left too (the tree law) -- the S109c form, a STATE
                _r12.record_state(tree, node, why="s86e_aux_device")
            else:
                _r12.record_spill(tree, node)   # the workers released the same node themselves; a noop there
        released.extend(sl)
    for node, sl, _k, sub in hon:
        if any(int(agreed_veto[s]) for s in sub):
            continue
        if not tree._is_host_leaf(node):
            # cannot happen when the veto saw the same tree (every descendant is a released candidate, deeper
            # first); kept -- the residual check names it (W86) instead of a silent half-release
            logger.warning("%s S86e host-only node %s is no host leaf at its release: kept", MARK,
                           getattr(node, "id", "?"))
            continue
        tracker = {ct: 0 for ct in getattr(tree, "tree_components", (BASE_COMPONENT_TYPE,))}
        tracker.setdefault(BASE_COMPONENT_TYPE, 0)
        tree._evict_host_leaf(node, tracker)
        if host:
            _r12.record_state(tree, node, why="s86e_host_only")   # exists=0: the K23 form
        released.extend(sl)
    return released


FIX_ROUNDS = 16


def _mixed_block(cands: List[tuple], agreed: torch.Tensor) -> torch.Tensor:
    """S86c: this rank's slots that the agreed set cannot release here -- the agreed slots of a candidate node that
    also holds a vetoed slot (the node is kept whole, so its agreed slots stay referenced). S86e: a host-only
    candidate is kept whole when any slot of its host-only SUBTREE is vetoed (it can leave the tree only after every
    descendant). An error blocks every slot (fail closed; the collective is still posted)."""
    out = torch.zeros_like(agreed)
    try:
        for c in cands:
            _node, sl, _kind, sub = _cand(c)
            if not sl:
                continue
            flags = [int(agreed[s]) for s in sl]
            sub_vetoed = any(int(agreed[s]) for s in sub)
            if (any(flags) and not all(flags)) or (sub_vetoed and not all(flags)):
                out[torch.tensor(sl, dtype=torch.int64)] = 1
    except Exception:  # noqa: BLE001
        logger.warning("%s mixed-node scan raised: every slot kept", MARK, exc_info=True)
        out.fill_(1)
    return out


def _residual_after(tree: Any, pool: Any, agreed: torch.Tensor) -> int:
    """S86d (review S86c A1): the agreed slots whose reference still EXISTS on this rank AFTER the release, read from
    the tree's state, not from the release's own bookkeeping: every node of this rank's tree that still names an
    agreed slot in its Full host rows (a release that "gave back" a node but kept its host rows, a node the
    candidate scan never saw). 0 on every rank when the group is one.

    Deliberately NOT the arena's per-process RefLedger: on a Form A group the workers QUEUE their arena references
    (host_mem_release_queue, S83) and give them back before each #1470 round -- right after the release their
    ledger still holds them, a correct release would read as a residual and stop the group (false W86)."""
    from flliper.srt.mem_cache.unified_radix_cache import BASE_COMPONENT_TYPE

    left = set()
    for node in tree._collect_all_nodes():
        if node is tree.root_node:
            continue
        hv = node.component_data[BASE_COMPONENT_TYPE].host_value
        if hv is None or int(hv.numel()) == 0:
            continue
        for s in _slots(pool, hv):
            if not int(agreed[s]):
                left.add(int(s))
    return len(left)


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
    # S86c (rc23 D 11:05:06Z, W86 DIVERGED released 431..433): the node structure is RANK-LOCAL -- TP0 (share-0 host)
    # holds other nodes than the token owners (ARENA-REF-HOLDERS tree=259 vs 433; kept aux_host_only:24 vs 8) and
    # splits are rank-local. A node holding an agreed slot AND a vetoed one is kept whole on its rank while a peer
    # whose node holds only the agreed slot lets it go. Fixed point over the group: the slots of such a mixed node
    # join the veto until the reduced vector stands; then every node that holds an agreed slot can release it on
    # every rank. The loop ends on a group-reduced comparison only (same rounds on every rank); past the bound every
    # slot is vetoed (uniform).
    rounds = 0
    while True:
        rounds += 1
        nxt = torch.tensor(group_max(torch.maximum(agreed, _mixed_block(cands, agreed)).tolist(),
                                     label="s86_sleep_l3_release/veto_fix"), dtype=torch.int64)
        if torch.equal(nxt, agreed):
            break
        agreed = nxt
        if rounds >= FIX_ROUNDS:
            logger.warning("%s veto fixed point not reached in %d rounds: every slot kept", MARK, rounds)
            agreed = torch.ones_like(agreed)
            break
    released: List[int] = []
    err = 0
    try:
        apply(tree, pool, cands, agreed, released)
    except Exception:  # noqa: BLE001 - carried to the digest: the whole group stops by name (A1)
        logger.error("%s release raised on this rank after %d slots", MARK, len(released), exc_info=True)
        err = 1
    try:
        cnt = len(set(released))
        # S86c: the unity check reads what must be one -- the agreed release set (its digest, the same on every rank)
        # and the RESIDUAL: agreed slots this rank holds but did not let go (0 on every rank). A rank that holds a
        # slot nobody else holds (TP0's tree is not the workers') releases a different COUNT, which is no divergence.
        res = _residual_after(tree, pool, agreed)
        dig = _digest(torch.nonzero(agreed == 0).reshape(-1).tolist())
    except Exception:  # noqa: BLE001
        cnt, res, dig, err = len(released), 0, 0, 1
    mx = group_max([cnt, -cnt, dig, -dig, err, res], label="s86_sleep_l3_release/digest")
    g_max, g_min, d_max, d_min, g_err, g_res = (int(mx[0]), -int(mx[1]), int(mx[2]), -int(mx[3]), int(mx[4]),
                                                int(mx[5]))
    out.update(nodes=sum(1 for c in map(_cand, cands) if not any(int(agreed[s]) for s in c[3])), pages=cnt,
               group_released=g_max, agreed=(d_max == d_min and g_res == 0),
               vetoed=int((agreed > 0).sum()), why=why, ms=(time.monotonic() - t0) * 1000.0, group_error=g_err,
               fix_rounds=rounds, residual=res)
    _N["calls"] += 1
    kept_line = ",".join("%s:%d" % kv for kv in sorted(why.items(), key=lambda kv: -kv[1])[:6]) or "-"
    msg = ("%s n=%d nodes=%d slots=%d of %d vetoed_by_group=%d group_released=%d..%d fix_rounds=%d residual=%d "
           "digest=%s kept=%s ms=%.1f "
           "(the sleep's reset drops every tree reference; the pages L3 holds go first, agreed over the group, so "
           "the #1470 flush finds room for the nodes with no copy -- the park-end chains; a page without an L3 "
           "copy is never released; released counts differ where a rank holds slots its peers do not)") % (
        MARK, _N["calls"], out["nodes"], cnt, n_slots, out["vetoed"], g_min, g_max, rounds, g_res,
        "agreed" if out["agreed"] and not g_err else ("ERROR" if g_err else "DIVERGED"), kept_line, out["ms"])
    if out["agreed"] and not g_err:
        logger.info(msg)
        return out
    logger.error(msg)
    raise SleepL3ReleaseDivergence(
        "W86 SleepL3ReleaseDivergence: %s %s -- agreed-set digest %d..%d, residual (agreed slots a rank kept) %d, "
        "rank error %d, released %d..%d: the ranks are not one (RAENGE-NIE-UNEINS); every rank reads this reduced "
        "digest and stops in the same pass"
        % (MARK, "DIVERGED" if not out["agreed"] else "ERROR", d_min, d_max, g_res, g_err, g_min, g_max))
