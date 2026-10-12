"""SF SLOT FIDELITY (27B rc12k27 b23, 27.09. 10:27:56Z, rid pdflip-68-235): on the
carrierless PP form every stage admits a request in the SAME pass and at the SAME
width, or none does.

THE DEATH. P log boot_weg2_dkr27bparkodirectdraftbar1w2309270957 (9417507cd2):
PP1 ``#1004 SLOT DISAGREEMENT: PP1 is launching slot 0 (fwd_ct=1582,
rids=[pdflip-68-235], extend=512) but the upstream's proxy names slot 1 (rows=1024,
extent=('pdflip-68-235', 40958, 41982))``. The told was 40958 on every rank, every
rank planned the head at 40958 (``P-CHUNK-POLICY ... start=40958 widths=[512x39,469]``),
the host read was complete on all three (``#905 ... completed=40958``). Then:

  PP0   PDFLIP-LOADBACK-EVICT kv_tokens=27095 floor=13035: the uniform floor is short by
        14060 ... this pass refuses (n=4)
  PP0   PDFLIP-LOADBACK-WAIT extent=40958 applied=0 ... the request waits
  PP1/2 #988 LOADBACK prefix moved to 40958 -> #969N ADMIT slot=0 extend=512
  PP0   (next pass) P-CHUNK-POLICY start=0 widths=[1024x59,979] pick=mid=1024
  PP0   H91 STORE-TOLD KEPT told=40958 -> #988 LOADBACK prefix moved to 40958
        -> #969N ADMIT slot=1 extend=1024

TERM (a) -- THE ROOM VERDICT WAS RANK-LOCAL. The "uniform floor" the load-back
decides from is the MIN over ``tp_cpu_group`` -- and on TP=1/PP=3 that group has one
member (boot line 1810: ``#788 UNIFORM-FLOOR SCOPE: tp_cpu_group world=1 -> floors
OFF ... the ranks that must agree are NOT in this reduce group``). The single-rank
path publishes THIS rank's ``available_size()`` (#1045), so the refuse-this-pass /
retry-next-pass rule built for a TP group's published MIN (xsn285) turns a local
shortfall into a PASS SKEW: PP0 was short, its peers were not, PP0 admitted one
pass later.

THE RULE (a). On a floor that is this rank's own value under pp > 1, the load-back
does what upstream's ``load_back`` does: evict what is short and load in the SAME
pass. The retry-next-pass exists so that a TP group re-reads a new MIN; a group of
one has nothing to re-read, so waiting a pass only buys the skew. Every rank then
admits in the same pass whenever free + evictable covers the host hit -- 218004
evictable against a 14060 shortfall on the specimen. Only when this rank cannot
hold the load-back even with every evictable row freed does it still refuse (the
residual, named ``SF LOADBACK-ROOM PP-RESIDUAL``; on this form a peer with room
admits in this pass, so that line is the cause of the ``#1004`` that follows).

TERM (b) -- THE CHUNK PLAN SURVIVED THE PREFIX MOVE. #1400's ``admission`` pops
``_pdflip_store_told[rid]`` on the first visit; H91 keeps the verdict in
``_pdflip_told_kept``. Fix A's ``budget_head`` read only the told map, so PP0's second
visit sized the pass from position 0 (``pos_src=zero``) -- a plan for start=0 whose
width (1024) then ran on an extent that #988 moved to 40958, where the plan says 512.

THE RULE (b). (b1) ``budget_head`` reads the KEPT told when the told map no longer
has the rid (same request object): every rank keeps the same verdict the same way,
so the position stays rank-identical. (b2) Whenever #988 moves the prefix of the
request the pass budget was planned for to a position OTHER than the planned one,
the plan is replanned at the moved prefix and the pass budget NARROWED to it (never
widened: the corridor granted the old width, not a wider one). Deterministic in the
planner's call sequence, which is the same on every rank when the move is.

NOT BUILT: followers taking PP0's width from PP0's row. On this form
(``pp_row_carrier_present`` False: no ``pp_flip_counters`` side channel, followers
log ``#1460 FOLLOWER-GATE gate=None``) PP0's row reaches a follower only with the
proxy of the pass it already admitted (Fix B, unbuilt). The rule here is the other
half of the same law: every input of the width and of the pass is rank-identical,
and ``#1004`` / ``#1233 W27`` stay the named stops if they ever are not.

Switch ``FLLIPER_PDFLIP_PP_SLOT_FIDELITY``, default on; ``0`` = the old behaviour byte
for byte (no flag is written, no hook installed, no extra lookup).
"""

from __future__ import annotations

import logging
import os
from typing import NamedTuple, Optional

logger = logging.getLogger(__name__)

ENV = "FLLIPER_PDFLIP_PP_SLOT_FIDELITY"
#: tree attribute: True while this iteration's evict floor is THIS rank's own
#: ``available_size()`` on a multi-stage PP form (tp group of one).
FLOOR_LOCAL_PP_ATTR = "pdflip_sf_floor_local_pp"
#: scheduler attribute: the plan this pass's budget was sized from (cleared at
#: the top of every budget sizing, consumed by the adder's move hook).
PLAN_ATTR = "_pdflip_sf_budget_plan"
#: adder attribute: the chunk budget the adder was built with.
ADDER_INITIAL_ATTR = "sf_rem_chunk_initial"

_LOG_FIRST = 8


def enabled(env=None) -> bool:
    e = os.environ if env is None else env
    raw = (e.get(ENV, "") or "").strip().lower()
    return raw not in ("0", "false", "no", "off")


def _sampled(holder, attr: str) -> Optional[int]:
    """Counter on ``holder``; returns n when this occurrence prints (first 8,
    then powers of two), else None. Never raises."""
    try:
        n = int(getattr(holder, attr, 0) or 0) + 1
        setattr(holder, attr, n)
    except Exception:  # noqa: BLE001 -- a slotted double
        return None
    return n if (n <= _LOG_FIRST or (n & (n - 1)) == 0) else None


# ---------------------------------------------------------------------------
# (a) the floor's scope, published by the scheduler
# ---------------------------------------------------------------------------


def mark_floor_scope(tree, local_pp: bool) -> None:
    """Record on the tree whether this iteration's evict floor is this rank's own
    value on a multi-stage PP form. Switch off: nothing is written."""
    if tree is None or not enabled():
        return
    try:
        setattr(tree, FLOOR_LOCAL_PP_ATTR, bool(local_pp))
    except Exception:  # noqa: BLE001 -- a slotted double
        pass


def pp_size_of(holder) -> int:
    args = getattr(holder, "server_args", None)
    try:
        return int(getattr(args, "pp_size", 1) or 1) if args is not None else 1
    except (TypeError, ValueError):
        return 1


#: PR (pdflip/pp_room_vote.py; mem_cache.common.PP_ROOM_CAP_ATTR): the room the
#: stages agreed for this pass (R_m), read by its literal name.
_AGREED_ATTR = "pdflip_pp_room_cap"


def agreed_room(tree) -> Optional[int]:
    cap = getattr(tree, _AGREED_ATTR, None)
    if isinstance(cap, int) and not isinstance(cap, bool):
        return max(0, cap)
    return None


def agreed_floor(tree, floor: int) -> int:
    """PR: under an agreed R_m the load-back's floor is at most R_m, on every
    stage alike (14:11Z int20: PP1 loaded pdflip-8-75 a pass before PP0 -> #1004)."""
    cap = agreed_room(tree)
    return int(floor) if cap is None else min(int(floor), cap)


class AgreedRoomShort(RuntimeError):
    """PR: this stage cannot hold a load-back the stages agreed it can."""


def local_pp_room(tree, kv_tokens: int, floor: int, rid=None) -> Optional[bool]:
    """The load-back room verdict on a local floor under pp > 1.

    None  -- not this form (switch off, a group floor, pp == 1): the caller keeps
             its path unchanged.
    True  -- room is there now (after evicting the shortfall): load in THIS pass.
    False -- the residual: even every evictable row does not make room; the
             caller refuses as before."""
    if not enabled() or not getattr(tree, FLOOR_LOCAL_PP_ATTR, False):
        return None
    alloc = getattr(tree, "token_to_kv_pool_allocator", None)
    if alloc is None:
        return None
    kv_tokens = int(kv_tokens)
    agreed = agreed_room(tree)
    if agreed is not None and kv_tokens > agreed:
        # every stage reads the same R_m: every stage refuses this pass
        return False
    avail0 = int(alloc.available_size())
    evictable = int(tree.evictable_size())
    evicted = 0
    short = kv_tokens - avail0
    if short > 0 and evictable > 0:
        from flliper.srt.mem_cache.base_prefix_cache import EvictParams

        res = tree.evict(EvictParams(num_tokens=min(short, evictable)))
        evicted = int(getattr(res, "num_tokens_evicted", 0) or 0)
    avail1 = int(alloc.available_size())
    ok = avail1 >= kv_tokens
    if ok:
        n = _sampled(tree, "_pdflip_sf_room_same_pass")
        if n is not None:
            logger.info(
                "SF LOADBACK-ROOM SAME-PASS rid=%s kv_tokens=%d floor=%d avail=%d "
                "evictable=%d evicted=%d avail_after=%d (n=%d): the floor is this "
                "rank's own value (tp group of one, pp>1), so the shortfall is "
                "evicted and the host hit loads in THIS pass -- a next-pass retry "
                "would only skew this stage one pass behind its peers (#1004, b23)",
                rid, kv_tokens, int(floor), avail0, evictable, evicted, avail1, n,
            )
    elif agreed is not None:
        # the stages agreed this load-back fits; this one cannot pay it. A local
        # refusal would start it on the peers alone (#1004) -- stop by name.
        from flliper.srt.pdflip.pp_room_vote import agreed_short, vote_diagnostic

        n = agreed_short("loadback", rid, kv_tokens, agreed, avail1)
        raise AgreedRoomShort(
            f"PR AGREED-ROOM SHORT rid={rid} kv_tokens={kv_tokens} agreed={agreed} "
            f"avail={avail0} evictable={evictable} evicted={evicted} avail_after={avail1} "
            f"(n={n}): the stages agreed this pass can hold the load-back and this "
            "stage cannot; refusing alone would start it on the peers only (#1004)."
            + vote_diagnostic(tree)
        )
    else:
        n = _sampled(tree, "_pdflip_sf_room_residual")
        if n is not None:
            logger.warning(
                "SF LOADBACK-ROOM PP-RESIDUAL rid=%s kv_tokens=%d floor=%d avail=%d "
                "evictable=%d evicted=%d avail_after=%d (n=%d): this rank cannot hold "
                "the load-back even with every evictable row freed and refuses this "
                "pass. The verdict is RANK-LOCAL on a carrierless PP form: a peer "
                "with room admits in this pass, and if one does the #1004 SLOT "
                "DISAGREEMENT that follows has THIS line as its cause",
                rid, kv_tokens, int(floor), avail0, evictable, evicted, avail1, n,
            )
    return ok


def note_loaded(tree, rows: int) -> None:
    """SF-X (rc12q PP2 15:38:49Z, pdflip-4-37): charge a load-back's device rows
    against this iteration's floor when the floor is this rank's own value.

    The floor (5336) was published at the top of the iteration; the same-pass
    room evicted 810 and the load took 5932 rows, so the live pool held 214 --
    while the extend's own eviction trigger (``uniform_avail_for_evict`` =
    floor - admitted ledger) still read 5336 >= 775 and SKIPPED, and the
    775-token extend raised with 245003 evictable. Charged here, the trigger
    reads 5336 - 5932 < 775 and evicts what the extend needs. Only on the
    local-PP form: on a TP group the floor is a group MIN and its ledger is
    charged by replicated allocations only (unchanged)."""
    if not enabled() or not getattr(tree, FLOOR_LOCAL_PP_ATTR, False) or rows <= 0:
        return
    try:
        from flliper.srt.mem_cache.common import note_uniform_admitted

        note_uniform_admitted(tree, int(rows))
    except Exception:  # noqa: BLE001 -- a ledger note never breaks a load
        pass


def node_in_flight(ongoing, node) -> bool:
    """A write of ``node`` is in flight: tracked by its id, or by the id a split
    node kept (``write_through_pending_id`` names the OLD id)."""
    if getattr(node, "id", None) in (ongoing or {}):
        return True
    return getattr(node, "write_through_pending_id", None) is not None


def node_host_locked(node) -> bool:
    """A load-back reads a host row of ``node`` (any component)."""
    return any(int(getattr(cd, "host_lock_ref", 0) or 0) > 0
               for cd in (getattr(node, "component_data", None) or ()))


def unbacked_drop_floor(tree, node) -> bool:
    """UD's floor half: the local-PP floor (the tree is this rank's own; on a TP
    group a rank-local drop would split the replicas) and no write-through of
    ``node`` in flight. Children are not looked at here."""
    if not enabled() or not getattr(tree, FLOOR_LOCAL_PP_ATTR, False):
        return False
    return not node_in_flight(getattr(tree, "ongoing_write_through", None) or {}, node)


def unbacked_drop_allowed(tree, node) -> bool:
    """UD (NF rc12q PP2 16:28:19Z): may an eviction DROP a write_back leaf whose
    backup was refused? Only on the local-PP floor, only a node with no children
    (#841: an un-backed node with children would orphan them) and no
    write-through in flight."""
    if not unbacked_drop_floor(tree, node):
        return False
    return not getattr(node, "children", None)


# ---------------------------------------------------------------------------
# ROOM-VOTE 1011 (NF int24e 09:29:33Z, death 09:39:42Z): ONE evictability check
# for the room vote (pdflip/pp_room_vote.estimate_payable) and the peel
# (UnifiedRadixCache._evict_device_leaf).
#
# THE DEATH. PP0's vote paid node 181 (120448 tokens) -- ``agreed=140416 =
# avail 19008 + reported 121408`` -- and the peel paid 960
# (``AgreedRoomShort ... evicted=960 avail_after=19968``).
#
# BELEGT (P.log of boot ..._6c53a955c7_1011_092933, PP0):
#   * agreed - avail_after = 140416 - 19968 = 120448 = node 181 exactly: the vote
#     counted all of ``reported`` and the peel paid everything but 181. The binding
#     number is PP0's OWN measurement (avail + reported to the token), not a stale
#     follower fact (that is Room-Short 1008, see below).
#   * 181: un-backed, ``l3_present`` after ``W3-ARENA SPILL released_pages=1882``
#     (39846, 09:39:32); at the peel ``#1421 BACKUP-REFUSED why=arena_claim node=181``
#     x2 (40132-40134, ``ARENA-DROP need=1006 freed=0``); no ``EVICT-UNBACKED-DROP``
#     (0 in the boot), so the drop did not happen.
#   * 194/195: their Full and aux host copies left at 09:39:32 (39853/39855
#     ``AUX-HOST-WITH-FULL ... released=1``). In the death pass the peel ran
#     ``#1465 WRITE-BACK DRAIN`` -> ``#1469 EVICT node=195/194 backuped=True`` ->
#     ``HOSTLOCK-LRU-SKIP ct=MAMBA host_lock_ref=1``: the peel's own backup preceded
#     the demotion (so they were NOT backed when the peel began; whether they were
#     when the vote ran is unbelegt), and the demoted nodes carried a MAMBA host lock.
#   * 181's children 194/195 (960 tokens) were paid; ``EVICT-FRONTIER-CENSUS leaves 1->1``.
#
# UNBELEGT -- WHY 181 COULD NOT LEAVE AT THE PEEL. The log does not name it. Candidates:
#   H1  the demoted children carried a host lock (``HOSTLOCK-LRU-SKIP``), so UD-H could not
#       clear them and 181 could not drop. WHEN the lock appeared (before or after the vote)
#       is not in the log. Code fact: ``load_back`` takes ``inc_host_lock_ref`` of its own
#       anchor (unified_radix_cache.py:6781) BEFORE ``local_pp_room`` (6905), i.e. after the
#       pass-top vote; whether 194/195 lie on that anchor path is unbelegt.
#   H2  181 was in ``ongoing_write_through`` at the peel (``PUBLISH-SWEEP unbacked=3
#       issued=3`` in the same pass, 40101; node ids not logged; the analysis report's own
#       hypothesis) -- the drop floor reads ``node_in_flight``.
#   H3  the host room the vote priced a backup with was spent/refused (host_free unbelegt).
# What this commit guarantees: for the tree state the vote SEES (pass top) the vote and the
# peel ask ONE function; a fresh walk equals the peel then (property tests). It does NOT
# close a state change between the vote and the peel (H1 with a late lock, H2 with a late
# write): the vote then over-counts, and ``AgreedRoomShort`` stays the stop for it. The stop
# now prints ``vote_payable`` next to ``walk_now`` and the refused nodes
# (``pp_room_vote.vote_diagnostic``): the next occurrence names the window instead of a guess.
#
# Earlier instances of "room estimate != peel": PP-ROOM-VOTE 24ed9b508b (int19 13:37:49 PP2),
# UD-H bff5913a62 (int19), PR-ARENA c59353374a (int22 18:34:20 PP1); and a different
# mechanism of the same stop, Room-Short 1008 (desk/nf-room-short-1008): a FOLLOWER's fact
# goes stale (``FACT_MAX_AGE_PASSES=8``; ``RoomBook.cap`` subtracts only PP0-started batches),
# counted as ``PR AGREED-ABOVE-LOCAL where=pass`` (5 of them, 64-128 tokens on PP1/PP2 in
# this boot, none fatal). This commit does not touch it.
#
# WHAT THE WALK DOES NOW. It carries per node ``clr`` (gone, or a host-only node UD-H could
# clear afterwards); a node the peel DEMOTES counts as clearable only when
# ``demoted_node_clearable``: no host lock, children clearable -- the test
# ``host_node_local_ok`` applies once the node IS host-only; a node's drop (UD / L3) needs
# all its children ``clr``. (Not proven to be the cause of 092933, but it is the rule the
# peel applies; the walk must not count what that rule refuses.)
#
# THE RULE. A device leaf leaves the device by exactly one of these routes, and
# BOTH sides ask this one function (the peel with ``host_room=None``: the claim
# is its own first attempt; the vote with its remaining host room):
#
#   DEMOTE   backed (host copy): the node goes down, rows paid.
#   L3_DROP  not backed, but its pages are in L3 (``l3_present``, e.g. spilled by
#            W3): it leaves the tree WITHOUT a new L2 copy -- no arena claim is
#            made or priced. Same legality as the UD drop (local-PP floor, no write
#            in flight, host-only subtree clearable).
#   BACKUP   not backed, write_back: a claim into the host room. Certain for
#            nobody; the peel tries it, the vote prices it by ``host_room``. The
#            vote does not price a claim for an l3_present node (conservative:
#            its slots were freed by the spill under claim pressure; a payable
#            under-count only refuses, an over-count kills). Not a proven cause of
#            092933 (H3 above is unbelegt); kept so the L3 route is the one the
#            vote offers such a node.
#   UD_DROP  not backed, claim refused: dropped on the local-PP floor (UD / UD-H).
#   DELETE   write_through: deleted, only without surviving children (#841).
#
# AgreedRoomShort (``local_pp_room``) stays the stop for a REAL contradiction.
# ---------------------------------------------------------------------------

ROUTE_DEMOTE = "demote"
ROUTE_L3_DROP = "l3_drop"
ROUTE_BACKUP = "backup"
ROUTE_UD_DROP = "ud_drop"
ROUTE_DELETE = "delete"
#: routes after which the node is GONE from the tree (its parent sees no child)
DROP_ROUTES = (ROUTE_L3_DROP, ROUTE_UD_DROP, ROUTE_DELETE)


def host_node_local_ok(node) -> bool:
    """One host-only node UD-H may remove, ignoring what hangs below it: evicted,
    backed, no device lock, no host lock. (``_ud_clear_host_children``'s per-node
    test; ``pp_room_vote``'s walk composes the same test.)"""
    if not getattr(node, "evicted", False) or not getattr(node, "backuped", False):
        return False
    return not any(int(getattr(cd, "lock_ref", 0) or 0) > 0 or int(getattr(cd, "host_lock_ref", 0) or 0) > 0
                   for cd in (getattr(node, "component_data", None) or ()))


def demoted_node_clearable(node, children_clearable: bool) -> bool:
    """A device node the peel DEMOTES (backed) is, afterwards, a host-only node:
    UD-H of its parent may clear it iff ``host_node_local_ok`` then holds -- no
    host lock (a load-back reads its host rows) and its children clearable. The
    vote's walk asks this where the peel will ask ``host_node_local_ok`` (int24e
    092933: nodes 194/195, demoted, MAMBA host_lock_ref=1, blocked 181's drop)."""
    return bool(children_clearable) and not node_host_locked(node)


def host_children_clearable(node) -> bool:
    """Every node below ``node`` is a clearable host-only node (UD-H's
    all-or-nothing test); no children: True."""
    stack = list(getattr(node, "children", {}).values())
    while stack:
        d = stack.pop()
        if not host_node_local_ok(d):
            return False
        stack.extend(d.children.values())
    return True


def device_exit_route(node, *, write_back: bool, local_floor: bool, ongoing, host_room: Optional[int],
                      children_clearable: bool, surviving_children: bool, tokens: int) -> Optional[str]:
    """How a DEVICE LEAF (every device child already gone) leaves the device, or
    None when it cannot now. ``children_clearable``: every child ends up a
    clearable host-only node (an UD-H subtree clear would succeed);
    ``surviving_children``: a child outlives the peel (write_through's delete
    refuses with one, #841). ``host_room``: tokens of host room a claim may
    take (vote), None = the peel itself (the claim is attempted)."""
    if getattr(node, "backuped", False):
        return ROUTE_DEMOTE
    if not write_back:
        return None if surviving_children else ROUTE_DELETE
    droppable = (local_floor and not node_in_flight(ongoing, node) and not node_host_locked(node)
                 and children_clearable)
    l3 = bool(getattr(node, "l3_present", False))
    if l3 and droppable:
        return ROUTE_L3_DROP
    if host_room is None:
        return ROUTE_BACKUP
    if not l3 and int(host_room) >= int(tokens):
        return ROUTE_BACKUP
    return ROUTE_UD_DROP if droppable else None


def route_refusal(node, *, write_back: bool, local_floor: bool, ongoing, children_clearable: bool,
                  surviving_children: bool) -> str:
    """Why :func:`device_exit_route` returned None (diagnostics only): the terms of the
    refusal, '+'-joined."""
    if not write_back:
        return "write_through_with_children" if surviving_children else "none"
    why = []
    if not local_floor:
        why.append("no_local_floor")
    if node_in_flight(ongoing, node):
        why.append("write_in_flight")
    if node_host_locked(node):
        why.append("host_locked")
    if not children_clearable:
        why.append("children_not_clearable")
    return "+".join(why) or "none"


def peel_route(tree, node, *, host_room: Optional[int] = None) -> Optional[str]:
    """The peel's side of :func:`device_exit_route` for ``node`` (a device leaf of
    ``tree``): the tree's write policy, floor scope and in-flight table.
    Switch ``FLLIPER_PDFLIP_PP_SLOT_FIDELITY`` off: no local floor, so the routes
    are the old ones (backup, no drop)."""
    cc = getattr(tree, "cache_controller", None)
    write_back = cc is not None and getattr(cc, "write_policy", None) == "write_back"
    local = bool(enabled() and getattr(tree, FLOOR_LOCAL_PP_ATTR, False))
    return device_exit_route(
        node, write_back=write_back, local_floor=local,
        ongoing=getattr(tree, "ongoing_write_through", None) or {}, host_room=host_room,
        children_clearable=host_children_clearable(node),
        surviving_children=bool(getattr(node, "children", None)),
        tokens=max(1, len(getattr(node, "key", None) or ())))


def note_l3_drop(tree, node, tokens: int) -> None:
    n = _sampled(tree, "_pdflip_sf_l3_drop")
    if n is not None:
        logger.warning(
            "ROOM-VOTE L3-DROP node=%s tokens=%d freed=%d (n=%d): a node whose L2 copy was spilled "
            "to L3 (l3_present) leaves the device without a new arena claim -- the vote and the "
            "peel read one evictability check (int24e 09:39:42Z: the vote paid it, the claim "
            "was refused)",
            getattr(node, "id", "?"), len(getattr(node, "key", []) or []), int(tokens), n)


def note_unbacked_drop(tree, node, tokens: int) -> None:
    n = _sampled(tree, "_pdflip_sf_unbacked_drop")
    if n is not None:
        logger.warning(
            "EVICT-UNBACKED-DROP node=%s tokens=%d freed=%d (n=%d): a write_back leaf whose "
            "backup was refused (arena full / parent un-backed) is dropped on the local-PP "
            "floor instead of blocking the whole chain behind it (NF rc12q: 249856 tokens behind "
            "one un-backable leaf)",
            getattr(node, "id", "?"), len(getattr(node, "key", []) or []), int(tokens), n)


def ensure_relief_provider(scheduler) -> None:
    """SF-X: register ONE rank-local extend relief provider for this process
    (``common.register_extend_relief_provider``). It acts only while the tree's
    floor is marked local-PP (tp group of one, pp>1: the floor is rank-local
    anyway, so a rank-local eviction cannot split a replica group); on a TP
    group it returns 0 without touching the tree. Evicts exactly the tokens
    asked, applies staged frees, logs ``EXTEND-RELIEF``."""
    if not enabled() or getattr(scheduler, "_sf_relief_registered", False):
        return
    tree = getattr(scheduler, "tree_cache", None)
    if tree is None:
        return

    def _relief(num_tokens: int) -> int:
        if not enabled() or not getattr(tree, FLOOR_LOCAL_PP_ATTR, False):
            return 0
        from flliper.srt.mem_cache.base_prefix_cache import EvictParams
        from flliper.srt.mem_cache import common as _c

        alloc = tree.token_to_kv_pool_allocator
        before = int(alloc.available_size())
        need = max(0, int(num_tokens) - before)
        res = tree.evict(EvictParams(num_tokens=need)) if need > 0 else None
        try:
            _c._flush_deferred_frees(alloc)
        except Exception:  # noqa: BLE001
            pass
        after = int(alloc.available_size())
        n = _sampled(tree, "_pdflip_sf_extend_relief")
        if n is not None:
            logger.warning(
                "EXTEND-RELIEF evicted=%d asked=%d avail %d->%d evictable_left=%d (n=%d): "
                "rank-local relief on the local-PP floor (tp group of one) -- the extend's "
                "own trigger read a floor the pass had already spent",
                int(getattr(res, "num_tokens_evicted", 0) or 0), int(num_tokens), before, after,
                int(tree.evictable_size()), n)
        return max(0, after - before)

    from flliper.srt.mem_cache.common import register_extend_relief_provider

    register_extend_relief_provider(_relief)
    scheduler._sf_relief_registered = True


# ---------------------------------------------------------------------------
# (b1) the budget head's position from the KEPT told
# ---------------------------------------------------------------------------


def kept_told(scheduler, req) -> Optional[int]:
    """The H91 kept told of ``req`` (same request object), or None. Switch off:
    None, the lookup is not made."""
    if not enabled():
        return None
    kept = getattr(scheduler, "_pdflip_told_kept", None)
    if not kept:
        return None
    entry = kept.get(str(getattr(req, "rid", "")))
    if entry is None or getattr(entry, "req", None) is not req:
        return None
    try:
        return int(entry.told)
    except (TypeError, ValueError, AttributeError):
        return None


# ---------------------------------------------------------------------------
# (b2) replan at the prefix move
# ---------------------------------------------------------------------------


class BudgetPlan(NamedTuple):
    rid: str
    pos: int
    end: int
    src: str
    width: int


def note_budget(scheduler, head, width: int) -> None:
    """Remember the position/width this pass's budget was sized from."""
    if not enabled():
        return
    try:
        setattr(scheduler, PLAN_ATTR, BudgetPlan(
            str(getattr(head.req, "rid", "")), int(head.pos), int(head.end),
            str(head.src), int(width)))
    except Exception:  # noqa: BLE001 -- an instrument never stops a pass
        pass


def clear_budget(scheduler) -> None:
    """Start of a pass's budget sizing: no plan is carried over from an earlier
    pass (a static / layer-split pass notes none). Switch off: nothing."""
    if not enabled():
        return
    try:
        setattr(scheduler, PLAN_ATTR, None)
    except Exception:  # noqa: BLE001
        pass


def replan_hook(scheduler, adder):
    """The #988 move hook for this pass's adder, or None (switch off, no plan
    policy, no plan noted this pass): ``hook(req, new_prefix_len)``. The plan is
    CONSUMED here -- bound into this adder's hook, gone from the scheduler -- so
    it never serves a second adder."""
    if not enabled() or getattr(scheduler, "_p_chunk_planner", None) is None:
        return None
    plan = getattr(scheduler, PLAN_ATTR, None)
    if plan is None:
        return None
    try:
        setattr(scheduler, PLAN_ATTR, None)
        setattr(adder, ADDER_INITIAL_ATTR, adder.rem_chunk_tokens)
    except Exception:  # noqa: BLE001
        return None
    box = [plan]

    def _hook(req, new_pos):
        return replan_after_move(scheduler, adder, req, new_pos, box)

    return _hook


def replan_after_move(scheduler, adder, req, new_pos: int, box) -> Optional[int]:
    """#988 moved ``req``'s prefix to ``new_pos``. If the pass budget was planned
    for this request at a different position, replan at ``new_pos`` and narrow
    the adder's chunk budget to the new plan's width. ``box`` holds this pass's
    plan. Returns the new budget, or None when nothing changed. Never raises."""
    try:
        plan = box[0]
        rid = str(getattr(req, "rid", ""))
        if plan is None or plan.rid != rid or plan.src == "chunked":
            return None
        new_pos = int(new_pos)
        if new_pos == plan.pos:
            return None
        rem = getattr(adder, "rem_chunk_tokens", None)
        initial = getattr(adder, ADDER_INITIAL_ATTR, None)
        if rem is None or initial is None:
            return None
        if getattr(adder, "can_run_list", None) or int(rem) != int(initial):
            n = _sampled(scheduler, "_pdflip_sf_replan_skipped")
            if n is not None:
                logger.warning(
                    "SF P-CHUNK REPLAN-AT-MOVE SKIPPED rid=%s planned_start=%d moved_to=%d "
                    "rem_chunk=%s initial=%s can_run=%d (n=%d): the pass budget was "
                    "already spent on another request; the next pass replans at the "
                    "executed position",
                    rid, plan.pos, new_pos, rem, initial,
                    len(getattr(adder, "can_run_list", None) or ()), n,
                )
            return None
        from flliper.srt.pdflip.p_chunk_policy import forward_budget

        planner = getattr(scheduler, "_p_chunk_planner", None)
        if planner is None:
            return None
        width = int(forward_budget(planner, req.rid, new_pos, plan.end, queued=True))
        cap = int(getattr(scheduler, "chunked_prefill_size", 0) or 0)
        if cap > 0:
            width = min(width, cap)
        new_budget = min(width, int(initial)) if width > 0 else int(initial)
        adder.rem_chunk_tokens = new_budget
        box[0] = plan._replace(pos=new_pos, width=new_budget)
        n = _sampled(scheduler, "_pdflip_sf_replan_n")
        if n is not None:
            logger.info(
                "SF P-CHUNK REPLAN-AT-MOVE rid=%s planned_start=%d (src=%s) moved_to=%d "
                "width %d -> %d (plan %d, never widened) (n=%d): the pass budget was "
                "planned for another start; #988 moved the prefix, so the plan is "
                "redone at the moved prefix on every rank alike",
                rid, plan.pos, plan.src, new_pos, int(initial), new_budget, width, n,
            )
        return new_budget
    except Exception as exc:  # noqa: BLE001 -- a plan must never stop a pass
        n = _sampled(scheduler, "_pdflip_sf_replan_err")
        if n is not None:
            logger.warning("SF P-CHUNK REPLAN-AT-MOVE failed (n=%d): %r", n, exc)
        return None
