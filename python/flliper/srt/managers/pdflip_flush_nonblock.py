"""B1 (30.09., NF y4i): the HiCache publish leaves the D->P flip's quiesce.

THE MEASUREMENT (boot y4i 09301011, 12 D->P flips, front clock ms-exact).
Between ``PDFLIP-FLIP begin`` and the sleep legs (``PDFLIP-FLIP-ORDER``) the
median is 233 ms. In 11 of 12 flips the front's first quiesce
``/flush_cache`` was REFUSED ("not-idle because: hicache_backup(n)"): the
flush first ran ``#1470 FLUSH-PUBLISH`` over the nodes D's finished requests
left un-backed (write_back policy, n = 1 -> 6 over the boot, ~28 ms per
node, waited_ms 32-166), then the sidecar store writes of those nodes were
still in flight, the idle verdict said no, the front slept its poll interval
and asked again (7-22 ms). ``begin -> quiesce done`` 53-410 ms, median 98.
Before every one of those flips D was IDLE: last D decode -> flip begin
137-333 ms (the client's gap plus the front's tokenizer count), longer than
the publish in each flip.

TWO PARTS, each behind its own switch (default off until metal), groups
``FLLIPER_PDFLIP_FLUSH_NONBLOCK_GROUPS`` (default ``D``):

1. D-IDLE-PUBLISH (``FLLIPER_PDFLIP_ENABLE_D_IDLE_PUBLISH``). The user's
   bubble publisher (pdflip_bubble_publish, decision 2026-09-16: "nothing is
   left for the flip, the forwards are not competed with") runs only in the
   PP event loop; group D (TP only) never had a bubble, so every node its
   requests left un-backed waited for the flip. ``idle_publish`` issues the
   same ``publish_unbacked_sweep`` from ``Scheduler.on_idle`` -- no batch,
   nothing running, nothing waiting, not dormant (the KV pool is mapped).
   The gate reads REPLICATED state only (TP ranks reach ``on_idle`` in the
   same loop iteration: the request broadcast is their lockstep), no wall
   clock, a node cap per pass (default 1: an arriving request waits at
   most one node's issue). The write-through acks drain through the
   loop's own ``check_hicache_events``; nothing waits here.

2. FLUSH-QUIESCE-NONBLOCK (``FLLIPER_PDFLIP_ENABLE_FLUSH_QUIESCE_NONBLOCK``).
   The front's quiesce ``/flush_cache`` (the tp-group verdict path) no longer
   refuses because of the group's OWN HiCache publish still in flight
   (``hicache_write_through`` / ``hicache_backup``) when nothing else blocks
   on any rank (a second group reduction). It answers "quiesced" and leaves
   the tree, the pools and the in-flight writes exactly as they are; the
   RESET moves to the sleep leg, which already runs, before any page goes:
   (a) ``_pdflip_drain_hicache_before_sleep`` -- the group loop over
   ``check_hicache_events`` until every write-through and store write is
   acked (W120 by name at its bound), (b) the idle assert, (c)
   ``flush_cache(zero_kv=False)`` -- #1470 publish of anything still
   un-backed, joined, #1470b store-queue join, ANCHOR-LOST, the H81 arena
   reference release inside ``tree_cache.reset()``, the reset -- and only
   then (d) the kv_cache pause. No node is dropped un-backed and no page is
   unmapped before its copy landed: the order is the sleep leg's own.
   In this mode the quiesce's FLUSH-PUBLISH issues its sweep without the
   blocking write-back drain; a sweep that leaves nodes un-issued (pin
   budget, arena) keeps the blocking #1470 loop for that flush.
"""

from __future__ import annotations

import os
import logging
import time
from typing import Iterable, List, Optional

logger = logging.getLogger(__name__)

LINE_IDLE = "PDFLIP-D-IDLE-PUBLISH"
LINE_QUIESCE = "PDFLIP-FLUSH-NONBLOCK"

#: the idle-blocker clauses (Scheduler.idle_blockers) that are the group's
#: own HiCache publish -- write-throughs and the store writes their acks
#: issue. Every other clause (running, waiting, grammar, load-back,
#: prefetch, ...) still refuses the quiesce.
SOFT_PREFIXES = ("hicache_write_through(", "hicache_backup(")


def _group() -> str:
    import os

    return str(os.environ.get("FLLIPER_PDFLIP_GROUP", "")).strip().upper()


def _group_on() -> bool:
    from flliper.srt.environ import envs

    groups = {g.strip().upper() for g in str(envs.FLLIPER_PDFLIP_FLUSH_NONBLOCK_GROUPS.get() or "").split(",")}
    return _group() in groups


def idle_publish_on() -> bool:
    from flliper.srt.environ import envs

    return bool(envs.FLLIPER_PDFLIP_ENABLE_D_IDLE_PUBLISH.get()) and _group_on()


def quiesce_nonblock_on() -> bool:
    from flliper.srt.environ import envs

    return bool(envs.FLLIPER_PDFLIP_ENABLE_FLUSH_QUIESCE_NONBLOCK.get()) and _group_on()


def hard_blockers(blockers: Iterable[str]) -> List[str]:
    """The clauses that are NOT the group's own HiCache publish."""
    return [b for b in blockers if not str(b).startswith(SOFT_PREFIXES)]


# ---- part 1: the idle publisher --------------------------------------------------


def idle_gate(scheduler) -> Optional[str]:
    """None = publish now; else why not (replicated state only)."""
    if not getattr(scheduler, "enable_hierarchical_cache", False):
        return "no_hicache"
    tc = getattr(scheduler, "tree_cache", None)
    if tc is None or not hasattr(tc, "publish_unbacked_sweep"):
        return "no_sweep"
    if getattr(scheduler, "pdflip_dormant", False):
        return "dormant"  # the KV pool is paused: nothing may read its pages
    if getattr(scheduler, "_engine_paused", False):
        return "paused"
    rb = getattr(scheduler, "running_batch", None)
    if rb is not None and not rb.is_empty():
        return "running"
    if len(getattr(scheduler, "waiting_queue", ()) or ()) > 0:
        return "waiting"
    if getattr(scheduler, "chunked_req", None) is not None or getattr(scheduler, "anchor_tails", None):
        return "chunked"
    return None


def idle_publish(scheduler) -> Optional[dict]:
    """One bounded sweep from ``on_idle`` (part 1). Returns the sweep's stats,
    or None when the gate is shut / the switch is off. Never raises."""
    if not idle_publish_on():
        return None
    if idle_gate(scheduler) is not None:
        return None
    # A pass that left nothing to issue sleeps until the next forward: the
    # sweep is a walk over the tree, and an idle loop at its zero-poll rung
    # would repeat it every iteration. ``forward_ct`` is replicated (TP
    # lockstep), so every rank skips the same passes -- no wall clock.
    fct = int(getattr(scheduler, "forward_ct", 0) or 0)
    if getattr(scheduler, "_pdflip_idle_publish_clean_at", None) == fct:
        return None
    from flliper.srt.environ import envs

    try:
        cap = max(1, int(envs.FLLIPER_PDFLIP_D_IDLE_PUBLISH_MAX_ISSUE.get()))
    except Exception:  # noqa: BLE001
        cap = 1
    try:
        stats = scheduler.tree_cache.publish_unbacked_sweep(max_issue=cap) or {}
    except Exception as e:  # noqa: BLE001 -- a publisher never takes the loop down
        logger.warning("%s raised %s: %s", LINE_IDLE, type(e).__name__, e)
        return None
    issued = int(stats.get("issued", 0) or 0)
    if int(stats.get("unbacked", 0) or 0) <= issued or issued == 0:
        # everything this pass saw is issued, nothing was un-backed, or
        # nothing COULD be issued (pin budget / arena refusals): no repeat
        # walk before the next forward -- the flip's flush keeps its #1470
        # loop for whatever is left
        scheduler._pdflip_idle_publish_clean_at = fct
    n = int(getattr(scheduler, "_pdflip_idle_publish_n", 0) or 0) + 1
    scheduler._pdflip_idle_publish_n = n
    if issued:
        total = int(getattr(scheduler, "_pdflip_idle_publish_issued", 0) or 0) + issued
        scheduler._pdflip_idle_publish_issued = total
        logger.info(
            "%s sweep=%d issued=%d unbacked=%s pending=%s refused=%s cumulative_issued=%d "
            "(D idle: nothing running or waiting; the nodes the flip's flush would "
            "have published, published now)",
            LINE_IDLE, n, issued, stats.get("unbacked"), stats.get("pending"),
            stats.get("refused"), total)
    return stats


# ---- part 1b: the publisher between D decode rounds (PUBLISH-SWEEP-BG) ----------

LINE_BG = "PDFLIP-PUBLISH-SWEEP-BG"


def bg_publish_on() -> bool:
    from flliper.srt.environ import envs

    if str(os.environ.get("FLLIPER_PDFLIP_DUAL_LAYOUT", "")).strip() == "1":
        # y9e port (flip/INT8 line): the dual layout never flips, its D publishes through the
        # told/anchor path; the default-ON sweep must not change the dual default (Dual-Boots
        # b9j-b9l ran without it). Dual stays as it was.
        return False
    return bool(envs.FLLIPER_PDFLIP_PUBLISH_SWEEP_BG.get()) and _group_on()


def bg_gate(scheduler) -> Optional[str]:
    """None = a background pass may run now; else why not. Replicated state
    only (every TP rank reaches the hook once per loop iteration)."""
    if not getattr(scheduler, "enable_hierarchical_cache", False):
        return "no_hicache"
    tc = getattr(scheduler, "tree_cache", None)
    if tc is None or not hasattr(tc, "publish_unbacked_sweep"):
        return "no_sweep"
    if getattr(scheduler, "pdflip_dormant", False):
        return "dormant"  # the KV pool is paused: nothing may read its pages
    if getattr(scheduler, "_engine_paused", False):
        return "paused"
    rb = getattr(scheduler, "running_batch", None)
    if rb is None or rb.is_empty():
        return "idle"  # D-IDLE-PUBLISH's case (on_idle)
    if len(getattr(scheduler, "waiting_queue", ()) or ()) > 0:
        return "waiting"  # an admission is pending: no publish work before it
    if getattr(scheduler, "chunked_req", None) is not None or getattr(scheduler, "anchor_tails", None):
        return "chunked"
    return None


def bg_publish_tick(scheduler) -> Optional[dict]:
    """One bounded background pass from the scheduler's group-uniform point
    (after ``check_hicache_events``). The nodes the flip's flush would publish
    (finished requests' tails, left un-backed by the write_back policy) are
    published a few at a time WHILE D decodes, so the flush finds a small
    backlog instead of 110-121 nodes (1.9-2.8 s of issue, INT8 boot
    4cf740ad50). Cadence = ``forward_ct`` (replicated under TP lockstep, no
    wall clock); at most ``..._MAX_ISSUE`` node(s) per pass; only nodes no
    running request references whose parent is backed
    (``publish_unbacked_sweep(background=True)``). The flush is untouched: what
    is still un-backed at the flip is published there exactly as before, and
    the write-throughs this issues are drained by the loop's own
    ``check_hicache_events`` / the flush's join. Never raises."""
    if not bg_publish_on():
        return None
    try:
        from flliper.srt.environ import envs

        every = max(1, int(envs.FLLIPER_PDFLIP_PUBLISH_SWEEP_BG_EVERY.get()))
        cap = max(1, int(envs.FLLIPER_PDFLIP_PUBLISH_SWEEP_BG_MAX_ISSUE.get()))
        max_tokens = max(0, int(envs.FLLIPER_PDFLIP_PUBLISH_SWEEP_BG_MAX_TOKENS.get()))
    except Exception:  # noqa: BLE001
        every, cap, max_tokens = 64, 1, 8192
    fct = int(getattr(scheduler, "forward_ct", 0) or 0)
    if fct <= 0 or fct % every != 0:
        return None
    if getattr(scheduler, "_pdflip_bg_publish_last_fct", None) == fct:
        return None  # an iteration that ran no forward: once per forward count
    if bg_gate(scheduler) is not None:
        return None
    scheduler._pdflip_bg_publish_last_fct = fct
    try:
        stats = scheduler.tree_cache.publish_unbacked_sweep(
            max_issue=cap, background=True, bg_max_tokens=max_tokens) or {}
    except Exception as e:  # noqa: BLE001 -- a publisher never takes the loop down
        logger.warning("%s raised %s: %s", LINE_BG, type(e).__name__, e)
        return None
    issued = int(stats.get("issued", 0) or 0)
    n = int(getattr(scheduler, "_pdflip_bg_publish_n", 0) or 0) + 1
    scheduler._pdflip_bg_publish_n = n
    if issued:
        total = int(getattr(scheduler, "_pdflip_bg_publish_issued", 0) or 0) + issued
        scheduler._pdflip_bg_publish_issued = total
        # one marker line per 30 s (review 1270 #8; monotonic, log only)
        _now = time.monotonic()
        if _now - float(getattr(scheduler, "_pdflip_bg_publish_log_t", -1e9)) >= 30.0:
            scheduler._pdflip_bg_publish_log_t = _now
            logger.info(
                "%s pass=%d issued=%d unbacked=%s skipped_bg=%s pending=%s refused=%s "
                "issue_ms=%s cumulative_issued=%d (D decoding: finished requests' un-backed "
                "nodes published between rounds, not in the flip's flush)",
                LINE_BG, n, issued, stats.get("unbacked"), stats.get("skipped_bg"),
                stats.get("pending"), stats.get("refused"), stats.get("issue_ms"), total)
    return stats


# ---- part 2: the quiesce that does not wait for its own publish ------------------


def quiesce_verdict(scheduler, group_idle: bool, tp_group_verdict: bool) -> Optional[str]:
    """Part 2: the detail of a QUIESCED answer, or None (the flush proceeds
    as before: reset if idle, refuse otherwise).

    Called on EVERY rank of the group in the same pass (the tp-group verdict
    path) whenever ``group_idle`` is False -- that bit is already reduced over
    the group, so the second reduction below is posted uniformly."""
    if group_idle or not tp_group_verdict or not quiesce_nonblock_on():
        return None
    if not getattr(scheduler, "enable_hierarchical_cache", False):
        return None
    pp_size = int(getattr(getattr(scheduler, "ps", None), "pp_size", 1) or 1)
    if pp_size > 1:
        return None  # the PP vote lap is its own protocol; D is TP only
    own = list(scheduler.idle_blockers())
    hard = hard_blockers(own)
    (any_hard,) = scheduler.tree_cache.hicache_group_max(
        [int(bool(hard))], label="flush_cache/nonblock_hard"
    )
    if int(any_hard) != 0:
        return None
    return (
        "%s quiesced: group has no blocker but its own HiCache publish in flight "
        "(this rank %s); tree, pools and in-flight writes kept -- the sleep leg drains "
        "them (group loop), publishes the rest (#1470), joins the store queue (#1470b) "
        "and resets BEFORE the kv_cache pause" % (LINE_QUIESCE, ", ".join(own) or "none")
    )


LINE_S104_DEFER = "S104 FLUSH-DEFER-RESET"
_S104_N = [0]


def s104_defer_reset_verdict(scheduler, group_idle: bool, tp_group_verdict: bool,
                             unbacked_left: int = 0, flip_quiesce: bool = False) -> Optional[str]:
    """S104: the detail of a DEFERRED reset, or None (the flush resets as before).

    THE METAL (rc22 boot ..._9c40fe40a2_1020_094747, D TP=3 Form A, 09:54:35Z). The front's quiesce
    ``/flush_cache`` ran its #1470 loop against a KV arena held in full by the trees (``S83 VICTIM-CENSUS
    site=flush_end slots=6485 free=0 pinned=6485 tree_slots=6485 unpinned_complete=0``): every claim refused
    (``#1421 BACKUP-REFUSED why=arena_claim`` / ``parent_unbacked``), ``#1470 FLUSH-PUBLISH issued=0
    unbacked_left=14 in_flight_after=0``. Nothing was in flight, so the group verdict was IDLE, B1 had nothing to
    hand over, and THIS flush reset the tree: ``#1470 FLUSH-UNBACKED site=flush nodes=14 tokens=10432``, five park
    ends ``S85 PARK-END-UNBACKED lost_tokens=1536/1792/2496/1920/1920``. The sleep leg's S86 SLEEP-L3-RELEASE --
    the step built for exactly this arena (it gives back the slots L3 holds, agreed over the group, BEFORE the
    #1470 loop claims) -- ran one second later on the empty tree (``PDFLIP-RELEASE-FLUSH-DEDUP ... the quiesce
    flush's reset is current``; rc22: 3 of 17 sleeps, rc24: 1 of 9, rc25: 1 of 2 took this quiesce reset).

    THE RULE. On group D, in the tp-group verdict path (every rank runs this flush in the same pass, ``group_idle``
    already reduced over the group), an idle quiesce asks the group once (one MAX, posted uniformly) whether any
    rank's tree still holds device KV the reset would drop (``unbacked_census``: neither ``backuped`` nor
    ``l3_present`` -- the FLUSH-UNBACKED predicate, so a byteless anchor or a host-backed node never defers). If so
    the reset moves to the sleep leg, exactly as B1 moves it for a publish in flight: the tree, pools and arena
    references stay, the answer is "quiesced" (200), and the release leg runs S86 (release the L3-proven slots),
    its own #1470 loop (now with room), the store join and the reset. Nothing is kept back and no reserve is made:
    the reset that would have dropped the nodes simply runs after the step that frees their room. Other groups,
    a rank-local verdict and an idle tree with nothing to drop answer None (the flush as before).

    Review A1 (nf-review-s104-1020): ONLY the front's flip quiesce (``flip_quiesce``, the replicated
    ``FlushCacheReqInput.pdflip_flip_quiesce`` the front sets in ``Front.quiesce``) may defer -- a sleep leg follows
    it. Every other /flush_cache (operator, test, removing wrong data, user order 15.10.) resets as before; it
    posts no collective here."""
    if not flip_quiesce or not group_idle or not tp_group_verdict:
        return None
    if _group() != "D" or not getattr(scheduler, "enable_hierarchical_cache", False):
        return None
    pp_size = int(getattr(getattr(scheduler, "ps", None), "pp_size", 1) or 1)
    if pp_size > 1:
        return None
    tc = getattr(scheduler, "tree_cache", None)
    gm = getattr(tc, "hicache_group_max", None)
    if not callable(gm):
        return None
    try:
        cen = unbacked_census(tc)
    except Exception:  # noqa: BLE001 - a census that cannot answer counts as "drops something": defer (fail safe)
        logger.warning("%s census raised: this rank votes defer", LINE_S104_DEFER, exc_info=True)
        cen = {"nodes": 1, "tokens": 0, "ids": []}
    mine = int(cen.get("nodes", 0) or 0)
    (any_drop,) = gm([int(mine > 0)], label="flush_cache/s104_defer_reset")
    if int(any_drop) == 0:
        return None
    _S104_N[0] += 1
    return (
        "%s n=%d quiesced: the group is idle but a reset now would drop device KV with no copy (this rank "
        "nodes=%d tokens=%d ids=%s unbacked_left=%d) -- tree, pools and arena references kept; the sleep leg "
        "runs S86 SLEEP-L3-RELEASE, publishes the rest (#1470), joins the store queue (#1470b) and resets "
        "BEFORE the kv_cache pause" % (LINE_S104_DEFER, _S104_N[0], mine, int(cen.get("tokens", 0) or 0),
                                       list(cen.get("ids", []) or [])[:16], int(unbacked_left or 0))
    )


def quiesce_sweep_blocking(scheduler, stats: dict, tp_group_verdict: bool) -> bool:
    """Part 2: must the quiesce's FLUSH-PUBLISH keep its blocking #1470 loop
    (drain, re-sweep)? False = one issuing pass was enough: nothing is left
    un-issued, the in-flight writes are the sleep leg's to drain."""
    if not tp_group_verdict or not quiesce_nonblock_on():
        return True
    if not getattr(scheduler, "running_batch", None) or not scheduler.running_batch.is_empty():
        return True
    left = int((stats or {}).get("unbacked", 0) or 0) - int((stats or {}).get("issued", 0) or 0)
    return left > 0


# ---- 1403 (NF nf9 1004_105315): the flush sweep's no-progress memo ------------


def sweep_memo_on() -> bool:
    """FLLIPER_PDFLIP_FLUSH_SWEEP_MEMO: skip the quiesce poll's re-sweep while a
    measured no-progress round is still in force. Default off until metal."""
    from flliper.srt.environ import envs

    return bool(envs.FLLIPER_PDFLIP_FLUSH_SWEEP_MEMO.get())


def inflight_count(tree_cache) -> int:
    """Replicated in-flight count: write-throughs plus store writes."""
    if tree_cache is None:
        return 0
    return (len(getattr(tree_cache, "ongoing_write_through", {}) or {})
            + len(getattr(tree_cache, "ongoing_backup", {}) or {}))


def sweep_no_progress(stats: dict) -> bool:
    """A round that left nodes un-backed, issued nothing and held nothing in
    flight: by kvs2 W3 (no pin frees, no write lands) a repeat round cannot do
    better -- across polls this is the 1.2 s sweep per quiesce poll measured on
    nf9 (P.log 144297: unbacked=18 issued=0 sweep_ms=1221.1, every poll)."""
    return (int((stats or {}).get("issued", 0) or 0) == 0
            and int((stats or {}).get("pending", 0) or 0) == 0
            and int((stats or {}).get("unbacked", 0) or 0) > 0)


def arena_free(tree_cache):
    """Room state of the host KV arena as a stamp part (None if the pool does not say). Cheap, no tree walk.
    S13 A2: NOT ``available_size()`` -- on the arena pool that is ``slots - claimed`` (COMPLETE slots count as
    free, arena_pool.py #1440b), so a COMPLETE slot turning free moves nothing there. Arena pool: the tuple
    ``(complete, claimed)`` of ``arena.stats()`` (free = slots - complete - claimed, every claim, completion and
    free moves it). Pools without an arena: ``available_size()``."""
    try:
        pool = getattr(getattr(tree_cache, "cache_controller", None), "mem_pool_host", None)
        arena = getattr(pool, "arena", None)
        if arena is not None:
            st = arena.stats()
            return (int(st["complete"]), int(st["claimed"]))
        fn = getattr(pool, "available_size", None)
        return int(fn()) if callable(fn) else None
    except Exception:  # noqa: BLE001 - a stamp part never takes the poll down
        return None


def sweep_state_stamp(tree_cache, scheduler=None) -> tuple:
    """h16v S12 (Nachreview 2, V2) + S13 A1/A2: in-flight count, arena room (complete, claimed) AND the
    scheduler's ``forward_ct`` -- any state change moves the stamp. ``forward_ct`` moves at every executed
    batch in EVERY event loop (Scheduler._run_batch_forward), so a batch between two polls voids the memo also on D
    (pp=1 overlap loop, which never calls quiesce_end('batch'))."""
    return (inflight_count(tree_cache), arena_free(tree_cache), int(getattr(scheduler, "forward_ct", -1)))


def sweep_memo_set(scheduler, tree_cache) -> None:
    scheduler._pdflip_flush_sweep_memo = sweep_state_stamp(tree_cache, scheduler)


def sweep_memo_clear(scheduler) -> None:
    """h16v S12 V2: the stamp belongs to ONE quiesce on ONE tree; reset, batch and quiesce end drop it."""
    scheduler._pdflip_flush_sweep_memo = None


def sweep_memo_hit(scheduler, tree_cache) -> bool:
    """True while in-flight count and free arena room stand where the no-progress round left them -- under the
    sweep's own gate (nothing running/waiting/chunked) every un-backing and pin release moves the count, every
    arena claim/free moves the room, so an unchanged stamp means the tree and the arena are exactly as futile as
    they were. Cleared at quiesce end (reset or batch) and at the tree reset."""
    stamp = getattr(scheduler, "_pdflip_flush_sweep_memo", None)
    return stamp is not None and stamp == sweep_state_stamp(tree_cache, scheduler)


# ---- F5c (NF-PGAP 1016): what goes into the reset un-backed, named -----------------

LINE_UNBACKED = "#1470 FLUSH-UNBACKED"


def unbacked_census(tree_cache) -> dict:
    """Device nodes whose KV is neither on the host (``backuped``) nor in the store (``l3_present``) -- the reset
    drops their KV and a later request recomputes it. The sweep's own predicate (``publish_unbacked_sweep``);
    ``pending`` = of those, a write-through still in flight. Read-only."""
    out = {"nodes": 0, "tokens": 0, "pending": 0, "ids": []}
    walk = getattr(tree_cache, "_collect_all_nodes", None)
    if not callable(walk):
        return out
    try:
        from flliper.srt.mem_cache.unified_radix_cache import BASE_COMPONENT_TYPE
    except Exception:  # noqa: BLE001
        return out
    root = getattr(tree_cache, "root_node", None)
    for node in walk():
        if node is root or getattr(node, "evicted", False) or getattr(node, "backuped", False) \
                or getattr(node, "l3_present", False):
            continue
        try:
            if node.component_data[BASE_COMPONENT_TYPE].value is None:
                continue
        except Exception:  # noqa: BLE001
            continue
        out["nodes"] += 1
        out["tokens"] += len(getattr(node, "key", ()) or ())
        if getattr(node, "write_through_pending_id", None) is not None:
            out["pending"] += 1
        if len(out["ids"]) < 16:
            out["ids"].append(getattr(node, "id", "?"))
    return out


def _dropped_by_reset(node, root) -> bool:
    """``unbacked_census``'s predicate for one node: device KV neither on the host nor in the store."""
    if node is None or node is root or getattr(node, "evicted", False) or getattr(node, "backuped", False) \
            or getattr(node, "l3_present", False):
        return False
    try:
        from flliper.srt.mem_cache.unified_radix_cache import BASE_COMPONENT_TYPE

        return node.component_data[BASE_COMPONENT_TYPE].value is not None
    except Exception:  # noqa: BLE001
        return False


#: S85 (rc19 boot ..._10e9b2b492_1019_230333, pdflip-106-2655): the park-end chain of a parked request went into
#: the sleep reset un-backed -- the KV arena was pinned in full (complete=6485 pinned=6485), every write-through of
#: the chain refused (``#1421 BACKUP-REFUSED why=arena_claim node=6298 depth=199232``), and the reset dropped it
#: (``#1470 FLUSH-UNBACKED nodes=27 ... ids=[6298, 6305, ..., 6333]``, 6333 = the PARK-END-ANCHOR-FIRST node,
#: depth 206080). The resume read the store to 198592 and 7488 tokens were recomputed; no line said so: S80 asks
#: only at P's wake probe of a read that ran during the flip, S81 watches only CLAIMED share-0 slots (these pages
#: were never claimed). Named here, at the moment of the loss, per park-end node.
LINE_PARK_END = "S85 PARK-END-UNBACKED"


def park_end_unbacked(tree_cache) -> list:
    """S85: every park-end node (``_pdflip_park_end_nodes``, PARK-END-ANCHOR-FIRST) whose chain loses KV with this
    reset: the store then ends at the START of the shallowest dropped node of the chain -- everything from there to
    the park-end depth is unreadable for the resume (exact, from the tree; read-only)."""
    out = []
    ends = list(getattr(tree_cache, "_pdflip_park_end_nodes", None) or ())
    root = getattr(tree_cache, "root_node", None)
    depth_of = getattr(tree_cache, "pdflip_node_depth", None)
    if not ends or not callable(depth_of):
        return out
    for end in ends:
        if getattr(end, "evicted", False):
            continue
        chain = []
        n = end
        while n is not None and n is not root:
            chain.append(n)
            n = getattr(n, "parent", None)
        dropped = [n for n in chain if _dropped_by_reset(n, root)]
        if not dropped:
            continue
        top = dropped[-1]                                      # the shallowest dropped node of the chain
        depth = int(depth_of(end))
        store_end = int(depth_of(top)) - len(getattr(top, "key", ()) or ())
        out.append({"rid": getattr(end, "pdflip_park_rid", "?"), "node": getattr(end, "id", "?"),
                    "depth": depth, "store_end": store_end, "lost_tokens": depth - store_end,
                    "unbacked_nodes": len(dropped)})
    return out


_UNBACKED_N = [0, 0]   # resets seen, resets with un-backed KV


def note_unbacked_at_reset(tree_cache, site: str = "flush", paused: int = -1, open_ms: float = -1.0) -> dict:
    """F5c: one line per reset that drops un-backed KV (and the denominator every 64th clean reset), so the store
    loss of a sleep reset is measured, never silent."""
    c = unbacked_census(tree_cache)
    _UNBACKED_N[0] += 1
    if c["nodes"]:
        _UNBACKED_N[1] += 1
    if c["nodes"] or _UNBACKED_N[0] % 64 == 1:
        (logger.warning if c["nodes"] else logger.info)(
            "%s site=%s nodes=%d tokens=%d pending=%d ids=%s resets=%d with_unbacked=%d paused_sweeps=%d "
            "quiesce_open_ms=%.0f (device KV neither on the host nor in the store goes with this reset -- a later "
            "request recomputes it; paused_sweeps = bubble sweeps the open quiesce left to the #1470 loop)",
            LINE_UNBACKED, site, c["nodes"], c["tokens"], c["pending"], c["ids"], _UNBACKED_N[0], _UNBACKED_N[1],
            int(paused), float(open_ms))
    try:
        c["park_end"] = park_end_unbacked(tree_cache) if c["nodes"] else []
    except Exception:  # noqa: BLE001 - an instrument never blocks the reset
        c["park_end"] = []
    for pe in c["park_end"]:
        logger.warning(
            "%s rid=%s node=%s depth=%d store_end=%d lost_tokens=%d unbacked_nodes=%d site=%s (the park-end chain of "
            "a parked request goes into this reset un-backed: its resume reads the store only to store_end -- P or D "
            "recomputes lost_tokens; target 0)", LINE_PARK_END, pe["rid"], pe["node"], pe["depth"], pe["store_end"],
            pe["lost_tokens"], pe["unbacked_nodes"], site)
    return c
