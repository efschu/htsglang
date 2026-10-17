"""B1 (30.09., NF y4i): the HiCache publish leaves the D->P flip's quiesce.

THE MEASUREMENT (boot y4i 09301011, 12 D->P flips, front clock ms-exact).
Between ``WEG2-FLIP begin`` and the sleep legs (``WEG2-FLIP-ORDER``) the
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
``SGLANG_WEG2_FLUSH_NONBLOCK_GROUPS`` (default ``D``):

1. D-IDLE-PUBLISH (``SGLANG_WEG2_ENABLE_D_IDLE_PUBLISH``). The user's
   bubble publisher (weg2_bubble_publish, decision 2026-09-16: "nothing is
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

2. FLUSH-QUIESCE-NONBLOCK (``SGLANG_WEG2_ENABLE_FLUSH_QUIESCE_NONBLOCK``).
   The front's quiesce ``/flush_cache`` (the tp-group verdict path) no longer
   refuses because of the group's OWN HiCache publish still in flight
   (``hicache_write_through`` / ``hicache_backup``) when nothing else blocks
   on any rank (a second group reduction). It answers "quiesced" and leaves
   the tree, the pools and the in-flight writes exactly as they are; the
   RESET moves to the sleep leg, which already runs, before any page goes:
   (a) ``_weg2_drain_hicache_before_sleep`` -- the group loop over
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

import logging
from typing import Iterable, List, Optional

logger = logging.getLogger(__name__)

LINE_IDLE = "WEG2-D-IDLE-PUBLISH"
LINE_QUIESCE = "WEG2-FLUSH-NONBLOCK"

#: the idle-blocker clauses (Scheduler.idle_blockers) that are the group's
#: own HiCache publish -- write-throughs and the store writes their acks
#: issue. Every other clause (running, waiting, grammar, load-back,
#: prefetch, ...) still refuses the quiesce.
SOFT_PREFIXES = ("hicache_write_through(", "hicache_backup(")


def _group() -> str:
    import os

    return str(os.environ.get("SGLANG_WEG2_GROUP", "")).strip().upper()


def _group_on() -> bool:
    from sglang.srt.environ import envs

    groups = {g.strip().upper() for g in str(envs.SGLANG_WEG2_FLUSH_NONBLOCK_GROUPS.get() or "").split(",")}
    return _group() in groups


def idle_publish_on() -> bool:
    from sglang.srt.environ import envs

    return bool(envs.SGLANG_WEG2_ENABLE_D_IDLE_PUBLISH.get()) and _group_on()


def quiesce_nonblock_on() -> bool:
    from sglang.srt.environ import envs

    return bool(envs.SGLANG_WEG2_ENABLE_FLUSH_QUIESCE_NONBLOCK.get()) and _group_on()


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
    if getattr(scheduler, "weg2_dormant", False):
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
    if getattr(scheduler, "_weg2_idle_publish_clean_at", None) == fct:
        return None
    from sglang.srt.environ import envs

    try:
        cap = max(1, int(envs.SGLANG_WEG2_D_IDLE_PUBLISH_MAX_ISSUE.get()))
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
        scheduler._weg2_idle_publish_clean_at = fct
    n = int(getattr(scheduler, "_weg2_idle_publish_n", 0) or 0) + 1
    scheduler._weg2_idle_publish_n = n
    if issued:
        total = int(getattr(scheduler, "_weg2_idle_publish_issued", 0) or 0) + issued
        scheduler._weg2_idle_publish_issued = total
        logger.info(
            "%s sweep=%d issued=%d unbacked=%s pending=%s refused=%s cumulative_issued=%d "
            "(D idle: nothing running or waiting; the nodes the flip's flush would "
            "have published, published now)",
            LINE_IDLE, n, issued, stats.get("unbacked"), stats.get("pending"),
            stats.get("refused"), total)
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
    """SGLANG_WEG2_FLUSH_SWEEP_MEMO: skip the quiesce poll's re-sweep while a
    measured no-progress round is still in force. Default off until metal."""
    from sglang.srt.environ import envs

    return bool(envs.SGLANG_WEG2_FLUSH_SWEEP_MEMO.get())


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
    """Free room of the host KV arena (None if the pool does not say). Cheap, no tree walk."""
    try:
        pool = getattr(getattr(tree_cache, "cache_controller", None), "mem_pool_host", None)
        fn = getattr(pool, "available_size", None)
        return int(fn()) if callable(fn) else None
    except Exception:  # noqa: BLE001 - a stamp part never takes the poll down
        return None


def sweep_state_stamp(tree_cache) -> tuple:
    """h16v S12 (Nachreview 2, V2): in-flight count AND free arena room -- any state change moves the stamp."""
    return (inflight_count(tree_cache), arena_free(tree_cache))


def sweep_memo_set(scheduler, tree_cache) -> None:
    scheduler._weg2_flush_sweep_memo = sweep_state_stamp(tree_cache)


def sweep_memo_clear(scheduler) -> None:
    """h16v S12 V2: the stamp belongs to ONE quiesce on ONE tree; reset, batch and quiesce end drop it."""
    scheduler._weg2_flush_sweep_memo = None


def sweep_memo_hit(scheduler, tree_cache) -> bool:
    """True while in-flight count and free arena room stand where the no-progress round left them -- under the
    sweep's own gate (nothing running/waiting/chunked) every un-backing and pin release moves the count, every
    arena claim/free moves the room, so an unchanged stamp means the tree and the arena are exactly as futile as
    they were. Cleared at quiesce end (reset or batch) and at the tree reset."""
    stamp = getattr(scheduler, "_weg2_flush_sweep_memo", None)
    return stamp is not None and stamp == sweep_state_stamp(tree_cache)


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
        from sglang.srt.mem_cache.unified_radix_cache import BASE_COMPONENT_TYPE
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
    return c
