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


# ---- part 1b: the publisher between D decode rounds (PUBLISH-SWEEP-BG) ----------

LINE_BG = "WEG2-PUBLISH-SWEEP-BG"


def bg_publish_on() -> bool:
    from sglang.srt.environ import envs

    return bool(envs.SGLANG_WEG2_PUBLISH_SWEEP_BG.get()) and _group_on()


def bg_gate(scheduler) -> Optional[str]:
    """None = a background pass may run now; else why not. Replicated state
    only (every TP rank reaches the hook once per loop iteration)."""
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
        from sglang.srt.environ import envs

        every = max(1, int(envs.SGLANG_WEG2_PUBLISH_SWEEP_BG_EVERY.get()))
        cap = max(1, int(envs.SGLANG_WEG2_PUBLISH_SWEEP_BG_MAX_ISSUE.get()))
    except Exception:  # noqa: BLE001
        every, cap = 16, 1
    fct = int(getattr(scheduler, "forward_ct", 0) or 0)
    if fct <= 0 or fct % every != 0:
        return None
    if getattr(scheduler, "_weg2_bg_publish_last_fct", None) == fct:
        return None  # an iteration that ran no forward: once per forward count
    if bg_gate(scheduler) is not None:
        return None
    scheduler._weg2_bg_publish_last_fct = fct
    try:
        stats = scheduler.tree_cache.publish_unbacked_sweep(max_issue=cap, background=True) or {}
    except Exception as e:  # noqa: BLE001 -- a publisher never takes the loop down
        logger.warning("%s raised %s: %s", LINE_BG, type(e).__name__, e)
        return None
    issued = int(stats.get("issued", 0) or 0)
    n = int(getattr(scheduler, "_weg2_bg_publish_n", 0) or 0) + 1
    scheduler._weg2_bg_publish_n = n
    if issued:
        total = int(getattr(scheduler, "_weg2_bg_publish_issued", 0) or 0) + issued
        scheduler._weg2_bg_publish_issued = total
        if total <= 10 or total % 20 == 0:
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
