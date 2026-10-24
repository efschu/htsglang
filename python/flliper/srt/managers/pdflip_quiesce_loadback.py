"""QUIESCE-LOADBACK-1023: the flip quiesce drains finished load-back acks before it votes.

THE DEATH (NF rc36p, boot ..._1023_215911, D epoch 4 -> P, 22:29:25 -> 22:30:55Z).
The front parked D's running decode (``PARK-RUNNING parked=3 rids=[pdflip-4-207,
-218, -219] still_running=[]``, front.log 22:29:25,686) and polled ``/flush_cache``.
Every one of 1602 answers in the next 31 s was refused on every D rank with one
blocker, ``not-idle because: hicache_load_back(1)`` (D.log 22:29:25-22:29:56),
with ``#queue-req: 0, #running-req: 0``. The load-back belonged to an admission
that had been REVERTED a moment before the park (``D-ADMIT-REFUSED rid=pdflip-4-218
gate=trunc_len_zero ... site=admission_revert``; its prefix match had issued the
load-back, ``MAMBA-HOST-RESUME ... this match triggers load_back``). Its copy was
long done; its ack was not drained, because ``loading_check`` -- the ONLY place
``ongoing_load_back`` shrinks -- runs inside ``check_hicache_events``, i.e. inside a
PLANNED prefill pass, and a parked D plans nothing. After 30 s D's own awake
re-queue (``FLLIPER_PDFLIP_D_PARK_AWAKE_REQUEUE_S``) put the park back
(``PDFLIP-D-PARK requeue (awake) ['pdflip-4-207']`` 22:29:55); that first planned
pass drained the ack in the same second -- and admitted pdflip-4-207, which decoded
on (``running_batch, last_batch, overlap_result_queue``, 2091 refusals). At the
90 s quiesce deadline the front's two witnesses both said "busy", which
``witness_verdict`` reads as agreement, and the sleep RPC went out: W120 on every
rank (a non-HiCache blocker stops the sleep drain at once, by design), W29, RANK-DEATH.

THE FIX (this module). The idle vote of a flush is a statement about the group NOW,
so a finished load-back ack must be counted as finished: right before the vote, a
rank drains its finished load-back acks. ``UnifiedRadixCache.loading_check`` is
rank-local since #737 (``_count_ready_acks``: only this rank's own finished events,
no collective), so it may run here on any rank, in any pass; an unfinished copy
stays in ``ongoing_load_back`` and still votes not-idle. The same drain already
exists for P followers under the row authority
(``scheduler_pp_mixin.pdflip_loadback_drain_on_idle_vote``, 27B rc12z28); D (TP
group, pp_rank 0) was excluded from it. A tree whose ``loading_check`` carries a
collective (upstream ``HiRadixCache``: MIN all_reduce) does not declare
``LOADING_CHECK_RANK_LOCAL`` and is never polled from here -- a rank-local
condition in front of a collective is the desync this tree already died of.

Default ON (``FLLIPER_PDFLIP_QUIESCE_LOADBACK_DRAIN=0`` restores the old vote).
"""

from __future__ import annotations

import logging
import os

logger = logging.getLogger(__name__)

ENV = "FLLIPER_PDFLIP_QUIESCE_LOADBACK_DRAIN"
#: the class attribute a radix cache sets when its ``loading_check`` is rank-local
RANK_LOCAL_ATTR = "LOADING_CHECK_RANK_LOCAL"

_N = [0]


def enabled(env=None) -> bool:
    e = os.environ if env is None else env
    return str(e.get(ENV, "1")).strip() not in ("0", "false", "False", "no", "off")


def drain_finished_loadbacks(sched, env=None) -> int:
    """Drain this rank's FINISHED load-back acks before an idle vote.

    Returns how many entries left ``ongoing_load_back``. Never raises (a vote
    that cannot drain votes on the undrained state, as before) and never waits
    (an unfinished event stays and votes not-idle)."""
    if not enabled(env) or not getattr(sched, "enable_hierarchical_cache", False):
        return 0
    tc = getattr(sched, "tree_cache", None)
    pending = getattr(tc, "ongoing_load_back", None)
    if not pending:
        return 0
    if not getattr(type(tc), RANK_LOCAL_ATTR, False):
        return 0
    check = getattr(tc, "loading_check", None)
    if not callable(check):
        return 0
    before = len(pending)
    try:
        check()
    except Exception:  # noqa: BLE001 - the drain never breaks the flush
        logger.warning("PDFLIP-QUIESCE-LOADBACK-DRAIN raised; the vote reads the undrained state",
                       exc_info=True)
        return 0
    left = len(getattr(tc, "ongoing_load_back", None) or ())
    drained = before - left
    if drained > 0:
        _N[0] += 1
        if _N[0] <= 16 or _N[0] % 256 == 0:
            logger.info("PDFLIP-QUIESCE-LOADBACK-DRAIN drained=%d left=%d (n=%d): finished load-back "
                        "acks polled at the flush's idle vote -- a parked D plans no pass that would "
                        "poll them (QUIESCE-LOADBACK-1023)", drained, left, _N[0])
    return drained
