"""REPROBE-1023: D's store arm re-asks a SHORT store answer on a group event, not in every round.

Metal 27B AutoRound (boot ..._a45112428e_1023_144422.D.log, D = TP2 on 2x3080, TP0 and TP1 alike),
14:49:15-14:49:35Z: two P hand-backs, pdflip-1-3 (N=118937) and pdflip-1-4 (N=107847), waited 20.1 s in D's
queue, refused 37x each by the lifetime gate (``D-ADMIT-AFTER-REFUSALS refusals=37 gates={lifetime:37}``,
price 123034 against a budget of ~64800 while pdflip-1-1/1-2 decoded). In EVERY decode round D asked the store
for both of them again: ``#1472 READ-TRACE asked=512 readable=0 why=no-file`` and ``#1035 R13 EMPTY KV PREFIX``
for ``keys=118709`` and ``keys=107619`` -- exactly the anchor depth minus the 227-token host match, i.e. the span
``Scheduler._pdflip_local_store_matches`` (#1317 X-STORE-VOTE, the S64 fresh store arm) asks. THE CALLER IS
THAT ARM, not ``_retry_deferred_prefetches``: ``S64 STORE-ARM COST probes=64 ms_sum=13497.3 ms_mean=210.9``
at 14:49:31 on TP0 (13510.4 on TP1), two probes per round, ~420 ms of the 400-500 ms ``rest`` of every
``D-DECODE-GAP cause=other``. Over the whole boot: 1984 probes, 129.0 s of scheduler thread, and
``S64 STORE-ARM STALE`` 0 times -- not one fresh probe returned anything the previous one had not.

THE ROOT. S64 S-ARM (NF rc15, 19.10.) made the arm ask the store fresh in every vote for every SHORT answer,
because the store can fill under an unchanged (generation, local_match, span) key. That is right while the store
moves and pure cost while it does not: a refused hand-back whose answer is 0 pages is hashed (27B page_size=1:
118709 keys) and looked up again in every round for as long as it waits.

THE RULE (SOLL R11 -- no rank-local decision): a short cached answer is re-asked on a pass of the GROUP's
schedule, and that schedule is built only from replicated values:

* ``tick`` -- the number of packed head-vote reduces this process completed. Every rank of the group calls the
  same ``all_reduce`` the same number of times (it is a collective), so the tick is identical on every rank.
* ``group_epoch`` -- the MIN over the group of each rank's STORE EPOCH, carried as one more slot of the SAME
  packed MIN reduce (no new collective). A rank's epoch counts the storage writes it has seen land (the backup
  acks its tree drains -- D's own write-through, turn-end L2 copies, park/anchor demotes). MIN = the epoch moved
  on ALL ranks, which is exactly when the group's MIN store answer can have grown (each rank asks its own shard).
* ``group_vote`` -- the group's MIN store match for the rid, read back from the same reduce.

After the reduce (``after_reduce``) every rank updates the per-rid schedule identically: an event -- the group
epoch differs from the one the rid's schedule was taken under (also when it moved while the rid was not voted),
the group vote changed, or the rid is new to the schedule -- resets the interval to 1 (the next pass asks again);
a scheduled re-ask that changed nothing doubles it (1, 2, 4, ... ``max_passes``). An ABSENT group vote (-1: some
rank cast none this pass, e.g. an S42 hold) is no event and leaves the schedule as it is, on every rank alike. The vote site
(``held``) reuses the cached answer while ``tick`` is before the rid's next scheduled pass. So two ranks with the
same queue ask on the same passes, and a rank can never ask alone because of a value only it has seen.

The bound ``max_passes`` stays for what no D rank observes (group P writing a span D is waiting on): the arm is
never older than that many passes -- the price against S64's every-pass ask, named (review 1023): such a write is
seen 1..``max_passes`` passes later, never later than that. On a group where a rank drains no backup acks (Form A
token cut: TP0 writes nothing) the MIN epoch does not move; the vote-change and bound paths still hold there. What this module does NOT change: the cache key (a changed local match or
binding generation asks at once, as before -- that term is rank-local by its own nature and the MIN reduce absorbs
it; the probe carries no collective), the same-pass reuse, the covered-span reuse, and the probe itself. The #580
group vote of ``prefetch_from_storage`` is not on this path at all.

``FLLIPER_PDFLIP_S64_REPROBE_EPOCH`` (default ON; 0 = ask every pass as S64 did),
``FLLIPER_PDFLIP_S64_REPROBE_MAX_PASSES`` (default 32).
"""

from __future__ import annotations

import logging
import os
from typing import Any, Dict, Optional, Sequence

logger = logging.getLogger(__name__)

ENV = "FLLIPER_PDFLIP_S64_REPROBE_EPOCH"
MAX_ENV = "FLLIPER_PDFLIP_S64_REPROBE_MAX_PASSES"
DEFAULT_MAX_PASSES = 32

#: the tree attribute of this rank's store epoch (storage writes seen land)
TREE_EPOCH_ATTR = "_pdflip_store_write_epoch"
#: scheduler attributes: the replicated reduce tick and the group epoch of the last reduce
TICK_ATTR = "_pdflip_store_reduce_tick"
GROUP_EPOCH_ATTR = "_pdflip_store_group_epoch"
#: the per-request schedule (written only in ``after_reduce``, from replicated values)
SCHED_ATTR = "_pdflip_store_reprobe"

_STATS = {"asked": 0, "held": 0, "events": 0, "log_n": 0, "raised": 0}


def enabled(env=None) -> bool:
    e = os.environ if env is None else env
    return (e.get(ENV, "1") or "1").strip().lower() not in ("0", "false", "no", "off")


def max_passes(env=None) -> int:
    e = os.environ if env is None else env
    try:
        v = int(str(e.get(MAX_ENV, "") or DEFAULT_MAX_PASSES).strip())
    except ValueError:
        return DEFAULT_MAX_PASSES
    return max(1, v)


def note_store_write(tree: Any, n: int = 1) -> None:
    """This rank saw ``n`` storage writes land (backup acks drained). Never raises."""
    try:
        if int(n) > 0:
            setattr(tree, TREE_EPOCH_ATTR, int(getattr(tree, TREE_EPOCH_ATTR, 0) or 0) + int(n))
    except Exception:  # noqa: BLE001 -- an epoch never breaks a drain
        pass


def local_epoch(sched: Any) -> int:
    """This rank's store epoch, the slot it puts into the packed MIN reduce."""
    try:
        return int(getattr(getattr(sched, "tree_cache", None), TREE_EPOCH_ATTR, 0) or 0)
    except Exception:  # noqa: BLE001
        return 0


def build_epoch_payload(sched: Any) -> list:
    """ONE slot, on every rank, whatever the rank holds -- the payload width never depends on the rank."""
    return [local_epoch(sched)]


def tick(sched: Any) -> int:
    return int(getattr(sched, TICK_ATTR, 0) or 0)


def held(sched: Any, req: Any, env=None) -> bool:
    """Vote site: True = reuse the cached short answer for ``req`` on this pass (the group's schedule says the
    store is not re-asked yet). Reads only the replicated schedule and the replicated tick. A request with no
    schedule (never voted through a reduce) is asked."""
    if not enabled(env):
        return False
    s = getattr(req, SCHED_ATTR, None)
    if not s:
        return False
    try:
        # a group epoch that moved since the schedule was taken asks now (replicated: the last reduce's MIN)
        ge = getattr(sched, GROUP_EPOCH_ATTR, None)
        if ge is not None and int(s.get("epoch", -1)) != int(ge):
            return False
        return tick(sched) < int(s["next"])
    except Exception:  # noqa: BLE001
        return False


def note_vote(asked: bool) -> None:
    _STATS["asked" if asked else "held"] += 1


def after_reduce(sched: Any, canonical: Sequence[str], group_store_lens: Sequence[int], group_epoch: int,
                 env=None) -> None:
    """Right after the packed MIN reduce, on every rank alike: advance the tick, take the group epoch, and
    re-schedule every canonical rid this rank holds from the GROUP's values only. Never raises (it sits on the
    collective path; a failure leaves the schedule as it was -- the next pass asks at the latest at ``next``)."""
    try:
        t = tick(sched) + 1
        setattr(sched, TICK_ATTR, t)
        setattr(sched, GROUP_EPOCH_ATTR, int(group_epoch))
        if not enabled(env):
            return
        cap = max_passes(env)
        by_rid = {}
        for r in getattr(sched, "waiting_queue", None) or ():
            by_rid[str(getattr(r, "rid", ""))] = r
        for i, rid in enumerate(canonical):
            req = by_rid.get(str(rid))
            if req is None or i >= len(group_store_lens):
                continue
            g = int(group_store_lens[i])
            if g < 0:
                # ABSENT (-1, the MIN-neutral "no vote"): some rank cast no store vote this pass (an S42 hold,
                # a rank without the rid) -- not an event, the schedule stands, the same on every rank
                continue
            s = getattr(req, SCHED_ATTR, None)
            # the epoch the schedule was taken under, per rid: a move while the rid was not voted counts too
            if not s or int(s.get("epoch", -1)) != int(group_epoch) or int(s.get("vote", g)) != g:
                interval = 1
                if s:
                    _STATS["events"] += 1
            elif t - 1 >= int(s["next"]):   # this pass was a scheduled re-ask and nothing changed
                interval = min(cap, 2 * int(s["interval"]))
            else:                             # a held pass: the schedule stands
                continue
            setattr(req, SCHED_ATTR, {"next": t + interval - 1, "interval": interval, "vote": g,
                                      "epoch": int(group_epoch)})
        _log(sched, t, int(group_epoch))
    except Exception as e:  # noqa: BLE001 -- the collective path never raises from here
        _STATS["raised"] = _STATS.get("raised", 0) + 1
        n = _STATS["raised"]
        if n <= 5 or n % 256 == 0:
            logger.warning("S64 STORE-ARM REPROBE after_reduce raised n=%d %s: %s -- the schedules stand as they "
                           "were (every held rid is re-asked at the latest at its next scheduled pass)",
                           n, type(e).__name__, e)


def _log(sched: Any, t: int, epoch: int) -> None:
    if _STATS["asked"] + _STATS["held"] <= 0:
        return
    k = _STATS["log_n"] = _STATS["log_n"] + 1
    if k > 3 and k % 256 != 0:
        return
    logger.info(
        "S64 STORE-ARM REPROBE asked=%d held=%d events=%d tick=%d group_epoch=%d max_passes=%d (REPROBE-1023: a "
        "short store answer is re-asked on the group's schedule -- an epoch move, a changed group vote, or the "
        "back-off bound; denominator: every store-arm vote of a short answer, since process start)",
        _STATS["asked"], _STATS["held"], _STATS["events"], t, epoch, max_passes())


def stats() -> Dict[str, int]:
    return dict(_STATS)


def reset_stats() -> None:
    for k in _STATS:
        _STATS[k] = 0
