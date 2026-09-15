"""#1400 (Weg 2, HiCache-Rueckweg D->P): PP0's store verdict TOLD on the wire.

WHY THIS EXISTS. On the carrierless PP form (group P of Weg 2: ``--pp-size 3
--tp-size 1``, ``pp_flip_counters`` hard-coded ``None`` since fb3631c434) every
term that let PP0 decide the prefix for its followers is disarmed by design:
the #631 row (``pp_row_carrier_present`` False), #1066's own-prefetch wait,
#1175's group completion. The consequence, measured on boot xsn116
(2026-09-15 17:32:17, rid e10588ca): every rank registers its HiCache storage
prefetch, no rank waits for it (``#973 PP0 PREFETCH WAIT DISARMED``, ``#969Z``),
and the hit that lands later is dropped as ``undistributable`` (#1245) so the
ranks stay equal -- P re-prefills a prefix that is byte-complete in the store
(``PHASE-PURITY STORE WITNESS ... state=unprobed ... loaded=0``,
``[#928 anchor] REFUSING resume ... re-prefilling``). D's leg 2 read the same
store fine (``cached_tokens=4314/4316``) because the TP path WAITS (policy
``timeout``) and MIN-reduces the loaded count across its ranks.

THE CARRIER IS THE REQUEST WIRE THAT ALREADY EXISTS. PP0 forwards every
pass's ``recv_reqs`` list to PP1, PP1 to PP2 (``_pp_send_pyobj_to_next_stage``,
``CHAN_REQ``), and the list is pickled as a whole, so a small control object
rides it for free and reaches PP2 one pass after PP1 (#1268 fix 1c rides the
same wire with ``Weg2IdleVoteReq``). The lag matches the plan lag: a follower
plans PP0's pass ``m`` batch at its own pass ``m+1``, exactly when a
``Weg2StoreTold`` sent at the top of PP0's pass ``m`` arrives.

THE PROTOCOL (one rid, in order):
  1. intake -- PP0 registers its prefetch as today and HOLDS the rid; a
     follower registers NOTHING and holds the request (``declined:weg2_held``).
  2. top of a PP0 pass, BEFORE the forward -- for every held rid whose prefetch
     has terminated (``check_prefetch_progress`` True; collective-free on
     tp_size 1, which is the only form this arms on), PP0 reads the loaded
     count WITHOUT popping it, stores ``told``, and appends
     ``Weg2StoreTold(rid, told)`` to the outgoing list. PP0's own admission
     loop admits a request only once its told is stored, i.e. never before the
     verdict has been put on the wire.
  3. follower absorb, AFTER the forward, BEFORE dispatch -- the told objects
     leave ``recv_reqs`` (``process_input_requests`` has no handler for them),
     the told is stored, and the held request now registers a prefetch for
     EXACTLY ``told`` tokens (``_prefetch_kvcache(req, limit_tokens=told)``).
     Same page keys, same length, same store: the follower's host tree ends
     where PP0's does, so ``host_hit_length`` and the load-back extent are
     uniform BY CONTENT and no truncation of a Mamba-anchored prefix is ever
     needed.
  4. admission on every rank -- no told: skip (``weg2_store_told_pending``).
     Told present: wait (bounded) for THIS rank's own prefetch to terminate,
     pop its loaded count, and REFUSE BY NAME when it differs from told
     (``Weg2StoreToldMismatch``): a follower that could not load what PP0
     admitted cannot compute the rows PP0's batch omits, and a silent
     divergence here is the W27 width split of weg2rg3. Ranks are never
     allowed to disagree quietly (memory: RAENGE-NIE-UNEINS).

WHAT IT COSTS. One pass of admission latency per request on P (register at
pass n, publish at the earliest at n+1), plus the followers' store read, which
starts one pass after PP0's verdict and is waited for at their admission
(serialised down the pipe, ~one read per stage). Against a re-prefill of the
whole prefix that is the cheap side.

NOT TOUCHED. ``pp_size <= 1`` (group D, every non-PP boot), any ``tp_size > 1``
stage (``check_prefetch_progress`` carries a TP all_reduce there and a
rank-local wait loop would call it a rank-dependent number of times -- the
#580 class), and every form where a row carrier exists (the #631 machinery
governs there). Kill switch: ``SGLANG_WEG2_STORE_TOLD=0``.
"""

from __future__ import annotations

import logging
import os
import time
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional

logger = logging.getLogger(__name__)

ENV_ARMED = "SGLANG_WEG2_STORE_TOLD"
#: admission skip-census key on every rank while PP0's verdict is outstanding.
SKIP_TOLD_PENDING = "weg2_store_told_pending"
#: #915 intake-partition term for a follower that registers nothing at intake.
GATE_HELD = "weg2_held"
#: hard cap on a follower's wait for its own store read at admission. The
#: prefetch policy (``timeout`` on this form) terminates the read long before;
#: this only bounds a stuck storage thread, and it stays below the ring-commit
#: budget (120 s) so the wedge is named here rather than there.
WAIT_CAP_S = 60.0
_LOG_FIRST = 8
_LOG_EVERY = 256


@dataclass
class Weg2StoreTold:
    """PP0's store verdict for one request: the page-aligned token count its
    storage prefetch loaded (0 when nothing was registered or nothing hit)."""

    rid: str
    told: int


class Weg2StoreToldMismatch(RuntimeError):
    """This rank's own store read does not reproduce PP0's told count."""


def env_armed() -> bool:
    return os.environ.get(ENV_ARMED, "1").strip().lower() not in (
        "0",
        "false",
        "no",
        "off",
    )


def armed(scheduler) -> bool:
    """Resolved ONCE per scheduler (form and kill switch are boot constants);
    a value that could flip mid-pass would split the ranks."""
    cached = getattr(scheduler, "_weg2_store_told_armed", None)
    if cached is not None:
        return cached
    value = _resolve_armed(scheduler)
    scheduler._weg2_store_told_armed = value
    if value:
        scheduler._weg2_store_told = {}
        scheduler._weg2_store_held = {}
        logger.warning(
            "#1400 STORE-TOLD ARMED rank pp=%s: carrierless PP form with HiCache "
            "storage -- PP0 publishes its store verdict per request on the "
            "request wire; followers register exactly the told span and every "
            "rank admits only on a told verdict.",
            getattr(getattr(scheduler, "ps", None), "pp_rank", "?"),
        )
    return value


def _resolve_armed(scheduler) -> bool:
    if not env_armed():
        return False
    if not getattr(scheduler, "enable_hicache_storage", False):
        return False
    ps = getattr(scheduler, "ps", None)
    if ps is None:
        return False
    if int(getattr(ps, "pp_size", 1) or 1) <= 1:
        return False
    if int(getattr(ps, "tp_size", 1) or 1) != 1:
        return False
    from sglang.srt.managers.pp_admission_congruence import pp_row_carrier_present

    if pp_row_carrier_present(scheduler):
        return False
    return True


def is_pp0(scheduler) -> bool:
    """Only meaningful once :func:`armed` said True (``ps`` then exists)."""
    return int(scheduler.ps.pp_rank) == 0


def _rid(req) -> str:
    return str(getattr(req, "rid", ""))


def intake(scheduler, req, note_gate: Callable[[str], None]) -> str:
    """The intake step. PP0: register as today and hold. Follower: hold only;
    the registration happens in :func:`follower_absorb` with PP0's told."""
    held: Dict[str, Any] = scheduler._weg2_store_held
    rid = _rid(req)
    if int(scheduler.ps.pp_rank) == 0:
        verdict = scheduler._prefetch_kvcache(req)
        held[rid] = req
        return verdict
    told = scheduler._weg2_store_told.get(rid)
    if told is not None:
        # The verdict arrived before the request did (never on the ring's
        # order, but a re-queued request can find its told already stored):
        # register now with the told span.
        return _follower_register(scheduler, req, told)
    held[rid] = req
    note_gate(GATE_HELD)
    return f"declined:{GATE_HELD}"


def _follower_register(scheduler, req, told: int) -> str:
    if told <= 0:
        return "declined:weg2_told_zero"
    verdict = scheduler._prefetch_kvcache(req, limit_tokens=int(told))
    if not str(verdict).startswith("issued"):
        logger.warning(
            "#1400 FOLLOWER REGISTRATION DECLINED rid=%s told=%d verdict=%s: this "
            "rank cannot load what PP0 admitted; admission will refuse by name "
            "unless the read still completes.",
            rid8(req),
            int(told),
            verdict,
        )
    return verdict


def rid8(req) -> str:
    return _rid(req)[:8]


def pp0_publish(scheduler, recv_reqs: List) -> List:
    """Top of a PP0 pass, before the forward: turn terminated prefetches of
    held rids into ``Weg2StoreTold`` objects appended to the outgoing list.
    Returns the list to SEND; the caller keeps dispatching ``recv_reqs``."""
    held: Dict[str, Any] = scheduler._weg2_store_held
    if not held:
        return recv_reqs
    told_map: Dict[str, int] = scheduler._weg2_store_told
    tree = scheduler.tree_cache
    queued = {_rid(r) for r in scheduler.waiting_queue}
    out: List[Weg2StoreTold] = []
    for rid in list(held):
        req = held[rid]
        if rid not in queued:
            # Left the queue without an admission (abort, deferral elsewhere);
            # a re-queue holds it again through intake.
            held.pop(rid, None)
            continue
        if getattr(req, "prefetch_deferred", None) is not None:
            # A12.2 deferral: PP0 will re-issue; the verdict is not final.
            continue
        if not tree.check_prefetch_progress(rid):
            continue
        told = int(tree.prefetch_loaded_tokens_by_reqid.get(rid, 0) or 0)
        told_map[rid] = told
        held.pop(rid, None)
        out.append(Weg2StoreTold(rid=rid, told=told))
        n = getattr(scheduler, "_weg2_store_told_published", 0) + 1
        scheduler._weg2_store_told_published = n
        if n <= _LOG_FIRST or n % _LOG_EVERY == 0:
            logger.info(
                "#1400 STORE-TOLD PUBLISHED rid=%s told=%d (n=%d held_left=%d): "
                "PP0's terminated store prefetch loaded this many page-aligned "
                "tokens; the followers register exactly this span.",
                rid[:8],
                told,
                n,
                len(held),
            )
    if not out:
        return recv_reqs
    return list(recv_reqs) + out


def follower_absorb(scheduler, recv_reqs: List) -> List:
    """After the forward, before dispatch: take the told objects off the list,
    store them, and register the held requests' prefetch with the told span."""
    if not any(isinstance(r, Weg2StoreTold) for r in recv_reqs):
        return recv_reqs
    told_map: Dict[str, int] = scheduler._weg2_store_told
    held: Dict[str, Any] = scheduler._weg2_store_held
    rest = []
    for item in recv_reqs:
        if not isinstance(item, Weg2StoreTold):
            rest.append(item)
            continue
        rid = str(item.rid)
        told = int(item.told)
        told_map[rid] = told
        req = held.pop(rid, None)
        n = getattr(scheduler, "_weg2_store_told_absorbed", 0) + 1
        scheduler._weg2_store_told_absorbed = n
        if req is not None:
            verdict = _follower_register(scheduler, req, told)
        else:
            verdict = "held:not_yet_queued"
        if n <= _LOG_FIRST or n % _LOG_EVERY == 0:
            logger.info(
                "#1400 STORE-TOLD ABSORBED rank pp=%s rid=%s told=%d verdict=%s (n=%d)",
                scheduler.ps.pp_rank,
                rid[:8],
                told,
                verdict,
                n,
            )
    return rest


def admission(scheduler, req, note_skip: Callable[[str, Any], None]) -> Optional[int]:
    """The admission gate on every rank. ``None`` = skip this pass (verdict
    outstanding). Otherwise the told count, which is also this rank's own
    loaded count -- or a named refusal."""
    told_map: Dict[str, int] = scheduler._weg2_store_told
    rid = _rid(req)
    told = told_map.get(rid)
    if told is None:
        note_skip(SKIP_TOLD_PENDING, rid)
        return None
    tree = scheduler.tree_cache
    deadline = time.monotonic() + WAIT_CAP_S
    waited = False
    while not tree.check_prefetch_progress(rid):
        waited = True
        if time.monotonic() > deadline:
            raise Weg2StoreToldMismatch(
                f"#1400 STORE-TOLD WAIT EXCEEDED rank pp={scheduler.ps.pp_rank} "
                f"rid={rid[:8]} told={told}: this rank's own store read did not "
                f"terminate within {WAIT_CAP_S:g}s; the prefetch policy should "
                f"have cut it long before, so the storage thread is stuck."
            )
        time.sleep(0.002)
    own = int(tree.pop_prefetch_loaded_tokens(rid) or 0)
    told_map.pop(rid, None)
    if own != told:
        raise Weg2StoreToldMismatch(
            f"#1400 STORE-TOLD MISMATCH rank pp={scheduler.ps.pp_rank} "
            f"rid={rid[:8]} told={told} own_loaded={own}: this rank's store "
            f"read does not reproduce PP0's verdict, so its prefix would "
            f"diverge from the batch PP0 built (the W27 width split of "
            f"weg2rg3). Refusing by name instead of planning a different pass."
        )
    if waited:
        n = getattr(scheduler, "_weg2_store_told_waited", 0) + 1
        scheduler._weg2_store_told_waited = n
        if n <= _LOG_FIRST or n % _LOG_EVERY == 0:
            logger.info(
                "#1400 STORE-TOLD WAITED rank pp=%s rid=%s told=%d (n=%d): the own "
                "read terminated inside the admission wait.",
                scheduler.ps.pp_rank,
                rid[:8],
                told,
                n,
            )
    return told
