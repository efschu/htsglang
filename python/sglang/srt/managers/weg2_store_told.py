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


def _completed_prefix(tree, rid: str) -> int:
    """The HOST-TREE PREFIX this rank's terminated prefetch leaves behind, in
    tokens from position 0 -- NOT the loaded increment.

    Boot xsn119 (rid eb03bfc5): PP0's read completed 4095 tokens (anchor at
    4094) of which 64 were already in its host tree, so ``loaded`` said 4031;
    a follower told 4031 registered [0, 4031), found no anchor in range, and
    claimed 0 -- the named mismatch. The prefix is the uniform quantity
    (``#1175 completed_prefetch_tokens`` = matched + loaded on the tree that
    has it); the increment is rank-local history.
    """
    fn = getattr(tree, "completed_prefetch_tokens", None)
    if callable(fn):
        value = fn(rid)
        if value is not None:
            return int(value)
    return int(tree.prefetch_loaded_tokens_by_reqid.get(rid, 0) or 0)


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
        n = getattr(scheduler, "_weg2_store_told_intake_n", 0) + 1
        scheduler._weg2_store_told_intake_n = n
        if n <= _LOG_FIRST or n % _LOG_EVERY == 0:
            logger.info(
                "#1400 STORE-TOLD INTAKE rid=%s verdict=%s span=%s matched=%s (n=%d)",
                rid[:8],
                verdict,
                getattr(req, "_prefetch_span_tokens", None),
                getattr(req, "_prefetch_registered_prefix_len", None),
                n,
            )
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


def follower_limit_tokens(tree, told: int) -> int:
    """Token position up to which a follower registers so that its KEY count
    equals PP0's completed count. Under a bigram key (``is_eagle``: MTP draft
    on this form) n tokens make n-1 keys, and the completed count -- like the
    anchor index -- is in keys: boot xsn120, told=4095, follower registered
    4095 tokens = 4094 keys, anchor at key 4094 outside the range, claim 0."""
    bigram = 1 if bool(getattr(tree, "is_eagle", False)) else 0
    return int(told) + bigram


def _follower_register(scheduler, req, told: int) -> str:
    if told <= 0:
        return "declined:weg2_told_zero"
    verdict = scheduler._prefetch_kvcache(
        req, limit_tokens=follower_limit_tokens(scheduler.tree_cache, told)
    )
    if str(verdict).startswith("declined") and _local_prefix(req) >= int(told):
        # xsn155: the same shape with verdict 'declined:store_absent' -- the
        # follower held 94,206 tokens locally against told=53,246 (PP0's
        # smaller pool had evicted and re-read), its probe started past the
        # told span and found nothing to ask for. Any decline while the
        # local prefix covers told is satisfied, whatever the label.
        # Boot xsn141 (2026-09-16, the first with the shared arena): PP0's
        # device tree had evicted part of a prefix its followers still held
        # (PP0 carries 39 of 64 layers, so its pool of 487k tokens is the
        # first to evict), PP0 read 686 tokens from the store to reach
        # told=3615, and the followers -- already holding >= 3616 locally --
        # had nothing to fetch: prefetch_length 0 < threshold, "too_short",
        # declined, and admission raised the mismatch on own_prefix=0. A
        # follower that already HOLDS the told span has satisfied it; it
        # registers nothing and admits at told.
        satisfied = getattr(scheduler, "_weg2_store_told_satisfied", None)
        if satisfied is None:
            satisfied = scheduler._weg2_store_told_satisfied = {}
        satisfied[_rid(req)] = int(told)
        logger.info(
            "#1400 FOLLOWER SATISFIED LOCALLY rid=%s told=%d local_prefix=%d: "
            "nothing to read, this rank already holds the told span",
            rid8(req), int(told), _local_prefix(req),
        )
        return "satisfied:local_prefix"
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


def _local_prefix(req) -> int:
    """Tokens this rank already holds for ``req`` (device + host tier), the
    same two terms ``_prefetch_kvcache`` subtracts before it reads."""
    try:
        return int(len(getattr(req, "prefix_indices", []) or [])) + int(
            getattr(req, "host_hit_length", 0) or 0
        )
    except Exception:  # noqa: BLE001 - a double without the fields holds nothing
        return 0


def _anchored_pages_full_span(cc, ids, page_size: int):
    """#1416c (boot xsn174): ``store_presence_pages`` asks the store about
    the FIRST ``STORAGE_BATCH_SIZE`` (128) pages only -- a 98,550-token span
    whose anchor sits on its last page answered 0, so told was clamped to 0
    and P recomputed 98k tokens it had just read back from the arena. Ask
    the whole span, the way ``_storage_hit_query`` does: one
    ``batch_exists_v2`` over every page key with the tree's component
    transfers (the mamba anchor is the trailing-pages pool). None = the
    question could not be asked.
    """
    try:
        hashes = cc.get_hash_str(list(ids), None, page_size=page_size)
        if not hashes:
            return 0
        transfers = cc._presence_pool_transfers()
        backend = cc.storage_backend
        from sglang.srt.mem_cache.hicache_storage import HiCacheStorageExtraInfo
        extra = HiCacheStorageExtraInfo(prefix_keys=None)
        if transfers:
            return int(backend.batch_exists_v2(list(hashes), transfers, extra).kv_hit_pages or 0)
        return int(backend.batch_exists(list(hashes), extra) or 0)
    except Exception as exc:  # noqa: BLE001 - see the docstring
        logger.warning("#1416c full-span anchor probe unavailable: %r", exc)
        return None


def _anchor_clamp(scheduler, req, told: int) -> int:
    """#1416 (boots xsn159/162/167): the completed prefix counts KV pages; the
    admission match accepts a prefix only up to the deepest page that also
    carries a mamba anchor. PP0 published told=53247 from a host-budget-
    truncated read (no anchor inside the span), its own match then refused
    the whole span ("MambaComponent:absent"), and the followers -- whose
    presence probe IS anchor-clamped since #869b -- answered store_absent:
    told mismatch, rank exit. Ask the same anchor-clamped question here, so
    told never names a prefix no rank can admit. Unavailable probe = no
    clamp (the pre-#1416 number), never a silent zero.
    """
    if told <= 0:
        return int(told)
    try:
        cc = getattr(scheduler, "cache_controller", None) or getattr(
            getattr(scheduler, "tree_cache", None), "cache_controller", None
        )
        ids = getattr(req, "origin_input_ids", None)
        if cc is None or not callable(getattr(cc, "get_hash_str", None)) or not ids:
            return int(told)
        page_size = int(getattr(cc, "page_size", 1) or 1)
        pages = _anchored_pages_full_span(cc, list(ids[: int(told)]), page_size)
        if pages is None:
            # the full-span question could not be asked: no clamp (the
            # pre-#1416 number; #1419 caps every rank's match to told, so a
            # too-large told recomputes on all ranks alike, it never diverges)
            return int(told)
        anchored = min(int(told), pages * page_size)
        if anchored < int(told):
            logger.warning(
                "#1416 STORE-TOLD ANCHOR-CLAMP rid=%s completed=%d anchored=%d: the "
                "span beyond the deepest mamba anchor is not admissible on any rank",
                rid8(req), int(told), anchored,
            )
        return anchored
    except Exception as exc:  # noqa: BLE001 - a probe never breaks publication
        logger.warning("#1416 anchor clamp skipped for rid=%s: %r", rid8(req), exc)
        return int(told)


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
        told = _completed_prefix(tree, rid)
        clamped = _anchor_clamp(scheduler, req, told)
        if clamped != told:
            # #1416b (boot xsn169): PP0's OWN admission compares its recorded
            # completed prefix with told; a clamped told against the stale
            # record (told=0 vs own=4095) stopped PP0 itself. The record is
            # "what this rank can admit" -- clamp it with the same number.
            try:
                tree._prefetch_completed_tokens[rid] = int(clamped)
            except Exception:  # noqa: BLE001 - a tree without the dict keeps its number
                pass
        told = clamped
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
    outstanding). Otherwise this rank's loaded credit, after its completed
    prefix was checked equal to told -- or a named refusal."""
    told_map: Dict[str, int] = scheduler._weg2_store_told
    rid = _rid(req)
    told = told_map.get(rid)
    if told is None:
        note_skip(SKIP_TOLD_PENDING, rid)
        return None
    tree = scheduler.tree_cache
    # #1419: told bounds this rank's radix match (schedule_batch
    # _weg2_cap_key_limit) so no rank -- PP0 included -- admits more than told.
    try:
        req._weg2_prefix_cap = int(told)
    except Exception:  # noqa: BLE001
        pass
    satisfied = getattr(scheduler, "_weg2_store_told_satisfied", None) or {}
    if rid in satisfied:
        # registered nothing because it already held the span (see
        # _follower_register): admit at told, no read to wait for.
        satisfied.pop(rid, None)
        told_map.pop(rid, None)
        return 0
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
    own = _completed_prefix(tree, rid)
    credit = int(tree.pop_prefetch_loaded_tokens(rid) or 0)
    told_map.pop(rid, None)
    if own != told:
        raise Weg2StoreToldMismatch(
            f"#1400 STORE-TOLD MISMATCH rank pp={scheduler.ps.pp_rank} "
            f"rid={rid[:8]} told={told} own_prefix={own} own_loaded={credit}: "
            f"this rank's store "
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
    # The CREDIT is this rank's loaded increment (rides into
    # `storage_hit_length` / cached_tokens_storage, informational); the
    # uniform fact -- the prefix -- was just checked against told.
    return credit
