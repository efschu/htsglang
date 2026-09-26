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

from sglang.srt.managers import weg2_told_fallback as _fb
from sglang.srt.weg2 import p_twin_defer as _twin
from sglang.srt.weg2 import prefix_trace as _pt

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
    #: #1416e (paced, SGLANG_WEG2_TOLD_PACED): True = READ-AHEAD only -- the
    #: followers register their read of the told span now, but the request's
    #: MEMBERSHIP arrives later as :class:`Weg2StoreAdmit`. False = the
    #: pre-#1416e single-phase told (read + membership in one object).
    paced: bool = False
    #: TK: True = ``told`` is ABSOLUTE (PP0's registered head + its store span,
    #: TW's twin form for every request, SGLANG_WEG2_TOLD_ABSOLUTE); a follower
    #: compares its own registered head + its own span. False = the span-
    #: relative #1400 record (the host insert is rooted at last_host_node).
    absolute: bool = False
    #: TK (#1442 PP0-authoritative): digest of the hand-off chain PP0's read
    #: used (``handoff_keys.chain_digest``; "" = own hashes). The follower
    #: adopts it instead of deciding from its own file read. None = an old
    #: sender (the follower decides itself, as before).
    keys_digest: Optional[str] = None


@dataclass
class Weg2StoreAdmit:
    """#1416e: PP0's membership verdict for a PACED told -- PP0 admits the
    request in the pass that puts this on the wire, every follower in the
    pass that absorbs it (the lag of the single-phase told, unchanged)."""

    rid: str
    told: int


class Weg2StoreToldTwin(Weg2StoreTold):
    """TW (weg2.p_twin_defer): a told for a fork twin released after its
    sibling finished. ``told`` is ABSOLUTE -- the head PP0's registration
    matched locally plus the store span -- so a follower compares its own
    registered head plus its own span against it. Only ever sent with
    SGLANG_WEG2_P_TWIN_DEFER on; a class attribute, so the pickled fields
    are the parent's."""

    twin = True


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
        # #1416e: resolved ONCE, read on PP0 only -- a follower follows the
        # `paced` flag of the object PP0 put on the wire, never its own env,
        # so a launcher that armed the switch on one rank only cannot split
        # the group.
        scheduler._weg2_told_paced_on = _paced_env()
        # PF: the group told=0 fallback -- PP0's switch, effective only on
        # the paced form (only there does PP0 hold admission for an Admit).
        scheduler._weg2_told_fallback_on = bool(
            scheduler._weg2_told_paced_on and _fb.env_on()
        )
        if _fb.env_on() and not scheduler._weg2_told_paced_on:
            logger.warning(
                "PF %s=1 WITHOUT %s: no effect -- the group told=0 fallback "
                "needs the paced form", _fb.ENV_FALLBACK, ENV_PACED,
            )
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
        if _twin.intake_defer(scheduler, req):
            # TW: a fork twin of a request in flight on P -- held WITHOUT a
            # store read until that sibling finished (pp0_publish releases).
            held[rid] = req
            return _twin.VERDICT_DEFERRED
        verdict = scheduler._prefetch_kvcache(req)
        held[rid] = req
        if getattr(scheduler, "_weg2_told_paced_on", False):
            # #1416e: PP0's own read time is the pacing window's measure.
            _pace_intake_t(scheduler).setdefault(rid, _clock())
        n = getattr(scheduler, "_weg2_store_told_intake_n", 0) + 1
        scheduler._weg2_store_told_intake_n = n
        if _log_due(n):
            logger.info(
                "#1400 STORE-TOLD INTAKE rid=%s verdict=%s span=%s matched=%s (n=%d)",
                _rt(rid),
                verdict,
                getattr(req, "_prefetch_span_tokens", None),
                getattr(req, "_prefetch_registered_prefix_len", None),
                n,
            )
        return verdict
    told = scheduler._weg2_store_told.get(rid)
    if told is None:
        # #1416e: a paced read-ahead that arrived before the request did.
        told = (getattr(scheduler, "_weg2_told_early", None) or {}).get(rid)
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


def prefix_cap_tokens(tree, told: int) -> int:
    """TK (#1419 with told > 0): the RAW-token cap that lets the radix match
    reach exactly ``told`` KEYS. ``_weg2_cap_key_limit`` feeds it to
    ``RadixKey(limit=...)`` and the tree's bigram view of a raw limit L has
    L-1 keys: a cap of ``told`` stopped one key short of the anchor told
    names, and the match fell back to an earlier anchor (or 0). Same +1 as
    :func:`follower_limit_tokens`; told 0 still matches nothing."""
    return follower_limit_tokens(tree, told)


def _follower_register(scheduler, req, told: int) -> str:
    if getattr(scheduler, "_weg2_fb_follower", None) is not None:
        # PF: whatever this registration's outcome (issued, satisfied,
        # declined), its read state is what PP0's fallback asked for.
        _fb.follower_note_registered(scheduler, req)
    _adopt_keys(scheduler, req)
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
    """The rid as a log field: 8 characters, the FULL rid under the prefix
    trace (SGLANG_WEG2_PREFIX_TRACE, IN 26.09.)."""
    return _pt.rid_text(_rid(req))


def _rt(rid) -> str:
    return _pt.rid_text(rid)


def _log_due(n: int) -> bool:
    """First _LOG_FIRST, then every _LOG_EVERY-th -- every one under the
    prefix trace (one line per request and event, never per pass)."""
    return _pt.sampled(n, _LOG_FIRST, _LOG_EVERY)


def _local_prefix(req) -> int:
    """Tokens this rank already holds for ``req`` (device + host tier), the
    same two terms ``_prefetch_kvcache`` subtracts before it reads."""
    try:
        return int(len(getattr(req, "prefix_indices", []) or [])) + int(
            getattr(req, "host_hit_length", 0) or 0
        )
    except Exception:  # noqa: BLE001 - a double without the fields holds nothing
        return 0


#: #1416d switch (default OFF = the pre-#1416d probe, byte-identical): the
#: told anchor clamp hashes the span the way the FETCH hashes it.
ENV_TREE_KEY = "SGLANG_WEG2_TOLD_PROBE_TREE_KEY"


def _tree_key_probe_armed() -> bool:
    return os.environ.get(ENV_TREE_KEY, "0").strip().lower() in (
        "1", "true", "yes", "on",
    )


def _probe_key(scheduler, req, ids, told: int):
    """#1416d (27B agent boots 0925/0926, every P leg cached_tokens=0): the
    probe below hashed ``origin_input_ids[:told]`` as plain UNIGRAM ids, while
    the store read it clamps (``prefetch_from_storage``) keys the span as the
    tree does -- ``RadixKey(..., is_bigram=tree.is_eagle)``, bigram on every
    EAGLE/DFLASH/NEXTN group and on P under SGLANG_HICACHE_BIGRAM_KEYS=1
    (#1233: page 0 bigram -> c90c5e64dea9, unigram -> f9ec16cdf22e). The
    probe therefore asked the arena about keys nobody writes
    (``#1439 ARENA-PRESENT ... leading_complete=0 first_stem=f9ec16cd``
    right after the fetch's ``leading_complete=22934 first_stem=c90c5e64``),
    answered 0 and clamped every told to 0 since #1416 (xsn171 on): P
    re-prefilled prefixes it had just read back. Ask with the fetch's key:
    ``told`` counts KEYS, so a bigram span needs ``told + 1`` raw tokens
    (the same +1 as ``follower_limit_tokens``); P's handed-over page keys
    (#1442), when registered for this rid, replace the covered prefix
    exactly as ``_storage_hit_query`` does."""
    from sglang.srt.mem_cache.radix_cache import RadixKey

    tree = getattr(scheduler, "tree_cache", None)
    bigram = bool(getattr(tree, "is_eagle", False))
    raw = list(ids[: int(told) + (1 if bigram else 0)])
    key = RadixKey(raw, extra_key=getattr(req, "extra_key", None), is_bigram=bigram)
    return key, bigram


def _handoff_keys(rid: str):
    """The scheduler's registry entry: P's keys SLICED AT THE MATCHED LENGTH
    of the last registration (``keys_for_span``). Not a chain from token 0 --
    the probe below takes :func:`_handoff_chain` instead."""
    try:
        from sglang.srt.managers.cache_controller import WEG2_HANDOFF_PAGE_KEYS

        return WEG2_HANDOFF_PAGE_KEYS.get(rid)
    except Exception:  # noqa: BLE001 - no registry = own hashes only
        return None


def _handoff_chain(req):
    """TK: the hand-off chain this request's registration read with, indexed
    from token 0 (``req._weg2_handoff_page_keys``), or None. The #1416d probe
    hashes the span from position 0; the registry (:func:`_handoff_keys`) is
    sliced at the matched head, so a PP0 with a head > 0 put P's keys of
    pages [head, ...) at page 0 and asked the store about keys nobody wrote."""
    from sglang.srt.weg2.handoff_keys import CHAIN_ATTR, OFF_ATTR

    if getattr(req, OFF_ATTR, False):
        return None
    chain = getattr(req, CHAIN_ATTR, None)
    return list(chain) if chain else None


#: TK switch: every told ABSOLUTE (head + span). Default follows the #1416d
#: tree-key switch -- the one that makes told > 0 -- because a span-relative
#: told > 0 against a follower whose device head is shorter than told is a
#: named mismatch (rank exit), and any head is lost to the #1419 cap.
ENV_ABSOLUTE = "SGLANG_WEG2_TOLD_ABSOLUTE"


def _absolute_armed() -> bool:
    default = "1" if _tree_key_probe_armed() else "0"
    return os.environ.get(ENV_ABSOLUTE, default).strip().lower() in (
        "1", "true", "yes", "on",
    )


def _pp0_keys_digest(req) -> str:
    from sglang.srt.weg2.handoff_keys import chain_digest

    return chain_digest(_handoff_chain(req))


def _digests(scheduler) -> Dict[str, str]:
    d = getattr(scheduler, "_weg2_told_keys_digest", None)
    if d is None:
        d = scheduler._weg2_told_keys_digest = {}
    return d


def _note_digest(scheduler, rid: str, digest) -> None:
    if digest is None:
        return
    d = _digests(scheduler)
    d[rid] = str(digest)
    while len(d) > 4096:
        d.pop(next(iter(d)))


def _adopt_keys(scheduler, req) -> None:
    """Follower, right before a registration with PP0's told: take PP0's
    hand-off key source (#1442, PP0-authoritative)."""
    d = getattr(scheduler, "_weg2_told_keys_digest", None)
    if not d:
        return
    digest = d.pop(_rid(req), None)
    if digest is None:
        return
    try:
        from sglang.srt.weg2 import handoff as _ho
        from sglang.srt.weg2.handoff_keys import ADOPT_DISAGREE, adopt_pp0_decision

        verdict = adopt_pp0_decision(req, digest, _ho.read)
    except Exception as exc:  # noqa: BLE001 - never leave the registration undone
        logger.warning("#1442 HANDOFF-KEYS adopt n/a rid=%s: %r", rid8(req), exc)
        return
    if verdict == ADOPT_DISAGREE:
        logger.warning(
            "#1442 HANDOFF-KEYS PP0-DISAGREE rank pp=%s rid=%s pp0=%s: this rank "
            "cannot reproduce the hand-off chain PP0 read with (file gone or "
            "different); reading with own hashes -- the told comparison at "
            "admission names any shortfall.",
            getattr(getattr(scheduler, "ps", None), "pp_rank", "?"), rid8(req), digest,
        )
    else:
        n = getattr(scheduler, "_tk_adopt_n", 0) + 1
        scheduler._tk_adopt_n = n
        if _log_due(n):
            logger.info("#1442 HANDOFF-KEYS ADOPT rid=%s pp0=%r verdict=%s (n=%d)",
                        rid8(req), digest, verdict, n)


def _anchored_pages_full_span(cc, ids, page_size: int, handoff_keys=None):
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
        # a RadixKey (#1416d) goes in as is -- the hash reads its bigram flag;
        # a plain id list keeps the pre-#1416d call.
        hashes = cc.get_hash_str(
            ids if not isinstance(ids, list) else list(ids), None, page_size=page_size
        )
        if not hashes:
            return 0
        if handoff_keys:
            _k = min(len(handoff_keys), len(hashes))
            hashes = list(handoff_keys[:_k]) + list(hashes[_k:])
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
        if _tree_key_probe_armed():
            key, bigram = _probe_key(scheduler, req, ids, told)
            pages = _anchored_pages_full_span(
                cc, key, page_size, handoff_keys=_handoff_chain(req)
            )
            n = getattr(scheduler, "_1416d_probe_n", 0) + 1
            try:
                scheduler._1416d_probe_n = n
            except Exception:  # noqa: BLE001 - a frozen double keeps no count
                pass
            if _log_due(n):
                logger.info(
                    "#1416d TOLD-PROBE rid=%s told=%d keys=%d bigram=%s pages=%s "
                    "(n=%d): the clamp asks the store with the fetch's key form",
                    rid8(req), int(told), len(key), bigram, pages, n,
                )
        else:
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


def _pp0_told(scheduler, tree, req, rid: str) -> int:
    """PP0's told for a terminated prefetch: the completed prefix, anchor-
    clamped (#1416), with PP0's own record clamped alike (#1416b)."""
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
    return int(clamped)


def _twin_register(scheduler, req, twin: bool) -> str:
    """TW: the store read a held twin did not register at its intake --
    the same call and the same A12.2 routing as the intake site."""
    rid = _rid(req)
    try:
        verdict = scheduler._prefetch_kvcache(req)
    except Exception as exc:  # noqa: BLE001 - never leave it unregistered AND held
        logger.warning("#TW twin registration raised for rid=%s: %r", rid[:12], exc)
        _twin.take_pp0_twin(scheduler, rid)
        return "declined:twin_register_raised"
    try:
        req._969c_verdict = verdict
    except Exception:  # noqa: BLE001
        pass
    apply = getattr(scheduler, "_apply_prefetch_deferral", None)
    if callable(apply):
        apply(req, verdict, site="twin_release")
    if getattr(scheduler, "_weg2_told_paced_on", False):
        # #1416e: PP0's own read time is the pacing window's measure.
        _pace_intake_t(scheduler)[rid] = _clock()
    logger.info(
        "#TW TWIN-REGISTER rid=%s twin=%s verdict=%s head=%d span=%s",
        rid[:12], twin, verdict, _twin.registered_head(req),
        getattr(req, "_prefetch_span_tokens", None),
    )
    return verdict


def _twin_pp0_told(scheduler, tree, req, rid: str, label: str = "TW TWIN-TOLD") -> int:
    """TW: PP0's told for a released fork twin. The #1400 record counts only
    the span beyond the registration's match (the host insert is rooted at
    ``last_host_node``); a twin's point is the head it matched on the device
    -- its sibling's rows -- so its told is head + span, ABSOLUTE, anchor-
    clamped like every told, and PP0's own record is set to it."""
    span = int(_completed_prefix(tree, rid))
    head = _twin.registered_head(req)
    told = head + span
    clamped = int(_anchor_clamp(scheduler, req, told))
    try:
        tree._prefetch_completed_tokens[rid] = clamped
    except Exception:  # noqa: BLE001 - a tree without the dict keeps its number
        pass
    if label == "TW TWIN-TOLD":
        logger.info(
            "#TW TWIN-TOLD rid=%s head=%d span=%d told=%d clamped=%d: the twin's told "
            "is absolute (the followers compare head + span)",
            rid[:12], head, span, told, clamped,
        )
    else:
        n = getattr(scheduler, "_tk_abs_told_n", 0) + 1
        scheduler._tk_abs_told_n = n
        if _log_due(n):
            logger.info(
                "#%s rid=%s head=%d span=%d told=%d clamped=%d (n=%d): absolute told, "
                "the followers compare head + span",
                label, _pt.rid_text(rid, 12), head, span, told, clamped, n,
            )
    return clamped


def _pp0_told_any(scheduler, tree, req, rid: str, twin: bool):
    """(told, absolute): the twin's and -- switch on -- every request's told
    is ABSOLUTE (TW's form); otherwise the span-relative #1400 record."""
    if twin:
        return _twin_pp0_told(scheduler, tree, req, rid), True
    if _absolute_armed():
        return _twin_pp0_told(scheduler, tree, req, rid, label="TK ABS-TOLD"), True
    return _pp0_told(scheduler, tree, req, rid), False


def _parked(scheduler) -> set:
    """TK path 4: rids that left the waiting queue only for the dormant hold
    (#1443/#1455) or the post-wake settle (#1471). They come back through the
    release, NOT through intake -- dropping them from ``held`` left them
    without a told for ever (weg2_store_told_pending on every rank)."""
    out = set()
    for attr in ("weg2_dormant_hold", "weg2_post_wake_settle"):
        for r in getattr(scheduler, attr, None) or ():
            out.add(_rid(r))
    return out


def pp0_publish(scheduler, recv_reqs: List) -> List:
    """Top of a PP0 pass, before the forward: turn terminated prefetches of
    held rids into ``Weg2StoreTold`` objects appended to the outgoing list.
    Returns the list to SEND; the caller keeps dispatching ``recv_reqs``."""
    held: Dict[str, Any] = scheduler._weg2_store_held
    paced_on = bool(getattr(scheduler, "_weg2_told_paced_on", False))
    if paced_on:
        return _pp0_publish_paced(scheduler, recv_reqs)
    if not held:
        return recv_reqs
    told_map: Dict[str, int] = scheduler._weg2_store_told
    tree = scheduler.tree_cache
    queued = {_rid(r) for r in scheduler.waiting_queue}
    # TW: held fork twins whose sibling finished (or whose Frist ran out)
    # register their store read NOW, exactly as the intake would have.
    for _treq, _is_twin in _twin.release_due(scheduler, queued):
        _twin_register(scheduler, _treq, _is_twin)
    out: List[Weg2StoreTold] = []
    parked = _parked(scheduler)
    for rid in list(held):
        req = held[rid]
        if rid not in queued:
            if rid in parked:
                # TK path 4: held by the dormant hold / settle; published
                # once the release has queued it (PP0's record is final then).
                continue
            # Left the queue without an admission (abort, deferral elsewhere);
            # a re-queue holds it again through intake.
            held.pop(rid, None)
            _twin.take_pp0_twin(scheduler, rid)
            continue
        if _twin.is_deferred(scheduler, rid):
            # TW: no store read registered yet -- nothing to publish.
            continue
        if getattr(req, "prefetch_deferred", None) is not None:
            # A12.2 deferral: PP0 will re-issue; the verdict is not final.
            continue
        if not tree.check_prefetch_progress(rid):
            continue
        twin = _twin.take_pp0_twin(scheduler, rid)
        told, absolute = _pp0_told_any(scheduler, tree, req, rid, twin)
        told_map[rid] = told
        held.pop(rid, None)
        out.append((Weg2StoreToldTwin if twin else Weg2StoreTold)(
            rid=rid, told=told, absolute=absolute, keys_digest=_pp0_keys_digest(req)))
        n = getattr(scheduler, "_weg2_store_told_published", 0) + 1
        scheduler._weg2_store_told_published = n
        if _log_due(n):
            logger.info(
                "#1400 STORE-TOLD PUBLISHED rid=%s told=%d (n=%d held_left=%d): "
                "PP0's terminated store prefetch loaded this many page-aligned "
                "tokens; the followers register exactly this span.",
                _rt(rid),
                told,
                n,
                len(held),
            )
    if not out:
        return recv_reqs
    return list(recv_reqs) + out


def _follower_absorb_impl(scheduler, recv_reqs: List) -> List:
    """After the forward, before dispatch: take the told objects off the list,
    store them, and register the held requests' prefetch with the told span."""
    if not any(isinstance(r, (Weg2StoreTold, Weg2StoreAdmit)) for r in recv_reqs):
        return recv_reqs
    told_map: Dict[str, int] = scheduler._weg2_store_told
    held: Dict[str, Any] = scheduler._weg2_store_held
    rest = []
    for item in recv_reqs:
        if isinstance(item, Weg2StoreAdmit):
            _follower_admit(scheduler, item)
            continue
        if not isinstance(item, Weg2StoreTold):
            rest.append(item)
            continue
        rid = str(item.rid)
        told = int(item.told)
        if getattr(item, "paced", False):
            # #1416e: read-ahead only. The read starts NOW, one pass behind
            # PP0's verdict; admission still skips until the Admit arrives.
            early = _early(scheduler)
            early[rid] = told
            if getattr(item, _fb.WIRE_ACK, 0):
                # PF: PP0 armed the group fallback -- report this read.
                _fb.follower_expect(scheduler, rid, told)
            if len(early) > 256:
                # an aborted request never gets its Admit: keep the table
                # to what is still queued (or just arrived, this rid)
                queued = {_rid(r) for r in scheduler.waiting_queue}
                for k in [k for k in early if k != rid and k not in queued]:
                    early.pop(k, None)
        else:
            told_map[rid] = told
        if getattr(item, "twin", False) or getattr(item, "absolute", False):
            # TW: an absolute twin told (read-ahead or single-phase alike);
            # TK: every absolute told takes the same follower mark.
            _twin.note_follower_twin(scheduler, rid)
        # TK (#1442): PP0's key source rides to whichever registration of
        # this rid comes next (here, at the paced Admit, or at intake).
        _note_digest(scheduler, rid, getattr(item, "keys_digest", None))
        req = held.pop(rid, None)
        n = getattr(scheduler, "_weg2_store_told_absorbed", 0) + 1
        scheduler._weg2_store_told_absorbed = n
        if req is not None:
            verdict = _follower_register(scheduler, req, told)
        else:
            verdict = "held:not_yet_queued"
        if _log_due(n):
            logger.info(
                "#1400 STORE-TOLD ABSORBED rank pp=%s rid=%s told=%d verdict=%s (n=%d)",
                scheduler.ps.pp_rank,
                _rt(rid),
                told,
                verdict,
                n,
            )
    return rest


# ---------------------------------------------------------------------------
# #1416e PACED TOLD (switch SGLANG_WEG2_TOLD_PACED, default OFF)
# ---------------------------------------------------------------------------
#
# THE RISK IT REMOVES. With told > 0 (#1416d) the follower path runs again:
# a follower registers its read when the told ARRIVES and then, in the same
# pass, busy-waits for it in :func:`admission` (bounded by WAIT_CAP_S). That
# wait sits in the follower's scheduler thread, so the whole stage stops --
# every other request in pipeline flight on that stage with it (user order
# 24.09.: HiCache work never slows a running prefill/decode).
#
# WHY THE FOLLOWER CANNOT SIMPLY DEFER. On the carrierless form every rank
# plans rank-locally and the told arrival IS the membership signal: PP0
# admitted the request in the pass it published, a follower that skipped it
# would build a different batch (the W27 width split). A deferral with a
# told=0 fallback must therefore be PP0's decision, and PP0 would need each
# follower's read state -- a follower->PP0 channel this form does not have
# live (#1175's return trip has no caller, the output ring only runs on
# passes with a batch, the #1268 home hop is the idle vote's). So:
#
# THE PACED FORM, PP0-AUTHORITATIVE AND WIRELESS. Two objects instead of one:
#   1. ``Weg2StoreTold(rid, told, paced=True)`` as soon as PP0's own read has
#      terminated -- a READ-AHEAD: followers register exactly the told span,
#      nobody admits.
#   2. ``Weg2StoreAdmit(rid, told)`` once PP0's pacing window has passed --
#      PP0 admits in that pass, every follower in the pass that absorbs it
#      (the single-phase told's lag, unchanged). While the window runs, PP0
#      skips the request (``weg2_store_told_pending``) and admits everything
#      behind it; no rank waits.
# The window is PP0's estimate of the followers' read: max(factor x PP0's own
# read time, told tokens x the measured per-100k read rate), capped. A
# follower whose read is still short at the Admit falls into the unchanged
# bounded wait (named, WAIT_CAP_S) -- the residual, not the normal case.
# Told 0 needs no read and is published single-phase as before.
# PF (SGLANG_WEG2_TOLD_GROUP_FALLBACK, weg2_told_fallback): with that switch
# the followers ack their terminated reads to PP0 on a gloo tag of their own,
# PP0 admits on the acks and switches the rid to told=0 for EVERY rank on a
# short read or at its Frist -- the residual wait is gone as well.

ENV_PACED = "SGLANG_WEG2_TOLD_PACED"
ENV_PACE_FACTOR = "SGLANG_WEG2_TOLD_PACE_FACTOR"
ENV_PACE_S_PER_100K = "SGLANG_WEG2_TOLD_PACE_S_PER_100K"
ENV_PACE_CAP_S = "SGLANG_WEG2_TOLD_PACE_CAP_S"
PACE_FACTOR_DEFAULT = 1.25
#: ~1 s per 100k tokens per stage (follower store read, 27B P, 0926 estimate)
PACE_S_PER_100K_DEFAULT = 1.0
#: well below the #699 admission-wedge threshold (ADMISSION_WEDGE_SECONDS =
#: 20 s of "queued, 0 running"): a window near it would have the idle P answer
#: the paced request 503 WEG2-INTAKE-STALL and flip (PP0's own read time sits
#: on the same queue clock). A follower slower than the cap takes the residual
#: bounded wait at admission instead.
PACE_CAP_S_DEFAULT = 10.0

_clock = time.monotonic


def _paced_env() -> bool:
    return os.environ.get(ENV_PACED, "0").strip().lower() in ("1", "true", "yes", "on")


def _env_float(name: str, default: float) -> float:
    try:
        value = float(os.environ.get(name, "") or default)
    except ValueError:
        return default
    return value if value >= 0 else default


def pace_window_s(own_read_s: float, told: int) -> float:
    """PP0's estimate of how long a follower's read of ``told`` tokens takes:
    the larger of (factor x PP0's own read time) and (told x per-100k rate),
    capped. The told floor covers the case PP0 read little itself (its host
    tree already held the prefix) while the followers read it all."""
    factor = _env_float(ENV_PACE_FACTOR, PACE_FACTOR_DEFAULT)
    rate = _env_float(ENV_PACE_S_PER_100K, PACE_S_PER_100K_DEFAULT)
    cap = _env_float(ENV_PACE_CAP_S, PACE_CAP_S_DEFAULT)
    est = max(factor * max(0.0, float(own_read_s)), rate * max(0, int(told)) / 100000.0)
    return min(cap, est)


@dataclass
class _Pace:
    req: Any
    told: int
    published_at: float
    published_pass: int
    window_s: float


def _pace_intake_t(scheduler) -> Dict[str, float]:
    d = getattr(scheduler, "_weg2_told_intake_t", None)
    if d is None:
        d = scheduler._weg2_told_intake_t = {}
    return d


def _pacing(scheduler) -> Dict[str, _Pace]:
    d = getattr(scheduler, "_weg2_told_pacing", None)
    if d is None:
        d = scheduler._weg2_told_pacing = {}
    return d


def _early(scheduler) -> Dict[str, int]:
    d = getattr(scheduler, "_weg2_told_early", None)
    if d is None:
        d = scheduler._weg2_told_early = {}
    return d


def _pp0_publish_paced(scheduler, recv_reqs: List) -> List:
    """PP0, paced form: publish read-aheads for terminated reads, then the
    Admits whose window has passed. Never waits."""
    held: Dict[str, Any] = scheduler._weg2_store_held
    pacing = _pacing(scheduler)
    if not held and not pacing:
        return recv_reqs
    told_map: Dict[str, int] = scheduler._weg2_store_told
    tree = scheduler.tree_cache
    intake_t = _pace_intake_t(scheduler)
    queued = {_rid(r) for r in scheduler.waiting_queue}
    now = _clock()
    pass_n = int(getattr(scheduler, "_weg2_told_pass_n", 0)) + 1
    scheduler._weg2_told_pass_n = pass_n
    fb_on = bool(getattr(scheduler, "_weg2_told_fallback_on", False))
    fb_parked = set()
    if fb_on and pacing:
        # PF: the followers' read acks that have landed (no wait), BEFORE
        # the verdicts below read them.
        _fb.pp0_harvest(scheduler)
        fb_parked = _parked(scheduler)
    # TW: held fork twins whose sibling finished (or whose Frist ran out)
    # register their store read now (the paced read clock starts here).
    for _treq, _is_twin in _twin.release_due(scheduler, queued):
        _twin_register(scheduler, _treq, _is_twin)
    out: List[Any] = []
    # (a) Admits first: an entry created in THIS pass is never admitted in it,
    # so the read-ahead always precedes its Admit by at least one pass.
    for rid in list(pacing):
        p = pacing[rid]
        if rid not in queued:
            if fb_on and rid in fb_parked:
                # PF, DEFENSIVE ONLY: a paced rid cannot be parked on the
                # current tree -- the #1443 hold is entered only at intake
                # (before waiting_queue.append) and the #1471 settle only from
                # the hold release, and (b) below never publishes a parked
                # rid (TK path 4). Should a later path park a published rid,
                # the verdict waits for its re-queue instead of dropping it.
                continue
            # left the queue during the window (abort): no Admit, nothing
            # admits; the followers' reads end in their own drain.
            pacing.pop(rid, None)
            if fb_on:
                _fb.pp0_forget(scheduler, rid)
            logger.info("#1416e PACED-DROP rid=%s told=%d: left the queue inside the window", _rt(rid), p.told)
            continue
        if fb_on:
            # PF: PP0 alone decides -- told once every follower's read
            # reproduced it, told=0 for every rank on a short read or at the
            # Frist. Never the window's guess.
            verdict = _fb.pp0_decide(scheduler, rid, now)
            if verdict is None:
                continue
            told_final, reason = verdict
            pacing.pop(rid, None)
            _fb.pp0_note_verdict(scheduler, rid, p.told, told_final, reason, now, p.published_at)
            admit = Weg2StoreAdmit(rid=rid, told=told_final)
            if told_final != p.told:
                # PP0's own read is released like a follower's: its record
                # said told, its admission now compares 0 with 0.
                _fb.release_own_read(scheduler, rid)
                setattr(admit, _fb.WIRE_FALLBACK, 1)
            told_map[rid] = told_final
            out.append(admit)
            continue
        if now - p.published_at < p.window_s:
            continue
        pacing.pop(rid, None)
        told_map[rid] = p.told
        out.append(Weg2StoreAdmit(rid=rid, told=p.told))
        n = getattr(scheduler, "_1416e_admit_n", 0) + 1
        scheduler._1416e_admit_n = n
        if _log_due(n):
            logger.info(
                "#1416e PACED-ADMIT rid=%s told=%d window=%.2fs waited=%.2fs passes=%d (n=%d): "
                "PP0 admits in this pass, the followers in the pass that absorbs it; "
                "nobody waited while the window ran",
                _rt(rid), p.told, p.window_s, now - p.published_at, pass_n - p.published_pass, n,
            )
    # (b) read-aheads for terminated reads (the single-phase checks)
    parked = _parked(scheduler)
    for rid in list(held):
        req = held[rid]
        if rid not in queued:
            if rid in parked:
                continue  # TK path 4: back through the hold release
            held.pop(rid, None)
            intake_t.pop(rid, None)
            _twin.take_pp0_twin(scheduler, rid)
            continue
        if _twin.is_deferred(scheduler, rid):
            continue  # TW: no store read registered yet
        if getattr(req, "prefetch_deferred", None) is not None:
            continue
        if not tree.check_prefetch_progress(rid):
            continue
        twin = _twin.take_pp0_twin(scheduler, rid)
        told, absolute = _pp0_told_any(scheduler, tree, req, rid, twin)
        _cls = Weg2StoreToldTwin if twin else Weg2StoreTold
        _extra = {"absolute": absolute, "keys_digest": _pp0_keys_digest(req)}
        held.pop(rid, None)
        own_read_s = now - intake_t.pop(rid, now)
        if told <= 0:
            # nothing to read on any rank: single-phase, as before
            told_map[rid] = told
            out.append(_cls(rid=rid, told=told, **_extra))
            continue
        window = pace_window_s(own_read_s, told)
        pacing[rid] = _Pace(req=req, told=told, published_at=now, published_pass=pass_n, window_s=window)
        ahead = _cls(rid=rid, told=told, paced=True, **_extra)
        if fb_on:
            # PF: ask the followers for their read state (wire marker, set
            # only here) and start the Frist.
            setattr(ahead, _fb.WIRE_ACK, 1)
            _frist = _fb.pp0_open(scheduler, rid, told, now, own_read_s, window)
            n_fb = getattr(scheduler, "_pf_open_n", 0) + 1
            scheduler._pf_open_n = n_fb
            if n_fb <= _LOG_FIRST or n_fb % _LOG_EVERY == 0:
                logger.info(
                    "PF TOLD-OPEN rid=%s told=%d window=%.2fs frist=%.2fs (n=%d): "
                    "admission follows the followers' read acks, told=0 at the Frist",
                    rid[:8], told, window, _frist, n_fb,
                )
        out.append(ahead)
        n = getattr(scheduler, "_1416e_ahead_n", 0) + 1
        scheduler._1416e_ahead_n = n
        if _log_due(n):
            logger.info(
                "#1416e PACED-TOLD rid=%s told=%d own_read=%.2fs window=%.2fs (n=%d pacing=%d): "
                "read-ahead on the wire, admission follows the window",
                _rt(rid), told, own_read_s, window, n, len(pacing),
            )
    if not out:
        return recv_reqs
    return list(recv_reqs) + out


def _follower_admit(scheduler, item: Weg2StoreAdmit) -> None:
    """Follower: PP0 admitted this rid -- the told becomes the admission
    verdict (the single-phase told's role from here on)."""
    rid = str(item.rid)
    told = int(item.told)
    fallback = bool(getattr(item, _fb.WIRE_FALLBACK, 0))
    if fallback:
        # PF: PP0 switched this rid to told=0 for every rank -- cut this
        # rank's read (abort path: rows, lock, reader references) first.
        _fb.follower_release(scheduler, rid)
    elif getattr(scheduler, "_weg2_fb_follower", None) is not None:
        _fb.follower_forget(scheduler, rid)
    early = _early(scheduler)
    ahead = early.pop(rid, None)
    scheduler._weg2_store_told[rid] = told
    held: Dict[str, Any] = scheduler._weg2_store_held
    req = held.pop(rid, None)
    if req is not None:
        # no read-ahead reached this request (it was not queued yet): register
        # now; admission then waits for it as the single-phase form does.
        _follower_register(scheduler, req, told)
    if ahead is not None and int(ahead) != told and not fallback:
        logger.warning(
            "#1416e PACED-ADMIT rid=%s told=%d differs from the read-ahead %d; admission "
            "compares against the admitted value", _rt(rid), told, int(ahead),
        )
    n = getattr(scheduler, "_1416e_absorb_n", 0) + 1
    scheduler._1416e_absorb_n = n
    if _log_due(n):
        logger.info(
            "#1416e PACED-ADMIT ABSORBED rank pp=%s rid=%s told=%d read_ahead=%s (n=%d)",
            scheduler.ps.pp_rank, _rt(rid), told, ahead, n,
        )


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
    # TK: the cap is RAW tokens and told counts KEYS (prefix_cap_tokens).
    try:
        req._weg2_prefix_cap = prefix_cap_tokens(tree, told)
    except Exception:  # noqa: BLE001
        pass
    satisfied = getattr(scheduler, "_weg2_store_told_satisfied", None) or {}
    if rid in satisfied:
        # registered nothing because it already held the span (see
        # _follower_register): admit at told, no read to wait for.
        satisfied.pop(rid, None)
        told_map.pop(rid, None)
        _twin.take_follower_twin(scheduler, rid)
        return 0
    # The single-phase form waits here for a read registered THIS pass (the
    # stage stops for it); the paced form (#1416e) registered it a window
    # earlier, so this loop normally finds it terminated on the first call.
    deadline = time.monotonic() + WAIT_CAP_S
    waited = False
    while not tree.check_prefetch_progress(rid):
        waited = True
        if time.monotonic() > deadline:
            raise Weg2StoreToldMismatch(
                f"#1400 STORE-TOLD WAIT EXCEEDED rank pp={scheduler.ps.pp_rank} "
                f"rid={_rt(rid)} told={told}: this rank's own store read did not "
                f"terminate within {WAIT_CAP_S:g}s; the prefetch policy should "
                f"have cut it long before, so the storage thread is stuck."
            )
        time.sleep(0.002)
    own = _completed_prefix(tree, rid)
    if _twin.take_follower_twin(scheduler, rid):
        # TW: an absolute twin told -- this rank's own prefix is the head its
        # registration matched plus the span its read completed.
        own = _twin.registered_head(req) + int(own)
    credit = int(tree.pop_prefetch_loaded_tokens(rid) or 0)
    told_map.pop(rid, None)
    if own != told:
        raise Weg2StoreToldMismatch(
            f"#1400 STORE-TOLD MISMATCH rank pp={scheduler.ps.pp_rank} "
            f"rid={_rt(rid)} told={told} own_prefix={own} own_loaded={credit}: "
            f"this rank's store "
            f"read does not reproduce PP0's verdict, so its prefix would "
            f"diverge from the batch PP0 built (the W27 width split of "
            f"weg2rg3). Refusing by name instead of planning a different pass."
        )
    if waited:
        n = getattr(scheduler, "_weg2_store_told_waited", 0) + 1
        scheduler._weg2_store_told_waited = n
        if _log_due(n):
            logger.info(
                "#1400 STORE-TOLD WAITED rank pp=%s rid=%s told=%d (n=%d): the own "
                "read terminated inside the admission wait.",
                scheduler.ps.pp_rank,
                _rt(rid),
                told,
                n,
            )
    # The CREDIT is this rank's loaded increment (rides into
    # `storage_hit_length` / cached_tokens_storage, informational); the
    # uniform fact -- the prefix -- was just checked against told.
    return credit


#: :func:`refetch_plan` verdict: this rank must not re-read the rid at all.
REFETCH_SKIP = "skip"


def refetch_plan(scheduler, req):
    """TK (#1456/#1471 on the told form): may the dormant hold's re-read or
    the post-wake settle re-register ``req`` on THIS rank, and how?

    ``None``  = not the told form: re-read as before (group D, every non-PP
                boot -- byte-identical).
    ``REFETCH_SKIP`` = never: a follower without PP0's told (it registers only
                with the told span), or PP0 once its told is on the wire (a
                re-read would move PP0's record under the followers' feet).
    ``int``   = the ``limit_tokens`` of the re-read: a follower re-reads the
                told span again, exactly like its first read. Without it the
                re-read of a follower still in the hold when the told arrived
                ran to the prompt end, past told -> Weg2StoreToldMismatch.
    """
    try:
        if not armed(scheduler):
            return None
    except Exception:  # noqa: BLE001 - a stand-in without the form
        return None
    rid = _rid(req)
    told_map = getattr(scheduler, "_weg2_store_told", None) or {}
    if is_pp0(scheduler):
        published = rid in told_map or rid in (getattr(scheduler, "_weg2_told_pacing", None) or {})
        return REFETCH_SKIP if published else None
    told = told_map.get(rid)
    if told is None:
        told = (getattr(scheduler, "_weg2_told_early", None) or {}).get(rid)
    if told is None or int(told) <= 0:
        return REFETCH_SKIP
    return follower_limit_tokens(getattr(scheduler, "tree_cache", None), int(told))


from sglang.srt.managers.weg2_pass_timer import timed as _pass_timed  # noqa: E402


def _follower_absorb_pass(scheduler, recv_reqs: List) -> List:
    rest = _follower_absorb_impl(scheduler, recv_reqs)
    if getattr(scheduler, "_weg2_fb_follower", None) is not None:
        # PF: report terminated reads to PP0, finish the last send; no wait.
        _fb.follower_pump(scheduler)
    return rest


follower_absorb = _pass_timed("_1475_absorb_ms")(_follower_absorb_pass)  # #1475
