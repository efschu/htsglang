"""TW (26.09.2026): group P holds a fork twin behind its sibling until the
sibling's prefix is published, then reads it -- instead of both prefilling it.

THE SPECIMEN. NVFP4 agent run dkr27bnvfp4bar1agent09252328 (5425830c95), P log
lines 119883-134601: subagent forks with the same parent context arrive as
pairs sharing ~64k-102k tokens of prompt (weg2-22-60/61 at 104,555/104,713,
weg2-22-63/23-64 at 68,527/68,685). P runs ``schedule_policy=fcfs``,
``max_running_requests=2``; upstream SGLang only looks for shared prefixes
under ``lpm``. The second twin's #1400 store read was registered at ITS
intake, while the first twin had not yet computed the shared span, so its
told was the 4095 tokens already in the arena, #1419 capped its match there
(``#1040 EXTENT ... kv=4095 device_len=0``) and P prefilled the same ~64k-102k
tokens twice (PX3 model: ~42 s in that run).

THE RULE (PP0 only; switch ``SGLANG_WEG2_P_TWIN_DEFER``, default off):
  1. intake -- a new request that shares >= ``SGLANG_WEG2_P_TWIN_MIN_TOKENS``
     (default 8192) leading token ids (and the extra key) with a request still
     in flight on P (queued, chunk-prefilling, in a microbatch, running) does
     NOT register its store read yet; it is held like every #1400 request.
  2. top of every PP0 pass (``weg2_store_told.pp0_publish``) -- a held twin is
     released once every sibling it waits for has FINISHED on PP0 (its rows
     are in the radix tree and its chunk/retain publish put its pages in the
     arena, weg2.retain_publish) and ``SGLANG_WEG2_P_TWIN_SETTLE_MS`` (500) and
     ``pp_size`` PP0 passes have gone by (the followers have processed the
     same finish, the retain writes have landed). Then it registers exactly
     as at intake and its told is published as a TWIN told: absolute (the
     local head the registration matched PLUS the store span), because a
     prefix that is already on the device is the whole point here and the
     #1400 completion record counts only the span beyond it
     (unified_radix_cache ``_insert_helper_host`` is rooted at
     ``last_host_node`` = the match's ``best_match_node``).
  3. the FRIST -- ``SGLANG_WEG2_P_TWIN_WAIT_S`` (120 s) after intake, or when
     the twin left the queue, it is released as an ordinary request (plain
     told, byte-for-byte the pre-TW path).

RANK AGREEMENT. Nothing new is decided off PP0. A follower already holds
every request until PP0's told arrives on the request wire (#1400,
``declined:weg2_held``) and skips it at admission until then
(``weg2_store_told_pending``); the deferral only moves WHEN PP0 puts the
told on the wire, so every stage skips and admits the twin on the same
decision. The twin told is a subclass of ``Weg2StoreTold`` so a follower
knows to compare its own prefix absolutely (its registered head + its span).

NEVER BRAKES ANYTHING. The deferral is a skip of ONE queue entry, never a
pass hold: the admission loop moves on to the next request (fcfs order is
kept for everyone else), and a twin could not have run before its sibling's
last chunk anyway (one chunked request at a time). The checks are host-side,
O(in-flight) slice compares at intake and O(deferred) finished() reads per
pass -- no collective, no device sync, no store I/O.
"""

from __future__ import annotations

import logging
import os
import time
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Tuple

logger = logging.getLogger(__name__)

ENV = "SGLANG_WEG2_P_TWIN_DEFER"
ENV_MIN_TOKENS = "SGLANG_WEG2_P_TWIN_MIN_TOKENS"
ENV_WAIT_S = "SGLANG_WEG2_P_TWIN_WAIT_S"
ENV_SETTLE_MS = "SGLANG_WEG2_P_TWIN_SETTLE_MS"

DEFAULT_MIN_TOKENS = 8192
DEFAULT_WAIT_S = 120.0
DEFAULT_SETTLE_MS = 500.0

#: intake verdict of a held twin (no store read registered yet).
VERDICT_DEFERRED = "declined:weg2_twin_deferred"
#: release reasons
REL_PUBLISHED = "published"
REL_DEADLINE = "deadline"

_ATTR = "_weg2_twin_state"
_LOG_FIRST = 8
_LOG_EVERY = 256
#: a request never admitted (abort) must not grow the flag sets without bound
_FLAG_CAP = 4096

_now = time.monotonic  # patched by the tests


def _env_on(env=None) -> bool:
    env = os.environ if env is None else env
    return str(env.get(ENV, "0")).strip().lower() in ("1", "true", "yes", "on")


def _env_int(name: str, default: int, env=None) -> int:
    env = os.environ if env is None else env
    try:
        return max(1, int(str(env.get(name, default)).strip()))
    except (TypeError, ValueError):
        return default


def _env_float(name: str, default: float, env=None) -> float:
    env = os.environ if env is None else env
    try:
        return max(0.0, float(str(env.get(name, default)).strip()))
    except (TypeError, ValueError):
        return default


@dataclass
class _Wait:
    req: Any
    sources: List[Any]
    since: float
    shared: int
    #: PP0 pass / monotonic time at which the LAST source was first seen finished
    done_pass: Optional[int] = None
    done_t: Optional[float] = None


@dataclass
class _State:
    min_tokens: int
    wait_s: float
    settle_s: float
    settle_passes: int
    waits: Dict[str, _Wait] = field(default_factory=dict)
    #: PP0: rids released as twins whose told is still to be published
    twin_pp0: Dict[str, int] = field(default_factory=dict)
    #: follower: rids whose absorbed told is a twin (absolute) told
    twin_follower: Dict[str, int] = field(default_factory=dict)
    passes: int = 0
    n_defer: int = 0
    n_release: int = 0
    n_deadline: int = 0


def state(scheduler) -> Optional[_State]:
    """Resolved ONCE per scheduler (a boot constant): None = switch off."""
    st = getattr(scheduler, _ATTR, False)
    if st is not False:
        return st
    st = None
    if _env_on():
        pp_size = int(getattr(getattr(scheduler, "ps", None), "pp_size", 1) or 1)
        st = _State(
            min_tokens=_env_int(ENV_MIN_TOKENS, DEFAULT_MIN_TOKENS),
            wait_s=_env_float(ENV_WAIT_S, DEFAULT_WAIT_S),
            settle_s=_env_float(ENV_SETTLE_MS, DEFAULT_SETTLE_MS) / 1000.0,
            settle_passes=max(1, pp_size),
        )
        logger.warning(
            "#TW P-TWIN-DEFER ARMED rank pp=%s min_tokens=%d wait_s=%g settle_ms=%g "
            "settle_passes=%d: a request sharing >= min_tokens leading ids with one "
            "in flight on P registers its store read only after that sibling "
            "finished (twin told absolute), bounded by wait_s.",
            getattr(getattr(scheduler, "ps", None), "pp_rank", "?"),
            st.min_tokens, st.wait_s, st.settle_s * 1000.0, st.settle_passes,
        )
    try:
        setattr(scheduler, _ATTR, st)
    except Exception:  # noqa: BLE001 - a frozen double just resolves again
        pass
    return st


def _ids(req):
    ids = getattr(req, "origin_input_ids", None)
    return ids if ids is not None else ()


def _same(a, b, n: int) -> bool:
    x, y = a[:n], b[:n]
    if type(x) is not type(y):
        x, y = list(x), list(y)
    return x == y


def shared_prefix_len(a, b) -> int:
    """Leading ids ``a`` and ``b`` share (binary search over C-level slice
    compares; only called for a pair that already passed the threshold)."""
    lo, hi = 0, min(len(a), len(b))
    while lo < hi:
        mid = (lo + hi + 1) // 2
        if _same(a, b, mid):
            lo = mid
        else:
            hi = mid - 1
    return lo


def is_twin(a, b, min_tokens: int) -> bool:
    if getattr(a, "extra_key", None) != getattr(b, "extra_key", None):
        return False
    ia, ib = _ids(a), _ids(b)
    if len(ia) < min_tokens or len(ib) < min_tokens:
        return False
    return _same(ia, ib, min_tokens)


def _batch_reqs(batch) -> Iterable[Any]:
    return getattr(batch, "reqs", None) or ()


def inflight(scheduler) -> List[Any]:
    """Every request PP0 knows as in flight on P and not finished: queued,
    the chunked one, the running batch, the PP microbatch rings."""
    seen, out = set(), []

    def add(r):
        if r is None or id(r) in seen:
            return
        seen.add(id(r))
        fin = getattr(r, "finished", None)
        if callable(fin) and fin():
            return
        out.append(r)

    for r in list(getattr(scheduler, "waiting_queue", None) or ()):
        add(r)
    add(getattr(scheduler, "chunked_req", None))
    for r in _batch_reqs(getattr(scheduler, "running_batch", None)):
        add(r)
    for r in _batch_reqs(getattr(scheduler, "cur_batch", None)):
        add(r)
    for ring in ("running_mbs", "mbs", "last_mbs"):
        for b in list(getattr(scheduler, ring, None) or ()):
            for r in _batch_reqs(b):
                add(r)
    return out


def _say(n: int) -> bool:
    return n <= _LOG_FIRST or n % _LOG_EVERY == 0


def intake_defer(scheduler, req) -> bool:
    """PP0 intake: True = hold ``req`` without registering its store read."""
    st = state(scheduler)
    if st is None:
        return False
    rid = str(getattr(req, "rid", ""))
    if len(_ids(req)) < st.min_tokens:
        return False
    sources = [
        s for s in inflight(scheduler)
        if s is not req and str(getattr(s, "rid", "")) != rid
        and is_twin(s, req, st.min_tokens)
    ]
    if not sources:
        st.waits.pop(rid, None)
        return False
    shared = max(shared_prefix_len(_ids(s), _ids(req)) for s in sources)
    st.waits[rid] = _Wait(req=req, sources=sources, since=_now(), shared=shared)
    st.n_defer += 1
    if _say(st.n_defer):
        logger.info(
            "#TW TWIN-DEFER rid=%s len=%d shared=%d sources=%s (n=%d): store read "
            "held until the sibling finished; admission skips it meanwhile, "
            "nothing else waits.",
            rid[:12], len(_ids(req)), shared,
            [str(getattr(s, "rid", "?"))[:12] for s in sources], st.n_defer,
        )
    return True


def is_deferred(scheduler, rid: str) -> bool:
    st = getattr(scheduler, _ATTR, None)
    return bool(st) and str(rid) in st.waits


def release_due(scheduler, queued) -> List[Tuple[Any, bool]]:
    """Top of a PP0 pass: the held twins whose wait is over, as
    ``(req, twin)``; ``twin`` False = the Frist fired (ordinary request).
    A twin that left the queue is forgotten (the #1400 held map drops it)."""
    st = getattr(scheduler, _ATTR, None)
    if not st:
        return []
    st.passes += 1
    if not st.waits:
        return []
    now = _now()
    out: List[Tuple[Any, bool]] = []
    for rid in list(st.waits):
        w = st.waits[rid]
        if rid not in queued:
            st.waits.pop(rid, None)
            continue
        pending = [
            s for s in w.sources
            if not (callable(getattr(s, "finished", None)) and s.finished())
        ]
        if not pending and w.done_pass is None:
            w.done_pass, w.done_t = st.passes, now
        reason = None
        if (
            not pending
            and st.passes - w.done_pass >= st.settle_passes
            and now - w.done_t >= st.settle_s
        ):
            reason = REL_PUBLISHED
        elif now - w.since >= st.wait_s:
            reason = REL_DEADLINE
        if reason is None:
            continue
        st.waits.pop(rid, None)
        twin = reason == REL_PUBLISHED
        if twin:
            st.n_release += 1
            n = st.n_release
        else:
            st.n_deadline += 1
            n = st.n_deadline
        if twin and len(st.twin_pp0) < _FLAG_CAP:
            st.twin_pp0[rid] = w.shared
        if _say(n):
            logger.info(
                "#TW TWIN-RELEASE rid=%s reason=%s waited_s=%.2f shared=%d "
                "pending_sources=%d (n=%d)%s",
                rid[:12], reason, now - w.since, w.shared, len(pending), n,
                "" if twin else ": FRIST -- registered as an ordinary request",
            )
        out.append((w.req, twin))
    return out


def take_pp0_twin(scheduler, rid: str) -> bool:
    """PP0 publish: True once for a rid released as a twin."""
    st = getattr(scheduler, _ATTR, None)
    return bool(st) and st.twin_pp0.pop(str(rid), None) is not None


def note_follower_twin(scheduler, rid: str) -> None:
    st = getattr(scheduler, _ATTR, None)
    if st is None:
        st = state(scheduler)
    if st is None:
        # The wire said twin: the told is absolute whatever this rank's env
        # says (one boot = one env, but PP0 is the authority on the told).
        st = _State(DEFAULT_MIN_TOKENS, DEFAULT_WAIT_S, 0.0, 1)
        try:
            setattr(scheduler, _ATTR, st)
        except Exception:  # noqa: BLE001
            return
    while len(st.twin_follower) >= _FLAG_CAP:
        st.twin_follower.pop(next(iter(st.twin_follower)))
    st.twin_follower[str(rid)] = 1


def take_follower_twin(scheduler, rid: str) -> bool:
    st = getattr(scheduler, _ATTR, None)
    return bool(st) and st.twin_follower.pop(str(rid), None) is not None


def registered_head(req) -> int:
    """The local prefix (device + host) the registration matched before its
    span -- stamped by ``_prefetch_kvcache`` (#1176) on every rank."""
    try:
        return max(0, int(getattr(req, "_prefetch_registered_prefix_len", 0) or 0))
    except (TypeError, ValueError):
        return 0
