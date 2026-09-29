"""TW (26.09.2026): group P holds a fork twin behind its sibling until the
sibling's prefix is published, then reads it -- instead of both prefilling it.

THE SPECIMEN. NVFP4 agent run dkr27bnvfp4bar1agent09252328 (5425830c95), P log
lines 119883-134601: subagent forks with the same parent context arrive as
pairs sharing ~64k-102k tokens of prompt (pdflip-22-60/61 at 104,555/104,713,
pdflip-22-63/23-64 at 68,527/68,685). P runs ``schedule_policy=fcfs``,
``max_running_requests=2``; upstream SGLang only looks for shared prefixes
under ``lpm``. The second twin's #1400 store read was registered at ITS
intake, while the first twin had not yet computed the shared span, so its
told was the 4095 tokens already in the arena, #1419 capped its match there
(``#1040 EXTENT ... kv=4095 device_len=0``) and P prefilled the same ~64k-102k
tokens twice (PX3 model: ~42 s in that run).

THE RULE (PP0 only; switch ``FLLIPER_PDFLIP_P_TWIN_DEFER``, default off):
  1. intake -- a new request that shares >= ``FLLIPER_PDFLIP_P_TWIN_MIN_TOKENS``
     (default 8192) leading token ids (and the extra key) with a request still
     in flight on P (queued, chunk-prefilling, in a microbatch, running) does
     NOT register its store read yet; it is held like every #1400 request.
  2. top of every PP0 pass (``pdflip_store_told.pp0_publish``) -- a held twin is
     released once every sibling it waits for has FINISHED on PP0 (its rows
     are in the radix tree and its chunk/retain publish put its pages in the
     arena, pdflip.retain_publish) and ``FLLIPER_PDFLIP_P_TWIN_SETTLE_MS`` (500) and
     ``pp_size`` PP0 passes have gone by (the followers have processed the
     same finish, the retain writes have landed). Then it registers exactly
     as at intake and its told is published as a TWIN told: absolute (the
     local head the registration matched PLUS the store span), because a
     prefix that is already on the device is the whole point here and the
     #1400 completion record counts only the span beyond it
     (unified_radix_cache ``_insert_helper_host`` is rooted at
     ``last_host_node`` = the match's ``best_match_node``).
  3. the FRIST -- ``FLLIPER_PDFLIP_P_TWIN_WAIT_S`` (120 s) after intake, or when
     the twin left the queue, it is released as an ordinary request (plain
     told, byte-for-byte the pre-TW path).
  4. NO GAIN, NO WAIT (28.09., NF rc12z30e ca2a9706ec, P log
     ...09282117_ca2a9706ec_0928_211748): a hybrid model resumes only at a
     Mamba anchor, and a sibling writes anchors only at its chunk ends and at
     its END ANCHOR ``floor((len - 1) / page) * page`` ('PDFLIP END-ANCHOR ...
     anchor=20032 target=20032'). An anchor the sibling writes past ``shared``
     is off the twin's path, one at or below the sibling's own start depth
     ``s0`` is there for the twin already. So the sibling brings the twin
     something only if ``s0 < end <= shared`` or a whole chunk fits into
     ``(s0, shared]`` (``shared - s0 >= chunked_prefill_size``). Specimen
     pdflip-8-30 (20171 tokens, shared 20029 with pdflip-8-29 of 20033 tokens,
     s0 16384): '#TW TWIN-RELEASE ... waited_s=6.34', then 'TWIN-TOLD
     head=16384' and 3787 tokens computed from 16384 -- exactly what it would
     have computed without the wait. 6 of that boot's 8 deferrals had this
     shape (0-7, 2-10, 2-11, 8-30, 12-35, 30-63: 51 s of held twins, 0 tokens
     saved); the two that gained (23-50, 37-68: the finished sibling's end
     anchor 49536 / 81600 at or below shared) keep holding. ``s0`` is the
     sibling's prefix when PP0 first sees it admitted (in flight, not in the
     waiting queue); a sibling never seen admitted is undecidable and holds
     as before, unless neither its end nor a whole chunk can land inside
     ``shared``. A twin whose sources all bring nothing registers at intake
     ('#TW TWIN-NO-GAIN ... at=intake'); a pass that learns ``s0`` releases
     a held one as an ordinary request ('... at=release').

RANK AGREEMENT. Nothing new is decided off PP0. A follower already holds
every request until PP0's told arrives on the request wire (#1400,
``declined:pdflip_held``) and skips it at admission until then
(``pdflip_store_told_pending``); the deferral only moves WHEN PP0 puts the
told on the wire, so every stage skips and admits the twin on the same
decision. The twin told is a subclass of ``PdFlipStoreTold`` so a follower
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

ENV = "FLLIPER_PDFLIP_P_TWIN_DEFER"
ENV_MIN_TOKENS = "FLLIPER_PDFLIP_P_TWIN_MIN_TOKENS"
ENV_WAIT_S = "FLLIPER_PDFLIP_P_TWIN_WAIT_S"
ENV_SETTLE_MS = "FLLIPER_PDFLIP_P_TWIN_SETTLE_MS"

DEFAULT_MIN_TOKENS = 8192
DEFAULT_WAIT_S = 120.0
DEFAULT_SETTLE_MS = 500.0

#: intake verdict of a held twin (no store read registered yet).
VERDICT_DEFERRED = "declined:pdflip_twin_deferred"
#: release reasons
REL_PUBLISHED = "published"
REL_DEADLINE = "deadline"
REL_NO_GAIN = "no_gain"

_ATTR = "_pdflip_twin_state"
_LOG_FIRST = 8
_LOG_EVERY = 256
#: a request never admitted (abort) must not grow the flag sets without bound
_FLAG_CAP = 4096

_now = time.monotonic  # patched by the tests


def _env_on(env=None) -> bool:
    # RG 26.09.: unset = the registry row of the published form's profile
    # (pdflip/form.py PREFIX_SWITCHES: qwen27b on, nextflash/no form off).
    from flliper.srt.pdflip.form import prefix_switch_armed

    return prefix_switch_armed(ENV, env)


def _env_int(name: str, default: int, env=None) -> int:
    env = os.environ if env is None else env
    try:
        return max(1, int(str(env.get(name, default)).strip()))
    except (TypeError, ValueError):
        return default


def _pos_int(v, default: int) -> int:
    try:
        v = int(v)
    except (TypeError, ValueError):
        return default
    return v if v > 0 else default


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
    #: rule 4: rid -> leading ids shared with THAT source (``shared`` is the max)
    shared_by: Dict[str, int] = field(default_factory=dict)
    #: PP0 pass / monotonic time at which the LAST source was first seen finished
    done_pass: Optional[int] = None
    done_t: Optional[float] = None


@dataclass
class _State:
    min_tokens: int
    wait_s: float
    settle_s: float
    settle_passes: int
    #: rule 4: the page the end anchor is floored to, and the chunk budget
    #: (0 = unchunked: no anchor between a sibling's start and its end)
    page: int = 1
    chunk: int = 0
    waits: Dict[str, _Wait] = field(default_factory=dict)
    #: rule 4: rid -> prefix length when PP0 first saw it admitted (s0)
    s0: Dict[str, int] = field(default_factory=dict)
    #: PP0: rids released as twins whose told is still to be published
    twin_pp0: Dict[str, int] = field(default_factory=dict)
    #: follower: rids whose absorbed told is a twin (absolute) told
    twin_follower: Dict[str, int] = field(default_factory=dict)
    passes: int = 0
    #: #56: the in-flight requests PP0 saw at its last look (id -> req), and
    #: the ones that FINISHED since, rid -> (req, pass, t), kept for the settle
    #: window only: a twin arriving right after its sibling's finish is held
    #: until the sibling's retain publish (its end anchor) has landed.
    last_seen: Dict[int, Any] = field(default_factory=dict)
    recent: Dict[str, Tuple[Any, int, float]] = field(default_factory=dict)
    n_defer: int = 0
    n_recent: int = 0
    n_release: int = 0
    n_deadline: int = 0
    n_no_gain: int = 0


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
            page=_pos_int(getattr(scheduler, "page_size", 1), 1),
            chunk=_pos_int(getattr(scheduler, "chunked_prefill_size", 0), 0),
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


def _settled(st: "_State", done_pass: int, done_t: float, now: float) -> bool:
    return st.passes - done_pass >= st.settle_passes and now - done_t >= st.settle_s


def _refresh_recent(st: "_State", scheduler, now: float) -> List[Any]:
    """#56 (NF rc12z 09280209, 03:15:02 PP0: ``PDFLIP END-ANCHOR rid=pdflip-20-25
    anchor=37952``, the same second ``#1416 STORE-TOLD ANCHOR-CLAMP
    rid=pdflip-21-27 completed=37952 anchored=30528``): a sibling that has just
    FINISHED on PP0 is no longer in flight, but its end anchor is not yet in
    the store (the retain publish is the step TW's own settle waits for). A
    twin arriving in that window registered at once, its store read found the
    KV pages and no anchor, and told fell back to the previous anchor. Record
    the requests that left the in-flight set FINISHED since the last look,
    keep them for the settle window, and return the current in-flight set."""
    cur = inflight(scheduler)
    ids_now = {id(r) for r in cur}
    _note_s0(st, scheduler, cur)
    for key, r in st.last_seen.items():
        if key in ids_now:
            continue
        fin = getattr(r, "finished", None)
        if callable(fin) and fin() and len(st.recent) < _FLAG_CAP:
            st.recent[str(getattr(r, "rid", ""))] = (r, st.passes, now)
    st.last_seen = {id(r): r for r in cur}
    for rid in [k for k, (_r, p, t) in st.recent.items() if _settled(st, p, t, now)]:
        st.recent.pop(rid, None)
    return cur


def _rid(r) -> str:
    return str(getattr(r, "rid", ""))


def _note_s0(st: "_State", scheduler, cur: List[Any]) -> None:
    """Rule 4: the prefix a request holds when PP0 first sees it admitted
    (in flight, out of the waiting queue) -- its start depth ``s0``; every
    anchor it writes later lies above it. Kept while the rid is in flight,
    just finished, or a source of a held twin."""
    queued = {id(r) for r in (getattr(scheduler, "waiting_queue", None) or ())}
    for r in cur:
        if id(r) in queued:
            continue
        rid = _rid(r)
        if rid and rid not in st.s0 and len(st.s0) < _FLAG_CAP:
            # prefix_indices is a torch tensor on the metal: never its truth
            # value (rc12z30i 27B P 00:04:33Z died on `tensor or ()`)
            _pi = getattr(r, "prefix_indices", None)
            try:
                st.s0[rid] = 0 if _pi is None else len(_pi)
            except TypeError:
                pass
    if len(st.s0) > len(cur) + len(st.recent):
        keep = {_rid(r) for r in cur} | set(st.recent)
        for w in st.waits.values():
            keep.update(_rid(s) for s in w.sources)
        for rid in [k for k in st.s0 if k not in keep]:
            st.s0.pop(rid, None)


def _gain(st: "_State", src, shared: int) -> Optional[bool]:
    """Rule 4: can ``src`` (in flight or finished) put an anchor inside the
    twin's usable band (s0, shared]? True / False / None = not decidable yet
    (``s0`` unknown: never seen admitted)."""
    n = len(_ids(src))
    end = ((n - 1) // st.page) * st.page if n > 0 else 0
    whole_chunk_fits = st.chunk > 0 and shared >= st.chunk
    s0 = st.s0.get(_rid(src))
    if s0 is None:
        if end > shared and not whole_chunk_fits:
            return False  # neither its end nor a whole chunk can land inside
        return None
    if s0 < end <= shared:
        return True
    return bool(st.chunk > 0 and shared - s0 >= st.chunk)


def _say_no_gain(st: "_State", rid: str, w_shared: Dict[str, int], at: str) -> None:
    st.n_no_gain += 1
    if _say(st.n_no_gain):
        logger.info(
            "#TW TWIN-NO-GAIN rid=%s at=%s sources=%s page=%d chunk=%d (n=%d): no "
            "source writes an anchor in (its start, shared] -- a hybrid model "
            "resumes only at an anchor, so waiting reads nothing; registered as an "
            "ordinary request.",
            rid[:12], at,
            [(k[:12], sh, st.s0.get(k)) for k, sh in w_shared.items()],
            st.page, st.chunk, st.n_no_gain,
        )


def intake_defer(scheduler, req) -> bool:
    """PP0 intake: True = hold ``req`` without registering its store read."""
    st = state(scheduler)
    if st is None:
        return False
    rid = str(getattr(req, "rid", ""))
    if len(_ids(req)) < st.min_tokens:
        return False
    now = _now()
    live = _refresh_recent(st, scheduler, now)
    sources = [
        s for s in live
        if s is not req and str(getattr(s, "rid", "")) != rid
        and is_twin(s, req, st.min_tokens)
    ]
    # #56: siblings that finished within the settle window count too; their
    # finish is the start of the settle (not this intake).
    recent = [
        (r, p, t) for k, (r, p, t) in st.recent.items()
        if r is not req and k != rid and is_twin(r, req, st.min_tokens)
    ]
    if not sources and not recent:
        st.waits.pop(rid, None)
        return False
    all_src = sources + [r for r, _p, _t in recent]
    shared_by = {_rid(s): shared_prefix_len(_ids(s), _ids(req)) for s in all_src}
    # Rule 4: a source that cannot put an anchor into the twin's band is no
    # reason to wait.
    gain_src = [s for s in all_src if _gain(st, s, shared_by[_rid(s)]) is not False]
    if not gain_src:
        st.waits.pop(rid, None)
        _say_no_gain(st, rid, shared_by, "intake")
        return False
    keep = {id(s) for s in gain_src}
    sources = [s for s in sources if id(s) in keep]
    recent = [(r, p, t) for r, p, t in recent if id(r) in keep]
    all_src = gain_src
    shared = max(shared_by[_rid(s)] for s in all_src)
    w = _Wait(req=req, sources=all_src, since=now, shared=shared, shared_by=shared_by)
    if not sources:
        w.done_pass = max(p for _r, p, _t in recent)
        w.done_t = max(t for _r, _p, t in recent)
        st.n_recent += 1
    st.waits[rid] = w
    st.n_defer += 1
    if _say(st.n_defer):
        logger.info(
            "#TW TWIN-DEFER rid=%s len=%d shared=%d sources=%s just_finished=%s (n=%d): "
            "store read held until the sibling finished and its publish settled; "
            "admission skips it meanwhile, nothing else waits.",
            rid[:12], len(_ids(req)), shared,
            [str(getattr(s, "rid", "?"))[:12] for s in sources],
            [str(getattr(r, "rid", "?"))[:12] for r, _p, _t in recent], st.n_defer,
        )
    return True


def is_deferred(scheduler, rid: str) -> bool:
    st = getattr(scheduler, _ATTR, None)
    return bool(st) and str(rid) in st.waits


def tick(scheduler) -> None:
    """Top of EVERY PP0 pass (``pdflip_store_told.pp0_publish``, before any early
    return): count the pass and note who finished since the last one (#56).
    Nothing when the switch is off; O(in-flight) host bookkeeping otherwise."""
    st = state(scheduler)
    if not st:
        return
    st.passes += 1
    _refresh_recent(st, scheduler, _now())


def release_due(scheduler, queued) -> List[Tuple[Any, bool]]:
    """Top of a PP0 pass: the held twins whose wait is over, as
    ``(req, twin)``; ``twin`` False = the Frist fired (ordinary request).
    A twin that left the queue is forgotten (the #1400 held map drops it)."""
    st = getattr(scheduler, _ATTR, None)
    if not st:
        return []
    now = _now()
    if not st.waits:
        return []
    out: List[Tuple[Any, bool]] = []
    for rid in list(st.waits):
        w = st.waits[rid]
        if rid not in queued:
            st.waits.pop(rid, None)
            continue
        # Rule 4: drop the sources that turned out to bring nothing (their
        # start depth is known now); none left -> no reason to wait.
        w.sources = [
            s for s in w.sources
            if _gain(st, s, w.shared_by.get(_rid(s), w.shared)) is not False
        ]
        if not w.sources:
            st.waits.pop(rid, None)
            _say_no_gain(st, rid, w.shared_by, "release")
            out.append((w.req, False))
            continue
        pending = [
            s for s in w.sources
            if not (callable(getattr(s, "finished", None)) and s.finished())
        ]
        if not pending and w.done_pass is None:
            w.done_pass, w.done_t = st.passes, now
        reason = None
        if not pending and _settled(st, w.done_pass, w.done_t, now):
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
