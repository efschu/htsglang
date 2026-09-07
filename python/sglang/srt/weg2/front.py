"""The Weg-2 front on :30030 (spec section 3, record sections 1c B2 / 1f B7).

The only port a client sees.  It owns:

* the REQUEST LEDGER: every request the front routed, per group, until its
  final response -- witness A of the drain double witness (spec 3.4);
* the SEQUENTIAL TWO-LEG ROUTE: leg 1 = the prompt to group P with
  ``max_new_tokens=1`` so the prefill lands in the canonical page store;
  leg 2 = the same prompt to group D, which reads the prefix from the store
  and decodes.  ``local_proxy.py``'s fan-out (both arms at once) is deleted:
  in a phase router that is a double prefill by construction; its
  ``max_new_tokens = 1`` idiom is lifted verbatim;
* DRAIN-AND-FLIP on the #1011 work-exhaustion clocks: P leaves when the
  prefill backlog is empty, D leaves when it has no decodable work left --
  plus the V1 FAIRNESS BOUND ``W`` (record 1f B7, "operator V1 fairness
  bound", 45 s, ``--fairness-w-s``): when the oldest request waiting for P
  is older than W the front stops admitting NEW short work to D, drains D's
  running decodes to completion, then flips;
* the route SHORT grant (record 1c B2, spec 3.5.2): a request whose
  estimated uncached remainder is at most ONE chunk (4096 tokens) is served
  in D directly while D is awake;
* SLEEP/WAKE CONTROL through S1's endpoints, with the quiesce (F4) before
  every sleep: witness B is the rank's own ``is_fully_idle`` as answered by
  ``/flush_cache`` (200 only when every HiCache in-flight term is zero),
  polled to a deadline; disagreement in either direction is W3;
* the named refusals W1, W2, W3, W4, W9, W16, W17, W19, W22 (see
  ``Weg2Stop``), ``/abort_request``, sessions refused 501, a corridor
  sampler line per phase, and the group-identifying marker ``WEG2-SERVED
  group=P|D`` that R2 counts.

V1 simplifications, each declared (also in the launcher's deviation list):
* pricing is a character estimate corrected by realised outcomes: the span
  LRU holds the realised (text, prompt_tokens) of leg-1/leg-2 outcomes and a
  new request's span is the longest common text prefix with an entry,
  converted proportionally; a request with no matching entry is priced at
  the full prompt (W22 ``Weg2SpanUnknownPricedFull``, counted);
* W16 is evaluated from the realised ``usage`` of a non-streamed leg-2
  response (``prompt_tokens - cached_tokens``); a streamed leg 2 is served
  but not priced (counted as unpriced);
* W17's serving-fact confirmation is the group's process liveness (its
  session id from the launcher), not an NVML per-process read;
* the controller epoch lives in this ledger only (no group echo).
"""

from __future__ import annotations

import argparse
import asyncio
import collections
import hashlib
import json
import logging
import os
import subprocess
import time
from dataclasses import dataclass, field
from typing import Any, Deque, Dict, List, Optional, Tuple

from aiohttp import ClientSession, ClientTimeout, web

from sglang.srt.managers.weg2_memory_saver import weights_family_tags

logger = logging.getLogger("weg2.front")

FORWARD_PATHS = ("/generate", "/v1/completions", "/v1/chat/completions")
PASSTHROUGH_GET = ("/v1/models", "/get_model_info", "/get_server_info", "/model_info", "/metrics")
CHARS_PER_TOKEN = 3.0  # conservative: over-estimates tokens, never under-prices
# #1233 zero-remainder: the CARRIER-EXCEEDS route must not UNDER-estimate --
# measured boot weg2zr1: 80,000 chars of markdown = 30,100 tokens (2.66
# chars/token), priced 26,701 by CHARS_PER_TOKEN and routed BATCH past the
# 27,466-token carrier bound. Route by a lower divisor; the realised leg-1
# count corrects any prompt that still slips through (see leg1).
CARRIER_CHARS_PER_TOKEN = 2.4
#: WEG2_SCHEDULING_SPEC_0907 C13/K10: the drain deadline is a FLAG
#: (``--drain-deadline-s``); this is the shipped value it preserves (spec
#: 3.5.4, the #111 link-seam bound reused).  No code reads it except the
#: argparse default -- the runtime reads ``Front.drain_deadline_s``.
DRAIN_DEADLINE_DEFAULT_S = 120.0
#: Spec C10/K5: the recorded PRE-BARLINK break-even inputs (record 1l/1o
#: weg2zr2 pair) that produce X's fallback.  Only the FRONT's default; the
#: launcher recomputes from this boot's own lines and tells the front (C2).
X_FALLBACK_TOKENS = 22000
QUIESCE_DEADLINE_S = 90.0
#: Spec C4/R-15/O9: the admitter resolves ONE future and then waits for that
#: request's coroutine to actually POST to D before resolving the next.  A
#: disconnected client never posts, so the wait carries a deadline and a
#: named refusal (W36) instead of stalling the whole queue.  O9 leaves it a
#: constant rather than a flag until a boot shows T5 firing in practice.
POST_BARRIER_S = 30.0
RPC_TIMEOUT_S = 900.0
SPAN_LRU = 512
SLEEP_TAGS = ["kv_cache", "weights"]
KV_TAG = "kv_cache"


class Weg2Stop(Exception):
    """A named refusal that STOPS the front (spec section 5)."""

    def __init__(self, name: str, detail: str):
        super().__init__(f"{name}: {detail}")
        self.name = name
        self.detail = detail


# --------------------------------------------------------------------------
# pure policy pieces (tested without a server)
# --------------------------------------------------------------------------


def request_text(payload: dict) -> str:
    """The prompt as ONE string, for the span estimate and the ledger."""
    if "messages" in payload and isinstance(payload["messages"], list):
        parts = []
        for m in payload["messages"]:
            c = m.get("content", "")
            if isinstance(c, list):
                c = " ".join(str(x.get("text", "")) if isinstance(x, dict) else str(x) for x in c)
            parts.append(f"{m.get('role', '')}:{c}\n")
        return "".join(parts)
    p = payload.get("prompt", payload.get("text", ""))
    if isinstance(p, list):
        return "\n".join(str(x) for x in p)
    return str(p)


def common_prefix_len(a: str, b: str) -> int:
    n = min(len(a), len(b))
    lo, hi = 0, n
    while lo < hi:
        mid = (lo + hi + 1) // 2
        if a[:mid] == b[:mid]:
            lo = mid
        else:
            hi = mid - 1
    return lo


class SpanLRU:
    """Realised (text, prompt_tokens) outcomes; the front's only price source."""

    def __init__(self, cap: int = SPAN_LRU):
        self.cap = cap
        self.entries: collections.OrderedDict[str, Tuple[str, int]] = collections.OrderedDict()

    def record(self, text: str, prompt_tokens: int) -> None:
        if not text or prompt_tokens <= 0:
            return
        key = hashlib.sha1(text.encode()).hexdigest()
        self.entries.pop(key, None)
        self.entries[key] = (text, prompt_tokens)
        while len(self.entries) > self.cap:
            self.entries.popitem(last=False)

    def span_tokens(self, text: str) -> Tuple[int, bool]:
        """(estimated tokens already in the store for this text's prefix, known)."""
        best = 0
        known = False
        for etext, etok in self.entries.values():
            cp = common_prefix_len(etext, text)
            if cp <= 0:
                continue
            known = True
            est = int(etok * (cp / max(1, len(etext))))
            best = max(best, est)
        return best, known


def price_remainder(text: str, spans: SpanLRU) -> Tuple[int, int, bool]:
    """(estimated uncached tokens, estimated prompt tokens, span_known)."""
    est_prompt = int(len(text) / CHARS_PER_TOKEN) + 1
    span, known = spans.span_tokens(text)
    return max(0, est_prompt - span), est_prompt, known


def usage_of(body: Any) -> Tuple[int, int, int, bool]:
    """(prompt_tokens, cached_tokens, completion_tokens, priced) from a response body.

    #1233 zero-remainder (1j finding 3): a body without usage/meta_info is
    NOT (0, 0) -- it is unpriced, and `priced=False` says so; the caller
    refuses it by name instead of serving it on a fail-open 'serve'.
    """
    if not isinstance(body, dict):
        return 0, 0, 0, False
    u = body.get("usage") or {}
    if not u and "meta_info" in body:
        mi = body["meta_info"] or {}
        if not isinstance(mi, dict) or "prompt_tokens" not in mi:
            return 0, 0, 0, False
        return (int(mi.get("prompt_tokens", 0) or 0), int(mi.get("cached_tokens", 0) or 0),
                int(mi.get("completion_tokens", 0) or 0), True)
    if not isinstance(u, dict) or "prompt_tokens" not in u:
        return 0, 0, 0, False
    pt = int(u.get("prompt_tokens", 0) or 0)
    ct = 0
    det = u.get("prompt_tokens_details") or {}
    if isinstance(det, dict):
        ct = int(det.get("cached_tokens", 0) or 0)
    ct = int(u.get("cached_tokens", ct) or ct)
    return pt, ct, int(u.get("completion_tokens", 0) or 0), True


def usage_of_stream_tail(tail: bytes) -> Tuple[int, int, int, bool]:
    """Price a STREAMED leg 2 from its final SSE chunks (1j finding 2).

    /v1/* streams carry one trailing `data: {... "usage": {...}}` chunk when
    `stream_options.include_usage` was requested (the front requests it on
    every BATCH stream); /generate streams carry `meta_info` in every chunk.
    Scans the retained tail from the end for the last priced chunk.
    """
    for raw in reversed(tail.split(b"\n")):
        line = raw.strip()
        if not line.startswith(b"data:"):
            continue
        data = line[5:].strip()
        if not data or data == b"[DONE]":
            continue
        try:
            js = json.loads(data)
        except Exception:  # noqa: BLE001
            continue
        if isinstance(js, dict) and (js.get("usage") or js.get("meta_info")):
            pt, ct, comp, priced = usage_of(js)
            if priced and pt:
                return pt, ct, comp, True
    return 0, 0, 0, False


def double_prefill_verdict(
    prompt_tokens: int, cached_tokens: int, reroutes: int, x_tokens: int
) -> str:
    """spec 3.6: 'serve' | 'reroute' | 'W16', priced against X.

    ``x_tokens`` is ``--tp-prefill-max-tokens`` (law 4): the number of
    uncached tokens D may prefill itself before the round trip through P is
    the cheaper answer.  It replaced the literal one-chunk bound
    (the deleted one-chunk module constant) at all four front sites, so the
    front cannot price a request against one number and route it against
    another (WEG2_SCHEDULING_SPEC_0907 C9).
    """
    uncached = max(0, prompt_tokens - cached_tokens)
    if uncached <= x_tokens:
        return "serve"
    if reroutes >= 1:
        return "W16"
    return "reroute"


def is_x_refusal(status: int, body_text: str) -> bool:
    """True iff D answered this leg 2 with the named W31 refusal (C11/C12).

    The front does not re-price the body: D's gate is the authority (the
    front's own number is an ESTIMATE, L10), so the only question here is
    whether the group refused BY NAME.  A 503 that is not W31 stays a 503.
    """
    return status == 503 and "W31 Weg2TpPrefillExceeded" in (body_text or "")


def witness_verdict(front_outstanding: int, rank_idle: bool) -> Optional[str]:
    """W3 in either direction; None when the two witnesses agree."""
    if front_outstanding == 0 and rank_idle:
        return None
    if front_outstanding > 0 and not rank_idle:
        return None
    if front_outstanding == 0 and not rank_idle:
        return "front drained, rank NOT idle"
    return "rank idle, front still holds requests"


def health_is_serving_fact(http_200: bool, process_alive: bool) -> bool:
    """W17: an HTTP 200 is a transport fact; liveness needs the process too."""
    return bool(http_200 and process_alive)


def fairness_reached(oldest_arrival: Optional[float], now: float, w_s: float) -> bool:
    return oldest_arrival is not None and (now - oldest_arrival) >= w_s


# --------------------------------------------------------------------------
# runtime
# --------------------------------------------------------------------------


@dataclass
class Group:
    name: str
    url: str
    sid: int = 0
    outstanding: Dict[str, float] = field(default_factory=dict)
    health_fail_streak: int = 0
    served: int = 0

    @property
    def phase(self) -> str:
        return "prefill" if self.name == "P" else "decode"


@dataclass
class Pending:
    rid: str
    path: str
    payload: dict
    text: str
    t_arrive: float
    fut: asyncio.Future
    reroutes: int = 0
    est_prompt: int = 0
    span_known: bool = False
    leg1_prompt_tokens: int = 0
    skip_leg1: bool = False  # #1233 route CARRIER-EXCEEDS: one prefill on D, no leg 1
    leg1_done: bool = False
    #: C4: the D seat this request holds while its leg 2 is in flight, and the
    #: event the admitter waits on before it resolves the next future (R-15).
    seat: Optional[Seat] = None
    posted_evt: Optional[asyncio.Event] = None
    #: C12: how often D refused this rid with W31.  A second one is W35.
    x_requeues: int = 0


class Seat:
    """One of ``--d-bs`` concurrency seats on group D (C4/C5).

    ONE acquire site per path (the admitter for BATCH, ``handle_generate``
    for SHORT) and ONE release site (:meth:`release`, called from ``leg2``'s
    ``finally`` and from the two paths that hand the request back before a
    leg 2 exists).  ``held`` makes the release idempotent, so a path that
    releases early and then falls through the ``finally`` cannot return a
    seat twice and inflate D's concurrency past its own bs.
    """

    __slots__ = ("front", "rid", "source", "held")

    def __init__(self, front: Front, rid: str, source: str):
        self.front = front
        self.rid = rid
        self.source = source
        self.held = True

    def release(self, freed_by: str) -> None:
        if not self.held:
            return
        self.held = False
        self.front._d_seat.release()
        self.front.counters["d_seat_released"] += 1
        # L3: the "wenn ein slot frei wird, wird nachgezogen" instrument.
        logger.info(
            "WEG2 D-REFILL rid=%s freed_by=%s seats_free=%d queued_d=%d",
            self.rid, freed_by, self.front.seats_free(), len(self.front._ready_for_d),
        )


def _sid_alive(sid: int) -> bool:
    if not sid:
        return True
    try:
        out = subprocess.run(["ps", "-eo", "sid"], capture_output=True, text=True, timeout=10).stdout
        return any(line.strip() == str(sid) for line in out.splitlines()[1:])
    except Exception:  # noqa: BLE001
        return True


def _nvml_free() -> List[Tuple[int, str, int]]:
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=index,uuid,memory.used,memory.total", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=10,
        ).stdout
    except Exception:  # noqa: BLE001
        return []
    res = []
    for line in out.strip().splitlines():
        idx, uuid, used, total = [x.strip() for x in line.split(",")]
        res.append((int(idx), uuid, int(total) - int(used)))
    return res


def _nvml_process_mib(pids: set) -> Dict[str, int]:
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-compute-apps=pid,used_memory,gpu_uuid", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=10,
        ).stdout
    except Exception:  # noqa: BLE001
        return {}
    res: Dict[str, int] = {}
    for line in out.strip().splitlines():
        if not line.strip():
            continue
        pid, used, uuid = [x.strip() for x in line.split(",")]
        if int(pid) in pids:
            res[uuid] = res.get(uuid, 0) + int(used)
    return res


def _session_pids(sid: int) -> set:
    out = subprocess.run(["ps", "-eo", "pid,sid"], capture_output=True, text=True).stdout
    return {int(a) for a, b in (l.split() for l in out.splitlines()[1:] if len(l.split()) == 2) if b == str(sid)}


class Front:
    def __init__(self, prefill: str, decode: str, awake: str, tag: str, store_dir: str,
                 prefill_sid: int, decode_sid: int, dc_reserve: Dict[str, int], w_s: float,
                 weight_chunks: int = 0, carrier_max_tokens: int = 0,
                 p_concurrency: int = 8, d_bs: int = 8,
                 tp_prefill_max_tokens: int = X_FALLBACK_TOKENS,
                 flip_min_work_tokens: Optional[int] = None,
                 min_dwell_ms: Optional[float] = None,
                 idle_layout: str = "D",
                 drain_deadline_s: float = DRAIN_DEADLINE_DEFAULT_S):
        self.groups = {"P": Group("P", prefill.rstrip("/"), prefill_sid), "D": Group("D", decode.rstrip("/"), decode_sid)}
        self.awake = awake
        # #1233 one-backup flip: the weights tag family both groups were built
        # with (launcher: SGLANG_WEG2_WEIGHT_CHUNKS), chunks first, base last.
        self.weight_chunks = int(weight_chunks)
        self.weights_tags = weights_family_tags(self.weight_chunks)
        self.tag = tag
        self.store_dir = store_dir
        self.dc_reserve = dc_reserve
        self.w_s = w_s
        self.epoch = 0
        self.state = "serving"  # serving | flipping | STOP
        self.stop: Optional[Weg2Stop] = None
        self.admit_d = True
        self.queue: Deque[Pending] = collections.deque()
        self.spans = SpanLRU()
        self.session: Optional[ClientSession] = None
        self.counters: Dict[str, int] = collections.Counter()
        self.corridor_min: Dict[str, Dict[int, int]] = {"P": {}, "D": {}}
        self.flip_log: List[dict] = []
        self.drain_refusals_in_a_row = 0
        self.identity_checked = False
        self.dc_measured_d: Dict[str, int] = {}
        self.t0 = time.time()
        self._rid = 0
        # ---- WEG2_SCHEDULING_SPEC_0907 slice A, laws 1/2/4/5 ----------------
        # law 1: P's bs is CONCURRENCY ONLY.  The drain below re-reads the
        # deque and exits on empty; this number bounds how many leg-1 POSTs
        # are in flight at once, never how many requests a P phase prefills.
        self.p_concurrency = max(1, int(p_concurrency))
        # law 2: D's own bs, independent of P's by construction (the launcher
        # writes both, R-6/R-12).  It is the front's D concurrency AND D's
        # own --max-running-requests, one number.
        self.d_bs = max(1, int(d_bs))
        # law 4: X.  ONE value for all four front sites (C9); the front's use
        # is an ESTIMATE (no tokenizer here) -- D enforces it for real at
        # get_new_batch_prefill after match_prefix (C11, W31).
        self.tp_prefill_max_tokens = max(1, int(tp_prefill_max_tokens))
        # C7/R-5: the SAME break-even quantity at aggregate granularity --
        # the D->P departure latch.  Defaults to X because it IS X.
        self.flip_min_work_tokens = (
            int(flip_min_work_tokens) if flip_min_work_tokens is not None
            else self.tp_prefill_max_tokens
        )
        # C8/K7: None = derive from the last completed flip in that direction.
        self.min_dwell_ms = None if min_dwell_ms is None else float(min_dwell_ms)
        # law 5 / C6: which group is awake when nothing is pending.
        self.idle_layout = "P" if str(idle_layout).upper().startswith("P") else "D"
        self.drain_deadline_s = float(drain_deadline_s)
        # C4: oldest-first, one seat per running request on D.
        self._ready_for_d: Deque[Pending] = collections.deque()
        self._d_seat = asyncio.Semaphore(self.d_bs)
        # C5/R-16: an asyncio.Semaphore is FIFO among waiters with NO
        # priority, so a SHORT arrival would take a seat ahead of BATCH work
        # that has already been prefilled.  The gate is the explicit
        # priority the semaphore does not have: cleared while _ready_for_d is
        # non-empty, set when it empties.
        self._batch_gate = asyncio.Event()
        self._batch_gate.set()
        self._admitted_this_epoch = 0
        # C12: per-rid W31 re-queue counter, ONE increment site.
        self._x_requeues: Dict[str, int] = {}
        # C8: when the currently awake group woke.  Phase DWELL, not the
        # interval between same-direction flips (R-5).
        self.t_awake = time.time()
        # #1233 zero-remainder: the longest prompt group D can READ from the
        # store (its host staging pool x the prefetch rate bound, launcher-
        # measured from D's log). Above it a BATCH prompt would be prefilled
        # by P and then prefilled AGAIN by D (measured boot weg2ls4b2: 84,027
        # tokens vs a 30,518-token D host pool -> '#915 PREFETCH REFUSED',
        # cached_tokens=0, W16 after 6 min of GPU time). Such a prompt is
        # routed to ONE prefill on D instead (no leg 1) -- served, single
        # prefill, and named as the carrier bound it is.
        self.carrier_max_tokens = int(carrier_max_tokens)
        self.exact_tokens: Dict[str, int] = {}

    # ---------------- seat / gate bookkeeping (C4, C5) ----------------
    def seats_free(self) -> int:
        """Seats not currently held, for the L2/L3 denominators."""
        return max(0, self.d_bs - self._seats_in_use())

    def _seats_in_use(self) -> int:
        return max(0, self.d_bs - self._d_seat._value)

    def _sync_batch_gate(self) -> None:
        """THE ONLY writer of ``_batch_gate`` (C5).

        Called at every site that changes ``_ready_for_d``'s emptiness: the
        controller's append (non-empty -> clear), the admitter's popleft
        (empty -> set) and ``do_stop``'s clear (empty -> set).  One writer
        rather than three ``set()``/``clear()`` calls, so the gate cannot be
        left in a state that contradicts the deque.
        """
        if self._ready_for_d:
            self._batch_gate.clear()
        else:
            self._batch_gate.set()

    # ---------------- lifecycle ----------------
    async def startup(self, app):
        self.session = ClientSession(timeout=ClientTimeout(total=3600))
        app["controller"] = asyncio.create_task(self.controller())
        app["admitter"] = asyncio.create_task(self.d_admitter())
        app["health"] = asyncio.create_task(self.health_poller())
        app["corridor"] = asyncio.create_task(self.corridor_sampler())
        logger.info("WEG2-FRONT up tag=%s awake=%s P=%s D=%s W=%.0f s (operator V1 fairness bound, 0 = off) "
                    "carrier_max_tokens=%d p_concurrency=%d d_bs=%d X=%d flip_min_work_tokens=%d "
                    "idle_layout=%s min_dwell_ms=%s drain_deadline_s=%.0f",
                    self.tag, self.awake, self.groups["P"].url, self.groups["D"].url, self.w_s,
                    self.carrier_max_tokens, self.p_concurrency, self.d_bs, self.tp_prefill_max_tokens,
                    self.flip_min_work_tokens, self.idle_layout,
                    "derived" if self.min_dwell_ms is None else f"{self.min_dwell_ms:.0f}",
                    self.drain_deadline_s)

    async def cleanup(self, app):
        for k in ("controller", "admitter", "health", "corridor"):
            t = app.get(k)
            if t:
                t.cancel()
        if self.session:
            await self.session.close()

    def do_stop(self, name: str, detail: str) -> None:
        if self.state == "STOP":
            return
        self.stop = Weg2Stop(name, detail)
        self.state = "STOP"
        self.counters["stop"] += 1
        logger.error("WEG2 STOP %s -- %s", name, detail)
        for p in list(self.queue) + list(self._ready_for_d):
            if not p.fut.done():
                p.fut.set_exception(self.stop)
        self.queue.clear()
        self._ready_for_d.clear()
        self._sync_batch_gate()

    # ---------------- HTTP handlers ----------------
    async def handle_health(self, request: web.Request) -> web.Response:
        results = {}
        for g in self.groups.values():
            try:
                async with self.session.get(f"{g.url}/health", timeout=ClientTimeout(total=25)) as r:
                    results[g.name] = r.status
            except Exception as e:  # noqa: BLE001
                results[g.name] = f"error: {type(e).__name__}"
        ok = all(v == 200 for v in results.values()) and self.state != "STOP"
        results.update({"state": self.state, "awake": self.awake, "epoch": self.epoch,
                        "stop": str(self.stop) if self.stop else None})
        return web.json_response(results, status=200 if ok else 503)

    async def handle_health_generate(self, request: web.Request) -> web.Response:
        # NEVER to a dormant group (K2).  During a flip: 503 (busy), not a fault.
        if self.state != "serving":
            return web.json_response({"state": self.state, "stop": str(self.stop) if self.stop else None}, status=503)
        g = self.groups[self.awake]
        try:
            async with self.session.get(f"{g.url}/health_generate", timeout=ClientTimeout(total=60)) as r:
                body = await r.read()
                return web.Response(body=body, status=r.status, content_type=r.content_type)
        except Exception as e:  # noqa: BLE001
            return web.json_response({"error": f"{type(e).__name__}: {e}", "group": g.name}, status=503)

    async def handle_state(self, request: web.Request) -> web.Response:
        return web.json_response(self.state_dict())

    def state_dict(self) -> dict:
        return {
            "tag": self.tag, "state": self.state, "awake": self.awake, "epoch": self.epoch,
            "stop": str(self.stop) if self.stop else None, "admit_d": self.admit_d,
            "queue": len(self.queue),
            "outstanding": {g.name: len(g.outstanding) for g in self.groups.values()},
            "served": {g.name: g.served for g in self.groups.values()},
            "counters": dict(self.counters),
            "corridor_min_mib": {k: dict(v) for k, v in self.corridor_min.items()},
            "flips": self.flip_log[-20:],
            "dc_measured_d_mib": self.dc_measured_d,
            "uptime_s": round(time.time() - self.t0, 1),
            "fairness_w_s": self.w_s,
        }

    async def handle_passthrough_get(self, request: web.Request) -> web.Response:
        g = self.groups[self.awake]
        async with self.session.get(f"{g.url}{request.path_qs}") as r:
            body = await r.read()
            return web.Response(body=body, status=r.status, content_type=r.content_type)

    async def handle_session_refused(self, request: web.Request) -> web.Response:
        self.counters["session_refused_501"] += 1
        return web.json_response(
            {"error": "Weg2SessionsRefused: /open_session and /close_session are refused with 501 "
                      "(spec 3.7): session state is per-server and a session opened while D is awake "
                      "does not exist in P. Re-issue with the cached prefix instead."},
            status=501,
        )

    async def handle_abort(self, request: web.Request) -> web.Response:
        payload = await request.json()
        rid = payload.get("rid")
        g = self.groups[self.awake]
        self.counters["abort_requests"] += 1
        for grp in self.groups.values():
            if rid in grp.outstanding:
                grp.outstanding.pop(rid, None)
        for p in list(self.queue):
            if p.rid == rid or p.payload.get("rid") == rid:
                self.queue.remove(p)
                if not p.fut.done():
                    p.fut.set_exception(web.HTTPRequestTimeout(text="aborted"))
        try:
            async with self.session.post(f"{g.url}/abort_request", json=payload) as r:
                body = await r.read()
                return web.Response(body=body, status=r.status, content_type=r.content_type)
        except Exception as e:  # noqa: BLE001
            return web.json_response({"error": str(e)}, status=503)

    async def handle_generate(self, request: web.Request) -> web.StreamResponse:
        if self.state == "STOP":
            return web.json_response({"error": f"WEG2 STOP {self.stop}"}, status=503)
        payload = await request.json()
        self._rid += 1
        rid = f"weg2-{self.epoch}-{self._rid}"
        text = request_text(payload)
        remainder, est_prompt, known = price_remainder(text, self.spans)
        if not known:
            self.counters["W22_Weg2SpanUnknownPricedFull"] += 1
        self.counters["requests"] += 1
        stream = bool(payload.get("stream"))
        exact = self.exact_tokens.get(hashlib.sha1(text.encode(errors="replace")).hexdigest())
        carrier_est = exact if exact else int(len(text) / CARRIER_CHARS_PER_TOKEN) + 1
        if self.carrier_max_tokens > 0 and carrier_est > self.carrier_max_tokens:
            self.counters["route_carrier_exceeds"] += 1
            logger.warning("WEG2-ROUTE rid=%s CARRIER-EXCEEDS -> D single prefill est_prompt=%d exact=%s > carrier_max=%d "
                           "(group D host staging pool bound: the store cannot be read into D for a prompt this long; "
                           "ONE prefill on D, no leg 1, no double prefill)", rid, est_prompt, exact, self.carrier_max_tokens)
            if self.awake == "D" and self.admit_d and self.state == "serving":
                seat = await self._acquire_short_seat(rid)
                if seat is not None:
                    self._log_admit(rid, source="short", t_arrive=time.time())
                    return await self.leg2(request, rid, payload, text, stream, pending=None,
                                           single_prefill=True, seat=seat)
            fut = asyncio.get_event_loop().create_future()
            p = Pending(rid, request.path, payload, text, time.time(), fut, est_prompt=est_prompt, span_known=known, skip_leg1=True)
            self.queue.append(p)
            try:
                await fut
            except Weg2Stop as e:
                return web.json_response({"error": str(e)}, status=503)
            except Exception as e:  # noqa: BLE001
                return web.json_response({"error": f"{type(e).__name__}: {e}"}, status=503)
            self._mark_posted(p)
            return await self.leg2(request, rid, payload, text, stream, pending=p, seat=p.seat)
        short_ok = remainder <= self.tp_prefill_max_tokens
        # L10 (C9): the front's X verdict, LABELLED as the estimate it is --
        # price_remainder is len(text)/3.0 minus an LRU prefix guess, with no
        # tokenizer at the front.  D re-derives the real extent after
        # match_prefix and refuses by name there (L9/W31).
        logger.info("WEG2 X-ROUTE rid=%s est_uncached=%d X=%d (ESTIMATE, front pricing, no tokenizer)",
                    rid, remainder, self.tp_prefill_max_tokens)
        if self.awake == "D" and self.admit_d and self.state == "serving" and short_ok:
            seat = await self._acquire_short_seat(rid)
            if seat is not None:
                self.counters["route_short"] += 1
                logger.info("WEG2-ROUTE rid=%s SHORT -> D est_prompt=%d remainder=%d span_known=%s", rid, est_prompt, remainder, known)
                self._log_admit(rid, source="short", t_arrive=time.time())
                return await self.leg2(request, rid, payload, text, stream, pending=None, seat=seat)
        self.counters["route_batch"] += 1
        if self.awake != "D" and short_ok:
            # L4/R-10: law 1 read literally means every arrival during a P
            # drain is queued BATCH, SHORT ones included.  Counted here,
            # printed by the drain (n=0 printed too), never discovered.
            self.counters["short_behind_p"] += 1
        elif self.awake == "D" and not short_ok:
            # L5: a BATCH arrival during a D phase -- named and left; it is
            # served by the NEXT P phase, whose epoch this line names.
            logger.info("WEG2 LATE-BATCH rid=%s deferred_to_epoch=%d", rid, self.epoch + 1)
        fut: asyncio.Future = asyncio.get_event_loop().create_future()
        p = Pending(rid, request.path, payload, text, time.time(), fut, est_prompt=est_prompt, span_known=known)
        self.queue.append(p)
        logger.info("WEG2-ROUTE rid=%s BATCH queued (awake=%s admit_d=%s est_prompt=%d remainder=%d queue=%d)",
                    rid, self.awake, self.admit_d, est_prompt, remainder, len(self.queue))
        try:
            await fut  # leg 1 done and D awake
        except Weg2Stop as e:
            return web.json_response({"error": str(e)}, status=503)
        except Exception as e:  # noqa: BLE001
            return web.json_response({"error": f"{type(e).__name__}: {e}"}, status=503)
        self._mark_posted(p)
        return await self.leg2(request, rid, payload, text, stream, pending=p, seat=p.seat)

    # ---------------- D admission (C4, C5) ----------------
    @staticmethod
    def _mark_posted(p: Optional[Pending]) -> None:
        """R-15: this request has reached its POST; the admitter may resolve
        the next future.  Set BEFORE leg 2 runs -- the barrier bounds the
        hand-off, not the decode."""
        if p is not None and p.posted_evt is not None and not p.posted_evt.is_set():
            p.posted_evt.set()

    async def _acquire_short_seat(self, rid: str) -> Optional[Seat]:
        """A SHORT arrival's seat -- behind the BATCH gate (C5/R-16).

        Returns ``None`` when the request must fall through to route BATCH:
        either the gate did not open inside the drain deadline, or the phase
        changed while waiting.  Never an unbounded wait (MUST NOT 8): the
        bound is ``--drain-deadline-s``, the same number that already says
        "D is not making progress" everywhere else in this front.
        """
        try:
            await asyncio.wait_for(self._batch_gate.wait(), self.drain_deadline_s)
        except asyncio.TimeoutError:
            self.counters["short_gate_timeout_to_batch"] += 1
            logger.warning("WEG2 SHORT-GATE rid=%s: queued BATCH work held the gate for %.0f s; "
                           "routing BATCH instead of overtaking it", rid, self.drain_deadline_s)
            return None
        if not (self.awake == "D" and self.admit_d and self.state == "serving"):
            return None
        await self._d_seat.acquire()
        if not (self.awake == "D" and self.admit_d and self.state == "serving"):
            self._d_seat.release()
            return None
        return Seat(self, rid, "short")

    def _log_admit(self, rid: str, source: str, t_arrive: float, rank: Optional[int] = None) -> None:
        """L2.  ``rank`` is this admission's ORDINAL in the current epoch.

        Deviation from the spec's wording, stated: the spec asks for the
        ``_ready_for_d`` POSITION, which a ``popleft`` admitter makes
        constantly 0 and therefore unreadable.  The ordinal, printed beside
        ``oldest_wait_s``, is what actually makes oldest-first checkable in
        the log: ordinals ascend while ``t_arrive`` ascends.
        """
        if rank is None:
            rank = self._admitted_this_epoch
        self._admitted_this_epoch += 1
        logger.info("WEG2 D-ADMIT rid=%s seat=%d/%d rank=%d oldest_wait_s=%.1f source=%s",
                    rid, self._seats_in_use(), self.d_bs, rank, max(0.0, time.time() - t_arrive), source)

    async def d_admitter(self) -> None:
        """Law 2: admit the OLDEST first, ``--d-bs`` at a time, refill on a
        freed seat.

        Replaces the release-all loop the controller ran after every
        ``flip("P","D")`` (C4).  That loop resolved every drained request's
        future at once, so D's own bs was the only thing bounding
        concurrency and the ARRIVAL ORDER was lost in the resolution
        stampede.  Here: one seat per running request, ``popleft`` (the
        deque keeps the order the drain popped them in, which is the order
        they arrived), and the seat is returned in ``leg2``'s ``finally``
        -- which is what "wenn ein slot frei wird, wird nachgezogen" means.

        Guarded on ``awake == "D"``, so a W1-refused flip that leaves P
        awake still releases nothing (the 1j finding-1 fix, preserved).
        """
        while True:
            await asyncio.sleep(0.05)
            try:
                if self.state != "serving" or self.awake != "D":
                    continue
                if not self._ready_for_d:
                    continue
                p = self._ready_for_d.popleft()
                self._sync_batch_gate()
                if p.fut.done():
                    # Already resolved, failed or cancelled (leg 1 error, an
                    # abort, a STOP): no seat is spent on it.
                    self.counters["d_admit_skipped_done"] += 1
                    continue
                await self._d_seat.acquire()
                if p.fut.done():
                    self._d_seat.release()
                    self.counters["d_admit_skipped_done"] += 1
                    continue
                p.seat = Seat(self, p.rid, "batch")
                p.posted_evt = asyncio.Event()
                self._log_admit(p.rid, source="batch", t_arrive=p.t_arrive)
                p.fut.set_result(True)
                try:
                    await asyncio.wait_for(p.posted_evt.wait(), POST_BARRIER_S)
                except asyncio.TimeoutError:
                    # W36: the client behind this rid never reached its POST
                    # (disconnected, cancelled).  One dead client may not
                    # stall the queue -- release the seat, count it, go on.
                    self.counters["W36_Weg2AdmitterBarrierExpired"] += 1
                    logger.error("W36 Weg2AdmitterBarrierExpired rid=%s: no POST within %.0f s of the "
                                 "hand-off; seat released, admission continues", p.rid, POST_BARRIER_S)
                    if p.seat is not None:
                        p.seat.release("W36_barrier_expired")
            except asyncio.CancelledError:
                raise
            except Exception as e:  # noqa: BLE001
                logger.exception("d_admitter error: %s", e)

    # ---------------- legs ----------------
    async def leg1(self, p: Pending) -> None:
        g = self.groups["P"]
        payload = dict(p.payload)
        payload.pop("stream", None)
        payload.pop("stream_options", None)
        if p.path == "/generate":
            sp = dict(payload.get("sampling_params") or {})
            sp["max_new_tokens"] = 1
            payload["sampling_params"] = sp
        else:
            payload["max_tokens"] = 1
            payload.pop("max_completion_tokens", None)
        g.outstanding[p.rid] = time.time()
        t0 = time.time()
        try:
            async with self.session.post(f"{g.url}{p.path}", json=payload) as r:
                body = await r.read()
                if r.status != 200:
                    raise RuntimeError(f"leg1 on P returned {r.status}: {body[:300]!r}")
                try:
                    js = json.loads(body)
                except Exception:  # noqa: BLE001
                    js = {}
                pt, ct, _, _ = usage_of(js)
                p.leg1_prompt_tokens = pt
                self.spans.record(p.text, pt)
                self._note_exact(p.text, pt)
                if self.carrier_max_tokens > 0 and pt > self.carrier_max_tokens and not p.skip_leg1:
                    # The realised count says D cannot read this prompt from the
                    # store (host staging pool bound): leg 2 is ONE prefill on D,
                    # not a reroute/W16 loop. P's prefill was spent; counted.
                    p.skip_leg1 = True
                    self.counters["carrier_exceeds_after_leg1"] += 1
                    logger.warning("WEG2-ROUTE rid=%s CARRIER-EXCEEDS after leg 1: prompt_tokens=%d > carrier_max=%d; leg 2 = single prefill on D",
                                   p.rid, pt, self.carrier_max_tokens)
                g.served += 1
                logger.info("WEG2-SERVED group=P leg=1 rid=%s prompt_tokens=%d cached_tokens=%d wall=%.2fs epoch=%d",
                            p.rid, pt, ct, time.time() - t0, self.epoch)
        finally:
            g.outstanding.pop(p.rid, None)

    def _note_exact(self, text: str, prompt_tokens: int) -> None:
        if prompt_tokens <= 0:
            return
        if len(self.exact_tokens) >= SPAN_LRU:
            self.exact_tokens.pop(next(iter(self.exact_tokens)))
        self.exact_tokens[hashlib.sha1(text.encode(errors="replace")).hexdigest()] = int(prompt_tokens)

    def _leg2_verdict(self, pt: int, ct: int, priced: bool, pending: Optional[Pending],
                      single_prefill: bool, stream: bool, rid: str) -> str:
        """spec 3.6 verdict with the 1j holes closed (#1233 zero-remainder).

        * single_prefill (route CARRIER-EXCEEDS): no leg 1 ever ran, so there
          is no double prefill to price -- 'single_prefill', never W16;
        * pending is None (route SHORT): no leg 1 either; a mis-priced SHORT
          is counted as 'short_mispriced' and served, never logged as W16
          (finding 4: SHORT passed reroutes=1 and produced verdict=W16 on a
          served request);
        * a stream cannot be re-routed or refused after its first byte, so a
          streamed BATCH leg 2 whose realised usage exceeds the bound is
          counted under W16 by name and reported, not refused (finding 2).
        """
        uncached = max(0, pt - ct)
        if single_prefill:
            self.counters["single_prefill_served"] += 1
            return "single_prefill"
        if pending is None:
            if uncached > self.tp_prefill_max_tokens:
                self.counters["short_mispriced"] += 1
                return "short_mispriced"
            return "serve"
        if not priced:
            self.counters["W28_Weg2Leg2Unpriced_stream_served"] += 1
            logger.error("W28 Weg2Leg2Unpriced rid=%s: STREAMED leg 2 ended without a usage/meta_info chunk; served, unpriced, counted", rid)
            return "unpriced"
        v = double_prefill_verdict(pt, ct, pending.reroutes, self.tp_prefill_max_tokens)
        if stream and v != "serve":
            self.counters["W16_Weg2DoublePrefillExceeded"] += 1
            self.counters["W16_stream_served"] += 1
            logger.error("W16 Weg2DoublePrefillExceeded rid=%s (STREAM, served): %d > %d uncached on a streamed leg 2 -- "
                         "reroute impossible after the first byte; counted by name, not refused",
                         rid, uncached, self.tp_prefill_max_tokens)
            return "W16"
        return v

    async def leg2(self, request: web.Request, rid: str, payload: dict, text: str, stream: bool,
                   pending: Optional[Pending], single_prefill: bool = False,
                   seat: Optional[Seat] = None) -> web.StreamResponse:
        g = self.groups["D"]
        g.outstanding[rid] = time.time()
        t0 = time.time()
        if pending is not None and pending.skip_leg1:
            single_prefill = True
        if stream and pending is not None and request.path.startswith("/v1/"):
            # 1j finding 2: a STREAMED leg 2 is priced like a non-streamed one.
            # OpenAI's stream_options.include_usage makes D append one usage
            # chunk (empty choices) -- standard, and the only post-hoc price.
            payload = dict(payload)
            so = dict(payload.get("stream_options") or {})
            so["include_usage"] = True
            payload["stream_options"] = so
        try:
            async with self.session.post(f"{g.url}{request.path}", json=payload) as r:
                if stream:
                    resp = web.StreamResponse(status=r.status)
                    resp.content_type = r.content_type
                    await resp.prepare(request)
                    tail = bytearray()
                    async for chunk in r.content.iter_any():
                        await resp.write(chunk)
                        tail += chunk
                        if len(tail) > 262144:
                            del tail[:-131072]
                    await resp.write_eof()
                    g.served += 1
                    pt, ct, comp, priced = usage_of_stream_tail(bytes(tail))
                    verdict = self._leg2_verdict(pt, ct, priced, pending, single_prefill, True, rid)
                    if pt:
                        self.spans.record(text, pt)
                        self._note_exact(text, pt)
                    logger.info("WEG2-SERVED group=D leg=2 rid=%s stream=1 status=%d prompt_tokens=%d cached_tokens=%d completion_tokens=%d "
                                "uncached=%d verdict=%s priced=%s wall=%.2fs epoch=%d", rid, r.status, pt, ct, comp, max(0, pt - ct),
                                verdict, priced, time.time() - t0, self.epoch)
                    if pending is not None and ct > 0:
                        self.counters["cross_group_prefix_hits"] += 1
                    return resp
                body = await r.read()
                try:
                    js = json.loads(body)
                except Exception:  # noqa: BLE001
                    js = {}
                if is_x_refusal(r.status, body.decode(errors="replace")):
                    # C11/C12 -- W31 came back from D's own gate, where the
                    # UNCACHED EXTENT IS REAL (after match_prefix).  Law 4
                    # says such a request is prefilled by P, so it re-joins
                    # route BATCH; it is never silently prefilled on D and
                    # never re-offered to D a third time (W35).
                    g.outstanding.pop(rid, None)
                    return await self._requeue_after_x_refusal(
                        request, rid, payload, text, stream, pending, seat, body
                    )
                pt, ct, comp, priced = usage_of(js)
                if r.status == 200 and not priced:
                    # 1j finding 3: never price a body without usage/meta_info as (0,0).
                    self.counters["W28_Weg2Leg2Unpriced"] += 1
                    logger.error("W28 Weg2Leg2Unpriced rid=%s: D answered 200 without usage/meta_info (%d bytes); refusing by name "
                                 "rather than serving on a fail-open (0,0) price", rid, len(body))
                    return web.json_response({"error": f"W28 Weg2Leg2Unpriced rid={rid}"}, status=503)
                verdict = self._leg2_verdict(pt, ct, priced, pending, single_prefill, False, rid)
                g.served += 1
                logger.info("WEG2-SERVED group=D leg=2 rid=%s status=%d prompt_tokens=%d cached_tokens=%d completion_tokens=%d "
                            "uncached=%d verdict=%s wall=%.2fs epoch=%d", rid, r.status, pt, ct, comp, max(0, pt - ct),
                            verdict, time.time() - t0, self.epoch)
                if pt:
                    self.spans.record(text, pt)
                    self._note_exact(text, pt)
                if r.status == 200 and pending is not None and verdict == "reroute":
                    self.counters["reroute"] += 1
                    pending.reroutes += 1
                    pending.fut = asyncio.get_event_loop().create_future()
                    pending.t_arrive = time.time()
                    # The seat goes back BEFORE the re-queue: the admitter
                    # takes a fresh one when this rid is admitted again, and
                    # a request that holds two seats has taken a running
                    # slot from another request for its whole round trip.
                    if seat is not None:
                        seat.release("reroute")
                    pending.seat = None
                    self.queue.append(pending)
                    logger.warning("WEG2-REROUTE rid=%s uncached=%d > %d: rejoining route BATCH once (spec 3.6)",
                                   rid, pt - ct, self.tp_prefill_max_tokens)
                    g.outstanding.pop(rid, None)
                    try:
                        await pending.fut
                    except Weg2Stop as e:
                        return web.json_response({"error": str(e)}, status=503)
                    self._mark_posted(pending)
                    return await self.leg2(request, rid, payload, text, stream, pending, seat=pending.seat)
                if r.status == 200 and pending is not None and verdict == "W16":
                    self.counters["W16_Weg2DoublePrefillExceeded"] += 1
                    logger.error("W16 Weg2DoublePrefillExceeded rid=%s: re-routed once and the prefix is still %d > %d uncached; refusing and reporting",
                                 rid, pt - ct, self.tp_prefill_max_tokens)
                    return web.json_response({"error": f"W16 Weg2DoublePrefillExceeded rid={rid} uncached={pt - ct}"}, status=503)
                if pending is not None and ct > 0:
                    self.counters["cross_group_prefix_hits"] += 1
                if not self.identity_checked and pending is not None:
                    self.check_identity()
                return web.Response(body=body, status=r.status, content_type=r.content_type)
        except Weg2Stop:
            raise
        except Exception as e:  # noqa: BLE001
            self.counters["leg2_failures"] += 1
            logger.error("WEG2 leg2 rid=%s failed: %s: %s", rid, type(e).__name__, e)
            return web.json_response({"error": f"{type(e).__name__}: {e}"}, status=503)
        finally:
            g.outstanding.pop(rid, None)
            if seat is not None:
                # C4: THE refill point.  Every exit of leg 2 -- served,
                # refused, raised, cancelled -- passes here, so a freed seat
                # is always visible to the admitter within one tick.
                seat.release("leg2_finished")

    async def _requeue_after_x_refusal(self, request: web.Request, rid: str, payload: dict, text: str,
                                       stream: bool, pending: Optional[Pending], seat: Optional[Seat],
                                       body: bytes) -> web.StreamResponse:
        """W31 came back from D: re-queue BATCH once, then W35 (C12).

        The counter is per rid and has exactly ONE increment site, here, so
        the "never a third pass" bound cannot be widened by a second writer.
        """
        n = self._x_requeues.get(rid, 0) + 1
        self._x_requeues[rid] = n
        if pending is not None:
            pending.x_requeues = n
        self.counters["W31_Weg2TpPrefillExceeded"] += 1
        logger.warning("WEG2 X-REQUEUE rid=%s n=%d verdict=%s", rid, n,
                       "requeue" if n <= 1 else "W35")
        if n > 1:
            self.counters["W35_Weg2XReQueueLoop"] += 1
            logger.error("W35 Weg2XReQueueLoop rid=%s: D refused this rid with W31 a second time after a full "
                         "P prefill; refusing by name rather than a third pass. D said: %s",
                         rid, body.decode(errors="replace")[:400])
            if seat is not None:
                seat.release("W35")
            return web.json_response({"error": f"W35 Weg2XReQueueLoop rid={rid}"}, status=503)
        p = pending
        if p is None:
            # A SHORT (or CARRIER-EXCEEDS) arrival the front mis-priced: it
            # has no Pending, so it gets one now and joins the BATCH queue.
            # This is the weg2zr2 `weg2-4-8` shape (front log line 186:
            # est remainder 0, realised uncached 19,401) -- served through a
            # 4,096-token grant then, refused by name now.
            p = Pending(rid, request.path, payload, text, time.time(),
                        asyncio.get_event_loop().create_future(),
                        est_prompt=len(text) // int(CHARS_PER_TOKEN) + 1, span_known=False)
            p.x_requeues = n
        else:
            p.fut = asyncio.get_event_loop().create_future()
            p.t_arrive = time.time()
        if seat is not None:
            seat.release("W31_requeue")
        p.seat = None
        self.queue.append(p)
        try:
            await p.fut
        except Weg2Stop as e:
            return web.json_response({"error": str(e)}, status=503)
        except Exception as e:  # noqa: BLE001
            return web.json_response({"error": f"{type(e).__name__}: {e}"}, status=503)
        self._mark_posted(p)
        return await self.leg2(request, rid, payload, text, stream, pending=p, seat=p.seat)

    def check_identity(self) -> None:
        """W9 from the store directory: exactly ONE identity suffix on disk."""
        self.identity_checked = True
        try:
            names = [n for n in os.listdir(self.store_dir) if n.endswith(".bin")][:20000]
        except OSError:
            return
        ids = set()
        for n in names:
            stem = n[:-4]
            parts = stem.split("_")
            if len(parts) >= 3:
                ids.add(parts[-1])
        logger.info("W9 identity check from store dir %s: %d files, identity suffixes %s", self.store_dir, len(names), sorted(ids))
        if len(ids) > 1:
            self.do_stop("W9 Weg2StoreIdentityMismatch", f"identity suffixes on the sole carrier: {sorted(ids)}")

    # ---------------- flip machinery ----------------
    async def rpc(self, g: Group, path: str, body: Optional[dict], timeout: float) -> Tuple[int, str]:
        try:
            async with self.session.post(f"{g.url}{path}", json=body or {}, timeout=ClientTimeout(total=timeout)) as r:
                return r.status, (await r.read()).decode(errors="replace")
        except Exception as e:  # noqa: BLE001
            return 0, f"{type(e).__name__}: {e}"

    async def drain(self, g: Group) -> bool:
        t0 = time.time()
        while g.outstanding:
            if time.time() - t0 > self.drain_deadline_s:
                return False
            await asyncio.sleep(0.25)
        return True

    async def quiesce(self, g: Group) -> Tuple[bool, str]:
        """Witness B: poll /flush_cache (200 iff the rank's is_fully_idle,
        including the HiCache in-flight terms) to a deadline."""
        t0 = time.time()
        last = ""
        while time.time() - t0 < QUIESCE_DEADLINE_S:
            code, body = await self.rpc(g, "/flush_cache", None, 60)
            if code == 200:
                return True, body
            last = body
            await asyncio.sleep(0.5)
        return False, last

    async def flip(self, src: str, dst: str) -> None:
        S, D = self.groups[src], self.groups[dst]
        self.state = "flipping"
        t_flip0 = time.time()
        logger.info("WEG2-FLIP begin epoch=%d sleep=%s wake=%s outstanding=%d queue=%d", self.epoch, src, dst, len(S.outstanding), len(self.queue))
        # 1. drain (W1/W2)
        if not await self.drain(S):
            self.drain_refusals_in_a_row += 1
            self.counters["W1_Weg2DrainRefused"] += 1
            logger.error("W1 Weg2DrainRefused: %s still holds %d request(s) after %.0f s (rids %s) suspended=%d; "
                         "not flipping (%d in a row)",
                         src, len(S.outstanding), self.drain_deadline_s, sorted(S.outstanding)[:8],
                         self.counters.get("suspended_now", 0), self.drain_refusals_in_a_row)
            if self.drain_refusals_in_a_row >= 3:
                self.do_stop("W2 Weg2DrainStuck", f"three W1 in a row on {src}: {sorted(S.outstanding)[:8]}")
            self.state = "serving" if self.state != "STOP" else "STOP"
            return
        self.drain_refusals_in_a_row = 0
        # 2. quiesce + double witness (W3)
        idle, msg = await self.quiesce(S)
        wv = witness_verdict(len(S.outstanding), idle)
        if wv is not None:
            self.do_stop("W3 Weg2DrainWitnessDisagreement",
                         f"{wv}: front ledger {sorted(S.outstanding)} vs rank({src}) flush_cache -> {msg[:400]!r}")
            return
        t_q = time.time()
        # 3. THE ONE-BACKUP INTERLEAVE (#1233, record 1h): src.pause(kv_cache),
        # then per weights tag k: src.pause(w_k) -> dst.resume(w_k) (the
        # patched saver frees dst's chunk image on resume), then
        # dst.resume(kv_cache).  The host holds at most ONE full image (the
        # dormant group's) plus ONE chunk at any moment.  Serial by
        # construction: the D2H of chunk k+1 never overlaps the H2D of chunk
        # k (the rank-side PCIe lock would serialise them anyway).
        sleep_ms = 0.0
        wake_ms = 0.0
        chunk_recs: List[dict] = []
        t0 = time.time()
        code, body = await self.rpc(S, "/release_memory_occupation", {"tags": [KV_TAG]}, RPC_TIMEOUT_S)
        sleep_ms += (time.time() - t0) * 1000
        if code != 200:
            self.do_stop("W4 Weg2WakeRefused", f"sleep({src}, {KV_TAG}) failed HTTP {code}: {body[:400]!r} -- VRAM state undefined, no retry")
            return
        for tag in self.weights_tags:
            t0 = time.time()
            code, body = await self.rpc(S, "/release_memory_occupation", {"tags": [tag]}, RPC_TIMEOUT_S)
            t1 = time.time()
            if code != 200:
                self.do_stop("W4 Weg2WakeRefused", f"sleep({src}, {tag}) failed HTTP {code}: {body[:400]!r} -- VRAM state undefined, no retry")
                return
            code, body = await self.rpc(D, "/resume_memory_occupation", {"tags": [tag]}, RPC_TIMEOUT_S)
            t2 = time.time()
            if code != 200:
                self.do_stop("W4 Weg2WakeRefused", f"wake({dst}, {tag}) failed HTTP {code}: {body[:400]!r} -- group-fatal, recovery = teardown + relaunch")
                return
            sleep_ms += (t1 - t0) * 1000
            wake_ms += (t2 - t1) * 1000
            chunk_recs.append({"tag": tag, "sleep_ms": round((t1 - t0) * 1000), "wake_ms": round((t2 - t1) * 1000)})
            logger.info("WEG2-FLIP-CHUNK epoch=%d tag=%s %s.pause=%d ms %s.resume=%d ms", self.epoch, tag, src, chunk_recs[-1]["sleep_ms"], dst, chunk_recs[-1]["wake_ms"])
        t_s = time.time()
        # 4. measure D_c(src); W19 for D at its first sleep
        pids = _session_pids(S.sid) if S.sid else set()
        dc = _nvml_process_mib(pids) if pids else {}
        for uuid, mib in sorted(dc.items()):
            logger.info("WEG2-DC group=%s uuid=%s measured=%d MiB reserve=%s", src, uuid, mib, self.dc_reserve.get(uuid))
        if src == "D" and not self.dc_measured_d and dc:
            self.dc_measured_d = dc
            over = {u: (m, self.dc_reserve.get(u)) for u, m in dc.items() if self.dc_reserve.get(u) is not None and m > self.dc_reserve[u]}
            if over:
                self.do_stop("W19 DormantResidueRefused",
                             f"measured D_c(D) exceeds the reserve P's budget assumed: {over} (measured, reserved) MiB -- waking P would overcommit the card")
                return
        # 5. wake dst kv (W4)
        t0 = time.time()
        code, body = await self.rpc(D, "/resume_memory_occupation", {"tags": [KV_TAG]}, RPC_TIMEOUT_S)
        t_w = time.time()
        wake_ms += (t_w - t0) * 1000
        if code != 200:
            self.do_stop("W4 Weg2WakeRefused", f"wake({dst}, {KV_TAG}) failed HTTP {code}: {body[:400]!r} -- group-fatal, recovery = teardown + relaunch")
            return
        self.awake = dst
        self.epoch += 1
        self.admit_d = True
        self.state = "serving"
        # C8: phase dwell restarts here, and the L2 admission ordinal with it.
        self.t_awake = time.time()
        self._admitted_this_epoch = 0
        rec = {"epoch": self.epoch, "sleep": src, "wake": dst, "drain_quiesce_ms": round((t_q - t_flip0) * 1000),
               "sleep_ms": round(sleep_ms), "wake_ms": round(wake_ms), "flip_ms": round((t_w - t_flip0) * 1000),
               "interleave_ms": round((t_s - t_q) * 1000), "chunks": chunk_recs,
               "dc_mib": dc, "t": time.time()}
        self.flip_log.append(rec)
        self.counters["flips"] += 1
        logger.info("WEG2-FLIP done epoch=%d slept=%s woke=%s drain+quiesce=%d ms sleep=%d ms (sum of %d %s RPCs) wake=%d ms (sum of %d %s RPCs) "
                    "interleave=%d ms flip_total=%d ms weights_tags=%d dc=%s",
                    rec["epoch"], src, dst, rec["drain_quiesce_ms"], rec["sleep_ms"], len(self.weights_tags) + 1, src,
                    rec["wake_ms"], len(self.weights_tags) + 1, dst, rec["interleave_ms"], rec["flip_ms"], len(self.weights_tags), dc)

    # ---------------- phase economics (C7, C8) ----------------
    def _derived_min_dwell_ms(self, src: str, dst: str) -> Tuple[float, str]:
        """K7: how long ``src`` must have been awake before it may leave.

        DERIVED from the last completed flip in the SAME direction -- the
        price of the round trip this flip would start -- not from the
        interval between same-direction flips (R-5: the measured weg2zr2
        thrash was a 29.4 s round trip around a 4.3 s P phase, which an
        interval latch does not see).  0 before the first flip, and the
        provenance string says which of the two it is.
        """
        if self.min_dwell_ms is not None:
            return self.min_dwell_ms, "flag"
        for rec in reversed(self.flip_log):
            if rec.get("sleep") == src and rec.get("wake") == dst:
                return float(rec.get("flip_ms") or 0.0), f"last-flip-{src}->{dst}"
        return 0.0, "none-first-flip"

    def _dwell_ok(self, src: str, dst: str, fairness_fired: bool,
                  work_exhausted: bool, oldest_wait_s: float) -> bool:
        """C8, with its two NAMED overrides.

        The fairness bound W wins over min-dwell, and min-dwell never holds
        a phase whose work is exhausted while the opposite queue's oldest
        has already waited >= W.  Prints L12 on every evaluation, including
        the ones that hold, so a short phase is never a mystery.
        """
        need, prov = self._derived_min_dwell_ms(src, dst)
        awake_ms = (time.time() - self.t_awake) * 1000.0
        overridden = "none"
        if fairness_fired:
            overridden = "fairness"
        elif work_exhausted and self.w_s > 0 and oldest_wait_s >= self.w_s:
            overridden = "work"
        ok = awake_ms >= need or overridden != "none"
        logger.info("WEG2 MIN-DWELL src=%s dst=%s awake_ms=%d derived_from_flip_ms=%d overridden_by=%s "
                    "provenance=%s verdict=%s",
                    src, dst, int(awake_ms), int(need), overridden, prov, "flip" if ok else "hold")
        return ok

    def _flip_economics_ok(self, fairness_fired: bool) -> bool:
        """C7/L13: is the queued work worth a round trip?

        The PRIMARY anti-thrash, and the one the measured 4.3 s P phase
        needed.  The threshold is X at aggregate granularity -- the same
        break-even quantity law 4 applies per request.
        """
        queued_tokens = sum(int(p.est_prompt) for p in self.queue)
        ok = queued_tokens >= self.flip_min_work_tokens or fairness_fired or not self.admit_d
        logger.info("WEG2 FLIP-ECONOMICS queued_tokens=%d threshold=%d fairness=%s verdict=%s",
                    queued_tokens, self.flip_min_work_tokens, fairness_fired, "flip" if ok else "hold")
        return ok

    def _fairness_switch(self, oldest_arrival: Optional[float], queue_name: str) -> bool:
        """A1-1: the ONE sanctioned pre-emption, and it names itself.

        ``--fairness-w-s 0`` disables it.  Every fire prints the switch, the
        oldest wait and the queue it pre-empted; nothing else in this front
        may pre-empt a phase.
        """
        if self.w_s <= 0 or not self.admit_d:
            return not self.admit_d
        if not fairness_reached(oldest_arrival, time.time(), self.w_s):
            return False
        self.admit_d = False
        self.counters["fairness_bound_hits"] += 1
        logger.warning("WEG2-FAIRNESS switch=--fairness-w-s value=%.0f s FIRED: oldest %s waiter has waited "
                       "%.1f s (queue=%s, n=%d); stop admitting NEW work to D, drain the running decodes, "
                       "then flip. This is the only sanctioned pre-emption (A1-1); 0 disables it.",
                       self.w_s, queue_name, time.time() - (oldest_arrival or time.time()),
                       queue_name, len(self.queue))
        return True

    async def controller(self) -> None:
        sem = asyncio.Semaphore(self.p_concurrency)
        while True:
            await asyncio.sleep(0.2)
            try:
                if self.state != "serving":
                    continue
                if self.awake == "D":
                    D = self.groups["D"]
                    oldest = self.queue[0].t_arrive if self.queue else None
                    fairness_fired = self._fairness_switch(oldest, "batch")
                    if not self.queue:
                        # law 5 / C6 / R-34: the idle mirror.  D->P at rest
                        # only under --idle-layout pp, only when D holds
                        # nothing (so no request is handed to a sleeping
                        # group) and only when the dwell latch has expired
                        # -- an ungated mirror makes every SHORT arrival pay
                        # two flips (~30 s measured).
                        if (self.idle_layout == "P" and not D.outstanding
                                and not self._ready_for_d and self.state == "serving"):
                            logger.info("WEG2 IDLE-REST layout=%s queue=0 ready_for_d=%d d_outstanding=%d held_s=%.1f",
                                        self.awake, len(self._ready_for_d), len(D.outstanding),
                                        time.time() - self.t_awake)
                            if self._dwell_ok("D", "P", fairness_fired, work_exhausted=True, oldest_wait_s=0.0):
                                await self.flip("D", "P")
                        continue
                    if not D.outstanding or not self.admit_d:
                        if not self._flip_economics_ok(fairness_fired):
                            continue
                        if not self._dwell_ok("D", "P", fairness_fired, work_exhausted=not D.outstanding,
                                              oldest_wait_s=time.time() - (oldest or time.time())):
                            continue
                        await self.flip("D", "P")
                    continue
                # awake == P: prefill the backlog until empty (#1011 PP exit
                # clock).  LAW 1: the phase ends on an EMPTY queue, never on
                # p_concurrency and never on a timer -- p_concurrency bounds
                # only how many leg-1 POSTs are in flight at once.
                t_drain0 = time.time()
                queue_at_entry = len(self.queue)
                prefilled = 0
                passes = 0
                short_behind_p0 = self.counters.get("short_behind_p", 0)
                while self.queue and self.state == "serving":
                    passes += 1
                    batch = [self.queue.popleft()
                             for _ in range(min(self.p_concurrency, len(self.queue)))]

                    async def one(p: Pending):
                        if p.skip_leg1:  # route CARRIER-EXCEEDS: no leg 1, D prefills once
                            p.leg1_done = True
                            return
                        async with sem:
                            try:
                                await self.leg1(p)
                            except Exception as e:  # noqa: BLE001
                                self.counters["leg1_failures"] += 1
                                logger.error("WEG2 leg1 rid=%s failed: %s", p.rid, e)
                                if not p.fut.done():
                                    p.fut.set_exception(e)
                                return
                            p.leg1_done = True
                    await asyncio.gather(*(one(p) for p in batch))
                    for p in batch:
                        if not p.fut.done():
                            self._ready_for_d.append(p)
                            self._sync_batch_gate()
                            prefilled += 1
                if passes:
                    oldest_short = 0.0
                    if self._ready_for_d:
                        oldest_short = time.time() - self._ready_for_d[0].t_arrive
                    logger.info("WEG2 P-DRAIN epoch=%d prefilled=%d arrived_during=%d p_concurrency=%d "
                                "passes=%d queue_at_exit=%d drain_s=%.1f",
                                self.epoch, prefilled, max(0, prefilled - queue_at_entry),
                                self.p_concurrency, passes, len(self.queue), time.time() - t_drain0)
                    # L4/R-10: the counted consequence of law 1 -- SHORT work
                    # that arrived while P was draining and had to queue
                    # BATCH.  n=0 is printed too, so absence is a reading.
                    logger.info("WEG2 SHORT-BEHIND-P epoch=%d n=%d oldest_wait_s=%.1f",
                                self.epoch, self.counters.get("short_behind_p", 0) - short_behind_p0,
                                oldest_short)
                # C6/R-2: the _ready_for_d term is NOT optional.  Without it,
                # --idle-layout pp keeps P awake with requests P has just
                # prefilled sitting on `await fut` behind a one-hour client
                # timeout -- a LOST-REQUEST class introduced by the fix for
                # law 5.  The admitter (C4) releases them once D is awake.
                if self.queue or self._ready_for_d or self.idle_layout == "D":
                    await self.flip("P", "D")
                else:
                    logger.info("WEG2 IDLE-REST layout=%s queue=0 ready_for_d=%d d_outstanding=%d held_s=%.1f",
                                self.awake, len(self._ready_for_d),
                                len(self.groups["D"].outstanding), time.time() - self.t_awake)
            except Weg2Stop as e:
                self.do_stop(e.name, e.detail)
            except Exception as e:  # noqa: BLE001
                logger.exception("controller error: %s", e)

    async def health_poller(self) -> None:
        while True:
            await asyncio.sleep(15)
            for g in self.groups.values():
                ok = False
                try:
                    async with self.session.get(f"{g.url}/health", timeout=ClientTimeout(total=25)) as r:
                        ok = r.status == 200
                except Exception:  # noqa: BLE001
                    ok = False
                alive = _sid_alive(g.sid)
                if ok and alive:
                    g.health_fail_streak = 0
                    continue
                g.health_fail_streak += 1
                logger.warning("WEG2-HEALTH group=%s http_ok=%s process_alive=%s streak=%d", g.name, ok, alive, g.health_fail_streak)
                if g.health_fail_streak >= 2 and not health_is_serving_fact(ok, alive):
                    self.do_stop("W17 Weg2GroupDead", f"group {g.name}: /health failed {g.health_fail_streak}x and process_alive={alive} (a 200 alone is a transport fact)")

    async def corridor_sampler(self) -> None:
        while True:
            await asyncio.sleep(10)
            if self.state != "serving":
                continue
            phase = self.awake
            for idx, uuid, free in _nvml_free():
                cur = self.corridor_min[phase].get(idx)
                self.corridor_min[phase][idx] = free if cur is None else min(cur, free)
            logger.info("WEG2-CORRIDOR phase=%s(awake) epoch=%d %s min_so_far=%s", phase, self.epoch,
                        " ".join(f"nvml{idx}:free={free}MiB" for idx, _, free in _nvml_free()),
                        dict(self.corridor_min[phase]))

    async def handle_manual_flip(self, request: web.Request) -> web.Response:
        if self.state != "serving":
            return web.json_response({"error": self.state}, status=503)
        src, dst = self.awake, ("P" if self.awake == "D" else "D")
        self.admit_d = False
        await self.flip(src, dst)
        if self.awake == "P" and self.state == "serving":
            await self.flip("P", "D")
        return web.json_response(self.state_dict())


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--prefill", required=True)
    ap.add_argument("--decode", required=True)
    ap.add_argument("--awake", choices=["P", "D"], default="D")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=30030)
    ap.add_argument("--tag", default="weg2")
    ap.add_argument("--store-dir", default="")
    ap.add_argument("--prefill-sid", type=int, default=0)
    ap.add_argument("--decode-sid", type=int, default=0)
    ap.add_argument("--dc-reserve", default="", help="uuid=mib,uuid=mib")
    ap.add_argument("--fairness-w-s", type=float, default=45.0,
                    help="A1-1: the ONLY sanctioned pre-emption of a P drain or a D exhaustion. "
                         "Seconds the oldest waiter may wait before the front stops admitting new "
                         "work to D and flips. 0 DISABLES it. Every fire names this switch, the "
                         "oldest wait and the queue it pre-empted.")
    ap.add_argument("--p-concurrency", type=int, default=8,
                    help="law 1 (K3): how many leg-1 POSTs group P runs at once. CONCURRENCY ONLY -- "
                         "the P phase ends when the backlog is empty, never on this number. Written "
                         "by the launcher from --p-bs; the front never derives it over HTTP.")
    ap.add_argument("--d-bs", type=int, default=8,
                    help="law 2 (K4): group D's own batch size, independent of P's. It is both D's "
                         "--max-running-requests and the number of front seats, so the front cannot "
                         "hand D more concurrent requests than D can run. Written by the launcher.")
    ap.add_argument("--tp-prefill-max-tokens", type=int, default=X_FALLBACK_TOKENS,
                    help="law 4 (K5, X): uncached tokens D may prefill itself before the round trip "
                         "through P is cheaper. DERIVED by the launcher as 2*flip_s/(1/r_D - 1/r_P) "
                         "from this boot's own rate and flip lines (floor = D's --chunked-prefill-size); "
                         "the front's use of it is an ESTIMATE (no tokenizer here) and D enforces it "
                         "for real after match_prefix (W31).")
    ap.add_argument("--flip-min-work-tokens", type=int, default=None,
                    help="C7/K6: queued prompt tokens that make a D->P round trip worth its cost. "
                         "Defaults to --tp-prefill-max-tokens because it IS the same break-even "
                         "quantity at aggregate granularity. The primary anti-thrash latch.")
    ap.add_argument("--min-dwell-ms", type=float, default=None,
                    help="C8/K7: how long a group must have been AWAKE before it may leave (phase "
                         "DWELL, not the interval between same-direction flips). Unset = derived "
                         "from the last completed flip in that direction; 0 before the first flip. "
                         "Never overrides the fairness bound and never holds an exhausted phase.")
    ap.add_argument("--idle-layout", choices=["D", "P"], default="D",
                    help="law 5 (K8): which group is awake when nothing is pending. D = today's "
                         "unconditional flip back to the decode group; P = rest on the prefill "
                         "group. The idle guard always includes the requests P has just prefilled, "
                         "so no request is left behind a sleeping group.")
    ap.add_argument("--drain-deadline-s", type=float, default=DRAIN_DEADLINE_DEFAULT_S,
                    help="C13/K10: seconds a group may still hold requests before a flip is refused "
                         "by name (W1 -> W2). Today's shipped value, promoted from a literal.")
    ap.add_argument("--weight-chunks", type=int, default=0, help="#1233: number of weights_<k> chunk tags both groups were built with (0 = single weights tag)")
    ap.add_argument("--carrier-max-tokens", type=int, default=0, help="#1233 zero-remainder: longest prompt group D can read from the store (0 = no CARRIER-EXCEEDS route)")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(levelname)s %(name)s: %(message)s")
    dc = {}
    for kv in filter(None, args.dc_reserve.split(",")):
        k, v = kv.split("=")
        dc[k] = int(v)
    front = Front(args.prefill, args.decode, args.awake, args.tag, args.store_dir, args.prefill_sid, args.decode_sid, dc, args.fairness_w_s,
                  weight_chunks=args.weight_chunks, carrier_max_tokens=args.carrier_max_tokens,
                  p_concurrency=args.p_concurrency, d_bs=args.d_bs,
                  tp_prefill_max_tokens=args.tp_prefill_max_tokens,
                  flip_min_work_tokens=args.flip_min_work_tokens,
                  min_dwell_ms=args.min_dwell_ms, idle_layout=args.idle_layout,
                  drain_deadline_s=args.drain_deadline_s)
    app = web.Application(client_max_size=1024**3)
    app.on_startup.append(front.startup)
    app.on_cleanup.append(front.cleanup)
    app.router.add_get("/health", front.handle_health)
    app.router.add_get("/health_generate", front.handle_health_generate)
    app.router.add_get("/weg2/state", front.handle_state)
    app.router.add_get("/metrics_summary", front.handle_state)
    app.router.add_post("/weg2/flip", front.handle_manual_flip)
    app.router.add_post("/abort_request", front.handle_abort)
    app.router.add_post("/open_session", front.handle_session_refused)
    app.router.add_post("/close_session", front.handle_session_refused)
    for path in PASSTHROUGH_GET:
        app.router.add_get(path, front.handle_passthrough_get)
    for path in FORWARD_PATHS:
        app.router.add_post(path, front.handle_generate)
    logger.info("WEG2-FRONT %s:%s -> P=%s D=%s awake=%s weights_tags=%s", args.host, args.port, args.prefill, args.decode, args.awake, front.weights_tags)
    web.run_app(app, host=args.host, port=args.port, print=None)


if __name__ == "__main__":
    main()
