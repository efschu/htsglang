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

from sglang.srt.managers.weg2_memory_saver import (
    WEIGHT_CHUNK_PREFIX,
    credit_epoch,
    weights_family_tags,
)

logger = logging.getLogger("weg2.front")

FORWARD_PATHS = ("/generate", "/v1/completions", "/v1/chat/completions")
PASSTHROUGH_GET = ("/v1/models", "/get_model_info", "/get_server_info", "/model_info", "/metrics")
CHUNK_TOKENS = 4096
CHARS_PER_TOKEN = 3.0  # conservative: over-estimates tokens, never under-prices
# #1233 zero-remainder: the CARRIER-EXCEEDS route must not UNDER-estimate --
# measured boot weg2zr1: 80,000 chars of markdown = 30,100 tokens (2.66
# chars/token), priced 26,701 by CHARS_PER_TOKEN and routed BATCH past the
# 27,466-token carrier bound. Route by a lower divisor; the realised leg-1
# count corrects any prompt that still slips through (see leg1).
CARRIER_CHARS_PER_TOKEN = 2.4
T_DRAIN_S = 120.0  # spec 3.5.4: the #111 link-seam bound reused
QUIESCE_DEADLINE_S = 90.0
RPC_TIMEOUT_S = 900.0
SPAN_LRU = 512
SLEEP_TAGS = ["kv_cache", "weights"]
KV_TAG = "kv_cache"

MIB = 1024 * 1024

#: HOW THE FLIP ORDERS ITS TWO LEGS -- the one fact the host-ring launch check
#: needs and cannot infer, declared HERE because this file is what does it.
#:
#: ``"interleave"`` (C9, spec Amendment A1-1): :meth:`Weg2Front.flip` step 3
#: sends ONE ``/release_memory_occupation`` to S and ONE
#: ``/resume_memory_occupation`` to W, both carrying the WHOLE weights family,
#: in a single :func:`asyncio.gather`.  The two legs are therefore concurrently
#: in flight, which is the premise of spec R5's corridor inequality: W's per-tag
#: releases are what fund S's acquires, so the host never has to hold both whole
#: images and ``H(c) = max_g image_g(c)`` suffices.
#:
#: The predecessor value ``"serial"`` -- one RPC per tag, ``S.pause(tag)``
#: completing before ``W.resume(tag)`` was even sent -- required the strictly
#: larger ``H(c) >= image_W(c) + max_tag_S(c)``, and boot weg2rg1 (2026-09-08
#: 01:2xZ) REFUSED the ring by name on 4 of 6 (card x direction) cases under it
#: (W34).  There is no longer a code path in this module that produces it: the
#: gathered pair below is the only leg form, and the launcher's W34 arm survives
#: only as the assertion that a launcher which finds this constant saying
#: anything else refuses instead of arming a ring the flip cannot walk.
#:
#: This constant is READ by ``weg2/launcher.py`` and must never be copied: it is
#: the single bookkeeping of the fact, and its value changing is what a launch
#: check is allowed to key on.
FLIP_LEG_FORM = "interleave"


def completed_tags(body: str) -> Tuple[List[str], Dict[str, List[float]], str]:
    """``(tags this group completed, the per-tag map, the critical-path note)``.

    C10/C17: the group's answer now carries ``per_tag`` -- ``{tag: [bytes, ms]}``
    reduced over the group's ranks by the fence's own ``all_gather_object``, and
    ``critical_path``, the rank and card that took longest.  A W4 that says only
    "HTTP 500" cannot tell the operator whether the family was half-parked; the
    tag list is what makes the "VRAM state undefined" sentence checkable.

    DENOMINATOR, stated because this is a population figure: the list is the
    tags the ANSWERING group reported, which with a gathered leg is every tag of
    the family it finished, and it is EMPTY -- never "none completed" -- when the
    body is not the JSON this tree emits (an upstream error page, a connection
    error string from :meth:`Weg2Front.rpc`, or a build without C17).  The caller
    prints the reason rather than a bare empty list.
    """
    try:
        payload = json.loads(body)
    except Exception:  # noqa: BLE001
        return [], {}, "no per-tag report (the answer is not JSON)"
    if not isinstance(payload, dict):
        return [], {}, "no per-tag report (the answer is not an object)"
    per_tag = payload.get("per_tag")
    crit = payload.get("critical_path") or ""
    if not isinstance(per_tag, dict) or not per_tag:
        return [], {}, crit or "no per-tag report in the answer (C17 field empty)"
    clean = {
        str(k): [float(v[0]), float(v[1])]
        for k, v in per_tag.items()
        if isinstance(v, (list, tuple)) and len(v) >= 2
    }
    return sorted(clean), clean, str(crit)


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


def double_prefill_verdict(prompt_tokens: int, cached_tokens: int, reroutes: int) -> str:
    """spec 3.6: 'serve' | 'reroute' | 'W16'."""
    uncached = max(0, prompt_tokens - cached_tokens)
    if uncached <= CHUNK_TOKENS:
        return "serve"
    if reroutes >= 1:
        return "W16"
    return "reroute"


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


def interleave_pause_order(
    tags: List[str],
    tag_cards: Dict[str, Any],
    free_mib: Dict[int, int],
) -> Tuple[List[str], str]:
    """The order the SOURCE group pauses its weights family in: TIGHTEST CARD
    FIRST.  Returns ``(order, why)``; ``why`` names the reason in the log.

    The interleave (``flip``) pauses one source tag and resumes one destination
    tag per step, so per card the destination's demand is only paid for by the
    source's release IF the two tags name the same card.  They do not when the
    two groups have different parallelism: the source's chunk tag is a LAYER
    band (PP: one card), the destination's is a shard of every layer (TP: all
    cards).  Boot weg2dk4 measured the consequence on the PP source's LAST
    stage -- 4,511 -> 277.8 MiB driver_free in five steps, then CUDA OOM in
    ``cu_mem_create`` on the sixth resume, group D fatal.

    The order below is the only free variable that fixes it without touching a
    budget: the ENDPOINT of the flip is unchanged (the same tags are paused and
    resumed), only the PATH is.  Releasing the tightest card's bands first pays
    that card's demand up front and moves the drawdown onto the cards that have
    the free memory to absorb it -- which is measured here, per flip, not
    assumed.

    Contracts kept: the base weights tag closes the sleep (``weights_family_tags``),
    the result is always a permutation of ``tags``, and an incomplete input is a
    NAMED refusal to reorder (identity), never a partial order.
    """
    tags = list(tags)
    chunks = [t for t in tags if t.startswith(WEIGHT_CHUNK_PREFIX)]
    rest = [t for t in tags if not t.startswith(WEIGHT_CHUNK_PREFIX)]
    if not tag_cards:
        return tags, "identity: the source has no chunk->card map (uniform/TP source, or no map passed)"
    if not free_mib:
        return tags, "identity: no NVML free sample for this flip"
    missing = [t for t in chunks if t not in tag_cards]
    if missing:
        return tags, f"identity REFUSED to reorder: chunk tags absent from the map {missing}"
    unknown = sorted({int(c) for t in chunks for c in tag_cards[t]} - set(free_mib))
    if unknown:
        return tags, f"identity REFUSED to reorder: cards {unknown} absent from the NVML free sample {sorted(free_mib)}"
    index = {t: i for i, t in enumerate(chunks)}
    order = sorted(chunks, key=lambda t: (min(free_mib[int(c)] for c in tag_cards[t]), index[t]))
    return order + rest, "tightest-card-first"


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
                 src_chunk_cards: Optional[Dict[str, Dict[str, List[int]]]] = None):
        self.groups = {"P": Group("P", prefill.rstrip("/"), prefill_sid), "D": Group("D", decode.rstrip("/"), decode_sid)}
        self.awake = awake
        # #1233 one-backup flip: the weights tag family both groups were built
        # with (launcher: SGLANG_WEG2_WEIGHT_CHUNKS), chunks first, base last.
        self.weight_chunks = int(weight_chunks)
        self.weights_tags = weights_family_tags(self.weight_chunks)
        # #1233 boot weg2dk4: per group, which cards (NVML index) hold each
        # chunk tag's bytes -- derived by the launcher from that group's
        # parallelism, EMPTY for a group whose tags are uniform across cards
        # (TP).  Read by interleave_pause_order when that group is the source.
        self.src_chunk_cards: Dict[str, Dict[str, List[int]]] = dict(src_chunk_cards or {})
        self.tag = tag
        self.store_dir = store_dir
        self.dc_reserve = dc_reserve
        self.w_s = w_s
        self.epoch = 0
        # C14 / FIX 2 round 2, finding 1: THE BOOT HALF OF THE CREDIT EPOCH.
        # ``self.epoch`` alone dates a flip only WITHIN this front; the counter
        # file it dates lives in /dev/shm and outlives the boot, so flip 7 of
        # this boot used to read flip 7 of the last boot's terminal state as its
        # own funding.  The boot nonce is the launcher's ring epoch, published
        # to the ranks as TMS_HOST_RING_EPOCH (launcher output, R19 -- not an
        # operator knob); with no armed ring there is none to inherit and this
        # process's own start time serves, boot-unique for the same reason.
        self.boot_epoch = os.environ.get("TMS_HOST_RING_EPOCH", "").strip() or str(int(time.time()))
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
        self._ready_for_d: List[Pending] = []
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

    # ---------------- lifecycle ----------------
    async def startup(self, app):
        self.session = ClientSession(timeout=ClientTimeout(total=3600))
        app["controller"] = asyncio.create_task(self.controller())
        app["health"] = asyncio.create_task(self.health_poller())
        app["corridor"] = asyncio.create_task(self.corridor_sampler())
        logger.info("WEG2-FRONT up tag=%s awake=%s P=%s D=%s W=%.0f s (operator V1 fairness bound) carrier_max_tokens=%d",
                    self.tag, self.awake, self.groups["P"].url, self.groups["D"].url, self.w_s, self.carrier_max_tokens)

    async def cleanup(self, app):
        for k in ("controller", "health", "corridor"):
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
        self._ready_for_d = []

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
                return await self.leg2(request, rid, payload, text, stream, pending=None, single_prefill=True)
            fut = asyncio.get_event_loop().create_future()
            p = Pending(rid, request.path, payload, text, time.time(), fut, est_prompt=est_prompt, span_known=known, skip_leg1=True)
            self.queue.append(p)
            try:
                await fut
            except Weg2Stop as e:
                return web.json_response({"error": str(e)}, status=503)
            except Exception as e:  # noqa: BLE001
                return web.json_response({"error": f"{type(e).__name__}: {e}"}, status=503)
            return await self.leg2(request, rid, payload, text, stream, pending=p)
        if self.awake == "D" and self.admit_d and self.state == "serving" and remainder <= CHUNK_TOKENS:
            self.counters["route_short"] += 1
            logger.info("WEG2-ROUTE rid=%s SHORT -> D est_prompt=%d remainder=%d span_known=%s", rid, est_prompt, remainder, known)
            return await self.leg2(request, rid, payload, text, stream, pending=None)
        self.counters["route_batch"] += 1
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
        return await self.leg2(request, rid, payload, text, stream, pending=p)

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
            if uncached > CHUNK_TOKENS:
                self.counters["short_mispriced"] += 1
                return "short_mispriced"
            return "serve"
        if not priced:
            self.counters["W28_Weg2Leg2Unpriced_stream_served"] += 1
            logger.error("W28 Weg2Leg2Unpriced rid=%s: STREAMED leg 2 ended without a usage/meta_info chunk; served, unpriced, counted", rid)
            return "unpriced"
        v = double_prefill_verdict(pt, ct, pending.reroutes)
        if stream and v != "serve":
            self.counters["W16_Weg2DoublePrefillExceeded"] += 1
            self.counters["W16_stream_served"] += 1
            logger.error("W16 Weg2DoublePrefillExceeded rid=%s (STREAM, served): %d > %d uncached on a streamed leg 2 -- "
                         "reroute impossible after the first byte; counted by name, not refused", rid, uncached, CHUNK_TOKENS)
            return "W16"
        return v

    async def leg2(self, request: web.Request, rid: str, payload: dict, text: str, stream: bool,
                   pending: Optional[Pending], single_prefill: bool = False) -> web.StreamResponse:
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
                    dterms = await self._draft_terms(g, None)
                    logger.info("WEG2-SERVED group=D leg=2 rid=%s stream=1 status=%d prompt_tokens=%d cached_tokens=%d completion_tokens=%d "
                                "uncached=%d verdict=%s priced=%s wall=%.2fs epoch=%d draft_pages=%d draft_miss=%d accept_len=%.3f accept_src=%s",
                                rid, r.status, pt, ct, comp, max(0, pt - ct), verdict, priced, time.time() - t0, self.epoch,
                                dterms["draft_pages"], dterms["draft_miss"], dterms["accept_len"], dterms["accept_src"])
                    if pending is not None and ct > 0:
                        self.counters["cross_group_prefix_hits"] += 1
                    return resp
                body = await r.read()
                try:
                    js = json.loads(body)
                except Exception:  # noqa: BLE001
                    js = {}
                pt, ct, comp, priced = usage_of(js)
                if r.status == 200 and not priced:
                    # 1j finding 3: never price a body without usage/meta_info as (0,0).
                    self.counters["W28_Weg2Leg2Unpriced"] += 1
                    logger.error("W28 Weg2Leg2Unpriced rid=%s: D answered 200 without usage/meta_info (%d bytes); refusing by name "
                                 "rather than serving on a fail-open (0,0) price", rid, len(body))
                    return web.json_response({"error": f"W28 Weg2Leg2Unpriced rid={rid}"}, status=503)
                verdict = self._leg2_verdict(pt, ct, priced, pending, single_prefill, False, rid)
                g.served += 1
                dterms = await self._draft_terms(g, js)
                logger.info("WEG2-SERVED group=D leg=2 rid=%s status=%d prompt_tokens=%d cached_tokens=%d completion_tokens=%d "
                            "uncached=%d verdict=%s wall=%.2fs epoch=%d draft_pages=%d draft_miss=%d accept_len=%.3f accept_src=%s",
                            rid, r.status, pt, ct, comp, max(0, pt - ct), verdict, time.time() - t0, self.epoch,
                            dterms["draft_pages"], dterms["draft_miss"], dterms["accept_len"], dterms["accept_src"])
                if pt:
                    self.spans.record(text, pt)
                    self._note_exact(text, pt)
                if r.status == 200 and pending is not None and verdict == "reroute":
                    self.counters["reroute"] += 1
                    pending.reroutes += 1
                    pending.fut = asyncio.get_event_loop().create_future()
                    pending.t_arrive = time.time()
                    self.queue.append(pending)
                    logger.warning("WEG2-REROUTE rid=%s uncached=%d > %d: rejoining route BATCH once (spec 3.6)", rid, pt - ct, CHUNK_TOKENS)
                    g.outstanding.pop(rid, None)
                    try:
                        await pending.fut
                    except Weg2Stop as e:
                        return web.json_response({"error": str(e)}, status=503)
                    return await self.leg2(request, rid, payload, text, stream, pending)
                if r.status == 200 and pending is not None and verdict == "W16":
                    self.counters["W16_Weg2DoublePrefillExceeded"] += 1
                    logger.error("W16 Weg2DoublePrefillExceeded rid=%s: re-routed once and the prefix is still %d > %d uncached; refusing and reporting",
                                 rid, pt - ct, CHUNK_TOKENS)
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

    async def _draft_terms(self, g, body) -> dict:
        """#1233 (C16, L12): the draft terms of one served request.

        ``accept_len`` comes from ``meta_info.spec_accept_length`` when the
        body carries it (``/generate``), else from the ``/get_server_info``
        ``avg_spec_accept_length`` (cumulative average, named as such);
        ``draft_pages``/``draft_miss`` are the DELTA of D's L3 draft read
        counters (``internal_states[0]`` of the body) between this call and the previous one (one request in
        flight at a time on the leg-2 route, so the delta is this request's).
        Never raises: a missing instrument is ``accept_src=none``.
        """
        out = {"draft_pages": 0, "draft_miss": 0, "accept_len": 0.0, "accept_src": "none"}
        try:
            mi = (body or {}).get("meta_info") if isinstance(body, dict) else None
            if isinstance(mi, dict) and mi.get("spec_accept_length") is not None:
                out["accept_len"] = float(mi["spec_accept_length"])
                out["accept_src"] = "meta"
            async with self.session.get(f"{g.url}/get_server_info") as r:
                info = await r.json() if r.status == 200 else {}
            if isinstance(info, list) and info:
                info = info[0]
            if not isinstance(info, dict):
                return out
            # /server_info (http_server.py) puts the server_args at the top
            # level and the SCHEDULER's counters -- these -- one level down
            # under `internal_states[0]` (fix 2: read at the top level, the
            # terms were always 0 and accept_len never reached its fallback).
            st = info.get("internal_states") or []
            if isinstance(st, list) and st and isinstance(st[0], dict):
                info = st[0]
            hits = int(info.get("draft_l3_hits", 0) or 0)
            miss = int(info.get("draft_l3_misses", 0) or 0)
            prev = getattr(self, "_draft_prev", (0, 0))
            self._draft_prev = (hits, miss)
            out["draft_pages"] = max(0, hits - prev[0])
            out["draft_miss"] = max(0, miss - prev[1])
            if out["accept_src"] == "none" and info.get("avg_spec_accept_length") is not None:
                out["accept_len"] = float(info["avg_spec_accept_length"])
                out["accept_src"] = "server_info_avg"
        except Exception as e:  # noqa: BLE001 - an instrument never breaks serving
            logger.debug("draft terms unavailable: %s: %s", type(e).__name__, e)
        return out

    def check_identity(self) -> None:
        """W9 from the store directory: exactly ONE identity suffix on disk.

        #1233 draft KV across the flip (C16): walks the ``page_shard``
        subdirectories (``hicache_storage.page_shard``: the first two hex
        characters of the key) instead of the flat listdir that saw no page
        at all, and censuses kv / mamba / draft files by name (L11).
        """
        self.identity_checked = True
        names = []
        try:
            for root, _dirs, files in os.walk(self.store_dir):
                for n in files:
                    if n.endswith(".bin"):
                        names.append(n)
                        if len(names) >= 200000:
                            break
                if len(names) >= 200000:
                    break
        except OSError:
            return
        ids, suffixes = set(), set()
        kv = mamba = draft = 0
        for n in names:
            stem = n[:-4]
            head, _, tail = stem.partition("_")
            if ".draft" in head:
                draft += 1
            elif ".mamba" in head:
                mamba += 1
            elif "." not in head:
                kv += 1
            parts = stem.split("_")
            if len(parts) >= 3:
                ids.add(parts[-1])
            suffixes.add("_" + tail if tail else "")
        logger.info("W9 store census (shard-walked) files=%d kv=%d mamba=%d draft=%d suffixes=%s identity suffixes %s (store dir %s)",
                    len(names), kv, mamba, draft, sorted(suffixes)[:8], sorted(ids), self.store_dir)
        if len(ids) > 1:
            self.do_stop("W9 Weg2StoreIdentityMismatch", f"identity suffixes on the sole carrier: {sorted(ids)}")

    # ---------------- flip machinery ----------------
    async def rpc(self, g: Group, path: str, body: Optional[dict], timeout: float) -> Tuple[int, str]:
        try:
            async with self.session.post(f"{g.url}{path}", json=body or {}, timeout=ClientTimeout(total=timeout)) as r:
                return r.status, (await r.read()).decode(errors="replace")
        except Exception as e:  # noqa: BLE001
            return 0, f"{type(e).__name__}: {e}"

    async def timed_rpc(self, g: Group, path: str, body: Optional[dict],
                        timeout: float) -> Tuple[int, str, float]:
        """:meth:`rpc` plus the wall time of THIS leg alone.

        C9 gathers the two legs, so the flip's own ``interleave`` wall clock is
        no longer the sum of the parts and neither leg's cost can be read off
        it.  Each leg times itself; the sum and the wall are then two different
        measured quantities and L5 prints both plus their difference (spec C11).
        """
        t0 = time.perf_counter()
        code, text = await self.rpc(g, path, body, timeout)
        return code, text, (time.perf_counter() - t0) * 1000

    async def drain(self, g: Group) -> bool:
        t0 = time.time()
        while g.outstanding:
            if time.time() - t0 > T_DRAIN_S:
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
            logger.error("W1 Weg2DrainRefused: %s still holds %d request(s) after %.0f s (rids %s); not flipping (%d in a row)",
                         src, len(S.outstanding), T_DRAIN_S, sorted(S.outstanding)[:8], self.drain_refusals_in_a_row)
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
        # 3. THE GATHERED LEGS (C9, spec Amendment A1-1).  src.pause(kv_cache)
        # first, then ONE /release_memory_occupation to S and ONE
        # /resume_memory_occupation to W, both carrying the WHOLE weights
        # family, issued together and awaited together; dst.resume(kv_cache)
        # last.
        #
        # WHAT ORDERS THE LEGS, now that this driver no longer does.  The old
        # per-tag loop ordered them here, at the cost of the strictly larger
        # host requirement H(c) >= image_W(c) + max_tag_S(c) -- which boot
        # weg2rg1 refused by name on 4 of 6 (card x direction) cases (W34).
        # With both legs in flight the ordering is enforced where the bytes
        # actually are, by two mechanisms that are checked at launch, not
        # assumed here:
        #
        #   * THE RING BITMAP (spec R11, tms_csrc/host_ring.cpp).  A granule is
        #     TAKEN from W's pause until ring_->release() runs, strictly after
        #     the leg's cudaStreamSynchronize; acquire() only ever returns bits
        #     that are 0.  So S can never take host bytes whose H2D is still in
        #     flight, and S's blocking acquire is funded by W's releases as they
        #     land -- the corridor walk of R5, whose inequality the launcher
        #     checks per card per direction BEFORE either group starts (W32).
        #   * THE VRAM CREDIT (C14, weg2_memory_saver.vram_credit).  The mirror
        #     of the same argument on the DEVICE axis: W may need device bytes
        #     that only S's release frees.  S publishes the bytes it released
        #     per tag and a leg_complete flag; W waits on the counter and
        #     REFUSES BY NAME (W35/W30) when S's leg finishes without ever
        #     funding it, instead of turning into a CUDA OOM or a hang.
        #
        # And the per-card PCIe lock, which used to serialise the two legs into
        # each other whatever this driver did, now keys on <uuid>.<d2h|h2d>
        # (C12/C13) on EVERY card while the legs are gathered.  FIX 2 round 2,
        # finding 4: what stood here instead was that the x4-linked card, having
        # measured 1.316, kept one key and so still serialised the pair by
        # design.  RETRACTED -- it is false on this tree and was already false
        # when it was written.  The ratio is an expectation about
        # THROUGHPUT; the split is a CORRECTNESS decision, taken once in
        # launcher._split_decisions, which returns split for every card under
        # leg_form == "interleave" because a one-key card deadlocks the gathered
        # pair (boot weg2rg2: W31 -> W29 -> group-fatal W4).  What the launcher
        # prints per card is that decision (WEG2-HOST-RING CHECK ... key=), and
        # a card printed SINGLE there does not arm -- it refuses W34.
        sleep_ms = 0.0
        wake_ms = 0.0
        chunk_recs: List[dict] = []
        t0 = time.time()
        code, body = await self.rpc(S, "/release_memory_occupation", {"tags": [KV_TAG]}, RPC_TIMEOUT_S)
        sleep_ms += (time.time() - t0) * 1000
        if code != 200:
            self.do_stop("W4 Weg2WakeRefused", f"sleep({src}, {KV_TAG}) failed HTTP {code}: {body[:400]!r} -- VRAM state undefined, no retry")
            return
        family = list(self.weights_tags)
        # #1233 fix 4 ON THE RING FORM.  The tight-card-first order survives the
        # move to gathered legs (C9); the serial per-tag RPC loop it used to
        # drive does NOT.  Under the gathered legs the front issues ONE
        # /release_memory_occupation per group, so the pause ORDER is no longer
        # an RPC sequence the front controls step by step -- it is the ORDER OF
        # THE TAG LIST that leg carries, which the group walks unchanged
        # (`weights_tags = [t for t in tags if is_weights_family_tag(t)]` in
        # weight_updater, both legs).  So the ordering is applied HERE, to the
        # source leg's list, and the destination keeps the natural order.  That
        # is ONE flip mechanism and ONE ordering function, not two.
        #
        # Why the order still matters with the legs gathered: the two legs now
        # run CONCURRENTLY, so the destination's demand on a card overlaps the
        # source's release on that same card.  Pausing the tightest card's
        # bands first is what makes the source's bytes come free on the card
        # the destination is about to want them on.  boot weg2dk4 died in
        # cu_mem_create on exactly that card (driver_free 4,511 -> 277.8 MiB
        # in five steps, BOOT_weg2dk4_0907.md).
        #
        # The free sample is taken HERE, after the source's kv_cache is already
        # released, so it is the state the flip actually starts from.
        free_mib = {idx: free for idx, _uuid, free in _nvml_free()}
        pause_order, why = interleave_pause_order(
            self.weights_tags, self.src_chunk_cards.get(src, {}), free_mib
        )
        logger.info(
            "WEG2-FLIP-ORDER epoch=%d src=%s driver_free=%s pause_order=%s resume_order=%s (%s) "
            "-- applied to the GATHERED sleep leg's tag list (C9), not to a per-tag RPC loop",
            self.epoch, src, free_mib, pause_order, self.weights_tags, why,
        )
        if sorted(pause_order) != sorted(self.weights_tags):
            self.do_stop(
                "W4 Weg2WakeRefused",
                f"pause order {pause_order} is not a permutation of the weights family "
                f"{self.weights_tags} -- a tag would be resumed on {dst} that was never "
                f"paused on {src}; VRAM state untouched, no flip",
            )
            return
        # FIX 2 round 2: the token names the BOOT and the flip, not the flip
        # alone -- see weg2_memory_saver.credit_epoch for the leftover counters
        # a bare flip index made this boot inherit.
        flip_epoch = credit_epoch(self.boot_epoch, self.epoch)
        t_gather0 = time.perf_counter()
        # C14 / FIX 1 round 1: the FLIP'S epoch rides on BOTH legs.  It is the
        # only thing that dates the per-card VRAM credit counter, and this
        # gather is precisely why one is needed -- there is no happens-before
        # between S's begin_leg and W's first read, so without it W reads the
        # previous flip's terminal state as this flip's funding.
        (s_code, s_body, s_ms), (w_code, w_body, w_ms) = await asyncio.gather(
            self.timed_rpc(S, "/release_memory_occupation",
                           {"tags": pause_order, "epoch": flip_epoch}, RPC_TIMEOUT_S),
            self.timed_rpc(D, "/resume_memory_occupation",
                           {"tags": family, "epoch": flip_epoch}, RPC_TIMEOUT_S),
        )
        legs_wall_ms = (time.perf_counter() - t_gather0) * 1000
        s_done, s_per_tag, s_crit = completed_tags(s_body)
        w_done, w_per_tag, w_crit = completed_tags(w_body)
        sleep_ms += s_ms
        wake_ms += w_ms
        if s_code != 200 or w_code != 200:
            # C10.  Both legs were in flight, so BOTH groups' completion state
            # is part of the fault and both are named -- "VRAM state undefined"
            # is now a statement about a whole family on each side, and the tag
            # lists are what make it checkable rather than a slogan.
            which = []
            if s_code != 200:
                which.append(f"sleep({src}, family) HTTP {s_code}: {s_body[:400]!r}")
            if w_code != 200:
                which.append(f"wake({dst}, family) HTTP {w_code}: {w_body[:400]!r}")
            self.do_stop(
                "W4 Weg2WakeRefused",
                "; ".join(which)
                + f" -- gathered legs (C9), family={family}; "
                + f"{src} completed {s_done or '[]'} ({s_crit}); "
                + f"{dst} completed {w_done or '[]'} ({w_crit}); "
                + "VRAM state undefined on BOTH sides, no retry, "
                + "recovery = teardown + relaunch",
            )
            return
        if not s_per_tag and not w_per_tag:
            # DENOMINATOR: no group reported per-tag, so there are no chunk
            # records to print.  Printing zeros here would read as "this tag
            # moved nothing", which is the opposite of "nobody measured".
            logger.info(
                "WEG2-FLIP-CHUNK epoch=%d SUPPRESSED for all %d tags: neither group "
                "returned a per-tag report (%s / %s) -- absence, not zero",
                self.epoch, len(family), s_crit, w_crit,
            )
        for tag in family if (s_per_tag or w_per_tag) else ():
            rec = {
                "tag": tag,
                "sleep_ms": round(s_per_tag.get(tag, [0.0, 0.0])[1]),
                "wake_ms": round(w_per_tag.get(tag, [0.0, 0.0])[1]),
                "sleep_mib": round(s_per_tag.get(tag, [0.0, 0.0])[0] / MIB),
                "wake_mib": round(w_per_tag.get(tag, [0.0, 0.0])[0] / MIB),
            }
            chunk_recs.append(rec)
            logger.info(
                "WEG2-FLIP-CHUNK epoch=%d tag=%s %s.pause=%d ms %d MiB %s.resume=%d ms %d MiB "
                "(source: the group's own per-tag report, NOT a front-side RPC boundary -- "
                "the legs are gathered, so the front no longer sees a tag edge)",
                self.epoch, tag, src, rec["sleep_ms"], rec["sleep_mib"], dst, rec["wake_ms"], rec["wake_mib"],
            )
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
        # C11 / L5.  interleave_ms IS NOT sleep+wake any more, and saying so is
        # the point of the line: with the legs gathered the two are different
        # measured quantities and their difference is the achieved overlap.
        #   sleep_ms / wake_ms  -- each leg's OWN wall time, timed_rpc
        #   legs_wall_ms        -- the wall time of the gather, both in flight
        #   overlap_ms          -- (sleep leg + wake leg) - gather wall, i.e.
        #                          the part of the shorter leg the longer one
        #                          hid.  Its denominator is the SHORTER leg:
        #                          100 % means the shorter leg cost nothing in
        #                          wall time, and it can never exceed that.
        # ZERO OVERLAP ON THIS TREE IS A FINDING, NOT A READING (FIX 2 round 2,
        # finding 4).  The comment that stood here pre-authorised it: it told the
        # reader that on a card whose measured duplex ratio did not earn the
        # direction split (the x4-linked 3080, 1.316) the per-card PCIe lock
        # ordered the two legs one after the other on purpose.  RETRACTED -- that
        # premise is dead: under gathered legs the launcher
        # splits the key on EVERY card whatever its ratio.  So if section 9.6's
        # L5 acceptance reading comes back at ~0 %, do not dismiss it; check, in
        # this order, (1) the launcher's WEG2-HOST-RING CHECK line per card for
        # key=SPLIT per direction, (2) the rank's RESOLVED lock path (a
        # <uuid>.d2h / <uuid>.h2d pair in /dev/shm, not a bare <uuid>), (3)
        # W36 Weg2DuplexDecisionRefused in the rank logs.  A ratio below R17's
        # gate predicts a small SPEEDUP from the split on that card (A1-4), not
        # a serialisation.
        legs_ms = [("sleep/" + src, s_ms, s_crit), ("wake/" + dst, w_ms, w_crit)]
        crit_leg, crit_ms, crit_note = max(legs_ms, key=lambda x: x[1])
        overlap_ms = max(0.0, s_ms + w_ms - legs_wall_ms)
        overlap_pct = 100.0 * overlap_ms / max(1.0, min(s_ms, w_ms))
        rec = {"epoch": self.epoch, "sleep": src, "wake": dst, "drain_quiesce_ms": round((t_q - t_flip0) * 1000),
               "sleep_ms": round(sleep_ms), "wake_ms": round(wake_ms), "flip_ms": round((t_w - t_flip0) * 1000),
               "interleave_ms": round((t_s - t_q) * 1000), "chunks": chunk_recs,
               "legs_wall_ms": round(legs_wall_ms), "sleep_leg_ms": round(s_ms), "wake_leg_ms": round(w_ms),
               "overlap_ms": round(overlap_ms), "overlap_pct": round(overlap_pct, 1),
               "critical_path": f"{crit_leg} {crit_note}",
               "dc_mib": dc, "t": time.time()}
        self.flip_log.append(rec)
        self.counters["flips"] += 1
        logger.info("WEG2-FLIP done epoch=%d slept=%s woke=%s drain+quiesce=%d ms sleep=%d ms (kv RPC + the %s leg of the gathered pair) "
                    "wake=%d ms (the %s leg + kv RPC) "
                    "interleave=%d ms (NOT sleep+wake: the legs overlap -- gather wall %d ms against %d + %d ms of legs) "
                    "overlap=%d ms (%.0f%% of the shorter leg) critical_path=%s "
                    "flip_total=%d ms weights_tags=%d dc=%s",
                    rec["epoch"], src, dst, rec["drain_quiesce_ms"], rec["sleep_ms"], src,
                    rec["wake_ms"], dst, rec["interleave_ms"], rec["legs_wall_ms"],
                    rec["sleep_leg_ms"], rec["wake_leg_ms"], rec["overlap_ms"], rec["overlap_pct"],
                    rec["critical_path"], rec["flip_ms"], len(self.weights_tags), dc)

    async def controller(self) -> None:
        sem = asyncio.Semaphore(8)
        while True:
            await asyncio.sleep(0.2)
            try:
                if self.state != "serving":
                    continue
                if self.awake == "D":
                    if not self.queue:
                        continue
                    oldest = self.queue[0].t_arrive
                    D = self.groups["D"]
                    if fairness_reached(oldest, time.time(), self.w_s) and self.admit_d:
                        self.admit_d = False
                        self.counters["fairness_bound_hits"] += 1
                        logger.warning("WEG2-FAIRNESS W=%.0f s reached (operator V1 fairness bound): oldest waiting %.1f s; "
                                       "stop admitting NEW work to D, drain running decodes, then flip", self.w_s, time.time() - oldest)
                    if not D.outstanding or not self.admit_d:
                        await self.flip("D", "P")
                    continue
                # awake == P: prefill the backlog until empty (#1011 PP exit clock)
                while self.queue and self.state == "serving":
                    batch = [self.queue.popleft() for _ in range(min(8, len(self.queue)))]

                    async def one(p: Pending):
                        if p.skip_leg1:  # route CARRIER-EXCEEDS: no leg 1, D prefills once
                            p.leg1_done = True  # type: ignore[attr-defined]
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
                            p.leg1_done = True  # type: ignore[attr-defined]
                    await asyncio.gather(*(one(p) for p in batch))
                    for p in batch:
                        if not p.fut.done():
                            self._ready_for_d.append(p)
                await self.flip("P", "D")
                # 1j finding 1: release leg 2 only when D is the awake group -- a
                # W1-refused flip leaves state=='serving' with P still awake.
                if self.state == "serving" and self.awake == "D":
                    ready, self._ready_for_d = self._ready_for_d, []
                    for p in ready:
                        if not p.fut.done():
                            p.fut.set_result(True)
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
    ap.add_argument("--fairness-w-s", type=float, default=45.0)
    ap.add_argument("--weight-chunks", type=int, default=0, help="#1233: number of weights_<k> chunk tags both groups were built with (0 = single weights tag)")
    ap.add_argument("--carrier-max-tokens", type=int, default=0, help="#1233 zero-remainder: longest prompt group D can read from the store (0 = no CARRIER-EXCEEDS route)")
    ap.add_argument("--src-chunk-cards", default="",
                    help="#1233 weg2dk4: JSON {group: {weights_k: [nvml_index, ...]}} -- which cards hold each chunk tag's "
                         "bytes, per group, derived by the launcher from that group's parallelism. A group that is absent "
                         "(or an empty map) pauses in the natural tag order, exactly as before.")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(levelname)s %(name)s: %(message)s")
    dc = {}
    for kv in filter(None, args.dc_reserve.split(",")):
        k, v = kv.split("=")
        dc[k] = int(v)
    front = Front(args.prefill, args.decode, args.awake, args.tag, args.store_dir, args.prefill_sid, args.decode_sid, dc, args.fairness_w_s,
                  weight_chunks=args.weight_chunks, carrier_max_tokens=args.carrier_max_tokens,
                  src_chunk_cards=json.loads(args.src_chunk_cards) if args.src_chunk_cards else {})
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
    logger.info("WEG2-FRONT %s:%s -> P=%s D=%s awake=%s weights_tags=%s src_chunk_cards=%s", args.host, args.port,
                args.prefill, args.decode, args.awake, front.weights_tags, front.src_chunk_cards)
    web.run_app(app, host=args.host, port=args.port, print=None)


if __name__ == "__main__":
    main()
