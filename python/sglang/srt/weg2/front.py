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
import contextvars
import hashlib
import json
import logging
import os
import re
import subprocess
import time
from dataclasses import dataclass, field
from typing import Any, Deque, Dict, List, Optional, Set, Tuple

from aiohttp import (
    ClientConnectionError,
    ClientSession,
    ClientTimeout,
    ServerDisconnectedError,
    TCPConnector,
    TraceConfig,
    web,
)

from sglang.srt.managers import corridor_guard
from sglang.srt.managers.corridor_guard import (
    corridor_band_ceiling_mib,
    corridor_band_floor_mib,
    corridor_floor_mib,
)
from sglang.srt.managers.corridor_guard import USER_RESERVE_ENV
from sglang.srt.managers.weg2_memory_saver import (
    credit_epoch,
    is_weights_chunk_tag,
    weights_family_tags,
)
from sglang.srt.registry import nvml as nvml_registry
from sglang.srt.weg2 import DEFAULT_D_BS, DEFAULT_P_BS
from sglang.srt.weg2 import admin_key as admin_key_mod
from sglang.srt.weg2 import host_ledger

logger = logging.getLogger("weg2.front")

#: Q0-B: ``/v1/messages`` IS FORWARDED LIKE ``/v1/chat/completions``.  Both
#: groups serve the Anthropic Messages API natively (measured 2026-09-09 on
#: boot weg2sn5: ``POST :30032/v1/messages`` -> 200 while the front answered
#: 404), and the split router at :30099 passes the path through unchanged, so
#: a front without this entry makes every Messages-API client -- i.e. the
#: whole Claude-Code-shaped agent fleet -- unreachable while the groups behind
#: it are healthy.  The three places that had to learn the second wire shape
#: are named at their own sites: ``request_text`` (pricing), ``usage_of`` /
#: ``AnthropicStreamUsage`` (the leg-2 price), and the ``stream_options``
#: injection in ``leg2``.  The ROUTING LOGIC is untouched by this addition.
FORWARD_PATHS = ("/generate", "/v1/completions", "/v1/chat/completions", "/v1/messages")
#: ``count_tokens`` is a PRICING call, not a generation: it decodes no token,
#: needs no seat, must not open a Pending and must never provoke a flip.  It
#: is forwarded verbatim to whichever group is awake, exactly like the GET
#: passthroughs beside it -- routing it through ``handle_generate`` would
#: allocate a seat for a request that never generates.
PASSTHROUGH_POST = ("/v1/messages/count_tokens",)
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
#: FIX 4 (round 4): how fresh D's host-pool reading must be before the front
#: decides an admission on it.  Provenance, boot weg2sc1: the front granted 16
#: admissions across a 42.1 s drain with six seats per epoch, i.e. GRANTS ARE
#: SECONDS APART, while a `/server_info` round trip on loopback is
#: milliseconds.  1.0 s sits two orders above the cost of the read and an
#: order below the interval between the decisions it informs.  The read is
#: ON DEMAND (the admitter refreshes a stale reading before it decides), never
#: a background poller -- there is no sampler task to leave running.
D_POOL_MAX_AGE_S = 1.0
#: FIX 5 (round 5): the request timeout of that read, DERIVED from its own
#: freshness bound rather than picked.  `t` is stamped BEFORE the request goes
#: out, so a reply that takes longer than `D_POOL_MAX_AGE_S` describes a pool
#: state already older than the age bound above -- it would be discarded by
#: the very next freshness check, and waiting for it only adds its own latency
#: to an admission decision.  A read that cannot answer inside its own
#: usefulness window is therefore a FAILED read by construction, and the
#: honest response is the named `WEG2 D-POOL UNREADABLE` refusal (gate off),
#: not a longer wait.  This replaces the bare `ClientTimeout(total=5)` of
#: round 4, which was a hand number and could add 5 s to a D admission
#: decision -- on the critical path, and even with the gate flag off.
D_POOL_READ_TIMEOUT_S = D_POOL_MAX_AGE_S
RPC_TIMEOUT_S = 900.0

# ---------------------------------------------------------------------------
# #1285: WHICH SOCKET DID THIS RPC GO OUT ON, AND WAS IT A FRESH ONE
#
# Boot weg2sb5e died with BOTH gathered flip legs returning
# `ServerDisconnectedError` after ~117 s, while P answered `/health` on new
# connections in the same seconds and logged NO access line for the resume.
# RPC_TIMEOUT_S is 900 s, so it was not a timeout; the front's ONE shared
# `ClientSession` has no `TCPConnector` of its own, i.e. aiohttp's defaults --
# keepalive on, `force_close=False`, `limit_per_host=0`.  The leading
# hypothesis is a STALE POOLED KEEPALIVE connection reused after the peer had
# closed it, but that shape normally raises at once rather than after 117 s,
# so it stays a HYPOTHESIS.  Nothing in the old logs can decide it, because
# nothing recorded which socket an RPC used.  These three lines record it.
#
# INSTRUMENT LIMITS, stated because a field that silently degrades is worse
# than an absent one (aiohttp 3.14.1, verified against `__slots__`):
#   * `conn` comes from the TraceConfig hooks `on_connection_reuseconn` /
#     `on_connection_create_end`, which fire reliably but carry NO payload in
#     this aiohttp -- both param classes have only `__weakref__`.  So they can
#     say WHICH of the two happened and nothing more.
#   * `sport` therefore comes from the RESPONSE's own connection transport
#     (`sockname`), which exists only once a response was received.  On the
#     RAISED path there is no response and `sport=n/a` -- absent, never 0.
#   * `pool_idle` / `pool_total` read the connector's private `_conns` /
#     `_acquired`.  Both are best-effort: an aiohttp that renames them yields
#     `-1`, which is "unreadable", not "empty".
#: The two flip legs, by path.  Everything else is `other` -- the leg name is
#: what makes the log greppable per direction, and only these two are the
#: gathered pair of `interleave`.
RPC_LEG_BY_PATH = {
    "/release_memory_occupation": "sleep",
    "/resume_memory_occupation": "wake",
}
#: Connection-level failures that are, by construction, "the request never
#: reached a handler": both are raised by aiohttp BEFORE any response line.
#: `ServerDisconnectedError` is a subclass of `ClientConnectionError`; both are
#: named so the tuple reads as the intent rather than as a class hierarchy.
RPC_CONN_ERRORS = (ServerDisconnectedError, ClientConnectionError)
#: Whether the LAST attempt in THIS task failed in the never-reached-a-handler
#: shape.  A ContextVar and not an attribute on the front: the two flip legs run
#: as concurrent tasks under one `asyncio.gather`, so an instance attribute
#: would be read by whichever leg finished last.  Each task gets its own copy of
#: the context, and `leg_rpc` awaits `rpc` in its OWN task, so it reads its own
#: leg's flag and no other's.  It exists so that `rpc` stays the single seam
#: every caller and every test already stubs, instead of `leg_rpc` reaching past
#: it into `_rpc_attempt`.
RPC_LAST_RETRYABLE: contextvars.ContextVar[bool] = contextvars.ContextVar(
    "weg2_rpc_last_retryable", default=False)


def rpc_leg_name(path: str) -> str:
    return RPC_LEG_BY_PATH.get(path, "other")


def rpc_pool_counts(session: Optional[ClientSession]) -> Tuple[int, int]:
    """(idle, total) pooled connections, or (-1, -1) when unreadable."""
    try:
        conn = session.connector  # type: ignore[union-attr]
        idle = sum(len(v) for v in conn._conns.values())  # noqa: SLF001
        acquired = conn._acquired  # noqa: SLF001
        return idle, idle + len(acquired)
    except Exception:  # noqa: BLE001 - an instrument never breaks serving
        return -1, -1


class SportTCPConnector(TCPConnector):
    """aiohttp's default connector, plus the local port of each connection.

    THE PORT IS THE JOIN KEY.  It is the only field that lets a front log line
    be matched against the peer's uvicorn access log -- which is exactly the
    read weg2sb5e needed and could not make (P logged no access line for the
    resume; without a port there was no way to say whether the request had left
    on a socket P had ever seen).

    It has to be captured HERE and not off the response: measured on aiohttp
    3.14.1, by the time an `async with session.post(...)` body is available
    `resp.connection` is already None and `resp._protocol.transport` is None
    too -- a small Content-Length body completes during the header read, and
    `_response_eof` releases the connection immediately.  The protocol object,
    however, survives and is reused with the pooled connection, so the port
    stamped at creation is still correct on a REUSED connection (verified: two
    requests, one connection, the same port on both).

    No behaviour is changed: every constructor argument is aiohttp's own.
    """

    async def _create_connection(self, req, traces, timeout):
        proto = await super()._create_connection(req, traces, timeout)
        try:
            proto._weg2_sport = str(  # noqa: SLF001
                proto.transport.get_extra_info("sockname")[1])
        except Exception:  # noqa: BLE001 - an instrument never breaks serving
            proto._weg2_sport = "n/a"  # noqa: SLF001
        return proto


def rpc_response_sport(resp: Any) -> str:
    """The LOCAL port this response came back on, or `n/a`.

    `n/a` means unreadable -- an aiohttp whose internals moved, or a session
    not built on :class:`SportTCPConnector`.  Never a port number of 0.
    """
    try:
        return str(resp._protocol._weg2_sport)  # noqa: SLF001
    except Exception:  # noqa: BLE001 - an instrument never breaks serving
        pass
    try:
        return str(resp.connection.transport.get_extra_info("sockname")[1])
    except Exception:  # noqa: BLE001
        return "n/a"


def make_rpc_trace_config() -> TraceConfig:
    """Records reused-vs-new into the per-request ctx dict passed by `rpc`."""

    tc = TraceConfig()

    async def on_reuse(_session, ctx, _params):
        info = getattr(ctx, "trace_request_ctx", None)
        if isinstance(info, dict):
            info["conn"] = "reused"

    async def on_create_end(_session, ctx, _params):
        info = getattr(ctx, "trace_request_ctx", None)
        if isinstance(info, dict):
            info["conn"] = "new"

    tc.on_connection_reuseconn.append(on_reuse)
    tc.on_connection_create_end.append(on_create_end)
    return tc
#: #1262 TIER 3.  How many times its OWN measured cost a flip may take before
#: the front says it is not progressing.  Dimensionless on purpose: the bound
#: itself is this boot's last measured flip in the same direction, so it is
#: correct on a 3 s flip and on a 17 s one without a second number, and there
#: is no literal seconds figure anywhere in the rule.  4x is chosen against the
#: measured spread of this campaign -- weg2rg3 3.1-4.7 s, weg2dk5 13.4/18.4 s,
#: i.e. under 1.5x between the fast and slow arms of the SAME form -- so 4x is
#: comfortably outside the population of flips that merely ran slowly, and the
#: specimen it must catch overran by more than 100x (weg2t2a: 7 min against a
#: 3-5 s flip).
FLIP_STALL_SLACK = 4.0
#: How often the stall watcher LOOKS.  A poll period, never the bound: the
#: bound is derived per flip by `Front._flip_stall_bound_s`.  Matches the
#: corridor sampler's existing cadence so the front gains no new timing
#: behaviour, only a new question asked on the same beat.
FLIP_STALL_POLL_S = 10.0
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


def _block_text(x: Any) -> str:
    """One content BLOCK as text, on both wire shapes.

    Q0-B: reading only ``text`` -- what this did while ``/v1/messages`` was
    404 and only OpenAI bodies arrived -- prices a Claude-Code turn at the
    prose half of its own conversation and silently drops the tool half.  An
    Anthropic ``tool_use`` block carries its arguments in ``input`` (a JSON
    object) and a ``tool_result`` carries the whole tool output in
    ``content``; both are PROMPT BYTES the model pays for on the next turn.
    Under-pricing here is not cosmetic: ``price_remainder`` feeds the SHORT
    bound, so an under-priced long prompt routes SHORT to D, exceeds D's own
    prefill cap and comes back as W50 -- the wall this front already counts
    by name.  Over-pricing is the safe direction and the module says so at
    CHARS_PER_TOKEN.
    """
    if not isinstance(x, dict):
        return str(x)
    t = x.get("type")
    if t == "tool_use":
        # name + the arguments object; `input` is a dict, not a string.
        return f"{x.get('name', '')} {json.dumps(x.get('input') or {}, ensure_ascii=False, sort_keys=True)}"
    if t == "tool_result":
        c = x.get("content")
        if isinstance(c, list):
            return " ".join(_block_text(b) for b in c)
        return str(c if c is not None else "")
    if t == "thinking":
        return str(x.get("thinking", "") or "")
    if "text" in x:
        return str(x.get("text", "") or "")
    # An unknown block (image, document, ...) has no priceable char count.
    # Return its type rather than its base64 payload: a data URI would price
    # as tens of thousands of phantom prompt chars.
    return str(t or "")


def _content_text(c: Any) -> str:
    """A message's ``content`` (str, or a list of blocks) as one string."""
    if isinstance(c, list):
        return " ".join(_block_text(x) for x in c)
    return str(c if c is not None else "")


def request_text(payload: dict) -> str:
    """The prompt as ONE string, for the span estimate and the ledger.

    Q0-B: covers BOTH forwarded chat shapes. OpenAI carries the system turn
    as ``messages[0]`` with ``role="system"``; Anthropic carries it in a
    top-level ``system`` field. Emitting it as a leading ``system:<text>``
    line makes the two shapes price IDENTICALLY for the same conversation,
    which is what the equivalence test in
    ``test/registered/unit/weg2/test_front_messages_pricing.py`` pins.
    ``tools`` is priced for both shapes -- it was priced for NEITHER before,
    and a Claude-Code agent carries 10-20k tokens of tool schemas.
    """
    if "messages" in payload and isinstance(payload["messages"], list):
        parts = []
        sys_field = payload.get("system")
        if sys_field:
            parts.append(f"system:{_content_text(sys_field)}\n")
        for m in payload["messages"]:
            if not isinstance(m, dict):
                parts.append(f"{m}\n")
                continue
            parts.append(f"{m.get('role', '')}:{_content_text(m.get('content', ''))}\n")
        tools = payload.get("tools")
        if isinstance(tools, list) and tools:
            # The whole schema list, serialized once. Deterministic key order
            # so the span LRU's prefix match is stable across identical turns.
            parts.append(
                "tools:" + json.dumps(tools, ensure_ascii=False, sort_keys=True) + "\n"
            )
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
    """Realised (text, CACHED-ON-D) outcomes; the front's only price source.

    #1324: WHAT THIS RECORDS IS A PRESENCE WITNESS, NOT A PREFILL WITNESS,
    and the two used to share one spelling. That conflation is the whole of
    boot weg2sn6s's wall (k), so the rule is stated here rather than left to
    the call sites:

      "P prefilled this text" and "D can serve this text without prefilling
      it" are DIFFERENT FACTS and must not share a spelling.

    MEASURED, weg2sn6s rid weg2-2-2 (2026-09-10). ``record`` was fed P's
    leg-1 ``prompt_tokens`` -- 109,132 at 15:21:49, the moment P FINISHED
    PREFILLING and long before its write-through had landed. The repeat
    arrived 46 s later, ``span_tokens`` credited 98,261 of it, the front
    priced ``uncached=10871`` against ``X=11101`` and routed SHORT with
    ``span_known=True``. D then read what the store actually HELD -- 53,247
    of 109,132, because the write-through is asynchronous -- priced the
    remaining 55,885 as uncached and refused by name (W31 -> W50 after the
    first stream byte, so no re-route). Divergence 45,014 tokens, and
    ``span_known=True`` was the assurance that carried it.

    So the entries are now fed the MEASURED cached-on-D quantity: a D leg-2
    response's own ``cached_tokens`` (``matched + loaded`` on the tree's
    emitter line), which is exactly "what D did not have to prefill". P's
    leg-1 numbers feed NOTHING here: under W38 P reads no store at all, so
    its ``cached_tokens`` witnesses P's own device tier, and its
    ``prompt_tokens`` witnesses a prefill whose write-through may still be in
    flight. Neither is a statement about what D can read back.

    THE DANGER DIRECTION IS OVER-CREDITING, and this feed can only
    under-credit: a text D has never served carries no entry, so the whole
    prompt prices as uncached and the request takes the P route (one prefill
    on P, the soft no-double-prefill goal paying for a hard correctness
    bound) instead of a SHORT that D refuses by construction. An
    under-credited span costs a leg; an over-credited one costs the request.

    NOT A STORE INDEX. The front has no tokenizer and no store key index, and
    building one here would be the second bookkeeping this tree deletes on
    sight (``_note_p_prefix_reuse`` says so at length). This is one measured
    outcome per text, and the ONE witness for "cached on D" -- the same
    quantity D's own store-priced match (``_weg2_local_store_matches`` over
    ``store_presence_pages``) votes on, observed from the response instead of
    re-derived at the front.
    """

    def __init__(self, cap: int = SPAN_LRU):
        self.cap = cap
        self.entries: collections.OrderedDict[str, Tuple[str, int]] = collections.OrderedDict()

    def record_presence(self, text: str, cached_tokens: int) -> None:
        """One MEASURED cached-on-D outcome for ``text``.

        ``cached_tokens`` is a D leg-2 response's own ``cached_tokens``.
        NAMED ``record_presence`` and not ``record`` on purpose: the old name
        took whatever token count a caller had to hand, and the caller that
        had ``prompt_tokens`` passed it. A name that states the quantity is
        the only guard that survives the next reader.

        A MEASURED ZERO RETRACTS, it does not abstain. ``cached_tokens == 0``
        for a text is D saying "I hold none of this", and leaving an older,
        larger entry standing under that measurement is precisely the stale
        credit that routes the next repeat SHORT into a W50. The old guard
        (``prompt_tokens <= 0`` -> return) could not distinguish "no reading"
        from "a reading of nothing".
        """
        if not text:
            return
        key = hashlib.sha1(text.encode()).hexdigest()
        self.entries.pop(key, None)
        if int(cached_tokens) <= 0:
            return
        self.entries[key] = (text, int(cached_tokens))
        while len(self.entries) > self.cap:
            self.entries.popitem(last=False)

    def span_tokens(self, text: str) -> Tuple[int, bool]:
        """(tokens D is MEASURED to hold for this text's prefix, known).

        ``known`` says a PRESENCE witness exists for a prefix of this text --
        never that a prefill happened (#1324). The scaling stays what it was:
        the longest common character prefix against a text D served, times
        that text's realised cached share.
        """
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
    """(estimated uncached tokens, estimated prompt tokens, presence_known).

    #1324: the subtracted span is the MEASURED cached-on-D presence, never a
    prefill. The third term is therefore "a presence witness exists", which
    is what the SHORT bound needs to hear; see :class:`SpanLRU`.
    """
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
    if isinstance(u, dict) and "prompt_tokens" not in u and "input_tokens" in u:
        # Q0-B: the ANTHROPIC shape. `/v1/messages` prices a non-streamed
        # answer with `usage.input_tokens` / `output_tokens`, and reports the
        # cached half as `cache_read_input_tokens`. Without this branch the
        # body reads as UNPRICED and `leg2` refuses a healthy 200 by W28 --
        # i.e. forwarding the path without teaching the pricer would turn the
        # front's 404 into a 503, which is not an improvement.
        pt = int(u.get("input_tokens", 0) or 0)
        ct = int(u.get("cache_read_input_tokens", 0) or 0)
        return pt, ct, int(u.get("output_tokens", 0) or 0), True
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


class AnthropicStreamUsage:
    """Roll up an Anthropic SSE stream's usage ACROSS THE WHOLE STREAM.

    Q0-B, and the reason this is a fed accumulator rather than one more tail
    scan: on the Anthropic wire the PROMPT count is announced exactly once,
    in ``message_start``, at the very FRONT of the stream, while
    ``message_delta`` carries only the running ``output_tokens`` and
    ``message_stop`` carries none. ``leg2`` retains a bounded tail (256 KiB,
    trimmed to the last 128 KiB), so on any answer longer than that window
    ``message_start`` has already been discarded by the time
    ``usage_of_stream_tail`` runs -- the price would silently read 0 and the
    request would land in W28 as 'unpriced'. Feeding every chunk past this
    object as it is written to the client costs one line-split per chunk and
    cannot be trimmed away.

    Chunk boundaries do not respect SSE line boundaries, so the trailing
    partial line is buffered rather than parsed and discarded.
    """

    def __init__(self) -> None:
        self.input_tokens = 0
        self.cached_tokens = 0
        self.output_tokens = 0
        self.saw_start = False
        self.saw_stop = False
        self._buf = b""

    def feed(self, chunk: bytes) -> None:
        self._buf += chunk
        # Keep the last (possibly partial) line in the buffer.
        lines = self._buf.split(b"\n")
        self._buf = lines.pop()
        for raw in lines:
            self._line(raw)

    def _line(self, raw: bytes) -> None:
        line = raw.strip()
        if not line.startswith(b"data:"):
            return
        data = line[5:].strip()
        if not data or data == b"[DONE]":
            return
        try:
            js = json.loads(data)
        except Exception:  # noqa: BLE001
            return
        if not isinstance(js, dict):
            return
        kind = js.get("type")
        if kind == "message_start":
            self.saw_start = True
            u = ((js.get("message") or {}).get("usage")) or {}
            if isinstance(u, dict):
                self.input_tokens = int(u.get("input_tokens", 0) or 0)
                self.cached_tokens = int(u.get("cache_read_input_tokens", 0) or 0)
                # message_start already carries the first output token.
                self.output_tokens = max(
                    self.output_tokens, int(u.get("output_tokens", 0) or 0)
                )
        elif kind == "message_delta":
            u = js.get("usage") or {}
            if isinstance(u, dict):
                # Cumulative on the Anthropic wire; max() keeps it monotone
                # even if a producer ever sends it per-delta.
                self.output_tokens = max(
                    self.output_tokens, int(u.get("output_tokens", 0) or 0)
                )
                if not self.input_tokens:
                    self.input_tokens = int(u.get("input_tokens", 0) or 0)
        elif kind == "message_stop":
            self.saw_stop = True

    def result(self) -> Tuple[int, int, int, bool]:
        """(prompt_tokens, cached_tokens, completion_tokens, priced).

        ``priced`` is the same contract as ``usage_of``: a prompt count was
        actually seen on the wire. ``saw_stop`` is NOT required -- a stream cut
        short still carries a real input count from ``message_start``, and
        refusing to price it would re-introduce the W28 fail-closed this class
        exists to prevent.
        """
        if self.saw_start and self.input_tokens > 0:
            return self.input_tokens, self.cached_tokens, self.output_tokens, True
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


#: Provenance of the seat gate's ``need``, printed on its own L-line.
NEED_REALISED = "realised"
NEED_ESTIMATE = "estimate"


def d_seat_need(est_tokens: int, realised_tokens: int = 0):
    """``(need, source)`` for the D seat gate -- FIX 7 (round 7), half 2.

    THE OVER-PRICE, MEASURED.  Boot weg2sc3 held requests at ``need=15047``
    whose realised extent was 8,865 tokens: 1.71x, because ``need`` was the
    front's ``len(text)/3`` arrival estimate and nothing ever replaced it.
    Two of those charge 30,094 against a 27,466-token limit, so the effective
    concurrency of a six-seat group was TWO -- the estimate, not the pool,
    was the binding constraint.

    THE REALISED COUNT IS ALREADY IN HAND and costs nothing to use: group P's
    leg 1 answers with the tokenizer's own ``prompt_tokens`` for this exact
    prompt (``WEG2-SERVED group=P leg=1 ... prompt_tokens=``), and a
    re-queued request has been through leg 1 by definition.  So the estimate
    is what a COLD arrival is priced at, and only that.

    The source is RETURNED rather than inferred at the log site, because
    "estimate" and "realised" are the same integer with different error bars
    and a line that cannot say which cannot be read at all.
    """
    realised = max(0, int(realised_tokens or 0))
    if realised > 0:
        return realised, NEED_REALISED
    return max(0, int(est_tokens or 0)), NEED_ESTIMATE


#: The name D's own gate refuses with (C11).  The front never re-prices the
#: body -- D's gate is the authority -- so the only question anywhere on the
#: leg-2 path is whether this NAME is present.
#:
#: RENUMBERED TWICE, W31 -> W47 (train fix 5) -> W50 (this branch).  W31 named
#: TWO unrelated refusals on this rig: this serving-path re-route and
#: ``Weg2HostRingExhausted`` in host_ring.cpp, the host-ring exhaustion that
#: killed boot weg2tr2.  A census that greps ``W31`` reports one number for a
#: fatal host-ring exhaustion and a routing event, which is how a boot
#: postmortem merges a killer into noise.
#:
#: THE FIRST RENUMBER LANDED ON ANOTHER TAKEN CODE: the #1235 argv slice had
#: already assigned W47 to ``Weg2TpObjectiveRefused`` (launcher.py), so fix 5
#: swapped one collision for another -- which is what a hand-picked "next"
#: number does when nobody enumerates the used set first.  W50 was chosen by
#: ENUMERATING every ``W<nn>`` token in the weg2 surface (highest real code 49;
#: 15, 18, 23 and 39 are also free) and the enumeration is now a registered
#: guard, ``test_weg2_wcode_uniqueness_1263``, so the third instance of this
#: class cannot be found by a reader again.
#:
#: THE DETECTOR IS KEYED ON THE EXCEPTION NAME, NOT THE NUMBER (:data:
#: `X_REFUSAL_MARKER`).  That is the durable half of this fix: the number is a
#: label humans and greps read, the name is what identifies the refusal, and a
#: wire protocol keyed on a renumberable label breaks silently at exactly the
#: renumber that fixes the collision.
X_REFUSAL_MARKER = "Weg2TpPrefillExceeded"
X_REFUSAL_NAME = "W50 " + X_REFUSAL_MARKER

#: #1290. W52 is free: the used set at this tip is W50 (Weg2TpPrefillExceeded)
#: and W51 (Weg2HostRingUnfunded), enumerated before choosing rather than
#: guessed -- the renumbering note above W50 is about exactly that mistake.
NO_ROUTE_MARKER = "Weg2NoServiceableRoute"
NO_ROUTE_NAME = "W52 " + NO_ROUTE_MARKER

#: #1291. W53 is free: the used set is W50/W51/W52, enumerated before
#: choosing. This names a DIFFERENT fault from W52 on purpose -- W52 is "no
#: route can serve this", W53 is "the route ran and its result did not reach
#: the group that needed it", which is a store/carrier fault and points at a
#: different repair.
HANDBACK_MARKER = "Weg2StoreHandbackFailed"
HANDBACK_NAME = "W53 " + HANDBACK_MARKER

#: D states its priced extent in the W50 body it sends back
#: (`scheduler.py::_weg2_answer_x_refusals`). Parsed rather than re-derived:
#: it is the only number that says whether the store handed anything back.
_D_EXTENT_RE = re.compile(r"extent after prefix matching is (\d+)")


def _d_refusal_extent(body: bytes) -> Optional[int]:
    """The uncached extent D priced, or None when its message does not say.

    None means UNKNOWN and never 0: a caller must not read an unparsed body
    as "the store returned nothing" -- that is the estimate-terminates trap
    #1290 round 2 closed one layer up.
    """
    try:
        m = _D_EXTENT_RE.search(body.decode(errors="replace"))
    except Exception:  # noqa: BLE001 - a parser may never break admission
        return None
    return int(m.group(1)) if m else None


def serviceable_route(uncached: int, carrier_est: int, x_tokens: int,
                      carrier_max: int, carrier_exact: bool = False) -> str:
    """#1290: WHICH ROUTE CAN SERVE THIS REQUEST -- decided ONCE, up front.

    Returns one of ``short`` / ``long`` / ``carrier_single`` / ``none``.

    THE TWO BOUNDS ASK DIFFERENT QUESTIONS OF DIFFERENT TOKEN BASES, and
    conflating them is the defect this function exists to end:

    * ``x_tokens`` (``--tp-prefill-max-tokens``) bounds what group D can
      PREFILL. D re-derives the extent after ``match_prefix``, so the base is
      the UNCACHED remainder -- the tokens D would actually have to compute.
    * ``carrier_max`` bounds what can move through the host staging pool as
      KV. That is the WHOLE prompt's KV, cached prefix included, so its base
      is the total estimate ``carrier_est``.

    Both bases are correct FOR THEIR OWN QUESTION. What was missing is that
    nobody asked them TOGETHER before committing to a route:

    MEASURED, boot weg2sb5f long-prompt arm (2026-09-09, 3 workers x 12.45
    min): every request carried ~22,200 UNCACHED tokens against X=12,944 and
    carrier_max=27,466, with a total estimate above the carrier. The router's
    first branch saw only the carrier, printed ``CARRIER-EXCEEDS -> D single
    prefill``, and handed D a prefill 1.7x its own cap. D refused all 93 by
    name (W50 x159 including the re-offers), the front re-queued in band, and
    93 of 93 ended in HTTP 503 after a median 22.0 s with ZERO tokens
    streamed. Route census for the whole boot: SHORT 487, CARRIER-EXCEEDS 97,
    BATCH 7, LONG 0 -- the P route was never chosen for a long prompt at all.

    THE INVARIANT: a request whose uncached extent exceeds D's prefill cap is
    NEVER routed to a D single prefill, because D refuses it BY CONSTRUCTION
    and no amount of retrying changes a static cap.

    #1317d AMENDED THE SECOND HALF. It used to read: "If the carrier also
    refuses it, no route exists and the honest answer is ONE named refusal at
    admission." That was true while D's host staging pool was a CAP on prompt
    length. It is not any more -- design A prices D's extent against store
    presence and the window loop streams the span through the pool in windows
    -- so "the carrier refuses it" no longer means "no route exists". The
    two-leg P route serves that population, and ``none`` is now unreachable
    from the carrier bound. It survives in this function's vocabulary only for
    a caller that disables the P route entirely.
    """
    fits_d_prefill = x_tokens <= 0 or uncached <= x_tokens
    fits_carrier = carrier_max <= 0 or carrier_est <= carrier_max
    # #1317d SCOPE 3 (user ruling 2026-09-10): THE CARRIER IS NO LONGER A CAP
    # ON PROMPT LENGTH, SO IT NO LONGER TERMINATES A ROUTE.
    #
    # `carrier_max` was "what can move through D's host staging pool AS ONE
    # PIECE", and the whole of #1317 exists to end that: design A prices D's
    # uncached extent against STORE presence, and the window loop streams the
    # store-backed span through the staging pool in W-sized windows. The pool
    # is TRANSIT now, not a ceiling. So a prompt above the carrier gets the
    # two-leg route it always should have had -- leg 1 on P, write-through to
    # L3, then offered to D -- instead of a D single prefill or a 413.
    #
    # THE POPULATION THIS IS FOR, measured: the user's prompts are 50-100k
    # tokens. Boot weg2sn6c, rid weg2-2-4: `W52 Weg2NoServiceableRoute
    # uncached=18453 exceeds X=8742 ... carrier_est=48011 exceeds
    # carrier_max=27466` -- a 413 at admission for a prompt the rig can serve,
    # and W52 appeared 1x in the front log and 0x in D's, i.e. D never even
    # got to price it. That request is exactly what this branch now routes.
    #
    # THE FALLBACK CHANGED IN #1317n, and this paragraph is corrected rather
    # than left standing: it used to end at D's `exempt_carrier_exceeds` arm,
    # which admitted an unvouched prompt as a single prefill on D. That arm is
    # DELETED -- D's L2 is now derived from `--max-kv-per-request`, so below the
    # cap the store carries the whole prefix in ONE prefetch and there is no
    # band in which a request is both servable and exempt. If the store cannot
    # vouch for the prefix when D prices it (a cold first pass whose
    # write-through has not landed, a failed backup), D's uncached extent stays
    # large and it takes the ordinary NAMED exit (W31/W50), which the front
    # re-routes once under W35 -- never a silent prefill over X. So the chain
    # is: store vouches -> A credits -> D decodes with a remainder <= one
    # chunk; store silent -> named refusal, one re-route. Still never a
    # hand-priced guess at the front: the front stops deciding what D can do
    # and lets D's own admission verdict decide, which is what the ruling
    # asked for.
    #
    # PHASE LAW, STATED PLAINLY RATHER THAN GLOSSED: on the credited path D
    # prefills at most one chunk (anchor-capped store depth + the #939 law,
    # and X >= C on every path), so the law holds. On the FALLBACK path D
    # prefills above X -- that is the pre-existing exemption, it is the ONLY
    # path on which it happens, and it is retained here by explicit user
    # ruling ("exemption + 413 band stay until A is proven on metal"). It is
    # not introduced by this commit and it is not hidden by it.
    if not fits_carrier:
        # The store cannot be read into D, so the two-leg route is out: only a
        # single prefill on D could serve it -- and only if D can prefill it.
        if fits_d_prefill:
            return "carrier_single"
        # A TERMINAL REFUSAL MAY NOT REST ON AN ESTIMATE (#1290, round 2).
        # `carrier_est` is `len(text) / CARRIER_CHARS_PER_TOKEN` whenever the
        # front has no EXACT prompt-token count for this text -- and it never
        # has one for a first-time prompt, which every long request is. That
        # constant (2.4) deliberately OVER-prices tokens so the carrier is
        # never under-estimated; on the sb5f salad the real ratio was ~3.0, so
        # a 22,169-token prompt priced out at ~27.5k+ against carrier_max
        # 27,466 -- refused on ~25% of estimator conservatism.
        #
        # Refusing a request outright on that number would turn a deliberate
        # over-estimate into a hard 413 for prompts the rig can actually
        # serve, which is a worse failure than the slow one. So an estimate
        # may only DOWNGRADE the route, never terminate it: the request goes
        # to D as before, and the exact count taken at the response
        # (`_note_exact`) makes the NEXT decision on this text terminal.
        # #1317d: WAS `return "none" if carrier_exact else "carrier_single"`.
        # Both answers were wrong once the window loop landed: "none" is a 413
        # for a prompt P can prefill and the store can hand back, and
        # "carrier_single" hands D a prefill 1.7x its own cap (the sb5f
        # measurement in this docstring). The P route serves it, and the
        # `carrier_exact` distinction stops mattering here -- it existed only
        # to keep an ESTIMATE from producing a terminal refusal, and this
        # branch no longer produces one at all.
        return "long"
    if fits_d_prefill:
        return "short"
    # X < uncached <= carrier: THE P ROUTE. This is the verdict that produced
    # 0 of 591 on sb5f; P prefills leg 1 and the KV comes back to D through
    # the carrier, which by this branch it fits.
    return "long"


def weg2_drain_progress_delta(
    before: Optional[dict], after: Optional[dict]
) -> Optional[dict]:
    """#1317c: did D make decode progress across a drain window?

    Pure, so the drain's verdict can be tested without a front, a group or an
    event loop -- which is the whole reason the predicate lives here and not
    inline in `flip`.

    Returns None when EITHER sample is missing: an unreadable counter is an
    absence and the caller must fall back to the old residency behaviour, never
    read it as "no progress".

    WHAT COUNTS AS PROGRESS, and why ``forward_ct`` deliberately does NOT.
    Progress is WORK DELIVERED: decode tokens emitted (``gen_tokens_total``) or
    prefill tokens consumed (``prefill_tokens_total``, so a request still
    chunk-prefilling a long prompt is not called stalled either). ``forward_ct``
    is REPORTED as a second witness but never licenses a wait on its own,
    because a livelock that spins forward passes and emits nothing is exactly
    the wedge this guard exists to catch -- crediting it as progress would take
    the guard's teeth out while looking like a safety improvement. So:
    tokens move -> WAIT; passes move but no tokens -> still a refusal, and the
    line prints both numbers so a reader can see which shape it was.

    A counter that went BACKWARDS is treated as no progress, not as a negative
    delta -- that means a restart or a rebind, and the honest answer for the
    window is "cannot say it worked".
    """
    if not isinstance(before, dict) or not isinstance(after, dict):
        return None
    tok = int(after.get("gen_tokens_total", 0) or 0) - int(
        before.get("gen_tokens_total", 0) or 0
    )
    pre = int(after.get("prefill_tokens_total", 0) or 0) - int(
        before.get("prefill_tokens_total", 0) or 0
    )
    fwd = int(after.get("forward_ct", 0) or 0) - int(
        before.get("forward_ct", 0) or 0
    )
    return {
        "tokens": max(0, tok),
        "prefill_tokens": max(0, pre),
        "forward": max(0, fwd),
        "running": int(after.get("running", 0) or 0),
        "progressed": tok > 0 or pre > 0,
    }


def x_refusal_marker_in(body_text: str) -> bool:
    """True iff this text carries D's named Weg2TpPrefillExceeded refusal.

    FIX 2 (round 1): the refusal reaches the front in TWO wire shapes and
    only one of them carries a status.  `tokenizer_manager.py:1518-1537`
    raises `HTTPException(503, detail=message)` only when the request is NOT
    streamed; for a streamed request the same abort is yielded as an
    IN-BAND chunk on an otherwise 200 response.  A status-only test
    therefore sees exactly half of law 4's traffic, and OpenAI chat
    completions under agent load are the streamed half.

    Matched on :data:`X_REFUSAL_MARKER` -- the exception NAME -- so the W-code
    renumber that resolved the W31 collision cannot silently stop the front
    recognising D's refusal (train fix 5).
    """
    return X_REFUSAL_MARKER in (body_text or "")


def is_x_refusal(status: int, body_text: str) -> bool:
    """True iff D answered this leg 2 with the named W31 refusal (C11/C12).

    The front does not re-price the body: D's gate is the authority (the
    front's own number is an ESTIMATE, L10), so the only question here is
    whether the group refused BY NAME.  A 503 that is not W31 stays a 503.
    """
    return status == 503 and x_refusal_marker_in(body_text)


async def _first_stream_chunk(r) -> Optional[bytes]:
    """The first body chunk of a streamed response, or None when empty.

    Read BEFORE `resp.prepare()` so the in-band W31 shape is still
    re-routable (FIX 2).  Bounded by the response itself: exactly one
    `__anext__`, no buffering, no timeout of its own -- the leg-2 session
    timeout is the bound, as it is for every other chunk.
    """
    async for chunk in r.content.iter_any():
        return chunk
    return None


#: Q0-B: how far past the envelope the re-route lookahead may read on the
#: Anthropic wire, before it gives up and commits the stream to the client.
#: Bounded in BOTH dimensions so a well-behaved stream can never be delayed
#: by more than one small envelope: the refusal, when it comes, is the event
#: immediately after ``message_start``.
ANTHROPIC_LOOKAHEAD_MAX_CHUNKS = 8
ANTHROPIC_LOOKAHEAD_MAX_BYTES = 65536
#: The events that mean "generation has actually begun". Reaching one of
#: these ends the lookahead: from here on there is content to lose, so the
#: stream is committed and a later refusal is counted by name (W50), exactly
#: as it is on the OpenAI wire.
ANTHROPIC_CONTENT_MARKERS = (b"content_block_delta", b"content_block_start")


async def _anthropic_refusal_lookahead(r) -> Tuple[Optional[bytes], bool]:
    """(buffered head, refused) for a streamed Anthropic leg 2.

    Q0-B, measured 2026-09-09 on boot weg2sn5m: ``_first_stream_chunk`` above
    is calibrated for the OPENAI wire, where a request refused at admission
    is aborted before it decodes anything, so the refusal IS the first chunk.
    On the ANTHROPIC wire it structurally CANNOT be: ``message_start`` is
    always emitted first, carrying the id, model and input usage, and the
    ``error`` event follows it. The one-chunk test therefore saw
    ``message_start``, found no marker, committed the response, and the
    refusal arrived after the first byte -- where re-routing is impossible.
    Measured shape of that boot: ``events={'message_start': 1, 'error': 1,
    'message_stop': 1}``, 0 chars of text, front logged
    ``W50 ... (STREAM, served) ... re-route impossible``.

    So the decision this lookahead has to make is not "is the FIRST CHUNK a
    refusal" but "does the refusal arrive before the first CONTENT" -- and on
    a wire whose envelope precedes its content, those differ by exactly one
    event. Everything read here is returned and forwarded verbatim, so the
    client sees a byte-identical stream either way; the only cost on the
    happy path is buffering one envelope.
    """
    head = bytearray()
    for _ in range(ANTHROPIC_LOOKAHEAD_MAX_CHUNKS):
        chunk = await _first_stream_chunk(r)
        if chunk is None:
            break
        head.extend(chunk)
        if x_refusal_marker_in(bytes(head).decode(errors="replace")):
            return bytes(head), True
        if any(m in head for m in ANTHROPIC_CONTENT_MARKERS):
            break
        if len(head) >= ANTHROPIC_LOOKAHEAD_MAX_BYTES:
            break
    return (bytes(head) if head else None), False


def witness_verdict(front_outstanding: int, rank_idle: bool) -> Optional[str]:
    """W3 in either direction; None when the two witnesses agree."""
    if front_outstanding == 0 and rank_idle:
        return None
    if front_outstanding > 0 and not rank_idle:
        return None
    if front_outstanding == 0 and not rank_idle:
        return "front drained, rank NOT idle"
    return "rank idle, front still holds requests"


def flip_stage_report(stage: str, age_s: Optional[float]) -> str:
    """``stage_last_known=<stage> age_s=<n>`` -- NEVER a bare ``stage=``.

    #1264 fix 2b (1).  The weg2t2b stall line said ``stage=sleep-kv`` at
    12:45:07Z about a stage that had COMPLETED at 12:43:04Z (D logged
    ``WEG2-DORMANT set`` and the RPC returned 200) and a controller that had
    been dead since 12:43:04,131Z.  ``_flip_stage`` is only advanced at the
    NEXT stage, so after an escape it names the last stage ENTERED, not the
    stage running -- and the bare ``stage=`` spelling asserted the second.
    That one word sent the triage two minutes and one whole boot analysis down
    the HiCache drain, which was a consequence and not the cause.

    So the name carries the epistemics: ``stage_last_known`` says "this is the
    last stage that was entered", and ``age_s`` is what makes it checkable --
    an age far larger than any stage's plausible duration IS the signal that
    nothing is advancing it.  On weg2t2b this would have read
    ``stage_last_known=sleep-kv age_s=123.1``, against a sleep RPC that had
    answered in 0.6 s.

    ``age_s`` is ``None`` only when no stage was ever stamped; it prints
    ``unknown`` rather than a zero, because a zero here reads as "just now".
    """
    age = "unknown" if age_s is None else f"{age_s:.1f}"
    return f"stage_last_known={stage} age_s={age}"


def controller_dead_line(
    epoch: int, stage: str, age_s: Optional[float], exc: BaseException
) -> str:
    """The one line a controller death must produce, at the moment it dies.

    #1264 fix 2b (2).  On weg2t2b the death produced ``controller error:
    cannot unpack non-iterable CardFree object`` -- a generic handler message
    with no marker, no stage and no verdict -- and the loop continued into an
    idle ``serving`` check forever.  Only the 4x stall timer spoke, 123.7 s
    later, and it spoke about the wrong thing.  A tool is built only when it is
    wired: this line is the deadman's tier-3 pattern (boot_deadman.sh), so the
    death is a VERDICT within one poll instead of a log entry someone reads
    afterwards.

    Pure, so both the wording and the deadman's pattern can be tested against
    the same string without a front, a socket or a boot.
    """
    return (
        f"WEG2-FLIP CONTROLLER-DEAD epoch={epoch} "
        f"{flip_stage_report(stage, age_s)} exc={type(exc).__name__} "
        f"-- the controller loop raised while a flip was OPEN, so the flip took "
        f"none of its named exits and `state` stays 'flipping'; the loop's own "
        f"guard then skips every later iteration. VRAM occupancy is undefined "
        f"on both groups. No retry; recovery = teardown + relaunch"
    )


def flip_escape_verdict(
    state: str, stage: str, epoch: int, exc: BaseException
) -> Optional[Tuple[str, str]]:
    """``(W-code, detail)`` when an exception escaped an OPEN flip, else None.

    #1264 (A).  ``Weg2Front.flip`` sets ``state="flipping"`` as its first
    statement and clears it only on its NAMED exits (W1 refusal, W3 witness
    disagreement, W4 RPC failure, W19, success).  An unexpected exception
    escapes past every one of them, and the controller's ``except Exception``
    used to log it and ``continue`` -- but the loop's own first statement is
    ``if self.state != "serving": continue``, so "continue" means the front
    does nothing for the rest of the boot while ``/health`` keeps answering
    200.  Measured weg2t2b (2026-09-08): a ``TypeError`` at 12:43:04,131Z, 0.6 s
    after ``WEG2-FLIP begin``, and the front never flipped again; weg2t2a held
    the same shape for seven minutes.

    An exception that escaped mid-flip is not a recoverable error.  The source's
    kv_cache (and possibly part of its weights family) is paused and the
    destination is not resumed, so VRAM occupancy is undefined on both groups --
    exactly the state W4 already names when an RPC fails at the same point.  So
    the verdict is W4, and the STAGE is carried because it is what says how far
    the flip got.

    Pure so it can be tested without a front, a session or an event loop: the
    caller does the stopping.
    """
    if state != "flipping":
        return None
    return (
        "W4 Weg2WakeRefused",
        f"unhandled {type(exc).__name__} during the flip of epoch {epoch} at "
        f"stage {stage!r}: {exc} -- the flip neither completed nor took one of "
        f"its named exits, so VRAM occupancy is undefined on both groups and "
        f"the front would otherwise sit in state='flipping' forever (the "
        f"controller's own guard skips every iteration while it is not "
        f"'serving'). No retry; recovery = teardown + relaunch",
    )


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
    # #1273 B4k: THE INTEGER PREDICATE, not the raw prefix.  A CHUNK is a LAYER
    # BAND and only a band has a card list in ``tag_cards``; the raw
    # ``startswith("weights_")`` is also True for ``weights_draft`` (and for
    # S8's planned ``weights_vision``), which is the same defect
    # ``is_weights_family_tag`` was corrected for one layer up.  Left as the
    # prefix, the draft tag joining the family would land in ``chunks``, find no
    # entry in the chunk->card map, and take the ``missing`` branch below -- so
    # every flip would fall back to the IDENTITY order and lose the
    # tightest-card-first ordering boot weg2dk4 paid for.  Non-chunk family
    # members belong in ``rest``, with the base tag still closing the sleep.
    chunks = [t for t in tags if is_weights_chunk_tag(t)]
    rest = [t for t in tags if not is_weights_chunk_tag(t)]
    if not tag_cards:
        return tags, "identity: the source has no chunk->card map (uniform/TP source, or no map passed)"
    if not free_mib:
        return tags, "identity: no NVML free sample for this flip"
    # #1233 fix 6: an EMPTY card list is as unusable as an absent tag, and the
    # difference used to be a crash instead of a refusal -- ``min()`` over an
    # empty sequence raises ValueError inside ``flip``, i.e. at the one moment
    # the flip must not fail.  ``chunk_tag_cards`` cannot emit an empty tuple
    # today (a tag exists only once a layer lands in it), so this is a latent
    # shape, and a latent shape guarded by nothing is how weg2dk4's order came
    # to be trusted.
    missing = [t for t in chunks if not tag_cards.get(t)]
    if missing:
        return tags, (
            "identity REFUSED to reorder: chunk tags absent from the map or with "
            f"no cards {missing}"
        )
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
    #: #1271 (c): the UNCACHED remainder priced at route time -- the
    #: quantity the flip actually has to redo. `est_prompt` counts the
    #: cached head too, which P does not recompute, so a backlog summed
    #: on it over-states the work and flips on prefixes already resident.
    est_uncached: int = 0
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
    #: MF-3: the tokens of this prompt whose prefix the front's OWN ROUTING
    #: PROBE already priced as store-resident, captured at arrival because
    #: the probe's source (:class:`SpanLRU`) is mutated by this very request
    #: once leg 1 answers.  See :meth:`Front._note_p_prefix_reuse`.
    store_span_est: int = 0


class Seat:
    """One of ``--d-bs`` concurrency seats on group D (C4/C5).

    ONE acquire site per path (the admitter for BATCH, ``handle_generate``
    for SHORT) and ONE release site (:meth:`release`, called from ``leg2``'s
    ``finally`` and from the two paths that hand the request back before a
    leg 2 exists).  ``held`` makes the release idempotent, so a path that
    releases early and then falls through the ``finally`` cannot return a
    seat twice and inflate D's concurrency past its own bs.
    """

    __slots__ = ("front", "rid", "source", "held", "tokens", "t_taken")

    def __init__(self, front: Front, rid: str, source: str, tokens: int = 0):
        self.front = front
        self.rid = rid
        self.source = source
        self.held = True
        # FIX 4a (round 1): a seat is a COUNT of one AND a charge against
        # D's host staging pool.
        #
        # FIX 4 (round 4) -- THE CHARGE IS DERIVED, NOT RECORDED.  Round 1
        # kept a `_d_inflight_tokens` counter here, and a counter written
        # only by this class is by construction back to 0 at the start of
        # every D epoch: it could never see the standing residency that
        # actually refuses the store read (the INDIKATOR-GESETZ finding of
        # round 3).  The front now prices the pool by D's OWN reading and
        # uses the live seats only for the grants that reading cannot yet
        # contain, so what a seat needs to carry is its size and the MOMENT
        # it was granted -- both immutable, both read by summation over the
        # live set.  No counter, no reconcile, no drift.
        self.tokens = max(0, int(tokens))
        self.t_taken = time.time()
        self.front._d_seats_live.add(self)

    def release(self, freed_by: str) -> None:
        if not self.held:
            return
        self.held = False
        self.front._d_seat.release()
        self.front._d_seats_live.discard(self)
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


#: THE CORRIDOR BAND IS NOT DECLARED HERE.  ``managers.corridor_guard`` is THE
#: ONE DECLARATION (``CORRIDOR_LAW_MIB`` +- ``CORRIDOR_BAND_FRACTION``, and the
#: ``SGLANG_CORRIDOR_LAW_FLOOR_MIB`` override read per call); this module reads
#: it and never repeats the literal.
#:
#: FIX 2, finding 2: the predecessor of this line froze ``819`` / ``1229`` here
#: with the comment "the corridor law, verbatim" -- a fourth private copy of a
#: number whose own authority says, at ``corridor_guard.py:141``, that "every
#: other module that needs the law imports it from here rather than repeating
#: the literal".  A frozen copy cannot follow the override: with
#: ``SGLANG_CORRIDOR_LAW_FLOOR_MIB=1536`` the guard grades against 1228-1843
#: while this file would still have printed ``band=819-1229MiB`` and graded
#: every ``verdict=`` against a law not in force.  Imported as FUNCTIONS, not
#: as values, for the same reason the guard reads its env per call.
#:
#: The band is stated in ALLOCATABLE free, which is the only quantity the
#: driver will actually hand to an allocation -- see :data:`CORRIDOR_INSTRUMENT`.
#: What every WEG2-CORRIDOR line says it measured, printed in the line itself.
#:
#: DEFECT THIS NAME EXISTS TO CLOSE (boot weg2rg6, 2026-09-08): this sampler
#: used to shell out to ``nvidia-smi --query-gpu=memory.used,memory.total`` and
#: print ``total - used``.  That subtraction is FREE PLUS THE DRIVER CARVE-OUT,
#: because nvidia-smi's ``memory.used`` (like NVML's v2 ``used``) already
#: EXCLUDES the carve-out.  Measured at one instant, 07:31:30Z, this line
#: against ``memory.free``: nvml0 1454 vs 1030, nvml1 859 vs 341, nvml2 1276 vs
#: 852 -- overstatements of exactly 424 / 518 / 424 MiB.  The 5090 was 478 MiB
#: BELOW the corridor floor while this line showed it inside the band.  The
#: corridor rule names that subtraction and forbids it by name; the rule is
#: older than the sampler (RUNSHEET_363 sec 4.3, and the August corridor
#: sampler carried the comment "FREE column only -- never total minus used").
CORRIDOR_INSTRUMENT = f"{nvml_registry.FREE_INSTRUMENT_V2},allocatable"
#: The SAME quantity read WITHOUT the v2 struct: still allocatable free (both
#: structs report it), but the carve-out beside it is unknown, so the line must
#: not claim v2.  FIX 2, degradation half of finding 3: the predecessor printed
#: ``reserved=0MiB`` under an ``instrument=nvml_v2_free`` claim in that mode.
CORRIDOR_INSTRUMENT_NO_V2 = f"{nvml_registry.FREE_INSTRUMENT_V1},allocatable(carve-out-unknown)"
#: One-shot latch so a rig without NVML says so once instead of every 10 s.
_nvml_unavailable_logged = False


@dataclass(frozen=True)
class CardFree:
    """One card's allocatable free at one instant, with its carve-out beside it."""

    nvml_index: int
    uuid: str
    free_mib: int
    reserved_mib: int
    #: False when this card's carve-out could not be read (no NVML v2 struct),
    #: which downgrades the whole line's instrument token.
    carve_out_known: bool = True


def corridor_band_mib() -> Tuple[int, int]:
    """``(floor, ceiling)`` of the corridor band IN FORCE, per call.

    Reads ``managers.corridor_guard``, the one declaration, every time --
    never a value frozen at import here.
    """
    return corridor_band_floor_mib(), corridor_band_ceiling_mib()


def corridor_instrument(cards: List[CardFree]) -> str:
    """The instrument token for a sample of these cards.

    Degrades to :data:`CORRIDOR_INSTRUMENT_NO_V2` if ANY card in the sample
    lost its carve-out: one token per line, and the weakest card sets it,
    because a reader takes the token to cover the whole line.
    """
    if cards and not all(c.carve_out_known for c in cards):
        return CORRIDOR_INSTRUMENT_NO_V2
    return CORRIDOR_INSTRUMENT


#: The user reserve this front was launched with, ``{card_uuid: MiB}``.
#: Injected by the launcher; empty means 0 on every card, which is the default
#: since #1257c. The NAME is the guard's, imported and not repeated.
RESERVE_ENV = USER_RESERVE_ENV


def _reserve_by_card() -> Dict[str, int]:
    """Delegates to ``corridor_guard.user_reserve_by_card`` -- ONE reader.

    REFUTER FINDING 6: this parse used to live here alone, which is why a
    second consumer of the derived floor (``vram_dial``) could not see the
    reserve at all.  The variable is declared in ``corridor_guard`` beside the
    floor it raises, so its reader belongs there too.
    """
    return corridor_guard.user_reserve_by_card()


def corridor_floor_for_card(card_uuid: str, group: str):
    """This card's derived corridor floor, for the group AWAKE on it.

    #1257c.  THE FLOOR IS NO LONGER RIG-UNIFORM and it is no longer a hand
    number: it is ``measured transient peak(awake group) + user reserve``, and
    where the transient is not measured it is the named
    ``UNMEASURED-FALLBACK`` 1024 -- which grades and prints but may never
    actuate a budget cut.  Read from ``managers.corridor_guard``, the ONE
    declaration, on every call, for the same reason the band always was: a
    value frozen here cannot follow an override.
    """
    return corridor_floor_mib(
        card_uuid,
        group=group,
        user_reserve_mib=_reserve_by_card().get(card_uuid, 0),
    )


def corridor_verdict(free_mib: int, floor=None) -> str:
    """``IN`` / ``BELOW`` / ``ABOVE`` against the corridor floor IN FORCE.

    Graded on ALLOCATABLE free only.  Inclusive at both edges.

    #1257c.  ``floor`` is a :class:`~sglang.srt.managers.corridor_guard.CorridorFloor`
    for THIS card; passing none keeps the pre-#1257c rig-wide band (819-1229
    at the stated law), which is what a caller without a card identity can
    honestly grade against.

    ``ABOVE`` IS A FINDING, NOT A FAIL (user decision, 2026-09-09, consequence
    5).  It says MiB are sitting unmobilised; it never fails an acceptance on
    its own.  Consumers print it as ``unmobilised_free_mib=`` and must not
    turn it into a problem -- see :func:`corridor_arm.arm_report`.
    """
    if floor is None:
        lo, hi = corridor_band_mib()
    else:
        lo, hi = floor.verdict_floor_mib, floor.ceiling_mib
    if free_mib < lo:
        return "BELOW"
    if free_mib > hi:
        return "ABOVE"
    return "IN"


def _nvml_free() -> List[CardFree]:
    """Allocatable free per card, from the registry's ONE NVML reader.

    No second reader and no local arithmetic: ``nvml_registry.memory_snapshot``
    returns the driver's own ``free`` (which already excludes the carve-out)
    plus the carve-out itself, all cards in one NVML session so the printed
    numbers come from a single instant.
    """
    global _nvml_unavailable_logged
    try:
        snap = nvml_registry.memory_snapshot()
    except Exception as e:  # noqa: BLE001 - a corridor sample never kills the front
        if not _nvml_unavailable_logged:
            _nvml_unavailable_logged = True
            logger.warning(
                "WEG2-CORRIDOR unavailable: NVML could not be read (%s). No corridor "
                "samples will be printed this boot -- and there is deliberately no "
                "nvidia-smi fallback, because the only fallback this sampler ever had "
                "was the total-minus-used subtraction the corridor rule forbids.", e,
            )
        return []
    # FIX 2 (nonblocking sibling): the latch is a RATE LIMIT on one outage, not
    # a boot-long mute.  A read that succeeds ends the outage, so the NEXT one
    # that fails warns again -- otherwise a single transient failure silenced
    # every later one and ``corridor_sample`` returned ``None`` in silence for
    # the rest of the boot, which is the denominator law's suppressed-count
    # trap in its worst form: no line at all.
    _nvml_unavailable_logged = False
    return [
        CardFree(
            nvml_index=dev.index,
            uuid=dev.uuid,
            free_mib=mem.free_mib,
            reserved_mib=mem.reserved_mib,
            carve_out_known=mem.carve_out_known,
        )
        for dev, mem in snap
    ]


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
                 p_concurrency: int = DEFAULT_P_BS, d_bs: int = DEFAULT_D_BS,
                 tp_prefill_max_tokens: int = X_FALLBACK_TOKENS,
                 flip_min_work_tokens: Optional[int] = None,
                 min_dwell_ms: Optional[float] = None,
                 idle_layout: str = "D",
                 drain_deadline_s: float = DRAIN_DEADLINE_DEFAULT_S,
                 d_admit_max_tokens: Optional[int] = None,
                 src_chunk_cards: Optional[Dict[str, Dict[str, List[int]]]] = None,
                 measured_record: str = "", commit: str = "",
                 ledger_arm: Optional[Dict[str, float]] = None,
                 admin_key_file: str = ""):
        # #1275: the key arrives as a PATH, never as an argv value. The groups
        # have no choice (`server_args` offers only `--admin-api-key`, so their
        # key is world-readable in /proc/<pid>/cmdline), but the front does, and
        # a secret that appears in one more process's argv for no reason is a
        # second exposure bought with nothing. None here == the groups are
        # unkeyed == send no header, which is the pre-#1275 behaviour byte for
        # byte.
        self.admin_key_file = admin_key_file or ""
        self.admin_key = admin_key_mod.read(admin_key_file) if admin_key_file else None
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
        #: #1350: the non-reclaimable reading (anon+shmem+slab_unreclaimable)
        #: taken at `WEG2-FLIP begin epoch=0`, and its timestamp. The FIRST
        #: FLIP PAIR's permanent step is `post - pre`, and it is written into
        #: the sidecar at `WEG2-FLIP done epoch=2` so the NEXT boot's ledger can
        #: CHARGE it instead of leaving it to a margin term measured by an
        #: instrument that is structurally blind to a monotone rise
        #: (ANALYSE_1350_HOST_TERM_0912.md SS1.4). No new sampler and no new
        #: hook: both moments are lines this front already logs.
        self._flip_ratchet_pre_gib: Optional[float] = None
        self._flip_ratchet_pre_at: str = ""
        self._flip_ratchet_written = False
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
        #: #1269: how often the host-watermark guard samples, and the drift
        #: rate it prices the margin with.  None => host_ledger's sb4 default
        #: until a boot's WEG2-IDLE-CENSUS line supplies a measured one.
        self.host_watermark_period_s = 15.0
        self._observed_anon_drift_mib_per_min: Optional[float] = None
        #: #1269 fix 3: cgroup anon BEFORE this boot existed -- the only valid
        #: foreign baseline.  Set by the launcher's preflight (weg2sb5b measured
        #: 10.05 GiB and the guard did not use it).  None => no split is printed
        #: rather than an invented one.
        self._anon_preboot_bytes: Optional[int] = None
        self.stop: Optional[Weg2Stop] = None
        self.admit_d = True
        self.queue: Deque[Pending] = collections.deque()
        self.spans = SpanLRU()
        self.session: Optional[ClientSession] = None
        self.counters: Dict[str, int] = collections.Counter()
        self.corridor_min: Dict[str, Dict[int, int]] = {"P": {}, "D": {}}
        #: The instrument of the LAST corridor sample taken, which is what
        #: ``/weg2/state`` reports.  Starts at the nominal token so a state read
        #: before the first sample is not blank; every sample overwrites it with
        #: what that read actually was.
        self.corridor_instrument: str = CORRIDOR_INSTRUMENT
        #: #1257c: ``{card_uuid: CORRIDOR-FLOOR provenance line}`` from the
        #: last sample. Empty until the first one -- never a nominal
        #: constant, for the same reason ``corridor_instrument`` is not.
        self.corridor_floors: Dict[str, str] = {}
        self.flip_log: List[dict] = []
        # ---- #1262 TIER 3: the flip's own stall detector ------------------
        # Deadman tier 1 (a process exists) and tier 2 (/health_generate) both
        # scored boot weg2t2a ALIVE for the seven minutes its first flip hung:
        # all six schedulers were burning CPU inside an idle-time instrument
        # and this front answered 200 on all three ports throughout. Neither
        # tier can see that, and that is a coverage gap, not a deadman fault.
        # The front CAN see it -- it already owns `state`, `epoch` and the
        # measured cost of every completed flip -- so the third signal lives
        # here, in the one process that knows a flip began and has not ended.
        self._flip_t0: Optional[float] = None
        #: #1264 fix 2b (1): the stage's VALUE and the monotonic instant it was
        #: assigned, always written together -- see the `_flip_stage` property.
        self._flip_stage_value: str = "none"
        self._flip_stage_t: float = time.monotonic()
        self._flip_stage = "none"
        #: the epoch a STALL line has already been emitted for; -1 = none, so
        #: it can never collide with a real epoch (which starts at 0). One
        #: line per flip: a repeating alarm is a monitor, and persistent
        #: monitors are banned on this rig.
        self._flip_stall_reported_epoch: int = -1
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
        # #1271 (b): the live X estimator's state.
        self._x_min_work_follows = flip_min_work_tokens is None
        #: The break-even's floor is ONE CHUNK: below it the round trip
        #: cannot pay whatever the rates say. Deliberately NOT the seed X --
        #: a floor at the seed would make the re-solve monotonically
        #: non-decreasing, i.e. unable to correct an X that was too high,
        #: which is exactly the sb2 direction.
        self.x_floor_tokens = 4096
        self._x_samples = {
            "r_d": collections.deque(maxlen=self.X_SAMPLE_WINDOW),
            "r_p": collections.deque(maxlen=self.X_SAMPLE_WINDOW),
            "flip_s": collections.deque(maxlen=self.X_SAMPLE_WINDOW),
        }
        self._x_since_resolve = 0
        self._x_seed_note = f"launcher solve X={self.tp_prefill_max_tokens}"
        #: #1289: which inputs were missing at the last NO-SOLVE, so the line
        #: is emitted on every CHANGE of that set rather than once per flip.
        self._x_last_missing: List[str] = []
        #: #1289 round 2: arrivals at D, the denominator of the solo witness.
        self._d_admissions = 0
        #: Where the last r_D sample came from, printed with every X decision.
        self._x_r_d_src = "none yet"
        # C8/K7: None = derive from the last completed flip in that direction.
        self.min_dwell_ms = None if min_dwell_ms is None else float(min_dwell_ms)
        # law 5 / C6: which group is awake when nothing is pending.
        self.idle_layout = "P" if str(idle_layout).upper().startswith("P") else "D"
        # MF-1: the IDLE-REST line's edge trigger (see _idle_disposition).
        self._idle_rest_shown = False
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
        # #1233 fix 8: the DORMANT-IMAGE measurement, one per group at its FIRST
        # sleep.  The launcher takes P's (un-interleaved, before D exists); the
        # front takes each group's first sleep it sees, which for D is a flip
        # and is therefore INTERLEAVED -- the sample says so, and only the
        # RssShmem instrument measures that group's image there.
        self.measured_record = measured_record
        self.commit = commit
        self.ledger_arm = dict(ledger_arm or {})
        self.dormant_image: Dict[str, dict] = {}
        # FIX 4a (round 1), boot weg2sc1 LINK 1 -- D's CONCURRENCY IS A
        # TOKEN BUDGET, NOT ONLY A COUNT.
        #
        # C4 opens `--d-bs` seats at once and nothing couples that number to
        # D's host staging pool, which is a TOKEN budget.  Measured on
        # weg2sc1: six seats opened, three requests staged
        # (occupied=25100 = 8203+8397+8500 against limit=27466), and the
        # fourth met `#915 PREFETCH REFUSED reason=vote_negative need=8629
        # available=5418`.  A refused prefetch means `match_prefix` finds
        # nothing, `extend_input_len` becomes the WHOLE prompt, and C11's X
        # gate then refuses it correctly -- for a reason that is not the
        # request's: its prefix IS in the store, the staging pool merely had
        # no room this pass.  C12 re-queued it to P, P re-prefilled a
        # store-resident rid, and P's PP ranks then diverged on its
        # re-admission extent (W27, the boot killer).
        #
        # FIX 4 (round 4) -- WHICH QUANTITY.  Round 1 compared a front-local
        # admission tally against `carrier_max_tokens`, and boot weg2sc1's own
        # two logs refute that pairing in the same second: at the FIRST
        # admission of the epoch the tally is 0, so the gate read 27,466 rows
        # free while D read `available=5418 occupied=25100`.  Replayed, the
        # round-1 gate admits three of the burst and every one of them is
        # still #915-refused -- exactly the failure it exists to remove.  The
        # front therefore no longer prices the pool at all: it READS D's own
        # `#915` terms off `/server_info` (`prefetch_residency`) and charges
        # only the grants that reading cannot yet contain.
        #
        # `--d-admit-max-tokens` survives as the OPERATOR CEILING it always
        # was, no longer as the budget: 0 = gate off, >0 = an extra bound on
        # the group's own reading, None = the reading alone.  The gate only
        # ever DELAYS an admission, never forces one, and never starves: a
        # request with no seat in use is admitted whatever it costs (the
        # truly-oversized case is CARRIER-EXCEEDS's, not this one).
        self.d_admit_max_tokens = (
            None if d_admit_max_tokens is None else max(0, int(d_admit_max_tokens))
        )
        # The live seats, the ONLY thing the front counts itself.  A set, not
        # a counter: `Seat.__init__` adds and `Seat.release` discards, so the
        # charge is a SUM over live seats at read time and cannot drift out of
        # step with the seats it describes (round 3's "reconciled at each
        # D-epoch start rather than assumed 0", satisfied structurally).
        self._d_seats_live: Set[Seat] = set()
        # D's last published host-pool reading: the terms of its own #915
        # gate plus `t`, the moment the READ WAS ISSUED (not answered), so a
        # grant made during the round trip is charged rather than lost.
        self._d_pool: Optional[Dict[str, Any]] = None
        self._d_pool_unreadable: bool = False
        # FIX 5 (round 5): NEGATIVE CACHING.  A failed read used to leave
        # `_d_pool = None`, so the next admission decision re-issued the
        # request immediately -- a silent or slow group D then charged its
        # timeout to EVERY D admission, one after the other.  Until this
        # stamp passes, the front answers "no reading" from memory and issues
        # no HTTP at all.  Bounded by the same age constant: a failure is
        # remembered exactly as long as a success would have been.
        self._d_pool_retry_after: float = 0.0
        self._d_token_hold_rid: Optional[str] = None

    # ---------------- seat / gate bookkeeping (C4, C5) ----------------
    def seats_free(self) -> int:
        """Seats not currently held, for the L2/L3 denominators."""
        return max(0, self.d_bs - self._seats_in_use())

    def _seats_in_use(self) -> int:
        return max(0, self.d_bs - self._d_seat._value)

    def _handoff_in_flight(self) -> int:
        """FIX 2 (round 2): requests HANDED to D that D does not hold yet.

        The window the two D->P flip guards could not see.  A seat is taken
        in the same synchronous step that resolves the client's future
        (``d_admitter``, :932-937) or, on the SHORT path, immediately before
        ``leg2`` is awaited; ``leg2`` registers the rid in ``D.outstanding``
        at its first line and ``leg2``'s ``finally`` returns the seat.  So a
        seat with no matching ``outstanding`` entry is exactly a request that
        has been promised to D and has not arrived there -- invisible to
        ``_ready_for_d`` (already popped) and to ``D.outstanding`` (not yet
        registered), which are the only two sets the controller read.

        Derived, not recorded: no second ledger to keep in step with the
        seat's own lifecycle, and no state whose writer and deleter could be
        separated by an event.  ``max(0, ...)`` because a leg 2 entered
        without a seat counts in ``outstanding`` alone, and that is D work
        the drain already covers.
        """
        return max(0, self._seats_in_use() - len(self.groups["D"].outstanding))

    def _d_charged_since(self, t: float) -> int:
        """Tokens of the seats granted at or after ``t``.

        The ONLY quantity the front counts itself, and it is DERIVED from the
        live seats rather than accumulated (see :class:`Seat`).  Its whole job
        is to cover the window a reading cannot: a seat granted after D
        sampled its pool is a store read D has not yet registered, so it is
        invisible in the reading and must be charged on top of it.
        """
        return sum(s.tokens for s in self._d_seats_live if s.t_taken >= t)

    def _d_inflight_tokens(self) -> int:
        """Tokens of ALL live seats -- the L-line denominator, never a bound."""
        return sum(s.tokens for s in self._d_seats_live)

    async def _d_pool_reading(self) -> Optional[Dict[str, Any]]:
        """D's host-pool reading, refreshed ON DEMAND when it is stale.

        Called from the two admission paths immediately before they decide,
        so the read happens exactly as often as decisions are taken near the
        bound and never as a background poller.  ``t`` is stamped BEFORE the
        request goes out: a grant made while the reply is in flight then
        satisfies ``t_taken >= t`` and is charged, which is the conservative
        direction (the gate may delay, never admit on a stale reading).

        ONE BOUND, ONE MEANING: a reading older than ``D_POOL_MAX_AGE_S`` is
        treated as no reading at all, and reaching the read below therefore
        already means the last one has aged out.  A failed read leaves no
        reading and the gate goes off by name.

        FIX 5 (round 5), TWO WINDOWS THIS METHOD MUST NOT OPEN, both on the D
        admission critical path:

        * it is now called only when :meth:`_d_gate_armed` is true, so a
          disabled gate (`--d-admit-max-tokens 0`) and the never-starve exit
          (no seat in use) cost NO round trip -- see the two call sites;
        * a FAILED read is remembered for ``D_POOL_MAX_AGE_S``
          (``_d_pool_retry_after``) instead of being retried at the next
          decision, so a silent group D charges its timeout once per age
          window rather than once per admission.
        """
        r = self._d_pool
        now = time.time()
        if r is not None and now - r["t"] <= D_POOL_MAX_AGE_S:
            # THE AGE TERM IS THE SAFETY PROPERTY, not a refinement, and this
            # is its ONLY site.  Drop it and the front decides for ever on an
            # arbitrarily old residency -- a number that cannot see what
            # actually refuses the store read, which is the indicator class
            # round 3 killed.  Reaching past this check therefore ALREADY
            # means "the last reading has aged out", which is why the failure
            # path below drops to the named refusal instead of re-testing the
            # same bound: round 4 wrote that second test with `t0 = now`, so
            # it could never be true and the "a failed read does not clobber
            # the last one" half of this docstring was dead code.
            return r
        if now < self._d_pool_retry_after:
            # A read failed less than one age window ago and the last reading
            # (if any) has aged out with it: no reading, and no HTTP to find
            # that out again.  Counted separately from a failure so the
            # UNREADABLE line's denominators stay honest (a suppressed read
            # is not evidence that D answered, nor that it did not).
            self.counters["d_pool_read_suppressed"] += 1
            return None
        t0 = now
        got = None
        # #1288: WHY the read failed, not just THAT it did.  On weg2sb5f the
        # only clue in the whole front log was one line saying group D
        # "published no reading" -- while D was in fact answering 401 to every
        # request.  A refusal that cannot name its own cause costs a boot.
        why = "no exception, no 200, no reading"
        try:
            g = self.groups["D"]
            # #1288: NO BEARER, ON PURPOSE. `/server_info` is ADMIN_OPTIONAL
            # and this read is front->group over loopback, which the auth
            # decision now trusts by TRANSPORT PEER (utils/auth.py,
            # `peer_is_loopback`). Adding a header here would be a second
            # copy of the key with a second way to get it wrong -- and the
            # first way is what #1288 is: fix 5 moved the route behind the
            # gate and this caller kept sending nothing.
            async with self.session.get(
                    f"{g.url}/server_info",
                    timeout=ClientTimeout(total=D_POOL_READ_TIMEOUT_S)) as resp:
                status = resp.status
                if status == 200:
                    body = await resp.json()
                    for st in (body.get("internal_states") or []):
                        got = st.get("hicache_prefetch")
                        if got:
                            break
                    if not got:
                        why = "200 but no `hicache_prefetch` in internal_states"
                else:
                    why = f"HTTP {status}" + (
                        " -- group D refused this read; loopback should be "
                        "trusted (WEG2-AUTH line in D's log), so check that "
                        "D is running a tree that carries #1288"
                        if status in (401, 403) else "")
        except Exception as e:  # noqa: BLE001 - an instrument may never break admission
            got = None
            why = f"{type(e).__name__}: {e}"
        if got:
            self.counters["d_pool_reads"] += 1
            self._d_pool = dict(got)
            self._d_pool["t"] = t0
            self._d_pool_retry_after = 0.0
            if self._d_pool_unreadable:
                self._d_pool_unreadable = False
                logger.info("WEG2 D-POOL READABLE again: available=%d occupied=%d limit=%d "
                            "(the seats-vs-pool gate is back on)",
                            self._d_pool["available"], self._d_pool["occupied"],
                            self._d_pool["limit"])
            return self._d_pool
        self.counters["d_pool_read_failed"] += 1
        self._d_pool_retry_after = t0 + D_POOL_MAX_AGE_S
        self._d_pool = None
        if not self._d_pool_unreadable:
            self._d_pool_unreadable = True
            # NAMED REFUSAL, not a silent fallback: the round-1 gate's
            # front-local proxy is exactly the wrong quantity, so with no
            # reading the gate is OFF and says so.  Admissions then behave as
            # they did before this coupling existed -- never worse -- and D's
            # own #915 gate remains the enforcement point.
            logger.warning("WEG2 D-POOL UNREADABLE (cause=%s): group D published no "
                           "`hicache_prefetch` "
                           "reading within %.1f s (timeout=%.1f s reads=%d failed=%d "
                           "suppressed=%d); the aggregate seats-vs-pool gate is OFF until it "
                           "answers -- the front will not substitute its own admission tally "
                           "for the pool's residency, and it will not re-issue the read "
                           "before %.1f s have passed",
                           why, D_POOL_MAX_AGE_S, D_POOL_READ_TIMEOUT_S,
                           self.counters["d_pool_reads"],
                           self.counters["d_pool_read_failed"],
                           self.counters["d_pool_read_suppressed"],
                           D_POOL_MAX_AGE_S)
        return None

    def _d_gate_armed(self) -> bool:
        """Can the seats-vs-pool gate refuse anything at all, right now?

        FIX 5 (round 5).  THE READ IS AN ARGUMENT, AND PYTHON EVALUATES
        ARGUMENTS FIRST.  Both admission paths used to call
        ``_d_token_budget_blocks(rid, est, await self._d_pool_reading())``, so
        the `/server_info` round trip happened BEFORE the two cheap guards
        inside that method could decline to use it.  Consequence measured on
        the desk: ``--d-admit-max-tokens 0``, documented in ``front.py`` and
        ``launcher.py`` as "0 disables the gate", still paid a full
        `/server_info` GET per admission decision -- and so did the gate's own
        never-starve exit, which is the path taken at every epoch's FIRST
        admission and at every unblock, i.e. exactly inside the 0.05 s window
        the admitter races ``controller()``'s D->P arm in.

        The two terms are the disabling conditions of
        :meth:`_d_token_budget_blocks`, kept there as well: this method is the
        cheap PRE-check that decides whether to pay for a reading, never a
        second copy of the decision.  Synchronous by design -- a guard that
        may await is a guard that can cost what it exists to avoid.
        """
        return self.d_admit_max_tokens != 0 and bool(self._d_seats_live)

    async def _d_reading_if_armed(self) -> Optional[Dict[str, Any]]:
        """The reading, or ``None`` without a round trip when the gate is off.

        ONE helper rather than the same conditional at both call sites, so the
        BATCH admitter and the SHORT path cannot drift apart on it.
        """
        if not self._d_gate_armed():
            return None
        return await self._d_pool_reading()

    def _d_token_budget_blocks(self, rid: str, est_tokens: int,
                               reading: Optional[Dict[str, Any]],
                               realised_tokens: int = 0) -> bool:
        """Would granting this request a seat ask D for a store read it cannot
        issue?

        THE AGGREGATE SEATS-VS-POOL COUPLING (FIX 4a round 1; the L-line and
        its denominators round 3; THE QUANTITY, round 4).  ``--d-bs`` seats
        are a COUNT and the store read is a ROW budget, so six seats times one
        prompt can exceed the pool the group has to prefetch into -- and a
        request whose prefix IS in the store then prices as wholly uncached
        (`#915 PREFETCH REFUSED reason=vote_negative`, `cached_tokens=0`) and
        is W31-refused for a reason that is not its own.  EFFECTIVE SEATS ARE
        THEREFORE ``min(d_bs, what the pool can still prefetch)``: the head of
        ``_ready_for_d`` WAITS in arrival order (law 2 is "oldest first", not
        "six at once"), IT IS NEVER SKIPPED OVER -- the admitter peeks and
        `continue`s, so no younger request can take the seat the head is
        waiting for -- and its seat is not taken while its store read cannot
        be issued.

        THE TWO TERMS ARE D'S OWN, in the order D applies them:

        * ALLOC -- ``need <= available``.  ``available`` is D's
          ``mem_pool_host.available_size()``, the reading that refused boot
          weg2sc1 (5418 rows against need 8629); it is the only term that
          sees rows retained by earlier requests, which is precisely what a
          front-local admission tally can never see.
        * RATE -- ``occupied < limit``, D's ``prefetch_rate_limited``.

        Both are charged with ``_d_charged_since(reading["t"])``: the grants
        this front made after D sampled, which its reading cannot contain.

        THE RESIDUAL WINDOW, NAMED.  A grant made BEFORE the read was issued
        whose prefetch had not yet registered when D sampled is in neither
        term.  That window is D's intake latency (front resolve -> client POST
        -> tokenizer -> scheduler intake), it is bounded by the age bound
        above, and its only effect is that ONE request may still meet D's own
        #915 gate -- the behaviour that existed before this coupling.  The
        gate can therefore be optimistic by at most one in-flight grant and is
        never pessimistic about rows that exist.

        False whenever no seat is in use (a sole request is never starved by a
        bound its own size -- the truly oversized case is CARRIER-EXCEEDS's,
        R-3), whenever the ceiling flag is 0, whenever there is no reading
        (named above), or whenever it fits.

        L-LINE ``WEG2 D-SEAT-WAIT`` with every denominator named and its
        PROVENANCE on the line: ``need`` carries ``source=realised|estimate``
        (:func:`d_seat_need`), ``available`` is what D last reported its pool
        could still allocate MINUS the grants made since, ``limit`` is D's own
        prefetch capacity, and ``reading_age_s`` says how old the numbers are.
        Rate-limited to one line per newly-held rid, then every 200th pass,
        and the SUPPRESSED count rides on the line (``held_passes``), so an
        absence of lines is readable as an absence of waits rather than as a
        silenced emitter.

        FIX 7 (round 7), TWO CORRECTIONS TO THE LEFT-HAND SIDE.  Round 4
        fixed the RIGHT-hand side of this comparison (``available`` became
        D's own #915 reading rather than a front tally) and left the left one
        alone; boot weg2sc3 measured what that costs.

        * ``need`` is the REALISED count once leg 1 has answered.  The 1.71x
          arrival estimate (15,047 for a realised 8,865) made two requests
          charge 30,094 against a 27,466 limit, so effective concurrency was
          2 of 6 -- the estimate was the binding constraint, not the pool.
        * ``available`` CLAMPS AT 0.  It read ``-1663`` on metal, which is
          ``reading["available"] - charged_since_reading`` extrapolated past
          the reading it is anchored to.  A negative availability is not a
          physical quantity; the honest statement is "none, and the
          extrapolation says we are ``over_charged`` beyond that", and the
          over-charge is PRINTED rather than folded into a sign.

        THIS IS THE HONEST INTERIM.  The wall itself is the #974 host-pool
        bound; it lifts when the shared-ring carrier gives D a windowed store
        read (WEG2_CARRIER_SPEC_0907 Amendment 2), and this gate then never
        fires, with no code to remove.
        """
        if self.d_admit_max_tokens == 0 or not self._d_seats_live:
            # The same two terms as :meth:`_d_gate_armed`, which is what the
            # call sites consult BEFORE paying for a reading (FIX 5).  Kept
            # here too because this method is called directly, and because a
            # guard whose only copy lives at the call site is a guard the next
            # call site forgets.
            self._d_token_hold_rid = None
            return False
        if reading is None:
            self._d_token_hold_rid = None
            return False
        need, need_source = d_seat_need(est_tokens, realised_tokens)
        charged = self._d_charged_since(reading["t"])
        raw_available = int(reading["available"]) - charged
        limit = int(reading["limit"])
        occupied = int(reading["occupied"]) + charged
        if self.d_admit_max_tokens is not None:
            # The operator ceiling bounds the SAME quantity, never replaces it.
            raw_available = min(raw_available, self.d_admit_max_tokens - occupied)
        # THE CLAMP, and the over-charge kept rather than swallowed by it: the
        # comparison below wants "rows this request may have", which cannot be
        # less than none, while the diagnosis wants "by how much the
        # extrapolation has overshot the reading it hangs on".  Two questions,
        # so two numbers -- folding them into one signed integer is what put
        # `available=-1663` on a line whose reader had no way to tell an
        # exhausted pool from a stale anchor.
        available = max(0, raw_available)
        over_charged = max(0, -raw_available)
        if need <= available and occupied < limit:
            self._d_token_hold_rid = None
            return False
        self.counters["d_admit_token_held"] += 1
        held = self.counters["d_admit_token_held"]
        if self._d_token_hold_rid != rid or held % 200 == 0:
            self._d_token_hold_rid = rid
            logger.info(
                "WEG2 D-SEAT-WAIT rid=%s need=%d source=%s available=%d "
                "over_charged=%d limit=%d occupied=%d "
                "charged_since_reading=%d reading_age_s=%.2f inflight_tokens=%d "
                "seats_free=%d held_passes=%d (group D's own #915 terms, read from "
                "/server_info; the store read for this request cannot be issued yet, "
                "so it keeps the head of the arrival queue and takes no seat -- "
                "admitting it here is the #915 vote_negative shape that mis-prices "
                "the X gate. need names its own provenance; available is clamped at "
                "0 and the extrapolation's overshoot rides beside it)",
                rid, need, need_source, available, over_charged, limit, occupied,
                charged, max(0.0, time.time() - reading["t"]),
                self._d_inflight_tokens(), self.seats_free(), held,
            )
        return True

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
        # #1285: the trace config is the ONLY way to learn whether a request
        # went out on a pooled connection or a fresh one; it adds two awaits
        # per request and no behaviour.  The connector stays aiohttp's default
        # ON PURPOSE in this commit -- changing pooling and instrumenting it in
        # the same step would leave the boot unable to say which of the two
        # moved the symptom.
        self.session = ClientSession(timeout=ClientTimeout(total=3600),
                                     connector=SportTCPConnector(),
                                     trace_configs=[make_rpc_trace_config()])
        app["controller"] = asyncio.create_task(self.controller())
        app["admitter"] = asyncio.create_task(self.d_admitter())
        app["health"] = asyncio.create_task(self.health_poller())
        app["corridor"] = asyncio.create_task(self.corridor_sampler())
        # #1262 tier 3 -- the deadman's third signal, see flip_stall_check.
        app["flip_stall"] = asyncio.create_task(self.flip_stall_sampler())
        app["host_watermark"] = asyncio.create_task(self.host_watermark_sampler())
        logger.info("WEG2-FRONT up tag=%s awake=%s P=%s D=%s W=%.0f s (operator V1 fairness bound, 0 = off) "
                    "carrier_max_tokens=%d p_concurrency=%d d_bs=%d X=%d flip_min_work_tokens=%d "
                    "idle_layout=%s min_dwell_ms=%s drain_deadline_s=%.0f d_admit_max_tokens=%d "
                    "(derived_from=%s)",
                    self.tag, self.awake, self.groups["P"].url, self.groups["D"].url, self.w_s,
                    self.carrier_max_tokens, self.p_concurrency, self.d_bs, self.tp_prefill_max_tokens,
                    self.flip_min_work_tokens, self.idle_layout,
                    "derived" if self.min_dwell_ms is None else f"{self.min_dwell_ms:.0f}",
                    self.drain_deadline_s,
                    -1 if self.d_admit_max_tokens is None else self.d_admit_max_tokens,
                    "group D's own #915 reading (/server_info hicache_prefetch), "
                    "no operator ceiling" if self.d_admit_max_tokens is None
                    else "that reading under an operator ceiling")

    async def cleanup(self, app):
        for k in ("controller", "admitter", "health", "corridor", "flip_stall"):
            t = app.get(k)
            if t:
                t.cancel()
        if self.session:
            await self.session.close()

    # ------------------------------------------------------------------
    # #1264 fix 2b (1): the stage carries its own timestamp, STRUCTURALLY.
    #
    # A property rather than "remember to stamp it at each assignment": there
    # are seven assignment sites today and the next stage anyone inserts would
    # otherwise ship unstamped, which is the same shape as the defect -- an
    # instrument that quietly reports a stale value as a current one. Written
    # through the setter, the two can never disagree, and `age_s` is a fact
    # about the WRITE, not about whoever remembered to record one.
    # ------------------------------------------------------------------
    @property
    def _flip_stage(self) -> str:
        return self._flip_stage_value

    @_flip_stage.setter
    def _flip_stage(self, stage: str) -> None:
        self._flip_stage_value = stage
        self._flip_stage_t = time.monotonic()

    def flip_stage_age_s(self, now: Optional[float] = None) -> Optional[float]:
        """Seconds since the last stage assignment, on the MONOTONIC clock.

        Monotonic because this is a duration and the wall clock can step; the
        stall line's own `elapsed` uses `time.time()` for a different reason
        (it is compared against the flip's start, which the flip log records in
        wall time). ``None`` when nothing was ever stamped.
        """
        t = getattr(self, "_flip_stage_t", None)
        if t is None:
            return None
        return (time.monotonic() if now is None else now) - t

    async def host_watermark_sampler(self) -> None:
        """#1269 / standing order 2026-09-08: THE HOST THRESHOLD IS NEVER CROSSED.

        Reads the cgroup's own ``memory.current`` -- the quantity the reaper
        watches -- and when it crosses the HARD BOUND (reap watermark minus the
        named margin) performs a CONTROLLED TEARDOWN down the same path a
        killer takes (:meth:`do_stop`).  Never "accept the risk", never a
        silent restart.

        The order exists because that offer was made and taken: base weg2sb4
        stood green at 95.92 -> 96.36 -> 96.47 GiB against a 95.90 GiB reap
        mark, growth entirely anon and entirely at idle, and the box went into
        the OOM anyway.  User, verbatim: "kein uebertreten mehr der schwelle.
        fuehrt nur zum absturz."

        The margin is NAMED, not a cushion (host_ledger.resolve_margin): the
        flip transient this form actually spends plus the idle anon drift it
        actually accumulates over the planned window.  Once a boot carries
        WEG2-IDLE-CENSUS the drift term comes from that line instead of the
        sb4 default.
        """
        margin = host_ledger.resolve_margin(
            drift_mib_per_min=self._observed_anon_drift_mib_per_min
        )
        logger.info("%s", host_ledger.watermark_provenance(margin))
        while True:
            try:
                cg = host_ledger.read_cgroup()
                current = (cg or {}).get("current")
                if current:
                    # The cgroup that reaps is SHARED with the Claude sessions
                    # and their desk work, so the verdict carries the split:
                    # a breach the operator's own pytest run caused is named,
                    # not charged to the boot.
                    # #1269 fix 3: test NON-RECLAIMABLE pressure, not raw
                    # memory.current -- the raw reading counts page cache the
                    # kernel drops before it OOMs, and on weg2sb5b that refused
                    # a boot 28 GiB below danger.  The split comes from the
                    # PRE-BOOT anon baseline; `cgroup anon - sum(RssAnon)` is
                    # not subtractable and printed foreign=-30.91 on that boot.
                    pr = host_ledger.read_cgroup_pressure()
                    verdict = host_ledger.watermark_breach_verdict(
                        int(current),
                        margin=margin,
                        nonreclaim_gib=pr.get("nonreclaim_gib"),
                        file_reclaimable_gib=pr.get("file_reclaimable_gib"),
                        cgroup_anon_bytes=host_ledger.read_cgroup_anon_bytes(),
                        anon_preboot_bytes=self._anon_preboot_bytes,
                        composition={
                            k[:-4]: v for k, v in pr.items()
                            if k.endswith("_gib") and k != "nonreclaim_gib"
                        },
                    )
                    if verdict is not None:
                        self.counters["host_watermark_breach"] += 1
                        logger.error("%s", verdict)
                        self.do_stop("W22 Weg2HostWatermarkBreached", verdict)
                        return
            except Exception:  # noqa: BLE001 - a guard may never kill the front
                logger.exception("host_watermark_sampler")
            await asyncio.sleep(self.host_watermark_period_s)

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
            # Same instrument and band as the WEG2-CORRIDOR log line: a reader
            # of this dict must not have to guess which "free" it holds.  Both
            # come from the SAME source the line uses -- the instrument from
            # the last actual sample (never a nominal constant, which would
            # claim v2 on a rig whose samples had to fall back), the band from
            # the corridor guard, read now.
            "corridor_instrument": self.corridor_instrument,
            "corridor_band_mib": list(corridor_band_mib()),
            # #1257c: the DERIVED per-card floor and both of its terms, so a
            # machine reader gets the same provenance the log line carries and
            # never has to assume the rig-wide band above still describes the
            # verdict.  ``corridor_band_mib`` stays for compatibility and is
            # the pre-#1257c rig-wide band, which is NOT what a verdict is
            # taken against when a card's transient is measured.
            "corridor_floor": dict(self.corridor_floors),
            "flips": self.flip_log[-20:],
            "dc_measured_d_mib": self.dc_measured_d,
            "uptime_s": round(time.time() - self.t0, 1),
            "fairness_w_s": self.w_s,
            # #1289: X AND ITS PROVENANCE TRAVEL TOGETHER. An acceptance that
            # reads only the number cannot tell the launcher's carried-in
            # constant from a value this boot measured -- which is exactly how
            # sb5f shipped `flip_s=3.79 (median of 30)` from a different
            # layout's flips and nobody noticed for a whole run.
            "x_tokens": self.tp_prefill_max_tokens,
            "flip_min_work_tokens": self.flip_min_work_tokens,
            "x_flip_s": self.x_flip_s_provenance(),
        }

    async def handle_passthrough_get(self, request: web.Request) -> web.Response:
        g = self.groups[self.awake]
        async with self.session.get(f"{g.url}{request.path_qs}") as r:
            body = await r.read()
            return web.Response(body=body, status=r.status, content_type=r.content_type)

    async def handle_passthrough_post(self, request: web.Request) -> web.Response:
        """Forward a NON-GENERATING POST to the awake group, verbatim.

        Q0-B: ``/v1/messages/count_tokens`` is the only member. It decodes
        nothing, so it takes no seat, opens no Pending, records no span and
        must not be able to provoke a flip -- routing it through
        ``handle_generate`` would do all four for a request that never
        generates a token. The group server answers it natively (measured:
        ``POST :30032/v1/messages/count_tokens`` -> 200).
        """
        g = self.groups[self.awake]
        payload = await request.read()
        async with self.session.post(
            f"{g.url}{request.path_qs}", data=payload,
            headers={"Content-Type": request.content_type or "application/json"},
        ) as r:
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
        # MF-3: the routing probe's OTHER half, taken here and nowhere else.
        # `price_remainder` already asked the span LRU how much of this prompt
        # D is MEASURED to hold, and subtracted it.  That difference is the
        # presence estimate, so MF-3's denominator costs no second probe.  It
        # must be captured HERE: leg 2 records this very text into the same
        # LRU, after which the probe would answer with this request's own
        # outcome.
        #
        # #1324 CORRECTION, and the old wording is replaced rather than left
        # standing because it named the defect as the design: it read "a
        # prefix P prefilled and WROTE THROUGH to the store". The LRU
        # witnessed the PREFILL and asserted the WRITE-THROUGH, which is the
        # 45,014-token divergence of boot weg2sn6s. The entries are now fed D's
        # own realised `cached_tokens`, so this term is a measurement.
        store_span = max(0, est_prompt - remainder)
        # #1324: the ROUTE-VERDICT must name WHAT VOUCHED for the credit it
        # routes on. `span_known=True` alone read as an assurance about the
        # store; these two fields say which reading it is and how big, so a
        # SHORT verdict can never again be traced back to a witness that only
        # ever saw a prefill.
        presence_src = "d_leg2_cached" if known else "none"
        if not known:
            self.counters["W22_Weg2SpanUnknownPricedFull"] += 1
        self.counters["requests"] += 1
        stream = bool(payload.get("stream"))
        exact = self.exact_tokens.get(hashlib.sha1(text.encode(errors="replace")).hexdigest())
        carrier_est = exact if exact else int(len(text) / CARRIER_CHARS_PER_TOKEN) + 1
        # #1290: ONE ROUTE VERDICT, TAKEN ONCE, ON BOTH BOUNDS TOGETHER.  The
        # branch below used to test the carrier ALONE and send anything over
        # it to a D single prefill -- including requests whose uncached extent
        # was 1.7x D's own prefill cap, which D then refused by construction.
        # The verdict names the two bases so a reader never has to work out
        # which number each bound was compared against.
        route = serviceable_route(remainder, carrier_est,
                                  self.tp_prefill_max_tokens,
                                  self.carrier_max_tokens,
                                  carrier_exact=exact is not None)
        # THE COMPARED NUMBER IS PRINTED (#1290 round 2). The CARRIER-EXCEEDS
        # line below printed `est_prompt=... exact=None > carrier_max=...`,
        # and NEITHER of those is the value the branch compares -- `est_prompt`
        # is the /3.0 total, `exact` is a cache miss, and `carrier_est` (the
        # /2.4 figure actually tested) did not appear at all. On sb5f that
        # rendered as `est_prompt=22169 exact=None > carrier_max=27466`, a
        # printed inequality that is FALSE as printed (22169 < 27466), and it
        # cost a reader a whole wrong causal chain -- "None was read as
        # exceeds". Nothing read None as a number; the line simply never
        # showed the number. Instrument-text-lies, class A.
        logger.info(
            "WEG2 ROUTE-VERDICT rid=%s verdict=%s uncached=%d (base for X=%d, "
            "what D must PREFILL, at CHARS_PER_TOKEN=%.1f minus the MEASURED "
            "cached-on-D presence) presence_span=%d presence_src=%s (#1324: the "
            "credit's witness -- d_leg2_cached is a realised cached_tokens "
            "reading from group D, never a prefill on P) carrier_est=%d src=%s "
            "(THE COMPARED VALUE for carrier_max=%d: the WHOLE prompt's KV "
            "through the host staging pool, at CARRIER_CHARS_PER_TOKEN=%.1f) "
            "est_prompt=%d chars=%d (#1290)",
            rid, route, remainder, self.tp_prefill_max_tokens, CHARS_PER_TOKEN,
            store_span, presence_src,
            carrier_est, "exact" if exact is not None else "estimate",
            self.carrier_max_tokens, CARRIER_CHARS_PER_TOKEN, est_prompt,
            len(text),
        )
        if route == "none":
            # TERMINAL AT ADMISSION, and 4xx because it is the request that
            # does not fit this server, not the server that failed.  sb5f
            # spent a median 22.0 s per request discovering this by round trip
            # and answered 503, which reads as "try again" for a condition
            # that cannot change.
            self.counters["W52_Weg2NoServiceableRoute"] += 1
            detail = (
                f"{NO_ROUTE_NAME} rid={rid}: no route can serve this request. "
                f"uncached={remainder} exceeds D's prefill cap X="
                f"{self.tp_prefill_max_tokens} (--tp-prefill-max-tokens), so D "
                f"cannot single-prefill it; carrier_est={carrier_est} exceeds "
                f"carrier_max={self.carrier_max_tokens}, so P cannot hand its "
                f"KV back to D either. Refused at admission rather than "
                f"re-offered: D's cap is static, so no retry can change this "
                f"answer. Send at most {min(self.tp_prefill_max_tokens, self.carrier_max_tokens)} "
                f"tokens, or raise --tp-prefill-max-tokens / the carrier bound."
            )
            logger.error("%s", detail)
            return web.json_response(
                {"error": detail, "uncached": remainder,
                 "x_tokens": self.tp_prefill_max_tokens,
                 "carrier_est": carrier_est,
                 "carrier_max": self.carrier_max_tokens},
                status=413)
        if route == "carrier_single":
            self.counters["route_carrier_exceeds"] += 1
            logger.warning("WEG2-ROUTE rid=%s CARRIER-EXCEEDS -> D single prefill carrier_est=%d (%s) > carrier_max=%d "
                           "est_prompt=%d exact=%s "
                           "(group D host staging pool bound: the store cannot be read into D for a prompt this long; "
                           "ONE prefill on D, no leg 1, no double prefill; uncached=%d vs X=%d, checked #1290)",
                           rid, carrier_est,
                           "exact" if exact is not None else "ESTIMATE from chars, never terminal",
                           self.carrier_max_tokens, est_prompt, exact,
                           remainder, self.tp_prefill_max_tokens)
            if self.awake == "D" and self.admit_d and self.state == "serving":
                seat = await self._acquire_short_seat(rid, carrier_est)
                if seat is not None:
                    self._log_admit(rid, source="short", t_arrive=time.time())
                    return await self.leg2(request, rid, payload, text, stream, pending=None,
                                           single_prefill=True, seat=seat)
            fut = asyncio.get_event_loop().create_future()
            p = Pending(rid, request.path, payload, text, time.time(), fut, est_prompt=est_prompt,
                        est_uncached=remainder, span_known=known,
                        skip_leg1=True, store_span_est=store_span)
            self.queue.append(p)
            try:
                await fut
            except Weg2Stop as e:
                return web.json_response({"error": str(e)}, status=503)
            except Exception as e:  # noqa: BLE001
                return web.json_response({"error": f"{type(e).__name__}: {e}"}, status=503)
            self._mark_posted(p)
            return await self.leg2(request, rid, payload, text, stream, pending=p, seat=p.seat)
        short_ok = route == "short"
        # L10 (C9): the front's X verdict, LABELLED as the estimate it is --
        # price_remainder is len(text)/3.0 minus an LRU prefix guess, with no
        # tokenizer at the front.  D re-derives the real extent after
        # match_prefix and refuses by name there (L9/W31).
        logger.info("WEG2 X-ROUTE rid=%s est_uncached=%d X=%d (ESTIMATE, front pricing, no tokenizer)",
                    rid, remainder, self.tp_prefill_max_tokens)
        if self.awake == "D" and self.admit_d and self.state == "serving" and short_ok:
            seat = await self._acquire_short_seat(rid, est_prompt)
            if seat is not None:
                self.counters["route_short"] += 1
                # #1324: `span_known=True` used to stand alone here and read as
                # an assurance about the store. The witness is named instead.
                logger.info("WEG2-ROUTE rid=%s SHORT -> D est_prompt=%d remainder=%d "
                            "presence_span=%d presence_src=%s",
                            rid, est_prompt, remainder, store_span, presence_src)
                self._log_admit(rid, source="short", t_arrive=time.time())
                return await self.leg2(request, rid, payload, text, stream, pending=None, seat=seat)
        # #1290: NAME THE P ROUTE. `route == "long"` is X < uncached <=
        # carrier -- the case P exists for -- and it was counted only as
        # "batch" before, which is why the sb5f census read LONG 0 and nobody
        # could tell "queued because P is awake" from "queued because it is
        # too long for D". Same queue, two reasons, now two counters.
        # ADDITIVE, not a rename: `route_batch` keeps counting the queue as it
        # always did (the #1246 carrier-floor census reads it over the whole
        # length axis), and `route_long` names the SUBSET that is queued
        # because it is too long for D rather than because P is awake.
        self.counters["route_batch"] += 1
        if route == "long":
            self.counters["route_long"] += 1
            logger.info(
                "WEG2-ROUTE rid=%s LONG -> P leg 1 (uncached=%d > X=%d, so D "
                "cannot prefill it; carrier_est=%d <= carrier_max=%d, so P's "
                "KV can come back to D) est_prompt=%d queue=%d",
                rid, remainder, self.tp_prefill_max_tokens, carrier_est,
                self.carrier_max_tokens, est_prompt, len(self.queue))
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
        p = Pending(rid, request.path, payload, text, time.time(), fut, est_prompt=est_prompt,
                        est_uncached=remainder, span_known=known,
                    store_span_est=store_span)
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

    async def _acquire_short_seat(self, rid: str, est_tokens: int = 0) -> Optional[Seat]:
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
        if self._d_token_budget_blocks(rid, est_tokens, await self._d_reading_if_armed()):
            # FIX 4a: the same aggregate bound the BATCH admitter obeys.  A
            # SHORT arrival that does not fit falls through to route BATCH
            # rather than overcommitting the staging pool -- the return
            # contract this method already has for a held gate.
            return None
        await self._d_seat.acquire()
        if not (self.awake == "D" and self.admit_d and self.state == "serving"):
            self._d_seat.release()
            return None
        return Seat(self, rid, "short", tokens=est_tokens)

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

        FIX 1 (round 1) -- THE SEAT IS ACQUIRED WHILE THE REQUEST IS STILL
        IN THE DEQUE, and the phase is re-checked after the acquire.  The
        guard above is at the TOP of the loop only; ``_d_seat.acquire()``
        below it blocks for the whole lifetime of a running decode, so the
        phase read there is arbitrarily stale.  Popping first and resolving
        after opened three holes at once, all of them the same window:

        * the resolved request POSTs its leg 2 into a group that is
          flipping or asleep, and ``leg2`` re-registers it in
          ``D.outstanding`` in the middle of ``drain(D)`` -- the drain can
          then never terminate (W1, three times W2 STOP);
        * ``do_stop`` answers ``self.queue + self._ready_for_d``
          (:meth:`do_stop`) and the popped request is in NEITHER, so a STOP
          raised during the acquire never reaches it;
        * popping the LAST entry runs ``_sync_batch_gate`` and OPENS the
          batch gate, so a SHORT arrival may take the seat this admitter is
          queued for -- R-16 inverted.

        Peeking closes all three: the entry stays reachable, the gate stays
        closed and the C6/R-2 idle-guard term stays true for the whole wait,
        and the pop happens only in the same synchronous step that resolves
        the future.  This is the batch-side counterpart of the check
        :meth:`_acquire_short_seat` already performs after ITS acquire.
        """
        while True:
            await asyncio.sleep(0.05)
            try:
                if self.state != "serving" or self.awake != "D":
                    continue
                if not self._ready_for_d:
                    continue
                p = self._ready_for_d[0]
                if p.fut.done():
                    # Already resolved, failed or cancelled (leg 1 error, an
                    # abort, a STOP): no seat is spent on it.
                    self._ready_for_d.popleft()
                    self._sync_batch_gate()
                    self.counters["d_admit_skipped_done"] += 1
                    continue
                # FIX 7: the realised count when leg 1 has answered for this
                # rid, the arrival estimate only for a cold one -- and the
                # SAME number is charged to the seat below, so the gate and
                # `_d_charged_since` cannot price one request two ways.
                if self._d_token_budget_blocks(p.rid, p.est_prompt,
                                               await self._d_reading_if_armed(),
                                               p.leg1_prompt_tokens):
                    # FIX 4a: the seat is not even reached -- taking one and
                    # holding it while the tokens are unavailable would block
                    # the refill the budget is waiting for.  `continue` and
                    # not a pop: the head STAYS the head (law 2), so a younger
                    # request that would fit cannot be admitted past it.
                    continue
                await self._d_seat.acquire()
                if not (self.awake == "D" and self.admit_d and self.state == "serving"):
                    # The phase moved while this admitter was queued behind a
                    # running decode.  Give the seat back and leave the
                    # request where it is: the deque head is still the oldest
                    # (law 2) and is still answerable by do_stop.
                    self._d_seat.release()
                    self.counters["d_admit_phase_moved"] += 1
                    continue
                if not self._ready_for_d or self._ready_for_d[0] is not p or p.fut.done():
                    # do_stop cleared the deque, or answered this request,
                    # while the seat was being waited for.
                    self._d_seat.release()
                    self.counters["d_admit_skipped_done"] += 1
                    continue
                self._ready_for_d.popleft()
                self._sync_batch_gate()
                p.seat = Seat(self, p.rid, "batch",
                              tokens=d_seat_need(p.est_prompt, p.leg1_prompt_tokens)[0])
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
                self._note_p_prefix_reuse(p, ct)
                # #1324: NO PRESENCE RECORD HERE. This site used to call
                # `self.spans.record(p.text, pt)`, i.e. it credited the span
                # P had just PREFILLED as a span D could read back -- and P's
                # write-through is asynchronous, so at this instant the store
                # may hold none of it. Measured on weg2sn6s: recorded 109,132
                # at 15:21:49, D found 53,247 at 15:22:35, the repeat routed
                # SHORT on the difference and died W31 -> W50.
                #
                # Nothing replaces it, because P HAS no presence witness to
                # offer: under W38 (`Weg2CarrierlessPpStoreRead`) group P
                # reads no store at all, so its `cached_tokens` speaks only
                # for P's own device tier, and its `prompt_tokens` speaks for
                # a prefill, not for a landing. The witness is D's own
                # `cached_tokens` on leg 2, recorded there.
                #
                # `_note_exact` DOES stay: `prompt_tokens` is a TOKENISATION
                # fact (this text is 109,132 tokens), it feeds `carrier_est`,
                # and it was never wrong -- `carrier_est=109132 src=exact` was
                # the one correct number on the sn6s route line.
                self._note_exact(p.text, pt)
                # #1317n THE POST-LEG-1 CARRIER BAND IS GONE. It compared
                # the realised prompt against what D's host tier could carry AS
                # ONE READ and sent leg 2 to a single prefill on D -- which,
                # through D's carrier-exceeds exemption, prefilled it over X.
                # D's L2 is now derived from `--max-kv-per-request`, so below
                # the cap the store carries the whole prompt in one prefetch
                # and there is nothing to correct here; above the cap the front
                # refuses at admission, before P's prefill is spent.
                g.served += 1
                logger.info("WEG2-SERVED group=P leg=1 rid=%s prompt_tokens=%d cached_tokens=%d wall=%.2fs epoch=%d",
                            p.rid, pt, ct, time.time() - t0, self.epoch)
        finally:
            g.outstanding.pop(p.rid, None)

    def _note_p_prefix_reuse(self, p: Pending, leg1_cached_tokens: int) -> None:
        """MF-3: price ONE P prefill against the prefix reuse W38 forgoes.

        THE COST THIS MAKES VISIBLE.  ``#1234 W38 Weg2CarrierlessPpStoreRead``
        refuses EVERY storage read on group P (scheduler.py, and it is the
        right refusal: without the #631 row carrier a prefetch completing on
        one PP rank and not another splits the geometry -- the W27 divergence
        that killed boot weg2sc1).  The consequence is that a multi-turn
        follow-up whose prefix left P's device tier is prefilled WHOLE again,
        which is the user's soft no-double-prefill law paying for a hard
        correctness refusal.  MF-3 orders that cost MEASURED until the
        PP0-authoritative materialisation (#968 form) removes it, so that it
        is a number in the log rather than a sentence in a postmortem.

        WHAT THE THREE TERMS MEASURE, each with its instrument:

        * ``prefix_tokens_available_in_store`` -- the front's OWN ROUTING
          PROBE (:func:`price_remainder` over :class:`SpanLRU`), captured at
          arrival in ``Pending.store_span_est``.  It is an ESTIMATE at TEXT
          granularity: the longest common prefix with a prompt this front saw
          realised, scaled by that prompt's MEASURED cached-on-D share.
          #1324 CHANGED WHAT THAT SHARE IS, and this term's meaning with it:
          the LRU used to be fed any realised ``prompt_tokens``, so this
          counter measured "a prefix somebody PREFILLED" and asserted the
          write-through; it is now fed D's own leg-2 ``cached_tokens``, so it
          measures "a prefix D was MEASURED to hold".  The instrument is
          unchanged; its denominator became honest, and it now reads LOWER on
          a first pass (no measurement yet) than it used to.  It is a
          LOWER bound in two named ways -- a prefix from before the LRU's
          window is invisible, and a W31 re-queue deliberately contributes 0
          (D's own refusal is evidence the prefix did NOT come back, and since
          #1324 that refusal also RETRACTS any stale credit for the text) --
          and it
          is an upper bound in one: it counts the text prefix, not the store's
          page keys, so a prefix shorter than one page cannot actually be read
          back.  It is NOT a store key probe; the front has no tokenizer and
          no store index, and inventing one for an instrument would be the
          second bookkeeping this tree deletes on sight.
        * ``prefix_tokens_reused`` -- MEASURED, never assumed: P's own leg-1
          ``cached_tokens``.  Under W38 this can only come from P's DEVICE
          tier (its radix tree, fed by P's own prefills in this epoch); the
          store contributes nothing by construction.  It is written as a
          measurement precisely so the day the carrier arrives and the read
          re-arms, this line moves on its own instead of lying.
        * ``forgone_tokens`` -- ``available - reused``, floored at 0: prefix
          tokens the store held, P's device tier did not, and P therefore
          recomputed.  That is the double prefill, priced.
        """
        self.counters["p_prefill_requests"] += 1
        self.counters["p_prefix_tokens_in_store"] += max(0, int(p.store_span_est))
        self.counters["p_prefix_tokens_reused"] += max(0, int(leg1_cached_tokens))

    def _note_exact(self, text: str, prompt_tokens: int) -> None:
        if prompt_tokens <= 0:
            return
        if len(self.exact_tokens) >= SPAN_LRU:
            self.exact_tokens.pop(next(iter(self.exact_tokens)))
        self.exact_tokens[hashlib.sha1(text.encode(errors="replace")).hexdigest()] = int(prompt_tokens)

    def _leg2_verdict(self, pt: int, ct: int, priced: bool, pending: Optional[Pending],
                      single_prefill: bool, stream: bool, rid: str,
                      x_inband: bool = False) -> str:
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
        if x_inband:
            # FIX 2 (round 1): D refused this streamed request by name after
            # the first byte.  It carries no usage chunk, so the pre-existing
            # code priced it as W28 "unpriced" -- the right name for a
            # missing price, the wrong name for a refusal that has one.  The
            # caller has already counted W50_stream_served.
            return "W50_stream_served"
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
        # #1289 round 2: THE SOLO WITNESS for the r_D sample below. `r_D` is
        # D's PREFILL rate -- tokens over the wall of a prefill D ran ALONE --
        # so what qualifies a sample is CONCURRENCY, not a pricing verdict.
        # `_d_admissions` counts every arrival at D; if it has not moved and
        # D's outstanding set held only this rid at both ends, nothing else
        # was in flight and the wall is this group's own throughput.
        self._d_admissions += 1
        _solo_adm0 = self._d_admissions
        _solo_entry = len(g.outstanding) == 1
        t0 = time.time()
        if pending is not None and pending.skip_leg1:
            single_prefill = True
        if (stream and pending is not None and request.path.startswith("/v1/")
                and request.path != "/v1/messages"):
            # 1j finding 2: a STREAMED leg 2 is priced like a non-streamed one.
            # OpenAI's stream_options.include_usage makes D append one usage
            # chunk (empty choices) -- standard, and the only post-hoc price.
            #
            # Q0-B: EXCLUDING /v1/messages is not an omission. `stream_options`
            # is an OpenAI field; the Anthropic wire always carries usage
            # (message_start / message_delta), so nothing has to be asked for,
            # and injecting an unknown top-level key into a Messages body risks
            # a 400 from the very endpoint this ticket exists to reach.
            payload = dict(payload)
            so = dict(payload.get("stream_options") or {})
            so["include_usage"] = True
            payload["stream_options"] = so
        try:
            async with self.session.post(f"{g.url}{request.path}", json=payload) as r:
                # FIX 2 (round 1): LAW 4's RE-ROUTE IS DECIDED BEFORE THE
                # RESPONSE IS COMMITTED, ON BOTH WIRE SHAPES.  The check used
                # to sit below the `if stream:` branch, which returns; so for
                # every streamed request -- the normal shape for OpenAI chat
                # completions under agent load, and the gate is ON for every
                # boot because the launcher always passes
                # `--tp-prefill-max-tokens` to argv_d -- W31 was never
                # counted, `_requeue_after_x_refusal` was never called, W35
                # could never apply, and the client got D's refusal instead of
                # being prefilled by P.  Both shapes are handled here, above
                # `resp.prepare()`, because nothing can be re-routed after the
                # first byte has been committed to the client.
                early_body: Optional[bytes] = None
                first_chunk: Optional[bytes] = None
                if r.status != 200:
                    # Shape 1: the non-stream abort, `HTTPException(503)`.
                    early_body = await r.read()
                    if is_x_refusal(r.status, early_body.decode(errors="replace")):
                        g.outstanding.pop(rid, None)
                        return await self._requeue_after_x_refusal(
                            request, rid, payload, text, stream, pending, seat, early_body
                        )
                elif stream:
                    # Shape 2: the IN-BAND abort on a 200.  A request refused
                    # at admission is aborted before it decodes anything, so
                    # the refusal IS the first chunk -- reading it costs one
                    # chunk of head-of-line latency (the client sees nothing
                    # before the first token anyway) and buys the re-route.
                    #
                    # Q0-B: ...on the OPENAI wire. The Anthropic wire always
                    # sends `message_start` first, so the refusal is never the
                    # first chunk there and this test saw only the envelope
                    # (measured, boot weg2sn5m). Read to the first CONTENT
                    # event instead, bounded; everything read is forwarded
                    # verbatim below.
                    if request.path == "/v1/messages":
                        first_chunk, _refused = await _anthropic_refusal_lookahead(r)
                    else:
                        first_chunk = await _first_stream_chunk(r)
                        _refused = first_chunk is not None and x_refusal_marker_in(
                            first_chunk.decode(errors="replace")
                        )
                    if _refused:
                        self.counters["W50_stream_inband_requeued"] += 1
                        g.outstanding.pop(rid, None)
                        return await self._requeue_after_x_refusal(
                            request, rid, payload, text, stream, pending, seat, first_chunk
                        )
                if stream:
                    resp = web.StreamResponse(status=r.status)
                    resp.content_type = r.content_type
                    await resp.prepare(request)
                    tail = bytearray()
                    # Q0-B: the Anthropic prompt count arrives ONCE, at the
                    # head of the stream, and the bounded tail below trims the
                    # head away on any long answer. Accumulate as we forward.
                    anth = AnthropicStreamUsage() if request.path == "/v1/messages" else None

                    async def _push(chunk: bytes) -> None:
                        await resp.write(chunk)
                        if anth is not None:
                            anth.feed(chunk)
                        tail.extend(chunk)
                        if len(tail) > 262144:
                            del tail[:-131072]

                    if early_body is not None:
                        # A non-200 whose body this method already consumed
                        # for the refusal test: forward it verbatim.
                        await _push(early_body)
                    else:
                        if first_chunk is not None:
                            await _push(first_chunk)
                        async for chunk in r.content.iter_any():
                            await _push(chunk)
                    await resp.write_eof()
                    g.served += 1
                    if anth is not None:
                        pt, ct, comp, priced = anth.result()
                    else:
                        pt, ct, comp, priced = usage_of_stream_tail(bytes(tail))
                    x_inband = x_refusal_marker_in(bytes(tail).decode(errors="replace"))
                    if x_inband:
                        # The W16 precedent's counterpart (finding 2): the
                        # refusal arrived AFTER the first byte, so the
                        # re-route is impossible.  Counted BY NAME rather
                        # than landing in W28 as an unpriced stream.
                        self.counters["W50_Weg2TpPrefillExceeded"] += 1
                        self.counters["W50_stream_served"] += 1
                        logger.error(
                            "W50 Weg2TpPrefillExceeded rid=%s (STREAM, served): D refused this request "
                            "by name after the first byte -- re-route impossible, counted by name",
                            rid,
                        )
                    verdict = self._leg2_verdict(pt, ct, priced, pending, single_prefill, True, rid,
                                                 x_inband=x_inband)
                    if pt:
                        # #1324: the PRESENCE witness is D's own cached share,
                        # not the prompt length. `pt` still feeds the
                        # tokenisation fact (`carrier_est`); `ct` is what D
                        # did not have to prefill, and it is the only measured
                        # answer to "can D serve this text back".
                        self.spans.record_presence(text, ct)
                        self._note_exact(text, pt)
                    dterms = await self._draft_terms(g, None)
                    logger.info("WEG2-SERVED group=D leg=2 rid=%s stream=1 status=%d prompt_tokens=%d cached_tokens=%d completion_tokens=%d "
                                "uncached=%d verdict=%s priced=%s wall=%.2fs epoch=%d draft_pages=%d draft_miss=%d accept_len=%.3f accept_src=%s",
                                rid, r.status, pt, ct, comp, max(0, pt - ct), verdict, priced, time.time() - t0, self.epoch,
                                dterms["draft_pages"], dterms["draft_miss"], dterms["accept_len"], dterms["accept_src"])
                    if pending is not None and ct > 0:
                        self.counters["cross_group_prefix_hits"] += 1
                    return resp
                # C11/C12 -- a W31 that came back from D's own gate, where
                # the UNCACHED EXTENT IS REAL (after match_prefix), has
                # already been re-routed above; law 4 says such a request is
                # prefilled by P, so it re-joins route BATCH and is never
                # re-offered to D a third time (W35).
                body = early_body if early_body is not None else await r.read()
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
                # #1271 (b): r_D SAMPLE, and ONLY from a prefill D ran ALONE.
                # A concurrent leg-2 wall would be a latency and must never
                # enter this deque (#1271 (a)).
                #
                # #1289 ROUND 2 -- THE GATE TESTED THE WRONG PROPERTY, and
                # boot weg2sb5g measured the consequence: `r_d` stayed 0
                # across 64 flips and 84 admitted D prefills, so
                # `resolve_x_live` starved at its first line and X never
                # re-solved (`WEG2 X NO-SOLVE: no r_d sample yet ... have
                # r_d=0 r_p=1 flip_s=2`) even though the flip_s sampler this
                # ticket fixed was working.
                #
                # The old gate admitted the verdicts `single_prefill` and
                # `short_mispriced` and nothing else. Both are EXCEPTIONAL:
                # the first is route CARRIER-EXCEEDS, the second a SHORT the
                # front under-priced. The ORDINARY well-priced SHORT -- which
                # is a D prefill and is routinely the only thing on D --
                # returns `serve` (`_leg2_verdict`), and `serve` was excluded.
                # sb5g's census: `verdict=serve` 24, `verdict=short` 20, and
                # ZERO of either admitted verdict. The gate was also
                # incoherent: `short_mispriced` is a seated SHORT running at
                # exactly the same concurrency as `serve`, so the old rule
                # admitted and excluded the same physical situation depending
                # on how the front had priced it.
                #
                # CONCURRENCY IS THE PROPERTY, so measure concurrency. The
                # witness is exact rather than a snapshot: `_d_admissions`
                # not moving rules out an arrival that came and went inside
                # this window, which `len(outstanding)` at two instants
                # cannot.
                _unc = max(0, pt - ct)
                _w = time.time() - t0
                _solo = (_solo_entry
                         and self._d_admissions == _solo_adm0
                         and len(g.outstanding) == 1)
                if _solo and _unc > 0 and _w > 0:
                    self._x_r_d_src = f"solo leg2 verdict={verdict}"
                    self.note_x_sample("r_d", _unc / _w)
                elif _unc > 0 and _w > 0:
                    self.counters["r_d_skipped_concurrent"] += 1
                if pt:
                    # #1324, as on the streamed branch above: `ct` is the
                    # measured presence, `pt` the tokenisation fact. On a W31
                    # leg 2 this deliberately records D's SMALL reading and
                    # thereby RETRACTS any larger stale credit for this text
                    # -- D's own refusal is the strongest evidence the prefix
                    # did not come back.
                    self.spans.record_presence(text, ct)
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
        self.counters["W50_Weg2TpPrefillExceeded"] += 1
        # #1290/#1291/#1296: WHERE THE TERMINAL BELONGS -- AND WHERE IT DOES NOT.
        #
        # Two refusals live in this function and they are NOT the same kind
        # of claim:
        #
        #   W52 (below, FIRST refusal): STRUCTURAL. `carrier_est >
        #   carrier_max` means P's KV can never be read back into D at all,
        #   whatever happens next. No second pass can move it, so it is
        #   terminal on n=1 and stays that way.
        #
        #   W53 (SECOND refusal only): OBSERVATIONAL. "D priced the whole
        #   prompt again, so nothing came back" is a statement about what had
        #   landed AT THAT INSTANT. It is not a proof that nothing ever will.
        #
        # #1291 made W53 terminal on the FIRST refusal. Boot weg2sb5h refutes
        # that premise on the metal. All 13 W53s fired at `requeue_n=1`; the
        # 13 rids that were re-offered instead (the prose class, where the
        # char estimate happened to fall the other side of the comparand) are
        # the counterfactual #1291 never had:
        #
        #   weg2-28-259  leg 1 pt=18559 ct=0 | offer 1 uncached=18559 REFUSED
        #                -> re-offer -> offer 2 cached_tokens=18557 uncached=2
        #                   status=200 verdict=serve
        #   weg2-12-235  leg 1 pt=16522 ct=0 | offer 1 uncached=16522 REFUSED
        #                -> re-offer -> offer 2 cached_tokens=8190 (PARTIAL)
        #
        # 2 of those 13 re-offers paid, and one of them is the ONLY 200 that
        # population produced. `d_extent == the measured whole prompt` at
        # offer 1 is therefore indistinguishable, BY EXTENT ALONE, from the
        # f2a/f2b/t9c case the slice-A suite already protects: a store read
        # that simply had not LANDED yet. The read lands BETWEEN the two
        # offers, so no quantity available at the first refusal separates
        # them -- D's witness census for this boot is `state=unprobed` 186 /
        # `state=cold` 9 with `X-DEFER` 0, i.e. there is no read-state signal
        # to gate on either. (Design item: this terminal wants a READ-STATE
        # witness, not an extent. SECTION 1ax-b.)
        #
        # #1291's own justification carries the same refutation: the weg2sb5g
        # figures it quotes are "3 served out of 53 (natural 1/27, salad
        # 2/26)" -- three requests that exist only BECAUSE the re-offer ran,
        # and that a first-refusal terminal deletes. Salad pays at 2/26
        # there, so this is not a prose-only effect.
        #
        # So the terminal MOVES TO THE SECOND REFUSAL, where W35 already
        # stands, and W35's bare 503 becomes this named, measured 413 --
        # which is what #1291 actually wanted (name it, 4xx not 503) minus
        # the lap it should never have skipped. Cost, stated honestly: the
        # salad class spends its second P prefill again (13 on sb5h).
        # Benefit: the re-offer that sometimes pays is no longer deleted
        # before it is placed.
        #
        # THE COMPARAND IS P'S MEASURED COUNT, NOT THE FRONT'S ESTIMATE
        # (#1296 round 1, which stands). `est_uncached` is
        # `len(text) / CHARS_PER_TOKEN` (3.0) and its ERROR CHANGES SIGN WITH
        # THE PROSE: on sb5h one 62k-char body measured 2.747-2.798
        # chars/token as token salad (13 rids, estimate too LOW) and
        # 3.355-3.883 as natural prose (13 rids, estimate too HIGH), 3.0
        # sitting in the gap -- so the IDENTICAL empty handback read as
        # "empty" for one class and "partial" for the other. P's realised
        # count for THIS request is already on this object
        # (`Pending.leg1_prompt_tokens`, written in `leg1`), so measuring
        # against a measurement costs no new bookkeeping and no second store.
        # Same law as #1290 round 2: only a MEASURED quantity may terminate.
        # D's refusal carries its half verbatim
        # (`scheduler.py::_weg2_answer_x_refusals`: "...extent after prefix
        # matching is {uncached}"); when it cannot be parsed the answer is
        # UNKNOWN and the plain W35 503 stands, never a refusal on a guess.
        d_extent = _d_refusal_extent(body)
        measured_whole = int(getattr(pending, "leg1_prompt_tokens", 0) or 0)
        handback_empty = (
            d_extent is not None
            and pending is not None
            and getattr(pending, "leg1_done", False)
            # 0 is UNKNOWN, never "the store returned nothing": route
            # CARRIER-EXCEEDS sets `leg1_done` WITHOUT running a leg 1
            # (`p.skip_leg1` in the drain), so this is the only field that
            # separates "P ran and nothing came back" from "P never ran".
            # That class keeps the plain W35 503 -- named in SECTION 1ax-b.
            and measured_whole > 0
            and d_extent >= measured_whole
        )

        # SAME RULE AS THE ROUTER: only a MEASURED count may terminate. D has
        # just answered, so `_note_exact` has this text's real prompt_tokens
        # -- which is exactly the case the router could not have. An estimate
        # here would refuse on the same ~25% conservatism.
        exact = self.exact_tokens.get(
            hashlib.sha1(text.encode(errors="replace")).hexdigest())
        carrier_est = exact if exact is not None else None
        if (self.carrier_max_tokens > 0 and carrier_est is not None
                and carrier_est > self.carrier_max_tokens):
            self.counters["W52_Weg2NoServiceableRoute"] += 1
            detail = (
                f"{NO_ROUTE_NAME} rid={rid}: D refused this request with "
                f"{X_REFUSAL_NAME} and a re-offer cannot change that answer -- "
                f"carrier_est={carrier_est} exceeds carrier_max="
                f"{self.carrier_max_tokens}, so a P prefill's KV cannot be read "
                f"back into D and D would see the same extent again. Refused on "
                f"the FIRST refusal (n={n}) instead of spending a full P prefill "
                f"to reach the same verdict (#1290). D said: "
                f"{body.decode(errors='replace')[:300]}"
            )
            logger.error("%s", detail)
            if seat is not None:
                seat.release(NO_ROUTE_NAME)
            return web.json_response(
                {"error": detail, "x_tokens": self.tp_prefill_max_tokens,
                 "carrier_est": carrier_est,
                 "carrier_max": self.carrier_max_tokens},
                status=413)
        logger.warning(
            "WEG2 X-REQUEUE rid=%s n=%d verdict=%s", rid, n,
            "requeue" if n <= 1 else ("W53" if handback_empty else "W35"))
        if n > 1:
            # W35 counts the POPULATION -- every rid D refused a second time
            # after a full P prefill. W53 is the SUBSET of those for which
            # D's own number shows the store handed nothing back. So
            # W53 <= W35 always, both denominators are readable straight off
            # the census, and each still has exactly ONE increment site.
            self.counters["W35_Weg2XReQueueLoop"] += 1
            if handback_empty:
                self.counters["W53_Weg2StoreHandbackFailed"] += 1
                detail = (
                    f"{HANDBACK_NAME} rid={rid}: group P completed leg 1 for "
                    f"this request TWICE and group D refused it with "
                    f"{X_REFUSAL_NAME} both times -- so the prefill P "
                    f"performed did not reach D, and the one re-offer that "
                    f"could have changed that is now spent. Refusing by name "
                    f"rather than a third pass. D's refusal: "
                    f"{body.decode(errors='replace')[:300]}. Front terms: "
                    f"front_X={self.tp_prefill_max_tokens} carrier_max="
                    f"{self.carrier_max_tokens} leg1_done=True requeue_n={n} "
                    f"d_extent={d_extent} leg1_prompt_tokens={measured_whole} "
                    f"(d_extent >= leg1_prompt_tokens, the MEASURED whole "
                    f"prompt, so the store handed back NOTHING). "
                    f"est_uncached={pending.est_uncached} is the front's char "
                    f"estimate at CHARS_PER_TOKEN={CHARS_PER_TOKEN}, PRINTED "
                    f"NOT COMPARED (#1296). front_X is THIS process's "
                    f"--tp-prefill-max-tokens; D enforces its own and the two "
                    f"disagreed for all of sb5h, so read the `X=` inside D's "
                    f"refusal above for the value that actually gated. W53 is "
                    f"a SUBSET of W35_Weg2XReQueueLoop, its population. If "
                    f"D's `uncached` above is the WHOLE prompt, the store did "
                    f"not hand P's pages back: check D's PHASE-PURITY STORE "
                    f"WITNESS state for this rid (`unprobed` = no read was "
                    f"ever issued) and the store size against the P pool."
                )
                logger.error("%s", detail)
                if seat is not None:
                    seat.release(HANDBACK_NAME)
                return web.json_response(
                    {"error": detail, "x_tokens": self.tp_prefill_max_tokens,
                     "est_uncached": pending.est_uncached,
                     "leg1_prompt_tokens": measured_whole,
                     "d_extent": d_extent,
                     "carrier_max": self.carrier_max_tokens,
                     "leg1_done": True},
                    status=413)
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
            # #1271 (c) FOLLOW-UP -- THE THIRD CONSTRUCTION SITE. #1271's own
            # commit body says `est_uncached` is "set at BOTH construction
            # sites"; there are THREE, and this one was missed. Since
            # `_flip_economics_ok` sums `est_uncached`, a rid re-queued here
            # contributed 0 to the backlog -- so a W31 re-queue arriving on an
            # otherwise empty queue could never clear the threshold and simply
            # sat until the drain deadline. Found by test_t9c, which hung for
            # 30 s waiting for a 200 that a held flip could never produce.
            #
            # AND THIS IS THE WORST SITE TO MISS: D refused this rid with W31
            # precisely BECAUSE its uncached extent after match_prefix exceeded
            # X, so the one Pending whose uncached work is provably large was
            # the one contributing zero.
            #
            # The value is the whole estimate, not a store-credited remainder,
            # for the same evidence the `store_span_est=0` note below gives:
            # the prefix the span LRU would price as resident demonstrably did
            # not come back on D. `span_known=False` says that is an estimate.
            _est = len(text) // int(CHARS_PER_TOKEN) + 1
            p = Pending(rid, request.path, payload, text, time.time(),
                        asyncio.get_event_loop().create_future(),
                        est_prompt=_est, est_uncached=_est, span_known=False)
            # MF-3: `store_span_est` stays 0 on this path ON PURPOSE, and the
            # reason is evidence, not caution: D has just refused this rid
            # with W31, i.e. its uncached extent AFTER match_prefix was larger
            # than X, so the prefix the span LRU would price as store-resident
            # demonstrably did not come back on D.  Pricing it here would
            # inflate the forgone-reuse figure with tokens no store read was
            # going to save.  The P-PREFIX-REUSE line is therefore a LOWER
            # bound, and says so.
            p.x_requeues = n
        else:
            p.fut = asyncio.get_event_loop().create_future()
            p.t_arrive = time.time()
        if seat is not None:
            seat.release("W50_requeue")
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
            # #1288: no bearer here either -- same loopback trust. On boot
            # weg2sb5f this read answered 401 to 507 of 509, so `accept_len`
            # fell to `accept_src=none` and both draft counters read 0 for
            # the whole boot, silently.
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
        """#1275: EVERY rpc carries the admin bearer token when one is set.

        NOT a security addition -- a LIVENESS one. `/flush_cache`,
        `/release_memory_occupation`, `/resume_memory_occupation` and
        `/abort_request` are all `@auth_level(ADMIN_OPTIONAL)`, and that level
        means "require the ADMIN key once one is configured", not "optional".
        They answer today only because no key is set. The moment the launcher
        passes `--admin-api-key` to the groups (which is what buys the live
        `/hicache/storage-backend/resize` lever), an unauthenticated front gets
        401 on its very next quiesce and the flip dies. So the token goes on
        every RPC, and `admin_key` is None exactly when the groups are unkeyed.
        """
        code, text, _ = await self._rpc_attempt(self.session, g, path, body, timeout)
        return code, text

    async def leg_rpc(self, g: Group, path: str, body: Optional[dict],
                      timeout: float) -> Tuple[int, str]:
        """:meth:`rpc` plus ONE retry on a FRESH connection (#1285).

        ONLY for the two gathered flip legs, and only for the one failure shape
        that provably never reached a handler: a connection-level error raised
        BEFORE the response line (``_rpc_attempt``'s ``retryable``).  A partial
        response is never retried -- its handler ran.

        THE RETRY IS SAFE BECAUSE THE HANDLER MAKES IT SAFE, not because the
        legs are idempotent.  They are not: ``resume_memory_occupation`` opens
        with ``offload_tags.remove(tag)`` (KeyError on a repeat) and
        ``release_memory_occupation`` pauses unconditionally while deriving
        ``sleep_begins``/``family_paused_before`` from the offload set.  The
        epoch-scoped ledger added in the same commit
        (``weight_updater.Weg2LegLedger``) is what makes a repeat a replay: a
        leg identified by (op, epoch, tag set) is applied at most once per rank
        and a second arrival returns the recorded answer.  The front therefore
        retries ONLY requests that carry an epoch -- without one there is no
        dedup key on the far side and the retry would be a second application.

        A SEPARATE SESSION, not the shared one: the hypothesis under test is a
        stale POOLED connection, so the retry must not be able to draw from
        that pool.  ``force_close=True`` also guarantees it leaves nothing
        pooled behind, which is what the mutant test reads.
        """
        # Through `self.rpc`, not past it: that is the seam every caller and
        # every existing test stubs, and a stub that does not set the flag is
        # simply never retried -- the old behaviour, unchanged.
        RPC_LAST_RETRYABLE.set(False)
        code, text = await self.rpc(g, path, body, timeout)
        if not RPC_LAST_RETRYABLE.get():
            return code, text
        if (body or {}).get("epoch") is None:
            logger.info(
                "WEG2-RPC NO-RETRY leg=%s group=%s path=%s reason=no epoch on the "
                "request, so the far side has no dedup key and a retry could "
                "apply the leg twice (#1285)",
                rpc_leg_name(path), g.name, path,
            )
            return code, text
        logger.info(
            "WEG2-RPC RETRY leg=%s group=%s path=%s epoch=%s reason=%s "
            "(one retry, fresh force_close connection, no pool)",
            rpc_leg_name(path), g.name, path, (body or {}).get("epoch"), text,
        )
        connector = SportTCPConnector(force_close=True, limit=1)
        session = ClientSession(timeout=ClientTimeout(total=timeout),
                                connector=connector,
                                trace_configs=[make_rpc_trace_config()])
        try:
            code, text, _ = await self._rpc_attempt(session, g, path, body, timeout)
        finally:
            await session.close()
        return code, text

    async def _rpc_attempt(self, session: ClientSession, g: Group, path: str,
                           body: Optional[dict], timeout: float,
                           ) -> Tuple[int, str, bool]:
        """ONE attempt, instrumented (#1285).

        Returns ``(code, text, retryable)``.  ``retryable`` is True for exactly
        one shape: a connection-level failure raised BEFORE any response line,
        i.e. the request provably never reached a handler.  A drop DURING the
        body read is a PARTIAL RESPONSE -- the handler ran, its effect is
        applied, and re-sending would apply it twice -- so it comes back
        ``retryable=False`` however connection-shaped its exception is.  The
        `got_response` flag below is that discriminator, and it is the whole
        safety argument for the retry in :meth:`leg_rpc`.
        """
        leg = rpc_leg_name(path)
        epoch = (body or {}).get("epoch")
        info: Dict[str, str] = {"conn": "unknown"}
        idle, total = rpc_pool_counts(session)
        logger.info(
            "WEG2-RPC ISSUED leg=%s group=%s path=%s epoch=%s conn=%s sport=%s "
            "pool_idle=%d pool_total=%d",
            leg, g.name, path, epoch, "pending", "pending", idle, total,
        )
        t0 = time.perf_counter()
        got_response = False
        try:
            async with session.post(f"{g.url}{path}", json=body or {},
                                    headers=admin_key_mod.auth_headers(self.admin_key),
                                    timeout=ClientTimeout(total=timeout),
                                    trace_request_ctx=info) as r:
                got_response = True
                sport = rpc_response_sport(r)
                text = (await r.read()).decode(errors="replace")
                status = r.status
            idle, total = rpc_pool_counts(session)
            logger.info(
                "WEG2-RPC RETURNED leg=%s group=%s path=%s epoch=%s conn=%s sport=%s "
                "pool_idle=%d pool_total=%d code=%d ms=%.0f",
                leg, g.name, path, epoch, info["conn"], sport, idle, total,
                status, (time.perf_counter() - t0) * 1000,
            )
            RPC_LAST_RETRYABLE.set(False)
            return status, text, False
        except Exception as e:  # noqa: BLE001
            idle, total = rpc_pool_counts(session)
            # `sport=n/a` and not a number: there is no response, hence no
            # connection object to read a sockname off (instrument limits at
            # RPC_CONN_ERRORS above).  Absent, never 0.
            retryable = isinstance(e, RPC_CONN_ERRORS) and not got_response
            RPC_LAST_RETRYABLE.set(retryable)
            logger.info(
                "WEG2-RPC RAISED leg=%s group=%s path=%s epoch=%s conn=%s sport=n/a "
                "pool_idle=%d pool_total=%d after_ms=%.0f got_response=%s retryable=%s "
                "%s: %s",
                leg, g.name, path, epoch, info["conn"], idle, total,
                (time.perf_counter() - t0) * 1000, got_response, retryable,
                type(e).__name__, e,
            )
            return 0, f"{type(e).__name__}: {e}", retryable

    async def timed_rpc(self, g: Group, path: str, body: Optional[dict],
                        timeout: float) -> Tuple[int, str, float]:
        """:meth:`rpc` plus the wall time of THIS leg alone.

        C9 gathers the two legs, so the flip's own ``interleave`` wall clock is
        no longer the sum of the parts and neither leg's cost can be read off
        it.  Each leg times itself; the sum and the wall are then two different
        measured quantities and L5 prints both plus their difference (spec C11).
        """
        # #1285: the two gathered legs -- and ONLY they -- go through the
        # retry-once path.  The discriminator is the leg name, i.e. the path
        # itself, so a future third caller of ``timed_rpc`` on some other
        # endpoint does not silently inherit a retry it has no dedup for.
        t0 = time.perf_counter()
        if rpc_leg_name(path) == "other":
            code, text = await self.rpc(g, path, body, timeout)
        else:
            code, text = await self.leg_rpc(g, path, body, timeout)
        return code, text, (time.perf_counter() - t0) * 1000

    async def _weg2_decode_progress(self, g: Group) -> Optional[dict]:
        """#1317c: D's monotone progress counters, or None if unreadable.

        Read off the SAME endpoint the front already polls for the draft terms
        (`/get_server_info` -> `internal_states[0]`), so this adds no new poll
        and no new endpoint. None -- never a zero -- when the read fails or the
        group does not publish the block: an unreadable counter must reach the
        caller as an ABSENCE, so it can keep the old residency behaviour
        instead of reading "no progress" out of a failed HTTP call and
        manufacturing the very W2 this change exists to prevent.
        """
        try:
            async with self.session.get(f"{g.url}/get_server_info") as r:
                info = await r.json() if r.status == 200 else None
            if isinstance(info, list) and info:
                info = info[0]
            if not isinstance(info, dict):
                return None
            st = info.get("internal_states") or []
            if isinstance(st, list) and st and isinstance(st[0], dict):
                info = st[0]
            blk = info.get("weg2_decode_progress")
            return blk if isinstance(blk, dict) else None
        except Exception as e:  # noqa: BLE001 - an instrument never breaks the flip
            logger.debug("weg2 decode progress unavailable: %s: %s", type(e).__name__, e)
            return None

    async def drain(self, g: Group) -> bool:
        """Wait for the group to go empty, and RECORD whether it was working.

        #1317c: the return value still answers only "did it go empty", because
        that is what the flip needs. What changed is that a refusal is no
        longer self-evidently a wedge: this samples D's monotone counters at
        both ends of the window and leaves the delta on
        ``self._drain_progress`` for `flip` to read. A window in which D
        emitted tokens or ran forward passes is a WAIT, not a refusal.
        """
        t0 = time.time()
        before = await self._weg2_decode_progress(g)
        self._drain_progress = None
        while g.outstanding:
            if time.time() - t0 > self.drain_deadline_s:
                after = await self._weg2_decode_progress(g)
                self._drain_progress = weg2_drain_progress_delta(before, after)
                return False
            await asyncio.sleep(0.25)
        return True

    async def quiesce(self, g: Group) -> Tuple[bool, str]:
        """Witness B: poll /flush_cache (200 iff the GROUP's is_fully_idle,
        including the HiCache in-flight terms) to a deadline.

        #1268: this said "the rank's", and it meant it -- the entrypoint rank
        answered for the group. On boot weg2sb1 PP0 and PP1 said flushed while
        PP2 said `hicache_prefetch(1: 43c9af54)` in the same second, only PP0's
        answer left the group, and the sleep that followed killed it by W29.
        The verdict is now reduced over every rank at its source
        (`Scheduler.group_idle_verdict`), so this poll is a group fact and the
        deadline below is the bounded wait that precedes a NAMED refusal (W3)
        instead of an assert-death on a follower."""
        t0 = time.time()
        last = ""
        while time.time() - t0 < QUIESCE_DEADLINE_S:
            code, body = await self.rpc(g, "/flush_cache", None, 60)
            if code == 200:
                return True, body
            last = body
            await asyncio.sleep(0.5)
        return False, last

    # #1236: `_store_used_bytes` IS DELETED, not repaired. It read
    # `statvfs(store_dir)` and returned `(f_blocks - f_bfree) * f_frsize`,
    # which was the store's own content only because the store WAS a whole
    # tmpfs. The store is a plain directory on the shared ZFS dataset now, so
    # that same call answers with the ROOT FILESYSTEM's usage -- 1.7 TB, not a
    # store -- and its one consumer (`host_ledger.dormant_image_sample`'s run
    # residual) no longer wants the term at all: those bytes are not in this
    # cgroup, so subtracting them would credit the residual with memory the
    # reading never held. A quantity that stopped meaning anything is removed
    # rather than given a new definition beside its old name.

    def sample_dormant_image(self, group: str, shmem_before: Optional[int]) -> Optional[dict]:
        """Measure ``group``'s dormant host image, once, at its first sleep.

        The term boot weg2dk7 refuted: the ledger charged the weight-tag byte
        sum (28.83 GiB for P) while the sleeping group's per-rank ``RssShmem``
        measured 38.63 GiB -- everything with ``enable_cpu_backup`` is in the
        image, not only the ``weights_*`` tags.  Returns ``None`` (and measures
        nothing) once that group has a sample, so a boot's images are the ones
        its FIRST sleeps produced and not a moving average of its flips.
        """
        if group in self.dormant_image:
            return None
        g = self.groups[group]
        pids = sorted(_session_pids(g.sid)) if g.sid else []
        cg = host_ledger.read_cgroup()
        weight_tags = (
            host_ledger.WEIGHT_TAGS_P_BYTES if group == "P" else host_ledger.WEIGHT_TAGS_D_BYTES
        ) / host_ledger.GIB
        rec = host_ledger.dormant_image_sample(
            group=group,
            shmem_before_bytes=shmem_before,
            shmem_after_bytes=host_ledger.read_cgroup_shmem_bytes(),
            pids=pids,
            weight_tags_gib=weight_tags,
            # A flip's sleep IS interleaved: the destination is resuming and the
            # patched saver frees ITS image while this one is written, so the
            # cgroup shmem delta is the difference of two images.
            interleaved=True,
            boot_tag=self.tag,
            commit=self.commit,
            cg_current_bytes=cg.get("current"),
            reclaimable_bytes=cg.get("reclaimable"),
            arm=self.ledger_arm or None,
            # #1325: the MEASURED load witness at the sampling moment. Both
            # terms are already held here, so nothing new is instrumented: the
            # front's own queue depth and the number of requests in flight
            # across both groups. A first sleep during a flip under load reads
            # non-zero and the record is stamped `loaded` -- which is what
            # sn6s's 16.34 GiB sample was, taken at 15:21:55Z with the 120k
            # load running, while sn6p's 9.01 GiB was taken idle.
            # #1350: this sample is taken INSIDE a flip, so its run-moment
            # residual already contains that flip's permanent step. Stamping
            # the epoch is what lets `run_origin_gib` mark the floor and
            # `predicted_run_peak_gib` refuse (W95) rather than add the same
            # bytes to the origin and to the charges.
            sampled_at_flip_epoch=self.epoch,
            load_witness={
                "queued": len(self.queue),
                "outstanding": sum(
                    len(getattr(gr, "outstanding", ()) or ())
                    for gr in self.groups.values()
                ),
            },
        )
        self.dormant_image[group] = rec
        logger.info("%s", host_ledger.format_dormant_image(rec))
        if self.measured_record:
            try:
                host_ledger.append_measured_record(self.measured_record, rec)
            except OSError as e:  # noqa: BLE001
                logger.error("WEG2 DORMANT-IMAGE not persisted to %s: %s -- the next boot "
                             "will price the recorded dk7 reading instead", self.measured_record, e)
        return rec

    def _write_flip_ratchet(self, done_epoch: int) -> None:
        """#1350 READING 2 OF 2: the FIRST FULL PAIR's permanent step, recorded.

        Called from the `WEG2-FLIP done` path and does nothing unless this is
        ``done epoch=2`` -- one D->P leg plus one P->D leg after the ``begin
        epoch=0`` reading, i.e. exactly one pair, which is what the next boot's
        ledger prices (:func:`host_ledger.resolve_flip_ratchet_gib`).

        WHY THE PAIR AND NOT THE PEAK. The peak-minus-origin figure the analysis
        tabulates (weg2xsn20 +4.462 GiB) is not knowable until the boot is over,
        and a term the ledger charges must be readable from a record a
        PREDECESSOR wrote. The pair is both legs' step, measured at two moments
        this front already logs, available while the boot still runs, and it is
        the conservative-enough half: it captures 3.161 of weg2xsn20's 4.462 and
        4.049 of weg2xsn24's 4.280.

        A boot that never completes a pair writes NOTHING (weg2xsn21b died in
        leg 2 on W85) and the next arm refuses by name rather than inventing a
        number. Both readings unreadable -> the entry is still written, with
        ``flip_ratchet_gib: None``, because why a boot could not measure is
        evidence.
        """
        if self._flip_ratchet_written or int(done_epoch) != 2:
            return
        self._flip_ratchet_written = True
        post = host_ledger.read_flip_currency_gib()
        at = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        rec = host_ledger.flip_ratchet_record(
            pre_gib=self._flip_ratchet_pre_gib,
            post_gib=post,
            boot_tag=self.tag,
            commit=self.commit,
            at=at,
            pre_at=self._flip_ratchet_pre_at,
            post_at=at,
            # The same shape `dormant_image_sample` builds its form key from --
            # the terms that fix the host posten -- so a reader can tell whether
            # a recorded ratchet speaks for THIS boot's form.
            form_key=f"wtags={len(self.weights_tags)}",
        )
        val = rec["flip_ratchet_gib"]
        logger.info(
            "WEG2-FLIP-RATCHET post epoch=2 nonreclaim=%s GiB pre=%s GiB -> "
            "flip_ratchet_gib=%s (ONE full pair, anon+shmem+slab_unreclaimable; "
            "#1350: the next boot CHARGES this instead of leaving it to a margin "
            "term whose LOCAL instrument reads negative on a staircase)",
            "unreadable" if post is None else f"{post:.3f}",
            "unreadable" if self._flip_ratchet_pre_gib is None
            else f"{self._flip_ratchet_pre_gib:.3f}",
            "UNMEASURED (a reading was absent -- the next arm refuses W94, it "
            "does not read this as 0)" if val is None else f"{float(val):.3f} GiB",
        )
        if not self.measured_record:
            logger.error(
                "WEG2-FLIP-RATCHET not persisted: this front has no measured-record "
                "path, so the next boot will find no flip_ratchet_gib and refuse W94"
            )
            return
        try:
            host_ledger.append_measured_record(self.measured_record, rec)
        except OSError as e:
            logger.error(
                "WEG2-FLIP-RATCHET not persisted to %s: %s -- the next boot will "
                "find no flip_ratchet_gib and refuse W94 rather than price a 0",
                self.measured_record, e,
            )

    async def flip(self, src: str, dst: str) -> None:
        S, D = self.groups[src], self.groups[dst]
        self.state = "flipping"
        t_flip0 = time.time()
        # #1262 tier 3: the open flip's identity, for `flip_stall_check`. Not
        # cleared on the exits below -- every one of them leaves `state` at
        # "serving" or "STOP", and the detector fires only while it is
        # "flipping", so there is no path on which a stale t0 can be read.
        self._flip_t0 = t_flip0
        self._flip_stage = "drain"
        logger.info("WEG2-FLIP begin epoch=%d sleep=%s wake=%s outstanding=%d queue=%d", self.epoch, src, dst, len(S.outstanding), len(self.queue))
        # #1350 READING 1 OF 2, at a moment this front already owns. Only at
        # epoch 0: the term is the step the FIRST waking of each group adds, it
        # SATURATES after the first pair (weg2xsn20: 3.16 of 4.46 GiB in the
        # first two legs of eight), and re-taking it later would measure a
        # plateau against a plateau.
        if self.epoch == 0 and self._flip_ratchet_pre_gib is None:
            self._flip_ratchet_pre_gib = host_ledger.read_flip_currency_gib()
            self._flip_ratchet_pre_at = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
            logger.info(
                "WEG2-FLIP-RATCHET pre epoch=0 nonreclaim=%s GiB at=%s "
                "(anon+shmem+slab_unreclaimable, NEVER memory.current; #1350 -- "
                "the post reading lands at `WEG2-FLIP done epoch=2` and the "
                "difference is written to the measured record as flip_ratchet_gib)",
                "unreadable" if self._flip_ratchet_pre_gib is None
                else f"{self._flip_ratchet_pre_gib:.3f}",
                self._flip_ratchet_pre_at,
            )
        # 1. drain (W1/W2)
        if not await self.drain(S):
            # #1317c PROGRESS BEFORE VERDICT. A window in which D made decode
            # progress is a WAIT, not a refusal: under the standing user law
            # (#1011, "er decoded zuende ... und flippt dann zurueck") a
            # decoding request runs to the end and is never cut, so it can
            # never be a wedge. W1 counts only a NO-PROGRESS window, and the
            # streak resets on a working one -- so W2 still needs three
            # consecutive windows in which D moved nothing, which is what the
            # guard was always meant to catch. The 120 s window is unchanged.
            #
            # An UNREADABLE counter (None) keeps the OLD behaviour and says so:
            # it must not be read as "no progress", or a failed HTTP call would
            # manufacture the W2 this change exists to prevent.
            _prog = getattr(self, "_drain_progress", None)
            if _prog is not None and _prog.get("progressed"):
                self.counters["weg2_drain_waiting"] += 1
                logger.warning(
                    "WEG2 DRAIN WAITING decode progress rids=%s tokens=%d "
                    "prefill_tokens=%d forward=%d running=%d window=%.0fs "
                    "streak_reset_from=%d "
                    "(denominator: drain windows that ended non-empty; this one "
                    "is a WAIT, not a W1 -- D emitted tokens or ran forward "
                    "passes inside it, and #1011 says a decode is never cut)",
                    sorted(S.outstanding)[:8], _prog.get("tokens", 0),
                    _prog.get("prefill_tokens", 0), _prog.get("forward", 0),
                    _prog.get("running", 0),
                    self.drain_deadline_s, self.drain_refusals_in_a_row,
                )
                self.drain_refusals_in_a_row = 0
                self.state = "serving" if self.state != "STOP" else "STOP"
                return
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
        self._flip_stage = "quiesce"
        idle, msg = await self.quiesce(S)
        wv = witness_verdict(len(S.outstanding), idle)
        if wv is not None:
            # #1268: name it as the GROUP's verdict, because it now is one --
            # the reply carries the reduced answer and the lowest blocking rank.
            # The old wording ("rank(P) flush_cache") described the defect: one
            # rank's word standing for three.
            self.do_stop("W3 Weg2DrainWitnessDisagreement",
                         f"{wv}: front ledger {sorted(S.outstanding)} vs group({src}) "
                         f"flush_cache (reduced over every rank, #1268) -> {msg[:400]!r}")
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
        # fix 8: the cgroup shmem reading the source's image is written against.
        shmem_before = host_ledger.read_cgroup_shmem_bytes()
        t0 = time.time()
        self._flip_stage = "sleep-kv"
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
        # #1264 (A), THE weg2t2b KILLER, and it is a READER bug not a producer
        # one: `_nvml_free()` returns `CardFree` FROZEN DATACLASSES, which are
        # not iterable, so unpacking one as `idx, _uuid, free` raises
        # `TypeError: cannot unpack non-iterable CardFree object`.  Measured
        # boot weg2t2b 2026-09-08 12:43:04,131Z, 0.6 s after `WEG2-FLIP begin`
        # and immediately after D's kv sleep RPC returned 200 -- the controller
        # caught it, `_flip_stage` was still "sleep-kv", and the front sat in
        # state="flipping" for the rest of the boot (see the handler in
        # `controller` for the second half of this fix).
        #
        # The two readers of this producer diverged at a MERGE, not in one
        # edit: `1a1247f8e6` (2026-09-07) added this line against the old
        # 3-tuple shape while `02d9811adb` (2026-09-08) changed the producer to
        # `CardFree` and migrated the OTHER reader (`corridor_sample`, which
        # uses attribute access).  Neither branch was wrong alone.  Attribute
        # access is now the ONE shape both readers use, so a further field on
        # `CardFree` cannot break either.
        free_mib = {c.nvml_index: c.free_mib for c in _nvml_free()}
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
        self._flip_stage = "gathered-legs"
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
        # 4. measure D_c(src) on the DEVICE axis; W19 for D at its first sleep.
        # fix 8: and the HOST axis, once per group -- the dormant image the next
        # boot's ledger prices instead of the weight-tag census sum.
        self.sample_dormant_image(src, shmem_before)
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
        self._flip_stage = "wake-kv"
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
        # #1262 tier 3: the flip is closed, so there is nothing open to stall.
        self._flip_t0 = None
        self._flip_stage = "none"
        # C8: phase dwell restarts here, and the L2 admission ordinal with it.
        self.t_awake = time.time()
        self._admitted_this_epoch = 0
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
        # #1271 (b): this boot's own flip cost feeds the live X.
        # #1289: THE KEY IS `flip_ms`. `rec` has never had a `flip_total_ms`
        # -- the name only ever existed in the LOG LINE below ("flip_total=%d
        # ms", fed from `rec["flip_ms"]`), and `dict.get` with a default of 0
        # turned that mismatch into a silent 0.0 that `note_x_sample` then
        # dropped on its own `value <= 0` guard. Result on weg2sb5f: 14
        # completed flips, ZERO flip_s samples, and `resolve_x_live` returning
        # None at its first line for the whole boot. Read from the SAME key
        # the log line reads, so the two can never disagree again.
        self.note_x_sample("flip_s", float(rec["flip_ms"]) / 1000.0)
        self._write_flip_ratchet(rec["epoch"])
        logger.info("WEG2-FLIP done epoch=%d slept=%s woke=%s drain+quiesce=%d ms sleep=%d ms (kv RPC + the %s leg of the gathered pair) "
                    "wake=%d ms (the %s leg + kv RPC) "
                    "interleave=%d ms (NOT sleep+wake: the legs overlap -- gather wall %d ms against %d + %d ms of legs) "
                    "overlap=%d ms (%.0f%% of the shorter leg) critical_path=%s "
                    "flip_total=%d ms weights_tags=%d dc=%s",
                    rec["epoch"], src, dst, rec["drain_quiesce_ms"], rec["sleep_ms"], src,
                    rec["wake_ms"], dst, rec["interleave_ms"], rec["legs_wall_ms"],
                    rec["sleep_leg_ms"], rec["wake_leg_ms"], rec["overlap_ms"], rec["overlap_pct"],
                    rec["critical_path"], rec["flip_ms"], len(self.weights_tags), dc)

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

    # ------------------------------------------------------------------
    # #1271 (b): X IS RE-SOLVED FROM THIS BOOT'S OWN SAMPLES.
    #
    # The shipped X came from the PREVIOUS boot's medians, read once at launch
    # and never revisited, so a rate change could not act until the boot after
    # the one that measured it. Worse, the seed itself was unstable for a
    # reason that had nothing to do with the rig: r_P was a per-request latency
    # under concurrency (#1271 (a)), so sb1's own log read 1964 or 3180 tok/s
    # depending only on which legs passed a filter.
    #
    # So: seed from the launcher's solve, then keep the boot's OWN samples and
    # re-solve. Every input is the same unit as its launcher counterpart --
    # r_P per P-DRAIN window, r_D per single prefill, flip_s per completed
    # flip -- and the re-solve refuses mixed units before any arithmetic, on
    # the same rule as the launcher.
    # ------------------------------------------------------------------
    #: New r_P samples between re-solves. Eight P-DRAIN windows is ~one agent-
    #: load minute on the measured cadence: long enough that a median is not
    #: one outlier, short enough that a real rate change acts inside the boot.
    X_RESOLVE_EVERY = 8
    #: Bounded so a long boot re-solves on its RECENT behaviour, not on its
    #: whole history -- the flip cost and both rates move with the layout.
    X_SAMPLE_WINDOW = 32

    def note_x_sample(self, kind: str, value: float) -> None:
        """Record one live X input. ``kind`` in {r_d, r_p, flip_s}.

        #1289: A COMPLETED LEG PAIR IS ITSELF A TRIGGER. Before this the ONLY
        trigger was the eighth ``r_p`` sample, i.e. the eighth P-DRAIN window
        -- so on a boot whose load never routes long (weg2sb5f: 14 flips but
        only SEVEN drains, because every prompt priced below X) the re-solve
        could not fire even once, whatever the flip cost did. The legs are the
        measurement this ticket is about, so a leg pair re-solves on its own.
        The r_p trigger is kept unchanged beside it: two ways in, one
        arithmetic, and the smoothing window below still decides how much
        history each median sees.
        """
        if value is None or value <= 0:
            return
        buf = self._x_samples.get(kind)
        if buf is None:
            return
        was_empty = not buf
        buf.append(float(value))
        # #1291: THE SAMPLE THAT COMPLETES THE TRIPLE RESOLVES IMMEDIATELY.
        # Without this, an `r_d` arriving last (the sb5g order: flip_s and r_p
        # first, r_d starved) waits for the NEXT flip or the eighth drain
        # before the solve it just unblocked can run -- so the term that was
        # missing all boot buys nothing on the pass that finally supplies it.
        # NOT for `r_p`: that kind carries #1271's deliberate every-8
        # contract (a median over eight drain windows, not over one), and
        # `test_red_first_x_does_not_move_before_n_samples` is that contract.
        # The starved term on sb5g was `r_d`, and `flip_s` has its own
        # per-leg trigger below, so completing on those two is enough.
        if (kind != "r_p" and was_empty
                and all(self._x_samples[k] for k in ("r_d", "r_p", "flip_s"))):
            self.resolve_x_live()
            return
        if kind == "flip_s":
            # Every completed leg pair. The median over the window is what
            # smooths -- not a count of legs withheld from the solver.
            self.resolve_x_live()
        elif kind == "r_p":
            self._x_since_resolve += 1
            if self._x_since_resolve >= self.X_RESOLVE_EVERY:
                self._x_since_resolve = 0
                self.resolve_x_live()

    def x_flip_s_provenance(self) -> str:
        """``flip_s source=live|seed n=`` -- on EVERY X decision (#1289).

        ``seed`` means no leg of THIS boot has been measured yet and X is
        still the launcher's carried-in constant; ``live`` means the median
        below is this boot's own legs and names how many. sb5f shipped
        ``flip_s=3.79 (median of 30)`` -- thirty flips of the PREVIOUS boot,
        on a different layout, while its own eight legs measured 3.906 s
        (D->P) and 4.175 s (P->D). Without this line the reader cannot tell a
        carried-in number from a measured one, and that is the whole defect.
        """
        n = len(self._x_samples["flip_s"])
        nd = len(self._x_samples["r_d"])
        # #1289 round 2: r_D gets the same treatment as flip_s. sb5g proved
        # the value of naming the STARVED term: the NO-SOLVE line said
        # "no r_d sample yet" and that one word located the defect.
        return (f"flip_s source={'live' if n else 'seed'} n={n} "
                f"r_d source={'live' if nd else 'none'} n={nd} "
                f"src={self._x_r_d_src}")

    def resolve_x_live(self) -> Optional[int]:
        """Re-solve X from the boot's own medians; return the new X or None.

        Prints the re-solve WITH ITS PROVENANCE -- an X that changed silently
        is a routing threshold nobody can account for after the fact.

        #1289: AND IT PRINTS WHEN IT DOES NOT. The early return below used to
        be silent, so a boot with an empty ``flip_s`` deque (the whole of
        weg2sb5f) looked exactly like a boot where nothing needed re-solving.
        A missing input is now a named, rate-limited line that says WHICH
        input is missing.
        """
        import statistics as _st

        from sglang.srt.weg2.launcher import (
            RATE_UNIT_GROUP_THROUGHPUT,
            MixedRateUnits,
            derive_x_star,
        )

        s = self._x_samples
        if not (s["r_d"] and s["r_p"] and s["flip_s"]):
            missing = [k for k in ("r_d", "r_p", "flip_s") if not s[k]]
            if missing != self._x_last_missing:
                self._x_last_missing = missing
                logger.warning(
                    "WEG2 X NO-SOLVE: no %s sample yet, so X stays at the "
                    "carried-in %d (%s; have r_d=%d r_p=%d flip_s=%d). This "
                    "line exists because the same state was SILENT on "
                    "weg2sb5f for 14 flips (#1289)",
                    " and ".join(missing), self.tp_prefill_max_tokens,
                    self.x_flip_s_provenance(),
                    len(s["r_d"]), len(s["r_p"]), len(s["flip_s"]),
                )
            return None
        self._x_last_missing = []
        r_d = _st.median(s["r_d"])
        r_p = _st.median(s["r_p"])
        flip_s = _st.median(s["flip_s"])
        prev = self.tp_prefill_max_tokens
        try:
            # BOTH group_throughput: r_D is tokens/wall of a D prefill that ran
            # alone, r_P is a drain's tokens over the drain's own wall. Passing
            # the units explicitly is what makes a future estimator swap fail
            # loudly instead of silently re-introducing (a).
            x = derive_x_star(
                flip_s, r_d, r_p, self.x_floor_tokens,
                unit_d=RATE_UNIT_GROUP_THROUGHPUT,
                unit_p=RATE_UNIT_GROUP_THROUGHPUT,
            )
        except MixedRateUnits as e:
            logger.error("WEG2 X RE-SOLVE REFUSED (units): %s", e)
            return None
        except ValueError as e:
            # No break-even (r_D >= r_P) is a real state, not an error: the
            # round trip does not pay and X stays where it was.
            logger.warning(
                "WEG2 X RE-SOLVE held: %s -- X stays %d (%s; n=%d/%d/%d)",
                e, prev, self.x_flip_s_provenance(),
                len(s["r_d"]), len(s["r_p"]), len(s["flip_s"]),
            )
            return None
        self.tp_prefill_max_tokens = x
        if self._x_min_work_follows:
            self.flip_min_work_tokens = x
        self.counters["x_resolves"] += 1
        logger.info(
            "WEG2 X RE-SOLVE n=%d X=%d <- X_prev=%d r_D=%.0f r_P=%.0f flip_s=%.2f "
            + self.x_flip_s_provenance() +
            " source=live (medians over this boot's own samples: %d r_D, %d r_P "
            "drains, %d flips, window %d; seeded from %s. Both rates are "
            "group_throughput -- tokens the group moved over the wall it was "
            "busy -- so 1/r_D-1/r_P is a time-per-token difference; #1271)",
            len(s["r_p"]), x, prev, r_d, r_p, flip_s,
            len(s["r_d"]), len(s["r_p"]), len(s["flip_s"]), self.X_SAMPLE_WINDOW,
            self._x_seed_note,
        )
        return x

    def _flip_economics_ok(self, fairness_fired: bool) -> bool:
        """C7/L13: is the queued work worth a round trip?

        The PRIMARY anti-thrash, and the one the measured 4.3 s P phase
        needed.  The threshold is X at aggregate granularity -- the same
        break-even quantity law 4 applies per request.
        """
        # #1271 (c): THE BACKLOG SUM IS OVER UNCACHED TOKENS, not est_prompt.
        # `2*flip_s` is amortised ONCE over the whole queued prefill backlog, so
        # the break-even is the same X* law 4 applies per request -- but the
        # quantity it is applied to must be the work the flip actually causes.
        # `est_prompt` includes the cached head, which P does not recompute; a
        # backlog summed on it flips on prefixes that are already resident.
        queued_uncached = sum(int(p.est_uncached) for p in self.queue)
        queued_tokens = sum(int(p.est_prompt) for p in self.queue)
        threshold = self.flip_min_work_tokens
        ok = queued_uncached >= threshold or fairness_fired or not self.admit_d

        # THE STRANDED-DECODE TERM, REPORTED AND NEVER A VETO. Every decode
        # resident on D stops for the whole P phase; the user accepted unbounded
        # decode wait under a prefill stream, so this is priced into the log and
        # not into the verdict. It is printed on BOTH verdicts -- a `hold` that
        # strands nobody and a `hold` that strands eight are different states and
        # the line has to tell them apart.
        now = time.time()
        try:
            D = self.groups.get("D")
            stranded = len(getattr(D, "outstanding", ()) or ()) if D else 0
            oldest = max(
                (now - float(getattr(v, "t_arrive", now))
                 for v in (getattr(D, "outstanding", {}) or {}).values()),
                default=0.0,
            )
        except Exception:  # noqa: BLE001 - a report may never break the verdict
            stranded, oldest = -1, -1.0

        logger.info(
            "WEG2 FLIP-ECONOMICS queued_uncached=%d queued_tokens=%d threshold=%d "
            "(X*, amortising 2*flip_s once over the backlog) fairness=%s "
            "stranded_decodes=%d oldest_wait_s=%.1f verdict=%s "
            "(stranded is REPORTED, never a veto -- unbounded decode wait under a "
            "prefill stream is accepted; -1 means the census could not be taken)",
            queued_uncached, queued_tokens, threshold, fairness_fired,
            stranded, oldest, "flip" if ok else "hold",
        )
        return ok

    def _idle_disposition(self, awake: str, at_rest: bool) -> str:
        """LAW 5, ONE DECISION FOR BOTH DIRECTIONS: rest, flip, or busy.

        MF-1 (operator, boot weg2sc2, and it is a CODE FACT rather than a
        missed measurement).  The two idle arms this replaces were each gated
        on the OTHER layout: the D-awake mirror required
        ``idle_layout == "P"`` and the P-awake tail flipped whenever
        ``idle_layout == "D"``.  ``--idle-layout tp`` makes the launcher emit
        front ``--idle-layout D``, so under tp -- the DEFAULT -- neither arm
        could reach its ``WEG2 IDLE-REST``: the witness for law 5's own
        acceptance probe did not exist in that direction, which retro-explains
        weg2sc1's and weg2sc2's zero IDLE-REST lines.

        THE DECISION, keyed on the CONFIGURED layout and nothing else:

        * not at rest -> ``"busy"``; the caller does its ordinary work.
        * at rest and the awake group IS ``self.idle_layout`` -> ``"rest"``,
          and this method prints the L-line naming the layout and the reason.
        * at rest and it is NOT -> ``"flip"``; the caller flips (still behind
          its own dwell/economics latches, which this method does not
          second-guess), and the NEXT pass rests in the configured layout.
          One flip, then rest -- the mirror, stated once.

        THE LATCH is the printer's own de-dup, not a second scheduling
        ledger: the controller wakes every 0.2 s, so a resting front would
        otherwise emit five identical lines a second for as long as it is
        idle.  The line is edge-triggered -- printed on the transition into
        rest, re-armed by any pass that is not resting (work arriving, a
        flip, a hand-off in flight).

        WHAT WENT AWAY WITH THE OLD SHAPE, deliberately: the D-awake arm used
        to print ``IDLE-REST`` and then FLIP in the same pass, i.e. the line
        claimed a rest the front was in the act of leaving.  A flip announces
        itself with ``WEG2-FLIP begin``; ``IDLE-REST`` now means what it says.
        """
        if not at_rest:
            self._idle_rest_shown = False
            return "busy"
        if awake != self.idle_layout:
            self._idle_rest_shown = False
            return "flip"
        if not self._idle_rest_shown:
            self._idle_rest_shown = True
            logger.info("WEG2 IDLE-REST layout=%s configured=%s reason=%s queue=0 ready_for_d=%d "
                        "d_outstanding=%d handing_off=%d held_s=%.1f (no backlog and the awake "
                        "group IS the layout --idle-layout asked for; there is nothing to flip to)",
                        awake, self.idle_layout, "awake_group_is_configured_idle_layout",
                        len(self._ready_for_d), len(self.groups["D"].outstanding),
                        int(self._handoff_in_flight()), time.time() - self.t_awake)
        return "rest"

    def _log_p_prefix_reuse(self, before: Tuple[int, int, int]) -> None:
        """MF-3 (L15): this drain epoch's forgone prefix reuse on group P.

        The terms and their instruments are documented on
        :meth:`_note_p_prefix_reuse`, which is where they are counted.  Here
        they are only differenced against the epoch's entry reading and
        spoken -- with ``requests=0`` printed too, so a drain that prefilled
        nothing is a reading rather than a silence.
        """
        n = self.counters.get("p_prefill_requests", 0) - before[0]
        avail = self.counters.get("p_prefix_tokens_in_store", 0) - before[1]
        reused = self.counters.get("p_prefix_tokens_reused", 0) - before[2]
        logger.info("WEG2 P-PREFIX-REUSE epoch=%d requests=%d prefix_tokens_available_in_store=%d "
                    "prefix_tokens_reused=%d forgone_tokens=%d (#1245 carrierless PP: P READS the "
                    "store, but a host hit that lands on one rank alone is dropped at admission -- "
                    "W38, which refused the read outright, was retired into this arm on the 0908 "
                    "train; available= is the front's own routing probe, a text-span ESTIMATE and a "
                    "LOWER bound; reused= is MEASURED from P's leg-1 cached_tokens; forgone= is an "
                    "UPPER bound on what the drop costs, not a refused read; the remedy that ends "
                    "the drop is the PP0-authoritative materialisation, #968)",
                    self.epoch, n, avail, reused, max(0, avail - reused))

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
                        #
                        # FIX 2 (round 2): THE SEAT IS THE THIRD TERM, and
                        # without it "D holds nothing" was false.  Between
                        # the admitter's `popleft` + `fut.set_result(True)`
                        # (:932/:937) and `leg2`'s `D.outstanding[rid] = ...`
                        # the request is in NEITHER `_ready_for_d` NOR
                        # `D.outstanding` -- the two sets both guards read --
                        # so this arm could flip D->P on a request it had
                        # already handed to D, and `flip`'s own `drain(D)`
                        # could not protect it either: `D.outstanding` is
                        # empty by construction in that window, so the drain
                        # returns at once and D is put to sleep under it.
                        # The client's leg 2 then POSTs into a sleeping group
                        # and sits out the 3600 s ClientTimeout -- R-2's
                        # LOST-REQUEST class in the D->P direction.  The seat
                        # already spans exactly the missing interval (taken
                        # where the future is resolved, returned in `leg2`'s
                        # `finally`), so the hand-off is made visible with the
                        # state that exists rather than with a second ledger.
                        at_rest = (not D.outstanding and not self._handoff_in_flight()
                                   and not self._ready_for_d and self.state == "serving")
                        if self._idle_disposition("D", at_rest) == "flip":
                            if self._dwell_ok("D", "P", fairness_fired, work_exhausted=True, oldest_wait_s=0.0):
                                await self.flip("D", "P")
                        continue
                    # FIX 2 (round 2): the work arm reads the SAME hand-off
                    # window the idle mirror above does, and for the same
                    # reason -- "D has no outstanding work" is false while a
                    # seat is held for a request that has been resolved but
                    # has not yet reached `leg2`.  BOTH halves of the
                    # condition need the term: the `admit_d` half flips
                    # deliberately (the front is closing D down) and would
                    # otherwise carry a handed-off request into the sleep
                    # just as the idle mirror did.  It only ever DELAYS a
                    # D->P flip, by at most the W36 barrier that already
                    # bounds a dead client's seat.
                    handing_off = self._handoff_in_flight()
                    d_work_exhausted = not D.outstanding and not handing_off
                    if (d_work_exhausted or not self.admit_d) and not handing_off:
                        if not self._flip_economics_ok(fairness_fired):
                            continue
                        if not self._dwell_ok("D", "P", fairness_fired, work_exhausted=d_work_exhausted,
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
                # MF-3: the same delta idiom as `short_behind_p0` -- the
                # epoch's terms are read off the running counters rather than
                # carried in a second per-epoch structure.
                reuse0 = (self.counters.get("p_prefill_requests", 0),
                          self.counters.get("p_prefix_tokens_in_store", 0),
                          self.counters.get("p_prefix_tokens_reused", 0))
                _drain_uncached = 0
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
                    _drain_uncached += sum(int(q.est_uncached) for q in batch)
                    for p in batch:
                        if not p.fut.done():
                            self._ready_for_d.append(p)
                            self._sync_batch_gate()
                            prefilled += 1
                if passes:
                    oldest_short = 0.0
                    if self._ready_for_d:
                        oldest_short = time.time() - self._ready_for_d[0].t_arrive
                    # #1271 (b): ONE r_P SAMPLE PER DRAIN WINDOW, in the same
                    # unit the launcher now uses -- the uncached tokens this
                    # drain moved over the drain's own wall. NOT per request:
                    # the drain runs at p_concurrency, so a per-request wall
                    # counts queueing behind peers and is a latency, not a rate.
                    _drain_s = time.time() - t_drain0
                    if _drain_s > 0 and _drain_uncached > 0:
                        self.note_x_sample("r_p", _drain_uncached / _drain_s)
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
                    # MF-3 (L15): what group P's disarmed store read cost THIS
                    # drain, beside the drain it cost it in.
                    self._log_p_prefix_reuse(reuse0)
                # C6/R-2: the _ready_for_d term is NOT optional.  Without it,
                # --idle-layout pp keeps P awake with requests P has just
                # prefilled sitting on `await fut` behind a one-hour client
                # timeout -- a LOST-REQUEST class introduced by the fix for
                # law 5.  The admitter (C4) releases them once D is awake.
                if self.queue or self._ready_for_d:
                    await self.flip("P", "D")
                elif self._idle_disposition("P", at_rest=True) == "flip":
                    await self.flip("P", "D")
            except Weg2Stop as e:
                self.do_stop(e.name, e.detail)
            except Exception as e:  # noqa: BLE001
                # #1264 (A), THE CLASS behind the weg2t2b wedge -- the one-line
                # TypeError above was only its trigger.  `flip()` sets
                # `state="flipping"` at its first statement and clears it only
                # on its NAMED exits; an unexpected exception escapes past every
                # one of them.  Logging and continuing then leaves the front in
                # "flipping" FOREVER: the guard at the top of this loop is
                # `if self.state != "serving": continue`, so the controller
                # spins doing nothing, the queued request is never dispatched,
                # and /health keeps answering 200.  Measured weg2t2b: 123.7 s to
                # the stall line, then teardown; weg2t2a held the same shape for
                # seven minutes.
                #
                # A flip that died mid-way is not a recoverable error, it is the
                # W4 state the RPC-failure paths already name: the source's
                # kv_cache (and possibly its weights family) is paused and the
                # destination is not resumed, so VRAM occupancy is undefined on
                # both sides and no retry is legal.  So STOP BY NAME at the
                # stage that was open, and keep the plain log-and-continue only
                # for errors raised outside a flip, where the front's state
                # really is intact.
                verdict = flip_escape_verdict(
                    self.state, self._flip_stage, self.epoch, e
                )
                if verdict is not None:
                    # #1264 fix 2b (2): THE DEATH IS ITS OWN LINE, and it is
                    # emitted BEFORE the verdict and before this loop continues.
                    # `logger.exception` alone carried no marker a watcher could
                    # match, so on weg2t2b the only thing that ever spoke was the
                    # 4x stall timer, 123.7 s later and about the wrong stage.
                    # This line is boot_deadman.sh's tier-3 pattern.
                    logger.error(
                        "%s",
                        controller_dead_line(
                            self.epoch, self._flip_stage, self.flip_stage_age_s(), e
                        ),
                    )
                    self.counters["controller_dead"] += 1
                    logger.exception("controller error during flip: %s", e)
                    self.do_stop(*verdict)
                else:
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

    def corridor_sample(self) -> Optional[str]:
        """One corridor sample: read, fold into ``min_so_far``, log, return the line.

        Split out of the 10 s loop so the instrument can be tested without a
        boot.  Returns ``None`` when NVML could not be read (no sample was
        taken and nothing was folded in) -- never a partial or a stale line.
        """
        phase = self.awake
        # ONE read per sample.  The pre-fix form called the reader TWICE --
        # once to fold into min_so_far and once to print -- so the printed
        # numbers and the accumulated minimum were two different instants.
        cards = _nvml_free()
        if not cards:
            return None
        for c in cards:
            cur = self.corridor_min[phase].get(c.nvml_index)
            self.corridor_min[phase][c.nvml_index] = (
                c.free_mib if cur is None else min(cur, c.free_mib)
            )
        # #1257c: ONE floor per card, derived, with its provenance in the
        # line.  The band is no longer rig-uniform, so a single ``band=``
        # token can no longer describe the sample and each card carries its
        # own ``floor=``/``source=``/``reserve=``.  ``band=`` stays for the
        # log parsers that key on it (``ring_table``, ``corridor_arm``) and
        # now reports the SPAN of the per-card floors, which is what it
        # honestly is when they differ.
        floors = {c.uuid: corridor_floor_for_card(c.uuid, phase) for c in cards}
        per_card = " ".join(
            f"nvml{c.nvml_index}:free={c.free_mib}MiB reserved={c.reserved_mib}MiB "
            f"floor={floors[c.uuid].mib}MiB "
            # REFUTER FIX 1: the number the VERDICT on this same line is
            # graded against, printed BESIDE the floor it is derived from.
            # Pre-fix the segment carried only ``floor=`` while ``verdict=``
            # was graded against ``verdict_floor_mib``, and ``corridor_arm``
            # -- reading this very line back -- graded against the ``floor=``
            # it could see. Under an unmeasured 1024 (verdict floor 819) a
            # card at 852 MiB therefore printed ``verdict=IN`` here and was
            # failed as BELOW by the arm off the same token.
            f"verdict_floor={floors[c.uuid].verdict_floor_mib}MiB "
            f"source={floors[c.uuid].source} "
            f"reserve={floors[c.uuid].reserve_mib}MiB "
            f"verdict={corridor_verdict(c.free_mib, floors[c.uuid])}"
            + (
                f" unmobilised_free_mib={c.free_mib - floors[c.uuid].ceiling_mib}"
                if c.free_mib > floors[c.uuid].ceiling_mib
                else ""
            )
            for c in cards
        )
        instrument = corridor_instrument(cards)
        self.corridor_instrument = instrument
        self.corridor_floors = {u: f.line for u, f in floors.items()}
        lo = min(f.verdict_floor_mib for f in floors.values())
        hi = max(f.ceiling_mib for f in floors.values())
        line = (
            f"WEG2-CORRIDOR phase={phase}(awake) epoch={self.epoch} "
            f"instrument={instrument} band={lo}-{hi}MiB "
            f"{per_card} min_so_far={dict(self.corridor_min[phase])} ({instrument}, MiB)"
        )
        logger.info("%s", line)
        for f in floors.values():
            logger.info("%s", f.line)
        return line

    async def corridor_sampler(self) -> None:
        while True:
            await asyncio.sleep(10)
            if self.state != "serving":
                continue
            self.corridor_sample()

    # ---------------- #1262 TIER 3: the flip stall detector ----------------

    def _flip_stall_bound_s(self) -> Tuple[float, str]:
        """How long a flip may show no progress before it is named STALLED.

        DERIVED, never a literal, and the provenance travels with the number:

        * **After the first flip in this direction** -- ``FLIP_STALL_SLACK``
          times that flip's own measured ``flip_ms``.  This is the same
          quantity ``_derived_min_dwell_ms`` already prices a round trip with,
          read from the same ``flip_log`` records, so the detector and the
          scheduler agree on what a flip costs on THIS boot rather than on a
          recorded one.
        * **Before it** -- ``self.drain_deadline_s``, the front's OWN declared
          bound on a single flip phase (the one it refuses W1 against).  Epoch
          0 is exactly the case weg2t2a died in, so this branch is the
          load-bearing one and it deliberately reuses an EXISTING published
          front bound instead of introducing a number of its own.

        Any direction, not just this one: a flip that has not progressed is not
        a slow flip, and waiting for a same-direction sample before the
        detector can arm would leave the first flip of each direction
        unwatched -- which is the specimen.
        """
        for rec in reversed(self.flip_log):
            ms = float(rec.get("flip_ms") or 0.0)
            if ms > 0:
                return (
                    FLIP_STALL_SLACK * ms / 1000.0,
                    f"{FLIP_STALL_SLACK:.0f}x this boot's last measured flip "
                    f"({rec.get('sleep')}->{rec.get('wake')}, {ms:.0f} ms)",
                )
        return (
            self.drain_deadline_s,
            "no flip measured on this boot yet -- the front's own published "
            "drain deadline (--drain-deadline-s) stands in",
        )

    def flip_stall_check(self, now: Optional[float] = None) -> Optional[str]:
        """Emit ``WEG2-FLIP STALL`` once when a flip overruns its derived bound.

        Deadman tier 3.  Tiers 1 and 2 are structurally blind to this state --
        the processes exist and ``/health`` answers 200 while all six
        schedulers spin inside an idle-time instrument (boot weg2t2a,
        2026-09-08) -- so the signal has to come from the process that knows a
        flip is open.  Returns the line it emitted, or ``None``; the return is
        for the test, the log line is the product.

        ONCE PER FLIP, keyed on ``self.epoch`` (which does not advance until a
        flip completes, so it is the flip's identity while it is open).  A
        repeating alarm would be a persistent monitor, which this rig forbids;
        one named line is what the deadman needs and all it needs.
        """
        if self.state != "flipping" or self._flip_t0 is None:
            return None
        now = time.time() if now is None else now
        elapsed = now - self._flip_t0
        bound, provenance = self._flip_stall_bound_s()
        if elapsed < bound:
            return None
        if self._flip_stall_reported_epoch == self.epoch:
            return None
        self._flip_stall_reported_epoch = self.epoch
        line = (
            f"WEG2-FLIP STALL epoch={self.epoch} elapsed={elapsed:.1f} s "
            f"bound={bound:.1f} s "
            f"{flip_stage_report(self._flip_stage, self.flip_stage_age_s())} "
            f"awake={self.awake} "
            f"queue={len(self.queue)} flips={len(self.flip_log)} "
            f"(bound provenance: {provenance}). The flip began and has not "
            f"completed; tier 1 (a process exists) and tier 2 "
            f"(/health_generate) cannot see this state -- boot weg2t2a held it "
            f"for 7 min with all three ports answering 200"
        )
        logger.error("%s", line)
        self.counters["flip_stall"] += 1
        return line

    async def flip_stall_sampler(self) -> None:
        while True:
            await asyncio.sleep(FLIP_STALL_POLL_S)
            try:
                self.flip_stall_check()
            except Exception as exc:  # noqa: BLE001 -- a detector, never a gate
                logger.error("WEG2-FLIP STALL check raised: %r", exc)

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
    ap.add_argument("--p-concurrency", type=int, default=DEFAULT_P_BS,
                    help=f"law 1 (K3): how many leg-1 POSTs group P runs at once. CONCURRENCY ONLY -- "
                         f"the P phase ends when the backlog is empty, never on this number. Written "
                         f"by the launcher from --p-bs; the front never derives it over HTTP. This "
                         f"default ({DEFAULT_P_BS}) therefore binds ONLY when the front is run by "
                         f"hand, and it is the same provisional number the launcher ships (user "
                         f"order 2026-09-09), read from one place so the two cannot disagree.")
    ap.add_argument("--d-bs", type=int, default=DEFAULT_D_BS,
                    help=f"law 2 (K4): group D's own batch size, independent of P's. It is both D's "
                         f"--max-running-requests and the number of front seats, so the front cannot "
                         f"hand D more concurrent requests than D can run. Written by the launcher; "
                         f"this default ({DEFAULT_D_BS}) binds only for a hand-run front and is the "
                         f"same provisional number the launcher ships (user order 2026-09-09).")
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
    ap.add_argument("--carrier-max-tokens", type=int, default=0,
                    help="#1233 zero-remainder: longest prompt group D can read from the store. "
                         "0 = no CARRIER-EXCEEDS route -- and that is NOT an off switch for the "
                         "leg-1/leg-2 round trip but its opposite: both carrier guards are "
                         "'carrier_max_tokens > 0', so 0 removes the BYPASS and every prompt above the "
                         "SHORT grant round-trips with no bound at all on what the store is asked to "
                         "carry (the '#915 PREFETCH REFUSED'/W16 shape of boot weg2ls4b2). The weg2 "
                         "launcher never ships 0: its census floor refuses it (#1246).")
    ap.add_argument("--d-admit-max-tokens", type=int, default=None,
                    help="FIX 4 (round 4): OPERATOR CEILING on the AGGREGATE store-read budget "
                         "group D may hold in flight. The budget itself is not this flag and is "
                         "never front-derived: the front READS group D's own #915 terms "
                         "(available/occupied/limit) off /server_info and charges the grants that "
                         "reading cannot yet contain. Unset = that reading alone. >0 = the same "
                         "reading under this extra bound. 0 disables the gate and restores the "
                         "count-only admitter that overcommitted the pool on boot weg2sc1 (#915 "
                         "vote_negative -> a mis-priced X gate -> a store-resident rid re-queued "
                         "to P).")
    ap.add_argument("--src-chunk-cards", default="",
                    help="#1233 weg2dk4: JSON {group: {weights_k: [nvml_index, ...]}} -- which cards hold each chunk tag's "
                         "bytes, per group, derived by the launcher from that group's parallelism. A group that is absent "
                         "(or an empty map) pauses in the natural tag order, exactly as before.")
    ap.add_argument("--measured-record", default="",
                    help="#1233 fix 8: the JSON sidecar this line writes its DORMANT-IMAGE measurements "
                         "into (empty = measure and log, do not persist)")
    ap.add_argument("--admin-key-file", default="",
                    help="#1275: path of the 0600 file holding THIS boot's admin API key. "
                         "The front reads it and sends `Authorization: Bearer <key>` on every "
                         "group RPC, which is MANDATORY once the groups are started with "
                         "--admin-api-key: /flush_cache, /release_memory_occupation, "
                         "/resume_memory_occupation and /abort_request are all ADMIN_OPTIONAL, "
                         "and that level requires the admin key once one is configured. A path, "
                         "not a value, so the key does not appear in this process's argv too.")
    ap.add_argument("--commit", default="", help="#1233 fix 8: the tip this boot runs, stamped into every measurement")
    ap.add_argument("--anon-preboot-bytes", type=int, default=0,
                    help="#1269 fix 3: cgroup memory.stat `anon` measured by the "
                         "launcher preflight BEFORE this boot existed. The only "
                         "valid foreign baseline for the W22 split -- "
                         "`cgroup anon - sum(RssAnon)` is not subtractable and "
                         "printed foreign=-30.91 GiB on weg2sb5b. 0 = unset, and "
                         "then no split is printed rather than an invented one.")
    ap.add_argument("--ledger-arm", default="",
                    help="#1233 fix 8: JSON {s_gb, m_mib, store_gib} -- the arm the ledger chose, needed to "
                         "derive the RUN-MOMENT residual from the front's own cgroup reading")
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
                  drain_deadline_s=args.drain_deadline_s,
                  d_admit_max_tokens=args.d_admit_max_tokens,
                  src_chunk_cards=json.loads(args.src_chunk_cards) if args.src_chunk_cards else {},
                  measured_record=args.measured_record, commit=args.commit,
                  ledger_arm=json.loads(args.ledger_arm) if args.ledger_arm else {},
                  admin_key_file=args.admin_key_file)
    # #1269 fix 3: the pre-boot anon baseline the watermark's currency is split
    # against. Kept from weg2/idle-anon-0908.
    if args.anon_preboot_bytes > 0:
        front._anon_preboot_bytes = int(args.anon_preboot_bytes)
    # #1275: say ONCE whether this boot has the live levers, and say it with the
    # PATH and a redaction -- never the key. The front log is world-readable and
    # is routinely pasted into records.
    logger.info("WEG2 ADMIN-KEY file=%s key=%s -- admin routes (/hicache/storage-backend/resize, "
                "attach, detach, clear) are %s on this boot; the front authenticates its own "
                "flip RPCs with it (#1275)",
                args.admin_key_file or "(none)",
                admin_key_mod.redact(front.admin_key),
                "REACHABLE" if front.admin_key else "unreachable (groups started without --admin-api-key)")
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
    for path in PASSTHROUGH_POST:
        app.router.add_post(path, front.handle_passthrough_post)
    for path in FORWARD_PATHS:
        app.router.add_post(path, front.handle_generate)
    logger.info("WEG2-FRONT %s:%s -> P=%s D=%s awake=%s weights_tags=%s src_chunk_cards=%s", args.host, args.port,
                args.prefill, args.decode, args.awake, front.weights_tags, front.src_chunk_cards)
    web.run_app(app, host=args.host, port=args.port, print=None)


if __name__ == "__main__":
    main()
