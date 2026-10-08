"""fnFL2 H43: the front's hint that a long request is coming to group P.

WHY. The PLE read of a request's first chunk can start at its admission on P
(``qwen4_exp_ple_admit``) -- but in the flip case P never admits it early: the
front keeps a BATCH arrival in its own queue while D is awake, flips D->P, and
posts leg 1 only once P is awake. x146 (24.09.): pdflip-2-6 queued at the front
11:31:27.48, P's wake leg issued 11:31:27.70 and done 11:31:29.44, P's first
gather began ~11:31:29.5 and read cold for 4.47 s.

WHAT. When the front queues a BATCH request while P is not the awake group, it
POSTs the leg-1 body to P's ``/pdflip/ple_prefetch_hint`` (fire and forget). P's
HTTP server tokenizes it exactly as the leg-1 POST will be tokenized (the same
OpenAI serving conversion) and pushes a ``PlePrefetchHintReqInput`` down the
same socket its wake RPC takes -- so a hint that arrives before the wake RPC
is handled by P's scheduler before the wake, and PP0's pread workers read the
first chunk while P wakes. The later leg 1 of the same rid finds the admission
(``confirmed`` when the tokens are equal, re-admitted otherwise).

NF z30k (29.09., boot ...dauer09290122_5527da6564): 71x ``BATCH queued
(awake=D ...)``, 0x ``PDFLIP PLE-HINT``, 0x ``dormant=1`` -- the agent load comes
over ``/v1/messages`` (114 POSTs against 5 on ``/v1/chat/completions``) and the
path table knew only the OpenAI shapes, so every hint was dropped silently and
P's first cold chunk after a D->P flip waited ``ready=no wait_ms~1000`` on its
own gather. An Anthropic leg 1 is tokenized through the SAME conversion its
POST takes (``AnthropicServing._convert_to_chat_completion_request`` -- with
the process's INLINE_SYSTEM_IN_PLACE choice -- then the chat serving's
internal request), and a hint the front does not send is counted by name.

A hint is only a prefetch: nothing is queued, nothing is answered but 200, a
failed or late hint costs the gain and nothing else. Front switch
``FLLIPER_PDFLIP_PLE_ADMIT_HINT`` (default on); P needs
``FLLIPER_QWEN4_PLE_PREFETCH_ADMIT`` (the admission itself).
"""

from __future__ import annotations

import logging
from typing import Any, Callable, List, Optional

logger = logging.getLogger(__name__)

HINT_PATH = "/pdflip/ple_prefetch_hint"
_LEG1_PATHS = ("/v1/chat/completions", "/v1/completions", "/generate", "/v1/messages")


# ---------------------------------------------------------------- front side


def ple_hint_wanted(*, awake: Optional[str], skip_leg1: bool = False) -> bool:
    """A BATCH arrival P will prefill (leg 1) while P is not awake: the one
    case in which P learns of the request only after its wake. (With P awake
    the leg 1 is posted within a controller tick, or it waits for a free P
    slot while the request ahead of it on P is admitted there already.)"""
    return awake != "P" and not skip_leg1


def ple_hint_skip_reason(path: str, payload: Any) -> Optional[str]:
    """Why the front sends no hint for this leg 1 (None: it sends one)."""
    if path not in _LEG1_PATHS:
        return "path"
    if not isinstance(payload, dict):
        return "payload"
    if not payload.get("rid"):
        return "rid"
    return None


def ple_hint_body(path: str, payload: dict) -> Optional[dict]:
    """The hint body: the leg-1 path and payload (the payload is the one leg 1
    will POST: same rid, ``max_tokens`` 1, no stream)."""
    if ple_hint_skip_reason(path, payload) is not None:
        return None
    body = dict(payload)
    body.pop("stream", None)
    body.pop("stream_options", None)
    return {"path": path, "payload": body}


# ---------------------------------------------------------------- P side


#: FLIP-LEGS 02.10.: the hint is tokenized in a worker thread, never on P's
#: HTTP event loop. N5a ep7 (b49f0282c2 1002_114540): a hint for a 110k-token
#: prompt held P's loop 1366 ms (`PDFLIP PLE-HINT ... ms=1366`) and P's resume
#: RPC for the D->P flip reached PP0 720 ms after the front issued it (every
#: other D->P flip of N4p/N4q/N5a: 1-3 ms) -- the flip's legs 2036 ms against
#: ~1400. Unset = on; 0/false/no/off = tokenized on the loop as before.
OFF_LOOP_ENV = "FLLIPER_PDFLIP_PLE_HINT_OFF_LOOP"


def hint_off_loop_on(env=None) -> bool:
    import os

    env = os.environ if env is None else env
    return str(env.get(OFF_LOOP_ENV, "1") or "1").strip().lower() not in ("0", "false", "no", "off")


async def build_ple_prefetch_hint_off_loop(body: dict, **kwargs):
    """``(hint_or_None, ms, off_loop)``: :func:`build_ple_prefetch_hint_sync`
    in a worker thread (switch on) so the event loop keeps taking the
    group's control RPCs -- the wake among them -- while a long prompt is
    tokenized. A hint is a prefetch: arriving after the wake costs its gain
    at most, never the wake."""
    import asyncio
    import time as _time

    t0 = _time.perf_counter()
    if hint_off_loop_on():
        hint = await asyncio.to_thread(build_ple_prefetch_hint_sync, body, **kwargs)
        off = True
    else:
        hint = build_ple_prefetch_hint_sync(body, **kwargs)
        off = False
    return hint, (_time.perf_counter() - t0) * 1000.0, off


async def build_ple_prefetch_hint(body: dict, **kwargs):
    """The original coroutine form (the work inside is synchronous)."""
    return build_ple_prefetch_hint_sync(body, **kwargs)


def build_ple_prefetch_hint_sync(
    body: dict,
    *,
    serving_chat: Any,
    serving_completion: Any,
    encode: Callable[[str], List[int]],
    raw_request: Any = None,
    serving_anthropic: Any = None,
):
    """Tokenize the leg-1 body the way its POST will be tokenized; returns a
    ``PlePrefetchHintReqInput`` or None (not a leg-1 shape)."""
    from flliper.srt.managers.io_struct import PlePrefetchHintReqInput

    path = str(body.get("path") or "")
    payload = dict(body.get("payload") or {})
    rid = payload.get("rid")
    if not rid or path not in _LEG1_PATHS:
        return None
    if path == "/generate":
        ids = payload.get("input_ids")
        if ids is None:
            text = payload.get("text")
            if not isinstance(text, str):
                return None
            ids = encode(text)
    else:
        from flliper.srt.entrypoints.openai.protocol import (
            ChatCompletionRequest,
            CompletionRequest,
        )

        if path == "/v1/messages":
            if serving_anthropic is None:
                return None
            from flliper.srt.entrypoints.anthropic.protocol import (
                AnthropicMessagesRequest,
            )

            req = serving_anthropic._convert_to_chat_completion_request(
                AnthropicMessagesRequest(**payload)
            )
            serving = serving_chat
        elif path == "/v1/chat/completions":
            req = ChatCompletionRequest(**payload)
            serving = serving_chat
        else:
            req = CompletionRequest(**payload)
            serving = serving_completion
        adapted, _ = serving._convert_to_internal_request(req, raw_request)
        ids = getattr(adapted, "input_ids", None)
        if ids is None:
            text = getattr(adapted, "text", None)
            if not isinstance(text, str):
                return None
            ids = encode(text)
    if ids and isinstance(ids[0], list):  # a batch is not a leg 1
        return None
    return PlePrefetchHintReqInput(rid=str(rid), input_ids=[int(t) for t in ids])
