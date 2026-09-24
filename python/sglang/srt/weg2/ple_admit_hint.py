"""fnFL2 H43: the front's hint that a long request is coming to group P.

WHY. The PLE read of a request's first chunk can start at its admission on P
(``qwen4_exp_ple_admit``) -- but in the flip case P never admits it early: the
front keeps a BATCH arrival in its own queue while D is awake, flips D->P, and
posts leg 1 only once P is awake. x146 (24.09.): weg2-2-6 queued at the front
11:31:27.48, P's wake leg issued 11:31:27.70 and done 11:31:29.44, P's first
gather began ~11:31:29.5 and read cold for 4.47 s.

WHAT. When the front queues a BATCH request while P is not the awake group, it
POSTs the leg-1 body to P's ``/weg2/ple_prefetch_hint`` (fire and forget). P's
HTTP server tokenizes it exactly as the leg-1 POST will be tokenized (the same
OpenAI serving conversion) and pushes a ``PlePrefetchHintReqInput`` down the
same socket its wake RPC takes -- so a hint that arrives before the wake RPC
is handled by P's scheduler before the wake, and PP0's pread workers read the
first chunk while P wakes. The later leg 1 of the same rid finds the admission
(``confirmed`` when the tokens are equal, re-admitted otherwise).

A hint is only a prefetch: nothing is queued, nothing is answered but 200, a
failed or late hint costs the gain and nothing else. Front switch
``SGLANG_WEG2_PLE_ADMIT_HINT`` (default on); P needs
``SGLANG_QWEN4_PLE_PREFETCH_ADMIT`` (the admission itself).
"""

from __future__ import annotations

import logging
from typing import Any, Callable, List, Optional

logger = logging.getLogger(__name__)

HINT_PATH = "/weg2/ple_prefetch_hint"
_LEG1_PATHS = ("/v1/chat/completions", "/v1/completions", "/generate")


# ---------------------------------------------------------------- front side


def ple_hint_wanted(*, awake: Optional[str], skip_leg1: bool = False) -> bool:
    """A BATCH arrival P will prefill (leg 1) while P is not awake: the one
    case in which P learns of the request only after its wake. (With P awake
    the leg 1 is posted within a controller tick, or it waits for a free P
    slot while the request ahead of it on P is admitted there already.)"""
    return awake != "P" and not skip_leg1


def ple_hint_body(path: str, payload: dict) -> Optional[dict]:
    """The hint body: the leg-1 path and payload (the payload is the one leg 1
    will POST: same rid, ``max_tokens`` 1, no stream)."""
    if path not in _LEG1_PATHS:
        return None
    body = dict(payload)
    body.pop("stream", None)
    body.pop("stream_options", None)
    if not body.get("rid"):
        return None
    return {"path": path, "payload": body}


# ---------------------------------------------------------------- P side


async def build_ple_prefetch_hint(
    body: dict,
    *,
    serving_chat: Any,
    serving_completion: Any,
    encode: Callable[[str], List[int]],
    raw_request: Any = None,
):
    """Tokenize the leg-1 body the way its POST will be tokenized; returns a
    ``PlePrefetchHintReqInput`` or None (not a leg-1 shape)."""
    from sglang.srt.managers.io_struct import PlePrefetchHintReqInput

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
        from sglang.srt.entrypoints.openai.protocol import (
            ChatCompletionRequest,
            CompletionRequest,
        )

        if path == "/v1/chat/completions":
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
