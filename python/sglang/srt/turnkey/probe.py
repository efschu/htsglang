# Copyright 2026 SGLang Team
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# ==============================================================================
"""Probes: is the port open, does the API answer, and does it GENERATE.

The third one is the point. #622/#649 established that this stack can reach a
state where the process lives, the socket accepts, ``/health`` returns 200 and
no request ever produces a token. Every cheap liveness signal reports that
server as fine. The only probe that disagrees is one that asks for output and
insists on receiving it.

The generation probe is therefore written to be un-fool-able in the specific
ways the cheap ones are:

* it POSTs to ``/generate`` and requires non-empty decoded text back, not a
  200 and not a well-formed envelope;
* it is BOUNDED by an explicit timeout, because the wedge's signature is
  precisely a request that never returns -- an unbounded probe would hang the
  watchdog instead of reporting the hang (the wedge-detection trap: a
  detector that shares the failure mode it detects);
* it is deterministic and tiny (``temperature 0``, a handful of tokens) so
  that repeating it every couple of minutes costs the served model close to
  nothing.
"""

from __future__ import annotations

import dataclasses
import json
import socket
import urllib.error
import urllib.request
from typing import Callable, Optional

__all__ = ["ProbeResult", "port_open", "api_ok", "generation_ok",
           "DEFAULT_PROMPT"]

#: Short, deterministic, and content-free. The probe asserts that tokens come
#: back, never what they say -- asserting on content would make the watchdog
#: restart production over a sampling change.
DEFAULT_PROMPT = "Reply with the single word: ok"


@dataclasses.dataclass(frozen=True)
class ProbeResult:
    ok: bool
    detail: str
    elapsed_s: float = 0.0


def port_open(host: str, port: int, timeout: float = 0.25) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.settimeout(timeout)
        return s.connect_ex((host, port)) == 0


#: Reachability is probed on /get_model_info ALONE, deliberately not /health.
#: On this stack ``SGLANG_ENABLE_HEALTH_ENDPOINT_GENERATION`` defaults to True
#: (``environ.py:1630``), so ``/health`` runs a real ``max_new_tokens=1``
#: generation bounded by ``SGLANG_HEALTH_CHECK_TIMEOUT`` (default 20 s,
#: ``http_server.py:206``). That makes /health a THIRD thing: neither cheap
#: reachability nor the watchdog's own generation verdict, but a blocking
#: call whose budget the caller does not control.
#:
#: Probing it with a short timeout is an outright defect, and a live one: the
#: existing watchdog polls /health with ``curl --max-time 10`` against a
#: server allowed 20 s to answer, which manufactures false HTTP_DEAD verdicts
#: on a merely busy server -- the false-positive class its later hedges
#: (DEGRADED_HTTP, log-mtime cross-checks) were added to paper over.
#:
#: Splitting the layers removes the problem instead of hedging it:
#: /get_model_info answers from the HTTP process without touching the
#: scheduler, and generation gets its own probe with its own explicit budget.
LIVENESS_PATH = "/get_model_info"


def api_ok(base_url: str, timeout: float = 5.0, opener=None) -> ProbeResult:
    """Cheap reachability -- is an HTTP server answering there at all.

    NOT liveness. The module docstring is emphatic about why: a wedged server
    answers this perfectly. See :data:`LIVENESS_PATH` for why /health is
    excluded from this layer.
    """
    from sglang.srt.planner import server_state as ss

    p = ss.probe_http(base_url, LIVENESS_PATH, timeout=timeout, opener=opener)
    return ProbeResult(ok=bool(p.ok), detail=p.error or f"{p.path} ok")


def _post_json(url: str, payload: dict, timeout: float) -> str:
    req = urllib.request.Request(
        url, data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"}, method="POST")
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return resp.read().decode("utf-8", "replace")


def generation_ok(base_url: str, timeout: float = 60.0,
                  max_new_tokens: int = 8, prompt: str = DEFAULT_PROMPT,
                  poster: Optional[Callable[[str, dict, float], str]] = None,
                  clock: Callable[[], float] = None) -> ProbeResult:
    """Ask for tokens and require tokens.

    Returns ``ok=False`` on timeout, transport error, HTTP error, malformed
    body, or an empty completion. Every one of those is a lane that cannot
    serve a user, which is the only question the watchdog is asking.
    """
    import time as _time

    clock = clock or _time.monotonic
    poster = poster or _post_json
    url = base_url.rstrip("/") + "/generate"
    payload = {
        "text": prompt,
        "sampling_params": {"max_new_tokens": int(max_new_tokens),
                            "temperature": 0.0},
    }
    t0 = clock()
    try:
        body = poster(url, payload, timeout)
    except socket.timeout:
        return ProbeResult(False, f"timeout after {timeout:.0f}s (wedge "
                                  f"signature)", clock() - t0)
    except urllib.error.HTTPError as e:
        return ProbeResult(False, f"HTTP {e.code}", clock() - t0)
    except (urllib.error.URLError, OSError) as e:
        # URLError wraps socket.timeout on some paths; name it as such so the
        # log distinguishes "hung" from "refused".
        reason = getattr(e, "reason", e)
        if isinstance(reason, socket.timeout):
            return ProbeResult(False, f"timeout after {timeout:.0f}s (wedge "
                                      f"signature)", clock() - t0)
        return ProbeResult(False, f"transport: {reason}", clock() - t0)
    elapsed = clock() - t0

    try:
        data = json.loads(body)
    except ValueError:
        return ProbeResult(False, "response was not json", elapsed)

    text = _extract_text(data)
    if not text or not text.strip():
        return ProbeResult(False, "empty completion (HTTP ok, no tokens)",
                           elapsed)
    return ProbeResult(True, f"generated {len(text.strip())} chars in "
                             f"{elapsed:.1f}s", elapsed)


def _extract_text(data) -> str:
    """/generate answers with a dict, or a list of them for batched input."""
    if isinstance(data, list):
        return "".join(_extract_text(d) for d in data)
    if isinstance(data, dict):
        t = data.get("text")
        if isinstance(t, str):
            return t
    return ""
