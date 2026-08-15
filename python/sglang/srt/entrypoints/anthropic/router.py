"""Model-aware split proxy for the Anthropic Messages wire format.

Claude Code binds its endpoint process-globally: ``ANTHROPIC_BASE_URL`` and its
siblings are read once per process and there is no per-subagent endpoint key in
the subagent frontmatter schema. So a session cannot keep its parent turns on
api.anthropic.com while a single subagent runs against the local htsglang
server -- unless something in front of both makes that decision per request.

This is that something. It listens locally, forwards every request verbatim to
api.anthropic.com, EXCEPT requests whose ``model`` is in ``--local-model`` --
those go to the local htsglang Anthropic front instead. Claude Code passes the
``model`` string from ``--agents`` to the wire unvalidated, so naming a local
model id in an agent definition is enough to route just that agent here.

Two properties this proxy deliberately does NOT have:

* It does not translate. Both sides speak the same Anthropic Messages format;
  htsglang serves it natively (``entrypoints/http_server.py`` ``/v1/messages``).
  Bodies pass through byte-for-byte apart from the shims below, and the
  UPSTREAM path is never parsed at all -- it stays a pure byte pipe.
* It does not log credentials. ``Authorization``, ``x-api-key`` and every other
  header are forwarded but never written to the log, at any level.

The one RESPONSE edit is the message_start usage repair, and it exists because
of how a client accounts tokens while a turn is still running. Anthropic's
``message_start`` carries the real ``usage.input_tokens``; the local front
emits it as ``0`` on purpose, shipping ``message_start`` before the backend has
produced anything so the client sees a message id immediately, and correcting
the totals in the closing ``message_delta``. That trade is invisible to a
client that only reads the final message -- but Claude Code recomputes its
per-agent token readout from EVERY assistant event, as
``input_tokens + cache_creation + cache_read`` (latest, not summed) plus
cumulative ``output_tokens``. A zero-valued ``message_start`` therefore zeroes
the readout for the whole turn, and every intermediate content block recorded
before the final one is stored with ``{"input_tokens": 0, "output_tokens": 0}``.
The router repairs this by asking the local front's own ``count_tokens``
endpoint (same tokenizer, CPU only, ~30 ms) for the prompt length in parallel
with the real request, and filling the value into ``message_start`` if -- and
only if -- the server reported zero. Once the server learns to emit a non-zero
``input_tokens`` itself, this repair sees a non-zero value and becomes a no-op
without any change here.

The one body edit is the thinking shim: an Anthropic client omits ``thinking``
entirely when it does not want extended thinking, and a server booted with
``--reasoning-parser`` used to answer such a request with a leading thinking
block that ate the whole ``max_tokens`` budget, so a tool round trip never got
emitted. ``feat/anthropic-front-conformance`` fixes that IN the front (absent
== disabled); the shim here makes the router work against a server that
predates the fix and becomes a no-op once the fixed server is running, because
it only ever fills in a field the client did not send.

The mirror image of that shim is the thinking ALIAS. Claude Code has no way to
ask a subagent for extended thinking either -- the frontmatter carries a
``model`` key and nothing else -- so the only client-side lever on the thinking
mode is, again, the model string. For every id in ``--local-model`` the router
therefore also answers to ``<id>-think``: same local backend, ``model``
rewritten back to the real id, and ``thinking`` FORCED to
``{"type": "adaptive"}`` on ``/v1/messages``. Adaptive rather than ``enabled``
because ``enabled`` requires ``budget_tokens >= 1024`` while adaptive means
"thinking on, the model decides how much", which is the arm an A/B against the
no-thinking default wants. Forced rather than filled-in because naming the
alias IS the explicit request: a client value that silently won would turn the
thinking arm back into the default arm without anything in the log saying so.

WHICH ARM IS THE DEFAULT IS A PER-MODEL FACT, so it is a flag
(``--local-thinking``) rather than a constant. On Qwen3.6 thinking was measured
to cost tokens without improving answers, so the plain id forces it off. On
Qwen3.8 thinking helps materially, so the plain id forces it ON together with a
reasoning effort. Because that flips which arm the plain id represents, the
router also answers to ``<id>-nothink``: with the default on, a client whose
only lever is the model string would otherwise have no route back to the cheap
arm.

The effort knob has one non-obvious encoding, and it is worth stating here
because it looks like a bug otherwise: the STRONGEST effort is expressed by
sending NO effort field. Qwen3.8's chat template accepts ``xhigh`` (its
default), ``medium`` and ``low`` and raises on anything else, while the
Anthropic front maps ``output_config.effort`` onto the OpenAI
``reasoning_effort`` Literal, which has no ``xhigh`` and collapses it to
``max`` -- which the template then rejects. So ``high``, ``xhigh`` and ``max``
are all request failures, and omitting the field is the only encoding that
reaches the template's strongest arm. The router normalizes client efforts onto
that reality, so a client asking to think as hard as possible gets the hardest
thinking rather than a 400.

Finally, ``--policy-file`` re-reads the arm from a small JSON file whenever its
mtime changes. This process is the endpoint every Claude Code session points
at, so restarting it drops live turns; the policy file is how the arm gets
retuned without that.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
from typing import Iterable, Optional

import aiohttp
from aiohttp import web

logger = logging.getLogger(__name__)

DEFAULT_UPSTREAM = "https://api.anthropic.com"
DEFAULT_LOCAL = "http://127.0.0.1:30030"
DEFAULT_LISTEN_HOST = "127.0.0.1"
DEFAULT_LISTEN_PORT = 30099

# Never forwarded: connection-scoped per RFC 9110, or recomputed by aiohttp.
_HOP_BY_HOP = frozenset(
    {
        "connection",
        "content-length",
        "keep-alive",
        "proxy-authenticate",
        "proxy-authorization",
        "te",
        "trailer",
        "transfer-encoding",
        "upgrade",
        "host",
    }
)

STATS_PATH = "/__router/stats"

# message_start usage repair (see module docstring).
COUNT_TOKENS_PATH = "/v1/messages/count_tokens"
# Bound on how long message_start may be held back waiting for the token count.
# The count runs concurrently with the real request and normally lands in tens
# of milliseconds; on timeout the frame goes out unrepaired rather than late.
COUNT_TOKENS_TIMEOUT_SECONDS = 3.0
# Request fields that determine the prompt length. Sampling knobs are dropped:
# count_tokens rejects some of them, and none of them change the token count.
# ``thinking`` is NOT a sampling knob here -- it selects a chat template, so
# omitting it makes the count disagree with the server's own prompt_tokens by
# the size of the thinking preamble (measured: 15 vs 13 on the -think alias).
COUNT_TOKENS_FIELDS = frozenset(
    {"model", "messages", "system", "tools", "tool_choice", "thinking"}
)
# Guards on the framing window. We only ever re-frame the head of a local
# stream, and only until message_start is found; past these bounds the body
# reverts to a raw byte pipe so a surprising stream shape cannot make the
# router buffer without limit.
MAX_FRAMED_HEAD_BYTES = 262144
MAX_FRAMED_EVENTS = 8
SSE_FRAME_SEPARATOR = b"\n\n"

# Model-id suffix that selects the thinking arm of a local model.
THINKING_ALIAS_SUFFIX = "-think"
# Mirror suffix that selects the NO-thinking arm. It exists because the
# default arm is configurable (--local-thinking): once a backend defaults to
# thinking ON, the plain id is no longer the off arm, and a client whose only
# lever is the model string would have no way back to it.
NOTHINK_ALIAS_SUFFIX = "-nothink"
ADAPTIVE_THINKING = {"type": "adaptive"}
DISABLED_THINKING = {"type": "disabled"}

# Reasoning-effort levels this router can ASK FOR, strongest first.
#
# The vocabulary is the model template's, not the Anthropic SDK's, and the
# difference is load-bearing. Qwen3.8's chat template accepts exactly
# ``xhigh`` (its own default), ``medium`` and ``low``, and calls
# ``raise_exception`` on anything else. The Anthropic front maps
# ``output_config.effort`` onto the OpenAI ``reasoning_effort`` Literal, which
# has no ``xhigh`` and therefore collapses it to ``max`` -- a value the
# template then rejects. So ``high``, ``xhigh`` and ``max`` all fail the
# request, and the STRONGEST effort is reachable only by sending NO effort at
# all and letting the template's own default stand.
#
# That is why ``xhigh`` here means "omit the field" rather than "send the
# string": it is the one encoding that actually reaches the strongest arm
# through the existing chain, with no backend change and no server restart.
EFFORT_XHIGH = "xhigh"
EFFORT_LEVELS = (EFFORT_XHIGH, "medium", "low")
# Client-supplied effort values, normalized onto what the backend template
# accepts. Everything at or above "high" means "think as hard as you can",
# which is ``xhigh``, which is the omitted field.
EFFORT_NORMALIZATION = {
    "low": "low",
    "medium": "medium",
    "high": EFFORT_XHIGH,
    "xhigh": EFFORT_XHIGH,
    "max": EFFORT_XHIGH,
}

# Typed keys: aiohttp warns on bare-string app state.
LOCAL_MODELS = web.AppKey("local_models", set)
THINKING_ALIASES = web.AppKey("thinking_aliases", dict)
NOTHINK_ALIASES = web.AppKey("nothink_aliases", dict)
POLICY = web.AppKey("policy", dict)
POLICY_FILE = web.AppKey("policy_file", object)
UPSTREAM_BASE = web.AppKey("upstream_base", str)
LOCAL_BASE = web.AppKey("local_base", str)
APPLY_SHIM = web.AppKey("apply_shim", bool)
STATS = web.AppKey("stats", dict)
SESSION = web.AppKey("session", aiohttp.ClientSession)


def _filter_headers(headers: Iterable[tuple[str, str]]) -> dict[str, str]:
    return {k: v for k, v in headers if k.lower() not in _HOP_BY_HOP}


def _request_headers(headers: Iterable[tuple[str, str]]) -> dict[str, str]:
    """Client headers, plus a pinned Accept-Encoding.

    We forward the response body without decompressing it, so the content
    coding the client asked for is the one it gets. But aiohttp adds its own
    ``Accept-Encoding: gzip, deflate`` when the header is absent, which would
    make us hand a gzipped body to a client that never advertised gzip.
    Pinning ``identity`` in that case keeps the pass-through honest; a client
    that DID ask has its own value forwarded untouched.
    """
    out = _filter_headers(headers)
    if not any(k.lower() == "accept-encoding" for k in out):
        out["Accept-Encoding"] = "identity"
    return out


def _normalize_client_effort(payload: dict) -> None:
    """Rewrite a client ``output_config.effort`` onto a value the model accepts.

    Without this, a client that asks for the strongest reasoning gets a FAILED
    request rather than the strongest reasoning: ``high``/``xhigh``/``max`` all
    arrive at the template as something it rejects (see ``EFFORT_LEVELS``).
    Normalizing keeps the client's intent -- "think as hard as you can" -- and
    delivers it through the encoding that works, which is the absent field.

    Only ever applied on the local route. Upstream bodies are never parsed.
    """
    oc = payload.get("output_config")
    if not isinstance(oc, dict):
        return
    effort = oc.get("effort")
    if not isinstance(effort, str):
        return
    normalized = EFFORT_NORMALIZATION.get(effort.lower())
    if normalized is None:
        return
    if normalized == EFFORT_XHIGH:
        # The strongest arm IS the omitted field. Drop the key, and drop an
        # output_config that carried nothing else, so the body stays minimal.
        oc.pop("effort", None)
        if not oc:
            payload.pop("output_config", None)
    elif normalized != effort:
        oc["effort"] = normalized


def _client_effort(payload: dict) -> Optional[str]:
    """The effort the client asked for, or None if it expressed no preference.

    Read BEFORE any normalization. Normalizing first would erase the evidence
    that a preference existed: the strongest efforts normalize to an ABSENT
    field, which is indistinguishable from "the client said nothing" -- and the
    router would then helpfully fill in its own weaker default, turning a
    request for maximum reasoning into a downgrade.
    """
    oc = payload.get("output_config")
    if isinstance(oc, dict) and isinstance(oc.get("effort"), str):
        return oc["effort"]
    return None


def _set_effort(payload: dict, effort: str) -> None:
    """Overwrite the request's effort with the router's configured value."""
    oc = payload.get("output_config")
    if effort == EFFORT_XHIGH:
        # Strongest == omit the field; see EFFORT_LEVELS.
        if isinstance(oc, dict):
            oc.pop("effort", None)
            if not oc:
                payload.pop("output_config", None)
        return
    if not isinstance(oc, dict):
        oc = {}
        payload["output_config"] = oc
    oc["effort"] = effort


def _apply_thinking_policy(
    body: bytes,
    enabled: bool,
    effort: str,
    force: bool,
    target_model: Optional[str] = None,
) -> bytes:
    """Stamp the thinking arm onto a ``/v1/messages`` body.

    ``thinking`` is always OVERWRITTEN, on both arms. Claude Code attaches its
    own thinking config to subagent requests, so a fill-in-only rule let the
    client silently decide the arm -- observed live as "Thought for 14s" blocks
    on the no-thinking route. The arm is the deployment's decision (or the
    alias's, when one is named); the client's lever is the model string.

    ``effort`` is the opposite: filled in on the plain id (``force=False``) so
    a per-request ``output_config.effort`` still wins, forced on the alias.
    """
    payload = _rewrite_model(body, target_model) if target_model else body
    try:
        obj = json.loads(payload)
    except (ValueError, UnicodeDecodeError):
        return payload
    if not isinstance(obj, dict):
        return payload
    obj["thinking"] = dict(ADAPTIVE_THINKING if enabled else DISABLED_THINKING)
    if enabled:
        if _client_effort(obj) is not None and not force:
            # The client expressed a preference and this is the fill-in arm,
            # so its value stands -- normalized onto something the model can
            # actually accept, rather than passed through into a 400.
            _normalize_client_effort(obj)
        else:
            _set_effort(obj, effort)
    else:
        # Thinking is off, so there is no reasoning section for an effort to
        # modulate. Leaving one in the body would only risk a template
        # rejection for no behavioural gain. Other output_config fields
        # (task_budget) are the client's and stay.
        oc = obj.get("output_config")
        if isinstance(oc, dict):
            oc.pop("effort", None)
            if not oc:
                obj.pop("output_config", None)
    return json.dumps(obj).encode()


def _rewrite_model(body: bytes, target_model: str) -> bytes:
    """Replace the request's ``model`` with the id the local front knows.

    Applied on every path, not just ``/v1/messages``: the local server has no
    ``-think`` checkpoint, so ``count_tokens`` and friends must be un-aliased
    too or they answer 404.
    """
    try:
        payload = json.loads(body)
    except (ValueError, UnicodeDecodeError):
        return body
    if not isinstance(payload, dict):
        return body
    payload["model"] = target_model
    return json.dumps(payload).encode()


def _count_tokens_body(body: bytes) -> bytes | None:
    """Derive a ``count_tokens`` request from a ``/v1/messages`` body.

    Returns ``None`` for anything the repair does not apply to: a
    non-streaming request (whose single response already carries correct
    usage), or a body we cannot read.
    """
    try:
        payload = json.loads(body)
    except (ValueError, UnicodeDecodeError):
        return None
    if not isinstance(payload, dict) or not payload.get("stream"):
        return None
    counted = {k: v for k, v in payload.items() if k in COUNT_TOKENS_FIELDS}
    if "model" not in counted or "messages" not in counted:
        return None
    return json.dumps(counted).encode()


async def _count_input_tokens(
    session: aiohttp.ClientSession, base: str, body: bytes
) -> Optional[int]:
    """Ask the local front how long the prompt is. Never raises.

    A failure here must not affect the real request, so every error path
    degrades to ``None`` and the message_start frame is forwarded unrepaired.
    """
    try:
        async with session.post(
            base + COUNT_TOKENS_PATH,
            data=body,
            headers={
                "Content-Type": "application/json",
                "anthropic-version": "2023-06-01",
                # The shared session runs auto_decompress=False, so a
                # compressed reply would arrive unreadable here.
                "Accept-Encoding": "identity",
            },
        ) as response:
            if response.status != 200:
                return None
            payload = json.loads(await response.read())
    except (aiohttp.ClientError, asyncio.TimeoutError, ValueError, UnicodeDecodeError):
        return None
    if not isinstance(payload, dict):
        return None
    value = payload.get("input_tokens")
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        return None
    return value


async def _resolve_count(task: "asyncio.Task") -> Optional[int]:
    """Await the token count under a deadline. Never raises.

    On timeout the task is cancelled and the caller proceeds unrepaired: a
    late message_start is worse for the client than a zero one.
    """
    try:
        return await asyncio.wait_for(task, COUNT_TOKENS_TIMEOUT_SECONDS)
    except (asyncio.CancelledError, Exception):
        return None


def _repair_message_start(
    frame: bytes, input_tokens: Optional[int]
) -> tuple[bytes, bool]:
    """Fill ``usage.input_tokens`` into a message_start frame reporting zero.

    Returns the frame (repaired or untouched) and whether this frame was the
    message_start, which is what ends the framing window. ``input_tokens`` of
    ``None`` means the count was unavailable: the frame is then only
    inspected, never rewritten.

    The frame is re-serialised only on the repair path, so a frame we decide
    not to touch stays byte-identical -- including its ``event:`` line and any
    explicit nulls the front emitted deliberately.
    """
    marker = b"data: "
    start = frame.find(marker)
    if start == -1:
        return frame, False
    payload_start = start + len(marker)
    payload_end = frame.find(b"\n", payload_start)
    if payload_end == -1:
        payload_end = len(frame)
    try:
        event = json.loads(frame[payload_start:payload_end])
    except (ValueError, UnicodeDecodeError):
        return frame, False
    if not isinstance(event, dict) or event.get("type") != "message_start":
        return frame, False

    message = event.get("message")
    if not isinstance(message, dict):
        return frame, True
    usage = message.get("usage")
    if not isinstance(usage, dict):
        return frame, True
    # Only ever fill in a missing number. A server that already reports its
    # own input_tokens wins, which is what makes this repair self-retiring.
    if input_tokens is None or usage.get("input_tokens"):
        return frame, True

    usage["input_tokens"] = input_tokens
    repaired = frame[:payload_start] + json.dumps(event).encode() + frame[payload_end:]
    return repaired, True


def _extract_model(body: bytes) -> str | None:
    try:
        payload = json.loads(body)
    except (ValueError, UnicodeDecodeError):
        return None
    if not isinstance(payload, dict):
        return None
    model = payload.get("model")
    return model if isinstance(model, str) else None


def _effective_policy(app: web.Application) -> dict:
    """Flags, with any live policy-file override applied on top."""
    policy = dict(app[POLICY])
    policy.update(app[POLICY_FILE].overrides())
    return policy


def _load_policy_file(path: str) -> dict:
    """Read a policy override file, returning only the keys it validly sets.

    A malformed or unreadable file is IGNORED rather than fatal: this is a
    live-tuning hook on a process that must not fall over, and the flags it
    overrides are already a working configuration.
    """
    try:
        with open(path) as fh:
            obj = json.load(fh)
    except (OSError, ValueError) as e:
        logger.warning("policy file %s not applied: %s", path, e)
        return {}
    if not isinstance(obj, dict):
        logger.warning("policy file %s not applied: not a JSON object", path)
        return {}
    out: dict = {}
    thinking = obj.get("thinking")
    if thinking in ("on", "off"):
        out["thinking_enabled"] = thinking == "on"
    elif thinking is not None:
        logger.warning("policy file %s: bad thinking value %r", path, thinking)
    effort = obj.get("effort")
    if effort in EFFORT_LEVELS:
        out["effort"] = effort
    elif effort is not None:
        logger.warning("policy file %s: bad effort value %r", path, effort)
    return out


class _PolicyFile:
    """Re-reads the policy file when its mtime changes.

    The point of this class is that the deployment can retune the thinking arm
    WITHOUT restarting the router. The router holds no other mutable state and
    has no reload signal, so a restart is the only alternative -- and this
    process is the endpoint every Claude Code session is pointed at, so a
    restart drops live turns. Cost is one ``stat`` per local /v1/messages
    request, which is noise next to the model call it precedes.
    """

    def __init__(self, path: Optional[str]):
        self.path = path
        self._mtime: Optional[float] = None
        self._overrides: dict = {}

    def overrides(self) -> dict:
        if self.path is None:
            return {}
        try:
            mtime = os.stat(self.path).st_mtime
        except OSError:
            if self._mtime is not None:
                logger.warning("policy file %s disappeared: keeping last values", self.path)
            return self._overrides
        if mtime != self._mtime:
            self._mtime = mtime
            self._overrides = _load_policy_file(self.path)
            logger.info("policy file %s reloaded: %s", self.path, self._overrides)
        return self._overrides


def create_app(
    local_models: Iterable[str],
    upstream_base: str = DEFAULT_UPSTREAM,
    local_base: str = DEFAULT_LOCAL,
    apply_shim: bool = True,
    thinking_enabled: bool = False,
    effort: str = EFFORT_XHIGH,
    policy_file: Optional[str] = None,
) -> web.Application:
    """Build the proxy application.

    ``local_models`` is matched exactly against the request's ``model`` field.
    Each one additionally gets a ``<id>-think`` alias (thinking forced on) and
    a ``<id>-nothink`` alias (thinking forced off), so both arms stay reachable
    from a client whose only lever is the model string, whichever way
    ``thinking_enabled`` sets the default for the plain id.
    """
    app = web.Application(client_max_size=1024**3)
    app[LOCAL_MODELS] = set(local_models)
    app[THINKING_ALIASES] = {m + THINKING_ALIAS_SUFFIX: m for m in app[LOCAL_MODELS]}
    app[NOTHINK_ALIASES] = {m + NOTHINK_ALIAS_SUFFIX: m for m in app[LOCAL_MODELS]}
    app[UPSTREAM_BASE] = upstream_base.rstrip("/")
    app[LOCAL_BASE] = local_base.rstrip("/")
    app[APPLY_SHIM] = apply_shim
    app[POLICY] = {"thinking_enabled": thinking_enabled, "effort": effort}
    app[POLICY_FILE] = _PolicyFile(policy_file)
    app[STATS] = {"local": 0, "upstream": 0, "errors": 0}

    async def _session(app: web.Application):
        # auto_decompress=False keeps the response body byte-identical, so a
        # Content-Encoding we forward stays true.
        connector = aiohttp.TCPConnector(limit=0)
        app[SESSION] = aiohttp.ClientSession(
            connector=connector,
            auto_decompress=False,
            timeout=aiohttp.ClientTimeout(total=None, sock_connect=30),
        )
        yield
        await app[SESSION].close()

    app.cleanup_ctx.append(_session)

    async def stats(request: web.Request) -> web.Response:
        body = dict(request.app[STATS])
        body["local_models"] = sorted(request.app[LOCAL_MODELS])
        body["thinking_aliases"] = sorted(request.app[THINKING_ALIASES])
        body["nothink_aliases"] = sorted(request.app[NOTHINK_ALIASES])
        body["policy"] = _effective_policy(request.app)
        return web.json_response(body)

    async def proxy(request: web.Request) -> web.StreamResponse:
        body = await request.read()
        model = _extract_model(body)
        think_target = (
            request.app[THINKING_ALIASES].get(model) if model is not None else None
        )
        nothink_target = (
            request.app[NOTHINK_ALIASES].get(model) if model is not None else None
        )
        alias_target = think_target if think_target is not None else nothink_target
        to_local = alias_target is not None or (
            model is not None and model in request.app[LOCAL_MODELS]
        )

        if to_local:
            base = request.app[LOCAL_BASE]
            policy = _effective_policy(request.app)
            if alias_target is not None:
                # An alias is an explicit arm request: thinking forced either
                # way, effort forced with it. Only /v1/messages carries a
                # thinking field; every other path just needs the id
                # un-aliased, or the local front answers 404.
                if request.path == "/v1/messages":
                    body = _apply_thinking_policy(
                        body,
                        enabled=think_target is not None,
                        effort=policy["effort"],
                        force=True,
                        target_model=alias_target,
                    )
                else:
                    body = _rewrite_model(body, alias_target)
            # Only /v1/messages carries the max_tokens budget the stray
            # thinking block used to consume; leave other endpoints verbatim.
            elif request.app[APPLY_SHIM] and request.path == "/v1/messages":
                body = _apply_thinking_policy(
                    body,
                    enabled=policy["thinking_enabled"],
                    effort=policy["effort"],
                    force=False,
                )
            request.app[STATS]["local"] += 1
        else:
            base = request.app[UPSTREAM_BASE]
            request.app[STATS]["upstream"] += 1

        # model is a client-supplied identifier, never a credential.
        logger.info(
            "%s %s -> %s (model=%s)",
            request.method,
            request.path,
            "local" if to_local else "upstream",
            model,
        )

        url = base + request.raw_path
        headers = _request_headers(request.headers.items())

        # Start the token count NOW, so it overlaps the real request instead
        # of adding a round trip in front of it. Only the local streaming
        # /v1/messages path can need the repair.
        count_task: Optional[asyncio.Task] = None
        if to_local and request.path == "/v1/messages":
            count_body = _count_tokens_body(body)
            if count_body is not None:
                count_task = asyncio.ensure_future(
                    _count_input_tokens(request.app[SESSION], base, count_body)
                )

        try:
            async with request.app[SESSION].request(
                request.method,
                url,
                headers=headers,
                data=body if body else None,
            ) as upstream_response:
                out = web.StreamResponse(
                    status=upstream_response.status,
                    headers=_filter_headers(upstream_response.headers.items()),
                )
                await out.prepare(request)

                # Repair the head of the stream only, and only when there is
                # a count to repair it with. Every other response -- upstream,
                # non-streaming, errors -- goes through the untouched byte
                # pipe below, exactly as before.
                if count_task is not None and upstream_response.status == 200:
                    tokens = await _resolve_count(count_task)
                    pending = b""
                    events_seen = 0
                    framing = True
                    while framing:
                        # readany() is the primitive behind iter_any(); an
                        # empty read means the upstream body is finished.
                        chunk = await upstream_response.content.readany()
                        if not chunk:
                            break
                        pending += chunk
                        # Drain whole SSE frames out of the head buffer.
                        while framing:
                            cut = pending.find(SSE_FRAME_SEPARATOR)
                            if cut == -1:
                                break
                            frame = pending[: cut + len(SSE_FRAME_SEPARATOR)]
                            pending = pending[cut + len(SSE_FRAME_SEPARATOR) :]
                            events_seen += 1
                            frame, is_start = _repair_message_start(frame, tokens)
                            await out.write(frame)
                            # message_start is the only frame this touches, so
                            # stop re-framing the moment it is past.
                            if is_start or events_seen >= MAX_FRAMED_EVENTS:
                                framing = False
                        if len(pending) > MAX_FRAMED_HEAD_BYTES:
                            break
                    if pending:
                        await out.write(pending)

                async for chunk in upstream_response.content.iter_any():
                    await out.write(chunk)
                await out.write_eof()
                return out
        except aiohttp.ClientError as e:
            request.app[STATS]["errors"] += 1
            # Anthropic-shaped envelope so the client's error handling works.
            logger.warning(
                "proxy to %s failed: %s", "local" if to_local else "upstream", e
            )
            return web.json_response(
                {
                    "type": "error",
                    "error": {
                        "type": "api_error",
                        "message": f"router could not reach the {'local' if to_local else 'upstream'} endpoint: {e}",
                    },
                },
                status=502,
            )
        finally:
            # A count started for a request that never reached the repair path
            # (non-200, transport error, client disconnect) must not be left
            # pending on the event loop.
            if count_task is not None and not count_task.done():
                count_task.cancel()

    app.router.add_get(STATS_PATH, stats)
    app.router.add_route("*", "/{tail:.*}", proxy)
    return app


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--host", default=DEFAULT_LISTEN_HOST)
    parser.add_argument("--port", type=int, default=DEFAULT_LISTEN_PORT)
    parser.add_argument("--upstream-base", default=DEFAULT_UPSTREAM)
    parser.add_argument("--local-base", default=DEFAULT_LOCAL)
    parser.add_argument(
        "--local-model",
        action="append",
        default=[],
        help="model id to route to --local-base; repeatable. Everything else "
        "goes to --upstream-base. Each id also gains a '"
        + THINKING_ALIAS_SUFFIX
        + "' alias (thinking forced on) and a '"
        + NOTHINK_ALIAS_SUFFIX
        + "' alias (thinking forced off), reaching the same backend.",
    )
    parser.add_argument(
        "--no-thinking-shim",
        action="store_true",
        help="do not stamp a thinking arm onto the plain local-model route at "
        "all, leaving the client's own value to reach the backend.",
    )
    parser.add_argument(
        "--local-thinking",
        choices=("off", "on"),
        default="off",
        help="thinking arm for the PLAIN local-model id. 'off' forces "
        "thinking:disabled (the Qwen3.6 policy: measured to cost tokens "
        "without improving answers). 'on' forces adaptive thinking plus "
        "--local-effort (the Qwen3.8 policy: thinking helps materially). "
        "The '"
        + THINKING_ALIAS_SUFFIX
        + "'/'"
        + NOTHINK_ALIAS_SUFFIX
        + "' aliases reach the other arm either way.",
    )
    parser.add_argument(
        "--local-effort",
        choices=EFFORT_LEVELS,
        default=EFFORT_XHIGH,
        help="reasoning effort requested when thinking is on. 'xhigh' is the "
        "strongest and is sent by OMITTING the effort field, because the "
        "model template's default is xhigh while the Anthropic->OpenAI "
        "mapping cannot express it (it collapses to 'max', which the "
        "template rejects). On the plain id this is a DEFAULT that a "
        "per-request output_config.effort overrides; on the aliases it is "
        "forced.",
    )
    parser.add_argument(
        "--policy-file",
        default=None,
        help="optional JSON file, re-read whenever its mtime changes, that "
        'overrides the thinking arm live: {"thinking": "on"|"off", '
        '"effort": "xhigh"|"medium"|"low"}. Exists so the arm can be '
        "retuned without restarting this process, which is the endpoint "
        "live client sessions are pointed at.",
    )
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s router %(message)s"
    )
    if not args.local_model:
        logger.warning("no --local-model given: every request goes upstream")

    app = create_app(
        local_models=args.local_model,
        upstream_base=args.upstream_base,
        local_base=args.local_base,
        apply_shim=not args.no_thinking_shim,
        thinking_enabled=args.local_thinking == "on",
        effort=args.local_effort,
        policy_file=args.policy_file,
    )
    logger.info(
        "listening on %s:%d, local models %s (aliases %s) -> %s, "
        "everything else -> %s; default arm thinking=%s effort=%s%s",
        args.host,
        args.port,
        sorted(args.local_model),
        sorted(list(app[THINKING_ALIASES]) + list(app[NOTHINK_ALIASES])),
        args.local_base,
        args.upstream_base,
        args.local_thinking,
        args.local_effort,
        f" (policy file {args.policy_file})" if args.policy_file else "",
    )
    web.run_app(app, host=args.host, port=args.port, print=None)


if __name__ == "__main__":
    main()
