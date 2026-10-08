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
  Bodies pass through byte-for-byte apart from the one shim below.
* It does not log credentials. ``Authorization``, ``x-api-key`` and every other
  header are forwarded but never written to the log, at any level.

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
"""

from __future__ import annotations

import argparse
import json
import logging
from typing import Iterable

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

# Model-id suffix that selects the thinking arm of a local model.
THINKING_ALIAS_SUFFIX = "-think"
ADAPTIVE_THINKING = {"type": "adaptive"}

# Typed keys: aiohttp warns on bare-string app state.
LOCAL_MODELS = web.AppKey("local_models", set)
THINKING_ALIASES = web.AppKey("thinking_aliases", dict)
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


def _apply_thinking_shim(body: bytes) -> bytes:
    """Force ``thinking: disabled`` on the plain local-model route.

    This OVERWRITES a client-supplied ``thinking`` value. Claude Code attaches
    its own thinking config to subagent requests, so a fill-in-only shim let
    the local model think despite the no-thinking policy (observed live:
    "Thought for 14s" blocks and a stray ``</think>`` in visible text). The
    ``-think`` alias is the only route to the thinking arm; the plain id must
    always be the no-thinking arm, or the two arms become indistinguishable.
    """
    try:
        payload = json.loads(body)
    except (ValueError, UnicodeDecodeError):
        return body
    if not isinstance(payload, dict):
        return body
    payload["thinking"] = {"type": "disabled"}
    return json.dumps(payload).encode()


def _apply_thinking_alias(body: bytes, target_model: str) -> bytes:
    """Rewrite ``model`` to ``target_model`` and force adaptive thinking.

    Unlike the shim this OVERWRITES an existing ``thinking`` value: the alias
    is the client's explicit request for the thinking arm, and a client default
    that won here would make the arm indistinguishable from the plain id.
    """
    payload = _rewrite_model(body, target_model)
    try:
        obj = json.loads(payload)
    except (ValueError, UnicodeDecodeError):
        return payload
    if not isinstance(obj, dict):
        return payload
    obj["thinking"] = dict(ADAPTIVE_THINKING)
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


def _extract_model(body: bytes) -> str | None:
    try:
        payload = json.loads(body)
    except (ValueError, UnicodeDecodeError):
        return None
    if not isinstance(payload, dict):
        return None
    model = payload.get("model")
    return model if isinstance(model, str) else None


def create_app(
    local_models: Iterable[str],
    upstream_base: str = DEFAULT_UPSTREAM,
    local_base: str = DEFAULT_LOCAL,
    apply_shim: bool = True,
) -> web.Application:
    """Build the proxy application.

    ``local_models`` is matched exactly against the request's ``model`` field.
    Each one additionally gets a ``<id>-think`` alias that routes to the same
    backend with adaptive thinking forced on.
    """
    app = web.Application(client_max_size=1024**3)
    app[LOCAL_MODELS] = set(local_models)
    app[THINKING_ALIASES] = {m + THINKING_ALIAS_SUFFIX: m for m in app[LOCAL_MODELS]}
    app[UPSTREAM_BASE] = upstream_base.rstrip("/")
    app[LOCAL_BASE] = local_base.rstrip("/")
    app[APPLY_SHIM] = apply_shim
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
        return web.json_response(body)

    async def proxy(request: web.Request) -> web.StreamResponse:
        body = await request.read()
        model = _extract_model(body)
        alias_target = (
            request.app[THINKING_ALIASES].get(model) if model is not None else None
        )
        to_local = alias_target is not None or (
            model is not None and model in request.app[LOCAL_MODELS]
        )

        if to_local:
            base = request.app[LOCAL_BASE]
            if alias_target is not None:
                # The thinking arm. /v1/messages gets model + forced adaptive
                # thinking; every other path only needs the id un-aliased.
                if request.path == "/v1/messages":
                    body = _apply_thinking_alias(body, alias_target)
                else:
                    body = _rewrite_model(body, alias_target)
            # Only /v1/messages carries the max_tokens budget the stray
            # thinking block used to consume; leave other endpoints verbatim.
            elif request.app[APPLY_SHIM] and request.path == "/v1/messages":
                body = _apply_thinking_shim(body)
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
        + "' alias that reaches the same backend with adaptive thinking "
        "forced on.",
    )
    parser.add_argument(
        "--no-thinking-shim",
        action="store_true",
        help="do not fill in thinking:disabled when the client omits it. Safe "
        "once the server carries the absent-means-disabled front fix.",
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
    )
    logger.info(
        "listening on %s:%d, local models %s (thinking aliases %s) -> %s, "
        "everything else -> %s",
        args.host,
        args.port,
        sorted(args.local_model),
        sorted(app[THINKING_ALIASES]),
        args.local_base,
        args.upstream_base,
    )
    web.run_app(app, host=args.host, port=args.port, print=None)


if __name__ == "__main__":
    main()
