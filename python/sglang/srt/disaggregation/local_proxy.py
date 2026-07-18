"""Minimal single-node PD-disaggregation proxy.

A deliberately small (dependency-free beyond aiohttp) front for a fixed
prefill server + decode server pair on one machine. Chosen over
``sglang_router`` for the single-node case (#99 design ruling): no service
discovery, trivially inspectable, and the interface (plain HTTP, health
passthrough, per-phase metric counters exposed at ``/metrics_summary``) is
designed so a rig dashboard can later poll it directly.

Usage:
    python -m sglang.srt.disaggregation.local_proxy \
        --prefill http://127.0.0.1:30021 --decode http://127.0.0.1:30022 \
        --bootstrap-port 8998 --host 127.0.0.1 --port 8100

Every ``/generate`` (and OpenAI-compat ``/v1/*``) request is duplicated:
the prefill server receives it with ``max_new_tokens=1`` semantics implied
by disaggregation-mode prefill, the decode server streams/returns the real
completion. Both copies carry the same freshly minted ``bootstrap_room``.
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import random
import time
from urllib.parse import urlparse

from aiohttp import ClientSession, ClientTimeout, web

logger = logging.getLogger(__name__)

FORWARD_PATHS = (
    "/generate",
    "/v1/completions",
    "/v1/chat/completions",
)


class Metrics:
    """Per-phase counters, dashboard-friendly (all plain numbers)."""

    def __init__(self):
        self.requests = 0
        self.failures = 0
        self.prefill_latency_sum_s = 0.0
        self.e2e_latency_sum_s = 0.0
        self.last_prefill_latency_s = 0.0
        self.last_e2e_latency_s = 0.0

    def as_dict(self):
        n = max(1, self.requests)
        return {
            "requests": self.requests,
            "failures": self.failures,
            "avg_prefill_latency_s": round(self.prefill_latency_sum_s / n, 4),
            "avg_e2e_latency_s": round(self.e2e_latency_sum_s / n, 4),
            "last_prefill_latency_s": round(self.last_prefill_latency_s, 4),
            "last_e2e_latency_s": round(self.last_e2e_latency_s, 4),
        }


class PDProxy:
    def __init__(self, prefill_url: str, decode_url: str, bootstrap_port: int):
        self.prefill_url = prefill_url.rstrip("/")
        self.decode_url = decode_url.rstrip("/")
        self.bootstrap_host = urlparse(self.prefill_url).hostname
        self.bootstrap_port = bootstrap_port
        self.metrics = Metrics()
        self.session: ClientSession | None = None

    async def startup(self, app):
        self.session = ClientSession(timeout=ClientTimeout(total=3600))

    async def cleanup(self, app):
        if self.session:
            await self.session.close()

    async def handle_health(self, request: web.Request) -> web.Response:
        """Health = both backends healthy."""
        results = {}
        for name, url in (("prefill", self.prefill_url), ("decode", self.decode_url)):
            try:
                async with self.session.get(f"{url}/health") as r:
                    results[name] = r.status
            except Exception as e:
                results[name] = f"error: {e}"
        ok = all(v == 200 for v in results.values())
        return web.json_response(results, status=200 if ok else 503)

    async def handle_metrics_summary(self, request: web.Request) -> web.Response:
        return web.json_response(self.metrics.as_dict())

    async def handle_generate(self, request: web.Request) -> web.StreamResponse:
        payload = await request.json()
        room = random.randint(0, 2**63 - 1)
        bootstrap = {
            "bootstrap_host": self.bootstrap_host,
            "bootstrap_port": self.bootstrap_port,
            "bootstrap_room": room,
        }
        prefill_payload = {**payload, **bootstrap}
        decode_payload = {**payload, **bootstrap}
        # The prefill side never generates more than the first token.
        sp = dict(prefill_payload.get("sampling_params") or {})
        sp["max_new_tokens"] = 1
        prefill_payload["sampling_params"] = sp
        prefill_payload.pop("stream", None)

        self.metrics.requests += 1
        t0 = time.time()

        async def run_prefill():
            async with self.session.post(
                f"{self.prefill_url}{request.path}", json=prefill_payload
            ) as r:
                body = await r.read()
                if r.status != 200:
                    raise RuntimeError(
                        f"prefill returned {r.status}: {body[:400]!r}"
                    )
                self.metrics.last_prefill_latency_s = time.time() - t0
                self.metrics.prefill_latency_sum_s += (
                    self.metrics.last_prefill_latency_s
                )

        prefill_task = asyncio.create_task(run_prefill())
        try:
            async with self.session.post(
                f"{self.decode_url}{request.path}", json=decode_payload
            ) as r:
                if payload.get("stream"):
                    resp = web.StreamResponse(status=r.status)
                    resp.content_type = r.content_type
                    await resp.prepare(request)
                    async for chunk in r.content.iter_any():
                        await resp.write(chunk)
                    await resp.write_eof()
                else:
                    body = await r.read()
                    resp = web.Response(
                        body=body, status=r.status, content_type=r.content_type
                    )
                await prefill_task
                self.metrics.last_e2e_latency_s = time.time() - t0
                self.metrics.e2e_latency_sum_s += self.metrics.last_e2e_latency_s
                return resp
        except Exception:
            self.metrics.failures += 1
            prefill_task.cancel()
            raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prefill", required=True, help="prefill server URL")
    parser.add_argument("--decode", required=True, help="decode server URL")
    parser.add_argument("--bootstrap-port", type=int, default=8998)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8100)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)
    proxy = PDProxy(args.prefill, args.decode, args.bootstrap_port)
    app = web.Application(client_max_size=1024**3)
    app.on_startup.append(proxy.startup)
    app.on_cleanup.append(proxy.cleanup)
    app.router.add_get("/health", proxy.handle_health)
    app.router.add_get("/metrics_summary", proxy.handle_metrics_summary)
    for path in FORWARD_PATHS:
        app.router.add_post(path, proxy.handle_generate)
    logger.info(
        "PD proxy on %s:%s -> prefill=%s decode=%s",
        args.host, args.port, args.prefill, args.decode,
    )
    web.run_app(app, host=args.host, port=args.port, print=None)


if __name__ == "__main__":
    main()
