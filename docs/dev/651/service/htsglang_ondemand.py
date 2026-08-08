"""#651: an on-demand front door for the laptop's htsglang server.

WHY A FRONT DOOR AND NOT A SYSTEMD SOCKET UNIT. The model is 21.6 GiB of Q4
GGUF on a 29.5 GiB shared-memory APU. Loading it takes ~2.5 minutes and holds
~22.7 GiB of host RAM through GTT for as long as it is resident, which is the
whole machine. Keeping it hot so a coding assistant can ask one question every
few hours is not a trade anyone would make; refusing the request while it loads
is not one either, because a client that gets a connection error does not come
back. So the front door ACCEPTS the request immediately, starts the backend if
it is parked, HOLDS the request until the backend answers, and parks the
backend again once the machine goes quiet.

`systemd`'s own socket activation would give the accept-and-hold behaviour for
free, but it cannot express the second half: nothing tells it to stop the
service after an idle period, and a `RuntimeMaxSec` would park the model in the
middle of a conversation. The idle timer has to see the requests, so it lives
here.

WHAT PARKING MEANS HERE. The backend process is stopped outright and cold-loaded
on the next request. The cheaper #89 suspend-to-disk park is NOT available for
this checkpoint: its snapshot path raises NotImplementedError on any GGUF MoE
layer, and this model (Qwen3.6-35B-A3B) is a GGUF MoE. See FINAL_651.md 9.2 for
the code path and the scope of that claim. If that restriction is ever lifted, only `_park()` and the boot
script's flags change -- the front door does not care which mechanism freed the
memory.

WHAT IS DELIBERATELY NOT HERE. No request queueing policy, no batching, no
retry of a failed generation. The backend already schedules; this process only
decides whether the backend should exist. Every request is proxied byte for
byte, including streaming responses, so an OpenAI-compatible client cannot tell
it is talking through anything.
"""

from __future__ import annotations

import asyncio
import logging
import os
import signal
import subprocess
import time
from typing import Optional

import aiohttp
from aiohttp import web

LOG = logging.getLogger("ondemand")

#: Public port. Clients (and oh-my-pi) talk to this one.
LISTEN_PORT = int(os.environ.get("HTSGLANG_LISTEN_PORT", "31651"))
#: Port the real server binds. Never exposed; the front door owns the public one.
BACKEND_PORT = int(os.environ.get("HTSGLANG_BACKEND_PORT", "31661"))
BACKEND = f"http://127.0.0.1:{BACKEND_PORT}"

#: Idle seconds before the model is parked. The default is short on purpose:
#: on this machine a resident model costs the user their RAM, and a cold load
#: is expensive but bounded, so the bias is toward giving the machine back.
IDLE_PARK_SECONDS = float(os.environ.get("HTSGLANG_IDLE_PARK_SECONDS", "60"))

#: How long a held request waits for a cold load before giving up. A cold load
#: measured ~150 s; this leaves room for a slow disk without hanging a client
#: forever behind a backend that is never going to come up.
WAKE_TIMEOUT_SECONDS = float(os.environ.get("HTSGLANG_WAKE_TIMEOUT_SECONDS", "900"))

BOOT_SCRIPT = os.environ.get(
    "HTSGLANG_BOOT_SCRIPT", "/root/651-p2/scripts/boot_ondemand.sh"
)

#: Where the BACKEND's stdout/stderr is kept, one file per load.
BACKEND_LOG_DIR = os.environ.get("HTSGLANG_BACKEND_LOG_DIR", "/root/651-p2/logs")

#: Smallest KV pool this service will accept as "ready".
#:
#: A load on this machine does not merely succeed or fail -- it succeeds
#: with a POOL SIZE decided by how much host RAM was free at that instant,
#: and the range is enormous: 15070 tokens on a good boot, 1081 on a bad
#: one, both reporting /health 200. A 1081-token pool cannot serve a single
#: real request against an 8192-token context, so a server in that state is
#: healthy by every signal it emits and useless in fact. Treating it as a
#: failed load and reloading is the only way the caller ever sees a usable
#: server; without this the front door confidently proxies into one that
#: rejects everything.
MIN_KV_TOKENS = int(os.environ.get("HTSGLANG_MIN_KV_TOKENS", "4096"))

#: Paths that must NOT wake the model and must NOT count as activity.
#: A monitoring probe or a browser tab polling /health would otherwise pin
#: 22 GiB of RAM forever -- the exact failure this service exists to avoid.
NO_WAKE_PATHS = {"/health", "/metrics", "/ondemand/status", "/favicon.ico"}

#: Proxied verbatim; no timeout of our own on the body, because a long
#: generation is not a stall.
_PROXY_TIMEOUT = aiohttp.ClientTimeout(total=None, connect=30, sock_read=None)

_HOP_BY_HOP = {
    "connection",
    "keep-alive",
    "proxy-authenticate",
    "proxy-authorization",
    "te",
    "trailers",
    "transfer-encoding",
    "upgrade",
    "content-length",
    "content-encoding",
}


class Backend:
    """Owns the backend process and the single decision "should it exist".

    All transitions are serialised by one lock. Concurrent first requests are
    the normal case (an editor fires several completions at once) and they must
    produce ONE load, not one per request.
    """

    def __init__(self) -> None:
        self._lock = asyncio.Lock()
        self._proc: Optional[subprocess.Popen] = None
        self._last_activity = time.monotonic()
        self._inflight = 0
        #: Set while a load is running so /ondemand/status can be honest
        #: about the difference between "parked" and "coming up".
        self._loading = False
        self._loads = 0
        self._parks = 0
        self._log_path: Optional[str] = None
        self._kv_tokens: Optional[int] = None
        self._last_load_seconds: Optional[float] = None

    # -- state ------------------------------------------------------------
    def alive(self) -> bool:
        return self._proc is not None and self._proc.poll() is None

    def status(self) -> dict:
        return {
            "state": (
                "loading" if self._loading else ("up" if self.alive() else "parked")
            ),
            "pid": self._proc.pid if self.alive() else None,
            "inflight": self._inflight,
            "idle_seconds": round(time.monotonic() - self._last_activity, 1),
            "idle_park_seconds": IDLE_PARK_SECONDS,
            "loads": self._loads,
            "parks": self._parks,
            "last_load_seconds": self._last_load_seconds,
            "backend_log": self._log_path,
            "kv_tokens": self._kv_tokens,
        }

    def touch(self) -> None:
        self._last_activity = time.monotonic()

    # -- transitions ------------------------------------------------------
    async def ensure_up(self) -> None:
        """Return once the backend answers /health, loading it if needed."""
        if self.alive() and await self._healthy():
            return
        async with self._lock:
            # Re-check inside the lock: another request may have loaded it
            # while this one waited, and a second load would OOM the machine.
            if self.alive() and await self._healthy():
                return
            if self.alive():
                # Process exists but is not healthy -- a crashed or wedged
                # backend. Clear it out rather than proxying into a corpse.
                LOG.warning("backend pid %s unhealthy; restarting", self._proc.pid)
                self._stop_process()
            await self._start()

    @staticmethod
    def _child_env() -> dict:
        """Environment for the backend -- explicitly NOT this process's.

        The unit sets CUDA_VISIBLE_DEVICES="" so that a stray torch import in
        the supervisor can never create a HIP context and hold GTT the model
        needs. That setting must not reach the child: it inherits the empty
        value, the GPU sanity guard's first `.cuda()` raises "No HIP GPUs are
        available", and the boot script exits 1 about four seconds in. The
        failure reads exactly like a model that will not load, which is the
        wrong diagnosis entirely -- so the variable is REMOVED here rather than
        left to chance.
        """
        env = dict(os.environ)
        env.pop("CUDA_VISIBLE_DEVICES", None)
        env["PORT"] = str(BACKEND_PORT)
        return env

    async def _start(self) -> None:
        """Load the model, retrying once if the load lost a race for memory.

        The retry is not defensive padding. This machine loads 22.7 GiB into
        29.5 GiB of RAM shared with the GPU, and the loader's own KV-budget
        check is decided by how much happened to be free at that instant: the
        SAME configuration has produced a 15070-token pool on one boot and been
        refused on the next. The boot script drops the page cache first, which
        removes most of that variance; a single retry covers the rest. A second
        failure is a real refusal and is reported as one rather than looped on,
        because retrying a configuration that genuinely does not fit would turn
        a clear error into a hang.
        """
        last: Optional[Exception] = None
        for attempt in (1, 2, 3):
            try:
                await self._start_once(attempt)
                return
            except web.HTTPBadGateway as exc:
                last = exc
                if attempt < 3:
                    LOG.warning(
                        "load attempt %d failed (%s); retrying", attempt, exc.reason
                    )
                    self._stop_process()
                    await asyncio.sleep(5)
        assert last is not None
        raise last

    async def _start_once(self, attempt: int) -> None:
        t0 = time.monotonic()
        self._loading = True
        LOG.info("loading model (attempt %d, boot script %s)", attempt, BOOT_SCRIPT)
        try:
            # start_new_session so the whole tree (bash -> guards -> python)
            # can be signalled as one group; the boot script execs the server,
            # but the guards that run before it are separate children.
            # The backend's own output goes to a file, NOT to DEVNULL. When a
            # load fails or stalls, the loader's log is the only place that
            # says why -- the supervisor can only see that /health never came
            # up, which is the same symptom for a refused KV budget, a wedged
            # GPU and a slow disk. Discarding it once already cost a debugging
            # round here.
            self._log_path = f"{BACKEND_LOG_DIR}/backend_{time.strftime('%H%M%S')}.log"
            log = open(self._log_path, "ab", buffering=0)
            try:
                self._proc = subprocess.Popen(
                    ["bash", BOOT_SCRIPT],
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    stdin=subprocess.DEVNULL,
                    start_new_session=True,
                    env=self._child_env(),
                )
            finally:
                log.close()
            LOG.info("backend log: %s", self._log_path)
            deadline = time.monotonic() + WAKE_TIMEOUT_SECONDS
            while time.monotonic() < deadline:
                if self._proc.poll() is not None:
                    rc = self._proc.returncode
                    self._proc = None
                    raise web.HTTPBadGateway(
                        reason=f"model failed to load (boot script exit {rc})"
                    )
                if await self._healthy():
                    tokens = await self._kv_pool_tokens()
                    if tokens is not None and tokens < MIN_KV_TOKENS:
                        raise web.HTTPBadGateway(
                            reason=(
                                f"model loaded with only {tokens} KV tokens "
                                f"(minimum {MIN_KV_TOKENS}); the load lost the "
                                f"memory lottery and cannot serve"
                            )
                        )
                    self._last_load_seconds = round(time.monotonic() - t0, 1)
                    self._loads += 1
                    self._kv_tokens = tokens
                    LOG.info(
                        "model ready in %.1f s (kv pool %s tokens)",
                        self._last_load_seconds,
                        tokens,
                    )
                    self.touch()
                    return
                await asyncio.sleep(2)
            self._stop_process()
            raise web.HTTPGatewayTimeout(
                reason=f"model did not become ready within {WAKE_TIMEOUT_SECONDS:.0f}s"
            )
        finally:
            self._loading = False

    async def park(self, force: bool = False) -> None:
        """Stop the backend, but only if it is STILL idle once we hold the lock.

        Re-checking here is not belt-and-braces, it is the whole correctness of
        the idle timer. A load takes longer than the idle window, so while one
        is in progress the watcher sees an "alive" process with no in-flight
        requests and an idle clock well past the limit, and calls this. The
        call then blocks on the lock the loader holds -- and would, the instant
        the model finished loading, park it before the request that triggered
        the load was ever served. The conditions have to be re-read on this
        side of the lock, where they mean something.
        """
        async with self._lock:
            if not self.alive():
                return
            if not force and self._inflight:
                return
            if not force and self.idle_for() < IDLE_PARK_SECONDS:
                return
            LOG.info("parking model after %.0fs idle", self.idle_for())
            self._stop_process()
            self._parks += 1

    def _stop_process(self) -> None:
        proc = self._proc
        self._proc = None
        if proc is None or proc.poll() is not None:
            return
        try:
            # Signal the GROUP: the boot script's children must go too, or a
            # stale server keeps the GPU and the next load OOMs.
            os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
        except (ProcessLookupError, PermissionError):
            pass
        try:
            proc.wait(timeout=30)
        except subprocess.TimeoutExpired:
            LOG.warning("backend did not exit on SIGTERM; killing")
            try:
                os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
            except (ProcessLookupError, PermissionError):
                pass
            try:
                proc.wait(timeout=15)
            except subprocess.TimeoutExpired:
                LOG.error("backend pid %s will not die", proc.pid)

    async def _kv_pool_tokens(self) -> Optional[int]:
        """The KV pool the backend actually got, or None if it cannot be read.

        None means "no honest number", and the caller admits the load rather
        than refusing on a figure it does not have -- refusing a working server
        because a status endpoint changed shape would be the worse failure.
        """
        try:
            timeout = aiohttp.ClientTimeout(total=10)
            async with aiohttp.ClientSession(timeout=timeout) as s:
                async with s.get(f"{BACKEND}/get_server_info") as r:
                    if r.status != 200:
                        return None
                    info = await r.json()
            value = info.get("max_total_num_tokens")
            return int(value) if value is not None else None
        except Exception:  # noqa: BLE001
            return None

    async def _healthy(self) -> bool:
        try:
            timeout = aiohttp.ClientTimeout(total=5)
            async with aiohttp.ClientSession(timeout=timeout) as s:
                async with s.get(f"{BACKEND}/health") as r:
                    return r.status == 200
        except Exception:  # noqa: BLE001 - any failure means "not ready"
            return False

    # -- in-flight accounting --------------------------------------------
    def enter(self) -> None:
        self._inflight += 1
        self.touch()

    def leave(self) -> None:
        self._inflight = max(0, self._inflight - 1)
        self.touch()

    def idle_for(self) -> float:
        return time.monotonic() - self._last_activity


backend = Backend()


async def idle_watcher(_app: web.Application) -> None:
    """Park the model once the machine has been quiet long enough."""
    try:
        while True:
            await asyncio.sleep(5)
            if (
                backend.alive()
                and backend._inflight == 0
                and backend.idle_for() >= IDLE_PARK_SECONDS
            ):
                await backend.park()
    except asyncio.CancelledError:
        pass


async def handle_status(_request: web.Request) -> web.Response:
    return web.json_response(backend.status())


async def handle_health(_request: web.Request) -> web.Response:
    """The FRONT DOOR's health, deliberately not the model's.

    Answering 200 while the model is parked is the correct answer to "is the
    service working": it is, and it will load on demand. Reporting the model's
    state here would make every health check a wake-up call.
    """
    return web.json_response({"status": "ok", "backend": backend.status()["state"]})


async def handle_proxy(request: web.Request) -> web.StreamResponse:
    path = request.path
    if path in NO_WAKE_PATHS:
        raise web.HTTPNotFound()

    # Read the body BEFORE the load, not after. A cold load holds this handler
    # for ~150 s, and leaving the request unread for that long keeps the
    # client's send buffered the whole time for no reason -- the body is a few
    # KB of JSON and the client has already sent it.
    body = await request.read()

    await backend.ensure_up()
    backend.enter()
    try:
        headers = {
            k: v for k, v in request.headers.items() if k.lower() not in _HOP_BY_HOP
        }
        headers.pop("Host", None)
        url = f"{BACKEND}{path}"
        if request.query_string:
            url = f"{url}?{request.query_string}"

        async with aiohttp.ClientSession(timeout=_PROXY_TIMEOUT) as session:
            async with session.request(
                request.method, url, data=body or None, headers=headers
            ) as upstream:
                out_headers = {
                    k: v
                    for k, v in upstream.headers.items()
                    if k.lower() not in _HOP_BY_HOP
                }
                response = web.StreamResponse(
                    status=upstream.status, headers=out_headers
                )
                await response.prepare(request)
                # Stream in chunks so SSE tokens reach the client as they are
                # produced; buffering here would turn streaming into batch.
                async for chunk in upstream.content.iter_chunked(8192):
                    await response.write(chunk)
                    backend.touch()
                await response.write_eof()
                return response
    finally:
        backend.leave()


def build_app() -> web.Application:
    app = web.Application(client_max_size=1024**3)
    app.router.add_get("/health", handle_health)
    app.router.add_get("/ondemand/status", handle_status)
    app.router.add_route("*", "/{tail:.*}", handle_proxy)

    async def _start_watcher(a: web.Application) -> None:
        a["idle_task"] = asyncio.create_task(idle_watcher(a))

    async def _stop_watcher(a: web.Application) -> None:
        a["idle_task"].cancel()
        await asyncio.gather(a["idle_task"], return_exceptions=True)
        # Park on shutdown regardless of the idle clock: leaving 22 GiB
        # resident after the unit stops is exactly the state this service
        # exists to prevent.
        await backend.park(force=True)

    app.on_startup.append(_start_watcher)
    app.on_cleanup.append(_stop_watcher)
    return app


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    LOG.info(
        "front door on :%d -> backend :%d, park after %.0fs idle",
        LISTEN_PORT,
        BACKEND_PORT,
        IDLE_PARK_SECONDS,
    )
    web.run_app(build_app(), host="0.0.0.0", port=LISTEN_PORT, print=None)


if __name__ == "__main__":
    main()
