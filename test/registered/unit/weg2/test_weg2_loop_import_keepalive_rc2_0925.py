# SPDX-License-Identifier: Apache-2.0
"""RC2 Blocker A (27B boot weg2rc2, 2026-09-24 23:18Z) / NF H75+H78 (x172-x175):
the first LONG request of the boot got HTTP 503, `WEG2 leg1 rid=weg2-0-1 failed:
Server disconnected`.

The chain, from the front log:
1. resolve_x_live imported sglang.srt.weg2.launcher lazily at the first flip's
   end -- ON the event loop. The import pulls transformers and the mem_cache
   stack; the loop stood still 5.36 s (`WEG2-HOST RATE-GAP gap_s=5.9`).
2. P's uvicorn closed the idle pooled connection after its keep-alive
   (SGLANG_TIMEOUT_KEEP_ALIVE, default 5 s).
3. SGLANG_WEG2_CTL_KICK_AFTER_FLIP sent leg 1 1 ms after FLIP done; aiohttp
   reused the pooled socket whose FIN the stalled loop had not read yet.
4. Leg 1 has no retry (none is safe: P dedups a rid only while it is in
   flight), so the client got 503.

UNIFY S2 (desk/27b-unified-0926): both lines fixed this independently -- the
27B line with a synchronous preload in Front.__init__ and a 0.6x pool
keep-alive (d0fba8955f), the NF line with a prewarm in worker threads at
startup (GC paused, heap frozen) and keep-alive = server - min(1 s, half)
(33b69563f5, ba19830a17). ONE form, the NF one (it also keeps the loop free
of the import's GC passes). This file keeps the 27B line's two instruments on
that form:
* the serving-loop probe (flips both ways, the X re-solve, the admitter, the
  controller, the three routes) in a fresh interpreter, after the startup
  prewarm has run: no module is imported for the first time inside the loop
  (it found retain_publish, now in FLIP_PATH_IMPORTS);
* the failure itself: a server that closes idle keep-alive connections, a loop
  stalled past that time, then a request on the front's own session -- the
  pooled socket must not be handed out.
The keep-alive arithmetic is pinned in test_weg2_front_loop_blockers_h78.py.

Hermetic: no GPU, no boot, no shared-memory segment; one local HTTP server.
"""
from __future__ import annotations

import asyncio
import builtins
import json
import os
import subprocess
import sys
import threading
import time

import pytest

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

from aiohttp import web  # noqa: E402

from sglang.srt.weg2 import front as front_mod  # noqa: E402
from sglang.test.ci.ci_register import register_cpu_ci  # noqa: E402

register_cpu_ci(est_time=40, suite="base-a-test-cpu")

X = 4096


def _front():
    return front_mod.Front("http://p", "http://d", "D", "blockerA", "", 0, 0, {}, 45.0,
                           tp_prefill_max_tokens=X, weight_chunks=2)


# ------------------------------------------------------------ (a) no import in the loop
LOOP_PROBE = r'''
import asyncio, builtins, json, os, sys, time
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
from sglang.srt.weg2 import front as fm
MIB = 1024 * 1024

def mk(**kw):
    f = fm.Front("http://p", "http://d", "D", "probe", "", 0, 0, {}, 45.0,
                 tp_prefill_max_tokens=4096, weight_chunks=2, **kw)
    f.stops = []
    async def rpc(g, path, body, timeout):
        if path == "/flush_cache":
            return 200, "{}"
        await asyncio.sleep(0.001)
        tags = tuple((body or {}).get("tags", ()))
        return 200, json.dumps({"per_tag": {t: [MIB, 1.0] for t in tags},
                                "critical_path": "rank=0 card=GPU-x ms=1"})
    f.rpc = rpc
    f.do_stop = lambda n, d: f.stops.append((n, d))
    return f

class Req:
    def __init__(self, payload, path="/v1/chat/completions"):
        self._p, self.path = payload, path
    async def json(self):
        return self._p

# main() builds the Front(s) BEFORE web.run_app starts the loop -- so do we.
F, G = mk(), mk(vision="transient")
seen, depth, armed = [], [0], [False]
orig = builtins.__import__
def hook(name, globals=None, locals=None, fromlist=(), level=0):
    new, top = name not in sys.modules, depth[0] == 0
    depth[0] += 1
    try:
        return orig(name, globals, locals, fromlist, level)
    finally:
        depth[0] -= 1
        if armed[0] and top and new:
            seen.append(name)
builtins.__import__ = hook

async def body():
    # the startup prewarm (Front.startup starts these as tasks) -- awaited, so
    # the probe asks what the SERVING loop imports after it
    await F._prewarm_launcher_import()
    await F._prewarm_flip_path_imports()
    armed[0] = True
    F._x_samples["r_d"].append(900.0)
    F._x_samples["r_p"].append(3000.0)
    await F.flip("D", "P")          # flip end -> note_x_sample -> resolve_x_live
    await F.flip("P", "D")
    img = {"type": "image_url", "image_url": {"url": "data:image/png;base64,iVBORw0KGgo="}}
    for payload in ({"model": "m", "messages": [{"role": "user", "content": "hi " * 40}], "max_tokens": 4},
                    {"model": "m", "messages": [{"role": "user", "content": [{"type": "text", "text": "what"}, img]}]},
                    {"prompt": "x" * 40000}):
        t = asyncio.ensure_future(G.handle_generate(Req(payload)))
        for _ in range(300):
            await asyncio.sleep(0)
            if t.done() or G.queue:
                break
        t.cancel()
        await asyncio.gather(t, return_exceptions=True)
        G.queue.clear()
    tasks = [asyncio.ensure_future(F.d_admitter()), asyncio.ensure_future(F.controller())]
    await asyncio.sleep(0.4)
    for t in tasks:
        t.cancel()
    await asyncio.gather(*tasks, return_exceptions=True)
    armed[0] = False

asyncio.run(body())
print("RESULT " + json.dumps({"first_imports_in_loop": seen, "stops": [s[0] for s in F.stops],
                              "flips": F.counters.get("flips", 0)}))
'''


def test_the_serving_loop_starts_no_import():
    env = dict(os.environ, CUDA_VISIBLE_DEVICES="", PYTHONDONTWRITEBYTECODE="1")
    out = subprocess.run([sys.executable, "-c", LOOP_PROBE], env=env, capture_output=True,
                         text=True, timeout=600)
    lines = [ln for ln in out.stdout.splitlines() if ln.startswith("RESULT ")]
    assert lines, f"probe did not finish (rc={out.returncode}): {out.stderr[-2000:]}"
    res = json.loads(lines[-1][len("RESULT "):])
    assert res["stops"] == [] and res["flips"] == 2, res
    assert res["first_imports_in_loop"] == [], (
        "a module was imported for the first time INSIDE the event loop -- boot weg2rc2 lost "
        f"5.36 s of loop there and the next request its connection: {res['first_imports_in_loop']}")


def test_the_front_is_built_before_the_loop_and_prewarms_at_startup():
    """main() builds the Front before web.run_app, and startup starts the
    prewarm tasks (launcher + flip-path modules, incl. retain_publish)."""
    import inspect

    src = inspect.getsource(front_mod.main)
    assert src.index("front = Front(") < src.index("web.run_app(")
    st = inspect.getsource(front_mod.Front.startup)
    assert "self._prewarm_launcher_import()" in st
    assert "self._prewarm_flip_path_imports()" in st
    assert "sglang.srt.weg2.retain_publish" in front_mod.Front.FLIP_PATH_IMPORTS
    assert "sglang.srt.weg2.launcher" not in front_mod.Front.FLIP_PATH_IMPORTS  # own task, GC paused


def _keepalive_server(keepalive_s):
    """An HTTP server in its OWN thread and loop that closes idle keep-alive
    connections after ``keepalive_s`` -- what P's uvicorn does after
    SGLANG_TIMEOUT_KEEP_ALIVE. Returns (url, stop)."""
    info, ready = {}, threading.Event()
    loop = asyncio.new_event_loop()

    async def handler(request):
        await request.read()
        return web.json_response({"ok": True})

    async def run():
        app = web.Application()
        app.router.add_post("/generate", handler)
        runner = web.AppRunner(app, keepalive_timeout=keepalive_s)
        await runner.setup()
        site = web.TCPSite(runner, "127.0.0.1", 0)
        await site.start()
        info["port"] = site._server.sockets[0].getsockname()[1]  # noqa: SLF001
        info["runner"] = runner
        ready.set()

    def target():
        asyncio.set_event_loop(loop)
        loop.run_until_complete(run())
        loop.run_forever()

    threading.Thread(target=target, daemon=True).start()
    assert ready.wait(20), "test server did not start"

    def stop():
        asyncio.run_coroutine_threadsafe(info["runner"].cleanup(), loop).result(10)
        loop.call_soon_threadsafe(loop.stop)

    return f"http://127.0.0.1:{info['port']}/generate", stop


def test_a_stalled_loop_never_gets_the_socket_the_server_closed(monkeypatch):
    """The weg2rc2 failure, on the front's OWN session (built by startup):
    request, the server's keep-alive runs out while the loop is stalled, the
    very next request must not go out on the closed socket."""
    server_keepalive = 1.0
    monkeypatch.setenv("SGLANG_TIMEOUT_KEEP_ALIVE", "1")   # the groups' setting
    url, stop = _keepalive_server(server_keepalive)

    async def body():
        f = _front()
        app = {}
        await f.startup(app)
        for t in list(app.values()):        # only the session is under test
            if isinstance(t, asyncio.Task):
                t.cancel()
        await asyncio.gather(*[t for t in app.values() if isinstance(t, asyncio.Task)],
                             return_exceptions=True)
        try:
            async with f.session.post(url, json={"a": 1}) as r:
                assert r.status == 200
                await r.read()
            time.sleep(server_keepalive + 1.5)   # the loop stands still (the lazy import)
            async with f.session.post(url, json={"a": 2}) as r:  # no yield before this
                return r.status
        finally:
            await f.session.close()

    try:
        assert asyncio.run(body()) == 200
    finally:
        stop()
