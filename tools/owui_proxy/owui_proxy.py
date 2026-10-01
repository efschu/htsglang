"""OpenWebUI-Proxy vor der Rig-Front (Nutzer 01.10.): OpenWebUI soll Stundenwechsel NF <-> 27B, Boots und
Tode nicht bemerken.

  OpenWebUI --> :30032 (dieser Proxy) --> 127.0.0.1:30030 (Tunnel) --> Front (NF oder 27B, was gerade laeuft)

Drei Dinge:
  1. /v1/models nie leer: jedes je gesehene Modell bleibt in der Liste (state-Datei), dazu der Alias rig-auto.
     Der Server bedient, was gerade laeuft; die Modell-ID der Anfrage wird an das laufende Modell angeglichen.
  2. Halten statt Fehler: ist kein Backend oben, bleibt die Anfrage offen (bis HOLD_S); ein Stream bekommt
     SSE-Kommentare als Keepalive, damit Browser und OpenWebUI nicht abbrechen.
  3. Fortsetzen mitten im Stream: bricht das Backend ab, wartet der Proxy auf das naechste und schickt die
     Teilantwort als angefangene Assistant-Nachricht (continue_final_message) -- der Client sieht einen
     durchgehenden Stream.  Bricht es in der Denkphase ab, wird die Fortsetzung bis </think> wieder als
     reasoning_content ausgeliefert.

Aufruf: python3 owui_proxy.py [--port 30032] [--upstream http://127.0.0.1:30030] [--hold-s 1800]
"""
import argparse
import asyncio
import copy
import json
import logging
import os
import time

import aiohttp
from aiohttp import web

log = logging.getLogger("owui_proxy")

ALIAS = "rig-auto"
HOP_HEADERS = {"host", "content-length", "transfer-encoding", "connection", "keep-alive", "accept-encoding"}


class Cfg:
    def __init__(self, upstream: str, hold_s: float, keepalive_s: float, poll_s: float, state_file: str,
                 max_resumes: int):
        self.upstream = upstream.rstrip("/")
        self.hold_s = hold_s
        self.keepalive_s = keepalive_s
        self.poll_s = poll_s
        self.state_file = state_file
        self.max_resumes = max_resumes


class Catalog:
    """Every model id the upstream ever served, kept across proxy restarts."""

    def __init__(self, path: str):
        self.path = path
        self.models = {}
        try:
            with open(path) as f:
                self.models = json.load(f)
        except (OSError, ValueError):
            self.models = {}

    def add(self, entries: list) -> None:
        changed = False
        for e in entries:
            mid = e.get("id")
            if mid and mid not in self.models:
                self.models[mid] = {k: e[k] for k in ("id", "object", "owned_by", "max_model_len") if k in e}
                changed = True
        if changed and self.path:
            tmp = self.path + ".tmp"
            with open(tmp, "w") as f:
                json.dump(self.models, f, indent=1)
            os.replace(tmp, self.path)

    def listing(self, live_ids: list) -> dict:
        data = [{"id": ALIAS, "object": "model", "owned_by": "rig", "live": bool(live_ids)}]
        for mid, e in sorted(self.models.items()):
            data.append(dict(e, object="model", live=mid in live_ids))
        return {"object": "list", "data": data}


class Proxy:
    def __init__(self, cfg: Cfg):
        self.cfg = cfg
        self.catalog = Catalog(cfg.state_file)
        self.session = None

    async def start(self, app):
        self.session = aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=None, sock_connect=5))

    async def stop(self, app):
        await self.session.close()

    # ---- backend state -------------------------------------------------------------------------
    async def live_models(self) -> list:
        """Model ids the upstream serves right now; [] when it is down or booting."""
        try:
            async with self.session.get(self.cfg.upstream + "/v1/models",
                                        timeout=aiohttp.ClientTimeout(total=3)) as r:
                if r.status != 200:
                    return []
                data = (await r.json()).get("data") or []
        except (aiohttp.ClientError, asyncio.TimeoutError, ValueError):
            return []
        self.catalog.add(data)
        return [e.get("id") for e in data if e.get("id")]

    async def wait_backend(self, on_tick=None) -> list:
        """Block until the upstream serves a model or HOLD_S runs out; on_tick() is awaited every keepalive."""
        t_end = time.monotonic() + self.cfg.hold_s
        t_ka = time.monotonic()
        while True:
            ids = await self.live_models()
            if ids:
                return ids
            if time.monotonic() >= t_end:
                return []
            if on_tick and time.monotonic() - t_ka >= self.cfg.keepalive_s:
                await on_tick()
                t_ka = time.monotonic()
            await asyncio.sleep(self.cfg.poll_s)

    # ---- handlers ------------------------------------------------------------------------------
    async def models(self, request):
        return web.json_response(self.catalog.listing(await self.live_models()))

    async def chat(self, request):
        body = await request.json()
        if body.get("stream"):
            return await self._chat_stream(request, body)
        return await self._forward_held(request, json.dumps(body).encode(), "/v1/chat/completions", body)

    async def passthrough(self, request):
        raw = await request.read()
        body = None
        try:
            body = json.loads(raw) if raw else None
        except ValueError:
            pass
        return await self._forward_held(request, raw, request.rel_url.path_qs, body)

    async def _forward_held(self, request, raw: bytes, path: str, body):
        """Non-streaming: hold until a backend is up, retry the whole request after a backend failure."""
        for _ in range(self.cfg.max_resumes + 1):
            ids = await self.wait_backend()
            if not ids:
                return web.json_response({"error": {"message": "no backend within hold time"}}, status=503)
            if isinstance(body, dict) and "model" in body:
                body = dict(body, model=ids[0])
                raw = json.dumps(body).encode()
            hdrs = {k: v for k, v in request.headers.items() if k.lower() not in HOP_HEADERS}
            try:
                async with self.session.request(request.method, self.cfg.upstream + path, data=raw,
                                                headers=hdrs) as r:
                    payload = await r.read()
                    if r.status >= 500:
                        raise aiohttp.ClientError("upstream %d" % r.status)
                    return web.Response(body=payload, status=r.status,
                                        content_type=r.content_type or "application/json")
            except (aiohttp.ClientError, asyncio.TimeoutError) as e:
                log.warning("forward %s failed (%s), holding for the next backend", path, e)
        return web.json_response({"error": {"message": "backend failed repeatedly"}}, status=502)

    async def _chat_stream(self, request, body: dict):
        resp = web.StreamResponse(headers={"Content-Type": "text/event-stream", "Cache-Control": "no-cache",
                                           "X-Accel-Buffering": "no"})
        await resp.prepare(request)
        st = StreamState(body)

        async def keepalive():
            await resp.write(b": waiting for backend\n\n")

        for attempt in range(self.cfg.max_resumes + 1):
            ids = await self.wait_backend(keepalive)
            if not ids:
                await _sse(resp, {"error": {"message": "no backend within hold time"}})
                break
            req_body = st.request_body(ids[0])
            if attempt:
                log.info("resume %s attempt %d: %d chars content, %d chars reasoning", st.id, attempt,
                         len(st.content), len(st.reasoning))
            try:
                done = await self._pump(req_body, st, resp)
            except (aiohttp.ClientError, asyncio.TimeoutError, ConnectionError) as e:
                log.warning("stream %s broke (%s)", st.id, e)
                done = False
            if done:
                break
        await resp.write(b"data: [DONE]\n\n")
        return resp

    async def _pump(self, req_body: dict, st: "StreamState", resp) -> bool:
        """One upstream stream; True when it ended normally ([DONE] or a finish_reason other than abort)."""
        async with self.session.post(self.cfg.upstream + "/v1/chat/completions", json=req_body) as r:
            if r.status != 200:
                raise aiohttp.ClientError("upstream %d: %s" % (r.status, (await r.text())[:200]))
            finished = False
            async for line in r.content:
                line = line.strip()
                if not line.startswith(b"data:"):
                    continue
                data = line[5:].strip()
                if data == b"[DONE]":
                    return finished or True
                try:
                    chunk = json.loads(data)
                except ValueError:
                    continue
                if "error" in chunk:
                    raise aiohttp.ClientError("upstream error chunk: %s" % str(chunk["error"])[:200])
                out, fin = st.absorb(chunk)
                if fin == "abort":
                    raise aiohttp.ClientError("upstream aborted the request")
                if out is not None:
                    await _sse(resp, out)
                if fin:
                    finished = True
            return finished


class StreamState:
    """What the client has received so far, and how to ask the next backend to go on from there."""

    def __init__(self, body: dict):
        self.body = body
        self.content = ""
        self.reasoning = ""
        self.id = None
        self.model = body.get("model")
        self.chunks = 0
        self.resume_in_think = False   # continuation of a break inside the thinking phase

    def request_body(self, live_model: str) -> dict:
        b = copy.deepcopy(self.body)
        b["model"] = live_model
        if not (self.content or self.reasoning):
            return b
        if self.content:
            prefix = ("<think>\n" + self.reasoning + "\n</think>\n\n" if self.reasoning else "") + self.content
            self.resume_in_think = False
        else:
            prefix = "<think>\n" + self.reasoning
            self.resume_in_think = True
        b["messages"] = list(b.get("messages") or []) + [{"role": "assistant", "content": prefix}]
        b["continue_final_message"] = True
        b["add_generation_prompt"] = False
        for k in ("max_tokens", "max_completion_tokens"):
            if isinstance(b.get(k), int):
                b[k] = max(1, b[k] - self.chunks)
        return b

    def absorb(self, chunk: dict):
        """Record the chunk, rewrite it for the client; returns (chunk_or_None, finish_reason)."""
        self.id = self.id or chunk.get("id")
        chunk["id"] = self.id
        if self.model:
            chunk["model"] = self.model
        fin = None
        for ch in chunk.get("choices") or []:
            d = ch.get("delta") or {}
            if self.resume_in_think and d.get("reasoning_content"):
                # the backend's reasoning parser splits the continuation itself (htsglang front, 01.10.)
                self.resume_in_think = False
            if self.resume_in_think and d.get("content"):
                text = d.pop("content")
                if "</think>" in text:
                    before, after = text.split("</think>", 1)
                    d["reasoning_content"] = (d.get("reasoning_content") or "") + before
                    if after.lstrip("\n"):
                        d["content"] = after.lstrip("\n")
                    self.resume_in_think = False
                else:
                    d["reasoning_content"] = (d.get("reasoning_content") or "") + text
            if d.get("reasoning_content"):
                self.reasoning += d["reasoning_content"]
            if d.get("content"):
                self.content += d["content"]
            fin = ch.get("finish_reason") or fin
        self.chunks += 1
        return chunk, fin


async def _sse(resp, obj: dict) -> None:
    await resp.write(b"data: " + json.dumps(obj).encode() + b"\n\n")


def build_app(cfg: Cfg) -> web.Application:
    p = Proxy(cfg)
    app = web.Application(client_max_size=256 * 1024 * 1024)
    app.on_startup.append(p.start)
    app.on_cleanup.append(p.stop)
    app.router.add_get("/v1/models", p.models)
    app.router.add_post("/v1/chat/completions", p.chat)
    app.router.add_route("*", "/{tail:.*}", p.passthrough)
    return app


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--port", type=int, default=30032)
    ap.add_argument("--upstream", default="http://127.0.0.1:30030")
    ap.add_argument("--hold-s", type=float, default=1800.0)
    ap.add_argument("--keepalive-s", type=float, default=5.0)
    ap.add_argument("--poll-s", type=float, default=2.0)
    ap.add_argument("--max-resumes", type=int, default=6)
    ap.add_argument("--state-file", default=os.path.expanduser("~/.owui_proxy_models.json"))
    a = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    cfg = Cfg(a.upstream, a.hold_s, a.keepalive_s, a.poll_s, a.state_file, a.max_resumes)
    web.run_app(build_app(cfg), host=a.host, port=a.port, access_log=None)


if __name__ == "__main__":
    main()
