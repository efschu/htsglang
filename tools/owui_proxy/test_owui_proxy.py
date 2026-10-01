"""owui_proxy gegen ein nachgebautes Backend: Abbruch mitten im Stream (auch in der Denkphase), Backend eine Weile
weg, Modellliste nie leer, Halten vor dem ersten Token."""
import asyncio
import json
import os
import sys

import aiohttp
from aiohttp import web

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import owui_proxy  # noqa: E402

ANSWER = "Die Antwort hat genau zwoelf Woerter und endet hier mit Punkt."
THINK = "Erst denken, dann schreiben."


class FakeBackend:
    """Streams THINK as reasoning_content, then ANSWER word by word; dies after die_after chunks once."""

    def __init__(self, die_after=None, model="Fake-NF"):
        self.die_after = die_after
        self.model = model
        self.up = True
        self.requests = []

    def app(self):
        app = web.Application()
        app.router.add_get("/v1/models", self.models)
        app.router.add_post("/v1/chat/completions", self.chat)
        return app

    async def models(self, request):
        if not self.up:
            return web.Response(status=502)
        return web.json_response({"object": "list", "data": [{"id": self.model, "object": "model"}]})

    def pieces(self, body):
        """(kind, text) still to send, given the assistant prefix of a continuation."""
        msgs = body.get("messages") or []
        prefix = msgs[-1]["content"] if body.get("continue_final_message") else ""
        full = [("r", w + " ") for w in THINK.split()] + [("c", w + " ") for w in ANSWER.split()]
        done_r = ""
        done_c = ""
        if prefix.startswith("<think>\n"):
            rest = prefix[len("<think>\n"):]
            if "\n</think>\n\n" in rest:
                done_r, done_c = rest.split("\n</think>\n\n", 1)
            else:
                done_r = rest
        else:
            done_c = prefix
        out, r_acc, c_acc = [], "", ""
        for kind, t in full:
            if kind == "r":
                if len(r_acc) < len(done_r):
                    r_acc += t
                    continue
            else:
                if len(c_acc) < len(done_c):
                    c_acc += t
                    continue
            out.append((kind, t))
        # a continuation inside the thinking phase gets everything as plain content incl. </think>
        if prefix.startswith("<think>\n") and "\n</think>" not in prefix:
            text = "".join(t for k, t in out if k == "r") + "</think>\n\n" + "".join(t for k, t in out if k == "c")
            return [("c", w) for w in _split_keep(text)]
        return out

    async def chat(self, request):
        if not self.up:
            return web.Response(status=502)
        body = await request.json()
        self.requests.append(body)
        resp = web.StreamResponse(headers={"Content-Type": "text/event-stream"})
        await resp.prepare(request)
        sent = 0
        for kind, text in self.pieces(body):
            if self.die_after is not None and sent >= self.die_after:
                self.die_after = None
                self.up = False
                asyncio.get_event_loop().call_later(0.6, setattr, self, "up", True)
                request.transport.close()
                return resp
            delta = {"reasoning_content": text} if kind == "r" else {"content": text}
            await resp.write(b"data: " + json.dumps(
                {"id": "up-%d" % len(self.requests), "model": self.model,
                 "choices": [{"index": 0, "delta": delta, "finish_reason": None}]}).encode() + b"\n\n")
            sent += 1
            await asyncio.sleep(0.005)
        await resp.write(b"data: " + json.dumps(
            {"id": "x", "model": self.model, "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]}
        ).encode() + b"\n\n")
        await resp.write(b"data: [DONE]\n\n")
        return resp


def _split_keep(text):
    out, cur = [], ""
    for ch in text:
        cur += ch
        if ch == " ":
            out.append(cur)
            cur = ""
    if cur:
        out.append(cur)
    return out


async def _serve(app):
    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, "127.0.0.1", 0)
    await site.start()
    port = site._server.sockets[0].getsockname()[1]
    return runner, port


async def _run(backend, tmp_path, body_extra=None):
    brun, bport = await _serve(backend.app())
    cfg = owui_proxy.Cfg("http://127.0.0.1:%d" % bport, hold_s=10, keepalive_s=0.2, poll_s=0.1,
                         state_file=str(tmp_path / "models.json"), max_resumes=4)
    prun, pport = await _serve(owui_proxy.build_app(cfg))
    reasoning, content, comments, ids = "", "", 0, set()
    try:
        async with aiohttp.ClientSession() as s:
            body = dict({"model": "irgendwas", "stream": True, "messages": [{"role": "user", "content": "hi"}]},
                        **(body_extra or {}))
            async with s.post("http://127.0.0.1:%d/v1/chat/completions" % pport, json=body) as r:
                async for line in r.content:
                    line = line.strip()
                    if line.startswith(b":"):
                        comments += 1
                    if not line.startswith(b"data:") or line == b"data: [DONE]":
                        continue
                    c = json.loads(line[5:])
                    ids.add(c.get("id"))
                    for ch in c.get("choices") or []:
                        d = ch.get("delta") or {}
                        reasoning += d.get("reasoning_content") or ""
                        content += d.get("content") or ""
            async with s.get("http://127.0.0.1:%d/v1/models" % pport) as r:
                models = [m["id"] for m in (await r.json())["data"]]
    finally:
        await prun.cleanup()
        await brun.cleanup()
    return reasoning, content, comments, ids, models


def test_break_in_answer_is_resumed(tmp_path):
    b = FakeBackend(die_after=7)          # 4 reasoning pieces + 3 answer words, then the backend dies
    reasoning, content, comments, ids, models = asyncio.run(_run(b, tmp_path))
    assert content.strip() == ANSWER
    assert reasoning.strip() == THINK
    assert len(ids) == 1                   # one stream for the client
    assert comments >= 1                   # keepalive while the backend was gone
    assert b.requests[1]["continue_final_message"] is True
    assert "rig-auto" in models and "Fake-NF" in models


def test_break_in_thinking_is_resumed_as_reasoning(tmp_path):
    b = FakeBackend(die_after=2)
    reasoning, content, _, _, _ = asyncio.run(_run(b, tmp_path))
    assert reasoning.strip() == THINK
    assert content.strip() == ANSWER


def test_no_break_passes_through(tmp_path):
    b = FakeBackend()
    reasoning, content, comments, _, _ = asyncio.run(_run(b, tmp_path))
    assert (reasoning.strip(), content.strip(), comments) == (THINK, ANSWER, 0)
    assert len(b.requests) == 1 and "continue_final_message" not in b.requests[0]
    assert b.requests[0]["model"] == "Fake-NF"   # request model id follows the live model


def test_resume_in_think_when_backend_parses_reasoning_itself():
    """Live 01.10.: the htsglang front returns reasoning_content for a continuation that starts in <think>."""
    st = owui_proxy.StreamState({"model": "m", "messages": []})
    st.absorb({"choices": [{"delta": {"reasoning_content": "Erst denken"}}]})
    b = st.request_body("m")
    assert b["messages"][-1]["content"] == "<think>\nErst denken" and st.resume_in_think
    st.absorb({"choices": [{"delta": {"reasoning_content": ", dann schreiben."}}]})
    out, _ = st.absorb({"choices": [{"delta": {"content": "\n\nDie Antwort."}}]})
    assert out["choices"][0]["delta"] == {"content": "\n\nDie Antwort."}
    assert st.reasoning == "Erst denken, dann schreiben." and st.content == "\n\nDie Antwort."


def test_models_survive_backend_gone(tmp_path):
    async def go():
        b = FakeBackend(model="Fake-27B")
        brun, bport = await _serve(b.app())
        cfg = owui_proxy.Cfg("http://127.0.0.1:%d" % bport, 10, 0.2, 0.1, str(tmp_path / "m.json"), 2)
        prun, pport = await _serve(owui_proxy.build_app(cfg))
        try:
            async with aiohttp.ClientSession() as s:
                await (await s.get("http://127.0.0.1:%d/v1/models" % pport)).json()
                b.up = False
                data = (await (await s.get("http://127.0.0.1:%d/v1/models" % pport)).json())["data"]
        finally:
            await prun.cleanup()
            await brun.cleanup()
        return {m["id"]: m["live"] for m in data}
    assert asyncio.run(go()) == {"rig-auto": False, "Fake-27B": False}
