"""FastAPI app for the live interview copilot.

Routes live here rather than in ``entrypoints/http_server.py`` on purpose: that
file is the audited auth/CORS serving surface (#510, ``utils/auth.py:149-159``)
and a browser-facing app with its own static assets and WebSocket lifecycle has
no business widening it. This process owns no model, no VRAM and no CUDA
context -- it is a protocol adapter in front of the runtime's own public
endpoints, the same category as a browser, not a second serving engine.

ONE WRITER PER SOCKET. The session pushes events (transcript partials, hints,
briefing addenda) whenever it has them, so a connection is a reader task and a
writer task over one outbound queue -- never two tasks calling ``send_text``.
Before the handshake there is no queue and no writer, so the reader sends
directly; that is the only place a direct send is legal and it is the reason
the handshake is the only thing that may reply inline.

No route here is exposed publicly. The reverse-proxy template comes with P4,
on the #466 pattern including its ``location / { return 404; }`` catch-all.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import os
from contextlib import asynccontextmanager
from dataclasses import dataclass
from typing import Any, Dict, Optional, Set

from fastapi import FastAPI, HTTPException, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse, JSONResponse, Response

from sglang.srt.copilot.backends import BackendSet
from sglang.srt.copilot.briefing import Briefing
from sglang.srt.copilot.config import FRAME_MS, PIPELINE_SAMPLE_RATE, CopilotConfig
from sglang.srt.copilot.protocol import (
    ClientFrame,
    Event,
    ProtocolError,
    ServerFrame,
    decode_audio_frame,
    parse_client_frame,
    parse_track,
)
from sglang.srt.copilot.session import CopilotSession, SessionManager, Subscription

logger = logging.getLogger(__name__)

CLIENT_DIR = os.path.join(os.path.dirname(__file__), "client")


@dataclass(frozen=True)
class _Kill:
    """Ask a connection's writer to close the socket.

    Closing from another task would race the writer's ``send_text``, so the
    request travels through the same queue as everything else.
    """

    code: int
    reason: str


class CopilotService:
    """Everything the routes need, in one injectable object."""

    def __init__(
        self,
        config: CopilotConfig,
        backends: BackendSet,
        default_briefing: Optional[Briefing] = None,
    ) -> None:
        self.config = config
        self.backends = backends
        self.sessions = SessionManager(
            config=config,
            backends=backends,
            default_briefing=default_briefing,
        )
        self.connections: Set["_Connection"] = set()

    def client_html(self) -> str:
        path = os.path.join(CLIENT_DIR, "index.html")
        with open(path, "r", encoding="utf-8") as fh:
            return fh.read()

    def client_build(self) -> str:
        """Content hash of the served page.

        #466 learned this the hard way: a stale cached client is otherwise
        indistinguishable from a server bug (``translator/server.py:836-842``).
        """
        return hashlib.sha256(self.client_html().encode("utf-8")).hexdigest()[:12]

    def drop_connections(self, reason: str) -> int:
        """Close every live socket without touching session state.

        The acceptance affordance for "the connection died mid-session": the
        journal, transcript and hint history stay exactly where they are, so a
        reconnect must be able to reproduce them.
        """
        killed = 0
        for conn in list(self.connections):
            if conn.kill(1012, reason):
                killed += 1
        return killed


class _Connection:
    """One browser WebSocket: a reader, a writer, one outbound queue."""

    def __init__(self, service: CopilotService, websocket: WebSocket) -> None:
        self.service = service
        self.ws = websocket
        self.session: Optional[CopilotSession] = None
        self.sub: Optional[Subscription] = None
        self._writer: Optional[asyncio.Task] = None
        self._replay_upto = 0

    # --- outbound ---------------------------------------------------------

    async def _send_direct(self, payload: Dict[str, Any]) -> None:
        """Legal only before the writer exists. See the module docstring."""
        await self.ws.send_text(json.dumps(payload))

    async def _send(self, payload: Dict[str, Any]) -> None:
        if self.sub is None:
            await self._send_direct(payload)
        else:
            self.sub.offer_raw(payload)

    async def _error(self, stage: str, message: str) -> None:
        await self._send(
            {"kind": ServerFrame.ERROR.value, "stage": stage, "message": message}
        )

    def kill(self, code: int, reason: str) -> bool:
        if self.sub is None:
            return False
        self.sub.offer_raw(_Kill(code, reason))
        return True

    async def _writer_loop(self) -> None:
        sub = self.sub
        assert sub is not None
        try:
            while True:
                item = await sub.queue.get()
                if sub.overflowed:
                    # The client is slower than the conversation. Say so and
                    # close: the journal retains more than this queue holds, so
                    # the reconnect replays what was dropped. Serving a
                    # silently thinned event stream is the one option that is
                    # not allowed.
                    await self.ws.send_text(
                        json.dumps(
                            {
                                "kind": ServerFrame.ERROR.value,
                                "stage": "transport",
                                "message": "outbound queue overflowed; reconnect "
                                "and resume from your cursor",
                            }
                        )
                    )
                    await self.ws.close(code=1013, reason="outbound overflow")
                    return
                if isinstance(item, _Kill):
                    await self.ws.close(code=item.code, reason=item.reason)
                    return
                if isinstance(item, Event):
                    if item.seq < self._replay_upto:
                        # Already sent inline by the handshake replay.
                        continue
                    await self.ws.send_text(json.dumps(item.to_json()))
                else:
                    await self.ws.send_text(json.dumps(item))
        except (WebSocketDisconnect, RuntimeError):
            return

    # --- lifecycle --------------------------------------------------------

    async def run(self) -> None:
        await self.ws.accept()
        self.service.connections.add(self)
        try:
            await self._loop()
        except WebSocketDisconnect:
            logger.info("[copilot] client disconnected")
        except Exception:
            logger.exception("[copilot] connection failed")
            try:
                await self._error("internal", "internal error")
            except Exception:
                pass
        finally:
            self.service.connections.discard(self)
            if self._writer is not None:
                self._writer.cancel()
                try:
                    await self._writer
                except (asyncio.CancelledError, Exception):
                    pass
            if self.session is not None and self.sub is not None:
                self.session.unsubscribe(self.sub)

    async def _loop(self) -> None:
        while True:
            message = await self.ws.receive()
            if message.get("type") == "websocket.disconnect":
                return
            if message.get("bytes") is not None:
                await self._on_binary(message["bytes"])
                continue
            text = message.get("text")
            if text is None:
                continue
            try:
                raw = json.loads(text)
            except json.JSONDecodeError:
                await self._error("protocol", "frame is not valid JSON")
                continue
            if not isinstance(raw, dict):
                await self._error("protocol", "frame must be a JSON object")
                continue
            try:
                if not await self._on_json(raw):
                    return
            except ProtocolError as exc:
                await self._error("protocol", str(exc))

    async def _on_binary(self, data: bytes) -> None:
        if self.session is None:
            await self._error("protocol", "send hello before audio")
            return
        try:
            frame = decode_audio_frame(data)
        except ProtocolError as exc:
            await self._error("audio", str(exc))
            return
        await self.session.on_audio(frame)

    async def _on_json(self, raw: Dict[str, Any]) -> bool:
        kind = parse_client_frame(raw)

        if kind is ClientFrame.HELLO:
            if self.session is not None:
                await self._error("protocol", "hello already accepted on this socket")
                return True
            await self._handshake(raw)
            return True

        if kind is ClientFrame.PING:
            await self._send({"kind": ServerFrame.PONG.value})
            return True

        if kind is ClientFrame.CLOSE:
            return False

        if self.session is None:
            await self._error("protocol", "send hello first")
            return True

        session = self.session

        if kind is ClientFrame.TRACK_OPEN:
            await session.open_track(parse_track(raw.get("track")))
        elif kind is ClientFrame.TRACK_CLOSE:
            await session.close_track(parse_track(raw.get("track")))
        elif kind is ClientFrame.COMMIT:
            await session.on_commit(parse_track(raw.get("track")))
        elif kind is ClientFrame.BRIEFING_SET:
            text = raw.get("text")
            if not isinstance(text, str):
                await self._error("briefing", "briefing.set needs a 'text' string")
                return True
            session.set_briefing(text, source="client")
            await session.prime_due_topics()
        elif kind is ClientFrame.BRIEFING_GET:
            await self._send(
                {
                    "kind": ServerFrame.BRIEFING_UPDATE.value,
                    "reason": "briefing.get",
                    "briefing": session.briefing.to_json(),
                    "markdown": session.briefing.render(),
                    "topics": session.topics.states(),
                }
            )
        elif kind is ClientFrame.TOPIC_FOCUS:
            topic_id = raw.get("topic_id")
            try:
                await session.focus_topic(str(topic_id))
            except KeyError:
                await self._error("topic", f"unknown topic {topic_id!r}")
        elif kind is ClientFrame.STATE:
            await self._send(
                {
                    "kind": ServerFrame.SESSION_STATE.value,
                    **session.state_json(),
                }
            )
        elif kind is ClientFrame.ACK:
            # Advisory only, exactly as in #466: retention is the journal's
            # decision, never the client's.
            pass
        return True

    async def _handshake(self, raw: Dict[str, Any]) -> None:
        session_id = raw.get("session_id")
        try:
            session = await self.service.sessions.open(
                session_id if isinstance(session_id, str) else None
            )
        except RuntimeError as exc:
            await self._error("session", str(exc))
            return
        self.session = session

        cursor = raw.get("resume_from")
        cursor = int(cursor) if isinstance(cursor, int) else 0
        gap = session.journal.has_gap(cursor)
        replay = session.journal.since(cursor)

        # Subscribe BEFORE the replay is written out, with no await in between,
        # so no event can fall between the two. Anything the session emits from
        # here on is queued and delivered after the replay; the writer drops
        # sequences below the snapshot in case that invariant is ever broken.
        self.sub = session.subscribe()
        self._replay_upto = session.journal.next_seq
        self._writer = asyncio.create_task(
            self._writer_loop(), name=f"copilot-writer-{session.session_id}"
        )

        await self._send_direct(
            {
                "kind": ServerFrame.SESSION_READY.value,
                "session_id": session.session_id,
                # ``state_json`` already carries ``next_seq``: the first
                # sequence the client has NOT been given. Deliberately not
                # called "seq" -- a journalled event's "seq" identifies that
                # event, and one name for both costs one lost event per
                # reconnect.
                "replay_from": cursor,
                "replayed": len(replay),
                "sample_rate": PIPELINE_SAMPLE_RATE,
                "frame_ms": FRAME_MS,
                "build": self.service.client_build(),
                "stub": session.backends.stub,
                **session.state_json(),
            }
        )
        if gap:
            await self._send_direct(
                {
                    "kind": ServerFrame.RESUME_GAP.value,
                    "floor": session.journal.floor,
                    "requested": cursor,
                }
            )
        for event in replay:
            await self._send_direct({**event.to_json(), "replay": True})
        session.start_background()


def build_app(service: CopilotService) -> FastAPI:
    @asynccontextmanager
    async def lifespan(app: FastAPI):
        yield
        # Every session owns tasks (the expander, a stub ASR driver) and those
        # must be cancelled on the way out, or a test run leaves them behind and
        # the next one inherits a conversation that is still talking.
        await service.sessions.aclose()

    app = FastAPI(title="htsglang interview copilot", lifespan=lifespan)
    app.state.service = service

    @app.get("/api/copilot/health")
    async def health() -> JSONResponse:
        return JSONResponse(
            {
                "status": "ok",
                "backend": service.backends.name,
                "stub": service.backends.stub,
                "sessions": len(service.sessions.sessions),
                "connections": len(service.connections),
                "runtime": service.config.runtime_base_url,
                "build": service.client_build(),
                "prep": service.backends.prep.report(),
            }
        )

    @app.get("/api/copilot/sessions")
    async def list_sessions() -> JSONResponse:
        return JSONResponse(
            {"sessions": [s.state_json() for s in service.sessions.sessions.values()]}
        )

    @app.post("/api/copilot/sessions")
    async def create_session() -> JSONResponse:
        try:
            session = await service.sessions.open(None)
        except RuntimeError as exc:
            raise HTTPException(status_code=429, detail=str(exc)) from exc
        return JSONResponse(session.state_json())

    @app.delete("/api/copilot/sessions/{session_id}")
    async def delete_session(session_id: str) -> JSONResponse:
        return JSONResponse({"closed": await service.sessions.close(session_id)})

    @app.get("/api/copilot/sessions/{session_id}/briefing")
    async def get_briefing(session_id: str) -> JSONResponse:
        session = service.sessions.sessions.get(session_id)
        if session is None:
            raise HTTPException(status_code=404, detail="unknown session")
        return JSONResponse(
            {
                "briefing": session.briefing.to_json(),
                "markdown": session.briefing.render(),
            }
        )

    @app.get("/api/copilot/sessions/{session_id}/transcript")
    async def get_transcript(session_id: str) -> JSONResponse:
        session = service.sessions.sessions.get(session_id)
        if session is None:
            raise HTTPException(status_code=404, detail="unknown session")
        return JSONResponse({"lines": [ln.to_json() for ln in session.transcript]})

    @app.websocket("/api/copilot/stream")
    async def stream(websocket: WebSocket) -> None:
        await _Connection(service, websocket).run()

    @app.get("/", response_class=HTMLResponse)
    async def index(request: Request) -> HTMLResponse:
        html = service.client_html().replace("__CLIENT_BUILD__", service.client_build())
        return HTMLResponse(html)

    @app.get("/manifest.webmanifest")
    async def manifest() -> Response:
        body = {
            "name": "htsglang interview copilot",
            "short_name": "copilot",
            "start_url": "./",
            "scope": "./",
            "display": "standalone",
            "background_color": "#101014",
            "theme_color": "#101014",
        }
        return Response(
            content=json.dumps(body),
            media_type="application/manifest+json",
        )

    if service.backends.stub:
        _add_stub_routes(app, service)

    return app


def _add_stub_routes(app: FastAPI, service: CopilotService) -> None:
    """Fault injection, registered ONLY for a non-real backend set.

    Honest degradation cannot be demonstrated without a way to break the thing
    on purpose. These routes exist so the acceptance run can kill the transport
    and fail the hint backend from outside, and they are absent from any other
    backend's route table -- not merely refused, absent.
    """

    @app.post("/api/copilot/stub/disconnect")
    async def stub_disconnect() -> JSONResponse:
        return JSONResponse({"closed": service.drop_connections("stub disconnect")})

    @app.post("/api/copilot/stub/hint-fault")
    async def stub_hint_fault(payload: Dict[str, Any]) -> JSONResponse:
        fail = bool(payload.get("fail"))
        backend = service.backends.hints
        if not hasattr(backend, "fail"):
            raise HTTPException(
                status_code=409,
                detail=f"hint backend {type(backend).__name__} has no fault switch",
            )
        backend.fail = fail
        return JSONResponse({"fail": fail})
