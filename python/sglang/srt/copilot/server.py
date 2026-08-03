"""FastAPI app for the live interview copilot.

Routes live here rather than in ``entrypoints/http_server.py`` on purpose: that
file is the audited auth/CORS serving surface (#510, ``utils/auth.py:149-159``)
and a browser-facing app with its own static assets and WebSocket lifecycle has
no business widening it. This process owns no model, no VRAM and no CUDA
context -- it is a protocol adapter in front of the runtime's own public
endpoints, the same category as a browser, not a second serving engine.

No route here is exposed publicly in P1. The reverse-proxy template comes with
P4, on the #466 pattern including its ``location / { return 404; }`` catch-all.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
from typing import Any, Dict, Optional

from fastapi import FastAPI, HTTPException, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse, JSONResponse, Response

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
from sglang.srt.copilot.session import CopilotSession, SessionManager

logger = logging.getLogger(__name__)

CLIENT_DIR = os.path.join(os.path.dirname(__file__), "client")


class CopilotService:
    """Everything the routes need, in one injectable object."""

    def __init__(
        self,
        config: CopilotConfig,
        hint_backend: Any,
        asr_factory: Any = None,
        default_briefing: Any = None,
    ) -> None:
        self.config = config
        self.sessions = SessionManager(
            config=config,
            hint_backend=hint_backend,
            asr_factory=asr_factory,
            default_briefing=default_briefing,
        )

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


class _Connection:
    """One browser WebSocket. Owns the per-connection replay cursor."""

    def __init__(self, service: CopilotService, websocket: WebSocket) -> None:
        self.service = service
        self.ws = websocket
        self.session: Optional[CopilotSession] = None

    async def _send(self, event: Event) -> None:
        await self.ws.send_text(json.dumps(event.to_json()))

    async def _send_raw(self, payload: Dict[str, Any]) -> None:
        await self.ws.send_text(json.dumps(payload))

    async def _error(self, stage: str, message: str) -> None:
        await self._send_raw(
            {"kind": ServerFrame.ERROR.value, "stage": stage, "message": message}
        )

    async def run(self) -> None:
        await self.ws.accept()
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
        for event in await self.session.on_audio(frame):
            await self._send(event)

    async def _on_json(self, raw: Dict[str, Any]) -> bool:
        kind = parse_client_frame(raw)

        if kind is ClientFrame.HELLO:
            await self._handshake(raw)
            return True

        if kind is ClientFrame.PING:
            await self._send_raw({"kind": ServerFrame.PONG.value})
            return True

        if kind is ClientFrame.CLOSE:
            return False

        if self.session is None:
            await self._error("protocol", "send hello first")
            return True

        session = self.session

        if kind is ClientFrame.TRACK_OPEN:
            await self._send(session.open_track(parse_track(raw.get("track"))))
        elif kind is ClientFrame.TRACK_CLOSE:
            await self._send(session.close_track(parse_track(raw.get("track"))))
        elif kind is ClientFrame.COMMIT:
            for event in await session.on_commit(parse_track(raw.get("track"))):
                await self._send(event)
        elif kind is ClientFrame.BRIEFING_SET:
            text = raw.get("text")
            if not isinstance(text, str):
                await self._error("briefing", "briefing.set needs a 'text' string")
                return True
            await self._send(session.set_briefing(text, source="client"))
            for event in await session.prime_due_topics():
                await self._send(event)
        elif kind is ClientFrame.BRIEFING_GET:
            await self._send_raw(
                {
                    "kind": ServerFrame.BRIEFING_UPDATE.value,
                    "reason": "briefing.get",
                    "briefing": session.briefing.to_json(),
                    "markdown": session.briefing.render(),
                }
            )
        elif kind is ClientFrame.TOPIC_FOCUS:
            topic_id = raw.get("topic_id")
            try:
                await self._send(session.focus_topic(str(topic_id)))
            except KeyError:
                await self._error("topic", f"unknown topic {topic_id!r}")
        elif kind is ClientFrame.STATE:
            await self._send_raw(
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
            session = self.service.sessions.open(
                session_id if isinstance(session_id, str) else None
            )
        except RuntimeError as exc:
            await self._error("session", str(exc))
            return
        self.session = session

        cursor = raw.get("resume_from")
        cursor = int(cursor) if isinstance(cursor, int) else 0

        await self._send_raw(
            {
                "kind": ServerFrame.SESSION_READY.value,
                "session_id": session.session_id,
                "seq": session.journal.next_seq,
                "sample_rate": PIPELINE_SAMPLE_RATE,
                "frame_ms": FRAME_MS,
                "build": self.service.client_build(),
                **session.state_json(),
            }
        )

        if session.journal.has_gap(cursor):
            await self._send_raw(
                {
                    "kind": ServerFrame.RESUME_GAP.value,
                    "floor": session.journal.floor,
                    "requested": cursor,
                }
            )
        for event in session.journal.since(cursor):
            await self._send(event)


def build_app(service: CopilotService) -> FastAPI:
    app = FastAPI(title="htsglang interview copilot")
    app.state.service = service

    @app.get("/api/copilot/health")
    async def health() -> JSONResponse:
        return JSONResponse(
            {
                "status": "ok",
                "sessions": len(service.sessions.sessions),
                "runtime": service.config.runtime_base_url,
                "build": service.client_build(),
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
            session = service.sessions.open(None)
        except RuntimeError as exc:
            raise HTTPException(status_code=429, detail=str(exc)) from exc
        return JSONResponse(session.state_json())

    @app.delete("/api/copilot/sessions/{session_id}")
    async def delete_session(session_id: str) -> JSONResponse:
        return JSONResponse({"closed": service.sessions.close(session_id)})

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

    return app
