"""The app driven over the real ASGI stack.

    CUDA_VISIBLE_DEVICES=99 PYTHONPATH=<worktree>/python \
        python -m pytest test/registered/copilot/test_server_ws.py -v

This is the execution smoke for the P1 slice: the whole chain -- handshake,
two tagged audio tracks, explicit commits, transcript frames, hint frames,
briefing round trip -- runs through FastAPI's own routing and WebSocket
handling rather than through direct method calls. #466 learned that a green
unit suite can coexist with a WS route that does not exist in a real
deployment, which is exactly what this file rules out.
"""

import json

import pytest
from fastapi.testclient import TestClient

from sglang.srt.copilot.config import CopilotConfig
from sglang.srt.copilot.deskfakes import desk_fake_backend_set
from sglang.srt.copilot.protocol import Track, encode_audio_frame
from sglang.srt.copilot.server import CopilotService, build_app

BRIEFING = """# Client call

## Contract renewal
Ends in March.
"""


@pytest.fixture
def client():
    config = CopilotConfig()
    service = CopilotService(config=config, backends=desk_fake_backend_set(config))
    with TestClient(build_app(service)) as test_client:
        yield test_client


RECEIVE_TIMEOUT_S = 5.0


def _receive_json(ws, timeout=RECEIVE_TIMEOUT_S):
    """Bounded read.

    ``WebSocketTestSession.receive_json`` blocks forever when the expected
    frame never arrives, which turns a broken assertion into a wedged test run
    -- and a wedged run is indistinguishable from a slow one. The bound is
    enforced through the session's own anyio portal. If starlette ever removes
    the stream this reaches for, this helper FAILS LOUDLY rather than falling
    back to an unbounded read.
    """
    import anyio

    stream = getattr(ws, "_send_rx", None)
    if stream is None:  # pragma: no cover - version guard
        raise AssertionError(
            "starlette's WebSocketTestSession no longer exposes _send_rx; "
            "update the bounded-receive helper instead of blocking forever"
        )

    async def _recv():
        with anyio.fail_after(timeout):
            return await stream.receive()

    message = ws.portal.call(_recv)
    if message["type"] == "websocket.close":
        raise AssertionError(f"socket closed while waiting: {message}")
    return json.loads(message["text"])


def drain(ws, kind, limit=40):
    """Read frames until one of ``kind`` arrives. Returns it."""
    seen = []
    for _ in range(limit):
        msg = _receive_json(ws)
        seen.append(msg.get("kind"))
        if msg.get("kind") == kind:
            return msg
    raise AssertionError(f"no {kind} frame within {limit} frames; saw {seen}")


class TestHttpSurface:
    def test_health(self, client):
        body = client.get("/api/copilot/health").json()
        assert body["status"] == "ok"
        assert body["build"]

    def test_index_serves_the_client_with_its_build_stamped(self, client):
        html = client.get("/").text
        assert "interview copilot" in html
        assert "__CLIENT_BUILD__" not in html

    def test_manifest_is_relative_so_a_path_prefix_survives(self, client):
        body = client.get("/manifest.webmanifest").json()
        assert body["start_url"] == "./"
        assert body["scope"] == "./"

    def test_unknown_session_transcript_is_404(self, client):
        assert client.get("/api/copilot/sessions/nope/transcript").status_code == 404


class TestWebSocketFlow:
    def test_full_round_trip(self, client):
        with client.websocket_connect("/api/copilot/stream") as ws:
            ws.send_json({"kind": "hello"})
            ready = drain(ws, "session.ready")
            assert ready["sample_rate"] == 16000
            assert ready["frame_ms"] == 20

            ws.send_json({"kind": "briefing.set", "text": BRIEFING})
            update = drain(ws, "briefing.update")
            assert update["briefing"]["sections"][0]["anchor"] == "contract-renewal"
            # Priming follows the briefing immediately.
            primed = drain(ws, "topic.state")
            assert primed["reason"] == "primed"
            assert primed["primed_tokens"] > 0
            # A prime alone never claims warmth.
            assert primed["warmth"] == "unknown"

            ws.send_json({"kind": "track.open", "track": "self"})
            drain(ws, "track.state")
            ws.send_json({"kind": "track.open", "track": "other"})
            drain(ws, "track.state")

            ws.send_bytes(encode_audio_frame(Track.OTHER, bytes(640), 0))
            ws.send_json({"kind": "commit", "track": "other"})
            line = drain(ws, "transcript.line")
            assert line["track"] == "other"
            assert line["text"]

            hint = drain(ws, "hint")
            assert hint["topic_id"] == "contract-renewal"
            assert hint["bullets"]
            assert hint["desk_fake"] is True
            assert hint["warmth"] in ("warm", "partial", "cold")

    def test_audio_before_hello_is_refused(self, client):
        with client.websocket_connect("/api/copilot/stream") as ws:
            ws.send_bytes(encode_audio_frame(Track.SELF, bytes(640), 0))
            err = drain(ws, "error")
            assert "hello" in err["message"]

    def test_malformed_audio_frame_is_refused_named(self, client):
        with client.websocket_connect("/api/copilot/stream") as ws:
            ws.send_json({"kind": "hello"})
            drain(ws, "session.ready")
            ws.send_bytes(b"\x00")
            err = drain(ws, "error")
            assert err["stage"] == "audio"
            assert "header" in err["message"]

    def test_unknown_frame_kind_is_refused(self, client):
        with client.websocket_connect("/api/copilot/stream") as ws:
            ws.send_json({"kind": "hello"})
            drain(ws, "session.ready")
            ws.send_json({"kind": "not.a.frame"})
            err = drain(ws, "error")
            assert err["stage"] == "protocol"

    def test_non_json_text_frame_is_refused(self, client):
        with client.websocket_connect("/api/copilot/stream") as ws:
            ws.send_text("{not json")
            err = drain(ws, "error")
            assert "valid JSON" in err["message"]

    def test_ping_pong(self, client):
        with client.websocket_connect("/api/copilot/stream") as ws:
            ws.send_json({"kind": "ping"})
            assert drain(ws, "pong")["kind"] == "pong"

    def test_resume_replays_from_the_cursor(self, client):
        with client.websocket_connect("/api/copilot/stream") as ws:
            ws.send_json({"kind": "hello"})
            ready = drain(ws, "session.ready")
            session_id = ready["session_id"]
            ws.send_json({"kind": "track.open", "track": "self"})
            first = drain(ws, "track.state")

        with client.websocket_connect("/api/copilot/stream") as ws:
            ws.send_json(
                {"kind": "hello", "session_id": session_id, "resume_from": first["seq"]}
            )
            drain(ws, "session.ready")
            replayed = drain(ws, "track.state")
            assert replayed["seq"] == first["seq"]

    def test_unknown_topic_focus_is_refused(self, client):
        with client.websocket_connect("/api/copilot/stream") as ws:
            ws.send_json({"kind": "hello"})
            drain(ws, "session.ready")
            ws.send_json({"kind": "topic.focus", "topic_id": "nope"})
            err = drain(ws, "error")
            assert err["stage"] == "topic"
