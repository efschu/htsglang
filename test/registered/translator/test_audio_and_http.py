# Copyright 2025 SGLang Team
# Licensed under the Apache License, Version 2.0
"""Transport: codec negotiation, resampling, and the HTTP/WebSocket surface.

Hermetic. The WebSocket tests drive Starlette's in-process test client, so no
socket is opened and no model is loaded; the backends are the fakes.

    CUDA_VISIBLE_DEVICES=99 python -m pytest test/registered/translator/test_audio_and_http.py -v
"""

import json
import math
import unittest

import numpy as np
from fastapi.testclient import TestClient

from sglang.srt.translator.audio import (
    PIPELINE_SAMPLE_RATE,
    CodecError,
    Pcm16Codec,
    available_codecs,
    decode_event_audio,
    encode_event_audio,
    from_pcm16_bytes,
    negotiate_codec,
    resample,
    to_pcm16_bytes,
)
from sglang.srt.translator.backends import (
    AudioChunk,
    FakeAsr,
    FakeEmbedder,
    FakeMt,
    FakeTts,
)
from sglang.srt.translator.config import TranslatorConfig, TtsConfig
from sglang.srt.translator.server import Stack, TranslatorService, build_app

LANG_A = "aa"
LANG_B = "bb"
VOICE_A_HZ = 140.0
RATE = PIPELINE_SAMPLE_RATE


def tone(frequency, seconds, rate=RATE, amplitude=0.3):
    t = np.arange(int(seconds * rate), dtype=np.float32) / rate
    return (amplitude * np.sin(2.0 * math.pi * frequency * t)).astype(np.float32)


class TestPcmRoundTrip(unittest.TestCase):
    def test_round_trip_is_lossless_within_quantisation(self):
        original = AudioChunk(tone(220.0, 0.1), RATE)
        restored = from_pcm16_bytes(to_pcm16_bytes(original), RATE)
        self.assertEqual(len(restored.samples), len(original.samples))
        self.assertLess(float(np.abs(restored.samples - original.samples).max()), 1e-3)

    def test_an_odd_length_payload_is_refused_not_truncated(self):
        # Truncating would desynchronise the stream slowly and invisibly.
        with self.assertRaises(CodecError):
            from_pcm16_bytes(b"\x00\x01\x02", RATE)

    def test_event_audio_round_trip(self):
        original = AudioChunk(tone(300.0, 0.05), RATE)
        restored = decode_event_audio(encode_event_audio(original))
        self.assertEqual(restored.sample_rate, RATE)
        self.assertEqual(len(restored.samples), len(original.samples))


class TestResampling(unittest.TestCase):
    def test_it_preserves_pitch_across_the_rates_the_system_uses(self):
        for source_rate, target_rate in (
            (16000, 24000),
            (24000, 16000),
            (48000, 16000),
            (16000, 22050),
        ):
            with self.subTest(f"{source_rate}->{target_rate}"):
                audio = AudioChunk(tone(440.0, 0.5, rate=source_rate), source_rate)
                out = resample(audio, target_rate)
                self.assertEqual(out.sample_rate, target_rate)
                spectrum = np.abs(np.fft.rfft(out.samples))
                freqs = np.fft.rfftfreq(len(out.samples), 1.0 / target_rate)
                peak = float(freqs[int(np.argmax(spectrum))])
                # A wrong-pitch resample is the classic silent audio bug; the
                # assertion is on the frequency, not on the sample count.
                self.assertAlmostEqual(peak, 440.0, delta=15.0)

    def test_a_no_op_conversion_returns_the_same_object(self):
        audio = AudioChunk(tone(440.0, 0.1), RATE)
        self.assertIs(resample(audio, RATE), audio)

    def test_an_unsupported_ratio_is_refused_loudly(self):
        audio = AudioChunk(tone(440.0, 0.01, rate=44101), 44101)
        with self.assertRaises(CodecError):
            resample(audio, 40009)


class TestCodecNegotiation(unittest.TestCase):
    def test_pcm16_is_always_available(self):
        self.assertIn("pcm16", available_codecs())

    def test_negotiation_prefers_the_server_order(self):
        codec = negotiate_codec(["pcm16"])
        self.assertEqual(codec.name, "pcm16")

    def test_no_common_codec_is_an_error_naming_both_sides(self):
        with self.assertRaises(CodecError) as ctx:
            negotiate_codec(["speex", "gsm"])
        self.assertIn("speex", str(ctx.exception))
        self.assertIn("pcm16", str(ctx.exception))

    def test_pcm16_framing_matches_the_declared_frame_size(self):
        codec = Pcm16Codec(sample_rate=RATE, frame_ms=20)
        frames = list(codec.encode(AudioChunk(tone(220.0, 0.1), RATE)))
        self.assertEqual(len(frames), 5)
        self.assertTrue(all(len(f) == 2 * RATE * 20 // 1000 for f in frames))


def build_service(**overrides):
    tts_config = TtsConfig(min_reference_seconds=overrides.pop("min_ref", 1.0))
    config = TranslatorConfig(
        tts=tts_config,
        default_participants=(LANG_A, LANG_B),
        journal_events=128,
        journal_audio_mib=4,
        max_sessions=3,
        **overrides,
    )
    stack = Stack(
        asr=FakeAsr(
            languages=(LANG_A, LANG_B),
            pitch_map=[(VOICE_A_HZ, LANG_A), (240.0, LANG_B)],
        ),
        embedder=FakeEmbedder(min_seconds=0.5),
        mt=FakeMt(),
        tts=FakeTts(
            languages=(LANG_A, LANG_B),
            sample_rate=RATE,
            min_reference_seconds=tts_config.min_reference_seconds,
            chunk_seconds=0.2,
            seconds_per_char=0.01,
        ),
    )
    return TranslatorService(config, stack)


class TestReadEndpoints(unittest.TestCase):
    def setUp(self):
        self.client = TestClient(build_app(build_service()))

    def test_languages_is_derived_from_the_backends(self):
        payload = self.client.get("/api/translator/languages").json()
        self.assertEqual(payload["stages"]["asr"], [LANG_A, LANG_B])
        self.assertEqual(payload["stages"]["tts"], [LANG_A, LANG_B])
        self.assertIsNone(payload["stages"]["mt"])
        self.assertTrue(payload["unconstrained_mt"])
        self.assertEqual(payload["pair_count"], 2)
        self.assertTrue(payload["default_participants_supported"])
        self.assertEqual(payload["backends"]["asr"], "fake-asr")

    def test_a_default_pair_the_deployment_cannot_run_is_reported(self):
        service = build_service()
        service.stack.tts = FakeTts(languages=(LANG_A,), sample_rate=RATE)
        payload = TestClient(build_app(service)).get(
            "/api/translator/languages"
        ).json()
        self.assertFalse(payload["default_participants_supported"])
        self.assertIn("TTS cannot speak", payload["default_participants_error"])

    def test_health_reports_budgets_and_codecs(self):
        payload = self.client.get("/api/translator/health").json()
        self.assertEqual(payload["status"], "ok")
        self.assertIn("pcm16", payload["codecs"])
        self.assertEqual(payload["pipeline_sample_rate"], PIPELINE_SAMPLE_RATE)
        self.assertGreater(payload["budgets_mib"]["total"], 0)

    def test_opening_an_unsupported_conversation_is_a_400(self):
        response = self.client.post(
            "/api/translator/sessions", json={"participants": [LANG_A, "zz"]}
        )
        self.assertEqual(response.status_code, 400)
        self.assertIn("zz", response.json()["detail"])

    def test_session_lifecycle(self):
        created = self.client.post(
            "/api/translator/sessions", json={"session_id": "abc"}
        ).json()
        self.assertEqual(created["session_id"], "abc")
        listed = self.client.get("/api/translator/sessions").json()
        self.assertEqual([s["session_id"] for s in listed["sessions"]], ["abc"])
        self.assertTrue(
            self.client.delete("/api/translator/sessions/abc").json()["closed"]
        )
        self.assertFalse(
            self.client.delete("/api/translator/sessions/abc").json()["closed"]
        )

    def test_enrollment_registers_a_voice(self):
        self.client.post("/api/translator/sessions", json={"session_id": "abc"})
        payload = {
            "label": "user",
            "language": LANG_A,
            "text": "sample",
            "audio": encode_event_audio(AudioChunk(tone(VOICE_A_HZ, 6.0), RATE)),
        }
        response = self.client.post("/api/translator/enroll/abc", json=payload)
        self.assertEqual(response.status_code, 200)
        body = response.json()
        self.assertEqual(body["speaker_id"], "enrolled:user")
        speaker = body["state"]["speakers"][0]
        self.assertTrue(speaker["enrolled"])
        self.assertGreaterEqual(speaker["reference_seconds"], 6.0)

    def test_enrollment_into_a_missing_session_is_a_404(self):
        response = self.client.post(
            "/api/translator/enroll/nope",
            json={"audio": encode_event_audio(AudioChunk(tone(200.0, 1.0), RATE))},
        )
        self.assertEqual(response.status_code, 404)


def drain_until(ws, predicate, budget_s=15.0):
    """Read control frames until ``predicate`` matches, or the budget expires.

    ``ws.receive()`` on Starlette's test client blocks forever when nothing is
    coming, so a plain read loop wedges the suite exactly the way an unbounded
    wait wedges a real session -- the failure the project's robustness canon
    names explicitly, and one this helper hit for real before it was written
    this way.

    The fence is a protocol-level sentinel rather than a timeout: send a
    ``ping`` and read until the matching ``pong``. Every read is then
    guaranteed to terminate as long as the server's receive loop is alive, and
    the outer wall-clock budget bounds the number of rounds. Binary frames
    (synthesized audio) are skipped without counting against anything.
    """
    import time

    seen = []
    deadline = time.monotonic() + budget_s
    while time.monotonic() < deadline:
        ws.send_text(json.dumps({"kind": "ping"}))
        while True:
            message = ws.receive()
            if message.get("type") == "websocket.disconnect":
                return None, seen
            text = message.get("text")
            if text is None:
                continue  # a binary audio frame
            event = json.loads(text)
            seen.append(event)
            if predicate(event):
                return event, seen
            if event.get("kind") == "pong":
                break
        time.sleep(0.05)
    return None, seen


class TestWebSocketProtocol(unittest.TestCase):
    def setUp(self):
        self.service = build_service()
        self.client = TestClient(build_app(self.service))

    def _hello(self, ws, **overrides):
        payload = {
            "kind": "hello",
            "codecs": ["pcm16"],
            "participants": [LANG_A, LANG_B],
        }
        payload.update(overrides)
        ws.send_text(json.dumps(payload))
        return json.loads(ws.receive_text())

    def test_handshake_answers_with_the_negotiated_codec_and_languages(self):
        with self.client.websocket_connect("/api/translator/stream") as ws:
            ready = self._hello(ws, session_id="s1")
            self.assertEqual(ready["kind"], "ready")
            self.assertEqual(ready["session_id"], "s1")
            self.assertEqual(ready["codec"]["name"], "pcm16")
            self.assertEqual(ready["pipeline_sample_rate"], PIPELINE_SAMPLE_RATE)
            self.assertEqual(
                sorted(c["code"] for c in ready["languages"]["bidirectional"]),
                [LANG_A, LANG_B],
            )

    def test_a_non_hello_first_frame_is_refused(self):
        with self.client.websocket_connect("/api/translator/stream") as ws:
            ws.send_text(json.dumps({"kind": "audio"}))
            error = json.loads(ws.receive_text())
            self.assertEqual(error["kind"], "error")
            self.assertEqual(error["stage"], "codec")

    def test_a_full_turn_flows_over_the_socket(self):
        codec = Pcm16Codec(sample_rate=RATE, frame_ms=20)
        with self.client.websocket_connect("/api/translator/stream") as ws:
            self._hello(ws, session_id="s1")
            speech = AudioChunk(tone(VOICE_A_HZ, 2.0), RATE)
            for frame in codec.encode(speech):
                ws.send_bytes(frame)
            ws.send_text(json.dumps({"kind": "release"}))
            done, seen = drain_until(ws, lambda e: e.get("kind") == "turn.done")
            self.assertIsNotNone(done, [e.get("kind") for e in seen])
            kinds = [e["kind"] for e in seen]
            self.assertIn("turn.transcript", kinds)
            self.assertIn("turn.speaker", kinds)
            self.assertIn("turn.translation", kinds)
            self.assertEqual(done["source"], LANG_A)
            self.assertEqual(done["targets"], [LANG_B])
            self.assertIn("first_audio_ms", done["timings"])

    def test_reconnect_resumes_the_journal_from_the_cursor(self):
        codec = Pcm16Codec(sample_rate=RATE, frame_ms=20)
        with self.client.websocket_connect("/api/translator/stream") as ws:
            self._hello(ws, session_id="s1")
            for frame in codec.encode(AudioChunk(tone(VOICE_A_HZ, 2.0), RATE)):
                ws.send_bytes(frame)
            ws.send_text(json.dumps({"kind": "release"}))
            done, seen = drain_until(ws, lambda e: e.get("kind") == "turn.done")
            self.assertIsNotNone(done)
            cursor = max(e["seq"] for e in seen if "seq" in e) + 1

        # The session survives the dropped connection -- that is the point.
        self.assertIn("s1", self.service.sessions.ids())

        with self.client.websocket_connect("/api/translator/stream") as ws:
            ready = self._hello(ws, session_id="s1", resume_from=cursor)
            self.assertTrue(ready["resumed"])
            # The speaker registry survived, so the reference buffer did too.
            self.assertEqual(len(ready["state"]["speakers"]), 1)
            self.assertGreater(
                ready["state"]["speakers"][0]["reference_seconds"], 0.0
            )
            # Nothing before the cursor is replayed.
            replayed, _ = drain_until(
                ws, lambda e: e.get("seq", cursor) < cursor, budget_s=1.0
            )
            self.assertIsNone(replayed, "the server replayed acknowledged events")

    def test_a_cursor_below_the_floor_produces_an_explicit_gap(self):
        session = self.service.open_session([LANG_A, LANG_B], "s2")
        from sglang.srt.translator.session import EventKind

        for _ in range(300):
            session.journal.append(EventKind.SESSION_STATE, {"tick": True})
        self.assertGreater(session.journal.floor, 0)
        with self.client.websocket_connect("/api/translator/stream") as ws:
            self._hello(ws, session_id="s2", resume_from=0)
            gap, seen = drain_until(ws, lambda e: e.get("kind") == "resume.gap")
            self.assertIsNotNone(gap, [e.get("kind") for e in seen])
            self.assertEqual(gap["requested_from"], 0)
            self.assertGreater(gap["available_from"], 0)

    def test_ping_and_state_control_frames(self):
        with self.client.websocket_connect("/api/translator/stream") as ws:
            self._hello(ws, session_id="s1")
            ws.send_text(json.dumps({"kind": "ping"}))
            pong, _ = drain_until(ws, lambda e: e.get("kind") == "pong", budget_s=5.0)
            self.assertIsNotNone(pong)
            ws.send_text(json.dumps({"kind": "state"}))
            state, _ = drain_until(ws, lambda e: e.get("kind") == "state", budget_s=5.0)
            self.assertEqual(state["state"]["session_id"], "s1")

    def test_an_unknown_control_frame_is_reported_not_ignored(self):
        with self.client.websocket_connect("/api/translator/stream") as ws:
            self._hello(ws, session_id="s1")
            ws.send_text(json.dumps({"kind": "teleport"}))
            error, _ = drain_until(
                ws, lambda e: e.get("kind") == "error", budget_s=5.0
            )
            self.assertIsNotNone(error)
            self.assertEqual(error["stage"], "protocol")


class TestClientAsset(unittest.TestCase):
    def test_the_pwa_is_served_and_is_self_contained(self):
        client = TestClient(build_app(build_service()))
        page = client.get("/")
        self.assertEqual(page.status_code, 200)
        html = page.text
        # No build system, no CDN: the phone must be able to load this over a
        # tunnel with no internet route.
        for forbidden in ("http://", "https://cdn", "<script src="):
            self.assertNotIn(forbidden, html, forbidden)
        self.assertIn("getUserMedia", html)
        manifest = client.get("/manifest.webmanifest").json()
        self.assertEqual(manifest["display"], "standalone")


if __name__ == "__main__":
    unittest.main()
