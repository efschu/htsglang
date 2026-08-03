# Copyright 2025 SGLang Team
# Licensed under the Apache License, Version 2.0
"""Transport: codec negotiation, resampling, and the HTTP/WebSocket surface.

Hermetic. The WebSocket tests drive Starlette's in-process test client, so no
socket is opened and no model is loaded; the backends are the fakes.

    CUDA_VISIBLE_DEVICES=99 python -m pytest test/registered/translator/test_audio_and_http.py -v
"""

import time
import json
import re
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

    def test_a_stop_frame_is_acked_and_names_what_it_abandoned(self):
        """The wiring, which every session-level test would pass without.

        A frame the dispatcher does not know is silently ignored, so the
        button would be dead on a server whose `abort_playback` is perfect.
        """
        with self.client.websocket_connect("/api/translator/stream") as ws:
            self._hello(ws, session_id="stop-1")
            ws.send_text(json.dumps({"kind": "playback.stop"}))
            ack, seen = drain_until(
                ws, lambda e: e.get("kind") == "playback.stop.ack"
            )
            self.assertIsNotNone(ack, [e.get("kind") for e in seen])
            self.assertEqual(ack["dropped_queued"], 0)
            self.assertEqual(ack["stop_epoch"], 1)
            self.assertIn("aborted_turn_id", ack)

    def test_a_stop_is_delivered_to_the_client_as_an_event_too(self):
        """The ack is for the presser; the event is for the record."""
        with self.client.websocket_connect("/api/translator/stream") as ws:
            self._hello(ws, session_id="stop-2")
            ws.send_text(json.dumps({"kind": "playback.stop"}))
            event, seen = drain_until(
                ws, lambda e: e.get("kind") == "playback.stopped"
            )
            self.assertIsNotNone(event, [e.get("kind") for e in seen])
            self.assertEqual(event["reason"], "user")

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

    def test_the_announced_rate_is_the_rate_on_the_wire(self):
        """The Mickey Mouse falsifier.

        The codec resamples to its own rate on the way out, so announcing the
        session-side rate described samples that no longer existed: the TTS
        produces 24 kHz, the codec ships 16 kHz, and a client that trusts the
        announcement plays it 1.5x too fast and a fifth too high. The
        front-door harness cannot catch this -- it decodes with the codec and
        ignores the field -- so only a client ever suffered it.
        """
        codec = Pcm16Codec(sample_rate=RATE, frame_ms=20)
        with self.client.websocket_connect("/api/translator/stream") as ws:
            ready = self._hello(ws, session_id="s1")
            wire_rate = ready["codec"]["sample_rate"]
            speech = AudioChunk(tone(VOICE_A_HZ, 2.0), RATE)
            for frame in codec.encode(speech):
                ws.send_bytes(frame)
            ws.send_text(json.dumps({"kind": "release"}))
            done, seen = drain_until(ws, lambda e: e.get("kind") == "turn.done")
            self.assertIsNotNone(done, [e.get("kind") for e in seen])
            announced = [
                e["sample_rate"] for e in seen
                if e.get("audio_follows") and "sample_rate" in e
            ]
            self.assertTrue(announced, "an audio event must announce its rate")
            for rate in announced:
                self.assertEqual(
                    rate, wire_rate,
                    "the announced rate must be the codec's, not the "
                    "session's -- anything else is a pitch shift on a phone",
                )

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


class TestTranscriptOverTheWire(unittest.TestCase):
    """The written record through the socket and the REST surface (§17.2)."""

    def setUp(self):
        self.service = build_service()
        self.client = TestClient(build_app(self.service))
        self.codec = Pcm16Codec(sample_rate=RATE, frame_ms=20)

    def _hello(self, ws, **overrides):
        payload = {
            "kind": "hello",
            "codecs": ["pcm16"],
            "participants": [LANG_A, LANG_B],
        }
        payload.update(overrides)
        ws.send_text(json.dumps(payload))
        return json.loads(ws.receive_text())

    def _one_turn(self, ws):
        for frame in self.codec.encode(AudioChunk(tone(VOICE_A_HZ, 2.0), RATE)):
            ws.send_bytes(frame)
        ws.send_text(json.dumps({"kind": "release"}))
        done, seen = drain_until(ws, lambda e: e.get("kind") == "turn.done")
        self.assertIsNotNone(done, [e.get("kind") for e in seen])
        return seen

    def test_the_handshake_delivers_the_record_before_anything_else(self):
        with self.client.websocket_connect("/api/translator/stream") as ws:
            ready = self._hello(ws, session_id="s1")
            self.assertEqual(ready["kind"], "ready")
            record = json.loads(ws.receive_text())
            self.assertEqual(record["kind"], "transcript")
            self.assertEqual(record["lines"], [])

    def test_a_reconnect_from_zero_restores_the_conversation_whole(self):
        """The phone whose tab was evicted.

        It has no lines and no cursor, so it asks from zero. If the record
        lived on the client this would come back empty, and if it were
        derived from the journal it would come back truncated.
        """
        with self.client.websocket_connect("/api/translator/stream") as ws:
            self._hello(ws, session_id="s1")
            ws.receive_text()  # the empty record
            self._one_turn(ws)

        with self.client.websocket_connect("/api/translator/stream") as ws:
            ready = self._hello(ws, session_id="s1", transcript_from=0)
            self.assertTrue(ready["resumed"])
            record = json.loads(ws.receive_text())
            self.assertEqual(record["kind"], "transcript")
            self.assertEqual(len(record["lines"]), 1)
            line = record["lines"][0]
            self.assertEqual(line["source_language"], LANG_A)
            self.assertTrue(line["source_text"])
            self.assertIn(LANG_B, line["translations"])

    def test_a_client_that_kept_its_lines_asks_only_for_the_tail(self):
        with self.client.websocket_connect("/api/translator/stream") as ws:
            self._hello(ws, session_id="s1")
            ws.receive_text()
            self._one_turn(ws)
            self._one_turn(ws)

        with self.client.websocket_connect("/api/translator/stream") as ws:
            self._hello(ws, session_id="s1", transcript_from=1)
            record = json.loads(ws.receive_text())
            self.assertEqual([line["line_id"] for line in record["lines"]], [2])

    def test_the_rest_surface_reads_and_clears_the_record(self):
        with self.client.websocket_connect("/api/translator/stream") as ws:
            self._hello(ws, session_id="s1")
            ws.receive_text()
            self._one_turn(ws)

        got = self.client.get("/api/translator/sessions/s1/transcript").json()
        self.assertEqual(len(got["lines"]), 1)
        self.assertEqual(got["dropped"], 0)

        cleared = self.client.delete("/api/translator/sessions/s1/transcript").json()
        self.assertEqual(cleared["removed"], 1)
        after = self.client.get("/api/translator/sessions/s1/transcript").json()
        self.assertEqual(after["lines"], [])
        # And it is gone for a reconnecting client too -- the clear is
        # server-side, so a second device sees the same empty record.
        self.assertEqual(
            self.client.get(
                "/api/translator/sessions/s1/transcript?since=0"
            ).json()["lines"],
            [],
        )

    def test_reading_mode_switches_over_the_socket_and_over_rest(self):
        with self.client.websocket_connect("/api/translator/stream") as ws:
            ready = self._hello(ws, session_id="s1")
            self.assertEqual(ready["state"]["output_mode"], "voice")
            ws.receive_text()  # the record
            ws.send_text(json.dumps({"kind": "output.mode", "mode": "silent"}))
            answer, seen = drain_until(
                ws, lambda e: e.get("kind") == "output.mode", budget_s=5.0
            )
            self.assertIsNotNone(answer, [e.get("kind") for e in seen])
            self.assertEqual(answer["mode"], "silent")
            # An unknown mode is refused by name rather than ignored.
            ws.send_text(json.dumps({"kind": "output.mode", "mode": "mute"}))
            error, _ = drain_until(
                ws, lambda e: e.get("kind") == "error", budget_s=5.0
            )
            self.assertIsNotNone(error)
            self.assertIn("mute", error["message"])

        state = self.client.post(
            "/api/translator/sessions/s1/voice", json={"output_mode": "voice"}
        ).json()
        self.assertEqual(state["output_mode"], "voice")

    def test_transcript_of_an_unknown_session_is_a_404(self):
        self.assertEqual(
            self.client.get("/api/translator/sessions/nope/transcript").status_code,
            404,
        )

    def test_naming_a_speaker_over_the_socket_rewrites_the_record(self):
        with self.client.websocket_connect("/api/translator/stream") as ws:
            self._hello(ws, session_id="s1")
            ws.receive_text()
            self._one_turn(ws)
            speaker_id = self.service.sessions.get("s1").transcript.lines()[0].speaker_id
            ws.send_text(
                json.dumps(
                    {"kind": "speaker.name", "speaker_id": speaker_id,
                     "label": "Matthias"}
                )
            )
            named, seen = drain_until(
                ws, lambda e: e.get("kind") == "speaker.named", budget_s=5.0
            )
            self.assertIsNotNone(named, [e.get("kind") for e in seen])
            self.assertEqual(named["lines_updated"], 1)
        record = self.client.get("/api/translator/sessions/s1/transcript").json()
        self.assertEqual(record["lines"][0]["speaker_label"], "Matthias")

    def test_naming_an_unknown_speaker_is_reported_not_ignored(self):
        with self.client.websocket_connect("/api/translator/stream") as ws:
            self._hello(ws, session_id="s1")
            ws.receive_text()
            ws.send_text(
                json.dumps({"kind": "speaker.name", "speaker_id": "ghost",
                            "label": "x"})
            )
            error, _ = drain_until(
                ws, lambda e: e.get("kind") == "error", budget_s=5.0
            )
            self.assertIsNotNone(error)
            self.assertEqual(error["stage"], "speaker")

    def test_the_rest_naming_endpoint_matches_the_socket(self):
        with self.client.websocket_connect("/api/translator/stream") as ws:
            self._hello(ws, session_id="s1")
            ws.receive_text()
            self._one_turn(ws)
        speaker_id = self.service.sessions.get("s1").transcript.lines()[0].speaker_id
        answer = self.client.post(
            f"/api/translator/sessions/s1/speakers/{speaker_id}/name",
            json={"label": "Larisa"},
        ).json()
        self.assertEqual(answer["lines_updated"], 1)
        self.assertEqual(
            self.client.post(
                "/api/translator/sessions/s1/speakers/ghost/name",
                json={"label": "x"},
            ).status_code,
            404,
        )


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

        # The first real phone test failed here, and the server looked
        # perfectly healthy while it did: nine WebSocket connections, two
        # sessions, zero segments. The capture AudioContext is created after
        # an await on getUserMedia -- which on the first press also spans the
        # Android permission dialog -- so the user gesture has expired and
        # Chrome starts it SUSPENDED. A suspended context never runs the
        # worklet, so no frame is ever produced and nothing is ever sent.
        #
        # These are string assertions on an asset, which is a weak instrument;
        # they are here because the alternative is no instrument at all, and
        # because each one pins a specific line whose removal reproduces a
        # failure that is invisible from the server side.
        self.assertIn("await this.ctx.resume()", html,
                      "the capture context must be resumed, not assumed")
        self.assertIn('this.ctx.state !== "running"', html,
                      "a backgrounded page suspends capture; resume on reopen")
        self.assertIn("microphone unavailable", html,
                      "a refused microphone must be reported, not swallowed")
        self.assertIn("no audio is arriving", html,
                      "an open microphone producing nothing must say so")
        self.assertIn("released without any audio", html,
                      "a release with no audio is indistinguishable from "
                      "silence at the server, so the client must report it")
        # The build stamp must be substituted, not shipped as a placeholder:
        # a page that cannot say which build it is makes every phone-side
        # diagnosis a guess about which code is running.
        self.assertNotIn("__CLIENT_BUILD__", html)
        stamp = re.search(r'const CLIENT_BUILD = "([0-9a-f]{10})"', html)
        self.assertIsNotNone(stamp, "the client must carry a build identity")
        # Stable across requests, different when the asset differs.
        again = client.get("/")
        self.assertEqual(
            re.search(r'const CLIENT_BUILD = "([0-9a-f]{10})"', again.text)
            .group(1),
            stamp.group(1),
        )
        self.assertIn("no-cache", again.headers.get("cache-control", ""))

        # The section 17 surface must actually be on the screen. It lived in
        # the protocol and in 380 tests for a while, which made "implemented"
        # read as "usable" -- the phone could reach none of it.
        for marker in ('id="transcript"', 'id="speakers"', 'id="chips"'):
            self.assertIn(marker, html, marker)
        self.assertIn("bindTalkButton", html,
                      "each speaker needs its own hold-to-speak button")
        self.assertIn('kind: "speaker.arm"', html)
        self.assertIn('kind: "line.resolve"', html)
        self.assertIn('kind: "suggestion.confirm"', html)
        # Auto-scroll that yields to a reader who scrolled up.
        self.assertIn("function atBottom", html)
        # The BEHAVIOUR pinned here is unchanged and is the one that matters:
        # auto-scroll happens only when the reader was already at the bottom,
        # so it never yanks the page away from somebody reading back.
        #
        # Its MECHANISM changed and this pin changed with it. The old shape
        # sampled `atBottom()` at append time and scrolled once
        # (`if (wasAtBottom) streamBox().scrollTop = ...`), which is stale by
        # the time the bubble finishes growing -- the queued notice, the
        # clause partials and the final text all arrive after the append and
        # none of them scrolled. Measured on the pre-fix client: the newest
        # line ends up 5 px below the fold on the first growth and never
        # recovers, because `atBottom()` then reads false for good.
        # `scripts/translator/probe_autoscroll.py` is the executing arm; this
        # is the structural pin that the mechanism is still the observer one.
        self.assertIn("let following = true", html)
        self.assertIn("new MutationObserver(restick)", html)
        self.assertIn("if (following) scrollToEnd()", html)
        # And the reader who IS scrolled up is told that something arrived,
        # instead of the app looking frozen to the one person reading it.
        self.assertIn('id="unread"', html)
        # Tap-to-toggle, not press-and-hold: on Android a long press is the
        # text-selection gesture and it cancelled the recording mid-word.
        self.assertIn("function toggleTalk", html)
        self.assertIn("tap to stop", html)
        self.assertIn("MIN_TURN_MS", html, "a stray double tap must not "
                      "send an empty turn silently")
        for guard in ("-webkit-touch-callout", "user-select", "contextmenu",
                      "touch-action"):
            self.assertIn(guard, html, guard)
        # Both texts on every line. The original is the only thing a speaker
        # can check the translation against.
        # Same invariant after the §19.3 rebuild, different selectors: the
        # source text is rendered on every line beside its translation, per
        # target. The bilingual line is the point -- a speaker can only check
        # a translation against their own words.
        self.assertIn('.src").textContent = line.source_text', html)
        self.assertIn("text.textContent = line.translations[target]", html)

        manifest = client.get("/manifest.webmanifest").json()
        self.assertEqual(manifest["display"], "standalone")


if __name__ == "__main__":
    unittest.main()


class TestReplayIsMarkedAsHistory(unittest.TestCase):
    """A resumed journal must not be mistaken for something just said.

    A freshly loaded page resumes from cursor zero, so the journal answers
    with every ``turn.audio`` the conversation ever produced. Played as if
    live, that is the whole conversation at once out of the speaker at the
    first tap -- which is what a real device reported as "noise starts as
    soon as I press, before I have said anything". The client can only refuse
    to speak history if the server says which events are history.
    """

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

    def _run_one_turn(self):
        codec = Pcm16Codec(sample_rate=RATE, frame_ms=20)
        with self.client.websocket_connect("/api/translator/stream") as ws:
            self._hello(ws, session_id="replay")
            speech = AudioChunk(tone(VOICE_A_HZ, 2.0), RATE)
            for frame in codec.encode(speech):
                ws.send_bytes(frame)
            ws.send_text(json.dumps({"kind": "release"}))
            deadline = time.monotonic() + 20
            while time.monotonic() < deadline:
                message = ws.receive()
                if "text" in message and message["text"]:
                    event = json.loads(message["text"])
                    if event.get("kind") == "turn.done":
                        return

    def test_live_audio_is_not_marked_replayed(self):
        seen = []
        codec = Pcm16Codec(sample_rate=RATE, frame_ms=20)
        with self.client.websocket_connect("/api/translator/stream") as ws:
            self._hello(ws, session_id="live")
            speech = AudioChunk(tone(VOICE_A_HZ, 2.0), RATE)
            for frame in codec.encode(speech):
                ws.send_bytes(frame)
            ws.send_text(json.dumps({"kind": "release"}))
            deadline = time.monotonic() + 20
            while time.monotonic() < deadline:
                message = ws.receive()
                if "text" in message and message["text"]:
                    event = json.loads(message["text"])
                    seen.append(event)
                    if event.get("kind") == "turn.done":
                        break
        audio = [e for e in seen if e.get("kind") == "turn.audio"]
        self.assertTrue(audio, "the turn produced no audio to judge")
        for event in audio:
            self.assertNotIn(
                "replayed", event,
                "live audio must never be tagged as history, or the client "
                "would refuse to play the turn that just happened",
            )

    def test_resumed_audio_is_marked_replayed(self):
        self._run_one_turn()
        replayed = []
        with self.client.websocket_connect("/api/translator/stream") as ws:
            self._hello(ws, session_id="replay", resume_from=0)
            deadline = time.monotonic() + 10
            while time.monotonic() < deadline:
                message = ws.receive()
                if "text" not in message or not message["text"]:
                    continue
                event = json.loads(message["text"])
                if event.get("kind") == "turn.audio":
                    replayed.append(event)
                if event.get("kind") == "turn.done":
                    break
        self.assertTrue(replayed, "resuming from zero replayed no audio")
        for event in replayed:
            self.assertTrue(
                event.get("replayed"),
                "replayed audio was indistinguishable from live audio, which "
                "is what made a fresh page load play the whole conversation",
            )


class TestAStaleCursorCannotSwallowLiveDelivery(unittest.TestCase):
    """A cursor from a session that no longer exists must not silence the new one.

    Reported from a phone: after a reload the previously spoken text is there,
    but speaking again updates nothing and produces no sound -- and reloading
    again shows the new translation. So the uplink, the recognizer, the
    translator and the synthesizer all worked; only live delivery did not.

    The mechanism is one cursor. The client persists ``translator.cursor``
    beside ``translator.session`` but resets neither when the server hands back
    a DIFFERENT session id, which is exactly what happens once the idle
    collector has taken the old session. The handshake then seeds the delivery
    cursor from that stale number, and ``Journal.since()`` answers a cursor
    past the end with an empty list and no gap -- so ``_journal_pump`` has
    nothing to send until the new journal organically grows past a high-water
    mark belonging to a different conversation.

    The reload appears to work because the written record travels on the
    handshake's ``transcript`` frame, which is keyed on a separate cursor the
    client does not persist. One defect, both symptoms: no live text and no
    live audio.
    """

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

    def test_a_cursor_beyond_the_journal_still_delivers_the_next_turn(self):
        codec = Pcm16Codec(sample_rate=RATE, frame_ms=20)
        with self.client.websocket_connect("/api/translator/stream") as ws:
            # A brand-new session, and a cursor that can only have come from a
            # different one. Unfixed, this silences every live event forever.
            self._hello(ws, session_id="minted-fresh", resume_from=5000)
            speech = AudioChunk(tone(VOICE_A_HZ, 2.0), RATE)
            for frame in codec.encode(speech):
                ws.send_bytes(frame)
            ws.send_text(json.dumps({"kind": "release"}))
            done, seen = drain_until(ws, lambda e: e.get("kind") == "turn.done")

        kinds = [e.get("kind") for e in seen]
        self.assertIsNotNone(
            done,
            "a stale cursor swallowed the whole turn: the client spoke, the "
            f"pipeline ran, and nothing was delivered live. saw {kinds}",
        )
        self.assertIn("turn.transcript", kinds)
        self.assertIn("turn.audio", kinds)

    def test_the_handshake_reports_whether_it_actually_resumed(self):
        """``resumed`` must describe the session, not the request.

        It used to be ``bool(session_id)`` -- true whenever the client named
        any id, including one the server had already collected and was now
        minting fresh. A client cannot decide whether to keep or drop its
        cursor from an answer that is true either way.
        """
        with self.client.websocket_connect("/api/translator/stream") as ws:
            ready = self._hello(ws, session_id="never-seen-before")
            self.assertFalse(
                ready["resumed"],
                "a session the server had to mint was reported as resumed",
            )
        with self.client.websocket_connect("/api/translator/stream") as ws:
            ready = self._hello(ws, session_id="never-seen-before")
            self.assertTrue(
                ready["resumed"], "an existing session was reported as new"
            )
