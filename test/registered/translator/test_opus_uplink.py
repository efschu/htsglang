# Copyright 2025 SGLang Team
# Licensed under the Apache License, Version 2.0
"""The Opus uplink: negotiation, the directional split, and every fallback.

The uplink was PCM16 -- ~256 kbit/s continuously, in the direction the phone
pays for -- while the server had been able to decode Opus since the transport
was written. The client simply never offered it. Closing that gap is a
negotiation change, and a negotiation is exactly the kind of code that passes
its happy path and strands a user on the other three.

So the arms here are the four outcomes, not the one:

  1. the client offers Opus and the server takes it (the change);
  2. the client offers PCM16 only and nothing about the session moves --
     the regression arm for every deployed client and every harness script;
  3. the server has no Opus decoder (no PyAV) and answers an Opus offer with
     PCM16 -- the deployment where the dependency is missing must degrade,
     not fail at the first audio frame;
  4. the two directions disagree: the uplink runs Opus while the downlink
     stays PCM16, because the page's playback graph reads raw PCM
     synchronously and would render Opus packets as noise.

Arm 4 is the one with teeth. A single codec object served both directions
before, so an Opus offer used to switch the downlink too. The tests below
assert the downlink bytes are still PCM16 while the uplink is Opus, which is
the property that lets this ship without touching the playback path.

The client half is asserted as TEXT, which is weak and labelled as such: there
is no JavaScript runtime in this venv. The cross-stack question those pins
cannot reach -- do a browser's Opus packets decode here at all -- is answered
by ``scripts/translator/probe_opus_uplink.py``, which drives a real Chromium.

    CUDA_VISIBLE_DEVICES=99 python -m pytest \\
      test/registered/translator/test_opus_uplink.py -v
"""

import json
import math
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
from fastapi.testclient import TestClient

from sglang.srt.translator import audio as audio_module
from sglang.srt.translator.audio import (
    PIPELINE_SAMPLE_RATE,
    CodecError,
    OpusCodec,
    Pcm16Codec,
    available_codecs,
    from_pcm16_bytes,
    resample,
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
HAVE_OPUS = "opus" in available_codecs()

CLIENT = (
    Path(__file__).resolve().parents[3]
    / "python/sglang/srt/translator/client/index.html"
)


def tone(frequency, seconds, rate=RATE, amplitude=0.3):
    t = np.arange(int(seconds * rate), dtype=np.float32) / rate
    return (amplitude * np.sin(2.0 * math.pi * frequency * t)).astype(np.float32)


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


class TestTheHandshakeReportsBothDirections(unittest.TestCase):
    """`ready` has to say what it expects AND what it will send."""

    def setUp(self):
        self.client = TestClient(build_app(build_service()))

    def _hello(self, ws, codecs):
        ws.send_text(json.dumps({
            "kind": "hello", "codecs": codecs,
            "participants": [LANG_A, LANG_B],
        }))
        return json.loads(ws.receive_text())

    def test_a_pcm16_only_client_sees_exactly_what_it_always_saw(self):
        """The regression arm. Every deployed client and script is this one."""
        with self.client.websocket_connect("/api/translator/stream") as ws:
            ready = self._hello(ws, ["pcm16"])
            self.assertEqual(ready["codec"]["name"], "pcm16")
            self.assertEqual(ready["codec"]["sample_rate"], PIPELINE_SAMPLE_RATE)
            self.assertEqual(ready["downlink_codec"]["name"], "pcm16")

    @unittest.skipUnless(HAVE_OPUS, "no Opus decoder in this environment")
    def test_an_opus_offer_moves_the_uplink_and_only_the_uplink(self):
        with self.client.websocket_connect("/api/translator/stream") as ws:
            ready = self._hello(ws, ["opus", "pcm16"])
            self.assertEqual(ready["codec"]["name"], "opus")
            self.assertEqual(
                ready["downlink_codec"]["name"], "pcm16",
                "the page cannot decode Opus; answering with it is noise",
            )

    @unittest.skipUnless(HAVE_OPUS, "no Opus decoder in this environment")
    def test_the_uplink_codec_reports_the_bitrate_it_will_be_fed(self):
        with self.client.websocket_connect("/api/translator/stream") as ws:
            ready = self._hello(ws, ["opus", "pcm16"])
            self.assertEqual(ready["codec"]["bitrate_bps"], 24000)

    def test_a_server_without_opus_answers_an_opus_offer_with_pcm16(self):
        """The deployment where PyAV is missing must degrade, not refuse.

        Forced rather than waited for: this venv HAS PyAV, so the arm would
        otherwise never execute anywhere it matters. Patched at the module the
        server imported from, which is also the check that the negotiation
        really consults `available_codecs` instead of a hardcoded preference.
        """
        with mock.patch.object(
            audio_module, "available_codecs", return_value=("pcm16",)
        ):
            with self.client.websocket_connect("/api/translator/stream") as ws:
                ready = self._hello(ws, ["opus", "pcm16"])
                self.assertEqual(ready["codec"]["name"], "pcm16")
                self.assertEqual(ready["downlink_codec"]["name"], "pcm16")

    def test_a_client_offering_nothing_the_server_has_is_still_refused(self):
        """The negotiation must not have become permissive on the way."""
        with self.client.websocket_connect("/api/translator/stream") as ws:
            ws.send_text(json.dumps({"kind": "hello", "codecs": ["speex"]}))
            answer = json.loads(ws.receive_text())
            self.assertEqual(answer["kind"], "error")
            self.assertEqual(answer["stage"], "codec")

    def test_an_absent_codecs_field_still_means_pcm16(self):
        with self.client.websocket_connect("/api/translator/stream") as ws:
            ws.send_text(json.dumps({
                "kind": "hello", "participants": [LANG_A, LANG_B],
            }))
            ready = json.loads(ws.receive_text())
            self.assertEqual(ready["codec"]["name"], "pcm16")


class TestTheDirectionsDoNotBleedIntoEachOther(unittest.TestCase):
    """The bytes on each leg, read as the other end would read them."""

    def setUp(self):
        self.service = build_service()
        self.client = TestClient(build_app(self.service))

    def _hello(self, ws, codecs):
        ws.send_text(json.dumps({
            "kind": "hello", "codecs": codecs, "session_id": "split",
            "participants": [LANG_A, LANG_B],
        }))
        return json.loads(ws.receive_text())

    @unittest.skipUnless(HAVE_OPUS, "no Opus decoder in this environment")
    def test_opus_uplink_frames_reach_the_session_as_audio(self):
        """Not merely accepted: the samples have to arrive at the segmenter."""
        speech = AudioChunk(tone(VOICE_A_HZ, 1.5), RATE)
        packets = list(OpusCodec().encode(speech))
        self.assertGreater(len(packets), 10, "the fixture produced no packets")

        with self.client.websocket_connect("/api/translator/stream") as ws:
            self._hello(ws, ["opus", "pcm16"])
            ws.receive_text()  # the transcript frame
            for packet in packets:
                ws.send_bytes(packet)
            ws.send_text(json.dumps({"kind": "state"}))
            state = None
            while state is None:
                message = ws.receive()
                text = message.get("text")
                if text is None:
                    continue
                event = json.loads(text)
                if event.get("kind") == "state":
                    state = event
        session = self.service.sessions.get("split")
        self.assertIsNotNone(session)
        # Read out what actually landed in the segmenter. A PCM16 server would
        # have read these ~30-byte packets as 15-sample runs and accumulated a
        # few hundredths of a second; a working Opus decode yields ~1.5 s.
        segment = session.release()
        self.assertIsNotNone(segment, "no audio reached the segmenter at all")
        self.assertGreater(segment.duration_s, 0.8)

    @unittest.skipUnless(HAVE_OPUS, "no Opus decoder in this environment")
    def test_the_downlink_stays_readable_as_pcm16_under_an_opus_uplink(self):
        """The whole reason the split exists, asserted on the bytes.

        A shared codec would have made these frames Opus packets, which
        `from_pcm16_bytes` would read as noise of the wrong length. Reading
        them back as PCM16 and finding the announced duration is the proof
        the downlink did not move.
        """
        with self.client.websocket_connect("/api/translator/stream") as ws:
            ready = self._hello(ws, ["opus", "pcm16"])
            self.assertEqual(ready["codec"]["name"], "opus")
            ws.receive_text()  # the transcript frame
            for packet in OpusCodec().encode(AudioChunk(tone(VOICE_A_HZ, 3.0), RATE)):
                ws.send_bytes(packet)
            ws.send_text(json.dumps({"kind": "release"}))

            announced = None
            payload = b""
            import time
            deadline = time.monotonic() + 20.0
            while time.monotonic() < deadline:
                ws.send_text(json.dumps({"kind": "ping"}))
                message = ws.receive()
                if message.get("type") == "websocket.disconnect":
                    break
                if (data := message.get("bytes")) is not None:
                    if announced is not None:
                        payload += data
                    continue
                event = json.loads(message["text"])
                if event.get("audio_follows"):
                    announced = event["sample_rate"]
                if payload:
                    break
            self.assertIsNotNone(announced, "no audio was ever announced")
            self.assertEqual(
                announced, PIPELINE_SAMPLE_RATE,
                "the downlink announced a rate that is not the PCM16 codec's",
            )
            self.assertTrue(payload, "no audio frames arrived")
            # The decisive line: these bytes parse as PCM16 at the announced
            # rate. Opus packets would not have an even length by luck alone
            # over this many frames, and would not decode to a sane peak.
            chunk = from_pcm16_bytes(payload, announced)
            self.assertGreater(len(chunk.samples) / announced, 0.05)
            self.assertLessEqual(float(np.abs(chunk.samples).max()), 1.0)


class TestABadUplinkFrameDoesNotEndTheConversation(unittest.TestCase):
    """One lost packet on a mobile link must not cost the socket.

    Reachable only now. Any even-length byte string decodes as PCM16, so this
    class of failure arrived with the compressed uplink: a truncated Opus
    packet makes the decoder raise, and that exception is not a `CodecError`,
    so it would have escaped the receive loop.
    """

    def setUp(self):
        self.service = build_service()
        self.client = TestClient(build_app(self.service))

    @unittest.skipUnless(HAVE_OPUS, "no Opus decoder in this environment")
    def test_garbage_is_dropped_and_the_session_keeps_going(self):
        good = list(OpusCodec().encode(AudioChunk(tone(VOICE_A_HZ, 1.5), RATE)))
        with self.client.websocket_connect("/api/translator/stream") as ws:
            ws.send_text(json.dumps({
                "kind": "hello", "codecs": ["opus", "pcm16"],
                "session_id": "bad", "participants": [LANG_A, LANG_B],
            }))
            ready = json.loads(ws.receive_text())
            self.assertEqual(ready["codec"]["name"], "opus")
            ws.receive_text()  # the transcript frame
            ws.send_bytes(b"\xff\xfe\xfd\xfc\xfb")   # not an Opus packet
            ws.send_bytes(b"")                        # not anything
            for packet in good:
                ws.send_bytes(packet)
            ws.send_text(json.dumps({"kind": "state"}))
            state = None
            while state is None:
                message = ws.receive()
                self.assertNotEqual(
                    message.get("type"), "websocket.disconnect",
                    "a corrupt frame closed the socket",
                )
                if (text := message.get("text")) is None:
                    continue
                event = json.loads(text)
                if event.get("kind") == "state":
                    state = event
        session = self.service.sessions.get("bad")
        segment = session.release()
        self.assertIsNotNone(
            segment, "the good packets after the garbage were not decoded"
        )
        self.assertGreater(segment.duration_s, 0.8)

    def test_the_can_fail_control_an_undecodable_frame_would_otherwise_raise(self):
        """The guard is only worth having if what it catches really throws."""
        if not HAVE_OPUS:
            self.skipTest("no Opus decoder in this environment")
        with self.assertRaises(Exception):
            OpusCodec().decode(b"\xff\xfe\xfd\xfc\xfb")

    @unittest.skipUnless(HAVE_OPUS, "no Opus decoder in this environment")
    def test_an_empty_frame_is_refused_rather_than_draining_the_decoder(self):
        """The silent-deafness case, which is why the guard also RESETS.

        FFmpeg reads a zero-length packet as end-of-stream. Left to itself the
        decoder then answers every later packet with EOF, so one stray empty
        binary frame from a phone would end the uplink for the whole session
        with nothing anywhere reporting a fault.
        """
        codec = OpusCodec()
        with self.assertRaises(CodecError):
            codec.decode(b"")

    @unittest.skipUnless(HAVE_OPUS, "no Opus decoder in this environment")
    def test_the_can_fail_control_a_drained_decoder_really_stays_dead(self):
        """The measurement behind the paragraph above, not a claim about it.

        Reaches under `decode`'s refusal on purpose: the point is what the
        raw decoder does, because that is what the refusal and the reset are
        protecting against. Without a reset it decodes nothing ever again.
        """
        import av

        packets = list(OpusCodec().encode(AudioChunk(tone(220.0, 1.0), RATE)))
        codec = OpusCodec()
        codec._get_decoder().decode(av.Packet(b""))   # the flush
        with self.assertRaises(Exception):
            for packet in packets:
                codec.decode(packet)
        codec.reset()
        recovered = sum(len(codec.decode(p).samples) for p in packets)
        self.assertGreater(recovered, 0, "reset() did not revive the decoder")


class TestFramingIsRawPacketsNotAContainer(unittest.TestCase):
    """The decision this change turned on, pinned so it cannot drift.

    WebCodecs `AudioEncoder` emits one bare Opus packet per frame with the
    default `opus.format` of "opus" -- no Ogg page, no WebM cluster, no
    `OpusHead` extradata. The server decodes bare packets. Neither side had to
    move, which is why this stayed a client-side change; if either end ever
    grows a container, these are the assertions that fail first.
    """

    @unittest.skipUnless(HAVE_OPUS, "no Opus decoder in this environment")
    def test_a_bare_packet_decodes_without_any_container_header(self):
        packets = list(OpusCodec().encode(AudioChunk(tone(220.0, 0.5), RATE)))
        first = packets[0]
        for magic in (b"OggS", b"\x1a\x45\xdf\xa3", b"OpusHead"):
            self.assertFalse(
                first.startswith(magic),
                f"the encoder emitted a {magic!r} container, not a raw packet",
            )
        decoded = OpusCodec().decode(first)
        self.assertGreater(len(decoded.samples), 0)

    @unittest.skipUnless(HAVE_OPUS, "no Opus decoder in this environment")
    def test_sixteen_kilohertz_input_decodes_at_the_decoder_s_own_rate(self):
        """Why the client may encode at 16 kHz against a 48 kHz decoder.

        Opus is defined at 48 kHz internally and its packets carry their own
        bandwidth, so the input rate the encoder was fed is not a property of
        the wire. The client feeds 16 kHz -- the rate the capture loop already
        produces, so the PCM16 and Opus paths carry identical audio and a
        before/after WER comparison measures the codec rather than two
        different resamplers.
        """
        source = AudioChunk(tone(220.0, 1.0, rate=16000), 16000)
        packets = list(OpusCodec(sample_rate=16000).encode(source))
        decoder = OpusCodec()  # 48 kHz, exactly as the server constructs it
        samples = np.concatenate([decoder.decode(p).samples for p in packets])
        chunk = resample(
            AudioChunk(samples, decoder.sample_rate), PIPELINE_SAMPLE_RATE
        )
        spectrum = np.abs(np.fft.rfft(chunk.samples))
        freqs = np.fft.rfftfreq(len(chunk.samples), 1.0 / PIPELINE_SAMPLE_RATE)
        self.assertAlmostEqual(
            float(freqs[int(np.argmax(spectrum))]), 220.0, delta=15.0
        )
        self.assertAlmostEqual(
            len(chunk.samples) / PIPELINE_SAMPLE_RATE, 1.0, delta=0.1
        )

    @unittest.skipUnless(HAVE_OPUS, "no Opus decoder in this environment")
    def test_the_uplink_is_an_order_of_magnitude_cheaper(self):
        """The requirement, not a side effect. Measured on the wire bytes."""
        source = AudioChunk(tone(220.0, 2.0), RATE)
        opus_bytes = sum(len(p) for p in OpusCodec().encode(source))
        pcm_bytes = sum(len(f) for f in Pcm16Codec().encode(source))
        self.assertGreater(pcm_bytes, 8 * opus_bytes)
        self.assertLess(opus_bytes * 8 / 2.0, 40000, "over the link budget")


class TestTheShippedClientCarriesTheFallbackLadder(unittest.TestCase):
    """Text pins. Weak, and labelled: there is no JS runtime in this venv.

    They exist because the four fallbacks are the part of this change that
    nothing else can reach from here, and a fallback that gets edited away is
    invisible until a browser without WebCodecs loads the page.
    """

    def setUp(self):
        self.html = CLIENT.read_text(encoding="utf-8")

    def test_the_hello_offers_what_the_browser_can_encode(self):
        self.assertIn("codecs: uplinkOffer()", self.html)
        self.assertNotIn(
            'codecs: ["pcm16"]', self.html,
            "the hardcoded offer is the dead branch this change removes",
        )

    def test_the_offer_is_made_only_on_a_definite_probe_answer(self):
        self.assertIn("opusEncodeSupported === true", self.html)
        self.assertIn("let opusEncodeSupported = null", self.html)

    def test_the_probe_runs_before_the_first_connect(self):
        self.assertIn("probeOpusUplink().then(() => connection.connect())",
                      self.html)

    def test_a_missing_webcodecs_lands_on_pcm16(self):
        self.assertIn('typeof AudioEncoder === "undefined"', self.html)
        self.assertIn("isConfigSupported", self.html)

    def test_a_server_that_does_not_split_the_directions_is_refused(self):
        self.assertIn('downlink !== "pcm16"', self.html)
        self.assertIn("downgradeUplink", self.html)

    def test_the_downgrade_is_permanent_for_the_life_of_the_page(self):
        # A reconnect loop that re-offers Opus after a failure would downgrade
        # forever without ever sending a turn.
        self.assertIn("if (opusUplinkDisabled) return;", self.html)
        self.assertIn("opusUplinkDisabled = true;", self.html)

    def test_nothing_is_sent_between_the_offer_and_the_answer(self):
        self.assertIn("uplink = new StalledUplink();", self.html)

    def test_the_encoder_config_matches_the_link_budget(self):
        self.assertIn("const UPLINK_BITRATE = 24000;", self.html)
        self.assertIn("opus: { frameDuration: FRAME_MS * 1000 }", self.html)
        self.assertIn("sampleRate: PIPELINE_RATE", self.html)

    def test_enrolment_still_captures_pcm16_and_bypasses_the_codec(self):
        """The collateral this change nearly broke, pinned.

        Enrolment used to divert the capture by swapping out the microphone's
        send callback. Giving the uplink its own codec removed that callback,
        which would have left enrolment silently collecting nothing -- and
        enrolment posts base64 PCM16 to an endpoint that accepts no other
        encoding, so it must bypass the uplink codec rather than follow it.
        """
        self.assertIn("microphone.tap = capture;", self.html)
        self.assertIn("microphone.tap = null;", self.html)
        self.assertIn("if (this.tap) this.tap(packPcm16(samples));", self.html)
        self.assertIn("else uplink.frame(samples);", self.html)

    def test_the_uplink_cost_is_reported_as_a_rate(self):
        # A total without the time it accumulated over is not a bitrate, and
        # the before/after evidence for this change is read out of this field.
        self.assertIn("bytes_per_second", self.html)
        self.assertIn("uplink: uplinkReport()", self.html)


if __name__ == "__main__":
    unittest.main()
