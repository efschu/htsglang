# Copyright 2025 SGLang Team
# Licensed under the Apache License, Version 2.0
"""The real TTS adapter, against a stub of the serving surface.

Hermetic but NOT mocked: a real uvicorn server implementing vLLM-Omni's two
audio endpoints runs on a loopback port, and the adapter talks to it over real
HTTP. Mocking the client would test the adapter's shape and none of its
behaviour -- the interesting parts (streaming PCM reassembly across chunk
boundaries, voice-registry caching, error surfacing) only exist on the wire.

The language set is read from the actual downloaded checkpoint's config.json
when it is present, which is the requirement-5 contract being exercised against
a real artifact rather than a fixture.

    CUDA_VISIBLE_DEVICES=99 python -m pytest test/registered/translator/test_tts_backend.py -v
"""

import asyncio
import json
import math
import socket
import threading
import unittest
from pathlib import Path

import numpy as np
import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, StreamingResponse

from sglang.srt.translator.backends import AudioChunk, BackendError
from sglang.srt.translator.tts_backends import (
    OpenAiSpeechTts,
    TtsHttpConfig,
    languages_from_qwen3_tts_config,
)

MODEL_DIR = Path("/spinning/llm_stuff/translator-models/qwen3-tts-0.6b-base")
RATE = 24000


def free_port():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


class StubServer:
    """A minimal stand-in for vLLM-Omni's audio surface.

    Records what it was asked for, so the tests can assert on the REQUEST the
    adapter built (cross-lingual flag, language, voice) and not merely on the
    audio that came back.
    """

    def __init__(self, fail_speech=False, chunk_bytes=1023):
        self.voices = {}
        self.speech_requests = []
        self.voice_uploads = 0
        self.fail_speech = fail_speech
        # An ODD chunk size on purpose: PCM16 is two bytes per sample, so an
        # odd boundary splits a sample and exercises the adapter's carry.
        self.chunk_bytes = chunk_bytes
        self.app = FastAPI()

        @self.app.post("/v1/audio/voices")
        async def register(request: Request):
            form = await request.form()
            name = str(form.get("name", "unnamed"))
            self.voice_uploads += 1
            self.voices[name] = True
            return JSONResponse({"name": name, "status": "ready"})

        @self.app.post("/v1/audio/speech")
        async def speech(request: Request):
            body = await request.json()
            self.speech_requests.append(body)
            if self.fail_speech:
                return JSONResponse({"error": "model exploded"}, status_code=500)
            seconds = max(0.2, len(body.get("input", "")) * 0.01)
            n = int(seconds * RATE)
            t = np.arange(n, dtype=np.float32) / RATE
            wave = (0.3 * np.sin(2.0 * math.pi * 220.0 * t)).astype(np.float32)
            payload = (np.clip(wave, -1, 1) * 32767).astype("<i2").tobytes()

            def emit():
                for start in range(0, len(payload), self.chunk_bytes):
                    yield payload[start : start + self.chunk_bytes]

            return StreamingResponse(emit(), media_type="application/octet-stream")

    def start(self):
        self.port = free_port()
        config = uvicorn.Config(
            self.app, host="127.0.0.1", port=self.port, log_level="error"
        )
        self.server = uvicorn.Server(config)
        self.thread = threading.Thread(target=self.server.run, daemon=True)
        self.thread.start()
        # Bounded wait for readiness -- no unbounded spin.
        for _ in range(200):
            if getattr(self.server, "started", False):
                return
            threading.Event().wait(0.05)
        raise RuntimeError("stub server did not start")

    def stop(self):
        self.server.should_exit = True
        self.thread.join(timeout=10)

    @property
    def base_url(self):
        return f"http://127.0.0.1:{self.port}/v1"


def reference(seconds=5.0, frequency=140.0, rate=RATE):
    t = np.arange(int(seconds * rate), dtype=np.float32) / rate
    return AudioChunk((0.3 * np.sin(2.0 * math.pi * frequency * t)).astype(np.float32), rate)


class TestLanguageContract(unittest.TestCase):
    @unittest.skipUnless(MODEL_DIR.exists(), "checkpoint not downloaded")
    def test_the_language_set_comes_from_the_real_checkpoint(self):
        languages = languages_from_qwen3_tts_config(MODEL_DIR)
        # The requirement-5 contract, exercised against the artifact itself.
        self.assertIn("de", languages)
        self.assertIn("es", languages)
        self.assertEqual(len(languages), 10, languages)
        self.assertEqual(languages, tuple(sorted(languages)))

    def test_a_missing_config_is_refused_rather_than_guessed(self):
        with self.assertRaises(BackendError) as ctx:
            languages_from_qwen3_tts_config(Path("/nonexistent/model"))
        self.assertIn("config.json", str(ctx.exception))

    def test_a_config_without_the_table_is_refused(self):
        import tempfile

        with tempfile.TemporaryDirectory() as tmp:
            (Path(tmp) / "config.json").write_text('{"talker_config": {}}')
            with self.assertRaises(BackendError) as ctx:
                languages_from_qwen3_tts_config(Path(tmp))
            self.assertIn("codec_language_id", str(ctx.exception))

    def test_construction_without_any_language_source_is_refused(self):
        # Guessing would put unspeakable languages into the advertised set.
        with self.assertRaises(BackendError) as ctx:
            OpenAiSpeechTts(TtsHttpConfig(model_dir=None, languages=None))
        self.assertIn("language set", str(ctx.exception))

    def test_an_unknown_language_name_passes_through_rather_than_vanishing(self):
        import tempfile

        with tempfile.TemporaryDirectory() as tmp:
            (Path(tmp) / "config.json").write_text(
                json.dumps({"talker_config": {"codec_language_id":
                                              {"german": 1, "klingon": 2}}})
            )
            languages = languages_from_qwen3_tts_config(Path(tmp))
            self.assertEqual(languages, ("de", "klingon"))


class TestAdapterOverHttp(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.stub = StubServer()
        self.stub.start()
        self.tts = OpenAiSpeechTts(
            TtsHttpConfig(
                base_url=self.stub.base_url,
                languages=("de", "es", "en"),
                sample_rate=RATE,
                min_reference_seconds=3.0,
            )
        )

    async def asyncTearDown(self):
        await self.tts.aclose()
        self.stub.stop()

    async def test_a_full_synthesis_streams_audio_back(self):
        chunks = []
        async for piece in self.tts.synthesize(
            "hola que tal", "es", reference(), "guten tag"
        ):
            chunks.append(piece)
        self.assertTrue(chunks)
        self.assertTrue(all(c.sample_rate == RATE for c in chunks))
        merged = np.concatenate([c.samples for c in chunks])
        self.assertGreater(len(merged), 0)
        self.assertLessEqual(float(np.abs(merged).max()), 1.0)

    async def test_pcm_reassembly_survives_odd_chunk_boundaries(self):
        # The stub deliberately emits 1023-byte chunks, splitting samples.
        # A dropped carry byte would desynchronise the stream into noise --
        # slowly, so it would pass a smoke test and fail in Spain.
        expected_seconds = max(0.2, len("hola") * 0.01)
        total = 0
        async for piece in self.tts.synthesize("hola", "es", reference()):
            total += len(piece.samples)
        self.assertEqual(total, int(expected_seconds * RATE))

    async def test_the_reference_is_registered_once_and_cached(self):
        clip = reference()
        for _ in range(3):
            async for _piece in self.tts.synthesize("hola", "es", clip):
                pass
        self.assertEqual(
            self.stub.voice_uploads, 1,
            "an unchanged reference was re-uploaded, adding a round trip to "
            "the critical path of every turn",
        )
        voices = {r["voice"] for r in self.stub.speech_requests}
        self.assertEqual(len(voices), 1)

    async def test_a_different_reference_registers_a_new_voice(self):
        async for _p in self.tts.synthesize("a", "es", reference(frequency=140.0)):
            pass
        async for _p in self.tts.synthesize("b", "es", reference(frequency=240.0)):
            pass
        self.assertEqual(self.stub.voice_uploads, 2)

    async def test_concurrent_first_turns_do_not_double_register(self):
        clip = reference()

        async def one():
            async for _p in self.tts.synthesize("hola", "es", clip):
                pass

        await asyncio.gather(*(one() for _ in range(4)))
        self.assertEqual(self.stub.voice_uploads, 1)

    async def test_a_preset_voice_id_skips_registration(self):
        async for _p in self.tts.synthesize(
            "hola", "es", reference(), None, "preset-man-1"
        ):
            pass
        self.assertEqual(self.stub.voice_uploads, 0)
        self.assertEqual(self.stub.speech_requests[-1]["voice"], "preset-man-1")

    async def test_the_cross_lingual_flag_and_language_reach_the_server(self):
        async for _p in self.tts.synthesize("hola", "es", reference(), "guten tag"):
            pass
        body = self.stub.speech_requests[-1]
        self.assertEqual(body["language"], "es")
        self.assertTrue(body["x_vector_only_mode"])
        # In cross-lingual mode the reference TRANSCRIPT must not be sent:
        # keeping it out of the LM context is the mechanism itself.
        self.assertNotIn("reference_text", body)

    async def test_in_context_mode_does_send_the_reference_transcript(self):
        tts = OpenAiSpeechTts(
            TtsHttpConfig(
                base_url=self.stub.base_url,
                languages=("de", "es"),
                x_vector_only_mode=False,
            )
        )
        try:
            async for _p in tts.synthesize("hola", "es", reference(), "guten tag"):
                pass
        finally:
            await tts.aclose()
        body = self.stub.speech_requests[-1]
        self.assertFalse(body["x_vector_only_mode"])
        self.assertEqual(body["reference_text"], "guten tag")

    async def test_an_unspeakable_language_is_refused_before_the_request(self):
        with self.assertRaises(BackendError) as ctx:
            async for _p in self.tts.synthesize("bonjour", "fr", reference()):
                pass
        self.assertIn("cannot speak", str(ctx.exception))
        self.assertEqual(self.stub.speech_requests, [])

    async def test_a_short_reference_is_refused_before_the_request(self):
        with self.assertRaises(BackendError) as ctx:
            async for _p in self.tts.synthesize(
                "hola", "es", reference(seconds=1.0)
            ):
                pass
        self.assertIn("reference is", str(ctx.exception))
        self.assertEqual(self.stub.voice_uploads, 0)

    async def test_empty_text_produces_nothing_and_costs_nothing(self):
        chunks = [c async for c in self.tts.synthesize("   ", "es", reference())]
        self.assertEqual(chunks, [])
        self.assertEqual(self.stub.speech_requests, [])

    async def test_a_server_error_surfaces_as_a_named_backend_error(self):
        stub = StubServer(fail_speech=True)
        stub.start()
        tts = OpenAiSpeechTts(
            TtsHttpConfig(base_url=stub.base_url, languages=("es",))
        )
        try:
            with self.assertRaises(BackendError) as ctx:
                async for _p in tts.synthesize("hola", "es", reference()):
                    pass
            self.assertEqual(ctx.exception.stage, "tts")
        finally:
            await tts.aclose()
            stub.stop()


class TestAdapterSatisfiesTheProtocol(unittest.TestCase):
    def test_it_is_shaped_like_the_backend_the_session_expects(self):
        tts = OpenAiSpeechTts(TtsHttpConfig(languages=("de", "es")))
        for attribute in ("name", "sample_rate", "min_reference_seconds",
                          "supported_languages", "synthesize"):
            self.assertTrue(hasattr(tts, attribute), attribute)
        self.assertEqual(set(tts.supported_languages()), {"de", "es"})


if __name__ == "__main__":
    unittest.main()
