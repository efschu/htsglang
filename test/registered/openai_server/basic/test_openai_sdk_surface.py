# Copyright 2025 SGLang Team
# Licensed under the Apache License, Version 2.0
"""#335-M0: the OpenAI surface, driven by the official ``openai`` SDK.

A surface no real client can drive is not compatible, however closely its
handwritten JSON matches the documentation. So these tests use the same
``openai`` package a user would install, pointed at a live local server
(:mod:`openai_sdk_harness`) whose only mocked part is the engine.

What that buys over hand-rolled requests: the SDK builds the request bodies,
parses the SSE stream, validates responses against its own typed models, and
maps status codes onto typed exceptions. Every one of those is a place where
"looks right in curl" and "works in a client" come apart.

Requires no GPU. ``CUDA_VISIBLE_DEVICES=99`` is the intended way to run it.
"""

from __future__ import annotations

import base64
import json
import unittest

import numpy as np
import openai
import requests

# Imported at module scope on purpose: ``from __future__ import annotations``
# turns the fake lane's parameter annotations into strings, and FastAPI
# resolves them against module globals. A function-local import makes it treat
# ``Request`` as an unknown type and demand it as a query parameter.
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from openai_sdk_harness import CANNED_TEXT, MODEL_NAME, TOKENIZER_NAME, live_server

from sglang.srt.entrypoints.openai.registry_view import RegisteredEngine, RegistryView
from sglang.test.ci.ci_register import register_cpu_ci

# CPU: the engine is mocked, so this needs no card -- which is the point.
register_cpu_ci(est_time=30, suite="base-a-test-cpu")

_HOT_DIFFUSION = RegistryView(
    reachable=True,
    url="http://127.0.0.1:8500",
    engines=(
        RegisteredEngine(
            engine_id="qwen-image",
            klass=2,
            state="COLD",
            cards=("GPU-aaaa",),
            reserved_bytes=12 * (1 << 30),
            health="ok",
        ),
        RegisteredEngine(
            engine_id="qwen3-27b",
            klass=1,
            state="HOT",
            cards=("GPU-bbbb",),
            reserved_bytes=20 * (1 << 30),
            health="ok",
            promotion_cost_ms=41200.0,
        ),
    ),
)


class OpenAISDKSurfaceTest(unittest.TestCase):
    """One server for the whole class: booting it costs ~5s, the tests ~0."""

    @classmethod
    def setUpClass(cls):
        from transformers import AutoTokenizer

        cls.tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_NAME)
        cls._server = live_server(tokenizer=cls.tokenizer, registry_view=_HOT_DIFFUSION)
        cls.base_url = cls._server.__enter__()
        cls.client = openai.OpenAI(base_url=cls.base_url + "/v1", api_key="unused")

    @classmethod
    def tearDownClass(cls):
        cls._server.__exit__(None, None, None)

    # ---------------------------------------------------------------- models

    def test_models_list_is_spec_shaped(self):
        page = self.client.models.list()
        models = list(page)
        ids = [m.id for m in models]
        self.assertIn(MODEL_NAME, ids)
        for model in models:
            self.assertEqual(model.object, "model")
            self.assertIsInstance(model.created, int)
            self.assertTrue(model.owned_by)

    def test_models_list_includes_registered_engines_with_residency(self):
        raw = requests.get(self.base_url + "/v1/models", timeout=10).json()
        by_id = {card["id"]: card for card in raw["data"]}
        self.assertIn("qwen-image", by_id)
        self.assertIn("qwen3-27b", by_id)

        # Extensions ride namespaced, never at the top level.
        cold = by_id["qwen-image"]["x-htsglang"]
        self.assertEqual(cold["residency"], "COLD")
        self.assertFalse(cold["gpu_resident"])
        self.assertEqual(cold["engine_class"], 2)
        self.assertEqual(cold["reserved_mib"], 12 * 1024)

        hot = by_id["qwen3-27b"]["x-htsglang"]
        self.assertEqual(hot["residency"], "HOT")
        self.assertTrue(hot["gpu_resident"])
        self.assertEqual(hot["measured_promotion_ms"], 41200.0)

        # The locally served model keeps its own card, not the registry's.
        self.assertEqual(by_id[MODEL_NAME]["x-htsglang"]["served_by"], "local")

    def test_retrieve_model_and_typed_not_found(self):
        model = self.client.models.retrieve(MODEL_NAME)
        self.assertEqual(model.id, MODEL_NAME)

        with self.assertRaises(openai.NotFoundError) as ctx:
            self.client.models.retrieve("no-such-model")
        self.assertEqual(ctx.exception.code, "model_not_found")
        self.assertEqual(ctx.exception.param, "model")

    # ------------------------------------------------------------------ chat

    def test_chat_non_streaming(self):
        response = self.client.chat.completions.create(
            model=MODEL_NAME,
            messages=[{"role": "user", "content": "hi"}],
        )
        self.assertEqual(response.object, "chat.completion")
        self.assertEqual(response.choices[0].message.content, CANNED_TEXT)
        self.assertEqual(response.choices[0].finish_reason, "stop")
        self.assertEqual(response.usage.prompt_tokens, 7)
        self.assertEqual(response.usage.total_tokens, 10)

    def test_chat_streaming_deltas_and_usage_chunk(self):
        stream = self.client.chat.completions.create(
            model=MODEL_NAME,
            messages=[{"role": "user", "content": "hi"}],
            stream=True,
            stream_options={"include_usage": True},
        )
        chunks = list(stream)
        text = "".join(c.choices[0].delta.content or "" for c in chunks if c.choices)
        self.assertEqual(text, CANNED_TEXT)
        self.assertTrue(all(c.object == "chat.completion.chunk" for c in chunks))

        # The usage chunk is last and carries an empty choices array, per spec.
        usage_chunk = chunks[-1]
        self.assertEqual(usage_chunk.choices, [])
        self.assertEqual(usage_chunk.usage.prompt_tokens, 7)

        finish_reasons = [c.choices[0].finish_reason for c in chunks if c.choices]
        self.assertEqual(finish_reasons[-1], "stop")

    def test_completions_streaming_and_non_streaming(self):
        response = self.client.completions.create(model=MODEL_NAME, prompt="hi")
        self.assertEqual(response.object, "text_completion")
        self.assertEqual(response.choices[0].text, CANNED_TEXT)

        chunks = list(
            self.client.completions.create(model=MODEL_NAME, prompt="hi", stream=True)
        )
        text = "".join(c.choices[0].text for c in chunks if c.choices)
        self.assertEqual(text, CANNED_TEXT)

    # ------------------------------------------------------------ embeddings

    def test_embeddings_default_path_round_trips_through_base64(self):
        """The SDK asks for base64 unless told otherwise, then decodes it.

        ``openai/resources/embeddings.py`` sets ``encoding_format="base64"``
        whenever the caller did not, and decodes with
        ``np.frombuffer(..., dtype="float32")``. Getting the exact floats back
        here therefore proves the server's base64 encoding is byte-compatible
        with the official decoder -- this is the default path for every Python
        client in existence, not an opt-in one.
        """
        response = self.client.embeddings.create(model=MODEL_NAME, input=["a", "b"])
        self.assertEqual(response.object, "list")
        self.assertEqual([d.index for d in response.data], [0, 1])
        self.assertEqual(response.data[0].embedding, [0.5, 0.5, 0.5, 0.5])
        self.assertEqual(response.data[1].embedding, [1.0, 1.0, 1.0, 1.0])
        # Real counts from the engine, not a placeholder.
        self.assertEqual(response.usage.prompt_tokens, 10)
        self.assertEqual(response.usage.total_tokens, 10)

    def test_embeddings_base64_decodes_to_the_same_floats(self):
        # The SDK decodes base64 back into floats transparently, so compare
        # against the raw body to prove the encoding really happened.
        raw = requests.post(
            self.base_url + "/v1/embeddings",
            json={"model": MODEL_NAME, "input": "a", "encoding_format": "base64"},
            timeout=10,
        ).json()
        encoded = raw["data"][0]["embedding"]
        self.assertIsInstance(encoded, str)
        decoded = np.frombuffer(base64.b64decode(encoded), dtype="<f4")
        np.testing.assert_allclose(decoded, [0.5, 0.5, 0.5, 0.5])

        # An explicit encoding_format is left untouched by the SDK's parser,
        # so the typed model carries the string through unchanged.
        via_sdk = self.client.embeddings.create(
            model=MODEL_NAME, input="a", encoding_format="base64"
        )
        self.assertEqual(via_sdk.data[0].embedding, encoded)

    def test_embeddings_explicit_float_returns_a_float_list(self):
        raw = requests.post(
            self.base_url + "/v1/embeddings",
            json={"model": MODEL_NAME, "input": "a", "encoding_format": "float"},
            timeout=10,
        ).json()
        self.assertEqual(raw["data"][0]["embedding"], [0.5, 0.5, 0.5, 0.5])
        self.assertEqual(raw["object"], "list")
        self.assertEqual(raw["data"][0]["object"], "embedding")

    def test_embeddings_reject_unknown_encoding_format(self):
        response = requests.post(
            self.base_url + "/v1/embeddings",
            json={"model": MODEL_NAME, "input": "a", "encoding_format": "float16"},
            timeout=10,
        )
        self.assertEqual(response.status_code, 400)
        self.assertIn("encoding_format", response.json()["error"]["message"])

    # ---------------------------------------------------------- error bodies

    def test_error_body_is_the_openai_envelope(self):
        response = requests.post(
            self.base_url + "/v1/chat/completions",
            json={"model": MODEL_NAME},  # no messages
            timeout=10,
        )
        self.assertEqual(response.status_code, 400)
        body = response.json()
        self.assertIn("error", body)
        error = body["error"]
        self.assertEqual(sorted(error), ["code", "message", "param", "type"])
        self.assertEqual(error["type"], "invalid_request_error")

    def test_sdk_raises_bad_request_for_invalid_body(self):
        with self.assertRaises(openai.BadRequestError):
            self.client.chat.completions.create(model=MODEL_NAME, messages=[])

    def test_an_unhandled_server_error_is_still_json(self):
        """Starlette's default 500 is text/plain, which no SDK can parse.

        This harness leaves ``openai_serving_responses`` unset, which is the
        real failure mode the audit found: the route reaches for a missing
        ``app.state`` attribute. The client must still get an error object.
        """
        response = requests.post(
            self.base_url + "/v1/responses",
            json={"model": MODEL_NAME, "input": "hi"},
            timeout=20,
        )
        self.assertEqual(response.status_code, 500)
        self.assertTrue(response.headers["content-type"].startswith("application/json"))
        error = response.json()["error"]
        self.assertEqual(error["type"], "api_error")
        # The detail is scrubbed: a 5xx message can carry a stack frame.
        self.assertEqual(error["message"], "Internal server error")

    def test_streaming_error_uses_the_same_envelope(self):
        # An empty input on the streaming path errors before the first chunk;
        # the body must still be an error envelope, not a bare string.
        response = requests.post(
            self.base_url + "/v1/completions",
            json={"model": MODEL_NAME, "prompt": "", "stream": True},
            timeout=10,
        )
        payload = response.text
        if response.headers.get("content-type", "").startswith("text/event-stream"):
            first = payload.split("data: ", 1)[1].split("\n", 1)[0]
            body = json.loads(first)
        else:
            body = response.json()
        self.assertIn("error", body)
        self.assertIn("message", body["error"])

    # ------------------------------------------------------ images / rejects

    def test_image_generation_rejects_with_a_typed_not_found(self):
        with self.assertRaises(openai.NotFoundError) as ctx:
            self.client.images.generate(model="qwen-image", prompt="a cat")
        exc = ctx.exception
        self.assertEqual(exc.code, "model_not_found")
        extension = exc.body["x-htsglang"]
        self.assertEqual(extension["capability"], "image_generation")
        # The rejection names the registered engine and what is missing.
        self.assertEqual(extension["registered_diffusion_engines"], ["qwen-image"])
        self.assertEqual(extension["gpu_resident_diffusion_engines"], [])
        self.assertTrue(extension["what_would_make_it_work"])

    def test_image_variations_report_the_endpoint_as_unimplemented(self):
        response = requests.post(
            self.base_url + "/v1/images/variations",
            files={"image": ("x.png", b"\x89PNG", "image/png")},
            timeout=10,
        )
        # No lane configured, so the honest answer is still "no lane".
        self.assertEqual(response.status_code, 404)
        self.assertEqual(response.json()["error"]["code"], "model_not_found")

    def test_speech_rejects_with_a_typed_not_found(self):
        with self.assertRaises(openai.NotFoundError) as ctx:
            self.client.audio.speech.create(model="tts-1", voice="alloy", input="hello")
        extension = ctx.exception.body["x-htsglang"]
        self.assertEqual(extension["capability"], "text_to_speech")
        self.assertTrue(
            any(
                "transcriptions" in remedy
                for remedy in extension["what_would_make_it_work"]
            )
        )

    def test_speech_rejects_an_unsupported_response_format_as_400(self):
        with self.assertRaises(openai.BadRequestError) as ctx:
            self.client.audio.speech.create(
                model="tts-1",
                voice="alloy",
                input="hello",
                response_format="ogg",  # not in OpenAI's set
            )
        self.assertEqual(ctx.exception.param, "response_format")

    def test_transcription_refuses_a_non_asr_model(self):
        # The served model is a text LLM; the adapter registry would silently
        # fall back to Whisper and answer with fluent nonsense.
        response = requests.post(
            self.base_url + "/v1/audio/transcriptions",
            files={"file": ("a.wav", b"RIFF", "audio/wav")},
            data={"model": MODEL_NAME},
            timeout=10,
        )
        self.assertEqual(response.status_code, 400)
        message = response.json()["error"]["message"]
        self.assertIn("Qwen3ForCausalLM", message)
        self.assertIn("not a speech-recognition model", message)

    # --------------------------------------------------------- ollama surface

    def test_ollama_tags_and_chat_ride_the_same_lane(self):
        """The Ollama emulation is in-tree already; this pins it to the lane.

        Same server, same engine, a second protocol on top -- which is the
        whole point of adapters being thin. Ollama clients (the CLI, the
        ``ollama`` Python package, anything pointed at ``OLLAMA_HOST``) can use
        this server unchanged.
        """
        tags = requests.get(self.base_url + "/api/tags", timeout=10).json()
        self.assertEqual([m["name"] for m in tags["models"]], [MODEL_NAME])

        chat = requests.post(
            self.base_url + "/api/chat",
            json={
                "model": MODEL_NAME,
                "messages": [{"role": "user", "content": "hi"}],
                "stream": False,
            },
            timeout=30,
        ).json()
        self.assertEqual(chat["message"]["content"], CANNED_TEXT)
        self.assertTrue(chat["done"])

    def test_ollama_generate_non_streaming(self):
        body = requests.post(
            self.base_url + "/api/generate",
            json={"model": MODEL_NAME, "prompt": "hi", "stream": False},
            timeout=30,
        ).json()
        self.assertEqual(body["response"], CANNED_TEXT)

    def test_unsupported_transcription_format_is_a_typed_error(self):
        response = requests.post(
            self.base_url + "/v1/audio/transcriptions",
            files={"file": ("a.wav", b"RIFF", "audio/wav")},
            data={"model": MODEL_NAME, "response_format": "srt"},
            timeout=10,
        )
        self.assertEqual(response.status_code, 400)
        self.assertEqual(response.json()["error"]["param"], "response_format")


class ImageLaneRoutingTest(unittest.TestCase):
    """With a lane configured, the adapter forwards instead of rejecting."""

    @classmethod
    def setUpClass(cls):
        from transformers import AutoTokenizer

        cls.tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_NAME)
        cls.lane = _FakeImageLane()
        cls.lane.start()
        cls._server = live_server(tokenizer=cls.tokenizer, image_lane_url=cls.lane.url)
        cls.base_url = cls._server.__enter__()
        cls.client = openai.OpenAI(base_url=cls.base_url + "/v1", api_key="unused")

    @classmethod
    def tearDownClass(cls):
        cls._server.__exit__(None, None, None)
        cls.lane.stop()

    def test_generations_are_forwarded_and_the_answer_relayed(self):
        response = self.client.images.generate(model="qwen-image", prompt="a cat", n=1)
        self.assertEqual(response.data[0].b64_json, "aGVsbG8=")
        self.assertEqual(self.lane.last_body["prompt"], "a cat")

    def test_a_lane_error_is_relayed_with_its_own_status(self):
        self.lane.fail_next = (
            429,
            {
                "error": {
                    "message": "queue full",
                    "type": "rate_limit_error",
                    "param": None,
                    "code": None,
                }
            },
        )
        # max_retries=0: the SDK retries 429 by default and would consume the
        # single queued failure, then pass on the retry.
        with self.assertRaises(openai.RateLimitError) as ctx:
            self.client.with_options(max_retries=0).images.generate(
                model="qwen-image", prompt="a cat"
            )
        self.assertEqual(ctx.exception.message.count("queue full"), 1)

    def test_an_unreachable_lane_is_a_503_not_a_500(self):
        import requests as _requests

        stopped_port = self.lane.port
        self.lane.stop()
        try:
            response = _requests.post(
                self.base_url + "/v1/images/generations",
                json={"model": "qwen-image", "prompt": "a cat"},
                timeout=30,
            )
            self.assertEqual(response.status_code, 503)
            self.assertEqual(response.json()["error"]["code"], "lane_unreachable")
        finally:
            self.lane.start(port=stopped_port)


class _FakeImageLane:
    """A stand-in for a ``multimodal_gen`` server on its own port."""

    def __init__(self):
        import threading

        self._thread = None
        self._server = None
        self.port = None
        self.last_body = None
        self.fail_next = None
        self._lock = threading.Lock()

    @property
    def url(self) -> str:
        return f"http://127.0.0.1:{self.port}"

    def start(self, port: int | None = None):
        import threading

        import uvicorn
        from openai_sdk_harness import free_port

        app = FastAPI()

        @app.post("/v1/images/generations")
        async def generations(request: Request):
            self.last_body = await request.json()
            with self._lock:
                failure = self.fail_next
                self.fail_next = None
            if failure is not None:
                return JSONResponse(failure[1], status_code=failure[0])
            return {
                "created": 0,
                "data": [{"b64_json": "aGVsbG8=", "url": None}],
            }

        self.port = port or free_port()
        config = uvicorn.Config(
            app, host="127.0.0.1", port=self.port, log_level="warning", access_log=False
        )
        self._server = uvicorn.Server(config)
        self._thread = threading.Thread(target=self._server.run, daemon=True)
        self._thread.start()
        import time

        deadline = time.time() + 30
        while not self._server.started and time.time() < deadline:
            time.sleep(0.05)

    def stop(self):
        if self._server is not None:
            self._server.should_exit = True
            self._thread.join(timeout=15)
            self._server = None


if __name__ == "__main__":
    unittest.main()
