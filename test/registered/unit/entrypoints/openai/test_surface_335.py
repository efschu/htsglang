# Copyright 2025 SGLang Team
# Licensed under the Apache License, Version 2.0
"""#335-M0 unit coverage: error envelope, registry view, lane rejections.

The SDK-driven tests in ``test/registered/openai_server/basic/`` prove the
surface end to end; these pin the pieces that have branches a live server run
would not reach -- an unreachable registry, a malformed engine entry, each
rejection variant.
"""

from __future__ import annotations

import base64
import json
import unittest
from unittest.mock import patch

import numpy as np

from sglang.srt.entrypoints.openai.errors import (
    EXTENSION_KEY,
    LaneUnavailable,
    error_message_of,
    error_type_for_status,
    openai_error_dict,
    parse_error_body,
)
from sglang.srt.entrypoints.openai.protocol import ErrorResponse, UsageInfo
from sglang.srt.entrypoints.openai.registry_view import (
    RegisteredEngine,
    RegistryView,
    _parse,
    fetch_registry_view,
    registry_base_url,
    reset_cache,
)
from sglang.srt.entrypoints.openai.serving_embedding import encode_embedding
from sglang.srt.entrypoints.openai.serving_images import _reject_no_lane
from sglang.srt.entrypoints.openai.transcription_adapters import matched_adapter_key
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=12, suite="base-a-test-cpu")


class ErrorEnvelopeTest(unittest.TestCase):
    def test_envelope_has_exactly_the_four_spec_fields(self):
        body = openai_error_dict("boom", err_type="api_error", code=500)
        self.assertEqual(list(body), ["error"])
        self.assertEqual(sorted(body["error"]), ["code", "message", "param", "type"])

    def test_extension_is_namespaced_inside_the_error_object(self):
        body = openai_error_dict("boom", extension={"capability": "x"})
        self.assertEqual(body["error"][EXTENSION_KEY], {"capability": "x"})

    def test_error_response_model_serializes_into_the_envelope(self):
        error = ErrorResponse(message="m", type="invalid_request_error", code=400)
        self.assertEqual(
            error.to_openai_envelope(),
            {
                "error": {
                    "message": "m",
                    "type": "invalid_request_error",
                    "param": None,
                    "code": 400,
                }
            },
        )

    def test_status_to_type_mapping_uses_the_spec_vocabulary(self):
        self.assertEqual(error_type_for_status(404), "not_found_error")
        self.assertEqual(error_type_for_status(429), "rate_limit_error")
        self.assertEqual(error_type_for_status(503), "api_error")
        self.assertEqual(error_type_for_status(418), "api_error")

    def test_parse_error_body_reads_both_shapes(self):
        nested = {"error": {"message": "m", "code": 400}}
        flat = {"object": "error", "message": "m", "code": 400}
        self.assertEqual(parse_error_body(nested)["message"], "m")
        self.assertEqual(parse_error_body(flat)["message"], "m")
        self.assertEqual(parse_error_body({"error": "plain"})["message"], "plain")
        self.assertEqual(error_message_of({}, "fallback"), "fallback")

    def test_usage_mirrors_reasoning_tokens_into_the_spec_object(self):
        with_reasoning = UsageInfo(
            prompt_tokens=1, completion_tokens=2, total_tokens=3, reasoning_tokens=2
        ).model_dump()
        self.assertEqual(
            with_reasoning["completion_tokens_details"], {"reasoning_tokens": 2}
        )
        plain = UsageInfo(
            prompt_tokens=1, completion_tokens=2, total_tokens=3
        ).model_dump()
        self.assertNotIn("completion_tokens_details", plain)


class EmbeddingEncodingTest(unittest.TestCase):
    def test_float_format_is_passed_through(self):
        self.assertEqual(encode_embedding([1.0, 2.0], "float"), [1.0, 2.0])

    def test_base64_is_little_endian_float32(self):
        encoded = encode_embedding([1.0, -2.5, 3.25], "base64")
        self.assertIsInstance(encoded, str)
        decoded = np.frombuffer(base64.b64decode(encoded), dtype="<f4")
        np.testing.assert_allclose(decoded, [1.0, -2.5, 3.25])
        # 3 float32 values, 12 bytes, never 24: a float64 buffer would decode
        # to garbage in every client that follows the spec.
        self.assertEqual(len(base64.b64decode(encoded)), 12)


class RegistryViewTest(unittest.TestCase):
    def setUp(self):
        reset_cache()

    def tearDown(self):
        reset_cache()

    def test_empty_env_disables_the_lookup(self):
        with patch.dict("os.environ", {"SGLANG_REGISTRY_URL": ""}):
            self.assertEqual(registry_base_url(), "")
            view = fetch_registry_view()
        self.assertFalse(view.reachable)
        self.assertEqual(view.engines, ())

    def test_bare_host_port_is_normalized(self):
        with patch.dict("os.environ", {"SGLANG_REGISTRY_URL": "10.0.0.5:9000"}):
            self.assertEqual(registry_base_url(), "http://10.0.0.5:9000")

    def test_unreachable_registry_is_an_empty_view_not_an_exception(self):
        with patch.dict("os.environ", {"SGLANG_REGISTRY_URL": "http://127.0.0.1:1"}):
            view = fetch_registry_view()
        self.assertFalse(view.reachable)
        self.assertIsNotNone(view.error)
        self.assertEqual(view.engines, ())

    def test_snapshot_parsing_and_residency(self):
        payload = {
            "engines": [
                {
                    "engine_id": "a",
                    "klass": 1,
                    "state": "HOT",
                    "cards": ["GPU-1"],
                    "reserved_bytes": 2 * (1 << 20),
                    "promotion_cost_ms": 1234.5,
                    "health": "ok",
                },
                {"engine_id": "b", "klass": 2, "state": "COLD"},
            ]
        }
        view = _parse(payload, "http://x")
        self.assertEqual(len(view.engines), 2)
        self.assertTrue(view.by_id("a").is_hot)
        self.assertFalse(view.by_id("b").is_gpu_resident)
        self.assertEqual([e.engine_id for e in view.of_class(2)], ["b"])

        extension = view.by_id("a").to_extension()
        self.assertEqual(extension["reserved_mib"], 2)
        self.assertEqual(extension["measured_promotion_ms"], 1234.5)
        # Never guessed: a transition that has not been observed reports nothing.
        self.assertNotIn("measured_promotion_ms", view.by_id("b").to_extension())

    def test_a_malformed_entry_does_not_blank_the_listing(self):
        view = _parse({"engines": [{"no_id": 1}, {"engine_id": "ok"}]}, "http://x")
        self.assertEqual([e.engine_id for e in view.engines], ["ok"])


class LaneRejectionTest(unittest.TestCase):
    def test_rejection_names_the_registered_engine(self):
        view = RegistryView(
            reachable=True,
            url="http://127.0.0.1:8500",
            engines=(RegisteredEngine(engine_id="flux", klass=2, state="COLD"),),
        )
        rejection = _reject_no_lane(view, "flux")
        response = rejection.to_response()
        self.assertEqual(response.status_code, 404)
        body = json.loads(bytes(response.body))
        error = body["error"]
        self.assertEqual(error["code"], "model_not_found")
        self.assertEqual(error["type"], "not_found_error")
        extension = error[EXTENSION_KEY]
        self.assertEqual(extension["registered_diffusion_engines"], ["flux"])
        self.assertIn("flux", error["message"])
        self.assertTrue(extension["what_would_make_it_work"])

    def test_rejection_without_any_registration_says_so(self):
        rejection = _reject_no_lane(RegistryView(), None)
        self.assertIn("no diffusion engine is", str(rejection))
        body = json.loads(bytes(rejection.to_response().body))
        self.assertEqual(
            body["error"][EXTENSION_KEY]["registered_diffusion_engines"], []
        )

    def test_status_codes_map_onto_distinct_sdk_exceptions(self):
        # 404 / 503 / 501 are three different client situations: wrong model,
        # lane down, endpoint absent. They must not collapse into one code.
        for status, code in (
            (404, "model_not_found"),
            (503, "lane_unreachable"),
            (501, "endpoint_not_implemented"),
        ):
            rejection = LaneUnavailable(
                "x", capability="c", status_code=status, code=code
            )
            response = rejection.to_response()
            self.assertEqual(response.status_code, status)
            self.assertEqual(json.loads(bytes(response.body))["error"]["code"], code)


class TranscriptionArchitectureTest(unittest.TestCase):
    def test_known_asr_architectures_match(self):
        self.assertEqual(
            matched_adapter_key(["WhisperForConditionalGeneration"]), "Whisper"
        )

    def test_a_text_llm_matches_nothing(self):
        # resolve_adapter() would still hand back the Whisper adapter here;
        # that fallback is exactly why the serving layer asks this question.
        self.assertIsNone(matched_adapter_key(["Qwen3ForCausalLM"]))
        self.assertIsNone(matched_adapter_key([]))


if __name__ == "__main__":
    unittest.main()
