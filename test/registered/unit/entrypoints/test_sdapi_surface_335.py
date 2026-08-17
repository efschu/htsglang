"""#335 — the A1111 ``/sdapi/v1`` surface: conformance, composition, refusals.

Implemented against AUTOMATIC1111 ``stable-diffusion-webui``'s API models
(``modules/api/models.py``: ``StableDiffusionTxt2ImgProcessingAPI`` /
``TextToImageResponse``) as stock clients send and expect them.

WHAT THESE PIN, and why each matters:

* **composition, not a parallel path.** The Ollama front's documented mistake
  (``ANALYSE_335_compat_surfaces.md`` §2) is a compat surface that drives the
  engine itself; it is why ``format`` could not simply be wired there. A source
  pin forbids the vocabulary that would signal a relapse here.
* **refuse, never approximate.** A1111 carries diffusion controls the OpenAI
  images protocol has no field for. Dropping them renders an image the caller
  did not ask for, with nothing saying so -- the #710 tool-arg-loss family, and
  the exact defect this task's own Ollama fix was for.
* **protocol conformance of the REFUSALS too.** A stock client parses errors as
  well as successes, so the error envelope is pinned like the success one.
"""

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

import asyncio
import inspect
import json
import unittest

from fastapi.responses import ORJSONResponse

from sglang.srt.entrypoints.sdapi.protocol import Txt2ImgRequest
from sglang.srt.entrypoints.sdapi.serving import SUPPORTED_SIZES, SdapiServing
from sglang.test.test_utils import CustomTestCase


class _FakeImages:
    """The OpenAI images front, mocked at its real call shape."""

    def __init__(self, *, status=200, payload=None):
        self.status = status
        self.payload = (
            payload
            if payload is not None
            else {"created": 1, "data": [{"b64_json": "AAAA"}]}
        )
        self.called_with = None

    async def generations(self, body, raw_request):
        self.called_with = body
        return ORJSONResponse(status_code=self.status, content=self.payload)


def _run(coro):
    return asyncio.run(coro)


def _serving(images=None):
    return SdapiServing(images or _FakeImages(), model_name="test-model")


class TestTheHappyPathIsProtocolShaped(CustomTestCase):
    def test_a_default_request_reaches_the_images_front(self):
        images = _FakeImages()
        _run(_serving(images).handle_txt2img(Txt2ImgRequest(prompt="a cat"), None))
        self.assertEqual(images.called_with["prompt"], "a cat")

    def test_the_response_carries_the_three_protocol_keys(self):
        r = _run(_serving().handle_txt2img(Txt2ImgRequest(prompt="a cat"), None))
        body = json.loads(r.body)
        self.assertEqual(set(body), {"images", "parameters", "info"})

    def test_images_are_base64_strings_not_objects(self):
        r = _run(_serving().handle_txt2img(Txt2ImgRequest(prompt="a cat"), None))
        self.assertEqual(json.loads(r.body)["images"], ["AAAA"])

    def test_info_is_a_json_encoded_string(self):
        """A1111's ``info`` is a STRING that clients json.loads. Emitting an
        object breaks them far from the cause."""
        body = json.loads(
            _run(_serving().handle_txt2img(Txt2ImgRequest(prompt="a cat"), None)).body
        )
        self.assertIsInstance(body["info"], str)
        self.assertEqual(json.loads(body["info"])["prompt"], "a cat")

    def test_batch_size_times_n_iter_becomes_n(self):
        images = _FakeImages()
        _run(
            _serving(images).handle_txt2img(
                Txt2ImgRequest(prompt="x", batch_size=2, n_iter=3), None
            )
        )
        self.assertEqual(images.called_with["n"], 6)

    def test_width_and_height_become_size(self):
        images = _FakeImages()
        _run(
            _serving(images).handle_txt2img(
                Txt2ImgRequest(prompt="x", width=1024, height=1024), None
            )
        )
        self.assertEqual(images.called_with["size"], "1024x1024")

    def test_base64_is_requested_because_a1111_returns_base64(self):
        images = _FakeImages()
        _run(_serving(images).handle_txt2img(Txt2ImgRequest(prompt="x"), None))
        self.assertEqual(images.called_with["response_format"], "b64_json")


class TestRefuseNeverApproximate(CustomTestCase):
    """Every A1111 control with no OpenAI equivalent, one subtest each."""

    def test_each_unmappable_field_is_refused_by_name(self):
        cases = {
            "steps": dict(steps=20),
            "cfg_scale": dict(cfg_scale=12.0),
            "sampler_name": dict(sampler_name="DPM++ 2M"),
            "sampler_index": dict(sampler_index="Euler a"),
            "seed": dict(seed=1234),
            "subseed": dict(subseed=99),
            "denoising_strength": dict(denoising_strength=0.3),
            "restore_faces": dict(restore_faces=True),
            "tiling": dict(tiling=True),
        }
        for field, kwargs in cases.items():
            with self.subTest(field=field):
                r = _run(
                    _serving().handle_txt2img(
                        Txt2ImgRequest(prompt="x", **kwargs), None
                    )
                )
                self.assertEqual(r.status_code, 400)
                self.assertIn(field, json.dumps(json.loads(r.body)))

    def test_a_refused_request_never_reaches_the_backend(self):
        """The point of refusing: nothing is rendered from a request we could
        not honour."""
        images = _FakeImages()
        _run(
            _serving(images).handle_txt2img(Txt2ImgRequest(prompt="x", steps=20), None)
        )
        self.assertIsNone(images.called_with)

    def test_negative_prompt_is_refused_and_says_why(self):
        """Folding it into the prompt would send it as a POSITIVE -- wrong in
        the direction hardest to see in the output."""
        r = _run(
            _serving().handle_txt2img(
                Txt2ImgRequest(prompt="a cat", negative_prompt="blurry"), None
            )
        )
        self.assertEqual(r.status_code, 400)
        self.assertIn("POSITIVE", json.dumps(json.loads(r.body)))

    def test_an_unexpressible_canvas_is_refused_not_rounded(self):
        r = _run(
            _serving().handle_txt2img(
                Txt2ImgRequest(prompt="x", width=768, height=768), None
            )
        )
        self.assertEqual(r.status_code, 400)
        self.assertIn("768x768", json.dumps(json.loads(r.body)))

    def test_every_supported_size_is_accepted(self):
        """The falsifier for the size gate: it must be able to pass."""
        for size in SUPPORTED_SIZES:
            w, h = (int(x) for x in size.split("x"))
            with self.subTest(size=size):
                images = _FakeImages()
                r = _run(
                    _serving(images).handle_txt2img(
                        Txt2ImgRequest(prompt="x", width=w, height=h), None
                    )
                )
                self.assertEqual(r.status_code, 200)

    def test_the_refusal_envelope_is_a1111_shaped(self):
        """A stock client parses errors too."""
        r = _run(_serving().handle_txt2img(Txt2ImgRequest(prompt="x", steps=1), None))
        self.assertEqual(set(json.loads(r.body)), {"error", "detail", "errors"})

    def test_the_refusal_routes_to_the_working_path(self):
        r = _run(_serving().handle_txt2img(Txt2ImgRequest(prompt="x", steps=1), None))
        self.assertIn("/v1/images/generations", json.dumps(json.loads(r.body)))


class TestInertIsDeclaredNotSilent(CustomTestCase):
    """The Ollama ``keep_alive`` precedent: a field with no effect is stated,
    not left looking honoured by accident."""

    def test_the_inert_fields_are_enumerated_with_reasons(self):
        inert = _serving().inert_fields()
        self.assertIn("save_images", inert)
        for field, why in inert.items():
            with self.subTest(field=field):
                self.assertGreater(len(why), 20)

    def test_an_inert_field_does_not_refuse_the_request(self):
        r = _run(
            _serving().handle_txt2img(
                Txt2ImgRequest(prompt="x", save_images=True), None
            )
        )
        self.assertEqual(r.status_code, 200)


class TestRefusedEndpoints(CustomTestCase):
    def test_img2img_is_501_and_explains_the_difference(self):
        r = _serving().refuse_img2img()
        self.assertEqual(r.status_code, 501)
        self.assertIn("inpaint", json.dumps(json.loads(r.body)).lower())

    def test_progress_returns_honest_zeros_in_the_protocol_shape(self):
        """404ing an endpoint stock clients poll on a timer would look like a
        broken server; inventing a percentage would be a lie."""
        body = json.loads(_serving().progress().body)
        self.assertEqual(body["progress"], 0.0)
        self.assertIn("state", body)
        self.assertIn("writes once", body["textinfo"])

    def test_options_and_models_name_the_served_model(self):
        self.assertEqual(
            json.loads(_serving().options().body)["sd_model_checkpoint"],
            "test-model",
        )
        self.assertEqual(
            json.loads(_serving().sd_models().body)[0]["model_name"], "test-model"
        )


class TestABackendRefusalIsPassedThroughUnchanged(CustomTestCase):
    """The images front's own refusal already names the registry state that
    caused it. Re-dressing it in an A1111 envelope would hide which layer
    refused."""

    def test_a_non_200_is_returned_verbatim(self):
        images = _FakeImages(status=503, payload={"error": "no diffusion lane"})
        r = _run(_serving(images).handle_txt2img(Txt2ImgRequest(prompt="x"), None))
        self.assertEqual(r.status_code, 503)
        self.assertEqual(json.loads(r.body)["error"], "no diffusion lane")


class TestItComposesAndCannotServe(CustomTestCase):
    """The structural pin. Kobold's module carries the same one, and it is what
    keeps a compat surface from becoming a second serving path."""

    def test_the_module_holds_no_engine_vocabulary(self):
        from sglang.srt.entrypoints.sdapi import serving

        src = inspect.getsource(serving)
        for forbidden in (
            "tokenizer_manager",
            "sampling_params",
            "apply_chat_template",
            "httpx",
            "aiohttp",
        ):
            with self.subTest(token=forbidden):
                self.assertNotIn(forbidden, src)

    def test_it_calls_the_images_front_and_not_a_lane_url(self):
        from sglang.srt.entrypoints.sdapi import serving

        src = inspect.getsource(serving)
        self.assertIn("self._images.generations(", src)
        self.assertNotIn("image_lane_url", src)


if __name__ == "__main__":
    unittest.main()
