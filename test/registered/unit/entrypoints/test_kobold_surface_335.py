"""#335: the KoboldCpp surface, composed over the OpenAI front.

THIS IS THE SHAPE THE OLLAMA FRONT SHOULD HAVE USED, and these pins hold that
claim rather than let it be a comment. The Ollama adapter drives the tokenizer
manager directly and applies its own chat template, which is why `format` was
never wired and why every future sampling feature would have to be wired
twice. This adapter translates into a ``CompletionRequest`` and hands it to
``openai_serving_completion``: no sampling, no template, no second place for a
feature to be forgotten.

The refusal discipline is the Ollama cut's (9b5a72f826), applied from the
start instead of retrofitted:

  * unsupported fields refuse BY NAME before generation, and the refusal
    ROUTES -- it names the endpoint that does support the thing;
  * inert fields are declared inert, so they do not look honoured by accident;
  * nothing is silently dropped.

Streaming is refused rather than approximated. Kobold streams via a polled
``/api/extra/generate/check`` protocol; emitting an OpenAI-shaped stream under
a Kobold URL would parse until it did not, mid-generation, with no error to
read. A half-faithful stream is worse than an honest 501.

Hermetic: the refusal surface and the translation are pure functions. No
server, no tokenizer, no CUDA, and the running server is not probed.
"""

import unittest

from sglang.srt.entrypoints.kobold.protocol import KoboldGenerateRequest
from sglang.srt.entrypoints.kobold.serving import KoboldServing


def _req(**over):
    body = {"prompt": "Once upon a time"}
    body.update(over)
    return KoboldGenerateRequest(**body)


def _serving():
    return KoboldServing(model_name="test-model")


class TestItComposesTheOpenAIPath(unittest.TestCase):
    """The structural claim, pinned."""

    def test_it_translates_into_a_completion_request(self):
        completion = _serving().to_completion_request(_req())
        from sglang.srt.entrypoints.openai.protocol import CompletionRequest

        self.assertIsInstance(completion, CompletionRequest)
        self.assertEqual(completion.prompt, "Once upon a time")

    def test_it_holds_no_tokenizer_manager_of_its_own(self):
        """A second serving path starts by acquiring its own engine handle."""
        serving = _serving()
        self.assertFalse(hasattr(serving, "tokenizer_manager"))

    def test_it_applies_no_template_and_no_sampling_itself(self):
        import inspect

        from sglang.srt.entrypoints.kobold import serving as m

        src = inspect.getsource(m)
        for forbidden in (
            "apply_chat_template",
            "tokenizer_manager.generate_request",
            "sampling_params",
        ):
            with self.subTest(token=forbidden):
                self.assertNotIn(
                    forbidden,
                    src,
                    "this adapter must translate onto the OpenAI path, never "
                    "grow a second serving/template/sampling path",
                )

    def test_the_handler_calls_openai_serving_completion(self):
        import inspect

        src = inspect.getsource(KoboldServing.handle_generate)
        self.assertIn("openai_serving_completion", src)


class TestTheHonouredFieldsMap(unittest.TestCase):
    def test_max_length_becomes_max_tokens(self):
        c = _serving().to_completion_request(_req(max_length=64))
        self.assertEqual(c.max_tokens, 64)

    def test_stop_sequence_becomes_stop(self):
        c = _serving().to_completion_request(_req(stop_sequence=["\n\n"]))
        self.assertEqual(c.stop, ["\n\n"])

    def test_sampling_knobs_pass_through(self):
        c = _serving().to_completion_request(
            _req(temperature=0.7, top_p=0.9, top_k=40)
        )
        self.assertEqual(c.temperature, 0.7)
        self.assertEqual(c.top_p, 0.9)
        self.assertEqual(c.top_k, 40)

    def test_an_omitted_max_length_gets_a_DECLARED_default(self):
        """CompletionRequest defaults max_tokens to 16 -- an OpenAI
        completions convention that would truncate a Kobold client's first
        answer to a fragment. Choosing a number here is an adapter
        responsibility; hiding that it was chosen is not, so the default is a
        named constant and this pin holds it visible."""
        c = _serving().to_completion_request(_req())
        self.assertEqual(c.max_tokens, KoboldServing.DEFAULT_MAX_TOKENS)
        self.assertGreater(KoboldServing.DEFAULT_MAX_TOKENS, 16)

    def test_an_explicit_max_length_always_wins_over_the_default(self):
        c = _serving().to_completion_request(_req(max_length=7))
        self.assertEqual(c.max_tokens, 7)


class TestAnHonourableRequestIsNotRefused(unittest.TestCase):
    def test_a_plain_request_passes(self):
        self.assertEqual(_serving().unsupported_reasons(_req()), [])

    def test_every_mapped_field_passes(self):
        request = _req(
            max_length=64,
            temperature=0.7,
            top_p=0.9,
            top_k=40,
            stop_sequence=["x"],
            n=1,
        )
        self.assertEqual(_serving().unsupported_reasons(request), [])

    def test_inert_fields_are_not_refused(self):
        request = _req(max_context_length=8192, quiet=True)
        self.assertEqual(_serving().unsupported_reasons(request), [])
        self.assertIn("max_context_length", KoboldServing.INERT_FIELDS)
        self.assertIn("quiet", KoboldServing.INERT_FIELDS)


class TestUnsupportedFieldsRefuseByNameAndRoute(unittest.TestCase):
    def test_rep_pen_is_refused_with_the_reason_it_cannot_be_mapped(self):
        """The worked example: multiplicative vs additive is not a constant
        away, so mapping it would sample differently than asked."""
        reasons = _serving().unsupported_reasons(_req(rep_pen=1.1))
        self.assertEqual(len(reasons), 1)
        self.assertIn("MULTIPLICATIVE", reasons[0])
        self.assertIn("/v1/completions", reasons[0])

    def test_grammar_routes_to_structured_output(self):
        reasons = _serving().unsupported_reasons(_req(grammar="root ::= x"))
        self.assertIn("response_format", reasons[0])

    def test_images_route_to_the_chat_endpoint(self):
        reasons = _serving().unsupported_reasons(_req(images=["deadbeef"]))
        self.assertIn("/v1/chat/completions", reasons[0])

    def test_sampler_order_is_refused_because_ignoring_it_changes_the_output(self):
        reasons = _serving().unsupported_reasons(_req(sampler_order=[6, 0, 1]))
        self.assertIn("silently", reasons[0])

    def test_several_unsupported_fields_are_all_named(self):
        reasons = _serving().unsupported_reasons(_req(mirostat=2, tfs=0.95))
        self.assertEqual(len(reasons), 2)
        joined = " ".join(reasons)
        self.assertIn("mirostat", joined)
        self.assertIn("tfs", joined)

    def test_every_unsupported_field_names_a_route_or_says_not_implemented(self):
        """A refusal that only blocks teaches nothing."""
        for name, why in KoboldServing.UNSUPPORTED_FIELDS.items():
            with self.subTest(field=name):
                self.assertTrue(
                    "/v1/" in why or "not implemented" in why or "rep_pen" in why,
                    f"{name} refuses without routing or explaining",
                )


class TestStreamingIsRefusedNotApproximated(unittest.TestCase):
    def test_the_stream_endpoint_refuses_with_501(self):
        response = _serving().refuse_stream()
        self.assertEqual(response.status_code, 501)

    def test_the_refusal_names_both_working_routes(self):
        body = _serving().refuse_stream().body.decode()
        self.assertIn("/api/v1/generate", body)
        self.assertIn("/v1/completions", body)

    def test_it_says_why_a_half_faithful_stream_is_worse(self):
        body = _serving().refuse_stream().body.decode()
        self.assertIn("half-faithful", body)


class TestAbortIsRefusedBecauseItCannotBeHonoured(unittest.TestCase):
    """Kobold aborts THE generation, presuming one user. This server is
    multi-tenant, so a guess would cancel a stranger's request."""

    def test_abort_refuses_with_501(self):
        self.assertEqual(_serving().refuse_abort().status_code, 501)

    def test_the_refusal_explains_the_multi_tenant_reason(self):
        body = _serving().refuse_abort().body.decode()
        self.assertIn("multi-tenant", body)
        self.assertIn("another client", body)


class TestTheProbeEndpoints(unittest.TestCase):
    def test_model_returns_the_served_name_in_the_kobold_envelope(self):
        self.assertEqual(_serving().get_model(), {"result": "test-model"})

    def test_version_does_not_claim_to_be_a_koboldcpp_build(self):
        """Claiming a KoboldCpp version number asserts compatibility with a
        build whose behaviour this does not implement."""
        self.assertEqual(_serving().get_version()["result"], "sglang")


class TestDriftAndOrdering(unittest.TestCase):
    """The two pins carried over from the Ollama cut."""

    def test_no_field_is_both_mapped_and_refused(self):
        mapped = set(KoboldServing.FIELD_MAP)
        refused = set(KoboldServing.UNSUPPORTED_FIELDS)
        inert = set(KoboldServing.INERT_FIELDS)
        self.assertEqual(mapped & refused, set())
        self.assertEqual(mapped & inert, set())
        self.assertEqual(refused & inert, set())

    def test_every_declared_request_field_is_classified(self):
        """DRIFT PIN. A field added to the protocol and to no table is a
        silent drop again -- the exact defect this surface exists to avoid."""
        declared = set(KoboldGenerateRequest.model_fields) - {"prompt"}
        classified = (
            set(KoboldServing.FIELD_MAP)
            | set(KoboldServing.UNSUPPORTED_FIELDS)
            | set(KoboldServing.INERT_FIELDS)
        )
        self.assertEqual(
            declared - classified,
            set(),
            "unclassified request fields would be silently dropped",
        )

    def test_the_refusal_precedes_the_translation(self):
        """ORDERING PIN. Refusing after building the request would still be
        before generation, but the order states the intent: nothing is
        prepared for a request that will not run."""
        import inspect

        src = inspect.getsource(KoboldServing.handle_generate)
        self.assertLess(
            src.index("unsupported_reasons"),
            src.index("to_completion_request"),
            "the refusal must precede the translation",
        )


class TestTheRoutesAreMounted(unittest.TestCase):
    """#421: an adapter directory with no route registration is an
    advertised-but-unwired feature. Proven at the source, not assumed."""

    def setUp(self):
        import inspect

        from sglang.srt.entrypoints import http_server

        self.src = inspect.getsource(http_server)

    def test_every_kobold_route_is_registered(self):
        for route in (
            "/api/v1/generate",
            "/api/v1/model",
            "/api/extra/version",
            "/api/extra/generate/stream",
            "/api/extra/abort",
        ):
            with self.subTest(route=route):
                self.assertIn(f'"{route}"', self.src)

    def test_the_serving_object_is_constructed_on_app_state(self):
        self.assertIn("state.kobold_serving = KoboldServing(", self.src)


if __name__ == "__main__":
    unittest.main()
