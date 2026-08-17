"""#335 — /v1/embeddings on a generative-only checkpoint must refuse by name.

THE CANDIDATE, SETTLED. `ANALYSE_335` §9 recorded "no named refusal found, but
I did not trace the path". Traced now, and the answer is the second of the
three possibilities: **no gate, and the request fails LOUDLY but wrongly** --
as a 500 with a stack trace rather than a protocol-conformant error.

THE CHAIN, at code:

* ``ModelConfig.is_generation`` (`configs/model_config.py:540`, derived by
  ``is_generation_model`` at `:2010`) is the authority, and it is already
  reachable at the front as ``tokenizer_manager.is_generation`` -- the HTTP
  layer uses it at `http_server.py:979` and `:2911`. Nothing had to be
  invented.
* ``OpenAIServingEmbedding._validate_request``
  (`entrypoints/openai/serving_embedding.py:68`) checked encoding format and
  input shape and **nothing about capability**.
* ``ModelRunner`` sets ``get_embedding=True`` ONLY when the model is not a
  generation model (`model_executor/model_runner.py:4033-4034`), so a
  generative checkpoint never produces one.
* ``_build_embedding_response`` then does a BARE SUBSCRIPT,
  ``ret_item["embedding"]`` (`serving_embedding.py:309`), while the caller
  catches only ``ValueError`` (`:286`). A ``KeyError`` is not caught.

So the caller received an HTTP 500 and a stack trace in the server log for a
request that was simply not supported by the loaded model. It is not the
silent-falseness class -- no garbage vector was returned -- but a 500 tells a
client "this server is broken" when the truth is "this checkpoint does not do
embeddings", and those provoke very different next actions.
"""

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

import inspect
import unittest

from sglang.srt.entrypoints.openai.protocol import EmbeddingRequest
from sglang.srt.entrypoints.openai.serving_embedding import OpenAIServingEmbedding
from sglang.test.test_utils import CustomTestCase


class _TM:
    """Tokenizer manager, at the two attributes this gate reads."""

    def __init__(self, is_generation: bool):
        self.is_generation = is_generation
        self.model_path = "test-model"
        self.server_args = None


def _serving(is_generation: bool):
    s = OpenAIServingEmbedding.__new__(OpenAIServingEmbedding)
    s.tokenizer_manager = _TM(is_generation)
    return s


def _req(**kw):
    return EmbeddingRequest(model="m", input="hello", **kw)


class TestAGenerativeCheckpointRefusesByName(CustomTestCase):
    def test_a_generation_only_model_refuses(self):
        msg = _serving(is_generation=True)._validate_request(_req())
        self.assertIsNotNone(
            msg, "a generative checkpoint cannot embed and must say so"
        )

    def test_the_refusal_names_the_capability_not_the_input(self):
        """The caller's input was fine. Blaming it would send them editing a
        request that was never the problem."""
        msg = _serving(is_generation=True)._validate_request(_req())
        self.assertIn("embedding", msg.lower())
        self.assertNotIn("empty", msg.lower())

    def test_the_refusal_names_the_flag_that_would_serve_it(self):
        """A refusal that only blocks teaches nothing; this one routes."""
        msg = _serving(is_generation=True)._validate_request(_req())
        self.assertIn("--is-embedding", msg)

    def test_an_embedding_model_is_not_refused(self):
        """THE FALSIFIER. A gate that cannot pass would break every embedding
        deployment, which is a far worse defect than the one it fixes."""
        self.assertIsNone(_serving(is_generation=False)._validate_request(_req()))

    def test_the_capability_check_runs_before_the_input_checks(self):
        """An empty input on a generative checkpoint should hear about the
        checkpoint, not the input -- otherwise fixing the input just moves the
        caller to the next wrong answer."""
        msg = _serving(is_generation=True)._validate_request(
            EmbeddingRequest(model="m", input="")
        )
        self.assertIn("embedding", msg.lower())

    def test_the_input_checks_still_work_on_an_embedding_model(self):
        """The falsifier for the ordering change: the checks it now runs after
        must still run."""
        msg = _serving(is_generation=False)._validate_request(
            EmbeddingRequest(model="m", input="")
        )
        self.assertIsNotNone(msg)
        self.assertIn("empty", msg.lower())


class TestTheDefectThisReplaces(CustomTestCase):
    """Recorded so the fix is not mistaken for style. What the caller got
    before is reconstructible from the source, and it was a 500."""

    def test_the_response_builder_still_subscripts_bare(self):
        """If this ever becomes a .get() with a default, the failure mode
        changes from a 500 to a silently-empty vector -- which would be the
        WORSE class -- and this gate becomes the only thing standing between a
        caller and a plausible wrong answer."""
        src = inspect.getsource(OpenAIServingEmbedding._build_embedding_response)
        self.assertIn('ret_item["embedding"]', src)

    def test_the_handler_catches_only_value_error(self):
        """Which is why the KeyError above reached the client as a 500."""
        src = inspect.getsource(
            OpenAIServingEmbedding._handle_non_streaming_request
        )
        self.assertIn("except ValueError", src)
        self.assertNotIn("except KeyError", src)


if __name__ == "__main__":
    unittest.main()
