# Copyright 2025 SGLang Team
# Licensed under the Apache License, Version 2.0
"""A live OpenAI-compatible server with the engine mocked out (#335-M0).

The point of this harness is that everything between the socket and the
scheduler is real: the real FastAPI app from
``sglang.srt.entrypoints.http_server``, the real routes, the real exception
handlers, the real serving classes, the real protocol models, served by a real
uvicorn over a real TCP port. Only ``TokenizerManager.generate_request`` --
the one thing that needs a GPU -- is replaced.

That matters because the deliverable is compatibility with real clients, and
the failures compatibility work is trying to catch live exactly in the layers
a TestClient or a hand-rolled request would skip: SSE framing, error-body
shape as an SDK parses it, alias serialization, status codes as typed
exceptions.
"""

from __future__ import annotations

import contextlib
import socket
import tempfile
import threading
import time
from pathlib import Path
from typing import Any, Iterator, Optional
from unittest.mock import MagicMock

MODEL_NAME = "test-model"
#: Cached by the environment; no network at test time.
TOKENIZER_NAME = "Qwen/Qwen3-0.6B"

#: What the mocked engine "generates". Short and recognisable so a test can
#: assert on it without depending on any model behaviour.
CANNED_TEXT = "mocked completion"


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return int(s.getsockname()[1])


def _meta_info(
    prompt_tokens: int = 7,
    completion_tokens: int = 3,
    finish_reason: Optional[dict] = None,
) -> dict[str, Any]:
    return {
        "id": "req-0",
        # ``None`` until the last chunk, exactly as the scheduler reports it:
        # a finish reason on every chunk would make the serving layer treat
        # each one as final and the deltas would come out wrong.
        "finish_reason": finish_reason,
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "cached_tokens": 0,
        "e2e_latency": 0.01,
        "weight_version": "harness",
    }


class MockEngine:
    """Stands in for ``TokenizerManager`` where the scheduler would be.

    Deliberately not a full fake: the serving classes read a handful of config
    attributes and call ``generate_request``. Everything else is a MagicMock,
    so a serving class reaching for something new fails loudly here rather
    than silently taking a different branch.
    """

    def __init__(self, tokenizer, *, embedding_dim: int = 4):
        self.embedding_dim = embedding_dim
        self.generate_calls: list[Any] = []

    async def generate_request(self, obj, request=None):
        raise NotImplementedError


def build_mock_tokenizer_manager(tokenizer, *, embedding_dim: int = 4) -> MagicMock:
    """A ``TokenizerManager`` stand-in whose generation is canned.

    ``server_args`` is a real :class:`ServerArgs`, not a mock. Every flag then
    sits at its production default, so a serving class reading a flag this
    harness never heard of takes the same branch it would in a real server --
    a mocked ``server_args`` returns a truthy Mock for every unknown attribute
    and silently drives the opposite branch.
    """
    from sglang.srt.server_args import ServerArgs

    tm = MagicMock()
    tm.tokenizer = tokenizer
    tm.model_path = TOKENIZER_NAME
    tm.served_model_name = MODEL_NAME
    tm.is_generation = True
    tm.gracefully_exit = False
    tm.log_requests = False
    tm.server_args = ServerArgs(model_path=TOKENIZER_NAME, device="cpu")
    tm.request_logger.log_requests = False
    tm.request_logger.log_requests_level = 0
    # Streaming responses attach this as a Starlette background task and await
    # it after the last chunk; a Mock there raises inside the ASGI send path.
    tm.create_abort_task.return_value = None
    tm.model_config.get_default_sampling_params.return_value = {}
    tm.model_config.hf_config.model_type = "qwen3"
    tm.model_config.hf_config.architectures = ["Qwen3ForCausalLM"]
    tm.model_config.context_len = 4096

    embedding_dim_local = embedding_dim

    async def generate_request(obj, request=None):
        """Canned generation. One item per input, streaming or not."""
        is_embedding = obj.__class__.__name__ == "EmbeddingReqInput"
        if is_embedding:
            texts = obj.text if isinstance(obj.text, list) else [obj.text]
            yield [
                {
                    "embedding": [0.5 * (i + 1)] * embedding_dim_local,
                    "meta_info": _meta_info(
                        prompt_tokens=5,
                        completion_tokens=0,
                        finish_reason={"type": "stop"},
                    ),
                }
                for i, _ in enumerate(texts)
            ]
            return

        if getattr(obj, "stream", False):
            emitted = ""
            for piece in CANNED_TEXT.split(" "):
                emitted = (emitted + " " + piece).strip()
                yield {
                    "text": emitted,
                    "output_ids": [1],
                    "meta_info": _meta_info(
                        completion_tokens=len(emitted.split()),
                        finish_reason=(
                            {"type": "stop"} if emitted == CANNED_TEXT else None
                        ),
                    ),
                }
            return

        yield {
            "text": CANNED_TEXT,
            "output_ids": [1, 2, 3],
            "meta_info": _meta_info(finish_reason={"type": "stop"}),
        }

    tm.generate_request = generate_request
    return tm


@contextlib.contextmanager
def live_server(
    *,
    tokenizer,
    registry_view=None,
    image_lane_url: str = "",
    speech_lane_url: str = "",
    architectures: Optional[list[str]] = None,
    training_service=None,
    workbench_service=None,
) -> Iterator[str]:
    """Run the real app on a real port. Yields the base URL.

    ``registry_view`` / ``image_lane_url`` / ``speech_lane_url`` are injected
    into the adapters rather than set as process env, so a test can describe a
    rig state without a running registry or diffusion service.

    ``training_service`` is the #341 service the fine-tuning routes talk to.
    Injected for the same reason: a test describes a machine and an executor
    without a card, a ledger or an installed training suite, and everything
    between the socket and that service stays real. ``workbench_service`` is
    the #347 idle workbench, injected on the same terms.
    """
    import uvicorn

    from sglang.srt.entrypoints import http_server
    from sglang.srt.entrypoints.openai.registry_view import RegistryView
    from sglang.srt.entrypoints.openai.serving_chat import OpenAIServingChat
    from sglang.srt.entrypoints.openai.serving_completions import (
        OpenAIServingCompletion,
    )
    from sglang.srt.entrypoints.openai.serving_embedding import OpenAIServingEmbedding
    from sglang.srt.entrypoints.openai.serving_files import OpenAIServingFiles
    from sglang.srt.entrypoints.openai.serving_finetune import (
        OpenAIServingFineTuning,
    )
    from sglang.srt.entrypoints.openai.serving_images import OpenAIServingImages
    from sglang.srt.entrypoints.openai.serving_speech import OpenAIServingSpeech
    from sglang.srt.entrypoints.openai.serving_transcription import (
        OpenAIServingTranscription,
    )

    tm = build_mock_tokenizer_manager(tokenizer)
    if architectures is not None:
        tm.model_config.hf_config.architectures = architectures
    template_manager = MagicMock()
    template_manager.chat_template_name = None
    template_manager.jinja_chat_template = None
    template_manager.completion_template_name = None

    app = http_server.app
    previous_state = dict(app.state._state)
    previous_global = http_server.get_global_state()
    http_server.set_global_state(
        http_server._GlobalState(
            tokenizer_manager=tm, template_manager=template_manager, scheduler_info={}
        )
    )

    view = registry_view if registry_view is not None else RegistryView()

    app.state.openai_serving_chat = OpenAIServingChat(tm, template_manager)
    app.state.openai_serving_completion = OpenAIServingCompletion(tm, template_manager)
    app.state.openai_serving_embedding = OpenAIServingEmbedding(tm, template_manager)
    app.state.openai_serving_transcription = OpenAIServingTranscription(tm)
    app.state.openai_serving_images = OpenAIServingImages(
        lane_url_resolver=lambda: image_lane_url, view_resolver=lambda: view
    )
    app.state.openai_serving_speech = OpenAIServingSpeech(
        lane_url_resolver=lambda: speech_lane_url, view_resolver=lambda: view
    )

    if training_service is None:
        from sglang.srt.training.service import (
            TrainingService,
            TrainingServiceConfig,
        )

        # Disabled by default: a test that does not ask for the tenant gets
        # the production default, so the "switched off" rejection is covered
        # by every other test in the file for free.
        training_service = TrainingService(
            TrainingServiceConfig(
                enabled=False,
                artifact_root=Path(tempfile.mkdtemp(prefix="htsglang-training-")),
            )
        )
    app.state.training_service = training_service
    app.state.openai_serving_files = OpenAIServingFiles(training_service)
    app.state.openai_serving_fine_tuning = OpenAIServingFineTuning(training_service)

    if workbench_service is None:
        from sglang.srt.workbench.scheduler import WorkbenchConfig
        from sglang.srt.workbench.service import WorkbenchService

        # Disabled by default, same reasoning as the training service above.
        workbench_service = WorkbenchService(
            WorkbenchConfig(
                enabled=False,
                artifact_root=Path(tempfile.mkdtemp(prefix="htsglang-workbench-")),
            )
        )
    app.state.workbench_service = workbench_service
    # The Ollama emulation rides the same lanes; it is served by the same app
    # and therefore covered by the same harness.
    from sglang.srt.entrypoints.ollama.serving import OllamaServing  # noqa: PLC0415

    # #335: the Ollama surface COMPOSES the OpenAI fronts now, so the harness
    # hands it those rather than the tokenizer manager -- the same wiring
    # http_server does.
    app.state.ollama_serving = OllamaServing(
        app.state.openai_serving_chat,
        app.state.openai_serving_completion,
        model_name=getattr(tm, "served_model_name", "test-model"),
        context_len=getattr(getattr(tm, "model_config", None), "context_len", None),
    )

    # The production lifespan boots a tokenizer manager and a warmup thread;
    # this harness has already provided both halves it would build.
    @contextlib.asynccontextmanager
    async def _noop_lifespan(_app):
        # The training tenant is the one piece the production lifespan starts
        # that this harness still needs, because its scheduler is an asyncio
        # task and must be created on the server's own loop, not the test's.
        training_service.start(start_tenant=not workbench_service.config.enabled)
        workbench_service.start()
        try:
            yield
        finally:
            await workbench_service.stop()
            await training_service.stop()

    previous_lifespan = app.router.lifespan_context
    app.router.lifespan_context = _noop_lifespan

    # ``/v1/models`` reads the registry through the module-level function, so
    # describing a rig state means replacing that lookup rather than standing
    # up a control plane.
    previous_fetch = http_server.fetch_registry_view
    http_server.fetch_registry_view = lambda *a, **kw: view

    port = free_port()
    config = uvicorn.Config(
        app, host="127.0.0.1", port=port, log_level="warning", access_log=False
    )
    server = uvicorn.Server(config)
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()

    deadline = time.time() + 30
    while not server.started and time.time() < deadline:
        time.sleep(0.05)
    if not server.started:
        raise RuntimeError("harness server did not start within 30s")

    try:
        yield f"http://127.0.0.1:{port}"
    finally:
        server.should_exit = True
        thread.join(timeout=15)
        app.router.lifespan_context = previous_lifespan
        http_server.fetch_registry_view = previous_fetch
        app.state._state.clear()
        app.state._state.update(previous_state)
        http_server.set_global_state(previous_global)
