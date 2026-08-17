# Copyright 2023-2024 SGLang Team
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""
The entry point of inference server. (SRT = SGLang Runtime)

This file implements HTTP APIs for the inference engine via fastapi.
"""

import asyncio
import dataclasses
import json
import logging
import os
import tempfile
import threading
import time
import uuid
from contextlib import asynccontextmanager
from http import HTTPStatus
from pathlib import Path
from typing import (
    Annotated,
    Any,
    AsyncGenerator,
    AsyncIterator,
    Callable,
    Dict,
    List,
    Optional,
    Union,
)

import numpy as np
import requests
import uvicorn
import uvloop
from fastapi import (
    Body,
    Depends,
    FastAPI,
    File,
    Form,
    HTTPException,
    Query,
    Request,
    UploadFile,
    WebSocket,
)
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import ORJSONResponse, Response, StreamingResponse

from sglang.srt.constants import HEALTH_CHECK_RID_PREFIX
from sglang.srt.disaggregation.utils import FAKE_BOOTSTRAP_HOST, DisaggregationMode
from sglang.srt.entrypoints.anthropic.protocol import (
    AnthropicCountTokensRequest,
    AnthropicMessagesRequest,
)
from sglang.srt.entrypoints.anthropic.serving import AnthropicServing
from sglang.srt.entrypoints.engine import (
    Engine,
    init_tokenizer_manager,
    run_detokenizer_process,
    run_scheduler_process,
)
from sglang.srt.entrypoints.ollama.protocol import (
    OllamaChatRequest,
    OllamaGenerateRequest,
    OllamaShowRequest,
)
from sglang.srt.entrypoints.ollama.serving import OllamaServing
from sglang.srt.entrypoints.openai.errors import (
    LaneUnavailable,
    error_type_for_status,
    openai_error_response,
)
from sglang.srt.entrypoints.openai.protocol import (
    ChatCompletionRequest,
    ClassifyRequest,
    CompletionRequest,
    DetokenizeRequest,
    EmbeddingRequest,
    ErrorResponse,
    ModelCard,
    ModelList,
    ResponsesRequest,
    ScoringRequest,
    TokenizeRequest,
    V1RerankReqInput,
)
from sglang.srt.entrypoints.openai.registry_view import fetch_registry_view
from sglang.srt.entrypoints.openai.serving_classify import OpenAIServingClassify
from sglang.srt.entrypoints.openai.serving_completions import OpenAIServingCompletion
from sglang.srt.entrypoints.openai.serving_embedding import OpenAIServingEmbedding
from sglang.srt.entrypoints.openai.serving_files import OpenAIServingFiles
from sglang.srt.entrypoints.openai.serving_finetune import OpenAIServingFineTuning
from sglang.srt.entrypoints.openai.serving_images import OpenAIServingImages
from sglang.srt.entrypoints.openai.serving_rerank import OpenAIServingRerank
from sglang.srt.entrypoints.openai.serving_score import OpenAIServingScore
from sglang.srt.entrypoints.openai.serving_speech import OpenAIServingSpeech
from sglang.srt.entrypoints.openai.serving_tokenize import (
    OpenAIServingDetokenize,
    OpenAIServingTokenize,
)
from sglang.srt.entrypoints.openai.serving_transcription import (
    OpenAIServingTranscription,
)
from sglang.srt.entrypoints.request_headers import apply_header_overrides
from sglang.srt.entrypoints.warmup import execute_warmups
from sglang.srt.environ import envs
from sglang.srt.function_call.function_call_parser import FunctionCallParser
from sglang.srt.liveness import guard_generate_stream
from sglang.srt.managers.io_struct import (
    AbortReq,
    AttachHiCacheStorageReqInput,
    CheckWeightsReqInput,
    CloseSessionReqInput,
    ConfigureLoggingReq,
    ContinueGenerationReqInput,
    DestroyWeightsUpdateGroupReqInput,
    DumperControlReqInput,
    EmbeddingReqInput,
    GenerateReqInput,
    GetWeightsByNameReqInput,
    InitWeightsSendGroupForRemoteInstanceReqInput,
    InitWeightsUpdateGroupReqInput,
    KvReshardReqInput,
    PhaseFlipReqInput,
    SessionHandoverReqInput,
    LoadLoRAAdapterFromTensorsReqInput,
    LoadLoRAAdapterReqInput,
    OpenSessionReqInput,
    ParseFunctionCallReq,
    PauseGenerationReqInput,
    ProfileReq,
    ReleaseMemoryOccupationReqInput,
    ResizeHiCacheStorageReqInput,
    ResumeMemoryOccupationReqInput,
    SendWeightsToRemoteInstanceReqInput,
    SeparateReasoningReqInput,
    SessionCheckpointReqInput,
    SessionHandoverReqInput,
    SetInternalStateReq,
    SlowDownReqInput,
    UnloadLoRAAdapterReqInput,
    UpdateWeightFromDiskReqInput,
    UpdateWeightsFromDistributedReqInput,
    UpdateWeightsFromIPCReqInput,
    UpdateWeightsFromTensorReqInput,
    UpdateWeightVersionReqInput,
    VertexGenerateReqInput,
    VramBudgetReqInput,
)
from sglang.srt.managers.multi_tokenizer_mixin import (
    MultiTokenizerRouter,
    TokenizerWorker,
    get_main_process_id,
    get_tokenizer_worker_class,
    read_from_shared_memory,
    write_data_for_multi_tokenizer,
)
from sglang.srt.managers.tokenizer_manager import ServerStatus, TokenizerManager
from sglang.srt.observability.func_timer import enable_func_timer
from sglang.srt.observability.trace import (
    process_tracing_init,
    set_global_trace_level,
    trace_set_thread_info,
)
from sglang.srt.parser.reasoning_parser import ReasoningParser
from sglang.srt.parser.template_manager import TemplateManager
from sglang.srt.server_args import PortArgs, ServerArgs
from sglang.srt.training.activity import note_serving_activity
from sglang.srt.utils import (
    add_prometheus_middleware,
    add_prometheus_track_response_middleware,
    delete_directory,
    get_bool_env_var,
    is_mps,
    kill_process_tree,
    set_uvicorn_logging_configs,
)
from sglang.srt.utils.auth import AuthLevel, app_has_admin_force_endpoints, auth_level
from sglang.srt.utils.json_response import (
    SGLangORJSONResponse,
    dumps_json,
    orjson_response,
)
from sglang.srt.utils.msgspec_utils import msgspec_to_builtins
from sglang.srt.utils.watchdog import SubprocessWatchdog
from sglang.utils import get_exception_traceback
from sglang.version import __version__

logger = logging.getLogger(__name__)
asyncio.set_event_loop_policy(uvloop.EventLoopPolicy())

# Global constants
HEALTH_CHECK_TIMEOUT = int(os.getenv("SGLANG_HEALTH_CHECK_TIMEOUT", 20))
WAIT_WEIGHTS_READY_TIMEOUT = int(os.getenv("SGLANG_WAIT_WEIGHTS_READY_TIMEOUT", 120))


# Store global states
@dataclasses.dataclass
class _GlobalState:
    tokenizer_manager: Union[TokenizerManager, MultiTokenizerRouter, TokenizerWorker]
    template_manager: TemplateManager
    scheduler_info: Dict


_global_state: Optional[_GlobalState] = None


def set_global_state(global_state: _GlobalState):
    global _global_state
    _global_state = global_state


def get_global_state() -> _GlobalState:
    return _global_state


async def init_multi_tokenizer() -> ServerArgs:
    """
    Initialization function for multi-process tokenizer mode.
    It read args information from shm and inits tokenizer manager for current process.
    """

    # Read configuration from shared memory
    main_pid = get_main_process_id()
    port_args, server_args, scheduler_info = read_from_shared_memory(
        f"multi_tokenizer_args_{main_pid}"
    )
    server_args: ServerArgs
    port_args: PortArgs

    # API key authentication is not supported in multi-tokenizer mode
    assert (
        server_args.api_key is None
    ), "API key is not supported in multi-tokenizer mode"

    # Create a new ipc name for the current process
    port_args.tokenizer_ipc_name = (
        f"ipc://{tempfile.NamedTemporaryFile(delete=False).name}"
    )
    logger.info(
        f"Start multi-tokenizer worker process {os.getpid()}, "
        f"ipc_name={port_args.tokenizer_ipc_name}"
    )

    # Launch multi-tokenizer manager process
    tokenizer_worker_class = get_tokenizer_worker_class(server_args)
    tokenizer_manager = tokenizer_worker_class(server_args, port_args)
    template_manager = TemplateManager()
    template_manager.initialize_templates(
        tokenizer_manager=tokenizer_manager,
        model_path=server_args.model_path,
        chat_template=server_args.chat_template,
        completion_template=server_args.completion_template,
    )

    tokenizer_manager.max_req_input_len = scheduler_info["max_req_input_len"]

    set_global_state(
        _GlobalState(
            tokenizer_manager=tokenizer_manager,
            template_manager=template_manager,
            scheduler_info=scheduler_info,
        )
    )

    return server_args


def client_liveness_config(server_args: ServerArgs):
    """The #344 per-endpoint-class policy, assembled from the server flags.

    ``--training-event-stream-timeout-s`` predates the general flag and stays
    supported as shorthand for one class. The general flag wins where both
    name the same class, because an operator who wrote out an explicit
    per-class table meant it.
    """
    from sglang.srt.liveness import EndpointClass, LivenessConfig

    config = LivenessConfig.parse(
        server_args.client_liveness_timeouts,
        poll_interval_s=float(server_args.client_liveness_poll_interval_s),
        teardown_timeout_s=float(server_args.client_liveness_teardown_timeout_s),
        grace_fraction=float(server_args.client_liveness_grace_fraction),
    )
    config.timeouts_s.setdefault(
        EndpointClass.TRAINING_EVENTS.value,
        float(server_args.training_event_stream_timeout_s),
    )
    return config


def install_client_liveness(server_args: ServerArgs):
    """Install the process-wide liveness policy and its ledger bridge (#344).

    Two globals, both write-once at startup: the policy, which the streaming
    guards three layers down read instead of being handed a config through
    every constructor, and the grace bridge, which mirrors "this tenant's
    consumer went quiet" onto the ledger so out-of-process reclaimers can see
    it. A rig without a writable ledger keeps the policy and loses only the
    cross-process view.
    """
    from sglang.srt.liveness import (
        global_attachment_registry,
        set_global_liveness_config,
    )

    config = client_liveness_config(server_args)
    set_global_liveness_config(config)
    try:
        from sglang.srt.liveness import attach_ledger_grace_bridge
        from sglang.srt.registry.ledger import ReservationStore

        attach_ledger_grace_bridge(global_attachment_registry(), ReservationStore())
    except Exception as exc:  # noqa: BLE001 - reporting, not correctness
        logger.info(
            "client liveness: no VRAM ledger to publish grace into (%s: %s); "
            "detection and release are unaffected",
            type(exc).__name__,
            exc,
        )
    return config


def build_training_service(server_args: ServerArgs):
    """Assemble the #341 training service from the server arguments.

    Built unconditionally. A disabled tenant still answers the routes -- with
    a rejection that names ``--enable-training-tenant`` -- because a 404 on
    ``/v1/fine_tuning/jobs`` is indistinguishable from an old server, and a
    client cannot act on that.
    """
    from sglang.srt.training.feasibility import default_artifact_root
    from sglang.srt.training.service import TrainingService, TrainingServiceConfig

    root = (
        Path(server_args.training_artifact_root)
        if server_args.training_artifact_root
        else default_artifact_root()
    )
    config = TrainingServiceConfig(
        enabled=bool(server_args.enable_training_tenant),
        artifact_root=root,
        grace_seconds=float(server_args.training_idle_grace_seconds),
        poll_seconds=float(server_args.training_poll_seconds),
        preempt_timeout_s=float(server_args.training_preempt_timeout_s),
        save_steps=int(server_args.training_save_steps),
        default_backend=str(server_args.training_default_backend),
        default_method=str(server_args.training_default_method),
        model_root=str(server_args.training_model_root or ""),
        event_stream_timeout_s=float(server_args.training_event_stream_timeout_s),
    )
    reservation_store = None
    if config.enabled:
        try:
            from sglang.srt.registry.ledger import ReservationStore

            reservation_store = ReservationStore()
        except Exception as exc:  # noqa: BLE001 - the tenant runs without it
            logger.warning(
                "training tenant: no VRAM ledger (%s: %s); jobs will run "
                "without a cross-process reservation",
                type(exc).__name__,
                exc,
            )
    return TrainingService(config, reservation_store=reservation_store)


def build_workbench_service(server_args: ServerArgs, training_service):
    """Assemble the #347 idle workbench from the server arguments.

    Built unconditionally, like the training service and for the same reason:
    a disabled workbench answers ``GET /x-htsglang/workbench`` with
    ``enabled: false`` and the tenant list, which is what an operator asking
    "why is nothing being tuned" needs. A 404 answers nothing.
    """
    from sglang.srt.workbench.service import build_service

    return build_service(server_args, training_service=training_service)


@asynccontextmanager
async def lifespan(fast_api_app: FastAPI):
    grpc_handle = None
    warmup_thread = None
    if getattr(fast_api_app, "is_single_tokenizer_mode", False):
        server_args = fast_api_app.server_args
        warmup_thread_kwargs = fast_api_app.warmup_thread_kwargs
        thread_label = "Tokenizer"
    else:
        # Initialize multi-tokenizer support for worker processes
        server_args = await init_multi_tokenizer()
        warmup_thread_kwargs = dict(server_args=server_args)
        thread_label = f"MultiTokenizer-{_global_state.tokenizer_manager.worker_id}"

    # Add prometheus middleware
    if server_args.enable_metrics:
        add_prometheus_middleware(app)
        enable_func_timer()

    # Init tracing
    if server_args.enable_trace:
        process_tracing_init(
            server_args.otlp_traces_endpoint,
            "sglang",
            trace_modules=server_args.trace_modules,
        )
        if server_args.disaggregation_mode == "prefill":
            thread_label = "Prefill" + thread_label
        elif server_args.disaggregation_mode == "decode":
            thread_label = "Decode" + thread_label
        trace_set_thread_info(thread_label)

    # Universal client liveness (#344). Installed before the handlers, because
    # every streaming handler below reads the process-wide policy when it puts
    # a watchdog on its response.
    liveness_config = install_client_liveness(server_args)

    # Initialize OpenAI serving handlers
    fast_api_app.state.openai_serving_completion = OpenAIServingCompletion(
        _global_state.tokenizer_manager, _global_state.template_manager
    )
    fast_api_app.state.openai_serving_chat = (
        _global_state.tokenizer_manager.serving_chat_class(
            _global_state.tokenizer_manager, _global_state.template_manager
        )
    )
    fast_api_app.state.openai_serving_embedding = OpenAIServingEmbedding(
        _global_state.tokenizer_manager, _global_state.template_manager
    )
    fast_api_app.state.openai_serving_classify = OpenAIServingClassify(
        _global_state.tokenizer_manager, _global_state.template_manager
    )
    fast_api_app.state.openai_serving_score = OpenAIServingScore(
        _global_state.tokenizer_manager
    )
    fast_api_app.state.openai_serving_rerank = OpenAIServingRerank(
        _global_state.tokenizer_manager, _global_state.template_manager
    )
    fast_api_app.state.openai_serving_tokenize = OpenAIServingTokenize(
        _global_state.tokenizer_manager, _global_state.template_manager
    )
    fast_api_app.state.openai_serving_detokenize = OpenAIServingDetokenize(
        _global_state.tokenizer_manager
    )
    fast_api_app.state.openai_serving_transcription = OpenAIServingTranscription(
        _global_state.tokenizer_manager
    )
    # Adapters, not lanes: they forward to the diffusion / speech services when
    # those are configured and reject with the registry's numbers when not.
    fast_api_app.state.openai_serving_images = OpenAIServingImages()
    fast_api_app.state.openai_serving_speech = OpenAIServingSpeech()

    # The idle training tenant (#341). Built even when the tenant is disabled
    # so /v1/files and /v1/fine_tuning/jobs answer with a named rejection
    # rather than a 404 that looks like an old server.
    training_service = build_training_service(server_args)
    fast_api_app.state.training_service = training_service
    # With the #347 workbench, training is one tenant in a priority order and
    # the workbench owns its loop; the service comes up surface-only so two
    # schedulers are never deciding when training runs.
    workbench_enabled = bool(getattr(server_args, "enable_idle_workbench", False))
    training_service.start(start_tenant=not workbench_enabled)
    fast_api_app.state.openai_serving_files = OpenAIServingFiles(training_service)
    fast_api_app.state.openai_serving_fine_tuning = OpenAIServingFineTuning(
        training_service,
        liveness=liveness_config,
    )

    # The idle workbench (#347): the queue of useful idle work, of which the
    # training tenant above is entry #1. Built even when disabled so
    # /x-htsglang/workbench reports "enabled: false" instead of a 404.
    workbench_service = build_workbench_service(server_args, training_service)
    fast_api_app.state.workbench_service = workbench_service
    workbench_service.start()

    # Initialize Ollama-compatible serving handler
    fast_api_app.state.ollama_serving = OllamaServing(_global_state.tokenizer_manager)

    # Initialize Anthropic-compatible serving handler
    fast_api_app.state.anthropic_serving = AnthropicServing(
        fast_api_app.state.openai_serving_chat
    )

    # Launch tool server
    tool_server = None
    if server_args.tool_server == "demo":
        from sglang.srt.entrypoints.openai.tool_server import DemoToolServer

        tool_server = DemoToolServer()
    elif server_args.tool_server:
        from sglang.srt.entrypoints.openai.tool_server import MCPToolServer

        tool_server = MCPToolServer()
        await tool_server.add_tool_server(server_args.tool_server)
    elif envs.EXA_API_KEY.get():
        from sglang.srt.entrypoints.openai.tool_server import NativeToolServer

        tool_server = NativeToolServer()

    try:
        from sglang.srt.entrypoints.openai.serving_responses import (
            OpenAIServingResponses,
        )

        fast_api_app.state.openai_serving_responses = OpenAIServingResponses(
            _global_state.tokenizer_manager,
            _global_state.template_manager,
            enable_prompt_tokens_details=True,
            tool_server=tool_server,
        )
    except Exception as e:
        # Optional endpoint; a load failure (e.g. the gpt-oss harmony vocab
        # download) must not look like a fatal error. One-line WARNING, full
        # traceback at DEBUG.
        logger.warning(
            f"OpenAI Responses API (/v1/responses) disabled: "
            f"OpenAIServingResponses init failed ({type(e).__name__}: {e})"
        )
        logger.debug(
            f"OpenAIServingResponses init traceback:\n{get_exception_traceback()}"
        )

    # Execute custom warmups
    if server_args.warmups is not None:
        await execute_warmups(
            server_args.disaggregation_mode,
            server_args.warmups.split(","),
            _global_state.tokenizer_manager,
        )
        logger.info("Warmup ended")

    # Start the native gRPC server and warmup inside the try so a failure in
    # either still runs the finally cleanup below. Native gRPC is enabled via
    # --grpc-port / SGLANG_GRPC_PORT; only the single-tokenizer process is
    # gRPC-capable (__post_init__ rejects --tokenizer-worker-num > 1).
    try:
        if (
            getattr(fast_api_app, "is_single_tokenizer_mode", False)
            and server_args.grpc_port is not None
            and not (server_args.smg_grpc_mode or server_args.grpc_mode)
        ):
            grpc_handle = _start_native_grpc_server_for_runtime(
                server_args=server_args,
                tokenizer_manager=_global_state.tokenizer_manager,
                template_manager=_global_state.template_manager,
                scheduler_info=_global_state.scheduler_info,
            )

        # Execute the general warmup
        warmup_thread = threading.Thread(
            target=_wait_and_warmup,
            kwargs=warmup_thread_kwargs,
        )
        warmup_thread.start()

        # Start the HTTP server
        yield
    finally:
        _shutdown_native_grpc_server(grpc_handle)
        if tool_server is not None and hasattr(tool_server, "aclose"):
            await tool_server.aclose()
        workbench_service = getattr(fast_api_app.state, "workbench_service", None)
        if workbench_service is not None:
            # Stopped before the training service, because the workbench owns
            # the training loop when it is enabled: stopping it in the other
            # order would restart nothing but would log a confusing pair of
            # shutdowns.
            await workbench_service.stop()
        training_service = getattr(fast_api_app.state, "training_service", None)
        if training_service is not None:
            # Stops the scheduler and releases any VRAM lease a training job
            # is holding, so a shutdown does not leave the ledger claiming
            # memory for a process that is gone.
            await training_service.stop()
        if warmup_thread is not None:
            warmup_thread.join()


# Fast API
app = FastAPI(
    lifespan=lifespan,
    openapi_url=None if get_bool_env_var("DISABLE_OPENAPI_DOC") else "/openapi.json",
)
#: Origins the CORS policy falls back to before --cors-allow-origins is known
#: (the app object is built at import time, server args arrive at launch).
DEFAULT_CORS_ALLOW_ORIGINS = ["*"]


def cors_policy(cors_allow_origins) -> dict:
    """CORS keyword arguments for one origin list.

    #510: the previous policy was ``allow_origins=["*"]`` **with**
    ``allow_credentials=True``. That pair is illegal per the Fetch standard
    (a wildcard `Access-Control-Allow-Origin` may not be sent with
    credentials) and, more concretely, it defeats a loopback-only bind: any
    page the operator has open can then POST to `127.0.0.1:30000` *and read
    the response*, which reaches every state-changing route on this app
    (audit #506, finding A2-F3).

    Credentials are therefore enabled only for an explicit origin list. A
    wildcard keeps working for plain, uncredentialed cross-origin calls, so
    the default deployment is unchanged for everything except the case that
    was never legal.
    """
    origins = list(cors_allow_origins or DEFAULT_CORS_ALLOW_ORIGINS)
    wildcard = "*" in origins
    return dict(
        allow_origins=origins,
        allow_credentials=not wildcard,
        allow_methods=["*"],
        allow_headers=["*"],
    )


def configure_cors(app, server_args) -> dict:
    """Apply --cors-allow-origins to ``app``, replacing the import-time policy.

    Replaces rather than appends: two CORSMiddleware instances would both
    answer a preflight and the effective policy would depend on stack order.
    Safe to call before startup, which is when the middleware stack is built.
    """
    policy = cors_policy(getattr(server_args, "cors_allow_origins", None))
    app.user_middleware = [
        m for m in app.user_middleware if m.cls is not CORSMiddleware
    ]
    app.add_middleware(CORSMiddleware, **policy)
    if not policy["allow_credentials"] and "*" in policy["allow_origins"]:
        logger.info(
            "CORS: wildcard origins, credentials disabled. Pass "
            "--cors-allow-origins with an explicit list to allow credentialed "
            "cross-origin requests."
        )
    return policy


app.add_middleware(CORSMiddleware, **cors_policy(DEFAULT_CORS_ALLOW_ORIGINS))

if envs.SGLANG_ENABLE_REQUEST_DECOMPRESSION.get():
    from sglang.srt.entrypoints.http_request_decompression import (
        RequestDecompressionMiddleware,
    )

    app.add_middleware(RequestDecompressionMiddleware)

# Include routers
from sglang.srt.entrypoints.v1_loads import router as v1_loads_router

app.include_router(v1_loads_router)


def _anthropic_validation_message(raw_errors) -> str:
    """Render Pydantic-style errors for an Anthropic /v1/messages route.

    Builds a short ``loc: msg`` digest that names the offending fields without
    leaking file paths or Python internals (the default ``str(exc)`` includes
    the dispatcher's ``File "/.../http_server.py"`` line).
    """
    parts: list[str] = []
    for err in raw_errors or []:
        loc = err.get("loc") or ()
        if loc:
            loc_str = ".".join(str(p) for p in loc if p not in ("body",))
        else:
            loc_str = ""
        msg = (err.get("msg") or "").strip()
        if loc_str and msg:
            parts.append(f"{loc_str}: {msg}")
        elif msg:
            parts.append(msg)
    text = "; ".join(parts) or "Invalid request"
    if len(text) > 500:
        text = text[:500] + "…"
    return text


def _anthropic_error_response(*, status_code: int, error_type: str, message: str):
    """Anthropic-format error envelope: {"type":"error","error":{"type":...,"message":...}}."""
    return ORJSONResponse(
        status_code=status_code,
        content={
            "type": "error",
            "error": {"type": error_type, "message": message},
        },
    )


@app.exception_handler(HTTPException)
async def validation_exception_handler(request: Request, exc: HTTPException):
    """Enrich HTTP exception with status code and other details.

    For /v1/responses, emit OpenAI-style nested error envelope:
    {"error": {"message": "...", "type": "...", "param": null, "code": <status>}}
    For /v1/messages, emit Anthropic-style envelope so SDK clients can parse it.
    """
    if request.url.path.startswith("/v1/messages"):
        # Map HTTP status to Anthropic error.type; fall back to api_error.
        anthropic_type = {
            400: "invalid_request_error",
            401: "authentication_error",
            403: "permission_error",
            404: "not_found_error",
            413: "request_too_large",
            422: "invalid_request_error",
            429: "rate_limit_error",
            500: "api_error",
            502: "api_error",
            503: "overloaded_error",
            504: "api_error",
        }.get(exc.status_code, "api_error")
        # 5xx must never echo upstream detail (may contain stack/PII).
        message = (
            "Internal server error"
            if exc.status_code >= 500
            else (str(exc.detail) if exc.detail else "Request failed")
        )
        return _anthropic_error_response(
            status_code=exc.status_code,
            error_type=anthropic_type,
            message=message,
        )

    # adjust fmt for responses api
    if request.url.path.startswith("/v1/responses"):
        nested_error = {
            "message": exc.detail,
            "type": HTTPStatus(exc.status_code).phrase,
            "param": None,
            "code": exc.status_code,
        }
        return ORJSONResponse(
            content={"error": nested_error}, status_code=exc.status_code
        )

    error = ErrorResponse(
        object="error",
        message=exc.detail,
        type=error_type_for_status(exc.status_code),
        code=exc.status_code,
    )
    return ORJSONResponse(
        content=error.to_openai_envelope(), status_code=exc.status_code
    )


# Custom exception handlers to change validation error status codes
@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    """Override FastAPI's default 422 validation error with 400.

    For /v1/messages, emit Anthropic-style envelope and scrub the message so
    file paths or Python internals from the default ``str(exc)`` representation
    never reach the client. For /v1/responses, keep OpenAI-style. Otherwise
    use the legacy ErrorResponse shape.
    """
    if request.url.path.startswith("/v1/messages"):
        return _anthropic_error_response(
            status_code=HTTPStatus.BAD_REQUEST.value,
            error_type="invalid_request_error",
            message=_anthropic_validation_message(exc.errors()),
        )

    exc_str = str(exc)
    errors_str = str(exc.errors())

    if errors_str and errors_str != exc_str:
        message = f"{exc_str} {errors_str}"
    else:
        message = exc_str

    if request.url.path.startswith("/v1/responses"):
        # adapt specially, for v1/responses API only (notice the error key is different)
        nested_error = {
            "message": message,
            "type": HTTPStatus.BAD_REQUEST.phrase,
            "param": None,
            "code": HTTPStatus.BAD_REQUEST.value,
        }
        return ORJSONResponse(status_code=400, content={"error": nested_error})

    err = ErrorResponse(
        message=message,
        type=error_type_for_status(HTTPStatus.BAD_REQUEST.value),
        code=HTTPStatus.BAD_REQUEST.value,
    )

    return ORJSONResponse(
        status_code=400,
        content=err.to_openai_envelope(),
    )


@app.exception_handler(LaneUnavailable)
async def lane_unavailable_handler(request: Request, exc: LaneUnavailable):
    """A capability with no lane behind it, answered in the OpenAI error shape.

    The status is the adapter's: 404 when the model genuinely is not served
    here, 503 when a configured lane is down, 501 for an endpoint no lane
    implements. Each maps to a distinct typed exception in the official SDKs,
    which is the difference between a client that can retry and one that can
    only fail.
    """
    return exc.to_response()


@app.exception_handler(Exception)
async def unhandled_exception_handler(request: Request, exc: Exception):
    """Last resort: an OpenAI-shaped 500 instead of Starlette's plain text.

    Without this, any exception raised outside a serving handler's own
    try/except (a missing ``app.state.openai_serving_*`` attribute, for
    instance) reaches the client as a text/plain body that no SDK can parse --
    the client sees a decode failure rather than a server error. The detail is
    deliberately not echoed: a 5xx message may carry a stack frame or a path.
    """
    logger.exception("Unhandled exception on %s", request.url.path)
    if request.url.path.startswith("/v1/messages"):
        return _anthropic_error_response(
            status_code=500,
            error_type="api_error",
            message="Internal server error",
        )
    return openai_error_response(
        "Internal server error",
        status_code=500,
        err_type="api_error",
        code=500,
    )


async def validate_json_request(raw_request: Request):
    """Validate that the request content-type is application/json."""
    content_type = raw_request.headers.get("content-type", "").lower()
    media_type = content_type.split(";", maxsplit=1)[0]
    if media_type != "application/json":
        raise RequestValidationError(
            errors=[
                {
                    "loc": ["header", "content-type"],
                    "msg": "Unsupported Media Type: Only 'application/json' is allowed",
                    "type": "value_error",
                }
            ]
        )


##### Native API endpoints #####


def _dead_subprocess_response() -> Optional[Response]:
    """503 naming the dead component, or None when every subprocess is alive.

    #604 established that a dead scheduler must not read as healthy. #485/C40
    established WHEN that has to be checked: always. This helper is therefore
    mode-independent, and the two /health modes both consult it before doing
    anything else -- see ``health_generate``.
    """
    watchdog = getattr(
        _global_state.tokenizer_manager, "_subprocess_watchdog", None
    )
    if watchdog is None:
        return None
    for proc, name in watchdog.processes_with_names():
        # is_alive() calls os.waitpid(pid, WNOHANG) internally which
        # reaps zombies. After the call, proc.exitcode is set if the
        # process has exited. A long-running scheduler/detokenizer that
        # is not alive is always a failure, regardless of exit code.
        if not proc.is_alive():
            from sglang.srt.utils.watchdog import describe_exit

            logger.error(
                "Health check failed: subprocess %s (pid=%d) is dead: %s",
                name,
                proc.pid,
                describe_exit(proc.exitcode),
            )
            return Response(
                status_code=503,
                content=_make_health_error_json(name, proc.pid, proc.exitcode),
                media_type="application/json",
            )
    return None


def _health_fast_path() -> Response:
    """Fast-path liveness check when generation-based health is disabled.

    Returns 503 if any scheduler or detokenizer subprocess is dead, with a
    response body naming the dead component.  This closes the gap in #604
    where the fast path returned 200 unconditionally while scheduler
    subprocesses could be dead/zombie.
    """
    return _dead_subprocess_response() or Response(status_code=200)


def _make_health_error_json(name: str, pid: int, exitcode: int | None) -> str:
    """Build the JSON body for a 503 subprocess-dead health response."""
    detail = {
        "error": "subprocess not alive",
        "component": name,
        "pid": pid,
        "exit_code": exitcode,
    }
    return json.dumps(detail)


@app.get("/health")
@app.get("/health_generate")
async def health_generate(request: Request) -> Response:
    """
    Check the health of the inference server by sending a special request to generate one token.

    If the server is running something, this request will be ignored, so it creates zero overhead.
    If the server is not running anything, this request will be run, so we know whether the server is healthy.
    """

    if _global_state.tokenizer_manager.gracefully_exit:
        logger.info("Health check request received during shutdown. Returning 503.")
        return Response(status_code=503)

    if _global_state.tokenizer_manager.server_status == ServerStatus.Starting:
        return Response(status_code=503)

    # #485/C40: LIVENESS IS A PRECONDITION OF HEALTH, IN EVERY MODE.
    # A dead rank used to be invisible here on a default boot: the check below
    # lived behind the fast path, and the fast path is only taken when
    # SGLANG_ENABLE_HEALTH_ENDPOINT_GENERATION is off, which it is not by
    # default. The generation path then answered 200 as soon as
    # last_receive_tstamp moved, and a still-draining output stream moves it,
    # so an instance printed healthy with a rank gone -- the #622 wedge
    # signature. The subprocess check is cheap (a waitpid(WNOHANG) per handle)
    # and now runs before the mode is consulted.
    dead = _dead_subprocess_response()
    if dead is not None:
        return dead

    if (
        not envs.SGLANG_ENABLE_HEALTH_ENDPOINT_GENERATION.get()
        and request.url.path == "/health"
    ):
        return _health_fast_path()

    sampling_params = {"max_new_tokens": 1, "temperature": 0.0}
    # uuid keeps rids unique across tokenizer workers (a bare time.time() can
    # collide and crash the shared DetokenizerManager decode_status).
    rid = f"{HEALTH_CHECK_RID_PREFIX}_{uuid.uuid4().hex}"

    if _global_state.tokenizer_manager.is_generation:
        gri = GenerateReqInput(
            rid=rid,
            input_ids=[0],
            sampling_params=sampling_params,
            log_metrics=False,
        )
        if (
            _global_state.tokenizer_manager.server_args.disaggregation_mode
            != DisaggregationMode.NULL.value
        ):
            gri.bootstrap_host = FAKE_BOOTSTRAP_HOST
            gri.bootstrap_room = 0
    else:
        gri = EmbeddingReqInput(
            rid=rid, input_ids=[0], sampling_params=sampling_params, log_metrics=False
        )

    async def gen():
        async for _ in _global_state.tokenizer_manager.generate_request(gri, request):
            break

    task = asyncio.create_task(gen())

    # As long as we receive any response from the detokenizer/scheduler, we consider the server is healthy.
    tic = time.time()
    while time.time() < tic + HEALTH_CHECK_TIMEOUT:
        await asyncio.sleep(1)
        if _global_state.tokenizer_manager.last_receive_tstamp > tic:
            task.cancel()
            _global_state.tokenizer_manager.rid_to_state.pop(rid, None)
            _global_state.tokenizer_manager.server_status = ServerStatus.Up
            return Response(status_code=200)

    task.cancel()
    tic_time = time.strftime("%H:%M:%S", time.localtime(tic))
    last_receive_time = time.strftime(
        "%H:%M:%S", time.localtime(_global_state.tokenizer_manager.last_receive_tstamp)
    )
    logger.error(
        f"Health check failed. Server couldn't get a response from detokenizer for last "
        f"{HEALTH_CHECK_TIMEOUT} seconds. tic start time: {tic_time}. "
        f"last_heartbeat time: {last_receive_time}"
    )
    _global_state.tokenizer_manager.rid_to_state.pop(rid, None)
    _global_state.tokenizer_manager.server_status = ServerStatus.UnHealthy
    return Response(status_code=503)


@app.get("/get_model_info")
async def get_model_info():
    """Get the model information (deprecated - use /model_info instead)."""
    logger.warning(
        "Endpoint '/get_model_info' is deprecated and will be removed in a future version. "
        "Please use '/model_info' instead."
    )
    return await model_info()


@app.get("/model_info")
async def model_info():
    """Get the model information."""
    model_config = _global_state.tokenizer_manager.model_config
    result = {
        "model_path": _global_state.tokenizer_manager.model_path,
        "tokenizer_path": _global_state.tokenizer_manager.server_args.tokenizer_path,
        "is_generation": _global_state.tokenizer_manager.is_generation,
        "preferred_sampling_params": _global_state.tokenizer_manager.server_args.preferred_sampling_params,
        "weight_version": _global_state.tokenizer_manager.server_args.weight_version,
        "has_image_understanding": model_config.is_image_understandable_model,
        "has_audio_understanding": model_config.is_audio_understandable_model,
        "model_type": getattr(model_config.hf_config, "model_type", None),
        "architectures": getattr(model_config.hf_config, "architectures", None),
        "weight_version": _global_state.tokenizer_manager.server_args.weight_version,
        # "hf_config": model_config.hf_config.to_dict(),
    }
    return result


@app.get("/get_weight_version")
@app.get("/weight_version")
async def weight_version():
    """Get the current weight version."""
    raise HTTPException(
        status_code=404,
        detail="Endpoint '/get_weight_version' or '/weight_version' is deprecated. Please use '/model_info' instead.",
    )


@app.get("/get_server_info")
async def get_server_info():
    """Get the server information (deprecated - use /server_info instead)."""
    logger.warning(
        "Endpoint '/get_server_info' is deprecated and will be removed in a future version. "
        "Please use '/server_info' instead."
    )
    return await server_info()


@app.get("/server_info")
async def server_info():
    """Get the server information."""
    # Returns internal states per DP.
    internal_states: List[Dict[Any, Any]] = (
        await _global_state.tokenizer_manager.get_internal_state()
    )

    server_args = _global_state.tokenizer_manager.server_args

    # server_args.model_config is not serializable but should be excluded by asdict.
    return msgspec_to_builtins(
        {
            **dataclasses.asdict(server_args),
            **_global_state.scheduler_info,
            "internal_states": internal_states,
            "version": __version__,
            # Structured KV-event publisher descriptor for KV-aware routers.
            # `None` when publishing is disabled or misconfigured; see
            # `ServerArgs.describe_kv_events_publisher` for the precise contract.
            "kv_events": server_args.describe_kv_events_publisher(),
        }
    )


@app.get("/get_load")
async def get_load():
    """Get load metrics (deprecated - use /v1/loads instead).

    Legacy shim backed by /v1/loads. Projects the load snapshot down to the
    historical field shape (dp_rank, num_reqs, num_waiting_reqs, num_tokens,
    num_pending_tokens, ts_tic) so existing clients keep working.
    """
    logger.warning(
        "Endpoint '/get_load' is deprecated and will be removed in a future version. "
        "Please use '/v1/loads' instead."
    )
    load_results = await _global_state.tokenizer_manager.get_loads(include=["core"])
    ts = time.perf_counter()
    return [
        {
            "dp_rank": r.dp_rank,
            "num_reqs": r.num_running_reqs + r.num_waiting_reqs,
            "num_waiting_reqs": r.num_waiting_reqs,
            "num_tokens": r.num_total_tokens,
            "num_pending_tokens": r.num_total_tokens - r.num_used_tokens,
            "ts_tic": ts,
        }
        for r in load_results
    ]


# example usage:
# curl -s -X POST http://localhost:30000/set_internal_state -H "Content-Type: application/json" -d '{"server_args": {"pp_max_micro_batch_size": 8}}'
@app.api_route("/set_internal_state", methods=["POST", "PUT"])
@auth_level(AuthLevel.ADMIN_OPTIONAL)
async def set_internal_state(
    obj: Annotated[SetInternalStateReq, Body()], request: Request
):
    res = await _global_state.tokenizer_manager.set_internal_state(obj)
    return res


# Do not import `dumper.py` to avoid dependency
if os.environ.get("DUMPER_SERVER_PORT") == "reuse":

    @app.api_route("/dumper/{method}", methods=["POST"])
    @auth_level(AuthLevel.ADMIN_OPTIONAL)
    async def _dumper_control_handler(method: str, request: Request):
        body_bytes = await request.body()
        body = await request.json() if body_bytes else {}
        obj = DumperControlReqInput(method=method, body=body)
        results = await _global_state.tokenizer_manager.dumper_control(obj)
        if any(not r.success for r in results):
            errors = [r.error for r in results if not r.success]
            return ORJSONResponse(status_code=400, content={"error": errors})
        return [x for result in results for x in result.response]


# fastapi implicitly converts json in the request to obj (dataclass)
@app.api_route(
    "/generate",
    methods=["POST", "PUT"],
    response_class=SGLangORJSONResponse,
)
async def generate_request(obj: GenerateReqInput, request: Request):
    """Handle a generate request."""
    # Serving demand, for the idle training tenant (#341). The OpenAI routes
    # stamp it in OpenAIServingBase; the native route has to do it here.
    note_serving_activity()
    if envs.SGLANG_ENABLE_REQUEST_HEADER_OVERRIDES.get():
        apply_header_overrides(obj, request.headers)
    if obj.stream:

        async def stream_results() -> AsyncIterator[bytes]:
            try:
                async for out in _global_state.tokenizer_manager.generate_request(
                    obj, request
                ):
                    yield b"data: " + dumps_json(out) + b"\n\n"
            except ValueError as e:
                # A client disconnect also surfaces here. It's a client-side
                # cancellation, not a server error or bad input -- log it and
                # stop (the request was already aborted upstream) instead of
                # emitting a 400.
                if request is not None and await request.is_disconnected():
                    logger.info(f"[http_server] Client disconnected: {e}")
                    return
                out = {
                    "error": {
                        "message": str(e),
                        "type": "invalid_request_error",
                        "code": 400,
                        "retryable": False,
                    }
                }
                logger.error(f"[http_server] Error: {e}")
                yield b"data: " + dumps_json(out) + b"\n\n"
            yield b"data: [DONE]\n\n"

        # #344: the background abort task only runs once the response body
        # ends, which for a client that stopped reading without closing is
        # never. The watchdog is what bounds that case; the background task
        # stays for the ordinary end-of-stream path.
        return guard_generate_stream(
            StreamingResponse(
                stream_results(),
                media_type="text/event-stream",
                background=_global_state.tokenizer_manager.create_abort_task(obj),
            ),
            tokenizer_manager=_global_state.tokenizer_manager,
            obj=obj,
        )
    else:
        try:
            ret = await _global_state.tokenizer_manager.generate_request(
                obj, request
            ).__anext__()
            return orjson_response(ret)
        except ValueError as e:
            logger.error(f"[http_server] Error: {e}")
            return _create_error_response(e)


@app.api_route("/encode", methods=["POST", "PUT"])
async def encode_request(obj: EmbeddingReqInput, request: Request):
    """Handle an embedding request."""
    try:
        ret = await _global_state.tokenizer_manager.generate_request(
            obj, request
        ).__anext__()
        return ret
    except ValueError as e:
        return _create_error_response(e)


@app.api_route("/classify", methods=["POST", "PUT"])
async def classify_request(obj: EmbeddingReqInput, request: Request):
    """Handle a reward model request. Now the arguments and return values are the same as embedding models."""
    try:
        ret = await _global_state.tokenizer_manager.generate_request(
            obj, request
        ).__anext__()
        return ret
    except ValueError as e:
        return _create_error_response(e)


@app.api_route("/phase_flip", methods=["POST"])
@auth_level(AuthLevel.ADMIN_OPTIONAL)
async def phase_flip(obj: Annotated[PhaseFlipReqInput, Body()], request: Request):
    """#631: arm a PP-prefill <-> TP-decode phase flip (obj.direction is
    pp_to_tp or tp_to_pp; requires --enable-phase-flip). The flip commits
    at the next consensus boundary where every rank is quiescent; watch
    the PHASE-FLIP log lines for the DONE record."""
    try:
        ret = await _global_state.tokenizer_manager.phase_flip(obj.direction)
    except Exception as e:
        return _create_error_response(e)
    return ORJSONResponse(
        {"success": ret.success, "message": ret.message},
        status_code=200 if ret.success else HTTPStatus.BAD_REQUEST,
    )


@app.api_route("/kv_reshard", methods=["POST"])
@auth_level(AuthLevel.ADMIN_OPTIONAL)
async def kv_reshard(obj: Annotated[KvReshardReqInput, Body()], request: Request):
    """#297: arm a phase-boundary KV reshard to obj.target_vector (one entry
    per DCP rank, from the set declared via --kv-reshard-vectors). The move
    itself commits at the next consensus boundary where every rank is fully
    idle; watch the KV-RESHARD log lines for the DONE record."""
    try:
        ret = await _global_state.tokenizer_manager.kv_reshard(obj.target_vector)
    except Exception as e:
        return _create_error_response(e)
    return ORJSONResponse(
        {"success": ret.success, "message": ret.message},
        status_code=200 if ret.success else HTTPStatus.BAD_REQUEST,
    )


@app.api_route("/session_handover", methods=["POST"])
@auth_level(AuthLevel.ADMIN_OPTIONAL)
async def session_handover(
    obj: Annotated[SessionHandoverReqInput, Body()], request: Request
):
    """#261: live session handover without server stop. Actions: export
    (source: quiesce + snapshot one session, returns the manifest), commit /
    abort (source: release or roll back the parked prefix), verify_import
    (destination: request-less check that the migrated blobs are present and
    the model identity matches). Both servers keep serving other sessions
    throughout."""
    try:
        ret = await _global_state.tokenizer_manager.session_handover(obj)
    except Exception as e:
        return _create_error_response(e)
    body = {"success": ret.success, "message": ret.message}
    if ret.manifest_json:
        body["manifest"] = json.loads(ret.manifest_json)
    return ORJSONResponse(
        body, status_code=200 if ret.success else HTTPStatus.BAD_REQUEST
    )


@app.api_route("/session/{session_id}/checkpoint", methods=["POST"])
@auth_level(AuthLevel.ADMIN_OPTIONAL)
async def session_checkpoint_create(
    session_id: str,
    obj: Annotated[Optional[SessionCheckpointReqInput], Body()] = None,
    request: Request = None,
):
    """#410: freeze this session -- KV pages AND GDN/Mamba state -- and return
    a checkpoint id it can later be branched or rewound to WITHOUT
    re-prefilling. The snapshot is #261's; the tier it lands on comes from the
    #407 memory-tier registry (VRAM -> RAM -> Disk by age, provenance
    labelled, named refusal when nothing is admissible). Ids are
    content-addressed, so checkpointing an unchanged prefix is idempotent."""
    return await _session_checkpoint_call(
        obj, action="checkpoint", session_id=session_id
    )


@app.api_route("/session/{session_id}/branch", methods=["POST"])
@auth_level(AuthLevel.ADMIN_OPTIONAL)
async def session_checkpoint_branch(
    session_id: str,
    obj: Annotated[Optional[SessionCheckpointReqInput], Body()] = None,
    request: Request = None,
):
    """#410: open a NEW session that continues from ``checkpoint_id``. Nothing
    is copied -- the branch shares the checkpoint's radix nodes and the tree
    splits where it diverges. The response reports the page-sharing
    accounting."""
    return await _session_checkpoint_call(obj, action="branch", session_id=session_id)


@app.api_route("/session/{session_id}/rewind", methods=["POST"])
@auth_level(AuthLevel.ADMIN_OPTIONAL)
async def session_checkpoint_rewind(
    session_id: str,
    obj: Annotated[Optional[SessionCheckpointReqInput], Body()] = None,
    request: Request = None,
):
    """#410: move this session's continuation point back to ``checkpoint_id``.
    Refused while the session has a request in flight. The tokens beyond the
    checkpoint stay in the radix tree as ordinary cached pages."""
    return await _session_checkpoint_call(obj, action="rewind", session_id=session_id)


@app.api_route("/session/{session_id}/checkpoints", methods=["GET", "POST"])
@auth_level(AuthLevel.ADMIN_OPTIONAL)
async def session_checkpoint_list(session_id: str, request: Request = None):
    """#410: the checkpoints taken of this session, oldest first."""
    return await _session_checkpoint_call(None, action="list", session_id=session_id)


async def _session_checkpoint_call(
    obj: Optional[SessionCheckpointReqInput], *, action: str, session_id: str
):
    """One call shape for every #410 route. The path parameter is the session
    id -- a body that disagrees with the URL is refused rather than silently
    preferring one of them."""
    obj = obj if obj is not None else SessionCheckpointReqInput(action=action)
    if obj.session_id and obj.session_id != session_id:
        return ORJSONResponse(
            {
                "success": False,
                "message": (
                    f"body session_id {obj.session_id!r} disagrees with the URL "
                    f"{session_id!r}; refusing to guess which one was meant"
                ),
            },
            status_code=HTTPStatus.BAD_REQUEST,
        )
    obj.action = action
    obj.session_id = session_id
    try:
        ret = await _global_state.tokenizer_manager.session_checkpoint(obj)
    except Exception as e:
        return _create_error_response(e)
    body = {
        "success": ret.success,
        "message": ret.message,
        "checkpoint_id": ret.checkpoint_id,
        "session_id": ret.session_id,
    }
    if ret.info:
        body["info"] = ret.info
    if ret.manifest_json:
        body["manifest"] = json.loads(ret.manifest_json)
    return ORJSONResponse(
        body, status_code=200 if ret.success else HTTPStatus.BAD_REQUEST
    )


@app.api_route("/vram_budget", methods=["POST"])
@auth_level(AuthLevel.ADMIN_OPTIONAL)
async def vram_budget(obj: Annotated[VramBudgetReqInput, Body()], request: Request):
    """#330: dial one card's VRAM budget at runtime (device: rank:N / cuda:N /
    NVML UUID / all; exactly one of budget_mib, release_mib,
    release_fraction), or query the dial state with {"query": true}. Released
    memory is decommitted back to the driver at the next group-idle consensus
    boundary; watch the VRAM-DIAL log lines for the DONE record. Requires
    --enable-vram-dial."""
    try:
        ret = await _global_state.tokenizer_manager.vram_budget(obj)
    except Exception as e:
        return _create_error_response(e)
    return ORJSONResponse(
        {"success": ret.success, "message": ret.message, "state": ret.state},
        status_code=200 if ret.success else HTTPStatus.BAD_REQUEST,
    )


@app.api_route("/flush_cache", methods=["GET", "POST"])
@auth_level(AuthLevel.ADMIN_OPTIONAL)
async def flush_cache(timeout: float = Query(0.0, ge=0.0)):
    """Flush the radix cache."""
    ret = await _global_state.tokenizer_manager.flush_cache(timeout_s=timeout)
    if ret.success:
        content = (
            "Cache flushed.\nPlease check backend logs for more details. "
            "(When there are running or waiting requests, the operation will not be performed.)\n"
        )
    else:
        content = ret.message or "Flush cache failed.\n"
    return Response(
        content=content,
        status_code=200 if ret.success else HTTPStatus.BAD_REQUEST,
    )


@app.post("/add_external_corpus")
@auth_level(AuthLevel.ADMIN_OPTIONAL)
async def add_external_corpus(request: Request):
    """Add an external corpus for ngram speculative decoding."""
    from sglang.srt.managers.io_struct import AddExternalCorpusReqInput

    try:
        obj = AddExternalCorpusReqInput(**(await request.json()))
    except TypeError as e:
        return ORJSONResponse(
            {"success": False, "message": str(e)},
            status_code=HTTPStatus.BAD_REQUEST,
        )
    result = await _global_state.tokenizer_manager.add_external_corpus(obj)
    return ORJSONResponse(
        {
            "success": result.success,
            "corpus_id": result.corpus_id,
            "message": result.message,
            "loaded_token_count": result.loaded_token_count,
        },
        status_code=200 if result.success else HTTPStatus.BAD_REQUEST,
    )


@app.post("/remove_external_corpus")
@auth_level(AuthLevel.ADMIN_OPTIONAL)
async def remove_external_corpus(request: Request):
    """Remove an external corpus by ID."""
    body = await request.json()
    corpus_id = body.get("corpus_id")
    if not corpus_id:
        return ORJSONResponse(
            {"success": False, "message": "corpus_id is required."},
            status_code=HTTPStatus.BAD_REQUEST,
        )
    result = await _global_state.tokenizer_manager.remove_external_corpus(corpus_id)
    return ORJSONResponse(
        {"success": result.success, "message": result.message},
        status_code=200 if result.success else HTTPStatus.BAD_REQUEST,
    )


@app.get("/list_external_corpora")
@auth_level(AuthLevel.ADMIN_OPTIONAL)
async def list_external_corpora():
    """List all active external corpora."""
    result = await _global_state.tokenizer_manager.list_external_corpora()
    return ORJSONResponse(
        {
            "success": result.success,
            "corpus_token_counts": result.corpus_token_counts,
            "message": result.message,
        },
        status_code=200 if result.success else HTTPStatus.BAD_REQUEST,
    )


@app.api_route("/clear_hicache_storage_backend", methods=["GET", "POST"])
@auth_level(AuthLevel.ADMIN_OPTIONAL)
async def clear_hicache_storage_backend_deprecated():
    """Deprecated: use POST /hicache/storage-backend/clear."""
    ret = await _global_state.tokenizer_manager.clear_hicache_storage()
    return Response(
        content=(
            "Deprecated endpoint. Use POST /hicache/storage-backend/clear.\n"
            "Hierarchical cache storage backend cleared.\n"
        ),
        status_code=200 if ret.success else HTTPStatus.BAD_REQUEST,
    )


# example usage:
# curl -s -X POST http://127.0.0.1:30000/clear_hicache_storage_backend
@app.api_route("/hicache/storage-backend/clear", methods=["POST"])
@auth_level(AuthLevel.ADMIN_OPTIONAL)
async def clear_hicache_storage_backend():
    """Clear the hierarchical cache storage backend."""
    ret = await _global_state.tokenizer_manager.clear_hicache_storage()
    return Response(
        content="Hierarchical cache storage backend cleared.\n",
        status_code=200 if ret.success else HTTPStatus.BAD_REQUEST,
    )


# example usage:
# curl -s -X PUT http://127.0.0.1:30000/hicache/storage-backend \
#  -H 'Content-Type: application/json' \
#   -d '{
#     "hicache_storage_backend": "file",
#     "hicache_storage_backend_extra_config_json": "{}",
#     "hicache_storage_prefetch_policy": "timeout",
#     "hicache_write_policy": "write_through"
#   }'
@app.api_route("/hicache/storage-backend", methods=["PUT"])
@auth_level(AuthLevel.ADMIN_OPTIONAL)
async def attach_hicache_storage_backend(
    obj: Annotated[AttachHiCacheStorageReqInput, Body()],
):
    """Attach (enable) HiCache storage backend at runtime.

    Only allowed when there are NO running / queued requests.
    """
    if not _global_state.tokenizer_manager.server_args.admin_api_key:
        return _admin_api_key_missing_response()

    ret = await _global_state.tokenizer_manager.attach_hicache_storage(
        hicache_storage_backend=obj.hicache_storage_backend,
        hicache_storage_backend_extra_config_json=obj.hicache_storage_backend_extra_config_json,
        hicache_storage_prefetch_policy=obj.hicache_storage_prefetch_policy,
        hicache_write_policy=obj.hicache_write_policy,
    )
    msg = ret.message
    return Response(
        content=(
            (
                "HiCache storage backend attached.\n"
                if ret.success
                else "Failed to attach HiCache storage backend.\n"
            )
            + (msg + "\n" if msg else "")
        ),
        status_code=200 if ret.success else HTTPStatus.BAD_REQUEST,
    )


# example usage:
# curl -s -X DELETE http://127.0.0.1:30000/hicache/storage-backend
@app.api_route("/hicache/storage-backend", methods=["DELETE"])
@auth_level(AuthLevel.ADMIN_OPTIONAL)
async def detach_hicache_storage_backend():
    """Detach (disable) HiCache storage backend at runtime.

    Only allowed when there are NO running / queued requests.
    """
    if not _global_state.tokenizer_manager.server_args.admin_api_key:
        return _admin_api_key_missing_response()

    ret = await _global_state.tokenizer_manager.detach_hicache_storage()
    msg = ret.message
    return Response(
        content=(
            (
                "HiCache storage backend detached.\n"
                if ret.success
                else "Failed to detach HiCache storage backend.\n"
            )
            + (msg + "\n" if msg else "")
        ),
        status_code=200 if ret.success else HTTPStatus.BAD_REQUEST,
    )


# example usage:
# curl -s http://127.0.0.1:30000/hicache/storage-backend
@app.get("/hicache/storage-backend")
@auth_level(AuthLevel.ADMIN_OPTIONAL)
async def hicache_storage_backend_status():
    """Get current HiCache storage backend status (tokenizer-side view)."""
    if not _global_state.tokenizer_manager.server_args.admin_api_key:
        return _admin_api_key_missing_response()

    return {
        "hicache_storage_backend": _global_state.tokenizer_manager.server_args.hicache_storage_backend,
        "hicache_storage_backend_extra_config": _global_state.tokenizer_manager.server_args.hicache_storage_backend_extra_config,
        "hicache_storage_prefetch_policy": _global_state.tokenizer_manager.server_args.hicache_storage_prefetch_policy,
        "hicache_write_policy": _global_state.tokenizer_manager.server_args.hicache_write_policy,
    }


# example usage:
# curl -s -X POST http://127.0.0.1:30000/hicache/storage-backend/resize \
#  -H 'Content-Type: application/json' \
#   -d '{"max_size_gb": 100, "min_free_gb": 20}'
@app.api_route("/hicache/storage-backend/resize", methods=["POST"])
@auth_level(AuthLevel.ADMIN_OPTIONAL)
async def resize_hicache_storage_backend(
    obj: Annotated[ResizeHiCacheStorageReqInput, Body()],
):
    """Re-cap the attached HiCache storage backend without detaching it.

    Growing takes effect immediately. Shrinking evicts LRU entries inline and
    only returns once usage is back under the new cap, so a large shrink
    delays the next batch for the duration of the unlinks. Unlike attach and
    detach this does not require an idle scheduler.
    """
    if not _global_state.tokenizer_manager.server_args.admin_api_key:
        return _admin_api_key_missing_response()

    ret = await _global_state.tokenizer_manager.resize_hicache_storage(
        max_size_gb=obj.max_size_gb,
        min_free_gb=obj.min_free_gb,
    )
    return ORJSONResponse(
        {
            "success": ret.success,
            "message": ret.message,
            "stats": ret.stats,
        },
        status_code=200 if ret.success else HTTPStatus.BAD_REQUEST,
    )


@app.api_route("/start_profile", methods=["GET", "POST"])
@auth_level(AuthLevel.ADMIN_OPTIONAL)
async def start_profile_async(obj: Annotated[Optional[ProfileReq], Body()] = None):
    """Start profiling."""
    await _global_state.tokenizer_manager.start_profile(obj or ProfileReq())
    return Response(
        content="Start profiling.\n",
        status_code=200,
    )


@app.api_route("/stop_profile", methods=["GET", "POST"])
@auth_level(AuthLevel.ADMIN_OPTIONAL)
async def stop_profile_async():
    """Stop profiling."""
    await _global_state.tokenizer_manager.stop_profile()
    return Response(
        content="Stop profiling. This will take some time.\n",
        status_code=200,
    )


@app.api_route("/set_trace_level", methods=["GET", "POST"])
def set_trace_level(level: int = Query(..., ge=0)):
    set_global_trace_level(level)

    return Response(
        content="success",
        status_code=200,
    )


@app.api_route("/freeze_gc", methods=["GET", "POST"])
@auth_level(AuthLevel.ADMIN_OPTIONAL)
async def freeze_gc_async():
    """
    See engine.freeze_gc for more details.
    """
    await _global_state.tokenizer_manager.freeze_gc()
    return Response(
        content="Garbage collection frozen.\n",
        status_code=200,
    )


@app.api_route("/start_expert_distribution_record", methods=["GET", "POST"])
@auth_level(AuthLevel.ADMIN_OPTIONAL)
async def start_expert_distribution_record_async():
    """Start recording the expert distribution. Clear the previous record if any."""
    await _global_state.tokenizer_manager.start_expert_distribution_record()
    return Response(
        content="Start recording the expert distribution.\n",
        status_code=200,
    )


@app.api_route("/stop_expert_distribution_record", methods=["GET", "POST"])
@auth_level(AuthLevel.ADMIN_OPTIONAL)
async def stop_expert_distribution_record_async():
    """Stop recording the expert distribution."""
    await _global_state.tokenizer_manager.stop_expert_distribution_record()
    return Response(
        content="Stop recording the expert distribution.\n",
        status_code=200,
    )


@app.api_route("/dump_expert_distribution_record", methods=["GET", "POST"])
@auth_level(AuthLevel.ADMIN_OPTIONAL)
async def dump_expert_distribution_record_async():
    """Dump expert distribution record."""
    await _global_state.tokenizer_manager.dump_expert_distribution_record()
    return Response(
        content="Dump expert distribution record.\n",
        status_code=200,
    )


@app.post("/update_weights_from_disk")
@auth_level(AuthLevel.ADMIN_OPTIONAL)
async def update_weights_from_disk(
    obj: Annotated[UpdateWeightFromDiskReqInput, Body()], request: Request
):
    """Update the weights from disk inplace without re-launching the server."""
    (
        success,
        message,
        num_paused_requests,
    ) = await _global_state.tokenizer_manager.update_weights_from_disk(obj, request)

    content = {
        "success": success,
        "message": message,
        "num_paused_requests": num_paused_requests,
    }
    if success:
        return ORJSONResponse(
            content,
            status_code=HTTPStatus.OK,
        )
    else:
        return ORJSONResponse(
            content,
            status_code=HTTPStatus.BAD_REQUEST,
        )


@app.post("/init_weights_send_group_for_remote_instance")
@auth_level(AuthLevel.ADMIN_OPTIONAL)
async def init_weights_send_group_for_remote_instance(
    obj: Annotated[InitWeightsSendGroupForRemoteInstanceReqInput, Body()],
    request: Request,
):
    (
        success,
        message,
    ) = await _global_state.tokenizer_manager.init_weights_send_group_for_remote_instance(
        obj, request
    )
    content = {"success": success, "message": message}
    if success:
        return ORJSONResponse(content, status_code=200)
    else:
        return ORJSONResponse(content, status_code=HTTPStatus.BAD_REQUEST)


@app.post("/send_weights_to_remote_instance")
@auth_level(AuthLevel.ADMIN_OPTIONAL)
async def send_weights_to_remote_instance(
    obj: Annotated[SendWeightsToRemoteInstanceReqInput, Body()], request: Request
):
    (
        success,
        message,
    ) = await _global_state.tokenizer_manager.send_weights_to_remote_instance(
        obj, request
    )
    content = {"success": success, "message": message}
    if success:
        return ORJSONResponse(content, status_code=200)
    else:
        return ORJSONResponse(content, status_code=HTTPStatus.BAD_REQUEST)


@app.get("/get_remote_instance_transfer_engine_info")
@auth_level(AuthLevel.ADMIN_OPTIONAL)
async def get_remote_instance_transfer_engine_info(rank: int = None):
    """Get the server information (deprecated - use /remote_instance_transfer_engine_info instead)."""
    logger.warning(
        "Endpoint '/get_remote_instance_transfer_engine_info' is deprecated and will be removed in a future version. "
        "Please use '/remote_instance_transfer_engine_info' instead."
    )
    return await remote_instance_transfer_engine_info(rank=rank)


@app.get("/remote_instance_transfer_engine_info")
@auth_level(AuthLevel.ADMIN_OPTIONAL)
async def remote_instance_transfer_engine_info(rank: int = None):
    if rank is None or rank < 0:
        return ORJSONResponse(
            {"error": {"message": "Missing or invalid rank parameter"}},
            status_code=HTTPStatus.BAD_REQUEST,
        )

    server_args = _global_state.tokenizer_manager.server_args
    try:
        resp = requests.get(
            f"{server_args.engine_info_bootstrap_url}/get_transfer_engine_info",
            params={"rank": rank},
            timeout=5,
        )
        if resp.status_code == 200:
            return resp.json()
    except (requests.exceptions.RequestException, ValueError) as e:
        logger.warning(f"Failed to get transfer engine info for rank {rank}: {e}")

    return ORJSONResponse(
        {"error": {"message": f"Failed to get transfer engine info for rank {rank}"}},
        status_code=HTTPStatus.BAD_REQUEST,
    )


@app.post("/init_weights_update_group")
@auth_level(AuthLevel.ADMIN_OPTIONAL)
async def init_weights_update_group(
    obj: Annotated[InitWeightsUpdateGroupReqInput, Body()], request: Request
):
    """Initialize the parameter update group."""
    success, message = await _global_state.tokenizer_manager.init_weights_update_group(
        obj, request
    )
    content = {"success": success, "message": message}
    if success:
        return ORJSONResponse(content, status_code=200)
    else:
        return ORJSONResponse(content, status_code=HTTPStatus.BAD_REQUEST)


@app.post("/destroy_weights_update_group")
@auth_level(AuthLevel.ADMIN_OPTIONAL)
async def destroy_weights_update_group(
    obj: Annotated[DestroyWeightsUpdateGroupReqInput, Body()], request: Request
):
    """Destroy the parameter update group."""
    (
        success,
        message,
    ) = await _global_state.tokenizer_manager.destroy_weights_update_group(obj, request)
    content = {"success": success, "message": message}
    return ORJSONResponse(
        content, status_code=200 if success else HTTPStatus.BAD_REQUEST
    )


@app.post("/update_weights_from_tensor")
@auth_level(AuthLevel.ADMIN_OPTIONAL)
async def update_weights_from_tensor(
    obj: Annotated[UpdateWeightsFromTensorReqInput, Body()], request: Request
):
    """Update the weights from tensor inplace without re-launching the server.
    Notes:
    1. Ensure that the model is on the correct device (e.g., GPU) before calling this endpoint. If the model is moved to the CPU unexpectedly, it may cause performance issues or runtime errors.
    2. HTTP will transmit only the metadata of the tensor, while the tensor itself will be directly copied to the model.
    3. Any binary data in the named tensors should be base64 encoded.
    """

    success, message = await _global_state.tokenizer_manager.update_weights_from_tensor(
        obj, request
    )

    content = {"success": success, "message": message}
    return ORJSONResponse(
        content, status_code=200 if success else HTTPStatus.BAD_REQUEST
    )


@app.post("/update_weights_from_distributed")
@auth_level(AuthLevel.ADMIN_OPTIONAL)
async def update_weights_from_distributed(
    obj: Annotated[UpdateWeightsFromDistributedReqInput, Body()], request: Request
):
    """Update model parameter from distributed online."""
    (
        success,
        message,
    ) = await _global_state.tokenizer_manager.update_weights_from_distributed(
        obj, request
    )

    content = {"success": success, "message": message}
    if success:
        return ORJSONResponse(content, status_code=200)
    else:
        return ORJSONResponse(content, status_code=HTTPStatus.BAD_REQUEST)


@app.post("/update_weights_from_ipc")
@auth_level(AuthLevel.ADMIN_OPTIONAL)
async def update_weights_from_ipc(
    obj: Annotated[UpdateWeightsFromIPCReqInput, Body()], request: Request
):
    """Update the weights from IPC (Inter-Process Communication) for checkpoint-engine integration."""
    success, message = await _global_state.tokenizer_manager.update_weights_from_ipc(
        obj, request
    )

    content = {"success": success, "message": message}
    if success:
        if _global_state.tokenizer_manager.initial_weights_loaded is False:
            _global_state.tokenizer_manager.initial_weights_loaded = True
        return ORJSONResponse(content)
    else:
        return ORJSONResponse(content, status_code=HTTPStatus.BAD_REQUEST)


@app.post("/update_weight_version")
@auth_level(AuthLevel.ADMIN_OPTIONAL)
async def update_weight_version(
    obj: Annotated[UpdateWeightVersionReqInput, Body()], request: Request
):
    """Update the weight version. This operation requires no active requests."""
    if obj.abort_all_requests:
        _global_state.tokenizer_manager.abort_request(abort_all=True)

    # Use a simple approach without the complex lock mechanism for now
    # since weight_version update is a simple operation that doesn't affect model weights
    try:
        # Update the weight version in server args (the single source of truth)
        _global_state.tokenizer_manager.server_args.override(
            "http.update_weight_version", weight_version=obj.new_version
        )

        return ORJSONResponse(
            {
                "success": True,
                "message": f"Weight version updated to {obj.new_version}",
                "new_version": obj.new_version,
            },
            status_code=HTTPStatus.OK,
        )
    except Exception as e:
        return ORJSONResponse(
            {
                "success": False,
                "message": f"Failed to update weight version: {str(e)}",
            },
            status_code=HTTPStatus.BAD_REQUEST,
        )


@app.api_route("/get_weights_by_name", methods=["GET", "POST"])
@auth_level(AuthLevel.ADMIN_OPTIONAL)
async def get_weights_by_name(
    obj: Annotated[GetWeightsByNameReqInput, Body()], request: Request
):
    """Get model parameter by name."""
    try:
        ret = await _global_state.tokenizer_manager.get_weights_by_name(obj, request)
        if ret is None:
            return _create_error_response("Get parameter by name failed")
        else:
            return ORJSONResponse(ret, status_code=200)
    except Exception as e:
        return _create_error_response(e)


@app.api_route("/release_memory_occupation", methods=["GET", "POST"])
@auth_level(AuthLevel.ADMIN_OPTIONAL)
async def release_memory_occupation(
    obj: Annotated[ReleaseMemoryOccupationReqInput, Body()], request: Request
):
    """Release GPU memory occupation temporarily."""
    try:
        await _global_state.tokenizer_manager.release_memory_occupation(obj, request)
    except Exception as e:
        return _create_error_response(e)


@app.api_route("/resume_memory_occupation", methods=["GET", "POST"])
@auth_level(AuthLevel.ADMIN_OPTIONAL)
async def resume_memory_occupation(
    obj: Annotated[ResumeMemoryOccupationReqInput, Body()], request: Request
):
    """Resume GPU memory occupation."""
    try:
        await _global_state.tokenizer_manager.resume_memory_occupation(obj, request)
    except Exception as e:
        return _create_error_response(e)


@app.api_route("/hibernate", methods=["POST"])
@auth_level(AuthLevel.ADMIN_OPTIONAL)
async def hibernate(
    obj: Annotated[Optional[ReleaseMemoryOccupationReqInput], Body()] = None,
    request: Request = None,
):
    """#89 hibernate (suspend-to-disk): park each rank's FINAL post-transform
    weights to --hibernate-dir, then release. A later boot with a matching
    manifest restores them fast (LoadFormat.HIBERNATE). Requires the server to
    be fully idle (mirrors /release_memory_occupation).

    POST only (#510): parking the model is a state change, and the route used
    to accept GET, so a bare browser navigation or a link preview fetch parked
    the server. A ``hibernate_dir`` in the body is confined to the configured
    --hibernate-dir; see the confinement note below."""
    try:
        if obj is None:
            obj = ReleaseMemoryOccupationReqInput()
        # #510 (audit #506 A2-F1): a request-supplied hibernate_dir used to
        # override --hibernate-dir outright and reach os.makedirs(). Confine
        # it here so the caller gets a 400 with the reason, and again at the
        # scheduler-side sink (weight_updater._hibernate_park_weights) so the
        # Engine API and any future caller inherit the same rule.
        requested_dir = getattr(obj, "hibernate_dir", None)
        if requested_dir is not None:
            from sglang.srt.utils.path_confinement import (
                PathConfinementError,
                confine_to_root,
            )

            try:
                obj.hibernate_dir = confine_to_root(
                    requested_dir,
                    _global_state.tokenizer_manager.server_args.hibernate_dir,
                )
            except PathConfinementError as exc:
                return ORJSONResponse(
                    {"error": {"message": str(exc)}},
                    status_code=HTTPStatus.BAD_REQUEST,
                )
        obj.destination = "disk"
        obj.tags = ["weights"]
        await _global_state.tokenizer_manager.release_memory_occupation(obj, request)
        return ORJSONResponse({"message": "hibernate: weights parked to disk."})
    except Exception as e:
        return _create_error_response(e)


@app.api_route("/weights_checker", methods=["GET", "POST"])
@auth_level(AuthLevel.ADMIN_OPTIONAL)
async def check_weights(
    obj: Annotated[Optional[CheckWeightsReqInput], Body()] = None,
    request: Request = None,
):
    if obj is None:
        obj = CheckWeightsReqInput()
    success, message, ranks, per_engine_checksum = (
        await _global_state.tokenizer_manager.check_weights(obj, request)
    )
    body = {"success": success, "message": message}
    if ranks is not None:
        body["ranks"] = ranks
    if per_engine_checksum is not None:
        body["per_engine_checksum"] = per_engine_checksum
    return ORJSONResponse(body, status_code=200 if success else HTTPStatus.BAD_REQUEST)


@app.api_route("/slow_down", methods=["GET", "POST"])
@auth_level(AuthLevel.ADMIN_OPTIONAL)
async def slow_down(obj: Annotated[SlowDownReqInput, Body()], request: Request):
    """Slow down the system deliberately. Only for testing. Example scenario:
    when we want to test performance of D in large-scale PD disaggregation and have no enough nodes for P,
    we can use this to slow down D to let it have enough running sequences, and then disable slowdown
    to let it run in full batch size.
    """
    try:
        await _global_state.tokenizer_manager.slow_down(obj, request)
    except Exception as e:
        return _create_error_response(e)


@app.api_route("/load_lora_adapter", methods=["POST"])
@auth_level(AuthLevel.ADMIN_OPTIONAL)
async def load_lora_adapter(
    obj: Annotated[LoadLoRAAdapterReqInput, Body()], request: Request
):
    """Load a new LoRA adapter without re-launching the server."""
    result = await _global_state.tokenizer_manager.load_lora_adapter(obj, request)
    status_code = HTTPStatus.OK if result.success else HTTPStatus.BAD_REQUEST
    return ORJSONResponse(msgspec_to_builtins(result), status_code=status_code)


@app.api_route("/load_lora_adapter_from_tensors", methods=["POST"])
async def load_lora_adapter_from_tensors(
    obj: Annotated[LoadLoRAAdapterFromTensorsReqInput, Body()], request: Request
):
    """Load a new LoRA adapter from tensors without re-launching the server."""
    result = await _global_state.tokenizer_manager.load_lora_adapter_from_tensors(
        obj, request
    )
    status_code = HTTPStatus.OK if result.success else HTTPStatus.BAD_REQUEST
    return ORJSONResponse(msgspec_to_builtins(result), status_code=status_code)


@app.api_route("/unload_lora_adapter", methods=["POST"])
@auth_level(AuthLevel.ADMIN_OPTIONAL)
async def unload_lora_adapter(
    obj: Annotated[UnloadLoRAAdapterReqInput, Body()], request: Request
):
    """Load a new LoRA adapter without re-launching the server."""
    result = await _global_state.tokenizer_manager.unload_lora_adapter(obj, request)
    status_code = HTTPStatus.OK if result.success else HTTPStatus.BAD_REQUEST
    return ORJSONResponse(msgspec_to_builtins(result), status_code=status_code)


@app.api_route("/open_session", methods=["GET", "POST"])
async def open_session(obj: Annotated[OpenSessionReqInput, Body()], request: Request):
    """Open a session, and return its unique session id."""
    try:
        session_id = await _global_state.tokenizer_manager.open_session(obj, request)
        if session_id is None:
            raise Exception(
                "Failed to open the session. Check if a session with the same id is still open."
            )
        return session_id
    except Exception as e:
        return _create_error_response(e)


@app.api_route("/close_session", methods=["GET", "POST"])
async def close_session(obj: Annotated[CloseSessionReqInput, Body()], request: Request):
    """Close the session."""
    try:
        await _global_state.tokenizer_manager.close_session(obj, request)
        return Response(status_code=200)
    except Exception as e:
        return _create_error_response(e)


@app.api_route("/configure_logging", methods=["GET", "POST"])
@auth_level(AuthLevel.ADMIN_OPTIONAL)
async def configure_logging(
    obj: Annotated[ConfigureLoggingReq, Body()], request: Request
):
    """Configure the request logging options."""
    _global_state.tokenizer_manager.configure_logging(obj)
    return Response(status_code=200)


@app.post("/abort_request")
@auth_level(AuthLevel.ADMIN_OPTIONAL)
async def abort_request(obj: Annotated[AbortReq, Body()], request: Request):
    """Abort a request."""
    try:
        _global_state.tokenizer_manager.abort_request(
            rid=obj.rid, abort_all=obj.abort_all
        )
        return Response(status_code=200)
    except Exception as e:
        return _create_error_response(e)


@app.post("/parse_function_call")
async def parse_function_call_request(
    obj: Annotated[ParseFunctionCallReq, Body()], request: Request
):
    """
    A native API endpoint to parse function calls from a text.
    """
    # 1) Initialize the parser based on the request body
    parser = FunctionCallParser(
        tools=obj.tools,
        tool_call_parser=obj.tool_call_parser,
        tokenizer=get_global_state().tokenizer_manager.tokenizer,
    )

    # 2) Call the non-stream parsing method (non-stream)
    normal_text, calls = parser.parse_non_stream(obj.text)

    # 3) Organize the response content
    response_data = {
        "normal_text": normal_text,
        "calls": [
            call.model_dump() for call in calls
        ],  # Convert pydantic objects to dictionaries
    }

    return ORJSONResponse(content=response_data, status_code=200)


@app.post("/separate_reasoning")
async def separate_reasoning_request(
    obj: Annotated[SeparateReasoningReqInput, Body()], request: Request
):
    """
    A native API endpoint to separate reasoning from a text.
    """
    # 1) Initialize the parser based on the request body
    parser = ReasoningParser(
        model_type=obj.reasoning_parser,
        request=request,
        tokenizer=get_global_state().tokenizer_manager.tokenizer,
    )

    # 2) Call the non-stream parsing method (non-stream)
    if obj.return_blocks:
        blocks = parser.parse_non_stream_blocks(obj.text)
        reasoning_blocks = [b["text"] for b in blocks if b["type"] == "reasoning"]
        text_blocks = [b["text"] for b in blocks if b["type"] == "text"]
        reasoning_text = "".join(reasoning_blocks)
        normal_text = "".join(text_blocks)
    else:
        reasoning_text, normal_text = parser.parse_non_stream(obj.text)

    # 3) Organize the response content
    response_data = {
        "reasoning_text": reasoning_text,
        "text": normal_text,
    }
    if obj.return_blocks:
        response_data["reasoning_blocks"] = reasoning_blocks
        response_data["text_blocks"] = text_blocks
        response_data["blocks"] = blocks

    return ORJSONResponse(content=response_data, status_code=200)


@app.post("/pause_generation")
@auth_level(AuthLevel.ADMIN_OPTIONAL)
async def pause_generation(
    obj: Annotated[PauseGenerationReqInput, Body()], request: Request
):
    """Pause generation."""
    await _global_state.tokenizer_manager.pause_generation(obj)
    return ORJSONResponse(
        content={"message": "Generation paused successfully.", "status": "ok"},
        status_code=200,
    )


@app.post("/continue_generation")
@auth_level(AuthLevel.ADMIN_OPTIONAL)
async def continue_generation(
    obj: Annotated[ContinueGenerationReqInput, Body()], request: Request
):
    """Continue generation."""
    await _global_state.tokenizer_manager.continue_generation(obj)
    return ORJSONResponse(
        content={"message": "Generation continued successfully.", "status": "ok"},
        status_code=200,
    )


##### OpenAI-compatible API endpoints #####


@app.post("/v1/completions", dependencies=[Depends(validate_json_request)])
async def openai_v1_completions(request: CompletionRequest, raw_request: Request):
    """OpenAI-compatible text completion endpoint."""
    return await raw_request.app.state.openai_serving_completion.handle_request(
        request, raw_request
    )


@app.post("/v1/chat/completions", dependencies=[Depends(validate_json_request)])
async def openai_v1_chat_completions(
    request: ChatCompletionRequest, raw_request: Request
):
    """OpenAI-compatible chat completion endpoint."""
    return await raw_request.app.state.openai_serving_chat.handle_request(
        request, raw_request
    )


@app.post(
    "/v1/embeddings",
    response_class=ORJSONResponse,
    dependencies=[Depends(validate_json_request)],
)
async def openai_v1_embeddings(request: EmbeddingRequest, raw_request: Request):
    """OpenAI-compatible embeddings endpoint."""
    return await raw_request.app.state.openai_serving_embedding.handle_request(
        request, raw_request
    )


@app.post(
    "/v1/classify",
    response_class=ORJSONResponse,
    dependencies=[Depends(validate_json_request)],
)
async def openai_v1_classify(request: ClassifyRequest, raw_request: Request):
    """OpenAI-compatible classification endpoint."""
    return await raw_request.app.state.openai_serving_classify.handle_request(
        request, raw_request
    )


@app.post(
    "/v1/tokenize",
    response_class=ORJSONResponse,
    dependencies=[Depends(validate_json_request)],
)
@app.post(
    "/tokenize",
    response_class=ORJSONResponse,
    dependencies=[Depends(validate_json_request)],
    include_in_schema=False,
)
async def openai_v1_tokenize(request: TokenizeRequest, raw_request: Request):
    """OpenAI-compatible tokenization endpoint."""
    return await raw_request.app.state.openai_serving_tokenize.handle_request(
        request, raw_request
    )


@app.post(
    "/v1/detokenize",
    response_class=ORJSONResponse,
    dependencies=[Depends(validate_json_request)],
)
@app.post(
    "/detokenize",
    response_class=ORJSONResponse,
    dependencies=[Depends(validate_json_request)],
    include_in_schema=False,
)
async def openai_v1_detokenize(request: DetokenizeRequest, raw_request: Request):
    """OpenAI-compatible detokenization endpoint."""
    return await raw_request.app.state.openai_serving_detokenize.handle_request(
        request, raw_request
    )


@app.post("/v1/images/generations", dependencies=[Depends(validate_json_request)])
async def openai_v1_images_generations(raw_request: Request):
    """OpenAI-compatible image generation. Routed to the diffusion lane."""
    body = await raw_request.json()
    return await raw_request.app.state.openai_serving_images.generations(
        body, raw_request
    )


@app.post("/v1/images/edits")
async def openai_v1_images_edits(
    raw_request: Request,
    image: UploadFile = File(...),
    prompt: str = Form(...),
    mask: Optional[UploadFile] = File(default=None),
    model: Optional[str] = Form(default=None),
    n: Optional[int] = Form(default=1),
    size: Optional[str] = Form(default=None),
    response_format: Optional[str] = Form(default=None),
    user: Optional[str] = Form(default=None),
):
    """OpenAI-compatible image edit. Routed to the diffusion lane."""
    files = {
        "image": (image.filename or "image.png", await image.read(), image.content_type)
    }
    if mask is not None:
        files["mask"] = (
            mask.filename or "mask.png",
            await mask.read(),
            mask.content_type,
        )
    form = {
        "prompt": prompt,
        "model": model,
        "n": n,
        "size": size,
        "response_format": response_format,
        "user": user,
    }
    return await raw_request.app.state.openai_serving_images.edits(
        form, files, model, raw_request
    )


@app.post("/v1/images/variations")
async def openai_v1_images_variations(
    raw_request: Request,
    image: UploadFile = File(...),
    model: Optional[str] = Form(default=None),
):
    """OpenAI-compatible image variations. No lane implements this."""
    return await raw_request.app.state.openai_serving_images.variations(model)


@app.post("/v1/audio/speech", dependencies=[Depends(validate_json_request)])
async def openai_v1_audio_speech(raw_request: Request):
    """OpenAI-compatible text-to-speech. Routed to a speech lane when one exists."""
    body = await raw_request.json()
    return await raw_request.app.state.openai_serving_speech.create_speech(
        body, raw_request
    )


##### Files and fine-tuning (#341-M1) #####


@app.post("/v1/files")
@auth_level(AuthLevel.ADMIN_OPTIONAL)  # #510: state-changing fork route
async def openai_v1_files(
    raw_request: Request,
    file: UploadFile = File(...),
    purpose: str = Form(default="fine-tune"),
):
    """OpenAI-compatible file upload. Training data for the fine-tuning API."""
    return await raw_request.app.state.openai_serving_files.create(
        filename=file.filename or "upload.jsonl",
        content=await file.read(),
        purpose=purpose,
    )


@app.get("/v1/files", response_class=ORJSONResponse)
async def openai_v1_files_list(raw_request: Request, purpose: Optional[str] = None):
    return await raw_request.app.state.openai_serving_files.list(purpose=purpose)


@app.get("/v1/files/{file_id}", response_class=ORJSONResponse)
async def openai_v1_files_retrieve(file_id: str, raw_request: Request):
    return await raw_request.app.state.openai_serving_files.retrieve(file_id)


@app.delete("/v1/files/{file_id}", response_class=ORJSONResponse)
@auth_level(AuthLevel.ADMIN_OPTIONAL)  # #510: state-changing fork route
async def openai_v1_files_delete(file_id: str, raw_request: Request):
    return await raw_request.app.state.openai_serving_files.delete(file_id)


@app.get("/v1/files/{file_id}/content")
async def openai_v1_files_content(file_id: str, raw_request: Request):
    return await raw_request.app.state.openai_serving_files.content(file_id)


@app.post("/v1/fine_tuning/jobs", dependencies=[Depends(validate_json_request)])
@auth_level(AuthLevel.ADMIN_OPTIONAL)  # #510: state-changing fork route
async def openai_v1_fine_tuning_jobs(raw_request: Request):
    """OpenAI-compatible fine-tuning job submission. Run by the idle tenant."""
    body = await raw_request.json()
    return await raw_request.app.state.openai_serving_fine_tuning.create(body)


@app.get("/v1/fine_tuning/jobs", response_class=ORJSONResponse)
async def openai_v1_fine_tuning_jobs_list(
    raw_request: Request, after: Optional[str] = None, limit: int = 20
):
    return await raw_request.app.state.openai_serving_fine_tuning.list(
        after=after, limit=limit
    )


@app.get("/v1/fine_tuning/jobs/{job_id}", response_class=ORJSONResponse)
async def openai_v1_fine_tuning_job(job_id: str, raw_request: Request):
    return await raw_request.app.state.openai_serving_fine_tuning.retrieve(job_id)


@app.post("/v1/fine_tuning/jobs/{job_id}/cancel", response_class=ORJSONResponse)
@auth_level(AuthLevel.ADMIN_OPTIONAL)  # #510: state-changing fork route
async def openai_v1_fine_tuning_job_cancel(job_id: str, raw_request: Request):
    return await raw_request.app.state.openai_serving_fine_tuning.cancel(job_id)


@app.get("/v1/fine_tuning/jobs/{job_id}/events")
async def openai_v1_fine_tuning_job_events(
    job_id: str,
    raw_request: Request,
    after: Optional[str] = None,
    limit: int = 20,
    stream: bool = False,
):
    """The job's event log. ``stream=true`` turns it into a live SSE tap."""
    return await raw_request.app.state.openai_serving_fine_tuning.events(
        job_id, after=after, limit=limit, stream=stream
    )


@app.get("/v1/fine_tuning/jobs/{job_id}/checkpoints", response_class=ORJSONResponse)
async def openai_v1_fine_tuning_job_checkpoints(
    job_id: str, raw_request: Request, after: Optional[str] = None, limit: int = 10
):
    return await raw_request.app.state.openai_serving_fine_tuning.checkpoints(
        job_id, after=after, limit=limit
    )


@app.get("/v1/fine_tuning/tenant", response_class=ORJSONResponse)
async def training_tenant_state(raw_request: Request):
    """The fork's own view: idle verdict, machine, backend probes.

    Namespaced under the fine-tuning prefix but not part of the OpenAI
    protocol -- it answers the questions the protocol has no field for, which
    is what the frontend and an operator both need.
    """
    return ORJSONResponse(raw_request.app.state.training_service.snapshot())


# ---------------------------------------------------------------------------
# The idle workbench (#347). Namespaced under x-htsglang because none of it is
# in anybody's standard protocol; the routes exist whether or not the feature
# is enabled, and a disabled bench says so rather than 404ing.
# ---------------------------------------------------------------------------


def _workbench_response(fn, *args, **kwargs) -> ORJSONResponse:
    from sglang.srt.workbench.http_api import error_payload
    from sglang.srt.workbench.service import WorkbenchError

    try:
        return ORJSONResponse(fn(*args, **kwargs))
    except WorkbenchError as exc:
        status, body = error_payload(exc)
        return ORJSONResponse(body, status_code=status)


@app.get("/x-htsglang/workbench", response_class=ORJSONResponse)
async def workbench_state(raw_request: Request):
    """Queue state, tenant table, idle verdict and cross-session claim."""
    from sglang.srt.workbench.http_api import snapshot_payload

    return _workbench_response(
        snapshot_payload, raw_request.app.state.workbench_service
    )


@app.get("/x-htsglang/workbench/events", response_class=ORJSONResponse)
async def workbench_events(raw_request: Request, after: int = 0, limit: int = 200):
    """The idle-work event log, cursor-paginated by sequence number."""
    from sglang.srt.workbench.http_api import events_payload

    return _workbench_response(
        events_payload,
        raw_request.app.state.workbench_service,
        {"after": after, "limit": limit},
    )


@app.post("/x-htsglang/workbench/pause", dependencies=[Depends(validate_json_request)])
@auth_level(AuthLevel.ADMIN_OPTIONAL)  # #510: state-changing fork route
async def workbench_pause(raw_request: Request):
    """Pause or resume the whole bench, or one tenant by name."""
    from sglang.srt.workbench.http_api import pause_payload

    body = await raw_request.json()
    return _workbench_response(
        pause_payload, raw_request.app.state.workbench_service, body
    )


@app.post(
    "/x-htsglang/workbench/enqueue", dependencies=[Depends(validate_json_request)]
)
@auth_level(AuthLevel.ADMIN_OPTIONAL)  # #510: state-changing fork route
async def workbench_enqueue(raw_request: Request):
    """Add one work item to one tenant's queue."""
    from sglang.srt.workbench.http_api import enqueue_payload

    body = await raw_request.json()
    return _workbench_response(
        enqueue_payload, raw_request.app.state.workbench_service, body
    )


@app.post("/v1/audio/transcriptions")
async def openai_v1_audio_transcriptions(
    raw_request: Request,
    file: UploadFile = File(...),
    model: str = Form(default="default"),
    language: Optional[str] = Form(default=None),
    response_format: str = Form(default="json"),
    temperature: float = Form(default=0.0),
    stream: bool = Form(default=False),
    timestamp_granularities: Optional[List[str]] = Form(
        default=None, alias="timestamp_granularities[]"
    ),
):
    """OpenAI-compatible audio transcription endpoint."""
    if response_format not in ["json", "text", "verbose_json"]:
        return openai_error_response(
            f"Unsupported response_format {response_format!r}: this endpoint "
            "supports 'json', 'text' and 'verbose_json'. 'srt' and 'vtt' are "
            "not implemented.",
            status_code=400,
            err_type="invalid_request_error",
            param="response_format",
            code="unsupported_value",
        )

    audio_data = await file.read()

    return (
        await raw_request.app.state.openai_serving_transcription.create_transcription(
            audio_data=audio_data,
            model=model,
            language=language,
            response_format=response_format,
            temperature=temperature,
            stream=stream,
            timestamp_granularities=timestamp_granularities,
            raw_request=raw_request,
        )
    )


@app.websocket("/v1/realtime")
async def openai_v1_realtime_transcription(ws: WebSocket):
    """OpenAI Realtime transcription WebSocket endpoint."""
    # /v1/realtime is OpenAI's unified Realtime URL covering transcription +
    # chat modes. This handler implements the transcription subset only;
    # chat-mode session.update payloads are rejected by the
    # `Literal["transcription"]` constraint on TranscriptionSessionConfig.type
    # (see realtime/protocol.py).
    await ws.app.state.openai_serving_transcription.handle_websocket(ws)


def _local_model_cards() -> List[ModelCard]:
    """The models this process itself serves: the base model and its LoRAs."""
    served_model_name = _global_state.tokenizer_manager.served_model_name
    cards = [
        ModelCard(
            id=served_model_name,
            root=served_model_name,
            max_model_len=_global_state.tokenizer_manager.model_config.context_len,
            htsglang={
                "served_by": "local",
                "residency": "HOT",
                "gpu_resident": True,
            },
        )
    ]

    if _global_state.tokenizer_manager.server_args.enable_lora:
        lora_registry = _global_state.tokenizer_manager.lora_registry
        for _, lora_ref in lora_registry.get_all_adapters().items():
            cards.append(
                ModelCard(
                    id=lora_ref.lora_name,
                    root=lora_ref.lora_path,
                    parent=served_model_name,
                    max_model_len=None,
                    htsglang={
                        "served_by": "local",
                        "kind": "lora_adapter",
                        "residency": "HOT",
                        "gpu_resident": True,
                    },
                )
            )
    return cards


def _registry_model_cards(view) -> List[ModelCard]:
    """One card per registered engine, residency reported as the registry has it.

    Cold engines are listed. A registered engine is a model this deployment can
    serve; that it currently holds no device memory is a fact about right now,
    carried in ``x-htsglang.residency``, not a reason to pretend it does not
    exist. Omitting it would make the listing disagree with the control plane,
    and a client that then asks for it would get a "model not found" for a
    model that is registered.
    """
    cards = []
    for engine in view.engines:
        extension = engine.to_extension()
        extension["served_by"] = "registry"
        cards.append(
            ModelCard(id=engine.engine_id, root=engine.engine_id, htsglang=extension)
        )
    return cards


@app.get("/v1/models", response_class=ORJSONResponse)
async def available_models():
    """Show available models. OpenAI-compatible endpoint.

    Backed by the engine registry when one is configured (#305-M1): every
    registered engine appears, each with its residency state in the namespaced
    ``x-htsglang`` block. Without a registry this is the single served model
    plus its LoRA adapters, exactly as before.
    """
    cards = _local_model_cards()
    local_ids = {card.id for card in cards}

    view = fetch_registry_view()
    for card in _registry_model_cards(view):
        # The locally served model is usually also a registered engine. The
        # local card wins: this process knows its own context length and LoRAs.
        if card.id not in local_ids:
            cards.append(card)

    return ModelList(data=cards)


@app.get("/v1/models/{model:path}", response_class=ORJSONResponse)
async def retrieve_model(model: str):
    """Retrieves a model instance, providing basic information about the model."""
    for card in _local_model_cards():
        if card.id == model:
            return card

    engine = fetch_registry_view().by_id(model)
    if engine is not None:
        extension = engine.to_extension()
        extension["served_by"] = "registry"
        return ModelCard(id=model, root=model, htsglang=extension)

    return openai_error_response(
        f"The model '{model}' does not exist",
        status_code=404,
        err_type="invalid_request_error",
        param="model",
        code="model_not_found",
    )


@app.post("/v1/score", dependencies=[Depends(validate_json_request)])
async def v1_score_request(request: ScoringRequest, raw_request: Request):
    """Endpoint for the scoring API. Supports CausalLM (logprob-based) and SequenceClassification (class logit-based) models. See Engine.score() for documentation."""
    return await raw_request.app.state.openai_serving_score.handle_request(
        request, raw_request
    )


@app.post("/v1/responses", dependencies=[Depends(validate_json_request)])
async def v1_responses_request(request: ResponsesRequest, raw_request: Request):
    """Endpoint for the responses API with reasoning support."""

    result = await raw_request.app.state.openai_serving_responses.create_responses(
        request, raw_request
    )

    # Handle streaming responses
    if isinstance(result, AsyncGenerator):
        # #344: this route never had an abort path at all -- not even the
        # two-second background task the other OpenAI-shaped streams carry --
        # so until now a client that hung up mid-stream held its KV blocks
        # until the generation reached EOS on its own.
        return guard_generate_stream(
            StreamingResponse(
                result,
                media_type="text/event-stream",
                headers={"Cache-Control": "no-cache", "Connection": "keep-alive"},
            ),
            tokenizer_manager=_global_state.tokenizer_manager,
            rids=[request.request_id] if getattr(request, "request_id", None) else None,
        )

    return result


@app.get("/v1/responses/{response_id}")
async def v1_retrieve_responses(response_id: str, raw_request: Request):
    """Retrieve a response by ID."""
    return await raw_request.app.state.openai_serving_responses.retrieve_responses(
        response_id
    )


@app.post("/v1/responses/{response_id}/cancel")
async def v1_cancel_responses(response_id: str, raw_request: Request):
    """Cancel a background response."""
    return await raw_request.app.state.openai_serving_responses.cancel_responses(
        response_id
    )


@app.api_route(
    "/v1/rerank", methods=["POST", "PUT"], dependencies=[Depends(validate_json_request)]
)
async def v1_rerank_request(request: V1RerankReqInput, raw_request: Request):
    """Endpoint for reranking documents based on query relevance."""
    return await raw_request.app.state.openai_serving_rerank.handle_request(
        request, raw_request
    )


##### Ollama-compatible API endpoints #####

_ollama_root_route = os.environ.get("SGLANG_OLLAMA_ROOT_ROUTE")
if _ollama_root_route is not None:

    @app.get(_ollama_root_route)
    @app.head(_ollama_root_route)
    async def ollama_root():
        """Ollama-compatible root endpoint."""
        return "Ollama is running"

else:

    @app.get("/")
    @app.head("/")
    async def sglang_root():
        """Default root endpoint."""
        return "SGLang is running"


@app.post(os.environ.get("SGLANG_OLLAMA_CHAT_ROUTE", "/api/chat"))
async def ollama_chat(request: OllamaChatRequest, raw_request: Request):
    """Ollama-compatible chat endpoint."""
    return await raw_request.app.state.ollama_serving.handle_chat(request, raw_request)


@app.post(os.environ.get("SGLANG_OLLAMA_GENERATE_ROUTE", "/api/generate"))
async def ollama_generate(request: OllamaGenerateRequest, raw_request: Request):
    """Ollama-compatible generate endpoint."""
    return await raw_request.app.state.ollama_serving.handle_generate(
        request, raw_request
    )


@app.get(os.environ.get("SGLANG_OLLAMA_TAGS_ROUTE", "/api/tags"))
async def ollama_tags(raw_request: Request):
    """Ollama-compatible list models endpoint."""
    return raw_request.app.state.ollama_serving.get_tags()


@app.post(os.environ.get("SGLANG_OLLAMA_SHOW_ROUTE", "/api/show"))
async def ollama_show(request: OllamaShowRequest, raw_request: Request):
    """Ollama-compatible show model info endpoint."""
    return raw_request.app.state.ollama_serving.get_show(request.model)


##### Anthropic-compatible API endpoints #####


@app.post("/v1/messages", dependencies=[Depends(validate_json_request)])
async def anthropic_v1_messages(
    request: AnthropicMessagesRequest, raw_request: Request
):
    """Anthropic-compatible Messages API endpoint."""
    return await raw_request.app.state.anthropic_serving.handle_messages(
        request, raw_request
    )


@app.post("/v1/messages/count_tokens", dependencies=[Depends(validate_json_request)])
async def anthropic_v1_count_tokens(
    request: AnthropicCountTokensRequest, raw_request: Request
):
    """Anthropic-compatible token counting endpoint."""
    return await raw_request.app.state.anthropic_serving.handle_count_tokens(
        request, raw_request
    )


## SageMaker API
@app.get("/ping")
async def sagemaker_health() -> Response:
    """Check the health of the http server."""
    return Response(status_code=200)


@app.post("/invocations")
async def sagemaker_chat_completions(
    request: ChatCompletionRequest, raw_request: Request
):
    """OpenAI-compatible chat completion endpoint."""
    return await raw_request.app.state.openai_serving_chat.handle_request(
        request, raw_request
    )


## Vertex AI API
@app.post(os.environ.get("AIP_PREDICT_ROUTE", "/vertex_generate"))
async def vertex_generate(
    vertex_req: Annotated[VertexGenerateReqInput, Body()], raw_request: Request
):
    if not vertex_req.instances:
        return []
    inputs = {}
    for input_key in ("text", "input_ids", "input_embeds"):
        if vertex_req.instances[0].get(input_key):
            inputs[input_key] = [
                instance.get(input_key) for instance in vertex_req.instances
            ]
            break
    image_data = [
        instance.get("image_data")
        for instance in vertex_req.instances
        if instance.get("image_data") is not None
    ] or None
    req = GenerateReqInput(
        **inputs,
        image_data=image_data,
        **(vertex_req.parameters or {}),
    )
    ret = await generate_request(req, raw_request)
    if isinstance(ret, Response):
        return ret
    return ORJSONResponse({"predictions": ret})


def _create_error_response(e):
    # Native (non-/v1) endpoints share the OpenAI envelope: one error shape for
    # the whole server means a client that already handles /v1 errors handles
    # these too, and the four fields are always present rather than message-only.
    return openai_error_response(
        str(e),
        status_code=HTTPStatus.BAD_REQUEST.value,
        err_type=error_type_for_status(HTTPStatus.BAD_REQUEST.value),
        code=HTTPStatus.BAD_REQUEST.value,
    )


# FIXME: In theory we should configure ADMIN_FORCE for some entrypoints, but doing so
# would currently cause all endpoints to go through add_api_key_middleware
# (even when neither api-key nor admin-api-key is configured).
#
# For now, we simulate ADMIN_FORCE by explicitly checking the admin API key parameter.
# Once the auth wiring is refactored so ADMIN_FORCE only affects the intended
# admin endpoints, we should switch this logic to use ADMIN_FORCE directly.
def _admin_api_key_missing_response(
    status_code: HTTPStatus = HTTPStatus.BAD_REQUEST,
) -> ORJSONResponse:
    # ``error`` is an object, never a bare string: a client that does
    # ``body["error"]["message"]`` must not have to special-case this endpoint.
    return openai_error_response(
        "This endpoint requires admin API key, but this server was started "
        "without one (admin-api-key). Restart with --admin-api-key to enable.",
        status_code=status_code.value,
        err_type=error_type_for_status(status_code.value),
        code="admin_api_key_missing",
    )


# Minimal 32x32 black PNG (base64, GLM4v requires at least 32x32 sized image)
MINIMUM_PNG_PICTURE_BASE64 = "iVBORw0KGgoAAAANSUhEUgAAACAAAAAgCAYAAABzenr0AAAACXBIWXMAAA7EAAAOxAGVKw4bAAAAbUlEQVRYhe3VsQ2AMAxE0Y/lIgNQULD/OqyCMgCihCKSG4yRuKuiNH6JLsoEbMACOGBcua9HOR7Y6w6swBwMy0qLTpkeI77qdEBpBFAHBBDAGH8WrwJKI4AAegUCfAKgEgpQDvh3CR3oQCuav58qlAw73kKCSgAAAABJRU5ErkJggg=="


def _execute_server_warmup(server_args: ServerArgs):
    headers = {}
    url = server_args.url()
    if server_args.api_key:
        headers["Authorization"] = f"Bearer {server_args.api_key}"

    ssl_verify = server_args.ssl_verify()

    # Wait until the server is launched
    success = False
    for _ in range(120):
        time.sleep(1)
        try:
            res = requests.get(
                url + "/model_info", timeout=5, headers=headers, verify=ssl_verify
            )
            assert res.status_code == 200, f"{res=}, {res.text=}"
            success = True
            break
        except (AssertionError, requests.exceptions.RequestException):
            last_traceback = get_exception_traceback()
            pass

    if not success:
        logger.error(f"Initialization failed. warmup error: {last_traceback}")
        kill_process_tree(os.getpid())
        return success

    model_info = res.json()

    # Construct a warmup request (MLX: text warmup for VLM-advertising models; TODO: enable image warmup).
    is_vlm = bool(model_info.get("has_image_understanding", False)) and not is_mps()
    # #186 route 1: the image warmup below is a real image request (275 tokens,
    # 256 of them soft tokens, measured), so on a lane that cannot serve image
    # requests it would kill the server during boot -- before any user request,
    # and for a reason that has nothing to do with the workload. Fall back to
    # the text warmup instead, which exercises the same path a text-only
    # deployment on this lane will actually use. `--skip-server-warmup` remains
    # the manual escape hatch.
    # Read the decision the tokenizer manager already made, rather than
    # recomputing it, so the warmup and the admission check can never disagree.
    mm_lane_refusal = getattr(_global_state.tokenizer_manager, "mm_lane_refusal", None)
    if is_vlm and mm_lane_refusal is not None:
        logger.warning(
            "Boot warmup: using a TEXT warmup instead of the image one, because "
            "image requests are not supported on this configuration. %s",
            mm_lane_refusal,
        )
        is_vlm = False
    if model_info["is_generation"]:
        if is_vlm and not server_args.skip_tokenizer_init:
            request_name = "/v1/chat/completions"
        else:
            request_name = "/generate"
    else:
        request_name = "/encode"
    max_new_tokens = 8 if model_info["is_generation"] else 1
    json_data = {
        "sampling_params": {
            "temperature": 0,
            "max_new_tokens": max_new_tokens,
        },
    }
    if server_args.skip_tokenizer_init:
        json_data["input_ids"] = [[10, 11, 12] for _ in range(server_args.dp_size)]
        # TODO Workaround the bug that embedding errors for list of size 1
        if server_args.dp_size == 1:
            json_data["input_ids"] = json_data["input_ids"][0]
    elif (
        is_vlm
        and server_args.disaggregation_mode == "null"
        and model_info["is_generation"]
    ):
        # TODO: ChatCompletionRequest does not have bootstrap info required by disaggregation mode, disable image-warmup for now
        # Only use chat completions format for generation models, not embedding models
        json_data = {
            "model": _global_state.tokenizer_manager.served_model_name,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/png;base64,{MINIMUM_PNG_PICTURE_BASE64}"
                            },
                        },
                        {
                            "type": "text",
                            "text": "Describe the image.",
                        },
                    ],
                }
            ],
            "max_tokens": max_new_tokens,
            "stream": False,
            "temperature": 0.0,
        }
    else:
        json_data["text"] = ["The capital city of France is"] * server_args.dp_size
        # TODO Workaround the bug that embedding errors for list of size 1
        if server_args.dp_size == 1:
            json_data["text"] = json_data["text"][0]

    # Config debug dumping
    if server_args.debug_tensor_dump_input_file:
        json_data.pop("text", None)
        json_data["input_ids"] = np.load(
            server_args.debug_tensor_dump_input_file
        ).tolist()
        json_data["sampling_params"]["max_new_tokens"] = 0

    # Send a warmup request
    warmup_timeout = envs.SGLANG_WARMUP_TIMEOUT.get()
    try:
        if server_args.disaggregation_mode == "null":
            res = requests.post(
                url + request_name,
                json=json_data,
                headers=headers,
                timeout=warmup_timeout if warmup_timeout > 0 else 600,
                verify=ssl_verify,
            )
            assert res.status_code == 200, f"{res.text}"
            _global_state.tokenizer_manager.server_status = ServerStatus.Up

        else:
            logger.info(f"Start of pd disaggregation warmup ...")
            request_name = "/generate"
            json_data = {
                "sampling_params": {
                    "temperature": 0.0,
                    "max_new_tokens": 8,
                    "ignore_eos": True,
                },
                "bootstrap_host": [FAKE_BOOTSTRAP_HOST] * server_args.dp_size,
                # This is a hack to ensure fake transfer is enabled during prefill warmup
                # ensure each dp rank has a unique bootstrap_room during prefill warmup
                "bootstrap_room": [
                    i * (2**63 // server_args.dp_size) + (i % server_args.tp_size)
                    for i in range(server_args.dp_size)
                ],
                "input_ids": [[10, 11, 12, 13]] * server_args.dp_size,
            }
            res = requests.post(
                url + request_name,
                json=json_data,
                headers=headers,
                timeout=(
                    warmup_timeout if warmup_timeout > 0 else 1800
                ),  # because of deep gemm precache is very long if not precache.
                verify=ssl_verify,
            )
            if res.status_code == 200:
                logger.info(
                    f"Disaggregation warmup request completed with status {res.status_code}, resp: {res.json()}"
                )
                logger.info("End of disaggregation warmup")
                _global_state.tokenizer_manager.server_status = ServerStatus.Up
            else:
                logger.info(
                    "Disaggregation warmup failed (mode=%s), status code: %s",
                    server_args.disaggregation_mode,
                    res.status_code,
                )
                _global_state.tokenizer_manager.server_status = ServerStatus.UnHealthy

    except Exception:
        last_traceback = get_exception_traceback()
        logger.error(f"Initialization failed. warmup error: {last_traceback}")
        kill_process_tree(os.getpid())
        return False

    # Debug print
    # logger.info(f"warmup request returns: {res.json()=}")
    return success


def _wait_and_warmup(
    server_args: ServerArgs,
    launch_callback: Optional[Callable[[], None]] = None,
    execute_warmup_func: Callable = _execute_server_warmup,
):
    if server_args.checkpoint_engine_wait_weights_before_ready:
        _wait_weights_ready()

    # Send a warmup request
    if not server_args.skip_server_warmup:
        from sglang.srt.observability.startup_func_log_and_timer import startup_timer

        # Runs in the parent process, where launch_server has already called
        # enable_startup_timer(). The context manager still times and logs the
        # phase when warmup fails and this function returns early.
        with startup_timer("server_warmup"):
            warmup_ok = execute_warmup_func(server_args)
        if not warmup_ok:
            return
    else:
        _global_state.tokenizer_manager.server_status = ServerStatus.Up

    # The server is ready for requests
    logger.info("The server is fired up and ready to roll!")

    if server_args.delete_ckpt_after_loading:
        delete_directory(server_args.model_path)

    if server_args.debug_tensor_dump_input_file:
        kill_process_tree(os.getpid())

    if launch_callback is not None:
        launch_callback()


def _wait_weights_ready():
    """Wait for weights to be ready within the specified timeout."""
    timeout = WAIT_WEIGHTS_READY_TIMEOUT
    start_time = time.time()

    for _ in range(timeout):
        if _global_state.tokenizer_manager.initial_weights_loaded:
            logger.info(
                f"Weights are ready after {time.time() - start_time:.2f} seconds"
            )
            return
        time.sleep(1)

    # Timeout reached without weights being ready
    logger.error(
        f"Weights are not ready after waiting {timeout} seconds. "
        f"Consider increasing SGLANG_WAIT_WEIGHTS_READY_TIMEOUT environment variable. "
        f"Current status: initial_weights_loaded={_global_state.tokenizer_manager.initial_weights_loaded}"
    )


def _run_granian_server(
    host,
    port,
    log_level,
    tokenizer_worker_num=1,
    ssl_certfile=None,
    ssl_keyfile=None,
    ssl_ca_certs=None,
    ssl_keyfile_password=None,
    ssl_verify=False,  # MTls is not supported
    backlog=2048,
    backpressure=2048,
):
    """Serve the in-process ASGI app with Granian (embedded mode) over HTTP/2.

    Unlike Granian's default multi-process server, the embedded server runs a
    single worker as an asyncio task inside the current process. It therefore
    serves the live ``app`` object directly and reuses the already-initialized
    global state (tokenizer manager, templates, ...) through the normal
    single-tokenizer lifespan path -- no shared memory or worker re-init needed.
    The event loop is uvloop. The default backlog and backpressure values are set
    exactly like uvicorn's defaults.
    """
    import signal

    from granian import Granian
    from granian.constants import HTTPModes, Interfaces, Loops
    from granian.server.embed import Server as GranianEmbeddedServer

    Server = GranianEmbeddedServer if tokenizer_worker_num == 1 else Granian
    target = (
        app if tokenizer_worker_num == 1 else "sglang.srt.entrypoints.http_server:app"
    )
    granian_kwargs = dict(
        target=target,
        address=host,
        port=port,
        interface=Interfaces.ASGI,
        http=HTTPModes.auto,
        log_level=log_level,
        ssl_cert=ssl_certfile,
        ssl_key=ssl_keyfile,
        ssl_key_password=ssl_keyfile_password,
        ssl_ca=ssl_ca_certs,
        ssl_client_verify=ssl_verify,
        backlog=backlog,
        backpressure=backpressure,
    )

    if tokenizer_worker_num > 1:
        granian_kwargs["workers"] = tokenizer_worker_num
        granian_kwargs["loop"] = Loops.uvloop

    server = Server(**granian_kwargs)

    if tokenizer_worker_num == 1:

        async def serve():
            # The embedded server does not install its own signal handlers, so wire
            # SIGINT/SIGTERM to a graceful stop, mirroring uvicorn's behavior.
            loop = asyncio.get_running_loop()
            for sig in (signal.SIGINT, signal.SIGTERM):
                try:
                    loop.add_signal_handler(sig, server.stop)
                except (NotImplementedError, ValueError):
                    pass
            await server.serve()

        uvloop.run(serve())
    else:
        server.serve()


def _setup_and_run_http_server(
    server_args: ServerArgs,
    tokenizer_manager,
    template_manager,
    port_args: PortArgs,
    scheduler_infos: List[Dict],
    subprocess_watchdog: Optional[SubprocessWatchdog],
    execute_warmup_func: Callable = _execute_server_warmup,
    launch_callback: Optional[Callable[[], None]] = None,
):
    """Set up global state, configure middleware, and run uvicorn.

    Called by launch_server after subprocesses have been launched.
    """
    # Set global states
    set_global_state(
        _GlobalState(
            tokenizer_manager=tokenizer_manager,
            template_manager=template_manager,
            scheduler_info=scheduler_infos[0],
        )
    )

    # Store watchdog on tokenizer_manager (single source of truth for SIGQUIT handler)
    if tokenizer_manager is not None:
        tokenizer_manager._subprocess_watchdog = subprocess_watchdog

    if server_args.enable_metrics:
        add_prometheus_track_response_middleware(app)

    # #510: apply the configured CORS policy. Done for both tokenizer modes,
    # unlike the api-key middleware below, because CORS is a browser-facing
    # property of the app and not a single-tokenizer feature.
    configure_cors(app, server_args)

    # Pass additional arguments to the lifespan function.
    # They will be used for additional initialization setups.
    if server_args.tokenizer_worker_num == 1:
        # If it is single tokenizer mode, we can pass the arguments by attributes of the app object.
        app.is_single_tokenizer_mode = True
        app.server_args = server_args
        app.warmup_thread_kwargs = dict(
            server_args=server_args,
            launch_callback=launch_callback,
            execute_warmup_func=execute_warmup_func,
        )

        # Add api key authorization
        # This is only supported in single tokenizer mode.
        #
        # Backward compatibility:
        # - api_key only: behavior matches legacy (all endpoints require api_key)
        # - no keys: legacy had no restriction; ADMIN_FORCE endpoints must still be rejected when
        #   admin_api_key is not configured.
        if (
            server_args.api_key
            or server_args.admin_api_key
            or app_has_admin_force_endpoints(app)
        ):
            from sglang.srt.utils.auth import add_api_key_middleware

            add_api_key_middleware(
                app,
                api_key=server_args.api_key,
                admin_api_key=server_args.admin_api_key,
            )
    else:
        # If it is multi-tokenizer mode, we need to write the arguments to shared memory
        # for other worker processes to read.
        app.is_single_tokenizer_mode = False
        multi_tokenizer_args_shm = write_data_for_multi_tokenizer(
            port_args, server_args, scheduler_infos[0]
        )

    try:
        # Update logging configs
        set_uvicorn_logging_configs(server_args)

        if server_args.ssl_certfile:
            logger.info(
                f"SSL enabled: certfile={server_args.ssl_certfile}, "
                f"keyfile={server_args.ssl_keyfile}"
            )

        # Listen for HTTP requests
        if server_args.tokenizer_worker_num == 1:
            if server_args.enable_http2:
                logger.info(
                    f"Starting embedded Granian HTTP/2 server on "
                    f"{server_args.host}:{server_args.port}"
                )
                _run_granian_server(
                    host=server_args.host,
                    port=server_args.port,
                    log_level=server_args.log_level_http or server_args.log_level,
                    ssl_certfile=server_args.ssl_certfile,
                    ssl_keyfile=server_args.ssl_keyfile,
                    ssl_ca_certs=server_args.ssl_ca_certs,
                    ssl_keyfile_password=server_args.ssl_keyfile_password,
                    ssl_verify=False,  # No MTLS supported for now.
                )
            elif server_args.enable_ssl_refresh:
                # Use Config/Server API for access to the SSLContext.
                config = uvicorn.Config(
                    app,
                    host=server_args.host,
                    port=server_args.port,
                    root_path=server_args.fastapi_root_path,
                    log_level=server_args.log_level_http or server_args.log_level,
                    timeout_keep_alive=envs.SGLANG_TIMEOUT_KEEP_ALIVE.get(),
                    loop="uvloop",
                    ssl_keyfile=server_args.ssl_keyfile,
                    ssl_certfile=server_args.ssl_certfile,
                    ssl_ca_certs=server_args.ssl_ca_certs,
                    ssl_keyfile_password=server_args.ssl_keyfile_password,
                )
                config.load()  # Creates the SSLContext

                from sglang.srt.entrypoints.ssl_utils import SSLCertRefresher

                server = uvicorn.Server(config)

                async def _run_with_ssl_refresh():
                    refresher = SSLCertRefresher(
                        config.ssl,
                        server_args.ssl_keyfile,
                        server_args.ssl_certfile,
                        server_args.ssl_ca_certs,
                    )
                    logger.info("SSL certificate auto-refresh enabled.")
                    try:
                        await server.serve()
                    finally:
                        refresher.stop()

                import asyncio

                asyncio.run(_run_with_ssl_refresh())
            else:
                # Default case, one tokenizer process
                uvicorn.run(
                    app,
                    host=server_args.host,
                    port=server_args.port,
                    root_path=server_args.fastapi_root_path,
                    log_level=server_args.log_level_http or server_args.log_level,
                    timeout_keep_alive=envs.SGLANG_TIMEOUT_KEEP_ALIVE.get(),
                    loop="uvloop",
                    ssl_keyfile=server_args.ssl_keyfile,
                    ssl_certfile=server_args.ssl_certfile,
                    ssl_ca_certs=server_args.ssl_ca_certs,
                    ssl_keyfile_password=server_args.ssl_keyfile_password,
                )
        else:
            # Multiple tokenizer and http processes
            from uvicorn.config import LOGGING_CONFIG

            LOGGING_CONFIG["loggers"]["sglang.srt.entrypoints.http_server"] = {
                "handlers": ["default"],
                "level": "INFO",
                "propagate": False,
            }

            if server_args.enable_ssl_refresh:
                logger.warning(
                    "--enable-ssl-refresh is not supported with multiple "
                    "tokenizer workers (--tokenizer-worker-num > 1). "
                    "SSL refresh will be disabled."
                )

            if server_args.enable_http2:
                logger.info(
                    f"Starting embedded Granian HTTP/2 server on "
                    f"{server_args.host}:{server_args.port}"
                )
                _run_granian_server(
                    host=server_args.host,
                    port=server_args.port,
                    log_level=server_args.log_level_http or server_args.log_level,
                    tokenizer_worker_num=server_args.tokenizer_worker_num,
                    ssl_certfile=server_args.ssl_certfile,
                    ssl_keyfile=server_args.ssl_keyfile,
                    ssl_ca_certs=server_args.ssl_ca_certs,
                    ssl_keyfile_password=server_args.ssl_keyfile_password,
                )
            else:
                uvicorn.run(
                    "sglang.srt.entrypoints.http_server:app",
                    host=server_args.host,
                    port=server_args.port,
                    root_path=server_args.fastapi_root_path,
                    log_level=server_args.log_level_http or server_args.log_level,
                    timeout_keep_alive=envs.SGLANG_TIMEOUT_KEEP_ALIVE.get(),
                    timeout_worker_healthcheck=envs.SGLANG_UVICORN_WORKER_HEALTHCHECK_TIMEOUT.get(),
                    loop="uvloop",
                    workers=server_args.tokenizer_worker_num,
                    ssl_keyfile=server_args.ssl_keyfile,
                    ssl_certfile=server_args.ssl_certfile,
                    ssl_ca_certs=server_args.ssl_ca_certs,
                    ssl_keyfile_password=server_args.ssl_keyfile_password,
                )
    finally:
        if server_args.tokenizer_worker_num > 1:
            if multi_tokenizer_args_shm is not None:
                multi_tokenizer_args_shm.unlink()
            if _global_state is not None:
                _global_state.tokenizer_manager.socket_mapping.clear_all_sockets()


def _start_native_grpc_server_for_runtime(
    server_args,
    tokenizer_manager,
    template_manager,
    scheduler_info,
):
    try:
        from sglang.srt.entrypoints.grpc_bridge import RuntimeHandle
        from sglang.srt.grpc import _core as grpc_native
    except ImportError as e:
        raise RuntimeError(
            "Native gRPC extension (sglang.srt.grpc._core) not found in this wheel, "
            "but --grpc-port was set. The extension is built from "
            "rust/sglang-grpc/ via setuptools-rust during wheel build. Either "
            "install a wheel that includes the extension or unset --grpc-port."
        ) from e

    runtime_handle = RuntimeHandle(
        tokenizer_manager=tokenizer_manager,
        template_manager=template_manager,
        server_args=server_args,
        scheduler_info=scheduler_info or {},
    )

    grpc_handle = grpc_native.start_server(
        host=server_args.host,
        port=server_args.grpc_port,
        runtime_handle=runtime_handle,
        worker_threads=server_args.grpc_worker_threads,
    )
    logger.info(
        f"Native gRPC server started on {server_args.host}:{server_args.grpc_port}"
    )
    return grpc_handle


def _shutdown_native_grpc_server(grpc_handle) -> None:
    if grpc_handle is None:
        return
    try:
        grpc_handle.shutdown()
    except Exception as e:
        logger.warning(f"Failed to shut down native gRPC server: {e}")


def launch_server(
    server_args: ServerArgs,
    init_tokenizer_manager_func: Callable = init_tokenizer_manager,
    run_scheduler_process_func: Callable = run_scheduler_process,
    run_detokenizer_process_func: Callable = run_detokenizer_process,
    execute_warmup_func: Callable = _execute_server_warmup,
    launch_callback: Optional[Callable[[], None]] = None,
):
    """
    Launch SRT (SGLang Runtime) Server.

    The SRT server consists of an HTTP server and an SRT engine.

    - HTTP server: A FastAPI server that routes requests to the engine.
    - The engine consists of three components:
        1. TokenizerManager: Tokenizes the requests and sends them to the scheduler.
        2. Scheduler (subprocess): Receives requests from the Tokenizer Manager, schedules batches, forwards them, and sends the output tokens to the Detokenizer Manager.
        3. DetokenizerManager (subprocess): Detokenizes the output tokens and sends the result back to the Tokenizer Manager.

    Note:
    1. The HTTP server, Engine, and TokenizerManager all run in the main process.
    2. Inter-process communication is done through IPC (each process uses a different port) via the ZMQ library.
    """
    # Enable startup latency timer for Prometheus metrics when metrics are enabled.
    # startup_timer is a safe no-op when this is not called, but calling it here
    # ensures gauge emission for callers that use launch_server.
    if server_args.enable_metrics:
        from sglang.srt.observability.startup_func_log_and_timer import (
            enable_startup_timer,
        )

        enable_startup_timer()

    # Launch subprocesses
    (
        tokenizer_manager,
        template_manager,
        port_args,
        scheduler_init_result,
        subprocess_watchdog,
    ) = Engine._launch_subprocesses(
        server_args=server_args,
        init_tokenizer_manager_func=init_tokenizer_manager_func,
        run_scheduler_process_func=run_scheduler_process_func,
        run_detokenizer_process_func=run_detokenizer_process_func,
    )

    _setup_and_run_http_server(
        server_args,
        tokenizer_manager,
        template_manager,
        port_args,
        scheduler_init_result.scheduler_infos,
        subprocess_watchdog,
        execute_warmup_func=execute_warmup_func,
        launch_callback=launch_callback,
    )
