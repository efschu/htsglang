from __future__ import annotations

import json
import logging
import uuid
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any, List, Optional, Tuple, Union

import orjson
from fastapi import HTTPException, Request
from fastapi.responses import ORJSONResponse, StreamingResponse

from sglang.srt.entrypoints.openai import request_binding
from sglang.srt.entrypoints.openai.encoding_dsv32 import DS32EncodingError
from sglang.srt.entrypoints.openai.protocol import ErrorResponse, OpenAIServingRequest
from sglang.srt.liveness import EndpointClass, guard_generate_stream
from sglang.srt.managers.io_struct import EmbeddingReqInput, GenerateReqInput
from sglang.srt.managers.shutdown_gate import ServerShuttingDown
from sglang.srt.observability.req_time_stats import monotonic_time
from sglang.srt.server_args import ServerArgs
from sglang.srt.training.activity import note_serving_activity

if TYPE_CHECKING:
    from sglang.srt.managers.tokenizer_manager import TokenizerManager

logger = logging.getLogger(__name__)


async def _release_after_stream(body_iterator, hold):
    """Pass a stream through, then give the engine back -- on any ending.

    ``finally`` rather than a trailing call, because the ending that matters is
    the one nobody plans: a client that stops reading raises
    ``GeneratorExit``/``CancelledError`` into this generator, and a hold that
    leaked there would pin the engine hot until the process restarted.
    """
    try:
        async for chunk in body_iterator:
            yield chunk
    finally:
        hold.release()


# Base class for specific endpoint handlers
class OpenAIServingBase(ABC):
    """Abstract base class for OpenAI endpoint handlers"""

    def __init__(self, tokenizer_manager: TokenizerManager):
        self.tokenizer_manager = tokenizer_manager
        self.allowed_custom_labels = (
            set(
                self.tokenizer_manager.server_args.tokenizer_metrics_allowed_custom_labels
            )
            if isinstance(self.tokenizer_manager.server_args, ServerArgs)
            and self.tokenizer_manager.server_args.tokenizer_metrics_allowed_custom_labels
            else None
        )

    def _parse_model_parameter(self, model: str) -> Tuple[str, Optional[str]]:
        """Parse 'base-model:adapter-name' syntax to extract LoRA adapter.

        Returns (base_model, adapter_name) or (model, None) if no colon present.
        """
        if ":" not in model:
            return model, None

        # Split on first colon only to handle model paths with multiple colons
        parts = model.split(":", 1)
        base_model = parts[0].strip()
        adapter_name = parts[1].strip() or None

        return base_model, adapter_name

    def _resolve_lora_path(
        self,
        request_model: str,
        explicit_lora_path: Optional[Union[str, List[Optional[str]]]],
    ) -> Optional[Union[str, List[Optional[str]]]]:
        """Resolve LoRA adapter with priority: model parameter > explicit lora_path.

        Returns adapter name or None. Supports both single values and lists (batches).
        """
        _, adapter_from_model = self._parse_model_parameter(request_model)

        # Model parameter adapter takes precedence
        if adapter_from_model is not None:
            return adapter_from_model

        # Fall back to explicit lora_path
        return explicit_lora_path

    async def handle_request(
        self, request: OpenAIServingRequest, raw_request: Request
    ) -> Union[Any, StreamingResponse, ErrorResponse]:
        """Handle the specific request type with common pattern
        If you want to override this method, you should be careful to record the validation time.
        """
        # #305 request-path binding. ``binding_enabled()`` is a single
        # module-global boolean read and it is False unless a control plane was
        # configured, so the single-model boot -- which is every boot on this
        # rig today -- reaches ``_serve`` with nothing between it and the old
        # code, pays no registry lookup, and returns the same bytes as before.
        if not request_binding.binding_enabled():
            return await self._serve(request, raw_request)
        return await self._serve_bound(request, raw_request)

    async def _serve_bound(
        self, request: OpenAIServingRequest, raw_request: Request
    ) -> Union[Any, StreamingResponse, ErrorResponse]:
        """Acquire the named engine, serve, release when the response is done.

        The hold outlives this frame for a stream: a generation is in flight
        until its last chunk, and the control tick reads exactly this count
        when it decides whether an engine may step down a rung. Releasing at
        ``return`` would let a long generation be demoted mid-stream.
        """
        try:
            hold = request_binding.acquire(getattr(request, "model", "") or "")
        except request_binding.BindingRefused as refusal:
            logger.info(
                "request binding: refusing %s (%s): %s",
                refusal.detail.get("engine_id", "?"),
                refusal.code,
                refusal.message,
            )
            return self.create_error_response(
                message=refusal.message,
                err_type=refusal.err_type,
                status_code=refusal.status_code,
            )
        released = False
        try:
            response = await self._serve(request, raw_request)
        except BaseException:
            released = True
            hold.release()
            raise
        if isinstance(response, StreamingResponse):
            response.body_iterator = _release_after_stream(response.body_iterator, hold)
            return response
        if not released:
            hold.release()
        return response

    async def _serve(
        self, request: OpenAIServingRequest, raw_request: Request
    ) -> Union[Any, StreamingResponse, ErrorResponse]:
        """The request path itself, unchanged by #305."""
        received_time = monotonic_time()
        # One float store, on the one path every OpenAI-shaped generation
        # request goes through. The idle training tenant (#341) reads it to
        # decide whether this process has been quiet long enough to train on;
        # without it, a server that acquired its engine an hour ago and is
        # answering right now looks idle from the registry's side.
        note_serving_activity()

        try:
            # Validate request
            error_msg = self._validate_request(request)
            if error_msg:
                return self.create_error_response(error_msg)

            # Log the raw OpenAI request payload before conversion to tokenized form.
            request_logger = self.tokenizer_manager.request_logger
            if request_logger.log_requests and request_logger.log_requests_level >= 2:
                request_logger.log_openai_received_request(request, request=raw_request)

            # Convert to internal format
            adapted_request, processed_request = self._convert_to_internal_request(
                request, raw_request
            )

            if isinstance(adapted_request, (GenerateReqInput, EmbeddingReqInput)):
                # Only set timing fields if adapted_request supports them
                adapted_request.received_time = received_time

            # Note(Xinyuan): raw_request below is only used for detecting the connection of the client
            if hasattr(request, "stream") and request.stream:
                response = await self._handle_streaming_request(
                    adapted_request, processed_request, raw_request
                )
                # #344: one place for every OpenAI-shaped streaming endpoint.
                # Each of them already schedules a background abort, but that
                # runs after the response body ends -- which is exactly what
                # does not happen when a client stops reading without closing.
                # A non-streaming or error response passes through untouched.
                return guard_generate_stream(
                    response,
                    tokenizer_manager=self.tokenizer_manager,
                    obj=adapted_request,
                    endpoint_class=self._liveness_endpoint_class(),
                )
            else:
                return await self._handle_non_streaming_request(
                    adapted_request, processed_request, raw_request
                )
        except HTTPException as e:
            return self.create_error_response(
                message=e.detail, err_type=str(e.status_code), status_code=e.status_code
            )
        except ServerShuttingDown as e:
            # #840: ahead of the ValueError arm below, which this subclasses so
            # that untouched routes still refuse. A shutdown refusal is 503:
            # the request was valid and is retryable, and a load balancer must
            # re-route it rather than drop it as a client error.
            return self.create_error_response(
                message=str(e),
                err_type="ServiceUnavailable",
                status_code=503,
            )
        except ValueError as e:
            return self.create_error_response(
                message=str(e),
                err_type="BadRequest",
                status_code=400,
            )
        except DS32EncodingError as e:
            logger.info(f"DS32EncodingError: {e}")
            return self.create_error_response(
                message=str(e),
                err_type="BadRequest",
                status_code=400,
            )
        except Exception as e:
            logger.exception(f"Error in request: {e}")
            return self.create_error_response(
                message=f"Internal server error: {str(e)}",
                err_type="InternalServerError",
                status_code=500,
            )

    def _liveness_endpoint_class(self) -> EndpointClass:
        """Which #344 timeout applies to this endpoint's streams.

        Token streams by default, which is what chat, completions and the
        Anthropic and Ollama shapes all are. A subclass whose stream is a
        different kind of thing -- transcription writes per audio chunk, not
        per token -- overrides this to get its own budget.
        """
        return EndpointClass.LLM_STREAM

    @abstractmethod
    def _request_id_prefix(self) -> str:
        """Generate request ID based on request type"""
        pass

    def _generate_request_id_base(self, request: OpenAIServingRequest) -> Optional[str]:
        """Generate request ID based on request type"""
        return None

        # TODO(chang): the rid is used in io_strcut check and often violates `The rid should be a list` AssertionError
        # Temporarily return None in this function until the rid logic is clear.
        if rid := getattr(request, "rid", None):
            return rid

        return f"{self._request_id_prefix()}{uuid.uuid4().hex}"

    def _compute_extra_key(self, request: OpenAIServingRequest) -> Optional[str]:
        """Compute the final extra_key by concatenating cache_salt and extra_key if both are provided."""
        parts = []
        for key in ["cache_salt", "extra_key"]:
            value = getattr(request, key, None)
            if value:
                if not isinstance(value, str):
                    raise TypeError(
                        f"Value of {key} must be a string, but got {type(value).__name__}"
                    )
                parts.append(value)
        return "".join(parts) if parts else None

    @abstractmethod
    def _convert_to_internal_request(
        self,
        request: OpenAIServingRequest,
        raw_request: Request = None,
    ) -> tuple[GenerateReqInput, OpenAIServingRequest]:
        """Convert OpenAI request to internal format"""
        pass

    async def _handle_streaming_request(
        self,
        adapted_request: GenerateReqInput,
        request: OpenAIServingRequest,
        raw_request: Request,
    ) -> Union[StreamingResponse, ErrorResponse, ORJSONResponse]:
        """Handle streaming request

        Override this method in child classes that support streaming requests.
        """
        return self.create_error_response(
            message=f"{self.__class__.__name__} does not support streaming requests",
            err_type="NotImplementedError",
            status_code=501,
        )

    async def _handle_non_streaming_request(
        self,
        adapted_request: GenerateReqInput,
        request: OpenAIServingRequest,
        raw_request: Request,
    ) -> Union[Any, ErrorResponse, ORJSONResponse]:
        """Handle non-streaming request

        Override this method in child classes that support non-streaming requests.
        """
        return self.create_error_response(
            message=f"{self.__class__.__name__} does not support non-streaming requests",
            err_type="NotImplementedError",
            status_code=501,
        )

    def _validate_request(self, _: OpenAIServingRequest) -> Optional[str]:
        """Validate request"""
        pass

    def create_error_response(
        self,
        message: str,
        err_type: str = "BadRequestError",
        status_code: int = 400,
        param: Optional[str] = None,
    ) -> ORJSONResponse:
        """Create an error response in OpenAI's envelope shape.

        ``{"error": {message, type, param, code}}`` -- see
        :mod:`sglang.srt.entrypoints.openai.errors`. Handlers keep passing the
        flat arguments; the envelope is applied in one place so no endpoint can
        drift out of it.
        """
        # TODO: remove fastapi dependency in openai and move response handling to the entrypoint
        error = ErrorResponse(
            object="error",
            message=message,
            type=err_type,
            param=param,
            code=status_code,
        )
        return ORJSONResponse(
            content=error.to_openai_envelope(), status_code=status_code
        )

    def create_streaming_error_response(
        self,
        message: str,
        err_type: str = "BadRequestError",
        status_code: int = 400,
    ) -> str:
        """Create a streaming error response"""
        error = ErrorResponse(
            object="error",
            message=message,
            type=err_type,
            param=None,
            code=status_code,
        )
        return json.dumps(error.to_openai_envelope())

    def extract_custom_labels(self, raw_request):
        if (
            not self.allowed_custom_labels
            or not self.tokenizer_manager.server_args.tokenizer_metrics_custom_labels_header
        ):
            return None

        custom_labels = None
        header = (
            self.tokenizer_manager.server_args.tokenizer_metrics_custom_labels_header
        )
        try:
            raw_labels = (
                orjson.loads(raw_request.headers.get(header))
                if raw_request and raw_request.headers.get(header)
                else None
            )
        except json.JSONDecodeError as e:
            logger.exception(f"Error in request: {e}")
            raw_labels = None

        if isinstance(raw_labels, dict):
            custom_labels = {
                label: value
                for label, value in raw_labels.items()
                if label in self.allowed_custom_labels
            }
        return custom_labels

    def extract_routing_key(self, raw_request):
        if raw_request is None:
            return None
        return raw_request.headers.get("x-smg-routing-key")

    def extract_routed_dp_rank_from_header(
        self, raw_request: Request, body_routed_dp_rank: Optional[int] = None
    ) -> Optional[int]:
        """Extract routed_dp_rank from HTTP header, with higher priority than routed_dp_rank in body.

        Header name: X-Data-Parallel-Rank (case-insensitive in HTTP/1.1/2)
        """
        if raw_request is None:
            return body_routed_dp_rank

        header_value = raw_request.headers.get("x-data-parallel-rank")
        if header_value is not None:
            try:
                header_dp_rank = int(header_value)
                if (
                    body_routed_dp_rank is not None
                    and header_dp_rank != body_routed_dp_rank
                ):
                    logger.debug(
                        f"X-Data-Parallel-Rank header ({header_dp_rank}) overrides "
                        f"body routed_dp_rank ({body_routed_dp_rank})"
                    )
                return header_dp_rank
            except ValueError:
                raise HTTPException(
                    status_code=400,
                    detail=f"Invalid X-Data-Parallel-Rank header: must be an integer, got '{header_value}'",
                )

        return body_routed_dp_rank
