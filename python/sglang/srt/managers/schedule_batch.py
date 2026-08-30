from __future__ import annotations

from sglang.srt.dllm.config import DllmConfig
from sglang.srt.model_executor.forward_batch_info import ForwardBatch
from sglang.srt.utils.common import (
    Range,
    ceil_align,
    flatten_arrays_to_pinned_cpu,
    is_pin_memory_available,
)

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
Store information about requests and batches.

The following is the flow of data structures for a batch:

ScheduleBatch -> ForwardBatch

- ScheduleBatch is managed by `scheduler.py::Scheduler`.
  It contains high-level scheduling data. Most of the data is on the CPU.
- ForwardBatch is managed by `model_runner.py::ModelRunner`.
  It contains low-level tensor data. Most of the data consists of GPU tensors.
  It is constructed directly from a ScheduleBatch by `ForwardBatch.init_new`.
"""

import copy
import dataclasses
import logging
import os
import re
import sys
from array import array
from concurrent.futures import Future
from enum import Enum, auto
from functools import lru_cache
from http import HTTPStatus
from typing import (
    TYPE_CHECKING,
    Any,
    Dict,
    List,
    NamedTuple,
    Optional,
    Set,
    Tuple,
    Union,
)

import msgspec
import numpy as np
import torch

from sglang.srt.constrained.base_grammar_backend import BaseGrammarObject
from sglang.srt.disaggregation.base import BaseKVSender
from sglang.srt.disaggregation.decode_schedule_batch_mixin import (
    ScheduleBatchDisaggregationDecodeMixin,
)
from sglang.srt.disaggregation.utils import FAKE_BOOTSTRAP_HOST, DisaggregationMode
from sglang.srt.distributed.device_communicators import lockstep_sentinel
from sglang.srt.dllm.mixin.req import ReqDllmMixin
from sglang.srt.environ import envs
from sglang.srt.hardware_backend.npu.dsv4.dsv4_common_hooks import (
    maybe_evict_dsv4_state,
)
from sglang.srt.managers.embed_types import PositionalEmbeds
from sglang.srt.managers.scheduler_components.new_token_ratio_tracker import (
    NewTokenRatioTracker,
)
from sglang.srt.mem_cache import seam_layer_carry
from sglang.srt.mem_cache.allocator import BaseTokenToKVPoolAllocator
from sglang.srt.mem_cache.base_prefix_cache import (
    BasePrefixCache,
    EvictParams,
    MatchPrefixParams,
    zero_match_result,
)
from sglang.srt.mem_cache.common import (
    alloc_for_decode,
    alloc_for_extend,
    evict_from_tree_cache,
    free_swa_out_of_window_slots,
    get_alloc_reserve_per_decode,
    peer_needs_mamba_evict,
    release_kv_cache,
)
from sglang.srt.mem_cache.memory_pool import (
    CpuCopyIdsUnreadable,
    CpuCopyUnmappedRows,
    ReqToTokenPool,
)
from sglang.srt.mem_cache.radix_cache import RadixKey
from sglang.srt.model_executor.forward_batch_info import (
    CaptureHiddenMode,
    ForwardBatch,
    ForwardMode,
)
from sglang.srt.observability.metrics_collector import (
    DPCooperationInfo,
    SchedulerMetricsCollector,
)
from sglang.srt.observability.req_time_stats import (
    APIServerReqTimeStats,
    DPControllerReqTimeStats,
    SchedulerReqTimeStats,
)
from sglang.srt.runtime_context import get_parallel, get_server_args
from sglang.srt.sampling.sampling_batch_info import SamplingBatchInfo
from sglang.srt.sampling.sampling_params import SamplingParams
from sglang.srt.sampling.thinking_budget import THINKING_BUDGET_INTERNAL_KEY
from sglang.srt.server_args import ServerArgs
from sglang.srt.utils import flatten_nested_list
from sglang.srt.utils.cuda_ipc_transport_utils import CudaIpcTensorTransportProxy

if TYPE_CHECKING:
    from typing import Any, Dict

    from sglang.srt.configs.model_config import ModelConfig
    from sglang.srt.managers.hisparse_coordinator import HiSparseCoordinator
    from sglang.srt.managers.scheduler_components.metrics_reporter import PrefillStats
    from sglang.srt.session.session_controller import Session
    from sglang.srt.speculative.spec_info import SpecInput, SpeculativeAlgorithm

INIT_INCREMENTAL_DETOKENIZATION_OFFSET = 5

# Constant used as the base offset for MM (multimodal) pad values.
# This ensures pad_values don't overlap with valid text token IDs.
MM_PAD_SHIFT_VALUE = 1_000_000

logger = logging.getLogger(__name__)

#: #1036: per-callsite census of admitted-prefix demotions. site -> count.
_1036_PREFIX_DEMOTIONS: Dict[str, int] = {}


def _note_1036_prefix_demotion(req, start: int) -> None:
    """#1036 INSTRUMENT: name whoever admits a prefix below `protected`.

    THE TWO-LINE GAP THIS EXISTS TO FILL. Boot `f178a94c51` died on a rank
    disagreement whose mutation site is not instrumented anywhere. PP0's own
    consult and its admitted geometry are ADJACENT log lines -- 951701 and
    951702 -- with nothing between them:

        #969B READMIT-MATCH n=61 rid=6da98e29 prefix_len=8192 protected=8192
        #969 EXTENT        n=28          reqs=[('6da98e29', 0, 4096, 0, 4096)]

    while PP1 and PP2 admitted the same rid in the same pass at
    `(8192, 8868, 8192, 676)`. Something dropped 8192 to 0 on PP0 alone and
    left no trace, so the writer cannot be named from any boot log we have.
    `set_extend_range` is the chokepoint every admitting path goes through
    (`schedule_policy.py` :1202, :1287, :1428, :1719, :1743, :1762, :2115),
    which makes it the one place that sees every candidate writer.

    AN INSTRUMENT, NOT A GATE -- deliberately, and this is the whole design.
    The refusal that belongs here cannot be written yet: gating a mechanism
    nobody has identified is the symptom-patching the standing order forbids,
    and a boot could not validate it (silence would not distinguish "gated the
    right path" from "gated nothing"). This names the path first; the #1035
    refusal template then applies to whatever it names.

    PER-SITE COUNTERS, NOT PER-EVENT LINES, and that is a lesson paid for
    once already: #1035's single rate-limited counter went quiet after five
    occurrences, so ABSENCE OF THE LINE IS NOT ABSENCE OF THE EVENT. Here every
    distinct callsite carries its own counter and its own first-three budget,
    so a writer that appears for the first time on occurrence 900 still prints,
    and the census below is the number to read rather than the line count.

    CHEAP ON THE HOT PATH: the caller compares two ints and only calls this on
    a demotion; the frame walk happens exclusively inside a real event.
    """
    try:
        protected = int(getattr(req, "cache_protected_len", 0) or 0)
        # The frame walk skips this module so the site named is the ADMITTING
        # caller, not `set_extend_range` itself -- the same correction the
        # #1034 provenance walk needed when it first printed `base.py:101`.
        site = "?"
        frame = sys._getframe(2) if hasattr(sys, "_getframe") else None
        hops = 0
        while frame is not None and hops < 12:
            name = frame.f_code.co_filename.rsplit("/", 1)[-1]
            if name != "schedule_batch.py":
                site = f"{name}:{frame.f_lineno} in {frame.f_code.co_name}"
                break
            frame = frame.f_back
            hops += 1
        n = _1036_PREFIX_DEMOTIONS.get(site, 0) + 1
        _1036_PREFIX_DEMOTIONS[site] = n
        if n <= 3 or n % 64 == 0:
            logger.warning(
                "#1036 PREFIX DEMOTED BELOW PROTECTED rid=%s site=%s "
                "protected=%d admitted_start=%d lost=%d prefix_indices=%d "
                "host_hit=%s mamba_host_hit=%s seam_readmit=%s "
                "site_occurrence=%d census=%s",
                str(getattr(req, "rid", "?"))[:8],
                site,
                protected,
                start,
                protected - start,
                0 if req.prefix_indices is None else len(req.prefix_indices),
                getattr(req, "host_hit_length", None),
                getattr(req, "mamba_host_hit_length", None),
                getattr(req, "seam_readmit_epoch", None),
                n,
                dict(sorted(_1036_PREFIX_DEMOTIONS.items())),
            )
    except Exception:  # noqa: BLE001
        # An instrument may never be the thing that kills a boot.
        logger.warning("#1036 PREFIX-DEMOTION PROBE RAISED", exc_info=True)

#: #783 seam state transfer log prefix, so the acceptance lines are greppable.
SEAM_STATE_PREFIX = "[#783 seam-state]"

# #622: per-finish trace for cross-rank finish-divergence attribution.
_FINISH_TRACE = os.environ.get("SGLANG_FINISH_TRACE", "0") not in ("0", "", "false")


@lru_cache(maxsize=1)
def sanity_check_mm_pad_shift_value(vocab_size: int) -> None:
    if vocab_size > MM_PAD_SHIFT_VALUE:
        raise ValueError(
            f"Model vocab_size ({vocab_size}) exceeds MM_PAD_SHIFT_VALUE ({MM_PAD_SHIFT_VALUE}). "
            f"MM pad_values may overlap with valid token IDs. "
            f"Please increase MM_PAD_SHIFT_VALUE in schedule_batch.py."
        )


def _compute_pad_value(hash: int) -> int:
    """Compute pad value from hash."""
    return MM_PAD_SHIFT_VALUE + (hash % (1 << 30))


class BaseFinishReason:
    def to_json(self):
        raise NotImplementedError()


class FINISH_MATCHED_TOKEN(BaseFinishReason):
    def __init__(self, matched: Union[int, List[int]]):
        super().__init__()
        self.matched = matched

    def to_json(self):
        return {
            "type": "stop",  # to match OpenAI API's return value
            "matched": self.matched,
        }


class FINISH_MATCHED_STR(BaseFinishReason):
    def __init__(self, matched: str):
        super().__init__()
        self.matched = matched

    def to_json(self):
        return {
            "type": "stop",  # to match OpenAI API's return value
            "matched": self.matched,
        }


class FINISHED_MATCHED_REGEX(BaseFinishReason):
    def __init__(self, matched: str):
        super().__init__()
        self.matched = matched

    def to_json(self):
        return {
            "type": "stop",  # to match OpenAI API's return value
            "matched": self.matched,
        }


class FINISH_LENGTH(BaseFinishReason):
    def __init__(self, length: int):
        super().__init__()
        self.length = length

    def to_json(self):
        return {
            "type": "length",  # to match OpenAI API's return value
            "length": self.length,
        }


class FINISH_ABORT(BaseFinishReason):
    def __init__(self, message=None, status_code=None, err_type=None):
        super().__init__()
        self.message = message or "Aborted"
        self.status_code = status_code
        self.err_type = err_type

    def to_json(self):
        return {
            "type": "abort",
            "message": self.message,
            "status_code": self.status_code,
            "err_type": self.err_type,
        }


class Modality(Enum):
    IMAGE = auto()
    VIDEO = auto()
    AUDIO = auto()

    @staticmethod
    def from_str(modality_str: str):
        try:
            return Modality[modality_str.upper()]
        except KeyError:
            raise ValueError(
                f"Invalid modality string: {modality_str}. Valid modalities are: {[m.name for m in Modality]}"
            )

    @staticmethod
    def all():
        return [Modality.IMAGE, Modality.VIDEO, Modality.AUDIO]


class MultimodalInputFormat(Enum):
    NORMAL = auto()
    PROCESSOR_OUTPUT = auto()
    PRECOMPUTED_EMBEDDING = auto()


@dataclasses.dataclass
class MultimodalDataItem:
    """
    One MultimodalDataItem represents a single multimodal input (one image, one video, or one audio).
    For example, if there are 3 images and 1 audio, there will be 4 MultimodalDataItems.

    Each item has its own hash and pad_value, enabling per-image RadixAttention caching.

    We put the common fields first and the model-specific fields in model_specific_data.
    """

    modality: Modality
    hash: int = None
    pad_value: int = None
    offsets: Optional[list] = None

    format: MultimodalInputFormat = MultimodalInputFormat.NORMAL

    # the raw features returned by processor, e.g. pixel_values or audio_features
    feature: Union[torch.Tensor, np.ndarray] = None
    # the precomputed embeddings, passed as final encoder embeddings
    # One and only one of the feature and precomputed_embeddings will be empty
    precomputed_embeddings: Optional[Union[torch.Tensor, np.ndarray]] = None

    # Model-specific data stored in a dictionary
    model_specific_data: dict[str, Any] = dataclasses.field(default_factory=dict)

    def __getattr__(self, name: str):
        if (
            "model_specific_data" in self.__dict__
            and name in self.__dict__["model_specific_data"]
        ):
            return self.__dict__["model_specific_data"][name]
        else:
            raise AttributeError(
                f"'{self.__class__.__name__}' object has no attribute '{name}'"
            )

    def __setitem__(self, key: str, value: Any):
        if key in self.__dict__:
            self.__dict__[key] = value
        else:
            self.model_specific_data[key] = value

    def set(self, key: str, value: Any):
        self.__setitem__(key, value)

    @staticmethod
    def is_empty_list(l):
        if l is None:
            return True
        return len([item for item in flatten_nested_list(l) if item is not None]) == 0

    def set_pad_value(self):
        """
        Set the pad value after first hashing the data
        """
        if self.pad_value is not None:
            return

        from sglang.srt.managers.mm_utils import hash_feature

        if envs.SGLANG_MM_SKIP_COMPUTE_HASH.get():
            import uuid

            self.hash = uuid.uuid4().int
            self.pad_value = _compute_pad_value(self.hash)
            return
        if self.hash is None:
            if self.feature is not None:
                hashed_feature = self.feature
            else:
                hashed_feature = self.precomputed_embeddings
            self.hash = hash_feature(hashed_feature)
        assert self.hash is not None
        self.pad_value = _compute_pad_value(self.hash)

    def is_modality(self, modality: Modality) -> bool:
        return self.modality == modality

    def is_audio(self):
        return self.modality == Modality.AUDIO

    def is_image(self):
        return self.modality == Modality.IMAGE

    def is_video(self):
        return self.modality == Modality.VIDEO

    def is_valid(self) -> bool:
        return self.is_image() or self.is_video() or self.is_audio()

    def validate(self):
        ...
        # TODO

    def is_precomputed_embedding(self):
        return self.format == MultimodalInputFormat.PRECOMPUTED_EMBEDDING

    @staticmethod
    def from_dict(obj: dict):
        kwargs = dict(obj)
        modality = kwargs.pop("modality")
        if isinstance(modality, str):
            modality = Modality[modality]
        ret = MultimodalDataItem(modality=modality, **kwargs)
        ret.validate()
        return ret

    def has_cuda_ipc_proxy(self):
        return (
            isinstance(self.feature, CudaIpcTensorTransportProxy)
            or isinstance(self.precomputed_embeddings, CudaIpcTensorTransportProxy)
            or any(
                isinstance(value, CudaIpcTensorTransportProxy)
                for value in self.model_specific_data.values()
            )
        )

    def reconstruct(self, target_device: int):
        """materialize cuda ipc proxy tensors in-place on target_device"""
        if isinstance(self.feature, CudaIpcTensorTransportProxy):
            self.feature = self.feature.reconstruct_on_target_device(target_device)
        if isinstance(self.precomputed_embeddings, CudaIpcTensorTransportProxy):
            self.precomputed_embeddings = (
                self.precomputed_embeddings.reconstruct_on_target_device(target_device)
            )
        for extra_key in self.model_specific_data:
            if isinstance(
                self.model_specific_data[extra_key], CudaIpcTensorTransportProxy
            ):
                extra_data = self.model_specific_data[
                    extra_key
                ].reconstruct_on_target_device(target_device)
                self.model_specific_data[extra_key] = extra_data


@dataclasses.dataclass
class MultimodalProcessorOutput:
    """Raw output from multimodal processors before scheduler-side preparation (pad, hash).

    This is the typed replacement for the dict previously returned by
    ``BaseMultimodalProcessor.process_mm_data_async``.  Preprocessed inputs may
    already carry ``pad_value`` and ``hash`` to avoid hashing the same tensor once
    per scheduler TP rank.
    """

    mm_items: List[MultimodalDataItem]
    input_ids: Optional[List[int]] = None
    padded_input_ids: Optional[List[int]] = None

    # image
    im_token_id: Optional[int] = None
    im_start_id: Optional[int] = None
    im_end_id: Optional[int] = None
    slice_start_id: Optional[int] = None
    slice_end_id: Optional[int] = None

    # video
    video_token_id: Optional[int] = None

    # audio
    audio_token_id: Optional[int] = None
    audio_start_id: Optional[int] = None
    audio_end_id: Optional[int] = None

    # QWen2-VL related
    mrope_positions: Optional[torch.Tensor] = None
    mrope_position_delta: Optional[torch.Tensor] = None

    # Moss-VL related
    vision_position_ids: Optional[torch.Tensor] = None
    media_nums_per_sample: Optional[List[int]] = None
    visible_frame_counts: Optional[torch.Tensor] = None

    # for transformers-compatibility
    token_type_ids: Optional[torch.Tensor] = None

    @staticmethod
    def from_dict(d: dict) -> MultimodalProcessorOutput:
        return MultimodalProcessorOutput(
            mm_items=d["mm_items"],
            input_ids=d.get("input_ids"),
            padded_input_ids=d.get("padded_input_ids"),
            im_token_id=d.get("im_token_id"),
            im_start_id=d.get("im_start_id"),
            im_end_id=d.get("im_end_id"),
            slice_start_id=d.get("slice_start_id"),
            slice_end_id=d.get("slice_end_id"),
            video_token_id=d.get("video_token_id"),
            audio_token_id=d.get("audio_token_id"),
            audio_start_id=d.get("audio_start_id"),
            audio_end_id=d.get("audio_end_id"),
            mrope_positions=d.get("mrope_positions"),
            mrope_position_delta=d.get("mrope_position_delta"),
            vision_position_ids=d.get("vision_position_ids"),
            media_nums_per_sample=d.get("media_nums_per_sample"),
            visible_frame_counts=d.get("visible_frame_counts"),
        )

    @staticmethod
    def build_padded_input_ids(input_ids, mm_items: List[MultimodalDataItem]):
        """pad the input_ids with mm_items if it's not already padded"""
        if input_ids is None or not mm_items:
            return None

        for item in mm_items:
            if item.pad_value is None or item.offsets is None:
                return None

        if isinstance(input_ids, torch.Tensor):
            padded_input_ids = input_ids.flatten().tolist()
        else:
            padded_input_ids = list(input_ids)

        for item in mm_items:
            for start, end in item.offsets:
                padded_input_ids[start : end + 1] = [item.pad_value] * (end - start + 1)
        return padded_input_ids


@dataclasses.dataclass
class MultimodalInputs:
    """The multimodal data related inputs."""

    # items of data
    mm_items: List[MultimodalDataItem]
    padded_input_ids: Optional[List[int]] = None
    image_pad_len: Optional[list] = None
    num_image_tokens: Optional[int] = None

    # image
    im_token_id: Optional[int] = None
    im_start_id: Optional[int] = None
    im_end_id: Optional[int] = None
    slice_start_id: Optional[int] = None
    slice_end_id: Optional[int] = None

    # video
    video_token_id: Optional[int] = None

    # audio
    audio_token_id: Optional[int] = None
    audio_start_id: Optional[int] = None
    audio_end_id: Optional[int] = None

    # QWen2-VL related
    mrope_positions: Optional[torch.Tensor] = None
    mrope_position_delta: Optional[torch.Tensor] = None
    mrope_position_delta_repeated_cache: Optional[torch.Tensor] = None

    # Moss-VL related
    vision_position_ids: Optional[torch.Tensor] = None
    media_nums_per_sample: Optional[List[int]] = None
    visible_frame_counts: Optional[torch.Tensor] = None

    def release_features(self):
        """Release feature tensors to free GPU memory."""
        for item in self.mm_items:
            item.feature = None

    @staticmethod
    def from_processor_output(obj: MultimodalProcessorOutput):
        mm_items = obj.mm_items
        assert isinstance(mm_items, list)
        mm_items = [item for item in mm_items if item.is_valid()]

        # try reconstructing from cuda-ipc
        reconstruct_device = None
        for mm_item in mm_items:
            if mm_item.has_cuda_ipc_proxy():
                if reconstruct_device is None:
                    reconstruct_device = torch.cuda.current_device()
                mm_item.reconstruct(reconstruct_device)

        if envs.SGLANG_MM_BUFFER_SIZE_MB.get() > 0:
            # Multi-modal feature hashing optimization:
            # When SGLANG_MM_BUFFER_SIZE_MB > 0, we temporarily move feature tensors to GPU
            # for faster hash computation, while avoiding OOM issues.
            from sglang.srt.managers.mm_utils import (
                init_feature_buffer,
                is_feature_buffer_initialized,
                reset_buffer_offset,
                try_add_to_buffer,
            )

            device = torch.cuda.current_device() if torch.cuda.is_available() else "cpu"
            if not is_feature_buffer_initialized():
                init_feature_buffer(device)
            reset_buffer_offset()
            for item in mm_items:
                if item.feature is not None:
                    if isinstance(item.feature, torch.Tensor):
                        item.feature = try_add_to_buffer(item.feature)

        for item in mm_items:
            item.set_pad_value()

        if envs.SGLANG_MM_BUFFER_SIZE_MB.get() > 0:
            for item in mm_items:
                if item.feature is not None:
                    item.feature = item.feature.to("cpu", non_blocking=True)

        mm_inputs = MultimodalInputs(
            mm_items=mm_items,
            padded_input_ids=obj.padded_input_ids,
        )
        optional_args = [
            "mrope_positions",
            "mrope_position_delta",
            "im_token_id",
            "im_start_id",
            "im_end_id",
            "video_token_id",
            "slice_start_id",
            "slice_end_id",
            "audio_start_id",
            "audio_end_id",
            "audio_token_id",
            "vision_position_ids",
            "media_nums_per_sample",
            "visible_frame_counts",
        ]
        for arg in optional_args:
            val = getattr(obj, arg, None)
            if val is not None:
                setattr(mm_inputs, arg, val)

        return mm_inputs

    def contains_image_inputs(self) -> bool:
        return any(item.is_image() for item in self.mm_items)

    def contains_video_inputs(self) -> bool:
        return any(item.is_video() for item in self.mm_items)

    def contains_audio_inputs(self) -> bool:
        return any(item.is_audio() for item in self.mm_items)

    def contains_mm_input(self) -> bool:
        return any(True for item in self.mm_items if item.is_valid())

    def compute_mm_token_counts(self) -> Tuple[int, int, int]:
        """Count prompt tokens consumed by each modality (image, audio, video).

        A modality's token count is the total span covered by its items'
        offsets. Returns a (image_tokens, audio_tokens, video_tokens) tuple.
        """
        image_tokens = audio_tokens = video_tokens = 0
        for item in self.mm_items:
            if not item.offsets:
                continue
            num_tokens = sum(end - start + 1 for start, end in item.offsets)
            if item.is_image():
                image_tokens += num_tokens
            elif item.is_audio():
                audio_tokens += num_tokens
            elif item.is_video():
                video_tokens += num_tokens
        return image_tokens, audio_tokens, video_tokens

    def merge(self, other: MultimodalInputs):
        """
        merge image inputs when requests are being merged
        """

        # args needed to be merged
        optional_args = [
            "mm_items",
            "image_pad_len",
        ]
        for arg in optional_args:
            self_arg = getattr(self, arg, None)
            if self_arg is not None:
                setattr(self, arg, self_arg + getattr(other, arg))

        mrope_positions = self.mrope_positions
        if mrope_positions is not None:
            if other.mrope_positions is None:
                self.mrope_positions = mrope_positions
            else:
                self.mrope_positions = torch.cat(
                    [self.mrope_positions, other.mrope_positions], dim=1
                )

        mrope_position_delta = self.mrope_position_delta
        if mrope_position_delta is not None:
            if other.mrope_position_delta is None:
                self.mrope_position_delta = mrope_position_delta
            else:
                self.mrope_position_delta = torch.cat(
                    [self.mrope_position_delta, other.mrope_position_delta], dim=0
                )

        for key, val in other.__dict__.items():
            if "_id" in key:
                # set token_ids
                if getattr(self, key, None) is None:
                    setattr(self, key, getattr(other, key, None))
        # other args would be kept intact


@dataclasses.dataclass(slots=True, kw_only=True)
class ReqLogprob:
    top_logprobs_num: int
    token_ids_logprob: Optional[List[int]]
    input_token_logprobs_val: Optional[List[float]] = None
    input_token_logprobs_idx: Optional[List[int]] = None
    input_top_logprobs_val: Optional[List[List[float]]] = None
    input_top_logprobs_idx: Optional[List[List[int]]] = None
    input_token_ids_logprobs_val: Optional[List[List[float]]] = None
    input_token_ids_logprobs_idx: Optional[List[List[int]]] = None
    output_token_logprobs_val: Optional[list] = None
    output_token_logprobs_idx: Optional[list] = None
    output_top_logprobs_val: Optional[list] = None
    output_top_logprobs_idx: Optional[list] = None
    # Can contain either lists or GPU tensors (delayed copy optimization for prefill-only scoring)
    output_token_ids_logprobs_val: Optional[List[Union[List[float], torch.Tensor]]] = (
        None
    )
    output_token_ids_logprobs_idx: Optional[list] = None


class Req(ReqDllmMixin):
    """The input and output status of a request."""

    def __init__(
        self,
        rid: str,
        origin_input_text: str,
        origin_input_ids: array[int],
        sampling_params: SamplingParams,
        return_logprob: bool = False,
        top_logprobs_num: int = 0,
        dllm_config: Optional[DllmConfig] = None,
        token_ids_logprob: List[int] = None,
        stream: bool = False,
        origin_input_ids_unpadded: Optional[array[int]] = None,
        lora_id: Optional[str] = None,
        input_embeds: Optional[List[List[float]]] = None,
        positional_embed_overrides: Optional[PositionalEmbeds] = None,
        token_type_ids: List[int] = None,
        session: Optional[Session] = None,
        custom_logit_processor: Optional[str] = None,
        require_reasoning: bool = False,
        return_hidden_states: bool = False,
        return_routed_experts: bool = False,
        routed_experts_start_len: int = 0,
        return_indexer_topk: bool = False,
        eos_token_ids: Optional[Set[int]] = None,
        bootstrap_host: Optional[str] = None,
        bootstrap_port: Optional[int] = None,
        bootstrap_room: Optional[int] = None,
        disagg_mode: Optional[DisaggregationMode] = None,
        routed_dp_rank: Optional[int] = None,
        disagg_prefill_dp_rank: Optional[int] = None,
        vocab_size: Optional[int] = None,
        priority: Optional[int] = None,
        metrics_collector: Optional[SchedulerMetricsCollector] = None,
        extra_key: Optional[str] = None,
        routing_key: Optional[str] = None,
        dimensions: Optional[int] = None,
        http_worker_ipc: Optional[str] = None,
        time_stats: Optional[
            Union[APIServerReqTimeStats, DPControllerReqTimeStats]
        ] = None,
        return_pooled_hidden_states: bool = False,
        multi_item_delimiter_indices: Optional[List[int]] = None,
        session_id: Optional[str] = None,
    ):
        # Input and output info
        self.rid = rid
        self.origin_input_ids = origin_input_ids
        self.origin_input_ids_unpadded = (
            origin_input_ids_unpadded
            if origin_input_ids_unpadded
            else self.origin_input_ids
        )  # Before image padding
        # Each decode stage's output ids. Append-only by contract:
        # _refresh_fill_ids infers how many output tokens are already in
        # full_untruncated_fill_ids from lengths alone, so in-place rewrites
        # that preserve length would silently corrupt fill_ids.
        self.output_ids = array("q")
        # Full untruncated sequence: origin + output (+ DLLM mask block).
        # Kept in sync by _refresh_fill_ids; admission only updates
        # extend_range, never mutates this array's length.
        self.full_untruncated_fill_ids = array("q")
        # #987 THE SHADOW FILL TAIL: output tokens an UPSTREAM PP rank holds
        # for this request that this rank has not produced itself, carried on
        # the admission decision (`PPAdmissionEntry.fill_len` / `fill_tail`)
        # and adopted by `pp_admission_congruence.adopt_carried_fill`.
        #
        # DELIBERATELY NOT `output_ids`, and this is the whole reason the pair
        # exists. `output_ids` is what this rank GENERATED: the client stream
        # slices it (scheduler_components/output_streamer.py:383-384) and
        # `_update_finish_state_impl` below counts it against
        # `max_new_tokens`, both unguarded by PP rank. A token this rank never
        # sampled must not enter it. It belongs to the FILL -- the length two
        # ranks were disagreeing about across the `tp_to_pp` seam -- and
        # `_refresh_fill_ids` is the one reader that honours it.
        #
        # `pp_carried_fill_len` is the upstream's total fill length and is
        # kept only so a reader can tell a live carry from a stale one;
        # `pp_carried_fill_tail` is the suffix itself, newest last. Both stay
        # None/() on every rank that never adopts, which is every rank on a
        # healthy pass and every rank of a `pp_size <= 1` boot.
        self.pp_carried_fill_len: Optional[int] = None
        self.pp_carried_fill_tail: Tuple[int, ...] = ()
        self.extend_range: Optional[Range] = None
        self.dllm_initialized: bool = False

        self.session = session
        self.session_id = session_id
        self.input_embeds = input_embeds
        self.positional_embed_overrides = positional_embed_overrides
        self.multi_item_delimiter_indices = multi_item_delimiter_indices

        # For req-level memory management
        self.kv_committed_len = 0
        self.kv_allocated_len = 0
        self.kv_committed_freed = False
        self.kv_overallocated_freed = False

        # kv-session-offload (S1): FCFS arrival order (monotonic counter,
        # assigned once at scheduler admission -- identical on every TP rank)
        # and the spill tier ("host" while the session's full-attention KV
        # lives in the pinned host pool; None on the default path).
        self.kv_arrival_seq: Optional[int] = None
        self.kv_spill_state: Optional[str] = None
        # S1b partial spill: number of device-resident head tokens [0, boundary);
        # the tail [boundary, seq) lives on host (0 == not partially spilled).
        self.kv_spill_boundary: int = 0

        # for cross-encoder model
        self.token_type_ids = token_type_ids

        # The length of KV that have been removed in swa cache.
        # SWA KV cache eviction behavior differs by cache type:
        # - Radix cache: KV in range [cache_protected_len, swa_evicted_seqlen) is freed manually in
        #   `ScheduleBatch.maybe_evict_swa`; KV in range [0, cache_protected_len) is freed during radix cache eviction.
        # - Chunk cache: KV in range [0, swa_evicted_seqlen) is freed manually in `ScheduleBatch.maybe_evict_swa`.
        self.swa_evicted_seqlen = 0
        # Tokens in [0, swa_evict_floor) are protected from SWA window eviction.
        # This is used by prefill-aware SWA models such as Unlimited-OCR to keep prompt/image KV visible during decode.
        self.swa_evict_floor: int = 0

        # The index of the extend / decode batch
        self.extend_batch_idx = 0
        self.decode_batch_idx = 0

        # For multi-http worker
        self.http_worker_ipc = http_worker_ipc

        # Require reasoning for the request
        self.require_reasoning = require_reasoning

        # State indicating whether the reasoning phase has finished (only meaningful when require_reasoning is True)
        self._is_reasoning_over = False
        self.reasoning_tokens = 0

        # Sampling info
        if isinstance(sampling_params.custom_params, dict):
            sampling_params = copy.copy(sampling_params)
            sampling_params.custom_params = sampling_params.custom_params | {
                "__req__": self
            }
        self.sampling_params = sampling_params
        self.custom_logit_processor = custom_logit_processor
        # Thinking budget (#540): the tokenizer manager attaches the built-in
        # ThinkingBudgetLogitProcessor itself, so such a request carries a
        # processor without --enable-custom-logit-processor being set. The
        # marker key is server-owned (stripped from client input in
        # attach_thinking_budget), so it cannot be forged by a caller.
        self.custom_logit_processor_internal = bool(
            custom_logit_processor is not None
            and isinstance(self.sampling_params.custom_params, dict)
            and self.sampling_params.custom_params.get(THINKING_BUDGET_INTERNAL_KEY)
            is True
        )
        self.return_hidden_states = return_hidden_states

        # extra key for classifying the request (e.g. cache_salt)
        if lora_id is not None:
            extra_key = (
                extra_key or ""
            ) + lora_id  # lora_id is concatenated to the extra key

        self.extra_key = extra_key
        self.lora_id = lora_id
        self.routing_key = routing_key

        # Memory pool info
        self.req_pool_idx: Optional[int] = None
        self.mamba_pool_idx: Optional[torch.Tensor] = None  # shape (1)
        self.mamba_ping_pong_track_buffer: Optional[torch.Tensor] = None  # shape (2)
        self.mamba_next_track_idx: Optional[int] = None  # 0 or 1
        self.mamba_last_track_seqlen: Optional[int] = (
            None  # seq len of the last cached mamba state
        )
        # the branching point seqlen to track mamba state. If set, given by prefix match,
        # it will be the tracked seqlen in the ping pong buffer for the right prefill pass.
        self.mamba_branching_seqlen: Optional[int] = None
        # Deferred COW: source mamba pool index from radix cache node (copy on forward stream)
        self.mamba_cow_src_index: Optional[torch.Tensor] = None
        # Deferred clear: newly allocated mamba slot needs zeroing on forward stream
        self.mamba_needs_clear: bool = False
        # #991 PROVENANCE OF THE ACTIVE MAMBA SLOT.
        #
        # True only while the slot in `mamba_pool_idx` was acquired
        # SPECULATIVELY by THIS admission round's prefix match (the COW /
        # host-load-back resume sites), i.e. while a rejection by
        # `add_one_req` still owes a give-back for it. Cleared the moment the
        # slot becomes batch-owned (`HybridReqToTokenPool.alloc`), and on
        # every path that releases or drops the slot.
        #
        # It exists because `not req.session` -- upstream's predicate at
        # `scheduler.py`'s revert site -- is a PROXY for provenance that this
        # fork invalidated: #984/#968b/#971 re-queue voided and displaced
        # requests UNRESET, so "in the waiting queue holding a live slot it
        # did not acquire this round" became the ordinary case rather than a
        # session-only one. This is #984's own "pages versus claim" doctrine
        # applied to the third increment; the lock ref and the chunk already
        # carry provenance, the mamba slot was the one that did not.
        self.mamba_slot_acquired_this_admission: bool = False
        # Deferred clear for freshly claimed ping-pong track slots: every
        # claimed mamba slot must be zeroed before any kernel can observe it
        # (fresh-boot semantics; recycled slots otherwise expose the previous
        # occupant's state to any premature/partial read).
        self.mamba_pingpong_clear_indices: Optional[torch.Tensor] = None
        # Lazy extra buffer: skip radix cache insert when prealloc failed at
        # boundary — the forward overwrites the only slot, corrupting the state.
        self.mamba_lazy_is_insert: bool = True

        # Check finish
        self.tokenizer = None
        self.finished_reason: Optional[BaseFinishReason] = None
        # finished position (in output_ids), used when checking stop conditions with speculative decoding
        self.finished_len = None
        # Whether this request has finished output
        self.finished_output = None
        # If we want to abort the request in the middle of the event loop,
        # set to_finish instead of directly setting finished_reason.
        # Note: We should never set finished_reason in the middle, the req will get filtered and never respond
        self.to_finish: Optional[BaseFinishReason] = None
        self.stream = stream
        self.eos_token_ids = eos_token_ids
        self.vocab_size = vocab_size
        self.priority = priority
        # Fast lane (Variant C Stage 0): set True for lane='fast' requests by the
        # scheduler at admission. Used by the anti-starvation reserved-heavy-slots
        # floor. Default False keeps the standard (heavy) path unchanged.
        self.is_fast_lane = False
        # kv-session-offload spill (latency) class, tagged by the scheduler at
        # admission from the request's `spill_class` field: "never" | "normal"
        # | "preferred". Consulted only by the offload manager's victim
        # selection; the default "normal" keeps the stock FCFS order. Spelled
        # as a literal rather than importing kv_session_offload.SPILL_CLASS_
        # NORMAL: this module's import block is already an E402 region and Req
        # construction is on the per-request path. The two are pinned together
        # by test_kv_spill_class_unit.test_req_defaults_to_normal.
        self.spill_class = "normal"

        # For incremental decoding
        # ----- | --------- read_ids -------|
        # ----- |   surr_ids  |
        # xxxxx | xxxxxxxxxxx | xxxxxxxxxxx |
        # ----- ^ ----------- ^ ----------- ^
        # ----- 1 ----------- 2 ----------- 3
        # 1: surr_offset
        # 2: read_offset
        # 3: last token
        self.surr_offset = None  # Surrounding offset to defeat the cleanup algorithm
        self.read_offset = None
        self.decoded_text = ""

        # For multimodal inputs
        self.multimodal_inputs: Optional[MultimodalInputs] = None
        # Pre-computed multimodal prompt token counts; populated on the prefill
        # node and transferred to decode via the metadata buffer in disagg (PD) mode.
        self.mm_image_tokens: int = 0
        self.mm_audio_tokens: int = 0
        self.mm_video_tokens: int = 0

        # Prefix info
        # The indices to kv cache for the shared prefix.
        self.prefix_indices: torch.Tensor = torch.empty((0,), dtype=torch.int64)
        # TODO(ispobock): rename to last_device_node
        self.last_node: Any = None
        self.last_host_node: Any = None
        self.best_match_node: Any = None
        # Per-component host hit lengths split off from host_hit_length:
        self.host_hit_length = 0
        self.swa_host_hit_length = 0
        self.mamba_host_hit_length = 0
        # Total cached prefix length (on-device prefix_indices + host_hit_length),
        # capped at the max allowed prefix. Set during prefix matching at schedule
        # time and used to estimate uncached tokens / sort by longest prefix for
        # load reporting.
        self.num_matched_prefix_tokens = 0
        # Tokens loaded from storage backend (L3) during prefetch for this request
        self.storage_hit_length = 0
        # The node to lock until for swa radix tree lock ref
        self.swa_uuid_for_lock: Optional[int] = None
        # Whether the prefill-time SWA tree lock has been released early
        self.swa_prefix_lock_released: bool = False
        # The prefix length that is inserted into the tree cache
        self.cache_protected_len: int = 0

        # Whether or not if it is chunked. It increments whenever
        # it is chunked, and decrement whenever chunked request is
        # processed.
        self.inflight_middle_chunks = 0

        # For retraction
        self.is_retracted = False
        # Indicates if the req has ever been retracted.
        self.retracted_stain = False
        # W30: the PHASE-FLIP SEAM's own retraction stamp -- the epoch of the
        # #856 no-carry cutover that retracted this request, or None.
        #
        # DELIBERATELY SEPARATE FROM `is_retracted`, which is set by four
        # unrelated paths (this seam, decode-OOM preemption in
        # `retract_decode`, the PD prefill path, the PP void path). Only the
        # seam's retraction makes a request's re-admission flip TRANSPORT --
        # a read-through of tokens already computed and already fenced to the
        # canonical store. An OOM-preempted request's re-prefill is real work
        # and must stay subject to the purity rule, so the two populations
        # need two different marks. Set only in
        # `phase_flip_runtime.build_cutover_release._retract`; spent (cleared)
        # on the one re-admission it licenses.
        self.seam_readmit_epoch = None

        # #890: DID THE LAST SEAM RESTORE ACTUALLY RESTORE?
        #
        # The exemption above is granted on the claim that a re-admission
        # "recomputes nothing -- it is a cache restore". `restore_seam_state`
        # has two branches that refuse the copy and send the tokens back to be
        # RECOMPUTED, which falsifies that claim for this request; W38 counted
        # 90 and 21 of them in two boots. Set there and only there, and cleared
        # there by a restore that actually happens, so this says what the LAST
        # attempt did rather than passing a life sentence. Read by
        # `phase_purity.seam_transport_premise_holds`
        # (`SEAM_RESTORE_REFUSED_ATTR`), which is where the permission is
        # issued and therefore where it has to be withdrawn.
        self.seam_restore_refused = False

        # kv-session-offload Prefill-Spill (born-spilled, PS1-V1a): set at
        # admission when the prompt's lifetime KV would not fit VRAM but its
        # prefill input transiently fits. The prompt is admitted (instead of
        # wedged) and rides the existing decode-OOM spill (try_spill) into the
        # host pool. Default False -> byte-identical when the feature is off.
        self.born_spilled = False

        # kv-session-offload Prefill-Spill DEEP (PS2): set at admission when
        # not even the prefill INPUT fits the device budget -- the strict
        # complement of born_spilled's window. Such a prompt never gets device
        # KV slots at all: prepare_for_extend takes spill_extend_alloc, the
        # chunk's K/V is written straight into the session's host region, and
        # the session is handed to the spill tick one iteration later. Default
        # False -> byte-identical when the feature is off.
        self.born_spilled_deep = False

        # Incremental streamining
        self.send_token_offset: int = 0
        self.send_decode_id_offset: int = 0
        # TODO (Byron): send_output_token_logprobs_offset and send_decode_id_offset can be different in disaggregation mode
        # because the decode server does not have the first output token logprobs
        self.send_output_token_logprobs_offset: int = 0

        # Logprobs (arguments)
        self.return_logprob = return_logprob
        # Start index to compute logprob from.
        self.logprob_start_len = 0
        self.logprob = ReqLogprob(
            top_logprobs_num=top_logprobs_num,
            token_ids_logprob=token_ids_logprob,
        )

        # Logprobs (return values)
        # True means the input logprob has been already sent to detokenizer.
        self.input_logprob_sent: bool = False
        # Temporary holder to store input_token_logprobs.
        self.input_token_logprobs: Optional[List[Tuple[int]]] = None
        self.temp_input_top_logprobs_val: Optional[List[torch.Tensor]] = None
        self.temp_input_top_logprobs_idx: Optional[List[int]] = None
        self.temp_input_token_ids_logprobs_val: Optional[List[float]] = None
        self.temp_input_token_ids_logprobs_idx: Optional[List[int]] = None

        if return_logprob:
            # shape: (bs, 1)
            self.logprob.output_token_logprobs_val = []
            self.logprob.output_token_logprobs_idx = []
            # shape: (bs, k)
            self.logprob.output_top_logprobs_val = []
            self.logprob.output_top_logprobs_idx = []
            # Can contain either lists or GPU tensors (delayed copy optimization for prefill-only scoring)
            self.logprob.output_token_ids_logprobs_val = []
            self.logprob.output_token_ids_logprobs_idx = []
        self.hidden_states: List[List[float]] = []
        self.hidden_states_tensor = None  # Note: use tensor instead of list to transfer hidden_states when PD + MTP
        self.output_topk_p = None
        self.output_topk_index = None

        # capture routed experts
        self.return_routed_experts = return_routed_experts
        self.routed_experts_start_len = routed_experts_start_len
        self.routed_experts: Optional[torch.Tensor] = (
            None  # cpu tensor: shape (seqlen, topk)
        )

        self.return_indexer_topk = return_indexer_topk
        self.indexer_topk: Optional[torch.Tensor] = (
            None  # cpu tensor: shape (seqlen, num_indexer_layers, index_topk)
        )
        # Customized info
        self.customized_info: Optional[Dict[str, List[Any]]] = None

        # Embedding (return values)
        self.embedding = None

        # Constrained decoding
        self.grammar_key: Optional[Tuple[str, str]] = None
        self.grammar: Optional[Union[BaseGrammarObject, Future[BaseGrammarObject]]] = (
            None
        )
        self.grammar_wait_ct = 0

        # The number of cached tokens that were already cached in the KV cache
        self.cached_tokens = 0
        self.already_computed = 0

        # Detailed breakdown of cached tokens by source (for HiCache)
        self.cached_tokens_device = 0  # Tokens from device cache (GPU)
        self.cached_tokens_host = 0  # Tokens from host cache (CPU memory)
        self.cached_tokens_storage = 0  # Tokens from L3 storage backend
        self._cache_breakdown_computed = (
            False  # Track if breakdown was already computed
        )

        # Per-request count of verification forward passes.
        self.spec_verify_ct = 0

        # Per-request count of accepted draft tokens (excludes the bonus token).
        self.spec_num_correct_drafts = 0

        self.spec_num_block_accept_tokens = 0

        self.spec_num_cap_tokens = 0

        # Acceptance histogram for speculative decoding.
        # List index = number of accepted tokens in a step, List value = count of steps with that many accepted tokens.
        # Example: histogram[0] = 5 means 5 steps with 0 accepted tokens, histogram[3] = 10 means 10 steps with 3 accepted tokens.
        self.spec_correct_drafts_histogram: List[int] = []

        self.spec_cap_lens_histogram: List[int] = []

        # The number of times this request has been retracted / preempted.
        self.retraction_count = 0
        self.retraction_mb_id = None

        # #273: lifetime count of "sole survivor of retract_decode and still
        # does not fit" events for THIS request. Distinct from
        # retraction_count (a client-visible total covering every kind of
        # retraction): this one gates a bounded retry-vs-fail decision for
        # the specific solo-OOM corner (see retract_decode) and is never
        # reset, so a request that keeps losing this exact race -- even
        # with ordinary retractions in between -- eventually fails cleanly
        # instead of retrying forever.
        self.solo_oom_count = 0

        # For observability
        self.metrics_collector = metrics_collector
        if time_stats is not None:
            self.time_stats = SchedulerReqTimeStats.new_from_obj(time_stats)
        else:
            self.time_stats = SchedulerReqTimeStats(disagg_mode=disagg_mode)
        self.time_stats.set_metrics_collector(metrics_collector)
        self.time_stats.set_scheduler_recv_time()
        self.has_log_time_stats: bool = False

        # For disaggregation
        self.bootstrap_host: str = bootstrap_host
        self.bootstrap_port: Optional[int] = bootstrap_port
        self.bootstrap_room: Optional[int] = bootstrap_room
        # Decode-local: the already-emitted boundary token to replay when a
        # retracted request is rebootstrapped. Set in pause_generation(retract)
        # and consumed in the decode transfer commit; never plumbed to prefill.
        self.pd_rebootstrap_forced_output_id: Optional[int] = None
        self.skip_radix_cache_insert = bootstrap_host == FAKE_BOOTSTRAP_HOST
        self.disagg_kv_sender: Optional[BaseKVSender] = None

        self.routed_dp_rank: Optional[int] = routed_dp_rank
        self.disagg_prefill_dp_rank: Optional[int] = disagg_prefill_dp_rank

        # the start index of the sent kv cache
        # We want to send it chunk by chunk for chunked prefill.
        # After every chunk forward, we do the following:
        # kv_send(req.input_ids[req.start_send_idx:req.extend_range.end])
        # start_send_idx = req.extend_range.end
        self.start_send_idx: int = 0

        # For overlap schedule, we delay the kv transfer until `process_batch_result_disagg_prefill` rather than `process_prefill_chunk` in non-overlap
        # This is because kv is not ready in `process_prefill_chunk`.
        # We use `tmp_end_idx` to store the end index of the kv cache to send.
        self.tmp_end_idx: int = -1
        self.metadata_buffer_index: int = -1
        # Used in overlap sequence to signal that an optimistic request should
        # abort chunking. Set in create_sender, consumed in process_batch_result.
        self.pending_bootstrap = False
        # Number of optimistic prefill forward passes started. preserved across retracts.
        self.prefill_attempt_count = 0

        # For Matryoshka embeddings
        self.dimensions = dimensions

        # Whether to return pooled hidden states (pre-head transformer output)
        self.return_pooled_hidden_states = return_pooled_hidden_states
        self.pooled_hidden_state = None

        # For diffusion LLM
        self.init_diffusion_llm(dllm_config)

        # For hisparse
        self.hisparse_staging = False

    @property
    def seqlen(self) -> int:
        """Get the current sequence length of the request."""
        return len(self.origin_input_ids) + len(self.output_ids)

    @property
    def is_prefill_only(self) -> bool:
        """Check if this request is prefill-only (no token generation needed)."""
        # NOTE: when spec is enabled, prefill_only optimizations are disabled

        spec_alg = get_server_args().speculative_algorithm
        return self.sampling_params.max_new_tokens == 0 and spec_alg is None

    @property
    def output_ids_through_stop(self) -> array[int]:
        """Get the output ids through the stop condition. Stop position is included."""
        if self.finished_len is not None:
            return self.output_ids[: self.finished_len]
        return self.output_ids

    def needs_host_load_back(self) -> bool:
        """Whether any cache layer has a host hit that needs L2 H2D load_back."""
        return (
            self.host_hit_length > 0
            or self.swa_host_hit_length > 0
            or self.mamba_host_hit_length > 0
        )

    def _cache_commit_len(self) -> int:
        # Report only the prompt prefix so thinking + answer fall into the
        # overallocated range and are reclaimed by release_kv_cache. #22373.
        if get_server_args().strip_thinking_cache and self.reasoning_tokens > 0:
            return min(self.kv_committed_len, len(self.origin_input_ids))
        return self.kv_committed_len

    def pop_committed_kv_cache(self) -> int:
        """Return the length of committed KV cache and mark them as freed."""
        assert not self.kv_committed_freed, (
            f"Committed KV cache already freed ({self.kv_committed_len=})"
        )
        self.kv_committed_freed = True
        return self._cache_commit_len()

    def pop_overallocated_kv_cache(self) -> Tuple[int, int]:
        """Return the range of over-allocated KV cache and mark them as freed."""

        # NOTE: This function is called when there is over-allocation of KV cache.
        # Over-allocation: we allocate more KV cache than the committed length.
        # e.g., speculative decoding may allocate more KV cache than actually used.
        assert not self.kv_overallocated_freed, (
            f"Overallocated KV cache already freed, {self.kv_committed_len=}, {self.kv_allocated_len=}"
        )
        self.kv_overallocated_freed = True
        return self._cache_commit_len(), self.kv_allocated_len

    def update_spec_correct_drafts_histogram(self, num_correct_drafts: int):
        """Update the speculative decoding acceptance histogram.

        Args:
            num_correct_drafts: Number of correct draft tokens (no bonus) in this step.
        """
        if len(self.spec_correct_drafts_histogram) <= num_correct_drafts:
            self.spec_correct_drafts_histogram.extend(
                [0] * (num_correct_drafts - len(self.spec_correct_drafts_histogram) + 1)
            )
        self.spec_correct_drafts_histogram[num_correct_drafts] += 1

    def update_spec_cap_lens_histogram(self, cap_len: int):
        cap_len = int(cap_len)
        if len(self.spec_cap_lens_histogram) <= cap_len:
            self.spec_cap_lens_histogram.extend(
                [0] * (cap_len - len(self.spec_cap_lens_histogram) + 1)
            )
        self.spec_cap_lens_histogram[cap_len] += 1

    def extend_image_inputs(self, image_inputs):
        if self.multimodal_inputs is None:
            self.multimodal_inputs = image_inputs
        else:
            self.multimodal_inputs.merge(image_inputs)

    def finished(self) -> bool:
        # Whether request reached finished condition
        return self.finished_reason is not None

    def set_extend_range(self, start: int, end: int) -> None:
        # #1036 INSTRUMENT ONLY -- no behaviour change, no refusal, nothing
        # below reads its result. See `_note_1036_prefix_demotion`.
        if start < getattr(self, "cache_protected_len", 0):
            _note_1036_prefix_demotion(self, start)
        self.extend_range = Range(start, end)

    def get_fill_ids(self) -> array:
        return self.full_untruncated_fill_ids[: self.extend_range.end]

    def _refresh_fill_ids(self) -> None:
        """Keep full_untruncated_fill_ids == origin_input_ids + output_ids by
        appending only the new output tokens.

        Falls back to a full rebuild when the in-place append is invalid:
        - aliasing: scheduler_pp_mixin assigns full_untruncated_fill_ids =
          origin_input_ids directly, so extending in place would write output
          tokens into the origin;
        - lengths disagree: fresh req (array still empty), retraction
          (output_ids reset to empty), or set_finish_with_abort (origin
          replaced by a 1-token stub).
        """
        # #987: THE ONE READER OF THE CARRIED TAIL. When an upstream PP rank
        # holds output tokens this rank has not produced, the fill is
        # `origin + output + carried_tail` -- and it is rebuilt whole rather
        # than extended, because the in-place branch below infers how many
        # output tokens are already present FROM LENGTHS ALONE and a suffix it
        # does not know about makes that inference wrong by exactly the length
        # of the tail. The rebuild is O(fill) but reachable only for a request
        # that is actually mid-divergence at the seam, which is one request on
        # the passes that would otherwise all have voided.
        #
        # SELF-CANCELLING: `adopt_carried_fill` recomputes the deficit against
        # `origin + output` on every pass and clears the pair the moment this
        # rank produces the token itself, so this branch stops being taken
        # without anything here having to detect it.
        carried_tail = getattr(self, "pp_carried_fill_tail", None)
        if carried_tail:
            self.full_untruncated_fill_ids = (
                self.origin_input_ids + self.output_ids + array("q", carried_tail)
            )
            return
        n_have_output = len(self.full_untruncated_fill_ids) - len(self.origin_input_ids)
        if (
            self.full_untruncated_fill_ids is not self.origin_input_ids
            and 0 <= n_have_output <= len(self.output_ids)
        ):
            self.full_untruncated_fill_ids.extend(self.output_ids[n_have_output:])
        else:
            self.full_untruncated_fill_ids = self.origin_input_ids + self.output_ids

    def init_next_round_input(
        self,
        tree_cache: Optional[BasePrefixCache] = None,
        cow_mamba: Optional[bool] = None,
    ):
        if self.is_dllm():
            self._init_fill_ids_for_dllm()
            self.determine_dllm_phase()
        else:
            self._refresh_fill_ids()

        input_len = len(self.full_untruncated_fill_ids)

        # Streaming sessions reuse committed KV from the session slot, so
        # custom logprob_start_len is not supported — override to -1.
        if (
            self.session is not None
            and self.session.streaming
            and self.return_logprob
            and self.logprob_start_len >= 0
        ):
            logger.warning(
                "logprob_start_len=%d is not supported for streaming sessions "
                "and will be ignored (rid=%s). Only new-token logprobs are returned.",
                self.logprob_start_len,
                self.rid,
            )
            self.logprob_start_len = -1

        # Pass the full array with a raw-token cap (limit) instead of slicing,
        # avoiding an O(context) copy per prefill-batch build.
        token_ids_to_match = self.full_untruncated_fill_ids
        key_limit: Optional[int] = self._compute_max_prefix_len(input_len)

        # SWA lives in a per-request ring that's not content-stable and is never
        # stored in the radix tree, so a reused prefix carries stale SWA. Cap the
        # match by the trailing sliding window so it gets re-prefilled, rewriting
        # this request's SWA ring. No-op for other layouts.
        if tree_cache is not None:
            reprefill_tail = tree_cache.swa_reprefill_tail_tokens()
            if reprefill_tail:
                capped = max(0, input_len - reprefill_tail)
                key_limit = capped if key_limit is None else min(key_limit, capped)

        # Disable prefix caching when embed overrides are present: same token IDs
        # with different override vectors must not share cached KV values.
        if self.positional_embed_overrides is not None:
            token_ids_to_match = array("q")
            key_limit = None

        if tree_cache is not None:
            if cow_mamba is None:
                cow_mamba = tree_cache.supports_mamba()
            # unified_kv SWA lives in a per-request ring that is not content-stable
            # and never cached in the radix tree, so a reused prefix carries stale
            # SWA. Cap the match by the trailing sliding window so it is re-prefilled
            # into this request's ring. No-op for other layouts (returns 0).
            reprefill_tail = tree_cache.swa_reprefill_tail_tokens()
            if reprefill_tail:
                capped = max(0, input_len - reprefill_tail)
                key_limit = capped if key_limit is None else min(key_limit, capped)
            match_result = tree_cache.match_prefix(
                MatchPrefixParams(
                    key=RadixKey(
                        token_ids=token_ids_to_match,
                        extra_key=self.extra_key,
                        limit=key_limit,
                    ),
                    req=self,
                    cow_mamba=cow_mamba,
                )
            )
            if envs.SGLANG_RADIX_FORCE_MISS.get():
                match_result = zero_match_result(tree_cache, match_result)
            (
                self.prefix_indices,
                self.last_node,
                self.last_host_node,
                self.best_match_node,
                self.host_hit_length,
                self.swa_host_hit_length,
                self.mamba_host_hit_length,
                self.mamba_branching_seqlen,
            ) = (
                match_result.device_indices,
                match_result.last_device_node,
                match_result.last_host_node,
                match_result.best_match_node,
                match_result.host_hit_length,
                match_result.swa_host_hit_length,
                match_result.mamba_host_hit_length,
                match_result.mamba_branching_seqlen,
            )
            if match_result.cache_protected_len is not None:
                self.cache_protected_len = match_result.cache_protected_len
            else:
                self.cache_protected_len = len(self.prefix_indices)
            # #969B RE-ADMISSION DECISION PROBE (temporary). The open question
            # from §H3: when PP0 rebuilt a re-admitted request FROM ZERO while
            # its peers continued at (4096, N), was PP0 EARLY (it consulted the
            # match before its own prefetch landed) or was its tree dropped and
            # never re-read? Both look identical downstream -- an empty
            # prefix_indices -- so the answer has to be taken HERE, at the
            # consult, with the prefetch registry read in the same breath.
            # Logs only re-admitted requests, so the ordinary path is silent.
            # Grep: "#969B READMIT-MATCH".
            try:
                from sglang.srt.managers.phase_purity import (
                    SEAM_READMIT_ATTR as _SRA,
                )

                if getattr(self, _SRA, None) is not None:
                    _og = getattr(tree_cache, "ongoing_prefetch", None)
                    _rid = str(getattr(self, "rid", "?"))
                    _n = getattr(Req, "_969b_n", 0) + 1
                    Req._969b_n = _n
                    logger.warning(
                        "#969B READMIT-MATCH n=%d rid=%s prefix_len=%d "
                        "host_hit=%s mamba_host_hit=%s protected=%s "
                        "prefetch_registered=%s prefetch_keys=%d "
                        "readmit_epoch=%s input_len=%d "
                        # #969AC LAP PROVENANCE, read where §AC measured the
                        # divergence. `lap` is how many times THIS rank has
                        # put this rid back on the admission path; `site` is
                        # WHICH code site did it last -- "own-void" means this
                        # rank decided the void for itself (self-initiated),
                        # "void-output" means the void arrived on the stream
                        # (told), "intake"/"retract-intake" is ordinary
                        # queueing. `from_fwd` is the forward_ct of the pass
                        # that started the lap. The question this answers:
                        # WHO runs the extra lap the peers do not, and out of
                        # which site does that lap originate.
                        # #969AF: the field is a STICKY LABEL on the Req, not
                        # an event. It says how this request was LAST put on
                        # the queue and is re-printed on every consult, so
                        # COUNTING THESE LINES IS NOT A SITE CENSUS -- §AD and
                        # §AE both read it as one and drew a 36/24/24 and a
                        # 24/6/6 "retraction asymmetry" out of a print count.
                        # The retraction census is `#969AD RETRACT`, which is
                        # one line per actual retraction (12/12/12, symmetric).
                        "#969AC lap=%s last_queued_as=%s from_fwd=%s "
                        "stamped_rank=%s",
                        _n,
                        _rid[:8],
                        0 if self.prefix_indices is None else len(self.prefix_indices),
                        getattr(self, "host_hit_length", None),
                        getattr(self, "mamba_host_hit_length", None),
                        getattr(self, "cache_protected_len", None),
                        (_rid in _og) if isinstance(_og, dict) else "n/a",
                        len(_og) if isinstance(_og, dict) else -1,
                        getattr(self, _SRA, None),
                        input_len,
                        getattr(self, "_969ac_lap", None),
                        getattr(self, "_969ac_site", None),
                        getattr(self, "_969ac_fwd", None),
                        getattr(self, "_969ac_rank", None),
                    )
            except Exception:  # noqa: BLE001
                logger.warning("#969B READMIT-MATCH PROBE RAISED", exc_info=True)


            if self.is_dllm():
                self._update_block_offset_for_dllm()

        if (
            self.is_retracted
            and self.multimodal_inputs is not None
            and self.multimodal_inputs.mrope_positions is not None
        ):
            from sglang.srt.managers.mm_utils import (
                extend_mrope_positions_for_retracted_request,
            )

            self.multimodal_inputs.mrope_positions = (
                extend_mrope_positions_for_retracted_request(
                    self.multimodal_inputs.mrope_positions, len(self.output_ids)
                )
            )

    def _compute_max_prefix_len(self, input_len: int) -> int:
        # NOTE: the matched length is at most 1 less than the input length to enable logprob computation
        max_prefix_len = input_len - 1
        if self.return_logprob and self.logprob_start_len >= 0:
            max_prefix_len = min(max_prefix_len, self.logprob_start_len)
        return max(max_prefix_len, 0)

    # Based on https://github.com/vllm-project/vllm/blob/7a64d24aad69e4d2548aa0bf528d9fe63428ab01/vllm/transformers_utils/detokenizer.py#L194-L313
    def init_incremental_detokenize(self):
        first_iter = self.surr_offset is None or self.read_offset is None

        output_ids = self.output_ids_through_stop

        if first_iter:
            self.read_offset = len(self.origin_input_ids_unpadded)
            self.surr_offset = max(
                self.read_offset - INIT_INCREMENTAL_DETOKENIZATION_OFFSET, 0
            )
            self.surr_and_decode_ids = (
                self.origin_input_ids_unpadded[self.surr_offset :] + output_ids
            )
            self.cur_decode_ids_len = len(output_ids)
        else:
            self.surr_and_decode_ids.extend(output_ids[self.cur_decode_ids_len :])
            self.cur_decode_ids_len = len(output_ids)

        return self.surr_and_decode_ids, self.read_offset - self.surr_offset

    def _stop_match_tail_len(self, new_accepted_len: int) -> int:
        max_len_tail_str = max(
            self.sampling_params.stop_str_max_len + 1,
            self.sampling_params.stop_regex_max_len + 1,
        )
        # Cover all newly accepted tokens so an early stop string is not missed
        # when speculative decoding accepts multiple tokens per step.
        return min(
            max_len_tail_str + max(new_accepted_len - 1, 0), len(self.output_ids)
        )

    def tail_str(self, new_accepted_len: int = 1) -> str:
        # Check stop strings and stop regex patterns together
        if (
            len(self.sampling_params.stop_strs) == 0
            and len(self.sampling_params.stop_regex_strs) == 0
        ):
            return ""

        tail_len = self._stop_match_tail_len(new_accepted_len)
        return self.tokenizer.decode(self.output_ids[-tail_len:])

    def check_match_stop_str_prefix(self) -> bool:
        """
        Check if the suffix of tail_str overlaps with any stop_str prefix
        """
        if not self.sampling_params.stop_strs:
            return False

        tail_str = self.tail_str()

        # Early return if tail_str is empty
        if not tail_str:
            return False

        for stop_str in self.sampling_params.stop_strs:
            if not stop_str:
                continue
            # Check if stop_str is contained in tail_str (fastest check first)
            if stop_str in tail_str:
                return True

            # Check if tail_str suffix matches stop_str prefix
            # Only check if stop_str is not empty, it's for stream output
            min_len = min(len(tail_str), len(stop_str))
            for i in range(1, min_len + 1):
                if tail_str[-i:] == stop_str[:i]:
                    return True

        return False

    def _check_token_based_finish(self, new_accepted_tokens: List[int]) -> bool:
        if self.sampling_params.ignore_eos:
            return False

        # Check stop token ids
        matched_eos = False

        for i, token_id in enumerate(new_accepted_tokens):
            if self.sampling_params.stop_token_ids:
                matched_eos |= token_id in self.sampling_params.stop_token_ids
            if self.eos_token_ids:
                matched_eos |= token_id in self.eos_token_ids
            if self.tokenizer is not None:
                matched_eos |= token_id == self.tokenizer.eos_token_id
                if self.tokenizer.additional_stop_token_ids:
                    matched_eos |= token_id in self.tokenizer.additional_stop_token_ids
            if matched_eos:
                self.finished_reason = FINISH_MATCHED_TOKEN(matched=token_id)
                matched_pos = len(self.output_ids) - len(new_accepted_tokens) + i
                self.finished_len = matched_pos + 1
                return True

        return False

    def _locate_str_stop_finished_len(
        self,
        new_accepted_len: int,
        *,
        stop_str: Optional[str] = None,
        stop_regex: Optional[str] = None,
    ) -> int:
        """Map a matched stop string/regex to output_ids length (stop included)."""

        def matched(text: str) -> bool:
            if stop_str is not None:
                return stop_str in text
            return re.search(stop_regex, text) is not None

        tail_len = self._stop_match_tail_len(new_accepted_len)
        start = len(self.output_ids) - tail_len
        token_window = self.output_ids[start:]

        # Old prefixes were checked in the previous step.
        for token_count in range(
            max(1, len(token_window) - new_accepted_len + 1), len(token_window)
        ):
            if matched(self.tokenizer.decode(token_window[:token_count])):
                return start + token_count

        # The full tail window is already known to match by the caller.
        return len(self.output_ids)

    def _check_str_based_finish(self, new_accepted_len: int = 1):
        if (
            len(self.sampling_params.stop_strs) > 0
            or len(self.sampling_params.stop_regex_strs) > 0
        ):
            tail_str = self.tail_str(new_accepted_len)

            # Check stop strings
            if len(self.sampling_params.stop_strs) > 0:
                for stop_str in self.sampling_params.stop_strs:
                    stop_str_in_tail = stop_str in tail_str
                    if stop_str_in_tail or stop_str in self.decoded_text:
                        self.finished_reason = FINISH_MATCHED_STR(matched=stop_str)
                        if stop_str_in_tail:
                            self.finished_len = self._locate_str_stop_finished_len(
                                new_accepted_len, stop_str=stop_str
                            )
                        return True

            # Check stop regex
            if len(self.sampling_params.stop_regex_strs) > 0:
                for stop_regex_str in self.sampling_params.stop_regex_strs:
                    if re.search(stop_regex_str, tail_str):
                        self.finished_reason = FINISHED_MATCHED_REGEX(
                            matched=stop_regex_str
                        )
                        self.finished_len = self._locate_str_stop_finished_len(
                            new_accepted_len, stop_regex=stop_regex_str
                        )
                        return True

        return False

    def _check_vocab_boundary_finish(self, new_accepted_tokens: List[int] = None):
        for i, token_id in enumerate(new_accepted_tokens):
            if token_id >= self.vocab_size or token_id < 0:
                offset = len(self.output_ids) - len(new_accepted_tokens) + i
                if self.sampling_params.stop_token_ids:
                    self.output_ids[offset] = next(
                        iter(self.sampling_params.stop_token_ids)
                    )
                if self.eos_token_ids:
                    self.output_ids[offset] = next(iter(self.eos_token_ids))
                self.finished_reason = FINISH_MATCHED_STR(matched="NaN happened")
                self.finished_len = offset + 1
                return True

        return False

    def update_finish_state(self, new_accepted_len: int = 1):
        # #622: env-gated finish trace. On a divergent finish (one rank drops
        # a request its peers keep — proven family injury), the three ranks'
        # FINISH-TRACE lines for the same rid diff to show WHICH input
        # diverged: the token tail, or the accept window length.
        if not _FINISH_TRACE:
            return self._update_finish_state_impl(new_accepted_len)
        was_finished = self.finished()
        self._update_finish_state_impl(new_accepted_len)
        if not was_finished and self.finished():
            _fr = self.finished_reason
            logger.info(
                "FINISH-TRACE rid=%s reason=%s matched=%r acc_len=%d out_len=%d tail=%s",
                self.rid[:16] if isinstance(self.rid, str) else self.rid,
                type(_fr).__name__ if _fr is not None else None,
                getattr(_fr, "matched", None),
                new_accepted_len,
                len(self.output_ids) if self.output_ids else 0,
                self.output_ids[-6:] if self.output_ids else [],
            )

    def _update_finish_state_impl(self, new_accepted_len: int = 1):
        if self.finished():
            return

        if self.to_finish:
            self.finished_reason = self.to_finish
            self.to_finish = None
            return

        if len(self.output_ids) >= self.sampling_params.max_new_tokens:
            self.finished_reason = FINISH_LENGTH(
                length=self.sampling_params.max_new_tokens
            )
            self.finished_len = self.sampling_params.max_new_tokens
            return

        if self.grammar is not None:
            if self.grammar.is_terminated():
                self.finished_reason = FINISH_MATCHED_TOKEN(matched=self.output_ids[-1])
                return

        new_accepted_tokens = self.output_ids[-new_accepted_len:]

        # Sanitize out-of-range / NaN token ids before any decode.
        if self._check_vocab_boundary_finish(new_accepted_tokens):
            return

        # Stop string beats EOS/stop-token matched in the same step (speculative
        # decoding can accept >1 token): token-based would trim only the last
        # token and leak the stop string.
        if self._check_str_based_finish(new_accepted_len):
            return

        if self._check_token_based_finish(new_accepted_tokens):
            return

    def truncate_prefix_to(self, told: int) -> None:
        """Shorten the reused prefix to ``told`` AND keep the protected length
        consistent with it (#930).

        ONE HELPER FOR TWO SITES, because the two sites are siblings of each
        other and drifted identically. ``get_new_batch_prefill``'s #791
        admission-uniformity block truncates ``prefix_indices`` on PP0 (from
        the guard's clamped candidate) and on every downstream rank (from PP0's
        decision), and NEITHER touched ``cache_protected_len``.

        WHY THAT IS NOT MERELY UNTIDY. ``cache_protected_len`` means "how many
        LEADING rows of this request's KV the TREE owns". ``init_next_round_
        input`` had just set it equal to ``len(prefix_indices)``; truncating
        one and not the other leaves the request claiming more tree-owned rows
        than it holds. That surplus is the SAFE direction for
        ``_insert_helper``'s duplicate free (a larger ``dup_start`` frees less)
        -- which is how it stayed invisible -- and it is the DANGEROUS
        direction for ``cache_finished_req``'s truncate branch:

            free_start = max(effective_cache_len, req.cache_protected_len)
            free(kv_indices[free_start:])   # starts ABOVE the gap
            ...                             # insert covers only up to ecl

        so when ``cache_protected_len > effective_cache_len`` the rows in
        ``[effective_cache_len, cache_protected_len)`` are neither freed nor
        inserted, and belong to nobody afterwards. That interval is #935's
        per-request row leak. The gap is the root and must not be able to leak
        whatever the value is; this closes one of the two PRODUCERS that make
        it reachable (the other is the #928 refusal re-prefill).

        MIN, NEVER ASSIGN: this may only LOWER the claim. A request whose
        protected length was already below ``told`` owns exactly that many, and
        raising it here would invent protection the tree never granted.

        #958 AND THE EXECUTED GEOMETRY GOES WITH IT -- the third sibling, and
        the one that cost the boot. ``pp_admission_congruence._executed_extent``
        builds PP0's production offer out of ``extend_range.start`` ALONE, and
        its docstring rests on ``extend_range.start == len(prefix_indices)``
        being ONE quantity with one expression. Truncating the prefix and
        leaving ``extend_range`` behind breaks exactly that: the offer is then
        read off a geometry describing a pass that no longer exists, and it is
        read as though it were a fresh measurement.

        MEASURED (window-955-boot, boot_943bx_27bcb4884f_0828_024615.log). The
        recompute terminator spent itself on rid ``a6132c5de5...``, discarding
        8192 tokens as a NAMED double prefill -- and the next 336 offers for
        that rid were ``told=8192``, unchanged. Not one genuine ``told=0``
        offer exists in the boot. Since ``reconcile_pp_admission_decision``
        admits ``told <= 0`` UNCONDITIONALLY (the one exit that survives a
        downstream lookup miss), the escape was built to reach that branch and
        never delivered it: PP1 retracted every pass, and PP2 voided 512
        CONSECUTIVE passes into the ``#801-spin`` refusal.

        A ZERO-LENGTH RANGE AT THE NEW PREFIX, NOT THE OLD ``end`` AND NOT
        ``None``. #958 refused ``Range(told, old_end)`` and that refusal
        stands: keeping ``end`` would INVENT a pass -- the discarded tokens
        would have to be computed in this chunk, inflating it past the budget
        the adder already decided, instr21 in the other direction (a report
        naming rows no rank will run). ``Range(told, told)`` invents nothing
        either: it says ZERO ROWS at the prefix that now exists, which is
        exactly what a request whose premise was just dropped will run this
        pass. It is not a new state -- ``_park_chunked_prefill_chunk`` writes
        ``Range(start, start)`` for the same purpose
        (scheduler_pp_mixin.py:592), ``add_chunked_req``'s #679 park writes it
        (schedule_policy.py:1434-1436), and ``_executed_extent`` declares the
        shape first-class ("A ZERO-LENGTH RANGE IS REPORTED, NOT SUPPRESSED").

        #961 AND WHY ``None`` COULD NOT STAY. The sentence that used to stand
        here -- "``reset_for_retract`` already sets ``extend_range = None``
        (below), every reader on this path is already None-safe" -- was FALSE,
        and it was refuted on metal 25 s into window-958-boot's first boot:

            scheduler.py:7010, in get_next_batch_to_run
              if self.chunked_req.extend_range.end > len(...prefix_indices):
            AttributeError: 'NoneType' object has no attribute 'end'

        The equivocation was on "this path". ``reset_for_retract``'s None is
        safe because the two disposal sites that see it CLEAR the resident
        pointer (scheduler_pp_mixin.py:6061-6075 and :7222-7236); those sites
        are upstream of this producer and are not on its path. Three readers
        take the geometry off the RESIDENT cross-round pointer behind a guard
        that tests the REQUEST and not the geometry -- scheduler.py:7010,
        scheduler.py:9424-9428 and ``_compute_chunked_req_next_prompt_token``
        (below) -- and none of them was None-safe.

        AND THE ADDER IS NOT THE RE-DERIVER IT WAS TAKEN FOR. The old argument
        was that the truncation runs at "#946 ACT HERE, AND ONLY HERE", just
        before ``add_chunked_req`` re-derives. #948 then MOVED the act to
        ``pp_apply_dead_premise_anywhere`` (called from
        scheduler_pp_mixin.py:2159), for a measured reason: the old site was
        entered ~6 times while 9471 passes voided. Two statements later, :2161
        calls ``get_next_batch_to_run``. The legality argument was a property
        of the old neighbourhood and did not travel with the call, and three
        further producers reach a reader without an adder in between -- the
        #906 seam refusal (scheduler.py:8916), ``add_chunked_req``'s
        hybrid-SWA zero-budget return (schedule_policy.py:1396), and the #791
        clamp sites when the admission loop breaks on ``NO_TOKEN``. Re-deriving
        HERE closes all four at the writer instead of one branch per boot.

        SO THE REQUEST STILL GETS EXACTLY THE THREE HONEST EXITS, and the
        offer still moves: ``_executed_extent`` now reads ``(told, 0)`` instead
        of refusing to read at all, so PP0 offers ``told`` -- 0 after the
        terminator -- which is the value ``reconcile_pp_admission_decision``
        admits UNCONDITIONALLY. The fourth exit stays gone: there is no silent,
        unbounded re-offer of a geometry the request no longer has, because the
        geometry is re-derived rather than left behind. What is also gone is
        the reliance on ``PPScheduleRefused`` /
        ``require_executed_geometry=True`` as the net for this case. That net
        fired 0 times on metal while the unguarded dereference killed the
        process, because it iterates ``can_run_list`` and a resident
        continuation the adder did not add is not in it. It is not made
        reachable here; it is made unnecessary, because the state it was meant
        to name can no longer be produced.

        ONLY WHEN THE PREFIX ACTUALLY MOVES. A no-op truncation
        (``told >= len(prefix_indices)``) leaves a valid geometry valid --
        clearing it there would void healthy passes for nothing.
        """
        told = int(told)
        if told < len(self.prefix_indices):
            self.prefix_indices = self.prefix_indices[:told]
            # #958/#961: the geometry was DERIVED from the prefix that just
            # moved, so it is now a stale reading rather than a report. It is
            # RE-DERIVED here, to the only honest value available at this
            # point, rather than nulled and left for a caller to fix up.
            if getattr(self, "extend_range", None) is not None:
                self.extend_range = Range(told, told)
            # #965 THE WHOLE CO-DERIVED GROUP, AS A GROUP.
            #
            # `init_next_round_input` reads the tree ONCE and unpacks that one
            # `match_result` into EIGHT attributes in a single tuple assignment
            # (the statement above `if match_result.cache_protected_len is not
            # None`). They are one reading of one geometry wearing eight names,
            # and the sentence four lines up -- "the geometry was DERIVED from
            # the prefix that just moved, so it is now a stale reading rather
            # than a report" -- is true, word for word, of every one of them.
            #
            # It was paid for one field at a time, twice, each found by a boot:
            # #930 carried `cache_protected_len` through this truncation, #958
            # added `extend_range`. Clearing the rest here, together, is what
            # stops a third window buying the same lesson.
            #
            # WHAT THE STALE READINGS DO, at `PrefillAdder.add_one_req` twelve
            # lines further down the path (`scheduler.py` derives them, both
            # arms of the `pp_size > 1` fork call THIS method, then
            # `adder.add_one_req` reads them, with no `match_prefix` between):
            #   * `real_input_tokens = cand_extend_input_len -
            #     req.host_hit_length` subtracts a host hit that is no longer
            #     part of this prefix, under-counting the input against the
            #     budget gates;
            #   * `needs_host_load_back()` is still true, so `init_load_back`
            #     runs on a stale `best_match_node`/`host_hit_length` and does
            #     `prefix_indices = torch.cat([prefix_indices, new_indices])`
            #     -- leaving `[0, told)` and then `[L_dev, L_dev+H)` with a HOLE
            #     between them, while `prepare_for_extend` sizes the
            #     cross-stage tensor off `len(prefix_indices)` as though it
            #     were contiguous. That is the silently-wrong-context class;
            #   * that same branch then re-raises `cache_protected_len` to the
            #     grown length, undoing #930 nine lines after it was applied.
            #
            # `last_node` IS NOT CLEARED, and the asymmetry is the point. It is
            # not a reading, it is a RESOURCE HANDLE: `cache_unfinished_req`
            # took `inc_lock_ref` on it and this attribute is the only surviving
            # reference to that ref. Nulling it would leak the lock ref and make
            # the node permanently unevictable -- a defect that already exists
            # on the PP void path and must not be manufactured here as well. It
            # is invalidated by RELEASE, which is not this method's job.
            self.last_host_node = None
            self.best_match_node = None
            self.host_hit_length = 0
            self.swa_host_hit_length = 0
            self.mamba_host_hit_length = 0
            self.mamba_branching_seqlen = None
        if self.cache_protected_len > told:
            self.cache_protected_len = told

    def reset_for_retract(self):
        # Increment retraction count before resetting other state. We should not reset this
        # since we are tracking the total number of retractions for each request.
        self.retraction_count += 1

        # #856: RECORD WHAT WAS ALREADY COMPUTED, BEFORE THE FIELDS THAT SAY SO
        # ARE CLEARED THREE LINES DOWN.
        #
        # A retracted request goes back to the waiting queue and its FULL
        # prompt reappears in `Scheduler._pending_prefill_tokens`, which is the
        # quantity the phase policy compares against the break-even N. But N is
        # priced from the UNCACHED prefill throughputs, and these tokens are
        # not uncached: their KV was computed, and under the phase-flip fence
        # it has been persisted to the canonical store, so re-prefilling them
        # is a cache read in EITHER layout. Tokens that cost the same in both
        # layouts cannot make one layout cheaper than the other, so they must
        # not enter that comparison at all.
        #
        # #731 measured what happens when this figure is inflated across a
        # cutover -- "51,369 -> 102,307 tokens ... six cutovers, nothing
        # served" -- from a different cause (double-counting). This is the same
        # shape by a route that dedup cannot catch, because nothing is counted
        # twice here; it is counted ONCE at a price that is wrong.
        #
        # Captured here rather than looked up later because the lookup is
        # impossible later: `prefix_indices`, `num_matched_prefix_tokens` and
        # `extend_range` are all cleared below. It is also FREE here, where a
        # `match_prefix` walk per queued request per policy round would not be.
        #
        # `extend_range.end` is the fill boundary, the same notion of
        # "computed" the pending counter itself uses for a chunked remainder.
        # A request that has emitted output finished its prefill by
        # construction, so its whole prompt is covered.
        if self.output_ids:
            computed = len(self.origin_input_ids)
        elif self.extend_range is not None:
            computed = int(self.extend_range.end)
        else:
            computed = 0
        self.cached_prompt_tokens_at_retract = max(
            0, min(int(computed), len(self.origin_input_ids))
        )
        # #861f: THE EXISTENCE STAMP, beside the economics credit above.
        #
        # `cached_prompt_tokens_at_retract` answers "how much of this prompt
        # will be cheap to redo" -- an ECONOMICS question, and it credits the
        # ENTIRE prompt to any request that produced even one output token
        # (see the branch above). W37-E showed why that cannot double as an
        # existence signal: seven retracted requests each carried n>=1 output
        # tokens, were credited their whole prompt, and every backlog counter
        # therefore read 0 while they sat unservable in the queue for 198 s.
        #
        # This field answers the different question: does this request still
        # need a PREFILL PASS before it can decode again? After a retraction
        # the answer is always yes -- the seam drops the prefix tree, so the
        # KV must be re-materialised whatever the credit says. Cleared by the
        # admission path when the pass has actually run.
        #
        # Declared in cutover_participants.MUTATED_STATE as DURING_CUTOVER:
        # it is written BY the retraction and is only meaningful afterwards.
        self.needs_prefill_pass = True

        self.prefix_indices = torch.empty((0,), dtype=torch.int64)
        self.routed_experts = None
        self.indexer_topk = None
        self.last_node = None
        self.cache_protected_len = 0
        self.num_matched_prefix_tokens = 0
        self.swa_uuid_for_lock = None
        self.swa_prefix_lock_released = False
        self.extend_range = None
        # #987: the carried fill tail is a statement about ONE pass's seam
        # divergence, and a retraction ends that pass. Left standing it would
        # survive into a re-prefill whose `output_ids` may be shorter (the
        # input_embeds branch below empties them outright), where the deficit
        # it implies names no seam at all. `adopt_carried_fill` re-establishes
        # it on the next pass that still needs it, from the decision, which is
        # the only place the fact is authoritative.
        self.pp_carried_fill_len = None
        self.pp_carried_fill_tail = ()
        self.dllm_initialized = False
        self.is_retracted = True
        self.retracted_stain = True
        self.input_token_logprobs = None
        self.temp_input_top_logprobs_val = None
        self.temp_input_top_logprobs_idx = None
        self.temp_input_token_ids_logprobs_val = None
        self.temp_input_token_ids_logprobs_idx = None
        self.inflight_middle_chunks = 0
        self.mamba_pool_idx = None
        # #991: the stamp describes the slot, so it dies with the slot.
        self.mamba_slot_acquired_this_admission = False
        self.mamba_ping_pong_track_buffer = None
        self.mamba_next_track_idx = None
        self.mamba_last_track_seqlen = None
        self.mamba_branching_seqlen = None
        self.mamba_cow_src_index = None
        self.mamba_needs_clear = False
        self.already_computed = 0
        self.kv_allocated_len = 0
        self.kv_committed_len = 0
        self._kvc_src = "reset_for_retract"  # #969L writer stamp
        self.kv_committed_freed = False
        self.kv_overallocated_freed = False
        self.swa_evicted_seqlen = 0
        self.extend_batch_idx = 0
        self.decode_batch_idx = 0

        # When using input_embeds, we cannot easily mix the original input embeddings
        # with the newly generated output token IDs during re-prefill of retracted request.
        # output_ids will have no use, but will lead to wrong size cache indexes.
        # Therefore, we discard the generated output_ids and restart prefill and generation
        # to ensure shape consistency in KV cache.
        if self.input_embeds is not None:
            self.output_ids = array("q")

    #: #783: the mamba half of the copy, when the pool does not do it itself.
    #:
    #: SHAPE DECISION, MADE DELIBERATELY. `HybridLinearKVPool.get_cpu_copy`
    #: returns a TUPLE `(kv_cpu, mamba_cpu)` and `UnifiedSWAKVPool` a DICT
    #: `{"full", "swa"}` -- two shapes for one role, which is the next naming
    #: error waiting to happen. This does NOT add a third, and does NOT unify
    #: them: `Req` never inspects the allocator's payload, it only hands the
    #: same object back to `load_cpu_copy`. The mamba copy therefore lives in
    #: its own attribute, which works for every pool regardless of shape and
    #: keeps `Req` independent of a decision that belongs to the pools.
    #: Unifying the two payload shapes is a real cleanup and a separate one; it
    #: touches every caller of both pools and is filed rather than smuggled in
    #: here.
    mamba_state_cpu: Optional[object] = None

    #: #783: how many token rows `kv_cache_cpu` actually covers. None = no copy.
    #: RECORDED rather than re-derived, because `seqlen` is a LOGICAL length
    #: (`len(origin_input_ids) + len(output_ids)`) and says nothing about how
    #: much of `req_to_token` has been written. W38-A crashed on exactly that
    #: gap: a restore whose extent had grown walked off the end of the saved
    #: chunk list (memory_pool.py:3295). Every pool-level `load_cpu_copy` just
    #: forwards the indices it is handed, so there is NO backstop below this
    #: class -- the guarantee has to be established here.
    kv_cache_cpu_extent: Optional[int] = None

    #: #861c: WHICH per-layer layout `kv_cache_cpu` was taken from. None = the
    #: pool could not say (see `BaseTokenToKVPoolAllocator.cpu_copy_layout`).
    #:
    #: The sibling of `kv_cache_cpu_extent`, one axis over. That field records
    #: how many ROWS the copy covers; this one records how many LAYERS, and
    #: which. W40 crashed on the axis that was not recorded: the copy was taken
    #: from the PP-stage pool (18 layers) and applied to the TP pool (64), and
    #: the extent contract passed because the ROW count had not changed.
    #:
    #: NEVER PARSED HERE, only compared for equality -- the same discipline that
    #: keeps `Req` independent of the copy payload's shape (see the note on
    #: `mamba_state_cpu` above). The pools decide what a layout IS.
    kv_cache_cpu_layout: Optional[object] = None

    #: #861c: and the same for the mamba half, when `Req` owns that copy. It is
    #: a separate field because it is a separate copy taken from a separate pool
    #: -- the flip can change the mamba-layer split independently of the KV one.
    mamba_state_cpu_layout: Optional[object] = None

    def _mamba_cpu_copy_is_mine(self, token_to_kv_pool_allocator) -> bool:
        """#783: does THIS caller own the mamba copy, or does the pool?

        Exactly one of the two moves it -- Ein-Job-ein-Mover, enforced by a
        declaration rather than promised by a comment. The pool that owns a
        mamba pool declares `supports_mamba_cpu_copy()` and keeps its mover;
        this path covers the pools that have no mamba pool to delegate to.

        NO getattr DEFAULT HERE (#606/#608). Since `BaseTokenToKVPoolAllocator`
        carries the method, every allocator that can be passed as
        `token_to_kv_pool_allocator` ANSWERS: TokenToKVPoolAllocator,
        PagedTokenToKVPoolAllocator, SWATokenToKVPoolAllocator (and its
        PureSWA subclass), HiSparseTokenToKVPoolAllocator,
        DeepSeekV4HiSparseTokenToKVPoolAllocator, MultiEndedAllocator,
        UnifiedMambaTokenToKVPoolAllocator, UnifiedSWATokenToKVPoolAllocator --
        all of them inherit it. The only allocator classes in this tree that do
        NOT are `MambaSlotAllocator` and `UnifiedMambaSlotAllocator`, which
        allocate mamba SLOTS and are never passed here.

        So a missing attribute would be a real defect, and an AttributeError
        saying so is worth more than a silent False that would quietly make
        `Req` copy state the pool had already copied.
        """
        if self.mamba_pool_idx is None:
            return False
        return not bool(token_to_kv_pool_allocator.supports_mamba_cpu_copy())

    def offload_kv_cache(self, req_to_token_pool, token_to_kv_pool_allocator):
        token_indices = req_to_token_pool.req_to_token[
            self.req_pool_idx, : self.seqlen - 1
        ]
        # Copies over both the kv cache and mamba state if available
        self.kv_cache_cpu = token_to_kv_pool_allocator.get_cpu_copy(
            token_indices, mamba_indices=self.mamba_pool_idx
        )
        # #783: say what was covered, so a later restore can tell drift from
        # agreement instead of discovering it as an IndexError.
        self.kv_cache_cpu_extent = int(token_indices.numel())
        # #861c: say WHICH per-layer layout it was taken from, on the same
        # principle and for the axis the extent does not cover. Asked of the
        # allocator rather than derived from the payload: the payload's shape is
        # the pool's business (tuple, dict, list of lists), and inspecting it
        # here is exactly the coupling the `mamba_state_cpu` note refuses.
        self.kv_cache_cpu_layout = token_to_kv_pool_allocator.cpu_copy_layout()
        # #783: and when the pool declares it does NOT move mamba, move it here.
        # Without this the comment above was false on this rig's pool: the KV
        # came back and the GDN state did not.
        self.mamba_state_cpu = None
        self.mamba_state_cpu_layout = None
        if self._mamba_cpu_copy_is_mine(token_to_kv_pool_allocator):
            mamba_pool = getattr(req_to_token_pool, "mamba_pool", None)
            if mamba_pool is not None:
                translate = getattr(
                    req_to_token_pool, "translate_mamba_indices", lambda ids: ids
                )
                self.mamba_state_cpu = mamba_pool.get_cpu_copy(
                    translate(self.mamba_pool_idx)
                )
                # #861c: the mamba copy is a second copy from a second pool, so
                # it needs its own layout stamp. `MambaPool` is not a `KVCache`
                # and does not inherit the allocator hop.
                layout_fn = getattr(mamba_pool, "cpu_copy_layout", None)
                self.mamba_state_cpu_layout = (
                    layout_fn() if layout_fn is not None else None
                )

    def load_kv_cache(self, req_to_token_pool, token_to_kv_pool_allocator):
        token_indices = req_to_token_pool.req_to_token[
            self.req_pool_idx, : self.seqlen - 1
        ]
        # Loads both the kv cache and mamba state if exists
        token_to_kv_pool_allocator.load_cpu_copy(
            self.kv_cache_cpu, token_indices, mamba_indices=self.mamba_pool_idx
        )
        if self.mamba_state_cpu is not None:
            mamba_pool = getattr(req_to_token_pool, "mamba_pool", None)
            if mamba_pool is not None:
                translate = getattr(
                    req_to_token_pool, "translate_mamba_indices", lambda ids: ids
                )
                mamba_pool.load_cpu_copy(
                    self.mamba_state_cpu, translate(self.mamba_pool_idx)
                )
            self.mamba_state_cpu = None
            self.mamba_state_cpu_layout = None
        del self.kv_cache_cpu
        self.kv_cache_cpu_layout = None

    def build_rebootstrap_payload(self) -> dict:
        """Build the prefill ``/generate`` payload that asks the original prefill
        worker to recompute this request's prefix KV under the current weights
        (PD true-retraction rebootstrap).

        ``input_ids`` are coerced to plain ``int`` so the payload is always
        JSON-serializable even when ``origin_input_ids``/``output_ids`` hold
        numpy scalars. The sampling-param allow-list forces ``max_new_tokens=1``
        and drops stop/grammar/min_new_tokens so the recompute only re-derives
        the prefix KV and samples a single handoff token. The already-emitted
        boundary token is replayed on the *decode* side (the transfer commit
        overrides the sampled handoff with it), so it is intentionally not sent
        to the prefill here.
        """
        # TODO: multi-modal requests are not supported here. The payload only
        # carries token ``input_ids`` and drops any image/audio/video inputs, so
        # the rebootstrap recompute would not reproduce the original prefix KV
        # for multi-modal requests. Add multi-modal support before enabling it.
        sp = self.sampling_params
        return {
            "input_ids": [int(x) for x in self.origin_input_ids]
            + [int(x) for x in self.output_ids],
            "sampling_params": {
                "max_new_tokens": 1,
                "temperature": sp.temperature,
                "top_p": sp.top_p,
                "top_k": sp.top_k,
                "min_p": sp.min_p,
                "frequency_penalty": sp.frequency_penalty,
                "presence_penalty": sp.presence_penalty,
                "repetition_penalty": sp.repetition_penalty,
                "ignore_eos": sp.ignore_eos,
                "skip_special_tokens": sp.skip_special_tokens,
                "spaces_between_special_tokens": sp.spaces_between_special_tokens,
                "no_stop_trim": sp.no_stop_trim,
            },
            "return_logprob": False,
            "stream": False,
            "rid": self.rid,
            "bootstrap_host": self.bootstrap_host,
            "bootstrap_port": self.bootstrap_port,
            "bootstrap_room": self.bootstrap_room,
            "priority": self.priority,
            "extra_key": self.extra_key,
            "routing_key": self.routing_key,
            "disagg_prefill_dp_rank": self.disagg_prefill_dp_rank,
        }

    def log_time_stats(self):
        # If overlap schedule, we schedule one decode batch ahead so this gets called twice.
        if self.has_log_time_stats:
            return

        bootstrap_info = (
            f", bootstrap_room={self.bootstrap_room}"
            if self.bootstrap_room is not None
            else ""
        )
        prefix = (
            f"ReqTimeStats("
            f"rid={self.rid}{bootstrap_info}, "
            f"input_len={len(self.origin_input_ids)}, "
            f"cached_input_len={self.cached_tokens}, "
            f"output_len={len(self.output_ids)}, "
            f"attempts={self.prefill_attempt_count}, "
            f"type={self.time_stats.disagg_mode_str()})"
        )
        logger.info(f"{prefix}: {self.time_stats.convert_to_duration()}")
        self.has_log_time_stats = True

    def set_finish_with_abort(self, error_msg: str):
        if get_parallel().tp_rank == 0:
            logger.error(f"{error_msg}, {self.rid=}")
        self.multimodal_inputs = None
        self.grammar = None
        self.origin_input_ids = array(
            "q", [0]
        )  # set it to one token to skip the long prefill
        self.return_logprob = False
        self.logprob_start_len = -1
        self.to_finish = FINISH_ABORT(
            error_msg, HTTPStatus.BAD_REQUEST, "BadRequestError"
        )

    def update_reasoning_tokens(self, token_id, think_end_id):
        if self._is_reasoning_over:
            return

        if not isinstance(token_id, list):
            token_id = [token_id]

        try:
            end_pos = token_id.index(think_end_id)
            self.reasoning_tokens += end_pos + 1
            self._is_reasoning_over = True
        except ValueError:
            self.reasoning_tokens += len(token_id)

    def __repr__(self):
        return (
            f"Req(rid={self.rid}, "
            f"input_ids={self.origin_input_ids}, output_ids={self.output_ids}, "
            f"{self.grammar=}, "
            f"{self.sampling_params=})"
        )


class _MambaRadixCacheV2TrackEntry(NamedTuple):
    track_mask: bool
    track_index: int
    track_seqlen: int


def set_mamba_track_indices_from_reqs(batch):
    """Build mamba_track_indices from req objects (authoritative source)."""
    req_to_token_pool = batch.req_to_token_pool
    all_buffers = req_to_token_pool.req_index_to_mamba_ping_pong_track_buffer_mapping[
        batch.req_pool_indices
    ]  # (bs, ping_pong_size), int64, on device
    idx = (
        torch.tensor(
            # A req retracted mid-flight under overlap scheduling has its mamba
            # ping-pong track nulled (see release-for-retract: mamba_pool_idx /
            # mamba_ping_pong_track_buffer / mamba_next_track_idx -> None) while
            # still present in this in-flight verify batch. Its verify result is
            # discarded, so column 0 is a harmless placeholder; legitimate reqs
            # always carry a real index, so this default never masks a real bug.
            [
                (
                    req.mamba_next_track_idx
                    if req.mamba_next_track_idx is not None
                    else 0
                )
                for req in batch.reqs
            ],
            dtype=torch.int64,
            pin_memory=True,
        )
        .unsqueeze(1)
        .to(device=all_buffers.device, non_blocking=True)
    )
    batch.mamba_track_indices = (
        torch.gather(all_buffers, 1, idx).squeeze(1).to(torch.int64)
    )


def release_req(
    *,
    req: Req,
    remaing_req_count: int,
    server_args: ServerArgs,
    req_to_token_pool: ReqToTokenPool,
    token_to_kv_pool_allocator: BaseTokenToKVPoolAllocator,
    tree_cache: BasePrefixCache,
    hisparse_coordinator: Optional[HiSparseCoordinator],
    offload_kv: bool = True,
    copy_state: bool = False,
    retain: bool = False,
) -> None:
    if hisparse_coordinator is not None and not req.finished():
        hisparse_coordinator.retract_req(req)

    # #783 half 1: the #856 cutover copies its state out before letting go.
    # Default False, set at EXACTLY ONE site (build_cutover_release._retract);
    # `retract_all`'s only other caller is the decode-pressure path, whose rate
    # is load-dependent and outside the flip-cadence host budget. Not a flag --
    # the condition is structural. Routed through `seam_copy_state` so a
    # mid-chunk request is DECLINED rather than copied at an extent that cannot
    # be restored. Runs before `release_kv_cache` below, while the rows still
    # hold live bytes, and transfers no ownership (`is_insert` stays False).
    if copy_state:
        seam_copy_state(req, req_to_token_pool, token_to_kv_pool_allocator)

    # In decode disaggregation the retracted KV is offloaded to host so it can be
    # restored later without recompute (see resume_retracted_reqs/load_kv_cache).
    # Callers that will recompute the KV instead (PD true-retraction rebootstrap)
    # pass offload_kv=False to skip the wasteful device->host copy.
    #
    # #920 SIBLING, NAMED RATHER THAN GUESSED. This branch reaches the SAME
    # `Req.offload_kv_cache` with the SAME raw `req_to_token` slice as the seam
    # copy above, so under a layout whose pool is indexed by compacted rows --
    # this fork's TP stack, see `seam_copy_addresses_the_bound_pool`
    # (phase_flip_runtime.py) -- it carries the identical defect: a global slot
    # id handed to a pool that does not store it at that row. It is NOT gated
    # here because the gate's input is the resident layout, which is a property
    # of the scheduler and is not reachable from this frame, and because no
    # measured specimen exists for it (this branch needs
    # `disaggregation_mode == "decode"`, which no boot in the #918/#920 family
    # ran). Gating it on a guess would put an ungrounded refusal on a live
    # decode-disagg path; recording it is what keeps it from being rediscovered
    # as a new finding.
    if server_args.disaggregation_mode == "decode" and offload_kv:
        req.offload_kv_cache(req_to_token_pool, token_to_kv_pool_allocator)
    # TODO (csy): for preempted requests, we may want to insert into the tree
    # #969D RETAIN AT THE CUTOVER. `is_insert=False` is CORRECT for upstream's
    # retraction, which is a deliberate discard under memory pressure: the
    # request will be re-prefilled and throwing the span away is the point.
    #
    # Our cutover uses the same call for the OPPOSITE purpose. The user design
    # is drain -> zero everything -> re-admit by HiCache prefix READ, and that
    # read can only hit what reached the store. With `is_insert=False` the
    # computed prefix is freed without ever entering the tree, so it is never
    # hashed, never staged by the fence, never in the store -- and the
    # re-admitted request's prefetch fetches 0 tokens and re-prefills in full.
    #
    # MEASURED, boot_969cutflip_92beb67982_0829_145759.log: across 102
    # post-retract fences (34 cutovers x 3 ranks) `staged>0` NEVER ONCE -- 51
    # found `eligible=0` (nothing in the tree at all) and 51 found only
    # already-staged nodes. The fence was never the defect; it had nothing to
    # write, because this line had already thrown it away. Downstream:
    # 132 `#937 STALE PREFETCH INSERT REFUSED ... 0 token(s) fetched`,
    # `#cached-token > 0` on 0 of 335 prefill lines, and 141 of 148
    # re-admissions matching an empty tree (#969B).
    #
    # The cutover ALREADY stamps `FORCE_HOST_WRITE_THROUGH_ATTR` on every
    # retracted request (phase_flip_runtime.py:1934) -- an intent that had
    # nothing to act on, because nothing was inserted for write-through to
    # carry.
    #
    # `retain` is threaded from the cutover only; every other caller keeps
    # upstream's discard semantics unchanged.
    release_kv_cache(req, tree_cache, is_insert=bool(retain))

    if not retain:
        # Evicting is right when the retraction is a discard under pressure. At
        # the cutover it would free exactly the span we just retained, before
        # the fence that has to stage it -- and the pools are reset wholesale
        # moments later anyway.
        num_tokens = remaing_req_count * envs.SGLANG_RETRACT_DECODE_STEPS.get()
        evict_from_tree_cache(tree_cache, num_tokens)

    req.reset_for_retract()


#: #783: seam state-transfer counters. AFFIRMATIVE REPORTING -- these exist so
#: "did the restore happen" is a grep and not an inference from #cached-token.
#: `refused` and `declined` are the numbers that matter most: a contract never
#: exercised in the failure direction is untested, and `declined` doubles as the
#: measurement of how often a mid-chunk request meets the seam at all -- i.e. it
#: MEASURES the PS3 question instead of guessing it.
# `refused` is the #783 extent (ROW) refusal; `refused_layout` is the #861c
# per-LAYER one. Counted apart on purpose: they diagnose different things. An
# extent refusal says a request grew across the seam; a layout refusal says the
# seam carry is structurally impossible in that direction, i.e. every flip loses
# its prefixes. One number for both would let the second hide inside the first.
# #875d: `carried` is the layout drift that was ANSWERED rather than refused --
# a copy whose global layers cover the destination's, re-selected onto them
# rank-locally. It is counted apart from `restored` for the reason `refused` and
# `refused_layout` are counted apart: folded into `restored` it would be
# indistinguishable from a same-layout restore, and the one thing an operator
# needs to read off these numbers is which flips keep their prefixes and by
# which route. `carried + refused_layout` is every flip that crossed a geometry.
#: #998 reader-side invariant probe: rid -> (start, end, len_prefix,
#: len_input, break). `break` is `start - len(prefix_indices)`; 0 means the
#: invariant holds, -1 means unreadable (a DISTINCT sentinel, because 0 is a
#: legitimate value). Emitted from `pp_ring_note`, which runs independently
#: of this probe.
_998_LAST: dict = {}
_998_SEEN = [0]
_998_BREAKS = [0]

_SEAM_STATE_COUNTS = {
    "copied": 0,
    "declined": 0,
    "restored": 0,
    "refused": 0,
    "refused_layout": 0,
    "carried": 0,
    # #913: counted apart from `declined` for the reason `refused_layout` is
    # counted apart from `refused` -- they diagnose different things. A
    # `declined` says the request was mid-chunk, which is normal traffic; a
    # `declined_unmapped` says the backing dial released pages under a live
    # row, which is a defect upstream of this file. Folded into one number the
    # second would be invisible inside the first at exactly the rate the first
    # is common.
    "declined_unmapped": 0,
    # #916: and apart from THAT one again, for the same reason one more time.
    # `declined_unmapped` is a verdict about the ids -- they were read and they
    # sit above the backing. `declined_unreadable` is the absence of a verdict:
    # the device would not answer at all, because the context was already
    # faulted when the copy was requested (0826 rerun boot #2, 21:53:36). One
    # says the dial released a page under a live row; the other says something
    # else had already gone wrong and this path is downstream of it. Reading
    # the second as the first would send the next window hunting the dial.
    "declined_unreadable": 0,
}


def _seam_extent_of(req: Req) -> int:
    """The number of token rows a copy of this request would cover."""
    return int(req.seqlen) - 1


def _seam_prefill_is_complete(req: Req) -> bool:
    """#783: is this request's row actually filled to its logical length?

    `Req.seqlen` is LOGICAL (`len(origin_input_ids) + len(output_ids)`); the row
    is filled only to `kv_allocated_len` (== `extend_range.end`), which is
    strictly less while the prompt is still being chunk-prefilled. Indexing by
    `seqlen - 1` is therefore only sound once those agree.
    """
    allocated = getattr(req, "kv_allocated_len", None)
    if allocated is None:
        return False
    return int(allocated) >= _seam_extent_of(req)


def seam_copy_state(req, req_to_token_pool, token_to_kv_pool_allocator) -> bool:
    """#783 half 1: copy this request's state out at the cutover, or decline.

    DECLINES a request whose prefill is still chunked. Such a request has no
    well-defined full extent, and the tree has already settled what to do about
    that: `kv_session_offload` refuses chunked admission outright rather than
    restoring into it (":445 return False  # would be CHUNKED -> needs PS3";
    the assert at :4496 names "PS3 (host-prefix extend read)"). PS3 is
    unimplemented -- four mentions, all comments. So mid-chunk is declined here
    too, loudly and counted, rather than half-built in passing.

    Declining costs a recompute of work that was unfinished anyway; a
    decode-phase resident, which is the population the cutover actually
    retracts, loses real session state and IS covered.
    """
    if not _seam_prefill_is_complete(req):
        _SEAM_STATE_COUNTS["declined"] += 1
        n = _SEAM_STATE_COUNTS["declined"]
        if n <= 3 or n % 100 == 0:
            logger.info(
                "%s SEAM COPY DECLINED rid=%s: prefill still chunked "
                "(allocated=%s, needs=%d), so there is no full extent to copy. "
                "Its tokens are recomputed after the flip. occurrence=%d",
                SEAM_STATE_PREFIX,
                getattr(req, "rid", None),
                getattr(req, "kv_allocated_len", None),
                _seam_extent_of(req),
                n,
            )
        return False
    # #913: DECLINE A ROW WHOSE PAGE IS GONE, do not read it.
    #
    # The backing dial releases pages under ids that live requests still hold
    # (`runtime_set_backing_tokens` states "rows above n are dead the moment
    # size is n", which is false while a resident holds one), and this copy is
    # the consumer that turns that into a CUDA illegal memory access: R7 of the
    # 0826 window, 18:27:14Z, PP2, live high-water 122898 against a backing of
    # 114688, the whole instance down with a traceback naming `synchronize()`.
    #
    # CAUGHT HERE AND NOT LOWER because this is the frame that owns the
    # response. `check_cpu_copy_rows` can only refuse; only the seam knows that
    # a refused copy is survivable -- it is the DECLINE path three lines above,
    # already built, already counted, whose cost is a recompute. Letting the
    # refusal propagate would replace an unrecoverable rank death with a
    # recoverable-in-principle one that still kills the flip, which is not the
    # improvement it looks like: `release_residents_for_cutover` is past the
    # no-return point and its caller has no abort left.
    #
    # NARROW ON PURPOSE. Only `CpuCopyUnmappedRows` is caught. A plain
    # ValueError from the same guard means the id addresses NO pool -- the
    # #783b defect -- and must still surface loudly; swallowing it here would
    # hide an addressing bug behind a counter that reads as normal traffic.
    try:
        req.offload_kv_cache(req_to_token_pool, token_to_kv_pool_allocator)
    except CpuCopyUnmappedRows as refusal:
        # #916: the unreadable-ids refusal is a SUBCLASS of this one on purpose
        # -- one `except` keeps the decline path unforgettable -- but it is
        # counted on its own line so the two cannot be read as one population.
        key = (
            "declined_unreadable"
            if isinstance(refusal, CpuCopyIdsUnreadable)
            else "declined_unmapped"
        )
        _SEAM_STATE_COUNTS[key] += 1
        n = _SEAM_STATE_COUNTS[key]
        if n <= 3 or n % 100 == 0:
            logger.error(
                "%s SEAM COPY DECLINED (%s) rid=%s: %s %s Declined; its tokens "
                "are recomputed after the flip. occurrence=%d",
                SEAM_STATE_PREFIX,
                "IDS UNREADABLE" if key == "declined_unreadable" else "UNMAPPED",
                getattr(req, "rid", None),
                refusal,
                (
                    "Nothing is claimed about these ids -- the decline is about "
                    "the device, not about them (#916)."
                    if key == "declined_unreadable"
                    else "The rows were minted at a larger backing and a dial "
                    "shrink released their pages while this request still held "
                    "them, so the copy would read unmapped device memory."
                ),
                n,
            )
        # Leave nothing half-taken: a partially populated copy would be applied
        # at restore against an extent it does not cover.
        req.kv_cache_cpu = None
        req.kv_cache_cpu_extent = None
        req.kv_cache_cpu_layout = None
        req.mamba_state_cpu = None
        req.mamba_state_cpu_layout = None
        return False
    _SEAM_STATE_COUNTS["copied"] += 1
    return True


def restore_seam_state(req, req_to_token_pool, token_to_kv_pool_allocator) -> bool:
    """#783 half 2: put back what the cutover copied, or REFUSE on extent drift.

    THE CONTRACT: a copy may only be applied to the extent it was taken from.
    W38-A applied one to a longer extent and the pool walked off the end of its
    saved chunk list (IndexError, memory_pool.py:3295, three ranks, 14 s into
    the load). The refusal happens HERE, at the caller that knows both numbers,
    because every pool-level `load_cpu_copy` merely forwards indices and adds no
    length check of its own beyond a per-chunk one (:3298) on the wrong axis.

    NOT A CLAMP. The two extents describe different things, so a `min()` would
    write a prefix's KV into the wrong rows -- a wrong ANSWER rather than a
    crash. A refusal costs a recompute, which is merely slow.

    A REFUSED COPY IS DROPPED, not kept: it is stale against a request the model
    has since advanced, and holding it would let a later coincidentally-matching
    extent restore ancient bytes.

    #890: A REFUSAL ALSO REVOKES THE PERMISSION THAT BROUGHT THE REQUEST HERE.
    Dropping the copy is only half of it. The request was admitted into the TP
    layout under `phase_purity.seam_transport_exempt`, whose premise -- verified
    at the GRANT by `seam_transport_premise_holds` -- is that the re-admission
    "recomputes nothing". Each refusal below says in its own log line that the
    tokens ARE recomputed, so the premise is false for this request and the
    permission must not be issued to it again on the same evidence. The
    evidence field (`cached_prompt_tokens_at_retract`) cannot carry that: the
    recompute the refusal forces re-stamps it at the next retraction, so it
    reads "computed and fenced" precisely when the copy has just proven
    unusable. Hence a separate mark, set here and cleared on the success path
    below, where the claim becomes true again.

    NOTHING IS MARKED WHEN THERE WAS NO COPY. This function runs for every
    request in an extend batch and `kv_cache_cpu` is None for almost all of
    them -- a request that never went through the seam, or one whose copy the
    cutover DECLINED. Marking that path would revoke the exemption for the
    whole world and put the W30 livelock back.
    """
    saved = getattr(req, "kv_cache_cpu", None)
    if saved is None:
        return False

    covered = getattr(req, "kv_cache_cpu_extent", None)
    now = _seam_extent_of(req)
    if (
        covered is None
        or int(covered) != int(now)
        or not _seam_prefill_is_complete(req)
    ):
        _SEAM_STATE_COUNTS["refused"] += 1
        n = _SEAM_STATE_COUNTS["refused"]
        logger.warning(
            "%s SEAM RESTORE REFUSED rid=%s: the copy covers %s row(s) but the "
            "request now needs %d (allocated=%s). Applying it would index past "
            "the saved chunks (the W38-A crash) or write a prefix into the "
            "wrong rows. Dropped; these tokens are recomputed. occurrence=%d",
            SEAM_STATE_PREFIX,
            getattr(req, "rid", None),
            covered,
            now,
            getattr(req, "kv_allocated_len", None),
            n,
        )
        req.kv_cache_cpu = None
        req.kv_cache_cpu_extent = None
        req.kv_cache_cpu_layout = None
        req.mamba_state_cpu = None
        req.mamba_state_cpu_layout = None
        # #890: the tokens this line just sent back to be recomputed are the
        # ones the exemption promised would not be. Rank-uniform: both sides of
        # the comparison above are the LOGICAL extent (`kv_cache_cpu_extent` is
        # stamped in `offload_kv_cache` from `[: seqlen - 1]`), which is
        # replicated across the group.
        req.seam_restore_refused = True
        return False

    # #875: THIS REFUSAL WAS A NON-ANSWER FOR BOTH DIRECTIONS AND IS NOW THE
    # ANSWER FOR ONE. The counter comment above says what it costs -- a layout
    # refusal means every flip in that direction loses its prefixes. #875d
    # splits the two directions apart (see the carry attempt below):
    #   * the copy is MISSING layers (PP stage -> TP pool): they are on a peer,
    #     the exchange is an all-to-all in the cutover's no-return region, and
    #     #875 measured it against the recompute it saves and returned DO NOT
    #     BUILD. This refusal is the answer there, and it names the layers.
    #   * the copy is a SUPERSET (TP -> PP stage): nothing is missing and the
    #     answer is a rank-local slice. Carried, not refused.
    # The TOKEN axis (PP at allocator slots, TP under the owner rule,
    # layers/dcp/owner.py:159) is untouched by either: the carry moves nothing
    # on the row axis, so it cannot produce the "matching row ids at mismatched
    # widths" shape #719 walked into. The extent contract above still owns that
    # axis and still runs first.
    #
    # #861c: the SECOND axis, and the one W40 died on. The extent check above
    # compares ROW counts; a phase flip does not change those, so it passed and
    # handed the copy straight to a pool with a different LAYER count. See
    # `check_cpu_copy_layers` (memory_pool.py) for the mechanism and for why a
    # remap is refused rather than built.
    #
    # THE REFUSAL LIVES HERE AND NOT ONLY IN THE POOL. The pool-level guard
    # turns the IndexError into a ValueError -- still a dead scheduler, because
    # `load_kv_cache` is called unguarded. This is the check that keeps the
    # instance up; the pool's is the backstop for every other caller.
    #
    # A None layout on either side means the pool could not state one. That is
    # tolerated rather than refused: refusing on silence would switch the seam
    # carry off for pools that never had this defect, and the pool-level count
    # guard still covers them.
    saved_layout = getattr(req, "kv_cache_cpu_layout", None)
    live_layout = token_to_kv_pool_allocator.cpu_copy_layout()
    saved_mamba_layout = getattr(req, "mamba_state_cpu_layout", None)
    live_mamba_layout = None
    if saved_mamba_layout is not None:
        mamba_pool = getattr(req_to_token_pool, "mamba_pool", None)
        layout_fn = getattr(mamba_pool, "cpu_copy_layout", None)
        live_mamba_layout = layout_fn() if layout_fn is not None else None
    kv_drifted = (
        saved_layout is not None
        and live_layout is not None
        and saved_layout != live_layout
    )
    mamba_drifted = (
        saved_mamba_layout is not None
        and live_mamba_layout is not None
        and saved_mamba_layout != live_mamba_layout
    )
    if kv_drifted or mamba_drifted:
        # #875d: TRY THE CARRY BEFORE PAYING THE RECOMPUTE. Drift is not one
        # situation, it is two, and only one of them is unanswerable here.
        #
        #   the copy is MISSING layers (PP stage -> TP pool). They are on a
        #   peer. Completing this needs an all-to-all inside the cutover's
        #   no-return region, measured and refused (#875, DO NOT BUILD). The
        #   refusal below is the answer, and it now names the layers.
        #
        #   the copy has a SUPERSET (TP -> PP stage). Nothing is missing; the
        #   destination's layers are all in hand and the answer is a rank-local
        #   SLICE -- no group, no peer, no cutover cost. This was ALSO the
        #   silent arm: the pool loop runs fewer iterations and writes global
        #   0..7 into global 8..15 with no crash and no log. So the direction
        #   nobody could see was the one that never needed a collective.
        #
        # The attempt is confined to `seam_layer_carry`, which acts only on
        # layouts it can identify BY TYPE and payload shapes it can NAME.
        # Anything else raises and lands on the refusal below, unchanged -- the
        # opaque layouts #861c's contract test passes here on purpose included.
        carry_refusal = None
        try:
            carried_kv = (
                seam_layer_carry.carry_payload(
                    saved_layout, live_layout, req.kv_cache_cpu
                )
                if kv_drifted
                else None
            )
            carried_mamba = (
                seam_layer_carry.carry_payload(
                    saved_mamba_layout, live_mamba_layout, req.mamba_state_cpu
                )
                if mamba_drifted
                else None
            )
        except seam_layer_carry.SeamCarryError as exc:
            carry_refusal = str(exc)
        else:
            # BOTH HALVES OR NEITHER, and the assignment happens only after both
            # have been built. A carry that wrote the KV half and then refused
            # the mamba half would leave the request holding attention state
            # from one geometry and GDN state from another -- the partial
            # restore this whole path exists to forbid.
            if kv_drifted:
                req.kv_cache_cpu = carried_kv
                req.kv_cache_cpu_layout = live_layout
            if mamba_drifted:
                req.mamba_state_cpu = carried_mamba
                req.mamba_state_cpu_layout = live_mamba_layout
            _SEAM_STATE_COUNTS["carried"] += 1
            n = _SEAM_STATE_COUNTS["carried"]
            if n <= 5 or n % 50 == 0:
                logger.info(
                    "%s SEAM RESTORE CARRIED rid=%s: the copy was taken from %s "
                    "and this pool is %s (mamba: %s -> %s). The copy covers "
                    "every global layer this pool holds, so it is re-selected "
                    "onto them rank-locally -- no collective, no peer. These "
                    "tokens are NOT recomputed. occurrence=%d",
                    SEAM_STATE_PREFIX,
                    getattr(req, "rid", None),
                    saved_layout,
                    live_layout,
                    saved_mamba_layout,
                    live_mamba_layout,
                    n,
                )
            kv_drifted = mamba_drifted = False

    if kv_drifted or mamba_drifted:
        _SEAM_STATE_COUNTS["refused_layout"] += 1
        n = _SEAM_STATE_COUNTS["refused_layout"]
        # #941: THE EXTENT IS PRINTED HERE BECAUSE IT IS THE ONE NUMBER THAT
        # CAN REOPEN #875's DO-NOT-BUILD, AND NOTHING EMITS IT ON THIS PATH.
        # `ec1717491f` refused the PP->TP completion (an all-to-all in the
        # cutover's no-return region) on an explicitly FALSIFIABLE ground: it
        # priced the collective against a 13-row specimen and said so --
        # "payload is linear in extent while the collective is latency-
        # dominated, so a break-even exists somewhere in the hundreds-to-
        # thousands of tokens. What would settle it is the DISTRIBUTION of
        # `extent` over requests actually retracted at a flip. I do not have it
        # and am not entitled to infer it."
        #
        # That distribution was not harvestable: this line named the missing
        # LAYERS (via `carry_refusal`) but never the ROWS, and the extent-
        # mismatch refusal above -- the only other emitter of `covered` on a
        # refusal -- by construction never fires for a request that reaches
        # here. So the gate that governs whether the carry may ever be built
        # could not be measured from a boot log, only re-argued from the same
        # single specimen. One value per refusal closes that: 57 refusals is 57
        # samples, and `#941 extent=` greps the distribution straight out.
        #
        # AN INSTRUMENT, NOT A GATE. Nothing below reads it and no behaviour
        # depends on it; the refusal is unchanged in every branch. The payload
        # this prices is (missing_layers x extent x row_bytes) -- `carry_refusal`
        # already carries the layer count, this supplies the extent, and the
        # row width is a property of the checkpoint. The DO-NOT-BUILD stands
        # until that product is measured against the #656 PHB numbers; this
        # only makes measuring it possible without a new boot instrument.
        logger.warning(
            "%s SEAM RESTORE REFUSED (LAYOUT) rid=%s #941 extent=%s row(s): the "
            "copy was taken from "
            "%s and this pool is %s (mamba: %s -> %s). Applying it would index "
            "past the saved per-layer list (the W40 IndexError) or write the "
            "copy's layers into the wrong global layers -- a wrong answer with "
            "no crash. No rank-local carry was available either: %s Dropped; "
            "these tokens are recomputed. occurrence=%d",
            SEAM_STATE_PREFIX,
            getattr(req, "rid", None),
            covered,
            saved_layout,
            live_layout,
            saved_mamba_layout,
            live_mamba_layout,
            carry_refusal,
            n,
        )
        req.kv_cache_cpu = None
        req.kv_cache_cpu_extent = None
        req.kv_cache_cpu_layout = None
        req.mamba_state_cpu = None
        req.mamba_state_cpu_layout = None
        # #890: THE AXIS THE MEASUREMENT ACTUALLY LANDED ON -- W38 logged 90 and
        # 21 of exactly this line, each one an exempt admission whose tokens
        # were then recomputed in the decode layout. Rank-uniform: a flip that
        # repartitions layers changes `layer_num` on EVERY rank (a stage's slice
        # against the whole), so this verdict is a property of the flip and not
        # of the rank; a flip that repartitions nothing leaves the two equal on
        # every rank. Whether a pool can state a layout at all is a property of
        # the pool CLASS, which is likewise the same on every rank.
        req.seam_restore_refused = True
        return False

    # #783b: ANNOUNCE BEFORE THE DANGEROUS CALL. This emitter sat only
    # AFTER `load_kv_cache`, so the one event it exists to explain -- a
    # restore that does not return -- was the one it structurally could not
    # witness. W40a logged 18 seam lines; W40b crashed INSIDE this call and
    # logged ZERO, leaving the traceback to name `synchronize()` rather than
    # the restore. Rate-limited on the same cadence as the success line
    # below, so the pair costs one extra line per restore at the head and
    # nothing in steady state.
    n_try = _SEAM_STATE_COUNTS["restored"] + 1
    if n_try <= 5 or n_try % 50 == 0:
        logger.info(
            "%s SEAM RESTORE ATTEMPT rid=%s extent=%s rows -- entering load_kv_cache",
            SEAM_STATE_PREFIX,
            getattr(req, "rid", None),
            covered,
        )
    req.load_kv_cache(req_to_token_pool, token_to_kv_pool_allocator)
    req.kv_cache_cpu_extent = None
    # #890: THE REVOCATION IS NOT A LIFE SENTENCE. A restore that actually
    # happens is the premise coming true again for this request, so the mark
    # clears here. Without this one flip whose geometry did not match would
    # exile the request from an exemption that exists to keep the instance out
    # of the W30 livelock, for the rest of its life.
    req.seam_restore_refused = False
    _SEAM_STATE_COUNTS["restored"] += 1
    n = _SEAM_STATE_COUNTS["restored"]
    if n <= 5 or n % 50 == 0:
        logger.info(
            "%s SEAM RESTORE rid=%s extent=%d restored from the host copy "
            "instead of recomputing (copied=%d declined=%d restored=%d "
            "refused=%d)",
            SEAM_STATE_PREFIX,
            getattr(req, "rid", None),
            now,
            _SEAM_STATE_COUNTS["copied"],
            _SEAM_STATE_COUNTS["declined"],
            n,
            _SEAM_STATE_COUNTS["refused"],
        )
    return True


def retract_all(
    *,
    reqs: List[Req],
    server_args: ServerArgs,
    req_to_token_pool: ReqToTokenPool,
    token_to_kv_pool_allocator: BaseTokenToKVPoolAllocator,
    tree_cache: BasePrefixCache,
    hisparse_coordinator: Optional[HiSparseCoordinator],
    offload_kv: bool = True,
    copy_state: bool = False,
    retain: bool = False,
) -> List[Req]:
    retracted_reqs = reqs
    for idx in range(len(reqs)):
        release_req(
            req=reqs[idx],
            remaing_req_count=len(reqs) - idx,
            server_args=server_args,
            req_to_token_pool=req_to_token_pool,
            token_to_kv_pool_allocator=token_to_kv_pool_allocator,
            tree_cache=tree_cache,
            hisparse_coordinator=hisparse_coordinator,
            offload_kv=offload_kv,
            copy_state=copy_state,
            retain=retain,
        )
    return retracted_reqs


def compute_extend_logprob_start_len(
    *,
    logprob_start_len: int,
    prefix_len: int,
    extend_len: int,
    full_untruncated_fill_len: int,
) -> int:
    # Key variables:
    # - logprob_start_len: Absolute position in full sequence where logprob computation begins
    # - extend_logprob_start_len: Relative position within current extend batch where logprob computation begins
    # - extend_input_len: Number of tokens that need to be processed in this extend batch
    if logprob_start_len == -1:
        resolved_start = full_untruncated_fill_len
    else:
        # logprob_start_len should be at least the length of the prefix indices
        resolved_start = max(logprob_start_len, prefix_len)
    return min(resolved_start - prefix_len, extend_len)


def _compute_chunked_req_next_prompt_token(
    chunked_req: Optional[Req],
    vocab_size: int,
) -> Optional[int]:
    """Return the next real prompt token after the fill boundary, skipping
    multimodal placeholder (hash) tokens that lie outside the model vocab."""
    if chunked_req is None:
        return None
    fill_len = chunked_req.extend_range.end
    origin_ids = chunked_req.origin_input_ids
    if fill_len >= len(origin_ids):
        return None
    if origin_ids[fill_len] < vocab_size:
        return int(origin_ids[fill_len])
    return None


def _group_world_size() -> int:
    """TP world size, or 1 when the group is not up yet.

    #583 follow-up. Used only to decide whether a MISSING rank-uniform value
    is survivable (single rank) or must be refused (a group). Never raises:
    a probe that can itself fail would just move the silent-default problem
    one level down.
    """
    try:
        return int(get_parallel().tp_size or 1)
    except Exception:  # noqa: BLE001 - no parallel context yet == single rank
        return 1


@dataclasses.dataclass
class ScheduleBatch(ScheduleBatchDisaggregationDecodeMixin):
    """Store all information of a batch on the scheduler."""

    # === Core: request list (ForwardBatch derives lora_ids / rids / grammars / positions from it) ===
    reqs: List[Req]

    # === Global config and shared resources (engine-lifetime; identical across batches) ===
    # Memory pool and cache
    req_to_token_pool: ReqToTokenPool = None
    token_to_kv_pool_allocator: BaseTokenToKVPoolAllocator = None
    tree_cache: BasePrefixCache = None

    # #583 (desync site 2): this iteration's RANK-UNIFORM pool headroom, set
    # by the scheduler from `Scheduler.uniform_min_avail()` before any
    # decode-mem decision is taken. `None` means "not supplied" -- the local
    # value is then used, which is correct for a single rank and for tests,
    # and is the same fallback shape `uniform_min_avail` itself documents.
    # See `check_decode_mem` for why a decision must never read the local
    # `available_size()` under uneven DCP/TP.
    uniform_avail_floor: Optional[int] = None

    # Batch configs
    model_config: ModelConfig = None
    enable_overlap: bool = False

    # Device
    device: str = "cuda"

    # HiSparse (engine-level coordinator ref, same across batches)
    hisparse_coordinator: Optional[HiSparseCoordinator] = None

    # === Batch-variant scheduler state (per-batch; not read by ForwardBatch) ===
    # Tell whether the current running batch is full so that we can skip
    # the check of whether to prefill new requests.
    # This is an optimization to reduce the overhead of the prefill check.
    batch_is_full: bool = False

    # For chunked prefill in PP
    chunked_req: Optional[Req] = None
    chunked_req_next_prompt_token: Optional[int] = None
    contains_last_prefill_chunk: bool = True

    # For DP attention
    inner_idle_batch: Optional[ScheduleBatch] = None
    # Decode requests carried alongside a chunked-prefill batch
    decoding_reqs: List[Req] = None

    # For split prefill
    split_index: int = 0
    split_prefill_finished: bool = False
    split_forward_count: int = 1
    split_forward_batch: ForwardBatch = None

    # CPU mirror of req_pool_indices; schedule-path only (used in overlap_utils,
    # not read by ForwardBatch), stale in spec draft window
    req_pool_indices_cpu: torch.Tensor = None  # shape: [b], int64

    # Forward-pass metrics
    fpm_start_time: float = 0.0

    # hicache pointer for synchronizing data loading from CPU to GPU
    hicache_consumer_index: int = -1

    # Metrics
    dp_cooperation_info: Optional[DPCooperationInfo] = None
    prefill_stats: Optional[PrefillStats] = None
    forward_iter: Optional[int] = None

    # === GPU tensors crossing to ForwardBatch (clone targets for stream isolation) ===
    # Batched arguments to model runner
    input_ids: torch.Tensor = None  # shape: [b], int64
    # Staging consumed by resolve_forward_inputs (prefill H2D / mixed gather).
    prefill_input_ids_cpu: Optional[torch.Tensor] = None
    mix_running_indices: Optional[torch.Tensor] = None
    input_embeds: torch.Tensor = None  # shape: [b, hidden_size], float32

    # Token replacement embeddings and absolute positions (optional).
    replace_embeds: Optional[torch.Tensor] = None
    replace_positions: Optional[torch.Tensor] = None

    # Read by ForwardBatch ngram embedding init
    ne_token_table: torch.Tensor = None
    # Mask marking chunked (not-yet-finished) prefill requests whose sampled
    # pseudo next-token must NOT be written into the ngram token table.
    ne_skip_token_table_update: torch.Tensor = None

    req_pool_indices: torch.Tensor = None  # shape: [b], int64
    seq_lens: torch.Tensor = None  # shape: [b], int64

    # The original sequence lengths, Qwen-1M related
    orig_seq_lens: torch.Tensor = None  # shape: [b], int32

    # The output locations of the KV cache
    out_cache_loc: torch.Tensor = None  # shape: [b], int64
    # DSV4-NPU: per-pool slot bundle from DSV4NPUTokenToKVPoolAllocator (None
    # elsewhere); c4/c128 state lens ride on ``batch.dsv4_state_lens``.
    out_cache_loc_dsv4: Optional[Any] = None

    # For hybrid GDN prefix cache
    mamba_track_indices: torch.Tensor = None  # shape: [b], int64
    mamba_track_mask: torch.Tensor = None  # shape: [b], bool
    mamba_track_seqlens: torch.Tensor = None  # shape: [b], int64
    # Deferred mamba init ops: COW pairs and clear indices (performed on forward stream)
    mamba_cow_src_indices: torch.Tensor = None
    mamba_cow_dst_indices: torch.Tensor = None
    mamba_clear_indices: torch.Tensor = None

    # Encoder-decoder device tensors (host fields in the host metadata group)
    encoder_lens: Optional[torch.Tensor] = None
    encoder_out_cache_loc: Optional[torch.Tensor] = None

    # It comes empty list if logprob is not required.
    extend_input_logprob_token_ids: Optional[torch.Tensor] = None

    # === Config / flags crossing to ForwardBatch (by-value) ===
    forward_mode: ForwardMode = None
    global_forward_mode: Optional[ForwardMode] = None

    # For DP attention
    is_extend_in_batch: bool = False
    all_extend_in_batch: bool = False  # plumbing for downstream forks (PR #19639)
    can_run_dp_cuda_graph: bool = False
    can_run_dp_breakable_cuda_graph: bool = False
    tbo_split_seq_index: Optional[int] = None
    spec_verify_tier_num_tokens: int = -1

    # For processing logprobs
    return_logprob: bool = False

    # Whether this batch is prefill-only (no token generation needed)
    is_prefill_only: bool = False

    # kv-session-offload (S1): True only on the separate, eager bs=1 decode
    # batch of a host-spilled session ("spill tick"). Gates the sentinel
    # decode allocation, disables CUDA-graph replay, and routes the
    # flashinfer decode to the host-streamed block-attention path.
    kv_session_spill_tick: bool = False

    # kv-session-offload PS2 (deep prefill-spill): True only on an EXTEND batch
    # whose requests are all born-spilled-deep. Gates the sentinel extend
    # allocation here and the staging-carve owner write in the attention
    # backend. All-or-nothing per batch (a mixed batch would put two disjoint
    # address spaces into one out_cache_loc).
    kv_session_prefill_spill: bool = False

    # Speculative decoding
    spec_algorithm: SpeculativeAlgorithm = None

    # Whether to return hidden states
    return_hidden_states: bool = False

    # Has grammar
    has_grammar: bool = False

    # The sum of all sequence lengths
    seq_lens_sum: int = None
    extend_num_tokens: Optional[int] = None

    # Diffusion LLM
    dllm_config: Optional[DllmConfig] = None

    # === Host metadata crossing to ForwardBatch (CPU lists / mirrors) ===
    seq_lens_cpu: torch.Tensor = None  # shape: [b], int64

    # For multimodal inputs
    multimodal_inputs: Optional[List] = None

    # For processing logprobs
    top_logprobs_nums: Optional[List[int]] = None
    token_ids_logprobs: Optional[List[List[int]]] = None

    # For encoder-decoder architectures
    encoder_cached: Optional[List[bool]] = None
    encoder_lens_cpu: Optional[List[int]] = None

    # For extend and mixed chunekd prefill
    prefix_lens: List[int] = None
    extend_lens: List[int] = None
    extend_logprob_start_lens: List[int] = None

    # For DP attention
    global_num_tokens: Optional[List[int]] = None
    global_num_tokens_for_logprob: Optional[List[int]] = None
    global_spec_verify_tier_num_tokens: Optional[List[int]] = None

    # === Compound crossing to ForwardBatch (carry their own device tensors) ===
    # Sampling info
    sampling_info: SamplingBatchInfo = None

    # Speculative decoding
    # spec_info: Optional[SpecInput] = None
    spec_info: Optional[SpecInput] = None

    # === One-shot per-forward overrides; init_new consumes and resets ===
    seq_lens_cpu_cache: torch.Tensor = None
    capture_hidden_mode: Optional[CaptureHiddenMode] = None
    return_hidden_states_before_norm: bool = False

    @classmethod
    def init_new(
        cls,
        reqs: List[Req],
        req_to_token_pool: ReqToTokenPool,
        token_to_kv_pool_allocator: BaseTokenToKVPoolAllocator,
        tree_cache: BasePrefixCache,
        model_config: ModelConfig,
        enable_overlap: bool,
        spec_algorithm: SpeculativeAlgorithm,
        chunked_req: Optional[Req] = None,
        dllm_config: Optional[DllmConfig] = None,
    ):
        return_logprob = any(req.return_logprob for req in reqs)

        batch = cls(
            reqs=reqs,
            req_to_token_pool=req_to_token_pool,
            token_to_kv_pool_allocator=token_to_kv_pool_allocator,
            tree_cache=tree_cache,
            model_config=model_config,
            enable_overlap=enable_overlap,
            return_logprob=return_logprob,
            has_grammar=any(req.grammar for req in reqs),
            device=req_to_token_pool.device,
            spec_algorithm=spec_algorithm,
            return_hidden_states=any(req.return_hidden_states for req in reqs),
            is_prefill_only=all(req.is_prefill_only for req in reqs),
            chunked_req=chunked_req,
            chunked_req_next_prompt_token=_compute_chunked_req_next_prompt_token(
                chunked_req,
                model_config.vocab_size,
            ),
            dllm_config=dllm_config,
        )
        return batch

    def batch_size(self):
        return len(self.reqs)

    def is_empty(self):
        return len(self.reqs) == 0

    def is_dllm(self):
        return self.dllm_config is not None

    def prepare_encoder_info_extend(
        self, input_ids: List[array[int]], seq_lens: List[int]
    ):
        _pin = is_pin_memory_available(self.device)
        self.encoder_lens_cpu = []
        self.encoder_cached = []

        for req in self.reqs:
            im = req.multimodal_inputs
            if im is None or im.num_image_tokens is None:
                # No image input
                self.encoder_lens_cpu.append(0)
                self.encoder_cached.append(True)
            else:
                self.encoder_lens_cpu.append(im.num_image_tokens)
                self.encoder_cached.append(
                    self.forward_mode.is_decode()
                    or len(req.prefix_indices) >= im.num_image_tokens
                )

        self.encoder_lens = torch.tensor(
            self.encoder_lens_cpu, dtype=torch.int64, pin_memory=_pin
        ).to(self.device, non_blocking=True)

        # Strip encoder infos
        pt = 0
        decoder_out_cache_loc = []
        encoder_out_cache_loc = []
        for i, req in enumerate(self.reqs):
            encoder_len = self.encoder_lens_cpu[i]
            seq_lens[i] -= encoder_len

            if len(req.prefix_indices) < encoder_len:
                # NOTE: the encoder part should be considered as a whole
                assert len(req.prefix_indices) == 0
                input_ids[i] = input_ids[i][encoder_len:]
                encoder_out_cache_loc.append(self.out_cache_loc[pt : pt + encoder_len])
                decoder_out_cache_loc.append(
                    self.out_cache_loc[pt + encoder_len : pt + req.extend_range.length]
                )
                self.extend_lens[i] -= encoder_len
                self.extend_num_tokens -= encoder_len
            else:
                decoder_out_cache_loc.append(
                    self.out_cache_loc[pt : pt + req.extend_range.length]
                )
                self.prefix_lens[i] -= encoder_len

            pt += req.extend_range.length

        # Reassign: ED stripping rebuilds prefill_input_ids_cpu (CPU pinned);
        # resolve_forward_inputs will H2D this on forward stream. self.input_ids
        # stays None.
        self.prefill_input_ids_cpu = flatten_arrays_to_pinned_cpu(input_ids, _pin)
        self.seq_lens = torch.tensor(seq_lens, dtype=torch.int64, pin_memory=_pin).to(
            self.device, non_blocking=True
        )
        self.seq_lens_cpu = torch.tensor(seq_lens, dtype=torch.int64)

        if not decoder_out_cache_loc:
            self.out_cache_loc = torch.zeros(0, dtype=torch.int64).to(
                self.device, non_blocking=True
            )
        else:
            self.out_cache_loc = torch.cat(decoder_out_cache_loc)

        if not encoder_out_cache_loc:
            self.encoder_out_cache_loc = torch.zeros(0, dtype=torch.int64).to(
                self.device, non_blocking=True
            )
        else:
            self.encoder_out_cache_loc = torch.cat(encoder_out_cache_loc)

        assert len(self.out_cache_loc) == self.extend_num_tokens, (
            f"Expected {len(self.out_cache_loc)}, got {self.extend_num_tokens}"
        )

        if self.extend_input_logprob_token_ids is not None:
            new_token_ids_parts = []
            offset = 0
            for i, req in enumerate(self.reqs):
                encoder_len = self.encoder_lens_cpu[i]
                old_start_len = self.extend_logprob_start_lens[i]
                old_contribution = req.extend_range.length - old_start_len

                if len(req.prefix_indices) < encoder_len:
                    tokens_to_strip = max(0, encoder_len - old_start_len)
                    new_token_ids_parts.append(
                        self.extend_input_logprob_token_ids[
                            offset + tokens_to_strip : offset + old_contribution
                        ]
                    )
                    self.extend_logprob_start_lens[i] = max(
                        0, old_start_len - encoder_len
                    )
                else:
                    new_token_ids_parts.append(
                        self.extend_input_logprob_token_ids[
                            offset : offset + old_contribution
                        ]
                    )

                offset += old_contribution

            if new_token_ids_parts:
                self.extend_input_logprob_token_ids = torch.cat(new_token_ids_parts)
            else:
                self.extend_input_logprob_token_ids = None

        for i, req in enumerate(self.reqs):
            encoder_len = self.encoder_lens_cpu[i]
            if encoder_len == 0:
                continue
            if len(req.prefix_indices) < encoder_len:
                assert len(req.prefix_indices) == 0
                req.extend_range = req.extend_range._replace(
                    start=req.extend_range.start + encoder_len
                )
            req.logprob_start_len = max(req.logprob_start_len, encoder_len)

    def prepare_for_extend(self):
        self.forward_mode = ForwardMode.EXTEND
        server_args = get_server_args()

        if self.is_dllm():
            # For DLLM, we use a separate forward mode
            self.forward_mode = ForwardMode.DLLM_EXTEND

        # Init tensors
        reqs = self.reqs
        input_ids = [r.get_fill_ids()[len(r.prefix_indices) :] for r in reqs]
        # #969 DECISION PROBE (temporary): what did THIS rank decide the extend
        # range is, for each request, at the one place the forward width is
        # actually built. The #998 invariant below checks the pair WITHIN a
        # rank; this line is for comparing the SAME rid ACROSS ranks, which is
        # the open question -- the #631 width mismatch showed a sender at
        # extend_range=(0,4096) against a receiver batch of 1302 tokens, and
        # nothing in the tree says whether that is a rank-local match verdict
        # or a temporal lag. Grep: "#969 EXTENT".
        try:
            _rk = getattr(getattr(self, "model_config", None), "pp_rank", None)
            if _rk is None:
                import os as _os
                _rk = _os.environ.get("SGLANG_DP_RANK", "?")
            _n = getattr(ScheduleBatch, "_969_extent_n", 0) + 1
            ScheduleBatch._969_extent_n = _n
            if _n <= 400 or _n % 64 == 0:
                logger.warning(
                    "#969 EXTENT n=%d fwd=%s reqs=%s",
                    _n,
                    getattr(self, "forward_mode", "?"),
                    [
                        (
                            str(getattr(r, "rid", "?"))[:8],
                            None if r.extend_range is None else int(r.extend_range.start),
                            None if r.extend_range is None else int(r.extend_range.end),
                            0 if r.prefix_indices is None else len(r.prefix_indices),
                            len(_i),
                        )
                        for r, _i in zip(reqs, input_ids)
                    ],
                )
        except Exception:  # noqa: BLE001 - a probe may never break a forward
            pass

        # #998 THE INVARIANT, CHECKED AT THE READER.
        #
        # `set_extend_range(prefix, prefix + new_len)` means
        # `extend_range.start == len(prefix_indices)`, and the line above
        # slices `fill[len(prefix) : extend_range.end]`. So
        #     len(input_ids) = end - len(prefix) = end - start = new_len
        # HOLDS ONLY WHILE start == len(prefix_indices). This is the one
        # place that reads both, so ONE condition here covers all 24
        # `extend_range` writers and all 27 `prefix_indices` writers -- and,
        # unlike reading them one by one, it also catches the case where
        # every writer is individually correct and their ORDER breaks the
        # pair. Two of the 24 were read (truncate_prefix_to, the #797b park)
        # and both carry the group; that is 2/24 and not a basis.
        #
        # Counter here, emission in `pp_ring_note` (independent of this
        # probe, and its execution on this config is established). -1 is the
        # sentinel; 0 is a legitimate break value, so it cannot be the
        # default.
        try:
            # #998c ALL rids, not the first four: two closing statements of
            # this window rested on a sample that excluded the dying request.
            for _r in reqs:
                _er = getattr(_r, "extend_range", None)
                # #796 AGAIN, AND THIS TIME I WROTE IT. `prefix_indices` is a
                # TENSOR; `x or ()` asks `bool(x)`, which torch refuses for a
                # multi-element tensor. scheduler.py:8300 documents this exact
                # idiom costing every admitting pass its trace line -- "the
                # docstring above already said len() is the right spelling;
                # the `or []` slipped in anyway" -- and I read that comment in
                # this same session before reproducing the bug. len() with an
                # explicit None test is the only correct spelling here.
                _pi = getattr(_r, "prefix_indices", None)
                _pl = 0 if _pi is None else len(_pi)
                if _er is None:
                    _rec = (-1, -1, _pl, -1, -1)
                else:
                    _st, _en = int(_er.start), int(_er.end)
                    _rec = (_st, _en, _pl, _en - _pl, _st - _pl)
                _998_SEEN[0] += 1
                # #998c HISTORY, not last-writer-wins. Every measurement in
                # this window compared RANKS AT ONE INSTANT and every one came
                # out uniform -- because the sender read its number one
                # forward EARLIER than the receiver (fwd_ct 2107/2106). The
                # two numbers the guard compares are two TIMESTAMPS, and the
                # axis "same rank, consecutive passes" was never measured.
                # `extend_range` is written from 24 sites -- (told,told) by the
                # truncation, (prefix, prefix+new_len) by add_one_req,
                # (len(prefix),len(prefix)) by the park -- so a request that
                # passes through admission, truncation and a park across
                # several passes MUST see different values. The question is
                # whether the change falls between the two passes the guard
                # compares.
                if len(_998_LAST) < 64 or _r.rid in _998_LAST:
                    _h = _998_LAST.setdefault(_r.rid, [])
                    if not _h or _h[-1][1:] != _rec:
                        _h.append((_998_SEEN[0],) + _rec)
                        del _h[:-4]
                if _rec[4] not in (0, -1):
                    _998_BREAKS[0] += 1
        except Exception:  # noqa: BLE001 - a probe may never break prepare_for_extend
            pass
        extend_num_tokens = sum(len(ids) for ids in input_ids)
        seq_lens = [r.extend_range.end for r in reqs]
        orig_seq_lens = [max(r.extend_range.end, len(r.origin_input_ids)) for r in reqs]
        prefix_lens = [len(r.prefix_indices) for r in reqs]
        # #639: this vector is the single rank-local input that decides both
        # the SHAPE of every per-layer TP collective in the forward (via
        # `extend_num_tokens` two lines up) and WHICH DCP collectives run at
        # all (via `weightless_has_prefix` -> `_forward_extend_dcp`'s
        # `if not has_prefix: ... return`). Check it here, once, where it is
        # first materialised -- not per layer, and not after the forward has
        # already entered a collective it cannot leave. Detector only: no
        # collective and no behaviour change without DCP over >1 rank.
        # Imported here rather than at module scope: this file's import block
        # is already an E402 region, and a local import keeps the new
        # dependency out of it without adding a finding.
        from sglang.srt.layers.dcp.prefix_lens_check import (
            assert_prefix_lens_rank_uniform,
        )

        assert_prefix_lens_rank_uniform(prefix_lens)
        extend_lens = [r.extend_range.length for r in reqs]
        extend_logprob_start_lens = [
            compute_extend_logprob_start_len(
                logprob_start_len=r.logprob_start_len,
                prefix_len=prefix_lens[i],
                extend_len=extend_lens[i],
                full_untruncated_fill_len=len(r.full_untruncated_fill_ids),
            )
            for i, r in enumerate(reqs)
        ]

        _pin = is_pin_memory_available(self.device)
        # Stay on pinned CPU; H2D is deferred to forward stream via
        # resolve_forward_inputs.
        pinned_input_ids = flatten_arrays_to_pinned_cpu(input_ids, _pin)
        seq_lens_tensor = torch.tensor(seq_lens, dtype=torch.int64, pin_memory=_pin).to(
            self.device, non_blocking=True
        )
        seq_lens_cpu = torch.tensor(seq_lens, dtype=torch.int64)
        orig_seq_lens_tensor = torch.tensor(
            orig_seq_lens, dtype=torch.int32, pin_memory=_pin
        ).to(self.device, non_blocking=True)

        # Set batch fields needed by alloc_for_extend
        self.prefix_lens = prefix_lens
        self.extend_lens = extend_lens
        self.seq_lens = seq_lens_tensor
        self.seq_lens_cpu = seq_lens_cpu
        self.extend_num_tokens = extend_num_tokens

        # Allocate memory
        #
        # kv-session-offload PS2 (deep prefill-spill): a born-spilled-deep
        # batch must NOT reach alloc_for_extend -- that call allocates real
        # device slots for every new token before any write happens, so a
        # prompt that does not transiently fit wedges right there and no write
        # retarget could ever help ("never materialize" is an ALLOCATION
        # property, not a write property). Instead the request slots are
        # allocated normally (cheap, and the row is needed) and the token slots
        # become host sentinels. Gated on the manager, so with the feature off
        # the default path below is byte-identical.
        _kv_sess_prefill = False
        if server_args.kv_session_offload_prefill and any(
            getattr(r, "born_spilled_deep", False) for r in reqs
        ):
            from sglang.srt.managers.kv_session_offload import (
                get_kv_session_offload_manager,
            )

            _mgr = get_kv_session_offload_manager()
            _kv_sess_prefill = _mgr.prefill_spill_extend_ready(self)
        if _kv_sess_prefill:
            from sglang.srt.mem_cache.common import alloc_req_slots

            self.kv_session_prefill_spill = True
            req_pool_indices = alloc_req_slots(
                self.req_to_token_pool, self.reqs, self.tree_cache
            )
            req_pool_indices_cpu = torch.tensor(req_pool_indices, dtype=torch.int64)
            req_pool_indices_tensor = req_pool_indices_cpu.to(
                self.device, non_blocking=True
            )
            # alloc_req_slots assigns req.req_pool_idx in place and returns one
            # index per request (reused slots included).
            self.req_pool_indices = req_pool_indices_tensor
            self.req_pool_indices_cpu = req_pool_indices_cpu
            out_cache_loc = _mgr.spill_extend_alloc(self)
        else:
            out_cache_loc, req_pool_indices_tensor, req_pool_indices_cpu = (
                alloc_for_extend(self)
            )

        # Set fields
        input_embeds = []
        all_replace_embeds: List[torch.Tensor] = []
        all_replace_positions: List[int] = []
        has_replace_embeds = False
        input_id_pointer = 0
        input_id_lens = [len(input_id) for input_id in input_ids]
        extend_input_logprob_token_ids = []
        multimodal_inputs = []
        mamba_track_mask_cpu = []
        mamba_track_indices_cpu = []
        mamba_track_seqlens_cpu = []

        for i, (req, seq_len, pre_len) in enumerate(zip(reqs, seq_lens, prefix_lens)):
            assert seq_len - pre_len == req.extend_range.length

            req.extend_batch_idx += 1

            # update req-level memory management fields
            req.kv_committed_len = seq_len
            req._kvc_src = "extend"  # #969L writer stamp
            req.kv_allocated_len = seq_len

            # If input_embeds are available, store them
            if req.input_embeds is not None:
                # Slice to match extend_input_len — PrefillAdder truncates
                # fill_len/extend_input_len on chunk overflow but not input_embeds.
                input_embeds.extend(
                    req.input_embeds[pre_len : pre_len + req.extend_range.length]
                )

            if req.positional_embed_overrides is not None:
                # Override positions are absolute in the full sequence.
                # Convert to extend-tensor coordinates by subtracting pre_len,
                # then skip any that fall within the cached prefix.
                embeds_to_add = []
                for embed_idx, pos in enumerate(
                    req.positional_embed_overrides.positions
                ):
                    extend_pos = pos - pre_len
                    if extend_pos < 0 or extend_pos >= req.extend_range.length:
                        continue  # Outside current extend chunk, skip
                    embeds_to_add.append((embed_idx, input_id_pointer + extend_pos))
                if embeds_to_add:
                    has_replace_embeds = True
                    indices, positions = zip(*embeds_to_add)
                    all_replace_embeds.append(
                        req.positional_embed_overrides.embeds[list(indices)]
                    )
                    all_replace_positions.extend(positions)
            input_id_pointer += input_id_lens[i]

            multimodal_inputs.append(req.multimodal_inputs)

            # Only calculate cached_tokens once. Once retracted, the 'retracted_stain'
            # flag will always True
            if not req.retracted_stain:
                new_cached = pre_len - req.already_computed
                req.cached_tokens += new_cached

                # Calculate detailed breakdown of cached tokens by source (for HiCache)
                # Only compute once on FIRST chunk - subsequent chunks in chunked prefill
                # would incorrectly count previously computed tokens as cache hits.
                if not req._cache_breakdown_computed:
                    # At this point, prefix_indices has been extended with host data
                    # via init_load_back in schedule_policy, so:
                    # - len(prefix_indices) = device_original + host_loaded
                    # - host_hit_length = total tokens from host cache (including storage-prefetched)
                    # - storage_hit_length = tokens loaded from storage backend (L3 hits)
                    # - device_portion = len(prefix_indices) - host_hit_length
                    #
                    # Storage hits are now tracked via scheduler after prefetch completes.
                    # storage_hit_length is set by scheduler.pop_prefetch_loaded_tokens()
                    host_total = req.host_hit_length
                    # Clamp storage to host_total to handle edge cases
                    storage_portion = min(host_total, req.storage_hit_length)
                    host_portion = host_total - storage_portion
                    device_portion = max(0, len(req.prefix_indices) - host_total)

                    req.cached_tokens_device = device_portion
                    req.cached_tokens_host = host_portion
                    req.cached_tokens_storage = storage_portion
                    req._cache_breakdown_computed = True

                req.already_computed = seq_len

            # #783 half 2: restore instead of recomputing, for the cutover
            # population only. Here and not in `readmit_seam_residents`, which
            # only re-queues: by then `reset_for_retract()` has cleared
            # `req_pool_idx` and there is nothing to load into. `alloc_for_extend`
            # above has just given this request rows back. The extent contract
            # inside REFUSES on drift rather than indexing, which is what W38-A
            # needed. Before `is_retracted` is cleared, so the order is visible.
            restore_seam_state(
                req, self.req_to_token_pool, self.token_to_kv_pool_allocator
            )
            req.is_retracted = False

            if server_args.enable_mamba_extra_buffer():
                track_entry = self._mamba_radix_cache_v2_req_prepare_for_extend(req)
                mamba_track_mask_cpu.append(track_entry.track_mask)
                mamba_track_indices_cpu.append(track_entry.track_index)
                mamba_track_seqlens_cpu.append(track_entry.track_seqlen)

            if self.return_logprob:
                # Find input logprob token ids.
                # First, find a global index within origin_input_ids and slide it by 1
                # to compute input logprobs. It is because you need the next token
                # to compute input logprobs. E.g., (chunk size 2)
                #
                # input_logprobs = [1, 2, 3, 4]
                # get_fill_ids() = [1, 2]
                # extend_input_logprob_token_id = [2, 3]
                #
                # Note that it can also overflow. In this case, we pad it with 0.
                # input_logprobs = [1, 2, 3, 4]
                # get_fill_ids() = [3, 4]
                # extend_input_logprob_token_id = [4, 0]
                global_start_idx, global_end_idx = (
                    len(req.prefix_indices),
                    req.extend_range.end,
                )
                if req.logprob_start_len == -1:
                    logprob_start_len = len(req.origin_input_ids)
                else:
                    logprob_start_len = req.logprob_start_len
                # Apply logprob_start_len
                if global_start_idx < logprob_start_len:
                    global_start_idx = logprob_start_len

                logprob_token_ids = req.origin_input_ids[
                    global_start_idx + 1 : global_end_idx + 1
                ]
                extend_input_logprob_token_ids.extend(logprob_token_ids)

                # We will need req.extend_range.length - extend_logprob_start_lens[i] number of
                # tokens, and logprob_token_ids is for input logprob, so pad the rest of them by 0.
                extend_input_logprob_token_ids.extend(
                    [0]
                    * (
                        req.extend_range.length
                        - extend_logprob_start_lens[i]
                        - len(logprob_token_ids)
                    )
                )

        if self.return_logprob:
            extend_input_logprob_token_ids = torch.tensor(
                extend_input_logprob_token_ids
            )
            # Clamp placeholder or out-of-range token IDs (e.g., multimodal hashes)
            # so they stay within the vocab boundary before being sent to GPU.
            extend_input_logprob_token_ids.clamp_(0, self.model_config.vocab_size - 1)
        else:
            extend_input_logprob_token_ids = None

        if has_replace_embeds:
            replace_embeds_tensor = torch.cat(all_replace_embeds, dim=0).to(
                self.device, non_blocking=True
            )
            replace_positions_tensor = torch.tensor(
                all_replace_positions, dtype=torch.long, device=self.device
            )
        else:
            replace_embeds_tensor = None
            replace_positions_tensor = None

        self.input_ids = None
        self.prefill_input_ids_cpu = pinned_input_ids
        self.req_pool_indices = req_pool_indices_tensor
        self.req_pool_indices_cpu = req_pool_indices_cpu
        self.orig_seq_lens = orig_seq_lens_tensor
        self.out_cache_loc = out_cache_loc
        self.input_embeds = (
            torch.tensor(input_embeds, pin_memory=_pin).to(
                self.device, non_blocking=True
            )
            if input_embeds
            else None
        )
        self.replace_embeds = replace_embeds_tensor
        self.replace_positions = replace_positions_tensor
        for mm_input in multimodal_inputs:
            if mm_input is None:
                continue
            if isinstance(mm_input.vision_position_ids, torch.Tensor):
                mm_input.vision_position_ids = mm_input.vision_position_ids.to(
                    self.device, non_blocking=True
                )
            if isinstance(mm_input.visible_frame_counts, torch.Tensor):
                mm_input.visible_frame_counts = mm_input.visible_frame_counts.to(
                    self.device, non_blocking=True
                )
        self.multimodal_inputs = multimodal_inputs
        self.seq_lens_sum = sum(seq_lens)

        if self.return_logprob:
            self.top_logprobs_nums = [r.logprob.top_logprobs_num for r in reqs]
            self.token_ids_logprobs = [r.logprob.token_ids_logprob for r in reqs]

        self.extend_logprob_start_lens = extend_logprob_start_lens
        self.extend_input_logprob_token_ids = extend_input_logprob_token_ids

        if server_args.enable_mamba_extra_buffer():
            self.mamba_track_indices = torch.tensor(
                mamba_track_indices_cpu,
                dtype=torch.int64,
                device=self.device,
            )
            self.mamba_track_mask = torch.tensor(
                mamba_track_mask_cpu,
                dtype=torch.bool,
                device=self.device,
            )
            self.mamba_track_seqlens = torch.tensor(
                mamba_track_seqlens_cpu,
                dtype=torch.int64,
                device=self.device,
            )

        # Collect mamba init info for deferred ops on forward stream
        if any(req.mamba_pool_idx is not None for req in reqs):
            self._collect_deferred_mamba_cow_and_clear(reqs)

        if self.model_config.is_encoder_decoder:
            self.prepare_encoder_info_extend(input_ids, seq_lens)

        # Build sampling info
        self.sampling_info = SamplingBatchInfo.from_schedule_batch(
            self,
            self.model_config.vocab_size,
        )

    def _mamba_radix_cache_v2_req_prepare_for_extend(
        self,
        req: Req,
    ) -> _MambaRadixCacheV2TrackEntry:
        server_args = get_server_args()
        mamba_cache_chunk_size = server_args.mamba_cache_chunk_size

        def _force_track_h(i: int) -> int:
            assert i % mamba_cache_chunk_size == 0
            # There are 3 cases for mamba_track_seqlen passed to mamba_track_seqlens_cpu:
            # 1) aligned with mamba_cache_chunk_size-> retrieve from last_recurrent_state
            #    a) is the last position -> retrieve from last_recurrent_state
            #    b) is NOT the last position -> retrieve from h
            # 2) unaligned with mamba_cache_chunk_size -> retrieve from h
            # Currently, the math calculation only supports case 1a and 2. So for 1b, we need to add 1
            # to force the math calculation to retrieve the correct mamba state from h.
            return i + 1

        ckpt_interval = server_args.mamba_checkpoint_interval
        if ckpt_interval is not None:
            return self._mamba_ckpt_interval_req_prepare_for_extend(
                req, ckpt_interval, mamba_cache_chunk_size
            )

        mask = req.extend_range.length >= mamba_cache_chunk_size
        track_index = req.mamba_ping_pong_track_buffer[req.mamba_next_track_idx].item()
        mamba_track_seqlen = -1
        if mask:
            # mamba_track_seqlen is used to calculate the indices to track in
            # hybrid_linear_attn_backend's _init_track_ssm_indices. Due to the
            # fact that the ssm state between aligned and non-aligned are retrieved differently,
            # if 1) last pos and 2) is aligned, then retrieved from the last_recurrent_state,
            # otherwise retrieved from h (i.e. unaligned).
            # We need to pass the non-aligned seqlen to the calculation. Even though
            # we pass in mamba_track_seqlen, the actual tracked seqlen is mamba_last_track_seqlen.
            mamba_track_seqlen = len(req.prefix_indices) + req.extend_range.length

            # mamba_track_seqlen_aligned/mamba_last_track_seqlen is actual tracked seqlen. Used to pass to
            # mamba radix cache to track which seqlen this mamba state should store at.
            mamba_track_seqlen_aligned = (
                len(req.prefix_indices)
                + (req.extend_range.length // mamba_cache_chunk_size)
                * mamba_cache_chunk_size
            )

            # mamba_track_fla_chunk_aligned is the aligned seqlen based on mamba_cache_chunk_size
            # If mamba_track_fla_chunk_aligned != mamba_track_seqlen_aligned, which can be true when
            # page_size > mamba_cache_chunk_size, we need to force the math calculation to retrieve the correct mamba state from h
            # by _force_track_h()
            mamba_track_fla_chunk_aligned = (
                len(req.prefix_indices)
                + (req.extend_range.length // mamba_cache_chunk_size)
                * mamba_cache_chunk_size
            )
            if mamba_track_fla_chunk_aligned != mamba_track_seqlen_aligned:
                # We want to track mamba_track_seqlen_aligned, and it's not the last position,
                # so we need to add 1 to the seqlen to retrieve the correct mamba state from h.
                mamba_track_seqlen = _force_track_h(mamba_track_seqlen_aligned)

            # In lazy mode, skip the swap — the second ping-pong slot is not
            # allocated yet; it will be allocated on demand at the track boundary
            # in mamba_lazy_prealloc_at_boundary during prepare_for_decode.
            if not server_args.enable_mamba_extra_buffer_lazy():
                req.mamba_next_track_idx = (
                    self.req_to_token_pool.get_mamba_ping_pong_other_idx(
                        req.mamba_next_track_idx
                    )
                )
            if req.mamba_branching_seqlen is not None:
                # track branching point in this forward if the branching point
                # is within the current extend batch.
                branching_seqlen_aligned_mask = (
                    req.mamba_branching_seqlen - len(req.prefix_indices)
                ) % mamba_cache_chunk_size == 0
                if (
                    req.mamba_branching_seqlen > len(req.prefix_indices)
                    and req.mamba_branching_seqlen < mamba_track_seqlen
                    and branching_seqlen_aligned_mask
                ):
                    # We want to track mamba_track_seqlen_aligned, and it's not the last position,
                    # so we need to add 1 to the seqlen to retrieve the correct mamba state from h.
                    # See _force_track_h() for more details.
                    mamba_track_seqlen = _force_track_h(req.mamba_branching_seqlen)
                    mamba_track_seqlen_aligned = req.mamba_branching_seqlen
            req.mamba_last_track_seqlen = mamba_track_seqlen_aligned

        return _MambaRadixCacheV2TrackEntry(
            track_mask=mask,
            track_index=track_index,
            track_seqlen=mamba_track_seqlen,
        )

    def _mamba_ckpt_interval_req_prepare_for_extend(
        self,
        req: Req,
        ckpt_interval: int,
        mamba_cache_chunk_size: int,
    ) -> _MambaRadixCacheV2TrackEntry:
        """--mamba-checkpoint-interval variant of the extend-track setup.

        Instead of the default's step-relative snapshot position
        (``prefix + floor(extend/chunk) * chunk`` — a function of the
        traffic-dependent prefill split), track the deepest ABSOLUTE
        multiple of the checkpoint interval inside this step, so the
        cached checkpoint positions of a request depend only on its token
        history. Mid-step targets reuse the existing intermediate-``h``
        retrieval (the ``+1`` force, see ``_force_track_h`` in the default
        path); a target at the exact step end reads
        ``last_recurrent_state``. Unreachable targets (no boundary in the
        step, or a chunk-unaligned offset after an unaligned resume edge)
        deterministically skip tracking for this step.
        """
        from sglang.srt.mem_cache.mamba_ckpt_utils import (
            mamba_checkpoint_track_target,
        )

        server_args = get_server_args()
        prefix_len = len(req.prefix_indices)
        end = prefix_len + req.extend_range.length
        target = mamba_checkpoint_track_target(
            prefix_len, req.extend_range.length, ckpt_interval, mamba_cache_chunk_size
        )
        track_index = req.mamba_ping_pong_track_buffer[req.mamba_next_track_idx].item()
        mamba_track_seqlen = -1
        # #758 emitter (1 of 3): ANCHOR WRITTEN, and the silent-inert question.
        #
        # The comp4 load ladder found ZERO log lines matching "anchor" across a
        # whole 300 s window with --mamba-checkpoint-interval 8192 armed. That
        # is indistinguishable between "the cadence is correct and silent" and
        # "the flag is inert" -- the #742 silently-inert-flag class. This is the
        # only place a checkpoint target is chosen, so it is where the question
        # is settled: every anchor is counted, the FIRST is always logged (so
        # "did any fire at all" is answerable from one grep), and thereafter one
        # line per 16 anchors carries the position and the running count, which
        # is what the 1-per-interval cadence is computed from.
        # Deliberately counts TARGETS CHOSEN rather than states copied: if the
        # copy silently no-ops the count still rises and the two disagree, which
        # is the more useful failure to be able to see.
        if target is not None:
            try:
                n = getattr(type(self), "_mamba_anchor_count", 0) + 1
                type(self)._mamba_anchor_count = n
                if n == 1 or n % 16 == 0:
                    logger.info(
                        "MAMBA-ANCHOR written n=%d at abs_pos=%d "
                        "(interval=%d, chunk=%d, prefix_len=%d, extend=%d) -- "
                        "cadence should be one per %d tokens",
                        n,
                        target,
                        ckpt_interval,
                        mamba_cache_chunk_size,
                        prefix_len,
                        req.extend_range.length,
                        ckpt_interval,
                    )
            except Exception:  # noqa: BLE001 - an instrument may never break a step
                pass
            if target == end:
                # Last position of the step: state comes from
                # last_recurrent_state (chunk-aligned routing).
                mamba_track_seqlen = end
            else:
                # Mid-step: retrieve from intermediate h. +1 forces the
                # unaligned routing that indexes h at (target - prefix) /
                # chunk (same trick as the default path's _force_track_h).
                assert target % mamba_cache_chunk_size == 0
                mamba_track_seqlen = target + 1
            mamba_track_seqlen_aligned = target

            if not server_args.enable_mamba_extra_buffer_lazy():
                req.mamba_next_track_idx = (
                    self.req_to_token_pool.get_mamba_ping_pong_other_idx(
                        req.mamba_next_track_idx
                    )
                )
            if req.mamba_branching_seqlen is not None:
                # Re-establish a missing checkpoint at the branching point if
                # it lies inside this step; it is on the interval grid by
                # construction (_match_post_processor) — verify defensively.
                branching = req.mamba_branching_seqlen
                if (
                    branching > prefix_len
                    and branching < target
                    and branching % ckpt_interval == 0
                    and (branching - prefix_len) % mamba_cache_chunk_size == 0
                ):
                    assert branching % mamba_cache_chunk_size == 0
                    mamba_track_seqlen = branching + 1
                    mamba_track_seqlen_aligned = branching
            req.mamba_last_track_seqlen = mamba_track_seqlen_aligned

        return _MambaRadixCacheV2TrackEntry(
            track_mask=target is not None,
            track_index=track_index,
            track_seqlen=mamba_track_seqlen,
        )

    def _collect_deferred_mamba_cow_and_clear(self, reqs):
        """Collect deferred COW/clear info from requests."""
        cow_src_tensors = []
        cow_dst_tensors = []
        clear_tensors = []
        for req in reqs:
            if req.mamba_cow_src_index is not None:
                cow_src_tensors.append(req.mamba_cow_src_index)
                cow_dst_tensors.append(req.mamba_pool_idx.unsqueeze(0))
                req.mamba_cow_src_index = None
                req.mamba_needs_clear = False
            elif req.mamba_needs_clear:
                clear_tensors.append(req.mamba_pool_idx.unsqueeze(0))
                req.mamba_needs_clear = False
            if req.mamba_pingpong_clear_indices is not None:
                # Freshly claimed ping-pong track slots: zero on the forward
                # stream like the active slot, so no premature read can ever
                # observe the previous occupant's state.
                clear_tensors.append(req.mamba_pingpong_clear_indices)
                req.mamba_pingpong_clear_indices = None
        self.mamba_cow_src_indices = (
            torch.cat(cow_src_tensors) if cow_src_tensors else None
        )
        self.mamba_cow_dst_indices = (
            torch.cat(cow_dst_tensors) if cow_dst_tensors else None
        )
        self.mamba_clear_indices = torch.cat(clear_tensors) if clear_tensors else None

    def prepare_for_split_prefill(self):
        self.prepare_for_extend()
        # For split prefill, we need to set the forward mode to SPLIT_PREFILL
        self.forward_mode = ForwardMode.SPLIT_PREFILL

    def mix_with_running(self, running_batch: ScheduleBatch):
        self.forward_mode = ForwardMode.MIXED
        running_bs = running_batch.batch_size()

        for req in running_batch.reqs:
            req._refresh_fill_ids()
            full_len = len(req.full_untruncated_fill_ids)
            req.set_extend_range(full_len - 1, full_len)

        # Decode tokens of the running portion live in future_map.output_tokens_buf.
        self.input_ids = None
        self.mix_running_indices = running_batch.req_pool_indices
        out_cache_loc = torch.cat([self.out_cache_loc, running_batch.out_cache_loc])

        self.merge_batch(running_batch)
        self.out_cache_loc = out_cache_loc

        # For overlap scheduler, the output_ids has one step delay
        delta = 0 if self.enable_overlap else -1

        # NOTE: prefix_indices is what has been cached, but we don't cache each decode step
        self.prefix_lens.extend(
            [
                len(r.origin_input_ids) + len(r.output_ids) + delta
                for r in running_batch.reqs
            ]
        )
        self.extend_lens.extend([1] * running_bs)
        self.extend_num_tokens += running_bs
        # TODO (lianmin): Revisit this. It should be seq_len - 1
        self.extend_logprob_start_lens.extend([0] * running_bs)
        self.is_prefill_only = False

    def new_tokens_required_next_decode(
        self, selected_indices: Optional[List[int]] = None
    ):
        page_size = self.token_to_kv_pool_allocator.page_size
        requests = (
            self.reqs
            if selected_indices is None
            else [self.reqs[i] for i in selected_indices]
        )

        if self.spec_algorithm.is_none():
            new_pages = sum(1 for r in requests if r.kv_committed_len % page_size == 0)
            return new_pages * page_size

        return self._new_tokens_required_next_decode_spec_v2(requests, page_size)

    def _new_tokens_required_next_decode_spec_v2(self, requests, page_size):
        """Tight estimate matching eagle_utils.eagle_prepare_for_decode allocation."""
        reserve = get_alloc_reserve_per_decode()
        total = 0
        for r in requests:
            x = max(0, r.kv_committed_len + reserve - r.kv_allocated_len)
            cur = r.kv_allocated_len
            nxt = cur + x
            total += ceil_align(nxt, page_size) - ceil_align(cur, page_size)
        return total

    def decode_mem_avail(self) -> int:
        """The headroom a decode-mem decision is allowed to read.

        #583 (desync site 2). ``token_to_kv_pool_allocator.available_size()``
        is a RANK-LOCAL quantity: under uneven DCP/TP the ranks own weighted
        shares of the token pool (this rig boots [332656, 177832, 177832]),
        so the same replicated token demand sits above one rank's headroom
        and below another's. Any BRANCH decided from it splits the group --
        the rank-local-test-before-a-group-collective family.

        #603 closed that for the decision of WHETHER to retract, by reducing
        once per iteration in ``Scheduler._update_uniform_pool_budget`` and
        comparing against the reduced value. It did not reach the decision of
        HOW MANY requests to retract, which ``retract_decode`` takes from
        ``check_decode_mem`` in a loop, nor the last-survivor test that can
        empty the batch outright. Those are the two call sites this closes.

        The reduced value is the group MINIMUM, so it is <= every rank's own
        headroom: the loop below therefore never under-retracts (it cannot
        leave a rank short and OOM it), and every rank pops the same victims
        in the same order and stops on the same iteration. No collective is
        taken here -- the reduce already happened, once, pre-branch.
        """
        floor = self.uniform_avail_floor
        if floor is not None:
            return int(floor)
        # #583 follow-up: the fallback must NOT silently reinstate the defect.
        #
        # This used to `return available_size()` whenever the floor was
        # unset, justified as "correct for a single rank and for tests". It
        # is -- but on a MULTI-RANK boot it is the original rank-local
        # predicate, restored silently by any path that forgets to set the
        # floor. A default that quietly means something else on the
        # configuration that matters is the getattr-default trap (#606): the
        # protection would still be present in the source and absent at
        # runtime, and nothing would say so.
        #
        # So: single rank keeps the local value (there is nothing to diverge
        # from), and a group refuses loudly instead of guessing.
        if _group_world_size() > 1:
            raise RuntimeError(
                "decode_mem_avail() was reached with uniform_avail_floor "
                "unset on a multi-rank boot. Falling back to this rank's "
                "local available_size() here would restore the exact "
                "rank-local decode-mem predicate #583/#603 removed, and the "
                "ranks would silently retract different numbers of requests. "
                "Scheduler.update_running_batch must set "
                "batch.uniform_avail_floor = self.uniform_min_avail() before "
                "any decode-mem decision."
            )
        return int(self.token_to_kv_pool_allocator.available_size())

    def check_decode_mem(self, selected_indices: Optional[List[int]] = None):
        num_tokens = self.new_tokens_required_next_decode(selected_indices)
        # Eviction stays LOCAL and unconditional: it is the side effect that
        # actually frees the space. Only the COMPARISON moves to the reduced
        # value, exactly as the #603 call site in the scheduler does it.
        evict_from_tree_cache(self.tree_cache, num_tokens)
        return self.decode_mem_avail() >= num_tokens

    def retract_all(self, server_args: ServerArgs, offload_kv: bool = True):
        retracted_reqs = retract_all(
            reqs=self.reqs,
            server_args=server_args,
            req_to_token_pool=self.req_to_token_pool,
            token_to_kv_pool_allocator=self.token_to_kv_pool_allocator,
            tree_cache=self.tree_cache,
            hisparse_coordinator=self.hisparse_coordinator,
            offload_kv=offload_kv,
        )
        self.reqs = []
        return retracted_reqs

    def retract_decode(
        self, server_args: ServerArgs
    ) -> Tuple[List[Req], float, List[Req]]:
        """Retract the decoding requests when there is not enough memory."""
        sorted_indices = self._get_decode_retraction_order(
            self.reqs,
            server_args,
            allow_policy_sort=(
                self.spec_algorithm is None or self.spec_algorithm.is_none()
            ),
        )

        retracted_reqs = []
        first_iter = True
        while first_iter or (
            not self.check_decode_mem(selected_indices=sorted_indices)
        ):
            if len(sorted_indices) == 1:
                # Always keep at least one request
                break

            first_iter = False
            idx = sorted_indices.pop()
            req = self.reqs[idx]
            retracted_reqs.append(req)
            # release memory and don't insert into the tree because we need the space instantly
            self.release_req(idx, len(sorted_indices), server_args)

        reqs_to_abort: List[Req] = []
        if len(sorted_indices) <= 1 and not self.check_decode_mem(
            selected_indices=sorted_indices
        ):
            # Even the last remaining request cannot fit in memory. Every
            # less-preferred request has already been retracted above, so by
            # construction this survivor is the OLDEST / most-progressed
            # request in the batch -- the one FCFS says must not be
            # sacrificed for someone else's pressure (#273). This is
            # reachable under ordinary extreme concurrent load once the
            # kv-session-offload spill budget is exhausted (try_spill
            # returns False when no host region is free -> stock retraction
            # runs, see scheduler.py's decode-OOM branch): transient, not a
            # sign this request is unfittable.
            last_idx = sorted_indices.pop()
            last_req = self.reqs[last_idx]
            last_req.solo_oom_count += 1
            max_retries = envs.SGLANG_RETRACT_SOLO_OOM_MAX_RETRIES.get()
            if last_req.solo_oom_count <= max_retries:
                # Retract it too, exactly like every other victim in this
                # function: release memory and send it back to the waiting
                # queue instead of killing the client. It gets another
                # scheduling turn once pressure eases (a spill region frees
                # up, another request finishes, ...).
                retracted_reqs.append(last_req)
                self.release_req(last_idx, 0, server_args)
                logger.warning(
                    "retract_decode: retracted the last remaining request "
                    "%s (solo-OOM #%d/%d) instead of aborting it -- the "
                    "pool could not fit even one request's next decode "
                    "step",
                    last_req.rid,
                    last_req.solo_oom_count,
                    max_retries,
                )
            else:
                # This exact request has now lost the solo-OOM race too many
                # times to be ordinary contention, which resolves within a
                # couple of scheduler iterations. Retracting it again would
                # silently turn a structurally oversized request into an
                # infinite retry loop with zero progress -- worse than a
                # clean failure. SERVICE_UNAVAILABLE, not
                # INTERNAL_SERVER_ERROR: nothing crashed, the pool is
                # (persistently) unable to seat this request's own
                # footprint, which is a capacity fact about the request, not
                # a server fault.
                last_req.to_finish = FINISH_ABORT(
                    f"Out of memory {last_req.solo_oom_count} times in a "
                    f"row as the sole remaining request in the decode "
                    f"batch. The pool cannot currently fit this request "
                    f"even alone.",
                    status_code=HTTPStatus.SERVICE_UNAVAILABLE,
                )
                reqs_to_abort.append(last_req)
                self.release_req(last_idx, 0, server_args)
                logger.warning(
                    "retract_decode: aborted request %s after %d "
                    "consecutive solo-OOM retractions (SERVICE_UNAVAILABLE)",
                    last_req.rid,
                    last_req.solo_oom_count,
                )

        self.filter_batch(keep_indices=sorted_indices)

        # Reqs in batch are filtered
        new_estimate_ratio = (
            NewTokenRatioTracker.estimate_new_token_ratio_after_retract(self.reqs)
        )

        return retracted_reqs, new_estimate_ratio, reqs_to_abort

    @staticmethod
    def _get_decode_retraction_order(
        reqs: List[Req], server_args: ServerArgs, *, allow_policy_sort: bool
    ) -> List[int]:
        """Return indices ordered from most-preferred to least-preferred to keep.

        The retraction loop pops from the end of this list, so the least-preferred
        request is retracted first.
        """
        sorted_indices = list(range(len(reqs)))

        # TODO(lsyin): improve retraction policy for radix cache
        # For spec decoding, filter_batch API can only filter requests from the
        # back, so we can only retract from the back.
        # TODO(sang): Clean up finish path and support better retract policy.
        if not allow_policy_sort:
            return sorted_indices

        def length_key(req: Req) -> Tuple[int, int]:
            return (len(req.output_ids), -len(req.origin_input_ids))

        if server_args.retraction_policy == "priority":
            priority_sign = 1 if server_args.schedule_low_priority_values_first else -1

            def retraction_key(req: Req) -> Tuple[int, int, int]:
                priority = req.priority
                if priority is None:
                    priority = (
                        sys.maxsize
                        if server_args.schedule_low_priority_values_first
                        else -sys.maxsize - 1
                    )
                return (priority * (-priority_sign), *length_key(req))

            sorted_indices.sort(
                key=lambda i: retraction_key(reqs[i]),
                reverse=True,
            )
            return sorted_indices

        sorted_indices.sort(
            key=lambda i: length_key(reqs[i]),
            reverse=True,
        )
        return sorted_indices

    def release_req(self, idx: int, remaing_req_count: int, server_args: ServerArgs):
        release_req(
            req=self.reqs[idx],
            remaing_req_count=remaing_req_count,
            server_args=server_args,
            req_to_token_pool=self.req_to_token_pool,
            token_to_kv_pool_allocator=self.token_to_kv_pool_allocator,
            tree_cache=self.tree_cache,
            hisparse_coordinator=self.hisparse_coordinator,
        )

    def prepare_encoder_info_decode(self):
        # Reset the encoder cached status
        self.encoder_cached = [True] * len(self.reqs)

    def prepare_for_idle(self):
        self.forward_mode = ForwardMode.IDLE
        self.input_ids = torch.empty(0, dtype=torch.int64, device=self.device)
        self.seq_lens = torch.empty(0, dtype=torch.int64, device=self.device)
        self.seq_lens_cpu = torch.empty(0, dtype=torch.int64)
        self.orig_seq_lens = torch.empty(0, dtype=torch.int32, device=self.device)
        self.out_cache_loc = torch.empty(0, dtype=torch.int64, device=self.device)
        self.req_pool_indices = torch.empty(0, dtype=torch.int64, device=self.device)
        self.req_pool_indices_cpu = torch.empty(0, dtype=torch.int64)
        self.seq_lens_sum = 0
        self.extend_num_tokens = 0
        self.sampling_info = SamplingBatchInfo.from_schedule_batch(
            self,
            self.model_config.vocab_size,
        )

    def mamba_lazy_prealloc_at_boundary(self, mamba_track_interval: int):
        """Allocate a temporary second ping-pong slot for reqs at a track boundary.

        In lazy mode each request normally holds only 1 ping-pong slot.
        When seq_len hits a track interval boundary, we allocate the
        second slot so the forward pass can write the new tracked state
        there. The old slot is freed after the forward in
        mamba_lazy_post_decode_at_boundary.
        """
        pool = self.req_to_token_pool
        for i, req in enumerate(self.reqs):
            buf = req.mamba_ping_pong_track_buffer
            assert buf is not None
            # Skip reqs not at a track boundary
            if self.seq_lens_cpu[i].item() % mamba_track_interval != 0:
                continue
            other_idx = 1 - req.mamba_next_track_idx
            if buf[other_idx].item() != -1:
                # With overlap the previous forward's post-processing
                # (which frees this slot) hasn't run yet. Skip.
                continue
            if envs.SGLANG_TEST_MAMBA_LAZY_ALLOC_FAIL.get():
                new_slot = None
            else:
                new_slot = pool.mamba_allocator.alloc(1)
                if new_slot is None:
                    self.tree_cache.evict(EvictParams(num_tokens=0, mamba_num=1))
                    new_slot = pool.mamba_allocator.alloc(1)
                elif peer_needs_mamba_evict(self.tree_cache):
                    # #639b: this rank got its boundary slot straight from the
                    # pool while a peer had to tombstone a cached mamba
                    # checkpoint for its own. The tombstone is visible to
                    # `match_prefix`, so not matching it here diverges the
                    # radix replicas and, one step later, the extend prefix
                    # vector. Only reachable when the scheduler published a
                    # floor, i.e. when the ranks' occupancy actually differs.
                    self.tree_cache.evict(EvictParams(num_tokens=0, mamba_num=1))
            if new_slot is not None:
                pool.set_mamba_ping_pong_slot(req, other_idx, new_slot[0])
                req.mamba_next_track_idx = other_idx

    def cumulate_penalty_output_tokens(self):
        # Under overlap batch.input_ids is just a placeholder here -- the
        # real token is relayed via future_map and resolved at forward
        # entry. So take the last output token from Req directly
        # (origin_input_ids[-1] on the first decode, before any output).
        last_tokens = [
            req.output_ids[-1] if len(req.output_ids) else req.origin_input_ids[-1]
            for req in self.reqs
        ]
        # Non-blocking H2D so this per-step copy doesn't sync behind the forward.
        # pin_memory (matching the prefill-path tensors) keeps the copy async;
        # is_pin_memory_available falls back to pageable on unsupported devices.
        latest_output_ids = torch.tensor(
            last_tokens,
            dtype=torch.int64,
            pin_memory=is_pin_memory_available(self.device),
        ).to(self.device, non_blocking=True)
        self.sampling_info.penalizer_orchestrator.cumulate_output_tokens(
            latest_output_ids
        )

    def prepare_for_decode(self):
        self.forward_mode = ForwardMode.DECODE
        server_args = get_server_args()
        # Decode embeds the last output token via embed_tokens; clear the stale
        # prefill-time tensor so it doesn't leak into ForwardBatch.
        self.input_embeds = None

        # Clear context parallel metadata - CP is only for prefill, not decode
        if hasattr(self, "attn_cp_metadata") and self.attn_cp_metadata is not None:
            self.attn_cp_metadata = None

        if not self.spec_algorithm.is_none():
            # Spec decoding owns decode preparation (allocation, seq-lens bookkeeping).
            from sglang.srt.speculative.spec_utils import spec_prepare_for_decode

            spec_prepare_for_decode(self)
            return

        if self.sampling_info.penalizer_orchestrator.is_required:
            self.cumulate_penalty_output_tokens()

        # input_ids is set at end of previous run_batch (placeholder for
        # overlap; next_token_ids cast for non-overlap).

        if self.model_config.is_encoder_decoder:
            self.prepare_encoder_info_decode()

        # Allocate memory (DSV4-NPU c{4,128}_state alloc lens are computed inside
        # the allocator, triggered from mem_cache/common.py.)
        if self.kv_session_spill_tick:
            # kv-session-offload spill tick: the session's KV lives on host;
            # no device allocation -- the manager assigns the next sentinel
            # slot and writes the req_to_token row. Counter/seq_lens updates
            # below are shared with the default path.
            from sglang.srt.managers.kv_session_offload import (
                get_kv_session_offload_manager,
            )

            self.out_cache_loc = get_kv_session_offload_manager().spill_decode_alloc(
                self
            )
        else:
            self.out_cache_loc = alloc_for_decode(self, token_per_req=1)

        # Update req-level memory management fields
        for req in self.reqs:
            req.decode_batch_idx += 1
            req.kv_committed_len += 1
            req.kv_allocated_len += 1

        if self.enable_overlap:
            # New-tensor avoids racing model_worker_batch refs queued for
            # overlap forward.
            self.seq_lens = self.seq_lens + 1
            self.seq_lens_cpu = self.seq_lens_cpu + 1
            self.orig_seq_lens = self.orig_seq_lens + 1
        else:
            self.seq_lens.add_(1)
            self.seq_lens_cpu.add_(1)
            self.orig_seq_lens.add_(1)
        # Sum is recomputed lazily by ForwardBatch.init_new.
        self.seq_lens_sum = None

        if self.hisparse_coordinator is not None:
            self.hisparse_coordinator.map_last_loc_to_buffer(
                self.seq_lens,
                self.out_cache_loc,
                self.req_pool_indices,
                self.seq_lens_cpu,
                self.req_pool_indices_cpu,
            )

        if server_args.enable_mamba_extra_buffer():
            mamba_track_interval = server_args.mamba_track_interval

            if len(self.reqs) == 0:
                self.mamba_track_indices = torch.empty(
                    (0,), dtype=torch.int64, device=self.device
                )
            else:
                if server_args.enable_mamba_extra_buffer_lazy():
                    self.mamba_lazy_prealloc_at_boundary(mamba_track_interval)
                set_mamba_track_indices_from_reqs(self)

            # async H2D
            self.mamba_track_mask = (
                (self.seq_lens_cpu % mamba_track_interval == 0)
                .pin_memory()
                .to(device=self.device, non_blocking=True)
            )

    def filter_batch(
        self,
        chunked_req_to_exclude: Optional[Union[Req, List[Req]]] = None,
        keep_indices: Optional[List[int]] = None,
    ):
        if keep_indices is None:
            if isinstance(chunked_req_to_exclude, Req):
                chunked_req_to_exclude = [chunked_req_to_exclude]
            elif chunked_req_to_exclude is None:
                chunked_req_to_exclude = []
            keep_indices = [
                i
                for i in range(len(self.reqs))
                if not self.reqs[i].finished()
                and self.reqs[i] not in chunked_req_to_exclude
            ]

        # #622: batch-membership drops are rank-uniformity-critical — feed
        # them to the lockstep sentinel WITH their reason before the list is
        # rebuilt. No-op unless the sentinel is armed.
        if lockstep_sentinel.armed() and len(keep_indices) != len(self.reqs):
            _kept = set(keep_indices)
            for _i, _req in enumerate(self.reqs):
                if _i not in _kept:
                    _fr = _req.finished_reason
                    # matched carries WHICH token/str ended the request — on a
                    # divergent drop this is the diverging value itself
                    # (discriminates sampling divergence from a torn D2H copy).
                    _matched = getattr(_fr, "matched", None)
                    lockstep_sentinel.note_decision(
                        "drop",
                        _req.rid[:16] if isinstance(_req.rid, str) else str(_req.rid),
                        type(_fr).__name__ if _fr is not None else "unfinished",
                        str(_matched) if _matched is not None else "",
                        len(_req.output_ids) if _req.output_ids else 0,
                    )

        if keep_indices is None or len(keep_indices) == 0:
            # Filter out all requests. Stale tensors are left as-is: is_empty()
            # keys off reqs, so callers drop the batch before a forward reads them.
            self.reqs = []
            return

        if len(keep_indices) == len(self.reqs):
            # No need to filter
            return

        keep_indices_device = torch.tensor(
            keep_indices,
            dtype=torch.int64,
            pin_memory=is_pin_memory_available(self.device),
        ).to(self.device, non_blocking=True)

        if self.model_config.is_encoder_decoder:
            self.encoder_lens = self.encoder_lens[keep_indices_device]
            self.encoder_lens_cpu = [self.encoder_lens_cpu[i] for i in keep_indices]

        self.reqs = [self.reqs[i] for i in keep_indices]
        if self.multimodal_inputs is not None:
            self.multimodal_inputs = [self.multimodal_inputs[i] for i in keep_indices]
        self.req_pool_indices = self.req_pool_indices[keep_indices_device]
        self.req_pool_indices_cpu = self.req_pool_indices_cpu[keep_indices]
        self.seq_lens = self.seq_lens[keep_indices_device]
        self.orig_seq_lens = self.orig_seq_lens[keep_indices_device]
        self.out_cache_loc = None
        # Sum is recomputed lazily by ForwardBatch.init_new.
        self.seq_lens_sum = None

        if self.input_ids is not None:
            self.input_ids = self.input_ids[keep_indices_device]
        # Optional under no-verify-sync; resolve_seq_lens repopulates before forward.
        if self.seq_lens_cpu is not None:
            self.seq_lens_cpu = self.seq_lens_cpu[keep_indices]

        self.mamba_track_indices = None
        self.mamba_track_mask = None
        self.mamba_track_seqlens = None
        self.mamba_cow_src_indices = None
        self.mamba_cow_dst_indices = None
        self.mamba_clear_indices = None
        self.return_logprob = any(req.return_logprob for req in self.reqs)
        if self.return_logprob:
            self.top_logprobs_nums = [self.top_logprobs_nums[i] for i in keep_indices]
            self.token_ids_logprobs = [self.token_ids_logprobs[i] for i in keep_indices]
        else:
            self.top_logprobs_nums = None
            self.token_ids_logprobs = None

        self.has_grammar = any(req.grammar for req in self.reqs)

        self.sampling_info.filter_batch(keep_indices, keep_indices_device)
        if self.spec_info:
            self.spec_info.filter_batch(
                new_indices=keep_indices_device,
                has_been_filtered=False,
            )

    def merge_batch(self, other: ScheduleBatch):
        # #631/#656 SELF-MERGE, guarded HERE because this is where it would
        # allocate. Every field below is a torch.cat or an extend, so if
        # ``other is self`` the batch DOUBLES -- req_pool_indices, seq_lens
        # and reqs all concatenate with themselves. Measured 2026-08-09
        # 23:42:45Z: a scheduler slot rebound running_batch and last_batch to
        # one object, the resident count walked 2^23 -> 2^24 -> 2^25 in three
        # seconds, and all three ranks died in sampling_info.merge_batch's
        # torch.cat asking 256 MiB with 138 MiB free.
        #
        # The caller that produced that death is fixed at its own site, so
        # this is deliberately a SECOND line rather than the only one: the
        # audit that followed found merge_batch reachable from seven call
        # sites, one of which (the non-PP overlap-resume path) has the same
        # shape and no identity guard of its own. A guard at one of several
        # call sites is a guard that does not run on the others -- the same
        # lesson the seam census learned when its probe was guarded at one
        # of two entry points.
        #
        # Returning is correct rather than lenient: merging a batch with
        # itself can only double-count requests that are already present.
        # It is logged so the condition stays attributable instead of
        # becoming a silent no-op that hides a caller's state bug.
        if other is self:
            logger.warning(
                "merge_batch called with other is self (bs=%d); refusing. "
                "Its requests are already in this batch and every field "
                "would concatenate with itself. See #631/#656.",
                self.batch_size(),
            )
            return

        # #622: admissions are the other half of batch membership (see the
        # drop notes in filter_batch); a rank admitting a request its peers
        # did not diverges just as hard as one dropping early.
        if lockstep_sentinel.armed():
            for _req in other.reqs:
                lockstep_sentinel.note_decision(
                    "admit",
                    _req.rid[:16] if isinstance(_req.rid, str) else str(_req.rid),
                )

        # Penalizer orchestrator must be merged before Batch.reqs is merged. This is because
        # orchestrator.merge() depends on Batch.reqs during preparation of each penalizers, so it
        # needs to be called with pre-merged Batch.reqs.
        self.sampling_info.merge_batch(other.sampling_info)

        # Encoder-decoder infos
        if self.model_config.is_encoder_decoder:
            self.encoder_lens = torch.cat([self.encoder_lens, other.encoder_lens])
            self.encoder_lens_cpu.extend(other.encoder_lens_cpu)
        self.req_pool_indices = torch.cat(
            [self.req_pool_indices, other.req_pool_indices]
        )
        self.req_pool_indices_cpu = torch.cat(
            [self.req_pool_indices_cpu, other.req_pool_indices_cpu]
        )
        self.seq_lens = torch.cat([self.seq_lens, other.seq_lens])
        self.orig_seq_lens = torch.cat([self.orig_seq_lens, other.orig_seq_lens])
        self.out_cache_loc = None
        # Sum is recomputed lazily by ForwardBatch.init_new.
        self.seq_lens_sum = None
        # Cat only when both sides hold a real token tensor; otherwise drop to
        # None and let resolve_forward_inputs rebuild from the merged
        # req_pool_indices. Mismatch arises e.g. with spec_v1, which keeps its
        # tensor while a relay-staged side is None -- there the worker rebuilds.
        if self.input_ids is not None and other.input_ids is not None:
            self.input_ids = torch.cat([self.input_ids, other.input_ids])
        else:
            self.input_ids = None
        # Optional under no-verify-sync; drop the mirror if either side absent.
        if self.seq_lens_cpu is None or other.seq_lens_cpu is None:
            self.seq_lens_cpu = None
        else:
            self.seq_lens_cpu = torch.cat([self.seq_lens_cpu, other.seq_lens_cpu])
        self.mamba_track_indices = None
        self.mamba_track_mask = None
        self.mamba_track_seqlens = None
        if self.return_logprob and other.return_logprob:
            self.top_logprobs_nums.extend(other.top_logprobs_nums)
            self.token_ids_logprobs.extend(other.token_ids_logprobs)
        elif self.return_logprob:
            self.top_logprobs_nums.extend([0] * len(other.reqs))
            self.token_ids_logprobs.extend([None] * len(other.reqs))
        elif other.return_logprob:
            self.top_logprobs_nums = [0] * len(self.reqs) + other.top_logprobs_nums
            self.token_ids_logprobs = [None] * len(self.reqs) + other.token_ids_logprobs
        self.reqs.extend(other.reqs)
        if self.multimodal_inputs is not None:
            self.multimodal_inputs.extend(other.multimodal_inputs)

        self.return_logprob |= other.return_logprob
        self.has_grammar |= other.has_grammar
        self.return_hidden_states |= other.return_hidden_states
        self.is_prefill_only = self.is_prefill_only and other.is_prefill_only

        if self.spec_info:
            if other.spec_info is None:
                # #631 corpse I. This used to dereference other.spec_info
                # straight through and die as
                #   AttributeError: 'NoneType' has no attribute 'topk_index'
                # three frames down in eagle_info.merge_batch, which named
                # neither batch nor the seam that produced them.
                #
                # It RAISES rather than skipping the merge on purpose. The
                # tempting "if other.spec_info is None: return" turns a
                # loud crash into a silent one: the two batches would be
                # merged as requests while one side's draft state was
                # dropped on the floor, and the wrong tokens would come out
                # of a server that looked healthy. One-sided spec state is
                # never a legal batch pair -- it means a phase seam let TP
                # draft state reach a phase that has no drafter -- so the
                # only correct action here is to say so by name. The seam
                # itself is fixed at the producer, in
                # phase_flip_draft_bootstrap.clear_spec_info_for_unspeculated_phase.
                raise ValueError(
                    "merge_batch: one-sided speculative state -- self has "
                    f"{type(self.spec_info).__name__} for {len(self.reqs)} "
                    f"request(s) (spec_algorithm={self.spec_algorithm}) but "
                    f"other has spec_info=None for {len(other.reqs)} "
                    f"request(s) (spec_algorithm={other.spec_algorithm}). "
                    "A phase seam left draft state reachable from a batch "
                    "built in the other phase; fix the seam, not this check."
                )
            self.spec_info.merge_batch(other.spec_info)
        elif other.spec_info is not None:
            # THE MIRROR, and the more dangerous of the two because the
            # pre-existing code reached it WITHOUT crashing: `if
            # self.spec_info:` is false, the merge completes, other's
            # requests enter self.reqs, and other's draft state is dropped
            # silently. Same illegal pair, opposite roles, no traceback.
            raise ValueError(
                "merge_batch: one-sided speculative state -- self has "
                f"spec_info=None for {len(self.reqs)} request(s) "
                f"(spec_algorithm={self.spec_algorithm}) but other has "
                f"{type(other.spec_info).__name__} for {len(other.reqs)} "
                f"request(s) (spec_algorithm={other.spec_algorithm}). "
                "Merging would silently discard other's draft state; fix "
                "the phase seam that produced the pair."
            )

    def copy(self):
        # Only contain fields that will be used by process_batch_result.
        # Shallow-copy the reqs list so that in-place mutations (filter_batch,
        # merge_batch) on the original don't corrupt this snapshot.
        return ScheduleBatch(
            reqs=self.reqs[:],
            # Per-request extend/prefix lens, snapshotted (sliced like reqs) so the
            # deferred prefill-stats report reads them after the original batch has
            # moved on. prepare_for_extend sets these; mix_with_running mutates them
            # in place. None for decode batches (no extend), which the reader skips.
            extend_lens=self.extend_lens[:] if self.extend_lens is not None else None,
            prefix_lens=self.prefix_lens[:] if self.prefix_lens is not None else None,
            req_to_token_pool=self.req_to_token_pool,
            req_pool_indices=self.req_pool_indices,
            model_config=self.model_config,
            forward_mode=self.forward_mode,
            out_cache_loc=self.out_cache_loc,
            return_logprob=self.return_logprob,
            decoding_reqs=self.decoding_reqs,
            spec_algorithm=self.spec_algorithm,
            spec_info=self.spec_info,
            global_num_tokens=self.global_num_tokens,
            global_num_tokens_for_logprob=self.global_num_tokens_for_logprob,
            can_run_dp_cuda_graph=self.can_run_dp_cuda_graph,
            can_run_dp_breakable_cuda_graph=self.can_run_dp_breakable_cuda_graph,
            is_extend_in_batch=self.is_extend_in_batch,
            all_extend_in_batch=self.all_extend_in_batch,
            is_prefill_only=self.is_prefill_only,
            seq_lens_cpu=self.seq_lens_cpu,
            enable_overlap=self.enable_overlap,
            mamba_track_indices=self.mamba_track_indices,
            mamba_track_mask=self.mamba_track_mask,
            mamba_track_seqlens=self.mamba_track_seqlens,
            dp_cooperation_info=self.dp_cooperation_info,
            prefill_stats=self.prefill_stats,
            fpm_start_time=self.fpm_start_time,
            forward_iter=self.forward_iter,
            kv_session_spill_tick=self.kv_session_spill_tick,
            kv_session_prefill_spill=self.kv_session_prefill_spill,
        )

    def maybe_evict_swa(self):
        if self.tree_cache.supports_swa():
            sliding_window_size = self.tree_cache.sliding_window_size
            server_args = get_server_args()

            release_leaf_lock = (
                envs.SGLANG_OPT_SWA_RELEASE_LEAF_LOCK_AFTER_WINDOW.get()
                and hasattr(self.tree_cache, "dec_swa_lock_only")
            )

            eviction_interval = max(1, envs.SGLANG_SWA_EVICTION_INTERVAL.get())
            swa_maintenance_step = (self.forward_iter or 0) % eviction_interval == 0
            for idx, req in enumerate(self.reqs):
                if self.forward_mode.is_decode():
                    # We set evict_swa condition here with two reasons:
                    # 1. In overlap scheduler, we cannot evict swa when req.decode_batch_idx == 0 since the prev extend batch is still running.
                    # 2. Evict swa every eviction_interval iterations to reduce the overhead.
                    if swa_maintenance_step and req.decode_batch_idx >= 1:
                        self._evict_swa(req, req.seqlen - 1)

                    # DSV4-NPU only (no-op elsewhere): the small paged compress-state
                    # pool must drain every decode step, independent of SWA cadence.
                    maybe_evict_dsv4_state(self, req, req.seqlen - 1)

                    # Once the decode position has moved past the sliding window,
                    # the SWA portion of the prefill-time tree lock is no longer
                    # needed by this request. Convert it from protected to
                    # evictable so SWA LRU can reclaim it under pressure.
                    if (
                        release_leaf_lock
                        and not req.swa_prefix_lock_released
                        and req.swa_uuid_for_lock is not None
                        and req.last_node is not None
                        and req.decode_batch_idx >= sliding_window_size
                    ):
                        self.tree_cache.dec_swa_lock_only(
                            req.last_node, req.swa_uuid_for_lock
                        )
                        req.swa_prefix_lock_released = True
                elif self.forward_mode.is_extend() and self.tree_cache.is_chunk_cache():
                    pre_len = self.prefix_lens[idx]
                    if self.enable_overlap:
                        # In chunked prefill case, when the second extend batch is scheduling, the first extend batch is still running, so we cannot evict swa tokens
                        if req.extend_batch_idx < 2:
                            continue
                        else:
                            pre_len = (
                                pre_len - server_args.chunked_prefill_size
                                if server_args.chunked_prefill_size > 0
                                else pre_len
                            )
                            self._evict_swa(req, pre_len)
                    else:
                        self._evict_swa(req, pre_len)

    def _evict_swa(self, req: Req, pre_len: int):
        assert self.tree_cache.supports_swa(), "prefix cache must support swa"
        free_swa_out_of_window_slots(
            req,
            pre_len,
            sliding_window_size=self.tree_cache.sliding_window_size,
            page_size=self.tree_cache.page_size,
            req_to_token_pool=self.req_to_token_pool,
            token_to_kv_pool_allocator=self.token_to_kv_pool_allocator,
            is_chunk_cache=self.tree_cache.is_chunk_cache(),
        )

    def __str__(self):
        return (
            f"ScheduleBatch(forward_mode={self.forward_mode.name if self.forward_mode else 'None'}, "
            f"#req={(len(self.reqs))})"
        )


class NextBatchPlan(msgspec.Struct):
    batch_to_run: Optional[ScheduleBatch]
    running_batch: ScheduleBatch
