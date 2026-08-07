from __future__ import annotations

import dataclasses
import enum
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, List, Optional

import numpy as np
import numpy.typing as npt

from sglang.srt.server_args import ServerArgs

if TYPE_CHECKING:
    from sglang.srt.disaggregation.utils import DisaggregationMode


class StateType(str, enum.Enum):
    MAMBA = "mamba"
    SWA = "swa"
    DSA = "dsa"
    MINIMAX_INDEX_K = "minimax_index_k"
    # DeepSeek-V4 unified_kv SWA ring: addressed per-row by ring slot
    # (req_pool_idx * ring_stride + pos % ring_stride), needs its own component.
    SWA_RING = "swa_ring"
    # DeepSeek-V4 online C128 request-scoped state.
    C128_STATE = "c128_state"


@dataclasses.dataclass
class KVTransferMetric:
    # Backends that cannot isolate transfer latency can leave this as None.
    transfer_latency_s: Optional[float] = None
    # Backends that cannot isolate allocation wait latency can leave this as None.
    alloc_latency_s: Optional[float] = None
    transfer_total_bytes: Optional[int] = None


class KVArgs:
    engine_rank: int
    kv_data_ptrs: List[int]
    kv_data_lens: List[int]
    kv_item_lens: List[int]
    aux_data_ptrs: List[int]
    aux_data_lens: List[int]
    aux_item_lens: List[int]
    state_types: List[StateType]
    state_data_ptrs: List[List[int]]
    state_data_lens: List[List[int]]
    state_item_lens: List[List[int]]
    # Per-tensor TP slice dim, used when prefill/decode attn_tp_size differ.
    state_dim_per_tensor: List[List[int]]
    # Per-tensor slice START offset (in dim units) of this rank's shard within
    # the unsharded state. Only populated on decode under an uneven-TP
    # (--rank-tp-ratio) plan, where `(rank * dim) % src_dim` is wrong; parallel
    # to state_dim_per_tensor. Empty lists -> uniform legacy formula.
    state_dim_offsets: List[List[int]]
    is_hybrid_mla_backend: bool
    ib_device: str
    ib_traffic_class: str
    gpu_id: int
    kv_head_num: int
    total_kv_head_num: int
    page_size: int
    # for system dp
    system_dp_rank: int
    # for pp prefill
    pp_rank: int
    prefill_start_layer: int
    # Absolute end layer (exclusive) for this prefill PP stage. Needed to
    # reconstruct PP sub-ranges when kv_data_ptrs does not use a flat
    # layer-indexed layout (e.g. DeepSeek V4's buffer-type-organized flat
    # list).
    prefill_end_layer: Optional[int]
    # For DeepSeek V4 (and other compressed-MLA) memory pools only.
    # Full-model compression ratio per layer (entries are 0/4/128). Used by
    # the connection layer to slice the buffer-type-organized flat list in a
    # PP-aware manner.
    mla_compression_ratios: Optional[List[int]]
    # #646: section boundary of the flat kv_data_ptrs / kv_item_lens lists.
    # The registration is [target K(L), target V(L)] followed, when this arm
    # holds a draft model, by [draft K(D), draft V(D)]. The transfer layer
    # must not infer the K/V boundary from the list length, because the draft
    # entries move it. Both are set by prefill.py / decode.py at registration;
    # code that reads them uses getattr with a default so a backend that never
    # sets them keeps its previous behaviour.
    num_target_kv_buffers: int
    num_draft_kv_buffers: int
    # Only used of npu, for kv buf groups
    kv_buf_groups: int
    # Only used of npu, for decode total kv layers
    total_kv_layers: int


class KVPoll:
    Failed = 0
    Bootstrapping = 1
    WaitingForInput = 2
    Transferring = 3
    Success = 4


class BaseKVManager(ABC):
    """Base class for managing transfer states"""

    @abstractmethod
    def __init__(
        self,
        args: KVArgs,
        disaggregation_mode: DisaggregationMode,
        server_args: ServerArgs,
        is_mla_backend: Optional[bool] = False,
    ): ...

    @abstractmethod
    def register_to_bootstrap(self):
        """Register prefill server info to the bootstrap server."""
        ...


class BaseKVSender(ABC):
    @abstractmethod
    def __init__(
        self,
        mgr: BaseKVManager,
        bootstrap_addr: str,
        bootstrap_room: int,
        dest_tp_ranks: List[int],
        pp_rank: int,
        req_has_disagg_prefill_dp_rank: bool = False,
    ): ...

    @abstractmethod
    def init(self, num_kv_indices: int, aux_index: Optional[int] = None):
        """
        Set req's index metadata locally or notify the decoder server about the kv indices length and aux index.
        """
        ...

    @abstractmethod
    def send(
        self,
        kv_indices: npt.NDArray[np.int32],
        state_indices: Optional[List] = None,
    ):
        """
        Send the kv cache at the given kv indices and the extra cache/state at the given indices to the decoder server.
        """
        ...

    def pop_decode_prefix_len(self) -> int:
        return 0

    def should_send_kv_chunk(self, num_pages: int, last_chunk: bool) -> bool:
        return num_pages > 0

    @abstractmethod
    def get_transfer_metric(self) -> KVTransferMetric:
        """Return backend-specific transfer metrics for this sender."""
        ...

    @abstractmethod
    def poll(self) -> KVPoll:
        """
        Check the status of the kv cache transfer.
        """
        ...

    @abstractmethod
    def failure_exception(self):
        """
        Raise an exception if the kv cache transfer fails.
        """
        ...


class BaseKVReceiver(ABC):
    @abstractmethod
    def __init__(
        self,
        mgr: BaseKVManager,
        bootstrap_addr: str,
        bootstrap_room: Optional[int] = None,
    ): ...

    @abstractmethod
    def init(
        self,
        prefill_dp_rank: int,
    ):
        """
        Resolve bootstrap metadata and mark the receiver ready for transfer metadata.
        """
        ...

    @abstractmethod
    def send_metadata(
        self,
        kv_indices: npt.NDArray[np.int32],
        aux_index: Optional[int] = None,
        state_indices: Optional[List] = None,
        decode_prefix_len: Optional[int] = None,
        owned_ordinals: Optional[npt.NDArray[np.int32]] = None,
    ):
        """
        Notify the prefill server about the kv indices, aux index, and state_indices.

        ``owned_ordinals``: when the decode side token-shards its KV pool
        (fork DCP with replicated kv-heads), the ordinals (0-based positions
        within the sent token range) that THIS rank owns, parallel to
        ``kv_indices`` (which are then this rank's compact pool rows). The
        prefill sender uses them to filter each chunk's source rows.
        """
        ...

    @abstractmethod
    def poll(self) -> KVPoll:
        """
        Check the status of the kv cache transfer.
        """
        ...

    @abstractmethod
    def failure_exception(self):
        """
        Raise an exception if the kv cache transfer fails.
        """
        ...

    def clear(self):
        """
        Clear any internal states.
        """
        pass

    def abort(self):
        """
        Abort the current transfer.
        """
        pass


class BaseKVBootstrapServer(ABC):
    @abstractmethod
    def __init__(self, host: str, port: int): ...
