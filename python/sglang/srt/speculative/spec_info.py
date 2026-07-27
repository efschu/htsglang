from __future__ import annotations

from abc import ABC, abstractmethod
from enum import Enum, IntEnum, auto
from typing import TYPE_CHECKING, Callable, List, Optional, Tuple, Type, Union

import torch

from sglang.srt.speculative.spec_registry import (
    CustomSpecAlgo,
    ServerArgsValidator,
    WorkerFactory,
)
from sglang.srt.speculative.spec_registry import get_spec as _get_registered_spec
from sglang.srt.speculative.spec_registry import (
    register_algorithm as _register_algorithm,
)

if TYPE_CHECKING:
    from sglang.srt.managers.overlap_utils import FutureMap
    from sglang.srt.managers.schedule_batch import ScheduleBatch
    from sglang.srt.managers.tp_worker import TpModelWorker
    from sglang.srt.server_args import ServerArgs
    from sglang.srt.speculative.base_spec_worker import BaseSpecWorker
    from sglang.srt.speculative.ngram_worker import NGRAMWorker
    from sglang.srt.speculative.ragged_verify import RaggedVerifyLayout


class SpeculativeAlgorithm(Enum):
    """Builtin speculative decoding algorithms. Plugin-registered ones are
    ``CustomSpecAlgo`` instances; ``from_string`` returns either type, and
    both expose the same ``is_*()`` / ``create_worker`` interface so callers
    dispatch uniformly without isinstance checks.
    """

    DFLASH = auto()
    DSPARK = auto()
    EAGLE = auto()
    EAGLE3 = auto()
    FROZEN_KV_MTP = auto()
    STANDALONE = auto()
    NGRAM = auto()
    NONE = auto()

    @classmethod
    def from_string(
        cls, name: Optional[str]
    ) -> Union[SpeculativeAlgorithm, CustomSpecAlgo]:
        if name is None:
            return cls.NONE
        upper = name.upper()
        try:
            return cls[upper]
        except KeyError:
            pass
        spec = _get_registered_spec(upper)
        if spec is not None:
            return spec
        raise ValueError(f"Unknown speculative algorithm name: {name}")

    @classmethod
    def register(
        cls,
        name: str,
        *,
        supports_overlap: bool = False,
        validate_server_args: Optional[ServerArgsValidator] = None,
        spec_class: Type[CustomSpecAlgo] = CustomSpecAlgo,
    ) -> Callable[[WorkerFactory], WorkerFactory]:
        """Decorator to register a plugin speculative algorithm. The factory
        takes ``server_args`` and returns the worker class. Pass a
        ``CustomSpecAlgo`` subclass via ``spec_class`` to override any
        ``is_*()`` / ``create_worker`` method.

        Example:
            @SpeculativeAlgorithm.register("MY_SPEC", supports_overlap=False)
            def _factory(server_args):
                return MySpecWorker
        """
        return _register_algorithm(
            name,
            supports_overlap=supports_overlap,
            validate_server_args=validate_server_args,
            spec_class=spec_class,
        )

    def is_some(self) -> bool:
        return self != SpeculativeAlgorithm.NONE

    def is_none(self) -> bool:
        return self == SpeculativeAlgorithm.NONE

    def is_speculative(self) -> bool:
        return self != SpeculativeAlgorithm.NONE

    def is_eagle(self) -> bool:
        # FIXME(kpham_sgl): Remove FROZEN_KV_MTP here once we
        # have established support for it in the scheduler.
        return self in (
            SpeculativeAlgorithm.EAGLE,
            SpeculativeAlgorithm.EAGLE3,
            SpeculativeAlgorithm.FROZEN_KV_MTP,
        )

    def is_eagle3(self) -> bool:
        return self == SpeculativeAlgorithm.EAGLE3

    def is_frozen_kv_mtp(self) -> bool:
        return self == SpeculativeAlgorithm.FROZEN_KV_MTP

    def is_dflash(self) -> bool:
        return self == SpeculativeAlgorithm.DFLASH

    def is_dspark(self) -> bool:
        return self == SpeculativeAlgorithm.DSPARK

    def is_dflash_family(self) -> bool:
        return self.is_dflash() or self.is_dspark()

    def is_standalone(self) -> bool:
        return self == SpeculativeAlgorithm.STANDALONE

    def is_ngram(self) -> bool:
        return self == SpeculativeAlgorithm.NGRAM

    def supports_target_verify_for_draft(self) -> bool:
        return self.is_dflash_family()

    def supports_ragged_verify(self) -> bool:
        """Whether this algorithm's verify step may carry a RaggedVerifyLayout
        (per-request verify lengths); gates the token-bucket-keyed verify
        graphs in the decode cuda graph runner."""
        return self.is_dspark()

    def has_draft_kv(self) -> bool:
        """Whether the draft phase writes KV chains. NGRAM does not (its tree
        lives only in the verify mask), so per-decode KV sizing needs no
        per-topk page rounding; see get_alloc_len_per_decode."""
        return not self.is_ngram()

    def carries_draft_hidden_states(self) -> bool:
        """Whether the disagg prefill->decode transfer carries draft hidden
        states (EAGLE-family only; STANDALONE's vanilla draft ignores them)."""
        return self.is_eagle()

    def create_future_map(
        self,
        device: torch.device,
        req_to_token_pool,
        needs_cpu_seq_lens: bool = True,
        needs_confidence_relay: bool = False,
    ) -> FutureMap:
        from sglang.srt.managers.overlap_utils import FutureMap

        return FutureMap(
            device,
            self,
            req_to_token_pool,
            needs_cpu_seq_lens,
            needs_confidence_relay,
        )

    def build_disagg_draft_input(
        self,
        batch: ScheduleBatch,
        server_args: ServerArgs,
        last_tokens_tensor: torch.Tensor,
        future_map: FutureMap,
    ) -> Optional[SpecInput]:
        if self.is_eagle():
            from sglang.srt.speculative.eagle_disaggregation import (
                build_eagle_disagg_draft_input,
            )

            return build_eagle_disagg_draft_input(
                batch, server_args, last_tokens_tensor, future_map
            )
        return None

    def need_topk(self) -> bool:
        return self.is_eagle() or self.is_standalone()

    def handle_server_args(self, server_args: ServerArgs) -> None:
        """Hook for per-algorithm server args mutation.

        In-place updated.
        """
        from sglang.srt.arg_groups.speculative_hook import (
            _handle_dflash,
            _handle_dspark,
            _handle_eagle_family,
            _handle_frozen_kv_mtp,
            _handle_ngram,
        )

        # Validate for every algorithm at startup: the metrics paths read the
        # ragged-verify mode env and must not be where a typo'd value raises.
        from sglang.srt.speculative.ragged_verify import read_ragged_verify_mode

        read_ragged_verify_mode()

        if self.is_dflash():
            _handle_dflash(server_args)
        elif self.is_dspark():
            _handle_dspark(server_args)
        elif self.is_frozen_kv_mtp():
            _handle_frozen_kv_mtp(server_args)
        elif self.is_eagle() or self.is_standalone():
            _handle_eagle_family(server_args)
        elif self.is_ngram():
            _handle_ngram(server_args)

    def get_num_tokens_per_bs_for_target_verify(
        self, num_draft_tokens: int, is_draft_worker: bool
    ) -> int:
        # FIXME: Remove this after the forward mode refactor. Target verify is
        # essentially a fixed sequence length prefill/extend with full cuda
        # graph support. We can use it for target verify, or we can use it for
        # other cases which is not target verify but fixed length prefill.
        # Here, we expose this interface to allow the other use cases.
        if self.is_dspark() and is_draft_worker:
            return num_draft_tokens - 1
        return num_draft_tokens

    def create_worker(
        self, server_args: ServerArgs
    ) -> Optional[Union[Type[BaseSpecWorker], Type[TpModelWorker], Type[NGRAMWorker]]]:
        assert (
            not self.is_none()
        ), "Cannot create worker for NONE speculative algorithm."

        if getattr(server_args, "speculative_cross_algorithm", False):
            # T156 stage 2: one meta-worker hosts both the NEXTN/MTP and the
            # DFLASH sub-worker; `self` is the FORCED rung's algorithm (the
            # normalized server args carry its shape).
            from sglang.srt.speculative.cross_algo_worker import CrossAlgoWorker

            return CrossAlgoWorker

        if self.is_dflash():
            # V2 worker drives both overlap and non-overlap (scheduler runs it
            # synchronously when overlap is disabled), same as EAGLE.
            from sglang.srt.speculative.dflash_worker_v2 import DFlashWorkerV2

            return DFlashWorkerV2

        if self.is_dspark():
            from sglang.srt.speculative.dspark_components.dspark_worker_v2 import (
                DSparkWorkerV2,
            )

            return DSparkWorkerV2

        if self.is_frozen_kv_mtp():
            # V2 worker drives both overlap and non-overlap (scheduler runs it
            # synchronously when overlap is disabled), same as EAGLE.
            from sglang.srt.speculative.frozen_kv_mtp_worker_v2 import (
                FrozenKVMTPWorkerV2,
            )

            return FrozenKVMTPWorkerV2

        # EAGLE / EAGLE3 / STANDALONE / MULTI_LAYER always use the V2 worker,
        # even with overlap disabled (scheduler drives it synchronously).
        if self.is_eagle() and server_args.enable_multi_layer_eagle:
            from sglang.srt.speculative.multi_layer_eagle_worker_v2 import (
                MultiLayerEagleWorkerV2,
            )

            return MultiLayerEagleWorkerV2

        elif self.is_eagle():
            from sglang.srt.speculative.eagle_worker_v2 import EAGLEWorkerV2

            return EAGLEWorkerV2
        elif self.is_standalone():
            from sglang.srt.speculative.standalone_worker_v2 import (
                StandaloneWorkerV2,
            )

            return StandaloneWorkerV2
        elif self.is_ngram():
            from sglang.srt.speculative.ngram_worker import NGRAMWorker

            return NGRAMWorker

        raise ValueError("Unreachable code path in create_worker.")


class SpecInputType(IntEnum):
    # NOTE: introduce this to distinguish the SpecInput types of multiple algorithms when asserting in attention backends.
    # If all algorithms can share the same datastrucutre of draft_input and verify_input, consider simplify it
    EAGLE_DRAFT = auto()
    EAGLE_DRAFT_EXTEND = auto()
    EAGLE_VERIFY = auto()
    FROZEN_KV_MTP_DRAFT = auto()
    FROZEN_KV_MTP_VERIFY = auto()
    DFLASH_DRAFT = auto()
    DFLASH_VERIFY = auto()
    NGRAM_VERIFY = auto()


def reject_ngram_verify_under_dcp(dcp_size: int) -> None:
    """Boot gate: NGRAM speculation has no DCP-aware target-verify build.

    ``NGRAM_VERIFY`` is deliberately outside both backends'
    ``_DCP_VERIFY_SPEC_INPUT_TYPES`` (#180.3): the M4 verify split assumes a
    uniform ``draft_token_num`` query block per request over a LINEAR draft
    chain, and an ngram draft is a corpus-matched TREE whose breadth comes
    from ``--speculative-ngram-{min,max}-bfs-breadth``. That exclusion is
    correct and stays.

    Both backends refuse the combination LOUDLY, so nothing is ever computed
    wrong -- but both refuse in the target-verify METADATA BUILD, i.e. at the
    first verify step of the first request, after the full weight load. The
    condition is decidable from ``server_args`` alone, so the primary call
    site is ``ServerArgs._handle_dcp_validation`` -- argument resolution, in
    the launcher, before anything is loaded. ``NGRAMWorker.__init__`` repeats
    the call as a backstop for directly constructed workers.

    ``dcp_size > 1`` is the exact reachable condition, not a widening of it:

    * Triton's target-verify arm gates on ``self.dcp_size > 1`` and raises
      ``NotImplementedError`` for any spec input type outside the set -- even
      and uneven DCP alike, because either way the pool rows are compact while
      the non-DCP verify metadata builds GLOBAL allocator ids.
    * flashinfer's split is an ``elif`` on ``uneven_dcp`` AND membership, so an
      ngram verify misses it and drops into the generic spec branch, which
      leaves ``use_ragged`` False while ``_forward_extend_dcp`` runs the ragged
      stage unconditionally -> ``AttributeError`` (the shape DFLASH verify hit
      before it joined the set). Its even-DCP case cannot arise:
      ``reject_silently_inert_dcp`` already refuses a ``--dcp-size`` this
      backend would ignore.

    ``dcp_size <= 1`` is inert, which is what keeps every stock boot untouched.
    """
    if dcp_size <= 1:
        return
    raise ValueError(
        f"Speculative algorithm 'ngram' cannot run under --dcp-size {dcp_size}: "
        "NGRAM_VERIFY is not one of the target-verify layouts the decode-context-"
        "parallel verify split serves (EAGLE_VERIFY, DFLASH_VERIFY). The split "
        "assumes a uniform draft-token query block per request over a LINEAR "
        "draft chain, while an ngram draft is a corpus-matched tree; reading the "
        "committed prefix the non-DCP way would dereference GLOBAL allocator ids "
        "against a compact token-sharded pool. Use --dcp-size 1, or a chain "
        "speculative algorithm (eagle / mtp with --speculative-eagle-topk 1)."
    )


def reject_frozen_kv_mtp_verify_under_dcp(dcp_size: int) -> None:
    """Boot gate: FROZEN_KV_MTP has no DCP-aware target-verify build.

    Same late-detection shape as ``reject_ngram_verify_under_dcp``, one door
    over: ``FROZEN_KV_MTP_VERIFY`` is outside both backends'
    ``_DCP_VERIFY_SPEC_INPUT_TYPES``, so under any ``dcp_size > 1`` the first
    target-verify metadata build refuses it (Triton raises
    ``NotImplementedError`` explicitly; flashinfer's verify split misses the
    type and fails in the ragged stage) -- after the weights are loaded, the
    pools are sized and the graphs are captured.

    The exclusion itself is structural, not an oversight in the set: the
    frozen draft reads the TARGET's KV cache in place (it owns no draft KV
    pool), and decode context parallelism shards that cache by token across
    the DCP ranks, so no single rank holds the rows the draft dereferences.
    Serving FROZEN_KV_MTP under DCP would need a DCP-aware draft read on top
    of the verify-split membership, which is why closing the gap is not a
    one-line set addition.

    Primary call site is ``ServerArgs._handle_dcp_validation`` (argument
    resolution, before any load); ``FrozenKVMTPWorkerV2.__init__`` repeats the
    call as a backstop for directly constructed workers. ``dcp_size <= 1``
    is inert, which keeps every stock boot untouched.
    """
    if dcp_size <= 1:
        return
    raise ValueError(
        f"Speculative algorithm 'FROZEN_KV_MTP' cannot run under --dcp-size "
        f"{dcp_size}: FROZEN_KV_MTP_VERIFY is not one of the target-verify "
        "layouts the decode-context-parallel verify split serves (EAGLE_VERIFY, "
        "DFLASH_VERIFY), and the frozen draft reads the target KV cache in "
        "place, which decode context parallelism shards by token across ranks "
        "-- no single rank holds the rows the draft would read. Use --dcp-size "
        "1, or a chain speculative algorithm that owns its draft KV (eagle / "
        "mtp with --speculative-eagle-topk 1)."
    )


class SpecInput(ABC):
    # Per-request verify lengths for the ragged-verify graphs (see
    # sglang.srt.speculative.ragged_verify); verify inputs of algorithms with
    # supports_ragged_verify() override it per step. Must stay a class-level
    # default, not an __init__ assignment: dataclass subclasses declare it as
    # a field and run __post_init__ -> super().__init__ *after* field
    # assignment, so an init-time default would clobber the passed layout.
    ragged_verify_layout: Optional[RaggedVerifyLayout] = None

    def __init__(self, spec_input_type: SpecInputType):
        self.spec_input_type = spec_input_type

    # Cross-algorithm phase guards. Used by attention backends and
    # ForwardBatch padding logic to dispatch on phase without hardcoding the
    # specific algo class (EAGLE / FROZEN_KV_MTP / DFLASH / NGRAM each have
    # their own draft / verify SpecInput subclasses).
    def is_draft_input(self) -> bool:
        return self.spec_input_type in {
            SpecInputType.EAGLE_DRAFT,
            SpecInputType.EAGLE_DRAFT_EXTEND,
            SpecInputType.FROZEN_KV_MTP_DRAFT,
            SpecInputType.DFLASH_DRAFT,
        }

    def is_verify_input(self) -> bool:
        return self.spec_input_type in {
            SpecInputType.EAGLE_VERIFY,
            SpecInputType.FROZEN_KV_MTP_VERIFY,
            SpecInputType.DFLASH_VERIFY,
            SpecInputType.NGRAM_VERIFY,
        }

    @abstractmethod
    def get_spec_adjust_token_coefficient(self) -> Tuple[int, int]:
        pass

    def get_spec_adjusted_global_num_tokens(
        self, batch: ScheduleBatch
    ) -> Tuple[List[int], List[int]]:
        c1, c2 = self.get_spec_adjust_token_coefficient()
        global_num_tokens = [x * c1 for x in batch.global_num_tokens]
        global_num_tokens_for_logprob = [
            x * c2 for x in batch.global_num_tokens_for_logprob
        ]
        return global_num_tokens, global_num_tokens_for_logprob


def create_dummy_verify_input(
    spec_algorithm: SpeculativeAlgorithm,
    server_args: ServerArgs,
    custom_mask: torch.Tensor,
    num_tokens_per_bs: int,
    is_draft_worker: bool,
) -> Optional[SpecInput]:
    """Dummy verify ``SpecInput`` for CUDA-graph capture (per-algorithm dispatch)."""
    from sglang.srt.model_executor.forward_batch_info import CaptureHiddenMode

    spec_info = None
    if spec_algorithm.is_eagle() or spec_algorithm.is_standalone():
        from sglang.srt.speculative.eagle_info import EagleVerifyInput

        if is_draft_worker:
            raise RuntimeError("This should not happen.")
        else:
            spec_info = EagleVerifyInput(
                draft_token=None,
                custom_mask=custom_mask,
                positions=None,
                retrieve_index=None,
                retrieve_next_token=None,
                retrieve_next_sibling=None,
                retrieve_cum_len=None,
                spec_steps=server_args.speculative_num_steps,
                topk=server_args.speculative_eagle_topk,
                draft_token_num=server_args.speculative_num_draft_tokens,
                capture_hidden_mode=CaptureHiddenMode.FULL,
                seq_lens_sum=None,
                seq_lens_cpu=None,
            )
    elif spec_algorithm.is_dflash_family():
        from sglang.srt.speculative.dflash_info import DFlashVerifyInput

        # Dummy warmup only needs shape metadata; avoid forcing custom-mask mode.
        spec_info = DFlashVerifyInput(
            draft_token=None,
            positions=None,
            draft_token_num=server_args.speculative_num_draft_tokens,
            custom_mask=None,
            capture_hidden_mode=(
                CaptureHiddenMode.NULL if is_draft_worker else CaptureHiddenMode.FULL
            ),
        )

    elif spec_algorithm.is_ngram():
        from sglang.srt.speculative.ngram_info import NgramVerifyInput

        spec_info = NgramVerifyInput(
            draft_token=None,
            custom_mask=custom_mask,
            positions=None,
            retrieve_index=None,
            retrieve_next_token=None,
            retrieve_next_sibling=None,
            draft_token_num=num_tokens_per_bs,
        )
        spec_info.capture_hidden_mode = CaptureHiddenMode.NULL

    return spec_info
