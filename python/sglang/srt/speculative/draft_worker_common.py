from __future__ import annotations

import logging
from copy import deepcopy
from typing import TYPE_CHECKING, Optional

import msgspec
import torch

from sglang.srt.layers.logits_processor import LogitsProcessorOutput
from sglang.srt.layers.moe.utils import speculative_moe_backend_context
from sglang.srt.managers.tp_worker import TpModelWorker
from sglang.srt.model_executor.forward_batch_info import CaptureHiddenMode
from sglang.srt.runtime_context import get_context, get_server_args
from sglang.srt.server_args import ServerArgs
from sglang.srt.speculative.dflash_info import DFlashVerifyInput
from sglang.srt.speculative.dflash_info_v2 import DFlashDraftInputV2

if TYPE_CHECKING:
    from sglang.srt.configs.model_config import ModelConfig
    from sglang.srt.model_executor.model_runner import ModelRunner

logger = logging.getLogger(__name__)

_SUPPORTED_DRAFT_BACKENDS = ("flashinfer", "fa3", "fa4", "triton", "ascend")


class DraftWorkerBundle(msgspec.Struct, frozen=True):
    draft_worker: TpModelWorker
    draft_model_runner: ModelRunner
    draft_model: torch.nn.Module
    resolved_attention_backend: str


def _resolve_draft_attention_backend_fallback(
    *, draft_server_args: ServerArgs, algo_label: str
) -> str:
    draft_backend = draft_server_args.speculative_draft_attention_backend
    if draft_backend is None:
        draft_backend, _ = draft_server_args.get_attention_backends()
    if draft_backend is None:
        return "triton" if torch.version.hip else "flashinfer"
    if draft_backend not in _SUPPORTED_DRAFT_BACKENDS:
        fallback = "triton" if torch.version.hip else "flashinfer"
        logger.warning(
            "%s draft worker only supports attention_backend in %s for now, "
            "but got %r. Falling back to '%s'.",
            algo_label,
            _SUPPORTED_DRAFT_BACKENDS,
            draft_backend,
            fallback,
        )
        return fallback
    return draft_backend


def _refuse_unsupported_speculative_moe_backend(
    *, server_args: ServerArgs, algo_label: str
) -> None:
    """Refuse a draft MoE runner backend this rank's card cannot run, by name.

    The kernel-level gate exists already, but it fires deep inside
    ``process_weights_after_loading`` as a bare
    ``RuntimeError("MXFP4 Marlin requires SM90 or SM120.")`` -- after the
    weights are on the card and with nothing said about which rank, which
    flag, or what to do. Under draft-solo placement the answer is usually
    "put the draft on the SM120 card", which is a placement flag away, so the
    refusal names it.

    Deliberately per-rank: a heterogeneous group (this rig is sm120 + 2x sm86)
    has different answers on different ranks, and only the ranks that actually
    build draft weights are affected -- a solo SHADOW builds on the ``meta``
    device and never reaches a kernel.
    """
    from sglang.srt.layers.moe.utils import get_speculative_moe_runner_backend

    if not torch.cuda.is_available():
        return
    backend = get_speculative_moe_runner_backend()
    if not backend.is_marlin():
        return
    from sglang.srt.utils.common import is_sm90_supported, is_sm120_supported

    if is_sm90_supported() or is_sm120_supported():
        return
    major, minor = torch.cuda.get_device_capability()
    raise ValueError(
        f"--speculative-moe-runner-backend marlin is not runnable on this "
        f"rank's GPU (compute capability {major}.{minor}, device "
        f"{torch.cuda.get_device_name()!r}): the MXFP4 Marlin MoE kernel "
        f"needs SM90 or SM120. The {algo_label} draft would load and then die "
        "in process_weights_after_loading. On a heterogeneous group, put the "
        "whole draft on the capable card with "
        "--speculative-draft-placement solo --speculative-draft-gpu <index>; "
        "otherwise choose a runner backend all ranks can run."
    )


def build_draft_tp_worker(
    *,
    server_args: ServerArgs,
    gpu_id: int,
    tp_rank: int,
    dp_rank: Optional[int],
    moe_ep_rank: int,
    attn_cp_rank: int,
    moe_dp_rank: int,
    nccl_port: int,
    target_model_config: ModelConfig,
    algo_label: str,
    attention_backend_override: Optional[str] = None,
) -> DraftWorkerBundle:
    draft_server_args = deepcopy(server_args)
    # An override names a draft-specific backend the caller has already
    # validated (e.g. a self-drafting architecture); it skips the generic
    # supported-backend fallback below.
    draft_backend = attention_backend_override or (
        _resolve_draft_attention_backend_fallback(
            draft_server_args=draft_server_args, algo_label=algo_label
        )
    )
    # Post-resolution ServerArgs rejects bare assignment; route the draft-copy
    # adjustments through the audited mutation point. Keep the resolved value
    # on speculative_draft_attention_backend: downstream draft-worker logic
    # keys on that field (backend selection in _get_attention_backend and the
    # fa4-draft KV dtype override in configure_kv_cache_dtype), so nulling it
    # would silently skip those paths. context_length keeps the draft aligned
    # with the target.
    draft_server_args.override(
        "draft_worker.build",
        skip_tokenizer_init=True,
        speculative_draft_attention_backend=draft_backend,
        prefill_attention_backend=None,
        decode_attention_backend=None,
        attention_backend=draft_backend,
        context_length=target_model_config.context_len,
    )

    _refuse_unsupported_speculative_moe_backend(
        server_args=server_args, algo_label=algo_label
    )

    saved_server_args = get_server_args()
    try:
        # --speculative-moe-runner-backend applies to the DRAFT model's MoE
        # layers. The EAGLE-family workers already enter this context around
        # their model build; the self-drafting block models (DFLASH / DSPARK)
        # did not, so their draft silently inherited the TARGET's runner
        # backend. That is load-bearing for a DSpark head whose routed experts
        # are MXFP4: Mxfp4MarlinMoEMethod is selected from
        # get_moe_runner_backend() at get_quant_method time (fp8.py), i.e.
        # inside this build, and a GGUF target's backend is not marlin.
        # Extending the existing flag rather than adding a draft-specific one
        # is deliberate (#382 canonical-flag rule).
        with speculative_moe_backend_context():
            draft_worker = TpModelWorker(
                server_args=draft_server_args,
                gpu_id=gpu_id,
                tp_rank=tp_rank,
                moe_ep_rank=moe_ep_rank,
                pp_rank=0,
                attn_cp_rank=attn_cp_rank,
                moe_dp_rank=moe_dp_rank,
                dp_rank=dp_rank,
                nccl_port=nccl_port,
                is_draft_worker=True,
            )
    finally:
        get_context().set_server_args(saved_server_args)

    draft_model_runner = draft_worker.model_runner
    draft_worker.draft_runner = draft_model_runner
    return DraftWorkerBundle(
        draft_worker=draft_worker,
        draft_model_runner=draft_model_runner,
        draft_model=draft_model_runner.model,
        resolved_attention_backend=draft_backend,
    )


def make_draft_input_v2(
    *,
    bonus_tokens: torch.Tensor,
    new_seq_lens: torch.Tensor,
) -> DFlashDraftInputV2:
    bs = int(new_seq_lens.numel())
    device = bonus_tokens.device
    return DFlashDraftInputV2(
        topk_p=torch.empty((bs, 0), device=device, dtype=torch.float32),
        topk_index=torch.empty((bs, 0), device=device, dtype=torch.int64),
        bonus_tokens=bonus_tokens.to(dtype=torch.int64),
        new_seq_lens=new_seq_lens.to(dtype=torch.int64),
        hidden_states=torch.empty((bs, 0), device=device, dtype=torch.float16),
    )


def make_draft_block_spec_info(
    *,
    draft_token_num: int,
    device: torch.device,
) -> DFlashVerifyInput:
    return DFlashVerifyInput(
        draft_token=torch.empty((0,), dtype=torch.long, device=device),
        positions=torch.empty((0,), dtype=torch.int64, device=device),
        draft_token_num=int(draft_token_num),
        custom_mask=None,
        capture_hidden_mode=CaptureHiddenMode.NULL,
    )


def make_draft_sampler_capture_hook(draft_sampler):

    def capture_hook(runner, out, forward_batch, num_tokens):
        del runner, num_tokens
        if not isinstance(out, LogitsProcessorOutput) or out.hidden_states is None:
            raise RuntimeError(
                "draft sampler set but the draft forward has no "
                "hidden_states to capture into the graph."
            )
        draft_sampler(out.hidden_states, forward_batch.input_ids)

    return capture_hook


def build_block_pos_offsets(*, length: int, device: torch.device) -> torch.Tensor:
    return torch.arange(int(length), device=device, dtype=torch.int64)
