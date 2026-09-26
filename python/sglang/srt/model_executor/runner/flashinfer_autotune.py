# Copyright 2023-2026 SGLang Team
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
from __future__ import annotations

import contextlib
import datetime
import hashlib
import logging
from pathlib import Path
from typing import TYPE_CHECKING, Callable, Optional

import torch

from sglang.srt.environ import envs
from sglang.srt.utils import cuda_sm_at_least

if TYPE_CHECKING:
    from sglang.srt.model_executor.model_runner import ModelRunner
    from sglang.srt.model_executor.runner.base_runner import BaseRunner

logger = logging.getLogger(__name__)


def should_run_flashinfer_autotune(
    model_runner: ModelRunner, *, for_speculative_draft: bool = False
) -> bool:
    """Check if flashinfer autotune should be run."""
    mr = model_runner
    if mr.device != "cuda":
        return False
    if mr.server_args.disable_flashinfer_autotune:
        return False

    # FORM A (F9): autotune drives a DUMMY MODEL FORWARD. On the host that
    # forward issues the 96 per-round collectives; the workers are not in a
    # forward at that moment, so the host would block in the carrier's
    # all-reduce with no peer -- the fnFA12 shape, one phase earlier. The
    # refusal is RANK-UNIFORM on purpose (every rank has the plan installed,
    # so every rank returns False here): excluding only the worker would
    # leave exactly the one-sided forward that hangs. On this rig the
    # predicate below is False anyway (the MoE backend is the offload/pool
    # route, not flashinfer_*), so this is a guard against a config change,
    # not a fix for a boot that failed -- named here rather than left to be
    # rediscovered as a hang.
    from sglang.srt.rank_role import installed_role_plan

    if installed_role_plan() is not None:
        logger.info(
            "Form A: flashinfer autotune SKIPPED on every rank. Its dummy "
            "forward is a full model forward, which under Form A issues the "
            "per-layer MoE-input carrier -- a collective the workers do not "
            "join outside a real round."
        )
        return False

    # CuteDSL v1 (cutedsl runner + deepep a2a) bypasses MoeRunner and must not
    # be autotuned -- its _dummy_run would dispatch more tokens per rank than
    # SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK, tripping a DeepEP assert.
    # Read server_args directly to avoid depending on initialize_moe_config()
    # having already populated the MoE backend globals.
    if (
        mr.server_args.moe_runner_backend == "flashinfer_cutedsl"
        and mr.server_args.moe_a2a_backend == "deepep"
    ):
        return False

    backend_str = mr.server_args.moe_runner_backend

    # TODO smor- support other cases for flashinfer autotune, such as, mamba backend

    moe_needs_autotune = backend_str in [
        "flashinfer_trtllm",
        "flashinfer_trtllm_routed",
        "flashinfer_mxfp4",
        "flashinfer_cutedsl",
        "flashinfer_cutlass",
    ]

    from sglang.srt.layers.quantization.fp4_utils import (
        get_fp4_gemm_runner_backend,
    )

    model_quantization = mr.model_config.quantization
    model_uses_fp4 = model_quantization in (
        "modelopt_fp4",
        "modelopt_mixed",
    )
    from sglang.srt.layers.quantization.nvfp4_sm12x_w4a16 import sm12x_w4a16_max_m

    fp4_gemm_needs_autotune = model_uses_fp4 and (
        get_fp4_gemm_runner_backend().is_flashinfer_cutlass()
        or get_fp4_gemm_runner_backend().is_flashinfer_cutedsl()
        # sm_12x small-M W4A16 (flashinfer cute-dsl-native) picks its tactic
        # through flashinfer's AutoTuner; untuned it runs the default tactic.
        or sm12x_w4a16_max_m() > 0
    )

    from sglang.srt.layers.quantization.fp8_utils import (
        get_fp8_gemm_runner_backend,
    )
    from sglang.srt.utils import is_sm100_supported

    model_uses_modelopt_fp8 = model_quantization in (
        "modelopt",
        "modelopt_fp8",
        "modelopt_mixed",
    )
    # Online MXFP8 (microscaling) linears dispatch to flashinfer's
    # ``mm_mxfp8``, which the flashinfer fp8 autotune dummy run does not
    # exercise correctly -- it triggers an illegal memory access inside the
    # mxfp8 cutlass cubin. The mxfp8 gemm is fixed-config and needs no
    # tuning, so skip autotune for these models.
    model_uses_mxfp8 = "mxfp8" in (model_quantization or "")
    fp8_gemm_needs_autotune = not model_uses_mxfp8 and (
        get_fp8_gemm_runner_backend().is_flashinfer_cutlass()
        or (model_uses_modelopt_fp8 and is_sm100_supported())
    )

    if not (moe_needs_autotune or fp4_gemm_needs_autotune or fp8_gemm_needs_autotune):
        return False

    # NVIDIA namespace (#171): FlashInfer autotuning is a CUDA-only concern and
    # "sm90+" is an NVIDIA statement, while the raw capability collides across
    # vendors (gfx942 reports (9, 4)).
    if not cuda_sm_at_least(9):
        return False

    if mr.spec_algorithm.is_speculative():
        return mr.is_draft_model_runner if for_speculative_draft else not mr.is_draft_model_runner

    return True


def flashinfer_autotune_cache_path(model_runner: ModelRunner) -> Path:
    import flashinfer

    mr = model_runner
    major, minor = torch.cuda.get_device_capability(mr.device)
    arch = f"sm{major}{minor}"
    flashinfer_version = getattr(flashinfer, "__version__", "unknown")

    server_args = mr.server_args
    model_key_parts = [
        str(server_args.model_path),
        str(mr.dtype),
        str(server_args.quantization),
        str(server_args.moe_runner_backend),
        str(mr.tp_size),
        str(mr.pp_size),
        str(mr.dp_size),
        str(mr.moe_ep_size),
        str(mr.model_config.hf_config.__class__.__name__),
    ]
    if mr.is_draft_model_runner:
        model_key_parts.append(f"draft_quant={mr.model_config.quantization}")
    model_key = "|".join(model_key_parts)
    cache_key = hashlib.sha256(model_key.encode()).hexdigest()[:16]
    cache_dir = (
        Path(envs.SGLANG_CACHE_DIR.get())
        / "flashinfer"
        / "autotune"
        / flashinfer_version
        / arch
        / cache_key
    )
    cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir / f"rank_tp{mr.tp_rank}_pp{mr.pp_rank}_dp{mr.dp_rank or 0}.json"


def agree_flashinfer_autotune_across_group(model_runner: ModelRunner, local: bool) -> bool:
    """Make the warmup-autotune decision GROUP-UNIFORM (OR over the TP group).

    ``should_run_flashinfer_autotune`` answers per rank, from that rank's own
    card and backend. The autotune dummy forward, however, is a full TP
    forward: ``enter_capture_group_barrier`` and every row-parallel
    all-reduce pair up across the TP group. On a heterogeneous group the
    answers differ -- rc9b D (RC9 e914e89fde, 25.09. 19:53Z): TP0 = 5090
    ``flashinfer_cutlass`` said yes, TP1/TP2 = 3080 ``w4a8_int8`` (sm_86, no
    FlashInfer backend) said no, so TP0 entered the dummy forward ALONE and
    died after 120 s in ``group barrier (tp:0) made no progress``.

    Any rank that tunes -> every rank runs the SAME dummy forward; only the
    ranks whose own answer is yes run it under flashinfer's autotuner (see
    ``run_flashinfer_autotune_forward(tune=...)``). One int all-reduce on the
    TP group's CPU (gloo) group, at the point where every rank already asks
    the question. Rank-local graph runners (``spec_solo_rank_local_graphs``,
    no group barrier by construction) and single-rank groups keep the local
    answer without any collective.
    """
    if getattr(model_runner, "spec_solo_rank_local_graphs", False):
        return bool(local)
    group = getattr(model_runner, "tp_group", None)
    if group is None or int(getattr(group, "world_size", 1) or 1) <= 1:
        return bool(local)
    import torch.distributed as dist

    flag = torch.tensor([1 if local else 0], dtype=torch.int32)
    dist.all_reduce(flag, op=dist.ReduceOp.MAX, group=group.cpu_group)
    agreed = bool(int(flag.item()))
    if agreed and not local:
        logger.info(
            "FlashInfer warmup autotune: another rank of this TP group tunes; this "
            "rank has no autotunable backend and runs the same dummy forward "
            "UNTUNED so the group's collectives stay paired."
        )
    return agreed


@contextlib.contextmanager
def flashinfer_autotune_context(
    model_runner: ModelRunner, *, skip_logits: bool, tune: bool = True
):
    """The autotune dummy forward's context. ``tune=False`` is the same
    forward (same stream, inference mode and logits skipping -- the logits
    path carries TP collectives of its own) WITHOUT flashinfer's autotuner,
    for a rank that only keeps its group's collectives paired."""
    mr = model_runner
    tune_ctx = contextlib.nullcontext()
    if tune:
        from flashinfer.autotuner import autotune

        cache_path = flashinfer_autotune_cache_path(mr)
        if envs.SGLANG_FLASHINFER_AUTOTUNE_CACHE.get():
            autotune_cache = cache_path
            logger.info("Running FlashInfer autotune with cache: %s", autotune_cache)
        else:
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            runs_dir = cache_path.parent / "runs"
            runs_dir.mkdir(parents=True, exist_ok=True)
            autotune_cache = runs_dir / f"{cache_path.stem}.{timestamp}{cache_path.suffix}"
            logger.info(
                "Running FlashInfer autotune (cache reuse DISABLED via "
                "SGLANG_FLASHINFER_AUTOTUNE_CACHE=0); writing fresh result to: %s",
                autotune_cache,
            )
        tune_ctx = autotune(True, cache=str(autotune_cache))
    else:
        logger.info("Running the FlashInfer autotune dummy forward untuned (group peer).")

    # Run warmup on the non-default stream to avoid NCCL 2.29+ cudaMemcpyBatchAsync
    # calls on default stream (unsupported by CUDA) when --enable-symm-mem is used.
    mr.forward_stream.wait_stream(torch.cuda.current_stream())
    with torch.get_device_module(mr.device).stream(mr.forward_stream):
        maybe_skip_logits = contextlib.nullcontext()
        if skip_logits:
            from sglang.srt.layers.logits_processor import autotune_dummy_run_mode

            maybe_skip_logits = autotune_dummy_run_mode()
        with torch.inference_mode(), tune_ctx, maybe_skip_logits:
            yield
    torch.cuda.current_stream().wait_stream(mr.forward_stream)
    logger.info(
        "FlashInfer autotune completed." if tune else "FlashInfer autotune dummy forward (untuned peer) completed."
    )


def run_flashinfer_autotune_forward(
    model_runner: ModelRunner,
    forward_fn: Callable[[], None],
    *,
    skip_logits: bool,
    tune: bool = True,
) -> None:
    """Run the autotune dummy forward (tuned, or untuned as a group peer).

    Inside the JIT cold-build window on every rank: the tuning rank spends
    extra time per GEMM profiling tactics (and may JIT-build them) while its
    peers already wait in the next TP collective.
    """
    from sglang.srt.utils.jit_cold_build import cold_build_window

    with cold_build_window("flashinfer warmup autotune"):
        with flashinfer_autotune_context(model_runner, skip_logits=skip_logits, tune=tune):
            forward_fn()


def maybe_flashinfer_autotune_speculative_draft(
    runner: BaseRunner,
    forward_fn: Callable[[], None],
    *,
    post_warmup_hook: Optional[Callable[[], None]] = None,
    skip_logits: bool = False,
) -> None:
    """Run speculative draft flashinfer autotune."""
    mr = runner.model_runner
    phase_key = f"{runner.__class__.__module__}.{runner.__class__.__qualname__}"
    tuned_phases = getattr(mr, "_flashinfer_spec_draft_autotuned_phases", None)
    if tuned_phases is None:
        tuned_phases = set()
        mr._flashinfer_spec_draft_autotuned_phases = tuned_phases
    if phase_key in tuned_phases:
        return
    if not mr.spec_algorithm.is_speculative() or not mr.is_draft_model_runner:
        return
    tune = should_run_flashinfer_autotune(mr, for_speculative_draft=True)
    if not agree_flashinfer_autotune_across_group(mr, tune):
        return

    def run_and_reset():
        forward_fn()
        if post_warmup_hook is not None:
            post_warmup_hook()

    run_flashinfer_autotune_forward(mr, run_and_reset, skip_logits=skip_logits, tune=tune)
    tuned_phases.add(phase_key)
