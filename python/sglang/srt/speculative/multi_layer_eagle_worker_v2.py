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

import contextlib
import logging
import time
from typing import TYPE_CHECKING, List, Optional, Tuple

import torch

from sglang.kernels.ops.speculative.eagle import fill_bonus_tokens_func
from sglang.srt.environ import envs
from sglang.srt.hardware_backend.npu.graph_runner.multi_layer_eagle_draft_extend_npu_graph_runner import (
    MultiLayerEagleMultiStepDraftExtendNpuGraphRunner,
)
from sglang.srt.hardware_backend.npu.graph_runner.npu_graph_runner import NPUGraphRunner
from sglang.srt.layers.moe.utils import (
    speculative_moe_a2a_backend_context,
    speculative_moe_backend_context,
)
from sglang.srt.layers.utils.logprob import compute_spec_v2_logprobs
from sglang.srt.managers.io_struct import (
    UpdateWeightFromDiskReqInput,
    UpdateWeightsFromIPCReqInput,
)
from sglang.srt.managers.schedule_batch import ScheduleBatch
from sglang.srt.managers.scheduler import GenerationBatchResult
from sglang.srt.managers.tp_worker import TpModelWorker
from sglang.srt.model_executor.cuda_graph_config import (
    Backend,
    Phase,
    check_cuda_graph_backend,
)
from sglang.srt.model_executor.forward_batch_info import (
    CaptureHiddenMode,
    ForwardBatch,
)
from sglang.srt.model_executor.runner import DecodeCudaGraphRunner
from sglang.srt.server_args import ServerArgs
from sglang.srt.speculative.adaptive_runtime_state import (
    AdaptiveController,
    SpecRuntimeState,
)
from sglang.srt.speculative.adaptive_spec_params import (
    adaptive_algorithm_key,
    adaptive_unsupported_reason,
)
from sglang.srt.speculative.base_spec_worker import BaseSpecWorker, EagleDraftWorkerBase
from sglang.srt.speculative.draft_utils import DraftBackendFactory
from sglang.srt.speculative.eagle_info import (
    EagleDraftExtendInput,
    EagleDraftInput,
    EagleVerifyInput,
)
from sglang.srt.speculative.eagle_utils import (
    build_tree_kernel_efficient,
    ensure_spec_kernel_backend,
    default_tree_mask_mode,
    eagle_prepare_for_verify,
    eagle_sample,
    get_draft_recurrent_hidden_state_spec,
)
from sglang.srt.speculative.multi_layer_eagle_draft_extend_cuda_graph_runner import (
    MultiLayerEagleMultiStepDraftExtendCudaGraphRunner,
)
from sglang.srt.speculative.multi_layer_eagle_utils import (
    adapt_draft_state_width,
    rotate_input_ids,
)
from sglang.srt.speculative.spec_info import SpeculativeAlgorithm
from sglang.srt.speculative.spec_utils import (
    draft_tp_context,
    record_stream_each,
    record_stream_for_v2_verify,
    sample_draft_proposal,
    select_top_k_tokens,
)
from sglang.srt.utils import is_cpu, is_npu
from sglang.srt.utils.async_probe import (
    maybe_detect_inf,
    maybe_detect_nan,
    maybe_detect_oob,
)
from sglang.srt.utils.common import (
    empty_context,
    fast_topk,
    get_available_gpu_memory,
    log_info_on_rank0,
)

_is_npu = is_npu()
_is_cpu = is_cpu()


if TYPE_CHECKING:
    from sglang.srt.model_executor.model_runner import ModelRunner, ModelRunnerOutput


logger = logging.getLogger(__name__)


def _get_plan_stream(
    device: str,
) -> Tuple[any, contextlib.AbstractContextManager]:
    if envs.SGLANG_ENABLE_OVERLAP_PLAN_STREAM.get():
        plan_stream = torch.get_device_module(device).Stream()
        plan_stream_ctx = torch.get_device_module(device).stream(plan_stream)
        return plan_stream, plan_stream_ctx
    else:
        return None, contextlib.nullcontext()


class MultiLayerEagleDraftWorker(EagleDraftWorkerBase):
    def __init__(
        self,
        server_args: ServerArgs,
        gpu_id: int,
        tp_rank: int,
        dp_rank: int,
        moe_ep_rank: int,
        attn_cp_rank: int,
        moe_dp_rank: int,
        nccl_port: int,
        target_worker: TpModelWorker,
    ):
        # copy args
        self.server_args = server_args
        self.gpu_id = gpu_id
        self.tp_rank = tp_rank
        self.dp_rank = dp_rank
        self.moe_ep_rank = moe_ep_rank
        self.nccl_port = nccl_port
        self.target_worker = target_worker
        self.draft_extend_attn_backend_list = []
        self.model_config = target_worker.model_config

        # Args for easy access
        self.device = server_args.device
        self.topk = server_args.speculative_eagle_topk

        # GROUP-WIDE spec kernel decision (collective). Runs here because the
        # draft-worker variants deliberately bypass each other's __init__
        # (#136a), so there is no shared constructor to hook -- and it must
        # happen before any draft round, on every rank, the same number of
        # times. If ANY rank lacks sgl_kernel's build_tree/verify ops (sm75,
        # gfx900), every rank switches to the Triton implementations: verify
        # decides accept counts, so a mixture would desync the group silently.
        ensure_spec_kernel_backend(self.topk)
        self.speculative_num_steps = server_args.speculative_num_steps
        self.speculative_num_draft_tokens = server_args.speculative_num_draft_tokens
        self.use_rejection_sampling = server_args.speculative_use_rejection_sampling
        assert self.speculative_num_draft_tokens == self.speculative_num_steps + 1, (
            "multi-layer EAGLE requires speculative_num_draft_tokens == "
            "speculative_num_steps + 1, "
            f"got {self.speculative_num_draft_tokens} and {self.speculative_num_steps}"
        )
        self.speculative_algorithm = SpeculativeAlgorithm.from_string(
            server_args.speculative_algorithm
        )

        # Set constant. ALLOC_LEN_PER_DECODE is a CLASS-level global, so under
        # adaptive draft length (#138) it must be sized for the LARGEST rung the
        # ladder can reach, not for the initial one -- a later upshift would
        # otherwise under-allocate the per-decode KV window.
        alloc_steps = max(
            self.speculative_num_steps,
            server_args.adaptive_max_candidate_steps or 0,
        )
        EagleDraftInput.ALLOC_LEN_PER_DECODE = max(
            alloc_steps * self.topk, alloc_steps + 1
        )

        # Load draft model weights only.
        with empty_context(), speculative_moe_backend_context():
            self.draft_worker = TpModelWorker(
                server_args=server_args,
                gpu_id=gpu_id,
                tp_rank=tp_rank,
                pp_rank=0,  # spec workers don't support pipeline parallelism
                dp_rank=dp_rank,
                moe_ep_rank=moe_ep_rank,
                attn_cp_rank=attn_cp_rank,
                moe_dp_rank=moe_dp_rank,
                nccl_port=nccl_port,
                is_draft_worker=True,
                is_multi_layer_eagle=True,
            )

        # Alias for better readability
        self.draft_runner_list: List[ModelRunner] = self.draft_worker.model_runner_list
        # Match `EagleDraftWorker.draft_runner` for generic draft-runner access.
        self.draft_runner: ModelRunner = self.draft_runner_list[0]

        # Chain-style MTP: each step propagates its own output hidden states to the
        # next step.  Non-chain: each step uses the target model's hidden states.
        draft_arch = self.draft_worker.model_config.hf_config.architectures[0]
        self.chain_mtp_hidden_states = draft_arch in ["Step3p5MTP"]
        self.draft_tp_context = (
            draft_tp_context if server_args.enable_dp_attention else empty_context
        )
        self.tree_mask_mode = default_tree_mask_mode()
        self.plan_stream, self.plan_stream_ctx = _get_plan_stream(self.device)

    def alloc_memory_pool(
        self,
        memory_pool_config=None,
        req_to_token_pool=None,
        token_to_kv_pool_allocator=None,
    ):
        """Allocate draft KV cache pools (called by scheduler)."""
        self.req_to_token_pool = req_to_token_pool
        self.token_to_kv_pool_allocator = token_to_kv_pool_allocator
        self.draft_worker.alloc_memory_pool(
            memory_pool_config=memory_pool_config,
            req_to_token_pool=req_to_token_pool,
            token_to_kv_pool_allocator=token_to_kv_pool_allocator,
        )
        self.init_lm_head()

    def init_attention_backends(self):
        with (
            self.draft_tp_context(self.draft_runner_list[0].tp_group),
            speculative_moe_backend_context(),
        ):
            super().init_attention_backends()

    def init_cuda_graphs(self):
        with (
            self.draft_tp_context(self.draft_runner_list[0].tp_group),
            speculative_moe_backend_context(),
        ):
            super().init_cuda_graphs()

    def mtp_model_runner(self, step: int):
        return self.draft_runner_list[step]

    def init_lm_head(self):
        embed, head = self.target_worker.model_runner.model.get_embed_and_head()
        # Share the embedding and lm_head with EVERY LOADED draft runner, not
        # just the first `speculative_num_steps` of them: under adaptive draft
        # length (#138) the boot k is only the INITIAL rung, while the loaded
        # runner count is the ladder ceiling. Runners above the initial rung
        # would otherwise never get the target's embed/head.
        for runner in self.draft_runner_list:
            runner.model.set_embed_and_head(embed, head)

    def init_attention_backend(self):
        # Create attn backends
        self.draft_extend_attn_backend_list = []
        for step in range(self.speculative_num_steps):
            draft_backend_factory = DraftBackendFactory(
                self.server_args,
                self.draft_runner_list[step],
                self.topk,
                self.speculative_num_steps,
            )
            self.draft_extend_attn_backend_list.append(
                draft_backend_factory.create_draft_extend_backend()
            )
            if self.draft_extend_attn_backend_list[-1] is not None:
                self.draft_runner_list[step].attn_backend = (
                    self.draft_extend_attn_backend_list[-1]
                )

    def _capture_cuda_graphs(self):
        self.cuda_graph_runner = None
        self.cuda_graph_runner_for_draft_extend = None

        if _is_cpu or check_cuda_graph_backend(Phase.DECODE, Backend.DISABLED):
            return

        if envs.SGLANG_DISABLE_DRAFT_EXTEND_CUDA_GRAPH.get():
            return

        if not _is_npu:
            self.cuda_graph_runner_for_draft_extend = (
                MultiLayerEagleMultiStepDraftExtendCudaGraphRunner(self)
            )
        else:
            self.cuda_graph_runner_for_draft_extend = (
                MultiLayerEagleMultiStepDraftExtendNpuGraphRunner(self)
            )

    def draft(self, batch: ScheduleBatch):
        draft_input: EagleDraftInput = batch.spec_info
        # Adaptive draft length (#138): the chain columns in `draft_input` were
        # produced at the END of the previous round, at the k that was active
        # then. A k switch lands between rounds, so bring the carried columns to
        # the active width first (slice on downshift, repeat-last on upshift --
        # see adapt_draft_columns for why padding cannot break correctness).
        # No-op (one integer compare) whenever the width already matches, which
        # is every round except the first one after a switch.
        #
        # UNCONDITIONAL, including on idle batches: draft_forward() runs BEFORE
        # the is_idle() early-return below and indexes the carried columns up to
        # speculative_num_steps, so an idle rank that skipped the adaptation
        # would index out of bounds after an upshift. It is also the
        # rank-uniformity rule ([[rank-lokaler-test-vor-kollektiv]]): the k a
        # rank drafts at must never depend on a rank-local condition such as
        # "is my slice of this batch empty".
        adapt_draft_state_width(draft_input, self.speculative_num_steps, self.topk)
        forward_batch, can_cuda_graph = self.prepare_for_draft(
            draft_input,
            self.req_to_token_pool,
            batch,
            self.cuda_graph_runner,
            self.draft_runner_list[0],
            self.topk,
            self.speculative_num_steps,
        )

        # Run draft
        parent_list, top_scores_index, draft_tokens = self.draft_forward(forward_batch)

        if batch.forward_mode.is_idle():
            return EagleVerifyInput.create_idle_input(
                self.topk,
                self.speculative_num_steps,
                self.speculative_num_draft_tokens,
                self.device,
            )

        # Build tree mask
        # Directly write to cuda graph buffers for verify attn
        tree_mask_buf, position_buf = (
            self.target_worker.model_runner.attn_backend.get_verify_buffers_to_fill_after_draft()
        )
        (
            tree_mask,
            position,
            retrieve_index,
            retrieve_next_token,
            retrieve_next_sibling,
            draft_tokens,
        ) = build_tree_kernel_efficient(
            draft_input.bonus_tokens,
            parent_list,
            top_scores_index,
            draft_tokens,
            batch.seq_lens,
            batch.seq_lens_sum,
            self.topk,
            self.speculative_num_steps,
            self.speculative_num_draft_tokens,
            self.tree_mask_mode,
            tree_mask_buf,
            position_buf,
        )

        return EagleVerifyInput(
            draft_token=draft_tokens,
            custom_mask=tree_mask,
            positions=position,
            retrieve_index=retrieve_index,
            retrieve_next_token=retrieve_next_token,
            retrieve_next_sibling=retrieve_next_sibling,
            retrieve_cum_len=None,
            spec_steps=self.speculative_num_steps,
            topk=self.topk,
            draft_token_num=self.speculative_num_draft_tokens,
            capture_hidden_mode=None,
            seq_lens_sum=None,
            seq_lens_cpu=None,
            draft_probs=draft_input.draft_probs,
        )

    def draft_forward(self, forward_batch: ForwardBatch):
        # Parse args
        spec_info: EagleDraftInput = forward_batch.spec_info
        topk_p, topk_index, hidden_states = (
            spec_info.topk_p,
            spec_info.topk_index,
            spec_info.hidden_states,
        )

        maybe_detect_nan(topk_p, "draft_forward: NaN in initial topk_p from spec_info")

        # Return values
        score_list: List[torch.Tensor] = []
        token_list: List[torch.Tensor] = []
        parents_list: List[torch.Tensor] = []

        # Forward multiple steps
        scores = None
        _, hidden_states, scores, tree_info = select_top_k_tokens(
            0, topk_p, topk_index, hidden_states, scores, self.topk
        )
        if self.speculative_num_steps == 1:
            score_list.append(tree_info[0])
            token_list.append(tree_info[1])
            parents_list.append(tree_info[2])
        else:
            for i in range(self.speculative_num_steps):
                score_list.append(tree_info[0][:, :, i].unsqueeze(-1))
                token_index = tree_info[1][:, i].unsqueeze(-1)
                token_list.append(token_index)
                if i == 0:
                    parents_list.append(tree_info[2])
                else:
                    parents_list.append(
                        torch.full(
                            (tree_info[2].size(0), 1),
                            i,
                            dtype=torch.long,
                            device=tree_info[2].device,
                        )
                    )

        # Organize the results
        score_list = torch.cat(score_list, dim=1).flatten(
            1
        )  # b, n, topk; n= 1 + (num_steps-1) * self.topk
        ss_token_list = torch.cat(
            token_list, dim=1
        )  # b, (self.topk + (num_steps-1) * self.topk)
        top_scores = torch.topk(
            score_list, self.speculative_num_draft_tokens - 1, dim=-1
        )
        top_scores_index = top_scores.indices
        top_scores_index = torch.sort(top_scores_index).values
        maybe_detect_oob(
            top_scores_index,
            0,
            ss_token_list.shape[1],
            "draft_forward: top_scores_index OOB for gather on ss_token_list",
        )
        draft_tokens = torch.gather(ss_token_list, index=top_scores_index, dim=1)

        if len(parents_list) > 1:
            parent_list = torch.cat(parents_list[:-1], dim=1)
        else:
            batch_size = parents_list[0].shape[0]
            parent_list = torch.empty(batch_size, 0, device=parents_list[0].device)

        return parent_list, top_scores_index, draft_tokens

    def draft_extend(self):
        pass

    def _draft_extend_for_prefill(
        self,
        batch: ScheduleBatch,
        target_hidden_states: torch.Tensor,
        next_token_ids: torch.Tensor,
    ):
        """
        Run draft model extend to correctly fill the KV cache.

        Args:
            batch: The batch to run.
            target_hidden_states: Hidden states from the target model forward
            next_token_ids: Next token ids generated from the target forward.
        """
        # The draft embed clamps unconditionally (to tolerate multimodal pad
        # sentinels), so probe next_token_ids here first -- otherwise a corrupted id
        # would be clamped away instead of surfacing.
        maybe_detect_oob(
            next_token_ids,
            0,
            self.model_config.vocab_size,
            "draft_extend_for_prefill: next_token_ids before draft embed",
        )

        # Draft-extend spec_info for the extend forward; carries only
        # hidden_states + shape info.
        extend_input = EagleDraftExtendInput(
            hidden_states=target_hidden_states,
            # draft mode is same with decode mode, only 1 token per req
            num_tokens_per_req=1,
            num_tokens_for_logprob_per_req=1,
        )
        batch.spec_info = extend_input

        # Chain-style MTP needs FULL to get all-token hidden states;
        # non-chain only needs LAST (the target model's hidden states).
        # STANDALONE skips hidden states end-to-end.
        if self.speculative_algorithm.is_standalone():
            draft_capture_hidden_mode = CaptureHiddenMode.NULL
        elif self.chain_mtp_hidden_states:
            draft_capture_hidden_mode = CaptureHiddenMode.FULL
        else:
            draft_capture_hidden_mode = CaptureHiddenMode.LAST

        # Run forward
        batch.capture_hidden_mode = draft_capture_hidden_mode
        batch.return_hidden_states_before_norm = True
        forward_batch = ForwardBatch.init_new(batch, self.draft_runner_list[0])

        # Construct input_ids
        # TODO: same chunked-prefill chain divergence as PR #26329.
        if not batch.forward_mode.is_idle():
            rotate_input_ids(
                forward_batch.input_ids,
                forward_batch.extend_start_loc,
                forward_batch.extend_seq_lens,
                next_token_ids,
            )

        topk_p_list = []
        topk_index_list = []
        draft_probs_list = []
        for step in range(self.speculative_num_steps):
            output: ModelRunnerOutput = self.draft_runner_list[step].forward(
                forward_batch
            )
            maybe_detect_nan(
                output.logits_output.next_token_logits,
                f"draft_extend_for_prefill step {step}",
            )
            maybe_detect_inf(
                output.logits_output.next_token_logits,
                f"draft_extend_for_prefill step {step}",
            )
            if self.use_rejection_sampling and self.topk == 1:
                # Sample X ~ q and stash q for the first verify's Leviathan step.
                probs, topk_p, topk_index = sample_draft_proposal(
                    output.logits_output.next_token_logits,
                    forward_batch.sampling_info.temperatures,
                )
                draft_probs_list.append(probs)
            else:
                probs = torch.softmax(output.logits_output.next_token_logits, dim=-1)
                topk_p, topk_index = fast_topk(probs, self.topk, dim=-1)
            topk_p_list.append(topk_p)
            topk_index_list.append(topk_index)
            # Chain-style: use this step's output hidden_states as next step's input
            if (
                self.chain_mtp_hidden_states
                and step < self.speculative_num_steps - 1
                and output.logits_output.hidden_states is not None
            ):
                forward_batch.spec_info.hidden_states = (
                    output.logits_output.hidden_states
                )
            if forward_batch.extend_seq_lens is not None:
                rotate_input_ids(
                    forward_batch.input_ids,
                    forward_batch.extend_start_loc,
                    forward_batch.extend_seq_lens,
                    topk_index,
                )

        next_draft_input = EagleDraftInput(
            topk_p=torch.cat(topk_p_list, dim=1),
            topk_index=torch.cat(topk_index_list, dim=1),
            # Chain-style left the last step's hidden_states on the extend
            # input; non-chain keeps the target hidden states.
            hidden_states=extend_input.hidden_states,
            bonus_tokens=next_token_ids,
            num_tokens_per_req=1,
            num_tokens_for_logprob_per_req=1,
        )
        # q [bs, num_steps, vocab] for the first verify's Leviathan step (RS only).
        next_draft_input.draft_probs = (
            torch.stack(draft_probs_list, dim=1)
            if self.use_rejection_sampling and draft_probs_list
            else None
        )

        return next_draft_input

    def _draft_extend_for_decode(
        self, batch: ScheduleBatch, batch_result: GenerationBatchResult
    ):
        # Batch 2: Draft extend
        draft_extend_input = EagleDraftExtendInput(
            hidden_states=batch_result.logits_output.hidden_states,
            num_tokens_per_req=self.speculative_num_steps + 1,
            num_tokens_for_logprob_per_req=1,
        )

        # Prepare for draft extend in a separate stream
        # Notice that here we use batch_result.next_token_ids as the input ids
        with self.plan_stream_ctx:
            forward_batch = self.prepare_for_draft_extend(
                draft_extend_input,
                batch,
                batch_result.next_token_ids,
                self.speculative_num_draft_tokens,
                self.draft_runner_list[0],
                self.cuda_graph_runner_for_draft_extend,
            )
            forward_batch.return_hidden_states_before_norm = True

        if self.plan_stream:
            torch.get_device_module(self.device).current_stream().wait_stream(
                self.plan_stream
            )
        # `batch_result.accept_lens` includes the bonus token, so drafts-only
        # is accept_lens - 1. Stash on spec_info for the cuda-graph prepare().
        forward_batch.spec_info.num_correct_drafts = batch_result.accept_lens - 1
        forward_batch.spec_info.num_accept_tokens = batch_result.accept_lens

        # Run draft extend batch in the main compute stream
        can_cuda_graph = (
            self.cuda_graph_runner_for_draft_extend
            and self.cuda_graph_runner_for_draft_extend.can_run_graph(forward_batch)
        )
        ret_topk_p_list = []
        ret_topk_index_list = []
        ret_draft_probs_list = []
        next_token_ids_backup = batch_result.next_token_ids.clone()

        if can_cuda_graph:
            cgr = self.cuda_graph_runner_for_draft_extend
            # Populate the single shared buffer set once; each step replays
            # against it and the chain is advanced in place between steps.
            cgr.prepare(forward_batch)
            for step in range(self.speculative_num_steps):
                _out, ret_topk_p, ret_topk_index = cgr.replay(step)
                if self.use_rejection_sampling and self.topk == 1:
                    # Re-pick X ~ q worker-side so the chain rotation carries it
                    # to step N+1 (per-step graph does not sample in-graph).
                    sel = cgr.buffers.select_index[: cgr.raw_bs]
                    probs, ret_topk_p, ret_topk_index = sample_draft_proposal(
                        _out.next_token_logits[sel],
                        forward_batch.sampling_info.temperatures,
                    )
                    ret_draft_probs_list.append(probs)
                ret_topk_p_list.append(ret_topk_p.clone())
                ret_topk_index_list.append(ret_topk_index.clone())
                # Advance the draft chain by rotating the shared input_ids window
                # in place; step N+1's graph then reads the rotated values.
                if step < self.speculative_num_steps - 1:
                    rotate_input_ids(
                        cgr.buffers.input_ids[: cgr.raw_num_tokens],
                        cgr.buffers.extend_start_loc[: cgr.raw_bs],
                        cgr.buffers.extend_seq_lens[: cgr.raw_bs],
                        ret_topk_index,
                        cgr.buffers.select_index[: cgr.raw_bs],
                    )
        else:
            logger.warning_once(
                "can't use cuda graph for draft extend! may have correctness issue!"
            )
            select_index = (
                torch.arange(len(batch.seq_lens), device=self.device)
                * self.speculative_num_draft_tokens
                + batch_result.accept_lens
                - 1
            )
            # NOTE: this non-graph path runs the per-step forwards without any
            # pre-plan (see warning above). Mark the batch so the forward path
            # keeps skipping metadata init — preserves the pre-existing
            # behavior; the latent issue is tracked by the warning.
            # On NPU with --disable-cuda-graph, leave each draft runner to init
            # its own metadata in forward_extend (post-pad), otherwise
            # per-runner attn_backend.forward_metadata is never initialized for
            # draft_runner_list[1+].
            if not _is_npu:
                forward_batch.mark_forward_metadata_ready()

            for step in range(self.speculative_num_steps):
                draft_logits_output = self.draft_runner_list[step].forward(
                    forward_batch
                )
                logits_sel = draft_logits_output.logits_output.next_token_logits[
                    select_index
                ]
                if self.use_rejection_sampling and self.topk == 1:
                    probs, ret_topk_p, ret_topk_index = sample_draft_proposal(
                        logits_sel, forward_batch.sampling_info.temperatures
                    )
                    ret_draft_probs_list.append(probs)
                else:
                    probs = torch.softmax(logits_sel, dim=-1)
                    ret_topk_p, ret_topk_index = fast_topk(probs, self.topk, dim=-1)
                # Chain-style: use this step's output hidden_states as next step's input
                if (
                    self.chain_mtp_hidden_states
                    and step < self.speculative_num_steps - 1
                    and draft_logits_output.logits_output.hidden_states is not None
                ):
                    forward_batch.spec_info.hidden_states = (
                        draft_logits_output.logits_output.hidden_states
                    )
                if forward_batch.extend_seq_lens is not None:
                    rotate_input_ids(
                        forward_batch.input_ids,
                        forward_batch.extend_start_loc,
                        forward_batch.extend_seq_lens,
                        ret_topk_index,
                        select_index,
                    )
                ret_topk_p_list.append(ret_topk_p)
                ret_topk_index_list.append(ret_topk_index)

        batch_result.next_token_ids = next_token_ids_backup
        # Construct the return values
        next_draft_input = batch_result.next_draft_input
        (
            next_draft_input.topk_p,
            next_draft_input.topk_index,
            next_draft_input.hidden_states,
        ) = (
            torch.cat(ret_topk_p_list, dim=1).clone(),
            torch.cat(ret_topk_index_list, dim=1).clone(),
            None,
        )
        # q [bs, num_steps, vocab] carries the per-chain-step draft distributions
        # to the next verify's Leviathan step (accept iff coin*q < p). None
        # otherwise (default target-only tree sampling).
        next_draft_input.draft_probs = (
            torch.stack(ret_draft_probs_list, dim=1) if ret_draft_probs_list else None
        )


class MultiLayerEagleWorkerV2(BaseSpecWorker):
    def __init__(
        self,
        server_args: ServerArgs,
        gpu_id: int,
        tp_rank: int,
        dp_rank: Optional[int],
        moe_ep_rank: int,
        attn_cp_rank: int,
        moe_dp_rank: int,
        nccl_port: int,
        target_worker: TpModelWorker,
    ):
        # Parse arguments
        self.server_args = server_args
        self.topk = server_args.speculative_eagle_topk
        self.speculative_num_steps = server_args.speculative_num_steps
        self.speculative_num_draft_tokens = server_args.speculative_num_draft_tokens
        self.gpu_id = gpu_id
        self.device = server_args.device
        self._target_worker = target_worker
        self.page_size = server_args.page_size
        self.speculative_algorithm = SpeculativeAlgorithm.from_string(
            server_args.speculative_algorithm
        )

        # Override the context length of the draft model to be the same as the target model.
        server_args.override(
            "spec_worker.match_target_context_length",
            context_length=target_worker.model_runner.model_config.context_len,
        )

        # BEFORE building the draft worker: that constructor loads one MTP
        # layer's weights per ladder rung, so an unsupported adaptive config
        # must be rejected while the failure is still cheap and legible.
        if server_args.speculative_adaptive:
            self._assert_adaptive_supported(server_args)

        self._draft_worker = MultiLayerEagleDraftWorker(
            server_args,
            gpu_id,
            tp_rank,
            dp_rank,
            moe_ep_rank,
            attn_cp_rank,
            moe_dp_rank,
            nccl_port,
            target_worker,
        )

        # Adaptive speculative (#138). Same shape as EAGLEWorkerV2 /
        # FrozenKVMTPWorkerV2: OFF unless --speculative-adaptive is passed.
        self.adaptive_controller: Optional[AdaptiveController] = None
        if server_args.speculative_adaptive:
            self.adaptive_controller = AdaptiveController(
                self,
                config_path=server_args.speculative_adaptive_config,
                algorithm=adaptive_algorithm_key(server_args),
            )

        # Some dummy tensors
        self.num_new_pages_per_topk = torch.empty(
            (), dtype=torch.int64, device=self.device
        )
        self.extend_lens = torch.empty((), dtype=torch.int64, device=self.device)

        self.plan_stream, self.plan_stream_ctx = _get_plan_stream(self.device)

    def alloc_memory_pool(
        self,
        memory_pool_config=None,
        req_to_token_pool=None,
        token_to_kv_pool_allocator=None,
    ):
        self._draft_worker.alloc_memory_pool(
            memory_pool_config, req_to_token_pool, token_to_kv_pool_allocator
        )
        self.req_to_token_pool = req_to_token_pool
        self.token_to_kv_pool_allocator = token_to_kv_pool_allocator

    def init_attention_backends(self):
        self._draft_worker.init_attention_backends()

    def init_cuda_graphs(self):
        self._draft_worker.init_cuda_graphs()
        # Build the adaptive ladder once target and draft backends exist. Every
        # candidate k gets a fully pre-captured graph set here; nothing is ever
        # re-captured at runtime (see MultiLayerEagleWorkerV2 adaptive notes).
        if self.adaptive_controller is not None:
            dw = self._draft_worker
            with (
                dw.draft_tp_context(dw.draft_runner_list[0].tp_group),
                speculative_moe_backend_context(),
                speculative_moe_a2a_backend_context(),
            ):
                self.adaptive_controller.register(
                    SpecRuntimeState(
                        speculative_num_steps=self.speculative_num_steps,
                        speculative_num_draft_tokens=self.speculative_num_draft_tokens,
                        # This worker never builds a draft-DECODE graph or
                        # backend: draft() runs no model (all k draft forwards
                        # happen in the draft-EXTEND stage of the prior round).
                        draft_attn_backend=None,
                        cuda_graph_runner=None,
                        target_attn_backend=self._target_worker.model_runner.attn_backend,
                        target_graph_runner=self._target_worker.model_runner.decode_cuda_graph_runner,
                        draft_extend_attn_backend=None,
                        cuda_graph_runner_for_draft_extend=dw.cuda_graph_runner_for_draft_extend,
                        draft_extend_attn_backend_list=list(
                            dw.draft_extend_attn_backend_list
                        ),
                    )
                )
                self.adaptive_controller.init_states(
                    cuda_graph_bs=(
                        None
                        if check_cuda_graph_backend(Phase.DECODE, Backend.DISABLED)
                        else self.server_args.cuda_graph_bs_decode
                    ),
                )

    @property
    def target_worker(self):
        return self._target_worker

    @property
    def draft_worker(self):
        return self._draft_worker

    @property
    def spec_v2_attn_backends(self) -> tuple:
        return (
            self._target_worker.model_runner.attn_backend,
            *(
                backend or runner.attn_backend
                for backend, runner in zip(
                    self._draft_worker.draft_extend_attn_backend_list,
                    self._draft_worker.draft_runner_list,
                )
            ),
        )

    def clear_cache_pool(self):
        # allocator and kv cache pool are shared with target worker, which are cleared in scheduler
        pass

    # -- Adaptive speculative decoding (#138) --
    #
    # Multi-layer EAGLE differs from EAGLE in one structural way that shapes all
    # of the code below: it runs ONE DRAFT MODEL PER CHAIN POSITION
    # (tp_worker._init_multi_layer_eagle_model_runners loads one ModelRunner per
    # step, each holding a different MTP layer of the draft checkpoint). So k
    # cannot grow past the number of LOADED layers -- it can only select a
    # prefix of them. Boot therefore loads max(candidate_steps) layers and every
    # rung uses draft_runner_list[:k]. Contrast EAGLE, where one draft model is
    # rolled out k times and growth is free.

    def _assert_adaptive_supported(self, server_args: ServerArgs) -> None:
        reason = adaptive_unsupported_reason(server_args)
        if reason is not None:
            raise ValueError(
                "Multi-layer EAGLE cannot enable adaptive speculative "
                f"decoding: {reason}"
            )
        from sglang.srt.speculative.adaptive_spec_params import (
            resolve_candidate_steps_from_config,
        )

        candidate_steps = resolve_candidate_steps_from_config(
            cfg_path=server_args.speculative_adaptive_config,
            algorithm=adaptive_algorithm_key(server_args),
        )
        if any(s < 1 for s in candidate_steps):
            raise ValueError(
                "Multi-layer EAGLE adaptive speculative decoding supports only "
                f"candidate steps >= 1, but the config resolves to {candidate_steps}. "
                "Step 0 (nospec) has no branch in this worker: draft() builds the "
                "chain by slicing k columns out of the carried draft state, and the "
                "draft-extend loop would produce none. Pass "
                '--speculative-adaptive-config with candidate_steps >= 1 (e.g. '
                '{"1": {"candidate_steps": [1, 2, 3]}}).'
            )
        # The remaining multi-layer constraint -- the ladder ceiling is a WEIGHT
        # budget, so rung k needs MTP layers 0..k-1 to EXIST in the draft
        # checkpoint -- is enforced in
        # TpModelWorker._num_multi_layer_eagle_draft_runners, which is where the
        # draft model config is first available and where the layers would
        # otherwise be loaded (a bad ceiling must fail before the weight load,
        # not as a missing-parameter error deep inside it).

    def build_adaptive_runtime_state(
        self,
        speculative_num_steps: int,
        speculative_num_draft_tokens: int,
        cuda_graph_bs=None,
    ) -> SpecRuntimeState:
        """Build a fully captured SpecRuntimeState for one candidate k."""
        tic = time.perf_counter()
        before_mem = get_available_gpu_memory(self.device, self.gpu_id)
        dw = self._draft_worker

        with self._override_worker_state(
            speculative_num_steps,
            speculative_num_draft_tokens,
            cuda_graph_bs=cuda_graph_bs,
        ):
            # Fresh per-step draft-extend backends + a fresh composite graph
            # runner. Both MUST be rebuilt per rung: the composite's shared
            # buffer window is speculative_num_draft_tokens wide and it holds
            # exactly `speculative_num_steps` per-step graphs, so nothing can be
            # shared across rungs (M16/#50 isolation, enforced by
            # assert_runtime_state_isolation over draft_extend_attn_backend_list).
            dw.init_attention_backend()
            dw._capture_cuda_graphs()

            target_model_runner = self._target_worker.model_runner
            backup_init = target_model_runner.init_new_workspace
            try:
                target_attn_backend = target_model_runner._get_attention_backend(
                    init_new_workspace=True
                )
            finally:
                target_model_runner.init_new_workspace = backup_init

            target_graph_runner = None
            if not check_cuda_graph_backend(Phase.DECODE, Backend.DISABLED):
                TargetGraphRunnerCls = (
                    NPUGraphRunner if _is_npu else DecodeCudaGraphRunner
                )
                target_graph_runner = TargetGraphRunnerCls(
                    target_model_runner,
                    attn_backend=target_attn_backend,
                    speculative_num_steps=speculative_num_steps,
                    speculative_num_draft_tokens=speculative_num_draft_tokens,
                )

            state = SpecRuntimeState(
                speculative_num_steps=speculative_num_steps,
                speculative_num_draft_tokens=speculative_num_draft_tokens,
                draft_attn_backend=None,
                cuda_graph_runner=None,
                target_attn_backend=target_attn_backend,
                target_graph_runner=target_graph_runner,
                draft_extend_attn_backend=None,
                cuda_graph_runner_for_draft_extend=dw.cuda_graph_runner_for_draft_extend,
                draft_extend_attn_backend_list=list(dw.draft_extend_attn_backend_list),
            )

        after_mem = get_available_gpu_memory(self.device, self.gpu_id)
        log_info_on_rank0(
            logger,
            f"Built multi-layer EAGLE adaptive runtime state "
            f"steps={speculative_num_steps}: "
            f"elapsed={time.perf_counter() - tic:.2f}s, "
            f"mem={(before_mem - after_mem):.2f}GB",
        )
        return state

    def apply_runtime_state(self, state: SpecRuntimeState) -> None:
        """Point the worker at a pre-built runtime state (pure pointer swap)."""
        if self.speculative_num_steps == state.speculative_num_steps:
            return

        log_info_on_rank0(
            logger,
            "Switch multi-layer EAGLE adaptive runtime state: "
            f"steps {self.speculative_num_steps} -> {state.speculative_num_steps}, "
            f"draft_tokens {self.speculative_num_draft_tokens} -> "
            f"{state.speculative_num_draft_tokens}",
        )

        self.speculative_num_steps = state.speculative_num_steps
        self.speculative_num_draft_tokens = state.speculative_num_draft_tokens

        dw = self._draft_worker
        dw.speculative_num_steps = state.speculative_num_steps
        dw.speculative_num_draft_tokens = state.speculative_num_draft_tokens
        dw.cuda_graph_runner_for_draft_extend = state.cuda_graph_runner_for_draft_extend
        self._bind_draft_extend_backends(state.draft_extend_attn_backend_list)

        self._target_worker.model_runner.attn_backend = state.target_attn_backend
        self._target_worker.model_runner.decode_cuda_graph_runner = (
            state.target_graph_runner
        )

        self.server_args.override(
            "adaptive_spec.restore",
            speculative_num_steps=state.speculative_num_steps,
            speculative_num_draft_tokens=state.speculative_num_draft_tokens,
        )

    def _bind_draft_extend_backends(self, backend_list) -> None:
        """Install a rung's per-step draft-extend backends.

        Mirrors MultiLayerEagleDraftWorker.init_attention_backend: the per-step
        draft forward reads draft_runner_list[step].attn_backend, so the runner
        rebind must travel with the list. Runners above the active rung keep
        whatever they had -- they are not touched by a k-step draft round.
        """
        if backend_list is None:
            # A state built with graphs/backends disabled carries no list;
            # clearing the live one would strand the draft runners.
            return
        dw = self._draft_worker
        dw.draft_extend_attn_backend_list = list(backend_list)
        for step, backend in enumerate(dw.draft_extend_attn_backend_list):
            if backend is not None:
                dw.draft_runner_list[step].attn_backend = backend

    @contextlib.contextmanager
    def _override_worker_state(
        self,
        speculative_num_steps: int,
        speculative_num_draft_tokens: int,
        cuda_graph_bs: "list[int] | None" = None,
    ):
        """Temporarily point server_args + worker at a target k for capture.

        server_args is part of the contract, not a convenience: the per-step
        draft-extend graph runner reads speculative_num_steps /
        speculative_num_draft_tokens / speculative_eagle_topk straight off
        model_runner.server_args, so mutating worker attributes alone would
        capture every rung at the boot geometry.
        """
        sa = self.server_args
        dw = self._draft_worker
        backup = (
            self.speculative_num_steps,
            self.speculative_num_draft_tokens,
            dw.speculative_num_steps,
            dw.speculative_num_draft_tokens,
            list(dw.draft_extend_attn_backend_list),
            [runner.attn_backend for runner in dw.draft_runner_list],
            dw.cuda_graph_runner,
            dw.cuda_graph_runner_for_draft_extend,
            sa.speculative_num_steps,
            sa.speculative_num_draft_tokens,
            sa.cuda_graph_bs_decode,
            sa.disable_cuda_graph,
        )

        self.speculative_num_steps = speculative_num_steps
        self.speculative_num_draft_tokens = speculative_num_draft_tokens
        dw.speculative_num_steps = speculative_num_steps
        dw.speculative_num_draft_tokens = speculative_num_draft_tokens
        sa.override(
            "adaptive_spec.capture_override",
            speculative_num_steps=speculative_num_steps,
            speculative_num_draft_tokens=speculative_num_draft_tokens,
        )
        if cuda_graph_bs is not None:
            # A rung no BS range can reach gets an empty pruned list; capture
            # nothing for it rather than capturing the full bs sweep.
            sa.override(
                "adaptive_spec.capture_override",
                cuda_graph_bs_decode=cuda_graph_bs,
                **({"disable_cuda_graph": True} if not cuda_graph_bs else {}),
            )

        try:
            yield
        finally:
            (
                self.speculative_num_steps,
                self.speculative_num_draft_tokens,
                dw.speculative_num_steps,
                dw.speculative_num_draft_tokens,
                dw.draft_extend_attn_backend_list,
                restored_runner_backends,
                dw.cuda_graph_runner,
                dw.cuda_graph_runner_for_draft_extend,
            ) = backup[:8]
            for runner, backend in zip(dw.draft_runner_list, restored_runner_backends):
                runner.attn_backend = backend
            sa.override(
                "adaptive_spec.capture_restore",
                speculative_num_steps=backup[8],
                speculative_num_draft_tokens=backup[9],
                cuda_graph_bs_decode=backup[10],
                disable_cuda_graph=backup[11],
            )

    def on_verify_complete_cpu(
        self,
        num_correct_drafts_per_req: list[int],
        batch_size: int = 0,
        steps: int | None = None,
    ) -> None:
        if self.adaptive_controller is not None:
            self.adaptive_controller.on_verify_complete(
                num_correct_drafts_per_req,
                batch_size=batch_size,
                result_steps=steps,
            )

    def activate_step_by_batch(self, batch_size: int) -> None:
        if self.adaptive_controller is not None:
            self.adaptive_controller.activate_step_by_batch(batch_size)

    def forward_batch_generation(self, batch: ScheduleBatch, on_publish=None):
        if batch.forward_mode.is_extend() or batch.is_extend_in_batch:
            # Target prefill
            target_capture_mode = (
                CaptureHiddenMode.NULL
                if self.speculative_algorithm.is_standalone()
                else CaptureHiddenMode.FULL
            )
            batch.capture_hidden_mode = target_capture_mode
            batch_output = self.target_worker.forward_batch_generation(batch)

            # Spec_v2 convention: batch.seq_lens = length BEFORE this iter's tokens.
            # Extend processed L prompt tokens; next verify iter expects same L.
            batch_output.new_seq_lens = batch.seq_lens
            # Publish before draft_extend so the fence is at target-end.
            if on_publish is not None:
                on_publish(batch_output.new_seq_lens)

            # Chain-style MTP needs FULL to get all-token hidden states;
            # non-chain only needs LAST (the target model's hidden states).
            batch_output.next_draft_input = self.draft_worker._draft_extend_for_prefill(
                batch,
                batch_output.logits_output.hidden_states,
                batch_output.next_token_ids,
            )
            return batch_output
        else:
            # Adaptive: pick this round's k BEFORE drafting, so draft / verify /
            # draft-extend of the round all run on one rung's graph set. The
            # batch size is rank-identical, and the k decision it routes to was
            # itself derived from the rank-0-broadcast accept counts, so every
            # TP rank activates the same rung on the same round.
            self.activate_step_by_batch(batch.seq_lens.shape[0])
            if batch.spec_info is None:
                capture_mode = (
                    CaptureHiddenMode.NULL
                    if self.speculative_algorithm.is_standalone()
                    else CaptureHiddenMode.LAST
                )
                hidden_size, hidden_dtype = get_draft_recurrent_hidden_state_spec(
                    self.draft_worker.draft_runner
                )
                batch.spec_info = EagleDraftInput.create_idle_input(
                    device=self.device,
                    hidden_size=hidden_size,
                    dtype=hidden_dtype,
                    topk=self.topk * self.speculative_num_steps,
                    capture_hidden_mode=capture_mode,
                )
            verify_input: EagleVerifyInput = self.draft_worker.draft(batch)
            assert verify_input.is_verify_input()
            batch.spec_info = verify_input
            batch_output = self.verify(batch)
            # Publish before draft_extend so the fence is at verify-end.
            if on_publish is not None:
                on_publish(batch_output.new_seq_lens)
            self.draft_worker._draft_extend_for_decode(batch, batch_output)
            return batch_output

    def verify(
        self,
        batch: ScheduleBatch,
    ):
        fwd_stream = torch.get_device_module(self.device).current_stream()
        verify_input: EagleVerifyInput = batch.spec_info
        record_stream_for_v2_verify(batch, verify_input, fwd_stream)

        bs = len(batch.seq_lens)

        # Batch 1: Target verify
        # Prepare for target verify in a separate stream
        with self.plan_stream_ctx:
            verify_forward_batch, can_run_cuda_graph = eagle_prepare_for_verify(
                verify_input,
                self.req_to_token_pool,
                batch,
                self.target_worker,
            )

        # Cover post-prepare rebinds: draft_token, plan_stream-allocated out_cache_loc.
        record_stream_each((batch.input_ids, batch.out_cache_loc), fwd_stream)

        # Correct some buffers due to the overlap plan
        if self.plan_stream:
            torch.get_device_module(self.device).current_stream().wait_stream(
                self.plan_stream
            )

            # Some values such as custom_mask and position depend on the output of draft,
            # so the previous plan step used the wrong values. Here, we need to run the related
            # computation again to update them to the correct values.
            self.target_worker.model_runner.attn_backend.update_verify_buffers_to_fill_after_draft(
                verify_input,
                (
                    self.target_worker.model_runner.decode_cuda_graph_runner.bs
                    if can_run_cuda_graph
                    else None
                ),
            )
        # NOTE: metadata init is skipped here unconditionally, although
        # eagle_prepare_for_verify only plans when cuda-graph load_batch ran.
        # eagle_worker_v2 re-inits the non-graph path instead (post-pad); this
        # worker has not adopted that fix, so preserve its behavior verbatim.
        # On NPU with --disable-cuda-graph, non-graph verify needs metadata init
        # in forward_extend (post-pad); only mark ready for the cuda-graph path.
        if not _is_npu or can_run_cuda_graph:
            verify_forward_batch.mark_forward_metadata_ready()
        # Run target verify batch in the main compute stream
        forward_batch_output = self.target_worker.forward_batch_generation(
            batch=None,
            forward_batch=verify_forward_batch,
            is_verify=True,
        )
        logits_output = forward_batch_output.logits_output

        # Sample
        maybe_detect_nan(logits_output.next_token_logits, "verify: target model logits")
        maybe_detect_inf(logits_output.next_token_logits, "verify: target model logits")
        (
            predict,
            accept_lens,
            accept_index,
        ) = eagle_sample(verify_input, batch, logits_output)
        new_seq_lens = batch.seq_lens + accept_lens

        if not batch.forward_mode.is_idle():
            accept_tokens = predict[accept_index]
            bonus_tokens = torch.empty_like(accept_lens, dtype=torch.int32)
            # stride = accept_tokens per-req width = accept_index.shape[1].
            fill_bonus_tokens_func(
                accept_tokens,
                accept_lens,
                bonus_tokens,
                accept_index.shape[1],
                bs,
            )
        else:
            bonus_tokens = torch.empty((0,), device=self.device, dtype=torch.int32)

        if batch.return_logprob and not batch.forward_mode.is_idle():
            compute_spec_v2_logprobs(
                batch, logits_output, predict, accept_index, self.speculative_num_steps
            )

        next_draft_input = EagleDraftInput(bonus_tokens=bonus_tokens)
        # verify_forward_batch transitively holds verify-time GPU tensors that
        # must outlive the imminent batch.input_ids rebind; scheduler pins it
        # in batch_record_buf via extra_keep_alive_refs. See EAGLEWorkerV2.verify.
        return GenerationBatchResult(
            logits_output=logits_output,
            next_token_ids=predict,
            can_run_cuda_graph=can_run_cuda_graph,
            speculative_num_draft_tokens=self.speculative_num_draft_tokens,
            next_draft_input=next_draft_input,
            accept_lens=accept_lens,
            new_seq_lens=new_seq_lens,
            routed_experts_output=forward_batch_output.routed_experts_output,
            indexer_topk_output=forward_batch_output.indexer_topk_output,
            extra_keep_alive_refs=[verify_forward_batch],
        )

    def update_weights_from_disk(self, recv_req: UpdateWeightFromDiskReqInput):
        # Every LOADED draft runner, not just the active rung's prefix: under
        # adaptive draft length the ladder ceiling exceeds the active k, and a
        # skipped runner would keep stale weights until the next upshift.
        for runner in self._draft_worker.draft_runner_list:
            success, message = runner.update_weights_from_disk(
                recv_req.model_path,
                recv_req.load_format,
                recapture_cuda_graph=recv_req.recapture_cuda_graph,
            )
            if not success:
                return success, message
        return True, "Succeeded to update model weights."

    def update_weights_from_ipc(self, recv_req: UpdateWeightsFromIPCReqInput):
        for runner in self._draft_worker.draft_runner_list:
            success, message = runner.update_weights_from_ipc(recv_req)
            if not success:
                return success, message
        return True, "Succeeded to update model weights."
