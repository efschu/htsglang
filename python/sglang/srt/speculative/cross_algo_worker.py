"""Cross-algorithm speculative meta-worker (T156 stage 2).

Hosts BOTH speculative rungs over one shared target TpModelWorker:

* NEXTN/MTP rung -- EAGLEWorkerV2, draft = the target checkpoint's MTP head,
  placement SPLIT (TP-sharded, like a plain NEXTN server);
* DFLASH rung   -- DFlashWorkerV2, draft = a separate DFLASH checkpoint,
  placement SOLO on TP rank 0 (forced: DFlashAttention needs
  heads % tp == 0, unsatisfiable under uneven tp3).

Stage-2 contract (static): ``--speculative-cross-algorithm-force`` pins the
ACTIVE rung; every batch routes to it and ``batch.spec_algorithm`` is stamped
with its algorithm in exactly ONE place (``forward_batch_generation``), so
stage 3 only has to change the SOURCE of that stamp. The other rung is fully
built -- draft weights loaded, draft KV pool allocated, attention backends
initialized, draft graphs AND a dedicated target-verify graph set captured --
but never runs.

Resource isolation of the secondary rung:

* its sub-worker is constructed from a deep-copied ServerArgs re-shaped to
  its algorithm (see cross_algo_utils), and the process-global server-args
  context points at that copy for the duration of every secondary build
  phase;
* its target-verify resources (private-workspace attention backend + decode
  graph runner with its own token width) are built the same way the adaptive
  k-ladder builds non-initial rungs (eagle_worker_v2.build_adaptive_runtime_state);
* when an AdaptiveGraphMemoryManager exists (NEXTN forced + adaptive +
  offload prerequisites), the whole secondary graph build runs inside a
  ``build_state((algorithm, draft_tokens))`` scope and is paused afterwards,
  so the inactive rung holds no physical VRAM (#93/#102 mechanics, rung key
  generalized from ``steps``);
* the target model's aux-hidden-state capture (needed by DFLASH, poison for
  NEXTN whose MTP draft consumes the FINAL hidden state) is toggled per
  build phase and left in the forced rung's runtime setting.
"""

from __future__ import annotations

import logging
from contextlib import contextmanager, nullcontext
from typing import Optional

from sglang.srt.managers.tp_worker import TpModelWorker
from sglang.srt.model_executor.cuda_graph_config import (
    Backend,
    Phase,
    check_cuda_graph_backend,
)
from sglang.srt.runtime_context import get_context, get_server_args
from sglang.srt.speculative.base_spec_worker import BaseSpecWorker
from sglang.srt.speculative.cross_algo_utils import (
    apply_shape_to_args_copy,
    get_cross_shapes,
)
from sglang.srt.speculative.spec_info import SpeculativeAlgorithm
from sglang.srt.utils.common import log_info_on_rank0

logger = logging.getLogger(__name__)


class CrossAlgoWorker(BaseSpecWorker):
    """BaseSpecWorker facade over the two co-resident sub-workers."""

    def __init__(
        self,
        server_args,
        gpu_id: int,
        tp_rank: int,
        dp_rank,
        moe_ep_rank: int,
        attn_cp_rank: int,
        moe_dp_rank: int,
        nccl_port: int,
        target_worker: TpModelWorker,
    ):
        self.server_args = server_args
        self.gpu_id = gpu_id
        self.tp_rank = tp_rank
        self._target_worker = target_worker
        self.device = target_worker.device

        shapes = get_cross_shapes(server_args)
        self._force: str = shapes["force"]
        self._forced_algo = SpeculativeAlgorithm.from_string(
            server_args.speculative_algorithm
        )
        self._secondary_name = "nextn" if self._force == "dflash" else "dflash"
        self._secondary_shape = shapes[self._secondary_name]
        self._secondary_algo = SpeculativeAlgorithm.from_string(
            self._secondary_shape["speculative_algorithm"]
        )
        # Rung key for the graph-memory manager: (algorithm, draft tokens).
        self._secondary_rung_key = (
            self._secondary_shape["speculative_algorithm"],
            int(self._secondary_shape["speculative_num_draft_tokens"]),
        )
        self._secondary_args = apply_shape_to_args_copy(
            server_args, self._secondary_shape
        )

        sub_kwargs = dict(
            gpu_id=gpu_id,
            tp_rank=tp_rank,
            dp_rank=dp_rank,
            moe_ep_rank=moe_ep_rank,
            attn_cp_rank=attn_cp_rank,
            moe_dp_rank=moe_dp_rank,
            nccl_port=nccl_port,
            target_worker=target_worker,
        )

        log_info_on_rank0(
            logger,
            f"CrossAlgoWorker (stage 2): forced rung={self._force} "
            f"({self._forced_algo}), secondary rung={self._secondary_name} "
            f"({self._secondary_algo}, draft_tokens="
            f"{self._secondary_rung_key[1]}, placement="
            f"{self._secondary_shape['speculative_draft_placement']}).",
        )

        # Build the PRIMARY sub-worker first, exactly as the single-algorithm
        # server would (same server_args object, same global context). For
        # force=nextn this also creates the AdaptiveController and thereby
        # the process-wide graph-memory manager the secondary build reuses.
        self._primary = self._make_sub_worker(
            self._forced_algo, server_args, sub_kwargs
        )

        # Build the SECONDARY sub-worker from the re-shaped args copy, with
        # the global server-args context pointing at that copy (deep code
        # paths -- e.g. qwen3_next_mtp reading get_server_args() during model
        # build -- must see the secondary shape).
        with self._secondary_ctx():
            self._secondary = self._make_sub_worker(
                self._secondary_algo, self._secondary_args, sub_kwargs
            )

        # Secondary target-verify resources; filled in init_cuda_graphs.
        self._secondary_target_attn_backend = None
        self._secondary_target_graph_runner = None

    # ------------------------------------------------------------------
    # Construction helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _make_sub_worker(algo: SpeculativeAlgorithm, args, sub_kwargs):
        if algo.is_dflash():
            from sglang.srt.speculative.dflash_worker_v2 import DFlashWorkerV2

            return DFlashWorkerV2(server_args=args, **sub_kwargs)
        assert algo.is_eagle(), f"unexpected cross-algo rung algorithm: {algo}"
        from sglang.srt.speculative.eagle_worker_v2 import EAGLEWorkerV2

        return EAGLEWorkerV2(server_args=args, **sub_kwargs)

    @contextmanager
    def _secondary_ctx(self):
        """Point the process-global server-args context at the secondary
        rung's re-shaped args copy for the duration of a build phase."""
        saved = get_server_args()
        get_context().set_server_args(self._secondary_args)
        try:
            yield
        finally:
            get_context().set_server_args(saved)

    def _graph_memory_manager(self):
        """The process-wide graph-memory manager; create one when no
        AdaptiveController did (e.g. force=dflash, where the primary has no
        k-ladder) so the secondary rung can still be built pauseable."""
        from sglang.srt.speculative.adaptive_graph_memory import (
            OFFLOAD_MODES,
            AdaptiveGraphMemoryManager,
            get_active_manager,
            resolve_adaptive_graph_memory_mode,
        )

        mgr = get_active_manager()
        if mgr is not None:
            return mgr
        mode = resolve_adaptive_graph_memory_mode(self.server_args)
        if mode not in OFFLOAD_MODES:
            return None
        mgr = AdaptiveGraphMemoryManager(mode=mode)
        log_info_on_rank0(
            logger,
            f"Cross-algo: created graph-memory manager (mode={mgr.mode}) for "
            "the secondary rung (no adaptive controller present).",
        )
        return mgr

    # ------------------------------------------------------------------
    # Target aux-hidden-state capture toggle
    # ------------------------------------------------------------------
    def _set_target_aux_capture(self, enabled: bool) -> None:
        """Enable/disable DFLASH aux-hidden-state capture on the target model.

        DFLASH needs the target to emit the concatenated aux-layer hidden
        states; the NEXTN/MTP draft needs the FINAL hidden state (width
        hidden_size) -- with aux capture on, logits_output.hidden_states
        becomes the aux concat and would poison the MTP draft. Graph captures
        bake the current setting in, so each rung's graph set is built under
        its own setting and the runtime setting follows the forced rung.
        """
        mr = self._target_worker.model_runner
        model = mr.model
        if enabled:
            layer_ids = mr.dflash_family_target_layer_ids
            assert layer_ids is not None, (
                "cross-algo: dflash_family_target_layer_ids not resolved on "
                "the target model runner"
            )
            # Idempotent: the setter recomputes layers_to_capture from the
            # raw ids (assignment, not append).
            model.set_dflash_layers_to_capture(layer_ids)
            return
        inner = getattr(model, "model", None)
        model.capture_aux_hidden_states = False
        if inner is not None:
            for lid in list(getattr(inner, "layers_to_capture", []) or []):
                setattr(inner.layers[lid], "_is_layer_to_capture", False)
            inner.layers_to_capture = []

    # ------------------------------------------------------------------
    # BaseSpecWorker interface -- boot phases
    # ------------------------------------------------------------------
    def alloc_memory_pool(
        self,
        memory_pool_config=None,
        req_to_token_pool=None,
        token_to_kv_pool_allocator=None,
    ):
        self.req_to_token_pool = req_to_token_pool
        self.token_to_kv_pool_allocator = token_to_kv_pool_allocator
        self._primary.alloc_memory_pool(
            memory_pool_config=memory_pool_config,
            req_to_token_pool=req_to_token_pool,
            token_to_kv_pool_allocator=token_to_kv_pool_allocator,
        )
        with self._secondary_ctx():
            self._secondary.alloc_memory_pool(
                memory_pool_config=memory_pool_config,
                req_to_token_pool=req_to_token_pool,
                token_to_kv_pool_allocator=token_to_kv_pool_allocator,
            )

    def init_attention_backends(self):
        self._primary.init_attention_backends()
        with self._secondary_ctx():
            self._secondary.init_attention_backends()
        # The target's init_attention_backends enabled aux capture whenever
        # the dflash planning fields are set (under the cross gate they
        # always are, via init_aux_hidden_state_capture). Leave the RUNTIME
        # setting to the forced rung; the secondary's graph build below
        # toggles it temporarily.
        self._set_target_aux_capture(enabled=self._force == "dflash")

    def init_cuda_graphs(self):
        # Secondary rung FIRST: when the primary is NEXTN+adaptive, its
        # controller's finalize_boot (reserve check with all states paused)
        # then runs AFTER the secondary rung exists and accounts for it.
        self._init_secondary_cuda_graphs()
        self._primary.init_cuda_graphs()

    def _init_secondary_cuda_graphs(self):
        import time

        from sglang.srt.utils.common import get_available_gpu_memory

        tic = time.perf_counter()
        before_mem = get_available_gpu_memory(self.device, self.gpu_id)

        mgr = self._graph_memory_manager()
        build_scope = (
            mgr.build_state(self._secondary_rung_key)
            if mgr is not None
            else nullcontext()
        )
        with self._secondary_ctx(), build_scope:
            self._set_target_aux_capture(enabled=self._secondary_algo.is_dflash())
            try:
                (
                    self._secondary_target_attn_backend,
                    self._secondary_target_graph_runner,
                ) = self._build_secondary_target_rung()
                self._secondary.init_cuda_graphs()
            finally:
                self._set_target_aux_capture(enabled=self._force == "dflash")
        if mgr is not None:
            mgr.pause_after_build(self._secondary_rung_key)

        after_mem = get_available_gpu_memory(self.device, self.gpu_id)
        log_info_on_rank0(
            logger,
            f"Cross-algo secondary rung {self._secondary_rung_key} built: "
            f"elapsed={time.perf_counter() - tic:.2f}s, "
            f"mem={(before_mem - after_mem):.2f}GB"
            + (
                " (paused via graph-memory manager)"
                if mgr is not None and mgr.offload_enabled
                else " (resident)"
            ),
        )

    def _build_secondary_target_rung(self):
        """Build the secondary rung's target-verify attention backend and
        decode(-verify) CUDA graph runner, mirroring
        EAGLEWorkerV2.build_adaptive_runtime_state's target part. The target
        model runner's algorithm identity and the REAL server args' spec
        fields are temporarily shaped to the secondary rung -- the decode
        graph runner reads both during capture."""
        from sglang.srt.model_executor.runner import DecodeCudaGraphRunner

        mr = self._target_worker.model_runner
        sa = self.server_args
        shape = self._secondary_shape
        backup = (
            mr.spec_algorithm,
            sa.speculative_algorithm,
            sa.speculative_num_steps,
            sa.speculative_eagle_topk,
            sa.speculative_num_draft_tokens,
        )
        sa.override(
            "cross_algo.secondary_target_capture",
            speculative_algorithm=shape["speculative_algorithm"],
            speculative_num_steps=shape["speculative_num_steps"],
            speculative_eagle_topk=shape["speculative_eagle_topk"],
            speculative_num_draft_tokens=shape["speculative_num_draft_tokens"],
        )
        mr.spec_algorithm = self._secondary_algo
        try:
            backup_init = mr.init_new_workspace
            try:
                target_attn_backend = mr._get_attention_backend(
                    init_new_workspace=True
                )
            finally:
                mr.init_new_workspace = backup_init

            target_graph_runner = None
            if not check_cuda_graph_backend(Phase.DECODE, Backend.DISABLED):
                target_graph_runner = DecodeCudaGraphRunner(
                    mr,
                    attn_backend=target_attn_backend,
                    speculative_num_steps=shape["speculative_num_steps"],
                    speculative_num_draft_tokens=shape[
                        "speculative_num_draft_tokens"
                    ],
                )
            return target_attn_backend, target_graph_runner
        finally:
            mr.spec_algorithm = backup[0]
            sa.override(
                "cross_algo.secondary_target_capture_restore",
                speculative_algorithm=backup[1],
                speculative_num_steps=backup[2],
                speculative_eagle_topk=backup[3],
                speculative_num_draft_tokens=backup[4],
            )

    # ------------------------------------------------------------------
    # BaseSpecWorker interface -- runtime (routes to the forced rung)
    # ------------------------------------------------------------------
    @property
    def target_worker(self) -> TpModelWorker:
        return self._target_worker

    @property
    def draft_worker(self):
        return self._primary.draft_worker

    @property
    def war_fastpath_runner(self):
        return self._primary.war_fastpath_runner

    @property
    def spec_v2_attn_backends(self) -> tuple:
        return self._primary.spec_v2_attn_backends

    def clear_cache_pool(self):
        self._primary.clear_cache_pool()
        self._secondary.clear_cache_pool()

    def forward_batch_generation(self, batch, *args, **kwargs):
        # THE stamp point (stage-2: constant == the scheduler's stamp; stage 3
        # changes only the SOURCE of this value to the per-batch active rung).
        batch.spec_algorithm = self._forced_algo
        return self._primary.forward_batch_generation(batch, *args, **kwargs)

    def on_verify_complete_cpu(
        self,
        num_correct_drafts_per_req: list,
        batch_size: int = 0,
        steps: Optional[int] = None,
    ) -> None:
        self._primary.on_verify_complete_cpu(
            num_correct_drafts_per_req, batch_size=batch_size, steps=steps
        )

    def note_request_finished(self, *, rid: str, natural_stop: bool) -> None:
        self._primary.note_request_finished(rid=rid, natural_stop=natural_stop)

    def activate_step_by_batch(self, batch_size: int) -> None:
        self._primary.activate_step_by_batch(batch_size)

    def __getattr__(self, name):
        # Delegate anything else (weight updates, dspark hooks, solo flags,
        # ...) to the forced rung's sub-worker. Guard the backing field so a
        # lookup before __init__ finishes raises cleanly.
        if name in ("_primary", "_secondary", "_target_worker"):
            raise AttributeError(name)
        return getattr(self._primary, name)
