"""Cross-algorithm speculative meta-worker (T156 stages 2-4).

Stage 4 (``--speculative-cross-algorithm-force auto``): the active rung is
chosen at runtime by an acceptance-driven bandit (cross_algo_bandit) over
{NEXTN k in the adaptive-config candidate set, DFLASH}. The NEXTN k rungs are
bandit ARMS with their own pre-built runtime states (the stage-1 k-ladder
controller stays deliberately unconstructed here -- see cross_algo_utils);
rank 0 decides, the rung id is broadcast over the TP CPU group at a
rank-uniform round boundary (design 4c), and switches land through the same
stage-3 round-boundary mechanics as schedule mode.

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
import time
from contextlib import contextmanager, nullcontext
from typing import Optional

import torch

from sglang.srt.layers.moe.utils import (
    speculative_moe_a2a_backend_context,
    speculative_moe_backend_context,
)
from sglang.srt.managers.tp_worker import TpModelWorker
from sglang.srt.model_executor.cuda_graph_config import (
    Backend,
    Phase,
    check_cuda_graph_backend,
)
from sglang.srt.model_executor.forward_batch_info import CaptureHiddenMode
from sglang.srt.runtime_context import get_context, get_server_args
from sglang.srt.speculative.base_spec_worker import BaseSpecWorker
from sglang.srt.speculative.cross_algo_utils import (
    SWITCHING_MODES,
    apply_runtime_shape,
    apply_shape_to_args_copy,
    get_cross_shapes,
    set_dual_capture_active,
)
from sglang.srt.speculative.spec_info import SpeculativeAlgorithm
from sglang.srt.speculative.spec_utils import spec_stage_span
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
        # Runtime-switching modes: stage-3 'schedule' (fixed round plan) and
        # stage-4 'auto' (bandit). The NEXTN rung is the primary (global
        # shape) in both; _switching False => the stage-2 static force modes,
        # byte-identical behavior.
        self._switching: bool = self._force in SWITCHING_MODES
        self._schedule_interval: Optional[int] = (
            int(shapes["schedule_interval"]) if self._force == "schedule" else None
        )
        if self._switching:
            # Dual hidden-state capture must be active BEFORE any target graph
            # capture (the scheduler captures target graphs right after worker
            # construction); process-local, scheduler process only.
            set_dual_capture_active(True)
            self._nextn_shape = shapes["nextn"]
            self._dflash_shape = shapes["dflash"]
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
        # Stage-3/4 switching-mode runtime state.
        # ------------------------------------------------------------------
        if self._switching:
            # Under the switching modes the primary is ALWAYS the NEXTN rung
            # and the secondary the DFLASH rung (cross_algo_utils guarantees
            # it).
            assert self._secondary_name == "dflash"
            self._dflash_sub = self._secondary
            self._nextn_sub = self._primary
            # DFLASH reads its context features from the dual-capture aux
            # field; the final hidden states stay for the NEXTN catch-up.
            self._dflash_sub._cross_dual_capture = True
            self._dflash_block_size = int(
                self._dflash_shape["speculative_num_draft_tokens"]
            )
            # Rung bookkeeping. The boot-k NEXTN rung is the graph-memory
            # BASELINE (untagged, always resident); the DFLASH tag -- and in
            # auto mode the non-primary NEXTN k tags -- are paused/resumed
            # across switches. The primary key must have NO built tag, so
            # ensure_active(("EAGLE", primary_k + 1)) pauses the mapped tag
            # and maps nothing.
            self._primary_k = int(self._nextn_shape["speculative_num_steps"])
            self._active_name = "nextn"
            self._active_k = self._primary_k
            self._rounds_total = 0
            self._rounds_at_switch = 0
            self._switch_count = 0
            # NEXTN k -> SpecRuntimeState; filled in init_cuda_graphs (the
            # primary k from the boot pointers, further ks -- auto mode's
            # bandit arms -- via build_adaptive_runtime_state).
            self._nextn_states = {}
            # Warm-keeping cost accounting (reported at each switch).
            self._warmkeep_ms_since_switch = 0.0
            self._warmkeep_rounds_since_switch = 0
            self._trace_path = None
            import os

            trace = os.environ.get("SGLANG_CROSS_ALGO_TRACE")
            if trace:
                self._trace_path = f"{trace}.rank{tp_rank}"

        # ------------------------------------------------------------------
        # Stage-4 auto mode (rung bandit, design 4b/4c) and the
        # deterministic drafter-policy mode share the NEXTN k arm set and
        # the ctx-gate bookkeeping.
        # ------------------------------------------------------------------
        if self._force in ("auto", "policy"):
            self._bandit_ks = [int(k) for k in shapes["bandit_ks"]]
            assert self._primary_k in self._bandit_ks
            self._rungs = [("nextn", k) for k in self._bandit_ks] + [
                ("dflash", self._dflash_block_size)
            ]
            # Context gate for the DFLASH rung (resolved at argument time
            # from the drafter's config.json; see cross_algo_utils). auto:
            # while the batch context is at/above the threshold, the DFLASH
            # rung is ineligible -- never selected, never probed. policy:
            # safety filter over the table lookup.
            gate = shapes.get("ctx_gate") or {}
            self._ctx_gate_threshold = gate.get("threshold")
            self._ctx_gate_near_frac = float(gate.get("near_frac", 0.8))
            self._ctx_gate_last_ctx = 0

        if self._force == "policy":
            # Deterministic ctx->rung table (T156 follow-up): no bandit, no
            # probing, no broadcast -- see _maybe_policy_switch.
            self._policy_table = [
                (int(start), (str(r[0]), int(r[1])))
                for start, r in shapes["policy_table"]
            ]
            self._policy_fallback_logged = False
            log_info_on_rank0(
                logger,
                "Cross-algo drafter policy (force=policy): "
                + ", ".join(
                    f"ctx>={start}: {fam}:{val}"
                    for start, (fam, val) in self._policy_table
                )
                + f"; ctx gate as safety filter: {self._ctx_gate_threshold}.",
            )

        if self._force == "auto":
            import os

            from sglang.srt.distributed import get_tp_group
            from sglang.srt.speculative.cross_algo_bandit import CrossAlgoBandit

            trace = os.environ.get("SGLANG_CROSS_BANDIT_TRACE")
            self._bandit = CrossAlgoBandit(
                rungs=self._rungs,
                initial=("nextn", self._primary_k),
                trace_path=(trace if (trace and tp_rank == 0) else None),
            )
            self._last_decide_round = 0
            # Rank-0-decides + broadcast (design 4c): the objective's round-
            # duration term is rank-local WALL CLOCK, so -- unlike the
            # stage-1 k-ladder and the stage-3 schedule plan -- the decision
            # CANNOT be recomputed identically per rank. One int64 over the
            # TP CPU (gloo) group at a rank-uniform round boundary, in eager
            # scheduler code (never inside a graph-capture region).
            coord = get_tp_group()
            self._decision_cpu_group = coord.cpu_group
            # Pure single-node TP is enforced by normalize_cross_algorithm_
            # args, so the group's first GLOBAL rank is TP rank 0.
            self._decision_src = coord.ranks[0]
            log_info_on_rank0(
                logger,
                f"Cross-algo bandit (stage 4): rungs={self._rungs}, "
                f"config={self._bandit.cfg}, ctx_gate="
                + (
                    f"{self._ctx_gate_threshold} tokens "
                    f"(near_frac={self._ctx_gate_near_frac}, "
                    f"source={gate.get('source')!r})"
                    if self._ctx_gate_threshold is not None
                    else f"off ({gate.get('source')!r})"
                )
                + ".",
            )

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
        # toggles it temporarily. Schedule mode: aux capture stays ON at all
        # times (dual capture separates final and aux fields), so both graph
        # sets are captured -- and every runtime round runs -- in one mode
        # and no runtime toggle exists.
        self._set_target_aux_capture(enabled=self._runtime_aux_enabled())

    def _runtime_aux_enabled(self) -> bool:
        """The target model's steady-state aux-capture setting."""
        return self._switching or self._force == "dflash"

    def init_cuda_graphs(self):
        # Secondary rung FIRST: when the primary is NEXTN+adaptive, its
        # controller's finalize_boot (reserve check with all states paused)
        # then runs AFTER the secondary rung exists and accounts for it.
        self._init_secondary_cuda_graphs()
        self._primary.init_cuda_graphs()
        if self._switching:
            from sglang.srt.speculative.adaptive_runtime_state import (
                SpecRuntimeState,
                assert_runtime_state_isolation,
            )

            # The primary (NEXTN, boot-k) rung's resources are whatever the
            # boot pipeline installed on the shared model runner / draft
            # worker; wrap them in a SpecRuntimeState so rung switches treat
            # every NEXTN k uniformly.
            mr = self._target_worker.model_runner
            dw = self._nextn_sub.draft_worker
            self._nextn_states[self._primary_k] = SpecRuntimeState(
                speculative_num_steps=self._primary_k,
                speculative_num_draft_tokens=self._primary_k + 1,
                draft_attn_backend=dw.draft_attn_backend,
                cuda_graph_runner=dw.cuda_graph_runner,
                target_attn_backend=mr.attn_backend,
                target_graph_runner=mr.decode_cuda_graph_runner,
                draft_extend_attn_backend=dw.draft_extend_attn_backend,
                cuda_graph_runner_for_draft_extend=(
                    dw.cuda_graph_runner_for_draft_extend
                ),
            )
            # Dedicated draft-extend graph at the DFLASH width for the
            # per-round NEXTN warm-keeping catch-up (kept RESIDENT: it runs
            # while the DFLASH rung is mapped). Eager fallback if None.
            (
                self._catchup_extend_backend,
                self._catchup_extend_runner,
            ) = self._nextn_sub.build_catchup_extend_state(
                self._dflash_block_size
            )
            if self._force in ("auto", "policy"):
                self._init_bandit_k_states()
                # M16/#50 isolation across the NEXTN k states (the DFLASH
                # rung's resources are isolated by construction, stage 2).
                # The boot-k state is the registered baseline whose global
                # workspace the others may alias (logged, not fatal).
                assert_runtime_state_isolation(
                    dict(self._nextn_states),
                    baseline_steps={self._primary_k},
                )

    def _init_bandit_k_states(self):
        """Build the non-primary NEXTN k rungs (bandit arms): draft backends
        + draft graphs + a private target-verify backend/graph set per k, via
        the k-ladder's build_adaptive_runtime_state -- inside a graph-memory
        build scope keyed ("EAGLE", k+1) and paused after build, so inactive
        k rungs hold no physical VRAM under offload. Mirrors
        AdaptiveController.init_states; the controller itself is deliberately
        NOT constructed in cross-auto mode (its _activate would pause the
        mapped DFLASH tag mid-segment -- the bandit owns activation)."""
        from sglang.srt.utils.common import get_available_gpu_memory

        mgr = self._graph_memory_manager()
        dw = self._nextn_sub.draft_worker
        extra_ks = [k for k in self._bandit_ks if k != self._primary_k]
        # Largest k first: under offload its capture runs with maximum
        # headroom (same rationale as AdaptiveController.init_states).
        for k in sorted(extra_ks, reverse=True):
            tic = time.perf_counter()
            before_mem = get_available_gpu_memory(self.device, self.gpu_id)
            key = ("EAGLE", k + 1)
            build_scope = (
                mgr.build_state(key) if mgr is not None else nullcontext()
            )
            with (
                dw.draft_tp_context(dw.draft_runner.tp_group),
                speculative_moe_backend_context(),
                speculative_moe_a2a_backend_context(),
                build_scope,
            ):
                state = self._nextn_sub.build_adaptive_runtime_state(
                    speculative_num_steps=k,
                    speculative_num_draft_tokens=k + 1,
                )
            self._nextn_states[k] = state
            if mgr is not None:
                mgr.pause_after_build(key)
            after_mem = get_available_gpu_memory(self.device, self.gpu_id)
            log_info_on_rank0(
                logger,
                f"Cross-algo bandit rung ('nextn', {k}) built: "
                f"elapsed={time.perf_counter() - tic:.2f}s, "
                f"mem={(before_mem - after_mem):.2f}GB"
                + (
                    " (paused via graph-memory manager)"
                    if mgr is not None and mgr.offload_enabled
                    else " (resident)"
                ),
            )

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
            self._set_target_aux_capture(
                enabled=self._secondary_algo.is_dflash()
                or self._runtime_aux_enabled()
            )
            try:
                (
                    self._secondary_target_attn_backend,
                    self._secondary_target_graph_runner,
                ) = self._build_secondary_target_rung()
                self._secondary.init_cuda_graphs()
            finally:
                self._set_target_aux_capture(enabled=self._runtime_aux_enabled())
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
    # BaseSpecWorker interface -- runtime (routes to the ACTIVE rung; under
    # the stage-2 static force modes the active rung is always the primary)
    # ------------------------------------------------------------------
    @property
    def _active_sub(self):
        if not self._switching or self._active_name == "nextn":
            return self._primary
        return self._dflash_sub

    @property
    def _active_algo(self) -> SpeculativeAlgorithm:
        if not self._switching:
            return self._forced_algo
        return (
            SpeculativeAlgorithm.DFLASH
            if self._active_name == "dflash"
            else self._forced_algo
        )

    @property
    def _active_rung(self) -> tuple:
        if self._active_name == "dflash":
            return ("dflash", self._dflash_block_size)
        return ("nextn", self._active_k)

    @property
    def target_worker(self) -> TpModelWorker:
        return self._target_worker

    @property
    def draft_worker(self):
        return self._active_sub.draft_worker

    @property
    def war_fastpath_runner(self):
        return self._active_sub.war_fastpath_runner

    @property
    def spec_v2_attn_backends(self) -> tuple:
        if not self._switching:
            return self._primary.spec_v2_attn_backends
        # Consulted once at boot (needs_cpu_seq_lens is an OR over backends):
        # both rungs run, so declare the union. (The auto-mode k states use
        # the same backend class as the primary; no new union members.)
        return tuple(self._primary.spec_v2_attn_backends) + tuple(
            self._secondary.spec_v2_attn_backends
        )

    def clear_cache_pool(self):
        self._primary.clear_cache_pool()
        self._secondary.clear_cache_pool()

    # ------------------------------------------------------------------
    # Stage 3: per-batch switching (schedule mode)
    # ------------------------------------------------------------------
    def maybe_switch_rung(self, batch) -> None:
        """Scheduler pre-decode hook (switching modes; called BEFORE
        ScheduleBatch.prepare_for_decode so the algorithm-dispatching prep and
        the KV headroom reservation already run under the incoming rung).

        Rank uniformity: under SCHEDULE mode the decision is a pure function
        of the executed verify-round counter, which is identical on every TP
        rank -- all ranks' schedulers run the same batch sequence (the accept
        counts that advance seq_lens are broadcast from rank 0, #50), so no
        decision broadcast is needed. Under AUTO mode the bandit objective
        includes per-rank wall time, so rank 0 decides and broadcasts the
        rung id (design 4c, see _maybe_bandit_switch).
        """
        if not self._switching:
            return
        if self._schedule_interval is not None:
            if (
                self._rounds_total - self._rounds_at_switch
            ) >= self._schedule_interval:
                self._apply_rung(
                    batch,
                    (
                        ("dflash", self._dflash_block_size)
                        if self._active_name == "nextn"
                        else ("nextn", self._primary_k)
                    ),
                )
        elif self._force == "policy":
            self._maybe_policy_switch(batch)
        else:
            self._maybe_bandit_switch(batch)
        # Re-stamp unconditionally: freshly (re)built running batches carry
        # the scheduler's boot-time stamp (the primary algorithm).
        batch.spec_algorithm = self._active_algo

    def _maybe_bandit_switch(self, batch) -> None:
        """Auto mode: every decide_interval executed verify rounds, rank 0
        runs the bandit and ALL ranks receive the target rung index over the
        TP CPU (gloo) group.

        The cadence counter (_rounds_total) advances once per executed decode
        round on every rank identically, so all ranks reach this collective
        at the same round boundary -- rank uniformity of the CALL. The
        broadcast VALUE is rank 0's decision; broadcasting the (unchanged)
        current rung is the explicit no-decision case, so ranks never have to
        infer "no news" from a missing message. Eager scheduler code, far
        from any graph-capture region (the stage-3 deadlock lesson).

        CONTEXT GATE: eligibility is computed LOCALLY on every rank from the
        batch's CPU-side sequence lengths -- rank-uniform inputs (all ranks
        run the same batch sequence; accept counts are broadcast, #50), so
        every rank derives the same verdict without communication. The
        DECISION stays rank-0 + broadcast, unchanged. When the ACTIVE rung
        falls out of eligibility (context crossed the threshold mid-stream),
        the decision point is PULLED FORWARD to this round boundary instead
        of waiting out the decide interval -- the trigger is rank-uniform,
        so the collective call stays rank-uniform."""
        gate_on = self._ctx_gate_threshold is not None
        dflash_eligible = True
        if gate_on:
            dflash_eligible = self._dflash_ctx_eligible(batch)
        due = (
            self._rounds_total - self._last_decide_round
        ) >= self._bandit.cfg.decide_interval_rounds
        gate_evict_now = (
            gate_on and not dflash_eligible and self._active_name == "dflash"
        )
        if not due and not gate_evict_now:
            return
        self._last_decide_round = self._rounds_total
        if self.tp_rank == 0:
            eligible = None
            gate_note = ""
            if gate_on and not dflash_eligible:
                eligible = frozenset(
                    r for r in self._rungs if r[0] != "dflash"
                )
                gate_note = (
                    f" (ctx {self._ctx_gate_last_ctx} vs threshold "
                    f"{self._ctx_gate_threshold})"
                )
            idx = self._rungs.index(
                self._bandit.decide(
                    self._rounds_total, eligible=eligible, gate_note=gate_note
                )
            )
        else:
            idx = 0  # placeholder; overwritten by the broadcast
        t = torch.tensor([idx], dtype=torch.int64)
        torch.distributed.broadcast(
            t, src=self._decision_src, group=self._decision_cpu_group
        )
        new_rung = self._rungs[int(t.item())]
        if new_rung != self._active_rung:
            self._apply_rung(batch, new_rung)

    @staticmethod
    def _batch_ctx_and_remaining(batch) -> list:
        """Per request: (current context length, remaining max_new_tokens
        budget or None). CPU-only (req.seqlen / sampling params; no device
        sync) and rank-uniform by construction (all ranks run the same batch
        sequence; accept counts are broadcast, #50)."""
        ctx_and_remaining = []
        for req in batch.reqs:
            mnt = getattr(req.sampling_params, "max_new_tokens", None)
            remaining = (
                None if mnt is None else max(0, int(mnt) - len(req.output_ids))
            )
            ctx_and_remaining.append((req.seqlen, remaining))
        return ctx_and_remaining

    def _dflash_ctx_eligible(self, batch) -> bool:
        """Context-gate eligibility of the DFLASH rung for this batch.

        bs>1: the max over the requests is the context the verify attention
        pays for; any single request violating the gate makes the rung
        ineligible. The turn-start pre-emption criterion lives in
        cross_algo_utils.ctx_gate_eligible."""
        from sglang.srt.speculative.cross_algo_utils import ctx_gate_eligible

        eligible, max_ctx = ctx_gate_eligible(
            self._batch_ctx_and_remaining(batch),
            self._ctx_gate_threshold,
            self._ctx_gate_near_frac,
        )
        self._ctx_gate_last_ctx = max_ctx
        return eligible

    def _maybe_policy_switch(self, batch) -> None:
        """force=policy (T156 follow-up): deterministic ctx -> rung lookup
        in the drafter-policy table at every round boundary. No bandit, no
        probing, no dwell: the active rung is a pure function of the batch's
        CPU-side max context over a static table, both rank-uniform -- so,
        like schedule mode and unlike the bandit, every rank computes the
        same verdict locally and NO decision broadcast is needed. Switches
        only happen when the context crosses a stage boundary (or the batch
        mix changes the max), so the per-round cost is a few comparisons.

        The ctx GATE stays active as a SAFETY filter on top of the table
        (policy_select): a policy-chosen but gate-ineligible DFLASH stage
        falls back to the next stage above, logged once per contiguous
        fallback phase. Gate threshold (4 * window: eligibility upper bound
        for the self-correcting bandit) and the default policy switch point
        (2 * window = drafter training ctx: forced assignment without a
        corrective) are deliberately different -- see
        cross_algo_utils.derive_policy_switch_ctx."""
        from sglang.srt.speculative.cross_algo_utils import policy_select

        ctx_and_remaining = self._batch_ctx_and_remaining(batch)
        target, max_ctx, fell_back = policy_select(
            self._policy_table,
            ctx_and_remaining,
            self._ctx_gate_threshold,
            self._ctx_gate_near_frac,
            ("nextn", self._primary_k),
        )
        self._ctx_gate_last_ctx = max_ctx
        if fell_back:
            if not self._policy_fallback_logged:
                self._policy_fallback_logged = True
                logger.info(
                    "Cross-algo policy: DFLASH stage gate-ineligible "
                    "(ctx %d vs gate %s); falling back to %s:%d.",
                    max_ctx,
                    self._ctx_gate_threshold,
                    target[0],
                    target[1],
                )
        else:
            self._policy_fallback_logged = False
        if target != self._active_rung:
            self._apply_rung(batch, target)

    def _apply_rung(self, batch, new_rung: tuple) -> None:
        old_rung = self._active_rung
        if new_rung == old_rung:
            return
        new_name, new_val = new_rung
        family_change = new_name != self._active_name
        mr = self._target_worker.model_runner
        tic = time.perf_counter()

        # 1) Graph memory: map the incoming rung's paused buffers and pause
        #    the outgoing rung's, BEFORE any pointer swap (synchronizes the
        #    device, so the outgoing rung's in-flight round is quiesced).
        #    The boot-k NEXTN rung is the untagged baseline: activating it
        #    maps nothing and just pauses the outgoing tag.
        ensure_ms = 0.0
        # Only the manager the BOOT created (states built/paused in it); never
        # create one at runtime -- a fresh manager would know no rung tags.
        from sglang.srt.speculative.adaptive_graph_memory import get_active_manager

        mgr = get_active_manager()
        if mgr is not None:
            mgr.ensure_active(
                self._secondary_rung_key
                if new_name == "dflash"
                else ("EAGLE", new_val + 1)
            )
            ensure_ms = mgr.last_swap_ms or 0.0
        else:
            # Resident mode: still quiesce the outgoing rung's round before
            # re-pointing the target's verify resources.
            torch.cuda.synchronize()

        # 2) Target-verify resource pointer swap (the adaptive-k pattern).
        if new_name == "dflash":
            # Park the NEXTN draft side on the PRIMARY (untagged, always-
            # resident) k state FIRST: the dual-prefill draft-extend and the
            # per-round warm-keeping catch-up run during DFLASH segments too,
            # and a non-primary k-state's extend backend plans into flash-
            # infer int-workspaces that belong to that k's now-PAUSED tag
            # (measured 2026-07-22: BatchPrefillWithKVCachePlan segfault on
            # the first prefill after a nextn:2 -> dflash switch). The
            # primary state's resources are permanently mapped -- the exact
            # stage-3 invariant.
            if self._active_k != self._primary_k:
                self._nextn_sub.apply_runtime_state(
                    self._nextn_states[self._primary_k]
                )
                self._active_k = self._primary_k
            mr.attn_backend = self._secondary_target_attn_backend
            mr.decode_cuda_graph_runner = self._secondary_target_graph_runner
            mr.spec_algorithm = self._secondary_algo
            new_shape = self._dflash_shape
        else:
            state = self._nextn_states[new_val]
            # Draft-side + numeric swap through the k-ladder path; no-op when
            # the worker already runs this k (i.e. on every family switch
            # back at an unchanged k).
            self._nextn_sub.apply_runtime_state(state)
            # Target-side pointers explicitly: apply_runtime_state early-
            # returns on an unchanged k, and after a DFLASH segment the
            # target pointers are the DFLASH rung's regardless of k.
            mr.attn_backend = state.target_attn_backend
            mr.decode_cuda_graph_runner = state.target_graph_runner
            mr.spec_algorithm = self._forced_algo
            new_shape = self._nextn_shape_for_k(new_val)

        # 3) Atomically re-point the REAL server args' numeric spec-shape
        #    fields (runtime readers: dflash decode prep block size, eager
        #    verify sizing, metrics). The algorithm STRING and every
        #    max_...-based boot sizing stay fixed.
        apply_runtime_shape(
            self.server_args,
            new_shape,
            f"cross_algo.switch_to_{new_name}_{new_val}",
        )

        # 4) Convert the carried draft input to the incoming rung's type.
        #    Family switches only (within the NEXTN family the
        #    EagleDraftInput is k-independent: one seed row per request).
        #    Only at round boundaries: the batch is BETWEEN verify rounds
        #    here (the outgoing round's commit is complete; prep for the
        #    incoming round has not run).
        if family_change:
            self._convert_spec_info(batch, new_name)

        self._active_name = new_name
        if new_name == "nextn":
            self._active_k = new_val
        self._rounds_at_switch = self._rounds_total
        self._switch_count += 1
        total_ms = (time.perf_counter() - tic) * 1e3
        wk_ms = self._warmkeep_ms_since_switch
        wk_rounds = self._warmkeep_rounds_since_switch
        self._warmkeep_ms_since_switch = 0.0
        self._warmkeep_rounds_since_switch = 0
        logger.info(
            "Cross-algo switch #%d @round %d: %s:%d -> %s:%d "
            "(ensure_active %.2f ms, total %.2f ms; warm-keep in closing "
            "segment: %.2f ms over %d rounds).",
            self._switch_count,
            self._rounds_total,
            old_rung[0],
            old_rung[1],
            new_name,
            new_val,
            ensure_ms,
            total_ms,
            wk_ms,
            wk_rounds,
        )

    def _nextn_shape_for_k(self, k: int) -> dict:
        if k == self._primary_k:
            return self._nextn_shape
        shape = dict(self._nextn_shape)
        shape["speculative_num_steps"] = k
        shape["speculative_num_draft_tokens"] = k + 1
        return shape

    def _convert_spec_info(self, batch, new_name: str) -> None:
        from sglang.srt.speculative.dflash_info_v2 import DFlashDraftInputV2
        from sglang.srt.speculative.eagle_info import EagleDraftInput

        si = batch.spec_info
        if si is None:
            return
        if new_name == "dflash":
            assert isinstance(si, EagleDraftInput), (
                "cross-algo switch to dflash expected an EagleDraftInput at "
                f"the round boundary, got {type(si).__name__}"
            )
            new = DFlashDraftInputV2(
                topk_p=si.topk_p,
                topk_index=si.topk_index,
                bonus_tokens=si.bonus_tokens,
                new_seq_lens=batch.seq_lens,
                hidden_states=si.hidden_states,
            )
        else:
            assert isinstance(si, DFlashDraftInputV2), (
                "cross-algo switch to nextn expected a DFlashDraftInputV2 at "
                f"the round boundary, got {type(si).__name__}"
            )
            # The per-round NEXTN warm-keeping catch-up attached fresh draft
            # seeds (topk/hidden) to every DFLASH round's draft input (and
            # stashed them into the FutureMap under overlap), so the first
            # NEXTN round drafts at full k immediately.
            new = EagleDraftInput(
                topk_p=si.topk_p,
                topk_index=si.topk_index,
                hidden_states=si.hidden_states,
                bonus_tokens=si.bonus_tokens,
                num_tokens_per_req=1,
                num_tokens_for_logprob_per_req=1,
            )
        new.future_indices = si.future_indices
        if new.future_indices is not None:
            assert new.future_indices.shape[0] == batch.batch_size(), (
                "cross-algo switch: future_indices/batch size mismatch "
                f"({new.future_indices.shape[0]} vs {batch.batch_size()})"
            )
        batch.spec_info = new

    # ------------------------------------------------------------------
    # Stage 3: per-round warm-keeping of the INACTIVE rung (schedule mode).
    #
    # Neither draft is stateless in this topology: the NEXTN/MTP draft owns a
    # 1-layer draft KV pool fed by its draft-extend, the solo DFLASH draft
    # (no compact cache) owns a pool fed with the target's aux hidden states.
    # Without warm-keeping, every switch would leave a permanent gap in the
    # inactive draft's KV for the tokens committed during the segment,
    # degrading its accept rate forever after switching back. Dual capture
    # provides both hidden variants on every verify round; the idle rung's
    # KV write is then cheap (DFLASH: pure append; NEXTN: an eager 1-layer
    # MTP extend, also yielding the next-round draft seeds).
    # ------------------------------------------------------------------
    def _warmkeep_nextn_after_dflash_round(self, batch, batch_output) -> None:
        logits_output = batch_output.logits_output
        hidden_final = logits_output.hidden_states
        assert hidden_final is not None, (
            "cross-algo dual capture: DFLASH verify returned no final hidden "
            "states for the NEXTN warm-keeping catch-up"
        )
        nx = self._nextn_sub
        dw = nx.draft_worker
        with (
            dw.draft_tp_context(dw.draft_runner.tp_group),
            speculative_moe_backend_context(),
            speculative_moe_a2a_backend_context(),
            spec_stage_span("cross_warmkeep_nextn"),
        ):
            topk_p, topk_index, hidden = dw.draft_extend_catchup(
                batch,
                hidden_final,
                batch_output.next_token_ids,
                batch_output.accept_lens,
                self._dflash_block_size,
                cuda_graph_runner=self._catchup_extend_runner,
            )
        # Attach the seeds to the DFLASH draft input: the overlap scheduler
        # relays them through the FutureMap (the DFlash input already carries
        # these eagle-shaped fields), the sync scheduler carries the object.
        ndi = batch_output.next_draft_input
        ndi.topk_p = topk_p
        ndi.topk_index = topk_index
        ndi.hidden_states = hidden
        logits_output.hidden_states = None

    def _warmkeep_dflash_after_nextn_round(self, batch, batch_output) -> None:
        logits_output = batch_output.logits_output
        aux = logits_output.cross_aux_hidden_states
        assert aux is not None, (
            "cross-algo dual capture: NEXTN verify returned no aux hidden "
            "states for the DFLASH warm-keeping append"
        )
        dsub = self._dflash_sub
        if not dsub._solo_is_shadow:
            bs = batch.batch_size()
            # Verify width of the round that JUST ran = the active NEXTN k+1
            # (switches land only at round boundaries before prep, so the
            # active rung cannot have changed since this batch was prepared).
            width = self._active_k + 1
            cache_loc = batch.out_cache_loc
            assert (
                cache_loc is not None and cache_loc.numel() == bs * width
            ), (
                "cross-algo warm-keep: unexpected verify cache-loc layout "
                f"({None if cache_loc is None else cache_loc.numel()} vs "
                f"{bs}*{width})"
            )
            # Chain verify (topk == 1): verify row r of request i sits at
            # position seq_lens[i] + r (spec_v2: batch.seq_lens is the
            # pre-round length).
            positions = (
                batch.seq_lens.to(torch.int64).unsqueeze(1)
                + self._nextn_pos_offsets(width)
            ).reshape(-1)
            dsub._append_target_hidden_to_draft_kv_by_loc(
                target_hidden=aux,
                cache_loc=cache_loc,
                cache_loc_2d=cache_loc.view(bs, width),
                positions=positions,
                commit_lens=batch_output.accept_lens,
            )
        logits_output.cross_aux_hidden_states = None

    def _nextn_pos_offsets(self, width: int) -> torch.Tensor:
        buf = getattr(self, "_nextn_pos_offsets_buf", None)
        if buf is None or buf.numel() != width:
            buf = torch.arange(width, dtype=torch.int64, device=self.device)
            self._nextn_pos_offsets_buf = buf
        return buf

    def _trace_round(self, batch, batch_output) -> None:
        if self._trace_path is None:
            return
        import json

        accepts = batch_output.accept_lens
        rec = {
            "round": self._rounds_total,
            "algo": self._active_name,
            "k": self._active_rung[1],
            "bs": batch.batch_size(),
            "rounds_since_switch": self._rounds_total - self._rounds_at_switch,
            "switch_count": self._switch_count,
            "accept_lens": (
                accepts.tolist() if accepts is not None else None
            ),
        }
        with open(self._trace_path, "a") as f:
            f.write(json.dumps(rec) + "\n")

    def _dual_prefill(self, batch, on_publish=None):
        """One shared target prefill forward feeding BOTH rungs' draft-prefill
        paths, so every request is warm in both drafts from birth and can be
        served by either rung after any later switch."""
        batch.capture_hidden_mode = CaptureHiddenMode.FULL
        batch_output = self._target_worker.forward_batch_generation(batch)
        batch_output.new_seq_lens = batch.seq_lens
        if on_publish is not None:
            on_publish(batch_output.new_seq_lens)

        # DFLASH first: it reads batch.extend_lens/prefix_lens/out_cache_loc,
        # which the NEXTN draft-extend below partially rewrites.
        dflash_input = self._dflash_sub.prefill_after_target(batch, batch_output)

        nx = self._nextn_sub
        dw = nx.draft_worker
        with (
            dw.draft_tp_context(dw.draft_runner.tp_group),
            speculative_moe_backend_context(),
            speculative_moe_a2a_backend_context(),
            spec_stage_span("draft_extend"),
        ):
            eagle_input = dw._draft_extend_for_prefill(
                batch,
                batch_output.logits_output.hidden_states,
                batch_output.next_token_ids,
                batch_output.logits_output.mm_input_embeds,
            )

        if self._active_name == "dflash":
            # Serve the DFLASH rung, but carry the NEXTN seeds along (relayed
            # through the FutureMap / the object itself) for a later switch.
            dflash_input.topk_p = eagle_input.topk_p
            dflash_input.topk_index = eagle_input.topk_index
            dflash_input.hidden_states = eagle_input.hidden_states
            batch_output.next_draft_input = dflash_input
        else:
            batch_output.next_draft_input = eagle_input
        return batch_output

    def forward_batch_generation(self, batch, *args, **kwargs):
        # THE stamp point: under the stage-2 static force modes the value is
        # the constant forced algorithm; under schedule mode it is the active
        # rung (already stamped by the scheduler's maybe_switch_rung hook for
        # decode batches -- re-stamped here so the invariant holds for every
        # batch that reaches a worker).
        batch.spec_algorithm = self._active_algo
        if not self._switching:
            return self._primary.forward_batch_generation(batch, *args, **kwargs)

        if batch.forward_mode.is_extend() or batch.is_extend_in_batch:
            return self._dual_prefill(batch, *args, **kwargs)

        self._rounds_total += 1
        batch_output = self._active_sub.forward_batch_generation(
            batch, *args, **kwargs
        )
        # Keep the INACTIVE rung warm (see the section comment above).
        tic = time.perf_counter()
        if self._active_name == "dflash":
            self._warmkeep_nextn_after_dflash_round(batch, batch_output)
        else:
            self._warmkeep_dflash_after_nextn_round(batch, batch_output)
        self._warmkeep_ms_since_switch += (time.perf_counter() - tic) * 1e3
        self._warmkeep_rounds_since_switch += 1
        self._trace_round(batch, batch_output)
        return batch_output

    def on_verify_complete_cpu(
        self,
        num_correct_drafts_per_req: list,
        batch_size: int = 0,
        steps: Optional[int] = None,
    ) -> None:
        if (
            self._force == "auto"
            and steps is not None
            and num_correct_drafts_per_req
        ):
            # Feed the bandit estimator. Attribution by draft-token stride
            # (steps == stride - 1, recovered by the result processor): the
            # DFLASH rung's stride is its block size, every other stride is
            # a NEXTN k rung (collision excluded at argument time). Correct
            # under overlap: a delayed result is attributed to the rung that
            # PRODUCED it, not the currently active one.
            rung = (
                ("dflash", self._dflash_block_size)
                if steps + 1 == self._dflash_block_size
                else ("nextn", int(steps))
            )
            self._bandit.observe_result(
                rung, num_correct_drafts_per_req, batch_size
            )
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
