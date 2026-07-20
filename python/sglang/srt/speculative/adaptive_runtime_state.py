import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol

from sglang.srt.speculative.adaptive_spec_params import AdaptiveSpeculativeParams
from sglang.srt.utils import log_info_on_rank0

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from sglang.srt.layers.attention.base_attn_backend import AttentionBackend
    from sglang.srt.model_executor.cpu_graph_runner import CPUGraphRunner
    from sglang.srt.model_executor.runner import DecodeCudaGraphRunner
    from sglang.srt.speculative.eagle_draft_cuda_graph_runner import (
        EAGLEDraftCudaGraphRunner,
    )
    from sglang.srt.speculative.eagle_draft_extend_cuda_graph_runner import (
        EAGLEDraftExtendCudaGraphRunner,
    )


@dataclass
class SpecRuntimeState:
    """A complete set of runtime resources bound to a specific speculative
    decoding configuration.

    Each decode round runs three stages — draft, verify, extend — and every
    stage has shape-dependent resources (attention backends and CUDA graphs)
    that must match the current configuration.  Switching adaptive steps
    means swapping the entire state atomically.
    """

    # -- Configuration (determines shapes for all stages) --
    speculative_num_steps: int
    speculative_num_draft_tokens: int

    # -- Draft stage: draft model multi-step autoregressive generation --
    draft_attn_backend: "AttentionBackend | None"
    cuda_graph_runner: "EAGLEDraftCudaGraphRunner | None"

    # -- Verify stage: target model one-pass tree verification --
    target_attn_backend: "AttentionBackend"
    target_graph_runner: "DecodeCudaGraphRunner | CPUGraphRunner | None"

    # -- Extend stage: draft model KV cache catch-up after verify --
    draft_extend_attn_backend: "AttentionBackend | None"
    cuda_graph_runner_for_draft_extend: "EAGLEDraftExtendCudaGraphRunner | None"


_WORKSPACE_ATTRS = ("workspace_buffer", "full_cg_prefill_workspace_buffer")


def _iter_state_backends(state: SpecRuntimeState):
    """Yield (role, backend) for every attention backend of *state*,
    unwrapping hybrid prefill/decode wrappers into their leaf backends."""
    stack = [
        (role, backend)
        for role, backend in (
            ("draft", state.draft_attn_backend),
            ("target", state.target_attn_backend),
            ("draft_extend", state.draft_extend_attn_backend),
        )
        if backend is not None
    ]
    while stack:
        role, backend = stack.pop()
        yield role, backend
        for attr in ("prefill_backend", "decode_backend"):
            inner = getattr(backend, attr, None)
            # Only descend into actual wrapped backend objects (e.g.
            # HybridAttnBackend children). FlashInferAttnBackend reuses these
            # attribute names for plain STRING kernel selectors ("fa2"), and
            # interned strings would alias across states and trip the
            # object-identity check as a false positive.
            if inner is not None and hasattr(inner, "init_forward_metadata"):
                stack.append((f"{role}.{attr}", inner))


def _workspace_ptrs(backend) -> list[tuple[str, int]]:
    """(attr, data_ptr) of every flashinfer-style workspace buffer on *backend*."""
    out = []
    for attr in _WORKSPACE_ATTRS:
        buf = getattr(backend, attr, None)
        data_ptr = getattr(buf, "data_ptr", None)
        if callable(data_ptr):
            out.append((attr, data_ptr()))
    return out


def assert_runtime_state_isolation(
    states: dict[int, "SpecRuntimeState"],
    baseline_steps: "set[int] | None" = None,
) -> None:
    """Enforce the M16/#50 invariant: adaptive runtime states must not share
    attention backends or PRIVATE flashinfer workspaces WITH EACH OTHER.

    Each state's CUDA graphs were captured against that state's backend
    graph-state buffers and flashinfer workspace/plan. A backend (or a private
    workspace buffer) reachable from two different states means an earlier
    captured graph replays against buffers that another state re-plans or
    overwrites — the stale-pointer / shared-workspace corruption class behind
    M16 and the #50 hetero-spec divergence. Sharing WITHIN one state is
    allowed (the registered static/initial state legitimately aliases the
    process-global workspace across its own backends).

    *baseline_steps* names the externally registered (static-path) states.
    Workspace pointers reachable from those states are the PROCESS-GLOBAL
    workspace family: built states may alias them (e.g. the EAGLE draft-extend
    backend is constructed without init_new_workspace and time-multiplexes the
    global workspace exactly like every non-adaptive forward does; it is
    re-planned before each use and its zero-state is maintained by
    zero_flashinfer_workspaces). Such aliasing is logged, not fatal. A pointer
    shared between two states that is NOT part of the baseline family is a
    genuine private-share bug and raises.

    Relation to adaptive graph-memory OFFLOAD (#93): the managed alias layer
    (adaptive_graph_memory) multiplexes PHYSICAL pages under per-state
    buffers whose VIRTUAL addresses stay distinct, so managed aliasing is
    invisible to -- and intentionally exempt from -- this data_ptr-identity
    check: it never produces equal pointers across states. Unmanaged private
    sharing therefore remains exactly as fatal as before. The physical-level
    counterpart of this invariant (pausing one state must not unmap memory
    referenced by another) is enforced separately by
    AdaptiveGraphMemoryManager.finalize_boot's segment-isolation audit.
    """
    baseline_steps = baseline_steps or set()
    baseline_ws: set[int] = set()
    for steps in baseline_steps:
        state = states.get(steps)
        if state is None:
            continue
        for _, backend in _iter_state_backends(state):
            baseline_ws.update(ptr for _, ptr in _workspace_ptrs(backend))

    seen_backend: dict[int, tuple[int, str]] = {}
    seen_ws: dict[int, tuple[int, str, str]] = {}
    for steps, state in sorted(states.items()):
        for role, backend in _iter_state_backends(state):
            prev = seen_backend.get(id(backend))
            if prev is not None and prev[0] != steps:
                raise RuntimeError(
                    "Adaptive runtime state isolation violated: attention "
                    f"backend shared between states steps={prev[0]} ({prev[1]}) "
                    f"and steps={steps} ({role}). Each state must build its own "
                    "backends (init_new_workspace=True); see M16/#50."
                )
            seen_backend[id(backend)] = (steps, role)
            for attr, ptr in _workspace_ptrs(backend):
                prev_ws = seen_ws.get(ptr)
                if prev_ws is not None and prev_ws[0] != steps:
                    if ptr in baseline_ws:
                        logger.info(
                            "Adaptive runtime states steps=%s (%s.%s) and "
                            "steps=%s (%s.%s) alias the process-global "
                            "flashinfer workspace (status-quo of the static "
                            "path; re-planned before each use).",
                            prev_ws[0],
                            prev_ws[1],
                            prev_ws[2],
                            steps,
                            role,
                            attr,
                        )
                    else:
                        raise RuntimeError(
                            "Adaptive runtime state isolation violated: "
                            "flashinfer workspace shared between states "
                            f"steps={prev_ws[0]} ({prev_ws[1]}.{prev_ws[2]}) "
                            f"and steps={steps} ({role}.{attr}) at data_ptr "
                            f"0x{ptr:x}. Captured graphs of one state would "
                            "replay against a workspace another state plans "
                            "into; see M16/#50."
                        )
                seen_ws[ptr] = (steps, role, attr)


class AdaptiveSpecWorker(Protocol):
    """Protocol that a worker must implement to use AdaptiveController."""

    speculative_num_steps: int

    def build_adaptive_runtime_state(
        self,
        speculative_num_steps: int,
        speculative_num_draft_tokens: int,
        cuda_graph_bs: list[int] | None = None,
    ) -> SpecRuntimeState: ...

    def apply_runtime_state(self, state: SpecRuntimeState) -> None: ...


class AdaptiveController:
    """Facade that owns adaptive decision-making and runtime state switching.

    Works with any worker that implements AdaptiveSpecWorker protocol:
      - build_adaptive_runtime_state(steps, draft_tokens) → runtime state
      - apply_runtime_state(state) → apply it to the worker

    The worker only needs to:
      1. Call register() for the initial state, then init_states()
         once during startup.
      2. Call on_verify_complete(num_correct_drafts_per_req) after each decode verify.
    """

    def __init__(
        self,
        worker: AdaptiveSpecWorker,
        config_path: str | None = None,
        algorithm: str | None = None,
    ):
        from sglang.srt.speculative.adaptive_graph_memory import (
            AdaptiveGraphMemoryManager,
            resolve_adaptive_graph_memory_mode,
        )

        self.worker = worker
        self.params = AdaptiveSpeculativeParams(
            initial_steps=worker.speculative_num_steps,
            cfg_path=config_path,
            algorithm=algorithm,
        )
        self._states: dict[int, SpecRuntimeState] = {}
        self._registered_steps: set[int] = set()
        self._forced_swap_counter = 0
        server_args = getattr(worker, "server_args", None)
        mode = (
            resolve_adaptive_graph_memory_mode(server_args)
            if server_args is not None
            else "resident"
        )
        self.graph_memory = AdaptiveGraphMemoryManager(mode=mode)
        log_info_on_rank0(
            logger, f"Adaptive graph memory mode: {self.graph_memory.mode}"
        )

    @property
    def candidate_steps(self) -> list[int]:
        return self.params.candidate_steps

    def register(self, state: SpecRuntimeState, steps: int | None = None) -> None:
        """Register a pre-built runtime state.

        *steps* defaults to state.speculative_num_steps when not given.
        """
        key = steps if steps is not None else state.speculative_num_steps
        self._states[key] = state
        # Externally registered states are the static-path baseline; their
        # workspace pointers define the process-global workspace family for
        # the isolation check in init_states.
        self._registered_steps.add(key)

    def init_states(self, cuda_graph_bs: list[int] | None = None) -> None:
        """Build and register runtime states for all candidate steps."""
        self.params.set_cuda_graph_bs(cuda_graph_bs)

        # Offload mode builds LARGEST k first: its capture runs with maximum
        # headroom (no other built state mapped yet, each state is paused as
        # soon as it is built), and its footprint is the max-state reserve
        # that finalize_boot verifies. Resident mode keeps the historical
        # ascending order (behavior-neutral there, and T75 measurements were
        # taken with it).
        offload = self.graph_memory.offload_enabled
        build_order = sorted(self.candidate_steps, reverse=offload)
        for steps in build_order:
            if steps in self._states:
                continue

            pruned_bs = self.params.cuda_graph_bs_for_step(steps)
            with self.graph_memory.build_state(steps):
                state = self.worker.build_adaptive_runtime_state(
                    speculative_num_steps=steps,
                    speculative_num_draft_tokens=steps + 1,
                    cuda_graph_bs=pruned_bs,
                )
            self._states[steps] = state
            self.graph_memory.pause_after_build(steps)

        # Enforced M16/#50 invariant: no backend or private flashinfer
        # workspace may be reachable from two different runtime states (each
        # state's graphs were captured against its own buffers/plans).
        # Aliasing of the process-global workspace (defined by the registered
        # static-path state) is logged, not fatal — see the assert docstring.
        assert_runtime_state_isolation(
            self._states, baseline_steps=self._registered_steps
        )

        # Offload mode: physical-isolation audit + max-state reserve check
        # (the no-OOM-on-swap guarantee), then map the initial state.
        self.graph_memory.finalize_boot(self.worker.speculative_num_steps)

        # Start on the initial step.
        self._activate(self.worker.speculative_num_steps)

    def activate_step_by_batch(self, batch_size: int) -> None:
        target = self.params.get_steps_for_batch(batch_size)
        if target != self.worker.speculative_num_steps:
            self._activate(target)

    def on_verify_complete(
        self, num_correct_drafts_per_req: list[int], batch_size: int
    ) -> None:
        """Feed verify results; switch runtime state if EMA warrants it.

        DETERMINISM INVARIANT (#50): *num_correct_drafts_per_req* MUST derive
        from the rank-0-broadcast accept counts (``result.accept_lens``, whose
        source tensor ``num_correct_drafts`` is broadcast from src=0 in
        ``eagle_utils.eagle_sample``) — never from a locally computed per-rank
        statistic. Together with *batch_size* (identical on all ranks) this
        keeps the step decision a pure function of rank-invariant inputs, so
        every TP/DCP rank swaps to the same graph on the same decode step.
        Guarded by the AST ratchet in test_draft_pick_rank_sync.py.
        """
        if self._maybe_forced_swap():
            return
        new_step = self.params.on_verify_complete(
            num_correct_drafts_per_req, batch_size
        )
        if new_step is not None:
            self._activate(new_step)

    def _maybe_forced_swap(self) -> bool:
        """TEST-ONLY stress hook (SGLANG_ADAPTIVE_FORCE_SWAP_INTERVAL=N):
        cycle through the candidate steps every N verify completions,
        overriding the EMA decision. Rank-deterministic: driven purely by
        the verify-call counter, which is identical on all ranks."""
        from sglang.srt.environ import envs

        interval = envs.SGLANG_ADAPTIVE_FORCE_SWAP_INTERVAL.get()
        if interval <= 0:
            return False
        self._forced_swap_counter += 1
        if self._forced_swap_counter % interval == 0:
            order = self.candidate_steps
            cur = self.worker.speculative_num_steps
            idx = (order.index(cur) + 1) % len(order) if cur in order else 0
            self._activate(order[idx])
        return True

    def _activate(self, speculative_num_steps: int) -> None:
        state = self._states.get(speculative_num_steps)
        if state is None:
            raise ValueError(
                f"Missing adaptive runtime state for steps={speculative_num_steps}"
            )
        # Offload mode: map the target state's physical pages (and unmap the
        # outgoing built state's) BEFORE the worker's pointer swap. No-op in
        # resident mode / when the target is already mapped.
        self.graph_memory.ensure_active(speculative_num_steps)
        self.worker.apply_runtime_state(state)
