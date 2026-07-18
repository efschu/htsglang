"""Physical-memory offload for adaptive speculative-decoding runtime states.

DESIGN (Task #93) -- reserve = max(one state), not sum(all states)
==================================================================

Problem
-------
Adaptive draft-length (``--speculative-adaptive``) pre-builds one complete
runtime state (attention backends + CUDA graphs) per candidate step count k.
Every state beyond the baseline costs ~1.1-1.35 GB of VRAM that sits idle
whenever that k is not active: on a 20 GB RTX 3080 rig, two extra states cost
+2.4 GB of graph reserve and shrank KV capacity from 261k to 39k tokens
(T75 GPU validation).

Key observation (user-originated, conceptually validated): the per-state
buffers are SCRATCH. Their content is disposable between decode steps -- only
the KV cache carries state, and the KV cache is shared across all k-states,
never duplicated. CUDA graphs bake in fixed VIRTUAL addresses, so the buffers
may never move; but the PHYSICAL pages behind them can be unmapped while a
state is inactive and remapped when it becomes active again.

Mechanism
---------
torch_memory_saver (an existing sglang dependency, the engine behind
``--enable-memory-saver``) allocates tagged regions through the CUDA
VirtualMemory API: ``pause(tag)`` unmaps and releases the physical pages
behind every allocation carrying that tag while keeping the virtual address
reservation intact; ``resume(tag)`` maps fresh physical pages at the same
virtual addresses. Captured CUDA graphs replay unchanged afterwards because
they only reference the (stable) virtual addresses.

torch_memory_saver has no API for mapping ONE physical allocation behind
MULTIPLE virtual ranges, so the literal "one shared physical pool, N virtual
aliases" variant is not directly expressible. pause/resume achieves the same
steady-state footprint -- only the active state's pages are mapped -- at the
cost of a physical map/unmap per swap instead of an access switch. Measured
swap rate is ~0.1/s (T75), so a ms-scale swap is irrelevant against the
~us pointer swap it replaces.

What is tagged (paused/resumed), what stays resident, what lives in RAM
-----------------------------------------------------------------------
Tagged per state (the dominant VRAM cost; each allocation is >= MiB-scale so
it occupies its own caching-allocator segment -- see "Isolation" below):

* the state's private flashinfer float workspace(s)
  (``init_new_workspace=True`` allocations, 384-512 MiB each),
* the flashinfer CUDA-graph state buffers allocated by
  ``init_cuda_graph_state``: ``cuda_graph_kv_indices`` (per wrapper /
  per draft step), ``cuda_graph_custom_mask``.

Resident (small, or shared with the static path, or not tensor memory):

* the process-global flashinfer workspace (time-multiplexed by ALL states
  and the non-spec path; never paused),
* static graph input buffers (EagleDraftInputBuffers / DecodeInputBuffers,
  few MiB; DecodeInputBuffers also references the SHARED logits buffer),
* small indptr / last-page-len buffers and their clones (KiB),
* flashinfer wrapper objects and their int workspaces (~8 MiB each),
* cudaGraphExec_t instantiation memory (driver-owned, not interceptable).

Host RAM: everything needed to re-activate a state is either already
host-resident (wrapper objects, host-side plan state inside flashinfer's
indices updaters, pinned staging buffers) or re-derivable on the device
(see re-init below). Nothing is copied device-to-host at swap time
(``enable_cpu_backup`` stays False): pause DISCARDS the tagged content.

Re-initialization on resume ("what needs re-init and why")
----------------------------------------------------------
After ``resume(tag)`` the tagged buffers contain undefined data. Exactly two
content classes live there, and both have a cheap deterministic re-init:

1. Float workspaces: the #50 investigation established that flashinfer's
   split-KV kernels read workspace regions the current forward did not write,
   and the validated contract is a ZEROED workspace (restored at request
   boundaries by ``zero_flashinfer_workspaces``). Zeroing on resume restores
   exactly the boot-state contract. (While a workspace is paused, the
   request-boundary zeroing must skip it -- see ``is_paused_tensor`` -- its
   zero happens at resume instead.)
2. Graph-state index/mask buffers: every replay's
   ``init_forward_metadata_replay_cuda_graph`` / draft ``common_template``
   rewrites the lanes its graph reads before launch, and plan data is
   re-planned from host state every forward. Zeroing on resume gives padded /
   not-yet-rewritten lanes a safe value (page index 0 is always a valid pool
   page), identical to the fresh-boot state before the first replay.

flashinfer plan buffers and semaphores live in the wrapper int workspaces,
which stay RESIDENT -- they are never paused, so no re-init is needed;
re-planning happens per forward exactly as on the static path.

Swap sequence (offload mode)
----------------------------
``torch.cuda.synchronize()`` (no in-flight kernel of the old state; the
existing swap sites run between scheduler steps, but attr-rebinding alone
never guaranteed quiescence -- the synchronize does), then ``pause(old)``,
then ``resume(new)`` + zero of the new state's noted tensors, then the
worker's ordinary pointer swap. Pausing BEFORE resuming guarantees the
physical pages freed by the outgoing state are available to the incoming one,
which combined with the boot-time reserve check makes a swap-time OOM
impossible (see below).

Reserve rule and the no-OOM-on-swap guarantee
---------------------------------------------
* resident mode: reserve = SUM of all built states (status quo).
* offload mode: reserve = MAX over the built states. Enforced at boot:
  states are built LARGEST k FIRST, each state is paused as soon as its build
  completes (so the boot peak is ~one state, not the sum), and after all
  builds ``finalize_boot`` verifies -- with every built state paused -- that
  free device memory >= the largest state's tagged bytes. Since at most one
  built state is ever resumed, and pause precedes resume on every swap, the
  physical pool is max-state-sized from boot onward: no lazy growth, no
  swap-time allocation beyond what pause just released. A failed check is a
  fatal, actionable error (raise the graph reserve or use resident mode).

Isolation / audit (G4 evolution)
--------------------------------
Virtual addresses are NEVER shared across states in this design, so the
existing ``assert_runtime_state_isolation`` (M16/#50) applies unchanged.
The new hazard class is PHYSICAL: pausing tag A must not unmap memory that
another state (or the static path) still uses. Two structural guarantees plus
one audit close it:

* Every tag allocates from its own private ``torch.cuda.MemPool``. This is
  the load-bearing guarantee: ``pause(tag)`` unmaps a tag's segments
  wholesale INCLUDING their free tails, and the caching allocator both
  packs 1-10 MiB allocations into shared 20 MiB segments (kLargeBuffer) and
  serves any large-enough free block regardless of who created the segment
  -- so with a shared pool, one tag's buffer can be placed into another
  (paused) tag's segment tail and fault on first touch (observed live on
  the 5-state boot). Per-tag pools confine a tag's free space to that tag,
  whose allocations happen only during its own build. Additionally only
  individually-noted, never-freed allocations >= MIN_TAGGED_BYTES are
  tagged; smaller ones stay resident in the default pool.
* Tags are only pausable through this manager, which enforces "at most one
  built state resumed".
* ``finalize_boot`` audits via ``torch.cuda.memory_snapshot()`` that every
  noted tensor of every tag lies in a segment created during THAT tag's build
  window; any cross-tag placement is a fatal error (fallback: resident mode).

Rank determinism
----------------
The swap decision is a pure function of rank-invariant inputs (rank-0
broadcast accept counts + batch size; AST-ratcheted in
test_draft_pick_rank_sync.py), so every TP/DCP rank performs the same
pause/resume on the same decode step. ``SGLANG_ADAPTIVE_ALIAS_VERIFY_RANK_SYNC=1``
additionally all-gathers (swap ordinal, target steps) across the TP CPU group
on every swap and asserts equality (stress/debug tool; a diverged rank turns
silent corruption into an immediate failure).

Modes
-----
``--speculative-adaptive-graph-memory {auto,resident,offload}``:

* ``resident``: status quo -- all states fully materialized, ~us pointer
  swaps, sum-of-states reserve. First-class mode for rigs with VRAM to spare.
* ``offload``: the tagged buffers of inactive states are unmapped -- ms-scale
  swaps, max-state reserve, KV capacity recovered.
* ``auto`` (default): offload when the prerequisites hold (CUDA device,
  flashinfer attention backend, torch_memory_saver importable, no
  expandable_segments), else resident. Offload is the intended default for
  adaptive mode because idle graph states are pure VRAM waste on the rigs
  adaptive mode targets; explicit ``offload`` with missing prerequisites is
  a hard error, ``auto`` degrades to resident with a log line.
"""

from __future__ import annotations

import logging
import os
import time
from contextlib import contextmanager, nullcontext
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Optional

import torch

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from sglang.srt.server_args import ServerArgs

# Only allocations at least this large are tagged. Large allocations receive
# dedicated caching-allocator segments, which is what makes per-tag
# pause/resume safe (a tagged segment hosts exactly one noted tensor); small
# allocations would share segments across tags and stay resident instead.
MIN_TAGGED_BYTES = 2 * 1024 * 1024

_MODES = ("auto", "resident", "offload")

# Process-wide active manager. One scheduler process owns at most one
# AdaptiveController and therefore one manager; the flashinfer wrap sites
# (tagged_state_alloc / note calls) and zero_flashinfer_workspaces reach it
# through this module handle.
_ACTIVE_MANAGER: Optional["AdaptiveGraphMemoryManager"] = None


def resolve_adaptive_graph_memory_mode(server_args: "ServerArgs") -> str:
    """Resolve {auto,resident,offload} -> {resident,offload}.

    Must be computable identically in the launcher process (which decides
    whether to spawn schedulers with the torch_memory_saver LD_PRELOAD hook)
    and in the scheduler process (which builds the manager), so it only looks
    at server args, imports, and inherited environment -- never CUDA state.
    """
    requested = getattr(server_args, "speculative_adaptive_graph_memory", "auto")
    if requested not in _MODES:
        raise ValueError(
            f"speculative_adaptive_graph_memory must be one of {_MODES}, "
            f"got {requested!r}"
        )
    if not server_args.speculative_adaptive:
        return "resident"
    if requested == "resident":
        return "resident"

    def _fail_or_resident(reason: str) -> str:
        if requested == "offload":
            raise ValueError(
                f"--speculative-adaptive-graph-memory offload is not usable: "
                f"{reason}. Use 'resident' (all states stay materialized in "
                "VRAM) or fix the prerequisite."
            )
        logger.info(
            "Adaptive graph memory: auto-resolving to 'resident' (%s).", reason
        )
        return "resident"

    if server_args.device not in (None, "cuda"):
        return _fail_or_resident(f"device={server_args.device} is not CUDA")
    # The offload wrap sites live in the flashinfer backend; other attention
    # backends would yield zero tagged bytes (silent no-op), so require it.
    attn_backend = (
        server_args.attention_backend
        or server_args.decode_attention_backend
        or "flashinfer"
    )
    if attn_backend != "flashinfer":
        return _fail_or_resident(
            f"attention backend {attn_backend!r} has no offload wrap sites "
            "(flashinfer required)"
        )
    if "expandable_segments:True" in os.environ.get("PYTORCH_CUDA_ALLOC_CONF", ""):
        return _fail_or_resident(
            "PYTORCH_CUDA_ALLOC_CONF expandable_segments is incompatible with "
            "torch_memory_saver"
        )
    try:
        import torch_memory_saver  # noqa: F401
    except ImportError as e:
        return _fail_or_resident(f"torch_memory_saver is not importable ({e})")
    return "offload"


@dataclass
class _StateRecord:
    tag: str
    steps: int
    tensors: list = field(default_factory=list)
    # [start, end) virtual-address ranges of allocator segments that appeared
    # during this state's build window (for the finalize_boot audit).
    segment_ranges: list = field(default_factory=list)

    @property
    def nbytes(self) -> int:
        return sum(t.untyped_storage().nbytes() for t in self.tensors)


def _snapshot_segment_addrs() -> dict[int, int]:
    """{segment start address: total size} from the caching allocator."""
    out = {}
    try:
        for seg in torch.cuda.memory_snapshot():
            addr = seg.get("address")
            size = seg.get("total_size")
            if addr is not None and size:
                out[addr] = size
    except Exception:  # pragma: no cover - snapshot is diagnostics-only
        logger.warning("torch.cuda.memory_snapshot failed", exc_info=True)
    return out


class AdaptiveGraphMemoryManager:
    """Owns the per-k-state tagged memory and the pause/resume swap path.

    In ``resident`` mode every method is a cheap no-op, preserving the
    pre-existing behavior bit-for-bit (states stay materialized, activation
    is a pure pointer swap in the worker).
    """

    def __init__(self, mode: str, tp_cpu_group=None):
        assert mode in ("resident", "offload"), mode
        self.mode = mode
        self._states: dict[str, _StateRecord] = {}
        self._pools: dict[str, "torch.cuda.MemPool"] = {}
        self._paused: set[str] = set()
        self._resumed_tag: Optional[str] = None  # at most one built tag mapped
        self._build_tag: Optional[str] = None
        self._pre_build_segments: Optional[dict[int, int]] = None
        self._finalized = False
        self._swap_ordinal = 0
        self.last_swap_ms: Optional[float] = None
        self._tp_cpu_group = tp_cpu_group
        self._adapter = None
        if self.mode == "offload":
            ld_preload = os.environ.get("LD_PRELOAD", "")
            if "torch_memory_saver" not in ld_preload:
                raise RuntimeError(
                    "Adaptive graph-memory offload requires the "
                    "torch_memory_saver preload hook, but LD_PRELOAD does not "
                    "contain it. The sglang launcher injects it automatically "
                    "when the offload mode resolves before scheduler spawn; "
                    "for direct scheduler runs set LD_PRELOAD to the "
                    "torch_memory_saver_hook_mode_preload library, or use "
                    "--speculative-adaptive-graph-memory resident."
                )
            from sglang.srt.utils.torch_memory_saver_adapter import (
                TorchMemorySaverAdapter,
            )

            self._adapter = TorchMemorySaverAdapter.create(enable=True)

        global _ACTIVE_MANAGER
        _ACTIVE_MANAGER = self

    # ------------------------------------------------------------------
    # Build phase
    # ------------------------------------------------------------------
    @staticmethod
    def tag_for_steps(steps: int) -> str:
        return f"adaptive_state_k{steps}"

    @contextmanager
    def build_state(self, steps: int):
        """Scope one candidate state's build. Wrap sites tag allocations
        only while a build scope is active (static-path init never is)."""
        if self.mode != "offload":
            yield
            return
        tag = self.tag_for_steps(steps)
        assert self._build_tag is None, "nested adaptive state builds"
        assert tag not in self._states, f"state {tag} built twice"
        rec = _StateRecord(tag=tag, steps=steps)
        self._states[tag] = rec
        self._pre_build_segments = _snapshot_segment_addrs()
        self._build_tag = tag
        try:
            yield
        finally:
            self._build_tag = None
            post = _snapshot_segment_addrs()
            pre = self._pre_build_segments or {}
            self._pre_build_segments = None
            rec.segment_ranges = [
                (addr, addr + size)
                for addr, size in post.items()
                if addr not in pre
            ]
        logger.info(
            "Adaptive graph memory: built %s, tagged %.1f MiB in %d buffers",
            tag,
            rec.nbytes / (1 << 20),
            len(rec.tensors),
        )

    @contextmanager
    def _tagged_region(self):
        if self.mode != "offload" or self._build_tag is None:
            yield
            return
        # One private MemPool PER TAG -- a hard correctness requirement, not
        # an optimization. pause(tag) unmaps a tag's segments wholesale,
        # including their free tails, and the caching allocator packs
        # 1-10 MiB allocations into shared 20 MiB segments (kLargeBuffer)
        # and serves any sufficiently large free block regardless of which
        # region entry created the segment. With a single shared pool, one
        # tag's allocation can land in another (already paused) tag's
        # segment tail -> illegal memory access on first touch (observed
        # live: state k2's multistep kv_indices landed in paused k4's 20 MiB
        # segment at 0x208af00000). Per-tag pools make cross-tag free-list
        # reuse structurally impossible: a tag's free space is only visible
        # to allocations of that same tag, which happen only during its own
        # build. The pools stay alive for the process lifetime.
        tag = self._build_tag
        pool = self._pools.get(tag)
        if pool is None:
            pool = self._pools[tag] = torch.cuda.MemPool()
        with torch.cuda.use_mem_pool(pool):
            with self._adapter.region_config(tag=tag):
                yield

    def note_tensor(self, t: Optional[torch.Tensor]) -> None:
        if self.mode != "offload" or self._build_tag is None or t is None:
            return
        if os.environ.get("SGLANG_ADAPTIVE_ALIAS_DEBUG"):
            logger.info(
                "AGM-DEBUG note %s ptr=0x%x nbytes=%d",
                self._build_tag,
                t.data_ptr(),
                t.untyped_storage().nbytes(),
            )
        self._states[self._build_tag].tensors.append(t)

    def pause_after_build(self, steps: int) -> None:
        """Pause a freshly built state so boot peak stays ~one state."""
        if self.mode != "offload":
            return
        tag = self.tag_for_steps(steps)
        rec = self._states.get(tag)
        if rec is None or not rec.tensors:
            return
        torch.cuda.synchronize()
        # Return freed temp segments to the driver between builds.
        torch.cuda.empty_cache()
        if os.environ.get("SGLANG_ADAPTIVE_ALIAS_DEBUG"):
            logger.info(
                "AGM-DEBUG pause %s segments=%s tensors=%s",
                tag,
                [(hex(lo), hex(hi)) for lo, hi in rec.segment_ranges],
                [
                    (hex(t.data_ptr()), t.untyped_storage().nbytes())
                    for t in rec.tensors
                ],
            )
        self._adapter.pause(tag)
        self._paused.add(tag)

    # ------------------------------------------------------------------
    # Boot finalize: audit + max-state reserve guarantee
    # ------------------------------------------------------------------
    def finalize_boot(self, initial_steps: int) -> None:
        """Audit tag/segment isolation, verify the max-state reserve, and
        bring the initial state up. Idempotence not needed (called once)."""
        if self.mode != "offload":
            return
        self._audit_segment_isolation()

        # Pause anything still mapped (defensive; builds already pause).
        for tag, rec in self._states.items():
            if tag not in self._paused and rec.tensors:
                torch.cuda.synchronize()
                self._adapter.pause(tag)
                self._paused.add(tag)

        sizes = {t: r.nbytes for t, r in self._states.items() if r.tensors}
        max_tag, max_bytes = (
            max(sizes.items(), key=lambda kv: kv[1]) if sizes else (None, 0)
        )
        free_bytes, total_bytes = torch.cuda.mem_get_info()
        logger.info(
            "Adaptive graph memory (offload): states=%s; reserve rule "
            "max(one state)=%.1f MiB (resident mode would keep the sum "
            "%.1f MiB mapped); free after pausing all states: %.1f MiB",
            {t: f"{b / (1 << 20):.1f} MiB" for t, b in sizes.items()},
            max_bytes / (1 << 20),
            sum(sizes.values()) / (1 << 20),
            free_bytes / (1 << 20),
        )
        # No-OOM-on-swap guarantee: with every built state unmapped, the
        # driver must be able to back the largest one. Since at most one
        # built state is ever resumed and pause precedes resume on every
        # swap, this check at boot implies every future swap succeeds.
        if free_bytes < max_bytes:
            raise RuntimeError(
                "Adaptive graph memory offload: free device memory "
                f"({free_bytes / (1 << 20):.0f} MiB) is below the largest "
                f"state's tagged footprint ({max_bytes / (1 << 20):.0f} MiB, "
                f"{max_tag}) even with all candidate states paused. A k-swap "
                "could OOM. Increase the graph/KV reserve so at least one "
                "state fits, or use --speculative-adaptive-graph-memory "
                "resident."
            )
        self._finalized = True
        # Map the initial state (a registered baseline has no tag and needs
        # no resume; a BUILT initial state does).
        self.ensure_active(initial_steps)

    def _audit_segment_isolation(self) -> None:
        """Every noted tensor must live in a segment created during its own
        tag's build window (G4-evolution physical-isolation audit).

        Only windows whose segments STILL exist participate: a segment freed
        after a build (empty_cache between builds) can have its virtual range
        recycled by a later build, which must not count as cross-tag
        placement."""
        live = _snapshot_segment_addrs()
        live_ranges = {
            tag: [
                (lo, hi)
                for lo, hi in rec.segment_ranges
                if live.get(lo) == hi - lo
            ]
            for tag, rec in self._states.items()
        }
        for tag, rec in self._states.items():
            for t in rec.tensors:
                ptr = t.data_ptr()
                home = None
                for other_tag, ranges in live_ranges.items():
                    if any(lo <= ptr < hi for lo, hi in ranges):
                        home = other_tag
                        break
                if home is not None and home != tag:
                    raise RuntimeError(
                        "Adaptive graph memory offload: tagged buffer of "
                        f"{tag} (ptr 0x{ptr:x}, {t.untyped_storage().nbytes()}"
                        f" bytes) lies in a segment created during {home}'s "
                        "build window. Pausing one state would unmap another "
                        "state's memory. This is a bug in the tagged-alloc "
                        "wrap sites; falling back to "
                        "--speculative-adaptive-graph-memory resident is safe."
                    )
                if home is None:
                    # Not fatal by itself (snapshot attribution can miss a
                    # segment) but worth surfacing: the disjointness proof
                    # does not cover this tensor.
                    logger.warning(
                        "Adaptive graph memory: could not attribute tagged "
                        "buffer of %s (ptr 0x%x) to a build-window segment; "
                        "physical-isolation audit incomplete for it.",
                        tag,
                        ptr,
                    )

    # ------------------------------------------------------------------
    # Swap path
    # ------------------------------------------------------------------
    def ensure_active(self, steps: int) -> None:
        """Make the state for *steps* the (only) mapped built state.

        Called on every runtime-state activation, BEFORE the worker's
        pointer swap. No-op in resident mode and when the target is already
        mapped (including registered baseline states, which are never
        tagged and always resident).
        """
        if self.mode != "offload":
            return
        tag = self.tag_for_steps(steps)
        rec = self._states.get(tag)
        target = tag if (rec is not None and rec.tensors) else None
        if target == self._resumed_tag:
            return

        tic = time.perf_counter()
        # Quiescence: attr-rebinding never required this, but unmapping the
        # outgoing state's pages under in-flight kernels (e.g. the previous
        # step's draft_extend still queued) would be use-after-unmap.
        torch.cuda.synchronize()
        if self._resumed_tag is not None:
            # Pause BEFORE resume: releases the outgoing pages so the
            # incoming state maps into memory the boot check accounted for.
            self._adapter.pause(self._resumed_tag)
            self._paused.add(self._resumed_tag)
            self._resumed_tag = None
        if target is not None:
            self._adapter.resume(target)
            self._paused.discard(target)
            self._resumed_tag = target
            # Re-init: restore the boot-state content contract (zeroed
            # workspaces per #50; zeroed index/mask buffers == fresh-boot
            # pre-first-replay state). Plan data is host-side and re-planned
            # per forward; nothing else is stateful. See module docstring.
            for t in rec.tensors:
                t.zero_()
        torch.cuda.synchronize()
        self.last_swap_ms = (time.perf_counter() - tic) * 1e3
        self._swap_ordinal += 1
        logger.info(
            "Adaptive graph memory swap #%d: mapped=%s (%.2f ms)",
            self._swap_ordinal,
            target or "<baseline>",
            self.last_swap_ms,
        )
        self._maybe_verify_rank_sync(steps)

    def _maybe_verify_rank_sync(self, steps: int) -> None:
        from sglang.srt.environ import envs

        if not envs.SGLANG_ADAPTIVE_ALIAS_VERIFY_RANK_SYNC.get():
            return
        try:
            import torch.distributed as dist

            if not dist.is_initialized() or dist.get_world_size() == 1:
                return
            if self._tp_cpu_group is None:
                from sglang.srt.distributed import get_tp_group

                self._tp_cpu_group = get_tp_group().cpu_group
            payload = (self._swap_ordinal, steps)
            gathered: list = [None] * dist.get_world_size(self._tp_cpu_group)
            dist.all_gather_object(gathered, payload, group=self._tp_cpu_group)
            if any(g != payload for g in gathered):
                raise RuntimeError(
                    "Adaptive graph memory: rank-divergent swap detected: "
                    f"local={payload}, all={gathered}. The swap decision must "
                    "be a pure function of rank-invariant inputs (#50/G5)."
                )
        except RuntimeError:
            raise
        except Exception:
            logger.warning(
                "SGLANG_ADAPTIVE_ALIAS_VERIFY_RANK_SYNC check skipped",
                exc_info=True,
            )

    # ------------------------------------------------------------------
    # Queries
    # ------------------------------------------------------------------
    def is_paused_tensor(self, t: torch.Tensor) -> bool:
        if self.mode != "offload":
            return False
        for tag in self._paused:
            rec = self._states.get(tag)
            if rec is not None and any(x is t for x in rec.tensors):
                return True
        return False

    @property
    def swap_count(self) -> int:
        return self._swap_ordinal


# ----------------------------------------------------------------------
# Module-level hooks for the (backend-side) wrap sites
# ----------------------------------------------------------------------
def tagged_state_alloc(nbytes: Optional[int] = None):
    """Context manager: route the enclosed allocation to the current
    adaptive build tag's pauseable region. No-op outside a build scope
    (static path), in resident mode, or -- CRITICALLY -- when *nbytes*
    is below MIN_TAGGED_BYTES.

    The size gate is a correctness requirement, not an optimization: the
    caching allocator SPLITS large blocks, so a tagged segment can carry a
    free tail of up to ~2 MiB. A later sub-2MiB allocation in another tag's
    region could be served from that tail -- and once the first tag is
    paused, the tail is UNMAPPED, so first touch is an illegal memory
    access (observed live on the 5-state high-accept boot: state k2's
    ~1.5 MiB custom_mask landed in paused k5's segment tail). Allocations
    >= MIN_TAGGED_BYTES always get their own segment, closing the hole;
    smaller ones stay resident in the default pool.
    """
    mgr = _ACTIVE_MANAGER
    if mgr is None:
        return nullcontext()
    if nbytes is not None and nbytes < MIN_TAGGED_BYTES:
        return nullcontext()
    return mgr._tagged_region()


def note_state_tensor(t: Optional[torch.Tensor]) -> None:
    """Register *t* as pauseable scratch of the state currently being built
    (zeroed on every resume). Only call for tensors allocated inside
    ``tagged_state_alloc``. Sub-MIN_TAGGED_BYTES tensors are skipped (they
    were allocated OUTSIDE the tagged region by the size gate above and
    must stay resident)."""
    mgr = _ACTIVE_MANAGER
    if mgr is not None and t is not None:
        if t.untyped_storage().nbytes() < MIN_TAGGED_BYTES:
            return
        mgr.note_tensor(t)


def in_offload_build() -> bool:
    """True while an offload-mode adaptive state build scope is active."""
    mgr = _ACTIVE_MANAGER
    return (
        mgr is not None and mgr.mode == "offload" and mgr._build_tag is not None
    )


def is_paused_tensor(t: torch.Tensor) -> bool:
    """True if *t* is currently unmapped scratch of an inactive adaptive
    state (its content contract is restored at resume; request-boundary
    zeroing must skip it)."""
    mgr = _ACTIVE_MANAGER
    return mgr is not None and mgr.is_paused_tensor(t)
