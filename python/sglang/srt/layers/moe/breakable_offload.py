# SPDX-License-Identifier: Apache-2.0
"""#462: the BREAKABLE MoE expert-offload graph route.

WHAT THIS IS
------------
The one route that puts a MoE expert offload under CUDA graphs without
re-importing the cost #452 refuted. Its whole content is a placement decision:

    the graph addresses SLOTS; the eager phase decides which expert occupies
    which slot, materialises its bytes there, and publishes the mapping --
    all of it BEFORE the replay that reads them.

``ANALYSE_456_dsv4f_matrix_sweep.md`` §2.2 cell #302b states the structural
fact this rests on: "a decode graph is captured against slot addresses, not
against which expert occupies a slot at replay time", so "any expert whose
bytes are materialised into the scratch slots eagerly, before replay runs, is
compatible with the graph". ``NOTE_452_desync_boot_refutation.md`` §3 calls the
same shape Option 3 and rates it "the only option with a plausibly positive
yield".

WHAT THIS IS NOT
----------------
It is NOT the capturable fetch. Fetching INSIDE the graph is refuted on
hardware and stays refuted: ``offload_capture_gate.refuse_capturable_offload_decode``
keeps refusing ``SGLANG_MOE_OFFLOAD_CUDA_GRAPH=1`` by name (B2 content
divergence, B4 6.60x). A graph cannot vary its work with the data, so a
captured gather must move the worst-case scratch set every layer every step
(2.128 GiB/token measured) where the eager fetch moves only the miss
(0.366-0.535 GiB/token measured). This module changes WHERE the fetch runs, not
whether the fetch is data-dependent -- the fetch stays eager, keeps its
measured volume, and the graph keeps only the launch-overhead saving on
everything around it.

HOW IT ATTACHES
---------------
``eager_on_graph`` (``runner_backend_utils/breakable_cuda_graph``) is the
existing mechanism: called during a breakable capture it ends the open segment,
runs the decorated callable eagerly, registers that callable as a break
function re-run on every replay, and opens a fresh segment. The decorated
callable here is :func:`breakable_moe_offload_fetch`, and the segment that
opens after it captures the MoE compute -- reading the slot arena and the
bridge buffer this call just filled.

This is why the route REQUIRES ``--cuda-graph-backend-decode=breakable``. Under
the ``full`` decode backend ``eager_on_graph`` is a pass-through (there is no
segment to split), so the host reads would execute inside a real stream capture,
where a D2H sync is illegal rather than merely slow. :func:`validate_breakable_boot`
refuses that combination by name instead of letting it fail at capture.

COST, STATED HONESTLY
---------------------
One host/device rendezvous plus one pinned H2D per MoE layer per step. The
rendezvous count is IRREDUCIBLE (see :data:`HOST_SYNCS_PER_LAYER_PER_STEP`).
Whether the graph's saving exceeds 43 rendezvous on DeepSeek-V4-Flash is the F2
measurement in ``docs/dev/TICKET_462_f2_and_replay.md``, and it is unmeasured
today -- which is why this route is OFF by default and every performance claim
about it is currently absent rather than optimistic.

DESK-WRITTEN, NEVER EXECUTED ON A CARD. Nothing here has served a token.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

from sglang.srt.layers.moe.offload_capture_gate import (
    ENV_GRAPH_MODE,
    MODE_BREAKABLE,
    MODE_CAPTURABLE,
    MODE_EAGER,
    MODES,
    BreakableModeRefused,
    env_graph_mode,
    validate_breakable_boot,
)
from sglang.srt.model_executor.runner_backend_utils.breakable_cuda_graph import (
    eager_on_graph,
)

logger = logging.getLogger(__name__)

#: Host/device RENDEZVOUS per MoE layer per forward on this route: exactly one,
#: ``topk_ids.tolist()``.
#:
#: It is irreducible, and the reason is worth stating once so nobody re-derives
#: it as a surprise: choosing which expert rows to fetch is host knowledge by
#: construction (that is what makes the fetch cheap -- it moves only the miss),
#: and MoE routing is SEQUENTIAL across layers, because layer L+1's router
#: consumes layer L's output. There is therefore no point in the step at which
#: several layers' routing decisions are simultaneously available to batch into
#: one sync. On DeepSeek-V4-Flash that is 43 rendezvous per step, and the
#: briefing's "43 baseline" is exactly this number: the route does not beat it,
#: it pays it. What the route removes is the eager path's ADDITIONAL host
#: blocking -- see :data:`HOST_BLOCKING_CROSSINGS_PER_LAYER_PER_STEP`.
HOST_SYNCS_PER_LAYER_PER_STEP = 1

#: Host-BLOCKING crossings per MoE layer per forward, this route vs the eager
#: path's ``run_waves``.
#:
#: eager (3): ``topk_ids.tolist()`` rendezvous, plus ``_build_lut``'s two
#: ``.to(device, non_blocking=True)`` copies out of numpy-backed PAGEABLE host
#: memory -- ``non_blocking`` is honoured only for pinned memory, so both block.
#: breakable (2): the same rendezvous, plus ONE copy out of a pinned staging
#: buffer whose bridge owns it exclusively.
#:
#: 43 layers x 3 = 129 crossings/step becomes 43 x 2 = 86, of which 43 are the
#: irreducible rendezvous. F2 prices what that is worth; this module claims
#: only the count.
HOST_BLOCKING_CROSSINGS_PER_LAYER_PER_STEP = 2
EAGER_HOST_BLOCKING_CROSSINGS_PER_LAYER_PER_STEP = 3

#: The ``OFFLOAD_CLASSES`` member the slot arena belongs to (#286).
ARENA_OFFLOAD_CLASS = "experts"


class BreakableScratchOverflow(RuntimeError):
    """A breakable step routed more distinct spill experts than it has slots.

    On the eager path this is recoverable: ``run_waves`` splits the forward into
    several waves. A captured segment cannot -- its work is fixed at capture
    time -- so the same condition here means the step would silently drop spill
    experts and return a WRONG result rather than a slow one. Refuse loudly.

    Carries the four numbers the remedy is computed from, because "raise the
    scratch" without them is a guess.
    """

    def __init__(
        self,
        layer_id: Any,
        spill: int,
        scratch: int,
        routed_slots: int,
        cold: int,
    ) -> None:
        self.layer_id = layer_id
        self.spill = int(spill)
        self.scratch = int(scratch)
        self.routed_slots = int(routed_slots)
        self.cold = int(cold)
        super().__init__(
            f"MoE breakable offload, layer {layer_id}: this step routes "
            f"{self.spill} distinct SPILL experts but the slot arena has only "
            f"{self.scratch} scratch slots. A captured segment cannot "
            f"wave-split, so continuing would drop spill experts and produce a "
            f"wrong result, not an approximate one. The worst case for this "
            f"capture shape is min(tokens x top_k = {self.routed_slots}, cold "
            f"set = {self.cold}) = {min(self.routed_slots, self.cold)}; set "
            f"SGLANG_MOE_SCRATCH_SLOTS to at least that, or lower "
            f"--cuda-graph-max-bs-decode. This should have been caught at boot "
            f"by validate_breakable_boot(); reaching it at runtime means the "
            f"captured shape is larger than the one that was validated."
        )


def breakable_opt_in() -> bool:
    """True when the operator selected the breakable route. Read per call."""
    return env_graph_mode() == MODE_BREAKABLE


@dataclass
class BreakableBridge:
    """One (bridge, stage) pair for ONE (layer, captured shape).

    ``buf`` is the device buffer the captured segment reads -- a fixed address,
    allocated in the eager region between segments, never reallocated. ``stage``
    is its pinned host mirror.

    The pair is allocated and owned TOGETHER, and that is deliberate rather than
    tidy: a single process-wide staging buffer shared by several bridges is
    precisely the shared-buffer family this fork keeps rediscovering (htccl
    ``_get_out_buf``, ``GraphSharedOutput``, ``_DEQUANT_WS``), where the
    ordering rule that makes reuse safe survives only as a comment. Here two
    captured buckets of the same layer cannot alias, by construction rather
    than by discipline, and ``test_breakable_route_462`` pins it.
    """

    buf: Any
    stage: Any = None


class BreakableOffloadArena:
    """Per-layer registry of the bridge buffers the captured segments read.

    The SLOT ARENA itself is not allocated here -- it already exists, and that
    is the point. ``MoEExpertOffloadCache.install()`` builds the ``[R+C]``-slot
    device buffer per expert tensor and binds it into the layer's parameters,
    so the captured compute addresses slots the moment it is captured. This
    class owns only the second half of the contract: the per-shape buffer that
    tells the captured kernels which expert is in which slot this step.

    #286 CONSUMPTION (asset class + capture refusal). The arena's payload is the
    ``experts`` offload class; :meth:`park` routes through the register's
    ``refuse_if_capture_active`` rather than inventing a local rule, because
    #452 settled that page movement belongs BETWEEN replays.

    OPEN FINDING, pinned as a test rather than fixed here (the register module
    is not this ticket's territory): the ``experts`` descriptor declares
    ``va_stable_required=False``, which is true of the eager offload -- the host
    pool is the source of truth and the VRAM copy is a droppable cache. Under
    THIS route it is not true: a captured graph holds the arena's device
    addresses, so moving the arena while a graph family addresses it invalidates
    the capture. The refusal below enforces the stronger rule locally; whether
    the descriptor should carry a per-route VA-stability qualifier is left to
    the #286 owner. See ``docs/dev/DESIGN_462_breakable_route.md`` §6.
    """

    def __init__(self, layer_id: Any = None) -> None:
        self.layer_id = layer_id
        self._bridges: Dict[Tuple[Tuple[int, ...], Any], BreakableBridge] = {}
        # Consume the #286 descriptor rather than restating what the class is.
        # An unknown class raises at import there, so this doubles as the
        # assertion that the arena's asset class still exists.
        from sglang.srt.model_executor.short_term_offload_register import (
            describe_class,
        )

        self.asset_class = describe_class(ARENA_OFFLOAD_CLASS)

    @property
    def shapes(self) -> Tuple[Tuple[int, ...], ...]:
        """Captured shapes this arena has a bridge for (test/diagnostic)."""
        return tuple(shape for shape, _ in self._bridges)

    def bridge_for(self, topk_ids) -> BreakableBridge:
        """The bridge for this capture shape, allocating it on first sight.

        Allocation happens in the eager break region -- the previous segment
        has ended and the next has not begun -- which is the one window where a
        fresh allocation is not captured into a graph mempool and therefore
        keeps a stable address for the graph's whole life. On every later call
        (each replay, and each further capture of the same shape) this is a dict
        hit and returns the SAME buffers, which is what makes the address the
        captured kernels baked in still correct.
        """
        import torch

        key = (tuple(topk_ids.shape), topk_ids.dtype)
        bridge = self._bridges.get(key)
        if bridge is not None:
            return bridge

        buf = torch.empty_like(topk_ids)
        stage = None
        if buf.is_cuda:
            stage = torch.empty(
                buf.numel(), dtype=buf.dtype, device="cpu"
            ).pin_memory()
        bridge = BreakableBridge(buf=buf, stage=stage)
        self._bridges[key] = bridge
        logger.debug(
            "MoE breakable arena layer %s: bridge allocated for shape %s (%s)",
            self.layer_id,
            key[0],
            key[1],
        )
        return bridge

    def park(self, target: Optional[str] = None) -> None:
        """Refuse to move the arena while a capture is active (#286 gate).

        The register owns this rule; calling its gate is the whole point of
        having one. A park unmaps physical pages, which is eager work, and the
        captured segments hold this arena's addresses.
        """
        from sglang.srt.model_executor.short_term_offload_register import (
            refuse_if_capture_active,
        )

        refuse_if_capture_active(
            "park",
            f"MoE breakable slot arena (layer {self.layer_id})",
            where=f"target={target}" if target else "",
        )
        raise BreakableModeRefused(
            reason=(
                f"the slot arena of layer {self.layer_id} is addressed by "
                f"captured decode graphs and has no park path while they exist"
            ),
            remedy=(
                "Drop the decode graphs first (the #286 rung-1 family eviction), "
                "then park. Parking the arena under a live capture would "
                "invalidate every graph that baked in its addresses."
            ),
        )


def _moe_offload_fetch_step(cache, arena: BreakableOffloadArena, topk_ids):
    """The eager pre-replay phase, as ONE decorated callable.

    Everything data-dependent lives in here and nowhere else, which is what
    makes the split clean: ``eager_on_graph`` re-runs exactly this on every
    replay, and the captured segment that follows contains no host dependency
    at all.

    Returns ``None`` on purpose. ``eager_on_graph`` copies a returned tensor
    into the bridge buffer it captured on the first pass; here the callable has
    ALREADY written the static buffers in place, so returning the bridge would
    make the decorator issue a redundant self-copy on every replay. The caller
    reads the bridge back off the arena instead, by the same key.
    """
    bridge = arena.bridge_for(topk_ids)
    cache.prepare_breakable(topk_ids, bridge.buf, bridge.stage)
    return None


#: The break point. Decorated once at module scope, matching the convention of
#: the other break points in the tree (``breakable_unified_attention_with_output``,
#: ``bcg_deepseek_v4_attention_with_output``, ...).
breakable_moe_offload_fetch = eager_on_graph(True)(_moe_offload_fetch_step)


__all__ = [
    "ARENA_OFFLOAD_CLASS",
    "BreakableBridge",
    "BreakableModeRefused",
    "BreakableOffloadArena",
    "BreakableScratchOverflow",
    "EAGER_HOST_BLOCKING_CROSSINGS_PER_LAYER_PER_STEP",
    "ENV_GRAPH_MODE",
    "HOST_BLOCKING_CROSSINGS_PER_LAYER_PER_STEP",
    "HOST_SYNCS_PER_LAYER_PER_STEP",
    "MODES",
    "MODE_BREAKABLE",
    "MODE_CAPTURABLE",
    "MODE_EAGER",
    "breakable_moe_offload_fetch",
    "breakable_opt_in",
    "validate_breakable_boot",
]
