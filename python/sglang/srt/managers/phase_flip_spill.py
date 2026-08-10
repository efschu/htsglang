"""#631 SPILL-DEPTH LADDER -- the TP-decode phase's exclusive assets leave
VRAM while the PP prefill phase owns the card.

WHY THIS EXISTS
---------------
The flip already treats two asset classes as phase-exclusive: the model
weights (one arena sized ``max(pp_bytes, tp_bytes)``, refilled per flip from
pinned host images) and the KV backing (``phase_flip_boot`` releases the PP
backing so the boot peak is ``max(PP, TP)``, not ``PP + TP``).

The speculative drafter is NOT in that set. Its weights (1.86-2.01 GB/rank
here) and its captured CUDA graphs (~0.55 GB/rank) are loaded once at boot
and stay resident in BOTH phases -- while the PP phase has no draft worker
at all (``build_flip_draft_worker`` returns None there, and the cutover
documents the PP phase as "bit-for-bit the state an instance without
speculation has"). In the PP phase those bytes are provably unreachable.
That is the strongest possible precondition for a spill: there is no
correctness question left, only a cost question.

The boot comment that says the draft weights "stay resident across both
phases ... there is no second layout for them to flip between" explains why
they were never ARENA-backed. It is not an argument that they must stay
resident -- a spill has no second layout either; it has a host image and an
empty device.

THE LADDER IS USER-SELECTABLE, and cumulative
---------------------------------------------
``--phase-flip-spill-depth {none,cache,draft,draft+graphs}`` (integers 0..3
also accepted) / ``SGLANG_PHASE_FLIP_SPILL_DEPTH``.

    0  none          nothing given up -- the pre-#656 seam, byte-identical
    1  cache         the outgoing phase's cached allocator segments go back
                     to the driver at the cutover      measured 2.5-3.5 GiB/card
    2  draft         + draft (MTP) weights             ~1.86-2.01 GB/rank
    3  draft+graphs  + draft CUDA graphs               ~0.55 GB/rank

Cumulative: depth N performs every rung up to N. Each rung buys corridor and
costs flip milliseconds; the measured trade is recorded per rung in
``docs/dev/631/PROD_BRINGUP_BENCH.md``. Higher flip time is an accepted price
per the ordering user -- but it IS a price, so it is measured and published
per rung rather than assumed small.

RUNG 1 IS THE DEFAULT under ``--enable-phase-flip``, and rung 1 only.
``IMPLEMENTED_DEPTH`` gates the rest: rungs 2 and 3 parse and are then
REFUSED, because the TP decode CUDA graphs bake the draft weights' addresses
and the restore below allocates a FRESH arena, which moves them. That is a
latent graph-corruption bug in this module as originally written, not a
missing feature -- wiring it as-is would have been worse than leaving it
dead. Those rungs need a VA-stable carrier (``KvVmmArena``, whose
commit/decommit the KV pools already use with addresses held fixed) before
they can be turned on.

THE PRIMITIVE IS NOT NEW
------------------------
``weights_arena`` already implements exactly this pair, and the boot path
already uses it on the two model layouts:

    plan_arena_layout -> image_from_tensors(pin=True) -> free the originals
    allocate_arena -> bind_arena_views -> arena_refill (verifies on device)

This module applies that pair to the drafter at the flip seam instead of to
the model layouts at boot. No new memory primitive is introduced.

THE IMMUTABILITY ASSUMPTION, named out loud
-------------------------------------------
The host image is built ONCE, on the first spill, and re-used by every
later restore. That is only correct if nothing writes to the draft weights
between a restore and the next spill -- true for inference, and exactly the
assumption the boot images ``image_pp`` / ``image_tp`` already make (they
are built once at boot and refilled at every flip forever).

Unlike those, this one ships with a FALSIFIER:
``SGLANG_PHASE_FLIP_SPILL_VERIFY=1`` re-checksums the live device bytes
before each spill and raises on a mismatch. Run it once on metal to prove
the assumption, then leave it off -- it costs a device-side reduction over
~2 GB per flip.

ORDERING LAW: SPILL ONLY AFTER THE CUTOVER HAS COMMITTED
--------------------------------------------------------
Never spill on a merely ARMED flip. An abandoned flip that had already
freed the draft weights would return to the TP phase with no drafter and
0-sized parameter placeholders -- a loud crash at best.

CORRECTION (#656 successor 21): the sentence that used to stand here claimed
"both call sites in ``phase_flip_runtime._cutover`` step 7b sit after the
active-stack swap". THAT WIRING DOES NOT EXIST AND NEVER DID -- a whole-tree
grep for ``get_spill_ladder``/``on_enter_pp``/``on_enter_tp`` matched only
this file. The draft rungs are unreferenced code with no tests. The statement
is left visible rather than deleted because a docstring asserting a call site
that is absent is exactly the kind of claim that let five successors believe
spec item 6 was implemented.

Rung 1 IS wired, at ``phase_flip_runtime._build_kv_backing_swap``, strictly
between the source pool's release and the destination's restore, and it is
covered by ``test_phase_flip_spill_depth_631.py``.

The restore leg runs BEFORE ``arm_draft_bootstrap_all_reachable``, which
needs a live drafter to scrub its pool.
"""

from __future__ import annotations

import logging
import os
from typing import Any, Dict, Optional

import torch

from sglang.srt.model_executor.weights_arena import (
    allocate_arena,
    arena_refill,
    bind_arena_views,
    image_from_tensors,
    plan_arena_layout,
    uint8_checksum,
)

logger = logging.getLogger(__name__)

LOG_PREFIX = "PHASE-FLIP-SPILL"

# The ladder is CUMULATIVE: depth N performs every rung up to and including
# N. Rung 1 was added by #656 successor 21 and deliberately sits BELOW the
# draft rungs, because it is the only rung whose worth was measured before it
# was written and because it is the one that matches the asset the corridor
# actually loses.
#
# WHY RUNG 1 IS FIRST, in the rig's own numbers. The corridor law is a
# CONTINUOUS minimum of NVML free memory, and NVML counts torch's cached-but-
# unused segments as USED. A prefill grows the caching allocator's reserve in
# proportion to the sequence length -- measured on this rig at ~19-26 MiB per
# 1000 prompt tokens per card -- and the allocator never returns those blocks
# to the driver on its own. So the corridor decays in steps, one step per
# longest-prefill-so-far, and never recovers. Measured 2026-08-10 on the live
# instance: free had decayed to 3911/4392/2911 MiB, and asking the allocator
# to return its cached segments restored 6605/7846/5405 MiB -- 2.5 to 3.5 GiB
# per card that no phase was using and no boot-time itemisation could see.
#
# The seam is the correct instant to ask, and not merely a convenient one:
# the outgoing layout's scratch is dead there by construction, the source
# pool's physical pages have just been handed back, and the destination's
# re-commit is about to ask the driver for RAW pages -- the exact allocation
# that has failed here before for want of them.
DEPTH_NONE = 0
DEPTH_ALLOCATOR_CACHE = 1
DEPTH_DRAFT_WEIGHTS = 2
DEPTH_DRAFT_GRAPHS = 3
MAX_DEPTH = DEPTH_DRAFT_GRAPHS

#: Spelling accepted by the CLI and the environment, in ladder order. The
#: integer form stays valid so an A/B can sweep the ladder numerically.
DEPTH_NAMES = {
    "none": DEPTH_NONE,
    "cache": DEPTH_ALLOCATOR_CACHE,
    "draft": DEPTH_DRAFT_WEIGHTS,
    "draft+graphs": DEPTH_DRAFT_GRAPHS,
}

#: The deepest rung that is WIRED AND EXERCISED. Anything above this parses
#: but is refused at resolution time rather than silently behaving as this
#: value -- a depth dial that quietly under-delivers is worse than one that
#: says no, because the A/B that compares two rungs would then compare a rung
#: against itself and report a real-looking zero.
IMPLEMENTED_DEPTH = DEPTH_ALLOCATOR_CACHE

DEPTH_ENV = "SGLANG_PHASE_FLIP_SPILL_DEPTH"
VERIFY_ENV = "SGLANG_PHASE_FLIP_SPILL_VERIFY"

_MIB = 1048576.0


class PhaseFlipSpillError(RuntimeError):
    """A spill/restore invariant broke.

    Raised rather than degraded: every failure mode here ends in a drafter
    whose parameters are 0-sized placeholders, and a forward through those
    is a far worse diagnostic than this exception.
    """


def resolve_spill_depth(server_args: Any = None) -> int:
    """The configured ladder depth, 0 when unset.

    An explicitly set server arg wins over the environment; the env
    fallback exists so an A/B does not have to edit the boot script.
    """
    depth = None
    if server_args is not None:
        depth = getattr(server_args, "phase_flip_spill_depth", None)
    if depth is None:
        raw = os.environ.get(DEPTH_ENV)
        if raw not in (None, ""):
            depth = raw
    if depth in (None, ""):
        return DEPTH_NONE
    if isinstance(depth, str) and depth.strip().lower() in DEPTH_NAMES:
        value = DEPTH_NAMES[depth.strip().lower()]
    else:
        try:
            value = int(depth)
        except (TypeError, ValueError):
            names = ", ".join(DEPTH_NAMES)
            raise PhaseFlipSpillError(
                f"{LOG_PREFIX} spill depth {depth!r} is neither an integer "
                f"0..{MAX_DEPTH} nor one of: {names}"
            )
    if not 0 <= value <= MAX_DEPTH:
        raise PhaseFlipSpillError(
            f"{LOG_PREFIX} spill depth {value} out of range; valid depths "
            f"are 0..{MAX_DEPTH} (cumulative: 2 implies 1)"
        )
    if value > IMPLEMENTED_DEPTH:
        # Refuse rather than clamp. Clamping would make a depth sweep report
        # that rung 3 is worth exactly what rung 1 is worth, which reads as a
        # measurement and is an artefact of the clamp.
        raise PhaseFlipSpillError(
            f"{LOG_PREFIX} spill depth {value} is defined but not wired; the "
            f"deepest implemented rung is {IMPLEMENTED_DEPTH} "
            f"('cache': return the outgoing phase's cached allocator segments "
            f"to the driver at the cutover). Rungs {IMPLEMENTED_DEPTH + 1}.."
            f"{MAX_DEPTH} spill the DRAFT model, whose weights the TP decode "
            f"CUDA graphs bake addresses into, so they need a VA-stable "
            f"carrier that this module does not yet have -- restoring into a "
            f"freshly allocated arena would move those addresses and corrupt "
            f"the graphs."
        )
    return value


def spill_verify_enabled() -> bool:
    return os.environ.get(VERIFY_ENV, "") not in ("", "0", "false", "False")


def release_allocator_cache(
    direction: str, *, depth: int, device_index: Any = None
) -> int:
    """Rung 1: hand the outgoing phase's cached segments back to the driver.

    Returns the number of bytes NVML reports as newly free, or 0 when the rung
    is not selected or CUDA is unavailable.

    WHAT THIS DOES NOT DO, stated because the difference decides what the rung
    is worth. It does not reduce the PEAK. The transient a long prefill needs
    while it runs is live memory, measured on this rig at ~30 MiB per 1000
    prompt tokens per card, and no allocator call can give that back while the
    prefill is in flight. What this returns is the RESIDUE: the segments the
    allocator kept after the request finished, which survive every subsequent
    shorter request and turn a one-off long prefill into a permanent corridor
    loss. Those two quantities were measured separately (a ladder run with and
    without a release between rungs) precisely so that this rung is not
    credited with the peak it cannot touch.

    The free-byte delta is read from the driver rather than from torch's own
    counters on purpose: the corridor law the user set is stated in the FREE
    column, and torch's ``reserved`` figure does not mean the same thing --
    under ``expandable_segments:True`` it counts a VIRTUAL extent and was
    observed at 36910 MiB on a 32607 MiB card, so it cannot be compared to a
    physical budget at all.

    ``device_index`` names the rank's own device explicitly. This is belt and
    braces rather than a bug fix, and the story of why it was written is worth
    keeping, because the reasoning that produced it was wrong.

    The figures this rung logs looked impossible next to ``nvidia-smi``: it
    reported 10371 MiB free on a card the driver showed with 3149. The
    conclusion drawn was "it is reading the wrong device", on the theory that
    every worker sees all three cards. **That theory was false in both
    halves.** ``--rank-gpu-id`` gives each worker process a
    ``CUDA_VISIBLE_DEVICES`` holding exactly ONE physical GPU, so device 0 is
    unambiguously this rank's card in every worker. And the two numbers were
    never comparable: this rung runs at the SEAM, immediately after the source
    pool handed its physical pages back, so free memory is legitimately some
    gigabytes higher there than in either phase's steady state. At pool 600000
    the source pool is ~8 GiB on rank 0, and the seam census independently
    records a seam maximum of 10974 MiB on that card.

    **Comparing an instrument's reading against a different instant is not a
    cross-check.** The device is still named explicitly because an absolute
    memory figure should say which card it is about, but do not read the
    argument as evidence that a bare call was reading the wrong one.
    """
    if depth < DEPTH_ALLOCATOR_CACHE:
        return 0
    try:
        import torch

        if not torch.cuda.is_available():
            return 0
        dev = device_index if device_index is not None else torch.cuda.current_device()
        free_before, _ = torch.cuda.mem_get_info(dev)
        torch.cuda.empty_cache()
        free_after, _ = torch.cuda.mem_get_info(dev)
    except Exception as e:  # pragma: no cover - driver/platform differences
        # A failed reclaim must never take the flip down: the flip is correct
        # without it, only tighter.
        logger.warning("%s allocator cache release failed: %s", LOG_PREFIX, e)
        return 0
    returned = int(free_after) - int(free_before)
    logger.info(
        "%s rung 1 (cache) at %s on device %s: returned %.1f MiB of cached "
        "segments to the driver; free %.1f -> %.1f MiB. This is residue, not "
        "peak.",
        LOG_PREFIX,
        direction,
        dev,
        returned / _MIB,
        free_before / _MIB,
        free_after / _MIB,
    )
    return returned


def draft_model_of(draft_worker: Any) -> Optional[Any]:
    """The drafter's ``torch.nn.Module``, or None.

    ``draft_worker`` is the ``EAGLEWorkerV2`` the cutover arms; its
    ``.draft_worker`` is the ``EagleDraftWorker`` and ``.draft_runner``
    that worker's ``ModelRunner``. Reached defensively at every hop: a
    phase-flip instance may be built with no speculation at all, in which
    case every rung must be a silent no-op and not an AttributeError
    inside the cutover's no-return region.
    """
    if draft_worker is None:
        return None
    inner = getattr(draft_worker, "draft_worker", None)
    runner = getattr(inner, "draft_runner", None)
    if runner is None:
        runner = getattr(inner, "model_runner", None)
    return getattr(runner, "model", None)


class DraftWeightSpill:
    """RUNG 1: the draft model's checkpoint parameters, out of VRAM for the
    duration of the PP phase.

    Lifecycle, and note that the FIRST spill is the only one that pays a
    device-to-host copy:

        spill()   #1  build pinned host image from the live originals,
                      rebind every param to a 0-sized placeholder, drop the
                      original storages.
        restore()     allocate ONE contiguous device arena, bind views,
                      one H2D refill, checksum verified on the device.
        spill()   #2+ drop the arena; the host image is already correct.

    After the first restore the parameters live in this module's arena and
    the boot-time storages are gone for good, which is incidentally a
    defragmentation win: ~2 GB of scattered per-tensor storages become one
    block that is allocated and released whole.
    """

    def __init__(self, model: Any) -> None:
        from sglang.srt.managers.phase_flip_boot import checkpoint_param_dict

        self._named = checkpoint_param_dict(model)
        if not self._named:
            raise PhaseFlipSpillError(
                f"{LOG_PREFIX} the draft model exposes no checkpoint "
                f"parameters; refusing to install a spill that would free "
                f"nothing and hide a wrong model handle"
            )
        self._layout = plan_arena_layout(self._named)
        self._image: Optional[torch.Tensor] = None
        self._arena: Optional[torch.Tensor] = None
        self._device = next(iter(self._named.values())).device
        self._spilled = False

    @property
    def spilled(self) -> bool:
        return self._spilled

    @property
    def payload_mib(self) -> float:
        return self._layout.total_bytes / _MIB

    def _checksum_live(self) -> int:
        """Checksum of the live device bytes in layout order.

        Only reachable under SGLANG_PHASE_FLIP_SPILL_VERIFY -- it is the
        falsifier for this module's immutability assumption, not a
        steady-state cost.
        """
        total = 0
        for slot in self._layout.slots:
            t = self._named[slot.name].data
            flat = t.reshape(-1).view(torch.uint8)
            total = (total + int(uint8_checksum(flat))) & 0xFFFFFFFFFFFFFFFF
        return total

    def spill(self) -> float:
        """Free the draft weights from the device. Returns MiB released."""
        if self._spilled:
            return 0.0
        verify = spill_verify_enabled()
        live_sum = self._checksum_live() if verify else None
        if self._image is None:
            # First spill: the originals are the only copy of the bytes.
            self._image = image_from_tensors(
                self._named, self._layout, pin=True
            )
            self._baseline_sum = live_sum
        elif verify and live_sum != getattr(self, "_baseline_sum", None):
            raise PhaseFlipSpillError(
                f"{LOG_PREFIX} the draft weights CHANGED between restore "
                f"and spill (checksum {live_sum} vs {self._baseline_sum}); "
                f"the re-used host image would silently revert them. This "
                f"module's immutability assumption is falsified -- rebuild "
                f"the image on every spill before shipping depth>=1."
            )
        for name, param in self._named.items():
            param.data = torch.empty(0, dtype=param.dtype, device=param.device)
        self._arena = None
        self._spilled = True
        released = self.payload_mib
        logger.info(
            "%s rung 1 SPILLED the draft weights: %d params, %.1f MiB "
            "released to the allocator (PP phase has no drafter)",
            LOG_PREFIX,
            len(self._named),
            released,
        )
        return released

    def restore(self) -> float:
        """Bring the draft weights back. Returns MiB re-materialized."""
        if not self._spilled:
            return 0.0
        if self._image is None:  # pragma: no cover - defensive
            raise PhaseFlipSpillError(
                f"{LOG_PREFIX} spilled with no host image; the draft "
                f"weights are unrecoverable"
            )
        self._arena = allocate_arena(self._layout.total_bytes, self._device)
        bind_arena_views(
            self._layout, self._arena, rebind=list(self._named.items())
        )
        arena_refill(self._arena, self._layout, self._image)
        self._spilled = False
        logger.info(
            "%s rung 1 RESTORED the draft weights: %.1f MiB refilled from "
            "the pinned host image, checksum verified on device",
            LOG_PREFIX,
            self.payload_mib,
        )
        return self.payload_mib


class PhaseFlipSpillLadder:
    """The depth-selected set of rungs, driven by the two cutover legs.

    Built lazily on first use so that an instance whose depth is 0 -- the
    default -- never touches the drafter at all, and so that an instance
    without speculation is a no-op rather than a failure.
    """

    def __init__(self, depth: int) -> None:
        self.depth = int(depth)
        self._weights: Optional[DraftWeightSpill] = None
        self._installed = False
        self._install_failed = False

    def _install(self, draft_worker: Any) -> None:
        if self._installed or self._install_failed:
            return
        model = draft_model_of(draft_worker)
        if model is None:
            # No speculation on this instance: every rung is a no-op. Say
            # it once, then stay quiet.
            logger.info(
                "%s depth=%d configured but this instance has no draft "
                "model; the ladder is a no-op",
                LOG_PREFIX,
                self.depth,
            )
            self._install_failed = True
            return
        self._weights = DraftWeightSpill(model)
        self._installed = True
        logger.info(
            "%s installed depth=%d, rung 1 payload %.1f MiB/rank",
            LOG_PREFIX,
            self.depth,
            self._weights.payload_mib,
        )

    def on_enter_pp(self, draft_worker: Any) -> float:
        """tp->pp leg, AFTER the cutover committed. Returns MiB released."""
        if self.depth < DEPTH_DRAFT_WEIGHTS:
            return 0.0
        self._install(draft_worker)
        if self._weights is None:
            return 0.0
        return self._weights.spill()

    def on_enter_tp(self, draft_worker: Any) -> float:
        """pp->tp leg, BEFORE the draft bootstrap. Returns MiB restored."""
        if self.depth < DEPTH_DRAFT_WEIGHTS:
            return 0.0
        self._install(draft_worker)
        if self._weights is None:
            return 0.0
        return self._weights.restore()

    def stats(self) -> Dict[str, Any]:
        return {
            "depth": self.depth,
            "installed": self._installed,
            "spilled": bool(self._weights and self._weights.spilled),
            "payload_mib": (
                round(self._weights.payload_mib, 1) if self._weights else 0.0
            ),
        }


def get_spill_ladder(scheduler: Any) -> Optional[PhaseFlipSpillLadder]:
    """The scheduler's ladder, built once, or None at depth 0.

    Cached on the scheduler because the host image must survive every flip
    -- a per-flip ladder would rebuild the pinned image on each seam and
    turn a 0-cost spill into a multi-GB device-to-host copy.
    """
    ladder = getattr(scheduler, "phase_flip_spill_ladder", None)
    if ladder is not None:
        return ladder
    depth = resolve_spill_depth(getattr(scheduler, "server_args", None))
    if depth <= DEPTH_NONE:
        return None
    ladder = PhaseFlipSpillLadder(depth)
    scheduler.phase_flip_spill_ladder = ladder
    return ladder
