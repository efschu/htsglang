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

RUNG 1 IS THE DEFAULT under ``--enable-phase-flip``. RUNG 2 IS IMPLEMENTED
(#656 successor 29) and must be asked for explicitly; rung 3 still parses and
is then REFUSED.

Rung 2 was refused for four shifts because the TP decode CUDA graphs bake the
draft weights' addresses and the restore allocated a FRESH arena, which moves
them -- a latent graph-corruption bug, not a missing feature. The fix is not
to stop capturing graphs (spec item 8 measured that at 41% of decode
throughput) but to make the ADDRESS survive the release:
``VmmDraftWeightCarrier`` puts the weights on a ``KvVmmArena`` reservation, so
a spill unmaps physical pages while the virtual addresses stand still. The
graphs stay valid because nothing they point at ever moved.

Rung 3 (the draft CUDA graphs themselves) is a genuinely different trade: a
captured graph cannot be refilled from a host image, it must be re-CAPTURED,
so rung 3 pays a capture per flip for graphs item 8 proved are worth 41% of
decode. It needs its own measurement before it is offered.

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
The host image is built ONCE, on the first spill, and reused by every
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

HISTORY, kept because it is the reason to distrust docstrings here.
Successor 21 found that the sentence which used to stand in this place --
"both call sites in ``phase_flip_runtime._cutover`` step 7b sit after the
active-stack swap" -- described wiring that DID NOT EXIST AND NEVER HAD. A
whole-tree grep for ``get_spill_ladder``/``on_enter_pp``/``on_enter_tp``
matched only this file. That false claim let five successors believe spec
item 6 was implemented.

IT IS WIRED NOW (#656 successor 29), and here is where, so the claim is
checkable:

    rung 1  ``WavedBackingSwap.reclaim_between``, called once per flip from
            ``PhaseFlipRuntime._execute``.
    rung 2  SPILL  ``_cutover``'s PP branch, after the active-stack swap has
                   installed ``scheduler.draft_worker = None``.
            RESTORE ``_cutover``'s TP branch, immediately before
                   ``arm_draft_bootstrap_all_reachable``, which needs a live
                   drafter to scrub its pool.
    gate    ``PhaseFlipRuntime._staging_bytes`` adds the pending restore to
            the pp->tp affordability verdict, so the commit that could fail
            inside the no-return region is priced before the flip commits.

Both rung-2 legs sit AFTER the abandon decision, which is what the ordering
law below requires.
"""

from __future__ import annotations

import logging
import os
from typing import Any, Dict, Optional

import torch

from sglang.srt.model_executor.weights_arena import (
    arena_refill,
    bind_arena_views,
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
# RUNG 3 (#656 successor 30): the weights arena's tail, idle in whichever
# phase has the smaller layout -- TP on all three ranks of this rig, which is
# also the phase that binds after rung 2. Inserted BELOW draft+graphs rather
# than renumbering "draft", so the integer form keeps its meaning in every
# piece of evidence recorded so far.
DEPTH_ARENA_TAIL = 3
DEPTH_DRAFT_GRAPHS = 4
MAX_DEPTH = DEPTH_DRAFT_GRAPHS

#: Spelling accepted by the CLI and the environment, in ladder order. The
#: integer form stays valid so an A/B can sweep the ladder numerically.
DEPTH_NAMES = {
    "none": DEPTH_NONE,
    "cache": DEPTH_ALLOCATOR_CACHE,
    "draft": DEPTH_DRAFT_WEIGHTS,
    "arena": DEPTH_ARENA_TAIL,
    "draft+graphs": DEPTH_DRAFT_GRAPHS,
}

#: The deepest rung that is WIRED AND EXERCISED. Anything above this parses
#: but is refused at resolution time rather than silently behaving as this
#: value -- a depth dial that quietly under-delivers is worse than one that
#: says no, because the A/B that compares two rungs would then compare a rung
#: against itself and report a real-looking zero.
#
# RAISED TO 2 BY #656 successor 29, and the reason the refusal could be
# lifted is the carrier, not a decision to tolerate the hazard. The refusal
# text below used to say the draft weights need "a VA-stable carrier that
# this module does not yet have". ``VmmDraftWeightCarrier`` is that carrier:
# the weights live on a ``KvVmmArena`` reservation whose virtual address is
# taken once at boot and freed only at close, so a spill releases PHYSICAL
# pages (``cuMemUnmap`` + ``cuMemRelease``) while every ``data_ptr()`` the
# TP decode graphs baked stays exactly where it was. Rung 3 (the draft
# CUDA graphs themselves) is still unwired -- and note that spec item 8 is
# now answered AGAINST removing those graphs, so rung 3 buys a phase-local
# spill of something the next TP phase must re-capture, which is a
# different and much worse trade than rung 2.
IMPLEMENTED_DEPTH = DEPTH_ARENA_TAIL

#: Where the boot parks the carrier on the draft worker. An attribute on the
#: worker rather than an entry on the scheduler because BOTH flip hooks
#: already hold the draft worker and neither holds a spill-aware scheduler
#: field; adding one would have meant touching the stacks dataclass, the
#: factory and two call sites to carry a pointer that the object at hand
#: already knows.
CARRIER_ATTR = "_phase_flip_weight_carrier"

#: Physical handle size for the carrier's commits. The arena maps ONE
#: monolithic handle per extension when this is unset, and a single ~2 GiB
#: cuMemCreate is the allocation most likely to fail on a card that is by
#: construction nearly full at this instant. Chunking makes the restore a
#: sequence of independently-satisfiable asks.
CARRIER_COMMIT_CHUNK = 64 * 1024 * 1024

#: Slack added to the reservation on top of the aligned payload. The pool's
#: bump allocator is granularity-aligned and torch rounds large-pool segments
#: up per tensor; one tensor is allocated here, so this is one rounding plus
#: a granule. VA costs nothing until committed.
CARRIER_VA_SLACK = 64 * 1024 * 1024

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
            f"('draft': the cached allocator segments AND the draft model's "
            f"weights, the latter on a VA-stable KvVmmArena carrier). Rung "
            f"{MAX_DEPTH} ('draft+graphs') additionally spills the draft CUDA "
            f"GRAPHS, which is not wired: a captured graph cannot be released "
            f"and re-materialised from a host image the way a weight tensor "
            f"can -- it would have to be re-CAPTURED on every pp->tp flip, and "
            f"#656 spec item 8 measured those graphs as worth 41% of decode "
            f"throughput, so paying a re-capture per flip is a different trade "
            f"from rung 2 and needs its own measurement before it is offered."
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


def allocate_carrier_tensor(arena: Any, nbytes: int, device_index: int):
    """A flat uint8 tensor occupying ``nbytes`` of ``arena``'s reservation.

    THE SPAN IS THE EXTENSION POINT. #656 spec items 11-14 make residency a
    function of (phase, load) rather than a boot constant, and the payload
    classes queued behind the drafter -- idle-slot GDN/mamba states, the cold
    layout's bytes, session KV -- are further rungs of THIS ladder, not new
    machinery. They differ only in which bytes they carry; the VA-stable
    mechanics (reserve once, commit/decommit underneath, refill from a host
    image) are identical. So the arena is injectable and the allocation is a
    function rather than an inlined ``use_mem_pool`` block: a payload class
    that already owns a span hands it in, and a test hands in a host-memory
    stand-in.

    An arena exposing ``allocate_carrier`` owns the allocation itself.
    Otherwise this takes the ``KvVmmArena`` path: allocate from its MemPool,
    whose pluggable allocator is a granularity-aligned bump pointer over the
    reservation and returns an ADDRESS ONLY -- no pages are mapped until
    ``commit_range``.
    """
    hook = getattr(arena, "allocate_carrier", None)
    if hook is not None:
        return hook(nbytes)
    with torch.cuda.use_mem_pool(arena.pool):
        return torch.empty(
            int(nbytes), dtype=torch.uint8, device=f"cuda:{int(device_index)}"
        )


class VmmDraftWeightCarrier:
    """RUNG 2: the draft model's checkpoint parameters, physically absent for
    the duration of the PP phase, at a virtual address that never moves.

    WHY A VMM ARENA AND NOT A torch ALLOCATION
    ------------------------------------------
    The version of this class that shipped dead in the tree freed the
    parameters by rebinding them to 0-sized placeholders and restored them
    into a FRESHLY allocated arena. That is correct for the bytes and fatal
    for the graphs: the TP decode CUDA graphs bake the drafter's parameter
    addresses at capture, and a fresh allocation lands wherever the caching
    allocator pleases. The refusal in ``resolve_spill_depth`` existed to keep
    that bug unreachable.

    ``KvVmmArena`` splits the two things a normal allocation fuses. The
    VIRTUAL range is reserved once (``cuMemAddressReserve``) at boot and
    released only at ``close()``; the PHYSICAL pages behind it are mapped and
    unmapped freely underneath (``cuMemMap`` / ``cuMemUnmap`` +
    ``cuMemRelease``). So:

        spill    = decommit_range(offset, 0)   pages go back to the DRIVER,
                                               NVML free rises, data_ptr()s
                                               are byte-for-byte unchanged
        restore  = commit_range(offset, n)     fresh pages behind the SAME
                   + arena_refill              addresses, then one H2D copy
                                               with a device-side checksum

    Nothing is ever rebound. ``bind_arena_views`` runs exactly once, at boot,
    before graph capture, and the parameter views it installs stay valid for
    the life of the process. That is what makes "draft graphs stay ON" (spec
    item 8's measured verdict) and "the drafter spills" compatible at all.

    WHAT IS UNSAFE, SAID PLAINLY
    ----------------------------
    Between ``spill()`` and ``restore()`` the parameter tensors point at
    virtual addresses with NO physical backing. A read is not a wrong answer,
    it is a fault. This is sound only because the drafter is provably idle
    for the whole PP phase under strict purity -- decode, and therefore MTP,
    runs only in TP. It is NOT sound under threshold purity (spec item 10),
    where decode may continue in the PP layout, and ``install_draft_weight_
    carrier`` refuses that combination at boot rather than leaving a fault to
    be discovered at the first threshold decode.

    THE IMMUTABILITY ASSUMPTION, and its falsifier
    ----------------------------------------------
    The pinned host image is built ONCE, at boot, and every restore refills
    from it. Correct only if nothing writes the draft weights during a TP
    phase -- true for inference, and the same assumption the boot's own
    ``image_pp``/``image_tp`` already make. ``SGLANG_PHASE_FLIP_SPILL_VERIFY=1``
    re-checksums the live device bytes before each spill and raises on
    mismatch. It costs a device reduction over ~2 GB per flip; run it once to
    prove the assumption, then leave it off.
    """

    def __init__(
        self,
        model: Any,
        device_index: int,
        *,
        arena: Any = None,
    ) -> None:
        from sglang.srt.managers.phase_flip_boot import checkpoint_param_dict

        self._named = checkpoint_param_dict(model)
        if not self._named:
            raise PhaseFlipSpillError(
                f"{LOG_PREFIX} the draft model exposes no checkpoint "
                f"parameters; refusing to install a spill that would free "
                f"nothing and hide a wrong model handle"
            )
        # NOT EVERYTHING THE DRAFTER NAMES IS THE DRAFTER'S TO GIVE.
        #
        # Found on the first depth=draft boot, 2026-08-10 21:02:04Z: the boot
        # refused with
        #
        #   'lm_head.weight' views 848035840 bytes of a 14137090816-byte
        #   storage; a partial view would smuggle unowned bytes into the
        #   arena (V1 scope)
        #
        # 14137090816 B is 13481 MiB -- the TARGET model's weights arena. The
        # drafter does not own its lm_head, it VIEWS the target's. Moving that
        # onto the carrier would have re-pointed a slice of the target's arena,
        # and spilling it would have released pages the TARGET still reads,
        # during the phase where the target is the only thing running. That is
        # a corruption, not a capacity win, and plan_arena_layout's V1-scope
        # check is what caught it.
        #
        # So the payload is the parameters the drafter EXCLUSIVELY owns: those
        # whose storage is exactly their own bytes. A partial view means the
        # storage belongs to someone else (or is shared), and shared bytes are
        # by definition not phase-exclusive, which is the entire precondition
        # for spilling anything.
        #
        # CONSEQUENCE FOR THE PRICE. HANDOFF_671 costed this rung at 1925
        # MiB/rank from the boot's "Load weight end ... mem usage=" delta. That
        # delta includes the shared bytes. The real spillable payload is
        # smaller and is logged below per rank -- do not quote 1925 without
        # re-reading the installed figure.
        shared = {}
        exclusive = {}
        for name, p in self._named.items():
            t = p.data
            own = t.numel() * t.element_size()
            try:
                storage = t.untyped_storage().nbytes()
            except Exception:  # pragma: no cover - exotic tensor types
                storage = own
            if storage > own:
                shared[name] = (own, storage)
            else:
                exclusive[name] = p
        if shared:
            biggest = sorted(shared.items(), key=lambda kv: -kv[1][0])[:4]
            logger.info(
                "%s excluding %d draft parameter(s) that VIEW a larger "
                "storage and are therefore not the drafter's to release "
                "(%.1f MiB kept resident). Largest: %s",
                LOG_PREFIX,
                len(shared),
                sum(v[0] for v in shared.values()) / _MIB,
                ", ".join(
                    f"{n} ({own/_MIB:.0f} MiB of a {st/_MIB:.0f} MiB storage)"
                    for n, (own, st) in biggest
                ),
            )
        if not exclusive:
            raise PhaseFlipSpillError(
                f"{LOG_PREFIX} every draft parameter views a larger storage; "
                f"the drafter owns no exclusive bytes on this build, so there "
                f"is nothing this rung can release"
            )
        self._named = exclusive
        self._layout = plan_arena_layout(self._named)
        self._device_index = int(device_index)
        self._nbytes = int(self._layout.total_bytes)
        # The arena is created on device_index and the parameters are rebound
        # onto it. If they do not already live there, the rebind silently
        # MIGRATES the drafter to another card -- which on this rig means a
        # rank computing against another rank's GPU. Refuse instead: a wrong
        # device here is a configuration bug, not something to paper over.
        param_devices = {
            p.device.index for p in self._named.values() if p.device.type == "cuda"
        }
        if param_devices and param_devices != {self._device_index}:
            raise PhaseFlipSpillError(
                f"{LOG_PREFIX} the draft parameters live on cuda device(s) "
                f"{sorted(d for d in param_devices if d is not None)} but the "
                f"carrier was asked for device {self._device_index}; binding "
                f"them onto an arena on the wrong card would migrate the "
                f"drafter across GPUs"
            )
        self._spilled = False
        self._baseline_sum: Optional[int] = None

        if arena is None:
            from sglang.srt.mem_cache.kv_vmm_backing import KvVmmArena, align_up

            # Reserve only what this payload needs. A 256 GiB default
            # reservation is free in principle, but three of these across
            # three ranks in one process tree is a lot of address space to
            # take for no reason, and a tight reservation makes an
            # out-of-bounds commit fail loudly instead of scribbling into
            # unrelated slack.
            probe_gran = _probe_granularity(self._device_index)
            reserve = align_up(self._nbytes, probe_gran) + CARRIER_VA_SLACK
            arena = KvVmmArena(
                self._device_index,
                reserve_bytes=reserve,
                commit_chunk_bytes=CARRIER_COMMIT_CHUNK,
                # Retention would PARK the physical handles instead of
                # returning them to the driver -- it makes the restore
                # allocation-free and the spill worth exactly 0 MiB of
                # corridor, which is the entire point of this rung.
                retain_handles=False,
            )
        self._arena = arena

        # VA first, pages second. The pool's allocator is a bump allocator
        # over the reservation: this hands back an address, it does not back
        # it. Touching the tensor before commit_range is a fault, so the
        # order here is load-bearing.
        self._carrier = allocate_carrier_tensor(
            self._arena, self._nbytes, self._device_index
        )
        self._offset = int(self._carrier.data_ptr()) - int(self._arena.base)
        if self._offset < 0 or self._offset % self._arena.granularity != 0:
            raise PhaseFlipSpillError(
                f"{LOG_PREFIX} carrier landed at offset {self._offset}, which "
                f"is negative or not aligned to the arena granularity "
                f"{self._arena.granularity}; commit_range would refuse it"
            )
        self._arena.commit_range(self._offset, self._nbytes)

        # Snapshot the live weights to a pinned host image and free the
        # device originals BEFORE binding onto the carrier, so the peak is
        # max(originals, carrier) and not their sum -- the same reordering
        # the boot already applies to the model layouts.
        from sglang.srt.managers.phase_flip_boot import snapshot_and_free

        # Pinning is what makes the restore's H2D copy a DMA rather than a
        # staged copy, and on the Gen4 x4 rank that difference is most of the
        # restore's cost. It is also only meaningful with a GPU present:
        # pin_memory() raises without CUDA, which would make every CPU unit
        # test of this class fail for a reason unrelated to what it tests.
        pin = bool(param_devices)
        self._image = snapshot_and_free(self._named, self._layout, pin=pin)
        bind_arena_views(
            self._layout, self._carrier, rebind=list(self._named.items())
        )
        arena_refill(self._carrier, self._layout, self._image)
        logger.info(
            "%s carrier installed on device %d: %d params, %.1f MiB on a "
            "VA-stable reservation at 0x%x+0x%x (chunk %d MiB). Draft "
            "parameter addresses are now FIXED for the life of the process.",
            LOG_PREFIX,
            self._device_index,
            len(self._named),
            self.payload_mib,
            int(self._arena.base),
            self._offset,
            CARRIER_COMMIT_CHUNK // (1024 * 1024),
        )

    @property
    def spilled(self) -> bool:
        return self._spilled

    @property
    def payload_mib(self) -> float:
        return self._nbytes / _MIB

    @property
    def payload_bytes(self) -> int:
        return self._nbytes

    def param_ptrs(self) -> Dict[str, int]:
        """Current device address of every carried parameter.

        The boot assertion compares this before and after graph capture; a
        difference means something reallocated the drafter behind our back
        and the captured graphs are addressing freed memory.
        """
        return {name: int(p.data.data_ptr()) for name, p in self._named.items()}

    def contains_all_params(self) -> bool:
        """Every carried parameter lies inside the carrier's reservation."""
        lo = int(self._arena.base) + self._offset
        hi = lo + self._nbytes
        for p in self._named.values():
            ptr = int(p.data.data_ptr())
            if p.data.numel() == 0:
                return False
            if not (lo <= ptr < hi):
                return False
        return True

    def _checksum_live(self) -> int:
        """Checksum of the live device bytes in layout order.

        Only reachable under SGLANG_PHASE_FLIP_SPILL_VERIFY -- the falsifier
        for the immutability assumption, not a steady-state cost.
        """
        total = 0
        for slot in self._layout.slots:
            t = self._named[slot.name].data
            flat = t.reshape(-1).view(torch.uint8)
            total = (total + int(uint8_checksum(flat))) & 0xFFFFFFFFFFFFFFFF
        return total

    def spill(self) -> float:
        """Release the draft weights' physical pages. Returns MiB released.

        The virtual addresses survive; only the pages go back to the driver,
        which is what NVML's free column -- and therefore the corridor law --
        actually measures.
        """
        if self._spilled:
            return 0.0
        if spill_verify_enabled():
            live_sum = self._checksum_live()
            if self._baseline_sum is None:
                self._baseline_sum = live_sum
            elif live_sum != self._baseline_sum:
                raise PhaseFlipSpillError(
                    f"{LOG_PREFIX} the draft weights CHANGED between restore "
                    f"and spill (checksum {live_sum} vs {self._baseline_sum}); "
                    f"the reused host image would silently revert them. The "
                    f"immutability assumption is falsified -- rebuild the "
                    f"image on every spill before trusting depth>=2."
                )
        released = int(self._arena.decommit_range(self._offset, 0))
        self._spilled = True
        logger.info(
            "%s rung 2 SPILLED the draft weights on device %d: %.1f MiB of "
            "physical pages returned to the driver; %d parameter addresses "
            "UNCHANGED (PP phase has no drafter)",
            LOG_PREFIX,
            self._device_index,
            released / _MIB,
            len(self._named),
        )
        return released / _MIB

    def restore(self) -> float:
        """Re-back and refill the draft weights. Returns MiB re-materialized.

        This is the allocation that runs inside the flip's no-return region.
        Its bytes are priced into the staging affordability verdict before
        the flip commits (``phase_flip_runtime._staging_bytes``), so a rank
        that cannot afford them abandons unanimously and cheaply instead of
        raising ``cuMemCreate failed: CUDA_ERROR_OUT_OF_MEMORY`` here, where
        there is nothing left to do but die.
        """
        if not self._spilled:
            return 0.0
        self._arena.commit_range(self._offset, self._nbytes)
        arena_refill(self._carrier, self._layout, self._image)
        self._spilled = False
        logger.info(
            "%s rung 2 RESTORED the draft weights on device %d: %.1f MiB "
            "re-committed behind the SAME addresses and refilled from the "
            "pinned host image, checksum verified on device",
            LOG_PREFIX,
            self._device_index,
            self.payload_mib,
        )
        return self.payload_mib


def _probe_granularity(device_index: int) -> int:
    """Allocation granularity of a device, with a 2 MiB fallback.

    Split out so the size arithmetic is testable without a driver.
    """
    try:
        from sglang.srt.mem_cache.kv_vmm_backing import query_granularity

        return int(query_granularity(device_index))
    except Exception:  # pragma: no cover - no driver in unit tests
        return 2 * 1024 * 1024


def carrier_of(draft_worker: Any) -> Optional[VmmDraftWeightCarrier]:
    """The carrier the boot parked on this draft worker, or None."""
    if draft_worker is None:
        return None
    return getattr(draft_worker, CARRIER_ATTR, None)


def pending_restore_bytes(draft_worker: Any) -> int:
    """Device bytes a pp->tp flip must be able to commit for the drafter.

    Zero unless a carrier is installed AND currently spilled. Read by the
    affordability gate; deliberately total-payload rather than a fraction,
    because ``commit_range`` asks for the whole span in one call and a
    partially satisfied commit is not a state this design has.
    """
    carrier = carrier_of(draft_worker)
    if carrier is None or not carrier.spilled:
        return 0
    return int(carrier.payload_bytes)


def install_draft_weight_carrier(
    draft_worker: Any,
    device_index: int,
    *,
    server_args: Any = None,
    arena: Any = None,
) -> Optional[VmmDraftWeightCarrier]:
    """Move the drafter's weights onto a VA-stable carrier, at BOOT.

    MUST run after the draft worker is built and BEFORE its CUDA graphs are
    captured. Between those two points the parameter addresses are still
    free to move; after graph capture they are baked into the graphs and
    moving them is silent corruption rather than a crash.

    Returns None when there is nothing to carry (no speculation on this
    instance). Raises rather than degrading: a depth>=2 boot that silently
    came up without a carrier would spill nothing, measure as a flat zero,
    and read as "the rung is not worth anything".
    """
    if draft_worker is None:
        return None
    model = draft_model_of(draft_worker)
    if model is None:
        logger.info(
            "%s depth>=%d configured but this instance has no draft model; "
            "no carrier installed",
            LOG_PREFIX,
            DEPTH_DRAFT_WEIGHTS,
        )
        return None
    purity = getattr(server_args, "phase_flip_purity", "strict")
    if purity is not None and str(purity).strip().lower() != "strict":
        raise PhaseFlipSpillError(
            f"{LOG_PREFIX} spill depth >= {DEPTH_DRAFT_WEIGHTS} requires "
            f"--phase-flip-purity strict, got {purity!r}. The draft weights "
            f"are released for the whole PP phase, which is only sound "
            f"because strict purity forbids decode there. Under threshold "
            f"purity a PP-phase decode would touch unbacked virtual memory "
            f"and fault, so this combination is refused at boot rather than "
            f"at the first threshold decode."
        )
    carrier = VmmDraftWeightCarrier(model, device_index, arena=arena)
    setattr(draft_worker, CARRIER_ATTR, carrier)
    return carrier


class VmmWeightsArenaCarrier:
    """RUNG 3: the WEIGHTS arena's tail, physically absent in the phase whose
    layout does not need it.

    THE BYTES. The weights arena is one flat allocation sized
    ``max(layout_pp.total_bytes, layout_tp.total_bytes)`` because both layouts
    bind views into it and only one is live at a time. Each layout occupies a
    PREFIX ``[0, layout.total_bytes)``, so in the phase with the smaller
    layout the span above it is committed and addressable by nothing.
    Measured on the 2026-08-10 boot:

        rank  arena/pp     tp          tail
        PP0   13482.18     13163.45     318.7 MiB
        PP1    8144.00      7923.95     220.1 MiB
        PP2    9114.95      7923.95    1191.0 MiB

    ``pp`` is the max on every rank, so the tail is idle in **TP** -- which is
    the phase that binds on all three cards after rung 2 moved the binding
    phase there. That is what makes this rung well-aimed where the drafter no
    longer is.

    WHY IT IS THE CHEAPEST PROVIDER IN THE SYSTEM. There is no host round
    trip. The tail holds no live bytes in the phase it is released in, and the
    content of the whole arena is rewritten by ``arena_refill`` on the way
    back regardless -- a refill that already runs on every flip. So the spill
    is a ``decommit_range`` and the restore is a ``commit_range``, with no
    copy on either side. Compare the drafter, which must stage a pinned host
    image both ways.

    ORDERING IS LOAD-BEARING IN BOTH DIRECTIONS, and getting it backwards is
    silent corruption rather than a crash:

    * ``PP_TO_TP``: refill FIRST, release after. ``arena_refill``'s
      ``restore=`` arm rewrites the PP layout on a checksum mismatch, and the
      PP layout extends into the tail. Releasing first would make that
      recovery path fault on unbacked memory.
    * ``TP_TO_PP``: commit FIRST, refill after. The refill writes the PP
      layout, which reaches into the tail.

    AND THE COMMIT MUST BE PRICED. The ``TP_TO_PP`` commit is an allocation
    inside the flip's no-return region -- precisely the kind that killed this
    instance on 2026-08-09. ``pending_tail_bytes`` exists so the affordability
    verdict can fold it in before the flip commits, exactly as
    ``pending_restore_bytes`` does for the drafter.
    """

    def __init__(self, device_index: int, total_bytes: int, arena: Any = None):
        self._device_index = int(device_index)
        self._nbytes = int(total_bytes)
        self._committed = self._nbytes
        self._released_mib = 0.0

        if arena is None:
            from sglang.srt.mem_cache.kv_vmm_backing import KvVmmArena, align_up

            probe_gran = _probe_granularity(self._device_index)
            reserve = align_up(self._nbytes, probe_gran) + CARRIER_VA_SLACK
            arena = KvVmmArena(
                self._device_index,
                reserve_bytes=reserve,
                commit_chunk_bytes=CARRIER_COMMIT_CHUNK,
                # Same reason as the drafter's: retention parks the handles
                # instead of returning them, which makes the release worth
                # exactly 0 MiB of corridor.
                retain_handles=False,
            )
        self._arena = arena

        self._tensor = allocate_carrier_tensor(
            self._arena, self._nbytes, self._device_index
        )
        self._offset = int(self._tensor.data_ptr()) - int(self._arena.base)
        if self._offset < 0 or self._offset % self._arena.granularity != 0:
            raise PhaseFlipSpillError(
                f"{LOG_PREFIX} weights arena landed at offset {self._offset}, "
                f"which is negative or not aligned to the arena granularity "
                f"{self._arena.granularity}; commit_range would refuse it"
            )
        # Fully backed at construction: the boot packs the PP layout, which is
        # the larger one, and every caller downstream expects a normal arena.
        self._arena.commit_range(self._offset, self._nbytes)
        logger.info(
            "%s weights arena on a VA-stable reservation at 0x%x+0x%x: "
            "%.1f MiB, fully committed for the boot layout",
            LOG_PREFIX,
            int(self._arena.base),
            self._offset,
            self._nbytes / _MIB,
        )

    @property
    def tensor(self):
        """The flat uint8 arena. Its address never moves."""
        return self._tensor

    @property
    def committed_bytes(self) -> int:
        return self._committed

    def pending_tail_bytes(self, active_bytes: int) -> int:
        """Bytes a commit to ``active_bytes`` would have to ask the driver for.

        Zero when the tail is already backed. Read by the affordability gate
        BEFORE the flip commits, because this allocation happens where a
        failure cannot be unwound.
        """
        want = int(active_bytes)
        return max(0, min(want, self._nbytes) - self._committed)

    def set_active_prefix(self, active_bytes: int) -> float:
        """Back exactly ``[0, active_bytes)`` and release the rest.

        Returns MiB RELEASED (0.0 when this call committed instead). The
        release is extent-granular, so the driver may keep less than one
        commit chunk more than asked -- ``committed_bytes`` stays truthful.
        """
        want = max(0, min(int(active_bytes), self._nbytes))
        if want > self._committed:
            self._arena.commit_range(self._offset, want)
            self._committed = want
            return 0.0
        if want < self._committed:
            released = int(self._arena.decommit_range(self._offset, want))
            self._committed = int(
                self._arena.committed_bytes(self._offset)
            )
            self._released_mib = released / _MIB
            return released / _MIB
        return 0.0


class PhaseFlipSpillLadder:
    """The depth-selected set of rungs, driven by the two cutover legs.

    Built lazily on first use so that an instance whose depth is 0 -- the
    default -- never touches the drafter at all, and so that an instance
    without speculation is a no-op rather than a failure.
    """

    def __init__(self, depth: int) -> None:
        self.depth = int(depth)
        self._weights: Optional[VmmDraftWeightCarrier] = None
        self._installed = False
        self._install_failed = False

    def _install(self, draft_worker: Any) -> None:
        """Bind to the carrier the BOOT installed. Never builds one.

        Constructing a carrier here would mean reserving virtual address
        space, allocating pages and copying ~2 GB to the host from inside
        the cutover's no-return region -- and, worse, it would move the
        parameter addresses that the TP decode graphs baked at capture. The
        carrier exists before graph capture or it does not exist at all.
        """
        if self._installed or self._install_failed:
            return
        carrier = carrier_of(draft_worker)
        if carrier is None:
            # No speculation on this instance, or no carrier was installed:
            # every draft rung is a no-op. Say it once, then stay quiet.
            logger.info(
                "%s depth=%d configured but no draft-weight carrier is "
                "installed on this worker; the draft rungs are a no-op",
                LOG_PREFIX,
                self.depth,
            )
            self._install_failed = True
            return
        self._weights = carrier
        self._installed = True
        logger.info(
            "%s bound depth=%d to the boot carrier, payload %.1f MiB/rank",
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


#: Cost ranks inside the REBALANCE tier. Spread out so a later rung can be
#: slotted between two of these without renumbering the others.
_COST_ARENA_TAIL = 10
_COST_DRAFT_WEIGHTS = 20

CORRIDOR_GUARD_ATTR = "phase_flip_corridor_guard"

#: Proof/soak override for the gate's arming floor. See get_corridor_guard.
CORRIDOR_FLOOR_ENV = "SGLANG_CORRIDOR_FLOOR_MIB"


def _late_bound_draft_provider(scheduler: Any):
    """A draft-carrier provider that finds its carrier when it is called.

    Two places hold the carrier and neither is reliable at guard-build time:
    the ladder binds it on the first cutover leg, and the draft worker holds
    it only while the instance is in TP -- under strict purity
    ``scheduler.draft_worker`` is None for the whole PP phase, by
    construction.

    So look in both, at call time, and return 0 when neither has it. Zero is
    the honest answer for "the drafter is not resident and cannot be spilled
    again", which is also true right after a spill; the guard treats a
    provider that yields nothing as spent and moves down the ladder.
    """

    def free_up_to(nbytes: int) -> int:
        ladder = getattr(scheduler, "phase_flip_spill_ladder", None)
        carrier = getattr(ladder, "_weights", None) if ladder is not None else None
        if carrier is None:
            carrier = carrier_of(getattr(scheduler, "draft_worker", None))
        if carrier is None:
            return 0
        from sglang.srt.managers.corridor_guard import draft_carrier_provider

        return draft_carrier_provider(carrier)(nbytes)

    return free_up_to


def get_corridor_guard(scheduler: Any):
    """The rank's spill-before-alloc gate (#656 items 15a/15b/16), built once.

    Cached on the scheduler for the same reason the ladder is: the providers
    it holds wrap payloads whose host images must survive every flip, and a
    per-flip guard would re-register them on each seam.

    The device is taken from ``model_runner.gpu_id`` rather than
    ``torch.cuda.current_device()``. Under ``--rank-gpu-id`` a worker's
    current device is ``cuda:0`` inside its own visible set, which is not the
    physical card the corridor law is stated about, and this chain has
    already shipped one defect from that conflation.

    Returns None only when there is no scheduler device to guard, which is a
    unit-test shape rather than a serving one.
    """
    guard = getattr(scheduler, CORRIDOR_GUARD_ATTR, None)
    if guard is not None:
        return guard

    from sglang.srt.managers import corridor_guard as cg

    model_runner = getattr(
        getattr(scheduler, "tp_worker", None), "model_runner", None
    )
    device_index = getattr(model_runner, "gpu_id", None)
    if device_index is None:
        return None

    server_args = getattr(scheduler, "server_args", None)
    # THE FLOOR IS OVERRIDABLE, and the reason is a test obligation rather
    # than a tuning knob. A gate that has never been observed to FIRE is not
    # a gate that works -- it is indistinguishable from a gate that is never
    # reached, and this chain has shipped seven such mechanisms. Raising the
    # floor above the measured resting headroom is the only way to make the
    # seam's own allocation cross it on demand, on the real rig, with the
    # real payload. The corridor law itself is unchanged: the sampler still
    # judges against 1024, so a proof run at a raised floor can demonstrate
    # spill-before-alloc without being able to launder a breach.
    floor_mib = int(
        os.environ.get(CORRIDOR_FLOOR_ENV)
        or getattr(server_args, "phase_flip_corridor_floor_mib", None)
        or cg.DEFAULT_FLOOR_MIB
    )
    if floor_mib != cg.DEFAULT_FLOOR_MIB:
        logger.warning(
            "%s corridor guard floor is %d MiB, NOT the %d MiB corridor law "
            "(%s is set). This makes the gate arm earlier than the law "
            "requires; it is a proof/soak setting, and the corridor verdict "
            "must still be read against %d MiB.",
            LOG_PREFIX,
            floor_mib,
            cg.DEFAULT_FLOOR_MIB,
            CORRIDOR_FLOOR_ENV,
            cg.DEFAULT_FLOOR_MIB,
        )
    guard = cg.CorridorGuard(
        int(device_index),
        floor_mib=floor_mib,
        # The LAW never moves with the proof setting: a raised arming floor
        # must make the gate work EARLIER, never make it refuse allocations
        # the corridor permits. See CorridorGuard.__init__.
        law_floor_mib=cg.DEFAULT_FLOOR_MIB,
        fleet_probe=cg.nvml_fleet_probe(),
    )

    # The drafter is a REBALANCE and not a HOST spill even though its image
    # lives in host RAM: it evacuates a non-layer-bound payload from the
    # binding card, which is the PP levelling move item 16 prescribes. See
    # the corridor_guard module docstring for the full argument.
    #
    # RESOLVED PER CALL, NOT AT REGISTRATION. The ladder binds its carrier
    # lazily, on the first cutover leg, and the guard is built at the first
    # gate -- which happens BEFORE that leg. Capturing the carrier here would
    # cache None into a provider list that is never rebuilt, and the guard
    # would spend the rest of the process with nothing to give: inert, and
    # indistinguishable in the logs from a guard that simply never needed to
    # arm. That failure mode has shipped in this chain before.
    guard.register(
        "draft-weights",
        _COST_DRAFT_WEIGHTS,
        _late_bound_draft_provider(scheduler),
        tier=cg.RELIEF_REBALANCE,
    )

    setattr(scheduler, CORRIDOR_GUARD_ATTR, guard)
    logger.info(
        "%s corridor guard armed on device %d, floor %d MiB, providers %s",
        LOG_PREFIX,
        int(device_index),
        floor_mib,
        list(guard.providers) or "[] (nothing to spend yet)",
    )
    return guard
