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
``--phase-flip-spill-depth {none,cache,draft,arena,draft+graphs}`` (integers
0..4 also accepted) / ``SGLANG_PHASE_FLIP_SPILL_DEPTH``.

    0  none          nothing given up -- the pre-#656 seam, byte-identical
    1  cache         the outgoing phase's cached allocator segments go back
                     to the driver at the cutover      measured 2.5-3.5 GiB/card
    2  draft         + draft (MTP) weights             439/285/285 MiB/rank
                     (EXCLUSIVELY-owned bytes; the 1.86-2.01 GB figure that
                     stood here was a memory-usage delta, most of which was
                     the TARGET's embed/lm_head aliased into the draft)
    3  arena         + the weights arena's idle tail   319/220/1191 MiB/rank
                     (measured; released in TP, the phase that binds)
    4  draft+graphs  + draft CUDA graphs               NOT WIRED, see below

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

#: #833: whether the "this rung cannot take the group backing floor" warning
#: has already been emitted. Once per process is enough to make an inert path
#: visible; once per seam round would drown the log it has to be found in.
_GROUP_FLOOR_UNSUPPORTED_LOGGED = False

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

#: Reverse of DEPTH_NAMES. Derived, never hand-maintained: the refusal message
#: names the implemented rung, and a hand-written name there is exactly how it
#: came to print "the deepest implemented rung is 3 ('draft')" when rung 3 was
#: the arena tail and 'draft' was rung 2.
DEPTH_NAMES_BY_VALUE = {v: k for k, v in DEPTH_NAMES.items()}

DEPTH_ENV = "SGLANG_PHASE_FLIP_SPILL_DEPTH"
VERIFY_ENV = "SGLANG_PHASE_FLIP_SPILL_VERIFY"

#: THE EXERCISE HATCH, and the reason a hatch exists at all.
#:
#: ``IMPLEMENTED_DEPTH`` is raised only after a rung has RUN on metal (the
#: desk-written-never-executed rule). That rule and the refusal below form a
#: deadlock for the rung being wired: the rung cannot be promoted until it has
#: been exercised, and it cannot be exercised while the refusal that guards
#: promotion is what stops the boot from reaching it.
#:
#: This env is the only way through, and it is deliberately not a server arg:
#: an operator sweeping ``--phase-flip-spill-depth`` must keep hitting the
#: refusal, because for them the rung genuinely is unexercised. It is loud
#: (WARNING on every resolution) and it is expected to become dead weight the
#: moment the rung it was opened for is promoted.
DEPTH_UNIMPLEMENTED_ENV = "SGLANG_PHASE_FLIP_SPILL_DEPTH_EXPERIMENTAL"

_MIB = 1048576.0


class PhaseFlipSpillError(RuntimeError):
    """A spill/restore invariant broke.

    Raised rather than degraded: every failure mode here ends in a drafter
    whose parameters are 0-sized placeholders, and a forward through those
    is a far worse diagnostic than this exception.
    """


def _unimplemented_hatch_open() -> bool:
    """Whether this process may request a wired-but-unexercised rung."""
    return os.environ.get(DEPTH_UNIMPLEMENTED_ENV, "").strip().lower() in (
        "1",
        "true",
        "yes",
        "on",
    )


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
    if value > IMPLEMENTED_DEPTH and _unimplemented_hatch_open():
        # The hatch is checked BEFORE the refusal is built, not caught after
        # it: the refusal text asserts the rung is unreachable, and a path
        # that raises it and then swallows it would leave that assertion
        # standing while being false.
        logger.warning(
            "%s spill depth %d (%r) exceeds the implemented rung %d and is "
            "being allowed by %s=1. This is the EXPERIMENTAL hatch: the rung "
            "is wired but has not yet been exercised on metal, which is the "
            "only reason it is not promoted. Do not use it for a measurement "
            "you intend to keep without saying that the hatch was open.",
            LOG_PREFIX,
            value,
            DEPTH_NAMES_BY_VALUE.get(value, value),
            IMPLEMENTED_DEPTH,
            DEPTH_UNIMPLEMENTED_ENV,
        )
        return value
    if value > IMPLEMENTED_DEPTH:
        # Refuse rather than clamp. Clamping would make a depth sweep report
        # that the refused rung is worth exactly what a lower one is worth,
        # which reads as a measurement and is an artefact of the clamp.
        raise PhaseFlipSpillError(
            f"{LOG_PREFIX} spill depth {value} is defined but not wired; the "
            f"deepest implemented rung is {IMPLEMENTED_DEPTH} "
            f"({DEPTH_NAMES_BY_VALUE[IMPLEMENTED_DEPTH]!r}: the cached "
            f"allocator segments, the draft model's weights, AND the weights "
            f"arena's idle tail, the latter two on VA-stable KvVmmArena "
            f"carriers). Rung "
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
                    f"{n} ({own / _MIB:.0f} MiB of a {st / _MIB:.0f} MiB storage)"
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
        bind_arena_views(self._layout, self._carrier, rebind=list(self._named.items()))
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
    # SEMANTIC check, not a name check. The precondition this guard exists
    # to enforce is "no decode ever runs in PP" -- that is what makes it
    # sound to release the draft weights for the whole PP phase. Comparing
    # the mode STRING refuses any future mode that provides the same
    # guarantee by another route; `prefill_in_tp` forbids PP decode exactly
    # as strict does and was rejected purely for being spelled differently.
    from sglang.srt.managers.phase_purity import PhasePurityError, parse_purity

    purity = getattr(server_args, "phase_flip_purity", None)
    try:
        guaranteed = parse_purity(purity).decode_forbidden_in_pp
    except PhasePurityError:
        # An unparsable mode is not a guarantee. Server-args validation is the
        # place that reports WHY it is malformed; this guard must still fail in
        # its OWN currency, because callers catch PhaseFlipSpillError -- letting
        # a PhasePurityError through here changes the exception type a caller
        # sees for the same refusal.
        guaranteed = False
    if not guaranteed:
        raise PhaseFlipSpillError(
            f"{LOG_PREFIX} spill depth >= {DEPTH_DRAFT_WEIGHTS} requires "
            f"a purity mode that forbids decode in PP (strict, "
            f"prefill_in_tp, or threshold:0), got {purity!r}. The draft weights "
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

    On THAT boot ``pp`` was the max on every rank, so the tail was idle in
    **TP** -- the phase that binds on all three cards after rung 2 moved the
    binding phase there. That is what makes this rung well-aimed where the
    drafter no longer is.

    **THAT IS A MEASUREMENT OF ONE STAGE RATIO, NOT A PROPERTY OF THE RUNG.**
    ``--pp-stage-ratio 15,9,8`` derives 32,16,16 layers over 64 and puts the
    MIDDLE rank's PP layout (6690 MiB) BELOW its TP layout (7924 MiB); the
    tail is then idle in **PP** on that rank and the phases swap roles. A
    previous version of this comment said "pp is the max on every rank" as
    though it were structural, the refill path was written to match, and the
    first flip after a tp->pp copied the larger TP image into the released
    tail: ``cudaErrorInvalidValue`` inside the no-return region, all three
    ranks down (metal, 2026-08-11). The carrier itself is symmetric and
    always has been -- it is sized ``max(layout_pp, layout_tp)`` and driven by
    ``PhaseFlipStacks.refill_high_water_bytes()`` -- so nothing here needs to
    know which layout is larger. Do not re-derive that it does.

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
        # Fully backed at construction, and note WHY that is safe regardless
        # of which layout is larger: ``total_bytes`` is already
        # max(layout_pp, layout_tp) at the call site (phase_flip_boot's
        # arena_total), so committing all of it backs whichever layout the
        # boot packs AND the one it will flip to. The earlier wording here --
        # "the boot packs the PP layout, which is the larger one" -- was true
        # of one stage ratio only; see the class docstring for the flip it
        # killed.

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
            self._committed = int(self._arena.committed_bytes(self._offset))
            self._released_mib = released / _MIB
            return released / _MIB
        return 0.0


def cold_stack_deferred(server_args: Any = None) -> bool:
    """ONE predicate for the rung-4 deferral: sizer credit AND boot behaviour.

    WHY ONE PREDICATE AND NOT TWO FLAGS. The credit
    (``arena_tail_probe.post_sizing_stack_bytes(cold_stack_deferred=True)``)
    and the deferral (``build_phase_flip_tp_stack`` skipping the cold posts)
    are the same claim seen from two sides: the pool may be sized as if the
    flip draft's attention workspaces and decode graphs are absent EXACTLY
    WHEN the boot actually leaves them unbuilt. Two independently-set flags
    can disagree, and the direction that disagrees silently -- credit taken,
    deferral not performed -- sizes a pool into memory the same boot then
    allocates. That is the #678 OOM class, and no test of either side alone
    can catch it. So both sides ask this function and nothing else.

    IT ANSWERS FALSE RATHER THAN RAISING on a depth this build refuses. The
    sizer calls this on every boot, including one configured for a rung that
    is not promoted yet; taking the boot down HERE would report the ladder's
    policy as a memory-sizing failure, at a line that has nothing to do with
    it. The refusal still happens -- loudly, with its own text -- when the
    scheduler resolves the ladder in ``get_spill_ladder``.
    """
    try:
        return resolve_spill_depth(server_args) >= DEPTH_DRAFT_GRAPHS
    except PhaseFlipSpillError:
        return False


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


#: Cost ranks inside a tier. Spread out so a later rung can be slotted
#: between two of these without renumbering the others.
_COST_ALLOCATOR_CACHE = 10  # tier LOCAL
_COST_KV_BACKING = 20  # tier LOCAL
_COST_ARENA_TAIL = 10  # tier REBALANCE
_COST_DRAFT_WEIGHTS = 20  # tier REBALANCE

CORRIDOR_GUARD_ATTR = "phase_flip_corridor_guard"
KV_BACKING_RELIEF_ATTR = "phase_flip_kv_backing_relief"


def recover_kv_backing(scheduler: Any, reduce_fn=None) -> int:
    """Re-back the scheduler's KV pool and lift the allocator cap.

    Called on the tp->pp leg, where the scheduler's pool becomes the ACTIVE
    layout again. Recovery is not optional bookkeeping: a cap that is never
    lifted turns dynamic residency into a permanently smaller pool, which is
    the one fix the standing rule forbids.

    #656 C22-e: AND THE RECOVERY LEVELS TO THE GROUP. ``recover`` grows as far
    as THIS rank's distance from the corridor law allows -- correctly, since
    the grow is an allocation on the card that needed relieving -- so the
    corridor-bounded rank comes back lower than its peers and, from that
    instant, the group has TWO DIFFERENT ID SPACES. The peers hand out row ids
    the poorest rank cannot map, and nothing downstream can repair it:

        05:40 PP2  proposal: current=585390 floor=585903 (max_live=585390)
        05:40 PP1  proposal: current=546236 floor=546749 (max_live=546236)

    measured on this rig, 2026-08-14. ``collective_cap_target`` then DECLINES
    by its own rule -- ``max_floor > capable``, the poorest rank cannot expose
    the rows a peer's live set requires -- and the seam's live-slot agreement
    cannot take the union either, because a union reaching row 585390 would
    have the rank whose backing ends at 546236 read unmapped memory. Both
    refusals are correct; both leave the flip refused for the rest of the
    boot. The only place the divergence can be prevented is the instant it is
    created, which is here.

    So after the grow, the group's MINIMUM backing is reduced and every rank
    caps its allocator to it. That is an ID decision and nothing else: no
    pages are released, no memory is committed, and a rank whose backing is
    already higher simply stops handing out ids above the level its poorest
    peer can map. It costs no real capacity for the reason the cap agreement
    already states -- under pure PP every rank holds the SAME token rows, so
    rows above the group minimum could never have been admitted against.

    #792: AND THE LEVEL MAY NEVER SIT BELOW THE GROUP'S LIVE SET. What the
    levelling above prevents is a rank exposing ids its peers cannot map; what
    it must not do in exchange is expose FEWER ids than the group's live set
    already occupies. Every id at or above the cap that a peel frees is taken
    straight back by ``KvRowCap``'s free listener, so a cap under the live
    high-water mark makes the pool unpayable: the tree empties, the allocator
    never moves, and the next prefill raises out-of-memory against a full
    evictable tree. The payload therefore carries each rank's floor, and the
    group declines when the MAX floor is above the MIN backing -- the same
    verdict ``kv_backing_relief.collective_cap_target`` already reaches for the
    same condition, on the seam's half of this mechanism.

    ``reduce_fn`` is the seam's element-wise MIN channel. Entered on the
    tp->pp post-cutover hook, which every rank reaches exactly once per
    tp->pp cutover -- the cutover itself is unanimous, so this is not a
    conditional path. Without a channel (single-rank shapes) the levelling is
    skipped, which is correct: one rank is level with itself.
    """
    #: #834 B step 1: THE TWO HALVES ARE NOW NAMED, and this function is their
    #: composition. Nothing about the shipped behaviour changes -- the same
    #: grow runs, then the same levelling, in the same order, on the same
    #: values -- but the halves are separately callable, which is the whole
    #: prerequisite the #830 F2 design note put first: "Split
    #: recover_kv_backing into grow (rank-local) and level (collective) as
    #: separate callables. The seam then runs grow-then-level as today, with no
    #: behaviour change, and the split is provable by test before anything
    #: moves."
    grown = grow_kv_backing_local(scheduler)
    level_kv_backing_to_group(scheduler, reduce_fn)
    return grown


def grow_kv_backing_local(scheduler: Any) -> int:
    """#834 B: the RANK-LOCAL half of the recovery. Grows, never agrees.

    This is the expensive half and the whole reason the split exists.
    ``relief.recover()`` reaches ``runtime_set_backing_rows`` and from there
    cuMemCreate/cuMemMap, and #830 F2 measured this call at 99-101% of the
    cutover term across 63 rank-flips in two independent boots -- 3448.9 and
    3773.0 ms mean, against 50.2 ms with HiCache off.

    NOT ONE COLLECTIVE IS REACHED HERE, and that is a checkable claim rather
    than a comfortable one: ``kv_backing_relief.py`` contains no collective
    anywhere, which is what makes this half safe to run at a rank-local cadence
    while the levelling below is not.

    IT DOES, HOWEVER, RAISE THIS RANK'S EXPOSURE. ``recover()`` ends in
    ``clamp_exposure_to_backing``, which lifts the allocator to this rank's own
    new backing -- correct when the levelling follows immediately, and the
    precise hazard when it does not. A caller that defers the levelling MUST
    re-clamp; see ``clamp_kv_exposure_to_level``.
    """
    relief = getattr(scheduler, KV_BACKING_RELIEF_ATTR, None)
    if relief is None:
        return 0
    try:
        return int(relief.recover())
    except Exception as e:
        logger.error(
            "%s KV backing recovery failed: %s. The pool is still capped, so "
            "admission capacity stays reduced until the next successful "
            "recovery -- this is a capacity loss, not a corridor risk",
            LOG_PREFIX,
            e,
        )
        return 0


def clamp_kv_exposure_to_level(scheduler: Any, level: int) -> int:
    """#834 B step 2: hold exposure at a level the GROUP has agreed to.

    THE INVARIANT THE DEFERRED GROW IS BUILT AGAINST, written as its own
    callable because the #830 F2 design note says it is the first thing to
    write a red arm against: "keep the pool's exposure clamped to the pre-grow
    group level until the levelling runs, so no rank can expose an id a peer
    has not backed."

    Why this is safe and cheap: it is an ID decision and nothing else. It uses
    the same ``KvRowCap`` primitive the levelling itself uses -- no pages are
    released, no memory is committed, live allocations are not enumerated, not
    moved and not touched. What it costs is capacity, temporarily, which is the
    correct thing to spend here: the alternative is a rank handing out an id a
    peer cannot map, which aborts ALL THREE inside ``store_kvcache``'s bounds
    assert.

    ``level`` None or <= 0 means "no agreed level to clamp to" and is a no-op --
    an unknown must never be turned into a cap, the same rule #721 and #830 F4
    both apply to refusals. None is the #792 DECLINE arriving here, and it is
    the commonest way this is called with nothing to do, so it is handled
    before the int() rather than inside it.
    """
    relief = getattr(scheduler, KV_BACKING_RELIEF_ATTR, None)
    if relief is None or level is None or int(level) <= 0:
        return 0
    try:
        return int(relief.level_recovery_to(int(level)))
    except Exception as e:
        logger.error(
            "%s could not clamp KV exposure to the agreed level %d (%s). This "
            "rank may now expose ids the group has not levelled to, which the "
            "seam's frame ballot refuses -- a lost flip, never a half-flipped "
            "group",
            LOG_PREFIX,
            int(level),
            e,
        )
        return 0


def level_kv_backing_to_group(scheduler: Any, reduce_fn) -> Optional[int]:
    """#834 B: the COLLECTIVE half. Agrees, never grows.

    Returns the level the group agreed on, or None when no level was applied
    (no channel, no relief, a declined level, or a group already even). The
    return value is what a deferred grow needs in order to know which level it
    must clamp itself back to, and it did not exist before this split.

    THIS HALF CANNOT MOVE TO A RANK-LOCAL CADENCE, and the reason is recorded
    at the call site in phase_flip_runtime.py rather than only here: entering a
    blocking reduction at a local cadence can pair with a peer blocked in a
    pipeline recv, which is the 2026-08-08 boots 9/10 PP wedge. "Just after a
    cutover" is approximately synchronized, and approximately is what that
    wedge punished.
    """
    relief = getattr(scheduler, KV_BACKING_RELIEF_ATTR, None)
    if relief is None or reduce_fn is None:
        return None
    applied: Optional[int] = None
    try:
        backed = int(relief.backed_rows())
        # #792: THE FLOOR RIDES ALONG, because the level this decides is
        # applied to the ID SPACE and the live set is what the id space owes.
        # The payload was [backed, -backed] -- how many rows each rank has
        # mapped, and nothing about the rows already in use -- so the group
        # could and did agree a level BELOW its own live high-water mark.
        # Sent negated, so the same MIN channel yields the group's MAX floor:
        # the most-loaded rank sets the limit, exactly as it does for
        # ``collective_cap_target``.
        floor = int(relief.live_floor_rows())
        reduced = list(reduce_fn([backed, -backed, -floor]))
        group_min = int(reduced[0])
        group_max = -int(reduced[1])
        # A channel that truncated (or a peer on an older tree) leaves the
        # floor UNKNOWN, and an unknown floor is not a low one: it declines
        # exactly as a floor above the level would.
        floor_known = len(reduced) > 2
        group_floor = -int(reduced[2]) if floor_known else -1
        unlevel = group_min > 0 and group_min < group_max
        # #834 B: THE AGREED LEVEL IS group_min WHETHER OR NOT IT HAD TO BE
        # APPLIED. An already-even group agreed just as much as an uneven one
        # that had to be capped -- it simply needed no cap to get there. A
        # deferred grow has to clamp itself back to that number either way, so
        # reporting it only in the branch that moved rows would leave the
        # commonest case (a level group) with no level to clamp to, which is
        # the shape that silently exposes unlevelled ids.
        if group_min > 0:
            applied = group_min
        if unlevel and (not floor_known or group_floor > group_min):
            # THE ONE STATE THAT CANNOT BE ANSWERED, and the levelling used to
            # answer it anyway. The poorest rank cannot expose the rows the
            # group's live set already occupies, so there is no honest level:
            # capping to the group minimum withholds ids the radix tree is
            # holding, and every eviction after it frees the TREE and pays the
            # POOL nothing (boot instr12, 2026-08-21: cap levelled to 40960
            # against a highest live row of 136720; 63641 ids confiscated; the
            # next prefill raised out-of-memory with 67674 evictable tokens in
            # the tree).
            #
            # Declining leaves the ranks on different id spaces, which the
            # seam's frame ballot refuses -- a lost flip, never a rank. That is
            # the identical trade ``collective_cap_target`` makes for the same
            # condition (``max_floor > capable``), and this path is the one
            # place in the chain that did not make it.
            logger.error(
                "%s DECLINED to level the recovery: the group's poorest rank "
                "backs %d rows while the group's live set needs %d "
                "(this rank backs %d, its own floor is %d). Capping the "
                "allocator to %d would withhold every id at or above it -- "
                "including the ones the radix tree is holding -- so eviction "
                "would pay the pool nothing and the next prefill would raise "
                "out-of-memory against a full evictable tree (#792). The id "
                "spaces stay apart and the frame ballot refuses the flip "
                "until a recovery closes the gap: a lost flip, never a dead "
                "rank",
                LOG_PREFIX,
                group_min,
                group_floor,
                backed,
                floor,
                group_min,
            )
            # DECLINED IS NOT AGREED. A caller holding a deferred grow must not
            # read this as a level it may expose to.
            applied = None
        elif unlevel:
            moved = int(relief.level_recovery_to(group_min))
            applied = group_min
            logger.warning(
                "%s KV recovery levelled to the group: this rank backs %d "
                "rows, the group's poorest backs %d, so the allocator is "
                "capped at %d (%+d exposed rows). A rank that came back from "
                "a phase with more backed rows than its peers would hand out "
                "ids they cannot map, and neither the cap agreement nor the "
                "seam's live-slot union can repair that afterwards -- the "
                "peers' own live rows put the agreement's floor above the "
                "poorest rank's ceiling. No pages are released here; this is "
                "an id decision",
                LOG_PREFIX,
                backed,
                group_min,
                group_min,
                moved,
            )
    except Exception as e:
        logger.error(
            "%s could not level the recovery to the group (%s). The ranks may "
            "now expose different id spaces, which the seam's frame ballot "
            "refuses -- a lost flip, never a half-flipped group",
            LOG_PREFIX,
            e,
        )
    return applied


def recover_kv_backing_on_abandon(
    scheduler: Any, reduce_fn, direction: str = "", why: str = ""
) -> int:
    """#814: lift the KV cap when a flip is ABANDONED, not only when one commits.

    THE RATCHET THIS BREAKS. ``recover_kv_backing`` is the only thing that
    lifts ``KvRowCap``, and until this existed its only two call sites were
    post-cutover hooks (phase_flip_runtime.py:1818 and :1835). That is fine
    while flips commit. It is a trap once one does not, and the trap closes on
    itself:

      1. a corridor-bounded recovery leaves the ranks unequal (measured on this
         rig, one boot: 210944 / 124928 / 131072 backed rows);
      2. the cap agreement levels the group to the poorest -- correct, since
         under pure PP an id a peer cannot map aborts all three ranks inside
         store_kvcache's bounds assert -- so the allocator is capped at 124928
         of 465190, 26.8% of the id space;
      3. the capped pool fills, so the group's live floor is 100% of what is
         left;
      4. the next flip is therefore DECLINED for want of fit;
      5. no cutover, so no recovery, so the cap is never lifted -- back to 3.

    Measured shape of the trap in that boot: 27 pool censuses, three ranks,
    NINE events, and not one of them ``post-cutover tp_to_pp``. All six return
    attempts stopped at ``at-arm``. ``recovered to N of M rows`` appears zero
    times. The pool sat at 26.8% for the life of the process and a user got an
    overloaded_error against a pool that had been sized 3.7x larger.

    ``recover``'s own docstring rests on the assumption this breaks -- "an
    admission-capacity loss, which is recoverable ON ANY LATER LEG" -- and
    ``recover_kv_backing``'s names the outcome as forbidden: "a cap that is
    never lifted turns dynamic residency into a permanently smaller pool, which
    is the one fix the standing rule forbids."

    WHY THE ABANDON EXIT IS THE RIGHT LEG, and not the arm or the shrink.
    Growing must not happen where the seam is about to spend the memory: the
    first metal boot of a GROWING cap agreement hit ``cuMemCreate failed:
    CUDA_ERROR_OUT_OF_MEMORY`` on all three ranks, rank 0 driven to 3 MiB free
    (kv_backing_relief.py:2710-2745, 2026-08-13). At an abandon there is no
    seam to fund -- the flip is over, nothing moved, and the layout the group
    is parked in is the one it will keep serving from. That is precisely where
    a withheld cap is pure loss.

    NOTHING HERE WEAKENS THAT GUARD. The bound lives inside ``recover`` itself
    (kv_backing_relief.py:2503-2505): a fresh per-rank ``mem_get_info`` read at
    call time, minus this card's own corridor law, and ``rows <= was`` defers
    without committing a byte. It is intrinsic to the function, not to the
    caller, so this call site inherits exactly the protection the two
    post-cutover sites already rely on. The split the design asks for is kept:
    the grow stays rank-local and corridor-bounded, the LEVELLING stays
    collective and strictly non-allocating.

    CALLER CONTRACT, and it is the load-bearing one: this enters a COLLECTIVE
    through ``reduce_fn``, so it may only be called where EVERY rank arrives.
    Its one call site is the group-unanimous abandon exit in ``_execute``,
    downstream of ``reduced_fit = self._collective_min(payload)`` -- a MIN
    reduction that is bit-identical on every rank, with no return and no raise
    between it and the exit. Without a channel this is a no-op, matching
    ``recover_kv_backing``'s own behaviour on single-rank shapes.

    NEVER RAISES. A lost flip must not become a dead rank: an abandon is
    already the unhappy path, and an exception escaping here would take down an
    instance that was merely declining to flip.
    """
    if reduce_fn is None:
        return 0
    try:
        grown = int(recover_kv_backing(scheduler, reduce_fn=reduce_fn))
    except Exception as e:  # noqa: BLE001 - an abandon must not kill the rank
        logger.error(
            "%s KV backing recovery at the abandoned %s seam failed (%s). The "
            "cap stays where it was, so admission capacity remains reduced -- "
            "a capacity loss, never a fault",
            LOG_PREFIX,
            direction or "flip",
            e,
        )
        return 0
    if grown:
        logger.warning(
            "%s recovered %d KV rows at the ABANDONED %s seam%s. The flip did "
            "not commit, so the post-cutover recovery hook never ran; without "
            "this leg the cap that helped refuse the flip would have stayed on "
            "for the life of the process (#814)",
            LOG_PREFIX,
            grown,
            direction or "flip",
            f" ({why})" if why else "",
        )
    return grown


def apply_cap_agreement(scheduler: Any, reduced_cap_fields) -> int:
    """#656 C22: ONE exposed row level for the whole group, every seam round.

    THE SHRINK WAS COLLECTIVE AND THE RECOVERY WAS NOT, and that asymmetry is
    the divergence the frame ballot caught on metal (boot ``boot_m1``,
    2026-08-13). ``KvBackingRelief.recover`` is bounded by each rank's own
    distance from the corridor law -- correctly, since recovery is an
    allocation on the card that needed relieving -- so the rank nearest the
    law recovers less than its peers and stays capped where they do not::

        14:42:25 PP1  KV-BACKING recovered to 537986 of 578390 rows (corridor-bounded)
        14:42:25 PP1  POOL CENSUS post-cutover: free=197527 unaccounted=40404
        14:42:25 PP0  POOL CENSUS post-cutover: free=237931 unaccounted=0
        14:42:25 PP2  POOL CENSUS post-cutover: free=237931 unaccounted=0

    ``578390 - 537986 = 40404``, which is the census delta to the row. From
    that moment the ranks enumerated different live slot sets, framed payloads
    of different lengths, and every subsequent ``pp_to_tp`` round was refused
    by the ballot -- decode's leg, so the instance wedged.

    Levelling costs the healthy ranks nothing real: under pure PP every rank
    holds the SAME token rows, so rows above the group minimum could never
    have been admitted against. It is also not a ratchet -- the level rises
    again as soon as the poorest rank can fund it.

    WHERE THIS RUNS AND WHY. It rides the KV rung's EXISTING reduction, whose
    payload is widened from 4 fields to 8 -- no second collective, because the
    collective COUNT is itself a load-bearing invariant here (``on_round``'s
    census diffs it across ranks to catch desyncs, and the gate's own tests
    pin one reduction per gate call). That site is the one place every rank
    reaches unconditionally, and it runs BEFORE ``_frame_digest`` in the same
    round -- so a divergence that opened during the previous phase is closed
    before the frame that would have tripped over it is computed. Both legs,
    unlike the shrink: the level has to be uniform whichever way the flip is
    going.

    ``reduced_cap_fields`` is the second half of that reduced payload.

    Returns this rank's signed change in exposed rows.
    """
    from sglang.srt.managers import kv_backing_relief as kbr

    relief = getattr(scheduler, KV_BACKING_RELIEF_ATTR, None) if scheduler else None
    target = kbr.collective_cap_target(reduced_cap_fields)
    if target is None or relief is None:
        return 0
    try:
        return int(relief.reconcile_to(target))
    except Exception as e:
        logger.error(
            "%s could not level this rank to the agreed %d rows: %s. The rank "
            "stays where it is and the seam's frame ballot refuses the flip "
            "until it agrees -- a lost flip, never a half-flipped group",
            LOG_PREFIX,
            target,
            e,
        )
        return 0


def _cached_relief_estimate(device_index: int) -> int:
    """Bytes the tiers BELOW the KV rung could still return, on this rank.

    TAKEN, THEN MEASURED -- no longer estimated. This used to return
    ``reserved - allocated``, an UPPER bound on what ``empty_cache`` really
    hands back (the allocator keeps whole segments it is still carving from),
    and the overestimate was deliberate on the argument that "under-shrinking
    is retried while over-shrinking costs admission capacity that bought
    nothing".

    THAT ARGUMENT IS FALSIFIED. Under-shrinking is not retried: eight refusals
    latch the direction "unfundable" and the instance holds in TP with tens of
    thousands of tokens pending at 1000-1600 tok/s where the PP layout does
    4000-7000. The rung's own trace had already recorded the mechanism -- the
    deficit was NEGATIVE on 100 % of arms across two acceptance runs, and
    dropping this term alone flipped every one of them positive by 260-832
    MiB. An overstated subtraction is why the only rung that can pay real
    bytes declined every single time it was asked.

    So the cheap tier is RECLAIMED HERE, before the rung is asked, and the ask
    is then sized against what the driver actually has. The reclaimed bytes
    land in the driver's free column, which the rung probes directly, so this
    function returns 0: subtracting them a second time is the double-count
    that caused the defect. Reclaiming here rather than after the verdict is
    the same reordering the guard already applies to this tier for the same
    reason.
    """
    try:
        import torch

        before = int(torch.cuda.mem_get_info(device_index)[0])
        torch.cuda.empty_cache()
        after = int(torch.cuda.mem_get_info(device_index)[0])
        returned = max(0, after - before)
        if returned:
            logger.info(
                "%s cheap tier taken before the KV ask: %.0f MiB returned to "
                "the driver, so the rung sizes against real free memory "
                "instead of an upper bound on this tier",
                LOG_PREFIX,
                returned / (1024 * 1024),
            )
        # Already in the free column the rung probes. Counting it again is the
        # subtraction that made the rung decline on every arm.
        return 0
    except Exception:
        return 0


#: The leg on which the KV rung shrinks WITHOUT QUESTION. See the driver's
#: docstring.
KV_RELIEF_DIRECTION = "pp_to_tp"

#: #662: may the rung ALSO fund the ``tp_to_pp`` leg?
#:
#: It must, or that leg has no funder at all. Measured on metal 2026-08-15,
#: max-KV vector, corridor at its law: every tp_to_pp arm abandoned with
#: "seam entry margin short: want 2194 MiB, free 2892 -> 3098 MiB, reclaimed
#: 206 MiB from [allocator-cache]" -- 206 MiB of torch cache against a
#: 2194 MiB ask, because the exclusion below set ``relief = None`` and
#: allocator-cache is the only other tier that pays. Eight such abandons
#: install the "seam unfundable" guard and THE INSTANCE NEVER ENTERS THE
#: PREFILL LAYOUT AGAIN -- which is exactly the reported defect: long
#: prompts prefilled at TP speed because the stack could not leave TP.
#:
#: The original exclusion's reasoning is sound about COST and wrong about
#: necessity. It says a shrink here "would be undone within the same flip"
#: by the post-cutover ``recover_kv_backing``. True -- and that undo is the
#: price of the flip happening at all. A cuMemCreate on the recovery path is
#: strictly better than a phase the instance can never reach.
#:
#: THE POOL IS INACTIVE AT THIS POINT, which is what makes it safe. The gate
#: runs BEFORE the cutover, so during the TP phase the PP layout's pool is
#: holding nothing and its backing is spendable -- the same argument the
#: pp_to_tp leg already makes about the TP phase. On this rig the pp_to_tp
#: leg had already left it at 413910 rows against a floor of 635, so the
#: bytes the tp_to_pp seam needed were sitting there unbacked-by-anything
#: and simply unreachable.
#:
#: 0 restores the shipped one-leg behaviour exactly, as a VALUE of the same
#: term rather than a second code path.
ENV_FUND_TP_TO_PP = "SGLANG_SEAM_FUND_TP_TO_PP"


def _may_fund_tp_to_pp() -> bool:
    return os.environ.get(ENV_FUND_TP_TO_PP, "1") not in ("0", "false", "False")


#: #662-F4: WHICH SEAM DIRECTIONS MUST BE TREATED AS UNFUNDABLE.
#:
#: A FAULT INJECTION, and the only one in this file. It exists because the
#: decode-stall SLO valve added in 1d1dbf9dba cannot be reached on metal by
#: any honest configuration. The invariant it guards -- decodes are never held
#: past the SLO by a funding failure -- needs the instance to be IN the PP
#: layout, with a decode resident, while ``pp_to_tp`` cannot be funded. Every
#: lever tried on 2026-08-15 made BOTH directions unfundable at once, so the
#: instance never entered PP and the shape under test never occurred:
#:
#:   rung disabled globally -> pp_to_tp funded from the allocator cache
#:                             anyway, and (because ``refusal_is_fatal`` opens
#:                             the host tier for exactly this leg) from system
#:                             RAM as well. Six flips DONE, nothing held.
#:   arming floor raised    -> tp_to_pp was blocked with it, the instance
#:                             rested in TP, and decode was never held at all.
#:
#: That is the catch-22 this term ends, and the property it needed is
#: DIRECTION. With ``pp_to_tp`` named here, tp_to_pp still funds and the
#: instance enters PP the ordinary way; the return leg then refuses, decode is
#: held, and the SLO is the only thing left that can free it.
#:
#: WHAT IT DOES NOT DO, stated because a proof is worth only the mechanism it
#: exercises. It does not simulate a card with no memory: the ladder still
#: runs and still spends. It overrides the VERDICT the gate reached, at the
#: single point where that verdict becomes the group's, so everything
#: downstream is the real abandon path -- the same ``too_small`` vote, the
#: same unanimous MIN, the same FLIP ABANDONED log, the same per-direction
#: refusal hold and backoff. What is injected is the refusal; what is proven
#: is what the system does with one.
#:
#: NOT A SERVING SETTING. Unset is the default and changes nothing.
ENV_SEAM_UNFUNDABLE = "SGLANG_SEAM_UNFUNDABLE_DIRECTIONS"

#: Resolved sets, keyed by the raw value, so the parse happens once per
#: distinct setting rather than once per seam.
_UNFUNDABLE_CACHE: dict = {}


def _parse_unfundable(raw: str) -> frozenset:
    """The named directions, or an EMPTY set on anything unrecognised.

    AN UNPARSABLE VALUE INJECTS NOTHING, and does not raise. This term is
    read on the seam's no-return path, where ``_abandon_parked_flip`` states
    the law: a raise from inside a cutover climbs into the event loop and
    takes the instance down. Refusing to guess is the gate's own currency --
    it declines to object -- and the ERROR below is what stops that being
    silent. An operator who mistypes a direction gets a healthy instance and
    a log line saying the injection is not armed, which is the failure a
    proof boot can notice; the alternative fails the whole instance to
    protect a diagnostic.
    """
    from sglang.srt.managers.phase_policy import PP_TO_TP, TP_TO_PP

    known = {PP_TO_TP, TP_TO_PP}
    names = {part.strip() for part in str(raw).split(",")}
    names.discard("")
    unknown = sorted(names - known)
    if unknown:
        logger.error(
            "%s %s names %s, which is not a flip direction; the unfundable "
            "injection is NOT armed. Valid directions are %s.",
            LOG_PREFIX,
            ENV_SEAM_UNFUNDABLE,
            ", ".join(repr(u) for u in unknown),
            ", ".join(sorted(known)),
        )
        return frozenset()
    return frozenset(names)


def unfundable_seam_directions() -> frozenset:
    """The configured set, announced ONCE when it is non-empty.

    The announcement is not decoration. This corpus has shipped seven terms
    that were present and inert, and a fault injection is the worst possible
    place for that: an unarmed injection makes a flip that funded normally
    look like the proof of a valve that never fired. So an armed injection
    says so in the boot log, by name.
    """
    raw = os.environ.get(ENV_SEAM_UNFUNDABLE, "")
    if not raw.strip():
        return frozenset()
    if raw not in _UNFUNDABLE_CACHE:
        resolved = _parse_unfundable(raw)
        _UNFUNDABLE_CACHE[raw] = resolved
        if resolved:
            logger.warning(
                "%s [#662-F4 UNFUNDABLE-SEAM] ARMED for %s: every flip in "
                "%s direction will be refused by the group as if no tier "
                "could pay for it. This is a fault injection for the "
                "decode-stall SLO proof and must not be set on an instance "
                "that is meant to serve.",
                LOG_PREFIX,
                ", ".join(sorted(resolved)),
                "that" if len(resolved) == 1 else "either",
            )
    return _UNFUNDABLE_CACHE[raw]


def seam_unfundable_objection(direction: str) -> Optional[str]:
    """The injected objection for ``direction``, or None.

    Returns a string in the same currency as the corridor gate's own refusal
    -- one clause naming why the seam may not enter -- so it joins
    ``too_small`` and votes exactly like a refusal the gate reached itself.
    It deliberately does NOT carry ``SEAM_MARGIN_DELAY_TAG``: a margin delay
    is a bounded wait the seam retries out of, while what is being injected
    is a funding failure, and calling it a delay would prove the wrong thing.
    """
    if str(direction) not in unfundable_seam_directions():
        return None
    return (
        f"{direction} is configured unfundable by {ENV_SEAM_UNFUNDABLE} "
        f"(fault injection for the decode-stall SLO proof), so the group "
        f"refuses this seam whatever the ladder returned"
    )


def _rank_verdict_terms(relief, target) -> str:
    """This rank's clause on the verdict line: applied when granted, proposed
    when declined.

    Tolerates a rung that predates ``explain_shrink_ppm`` (a stub, or an older
    rung surface) by falling back to the proposal summary, because a diagnostic
    may never be the thing that takes the seam down.
    """
    if target is not None:
        explain = getattr(relief, "explain_shrink_ppm", None)
        if explain is not None:
            return explain(target)
    return relief.last_proposal_summary()


def collective_kv_backing_relief(
    scheduler: Any,
    reduce_fn,
    *,
    want_bytes,
    guard,
    direction: str = KV_RELIEF_DIRECTION,
    discretionary_bytes: int = 0,
    slots_digest: int = 0,
    max_live_row: int = -1,
    slot_ballot_out: Optional[dict] = None,
) -> int:
    global _GROUP_FLOOR_UNSUPPORTED_LOGGED
    """#656 item 12, the device half: ONE shrink target for the whole group.

    A REFUSAL MAY BE DECIDED LOCALLY. A CAPACITY MAY NOT. The corridor guard
    reads NVML per rank and refuses per rank, which is safe because a refusal
    joins the seam's unanimous abandon. The KV cap is the opposite kind of
    decision: it changes ``available_size()``, which feeds ADMISSION, so three
    ranks that size their own shrink from their own free memory end up
    disagreeing about how much work the group may take -- measured on metal as
    449039 / 451037 / 175225 / 145734 rows, followed by a scheduler that
    stopped heartbeating while every rank was alive (HANDOFF_675 §1a).

    WHY HERE AND NOT INSIDE THE GUARD'S LADDER. A collective may only be
    entered at a point EVERY rank reaches unconditionally. The guard's
    providers run behind its ARM CONDITION, which is rank-local by
    construction (``free - want < floor`` against this rank's own NVML
    reading), so a reduction down there hangs the first time one rank arms and
    a peer does not -- strictly worse than the desync it would be fixing. This
    function is called from ``_corridor_gate``, on the same unconditional path
    as the seam's existing fit reduction, and every rank calls it whether or
    not it is under pressure.

    It runs BEFORE the guard, so the bytes it frees are money the guard's own
    probe then sees, and its ask already discounts the cheaper tiers.

    ``discretionary_bytes`` is the part of ``want_bytes`` the caller can do
    WITHOUT -- at the seam, the C20 entry margin, whose shortfall produces a
    graded delay rather than a failure. It is bounded here by what this rank's
    rung can return without crossing its admission floor, and the remainder is
    passed through untouched.

    WHY THE DISCRETIONARY HALF NEEDS A BOUND AT ALL, given that the floor
    already stops the shrink. The rung's ask sets the SIZE of the shrink, not
    only its limit: the deficit is ``floor + delta + want - free - cheap``, so
    an ask nothing can fund produces a deficit nothing can satisfy and the rung
    goes to its floor on EVERY seam -- spending all of its slack, every time,
    to close a gap that was never closable. That is how a one-digit operator
    typo became an instance death (HANDOFF_681 §1a) and it stays expensive even
    once the death is fixed. Bounded, the rung shrinks for what it can actually
    deliver and the gate takes its delay for the rest.

    THE MANDATORY HALF IS NEVER BOUNDED. If the rung cannot fund the seam's
    staging, the guard must see the full ask and refuse the seam -- a free,
    unanimous abandon with every request intact. Laundering that into a smaller
    ask would produce a seam that "fits" and an allocator that disagrees.

    ``slots_digest`` / ``max_live_row`` / ``slot_ballot_out`` carry #656 C22-d:
    the seam's LIVE SLOT SET agreement rides this same reduction, four fields
    wider again (12, not 8), for the reason the cap agreement rides it at all
    -- this is the one point every rank reaches unconditionally BEFORE
    ``_frame_digest`` runs in the same round. ``slot_ballot_out``, when given,
    is filled in place with :func:`kv_backing_relief.collective_slot_ballot`'s
    verdict so the caller can act on it without a second collective.

    Returns the bytes THIS rank gave back to the driver.
    """
    if reduce_fn is None:
        # No consensus channel: single-rank shapes only. Uniform across the
        # group by construction, since the channel is built for all or none.
        return 0
    from sglang.srt.managers import kv_backing_relief as kbr

    relief = getattr(scheduler, KV_BACKING_RELIEF_ATTR, None) if scheduler else None
    # #656 C22: the CAP AGREEMENT rides this same reduction, and it is NOT
    # subject to the exclusions below. The shrink happens on one leg and needs
    # a guard to propose against; the levelling has to happen on both legs and
    # needs neither, because it changes no bytes -- it only decides which ids
    # the allocators may hand out. Held in its own name so the exclusions
    # cannot silently disable it, which is exactly how the recovery came to be
    # rank-local in the first place.
    cap_relief = relief
    if str(direction) != KV_RELIEF_DIRECTION and not _may_fund_tp_to_pp():
        # ONE LEG ONLY, and the asymmetry is structural rather than cautious.
        #
        # The scheduler's KV pool is the PP LAYOUT'S pool. Capping it on the
        # pp->tp leg gives up backing that is about to hold nothing, because
        # the PP layout goes inactive for the whole TP phase -- and pp->tp is
        # also the leg whose refusal is fatal (strict purity forbids decode in
        # PP, so a refused pp->tp starves decode outright).
        #
        # On the tp->pp leg the same pool is about to become ACTIVE again, and
        # ``recover_kv_backing`` runs at that leg's post-cutover hook. A shrink
        # here would therefore be undone within the same flip -- churn whose
        # undo is a ``cuMemCreate``, which is the one operation this chain has
        # already paid 2.5 GiB to learn not to do near a tight card.
        #
        # The rank still ABSTAINS INTO THE REDUCTION rather than returning
        # early. The direction comes from the seam's reduced consensus payload
        # and is therefore already agreed, so skipping would in fact be safe --
        # but "safe because a value upstream is agreed" is precisely the
        # reasoning that produces collective hangs when the upstream changes.
        # One abstention per tp->pp leg is cheaper than that risk.
        relief = None
    if guard is None:
        # No floor to propose against. Abstain -- but still reduce, because a
        # rank that returns early here hangs the ones that did not.
        relief = None
    ask_bytes = int(want_bytes)
    discretionary = max(0, int(discretionary_bytes))
    if relief is not None and discretionary:
        mandatory = max(0, ask_bytes - discretionary)
        try:
            fundable = int(relief.fundable_bytes())
        except Exception as e:
            # An unreadable bound is not an unbounded one. Granting nothing
            # discretionary leaves the mandatory ask exactly as it was, which
            # is the behaviour of a boot with no margin configured at all.
            logger.warning(
                "%s could not price what the KV rung can fund (%s); the "
                "discretionary %d MiB is not asked for",
                LOG_PREFIX,
                e,
                discretionary // (1024 * 1024),
            )
            fundable = 0
        granted = max(0, min(discretionary, fundable - mandatory))
        if granted < discretionary:
            logger.info(
                "%s KV rung asked for %d of %d MiB discretionary (%s): it can "
                "return %d MiB above its admission floor, and asking for more "
                "would drive it to that floor for bytes it does not have",
                LOG_PREFIX,
                granted // (1024 * 1024),
                discretionary // (1024 * 1024),
                direction,
                max(0, fundable) // (1024 * 1024),
            )
        ask_bytes = mandatory + granted
    proposal = kbr.ABSTAIN
    if relief is not None:
        try:
            proposal = relief.propose(
                want_bytes=int(ask_bytes),
                floor_bytes=int(guard.floor_bytes),
                delta_bytes=int(guard.delta_bytes),
                cheap_relief_bytes=_cached_relief_estimate(guard.device_index),
            )
        except Exception as e:
            # EVERY RANK STILL ENTERS THE REDUCTION. A rank that returned
            # early here would leave its peers waiting on a collective that
            # never completes, which is a hang rather than a degraded rung.
            logger.warning(
                "%s KV relief proposal failed (%s); abstaining, which makes "
                "the whole group decline",
                LOG_PREFIX,
                e,
            )
            proposal = kbr.ABSTAIN
    # THE SECOND HALF OF THE SAME PAYLOAD (#656 C22). Built unconditionally,
    # including when this rank abstains from the shrink, because the payload
    # LENGTH must be identical on every rank or the reduction is malformed.
    cap_proposal = kbr.CAP_ABSTAIN
    if cap_relief is not None:
        try:
            cap_proposal = cap_relief.cap_proposal()
        except Exception as e:
            logger.warning(
                "%s KV cap proposal failed (%s); abstaining, which makes the "
                "whole group decline the levelling",
                LOG_PREFIX,
                e,
            )
            cap_proposal = kbr.CAP_ABSTAIN
    # THE THIRD QUARTER OF THE SAME PAYLOAD (#656 C22-d). Built
    # unconditionally, like the cap proposal above and for the same reason:
    # the payload LENGTH must be identical on every rank or the reduction is
    # malformed. A rank with no relief object contributes SLOT_ABSTAIN, whose
    # backing field is unbounded so the group's minimum is decided by whoever
    # can actually read theirs.
    if cap_relief is not None:
        try:
            slot_fields = kbr.slot_proposal(
                slots_digest, max_live_row, cap_relief.backed_rows()
            )
        except Exception as e:
            logger.warning(
                "%s live-slot proposal failed (%s); abstaining, which leaves "
                "the frame ballot to refuse a divergence instead of the "
                "agreement repairing it",
                LOG_PREFIX,
                e,
            )
            slot_fields = kbr.SLOT_ABSTAIN
    else:
        slot_fields = kbr.slot_proposal(slots_digest, max_live_row, kbr.SLOT_ABSTAIN[3])
    reduced = list(reduce_fn(list(proposal) + list(cap_proposal) + list(slot_fields)))
    verdict = kbr.collective_slot_ballot(reduced[8:12])
    if slot_ballot_out is not None and verdict is not None:
        slot_ballot_out.update(verdict)
    # #833: FEED THE GROUP FLOOR BACK TO THE THING THAT SETS EXPOSURE.
    #
    # The reduction already carries the group's MIN backed rows -- it is what
    # the frame ballot refuses a union against. Until now nothing told the
    # exposure clamp about it, so each rank capped its id space at its OWN
    # backing and the group's exposures diverged by exactly the amount its
    # pools differ (uneven vectors: by design, and never zero). The widest rank
    # then issues ids the narrowest cannot map, and every subsequent tp_to_pp
    # abandons on the union bound. Measured: boot_window3_0823_1733, seven
    # cutovers then a 22-minute drought, all twelve abandon lines naming the
    # same 120832.
    #
    # Every rank decodes this out of the SAME reduced payload in the SAME
    # round, so adopting it cannot make the ranks disagree -- it is the only
    # value in reach that makes them agree.
    #
    # A RUNG THAT CANNOT TAKE THE FLOOR MUST NOT RAISE INTO THE SEAM, AND MUST
    # NOT BE SILENT EITHER. A stub rung or a peer on an older tree has no such
    # method, and letting the AttributeError climb would kill a cutover over a
    # bookkeeping call. But swallowing it is the W9 failure -- 105 silent
    # AttributeError fallbacks that left a shipped fix inert with no log line
    # saying so. So it is skipped AND announced, once per process.
    if cap_relief is not None and verdict is not None:
        note = getattr(cap_relief, "note_group_backing_floor", None)
        if callable(note):
            note(int(verdict["min_backed_rows"]))
        elif not _GROUP_FLOOR_UNSUPPORTED_LOGGED:
            _GROUP_FLOOR_UNSUPPORTED_LOGGED = True
            logger.warning(
                "%s this rung cannot take the group backing floor (#833): %s "
                "has no note_group_backing_floor, so this rank will cap its "
                "exposure at its OWN backing. Under unequal pools that lets it "
                "issue ids a narrower peer cannot map, and every subsequent "
                "tp_to_pp will abandon on the union bound. This is the #816 "
                "behaviour, stated rather than assumed",
                LOG_PREFIX,
                type(cap_relief).__name__,
            )
    # #656 C22-d, THE SOURCE HALF: make the free-list ORDER a function of
    # MEMBERSHIP alone on EVERY rank, EVERY seam round, whatever the cap
    # agreement below decides to do. ``reconcile_to`` ends with this sort, but
    # it only runs when the group's exposed counts DISAGREE -- so the one
    # state nothing normalises is the state the counts are equal in, which is
    # exactly the state a rank that took a corridor-bounded ``recover()`` (and
    # therefore sorted) sits in next to peers that never shrank (and therefore
    # never did). See ``KvBackingRelief.normalize_free_lists``.
    if cap_relief is not None:
        try:
            cap_relief.normalize_free_lists()
        except Exception as e:
            logger.warning(
                "%s could not normalise this rank's free-list order (%s); the "
                "next allocation may hand out a different row id than a peer "
                "does, which the seam's live-slot agreement then has to repair",
                LOG_PREFIX,
                e,
            )
    target = kbr.collective_kv_shrink_ppm(reduced[:4])
    # #796: THE GROUP'S VERDICT, AND THIS RANK'S TERMS BEHIND IT, ON EVERY RANK.
    #
    # The shrink target is a MAX over per-rank floors, so the rank that decides
    # the outcome is frequently NOT the rank that needed the shrink -- and until
    # now only a rank that REFUSED ever printed its terms. On 2026-08-22 that
    # cost the flip outright: PP0 held a fundable plan (89119 rows of slack,
    # +1740 MiB deficit, SHRINK to 149126), PP2 was under no pressure at all and
    # its floor sat at its own cap, the group declined, and the whole boot
    # contains no evidence of the decline beyond a bare "returned NOTHING".
    #
    # Logged on EVERY rank, including the ones that fit, because the fitting
    # rank is exactly the one whose floor may be the veto. Logged at the same
    # cadence as the "returned NOTHING" line the gate already emits once per
    # seam, so a seam's funding story stays one grep and this adds no new
    # class of volume.
    try:
        logger.info(
            "%s KV shrink verdict (%s): %s | this rank: %s",
            LOG_PREFIX,
            direction,
            kbr.explain_kv_target(reduced[:4]),
            (
                # #796 FOLLOW-UP: on a GRANTED seam report what this rank will
                # APPLY, not what it proposed. The two diverge by design under a
                # proportional agreement -- a rank that asked for nothing still
                # pays its share -- and reusing the refusal summary here printed
                # "no change" on a rank that unmapped 702 MiB
                # (boot_798_0822_0646.log, 06:50:59Z). The declined path keeps
                # the proposal terms, which is what that reader needs.
                _rank_verdict_terms(relief, target)
                if relief is not None
                else "abstained from the shrink on this leg, so it has no terms"
            ),
        )
    except Exception as e:
        # THE GATE IS A NO-RETURN PATH: a raise from here climbs into the event
        # loop and takes the instance down (see _abandon_parked_flip). A
        # diagnostic may never be the thing that does that.
        logger.warning("%s could not report the KV shrink verdict: %s", LOG_PREFIX, e)
    if target is not None and relief is not None:
        # A SHRINK MAKES THE GROUP LEVEL BY CONSTRUCTION -- every rank caps to
        # the same absolute row target -- so the levelling has nothing to do
        # this round, and applying it from proposals made BEFORE the shrink
        # would act on a state that no longer exists.
        try:
            # #796: ``target`` is a PROPORTION of each rank's own cap, not a row
            # id. The group agrees the proportion; the rank converts it against
            # its own pool and its own floor, because the pools are uneven by
            # design and a row id is meaningless to a peer.
            return int(relief.apply_shrink_ppm(target))
        except Exception as e:
            logger.error(
                "%s KV relief failed to apply the agreed target %d: %s. This "
                "rank is now capped differently from its peers -- the cap "
                "agreement on a later round levels it.",
                LOG_PREFIX,
                target,
                e,
            )
            return 0
    levelled = apply_cap_agreement(scheduler, reduced[4:8])
    if levelled:
        logger.warning(
            "%s KV cap agreement moved this rank %+d exposed rows (%s). A "
            "rank that came back from a phase with fewer backed rows than its "
            "peers frames a different seam payload; the group now lives at "
            "the level its poorest rank can fund",
            LOG_PREFIX,
            int(levelled),
            direction,
        )
    return 0


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
        # THE PHASE IS A PRECONDITION, NOT AN ASSUMPTION (#656 successor 36).
        #
        # This provider's whole safety argument is that the drafter is
        # UNREACHABLE while the instance is in PP: no draft worker exists
        # there under strict purity, so decommitting its pages cannot be
        # observed. In TP the opposite holds -- the drafter is live and its
        # captured graphs point at those exact virtual addresses -- and the
        # only ``restore()`` in the tree is on the pp->tp cutover leg
        # (``PhaseFlipSpillLadder.on_enter_tp``). There is no lazy restore, so
        # a spill taken during TP would leave graphs replaying against
        # unbacked VA with nothing scheduled to fix it.
        #
        # That was survivable while the only caller was the seam gate, which
        # in practice only reaches this rung with PP current. It stopped being
        # survivable when the rebalance lender began spending the same ladder
        # on a 2 s clock in BOTH phases. The precondition the registration
        # comment already asserts is therefore enforced here rather than
        # trusted -- and enforced for every caller, because a TP spill is a
        # fault whoever asks for it.
        #
        # An UNKNOWN phase refuses. The carrier is not worth a guess.
        # Imported in the body, not at module scope: the runtime imports this
        # module, and the constant is not worth a cycle.
        from sglang.srt.managers.phase_flip_runtime import PHASE_PP

        if getattr(scheduler, "phase_flip_active_stack", None) != PHASE_PP:
            return 0
        ladder = getattr(scheduler, "phase_flip_spill_ladder", None)
        carrier = getattr(ladder, "_weights", None) if ladder is not None else None
        if carrier is None:
            carrier = carrier_of(getattr(scheduler, "draft_worker", None))
        if carrier is None:
            return 0
        from sglang.srt.managers.corridor_guard import draft_carrier_provider

        return draft_carrier_provider(carrier)(nbytes)

    return free_up_to


def _measured_seam_draw_mib(scheduler: Any, server_args: Any) -> int:
    """This rank's MEASURED seam draw in MiB, or 0 when none is on record.

    The seam reserve already measures and persists exactly this: the arena
    tail the cutover must re-commit plus the drafter's restore, both one-shot
    draws that happen while the seam runs. The gate's arming floor is defined
    as the law plus that draw, so reading it here is not a new measurement --
    it is the existing one finally reaching the number it was written for.

    Zero on anything unreadable: a cold boot with no record, no runtime, no
    rank. Zero leaves the shipped 512 MiB allowance in force, which is the
    previous behaviour exactly and the safe direction for a term that can only
    RAISE the floor.
    """
    try:
        from sglang.srt.managers import phase_flip_seam_reserve as seam

        runtime = getattr(scheduler, "phase_flip_runtime", None)
        rank = getattr(runtime, "_rank", None)
        if rank is None or server_args is None:
            return 0
        reserve = seam.read_seam_reserve(server_args, int(rank))
        if reserve is None or not reserve.active:
            return 0
        # #678: THE DRAW OF ONE LEG, not the sum of two cross-leg maxima.
        # total_fixed_bytes adds the arena tail (committed only on tp_to_pp)
        # to the drafter's restore (committed only on pp_to_tp) and so prices
        # a commit no seam makes -- 1595 MiB on this rig's rank 2 against a
        # worst leg of 1456. The sizer reads the same accessor, so the gate
        # and the pool cannot disagree about the number.
        return int(reserve.arming_draw_bytes()) // (1 << 20)
    except Exception as e:  # noqa: BLE001 - sizing must not raise
        logger.warning(
            "%s could not read this rank's measured seam draw (%s); the gate "
            "arms on the shipped allowance instead",
            LOG_PREFIX,
            e,
        )
        return 0


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

    model_runner = getattr(getattr(scheduler, "tp_worker", None), "model_runner", None)
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
    # #656 remediation: THE PAIR IS DERIVED AND DECLARED TOGETHER. The default
    # arming floor is the law plus the seam's expected draw (cg.arming_floor_
    # mib) rather than an unrelated number, and whichever way it is set, both
    # halves are logged in ONE line naming the reserve BETWEEN them. The
    # acceptance run armed at 1536 and judged at 1024 with nothing stating
    # that 512 MiB was supposed to be the seam's whole draw -- and the
    # measured draw was 1814-1852 MiB, so the gap between the two numbers was
    # exactly where five corridor breaches lived, unseen.
    # #781: THE FLAG NOW EXISTS AND WINS. `--phase-flip-corridor-floor-mib` is
    # a real ServerArgs field, so the hook this comment was left for is live and
    # the precedence is the right way round: flag first, env only as a
    # deprecated bridge.
    #
    # The order mattered more than it looks. SGLANG_CORRIDOR_FLOOR_MIB=1536 rode
    # in the boot env and silently overrode the corridor law that lives in code
    # (corridor_guard.CORRIDOR_LAW_MIB = 1024, band 819-1229) -- which is the
    # "armed at 1536, judged at 1024" split described just above. The env has
    # been removed from the boot env entirely; unset means the code law governs,
    # which is the intended state.
    configured = getattr(server_args, "phase_flip_corridor_floor_mib", None)
    if configured is None:
        _env_floor = os.environ.get(CORRIDOR_FLOOR_ENV)
        if _env_floor:
            try:
                from sglang.srt.environ import _warn_deprecated_env_to_cli_flag

                _warn_deprecated_env_to_cli_flag(
                    CORRIDOR_FLOOR_ENV, "--phase-flip-corridor-floor-mib"
                )
            except Exception:
                pass
            configured = _env_floor
    law_mib = cg.corridor_law_mib()
    # #662: THE RESERVE IS A MEASURED DRAW WHERE ONE EXISTS, and this rig has
    # one. `arming_floor_mib`'s own docstring says the reserve is "the MEASURED
    # draw a seam makes while it runs, where a measurement exists; the default
    # is the shipped allowance" -- but nothing was passing the measurement, so
    # the default 512 MiB was used on every boot that had a seam record sitting
    # on disk with the real number in it.
    #
    # Measured here 2026-08-15: the seam's own record puts the fixed draw at
    # 954 MiB on rank1 and 1595 MiB on rank2 (arena tail plus draft restore),
    # against a gate arming at 1024 + 512 = 1536. The gap between 1536 and
    # 1024 + 954 is exactly where the breach lived -- gpu0 reached 935 MiB at
    # the `weights_refill` stage while every allocation had cleared the gate.
    # That is the failure the threshold-pair log line below predicts in words
    # and the corridor audit predicts in its own text ("the arming floor is
    # the number to re-derive").
    #
    # HIGHEST WINS, never lowest. A raised arming floor makes the gate work
    # EARLIER and can only over-reclaim; a floor below the real draw launders
    # breaches as passed checks. So a configured value may raise the derived
    # one and may not lower it.
    measured_draw_mib = _measured_seam_draw_mib(scheduler, server_args)
    # #826: resolved reserve, not the shipped constant -- see the twin site
    # in phase_flip_seam_reserve.py. A flag that only moved the default
    # argument would be inert here, which is the defect being closed.
    derived_mib = cg.arming_floor_mib(
        seam_entry_reserve_mib=max(
            cg.seam_entry_reserve_mib_resolved(), measured_draw_mib
        ),
        law_mib=law_mib,
    )
    floor_mib = max(int(configured) if configured else 0, derived_mib)
    cg.check_threshold_pair(floor_mib, law_mib)
    logger.warning(
        "%s THRESHOLD PAIR on device %d: corridor LAW %d MiB (the verdict, "
        "and the only thing a refusal may be justified by), gate ARMS at %d "
        "MiB, i.e. the law plus a %d MiB seam-entry reserve%s. The reserve is "
        "what the gate assumes a seam DRAWS while it runs: if the measured "
        "draw exceeds it, entries clear this gate and still breach the law, "
        "and the breach lands in the gap between these two numbers where "
        "nothing looks. Measured on this rig (#656 acceptance): 1814-1852 MiB "
        "on the binding card.",
        LOG_PREFIX,
        int(device_index),
        law_mib,
        floor_mib,
        floor_mib - law_mib,
        (
            # #781: name the FLAG, since that is now the supported way to set
            # it. The env is still honoured as a deprecated bridge, but telling
            # an operator to go looking for an env var they should not be using
            # is how the 1536-vs-1024 split stayed alive.
            f" (set by --phase-flip-corridor-floor-mib/{CORRIDOR_FLOOR_ENV})"
            if configured and int(configured) >= derived_mib
            else (
                f" (derived from this rank's MEASURED seam draw of "
                f"{measured_draw_mib} MiB)"
                if measured_draw_mib
                else " (derived)"
            )
        ),
    )
    guard = cg.CorridorGuard(
        int(device_index),
        floor_mib=floor_mib,
        # The LAW never moves with the proof setting: a raised arming floor
        # must make the gate work EARLIER, never make it refuse allocations
        # the corridor permits. See CorridorGuard.__init__.
        law_floor_mib=law_mib,
        fleet_probe=cg.nvml_fleet_probe(),
    )

    # FIRST, AND ON BOTH LEGS: hand torch's unused cached blocks back to the
    # driver. The seam already did this, but inside _staging_affordable --
    # i.e. AFTER this gate had already formed its verdict. So the gate judged
    # against a free column that understated the truth by the size of the
    # hoard (1028-1426 MiB/card at idle on this rig), and could refuse a
    # pp->tp flip, the leg that starves decode outright, while a gibibyte of
    # nobody's memory sat on the card. It is registered in tier LOCAL because
    # it moves no payload anywhere and must precede everything that does.
    guard.register(
        "allocator-cache",
        _COST_ALLOCATOR_CACHE,
        cg.allocator_cache_provider(guard.free_bytes),
        tier=cg.RELIEF_LOCAL,
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

    # SPEC ITEM 12, THE DEVICE HALF: the KV pool's own unoccupied backing.
    #
    # This is the relief that funds the pp->tp leg, which was the deadlock
    # class: under strict purity the drafter is already spilled for the whole
    # PP phase, so the only REBALANCE provider returns 0 exactly when the
    # fatal leg needs it. The scheduler's pool IS the PP layout's pool, so
    # this is relief the binding leg can actually spend.
    #
    # IT IS DELIBERATELY NOT REGISTERED AS A GUARD PROVIDER. Every other
    # provider here answers a rank-local question -- "can I give bytes back" --
    # and may therefore run behind the gate's rank-local arm condition. The KV
    # cap answers a GROUP question, because it changes ``available_size()``
    # and therefore admission, and three ranks that answered it independently
    # wedged this instance (HANDOFF_675 §1a). So the target is agreed by one
    # reduction in ``collective_kv_backing_relief``, which ``_corridor_gate``
    # calls just BEFORE consulting this guard -- early enough that the bytes
    # it returns are money the guard's own probe sees, and unconditional
    # enough that a collective there is safe.
    #
    # The object is still built and cached here, because recovery on the
    # tp->pp leg reaches it through the same attribute.
    #
    # THIS IS NOT "A SMALLER POOL AS THE FIX", which the standing rule
    # forbids. The VA reservation does not move, the addresses a captured
    # graph baked in do not move, and the cap is lifted again on entry to PP.
    # What changes is PHYSICAL RESIDENCY, which is precisely what item 12
    # asks for: in VRAM lies exactly what has to lie there.
    try:
        from sglang.srt.managers.kv_backing_relief import kv_backing_provider

        relief = kv_backing_provider(
            scheduler,
            device_index=int(device_index),
            probe=guard.free_bytes,
            # THE LAW, never the arming floor. A proof run that raises the
            # floor must make the gate work earlier; it must not bound
            # recovery and leave the pool permanently smaller.
            law_floor_bytes=guard.law_floor_bytes,
        )
    except Exception as e:
        logger.warning("%s KV backing relief unavailable: %s", LOG_PREFIX, e)
        relief = None
    if relief is not None:
        setattr(scheduler, KV_BACKING_RELIEF_ATTR, relief)
        logger.info(
            "%s KV backing relief is available on device %d and will be driven "
            "by the COLLECTIVE target, not by the guard's ladder",
            LOG_PREFIX,
            int(device_index),
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
