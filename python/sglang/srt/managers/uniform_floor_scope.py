# Copyright 2026 SGLang Team
# SPDX-License-Identifier: Apache-2.0
"""#824: is the rank-uniformity floor actually covering the ranks that matter?

WHAT THE FLOORS ARE. #603/#610 made the decode-mem and prefill-admission
DECISIONS rank-uniform; #616g closed the level below them, because eviction
was still "an explicitly local side effect". Its own docstring states the
chain that costs:

    `evict_from_tree_cache` fires on `available_size() < num_tokens`, where
    the DEMAND is replicated but the AVAILABILITY is this rank's own shard.
    The rank with the roomiest pool declines to evict while the tight ranks
    evict, so the radix trees stop being replicas. `match_prefix` then
    returns a rank-dependent prefix, `prepare_for_extend` computes
    `extend_num_tokens` from it, and every per-layer TP all_reduce of that
    forward is entered with a rank-dependent token count.

THE SCOPE THIS MODULE IS ABOUT. All three floors are published off ONE
MIN-reduce over ``tp_cpu_group`` (`Scheduler._update_uniform_pool_budget`).
On a TP=1/PP=3 boot that group has a single member on every rank, so the
reduce is a no-op and every floor switches OFF -- while three PP ranks go on
evicting from purely local availability. `scheduler.py`'s own comment at that
site says it plainly: "With pp_size>1 the ranks that must agree are NOT in
this reduce group", and calls the condition "the measured cause of a pipeline
deadlock".

WHY A MODULE FOR SOMETHING THIS SMALL. Because under ``--enable-phase-flip``
the scope is not a boot constant. `phase_flip_runtime` rebuilds the TP group
per phase (``want_tp_size = n if tp_phase else 1``), so ``tp_cpu_group``
holds 3 members in the TP decode phase and 1 in the PP prefill phase: the
floors go ON and OFF at every cutover. The site reported that with a
once-per-process latch, so a boot that flipped four times in 55 s said it
once, at the first PP iteration, and never again.

Specimen /spinning/evidence-816-18f/wedge_0823_055757 (boot 0516): the log
carries the scope line exactly three times, once per rank, all reading
``world=1 -> floors OFF ... pp_size=3 tp_size=1``. Nineteen seconds later the
TP replicas' queue-head digests had parted (#791b), and by 05:56:18 the three
ranks were building prefill batches with #cached-token 0 against 16384 -- a
rank-dependent prefix match, which is the exact fingerprint the #616g
docstring predicts. Whether the floors were on or off during any particular
one of those four flips is not recoverable from that log.

This module holds only the DECISION -- when a scope change is worth a line --
as a pure function, so it can be driven directly in a test. The #823 lesson,
applied without having to relearn it: with this logic inline behind a real
``all_reduce``, the only thing a test could check is whether the source still
mentions a transition branch, and a source grep cannot tell a live branch
from ``elif False:``.

THIS MODULE ENFORCES NOTHING. It does not widen the reduce group, and it does
not change a single admission or eviction decision. Making the floors
actually cover the PP ranks is a change to what the group agrees on, which
needs a metal lane; this only makes the gap legible while that is decided.
"""

from __future__ import annotations

from typing import Optional, Tuple

__all__ = [
    "SCOPE_ON",
    "SCOPE_OFF",
    "scope_for_world",
    "scope_transition",
    "report_scope",
]

#: The floors are live: the reduce group has more than one member, so a MIN
#: over it actually constrains anything.
SCOPE_ON = "on"

#: The floors are inert: a one-member (or absent) reduce group. A MIN over a
#: singleton returns the rank's own value, so every floor published from it is
#: that rank's local number wearing a group's name.
SCOPE_OFF = "off"


def scope_for_world(world: Optional[int]) -> str:
    """Is a reduce over a group of this size capable of constraining anything?

    ``None`` (no group at all) reads as OFF for the same reason a singleton
    does: there is no peer whose value could lower this rank's own.
    """
    if world is None:
        return SCOPE_OFF
    return SCOPE_ON if int(world) > 1 else SCOPE_OFF


def scope_transition(
    previous: Optional[str], world: Optional[int]
) -> Tuple[str, Optional[str]]:
    """Return ``(scope, event)`` for this observation.

    ``event`` is the scope to report and is ``None`` when there is nothing new
    to say. The FIRST observation always reports, so a boot states its
    starting coverage; after that only genuine changes do.

    Reporting transitions rather than latching once is the whole point. Under
    a phase flip the scope alternates, and a latched report cannot distinguish
    "the floors were off for the entire run" from "the floors were off for the
    prefill half of every one of four cutovers" -- which are different
    situations with different fixes. It also cannot say when coverage came
    BACK, and a gap that closes and a gap that never closes call for opposite
    responses.

    Pure arithmetic on the previous value: no clock, no counter, no I/O, and
    nothing a rank could observe differently from its peers.
    """
    scope = scope_for_world(world)
    if previous == scope:
        return scope, None
    return scope, scope


def report_scope(holder, world: Optional[int], logger) -> None:
    """Log a scope CHANGE for ``holder``, if this observation is one.

    A MODULE FUNCTION AND NOT A `Scheduler` METHOD, deliberately. The first
    cut was a method, and it broke 61 tests across six files at once: the
    uniformity-floor suites drive `_update_uniform_pool_budget` bound onto
    `_FakeScheduler` stand-ins that bind only the methods they name, so a new
    `self._report_...` call raised AttributeError on every one of them. That
    is the same binding trap `_pp_wait_for_proxy_readiness`' alias comment
    documents for the PP holders. Reaching for the holder through ``getattr``
    from a module function needs no binding at all, so the stand-ins keep
    working untouched -- and the alternative, a `getattr(self, ..., None)`
    guard at the call site, would have been the silently-skip shape that hides
    a rename in production.

    State lives on the holder so each rank keeps its own; everything is read
    defensively because the stand-ins are deliberately partial objects.
    """
    scope, event = scope_transition(
        getattr(holder, "_uniform_floor_scope", None), world
    )
    holder._uniform_floor_scope = scope
    if event is None:
        return
    args = getattr(holder, "server_args", None)
    pp = int(getattr(args, "pp_size", 1) or 1) if args is not None else 1
    tp = int(getattr(args, "tp_size", 1) or 1) if args is not None else 1
    if event == SCOPE_OFF:
        logger.info(
            "#788 UNIFORM-FLOOR SCOPE: tp_cpu_group world=%s -> floors OFF "
            "(evict/host/mamba). pp_size=%d tp_size=%d. With pp_size>1 the "
            "ranks that must agree are NOT in this reduce group.",
            "none" if world is None else int(world),
            pp,
            tp,
        )
    else:
        logger.info(
            "#788 UNIFORM-FLOOR SCOPE: tp_cpu_group world=%d -> floors ON "
            "(evict/host/mamba). pp_size=%d tp_size=%d. The ranks that must "
            "agree are in this reduce group again.",
            int(world),
            pp,
            tp,
        )
