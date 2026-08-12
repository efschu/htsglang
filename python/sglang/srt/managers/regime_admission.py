# SPDX-License-Identifier: Apache-2.0
"""Corridor admission for a #363 stage flip: price it, then ask the guard.

THE HOLE THIS CLOSES
--------------------
Before this module, a stage flip reached the actuators without ever passing
the corridor. That is not an inference; it is three facts about the tree:

* ``regime_runtime.RegimeObserver._act_interlocks`` runs four interlocks --
  selectability, dwell, group agreement, stale group timing. None of them is
  a memory admission.
* ``regime_act.RegimeActuator.apply`` calls ``vram_apply(want_vram)``, i.e.
  ``KvCapacityRuntime.apply_budget_request(budget_mib=...)``, directly.
* the dial's GROW path validates a new budget against exactly two bounds --
  ``min_viable_budget_bytes`` (the weights+graphs+state floor) and
  ``effective_budget_ceiling_bytes`` (the VA-reservation ceiling) -- and
  spends the corridor relief ladder ONLY when ``my_reduction > 0``, i.e. only
  when the budget SHRINKS.

So the one direction that consumes free VRAM was the one direction nobody
priced against the corridor law. A controller that grows a budget into the
1024 MiB per-card corridor breaches it silently, and the corridor is the
admission authority for exactly this class of move.

WHY THE CONTROLLER CALLS THE GUARD, AND NOT THE REVERSE
-------------------------------------------------------
Register C18 constrains the DIAL and the GUARD not to carry two floors: the
dial imports ``DEFAULT_FLOOR_MIB`` from the guard rather than restating it,
and the dial calls ``guard.ensure_headroom`` for its own reductions
(``vram_dial._corridor_relief``). The guard holds no reference to the dial in
either direction -- there is no import path from ``corridor_guard`` to
``vram_dial`` at all. This module follows that established direction: the
CALLER prices its move and asks the guard, which is also the shape the seam
already uses (``phase_flip_runtime._execute``: price the staging bytes, gate
them through the guard, then reduce the verdict across the group before any
rank moves). Nothing here inverts that.

THE PRICE
---------
Two terms, and neither may be guessed:

1. RESIDENCY DELTA -- how much more memory the target stage's budget vector
   asks this rank to hold than the current one does. Read from the stages
   themselves (``Stage.vram_budget_mib``), which is the same vector the #330
   dial is about to be handed, so the price and the move cannot disagree.

2. TRANSIENT -- the peak draw ABOVE residency that the target stage takes
   while serving, under THE LOAD STATE THIS RIG IS IN RIGHT NOW. A transient
   is a property of (layout, load state) jointly; borrowing another load
   state's number prices a deep-prefill burst with a decode figure and admits
   a move that then breaks the corridor under the load it was admitted for.
   The measured spread that makes this concrete is recorded in ``pp_cut``:
   the same rank on this rig drew 956 MiB in a deep-prefill A/B and 1989-3148
   MiB over a mixed soak.

REFUSING TO PRICE AT ZERO
-------------------------
An absent transient census, an empty per-load-state table, or a load state
missing from that table are all REFUSALS here, never a 0.0. An unpriced term
reads as free memory, a move priced as free is always affordable, and a move
that is always affordable is not gated. This is the same rule the planner
already enforces for its own inputs (an empty ``transient_by_load_state``
raises rather than meaning "no transient"), applied at the point where the
controller -- not the planner -- is about to spend memory.

GROUP UNIFORMITY
----------------
The guard is per-rank and its verdict is rank-local: one rank's card can be
tight while another's is clear. A rank-local refusal that stopped only that
rank would half-flip the group, which is the hang class the observe phase
existed to rule out. So the local verdict is packed and reduced with the
house MIN idiom -- the value and its negation in one payload, so a single MIN
yields both the group minimum and the group maximum -- and the flip proceeds
only if EVERY rank was admitted. A rank with no guard wired ABSTAINS loudly
(it refuses) rather than silently voting yes.

Every outcome is a named return. Nothing here raises into the scheduler loop:
a controller that turns a bad price into an exception turns a tight card into
a dead server.

STATUS: desk code. Pinned by
``test/registered/unit/managers/test_regime_admission_363.py``; not yet run on
metal. See ``docs/dev/363/TICKET_363_STAGE_CLOCK.md``.
"""

from __future__ import annotations

import dataclasses
import logging
from typing import Callable, Dict, List, Mapping, Optional, Tuple

logger = logging.getLogger(__name__)

LOG_PREFIX = "REGIME-ADMIT"

MIB = 1024 * 1024

__all__ = [
    "LOG_PREFIX",
    "AdmissionVerdict",
    "CorridorAdmission",
    "StageFlipPrice",
    "build_corridor_admission",
    "price_stage_flip",
]


@dataclasses.dataclass(frozen=True)
class StageFlipPrice:
    """What one stage flip costs this rank, in the two terms that matter."""

    stage: str
    #: The load state the transient was read for. Named in every log line so
    #: a reader can check it against the load the rig was actually under.
    load_state: str
    #: Extra residency the target budget vector asks for, MiB. Zero when the
    #: target's budget equals the current one (a KV-vector-only move).
    residency_delta_mib: float
    #: Peak draw above residency for the TARGET stage under ``load_state``.
    transient_mib: float
    #: What the guard is asked for.
    want_bytes: int
    detail: str

    @property
    def want_mib(self) -> float:
        return self.want_bytes / float(MIB)

    def as_dict(self) -> Dict:
        return {
            "stage": self.stage,
            "load_state": self.load_state,
            "residency_delta_mib": self.residency_delta_mib,
            "transient_mib": self.transient_mib,
            "want_bytes": self.want_bytes,
            "want_mib": self.want_mib,
            "detail": self.detail,
        }


@dataclasses.dataclass(frozen=True)
class AdmissionVerdict:
    """Admitted or refused, with the numbers that decided it."""

    ok: bool
    reason: str
    price: Optional[StageFlipPrice] = None
    #: This rank's own guard verdict, before the group reduction.
    local_ok: Optional[bool] = None
    #: True when the group reduction is what refused (some other rank was
    #: tight). Distinguished from a local refusal because the operator fix
    #: is different: a different card needs the room.
    group_refused: bool = False
    free_before_mib: Optional[float] = None
    free_after_mib: Optional[float] = None
    reclaimed_mib: Optional[float] = None

    def as_dict(self) -> Dict:
        return {
            "ok": self.ok,
            "reason": self.reason,
            "price": self.price.as_dict() if self.price else None,
            "local_ok": self.local_ok,
            "group_refused": self.group_refused,
            "free_before_mib": self.free_before_mib,
            "free_after_mib": self.free_after_mib,
            "reclaimed_mib": self.reclaimed_mib,
        }


def price_stage_flip(
    *,
    current,
    target,
    rank: int,
    load_state: Optional[str],
    transient_by_load_state: Optional[Mapping[str, float]],
) -> Tuple[Optional[StageFlipPrice], str]:
    """Price the move, or refuse to.

    Returns ``(price, "")`` or ``(None, reason)``. Never raises and never
    substitutes a zero for a term it could not read.

    ``transient_by_load_state`` is the TARGET stage's transient census, keyed
    by load state -- the same shape ``transient_census.TransientCensus.
    draw_mib()`` produces and ``pp_cut.RankResources`` consumes.
    """
    want_vram = tuple(int(x) for x in target.vram_budget_mib)
    have_vram = tuple(int(x) for x in current.vram_budget_mib)
    if len(want_vram) != len(have_vram):
        return None, (
            f"{LOG_PREFIX} cannot price {target.name!r}: its budget vector has "
            f"{len(want_vram)} entries and the stage in force ({current.name!r}) "
            f"has {len(have_vram)}. Two different group geometries are not two "
            f"stages of one table."
        )
    if not 0 <= rank < len(want_vram):
        return None, (
            f"{LOG_PREFIX} cannot price {target.name!r}: rank {rank} is outside "
            f"the {len(want_vram)}-entry budget vector, so there is no entry "
            f"that describes this rank's card."
        )

    residency_delta_mib = float(want_vram[rank] - have_vram[rank])

    # -- the transient, which may not be guessed -----------------------------
    if load_state is None:
        return None, (
            f"{LOG_PREFIX} refuses to price {target.name!r}: the current load "
            f"state is unknown, and a transient is a property of (stage, load "
            f"state) jointly. Pricing it under some other load state would "
            f"admit this move against a burst it was never measured for."
        )
    if not transient_by_load_state:
        return None, (
            f"{LOG_PREFIX} refuses to price {target.name!r}: no transient "
            f"census for it (load state {load_state!r}). An EMPTY table is not "
            f"'no transient', it is an UNMEASURED one, and an unpriced term "
            f"reads as free memory -- which would make this move admissible "
            f"by construction. Run the transient census "
            f"(SGLANG_TRANSIENT_CENSUS) for this stage or leave it out of the "
            f"acting set."
        )
    if load_state not in transient_by_load_state:
        known = ", ".join(sorted(transient_by_load_state)) or "none"
        return None, (
            f"{LOG_PREFIX} refuses to price {target.name!r} under load state "
            f"{load_state!r}: the census covers {known}. Substituting a "
            f"neighbouring load state's draw is the exact error the per-load-"
            f"state table exists to prevent -- on this rig the same rank drew "
            f"956 MiB in a deep-prefill A/B and 1989-3148 MiB over a mixed "
            f"soak."
        )
    transient_mib = float(transient_by_load_state[load_state])
    if transient_mib < 0.0:
        return None, (
            f"{LOG_PREFIX} refuses to price {target.name!r}: transient census "
            f"reports {transient_mib} MiB for load state {load_state!r}; a "
            f"negative draw is a broken census, not a credit."
        )

    # A shrink is refused upstream by the actuator, so a negative residency
    # delta is never a reason to ask the guard for LESS than the transient:
    # the transient is drawn whether or not residency moved.
    want_mib = max(0.0, residency_delta_mib) + transient_mib
    want_bytes = int(round(want_mib * MIB))
    detail = (
        f"stage {target.name!r} under load state {load_state!r}: residency "
        f"delta {residency_delta_mib:+.1f} MiB (rank {rank}: "
        f"{have_vram[rank]} -> {want_vram[rank]} MiB) + transient "
        f"{transient_mib:.1f} MiB = {want_mib:.1f} MiB wanted"
    )
    return (
        StageFlipPrice(
            stage=target.name,
            load_state=load_state,
            residency_delta_mib=residency_delta_mib,
            transient_mib=transient_mib,
            want_bytes=want_bytes,
            detail=detail,
        ),
        "",
    )


class CorridorAdmission:
    """The gate a #363 stage flip must pass before an actuator is touched.

    Fully injected: ``guard_fn`` returns this rank's ``CorridorGuard`` (or
    ``None``), ``collective_min`` is the same packed-int MIN channel the
    observer already uses, ``load_state_fn`` names the load state the rig is
    in, and ``transient_fn`` returns a stage's per-load-state census. Every
    one of them is a stub in the unit tests, so the whole path is exercised
    without a GPU.
    """

    def __init__(
        self,
        *,
        guard_fn: Optional[Callable[[], object]] = None,
        collective_min: Optional[Callable[[List[int]], List[int]]] = None,
        load_state_fn: Optional[Callable[[], Optional[str]]] = None,
        transient_fn: Optional[
            Callable[[object], Optional[Mapping[str, float]]]
        ] = None,
        rank: int = 0,
        tp_size: int = 1,
    ):
        self._guard_fn = guard_fn
        self._collective_min = collective_min
        self._load_state_fn = load_state_fn
        self._transient_fn = transient_fn
        self._rank = int(rank)
        self._tp_size = int(tp_size)
        self.admitted = 0
        self.refused = 0
        self.last: Optional[AdmissionVerdict] = None

    @property
    def wired(self) -> bool:
        return self._guard_fn is not None and self._transient_fn is not None

    def admit(self, current, target) -> AdmissionVerdict:
        """Price the flip and ask the corridor. Never raises."""
        load_state = self._load_state_fn() if self._load_state_fn else None
        transient = self._transient_fn(target) if self._transient_fn else None
        if self._transient_fn is None:
            return self._refuse(
                AdmissionVerdict(
                    ok=False,
                    reason=(
                        f"{LOG_PREFIX} refuses {target.name!r}: no transient "
                        f"census reader is wired, so the move would be priced "
                        f"on residency alone -- i.e. its peak draw would be "
                        f"priced at zero."
                    ),
                )
            )

        price, why = price_stage_flip(
            current=current,
            target=target,
            rank=self._rank,
            load_state=load_state,
            transient_by_load_state=transient,
        )
        if price is None:
            return self._refuse(AdmissionVerdict(ok=False, reason=why))

        guard = self._guard_fn() if self._guard_fn else None
        if guard is None:
            # ABSTAIN LOUDLY. Passing a move that could not be checked is the
            # boot-then-starve failure: the flip lands, the corridor breaks,
            # and the first evidence is an admission storm on another tenant.
            local_ok = False
            local_reason = (
                f"{LOG_PREFIX} refuses {price.detail}: no corridor guard on "
                f"this rank, so the corridor law could not be checked. A move "
                f"admitted without a check is not an admitted move."
            )
            result = None
        else:
            try:
                result = guard.ensure_headroom(
                    price.want_bytes,
                    reason=f"#363 stage flip -> {target.name}",
                )
            except Exception as e:  # the guard's own refusal path, or a probe
                return self._refuse(
                    AdmissionVerdict(
                        ok=False,
                        reason=(
                            f"{LOG_PREFIX} refuses {price.detail}: the corridor "
                            f"guard raised {type(e).__name__}: {e}"
                        ),
                        price=price,
                        local_ok=False,
                    )
                )
            local_ok = bool(getattr(result, "ok", False))
            local_reason = (
                f"{LOG_PREFIX} {'admits' if local_ok else 'refuses'} "
                f"{price.detail}; guard: {getattr(result, 'detail', '')}"
            )

        # -- group reduction: unanimous or nothing moves ----------------------
        group_ok, group_refused = self._reduce(local_ok)

        free_before = getattr(result, "free_before", None) if result else None
        free_after = getattr(result, "free_after", None) if result else None
        reclaimed = getattr(result, "reclaimed", None) if result else None
        verdict = AdmissionVerdict(
            ok=group_ok,
            reason=(
                local_reason
                if not group_refused
                else (
                    f"{LOG_PREFIX} refuses {price.detail}: this rank was "
                    f"admitted but at least one other rank in the group was "
                    f"not. A stage flip is group-uniform; admitting it here "
                    f"alone would half-flip the group."
                )
            ),
            price=price,
            local_ok=local_ok,
            group_refused=group_refused,
            free_before_mib=(free_before / MIB) if free_before is not None else None,
            free_after_mib=(free_after / MIB) if free_after is not None else None,
            reclaimed_mib=(reclaimed / MIB) if reclaimed is not None else None,
        )
        if group_ok:
            self.admitted += 1
            self.last = verdict
            logger.warning("%s", verdict.reason)
            return verdict
        return self._refuse(verdict)

    def _reduce(self, local_ok: bool) -> Tuple[bool, bool]:
        """(group_ok, group_refused_despite_local_ok).

        Packed-pair MIN: one reduction yields both the group minimum and the
        group maximum, so a single collective answers "did everyone agree?"
        and "what did they agree on?". With no channel on a multi-rank group
        the answer is REFUSE -- unlike the observer, which may legitimately
        classify uncoordinated because it acts on nothing, this path is about
        to spend memory.
        """
        vote = 1 if local_ok else 0
        if self._tp_size <= 1:
            return bool(local_ok), False
        if self._collective_min is None:
            return False, False
        reduced = self._collective_min([vote, -vote])
        if len(reduced) != 2:
            return False, False
        lo = int(reduced[0])
        group_ok = lo == 1
        return group_ok, bool(local_ok and not group_ok)

    def _refuse(self, verdict: AdmissionVerdict) -> AdmissionVerdict:
        self.refused += 1
        self.last = verdict
        logger.warning("%s", verdict.reason)
        return verdict

    def summary(self) -> Dict:
        return {
            "admitted": self.admitted,
            "refused": self.refused,
            "wired": self.wired,
            "last": self.last.as_dict() if self.last else None,
        }


def build_corridor_admission(scheduler, *, stage_transients=None):
    """Bind the gate to whatever this server actually wired.

    ``stage_transients`` maps stage name -> {load state: MiB}. It is supplied
    by the boot path from the transient census; when it is absent the gate is
    still built and every admission REFUSES with the missing-census reason,
    which is the intended behaviour: the operator learns that the census is
    missing from a named refusal rather than from a corridor breach.
    """
    def guard_fn(_s=scheduler):
        try:
            from sglang.srt.managers.phase_flip_spill import get_corridor_guard

            return get_corridor_guard(_s)
        except Exception as e:
            logger.warning(
                "%s no corridor guard available (%s: %s); every stage-flip "
                "admission will refuse",
                LOG_PREFIX,
                type(e).__name__,
                e,
            )
            return None

    def load_state_fn(_s=scheduler):
        # The load state the transient census is keyed by. The observer's
        # phase label is the honest source: it is what the census itself is
        # bucketed by, and it is replicated across the group.
        obs = getattr(_s, "regime_observer", None)
        rec = getattr(obs, "last_record", None) if obs else None
        if isinstance(rec, dict):
            return rec.get("regime")
        return None

    transient_fn = None
    if stage_transients is not None:

        def transient_fn(stage, _t=stage_transients):
            return _t.get(stage.name)

    tp_size = int(getattr(scheduler, "tp_size", 1) or 1)
    rank = int(getattr(scheduler, "tp_rank", 0) or 0)

    collective_min = None
    cpu_group = getattr(scheduler, "tp_cpu_group", None)
    if cpu_group is not None and tp_size > 1:
        try:
            from sglang.srt.managers.kv_pressure_runtime import (
                default_collective_min,
            )

            collective_min = default_collective_min(cpu_group)
        except Exception:
            collective_min = None

    gate = CorridorAdmission(
        guard_fn=guard_fn,
        collective_min=collective_min,
        load_state_fn=load_state_fn,
        transient_fn=transient_fn,
        rank=rank,
        tp_size=tp_size,
    )
    logger.info(
        "%s stage-flip corridor admission armed (rank %d of %d): "
        "transient census %s",
        LOG_PREFIX,
        rank,
        tp_size,
        "wired" if transient_fn is not None else "MISSING (all flips refuse)",
    )
    return gate
