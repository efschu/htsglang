# SPDX-License-Identifier: Apache-2.0
"""The proposal-to-actuator path of the #363 controller (phase 3, desk half).

THIS MODULE IS THE ONLY PLACE THE CONTROLLER TOUCHES AN ACTUATOR, and it is a
separate module for exactly that reason. ``regime_runtime`` -- the observer --
imports nothing from here at module scope, so the F4 property established in
phase 2 survives the arrival of the action path as a fact about the import
graph rather than as a promise: a unit test walks both modules' syntax trees
and asserts that observe cannot reach #297 or #330 even now that #363 can.

WHAT IT DOES
------------
Takes a stage the classifier proposed and the boot table judged selectable,
and issues the two actuator calls that move a stage's reachable axes:

* ``KvReshardRuntime.arm(vector, source)`` -- #297. Replicated call; commits
  at the next group-wide idle consensus boundary. Arming is not moving: the
  bytes move later, or the arm is superseded.
* ``KvCapacityRuntime.apply_budget_request(...)`` -- #330, GROW ONLY. A shrink
  flushes the radix cache and ``vram_dial`` reserves that for an explicit
  dial, so the controller may hand VRAM back to a pool and may never take it
  away.

WHAT IT REFUSES
---------------
Every refusal is a named return, not an exception: a controller that raises
inside the scheduler loop turns a bad proposal into a dead server. The caller
logs the reason and holds.

1. The stage is not selectable per the boot table (weights differ, or the KV
   vector was never declared).
2. No actuator is wired for the axis the stage needs.
3. The move would shrink a VRAM budget.
4. The actuator itself refused (``arm`` returns ``(False, msg)``); the message
   is passed through verbatim.

The actuators are INJECTED, so the whole path is hermetically testable with
stubs and the production wiring is one function at the bottom.
"""

from __future__ import annotations

import dataclasses
import logging
from typing import Callable, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

LOG_PREFIX = "REGIME-ACT"

#: Source tag handed to #297 so a reshard's provenance is readable in its own
#: log line: the pressure ladder and the regime controller both arm the same
#: runtime, and "which of you did this" must not need a timestamp comparison.
ARM_SOURCE = "regime_controller"

__all__ = [
    "ARM_SOURCE",
    "LOG_PREFIX",
    "ActionResult",
    "RegimeActuator",
    "build_regime_actuator",
]


@dataclasses.dataclass(frozen=True)
class ActionResult:
    """What one proposal did, or why it did nothing."""

    stage: str
    #: True only when at least one actuator accepted work.
    armed: bool
    #: Per-axis outcome, e.g. {"kv": "armed 2,11,10", "vram": "not wired"}.
    detail: Dict[str, str]
    reason: str

    def as_dict(self) -> Dict:
        return {
            "stage": self.stage,
            "armed": self.armed,
            "detail": dict(self.detail),
            "reason": self.reason,
        }


class RegimeActuator:
    """Issues the reachable half of a stage flip. Injectable end to end.

    ``reshard_arm`` mirrors ``KvReshardRuntime.arm``: ``(vector, source) ->
    (ok, msg)``. ``vram_apply`` mirrors the dial's budget request:
    ``(budgets_mib) -> (ok, msg)``. Either may be ``None``, which is the
    honest state of a server booted without ``--kv-reshard-vectors`` or
    without ``--enable-vram-dial``, and the refusal says which one is missing.
    """

    def __init__(
        self,
        *,
        reshard_arm: Optional[
            Callable[[Tuple[int, ...], str], Tuple[bool, str]]
        ] = None,
        vram_apply: Optional[Callable[[Tuple[int, ...]], Tuple[bool, str]]] = None,
        selectable_fn: Optional[Callable[[str], Tuple[bool, str]]] = None,
    ):
        self._reshard_arm = reshard_arm
        self._vram_apply = vram_apply
        self._selectable_fn = selectable_fn
        self.arms = 0
        self.refusals = 0
        self.last: Optional[ActionResult] = None

    @property
    def wired_axes(self) -> List[str]:
        axes = []
        if self._reshard_arm is not None:
            axes.append("kv")
        if self._vram_apply is not None:
            axes.append("vram")
        return axes

    def apply(self, stage, current) -> ActionResult:
        """Move what can be moved from ``current`` to ``stage``.

        ``stage`` and ``current`` are :class:`regime_classifier.Stage`. Never
        raises: every failure is a named :class:`ActionResult`.
        """
        detail: Dict[str, str] = {}

        if self._selectable_fn is not None:
            ok, why = self._selectable_fn(stage.name)
            if not ok:
                return self._refuse(stage, detail, why)

        want_kv = tuple(int(x) for x in stage.kv_token_vector)
        have_kv = tuple(int(x) for x in current.kv_token_vector)
        want_vram = tuple(int(x) for x in stage.vram_budget_mib)
        have_vram = tuple(int(x) for x in current.vram_budget_mib)

        needs_kv = want_kv != have_kv
        needs_vram = want_vram != have_vram
        if not needs_kv and not needs_vram:
            return self._refuse(
                stage,
                detail,
                f"stage {stage.name!r} differs from the current configuration "
                f"in no reachable axis; nothing to arm",
            )

        # A shrink is never autonomous: it flushes the radix cache, and #330
        # reserves that for an explicit dial (vram_dial's own rule). Checked
        # BEFORE anything is armed, so a mixed move cannot half-commit.
        if needs_vram and any(w < h for w, h in zip(want_vram, have_vram)):
            return self._refuse(
                stage,
                detail,
                f"stage {stage.name!r} would SHRINK a VRAM budget "
                f"({list(have_vram)} -> {list(want_vram)}). A shrink flushes "
                f"the radix cache and only an explicit dial authorizes that; "
                f"the controller may grow a budget and never take one away.",
            )

        if needs_kv and self._reshard_arm is None:
            return self._refuse(
                stage,
                detail,
                f"stage {stage.name!r} needs a KV reshard to {list(want_kv)} "
                f"and no #297 runtime is wired (boot with "
                f"--kv-reshard-vectors covering that vector)",
            )
        if needs_vram and self._vram_apply is None:
            return self._refuse(
                stage,
                detail,
                f"stage {stage.name!r} needs a VRAM budget change to "
                f"{list(want_vram)} and no #330 dial is wired (boot with "
                f"--enable-vram-dial)",
            )

        armed = False
        # VRAM first when it GROWS: a bigger pool cannot make a reshard fail,
        # while a reshard into a pool that is about to grow would be sized
        # against the smaller one.
        if needs_vram:
            ok, msg = self._vram_apply(want_vram)
            detail["vram"] = f"{'applied' if ok else 'REFUSED'}: {msg}"
            armed = armed or ok
        if needs_kv:
            ok, msg = self._reshard_arm(want_kv, ARM_SOURCE)
            detail["kv"] = f"{'armed' if ok else 'REFUSED'}: {msg}"
            armed = armed or ok

        result = ActionResult(
            stage=stage.name,
            armed=armed,
            detail=detail,
            reason=(
                f"stage {stage.name!r}: "
                + "; ".join(f"{k} {v}" for k, v in sorted(detail.items()))
            ),
        )
        self.last = result
        if armed:
            self.arms += 1
            logger.warning("%s %s", LOG_PREFIX, result.reason)
        else:
            self.refusals += 1
            logger.warning("%s no axis accepted -- %s", LOG_PREFIX, result.reason)
        return result

    def _refuse(self, stage, detail: Dict[str, str], why: str) -> ActionResult:
        self.refusals += 1
        result = ActionResult(stage=stage.name, armed=False, detail=detail, reason=why)
        self.last = result
        logger.warning("%s refused: %s", LOG_PREFIX, why)
        return result

    def summary(self) -> Dict:
        return {
            "arms": self.arms,
            "refusals": self.refusals,
            "wired_axes": self.wired_axes,
            "last": self.last.as_dict() if self.last else None,
        }


def build_regime_actuator(scheduler, table_plan=None) -> RegimeActuator:
    """Bind the actuator to whatever this server actually wired.

    An axis with no runtime is left ``None`` rather than stubbed, so the
    refusal names the missing flag instead of the move silently doing half of
    what was asked.
    """
    reshard_arm = None
    reshard_rt = getattr(scheduler, "kv_reshard_runtime", None)
    if reshard_rt is not None:

        def reshard_arm(vector, source, _rt=reshard_rt):
            return _rt.arm(vector, source=source)

    vram_apply = None
    vram_rt = getattr(scheduler, "kv_capacity_runtime", None)
    if vram_rt is not None:

        def vram_apply(budgets, _rt=vram_rt):
            return _rt.apply_budget_request(budget_mib=list(budgets))

    selectable_fn = None
    if table_plan is not None:
        selectable_fn = table_plan.is_selectable

    actuator = RegimeActuator(
        reshard_arm=reshard_arm,
        vram_apply=vram_apply,
        selectable_fn=selectable_fn,
    )
    logger.info(
        "%s armed: wired axes %s",
        LOG_PREFIX,
        ", ".join(actuator.wired_axes) or "NONE (every proposal will be refused)",
    )
    return actuator
