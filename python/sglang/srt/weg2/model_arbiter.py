# SPDX-License-Identifier: Apache-2.0
"""Dual-model flip: the model arbiter's switch policy (pure, stdlib only).

Two models (e.g. Qwen3.8-27B-INT8 and Qwen3.8-Next-Flash-INT4) are both booted
on the same three cards, and only one of them computes at a time. The arbiter
is a host side-car that is NOT in the request path: each model keeps its own
front, requests for the sleeping model wait in that front, and the side-car
polls both fronts' ``GET /weg2/state`` and calls ``/weg2/model_sleep`` /
``/weg2/model_wake`` (DUAL-MODEL-FLIP-KONZEPT-1001.md section 8.5).

This module is the decision only: one :class:`Snapshot` in, one
:class:`Decision` out, no clock, no I/O. Rules, in order (section 4):

1. a switch in progress holds;
2. nobody awake (cold start) -> wake the model with the oldest waiter;
3. no waiter on any sleeping model -> stay;
4. dwell: less than ``T_min`` since the last switch -> stay, where
   ``T_min = max(t_min_floor_s, t_min_factor * measured switch time)``,
   clamped to ``T_max`` so the bound below survives a slow switch;
5. idle switch: the awake model has nothing outstanding -> switch;
6. anti-starvation: the oldest waiter waited ``>= T_max`` -> switch and park
   the awake model's running decodes (the park mechanism exists);
7. optional time slice: both busy and ``>= slice_s`` since the last switch ->
   switch with park;
8. otherwise stay.

Bound: no request waits longer than ``T_max`` plus one switch time, because
rule 6 fires at ``T_max`` and rule 4 can never hold longer than ``T_max``.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple

ACTIONS = ("stay", "switch", "hold")


class ArbiterConfigRefused(ValueError):
    """A configuration that would break the T_max bound, named."""


@dataclass(frozen=True)
class ArbiterConfig:
    t_max_s: float = 90.0
    t_min_factor: float = 5.0
    t_min_floor_s: float = 10.0
    slice_s: Optional[float] = None

    def __post_init__(self) -> None:
        if not self.t_max_s > 0:
            raise ArbiterConfigRefused(f"t_max_s must be > 0, got {self.t_max_s}")
        if self.t_min_factor < 0:
            raise ArbiterConfigRefused(f"t_min_factor must be >= 0, got {self.t_min_factor}")
        if self.t_min_floor_s < 0 or self.t_min_floor_s > self.t_max_s:
            raise ArbiterConfigRefused(
                f"t_min_floor_s={self.t_min_floor_s} must lie in [0, t_max_s={self.t_max_s}]: "
                f"a dwell above T_max breaks the wait bound")
        if self.slice_s is not None and (self.slice_s <= 0 or self.slice_s < self.t_min_floor_s):
            raise ArbiterConfigRefused(
                f"slice_s={self.slice_s} must be > 0 and >= t_min_floor_s={self.t_min_floor_s}: "
                f"a slice shorter than the dwell floor would pendulum")


@dataclass(frozen=True)
class ModelLoad:
    #: running + queued requests of this model, as its front reports them
    outstanding: int = 0
    #: age of the oldest request still waiting for this model (0 when none)
    oldest_wait_s: float = 0.0


@dataclass(frozen=True)
class Snapshot:
    now: float
    #: the model that computes right now, or None (both asleep / cold start)
    awake: Optional[str]
    last_switch_at: float
    switching: bool
    #: the last measured model-switch wall time, seconds
    switch_s: float
    models: Dict[str, ModelLoad] = field(default_factory=dict)


@dataclass(frozen=True)
class Decision:
    action: str
    target: Optional[str]
    reason: str
    park: bool = False
    note: str = ""


def t_min_s(cfg: ArbiterConfig, switch_s: float) -> Tuple[float, str]:
    """The dwell and, when it had to be clamped to T_max, why."""
    t = max(cfg.t_min_floor_s, cfg.t_min_factor * max(0.0, float(switch_s)))
    if t > cfg.t_max_s:
        return cfg.t_max_s, (f"T_min {t:.1f}s (factor {cfg.t_min_factor} x switch {switch_s:.1f}s) "
                             f"clamped to T_max {cfg.t_max_s:.1f}s")
    return t, ""


def _oldest_waiter(models: Dict[str, ModelLoad], exclude: Optional[str]) -> Optional[str]:
    cands = [(m.oldest_wait_s, m.outstanding, name) for name, m in models.items()
             if name != exclude and m.outstanding > 0]
    if not cands:
        return None
    return max(cands)[2]


def decide(s: Snapshot, cfg: ArbiterConfig) -> Decision:
    if s.switching:
        return Decision("hold", None, "switching")
    if s.awake is None:
        tgt = _oldest_waiter(s.models, None)
        if tgt is None:
            return Decision("stay", None, "no-demand")
        return Decision("switch", tgt, "cold-start")
    tgt = _oldest_waiter(s.models, s.awake)
    if tgt is None:
        return Decision("stay", None, "no-demand")
    dwell = s.now - s.last_switch_at
    t_min, note = t_min_s(cfg, s.switch_s)
    if dwell < t_min:
        return Decision("stay", None, "dwell", note=note)
    awake_load = s.models.get(s.awake, ModelLoad())
    if awake_load.outstanding == 0:
        return Decision("switch", tgt, "idle", park=False, note=note)
    if s.models[tgt].oldest_wait_s >= cfg.t_max_s:
        return Decision("switch", tgt, "t_max", park=True, note=note)
    if cfg.slice_s is not None and dwell >= cfg.slice_s:
        return Decision("switch", tgt, "slice", park=True, note=note)
    return Decision("stay", None, "busy", note=note)
