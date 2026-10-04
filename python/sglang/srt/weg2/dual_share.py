"""DUAL-TP3PP3 P/D SHARE: the selectable and dynamic precedence of P vs D (item 800).

User order 03.10. ~18:55-19:15Z: in the dual layout (27B NVFP4, P prefill PP3 and
D decode TP3 at once on the same cards, per MPS) the precedence P vs D is
SELECTABLE, and also DYNAMIC from (waiting prefill tokens, active decode bs):
many waiting tokens + D bs1 -> P gets more; few waiting + a large D bs -> D gets
more. Everything hardware-generic: SM counts, granularity, driver and MPS state
are read at runtime; a missing mechanism prints ONE named line
``W-DUAL-SHARE-FALLBACK mech=... card=... reason=...`` and the next cheaper
mechanism takes over, never silently.

MEASURED (03.10., 27B NVFP4 dual, P chunked_prefill_size=1024): D round 47.5 ms
without P, 72 ms with light P, 170 ms with P at full load. Research basis:
deskq/done/760-greenctx-mps.out (stages 1-4, control 5.1-5.5, matrix 3/6).

Shape of this module (pure stdlib; torch only inside the D-capture helper):

* the FRONT runs a :class:`ShareController` (mode, characteristic, hysteresis)
  every ``tick_s`` and writes its rung into ``<busy file>.ctl``
  (:class:`CtlWriter`, atomic rename like ``dual_duty.DBusyWriter``);
* the P ranks read it (:class:`CtlReader`, at most every 20 ms, a stale or
  missing file = P full) and act through the ACTUATORS of stage 2:
  ``chunk`` -- PP0's chunk cap (:class:`ChunkCap`, applied after
  ``Scheduler.dynamic_chunked_prefill_size``; only PP0 decides, downstream
  stages run PP0's #791 extents) and ``duty`` -- the existing duty throttle
  with the rung's fraction as its duty (:class:`ShareDuty`);
* stage 1 switches, each alone, default off: the D CUDA-graph capture on a
  high-priority stream (:func:`d_capture_stream`) and P's MPS client priority
  (launcher env, see ``launcher.dual_priority_env``);
* stage 3 (green-context ladder in P) lives in ``weg2/dual_green.py`` (item 1330) behind its own switch
  (``--dual-green-ladder``, default off): the ``green`` actuator without the switch still prints the named
  fallback below; with it the chunk cap and the duty actuator step aside for every rung the ladder serves
  (:func:`green_serves`) and stay as the named fallback for the rest. :func:`sm_ladder` is the older pure
  floor-rounding arithmetic; the ladder itself takes the DRIVER's group sizes (which round up).

Modes (``--dual-priority``): ``p`` = rung 0 always (P full, today's physics);
``balanced`` / ``d`` = a static rung while D holds decodes (Env ``..._STATIC``,
default balanced=2 (0.5), d=3 (0.25)); ``dynamic`` = the matrix tau-class x
D-bs-class. In every mode a D with no decodes gives P everything at once
(``reason=d_idle``): P loses nothing while D is empty.

No flip/NF/27B-INT8 path imports this module unless the launcher armed it: the
P side needs ``SGLANG_WEG2_DUAL_SHARE_CTL``, the D side
``SGLANG_WEG2_DUAL_D_CAPTURE_PRIO=1`` on group D, the front ``--dual-priority``.
"""

from __future__ import annotations

import hashlib
import logging
import math
import os
import statistics
import time
from collections import deque
from dataclasses import dataclass, field, replace
from typing import Callable, Deque, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

logger = logging.getLogger(__name__)

LOG_TAG = "DUAL-SHARE"
FALLBACK = "W-DUAL-SHARE-FALLBACK"
MODES = ("p", "balanced", "d", "dynamic")
ACTUATORS = ("chunk", "duty", "green")

#: P side: the control file the front writes (``<dbusy>.ctl``).
CTL_ENV = "SGLANG_WEG2_DUAL_SHARE_CTL"
#: P side: comma list of the actuators (subset of ACTUATORS).
ACT_ENV = "SGLANG_WEG2_DUAL_SHARE_ACTUATORS"
#: P side: the captured P prefill graph buckets (comma list), for the cap snap.
BUCKETS_ENV = "SGLANG_WEG2_DUAL_SHARE_BUCKETS"
#: P side: "1" = the duty actuator is armed (read by the PP forward site).
DUTY_ENV = "SGLANG_WEG2_DUAL_SHARE_DUTY"
#: D side (stage 1a): "1" = capture D's CUDA graphs on a high-priority stream.
D_CAPTURE_PRIO_ENV = "SGLANG_WEG2_DUAL_D_CAPTURE_PRIO"
#: Profile knobs (front side), all relative units; see :func:`config_from_env`.
ENV_PREFIX = "SGLANG_WEG2_DUAL_SHARE_"
#: STAGE 3 (weg2/dual_green.py): "1" = the green-context ladder serves the ``green`` actuator (P env, set by the
#: launcher from ``--dual-green-ladder``; default off = this module behaves exactly as without stage 3).
GREEN_LADDER_ENV = "SGLANG_WEG2_DUAL_GREEN_LADDER"

#: set by ``dual_green.maybe_arm`` once a ladder serves rungs: ``fn(rung) -> bool`` (True = the green stream
#: already throttles this rung, so the chunk cap and the duty actuator must NOT throttle a second time).
#: None (default, every process that never armed the ladder) = they act as before.
_GREEN_SERVES: Optional[Callable[[int], bool]] = None


def set_green_serves(fn: Optional[Callable[[int], bool]]) -> None:
    global _GREEN_SERVES
    _GREEN_SERVES = fn


def green_serves(rung: int) -> bool:
    fn = _GREEN_SERVES
    if fn is None:
        return False
    try:
        return bool(fn(int(rung)))
    except Exception:  # noqa: BLE001 - a question to the ladder never stops a pass
        return False

READ_EVERY_S = 0.02
#: A single duty pause never exceeds this (a stuck signal must not stall P).
MAX_SLEEP_S = 0.5


def ctl_path(tag: str) -> str:
    """``<busy>.ctl`` next to the duty signal (``dual_duty.dbusy_path``)."""
    return f"/dev/shm/wdb-{hashlib.sha1(str(tag).encode()).hexdigest()[:10]}.ctl"


def fallback_line(mech: str, card: str, reason: str) -> str:
    return f"{FALLBACK} mech={mech} card={card or 'all'} reason={reason}"


def parse_actuators(raw: str) -> Tuple[str, ...]:
    out = []
    for a in (raw or "").split(","):
        a = a.strip().lower()
        if not a:
            continue
        if a not in ACTUATORS:
            raise ValueError(f"{LOG_TAG}: unknown actuator {a!r} (allowed: {', '.join(ACTUATORS)})")
        if a not in out:
            out.append(a)
    if not out:
        raise ValueError(f"{LOG_TAG}: no actuator named (allowed: {', '.join(ACTUATORS)})")
    return tuple(out)


# ------------------------------------------------------------------ configuration

@dataclass(frozen=True)
class ShareConfig:
    """The characteristic and the hysteresis. Every threshold is relative:
    rungs are fractions of P's share, tau is seconds of P work, b is the
    fraction of D's seats in use."""

    rungs: Tuple[float, ...] = (1.0, 0.75, 0.5, 0.25)
    #: tau class edges in seconds of P work: < lo = low, > hi = high.
    tau_edges_s: Tuple[float, float] = (2.0, 10.0)
    #: b class edges as a fraction of D seats (B=0 is its own class).
    b_edges: Tuple[float, float] = (0.25, 0.60)
    #: rows tau class (low, mid, high), cols b class (0, low, mid, high).
    matrix: Tuple[Tuple[int, ...], ...] = ((0, 1, 2, 3), (0, 1, 2, 2), (0, 0, 1, 2))
    #: the static modes' rung while D holds decodes.
    static: Tuple[Tuple[str, int], ...] = (("balanced", 2), ("d", 3))
    deadband: float = 0.20
    dwell_to_d_s: float = 0.5
    dwell_to_p_s: float = 2.0
    ewma_s: float = 2.0
    tick_s: float = 0.25
    d_min_rate_tps: float = 0.0
    rate_band: float = 0.10
    rate_relax_s: float = 5.0
    p_min_share: float = 0.25
    starve_age_s: float = 60.0
    starve_max_rung: int = 1
    flap_window_s: float = 10.0
    heartbeat_s: float = 1.0
    stale_s: float = 5.0

    def __post_init__(self):
        r = tuple(float(x) for x in self.rungs)
        object.__setattr__(self, "rungs", r)
        if not r or abs(r[0] - 1.0) > 1e-9:
            raise ValueError(f"{LOG_TAG}: rung 0 must be 1.0 (P full), got {r}")
        if any(not (0.0 < x <= 1.0) for x in r) or any(b >= a for a, b in zip(r, r[1:])):
            raise ValueError(f"{LOG_TAG}: rungs must fall strictly inside (0, 1], got {r}")
        lo, hi = (float(x) for x in self.tau_edges_s)
        if not (0.0 < lo < hi):
            raise ValueError(f"{LOG_TAG}: tau edges need 0 < lo < hi, got {self.tau_edges_s}")
        blo, bhi = (float(x) for x in self.b_edges)
        if not (0.0 < blo < bhi <= 1.0):
            raise ValueError(f"{LOG_TAG}: b edges need 0 < lo < hi <= 1, got {self.b_edges}")
        if len(self.matrix) != 3 or any(len(row) != 4 for row in self.matrix):
            raise ValueError(f"{LOG_TAG}: matrix must be 3 rows (tau low/mid/high) x 4 cols "
                             f"(b 0/low/mid/high), got {self.matrix}")
        for row in self.matrix:
            for k in row:
                if not (0 <= int(k) < len(r)):
                    raise ValueError(f"{LOG_TAG}: matrix rung {k} outside 0..{len(r) - 1}")
        for m, k in self.static:
            if m not in MODES or not (0 <= int(k) < len(r)):
                raise ValueError(f"{LOG_TAG}: static rung {m}={k} invalid")
        if not (0.0 <= self.deadband < 1.0):
            raise ValueError(f"{LOG_TAG}: deadband {self.deadband} outside [0, 1)")
        if not (0.0 < self.p_min_share <= 1.0):
            raise ValueError(f"{LOG_TAG}: p_min_share {self.p_min_share} outside (0, 1]")
        if self.d_min_rate_tps < 0.0:
            raise ValueError(f"{LOG_TAG}: d_min_rate_tps {self.d_min_rate_tps} < 0")
        if self.dwell_to_d_s < 0 or self.dwell_to_p_s < 0 or self.tick_s <= 0 or self.ewma_s < 0:
            raise ValueError(f"{LOG_TAG}: dwell/tick/ewma must be >= 0 (tick > 0)")

    def static_rung(self, mode: str) -> Optional[int]:
        for m, k in self.static:
            if m == mode:
                return int(k)
        return None

    def max_rung(self) -> int:
        """The deepest rung P's share may fall to (``p_min_share`` floor)."""
        k = 0
        for i, f in enumerate(self.rungs):
            if f + 1e-9 >= self.p_min_share:
                k = i
        return k


def _floats(raw: str, n: Optional[int] = None) -> Tuple[float, ...]:
    vals = tuple(float(x) for x in raw.split(",") if x.strip())
    if n is not None and len(vals) != n:
        raise ValueError(f"{LOG_TAG}: expected {n} values, got {raw!r}")
    return vals


def config_from_env(env: Mapping[str, str], *, d_min_rate_tps: float = 0.0,
                    p_min_share: float = 0.25) -> ShareConfig:
    """The profile's knobs (``SGLANG_WEG2_DUAL_SHARE_<NAME>``); unset = default.

    RUNGS "1,0.75,0.5,0.25"; TAU_EDGES_S "2,10"; B_EDGES "0.25,0.6";
    MATRIX "0,1,2,3;0,1,2,2;0,0,1,2" (rows tau low;mid;high); STATIC
    "balanced=2,d=3"; DEADBAND; DWELL_TO_D_S; DWELL_TO_P_S; EWMA_S; TICK_S;
    RATE_BAND; RATE_RELAX_S; STARVE_AGE_S; STARVE_MAX_RUNG; FLAP_WINDOW_S;
    HEARTBEAT_S; STALE_S."""
    def g(name: str) -> str:
        return str(env.get(ENV_PREFIX + name, "") or "").strip()

    kw: Dict[str, object] = {"d_min_rate_tps": float(d_min_rate_tps), "p_min_share": float(p_min_share)}
    if g("RUNGS"):
        kw["rungs"] = _floats(g("RUNGS"))
    if g("TAU_EDGES_S"):
        kw["tau_edges_s"] = _floats(g("TAU_EDGES_S"), 2)
    if g("B_EDGES"):
        kw["b_edges"] = _floats(g("B_EDGES"), 2)
    if g("MATRIX"):
        kw["matrix"] = tuple(tuple(int(x) for x in row.split(",")) for row in g("MATRIX").split(";"))
    if g("STATIC"):
        kw["static"] = tuple((kv.split("=")[0].strip(), int(kv.split("=")[1]))
                             for kv in g("STATIC").split(",") if "=" in kv)
    for name, key, typ in (("DEADBAND", "deadband", float), ("DWELL_TO_D_S", "dwell_to_d_s", float),
                           ("DWELL_TO_P_S", "dwell_to_p_s", float), ("EWMA_S", "ewma_s", float),
                           ("TICK_S", "tick_s", float), ("RATE_BAND", "rate_band", float),
                           ("RATE_RELAX_S", "rate_relax_s", float), ("STARVE_AGE_S", "starve_age_s", float),
                           ("STARVE_MAX_RUNG", "starve_max_rung", int),
                           ("FLAP_WINDOW_S", "flap_window_s", float), ("HEARTBEAT_S", "heartbeat_s", float),
                           ("STALE_S", "stale_s", float)):
        if g(name):
            kw[key] = typ(g(name))
    return ShareConfig(**kw)


# ------------------------------------------------------------------ the controller

def _classify(value: float, edges: Sequence[float], prev: Optional[int], deadband: float) -> int:
    """Class index 0..len(edges) of ``value`` with a dead band around each edge:
    leaving the previous class needs the edge crossed by ``deadband`` (relative)."""
    raw = sum(1 for e in edges if value > e)
    if prev is None or raw == prev:
        return raw
    if raw > prev:
        # moving up: every edge between must be exceeded by the band
        k = prev
        while k < raw and value > edges[k] * (1.0 + deadband):
            k += 1
        return k
    k = prev
    while k > raw and value < edges[k - 1] * (1.0 - deadband):
        k -= 1
    return k


@dataclass
class Decision:
    mode: str
    rung: int
    prev_rung: int
    target: int
    fraction: float
    tau_s: Optional[float]
    b: int
    seats: int
    q_tokens: float
    p_rate_tps: Optional[float]
    d_rate_tps: Optional[float]
    reason: str
    changed: bool
    held: bool
    flaps: int
    dwell_s: float


class ShareController:
    """Front side. ``observe(...)`` once per tick returns the :class:`Decision`.

    Hysteresis: a step toward D (deeper rung) needs ``dwell_to_d_s`` since the
    last change, a step toward P ``dwell_to_p_s``; at most one rung per
    decision; the tau and b classes carry a +-``deadband`` band; Q, b and the D
    rate are EWMA-smoothed over ``ewma_s``. Immediate (no dwell, any distance):
    D idle -> rung 0, the starvation clamp, an operator mode change. Flaps (a
    reversal inside ``flap_window_s``) are counted and logged, never refused."""

    def __init__(self, cfg: ShareConfig, mode: str, clock: Callable[[], float] = time.monotonic):
        if mode not in MODES:
            raise ValueError(f"{LOG_TAG}: mode {mode!r} not in {MODES}")
        self.cfg = cfg
        self.mode = mode
        self._clock = clock
        self.rung = 0
        self.last_change_t = -1e9
        self._last_dir = 0
        self.flaps = 0
        self.changes = 0
        self._q = None
        self._b = None
        self._r = None
        self._t = None
        self._tau_cls: Optional[int] = None
        self._b_cls: Optional[int] = None
        self._rate_floor = 0
        self._rate_ok_since: Optional[float] = None
        self._rate_step_t = -1e9
        self._mode_jump = False
        self._held_key = None

    # -- operator --------------------------------------------------------
    def set_mode(self, mode: str) -> bool:
        if mode not in MODES:
            raise ValueError(f"{LOG_TAG}: mode {mode!r} not in {MODES}")
        if mode == self.mode:
            return False
        self.mode = mode
        self._mode_jump = True
        self._rate_floor = 0
        return True

    def set_limits(self, *, d_min_rate_tps: Optional[float] = None,
                   p_min_share: Optional[float] = None) -> None:
        kw = {}
        if d_min_rate_tps is not None:
            kw["d_min_rate_tps"] = float(d_min_rate_tps)
        if p_min_share is not None:
            kw["p_min_share"] = float(p_min_share)
        if kw:
            self.cfg = replace(self.cfg, **kw)

    # -- smoothing -------------------------------------------------------
    def _ewma(self, old: Optional[float], new: Optional[float], dt: float) -> Optional[float]:
        if new is None:
            return old
        if old is None or self.cfg.ewma_s <= 0.0:
            return float(new)
        a = 1.0 - math.exp(-max(0.0, dt) / self.cfg.ewma_s)
        return old + a * (float(new) - old)

    # -- the characteristic ----------------------------------------------
    def base_target(self, tau_s: Optional[float], b: int, seats: int, b_smooth: float) -> Tuple[int, str]:
        cfg = self.cfg
        if self.mode == "p":
            return 0, "mode_p"
        if b <= 0:
            return 0, "d_idle"
        st = cfg.static_rung(self.mode)
        if self.mode in ("balanced", "d"):
            return (st if st is not None else 0), f"static_{self.mode}"
        frac = (b_smooth / seats) if seats > 0 else 1.0
        self._b_cls = _classify(frac, cfg.b_edges, self._b_cls, cfg.deadband)
        if tau_s is None:
            tcls, tname = 1, "unknown"
            self._tau_cls = None
        else:
            self._tau_cls = _classify(tau_s, cfg.tau_edges_s, self._tau_cls, cfg.deadband)
            tcls, tname = self._tau_cls, ("low", "mid", "high")[self._tau_cls]
        bcol = 1 + self._b_cls  # col 0 is B=0
        return int(cfg.matrix[tcls][bcol]), f"matrix(tau={tname},b={('low', 'mid', 'high')[self._b_cls]})"

    def observe(self, *, q_tokens: float, b: int, seats: int, p_rate_tps: Optional[float],
                d_rate_tps: Optional[float], oldest_age_s: float = 0.0) -> Decision:
        cfg = self.cfg
        now = self._clock()
        dt = 0.0 if self._t is None else now - self._t
        self._t = now
        self._q = self._ewma(self._q, max(0.0, float(q_tokens)), dt)
        self._b = self._ewma(self._b, max(0, int(b)), dt)
        self._r = self._ewma(self._r, d_rate_tps if (d_rate_tps or 0) > 0 else None, dt)
        tau = (self._q / p_rate_tps) if (p_rate_tps and p_rate_tps > 0) else None
        target, reason = self.base_target(tau, int(b), int(seats), float(self._b or 0.0))
        immediate = reason == "d_idle" or self._mode_jump
        # D minimum-rate guard (any mode but p, only while D decodes)
        if self.mode != "p" and cfg.d_min_rate_tps > 0 and b > 0 and self._r is not None:
            lo = cfg.d_min_rate_tps * (1.0 - cfg.rate_band)
            hi = cfg.d_min_rate_tps * (1.0 + cfg.rate_band)
            if self._r < lo:
                self._rate_ok_since = None
                if now - self._rate_step_t >= cfg.dwell_to_d_s and self._rate_floor <= self.rung:
                    self._rate_floor = min(cfg.max_rung(), self.rung + 1)
                    self._rate_step_t = now
                reason += "+rate_low"
            elif self._r > hi:
                if self._rate_ok_since is None:
                    self._rate_ok_since = now
                elif now - self._rate_ok_since >= cfg.rate_relax_s and self._rate_floor > 0:
                    self._rate_floor -= 1
                    self._rate_ok_since = now
                    reason += "+rate_relax"
            else:
                self._rate_ok_since = None
            if self._rate_floor > target:
                target = self._rate_floor
                reason += "+rate_floor"
        if b <= 0:
            self._rate_floor = 0
            self._rate_ok_since = None
        if self.mode != "p" and oldest_age_s > cfg.starve_age_s and target > cfg.starve_max_rung:
            target = cfg.starve_max_rung
            reason += "+starve"
            immediate = True
        if target > cfg.max_rung():
            target = cfg.max_rung()
            reason += "+p_min_share"
        prev = self.rung
        held = False
        if target != self.rung:
            since = now - self.last_change_t
            if immediate:
                self.rung = target
            elif target > self.rung:
                if since >= cfg.dwell_to_d_s:
                    self.rung += 1
                else:
                    held = True
            else:
                if since >= cfg.dwell_to_p_s:
                    self.rung -= 1
                else:
                    held = True
        self._mode_jump = False
        changed = self.rung != prev
        dwell = now - self.last_change_t if self.last_change_t > -1e8 else 0.0
        if changed:
            d = 1 if self.rung > prev else -1
            if self._last_dir and d != self._last_dir and dwell < cfg.flap_window_s:
                self.flaps += 1
            self._last_dir = d
            self.last_change_t = now
            self.changes += 1
        if held:
            reason += "+dwell_hold"
        return Decision(mode=self.mode, rung=self.rung, prev_rung=prev, target=target,
                        fraction=cfg.rungs[self.rung], tau_s=tau, b=int(b), seats=int(seats),
                        q_tokens=float(self._q or 0.0), p_rate_tps=p_rate_tps, d_rate_tps=self._r,
                        reason=reason, changed=changed, held=held, flaps=self.flaps, dwell_s=dwell)

    def should_log(self, d: Decision) -> bool:
        """Every change; a held target once per (rung, target) pair."""
        if d.changed:
            self._held_key = None
            return True
        if d.held:
            key = (d.rung, d.target)
            if key != self._held_key:
                self._held_key = key
                return True
        return False


def decision_line(d: Decision, actuators: Iterable[str]) -> str:
    """The ONE measurement line per decision (grep ``DUAL-SHARE mode=``)."""
    tau = "n/a" if d.tau_s is None else f"{d.tau_s:.2f}s"
    rate = "n/a" if d.d_rate_tps is None else f"{d.d_rate_tps:.1f}"
    prate = "n/a" if d.p_rate_tps is None else f"{d.p_rate_tps:.0f}"
    return (f"{LOG_TAG} mode={d.mode} tau={tau} dbs={d.b}/{d.seats} q={d.q_tokens:.0f} "
            f"p_rate={prate} d_rate={rate} rung={d.prev_rung}->{d.rung} f={d.fraction:.2f} "
            f"target={d.target} actuator={','.join(actuators)} reason={d.reason} "
            f"flaps={d.flaps} dwell_s={d.dwell_s:.2f}")


# ------------------------------------------------------------------ front-side meters

class PRateMeter:
    """P's full rate for tau: uncached tokens over P's own prefill seconds
    (compute-honest, ``weg2_prefill_s``; the leg wall only when absent), from
    legs that ran while D was idle (the FULL rate); without such a sample the
    p90 of all legs, named in ``source``."""

    def __init__(self, window: int = 32, min_idle: int = 3):
        self.idle: Deque[float] = deque(maxlen=window)
        self.any: Deque[float] = deque(maxlen=window)
        self.min_idle = int(min_idle)

    def note(self, tokens: int, seconds: Optional[float], d_idle: bool) -> None:
        if not seconds or seconds <= 0 or tokens <= 0:
            return
        r = float(tokens) / float(seconds)
        self.any.append(r)
        if d_idle:
            self.idle.append(r)

    def rate(self) -> Tuple[Optional[float], str]:
        if len(self.idle) >= self.min_idle:
            return statistics.median(self.idle), f"idle_median(n={len(self.idle)})"
        if self.any:
            s = sorted(self.any)
            return s[min(len(s) - 1, int(0.9 * len(s)))], f"p90_all(n={len(s)})"
        return None, "none"


class DRateMeter:
    """D's per-request decode rate from the leg-2 token stream.

    The front sees SSE EVENTS, not tokens; a finished leg gives the exact
    completion tokens, so tokens-per-event is calibrated on completion
    (EWMA) and a live request's rate is its events in the last ``window_s``
    times that ratio. ``rate()`` = median over live requests with at least
    ``min_age_s`` of stream, else the median of the recently completed ones."""

    def __init__(self, window_s: float = 2.0, min_age_s: float = 1.0, done_window: int = 16,
                 clock: Callable[[], float] = time.monotonic):
        self.window_s = float(window_s)
        self.min_age_s = float(min_age_s)
        self._clock = clock
        self._live: Dict[str, Tuple[float, Deque[Tuple[float, int]], int]] = {}
        self._done: Deque[float] = deque(maxlen=done_window)
        self.tok_per_event = 1.0

    def chunk(self, rid: str, events: int) -> None:
        if events <= 0:
            return
        now = self._clock()
        ent = self._live.get(rid)
        if ent is None:
            ent = (now, deque(), 0)
        t0, dq, n = ent
        dq.append((now, int(events)))
        while dq and now - dq[0][0] > self.window_s:
            dq.popleft()
        self._live[rid] = (t0, dq, n + int(events))

    def done(self, rid: str, completion_tokens: int, seconds: Optional[float] = None) -> None:
        ent = self._live.pop(rid, None)
        now = self._clock()
        if ent is not None:
            t0, _dq, n = ent
            if n > 0 and completion_tokens > 0:
                self.tok_per_event += 0.2 * (completion_tokens / n - self.tok_per_event)
            span = now - t0
        else:
            span = seconds or 0.0
        if completion_tokens > 1 and span > 0:
            self._done.append(completion_tokens / span)

    def drop(self, rid: str) -> None:
        self._live.pop(rid, None)

    def prune(self, active) -> None:
        """Forget live requests D no longer holds (an aborted leg 2 never calls done)."""
        for rid in [r for r in self._live if r not in active]:
            self._live.pop(rid, None)

    def rate(self) -> Optional[float]:
        now = self._clock()
        live = []
        for t0, dq, _n in self._live.values():
            if now - t0 < self.min_age_s:
                continue
            while dq and now - dq[0][0] > self.window_s:
                dq.popleft()
            span = min(self.window_s, now - t0)
            ev = sum(e for _, e in dq)
            live.append(ev * self.tok_per_event / span if span > 0 else 0.0)
        if live:
            return statistics.median(live)
        if self._done:
            return statistics.median(self._done)
        return None


def sse_events(chunk: bytes) -> int:
    """SSE events in a streamed chunk (``\\n\\n`` separators); a non-SSE body counts 1."""
    n = chunk.count(b"\n\n")
    return n if n > 0 else (1 if chunk else 0)


# ------------------------------------------------------------------ the control file

def format_ctl(seq: int, d: Decision) -> str:
    tau_ms = -1 if d.tau_s is None else int(d.tau_s * 1000)
    rate = -1 if d.d_rate_tps is None else int(d.d_rate_tps * 100)
    extra = f" starve={1 if getattr(d, 'starve', False) else 0}" if getattr(d, "green", False) else ""
    return (f"v1 seq={seq} mode={d.mode} rung={d.rung} f={d.fraction:.4f} b={d.b} seats={d.seats} "
            f"q={int(d.q_tokens)} tau_ms={tau_ms} r_x100={rate}{extra}\n")


def parse_ctl(text: str) -> Optional[Dict[str, str]]:
    parts = text.strip().split()
    if not parts or parts[0] != "v1":
        return None
    out = {}
    for p in parts[1:]:
        if "=" in p:
            k, v = p.split("=", 1)
            out[k] = v
    return out if {"rung", "f", "b"} <= set(out) else None


class CtlWriter:
    """Front side: rewrite on a change of rung/mode/b-busy or every heartbeat."""

    def __init__(self, path: str, heartbeat_s: float = 1.0, clock: Callable[[], float] = time.monotonic):
        self.path = path
        self.heartbeat_s = float(heartbeat_s)
        self._clock = clock
        self.seq = 0
        self._last_key = None
        self._last_t = -1e9

    def update(self, d: Decision) -> bool:
        key = (d.mode, d.rung, d.b > 0)
        now = self._clock()
        if key == self._last_key and now - self._last_t < self.heartbeat_s:
            return False
        self.seq += 1
        tmp = f"{self.path}.{os.getpid()}.tmp"
        with open(tmp, "w") as f:
            f.write(format_ctl(self.seq, d))
        os.replace(tmp, self.path)
        self._last_key, self._last_t = key, now
        return True


@dataclass
class CtlState:
    rung: int = 0
    fraction: float = 1.0
    d_busy: bool = False
    mode: str = "p"
    seq: int = -1
    ok: bool = False
    starve: bool = False


class CtlReader:
    """P side: the front's rung, read at most every ``READ_EVERY_S``. A
    missing, unreadable or stale file (mtime older than ``stale_s``) is P
    FULL (rung 0) -- P never stalls on a signal that is not there -- and is
    named once per episode (``W-DUAL-SHARE-FALLBACK mech=ctl``)."""

    def __init__(self, path: str, stale_s: float = 5.0, clock: Callable[[], float] = time.monotonic,
                 wall: Callable[[], float] = time.time, log: Callable[[str], None] = logger.warning,
                 card: str = "all"):
        self.path = path
        self.stale_s = float(stale_s)
        self._clock = clock
        self._wall = wall
        self._log = log
        self.card = card
        self._read_t = -1e9
        self.state = CtlState()
        self._named = False

    def _fail(self, why: str) -> CtlState:
        if not self._named:
            self._named = True
            self._log(fallback_line("ctl", self.card, f"{why} -> rung 0 (P full) until the front writes again"))
        self.state = CtlState()
        return self.state

    def read(self) -> CtlState:
        now = self._clock()
        if now - self._read_t < READ_EVERY_S:
            return self.state
        self._read_t = now
        try:
            st = os.stat(self.path)
            with open(self.path) as f:
                txt = f.read(512)
        except OSError as e:
            return self._fail(f"ctl unreadable ({type(e).__name__})")
        age = self._wall() - st.st_mtime
        if age > self.stale_s:
            return self._fail(f"ctl stale (age {age:.1f}s > {self.stale_s:.1f}s)")
        kv = parse_ctl(txt)
        if kv is None:
            return self._fail("ctl malformed")
        try:
            self.state = CtlState(rung=int(kv["rung"]), fraction=max(0.0, min(1.0, float(kv["f"]))),
                                  d_busy=int(kv["b"]) > 0, mode=kv.get("mode", "?"),
                                  seq=int(kv.get("seq", -1)), ok=True,
                                  starve=int(kv.get("starve", "0")) > 0)
        except (ValueError, KeyError):
            return self._fail("ctl malformed")
        self._named = False
        return self.state


# ------------------------------------------------------------------ P actuators

def chunk_width_for(fraction: float, base: int, page: int, buckets: Sequence[int] = ()) -> Tuple[int, str]:
    """The chunk cap of one rung: ``fraction`` of the configured width, floored
    to the page, at least one page. A captured graph bucket in (cap/2, cap]
    is taken instead (graph replay instead of the eager host floor); without
    one the cap runs eager, and the second value says so."""
    base = int(base)
    page = max(1, int(page or 1))
    if fraction >= 1.0 or base <= 0:
        return base, "full"
    raw = int(math.floor(fraction * base / page)) * page
    cap = max(page, raw)
    fits = [int(b) for b in buckets if cap / 2.0 < int(b) <= cap and int(b) % page == 0]
    if fits:
        return max(fits), "graph"
    return cap, "eager"


class ChunkCap:
    """PP0 only: ``cap(width)`` = min(width, the rung's cap). Never widens."""

    def __init__(self, reader: CtlReader, base: int, page: int, buckets: Sequence[int] = (),
                 log: Callable[[str], None] = logger.info):
        self.reader = reader
        self.base = int(base)
        self.page = max(1, int(page or 1))
        self.buckets = tuple(sorted(int(b) for b in buckets))
        self._log = log
        self._last = None
        self.capped = 0

    def ladder(self, rungs: Sequence[float]) -> List[Tuple[float, int, str]]:
        return [(f,) + chunk_width_for(f, self.base, self.page, self.buckets) for f in rungs]

    def cap(self, width: int) -> int:
        st = self.reader.read()
        if st.fraction >= 1.0 or width is None or int(width) <= 0:
            out = width
            kind = "full"
        elif green_serves(st.rung):
            out = width               # the green ladder already throttles this rung: no second throttle
            kind = "green"
        else:
            c, kind = chunk_width_for(st.fraction, self.base, self.page, self.buckets)
            out = min(int(width), c)
            if out < int(width):
                self.capped += 1
        key = (st.rung, kind, st.ok)
        if key != self._last:
            self._last = key
            self._log(f"{LOG_TAG} P chunk cap rung={st.rung} f={st.fraction:.2f} width={width}->{out} "
                      f"({kind}) mode={st.mode} seq={st.seq} capped_total={self.capped}")
        return out


def maybe_chunk_cap(env: Mapping[str, str], *, first_pp_rank: bool, chunked_prefill_size: Optional[int],
                    page: Optional[int], planner_buckets: Sequence[int] = (),
                    log: Callable[[str], None] = logger.info,
                    warn: Callable[[str], None] = logger.warning) -> Optional[ChunkCap]:
    """The scheduler's construct step: None unless the launcher armed the share
    control with the ``chunk`` actuator, and None on every rank but PP0 (the
    downstream stages run PP0's forwarded extents, #791)."""
    path = str(env.get(CTL_ENV, "") or "").strip()
    if not path:
        return None
    acts = parse_actuators(str(env.get(ACT_ENV, "chunk") or "chunk"))
    if "chunk" not in acts and "green" not in acts:
        return None
    if not first_pp_rank:
        return None
    if "green" in acts:
        if str(env.get(GREEN_LADDER_ENV, "") or "").strip() == "1":
            log(f"{LOG_TAG} P green ladder armed (weg2/dual_green.py): the chunk cap stays only as the named "
                "fallback for rungs the ladder cannot serve" + (" (duty likewise)" if "duty" in acts else ""))
        else:
            warn(fallback_line("green", "all", "stage 3 (green-context ladder) not built -- waits for metal "
                               "probe M1 -> chunk cap" + (" + duty" if "duty" in acts else "")))
    if not chunked_prefill_size or int(chunked_prefill_size) <= 0:
        warn(fallback_line("chunk", "all", "no --chunked-prefill-size on P (no chunk to cap) -> "
                           + ("duty" if "duty" in acts else "none (P full)")))
        return None
    raw = str(env.get(BUCKETS_ENV, "") or "")
    buckets = sorted({int(x) for x in raw.split(",") if x.strip()} | {int(b) for b in planner_buckets})
    cap = ChunkCap(CtlReader(path, log=warn), int(chunked_prefill_size), int(page or 1), buckets, log=log)
    lad = cap.ladder(ShareConfig().rungs)
    log(f"{LOG_TAG} P chunk cap armed (PP0) ctl={path} base={cap.base} page={cap.page} buckets={list(buckets)} "
        "ladder(default rungs)=" + ", ".join(f"{f:.2f}->{w} {k}" for f, w, k in lad)
        + " -- costedge cap (760 4.3) not built: fixed fractions")
    return cap


def apply_chunk_cap(sched, width):
    """The scheduler's delegate: ``width`` unless PP0 holds an armed cap. A
    ``--p-layer-split dynamic`` leader owns width AND cut jointly, so it is
    left alone (named once)."""
    cap = getattr(sched, "_dual_share_chunk", None)
    if cap is None:
        return width
    try:
        from sglang.srt.weg2 import p_layer_split_runtime as _pls_rt  # noqa: F401
        _pls = _pls_rt.active()
        if _pls is not None and getattr(_pls, "leader", None) is not None:
            if not getattr(cap, "_pls_named", False):
                cap._pls_named = True
                logger.warning(fallback_line("chunk", "all", "--p-layer-split dynamic owns the width "
                                             "(joint chunk+cut plan) -> no cap, duty if armed"))
            return width
    except ImportError:
        pass
    return cap.cap(width)


class ShareDuty:
    """The ``duty`` actuator: ``dual_duty.DutyThrottle`` semantics with the
    rung's fraction as the duty and the ctl's b as the busy signal. Rung 0 or
    D idle: no pause."""

    def __init__(self, reader: CtlReader, sleep: Callable[[float], None] = time.sleep):
        self.reader = reader
        self._sleep = sleep
        self._last_fwd_s = 0.0
        self.slept_s = 0.0
        self.throttled = 0
        self.duty = 1.0

    def pause_s(self) -> float:
        st = self.reader.read()
        self.duty = st.fraction
        if self._last_fwd_s <= 0.0 or not st.d_busy or st.fraction >= 1.0 or st.fraction <= 0.0:
            return 0.0
        if green_serves(st.rung):
            return 0.0                # the green ladder already throttles this rung (duty = its fallback)
        return min(MAX_SLEEP_S, self._last_fwd_s * (1.0 - st.fraction) / st.fraction)

    def before_forward(self) -> float:
        s = self.pause_s()
        if s > 0.0:
            self._sleep(s)
            self.slept_s += s
            self.throttled += 1
        return s

    def after_forward(self, seconds: float) -> None:
        self._last_fwd_s = max(0.0, float(seconds))

    @classmethod
    def from_env(cls, env=None) -> Optional["ShareDuty"]:
        env = os.environ if env is None else env
        if str(env.get(DUTY_ENV, "") or "").strip() != "1":
            return None
        path = str(env.get(CTL_ENV, "") or "").strip()
        if not path:
            return None
        return cls(CtlReader(path))


# ------------------------------------------------------------------ stage 1a: D capture priority

def d_capture_armed(env: Optional[Mapping[str, str]] = None) -> bool:
    """Only on a group-D rank whose launcher set the switch (dual layout only)."""
    e = os.environ if env is None else env
    return (str(e.get(D_CAPTURE_PRIO_ENV, "") or "").strip() == "1"
            and str(e.get("SGLANG_WEG2_GROUP", "") or "").strip().upper() == "D")


def pick_capture_priority(prange: Optional[Tuple[int, int]]) -> Tuple[Optional[int], str]:
    """(priority, why) from the device's runtime range (least, greatest).
    Lower numbers are higher priority; one level only = no priority there."""
    if prange is None:
        return None, "stream priority range unreadable"
    least, greatest = int(prange[0]), int(prange[1])
    if greatest == least:
        return None, f"device reports one stream priority level ({least})"
    return greatest, f"range least={least} greatest={greatest}"


_capture_named = set()


def d_capture_stream(env: Optional[Mapping[str, str]] = None):
    """STAGE 1a: the stream ``parallel_state.graph_capture`` captures on, or
    None (= the stock ``Stream()``). PyTorch instantiates graphs with
    ``cudaGraphInstantiateFlagUseNodePriority``: a node keeps the priority of
    its CAPTURE stream, so replaying a priority-0 graph on a high-priority
    stream changes nothing (760 2.4) -- the priority must be there at capture."""
    if not d_capture_armed(env):
        return None
    import torch

    dev = torch.cuda.current_device()
    card = "?"
    try:
        card = str(getattr(torch.cuda.get_device_properties(dev), "uuid", dev))
    except Exception:  # noqa: BLE001 - the name of a card never stops a capture
        pass
    try:
        prange = tuple(torch.cuda.Stream.priority_range())
    except Exception:  # noqa: BLE001
        prange = None
    prio, why = pick_capture_priority(prange)
    if prio is None:
        if card not in _capture_named:
            _capture_named.add(card)
            logger.warning(fallback_line("d_capture_priority", card, why + " -> capture on the default-"
                                         "priority stream (MPS client priority / chunk cap remain)"))
        return None
    if card not in _capture_named:
        _capture_named.add(card)
        logger.info("%s D graph capture on a HIGH-priority stream card=%s priority=%d (%s)",
                    LOG_TAG, card, prio, why)
    return torch.cuda.Stream(priority=prio)


# ------------------------------------------------------------------ stage 1b: P MPS client priority

def driver_major(text: Optional[str]) -> Optional[int]:
    """The kernel module's major version from /proc/driver/nvidia/version text."""
    if not text:
        return None
    import re

    m = re.search(r"Kernel Module(?:\s+for\s+\S+)?\s+(\d+)\.(\d+)", text)
    if m is None:
        m = re.search(r"\b(\d{3})\.(\d+)(?:\.\d+)?\b", text)
    return int(m.group(1)) if m else None


#: CUDA 12.2 introduced MPS client priority; its driver branch is R535.
MPS_CLIENT_PRIORITY_MIN_DRIVER = 535


def mps_client_priority_env(*, mps_on: bool, driver_text: Optional[str]) -> Tuple[Dict[str, str], str]:
    """(env for P, the line to print). ``CUDA_MPS_CLIENT_PRIORITY=1`` (below
    normal) is read when the client connects -- boot only, not switchable."""
    if not mps_on:
        return {}, fallback_line("mps_client_priority", "all", "--dual-mps off (no MPS daemon, P is no MPS "
                                 "client) -> without priority")
    major = driver_major(driver_text)
    if major is not None and major < MPS_CLIENT_PRIORITY_MIN_DRIVER:
        return {}, fallback_line("mps_client_priority", "all", f"driver {major} < "
                                 f"{MPS_CLIENT_PRIORITY_MIN_DRIVER} (CUDA 12.2 MPS client priority) -> "
                                 "without priority")
    note = (f"driver {major}" if major is not None else "driver version unreadable, effect unverified")
    return ({"CUDA_MPS_CLIENT_PRIORITY": "1"},
            f"{LOG_TAG} P MPS client priority 1 (below normal) armed ({note}; a hint to the MPS "
            "scheduler, read at connect -- A/B per metal probe M7)")


# ------------------------------------------------------------------ stage 3 sketch: SM ladder

def sm_ladder(sm_total: int, fractions: Sequence[float], granularity: Optional[int],
              min_partition: Optional[int] = None) -> Tuple[List[Tuple[float, int]], Optional[str]]:
    """STAGE 3 PREP (not wired): each rung's SM count on one card, rounded DOWN
    to the driver's granularity (``smCoscheduledAlignment``) and at least
    ``minSmPartitionSize``; rung 0 is the whole card; duplicates after rounding
    are dropped. Returns (ladder, fallback reason or None) -- a ladder of fewer
    than two distinct rungs carries the reason. Every number is an input read
    at runtime (cuDeviceGetDevResource); nothing here knows a card."""
    sm_total = int(sm_total)
    if sm_total <= 0:
        return [], "SM count unreadable"
    if not granularity or int(granularity) <= 0:
        return [(1.0, sm_total)], "granularity not read (green-context API absent)"
    g = int(granularity)
    floor = max(g, int(min_partition or g))
    out: List[Tuple[float, int]] = []
    seen = set()
    for f in fractions:
        if f >= 1.0:
            n = sm_total
        else:
            n = int(math.floor(f * sm_total / g)) * g
            if n < floor:
                continue
        if n in seen:
            continue
        seen.add(n)
        out.append((float(f), n))
    if len(out) < 2:
        return out, f"ladder has {len(out)} distinct rung(s) after rounding to {g} SM"
    return out, None


# ------------------------------------------------------------------ the front's glue

class FrontShare:
    """Everything the front holds for the share control, so the front itself
    only delegates: the controller, the ctl writer, the P/D rate meters and
    the uncached rest of every leg 1 in flight on P (in the dual layout the
    queue empties into P at once, so Q = queue + P's in-flight legs)."""

    STATUS_EVERY_S = 30.0

    def __init__(self, *, ctl: str, mode: str, actuators: Sequence[str], cfg: ShareConfig,
                 clock: Callable[[], float] = time.monotonic, log: Callable[[str], None] = logger.info):
        self.ctl = ctl
        self.actuators = tuple(actuators)
        self.ctrl = ShareController(cfg, mode, clock=clock)
        self.writer = CtlWriter(ctl, heartbeat_s=cfg.heartbeat_s, clock=clock)
        self.p_rate = PRateMeter()
        self.d_rate = DRateMeter(clock=clock)
        self.p_rest: Dict[str, int] = {}
        self.last: Optional[Decision] = None
        self.ticks = 0
        self.write_errors = 0
        self._log = log
        self._clock = clock
        self._status_t = clock()
        self.green: Optional[str] = None       # "on" / "hold" once the green ladder front is armed (from_args)
        self._pobs_t = -1e9

    @classmethod
    def from_args(cls, *, ctl: str, mode: Optional[str], actuators: str, d_min_rate_tps: float,
                  p_min_share: float, env: Optional[Mapping[str, str]] = None,
                  log: Callable[[str], None] = logger.info, green_ladder: str = "off") -> Optional["FrontShare"]:
        if not ctl or not mode:
            return None
        e = os.environ if env is None else env
        cfg = config_from_env(e, d_min_rate_tps=d_min_rate_tps, p_min_share=p_min_share)
        acts = parse_actuators(actuators or "chunk")
        green = str(green_ladder or "off") != "off" and "green" in acts
        if green:
            # STAGE 3 (weg2/dual_green.py): the 5-stage automaton replaces the matrix controller; the tick is
            # faster (D empty must lift the throttle within ~50 ms, not 250) unless the profile set it.
            from sglang.srt.weg2 import dual_green as _dg

            if not str(e.get(ENV_PREFIX + "TICK_S", "") or "").strip():
                cfg = replace(cfg, tick_s=_dg.GreenConfig().tick_s)
        fs = cls(ctl=ctl, mode=mode, actuators=acts, cfg=cfg, log=log)
        if green:
            gcfg = _dg.GreenConfig.from_env(e)
            fs.ctrl = _dg.GreenController(cfg, mode, gcfg, clock=fs._clock)
            fs.d_rate = DRateMeter(window_s=0.5, clock=fs._clock)   # a short window: the round after a change
            fs.green = str(green_ladder)
            log(f"{LOG_TAG} GREEN ladder front armed ({green_ladder}): factors={list(gcfg.factors)} "
                f"tsolo_ms={[list(x) for x in gcfg.tsolo_ms]} accept_len={gcfg.accept_len:g} "
                f"desc_min_s={gcfg.desc_min_s:g} asc_ratio={gcfg.asc_ratio:g} asc_calm_s={gcfg.asc_calm_s:g} "
                f"asc_dwell_s={gcfg.asc_dwell_s:g} table={[list(r) for r in gcfg.table]} -- stage H (hold, 0 %) is "
                f"{'ARMED on PP0' if green_ladder == 'hold' else 'observer only (would_hold in P-STUFE)'}")
        log(f"{LOG_TAG} front controller armed mode={mode} ctl={ctl} actuators={','.join(acts)} "
            f"rungs={list(cfg.rungs)} tau_edges_s={list(cfg.tau_edges_s)} b_edges={list(cfg.b_edges)} "
            f"matrix={[list(r) for r in cfg.matrix]} static={dict(cfg.static)} deadband={cfg.deadband:g} "
            f"dwell_to_d_s={cfg.dwell_to_d_s:g} dwell_to_p_s={cfg.dwell_to_p_s:g} tick_s={cfg.tick_s:g} "
            f"d_min_rate_tps={cfg.d_min_rate_tps:g} p_min_share={cfg.p_min_share:g} max_rung={cfg.max_rung()}")
        return fs

    # -- feeds -----------------------------------------------------------
    def note_leg1_start(self, rid: str, uncached: int) -> None:
        self.p_rest[rid] = max(0, int(uncached or 0))

    def note_leg1_done(self, rid: str, tokens: int, seconds: Optional[float], d_idle: bool) -> None:
        self.p_rest.pop(rid, None)
        self.p_rate.note(int(tokens), seconds, d_idle)

    def note_d_chunk(self, rid: str, chunk: bytes) -> None:
        self.d_rate.chunk(rid, sse_events(chunk))

    def note_d_done(self, rid: str, completion_tokens: int, seconds: Optional[float] = None) -> None:
        self.d_rate.done(rid, int(completion_tokens or 0), seconds)

    # -- the tick --------------------------------------------------------
    def tick(self, *, queue: Iterable, p_outstanding: Mapping[str, float], d_outstanding: Mapping[str, float],
             seats: int, d_handoff: int = 0, now_wall: Optional[float] = None) -> Decision:
        """One decision. B = D's outstanding legs plus the hand-offs promised to
        D that it does not hold yet (the dbusy writer's own reading)."""
        now_wall = time.time() if now_wall is None else now_wall
        for rid in [r for r in self.p_rest if r not in p_outstanding]:
            self.p_rest.pop(rid, None)
        self.d_rate.prune(d_outstanding)
        q = 0
        oldest = now_wall
        for p in queue:
            q += int(getattr(p, "est_uncached", 0) or 0)
            oldest = min(oldest, float(getattr(p, "t_arrive", now_wall) or now_wall))
        q += sum(self.p_rest.get(r, 0) for r in p_outstanding)
        if p_outstanding:
            oldest = min(oldest, min(float(t) for t in p_outstanding.values()))
        rate, src = self.p_rate.rate()
        extra: Dict[str, object] = {}
        if self.green is not None:
            from sglang.srt.weg2 import dual_green as _dg

            if self._clock() - self._pobs_t >= 0.5:        # the P stages' observation files (SM, arena, would_hold)
                self._pobs_t = self._clock()
                self.ctrl.feed_pobs(_dg.read_pobs(self.ctl))
            extra["p_idle"] = (not p_outstanding) and q <= 0
        d = self.ctrl.observe(q_tokens=q, b=len(d_outstanding) + max(0, int(d_handoff)), seats=int(seats),
                              p_rate_tps=rate,
                              d_rate_tps=self.d_rate.rate(), oldest_age_s=max(0.0, now_wall - oldest), **extra)
        try:
            self.writer.update(d)
        except OSError as e:
            self.write_errors += 1
            if self.write_errors <= 3:
                self._log(fallback_line("ctl", "all", f"front cannot write {self.ctl} ({e}) -> P reads "
                                        "a stale file and falls back to rung 0 (P full)"))
        if self.ctrl.should_log(d):
            self._log(decision_line(d, self.actuators))
            if self.green is not None:
                from sglang.srt.weg2 import dual_green as _dg

                self._log(_dg.pstufe_line(d, self.ctrl.pobs_view()))
        now = self._clock()
        if now - self._status_t >= self.STATUS_EVERY_S:
            self._status_t = now
            self._log(f"{LOG_TAG} status mode={d.mode} rung={d.rung} changes={self.ctrl.changes} "
                      f"flaps={d.flaps} p_rate_src={src} tok_per_event={self.d_rate.tok_per_event:.2f} "
                      f"ctl_seq={self.writer.seq}")
        self.last = d
        self.ticks += 1
        return d

    def set_mode(self, mode: str, *, d_min_rate_tps: Optional[float] = None,
                 p_min_share: Optional[float] = None, who: str = "admin") -> Dict[str, object]:
        changed = self.ctrl.set_mode(mode)
        self.ctrl.set_limits(d_min_rate_tps=d_min_rate_tps, p_min_share=p_min_share)
        self._log(f"{LOG_TAG} mode -> {mode} (changed={changed}, by {who}) d_min_rate_tps="
                  f"{self.ctrl.cfg.d_min_rate_tps:g} p_min_share={self.ctrl.cfg.p_min_share:g}")
        return self.snapshot()

    def snapshot(self) -> Dict[str, object]:
        d = self.last
        return {"mode": self.ctrl.mode, "rung": self.ctrl.rung,
                "fraction": self.ctrl.cfg.rungs[self.ctrl.rung], "actuators": list(self.actuators),
                "d_min_rate_tps": self.ctrl.cfg.d_min_rate_tps, "p_min_share": self.ctrl.cfg.p_min_share,
                "changes": self.ctrl.changes, "flaps": self.ctrl.flaps, "ctl": self.ctl,
                "last_reason": d.reason if d is not None else None,
                "tau_s": d.tau_s if d is not None else None, "dbs": d.b if d is not None else None,
                **({"green": self.green} if self.green is not None else {})}
