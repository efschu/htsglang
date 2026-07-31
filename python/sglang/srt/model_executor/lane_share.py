"""Online card-equivalent estimator for the multi-group runtime (#274 slice D, S1).

The quantity is the one DESIGN_121 §11.5 already measures OFFLINE, and this
module is the same arithmetic done from inside the server, once per window::

    share_c = Rate_c(shared window) / Rate_c(solo floor)      per class c
    E       = SUM_c share_c                                   card equivalents

E = 1.0 means the classes split one card exactly; E > 1.0 means there was real
overlap.  DESIGN_201 addendum 12 (3) makes this the sensor of a future
controller as well as its objective, which is why it is worth having online:
``prefill_wait_ms`` measured 0.01 ms while device time rose from 583 to 638 ms,
so queueing is the wrong sensor -- the contention sits in compute time, and
share_c is where it shows up.

Four rules are inherited from measurements that have already gone wrong once,
and they are what most of the code below is for:

1. WORK PER WALL SECOND, never self-reported step time.  A lane tick's own ms
   says how fast a tick RUNS, not how much the lane GETS DONE; reading tick
   time as a rate reported E = 1.23 for a serial mode that is a zero-sum split
   of one wall clock by construction (DESIGN_121 §11.17).  Every rate here is
   a monotone work counter differenced over the wall time of its window.

2. ONE ARM PER WINDOW.  A prefill-shaped and a decode-shaped step have
   different solo floors, so a window in which a class did both has no defined
   share.  Such a window is dropped, named, and counted -- not averaged.

3. FLOORS FROM SOLO WINDOWS ONLY, and freezable.  A floor is only ever updated
   from a window in which exactly one class was busy.  ``freeze_floors()``
   exists for the self-conditioning trap of DESIGN_201 addendum 12 (4): once a
   controller changes the load it measures, floors may no longer be re-learned
   in flight -- they have to come from a controller-free boot.

4. EVERY WINDOW CARRIES ITS RUNG.  E is only defined per controller state, so
   a window whose rung id changed while it was open is unauswertbar and is
   dropped rather than reported.  Slice D1 builds no controller and always
   passes the same rung; the bookkeeping is here because addendum 12 requires
   it to exist BEFORE the first measurement, not after.

5. A SHARE IS A QUOTIENT, SO REPORT BOTH FACTORS (#284).  Round 8 measured the
   lane keeping 30 % of its solo rate under load where round 4 measured 100 %,
   and a rate ratio cannot say which of two opposite situations produced
   either number: a class denied the card, or a class that had the card and
   ran slower on it.  With ``occ = device_ms / wall_ms`` and
   ``cost = device_ms / work`` the definition of a rate makes
   ``share = (occ_shared/occ_solo) / (cost_shared/cost_solo)`` an identity, so
   each window can name which factor carries its loss -- see
   :attr:`ClassWindow.carrier`.  Classes that supply no device counters keep
   working exactly as before and simply get no attribution.

Pure Python by design: no torch, no CUDA, no server -- the whole estimator is
CPU-testable.  The device counters it consumes come from
:mod:`sglang.srt.model_executor.lane_device_clock`, which is where the CUDA
events live and which is equally torch-free at module level.
"""

from __future__ import annotations

import dataclasses
import math
from typing import Dict, Iterable, List, Mapping, Optional, Tuple

__all__ = [
    "CARRIERS",
    "DEFAULT_EMA_S",
    "DEFAULT_WINDOW_S",
    "ClassSample",
    "ClassWindow",
    "FloorEstimate",
    "LaneShareGate",
    "LaneShareMeter",
    "ShareWindow",
]

DEFAULT_WINDOW_S: float = 1.0
# EMA over ~1 s: the A-vs-A spans of the underlying quantities are 0.25-0.39 %,
# so a measurement window has to be long enough to sit above them
# (DESIGN_201 addendum 12 (3)).
DEFAULT_EMA_S: float = 1.0
DEFAULT_FLOOR_MIN_WINDOWS: int = 3
DEFAULT_FLOOR_ALPHA: float = 0.35
# Below this many work units per second a class counts as idle rather than
# slow: one counter tick landing inside a window would otherwise define a
# "solo floor" of one token per second.
MIN_RATE: float = 1e-6
# Device time below this per window is not an occupancy measurement, it is a
# rounding artefact of one event pair landing inside the window.
MIN_DEVICE_MS: float = 1e-3
# A duty cycle this far below its solo floor means the class was not HOLDING
# work for part of the window, which is a statement about the feeder, not about
# the card.  Deliberately generous: the question it decides is only which of
# three named carriers a window is filed under.
DUTY_STARVED_RATIO: float = 0.9

STATIC_RUNG: str = "static"

# The three ways a class can lose rate under sharing, named once so that a
# window, a gate verdict and a report all use the same word.  They are not
# opinions: each one is the term that dominates the exact decomposition in
# ``ClassWindow.carrier``.
CARRIER_SM: str = "sm_competition"
CARRIER_SUBMISSION: str = "submission_gap"
CARRIER_STARVED: str = "starved"
CARRIERS: Tuple[str, str, str] = (CARRIER_SM, CARRIER_SUBMISSION, CARRIER_STARVED)


@dataclasses.dataclass(frozen=True)
class ClassSample:
    """One class's monotone work counters at one instant.

    ``counters`` maps an ARM name (``"prefill_tokens"``, ``"decode_tokens"``,
    ...) to a monotone count of work completed.  Several arms may be reported;
    at most one of them may advance inside a window (rule 2).

    ``device`` carries the same class's monotone TIME counters in ms --
    ``device_ms`` (what its kernels took on its own stream) and
    ``busy_wall_ms`` (wall time it held work).  Optional: a class that cannot
    measure them still gets a share, it just gets no carrier attribution.
    """

    key: str
    counters: Mapping[str, float]
    device: Optional[Mapping[str, float]] = None


@dataclasses.dataclass(frozen=True)
class FloorEstimate:
    """A class+arm's solo behaviour: the denominator of share_c, and the two
    quantities share_c decomposes into.

    ``rate`` is the only one the share needs.  ``occupancy`` (device ms per
    wall ms) and ``cost`` (device ms per work unit) are carried alongside
    because their ratio IS the share -- see :meth:`ClassWindow.carrier` -- and
    a decomposition whose denominator came from a different window than its
    numerator would explain the wrong run.
    """

    rate: float
    windows: int
    frozen: bool = False
    occupancy: Optional[float] = None
    cost: Optional[float] = None
    duty: Optional[float] = None

    @property
    def ready(self) -> bool:
        return self.windows >= 1 and self.rate > MIN_RATE

    def to_json(self) -> Dict[str, object]:
        out: Dict[str, object] = {
            "rate": round(self.rate, 6),
            "windows": self.windows,
            "frozen": self.frozen,
        }
        for name, value in (
            ("occupancy", self.occupancy),
            ("cost_ms", self.cost),
            ("duty", self.duty),
        ):
            if value is not None:
                out[name] = round(value, 6)
        return out


@dataclasses.dataclass(frozen=True)
class ClassWindow:
    """What one class did in one window.

    Beyond the rate and its share, a window carries the two quantities the
    share is the quotient of.  With ``occ = device_ms / wall_ms`` (the fraction
    of the window the class's own kernels were executing) and
    ``cost = device_ms / work`` (device ms per token), the definition of a rate
    is an identity, not an approximation::

        rate = work / wall = occ / cost

    so::

        share = rate_shared / rate_solo = (occ_s / occ_0) / (cost_s / cost_0)

    That splits a lost share into exactly two named terms: the class got LESS
    OF THE CARD (occupancy fell) or it got the same card time and each token
    cost MORE on it (cost rose).  The first is submission or starvation, the
    second is SM competition -- and which of the two carries a given number is
    the question that a rate ratio alone cannot answer.
    """

    key: str
    arm: Optional[str]
    rate: float
    busy: bool
    mixed: bool
    floor: Optional[float]
    share: Optional[float]
    device_ms: Optional[float] = None
    busy_wall_ms: Optional[float] = None
    occupancy: Optional[float] = None
    duty: Optional[float] = None
    cost_ms: Optional[float] = None
    floor_occupancy: Optional[float] = None
    floor_cost: Optional[float] = None
    floor_duty: Optional[float] = None

    @property
    def occupancy_ratio(self) -> Optional[float]:
        if not self.floor_occupancy or self.occupancy is None:
            return None
        return self.occupancy / self.floor_occupancy

    @property
    def cost_ratio(self) -> Optional[float]:
        if not self.floor_cost or self.cost_ms is None:
            return None
        return self.cost_ms / self.floor_cost

    @property
    def duty_ratio(self) -> Optional[float]:
        if not self.floor_duty or self.duty is None:
            return None
        return self.duty / self.floor_duty

    @property
    def carrier(self) -> Optional[str]:
        """Which term carries the lost share, or ``None`` if nothing is lost.

        Not a heuristic ranking: the two candidate terms are the two factors of
        an identity, so whichever is larger IS the larger part of the loss.
        The only judgement in here is the third outcome -- a duty cycle below
        its solo floor means the class did not HOLD work for part of the
        window, which is a statement about its feeder and must not be read as
        the card denying it access.
        """
        occ_r, cost_r = self.occupancy_ratio, self.cost_ratio
        if occ_r is None or cost_r is None or self.share is None:
            return None
        if self.share >= 1.0:
            return None
        lost_access = -math.log(occ_r) if occ_r > 0 else float("inf")
        lost_speed = math.log(cost_r) if cost_r > 0 else 0.0
        if lost_speed >= lost_access:
            return CARRIER_SM
        duty_r = self.duty_ratio
        if duty_r is not None and duty_r < DUTY_STARVED_RATIO:
            return CARRIER_STARVED
        return CARRIER_SUBMISSION

    def to_json(self) -> Dict[str, object]:
        out: Dict[str, object] = {
            "key": self.key,
            "arm": self.arm,
            "rate": round(self.rate, 6),
            "busy": self.busy,
            "mixed": self.mixed,
            "floor": None if self.floor is None else round(self.floor, 6),
            "share": None if self.share is None else round(self.share, 6),
        }
        for name, value in (
            ("device_ms", self.device_ms),
            ("busy_wall_ms", self.busy_wall_ms),
            ("occupancy", self.occupancy),
            ("duty", self.duty),
            ("cost_ms", self.cost_ms),
            ("occupancy_ratio", self.occupancy_ratio),
            ("cost_ratio", self.cost_ratio),
            ("duty_ratio", self.duty_ratio),
        ):
            if value is not None:
                out[name] = round(value, 6)
        carrier = self.carrier
        if carrier is not None:
            out["carrier"] = carrier
        return out


@dataclasses.dataclass(frozen=True)
class ShareWindow:
    """One closed window: the unit of everything this module reports."""

    t_end: float
    wall_s: float
    rung: str
    classes: Tuple[ClassWindow, ...]
    kind: str  # "idle" | "solo" | "shared" | "dropped"
    solo_for: Optional[str] = None
    e: Optional[float] = None
    e_ema: Optional[float] = None
    dropped: Optional[str] = None

    @property
    def usable(self) -> bool:
        return self.kind == "shared" and self.e is not None

    def to_json(self) -> Dict[str, object]:
        return {
            "t_end": round(self.t_end, 3),
            "wall_s": round(self.wall_s, 4),
            "rung": self.rung,
            "kind": self.kind,
            "solo_for": self.solo_for,
            "e": None if self.e is None else round(self.e, 6),
            "e_ema": None if self.e_ema is None else round(self.e_ema, 6),
            "dropped": self.dropped,
            "classes": [c.to_json() for c in self.classes],
        }


def _ema_alpha(window_s: float, ema_s: float) -> float:
    if ema_s <= 0.0:
        return 1.0
    return 1.0 - math.exp(-max(window_s, 1e-9) / ema_s)


def _blend(old: Optional[float], new: Optional[float], alpha: float) -> Optional[float]:
    """EMA that survives a missing side.

    A window with no device counters must not erase a floor's occupancy, and a
    floor learned before the clock existed must not block one learned after.
    """
    if new is None:
        return old
    if old is None:
        return new
    return (1.0 - alpha) * old + alpha * new


def _device_deltas(
    before: Optional[Mapping[str, float]], after: Optional[Mapping[str, float]]
) -> Tuple[Optional[float], Optional[float]]:
    """``(device_ms, busy_wall_ms)`` consumed inside the window, or ``None``.

    ``None`` and ``0.0`` are different answers and are kept apart: a class
    without a device clock has no occupancy, while a class whose clock stood
    still has an occupancy of zero, and filing the first as the second would
    put a fabricated carrier on every window.
    """
    if not before or not after:
        return (None, None)
    out: List[Optional[float]] = []
    for name in ("device_ms", "busy_wall_ms"):
        if name not in before or name not in after:
            out.append(None)
            continue
        out.append(max(0.0, float(after[name]) - float(before[name])))
    return (out[0], out[1])


@dataclasses.dataclass(frozen=True)
class _Anchor:
    """The opening edge of the window that is currently accumulating."""

    t0: float
    work: Dict[str, Dict[str, float]]
    device: Dict[str, Dict[str, float]]
    rung: str


class LaneShareMeter:
    """Turns monotone per-class work counters into share_c and E.

    Feed it with :meth:`observe` as often as convenient (once per scheduler
    iteration is the intended rate); it closes a window every ``window_s``
    seconds and returns the closed :class:`ShareWindow`, or ``None`` while the
    current window is still open.  It never sleeps, never synchronizes and
    never touches the GPU.
    """

    def __init__(
        self,
        *,
        window_s: float = DEFAULT_WINDOW_S,
        ema_s: float = DEFAULT_EMA_S,
        floor_min_windows: int = DEFAULT_FLOOR_MIN_WINDOWS,
        floor_alpha: float = DEFAULT_FLOOR_ALPHA,
        history: int = 64,
    ) -> None:
        self.window_s = float(window_s)
        self.ema_s = float(ema_s)
        self.floor_min_windows = int(floor_min_windows)
        self.floor_alpha = float(floor_alpha)
        self._alpha = _ema_alpha(self.window_s, self.ema_s)
        self._history_cap = int(history)

        self._floors: Dict[str, FloorEstimate] = {}
        self._anchor: Optional[_Anchor] = None
        self._share_ema: Dict[str, float] = {}
        self._e_ema: Optional[float] = None
        self._history: List[ShareWindow] = []
        self._history_json: List[Dict[str, object]] = []
        self._last: Optional[ShareWindow] = None
        self._gates: List["LaneShareGate"] = []
        self.counts: Dict[str, int] = {
            "idle": 0,
            "solo": 0,
            "shared": 0,
            "dropped": 0,
            "shared_without_floor": 0,
        }

    # -- floors ----------------------------------------------------------

    @staticmethod
    def floor_key(key: str, arm: str) -> str:
        return f"{key}/{arm}"

    def floors(self) -> Dict[str, FloorEstimate]:
        return dict(self._floors)

    def freeze_floors(self) -> None:
        """Stop learning floors.  See rule 3: a controller must not re-learn
        the denominator of the quantity it is steering."""
        self._floors = {
            k: dataclasses.replace(v, frozen=True) for k, v in self._floors.items()
        }

    def load_floors(self, floors: Mapping[str, float], *, frozen: bool = True) -> None:
        """Install floors measured in a controller-free boot."""
        for k, rate in floors.items():
            self._floors[k] = FloorEstimate(
                rate=float(rate), windows=self.floor_min_windows, frozen=bool(frozen)
            )

    def _update_floor(self, key: str, arm: str, row: "ClassWindow") -> None:
        fk = self.floor_key(key, arm)
        cur = self._floors.get(fk)
        if cur is not None and cur.frozen:
            return
        if cur is None:
            self._floors[fk] = FloorEstimate(
                rate=row.rate,
                windows=1,
                occupancy=row.occupancy,
                cost=row.cost_ms,
                duty=row.duty,
            )
            return
        self._floors[fk] = FloorEstimate(
            rate=_blend(cur.rate, row.rate, self.floor_alpha),
            windows=cur.windows + 1,
            occupancy=_blend(cur.occupancy, row.occupancy, self.floor_alpha),
            cost=_blend(cur.cost, row.cost_ms, self.floor_alpha),
            duty=_blend(cur.duty, row.duty, self.floor_alpha),
        )

    def _floor_for(self, key: str, arm: str) -> Optional[FloorEstimate]:
        est = self._floors.get(self.floor_key(key, arm))
        if est is None or est.windows < self.floor_min_windows or not est.ready:
            return None
        return est

    # -- the window ------------------------------------------------------

    def observe(
        self,
        now: float,
        samples: Iterable[ClassSample],
        *,
        rung: str = STATIC_RUNG,
    ) -> Optional[ShareWindow]:
        samples = list(samples)
        snap = {s.key: {a: float(v) for a, v in s.counters.items()} for s in samples}
        dsnap = {
            s.key: {a: float(v) for a, v in (s.device or {}).items()} for s in samples
        }
        if self._anchor is None:
            self._anchor = _Anchor(now, snap, dsnap, rung)
            return None
        anchor = self._anchor
        wall = now - anchor.t0
        if wall < self.window_s:
            # The rung is part of the window's identity, so a change closes it
            # early -- as a DROPPED window, never as a measurement.
            if rung != anchor.rung:
                win = ShareWindow(
                    t_end=now,
                    wall_s=wall,
                    rung=anchor.rung,
                    classes=(),
                    kind="dropped",
                    dropped="rung_change",
                )
                self._anchor = _Anchor(now, snap, dsnap, rung)
                return self._record(win)
            return None
        self._anchor = _Anchor(now, snap, dsnap, rung)
        if rung != anchor.rung:
            return self._record(
                ShareWindow(
                    t_end=now,
                    wall_s=wall,
                    rung=anchor.rung,
                    classes=(),
                    kind="dropped",
                    dropped="rung_change",
                )
            )
        return self._record(self._close(now, wall, anchor, snap, dsnap))

    def _close(
        self,
        now: float,
        wall: float,
        anchor: _Anchor,
        snap: Mapping[str, Mapping[str, float]],
        dsnap: Mapping[str, Mapping[str, float]],
    ) -> ShareWindow:
        rung = anchor.rung
        base = anchor.work
        rows: List[ClassWindow] = []
        mixed_keys: List[str] = []
        wall_ms = wall * 1000.0
        for key in sorted(snap):
            device_ms, busy_ms = _device_deltas(anchor.device.get(key), dsnap.get(key))
            occupancy = None if device_ms is None else device_ms / wall_ms
            duty = None if busy_ms is None else busy_ms / wall_ms
            before = base.get(key)
            if before is None:
                # A class that appeared mid-window has no baseline.
                rows.append(ClassWindow(key, None, 0.0, False, False, None, None))
                continue
            active: List[Tuple[str, float]] = []
            for arm, value in snap[key].items():
                prev = before.get(arm)
                if prev is None:
                    continue
                delta = value - prev
                if delta / wall > MIN_RATE:
                    active.append((arm, delta / wall))
            if not active:
                rows.append(ClassWindow(key, None, 0.0, False, False, None, None))
                continue
            arm, rate = (
                max(active, key=lambda kv: kv[1]) if len(active) > 1 else active[0]
            )
            if len(active) > 1:
                mixed_keys.append(key)
            # Cost is device ms per WORK UNIT, so it needs the work of this
            # window and not its rate: rate * wall is that work, and dividing
            # occupancy by cost gives the rate back exactly (see ClassWindow).
            cost = None
            if device_ms is not None and device_ms > MIN_DEVICE_MS:
                work = rate * wall
                if work > MIN_RATE:
                    cost = device_ms / work
            rows.append(
                ClassWindow(
                    key,
                    arm,
                    rate,
                    True,
                    len(active) > 1,
                    None,
                    None,
                    device_ms=device_ms,
                    busy_wall_ms=busy_ms,
                    occupancy=occupancy,
                    duty=duty,
                    cost_ms=cost,
                )
            )

        busy = [r for r in rows if r.busy]
        if mixed_keys:
            return ShareWindow(
                t_end=now,
                wall_s=wall,
                rung=rung,
                classes=tuple(rows),
                kind="dropped",
                dropped="mixed_arms:" + ",".join(sorted(mixed_keys)),
            )
        if not busy:
            return ShareWindow(
                t_end=now, wall_s=wall, rung=rung, classes=tuple(rows), kind="idle"
            )
        if len(busy) == 1:
            row = busy[0]
            assert row.arm is not None
            self._update_floor(row.key, row.arm, row)
            est = self._floors[self.floor_key(row.key, row.arm)]
            rows = [
                (
                    dataclasses.replace(
                        r,
                        floor=est.rate,
                        floor_occupancy=est.occupancy,
                        floor_cost=est.cost,
                        floor_duty=est.duty,
                    )
                    if r is row
                    else r
                )
                for r in rows
            ]
            return ShareWindow(
                t_end=now,
                wall_s=wall,
                rung=rung,
                classes=tuple(rows),
                kind="solo",
                solo_for=row.key,
            )

        # Shared window: every busy class needs a ready floor for the SAME arm
        # it is running now, or E is undefined for this window.
        out: List[ClassWindow] = []
        shares: List[float] = []
        missing: List[str] = []
        for row in rows:
            if not row.busy or row.arm is None:
                out.append(row)
                continue
            est = self._floor_for(row.key, row.arm)
            if est is None:
                missing.append(self.floor_key(row.key, row.arm))
                out.append(row)
                continue
            share = row.rate / est.rate
            shares.append(share)
            out.append(
                dataclasses.replace(
                    row,
                    floor=est.rate,
                    share=share,
                    floor_occupancy=est.occupancy,
                    floor_cost=est.cost,
                    floor_duty=est.duty,
                )
            )
        if missing:
            return ShareWindow(
                t_end=now,
                wall_s=wall,
                rung=rung,
                classes=tuple(out),
                kind="shared",
                dropped="no_floor:" + ",".join(sorted(missing)),
            )
        e = sum(shares)
        for row in out:
            if row.share is not None:
                prev = self._share_ema.get(row.key)
                self._share_ema[row.key] = (
                    row.share
                    if prev is None
                    else prev + self._alpha * (row.share - prev)
                )
        self._e_ema = (
            e if self._e_ema is None else self._e_ema + self._alpha * (e - self._e_ema)
        )
        return ShareWindow(
            t_end=now,
            wall_s=wall,
            rung=rung,
            classes=tuple(out),
            kind="shared",
            e=e,
            e_ema=self._e_ema,
        )

    def attach_gate(self, gate: "LaneShareGate") -> None:
        """Let a gate see every window this meter closes.

        The gate is a consumer, never a producer: it cannot change a floor, a
        window or a verdict of the meter, so switching it on cannot move the
        number it judges.
        """
        self._gates.append(gate)

    def gates(self) -> List["LaneShareGate"]:
        return list(self._gates)

    def _record(self, win: ShareWindow) -> ShareWindow:
        for gate in self._gates:
            gate.observe(win)
        if win.kind == "shared" and win.e is None:
            self.counts["shared_without_floor"] += 1
        else:
            self.counts[win.kind] = self.counts.get(win.kind, 0) + 1
        self._last = win
        self._history.append(win)
        # Serialize ONCE, here, where it costs one window's worth of work per
        # window.  ``snapshot()`` is called from ``get_internal_state``, i.e.
        # from the scheduler thread in the middle of the event loop, several
        # times a second -- re-serializing the whole history there put
        # measurable latency into the serving group's iteration (see
        # DESIGN_121 §12.7).  A readout must not cost more than the thing it
        # reads.
        self._history_json.append(win.to_json())
        if len(self._history) > self._history_cap:
            drop = len(self._history) - self._history_cap
            del self._history[:drop]
            del self._history_json[:drop]
        return win

    # -- readout ---------------------------------------------------------

    @property
    def e_ema(self) -> Optional[float]:
        return self._e_ema

    def share_ema(self, key: str) -> Optional[float]:
        return self._share_ema.get(key)

    def history(self) -> List[ShareWindow]:
        return list(self._history)

    def snapshot(self, *, history: int = 16) -> Dict[str, object]:
        """The dict that goes into ``/get_server_info`` -> ``internal_states``.

        Cheap by construction: the windows are already serialized (see
        ``_record``), so this is a slice and a handful of small dicts.  It runs
        on the scheduler thread, so its cost is paid by the serving group.
        """
        return {
            "window_s": self.window_s,
            "ema_s": self.ema_s,
            "e_ema": None if self._e_ema is None else round(self._e_ema, 6),
            "share_ema": {k: round(v, 6) for k, v in self._share_ema.items()},
            "floors": {k: v.to_json() for k, v in sorted(self._floors.items())},
            "counts": dict(self.counts),
            "last": self._history_json[-1] if self._history_json else None,
            "windows": self._history_json[-history:] if history > 0 else [],
            "gates": [g.snapshot() for g in self._gates],
        }


class LaneShareGate:
    """A standing criterion of the form "class K keeps >= X of its solo rate".

    The gate the r8 report asked for, written down so that it can be checked
    continuously instead of reconstructed from a boot's raw data afterwards.
    Three properties are what make it a gate rather than a gauge:

    1. IT JUDGES ONLY SHARED WINDOWS.  A solo window trivially has share 1.0
       and an idle one has none; averaging either into the verdict would let a
       lane pass a load test by not being loaded.  ``load`` is a free-text NAME
       for the load the verdict is about, carried through to the readout,
       because "the lane keeps 30 %" means nothing without it.
    2. IT NEEDS ``min_windows`` OF THEM.  Below that the verdict is
       ``insufficient``, never ``pass``.  A criterion that reports green while
       it has no data is worse than no criterion.
    3. IT NAMES THE CARRIER WHEN IT FAILS.  A failing gate that says only "too
       low" sends the next round looking for the wrong lever; the carrier comes
       from the occupancy/cost decomposition of the same windows, so the
       diagnosis and the verdict cannot disagree.
    4. A MOVING DENOMINATOR VOIDS THE VERDICT.  ``share`` is a quotient, and
       the floor underneath it keeps being re-learned from solo windows
       (rule 3).  That is right while the class's own configuration holds
       still and wrong the moment it does not: measured on the #284 boot, a
       driver that ran a captured lane (56.8 tok/s solo), an eager lane
       (16.5) and a depth-1 feeder (50.4) through one meter blended them into
       a single floor of 33.7, and the gate reported a comfortable **pass** at
       a median share of 0.310 for a lane that was in truth keeping 0.185 of
       the floor it should have been judged against.  The number was not
       wrong; the denominator was.  So the gate watches its own floor and
       returns ``insufficient`` with ``floor_moved`` when the floor it divided
       by spans more than ``floor_tolerance`` across the windows it judged.
       Freeze the floors (:meth:`LaneShareMeter.freeze_floors`) or install
       them from a configuration-stable boot (:meth:`load_floors`) and the
       verdict comes back.

    The gate never enforces anything.  Nothing in the runtime reads its verdict
    to make a decision -- deliberately, because a controller reacting to a
    measurement makes that measurement self-conditioning (DESIGN_201 addendum
    12 (4)), and the whole point of this one is to survive being trusted.
    """

    def __init__(
        self,
        key: str,
        min_share: float,
        *,
        load: str = "unspecified",
        min_windows: int = 5,
        history: int = 64,
        floor_tolerance: float = 0.10,
    ) -> None:
        self.key = str(key)
        self.min_share = float(min_share)
        self.load = str(load)
        self.min_windows = int(min_windows)
        self.floor_tolerance = float(floor_tolerance)
        self._history_cap = int(history)
        self._shares: List[float] = []
        self._floors: List[float] = []
        self._carriers: Dict[str, int] = {}
        self.judged = 0
        self.failed = 0

    def observe(self, win: ShareWindow) -> None:
        if win.kind != "shared" or win.e is None:
            return
        for row in win.classes:
            if row.key != self.key or row.share is None:
                continue
            self.judged += 1
            self._shares.append(row.share)
            self._floors.append(row.floor if row.floor is not None else float("nan"))
            if len(self._shares) > self._history_cap:
                cut = len(self._shares) - self._history_cap
                del self._shares[:cut]
                del self._floors[:cut]
            if row.share < self.min_share:
                self.failed += 1
                carrier = row.carrier
                if carrier is not None:
                    self._carriers[carrier] = self._carriers.get(carrier, 0) + 1

    @property
    def floor_span(self) -> Optional[float]:
        """Relative spread of the floor across the judged windows.

        ``(max - min) / max``, so 0.0 is a floor that never moved.
        """
        floors = [f for f in self._floors if f == f and f > 0.0]
        if not floors:
            return None
        hi, lo = max(floors), min(floors)
        return (hi - lo) / hi

    @property
    def median_share(self) -> Optional[float]:
        if not self._shares:
            return None
        ordered = sorted(self._shares)
        mid = len(ordered) // 2
        if len(ordered) % 2:
            return ordered[mid]
        return 0.5 * (ordered[mid - 1] + ordered[mid])

    @property
    def carrier(self) -> Optional[str]:
        """The carrier that dominates the failing windows, if any failed."""
        if not self._carriers:
            return None
        return max(self._carriers.items(), key=lambda kv: kv[1])[0]

    @property
    def insufficient_reason(self) -> Optional[str]:
        if len(self._shares) < self.min_windows:
            return "too_few_windows"
        span = self.floor_span
        if span is not None and span > self.floor_tolerance:
            return "floor_moved"
        return None

    @property
    def verdict(self) -> str:
        """``pass`` | ``fail`` | ``insufficient``.

        The MEDIAN carries the verdict, not the mean: one window in which the
        lane happened to be between jobs is a long left tail on a bounded
        quantity, and a mean lets that tail decide a gate.
        """
        median = self.median_share
        if median is None or self.insufficient_reason is not None:
            return "insufficient"
        return "pass" if median >= self.min_share else "fail"

    def describe(self) -> str:
        return (
            f"lane share gate: {self.key} keeps >= {self.min_share:.3f} "
            f"of its solo rate under load '{self.load}'"
        )

    def snapshot(self) -> Dict[str, object]:
        return {
            "key": self.key,
            "min_share": self.min_share,
            "load": self.load,
            "min_windows": self.min_windows,
            "verdict": self.verdict,
            "insufficient_reason": self.insufficient_reason,
            "judged": self.judged,
            "failed": self.failed,
            "median_share": (
                None if self.median_share is None else round(self.median_share, 6)
            ),
            "floor_span": (
                None if self.floor_span is None else round(self.floor_span, 6)
            ),
            "floor_tolerance": self.floor_tolerance,
            "carrier": self.carrier,
            "carriers": dict(self._carriers),
            "describe": self.describe(),
        }
