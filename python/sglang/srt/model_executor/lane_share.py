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

Pure Python by design: no torch, no CUDA, no server -- the whole estimator is
CPU-testable.
"""

from __future__ import annotations

import dataclasses
import math
from typing import Dict, Iterable, List, Mapping, Optional, Tuple

__all__ = [
    "DEFAULT_EMA_S",
    "DEFAULT_WINDOW_S",
    "ClassSample",
    "ClassWindow",
    "FloorEstimate",
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

STATIC_RUNG: str = "static"


@dataclasses.dataclass(frozen=True)
class ClassSample:
    """One class's monotone work counters at one instant.

    ``counters`` maps an ARM name (``"prefill_tokens"``, ``"decode_tokens"``,
    ...) to a monotone count of work completed.  Several arms may be reported;
    at most one of them may advance inside a window (rule 2).
    """

    key: str
    counters: Mapping[str, float]


@dataclasses.dataclass(frozen=True)
class FloorEstimate:
    """A class+arm's solo rate: the denominator of share_c."""

    rate: float
    windows: int
    frozen: bool = False

    @property
    def ready(self) -> bool:
        return self.windows >= 1 and self.rate > MIN_RATE

    def to_json(self) -> Dict[str, object]:
        return {
            "rate": round(self.rate, 6),
            "windows": self.windows,
            "frozen": self.frozen,
        }


@dataclasses.dataclass(frozen=True)
class ClassWindow:
    """What one class did in one window."""

    key: str
    arm: Optional[str]
    rate: float
    busy: bool
    mixed: bool
    floor: Optional[float]
    share: Optional[float]

    def to_json(self) -> Dict[str, object]:
        return {
            "key": self.key,
            "arm": self.arm,
            "rate": round(self.rate, 6),
            "busy": self.busy,
            "mixed": self.mixed,
            "floor": None if self.floor is None else round(self.floor, 6),
            "share": None if self.share is None else round(self.share, 6),
        }


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
        self._anchor: Optional[Tuple[float, Dict[str, Dict[str, float]], str]] = None
        self._share_ema: Dict[str, float] = {}
        self._e_ema: Optional[float] = None
        self._history: List[ShareWindow] = []
        self._history_json: List[Dict[str, object]] = []
        self._last: Optional[ShareWindow] = None
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

    def _update_floor(self, key: str, arm: str, rate: float) -> None:
        fk = self.floor_key(key, arm)
        cur = self._floors.get(fk)
        if cur is not None and cur.frozen:
            return
        if cur is None:
            self._floors[fk] = FloorEstimate(rate=rate, windows=1)
            return
        blended = (1.0 - self.floor_alpha) * cur.rate + self.floor_alpha * rate
        self._floors[fk] = FloorEstimate(rate=blended, windows=cur.windows + 1)

    def _floor_for(self, key: str, arm: str) -> Optional[float]:
        est = self._floors.get(self.floor_key(key, arm))
        if est is None or est.windows < self.floor_min_windows or not est.ready:
            return None
        return est.rate

    # -- the window ------------------------------------------------------

    def observe(
        self,
        now: float,
        samples: Iterable[ClassSample],
        *,
        rung: str = STATIC_RUNG,
    ) -> Optional[ShareWindow]:
        snap = {s.key: {a: float(v) for a, v in s.counters.items()} for s in samples}
        if self._anchor is None:
            self._anchor = (now, snap, rung)
            return None
        t0, base, rung0 = self._anchor
        wall = now - t0
        if wall < self.window_s:
            # The rung is part of the window's identity, so a change closes it
            # early -- as a DROPPED window, never as a measurement.
            if rung != rung0:
                win = ShareWindow(
                    t_end=now,
                    wall_s=wall,
                    rung=rung0,
                    classes=(),
                    kind="dropped",
                    dropped="rung_change",
                )
                self._anchor = (now, snap, rung)
                return self._record(win)
            return None
        self._anchor = (now, snap, rung)
        if rung != rung0:
            return self._record(
                ShareWindow(
                    t_end=now,
                    wall_s=wall,
                    rung=rung0,
                    classes=(),
                    kind="dropped",
                    dropped="rung_change",
                )
            )
        return self._record(self._close(now, wall, rung0, base, snap))

    def _close(
        self,
        now: float,
        wall: float,
        rung: str,
        base: Mapping[str, Mapping[str, float]],
        snap: Mapping[str, Mapping[str, float]],
    ) -> ShareWindow:
        rows: List[ClassWindow] = []
        mixed_keys: List[str] = []
        for key in sorted(snap):
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
            if len(active) > 1:
                mixed_keys.append(key)
                arm, rate = max(active, key=lambda kv: kv[1])
                rows.append(ClassWindow(key, arm, rate, True, True, None, None))
                continue
            arm, rate = active[0]
            rows.append(ClassWindow(key, arm, rate, True, False, None, None))

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
            self._update_floor(row.key, row.arm, row.rate)
            est = self._floors[self.floor_key(row.key, row.arm)]
            rows = [
                dataclasses.replace(r, floor=est.rate) if r is row else r for r in rows
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
            floor = self._floor_for(row.key, row.arm)
            if floor is None:
                missing.append(self.floor_key(row.key, row.arm))
                out.append(row)
                continue
            share = row.rate / floor
            shares.append(share)
            out.append(dataclasses.replace(row, floor=floor, share=share))
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

    def _record(self, win: ShareWindow) -> ShareWindow:
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
        }
