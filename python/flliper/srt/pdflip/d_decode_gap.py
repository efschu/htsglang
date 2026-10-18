# SPDX-License-Identifier: Apache-2.0
"""S48 D-DECODE-GAP: every pause of group D's decode streams, with its cause, also below 5 s.

WHY. S47 (rc13 ``..._6767e25d89_1018_200305``) could explain 45-48 s of the 62-95 s of decode silence per 5 min by
the flips and P epochs; the rest stayed undivided because ``Decode batch`` is written every ~5 s and nothing names a
pause below that. The per-round ``Decode rank batch`` line carries a start stamp, so the pauses can be measured after
the fact (S48: 721 gaps >= 0.3 s in 70 min, 1503 s), but their CAUSE had to be re-derived by hand from the lines in
between. This instrument names it at run time, one line per gap, and a census per window.

WHAT A GAP IS. The start-to-start time between two batches that advance the running decodes (a decode batch, or a
MIXED-D extend that carries decode rows), counted only when the earlier one had decode rows (a group with nothing to
decode is idle, not paused). Start-to-start includes the earlier round's own forward, so the default bound (400 ms)
sits above a long bs=6 round (~110 ms gpu, rare 300-400 ms).

CAUSES (the dominant one is ``cause=``, every one present is listed in ``notes=``):

* ``flip``            -- D slept and woke between the two rounds (``_pdflip_wake_seq`` moved): the P epoch and the
                         flip pair;
* ``d_prefill``       -- pure extend batches ran between the rounds (R2 shorts, resumed tails): their summed wall,
                         token count and number are printed (``d_prefill_ms``, ``prefill_tokens``, ``extends``);
* ``skipfirst_hold``  -- ``PDFLIP-SKIP-FIRST DECODE-HOLD`` declined an extend for a pass;
* ``f3_decode_first`` -- the F3 decode-first gate deferred the wake's parked resumes;
* ``seat``            -- the D seat gate (d_seats.admission_gate barrier) held a request;
* ``wake_read``       -- a #248 wake read was issued in the gap;
* ``other``           -- none of these: the rest of the gap (``other_ms`` = gap minus the extends' wall).

Rank-local, read-only, no collective, no decision (nothing here can split the ranks; R11 untouched). Logged on the
first TP rank of group D only. Switch ``FLLIPER_PDFLIP_D_DECODE_GAP`` (default on), bound
``FLLIPER_PDFLIP_D_DECODE_GAP_MS`` (400), census window ``FLLIPER_PDFLIP_D_DECODE_GAP_CENSUS_S`` (300).
"""
from __future__ import annotations

import logging
import os
import time
from typing import Dict, Optional

logger = logging.getLogger(__name__)

MARK = "D-DECODE-GAP"
ENV = "FLLIPER_PDFLIP_D_DECODE_GAP"
ENV_MS = "FLLIPER_PDFLIP_D_DECODE_GAP_MS"
ENV_CENSUS_S = "FLLIPER_PDFLIP_D_DECODE_GAP_CENSUS_S"
_ATTR = "_pdflip_d_decode_gap"
NOTE_CAUSES = ("skipfirst_hold", "f3_decode_first", "seat", "wake_read")
CAUSES = ("flip", "d_prefill") + NOTE_CAUSES + ("other",)


def _env_float(name: str, default: float) -> float:
    try:
        v = float(os.environ.get(name, "") or default)
    except ValueError:
        return default
    return v if v >= 0 else default


def enabled() -> bool:
    return (os.environ.get(ENV, "1") or "1").strip().lower() not in ("0", "false", "no", "off")


class Tracker:
    """The state between two decode-advancing batches. Pure host bookkeeping."""

    def __init__(self, bound_ms: float, census_s: float, now: float) -> None:
        self.bound_ms = float(bound_ms)
        self.census_s = float(census_s)
        self.last_decode_t: Optional[float] = None
        self.last_rows = 0
        self.last_wake_seq = None
        self.ext_open: Optional[float] = None
        self.ext_ms = 0.0
        self.ext_n = 0
        self.ext_tokens = 0
        self.notes: Dict[str, int] = {}
        self.window_t0 = now
        self.census: Dict[str, list] = {c: [0, 0.0] for c in CAUSES}
        self.rounds = 0
        self.gaps = 0

    def note(self, cause: str) -> None:
        self.notes[cause] = self.notes.get(cause, 0) + 1

    def _close_extend(self, now: float) -> None:
        if self.ext_open is not None:
            self.ext_ms += max(0.0, now - self.ext_open) * 1000.0
            self.ext_open = None

    def on_batch(self, now: float, decode_rows: int, extend_tokens: int, wake_seq) -> Optional[dict]:
        """One batch at its launch. ``decode_rows`` > 0 = it advances the running decodes (decode or MIXED-D);
        ``extend_tokens`` > 0 with no decode rows = a pure extend. Returns the gap record when one closed."""
        self._close_extend(now)
        rec = None
        if decode_rows > 0:
            self.rounds += 1
            if self.last_decode_t is not None and self.last_rows > 0:
                gap_ms = (now - self.last_decode_t) * 1000.0
                if gap_ms >= self.bound_ms:
                    flip = wake_seq is not None and self.last_wake_seq is not None and wake_seq != self.last_wake_seq
                    rec = self._record(gap_ms, flip)
            self.last_decode_t = now
            self.last_rows = int(decode_rows)
            self.last_wake_seq = wake_seq
            self._reset_between()
        elif extend_tokens > 0:
            self.ext_open = now
            self.ext_n += 1
            self.ext_tokens += int(extend_tokens)
        return rec

    def _record(self, gap_ms: float, flip: bool) -> dict:
        notes = dict(self.notes)
        if flip:
            cause = "flip"
        elif self.ext_n > 0 and self.ext_ms >= 0.5 * gap_ms:
            cause = "d_prefill"
        else:
            cause = next((c for c in NOTE_CAUSES if c in notes), "d_prefill" if self.ext_n else "other")
        other_ms = max(0.0, gap_ms - self.ext_ms)
        rec = {"gap_ms": gap_ms, "cause": cause, "rows": self.last_rows, "d_prefill_ms": self.ext_ms,
               "extends": self.ext_n, "prefill_tokens": self.ext_tokens, "notes": notes, "other_ms": other_ms,
               "flip": flip}
        self.gaps += 1
        self.census[cause][0] += 1
        self.census[cause][1] += gap_ms
        return rec

    def _reset_between(self) -> None:
        self.ext_open = None
        self.ext_ms = 0.0
        self.ext_n = 0
        self.ext_tokens = 0
        self.notes = {}

    def census_due(self, now: float) -> Optional[dict]:
        if self.census_s <= 0 or now - self.window_t0 < self.census_s:
            return None
        out = {"window_s": now - self.window_t0, "gaps": self.gaps, "rounds": self.rounds,
               "by_cause": {c: (v[0], v[1]) for c, v in self.census.items() if v[0]}}
        self.window_t0 = now
        self.census = {c: [0, 0.0] for c in CAUSES}
        self.rounds = 0
        self.gaps = 0
        return out


def _is_log_rank(sched) -> bool:
    try:
        return int(getattr(sched, "tp_rank", 0) or 0) == 0
    except (TypeError, ValueError):
        return False


def tracker(sched) -> Optional[Tracker]:
    """The tracker of this scheduler: group D, first TP rank, switch on; else None (and nothing runs)."""
    tr = getattr(sched, _ATTR, None)
    if tr is not None:
        return tr if tr is not False else None
    ok = (enabled() and os.environ.get("FLLIPER_PDFLIP_GROUP", "").strip().upper() == "D" and _is_log_rank(sched))
    tr = Tracker(_env_float(ENV_MS, 400.0), _env_float(ENV_CENSUS_S, 300.0), time.monotonic()) if ok else False
    try:
        setattr(sched, _ATTR, tr)
    except Exception:  # noqa: BLE001 -- a slotted double
        return None
    return tr if tr is not False else None


def note(sched, cause: str) -> None:
    """A cause seen between two decode rounds (hook sites: skip-first hold, F3 gate, seat gate, wake read)."""
    try:
        tr = tracker(sched)
        if tr is not None:
            tr.note(str(cause))
    except Exception:  # noqa: BLE001 -- an instrument never breaks the pass
        pass


def _batch_terms(batch):
    mode = getattr(batch, "forward_mode", None)
    if mode is None:
        return 0, 0
    try:
        if mode.is_decode():
            return int(batch.batch_size()), 0
        if mode.is_extend():
            mixed = getattr(batch, "decoding_reqs", None) or ()
            return len(mixed), int(getattr(batch, "extend_num_tokens", 0) or 0)
    except Exception:  # noqa: BLE001
        return 0, 0
    return 0, 0


def on_batch(sched, batch, now: Optional[float] = None) -> None:
    """The run_batch funnel (every forward passes it once). Logs a closed gap and, per window, the census."""
    try:
        tr = tracker(sched)
        if tr is None:
            return
        now = time.monotonic() if now is None else float(now)
        rows, tokens = _batch_terms(batch)
        rec = tr.on_batch(now, rows, tokens, getattr(sched, "_pdflip_wake_seq", None))
        if rec is not None:
            logger.info(
                "%s gap_ms=%.0f cause=%s rows=%d d_prefill_ms=%.0f extends=%d prefill_tokens=%d other_ms=%.0f "
                "notes=%s (start-to-start between two decode-advancing batches; D group, rank-local)",
                MARK, rec["gap_ms"], rec["cause"], rec["rows"], rec["d_prefill_ms"], rec["extends"],
                rec["prefill_tokens"], rec["other_ms"], ",".join("%s:%d" % kv for kv in sorted(rec["notes"].items()))
                or "-")
        cen = tr.census_due(now)
        if cen is not None:
            logger.info(
                "%s CENSUS window_s=%.0f rounds=%d gaps=%d total_ms=%.0f by_cause=%s",
                MARK, cen["window_s"], cen["rounds"], cen["gaps"], sum(v[1] for v in cen["by_cause"].values()),
                " ".join("%s:%d/%.0fms" % (c, v[0], v[1]) for c, v in cen["by_cause"].items()) or "-")
    except Exception:  # noqa: BLE001 -- an instrument never breaks the pass
        pass
