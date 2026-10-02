"""#239 S3f miss record of the device-planned expert pool (group D), per rank.

``SGLANG_WEG2_OWNED_MISS_RECORD=<records root of the line>`` (unset = off; the
launcher sets it for D by default; the layout of the VRAM contract M3,
``records/<line>/<model_id>/<kind>``): each D rank accumulates, over its
phase, PAIRED per timed prefill forward (PR, 30.09.; see below):

* the device ms of THAT forward's host-plan expert fetches -- the collective
  clock's ``pool.host_fetch`` family (one span per ``_fetch``) of the
  per-rank prefill line's split-known, #691-paired duration
  (``RankPrefillLog.flush``), and
* the expert rows THOSE fetches loaded -- the host plan's own ``fetch_plan``
  (PR2, see below): no device counter, no host read, no decode row,

and writes one JSON record at D's sleep into ``<root>/<model_id>/owned_miss/``
(:func:`record_dir_for`), next to the #276 heat record, before any pause. ``ms per missed row`` = fetch ms / missed rows is the cost the owned
solve (``planner.expert_residency.solve_owned_cut``) prices a rank's round
with (``missed rows per layer x MoE layers x ms per row``); the launcher reads
these records (``read_owned_miss_rank_records`` on the same
:func:`record_dir_for`) instead of the seed.

A phase without a paired forward writes NOTHING (named in one warning) -- a
cost from half the numbers would be invented. Off, nothing is counted: one
cached bool test per timed prefill forward and one ``None`` test per fetch.
"""

from __future__ import annotations

import json
from contextlib import nullcontext
import logging
import os
import re
import time
from typing import Any, Dict, Mapping, Optional

logger = logging.getLogger(__name__)

MARKER = "OWNED-MISS-COST (#239 S3f)"
RECORD_KIND = "owned_miss_rank"
RECORD_VERSION = 3
RECORD_SUBDIR = "owned_miss"

_DIR: Optional[str] = None
_READ = False


def model_id(model: Optional[str]) -> str:
    """The ``<model_id>`` path segment of a checkpoint: its directory name,
    reduced to ``[A-Za-z0-9._-]`` (the same name on the host and in a
    container mount); ``unknown-model`` without a path."""
    name = os.path.basename(str(model or "").rstrip("/").strip("'\""))
    return re.sub(r"[^A-Za-z0-9._-]", "_", name) or "unknown-model"


def record_dir_for(root: str, model: Optional[str]) -> str:
    """``<root>/<model_id>/owned_miss`` -- the ONE producer of the path the
    ranks write and the launcher reads."""
    return os.path.join(root, model_id(model), RECORD_SUBDIR)


def record_dir() -> Optional[str]:
    """The records root of the line, or ``None`` when off (read once per
    process)."""
    global _DIR, _READ
    if not _READ:
        try:
            from sglang.srt.environ import envs

            value = (envs.SGLANG_WEG2_OWNED_MISS_RECORD.get() or "").strip()
        except Exception:  # noqa: BLE001 - an instrument never kills a boot
            value = ""
        _DIR = value or None
        _READ = True
    return _DIR


def _reset() -> None:
    _PAIR.update(fetch_ms=0.0, miss_rows=0, forwards=0, refused=0)


def _reset_for_test() -> None:
    global _DIR, _READ
    _DIR, _READ = None, False
    _reset()


# ---------------------------------------------------------------------------
# The PAIRED record -- one timed prefill forward's host->device expert fetch
# ms against the rows THE SAME fetches loaded.
#
# PR (30.09., 363c173670) paired the forward's ``pool.fetch`` family with the
# misses of the pool syncs inside its window. Both halves were foreign: a
# device-planned eager forward (D-Mini-Extend) never syncs, so its misses
# surface at the NEXT host-plan sync, among the graphed decode steps; and the
# host-planned forward (``_run_eager_host_plan`` -> run_waves) -- the big D
# prefills -- fetches through ``_fetch``, which carried no clock span, while
# its sync reports only the device steps before it. So no window paired
# anything, and a mixed one (device-step layers + host-plan layers) could pair
# one forward's fetch with earlier decode misses.
#
# PR2: the host plan KNOWS the rows it reloads (``fetch_plan``). Each
# ``_fetch`` inside a window adds its rows here and runs its copies under ONE
# clock span of the family ``pool.host_fetch`` (device events, no host read);
# the prefill line sums that family only. Numerator and denominator are the
# same calls: the pair holds when the harvested span count equals the window's
# fetch count. Device-step layers stay out of both halves (their rows are
# device counters, readable only by a host sync), pool syncs feed nothing.
# ---------------------------------------------------------------------------

PAIRING = "prefill_hostfetch_v3"
HOST_FETCH_FAMILY = "pool.host_fetch"
#: A window times at most this many fetches -- beyond it the forward is
#: dropped whole (never half a pair), and a token-major prefill's thousands of
#: waves cannot load the forward with events.
MAX_WINDOW_FETCHES = 4096
_WIN: Optional[Dict[str, Any]] = None
_PAIR: Dict[str, float] = {"fetch_ms": 0.0, "miss_rows": 0, "forwards": 0, "refused": 0}


def open_window(holder: Dict[str, Any]) -> None:
    """A timed prefill forward starts: its host-plan fetches fill ``holder``."""
    global _WIN
    if record_dir() is None:
        return
    holder.update(rows=0, fetches=0, clean=True)
    _WIN = holder


def close_window() -> None:
    global _WIN
    _WIN = None


def window_open() -> bool:
    return _WIN is not None


def host_fetch_span(rows: int, clock=None):
    """Around ONE host-plan ``_fetch`` (its copies and the join): inside an
    open window, add its ``rows`` and return the clock's ``pool.host_fetch``
    span; else a no-op context. A window whose clock is not armed (a refused
    arming) or that is past MAX_WINDOW_FETCHES is marked unclean -- dropped
    whole at the pairing. ``clock``: the process's collective clock unless
    given."""
    w = _WIN
    if w is None or int(rows) <= 0:
        return nullcontext()
    if clock is None:
        from sglang.srt.utils.collective_clock import collective_clock

        clock = collective_clock()
    if not clock.armed or int(w["fetches"]) >= MAX_WINDOW_FETCHES:
        w["clean"] = False
        return nullcontext()
    w["rows"] += int(rows)
    w["fetches"] += 1
    return clock.span(HOST_FETCH_FAMILY)


def note_paired(*, fetch_ms: float, fetch_count: int, window: Optional[Mapping[str, Any]]) -> bool:
    """The prefill line paired one duration with its record: add the forward's
    ``pool.host_fetch`` ms and the rows those fetches loaded. Refused (counted,
    nothing added) when the window is missing or empty, unclean, when the
    harvested span count differs from the window's fetch count (not the same
    calls), or when the forward fetched nothing."""
    if record_dir() is None:
        return False
    ok = (window is not None and bool(window.get("clean")) and int(window.get("rows", 0)) > 0
          and int(window.get("fetches", 0)) > 0 and int(fetch_count) == int(window["fetches"])
          and float(fetch_ms) > 0.0)
    if not ok:
        _PAIR["refused"] += 1
        return False
    _PAIR["fetch_ms"] += float(fetch_ms)
    _PAIR["miss_rows"] += int(window["rows"])
    _PAIR["forwards"] += 1
    return True


def rank_record(*, rank: int, group: str, reason: str, model: Optional[str],
                phase_index: Optional[int] = None) -> Optional[Dict[str, Any]]:
    """The PAIRED record of the phase so far, or ``None`` when a number is missing."""
    rounds = int(_PAIR["forwards"])
    rows = int(_PAIR["miss_rows"])
    fetch = float(_PAIR["fetch_ms"])
    if rounds <= 0 or rows <= 0 or fetch <= 0.0:
        return None
    return {
        "kind": RECORD_KIND,
        "version": RECORD_VERSION,
        "group": group,
        "rank": int(rank),
        "reason": reason,
        "phase_index": phase_index,
        "model": model,
        "time_unix": time.time(),
        "fetch_ms": round(fetch, 3),
        "miss_rows": rows,
        "rounds": rounds,
        "forwards": rounds,
        "paired_refused": int(_PAIR["refused"]),
        "pairing": PAIRING,
        "ms_per_row": fetch / rows,
        "scope": "pool.host_fetch ms of timed prefill forwards' host-plan fetches / the rows "
                 "the SAME fetches loaded (device-step layers and decode rows in neither)",
    }


def flush(*, rank: int, group: str, reason: str, model: Optional[str],
          phase_index: Optional[int] = None) -> Optional[str]:
    """At D's phase end: write the record, zero the counters. Returns the path;
    ``None`` when off, when a number is missing, or on any error (logged,
    never raised)."""
    try:
        directory = record_dir()
        if directory is None:
            return None
        rec = rank_record(rank=rank, group=group, reason=reason, model=model,
                          phase_index=phase_index)
        if rec is None:
            logger.warning(
                "%s rank %d: no record (paired prefill forwards=%d, refused=%d, missed rows=%d, "
                "fetch_ms=%.1f) -- no timed prefill forward of this phase ran a host-plan fetch "
                "the clock timed",
                MARKER, rank, int(_PAIR["forwards"]), int(_PAIR["refused"]),
                int(_PAIR["miss_rows"]), _PAIR["fetch_ms"])
            _reset()
            return None
        directory = record_dir_for(directory, model)
        os.makedirs(directory, exist_ok=True)
        stamp = time.strftime("%Y%m%dT%H%M%S", time.gmtime(rec["time_unix"]))
        path = os.path.join(directory, "owned_miss_%s_tp%d_%s_p%s.json" % (
            group, int(rank), stamp, "-" if phase_index is None else int(phase_index)))
        tmp = path + ".tmp"
        with open(tmp, "w") as fh:
            json.dump(rec, fh)
        os.replace(tmp, path)
        logger.info("%s wrote %s ms_per_row=%.4f rows=%d paired_forwards=%d refused=%d", MARKER,
                    path, rec["ms_per_row"], rec["miss_rows"], rec["rounds"], rec["paired_refused"])
        _reset()
        return path
    except Exception as exc:  # noqa: BLE001 - an instrument never kills a sleep
        logger.warning("%s flush failed: %s: %s", MARKER, type(exc).__name__, exc)
        return None
