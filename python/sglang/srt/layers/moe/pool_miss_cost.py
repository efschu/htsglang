"""#239 S3f miss record of the device-planned expert pool (group D), per rank.

``SGLANG_WEG2_OWNED_MISS_RECORD=<records root of the line>`` (unset = off; the
launcher sets it for D by default; the layout of the VRAM contract M3,
``records/<line>/<model_id>/<kind>``): each D rank accumulates, over its
phase, PAIRED per timed prefill forward (PR, 30.09.; see below):

* the device ms of the pool's host->device expert traffic of THAT forward --
  the collective clock's ``pool.fetch`` family of the per-rank prefill line's
  split-known, #691-paired duration (``RankPrefillLog.flush``), and
* the expert rows THAT forward missed -- the pool's own per-layer counters
  (``take_report`` in ``sync_pool_from_host``) of the syncs inside the
  forward's window, every pool layer, only when each reported exactly one
  forward since its last sync (no graphed decode row in the denominator),

and writes one JSON record at D's sleep into ``<root>/<model_id>/owned_miss/``
(:func:`record_dir_for`), next to the #276 heat record, before any pause. ``ms per missed row`` = fetch ms / missed rows is the cost the owned
solve (``planner.expert_residency.solve_owned_cut``) prices a rank's round
with (``missed rows per layer x MoE layers x ms per row``); the launcher reads
these records (``read_owned_miss_rank_records`` on the same
:func:`record_dir_for`) instead of the seed.

A phase without a paired forward writes NOTHING (named in one warning) -- a
cost from half the numbers would be invented. Off, nothing is counted: one
cached bool test per pool sync and per timed prefill forward.
"""

from __future__ import annotations

import json
import logging
import os
import re
import time
from typing import Any, Dict, Mapping, Optional

logger = logging.getLogger(__name__)

MARKER = "OWNED-MISS-COST (#239 S3f)"
RECORD_KIND = "owned_miss_rank"
RECORD_VERSION = 2
FETCH_FAMILY = "pool.fetch"
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


def note_sync(forwards: int, misses: int) -> None:
    """One pool layer's counters since its last sync (every layer, not the
    sampled ones the log line prints)."""
    if record_dir() is None:
        return
    _note_window(forwards, misses)


# ---------------------------------------------------------------------------
# PR (30.09.): the PAIRED record -- one timed prefill forward's pool.fetch ms
# against the rows THE SAME forward missed.
#
# The record above paired the fetch ms of split decode rounds with the misses
# of EVERY pool sync: a sync after an eager forward reports the device
# counters since the previous sync, i.e. the graphed decode rounds in between
# as well, whose fetch time no production round can split (graph reader off).
# Five NF boots (y3u-y3z) never wrote one: D-EIGENTUM priced its misses with
# the seed. Here a window opens and closes around each timed prefill forward
# (the per-rank prefill timer's bracket, metrics_reporter), the pool syncs of
# that forward land in it, and the forward's own pool.fetch family is added
# only when the prefill line pairs its duration (the #691 guard holds) and
# every pool layer reported exactly ONE forward since its last sync -- no
# decode step in between, so no graphed decode row reaches the denominator.
# ---------------------------------------------------------------------------

PAIRING = "prefill_window_v2"
_WIN: Optional[Dict[str, Any]] = None
_PAIR: Dict[str, float] = {"fetch_ms": 0.0, "miss_rows": 0, "forwards": 0, "refused": 0}


def open_window(holder: Dict[str, Any]) -> None:
    """A timed prefill forward starts: its pool syncs fill ``holder``."""
    global _WIN
    if record_dir() is None:
        return
    holder.update(layers=0, misses=0, clean=True)
    _WIN = holder


def close_window() -> None:
    global _WIN
    _WIN = None


def _note_window(forwards: int, misses: int) -> None:
    w = _WIN
    if w is None:
        return
    w["layers"] += 1
    w["misses"] += int(misses)
    if int(forwards) != 1:
        w["clean"] = False  # a decode step since the last sync: not this forward's rows alone


def note_paired(*, fetch_ms: float, fetch_count: int, window: Optional[Mapping[str, Any]]) -> bool:
    """The prefill line paired one duration with its record: add the forward's
    pool.fetch ms and its own missed rows. Refused (counted, nothing added)
    when the window is missing or empty (a forward the window did not cover),
    contaminated by a decode step, or when the forward fetched nothing."""
    if record_dir() is None:
        return False
    ok = (window is not None and int(window.get("layers", 0)) > 0 and bool(window.get("clean"))
          and int(window.get("misses", 0)) > 0 and int(fetch_count) > 0 and float(fetch_ms) > 0.0)
    if not ok:
        _PAIR["refused"] += 1
        return False
    _PAIR["fetch_ms"] += float(fetch_ms)
    _PAIR["miss_rows"] += int(window["misses"])
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
        "scope": "pool.fetch of timed prefill forwards / the rows the SAME forward missed "
                 "(no decode step since the last sync)",
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
                "fetch_ms=%.1f) -- no timed prefill forward of this phase fetched pool rows "
                "without a decode step since the last sync",
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
