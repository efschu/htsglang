"""#239 S3f miss record of the device-planned expert pool (D decode), per rank.

``SGLANG_WEG2_OWNED_MISS_RECORD=<records root of the line>`` (default unset =
off; the layout of the VRAM contract M3, ``records/<line>/<model_id>/<kind>``):
each D rank accumulates, over its decode phase,

* the device ms of the pool's host->device expert traffic -- the collective
  clock's ``pool.fetch`` family (``spec_verify:`` folded) of every decode
  round whose split is known (``DecodeRoundLog._emit``; graphed rounds need
  ``SGLANG_DEBUG_COLLECTIVE_CLOCK_GRAPH_NODES=1``), and
* the expert rows it missed -- the pool's own per-layer counters
  (``take_report`` in ``sync_pool_from_host``), summed over EVERY pool layer,

and writes one JSON record at D's sleep into ``<root>/<model_id>/owned_miss/``
(:func:`record_dir_for`), next to the #276 heat record, before any pause. ``ms per missed row`` = fetch ms / missed rows is the cost the owned
solve (``planner.expert_residency.solve_owned_cut``) prices a rank's round
with (``missed rows per layer x MoE layers x ms per row``); the launcher reads
these records (``read_owned_miss_rank_records`` on the same
:func:`record_dir_for`) instead of the seed.

A phase without a split round or without a miss writes NOTHING (named in one
warning) -- a cost from half the numbers would be invented. Off, nothing is
counted: one cached bool test per round and per pool sync.
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
RECORD_VERSION = 1
FETCH_FAMILY = "pool.fetch"
RECORD_SUBDIR = "owned_miss"

_DIR: Optional[str] = None
_READ = False
_ACC: Dict[str, float] = {"fetch_ms": 0.0, "rounds": 0, "miss_rows": 0, "forwards": 0}


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
    _ACC.update(fetch_ms=0.0, rounds=0, miss_rows=0, forwards=0)


def _reset_for_test() -> None:
    global _DIR, _READ
    _DIR, _READ = None, False
    _reset()


def note_round(families: Mapping[str, Any]) -> None:
    """One split decode round: add its ``pool.fetch`` device ms. ``families``
    maps family name -> ``[total_ms, count, ...]`` (DecodeRoundLog)."""
    if record_dir() is None:
        return
    ms = 0.0
    for name, acc in families.items():
        if str(name).split(":")[-1] == FETCH_FAMILY:
            ms += float(acc[0])
    _ACC["fetch_ms"] += ms
    _ACC["rounds"] += 1


def note_sync(forwards: int, misses: int) -> None:
    """One pool layer's counters since its last sync (every layer, not the
    sampled ones the log line prints)."""
    if record_dir() is None:
        return
    _ACC["miss_rows"] += int(misses)
    _ACC["forwards"] += int(forwards)


def rank_record(*, rank: int, group: str, reason: str, model: Optional[str],
                phase_index: Optional[int] = None) -> Optional[Dict[str, Any]]:
    """The record of the phase so far, or ``None`` when a number is missing."""
    rounds = int(_ACC["rounds"])
    rows = int(_ACC["miss_rows"])
    fetch = float(_ACC["fetch_ms"])
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
        "forwards": int(_ACC["forwards"]),
        "ms_per_row": fetch / rows,
        "scope": "pool.fetch of split decode rounds / missed rows of every pool layer",
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
                "%s rank %d: no record (split rounds=%d, missed rows=%d, fetch_ms=%.1f) -- "
                "the graph reader must be on (SGLANG_DEBUG_COLLECTIVE_CLOCK_GRAPH_NODES=1)",
                MARKER, rank, int(_ACC["rounds"]), int(_ACC["miss_rows"]), _ACC["fetch_ms"])
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
        logger.info("%s wrote %s ms_per_row=%.4f rows=%d rounds=%d", MARKER, path,
                    rec["ms_per_row"], rec["miss_rows"], rec["rounds"])
        _reset()
        return path
    except Exception as exc:  # noqa: BLE001 - an instrument never kills a sleep
        logger.warning("%s flush failed: %s: %s", MARKER, type(exc).__name__, exc)
        return None
