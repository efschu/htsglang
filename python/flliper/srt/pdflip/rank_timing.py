"""RANK-TIMING (user 01.10. ~08:45Z, expert view of the dashboard): the rank's
PLE latency and the L2 / L3 fetch-back times as IPC -- the rankstats file
(pdflip/rankstats.py) reads them; before this the rankstats ``cache`` block had
counters only (no duration, no bytes, no PLE).

Stage names as in ``request_done.cached``: device, host (= L2), storage (= L3),
l15.

* ``ple.prefill`` / ``ple.decode`` = {n, ms_sum, ms_max, last_ms, last_t, hit_n,
  miss_n, wait_ms_sum, bytes}; ``ple.recent`` = [[t, ms, hit, miss, phase], ...]
  (the last :data:`RECENT_KEEP`). Prefill: one per PLE gather of a prefill
  chunk (the serial pread gather, or the H32 prefetch gather: hit = rows the
  prefetch had ready). Decode: one per verify round's PLE stage (H40: the host
  gap the round waits; hit = rows staged in budget, miss = the rest) or per
  posted round (H73 autonomous: the post's host ms, hit/miss on the device).
* ``loadback`` (L2 host arena -> device): ms = the load stream's start ->
  finish event (device clock, read only once the ack's event has landed --
  the tree's ``loading_check`` already holds it, no new sync), pages, bytes.
* ``l3`` (store / disk -> host): one per prefetch read (the aux IO thread's
  ``_page_transfer``): read_ms = the read's own clock, pages, bytes; and
  ``prefetch`` = issue -> read end (queue + read) with ``landed`` = reads that
  landed at least one page.
* ``l15``: the schema and its writer (:func:`note_l15`); the block exists only
  once an L1,5 stage has written it -- no stage, no key (never zeros).
* ``vision`` (Nutzer 02.10. ~11:00Z: "der visiontower laden rechnen entladen
  auch mit in die phasenliste ins dashboard"): the transient tower stage on
  P's PP0 (pdflip/vision_rank_runner.py). ``live`` = {run, leg, since, rids}
  while a stage runs (the leg it is in, wall clock), null between stages;
  ``runs`` = the stages so far; ``recent`` = the last :data:`RECENT_KEEP`
  stages as {run, ok, code, t0, t1, rids, tower_mib, place, card,
  legs{leg: [t0, t1, ms]}} (wall clock per leg: build, reserve, load, encode,
  attach, teardown). Only on a rank that ran a stage.

RULES (operator 01.10.): host timestamps the paths take anyway, or device
events that have already landed; no CUDA sync, no lock, no I/O. A note is a
handful of dict operations in the calling thread; the rankstats timer copies
(``dict()`` / ``list()`` of builtins hold the GIL for the copy, so a reader
never sees a container mid-resize; a sum may lead its count by one sample).
Module state = this process = this rank (P and D are separate processes).
"""
from __future__ import annotations

import collections
import time
from typing import Any, Dict, Optional

#: entries kept in every ``recent`` list
RECENT_KEEP = 32
PLE_PHASES = ("prefill", "decode")


def _agg() -> Dict[str, Any]:
    return {"n": 0, "ms_sum": 0.0, "ms_max": 0.0, "last_ms": None, "last_t": None}


def _add(a: Dict[str, Any], ms: float, t: float) -> None:
    a["n"] += 1
    a["ms_sum"] += ms
    if ms > a["ms_max"]:
        a["ms_max"] = ms
    a["last_ms"] = ms
    a["last_t"] = t


class _State:
    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self.ple: Dict[str, Dict[str, Any]] = {}
        self.ple_recent: collections.deque = collections.deque(maxlen=RECENT_KEEP)
        self.loadback: Optional[Dict[str, Any]] = None
        self.loadback_recent: collections.deque = collections.deque(maxlen=RECENT_KEEP)
        self.l3: Optional[Dict[str, Any]] = None
        self.l3_recent: collections.deque = collections.deque(maxlen=RECENT_KEEP)
        self.prefetch: Optional[Dict[str, Any]] = None
        self.l15: Optional[Dict[str, Any]] = None
        self.vision_live: Optional[Dict[str, Any]] = None
        self.vision_runs = 0
        self.vision_recent: collections.deque = collections.deque(maxlen=RECENT_KEEP)


_S = _State()


def reset() -> None:
    """Tests only: a fresh process view."""
    _S.reset()


def _r(x: Optional[float], nd: int = 2) -> Optional[float]:
    return None if x is None else round(float(x), nd)


# ---------------- writers (the paths) ----------------
def note_ple(phase: str, ms: float, *, hit: int = 0, miss: int = 0, wait_ms: float = 0.0,
             nbytes: int = 0, t: Optional[float] = None) -> None:
    """One PLE gather (prefill chunk) or verify-round stage (decode)."""
    t = time.time() if t is None else float(t)
    a = _S.ple.get(phase)
    if a is None:
        a = _S.ple[phase] = dict(_agg(), hit_n=0, miss_n=0, wait_ms_sum=0.0, bytes=0)
    ms = float(ms)
    _add(a, ms, t)
    a["hit_n"] += int(hit)
    a["miss_n"] += int(miss)
    a["wait_ms_sum"] += float(wait_ms)
    a["bytes"] += int(nbytes)
    _S.ple_recent.append([round(t, 3), round(ms, 2), int(hit), int(miss), phase])


def note_loadback(ms: float, *, pages: int, nbytes: int, t: Optional[float] = None) -> None:
    """One L2 (host arena) -> device load whose finish event has landed."""
    t = time.time() if t is None else float(t)
    a = _S.loadback
    if a is None:
        a = _S.loadback = dict(_agg(), pages=0, bytes=0)
    ms = float(ms)
    _add(a, ms, t)
    a["pages"] += int(pages)
    a["bytes"] += int(nbytes)
    _S.loadback_recent.append([round(t, 3), round(ms, 2), int(pages), int(nbytes)])


def note_l3_read(read_ms: float, *, pages: int, nbytes: int, prefetch_ms: Optional[float] = None,
                 t: Optional[float] = None) -> None:
    """One prefetch read from the store (L3) into the host pool, at its end."""
    t = time.time() if t is None else float(t)
    a = _S.l3
    if a is None:
        a = _S.l3 = dict(_agg(), pages=0, bytes=0)
    ms = float(read_ms)
    _add(a, ms, t)
    a["pages"] += int(pages)
    a["bytes"] += int(nbytes)
    _S.l3_recent.append([round(t, 3), round(ms, 2), int(pages), int(nbytes)])
    p = _S.prefetch
    if p is None:
        p = _S.prefetch = dict(_agg(), bytes=0, landed=0, landed_pages=0, empty=0)
    _add(p, float(prefetch_ms if prefetch_ms is not None else read_ms), t)
    p["bytes"] += int(nbytes)
    if int(pages) > 0:
        p["landed"] += 1
        p["landed_pages"] += int(pages)
    else:
        p["empty"] += 1


#: the L1,5 block's fields (prepared; filled once the stage exists on the line)
L15_FIELDS = ("kv_pages", "anchors", "bytes", "hit_n", "hit_tokens", "p2p_read_n",
              "p2p_read_ms_sum", "p2p_bytes", "evict_n")


def note_l15(**fields: Any) -> None:
    """The L1,5 stage's state (gauges: kv_pages, anchors, bytes) and its
    counters (the rest, ADDED). Unknown keys are refused by name."""
    bad = [k for k in fields if k not in L15_FIELDS]
    if bad:
        raise KeyError(f"rank_timing.note_l15: unknown fields {bad} (schema {L15_FIELDS})")
    a = _S.l15
    if a is None:
        a = _S.l15 = {k: 0 for k in L15_FIELDS}
    for k, v in fields.items():
        if k in ("kv_pages", "anchors", "bytes"):
            a[k] = v
        else:
            a[k] = a[k] + v


def note_vision_leg(run: int, leg: str, rids, t: Optional[float] = None) -> None:
    """The tower stage entered ``leg`` (wall clock ``t``): the live phase."""
    _S.vision_live = {"run": int(run), "leg": str(leg),
                      "since": round(time.time() if t is None else float(t), 3),
                      "rids": [str(r) for r in (rids or ())][:8]}


def note_vision_run(run: int, *, ok: bool, code: str, rids, legs_wall: Dict[str, Any],
                    legs_ms: Dict[str, float], tower_bytes: int = 0, place: str = "",
                    card: Optional[int] = None, t: Optional[float] = None) -> None:
    """One finished tower stage. ``legs_wall`` = {leg: (t0, t1)} wall clock; a
    leg only in ``legs_ms`` (the async form's worker legs) is laid back from
    the stage end ``t`` in leg order."""
    t1 = time.time() if t is None else float(t)
    legs: Dict[str, list] = {}
    for k, w in (legs_wall or {}).items():
        a, b = float(w[0]), float(w[1])
        legs[k] = [round(a, 3), round(b, 3), round((b - a) * 1e3, 1)]
    end = t1
    for k in reversed(list(legs_ms or {})):
        if k in legs:
            end = min(end, legs[k][0])
            continue
        ms = float(legs_ms[k])
        legs[k] = [round(end - ms / 1e3, 3), round(end, 3), round(ms, 1)]
        end -= ms / 1e3
    t0 = min([v[0] for v in legs.values()] or [t1])
    _S.vision_runs += 1
    _S.vision_recent.append({"run": int(run), "ok": bool(ok), "code": str(code or ""),
                             "t0": round(t0, 3), "t1": round(t1, 3),
                             "rids": [str(r) for r in (rids or ())][:8],
                             "tower_mib": round(int(tower_bytes or 0) / (1 << 20), 1),
                             "place": str(place or ""), "card": card, "legs": legs})
    _S.vision_live = None


# ---------------- readers (the rankstats timer) ----------------
def _agg_out(a: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    if a is None:
        return None
    o = dict(a)
    for k in ("ms_sum", "ms_max", "last_ms", "wait_ms_sum"):
        if k in o:
            o[k] = _r(o[k])
    o["last_t"] = _r(o.get("last_t"), 3)
    return o


def ple_block() -> Optional[Dict[str, Any]]:
    """rankstats ``ple``; None while this rank has gathered no PLE row."""
    ple = dict(_S.ple)
    if not ple:
        return None
    out: Dict[str, Any] = {ph: _agg_out(ple.get(ph)) for ph in PLE_PHASES}
    out["recent"] = list(_S.ple_recent)
    return out


def cache_fields() -> Dict[str, Any]:
    """The keys rankstats merges into its ``cache`` block (each only once
    observed: a rank without load-backs / L3 reads has no such key)."""
    out: Dict[str, Any] = {}
    lb = _S.loadback
    if lb is not None:
        lb = dict(lb)
        out["loadback_ms_sum"] = _r(lb["ms_sum"])
        out["loadback_ms_max"] = _r(lb["ms_max"])
        out["loadback_pages"] = lb["pages"]
        out["loadback_bytes"] = lb["bytes"]
        out["loadback_count"] = lb["n"]
        rec = list(_S.loadback_recent)
        last = rec[-1] if rec else None
        out["loadback_last"] = (None if last is None else
                                {"t": last[0], "ms": last[1], "pages": last[2], "bytes": last[3]})
        out["loadback_recent"] = rec
    l3 = _S.l3
    if l3 is not None:
        l3 = dict(l3)
        rec = list(_S.l3_recent)
        last = rec[-1] if rec else None
        out["l3"] = {"read_n": l3["n"], "read_ms_sum": _r(l3["ms_sum"]), "read_ms_max": _r(l3["ms_max"]),
                     "read_pages": l3["pages"], "read_bytes": l3["bytes"],
                     "last": (None if last is None else
                              {"t": last[0], "ms": last[1], "pages": last[2], "bytes": last[3]}),
                     "recent": rec}
    if _S.l15 is not None:
        out["l15"] = dict(_S.l15)
    return out


def vision_block() -> Optional[Dict[str, Any]]:
    """rankstats ``vision``; None on a rank that never ran a tower stage."""
    if not _S.vision_runs and _S.vision_live is None:
        return None
    live = _S.vision_live
    return {"runs": _S.vision_runs, "live": None if live is None else dict(live),
            "recent": list(_S.vision_recent)}


def prefetch_fields() -> Dict[str, Any]:
    """The keys rankstats merges into ``cache.prefetch`` ({} before the first read)."""
    p = _S.prefetch
    if p is None:
        return {}
    p = dict(p)
    return {"ms_sum": _r(p["ms_sum"]), "ms_max": _r(p["ms_max"]), "bytes": p["bytes"],
            "last_ms": _r(p["last_ms"]), "reads": p["n"], "landed": p["landed"],
            "landed_pages": p["landed_pages"], "empty": p["empty"]}
