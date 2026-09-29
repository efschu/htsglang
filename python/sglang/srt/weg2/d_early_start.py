"""BOOTZEIT 3 (29.09.): group D starts WITH group P and waits at a gate
before its weight load, instead of starting after P's READY + sleep + D plan.

MEASURED (z30r3, 256 s boot): D's init -- spawn, imports, dist init -- is
30 s (05:50:30 -> 05:51:00) and starts only after P's READY (05:50:22) and
sleep (8 s incl. the D plan). Nothing in that init needs P asleep: D touches
no weight byte and no store row before ``load_model``. What D's LOAD needs
is (a) P's cards free up to D's budget (P asleep) and (b) P's H2 sentinels
written (P's presplit is over before its READY, so (a) implies (b)).

The protocol, one file, no log line in the loop (IPC never over logs):

  * the launcher starts D with ``SGLANG_WEG2_D_EARLY_GATE=<path>`` and a D
    plan made from the planner's expectation budgets (the planner seat owns
    that input -- this module never computes a budget);
  * every D rank, at the top of its first ``load_model``, waits for the file;
  * after ``sleep(P)`` the launcher measures P's dormant footprint, derives
    the real D budgets exactly as the serial path does, and writes
    ``{"verdict": "go"}`` iff every card's PLANNED budget is <= its MEASURED
    one (D was planned with at most what it gets), else ``"refuse"``: the D
    ranks raise ``DEarlyGateRefused`` and exit, the launcher starts D the old
    serial way from the measured budgets.

STAGE 0 (27B review of cb98c3d94a): P sizes its KV pool from two LIVE
free-memory readings (before its load, model_runner.py:2291, and after it,
model_runner_kv_cache_mixin ``used_by_me = pre_model_load - available``).
A D CUDA context created between them is charged to P as its own use -- P's
KV pool shrinks silently and nondeterministically and its records lie. So D
must not create a context before P is SIZED:

  * P ranks, once every sizing reading of theirs is taken (after the
    post-capture leftover note), write one record each into
    ``SGLANG_WEG2_P_MEM_SIZED_DIR`` (``p_sized.pp<k>tp<r>.json``, with the
    ``used_by_me`` they charged);
  * D ranks wait in ``run_scheduler_process`` -- before the Scheduler, i.e.
    before the first CUDA call of the rank -- for the stage-0 file
    ``SGLANG_WEG2_D_EARLY_STAGE0`` (same go/refuse format as the load gate);
  * the launcher writes stage 0 ``go`` only when every P rank has reported
    AND no D process held VRAM on any card while it waited
    (:func:`stage0_verdict`), and records the event in state.json.

Default off (``--weg2-d-early-start``); without the env vars nothing here runs.
"""

from __future__ import annotations

import json
import logging
import os
import time
from typing import Dict, List, Mapping, Optional, Tuple

logger = logging.getLogger(__name__)

GATE_ENV = "SGLANG_WEG2_D_EARLY_GATE"
GATE_TIMEOUT_ENV = "SGLANG_WEG2_D_EARLY_GATE_TIMEOUT_S"
MARKER = "BZ3 D-EARLY-GATE"
VERDICT_GO = "go"
VERDICT_REFUSE = "refuse"

STAGE0_ENV = "SGLANG_WEG2_D_EARLY_STAGE0"
P_SIZED_DIR_ENV = "SGLANG_WEG2_P_MEM_SIZED_DIR"
STAGE0_MARKER = "BZ3 D-EARLY-STAGE0"

_PASSED: Dict[str, object] = {"path": None, "waited_s": None}
_STAGE0_PASSED: Dict[str, object] = {"path": None}


class DEarlyGateRefused(RuntimeError):
    """The launcher measured less room than D was planned with."""


class DEarlyGateTimeout(RuntimeError):
    """No verdict within the deadline (the launcher died or P never slept)."""


def write_gate(path: str, verdict: str, reason: str,
               planned: Optional[Mapping[str, int]] = None,
               measured: Optional[Mapping[str, int]] = None) -> None:
    """Atomic: a reader sees either no file or the whole verdict."""
    if verdict not in (VERDICT_GO, VERDICT_REFUSE):
        raise ValueError(f"verdict {verdict!r}")
    doc = {"verdict": verdict, "reason": str(reason), "t": time.time(),
           "planned": dict(planned or {}), "measured": dict(measured or {})}
    tmp = f"{path}.tmp.{os.getpid()}"
    with open(tmp, "w") as fh:
        json.dump(doc, fh, sort_keys=True)
        fh.flush()
        os.fsync(fh.fileno())
    os.replace(tmp, path)


def read_gate(path: str) -> Optional[dict]:
    try:
        with open(path) as fh:
            doc = json.load(fh)
    except FileNotFoundError:
        return None
    except (OSError, ValueError):
        return None  # a torn read cannot happen (os.replace); treat as absent
    return doc if isinstance(doc, dict) else None


def wait_gate(path: str, timeout_s: float, poll_s: float = 0.2,
              clock=time.monotonic, sleep=time.sleep) -> Tuple[dict, float]:
    t0 = clock()
    while True:
        doc = read_gate(path)
        if doc is not None:
            waited = clock() - t0
            if doc.get("verdict") == VERDICT_GO:
                return doc, waited
            raise DEarlyGateRefused(
                f"{MARKER}: launcher refused the early D start after {waited:.1f} s: "
                f"{doc.get('reason', '?')} -- this rank exits, the launcher starts D serially")
        if clock() - t0 > timeout_s:
            raise DEarlyGateTimeout(
                f"{MARKER}: no verdict in {path} after {timeout_s:.0f} s")
        sleep(poll_s)


def wait_gate_from_env() -> Optional[float]:
    """Loader hook: block the FIRST load of this process until the gate says
    go. No-op (None) without the env var, and after the first pass."""
    path = os.environ.get(GATE_ENV, "").strip()
    if not path:
        return None
    if _PASSED["path"] == path:
        return None
    timeout_s = float(os.environ.get(GATE_TIMEOUT_ENV, "1800") or 1800)
    logger.info("%s waiting path=%s timeout_s=%.0f (D init done; the load waits for P asleep)",
                MARKER, path, timeout_s)
    doc, waited = wait_gate(path, timeout_s)
    _PASSED["path"] = path
    _PASSED["waited_s"] = waited
    logger.info("%s PASSED waited_s=%.1f reason=%s", MARKER, waited, doc.get("reason", ""))
    return waited


def wait_stage0_from_env() -> Optional[float]:
    """Scheduler-process hook, BEFORE the rank's first CUDA call: hold until
    the launcher says P is sized (stage 0). No-op without the env var."""
    path = os.environ.get(STAGE0_ENV, "").strip()
    if not path or _STAGE0_PASSED["path"] == path:
        return None
    timeout_s = float(os.environ.get(GATE_TIMEOUT_ENV, "1800") or 1800)
    logger.info("%s waiting path=%s timeout_s=%.0f (no CUDA context until P's KV is sized)",
                STAGE0_MARKER, path, timeout_s)
    doc, waited = wait_gate(path, timeout_s)
    _STAGE0_PASSED["path"] = path
    logger.info("%s PASSED waited_s=%.1f reason=%s", STAGE0_MARKER, waited, doc.get("reason", ""))
    return waited


def note_p_memory_sized(pp_rank: int, tp_rank: int, used_by_me_gb: Optional[float]) -> Optional[str]:
    """P rank: every free-memory reading its sizing depends on is taken.
    One atomic record per rank; no-op without ``SGLANG_WEG2_P_MEM_SIZED_DIR``."""
    d = os.environ.get(P_SIZED_DIR_ENV, "").strip()
    if not d:
        return None
    os.makedirs(d, exist_ok=True)
    path = os.path.join(d, f"p_sized.pp{int(pp_rank)}tp{int(tp_rank)}.json")
    doc = {"pp_rank": int(pp_rank), "tp_rank": int(tp_rank), "pid": os.getpid(), "t": time.time(),
           "used_by_me_mib": (None if used_by_me_gb is None else int(round(float(used_by_me_gb) * 1024)))}
    tmp = f"{path}.tmp.{os.getpid()}"
    with open(tmp, "w") as fh:
        json.dump(doc, fh, sort_keys=True)
        fh.flush()
        os.fsync(fh.fileno())
    os.replace(tmp, path)
    return path


def read_p_sized(d: str) -> List[dict]:
    """Every complete P-sized record of ``d`` (a partial name never matches)."""
    out: List[dict] = []
    try:
        names = sorted(os.listdir(d))
    except FileNotFoundError:
        return out
    for name in names:
        if not (name.startswith("p_sized.") and name.endswith(".json")):
            continue
        doc = read_gate(os.path.join(d, name))
        if doc is not None:
            out.append(doc)
    return out


def stage0_verdict(p_records: List[dict], expected_p_ranks: int,
                   d_vram_max_mib: Mapping[str, int],
                   names: Optional[Mapping[str, str]] = None) -> Tuple[Optional[bool], List[str]]:
    """``(None, lines)`` while P is not fully sized; ``(True, lines)`` = go;
    ``(False, lines)`` = refuse because a D process held VRAM on a card before
    P was sized (P's used_by_me then includes it: the sizing is tainted)."""
    held = {u: int(v) for u, v in d_vram_max_mib.items() if int(v) > 0}
    lines = [f"P rank pp{r.get('pp_rank')}tp{r.get('tp_rank')} sized used_by_me="
             f"{r.get('used_by_me_mib')} MiB" for r in p_records]
    if held:
        lines += [f"{(names or {}).get(u, u)}: D held {v} MiB before P was sized" for u, v in sorted(held.items())]
        return False, lines
    if len(p_records) < int(expected_p_ranks):
        return None, lines
    lines.append(f"all {len(p_records)} P ranks sized, D held 0 MiB on every card meanwhile")
    return True, lines


def budget_verdict(planned: Mapping[str, int], measured: Mapping[str, int],
                   names: Optional[Mapping[str, str]] = None) -> Tuple[bool, List[str]]:
    """go iff every card's planned D budget fits its measured one. A card the
    measurement does not name refuses (nothing measured = nothing proven)."""
    ok = True
    lines = []
    for uuid in sorted(planned):
        p = int(planned[uuid])
        label = (names or {}).get(uuid, uuid)
        if uuid not in measured:
            ok = False
            lines.append(f"{label}: planned {p} MiB, NOT MEASURED")
            continue
        m = int(measured[uuid])
        fits = p <= m
        ok = ok and fits
        lines.append(f"{label}: planned {p} <= measured {m} MiB (slack {m - p:+d})"
                     if fits else f"{label}: planned {p} > measured {m} MiB (short {p - m})")
    return ok, lines


def reset_for_tests() -> None:
    _PASSED["path"] = None
    _PASSED["waited_s"] = None
    _STAGE0_PASSED["path"] = None
