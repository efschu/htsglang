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

Default off (``--weg2-d-early-start``); without the env var nothing here runs.
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

_PASSED: Dict[str, object] = {"path": None, "waited_s": None}


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
