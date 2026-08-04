# Copyright 2025 SGLang Team
# Licensed under the Apache License, Version 2.0
"""Serving metrics for the translator tenant (`--enable-metrics`).

Every server boot in this repo carries `--enable-metrics` (standing order):
a server without `/metrics` is flying blind. The translator tenant was the
one boot that had no such flag at all, which is why the first real backlog it
produced could only be described, not located.

WHAT THIS IS FOR, concretely. A turn that takes eighty seconds can be stuck in
exactly three places, and the per-turn `Stopwatch` alone cannot tell them
apart because it only ever describes turns that FINISHED:

* the MT hop -- `mt_first_token` / `mt_total` grow;
* the synthesizer -- `tts_wait` grows while `tts_first_audio` does not, i.e.
  the turn is queued behind another synthesis rather than being slow;
* delivery -- the journal grows but the pump does not drain it, which no
  stage timing can see because it happens after the turn is done.

So the stage timings are aggregated AND three live depths are exposed. The
depths are read at scrape time from the objects that own them rather than
mirrored into counters, because a mirrored depth is wrong exactly when it
matters -- when the thing that should have updated it is the thing that is
stuck.

No new dependency: Prometheus' text format is a few lines of string building,
and pulling in a client library for that would be a worse trade than writing
it. Disabled by default, and when disabled every hook here is a cheap
boolean test on a module-level flag.
"""

from __future__ import annotations

import threading
from typing import Callable, Dict, List, Optional, Tuple

__all__ = ["enable", "enabled", "record_turn", "register_depths", "render"]

#: The stage keys carried by ``Stopwatch.to_json``. Named explicitly rather
#: than discovered from the payload so that a renamed stage shows up as a
#: missing series -- a metric that silently follows a rename is a metric that
#: cannot be alerted on.
_STAGES: Tuple[str, ...] = (
    "asr_ms",
    "embed_ms",
    "mt_first_token_ms",
    "mt_total_ms",
    "tts_first_audio_ms",
    "tts_wait_ms",
    "tts_total_ms",
    "first_audio_ms",
    "total_ms",
)

_lock = threading.Lock()
_enabled = False
_turns = 0
_sums: Dict[str, float] = {k: 0.0 for k in _STAGES}
_maxes: Dict[str, float] = {k: 0.0 for k in _STAGES}
#: Scrape-time callbacks returning live depths. Registered by the server so
#: this module needs no import of session or server (and no import cycle).
_depth_sources: List[Tuple[str, str, Callable[[], float]]] = []


def enable() -> None:
    global _enabled
    _enabled = True


def enabled() -> bool:
    return _enabled


def record_turn(timings: Optional[Dict[str, float]]) -> None:
    """Fold one completed turn's stage timings into the aggregates."""
    if not _enabled or not timings:
        return
    global _turns
    with _lock:
        _turns += 1
        for key in _STAGES:
            value = float(timings.get(key, 0.0) or 0.0)
            _sums[key] += value
            if value > _maxes[key]:
                _maxes[key] = value


def register_depths(
    name: str, help_text: str, source: Callable[[], float]
) -> None:
    """Expose a live depth, read at scrape time from whoever owns it."""
    with _lock:
        _depth_sources.append((name, help_text, source))


def reset_for_test() -> None:
    """Test hook: the aggregates are process-global by design."""
    global _turns, _enabled
    with _lock:
        _enabled = False
        _turns = 0
        for key in _STAGES:
            _sums[key] = 0.0
            _maxes[key] = 0.0
        _depth_sources.clear()


def render() -> str:
    """The Prometheus text exposition for everything recorded so far."""
    lines: List[str] = []
    with _lock:
        lines.append("# HELP translator_turns_total Completed translated turns.")
        lines.append("# TYPE translator_turns_total counter")
        lines.append(f"translator_turns_total {_turns}")
        for key in _STAGES:
            stage = key[:-3]
            base = f"translator_{stage}_seconds"
            lines.append(
                f"# HELP {base}_sum Cumulative {stage} time over all turns."
            )
            lines.append(f"# TYPE {base}_sum counter")
            lines.append(f"{base}_sum {_sums[key] / 1000.0:.6f}")
            lines.append(f"# HELP {base}_max Worst single turn's {stage} time.")
            lines.append(f"# TYPE {base}_max gauge")
            lines.append(f"{base}_max {_maxes[key] / 1000.0:.6f}")
        sources = list(_depth_sources)
    for name, help_text, source in sources:
        try:
            value = float(source())
        except Exception:  # noqa: BLE001 - a broken gauge must not kill a scrape
            continue
        lines.append(f"# HELP {name} {help_text}")
        lines.append(f"# TYPE {name} gauge")
        lines.append(f"{name} {value:.6f}")
    return "\n".join(lines) + "\n"
