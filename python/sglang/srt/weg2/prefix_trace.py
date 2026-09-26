"""Prefix trace (IN 26.09.): every prefix miss of an agent-load boot gets a token
receipt, behind ONE switch -- ``SGLANG_WEG2_PREFIX_TRACE`` (default off).

Agent PM's reading of the 27B agent boots (26.09.) with
``devtools/prefix_miss_classify.py``: the instruments that name WHY a prefix
missed are sampled per process (``#1420 WALK-STOP`` 24 lines, ``#1439`` 24,
``#1469`` 600, ``#1400``/``#1416d``/``#1416e`` 8, ``#1442 HANDOFF-KEYS REG``
12), ``#1416`` cut the rid to 8 characters, ``#1427 ARENA-DROP`` named no key
and ``#1469 EVICT`` only a node id. After the caps a render break and the
midnight date change were indistinguishable -- indiz, never a receipt.

With the switch ON (and only then):

* ``#1420 WALK-STOP`` -- one line per (rid, stop depth) for every radix walk
  whose unmatched rest is >= ``SGLANG_WEG2_PREFIX_TRACE_MIN_TOKENS`` (default
  1024), no process cap; the line carries the full rid, the stop node, its
  last page hash and the first tokens on both sides of the divergence.
* ``#1400`` / ``#1416`` / ``#1416d`` / ``#1416e`` -- full rid, no 8/256
  sampling (one line per request and event).
* ``#1442 HANDOFF-KEYS REG`` and ``#1040 EXTENT STATE-ALIGN`` -- no sampling.
* ``#1469 EVICT`` -- no cap, plus the parent node id (the node a later walk
  stops at) and the node's page hashes; ``#1427 ARENA-DROP`` -- no cap, plus
  the dropped slot keys and the claiming page's stem.

What it never does: log in the decode round path. Every traced line sits on a
prefill admission / store / eviction event, not on a per-round step. The
switch is read ONCE per process; off costs one module-global bool test.
"""

from __future__ import annotations

import logging
import os
from typing import Dict, Optional, Tuple

logger = logging.getLogger(__name__)

ENV = "SGLANG_WEG2_PREFIX_TRACE"
ENV_MIN_TOKENS = "SGLANG_WEG2_PREFIX_TRACE_MIN_TOKENS"
DEFAULT_MIN_TOKENS = 1024

#: bound on the (rid, depth) de-duplication table of the walk trace -- an
#: agent boot has a few hundred requests; FIFO eviction past this keeps the
#: memory flat on a boot that runs for days.
WALK_SEEN_MAX = 8192

_state: Optional[Tuple[bool, int]] = None
_walk_seen: Dict[Tuple[str, int], None] = {}
_once_seen: Dict[tuple, None] = {}


def _read() -> Tuple[bool, int]:
    try:
        from sglang.srt.environ import envs

        on = bool(envs.SGLANG_WEG2_PREFIX_TRACE.get())
        mn = int(envs.SGLANG_WEG2_PREFIX_TRACE_MIN_TOKENS.get())
    except Exception:  # noqa: BLE001 - a partial environ (unit doubles) reads raw
        on = str(os.environ.get(ENV, "0")).strip().lower() in ("1", "true", "yes", "on")
        try:
            mn = int(os.environ.get(ENV_MIN_TOKENS, DEFAULT_MIN_TOKENS))
        except ValueError:
            mn = DEFAULT_MIN_TOKENS
    return on, max(0, mn)


def _load() -> Tuple[bool, int]:
    global _state
    if _state is None:
        _state = _read()
        # The one "armed" line of this process: written at the first question,
        # which every scheduler asks at its first radix walk -- once, never on
        # a round path. Names the value, the group and the pid.
        logger.info(
            "#PT PREFIX-TRACE armed=%d min_tokens=%d group=%s pid=%d (%s)",
            int(_state[0]), _state[1], os.environ.get("SGLANG_WEG2_GROUP", "-"),
            os.getpid(), ENV,
        )
    return _state


def on() -> bool:
    """Is the prefix trace armed in this process (read once)."""
    s = _state
    return (s if s is not None else _load())[0]


def min_tokens() -> int:
    s = _state
    return (s if s is not None else _load())[1]


def sampled(n: int, first: int, every: int) -> bool:
    """The legacy ``n <= first or n % every == 0`` sampling, lifted when traced."""
    return on() or n <= first or (every > 0 and n % every == 0)


def rid_text(rid, width: int = 8) -> str:
    """The rid as a log field: full when traced, the legacy prefix otherwise."""
    s = str(rid)
    return s if on() else s[:width]


def walk_due(rid, depth: int, remaining: int) -> bool:
    """One traced walk line per (rid, stop depth), rest >= the minimum."""
    if rid is None or not on() or int(remaining) < min_tokens():
        return False
    k = (str(rid), int(depth))
    if k in _walk_seen:
        return False
    _walk_seen[k] = None
    if len(_walk_seen) > WALK_SEEN_MAX:
        _walk_seen.pop(next(iter(_walk_seen)))
    return True


def once(*key) -> bool:
    """Traced AND the first time for ``key`` in this process (bounded FIFO):
    a per-request line on a path a request may pass more than once (a
    re-issued prefetch, a re-admission) stays one line per distinct fact."""
    if not on():
        return False
    if key in _once_seen:
        return False
    _once_seen[key] = None
    if len(_once_seen) > WALK_SEEN_MAX:
        _once_seen.pop(next(iter(_once_seen)))
    return True


def page_hash(node, which: int = -1, width: int = 16) -> str:
    """A node's page hash (chained: the last one names the whole prefix up to
    the node) -- an attribute read, set when the node was backed up; '-' when
    the node was never hashed."""
    hv = getattr(node, "hash_value", None)
    if not hv:
        return "-"
    try:
        return str(hv[which])[:width]
    except Exception:  # noqa: BLE001
        return "-"


def token_run(key, n: int = 8):
    """The first ``n`` raw tokens of a RadixKey (or a sequence), as a list."""
    try:
        t = getattr(key, "token_ids", key)
        return [int(x) for x in t[:n]]
    except Exception:  # noqa: BLE001
        return []


def _reset_for_tests() -> None:
    global _state
    _state = None
    _walk_seen.clear()
    _once_seen.clear()
