# SPDX-License-Identifier: Apache-2.0
"""Replay-boundary check for the capturable MoE expert-offload fetch.

WHAT IT GUARDS
--------------
The capturable offload path (``expert_offload.prepare_capturable``) resolves a
routed expert to a row of THIS rank's pinned cold pool with pure on-device
index math, and gathers that row with a captured ``index_select``. Under the
#394 shared cold tier a routed expert can be DELEGATED: its bytes live in a
peer's segment and this rank's pool has no row for it, so the frozen
``spill_pool_row_lut`` holds ``-1`` for it. ``-1`` is not a diagnosis; it is an
out-of-bounds index.

The eager path never reaches that state, because ``_fetch`` branches on
``expert_id in self._remote_ids`` and copies from the peer view instead. That
branch is a host read of a Python set and cannot exist inside a capture, which
is exactly why the capturable installer refuses a cold tier by default.

``SGLANG_MOE_COLD_TIER_GRAPH_UNSAFE=1`` is the development window's switch past
that refusal. With this gate in place it is no longer a switch into undefined
behaviour: the capturable remap clamps the index to a valid row and increments
a DEVICE-RESIDENT breach counter instead, and the counter is read here -- on
the host, after the graph has finished replaying -- and turned into a named
exception. A window that trips it learns which layer and how many routed slots,
rather than reading a plausible wrong expert's weights.

WHY A SEPARATE MODULE
---------------------
Same reason as ``barlink_abort_gate`` (#431), whose shape this follows: the
check fires at the CUDA-graph replay boundary in ``model_executor``, which must
not grow an import of ``expert_offload`` -- 3.6k lines pulling in the whole
staging, planner and NVML stack -- in a process that never installed an
offload. This module's own import cost is ``logging``, ``os`` and
``threading``. Its PARENT package is not free (``layers/moe/__init__`` builds
the runner config), but the graph backends already import it transitively, so
the marginal cost of the line added there is the gate module alone; verified
by importing ``full_cuda_graph_backend`` and reading ``sys.modules``:
``layers.moe`` present, ``layers.moe.expert_offload`` absent.

CAPTURE SAFETY
--------------
Reading the counter is a device read and therefore synchronizes; inside a
stream capture that is illegal, not merely slow. Nothing in this module is
called from inside a capture: the counter is incremented by a captured kernel
and read only here, at the boundary, which is the next host point after the
replay.

DEFAULT PATH
------------
With no cold-tier offload cache in the process the registry is empty and
``check_after_graph_replay`` returns after one truth test on a list. A launch
that does not set ``SGLANG_MOE_COLD_TIER_SHM`` never registers anything.
"""

from __future__ import annotations

import logging
import os
import threading
from typing import Any, List

logger = logging.getLogger(__name__)

#: Kill switch for the whole family. "0" restores the pre-port behaviour: the
#: breach counter is still incremented by the captured kernel, but nothing
#: reads it, so a delegated expert routed into a captured step silently reads
#: row 0 of the local pool. It exists so an operator can bisect around the
#: check, not because running without it is a reasonable operating point.
ENV_ENABLE = "SGLANG_MOE_OFFLOAD_CAPTURE_GATE"

_FALSE = ("0", "false", "no", "off", "")

_lock = threading.Lock()
_caches: List[Any] = []


class OffloadCaptureBreach(RuntimeError):
    """A captured offload gather referenced a row this rank does not own.

    Carries the layer and the counts rather than a bare message: the operating
    question after one of these is always "which layer, and was it one slot or
    every slot", and a summary that drops that costs the reader the only two
    facts that distinguish a mis-built plan from a routing excursion.
    """

    def __init__(self, layer_id: Any, breaches: int, where: str):
        self.layer_id = layer_id
        self.breaches = int(breaches)
        self.where = where
        super().__init__(
            f"MoE offload capture gate ({where}): layer {layer_id} gathered "
            f"{self.breaches} scratch slot(s) whose expert has no row in this "
            f"rank's cold pool. Those experts are DELEGATED to a peer's #394 "
            f"segment, and the capturable gather has no source for them -- the "
            f"clamped index read row 0 of the local pool, so this step's MoE "
            f"output is wrong, not approximate. Run eager "
            f"(--disable-cuda-graph) or without SGLANG_MOE_COLD_TIER_SHM until "
            f"the capturable path sources peer rows."
        )


def _env_flag(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() not in _FALSE


def gate_enabled() -> bool:
    """Read per call, not at import: tests and operators change it in place."""
    return _env_flag(ENV_ENABLE, True)


def register(cache: Any) -> None:
    """Publish an offload cache that carries a device breach counter."""
    with _lock:
        if cache not in _caches:
            _caches.append(cache)


def unregister(cache: Any) -> None:
    with _lock:
        if cache in _caches:
            _caches.remove(cache)


def registered() -> List[Any]:
    with _lock:
        return list(_caches)


def reset_for_test() -> None:
    """Drop the registry. Tests only."""
    with _lock:
        _caches.clear()


def check_after_graph_replay(where: str = "cuda-graph replay") -> None:
    """Raise if any registered cache's captured gather clamped an index.

    The FIRST statement is the empty-registry test, and that is the whole
    default path: a process without a cold-tier offload pays one truth test on
    a list per replay.

    Raises :class:`OffloadCaptureBreach` for the first layer that reports.
    Deliberately not aggregated across layers -- the first one to breach is the
    one whose plan is wrong, and every later layer would report the same cause.
    """
    if not _caches:
        return
    if not gate_enabled():
        return
    for cache in registered():
        check = getattr(cache, "check_capture_breach", None)
        if check is not None:
            check(where)


__all__ = [
    "ENV_ENABLE",
    "OffloadCaptureBreach",
    "check_after_graph_replay",
    "gate_enabled",
    "register",
    "registered",
    "reset_for_test",
    "unregister",
]
