# SPDX-License-Identifier: Apache-2.0
"""Where a tripped BAR1 spin kernel becomes a raised exception (task #431).

THE DEFECT
----------
``barlink_bar1_ext`` kernels break out of their flag-equality wait when
``clock64() - t0 > A.capCycles``, write ``*A.ctlStatus = 1u`` and RETURN --
leaving the output buffer partially written. The word they write is read by
``BarlinkBar1Transport.status()``, and until this module existed nothing on a
production path ever called it: ``raise_if_aborted`` was reachable only from
three bring-up proofs. The #431 window measured what that costs. A run that
tripped the cap on essentially every collective for 22 minutes produced an
``abort_*.txt`` with ZERO matching lines
(``/spinning/gpu-battery-results/2026-08-02_431_repro/abort_fp8_bar1_decode.txt``):
nothing in the process could distinguish "no kernel tripped" from "every
kernel tripped", and the stream kept running over the partial buffers.

WHY A SEPARATE MODULE
---------------------
The check has to fire at two places that must not import each other:

* the barlink dispatch sites, which already hold the transport, and
* the CUDA-graph replay boundary in ``model_executor``, which holds nothing
  and must not grow an import of ``barlink_bar1`` (ctypes, mmap, dma-buf) in
  a process that never asked for barlink.

So the registry lives here, in a module whose entire import cost is
``threading`` + ``weakref``, and the graph backends import only this.

CAPTURE SAFETY
--------------
Reading ``ctlStatus`` is a device read and therefore synchronizes; inside a
stream capture it is not merely expensive but illegal. Every check in this
family is therefore skipped while ``graph_capture_running()`` is true, and
the transport records that the launch was CAPTURED instead. The kernels those
captures contain execute on replay, and the replay boundary -- host code, no
capture in progress -- is where they get checked.

DEFAULT PATH
------------
With no BAR1 transport in the process the registry is empty and
``check_aborts`` returns after one truth test on a list. Nothing else in this
module runs.
"""

from __future__ import annotations

import logging
import os
import threading
from typing import Any, List

logger = logging.getLogger(__name__)

#: Environment kill switch for the whole family. "0" restores the pre-#431
#: behaviour exactly: a tripped kernel stays silent and the stream continues
#: over the partially written buffer. It exists because a check that cannot
#: be switched off is a check nobody can bisect around, not because running
#: without it is a reasonable operating point.
ENV_ENABLE = "SGLANG_BARLINK_BAR1_ABORT_CHECK"

#: How many BAR1 collectives may run on the host path between two device
#: reads of the status word. 1 = check after every collective, which is the
#: default: the word is 4 bytes, but reading it synchronizes the current
#: stream, so an operator who measures a regression on an eager path has a
#: knob instead of an all-or-nothing switch. Values <= 0 read as 1.
ENV_EVERY = "SGLANG_BARLINK_BAR1_ABORT_CHECK_EVERY"

#: Whether the CUDA-graph replay boundary checks. Separate from ENV_ENABLE
#: because it is the one check that can cost a synchronization the host would
#: not otherwise have paid (an overlap-scheduled decode runs ahead of the
#: device on purpose). On by default -- a captured decode is exactly the case
#: where nothing else looks -- and documented so it can be turned off with a
#: reason rather than by disabling the whole family.
ENV_REPLAY = "SGLANG_BARLINK_BAR1_ABORT_CHECK_REPLAY"

_FALSE = ("0", "false", "no", "off", "")

_lock = threading.Lock()
_transports: List[Any] = []


def _env_flag(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() not in _FALSE


def abort_check_enabled() -> bool:
    """Read per call, not at import: tests and operators change it in place."""
    return _env_flag(ENV_ENABLE, True)


def replay_check_enabled() -> bool:
    return abort_check_enabled() and _env_flag(ENV_REPLAY, True)


def check_every() -> int:
    raw = os.environ.get(ENV_EVERY)
    if raw is None:
        return 1
    try:
        return max(int(raw), 1)
    except ValueError:
        logger.warning("%s=%r is not an integer; using 1.", ENV_EVERY, raw)
        return 1


def register(transport: Any) -> None:
    """Publish a live BAR1 transport. Called once per successful bring-up."""
    with _lock:
        if transport not in _transports:
            _transports.append(transport)


def unregister(transport: Any) -> None:
    """Withdraw a transport. Called from ``close()``, before the tensors go."""
    with _lock:
        if transport in _transports:
            _transports.remove(transport)


def registered() -> List[Any]:
    with _lock:
        return list(_transports)


def reset_for_test() -> None:
    """Drop the registry. Tests only."""
    with _lock:
        _transports.clear()


def check_aborts(where: str) -> None:
    """Raise if any live BAR1 transport's kernels took their abort path.

    The FIRST statement is the empty-registry test, and it is the whole
    default path: a process without a BAR1 transport pays one truth test on a
    list per call, which is why this is safe to put on a per-forward-step
    boundary.

    Raises the transport's own ``Bar1CollectiveAborted``. Deliberately not
    wrapped or aggregated: the first transport to report is the one whose
    buffers are corrupt, and a group-level summary would cost the reader the
    rank/op/rounds context that is the entire point.
    """
    if not _transports:
        return
    if not abort_check_enabled():
        return
    for transport in registered():
        check = getattr(transport, "check_aborted", None)
        if check is not None:
            check(where)


def check_after_graph_replay(where: str = "cuda-graph replay") -> None:
    """The replay-boundary check. See the module docstring for why it exists.

    A captured graph contains the BAR1 kernels but no host code, so the
    per-collective check cannot fire during a replay; this is the next host
    point after it. Separately switchable via ``ENV_REPLAY`` because it is
    the one place where the read can cost a synchronization that the host
    would not otherwise have paid.
    """
    if not _transports:
        return
    if not replay_check_enabled():
        return
    check_aborts(where)


__all__ = [
    "ENV_ENABLE",
    "ENV_EVERY",
    "ENV_REPLAY",
    "abort_check_enabled",
    "check_aborts",
    "check_after_graph_replay",
    "check_every",
    "register",
    "registered",
    "replay_check_enabled",
    "reset_for_test",
    "unregister",
]
