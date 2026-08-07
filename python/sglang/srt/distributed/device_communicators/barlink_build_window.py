# SPDX-License-Identifier: Apache-2.0
"""The build window, made visible to the PEERS of the rank that is building.

THE DEFECT (#615)
-----------------
``utils/jit_cold_build.cold_build_window`` already exists and already does the
right thing -- for the rank that opens it. It is RANK-LOCAL state (a module
depth counter, ``jit_cold_build.py:107``), so every deadline it relaxes is a
deadline evaluated in the builder's own process:

* ``barlink_liveness.wait_timeout_s`` (``barlink_liveness.py:168``) multiplies
  the host wait budget while ``in_cold_build_window()`` -- in THIS process;
* ``jit_cold_build.resolve_timeout_cycles`` multiplies the kernel cap cycles
  the caller is about to launch with -- in THIS process.

The rank that is stuck in nvcc is not the rank whose deadline matters. Its
PEERS are the ones sitting in a collective, and they are not in a cold-build
window, so they run the steady-state deadline and time each other out. The
only case this tree covers today is the one where the build is COORDINATED and
every rank opens the window together (``barlink_bar1_ext.py:162``, the grouped
BAR1 JIT build). An UNCOORDINATED lazy build -- gptq_marlin compiling on first
call inside a warmup forward, a flashinfer JIT on whichever rank reaches the
shape first -- is invisible to everyone but the builder.

#616f made that gap sharp. The BAR1 abort guard now raises
``Bar1CollectiveStalled`` after ``SGLANG_BARLINK_BAR1_STALL_RAISE_AFTER``
consecutive expiries of ``..._ABORT_SYNC_DEADLINE_MS`` (30 x 2000 ms = ~60 s
by default). A legitimate three-minute nvcc build on one rank now looks
exactly like a wedged collective, and the guard aborts the group for it.
Before the guard, the same situation was a multi-minute silent stall. Neither
outcome is a diagnosis.

THE SHAPE OF THE FIX
--------------------
One fact, published by the builder BEFORE it blocks, readable by the peers
WITHOUT its cooperation, consulted only by a wait that is already about to
give up.

* ``publish_building`` writes one small marker file named ``<host>.<pid>``.
  The builder writes it and then disappears into the compiler; nothing else
  is asked of it, which is the property that matters -- a rank inside
  ``nvcc`` cannot answer a probe, respond to a collective, or update a
  heartbeat.
* Peers read it with the pair they ALREADY have.
  ``barlink_liveness.PeerTable`` exchanges ``(rank, host, pid, boot)`` at
  transport bring-up (``barlink_liveness.py:247``), so a peer's marker path is
  derivable locally, from the table, with no new exchange and no new
  collective. This is the same cooperation-free shape as ``pid_alive``:
  ``PeerTable.state`` decides a peer's liveness with ``os.kill(pid, 0)`` and
  ``/proc/<pid>/stat``, and this decides a peer's BUILDING state with one
  ``stat`` of a path derived from the same tuple.
* A marker whose pid is dead is ignored. A dead builder is a dead peer, and
  that is ``PeerTable.dead_peers``' answer to give, not this module's -- a
  leaked marker must never be able to extend a deadline forever.

WHAT THE WAITERS DO WITH IT
---------------------------
They EXTEND, in increments, under an absolute cap:

* ``barlink_liveness.bounded_poll`` at its deadline, before raising
  ``CollectiveTimeoutError``;
* ``BarlinkBar1Transport._wait_ctl_event`` at its stall count, before raising
  ``Bar1CollectiveStalled``.

Both extend by their own increment and both stop at ``build_cap_s()``
(``SGLANG_BARLINK_BUILD_WINDOW_CAP_S``, default 900 s). The cap is what keeps
this a diagnosis rather than a mute button: a rank that publishes "building"
and then genuinely wedges is still caught, 15 minutes later, with the marker
named in the message. Every extension logs, at WARNING, naming the peer and
the reason it published -- so a boot that spends 12 minutes in nvcc says so in
the log instead of looking like a hang that got lucky.

DEFAULT PATH
------------
Nothing here runs unless a build window is opened or a deadline is about to
fire. ``peers_building`` is called from the two failure paths only; on a
healthy collective it is never reached. With no window ever opened the marker
directory is never created and ``peers_building`` finds nothing. The
mechanism is file writes plus local arithmetic: no collective, no CUDA, no
change to any hot path.

SCOPE
-----
Single node. The marker directory is local, so a CROSS-HOST peer's build is
not visible here -- deliberately, and consistent with ``PeerTable.state``,
which already reports a cross-host peer as UNKNOWN rather than guessing. A
multi-node build window would need the marker on shared storage or in the
rendezvous store; nothing in this module presumes it cannot be added, and
nothing in it pretends it already is.
"""

from __future__ import annotations

import logging
import os
import tempfile
import threading
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator, List, Optional, Sequence, Tuple

logger = logging.getLogger(__name__)

# --- environment ----------------------------------------------------------

#: Kill switch for the whole mechanism. "0" restores the pre-#615 behaviour
#: exactly: nothing is published, nothing is read, and both waiters raise at
#: their original deadline. It exists so the change can be bisected around,
#: not because running without it is a reasonable operating point on a rig
#: whose kernel cache is ever cold.
ENV_ENABLE = "SGLANG_BARLINK_BUILD_WINDOW"

#: Where the markers live. Default ``/dev/shm/sglang-barlink-build`` -- tmpfs,
#: so a write costs no disk I/O and a reboot cleans it; the same directory
#: family ``utils/stale_shm_cleanup.py`` already sweeps. Falls back to the
#: system temp directory where ``/dev/shm`` is not a directory.
ENV_DIR = "SGLANG_BARLINK_BUILD_WINDOW_DIR"

#: The ABSOLUTE cap on how long a published build may extend a peer's
#: deadline, in seconds. This is the number that keeps the extension a
#: diagnosis: past it, a rank that published "building" and then wedged is
#: raised on anyway, with the marker named.
#:
#: Default 900 s (15 minutes). Two measurements bound the choice from below.
#: The BAR1 grouped-build docstring records a many-minute build in the Docker
#: image, whose ``TORCH_CUDA_ARCH_LIST`` carries seven architectures
#: (``barlink_bar1_ext.py:128``); the r3 cold-cache finding that produced
#: ``jit_cold_build`` records a multi-minute nvcc build of gptq_marlin
#: (``test_jit_cold_build_window.py`` header). From above it is bounded by
#: what an operator will sit through before concluding the boot is dead, and
#: by the existing device-side relaxation it has to be commensurate with:
#: ``_DEFAULT_MULT = 40`` x 60e9 cycles is ~15 min at 2.6 GHz
#: (``jit_cold_build.py:99``). 900 s is that same number, which is deliberate
#: -- a host-side cap shorter than the device-side one would make the device
#: relaxation unreachable, and a longer one would let the host outlive the
#: kernels it is waiting for.
ENV_CAP_S = "SGLANG_BARLINK_BUILD_WINDOW_CAP_S"

_DEFAULT_CAP_S = 900.0
_DEFAULT_DIR = "/dev/shm/sglang-barlink-build"

#: How long a ``peers_building`` answer is reused before the directory is
#: stat'ed again. The callers are spin loops; without this a wait that is
#: extending would ``stat`` once per iteration for minutes. 0.25 s is two
#: orders of magnitude below the shortest increment either caller can add
#: (the BAR1 waiter's ~60 s stall run, the host waiter's 120 s budget), so
#: the cache cannot change any extension decision -- it only removes syscalls.
_SCAN_CACHE_S = 0.25

_FALSE = ("0", "false", "no", "off", "")


def _env_flag(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() not in _FALSE


def build_window_enabled() -> bool:
    """Read per call, not at import: tests and operators change it in place."""
    return _env_flag(ENV_ENABLE, True)


def build_cap_s() -> float:
    """The absolute ceiling on build-driven deadline extension, in seconds.

    ``0`` disables extension entirely while leaving publication intact, which
    is the setting that isolates "the marker is being written" from "the
    marker is being honoured" when a wedge is under investigation.
    """
    raw = os.environ.get(ENV_CAP_S)
    if raw is None:
        return _DEFAULT_CAP_S
    try:
        return max(float(raw), 0.0)
    except ValueError:
        logger.warning(
            "%s=%r is not a number; using %g.", ENV_CAP_S, raw, _DEFAULT_CAP_S
        )
        return _DEFAULT_CAP_S


def marker_dir() -> Path:
    """The directory the markers live in. Never raises."""
    raw = os.environ.get(ENV_DIR)
    if raw:
        return Path(raw)
    shm = Path("/dev/shm")
    if shm.is_dir():
        return Path(_DEFAULT_DIR)
    return Path(tempfile.gettempdir()) / "sglang-barlink-build"


def marker_path(host: str, pid: int) -> Path:
    """The marker path for ONE rank, derived from the tuple peers already have.

    ``(host, pid)`` is exactly what ``barlink_liveness.PeerIdentity`` carries,
    so a waiter computes this for each peer with no exchange of any kind. The
    host is part of the NAME rather than only of the check so that a marker
    directory shared by accident (a bind mount, a shared ``/tmp``) cannot make
    two hosts' identically numbered pids collide.
    """
    return marker_dir() / f"{host}.{int(pid)}"


# --- publication ----------------------------------------------------------

_lock = threading.Lock()
_depth = 0
_own_path: Optional[Path] = None


def _hostname() -> str:
    import socket

    return socket.gethostname()


def publish_building(reason: str) -> Optional[Path]:
    """Announce that THIS process is about to block in a build.

    Returns the marker path, or ``None`` if nothing was published (disabled,
    or the write failed). Never raises: a rank must not fail to build because
    it could not tell anybody it was building.

    Re-entrant. Nested windows publish once and the marker survives until the
    outermost one closes, which mirrors ``jit_cold_build.cold_build_window``'s
    depth counter -- the two are opened together and must agree about when the
    build is over.
    """
    global _depth, _own_path
    if not build_window_enabled():
        return None
    with _lock:
        _depth += 1
        if _depth > 1:
            return _own_path
        path = marker_path(_hostname(), os.getpid())
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            # Written whole, then renamed: a peer that reads the marker while
            # it is being written must never see a half line and conclude the
            # pid is 0.
            tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
            tmp.write_text(
                f"{_hostname()}\n{os.getpid()}\n{time.time()}\n{reason}\n"
            )
            os.replace(tmp, path)
        except Exception as e:  # noqa: BLE001 - a notice must not break a build
            logger.warning(
                "barlink build window: could not publish %s (%s); peers will "
                "run their steady-state deadlines through this build.",
                path,
                e,
            )
            _depth -= 1
            return None
        _own_path = path
    logger.info(
        "barlink build window OPEN (%s): published %s. Peers that reach a "
        "collective deadline while this exists extend it, up to %s=%gs.",
        reason,
        path,
        ENV_CAP_S,
        build_cap_s(),
    )
    return path


def clear_building() -> None:
    """Withdraw this process's marker. Idempotent, never raises.

    Called from the ``finally`` of ``barlink_build_window``, so it runs on the
    exception path too. A build that dies still clears -- and a build whose
    PROCESS dies leaves the marker behind, which is why every reader checks
    the pid.
    """
    global _depth, _own_path
    with _lock:
        if _depth <= 0:
            return
        _depth -= 1
        if _depth > 0:
            return
        path, _own_path = _own_path, None
    if path is None:
        return
    try:
        path.unlink()
    except FileNotFoundError:
        pass
    except Exception as e:  # noqa: BLE001
        logger.warning("barlink build window: could not remove %s (%s).", path, e)
    logger.info("barlink build window CLOSE: withdrew %s.", path)


def publishing() -> bool:
    """Whether THIS process currently has a marker published."""
    with _lock:
        return _depth > 0


@contextmanager
def barlink_build_window(reason: str) -> Iterator[None]:
    """Publish "building" for the duration of the block.

    Exception-safe and re-entrant. This is the whole publisher API; the
    production sites reach it through ``jit_cold_build.cold_build_window``,
    which opens both windows together so that a build site never relaxes one
    deadline family and not the other.
    """
    publish_building(reason)
    try:
        yield
    finally:
        clear_building()


# --- observation ----------------------------------------------------------


class PeerBuild:
    """One peer that is currently inside a build, as this process sees it."""

    __slots__ = ("rank", "host", "pid", "reason", "since")

    def __init__(self, rank: int, host: str, pid: int, reason: str, since: float):
        self.rank = rank
        self.host = host
        self.pid = pid
        self.reason = reason
        self.since = since

    def describe(self) -> str:
        age = max(time.time() - self.since, 0.0) if self.since else 0.0
        return (
            f"rank {self.rank} ({self.host}, pid {self.pid}) building for "
            f"{age:.0f}s: {self.reason or 'unnamed build'}"
        )

    def __repr__(self) -> str:  # pragma: no cover - diagnostics
        return f"<PeerBuild {self.describe()}>"


def _read_marker(path: Path) -> Optional[Tuple[str, int, float, str]]:
    """``(host, pid, started, reason)`` from a marker, or ``None``."""
    try:
        raw = path.read_text()
    except FileNotFoundError:
        return None
    except Exception:  # noqa: BLE001 - an unreadable marker is no marker
        return None
    lines = raw.split("\n")
    if len(lines) < 3:
        return None
    host = lines[0].strip()
    try:
        pid = int(lines[1].strip())
    except ValueError:
        return None
    try:
        started = float(lines[2].strip())
    except ValueError:
        started = 0.0
    reason = lines[3].strip() if len(lines) > 3 else ""
    return host, pid, started, reason


def _entries_of(table: Any) -> Sequence[Any]:
    return tuple(getattr(table, "entries", ()) or ())


def _scan(tables: Sequence[Any]) -> List[PeerBuild]:
    from sglang.srt.distributed.device_communicators.barlink_liveness import (
        pid_alive,
    )

    host = _hostname()
    mine = os.getpid()
    found: List[PeerBuild] = []
    seen = set()
    for table in tables:
        self_rank = getattr(table, "self_rank", None)
        for entry in _entries_of(table):
            rank = getattr(entry, "rank", -1)
            peer_host = getattr(entry, "host", "") or ""
            peer_pid = int(getattr(entry, "pid", 0) or 0)
            if rank == self_rank or peer_pid == mine:
                continue
            if not peer_host or peer_pid <= 0:
                continue
            if peer_host != host:
                # Cross-host: the marker directory is local, so absence here
                # is not evidence. Same rule as PeerTable.state's UNKNOWN.
                continue
            key = (peer_host, peer_pid)
            if key in seen:
                continue
            seen.add(key)
            info = _read_marker(marker_path(peer_host, peer_pid))
            if info is None:
                continue
            m_host, m_pid, started, reason = info
            if m_host != peer_host or m_pid != peer_pid:
                # A stale file under a recycled name. Not this peer's.
                continue
            if not pid_alive(m_pid):
                # A leaked marker from a process that is gone. The peer table
                # calls that DEAD and ends the wait; it must never be able to
                # extend one.
                continue
            found.append(PeerBuild(rank, m_host, m_pid, reason, started))
    return found


_scan_lock = threading.Lock()
_scan_at = 0.0
_scan_result: List[PeerBuild] = []


def peers_building(table: Any = None, *, cached: bool = True) -> List[PeerBuild]:
    """Which PEERS are currently inside a published build window.

    Reached only from a wait that is already at its deadline, never from a
    collective that completes. Costs one ``stat``/``read`` per same-host peer,
    rate-limited to ``_SCAN_CACHE_S``.

    ``table`` restricts the scan to one ``PeerTable``; the default consults
    every registered one, which is the same fallback ``_dead_for`` uses.
    """
    global _scan_at, _scan_result
    if not build_window_enabled():
        return []
    if table is not None:
        tables: Sequence[Any] = (table,)
        cached = False
    else:
        from sglang.srt.distributed.device_communicators.barlink_liveness import (
            registered_tables,
        )

        tables = registered_tables()
    if not tables:
        return []
    if cached:
        now = time.monotonic()
        with _scan_lock:
            if now - _scan_at < _SCAN_CACHE_S:
                return list(_scan_result)
        result = _scan(tables)
        with _scan_lock:
            _scan_at = time.monotonic()
            _scan_result = list(result)
        return result
    return _scan(tables)


def describe_peers_building(builds: Sequence[PeerBuild]) -> str:
    return "; ".join(b.describe() for b in builds)


def reset_for_test() -> None:
    """Drop all process-global state, including the scan cache. Tests only."""
    global _depth, _own_path, _scan_at, _scan_result
    with _lock:
        _depth = 0
        _own_path = None
    with _scan_lock:
        _scan_at = 0.0
        _scan_result = []


# --- the extension decision ------------------------------------------------


def extension_for(
    waited_s: float,
    increment_s: float,
    label: str,
    *,
    table: Any = None,
) -> Optional[List[PeerBuild]]:
    """Whether a wait that just hit its deadline may extend, and by how much.

    Returns the list of building peers when the wait may extend by
    ``increment_s``, and ``None`` when it must raise. Both callers share this
    so there is ONE definition of the decision -- a second copy in the second
    caller would keep passing across a revert of the first.

    ``waited_s`` is the total time this wait has already spent, measured from
    when it started, INCLUDING every previous extension. That is what the cap
    is applied to, which is what makes the bound absolute rather than
    per-extension: a peer that republishes its marker cannot buy more time
    than ``build_cap_s()`` from any single wait.
    """
    if not build_window_enabled():
        return None
    cap = build_cap_s()
    if cap <= 0.0:
        return None
    if waited_s >= cap:
        return None
    builds = peers_building(table)
    if not builds:
        return None
    logger.warning(
        "barlink build window: '%s' reached its deadline after %.1fs, but a "
        "peer is legitimately building -- %s. Extending by %.1fs. The "
        "extension is capped at %s=%gs; past that this raises anyway, so a "
        "rank that published a build and then wedged is still caught.",
        label,
        waited_s,
        describe_peers_building(builds),
        increment_s,
        ENV_CAP_S,
        cap,
    )
    return builds


__all__ = [
    "ENV_CAP_S",
    "ENV_DIR",
    "ENV_ENABLE",
    "PeerBuild",
    "barlink_build_window",
    "build_cap_s",
    "build_window_enabled",
    "clear_building",
    "describe_peers_building",
    "extension_for",
    "marker_dir",
    "marker_path",
    "peers_building",
    "publish_building",
    "publishing",
    "reset_for_test",
]
