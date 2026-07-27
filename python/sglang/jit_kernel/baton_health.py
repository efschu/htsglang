"""Self-healing for torch's cpp_extension build lock (the "baton").

THE DEFECT
----------
``torch.utils.cpp_extension._jit_compile`` serialises concurrent builds of one
extension with a ``FileBaton`` -- a zero-byte ``lock`` file in the build
directory, created with ``O_CREAT | O_EXCL``. The process that creates it
builds; everyone else calls ``FileBaton.wait()``, which polls
``os.path.exists`` with no bound, no owner information and no evidence of any
kind.

A builder killed between finishing and ``baton.release()`` leaves that file
behind, and from that moment every later process waits forever for a build
that is already done. Observed on this host: a run stood 18 minutes at 0 % CPU
holding 682 MiB of VRAM and outlived its own ``timeout 900``, with

    torch/utils/file_baton.py:51           wait
    torch/utils/cpp_extension.py:2286      _jit_compile
    mem_cache/cpp_utils/native_hash.py:33  _load_native_hash_module

``hicache_hash_cpp/lock`` was zero bytes from the previous day at 16:18, while
``hicache_hash_cpp.so`` in the same directory had been finished at 16:15.
Removing the lock by hand let the waiter through instantly.

This is the third member of a family: the tvm-ffi cache (#172b) and the
torch_extensions cache (#181) both turned one interrupted build into a
permanent failure. Those two fixes look at the build RESIDUE and are in
``cache_health``; this one looks at the LOCK, which neither of them touches --
in the observed case the entry was ``complete`` by cache_health's rule and
still unusable.

WHERE THE FIX SITS
------------------
In ``wait()``, not before ``load()``. That placement is what makes the race
argument work, and it is worth being precise about why:

    A process only reaches ``wait()`` when ``try_acquire()`` FAILED, and after
    ``wait()`` returns ``_jit_compile`` does not build -- it falls straight
    through to ``_import_module_from_library``.

So releasing a waiter early can never produce two concurrent builders. The
worst case is importing an artifact that is not the one that was wanted, which
is why every early exit below requires an artifact to be present, and why the
default exit requires evidence that it is the FINISHED one. Deleting a lock
BEFORE calling ``load()`` would not have that property -- it can promote the
deleting process to builder next to a live one -- so this module does not do
that, and no startup sweep is wired in.

WHEN A LOCK IS TREATED AS ORPHANED
----------------------------------
Evaluated on every poll, first match wins:

1. ``released``  -- the lock is gone. The normal path.
2. ``reclaim``   -- the target artifact exists, is not older than the
   registered sources, and nothing in the build directory has changed for
   ``quiet_seconds``. This is the observed shape and it fires on the first
   poll, because a day-old directory is quiescent by definition. Requires
   sources registered through ``jit_build_guard`` / ``register_sources``: with
   no sources there is nothing to compare the artifact against.
3. ``reclaim``   -- the artifact exists and has not changed since this waiter
   started, and the waiter has been watching for ``quiet_seconds``. Covers
   ``load_inline``, whose callers -- waiters included -- rewrite ``main.cpp``
   into the build directory before ``_jit_compile`` runs, so rule 2's absolute
   quiescence can never hold there.
4. ``reclaim`` / ``fail`` -- the lock is older than ``max_wait_seconds``.
   With an artifact the waiter takes over; without one there is nothing to
   import and it raises a NAMED error rather than waiting on. This is the
   backstop for build directories nobody registered.

A live ``cache_health`` build marker (host + live pid + fresh) vetoes rules 2
and 3: that marker exists precisely to say "a co-located rank is compiling
here". Rule 4 still applies, since the marker's own staleness bound is longer
than the wait limit.

Reclaiming is a rename, then a delete. ``os.rename`` is atomic, so exactly one
of N concurrent waiters observes the lock and removes it; the others get
ENOENT and take rule 1 on their next poll. ``release()`` is patched alongside
``wait()`` to tolerate an already-removed lock, so reclaiming can never turn
into a crash for a holder that was still alive.

The wait is never silent: a named warning is logged periodically for as long
as it lasts.

KNOBS
-----
``SGLANG_JIT_BATON_SELFHEAL``            1 (set to 0 to restore stock torch)
``SGLANG_JIT_BATON_QUIET_SECONDS``       120
``SGLANG_JIT_BATON_MAX_WAIT_SECONDS``    1800
"""

from __future__ import annotations

import errno
import logging
import os
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Dict, Iterable, List, NamedTuple, Optional, Sequence, Union

logger = logging.getLogger(__name__)

PathLike = Union[str, "os.PathLike[str]"]

#: Grep handle. Every message this module emits about an orphaned or expired
#: lock carries it, so the failure has a name in logs instead of being an
#: anonymous hang.
BATON_ORPHAN_MARKER = "SGLANG_JIT_BATON_ORPHANED"

#: A finished cpp_extension build. ``.pyd`` is torch's Windows name for it.
ARTIFACT_SUFFIXES = (".so", ".pyd")

#: Sources torch copies into the build directory itself (``load_inline``).
#: Listed so they are not mistaken for build products.
_SOURCE_SUFFIXES = (".cpp", ".cc", ".cxx", ".c", ".cu", ".hip", ".sycl")

#: How long a build directory must be motionless before "nobody is building
#: here" is a fair reading. Two orders of magnitude above the sub-second window
#: in which a warm-cache process holds the lock for a no-op ninja run, and
#: comfortably above the gap between two ninja edges.
DEFAULT_QUIET_SECONDS = 120.0

#: Hard bound on any wait. Chosen an order of magnitude above the slowest build
#: that goes through this path on this tree (the HTCCL device extension, ~150 s
#: of nvcc), so a genuine build is never interrupted, while an orphaned lock
#: costs minutes instead of the rest of the run.
DEFAULT_MAX_WAIT_SECONDS = 1800.0

#: Interval for the "still waiting" warning.
WARN_EVERY_SECONDS = 60.0

_POLL_SECONDS = 0.5

#: build directory -> source paths, filled in by ``jit_build_guard``. Sources
#: are what rule 2 compares the artifact against; without them a build
#: directory only gets the rule 4 backstop.
_SOURCES: Dict[str, List[str]] = {}


class Verdict(NamedTuple):
    action: str  # "released" | "wait" | "reclaim" | "fail"
    reason: str


def _env_flag(name: str, default: bool = True) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() not in ("0", "false", "no", "off")


def _env_float(name: str, default: float) -> float:
    raw = os.environ.get(name)
    if raw is None:
        return default
    try:
        value = float(raw)
    except ValueError:
        logger.warning("Ignoring %s=%r: not a number.", name, raw)
        return default
    return value if value > 0 else default


def selfheal_enabled() -> bool:
    # Read per call, not at import: the switch exists so an operator can turn
    # the healing off in place if it ever misjudges a lock.
    return _env_flag("SGLANG_JIT_BATON_SELFHEAL", True)


def register_sources(build_dir: PathLike, sources: Iterable[PathLike]) -> None:
    """Record which files ``build_dir`` is built from.

    Only meaningful for ``load()``, whose sources live outside the build
    directory. ``load_inline`` writes its sources INTO the directory on every
    call, waiters included, so their mtimes say nothing about who is building.
    """
    paths = [str(s) for s in sources]
    if paths:
        _SOURCES[str(Path(build_dir))] = paths


def _mtime(path: Path) -> Optional[float]:
    try:
        return path.stat().st_mtime
    except OSError:
        return None


def _newest_artifact(build_dir: Path) -> Optional[Path]:
    newest: Optional[Path] = None
    newest_mtime = -1.0
    try:
        children = list(build_dir.iterdir())
    except OSError:
        return None
    for child in children:
        if child.suffix not in ARTIFACT_SUFFIXES:
            continue
        m = _mtime(child)
        if m is not None and m > newest_mtime:
            newest, newest_mtime = child, m
    return newest


def _newest_mtime(build_dir: Path) -> float:
    """Newest mtime anywhere in the build directory, the lock included."""
    newest = _mtime(build_dir) or 0.0
    try:
        children = list(build_dir.iterdir())
    except OSError:
        return newest
    for child in children:
        m = _mtime(child)
        if m is not None and m > newest:
            newest = m
    return newest


def _newest_source_mtime(build_dir: Path) -> Optional[float]:
    paths = _SOURCES.get(str(build_dir))
    if not paths:
        return None
    newest: Optional[float] = None
    for p in paths:
        m = _mtime(Path(p))
        if m is None:
            # A source we cannot stat is not evidence of anything; refuse to
            # judge rather than guess in the permissive direction.
            return None
        if newest is None or m > newest:
            newest = m
    return newest


def _live_build_marker(build_dir: Path) -> bool:
    """Is a cache_health build marker here claiming a live, recent builder?

    Reused rather than reinvented: that marker is written by exactly the
    co-located ranks whose in-flight build must not be declared abandoned.
    """
    try:
        from sglang.jit_kernel import cache_health

        marker = build_dir / cache_health.MARKER_BUILDING
        if not marker.is_file():
            return False
        age = time.time() - (_mtime(marker) or 0.0)
        if age > cache_health.DEFAULT_STALE_SECONDS:
            return False
        info = cache_health._read_marker(marker)
        if info is None:
            return False
        host, pid = info
        # Another host on a shared cache volume: its pid means nothing here,
        # so freshness has to carry it -- bounded by the staleness check above.
        return host != cache_health._hostname() or cache_health._pid_alive(pid)
    except Exception:
        return False


def baton_verdict(
    lock_path: PathLike,
    *,
    quiet_seconds: Optional[float] = None,
    max_wait_seconds: Optional[float] = None,
    unchanged_since: Optional[float] = None,
    watching_since: Optional[float] = None,
    now: Optional[float] = None,
) -> Verdict:
    """Classify one build lock. See the module docstring for the ladder.

    ``unchanged_since`` / ``watching_since`` carry the caller's observation of
    the directory: the newest mtime seen when the wait began, and when that
    was. Rule 3 needs both; the other rules do not.
    """
    lock = Path(lock_path)
    build_dir = lock.parent
    now = time.time() if now is None else now
    quiet = DEFAULT_QUIET_SECONDS if quiet_seconds is None else quiet_seconds
    limit = DEFAULT_MAX_WAIT_SECONDS if max_wait_seconds is None else max_wait_seconds

    lock_mtime = _mtime(lock)
    if lock_mtime is None:
        return Verdict("released", "the lock is gone")

    artifact = _newest_artifact(build_dir)
    lock_age = now - lock_mtime
    marker_live = _live_build_marker(build_dir)

    if artifact is not None and not marker_live:
        artifact_mtime = _mtime(artifact) or 0.0
        source_mtime = _newest_source_mtime(build_dir)

        # Rule 2: the build is finished and the directory has gone cold.
        if source_mtime is not None and artifact_mtime >= source_mtime:
            still_for = now - _newest_mtime(build_dir)
            if still_for >= quiet:
                return Verdict(
                    "reclaim",
                    f"{artifact.name} is newer than its sources and nothing in "
                    f"{build_dir} has changed for {still_for:.0f}s, so the "
                    f"build that created this lock is over",
                )

        # Rule 3: no source comparison available, but nothing has moved since
        # this waiter started watching.
        if unchanged_since is not None and watching_since is not None:
            watched_for = now - watching_since
            if watched_for >= quiet and _newest_mtime(build_dir) <= unchanged_since:
                return Verdict(
                    "reclaim",
                    f"{artifact.name} is present and nothing in {build_dir} has "
                    f"changed in the {watched_for:.0f}s this process has been "
                    f"waiting, so no build is in progress",
                )

    # Rule 4: the backstop. Past the limit the wait ends either way.
    if lock_age > limit:
        detail = (
            f"the lock {lock} is {lock_age / 60:.0f} min old, past the "
            f"{limit / 60:.0f} min limit"
        )
        if marker_live:
            detail += " (a build marker here still claims a live builder)"
        if artifact is not None:
            return Verdict("reclaim", f"{detail}; {artifact.name} is present")
        return Verdict("fail", detail)

    return Verdict("wait", f"the lock is {lock_age:.0f}s old")


def claim_baton(lock_path: PathLike) -> bool:
    """Take an orphaned lock away, atomically. True when this caller won.

    Rename first: ``os.rename`` gives exactly one winner among concurrent
    claimants, and the losers see the lock already gone rather than racing on
    a half-removed file.
    """
    lock = Path(lock_path)
    claimed = lock.with_name(f"{lock.name}.__sglorphan-{os.getpid()}-{time.time_ns()}")
    try:
        os.rename(lock, claimed)
    except OSError:
        # Gone already: another waiter claimed it, or the holder released it.
        return False
    try:
        os.unlink(claimed)
    except OSError:
        pass
    return True


def _selfhealing_wait(baton) -> None:
    """Replacement for ``FileBaton.wait``. Bounded, and never silent."""
    lock = Path(baton.lock_file_path)
    poll = min(getattr(baton, "wait_seconds", _POLL_SECONDS) or _POLL_SECONDS, _POLL_SECONDS)
    quiet = _env_float("SGLANG_JIT_BATON_QUIET_SECONDS", DEFAULT_QUIET_SECONDS)
    limit = _env_float("SGLANG_JIT_BATON_MAX_WAIT_SECONDS", DEFAULT_MAX_WAIT_SECONDS)

    started = time.time()
    unchanged_since = _newest_mtime(lock.parent)
    warned_at = started

    while True:
        if not selfheal_enabled():
            # Stock behaviour, on purpose: an operator turned the healing off.
            if not lock.exists():
                return
            time.sleep(poll)
            continue

        verdict = baton_verdict(
            lock,
            quiet_seconds=quiet,
            max_wait_seconds=limit,
            unchanged_since=unchanged_since,
            watching_since=started,
        )

        if verdict.action == "released":
            return

        if verdict.action == "reclaim":
            if claim_baton(lock):
                logger.warning(
                    "%s: removed an abandoned build lock %s after %.0fs -- %s. "
                    "A process was killed between finishing the build and "
                    "releasing the lock; without this every later process "
                    "waits for it forever.",
                    BATON_ORPHAN_MARKER,
                    lock,
                    time.time() - started,
                    verdict.reason,
                )
            return

        if verdict.action == "fail":
            raise RuntimeError(
                f"{BATON_ORPHAN_MARKER}: giving up on the JIT build lock "
                f"{lock}. {verdict.reason}, and the build directory "
                f"{lock.parent} holds no {'/'.join(ARTIFACT_SUFFIXES)} to fall "
                f"back on, so waiting longer cannot help. Remove that "
                f"directory to force a clean rebuild, or raise "
                f"SGLANG_JIT_BATON_MAX_WAIT_SECONDS if a build here really "
                f"does take that long."
            )

        now = time.time()
        if now - warned_at >= WARN_EVERY_SECONDS:
            warned_at = now
            logger.warning(
                "%s: still waiting on the JIT build lock %s after %.0fs (%s). "
                "Waiting up to %.0f min in total.",
                BATON_ORPHAN_MARKER,
                lock,
                now - started,
                verdict.reason,
                limit / 60,
            )
        time.sleep(poll)


def _tolerant_release(baton) -> None:
    """``FileBaton.release`` that survives a lock somebody else reclaimed.

    Stock torch calls ``os.remove`` unconditionally, so a holder whose lock was
    taken away would die in its ``finally`` block. Reclaiming must not create
    that failure.
    """
    fd = getattr(baton, "fd", None)
    if fd is not None:
        try:
            os.close(fd)
        except OSError:
            pass
        baton.fd = None
    try:
        os.remove(baton.lock_file_path)
    except OSError as exc:
        if exc.errno != errno.ENOENT:
            raise


_installed = False


def install_baton_selfheal() -> bool:
    """Patch ``FileBaton`` so no build lock can stall a process forever.

    Idempotent and best-effort: cache hygiene must never be the reason a
    process fails to start, so a failure here leaves stock torch in place.

    Patching the class is what reaches the callsites that matter. ``wait()`` is
    called from inside ``_jit_compile``, several frames below every
    ``load()`` / ``load_inline()`` in this tree and in torch itself, so there
    is no argument to pass and no wrapper to interpose from outside.
    """
    global _installed
    if _installed:
        return True
    try:
        from torch.utils.file_baton import FileBaton

        stock_wait = FileBaton.wait
        stock_release = FileBaton.release

        def wait(self) -> None:
            _selfhealing_wait(self)

        def release(self) -> None:
            _tolerant_release(self)

        # Kept reachable so the falsifier test can still exercise stock torch.
        wait.__wrapped__ = stock_wait
        release.__wrapped__ = stock_release

        FileBaton.wait = wait
        FileBaton.release = release
        _installed = True
        return True
    except Exception as exc:  # pragma: no cover - defensive
        logger.warning("JIT build-lock self-heal not installed: %r", exc)
        return False


def torch_build_directory(name: str) -> Path:
    """The directory torch will build ``name`` into.

    Asked of torch rather than reconstructed: the layout depends on
    ``TORCH_EXTENSIONS_DIR``, the python tag and the accelerator tag, and a
    reimplementation that drifted would register sources for one directory
    while torch built into another.
    """
    from torch.utils.cpp_extension import _get_build_directory

    return Path(_get_build_directory(name, False))


@contextmanager
def jit_build_guard(
    name: str,
    sources: Sequence[PathLike] = (),
    build_directory: Optional[PathLike] = None,
):
    """Arm the build-lock self-heal around a ``load()`` / ``load_inline()``.

    Yields the build directory, or None when arming failed -- in which case the
    call behaves exactly as it did before this module existed.
    """
    build_dir: Optional[Path] = None
    try:
        install_baton_selfheal()
        build_dir = (
            Path(build_directory)
            if build_directory is not None
            else torch_build_directory(name)
        )
        register_sources(build_dir, sources)
    except Exception as exc:
        logger.warning("JIT build-lock self-heal not armed for %s: %r", name, exc)
        build_dir = None
    yield build_dir
