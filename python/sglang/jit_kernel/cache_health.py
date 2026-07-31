"""Self-healing for on-disk build caches (JIT kernels, torch extensions).

THE DEFECT
----------
sglang's jit_kernel modules are built by tvm-ffi into a content-addressed
directory under ``$TVM_FFI_CACHE_DIR`` (default ``~/.cache/tvm-ffi``). A
process killed mid-build leaves that directory holding ``build.ninja``,
``cuda.cu``, ``cuda_0.o.d`` -- and no ``.so``. Every later process that wants
the same module then dies with

    Check failed: (lib_handle_ != nullptr) ... Failed to load

The cache has no notion of a half-built entry, so it hands the wreckage back
forever. Observed on the r3 host: four such directories, three from one
afternoon of crashed boots and one from ten days earlier, all removed by hand
before the tree would boot again. That turns a transient failure -- an
interrupted first boot -- into a permanent one, which is the worst property a
cache can have.

THE RULE
--------
An entry is COMPLETE exactly when its ``.so`` exists. Everything else is
either work in progress (marked, by a live process, recently) or poison.
Poison is removed; complete entries are never touched, because rebuilding a
warm cache costs minutes of nvcc per module.

WHY A MARKER AND NOT JUST "no .so"
----------------------------------
Co-located ranks build into the same cache directory. A sweep keyed on "no
.so" alone would delete a peer rank's in-flight build and turn a fix into a
new bug. ``building_marker()`` records host + pid + timestamp; a marker whose
pid is dead, or which is older than the staleness bound, is poison rather
than progress -- a SIGKILLed builder cannot run its own cleanup, so the
liveness check is what stops such an entry from becoming immortal.

Removal is rename-then-delete: the directory first becomes a sibling with a
``.__sglpurge-`` prefix, so a concurrent reader can never observe a
half-deleted entry under the real name.

MORE THAN ONE WRITER (#181)
---------------------------
The tvm-ffi cache is not the only place a killed build leaves poison. The
barlink device extension is built by ``torch.utils.cpp_extension.load_inline``
into ``$TORCH_EXTENSIONS_DIR`` -- a different writer, a ~150 s build, and the
same failure shape: residue (``main.cpp``, ``cuda.cu``, ``build.ninja``,
``*.o``) with no ``.so``, handed back to every later boot. So the two
parameters that differ are arguments here rather than a second copy of this
file:

* ``artifact_suffixes`` -- what counts as the finished product. ``.so`` for
  both toolchains on Linux; torch would write ``.pyd`` on Windows.
* ``name_filter`` -- WHICH entries under a root this sweep is allowed to
  judge. The tvm-ffi root belongs to sglang alone, so the JIT sweep needs no
  filter. The torch extensions root is SHARED with every other cpp_extension
  in the process (and with other tools using the same cache volume), where a
  half-built entry may well be somebody else's live business. A sweep there
  must be scoped to the caller's own names, and an unscoped sweep of that root
  would be a new bug rather than a fix.
"""

from __future__ import annotations

import errno
import logging
import os
import shutil
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Callable, Iterator, List, Optional, Sequence, Union

logger = logging.getLogger(__name__)

#: Written while a build is running; removed when it finishes either way.
MARKER_BUILDING = ".sgl_jit_building"
#: Written after a successful build. Advisory only -- the `.so` is the truth,
#: so entries built before this change stay valid.
MARKER_COMPLETE = ".sgl_jit_complete"

_PURGE_PREFIX = ".__sglpurge-"

#: A build marker older than this is treated as abandoned even if some process
#: happens to hold its pid (pids are reused) or the marker came from another
#: host sharing the cache volume.
DEFAULT_STALE_SECONDS = 3600.0

#: What a finished build leaves behind. A loadable shared object, for tvm-ffi
#: and for torch's cpp_extension alike; ``.pyd`` is torch's Windows name for
#: the same thing.
DEFAULT_ARTIFACT_SUFFIXES = (".so",)

PathLike = Union[str, "os.PathLike[str]"]
NameFilter = Callable[[str], bool]


def _hostname() -> str:
    try:
        return os.uname().nodename
    except AttributeError:  # pragma: no cover - non-POSIX
        import socket

        return socket.gethostname()


def _pid_alive(pid: int) -> bool:
    if pid <= 0:
        return False
    try:
        os.kill(pid, 0)
    except OSError as exc:
        return exc.errno == errno.EPERM  # alive, just not ours
    except Exception:  # pragma: no cover - defensive
        return False
    return True


def _read_marker(path: Path) -> Optional[tuple]:
    try:
        parts = path.read_text().split("\n")
    except OSError:
        return None
    host = parts[0].strip() if parts else ""
    try:
        pid = int(parts[1].strip())
    except (IndexError, ValueError):
        pid = -1
    return host, pid


def _has_artifact(path: Path, suffixes: Sequence[str]) -> bool:
    try:
        return any(child.suffix in suffixes for child in path.iterdir())
    except OSError:
        return False


def entry_state(
    path: PathLike,
    *,
    artifact_suffixes: Sequence[str] = DEFAULT_ARTIFACT_SUFFIXES,
    stale_seconds: float = DEFAULT_STALE_SECONDS,
    now: Optional[float] = None,
) -> str:
    """Classify one cache entry: absent / complete / building / poisoned.

    ``complete`` is decided by the presence of a build artifact (by default a
    ``.so``) and nothing else, so entries written before this module existed
    are recognised as valid.
    """
    p = Path(path)
    if not p.is_dir():
        return "absent"
    try:
        children = list(p.iterdir())
    except OSError:
        return "absent"
    if not children:
        return "absent"
    if _has_artifact(p, artifact_suffixes):
        return "complete"

    marker = p / MARKER_BUILDING
    if marker.is_file():
        now = time.time() if now is None else now
        try:
            age = now - marker.stat().st_mtime
        except OSError:
            age = stale_seconds + 1
        if age <= stale_seconds:
            info = _read_marker(marker)
            if info is not None:
                host, pid = info
                # Another host on a shared cache volume: we cannot check its
                # pid, so freshness is all we have -- and it is enough, since
                # the staleness bound bounds how long a wreck can survive.
                if host != _hostname() or _pid_alive(pid):
                    return "building"
    return "poisoned"


def purge_entry(path: PathLike) -> bool:
    """Remove one cache entry. Rename first, then delete. Idempotent."""
    p = Path(path)
    if not p.exists():
        return False
    doomed = p.with_name(f"{_PURGE_PREFIX}{os.getpid()}-{time.time_ns()}-{p.name}")
    try:
        p.rename(doomed)
    except OSError:
        # Someone else got there first, or the rename is not permitted; fall
        # back to a direct delete rather than leaving the wreck in place.
        shutil.rmtree(p, ignore_errors=True)
        return not p.exists()
    shutil.rmtree(doomed, ignore_errors=True)
    return True


def _purge_debris_origin(name: str) -> Optional[str]:
    """The entry name a ``.__sglpurge-<pid>-<ns>-<name>`` leftover came from.

    Needed so a name-scoped sweep can decide whether interrupted-purge debris
    is ITS debris. None when the name cannot be parsed -- in which case a
    scoped sweep leaves it alone rather than guessing.
    """
    if not name.startswith(_PURGE_PREFIX):
        return None
    parts = name[len(_PURGE_PREFIX) :].split("-", 2)
    if len(parts) != 3:
        return None
    return parts[2]


def heal_entry(
    path: PathLike,
    *,
    artifact_suffixes: Sequence[str] = DEFAULT_ARTIFACT_SUFFIXES,
    stale_seconds: float = DEFAULT_STALE_SECONDS,
    label: str = "JIT cache",
) -> bool:
    """Remove ``path`` if and only if it is poisoned. Returns whether it was."""
    state = entry_state(
        path, artifact_suffixes=artifact_suffixes, stale_seconds=stale_seconds
    )
    if state != "poisoned":
        return False
    removed = purge_entry(path)
    if removed:
        logger.warning(
            "Discarded incomplete %s entry %s (build residue, no %s -- left "
            "behind by an interrupted build). It will be rebuilt.",
            label,
            path,
            "/".join(artifact_suffixes),
        )
    return removed


def sweep_cache_root(
    root: PathLike,
    *,
    artifact_suffixes: Sequence[str] = DEFAULT_ARTIFACT_SUFFIXES,
    stale_seconds: float = DEFAULT_STALE_SECONDS,
    name_filter: Optional[NameFilter] = None,
    label: str = "JIT cache",
) -> List[str]:
    """Discard every poisoned entry directly under ``root``.

    Returns the removed paths. Never touches an entry that holds an artifact,
    and never touches an entry a live process is building into.

    ``name_filter`` restricts the sweep to entries whose directory name it
    accepts. Pass one whenever ``root`` is shared with builds this caller does
    not own -- torch's extensions root is: a half-built entry there may be
    another extension's live business, and judging it is not this sweep's
    right. Without a filter every entry under ``root`` is in scope, which is
    correct only for a root the caller owns outright.
    """
    r = Path(root)
    if not r.is_dir():
        return []
    removed: List[str] = []
    try:
        children = sorted(r.iterdir())
    except OSError:
        return []
    for child in children:
        if child.name.startswith(_PURGE_PREFIX):
            # Debris from an interrupted purge; take it out too -- but only
            # ours, on a scoped sweep.
            if name_filter is not None:
                origin = _purge_debris_origin(child.name)
                if origin is None or not name_filter(origin):
                    continue
            shutil.rmtree(child, ignore_errors=True)
            continue
        if not child.is_dir():
            continue
        if name_filter is not None and not name_filter(child.name):
            continue
        if heal_entry(
            child,
            artifact_suffixes=artifact_suffixes,
            stale_seconds=stale_seconds,
            label=label,
        ):
            removed.append(str(child))
    if removed:
        logger.warning(
            "%s self-heal: removed %d incomplete entr%s under %s.",
            label,
            len(removed),
            "y" if len(removed) == 1 else "ies",
            r,
        )
    else:
        logger.info("%s self-heal: 0 poisoned entries under %s.", label, r)
    return removed


@contextmanager
def building_marker(path: PathLike) -> Iterator[Path]:
    """Mark ``path`` as a build in progress for the duration of the block.

    On success the marker is replaced by a completeness marker; on failure it
    is simply removed, so the directory is immediately recognisable as poison
    instead of masquerading as somebody's live build until the staleness bound
    expires. A hard kill skips both -- that case is covered by the pid
    liveness check in ``entry_state``.
    """
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    marker = p / MARKER_BUILDING
    try:
        marker.write_text(f"{_hostname()}\n{os.getpid()}\n{time.time()}\n")
    except OSError:  # pragma: no cover - unwritable cache; let the build try
        marker = None
    ok = False
    try:
        yield p
        ok = True
    finally:
        if marker is not None:
            try:
                marker.unlink()
            except OSError:
                pass
        if ok:
            try:
                (p / MARKER_COMPLETE).write_text(f"{time.time()}\n")
            except OSError:
                pass
