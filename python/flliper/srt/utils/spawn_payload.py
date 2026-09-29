"""Scheduler spawn payload by FILE REFERENCE instead of through the spawn pipe.

NF-Bootzeit H1 (28.09.): measured on the NF boots rc12z10 08:31 and 09:14
(``boot_weg2_dkrnfh91dprsavisbar1dauer09280831_*.P.log``, lines 32/35/51):
the parent logged its three rank starts 8 s apart on P and 4-5 s apart on D,
while a 27B boot of the same tree starts its three ranks in the same second
(``boot_weg2_dkr27b...0928_082020.P.log`` lines 24-26). P's PP2 therefore
reached ``Init torch distributed`` 14 s after PP0, D's TP2 9 s after TP0 --
every boot, before a single byte was loaded.

The mechanism is the ``spawn`` start method itself. ``Popen._launch`` writes
the pickled preparation data AND the pickled process object into ONE pipe
and returns only when the write completed. The child reads the preparation
data, then imports the parent's main module (``flliper.launch_server``, 5-8 s
here), and only THEN reads the process object. A payload larger than the
pipe (64 KiB) therefore holds the parent inside ``proc.start()`` until the
child finished that import -- rank starts become strictly serial. NF's
ServerArgs crosses the pipe size, 27B's does not.

The fix keeps the payload small: the object is pickled into a private file
(``mkstemp``, mode 0600) and the process receives a tiny reference whose
unpickling (in the child) reads, verifies (sha256) and unlinks that file.
The child gets the SAME object it got before; only the transport changed.
Anything that cannot be pickled standalone falls back to the inline object
(the old behaviour), named in the log.
"""

from __future__ import annotations

import atexit
import hashlib
import logging
import os
import pickle
import tempfile
from typing import Any, List

logger = logging.getLogger(__name__)

#: Files this process wrote and the child has not consumed yet; removed at
#: exit so a child that died before reading leaves no residue.
_PENDING: List[str] = []
_ATEXIT = {"armed": False}

FILE_PREFIX = "flliper-spawn-payload-"


def _cleanup_pending() -> None:
    while _PENDING:
        path = _PENDING.pop()
        try:
            os.unlink(path)
        except OSError:
            pass


def _load_by_reference(path: str, sha256: str, nbytes: int) -> Any:
    """Child side: read, verify and unlink the payload file, return the object."""
    with open(path, "rb") as fh:
        data = fh.read()
    try:
        os.unlink(path)
    except OSError:
        pass
    if len(data) != int(nbytes) or hashlib.sha256(data).hexdigest() != sha256:
        raise RuntimeError(
            f"spawn payload {path}: {len(data)} bytes / sha mismatch "
            f"(expected {nbytes} bytes, sha256 {sha256[:16]}...) -- the file "
            f"was changed between the parent's write and this child's read"
        )
    return pickle.loads(data)


class SpawnPayloadRef:
    """Pickles to ``_load_by_reference(path, sha, n)``: a few hundred bytes."""

    __slots__ = ("path", "sha256", "nbytes")

    def __init__(self, path: str, sha256: str, nbytes: int):
        self.path = path
        self.sha256 = sha256
        self.nbytes = int(nbytes)

    def __reduce__(self):
        return (_load_by_reference, (self.path, self.sha256, self.nbytes))


def _pid_alive(pid: int) -> bool:
    try:
        os.kill(int(pid), 0)
    except ProcessLookupError:
        return False
    except (PermissionError, OSError):
        return True  # exists, or cannot tell -> keep the file
    return True


def sweep_stale(directory: str = None) -> int:
    """Remove payload files whose WRITER is dead (a parent killed before its
    atexit ran, SIGKILL / OOM). The writer pid is part of the file name; a
    living writer's files are never touched. Returns the count removed."""
    d = directory or tempfile.gettempdir()
    n = 0
    try:
        names = os.listdir(d)
    except OSError:
        return 0
    for name in names:
        if not name.startswith(FILE_PREFIX):
            continue
        pid = name[len(FILE_PREFIX):].split("-", 1)[0]
        if not pid.isdigit() or _pid_alive(int(pid)):
            continue
        try:
            os.unlink(os.path.join(d, name))
            n += 1
        except OSError:
            pass
    return n


def by_reference(obj: Any, *, directory: str = None) -> Any:
    """Return a small picklable stand-in for ``obj`` (or ``obj`` itself when it
    cannot be pickled standalone). Call once PER child: the child unlinks the
    file it read; the parent removes unread ones at exit; a later parent
    removes those of a writer that died without its atexit (``sweep_stale``)."""
    if not _ATEXIT["armed"]:
        swept = sweep_stale(directory)
        if swept:
            logger.info("spawn payload: removed %d stale file(s) of dead writers", swept)
    try:
        data = pickle.dumps(obj, protocol=pickle.HIGHEST_PROTOCOL)
    except Exception as exc:  # noqa: BLE001 -- the inline pipe is the old path
        logger.info(
            "spawn payload: %s is not picklable standalone (%s: %s) -- passed "
            "inline through the spawn pipe as before",
            type(obj).__name__, type(exc).__name__, str(exc)[:160])
        return obj
    fd, path = tempfile.mkstemp(prefix=f"{FILE_PREFIX}{os.getpid()}-", suffix=".pkl",
                                dir=directory)
    try:
        with os.fdopen(fd, "wb") as fh:
            fh.write(data)
    except Exception:
        try:
            os.unlink(path)
        except OSError:
            pass
        raise
    _PENDING.append(path)
    if not _ATEXIT["armed"]:
        atexit.register(_cleanup_pending)
        _ATEXIT["armed"] = True
    # One line per child: the size is the reason this path exists (a payload
    # above the 64 KiB pipe serialised the rank starts), so the metal says it.
    logger.info("spawn payload: %s by reference, %d bytes (%s)",
                type(obj).__name__, len(data), path)
    return SpawnPayloadRef(path, hashlib.sha256(data).hexdigest(), len(data))
