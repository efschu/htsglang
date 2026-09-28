# SPDX-License-Identifier: Apache-2.0
"""The L3 store's write journal: the wake reads the delta, not the directory.

28.09. (27B): the wake's eviction-index rescan walked the whole persistent store
-- 821,711 files, 4.7 s at 700k, ~18 s projected at the 150 GB cap -- on every
wake. Every rank that writes into the store appends one line per committed page
(and the eviction owner one per unlinked page) to ITS OWN journal file beside
the pages; the waking owner reads only the lines appended since its last look.

Lines (one ``os.write`` each on an ``O_APPEND`` descriptor, far below
``PIPE_BUF``, so concurrent writers never interleave inside a line)::

    C <unix time> <allocated bytes> <stem>     a page landed
    E <unix time> <allocated bytes> <stem>     a page was unlinked

File name: ``L3_JOURNAL.<attach epoch>.<rank ident>.<token>.jnl`` at the store's
top level (``_iter_existing_files`` reads ``.bin`` only, so a journal is never
mistaken for a page; the staging census counts its bytes like any non-page file).
COMPACTION: a rank that attaches under launcher epoch E removes every journal
of an epoch below E -- those writers belong to a previous boot and are gone; the
attach itself walks the whole directory, so nothing they recorded is lost.

PURE: stdlib only.
"""

from __future__ import annotations

import glob
import os
import uuid
from typing import Dict, List, Optional, Tuple

PREFIX = "L3_JOURNAL."
SUFFIX = ".jnl"
ENV = "SGLANG_WEG2_STORE_JOURNAL"
FULL_WALK_EVERY_ENV = "SGLANG_WEG2_STORE_JOURNAL_FULL_WALK_EVERY"
FULL_WALK_EVERY_DEFAULT = 50


def enabled() -> bool:
    return str(os.environ.get(ENV, "1")).strip() != "0"


def full_walk_every() -> int:
    try:
        return max(0, int(os.environ.get(FULL_WALK_EVERY_ENV, FULL_WALK_EVERY_DEFAULT)))
    except ValueError:
        return FULL_WALK_EVERY_DEFAULT


def attach_epoch() -> int:
    try:
        return int(float(os.environ.get("SGLANG_WEG2_L3_EPOCH", "") or 0))
    except ValueError:
        return 0


def _epoch_of(path: str) -> Optional[int]:
    name = os.path.basename(path)
    try:
        return int(name[len(PREFIX):].split(".", 1)[0])
    except (ValueError, IndexError):
        return None


def journal_paths(store: str) -> List[str]:
    return sorted(glob.glob(os.path.join(store, PREFIX + "*" + SUFFIX)))


def compact(store: str, epoch: int) -> int:
    """Remove the journals of earlier boots (epoch below ``epoch``). Returns the
    count removed; ``epoch`` 0 (no launcher epoch) removes nothing."""
    if epoch <= 0:
        return 0
    n = 0
    for p in journal_paths(store):
        e = _epoch_of(p)
        if e is not None and e < epoch:
            try:
                os.unlink(p)
                n += 1
            except OSError:
                pass
    return n


class JournalWriter:
    """This rank's own journal. Never raises into the write path."""

    def __init__(self, store: str, ident: str):
        self.path = os.path.join(
            store, f"{PREFIX}{attach_epoch()}.{ident}.{uuid.uuid4().hex[:8]}{SUFFIX}")
        self._fd: Optional[int] = None
        self.failed = False
        self._open()

    def _open(self) -> None:
        try:
            self._fd = os.open(self.path, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o644)
        except OSError:
            self._fd = None
            self.failed = True

    def reopen(self) -> None:
        """After a clear removed every file under the store."""
        if self._fd is not None:
            try:
                os.close(self._fd)
            except OSError:
                pass
        self._open()

    def write(self, op: str, t: float, size: int, stem: str) -> None:
        if self._fd is None:
            return
        try:
            os.write(self._fd, f"{op} {t:.3f} {int(size)} {stem}\n".encode())
        except OSError:
            self.failed = True


Record = Tuple[float, str, int, str]  # (time, op, size, stem)


class JournalReader:
    """The eviction owner's view of every OTHER journal: ``(inode, offset)`` per
    file. ``mark`` records where each journal ends (before a full walk);
    ``read_delta`` returns the new records, time-ordered across journals, or
    ``None`` when a full walk is due (a known journal shrank in place or
    vanished, or one could not be read)."""

    def __init__(self, store: str, own_path: Optional[str]):
        self.store = store
        self.own_path = own_path
        self.state: Dict[str, Tuple[int, int]] = {}
        self.why_full = ""

    def mark(self) -> None:
        state: Dict[str, Tuple[int, int]] = {}
        for p in journal_paths(self.store):
            if p == self.own_path:
                continue
            try:
                st = os.stat(p)
            except OSError:
                continue
            state[p] = (st.st_ino, st.st_size)
        self.state = state

    def read_delta(self) -> Optional[Tuple[List[Record], int]]:
        paths = [p for p in journal_paths(self.store) if p != self.own_path]
        gone = set(self.state) - set(paths)
        if gone:
            self.why_full = f"journal vanished: {sorted(os.path.basename(g) for g in gone)[:3]}"
            return None
        records: List[Record] = []
        new_state: Dict[str, Tuple[int, int]] = {}
        nbytes = 0
        for p in paths:
            try:
                st = os.stat(p)
            except OSError as e:
                self.why_full = f"journal unreadable ({e})"
                return None
            ino, off = self.state.get(p, (st.st_ino, 0))
            if ino != st.st_ino:
                off = 0  # replaced by its writer: every line in it is new
            elif st.st_size < off:
                self.why_full = f"journal {os.path.basename(p)} truncated ({st.st_size} < {off})"
                return None
            if st.st_size == off:
                new_state[p] = (st.st_ino, off)
                continue
            try:
                with open(p, "rb") as f:
                    f.seek(off)
                    data = f.read(st.st_size - off)
            except OSError as e:
                self.why_full = f"journal unreadable ({e})"
                return None
            cut = data.rfind(b"\n") + 1  # a partial last line waits for its end
            nbytes += cut
            for line in data[:cut].splitlines():
                parts = line.decode(errors="replace").split(" ", 3)
                if len(parts) != 4 or parts[0] not in ("C", "E"):
                    continue
                try:
                    records.append((float(parts[1]), parts[0], int(parts[2]), parts[3]))
                except ValueError:
                    continue
            new_state[p] = (st.st_ino, off + cut)
        self.state = new_state
        records.sort(key=lambda r: r[0])
        return records, nbytes
