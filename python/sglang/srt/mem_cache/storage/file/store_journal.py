# SPDX-License-Identifier: Apache-2.0
"""The L3 store's persistent index: a snapshot plus write-ahead journals.

User order 28.09.: "einen index im systemram zum L3 ... der index wird auch im
L3 abgelegt und aktualisiert und ggf. EIN mal angelegt, falls missing". The
RAM index is the eviction owner's LRU index (it already existed; the walk built
it). It now lives in the store as

* ``L3_INDEX.snap`` -- the index as it is (every page file ``(mtime, stem,
  bytes)`` oldest first, plus epoch and kernel boot id), PICKLED behind one
  line ``WEG2-L3-INDEX v2 n=<N> bytes=<B> sha1=<of the payload>``; written
  tmp + fsync + rename by ONE rank (the D group's owner) when a boot attaches
  -- never in a flip -- and
* ``L3_JOURNAL.<epoch>.<rank ident>.<token>.jnl`` -- per writing rank, O_APPEND,
  ``# boot_id=<id>`` then one line per event::

      R <t> <bytes> <stem>    a write is ABOUT to create/extend the page (write-ahead)
      C <t> <bytes> <stem>    it landed (allocated bytes after the write)
      A <t> 0 <stem>          the reserved write did not happen
      E <t> <bytes> <stem>    the page was unlinked (logged AFTER the unlink)

CRASH ORDER, the whole of it: ``R`` precedes every file change and ``E`` follows
every unlink, so a crash can leave only an ENTRY WITHOUT A FILE (an ``R`` with no
``C``/``A`` is resolved at load by one stat; an ``E`` lost after its unlink leaves
an entry the next ENOENT strikes), never a file without an entry. A page whose
file is gone is a miss, never a wrong read: readers stat/open, never trust the
index. fsync: journals per sleep (off the flip, a thread), the snapshot always.

A boot loads snapshot + the previous boots' journals; it walks the directory
ONCE only when the snapshot is missing or broken, or the kernel boot id changed
(a host reboot/power loss can drop unsynced journal lines).

PURE: stdlib only.
"""

from __future__ import annotations

import glob
import hashlib
import os
import pickle
import uuid
from typing import Dict, List, Optional, Tuple

PREFIX = "L3_JOURNAL."
SUFFIX = ".jnl"
SNAP = "L3_INDEX.snap"
ENV = "SGLANG_WEG2_STORE_JOURNAL"
#: bytes of host RAM per index entry (Python OrderedDict, ~100-char stem),
#: measured 28.09.: 280 -- the host ledger books entries x this per owner.
RAM_BYTES_PER_ENTRY = 280


def enabled() -> bool:
    return str(os.environ.get(ENV, "1")).strip() != "0"


def attach_epoch() -> int:
    try:
        return int(float(os.environ.get("SGLANG_WEG2_L3_EPOCH", "") or 0))
    except ValueError:
        return 0


def kernel_boot_id() -> str:
    try:
        with open("/proc/sys/kernel/random/boot_id") as f:
            return f.read().strip()
    except OSError:
        return "unknown"


def epoch_of(path: str) -> Optional[int]:
    name = os.path.basename(path)
    try:
        return int(name[len(PREFIX):].split(".", 1)[0])
    except (ValueError, IndexError):
        return None


def journal_paths(store: str) -> List[str]:
    return sorted(glob.glob(os.path.join(store, PREFIX + "*" + SUFFIX)))


def remove_journals_before(store: str, epoch: int) -> int:
    """Compaction: the journals of boots before ``epoch`` (0 removes nothing)."""
    if epoch <= 0:
        return 0
    n = 0
    for p in journal_paths(store):
        e = epoch_of(p)
        if e is not None and e < epoch:
            try:
                os.unlink(p)
                n += 1
            except OSError:
                pass
    return n


Record = Tuple[float, str, int, str]  # (time, op, bytes, stem)


def parse_lines(data: bytes) -> Tuple[List[Record], Optional[str]]:
    """Records and the ``# boot_id=`` a journal declares (None if absent)."""
    out: List[Record] = []
    boot = None
    for line in data.splitlines():
        s = line.decode(errors="replace")
        if s.startswith("# boot_id="):
            boot = s[len("# boot_id="):].strip()
            continue
        parts = s.split(" ", 3)
        if len(parts) != 4 or parts[0] not in ("R", "C", "A", "E"):
            continue
        try:
            out.append((float(parts[1]), parts[0], int(parts[2]), parts[3]))
        except ValueError:
            continue
    return out, boot


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
            os.write(self._fd, f"# boot_id={kernel_boot_id()}\n".encode())
            self.failed = False
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

    def sync(self) -> None:
        if self._fd is not None:
            try:
                os.fsync(self._fd)
            except OSError:
                pass


class JournalReader:
    """The eviction owner's view of every OTHER journal: ``(inode, offset)`` per
    file; ``read_delta`` returns the lines appended since, time-ordered across
    journals. It never asks for a walk: a journal that shrank in place or
    vanished is NAMED (``notes``) and followed from where it now is."""

    def __init__(self, store: str, own_path: Optional[str]):
        self.store = store
        self.own_path = own_path
        self.state: Dict[str, Tuple[int, int]] = {}
        self.notes: List[str] = []

    def mark(self, epoch_below: Optional[int] = None) -> None:
        """Every other journal is consumed up to its current end -- except, with
        ``epoch_below``, journals of that epoch or later (this boot's), which
        stay at offset 0 so their lines are read at the next delta."""
        state: Dict[str, Tuple[int, int]] = {}
        for p in journal_paths(self.store):
            if p == self.own_path:
                continue
            try:
                st = os.stat(p)
            except OSError:
                continue
            e = epoch_of(p)
            fresh = epoch_below is not None and (e is None or e >= epoch_below)
            state[p] = (st.st_ino, 0 if fresh else st.st_size)
        self.state = state

    def read_delta(self) -> Tuple[List[Record], int]:
        self.notes = []
        paths = [p for p in journal_paths(self.store) if p != self.own_path]
        for g in sorted(set(self.state) - set(paths)):
            self.notes.append(f"journal vanished: {os.path.basename(g)}")
        records: List[Record] = []
        new_state: Dict[str, Tuple[int, int]] = {}
        nbytes = 0
        for p in paths:
            try:
                st = os.stat(p)
            except OSError:
                continue
            ino, off = self.state.get(p, (st.st_ino, 0))
            if ino != st.st_ino:
                off = 0
            elif st.st_size < off:
                self.notes.append(f"journal {os.path.basename(p)} shrank ({st.st_size} < {off})")
                off = st.st_size
            if st.st_size == off:
                new_state[p] = (st.st_ino, off)
                continue
            try:
                with open(p, "rb") as f:
                    f.seek(off)
                    data = f.read(st.st_size - off)
            except OSError as e:
                self.notes.append(f"journal {os.path.basename(p)} unreadable ({e})")
                new_state[p] = (st.st_ino, off)
                continue
            cut = data.rfind(b"\n") + 1  # a partial last line waits for its end
            nbytes += cut
            recs, _boot = parse_lines(data[:cut])
            records.extend(recs)
            new_state[p] = (st.st_ino, off + cut)
        self.state = new_state
        records.sort(key=lambda r: r[0])
        return records, nbytes


# ---------------------------------------------------------------------------
# the snapshot
# ---------------------------------------------------------------------------

MAGIC = b"WEG2-L3-INDEX v2"


def write_snapshot(store: str, items, epoch: int) -> Tuple[int, int]:
    """``items`` = list of (mtime, stem, bytes) oldest first -- the index as it
    is, pickled (no format of our own), behind one version/checksum line.
    Atomic: tmp + fsync + rename + directory fsync."""
    items = list(items)
    payload = pickle.dumps(
        {"epoch": int(epoch), "boot_id": kernel_boot_id(), "items": items},
        protocol=pickle.HIGHEST_PROTOCOL)
    total = sum(int(it[2]) for it in items)
    head = (MAGIC + f" n={len(items)} bytes={total} "
            f"sha1={hashlib.sha1(payload).hexdigest()}\n".encode())
    path = os.path.join(store, SNAP)
    tmp = f"{path}.w{os.getpid()}.{uuid.uuid4().hex[:6]}.tmp"
    with open(tmp, "wb") as f:
        f.write(head)
        f.write(payload)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)
    try:
        dfd = os.open(store, os.O_RDONLY)
        try:
            os.fsync(dfd)
        finally:
            os.close(dfd)
    except OSError:
        pass
    return len(items), total


def remove_snapshot_tmps(store: str) -> int:
    """A snapshot tmp left by a writer that died mid-write (user: ".tmp beim
    Boot weg"). Only the ONE snapshot writer calls this, before it writes."""
    n = 0
    for p in glob.glob(os.path.join(store, SNAP + ".w*.tmp")):
        try:
            os.unlink(p)
            n += 1
        except OSError:
            pass
    return n


def index_entries(store: str) -> Tuple[int, str]:
    """How many entries the RAM index of ``store`` holds, for the host ledger:
    the snapshot header's ``n=`` (one line read), else the page-file count of
    the directory (names only, no stat -- the first boot walks it once anyway)."""
    try:
        with open(os.path.join(store, SNAP), "rb") as f:
            head = f.readline(4096)
        if head.startswith(MAGIC + b" "):
            for kv in head[len(MAGIC):].decode(errors="replace").split():
                if kv.startswith("n="):
                    return int(kv[2:]), "snapshot header"
    except (OSError, ValueError):
        pass
    n = 0
    try:
        with os.scandir(store) as it:
            for e in it:
                if e.name.endswith(".bin"):
                    n += 1
    except OSError:
        return 0, "no store yet"
    return n, "file count, no snapshot yet"


class Snapshot:
    def __init__(self, epoch: int, boot_id: str, items: "Dict[str, Tuple[float, int]]", inode: int):
        self.epoch, self.boot_id, self.items, self.inode = epoch, boot_id, items, inode


def load_snapshot(store: str) -> Tuple[Optional[Snapshot], str]:
    """``(snapshot, "")`` or ``(None, why)`` -- missing, torn, or a sum mismatch."""
    path = os.path.join(store, SNAP)
    try:
        st = os.stat(path)
        with open(path, "rb") as f:
            data = f.read()
    except FileNotFoundError:
        return None, "no snapshot"
    except OSError as e:
        return None, f"snapshot unreadable ({e})"
    nl = data.find(b"\n")
    if nl < 0 or not data.startswith(MAGIC + b" "):
        return None, "snapshot torn or of another version (header)"
    head = dict(kv.split("=", 1) for kv in data[len(MAGIC):nl].decode(errors="replace").split()
                if "=" in kv)
    payload = data[nl + 1:]
    if head.get("sha1") != hashlib.sha1(payload).hexdigest():
        return None, "snapshot checksum mismatch"
    try:
        obj = pickle.loads(payload)
        items: Dict[str, Tuple[float, int]] = {
            stem: (float(m), int(n)) for m, stem, n in obj["items"]}
        epoch = int(obj.get("epoch", 0))
    except Exception as e:  # noqa: BLE001 -- any unpickling fault = a broken snapshot
        return None, f"snapshot unreadable ({type(e).__name__})"
    if int(head.get("n", -1)) != len(items):
        return None, "snapshot count mismatch"
    return Snapshot(epoch, str(obj.get("boot_id", "")), items, st.st_ino), ""


def replay(items: "Dict[str, Tuple[float, int]]", records: List[Record],
           stat_size=None) -> Tuple[int, int]:
    """Apply journal records (time-ordered) to a stem -> (mtime, bytes) map in
    place. An ``R`` left without its ``C``/``A`` is resolved by ONE stat through
    ``stat_size(stem) -> (mtime, bytes) | None``. Returns (applied, stat'ed)."""
    pending: Dict[str, float] = {}
    for t, op, size, stem in records:
        if op == "R":
            pending[stem] = t
        elif op == "C":
            pending.pop(stem, None)
            items.pop(stem, None)
            items[stem] = (t, size)
        elif op == "A":
            pending.pop(stem, None)
        elif op == "E":
            pending.pop(stem, None)
            items.pop(stem, None)
    stated = 0
    if stat_size is not None:
        for stem in pending:
            stated += 1
            got = stat_size(stem)
            if got is not None:
                items.pop(stem, None)
                items[stem] = got
    return len(records), stated
