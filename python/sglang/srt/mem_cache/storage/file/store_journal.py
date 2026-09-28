# SPDX-License-Identifier: Apache-2.0
"""The L3 store's persistent index: a snapshot plus write-ahead journals.

User order 28.09.: "einen index im systemram zum L3 ... der index wird auch im
L3 abgelegt und aktualisiert und ggf. EIN mal angelegt, falls missing". The
RAM index is the eviction owner's LRU index (it already existed; the walk built
it). It now lives in the store as

* ``L3_INDEX.snap`` -- the index as it is (every page file ``(mtime, stem,
  bytes)`` oldest first, plus epoch and kernel boot id), PICKLED in frames
  behind one line ``WEG2-L3-INDEX v3 n=<N>``, the last frame carrying n,
  bytes and the sha1 of the frames before it; written
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

import contextlib
import fcntl
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

    def __init__(self, store: str, ident: str, epoch: Optional[int] = None):
        self.path = os.path.join(
            store, f"{PREFIX}{attach_epoch() if epoch is None else int(epoch)}."
                   f"{ident}.{uuid.uuid4().hex[:8]}{SUFFIX}")
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

MAGIC = b"WEG2-L3-INDEX v3"
#: entries per pickled frame: load and write hold ONE frame beside the index,
#: never the whole payload (R4 28.09.: v2 held payload + list + dict + copy,
#: measured 419 B/entry transient over the 263 B/entry index at attach).
FRAME = 16384


class _HashIO:
    """File wrapper: sha1 over every byte read or written through it."""

    def __init__(self, f):
        self.f, self.h = f, hashlib.sha1()

    def write(self, b):
        self.h.update(b)
        return self.f.write(b)

    def read(self, n=-1):
        b = self.f.read(n)
        self.h.update(b)
        return b

    def readline(self, n=-1):
        b = self.f.readline(n)
        self.h.update(b)
        return b

    def readinto(self, buf):
        n = self.f.readinto(buf)
        self.h.update(memoryview(buf)[:n])
        return n


def write_snapshot(store: str, items, epoch: int, n: Optional[int] = None) -> Tuple[int, int]:
    """``items`` = iterable of (mtime, stem, bytes) oldest first -- the index as
    it is, pickled (no format of our own) in frames of ``FRAME`` entries behind
    one line ``WEG2-L3-INDEX v3 n=<N>``; the last frame carries n, bytes and
    the sha1 of every frame before it. Streamed: no payload, no list. Atomic:
    tmp + fsync + rename + directory fsync."""
    if n is None:
        items = list(items)
        n = len(items)
    path = os.path.join(store, SNAP)
    tmp = f"{path}.w{os.getpid()}.{uuid.uuid4().hex[:6]}.tmp"
    count = total = 0
    with open(tmp, "wb") as f:
        f.write(MAGIC + f" n={int(n)}\n".encode())
        w = _HashIO(f)
        pickle.dump({"epoch": int(epoch), "boot_id": kernel_boot_id()}, w,
                    protocol=pickle.HIGHEST_PROTOCOL)
        frame: list = []
        for it in items:
            frame.append(it)
            total += int(it[2])
            if len(frame) >= FRAME:
                pickle.dump(frame, w, protocol=pickle.HIGHEST_PROTOCOL)
                count += len(frame)
                frame = []
        if frame:
            pickle.dump(frame, w, protocol=pickle.HIGHEST_PROTOCOL)
            count += len(frame)
        pickle.dump({"end": 1, "n": count, "bytes": total, "sha1": w.h.hexdigest()}, f,
                    protocol=pickle.HIGHEST_PROTOCOL)
        f.flush()
        os.fsync(f.fileno())
    if count != int(n):
        os.unlink(tmp)
        raise ValueError(f"snapshot: {count} entries written, {n} announced")
    os.replace(tmp, path)
    try:
        dfd = os.open(store, os.O_RDONLY)
        try:
            os.fsync(dfd)
        finally:
            os.close(dfd)
    except OSError:
        pass
    return count, total


LOCK = "L3_INDEX.lock"


@contextlib.contextmanager
def snapshot_lock(store: str):
    """The store's ONE-WALK lock (NF metal rc12z30c: launcher, PP0-2 and D TP0
    each walked a store without a snapshot -- 5 walks for one directory).
    Whoever finds no usable snapshot takes it, looks again, and only then
    walks and writes; everyone behind it loads what it wrote. flock on a file
    in the store, so it holds across the containers that bind the store."""
    fd = None
    try:
        fd = os.open(os.path.join(store, LOCK), os.O_RDWR | os.O_CREAT, 0o644)
        fcntl.flock(fd, fcntl.LOCK_EX)
    except OSError:
        if fd is not None:
            os.close(fd)
        fd = None  # no lock possible: walk unlocked (the old behaviour), never hang
    try:
        yield fd is not None
    finally:
        if fd is not None:
            try:
                fcntl.flock(fd, fcntl.LOCK_UN)
            finally:
                os.close(fd)


def persist(store: str, items, n: int, epoch: int) -> Tuple[int, int, int]:
    """Snapshot + compaction in one step: the torn tmps of a dead writer go,
    the snapshot is written, the journals of earlier boots (``< epoch``) are
    folded into it. Returns (entries, bytes, journals removed). Callers hold
    ``snapshot_lock``."""
    remove_snapshot_tmps(store)
    n, nbytes = write_snapshot(store, items, epoch, n=n)
    return n, nbytes, remove_journals_before(store, epoch)


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


_SHARD_HEX = "0123456789abcdef"


def page_path(store: str, stem: str) -> str:
    """Where the page ``stem`` lives: its shard (``hicache_storage.page_shard``,
    same rule, kept torch-free here), else the pre-sharding flat path."""
    prefix = stem[:2]
    shard = prefix if len(prefix) == 2 and all(c in _SHARD_HEX for c in prefix) else "zz"
    p = os.path.join(store, shard, f"{stem}.bin")
    if os.path.exists(p):
        return p
    flat = os.path.join(store, f"{stem}.bin")
    if os.path.exists(flat):
        return flat
    # a page outside the shard rule (never written by the backend, which always
    # shards by it -- an offline copy or a hand-made store): one glob, on a miss only
    hits = glob.glob(os.path.join(glob.escape(store), "*", glob.escape(f"{stem}.bin")))
    return hits[0] if hits else p


#: seconds a journaled change may precede its journal line (R before a
#: rename, C after it) -- never enough for a whole other boot to hide in
FOREIGN_SLACK_S = 2.0


def foreign_writer_since_journals(store: str) -> Optional[str]:
    """A page shard changed after the newest snapshot/journal write: a writer
    WITHOUT a journal touched the store (an image before the journal -- the
    bridge boots of 28.09. -- or an offline tool). Its pages are invisible to
    snapshot + journals: never evicted against the cap, never moved by a
    revoked window. Every journaled change is followed by a journal write
    (C/E after the rename/unlink), so its shard is never newer than the
    newest journal. One stat per shard (<= 257), attach only."""
    bound = 0.0
    for p in [os.path.join(store, SNAP)] + journal_paths(store):
        try:
            bound = max(bound, os.stat(p).st_mtime)
        except OSError:
            pass
    try:
        with os.scandir(store) as it:
            for e in it:
                if len(e.name) != 2 or not (e.name == "zz" or all(
                        c in _SHARD_HEX for c in e.name)):
                    continue
                if not e.is_dir(follow_symlinks=False):
                    continue
                m = e.stat(follow_symlinks=False).st_mtime
                if m > bound + FOREIGN_SLACK_S:
                    return (f"shard {e.name} changed at {m:.3f}, after the newest "
                            f"snapshot/journal write {bound:.3f}: a writer without "
                            f"a journal touched the store")
    except OSError as e:
        return f"store unreadable ({e})"
    return None


def load_index(store: str, *, own_path: Optional[str] = None,
               skip_epoch_from: Optional[int] = None, stat_size=None,
               open_intents: Optional[List[str]] = None):
    """``(items, provenance)`` from snapshot + journals, or ``(None, why)``: the
    ONE load the ranks' attach and the launcher's attach share. ``items`` =
    stem -> (mtime, bytes). Journals from ``skip_epoch_from`` on (this boot's)
    are left for the wake; ``open_intents`` receives the stems whose write
    began and never ended. ``None`` means: walk once (missing/torn snapshot,
    another kernel boot)."""
    boot = kernel_boot_id()
    for _attempt in range(5):
        snap, why = load_snapshot(store)
        if snap is None:
            return None, why
        if snap.boot_id != boot:
            return None, ("kernel boot id changed since the snapshot (host reboot or "
                          "power loss can drop unsynced journal lines)")
        items = snap.items  # the loaded dict itself: no copy (R4)
        records: Optional[List[Record]] = []
        used = 0
        for p in journal_paths(store):
            if p == own_path:
                continue
            e = epoch_of(p) or 0
            if e < snap.epoch or (skip_epoch_from is not None and e >= skip_epoch_from):
                continue  # compacted into the snapshot / this boot's (read at the wake)
            try:
                with open(p, "rb") as f:
                    data = f.read()
            except OSError:
                records = None
                break
            recs, jboot = parse_lines(data[: data.rfind(b"\n") + 1])
            if jboot is not None and jboot != boot:
                return None, f"journal {os.path.basename(p)} from another kernel boot"
            records.extend(recs)
            used += 1
        try:
            same = os.stat(os.path.join(store, SNAP)).st_ino == snap.inode
        except OSError:
            same = False
        if records is None or not same:
            continue  # the snapshot writer compacted meanwhile: load its new one
        foreign = foreign_writer_since_journals(store)
        if foreign is not None:
            return None, foreign
        records.sort(key=lambda r: r[0])
        opened: List[str] = []
        applied, stated = replay(items, records, stat_size, opened)
        if open_intents is not None:
            open_intents.extend(opened)
        return items, (f"snapshot epoch {snap.epoch} ({len(snap.items)} files) + {used} "
                       f"journal(s), {applied} line(s), {stated} open intent(s) stat'ed")
    return None, "the snapshot kept changing while it was read"


def index_stems(store: str, epoch: Optional[int] = None) -> Tuple[Optional[list], str]:
    """The stems the persistent index names (snapshot + earlier boots'
    journals), for seeding the shared #1459 stem index without a walk;
    ``(None, why)`` when the index is unusable (the caller walks)."""
    items, why = load_index(store, skip_epoch_from=epoch if epoch else None)
    return (None, why) if items is None else (list(items), why)


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
            top = list(it)
    except OSError:
        return 0, "no store yet"
    for e in top:
        if e.name.endswith(".bin"):
            n += 1
        elif e.is_dir(follow_symlinks=False):  # the page shards
            try:
                with os.scandir(e.path) as sub:
                    n += sum(1 for x in sub if x.name.endswith(".bin"))
            except OSError:
                pass
    return n, "file count, no snapshot yet"


class Snapshot:
    def __init__(self, epoch: int, boot_id: str, items: "Dict[str, Tuple[float, int]]", inode: int):
        self.epoch, self.boot_id, self.items, self.inode = epoch, boot_id, items, inode


def load_snapshot(store: str) -> Tuple[Optional[Snapshot], str]:
    """``(snapshot, "")`` or ``(None, why)`` -- missing, torn, of another
    version, or a sum mismatch. Streamed frame by frame into the dict."""
    path = os.path.join(store, SNAP)
    try:
        f = open(path, "rb")
    except FileNotFoundError:
        return None, "no snapshot"
    except OSError as e:
        return None, f"snapshot unreadable ({e})"
    with f:
        try:
            ino = os.fstat(f.fileno()).st_ino
            head = f.readline(4096)
            if not head.startswith(MAGIC + b" "):
                return None, "snapshot torn or of another version (header)"
            r = _HashIO(f)
            meta = pickle.load(r)
            items: Dict[str, Tuple[float, int]] = {}
            total = 0
            while True:
                before = r.h.copy()
                obj = pickle.load(r)
                if isinstance(obj, dict):
                    break
                for m, stem, n in obj:
                    items[stem] = (float(m), int(n))
                    total += int(n)
            if f.read(1):
                return None, "snapshot: bytes after the end frame"
        except Exception as e:  # noqa: BLE001 -- any unpickling fault = a broken snapshot
            return None, f"snapshot torn ({type(e).__name__})"
    if obj.get("sha1") != before.hexdigest():
        return None, "snapshot checksum mismatch"
    if int(obj.get("n", -1)) != len(items) or int(obj.get("bytes", -1)) != total:
        return None, "snapshot count mismatch"
    return Snapshot(int(meta.get("epoch", 0)), str(meta.get("boot_id", "")), items, ino), ""


def replay(items: "Dict[str, Tuple[float, int]]", records: List[Record],
           stat_size=None, open_intents: Optional[List[str]] = None) -> Tuple[int, int]:
    """Apply journal records (time-ordered) to a stem -> (mtime, bytes) map in
    place. An ``R`` left without its ``C``/``A`` is resolved by ONE stat through
    ``stat_size(stem) -> (mtime, bytes) | None`` and, when asked, listed in
    ``open_intents`` (a write that died: its staging file may be left).
    Returns (applied, stat'ed)."""
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
    if open_intents is not None:
        open_intents.extend(pending)
    if stat_size is not None:
        for stem in pending:
            stated += 1
            got = stat_size(stem)
            if got is not None:
                items.pop(stem, None)
                items[stem] = got
    return len(records), stated
