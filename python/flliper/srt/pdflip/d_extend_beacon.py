"""S49 D-EXTEND-BEACON (NF 18.10.): group D tells the front WHICH requests it has really begun to prefill.

SOLL v2 R6 after the user's decision of 18.10. ~21:15Z (SCHEDULER-SOLL.md, R6, verbatim "Nur echt laufend zählt
(Recommended)"): only a D prefill that ALREADY COMPUTES counts into the set's S; granted-but-not-started does not, and
a big request waits its T_halb (R4) to the end.

THE GAP. The front knew three moments of a D prefill: the grant (its own verdict), the hand-over to D (``enter_leg2``,
the POST) and the first content (prefill over). Between the hand-over and D's first extend lie D's tokenizer, its
waiting queue, its X/KV gates and holds -- 80 ms (NF rc13 20:33:48, pdflip-22-249: front D-ADMIT .456, D recv .552,
R6 verdict .592, extend .633) up to 30 s (pdflip-20-244: recv 48.525, first extend 34:18.648). The book counted every
grant from the verdict on, so R6 rerouted a p_only member 3.1 s before its T_halb for a kurz prefill D had not begun
(S46 report (3).4, pdflip-20-244 TTFT 114 s).

THE SIGNAL (no clock threshold, no log line): every scheduler rank of group D keeps one mmap file
``<arena>/d_extend/D-pid<pid>.bin``. At a request's FIRST extend batch after its arrival on the rank
(``Scheduler._run_batch_forward``, the very hook D-LEAD-MS-1007 stamps ``extend_ms`` at) it writes
``(rid_key, t_ns)`` into a ring slot -- one ``struct.pack_into``, no syscall, no collective, no new endpoint, nothing on
the forward path but a dict lookup per rid of an extend batch. The front reads the group's files (live pids of the
group's session, like the progress beacon) and counts a granted row as RUNNING iff a rank recorded an extend of that
rid at or after the row's grant (``t_ns >= t_grant``: a PARK-HANDBACK re-grant of the same rid is not "started" by
its earlier leg).

R11: D only reports a fact; the verdict stays the front's. Every rank of the group runs the same extend batch, so any
rank's record is the group's (the front takes the union).

Switch ``FLLIPER_PDFLIP_D_EXTEND_BEACON`` (default on). Off: no file, and the front's R6 counts the book as before.
On without a readable file (no arena dir, D not yet written): nothing is proven running, nothing counts (the user's
rule), named by a counter.
"""
from __future__ import annotations

import collections
import glob
import hashlib
import mmap
import os
import struct
import time
from typing import Callable, Dict, Iterable, Optional

ENV = "FLLIPER_PDFLIP_D_EXTEND_BEACON"
SUBDIR = "d_extend"
#: header: magic, version, slots, head (records written), pid
_HDR_FMT = "<IIIqq"
_HDR_SIZE = struct.calcsize(_HDR_FMT)
_MAGIC = 0x44455842  # "DEXB"
_VERSION = 1
#: one record: rid key (u64, 0 = empty), extend start (epoch ns)
_REC_FMT = "<Qq"
_REC_SIZE = struct.calcsize(_REC_FMT)
#: records kept: a grant lives seconds to minutes, D starts far fewer than this many prefills in that time
SLOTS = 1024
#: rids received but not yet extended, dropped oldest-first (an aborted request never extends)
ARMED_CAP = 4096


def enabled(env=None) -> bool:
    """``FLLIPER_PDFLIP_D_EXTEND_BEACON`` (environ.py, default on); ``env`` = a mapping to read instead."""
    if env is None:
        from flliper.srt.environ import envs

        return bool(envs.FLLIPER_PDFLIP_D_EXTEND_BEACON.get())
    raw = (env.get(ENV, "") or "").strip().lower()
    return raw not in ("0", "false", "no", "off")


def rid_key(rid: str) -> int:
    """A rid's 64-bit key (blake2b); 0 is reserved for an empty slot."""
    k = int.from_bytes(hashlib.blake2b(str(rid).encode("utf-8", "replace"), digest_size=8).digest(), "little")
    return k or 1


def beacon_dir(tag: str = "", env=None) -> str:
    from flliper.srt.pdflip.resume_via_p import arena_dir

    base = arena_dir(tag, env)
    return os.path.join(base, SUBDIR) if base else ""


class DExtendBeacon:
    """The writer on one scheduler rank of group D."""

    def __init__(self, path: str, *, slots: int = SLOTS, clock_ns: Callable[[], int] = time.time_ns):
        self._clock_ns = clock_ns
        self._slots = int(slots)
        self._armed: "collections.OrderedDict[str, None]" = collections.OrderedDict()
        self._head = 0
        self.mm = None
        size = _HDR_SIZE + self._slots * _REC_SIZE
        try:
            os.makedirs(os.path.dirname(path), exist_ok=True)
            fd = os.open(path, os.O_RDWR | os.O_CREAT | os.O_TRUNC, 0o644)
            try:
                os.ftruncate(fd, size)
                self.mm = mmap.mmap(fd, size)
            finally:
                os.close(fd)
            struct.pack_into(_HDR_FMT, self.mm, 0, _MAGIC, _VERSION, self._slots, 0, os.getpid())
        except Exception:  # noqa: BLE001 -- a beacon never stops the scheduler; the front then counts nothing
            self.mm = None

    def note_recv(self, *, rid: str) -> None:
        """The rid reached this rank's scheduler: its next extend is its start."""
        self._armed.pop(rid, None)
        self._armed[rid] = None
        while len(self._armed) > ARMED_CAP:
            self._armed.popitem(last=False)

    def note_extend(self, *, rids: Iterable[str]) -> None:
        """An extend batch runs: record every rid of it that has not extended since its arrival."""
        if self.mm is None:
            return
        now = None
        for rid in rids:
            if rid not in self._armed:
                continue
            del self._armed[rid]
            if now is None:
                now = int(self._clock_ns())
            try:
                off = _HDR_SIZE + (self._head % self._slots) * _REC_SIZE
                struct.pack_into(_REC_FMT, self.mm, off, rid_key(rid), now)
                self._head += 1
                struct.pack_into("<q", self.mm, 12, self._head)
            except Exception:  # noqa: BLE001
                return


def build(*, group: str, tag: str = "", env=None) -> Optional[DExtendBeacon]:
    """The writer for a rank of group D with the switch on and an arena dir, else None."""
    if not enabled(env) or (group or "").strip().upper() != "D":
        return None
    d = beacon_dir(tag, env)
    if not d:
        return None
    return DExtendBeacon(os.path.join(d, f"D-pid{os.getpid()}.bin"))


def read_group(directory: str, group: str, sid: int,
               session_of: Optional[Callable[[int], Optional[int]]] = None) -> Optional[Dict[int, int]]:
    """``{rid_key: newest extend start ns}`` over the group's LIVE ranks (pid in the group's session); None when no
    such file is readable (no channel -- nothing is proven running)."""
    if not directory or not sid:
        return None
    if session_of is None:
        from flliper.srt.pdflip.front_health import pid_session as session_of
    out: Dict[int, int] = {}
    seen = False
    for path in glob.glob(os.path.join(directory, f"{group}-pid*.bin")):
        try:
            with open(path, "rb") as fh:
                raw = fh.read()
            magic, ver, slots, _head, pid = struct.unpack_from(_HDR_FMT, raw, 0)
        except Exception:  # noqa: BLE001
            continue
        if magic != _MAGIC or ver != _VERSION or session_of(int(pid)) != int(sid):
            continue
        seen = True
        n = min(int(slots), max(0, (len(raw) - _HDR_SIZE) // _REC_SIZE))
        for key, t_ns in struct.iter_unpack(_REC_FMT, raw[_HDR_SIZE:_HDR_SIZE + n * _REC_SIZE]):
            if key and t_ns > out.get(key, 0):
                out[key] = int(t_ns)
    return out if seen else None


def started(reading: Optional[Dict[int, int]], rid: str, t_grant_s: float) -> bool:
    """D began the extend of ``rid`` at or after its grant at ``t_grant_s`` (epoch s; same host clock)."""
    if not reading:
        return False
    t_ns = reading.get(rid_key(rid))
    return t_ns is not None and t_ns >= int(float(t_grant_s) * 1e9)
