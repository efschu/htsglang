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

S56 (NF rc15 19.10., user: "um 9:20:26 wird nur EIN request prefillt und zwar 4492 neue token"; SOLL R6 + R11):
VERSION 2 carries D's REAL REST. The front counted a running D prefill into R6's S with its own arrival price
(pdflip-6-11: 3513), while D matched 98240 of 99217 and had 977 to compute (``#969 EXTENT ... 977``, X-GATE
uncached=977) -- S=8005 > X=6216 flipped for 4492 tokens. Every extend batch of a rid now writes
``(rid_key, t_first_extend_ns, rest, seq)``: ``rest`` = the tokens of the request D has NOT computed when the batch
starts (``len(origin_input_ids) + len(output_ids) - len(prefix_indices)``, the in-flight chunk included), ``seq`` the
rank's write counter (the newest record of a rid wins). The ranks only report the number; the front decides (R11).
Several ranks: the newest record per rank, the MAX over ranks (deterministic, independent of the file order).

EZ-S2b review A2 (VERSION 3, SOLL R6 "laufend"; NF rc19: 26 of 43 counted rests were ALREADY COMPUTED, the verdict fell 22-211 ms
after the forward ended -- pdflip-104-2646 05:26:34.960 vs forward end .918, pdflip-0-1 23:06:43.576 vs .498): v2 wrote ``rest``
only at the batch START, and a decode batch writes no extend record, so the record kept a rest D had long computed. v3 writes a
SECOND record when the extend batch's forward has ended (``note_extend_done``, called from ``Scheduler.process_batch_result``
after the result is synchronised): same rid, same first-extend start, newer ``seq``, ``rest`` = ``rest_at_start - extend_len``
(0 for a single-chunk prefill; the later chunks of a chunked prefill report their own start rest again). The front counts a
rest only while it is > 0. Ranks still only report a number (R11); with several ranks the MAX rest is kept, so a rank that has
not finished its forward keeps the rest alive (agreement by MAX, as before).
"""
from __future__ import annotations

import collections
import glob
import hashlib
import mmap
import os
import struct
import time
from typing import Callable, Dict, Iterable, Optional, Tuple

ENV = "FLLIPER_PDFLIP_D_EXTEND_BEACON"
SUBDIR = "d_extend"
#: header: magic, version, slots, head (records written), pid
_HDR_FMT = "<IIIqq"
_HDR_SIZE = struct.calcsize(_HDR_FMT)
_MAGIC = 0x44455842  # "DEXB"
#: v3 (EZ-S2b review A2): the record LAYOUT is v2's; the SEMANTICS of ``rest`` gain the forward end -- see ``note_extend_done``
_VERSION = 3
#: one record (v2): rid key (u64, 0 = empty), FIRST extend start (epoch ns), rest tokens D has not computed at the
#: batch start (-1 = not known), the rank's write counter (newest record of a key wins)
_REC_FMT = "<Qqqq"
#: v1 record (read only): rid key, extend start
_REC_FMT_V1 = "<Qq"
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
        #: S56: rid -> its FIRST extend start (ns) while it keeps extending (chunked prefill: every chunk re-reports
        #: the rest under the same start); bounded like ``_armed``
        self._live: "collections.OrderedDict[str, int]" = collections.OrderedDict()
        #: v3: rid -> the rest once the CURRENT extend batch's forward has ended (rest at start - extend_len)
        self._after: "collections.OrderedDict[str, int]" = collections.OrderedDict()
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

    def note_extend(self, *, rids: Iterable[str], rests: Optional[Iterable[int]] = None,
                    extend_lens: Optional[Iterable[int]] = None) -> None:
        """An extend batch runs: record every rid of it. A rid's FIRST extend after its arrival stamps its start; every
        later extend (the next chunk) re-reports the rest under that start. ``rests`` (S56, parallel to ``rids``): the
        tokens D has not computed for each rid when this batch starts; None = not known (-1 in the record).
        ``extend_lens`` (v3, parallel): the tokens this batch computes for each rid; with it the rest after the forward
        is known (``note_extend_done`` writes it)."""
        if self.mm is None:
            return
        now = None
        rest_list = list(rests) if rests is not None else None
        len_list = list(extend_lens) if extend_lens is not None else None
        for i, rid in enumerate(rids):
            if rid in self._armed:
                del self._armed[rid]
                if now is None:
                    now = int(self._clock_ns())
                t_first = now
                self._live.pop(rid, None)
                self._live[rid] = t_first
                while len(self._live) > ARMED_CAP:
                    self._live.popitem(last=False)
            elif rid in self._live:
                t_first = self._live[rid]
            else:
                continue
            rest = -1
            if rest_list is not None and i < len(rest_list):
                try:
                    rest = max(0, int(rest_list[i]))
                except (TypeError, ValueError):
                    rest = -1
            self._after.pop(rid, None)
            if rest >= 0 and len_list is not None and i < len(len_list):
                try:
                    self._after[rid] = max(0, rest - max(0, int(len_list[i])))
                    while len(self._after) > ARMED_CAP:
                        self._after.popitem(last=False)
                except (TypeError, ValueError):
                    pass
            try:
                off = _HDR_SIZE + (self._head % self._slots) * _REC_SIZE
                self._head += 1
                struct.pack_into(_REC_FMT, self.mm, off, rid_key(rid), int(t_first), int(rest), int(self._head))
                struct.pack_into("<q", self.mm, 12, self._head)
            except Exception:  # noqa: BLE001
                return

    def note_extend_done(self, *, rids: Iterable[str]) -> None:
        """v3: the forward of the extend batch ``rids`` has ended on this rank: re-report each rid under its first-extend
        start with the rest AFTER the batch (``rest_at_start - extend_len``; 0 when the batch finished the prefill)."""
        if self.mm is None:
            return
        for rid in rids:
            after = self._after.pop(rid, None)
            t_first = self._live.get(rid)
            if after is None or t_first is None:
                continue
            try:
                off = _HDR_SIZE + (self._head % self._slots) * _REC_SIZE
                self._head += 1
                struct.pack_into(_REC_FMT, self.mm, off, rid_key(rid), int(t_first), int(after), int(self._head))
                struct.pack_into("<q", self.mm, 12, self._head)
            except Exception:  # noqa: BLE001
                return

    def note_done(self, *, rid: str) -> None:
        """The rid left this rank (finished, aborted): no later extend re-reports it."""
        self._live.pop(rid, None)
        self._after.pop(rid, None)
        self._armed.pop(rid, None)


def req_rest(req) -> int:
    """S56: the tokens of ``req`` D has not computed at a batch start -- its whole sequence (prompt + output, a park
    resume included) minus the prefix the batch extends from (cached + earlier chunks). Never raises; -1 = unknown."""
    try:
        total = len(req.origin_input_ids) + len(getattr(req, "output_ids", None) or ())
        pre = getattr(req, "prefix_indices", None)
        return max(0, int(total) - (len(pre) if pre is not None else 0))
    except Exception:  # noqa: BLE001
        return -1


def req_extend_len(req) -> int:
    """v3: the tokens the extend batch computes for ``req`` (``extend_input_len``: the chunk). Never raises; -1 = unknown."""
    try:
        return max(0, int(req.extend_input_len))
    except Exception:  # noqa: BLE001
        return -1


def build(*, group: str, tag: str = "", env=None) -> Optional[DExtendBeacon]:
    """The writer for a rank of group D with the switch on and an arena dir, else None."""
    if not enabled(env) or (group or "").strip().upper() != "D":
        return None
    d = beacon_dir(tag, env)
    if not d:
        return None
    return DExtendBeacon(os.path.join(d, f"D-pid{os.getpid()}.bin"))


def read_group_full(directory: str, group: str, sid: int,
                    session_of: Optional[Callable[[int], Optional[int]]] = None
                    ) -> Optional[Dict[int, Tuple[int, int]]]:
    """``{rid_key: (first extend start ns, rest)}`` over the group's LIVE ranks (pid in the group's session); None when
    no such file is readable (no channel -- nothing is proven running). Per rank the NEWEST record of a key (its write
    counter); over ranks the larger start, then the larger rest (deterministic: the file order never decides).
    ``rest`` -1 = the rank did not report one (a v1 file)."""
    if not directory or not sid:
        return None
    if session_of is None:
        from flliper.srt.pdflip.front_health import pid_session as session_of
    out: Dict[int, Tuple[int, int]] = {}
    seen = False
    for path in sorted(glob.glob(os.path.join(directory, f"{group}-pid*.bin"))):
        try:
            with open(path, "rb") as fh:
                raw = fh.read()
            magic, ver, slots, _head, pid = struct.unpack_from(_HDR_FMT, raw, 0)
        except Exception:  # noqa: BLE001
            continue
        if magic != _MAGIC or ver not in (1, 2, _VERSION) or session_of(int(pid)) != int(sid):
            continue
        seen = True
        own: Dict[int, Tuple[int, int, int]] = {}  # key -> (seq, t_first, rest)
        if ver == 1:
            rs = struct.calcsize(_REC_FMT_V1)
            n = min(int(slots), max(0, (len(raw) - _HDR_SIZE) // rs))
            for key, t_ns in struct.iter_unpack(_REC_FMT_V1, raw[_HDR_SIZE:_HDR_SIZE + n * rs]):
                if key and (key not in own or t_ns > own[key][1]):
                    own[key] = (0, int(t_ns), -1)
        else:
            n = min(int(slots), max(0, (len(raw) - _HDR_SIZE) // _REC_SIZE))
            for key, t_ns, rest, seq in struct.iter_unpack(_REC_FMT, raw[_HDR_SIZE:_HDR_SIZE + n * _REC_SIZE]):
                if key and (key not in own or seq > own[key][0]):
                    own[key] = (int(seq), int(t_ns), int(rest))
        for key, (_seq, t_ns, rest) in own.items():
            prev = out.get(key)
            if prev is None or (t_ns, rest) > prev:
                out[key] = (t_ns, rest)
    return out if seen else None


def read_group(directory: str, group: str, sid: int,
               session_of: Optional[Callable[[int], Optional[int]]] = None) -> Optional[Dict[int, int]]:
    """``{rid_key: newest extend start ns}`` (the pre-S56 reading, kept for its readers)."""
    full = read_group_full(directory, group, sid, session_of)
    return None if full is None else {k: v[0] for k, v in full.items()}


def _entry(reading, rid: str) -> Optional[Tuple[int, int]]:
    v = reading.get(rid_key(rid)) if reading else None
    if v is None:
        return None
    if isinstance(v, tuple):
        return int(v[0]), int(v[1])
    return int(v), -1  # a start-only reading (v1 file, or a caller's stub)


def started(reading: Optional[Dict[int, object]], rid: str, t_grant_s: float) -> bool:
    """D began the extend of ``rid`` at or after its grant at ``t_grant_s`` (epoch s; same host clock)."""
    e = _entry(reading, rid)
    return e is not None and e[0] >= int(float(t_grant_s) * 1e9)


def rest_of(reading: Optional[Dict[int, object]], rid: str, t_grant_s: float) -> Optional[int]:
    """S56: D's REAL rest of ``rid`` (tokens not yet computed, as the newest extend batch saw it) when D began it at or
    after its grant; None = not started, or started without a reported rest."""
    e = _entry(reading, rid)
    if e is None or e[0] < int(float(t_grant_s) * 1e9) or e[1] < 0:
        return None
    return int(e[1])
