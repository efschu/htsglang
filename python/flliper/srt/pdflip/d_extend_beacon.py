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

NF-R6-REST-EXAKT (20.10., SOLL R6 + R11, user order 19.10. "NICHTS SCHÄTZEN, EXAKTE ZAHLEN: Einstufung uncached/S/R6-Rest/Sitz nur
aus D-Radix/L2/L3 zum Entscheidungszeitpunkt"): v3 wrote the after-rest only from ``process_batch_result`` -- and with overlap
scheduling (D runs ``overlap=True``) that runs AFTER the NEXT batch was built and launched. rc28 _1020_174205 pdflip-12-275: chunk 1
(768 of 1394) ended its forward at 18:36:32.918-.930 on all three ranks, the next batch (chunk 2, 626) was launched only at
18:36:33.x, so the record still said 1394 when the front decided at 32.977-.983 (``r6_rest=1394``; D's exact rest was 626). Now the
rank also ARMS the done of every extend batch with a device event recorded behind its forward (``arm_done``) and POLLS it
(``poll_done``, ``event.query()``: no sync, no wait) at its loop's top and again before the next launch: the after-rest is written
as soon as the forward has really ended on the device. ``note_extend_done`` (the result path) stays the backstop: what the poll
already wrote is not written twice, what no event armed is written there as before, and a result always flushes the older armed
batches first (one stream: an earlier forward has ended when a later one's result is in) -- the per-rid FIFO stays aligned. The
record layout and ``_VERSION`` are unchanged; only the moment of the after-rest record moves to the forward's real end.

NF-RC30-R6STALE (21.10., VERSION 4, SOLL R6 + R6k + R7 + R11; user order 19.10. "NICHTS SCHÄTZEN, EXAKTE ZAHLEN"): ``rest`` says
what D still COMPUTES -- the start rest of the oldest batch whose forward has not ended (review A1, FIFO head), so it includes the
chunk(s) D has already handed to a forward. Those chunks never reach P: an in-flight forward ends on D (an abort lands at D's next
loop top, after it), its KV stays in D's tree and its write-through, and P reads it from the store. NF boot ``..._1021_063252``:
pdflip-48-687 (u=3280 halb, chunk 1 = 2496 launched 07:53:11.128, chunk 2 = 784 launched 14.365) was counted with r6_rest=3280 at
the R6 verdict 14,400 (chunk 1's done not yet polled when chunk 2 launched, chunk 2's forward then held the host 1.25 s), P computed
16 tokens; pdflip-42-671 counted 4681 -> P 9; pdflip-46-680 counted 5151 -> P 2911 (= 5151 - chunk 1 of 2304, plus the store's
page). v4 adds ``free`` to every record: the tokens of the request NO forward has been launched for (the newest launched batch's
``rest_at_start - extend_len``; -1 = not known). That is what an R6 redirect can move to P, so it is what R6 counts into S.
``rest`` keeps its meaning for the readers that ask "does D still compute it" (R6k finish hold, NF-1b ``_d_extending_rids``).
Over the ranks ``free`` is the MIN: chunk boundaries are group-agreed (#794 GROUP-NARROWED, every rank cuts the same number) and
a forward one rank launched runs on every rank (its collectives), so those tokens are D's on all of them.

Review NF-RC30-R6STALE (21.10., Auflagen A1/A2; the v4 layout below was never booted, it replaces the first v4 draft):
- A1 (chunk-boundary window): D answers a probe in its recv step, BEFORE it builds and launches the next chunk of a running
  chunked prefill (``Scheduler.event_loop_*``: recv -> poll -> get_next_batch_to_run -> run_batch/note_extend). A verdict made
  on that answer saw ``free`` = the rest after chunk k while D launched chunk k+1 in the same iteration, before any abort could
  land (live 42-671: +6 ms, 48-687: +1 ms, 50-698: +36 ms). Now the rank marks, at the END of its recv step (aborts applied),
  every rid whose next chunk is fixed (``commit_next``: the scheduler's ``chunked_req``): its records carry ``free`` =
  ``COMMITTED`` (-2) until that chunk's start record (``note_extend``) writes the exact free; a rid the built batch does NOT
  launch is released at once (``resolve_commit``). The front counts and redirects nothing of a committed rid and holds a final
  set verdict for that one D iteration (R11: D reports a fact, the front decides).
- A2: every record carries ``n_launch`` (extend batches launched for the rid in this leg); the reader keeps the per-rank values
  (``_Entry.ranks``) and ``free_of``/``rest_of`` merge ONLY the ranks whose record is of the asked grant (``t_first >=
  t_grant``) -- a lagging rank's record of an EARLIER leg (park-handback / resume re-grant) no longer pulls free to 0.
  Merge of ``free``: the ranks with the highest ``n_launch`` decide (the others have not noted the newest chunk yet); any of
  them COMMITTED -> COMMITTED, else the MIN of their known values.

NF-RC31-R6REST (21.10., SOLL R6 + R7 + R11; user order 19.10. "NICHTS SCHÄTZEN, EXAKTE ZAHLEN"): ``free`` is exact at the moment it
is read -- but an R6 redirect moves what is free when D APPLIES the abort, and D keeps launching chunks until then. Live NF boot
``..._82784e1138_1021_202608`` pdflip-6-401 (halb, u=6550): chunk 1 (2240) ended 21:15:36,826, the front read free=4310 (no
COMMITTED, nothing in flight) and decided REROUTE_P at 36,982 (S = 3786 + 4310 = 8096 > X = 7370); D's next recv step came before
the abort (sent 37,114, the leg's ROS poll), D launched chunk 2 (2300, forward 36,9.. -> 40,066), the abort applied after it, P got
the tail 2069/2070: real S = 5855 < X (SOLL R7 violated). The rank now writes a HALT record (``note_halt``) when the abort of a rid
it extended is APPLIED (no forward of it launches any more): under the derived key ``halt_key(rid)`` -- the v4 layout and
``_VERSION`` are unchanged, a v4 reader ignores the key -- with ``free`` = the tokens no forward was launched for AT THE HALT (a
chunk in flight ends on D). That is the number the front's R6 decides on (``halted_free``): what the redirect really moves to P.
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
_VERSION = 4
#: one record (v4): rid key (u64, 0 = empty), FIRST extend start (epoch ns), rest tokens D has not computed at the
#: batch start (-1 = not known), the rank's write counter (newest record of a key wins), free = tokens no forward has been
#: launched for (NF-RC30-R6STALE; -1 = not known, -2 = COMMITTED: the next chunk is fixed, its start record follows),
#: n_launch = extend batches launched for the rid in this leg on this rank
_REC_FMT = "<Qqqqqq"
#: ``free`` of a rid whose next chunk the rank has fixed in its recv step and not launched yet (review A1)
COMMITTED = -2
#: v2/v3 record (read only): the v4 record without ``free``
_REC_FMT_V2 = "<Qqqq"
#: v1 record (read only): rid key, extend start
_REC_FMT_V1 = "<Qq"
_REC_SIZE = struct.calcsize(_REC_FMT)
#: records kept: a grant lives seconds to minutes, D starts far fewer than this many prefills in that time
SLOTS = 1024
#: rids received but not yet extended, dropped oldest-first (an aborted request never extends)
ARMED_CAP = 4096
#: NF-R6-REST-EXAKT: extend batches armed behind their forward and not yet written (a handful in flight at most)
PENDING_CAP = 256


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
        #: v4 (NF-RC30-R6STALE): rid -> the tokens no forward has been launched for (the NEWEST launched batch's after-rest)
        self._free: "collections.OrderedDict[str, int]" = collections.OrderedDict()
        #: review A1/A2: rid -> extend batches launched in this leg; rids whose next chunk is fixed (COMMITTED); last rest written
        self._nl: "collections.OrderedDict[str, int]" = collections.OrderedDict()
        self._committed: set = set()
        self._last_rest: "collections.OrderedDict[str, int]" = collections.OrderedDict()
        #: NF-R6-REST-EXAKT: batch key (the rank's forward counter) -> (rids, ready) armed behind its forward, oldest first
        self._pending: "collections.OrderedDict[int, Tuple[list, Callable[[], bool]]]" = collections.OrderedDict()
        #: keys whose done the poll already wrote (the result path then writes nothing)
        self._polled: "collections.OrderedDict[int, None]" = collections.OrderedDict()
        self._head = 0
        #: NF-RC31-R6REST: this rank applies halts (``announce_halt``) -- the cap record is re-written before the ring wraps it
        self._halt_cap = False
        self._cap_at = 0
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
                self._nl.pop(rid, None)
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
            # review A1 (SOLL R6, danger direction "rest too small"): under overlap chunk k+1 is launched on the host while
            # chunk k still runs on the device -- its prefix already moved past chunk k. The record then reports the start
            # rest of the OLDEST batch of this rid whose done is not written yet (the FIFO head), never less
            q0 = self._after.get(rid)
            report = int(q0[0][0]) if (q0 and rest >= 0) else rest
            # v4 (NF-RC30-R6STALE): what no forward has been launched for -- this batch's prefix already covers every earlier
            # launched chunk, so its start rest minus its own length is exactly the part an R6 redirect can still move to P
            free = -1
            if rest >= 0 and len_list is not None and i < len(len_list):
                try:
                    free = max(0, rest - max(0, int(len_list[i])))
                except (TypeError, ValueError):
                    free = -1
            self._free.pop(rid, None)
            self._free[rid] = free
            while len(self._free) > ARMED_CAP:
                self._free.popitem(last=False)
            if rest >= 0 and len_list is not None and i < len(len_list):
                try:
                    # FIFO per rid (review A2a): with overlap scheduling chunk k+1 is launched before chunk k's result is
                    # processed, so one value would hand chunk k's done the after-rest of the LAST chunk
                    q = self._after.get(rid)
                    if q is None:
                        q = self._after[rid] = collections.deque(maxlen=64)
                    q.append((rest, max(0, rest - max(0, int(len_list[i])))))
                    while len(self._after) > ARMED_CAP:
                        self._after.popitem(last=False)
                except (TypeError, ValueError):
                    pass
            rest = report
            # review A1/A2: this launch fixes the chunk the recv step committed -> the exact free replaces the marker
            self._committed.discard(rid)
            n = self._nl.pop(rid, 0) + 1
            self._nl[rid] = n
            while len(self._nl) > ARMED_CAP:
                self._nl.popitem(last=False)
            if not self._put(rid, t_first, rest, free, n):
                return

    def arm_done(self, *, key: int, rids: Iterable[str], ready: Callable[[], bool]) -> None:
        """NF-R6-REST-EXAKT: the extend batch ``key`` (the rank's forward counter) was issued; ``ready()`` (a device event's
        non-blocking ``query``) says when its forward has ended on the device. ``poll_done`` then writes its after-rest."""
        if self.mm is None:
            return
        self._pending.pop(int(key), None)
        self._pending[int(key)] = (list(rids), ready)
        while len(self._pending) > PENDING_CAP:
            # dropped unwritten: the result path writes it (the key is neither pending nor polled)
            self._pending.popitem(last=False)

    def poll_done(self) -> int:
        """NF-R6-REST-EXAKT: write the after-rest of every armed batch whose forward has ended, oldest first, and stop at the
        first that has not (one stream: a later forward cannot have ended before it). Never blocks; returns the batches
        written."""
        n = 0
        while self._pending:
            key, (rids, ready) = next(iter(self._pending.items()))
            try:
                done = bool(ready())
            except Exception:  # noqa: BLE001 -- an unreadable event: the result path writes it
                done = False
            if not done:
                break
            self._pending.popitem(last=False)
            self._write_done(rids)
            self._polled[key] = None
            while len(self._polled) > PENDING_CAP:
                self._polled.popitem(last=False)
            n += 1
        return n

    def note_extend_done(self, *, rids: Iterable[str], key: Optional[int] = None) -> None:
        """v3: the forward of the extend batch ``rids`` has ended on this rank: re-report each rid under its first-extend
        start with the rest AFTER the batch (``rest_at_start - extend_len``; 0 when the batch finished the prefill).
        NF-R6-REST-EXAKT: with ``key`` (the batch's forward counter) a batch the poll already wrote is not written again, and
        the older armed batches are written first (their forwards have ended: this result is in)."""
        if self.mm is None:
            return
        if key is not None:
            k = int(key)
            # the older armed batches first (keys are the rank's monotone forward counter): their forwards have ended
            while self._pending:
                k0 = next(iter(self._pending))
                if k0 >= k:
                    break
                _k0, (r0, _ready) = self._pending.popitem(last=False)
                self._write_done(r0)
                self._polled[k0] = None
            if k in self._polled:
                del self._polled[k]
                return
            self._pending.pop(k, None)
        self._write_done(rids)

    def _write_done(self, rids: Iterable[str]) -> None:
        for rid in rids:
            q = self._after.get(rid)
            t_first = self._live.get(rid)
            if not q or t_first is None:
                continue
            after = q.popleft()[1]
            if not q:
                self._after.pop(rid, None)
            # v4: a later launched chunk keeps its own (smaller) free; with none launched it is this batch's after-rest; a rid
            # whose next chunk the recv step fixed stays COMMITTED (review A1)
            free = COMMITTED if rid in self._committed else self._free.get(rid, after)
            if not self._put(rid, t_first, after, free, self._nl.get(rid, 0)):
                return

    def _put(self, rid: str, t_first: int, rest: int, free: int, n: int) -> bool:
        """One record of ``rid``; False = the mmap failed (the beacon then writes nothing more this call)."""
        if not self._put_key(rid_key(rid), t_first, rest, free, n):
            return False
        self._last_rest.pop(rid, None)
        self._last_rest[rid] = int(rest)
        while len(self._last_rest) > ARMED_CAP:
            self._last_rest.popitem(last=False)
        return True

    def _put_key(self, key: int, t_first: int, rest: int, free: int, n: int) -> bool:
        try:
            off = _HDR_SIZE + (self._head % self._slots) * _REC_SIZE
            self._head += 1
            struct.pack_into(_REC_FMT, self.mm, off, int(key), int(t_first), int(rest), int(self._head), int(free), int(n))
            struct.pack_into("<q", self.mm, 12, self._head)
        except Exception:  # noqa: BLE001
            return False
        if self._halt_cap and key != HALT_CAP_KEY and self._head - self._cap_at >= self._slots // 2:
            self._put_cap()
        return True

    def _put_cap(self) -> None:
        if self._put_key(HALT_CAP_KEY, int(self._clock_ns()), 0, 0, 0):
            self._cap_at = self._head

    def announce_halt(self) -> None:
        """NF-RC31-R6REST: this rank's scheduler writes a halt record when it APPLIES an abort (``note_halt``, wired in
        ``Scheduler.process_pending_chunked_abort`` / ``_abort_request_now``). The front stops an R6-counted prefill first and
        decides on its halted rest only for a D that announces it -- a D without halt records would leave every R6 verdict
        waiting for the liveness guard. The cap record rides the ring (re-written before the ring can wrap it)."""
        if self.mm is None:
            return
        self._halt_cap = True
        self._put_cap()

    def note_halt(self, *, rid: str) -> bool:
        """NF-RC31-R6REST (SOLL R6 + R7, R11): the abort of ``rid`` is APPLIED on this rank -- no forward of it is launched any
        more (a chunked prefill: ``process_pending_chunked_abort`` cleared ``chunked_req`` at the top of the scheduling step; any
        other: the abort took it out of the waiting queue / running batch). ONE halt record under ``halt_key(rid)``: ``free`` =
        the tokens no forward was launched for at the halt -- final, what an R6 redirect really moves to P (a chunk already in
        flight ends on D, its KV stays in D's tree and the store); ``rest`` = what D still computes (the in-flight chunk
        included). A COMMITTED marker of the rid is resolved (its next chunk never launches). A rid this leg never extended
        writes nothing (D never began it; the front's R6 never counted it). Returns True when a halt record was written."""
        if self.mm is None:
            return False
        t_first = self._live.get(rid)
        if t_first is None:
            return False
        n = self._nl.get(rid, 0)
        free = int(self._free.get(rid, -1))
        rest = int(self._last_rest.get(rid, -1))
        q0 = self._after.get(rid)
        if q0:
            rest = int(q0[0][0])  # the oldest launched batch whose forward has not ended (review A1): D still computes it
        if free < 0:
            free = rest  # no launch length known (a v2-style caller): the rest is all D has not computed
        if rid in self._committed:
            self._committed.discard(rid)
            self._put(rid, t_first, self._last_rest.get(rid, -1), free, n)
        return self._put_key(halt_key(rid), t_first, rest, free, n)

    def commit_next(self, *, rids: Iterable[str]) -> None:
        """Review A1 (SOLL R6/R7, R11): the END of the rank's recv step -- aborts applied, the next chunk of every ``rids``
        (the scheduler's running ``chunked_req``) is fixed for this iteration. Its record says COMMITTED until that chunk's
        start record (``note_extend``) or until ``resolve_commit`` finds it was not launched. Only rids this leg extended."""
        if self.mm is None:
            return
        for rid in rids:
            t_first = self._live.get(rid)
            if t_first is None or rid in self._committed:
                continue
            self._committed.add(rid)
            self._put(rid, t_first, self._last_rest.get(rid, -1), COMMITTED, self._nl.get(rid, 0))

    def resolve_commit(self, *, launched: Iterable[str]) -> None:
        """Review A1: the batch is built. A committed rid it launches keeps the marker until its start record (written in
        ``run_batch`` a few statements later); a committed rid it does NOT launch (held, parked, aborted) gets its real free
        back now -- the marker never outlives the iteration that set it."""
        if self.mm is None or not self._committed:
            return
        go = set(launched)
        for rid in [r for r in self._committed if r not in go]:
            self._committed.discard(rid)
            t_first = self._live.get(rid)
            if t_first is not None:
                self._put(rid, t_first, self._last_rest.get(rid, -1), self._free.get(rid, -1), self._nl.get(rid, 0))

    def note_done(self, *, rid: str) -> None:
        """The rid left this rank (finished, aborted): no later extend re-reports it."""
        self._live.pop(rid, None)
        self._after.pop(rid, None)
        self._free.pop(rid, None)
        self._nl.pop(rid, None)
        self._committed.discard(rid)
        self._last_rest.pop(rid, None)
        self._armed.pop(rid, None)


#: NF-RC31-R6REST: the key of the "this rank applies halts" record (``announce_halt``)
HALT_CAP_KEY = int.from_bytes(hashlib.blake2b(b"\x00d-extend-halt-capable", digest_size=8).digest(), "little") or 1


def halt_key(rid: str) -> int:
    """NF-RC31-R6REST: the key of ``rid``'s HALT record (``note_halt``) -- a derived key, so the v4 layout carries it and a v4
    reader that does not ask for it never sees it."""
    return rid_key(str(rid) + "\x00halt")


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
    w = DExtendBeacon(os.path.join(d, f"D-pid{os.getpid()}.bin"))
    w.announce_halt()  # NF-RC31-R6REST: the scheduler wires note_halt -- the front may stop R6 prefills first
    return w


def read_group_full(directory: str, group: str, sid: int,
                    session_of: Optional[Callable[[int], Optional[int]]] = None
                    ) -> Optional[Dict[int, Tuple[int, int, int]]]:
    """``{rid_key: (first extend start ns, rest, free)}`` over the group's LIVE ranks (pid in the group's session); None when
    no such file is readable (no channel -- nothing is proven running). Per rank the NEWEST record of a key (its write
    counter); over ranks the larger start and, separately, the larger rest (deterministic: the file order never decides).
    ``rest`` -1 = the rank did not report one (a v1 file). ``free`` (v4): the MIN over the ranks that report one (a forward one
    rank launched runs on all), -1 = no rank reported one (a v1-v3 file)."""
    if not directory or not sid:
        return None
    if session_of is None:
        from flliper.srt.pdflip.front_health import pid_session as session_of
    out: Dict[int, Tuple[int, int, int]] = {}
    per: Dict[int, list] = {}
    seen = False
    for path in sorted(glob.glob(os.path.join(directory, f"{group}-pid*.bin"))):
        try:
            with open(path, "rb") as fh:
                raw = fh.read()
            magic, ver, slots, _head, pid = struct.unpack_from(_HDR_FMT, raw, 0)
        except Exception:  # noqa: BLE001
            continue
        if magic != _MAGIC or ver not in (1, 2, 3, _VERSION) or session_of(int(pid)) != int(sid):
            continue
        seen = True
        own: Dict[int, Tuple[int, int, int, int, int]] = {}  # key -> (seq, t_first, rest, free, n_launch)
        if ver == 1:
            rs = struct.calcsize(_REC_FMT_V1)
            n = min(int(slots), max(0, (len(raw) - _HDR_SIZE) // rs))
            for key, t_ns in struct.iter_unpack(_REC_FMT_V1, raw[_HDR_SIZE:_HDR_SIZE + n * rs]):
                if key and (key not in own or t_ns > own[key][1]):
                    own[key] = (0, int(t_ns), -1, -1, 0)
        elif ver in (2, 3):
            rs = struct.calcsize(_REC_FMT_V2)
            n = min(int(slots), max(0, (len(raw) - _HDR_SIZE) // rs))
            for key, t_ns, rest, seq in struct.iter_unpack(_REC_FMT_V2, raw[_HDR_SIZE:_HDR_SIZE + n * rs]):
                if key and (key not in own or seq > own[key][0]):
                    own[key] = (int(seq), int(t_ns), int(rest), -1, 0)
        else:
            n = min(int(slots), max(0, (len(raw) - _HDR_SIZE) // _REC_SIZE))
            for key, t_ns, rest, seq, free, nl in struct.iter_unpack(_REC_FMT, raw[_HDR_SIZE:_HDR_SIZE + n * _REC_SIZE]):
                if key and (key not in own or seq > own[key][0]):
                    own[key] = (int(seq), int(t_ns), int(rest), int(free), int(nl))
        for key, (_seq, t_ns, rest, free, nl) in own.items():
            per.setdefault(key, []).append((t_ns, rest, free, nl))
    for key, ranks in per.items():
        # review A2b (R11): the start stamp and the rest are merged SEPARATELY -- the later start never decides the rest;
        # the MAX rest over the ranks does (a rank that has not finished keeps it alive). ``free``: ``_merge_free``. The
        # per-rank values ride along (``ranks``) so a reader can keep only the ranks of ITS grant (review A2)
        out[key] = _Entry((max(r[0] for r in ranks), max(r[1] for r in ranks), _merge_free(ranks)), ranks)
    return out if seen else None


class _Entry(tuple):
    """``(first extend start ns, rest, free)`` -- a plain tuple to every reader -- plus ``ranks``: the per-rank
    ``(t_first, rest, free, n_launch)`` it was merged from (review A2)."""

    def __new__(cls, vals, ranks):
        obj = super().__new__(cls, vals)
        obj.ranks = tuple(ranks)
        return obj


def _merge_free(ranks) -> int:
    """Review A1/A2 (R11): the ranks that noted the most chunks of this leg decide (a rank with fewer has not noted the newest
    launch yet; chunk boundaries are group-agreed, #794); any of them COMMITTED -> COMMITTED, else the MIN of their known
    values; nothing known -> -1."""
    if not ranks:
        return -1
    top = max(int(r[3]) for r in ranks)
    cands = [r for r in ranks if int(r[3]) == top]
    if any(int(r[2]) == COMMITTED for r in cands):
        return COMMITTED
    known = [int(r[2]) for r in cands if int(r[2]) >= 0]
    return min(known) if known else -1


def _of_grant(reading, rid: str, t_grant_s: float, key: Optional[int] = None):
    """Review A2: ``(rest, free)`` of ``rid`` over the ranks whose record is of the grant at ``t_grant_s`` (first extend at or
    after it); None = no such rank. A reading without per-rank values (a stub, a caller's tuple) uses the merged entry.
    ``key``: read that key instead of ``rid_key(rid)`` (NF-RC31-R6REST: the halt record)."""
    v = reading.get(rid_key(rid) if key is None else int(key)) if reading else None
    if v is None:
        return None
    t0 = int(float(t_grant_s) * 1e9)
    ranks = getattr(v, "ranks", None)
    if ranks:
        mine = [r for r in ranks if int(r[0]) >= t0]
        if not mine:
            return None
        return max(int(r[1]) for r in mine), _merge_free(mine)
    if not isinstance(v, tuple):
        return None if int(v) < t0 else (-1, -1)
    if int(v[0]) < t0:
        return None
    return int(v[1]), (int(v[2]) if len(v) >= 3 else -1)


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


def free_of(reading: Optional[Dict[int, object]], rid: str, t_grant_s: float) -> Optional[int]:
    """NF-RC30-R6STALE (v4, SOLL R6 + R7): the tokens of ``rid`` NO forward of D has been launched for -- what an R6 redirect can
    still move to P -- when D began it at or after its grant. A reading without ``free`` (v1-v3 file, a caller's 2-tuple stub, a
    rank that did not report it) falls back to the rest (``rest_of``, the pre-v4 figure); None = not started / no rest."""
    rest = rest_of(reading, rid, t_grant_s)
    if rest is None:
        return None
    g = _of_grant(reading, rid, t_grant_s)
    free = -1 if g is None else int(g[1])
    if free == COMMITTED:
        return COMMITTED  # review A1: the next chunk is fixed on D, its exact free follows with its start record
    if free >= 0:
        return min(free, int(rest))
    return int(rest)


def started(reading: Optional[Dict[int, object]], rid: str, t_grant_s: float) -> bool:
    """D began the extend of ``rid`` at or after its grant at ``t_grant_s`` (epoch s; same host clock)."""
    e = _entry(reading, rid)
    return e is not None and e[0] >= int(float(t_grant_s) * 1e9)


def rest_of(reading: Optional[Dict[int, object]], rid: str, t_grant_s: float) -> Optional[int]:
    """S56: D's REAL rest of ``rid`` (tokens not yet computed, as the newest extend batch saw it) when D began it at or
    after its grant; None = not started, or started without a reported rest."""
    g = _of_grant(reading, rid, t_grant_s)
    if g is None or g[0] < 0:
        return None
    return int(g[0])


def halted_free(reading: Optional[Dict[int, object]], rid: str, t_grant_s: float) -> Optional[int]:
    """NF-RC31-R6REST (SOLL R6 + R7, R11; user 19.10. "NICHTS SCHÄTZEN, EXAKTE ZAHLEN"): the tokens of ``rid`` no forward of D was
    launched for when D APPLIED its abort -- the exact number an R6 redirect moves to P -- from the halt record (``note_halt``) of
    the grant at ``t_grant_s``; None = no rank has applied the abort of this grant yet (D may still launch a chunk). The abort
    reaches every TP rank in the same recv step and chunk boundaries are group-agreed (#794), so the ranks that halted agree;
    their merge is ``_merge_free`` (MIN of the ranks with the most launches)."""
    g = _of_grant(reading, rid, t_grant_s, key=halt_key(rid))
    if g is None:
        return None
    # Review NF-RC31-R6REST A3 (SOLL R11, rank unanimity): the halt counts only once EVERY live rank that holds this grant's
    # extend record has applied the abort. A rank that has not may still launch a chunk (its free would be smaller); reading the
    # first rank alone would rest the verdict on one rank's state. Until then: None -> the verdict holds (liveness guard
    # ``Front.R6_HALT_STALE_S`` names a rank that never halts). A reading without per-rank values (a stub) keeps the merged entry.
    n_ext = _n_ranks_of_grant(reading, rid_key(rid), t_grant_s)
    n_halt = _n_ranks_of_grant(reading, halt_key(rid), t_grant_s)
    if n_ext is not None and n_halt is not None and n_halt < n_ext:
        return None
    free = int(g[1])
    if free < 0:
        return None if int(g[0]) < 0 else int(g[0])
    return free


def _n_ranks_of_grant(reading, key: int, t_grant_s: float) -> Optional[int]:
    """Review A3: how many ranks carry a record under ``key`` of the grant at ``t_grant_s``; None = the reading has no per-rank
    values (a stub, a caller's tuple) -- the count is not known."""
    v = reading.get(int(key)) if reading else None
    if v is None:
        return 0
    ranks = getattr(v, "ranks", None)
    if not ranks:
        return None
    t0 = int(float(t_grant_s) * 1e9)
    return sum(1 for r in ranks if int(r[0]) >= t0)


def halt_capable(reading: Optional[Dict[int, object]]) -> bool:
    """NF-RC31-R6REST: a live rank of the group announced halt records (``DExtendBeacon.announce_halt``)."""
    return bool(reading) and HALT_CAP_KEY in reading
