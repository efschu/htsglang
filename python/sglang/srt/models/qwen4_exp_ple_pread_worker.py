"""fnFL2 H32: the PLE pread gather's worker PROCESS (stdlib only, run as a script).

Started by :class:`sglang.srt.models.qwen4_exp_ple_prefetch.PlePreadProcs` as
``python -I -S <this file>`` -- no site-packages, no torch, no sglang import --
so its RSS is the interpreter's and its GIL is its own. Why a process and not a
thread: the pread gather is one Python ``preadv`` per row, and 32 threads of it
in the scheduler process starve the scheduler thread of the GIL. Measured on
this rig (desk, 24.09., 262144 rows x 320 B, ARC-warm, 32 threads): a torch-op
loop of 20000 small ops takes 0.054 s alone and 1.495 s while the threaded
gather runs beside it (x28); with the same gather in a child process it takes
0.060 s. A gather overlapped with the forward therefore has to leave the
process.

Protocol (little endian, over stdin/stdout pipes):

* start: ``<I`` length + JSON ``{"files": [...], "slot_fds": [...],
  "row_bytes": int, "threads": int, "delay_s": float}``; reply ``<I`` 0x504C4531.
* request: ``<IQIIQ`` (kind, seq, slot, n, arg) + payload.
  - kind 1 GATHER: payload = n x int64 destination rows, then n x int64 keys;
    key = (file index << 48) | byte offset, key < 0 = a zero row. Every row is
    ``row_bytes`` bytes, written at ``dest * row_bytes`` of slot ``slot``.
    ``arg`` > 0 = rows per thread task (fnFL2 H40), 0 = the default split.
  - kind 2 MAP: (re)map slot ``slot`` at ``arg`` bytes (the owner has already
    sized the memfd with ftruncate).
  - kind 3 QUIT.
  - kind 4 AUTO (fnFL2 H73): payload = ``n`` bytes of JSON, this worker's
    share of the decode stage in slot ``slot`` that it fills ON ITS OWN
    (:class:`AutoStage`): layout offsets, its index among the workers, the
    n-gram hash constants and the shard -> (file, offset) map.
  - kind 5 ROUND (fnFL2 H73): "the device posts round ``seq`` into the
    mailbox of part ``arg >> 32`` (0 = the verify round, 1 = the next round's
    bonus rows)"; ``n`` = requests, ``arg & 0xFFFF`` = tokens per request in
    the mailbox, ``(arg >> 16) & 0xFFFF`` = the verify width (bonus part).
    NO reply: the owner never waits on a round.
* reply to every request but ROUND: ``<QiId`` (seq, status, rows, seconds);
  status 0 = ok, otherwise an errno-like code (-1 = short read).

EOF on stdin (the owner died) ends the process.
"""

import json
import mmap
import os
import struct
import sys
import time
from array import array
from concurrent.futures import ThreadPoolExecutor

REQ = struct.Struct("<IQIIQ")
REP = struct.Struct("<QiId")
HELLO = 0x504C4531
KIND_GATHER = 1
KIND_MAP = 2
KIND_QUIT = 3
#: fnFL2 H73: configure / announce the autonomous decode stage (AutoStage)
KIND_AUTO = 4
KIND_ROUND = 5
PART_VERIFY = 0
PART_BONUS = 1
KEY_SHIFT = 48
KEY_MASK = (1 << KEY_SHIFT) - 1
_MASK64 = (1 << 64) - 1
_SIGN64 = 1 << 63
#: fnFL2 H73: int64 words per worker in the stage's stats block, and their meaning
STATS_WORDS = 16
(ST_LAST, ST_ROUNDS, ST_READ, ST_STAGE_NS, ST_STAGE_MAX_NS, ST_POLL_NS, ST_REUSED,
 ST_MISSED, ST_ERRORS, ST_BONUS_ROUNDS, ST_BONUS_READ) = range(11)
_PR_SET_TIMERSLACK = 29


def _read_exact(fd, n):
    parts = []
    while n:
        b = os.read(fd, min(n, 1 << 20))
        if not b:
            return None
        parts.append(b)
        n -= len(b)
    return b"".join(parts)


def _write_all(fd, data):
    view = memoryview(data)
    while view:
        k = os.write(fd, view)
        view = view[k:]


def _read_rows(buf, rb, fds, dest, keys, lo, hi, zero):
    for j in range(lo, hi):
        k = keys[j]
        d = dest[j] * rb
        if k < 0:
            buf[d : d + rb] = zero
            continue
        got = os.preadv(fds[k >> KEY_SHIFT], [buf[d : d + rb]], k & KEY_MASK)
        if got != rb:
            return -1
    return 0


def _wrap64(x):
    """int64 two's-complement wrap-around (torch's int64 multiply)."""
    x &= _MASK64
    return x - (1 << 64) if x & _SIGN64 else x


def row_ids(rows, mult, sizes, offsets, hpn, n, eos):
    """fnFL2 H73: ``ple_verify_row_ids`` of ``qwen4_exp_ple_decode_pread``, stdlib
    only (this process imports neither torch nor sglang): ``[history (n-1) |
    tokens]`` per request -> the gather's row ids, token-major, bigram heads
    then trigram heads per token. The desk tests hold it equal to the model's
    ``_hash_contexts``."""
    out = []
    for row in rows:
        for j in range(len(row) - n + 1):
            i = j + n - 1
            shifted = [row[i]]
            broken = False
            for s in range(1, n):
                v = row[i - s]
                if broken or v == eos:
                    broken = True
                    v = eos
                shifted.append(v)
            for ngram in range(2, n + 1):
                mix = _wrap64(shifted[0] * mult[0])
                for pos in range(1, ngram):
                    mix ^= _wrap64(shifted[pos] * mult[pos])
                start = (ngram - 2) * hpn
                for h in range(start, start + hpn):
                    out.append(mix % sizes[h] + offsets[h])
    return out


def _timer_slack_ns(ns):
    """Sleep granularity of this process (the default slack is 50 us, longer
    than the stage's poll interval). Best effort."""
    try:
        import ctypes

        ctypes.CDLL(None, use_errno=True).prctl(_PR_SET_TIMERSLACK, int(ns), 0, 0, 0)
    except (OSError, AttributeError):
        pass


class AutoStage:
    """fnFL2 H73: this worker's share of a decode stage the workers fill ON
    THEIR OWN -- no request per row from the owner, no reply.

    The device posts a round into a mailbox of the (shared, page-locked) stage:
    the token windows ``[history | tokens]`` of every request, then -- a
    second copy, ordered behind the first on the same stream -- the round's
    number into the part's ``flag`` word. The owner announces the round with a
    ROUND request (no payload); this worker polls ``flag`` until it reaches the
    round, hashes the windows (``row_ids``), and reads the rows of ITS stage
    positions (position ``r`` belongs to worker ``r % procs``, always the same
    one, so no two workers ever write one row) straight into the stage: id -1,
    bytes, id -- the id after its bytes, as H40. A position whose id already
    names the wanted row is kept (the bonus rows pre-staged by part 1, or the
    same n-gram again). Part 0 then publishes ``done[index] <- round``; the
    device's gate lets the gather read the stage once every worker did.

    Stats (int64 words at ``stats_off + 8 * STATS_WORDS * index``, read by the
    owner's proof line): last round, rounds, rows read, stage ns (flag seen ->
    done), max stage ns, poll ns (announced -> flag seen), rows kept, rounds
    missed (the flag did not reach the round in time, or a newer round had
    overwritten the mailbox), read errors, bonus rounds, bonus rows read.
    """

    def __init__(self, cfg, buf, rb, threads):
        n_bytes = len(buf)

        def q(off, count):
            off = int(off)
            if off < 0 or off % 8 or off + 8 * count > n_bytes:
                raise ValueError(f"auto stage field at {off} (+{count} words) outside the slot")
            return buf[off : off + 8 * count].cast("q")

        self._views = []
        self.p = int(cfg["index"])
        self.P = max(1, int(cfg["procs"]))
        if not 0 <= self.p < self.P:
            raise ValueError(f"auto stage index {self.p} of {self.P}")
        self.cap = int(cfg["capacity"])
        self.rb = int(rb)
        self.buf = buf
        self.threads = max(1, int(threads))
        self.ctx_cap = int(cfg["ctx_cap"])
        if self.cap * self.rb > int(cfg["ids_off"]):
            raise ValueError("auto stage rows overlap the ids")
        self.ids = q(cfg["ids_off"], self.cap)
        self.done = q(cfg["done_off"], self.P)
        self.flag = (q(cfg["flag_off"], 1), q(cfg["flag_bonus_off"], 1))
        self.ctx = (q(cfg["ctx_off"], self.ctx_cap), q(cfg["ctx_bonus_off"], self.ctx_cap))
        self.stats = q(int(cfg["stats_off"]) + 8 * STATS_WORDS * self.p, STATS_WORDS)
        self._views = [self.ids, self.done, *self.flag, *self.ctx, self.stats]
        h = cfg["hash"]
        self.mult = [int(x) for x in h["mult"]]
        self.sizes = [int(x) for x in h["sizes"]]
        self.offsets = [int(x) for x in h["offsets"]]
        self.hpn = int(h["hpn"])
        self.ngram = int(h["ngram"])
        self.eos = int(h["eos"])
        self.heads = (self.ngram - 1) * self.hpn
        self.lo, self.hi = int(cfg["vocab"][0]), int(cfg["vocab"][1])
        self.shard_rows = int(cfg["shard_rows"])
        self.shard_off = [int(x) for x in cfg["shard_offsets"]]
        self.shard_file = [int(x) for x in cfg["shard_file"]]
        if len(self.shard_off) != len(self.shard_file) or self.shard_rows <= 0:
            raise ValueError("auto stage shard map is inconsistent")
        self.poll_s = (float(cfg["poll_us"]) / 1e6, float(cfg["poll_bonus_us"]) / 1e6)
        self.deadline_ns = int(float(cfg["deadline_ms"]) * 1e6)
        self.seq = [0, 0]
        # own positions at or above this bound hold no id (hygiene only:
        # a stale (id, bytes) pair is still a true pair)
        self.high = 0

    def release(self):
        for v in self._views:
            v.release()
        self._views = []

    def _read_part(self, part, fds):
        buf, rb, sid = self.buf, self.rb, self.ids
        sr, soff, sfile = self.shard_rows, self.shard_off, self.shard_file
        for r, gid in part:
            s = gid // sr
            d = r * rb
            got = os.preadv(fds[sfile[s]], [buf[d : d + rb]], soff[s] + (gid - s * sr) * rb)
            if got != rb:
                return -1
            sid[r] = gid  # the id AFTER its bytes
        return 0

    def _read(self, todo, pool, fds):
        if not todo:
            return 0
        per = max(1, -(-len(todo) // self.threads))
        futs = [
            pool.submit(self._read_part, todo[a : a + per], fds)
            for a in range(0, len(todo), per)
        ]
        err = 0
        for f in futs:
            try:
                rc = f.result()
            except OSError as exc:
                rc = exc.errno or 5
            if rc and not err:
                err = rc
        return err

    def round(self, part, seq, bs, cols, width, pool, fds):
        if part not in (PART_VERIFY, PART_BONUS) or seq <= self.seq[part]:
            return
        st = self.stats
        t0 = time.monotonic_ns()
        flag = self.flag[part]
        end = t0 + self.deadline_ns
        poll = self.poll_s[part]
        f = flag[0]
        while f < seq and time.monotonic_ns() < end:
            time.sleep(poll)
            f = flag[0]
        k = bs * cols
        if f != seq or bs <= 0 or cols < self.ngram or k > self.ctx_cap:
            st[ST_MISSED] += 1  # never posted in time, overtaken, or a foreign shape
            return
        t_seen = time.monotonic_ns()
        vals = self.ctx[part][:k].tolist()
        if flag[0] != seq:  # the next round's post landed while this one was read
            st[ST_MISSED] += 1
            return
        ids = row_ids(
            [vals[i * cols : (i + 1) * cols] for i in range(bs)],
            self.mult, self.sizes, self.offsets, self.hpn, self.ngram, self.eos,
        )
        H, P, p, cap = self.heads, self.P, self.p, self.cap
        if part == PART_VERIFY:
            pos_ids = enumerate(ids)
        else:
            # the NEXT verify row starts with this bonus: position (i * width) * H + h
            pos_ids = (((i * width) * H + h, ids[i * H + h]) for i in range(bs) for h in range(H))
        sid, lo, hi = self.ids, self.lo, self.hi
        todo = []
        kept = 0
        top = 0
        for r, gid in pos_ids:
            if r % P != p or r >= cap:
                continue
            if r + 1 > top:
                top = r + 1
            if lo <= gid < hi:
                if sid[r] == gid:
                    kept += 1
                else:
                    sid[r] = -1
                    todo.append((r, gid))
            else:
                sid[r] = -1
        if part == PART_VERIFY:
            n = min(len(ids), cap)
            for r in range(n + (p - n) % P, self.high, P):
                sid[r] = -1
            self.high = n
        elif top > self.high:
            self.high = top
        err = self._read(todo, pool, fds)
        self.seq[part] = seq
        if err:
            st[ST_ERRORS] += 1
        st[ST_REUSED] += kept
        if part == PART_BONUS:
            st[ST_BONUS_ROUNDS] += 1
            st[ST_BONUS_READ] += len(todo)
            return
        self.done[p] = seq  # this worker's rows of round ``seq`` are in
        t1 = time.monotonic_ns()
        st[ST_LAST] = seq
        st[ST_ROUNDS] += 1
        st[ST_READ] += len(todo)
        st[ST_STAGE_NS] += t1 - t_seen
        if t1 - t_seen > st[ST_STAGE_MAX_NS]:
            st[ST_STAGE_MAX_NS] = t1 - t_seen
        st[ST_POLL_NS] += t_seen - t0


def main():
    fin = sys.stdin.fileno()
    fout = sys.stdout.fileno()
    raw = _read_exact(fin, 4)
    if raw is None:
        return 0
    cfg = json.loads(_read_exact(fin, struct.unpack("<I", raw)[0]))
    rb = int(cfg["row_bytes"])
    fds = [os.open(p, os.O_RDONLY) for p in cfg["files"]]
    slot_fds = [int(f) for f in cfg["slot_fds"]]
    threads = max(1, int(cfg.get("threads", 8)))
    delay_s = float(cfg.get("delay_s", 0.0))
    maps = [None] * len(slot_fds)
    views = [None] * len(slot_fds)
    autos = [None] * len(slot_fds)  # fnFL2 H73
    zero = bytes(rb)
    pool = ThreadPoolExecutor(max_workers=threads)
    _write_all(fout, struct.pack("<I", HELLO))
    while True:
        head = _read_exact(fin, REQ.size)
        if head is None:
            return 0
        kind, seq, slot, n, arg = REQ.unpack(head)
        if kind == KIND_QUIT:
            _write_all(fout, REP.pack(seq, 0, 0, 0.0))
            return 0
        if kind == KIND_ROUND:
            # fnFL2 H73: no reply -- the owner never waits on a round
            if 0 <= slot < len(autos) and autos[slot] is not None:
                autos[slot].round(
                    (arg >> 32) & 0xFF, seq, n, arg & 0xFFFF, (arg >> 16) & 0xFFFF, pool, fds
                )
            continue
        if kind == KIND_AUTO:
            raw = _read_exact(fin, n) if n else b""
            if raw is None:
                return 0
            status = 0
            try:
                if autos[slot] is not None:
                    autos[slot].release()
                    autos[slot] = None
                if views[slot] is None:
                    raise ValueError("auto stage on an unmapped slot")
                autos[slot] = AutoStage(json.loads(raw), views[slot], rb, threads)
                _timer_slack_ns(1000)
            except (ValueError, KeyError, TypeError, IndexError):
                status = 22  # EINVAL
            _write_all(fout, REP.pack(seq, status, 0, 0.0))
            continue
        if kind == KIND_MAP:
            if autos[slot] is not None:
                # fnFL2 H73: its views pin the old mapping
                autos[slot].release()
                autos[slot] = None
            if views[slot] is not None:
                views[slot].release()
                maps[slot].close()
                views[slot] = maps[slot] = None
            if arg:
                maps[slot] = mmap.mmap(slot_fds[slot], int(arg), mmap.MAP_SHARED,
                                       mmap.PROT_READ | mmap.PROT_WRITE)
                views[slot] = memoryview(maps[slot])
            _write_all(fout, REP.pack(seq, 0, 0, 0.0))
            continue
        payload = _read_exact(fin, 16 * n) if n else b""
        if payload is None:
            return 0
        t0 = time.monotonic()
        dest = array("q")
        dest.frombytes(payload[: 8 * n])
        keys = array("q")
        keys.frombytes(payload[8 * n :])
        status = 0
        buf = views[slot]
        if n and (buf is None or (max(dest) + 1) * rb > len(buf)):
            status = 22  # EINVAL: slot not mapped or too small
        elif n:
            # fnFL2 H40: ``arg`` > 0 is the rows per thread task -- a
            # decode-sized gather (tens of rows) spreads over the threads
            # instead of landing whole on one of them
            step = int(arg) if arg else max(256, (n + threads - 1) // threads)
            futs = [
                pool.submit(_read_rows, buf, rb, fds, dest, keys, lo, min(n, lo + step), zero)
                for lo in range(0, n, step)
            ]
            for f in futs:
                try:
                    rc = f.result()
                except OSError as exc:
                    rc = exc.errno or 5
                if rc and not status:
                    status = rc
        if delay_s:
            # desk tests only: a gather that takes a known time
            time.sleep(delay_s)
        _write_all(fout, REP.pack(seq, status, n, time.monotonic() - t0))


if __name__ == "__main__":
    sys.exit(main())
