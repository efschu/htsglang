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
  - kind 2 MAP: (re)map slot ``slot`` at ``arg`` bytes (the owner has already
    sized the memfd with ftruncate).
  - kind 3 QUIT.
* reply to every request: ``<QiId`` (seq, status, rows, seconds); status 0 =
  ok, otherwise an errno-like code (-1 = short read).

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
KEY_SHIFT = 48
KEY_MASK = (1 << KEY_SHIFT) - 1


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
        if kind == KIND_MAP:
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
            step = max(256, (n + threads - 1) // threads)
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
