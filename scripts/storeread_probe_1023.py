#!/usr/bin/env python3
"""Store-read micro probe (27b-storeread-1023).

Reads a COPY of a page group of the real L3 store (one 32 KiB file per KV
page, 256 hash shards) the way the server's L3->L2 fill does, and the way the
fixed path does, under the contention the boot had:

* ``--gil-hog K``   K pure-Python busy threads (the scheduler thread / any
                    CPU-bound Python thread of the rank process)
* ``--evict-storm`` an in-process thread that does what the background L3
                    evictor did on boot 15 (408438 files in 22 min, one
                    logger.info each): stat + unlink + log line, at full speed,
                    on files it creates itself in a scratch dir of the SAME
                    filesystem (never in the store)

Every run starts cold: posix_fadvise(DONTNEED) over the group first.
Modes:
  server : per-page Python os.stat (``_existing_path``) + 16 pool threads,
           strided groups, 16-page ctypes sub-calls, touch=1 (futimens)
  fixed  : no per-page Python stat (paths from the batched C stat), ONE
           ctypes call that reads the whole batch on C threads
           (``hicache_read_pages_mt``), the caller touches progress <= 10/s
Only stdlib + the worktree's pageio.c (loaded by path, no flliper import).
"""
from __future__ import annotations

import argparse
import importlib.util
import logging
import os
import random
import threading
import time
from concurrent.futures import ThreadPoolExecutor

HERE = os.path.dirname(os.path.abspath(__file__))
PAGEIO_PY = os.path.join(HERE, "..", "python", "flliper", "srt", "mem_cache", "storage", "file", "pageio.py")


def load_pageio():
    spec = importlib.util.spec_from_file_location("pageio_probe", PAGEIO_PY)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    pio = mod.load()
    assert pio is not None, "pageio helper did not load"
    return pio


def group_files(root, n):
    out = []
    for sh in sorted(os.listdir(root)):
        d = os.path.join(root, sh)
        if not os.path.isdir(d):
            continue
        out += [os.path.join(d, x) for x in sorted(os.listdir(d))]
    random.Random(7).shuffle(out)
    return out[:n]


def drop_cache(paths):
    for p in paths:
        try:
            fd = os.open(p, os.O_RDONLY)
            try:
                os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
            finally:
                os.close(fd)
        except OSError:
            pass


def gil_hog(stop):
    x = 0
    while not stop.is_set():
        for i in range(2000):
            x += i * i


def evict_storm(stop, scratch, counter):
    log = logging.getLogger("storm")
    os.makedirs(scratch, exist_ok=True)
    blob = b"\0" * 32768
    k = 0
    # pre-create a pool so the storm is mostly unlink (what the evictor does)
    pool = []
    for i in range(4096):
        p = os.path.join(scratch, f"{i:06d}.bin")
        with open(p, "wb") as f:
            f.write(blob)
        pool.append(p)
    while not stop.is_set():
        p = pool[k % len(pool)]
        try:
            st = os.stat(p).st_mtime
            os.remove(p)
            log.info("L3-EVICT-FILE stem=%s comp=kv bytes=%d age_s=%.0f why=cap_bg run=%d n=%d", p, 32768,
                     time.time() - st, k, 1)
            with open(p, "wb") as f:   # recreate so the storm can go on (a write the real evictor does not do)
                f.write(blob)
        except OSError:
            pass
        k += 1
        counter[0] = k


def run_server(pio, paths, threads=16, step=16, touch=True, py_stat=True):
    ptrs_buf = bytearray(32768 * len(paths))
    import ctypes
    base = ctypes.addressof((ctypes.c_char * len(ptrs_buf)).from_buffer(ptrs_buf))
    ptrs = [base + i * 32768 for i in range(len(paths))]
    t0 = time.perf_counter()
    if py_stat:
        for p in paths:            # _existing_path -> _stat_stem: one Python os.stat per page
            try:
                os.stat(p)
            except OSError:
                pass
    t1 = time.perf_counter()
    n = len(paths)
    ext = ((0, 32768),)
    groups = [list(range(g, n, threads)) for g in range(threads)]
    last = [0.0]

    def progress():
        now = time.monotonic()
        if now - last[0] >= 0.1:
            last[0] = now

    def one(idx):
        out = []
        for a in range(0, len(idx), step):
            sub = idx[a:a + step]
            out += list(pio.read_pages([paths[i] for i in sub], [32768] * len(sub), [ext] * len(sub),
                                       [ptrs[i] for i in sub], touch))
            progress()
        return out

    with ThreadPoolExecutor(max_workers=threads) as ex:
        rc = [r for res in ex.map(one, groups) for r in res]
    t2 = time.perf_counter()
    return {"ok": sum(1 for r in rc if r == 0), "stat_s": t1 - t0, "read_s": t2 - t1, "total_s": t2 - t0}


def run_fixed(pio, paths, threads=16, touch=True):
    import ctypes
    buf = bytearray(32768 * len(paths))
    base = ctypes.addressof((ctypes.c_char * len(buf)).from_buffer(buf))
    ptrs = [base + i * 32768 for i in range(len(paths))]
    t0 = time.perf_counter()
    rc = pio.read_pages_mt(paths, [32768] * len(paths), [((0, 32768),)] * len(paths), ptrs, touch, threads,
                           progress=lambda: None)
    t2 = time.perf_counter()
    return {"ok": sum(1 for r in rc if r == 0), "stat_s": 0.0, "read_s": t2 - t0, "total_s": t2 - t0}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="/spinning/docker-acceptance/27b/store/_storeread_probe_1023/group")
    ap.add_argument("--pages", type=int, default=8192)
    ap.add_argument("--mode", default="server", choices=["server", "server-nostat", "server-notouch", "fixed",
                                                          "fixed-notouch"])
    ap.add_argument("--gil-hog", type=int, default=0)
    ap.add_argument("--evict-storm", action="store_true")
    ap.add_argument("--warm", action="store_true", help="do not drop the page cache first")
    ap.add_argument("--threads", type=int, default=16)
    a = ap.parse_args()
    logging.basicConfig(filename="/dev/null", level=logging.INFO)
    pio = load_pageio()
    paths = group_files(a.root, a.pages)
    if not a.warm:
        drop_cache(paths)
    stop = threading.Event()
    bg = []
    storm_n = [0]
    for _ in range(a.gil_hog):
        bg.append(threading.Thread(target=gil_hog, args=(stop,), daemon=True))
    if a.evict_storm:
        bg.append(threading.Thread(target=evict_storm, args=(stop, os.path.join(os.path.dirname(a.root), "storm"),
                                                             storm_n), daemon=True))
    for t in bg:
        t.start()
    time.sleep(0.5 if bg else 0.0)
    s0 = storm_n[0]
    t_s = time.perf_counter()
    if a.mode == "server":
        r = run_server(pio, paths, a.threads)
    elif a.mode == "server-nostat":
        r = run_server(pio, paths, a.threads, py_stat=False)
    elif a.mode == "server-notouch":
        r = run_server(pio, paths, a.threads, touch=False)
    elif a.mode == "fixed":
        r = run_fixed(pio, paths, a.threads)
    else:
        r = run_fixed(pio, paths, a.threads, touch=False)
    el = time.perf_counter() - t_s
    storm_rate = (storm_n[0] - s0) / max(1e-6, el)
    stop.set()
    pps = r["ok"] / max(1e-9, r["total_s"])
    print(f"mode={a.mode} pages={len(paths)} ok={r['ok']} cold={not a.warm} gil_hog={a.gil_hog} "
          f"evict_storm={a.evict_storm}(unlinks/s={storm_rate:.0f}) stat_s={r['stat_s']:.3f} read_s={r['read_s']:.3f} "
          f"total_s={r['total_s']:.3f} pages_per_s={pps:.0f} MB_per_s={pps * 32768 / 1e6:.1f}")


if __name__ == "__main__":
    main()
