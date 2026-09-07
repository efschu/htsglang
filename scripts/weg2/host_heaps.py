#!/usr/bin/env python3
"""Host-RAM attribution per process from /proc/<pid>/smaps (record 1h item 5).

Two modes, both read-only, both print their denominator:

  host_heaps.py table PID [PID ...]
      one row per pid: Rss split by mapping class -- [heap] (glibc brk),
      anon-private (python/torch CPU tensors, arenas), anon-shared/shmem
      (the torch_memory_saver cpu backup = cudaMallocHost pinned pages, the
      hicache host pools, /dev/shm rings), nvidia (/dev/nvidia* mappings =
      CUDA context + host-mapped device pages), file-backed (libraries,
      mmapped checkpoints), other.  Units MiB.  Denominator: Rss of every
      mapping in smaps at the sampling instant, so the classes sum to Rss.

  host_heaps.py watch SECONDS INTERVAL OUT.csv PID [PID ...]
      RssAnon/RssFile/RssShmem from /proc/<pid>/status every INTERVAL s
      for SECONDS s into OUT.csv (bounded; exits by itself).
"""
import os, sys, time


def classify(path: str, flags: str) -> str:
    if path == "[heap]":
        return "heap"
    if path.startswith("/dev/nvidia") or path.startswith("/dev/dri"):
        return "nvidia"
    if path.startswith("/dev/shm") or path.startswith("/memfd") or "SYSV" in path or path.startswith("/run/shm"):
        return "shmem"
    if path.startswith("[stack") or path == "[vvar]" or path == "[vdso]" or path == "[vsyscall]":
        return "stack/vdso"
    if path == "" or path == "[anon]" or path.startswith("[anon:"):
        return "anon-shared" if "s" in flags else "anon-private"
    if path.startswith("/"):
        return "file"
    return "other"


def table(pids):
    print("pid       cmd                          rss_MiB    heap  anon-priv  anon-shared   shmem  nvidia    file   other  | RssShmem(status)")
    for pid in pids:
        cls = {}
        rss_total = 0
        try:
            with open(f"/proc/{pid}/smaps") as f:
                cur = None
                for ln in f:
                    parts = ln.split()
                    if len(parts) >= 5 and "-" in parts[0] and len(parts[1]) == 4:
                        path = parts[5] if len(parts) > 5 else ""
                        cur = classify(path, parts[1])
                    elif ln.startswith("Rss:") and cur is not None:
                        kb = int(parts[1])
                        cls[cur] = cls.get(cur, 0) + kb
                        rss_total += kb
            cmd = open(f"/proc/{pid}/cmdline", "rb").read().replace(b"\0", b" ").decode(errors="replace")[:28]
            shm = 0
            for ln in open(f"/proc/{pid}/status"):
                if ln.startswith("RssShmem:"):
                    shm = int(ln.split()[1])
        except OSError as e:
            print(f"{pid:<9} unreadable: {e}")
            continue
        m = lambda k: cls.get(k, 0) / 1024
        print(f"{pid:<9} {cmd:<28} {rss_total/1024:8.0f} {m('heap'):7.0f} {m('anon-private'):10.0f} {m('anon-shared'):12.0f} {m('shmem'):7.0f} {m('nvidia'):7.0f} {m('file'):7.0f} {m('other')+m('stack/vdso'):7.0f}  | {shm/1024:.0f}")
    print("denominator: Rss of every smaps mapping at the sampling instant (classes sum to rss_MiB); MiB")


def watch(seconds, interval, out, pids):
    t_end = time.time() + seconds
    with open(out, "w") as f:
        f.write("ts_utc,pid,rss_anon_mib,rss_file_mib,rss_shmem_mib\n")
        while time.time() < t_end:
            ts = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
            for pid in pids:
                try:
                    v = {}
                    for ln in open(f"/proc/{pid}/status"):
                        if ln.startswith(("RssAnon:", "RssFile:", "RssShmem:")):
                            v[ln.split(":")[0]] = int(ln.split()[1]) / 1024
                    f.write(f"{ts},{pid},{v.get('RssAnon', -1):.0f},{v.get('RssFile', -1):.0f},{v.get('RssShmem', -1):.0f}\n")
                except OSError:
                    f.write(f"{ts},{pid},-1,-1,-1\n")
            f.flush()
            time.sleep(interval)


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print(__doc__); sys.exit(2)
    if sys.argv[1] == "table":
        table([int(p) for p in sys.argv[2:]])
    elif sys.argv[1] == "watch":
        watch(float(sys.argv[2]), float(sys.argv[3]), sys.argv[4], [int(p) for p in sys.argv[5:]])
    else:
        print(__doc__); sys.exit(2)
