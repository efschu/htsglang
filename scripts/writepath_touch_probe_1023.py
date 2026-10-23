"""WAKEEXT-1023 micro-measurement: the write path's recency touch of pages already on disk
(``HiCacheFile._touch_present``), per-page Python stat + utime (FLLIPER_HICACHE_L3_WRITE_NOSTAT=0) against
sharded path + one C utime call (default), with and without one busy Python thread in the process.

usage: python3 scripts/writepath_touch_probe_1023.py <dir on the store's filesystem> [pages]
"""
import os
import sys
import threading
import time
import types

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "python"))

from flliper.srt.mem_cache.hicache_storage import HiCacheFile  # noqa: E402


class _Ev:
    enabled = True

    def touch(self, stem, path, mtime=True):
        if mtime:
            try:
                os.utime(path, None)
            except OSError:
                pass


def _backend(root):
    be = object.__new__(HiCacheFile)
    be.file_path = root
    be._known_shards = set()
    be._legacy_flat = False
    be._evictor = _Ev()
    return be


def _busy(stop):
    x = 0
    while not stop.is_set():
        x += sum(range(200))


def _run(be, stems, mode, busy, swi):
    os.environ["FLLIPER_HICACHE_L3_WRITE_NOSTAT"] = "1" if mode == "new" else "0"
    sys.setswitchinterval(swi)
    stop = threading.Event()
    th = threading.Thread(target=_busy, args=(stop,), daemon=True) if busy else None
    if th:
        th.start()
        time.sleep(0.05)
    t0 = time.perf_counter()
    for i in range(0, len(stems), 512):          # backup batches of 512 stems
        be._touch_present(stems[i:i + 512])
    dt = time.perf_counter() - t0
    if th:
        stop.set()
        th.join()
    return len(stems) / dt


def main():
    root = sys.argv[1]
    n = int(sys.argv[2]) if len(sys.argv) > 2 else 8192
    be = _backend(root)
    stems = ["%064x_probe1023" % i for i in range(n)]
    for st in stems:
        p = be._sharded_path(st)
        os.makedirs(os.path.dirname(p), exist_ok=True)
        if not os.path.exists(p):
            with open(p, "wb") as f:
                f.write(b"\0" * 32768)
    print("pages=%d dir=%s" % (n, root))
    for busy, swi in ((False, 0.005), (True, 0.005), (True, 0.0002)):
        for mode in ("old", "new"):
            r = _run(be, stems, mode, busy, swi)
            print("busy_python_thread=%d switchinterval_ms=%.1f mode=%s pages_per_s=%.0f" % (busy, swi * 1e3, mode, r))


if __name__ == "__main__":
    main()
