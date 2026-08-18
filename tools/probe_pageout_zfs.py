"""#738 metal-truth probe (reusable; ~2 minutes, no GPU, no serving).

VERDICT IT PRODUCED 2026-08-18 on this pool (OpenZFS, 4 GiB urandom file,
mincore): A touched mmap 100.0%; B fresh-mapping MADV_PAGEOUT rc=0 and
residency STILL 100.0% (accepted-and-did-nothing -- the silent-success
shape); C the ladder's live-mapping call 100.0% -> 0.0%; D read_file_direct
4096 MiB with 0.0% -> 0.0% residency; E buffered read() landed in ARC, not
page cache (0.0% resident before fadvise) -- so the fadvise no-op claim is
untestable this way, and the 99G load peak was the MMAP path, not buffered
reads. Re-run me whenever "is the drop flag inert" comes up again.

Original module docstring:

#738 metal-truth probe: does the in-tree ladder actually work on THIS
ZFS pool, today? Drives the REAL functions (drop_page_cache_range,
read_file_direct), measures residency via mincore. No GPU, no serving.

Phases:
  A  buffered read + touched mmap        -> baseline residency (~100%)
  B  MADV_PAGEOUT via a FRESH mapping    -> in-tree claim: near no-op
  C  MADV_PAGEOUT via the LIVE mapping   -> in-tree claim: ~full drop
  D  read_file_direct (O_DIRECT)         -> in-tree claim: ~zero re-cache
  E  plain fadvise(DONTNEED)             -> in-tree claim: no-op on ZFS
"""

import ctypes
import mmap
import os
import sys

# Run from a repo checkout: PYTHONPATH=<repo>/python python tools/probe_pageout_zfs.py
_here = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_here, "..", "python"))

SIZE = 4 << 30  # 4 GiB
PATH = os.environ.get("PROBE738_PATH", "/spinning/probe738.bin")
PAGE = os.sysconf("SC_PAGESIZE")

libc = ctypes.CDLL("libc.so.6", use_errno=True)


def residency(path: str) -> float:
    """Fraction of the file's pages resident in page cache (mincore)."""
    size = os.path.getsize(path)
    npages = (size + PAGE - 1) // PAGE
    with open(path, "rb") as f:
        mm = mmap.mmap(f.fileno(), size, prot=mmap.PROT_READ | mmap.PROT_WRITE, flags=mmap.MAP_PRIVATE)
        try:
            buf = ctypes.addressof(ctypes.c_char.from_buffer(mm))
            vec = (ctypes.c_ubyte * npages)()
            rc = libc.mincore(
                ctypes.c_void_p(buf), ctypes.c_size_t(size), vec
            )
            if rc != 0:
                raise OSError(ctypes.get_errno(), "mincore failed")
            resident = sum(1 for b in vec if b & 1)
        finally:
            del vec
            mm.close()
    return resident / npages


def main():
    if not os.path.exists(PATH) or os.path.getsize(PATH) != SIZE:
        print("writing 4 GiB of urandom (defeats ZFS compression)...")
        with open("/dev/urandom", "rb") as src, open(PATH, "wb") as dst:
            left = SIZE
            while left:
                chunk = src.read(64 << 20)
                dst.write(chunk[: min(len(chunk), left)])
                left -= min(len(chunk), left)
        os.system("sync")

    from sglang.srt.model_loader.gguf_shards import (
        _page_cache_advice_available,
        drop_page_cache_range,
    )
    from sglang.srt.model_loader.weight_utils import read_file_direct

    print("advice available:", _page_cache_advice_available())

    # Phase A: fault everything in through a LIVE mapping and keep it.
    f = open(PATH, "rb")
    live = mmap.mmap(f.fileno(), SIZE, prot=mmap.PROT_READ | mmap.PROT_WRITE, flags=mmap.MAP_PRIVATE)
    live.madvise(mmap.MADV_SEQUENTIAL)
    touch = 0
    for off in range(0, SIZE, PAGE):
        touch += live[off]
    live_addr = ctypes.addressof(ctypes.c_char.from_buffer(live))
    ra = residency(PATH)
    print(f"A baseline after touched mmap: residency={ra:.1%}")

    # Phase B: PAGEOUT through a FRESH mapping while the live one stands.
    with open(PATH, "rb") as f2:
        fresh = mmap.mmap(f2.fileno(), SIZE, prot=mmap.PROT_READ | mmap.PROT_WRITE, flags=mmap.MAP_PRIVATE)
        fresh_addr = ctypes.addressof(ctypes.c_char.from_buffer(fresh))
        MADV_PAGEOUT = 21
        rc = libc.madvise(
            ctypes.c_void_p(fresh_addr), ctypes.c_size_t(SIZE), MADV_PAGEOUT
        )
        print(f"B fresh-mapping MADV_PAGEOUT rc={rc}")
        fresh.close()
    rb = residency(PATH)
    print(f"B residency after fresh-mapping PAGEOUT: {rb:.1%} (claim: near no-op)")

    # Phase C: the ladder's own call with the LIVE mapping's address.
    drop_page_cache_range(PATH, 0, SIZE, address=live_addr, map_len=SIZE)
    rc_res = residency(PATH)
    print(f"C residency after LIVE-mapping ladder drop: {rc_res:.1%} (claim: ~0)")
    live.close()
    f.close()

    # Phase D: O_DIRECT read of the (now uncached) file.
    before = residency(PATH)
    data = read_file_direct(PATH)
    n = len(data)
    del data
    after = residency(PATH)
    print(
        f"D O_DIRECT read of {n >> 20} MiB: residency {before:.1%} -> "
        f"{after:.1%} (claim: ~zero growth)"
    )

    # Phase E: re-fault a slice buffered, then plain fadvise(DONTNEED).
    with open(PATH, "rb") as f3:
        f3.read(1 << 30)  # 1 GiB buffered
    re1 = residency(PATH)
    fd = os.open(PATH, os.O_RDONLY)
    os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
    os.close(fd)
    re2 = residency(PATH)
    print(f"E buffered 1GiB re-cache: {re1:.1%}; after fadvise DONTNEED: {re2:.1%} (claim: no-op)")


if __name__ == "__main__":
    main()
