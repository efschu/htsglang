#!/usr/bin/env python3
"""Where does the pinned expert pool land in cgroup accounting? (#391)

WHY THIS EXISTS
---------------------------------------------------------------------------
Boot 10's staging ledger reported 44.33 GiB of pinned host pool across the
three ranks -- a figure the PREP model predicted independently to 0.36% -- while
the cgroup's `anon` never exceeded 16.3 GiB and `unevictable` read 0.00. The
guard rammon.sh fires on is `memory.current`, so "which term of memory.current
holds the pinned pool, if any" decides whether the guard counts the feature's
own memory once, twice, or not at all -- and therefore whether the ~59 GiB
end-of-load ceiling the recipe expects is even the right number.

The log correlation in the runbook (4.5.6) already answers it operationally.
This probe answers it DIRECTLY, and it is written to be run in the next card
window: allocating pinned host memory needs a CUDA context, which needs a card
for a few seconds. Without one it refuses instead of guessing.

    python3 scripts/dsv4/pinned_pool_accounting.py --gib 4          # needs a card
    python3 scripts/dsv4/pinned_pool_accounting.py --mlock-control  # no card

The `--mlock-control` arm allocates ordinary anonymous memory and mlock()s it.
That is the OTHER way host memory can become unevictable, and it is the control
the pinned arm is read against: mlocked anon shows up as `anon` AND as
`unevictable`, which is exactly the signature boot 10 did NOT have.
"""

import argparse
import ctypes
import os
import sys

CG = os.environ.get("DSV4_CGROUP_ROOT", "/sys/fs/cgroup")
GIB = 1 << 30
TERMS = (
    "anon",
    "file",
    "kernel",
    "slab",
    "shmem",
    "file_mapped",
    "unevictable",
    "pagetables",
)


def cgroup_sample():
    out = {}
    try:
        with open(os.path.join(CG, "memory.current")) as fh:
            out["memory.current"] = int(fh.read().strip())
    except OSError:
        out["memory.current"] = -1
    try:
        with open(os.path.join(CG, "memory.stat")) as fh:
            stat = dict(
                (line.split()[0], int(line.split()[1]))
                for line in fh
                if len(line.split()) == 2
            )
    except OSError:
        stat = {}
    for key in TERMS:
        out[key] = stat.get(key, -1)
    return out


def proc_sample():
    out = {}
    try:
        with open("/proc/self/status") as fh:
            for line in fh:
                key = line.split(":")[0]
                if key in ("VmRSS", "VmLck", "VmPin", "VmSize"):
                    out[key] = int(line.split()[1]) * 1024
    except OSError:
        pass
    return out


def smaps_for(address: int):
    """The /proc/self/smaps entry containing `address`, as a short dict."""
    entry = {}
    try:
        with open("/proc/self/smaps") as fh:
            keep = False
            for line in fh:
                if "-" in line.split()[0] and ":" not in line.split()[0]:
                    start, _, end = line.split()[0].partition("-")
                    keep = int(start, 16) <= address < int(end, 16)
                    if keep:
                        entry["mapping"] = line.strip()
                elif keep:
                    key, _, value = line.partition(":")
                    if key in ("Rss", "Pss", "Locked", "Private_Dirty", "VmFlags"):
                        entry[key] = value.strip()
    except OSError:
        pass
    return entry


def report(before, after, label):
    print(f"\n-- {label} --")
    for key in ["memory.current"] + list(TERMS):
        b, a = before.get(key, -1), after.get(key, -1)
        if b < 0 or a < 0:
            print(f"  {key:<14} unavailable")
            continue
        print(
            f"  {key:<14} {b / GIB:8.3f} -> {a / GIB:8.3f} GiB "
            f"(delta {(a - b) / GIB:+7.3f})"
        )


def verdict(before, after, allocated_bytes):
    """Name the term that absorbed the allocation, or say that none did."""
    deltas = {
        key: after[key] - before[key]
        for key in TERMS
        if before.get(key, -1) >= 0 and after.get(key, -1) >= 0
    }
    threshold = 0.6 * allocated_bytes
    absorbed = [key for key, value in deltas.items() if value >= threshold]
    current_delta = after["memory.current"] - before["memory.current"]
    print("\nVERDICT")
    if not absorbed:
        print(
            f"  no memory.stat term grew by >= 60% of the {allocated_bytes / GIB:.2f} "
            "GiB allocated."
        )
    else:
        print(f"  absorbed by: {', '.join(absorbed)}")
    print(
        f"  memory.current moved {current_delta / GIB:+.3f} GiB against "
        f"{allocated_bytes / GIB:+.3f} GiB allocated -> the rammon guard "
        + (
            "SEES it"
            if current_delta >= threshold
            else "does NOT see it (the guard under-counts the pool)"
        )
    )


def run_pinned(gib):
    try:
        import torch
    except ImportError:
        print("torch is not importable here; nothing to measure.", file=sys.stderr)
        return 4
    if not torch.cuda.is_available():
        print(
            "REFUSED: pinned host memory comes from cudaHostAlloc, which needs a\n"
            "CUDA context, which needs a visible card. This probe therefore needs\n"
            "ONE card for a few seconds -- it allocates host memory, samples the\n"
            "cgroup and exits; it runs no kernels and touches no VRAM beyond the\n"
            "context. Run it inside the next card window, or run\n"
            "--mlock-control now for the anon/unevictable reference arm.",
            file=sys.stderr,
        )
        return 3
    nbytes = int(gib * GIB)
    torch.zeros(1, device="cuda")  # force the context before the baseline
    before, before_proc = cgroup_sample(), proc_sample()
    buf = torch.empty(nbytes, dtype=torch.uint8).pin_memory()
    buf.fill_(1)  # touch every page: an untouched mapping proves nothing
    after, after_proc = cgroup_sample(), proc_sample()
    report(before, after, f"torch pin_memory({gib} GiB)")
    print(f"  /proc/self/status before={before_proc} after={after_proc}")
    print(f"  smaps of the pinned buffer: {smaps_for(buf.data_ptr())}")
    verdict(before, after, nbytes)
    del buf
    return 0


def run_mlock_control(mib):
    nbytes = int(mib) * (1 << 20)
    libc = ctypes.CDLL("libc.so.6", use_errno=True)
    before, before_proc = cgroup_sample(), proc_sample()
    buf = ctypes.create_string_buffer(nbytes)
    ctypes.memset(buf, 1, nbytes)
    address = ctypes.addressof(buf)
    locked = libc.mlock(ctypes.c_void_p(address), ctypes.c_size_t(nbytes)) == 0
    if not locked:
        err = ctypes.get_errno()
        print(
            f"  mlock({mib} MiB) failed with errno {err} "
            f"({os.strerror(err)}); RLIMIT_MEMLOCK is "
            f"{__import__('resource').getrlimit(__import__('resource').RLIMIT_MEMLOCK)}"
        )
    after, after_proc = cgroup_sample(), proc_sample()
    report(
        after=after, before=before, label=f"mlock control ({mib} MiB, locked={locked})"
    )
    print(f"  /proc/self/status before={before_proc} after={after_proc}")
    print(f"  smaps of the locked buffer: {smaps_for(address)}")
    verdict(before, after, nbytes)
    if locked:
        libc.munlock(ctypes.c_void_p(address), ctypes.c_size_t(nbytes))
    return 0


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gib", type=float, default=4.0, help="pinned GiB to allocate")
    parser.add_argument("--mib", type=int, default=256, help="mlock control size (MiB)")
    parser.add_argument(
        "--mlock-control",
        action="store_true",
        help="run the no-CUDA reference arm instead of the pinned arm",
    )
    args = parser.parse_args(argv)
    print(f"cgroup root: {CG}")
    if args.mlock_control:
        return run_mlock_control(args.mib)
    return run_pinned(args.gib)


if __name__ == "__main__":
    sys.exit(main())
