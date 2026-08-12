#!/usr/bin/env python3
"""#695: measure the page-locked host shmem a running boot holds, per rank.

THE MEASUREMENT THIS EXISTS FOR
-------------------------------
The phase-flip weight images were allocated with ``torch.zeros(pin_memory=
True)``, and PyTorch's pinned-host caching allocator rounds every request up
to the next power of two before ``cudaHostAlloc``
(``ATen/core/CachingHostAllocator.h:302``). Two images per rank, held for the
life of the process. On the 2026-08-12 PP=3 boot that turned 58.35 GiB of
payload into 72 GiB of mappings.

The fix allocates the images at their exact size instead. It cannot be
validated by a free-memory metric: free memory moves for a dozen reasons on a
shared box, and "more free" is exactly the reading a fix like this produces
whether or not it worked. It is validated by THE MAPPING ITSELF changing size
-- 16384.000 MiB becoming 13482.xx MiB -- which is a statement about the
allocation and nothing else. That is what ``--compare`` prints.

HOW TO RUN IT, in the next GPU window
-------------------------------------
Both halves must be the SAME launch command; only the commit differs.

  1. BEFORE. Boot on the parent commit (the one without this branch merged),
     wait for the server to answer, then::

         python3 scripts/vram_ledger/host_shmem_695.py --save /tmp/shmem_before.json

  2. Record a generation. Content correctness is half the acceptance -- a fix
     that frees memory and changes the output is not a fix::

         curl -s localhost:30030/generate \\
           -H 'Content-Type: application/json' \\
           -d '{"text":"Count from 1 to 20.","sampling_params":{"temperature":0,"max_new_tokens":64}}' \\
           > /tmp/gen_before.json

  3. Stop serving, boot on ``fix/scheduler-shmem-residency``, same flags.

  4. AFTER::

         python3 scripts/vram_ledger/host_shmem_695.py --save /tmp/shmem_after.json
         curl -s ... > /tmp/gen_after.json     # same request as step 2

  5. VERDICT::

         python3 scripts/vram_ledger/host_shmem_695.py \\
             --compare /tmp/shmem_before.json /tmp/shmem_after.json
         diff <(jq -r .text /tmp/gen_before.json) <(jq -r .text /tmp/gen_after.json)

WHAT PASSES
-----------
* Every ``anon-shared`` mapping at or above 1 GiB is NO LONGER a power of two.
  This is the load-bearing assertion; ``--compare`` fails loudly without it.
* Total anon-shared drops by roughly 13.6 GiB across the three ranks, with the
  largest single drop on the rank whose PP layout was 9114.95 MiB (it alone
  was rounded to 16384 MiB).
* The generation diff is EMPTY.
* The boot log carries one ``HOST-SHMEM rank<N> ...`` line per rank, and its
  ``residual`` is small -- the images now register a post, so measured and
  declared should agree to within workspaces.

Read-only: this script opens ``/proc/<pid>/smaps`` and nothing else. It sends
no signals and writes nothing outside the paths you name.
"""

from __future__ import annotations

import argparse
import glob
import json
import os

import sys
from typing import Dict, List, Optional

_MIB = 1 << 20
_GIB = 1 << 30

_HERE = os.path.dirname(os.path.abspath(__file__))
_PYTHON_DIR = os.path.abspath(os.path.join(_HERE, "..", "..", "python"))
if _PYTHON_DIR not in sys.path:
    sys.path.insert(0, _PYTHON_DIR)

from sglang.srt.mem_ledger.host_shmem import (  # noqa: E402
    CLASS_ANON_SHARED,
    HOST_RAM_CLASSES,
    parse_shared_mappings,
)

#: Mappings at or above this size must not be powers of two after the fix.
BIG_ENOUGH_TO_JUDGE = 1 * _GIB


def _is_power_of_two(n: int) -> bool:
    return n > 0 and (n & (n - 1)) == 0


#: What a rank's ``comm`` can actually be compared against.
#:
#: The ranks ask to be called ``sglang::scheduler_PP0`` and the like, but the
#: kernel stores ``comm`` in ``TASK_COMM_LEN`` = 16 bytes, so what
#: ``/proc/<pid>/comm`` returns is the 15-character prefix
#: ``sglang::schedul``. Testing for ``"sglang::scheduler"`` (17 chars) is
#: therefore False for every process that ever existed -- which is what this
#: script did, and why it reported "no sglang::scheduler process found"
#: against a healthy three-rank boot.
#:
#: 13 characters, so it survives the truncation with room to spare, and long
#: enough not to collide with the launcher's other children (the sibling
#: ``sglang::detoken`` does not share this prefix).
_SCHEDULER_COMM_PREFIX = "sglang::sched"


def is_scheduler_comm(comm: str) -> bool:
    """Is this ``/proc/<pid>/comm`` value one of the ranks?

    Compares against the TRUNCATED prefix on purpose -- see
    ``_SCHEDULER_COMM_PREFIX``. Matching on comm rather than the command line
    is still deliberate: it cannot match this script itself.
    """
    return comm.startswith(_SCHEDULER_COMM_PREFIX)


def _all_proc_pids() -> List[int]:
    out: List[int] = []
    for path in glob.glob("/proc/[0-9]*/comm"):
        try:
            out.append(int(path.split("/")[2]))
        except (IndexError, ValueError):
            continue
    return out


def _read_comm(pid: int) -> Optional[str]:
    try:
        with open(f"/proc/{pid}/comm", "r", encoding="utf-8") as handle:
            return handle.read().strip()
    except OSError:
        return None


def discover_scheduler_pids(read_comm=None, pids=None) -> List[int]:
    """Every scheduler rank, by comm.

    ``read_comm``/``pids`` exist so the matching can be tested without a
    running server. The desk half of this branch was hermetic and never
    exercised this function against a real ``comm``, which is exactly how the
    truncation bug reached a GPU window.
    """
    read_comm = read_comm or _read_comm
    pids = _all_proc_pids() if pids is None else pids
    found: List[int] = []
    for pid in pids:
        comm = read_comm(pid)
        if comm and is_scheduler_comm(comm):
            found.append(pid)
    return sorted(found)


def census_for(pid: int) -> Optional[Dict]:
    mappings = parse_shared_mappings(f"/proc/{pid}/smaps")
    if not mappings:
        return None
    try:
        with open(f"/proc/{pid}/comm", "r", encoding="utf-8") as handle:
            comm = handle.read().strip()
    except OSError:
        comm = f"pid{pid}"
    by_class: Dict[str, int] = {}
    for m in mappings:
        by_class[m.owner_class] = by_class.get(m.owner_class, 0) + m.pss_bytes
    big = [
        {
            "size_bytes": m.size_bytes,
            "size_mib": round(m.size_bytes / _MIB, 3),
            "pss_bytes": m.pss_bytes,
            "class": m.owner_class,
            "path": m.path,
            "power_of_two": _is_power_of_two(m.size_bytes),
        }
        for m in sorted(mappings, key=lambda x: -x.size_bytes)
        if m.size_bytes >= 128 * _MIB
    ]
    return {
        "pid": pid,
        "comm": comm,
        "by_class_pss": by_class,
        "host_ram_pss": sum(by_class.get(c, 0) for c in HOST_RAM_CLASSES),
        "anon_shared_pss": by_class.get(CLASS_ANON_SHARED, 0),
        "big_mappings": big,
    }


def cgroup_snapshot() -> Dict:
    rel = ""
    try:
        with open("/proc/self/cgroup", "r", encoding="utf-8") as handle:
            for line in handle:
                parts = line.strip().split(":", 2)
                if len(parts) == 3 and parts[0] == "0":
                    rel = parts[2].lstrip("/")
                    break
    except OSError:
        pass
    root = os.path.join("/sys/fs/cgroup", rel) if rel else "/sys/fs/cgroup"
    out: Dict[str, Optional[int]] = {"shmem": None, "current": None, "oom_kill": None}
    try:
        with open(os.path.join(root, "memory.stat"), "r", encoding="utf-8") as h:
            for line in h:
                key, _, rest = line.partition(" ")
                if key == "shmem":
                    out["shmem"] = int(rest.strip())
    except (OSError, ValueError):
        pass
    for key, fname in (("current", "memory.current"),):
        try:
            with open(os.path.join(root, fname), "r", encoding="utf-8") as h:
                out[key] = int(h.read().strip())
        except (OSError, ValueError):
            pass
    try:
        with open(os.path.join(root, "memory.events"), "r", encoding="utf-8") as h:
            for line in h:
                key, _, rest = line.partition(" ")
                if key == "oom_kill":
                    out["oom_kill"] = int(rest.strip())
    except (OSError, ValueError):
        pass
    out["cgroup"] = root
    return out


def collect(pids: List[int]) -> Dict:
    ranks = []
    for pid in pids:
        entry = census_for(pid)
        if entry is not None:
            ranks.append(entry)
    return {
        "ranks": ranks,
        "cgroup": cgroup_snapshot(),
        "total_anon_shared": sum(r["anon_shared_pss"] for r in ranks),
        "total_host_ram": sum(r["host_ram_pss"] for r in ranks),
    }


def print_report(snap: Dict) -> None:
    print(f"{'pid':>8}  {'comm':<24} {'anon-shared':>12} {'host-ram':>10}")
    for r in snap["ranks"]:
        print(
            f"{r['pid']:>8}  {r['comm']:<24} "
            f"{r['anon_shared_pss'] / _GIB:>9.2f}GiB "
            f"{r['host_ram_pss'] / _GIB:>7.2f}GiB"
        )
        for m in r["big_mappings"]:
            flag = "  <-- POWER OF TWO" if m["power_of_two"] else ""
            print(f"           {m['size_mib']:>12.3f} MiB  {m['class']}{flag}")
    print(
        f"\nTOTAL anon-shared {snap['total_anon_shared'] / _GIB:.2f} GiB   "
        f"host-ram {snap['total_host_ram'] / _GIB:.2f} GiB"
    )
    cg = snap["cgroup"]
    if cg.get("shmem") is not None:
        print(
            f"cgroup {cg.get('cgroup')}: shmem "
            f"{cg['shmem'] / _GIB:.2f} GiB  current "
            f"{(cg.get('current') or 0) / _GIB:.2f} GiB  oom_kill "
            f"{cg.get('oom_kill')}"
        )
        drift = snap["total_anon_shared"] - cg["shmem"]
        print(
            f"reconciliation: sum(Pss anon-shared) - cgroup.shmem = "
            f"{drift / _GIB:+.2f} GiB (small is good; a large gap means the "
            f"classification missed an owner)"
        )


def compare(before: Dict, after: Dict) -> int:
    """Print the verdict. Returns a process exit code."""
    b_total = before["total_anon_shared"]
    a_total = after["total_anon_shared"]
    print("#695 HOST-SHMEM LEVER -- before/after\n")
    print(f"  anon-shared BEFORE : {b_total / _GIB:8.2f} GiB")
    print(f"  anon-shared AFTER  : {a_total / _GIB:8.2f} GiB")
    print(f"  DROP               : {(b_total - a_total) / _GIB:8.2f} GiB\n")

    failures: List[str] = []

    # The load-bearing assertion: the MAPPINGS changed, not merely the free
    # memory. A fix that frees memory cannot be validated by a free-memory
    # metric -- this is the statement about the allocation itself.
    offenders = []
    for rank in after["ranks"]:
        for m in rank["big_mappings"]:
            if (
                m["class"] == CLASS_ANON_SHARED
                and m["size_bytes"] >= BIG_ENOUGH_TO_JUDGE
                and m["power_of_two"]
            ):
                offenders.append((rank["pid"], m["size_mib"]))
    if offenders:
        failures.append(
            "these large anon-shared mappings are STILL powers of two, so the "
            "image did not go through the exact-size allocator: "
            + ", ".join(f"pid {p}: {s:.3f} MiB" for p, s in offenders)
        )
    else:
        print("  OK: no large anon-shared mapping is a power of two any more.")

    if a_total >= b_total:
        failures.append(
            f"anon-shared did not fall ({b_total / _GIB:.2f} -> "
            f"{a_total / _GIB:.2f} GiB). Expected roughly -13.6 GiB."
        )

    print("\n  per-rank mappings AFTER:")
    for rank in after["ranks"]:
        sizes = ", ".join(
            f"{m['size_mib']:.3f} MiB"
            for m in rank["big_mappings"]
            if m["class"] == CLASS_ANON_SHARED
            and m["size_bytes"] >= BIG_ENOUGH_TO_JUDGE
        )
        print(f"    pid {rank['pid']}: {sizes or '(none >= 1 GiB)'}")

    if failures:
        print("\nFAILED:")
        for f in failures:
            print(f"  - {f}")
        return 1
    print(
        "\nPASSED (memory half). Now confirm the CONTENT half: the generation "
        "diff from steps 2 and 4 must be empty."
    )
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--pid", type=int, action="append", help="rank pid (repeatable)")
    ap.add_argument("--save", metavar="PATH", help="write the snapshot as JSON")
    ap.add_argument(
        "--compare",
        nargs=2,
        metavar=("BEFORE", "AFTER"),
        help="print the verdict for two saved snapshots",
    )
    args = ap.parse_args()

    if args.compare:
        with open(args.compare[0], encoding="utf-8") as h:
            before = json.load(h)
        with open(args.compare[1], encoding="utf-8") as h:
            after = json.load(h)
        return compare(before, after)

    pids = args.pid or discover_scheduler_pids()
    if not pids:
        print(
            "no sglang::scheduler process found. Boot the server first, or "
            "name the pids with --pid.",
            file=sys.stderr,
        )
        return 2
    snap = collect(pids)
    print_report(snap)
    if args.save:
        with open(args.save, "w", encoding="utf-8") as h:
            json.dump(snap, h, indent=2)
        print(f"\nsaved: {args.save}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
