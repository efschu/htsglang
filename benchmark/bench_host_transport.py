#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""HTCCL against NCCL, all_reduce and all_to_all, interleaved in the same run.

WHY
===
The host transport (`SGLANG_HTCCL_TRANSPORT=host`, see
python/sglang/srt/distributed/device_communicators/htccl_host.py) moves the
payload over ONE pinned, portable host segment and is driven entirely by two
kernels -- no host sync, no allocation in the hot path, completion signaled
via flags in pinned memory. On this rig the plain pinned host path measured
7.30 us against 37.41 us for NCCL at 20 KiB ping-pong. This program checks
how much of that arrives at the COLLECTIVE level.

    All times are the FULL operation duration of ONE operation. Neither
    all_reduce nor all_to_all has a "half path". Do NOT compare these
    against point-to-point numbers (half round-trip).

WHAT IS MEASURED (--op)
========================
``all_reduce``  as usual.
``all_to_all``  ``all_to_all_single`` along axis 0, in two
                distribution patterns, because MoE produces both:

                * ``gleich`` ("equal"): every destination rank gets the
                  same number of rows.
                * ``schief`` ("skewed"): rank 0 gets ``--schief`` times as
                  many rows from EVERY sender as the others do -- the hot
                  expert, the incast case.

                The row width comes from ``--hidden`` (elements per row);
                the actually measured payload is rounded down to a
                multiple of row width times rank count, and REPORTED as
                such.
``beide`` ("both")  one after the other, in a single run.

For all_to_all, the BAR1 direct path is asked **before** the measurement
whether it genuinely runs this cell (``handles`` plus ``supports_a2a`` with
the largest block over all pairs). If it says no -- typically because a
skewed block exceeds the slot limit -- the cell is carried as dead, with a
reason, and is NOT measured over the gloo layer instead and reported as BAR1.

MEASUREMENT DISCIPLINE (built in, not optional)
=================================================
* **Warmup is mandatory.** Without it, this rig measured 726 us instead of
  95 us -- a factor of 7.6 from the P-state ramp alone. The warmup budget
  per cell is pinned to at least 3 s; a smaller --warmup is raised, and the
  raise is reported.
* **Interleaved, not block-wise.** Each round runs a short series per cell,
  with the order rotated per round. That way drift, external load and clock
  behavior hit every backend equally.
* **Correctness first, then timing.** Every round starts with a
  known-answer check per cell. Whichever cell fails it is no longer measured
  from that round on, and is carried in the report with a reason.
* **Dead variants with a reason, never a silent fallback.** If a transport
  fails to come up, or the active transport is not the one requested
  (htccl.py silently falls back to the gloo inline layer for some names),
  the backend is dropped on ALL ranks and the reason is printed. Something
  else is never measured as a substitute and reported as the one requested.

COMPARABILITY, honestly
=========================
* HTCCL's `all_reduce` is OUT-OF-PLACE (returns a new tensor), NCCL's
  `dist.all_reduce` is IN-PLACE. The host transport therefore additionally
  writes a full result into a fresh buffer. If anything, the comparison is
  skewed AGAINST HTCCL -- not in its favor.
* The completion signal in the measurement path is the same for BOTH
  backends: `--completion sync` (default) calls `torch.cuda.synchronize`,
  `--completion event` polls a `cuda.Event`. The transport itself never
  synchronizes with the host; the measurement path only does so to get a
  timestamp, and does so identically for every backend.

DEADLOCK CAUTION
================
This project already had a deadlock once, because every rank computed its
own round count locally. Here the rule is: a pilot run with a FIXED
iteration count identical across all ranks; rank 0 computes the plan from
it and distributes it via broadcast_object_list; every backend dropout is
decided jointly via all_gather_object. On top of that: a gloo timeout, a
hard watchdog per rank, and an overall time limit in the parent process.

INVOCATION
==========
    python3 benchmark/bench_host_transport.py --devices 1,2
    python3 benchmark/bench_host_transport.py --devices 1,2,3 \
        --op all_to_all --backends htccl:bar1,nccl

The rank count follows from ``--devices``; it is no longer fixed at two.

Other flags: --op, --hidden, --schief, --sizes, --backends, --secs,
--warmup, --rounds, --dtype, --slot-mib, --completion, --out, --devices. The
cards must be free; the program does not preempt anything and aborts if more
than --max-used-mib is in use on a target card.
"""
from __future__ import annotations

import argparse
import faulthandler
import json
import os
import subprocess
import sys
import threading
import time
from datetime import timedelta

# Must come BEFORE the torch import, otherwise the ordering follows compute
# capability instead of the PCI bus.
os.environ.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")

#: 20 KiB, 80 KiB, 1 MiB, 4 MiB, 16 MiB.
DEFAULT_SIZES = "20480,81920,1048576,4194304,16777216"

#: name -> (SGLANG_HTCCL_TRANSPORT value, expected transport class). The class
#: name is the probe against the silent fallback: htccl.py lets an unknown or
#: failed transport drop to the gloo inline layer, and then you measure gloo
#: and call it "host".
HTCCL_BACKENDS = {
    "htccl:host": ("host", "HTCCLHostTransport"),
    "htccl:device": ("device", "HTCCLDeviceTransport"),
    "htccl:shm": ("shm", "HTCCLShmTransport"),
    "htccl:bar1": ("bar1", "HTCCLBar1Transport"),
    "htccl:matrix": ("matrix", "HTCCLMatrixTransport"),
}
DEFAULT_BACKENDS = "htccl:host,nccl"

#: Lower bound for the warmup per cell. See the header: without warmup, a
#: factor of 7.6.
MIN_WARMUP_S = 3.0

#: Distribution patterns for all_to_all. all_reduce only knows "-".
#: The two literal values are the CLI/report contract and stay German.
PATTERNS = ("gleich", "schief")


def cell_key(cell) -> str:
    """One cell as a flat string -- the collectives carry dicts.

    ``(op, bytes, pattern, backend)``. Deliberately ONE function and not the
    same f-string in five places: the drop vote and the result round have to
    build the same key, otherwise one rank drops a cell the other keeps
    measuring -- and that is a hang.
    """
    op, nb, pattern, be = cell
    return f"{op}|{nb}|{pattern}|{be}"


def split_sizes(rows: int, world: int, pattern: str, skew: int) -> list:
    """Send row counts per destination rank. The same math on ALL ranks.

    ``gleich`` (equal): ``rows/world`` to everyone.
    ``schief`` (skewed): rank 0 gets ``skew`` times as many as the others --
    a hot expert everybody writes to at once. The remainder of the division
    goes to rank 0 so that the sum is exactly ``rows``; a sum that does not
    add up would be a buffer that does not fit.
    """
    if pattern == "gleich":
        g = rows // world
        s = [g] * world
    else:
        weights = [skew] + [1] * (world - 1)
        total = sum(weights)
        base = rows // total
        s = [g * base for g in weights]
    s[0] += rows - sum(s)
    return s


# ----------------------------------------------------------------------
# Watchdog: a clear abort beats a hanging rank.
# ----------------------------------------------------------------------
def arm_watchdog(seconds: float, tag: str) -> None:
    def boom() -> None:
        sys.stderr.write(
            f"\n[{tag}] ABORT: watchdog after {seconds:.0f} s. "
            f"A collective did not come back. Traceback:\n"
        )
        sys.stderr.flush()
        faulthandler.dump_traceback()
        os._exit(3)

    t = threading.Timer(seconds, boom)
    t.daemon = True
    t.start()


# ----------------------------------------------------------------------
# Statistik
# ----------------------------------------------------------------------
def summarize(lat_us: list[float], nbytes: int) -> dict:
    s = sorted(lat_us)
    n = len(s)

    def pct(p: float) -> float:
        if n == 0:
            return float("nan")
        return s[min(n - 1, max(0, int(round(p * (n - 1)))))]

    p50 = pct(0.50)
    return {
        "n": n,
        "min": s[0] if n else float("nan"),
        "p50": p50,
        "p90": pct(0.90),
        "p99": pct(0.99),
        "mean": sum(s) / n if n else float("nan"),
        # Payload per operation divided by the duration. Deliberately NOT the
        # "bus bandwidth" of the NCCL tests (2*(N-1)/N * payload): this is
        # meant to be a handy second number next to the latency, not a measure
        # of link utilization.
        "mbps": (nbytes / (p50 * 1e-6)) / 1e6 if p50 == p50 and p50 else float("nan"),
    }


# ----------------------------------------------------------------------
# Child process: one rank
# ----------------------------------------------------------------------
def run_rank(a: argparse.Namespace) -> int:
    import torch
    import torch.distributed as dist

    rank = a.rank
    # The rank count follows from --devices. It used to be hardwired to 2
    # while --devices had always been a list: whoever passed three ordinals
    # silently got two ranks and one unused card -- which is why the 3-rank
    # numbers of this effort do NOT come from this program.
    world = len([x for x in a.devices.split(",") if x.strip()])
    tag = f"r{rank}"
    arm_watchdog(a.max_secs, tag)

    dev_ord = [int(x) for x in a.devices.split(",")][rank]
    torch.cuda.set_device(dev_ord)
    dev = torch.device("cuda", dev_ord)

    def log(msg: str) -> None:
        if rank == 0:
            print(msg, flush=True)

    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ["RANK"] = str(rank)
    os.environ["WORLD_SIZE"] = str(world)

    to = timedelta(seconds=a.pg_timeout)
    dist.init_process_group(
        backend="cpu:gloo,cuda:nccl", rank=rank, world_size=world, timeout=to,
        device_id=dev,
    )
    # Separate groups: HTCCL needs a pure gloo CPU group, NCCL gets its own,
    # so that the two data planes do not serialize against each other over a
    # shared group.
    gloo_pg = dist.new_group(ranks=list(range(world)), backend="gloo", timeout=to)
    nccl_pg = dist.new_group(ranks=list(range(world)), backend="nccl", timeout=to)

    dtype = {"bfloat16": torch.bfloat16, "float16": torch.float16,
             "float32": torch.float32}[a.dtype]
    esize = torch.tensor([], dtype=dtype).element_size()
    sizes = [int(x) for x in a.sizes.split(",")]
    for nb in sizes:
        if nb % esize:
            raise SystemExit(f"{nb} B is not a multiple of {esize} B ({a.dtype})")

    p = torch.cuda.get_device_properties(dev_ord)
    print(f"# rank{rank} -> cuda:{dev_ord} {p.name} sm_{p.major}{p.minor}", flush=True)

    # ---------------- build the HTCCL communicators ----------------
    sys.path.insert(0, a.sglang_python)
    import sglang.srt.distributed.device_communicators.htccl as htccl_mod

    comms: dict[str, object] = {}
    dead: dict[str, str] = {}
    for name in a.backends:
        if name == "nccl":
            continue
        transport, want = HTCCL_BACKENDS[name]
        err = ""
        obj = None
        t0 = time.time()
        try:
            # In htccl.py the transport name comes from the module variable
            # `_TRANSPORT`, filled once at import time from
            # SGLANG_HTCCL_TRANSPORT; a single process therefore cannot see
            # the env var with two different values. Setting `_TRANSPORT` is
            # exactly the value the variable would have produced -- same code
            # path.
            htccl_mod._TRANSPORT = transport
            obj = htccl_mod.HTCCLCommunicator(cpu_group=gloo_pg, device=dev)
            got = type(obj.transport).__name__ if obj.transport is not None else None
            if got != want:
                err = (
                    f"transport not active (active: {got or 'gloo inline layer'}, "
                    f"expected: {want}); nothing is measured as a substitute"
                )
        except Exception as e:  # noqa: BLE001 - the reason belongs in the report
            err = f"{type(e).__name__}: {e}"
        build_s = time.time() - t0
        # Decide uniformly: otherwise one rank measures a backend the other
        # does not even have -> deadlock in the first collective.
        votes: list = [None] * world
        dist.all_gather_object(votes, err, group=gloo_pg)
        bad = [f"r{i}: {v}" for i, v in enumerate(votes) if v]
        if bad:
            dead[name] = "; ".join(bad)
            if obj is not None:
                try:
                    obj.close()
                except Exception:  # noqa: BLE001
                    pass
        else:
            comms[name] = obj
            log(f"# {name}: up in {build_s:.1f} s ({want})")

    active = [b for b in a.backends if b == "nccl" or b in comms]
    if not active:
        for k, why in dead.items():
            print(f"# {k}: did NOT come up -- {why}", flush=True)
        raise SystemExit("ABORT: not a single backend came up.")

    # ---------------- buffers: allocated once, never in the measured path ---
    # The timing buffers are zeros: NCCL's all_reduce is in-place, and
    # repeatedly summing ones runs into inf in bf16 after a few hundred
    # rounds. The values influence none of the measured DMA/kernel paths. The
    # correctness check uses its own buffers with real content.
    timing_buf = {nb: torch.zeros(nb // esize, dtype=dtype, device=dev)
                  for nb in sizes}
    # Known answer: rank r contributes (r+1) * pattern, so the sum over the
    # group is (W(W+1)/2) * pattern. Small integers -> exactly representable
    # in bf16, so the check is a torch.equal and not a tolerance comparison
    # that could wave a half-broken data path through.
    check_in, check_want = {}, {}
    for nb in sizes:
        n = nb // esize
        pattern = (torch.arange(n, device=dev, dtype=torch.int32) % 7 + 1).to(dtype)
        check_in[nb] = (pattern * (rank + 1)).contiguous()
        check_want[nb] = (pattern * (world * (world + 1) // 2)).contiguous()

    # ---------------- all_to_all: geometry, buffers, known answer ----------
    #
    # A block's marker is `sender*world + receiver + 1` plus
    # `0.5 * (column % 4)`. Both are EXACTLY representable in bf16 (the values
    # stay below 128, where the step size is 0.5), so the check is a
    # torch.equal and not a tolerance comparison. The column term is not
    # decoration: without it a whole block carries the same value and a
    # transposition INSIDE a block goes unnoticed.
    row_elems = int(a.hidden)
    row_bytes = row_elems * esize
    a2a: dict = {}                # (nb, pattern) -> dict of buffers and sizes
    a2a_tot: dict = {}            # (nb, pattern) -> actual payload in bytes

    def _marker(sender: int, receiver: int) -> int:
        return sender * world + receiver + 1

    if a.op in ("all_to_all", "beide"):
        column = (torch.arange(row_elems, device=dev, dtype=torch.int32) % 4)
        column = (column.to(torch.float32) * 0.5).to(dtype)
        for nb in sizes:
            rows = (nb // row_bytes // world) * world
            if rows < world:
                raise SystemExit(
                    f"ABORT: at --hidden {row_elems} and {world} ranks, "
                    f"{nb} B do not carry {world} rows. Use larger --sizes "
                    f"or a smaller --hidden."
                )
            for pattern in PATTERNS:
                send = split_sizes(rows, world, pattern, a.schief)
                # What I receive is column `rank` of the send matrix. Every
                # rank runs the same function, so this needs no collective --
                # in a benchmark the pattern is known up front.
                recv = [split_sizes(rows, world, pattern, a.schief)[rank]
                        for _ in range(world)]
                x = torch.empty((rows, row_elems), dtype=dtype, device=dev)
                o = 0
                for j, n in enumerate(send):
                    if n:
                        x[o:o + n] = column + float(_marker(rank, j))
                    o += n
                want = torch.empty((sum(recv), row_elems), dtype=dtype,
                                   device=dev)
                o = 0
                for i, n in enumerate(recv):
                    if n:
                        want[o:o + n] = column + float(_marker(i, rank))
                    o += n
                a2a[(nb, pattern)] = {
                    "send": send, "recv": recv, "rows": rows,
                    "x": x, "want": want,
                    "zeros": torch.zeros((rows, row_elems), dtype=dtype,
                                         device=dev),
                    "out": torch.empty((sum(recv), row_elems), dtype=dtype,
                                       device=dev),
                    "out_check": torch.empty((sum(recv), row_elems),
                                             dtype=dtype, device=dev),
                    # Largest block over ALL pairs -- the number the direct
                    # path is asked with, to say whether it runs this cell.
                    "max_block": max(send) * row_bytes,
                }
                a2a_tot[(nb, pattern)] = rows * row_bytes

    def do_op(cell, x, out=None):
        """One operation. Returns the result."""
        op, nb, pattern, backend = cell
        if op == "all_reduce":
            if backend == "nccl":
                dist.all_reduce(x, group=nccl_pg)
                return x
            return comms[backend].all_reduce(x)
        g = a2a[(nb, pattern)]
        if backend == "nccl":
            dist.all_to_all_single(
                out, x, output_split_sizes=g["recv"],
                input_split_sizes=g["send"], group=nccl_pg,
            )
            return out
        return comms[backend].all_to_all_single(out, x, g["recv"], g["send"])

    # Created once, never in the measured path: a `torch.cuda.Event()` per
    # iteration would be exactly the hot-path allocation this program is
    # supposed to demonstrate the absence of.
    done_ev = torch.cuda.Event()

    def wait_done() -> None:
        """Completion signal, identical for every backend."""
        if a.completion == "event":
            done_ev.record()
            while not done_ev.query():
                pass
        else:
            torch.cuda.synchronize(dev)

    ops = ["all_reduce", "all_to_all"] if a.op == "beide" else [a.op]
    cells: list = []
    for op in ops:
        for nb in sizes:
            for pattern in (PATTERNS if op == "all_to_all" else ("-",)):
                for be in active:
                    cells.append((op, nb, pattern, be))

    # ---------------- ask the direct path up front, for all_to_all --------
    #
    # At large payloads a skewed block exceeds the slot limit. Then
    # HTCCLCommunicator.all_to_all_single falls back to the CPU layer -- which
    # is correct, but here it would be a measurement of the gloo layer under
    # the name of the direct path. So ask first, and otherwise drop the cell
    # with a reason.
    for cell in list(cells):
        op, nb, pattern, be = cell
        if op != "all_to_all" or be == "nccl":
            continue
        tr = getattr(comms[be], "transport", None)
        if tr is None:
            continue
        g = a2a[(nb, pattern)]
        reason = ""
        if not tr.handles("all_to_all", a2a_tot[(nb, pattern)]):
            reason = (
                f"handles('all_to_all', {a2a_tot[(nb, pattern)]}) says False"
            )
        elif not getattr(tr, "supports_a2a", lambda _: False)(g["max_block"]):
            slot = getattr(tr, "a2a_slot_bytes", lambda: 0)()
            reason = (
                f"largest block {g['max_block']} B does not fit into the "
                f"slot of {slot} B"
            )
        votes: list = [None] * world
        dist.all_gather_object(votes, reason, group=gloo_pg)
        bad = [f"r{i}: {v}" for i, v in enumerate(votes) if v]
        if bad:
            cells.remove(cell)
            dead[f"{be} @ all_to_all/{pattern} {a2a_tot[(nb, pattern)]} B"] = (
                "direct path opts out -- " + "; ".join(bad)
                + ". The CPU layer is NOT measured as a substitute. "
                "For a larger window: raise --bar1-window-mib (the a2a slot "
                "is roughly window/(6(R-1)))."
            )
    if not cells:
        log("# ABORT: no cell left that can genuinely be measured.")
        for k, why in dead.items():
            print(f"# {k}: NOT measured -- {why}", flush=True)
        raise SystemExit(2)

    def series(cell, iters: int, timed: bool) -> list[float]:
        op, nb, pattern, be = cell
        if op == "all_reduce":
            x, out = timing_buf[nb], None
        else:
            g = a2a[(nb, pattern)]
            x, out = g["zeros"], g["out"]
        res: list[float] = []
        if not timed:
            for _ in range(iters):
                do_op(cell, x, out)
            wait_done()
            return res
        for _ in range(iters):
            t0 = time.perf_counter()
            do_op(cell, x, out)
            wait_done()
            res.append((time.perf_counter() - t0) * 1e6)
        return res

    def check(cell) -> str:
        """Known-answer check. Empty string = fine."""
        op, nb, pattern, be = cell
        if op == "all_reduce":
            # NCCL computes in place, HTCCL out of place -- the copy sits
            # outside any timing.
            x, out, want = check_in[nb].clone(), None, check_want[nb]
        else:
            g = a2a[(nb, pattern)]
            # The output buffer is poisoned BEFORE every check: otherwise a
            # collective that writes nothing at all looks like a hit as soon
            # as the buffer happened to be right once.
            g["out_check"].fill_(-1.0)
            x, out, want = g["x"], g["out_check"], g["want"]
        try:
            got = do_op(cell, x, out)
            wait_done()
        except Exception as e:  # noqa: BLE001
            return f"{type(e).__name__}: {e}"
        if got.shape != want.shape:
            return f"shape {tuple(got.shape)} instead of {tuple(want.shape)}"
        if not torch.equal(got, want):
            wrong_mask = got != want
            wrong = int(wrong_mask.sum())
            first = [int(v) for v in wrong_mask.nonzero()[0]]
            return (
                f"{wrong}/{got.numel()} elements wrong, first at {first}: "
                f"{got[tuple(first)].item()} instead of "
                f"{want[tuple(first)].item()}"
            )
        return ""

    def cell_bytes(cell) -> int:
        """Payload of ONE operation. For all_to_all the ACTUAL one.

        It is not the requested size: the row count is rounded down to a
        multiple of row width times rank count. Writing the requested number
        into the MB/s column would mean computing with bytes that never
        flowed.
        """
        op, nb, pattern, _ = cell
        return nb if op == "all_reduce" else a2a_tot[(nb, pattern)]

    # ---------------- pilot: FIXED iteration count on all ranks -----------
    log(f"# Pilot ({a.pilot} iterations per cell, same count on all ranks)")
    pilot: dict[tuple, float] = {}
    for cell in cells:
        dist.barrier(group=gloo_pg)
        series(cell, 3, timed=False)  # first contact / JIT / allocator
        lat = series(cell, a.pilot, timed=True)
        pilot[cell] = sorted(lat)[len(lat) // 2] if lat else 1000.0

    # ---------------- plan: rank 0 decides, broadcast to everyone ---------
    plan_box: list = [None]
    if rank == 0:
        plan = {}
        per_round = max(0.2, a.secs) / max(1, a.rounds)
        for cell in cells:
            us = max(pilot[cell], 1.0)
            warm = int(min(200000, max(20, a.warmup * 1e6 / us)))
            meas = int(min(200000, max(5, per_round * 1e6 / us)))
            plan[cell] = (warm, meas)
        plan_box[0] = plan
    dist.broadcast_object_list(plan_box, src=0, group=gloo_pg)
    plan = plan_box[0]
    est = sum((w + a.rounds * (m + 1)) * pilot[c] for c, (w, m) in plan.items()) / 1e6
    log(f"# Plan distributed by rank 0; estimated measurement time ~{est:.0f} s")

    # ---------------- warmup (mandatory) ----------------
    t0 = time.time()
    for cell in cells:
        dist.barrier(group=gloo_pg)
        series(cell, plan[cell][0], timed=False)
    log(f"# Warmup done in {time.time() - t0:.1f} s "
        f"(budget {a.warmup:.1f} s per cell)")

    # ---------------- measurement, interleaved ----------------
    lat: dict[tuple, list[float]] = {c: [] for c in cells}
    live = list(cells)
    t0 = time.time()
    for rnd in range(a.rounds):
        # Correctness first, then timing -- in EVERY round.
        verdict = {}
        for cell in live:
            dist.barrier(group=gloo_pg)
            verdict[cell] = check(cell)
        votes: list = [None] * world
        dist.all_gather_object(
            votes,
            {cell_key(c): v for c, v in verdict.items()},
            group=gloo_pg,
        )
        drop = []
        for cell in live:
            key = cell_key(cell)
            why = [f"r{i}: {v[key]}" for i, v in enumerate(votes) if v.get(key)]
            if why:
                drop.append(cell)
                dead[f"{cell[3]} @ {cell[0]}/{cell[2]} {cell_bytes(cell)} B"] = (
                    f"correctness failed in round {rnd + 1} -- " + "; ".join(why)
                )
        for cell in drop:  # decided jointly, hence the same on all ranks
            live.remove(cell)
        if not live:
            log("# ABORT: no cell survived the correctness check.")
            break

        # Round robin in short series, order rotated per round, so that no
        # backend always runs first after the barrier.
        order = live[rnd % len(live):] + live[: rnd % len(live)]
        for cell in order:
            dist.barrier(group=gloo_pg)
            # One settling iteration per series is discarded: it carries the
            # rank skew of the barrier, not the transport time.
            s = series(cell, plan[cell][1] + 1, timed=True)
            lat[cell].extend(s[1:])
        if rank == 0:
            print(f"# Round {rnd + 1}/{a.rounds} after {time.time() - t0:.1f} s",
                  flush=True)
        if time.time() - t0 > a.secs * len(cells) * 3 + 60:
            log("# Time budget exhausted, the measurement stops here.")
            break
    log(f"# Measurement done in {time.time() - t0:.1f} s")

    # ---------------- result ----------------
    res = {c: summarize(v, cell_bytes(c)) for c, v in lat.items() if v}
    peer: list = [None] * world
    dist.all_gather_object(
        peer, {cell_key(c): v for c, v in res.items()}, group=gloo_pg
    )
    if rank == 0:
        meta = {
            "dtype": a.dtype,
            "devices": a.devices,
            "completion": a.completion,
            "gpus": [],
            "nccl": ".".join(map(str, torch.cuda.nccl.version())),
            "torch": torch.__version__,
            "rounds": a.rounds,
            "warmup_s_per_cell": a.warmup,
            "measure_s_per_cell": a.secs,
            "htccl_slot_mib": os.environ.get("SGLANG_HTCCL_SLOT_MIB", "64 (default)"),
            "htccl_host_blocks": os.environ.get("SGLANG_HTCCL_HOST_BLOCKS",
                                                "32 (default)"),
            "world": world,
            "op": a.op,
            "hidden": a.hidden,
            "skew_factor": a.schief,
            "a2a_geometry": {
                f"{nb}|{m}": {
                    "rows": g["rows"], "send": g["send"], "recv": g["recv"],
                    "bytes": a2a_tot[(nb, m)], "max_block": g["max_block"],
                }
                for (nb, m), g in a2a.items()
            },
            "dead_variants": dead,
        }
        for r, o in enumerate([int(x) for x in a.devices.split(",")]):
            q = torch.cuda.get_device_properties(o)
            meta["gpus"].append(f"rank{r}=cuda:{o} {q.name} sm_{q.major}{q.minor}")
        report(a, ops, sizes, active, res, peer, meta, a2a_tot)

    for c in comms.values():
        try:
            c.close()
        except Exception:  # noqa: BLE001
            pass
    dist.barrier(group=gloo_pg)
    dist.destroy_process_group()
    return 0


# ----------------------------------------------------------------------
# Report (rank 0 only)
# ----------------------------------------------------------------------
def report(a, ops, sizes, backends, res, peer_res, meta, a2a_tot) -> None:
    print()
    print("=" * 78)
    print("HTCCL against NCCL -- one run, interleaved, "
          + "/".join(ops))
    print("=" * 78)
    for k, v in meta.items():
        if k == "gpus":
            for g in v:
                print(f"  {g}")
        elif k == "dead_variants":
            for kk, vv in v.items():
                print(f"  NOT measured: {kk} -- {vv}")
        elif k == "a2a_geometry":
            for kk, vv in v.items():
                print(f"  a2a {kk}: {vv['rows']} rows, sending {vv['send']}, "
                      f"receiving {vv['recv']}, {vv['bytes']} B, largest "
                      f"block {vv['max_block']} B")
        else:
            print(f"  {k}: {v}")
    if not meta["dead_variants"]:
        print("  dead variants: none")
    print()
    print("  All times in microseconds and FULL OPERATION DURATION.")
    print("  Neither all_reduce nor all_to_all has a half path --")
    print("  do not mix these with the point-to-point numbers.")
    print("  MB/s = payload per operation / p50, not bus bandwidth.")
    print("  all_reduce: HTCCL computes OUT of place, NCCL IN place -- HTCCL")
    print("  additionally writes a full result. The bias runs against HTCCL,")
    print("  not in its favor.")
    print("  all_to_all: BOTH out of place, same split sizes, same buffers.")
    print("  The comparison is unbiased there.")
    print("  Under --op all_to_all the bytes are the ones ACTUALLY sent")
    print("  (rounded down to row width times rank count), not the ones")
    print("  requested via --sizes.")
    print()

    def _patterns(op):
        return PATTERNS if op == "all_to_all" else ("-",)

    for op in ops:
        for nb in sizes:
            for m in _patterns(op):
                row_cells = [(be, res[(op, nb, m, be)]) for be in backends
                             if (op, nb, m, be) in res]
                real = nb if op == "all_reduce" else a2a_tot.get((nb, m), nb)
                title = (f"--- {op}"
                         + (f"/{m}" if op == "all_to_all" else "")
                         + f", {real} B ({real / 1024:.0f} KiB), "
                         + f"{meta['dtype']} ---")
                if not row_cells:
                    print(f"{title} no cell survived\n")
                    continue
                best = min(r[1]["p50"] for r in row_cells)
                ref = dict(row_cells).get("nccl", {}).get("p50")
                print(title)
                print(f"{'Backend':<14}{'n':>8}{'p50 us':>10}{'p99 us':>10}"
                      f"{'MB/s':>10}{'x best':>8}{'x NCCL':>9}")
                for be, r in sorted(row_cells, key=lambda x: x[1]["p50"]):
                    vs = f"{ref / r['p50']:.2f}" if ref and be != "nccl" else "-"
                    print(f"{be:<14}{r['n']:>8}{r['p50']:>10.2f}"
                          f"{r['p99']:>10.2f}{r['mbps']:>10.1f}"
                          f"{r['p50'] / best:>8.2f}{vs:>9}")
                print()

    # The HOSTBENCH lines below are the machine-readable contract of this
    # benchmark; their key names stay as they are so existing log scrapers
    # keep working.
    print("# Machine-readable (FULL operation duration, us):")
    for op in ops:
        for nb in sizes:
            for m in _patterns(op):
                real = nb if op == "all_reduce" else a2a_tot.get((nb, m), nb)
                for be in backends:
                    r = res.get((op, nb, m, be))
                    if not r:
                        continue
                    key = f"{op}|{nb}|{m}|{be}"
                    # All peers, not just rank 1: with three ranks the slowest
                    # rank is the operation duration, and that one used to be
                    # missing from the report.
                    peers = [pr.get(key, {}).get("p50", float("nan"))
                             for pr in (peer_res or [])]
                    print(
                        f"HOSTBENCH op={op} verteilung={m} backend={be} "
                        f"bytes={real} dtype={meta['dtype']} n={r['n']} "
                        f"min_us={r['min']:.2f} p50_us={r['p50']:.2f} "
                        f"p90_us={r['p90']:.2f} p99_us={r['p99']:.2f} "
                        f"mean_us={r['mean']:.2f} mbps={r['mbps']:.1f} "
                        f"peer_p50_us=" + ",".join(f"{x:.2f}" for x in peers)
                        + " half_rt=no",
                        flush=True,
                    )
    for k, v in meta["dead_variants"].items():
        print(f"HOSTBENCH dead={k!r} reason={v!r}", flush=True)

    if a.out:
        blob = {"meta": meta,
                "ranks": [{cell_key(c): v for c, v in res.items()}]
                + [dict(pr or {}) for pr in (peer_res or [])[1:]]}
        os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
        with open(a.out, "w") as f:
            json.dump(blob, f, indent=1)
        print(f"\n# JSON: {a.out}")


# ----------------------------------------------------------------------
# Parent process
# ----------------------------------------------------------------------
def preflight(devices: list[int], max_used_mib: int) -> None:
    q = subprocess.run(
        ["nvidia-smi", "--query-gpu=index,name,pci.bus_id,memory.used",
         "--format=csv,noheader,nounits"],
        capture_output=True, text=True, check=True,
        env={**os.environ, "CUDA_DEVICE_ORDER": "PCI_BUS_ID"},
    ).stdout.strip().splitlines()
    busy = []
    for line in q:
        idx, name, bus, used = [x.strip() for x in line.split(",")]
        if int(idx) in devices:
            print(f"# GPU {idx} {name} {bus}: {used} MiB in use")
            if int(used) > max_used_mib:
                busy.append(f"GPU {idx} ({bus}) holds {used} MiB")
    apps = subprocess.run(
        ["nvidia-smi", "--query-compute-apps=pid,used_memory,gpu_bus_id",
         "--format=csv,noheader"],
        capture_output=True, text=True, check=True,
    ).stdout.strip()
    if apps:
        print(f"# Foreign compute processes:\n{apps}")
    if busy:
        raise SystemExit(
            "ABORT: target cards are not free -- " + "; ".join(busy)
            + ". Nothing is preempted."
        )


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--rank", type=int, default=-1, help="internal: child process")
    ap.add_argument("--devices", default="1,2",
                    help="CUDA ordinals under PCI_BUS_ID, one rank per entry")
    ap.add_argument("--op", default="all_reduce",
                    choices=["all_reduce", "all_to_all", "beide"],
                    help="which collectives are measured ('beide' = both)")
    ap.add_argument("--hidden", type=int, default=512,
                    help="elements per row for all_to_all; the split sizes "
                         "are row counts, as with the MoE dispatchers")
    ap.add_argument("--schief", type=int, default=4,
                    help="multiple that rank 0 receives from EVERY sender in "
                         "the 'schief' (skewed) pattern -- the hot expert")
    ap.add_argument("--sizes", default=DEFAULT_SIZES)
    ap.add_argument("--dtype", default="bfloat16",
                    choices=["bfloat16", "float16", "float32"])
    ap.add_argument("--backends", default=DEFAULT_BACKENDS,
                    help="comma list of nccl," + ",".join(HTCCL_BACKENDS))
    ap.add_argument("--secs", type=float, default=3.0,
                    help="measurement budget per cell in seconds (over all "
                         "rounds)")
    ap.add_argument("--warmup", type=float, default=MIN_WARMUP_S,
                    help=f"warmup budget per cell in seconds (minimum "
                         f"{MIN_WARMUP_S:.0f})")
    ap.add_argument("--rounds", type=int, default=10,
                    help="interleaving rounds; one short series per cell and "
                         "round")
    ap.add_argument("--pilot", type=int, default=15,
                    help="fixed pilot iterations per cell (used to estimate "
                         "the round counts)")
    ap.add_argument("--completion", default="sync", choices=["sync", "event"],
                    help="completion signal in the measured path, the same "
                         "for every backend")
    ap.add_argument("--slot-mib", type=int, default=32,
                    help="SGLANG_HTCCL_SLOT_MIB for the child processes; must "
                         "hold at least the largest payload")
    ap.add_argument("--bar1-window-mib", type=int, default=0,
                    help="SGLANG_HTCCL_BAR1_WINDOW_MIB for the child "
                         "processes (0 = the module default, 96). The a2a "
                         "slot per directed pair is roughly window/(6(R-1)); "
                         "a skewed block above that makes the direct path "
                         "opt out.")
    ap.add_argument("--max-secs", type=float, default=2400.0,
                    help="hard watchdog per rank")
    ap.add_argument("--pg-timeout", type=float, default=180.0)
    ap.add_argument("--port", type=int, default=29593)
    ap.add_argument("--sglang-python", default="/spinning/wt-gdr-loadsym/python")
    ap.add_argument("--out", default="")
    ap.add_argument("--max-used-mib", type=int, default=64)
    a = ap.parse_args()
    a.backends = [b for b in a.backends.split(",") if b]
    for b in a.backends:
        if b != "nccl" and b not in HTCCL_BACKENDS:
            raise SystemExit(f"unknown backend {b!r}")
    if a.warmup < MIN_WARMUP_S:
        print(f"# Warmup raised from {a.warmup:.1f} s to {MIN_WARMUP_S:.1f} s: "
              f"without warmup this rig measured 726 us instead of 95 us "
              f"(factor 7.6 from the P-state ramp).", flush=True)
        a.warmup = MIN_WARMUP_S

    if a.rank >= 0:
        return run_rank(a)

    max_bytes = max(int(x) for x in a.sizes.split(","))
    if max_bytes > a.slot_mib * 1024 * 1024:
        # No silent fallback: the host transport does split larger payloads
        # into slot-sized portions, but then you measure a chain and call it
        # one operation. Better to pick a slot that fits.
        raise SystemExit(
            f"ABORT: largest payload {max_bytes} B does not fit into a slot "
            f"of {a.slot_mib} MiB. Raise --slot-mib."
        )

    devices = [int(x) for x in a.devices.split(",") if x.strip()]
    if len(devices) < 2:
        raise SystemExit("ABORT: --devices needs at least two ordinals.")
    if len(devices) != len(set(devices)):
        raise SystemExit(
            f"ABORT: --devices {a.devices} names a card twice. Two ranks on "
            f"the same card do not measure a transport."
        )
    if a.op in ("all_to_all", "beide") and a.schief < 1:
        raise SystemExit("ABORT: --schief must be at least 1.")
    preflight(devices, a.max_used_mib)

    env = {**os.environ,
           "CUDA_DEVICE_ORDER": "PCI_BUS_ID",
           "MASTER_ADDR": "127.0.0.1",
           "MASTER_PORT": str(a.port),
           "SGLANG_HTCCL_SLOT_MIB": str(a.slot_mib),
           "PYTHONPATH": a.sglang_python + ":" + os.environ.get("PYTHONPATH", ""),
           "PYTHONUNBUFFERED": "1"}
    if a.bar1_window_mib:
        env["SGLANG_HTCCL_BAR1_WINDOW_MIB"] = str(a.bar1_window_mib)
    base = [sys.executable, os.path.abspath(__file__)]
    for k, v in vars(a).items():
        if k == "rank":
            continue
        key = "--" + k.replace("_", "-")
        val = ",".join(map(str, v)) if isinstance(v, list) else str(v)
        if val:
            base += [key, val]
    procs = [subprocess.Popen(base + ["--rank", str(r)], env=env)
             for r in range(len(devices))]
    rc = 0
    deadline = time.time() + a.max_secs + 120
    try:
        for pr in procs:
            left = max(1.0, deadline - time.time())
            rc |= abs(pr.wait(timeout=left))
    except subprocess.TimeoutExpired:
        sys.stderr.write("\nABORT: overall time limit exceeded, terminating "
                         "our own ranks.\n")
        for pr in procs:
            pr.kill()
        rc = 4
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
