#!/usr/bin/env python3
"""NCCL / system-RAM reference measurement in the #279 format.

The HTCCL path dispatcher's third rate source does not exist yet; its FORMAT
does (htccl_path_rates.new_nccl_reference_envelope, schema_version 1). This
script produces exactly that, so the result is loadable by load_nccl_reference
without a single line of glue.

WHAT IS MEASURED, AND WHY IN THIS SHAPE

* Per row BOTH p50 and p99, in every arm. The #278 wrap-up flagged that its
  load axis had been taken asymmetrically -- p50 on one side, p99 on the other
  -- which made the two sides incomparable and the load verdict unusable. The
  schema makes both mandatory; this script fills both mandatory fields in both
  arms rather than filling p99 only where a tail was expected.

* An explicit LOAD arm next to the idle arm, over the same pairs, sizes and
  iteration counts. Idle rows feed the cost model (p50), load rows feed the
  pressure view (p99). The load is a continuous pinned-host <-> device stream
  on both participating cards: the PCIe bus is the contended resource on this
  rig, and a load arm that does not contend for it measures nothing.

* DIRECTED send/recv, not only symmetric all_reduce. The rig is asymmetric by
  construction (full-BAR 5090 against two windowed 3080s) and Q6 of the
  re-probe asks whether a broadcast topology can exploit that. A symmetric
  collective averages the asymmetry away.

TIMING CONVENTION, stated because it is not self-evident:

* all_reduce rows carry the duration of the collective itself
  (timing="op_duration"), measured with CUDA events on rank 0.
* send_recv rows carry a DIRECTED payload transfer followed by a 4-byte ack
  (timing="directed_ack_round_trip"), timed on the sender. The constant ack
  overhead is also recorded per row as baseline_ack_us. The consumer fits an
  affine model base + per_byte*bytes, so a constant lands in `base` where it
  belongs and does not distort the per-byte term. Halving a symmetric
  ping-pong instead would have averaged exactly the asymmetry we came for.

Cards are named by PCI bus id everywhere. Bare indices are not quotable on
this rig: torch's order and NVML's order differ and the mapping shifts.

Usage:
    python s06_nccl_reference.py --out <dir>/nccl_reference.json [--dry-run]
"""

from __future__ import annotations

import argparse
import itertools
import json
import os
import socket
import subprocess
import sys
import time

os.environ.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")

KIB = 1024
MIB = 1024 * 1024

# Kept short on purpose: the whole step has a ~15 minute budget and a ladder
# that cannot be finished is worth less than a shorter one that can. Four
# decades are enough to fit base + per_byte.
SIZE_LADDER = (64 * KIB, 1 * MIB, 8 * MIB, 64 * MIB)
ITERS = {64 * KIB: 60, 1 * MIB: 50, 8 * MIB: 30, 64 * MIB: 15}
WARMUP = 5

IDLE_ARM = "idle"
LOAD_ARM = "host_stream_64mib"

WORKER_FLAG = "--worker-rank"


# ---------------------------------------------------------------------------
# statistics without a numpy dependency
# ---------------------------------------------------------------------------


def percentile(values, q: float) -> float:
    if not values:
        return float("nan")
    ordered = sorted(values)
    if len(ordered) == 1:
        return float(ordered[0])
    pos = (len(ordered) - 1) * q
    low = int(pos)
    high = min(low + 1, len(ordered) - 1)
    frac = pos - low
    return float(ordered[low] * (1 - frac) + ordered[high] * frac)


# ---------------------------------------------------------------------------
# worker: runs inside a subprocess pinned to exactly two cards
# ---------------------------------------------------------------------------


def _host_load(stop_flag, device):
    """A continuous pinned-host <-> device stream on a side stream.

    This is the foreign load. It runs on its own stream so it contends for the
    bus and the copy engines without serialising against the collective's
    stream, which is what a real neighbour process does.
    """
    import torch

    stream = torch.cuda.Stream(device=device)
    host = torch.empty(64 * MIB, dtype=torch.uint8, pin_memory=True)
    dev = torch.empty(64 * MIB, dtype=torch.uint8, device=device)
    with torch.cuda.stream(stream):
        while not stop_flag["stop"]:
            dev.copy_(host, non_blocking=True)
            host.copy_(dev, non_blocking=True)
            stream.synchronize()


def worker(rank: int, out_path: str) -> int:
    import threading

    import torch
    import torch.distributed as dist

    dist.init_process_group("nccl", rank=rank, world_size=2)
    torch.cuda.set_device(0)  # pinned by CUDA_VISIBLE_DEVICES: always cuda:0
    device = torch.device("cuda:0")

    rows = []
    stop_flag = {"stop": True}
    load_thread = None

    def start_load():
        nonlocal load_thread
        stop_flag["stop"] = False
        load_thread = threading.Thread(
            target=_host_load, args=(stop_flag, device), daemon=True
        )
        load_thread.start()

    def stop_load():
        nonlocal load_thread
        stop_flag["stop"] = True
        if load_thread is not None:
            load_thread.join(timeout=30)
            load_thread = None

    def timed_all_reduce(nbytes, iters):
        buf = torch.ones(nbytes // 4, dtype=torch.float32, device=device)
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        samples = []
        for i in range(iters + WARMUP):
            dist.barrier()
            start.record()
            dist.all_reduce(buf)
            end.record()
            torch.cuda.synchronize()
            if i >= WARMUP:
                samples.append(start.elapsed_time(end) * 1000.0)  # ms -> us
        del buf
        torch.cuda.empty_cache()
        return samples

    def timed_directed(nbytes, iters, sender: int):
        """One directed payload plus a 4-byte ack, timed on the sender."""
        payload = torch.empty(max(nbytes, 4) // 4, dtype=torch.float32, device=device)
        ack = torch.zeros(1, dtype=torch.float32, device=device)
        peer = 1 - rank
        samples = []
        for i in range(iters + WARMUP):
            dist.barrier()
            t0 = time.perf_counter()
            if rank == sender:
                dist.send(payload, peer)
                dist.recv(ack, peer)
            else:
                dist.recv(payload, peer)
                dist.send(ack, peer)
            torch.cuda.synchronize()
            t1 = time.perf_counter()
            if i >= WARMUP and rank == sender:
                samples.append((t1 - t0) * 1e6)
        del payload, ack
        torch.cuda.empty_cache()
        return samples

    pci = [os.environ["BATTERY_PCI_0"], os.environ["BATTERY_PCI_1"]]

    for arm in (IDLE_ARM, LOAD_ARM):
        if arm == LOAD_ARM:
            start_load()
            time.sleep(2)  # let the stream reach steady state before measuring

        # all_reduce: symmetric, recorded against the sorted pair
        for nbytes in SIZE_LADDER:
            samples = timed_all_reduce(nbytes, ITERS[nbytes])
            if rank == 0:
                src, dst = sorted(pci)
                rows.append(
                    {
                        "op": "all_reduce",
                        "transport": None,  # filled by the parent from the log
                        "world": 2,
                        "src_pci": src,
                        "dst_pci": dst,
                        "size_bytes": nbytes,
                        "iters": len(samples),
                        "p50_us": round(percentile(samples, 0.50), 3),
                        "p99_us": round(percentile(samples, 0.99), 3),
                        "load": arm,
                        "timing": "op_duration",
                    }
                )

        # send/recv: both directions, so the asymmetry survives
        baseline = {}
        for sender in (0, 1):
            base_samples = timed_directed(4, 30, sender)
            if rank == sender:
                baseline[sender] = round(percentile(base_samples, 0.50), 3)

        for sender in (0, 1):
            for nbytes in SIZE_LADDER:
                samples = timed_directed(nbytes, ITERS[nbytes], sender)
                if rank == sender:
                    rows.append(
                        {
                            "op": "send_recv",
                            "transport": None,
                            "world": 2,
                            "src_pci": pci[sender],
                            "dst_pci": pci[1 - sender],
                            "size_bytes": nbytes,
                            "iters": len(samples),
                            "p50_us": round(percentile(samples, 0.50), 3),
                            "p99_us": round(percentile(samples, 0.99), 3),
                            "load": arm,
                            "timing": "directed_ack_round_trip",
                            "baseline_ack_us": baseline.get(sender),
                        }
                    )

        if arm == LOAD_ARM:
            stop_load()

    # Rank 1 owns the rows it timed (the sender side of its direction), so both
    # ranks write and the parent merges. A single writer would silently drop
    # one direction.
    with open(f"{out_path}.rank{rank}", "w") as f:
        json.dump(rows, f)

    dist.barrier()
    dist.destroy_process_group()
    return 0


# ---------------------------------------------------------------------------
# parent
# ---------------------------------------------------------------------------


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def run_pair(cuda_ids, pci_ids, frag_path, timeout_s):
    port = free_port()
    env = dict(os.environ)
    env.update(
        {
            "CUDA_VISIBLE_DEVICES": f"{cuda_ids[0]},{cuda_ids[1]}",
            "CUDA_DEVICE_ORDER": "PCI_BUS_ID",
            "NCCL_DEBUG": "INFO",
            "MASTER_ADDR": "127.0.0.1",
            "MASTER_PORT": str(port),
            "BATTERY_PCI_0": pci_ids[0],
            "BATTERY_PCI_1": pci_ids[1],
        }
    )
    procs = [
        subprocess.Popen(
            [
                sys.executable,
                os.path.abspath(__file__),
                WORKER_FLAG,
                str(rank),
                "--frag",
                frag_path,
            ],
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )
        for rank in (0, 1)
    ]
    logs, status = [], "ok"
    for p in procs:
        try:
            out, _ = p.communicate(timeout=timeout_s)
            logs.append(out or "")
            if p.returncode != 0:
                status = f"rank exited {p.returncode}"
        except subprocess.TimeoutExpired:
            p.kill()
            out, _ = p.communicate()
            logs.append(out or "")
            status = f"timeout after {timeout_s}s"
    return status, "\n".join(logs)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(WORKER_FLAG, type=int, default=None, help=argparse.SUPPRESS)
    ap.add_argument("--frag", default=None, help=argparse.SUPPRESS)
    ap.add_argument("--out", default="nccl_reference.json")
    ap.add_argument(
        "--log", default=None, help="NCCL_DEBUG output goes here, not to stdout"
    )
    ap.add_argument("--timeout", type=int, default=600)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    if args.worker_rank is not None:
        return worker(args.worker_rank, args.frag)

    if args.dry_run:
        print(
            "Plan: je ungeordnetem Kartenpaar zwei gepinnte Subprozesse "
            "(CUDA_VISIBLE_DEVICES=<Paar>, NCCL_DEBUG=INFO); je Arm "
            f"({IDLE_ARM}, {LOAD_ARM}) all_reduce ueber die Leiter "
            f"{[s // KIB for s in SIZE_LADDER]} KiB plus send_recv in BEIDE "
            "Richtungen; p50 UND p99 je Zeile; Ausgabe im "
            "nccl_reference-Envelope (schema_version 1)."
        )
        return 0

    sys.path.insert(
        0,
        os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "p2p_readiness"),
    )
    from p2p_common import cuda_device_count, cuda_pci_bus_id  # noqa: E402

    repo_python = os.path.abspath(
        os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "python")
    )
    sys.path.insert(0, repo_python)
    from sglang.srt.distributed.device_communicators.htccl_path_rates import (  # noqa: E402
        new_nccl_reference_envelope,
    )
    from p2p_common import parse_nccl_transports, summarize_transport_classes  # noqa: E402

    n = cuda_device_count()
    if n < 2:
        print(f"brauche >= 2 Karten, gefunden {n}", file=sys.stderr)
        return 4
    idx_pci = {i: cuda_pci_bus_id(i) for i in range(n)}

    payload = new_nccl_reference_envelope()
    payload["host"] = os.uname().nodename
    payload["timing_conventions"] = {
        "op_duration": "Dauer des Kollektivs, CUDA-Events auf Rang 0",
        "directed_ack_round_trip": (
            "gerichtete Nutzlast plus 4-Byte-Ack, auf dem Sender gemessen; der "
            "konstante Ack-Anteil steht je Zeile in baseline_ack_us und landet "
            "im affinen Fit im base-Term"
        ),
    }
    payload["load_arms"] = {
        IDLE_ARM: "keine Fremdlast",
        LOAD_ARM: "durchgehender 64-MiB-pinned-Host<->Device-Strom auf beiden Karten",
    }
    payload["pairs_status"] = []

    log_path = args.log or (os.path.splitext(args.out)[0] + ".nccl.log")
    all_rows = []

    with open(log_path, "w") as logf:
        for a, b in itertools.combinations(range(n), 2):
            frag = f"{args.out}.frag.{a}-{b}"
            status, log = run_pair((a, b), (idx_pci[a], idx_pci[b]), frag, args.timeout)
            logf.write(f"===== pair cuda:{a} <-> cuda:{b} ({status}) =====\n{log}\n")
            transports = summarize_transport_classes(parse_nccl_transports(log))
            chosen = max(transports, key=transports.get) if transports else None

            rows = []
            for rank in (0, 1):
                part = f"{frag}.rank{rank}"
                if os.path.exists(part):
                    with open(part) as f:
                        rows.extend(json.load(f))
                    os.unlink(part)
            for row in rows:
                row["transport"] = chosen
            all_rows.extend(rows)

            payload["pairs_status"].append(
                {
                    "pci_pair": [idx_pci[a], idx_pci[b]],
                    "status": status,
                    "transport_summary": transports,
                    "rows": len(rows),
                }
            )
            print(
                f"{idx_pci[a]} <-> {idx_pci[b]}: {status}, {len(rows)} Zeilen, {transports}"
            )

    payload["rows"] = all_rows
    with open(args.out, "w") as f:
        json.dump(payload, f, indent=2)
        f.write("\n")
    print(f"geschrieben: {args.out} ({len(all_rows)} Zeilen); NCCL-Log: {log_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
