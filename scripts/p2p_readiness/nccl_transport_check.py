#!/usr/bin/env python3
"""Which transport does NCCL pick per pair, after the update?

For every unordered card pair: a 2-rank torch.distributed all_reduce with
NCCL_DEBUG=INFO, the chosen transport (P2P / SHM / NET) grepped from the log.
Before/after comparison: pass --baseline <old json> to get a per-pair diff.

NO expected outcome is encoded here. Whether NCCL P2P engages at all over a
256-MiB BAR window, and for which message sizes, is exactly what the run
determines (see verdict_diff.md).

Each pair runs in fresh subprocesses with CUDA_VISIBLE_DEVICES pinned to the
two PCI-selected cards, so ranks are process-isolated and cuda:0/cuda:1 are
unambiguous inside each worker.

Usage:
    python nccl_transport_check.py --out results/<date>/nccl_transport.json \
        [--baseline results/<older>/nccl_transport.json] [--dry-run]

Runtime: a few seconds per pair (one tiny all_reduce each).
"""

import argparse
import itertools
import json
import os
import socket
import subprocess
import sys

os.environ.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")

from p2p_common import (
    normalize_pci,
    parse_nccl_transports,
    result_envelope,
    summarize_transport_classes,
    write_json,
)

WORKER_FLAG = "--worker-rank"


def free_port():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def worker(rank: int) -> int:
    """Runs INSIDE the pinned subprocess: rank 0/1, one small all_reduce.

    CUDA_VISIBLE_DEVICES lists BOTH cards of the pair, so the rank picks its
    own card out of that pair -- rank 0 -> cuda:0, rank 1 -> cuda:1. Pinning
    both ranks to cuda:0 puts two ranks on one device and NCCL refuses the
    communicator outright ("Duplicate GPU detected"), which is what the
    2026-07-30 run recorded for all three pairs.
    """
    import torch
    import torch.distributed as dist

    torch.cuda.set_device(rank)
    dist.init_process_group("nccl", rank=rank, world_size=2)
    t = torch.ones(1024 * 256, dtype=torch.float32, device=f"cuda:{rank}")
    dist.all_reduce(t)
    torch.cuda.synchronize()
    ok = float(t[0].item()) == 2.0
    dist.destroy_process_group()
    return 0 if ok else 1


def run_pair(cuda_ids, timeout_s):
    """Launch both ranks with CUDA_VISIBLE_DEVICES=<the two cards>; collect
    NCCL_DEBUG=INFO output. A hang or crash is a recorded result."""
    port = free_port()
    env = dict(os.environ)
    env.update(
        {
            "CUDA_VISIBLE_DEVICES": f"{cuda_ids[0]},{cuda_ids[1]}",
            "CUDA_DEVICE_ORDER": "PCI_BUS_ID",
            "NCCL_DEBUG": "INFO",
            "MASTER_ADDR": "127.0.0.1",
            "MASTER_PORT": str(port),
        }
    )
    procs = [
        subprocess.Popen(
            [sys.executable, os.path.abspath(__file__), WORKER_FLAG, str(rank)],
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


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(WORKER_FLAG, type=int, default=None, help=argparse.SUPPRESS)
    ap.add_argument("--out", default="nccl_transport.json")
    ap.add_argument("--baseline", default=None)
    ap.add_argument("--timeout", type=int, default=120)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    if args.worker_rank is not None:
        return worker(args.worker_rank)

    if args.dry_run:
        print(
            "plan: per unordered pair -> 2 pinned subprocesses "
            "(CUDA_VISIBLE_DEVICES=<pair>, NCCL_DEBUG=INFO) -> tiny "
            "all_reduce -> grep transport (P2P/SHM/NET) -> JSON; "
            "--baseline gives the before/after diff"
        )
        return 0

    from p2p_common import cuda_device_count, cuda_pci_bus_id

    n = cuda_device_count()
    idx_pci = {i: cuda_pci_bus_id(i) for i in range(n)}

    pair_rows = []
    for a, b in itertools.combinations(range(n), 2):
        status, log = run_pair((a, b), args.timeout)
        rows = parse_nccl_transports(log)
        pair_rows.append(
            {
                "cuda_pair": [a, b],
                "pci_pair": [idx_pci[a], idx_pci[b]],
                "status": status,
                "transports": rows,
                "transport_summary": summarize_transport_classes(rows),
                "log_tail": log[-4000:],
            }
        )
        print(
            f"{idx_pci[a]} <-> {idx_pci[b]}: {status} "
            f"{pair_rows[-1]['transport_summary']}"
        )

    payload = result_envelope("nccl_transport")
    payload["pairs"] = pair_rows

    if args.baseline and os.path.exists(args.baseline):
        with open(args.baseline) as f:
            base = json.load(f)
        base_by_pci = {
            tuple(sorted(map(normalize_pci, p["pci_pair"]))): p.get(
                "transport_summary", {}
            )
            for p in base.get("pairs", [])
        }
        payload["diff_vs_baseline"] = [
            {
                "pci_pair": p["pci_pair"],
                "before": base_by_pci.get(
                    tuple(sorted(map(normalize_pci, p["pci_pair"])))
                ),
                "after": p["transport_summary"],
            }
            for p in pair_rows
        ]

    write_json(args.out, payload)
    print(f"written: {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
