#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Symmetrischer Lastvergleich (04_OFFEN 1a) -- NCCL-Seite.

Der bisherige Lastvergleich taugt nicht fuer eine Bauentscheidung: nur eine
Groesse, NCCL p50 gegen NIC p99, und die Fremdlast traf NCCL nur indirekt
ueber PCIe/RAM. Hier wird die Last SYMMETRISCH definiert: derselbe zweite
Strom (dauerhaftes 1-MiB-send/recv-Ping-Pong zwischen demselben Kartenpaar),
getragen vom Pfad unter Test -- fuer NCCL also ueber den System-RAM (SHM),
fuer die NIC-Seite (Phase loadsym der Leiter) ueber die NIC. Gemessen wird
beidseitig p99 auf 20/80 KiB und 1 MiB.

Zwei Rollen, jede ein eigenes Zwei-Prozess-Paar mit eigenem MASTER_PORT auf
DEMSELBEN Kartenpaar:

  measure  send/recv-Ping-Pong 0<->1 je Groesse mit Zeitbudget; berichtet
           p10/p50/p90/p99 als HALBEN Round-trip (Konvention des C-Benchs)
           plus Fensterzeiten (Epoch), damit die Ueberlappung mit der Last
           offline PRUEFBAR ist statt behauptet.
  load     Ping-Pong fester Groesse in Bloecken, bis Rang 0 das Ende
           beschliesst (Zeitbudget oder Stop-Datei) und es per Broadcast
           verteilt. Je Block eine LOAD-Zeile mit Epoch und Durchsatz --
           das ist die zweite Haelfte der Ueberlappungspruefung.

Deadlock-Familie dieses Projekts (rang-lokale Bedingung vor einem
Gruppen-Kollektiv): sowohl die Rundenzahl der Messung als auch das Last-Ende
entscheidet ausschliesslich Rang 0 und verteilt sie per Broadcast.

MB/s-Konvention wie im C-Bench: Nutzlast einer Richtung durch die Zeit der
VOLLEN Runde.
"""
import argparse
import json
import os
import time

import torch
import torch.distributed as dist


def pingpong(rank, x):
    if rank == 0:
        dist.send(x, 1)
        dist.recv(x, 1)
    else:
        dist.recv(x, 0)
        dist.send(x, 0)


def role_measure(a, dev):
    out = {
        "role": "measure",
        "nccl": ".".join(map(str, torch.cuda.nccl.version())),
        "sizes": a.sizes,
    }
    for nbytes in [int(s) for s in a.sizes.split(",")]:
        x = torch.ones(nbytes // 4, dtype=torch.float32, device=dev)
        for _ in range(a.warmup):
            pingpong(a.rank, x)
        torch.cuda.synchronize()

        # Kalibrieren und Rundenzahl festlegen -- Rang 0 entscheidet,
        # Broadcast verteilt (siehe Kopf).
        t0 = time.perf_counter()
        for _ in range(50):
            pingpong(a.rank, x)
        torch.cuda.synchronize()
        per = (time.perf_counter() - t0) / 50
        n = max(50, min(50000, int(a.secs / per))) if per > 0 else 500
        t = torch.tensor([n], dtype=torch.int64, device=dev)
        dist.broadcast(t, 0)
        n = int(t.item())

        t_begin = time.time()
        lat = []
        for _ in range(n):
            t1 = time.perf_counter()
            pingpong(a.rank, x)
            torch.cuda.synchronize()
            lat.append((time.perf_counter() - t1) * 1e6)
        t_end = time.time()
        lat.sort()
        if a.rank == 0:
            # Halber Round-trip, Konvention des C-Benchs.
            out[f"sendrecv/{nbytes}"] = {
                "n": n,
                "p10": lat[n // 10] / 2,
                "p50": lat[n // 2] / 2,
                "p90": lat[(n * 9) // 10] / 2,
                "p99": lat[int(n * 0.99)] / 2,
                "t_begin": round(t_begin, 3),
                "t_end": round(t_end, 3),
            }
    if a.rank == 0 and a.out:
        with open(a.out, "w") as f:
            json.dump(out, f, indent=1)


def role_load(a, dev):
    x = torch.ones(a.load_bytes // 4, dtype=torch.float32, device=dev)
    for _ in range(a.warmup):
        pingpong(a.rank, x)
    torch.cuda.synchronize()
    blk = a.block
    t_stop = time.time() + a.secs
    flag = torch.zeros(1, dtype=torch.int64, device=dev)
    total = 0
    t_begin = time.time()
    while True:
        if a.rank == 0:
            go = 1 if (time.time() < t_stop
                       and not (a.stop_file and os.path.exists(a.stop_file))) else 0
            flag.fill_(go)
        dist.broadcast(flag, 0)
        if int(flag.item()) == 0:
            break
        tb = time.perf_counter()
        for _ in range(blk):
            pingpong(a.rank, x)
        torch.cuda.synchronize()
        dt = time.perf_counter() - tb
        total += blk
        if a.rank == 0:
            print("LOAD %.3f rounds=%d us_per_round=%.1f MBps=%.0f"
                  % (time.time(), blk, dt / blk * 1e6,
                     a.load_bytes * blk / dt / 1e6), flush=True)
    dur = time.time() - t_begin
    if a.rank == 0:
        print("LOADSUM rounds=%d secs=%.1f avg_MBps=%.0f"
              % (total, dur, (a.load_bytes * total / dur / 1e6) if dur > 0 else 0),
              flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--role", choices=("measure", "load"), required=True)
    ap.add_argument("--rank", type=int, required=True)
    ap.add_argument("--world", type=int, default=2)
    ap.add_argument("--secs", type=float, default=5.0)
    ap.add_argument("--sizes", default="20480,81920,1048576")
    ap.add_argument("--warmup", type=int, default=200)
    ap.add_argument("--load-bytes", type=int, default=1048576)
    ap.add_argument("--block", type=int, default=50)
    ap.add_argument("--stop-file", default="")
    ap.add_argument("--out", default="")
    a = ap.parse_args()

    torch.cuda.set_device(a.rank)
    dev = torch.device("cuda", a.rank)
    os.environ.setdefault("RANK", str(a.rank))
    os.environ.setdefault("WORLD_SIZE", str(a.world))
    dist.init_process_group("nccl", rank=a.rank, world_size=a.world)

    if a.role == "measure":
        role_measure(a, dev)
    else:
        role_load(a, dev)
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
