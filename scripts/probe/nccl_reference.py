#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""NCCL-Referenz: der HEUTE gefahrene Weg zwischen zwei Karten dieses Rigs.

Wozu: die Cross-Rig-/Intra-Rig-Leiter misst den NIC-Relay-Pfad (GPU -> CX-4 ->
GPU) direkt gegen Host-Staging ueber dieselbe NIC. Beides ist NICHT der Weg,
den der Betrieb heute nimmt: dieses Rig kann kein GPUDirect-P2P, also schiebt
NCCL zwischen zwei Karten ueber den System-RAM (SHM-Transport). Ohne diese
Zahl ist "schlaegt der Direktpfad den echten Pfad" nicht beantwortbar -- der
Staging-Arm der Leiter laeuft ueber die NIC und ist eben kein Ersatz dafuer.

Gemessen werden, auf denselben Groessen wie die Leiter (20/80 KiB, 1 MiB) und
mit demselben Zeitbudget je Punkt:

  all_reduce   das Kollektiv, wie es im Betrieb vorkommt. Die Zahl ist die
               VOLLE Operationsdauer, kein halber Round-trip -- ein
               all_reduce hat keinen "halben" Weg. Nicht mit den
               Punkt-zu-Punkt-Zahlen der Leiter verrechnen.
  send/recv    Punkt-zu-Punkt-Ping-Pong Rang 0 <-> Rang 1, berichtet als
               HALBER Round-trip -- exakt die Konvention des C-Benchs, damit
               diese eine Zeile direkt vergleichbar ist. Bei world=3 wartet
               Rang 2 an einer Barriere; gemessen wird bewusst dieselbe
               Paarstrecke, damit sich zwischen world=2 und world=3 nur die
               Gruppengroesse aendert.

Rang r bindet Karte r (torch.cuda.set_device). Rang 0 schreibt das Ergebnis
als JSON.
"""
import argparse
import json
import os
import time

import torch
import torch.distributed as dist


def bench(fn, secs, warmup=20, dev=None, bcast_group=None):
    """fn() einmal = eine Runde. Erst warmup Runden, dann nach Zeitbudget.

    Die Rundenzahl wird NICHT vorab festgelegt, sondern aus dem Warmup
    hochgerechnet und dann fest gefahren -- alle Raenge muessen exakt gleich
    viele Kollektive absetzen, sonst haengt der letzte in seinem eigenen.
    """
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    per = (time.perf_counter() - t0) / warmup
    n = max(20, min(20000, int(secs / per))) if per > 0 else 100
    # ALLE Raenge muessen exakt dieselbe Rundenzahl fahren. Jeder Rang
    # kalibriert lokal und kommt auf eine leicht andere Zahl -- wer mehr
    # Runden faehrt, wartet am Ende ewig auf einen Partner, der schon fertig
    # ist. Genau dieser Deadlock ist hier einmal passiert; deshalb legt Rang 0
    # die Zahl fest und verteilt sie.
    t = torch.tensor([n], dtype=torch.int64, device=dev)
    dist.broadcast(t, 0, group=bcast_group)
    n = int(t.item())

    lat = []
    for _ in range(n):
        t = time.perf_counter()
        fn()
        torch.cuda.synchronize()
        lat.append((time.perf_counter() - t) * 1e6)
    lat.sort()
    return {
        "n": n,
        "p10": lat[n // 10],
        "p50": lat[n // 2],
        "p90": lat[(n * 9) // 10],
        "p99": lat[int(n * 0.99)],
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rank", type=int, required=True)
    ap.add_argument("--world", type=int, required=True)
    ap.add_argument("--secs", type=float, default=5.0)
    ap.add_argument("--sizes", default="20480,81920,1048576")
    ap.add_argument("--out", default="")
    a = ap.parse_args()

    torch.cuda.set_device(a.rank)
    dev = torch.device("cuda", a.rank)
    os.environ.setdefault("RANK", str(a.rank))
    os.environ.setdefault("WORLD_SIZE", str(a.world))
    dist.init_process_group("nccl", rank=a.rank, world_size=a.world)

    pair_group = dist.new_group([0, 1])
    out = {"world": a.world, "nccl": ".".join(map(str, torch.cuda.nccl.version()))}
    for nbytes in [int(x) for x in a.sizes.split(",")]:
        n_el = nbytes // 4
        x = torch.ones(n_el, dtype=torch.float32, device=dev)

        dist.barrier()
        r = bench(lambda: dist.all_reduce(x), a.secs, dev=dev)
        if a.rank == 0:
            out[f"all_reduce/{nbytes}"] = r

        # Punkt-zu-Punkt-Ping-Pong 0 <-> 1; alle anderen warten.
        dist.barrier()
        if a.rank in (0, 1):
            def pp(_x=x):
                if a.rank == 0:
                    dist.send(_x, 1)
                    dist.recv(_x, 1)
                else:
                    dist.recv(_x, 0)
                    dist.send(_x, 0)
            r2 = bench(pp, a.secs, dev=dev, bcast_group=pair_group)
            if a.rank == 0:
                # halber Round-trip, Konvention des C-Benchs
                out[f"sendrecv/{nbytes}"] = {k: (v / 2 if k != "n" else v)
                                             for k, v in r2.items()}
        dist.barrier()

    if a.rank == 0 and a.out:
        with open(a.out, "w") as f:
            json.dump(out, f, indent=1)
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
