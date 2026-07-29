#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""HTCCL gegen NCCL auf der KOLLEKTIV-Ebene, verschraenkt im selben Lauf.

WOZU
====
Fuer den neuen GPU-zu-GPU-Direktpfad (BAR1) und fuer ein nacktes Ping-Pong
ueber gepinnten Host-Speicher liegen Punkt-zu-Punkt-Zahlen vor. Was fehlt,
sind die beiden Wege, die im Betrieb wirklich laufen: HTCCL (der eigene
Transport dieses Forks) und NCCL. Diese Sonde liefert genau die.

WARUM KOLLEKTIVE UND KEIN send/recv
===================================
HTCCL implementiert ausschliesslich Kollektive (all_reduce, broadcast,
all_gather, reduce_scatter) und KEIN send/recv. Ein Vergleich mit dem
Punkt-zu-Punkt-Ping-Pong waere deshalb schief. Gemessen wird daher
all_reduce gegen all_reduce und broadcast gegen broadcast.

    Alle Zeiten sind die VOLLE Operationsdauer.
    Ein all_reduce hat keinen "halben Weg".
    NICHT mit den Punkt-zu-Punkt-Zahlen (halber Round-trip) verrechnen.

AUFBAU
======
Zwei Raenge auf zwei Karten (Vorgabe: CUDA-Ordinals 1 und 2 unter
CUDA_DEVICE_ORDER=PCI_BUS_ID, also RTX 5090 0a:00.0 und RTX 3080 0b:00.0 --
beide am CPU-Root-Port). Backends:

    nccl           torch.distributed ueber die NCCL-Prozessgruppe
    htccl:device   HTCCLCommunicator, SGLANG_HTCCL_TRANSPORT=device
    htccl:shm      HTCCLCommunicator, SGLANG_HTCCL_TRANSPORT=shm

Beide HTCCL-Transporte werden IM SELBEN PROZESS aufgebaut. Der Transportname
kommt in htccl.py aus der Modulvariablen `_TRANSPORT`, die beim Import einmal
aus SGLANG_HTCCL_TRANSPORT gefuellt und in HTCCLCommunicator.__init__ gelesen
wird; ein Prozess kann die Umgebungsvariable also nicht zweimal verschieden
sehen. Deshalb wird `_TRANSPORT` vor jeder Konstruktion gesetzt -- exakt der
Wert, den die Umgebungsvariable gesetzt haette, gleicher Codepfad.

VERSCHRAENKUNG
==============
Nicht Block fuer Block, sondern reihum: pro Runde eine kurze Serie je
(Operation, Groesse, Backend), dann die naechste Runde. Drift und Fremdlast
treffen so alle Backends gleich.

DEADLOCK-VORSICHT
=================
In diesem Projekt gab es bereits einen Deadlock, weil jeder Rang seine
Rundenzahl lokal aus dem Warmup hochrechnete und die Raenge dann verschieden
viele Kollektive absetzten. Hier gilt:

  * Ein Pilotlauf mit FESTER, fuer alle Raenge identischer Iterationszahl.
  * Rang 0 rechnet daraus den kompletten Plan (Warmup- und Messiterationen
    je Zelle) und verteilt ihn per broadcast_object_list. Kein Rang leitet
    eine Rundenzahl selbst ab.
  * Kommt ein Backend auf einem Rang nicht hoch, wird das per all_gather_object
    bekannt gemacht und das Backend auf ALLEN Raengen verworfen.
  * Kein unbegrenztes Warten: gloo-Timeout, harter Watchdog je Rang, Timeout
    im Elternprozess.
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

# Muss VOR dem torch-Import stehen, sonst zaehlt die Reihenfolge der
# Rechenleistung und nicht die des PCI-Busses.
os.environ.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")

BACKENDS = ("nccl", "htccl:device", "htccl:shm")
OPS = ("all_reduce", "broadcast")


# ----------------------------------------------------------------------
# Watchdog: lieber ein klarer Abbruch als ein haengender Rang.
# ----------------------------------------------------------------------
def arm_watchdog(seconds: float, tag: str) -> None:
    def boom() -> None:
        sys.stderr.write(
            f"\n[{tag}] ABBRUCH: Watchdog nach {seconds:.0f} s. "
            f"Ein Kollektiv ist nicht zurueckgekommen. Traceback:\n"
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
        # Nutzlast je Operation geteilt durch die Dauer. Bewusst NICHT die
        # "bus bandwidth" der NCCL-Tests (2*(N-1)/N * Nutzlast): hier soll
        # nur eine handliche Zweitgroesse zur Latenz stehen, kein Mass fuer
        # die Leitungsauslastung.
        "mbps": (nbytes / (p50 * 1e-6)) / 1e6 if p50 and p50 == p50 else float("nan"),
    }


# ----------------------------------------------------------------------
# Kindprozess: ein Rang
# ----------------------------------------------------------------------
def run_rank(a: argparse.Namespace) -> int:
    import torch
    import torch.distributed as dist

    rank = a.rank
    world = 2
    tag = f"r{rank}"
    arm_watchdog(a.max_secs, tag)

    dev_ord = [int(x) for x in a.devices.split(",")][rank]
    torch.cuda.set_device(dev_ord)
    dev = torch.device("cuda", dev_ord)
    props = torch.cuda.get_device_properties(dev_ord)

    def log(msg: str) -> None:
        if rank == 0:
            print(msg, flush=True)

    def warn(msg: str) -> None:
        sys.stderr.write(f"[{tag}] {msg}\n")
        sys.stderr.flush()

    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ["RANK"] = str(rank)
    os.environ["WORLD_SIZE"] = str(world)

    to = timedelta(seconds=a.pg_timeout)
    dist.init_process_group(
        backend="cpu:gloo,cuda:nccl", rank=rank, world_size=world, timeout=to,
        device_id=dev,
    )
    # Getrennte Gruppen: HTCCL braucht eine reine gloo-CPU-Gruppe, NCCL
    # bekommt seine eigene, damit die beiden Datenebenen sich nicht ueber
    # eine gemeinsame Gruppe serialisieren.
    gloo_pg = dist.new_group(ranks=list(range(world)), backend="gloo", timeout=to)
    nccl_pg = dist.new_group(ranks=list(range(world)), backend="nccl", timeout=to)

    dtype = {"bfloat16": torch.bfloat16, "float32": torch.float32}[a.dtype]
    esize = torch.tensor([], dtype=dtype).element_size()
    sizes = [int(x) for x in a.sizes.split(",")]
    for nb in sizes:
        if nb % esize:
            raise SystemExit(f"{nb} B ist kein Vielfaches von {esize} B ({a.dtype})")

    log(f"# rank{rank} -> cuda:{dev_ord} {props.name} sm_{props.major}{props.minor}")

    # ---------------- HTCCL-Kommunikatoren aufbauen ----------------
    sys.path.insert(0, a.sglang_python)
    import sglang.srt.distributed.device_communicators.htccl as htccl_mod

    comms: dict[str, object] = {}
    fail_reason: dict[str, str] = {}
    for name in ("device", "shm"):
        key = f"htccl:{name}"
        if key not in a.backends:
            continue
        err = ""
        obj = None
        t0 = time.time()
        try:
            htccl_mod._TRANSPORT = name  # == SGLANG_HTCCL_TRANSPORT=<name>
            obj = htccl_mod.HTCCLCommunicator(cpu_group=gloo_pg, device=dev)
            # Der Communicator faellt bei einem nicht verfuegbaren Transport
            # still auf die inline-gloo-Ebene zurueck. Das waere dann NICHT
            # der Transport, der gemessen werden soll -- also pruefen.
            got = type(obj.transport).__name__ if obj.transport is not None else None
            want = {"device": "HTCCLDeviceTransport", "shm": "HTCCLShmTransport"}[name]
            if got != want:
                err = (
                    f"Transport nicht aktiv (aktiv: {got or 'gloo-Inline-Ebene'}); "
                    f"htccl.py faellt bei Aufbaufehlern still auf gloo zurueck"
                )
        except Exception as e:  # noqa: BLE001 - Grund soll in den Bericht
            err = f"{type(e).__name__}: {e}"
        build_s = time.time() - t0
        # Uniform entscheiden: sonst misst ein Rang ein Backend, das der
        # andere gar nicht hat -> Deadlock im ersten Kollektiv.
        votes: list = [None] * world
        dist.all_gather_object(votes, err, group=gloo_pg)
        bad = [f"r{i}: {v}" for i, v in enumerate(votes) if v]
        if bad:
            fail_reason[key] = "; ".join(bad)
            if obj is not None:
                try:
                    obj.close()
                except Exception:  # noqa: BLE001
                    pass
        else:
            comms[key] = obj
            log(f"# {key}: hochgekommen in {build_s:.1f} s "
                f"({type(obj.transport).__name__})")

    active = [b for b in a.backends if b == "nccl" or b in comms]
    for k, why in fail_reason.items():
        log(f"# {k}: NICHT hochgekommen -- {why}")

    # ---------------- Puffer und Operationen ----------------
    # Nullen als Inhalt: NCCLs all_reduce ist in-place, ein wiederholtes
    # Aufsummieren von Einsen laeuft in bf16 nach wenigen hundert Runden in
    # inf. Die Datenwerte beeinflussen keine der gemessenen DMA-/Kernelpfade.
    bufs = {nb: torch.zeros(nb // esize, dtype=dtype, device=dev) for nb in sizes}

    def make_op(op: str, nb: int, backend: str):
        x = bufs[nb]
        if backend == "nccl":
            if op == "all_reduce":
                return lambda: dist.all_reduce(x, group=nccl_pg)
            return lambda: dist.broadcast(x, src=0, group=nccl_pg)
        comm = comms[backend]
        if op == "all_reduce":
            return lambda: comm.all_reduce(x)
        return lambda: comm.broadcast(x, src=0)

    cells: list[tuple[str, int, str]] = [
        (op, nb, be) for op in a.ops for nb in sizes for be in active
    ]
    ops_cache = {c: make_op(*c) for c in cells}

    def series(cell, iters: int, timed: bool) -> list[float]:
        fn = ops_cache[cell]
        out: list[float] = []
        if not timed:
            for _ in range(iters):
                fn()
            torch.cuda.synchronize(dev)
            return out
        for _ in range(iters):
            t0 = time.perf_counter()
            fn()
            torch.cuda.synchronize(dev)
            out.append((time.perf_counter() - t0) * 1e6)
        return out

    # ---------------- Pilot: FESTE Iterationszahl auf allen Raengen -------
    log(f"# Pilot ({a.pilot} Iterationen je Zelle, feste Zahl auf beiden Raengen)")
    pilot: dict[tuple, float] = {}
    for cell in cells:
        dist.barrier(group=gloo_pg)
        series(cell, 3, timed=False)          # erster Kontakt / Allokation
        lat = series(cell, a.pilot, timed=True)
        pilot[cell] = sorted(lat)[len(lat) // 2] if lat else 1000.0

    # ---------------- Plan: Rang 0 legt fest, Broadcast an alle -----------
    plan_box: list = [None]
    if rank == 0:
        plan = {}
        per_round = max(1.0, a.secs) / max(1, a.rounds)
        for cell in cells:
            us = max(pilot[cell], 1.0)
            warm = int(min(20000, max(20, a.warmup * 1e6 / us)))
            meas = int(min(20000, max(5, per_round * 1e6 / us)))
            plan[cell] = (warm, meas)
        plan_box[0] = plan
    dist.broadcast_object_list(plan_box, src=0, group=gloo_pg)
    plan = plan_box[0]

    est = sum((w + a.rounds * (m + 1)) * pilot[c] for c, (w, m) in plan.items()) / 1e6
    log(f"# Plan von Rang 0 verteilt; geschaetzte Messdauer ~{est:.0f} s")

    # ---------------- Warmup ----------------
    # Fehlende Aufwaermung hat auf diesem Rig schon Werte um Faktor 7
    # verfaelscht (JIT, Allokator, Clock/Power-State, erste Registrierung).
    t0 = time.time()
    for cell in cells:
        dist.barrier(group=gloo_pg)
        series(cell, plan[cell][0], timed=False)
    log(f"# Warmup fertig in {time.time() - t0:.1f} s")

    # ---------------- Messung, verschraenkt ----------------
    lat: dict[tuple, list[float]] = {c: [] for c in cells}
    t0 = time.time()
    for rnd in range(a.rounds):
        for cell in cells:
            dist.barrier(group=gloo_pg)
            # Eine Einschwing-Iteration je Serie wird verworfen: sie traegt
            # den Rangversatz der Barriere, nicht die Transportzeit.
            s = series(cell, plan[cell][1] + 1, timed=True)
            lat[cell].extend(s[1:])
        if rank == 0:
            print(f"# Runde {rnd + 1}/{a.rounds} nach {time.time() - t0:.1f} s",
                  flush=True)
    log(f"# Messung fertig in {time.time() - t0:.1f} s")

    # ---------------- Ergebnis ----------------
    res = {c: summarize(v, c[1]) for c, v in lat.items()}
    peer = [None] * world
    dist.all_gather_object(peer, {f"{c[0]}|{c[1]}|{c[2]}": v for c, v in res.items()},
                           group=gloo_pg)

    if rank == 0:
        meta = {
            "dtype": a.dtype,
            "devices": a.devices,
            "gpus": [],
            "nccl": ".".join(map(str, torch.cuda.nccl.version())),
            "torch": torch.__version__,
            "rounds": a.rounds,
            "failed_backends": fail_reason,
            "htccl_slot_mib": os.environ.get("SGLANG_HTCCL_SLOT_MIB", "64 (Vorgabe)"),
        }
        for r, o in enumerate([int(x) for x in a.devices.split(",")]):
            p = torch.cuda.get_device_properties(o)
            meta["gpus"].append(f"rank{r}=cuda:{o} {p.name} sm_{p.major}{p.minor}")
        report(a, cells, res, peer[1], meta)

    for c in comms.values():
        try:
            c.close()
        except Exception:  # noqa: BLE001
            pass
    dist.barrier(group=gloo_pg)
    dist.destroy_process_group()
    return 0


# ----------------------------------------------------------------------
# Bericht (nur Rang 0)
# ----------------------------------------------------------------------
def report(a, cells, res, peer_res, meta) -> None:
    sizes = sorted({c[1] for c in cells})
    ops = [o for o in a.ops]
    backends = []
    for c in cells:
        if c[2] not in backends:
            backends.append(c[2])

    print()
    print("=" * 78)
    print("HTCCL gegen NCCL -- Kollektive, zwei Raenge, ein Lauf, verschraenkt")
    print("=" * 78)
    for k, v in meta.items():
        if k == "gpus":
            for g in v:
                print(f"  {g}")
        elif k == "failed_backends":
            for kk, vv in v.items():
                print(f"  nicht hochgekommen: {kk} -- {vv}")
        else:
            print(f"  {k}: {v}")
    print()
    print("  Alle Zeiten in Mikrosekunden und VOLLE OPERATIONSDAUER.")
    print("  Ein all_reduce hat keinen halben Weg -- nicht mit den")
    print("  Punkt-zu-Punkt-Zahlen (halber Round-trip) verrechnen.")
    print("  MB/s = Nutzlast je Operation / p50, keine Busbandbreite.")
    if "htccl:shm" in backends and "broadcast" in ops:
        print()
        print("  ACHTUNG zu 'broadcast / htccl:shm': der shm-Transport bedient")
        print("  laut HTCCLShmTransport.HTCCL_OPS nur all_reduce. Ein broadcast")
        print("  faellt dort auf die host-gestagte gloo-Ebene in htccl.py zurueck.")
        print("  Die Zeile misst also NICHT den shm-Datenpfad, sondern gloo.")
    print()

    for op in ops:
        for nb in sizes:
            rows = [(be, res[(op, nb, be)]) for be in backends if (op, nb, be) in res]
            if not rows:
                continue
            best = min(r[1]["p50"] for r in rows)
            print(f"--- {op}, {nb} B ({nb / 1024:.0f} KiB), {meta['dtype']} "
                  f"---------------------------")
            print(f"{'Backend':<14}{'n':>7}{'p50 us':>10}{'p99 us':>10}"
                  f"{'MB/s':>10}{'Faktor':>9}")
            for be, r in sorted(rows, key=lambda x: x[1]["p50"]):
                print(f"{be:<14}{r['n']:>7}{r['p50']:>10.2f}{r['p99']:>10.2f}"
                      f"{r['mbps']:>10.1f}{r['p50'] / best:>9.2f}")
            print()

    print("# Maschinenlesbar (VOLLE Operationsdauer, us):")
    for op in ops:
        for nb in sizes:
            rows = [(be, res[(op, nb, be)]) for be in backends if (op, nb, be) in res]
            if not rows:
                continue
            best = min(r[1]["p50"] for r in rows)
            for be, r in rows:
                pr = (peer_res or {}).get(f"{op}|{nb}|{be}", {})
                print(
                    f"COLLDATA op={op} backend={be} bytes={nb} dtype={meta['dtype']} "
                    f"n={r['n']} min_us={r['min']:.2f} p50_us={r['p50']:.2f} "
                    f"p90_us={r['p90']:.2f} p99_us={r['p99']:.2f} "
                    f"mean_us={r['mean']:.2f} mbps={r['mbps']:.1f} "
                    f"factor_vs_best={r['p50'] / best:.2f} "
                    f"peer_p50_us={pr.get('p50', float('nan')):.2f} half_rt=no",
                    flush=True,
                )

    if a.out:
        blob = {
            "meta": meta,
            "rank0": {f"{c[0]}|{c[1]}|{c[2]}": v for c, v in res.items()},
            "rank1": peer_res,
        }
        os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
        with open(a.out, "w") as f:
            json.dump(blob, f, indent=1)
        print(f"\n# JSON: {a.out}")


# ----------------------------------------------------------------------
# Elternprozess
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
            print(f"# GPU {idx} {name} {bus}: {used} MiB belegt")
            if int(used) > max_used_mib:
                busy.append(f"GPU {idx} ({bus}) haelt {used} MiB")
    apps = subprocess.run(
        ["nvidia-smi", "--query-compute-apps=pid,used_memory,gpu_bus_id",
         "--format=csv,noheader"],
        capture_output=True, text=True, check=True,
    ).stdout.strip()
    if apps:
        print(f"# Fremde Rechenprozesse:\n{apps}")
    if busy:
        raise SystemExit(
            "ABBRUCH: Zielkarten sind nicht frei -- " + "; ".join(busy)
            + ". Es wird nichts verdraengt."
        )


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--rank", type=int, default=-1, help="intern: Kindprozess")
    ap.add_argument("--devices", default="1,2",
                    help="CUDA-Ordinals unter PCI_BUS_ID, rank0,rank1")
    ap.add_argument("--sizes", default="20480,81920,1048576")
    ap.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float32"])
    ap.add_argument("--ops", default="all_reduce,broadcast")
    ap.add_argument("--backends", default=",".join(BACKENDS))
    ap.add_argument("--secs", type=float, default=2.5,
                    help="Messbudget je Zelle in Sekunden (ueber alle Runden)")
    ap.add_argument("--warmup", type=float, default=0.7,
                    help="Warmup-Budget je Zelle in Sekunden")
    ap.add_argument("--rounds", type=int, default=8,
                    help="Verschraenkungsrunden; je Runde eine kurze Serie je Zelle")
    ap.add_argument("--pilot", type=int, default=15,
                    help="feste Pilotiterationen je Zelle (Rundenzahl-Schaetzung)")
    ap.add_argument("--max-secs", type=float, default=1500.0,
                    help="harter Watchdog je Rang")
    ap.add_argument("--pg-timeout", type=float, default=120.0)
    ap.add_argument("--port", type=int, default=29591)
    ap.add_argument("--sglang-python",
                    default="/spinning/wt-gdr-loadsym/python")
    ap.add_argument("--out", default="")
    ap.add_argument("--max-used-mib", type=int, default=64)
    a = ap.parse_args()
    a.ops = [o for o in a.ops.split(",") if o]
    a.backends = [b for b in a.backends.split(",") if b]

    if a.rank >= 0:
        return run_rank(a)

    devices = [int(x) for x in a.devices.split(",")]
    preflight(devices, a.max_used_mib)

    env = {**os.environ,
           "CUDA_DEVICE_ORDER": "PCI_BUS_ID",
           "MASTER_ADDR": "127.0.0.1",
           "MASTER_PORT": str(a.port),
           "PYTHONPATH": a.sglang_python + ":" + os.environ.get("PYTHONPATH", ""),
           "PYTHONUNBUFFERED": "1"}
    base = [sys.executable, os.path.abspath(__file__)]
    for k, v in vars(a).items():
        if k in ("rank",):
            continue
        key = "--" + k.replace("_", "-")
        val = ",".join(map(str, v)) if isinstance(v, list) else str(v)
        if val:
            base += [key, val]
    procs = [subprocess.Popen(base + ["--rank", str(r)], env=env)
             for r in range(2)]
    rc = 0
    deadline = time.time() + a.max_secs + 120
    try:
        for p in procs:
            left = max(1.0, deadline - time.time())
            rc |= abs(p.wait(timeout=left))
    except subprocess.TimeoutExpired:
        sys.stderr.write("\nABBRUCH: Gesamtzeitlimit ueberschritten, "
                         "beende die eigenen Raenge.\n")
        for p in procs:
            p.kill()
        rc = 4
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
