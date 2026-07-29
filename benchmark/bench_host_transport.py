#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""HTCCL-Host-Transport gegen NCCL, all_reduce, verschraenkt im selben Lauf.

WOZU
====
Der Host-Transport (`SGLANG_HTCCL_TRANSPORT=host`, siehe
python/sglang/srt/distributed/device_communicators/htccl_host.py) bewegt die
Nutzlast ueber EIN gepinntes, portables Host-Segment und wird komplett von
zwei Kerneln getrieben -- kein Host-Sync, keine Allokation im heissen Pfad,
Fertigmeldung ueber Flaggen im gepinnten Speicher. Auf diesem Rig lag der
schlichte gepinnte Host-Weg bei 20 KiB Ping-Pong bei 7,30 us gegen 37,41 us
fuer NCCL. Dieses Programm prueft, was davon auf der KOLLEKTIV-Ebene ankommt.

    Alle Zeiten sind die VOLLE Operationsdauer eines all_reduce.
    Ein all_reduce hat keinen "halben Weg". NICHT mit den
    Punkt-zu-Punkt-Zahlen (halber Round-trip) verrechnen.

MESSDISZIPLIN (eingebaut, nicht optional)
=========================================
* **Vorlauf ist Pflicht.** Ohne ihn wurden auf diesem Rig 726 us statt 95 us
  gemessen -- Faktor 7,6 allein aus der P-State-Rampe. Das Vorlaufbudget je
  Zelle ist auf mindestens 3 s festgenagelt; ein kleinerer --warmup wird
  hochgesetzt und das Hochsetzen gemeldet.
* **Verschraenkt, nicht blockweise.** Je Runde eine kurze Serie pro Zelle,
  Reihenfolge je Runde rotiert. Drift, Fremdlast und Taktverhalten treffen
  damit alle Backends gleich.
* **Erst Korrektheit, dann Zeit.** Jede Runde beginnt mit einer
  Bekannt-Antwort-Pruefung je Zelle. Wer sie nicht besteht, wird ab dieser
  Runde nicht mehr gemessen und im Bericht mit Grund gefuehrt.
* **Tote Varianten mit Grund, nie stiller Rueckfall.** Kommt ein Transport
  nicht hoch, oder ist der aktive Transport nicht der angeforderte (htccl.py
  faellt bei manchen Namen still auf die gloo-Inline-Ebene zurueck), wird das
  Backend auf ALLEN Raengen verworfen und der Grund gedruckt. Es wird nie
  ersatzweise etwas anderes gemessen und als das Angeforderte ausgegeben.

VERGLEICHBARKEIT, ehrlich
=========================
* HTCCL `all_reduce` ist AUSSER-PLATZ (liefert einen neuen Tensor), NCCLs
  `dist.all_reduce` ist AN-PLATZ. Der Host-Transport schreibt also zusaetzlich
  ein volles Ergebnis in einen frischen Puffer. Der Vergleich ist damit, wenn
  ueberhaupt, zu Ungunsten von HTCCL verzerrt -- nicht zu seinen Gunsten.
* Die Fertigmeldung im Messpfad ist fuer BEIDE Backends dieselbe:
  `--completion sync` (Vorgabe) ruft `torch.cuda.synchronize`,
  `--completion event` pollt ein `cuda.Event`. Der Transport selbst
  synchronisiert nie mit dem Host; der Messpfad tut es nur, um einen
  Zeitstempel zu bekommen, und fuer alle Backends identisch.

DEADLOCK-VORSICHT
=================
In diesem Projekt gab es bereits einen Deadlock, weil jeder Rang seine
Rundenzahl lokal hochrechnete. Hier gilt: Pilotlauf mit FESTER, fuer alle
Raenge identischer Iterationszahl; Rang 0 rechnet daraus den Plan und
verteilt ihn per broadcast_object_list; jede Verwerfung eines Backends wird
per all_gather_object gemeinsam beschlossen. Dazu gloo-Timeout, harter
Watchdog je Rang und Gesamtzeitlimit im Elternprozess.

AUFRUF
======
    python3 benchmark/bench_host_transport.py --devices 1,2

Weitere Schalter: --sizes, --backends, --secs, --warmup, --rounds, --dtype,
--slot-mib, --completion, --out, --devices. Die Karten muessen frei sein;
das Programm verdraengt nichts und bricht ab, wenn auf einer Zielkarte mehr
als --max-used-mib belegt ist.
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

#: 20 KiB, 80 KiB, 1 MiB, 4 MiB, 16 MiB.
DEFAULT_SIZES = "20480,81920,1048576,4194304,16777216"

#: name -> (SGLANG_HTCCL_TRANSPORT-Wert, erwartete Transportklasse). Der
#: Klassenname ist die Probe gegen den stillen Rueckfall: htccl.py laesst
#: einen unbekannten oder fehlgeschlagenen Transport auf die gloo-Inline-Ebene
#: fallen, und dann misst man gloo und nennt es "host".
HTCCL_BACKENDS = {
    "htccl:host": ("host", "HTCCLHostTransport"),
    "htccl:device": ("device", "HTCCLDeviceTransport"),
    "htccl:shm": ("shm", "HTCCLShmTransport"),
}
DEFAULT_BACKENDS = "htccl:host,nccl"

#: Untergrenze des Vorlaufs je Zelle. Siehe Kopf: ohne Vorlauf Faktor 7,6.
MIN_WARMUP_S = 3.0


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
        # "bus bandwidth" der NCCL-Tests (2*(N-1)/N * Nutzlast): hier soll nur
        # eine handliche Zweitgroesse zur Latenz stehen, kein Mass fuer die
        # Leitungsauslastung.
        "mbps": (nbytes / (p50 * 1e-6)) / 1e6 if p50 == p50 and p50 else float("nan"),
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
    # Getrennte Gruppen: HTCCL braucht eine reine gloo-CPU-Gruppe, NCCL
    # bekommt seine eigene, damit die beiden Datenebenen sich nicht ueber eine
    # gemeinsame Gruppe serialisieren.
    gloo_pg = dist.new_group(ranks=list(range(world)), backend="gloo", timeout=to)
    nccl_pg = dist.new_group(ranks=list(range(world)), backend="nccl", timeout=to)

    dtype = {"bfloat16": torch.bfloat16, "float16": torch.float16,
             "float32": torch.float32}[a.dtype]
    esize = torch.tensor([], dtype=dtype).element_size()
    sizes = [int(x) for x in a.sizes.split(",")]
    for nb in sizes:
        if nb % esize:
            raise SystemExit(f"{nb} B ist kein Vielfaches von {esize} B ({a.dtype})")

    p = torch.cuda.get_device_properties(dev_ord)
    print(f"# rank{rank} -> cuda:{dev_ord} {p.name} sm_{p.major}{p.minor}", flush=True)

    # ---------------- HTCCL-Kommunikatoren aufbauen ----------------
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
            # Der Transportname kommt in htccl.py aus der Modulvariablen
            # `_TRANSPORT`, die beim Import einmal aus SGLANG_HTCCL_TRANSPORT
            # gefuellt wird; ein Prozess kann die Umgebungsvariable also nicht
            # zweimal verschieden sehen. Setzen von `_TRANSPORT` ist exakt der
            # Wert, den die Variable gesetzt haette -- gleicher Codepfad.
            htccl_mod._TRANSPORT = transport
            obj = htccl_mod.HTCCLCommunicator(cpu_group=gloo_pg, device=dev)
            got = type(obj.transport).__name__ if obj.transport is not None else None
            if got != want:
                err = (
                    f"Transport nicht aktiv (aktiv: {got or 'gloo-Inline-Ebene'}, "
                    f"erwartet: {want}); es wird NICHT ersatzweise gemessen"
                )
        except Exception as e:  # noqa: BLE001 - der Grund soll in den Bericht
            err = f"{type(e).__name__}: {e}"
        build_s = time.time() - t0
        # Uniform entscheiden: sonst misst ein Rang ein Backend, das der
        # andere gar nicht hat -> Deadlock im ersten Kollektiv.
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
            log(f"# {name}: hochgekommen in {build_s:.1f} s ({want})")

    active = [b for b in a.backends if b == "nccl" or b in comms]
    if not active:
        for k, why in dead.items():
            print(f"# {k}: NICHT hochgekommen -- {why}", flush=True)
        raise SystemExit("ABBRUCH: kein einziges Backend hochgekommen.")

    # ---------------- Puffer: einmal, nie im Messpfad ----------------
    # Zeitmesspuffer sind Nullen: NCCLs all_reduce ist an Platz, ein
    # wiederholtes Aufsummieren von Einsen laeuft in bf16 nach wenigen hundert
    # Runden in inf. Die Werte beeinflussen keinen der gemessenen DMA-/
    # Kernelpfade. Die Korrektheitspruefung nutzt eigene Puffer mit Inhalt.
    timing_buf = {nb: torch.zeros(nb // esize, dtype=dtype, device=dev)
                  for nb in sizes}
    # Bekannte Antwort: Rang r traegt (r+1) * muster bei, Summe ueber die
    # Gruppe ist also (W(W+1)/2) * muster. Kleine ganze Zahlen -> in bf16
    # exakt darstellbar, die Pruefung ist daher ein torch.equal und kein
    # Toleranzvergleich, der einen halb kaputten Datenpfad durchwinken kann.
    check_in, check_want = {}, {}
    for nb in sizes:
        n = nb // esize
        pattern = (torch.arange(n, device=dev, dtype=torch.int32) % 7 + 1).to(dtype)
        check_in[nb] = (pattern * (rank + 1)).contiguous()
        check_want[nb] = (pattern * (world * (world + 1) // 2)).contiguous()

    def do_op(nb: int, backend: str, x):
        """Eine all_reduce-Operation, Rueckgabe ist das Ergebnis."""
        if backend == "nccl":
            dist.all_reduce(x, group=nccl_pg)
            return x
        return comms[backend].all_reduce(x)

    # Einmal angelegt, nie im Messpfad: ein `torch.cuda.Event()` je Iteration
    # waere genau die Allokation im heissen Pfad, die dieses Programm messen
    # soll, dass es sie nicht gibt.
    done_ev = torch.cuda.Event()

    def wait_done() -> None:
        """Fertigmeldung, fuer alle Backends identisch."""
        if a.completion == "event":
            done_ev.record()
            while not done_ev.query():
                pass
        else:
            torch.cuda.synchronize(dev)

    cells: list[tuple[int, str]] = [(nb, be) for nb in sizes for be in active]

    def series(cell, iters: int, timed: bool) -> list[float]:
        nb, be = cell
        x = timing_buf[nb]
        out: list[float] = []
        if not timed:
            for _ in range(iters):
                do_op(nb, be, x)
            wait_done()
            return out
        for _ in range(iters):
            t0 = time.perf_counter()
            do_op(nb, be, x)
            wait_done()
            out.append((time.perf_counter() - t0) * 1e6)
        return out

    def check(cell) -> str:
        """Bekannt-Antwort-Pruefung. Leerer String = in Ordnung."""
        nb, be = cell
        # NCCL rechnet an Platz, HTCCL ausser Platz -- die Kopie liegt
        # ausserhalb jeder Zeitmessung.
        x = check_in[nb].clone()
        try:
            got = do_op(nb, be, x)
            wait_done()
        except Exception as e:  # noqa: BLE001
            return f"{type(e).__name__}: {e}"
        if got.shape != check_want[nb].shape:
            return f"Form {tuple(got.shape)} statt {tuple(check_want[nb].shape)}"
        if not torch.equal(got, check_want[nb]):
            wrong = int((got != check_want[nb]).sum())
            first = int((got != check_want[nb]).nonzero()[0][0])
            return (
                f"{wrong}/{got.numel()} Elemente falsch, erstes bei {first}: "
                f"{got[first].item()} statt {check_want[nb][first].item()}"
            )
        return ""

    # ---------------- Pilot: FESTE Iterationszahl auf allen Raengen -------
    log(f"# Pilot ({a.pilot} Iterationen je Zelle, feste Zahl auf beiden Raengen)")
    pilot: dict[tuple, float] = {}
    for cell in cells:
        dist.barrier(group=gloo_pg)
        series(cell, 3, timed=False)  # erster Kontakt / JIT / Allokator
        lat = series(cell, a.pilot, timed=True)
        pilot[cell] = sorted(lat)[len(lat) // 2] if lat else 1000.0

    # ---------------- Plan: Rang 0 legt fest, Broadcast an alle -----------
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
    log(f"# Plan von Rang 0 verteilt; geschaetzte Messdauer ~{est:.0f} s")

    # ---------------- Vorlauf (Pflicht) ----------------
    t0 = time.time()
    for cell in cells:
        dist.barrier(group=gloo_pg)
        series(cell, plan[cell][0], timed=False)
    log(f"# Vorlauf fertig in {time.time() - t0:.1f} s "
        f"(Budget {a.warmup:.1f} s je Zelle)")

    # ---------------- Messung, verschraenkt ----------------
    lat: dict[tuple, list[float]] = {c: [] for c in cells}
    live = list(cells)
    t0 = time.time()
    for rnd in range(a.rounds):
        # Erst Korrektheit, dann Zeit -- in JEDER Runde.
        verdict = {}
        for cell in live:
            dist.barrier(group=gloo_pg)
            verdict[cell] = check(cell)
        votes: list = [None] * world
        dist.all_gather_object(
            votes, {f"{c[0]}|{c[1]}": v for c, v in verdict.items()}, group=gloo_pg
        )
        drop = []
        for cell in live:
            key = f"{cell[0]}|{cell[1]}"
            why = [f"r{i}: {v[key]}" for i, v in enumerate(votes) if v.get(key)]
            if why:
                drop.append(cell)
                dead[f"{cell[1]} @ {cell[0]} B"] = (
                    f"Korrektheit in Runde {rnd + 1} gescheitert -- " + "; ".join(why)
                )
        for cell in drop:  # gemeinsam beschlossen, also auf allen Raengen gleich
            live.remove(cell)
        if not live:
            log("# ABBRUCH: keine Zelle hat die Korrektheitspruefung ueberlebt.")
            break

        # Reihum in kurzen Serien, Reihenfolge je Runde rotiert, damit kein
        # Backend immer als erstes nach der Barriere laeuft.
        order = live[rnd % len(live):] + live[: rnd % len(live)]
        for cell in order:
            dist.barrier(group=gloo_pg)
            # Eine Einschwing-Iteration je Serie wird verworfen: sie traegt
            # den Rangversatz der Barriere, nicht die Transportzeit.
            s = series(cell, plan[cell][1] + 1, timed=True)
            lat[cell].extend(s[1:])
        if rank == 0:
            print(f"# Runde {rnd + 1}/{a.rounds} nach {time.time() - t0:.1f} s",
                  flush=True)
        if time.time() - t0 > a.secs * len(cells) * 3 + 60:
            log("# Zeitbudget erschoepft, Messung wird hier beendet.")
            break
    log(f"# Messung fertig in {time.time() - t0:.1f} s")

    # ---------------- Ergebnis ----------------
    res = {c: summarize(v, c[0]) for c, v in lat.items() if v}
    peer: list = [None] * world
    dist.all_gather_object(
        peer, {f"{c[0]}|{c[1]}": v for c, v in res.items()}, group=gloo_pg
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
            "warmup_s_pro_zelle": a.warmup,
            "mess_s_pro_zelle": a.secs,
            "htccl_slot_mib": os.environ.get("SGLANG_HTCCL_SLOT_MIB", "64 (Vorgabe)"),
            "htccl_host_blocks": os.environ.get("SGLANG_HTCCL_HOST_BLOCKS",
                                                "32 (Vorgabe)"),
            "tote_varianten": dead,
        }
        for r, o in enumerate([int(x) for x in a.devices.split(",")]):
            q = torch.cuda.get_device_properties(o)
            meta["gpus"].append(f"rank{r}=cuda:{o} {q.name} sm_{q.major}{q.minor}")
        report(a, sizes, active, res, peer[1], meta)

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
def report(a, sizes, backends, res, peer_res, meta) -> None:
    print()
    print("=" * 78)
    print("HTCCL-Host-Transport gegen NCCL -- all_reduce, ein Lauf, verschraenkt")
    print("=" * 78)
    for k, v in meta.items():
        if k == "gpus":
            for g in v:
                print(f"  {g}")
        elif k == "tote_varianten":
            for kk, vv in v.items():
                print(f"  NICHT gemessen: {kk} -- {vv}")
        else:
            print(f"  {k}: {v}")
    if not meta["tote_varianten"]:
        print("  tote Varianten: keine")
    print()
    print("  Alle Zeiten in Mikrosekunden und VOLLE OPERATIONSDAUER.")
    print("  Ein all_reduce hat keinen halben Weg -- nicht mit den")
    print("  Punkt-zu-Punkt-Zahlen (halber Round-trip) verrechnen.")
    print("  MB/s = Nutzlast je Operation / p50, keine Busbandbreite.")
    print("  HTCCL rechnet AUSSER Platz, NCCL AN Platz: HTCCL schreibt")
    print("  zusaetzlich ein volles Ergebnis. Die Verzerrung geht damit")
    print("  gegen HTCCL, nicht fuer HTCCL.")
    print()

    for nb in sizes:
        rows = [(be, res[(nb, be)]) for be in backends if (nb, be) in res]
        if not rows:
            print(f"--- all_reduce, {nb} B: keine Zelle ueberlebt ---\n")
            continue
        best = min(r[1]["p50"] for r in rows)
        ref = dict(rows).get("nccl", {}).get("p50")
        print(f"--- all_reduce, {nb} B ({nb / 1024:.0f} KiB), {meta['dtype']} "
              f"---------------------")
        print(f"{'Backend':<14}{'n':>8}{'p50 us':>10}{'p99 us':>10}{'MB/s':>10}"
              f"{'x best':>8}{'x NCCL':>9}")
        for be, r in sorted(rows, key=lambda x: x[1]["p50"]):
            vs = f"{ref / r['p50']:.2f}" if ref and be != "nccl" else "-"
            print(f"{be:<14}{r['n']:>8}{r['p50']:>10.2f}{r['p99']:>10.2f}"
                  f"{r['mbps']:>10.1f}{r['p50'] / best:>8.2f}{vs:>9}")
        print()

    print("# Maschinenlesbar (VOLLE Operationsdauer, us):")
    for nb in sizes:
        for be in backends:
            r = res.get((nb, be))
            if not r:
                continue
            pr = (peer_res or {}).get(f"{nb}|{be}", {})
            print(
                f"HOSTBENCH op=all_reduce backend={be} bytes={nb} "
                f"dtype={meta['dtype']} n={r['n']} min_us={r['min']:.2f} "
                f"p50_us={r['p50']:.2f} p90_us={r['p90']:.2f} "
                f"p99_us={r['p99']:.2f} mean_us={r['mean']:.2f} "
                f"mbps={r['mbps']:.1f} "
                f"peer_p50_us={pr.get('p50', float('nan')):.2f} half_rt=no",
                flush=True,
            )
    for k, v in meta["tote_varianten"].items():
        print(f"HOSTBENCH dead={k!r} reason={v!r}", flush=True)

    if a.out:
        blob = {"meta": meta,
                "rank0": {f"{c[0]}|{c[1]}": v for c, v in res.items()},
                "rank1": peer_res}
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
    ap.add_argument("--sizes", default=DEFAULT_SIZES)
    ap.add_argument("--dtype", default="bfloat16",
                    choices=["bfloat16", "float16", "float32"])
    ap.add_argument("--backends", default=DEFAULT_BACKENDS,
                    help="Komma-Liste aus nccl," + ",".join(HTCCL_BACKENDS))
    ap.add_argument("--secs", type=float, default=3.0,
                    help="Messbudget je Zelle in Sekunden (ueber alle Runden)")
    ap.add_argument("--warmup", type=float, default=MIN_WARMUP_S,
                    help=f"Vorlaufbudget je Zelle in Sekunden (Minimum "
                         f"{MIN_WARMUP_S:.0f})")
    ap.add_argument("--rounds", type=int, default=10,
                    help="Verschraenkungsrunden; je Runde eine kurze Serie je Zelle")
    ap.add_argument("--pilot", type=int, default=15,
                    help="feste Pilotiterationen je Zelle (Rundenzahl-Schaetzung)")
    ap.add_argument("--completion", default="sync", choices=["sync", "event"],
                    help="Fertigmeldung im Messpfad, fuer alle Backends gleich")
    ap.add_argument("--slot-mib", type=int, default=32,
                    help="SGLANG_HTCCL_SLOT_MIB fuer die Kindprozesse; muss "
                         "mindestens die groesste Nutzlast fassen")
    ap.add_argument("--max-secs", type=float, default=2400.0,
                    help="harter Watchdog je Rang")
    ap.add_argument("--pg-timeout", type=float, default=180.0)
    ap.add_argument("--port", type=int, default=29593)
    ap.add_argument("--sglang-python", default="/spinning/wt-gdr-loadsym/python")
    ap.add_argument("--out", default="")
    ap.add_argument("--max-used-mib", type=int, default=64)
    a = ap.parse_args()
    a.backends = [b for b in a.backends.split(",") if b]
    for b in a.backends:
        if b != "nccl" and b not in HTCCL_BACKENDS:
            raise SystemExit(f"unbekanntes Backend {b!r}")
    if a.warmup < MIN_WARMUP_S:
        print(f"# Vorlauf von {a.warmup:.1f} s auf {MIN_WARMUP_S:.1f} s "
              f"hochgesetzt: ohne Vorlauf wurden auf diesem Rig 726 us statt "
              f"95 us gemessen (Faktor 7,6 aus der P-State-Rampe).", flush=True)
        a.warmup = MIN_WARMUP_S

    if a.rank >= 0:
        return run_rank(a)

    max_bytes = max(int(x) for x in a.sizes.split(","))
    if max_bytes > a.slot_mib * 1024 * 1024:
        # Kein stiller Rueckfall: der Host-Transport zerlegt zwar groessere
        # Nutzlasten in Slot-Portionen, aber dann misst man eine Kette und
        # nennt sie eine Operation. Lieber den Slot passend waehlen.
        raise SystemExit(
            f"ABBRUCH: groesste Nutzlast {max_bytes} B passt nicht in einen "
            f"Slot von {a.slot_mib} MiB. --slot-mib erhoehen."
        )

    devices = [int(x) for x in a.devices.split(",")]
    preflight(devices, a.max_used_mib)

    env = {**os.environ,
           "CUDA_DEVICE_ORDER": "PCI_BUS_ID",
           "MASTER_ADDR": "127.0.0.1",
           "MASTER_PORT": str(a.port),
           "SGLANG_HTCCL_SLOT_MIB": str(a.slot_mib),
           "PYTHONPATH": a.sglang_python + ":" + os.environ.get("PYTHONPATH", ""),
           "PYTHONUNBUFFERED": "1"}
    base = [sys.executable, os.path.abspath(__file__)]
    for k, v in vars(a).items():
        if k == "rank":
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
        for pr in procs:
            left = max(1.0, deadline - time.time())
            rc |= abs(pr.wait(timeout=left))
    except subprocess.TimeoutExpired:
        sys.stderr.write("\nABBRUCH: Gesamtzeitlimit ueberschritten, "
                         "beende die eigenen Raenge.\n")
        for pr in procs:
            pr.kill()
        rc = 4
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
