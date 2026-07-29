#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Der Beleg: laesst sich ein ``all_reduce`` ueber BAR1 aufzeichnen und
mehrfach wiedergeben -- und stimmen die Bytes nach JEDER Wiedergabe?

    python benchmark/bar1_graph_check.py 1,2,3 [port]

Ohne Server, ohne Modell, ohne Planer. Gebaut wird ``HTCCLBar1Transport``
direkt, wie in ``bar1_diag.py``, damit eine Ausnahme samt Rueckverfolgung
sichtbar bleibt statt in ein ``logger.info`` uebersetzt zu werden.

Warum es dieses Programm gibt
-----------------------------
``bar1`` steht nicht in ``CAPTURABLE_HTCCL_TRANSPORTS``. Der Riegel ist
vorsorglich gesetzt: der Datenweg beruehrt den Host nicht, die Rundennummer
liegt in einem Geraetewort (``htccl_bar1_ext.py``, Kopfkommentar) und die
Peer-Zeiger stehen seit dem Bootstrap fest -- capturable **by construction**.
Was fehlte, war der Beleg. Diesen liefert dieses Programm, und erst danach
darf ``SGLANG_HTCCL_GRAPH_FREIGABE=1`` gesetzt werden.

Was genau geprueft wird
-----------------------
1. **Beide Kernvarianten einzeln.** ``1blk`` ist ein gewoehnlicher
   ``<<<1, threads>>>``-Start. ``gitter`` ist ein
   ``cudaLaunchCooperativeKernel`` mit ``grid.sync()``. Die Kopfdateien auf
   diesem Rig (CUDA 12.9) sagen zu ``CU_LAUNCH_ATTRIBUTE_COOPERATIVE``
   ausdruecklich "Valid for graph nodes, launches" (``cuda.h:2043``), also
   ist ein cooperative Start als Graphknoten darstellbar -- ob der Treiber
   ihn auch aus einem Stream-Capture heraus annimmt, steht dort nicht. Fall
   ``gitter`` beantwortet genau das und weist den Fehlercode aus, wenn nicht.

2. **Byte-Beleg nach JEDER Wiedergabe, nicht nur nach der ersten.** Jede
   Wiedergabe bekommt eine ANDERE Eingabe. Eine Aufzeichnung, die eine
   Flagge, eine Rundennummer oder einen Ringplatz eingebrannt hat, liefert
   beim ersten Mal richtig und danach falsch -- das ist der ganze Punkt.

3. **Mehrere Graphen.** sglang zeichnet je Stapelgroesse einen Graphen auf.
   Fall ``zwei-graphen`` zeichnet zwei auf und gibt sie ABWECHSELND wieder.
   Damit faellt auf, wenn zwei Aufzeichnungen sich denselben Flaggen- oder
   Ergebnisplatz teilen.

4. **Der Vorbehalt selbst.** Fall ``vorbehalt`` prueft, dass ``_kern``
   oberhalb der gitter-Schwelle unter Aufzeichnung wirklich auf ``1blk``
   ausweicht, statt zu scheitern.

Jeder Fall laeuft in FRISCHEN Prozessen. Eine misslungene Aufzeichnung
laesst den Strom im Capture-Zustand zurueck und macht alles danach
unbrauchbar; ein Fall, der einen anderen vergiftet, waere kein Beleg.
"""
from __future__ import annotations

import json
import os
import pathlib
import sys
import tempfile
import traceback

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

# Wiedergaben je Graph. Mehr als zwei, weil ein Ringplatz mit L = 2 erst
# ab der dritten Runde auffaellt.
WIEDERGABEN = 5


# ===========================================================================
# Die Faelle
# ===========================================================================
#
# `umgebung` wird VOR dem Bau des Transports gesetzt -- die Knopfwerte liest
# `HTCCLBar1Transport.__init__` einmal.
FAELLE = [
    {
        "name": "1blk-klein",
        "zweck": "Der gewoehnliche Start, kleine Nutzlast. Faellt der, "
                 "ist bar1 mit Graphen grundsaetzlich fertig.",
        "umgebung": {"SGLANG_HTCCL_BAR1_GITTER_AB": str(1 << 40)},
        "groessen": [64 << 10],
        "gate": True,
    },
    {
        "name": "1blk-gross",
        "zweck": "Derselbe Start ueber der gitter-Schwelle, damit die "
                 "Groesse und nicht die Variante die Variable ist.",
        "umgebung": {"SGLANG_HTCCL_BAR1_GITTER_AB": str(1 << 40)},
        "groessen": [8 << 20],
        "gate": True,
    },
    {
        "name": "gitter",
        "zweck": "cudaLaunchCooperativeKernel UNTER Aufzeichnung. Die eine "
                 "Frage, die sich ohne freie Karten nicht klaeren liess.",
        "umgebung": {
            "SGLANG_HTCCL_BAR1_GITTER_AB": "0",
            "SGLANG_HTCCL_BAR1_GRAPH_GITTER": "1",
        },
        "groessen": [64 << 10, 8 << 20],
        # KEIN Gate: faellt er, ist das kein Grund gegen bar1 -- dann greift
        # der Vorbehalt, und genau der wird im naechsten Fall geprueft.
        "gate": False,
    },
    {
        "name": "vorbehalt",
        "zweck": "Ueber der Schwelle, aber ohne SGLANG_HTCCL_BAR1_GRAPH_"
                 "GITTER: _kern muss aufgezeichnet auf 1blk ausweichen "
                 "statt zu scheitern.",
        "umgebung": {"SGLANG_HTCCL_BAR1_GITTER_AB": "0"},
        "groessen": [64 << 10, 8 << 20],
        "gate": True,
    },
    {
        "name": "zwei-graphen",
        "zweck": "Zwei Aufzeichnungen, abwechselnd wiedergegeben. Zeigt "
                 "geteilte Flaggen- oder Ergebnisplaetze.",
        "umgebung": {"SGLANG_HTCCL_BAR1_GITTER_AB": str(1 << 40)},
        "groessen": [64 << 10, 256 << 10],
        "verschraenkt": True,
        "gate": True,
    },
    {
        "name": "pipe",
        "zweck": "netz_pipe aufgezeichnet. Der Direkt-Modus schaltet sich "
                 "unter Aufzeichnung selbst ab; geprueft wird, dass der "
                 "direkt=0-Weg traegt.",
        "umgebung": {
            "SGLANG_HTCCL_BAR1_PIPE": "1",
            "SGLANG_HTCCL_BAR1_GITTER_AB": str(1 << 40),
            "SGLANG_HTCCL_BAR1_PIPE_GITTER_AB": str(1 << 40),
        },
        # Zwischen pipe_ab (256 KiB) und ring_ab (1 MiB) -- nur dort waehlt
        # `algorithmus_fuer` ueberhaupt netz_pipe.
        "groessen": [512 << 10],
        "gate": True,
    },
    {
        "name": "pipe-direkt",
        "zweck": "Der Direkt-Modus AUFGEZEICHNET, mit ausdruecklich "
                 "aufgehobenem Vorbehalt und zwei Graphen. Erwartet wird, "
                 "dass er auffaellt -- kein Gate, sondern der Beleg fuer "
                 "die Begruendung in _erg_platz.",
        "umgebung": {
            "SGLANG_HTCCL_BAR1_PIPE": "1",
            "SGLANG_HTCCL_BAR1_PIPE_DIREKT": "1",
            "SGLANG_HTCCL_BAR1_PIPE_DIREKT_GRAPH": "1",
            "SGLANG_HTCCL_BAR1_GITTER_AB": str(1 << 40),
            "SGLANG_HTCCL_BAR1_PIPE_GITTER_AB": str(1 << 40),
        },
        "groessen": [512 << 10, 768 << 10],
        "verschraenkt": True,
        "gate": False,
    },
]


# ===========================================================================
# Eingabemuster und Sollwert -- unabhaengig vom Transport gerechnet
# ===========================================================================


def _muster(n: int, rang: int, runde: int, geraet) -> torch.Tensor:
    """Die Eingabe von Rang ``rang`` in Wiedergabe ``runde``.

    float32 mit kleinen ganzen Zahlen: die Summe ueber bis zu acht Raenge
    bleibt weit unter 2^24 und ist damit BITGENAU. Ein Vergleich mit
    Toleranz waere hier der Fehler -- er verzieht genau die Abweichung, die
    ein halb geschriebener Puffer erzeugt.

    Der Term ``i % 97`` haengt am Index, nicht nur am Rang: eine
    Chunkzerlegung, die einen Bereich auslaesst oder verschiebt, faellt
    sonst nicht auf, weil alle Elemente gleich aussaehen.
    """
    i = torch.arange(n, dtype=torch.float32, device=geraet)
    return (rang + 1) * 1000.0 + runde * 7.0 + (i % 97)


def _soll(n: int, welt: int, runde: int, geraet) -> torch.Tensor:
    i = torch.arange(n, dtype=torch.float32, device=geraet)
    kopf = sum((r + 1) * 1000.0 for r in range(welt))
    return kopf + welt * (runde * 7.0 + (i % 97))


def _vergleiche(ist: torch.Tensor, soll: torch.Tensor) -> tuple[int, str]:
    falsch = torch.ne(ist, soll)
    n = int(falsch.sum().item())
    if n == 0:
        return 0, ""
    erste = int(falsch.nonzero()[0].item())
    return n, (
        f"{n} von {ist.numel()} Elementen falsch, erstes bei {erste}: "
        f"ist {float(ist[erste]):.1f}, soll {float(soll[erste]):.1f}"
    )


# ===========================================================================
# Ein Graph
# ===========================================================================


def _zeichne_auf(t, n: int, rang: int, geraet):
    """Aufwaermen, aufzeichnen, ``(graph, eingabe, ausgabe)`` zurueck.

    Das Aufwaermen laeuft auf einem Nebenstrom -- die uebliche Vorschrift
    fuer ``torch.cuda.graph``, und hier zusaetzlich noetig, weil der Kern
    beim ersten Lauf die Flaggenzeilen der Gegenseite trifft.

    Die Barriere davor und danach ist eine HOSTBARRIERE ueber die
    gloo-Gruppe. Sie steht nur um die Aufwaermung und um die Aufzeichnung
    herum; innerhalb der Aufzeichnung gibt es keine, sonst pruefte das
    Programm etwas anderes als den heissen Pfad.
    """
    eingabe = _muster(n, rang, 0, geraet)

    strom = torch.cuda.Stream(device=geraet)
    strom.wait_stream(torch.cuda.current_stream(geraet))
    with torch.cuda.stream(strom):
        for _ in range(3):
            t.htccl_all_reduce(None, eingabe)
    torch.cuda.current_stream(geraet).wait_stream(strom)
    torch.cuda.synchronize(geraet)
    dist.barrier()

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        ausgabe = t.htccl_all_reduce(None, eingabe)
    torch.cuda.synchronize(geraet)
    dist.barrier()
    return graph, eingabe, ausgabe


def _pruefe_graphen(t, groessen, verschraenkt, rang, welt, geraet, protokoll):
    """Aufzeichnen, wiedergeben, nach JEDER Wiedergabe belegen."""
    graphen = []
    for n_bytes in groessen:
        n = n_bytes // 4                       # float32
        if not t.handles("all_reduce", n_bytes):
            protokoll.append(
                f"UEBERSPRUNGEN {n_bytes} Byte: handles() -> False "
                f"(Fenster {t.fenster_minimum()} Byte, max_bytes "
                f"{t.max_bytes} Byte, min_bytes {t.min_bytes}). Kein "
                f"Befund, sondern eine Groesse, die dieser Weg nicht faehrt."
            )
            continue
        algo = t.algorithmus_fuer(n_bytes)
        graph, eingabe, ausgabe = _zeichne_auf(t, n, rang, geraet)
        protokoll.append(
            f"aufgezeichnet: {n_bytes} Byte, Algorithmus {algo!r}"
        )
        graphen.append((n_bytes, n, graph, eingabe, ausgabe))

    if not graphen:
        raise RuntimeError(
            "kein einziger Graph aufgezeichnet -- alle Groessen dieses "
            "Falles wurden von handles() abgelehnt"
        )

    # Wiedergabe. Verschraenkt heisst: Runde fuer Runde ALLE Graphen, damit
    # ein geteilter Platz zwischen zwei Aufzeichnungen auffaellt. Sonst
    # jeder Graph fuer sich, damit ein Befund einem Graphen zuzuordnen ist.
    folgen = ([[(runde, g) for g in graphen] for runde in range(1, WIEDERGABEN + 1)]
              if verschraenkt
              else [[(runde, g) for runde in range(1, WIEDERGABEN + 1)]
                    for g in graphen])

    for folge in folgen:
        for runde, (n_bytes, n, graph, eingabe, ausgabe) in folge:
            eingabe.copy_(_muster(n, rang, runde, geraet))
            torch.cuda.synchronize(geraet)
            dist.barrier()
            graph.replay()
            torch.cuda.synchronize(geraet)
            schlecht, text = _vergleiche(ausgabe, _soll(n, welt, runde, geraet))
            if schlecht:
                raise AssertionError(
                    f"Wiedergabe {runde} von {n_bytes} Byte: {text}"
                )
            protokoll.append(
                f"Wiedergabe {runde}, {n_bytes} Byte: 0 von {n} Elementen "
                f"falsch"
            )
            dist.barrier()

    for _, _, graph, _, _ in graphen:
        del graph


# ===========================================================================
# Arbeiter
# ===========================================================================


def arbeiter(lokal: int, devs: list, port: str, fall: dict, ablage: str) -> None:
    rang = lokal
    welt = len(devs)
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = port
    os.environ["RANK"] = str(rang)
    os.environ["WORLD_SIZE"] = str(welt)
    for k, v in fall.get("umgebung", {}).items():
        os.environ[k] = v

    import logging
    logging.basicConfig(
        level=logging.INFO,
        format=f"[r{rang}] %(levelname)s %(name)s: %(message)s",
        stream=sys.stderr,
        force=True,
    )

    dist.init_process_group("gloo", rank=rang, world_size=welt)
    torch.cuda.set_device(devs[rang])
    geraet = torch.device("cuda", devs[rang])
    torch.cuda.init()
    torch.zeros(1, device=geraet)

    from sglang.srt.distributed.device_communicators.htccl_bar1 import (
        HTCCLBar1Transport,
    )
    from sglang.srt.distributed.device_communicators.htccl_matrix_transport import (
        _fenster_bytes,
    )

    protokoll: list[str] = []
    ergebnis = {"fall": fall["name"], "rang": rang, "ok": False,
                "grund": "", "protokoll": protokoll}
    t = None
    try:
        t = HTCCLBar1Transport(dist.group.WORLD, geraet, _fenster_bytes())
        belege = t.byte_beleg_alle()
        if not all(belege.values()):
            raise RuntimeError(
                f"Byte-Beleg des Transports gefallen: {belege}. Ohne ihn "
                f"sagt handles() zu allem False, und ein Graph ueber einen "
                f"Weg, der Bytes verliert, belegt gar nichts."
            )
        if t.pipe_an and not t.byte_beleg_pipe():
            raise RuntimeError(
                "Byte-Beleg von netz_pipe gefallen -- dieser Fall braucht "
                "ihn, sonst waehlt algorithmus_fuer netz_pipe gar nicht"
            )
        _pruefe_graphen(
            t, fall["groessen"], fall.get("verschraenkt", False),
            rang, welt, geraet, protokoll,
        )
        ergebnis["ok"] = True
    except BaseException as e:
        ergebnis["grund"] = f"{type(e).__name__}: {e}"
        sys.stderr.write(
            f"\n===== [r{rang}] FALL {fall['name']!r} GEFALLEN =====\n"
        )
        traceback.print_exc()
        sys.stderr.flush()
    finally:
        try:
            if t is not None:
                t.close()
        except Exception:
            pass
        pathlib.Path(ablage, f"r{rang}.json").write_text(
            json.dumps(ergebnis, indent=1)
        )
        try:
            dist.destroy_process_group()
        except Exception:
            pass
    if not ergebnis["ok"]:
        os._exit(1)


# ===========================================================================
# Aufruf
# ===========================================================================


def main() -> int:
    devs = [int(x) for x in
            (sys.argv[1] if len(sys.argv) > 1 else "1,2").split(",")]
    port = int(sys.argv[2]) if len(sys.argv) > 2 else 29593
    nur = sys.argv[3].split(",") if len(sys.argv) > 3 else None

    print(f"BAR1-Graph-Beleg: Geraete {devs}, {len(devs)} Raenge, "
          f"{WIEDERGABEN} Wiedergaben je Graph.\n")

    stand = []
    for i, fall in enumerate(FAELLE):
        if nur and fall["name"] not in nur:
            continue
        print(f"--- Fall {fall['name']!r} " + "-" * 40)
        print(f"    {fall['zweck']}")
        print(f"    Umgebung: {fall.get('umgebung', {})}")
        with tempfile.TemporaryDirectory() as ablage:
            try:
                mp.spawn(
                    arbeiter,
                    args=(devs, str(port + i), fall, ablage),
                    nprocs=len(devs), join=True,
                )
                ok, grund = True, ""
            except Exception as e:
                ok, grund = False, str(e)
            zeilen = []
            for r in range(len(devs)):
                p = pathlib.Path(ablage, f"r{r}.json")
                if p.is_file():
                    d = json.loads(p.read_text())
                    zeilen.append(d)
                    if not d["ok"]:
                        ok = False
                        grund = grund or d["grund"]
        for d in zeilen:
            for z in d["protokoll"]:
                print(f"    [r{d['rang']}] {z}")
        print(f"    => {'BESTANDEN' if ok else 'GEFALLEN'}"
              + (f": {grund}" if grund else ""))
        print()
        stand.append((fall["name"], fall.get("gate", True), ok, grund))

    print("=" * 62)
    print("Zusammenfassung")
    print("=" * 62)
    for name, gate, ok, grund in stand:
        marke = "BESTANDEN" if ok else "GEFALLEN "
        print(f"  {marke}  {'[Gate]' if gate else '[Info]'}  {name}"
              + (f"  -- {grund[:80]}" if grund else ""))

    gates = [(n, ok) for n, gate, ok, _ in stand if gate]
    fehlend = [n for n, ok in gates if not ok]
    print()
    if not gates:
        print("Kein Gate-Fall gelaufen -- das ist KEINE Freigabe.")
        return 2
    if fehlend:
        print(f"Gefallene Gate-Faelle: {', '.join(fehlend)}.")
        print("SGLANG_HTCCL_GRAPH_FREIGABE bleibt AUS.")
        return 1
    print("Alle Gate-Faelle bestanden.")
    print("Erst jetzt darf SGLANG_HTCCL_GRAPH_FREIGABE=1 gesetzt werden;")
    print("bar1/matrix gelten dann in parallel_state als capturable.")
    info = [(n, ok) for n, gate, ok, _ in stand if not gate]
    for n, ok in info:
        if n == "gitter" and ok:
            print()
            print("Und: der Fall 'gitter' ist bestanden -- der cooperative "
                  "Start laesst sich auf diesem Rig aufzeichnen. Damit ist "
                  "SGLANG_HTCCL_BAR1_GRAPH_GITTER=1 belegt und der "
                  "Vorbehalt in HTCCLBar1Transport._kern kann als Vorgabe "
                  "fallen.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
