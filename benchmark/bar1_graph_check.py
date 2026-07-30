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

5. **Nicht nur all_reduce.** Der Abbruch, der diesen Weg noetig gemacht
   hat, kam von einem ``broadcast`` mit 128 Byte im Draft-Graphen -- nicht
   von einem all_reduce. Die Faelle ``broadcast`` und
   ``broadcast-zwei-graphen`` fahren deshalb genau dieses Kollektiv, in
   genau dieser Groesse, und zwar mit JEDEM Rang einmal als Quelle. Die
   Pruefung ist dort schaerfer als beim all_reduce: ein Nicht-Quellen-Rang
   startet mit SEINEM Muster im Puffer, und der ist an Ort. Eine
   Aufzeichnung, die bei der Wiedergabe nichts mehr bewegt, laesst also das
   eigene Muster stehen -- und das ist von dem der Quelle unterscheidbar.

6. **Der Direkt-Modus unter Aufzeichnung.** Fall ``pipe-direkt`` zeichnet
   drei Graphen auf, von denen jeder einen reservierten Ringplatz im
   BAR1-Fenster bekommt, und gibt sie verschraenkt wieder. Dazu zwei
   Nachweise, die die anderen Faelle nicht brauchen: der Ergebnistensor
   muss WIRKLICH im Fenster liegen (sonst hat der Fall den
   ``direkt=0``-Kontrollpfad gemessen und nichts belegt), und
   zurueckgelesen wird zusaetzlich ueber den Host statt ueber den L2 der
   Empfaengerkarte -- ein anderer Weg zu denselben Bytes, weil derselbe Weg
   einen defekten Pfad verdecken wuerde. Fall
   ``pipe-direkt-vorrat-leer`` ist die Negativkontrolle dazu: ohne
   Graph-Plaetze muss jeder aufgezeichnete Aufruf auf ``direkt=0``
   zurueckfallen und trotzdem stimmen.

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
        "zweck": "Ueber der Schwelle, aber mit SGLANG_HTCCL_BAR1_GRAPH_"
                 "GITTER=0: _kern muss aufgezeichnet auf 1blk ausweichen "
                 "statt zu scheitern.",
        # Die 0 steht AUSDRUECKLICH da, seit die Vorgabe aus
        # SGLANG_HTCCL_GRAPH_FREIGABE kommt. Ohne sie haenge dieser Fall
        # daran, ob die Freigabe in der Umgebung des Aufrufers steht -- und
        # ein Fall, der je nach Umgebung etwas anderes prueft, prueft nichts.
        "umgebung": {"SGLANG_HTCCL_BAR1_GITTER_AB": "0",
                     "SGLANG_HTCCL_BAR1_GRAPH_GITTER": "0"},
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
        "zweck": "Der Direkt-Modus AUFGEZEICHNET: drei Graphen, drei "
                 "reservierte Ringplaetze, verschraenkt wiedergegeben. "
                 "Zusaetzlich wird ueber einen ANDEREN Lesepfad "
                 "zurueckgelesen (Host statt Geraet) und belegt, dass der "
                 "Ergebnistensor wirklich im BAR1-Fenster liegt.",
        "umgebung": {
            "SGLANG_HTCCL_BAR1_PIPE": "1",
            "SGLANG_HTCCL_BAR1_PIPE_DIREKT": "1",
            "SGLANG_HTCCL_BAR1_PIPE_DIREKT_GRAPH": "1",
            # 2 eager + 3 Graph-Plaetze. Ohne die drei oben waere der Vorrat
            # leer und jeder aufgezeichnete Aufruf fiele auf direkt=0
            # zurueck -- der Fall wuerde dann bestehen, ohne den
            # Direkt-Modus je gefahren zu haben. Deshalb prueft `direkt`
            # unten die Lage des Ergebnistensors nach.
            "SGLANG_HTCCL_BAR1_PIPE_ERG_RING": "5",
            "SGLANG_HTCCL_BAR1_GITTER_AB": str(1 << 40),
            "SGLANG_HTCCL_BAR1_PIPE_GITTER_AB": str(1 << 40),
        },
        "groessen": [512 << 10, 640 << 10, 768 << 10],
        "verschraenkt": True,
        "direkt": "alle",
        "gate": True,
    },
    {
        "name": "pipe-direkt-vorrat-leer",
        "zweck": "Negativkontrolle zum Fall davor: derselbe Aufbau, aber "
                 "ein Ring ohne Graph-Plaetze (L=2). Jeder aufgezeichnete "
                 "Aufruf MUSS auf direkt=0 zurueckfallen und trotzdem die "
                 "richtigen Bytes liefern -- ein Rueckfall, der falsche "
                 "Zahlen liefert, waere schlimmer als gar keiner.",
        "umgebung": {
            "SGLANG_HTCCL_BAR1_PIPE": "1",
            "SGLANG_HTCCL_BAR1_PIPE_DIREKT": "1",
            "SGLANG_HTCCL_BAR1_PIPE_DIREKT_GRAPH": "1",
            "SGLANG_HTCCL_BAR1_PIPE_ERG_RING": "2",
            "SGLANG_HTCCL_BAR1_GITTER_AB": str(1 << 40),
            "SGLANG_HTCCL_BAR1_PIPE_GITTER_AB": str(1 << 40),
        },
        "groessen": [512 << 10, 768 << 10],
        "verschraenkt": True,
        "direkt": "keiner",
        "gate": True,
    },
    {
        "name": "broadcast",
        "kollektiv": "broadcast",
        "zweck": "Die Faelle aus dem Standardlauf: 12 und 128 Byte "
                 "broadcast im Draft-Graphen. Jeder Rang einmal Quelle, "
                 "damit keine Kante ungeprueft bleibt.",
        "umgebung": {"SGLANG_HTCCL_BAR1_GITTER_AB": str(1 << 40)},
        # 12 steht vorn und nicht aus Ordnungsliebe: der erste Anlauf deckte
        # 128 und lehnte 12 ab (Untergrenze 16), und weil das Tor nur 128
        # fuhr, ging es gruen durch, waehrend der Standardlauf abbrach. Eine
        # Groesse UNTER einem 16-Byte-Paket gehoert seitdem in jeden
        # broadcast-Fall.
        "groessen": [12, 128, 64 << 10],
        "gate": True,
    },
    {
        "name": "broadcast-zwei-graphen",
        "kollektiv": "broadcast",
        "zweck": "Zwei broadcast-Aufzeichnungen, abwechselnd wiedergegeben. "
                 "Ein geteilter Schlitz oder eine eingebrannte Haelfte faellt "
                 "erst hier auf.",
        "umgebung": {"SGLANG_HTCCL_BAR1_GITTER_AB": str(1 << 40)},
        "groessen": [12, 128, 64 << 10],
        "verschraenkt": True,
        "gate": True,
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


def _zeichne_auf_broadcast(t, n: int, rang: int, geraet, src: int):
    """Dasselbe fuer ``broadcast`` -- und der ist AN ORT.

    Ein- und Ausgabe sind derselbe Puffer, weil die Naht das so zusagt
    (``broadcast(tensor, src)`` gibt denselben Tensor zurueck). Genau
    deshalb ist die Wiedergabe hier schaerfer zu pruefen als beim
    all_reduce: der Puffer wird vor jeder Wiedergabe mit dem Muster DIESES
    Ranges gefuellt, und wenn die Wiedergabe nichts bewegt, steht es
    danach noch da.

    ``src`` steht zur Aufzeichnungszeit fest und geht als Kernelargument
    mit -- das ist die Bedingung, unter der ein broadcast ueberhaupt
    aufzeichenbar ist.
    """
    puffer = _muster(n, rang, 0, geraet)

    strom = torch.cuda.Stream(device=geraet)
    strom.wait_stream(torch.cuda.current_stream(geraet))
    with torch.cuda.stream(strom):
        for _ in range(3):
            t.htccl_broadcast(None, puffer, src)
    torch.cuda.current_stream(geraet).wait_stream(strom)
    torch.cuda.synchronize(geraet)
    dist.barrier()

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        t.htccl_broadcast(None, puffer, src)
    torch.cuda.synchronize(geraet)
    dist.barrier()
    return graph, puffer


def _im_fenster(t, tensor) -> bool:
    """Liegt ``tensor`` im exportierten Ergebnisring dieses Rangs?

    Das ist die Antwort auf "ist der Direkt-Modus wirklich gefahren". Der
    Transportname und die Umgebungsvariable sagen nur, was ANGEFORDERT war;
    ob der Ringplatz auch vergeben wurde, steht allein an der Adresse des
    Ergebnistensors. Ohne diese Pruefung koennte ein Fall bestehen, der in
    Wahrheit durchgehend den ``direkt=0``-Kontrollpfad gemessen hat
    (Messdisziplin, Regel 5: nie still auf einen anderen Weg zurueckfallen).
    """
    fenster = t.erg_fenster()
    if fenster is None:
        return False
    anfang, laenge = fenster
    p = int(tensor.data_ptr())
    return anfang <= p < anfang + laenge


def _rueckgelesen_ueber_host(ausgabe, soll_cpu) -> tuple[int, str]:
    """Byte-Beleg ueber einen ANDEREN Lesepfad als den geschriebenen.

    Messdisziplin, Regel 3: "Muster schreiben, ueber einen **anderen** Weg
    zurueckleseen, jedes Byte vergleichen. Nimmt die Ruecklesung denselben
    Weg wie das Schreiben, verdeckt ein defekter Pfad seinen eigenen
    Fehler."

    Geschrieben wird hier vom Rechenkern der Nachbarkarte per PCIe in die
    BAR1-Apertur. Der Vergleich davor liest dieselben Bytes mit einem
    Geraetekernel -- also ueber den L2 der Empfaengerkarte, und genau dieser
    L2 ist mit eingehenden PCIe-Schreibvorgaengen NICHT kohaerent
    (BEFUND_L2_NICHT_KOHAERENT.md). Diese zweite Ruecklesung geht statt
    dessen ueber die Host-Kopie: eine DMA-Leseanforderung des Hosts an
    dieselbe Region, ein anderer Weg zu denselben Bytes.

    Der Sollwert kommt als CPU-Tensor herein, damit die Berechnung des
    Erwarteten nicht selbst wieder ueber die Karte laeuft.
    """
    ist = ausgabe.detach().to("cpu", copy=True)
    falsch = torch.ne(ist, soll_cpu)
    n = int(falsch.sum().item())
    if n == 0:
        return 0, ""
    erste = int(falsch.nonzero()[0].item())
    return n, (
        f"Host-Ruecklesung: {n} von {ist.numel()} Elementen falsch, erstes "
        f"bei {erste}: ist {float(ist[erste]):.1f}, soll "
        f"{float(soll_cpu[erste]):.1f}"
    )


def _pruefe_graphen(t, groessen, verschraenkt, rang, welt, geraet, protokoll,
                    kollektiv: str = "all_reduce", direkt_erwartung=None):
    """Aufzeichnen, wiedergeben, nach JEDER Wiedergabe belegen.

    ``kollektiv`` waehlt, WAS aufgezeichnet wird. Die Wiedergabe- und
    Belegschleife ist fuer beide dieselbe -- was sich unterscheidet, ist der
    Sollwert und die Frage, ob Ein- und Ausgabe derselbe Puffer sind.

    ``direkt_erwartung`` ist ``None`` (egal), ``"alle"`` (jeder
    aufgezeichnete Ergebnistensor muss im BAR1-Fenster liegen) oder
    ``"keiner"`` (keiner darf). Sie wird beim Aufzeichnen geprueft, also
    bevor eine einzige Wiedergabe gelaufen ist -- ein Fall, der den
    Direkt-Modus gar nicht erreicht hat, soll nicht erst am Ende auffallen.
    """
    graphen = []
    for n_bytes in groessen:
        n = n_bytes // 4                       # float32
        if not t.handles(kollektiv, n_bytes):
            protokoll.append(
                f"UEBERSPRUNGEN {n_bytes} Byte ({kollektiv}): handles() -> "
                f"False (Fenster {t.fenster_minimum()} Byte, max_bytes "
                f"{t.max_bytes} Byte, min_bytes {t.min_bytes}). Kein "
                f"Befund, sondern eine Groesse, die dieser Weg nicht faehrt."
            )
            continue
        if kollektiv == "broadcast":
            # Jeder Rang einmal Quelle: sonst bliebe genau die Kante
            # ungeprueft, ueber die im Ernstfall der Draft-Pick laeuft.
            for src in range(welt):
                graph, puffer = _zeichne_auf_broadcast(t, n, rang, geraet, src)
                protokoll.append(
                    f"aufgezeichnet: broadcast, {n_bytes} Byte, src={src}, "
                    f"{t.bc_runden(n_bytes)} Runden"
                )
                graphen.append((n_bytes, n, graph, puffer, puffer, src))
            continue
        algo = t.algorithmus_fuer(n_bytes)
        graph, eingabe, ausgabe = _zeichne_auf(t, n, rang, geraet)
        im_fenster = _im_fenster(t, ausgabe)
        protokoll.append(
            f"aufgezeichnet: {n_bytes} Byte, Algorithmus {algo!r}, "
            f"Ergebnistensor {'IM' if im_fenster else 'NICHT im'} BAR1-Fenster"
        )
        if direkt_erwartung == "alle" and not im_fenster:
            raise AssertionError(
                f"{n_bytes} Byte: der Ergebnistensor liegt NICHT im "
                f"BAR1-Fenster, der Direkt-Modus ist also nicht gefahren. "
                f"Dieser Fall haette den direkt=0-Kontrollpfad gemessen und "
                f"bestanden, ohne die Frage zu beantworten. Ursache "
                f"pruefen: Graph-Vorrat des Ergebnisrings "
                f"(SGLANG_HTCCL_BAR1_PIPE_ERG_RING), "
                f"SGLANG_HTCCL_BAR1_PIPE_DIREKT_GRAPH."
            )
        if direkt_erwartung == "keiner" and im_fenster:
            raise AssertionError(
                f"{n_bytes} Byte: der Ergebnistensor liegt im BAR1-Fenster, "
                f"obwohl der Graph-Vorrat leer sein sollte. Die Aufteilung "
                f"des Ergebnisrings stimmt nicht."
            )
        graphen.append((n_bytes, n, graph, eingabe, ausgabe, None))

    if not graphen:
        raise RuntimeError(
            f"kein einziger Graph aufgezeichnet ({kollektiv}) -- alle "
            f"Groessen dieses Falles wurden von handles() abgelehnt"
        )

    # Wiedergabe. Verschraenkt heisst: Runde fuer Runde ALLE Graphen, damit
    # ein geteilter Platz zwischen zwei Aufzeichnungen auffaellt. Sonst
    # jeder Graph fuer sich, damit ein Befund einem Graphen zuzuordnen ist.
    folgen = ([[(runde, g) for g in graphen] for runde in range(1, WIEDERGABEN + 1)]
              if verschraenkt
              else [[(runde, g) for runde in range(1, WIEDERGABEN + 1)]
                    for g in graphen])

    for folge in folgen:
        for runde, (n_bytes, n, graph, eingabe, ausgabe, src) in folge:
            eingabe.copy_(_muster(n, rang, runde, geraet))
            if src is None:
                soll = _soll(n, welt, runde, geraet)
                soll_cpu = _soll(n, welt, runde, "cpu")
                wer = ""
            else:
                soll = _muster(n, src, runde, geraet)
                soll_cpu = _muster(n, src, runde, "cpu")
                wer = f", src={src}"
            torch.cuda.synchronize(geraet)
            dist.barrier()
            graph.replay()
            torch.cuda.synchronize(geraet)
            schlecht, text = _vergleiche(ausgabe, soll)
            if schlecht:
                raise AssertionError(
                    f"Wiedergabe {runde} von {n_bytes} Byte{wer}: {text}"
                )
            wege = "Geraet"
            if direkt_erwartung is not None:
                schlecht, text = _rueckgelesen_ueber_host(ausgabe, soll_cpu)
                if schlecht:
                    raise AssertionError(
                        f"Wiedergabe {runde} von {n_bytes} Byte{wer}: {text}"
                    )
                wege = "Geraet+Host"
            protokoll.append(
                f"Wiedergabe {runde}, {n_bytes} Byte{wer}: 0 von {n} "
                f"Elementen falsch ({wege})"
            )
            dist.barrier()

    for eintrag in graphen:
        del eintrag


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
        kollektiv = fall.get("kollektiv", "all_reduce")
        if kollektiv == "broadcast":
            # Der Transport wird hier direkt gebaut, nicht ueber `baue_bar1`
            # -- also stehen die Belege, die die Fabrik sonst fuehrt, noch
            # nicht. broadcast haengt an beiden: am a2a-Beleg (derselbe
            # Kern, dieselben Schlitze) und am eigenen.
            if not t.byte_beleg_a2a():
                raise RuntimeError(
                    "a2a-Byte-Beleg gefallen -- broadcast faehrt auf diesem "
                    "Kern, ohne ihn sagt handles() False"
                )
            if not t.byte_beleg_broadcast():
                raise RuntimeError(
                    "broadcast-Byte-Beleg gefallen -- ein Graph ueber einen "
                    "Weg, der Bytes verliert, belegt gar nichts"
                )
        _pruefe_graphen(
            t, fall["groessen"], fall.get("verschraenkt", False),
            rang, welt, geraet, protokoll, kollektiv, fall.get("direkt"),
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
                  "Start laesst sich auf diesem Rig aufzeichnen. Der "
                  "Vorbehalt in HTCCLBar1Transport._kern faellt damit von "
                  "selbst: seine Vorgabe haengt an SGLANG_HTCCL_GRAPH_"
                  "FREIGABE. Wer ihn einzeln zurueckholen will, setzt "
                  "SGLANG_HTCCL_BAR1_GRAPH_GITTER=0.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
