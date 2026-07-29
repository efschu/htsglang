# SPDX-License-Identifier: Apache-2.0
"""Gepipelineter BAR1-Kollektivkern: ``netz_pipe``.

Warum es diese Datei gibt
-------------------------
``htccl_bar1_ext.py`` traegt die aus der Messsonde ``bar1_kollektiv.cu``
portierten Kerne ``netz`` (zwei gitterweite Sperren) und ``ring``
(``2(R-1)`` Sperren). In der Sonde waren saubere Phasengrenzen
**erwuenscht** -- eine Phasenuhr sollte Senden, Reduzieren und Verteilen
getrennt ausweisen. Im Transport hat diese Entscheidung keinen Zweck mehr,
sie kostet nur: zwischen dem Setzen einer Flagge und ihrem Eintreffen steht
die Fabric still. Genau im Bereich 1-8 MiB ist das der Unterschied zwischen
dauernd und schubweise belegter Leitung, und genau dort verliert der
Direktpfad gegen NCCL (0,97x bei 1 MiB, 0,86x bei 4 MiB, zwei Karten).

Dieser Kern zerlegt die Nutzlast in ``K`` Chunks und laesst Senden,
Reduzieren und Uebernehmen ueberlappen. Es gibt **keinen gitterweiten
Gleichschritt je Phase** mehr.

Warum eine eigene Datei und nicht ein dritter Kern in ``htccl_bar1_ext.py``:
an jener Datei arbeitet parallel ein anderer Vorgang. Die Primitiven
(``leseV4``, ``schreibeV4``, ``flaggeLesen``, ``chunkGrenzen``, ``addV4``)
sind deshalb hier **woertlich dupliziert** statt geteilt. Das ist bewusst
und es ist eine Schuld: sobald beide Straenge zusammengefuehrt sind, gehoert
der gemeinsame Teil in EINE Kopfdatei. Bis dahin gilt: wer eine Primitive
hier aendert, muss sie dort mitaendern -- sie muessen bitgleich bleiben,
weil beide Kerne in dieselben Schlitze schreiben koennen (nicht
gleichzeitig, aber in aufeinanderfolgenden Runden).

Das Schiebefenster (uebernommen von NCCL, Apache-2)
---------------------------------------------------
Die Flaggenmechanik ist **nicht** die der Sonde und auch nicht die naive
"eine Flagge je (Chunk, Sender)". Sie ist NCCLs Schiebefenster aus
``src/device/prims_simple.h`` (NCCL, Apache-2.0, NVIDIA):

* ``NCCL_STEPS 8`` (``src/include/device.h:26``) -- feste Ringtiefe.
* ``connStepPtr`` / ``connStepCache`` (``prims_simple.h:40-41``) -- EIN
  monoton wachsender Zaehler je Verbindung, plus eine **lokale Kopie**.
* ``waitPeer`` (``prims_simple.h:101-171``), Wartebedingung Zeile 107:
  ``while (connStepCache + (isSendNotRecv ? NCCL_STEPS : 0) < step + StepPerSlice)``
* ``postPeer`` (``:165-171``) -- veroeffentlichen mit
  ``st_relaxed_sys_global``.

Uebernommen ist das **Prinzip**, nicht der Quelltext: je gerichteter
Verbindung und je Ring (RS, AG) gibt es zwei absolute Zaehler,

``tail``  vom Erzeuger geschrieben  = "soviele Chunks habe ich abgelegt",
``head``  vom Verbraucher geschrieben = "soviele Chunks habe ich gelesen".

Beide werden vom Erzeuger in den Speicher des **Lesers** geschrieben (ein
gebuchter PCIe-Schreibvorgang, billig) und dort **lokal** gelesen. Ein
Lesevorgang aus einer fremden BAR waere ein Umlauf; das ist die Stelle, an
der NCCLs Annahme (NVLink, kohaerent) bei uns anders liegt und an der die
Richtung deshalb nicht beliebig ist. ``connStepCache`` ist hier
``sTailC``/``sHeadC`` im gemeinsamen Speicher: solange der zwischengespeicherte
Wert reicht, wird gar nicht nachgelesen.

Die Zaehler sind **absolut ueber die Lebensdauer des Transports**, nicht je
Runde zurueckgesetzt. Ein je Runde zurueckgesetzter Zaehler haette genau an
der Rundengrenze eine Luecke: wenn Rang A seine Runde ``n`` beendet, kann
Rang B noch in Runde ``n`` lesen, und ein Fenstervergleich gegen einen
rundenlokalen Zaehler haette das nicht gesehen. Der Zaehler liegt deshalb in
``schrittDev`` und waechst um ``K`` je Aufruf.

Was NICHT uebernommen ist und warum
-----------------------------------
* **Direkt-Modus** (``prims_simple.h:39`` ``directBuff``, ``:135-148``).
  NCCL schreibt bei verfuegbarem P2P am FIFO vorbei direkt in den
  Zielpuffer. Bei uns scheitert das nicht am Kern, sondern am Bootstrap:
  erreichbar ist nur, was beim Aufbau als dma-buf exportiert und per
  ``mmap`` + ``cudaHostRegister`` gebunden wurde. Der Ergebnistensor
  wechselt je Aufruf; ihn je Aufruf zu binden hiesse ioctl + mmap +
  Registrierung im heissen Pfad. Die Bewertung steht im Bericht -- kurz:
  gespart wuerde ein lokaler VRAM-Durchgang, und der ist gegen den
  PCIe-Engpass zwei Groessenordnungen zu billig, um die 0,86x zu erklaeren.
  Was dieser Kern dagegen sehr wohl spart: die Verteilung liest das
  Ergebnis **nicht erneut aus dem VRAM**, sondern schreibt es aus dem
  Register, in dem es gerade entstanden ist (``reduziereUndVerteile``).
  ``bar1_netz_kernel`` macht daraus zwei Durchgaenge
  (``reduziereNPhase`` + ``verteilePhase``).
* **LL / LL128** (``prims_ll.h:111-119``, ``:154``). Die Flagge im
  Datenwort setzt voraus, dass man nie "Flagge neu, Daten alt" sieht. Ueber
  NVLink ist das gegeben, ueber PCIe in eine Write-Combining-Apertur ist es
  **ungeprueft**, und auf diesem Rig ist belegt, dass der L2 mit
  eingehenden PCIe-Schreibvorgaengen nicht kohaerent ist
  (``BEFUND_L2_NICHT_KOHAERENT.md``). Ohne Beleg nicht gebaut.
"""

from __future__ import annotations

import logging
import os
import time
from typing import Optional

logger = logging.getLogger(__name__)

_ext = None

#: Groesster Verbund, fuer den die Argumentfelder Platz haben. Gleich
#: ``htccl_bar1_ext.MAX_RANGE`` -- die beiden Kerne teilen sich die
#: Peer-Tabellen des Transports und muessen dieselbe Grenze haben.
MAX_RANGE = 8

#: Obergrenze fuer die Chunkzahl. Nur eine Schleifengrenze; die
#: Schlitzgeometrie haengt an ``T``, nicht an ``K``.
MAX_CHUNKS = 1024

#: Kleinste sinnvolle Tiefe. ``T = 1`` ist NICHT nur langsam, sondern
#: verklemmt: bei ``T = 1`` faellt das Senden von Chunk ``c`` und sein
#: Verbrauch in dieselbe Schleifenrunde, und die Wartebedingung stuende vor
#: der Veroeffentlichung, auf die sie wartet. Siehe die Herleitung in
#: :func:`pipe_plan`.
MIN_TIEFE = 2

#: Groesste Tiefe. Aus NCCL: ``NCCL_STEPS 8``.
MAX_TIEFE = 8


# ===========================================================================
# Geometrie -- alles, was der Transport ueber diesen Kern wissen muss.
#
# Absichtlich HIER und nicht in htccl_bar1.py: der Kern und seine
# Speicherordnung gehoeren zusammen, und htccl_bar1.py soll nur die
# Ergebnisse einsetzen.
# ===========================================================================


def pipe_flaggen_zusatz(welt: int) -> int:
    """Zusaetzliche Flaggenbytes fuer ``netz_pipe``: ``4 * R * 256``.

    Vier Familien zu je einer 256-Byte-Zeile je Rang:

    ==== ================= ============================================
    Fam. Name              geschrieben von / gelesen von
    ==== ================= ============================================
    0    ``tailRS``        Erzeuger des RS-Chunks / dessen Empfaenger
    1    ``tailAG``        Erzeuger des AG-Chunks / dessen Empfaenger
    2    ``headRS``        Verbraucher des RS-Chunks / dessen Erzeuger
    3    ``headAG``        Verbraucher des AG-Chunks / dessen Erzeuger
    ==== ================= ============================================

    **Unabhaengig von K und T.** Das ist der eigentliche Gewinn des
    Schiebefensters gegenueber "eine Flagge je (Chunk, Sender)": dort waere
    der Bedarf ``2 * K * R`` Zeilen gewesen und haette bei jeder
    Groessenaenderung die Flaggenregion mitbewegt.

    256 Byte je Zeile, wie ueberall in diesem Transport: kein False Sharing
    zwischen Sendern, keins zwischen Familien.
    """
    return 4 * welt * 256


def pipe_fbasis(welt: int, mit_a2a: bool = True) -> int:
    """Versatz der Pipe-Zeilen in der Flaggenregion.

    **Hinter** Netz, Ring und a2a. Damit liegen alle gemessenen Topologien
    Byte fuer Byte dort, wo sie lagen; die Pipe haengt sich nur hinten an.
    Die Rechnung steht im Kern NICHT noch einmal -- sie wird als Argument
    hereingereicht, weil eine zweite Fassung genau die Stelle waere, an der
    Sender und Empfaenger auf verschiedene Zeilen zeigen.
    """
    return (2 + 2 * (welt - 1) + (1 if mit_a2a else 0)) * welt * 256


def pipe_schlitz_bytes(chunk_max: int, tiefe: int) -> int:
    """Groesse EINES Pipe-Schlitzes.

    Der Pipe-Bereich ist genau **ein Schlitzsatz** gross, also
    ``2(R-1) * chunk_max`` Byte -- dieselbe Groesse, die Netz, Ring und a2a
    je fuer sich belegen. Darin liegen ``2 * T * (R-1)`` Schlitze
    (``T(R-1)`` je Phase, zwei Phasen), also

        ``schlitz = chunk_max / T``, auf 16 Byte **abgerundet**.

    Abgerundet und nicht aufgerundet, weil die Summe der Schlitze in den
    Bereich passen muss und nicht umgekehrt. 16 Byte, weil die
    Zugriffsbreite 128 Bit ist: ein Schlitz, dessen Groesse kein Vielfaches
    von 16 waere, brauchte einen schiefen Anfang fuer den naechsten. Der
    Bereichsanfang liegt auf einer Seitengrenze (``chunk_max`` ist ein
    Vielfaches von ``SEITE``), also ist jeder Schlitzanfang
    16-Byte-ausgerichtet.
    """
    if tiefe < 1:
        return 0
    return (chunk_max // tiefe // 16) * 16


def pipe_bereich_bytes(welt: int, chunk_max: int) -> int:
    """Bytes, die der Pipe-Bereich in der Empfangsregion belegt."""
    return 2 * (welt - 1) * chunk_max


#: Seitengroesse. Bewusst eine eigene Konstante und kein Import aus
#: ``htccl_bar1`` -- jenes Modul importiert dieses, und ein Import in beide
#: Richtungen auf Modulebene waere ein Zirkel.
SEITE = 4096

#: Wieviele Ergebnispuffer der Ring haelt. Zwei ist das Minimum, mit dem
#: Runde ``n`` nicht in den Puffer schreibt, den der Aufrufer aus Runde
#: ``n-1`` noch in der Hand haelt.
ERG_RING_VORGABE = 2


def erg_stride_bytes(max_bytes: int) -> int:
    """Abstand zweier Ergebnispuffer im Ring.

    Auf eine Seite aufgerundet. Damit beginnt jeder Puffer auf einer
    Seitengrenze, ist also 16-Byte-ausgerichtet -- die Bedingung, an der
    jeder Schreibvorgang in die WC-Apertur haengt -- und ein Puffer teilt
    nie eine Seite mit seinem Nachbarn.
    """
    return ((max_bytes + SEITE - 1) // SEITE) * SEITE


def erg_ring_bytes(max_bytes: int, ring: int) -> int:
    """Was der Ergebnisring im BAR-Fenster kostet: ``L * stride``.

    Das ist die Rechnung, die gegen die TATSAECHLICH abgebildete Laenge
    geprueft werden muss, nicht gegen die Bruttogroesse aus sysfs. Sie ist
    nicht klein: bei ``R = 2``, a2a und Pipe eingeschaltet, stehen
    ``6 (R-1) = 6`` Schlitze zu je ``chunk_max`` gegen ``L * R * chunk_max``
    Ergebnispuffer -- bei ``L = 2`` also 10 statt 6 Einheiten, und die
    groesste Nutzlast sinkt entsprechend auf 6/10. Der Direkt-Modus ist
    nicht umsonst; er tauscht BAR1-Fenster gegen einen gesparten
    VRAM-Durchgang und gegen den Wegfall des Umkopierens.
    """
    return max(0, int(ring)) * erg_stride_bytes(max_bytes)


def pipe_fensterbedarf(welt: int, chunk_max: int, tiefe: int) -> int:
    """Der **gerechnete** Bedarf: ``2 * T * (R-1)`` Schlitze.

    Nicht "ein Schlitzsatz, passt schon". Diese Zahl geht in
    ``handles()`` gegen die TATSAECHLICH abgebildete Laenge, und sie ist
    kleiner oder gleich :func:`pipe_bereich_bytes` -- die Differenz ist der
    Verschnitt aus dem Abrunden in :func:`pipe_schlitz_bytes`.
    """
    return 2 * tiefe * (welt - 1) * pipe_schlitz_bytes(chunk_max, tiefe)


def _chunk_grenzen(n: int, j: int, teile: int) -> tuple[int, int]:
    """Dieselbe Zerlegung wie ``chunkGrenzen`` im Kern: Rest nach vorn.

    Sie steht hier NUR fuer die Planung und den Byte-Beleg. Die
    Nahtstellenpruefung im Kern rechnet mit ``chunkGrenzen`` selbst -- eine
    Naht, die auf beiden Seiten mit derselben falschen Formel geprueft
    wuerde, faellt nicht auf.
    """
    basis = n // teile
    rest = n - basis * teile
    laenge = basis + (1 if j < rest else 0)
    versatz = j * basis + (j if j < rest else rest)
    return versatz, laenge


def groesstes_stueck4(n4: int, welt: int, k: int) -> int:
    """Groesstes Stueck (in 128-Bit-Paketen), das je in einen Schlitz muss.

    Ausgerechnet ueber ALLE (Chunk, Rang)-Paare, nicht ueber
    ``ceil(ceil(n4/K)/R)``. Die geschlossene Form stimmt hier zwar, aber sie
    ist eine zweite Fassung derselben Zerlegung, und genau das ist die
    Fehlerklasse, die dieser Transport an mehreren Stellen ausdruecklich
    vermeidet.
    """
    gr = 0
    for c in range(k):
        _, clen = _chunk_grenzen(n4, c, k)
        for z in range(welt):
            _, plen = _chunk_grenzen(clen, z, welt)
            if plen > gr:
                gr = plen
    return gr


def pipe_plan(nbytes: int, welt: int, chunk_max: int, tiefe: int,
              k_wunsch: int, chunk_ziel_bytes: int,
              k_max: int = 64) -> Optional[int]:
    """Die Chunkzahl ``K`` fuer diese Nutzlast, oder ``None``.

    ``None`` heisst: diese Groesse traegt der gepipelinete Weg nicht -- der
    Aufrufer meldet sich ueber ``handles()`` ab, statt still kleiner zu
    rechnen.

    **Rangeinheitlich.** Jeder Eingang ist gruppenweit gleich (``nbytes``
    und ``welt`` ohnehin, ``chunk_max`` aus dem Aufbau, der Rest aus
    rangeinheitlichen Umgebungsvariablen). Zwei Raenge duerfen hier nie
    verschieden antworten.

    Zwei Bedingungen, beide hart:

    1. ``K >= T``. Bei ``K < T`` gaebe es Schlitzklassen, die nie belegt
       werden -- harmlos --, aber die Schlitzgroesse ist ``chunk_max/T``,
       waehrend das groesste Stueck mit ``1/K`` schrumpft; bei ``K < T``
       passt es nicht. Ausserdem ist ``T >= 2`` erzwungen (siehe
       :data:`MIN_TIEFE`).
    2. Das groesste Stueck muss in einen Schlitz passen. Geprueft mit
       :func:`groesstes_stueck4`, also mit der Zerlegung selbst.

    Die Vorgabe fuer ``K`` (``k_wunsch <= 0``) leitet sich aus
    ``chunk_ziel_bytes`` ab. Sie ist eine **Ausgangszahl fuer die
    Messreihe**, keine Messaussage: begruendet ist sie damit, dass in
    ``MESSUNG_ALLES_IM_SELBEN_LAUF.md`` (drei Raenge, Rig 1) ``netz`` bei
    1 MiB 330,30 us braucht und die Leerrunde 25,74 us; ein 256-KiB-Chunk
    kostet damit rund 82 us Uebertragung gegen rund 26 us Umlauf, der
    Umlauf bleibt also hinter dem Verkehr verborgen. Auf zwei Karten und am
    x8-Port ist das ungemessen.
    """
    if nbytes % 16 != 0:
        return None
    n4 = nbytes // 16
    if tiefe < MIN_TIEFE or tiefe > MAX_TIEFE:
        return None
    schlitz = pipe_schlitz_bytes(chunk_max, tiefe)
    if schlitz <= 0:
        return None
    obergrenze = min(k_max, MAX_CHUNKS, max(1, n4 // welt))
    if obergrenze < tiefe:
        # Weniger Chunks moeglich als die Tiefe verlangt: zu wenig Nutzlast
        # je Rang. Kein Herunterregeln der Tiefe -- sie bestimmt die
        # Schlitzgeometrie und die steht seit dem Aufbau fest.
        return None

    if k_wunsch > 0:
        kandidaten = [k_wunsch]
    else:
        ziel = max(1, chunk_ziel_bytes)
        k0 = max(tiefe, min(obergrenze, max(1, nbytes // ziel)))
        # Aufsteigend probieren: passt das groesste Stueck nicht in einen
        # Schlitz, hilft MEHR Zerlegung, nie weniger.
        kandidaten = list(range(k0, obergrenze + 1))
    for k in kandidaten:
        if k < tiefe or k > obergrenze:
            continue
        if groesstes_stueck4(n4, welt, k) * 16 <= schlitz:
            return int(k)
    return None


# ===========================================================================
# Kernelquelltext
# ===========================================================================

_CUDA_SRC = r"""
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <cooperative_groups.h>
#include <cstring>
#include <vector>

namespace cg = cooperative_groups;

#define HTCCL_PIPE_MAX_RANKS 8
#define HTCCL_PIPE_MAX_TIEFE 8

#define K_1BLK   0
#define K_GITTER 1
#define LA_CV    0
#define LA_MMIO  2

using u64 = unsigned long long;

// ===========================================================================
// Primitiven -- WOERTLICH aus htccl_bar1_ext.py dupliziert. Siehe Moduldoku:
// die Duplikation ist bewusst (Dateikonflikt) und ist eine Schuld. Wer hier
// etwas aendert, muss es dort mitaendern.
// ===========================================================================

__device__ __forceinline__ uint4 leseV4(const void *p)
{
    uint4 v;
    asm volatile("ld.global.cv.v4.u32 {%0,%1,%2,%3}, [%4];"
                 : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w)
                 : "l"(p) : "memory");
    return v;
}

__device__ __forceinline__ void schreibeV4(void *p, uint4 v)
{
    asm volatile("st.global.wt.v4.u32 [%0], {%1,%2,%3,%4};"
                 :: "l"(p), "r"(v.x), "r"(v.y), "r"(v.z), "r"(v.w) : "memory");
}

template<int LA>
__device__ __forceinline__ u64 flaggeLesen(const u64 *p)
{
    u64 v;
    if (LA == LA_MMIO) {
        asm volatile("ld.mmio.relaxed.sys.global.u64 %0, [%1];"
                     : "=l"(v) : "l"(p) : "memory");
    } else {
        asm volatile("ld.global.cv.u64 %0, [%1];" : "=l"(v) : "l"(p) : "memory");
    }
    return v;
}

__device__ __forceinline__ void schreibeU64(void *p, u64 v)
{
    asm volatile("st.global.wt.u64 [%0], %1;" :: "l"(p), "l"(v) : "memory");
}

__device__ __forceinline__ unsigned int addF(unsigned int a, unsigned int b)
{
    return __float_as_uint(__uint_as_float(a) + __uint_as_float(b));
}

__device__ __forceinline__ unsigned int addH2(unsigned int a, unsigned int b)
{
    __half2 x = *(const __half2 *)&a;
    __half2 y = *(const __half2 *)&b;
    __half2 z = __hadd2(x, y);
    return *(const unsigned int *)&z;
}

__device__ __forceinline__ unsigned int addBF2(unsigned int a, unsigned int b)
{
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
    __nv_bfloat162 x = *(const __nv_bfloat162 *)&a;
    __nv_bfloat162 y = *(const __nv_bfloat162 *)&b;
    __nv_bfloat162 z = __hadd2(x, y);
    return *(const unsigned int *)&z;
#else
    const __nv_bfloat16 *x = (const __nv_bfloat16 *)&a;
    const __nv_bfloat16 *y = (const __nv_bfloat16 *)&b;
    __nv_bfloat16 r[2];
    r[0] = __float2bfloat16(__bfloat162float(x[0]) + __bfloat162float(y[0]));
    r[1] = __float2bfloat16(__bfloat162float(x[1]) + __bfloat162float(y[1]));
    return *(const unsigned int *)r;
#endif
}

template<typename T>
__device__ __forceinline__ uint4 addV4(uint4 a, uint4 b);

template<>
__device__ __forceinline__ uint4 addV4<float>(uint4 a, uint4 b)
{
    uint4 r;
    r.x = addF(a.x, b.x); r.y = addF(a.y, b.y);
    r.z = addF(a.z, b.z); r.w = addF(a.w, b.w);
    return r;
}

template<>
__device__ __forceinline__ uint4 addV4<__half>(uint4 a, uint4 b)
{
    uint4 r;
    r.x = addH2(a.x, b.x); r.y = addH2(a.y, b.y);
    r.z = addH2(a.z, b.z); r.w = addH2(a.w, b.w);
    return r;
}

template<>
__device__ __forceinline__ uint4 addV4<__nv_bfloat16>(uint4 a, uint4 b)
{
    uint4 r;
    r.x = addBF2(a.x, b.x); r.y = addBF2(a.y, b.y);
    r.z = addBF2(a.z, b.z); r.w = addBF2(a.w, b.w);
    return r;
}

// Chunk j von n Einheiten ueber `teile`; Rest auf die VORDEREN Chunks.
// Bitgleich zu htccl_bar1_ext.py::chunkGrenzen -- die Pipe zerlegt zweimal
// mit derselben Regel (erst in K Chunks, dann in R Stuecke).
__device__ __host__ __forceinline__ void chunkGrenzen(int n, int j, int teile,
                                                      int *off, int *len)
{
    const int basis = n / teile;
    const int rest  = n - basis * teile;
    *len = basis + (j < rest ? 1 : 0);
    *off = j * basis + (j < rest ? j : rest);
}

template<int GRID>
__device__ __forceinline__ void barriere(void)
{
    if (GRID == K_GITTER) cg::this_grid().sync();
    else                  __syncthreads();
}

// ===========================================================================
// Umkehrung von chunkGrenzen: zu einem flachen Index j in [0, n) das Stueck
// z und den Versatz darin. Geschlossene Form, kein Scan.
//
// Sie erlaubt es, den Reduce-Scatter-Sendevorgang ueber ALLE Ziele in EINER
// flachen Schleife zu fahren -- verschiedene Warps schreiben dann
// gleichzeitig an verschiedene Karten, statt Ziel fuer Ziel. Dasselbe
// Verfahren wie im a2a-Kern, dort ueber einen Praefixscan im gemeinsamen
// Speicher; hier geht es geschlossen, weil die Zerlegung gleichmaessig ist.
//
// basis == 0 (n < teile) ist NICHT der Sonderfall, der er zu sein scheint:
// dann ist rest == n und grenz == n, der else-Zweig also unerreichbar. Eine
// Division durch 0 kann nicht auftreten.
// ===========================================================================
__device__ __forceinline__ void stueckAus(int n, int teile, int j,
                                          int *z, int *lok)
{
    const int basis = n / teile;
    const int rest  = n - basis * teile;
    const int grenz = rest * (basis + 1);
    if (j < grenz) {
        const int q = j / (basis + 1);
        *z = q; *lok = j - q * (basis + 1);
    } else {
        const int d = j - grenz;
        const int q = d / basis;               // basis > 0, s.o.
        *z = rest + q; *lok = d - q * basis;
    }
}

// ===========================================================================
// Argumente
// ===========================================================================
struct PipeArgs {
    const uint4 *in;
    uint4       *out;
    u64         *rundeDev;
    u64         *schrittDev;   // absoluter Chunkzaehler, ueberlebt den Aufruf
    unsigned int *ctlStatus;
    unsigned int *abbruchDev;
    int          n4;
    int          R;
    int          rang;
    int          K;            // Chunkzahl dieses Aufrufs
    int          TT;           // Ringtiefe = Schlitzklassen je Phase
    int          PP;           // Zeitplan-Vorlauf, 2 <= PP <= TT
    int          quittung;     // 1 = head veroeffentlichen (Vorgabe)
    long long    schlitz4;     // Schlitzgroesse in 128-Bit-Paketen
    long long    klasse4;      // Abstand zweier Schlitzklassen = (R-1)*schlitz4
    u64          deckelZyklen;

    int          direkt;       // 1 = Allgather schreibt in den Ergebnispuffer

    // Nutzlastschlitze, Klasse 0. Klasse t liegt bei + t*klasse4.
    uint4       *sendRS[HTCCL_PIPE_MAX_RANKS];
    uint4       *sendAG[HTCCL_PIPE_MAX_RANKS];
    const uint4 *recvRS[HTCCL_PIPE_MAX_RANKS];
    const uint4 *recvAG[HTCCL_PIPE_MAX_RANKS];
    // Direkt-Modus: der Ergebnispuffer des Peers z fuer DIESE Runde. Er
    // liegt in derselben exportierten Region wie die Schlitze, also ist er
    // ohne jede Registrierung im heissen Pfad erreichbar. Der Versatz
    // innerhalb des Puffers ist derselbe wie im eigenen `out` -- der
    // Allgather kopiert nichts um, er schreibt an die Endstelle.
    uint4       *ergAn[HTCCL_PIPE_MAX_RANKS];

    // Zaehler. [0] = RS-Ring, [1] = AG-Ring.
    u64         *tailAn [2][HTCCL_PIPE_MAX_RANKS];  // ich -> Empfaenger z
    const u64   *tailVon[2][HTCCL_PIPE_MAX_RANKS];  // lokal, von Sender s
    u64         *headAn [2][HTCCL_PIPE_MAX_RANKS];  // ich -> Erzeuger s
    const u64   *headVon[2][HTCCL_PIPE_MAX_RANKS];  // lokal, von Empfaenger z
};

// ===========================================================================
// Der Kern
// ===========================================================================
template<typename T, int LA, int GRID>
__global__ void bar1_netz_pipe_kernel(PipeArgs A)
{
    const int tid = (GRID == K_GITTER)
                        ? (int)(blockIdx.x * blockDim.x + threadIdx.x)
                        : (int)threadIdx.x;
    const int nth = (GRID == K_GITTER)
                        ? (int)(gridDim.x * blockDim.x)
                        : (int)blockDim.x;
    const bool erster = (tid == 0);
    const int  n4 = A.n4, R = A.R, r = A.rang, K = A.K, TT = A.TT, PP = A.PP;
    const u64  runde  = *(const volatile u64 *)A.rundeDev + 1ull;
    const u64  basis  = *(const volatile u64 *)A.schrittDev;

    // Alles, was mit LAUFENDEM Index indiziert wird, in den gemeinsamen
    // Speicher. Ein dynamisch indiziertes Feld der Argumentstruktur zwingt
    // nvcc, den ganzen Parameterblock je Thread in den local memory zu
    // legen -- derselbe Grund wie im a2a-Kern.
    __shared__ uint4       *sSendRS[HTCCL_PIPE_MAX_RANKS];
    __shared__ uint4       *sSendAG[HTCCL_PIPE_MAX_RANKS];
    __shared__ const uint4 *sRecvRS[HTCCL_PIPE_MAX_RANKS];
    __shared__ const uint4 *sRecvAG[HTCCL_PIPE_MAX_RANKS];
    __shared__ uint4       *sErgAn [HTCCL_PIPE_MAX_RANKS];
    // Zaehlerzeiger und die lokalen Kopien (NCCL: connStepCache). Nur der
    // erste Thread fasst sie an; sie liegen trotzdem im gemeinsamen
    // Speicher, weil sonst 4*8 u64 als Register je Thread anfielen.
    __shared__ u64         *sTailAn [2][HTCCL_PIPE_MAX_RANKS];
    __shared__ const u64   *sTailVon[2][HTCCL_PIPE_MAX_RANKS];
    __shared__ u64         *sHeadAn [2][HTCCL_PIPE_MAX_RANKS];
    __shared__ const u64   *sHeadVon[2][HTCCL_PIPE_MAX_RANKS];
    __shared__ u64          sTailC[2][HTCCL_PIPE_MAX_RANKS];
    __shared__ u64          sHeadC[2][HTCCL_PIPE_MAX_RANKS];
    __shared__ int          abbruchS;

    if (threadIdx.x == 0) {
        for (int i = 0; i < R; ++i) {
            sSendRS[i] = A.sendRS[i]; sSendAG[i] = A.sendAG[i];
            sRecvRS[i] = A.recvRS[i]; sRecvAG[i] = A.recvAG[i];
            sErgAn[i]  = A.ergAn[i];
            for (int ph = 0; ph < 2; ++ph) {
                sTailAn [ph][i] = A.tailAn [ph][i];
                sTailVon[ph][i] = A.tailVon[ph][i];
                sHeadAn [ph][i] = A.headAn [ph][i];
                sHeadVon[ph][i] = A.headVon[ph][i];
                // Der Anfangswert des Zwischenspeichers ist BASIS, nicht 0:
                // was vor diesem Aufruf geschehen ist, steht schon in
                // schrittDev. Ein mit 0 anfangender Zwischenspeicher waere
                // nicht falsch, aber er erzwaenge in der ersten
                // Schleifenrunde ein Nachlesen je Verbindung.
                sTailC[ph][i] = basis;
                sHeadC[ph][i] = basis;
            }
        }
        abbruchS = 0;
    }
    __syncthreads();

    if (GRID == K_GITTER && erster) {
        *(volatile unsigned int *)A.abbruchDev = 0u;
        __threadfence();
    }
    barriere<GRID>();

    // -- Wartebedingungen ---------------------------------------------------
    //
    // Beide lesen ZUERST den Zwischenspeicher und fassen die Zeile nur an,
    // wenn er nicht reicht. Ueber PCIe waere jedes Nachlesen ein Umlauf --
    // ohne den Zwischenspeicher kostete das Fenster mehr als es bringt
    // (NCCL prims_simple.h:40-41, 107).
    //
    // `ziel` ist immer ein ABSOLUTER Schritt (basis + c + 1).

#define PIPE_WARTE_DATEN(PH, ZIEL)                                             \
    do {                                                                       \
        const u64 _z = (ZIEL);                                                 \
        long long _t0 = clock64();                                             \
        for (;;) {                                                             \
            bool _alle = true;                                                 \
            for (int _s = 0; _s < R; ++_s) {                                   \
                if (_s == r) continue;                                         \
                if (sTailC[PH][_s] < _z) {                                     \
                    sTailC[PH][_s] = flaggeLesen<LA>(sTailVon[PH][_s]);        \
                    if (sTailC[PH][_s] < _z) { _alle = false; break; }         \
                }                                                              \
            }                                                                  \
            if (_alle) break;                                                  \
            if ((u64)(clock64() - _t0) > A.deckelZyklen) { abbruchS = 1; break; } \
        }                                                                      \
    } while (0)

    // Fenster: der Empfaenger z hat mindestens (ZIEL - TT) Chunks gelesen.
    // Unsigned-Vergleich in der Form `cache + TT < ziel` -- genau wie NCCL,
    // damit ziel < TT nicht unterlaeuft.
#define PIPE_WARTE_FENSTER(PH, ZIEL)                                           \
    do {                                                                       \
        if (A.quittung) {                                                      \
            const u64 _z = (ZIEL);                                             \
            long long _t0 = clock64();                                         \
            for (;;) {                                                         \
                bool _alle = true;                                             \
                for (int _q = 0; _q < R; ++_q) {                               \
                    if (_q == r) continue;                                     \
                    if (sHeadC[PH][_q] + (u64)TT < _z) {                       \
                        sHeadC[PH][_q] = flaggeLesen<LA>(sHeadVon[PH][_q]);    \
                        if (sHeadC[PH][_q] + (u64)TT < _z) { _alle = false; break; } \
                    }                                                          \
                }                                                              \
                if (_alle) break;                                              \
                if ((u64)(clock64() - _t0) > A.deckelZyklen) { abbruchS = 1; break; } \
            }                                                                  \
        }                                                                      \
    } while (0)

    // -- Schleife -----------------------------------------------------------
    //
    // Drei Stufen, um PP-1 bzw. PP Schleifenrunden versetzt:
    //
    //   cs = i        Reduce-Scatter-Stueck von Chunk cs an alle Peers
    //   cr = i-(PP-1) Chunk cr reduzieren UND das Ergebnis sofort verteilen
    //   cg = i-PP     Allgather-Stuecke von Chunk cg uebernehmen
    //
    // Ein Rang schiebt also Chunk i hinaus, waehrend er Chunk i-PP+1
    // reduziert und Chunk i-PP einsammelt. Zwischen den Stufen steht KEINE
    // gitterweite Sperre je Phase mehr; die beiden Sperren je Schleifenrunde
    // trennen nur "erster hat gewartet" von "alle arbeiten" und "alle sind
    // fertig" von "erster veroeffentlicht".
    //
    // WARUM VORLAUF UND RINGTIEFE GETRENNT SIND
    // -----------------------------------------
    // Sie waren erst dasselbe (PP == TT), und das war ein Entwurfsfehler.
    // Die Fensterbedingung fuer das Senden von Chunk i verlangt, dass der
    // Empfaenger Chunk i-TT verbraucht hat; verbraucht wird er bei ihm in
    // Schleifenrunde (i-TT)+(PP-1). Der Empfaenger darf also um
    //
    //     TT - PP + 1  Schleifenrunden
    //
    // zurueckliegen, bevor der Sender blockiert. Mit PP == TT ist das genau
    // EINE Runde -- der Ring waere TT Schlitze tief, aber der Zeitplan
    // verbraeuchte sie sofort wieder, und zwei ungleich schnelle Karten
    // liefen faktisch im Gleichschritt. Genau die zeitliche Entkopplung, die
    // dem Host-Weg seine einzige echte Staerke gibt (der Sender kippt ins
    // RAM und ist fertig), waere damit verschenkt.
    //
    // Mit PP = 2 (dem Minimum, das ueberhaupt pipelinet) und TT = 4 sind es
    // drei Runden Versatz. Das ist NCCLs Bauform: acht Schritte im Ring
    // (NCCL_STEPS, src/include/device.h:26) gegen zwei Schritte je Scheibe
    // (ALLREDUCE_SLICESTEPS, src/include/collectives.h:19) -- Ringtiefe und
    // Vorlauf sind auch dort verschiedene Zahlen.
    for (int i = 0; i < K + PP; ++i) {
        const int cs = i;
        const int cr = i - (PP - 1);
        const int cg = i - PP;
        // SCHLITZKLASSE AUS DEM ABSOLUTEN SCHRITT, nicht aus dem Chunkindex
        // dieses Aufrufs. Mit `c % TT` verschoebe sich die Zuordnung
        // Klasse <-> Schlitz an jeder Rundengrenze, an der K kein Vielfaches
        // von TT ist -- die Fensterbedingung "head + TT >= schritt" spraeche
        // dann ueber einen anderen Schlitz als den, in den geschrieben wird.
        // Beispiel TT=2, K=3: Runde n legt die Schritte 1,2,3 in die Klassen
        // 0,1,0; Runde n+1 legt Schritt 4 in Klasse 0, waehrend dort noch
        // Schritt 3 liegt -- die Bedingung haette aber nur Schritt 2
        // verlangt. Mit dem absoluten Schritt ist die Klasse
        // `(schritt-1) mod TT` und die Bedingung trifft immer den richtigen
        // Vorgaenger.
        const int ts = (int)((basis + (u64)cs) % (u64)TT);
        const int tr = (cr >= 0) ? (int)((basis + (u64)cr) % (u64)TT) : 0;
        const int tg = (cg >= 0) ? (int)((basis + (u64)cg) % (u64)TT) : 0;

        // (a) nur der erste Thread: warten
        if (erster) {
            if (cs < K)             PIPE_WARTE_FENSTER(0, basis + (u64)cs + 1ull);
            if (!abbruchS && cr >= 0 && cr < K) {
                                    PIPE_WARTE_DATEN  (0, basis + (u64)cr + 1ull);
                if (!abbruchS)      PIPE_WARTE_FENSTER(1, basis + (u64)cr + 1ull);
            }
            if (!abbruchS && cg >= 0) PIPE_WARTE_DATEN(1, basis + (u64)cg + 1ull);
            if (abbruchS && GRID == K_GITTER) {
                *(volatile unsigned int *)A.abbruchDev = 1u;
                __threadfence();
            }
        }
        barriere<GRID>();
        {
            const int ab = (GRID == K_1BLK)
                               ? abbruchS
                               : (int)*(volatile unsigned int *)A.abbruchDev;
            if (ab) {
                // ALLE Bloecke kehren gemeinsam zurueck -- ein einzelner
                // Block, der eine grid.sync() nicht mehr erreicht, haengt
                // den Rest auf.
                if (erster) {
                    *A.ctlStatus = 1u;
                    *(volatile u64 *)A.rundeDev   = runde;
                    *(volatile u64 *)A.schrittDev = basis + (u64)K;
                    __threadfence_system();
                }
                return;
            }
        }
        // OHNE DIESEN ZAUN LIEST DIE REDUKTION ALTE ZEILEN.
        //
        // `leseV4` ist `ld.global.cv`, und `ld.global.cv` allein sieht
        // eingehende PCIe-Schreibvorgaenge auf diesem Rig NICHT: der L2 der
        // Empfaengerkarte ist mit ihnen nicht kohaerent
        // (BEFUND_L2_NICHT_KOHAERENT.md; BEFUND_L2_UMGEHBAR.md, Zeile (1):
        // 46,5 Millionen Leseversuche, nie sichtbar). Sichtbar wird es erst
        // mit `__threadfence_system()` DAVOR -- Zeile (6) derselben Messung,
        // SASS `MEMBAR.SC.SYS` + `CCTL.IVALL`, sechs Leseversuche. Die
        // Sperre allein genuegt nicht: `__syncthreads()` und
        // `grid.sync()` sind geraeteweit, nicht systemweit.
        //
        // Von ALLEN Threads, nicht nur vom ersten: der erste hat die Flagge
        // gesehen, aber jeder Thread liest gleich seine eigenen Schlitze.
        // `bar1_netz_kernel` hat denselben Zaun an derselben Stelle
        // (htccl_bar1_ext.py, hinter jeder Abbruchpruefung).
        __threadfence_system();

        // (b) alle Threads: die Arbeit dieser Schleifenrunde.
        //
        // Die drei Stufen stehen ohne Sperre nebeneinander. Sie kommen sich
        // nicht ins Gehege:
        //   - cs schreibt in FREMDE RS-Schlitze,
        //   - cr liest EIGENE RS-Schlitze und schreibt out + FREMDE AG-Schlitze,
        //   - cg liest EIGENE AG-Schlitze und schreibt out.
        // cr und cg schreiben beide out, aber in verschiedene Chunks
        // (cr = cg+1). cs und cg liegen in derselben Schlitzklasse
        // ((i) mod TT == (i-TT) mod TT), aber in verschiedenen Ringen
        // (RS gegen AG) und auf verschiedenen Karten (fremd gegen eigen).
        if (cs < K) {
            int coff, clen;
            chunkGrenzen(n4, cs, K, &coff, &clen);
            const long long kv = (long long)ts * A.klasse4;
            for (int j = tid; j < clen; j += nth) {
                int z, lok;
                stueckAus(clen, R, j, &z, &lok);
                if (z == r) continue;      // eigenes Stueck bleibt in `in`
                schreibeV4(sSendRS[z] + kv + lok, A.in[coff + j]);
            }
        }
        if (cr >= 0 && cr < K) {
            int coff, clen, poff, plen;
            chunkGrenzen(n4, cr, K, &coff, &clen);
            chunkGrenzen(clen, r, R, &poff, &plen);
            const long long kv = (long long)tr * A.klasse4;
            const uint4 *q = A.in  + coff + poff;
            uint4       *o = A.out + coff + poff;
            for (int j = tid; j < plen; j += nth) {
                uint4 s = q[j];
                for (int p = 0; p < R; ++p) {
                    if (p == r) continue;
                    s = addV4<T>(s, leseV4(sRecvRS[p] + kv + j));
                }
                // Ergebnis EINMAL bilden, dann aus dem Register sowohl in
                // den eigenen Ausgabepuffer als auch an alle Peers. Der
                // Sondenkern liest dafuer `out` ein zweites Mal
                // (reduziereNPhase, dann verteilePhase).
                // Im Direkt-Modus liegt `out` in der exportierten Region,
                // und in DIESELBE Region schreiben die Peers per PCIe. Ein
                // gewoehnlicher Store liesse eine schmutzige L2-Zeile
                // zurueck; eine Zeile ist 128 Byte, eine Stueckgrenze aber
                // nur 16-Byte-granular, also enthaelt die Zeile am Rand
                // meines Stuecks auch Bytes, die der Nachbar per PCIe
                // schreibt. Wird sie spaeter zurueckgeschrieben, ueberbuegelt
                // sie dessen Bytes -- falsche Zahlen ohne Absturz. `st.wt`
                // hinterlaesst keine schmutzige Zeile. Ausserhalb des
                // Direkt-Modus ist `out` ein gewoehnlicher torch-Puffer, den
                // kein Peer anfasst; dort bleibt der normale Store.
                if (A.direkt) schreibeV4(o + j, s);
                else          o[j] = s;
                if (A.direkt) {
                    // DIREKT-MODUS: an die Endstelle im Ergebnispuffer des
                    // Empfaengers, nicht in einen Schlitz. Damit entfaellt
                    // beim Empfaenger der Auslese- und Umkopierdurchgang
                    // ERSATZLOS -- nicht halb. Derselbe Versatz wie im
                    // eigenen `out`, also 16-Byte-ausgerichtet, weil der
                    // Puffer auf einer Seitengrenze beginnt.
                    const int ziel = coff + poff + j;
                    for (int z = 0; z < R; ++z) {
                        if (z == r) continue;
                        schreibeV4(sErgAn[z] + ziel, s);
                    }
                } else {
                    for (int z = 0; z < R; ++z) {
                        if (z == r) continue;
                        schreibeV4(sSendAG[z] + kv + j, s);
                    }
                }
            }
        }
        if (cg >= 0 && !A.direkt) {
            int coff, clen;
            chunkGrenzen(n4, cg, K, &coff, &clen);
            const long long kv = (long long)tg * A.klasse4;
            for (int j = tid; j < clen; j += nth) {
                int s, lok;
                stueckAus(clen, R, j, &s, &lok);
                if (s == r) continue;      // eigenes Stueck steht schon in out
                A.out[coff + j] = leseV4(sRecvAG[s] + kv + lok);
            }
        }
        // Im Direkt-Modus gibt es hier NICHTS zu tun: die Stuecke der Peers
        // stehen bereits an ihrer Endstelle. Die Wartebedingung auf tailAG
        // in Phase (a) bleibt trotzdem -- sie ist es, die sicherstellt, dass
        // der Kern nicht endet, bevor die letzten Pakete der Peers
        // angekommen sind. Ohne sie gaebe der Aufruf einen halb gefuellten
        // Ergebnistensor zurueck.
        __threadfence_system();
        barriere<GRID>();

        // (c) nur der erste Thread: veroeffentlichen.
        //
        // Reihenfolge: erst die Nutzlast (oben, vor der Sperre und vor dem
        // Zaun), dann der Zaehler. Nutzlast und Zaehler gehen an DIESELBE
        // Zielkarte, also gilt die PCIe-Reihenfolge je
        // Requester/Completer-Paar -- dieselbe Annahme, auf der schon
        // netz/ring/a2a stehen.
        if (erster) {
            if (cs < K) {
                const u64 m = basis + (u64)cs + 1ull;
                for (int z = 0; z < R; ++z)
                    if (z != r) schreibeU64(sTailAn[0][z], m);
            }
            if (cr >= 0 && cr < K) {
                const u64 m = basis + (u64)cr + 1ull;
                for (int z = 0; z < R; ++z) {
                    if (z == r) continue;
                    schreibeU64(sTailAn[1][z], m);          // AG-Daten liegen
                    if (A.quittung) schreibeU64(sHeadAn[0][z], m);  // RS gelesen
                }
            }
            if (cg >= 0 && A.quittung) {
                const u64 m = basis + (u64)cg + 1ull;
                for (int z = 0; z < R; ++z)
                    if (z != r) schreibeU64(sHeadAn[1][z], m);      // AG gelesen
            }
            __threadfence_system();
        }
    }

    barriere<GRID>();
    if (erster) {
        *(volatile u64 *)A.rundeDev   = runde;
        *(volatile u64 *)A.schrittDev = basis + (u64)K;
        __threadfence_system();
    }
#undef PIPE_WARTE_DATEN
#undef PIPE_WARTE_FENSTER
}

// ===========================================================================
// Hostseite
// ===========================================================================

static int gitterGroesse(const void *fn, int threads, int n4)
{
    int proSM = 0;
    if (cudaOccupancyMaxActiveBlocksPerMultiprocessor(&proSM, fn, threads, 0)
            != cudaSuccess || proSM < 1)
        return 1;
    int dev = 0;
    if (cudaGetDevice(&dev) != cudaSuccess) return 1;
    int sms = 0;
    if (cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, dev)
            != cudaSuccess || sms < 1)
        return 1;
    int g = proSM * sms;
    int noetig = (n4 + threads - 1) / threads;
    if (noetig < 1) noetig = 1;
    if (noetig < g) g = noetig;
    if (g < 1) g = 1;
    return g;
}

template<typename T>
static void startePipe(int kern, int la, PipeArgs &A, int threads,
                       int grid_n4, cudaStream_t strom)
{
#define HTCCL_PIPE_STARTE(LA)                                                  \
    do {                                                                       \
        if (kern == K_GITTER) {                                                \
            const void *fn =                                                   \
                (const void *)bar1_netz_pipe_kernel<T, LA, K_GITTER>;          \
            int g = gitterGroesse(fn, threads, grid_n4);                       \
            dim3 gd((unsigned)g), bd((unsigned)threads);                       \
            void *args[1] = { &A };                                            \
            cudaError_t e = cudaLaunchCooperativeKernel(fn, gd, bd, args, 0,   \
                                                        strom);                \
            TORCH_CHECK(e == cudaSuccess,                                      \
                        "htccl-bar1 pipe: cudaLaunchCooperativeKernel -> ",    \
                        cudaGetErrorString(e));                                \
            return;                                                            \
        }                                                                      \
        bar1_netz_pipe_kernel<T, LA, K_1BLK><<<1, threads, 0, strom>>>(A);     \
        TORCH_CHECK(cudaGetLastError() == cudaSuccess,                         \
                    "htccl-bar1 pipe: Kernelstart fehlgeschlagen");            \
        return;                                                                \
    } while (0)

    if (la == LA_MMIO) HTCCL_PIPE_STARTE(LA_MMIO);
    else               HTCCL_PIPE_STARTE(LA_CV);
#undef HTCCL_PIPE_STARTE
}

// Ein Tensor UEBER die exportierte Empfangsregion, ohne Kopie.
//
// Das ist der Angelpunkt des Direkt-Modus: `all_reduce` arbeitet ausser
// Platz und gibt einen frischen Tensor zurueck -- also bestimmen WIR, wo er
// liegt. Statt torch einen Puffer aussuchen zu lassen, den die Nachbarn
// nicht kennen, zeigt der Ergebnistensor in einen beim Aufbau exportierten
// Ringpuffer. Der Aufrufer bekommt einen ganz gewoehnlichen Tensor; er liegt
// nur in Speicher, in den die Nachbarkarten unmittelbar schreiben duerfen.
// Keine Registrierung im heissen Pfad, kein Umkopieren.
//
// `at::from_blob` OHNE Deleter: der Speicher gehoert dem Transport und wird
// von `close()` freigegeben. Ein Deleter waere hier der Fehler -- torch
// wuerde versuchen, eine cuMemMap-Adresse an den Caching-Allokator
// zurueckzugeben.
at::Tensor bar1_erg_tensor(int64_t ptr, at::Tensor muster)
{
    TORCH_CHECK(ptr != 0, "htccl-bar1 pipe: Ergebnispuffer ist ein Nullzeiger");
    TORCH_CHECK((ptr % 16) == 0,
                "htccl-bar1 pipe: Ergebnispuffer ", ptr,
                " ist nicht 16-Byte-ausgerichtet -- jeder Schreibvorgang in "
                "die WC-Apertur ist 128 Bit breit");
    return at::from_blob((void *)(uintptr_t)ptr, muster.sizes(),
                         muster.options());
}

void bar1_netz_pipe(at::Tensor inp, at::Tensor out,
                    int64_t rank, int64_t world,
                    std::vector<int64_t> peer_nutz,
                    std::vector<int64_t> peer_flag,
                    std::vector<int64_t> peer_erg,
                    int64_t eigen_nutz, int64_t eigen_flag,
                    int64_t schlitz, int64_t off_pipe, int64_t fbasis_pipe,
                    int64_t k_chunks, int64_t tiefe, int64_t vorlauf,
                    int64_t quittung, int64_t direkt,
                    at::Tensor runde_dev, at::Tensor schritt_dev,
                    at::Tensor ctl_dev,
                    int64_t deckel_zyklen, int64_t threads, int64_t kern,
                    int64_t ladeform)
{
    const int R = (int)world, r = (int)rank;
    const int K = (int)k_chunks, TT = (int)tiefe, PP = (int)vorlauf;
    TORCH_CHECK(R >= 2 && R <= HTCCL_PIPE_MAX_RANKS,
                "htccl-bar1 pipe: world ", R, " ausserhalb 2..",
                HTCCL_PIPE_MAX_RANKS);
    TORCH_CHECK(r >= 0 && r < R, "htccl-bar1 pipe: rank ausserhalb");
    TORCH_CHECK(inp.is_contiguous() && out.is_contiguous(),
                "htccl-bar1 pipe: nur zusammenhaengende Tensoren");
    TORCH_CHECK(inp.numel() == out.numel() &&
                inp.scalar_type() == out.scalar_type(),
                "htccl-bar1 pipe: in und out passen nicht zusammen");
    TORCH_CHECK(inp.data_ptr() != out.data_ptr(),
                "htccl-bar1 pipe: in und out duerfen nicht dasselbe sein -- "
                "die Reduktion liest 'in' noch, waehrend 'out' schon steht");
    TORCH_CHECK((int64_t)peer_nutz.size() == world &&
                (int64_t)peer_flag.size() == world &&
                (int64_t)peer_erg.size() == world,
                "htccl-bar1 pipe: Peer-Tabelle hat die falsche Laenge");
    // PP >= 2: bei PP == 1 fielen das Senden von Chunk c und sein Verbrauch
    // in dieselbe Schleifenrunde, und die Wartebedingung stuende VOR der
    // Veroeffentlichung, auf die sie wartet -- ein Selbstblock, den kein
    // Zeitdeckel zu einem Fehler macht, sondern nur zu einem langsamen.
    // PP <= TT: der Ring muss mindestens so tief sein wie der Vorlauf, sonst
    // ueberholt der Zeitplan die Schlitze.
    TORCH_CHECK(TT >= 2 && TT <= HTCCL_PIPE_MAX_TIEFE,
                "htccl-bar1 pipe: Ringtiefe ", TT, " ausserhalb 2..",
                HTCCL_PIPE_MAX_TIEFE);
    TORCH_CHECK(PP >= 2 && PP <= TT,
                "htccl-bar1 pipe: Vorlauf ", PP, " ausserhalb 2..", TT,
                " -- er darf die Ringtiefe nicht ueberschreiten");
    TORCH_CHECK(K >= TT, "htccl-bar1 pipe: K=", K, " < T=", TT,
                " -- weniger Chunks als Ringtiefe");
    TORCH_CHECK(schlitz > 0 && (schlitz % 16) == 0,
                "htccl-bar1 pipe: Schlitzgroesse ", schlitz,
                " ist kein positives Vielfaches von 16");
    TORCH_CHECK((off_pipe % 16) == 0,
                "htccl-bar1 pipe: off_pipe ", off_pipe,
                " ist kein Vielfaches von 16 -- jeder Schreibvorgang in die "
                "WC-Apertur ist 128 Bit breit und muss ausgerichtet sein");

    const size_t nbytes = (size_t)inp.numel() * (size_t)inp.element_size();
    TORCH_CHECK(nbytes % 16 == 0,
                "htccl-bar1 pipe: Nutzlast ", nbytes,
                " ist kein Vielfaches von 16 Byte");
    TORCH_CHECK(((uintptr_t)inp.data_ptr() % 16) == 0 &&
                ((uintptr_t)out.data_ptr() % 16) == 0,
                "htccl-bar1 pipe: Puffer nicht auf 16 Byte ausgerichtet");
    const int n4 = (int)(nbytes / 16);
    TORCH_CHECK(n4 >= R, "htccl-bar1 pipe: weniger als ein Paket je Rang");

    // Nahtstellenpruefung. Gerechnet wird mit chunkGrenzen SELBST, ueber
    // alle (Chunk, Rang)-Paare -- nicht mit einer geschlossenen Zweitfassung.
    int maxStueck = 0, maxChunk = 0;
    for (int c = 0; c < K; ++c) {
        int coff, clen;
        chunkGrenzen(n4, c, K, &coff, &clen);
        if (clen > maxChunk) maxChunk = clen;
        for (int z = 0; z < R; ++z) {
            int poff, plen;
            chunkGrenzen(clen, z, R, &poff, &plen);
            if (plen > maxStueck) maxStueck = plen;
        }
    }
    TORCH_CHECK((int64_t)maxStueck * 16 <= schlitz,
                "htccl-bar1 pipe: groesstes Stueck ", (int64_t)maxStueck * 16,
                " Byte passt nicht in den Pipe-Schlitz von ", schlitz,
                " Byte (K=", K, ", T=", TT, ", R=", R,
                "). Der Aufrufer haette handles() fragen muessen.");

    PipeArgs A;
    std::memset(&A, 0, sizeof(A));
    A.in           = (const uint4 *)inp.data_ptr();
    A.out          = (uint4 *)out.data_ptr();
    A.rundeDev     = (u64 *)runde_dev.data_ptr();
    A.schrittDev   = (u64 *)schritt_dev.data_ptr();
    A.ctlStatus    = (unsigned int *)ctl_dev.data_ptr();
    A.abbruchDev   = ((unsigned int *)ctl_dev.data_ptr()) + 1;
    A.n4           = n4;
    A.R            = R;
    A.rang         = r;
    A.K            = K;
    A.TT           = TT;
    A.PP           = PP;
    A.quittung     = (int)quittung;
    A.direkt       = (int)direkt;
    A.deckelZyklen = (u64)deckel_zyklen;
    A.schlitz4     = (long long)(schlitz / 16);
    A.klasse4      = (long long)(R - 1) * A.schlitz4;

    // Der AG-Ring liegt hinter dem RS-Ring; jeder Ring haelt T*(R-1)
    // Schlitze. Innerhalb eines Rings ist die Ordnung (Klasse, Position),
    // damit die Klasse mit einem einzigen Vielfachen von klasse4 erreicht
    // wird.
    const long long ringBytes = (long long)TT * (R - 1) * schlitz;
    for (int z = 0; z < R; ++z) {
        if (z == r) continue;
        // Meine Position in der aufsteigenden Peer-Liste des Empfaengers z.
        // NICHT (r < z) -- dieselbe Herleitung wie bei netz und a2a.
        const long long p = (long long)(r - (r > z ? 1 : 0));
        char *zb = (char *)(uintptr_t)peer_nutz[z] + off_pipe;
        A.sendRS[z] = (uint4 *)(zb + p * schlitz);
        A.sendAG[z] = (uint4 *)(zb + ringBytes + p * schlitz);
        if (direkt) {
            TORCH_CHECK(peer_erg[z] != 0,
                        "htccl-bar1 pipe: Direkt-Modus ohne Ergebnispuffer "
                        "von Rang ", z);
            TORCH_CHECK((peer_erg[z] % 16) == 0,
                        "htccl-bar1 pipe: Ergebnispuffer von Rang ", z,
                        " ist nicht 16-Byte-ausgerichtet");
            A.ergAn[z] = (uint4 *)(uintptr_t)peer_erg[z];
        }
    }
    if (direkt) {
        // Der eigene Ergebnispuffer MUSS `out` sein -- sonst schriebe der
        // Peer an eine Stelle, die der Aufrufer nie zu sehen bekommt. Das
        // ist die Naht zwischen Python und Kern, und sie wird geprueft, nicht
        // vorausgesetzt.
        TORCH_CHECK((uintptr_t)out.data_ptr() == (uintptr_t)peer_erg[r],
                    "htccl-bar1 pipe: im Direkt-Modus muss `out` der eigene "
                    "Ergebnispuffer sein (out=", (int64_t)(uintptr_t)out.data_ptr(),
                    ", erg=", peer_erg[r], ")");
    }
    {
        char *mb = (char *)(uintptr_t)eigen_nutz + off_pipe;
        for (int s = 0; s < R; ++s) {
            if (s == r) continue;
            const long long p = (long long)(s - (s > r ? 1 : 0));
            A.recvRS[s] = (const uint4 *)(mb + p * schlitz);
            A.recvAG[s] = (const uint4 *)(mb + ringBytes + p * schlitz);
        }
    }

    // Zaehlerzeilen: vier Familien zu je R Zeilen a 256 Byte.
    //   0 tailRS  1 tailAG  2 headRS  3 headAG
    // tail schreibe ich beim EMPFAENGER in MEINE Zeile und lese bei mir
    // DESSEN Zeile; head genau andersherum. Beides sind gebuchte
    // Schreibvorgaenge und lokale Lesevorgaenge -- ein Lesevorgang aus einer
    // fremden BAR waere ein Umlauf.
    for (int ph = 0; ph < 2; ++ph) {
        for (int q = 0; q < R; ++q) {
            if (q == r) continue;
            char *pf = (char *)(uintptr_t)peer_flag[q] + fbasis_pipe;
            char *ef = (char *)(uintptr_t)eigen_flag   + fbasis_pipe;
            A.tailAn [ph][q] = (u64 *)(pf + (size_t)(ph * R + r) * 256u);
            A.tailVon[ph][q] = (const u64 *)(ef + (size_t)(ph * R + q) * 256u);
            A.headAn [ph][q] = (u64 *)(pf + (size_t)((2 + ph) * R + r) * 256u);
            A.headVon[ph][q] = (const u64 *)(ef + (size_t)((2 + ph) * R + q) * 256u);
        }
    }

    auto strom = at::cuda::getCurrentCUDAStream().stream();
    // Das Gitter wird nach dem GROESSTEN CHUNK bemessen, nicht nach der
    // ganzen Nutzlast: mehr Bloecke als ein Chunk Arbeit hat, warten nur an
    // der grid.sync().
    switch (inp.scalar_type()) {
        case at::kFloat:
            startePipe<float>((int)kern, (int)ladeform, A, (int)threads,
                              maxChunk, strom);
            break;
        case at::kHalf:
            startePipe<__half>((int)kern, (int)ladeform, A, (int)threads,
                               maxChunk, strom);
            break;
        case at::kBFloat16:
            startePipe<__nv_bfloat16>((int)kern, (int)ladeform, A, (int)threads,
                                      maxChunk, strom);
            break;
        default:
            TORCH_CHECK(false, "htccl-bar1 pipe: Datentyp ", inp.scalar_type(),
                        " wird nicht unterstuetzt (float32/float16/bfloat16)");
    }
}
"""

_CPP_SRC = """
at::Tensor bar1_erg_tensor(int64_t ptr, at::Tensor muster);

void bar1_netz_pipe(at::Tensor inp, at::Tensor out,
                    int64_t rank, int64_t world,
                    std::vector<int64_t> peer_nutz,
                    std::vector<int64_t> peer_flag,
                    std::vector<int64_t> peer_erg,
                    int64_t eigen_nutz, int64_t eigen_flag,
                    int64_t schlitz, int64_t off_pipe, int64_t fbasis_pipe,
                    int64_t k_chunks, int64_t tiefe, int64_t vorlauf,
                    int64_t quittung, int64_t direkt,
                    at::Tensor runde_dev, at::Tensor schritt_dev,
                    at::Tensor ctl_dev,
                    int64_t deckel_zyklen, int64_t threads, int64_t kern,
                    int64_t ladeform);
"""


# ===========================================================================
# Uebersetzen
# ===========================================================================


def lade_pipe_ext(cpu_group, ptxas_verbose: bool = False):
    """Die CUDA-Erweiterung mit ``bar1_netz_pipe``.

    Eigener Name, eigenes Bauverzeichnis, eigener Cache-Schluessel: torch
    schluesselt sein Bauverzeichnis nur nach dem Namen, und eine .so unter
    fremdem Namen bekaeme ein Rang gereicht, dessen Karte sie nicht
    ausfuehren kann. Mechanik und Cache-Hygiene kommen aus ``htccl_device``,
    genau wie bei ``htccl_bar1_ext``.
    """
    global _ext
    if _ext is not None:
        return _ext

    from torch.utils.cpp_extension import load_inline

    from sglang.srt.distributed.device_communicators.htccl_device import (
        _build_flags,
        _ext_cache_guarded,
        _local_vendor,
        _resolve_build_arches,
    )

    vendor = _local_vendor()
    if vendor != "cuda":
        raise RuntimeError(
            f"HTCCL-BAR1-PIPE: der Direktpfad ist NVIDIA-eigen (BAR1-Apertur, "
            f"PTX-Einschuebe wie ld.mmio.relaxed.sys). Dieser Rang meldet "
            f"{vendor!r}."
        )
    by_vendor = _resolve_build_arches(cpu_group)
    arches = by_vendor.get(vendor, [])
    flags = list(_build_flags(vendor, arches))
    if ptxas_verbose or os.environ.get("SGLANG_HTCCL_BAR1_PTXAS_V", "0") == "1":
        flags += ["-Xptxas", "-v"]
    name = "htccl_bar1_pipe_ext_" + vendor
    if arches:
        name += "_" + "_".join(a.replace(".", "") for a in arches)
    t0 = time.time()
    with _ext_cache_guarded(name) as build_dir:
        _ext = load_inline(
            name=name,
            cpp_sources=_CPP_SRC,
            cuda_sources=_CUDA_SRC,
            functions=["bar1_netz_pipe", "bar1_erg_tensor"],
            extra_cuda_cflags=flags or None,
            verbose=bool(ptxas_verbose),
            build_directory=str(build_dir) if build_dir is not None else None,
        )
    logger.info(
        "HTCCL-BAR1-PIPE: Erweiterung %r fuer Boegen %s in %.1f s gebaut.",
        name, ",".join(arches) or "<torch-Vorgabe>", time.time() - t0,
    )
    return _ext


# ===========================================================================
# Byte-Beleg
# ===========================================================================


def byte_beleg_pipe(transport, runden: int = 0) -> bool:
    """Der Beleg fuer ``netz_pipe``: mehrere Runden, ungleiche Zerlegung.

    Ein Beleg, der nur EINE Runde prueft, findet genau den gefaehrlichsten
    Fehler nicht -- die Schlitz-Wiederverwendung schlaegt erst zu, wenn eine
    Schlitzklasse ein zweites Mal drankommt. Deshalb:

    * **mehrere aufeinanderfolgende Runden** (Vorgabe ``2T + 3``, mindestens
      so viele, dass jede Klasse mehrfach getroffen wird), jede mit ANDEREN
      Zahlen -- ein liegengebliebener Schlitz faellt sonst nicht auf, weil
      der alte Inhalt zufaellig richtig waere;
    * **ungleiche Chunkzahlen**: Nutzlasten, deren Paketzahl weder durch
      ``K`` noch durch ``R`` teilbar ist, sodass ``chunkGrenzen`` den Rest
      auf die vorderen Chunks legt und die Stuecke verschieden lang sind;
    * **Restchunks**: die kleinste erlaubte Nutzlast und Groessen kurz
      ueber einer Zweierpotenz.

    Zur Frage "Chunk kein Vielfaches von 16": bei ``all_reduce`` kann das
    **nicht** auftreten und das ist nachgerechnet, nicht angenommen. Die
    Einheit der Zerlegung ist das 128-Bit-Paket (``n4 = nbytes/16``,
    ``nbytes % 16 == 0`` wird an der Naht erzwungen), also ist jede
    Chunk- und Stueckgrenze ein Vielfaches von 16 Byte. Der Beleg prueft
    diese Behauptung ausdruecklich, statt sie zu glauben.

    Verglichen wird **bitgleich**. Die Eingaben sind kleine ganze Zahlen,
    in float32/float16/bfloat16 exakt darstellbar, und der Kern summiert in
    fester Reihenfolge (eigener Beitrag, dann Peers aufsteigend). Ein
    Vergleich mit Toleranz wuerde genau die Fehlerklasse verdecken, um die
    es hier geht: ein Schlitz, der einen Beitrag zu frueh oder zu spaet
    traegt, verschiebt das Ergebnis um einen ganzen Summanden.
    """
    import torch
    import torch.distributed as dist

    if not transport.pipe_an:
        return False
    tiefe = int(transport.pipe_t)
    if runden <= 0:
        runden = 2 * tiefe + 3

    welt = transport.welt
    rang = transport.rank
    geraet = transport.device
    chunk_max = int(transport._geo["chunk_max"])
    max_bytes = int(transport.max_bytes)

    # Groessen: absichtlich krumm. 16*R*T ist die kleinste, bei der jeder
    # Chunk jedes Rangs mindestens ein Paket bekommt; die uebrigen sind so
    # gewaehlt, dass n4 weder durch K noch durch R aufgeht.
    kandidaten = [
        16 * welt * tiefe,
        16 * (welt * tiefe + 1),
        16 * 1021,                    # Primzahl an Paketen
        (1 << 20) + 16 * 3,
        (4 << 20) + 16 * 7,
    ]
    groessen = []
    for n in kandidaten:
        if n < transport.min_bytes or n > max_bytes:
            continue
        if n % 16 or n // 16 < welt:
            continue
        groessen.append(n)
    if not groessen:
        logger.warning(
            "HTCCL-BAR1-PIPE: keine Probengroesse passt zwischen %d und %d "
            "Byte -- der Beleg kann nichts aussagen, also meldet er sich ab.",
            transport.min_bytes, max_bytes,
        )
        return False

    alles_gut = True
    for nbytes in groessen:
        n4 = nbytes // 16
        k = transport._pipe_k(nbytes)
        if k is None:
            logger.info(
                "HTCCL-BAR1-PIPE: %d Byte traegt der gepipelinete Weg nicht "
                "(kein K passt) -- uebersprungen, nicht gefallen.", nbytes,
            )
            continue

        # Behauptung 1, nachgerechnet statt geglaubt: keine Chunk- oder
        # Stueckgrenze faellt auf eine Adresse, die kein Vielfaches von 16
        # ist, und kein Stueck sprengt den Schlitz.
        schlitz = pipe_schlitz_bytes(chunk_max, tiefe)
        for c in range(k):
            coff, clen = _chunk_grenzen(n4, c, k)
            if (coff * 16) % 16 or (clen * 16) % 16:
                raise AssertionError("Chunkgrenze nicht 16-Byte-ausgerichtet")
            for z in range(welt):
                poff, plen = _chunk_grenzen(clen, z, welt)
                if ((coff + poff) * 16) % 16:
                    raise AssertionError("Stueckgrenze nicht ausgerichtet")
                if plen * 16 > schlitz:
                    raise AssertionError(
                        f"Stueck {plen * 16} > Schlitz {schlitz}"
                    )

        n = nbytes // 4
        for runde in range(runden):
            # Jede Runde andere Zahlen, und je Rang verschieden -- ein
            # Schlitz, der den Inhalt der Vorrunde traegt, faellt damit auf.
            welle = torch.arange(n, dtype=torch.float32, device=geraet)
            eig = ((welle % 7.0) + 1.0) * float(rang + 1) + float(runde * 13)
            erwartet = torch.zeros(n, dtype=torch.float32, device=geraet)
            for q in range(welt):
                erwartet += ((welle % 7.0) + 1.0) * float(q + 1) + float(runde * 13)
            dist.barrier(group=transport.cpu_group)
            ist = transport._pipe_all_reduce(eig, k)
            torch.cuda.synchronize(geraet)
            schlecht = int((ist != erwartet).sum().item())
            if schlecht:
                erste = int((ist != erwartet).nonzero()[0].item())
                logger.warning(
                    "HTCCL-BAR1-PIPE: Byte-Beleg GEFALLEN bei %d Byte, K=%d, "
                    "T=%d, Runde %d/%d: %d von %d Werten falsch, erster bei "
                    "Index %d (ist %r, soll %r).",
                    nbytes, k, tiefe, runde + 1, runden, schlecht, n, erste,
                    float(ist[erste]), float(erwartet[erste]),
                )
                alles_gut = False
                break
        else:
            logger.info(
                "HTCCL-BAR1-PIPE: Byte-Beleg bestanden: %d Byte (n4=%d, K=%d, "
                "T=%d), %d Runden, 0 von %d Werten falsch.",
                nbytes, n4, k, tiefe, runden, n,
            )
    # Gruppenweit EINE Antwort -- sonst antwortete `handles` rangabhaengig
    # und der eine Rang liefe ins Kollektiv, waehrend der andere abbiegt.
    # Ein Broadcast von Rang 0 taete das NICHT: gefallen ist der Beleg,
    # sobald IRGENDEIN Rang ein falsches Wort gesehen hat.
    traeger: list = [None] * welt
    dist.all_gather_object(traeger, bool(alles_gut), group=transport.cpu_group)
    return all(bool(x) for x in traeger)
