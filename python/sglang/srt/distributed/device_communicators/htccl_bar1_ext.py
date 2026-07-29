# SPDX-License-Identifier: Apache-2.0
"""Native Anteile des BAR1-Direkttransports.

Zwei getrennte Erweiterungen, weil sie verschiedene Voraussetzungen haben:

``htccl_bar1_ext``  (CUDA)
    Die beiden Kollektiv-Kernel ``netz`` und ``ring``, portiert aus
    ``/spinning/nvidia-open-595/bar1_kollektiv.cu``. Braucht nur nvcc und
    torch, laeuft ueberall, wo ``htccl_device`` auch baut.

``htccl_bar1_dmabuf_ext``  (nur C++)
    Der dma-buf-Export ueber die RM-Ioctls. Braucht die **Kopfdateien der
    quelloffenen NVIDIA-Kernelmodule**; ohne sie wird er nicht gebaut und
    der Transport meldet sich ab. Aus Python ist der Weg nicht nachbaubar:
    ``NV0000_CTRL_CMD_OS_UNIX_IMPORT_OBJECT_FROM_FD`` und
    ``NV_ESC_EXPORT_TO_DMABUF_FD`` verlangen die UAPI-Strukturen
    (``nvos.h``, ``nv-ioctl.h``, ``class/cl0000.h``), und deren Groessen und
    Feldversaetze sind versionsgebunden.

Was aus der Sonde uebernommen ist und was sich geaendert hat
------------------------------------------------------------
UEBERNOMMEN, moeglichst buchstaeblich: ``schreibeV4`` (``st.global.wt``),
``leseV4`` (``ld.global.cv``), ``flaggeLesen`` (``ld.mmio.relaxed.sys`` bzw.
``ld.global.cv``), ``leseFluss``, ``barriere<GRID>``, die Schrittfolge und
die Sperrenzahl beider Kernel, die Chunkgeometrie (Rest auf die vorderen
Chunks), die Flaggenzeilen zu 256 Byte je (Topologie, Schritt, Sender) und
die Gittergroessenrechnung.

GEAENDERT, jeweils mit Grund:

* **Beliebiges R statt fest drei.** ``RANGE_N`` war eine
  Uebersetzungskonstante; hier ist ``R`` ein Feld der Argumentstruktur. Die
  Schleifen laufen ueber ``R``, die Felder sind auf
  ``HTCCL_BAR1_MAX_RANKS`` dimensioniert.
* **Datentypen.** Die Sonde rechnete ausschliesslich ``float``. Hier ist
  die Reduktion ueber ``float``/``half``/``bfloat16`` templatisiert. Die
  **Zugriffsbreite bleibt 128 Bit** (``uint4``) -- das ist die Eigenschaft,
  an der die Messung haengt, nicht der Elementtyp. Der ``float``-Pfad ist
  bitgleich zur Sonde.
* **Keine Selbstpruefung.** ``sollWert``/``basisWert``/``fehlerZahl`` waren
  Messwerkzeug der Sonde (sie kannte ihre Eingabe). Ein Transport kennt sie
  nicht; die Pruefung ist ersatzlos entfallen statt durch eine Attrappe
  ersetzt zu werden. Der Byte-Beleg beim Aufbau bleibt.
* **Kein Rundenterm ``rt``.** Ebenfalls Messwerkzeug (er machte die Nutzlast
  je Runde verschieden, damit ein liegengebliebener Puffer auffaellt).
* **Rundennummer im Geraetespeicher.** Die Sonde reichte ``runde`` je Start
  als Wert herein. Das ueberlebt keine CUDA-Graph-Aufzeichnung -- ein
  aufgezeichneter Start wiederholte ewig dieselbe Zahl und jede Flagge
  stuende sofort. Die Runde liegt deshalb in einem Geraetewort und wird im
  Kernel gelesen und am Ende fortgeschrieben, genau wie ``htccl_device`` es
  mit seiner Folgenummer macht.
* **Keine Phasenuhr.** ``%globaltimer``-Stempel und der Ablageschreibvorgang
  am Kernelende gehoeren in die Sonde, nicht in den heissen Pfad.
"""

from __future__ import annotations

import logging
import os
import pathlib
import time
from typing import Optional

logger = logging.getLogger(__name__)

_ext = None
_dmabuf_ext = None
_dmabuf_grund = ""

#: Groesster Verbund, fuer den die Argumentfelder Platz haben. Acht, weil
#: die Struktur damit rund 1 KiB gross bleibt und sicher in den 4-KiB-Bereich
#: fuer Kernelparameter passt.
MAX_RANGE = 8

#: Wo die Kopfdateien der quelloffenen Kernelmodule liegen. Rangeinheitlich
#: wie jede andere SGLANG_HTCCL*-Variable.
NV_QUELLE_VORGABE = "/spinning/nvidia-open-595"

#: Unterverzeichnisse relativ dazu -- exakt die drei, mit denen
#: sonden/dmabuf_p2p_probe.cpp uebersetzt wird.
NV_INCLUDES = (
    "kernel-open/common/inc",
    "src/common/sdk/nvidia/inc",
    "src/nvidia/arch/nvalloc/unix/include",
)


# ===========================================================================
# Kernelquelltext
# ===========================================================================

_CUDA_SRC = r"""
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <cooperative_groups.h>
#include <vector>

namespace cg = cooperative_groups;

#define HTCCL_BAR1_MAX_RANKS 8
#define HTCCL_BAR1_MAX_STEPS (2 * (HTCCL_BAR1_MAX_RANKS - 1))

// Kernvarianten, unveraendert aus der Sonde.
#define K_1BLK   0
#define K_GITTER 1
// Ladeform der Flagge. LA_MMIO ist die einzige echte Cache-Umgehung und die
// Vorgabe der Sonde (Lauf-Feld uncachedLeseart = LA_MMIO).
#define LA_CV    0
#define LA_MMIO  2

using u64 = unsigned long long;

// ---------------------------------------------------------------------------
// Zugriffe, die keine Cache-Stufe verschlucken darf. Buchstaeblich aus
// bar1_kollektiv.cu: die Bytes kommen per DMA von einer FREMDEN Karte in den
// Framebuffer, eine wiederverwendete Cachezeile zeigt den alten Stand.
// ---------------------------------------------------------------------------
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

// Lesefluss fuer den Fall, dass Nutzlast und Flagge an VERSCHIEDENEN
// PCIe-Zielen liegen. Bleibt uebernommen, wird von diesem Transport aber
// nicht benutzt (beide liegen auf derselben Zielkarte, Reihenfolge je
// Requester/Completer-Paar gilt); der Schalter existiert, damit die
// Quittungsart spaeter ohne Codeaenderung geprueft werden kann.
__device__ __forceinline__ void leseFluss(const uint4 *peerRecv, int n4)
{
    if (n4 <= 0) return;
    unsigned int d = 0;
    asm volatile("ld.mmio.relaxed.sys.global.u32 %0, [%1];"
                 : "+r"(d) : "l"(peerRecv + (n4 - 1)) : "memory");
}

// ---------------------------------------------------------------------------
// Elementweise Addition INNERHALB eines 128-Bit-Pakets.
//
// Die Sonde kannte nur float4. Die Zugriffsbreite bleibt hier dieselbe --
// geaendert hat sich nur, wie die 16 Byte gedeutet werden. Fuer float ist der
// erzeugte Code identisch zur Sonde.
// ---------------------------------------------------------------------------
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
    // Vor sm_80 gibt es keine bf16-Rechenwerke. Ueber float, damit derselbe
    // Quelltext uebersetzt -- gemessen wurde bf16 nur auf sm_86 und sm_120.
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

// ---------------------------------------------------------------------------
// Chunkgeometrie. Chunk j von n4 Paketen ueber R Raenge; der Rest n4 % R geht
// auf die VORDEREN Chunks. Unveraendert aus der Sonde, nur R statt RANGE_N.
// Dieselbe Rechnung steht host-seitig noch einmal in der Nahtstellenpruefung
// (bar1_all_reduce): eine Naht, die auf beiden Seiten mit DERSELBEN falschen
// Formel geprueft wuerde, faellt nicht auf.
// ---------------------------------------------------------------------------
__device__ __host__ __forceinline__ void chunkGrenzen(int n4, int j, int R,
                                                      int *off, int *len)
{
    const int basis = n4 / R;
    const int rest  = n4 - basis * R;
    *len = basis + (j < rest ? 1 : 0);
    *off = j * basis + (j < rest ? j : rest);
}

// ---------------------------------------------------------------------------
// Argumente. Eine Struktur, damit der cooperative Start und der normale
// <<< >>>-Start nachweislich dasselbe uebergeben.
// ---------------------------------------------------------------------------
struct Bar1Args {
    const uint4 *in;          // lokal, VRAM
    uint4       *out;         // lokal, VRAM
    u64         *rundeDev;    // ein Wort, lokal: die laufende Rundennummer
    unsigned int *ctlStatus;  // 1 = Zeitlimit gerissen
    unsigned int *abbruchDev; // nur K_GITTER: gitterweites Abbruchbit
    int          n4;
    int          R;
    int          rang;
    u64          deckelZyklen;

    // netz: indiziert nach RANGNUMMER, eigener Eintrag bleibt leer.
    uint4       *nzSendRS[HTCCL_BAR1_MAX_RANKS];
    uint4       *nzSendAG[HTCCL_BAR1_MAX_RANKS];
    const uint4 *nzRecvRS[HTCCL_BAR1_MAX_RANKS];
    const uint4 *nzRecvAG[HTCCL_BAR1_MAX_RANKS];
    u64         *nzFlagAn [2][HTCCL_BAR1_MAX_RANKS];
    const u64   *nzFlagVon[2][HTCCL_BAR1_MAX_RANKS];

    // ring: indiziert nach SCHRITT, 0 .. 2*(R-1)-1.
    uint4       *rgSend   [HTCCL_BAR1_MAX_STEPS];
    const uint4 *rgRecv   [HTCCL_BAR1_MAX_STEPS];
    u64         *rgFlagAn [HTCCL_BAR1_MAX_STEPS];
    const u64   *rgFlagVon[HTCCL_BAR1_MAX_STEPS];
};

// ---------------------------------------------------------------------------
// Barriere -- der einzige Unterschied zwischen den Kernvarianten.
// ---------------------------------------------------------------------------
template<int GRID>
__device__ __forceinline__ void barriere(void)
{
    if (GRID == K_GITTER) cg::this_grid().sync();
    else                  __syncthreads();
}

// ---------------------------------------------------------------------------
// Phasen. Alle Kernel benutzen dieselben.
// ---------------------------------------------------------------------------

// Eigenen Beitrag per DMA in den Empfangsschlitz der Zielkarte.
__device__ __forceinline__ void sendePhase(const uint4 *__restrict__ in,
                                           uint4 *peerRecv,
                                           int n4, int tid, int nth)
{
    for (int j = tid; j < n4; j += nth) schreibeV4(peerRecv + j, in[j]);
}

// Ein Stueck an ALLE Peers, in EINER Schleife -- damit das Ergebnis nur
// einmal aus dem lokalen VRAM gelesen wird (verteilePhase der Sonde,
// verallgemeinert von zwei Zielen auf R-1).
__device__ __forceinline__ void verteilePhase(const uint4 *__restrict__ erg,
                                              uint4 *const *ziel,
                                              int R, int rang,
                                              int n4, int tid, int nth)
{
    for (int j = tid; j < n4; j += nth) {
        const uint4 v = erg[j];
        for (int z = 0; z < R; ++z) {
            if (z == rang) continue;
            schreibeV4(ziel[z] + j, v);
        }
    }
}

// Lokal ueber den eigenen Beitrag und ALLE R-1 empfangenen reduzieren.
// (reduziere3Phase der Sonde, verallgemeinert.)
template<typename T>
__device__ __forceinline__ void reduziereNPhase(const uint4 *__restrict__ in,
                                                uint4 *out,
                                                const uint4 *const *recv,
                                                int R, int rang,
                                                int n4, int tid, int nth)
{
    for (int j = tid; j < n4; j += nth) {
        uint4 s = in[j];
        for (int q = 0; q < R; ++q) {
            if (q == rang) continue;
            s = addV4<T>(s, leseV4(recv[q] + j));
        }
        out[j] = s;
    }
}

// Empfangenes uebernehmen (uebernehmePhase der Sonde, ohne die Pruefung).
__device__ __forceinline__ void uebernehmePhase(uint4 *out, const uint4 *recv,
                                                int n4, int tid, int nth)
{
    for (int j = tid; j < n4; j += nth) out[j] = leseV4(recv + j);
}

// Ring-Reduce-Scatter, Empfangsschritt: eigener Beitrag plus Teilsumme.
template<typename T>
__device__ __forceinline__ void ringAddiere(const uint4 *__restrict__ in,
                                            uint4 *out, const uint4 *recv,
                                            int n4, int tid, int nth)
{
    for (int j = tid; j < n4; j += nth) out[j] = addV4<T>(in[j], leseV4(recv + j));
}

// ---------------------------------------------------------------------------
// Rundennummer.
//
// Gelesen von JEDEM Thread am Kernelanfang (uniformer Wert, ein L2-Treffer),
// fortgeschrieben vom global ersten Thread am Kernelende -- auch auf dem
// Abbruchpfad. Wuerde bei Abbruch nicht fortgeschrieben, liefen die Raenge
// dauerhaft auseinander: der eine zaehlte den Aufruf, der andere nicht, und
// jede folgende Flagge stuende an der falschen Stelle.
// ---------------------------------------------------------------------------
__device__ __forceinline__ void rundeSchreiben(const Bar1Args &A, u64 runde)
{
    *(volatile u64 *)A.rundeDev = runde;
    __threadfence_system();
}

// ---------------------------------------------------------------------------
// TOPOLOGIE 'netz' -- Reduce-Scatter + Allgather ueber ALLE Paare, zwei
// Sperren. Aufbau unveraendert aus ar3_netz_kernel; die festen Raenge a und b
// sind durch Schleifen ueber alle Peers ersetzt.
//
// GETRENNTE SCHLITZE FUER RS UND AG: zwischen "ich lese meine RS-Schlitze"
// und "der andere schreibt seinen AG-Chunk" steht keine Ordnung. Mit einem
// gemeinsamen Schlitzsatz braeuchte es eine dritte Sperre.
// ---------------------------------------------------------------------------
template<typename T, int LA, int FLUSS, int GRID>
__global__ void bar1_netz_kernel(Bar1Args A)
{
    const int tid = (GRID == K_GITTER)
                        ? (int)(blockIdx.x * blockDim.x + threadIdx.x)
                        : (int)threadIdx.x;
    const int nth = (GRID == K_GITTER)
                        ? (int)(gridDim.x * blockDim.x)
                        : (int)blockDim.x;
    const bool erster = (tid == 0);
    const int  n4 = A.n4, R = A.R, r = A.rang;
    const u64  runde = *(const volatile u64 *)A.rundeDev + 1ull;

    __shared__ int abbruchS;
    if (GRID == K_1BLK) {
        if (threadIdx.x == 0) abbruchS = 0;
        __syncthreads();
    } else if (erster) {
        *(volatile unsigned int *)A.abbruchDev = 0u;
        __threadfence();
    }

    int offR, lenR;
    chunkGrenzen(n4, r, R, &offR, &lenR);

    // --- 1. Reduce-Scatter: Chunk z an Rang z ------------------------------
    for (int z = 0; z < R; ++z) {
        if (z == r) continue;
        int off, len;
        chunkGrenzen(n4, z, R, &off, &len);
        sendePhase(A.in + off, A.nzSendRS[z], len, tid, nth);
    }
    __threadfence_system();
    barriere<GRID>();

    if (erster) {
        for (int z = 0; z < R; ++z) {
            if (z == r) continue;
            if (FLUSS) {
                int off, len;
                chunkGrenzen(n4, z, R, &off, &len);
                leseFluss(A.nzSendRS[z], len);
            }
            schreibeU64(A.nzFlagAn[0][z], runde);
        }
        __threadfence_system();

        bool ab = false;
        long long t0 = clock64();
        for (;;) {
            bool alle = true;
            for (int s = 0; s < R; ++s) {
                if (s == r) continue;
                if (flaggeLesen<LA>(A.nzFlagVon[0][s]) != runde) { alle = false; break; }
            }
            if (alle) break;
            if ((u64)(clock64() - t0) > A.deckelZyklen) { ab = true; break; }
        }
        if (ab) {
            if (GRID == K_1BLK) abbruchS = 1;
            else { *(volatile unsigned int *)A.abbruchDev = 1u; __threadfence(); }
        }
    }
    barriere<GRID>();
    {
        const int abbruch = (GRID == K_1BLK)
                                ? abbruchS
                                : (int)*(volatile unsigned int *)A.abbruchDev;
        if (abbruch) {
            if (erster) {
                *A.ctlStatus = 1u;
                rundeSchreiben(A, runde);
            }
            return;
        }
    }
    __threadfence_system();

    // --- 2. eigenen Chunk reduzieren --------------------------------------
    {
        // Der Empfangsschlitz traegt NUR den eigenen Chunk, beginnt also bei 0
        // -- der Chunkversatz offR gilt fuer den LOKALEN Puffer, nicht fuer
        // den Schlitz.
        const uint4 *recv[HTCCL_BAR1_MAX_RANKS];
        for (int s = 0; s < R; ++s) recv[s] = A.nzRecvRS[s];
        reduziereNPhase<T>(A.in + offR, A.out + offR, recv, R, r, lenR, tid, nth);
    }
    barriere<GRID>();

    // --- 3. Allgather: der fertige eigene Chunk an alle anderen ------------
    verteilePhase(A.out + offR, A.nzSendAG, R, r, lenR, tid, nth);
    __threadfence_system();
    barriere<GRID>();

    if (erster) {
        for (int z = 0; z < R; ++z) {
            if (z == r) continue;
            if (FLUSS) leseFluss(A.nzSendAG[z], lenR);
            schreibeU64(A.nzFlagAn[1][z], runde);
        }
        __threadfence_system();

        bool ab = false;
        long long t0 = clock64();
        for (;;) {
            bool alle = true;
            for (int s = 0; s < R; ++s) {
                if (s == r) continue;
                if (flaggeLesen<LA>(A.nzFlagVon[1][s]) != runde) { alle = false; break; }
            }
            if (alle) break;
            if ((u64)(clock64() - t0) > A.deckelZyklen) { ab = true; break; }
        }
        if (ab) {
            if (GRID == K_1BLK) abbruchS = 1;
            else { *(volatile unsigned int *)A.abbruchDev = 1u; __threadfence(); }
        }
    }
    barriere<GRID>();
    {
        const int abbruch = (GRID == K_1BLK)
                                ? abbruchS
                                : (int)*(volatile unsigned int *)A.abbruchDev;
        if (abbruch) {
            if (erster) {
                *A.ctlStatus = 1u;
                rundeSchreiben(A, runde);
            }
            return;
        }
    }
    __threadfence_system();

    for (int s = 0; s < R; ++s) {
        if (s == r) continue;
        int off, len;
        chunkGrenzen(n4, s, R, &off, &len);
        uebernehmePhase(A.out + off, A.nzRecvAG[s], len, tid, nth);
    }
    barriere<GRID>();
    if (erster) rundeSchreiben(A, runde);
}

// ---------------------------------------------------------------------------
// TOPOLOGIE 'ring' -- Ring-Reduce-Scatter + Ring-Allgather, 2*(R-1) Sperren.
// Gesendet wird IMMER an (r+1)%R, empfangen IMMER von (r-1+R)%R.
// Unveraendert aus ar3_ring_kernel, nur RANGE_N -> A.R und ohne Pruefung.
// ---------------------------------------------------------------------------
template<typename T, int LA, int FLUSS, int GRID>
__global__ void bar1_ring_kernel(Bar1Args A)
{
    const int tid = (GRID == K_GITTER)
                        ? (int)(blockIdx.x * blockDim.x + threadIdx.x)
                        : (int)threadIdx.x;
    const int nth = (GRID == K_GITTER)
                        ? (int)(gridDim.x * blockDim.x)
                        : (int)blockDim.x;
    const bool erster = (tid == 0);
    const int  n4 = A.n4, R = A.R, r = A.rang;
    const u64  runde = *(const volatile u64 *)A.rundeDev + 1ull;

    __shared__ int abbruchS;
    if (GRID == K_1BLK) {
        if (threadIdx.x == 0) abbruchS = 0;
        __syncthreads();
    } else if (erster) {
        *(volatile unsigned int *)A.abbruchDev = 0u;
        __threadfence();
    }

    // ------------------------- Reduce-Scatter ------------------------------
    for (int s = 0; s < R - 1; ++s) {
        const int cs = (r - s + 2 * R) % R;         // gesendeter Chunk
        const int cr = (r - s - 1 + 2 * R) % R;     // empfangener Chunk
        int offS, lenS, offE, lenE;
        chunkGrenzen(n4, cs, R, &offS, &lenS);
        chunkGrenzen(n4, cr, R, &offE, &lenE);

        // Schritt 0 sendet den eigenen Beitrag, jeder weitere die im vorigen
        // Schritt gebildete Teilsumme.
        const uint4 *quelle = (s == 0) ? (A.in + offS)
                                       : (const uint4 *)(A.out + offS);
        sendePhase(quelle, A.rgSend[s], lenS, tid, nth);
        __threadfence_system();
        barriere<GRID>();

        if (erster) {
            if (FLUSS) leseFluss(A.rgSend[s], lenS);
            schreibeU64(A.rgFlagAn[s], runde);
            __threadfence_system();

            bool ab = false;
            long long t0 = clock64();
            while (flaggeLesen<LA>(A.rgFlagVon[s]) != runde) {
                if ((u64)(clock64() - t0) > A.deckelZyklen) { ab = true; break; }
            }
            if (ab) {
                if (GRID == K_1BLK) abbruchS = 1;
                else { *(volatile unsigned int *)A.abbruchDev = 1u; __threadfence(); }
            }
        }
        barriere<GRID>();
        {
            const int abbruch = (GRID == K_1BLK)
                                    ? abbruchS
                                    : (int)*(volatile unsigned int *)A.abbruchDev;
            if (abbruch) {
                if (erster) { *A.ctlStatus = 1u; rundeSchreiben(A, runde); }
                return;
            }
        }
        __threadfence_system();

        ringAddiere<T>(A.in + offE, A.out + offE, A.rgRecv[s], lenE, tid, nth);
        barriere<GRID>();
    }

    // --------------------------- Allgather ---------------------------------
    for (int s = 0; s < R - 1; ++s) {
        const int sl = (R - 1) + s;                 // Schlitz + Flaggenzeile
        const int cs = (r + 1 - s + 2 * R) % R;     // gesendeter Chunk
        const int cr = (r - s + 2 * R) % R;         // uebernommener Chunk
        int offS, lenS, offE, lenE;
        chunkGrenzen(n4, cs, R, &offS, &lenS);
        chunkGrenzen(n4, cr, R, &offE, &lenE);

        sendePhase((const uint4 *)(A.out + offS), A.rgSend[sl], lenS, tid, nth);
        __threadfence_system();
        barriere<GRID>();

        if (erster) {
            if (FLUSS) leseFluss(A.rgSend[sl], lenS);
            schreibeU64(A.rgFlagAn[sl], runde);
            __threadfence_system();

            bool ab = false;
            long long t0 = clock64();
            while (flaggeLesen<LA>(A.rgFlagVon[sl]) != runde) {
                if ((u64)(clock64() - t0) > A.deckelZyklen) { ab = true; break; }
            }
            if (ab) {
                if (GRID == K_1BLK) abbruchS = 1;
                else { *(volatile unsigned int *)A.abbruchDev = 1u; __threadfence(); }
            }
        }
        barriere<GRID>();
        {
            const int abbruch = (GRID == K_1BLK)
                                    ? abbruchS
                                    : (int)*(volatile unsigned int *)A.abbruchDev;
            if (abbruch) {
                if (erster) { *A.ctlStatus = 1u; rundeSchreiben(A, runde); }
                return;
            }
        }
        __threadfence_system();

        uebernehmePhase(A.out + offE, A.rgRecv[sl], lenE, tid, nth);
        barriere<GRID>();
    }

    if (erster) rundeSchreiben(A, runde);
}

// ===========================================================================
// Hostseite
// ===========================================================================

// Gittergroesse der Variante 'gitter'. Zwei Deckel, beide notwendig
// (unveraendert aus der Sonde):
//   1. Nie mehr Bloecke, als GLEICHZEITIG resident sein koennen -- sonst
//      wartet grid.sync() auf einen Block, der gar nicht laeuft.
//   2. Mehr Bloecke als Arbeit ist sinnlos: ceil(n4/threads).
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

// Startet den gewaehlten Kernel. Die Ladeform der Flagge, der Lesefluss und
// die Barrierenart sind Template-Argumente -- in den Warteschleifen bleibt
// kein Zweig uebrig.
template<typename T>
static void starte(int algo, int kern, int la, int fluss, Bar1Args &A,
                   int threads, cudaStream_t strom)
{
#define HTCCL_BAR1_STARTE(KERNFN, LA, FL)                                      \
    do {                                                                       \
        if (kern == K_GITTER) {                                                \
            const void *fn = (const void *)KERNFN<T, LA, FL, K_GITTER>;        \
            int g = gitterGroesse(fn, threads, A.n4);                          \
            dim3 gd((unsigned)g), bd((unsigned)threads);                       \
            void *args[1] = { &A };                                            \
            cudaError_t e = cudaLaunchCooperativeKernel(fn, gd, bd, args, 0,   \
                                                        strom);                \
            TORCH_CHECK(e == cudaSuccess,                                      \
                        "htccl-bar1: cudaLaunchCooperativeKernel -> ",         \
                        cudaGetErrorString(e));                                \
            return;                                                            \
        }                                                                      \
        KERNFN<T, LA, FL, K_1BLK><<<1, threads, 0, strom>>>(A);                \
        TORCH_CHECK(cudaGetLastError() == cudaSuccess,                         \
                    "htccl-bar1: Kernelstart fehlgeschlagen");                 \
        return;                                                                \
    } while (0)

#define HTCCL_BAR1_WAEHLE(KERNFN)                                              \
    do {                                                                       \
        if (la == LA_MMIO) {                                                   \
            if (fluss) HTCCL_BAR1_STARTE(KERNFN, LA_MMIO, 1);                  \
            else       HTCCL_BAR1_STARTE(KERNFN, LA_MMIO, 0);                  \
        } else {                                                               \
            if (fluss) HTCCL_BAR1_STARTE(KERNFN, LA_CV, 1);                    \
            else       HTCCL_BAR1_STARTE(KERNFN, LA_CV, 0);                    \
        }                                                                      \
    } while (0)

    if (algo == 0) HTCCL_BAR1_WAEHLE(bar1_netz_kernel);
    else           HTCCL_BAR1_WAEHLE(bar1_ring_kernel);
#undef HTCCL_BAR1_WAEHLE
#undef HTCCL_BAR1_STARTE
}

// ---------------------------------------------------------------------------
// bar1_all_reduce
//
// `peer_nutz` / `peer_flag` tragen je Rang den GERAETEZEIGER DIESER Karte auf
// die BAR1-Region des Ziels; der eigene Eintrag ist der lokale Zeiger auf die
// eigene Region. Die Schlitzversaetze werden hier gerechnet, nicht in Python:
// derselbe Code, der sie in die Kernelargumente schreibt, prueft sie auch.
// ---------------------------------------------------------------------------
void bar1_all_reduce(at::Tensor inp, at::Tensor out,
                     int64_t rank, int64_t world, int64_t algo,
                     std::vector<int64_t> peer_nutz,
                     std::vector<int64_t> peer_flag,
                     int64_t eigen_nutz, int64_t eigen_flag,
                     int64_t chunk_max, int64_t off_netz, int64_t off_ring,
                     at::Tensor runde_dev, at::Tensor ctl_dev,
                     int64_t deckel_zyklen, int64_t threads, int64_t kern,
                     int64_t ladeform, int64_t fluss)
{
    const int R = (int)world, r = (int)rank;
    TORCH_CHECK(R >= 2 && R <= HTCCL_BAR1_MAX_RANKS,
                "htccl-bar1: world ", R, " ausserhalb 2..", HTCCL_BAR1_MAX_RANKS);
    TORCH_CHECK(r >= 0 && r < R, "htccl-bar1: rank ausserhalb");
    TORCH_CHECK(algo == 0 || algo == 1, "htccl-bar1: algo 0=netz 1=ring");
    TORCH_CHECK(inp.is_contiguous() && out.is_contiguous(),
                "htccl-bar1: nur zusammenhaengende Tensoren");
    TORCH_CHECK(inp.numel() == out.numel() && inp.scalar_type() == out.scalar_type(),
                "htccl-bar1: in und out passen nicht zusammen");
    TORCH_CHECK(inp.data_ptr() != out.data_ptr(),
                "htccl-bar1: in und out duerfen nicht dasselbe sein -- der Ring "
                "liest 'in' noch, waehrend er 'out' fortschreibt");
    TORCH_CHECK((int64_t)peer_nutz.size() == world &&
                (int64_t)peer_flag.size() == world,
                "htccl-bar1: Peer-Tabelle hat die falsche Laenge");

    const size_t nbytes = (size_t)inp.numel() * (size_t)inp.element_size();
    TORCH_CHECK(nbytes % 16 == 0,
                "htccl-bar1: Nutzlast ", nbytes, " ist kein Vielfaches von 16 "
                "Byte -- die Zugriffsbreite ist 128 Bit");
    TORCH_CHECK(((uintptr_t)inp.data_ptr() % 16) == 0 &&
                ((uintptr_t)out.data_ptr() % 16) == 0,
                "htccl-bar1: Puffer nicht auf 16 Byte ausgerichtet");
    const int n4 = (int)(nbytes / 16);
    TORCH_CHECK(n4 >= R, "htccl-bar1: weniger als ein 128-Bit-Paket je Rang");

    // Nahtstellenpruefung: der GROESSTE Chunk muss in einen Schlitz passen.
    // Absichtlich hier und nicht nur in Python -- gerechnet wird mit
    // chunkGrenzen selbst, nicht mit einer zweiten Fassung der Formel.
    {
        int maxLen = 0;
        for (int j = 0; j < R; ++j) {
            int off, len;
            chunkGrenzen(n4, j, R, &off, &len);
            if (len > maxLen) maxLen = len;
        }
        TORCH_CHECK((int64_t)maxLen * 16 <= chunk_max,
                    "htccl-bar1: groesster Chunk ", (int64_t)maxLen * 16,
                    " Byte passt nicht in den Schlitz von ", chunk_max,
                    " Byte. Der Aufrufer haette handles() fragen muessen.");
    }

    Bar1Args A;
    std::memset(&A, 0, sizeof(A));
    A.in           = (const uint4 *)inp.data_ptr();
    A.out          = (uint4 *)out.data_ptr();
    A.rundeDev     = (u64 *)runde_dev.data_ptr();
    A.ctlStatus    = (unsigned int *)ctl_dev.data_ptr();
    A.abbruchDev   = ((unsigned int *)ctl_dev.data_ptr()) + 1;
    A.n4           = n4;
    A.R            = R;
    A.rang         = r;
    A.deckelZyklen = (u64)deckel_zyklen;

    const int schritte_netz = 2;
    const int schritte_ring = 2 * (R - 1);
    // FSLOT(topo, schritt, sender) = FBASIS[topo] + (schritt*R + sender)*256,
    // FBASIS = { 0, 2*R*256 }. Unveraendert aus der Sonde, nur ohne die
    // beiden dort zusaetzlich vermessenen Topologien.
    const size_t fbasis_netz = 0;
    const size_t fbasis_ring = (size_t)schritte_netz * (size_t)R * 256u;
#define FSLOT(BASIS, SCHRITT, SENDER) \
    ((BASIS) + (size_t)((SCHRITT) * R + (SENDER)) * 256u)

    if (algo == 0) {
        for (int z = 0; z < R; ++z) {
            if (z == r) continue;
            // Meine Position in der aufsteigenden Peer-Liste des Empfaengers z.
            // NICHT (r < z) -- siehe die Herleitung in der Sonde.
            const size_t p = (size_t)(r - (r > z ? 1 : 0));
            char *zb = (char *)(uintptr_t)peer_nutz[z];
            A.nzSendRS[z] = (uint4 *)(zb + off_netz +
                                      (size_t)(0 * (R - 1) + (int)p) * chunk_max);
            A.nzSendAG[z] = (uint4 *)(zb + off_netz +
                                      (size_t)(1 * (R - 1) + (int)p) * chunk_max);
        }
        {
            char *mb = (char *)(uintptr_t)eigen_nutz;
            for (int s = 0; s < R; ++s) {
                if (s == r) continue;
                const size_t p = (size_t)(s - (s > r ? 1 : 0));
                A.nzRecvRS[s] = (const uint4 *)(mb + off_netz +
                                  (size_t)(0 * (R - 1) + (int)p) * chunk_max);
                A.nzRecvAG[s] = (const uint4 *)(mb + off_netz +
                                  (size_t)(1 * (R - 1) + (int)p) * chunk_max);
            }
        }
        for (int ph = 0; ph < 2; ++ph) {
            for (int s = 0; s < R; ++s) {
                if (s == r) continue;
                A.nzFlagAn[ph][s] = (u64 *)((char *)(uintptr_t)peer_flag[s] +
                                            FSLOT(fbasis_netz, ph, r));
                A.nzFlagVon[ph][s] = (const u64 *)((char *)(uintptr_t)eigen_flag +
                                                   FSLOT(fbasis_netz, ph, s));
            }
        }
    } else {
        const int nach = (r + 1) % R;
        const int vor  = (r + R - 1) % R;
        char *zb = (char *)(uintptr_t)peer_nutz[nach];
        char *mb = (char *)(uintptr_t)eigen_nutz;
        for (int s = 0; s < schritte_ring; ++s) {
            A.rgSend[s]    = (uint4 *)(zb + off_ring + (size_t)s * chunk_max);
            A.rgRecv[s]    = (const uint4 *)(mb + off_ring + (size_t)s * chunk_max);
            A.rgFlagAn[s]  = (u64 *)((char *)(uintptr_t)peer_flag[nach] +
                                     FSLOT(fbasis_ring, s, r));
            A.rgFlagVon[s] = (const u64 *)((char *)(uintptr_t)eigen_flag +
                                           FSLOT(fbasis_ring, s, vor));
        }
    }
#undef FSLOT

    auto strom = at::cuda::getCurrentCUDAStream().stream();
    const int kernI = (int)kern, laI = (int)ladeform, flI = (int)fluss;
    switch (inp.scalar_type()) {
        case at::kFloat:
            starte<float>((int)algo, kernI, laI, flI, A, (int)threads, strom);
            break;
        case at::kHalf:
            starte<__half>((int)algo, kernI, laI, flI, A, (int)threads, strom);
            break;
        case at::kBFloat16:
            starte<__nv_bfloat16>((int)algo, kernI, laI, flI, A, (int)threads,
                                  strom);
            break;
        default:
            TORCH_CHECK(false, "htccl-bar1: Datentyp ", inp.scalar_type(),
                        " wird nicht unterstuetzt (float32/float16/bfloat16)");
    }
}
"""

_CPP_SRC = """
void bar1_all_reduce(at::Tensor inp, at::Tensor out,
                     int64_t rank, int64_t world, int64_t algo,
                     std::vector<int64_t> peer_nutz,
                     std::vector<int64_t> peer_flag,
                     int64_t eigen_nutz, int64_t eigen_flag,
                     int64_t chunk_max, int64_t off_netz, int64_t off_ring,
                     at::Tensor runde_dev, at::Tensor ctl_dev,
                     int64_t deckel_zyklen, int64_t threads, int64_t kern,
                     int64_t ladeform, int64_t fluss);
"""


# ===========================================================================
# dma-buf-Export ueber die RM-Ioctls
# ===========================================================================

_DMABUF_SRC = r"""
#include <torch/extension.h>

#include <cerrno>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

#include <fcntl.h>
#include <sys/ioctl.h>
#include <unistd.h>

// UAPI der quelloffenen NVIDIA-Kernelmodule. Die Include-Pfade werden beim
// Uebersetzen hereingereicht; fehlen sie, wird diese Erweiterung gar nicht
// erst gebaut und der Transport meldet sich ab.
#include <nvtypes.h>
#include <nvos.h>
#include <nv-ioctl.h>
#include <nv_escape.h>
#include <class/cl0000.h>
#include <class/cl0080.h>
#include <ctrl/ctrl0000/ctrl0000gpu.h>
#include <ctrl/ctrl0000/ctrl0000unix.h>

static int nvIoctl(int fd, int nr, void *p, size_t size)
{
    return ioctl(fd, _IOC(_IOC_READ | _IOC_WRITE, NV_IOCTL_MAGIC, nr,
                          (unsigned)size), p);
}

// Portiert aus sonden/dmabuf_p2p_probe.cpp::nvExportToDmabuf().
//
// Rueckgabe: { dmabuf_fd, ctl_fd, dev_fd }. Die beiden letzten MUESSEN offen
// bleiben, solange der dma-buf benutzt wird: an ctl_fd haengt der RM-Client,
// der das importierte Speicherobjekt besitzt. Wird er geschlossen, gibt RM
// das Objekt frei und der dma-buf zeigt ins Leere. Deshalb gibt diese
// Funktion sie heraus, statt sie stillschweigend lecken zu lassen -- der
// Aufrufer haelt sie und schliesst sie beim Abbau.
//
// objfd = fd aus cuMemExportToShareableHandle(POSIX_FILE_DESCRIPTOR)
// pci_bus = CU_DEVICE_ATTRIBUTE_PCI_BUS_ID der EXPORTIERENDEN Karte
std::vector<int64_t> bar1_export_dmabuf(int64_t objfd, int64_t pci_bus,
                                        int64_t size)
{
    const NvHandle hDevice = 0xbee00001;
    const NvHandle hMemory = 0xbee00010;
    NvHandle hClient = 0;
    int ctlFd = -1, devFd = -1;
    char fehler[256] = {0};

#define NVX_FAIL(...)                                                          \
    do {                                                                       \
        std::snprintf(fehler, sizeof(fehler), __VA_ARGS__);                    \
        if (devFd >= 0) close(devFd);                                          \
        if (ctlFd >= 0) close(ctlFd);                                          \
        TORCH_CHECK(false, "htccl-bar1 dma-buf-Export: ", fehler);             \
    } while (0)

    ctlFd = open("/dev/nvidiactl", O_RDWR);
    if (ctlFd < 0) NVX_FAIL("open(/dev/nvidiactl): %s", std::strerror(errno));

    {   // Versions-Handshake
        nv_ioctl_rm_api_version_t v;
        char buf[256] = {0}, ver[64] = {0};
        std::memset(&v, 0, sizeof(v));
        v.cmd = NV_RM_API_VERSION_CMD_RELAXED;
        FILE *f = std::fopen("/proc/driver/nvidia/version", "r");
        if (f) { if (!std::fgets(buf, sizeof(buf), f)) buf[0] = 0; std::fclose(f); }
        char *p = std::strstr(buf, "for x86_64");
        if (p && std::sscanf(p, "for x86_64 %63s", ver) == 1)
            std::strncpy(v.versionString, ver, sizeof(v.versionString) - 1);
        if (nvIoctl(ctlFd, NV_ESC_CHECK_VERSION_STR, &v, sizeof(v)) < 0)
            NVX_FAIL("NV_ESC_CHECK_VERSION_STR: %s", std::strerror(errno));
    }

    NvU32 gpuId = 0, minor = 0;
    {
        nv_ioctl_card_info_t ci[32];
        std::memset(ci, 0, sizeof(ci));
        if (nvIoctl(ctlFd, NV_ESC_CARD_INFO, ci, sizeof(ci)) < 0)
            NVX_FAIL("NV_ESC_CARD_INFO: %s", std::strerror(errno));
        bool found = false;
        for (int i = 0; i < 32; ++i) {
            if (!ci[i].valid) continue;
            if ((int64_t)ci[i].pci_info.bus == pci_bus) {
                gpuId = ci[i].gpu_id;
                minor = ci[i].minor_number;
                found = true;
            }
        }
        if (!found)
            NVX_FAIL("PCI-Bus 0x%02x nicht in NV_ESC_CARD_INFO",
                     (unsigned)pci_bus);
    }

    {   // eigener RM-Client
        NVOS21_PARAMETERS a;
        std::memset(&a, 0, sizeof(a));
        a.hClass = NV01_ROOT;
        if (nvIoctl(ctlFd, NV_ESC_RM_ALLOC, &a, sizeof(a)) < 0 || a.status != 0)
            NVX_FAIL("RM_ALLOC(root) status=0x%x", (unsigned)a.status);
        hClient = a.hObjectNew;
    }

    NvU32 devInst = 0;
    {
        NV0000_CTRL_GPU_GET_ID_INFO_V2_PARAMS p;
        NVOS54_PARAMETERS c;
        std::memset(&p, 0, sizeof(p)); std::memset(&c, 0, sizeof(c));
        p.gpuId   = gpuId;
        c.hClient = hClient; c.hObject = hClient;
        c.cmd     = NV0000_CTRL_CMD_GPU_GET_ID_INFO_V2;
        c.params  = (NvP64)(uintptr_t)&p; c.paramsSize = sizeof(p);
        if (nvIoctl(ctlFd, NV_ESC_RM_CONTROL, &c, sizeof(c)) < 0 || c.status != 0)
            NVX_FAIL("GPU_GET_ID_INFO_V2 status=0x%x", (unsigned)c.status);
        devInst = p.deviceInstance;
    }

    {
        NV0080_ALLOC_PARAMETERS dp;
        NVOS21_PARAMETERS a;
        std::memset(&dp, 0, sizeof(dp)); std::memset(&a, 0, sizeof(a));
        dp.deviceId = devInst; dp.hClientShare = hClient;
        a.hRoot = hClient; a.hObjectParent = hClient; a.hObjectNew = hDevice;
        a.hClass = NV01_DEVICE_0;
        a.pAllocParms = (NvP64)(uintptr_t)&dp;
        if (nvIoctl(ctlFd, NV_ESC_RM_ALLOC, &a, sizeof(a)) < 0 || a.status != 0)
            NVX_FAIL("RM_ALLOC(device) status=0x%x", (unsigned)a.status);
    }

    {   // Objekt-fd in den eigenen Client importieren
        NV0000_CTRL_OS_UNIX_IMPORT_OBJECT_FROM_FD_PARAMS p;
        NVOS54_PARAMETERS c;
        std::memset(&p, 0, sizeof(p)); std::memset(&c, 0, sizeof(c));
        p.fd = (NvS32)objfd;
        p.object.type = NV0000_CTRL_OS_UNIX_EXPORT_OBJECT_TYPE_RM;
        p.object.data.rmObject.hDevice = hDevice;
        p.object.data.rmObject.hParent = hDevice;
        p.object.data.rmObject.hObject = hMemory;
        c.hClient = hClient; c.hObject = hClient;
        c.cmd     = NV0000_CTRL_CMD_OS_UNIX_IMPORT_OBJECT_FROM_FD;
        c.params  = (NvP64)(uintptr_t)&p; c.paramsSize = sizeof(p);
        if (nvIoctl(ctlFd, NV_ESC_RM_CONTROL, &c, sizeof(c)) < 0 || c.status != 0)
            NVX_FAIL("IMPORT_OBJECT_FROM_FD status=0x%x", (unsigned)c.status);
    }

    int dmabufFd = -1;
    {
        char devpath[64];
        std::snprintf(devpath, sizeof(devpath), "/dev/nvidia%u", (unsigned)minor);
        devFd = open(devpath, O_RDWR);
        if (devFd < 0) NVX_FAIL("open(%s): %s", devpath, std::strerror(errno));

        nv_ioctl_export_to_dma_buf_fd_t p;
        std::memset(&p, 0, sizeof(p));
        p.fd           = -1;               // -1 = neuen dma-buf erzeugen
        p.hClient      = hClient;
        p.totalObjects = 1;
        p.numObjects   = 1;
        p.index        = 0;
        p.totalSize    = (NvU64)size;
        p.mappingType  = NV_DMABUF_EXPORT_MAPPING_TYPE_DEFAULT;
        p.bAllowMmap   = NV_FALSE;
        p.handles[0]   = hMemory;
        p.offsets[0]   = 0;
        p.sizes[0]     = (NvU64)size;

        if (nvIoctl(devFd, NV_ESC_EXPORT_TO_DMABUF_FD, &p, sizeof(p)) < 0)
            NVX_FAIL("NV_ESC_EXPORT_TO_DMABUF_FD: %s", std::strerror(errno));
        if (p.status != 0)
            NVX_FAIL("NV_ESC_EXPORT_TO_DMABUF_FD status=0x%08x",
                     (unsigned)p.status);
        dmabufFd = p.fd;
    }
#undef NVX_FAIL

    std::vector<int64_t> aus;
    aus.push_back((int64_t)dmabufFd);
    aus.push_back((int64_t)ctlFd);
    aus.push_back((int64_t)devFd);
    return aus;
}
"""

# Bewusst KEIN getrennter Deklarationsblock wie bei der CUDA-Erweiterung:
# `load_inline` haengt cpp_sources in der uebergebenen Reihenfolge aneinander,
# und eine Deklaration mit std::vector VOR den #includes uebersetzt nicht.
# Die Definition allein genuegt, pybind braucht nichts weiter.


# ===========================================================================
# Uebersetzen
# ===========================================================================


def nv_include_pfade() -> Optional[list]:
    """Die drei Include-Verzeichnisse des Treiberbaums, oder ``None``.

    ``None`` heisst: der Baum ist hier nicht da. Kein Raten und keine
    mitgelieferte Kopie der UAPI-Strukturen -- deren Feldversaetze sind
    versionsgebunden, und eine falsch geratene Struktur schickt Muell in
    einen Kernel-Ioctl.
    """
    wurzel = os.environ.get("SGLANG_HTCCL_BAR1_NV_QUELLE", NV_QUELLE_VORGABE)
    p = pathlib.Path(wurzel)
    pfade = [p / t for t in NV_INCLUDES]
    fehlend = [str(x) for x in pfade if not x.is_dir()]
    if fehlend:
        return None
    # Ein Kopfdatei-Stichprobentest: das Verzeichnis kann existieren und
    # trotzdem der falsche Baum sein.
    if not (pfade[1] / "nvos.h").is_file():
        return None
    return [str(x) for x in pfade]


def lade_kollektiv_ext(cpu_group):
    """Die CUDA-Erweiterung mit den Kernen ``netz`` und ``ring``.

    Cache-Hygiene, Bogenaufloesung und Baumarke kommen aus ``htccl_device``:
    dieselbe Mechanik, ein eigener Name. Der Name traegt Hersteller und
    Boegen, weil torch sein Bauverzeichnis nur nach dem Namen schluesselt --
    ohne das bekaeme ein Rang eine .so gereicht, die seine Karte nicht
    ausfuehren kann.
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
            f"HTCCL-BAR1: der Direktpfad ist NVIDIA-eigen (BAR1-Apertur, "
            f"dma-buf-Export ueber die RM-Ioctls, PTX-Einschuebe wie "
            f"ld.mmio.relaxed.sys). Dieser Rang meldet {vendor!r}. Es gibt "
            f"keine hipify-Uebersetzung dieser Zeilen, und eine vorzutaeuschen "
            f"waere schlimmer als die Absage."
        )
    by_vendor = _resolve_build_arches(cpu_group)
    arches = by_vendor.get(vendor, [])
    flags = _build_flags(vendor, arches)
    # KEIN -rdc=true und kein -lcudadevrt: die Sonde uebersetzt dieselben
    # cooperative-groups-Aufrufe ohne beides (Bauzeile im Kopf von
    # bar1_kollektiv.cu). Getrennte Uebersetzung braeuchte hier ein
    # Geraete-Linken, das load_inline nicht macht -- und sie wird nicht
    # gebraucht.
    name = "htccl_bar1_ext_" + vendor
    if arches:
        name += "_" + "_".join(a.replace(".", "") for a in arches)
    t0 = time.time()
    with _ext_cache_guarded(name) as build_dir:
        _ext = load_inline(
            name=name,
            cpp_sources=_CPP_SRC,
            cuda_sources=_CUDA_SRC,
            functions=["bar1_all_reduce"],
            extra_cuda_cflags=flags or None,
            verbose=False,
            build_directory=str(build_dir) if build_dir is not None else None,
        )
    logger.info(
        "HTCCL-BAR1: Kollektiv-Erweiterung %r fuer Boegen %s in %.1f s gebaut.",
        name, ",".join(arches) or "<torch-Vorgabe>", time.time() - t0,
    )
    return _ext


def lade_dmabuf_ext():
    """Die C++-Erweiterung fuer den dma-buf-Export, oder ``None``.

    ``None`` mit protokolliertem Grund, wenn der Treiberbaum fehlt. Der
    Aufrufer versucht vorher ``cuMemGetHandleForAddressRange``; nur wenn der
    scheitert -- auf GeForce meldet der Treiber
    ``CU_DEVICE_ATTRIBUTE_DMA_BUF_SUPPORTED = 0`` -- wird dieser Weg
    gebraucht.
    """
    global _dmabuf_ext, _dmabuf_grund
    if _dmabuf_ext is not None:
        return _dmabuf_ext
    if _dmabuf_grund:
        return None

    pfade = nv_include_pfade()
    if pfade is None:
        _dmabuf_grund = (
            f"Die Kopfdateien der quelloffenen NVIDIA-Kernelmodule liegen "
            f"nicht unter "
            f"{os.environ.get('SGLANG_HTCCL_BAR1_NV_QUELLE', NV_QUELLE_VORGABE)!r} "
            f"(erwartet: {', '.join(NV_INCLUDES)}). Ohne sie ist "
            f"NV_ESC_EXPORT_TO_DMABUF_FD nicht aufrufbar. Pfad ueber "
            f"SGLANG_HTCCL_BAR1_NV_QUELLE setzen."
        )
        logger.info("HTCCL-BAR1: %s", _dmabuf_grund)
        return None

    from torch.utils.cpp_extension import load_inline

    from sglang.srt.distributed.device_communicators.htccl_device import (
        _ext_cache_guarded,
    )

    name = "htccl_bar1_dmabuf_ext"
    t0 = time.time()
    try:
        with _ext_cache_guarded(name) as build_dir:
            _dmabuf_ext = load_inline(
                name=name,
                cpp_sources=_DMABUF_SRC,
                functions=["bar1_export_dmabuf"],
                extra_include_paths=pfade,
                verbose=False,
                build_directory=str(build_dir) if build_dir is not None else None,
            )
    except Exception as e:                     # Uebersetzungsfehler ist ein Grund
        _dmabuf_grund = f"Uebersetzung fehlgeschlagen: {e}"
        logger.info("HTCCL-BAR1: dma-buf-Erweiterung nicht gebaut -- %s", e)
        return None
    logger.info(
        "HTCCL-BAR1: dma-buf-Erweiterung in %.1f s gebaut (Treiberbaum %s).",
        time.time() - t0, pfade[0],
    )
    return _dmabuf_ext


def dmabuf_grund() -> str:
    return _dmabuf_grund
