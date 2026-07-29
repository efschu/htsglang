# SPDX-License-Identifier: Apache-2.0
"""HTCCL-Unterpfad: BAR1-Direkttransport.

Die Quellkarte schreibt per DMA in die BAR1-Apertur der Zielkarte. Kein
Host-Speicher, keine NIC, kein NCCL. Der Weg ist byte-belegt (siehe
``BYTE_BELEG_DMABUF.md``, vier Richtungen, ``bad_bytes = 0``), aber es gibt
**keine einzige Zeitmessung** davon -- alle Raten des Repos stammen vom
statischen Fenster. Dieses Modul baut den Weg auf; ob er sich lohnt,
entscheidet die Messung, die er ermoeglicht.

Ablauf des Aufbaus, genau einmal beim Start
-------------------------------------------
1. **Empfangspuffer per CUDA-VMM** (``cuMemCreate`` + ``cuMemAddressReserve``
   + ``cuMemMap`` + ``cuMemSetAccess``). Nicht ``cudaMalloc``: nur eine
   VMM-Allokation laesst sich als dma-buf exportieren.
2. **dma-buf-fd erzeugen** (``cuMemGetHandleForAddressRange``,
   ``CU_MEM_RANGE_HANDLE_TYPE_DMA_BUF_FD``). Der fd bleibt offen.
3. **fd an alle Peers reichen** ueber ``SCM_RIGHTS`` auf einem
   AF_UNIX-Socket. Ein fd ist prozesslokal; ueber gloo geht er nicht.
4. **Jeder Peer haengt sich an** -- ``/dev/dmabuf_holder``, ``dma_buf_attach``
   + ``dma_buf_map_attachment``. **Erst dieser Schritt programmiert die
   BAR1-Seiten der Zielkarte** (``nv-dmabuf.c:1066``). Ein offener fd ohne
   Importeur genuegt nachweislich nicht: der Musterscan fand ueber 65 536
   Sondierungen nichts, nach dem Anhaengen sass der Treffer sofort.
   Angehaengt wird als die **Quellkarte** -- sie ist das Geraet, das spaeter
   schreibt, damit stimmen Topologiepruefung und IOMMU-Domaene.
5. **BAR1-Versatz aus der sg-Tabelle**, die der Halter zurueckgibt -- kein
   Musterscan. Der Versatz ist die Differenz zum BAR1-Anfang aus sysfs.
6. **Fenster mappen und registrieren**: ``resource1_wc`` der Zielkarte, nur
   den benoetigten Ausschnitt (ein mmap ueber ein 32-GiB-Fenster scheitert
   mit ``EINVAL``), dann ``cudaHostRegister(..., cudaHostRegisterIoMemory)``
   und ``cudaHostGetDevicePointer`` auf der **Quellkarte**.
7. **Byte-Beleg je gerichtetem Paar.** Muster hinein, auf der Zielkarte
   ueber ihren **eigenen** VMM-Zeiger zurueckgelesen, jedes Byte
   verglichen. Nicht ueber die Apertur zurueck -- sonst verdeckt ein
   defekter Pfad seinen eigenen Fehler.

Danach steht die Peer-Zeiger-Tabelle fest. **Im heissen Pfad wird nichts
gemappt und nichts registriert** -- das ist der teure Teil, und um 7 us geht
es hier.

Nicht verhandelbare Auflagen
----------------------------
* **Sauberer Rueckfall.** Fehlt der Patch, das Haltermodul oder die
  Peer-Faehigkeit, meldet sich der Transport ab: ``handles(...)`` gibt
  ``False``. Er scheitert nicht und er veraendert ohne ausdrueckliche Wahl
  nichts. Ein Transport, der einen Patch VORAUSSETZT, waere auf den meisten
  Maschinen unbrauchbar.
* **Kein stiller Platzhalter.** Wo eine Treiberfunktion fehlt, die es noch
  nicht gibt, steht ein ``NotImplementedError`` mit Begruendung -- keine
  Attrappe, die scheinbar funktioniert.
* **Fenstergrenze.** Erreichbar ist nur, was gleichzeitig abgebildet ist.
  Der Bedarf wird explizit gerechnet (``fensterbedarf``) und beim Start
  gegen das geprueft, was sich tatsaechlich exportieren laesst -- nicht
  gegen die Bruttogroesse aus sysfs.
* **Nur phasengetrennt.** Der gelockerte Treiber-Guard existiert wegen
  eines dokumentierten Vollduplex-Deadlocks ueber BAR1 (Bug 1571948). Bis
  Gegenverkehr ueber die volle Kollektivdauer geprueft ist, darf die
  Zerlegung nicht gleichzeitig in beide Richtungen schreiben.

Die Kollektive
--------------
``all_reduce`` laeuft ueber zwei Kernel, die aus der Sonde
``/spinning/nvidia-open-595/bar1_kollektiv.cu`` portiert sind (Quelltext
und Uebersetzung: ``htccl_bar1_ext.py``):

``netz``
    Reduce-Scatter + Allgather ueber ALLE Paare, **zwei** Sperren.
``ring``
    Ring-Reduce-Scatter + Ring-Allgather, **2(R-1)** Sperren.

Beide sind in der Sonde byte-belegt und gegen NCCL gemessen (drei Raenge,
p50, volle Operationsdauer, uncached):

====== ========== ======== ========= =======
Groesse bester Arm     us   NCCL us  Faktor
====== ========== ======== ========= =======
 20 KiB hub          28,22     41,75   1,48x
 80 KiB netz         50,81     73,58   1,45x
  1 MiB ring        328,60    372,79   1,13x
  4 MiB netz       1301,05   1356,69   1,04x
 16 MiB ring       4077,43   5172,83   1,27x
====== ========== ======== ========= =======

**Der 20-KiB-Sieger ``hub`` ist hier NICHT portiert.** Er ist keine
Zerlegung, sondern eine Rolle (ein Rang sammelt alles ein und verteilt das
Ergebnis zurueck), er braucht R volle Puffer je Nabe statt Chunkschlitze,
und der Planer in ``htccl_matrix.py`` kennt dafuer den eigenen Algorithmus
``stern``. Bei 20 KiB liegt ``netz`` in derselben Messung bei 31,67 us --
der Verlust gegen ``hub`` ist klein, der Aufwand fuer eine zweite
Speichergeometrie (R volle Puffer statt Chunkschlitze) nicht. Wer
``stern`` will, faellt heute auf den Nicht-BAR1-Pfad zurueck; ``handles``
sagt das mit ``False``, es wird nicht still ``netz`` untergeschoben.

Welcher der beiden Kernel bei welcher Groesse laeuft, entscheidet **nicht**
dieses Modul, sondern der Plan aus ``htccl_matrix.py``. Das ist die
Schlussfolgerung der Messung selbst: zwischen 80 KiB und 16 MiB gibt es
laut ``MESSUNG_ALLES_IM_SELBEN_LAUF.md`` "keine saubere Schwelle" (netz
330,30 gegen ring 326,57 us bei 1 MiB), und der Ringvorteil bei 1 MiB
liegt innerhalb dessen, was ohne Wiederholungslaeufe nicht von Rauschen
zu unterscheiden ist. Ohne Plan gilt die Notschwelle
``SGLANG_HTCCL_BAR1_RING_AB`` (Vorgabe 1 MiB) -- eine Vorgabe, keine
Messaussage.

**Datentypen.** Gemessen ist ausschliesslich ``float32``. Die Kernel
rechnen zusaetzlich ``float16`` und ``bfloat16``, weil die Zugriffsbreite
(128 Bit) und damit der vermessene Teil des Pfades gleich bleibt und nur
die Deutung der 16 Byte sich aendert. Eine Zeitmessung dafuer gibt es
nicht.

``all_to_all``
--------------
``all_to_all_single`` laeuft ueber einen **dritten** Kern (``a2a``), der in
der Sonde keine Entsprechung hat und deshalb nicht portiert, sondern neu
geschrieben ist. Rang r schreibt seinen Block fuer Rang j direkt in dessen
Empfangsschlitz -- ein Schritt, alle Sendevorgaenge im selben flachen
Indexraum, dann **eine** Sperre. Kein Host-Umweg, kein Nachmappen.

Drei Eigenschaften, die ihn von ``all_reduce`` unterscheiden:

* **Keine Reduktion, also kein Datentyp.** Der Kern bewegt Bytes. fp8,
  bf16, int32, uint8 -- ein Pfad. Dass die sm_86-Karten keine
  fp8-Umwandlungsbefehle haben (die beginnen bei sm_89), ist hier
  gegenstandslos.
* **Ungleiche Teilgroessen sind der Normalfall.** Die Zahl der Token je
  Experte schwankt; ``sende_bytes``/``empfangs_bytes`` kommen je Rang
  herein. Passt ein Block nicht in einen Schlitz, meldet sich der
  Transport ueber ``traegt_a2a`` ab, statt zu scheitern.
* **Doppelte Schlitze statt zweiter Sperre.** ``2(R-1)`` Schlitze, die
  Rundennummer waehlt die Haelfte. Begruendung bei ``geometrie`` und beim
  Kern.

**Ungemessen.** Es gibt fuer diesen Kern bis heute **keine einzige
Zeitmessung** -- nur den Byte-Beleg (``byte_beleg_a2a``, gleichverteilt und
schief, jedes Byte, jedes gerichtete Paar). Die Tabelle oben gilt fuer
``all_reduce`` und ausschliesslich dafuer.

Was hier NICHT drin ist
-----------------------
``all_gather``, ``reduce_scatter`` und ``broadcast``. Sie waeren je eine
Haelfte der all_reduce-Kernel, aber keine davon ist gemessen -- und in
diesem Vorhaben zaehlt nur Gemessenes. ``handles`` gibt dafuer ``False``
und die ``htccl_*``-Methoden erklaeren, was fehlt.
"""

from __future__ import annotations

import ctypes
import fcntl
import logging
import mmap
import os
import struct
import tempfile
import time
from dataclasses import dataclass
from typing import Optional

logger = logging.getLogger(__name__)


class Bar1Unverfuegbar(RuntimeError):
    """Der BAR1-Pfad steht auf dieser Maschine nicht zur Verfuegung.

    Wird beim Aufbau geworfen und vom Aufrufer in ein ``handles() == False``
    uebersetzt. Traegt IMMER den Grund, weil "geht nicht" in diesem
    Vorhaben ohne Beleg nichts wert ist.
    """


# ===========================================================================
# CUDA-Bindungen (ctypes) -- nur, was der Aufbau braucht
# ===========================================================================

CU_MEM_ALLOCATION_TYPE_PINNED = 0x1
CU_MEM_LOCATION_TYPE_DEVICE = 0x1
CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR = 0x1
CU_MEM_ACCESS_FLAGS_PROT_READWRITE = 0x3
CU_MEM_RANGE_HANDLE_TYPE_DMA_BUF_FD = 0x1
# Erzwingt eine PCIe-Abbildung des dma-buf statt einer moeglichen
# Kurzschluss-Abbildung. Ohne das kann der Treiber eine Abbildung liefern,
# die fuer einen Peer nicht ueber PCIe erreichbar ist.
CU_MEM_RANGE_FLAG_DMA_BUF_MAPPING_TYPE_PCIE = 0x1

CUDA_HOST_REGISTER_IO_MEMORY = 0x04
CUDA_MEMCPY_DEFAULT = 4
CU_MEM_ALLOC_GRANULARITY_RECOMMENDED = 0x1
CU_DEVICE_ATTRIBUTE_PCI_BUS_ID = 33


class _CUmemLocation(ctypes.Structure):
    _fields_ = [("type", ctypes.c_int), ("id", ctypes.c_int)]


class _CUmemAllocFlags(ctypes.Structure):
    _fields_ = [
        ("compressionType", ctypes.c_ubyte),
        ("gpuDirectRDMACapable", ctypes.c_ubyte),
        ("usage", ctypes.c_ushort),
        ("reserved", ctypes.c_ubyte * 4),
    ]


class _CUmemAllocationProp(ctypes.Structure):
    _fields_ = [
        ("type", ctypes.c_int),
        ("requestedHandleTypes", ctypes.c_int),
        ("location", _CUmemLocation),
        ("win32HandleMetaData", ctypes.c_void_p),
        ("allocFlags", _CUmemAllocFlags),
    ]


class _CUmemAccessDesc(ctypes.Structure):
    _fields_ = [("location", _CUmemLocation), ("flags", ctypes.c_int)]


class _Cuda:
    """Schmale Huelle um libcuda/libcudart. Laedt faul, nie beim Import."""

    def __init__(self):
        try:
            self.drv = ctypes.CDLL("libcuda.so.1")
        except OSError as e:
            raise Bar1Unverfuegbar(f"libcuda.so.1 nicht ladbar: {e}") from e
        try:
            self.rt = ctypes.CDLL("libcudart.so")
        except OSError:
            try:
                self.rt = ctypes.CDLL("libcudart.so.12")
            except OSError as e:
                raise Bar1Unverfuegbar(f"libcudart nicht ladbar: {e}") from e

    def _d(self, name: str, *args) -> None:
        fn = getattr(self.drv, name, None)
        if fn is None:
            raise Bar1Unverfuegbar(
                f"{name} fehlt in libcuda -- der Treiber ist zu alt fuer den "
                f"VMM-/dma-buf-Weg."
            )
        rc = fn(*args)
        if rc != 0:
            text = ctypes.c_char_p()
            if hasattr(self.drv, "cuGetErrorString"):
                self.drv.cuGetErrorString(ctypes.c_int(rc), ctypes.byref(text))
            raise Bar1Unverfuegbar(
                f"{name} -> {rc} "
                f"({text.value.decode() if text.value else 'kein Text'})"
            )

    def _r(self, name: str, *args) -> None:
        fn = getattr(self.rt, name)
        rc = fn(*args)
        if rc != 0:
            raise Bar1Unverfuegbar(f"{name} -> cudaError {rc}")

    # -- VMM ---------------------------------------------------------------

    def _prop(self, ordinal: int) -> _CUmemAllocationProp:
        p = _CUmemAllocationProp()
        ctypes.memset(ctypes.byref(p), 0, ctypes.sizeof(p))
        p.type = CU_MEM_ALLOCATION_TYPE_PINNED
        p.requestedHandleTypes = CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR
        p.location.type = CU_MEM_LOCATION_TYPE_DEVICE
        p.location.id = ordinal
        return p

    def granularitaet(self, ordinal: int) -> int:
        gran = ctypes.c_size_t(0)
        p = self._prop(ordinal)
        self._d("cuMemGetAllocationGranularity", ctypes.byref(gran),
                ctypes.byref(p), ctypes.c_int(CU_MEM_ALLOC_GRANULARITY_RECOMMENDED))
        return int(gran.value) or (2 << 20)

    def vmm_alloc(self, ordinal: int, groesse: int) -> tuple[int, int, int]:
        """``(dptr, handle, groesse)`` -- exportfaehige Geraeteallokation."""
        gran = self.granularitaet(ordinal)
        groesse = ((groesse + gran - 1) // gran) * gran
        handle = ctypes.c_ulonglong(0)
        p = self._prop(ordinal)
        self._d("cuMemCreate", ctypes.byref(handle), ctypes.c_size_t(groesse),
                ctypes.byref(p), ctypes.c_ulonglong(0))
        dptr = ctypes.c_ulonglong(0)
        self._d("cuMemAddressReserve", ctypes.byref(dptr),
                ctypes.c_size_t(groesse), ctypes.c_size_t(gran),
                ctypes.c_ulonglong(0), ctypes.c_ulonglong(0))
        self._d("cuMemMap", ctypes.c_ulonglong(dptr.value),
                ctypes.c_size_t(groesse), ctypes.c_size_t(0), handle,
                ctypes.c_ulonglong(0))
        desc = _CUmemAccessDesc()
        desc.location.type = CU_MEM_LOCATION_TYPE_DEVICE
        desc.location.id = ordinal
        desc.flags = CU_MEM_ACCESS_FLAGS_PROT_READWRITE
        self._d("cuMemSetAccess", ctypes.c_ulonglong(dptr.value),
                ctypes.c_size_t(groesse), ctypes.byref(desc), ctypes.c_size_t(1))
        return int(dptr.value), int(handle.value), groesse

    def vmm_frei(self, dptr: int, handle: int, groesse: int) -> None:
        for name, args in (
            ("cuMemUnmap", (ctypes.c_ulonglong(dptr), ctypes.c_size_t(groesse))),
            ("cuMemRelease", (ctypes.c_ulonglong(handle),)),
            ("cuMemAddressFree", (ctypes.c_ulonglong(dptr),
                                  ctypes.c_size_t(groesse))),
        ):
            try:
                getattr(self.drv, name)(*args)
            except Exception:      # Abbau darf den echten Fehler nie verdecken
                pass

    def pci_bus(self, ordinal: int) -> int:
        """``CU_DEVICE_ATTRIBUTE_PCI_BUS_ID``. Der RM-Ioctl-Weg sucht die
        Karte ueber die Busnummer, nicht ueber das Ordinal."""
        dev = ctypes.c_int(0)
        self._d("cuDeviceGet", ctypes.byref(dev), ctypes.c_int(ordinal))
        wert = ctypes.c_int(0)
        self._d("cuDeviceGetAttribute", ctypes.byref(wert),
                ctypes.c_int(CU_DEVICE_ATTRIBUTE_PCI_BUS_ID), dev)
        return int(wert.value)

    def export_shareable(self, handle: int) -> int:
        """``cuMemExportToShareableHandle`` -- der Objekt-fd fuer den Ioctl-Weg."""
        fd = ctypes.c_int(-1)
        self._d("cuMemExportToShareableHandle", ctypes.byref(fd),
                ctypes.c_ulonglong(handle),
                ctypes.c_int(CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR),
                ctypes.c_ulonglong(0))
        return int(fd.value)

    def memset_d8(self, dptr: int, wert: int, n: int) -> None:
        # ``cuMemsetD8_v2``, NICHT ``cuMemsetD8``. In cuda.h ist der kurze
        # Name ein Makro auf die _v2-Form; ueber dlsym/ctypes bekommt man
        # dagegen den alten ABI-Einsprung mit 32-Bit-CUdeviceptr, und der
        # antwortet auf einem heutigen Treiber mit 201 (invalid device
        # context) -- auch wenn ein Kontext aktuell ist (nachgemessen:
        # cuCtxGetCurrent liefert einen gueltigen Kontext, cuMemsetD8 -> 201,
        # cuMemsetD8_v2 -> 0). Das gilt fuer jede _v2-Funktion der Treiber-API.
        self._d("cuMemsetD8_v2", ctypes.c_ulonglong(dptr), ctypes.c_ubyte(wert),
                ctypes.c_size_t(n))

    def dmabuf_fd(self, dptr: int, handle: int, groesse: int,
                  ordinal: int) -> tuple[int, list[int], str]:
        """``(dmabuf_fd, zu_haltende_fds, weg)``.

        Erst der bequeme Weg ``cuMemGetHandleForAddressRange``. Auf GeForce
        meldet der Treiber ``CU_DEVICE_ATTRIBUTE_DMA_BUF_SUPPORTED = 0`` und
        liefert ``CUDA_ERROR_INVALID_VALUE`` (=1), obwohl das Kernelmodul den
        Export kann (``nv->dma_buf_supported = 1``, osinit.c:671). Dann der
        Ioctl-Weg ueber die native Erweiterung -- portiert aus
        ``sonden/dmabuf_p2p_probe.cpp::nvExportToDmabuf()``.

        ``zu_haltende_fds`` sind ``/dev/nvidiactl`` und ``/dev/nvidia<N>``
        des Ioctl-Weges. An ersterem haengt der RM-Client, der das
        importierte Speicherobjekt besitzt; wird er geschlossen, gibt RM das
        Objekt frei und der dma-buf zeigt ins Leere. Sie werden deshalb
        herausgegeben und beim Abbau geschlossen, statt zu lecken.
        """
        from sglang.srt.distributed.device_communicators import htccl_bar1_ext

        fd = ctypes.c_int(-1)
        fn = getattr(self.drv, "cuMemGetHandleForAddressRange", None)
        rc = -1
        if fn is not None:
            rc = fn(ctypes.byref(fd), ctypes.c_ulonglong(dptr),
                    ctypes.c_size_t(groesse),
                    ctypes.c_int(CU_MEM_RANGE_HANDLE_TYPE_DMA_BUF_FD),
                    ctypes.c_ulonglong(
                        CU_MEM_RANGE_FLAG_DMA_BUF_MAPPING_TYPE_PCIE))
            if rc == 0:
                return int(fd.value), [], "cuMemGetHandleForAddressRange"

        ext = htccl_bar1_ext.lade_dmabuf_ext()
        if ext is None:
            raise Bar1Unverfuegbar(
                f"dma-buf-Export nicht moeglich. "
                f"cuMemGetHandleForAddressRange -> "
                f"{'fehlt in libcuda' if fn is None else rc}, und der "
                f"Ersatzweg ueber NV0000_CTRL_CMD_OS_UNIX_IMPORT_OBJECT_FROM_FD "
                f"+ NV_ESC_EXPORT_TO_DMABUF_FD steht auch nicht bereit: "
                f"{htccl_bar1_ext.dmabuf_grund()}"
            )
        objfd = self.export_shareable(handle)
        try:
            aus = ext.bar1_export_dmabuf(int(objfd), int(self.pci_bus(ordinal)),
                                         int(groesse))
        except Exception as e:
            raise Bar1Unverfuegbar(
                f"NV_ESC_EXPORT_TO_DMABUF_FD fehlgeschlagen: {e}"
            ) from e
        finally:
            # Der Objekt-fd ist nach dem Import nicht mehr noetig -- RM haelt
            # das Objekt jetzt ueber den eigenen Client.
            try:
                os.close(objfd)
            except OSError:
                pass
        return int(aus[0]), [int(aus[1]), int(aus[2])], "NV_ESC_EXPORT_TO_DMABUF_FD"

    # -- Runtime -----------------------------------------------------------

    def register_io(self, adresse: int, laenge: int) -> None:
        self._r("cudaHostRegister", ctypes.c_void_p(adresse),
                ctypes.c_size_t(laenge),
                ctypes.c_uint(CUDA_HOST_REGISTER_IO_MEMORY))

    def unregister(self, adresse: int) -> None:
        try:
            self.rt.cudaHostUnregister(ctypes.c_void_p(adresse))
        except Exception:
            pass

    def dev_ptr(self, host_adresse: int) -> int:
        p = ctypes.c_void_p(0)
        self._r("cudaHostGetDevicePointer", ctypes.byref(p),
                ctypes.c_void_p(host_adresse), ctypes.c_uint(0))
        return int(p.value or 0)

    def memcpy_async(self, ziel: int, quelle: int, n: int, stream: int) -> None:
        self._r("cudaMemcpyAsync", ctypes.c_void_p(ziel), ctypes.c_void_p(quelle),
                ctypes.c_size_t(n), ctypes.c_int(CUDA_MEMCPY_DEFAULT),
                ctypes.c_void_p(stream))

    def memcpy(self, ziel: int, quelle: int, n: int) -> None:
        self._r("cudaMemcpy", ctypes.c_void_p(ziel), ctypes.c_void_p(quelle),
                ctypes.c_size_t(n), ctypes.c_int(CUDA_MEMCPY_DEFAULT))


# ===========================================================================
# /dev/dmabuf_holder
# ===========================================================================

HALTER_PFAD = os.environ.get("SGLANG_HTCCL_BAR1_HALTER", "/dev/dmabuf_holder")

_HOLD_FMT = "=iIIBBBBIIQIIQQ"      # struct dmabuf_holder_hold
_HOLD_SIZE = struct.calcsize(_HOLD_FMT)
_REL_FMT = "=II"
_REL_SIZE = struct.calcsize(_REL_FMT)
_MAGIC = 0xDB
_F_BDF_VALID = 1 << 0


def _ioc(richtung: int, typ: int, nr: int, groesse: int) -> int:
    return (richtung << 30) | (groesse << 16) | (typ << 8) | nr


_IOC_WRITE, _IOC_READ = 1, 2
IOC_HOLD = _ioc(_IOC_READ | _IOC_WRITE, _MAGIC, 1, _HOLD_SIZE)
IOC_RELEASE = _ioc(_IOC_WRITE, _MAGIC, 2, _REL_SIZE)


def _ioc_arg(op: int) -> int:
    """``_IOWR`` setzt das oberste Bit; ``fcntl.ioctl`` will es vorzeichenbehaftet.

    ``_IOC_READ|_IOC_WRITE`` ist 3, also Bit 31 gesetzt -- die Zahl passt
    nicht in ein ``int`` und CPython lehnt sie je nach Version ab. Der
    Zweierkomplement-Wert ist derselbe 32-Bit-Wert, den der Kernel sieht.
    """
    return op - (1 << 32) if op >= (1 << 31) else op


@dataclass
class SgEintrag:
    dma_adresse: int
    laenge: int


class Halter:
    """Haelt fremde dma-bufs am Leben und liefert ihre BAR1-Adressen.

    Ohne Importeur programmiert der NVIDIA-Treiber die BAR1-Seiten gar
    nicht -- der offene fd allein genuegt nachweislich nicht. Frueher
    musste dafuer eine RDMA-Karte herhalten (``ibv_reg_dmabuf_mr``); das
    GPL-Modul ``dmabuf_holder`` uebernimmt die Rolle ohne NIC und gibt
    zusaetzlich die sg-Tabelle zurueck, wodurch der Musterscan entfaellt.
    """

    def __init__(self, pfad: str = HALTER_PFAD):
        if not os.path.exists(pfad):
            raise Bar1Unverfuegbar(
                f"{pfad} fehlt. Ohne Importeur bildet der Treiber die "
                f"BAR1-Seiten des exportierten Puffers nicht ab (belegt: der "
                f"Musterscan fand ueber 65 536 Sondierungen nichts, nach dem "
                f"Anhaengen sass der Treffer sofort). Modul aus "
                f"nvidia-smallbar-p2p/dmabuf_holder/ laden. KEIN stiller "
                f"Rueckfall auf eine RDMA-Karte -- das waere eine andere "
                f"Betriebsart, nicht dieselbe."
            )
        try:
            self.fd = os.open(pfad, os.O_RDWR)
        except OSError as e:
            raise Bar1Unverfuegbar(f"{pfad} nicht oeffenbar: {e}") from e
        self._handles: list[int] = []

    def halte(self, dmabuf_fd: int, bdf: str,
              max_eintraege: int = 1024) -> tuple[int, list[SgEintrag], int]:
        """``dma_buf_attach`` + ``dma_buf_map_attachment`` als ``bdf``.

        ``bdf`` ist die **Quellkarte** -- das Geraet, das spaeter schreibt.
        Ein Dummy-Geraet gibt es bewusst nicht: ``nv_dma_buf_attach`` ruft
        ``to_pci_dev(attachment->dev)`` ungeprueft auf (nv-dmabuf.c:1033)
        und ``nv_dma_map_peer`` greift auf ``->resource[]`` zu
        (nv-dma.c:749); ein Geraet ohne eingebettetes ``struct pci_dev``
        wuerde dort ausserhalb des Objekts lesen.

        Zwei Durchgaenge, falls noetig: das Modul traegt die WAHRE Zahl der
        sg-Eintraege in ``arg.nents`` ein, kopiert aber hoechstens
        ``max_entries`` davon heraus (dmabuf_holder.c:216). Eine
        abgeschnittene Tabelle sieht aus wie eine kurze zusammenhaengende
        Strecke -- der Aufbau scheiterte dadurch an einer Fenstergrenze, die
        es gar nicht gibt. Wer mehr Eintraege meldet als hineinpassten,
        bekommt einen zweiten Halt mit passendem Puffer; der erste bleibt
        so lange stehen, damit die BAR1-Abbildung dazwischen nie faellt.
        """
        handle_, eintraege, total_len, nents = self._halte_einmal(
            dmabuf_fd, bdf, max_eintraege
        )
        if nents > max_eintraege:
            alt = handle_
            try:
                handle_, eintraege, total_len, nents2 = self._halte_einmal(
                    dmabuf_fd, bdf, nents
                )
            finally:
                self.gib_frei(alt)
            if nents2 > nents:
                self.gib_frei(handle_)
                raise Bar1Unverfuegbar(
                    f"Der Halter meldet {nents2} sg-Eintraege, angefordert "
                    f"waren {nents} -- die Tabelle waechst zwischen zwei "
                    f"Haltevorgaengen. Ohne vollstaendige Tabelle ist die "
                    f"zusammenhaengende Laenge nicht bestimmbar."
                )
        if not eintraege:
            raise Bar1Unverfuegbar(
                "Der Halter meldet 0 sg-Eintraege -- die Abbildung ist leer. "
                "Ohne sg-Adresse ist der BAR1-Versatz nicht bestimmbar; der "
                "Musterscan waere der Rueckfall, er gehoert aber nicht in "
                "einen Transport."
            )
        return handle_, eintraege, total_len

    def _halte_einmal(self, dmabuf_fd: int, bdf: str,
                      max_eintraege: int) -> tuple[int, list[SgEintrag], int, int]:
        dom, bus, slot, func = _zerlege_bdf(bdf)
        puffer = ctypes.create_string_buffer(16 * max_eintraege)
        arg = bytearray(struct.pack(
            _HOLD_FMT,
            dmabuf_fd, _F_BDF_VALID, dom, bus, slot, func, 0,
            max_eintraege, 0, ctypes.addressof(puffer),
            0, 0, 0, 0,
        ))
        try:
            fcntl.ioctl(self.fd, _ioc_arg(IOC_HOLD), arg, True)
        except OSError as e:
            raise Bar1Unverfuegbar(
                f"DMABUF_HOLDER_IOC_HOLD fehlgeschlagen ({e}). Ohne "
                f"gehaltene Anhaftung gibt es keine BAR1-Abbildung und damit "
                f"keinen Direktpfad."
            ) from e
        werte = struct.unpack(_HOLD_FMT, bytes(arg))
        handle_, nents, _dmabuf_size, total_len = werte[10], werte[11], werte[12], werte[13]
        self._handles.append(handle_)
        eintraege = []
        gueltig = min(nents, max_eintraege)
        roh = bytes(puffer.raw[: 16 * gueltig])
        for i in range(gueltig):
            a, l = struct.unpack_from("=QQ", roh, 16 * i)
            eintraege.append(SgEintrag(a, l))
        return handle_, eintraege, int(total_len), int(nents)

    def gib_frei(self, handle_: int) -> None:
        # Aus der Liste nehmen, BEVOR der Ioctl laeuft: sonst gibt
        # `schliesse()` denselben Handle ein zweites Mal frei und
        # protokolliert eine Warnung ueber einen Fehler, den es nicht gibt.
        if handle_ in self._handles:
            self._handles.remove(handle_)
        try:
            fcntl.ioctl(self.fd, _ioc_arg(IOC_RELEASE),
                        struct.pack(_REL_FMT, handle_, 0))
        except OSError as e:
            logger.warning("HTCCL-BAR1: RELEASE(%d) fehlgeschlagen: %s", handle_, e)

    def schliesse(self) -> None:
        for h in list(self._handles):
            self.gib_frei(h)
        self._handles.clear()
        try:
            os.close(self.fd)
        except OSError:
            pass


def _zerlege_bdf(bdf: str) -> tuple[int, int, int, int]:
    s = bdf.strip().lower()
    if s.count(":") == 1:
        s = "0000:" + s
    dom, rest = s.split(":", 1)
    bus, rest = rest.split(":", 1)
    slot, func = rest.split(".", 1)
    return int(dom, 16), int(bus, 16), int(slot, 16), int(func, 16)


# ===========================================================================
# sysfs: BAR1-Lage und -Groesse
# ===========================================================================


@dataclass(frozen=True)
class Bar1Fenster:
    bdf: str
    basis: int
    groesse: int          # brutto laut sysfs

    @property
    def ende(self) -> int:
        return self.basis + self.groesse


def bar1_fenster(bdf: str) -> Bar1Fenster:
    """BAR1 aus ``/sys/bus/pci/devices/<bdf>/resource``, Zeile 1.

    ACHTUNG: das ist die **Bruttogroesse** der Apertur. Wieviel davon
    tatsaechlich fuer Peer-Abbildungen zur Verfuegung steht, ist ungemessen
    -- RM belegt Teile selbst. ``pruefe_fensterbedarf`` rechnet deshalb
    gegen das, was sich wirklich exportieren liess, und nicht gegen diese
    Zahl.
    """
    pfad = f"/sys/bus/pci/devices/{bdf}/resource"
    try:
        with open(pfad) as f:
            zeilen = f.read().strip().split("\n")
    except OSError as e:
        raise Bar1Unverfuegbar(f"{pfad} nicht lesbar: {e}") from e
    if len(zeilen) < 2:
        raise Bar1Unverfuegbar(f"{pfad}: keine BAR1-Zeile")
    start_s, ende_s, _flags = zeilen[1].split()
    start, ende = int(start_s, 16), int(ende_s, 16)
    if ende <= start:
        raise Bar1Unverfuegbar(f"{bdf}: BAR1 ist leer ({start_s}..{ende_s})")
    return Bar1Fenster(bdf=bdf, basis=start, groesse=ende - start + 1)


SEITE = 4096

#: Groesster Verbund, fuer den die Kernelargumente Platz haben.
MAX_RANGE = 8


def fensterbedarf(algorithmus: str, nbytes: int, welt: int) -> int:
    """Wieviel BAR1 die Zerlegung gleichzeitig abgebildet braucht.

    **Nachgezaehlt an den portierten Kernen, nicht geschaetzt.** Beide
    brauchen ``2(R-1)`` Schlitze zu je ``ceil(N/R)``, aus verschiedenen
    Gruenden:

    * **Gechunktes Netz**: ``R-1`` Schlitze fuer den Reduce-Scatter und noch
      einmal ``R-1`` fuer den Allgather. Die beiden Saetze duerfen sich
      **nicht** teilen: zwischen "ich lese meine RS-Schlitze" und "der
      andere schreibt seinen AG-Chunk" steht keine Ordnung. Mit einem
      gemeinsamen Satz braeuchte es eine dritte Sperre, und die kostet mehr
      als die Schlitze.
    * **Ring**: einer je Schritt, und es gibt ``2(R-1)`` Schritte. Zwei
      Schlitze abwechselnd zu benutzen ginge nur, wenn der Sender wuesste,
      dass der Empfaenger den Schlitz von vor zwei Schritten schon gelesen
      hat -- er beobachtet aber nur seinen VORGAENGER, nie seinen
      Nachfolger.

    Die frueheren Zahlen dieses Moduls (Netz ``R-1``, Ring ``2``, jeweils
    mal zwei fuer Doppelpufferung) ergaben bei ``R=3`` denselben Wert und
    waren nur deshalb nie aufgefallen. Ab ``R=4`` weichen sie ab, und zwar
    nach unten -- ein Fenster, das nicht reicht, haette wie ein Fenster
    ausgesehen, das reicht.

    * **Stern**: auf der Nabe ``R-1`` volle Puffer -- der Grund, warum er
      auf 256-MiB-Karten als erstes anschlaegt. Nicht portiert, siehe
      Moduldoku; die Zeile steht hier, damit der Planer denselben Wert
      rechnet.
    """
    if welt < 2:
        return 0
    anteil = -(-nbytes // welt)          # aufrunden
    # ``netz_pipe`` steht hier bei netz und ring, weil sein Bereich genauso
    # gross ist: 2*T*(R-1) Schlitze zu je chunk_max/T, also wieder
    # 2(R-1)*chunk_max. Die GENAUE Zahl -- mit dem Verschnitt aus dem
    # Abrunden der Schlitzgroesse -- rechnet
    # ``htccl_bar1_pipe_ext.pipe_fensterbedarf``; sie ist kleiner und wird
    # in ``handles`` zusaetzlich geprueft.
    if algorithmus in ("netz", "netz_pipe", "ring", "hierarchisch"):
        return 2 * (welt - 1) * anteil
    if algorithmus == "stern":
        return 2 * (welt - 1) * nbytes
    raise ValueError(f"unbekannter Algorithmus {algorithmus!r}")


def geometrie(welt: int, max_bytes: int, mit_a2a: bool = True,
              mit_pipe: bool = False, erg_ring: int = 0) -> dict:
    """Die Speicherordnung EINER Empfangsregion, fuer beliebiges R.

    Sie traegt alle Verfahren **gleichzeitig**, damit ein Plan je Groesse und
    je Operation umschalten kann, ohne dass irgendetwas neu abgebildet wird:

    ===========  =================  ========================================
    Versatz      Inhalt             Groesse
    ===========  =================  ========================================
    ``0``        Netz-RS-Schlitze   ``(R-1) * chunk_max``
    ...          Netz-AG-Schlitze   ``(R-1) * chunk_max``
    ``off_ring`` Ring-Schlitze      ``2(R-1) * chunk_max``
    ``off_a2a``  a2a-Schlitze       ``2(R-1) * chunk_max``
    ``off_pipe`` netz_pipe-Schlitze ``2(R-1) * chunk_max``
    ===========  =================  ========================================

    ``chunk_max`` ist auf eine Seite aufgerundet -- ein Schlitz, der auf
    einer Seitengrenze beginnt, kann nie mit dem Nachbarschlitz eine Seite
    teilen, und ein ueberlanger Schreibvorgang trifft dann eine eigene
    Seite statt fremde Nutzlast.

    **Warum a2a 2(R-1) Schlitze braucht und nicht (R-1).** Ein Schlitz je
    Sender wuerde genuegen, wenn der Sender wuesste, dass der Empfaenger den
    vorigen Inhalt schon gelesen hat. Die Flagge sagt aber nur
    "geschrieben". Die Alternativen sind eine zweite Sperre (bei
    MoE-Groessen die halbe Latenz) oder zwei Haelften, zwischen denen die
    Rundennummer wechselt. Es sind zwei Haelften; die Begruendung, warum
    zwei genuegen, steht beim Kernel in ``htccl_bar1_ext.py``.

    **Was das den all_reduce-Weg kostet.** Die Region waechst von
    ``4(R-1)`` auf ``6(R-1)`` Schlitze, die groesste all_reduce-Nutzlast in
    einem gegebenen Fenster sinkt also auf zwei Drittel. Keine gemessene
    Zahl aendert sich dadurch -- nur die Decke, ab der ``handles`` False
    sagt. Wer sie zurueck will, setzt ``SGLANG_HTCCL_BAR1_A2A=0``; dann ist
    ``mit_a2a`` False und die Ordnung ist Byte fuer Byte die alte.

    **Warum netz_pipe einen EIGENEN Bereich bekommt und nicht den von
    netz.** Die Bereiche der Verfahren muessen paarweise disjunkt sein, und
    zwar nicht nur innerhalb eines Aufrufs. Wenn Rang A seine Runde ``n``
    beendet, kann Rang B den Allgather-Schlitz dieser Runde noch lesen --
    A wartet vor dem Ende nur auf B's Flagge, nicht auf B's Lesevorgang.
    Bei ``netz`` faellt das nicht auf, weil A's naechster Schreibvorgang in
    die RS-Haelfte geht und B in der AG-Haelfte liest. Ein ``netz_pipe``,
    der den ganzen Netz-Bereich benutzte, traefe die AG-Haelfte sofort.
    Ein eigener Bereich macht die Frage gegenstandslos.

    Der Bereich wird nur angelegt, wenn ``mit_pipe`` gesetzt ist
    (``SGLANG_HTCCL_BAR1_PIPE=1``); ohne ihn ist die Ordnung Byte fuer Byte
    die gemessene.
    """
    if welt < 2:
        raise ValueError("welt < 2")
    n4_max = max_bytes // 16
    chunk4 = -(-n4_max // welt)
    chunk_max = ((chunk4 * 16 + SEITE - 1) // SEITE) * SEITE
    schlitze = 2 * (welt - 1)
    off_netz = 0
    off_ring = schlitze * chunk_max
    off_a2a = 2 * schlitze * chunk_max
    from sglang.srt.distributed.device_communicators.htccl_bar1_pipe_ext import (
        erg_stride_bytes,
    )

    saetze = 2 + (1 if mit_a2a else 0)
    off_pipe = saetze * schlitze * chunk_max
    if mit_pipe:
        saetze += 1
    off_erg = saetze * schlitze * chunk_max
    ring = int(erg_ring) if mit_pipe else 0
    erg_stride = erg_stride_bytes(max_bytes) if ring > 0 else 0
    region = off_erg + ring * erg_stride + SEITE
    return {
        "chunk_max": chunk_max,
        "off_netz": off_netz,
        "off_ring": off_ring,
        # -1 heisst ausdruecklich "gibt es nicht" und nicht "liegt bei 0" --
        # ein Versatz 0 waere der Netz-Bereich.
        "off_a2a": off_a2a if mit_a2a else -1,
        "a2a_schlitz": chunk_max if mit_a2a else 0,
        "off_pipe": off_pipe if mit_pipe else -1,
        "off_erg": off_erg if ring > 0 else -1,
        "erg_stride": erg_stride,
        "erg_ring": ring,
        "region_bytes": region,
        "max_bytes": max_bytes,
        "mit_a2a": bool(mit_a2a),
        "mit_pipe": bool(mit_pipe),
    }


def flaggen_bedarf(welt: int, mit_a2a: bool = True,
                   mit_pipe: bool = False) -> int:
    """``(2 + 2(R-1) [+ 1]) * R * 256`` Byte, plus ``4 R * 256`` fuer die Pipe.

    Eine 256-Byte-Zeile je (Topologie, Schritt, Sender): kein False Sharing
    zwischen Sendern, keins zwischen Schritten, keins zwischen Topologien.
    Netz hat 2 Schritte, Ring ``2(R-1)``, a2a genau **einen**. Bei R=8 sind
    das 34 KiB und damit weit unter einer Allokationsgranularitaet.

    ``netz_pipe`` haengt vier Zeilen je Rang hinten an (``tailRS``,
    ``tailAG``, ``headRS``, ``headAG``) -- **unabhaengig von K und T**, weil
    es ein Schiebefenster mit einem Zaehler je Verbindung ist und nicht eine
    Flagge je Chunk. Hinten angehaengt, damit jeder bestehende
    Zeilenversatz Byte fuer Byte bleibt.
    """
    from sglang.srt.distributed.device_communicators.htccl_bar1_pipe_ext import (
        pipe_flaggen_zusatz,
    )

    grund = (2 + 2 * (welt - 1) + (1 if mit_a2a else 0)) * welt * 256
    return grund + (pipe_flaggen_zusatz(welt) if mit_pipe else 0)


def fbasis_a2a(welt: int) -> int:
    """Versatz der a2a-Flaggenzeilen in der Flaggenregion.

    Hinter Netz und Ring, damit die beiden gemessenen Topologien Byte fuer
    Byte dort liegen, wo sie lagen. Dieselbe Rechnung steht im Kernel
    NICHT noch einmal -- sie wird als Argument hereingereicht, weil eine
    zweite Fassung genau die Stelle waere, an der Sender und Empfaenger auf
    verschiedene Zeilen zeigen.
    """
    return (2 + 2 * (welt - 1)) * welt * 256


def ag_plan(laengen, schlitz: int) -> list:
    """Die Rundenzerlegung eines ``all_gather``. Reine Arithmetik.

    ``laengen[i]`` ist die Scherbe von Rang ``i`` in **Byte**; das Ergebnis
    ist deren Aneinanderreihung, also ``sum(laengen)`` Byte, mit Rang ``i``
    bei Versatz ``sum(laengen[:i])``.

    Geliefert wird je Runde eine Liste von ``(sende_versatz, laenge,
    empfangs_versatz)`` je Rang -- alles in Byte, alles absolut, nichts als
    Praefixsumme zu erratenden:

    * ``sende_versatz`` zeigt in die EIGENE Scherbe (denselben Ausschnitt
      fuer jedes Ziel -- genau das unterscheidet all_gather von
      all_to_all),
    * ``empfangs_versatz`` in das Ergebnis, also ``basis[i] + k*schlitz``.

    **Warum ueberhaupt Runden.** Eine Scherbe kann groesser sein als ein
    Schlitz. Der Fehlerfall aus der Uebergabe ist genau der: 10 600 448 Byte
    all_gather gegen einen a2a-Schlitz von knapp 8 MiB bei 96 MiB Fenster.
    Statt sich ueber ``handles`` abzumelden -- was unter einer
    CUDA-Graph-Aufzeichnung den Lauf abbricht, weil es keinen Ausweichweg
    gibt -- laeuft die Scherbe in ``ceil(max(laengen)/schlitz)`` Runden.

    **Warum das eine Aufzeichnung ueberlebt.** Die Rundenzahl haengt nur an
    ``laengen`` und ``schlitz``. Beide sind gruppenweit gleich und fuer eine
    aufgezeichnete Form konstant, die Zahl der Kernelstarts ist also
    eingebrannt und bei jeder Wiedergabe dieselbe -- dasselbe Argument, mit
    dem ``htccl_device.all_reduce`` seine Schlitzschleife aufzeichnen darf.
    Kein Hostcode entscheidet hier je Runde etwas, was sich zwischen
    Aufzeichnung und Wiedergabe aendern koennte. Das ist der Unterschied zum
    Direkt-Modus der Pipe, dessen hostseitiger Ringindex genau daran
    scheitert (siehe ``_erg_platz``).

    **Rangeinheitlich.** Jeder Rang rechnet aus DEMSELBEN ``laengen``-Vektor,
    also fallen bei allen gleich viele Runden an. Zaehlte ein Rang anders,
    waere das kein Fehler, sondern ein Haenger: die anderen warteten in der
    Sperre einer Runde, die er nicht mehr fuehrt.

    **Ungleiche Scherben** sind hier Arithmetik, keine Umschreibung. Die
    heutige Naht (``HTCCLCommunicator.all_gather``) ist gleichverteilt -- ihr
    Ergebnis ist ``(R,) + form``, das GEHT nicht ungleich, und die ungleiche
    Form heisst in sglang ``all_gatherv`` und ist unter HTCCL ausdruecklich
    nicht gedeckt. Diese Funktion nimmt trotzdem einen Vektor: unter uneven
    TP sind ungleiche Scherben der Normalfall, und die Stelle, an der eine
    Gleichverteilung ANGENOMMEN wird, ist die Stelle, an der ein spaeteres
    ``all_gatherv`` still falsche Versaetze bekommt. Ein Rang, dessen
    Scherbe frueher endet, bekommt in den restlichen Runden Laenge 0 -- er
    faehrt die Sperre mit, ohne Bytes zu bewegen.
    """
    laengen = [int(x) for x in laengen]
    if not laengen:
        return []
    if schlitz <= 0:
        raise ValueError(f"Schlitzgroesse {schlitz} ist nicht positiv")
    if any(n < 0 for n in laengen):
        raise ValueError(f"negative Scherbenlaenge in {laengen}")
    basis, acc = [], 0
    for n in laengen:
        basis.append(acc)
        acc += n
    runden = max(1, -(-max(laengen) // schlitz))
    plan = []
    for k in range(runden):
        eine = []
        for i, n in enumerate(laengen):
            a = min(k * schlitz, n)
            b = min((k + 1) * schlitz, n)
            eine.append((a, b - a, basis[i] + a))
        plan.append(eine)
    return plan


def max_nutzlast(welt: int, region_bytes: int, mit_a2a: bool = True,
                 mit_pipe: bool = False, erg_ring: int = 0) -> int:
    """Groesste Nutzlast, deren Schlitze in eine Region dieser Groesse passen.

    Umkehrung von :func:`geometrie`. Bewusst konservativ gerundet und
    danach gegengerechnet -- eine Umkehrung, die um eine Seite danebenliegt,
    faellt sonst erst im heissen Pfad auf. Die Gegenrechnung ist genau der
    Grund, warum hier keine zweite Fassung der Faktorenrechnung stehen
    muss: ``geometrie`` selbst hat das letzte Wort.
    """
    if welt < 2 or region_bytes <= SEITE:
        return 0
    # 2 Saetze fuer netz, 2 fuer ring, je 2 fuer a2a und die Pipe -- also 4
    # als Sockel, nicht 2. Ausgeschrieben statt als "(6 wenn a2a sonst 4)",
    # damit der vierte Summand nicht wieder in einer Zahl verschwindet.
    schlitze = (4 + (2 if mit_a2a else 0) + (2 if mit_pipe else 0)) * (welt - 1)
    ring = int(erg_ring) if mit_pipe else 0
    # Der Ergebnisring kostet ``L * roundup(N, SEITE)``, und ``N`` ist
    # ``chunk_max * R``. In Einheiten von chunk_max sind das ``L * R``
    # zusaetzliche Einheiten zu den ``schlitze`` -- deshalb steht der Ring
    # hier IM NENNER und nicht als Abzug. Ein Abzug haette den Anfangswert
    # so weit danebengelegt, dass die Gegenrechnung unten in
    # 32-Byte-Schritten haette heruntersuchen muessen.
    nenner = schlitze + ring * welt
    chunk_max = ((region_bytes - SEITE) // nenner // SEITE) * SEITE
    if chunk_max <= 0:
        return 0
    n = (chunk_max // 16) * welt * 16
    while n > 0 and geometrie(welt, n, mit_a2a, mit_pipe,
                              ring)["region_bytes"] > region_bytes:
        n -= welt * 16
    return n


# ===========================================================================
# fd-Austausch ueber SCM_RIGHTS
# ===========================================================================


def _tausche_fds(cpu_group, rank: int, welt: int,
                 eigene_fds: list[int]) -> list[list[int]]:
    """Jeder Rang gibt seine dma-buf-fds an alle anderen.

    Es sind ZWEI je Rang: die Nutzlastregion und die Flaggenregion. Sie
    liegen in getrennten VMM-Allokationen, weil die Sonde sie so vermessen
    hat -- dieselbe Anordnung, dieselben Zahlen. Beide gehen in EINER
    ``SCM_RIGHTS``-Nachricht, damit der Austausch nicht zweimal laeuft und
    ein halb geglueckter Durchgang keinen Rang mit halber Tabelle
    zuruecklaesst.

    ``SCM_RIGHTS`` auf AF_UNIX, weil ein fd prozesslokal ist: ueber gloo
    laesst sich eine Zahl uebertragen, aber kein Zugriffsrecht. Der Ablauf
    ist bewusst rundenweise und seriell -- er laeuft genau einmal beim
    Start, und ein verklemmter Bootstrap kostet mehr als ein paar
    Millisekunden Startzeit.
    """
    import socket

    import torch.distributed as dist

    traeger = [None]
    if rank == 0:
        traeger = [tempfile.mkdtemp(prefix="htccl-bar1-")]
    dist.broadcast_object_list(
        traeger, src=dist.get_global_rank(cpu_group, 0), group=cpu_group
    )
    verz = str(traeger[0])
    pfad = os.path.join(verz, f"r{rank}.sock")

    anzahl = len(eigene_fds)
    fds: list[list[int]] = [[] for _ in range(welt)]
    fds[rank] = list(eigene_fds)
    horcher = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    try:
        if os.path.exists(pfad):
            os.unlink(pfad)
        horcher.bind(pfad)
        horcher.listen(welt)
        dist.barrier(group=cpu_group)

        for besitzer in range(welt):
            if besitzer == rank:
                for _ in range(welt - 1):
                    verb, _ = horcher.accept()
                    with verb:
                        socket.send_fds(verb, [b"x"], list(eigene_fds))
            else:
                ziel = os.path.join(verz, f"r{besitzer}.sock")
                verb = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
                letzter: Optional[Exception] = None
                for _ in range(200):        # der Peer bindet ggf. noch
                    try:
                        verb.connect(ziel)
                        letzter = None
                        break
                    except OSError as e:
                        letzter = e
                        time.sleep(0.01)
                if letzter is not None:
                    raise Bar1Unverfuegbar(
                        f"fd-Austausch: {ziel} nicht erreichbar ({letzter})"
                    )
                with verb:
                    _daten, empfangen, _fl, _adr = socket.recv_fds(
                        verb, 1, anzahl
                    )
                if len(empfangen) != anzahl:
                    raise Bar1Unverfuegbar(
                        f"fd-Austausch: Rang {besitzer} hat {len(empfangen)} "
                        f"statt {anzahl} fds gesendet"
                    )
                fds[besitzer] = list(empfangen)
            dist.barrier(group=cpu_group)
    finally:
        horcher.close()
        try:
            os.unlink(pfad)
        except OSError:
            pass
    return fds


# ===========================================================================
# Der Transport
# ===========================================================================


@dataclass
class Abbildung:
    """Eine abgebildete und registrierte fremde BAR1-Region."""

    bar1_basis: int
    bar1_versatz: int          # Versatz der Region in BAR1
    laenge: int                # TATSAECHLICH abgebildete, zusammenhaengende Laenge
    mmap_obj: object           # gehalten, damit die Abbildung lebt
    reg_adresse: int           # Adresse, unter der REGISTRIERT wurde
    host_adresse: int          # Nutzeradresse der Region (reg + Vorlauf)
    dev_ptr: int               # Geraetezeiger DIESER Karte auf die fremde BAR
    halter_handle: int


@dataclass
class PeerZiel:
    """Was der Aufbau je Peer festgestellt hat -- danach unveraenderlich.

    Zwei Regionen je Peer, in getrennten VMM-Allokationen und damit
    getrennt exportiert und abgebildet: die Nutzlastschlitze und die
    Flaggenzeilen. Genau die Anordnung, in der die Sonde gemessen hat.
    """

    rang: int
    bdf: str
    nutz: Abbildung
    flag: Abbildung
    byte_beleg: bool = False

    # Die beiden Namen, unter denen der Punkt-zu-Punkt-Weg (put/paar) die
    # Nutzlastregion kennt. Bleiben, damit der Messfuehler unveraendert ist.
    @property
    def dev_ptr(self) -> int:
        return self.nutz.dev_ptr

    @property
    def laenge(self) -> int:
        return self.nutz.laenge


class HTCCLBar1Transport:
    """BAR1-Direkttransport.

    Erfuellt die Transport-Naht aus ``htccl.py`` (Zeilen 67-80):
    ``handles(op, nbytes) -> bool`` plus ``htccl_<op>(comm, ...)`` je
    angebotener Operation.

    Was er bietet und was davon gemessen ist:

    * ``htccl_all_reduce`` ueber die portierten Kerne ``netz`` und
      ``ring``. In der Sonde durchgemessen (float32, drei Raenge, Rig 1);
      die Tabelle steht in der Moduldoku.
    * ``put(ziel, quell_ptr, nbytes, versatz)`` -- ein einzelner
      Schreibvorgang in die BAR der Zielkarte.
    * ``paar``/``paar_empfang`` -- der Messfuehler, den
      ``htccl_matrix.HTCCLMatrixPlaner`` fuer echte Kantenkapazitaeten
      braucht (statt der Eigenlast-Schaetzung).
    * ``byte_beleg_alle()`` -- der Byte-Beleg je gerichtetem Paar. Faellt
      er durch, wird die Kante gestrichen, egal was der Treiber meldet.
      Auf diesem Rig meldete der Treiber fuer ein Paar Peer-Zugriff und
      lieferte 4096 von 1048576 Byte.

    ``handles`` sagt ``True`` **nur**, wenn die Peer-Zeiger stehen, jeder
    Byte-Beleg gehalten hat, die Groesse in den Bereich passt und der
    Fensterbedarf in die TATSAECHLICH abgebildete Laenge passt. Sonst
    ``False`` -- ohne Ausnahme, ohne Notpfad.
    """

    #: all_reduce (gemessen), all_to_all (eigener Kern, ungemessen) und
    #: all_gather (auf demselben Kern, ungemessen -- siehe
    #: :meth:`htccl_all_gather`).
    #:
    #: ``reduce_scatter`` und ``broadcast`` fehlen weiter, und zwar mit
    #: Grund, nicht aus Versehen:
    #:
    #: * ``reduce_scatter`` braucht eine REDUKTION. Der a2a-Kern bewegt
    #:   Bytes und kennt keinen Datentyp; er traegt all_gather deshalb
    #:   gratis und reduce_scatter gar nicht. Die RS-Phase der beiden
    #:   all_reduce-Kerne koennte es, aber nur als eigener Einsprung mit
    #:   eigenem Schlitzsatz -- also nicht als Beifang.
    #: * ``broadcast`` ist in sglang AN ORT (``broadcast(tensor, src)``
    #:   gibt denselben Tensor zurueck), und die Erweiterung lehnt
    #:   ``in is out`` ab. Aus dem Beifang wuerde ein Zusatzpuffer plus
    #:   Kopie -- billiger als ein neuer Kern, aber nicht gratis, und
    #:   ausser Ort waere es eine andere Zusage als die der Naht.
    #:
    #: Fuer beide bleibt der laute Riegel in ``htccl._select`` zustaendig.
    #: Er nennt die Operation, und weil diese Menge hier die einzige
    #: Wahrheit darueber ist, was gedeckt ist, kann die Meldung nicht
    #: veralten.
    #:
    #: Beide Schreibweisen von all_to_all stehen hier, weil die Naht in
    #: htccl.py die Operation unter dem Namen ``all_to_all`` fragt, waehrend
    #: der einzige echte Aufrufer in sglang (GroupCoordinator, equal split)
    #: ``all_to_all_single`` heisst. Zwei Namen, ein Weg -- besser als eine
    #: Umbenennung an der Nahtstelle, die man beim Lesen uebersieht.
    HTCCL_OPS: frozenset = frozenset(
        {"all_reduce", "all_gather", "all_to_all", "all_to_all_single"}
    )

    def __init__(self, cpu_group, device, fenster_bytes: int,
                 aktiviert: Optional[bool] = None, gruppe: str = ""):
        import torch
        import torch.distributed as dist

        self.cpu_group = cpu_group
        self.device = device
        #: Name der Kommunikatorgruppe ("tp", "dcp", ...). Er steht hier,
        #: weil BAR1 eine PROZESSWEITE Ressource ist: was diese Gruppe
        #: festnagelt, fehlt der naechsten. Ohne den Namen liesse sich weder
        #: buchen noch sagen, wer den Platz hat.
        self.gruppe = gruppe
        self.rank = dist.get_rank(cpu_group)
        self.welt = dist.get_world_size(cpu_group)
        self.fenster_bytes = int(fenster_bytes)
        self._auf = False
        self._peers: dict[int, PeerZiel] = {}
        self._halter: Optional[Halter] = None
        self._cuda: Optional[_Cuda] = None
        self._eigen = (0, 0, 0)          # Nutzlast: dptr, handle, groesse
        self._eigen_flag = (0, 0, 0)     # Flaggen:  dptr, handle, groesse
        self._eigen_fuehler = None
        self._dmabuf_fds: list[int] = []       # eigene, exportierte
        self._halte_fds: list[int] = []        # /dev/nvidiactl, /dev/nvidiaN
        self._fremde_fds: list[list[int]] = []
        self._ext = None
        self._geo: dict = {}
        self._plan = None                      # optionaler Plan aus htccl_matrix
        # Faehigkeit, gruppenweit einheitlich. Erst nach _baue_auf gueltig.
        self._fenster_minimum = 0
        self._belege_stehen = False
        self._runde_dev = None
        self._ctl_dev = None

        if aktiviert is None:
            aktiviert = os.environ.get("SGLANG_HTCCL_MATRIX_DIRECT", "1") not in (
                "0", "nein", "aus", "false"
            )
        if not aktiviert:
            raise Bar1Unverfuegbar(
                "per SGLANG_HTCCL_MATRIX_DIRECT=0 abgeschaltet"
            )
        if self.welt > MAX_RANGE:
            raise Bar1Unverfuegbar(
                f"{self.welt} Raenge, aber die Kernelargumente fassen "
                f"hoechstens {MAX_RANGE}. Die Grenze steht in "
                f"htccl_bar1_ext.py (HTCCL_BAR1_MAX_RANKS) und ist dort "
                f"nachvollziehbar zu heben -- nicht hier zu umgehen."
            )
        self.ordinal = device.index if getattr(device, "index", None) is not None \
            else torch.cuda.current_device()
        # Betriebsparameter der Kerne. Alle rangeinheitlich, wie jede andere
        # SGLANG_HTCCL*-Variable.
        self.threads = int(os.environ.get("SGLANG_HTCCL_BAR1_THREADS", "256"))
        # ~30 s bei 2 GHz -- ein abgedrifteter Peer faengt sich im Kernel eine
        # Frist statt die Karte auf Dauer zu belegen. Gleiche Groessenordnung
        # wie HTCCLDeviceTransport._TIMEOUT_CYCLES, aus demselben Grund.
        self.deckel_zyklen = int(
            os.environ.get("SGLANG_HTCCL_BAR1_DECKEL_ZYKLEN", "60000000000")
        )
        # Ladeform der Flagge: 2 = ld.mmio.relaxed.sys (die einzige echte
        # Cache-Umgehung, Vorgabe der Sonde), 0 = ld.global.cv.
        self.ladeform = int(os.environ.get("SGLANG_HTCCL_BAR1_LADEFORM", "2"))
        # Lesefluss: nur noetig, wenn Nutzlast und Flagge an verschiedenen
        # PCIe-Zielen liegen. Hier liegen sie es nicht; Vorgabe aus.
        self.fluss = int(os.environ.get("SGLANG_HTCCL_BAR1_FLUSS", "0"))
        # Ab dieser Nutzlast der cooperative Mehrblockstart. 4 MiB, weil in
        # MESSUNG_ALLES_IM_SELBEN_LAUF.md ab 4 MiB die 'gitter'-Variante
        # gewinnt und darunter '1blk'.
        self.gitter_ab = int(
            os.environ.get("SGLANG_HTCCL_BAR1_GITTER_AB", str(4 << 20))
        )
        # Darf die cooperative Variante WAEHREND einer CUDA-Graph-Aufzeichnung
        # gestartet werden? Vorgabe NEIN -- und das ist eine Vorsichtsregel,
        # keine festgestellte Unvertraeglichkeit.
        #
        # Was die Kopfdateien auf diesem Rig hergeben (CUDA 12.9):
        # `CU_LAUNCH_ATTRIBUTE_COOPERATIVE = 2` ist ausdruecklich "Valid for
        # graph nodes, launches" (cuda.h:2043, driver_types.h:3800) -- ein
        # cooperative Start ist als Graphknoten also DARSTELLBAR. Ob der
        # Treiber ihn auch aus einem Stream-Capture heraus als solchen Knoten
        # aufnimmt, steht dort NICHT, und die Kopfdateien sind alles, was
        # ohne freie Karten zu haben ist. `benchmark/bar1_graph_check.py`
        # beantwortet genau diese Frage und weist den Fehlercode aus.
        #
        # Bis dahin kostet die Vorsicht im Decode nichts: dort liegen die
        # Nutzlasten unter `gitter_ab`, also faellt ohnehin `1blk`. Die
        # Umschaltung greift nur oberhalb der Schwelle, und dort ist sie
        # sichtbar (einmaliger Protokolleintrag), nicht still.
        self.graph_gitter = os.environ.get(
            "SGLANG_HTCCL_BAR1_GRAPH_GITTER", "0"
        ) not in ("0", "nein", "aus", "false")
        self._graph_gitter_gemeldet = False
        # Notschwelle netz->ring, falls kein Plan hereingereicht wird.
        self.ring_ab = int(
            os.environ.get("SGLANG_HTCCL_BAR1_RING_AB", str(1 << 20))
        )
        self.min_bytes = int(os.environ.get("SGLANG_HTCCL_BAR1_MIN_BYTES", "4096"))
        self.max_bytes = 0
        # all_to_all belegt einen dritten Schlitzsatz in derselben Region und
        # kostet damit ein Drittel der groessten all_reduce-Nutzlast (siehe
        # `geometrie`). Rangeinheitlich wie jede andere SGLANG_HTCCL*-Variable;
        # 0 stellt die alte Speicherordnung Byte fuer Byte wieder her.
        self.a2a_an = os.environ.get("SGLANG_HTCCL_BAR1_A2A", "1") not in (
            "0", "nein", "aus", "false"
        )
        #: Erst nach `byte_beleg_a2a` gueltig. Ohne bestandenen Beleg meldet
        #: sich all_to_all ab -- all_reduce bleibt davon unberuehrt.
        self._a2a_beleg = False

        # -- netz_pipe (gepipelinetes Netz, htccl_bar1_pipe_ext) ------------
        # AUS als Vorgabe. Eingeschaltet belegt es einen weiteren Schlitzsatz
        # und vier Flaggenzeilen je Rang; ausgeschaltet ist jede Zahl und
        # jeder Versatz dieses Moduls Byte fuer Byte der gemessene.
        self.pipe_an = os.environ.get("SGLANG_HTCCL_BAR1_PIPE", "0") not in (
            "0", "nein", "aus", "false"
        )
        # RINGTIEFE T -- Schlitze je Phase und Verbindung. 4 aus NCCL:
        # NCCL_STEPS 8 (src/include/device.h:26) geteilt durch
        # ALLREDUCE_SLICESTEPS 2 (src/include/collectives.h:19).
        self.pipe_t = int(os.environ.get("SGLANG_HTCCL_BAR1_PIPE_T", "4"))
        # ZEITPLAN-VORLAUF P -- um wieviele Schleifenrunden das Senden dem
        # Reduzieren vorauslaeuft. GETRENNT von der Ringtiefe, und darin
        # liegt die Zeitentkopplung: der Empfaenger darf um `T - P + 1`
        # Schleifenrunden zurueckliegen, bevor der Sender blockiert. Mit
        # P = T waere das genau EINE Runde, also faktisch Gleichschritt --
        # und auf einem Rig mit x4-, x8- und x8-Anbindung und drei
        # verschiedenen Kartenmodellen ist der Versatz zwischen ungleich
        # schnellen Raengen genau das, was das Fenster absorbieren soll.
        # P = 2 ist das Minimum, das ueberhaupt pipelinet; mit T = 4 sind es
        # drei Runden Versatz.
        self.pipe_vorlauf = int(
            os.environ.get("SGLANG_HTCCL_BAR1_PIPE_VORLAUF", "2")
        )
        # Chunkzahl K. 0 = automatisch aus `pipe_chunk_bytes`.
        self.pipe_k = int(os.environ.get("SGLANG_HTCCL_BAR1_PIPE_K", "0"))
        # Zielgroesse eines Chunks bei automatischem K.
        #
        # 1 MiB, gerechnet statt geraten. Je Schleifenrunde fallen zwei
        # Sperren an; bei der cooperative Variante ist das `grid.sync()`.
        # Der Leerlauf ist `2*t_sync*(K+P)` (Sperren) plus
        # `(P-1)*T_leitung/K` (Anlauf der Pipeline), minimal bei
        # `K = sqrt((P-1)*T_leitung/(2*t_sync))`. Bei 16 MiB und zwei
        # Raengen sind `T_leitung` rund 1985 us (33,55 MB ueber die
        # gemessene Duplexdecke von 16,90 GB/s), mit P=2 und t_sync=3 us
        # also K ~ 18 -- rund 1 MiB je Chunk.
        #
        # UNGEMESSEN: `t_sync` fuer `grid.sync()` ist auf diesem Rig nicht
        # gemessen; 3 us ist eine Annahme. Traegt die 1blk-Variante -- und
        # KORREKTUR_BANDBREITE.md misst mit 256 Threads in EINEM Block
        # bereits 12,64 GB/s, also die volle Schreibrate --, dann kostet die
        # Sperre rund 0,1 us und das optimale K ist rund fuenfmal groesser.
        # Das ist die erste Achse der Messreihe.
        self.pipe_chunk_bytes = int(
            os.environ.get("SGLANG_HTCCL_BAR1_PIPE_CHUNK_BYTES", str(1 << 20))
        )
        self.pipe_k_max = int(os.environ.get("SGLANG_HTCCL_BAR1_PIPE_K_MAX", "64"))
        # EIGENE 1blk/gitter-Schwelle fuer die Pipe.
        #
        # Getrennt von `gitter_ab`, weil die Rechnung fuer die Pipe anders
        # ausgeht: `netz` zahlt zwei `grid.sync()` je Kollektiv, die Pipe
        # zahlt zwei je SCHLEIFENRUNDE, also 2(K+P) statt 2. Und der Grund,
        # aus dem die cooperative Variante bei `netz` ab 4 MiB gewinnt, gilt
        # fuer den Datenweg gar nicht: KORREKTUR_BANDBREITE.md misst die
        # Schreibrate in die Peer-BAR ueber 1 bis 16 Bloecke und 32 bis 1024
        # Threads mit 12,6-12,7 GB/s -- ohne Streuung. 256 Threads in EINEM
        # Block erreichen bereits die volle Rate. Mehr Bloecke helfen der
        # Leitung nicht; sie helfen nur der lokalen Reduktion, deren
        # cache-umgehende Ladevorgaenge Nebenlaeufigkeit brauchen.
        #
        # Vorgabe deshalb wie bisher (kein stiller Unterschied), aber als
        # eigener Hebel: nach der obigen Rechnung sollte die Pipe mit 1blk
        # besser fahren, und das ist die erste Frage der Messreihe.
        self.pipe_gitter_ab = int(
            os.environ.get("SGLANG_HTCCL_BAR1_PIPE_GITTER_AB",
                           str(self.gitter_ab))
        )
        # Empfaengerquittung (head). 1 = an. Aus gemessen zeigt, was das
        # Schiebefenster kostet; aus GEFAHREN darf sie nur, wer den
        # Zeitplanbeweis in htccl_bar1_pipe_ext gelesen hat.
        self.pipe_quittung = int(
            os.environ.get("SGLANG_HTCCL_BAR1_PIPE_QUITTUNG", "1")
        )
        # Ab dieser Nutzlast wird netz_pipe statt netz gefahren. 256 KiB, weil
        # darunter ein einziger Chunk uebrigbliebe und die Pipe dann nur die
        # Buchfuehrung des Netzes waere.
        self.pipe_ab = int(
            os.environ.get("SGLANG_HTCCL_BAR1_PIPE_AB", str(256 << 10))
        )
        # Direkt-Modus: der Allgather schreibt in den Ergebnispuffer des
        # Empfaengers statt in einen Schlitz, den der Empfaenger danach
        # auslesen und umkopieren muesste. Vorgabe AN, sobald die Pipe an
        # ist -- das ist der Punkt der Pipe. 0 ist der Kontrollversuch mit
        # derselben Speicherordnung.
        self.pipe_direkt = os.environ.get(
            "SGLANG_HTCCL_BAR1_PIPE_DIREKT", "1"
        ) not in ("0", "nein", "aus", "false")
        # Direkt-Modus WAEHREND einer Graph-Aufzeichnung. Vorgabe AUS, und
        # anders als beim Kern ist das hier kein Vorbehalt, sondern eine
        # Herleitung -- die Begruendung steht bei `_erg_platz`.
        self.pipe_direkt_graph = os.environ.get(
            "SGLANG_HTCCL_BAR1_PIPE_DIREKT_GRAPH", "0"
        ) not in ("0", "nein", "aus", "false")
        self._direkt_graph_gemeldet = False
        # Wieviele Ergebnispuffer der Ring haelt. Kostet L*max_bytes im
        # BAR-Fenster; 2 ist das Minimum, mit dem Runde n nicht in den
        # Puffer schreibt, den der Aufrufer aus Runde n-1 noch haelt.
        self.pipe_erg_ring = int(
            os.environ.get("SGLANG_HTCCL_BAR1_PIPE_ERG_RING", "2")
        )
        #: Erst nach `byte_beleg_pipe` gueltig.
        self._pipe_beleg = False
        self._pipe_ext = None
        self._schritt_dev = None
        #: Laufender Index im Ergebnisring. HOSTSEITIG und rangeinheitlich,
        #: weil jeder Rang dieselbe Folge von Kollektiven sieht (SPMD) --
        #: dieselbe Annahme, auf der schon `algorithmus_fuer` steht. Der
        #: Kern kann ihn NICHT selbst waehlen: der Host muss den
        #: Ergebnistensor bauen, bevor der Kern laeuft.
        self._erg_i = -1
        #: Schwache Verweise auf die zuletzt herausgegebenen
        #: Ergebnistensoren, je Ringplatz. Sie sind die Lebensdauerpruefung.
        self._erg_lebt: list = []
        #: Untergrenze fuer all_to_all. Bewusst NICHT `min_bytes` (4096): der
        #: Reiz von a2a ueber BAR1 liegt gerade bei den kleinen
        #: MoE-Dispatchbloecken. 16 Byte = ein Paket.
        self.a2a_min_bytes = int(
            os.environ.get("SGLANG_HTCCL_BAR1_A2A_MIN_BYTES", "16")
        )
        #: all_gather ueber den a2a-Kern. VORGABE AN, und das ist Absicht:
        #: ohne sie bricht der Standardlauf in der Graph-Aufzeichnung ab
        #: (der Riegel in htccl._select, richtig und laut). Der Schalter
        #: existiert, damit der Messende ihn gegen die gloo-Ebene stellen
        #: kann -- und weil ein neuer Weg im heissen Pfad einen Ausschalter
        #: haben muss, den man ohne Uebersetzen erreicht. Er greift NUR
        #: innerhalb von SGLANG_HTCCL_TRANSPORT=bar1|matrix; ohne HTCCL
        #: aendert er nichts.
        self.ag_an = os.environ.get("SGLANG_HTCCL_BAR1_AG", "1") not in (
            "0", "nein", "aus", "false"
        )
        #: Untergrenze wie bei a2a und aus demselben Grund: ein Paket. Nicht
        #: `min_bytes` (4096) -- die all_gather der Spekulations- und
        #: DP-Attention-Pfade sind klein, und dort ist die Latenz der
        #: gloo-Ebene am teuersten.
        self.ag_min_bytes = int(
            os.environ.get("SGLANG_HTCCL_BAR1_AG_MIN_BYTES", "16")
        )
        #: Wieviele Runden eine Scherbe hoechstens kosten darf. Keine
        #: Fenstergrenze, sondern eine Rundengrenze: je Runde ein
        #: Kernelstart mit einer Sperre. 16 traegt bei knapp 8 MiB Schlitz
        #: (96-MiB-Fenster, R=3) eine Scherbe von ~128 MiB und damit jede
        #: Groesse, die in diesem Modell vorkommt -- die groesste gemessene
        #: ist 10,6 MB. Darueber meldet sich der Weg ab, statt eine
        #: Schleife als Transport auszugeben.
        self.ag_max_runden = int(
            os.environ.get("SGLANG_HTCCL_BAR1_AG_MAX_RUNDEN", "16")
        )
        try:
            self._baue_auf()
        except BaseException:
            # Ein halb aufgebauter Transport bleibt nicht stehen: die schon
            # gebundenen Peers halten Abbildungen, Registrierungen und
            # Anhaftungen, und die ueberlebten sonst bis zum Prozessende --
            # samt der BAR1-Seiten, die sie belegen.
            try:
                self.close()
            except Exception:
                pass
            raise

    # -- Faehigkeit --------------------------------------------------------

    @staticmethod
    def patchstand() -> dict:
        """Was der Treiber ueber sich verraet -- ohne jede Deutung.

        ``RMSmallBarP2PPeerBar1`` weitet den Guard von "innerhalb des
        statischen Fensters einer anderen GPU" auf "innerhalb der
        BAR1-Apertur einer anderen GPU". Vorgabe ist **0**; ohne den
        Regkey verhaelt sich der Guard exakt wie bisher, und
        ``cudaHostRegister(..., IoMemory)`` auf eine fremde BAR scheitert.

        Gemeldet wird nur, was in ``/proc/driver/nvidia/params`` steht.
        Ein leerer ``RegistryDwords``-Eintrag heisst: der Regkey ist nicht
        gesetzt. Das ist kein Beweis fuer "Pfad tot" -- der Beweis ist der
        fehlgeschlagene ``cudaHostRegister``, und genau darauf wartet der
        Aufbau.
        """
        aus = {"regkeys": "", "treiber": "", "halter": os.path.exists(HALTER_PFAD)}
        try:
            with open("/proc/driver/nvidia/params") as f:
                for z in f:
                    # Genau "RegistryDwords:" -- nicht "RegistryDwordsPerDevice:",
                    # das sonst als spaetere Zeile die echte ueberschreibt und
                    # einen gesetzten Regkey als leer meldet.
                    if z.startswith("RegistryDwords:"):
                        aus["regkeys"] = z.strip()
        except OSError:
            pass
        try:
            with open("/proc/driver/nvidia/version") as f:
                aus["treiber"] = f.readline().strip()
        except OSError:
            pass
        aus["smallbar_p2p_peerbar1"] = "RMSmallBarP2PPeerBar1" in aus["regkeys"]
        return aus

    # -- Aufbau ------------------------------------------------------------

    def _baue_auf(self) -> None:
        import torch
        import torch.distributed as dist

        from sglang.srt.distributed.device_communicators.htccl_matrix import (
            bdf_der_karte,
        )

        if self.welt < 2:
            raise Bar1Unverfuegbar("weniger als zwei Raenge -- nichts zu tun")

        t0 = time.perf_counter()
        self._cuda = _Cuda()
        self._halter = Halter()

        eigener_bdf = bdf_der_karte(self.device)
        # BDF und Fenstervorschlag in EINEM all_gather. Der Vorschlag muss
        # mit, weil die Karten der Gruppe verschieden grosse Aperturen haben
        # (3080: 256 MiB brutto) und in einem Prozess mit zwei Gruppen
        # ausserdem verschieden viel davon schon vergeben ist. Eine je Rang
        # verschiedene Region waere eine je Rang verschiedene Schlitzordnung
        # -- also nicht ein Fehler, sondern Schreibvorgaenge an die falsche
        # Stelle. Deshalb: gruppenweites MINIMUM, und das entscheidet.
        gesammelt: list[object] = [None] * self.welt
        dist.all_gather_object(
            gesammelt, (eigener_bdf, int(self.fenster_bytes)),
            group=self.cpu_group,
        )
        self.bdfs = [str(x[0]) for x in gesammelt]      # type: ignore[index]
        vorschlaege = [int(x[1]) for x in gesammelt]    # type: ignore[index]
        gemeinsam = min(vorschlaege)
        if gemeinsam != self.fenster_bytes:
            logger.warning(
                "HTCCL-BAR1: Fenstervorschlaege je Rang %s MiB -- massgeblich "
                "ist das gruppenweite Minimum %d MiB. Dieser Rang haette "
                "%d MiB gekonnt. Die Region ist rangeinheitlich, weil die "
                "Schlitzversaetze in beiden Kernen aus ihr gerechnet werden.",
                [v // 2**20 for v in vorschlaege], gemeinsam // 2**20,
                self.fenster_bytes // 2**20,
            )
        if gemeinsam <= 0:
            raise Bar1Unverfuegbar(
                "gruppenweit sind 0 Byte BAR1-Fenster uebrig. Ein anderer "
                "Kommunikator dieses Prozesses hat die Apertur belegt; die "
                "Rechnung dazu steht in der Warnung von "
                "htccl_matrix_transport.fenster_fuer. Entweder der anderen "
                "Gruppe weniger geben (SGLANG_HTCCL_BAR1_FENSTER_MIB_<NAME>) "
                "oder diese Gruppe ausdruecklich ueber NCCL fahren lassen."
            )
        self.fenster_bytes = gemeinsam

        # 0. Die Kerne. Zuerst, weil ein fehlgeschlagener Bau billiger
        # abzubrechen ist als eine halb aufgebaute Peer-Tabelle.
        from sglang.srt.distributed.device_communicators import htccl_bar1_ext

        try:
            self._ext = htccl_bar1_ext.lade_kollektiv_ext(self.cpu_group)
        except Exception as e:
            raise Bar1Unverfuegbar(
                f"Die Kollektiv-Erweiterung liess sich nicht uebersetzen: {e}"
            ) from e

        # 0b. Der gepipelinete Kern, wenn er eingeschaltet ist. Ein
        # fehlgeschlagener Bau schaltet ihn ab, statt den ganzen Transport zu
        # verlieren -- netz und ring sind davon unberuehrt.
        if self.pipe_an:
            from sglang.srt.distributed.device_communicators import (
                htccl_bar1_pipe_ext,
            )

            try:
                self._pipe_ext = htccl_bar1_pipe_ext.lade_pipe_ext(self.cpu_group)
            except Exception as e:
                logger.warning(
                    "HTCCL-BAR1: die gepipelinete Erweiterung liess sich nicht "
                    "uebersetzen (%s). netz_pipe faellt aus; netz und ring "
                    "laufen unveraendert weiter.", e,
                )
                self.pipe_an = False
                self._pipe_ext = None

        # 1. Speicherordnung. Aus dem Fenster, das der Aufrufer bewilligt,
        # folgt die groesste Nutzlast -- nicht umgekehrt.
        if self.pipe_an and not (2 <= self.pipe_vorlauf <= self.pipe_t):
            raise Bar1Unverfuegbar(
                f"SGLANG_HTCCL_BAR1_PIPE_VORLAUF={self.pipe_vorlauf} passt "
                f"nicht zu T={self.pipe_t}: erlaubt ist 2 <= P <= T. P=1 "
                f"verklemmt (Senden und Verbrauch eines Chunks fielen in "
                f"dieselbe Schleifenrunde), P>T liesse den Zeitplan die "
                f"Schlitze ueberholen."
            )
        if self.pipe_an:
            logger.info(
                "HTCCL-BAR1-PIPE: Ringtiefe T=%d, Vorlauf P=%d -- ein Peer "
                "darf um %d Schleifenrunden zurueckliegen, bevor der Sender "
                "blockiert. Direkt-Modus %s, Ergebnisring L=%d.",
                self.pipe_t, self.pipe_vorlauf,
                self.pipe_t - self.pipe_vorlauf + 1,
                "an" if self.pipe_direkt else "aus", self.pipe_erg_ring,
            )
        if not self.pipe_an or not self.pipe_direkt:
            self.pipe_erg_ring = 0
        elif self.pipe_erg_ring < 2:
            raise Bar1Unverfuegbar(
                f"SGLANG_HTCCL_BAR1_PIPE_ERG_RING={self.pipe_erg_ring}: der "
                f"Direkt-Modus braucht mindestens zwei Ergebnispuffer. Mit "
                f"einem einzigen schriebe Runde n in genau den Puffer, den "
                f"der Aufrufer aus Runde n-1 noch in der Hand haelt -- ein "
                f"still ueberschriebener Ergebnistensor, also falsche Zahlen "
                f"ohne Absturz. Wer den Ring nicht will, schaltet mit "
                f"SGLANG_HTCCL_BAR1_PIPE_DIREKT=0 den Direkt-Modus ab."
            )
        max_bytes = max_nutzlast(self.welt, self.fenster_bytes, self.a2a_an,
                                 self.pipe_an, self.pipe_erg_ring)
        if max_bytes < self.min_bytes:
            raise Bar1Unverfuegbar(
                f"Fenster von {self.fenster_bytes // 1024} KiB traegt bei "
                f"{self.welt} Raengen nur {max_bytes} Byte Nutzlast, "
                f"Mindestgroesse ist {self.min_bytes}. 4(R-1) Schlitze zu je "
                f"ceil(N/R) muessen hineinpassen."
            )
        self._geo = geometrie(self.welt, max_bytes, self.a2a_an, self.pipe_an,
                              self.pipe_erg_ring)
        self._erg_lebt = [None] * max(0, self.pipe_erg_ring)
        self.max_bytes = max_bytes
        region = self._geo["region_bytes"]
        flaggen = flaggen_bedarf(self.welt, self.a2a_an, self.pipe_an)

        # 2. Zwei Empfangsregionen, zwei Exporte. Getrennt, weil die Sonde
        # sie getrennt vermessen hat.
        dptr, handle, groesse = self._cuda.vmm_alloc(self.ordinal, region)
        self._eigen = (dptr, handle, groesse)
        fptr, fhandle, fgroesse = self._cuda.vmm_alloc(self.ordinal, flaggen)
        self._eigen_flag = (fptr, fhandle, fgroesse)
        # Flaggen auf 0. Runden beginnen bei 1, also kann keine alte Marke
        # als gueltige Quittung durchgehen.
        self._cuda.memset_d8(fptr, 0, fgroesse)

        weg = ""
        for adr, hnd, gr in ((dptr, handle, groesse), (fptr, fhandle, fgroesse)):
            fd, halte, weg = self._cuda.dmabuf_fd(adr, hnd, gr, self.ordinal)
            self._dmabuf_fds.append(fd)
            self._halte_fds.extend(halte)

        # 3. fds tauschen -- beide in einer Nachricht.
        self._fremde_fds = _tausche_fds(
            self.cpu_group, self.rank, self.welt, self._dmabuf_fds
        )

        # 4.-6. je Peer anhaengen, Versatz aus der sg-Tabelle, mappen,
        # registrieren. Das passiert GENAU HIER und nie wieder.
        for peer in range(self.welt):
            if peer == self.rank:
                continue
            self._peers[peer] = self._binde_peer(peer, self._fremde_fds[peer])

        # 7. Was TATSAECHLICH abgebildet ist -- gruppenweit das Minimum.
        # Nicht die Bruttogroesse aus sysfs und nicht die angeforderte:
        # massgeblich ist die zusammenhaengende Laenge, die der Halter je
        # Peer wirklich gemeldet hat. Ein Rang, dessen kleinstes Fenster
        # kleiner ist, entscheidet fuer alle -- sonst antwortete `handles`
        # rangabhaengig und die SPMD-Annahme waere verletzt.
        lokal_min = min(z.nutz.laenge for z in self._peers.values())
        lokal_flag_min = min(z.flag.laenge for z in self._peers.values())
        traeger: list[object] = [None] * self.welt
        dist.all_gather_object(
            traeger, (lokal_min, lokal_flag_min), group=self.cpu_group
        )
        self._fenster_minimum = min(int(x[0]) for x in traeger)   # type: ignore[index]
        flag_minimum = min(int(x[1]) for x in traeger)            # type: ignore[index]
        if self._fenster_minimum < region:
            raise Bar1Unverfuegbar(
                f"Angefordert waren {region} Byte Nutzlastregion, abgebildet "
                f"sind gruppenweit hoechstens {self._fenster_minimum} Byte "
                f"zusammenhaengend. Kein stilles Verkleinern der Nutzlast: "
                f"die Schlitzversaetze stehen in beiden Kernen fest, und ein "
                f"Rang mit anderer Ordnung schriebe an die falsche Stelle."
            )
        if flag_minimum < flaggen:
            raise Bar1Unverfuegbar(
                f"Flaggenregion: {flaggen} Byte noetig, abgebildet gruppenweit "
                f"hoechstens {flag_minimum}."
            )

        # 8. Rundenzaehler und Meldewort. Beide LOKAL im VRAM -- sie werden
        # nie von einem Peer angefasst.
        self._runde_dev = torch.zeros(1, dtype=torch.int64, device=self.device)
        self._ctl_dev = torch.zeros(2, dtype=torch.int32, device=self.device)
        # Absoluter Chunkzaehler des Schiebefensters. Getrennt vom
        # Rundenzaehler, weil er um K je Aufruf waechst und nur von
        # netz_pipe fortgeschrieben wird -- er ist der Bezug, gegen den die
        # head/tail-Zeilen der Peers verglichen werden, und muss deshalb
        # ueber Aufrufe hinweg absolut bleiben. Rangeinheitlich, weil jeder
        # Rang dieselbe Folge von Aufrufen mit demselben K sieht.
        self._schritt_dev = torch.zeros(1, dtype=torch.int64, device=self.device)

        dist.barrier(group=self.cpu_group)
        self._auf = True
        # In die Kasse. Erst JETZT, weil erst jetzt feststeht, dass die
        # Apertur den Platz wirklich hergegeben hat -- eine Buchung vor dem
        # Halter waere eine Zusage auf Verdacht, und der ENOMEM der zweiten
        # Gruppe kaeme dann von einer Reservierung, die es gar nicht gibt.
        from sglang.srt.distributed.device_communicators import (
            htccl_matrix_transport as _kasse,
        )

        _kasse.kasse_eintragen(self.device, self.gruppe, region + flaggen)
        dauer = time.perf_counter() - t0
        logger.info(
            "HTCCL-BAR1: Aufbau in %.0f ms, %d Peer-Ziele, Region %.1f MiB je "
            "Rang (%s), Schlitz %d KiB, groesste Nutzlast %d KiB, Flaggen %d "
            "Byte, Export ueber %s. Ab hier wird im heissen Pfad nichts mehr "
            "gemappt.",
            dauer * 1000, len(self._peers), region / 2**20,
            f"{(6 if self.a2a_an else 4) * (self.welt - 1)} Schlitze"
            + (" (davon 2(R-1) fuer all_to_all)" if self.a2a_an
               else ", all_to_all abgeschaltet"),
            self._geo["chunk_max"] // 1024,
            max_bytes // 1024, flaggen, weg,
        )
        # Und die Kasse mit ausgeben. Ohne sie ist die naechste Gruppe, die
        # mit ENOMEM scheitert, wieder auf Raten angewiesen.
        logger.info(
            "HTCCL-BAR1: BAR1-Kasse dieser Karte nach Gruppe %r: %s.",
            self.gruppe or "<ohne Namen>",
            ", ".join(f"{g or '<ohne Namen>'}: {b / 2**20:.1f} MiB"
                      for g, b in _kasse.kasse_stand(self.device)),
        )

    def _binde_peer(self, peer: int, fremde_fds: list) -> PeerZiel:
        """Beide Regionen eines Peers anhaengen, abbilden, registrieren."""
        ziel_bdf = self.bdfs[peer]
        nutz = self._binde_region(peer, ziel_bdf, fremde_fds[0], "Nutzlast",
                                  self._geo["region_bytes"])
        try:
            flag = self._binde_region(peer, ziel_bdf, fremde_fds[1], "Flaggen",
                                      flaggen_bedarf(self.welt, self.a2a_an,
                                                     self.pipe_an))
        except BaseException:
            # Die schon gebundene Nutzlastregion wieder los: sie steht noch
            # in keinem PeerZiel und wuerde von close() nicht gefunden.
            self._loese_region(nutz)
            raise
        return PeerZiel(rang=peer, bdf=ziel_bdf, nutz=nutz, flag=flag)

    def _loese_region(self, a: Abbildung) -> None:
        if self._cuda is not None:
            self._cuda.unregister(a.reg_adresse)
        try:
            a.mmap_obj.close()              # type: ignore[attr-defined]
        except Exception:
            pass
        if self._halter is not None:
            self._halter.gib_frei(a.halter_handle)

    def _binde_region(self, peer: int, ziel_bdf: str, fremder_fd: int,
                      was: str, mindestens: int) -> Abbildung:
        assert self._cuda is not None and self._halter is not None
        fenster = bar1_fenster(ziel_bdf)

        # Angehaengt wird als DIESE Karte -- sie schreibt spaeter.
        handle_, sg, total = self._halter.halte(fremder_fd, self.bdfs[self.rank])

        treffer = [e for e in sg if fenster.basis <= e.dma_adresse < fenster.ende]
        if not treffer:
            self._halter.gib_frei(handle_)
            raise Bar1Unverfuegbar(
                f"Keine der {len(sg)} sg-Adressen von Rang {peer} liegt in "
                f"dessen BAR1 [{fenster.basis:#x}, {fenster.ende:#x}). "
                f"Erste Adresse {sg[0].dma_adresse:#x}. Das heisst entweder, "
                f"die IOMMU bildet nicht identisch ab (dann traegt der aus der "
                f"sg-Tabelle abgeleitete Versatz nicht und es braeuchte den "
                f"Musterscan), oder der Treiber hat gar nicht in BAR1 "
                f"abgebildet. Kein Raten -- die Kante faellt aus."
            )
        treffer.sort(key=lambda e: e.dma_adresse)
        start = treffer[0].dma_adresse
        # Zusammenhaengend? Nur der zusammenhaengende Anfang ist als ein
        # Stueck mapbar; der Rest waere ein zweites Fenster.
        laenge = 0
        erwartet = start
        for e in treffer:
            if e.dma_adresse != erwartet:
                break
            laenge += e.laenge
            erwartet += e.laenge
        versatz = start - fenster.basis
        if laenge < mindestens:
            self._halter.gib_frei(handle_)
            raise Bar1Unverfuegbar(
                f"{was}region von Rang {peer} ({ziel_bdf}): "
                f"{mindestens} Byte noetig, aber nur {laenge} Byte "
                f"ZUSAMMENHAENGEND in BAR1 abgebildet (aus {len(treffer)} "
                f"sg-Eintraegen ab {start:#x}). Das ist die Laenge, gegen die "
                f"geprueft wird -- nicht die Bruttogroesse aus sysfs "
                f"({fenster.groesse} Byte), von der RM sich selbst bedient."
            )

        seite = mmap.PAGESIZE
        m_versatz = (versatz // seite) * seite
        vorlauf = versatz - m_versatz
        m_laenge = laenge + vorlauf
        res = f"/sys/bus/pci/devices/{ziel_bdf}/resource1_wc"
        try:
            res_fd = os.open(res, os.O_RDWR | os.O_SYNC)
        except OSError as e:
            self._halter.gib_frei(handle_)
            raise Bar1Unverfuegbar(
                f"{res} nicht oeffenbar ({e}). Ohne Write-Combining-Apertur "
                f"gibt es keinen Direktpfad."
            ) from e
        try:
            # NUR den benoetigten Ausschnitt: ein mmap ueber ein 32-GiB-Fenster
            # scheitert mit EINVAL (gemessen an der 5090).
            abb = mmap.mmap(res_fd, m_laenge, mmap.MAP_SHARED,
                            mmap.PROT_READ | mmap.PROT_WRITE, offset=m_versatz)
        except (OSError, ValueError) as e:
            self._halter.gib_frei(handle_)
            raise Bar1Unverfuegbar(
                f"mmap({res}, laenge={m_laenge}, versatz={m_versatz:#x}) "
                f"fehlgeschlagen: {e}"
            ) from e
        finally:
            os.close(res_fd)

        host = ctypes.addressof(ctypes.c_char.from_buffer(abb)) + vorlauf
        try:
            self._cuda.register_io(host - vorlauf, m_laenge)
        except Bar1Unverfuegbar as e:
            abb.close()
            self._halter.gib_frei(handle_)
            ps = self.patchstand()
            if not ps["smallbar_p2p_peerbar1"]:
                grund = (
                    f"Das ist der erwartete Ausgang OHNE den geweiteten Guard: "
                    f"der Regkey RMSmallBarP2PPeerBar1 hat die Vorgabe 0 und "
                    f"muss ueber NVreg_RegistryDwords gesetzt werden. "
                    f"Gefunden: {ps['regkeys'] or '<leer>'}."
                )
            else:
                # Der Regkey steht. Dann ist der Guard NICHT mehr der
                # wahrscheinliche Ablehner -- und die Verwechslung war teuer.
                # Vor dem Bereichs-Guard sitzt in
                # osCreateOsDescriptorFromIoMemory eine zweite Huerde: der
                # Zweig _PEER_MAP_OVERRIDE_REQUIRED verlangt
                # peerMappingOverride ODER osIsAdministrator(), und
                # osIsAdministrator() ist auf Linux capable(CAP_SYS_ADMIN)
                # (os-interface.c:380 -> nv-linux.h:499). Ein Docker-Container
                # laeuft als root, hat CAP_SYS_ADMIN aber NICHT in der
                # Vorgabemenge -- der Aufruf scheitert dann als
                # NV_ERR_INSUFFICIENT_PERMISSIONS, was hier als cudaError 800
                # (cudaErrorNotPermitted) ankommt.
                grund = (
                    f"Der Regkey steht ({ps['regkeys']}), der Bereichs-Guard "
                    f"ist es also nicht. Naechster Verdacht in dieser "
                    f"Reihenfolge: (1) CAP_SYS_ADMIN fehlt dem Prozess -- "
                    f"osCreateOsDescriptorFromIoMemory verlangt im Zweig "
                    f"_PEER_MAP_OVERRIDE_REQUIRED entweder "
                    f"peerMappingOverride oder osIsAdministrator(), und das "
                    f"ist capable(CAP_SYS_ADMIN); im Container also "
                    f"'--cap-add SYS_ADMIN' bzw. NVreg_RegistryDwords um "
                    f"PeerMappingOverride=1 ergaenzen. (2) der Bereich liegt "
                    f"nicht vollstaendig in der BAR1-Apertur des Peers. "
                    f"Welcher von beiden es war, sagt das Kernellog "
                    f"eindeutig: 'permission denied, allowPeermapping=0' "
                    f"gegen 'SMALLBAR_P2P: DENY ...'."
                )
            raise Bar1Unverfuegbar(
                f"cudaHostRegister(IoMemory) auf die {was}region von Rang "
                f"{peer} ({ziel_bdf}) fehlgeschlagen: {e}. {grund} "
                f"Der Transport meldet sich ab, er erzwingt nichts."
            ) from e
        dev = self._cuda.dev_ptr(host - vorlauf)
        return Abbildung(
            bar1_basis=fenster.basis, bar1_versatz=versatz, laenge=laenge,
            mmap_obj=abb, reg_adresse=host - vorlauf, host_adresse=host,
            dev_ptr=dev + vorlauf, halter_handle=handle_,
        )

    # -- Fensterrechnung ---------------------------------------------------

    def pruefe_fensterbedarf(self, algorithmus: str, nbytes: int) -> None:
        """Bedarf gegen das, was sich TATSAECHLICH exportieren liess.

        Nicht gegen die Bruttogroesse aus sysfs: die 3080er melden 256 MiB
        BAR1 brutto, wieviel davon netto fuer Peer-Abbildungen frei ist,
        ist ungemessen -- RM belegt Teile selbst. Massgeblich ist die
        Laenge, die der Halter je Peer wirklich zusammenhaengend abgebildet
        hat.
        """
        noetig = fensterbedarf(algorithmus, nbytes, self.welt)
        for peer, z in sorted(self._peers.items()):
            brutto = bar1_fenster(z.bdf).groesse
            if noetig > z.nutz.laenge:
                raise Bar1Unverfuegbar(
                    f"Fenster zu klein fuer '{algorithmus}' bei "
                    f"{nbytes // 1024} KiB und {self.welt} Raengen: noetig "
                    f"{noetig // 1024} KiB, abgebildet bei Rang {peer} "
                    f"({z.bdf}) aber nur {z.nutz.laenge // 1024} KiB "
                    f"(BAR1 brutto {brutto // 2**20} MiB). Entweder kleiner "
                    f"chunken oder diese Kante ausschliessen. Netz und Ring "
                    f"brauchen BEIDE 2(R-1) Schlitze -- der Ring ist hier "
                    f"kein Ausweg."
                )

    def fenster_minimum(self) -> int:
        """Kleinste tatsaechlich abgebildete Nutzlastregion in der GRUPPE.

        Das ist die Zahl, die als ``fenster_bytes`` in den Planer gehoert:
        eine **Faehigkeit**, ermittelt aus dem, was der Halter je Peer
        wirklich zusammenhaengend abgebildet hat, minimiert ueber alle
        Raenge. Ein je Rang verschiedener Wert ergaebe je Rang einen
        anderen Plan, und daran haengt die SPMD-Annahme der Kollektive.
        """
        return self._fenster_minimum

    def algorithmus_fuer(self, nbytes: int) -> str:
        """``netz``, ``netz_pipe`` oder ``ring`` fuer diese Groesse.

        Vorrang hat der Plan aus ``htccl_matrix.py``, wenn einer
        hereingereicht wurde (``setze_plan``). Er ist gruppenweit
        identisch -- geprueft ueber die Planpruefsumme -- und damit die
        einzige Quelle, die diese Wahl rangeinheitlich halten kann.

        Ohne Plan die Notschwelle ``SGLANG_HTCCL_BAR1_RING_AB``. Sie ist
        eine Vorgabe und keine Messaussage: zwischen 1 und 16 MiB liegen
        Netz und Ring in der Sonde 1 bis 7 Prozent auseinander, und der
        Befund dazu sagt ausdruecklich "keine saubere Schwelle -- der
        Planer sollte das messen, nicht einbauen".

        **DIE EINE STELLE, an der netz_pipe gewaehlt wird.** Der Planer
        kennt ``netz_pipe`` nicht und soll ihn vorerst auch nicht kennen:
        seine Kostenmodelle sind an den zwei gemessenen Topologien geeicht.
        Solange ``SGLANG_HTCCL_BAR1_PIPE`` aus ist -- und das ist die
        Vorgabe --, gibt diese Methode Buchstabe fuer Buchstabe dieselbe
        Antwort wie bisher.
        """
        if self._plan is not None:
            a = self._plan.algorithmus_fuer(nbytes)
            # 'stern' und 'hierarchisch' sind hier nicht portiert; sie
            # kommen ueber handles() gar nicht bis hierher.
        else:
            a = "ring" if nbytes >= self.ring_ab else "netz"
        if (self.pipe_an and a == "netz" and nbytes >= self.pipe_ab
                and self._pipe_k(nbytes) is not None):
            return "netz_pipe"
        return a

    def setze_plan(self, plan) -> None:
        """Den Plan des Matrix-Planers uebernehmen.

        Nur einmal und nur vor dem ersten Kollektiv: die Wahl muss zum
        Aufzeichnungszeitpunkt eines CUDA-Graphen feststehen.
        """
        self._plan = plan

    # -- Byte-Beleg --------------------------------------------------------

    def byte_beleg_alle(self, probe_bytes: int = 65536) -> dict[tuple[int, int], bool]:
        """Je gerichtetem Paar: Muster hinein, auf dem Ziel zurueckgelesen.

        Die Ruecklesung laeuft ueber den **eigenen VMM-Zeiger der
        Zielkarte**, nicht durch die Apertur -- sonst verdeckt ein defekter
        Pfad seinen eigenen Fehler. Genau daran ist auf diesem Rig der
        Mailbox-Weg aufgefallen: der Treiber meldete Peer-Zugriff, und von
        1 MiB kamen 4096 Byte an.
        """
        import torch
        import torch.distributed as dist

        assert self._cuda is not None
        n = min(probe_bytes, self._eigen[2])
        ergebnis: dict[tuple[int, int], bool] = {}
        rueck = torch.empty(n, dtype=torch.uint8, pin_memory=True)
        for quelle in range(self.welt):
            for ziel in range(self.welt):
                if quelle == ziel:
                    continue
                marke = (quelle * 251 + ziel * 37 + 1) & 0xFF
                dist.barrier(group=self.cpu_group)
                if self.rank == ziel:
                    # Ziel zuerst loeschen, damit ein NICHT geschriebener
                    # Puffer nicht zufaellig wie ein Treffer aussieht.
                    leer = torch.full((n,), (marke ^ 0xFF) & 0xFF,
                                      dtype=torch.uint8, pin_memory=True)
                    self._cuda.memcpy(self._eigen[0], leer.data_ptr(), n)
                dist.barrier(group=self.cpu_group)
                if self.rank == quelle:
                    muster = torch.full((n,), marke, dtype=torch.uint8,
                                        device=self.device)
                    self.put(ziel, muster.data_ptr(), n, 0)
                    torch.cuda.synchronize(self.device)
                dist.barrier(group=self.cpu_group)
                ok = True
                if self.rank == ziel:
                    self._cuda.memcpy(rueck.data_ptr(), self._eigen[0], n)
                    schlecht = int((rueck != marke).sum().item())
                    ok = schlecht == 0
                    if ok:
                        # Auch der bestandene Beleg gehoert ins Protokoll:
                        # "0 von N Byte falsch" ist die Aussage, auf der jede
                        # spaetere Zeitmessung steht.
                        logger.info(
                            "HTCCL-BAR1: Byte-Beleg %d->%d bestanden: 0 von "
                            "%d Byte falsch.", quelle, ziel, n,
                        )
                    else:
                        logger.warning(
                            "HTCCL-BAR1: Byte-Beleg %d->%d GEFALLEN: %d von %d "
                            "Byte falsch. Diese Kante wird gestrichen, "
                            "unabhaengig davon, was der Treiber meldet.",
                            quelle, ziel, schlecht, n,
                        )
                traeger: list[object] = [ok if self.rank == ziel else None]
                dist.broadcast_object_list(
                    traeger, src=dist.get_global_rank(self.cpu_group, ziel),
                    group=self.cpu_group,
                )
                ergebnis[(quelle, ziel)] = bool(traeger[0])
        for (q, z), ok in ergebnis.items():
            if q == self.rank and z in self._peers:
                self._peers[z].byte_beleg = ok
        # Gruppenweit EINE Antwort. `ergebnis` ist auf jedem Rang identisch
        # (jeder Eintrag wurde vom Ziel aus verteilt), also ist auch dieses
        # Und rangeinheitlich -- genau das braucht `handles`.
        self._belege_stehen = all(ergebnis.values())
        if not self._belege_stehen:
            gefallen = sorted(k for k, v in ergebnis.items() if not v)
            logger.warning(
                "HTCCL-BAR1: Byte-Beleg gefallen fuer %s. Die Kollektive "
                "melden sich ab (handles -> False); ein Kollektiv ueber eine "
                "Kante, die Bytes verliert, waere kein Kollektiv.", gefallen,
            )
        return ergebnis

    # -- Datenweg ----------------------------------------------------------

    def put(self, ziel: int, quell_ptr: int, nbytes: int, versatz: int = 0,
            stream: Optional[int] = None) -> None:
        """Ein Schreibvorgang in die BAR der Zielkarte. Posted, also schnell.

        Es gibt bewusst **kein** ``get``: Lesen aus einer fremden BAR ist
        non-posted und teuer (an der 2080 Ti gemessen 1132 MB/s heraus
        gegen 3254 MB/s hinein). Deshalb die Regel "jeder schiebt selbst".
        """
        if not self._auf:
            raise Bar1Unverfuegbar("Transport nicht aufgebaut")
        z = self._peers.get(ziel)
        if z is None:
            raise Bar1Unverfuegbar(f"kein Peer-Ziel fuer Rang {ziel}")
        if versatz + nbytes > z.laenge:
            raise Bar1Unverfuegbar(
                f"put({ziel}): {versatz}+{nbytes} ueberschreitet das "
                f"abgebildete Fenster von {z.laenge} Byte. Der Aufrufer muss "
                f"chunken; ein automatisches Nachmappen im heissen Pfad ist "
                f"ausgeschlossen -- es ist genau der teure Teil."
            )
        assert self._cuda is not None
        if stream is None:
            import torch

            stream = torch.cuda.current_stream(self.device).cuda_stream
        self._cuda.memcpy_async(z.dev_ptr + versatz, quell_ptr, nbytes, stream)

    # -- Messfuehler fuer htccl_matrix -------------------------------------

    def name(self) -> str:
        return "bar1"

    def eigenlast(self, nbytes: int, richtung: str) -> float:
        from sglang.srt.distributed.device_communicators.htccl_matrix import (
            EigenlastFuehler,
        )

        if getattr(self, "_eigen_fuehler", None) is None:
            self._eigen_fuehler = EigenlastFuehler(self.device, max_bytes=4 << 20)
        return self._eigen_fuehler.eigenlast(nbytes, richtung)

    def eigenlast_duplex(self, nbytes: int) -> Optional[float]:
        """Bewusst ``None``.

        Vollduplex ueber den Direktpfad ist NICHT ueber Host-Speicher
        messbar, und der gelockerte Treiber-Guard existiert wegen eines
        dokumentierten Vollduplex-Deadlocks (Bug 1571948). Solange
        Gegenverkehr ueber die volle Kollektivdauer nicht geprueft ist,
        wird hier nichts gemeldet, was wie eine Freigabe aussieht.
        """
        return None

    def paar(self, ziel: int, nbytes: int) -> Optional[float]:
        """Gerichtete Kante GB/s -- nur wenn der Byte-Beleg steht."""
        import torch

        if not self._auf:
            return None
        if ziel < 0:                       # Faehigkeitsanfrage des Planers
            return 0.0 if self._peers else None
        z = self._peers.get(ziel)
        if z is None or not z.byte_beleg:
            return None
        n = min(nbytes, z.laenge)
        quelle = torch.empty(n, dtype=torch.uint8, device=self.device)
        for _ in range(8):
            self.put(ziel, quelle.data_ptr(), n, 0)
        torch.cuda.synchronize(self.device)
        runden = 64 if n <= 65536 else 16
        t0 = time.perf_counter()
        for _ in range(runden):
            self.put(ziel, quelle.data_ptr(), n, 0)
        torch.cuda.synchronize(self.device)
        dt = time.perf_counter() - t0
        return (runden * n) / dt / 1e9 if dt > 0 else 0.0

    def paar_empfang(self, quelle: int, nbytes: int) -> None:
        """Nichts zu tun -- die Schreibvorgaenge sind einseitig.

        Die Zielkarte ist am Transfer nicht beteiligt; ihre BAR nimmt die
        Bytes ohne Zutun entgegen. Genau das ist der Grund, warum der Weg
        so wenig kostet.
        """
        return None

    # -- Transport-Naht ----------------------------------------------------

    def handles(self, op: str, nbytes: int) -> bool:
        """``True`` nur, wenn der Weg diese Operation wirklich fahren kann.

        Jede Bedingung ist **rangeinheitlich**: sie haengt nur an
        gruppenweit abgeglichenen Groessen (``_belege_stehen`` kommt aus
        einer Verteilung je gerichtetem Paar, ``_fenster_minimum`` und
        ``max_bytes`` aus einem ``all_gather``, die Schwellen aus
        rangeinheitlichen Umgebungsvariablen). Zwei Raenge duerfen hier nie
        verschieden antworten -- der eine liefe ins Kollektiv, der andere
        nicht, und das Ergebnis waere ein Haenger statt eines Fehlers.

        Der Datentyp geht NICHT ein: ``handles`` sieht ihn nicht. Die
        Erweiterung nimmt float32/float16/bfloat16 und lehnt alles andere
        mit Begruendung ab. Das ist derselbe Stand wie bei
        ``htccl_device``.
        """
        if op not in self.HTCCL_OPS:
            return False
        if not self._auf or self._ext is None:
            return False
        if not self._belege_stehen:
            return False
        if op in ("all_to_all", "all_to_all_single"):
            return self._handles_a2a(nbytes)
        if op == "all_gather":
            return self._handles_all_gather(nbytes)
        if nbytes < self.min_bytes or nbytes > self.max_bytes:
            return False
        if nbytes % 16 != 0:
            # Die Zugriffsbreite ist 128 Bit; ein Rest waere ein zweiter,
            # ungemessener Pfad im Kernel.
            return False
        if nbytes // 16 < self.welt:
            # Weniger als ein Paket je Rang -- die Chunkzerlegung liesse
            # Raenge leer ausgehen.
            return False
        # Passt der GROESSTE Chunk in einen Schlitz? Das ist die Bedingung,
        # an der die Abbildung wirklich haengt -- geprueft, nicht aus
        # `nbytes <= max_bytes` gefolgert. Die Erweiterung rechnet sie ein
        # zweites Mal, dort aber mit chunkGrenzen selbst statt mit dieser
        # Formel: eine Naht, die auf beiden Seiten mit derselben falschen
        # Formel geprueft wuerde, faellt nicht auf.
        groesster_chunk = -(-(nbytes // 16) // self.welt) * 16
        if groesster_chunk > self._geo["chunk_max"]:
            return False
        algo = self.algorithmus_fuer(nbytes)
        if algo not in ("netz", "netz_pipe", "ring"):
            # 'stern' und 'hierarchisch' sind hier nicht portiert. Kein
            # stilles Ausweichen auf 'netz'.
            return False
        if algo == "netz_pipe" and not self._pipe_traegt(nbytes):
            return False
        # Und derselbe Bedarf noch einmal in der Waehrung des Planers, gegen
        # die gruppenweit KLEINSTE tatsaechlich abgebildete Laenge. Redundant,
        # solange der Aufbau durchgelaufen ist -- und genau deshalb billig:
        # die Zeile faellt auf, wenn jemand die Regionsgroesse anfasst, ohne
        # den Fensterbegriff mitzuziehen.
        if fensterbedarf(algo, nbytes, self.welt) > self._fenster_minimum:
            return False
        return True

    def _kern(self, bewegt: int, schwelle: int, wo: str) -> int:
        """``1`` = cooperative Mehrblockstart (``gitter``), ``0`` = ``1blk``.

        **Die eine Stelle, an der diese Wahl faellt** -- vorher rechnete jedes
        der drei Kollektive `bewegt >= schwelle` selbst nach, und eine
        Aufzeichnungsregel haette man an drei Stellen einbauen und an einer
        vergessen koennen.

        Zwei Eingaben, beide rangeinheitlich: die Groesse (gruppenweit gleich)
        und die Schwelle (Umgebungsvariable). Dazu, wenn gerade aufgezeichnet
        wird, der Vorbehalt gegen den cooperative Start.

        Zur Rangeinheitlichkeit der Aufzeichnung: sie ist es, weil der
        Graph-Laeufer auf allen Raengen dieselben Formen in derselben
        Reihenfolge aufzeichnet. Waere sie es nicht, haetten wir bereits ein
        groesseres Problem als die Kernvariante -- ein Rang im Kollektiv, der
        andere nicht, also ein Haenger.
        """
        if bewegt < schwelle:
            return 0
        if self.graph_gitter:
            return 1
        from sglang.srt.distributed.device_communicators.htccl import (
            graph_erfassung_laeuft,
        )

        if not graph_erfassung_laeuft():
            return 1
        if not self._graph_gitter_gemeldet:
            self._graph_gitter_gemeldet = True
            logger.warning(
                "HTCCL-BAR1: %s mit %d Byte laege ueber der gitter-Schwelle "
                "(%d Byte), wird aber waehrend einer CUDA-Graph-Aufzeichnung "
                "auf die 1blk-Variante gelegt. Ob cudaLaunchCooperativeKernel "
                "sich aufzeichnen laesst, ist auf diesem Rig NICHT gemessen; "
                "benchmark/bar1_graph_check.py beantwortet es. Faellt der "
                "Beleg zugunsten von gitter aus, hebt "
                "SGLANG_HTCCL_BAR1_GRAPH_GITTER=1 diesen Vorbehalt auf. "
                "Dieser Hinweis erscheint einmal je Rang.",
                wo, bewegt, schwelle,
            )
        return 0

    def htccl_all_reduce(self, comm, inp):
        """Summen-Allreduce ueber ``netz`` oder ``ring``, ausser Ort.

        Ausser Ort ist keine Bequemlichkeit: der Ring liest ``in`` noch,
        waehrend er ``out`` bereits fortschreibt (Schritt s+1 sendet die in
        Schritt s gebildete Teilsumme). Die Erweiterung prueft das und
        lehnt gleiche Zeiger ab.
        """
        import torch

        if not self._auf or self._ext is None:
            raise Bar1Unverfuegbar(
                "htccl_all_reduce ohne aufgebauten Transport -- erreichbar "
                "nur, wenn jemand handles() umgangen hat."
            )
        inp = inp.contiguous()
        nbytes = inp.numel() * inp.element_size()
        algo = self.algorithmus_fuer(nbytes)
        if algo == "netz_pipe":
            k = self._pipe_k(nbytes)
            if k is None:
                raise Bar1Unverfuegbar(
                    "netz_pipe ohne passende Chunkzahl -- erreichbar nur, "
                    "wenn jemand handles() umgangen hat."
                )
            return self._pipe_all_reduce(inp, k)
        out = torch.empty_like(inp)
        # 'gitter' ist der cooperative Mehrblockstart. Die Schwelle ist
        # gemessen (ab 4 MiB gewinnt er), aber sie ist eine Zahl aus EINEM
        # Rig -- deshalb steht sie in einer Umgebungsvariablen. Unter
        # Graph-Aufzeichnung entscheidet zusaetzlich `_kern`.
        kern = self._kern(nbytes, self.gitter_ab, "all_reduce")
        peer_nutz = [0] * self.welt
        peer_flag = [0] * self.welt
        for r, z in self._peers.items():
            peer_nutz[r] = z.nutz.dev_ptr
            peer_flag[r] = z.flag.dev_ptr
        peer_nutz[self.rank] = self._eigen[0]
        peer_flag[self.rank] = self._eigen_flag[0]
        self._ext.bar1_all_reduce(
            inp, out, int(self.rank), int(self.welt),
            0 if algo == "netz" else 1,
            peer_nutz, peer_flag,
            int(self._eigen[0]), int(self._eigen_flag[0]),
            int(self._geo["chunk_max"]), int(self._geo["off_netz"]),
            int(self._geo["off_ring"]),
            self._runde_dev, self._ctl_dev,
            int(self.deckel_zyklen), int(self.threads), int(kern),
            int(self.ladeform), int(self.fluss),
        )
        return out

    # -- netz_pipe ---------------------------------------------------------
    #
    # Alles, was ueber die Wahl hinausgeht, steht in htccl_bar1_pipe_ext:
    # der Kern, die Schlitz- und Zaehlergeometrie, die Chunkplanung und der
    # Byte-Beleg. Hier stehen nur die drei Zeilen, mit denen der Transport
    # sie einsetzt.

    def _pipe_k(self, nbytes: int):
        """Chunkzahl fuer diese Nutzlast, oder ``None``.

        Rangeinheitlich: jeder Eingang ist gruppenweit gleich. ``None``
        heisst "traegt der gepipelinete Weg nicht" und fuehrt in
        ``handles`` zu False -- nicht zu einem stillen Ausweichen.

        **Gemerkt je Groesse.** ``pipe_plan`` rechnet die Zerlegung ueber
        alle (Chunk, Rang)-Paare durch -- absichtlich, statt mit einer
        geschlossenen Zweitfassung --, und das gehoert nicht in den heissen
        Pfad. ``handles`` und ``htccl_all_reduce`` fragen je Kollektiv; die
        Groessen wiederholen sich.
        """
        if not self.pipe_an or self._pipe_ext is None:
            return None
        if self._geo.get("off_pipe", -1) < 0:
            return None
        merk = getattr(self, "_pipe_k_merk", None)
        if merk is None:
            merk = {}
            self._pipe_k_merk = merk
        if nbytes in merk:
            return merk[nbytes]
        from sglang.srt.distributed.device_communicators.htccl_bar1_pipe_ext import (
            pipe_plan,
        )

        k = pipe_plan(
            int(nbytes), int(self.welt), int(self._geo["chunk_max"]),
            int(self.pipe_t), int(self.pipe_k), int(self.pipe_chunk_bytes),
            int(self.pipe_k_max),
        )
        merk[nbytes] = k
        return k

    def _pipe_traegt(self, nbytes: int) -> bool:
        """Fenstergrenze fuer ``netz_pipe`` -- gerechnet, nicht angenommen.

        Der Bedarf ist ``2 T (R-1)`` Schlitze zu je ``chunk_max/T``,
        gerechnet in ``pipe_fensterbedarf``. Geprueft wird gegen die
        gruppenweit KLEINSTE **tatsaechlich abgebildete** Laenge
        (``_fenster_minimum``), nicht gegen die Bruttogroesse aus sysfs und
        nicht gegen die angeforderte Region: massgeblich ist, was der
        Halter je Peer wirklich zusammenhaengend in BAR1 vorgefunden hat.
        Passt es nicht, meldet sich dieser Weg ueber ``handles`` ab.
        """
        if not self._pipe_beleg:
            return False
        from sglang.srt.distributed.device_communicators.htccl_bar1_pipe_ext import (
            erg_ring_bytes,
            pipe_fensterbedarf,
        )

        off = int(self._geo.get("off_pipe", -1))
        if off < 0:
            return False
        noetig = off + pipe_fensterbedarf(
            self.welt, int(self._geo["chunk_max"]), int(self.pipe_t)
        )
        # Und der Ergebnisring obendrauf: L * roundup(max_bytes, SEITE). Er
        # liegt HINTER den Schlitzen, also ist der Bedarf der Versatz des
        # Rings plus seine Laenge -- nicht das Maximum von beidem.
        if int(self._geo.get("off_erg", -1)) >= 0:
            noetig = max(noetig, int(self._geo["off_erg"]) + erg_ring_bytes(
                int(self.max_bytes), int(self._geo["erg_ring"])
            ))
        return noetig <= self._fenster_minimum

    def _erg_platz(self, inp):
        """Naechster Ergebnispuffer im Ring, als Tensor -- oder ``None``.

        ``None`` heisst "kein Direkt-Modus"; der Aufrufer nimmt dann
        ``torch.empty_like``. Sichtbar, nicht still.

        **Die Lebensdauerpruefung.** Der Ring vergibt Platz ``i`` reihum.
        Bevor Platz ``i`` neu beschrieben wird, muss der Tensor, den dieser
        Platz vor ``L`` Runden herausgegeben hat, tot sein. Geprueft wird
        das mit einem schwachen Verweis: lebt er noch, haelt ihn jemand, und
        ihn zu ueberschreiben hiesse, dem Aufrufer unter der Hand andere
        Zahlen in einen Tensor zu schreiben, den er fuer fertig haelt --
        falsche Ergebnisse ohne Absturz, der schlimmste denkbare Fehler
        dieses Vorhabens. Deshalb bricht es hier mit Grund ab und weicht
        nicht still aus.

        **Was diese Pruefung NICHT abdeckt** und was deshalb hier steht: sie
        sieht Python-Verweise, nicht laufende Kernel. Solange Ergebnis und
        Folgeschicht auf DEMSELBEN Strom liegen -- und das tun sie in
        sglang --, ordnet der Strom die Zugriffe. Wer den Ergebnistensor auf
        einem anderen Strom weiterverarbeitet, muss selbst synchronisieren;
        der Direkt-Modus gehoert dann abgeschaltet
        (``SGLANG_HTCCL_BAR1_PIPE_DIREKT=0``).
        """
        import weakref

        if not self.pipe_direkt or self._geo.get("off_erg", -1) < 0:
            return None
        # -- Aufzeichnung -------------------------------------------------
        #
        # Unter Stream-Capture ist der Direkt-Modus AUS. Das ist keine
        # Vorsicht, sondern eine Herleitung; sie steht hier ausgeschrieben,
        # weil sie sonst nirgends nachlesbar waere.
        #
        # Diese Methode ist HOSTCODE. Sie laeuft beim Aufzeichnen genau
        # einmal und bei keiner Wiedergabe wieder. Der gewaehlte Ringplatz
        # `_erg_i`, der daraus gerechnete Zeiger und die `peer_erg`-Tabelle
        # des Kerns werden in den Graphen eingebrannt. Drei Folgen, und die
        # dritte ist die gefaehrliche:
        #
        # 1. Der Ring entartet je Graph auf EINEN Platz. Fuer sich genommen
        #    ist das nur graphueblich (ein Graph hat feste Ausgabepuffer).
        # 2. Die Lebensdauerpruefung unten -- der schwache Verweis -- greift
        #    bei Wiedergabe NICHT MEHR, weil kein Hostcode laeuft. Wer den
        #    Ergebnistensor ueber eine Wiedergabe hinaus haelt, bekommt ihn
        #    unter der Hand ueberschrieben, und zwar ohne den Abbruch, den
        #    diese Methode im eager-Betrieb ausloest. Genau der Fehlerfall,
        #    den die Pruefung verhindern soll, kaeme durch die Aufzeichnung
        #    zurueck.
        # 3. Und das ist der stille: bei MEHREREN aufgezeichneten Graphen
        #    (sglang zeichnet je Stapelgroesse einen auf) laeuft `_erg_i`
        #    ueber die Ringplaetze weiter. Mit der Vorgabe `erg_ring = 2`
        #    ist beim dritten Graphen Platz 0 wieder an der Reihe. Haelt der
        #    Graph-Laeufer den Ausgabetensor des ersten Graphen noch -- was
        #    er tut --, bricht es hier ab, mitten in der Aufzeichnung. Haelt
        #    er ihn NICHT mehr, teilen sich zwei aufgezeichnete Graphen
        #    denselben BAR1-Platz, und wer sie abwechselnd wiedergibt,
        #    bekommt vom einen die Zahlen des anderen. Kein Absturz.
        #
        # Der Ausweg ist billig: ohne Direkt-Modus liefert `_pipe_all_reduce`
        # einen `torch.empty_like`, der waehrend der Aufzeichnung aus dem
        # privaten Speicherbecken des Graphen kommt und damit ohnehin eine
        # feste Adresse hat -- der Kern faehrt seinen `direkt=0`-Weg, der
        # derselbe gemessene Kontrollpfad ist. Es kostet den gesparten
        # VRAM-Durchgang, nicht die Richtigkeit.
        from sglang.srt.distributed.device_communicators.htccl import (
            graph_erfassung_laeuft,
        )

        if graph_erfassung_laeuft() and not self.pipe_direkt_graph:
            if not self._direkt_graph_gemeldet:
                self._direkt_graph_gemeldet = True
                logger.warning(
                    "HTCCL-BAR1-PIPE: Direkt-Modus waehrend einer "
                    "CUDA-Graph-Aufzeichnung abgeschaltet -- der Ringplatz "
                    "wuerde eingebrannt und die Lebensdauerpruefung liefe bei "
                    "keiner Wiedergabe mehr. netz_pipe faehrt aufgezeichnet "
                    "den direkt=0-Weg. SGLANG_HTCCL_BAR1_PIPE_DIREKT_GRAPH=1 "
                    "hebt das auf; wer das setzt, muss den Ergebnistensor "
                    "innerhalb derselben Wiedergabe verbrauchen und darf "
                    "hoechstens SGLANG_HTCCL_BAR1_PIPE_ERG_RING Graphen "
                    "aufzeichnen. Dieser Hinweis erscheint einmal je Rang."
                )
            return None
        ring = int(self._geo["erg_ring"])
        i = (self._erg_i + 1) % ring
        alt = self._erg_lebt[i]
        if alt is not None and alt() is not None:
            raise Bar1Unverfuegbar(
                f"Direkt-Modus: der Ergebnispuffer {i} von vor {ring} Runden "
                f"wird noch gehalten. Ihn jetzt zu beschreiben, hiesse einen "
                f"Tensor zu veraendern, den der Aufrufer fuer fertig haelt. "
                f"Entweder den Ring vergroessern "
                f"(SGLANG_HTCCL_BAR1_PIPE_ERG_RING) oder den Direkt-Modus "
                f"abschalten (SGLANG_HTCCL_BAR1_PIPE_DIREKT=0). Kein stilles "
                f"Ausweichen: ein ueberschriebenes Ergebnis faellt sonst "
                f"nirgends auf."
            )
        ptr = (self._eigen[0] + int(self._geo["off_erg"])
               + i * int(self._geo["erg_stride"]))
        out = self._pipe_ext.bar1_erg_tensor(int(ptr), inp)
        self._erg_lebt[i] = weakref.ref(out)
        self._erg_i = i
        return out

    def _pipe_all_reduce(self, inp, k: int):
        """Ein Aufruf des gepipelineten Kerns. Ausser Ort, wie netz/ring."""
        import torch

        inp = inp.contiguous()
        nbytes = inp.numel() * inp.element_size()
        out = self._erg_platz(inp)
        direkt = out is not None
        if not direkt:
            out = torch.empty_like(inp)
        kern = self._kern(nbytes, self.pipe_gitter_ab, "netz_pipe")
        peer_nutz = [0] * self.welt
        peer_flag = [0] * self.welt
        peer_erg = [0] * self.welt
        for r, z in self._peers.items():
            peer_nutz[r] = z.nutz.dev_ptr
            peer_flag[r] = z.flag.dev_ptr
        peer_nutz[self.rank] = self._eigen[0]
        peer_flag[self.rank] = self._eigen_flag[0]
        if direkt:
            # DERSELBE Ringplatz bei jedem Rang. Das ist keine Annahme ueber
            # den Nachbarn, sondern dieselbe SPMD-Voraussetzung, auf der
            # jedes Kollektiv dieses Moduls steht: alle Raenge sehen
            # dieselbe Folge von Aufrufen. Der Kern prueft ausserdem, dass
            # der eigene Eintrag wirklich `out` ist.
            versatz = (int(self._geo["off_erg"])
                       + self._erg_i * int(self._geo["erg_stride"]))
            for r in range(self.welt):
                peer_erg[r] = peer_nutz[r] + versatz
        from sglang.srt.distributed.device_communicators.htccl_bar1_pipe_ext import (
            pipe_fbasis,
            pipe_schlitz_bytes,
        )

        self._pipe_ext.bar1_netz_pipe(
            inp, out, int(self.rank), int(self.welt),
            peer_nutz, peer_flag, peer_erg,
            int(self._eigen[0]), int(self._eigen_flag[0]),
            int(pipe_schlitz_bytes(int(self._geo["chunk_max"]), int(self.pipe_t))),
            int(self._geo["off_pipe"]),
            int(pipe_fbasis(self.welt, self.a2a_an)),
            int(k), int(self.pipe_t), int(self.pipe_vorlauf),
            int(self.pipe_quittung), 1 if direkt else 0,
            self._runde_dev, self._schritt_dev, self._ctl_dev,
            int(self.deckel_zyklen), int(self.threads), int(kern),
            int(self.ladeform),
        )
        return out

    def byte_beleg_pipe(self, runden: int = 0) -> bool:
        """Byte-Beleg fuer ``netz_pipe``. Ohne ihn meldet sich der Weg ab.

        Getrennt von ``byte_beleg_alle``, weil er etwas anderes prueft: der
        Paarbeleg zeigt, dass Bytes ankommen; dieser zeigt, dass die
        Schlitz-Wiederverwendung ueber mehrere Runden traegt. Der zweite
        Punkt ist der gefaehrlichere, und er faellt in einer Einzelrunde
        nicht auf.
        """
        from sglang.srt.distributed.device_communicators import (
            htccl_bar1_pipe_ext,
        )

        if not self.pipe_an or self._pipe_ext is None:
            self._pipe_beleg = False
            return False
        # Vorlaeufig durchlassen, damit der Beleg selbst laufen kann; die
        # endgueltige Antwort steht unten. `handles` fragt waehrenddessen
        # niemand -- der Beleg laeuft beim Aufbau, vor dem ersten Kollektiv.
        self._pipe_beleg = True
        try:
            ok = htccl_bar1_pipe_ext.byte_beleg_pipe(self, runden)
        except Exception as e:
            logger.warning(
                "HTCCL-BAR1-PIPE: der Byte-Beleg ist mit %r abgebrochen. "
                "netz_pipe meldet sich ab; netz und ring bleiben unberuehrt.",
                e,
            )
            ok = False
        self._pipe_beleg = bool(ok)
        if not ok:
            logger.warning(
                "HTCCL-BAR1-PIPE: Byte-Beleg nicht bestanden -- netz_pipe "
                "meldet sich ueber handles() ab."
            )
        return self._pipe_beleg

    # -- all_gather --------------------------------------------------------
    #
    # DER STOPPER. Vor dieser Aenderung deckte HTCCL_OPS all_gather nicht,
    # und der Standardlauf brach ab:
    #
    #     RuntimeError: HTCCL: 'all_gather' mit 10600448 Byte waehrend einer
    #     CUDA-Graph-Aufzeichnung, aber bar1 meldet handles(...) -> False.
    #
    # Richtig abgebrochen -- unter HTCCL ist PyNccl nicht gebaut, der
    # Ausweichweg waere die host-gestaffelte gloo-Ebene, und die laeuft in
    # einer Aufzeichnung einmal und bei keiner Wiedergabe wieder. Nur ging es
    # eben nicht.
    #
    # WARUM KEIN NEUER KERN. Ein all_gather ist die AG-Phase des
    # Netz-Allreduce ohne die Reduktion, und genau das kann der a2a-Kern
    # schon: er bewegt Bytes, kennt keinen Datentyp, und er bekommt
    # Versaetze und Laengen JE RANG GETRENNT herein. Ein all_gather ist ein
    # all_to_all, bei dem jedes Ziel denselben Ausschnitt bekommt -- also
    # dieselbe Tabelle mit ``sende_versatz[z] = const``. Das ist kein
    # Kunstgriff, sondern die Zusage, die in htccl_bar1_ext.py schon
    # ausgeschrieben steht ("er hat nie angenommen, dass sie
    # zusammenhaengen").
    #
    # Was das mitbringt, ohne es zu bauen: den Byte-Beleg je gerichtetem
    # Paar (``byte_beleg_a2a``), die Haelftenwahl nach Rundenparitaet (die
    # Schlitzwiederverwendung ist damit ohne dritte Sperre sicher), den
    # lokalen Weg fuer den eigenen Block, den Restpfad fuer Laengen, die
    # kein Vielfaches von 16 sind, und die Grenzpruefungen der Erweiterung.
    # Ein zweiter Kern haette jedes dieser Stuecke ein zweites Mal gebraucht
    # -- und jedes waere eine Stelle, an der die beiden Fassungen
    # auseinanderlaufen.
    #
    # WAS DAS KOSTET, ehrlich: einen Zwischenschlitz. Der Empfaenger liest
    # aus seinem Schlitz in den Ausgabepuffer, statt dass der Sender direkt
    # in den Ausgabepuffer schreibt. Ohne Reduktion waere Letzteres
    # moeglich -- aber nur mit einem abgebildeten Ergebnispuffer, also mit
    # dem Direkt-Modus der Pipe, und der ueberlebt keine Aufzeichnung
    # (``_erg_platz``, Punkt 3: der hostseitige Ringindex wird je Graph
    # eingebrannt, bei erg_ring=2 teilen sich der erste und der dritte
    # Graph denselben BAR1-Platz). Der Abnahmefall IST eine Aufzeichnung.
    # Also dieselbe konservative Wahl wie beim Allreduce: Schlitz statt
    # Direkt, und der Direkt-Modus bleibt ein spaeterer, eigener Schritt
    # mit eigenem Ergebnisring und eigener Lebensdauerpruefung.

    def _handles_all_gather(self, nbytes: int) -> bool:
        """``nbytes`` ist die EIGENE Scherbe, nicht das Ergebnis.

        Die Naht fragt mit ``input_.numel() * element_size()``
        (``htccl.HTCCLCommunicator.all_gather``), also mit der Scherbe. Das
        Ergebnis ist ``R`` mal so gross und wird hier NICHT geprueft: es
        liegt im lokalen VRAM, nicht im Fenster.

        Jede Bedingung ist rangeinheitlich, aus demselben Grund wie in
        :meth:`handles`.

        Ueber den Schlitz hinaus wird NICHT abgelehnt, sondern in Runden
        zerlegt (:func:`ag_plan`) -- eine Ablehnung waere unter Aufzeichnung
        ein Abbruch ohne Ausweichweg, und genau daran hing der Stopper.
        Abgelehnt wird nur, was auch in Runden nicht geht.

        ``nbytes % 16 != 0`` wird ausdruecklich NICHT abgelehnt, anders als
        bei all_reduce. Der a2a-Kern hat dafuer den Restpfad (``VEK=0``,
        Paket byteweise zusammengesetzt): korrekt, langsamer, ungemessen.
        Das ist die richtige Wahl, weil die Alternative unter Aufzeichnung
        kein langsamerer Weg ist, sondern gar keiner.

        **Was eine krumme Scherbe kostet, genau gesagt:** nicht die letzten
        15 Byte, sondern alles. Der Ergebnisversatz von Rang ``i`` ist
        ``i * scherbe``; ist ``scherbe`` kein Vielfaches von 16, liegt jeder
        Versatz ausser dem von Rang 0 schief, und die Erweiterung schaltet
        auf ``VEK=0`` fuer den GANZEN Aufruf (sie prueft alle Versaetze
        gemeinsam, htccl_bar1_ext.py: "meistens ausgerichtet gibt es
        nicht"). Wer eine langsame Zahl sieht, sollte das zuerst pruefen,
        bevor er sie dem Transport zuschreibt.
        """
        if not self.ag_an:
            return False
        # Derselbe Bereich, derselbe Kern, derselbe Byte-Beleg. Ohne den
        # a2a-Bereich (SGLANG_HTCCL_BAR1_A2A=0) gibt es auch kein all_gather
        # -- gesagt, nicht still angenommen.
        if not self.a2a_an or not self._a2a_beleg:
            return False
        geo = self._geo
        if geo.get("off_a2a", -1) < 0 or geo.get("a2a_schlitz", 0) <= 0:
            return False
        if nbytes <= 0:
            return False
        if nbytes < self.ag_min_bytes:
            return False
        # Derselbe Fensterbegriff wie bei all_reduce und a2a: gegen die
        # gruppenweit KLEINSTE tatsaechlich abgebildete Laenge.
        if geo["region_bytes"] > self._fenster_minimum:
            return False
        # Eine Decke gibt es trotzdem, und sie ist keine Fenstergrenze,
        # sondern eine Rundengrenze: jede Runde ist ein Kernelstart mit einer
        # Sperre, und beliebig viele davon je Kollektiv waere kein Transport,
        # sondern eine Schleife. Rangeinheitlich, weil nbytes es ist.
        if -(-nbytes // int(geo["a2a_schlitz"])) > self.ag_max_runden:
            return False
        return True

    def ag_runden(self, nbytes: int) -> int:
        """Rundenzahl fuer eine Scherbe von ``nbytes`` -- fuer Protokoll/Test."""
        schlitz = int(self._geo.get("a2a_schlitz", 0))
        if schlitz <= 0:
            return 0
        return max(1, -(-int(nbytes) // schlitz))

    def htccl_all_gather(self, comm, inp, dim: int = -1):
        """``all_gather`` ueber den Direktpfad, notfalls in mehreren Runden.

        Ergebnisform und Achsenbehandlung sind Byte fuer Byte die der Naht
        (``htccl.HTCCLCommunicator.all_gather``) und die von
        ``htccl_device.all_gather``: erst ``(R,) + form``, dann
        ``movedim(0, dim)``, dann zusammenlegen. Nicht neu gedacht --
        derselbe Ausdruck, damit ein Transportwechsel nichts an den Zahlen
        aendert.
        """
        import torch

        if not self._auf or self._ext is None or not self.a2a_an:
            raise Bar1Unverfuegbar(
                "htccl_all_gather ohne aufgebauten a2a-Bereich -- erreichbar "
                "nur, wenn jemand handles() umgangen hat."
            )
        if dim < 0:
            dim += inp.dim()
        inp = inp.contiguous()
        form = tuple(inp.size())
        scherbe = inp.numel() * inp.element_size()
        out = torch.empty((self.welt,) + form, dtype=inp.dtype,
                          device=inp.device)
        # Gleichverteilt, weil die Naht es ist -- aber als VEKTOR an
        # ag_plan, nicht als Annahme in der Arithmetik. Die Begruendung
        # steht bei ag_plan.
        plan = ag_plan([scherbe] * self.welt, int(self._geo["a2a_schlitz"]))
        flach = out.view(-1)
        for runde in plan:
            s_off = [runde[self.rank][0]] * self.welt
            s_len = [runde[self.rank][1]] * self.welt
            e_off = [x[2] for x in runde]
            e_len = [x[1] for x in runde]
            self.htccl_all_to_all_single(
                comm, flach, inp, s_len, e_len, s_off, e_off,
            )
        out = out.movedim(0, dim)
        return out.reshape(form[:dim] + (self.welt * form[dim],) + form[dim + 1:])

    # -- all_to_all --------------------------------------------------------

    def _handles_a2a(self, nbytes: int) -> bool:
        """Die GROBE Antwort der Naht fuer ``all_to_all``.

        Sie kennt nur die Gesamtgroesse, nicht die Teilgroessen -- also
        prueft sie den gleichverteilten Fall. Fuer ungleiche Teilgroessen
        (bei MoE der Normalfall) entscheidet :meth:`traegt_a2a`, sobald die
        Zaehlwerte feststehen. Beide Antworten sind rangeinheitlich: diese
        haengt nur an gruppenweit abgeglichenen Groessen, jene an einem Wert,
        den der Aufrufer gruppenweit maximiert hereinreicht.
        """
        if not self.a2a_an or not self._a2a_beleg:
            return False
        geo = self._geo
        if geo.get("off_a2a", -1) < 0 or geo.get("a2a_schlitz", 0) <= 0:
            return False
        if nbytes < self.a2a_min_bytes:
            return False
        if -(-nbytes // self.welt) > geo["a2a_schlitz"]:
            return False
        # Derselbe Fensterbegriff wie bei all_reduce: gegen die gruppenweit
        # KLEINSTE tatsaechlich abgebildete Laenge, nicht gegen die
        # angeforderte. Redundant, solange der Aufbau durchgelaufen ist --
        # und genau deshalb billig.
        if geo["region_bytes"] > self._fenster_minimum:
            return False
        return True

    def a2a_schlitz_bytes(self) -> int:
        """Groesster Block, den EIN gerichtetes Paar tragen kann. 0 = kein a2a."""
        if not self.a2a_an or not self._a2a_beleg:
            return 0
        return int(self._geo.get("a2a_schlitz", 0))

    def traegt_a2a(self, groesster_block: int) -> bool:
        """Passt der groesste Block ueber ALLE Paare in einen Schlitz?

        ``groesster_block`` muss ein **gruppenweit identischer** Wert sein --
        das Maximum ueber alle R*R Bloecke, nicht ueber die eigene Zeile.
        Rechnet ihn jeder Rang nur aus seinen eigenen Teilgroessen, kann ein
        Rang ins Kollektiv laufen und ein anderer in den Rueckfall, und das
        Ergebnis ist ein Haenger statt eines Fehlers. Der Aufrufer
        (``HTCCLCommunicator.all_to_all_single``) maximiert vorher ueber die
        Gruppe; genau das ist der Grund, warum diese Pruefung nicht in
        ``handles`` liegt.
        """
        if not self.a2a_an or not self._a2a_beleg or not self._auf:
            return False
        if groesster_block < 0:
            return False
        return groesster_block <= int(self._geo.get("a2a_schlitz", 0))

    def htccl_all_to_all_single(self, comm, output, inp,
                                sende_bytes, empfangs_bytes,
                                sende_versatz=None, empfangs_versatz=None):
        """``all_to_all_single`` ueber den Direktpfad. Ein Schritt, eine Sperre.

        ``sende_bytes[j]`` ist der Block, der an Rang ``j`` geht,
        ``empfangs_bytes[i]`` der, der von Rang ``i`` kommt -- beides in
        **Byte**, nicht in Zeilen und nicht in Elementen. Der Kernel bewegt
        Bytes: es gibt keine Reduktion, also keinen Datentyp. fp8, bf16,
        int32, uint8 laufen denselben Weg, und die fehlenden
        fp8-Umwandlungsbefehle der sm_86-Karten sind hier gegenstandslos.

        ``sende_versatz`` / ``empfangs_versatz`` sind **optional** und in
        Byte. ``None`` heisst: die Praefixsumme der Laengen, also die
        lueckenlose Aneinanderreihung, die ``torch.distributed.
        all_to_all_single`` meint -- Byte fuer Byte der bisherige Weg.
        Angegeben werden sie von einem Aufrufer, der aus EINEM stehenden
        Puffer nur ein Stueck je Block bewegen will, ohne vorher umzukopieren:
        genau das braucht der MoE-Dispatcher, wenn ein Block groesser ist als
        der Schlitz und deshalb ueber mehrere Runden geht. Der Kernel selbst
        kennt beide Faelle schon -- er bekommt Versaetze und Laengen getrennt
        herein und hat nie angenommen, dass sie zusammenhaengen; nur diese
        Naht hat die Versaetze bisher selbst gerechnet.

        Der Aufrufer haftet dafuer, dass ``empfangs_bytes[i]`` auf diesem
        Rang gleich ``sende_bytes[rang]`` auf Rang ``i`` ist. Die Erweiterung
        prueft, was sie lokal pruefen kann (Puffergrenzen, Schlitzgrenze,
        eigener Block), aber nicht die Uebereinstimmung ueber die Gruppe --
        dafuer muesste sie ein Kollektiv fahren, und das waere genau der
        Host-Sync, den dieser Pfad vermeidet.
        """
        if not self._auf or self._ext is None or not self.a2a_an:
            raise Bar1Unverfuegbar(
                "htccl_all_to_all_single ohne aufgebauten a2a-Bereich -- "
                "erreichbar nur, wenn jemand handles() umgangen hat."
            )
        R = self.welt
        if len(sende_bytes) != R or len(empfangs_bytes) != R:
            raise Bar1Unverfuegbar(
                f"Teilgroessen haben Laenge {len(sende_bytes)}/"
                f"{len(empfangs_bytes)}, erwartet sind {R}."
            )
        inp = inp.contiguous()
        if not output.is_contiguous():
            raise Bar1Unverfuegbar("Ausgabepuffer ist nicht zusammenhaengend")

        if sende_versatz is None:
            sende_off, s = [], 0
            for n in sende_bytes:
                sende_off.append(s)
                s += int(n)
        else:
            if len(sende_versatz) != R:
                raise Bar1Unverfuegbar(
                    f"sende_versatz hat Laenge {len(sende_versatz)}, "
                    f"erwartet sind {R}."
                )
            sende_off = [int(x) for x in sende_versatz]
        if empfangs_versatz is None:
            empf_off, e = [], 0
            for n in empfangs_bytes:
                empf_off.append(e)
                e += int(n)
        else:
            if len(empfangs_versatz) != R:
                raise Bar1Unverfuegbar(
                    f"empfangs_versatz hat Laenge {len(empfangs_versatz)}, "
                    f"erwartet sind {R}."
                )
            empf_off = [int(x) for x in empfangs_versatz]

        # Der cooperative Mehrblockstart lohnt sich nach derselben Schwelle
        # wie bei all_reduce -- gemessen ist sie DORT und nur dort; hier ist
        # sie uebernommen, nicht bestaetigt. Massgeblich ist, was wirklich
        # ueber PCIe geht, also ohne den eigenen Block.
        bewegt = sum(int(n) for j, n in enumerate(sende_bytes) if j != self.rank)
        kern = self._kern(bewegt, self.gitter_ab, "all_to_all_single")

        peer_nutz = [0] * R
        peer_flag = [0] * R
        for rr, z in self._peers.items():
            peer_nutz[rr] = z.nutz.dev_ptr
            peer_flag[rr] = z.flag.dev_ptr
        peer_nutz[self.rank] = self._eigen[0]
        peer_flag[self.rank] = self._eigen_flag[0]

        self._ext.bar1_all_to_all(
            inp, output, int(self.rank), int(R),
            [int(x) for x in sende_off], [int(x) for x in sende_bytes],
            [int(x) for x in empf_off], [int(x) for x in empfangs_bytes],
            peer_nutz, peer_flag,
            int(self._eigen[0]), int(self._eigen_flag[0]),
            int(self._geo["a2a_schlitz"]), int(self._geo["off_a2a"]),
            int(fbasis_a2a(R)),
            self._runde_dev, self._ctl_dev,
            int(self.deckel_zyklen), int(self.threads), int(kern),
            int(self.ladeform),
        )
        return output

    @staticmethod
    def _a2a_marke(quelle: int, ziel: int) -> int:
        """Ein je gerichtetem Paar verschiedenes Byte, nie 0x00 und nie 0xFF.

        ``0x40 | (quelle*8 + ziel)`` -- fuer R <= 8 ist ``quelle*8+ziel``
        injektiv und passt in 6 Bit. 0xFF ist die Vorbelegung des
        Ausgabepuffers, 0x00 die des Empfangsschlitzes; beide sind damit
        vom Muster unterscheidbar, und ein NICHT geschriebener Block faellt
        als solcher auf statt zufaellig wie ein Treffer auszusehen.
        """
        return 0x40 | ((quelle * 8 + ziel) & 0x3F)

    def byte_beleg_a2a(self) -> bool:
        """Byte-Beleg fuer ``all_to_all``: jedes Byte, jedes gerichtete Paar.

        Zwei Durchgaenge ueber den ECHTEN Kernel -- nicht ueber ``put``, denn
        was belegt werden soll, ist der Weg, den die Kollektive nehmen,
        einschliesslich Schlitzwahl, Haelftenwahl und Sperre:

        1. **gleichverteilt** -- jeder Block gleich gross, alle Versaetze
           16-Byte-ausgerichtet. Das ist der Vektorpfad des Kernels.
        2. **schief und krumm** -- Blocklaengen ``block*(1+((q+z)%3)) +
           ((q*5+z*3)%7)``. Der Faktor macht die Teilgroessen ungleich (der
           MoE-Normalfall), der Summand macht sie zu Nicht-Vielfachen von 16
           und schiebt damit jeden folgenden Versatz aus der Ausrichtung --
           genau der Restpfad, der sonst nie liefe und in dem die letzten
           Bytes eines Blocks liegen.

        Geprueft wird auf der EMPFANGENDEN Karte gegen den lokalen
        Ausgabepuffer, Byte fuer Byte, fuer jeden Sender einzeln -- auch fuer
        den eigenen Block, der gar nicht ueber die Apertur laeuft (sonst
        faellt ein vertauschter Versatz nicht auf).

        Faellt der Beleg, meldet sich **nur** ``all_to_all`` ab.
        ``all_reduce`` benutzt andere Schlitze, andere Flaggenzeilen und
        einen gemessenen Kernel; es mit abzuschalten waere eine
        Schlussfolgerung, die die Probe nicht hergibt.
        """
        import torch.distributed as dist

        self._a2a_beleg = False
        if not self.a2a_an:
            logger.info(
                "HTCCL-BAR1: all_to_all ist per SGLANG_HTCCL_BAR1_A2A=0 "
                "abgeschaltet -- kein Byte-Beleg, handles() sagt False."
            )
            return False
        if not self._auf or self._ext is None:
            return False

        R, r = self.welt, self.rank
        schlitz = int(self._geo.get("a2a_schlitz", 0))
        # Der groesste Block des schiefen Durchgangs ist 3*block+6.
        block = min(8192, (schlitz - 6) // 3)
        block = (block // 16) * 16
        if block <= 0:
            logger.warning(
                "HTCCL-BAR1: a2a-Schlitz von %d Byte ist zu klein fuer den "
                "Byte-Beleg. all_to_all meldet sich ab.", schlitz,
            )
            return False

        # Ab hier laeuft der Abgleich ueber die Gruppe IN JEDEM Fall. Ein
        # Rang, der wegen einer lokalen Ausnahme vor dem all_gather_object
        # aussteigt, laesst die anderen darin stehen -- aus einem
        # fehlgeschlagenen Beleg wuerde ein Haenger.
        ok_lokal = True
        try:
            ok_lokal = self._a2a_beleg_durchgaenge(block)
        except Exception as ex:                # noqa: BLE001
            ok_lokal = False
            logger.warning("HTCCL-BAR1: a2a-Byte-Beleg abgebrochen: %r", ex)

        traeger: list[object] = [None] * R
        dist.all_gather_object(traeger, bool(ok_lokal), group=self.cpu_group)
        self._a2a_beleg = all(bool(x) for x in traeger)
        if not self._a2a_beleg:
            logger.warning(
                "HTCCL-BAR1: a2a-Byte-Beleg gruppenweit gefallen (Raenge %s). "
                "handles('all_to_all') gibt False; all_reduce bleibt "
                "unberuehrt.",
                [i for i, x in enumerate(traeger) if not x],
            )
        return self._a2a_beleg

    def _a2a_beleg_durchgaenge(self, block: int) -> bool:
        """Die beiden Probedurchgaenge. Rein lokal, ohne Gruppenabgleich.

        **Beide Durchgaenge laufen immer**, auch wenn der erste gefallen ist.
        In jedem steckt eine ``dist.barrier``; ein Rang, der nach einem
        Fehlschlag abkuerzt, laesst die anderen in der naechsten Barriere
        stehen. Aus einem gefallenen Beleg wuerde ein Haenger -- und ein
        Haenger sagt nicht, was kaputt ist.
        """
        import torch
        import torch.distributed as dist

        R, r = self.welt, self.rank
        ok_lokal = True
        for schief in (False, True):

            def laenge(q: int, z: int) -> int:
                if not schief:
                    return block
                return block * (1 + ((q + z) % 3)) + ((q * 5 + z * 3) % 7)

            sende = [laenge(r, z) for z in range(R)]
            empf = [laenge(q, r) for q in range(R)]
            inp = torch.empty(sum(sende), dtype=torch.uint8, device=self.device)
            o = 0
            for z in range(R):
                inp[o:o + sende[z]] = self._a2a_marke(r, z)
                o += sende[z]
            out = torch.full((sum(empf),), 0xFF, dtype=torch.uint8,
                             device=self.device)
            dist.barrier(group=self.cpu_group)
            gelaufen = True
            try:
                self.htccl_all_to_all_single(None, out, inp, sende, empf)
                torch.cuda.synchronize(self.device)
            except Exception as ex:            # noqa: BLE001 -- Grund ins Protokoll
                logger.warning(
                    "HTCCL-BAR1: a2a-Byte-Beleg (%s) nicht durchfuehrbar: %r",
                    "schief" if schief else "gleich", ex,
                )
                ok_lokal = False
                gelaufen = False
            if not gelaufen:
                continue
            rueck = out.cpu()
            o = 0
            schlecht_ges = 0
            for q in range(R):
                soll = self._a2a_marke(q, r)
                stueck = rueck[o:o + empf[q]]
                schlecht = int((stueck != soll).sum().item())
                if schlecht:
                    ok_lokal = False
                    schlecht_ges += schlecht
                    logger.warning(
                        "HTCCL-BAR1: a2a-Byte-Beleg %d->%d (%s) GEFALLEN: %d "
                        "von %d Byte falsch. all_to_all meldet sich ab.",
                        q, r, "schief" if schief else "gleich", schlecht,
                        empf[q],
                    )
                o += empf[q]
            if not schlecht_ges:
                # Auch der bestandene Beleg gehoert ins Protokoll -- er ist
                # die Aussage, auf der jede spaetere Zeitmessung steht.
                logger.info(
                    "HTCCL-BAR1: a2a-Byte-Beleg (%s) bestanden: 0 von %d Byte "
                    "falsch ueber %d Sender.",
                    "schief" if schief else "gleich", sum(empf), R,
                )
        return ok_lokal

    def status(self) -> int:
        """``1``, wenn ein Kernel je das Zeitlimit gerissen hat.

        Bewusst eine eigene Abfrage und nicht im heissen Pfad: sie liest ein
        Geraetewort und synchronisiert damit -- genau das, was der
        Direktpfad vermeidet. Wer sie ruft, will es wissen und zahlt dafuer.
        """
        if self._ctl_dev is None:
            return 0
        return int(self._ctl_dev[0].item())

    def _kein_kollektiv(self, comm, *args, **kwargs):
        """Der Platzhalter fuer die weiter ungedeckten Kollektive.

        ``*args`` ist kein Bequemlichkeitszeichen: die Nahtstellen rufen mit
        verschiedenen Signaturen (``htccl_reduce_scatter(comm, inp, dim)``,
        ``htccl_broadcast(comm, tensor, src)``). Mit dem frueheren festen
        ``(self, comm, inp)`` kam bei beiden ein ``TypeError`` heraus, bevor
        diese Meldung ueberhaupt zum Zug kam -- der Text war also unerreichbar
        und die Ursache stand nirgends.
        """
        raise NotImplementedError(
            f"Der BAR1-Transport deckt {', '.join(sorted(self.HTCCL_OPS))} "
            f"ab. reduce_scatter braucht eine Reduktion (der a2a-Kern bewegt "
            f"Bytes und traegt deshalb all_gather gratis, reduce_scatter gar "
            f"nicht), broadcast ist an dieser Naht an Ort und die Erweiterung "
            f"lehnt in==out ab. Erreichbar ist diese Zeile nur, wenn jemand "
            f"handles() umgangen hat."
        )

    # all_gather ist NICHT mehr hier. Die Zuweisung stand bis zur Einfuehrung
    # von htccl_all_gather in dieser Liste und haette die neue Methode
    # ueberschrieben -- eine Zuweisung im Klassenkoerper gewinnt gegen ein
    # weiter oben stehendes `def` desselben Namens, lautlos. Ruff (F811) hat
    # es gefunden; ohne den Lauf haette jedes all_gather ein
    # NotImplementedError geworfen, obwohl handles() zugesagt hatte, und der
    # Riegel haette wie ein Transportfehler ausgesehen.
    htccl_reduce_scatter = _kein_kollektiv
    htccl_broadcast = _kein_kollektiv

    # -- Abbau -------------------------------------------------------------

    def close(self) -> None:
        """Alles wieder abbauen. Reihenfolge zaehlt.

        Erst die Registrierungen und Abbildungen der fremden BARs, dann die
        Anhaftungen (die halten die BAR1-Seiten), dann die eigene
        Allokation. Umgekehrt zoege man dem Treiber Seiten unter einer
        lebenden Abbildung weg.
        """
        self._auf = False
        # Zuerst austragen: der Platz ist ab hier auf dem Weg zurueck, und
        # eine Kasse, die nach einem `close` noch belegt meldet, wuerde eine
        # spaeter gebaute Gruppe grundlos kuerzen.
        try:
            from sglang.srt.distributed.device_communicators import (
                htccl_matrix_transport as _kasse,
            )

            _kasse.kasse_austragen(self.device, self.gruppe)
        except Exception:
            pass
        for z in self._peers.values():
            for a in (z.nutz, z.flag):
                if self._cuda is not None:
                    # ABMELDEN AN DERSELBEN ADRESSE, UNTER DER REGISTRIERT
                    # WURDE. cudaHostUnregister auf einen Zeiger MITTEN in der
                    # Registrierung schlaegt fehl, und die Abmeldung waere
                    # stillschweigend ausgeblieben -- die Apertur bliebe beim
                    # naechsten Lauf registriert.
                    self._cuda.unregister(a.reg_adresse)
                try:
                    a.mmap_obj.close()      # type: ignore[attr-defined]
                except Exception:
                    pass
        if self._halter is not None:
            for z in self._peers.values():
                for a in (z.nutz, z.flag):
                    self._halter.gib_frei(a.halter_handle)
            self._halter.schliesse()
            self._halter = None
        self._peers.clear()
        eigene = set(self._dmabuf_fds)
        for liste in self._fremde_fds:
            for fd in liste or ():
                if fd is not None and fd >= 0 and fd not in eigene:
                    try:
                        os.close(fd)
                    except OSError:
                        pass
        self._fremde_fds = []
        for fd in self._dmabuf_fds:
            try:
                os.close(fd)
            except OSError:
                pass
        self._dmabuf_fds = []
        # ERST JETZT: an /dev/nvidiactl haengt der RM-Client, dem das
        # exportierte Speicherobjekt gehoert. Waere er vorher geschlossen
        # worden, haette RM das Objekt unter den noch lebenden Abbildungen
        # der Peers freigegeben.
        for fd in self._halte_fds:
            try:
                os.close(fd)
            except OSError:
                pass
        self._halte_fds = []
        if self._cuda is not None:
            for eig in ("_eigen", "_eigen_flag"):
                w = getattr(self, eig)
                if w[2]:
                    self._cuda.vmm_frei(*w)
                    setattr(self, eig, (0, 0, 0))
        self._runde_dev = None
        self._ctl_dev = None
        self._schritt_dev = None


def baue_bar1(cpu_group, device, fenster_bytes: int,
              bericht: Optional[dict] = None,
              gruppe: str = "") -> Optional[HTCCLBar1Transport]:
    """Fabrik mit sauberem Rueckfall.

    ``None`` heisst: dieser Rechner kann den Direktpfad nicht, mit
    protokolliertem Grund. Kein Werfen, kein stilles Ausweichen auf einen
    anderen Weg -- die Wahl des Ersatzpfades trifft der Aufrufer, nicht
    dieses Modul.

    ``fenster_bytes`` ist die **angeforderte** Groesse der Empfangsregion je
    Rang. Was daraus wirklich wird, sagt danach
    ``transport.fenster_minimum()`` -- und nur das gehoert in den Planer.

    ``bericht`` ist der GRUND, und er ist kein Beiwerk. Bisher endete jeder
    Ausfall in einem ``logger.info`` und einem ``None``, und der Aufrufer
    protokollierte danach ungerührt "transport=bar1". Genau so ist eine
    Messung entwertet worden: die tp-Gruppe fuhr ueber BAR1, die
    dcp-Gruppe ueber gloo, und das Protokoll sah in beiden Faellen gleich
    aus. Wer ``bericht`` mitgibt, bekommt hier ``grund`` und ``stufe``
    ("aufbau", "byte_beleg") hineingeschrieben und kann daraus eine laute
    Meldung machen.
    """
    if bericht is None:
        bericht = {}

    def _aus(stufe: str, text: str):
        bericht["stufe"] = stufe
        bericht["grund"] = text
        return None

    try:
        t = HTCCLBar1Transport(cpu_group, device, fenster_bytes, gruppe=gruppe)
    except Bar1Unverfuegbar as e:
        logger.info("HTCCL-BAR1: Direktpfad nicht verfuegbar -- %s", e)
        return _aus("aufbau", str(e))
    except NotImplementedError as e:
        logger.info("HTCCL-BAR1: Direktpfad braucht Treiberarbeit -- %s", e)
        return _aus("aufbau", f"Treiberarbeit noetig: {e}")
    except Exception as e:                 # ein halber Aufbau bleibt nicht stehen
        logger.info("HTCCL-BAR1: Aufbau fehlgeschlagen -- %r", e)
        return _aus("aufbau", f"{type(e).__name__}: {e}")
    # Der Byte-Beleg gehoert zum Aufbau, nicht zur Kuer: ohne ihn ist
    # `handles` gesperrt. Auf diesem Rig meldete der Treiber fuer ein Paar
    # Peer-Zugriff und lieferte 4096 von 1048576 Byte.
    try:
        belege = t.byte_beleg_alle()
    except Exception as e:
        logger.info("HTCCL-BAR1: Byte-Beleg nicht durchfuehrbar -- %r", e)
        t.close()
        return _aus("byte_beleg", f"nicht durchfuehrbar: {type(e).__name__}: {e}")
    if not t._belege_stehen:
        # Bisher kam der Transport hier UNVERSEHRT heraus und meldete sich
        # erst spaeter ueber `handles` ab -- also lief jedes Kollektiv still
        # ueber die gloo-Ebene, waehrend das Protokoll "transport=bar1"
        # sagte. Der Grund gehoert dem Aufrufer gemeldet, nicht verschwiegen.
        gefallen = sorted(k for k, v in belege.items() if not v)
        bericht["stufe"] = "byte_beleg"
        bericht["grund"] = (
            f"Byte-Beleg gefallen fuer die gerichteten Paare {gefallen}. "
            f"handles() sagt zu allem False; jedes Kollektiv dieser Gruppe "
            f"laeuft ueber die gloo-Ebene."
        )
        bericht["haelt_belegt"] = True
    # Und derselbe Grundsatz fuer all_to_all -- eigener Kern, eigene
    # Schlitze, eigene Flaggenzeilen, also eigener Beleg. Er wird NUR
    # versucht, wenn der all_reduce-Beleg steht: ein Kollektiv ueber eine
    # Kante, die schon dort Bytes verloren hat, braucht keine zweite Probe.
    # Faellt er, bleibt all_reduce trotzdem verfuegbar; deshalb wird der
    # Transport hier auch nicht abgeraeumt.
    if t._belege_stehen:
        try:
            t.byte_beleg_a2a()
        except Exception as e:
            logger.info(
                "HTCCL-BAR1: a2a-Byte-Beleg nicht durchfuehrbar -- %r. "
                "all_to_all meldet sich ab, all_reduce laeuft weiter.", e,
            )
        # Und derselbe Grundsatz noch einmal fuer netz_pipe: eigener Kern,
        # eigene Schlitze, eigene Zaehlerzeilen, also eigener Beleg -- und
        # zwar einer ueber MEHRERE Runden, weil die Schlitz-Wiederverwendung
        # in einer Einzelrunde gar nicht drankommt. Faellt er, bleiben netz
        # und ring verfuegbar.
        if t.pipe_an:
            try:
                t.byte_beleg_pipe()
            except Exception as e:
                logger.info(
                    "HTCCL-BAR1: pipe-Byte-Beleg nicht durchfuehrbar -- %r. "
                    "netz_pipe meldet sich ab, netz und ring laufen weiter.", e,
                )
    return t
