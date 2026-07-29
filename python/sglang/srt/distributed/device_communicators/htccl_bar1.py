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

Was hier NICHT drin ist
-----------------------
``all_gather``, ``reduce_scatter`` und ``broadcast``. Sie waeren je eine
Haelfte der obigen Kernel, aber keine davon ist gemessen -- und in diesem
Vorhaben zaehlt nur Gemessenes. ``handles`` gibt dafuer ``False`` und die
``htccl_*``-Methoden erklaeren, was fehlt.
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
        self._d("cuMemsetD8", ctypes.c_ulonglong(dptr), ctypes.c_ubyte(wert),
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
        """
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
        eintraege = []
        roh = bytes(puffer.raw[: 16 * min(nents, max_eintraege)])
        for i in range(min(nents, max_eintraege)):
            a, l = struct.unpack_from("=QQ", roh, 16 * i)
            eintraege.append(SgEintrag(a, l))
        self._handles.append(handle_)
        if not eintraege:
            raise Bar1Unverfuegbar(
                "Der Halter meldet 0 sg-Eintraege -- die Abbildung ist leer. "
                "Ohne sg-Adresse ist der BAR1-Versatz nicht bestimmbar; der "
                "Musterscan waere der Rueckfall, er gehoert aber nicht in "
                "einen Transport."
            )
        return handle_, eintraege, int(total_len)

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
    if algorithmus in ("netz", "ring", "hierarchisch"):
        return 2 * (welt - 1) * anteil
    if algorithmus == "stern":
        return 2 * (welt - 1) * nbytes
    raise ValueError(f"unbekannter Algorithmus {algorithmus!r}")


def geometrie(welt: int, max_bytes: int) -> dict:
    """Die Speicherordnung EINER Empfangsregion, fuer beliebiges R.

    Sie traegt beide Verfahren **gleichzeitig**, damit ein Plan je Groesse
    umschalten kann, ohne dass irgendetwas neu abgebildet wird:

    ==========  ================  =========================================
    Versatz     Inhalt            Groesse
    ==========  ================  =========================================
    ``0``       Netz-RS-Schlitze  ``(R-1) * chunk_max``
    ...         Netz-AG-Schlitze  ``(R-1) * chunk_max``
    ``off_ring`` Ring-Schlitze    ``2(R-1) * chunk_max``
    ==========  ================  =========================================

    ``chunk_max`` ist auf eine Seite aufgerundet -- ein Schlitz, der auf
    einer Seitengrenze beginnt, kann nie mit dem Nachbarschlitz eine Seite
    teilen, und ein ueberlanger Schreibvorgang trifft dann eine eigene
    Seite statt fremde Nutzlast.
    """
    if welt < 2:
        raise ValueError("welt < 2")
    n4_max = max_bytes // 16
    chunk4 = -(-n4_max // welt)
    chunk_max = ((chunk4 * 16 + SEITE - 1) // SEITE) * SEITE
    schlitze = 2 * (welt - 1)
    off_netz = 0
    off_ring = schlitze * chunk_max
    region = off_ring + schlitze * chunk_max + SEITE
    return {
        "chunk_max": chunk_max,
        "off_netz": off_netz,
        "off_ring": off_ring,
        "region_bytes": region,
        "max_bytes": max_bytes,
    }


def flaggen_bedarf(welt: int) -> int:
    """``(2 + 2(R-1)) * R * 256`` Byte.

    Eine 256-Byte-Zeile je (Topologie, Schritt, Sender): kein False Sharing
    zwischen Sendern, keins zwischen Schritten, keins zwischen Topologien.
    Netz hat 2 Schritte, Ring ``2(R-1)``. Bei R=8 sind das 32 KiB und damit
    weit unter einer Allokationsgranularitaet.
    """
    return (2 + 2 * (welt - 1)) * welt * 256


def max_nutzlast(welt: int, region_bytes: int) -> int:
    """Groesste Nutzlast, deren Schlitze in eine Region dieser Groesse passen.

    Umkehrung von :func:`geometrie`. Bewusst konservativ gerundet und
    danach gegengerechnet -- eine Umkehrung, die um eine Seite danebenliegt,
    faellt sonst erst im heissen Pfad auf.
    """
    if welt < 2 or region_bytes <= SEITE:
        return 0
    schlitze = 4 * (welt - 1)
    chunk_max = ((region_bytes - SEITE) // schlitze // SEITE) * SEITE
    if chunk_max <= 0:
        return 0
    n = (chunk_max // 16) * welt * 16
    while n > 0 and geometrie(welt, n)["region_bytes"] > region_bytes:
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

    #: Nur all_reduce. all_gather/reduce_scatter/broadcast waeren je eine
    #: Haelfte derselben Kerne, sind aber nicht gemessen.
    HTCCL_OPS: frozenset = frozenset({"all_reduce"})

    def __init__(self, cpu_group, device, fenster_bytes: int,
                 aktiviert: Optional[bool] = None):
        import torch
        import torch.distributed as dist

        self.cpu_group = cpu_group
        self.device = device
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
        # Notschwelle netz->ring, falls kein Plan hereingereicht wird.
        self.ring_ab = int(
            os.environ.get("SGLANG_HTCCL_BAR1_RING_AB", str(1 << 20))
        )
        self.min_bytes = int(os.environ.get("SGLANG_HTCCL_BAR1_MIN_BYTES", "4096"))
        self.max_bytes = 0
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
                    if z.startswith("RegistryDwords"):
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
        gesammelt: list[object] = [None] * self.welt
        dist.all_gather_object(gesammelt, eigener_bdf, group=self.cpu_group)
        self.bdfs = [str(x) for x in gesammelt]

        # 0. Die Kerne. Zuerst, weil ein fehlgeschlagener Bau billiger
        # abzubrechen ist als eine halb aufgebaute Peer-Tabelle.
        from sglang.srt.distributed.device_communicators import htccl_bar1_ext

        try:
            self._ext = htccl_bar1_ext.lade_kollektiv_ext(self.cpu_group)
        except Exception as e:
            raise Bar1Unverfuegbar(
                f"Die Kollektiv-Erweiterung liess sich nicht uebersetzen: {e}"
            ) from e

        # 1. Speicherordnung. Aus dem Fenster, das der Aufrufer bewilligt,
        # folgt die groesste Nutzlast -- nicht umgekehrt.
        max_bytes = max_nutzlast(self.welt, self.fenster_bytes)
        if max_bytes < self.min_bytes:
            raise Bar1Unverfuegbar(
                f"Fenster von {self.fenster_bytes // 1024} KiB traegt bei "
                f"{self.welt} Raengen nur {max_bytes} Byte Nutzlast, "
                f"Mindestgroesse ist {self.min_bytes}. 4(R-1) Schlitze zu je "
                f"ceil(N/R) muessen hineinpassen."
            )
        self._geo = geometrie(self.welt, max_bytes)
        self.max_bytes = max_bytes
        region = self._geo["region_bytes"]
        flaggen = flaggen_bedarf(self.welt)

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

        dist.barrier(group=self.cpu_group)
        self._auf = True
        dauer = time.perf_counter() - t0
        logger.info(
            "HTCCL-BAR1: Aufbau in %.0f ms, %d Peer-Ziele, Region %.1f MiB je "
            "Rang (%s), Schlitz %d KiB, groesste Nutzlast %d KiB, Flaggen %d "
            "Byte, Export ueber %s. Ab hier wird im heissen Pfad nichts mehr "
            "gemappt.",
            dauer * 1000, len(self._peers), region / 2**20,
            f"{4 * (self.welt - 1)} Schlitze", self._geo["chunk_max"] // 1024,
            max_bytes // 1024, flaggen, weg,
        )

    def _binde_peer(self, peer: int, fremde_fds: list) -> PeerZiel:
        """Beide Regionen eines Peers anhaengen, abbilden, registrieren."""
        ziel_bdf = self.bdfs[peer]
        nutz = self._binde_region(peer, ziel_bdf, fremde_fds[0], "Nutzlast",
                                  self._geo["region_bytes"])
        try:
            flag = self._binde_region(peer, ziel_bdf, fremde_fds[1], "Flaggen",
                                      flaggen_bedarf(self.welt))
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
            raise Bar1Unverfuegbar(
                f"cudaHostRegister(IoMemory) auf die {was}region von Rang "
                f"{peer} ({ziel_bdf}) fehlgeschlagen: {e}. Das ist der erwartete "
                f"Ausgang OHNE den geweiteten Guard: der Regkey "
                f"RMSmallBarP2PPeerBar1 hat die Vorgabe 0 und muss ueber "
                f"NVreg_RegistryDwords gesetzt werden. Gefunden: "
                f"{ps['regkeys'] or '<leer>'}. Der Transport meldet sich ab, "
                f"er erzwingt nichts."
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
        """``netz`` oder ``ring`` fuer diese Groesse.

        Vorrang hat der Plan aus ``htccl_matrix.py``, wenn einer
        hereingereicht wurde (``setze_plan``). Er ist gruppenweit
        identisch -- geprueft ueber die Planpruefsumme -- und damit die
        einzige Quelle, die diese Wahl rangeinheitlich halten kann.

        Ohne Plan die Notschwelle ``SGLANG_HTCCL_BAR1_RING_AB``. Sie ist
        eine Vorgabe und keine Messaussage: zwischen 1 und 16 MiB liegen
        Netz und Ring in der Sonde 1 bis 7 Prozent auseinander, und der
        Befund dazu sagt ausdruecklich "keine saubere Schwelle -- der
        Planer sollte das messen, nicht einbauen".
        """
        if self._plan is not None:
            a = self._plan.algorithmus_fuer(nbytes)
            # 'stern' und 'hierarchisch' sind hier nicht portiert; sie
            # kommen ueber handles() gar nicht bis hierher.
            return a
        return "ring" if nbytes >= self.ring_ab else "netz"

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
                    if not ok:
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
        if algo not in ("netz", "ring"):
            # 'stern' und 'hierarchisch' sind hier nicht portiert. Kein
            # stilles Ausweichen auf 'netz'.
            return False
        # Und derselbe Bedarf noch einmal in der Waehrung des Planers, gegen
        # die gruppenweit KLEINSTE tatsaechlich abgebildete Laenge. Redundant,
        # solange der Aufbau durchgelaufen ist -- und genau deshalb billig:
        # die Zeile faellt auf, wenn jemand die Regionsgroesse anfasst, ohne
        # den Fensterbegriff mitzuziehen.
        if fensterbedarf(algo, nbytes, self.welt) > self._fenster_minimum:
            return False
        return True

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
        out = torch.empty_like(inp)
        # 'gitter' ist der cooperative Mehrblockstart. Die Schwelle ist
        # gemessen (ab 4 MiB gewinnt er), aber sie ist eine Zahl aus EINEM
        # Rig -- deshalb steht sie in einer Umgebungsvariablen.
        kern = 1 if nbytes >= self.gitter_ab else 0
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

    def status(self) -> int:
        """``1``, wenn ein Kernel je das Zeitlimit gerissen hat.

        Bewusst eine eigene Abfrage und nicht im heissen Pfad: sie liest ein
        Geraetewort und synchronisiert damit -- genau das, was der
        Direktpfad vermeidet. Wer sie ruft, will es wissen und zahlt dafuer.
        """
        if self._ctl_dev is None:
            return 0
        return int(self._ctl_dev[0].item())

    def _kein_kollektiv(self, comm, inp):
        raise NotImplementedError(
            "Der BAR1-Transport bietet heute nur all_reduce an. all_gather, "
            "reduce_scatter und broadcast waeren je eine Haelfte derselben "
            "Kerne -- gemessen ist keine davon, und in diesem Vorhaben zaehlt "
            "nur Gemessenes. Erreichbar ist diese Zeile nur, wenn jemand "
            "handles() umgangen hat."
        )

    htccl_all_gather = _kein_kollektiv
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


def baue_bar1(cpu_group, device, fenster_bytes: int) -> Optional[HTCCLBar1Transport]:
    """Fabrik mit sauberem Rueckfall.

    ``None`` heisst: dieser Rechner kann den Direktpfad nicht, mit
    protokolliertem Grund. Kein Werfen, kein stilles Ausweichen auf einen
    anderen Weg -- die Wahl des Ersatzpfades trifft der Aufrufer, nicht
    dieses Modul.

    ``fenster_bytes`` ist die **angeforderte** Groesse der Empfangsregion je
    Rang. Was daraus wirklich wird, sagt danach
    ``transport.fenster_minimum()`` -- und nur das gehoert in den Planer.
    """
    try:
        t = HTCCLBar1Transport(cpu_group, device, fenster_bytes)
    except Bar1Unverfuegbar as e:
        logger.info("HTCCL-BAR1: Direktpfad nicht verfuegbar -- %s", e)
        return None
    except NotImplementedError as e:
        logger.info("HTCCL-BAR1: Direktpfad braucht Treiberarbeit -- %s", e)
        return None
    except Exception as e:                 # ein halber Aufbau bleibt nicht stehen
        logger.info("HTCCL-BAR1: Aufbau fehlgeschlagen -- %r", e)
        return None
    # Der Byte-Beleg gehoert zum Aufbau, nicht zur Kuer: ohne ihn ist
    # `handles` gesperrt. Auf diesem Rig meldete der Treiber fuer ein Paar
    # Peer-Zugriff und lieferte 4096 von 1048576 Byte.
    try:
        t.byte_beleg_alle()
    except Exception as e:
        logger.info("HTCCL-BAR1: Byte-Beleg nicht durchfuehrbar -- %r", e)
        t.close()
        return None
    return t
