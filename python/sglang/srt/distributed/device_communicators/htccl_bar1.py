# SPDX-License-Identifier: Apache-2.0
"""HTCCL-Unterpfad: BAR1-Direkttransport (Geruest).

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

Was hier NICHT drin ist
-----------------------
Die Kollektive selbst. Dieses Modul liefert den Punkt-zu-Punkt-Weg
(``put``) und den Messfuehler (``paar``/``paar_empfang``) fuer den Planer
in ``htccl_matrix.py``; die Zerlegung eines Allreduce in Schritte gehoert
in den zusammengesetzten Matrix-Transport. ``handles`` gibt fuer die
Kollektive deshalb ``False`` -- korrekt und ehrlich, statt eine
Zerlegung vorzutaeuschen, die es nicht gibt.
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

    def dmabuf_fd(self, dptr: int, groesse: int) -> int:
        fd = ctypes.c_int(-1)
        fn = getattr(self.drv, "cuMemGetHandleForAddressRange", None)
        if fn is None:
            raise NotImplementedError(
                "cuMemGetHandleForAddressRange fehlt in dieser libcuda. Der "
                "Ersatzweg ueber die RM-Ioctls "
                "(NV0000_CTRL_CMD_OS_UNIX_IMPORT_OBJECT_FROM_FD + "
                "NV_ESC_EXPORT_TO_DMABUF_FD) ist aus Python nicht sinnvoll "
                "nachbaubar -- er braucht die UAPI-Strukturen der "
                "quelloffenen Kernelmodule (nv-ioctl.h, nvos.h, class/cl0000.h) "
                "und ist versionsgebunden. Dafuer fehlt eine native "
                "Hilfsfunktion; sie waere aus sonden/dmabuf_p2p_probe.cpp, "
                "Funktion nvExportToDmabuf(), zu uebernehmen."
            )
        rc = fn(ctypes.byref(fd), ctypes.c_ulonglong(dptr),
                ctypes.c_size_t(groesse),
                ctypes.c_int(CU_MEM_RANGE_HANDLE_TYPE_DMA_BUF_FD),
                ctypes.c_ulonglong(CU_MEM_RANGE_FLAG_DMA_BUF_MAPPING_TYPE_PCIE))
        if rc != 0:
            raise NotImplementedError(
                f"cuMemGetHandleForAddressRange(DMA_BUF_FD) -> {rc}. Auf "
                f"GeForce meldet der Treiber CU_DEVICE_ATTRIBUTE_"
                f"DMA_BUF_SUPPORTED = 0 und liefert CUDA_ERROR_INVALID_VALUE "
                f"(=1), obwohl das Kernelmodul den Export kann "
                f"(nv->dma_buf_supported = 1, osinit.c:671). Der Ausweg ist "
                f"NV0000_CTRL_CMD_OS_UNIX_IMPORT_OBJECT_FROM_FD + "
                f"NV_ESC_EXPORT_TO_DMABUF_FD -- eine native Hilfsfunktion, die "
                f"es hier noch nicht gibt. Vorlage: "
                f"sonden/dmabuf_p2p_probe.cpp::nvExportToDmabuf()."
            )
        return int(fd.value)

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


def fensterbedarf(algorithmus: str, nbytes: int, welt: int,
                  doppelpuffer: bool = True) -> int:
    """Wieviel BAR1 die Zerlegung gleichzeitig abgebildet braucht.

    * **Gechunktes Netz**: ``R-1`` Empfangsschlitze zu je ``N/R``, zusammen
      also rund ``N`` -- **unabhaengig von R**. Genau das macht kleine BARs
      ueberhaupt skalierbar; naiv (ein voller N-Puffer je Nachbar) waeren es
      bei R=8 und N=16 MiB 112 MiB statt 16.
    * **Ring**: zwei Schlitze zu ``N/R``, also ein Bruchteil davon.
    * **Stern**: auf der Nabe ``R-1`` volle Puffer -- der Grund, warum er
      auf 256-MiB-Karten als erstes anschlaegt.

    Doppelpufferung verdoppelt, weil Chunk *i* geschrieben wird, waehrend
    *i-1* noch gelesen wird -- ohne sie gibt es kein Pipelining, und
    Pipelining ist nach der Rechnung der groessere Hebel als die
    Topologiewahl.
    """
    if welt < 2:
        return 0
    f = 2 if doppelpuffer else 1
    anteil = -(-nbytes // welt)          # aufrunden
    if algorithmus == "netz":
        bedarf = (welt - 1) * anteil
    elif algorithmus == "ring":
        bedarf = 2 * anteil
    elif algorithmus == "stern":
        bedarf = (welt - 1) * nbytes
    elif algorithmus == "hierarchisch":
        bedarf = (welt - 1) * anteil     # wie Netz innerhalb der Domaene
    else:
        raise ValueError(f"unbekannter Algorithmus {algorithmus!r}")
    return bedarf * f


# ===========================================================================
# fd-Austausch ueber SCM_RIGHTS
# ===========================================================================


def _tausche_fds(cpu_group, rank: int, welt: int, eigener_fd: int) -> list[int]:
    """Jeder Rang gibt seinen dma-buf-fd an alle anderen.

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

    fds: list[int] = [-1] * welt
    fds[rank] = eigener_fd
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
                        socket.send_fds(verb, [b"x"], [eigener_fd])
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
                    _daten, empfangen, _fl, _adr = socket.recv_fds(verb, 1, 1)
                if not empfangen:
                    raise Bar1Unverfuegbar(
                        f"fd-Austausch: Rang {besitzer} hat keinen fd gesendet"
                    )
                fds[besitzer] = empfangen[0]
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
class PeerZiel:
    """Was der Aufbau je Peer festgestellt hat -- danach unveraenderlich."""

    rang: int
    bdf: str
    bar1_basis: int
    bar1_versatz: int          # Versatz des Empfangspuffers in BAR1
    laenge: int                # abgebildete Laenge
    mmap_obj: object           # gehalten, damit die Abbildung lebt
    reg_adresse: int           # Adresse, unter der REGISTRIERT wurde
    host_adresse: int          # Nutzeradresse des Puffers (reg + Vorlauf)
    dev_ptr: int               # Geraetezeiger DIESER Karte auf die fremde BAR
    halter_handle: int
    byte_beleg: bool = False


class HTCCLBar1Transport:
    """BAR1-Direkttransport.

    Erfuellt die Transport-Naht aus ``htccl.py`` (Zeilen 67-80):
    ``handles(op, nbytes) -> bool`` plus ``htccl_<op>(comm, ...)`` je
    angebotener Operation. Er bietet heute **keine** Kollektive an --
    ``handles`` gibt dafuer ``False``, und die ``htccl_*``-Methoden
    erklaeren, was fehlt, statt eine Zerlegung vorzutaeuschen.

    Was er heute bietet und was echt ist:

    * ``put(ziel, quell_ptr, nbytes, versatz)`` -- ein Schreibvorgang in
      die BAR der Zielkarte.
    * ``paar``/``paar_empfang`` -- der Messfuehler, den
      ``htccl_matrix.HTCCLMatrixPlaner`` fuer echte Kantenkapazitaeten
      braucht (statt der Eigenlast-Schaetzung).
    * ``byte_beleg_alle()`` -- der Byte-Beleg je gerichtetem Paar. Faellt
      er durch, wird die Kante gestrichen, egal was der Treiber meldet.
      Auf diesem Rig meldete der Treiber fuer ein Paar Peer-Zugriff und
      lieferte 4096 von 1048576 Byte.
    """

    HTCCL_OPS: frozenset = frozenset()   # bewusst leer, siehe Klassendoku

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
        self._eigen = (0, 0, 0)          # dptr, handle, groesse
        self._eigen_fuehler = None
        self._dmabuf_fd = -1
        self._fremde_fds: list[int] = []

        if aktiviert is None:
            aktiviert = os.environ.get("SGLANG_HTCCL_MATRIX_DIRECT", "1") not in (
                "0", "nein", "aus", "false"
            )
        if not aktiviert:
            raise Bar1Unverfuegbar(
                "per SGLANG_HTCCL_MATRIX_DIRECT=0 abgeschaltet"
            )
        self.ordinal = device.index if getattr(device, "index", None) is not None \
            else torch.cuda.current_device()
        self._baue_auf()

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

        # 1./2. Empfangspuffer und Export. Das Fenster ist die Groesse, die
        # der Aufrufer aus `fensterbedarf` gerechnet hat.
        dptr, handle, groesse = self._cuda.vmm_alloc(self.ordinal, self.fenster_bytes)
        self._eigen = (dptr, handle, groesse)
        self._dmabuf_fd = self._cuda.dmabuf_fd(dptr, groesse)

        # 3. fds tauschen.
        self._fremde_fds = _tausche_fds(
            self.cpu_group, self.rank, self.welt, self._dmabuf_fd
        )

        # 4.-6. je Peer anhaengen, Versatz aus der sg-Tabelle, mappen,
        # registrieren. Das passiert GENAU HIER und nie wieder.
        for peer in range(self.welt):
            if peer == self.rank:
                continue
            self._peers[peer] = self._binde_peer(peer, self._fremde_fds[peer])

        dist.barrier(group=self.cpu_group)
        self._auf = True
        dauer = time.perf_counter() - t0
        logger.info(
            "HTCCL-BAR1: Aufbau in %.0f ms, %d Peer-Ziele, Fenster %d MiB je "
            "Rang. Ab hier wird im heissen Pfad nichts mehr gemappt.",
            dauer * 1000, len(self._peers), groesse // 2**20,
        )

    def _binde_peer(self, peer: int, fremder_fd: int) -> PeerZiel:
        assert self._cuda is not None and self._halter is not None
        ziel_bdf = self.bdfs[peer]
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
                f"cudaHostRegister(IoMemory) auf die BAR von Rang {peer} "
                f"({ziel_bdf}) fehlgeschlagen: {e}. Das ist der erwartete "
                f"Ausgang OHNE den geweiteten Guard: der Regkey "
                f"RMSmallBarP2PPeerBar1 hat die Vorgabe 0 und muss ueber "
                f"NVreg_RegistryDwords gesetzt werden. Gefunden: "
                f"{ps['regkeys'] or '<leer>'}. Der Transport meldet sich ab, "
                f"er erzwingt nichts."
            ) from e
        dev = self._cuda.dev_ptr(host - vorlauf)
        return PeerZiel(
            rang=peer, bdf=ziel_bdf, bar1_basis=fenster.basis,
            bar1_versatz=versatz, laenge=laenge, mmap_obj=abb,
            reg_adresse=host - vorlauf, host_adresse=host,
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
            if noetig > z.laenge:
                raise Bar1Unverfuegbar(
                    f"Fenster zu klein fuer '{algorithmus}' bei "
                    f"{nbytes // 1024} KiB und {self.welt} Raengen: noetig "
                    f"{noetig // 1024} KiB, abgebildet bei Rang {peer} "
                    f"({z.bdf}) aber nur {z.laenge // 1024} KiB "
                    f"(BAR1 brutto {brutto // 2**20} MiB). Entweder kleiner "
                    f"chunken, den Ring statt des Netzes fahren (zwei Schlitze "
                    f"statt R-1) oder diese Kante ausschliessen."
                )

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
        """Heute immer ``False`` -- und das ist die richtige Antwort.

        Der Transport bewegt Punkt-zu-Punkt-Bytes, er zerlegt keine
        Kollektive. Die Zerlegung gehoert in den zusammengesetzten
        Matrix-Transport, der je gerichteter Kante einen Unterpfad waehlt.
        Bis der existiert, waere ein ``True`` hier eine Zusage, die dieses
        Modul nicht einloesen kann -- und ein Transport, der zusagt und
        dann scheitert, ist schlechter als einer, der sich abmeldet.

        Sobald die Zerlegung da ist, lautet die Bedingung:
        ``op in HTCCL_OPS and self._auf and alle Byte-Belege stehen and
        fensterbedarf(algorithmus, nbytes, welt) <= kleinstes Peer-Fenster``.
        """
        return op in self.HTCCL_OPS and self._auf

    def htccl_all_reduce(self, comm, inp):
        raise NotImplementedError(
            "Der BAR1-Transport zerlegt keine Kollektive. Reduce-Scatter + "
            "Allgather ueber gerichtete Kanten gehoert in den "
            "zusammengesetzten Matrix-Transport, der je Kante einen "
            "Unterpfad waehlt (BAR1 / NIC / System-RAM). Erreichbar ist "
            "diese Zeile nur, wenn jemand handles() umgangen hat."
        )

    htccl_all_gather = htccl_all_reduce
    htccl_reduce_scatter = htccl_all_reduce
    htccl_broadcast = htccl_all_reduce

    # -- Abbau -------------------------------------------------------------

    def close(self) -> None:
        """Alles wieder abbauen. Reihenfolge zaehlt.

        Erst die Registrierungen und Abbildungen der fremden BARs, dann die
        Anhaftungen (die halten die BAR1-Seiten), dann die eigene
        Allokation. Umgekehrt zoege man dem Treiber Seiten unter einer
        lebenden Abbildung weg.
        """
        for z in self._peers.values():
            if self._cuda is not None:
                # ABMELDEN AN DERSELBEN ADRESSE, UNTER DER REGISTRIERT WURDE.
                # cudaHostUnregister auf einen Zeiger MITTEN in der
                # Registrierung schlaegt fehl, und die Abmeldung waere
                # stillschweigend ausgeblieben -- die Apertur bliebe beim
                # naechsten Lauf registriert.
                self._cuda.unregister(z.reg_adresse)
            try:
                z.mmap_obj.close()          # type: ignore[attr-defined]
            except Exception:
                pass
        if self._halter is not None:
            for z in self._peers.values():
                self._halter.gib_frei(z.halter_handle)
            self._halter.schliesse()
            self._halter = None
        self._peers.clear()
        for fd in self._fremde_fds:
            if fd is not None and fd >= 0 and fd != self._dmabuf_fd:
                try:
                    os.close(fd)
                except OSError:
                    pass
        self._fremde_fds = []
        if self._dmabuf_fd >= 0:
            try:
                os.close(self._dmabuf_fd)
            except OSError:
                pass
            self._dmabuf_fd = -1
        if self._cuda is not None and self._eigen[2]:
            self._cuda.vmm_frei(*self._eigen)
            self._eigen = (0, 0, 0)
        self._auf = False


def baue_bar1(cpu_group, device, fenster_bytes: int) -> Optional[HTCCLBar1Transport]:
    """Fabrik mit sauberem Rueckfall.

    ``None`` heisst: dieser Rechner kann den Direktpfad nicht, mit
    protokolliertem Grund. Kein Werfen, kein stilles Ausweichen auf einen
    anderen Weg -- die Wahl des Ersatzpfades trifft der Aufrufer, nicht
    dieses Modul.
    """
    try:
        return HTCCLBar1Transport(cpu_group, device, fenster_bytes)
    except Bar1Unverfuegbar as e:
        logger.info("HTCCL-BAR1: Direktpfad nicht verfuegbar -- %s", e)
        return None
    except NotImplementedError as e:
        logger.info("HTCCL-BAR1: Direktpfad braucht Treiberarbeit -- %s", e)
        return None
