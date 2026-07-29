# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Adapted from the shvllm fork (branch feature/htccl),
# vllm/distributed/device_communicators/htccl.py
"""HTCCL — heterogeneous collective communication layer.

Vendor-neutral collectives for TP groups that span GPUs which share no
common device-native collective library (e.g. NVIDIA/NCCL + AMD/RCCL).
Every collective is executed over the host-staging path that NCCL itself
falls back to when P2P is unavailable:

    GPU (D2H, async) -> pinned host buffer -> gloo collective on the
    group's CPU process group -> (H2D, async) -> GPU

gloo runs entirely CPU-side, so the two endpoints of the collective may
be CUDA and ROCm processes — the device only ever performs plain
``memcpy`` to/from its own pinned staging buffer, which both vendors
implement identically.

Large tensors are processed in chunks and pipelined: while gloo reduces
chunk *i* on the CPU, the D2H copy of chunk *i+1* is already in flight
on the device's copy stream. On systems without P2P between the GPUs
this is functionally the same transport NCCL would use, so forcing
HTCCL on an all-NVIDIA group (``SGLANG_HTCCL=1``) is a faithful test bed
for the mixed-vendor case.

Limitations (v1):
- Collectives synchronize with the CPU, so they cannot be captured in
  CUDA graphs — run with ``--enforce-eager``.
- Reduction happens on the CPU inside gloo (fp32 accumulation for
  half/bfloat16 inputs via upcast).
"""

import logging
import os

import torch
import torch.distributed as dist
from torch.distributed import ProcessGroup

logger = logging.getLogger(__name__)

# Chunk size for the D2H -> gloo -> H2D pipeline. Small enough to
# overlap copy and CPU reduction, large enough to amortize per-op
# latency. Tunable via SGLANG_HTCCL_CHUNK_MIB.
_CHUNK_BYTES = int(os.environ.get("SGLANG_HTCCL_CHUNK_MIB", "8")) * 1024 * 1024

# gloo reduces half/bfloat16 with fp32 accumulation only when the
# tensor is upcast explicitly; reducing bf16 directly through gloo
# accumulates in bf16 and loses precision vs NCCL. Upcast by default,
# disable with SGLANG_HTCCL_FP32_REDUCE=0 to trade accuracy for speed.
_FP32_REDUCE = bool(int(os.environ.get("SGLANG_HTCCL_FP32_REDUCE", "1")))


# Preferred data plane: "shm" (pinned shared-memory slots + GPU-side
# reduction, single-node — matches NCCL's no-P2P SHM path) or "gloo"
# (TCP data plane, also works multi-node; slower).
# "device" = GPU-driven kernels over the mapped segment (fastest,
# CUDA-graph-capturable), "shm" = CPU-orchestrated pinned staging,
# "gloo" = TCP data plane (also multi-node).
_TRANSPORT = os.environ.get("SGLANG_HTCCL_TRANSPORT", "device")
# Per-rank shm slot size; all_reduce payloads above this fall back to
# the gloo path. 64 MiB covers a 4096-token x 5120-hidden bf16 chunk.
_SLOT_BYTES = int(os.environ.get("SGLANG_HTCCL_SLOT_MIB", "64")) * 1024 * 1024


class Bar1Ausgefallen(RuntimeError):
    """Der Direktpfad ist nicht zustande gekommen -- MIT Grund.

    Der Unterschied zu einem stillen ``None`` ist der ganze Punkt: ein
    ``None`` waehlt die gloo-Ebene und sieht danach aus wie ein Erfolg. Diese
    Ausnahme traegt den Grund bis zu ``_build_transport``, das ihn in eine
    Warnung und in ``_STAND`` schreibt.
    """

    def __init__(self, grund: str, stufe: str = "aufbau"):
        super().__init__(grund)
        self.grund = grund
        self.stufe = stufe


#: Was je Gruppe WIRKLICH laeuft. Schluessel ist der Gruppenname
#: (``GroupCoordinator.unique_name``: "tp", "dcp", ...).
#:
#: Es gibt diese Tabelle, weil die Protokollzeile "HTCCL enabled for group
#: 'X' (transport=bar1)" den ANGEFORDERTEN Namen nannte und nicht den
#: erreichten. Am echten Modell hiess das: tp lief ueber BAR1, dcp fiel mit
#: ENOMEM auf gloo zurueck, beide Zeilen sahen gleich aus, und die daraus
#: gewonnene Zahl (22,83 tok/s) war teils gar kein BAR1-Wert. Ein Messwert,
#: dessen Arm man dem Protokoll nicht ansieht, ist kein Messwert.
_STAND: dict[str, dict] = {}


def melde_stand(gruppe: str, angefordert: str, erreicht: str,
                grund: str = "", stufe: str = "") -> dict:
    """Eintragen, was diese Gruppe wirklich fahren wird.

    Ein Eintrag je Gruppenname. Namenlose Gruppen (die es in sglang nicht
    gibt -- ``GroupCoordinator.unique_name`` ist immer gesetzt) bekommen
    einen durchnummerierten Ersatznamen, damit zwei von ihnen sich nicht
    gegenseitig ueberschreiben und eine davon unsichtbar wird.
    """
    schluessel = gruppe
    if not schluessel:
        i = 0
        while f"<ohne Namen #{i}>" in _STAND:
            i += 1
        schluessel = f"<ohne Namen #{i}>"
    eintrag = {
        "gruppe": schluessel,
        "angefordert": angefordert,
        "erreicht": erreicht,
        "grund": grund,
        "stufe": stufe,
        "direkt": erreicht == angefordert and erreicht not in ("gloo", ""),
    }
    _STAND[schluessel] = eintrag
    return eintrag


def gruppen_stand() -> dict[str, dict]:
    """Welche Gruppen wirklich ueber den angeforderten Transport laufen.

    Abfragbar gemacht und nicht nur protokolliert: ein Messprogramm soll das
    PRUEFEN koennen, statt Protokollzeilen zu lesen.
    """
    return dict(_STAND)


def stand_zusammenfassung() -> str:
    """Eine Zeile je Gruppe, fuer das Protokoll und fuer den Messbericht."""
    if not _STAND:
        return "HTCCL: keine Gruppe gemeldet."
    zeilen = []
    for name, e in sorted(_STAND.items()):
        if e["direkt"]:
            zeilen.append(f"  {name}: {e['erreicht']}")
        else:
            zeilen.append(
                f"  {name}: {e['erreicht']} (ANGEFORDERT WAR "
                f"{e['angefordert']}; {e['stufe']}: {e['grund']})"
            )
    voll = all(e["direkt"] for e in _STAND.values())
    kopf = ("HTCCL: alle Gruppen fahren den angeforderten Transport."
            if voll else
            "HTCCL: NICHT alle Gruppen fahren den angeforderten Transport -- "
            "eine Messung ueber diese Konfiguration ist gemischt.")
    return kopf + "\n" + "\n".join(zeilen)


# ----------------------------------------------------------------------
# Transport seam
#
# A transport is anything exposing:
#     handles(op: str, nbytes: int) -> bool
#     htccl_<op>(comm, ...)            for each op it declares
# The communicator ASKS `handles` rather than knowing a transport's limits, so
# no transport-specific condition (e.g. the shm slot-size ceiling) lives at a
# call site. Adding a transport -- `ucx` for RDMA is the expected next one --
# is one registry entry plus its module; no dispatch site changes.
#
# `None` means "no transport object": the inline gloo data plane below, which
# is always available and is the universal fallback.
# ----------------------------------------------------------------------


def _make_device_transport(cpu_group, device):
    from sglang.srt.distributed.device_communicators.htccl_device import (
        HTCCLDeviceTransport,
    )

    return HTCCLDeviceTransport(
        cpu_group=cpu_group, device=device, slot_bytes=_SLOT_BYTES
    )


def _make_shm_transport(cpu_group, device):
    from sglang.srt.distributed.device_communicators.htccl_shm import (
        HTCCLShmTransport,
    )

    return HTCCLShmTransport(
        cpu_group=cpu_group, device=device, slot_bytes=_SLOT_BYTES
    )


def _make_host_transport(cpu_group, device):
    from sglang.srt.distributed.device_communicators.htccl_host import (
        HTCCLHostTransport,
    )

    return HTCCLHostTransport(
        cpu_group=cpu_group, device=device, slot_bytes=_SLOT_BYTES
    )


def _make_ucx_transport(cpu_group, device):
    from sglang.srt.distributed.device_communicators.htccl_ucx import (
        HTCCLUcxTransport,
    )

    return HTCCLUcxTransport(cpu_group=cpu_group, device=device)


def _make_bar1_transport(cpu_group, device, gruppe: str = ""):
    """The BAR1 direct path on its own -- no planner, no measurement.

    The source card DMAs straight into the target card's BAR1 aperture: no
    host memory, no NIC, no NCCL. Which of the two ported kernels runs at
    which size follows the `SGLANG_HTCCL_BAR1_RING_AB` threshold here,
    because there is no plan to ask. That threshold is a default, not a
    finding -- between 1 and 16 MiB mesh and ring measured within 1..7 % of
    each other. Use "matrix" to have that decided by measurement instead.

    `baue_bar1` returns None with a logged reason on any machine that
    cannot do this (holder module absent, driver regkey unset, byte proof
    failed). None is not an error here: it selects the inline gloo plane,
    exactly as an unknown transport name would.
    """
    from sglang.srt.distributed.device_communicators.htccl_bar1 import baue_bar1
    from sglang.srt.distributed.device_communicators.htccl_matrix_transport import (
        fenster_fuer,
    )

    bericht: dict = {}
    t = baue_bar1(cpu_group, device, fenster_fuer(gruppe, device), bericht,
                  gruppe=gruppe)
    if t is None or bericht.get("haelt_belegt"):
        raise Bar1Ausgefallen(
            bericht.get("grund", "ohne gemeldeten Grund"),
            stufe=bericht.get("stufe", "unbekannt"),
        )
    return t


def _make_matrix_transport(cpu_group, device, gruppe: str = ""):
    """Planner + BAR1 direct path.

    Builds the direct path, hands its ACTUALLY mapped window to the planner
    as a capability, runs `plan()`, logs `plan.erklaerung()` on rank 0, and
    feeds the plan back so the kernel choice per size comes from the
    measurement rather than from a built-in number. htccl_matrix_transport
    explains why that is the only order that works.
    """
    from sglang.srt.distributed.device_communicators.htccl_matrix_transport import (
        HTCCLMatrixTransport,
    )

    t = HTCCLMatrixTransport(cpu_group=cpu_group, device=device, gruppe=gruppe)
    if t.bar1 is None:
        # Der Planer allein ist kein Transport: `handles` gaebe fuer alles
        # False, und jedes Kollektiv liefe ueber die gloo-Ebene -- bei einem
        # Protokoll, das "transport=matrix" sagt. Genau die Verwechslung,
        # die hier zu reparieren ist.
        grund = getattr(t, "bar1_grund", "") or "ohne gemeldeten Grund"
        t.close()
        raise Bar1Ausgefallen(grund, stufe=getattr(t, "bar1_stufe", "aufbau"))
    return t


# name -> factory. "gloo" is intentionally absent: it is the inline plane.
TRANSPORT_REGISTRY = {
    "device": _make_device_transport,
    "shm": _make_shm_transport,
    # Pinned, portable host memory, driven entirely by two kernels. The name
    # says where the BYTES live, not who drives: unlike shm/gloo/ucx this one
    # never synchronizes with the host, so it is capturable like "device".
    # Its slot geometry follows SGLANG_HTCCL_SLOT_MIB unless
    # SGLANG_HTCCL_HOST_SLOT_MIB overrides it; the rest of its knobs live in
    # its own module, because they describe its kernels, not the communicator.
    "host": _make_host_transport,
    # RDMA data plane for groups that span hosts. Same host-staged semantics
    # as gloo, UCX instead of TCP. Sizing/threshold knobs live in its own
    # module (SGLANG_HTCCL_UCX_*), not here, because they describe the wire,
    # not the communicator.
    "ucx": _make_ucx_transport,
    # GPU-to-GPU straight through the target's BAR1 aperture. Neither the
    # host nor a NIC touches the payload. Needs the relaxed driver guard
    # (RMSmallBarP2PPeerBar1), the dmabuf_holder module and a passing byte
    # proof; without any of them it opts out cleanly and the gloo plane
    # runs. Its knobs (SGLANG_HTCCL_BAR1_*) live in its own module because
    # they describe its kernels and its BAR1 geometry, not the communicator.
    "bar1": _make_bar1_transport,
    # The same direct path, but with the path-matrix planner deciding role,
    # algorithm and kernel per size from a start-up measurement instead of
    # from a threshold. Strictly more than "bar1"; "bar1" exists so the
    # transport can be measured WITHOUT the planner in the loop.
    "matrix": _make_matrix_transport,
}

# Transports that must NOT silently fall back to the gloo plane on failure.
# The rule is exactly CAPTURABLE_HTCCL_TRANSPORTS: the compilation config
# allowed CUDA graphs on the strength of these, so a CPU-orchestrated
# replacement would be captured and crash later -- and it would crash far from
# the transport that actually failed to come up. "host" is here for the same
# reason "device" is, not for a new one.
#
# "bar1" and "matrix" are deliberately NOT here, and correspondingly not in
# CAPTURABLE_HTCCL_TRANSPORTS. Their data path never touches the host and the
# round number lives in device memory precisely so a replayed graph would not
# reuse a stale one -- so they are capturable BY CONSTRUCTION. What is missing
# is a measurement: nobody has captured a cooperative launch
# (cudaLaunchCooperativeKernel, used above the SGLANG_HTCCL_BAR1_GITTER_AB
# threshold) into a CUDA graph on this rig and replayed it. Claiming
# capturability on a construction argument is exactly the kind of plausible
# assumption that has been failing against this hardware all day.
_NO_FALLBACK = frozenset({"device", "host"})


def _kein_ausweichen(name: str) -> bool:
    """Ob ``name`` bei einem Aufbaufehler werfen muss statt auszuweichen.

    Die Regel ist unveraendert "genau die capturable Menge" -- nur wird sie
    jetzt bei ``parallel_state`` erfragt statt hier ein zweites Mal
    hingeschrieben. Sonst wirkte der Freigabeschalter an einer Stelle und an
    der anderen nicht, und bar1 waere freigegeben UND wiche still nach gloo
    aus: die schlechteste denkbare Verbindung der beiden.
    """
    if name in _NO_FALLBACK:
        return True
    try:
        from sglang.srt.distributed.parallel_state import capturable_transports

        return name in capturable_transports()
    except Exception:
        return False


def _ruf_fabrik(factory, cpu_group, device, gruppe: str):
    """Die Fabrik aufrufen, mit Gruppennamen nur wenn sie ihn annimmt.

    Zwei Fabriken (bar1, matrix) brauchen ihn -- fuer die Fenstergroesse je
    Gruppe. Die anderen und jede von aussen registrierte Fabrik haben
    weiterhin die zweistellige Form. Geprueft wird die Signatur, nicht ein
    ``TypeError`` abgefangen: ein TypeError AUS der Fabrik heraus saehe
    genauso aus und wuerde stillschweigend als "alte Form" gedeutet.
    """
    import inspect

    try:
        nimmt = "gruppe" in inspect.signature(factory).parameters
    except (TypeError, ValueError):
        nimmt = False
    if nimmt:
        return factory(cpu_group, device, gruppe=gruppe)
    return factory(cpu_group, device)


def _build_transport(name: str, cpu_group, device, disabled: bool,
                     gruppe: str = ""):
    if disabled:
        melde_stand(gruppe, name, "keiner (world_size 1)")
        return None
    factory = TRANSPORT_REGISTRY.get(name)
    if factory is None:
        melde_stand(gruppe, name, "gloo",
                    grund="kein solcher Name in TRANSPORT_REGISTRY",
                    stufe="auswahl")
        return None  # "gloo" or an unknown name -> inline data plane
    if _kein_ausweichen(name):
        t = _ruf_fabrik(factory, cpu_group, device, gruppe)
        melde_stand(gruppe, name, name)
        return t
    try:
        t = _ruf_fabrik(factory, cpu_group, device, gruppe)
    except Exception as e:
        stufe = getattr(e, "stufe", "aufbau")
        grund = getattr(e, "grund", f"{type(e).__name__}: {e}")
        melde_stand(gruppe, name, "gloo", grund=grund, stufe=stufe)
        # WARNING, nicht INFO, und mit Gruppennamen. Der Ausfall EINER Gruppe
        # ist kein Randfall: er macht jede Messung ueber diesen Lauf zu einer
        # gemischten, und genau das ist heute unbemerkt passiert.
        logger.warning(
            "HTCCL: Gruppe %r bekommt den angeforderten Transport %r NICHT "
            "(%s: %s). Diese Gruppe faehrt ueber die host-gestaffelte "
            "gloo-Ebene. Jede Messung ueber diesen Lauf ist damit gemischt "
            "und darf NICHT als %r-Wert berichtet werden.",
            gruppe or "<ohne Namen>", name, stufe, grund, name,
        )
        return None
    if t is None:
        melde_stand(gruppe, name, "gloo",
                    grund="Fabrik lieferte None ohne Grund", stufe="aufbau")
        logger.warning(
            "HTCCL: Gruppe %r bekommt den angeforderten Transport %r nicht "
            "(die Fabrik lieferte None). gloo-Ebene.",
            gruppe or "<ohne Namen>", name,
        )
        return None
    melde_stand(gruppe, name, name)
    return t


def graph_erfassung_laeuft() -> bool:
    """``True``, solange der AKTUELLE Strom in einen CUDA-Graphen aufgezeichnet
    wird.

    **Eine** Definition, hier und nicht je Modul nachgebaut: sie entscheidet
    an drei Stellen (Ausweichriegel unten, Kernvariante und Direkt-Modus in
    ``htccl_bar1``), und zwei Fassungen derselben Frage waeren die Stelle, an
    der sie auseinanderlaufen.

    RANGEINHEITLICH nur, soweit die Aufzeichnung es ist -- und das ist sie in
    sglang: der Graph-Laeufer zeichnet auf allen Raengen dieselben Formen in
    derselben Reihenfolge auf. Wer diese Funktion benutzt, um eine
    KOLLEKTIVE Entscheidung zu treffen, verlaesst sich genau darauf; wo das
    nicht gilt, ist die Entscheidung falsch am Platz.

    Nie eine Ausnahme: ohne initialisiertes CUDA wirft
    ``is_current_stream_capturing`` (``cudaErrorNoDevice``), und diese Frage
    darf keinen Aufrufer umbringen, der sie nur vorsorglich stellt.
    """
    try:
        if not torch.cuda.is_available() or not torch.cuda.is_initialized():
            return False
        return bool(torch.cuda.is_current_stream_capturing())
    except Exception:
        return False


def _transport_name(t) -> str:
    """Der Name eines Transports fuer Fehlertexte, ohne je selbst zu werfen.

    ``name()`` ist die Zusage der Transport-Schnittstelle, aber ein Transport,
    der gerade als Ursache einer Fehlermeldung benannt werden soll, ist der
    letzte, dem man einen Aufruf zutrauen sollte.
    """
    try:
        holen = getattr(t, "name", None)
        if callable(holen):
            return str(holen())
    except Exception:  # noqa: BLE001 - ein Name darf nie der Grund sein
        pass
    return type(t).__name__


def _gedeckte_ops(t) -> str:
    """Die Operationen, die ``t`` ueberhaupt anbietet -- aus DER Quelle.

    Gelesen wird ``HTCCL_OPS`` des Transports selbst, nie eine im Fehlertext
    mitgefuehrte Liste. Eine Liste im Text waere genau die Sorte Zusage, die
    beim naechsten hinzugebauten Kollektiv veraltet, und dann sagt die
    Meldung "es fehlt X", waehrend X laengst da ist und etwas anderes klemmt.
    """
    ops = getattr(t, "HTCCL_OPS", None)
    if not ops:
        return "unbekannt (der Transport nennt kein HTCCL_OPS)"
    return ", ".join(sorted(str(o) for o in ops))


def _zeilen_bytes(t: torch.Tensor) -> int:
    """Bytes einer Zeile entlang Achse 0. Fuer 1-D ist das ein Element."""
    n = 1
    for d in t.shape[1:]:
        n *= int(d)
    return n * t.element_size()


def _gruppen_max(wert: int, cpu_group) -> int:
    """Maximum ueber die Gruppe, auf der CPU.

    Nur fuer den Fall gedacht, dass der Aufrufer BEIDE Teilgroessenlisten
    selbst mitbringt: dann kennt kein Rang die Bloecke der anderen Paare,
    und die Schlitzentscheidung muss trotzdem rangeinheitlich ausfallen.
    Ein int64 ueber gloo -- messbar teuer im Verhaeltnis zu einem
    MoE-Dispatch, aber billiger als ein Haenger, und der gleichverteilte
    Fall braucht es gar nicht.
    """
    t = torch.tensor([int(wert)], dtype=torch.int64)
    dist.all_reduce(t, op=dist.ReduceOp.MAX, group=cpu_group)
    return int(t.item())


class HTCCLCommunicator:
    """Host-staged collectives over the group's gloo CPU process group."""

    def __init__(
        self,
        cpu_group: ProcessGroup,
        device: torch.device,
        gruppe: str = "",
    ):
        self.cpu_group = cpu_group
        self.device = device
        self.gruppe = gruppe
        self.world_size = dist.get_world_size(cpu_group)
        self.rank = dist.get_rank(cpu_group)
        self.disabled = self.world_size == 1
        self.transport = _build_transport(
            _TRANSPORT, cpu_group, device, disabled=self.disabled, gruppe=gruppe,
        )
        #: Was diese Gruppe WIRKLICH faehrt -- nicht, was angefordert war.
        #: Namenlos heisst: der Eintrag, den `_build_transport` gerade
        #: angelegt hat, also der zuletzt eingefuegte.
        self.stand = (
            _STAND.get(gruppe, {}) if gruppe
            else (list(_STAND.values())[-1] if _STAND else {})
        )
        # #279 path dispatcher (skeleton, flag-gated, default None). With an
        # empty registry every decision is status quo, so building it does
        # not change any selection -- see htccl_path_dispatcher.
        from sglang.srt.distributed.device_communicators.htccl_path_dispatcher import (
            maybe_build_dispatcher,
        )

        self._path_dispatcher = None if self.disabled else maybe_build_dispatcher()
        # Dedicated copy stream: D2H of the next chunk overlaps with the
        # CPU-side gloo reduction of the current one.
        self._stream = torch.cuda.Stream(device=device)
        # Pinned staging buffers, grown on demand and reused. Two
        # buffers per direction so chunk i+1 can stage while chunk i is
        # still being reduced/written back.
        self._host_bufs: list[torch.Tensor] = []
        self._host_buf_bytes = 0

    def _select(self, op: str, nbytes: int):
        """The transport for ``op`` at this size, or None for the gloo plane.

        One attribute test plus the transport's own `handles` -- the same shape
        at every dispatch site, so op coverage can no longer differ silently
        between ops the way it did when each site hard-coded its own condition.

        **Und genau hier faellt der Ausweichriegel.** Ein ``None`` heisst
        "gloo-Ebene", und die gloo-Ebene ist host-gestaffelt: gepinnte
        Allokation, ``dist.*`` auf der CPU, ``Event.synchronize()``. Innerhalb
        einer CUDA-Graph-Aufzeichnung ist das entweder ein Abbruch mit einem
        Fehler, der nach etwas ganz anderem aussieht -- oder, schlimmer, eine
        CPU-Reduktion, die EINMAL zur Aufzeichnungszeit laeuft und bei jeder
        Wiedergabe fehlt: falsche Zahlen ohne Absturz.

        Das ist kein hypothetischer Fall. Ein Transport wie ``bar1`` deckt
        nicht jede Operation ab (``HTCCL_OPS`` in ``htccl_bar1.py``) und sagt
        selbst zu einer gedeckten unterhalb von ``min_bytes``, bei
        ``nbytes % 16 != 0`` oder oberhalb des abgebildeten Fensters ``False``.
        Unter Aufzeichnung landete jeder dieser Faelle lautlos in der Schleife
        weiter unten.

        Deshalb: unter Aufzeichnung gibt es kein Ausweichen, sondern eine
        Ansage mit Grund.

        **Zwei Mechanismen, in dieser Reihenfolge.** Der #279-Pfad-Dispatcher
        darf die Klassenwahl noch verfeinern, der Riegel huetet danach die
        ENDGUELTIGE Wahl:

        1. ``handles`` liefert die Klassenwahl (#240).
        2. ``refine_transport_choice`` verfeinert sie (#279). Ohne gemessene
           Raten ist jede Entscheidung Status quo, gibt also unveraendert
           zurueck -- die Platzhalter-Neutralitaet, die
           ``test_htccl_path_dispatcher.py`` festnagelt, bleibt exakt erhalten,
           weil ausserhalb einer Aufzeichnung nach Schritt 2 nichts mehr
           passiert.
        3. Erst dann der Riegel. Die Reihenfolge ist nicht beliebig: ein
           ``HINT_GLOO`` kann die Wahl auch dann noch auf ``None`` setzen, wenn
           ``handles`` zugesagt hatte. Vor dem Dispatcher haette der Riegel
           genau diesen einen Fall durchgelassen -- den einzigen, in dem eine
           gemessene Entscheidung unter Aufzeichnung in die gloo-Ebene fuehrt.
        """
        t = self.transport
        chosen = t if (t is not None and t.handles(op, nbytes)) else None
        dispatcher = getattr(self, "_path_dispatcher", None)
        if dispatcher is not None:
            # Thin #279 hook onto the existing #240 class choice: status-quo
            # decisions (today: all of them) return `chosen` unchanged.
            from sglang.srt.distributed.device_communicators.htccl_path_dispatcher import (  # noqa: E501
                refine_transport_choice,
            )

            chosen = refine_transport_choice(dispatcher, op, nbytes, chosen)
        if chosen is None and graph_erfassung_laeuft():
            # Der Riegel sitzt hinter dem Dispatcher, nicht davor: er huetet
            # die ENDGUELTIGE Wahl. Ein ``HINT_GLOO`` kann `chosen` auch dann
            # noch auf None setzen, wenn `handles` zugesagt hatte -- und genau
            # dieser Fall waere sonst der eine, der unter Aufzeichnung
            # ungeriegelt in die gloo-Ebene faellt.
            if t is None:
                grund = "es ist ueberhaupt kein Transport aufgebaut"
            elif t.handles(op, nbytes):
                grund = (
                    f"{_transport_name(t)} kann es, aber der Pfad-Dispatcher "
                    f"hat auf die gloo-Ebene entschieden"
                )
            else:
                grund = (
                    f"{_transport_name(t)} meldet handles({op!r}, {nbytes}) "
                    f"-> False; gedeckt sind dort {_gedeckte_ops(t)}"
                )
            raise RuntimeError(
                f"HTCCL: {op!r} mit {nbytes} Byte waehrend einer "
                f"CUDA-Graph-Aufzeichnung, aber {grund}. Der Ausweichweg ist "
                f"die host-gestaffelte gloo-Ebene (gepinnte Allokation, "
                f"dist.* auf der CPU, Event.synchronize()) -- die laeuft "
                f"EINMAL beim Aufzeichnen und bei keiner Wiedergabe wieder. "
                f"Das gaebe falsche Zahlen ohne Absturz, also bricht es hier "
                f"ab. Abhilfe: --disable-cuda-graph, oder einen Transport "
                f"waehlen, der diese Operation in dieser Groesse wirklich "
                f"fahren kann (SGLANG_HTCCL_TRANSPORT=device deckt "
                f"all_reduce/all_gather/reduce_scatter/broadcast lueckenlos "
                f"ab)."
            )
        return chosen

    def _get_out_buf(self, ref: torch.Tensor) -> torch.Tensor:
        """One FRESH output tensor per call — never a shape-keyed cache.

        This used to hand out a persistent per-(shape, dtype) buffer, on the
        theory that a piecewise CUDA graph captured downstream of the
        collective needs the result at a stable address. That reasoning does
        not survive contact with either path that exists:

        * the CPU transports (shm/gloo) synchronize with the host inside the
          collective, so they can never be inside a captured region at all;
        * the one graph-capturable transport (`htccl_device`) does NOT use
          this helper — it allocates with `torch.empty_like` per call, and is
          correct precisely because capture-time allocations come from the
          graph's private pool and therefore already replay at a stable
          address.

        Meanwhile the cache actively BREAKS the documented contract of
        `all_reduce` ("returns a new tensor, out-of-place"): two results of
        the same shape and dtype were the SAME tensor, so the second call
        silently overwrote the first while the model still held it. That is
        not a hypothetical — it corrupted the forward outright (garbage
        tokens, no crash, no hang) on every non-device transport.
        """
        return torch.empty_like(ref)

    def _get_host_bufs(self, nbytes: int, count: int = 2) -> list[torch.Tensor]:
        if self._host_buf_bytes < nbytes or len(self._host_bufs) < count:
            self._host_bufs = [
                torch.empty(nbytes, dtype=torch.uint8, pin_memory=True)
                for _ in range(count)
            ]
            self._host_buf_bytes = nbytes
        return self._host_bufs

    # ------------------------------------------------------------------
    # all_reduce
    # ------------------------------------------------------------------

    # ------------------------------------------------------------------
    # async all_reduce (issue/wait split; ucx transport only today)
    #
    # supports_async() is deliberately shaped like handles(): its answer
    # depends only on group-uniform state (env-selected transport, class of
    # that transport), never on the payload -- so no rank can decide to go
    # async while a peer goes sync. Callers must treat a None from
    # all_reduce_async as "issue unavailable" and fall back to the sync
    # all_reduce; wait_async must be called exactly once per handle.
    # ------------------------------------------------------------------

    def supports_async(self) -> bool:
        t = self.transport
        return (
            not self.disabled
            and t is not None
            and hasattr(t, "all_reduce_async")
            and t.handles("all_reduce", 0)
        )

    def all_reduce_async(self, input_: torch.Tensor):
        """Issue a sum-all-reduce; returns a handle for wait_async, or None.

        None means the async path is unavailable here (no transport, or the
        transport has no async support) -- the caller runs the sync
        all_reduce instead. That decision is group-uniform by construction
        (see supports_async).
        """
        if not self.supports_async():
            return None
        return self.transport.all_reduce_async(self, input_.contiguous())

    def wait_async(self, handle) -> torch.Tensor:
        """Complete an all_reduce_async handle; returns the fresh result."""
        return self.transport.wait_async(handle)

    def all_reduce(self, input_: torch.Tensor) -> torch.Tensor:
        """Sum-all-reduce ``input_`` across the group, host-staged.

        Returns a new tensor (out-of-place), matching the contract of
        the other vLLM all-reduce backends.
        """
        if self.disabled:
            return input_.clone()
        inp = input_.contiguous()
        nbytes = inp.numel() * inp.element_size()
        t = self._select("all_reduce", nbytes)
        if t is not None:
            return t.htccl_all_reduce(self, inp)
        out = self._get_out_buf(inp)

        reduce_dtype = (
            torch.float32
            if _FP32_REDUCE and inp.dtype in (torch.float16, torch.bfloat16)
            else inp.dtype
        )
        elem_bytes = torch.tensor([], dtype=reduce_dtype).element_size()
        chunk_elems = max(_CHUNK_BYTES // elem_bytes, 1)

        flat_in = inp.view(-1)
        flat_out = out.view(-1)
        n = flat_in.numel()
        n_chunks = (n + chunk_elems - 1) // chunk_elems

        bufs = self._get_host_bufs(min(n, chunk_elems) * elem_bytes)
        staged: list[tuple[int, int, torch.Tensor, torch.cuda.Event]] = []

        current = torch.cuda.current_stream(self.device)
        self._stream.wait_stream(current)

        def _stage(ci: int) -> None:
            start = ci * chunk_elems
            end = min(start + chunk_elems, n)
            host = (
                bufs[ci % len(bufs)][: (end - start) * elem_bytes]
                .view(reduce_dtype)[: end - start]
            )
            with torch.cuda.stream(self._stream):
                src = flat_in[start:end]
                if src.dtype != reduce_dtype:
                    src = src.to(reduce_dtype)
                host.copy_(src, non_blocking=True)
                ev = torch.cuda.Event()
                ev.record(self._stream)
            staged.append((start, end, host, ev))

        _stage(0)
        for ci in range(n_chunks):
            if ci + 1 < n_chunks:
                _stage(ci + 1)  # D2H of next chunk overlaps gloo below
            start, end, host, ev = staged[ci]
            ev.synchronize()
            dist.all_reduce(host, group=self.cpu_group)
            with torch.cuda.stream(self._stream):
                dst = flat_out[start:end]
                if host.dtype != dst.dtype:
                    dst.copy_(host.to(dst.dtype), non_blocking=False)
                else:
                    dst.copy_(host, non_blocking=True)

        current.wait_stream(self._stream)
        return out

    # ------------------------------------------------------------------
    # all_gather / reduce_scatter / broadcast
    # ------------------------------------------------------------------

    def all_gather(self, input_: torch.Tensor, dim: int = -1) -> torch.Tensor:
        if self.disabled:
            return input_
        t = self._select(
            "all_gather", input_.numel() * input_.element_size()
        )
        if t is not None:
            return t.htccl_all_gather(self, input_, dim)
        if dim < 0:
            dim += input_.dim()
        inp = input_.contiguous()
        input_size = inp.size()

        host_in = torch.empty(
            inp.shape, dtype=inp.dtype, pin_memory=True
        )
        host_in.copy_(inp, non_blocking=False)
        host_out = [torch.empty_like(host_in) for _ in range(self.world_size)]
        dist.all_gather(host_out, host_in, group=self.cpu_group)

        output = torch.empty(
            (self.world_size,) + tuple(input_size),
            dtype=inp.dtype,
            device=inp.device,
        )
        for i, h in enumerate(host_out):
            output[i].copy_(h, non_blocking=True)
        torch.cuda.current_stream(self.device).synchronize()

        output = output.movedim(0, dim)
        return output.reshape(
            input_size[:dim]
            + (self.world_size * input_size[dim],)
            + input_size[dim + 1 :]
        )

    def reduce_scatter(self, input_: torch.Tensor, dim: int = -1) -> torch.Tensor:
        if self.disabled:
            return input_
        t = self._select(
            "reduce_scatter", input_.numel() * input_.element_size()
        )
        if t is not None:
            return t.htccl_reduce_scatter(self, input_, dim)
        if dim < 0:
            dim += input_.dim()
        # Host-staged: full all-reduce, then slice this rank's shard.
        # For the small TP world sizes HTCCL targets (2-4 ranks) the
        # extra traffic vs a true reduce-scatter is bounded and the
        # code stays trivially correct. Axis handling mirrors the base
        # communicator's reduce_scatter exactly.
        reduced = self.all_reduce(input_)
        # movedim(dim, 0) -- NOT movedim(0, dim). The two agree only for dim
        # in {0, 1}; from dim >= 2 they differ, and the old form left the
        # ORIGINAL axis 1 in front, so the scatter sliced the wrong axis while
        # every shape check still passed. Measured: shape (4,6,2) dim=2 sliced
        # 6 instead of 2; (2,4,6,2) dim=3 sliced 4 instead of 2. The signature
        # defaults to dim=-1, so a bare reduce_scatter(x) on ndim >= 3
        # scattered the wrong axis SILENTLY. 2-D happened to be correct, which
        # is why it survived.
        moved = reduced.movedim(dim, 0).contiguous()
        assert moved.shape[0] % self.world_size == 0
        chunk = moved.shape[0] // self.world_size
        shard = moved[self.rank * chunk : (self.rank + 1) * chunk]
        return shard.movedim(0, dim).contiguous()

    # ------------------------------------------------------------------
    # all_to_all_single
    #
    # WER DAS WIRKLICH RUFT -- nachgesehen, nicht angenommen:
    #
    # * `GroupCoordinator.all_to_all_single(output, input)`
    #   (parallel_state.py:1199 -> :1196) ist die EINZIGE
    #   torch.distributed-a2a-Stelle im ganzen srt-Baum. Sie ist
    #   ausser-Ort, gleichverteilt, ohne Teilgroessen, ohne
    #   scatter/gather-Achse -- und sie hat heute keinen Aufrufer
    #   (Upstream-Oberflaeche aus #27492).
    # * Die MoE-Token-Dispatcher rufen NICHT hierher. deepep.py:578
    #   `buffer.dispatch(...)`, mooncake.py:236, nixl.py:293, moriep.py:724
    #   und flashinfer.py:259 `moe_a2a.dispatch(...)` gehen an
    #   torch.distributed vorbei in ihre eigenen Bibliotheken. Deren
    #   Semantik ist aber genau die ungleich geteilte: sie reichen
    #   `num_tokens_per_rank`/`num_tokens_per_expert` herein, weil die Zahl
    #   der Token je Experte schwankt.
    #
    # Daraus folgt die Signatur hier: die von `torch.distributed.
    # all_to_all_single`, also die gleichverteilte Form des einzigen echten
    # Aufrufers PLUS die Teilgroessen, ohne die MoE nicht abbildbar waere.
    # Geraten ist daran nichts -- die gleichverteilte Form ist ein
    # Sonderfall (beide Listen None) und laeuft denselben Weg.
    # ------------------------------------------------------------------

    def all_to_all_zaehlwerte(self, input_split_sizes) -> list[list[int]]:
        """Die volle R x R Zaehlwertmatrix aus den eigenen Sendezahlen.

        ``matrix[i][j]`` = was Rang i an Rang j schickt. Ein
        ``all_gather_object`` ueber die CPU-Gruppe -- genau der Schritt, den
        DeepEP vor dem Dispatch macht (``get_dispatch_layout`` ->
        ``num_tokens_per_rank``), und aus demselben Grund: der Empfaenger
        kann seine Puffergroesse nicht kennen, bevor der Sender gezaehlt hat.

        Das ist ein **Host-Kollektiv**. Es steht vor dem Datenpfad, nicht
        darin, und es ist der Grund, warum der ungleich geteilte Fall nicht
        CUDA-Graph-faehig ist. Der gleichverteilte Fall braucht ihn nicht
        und ist es deshalb.
        """
        matrix: list = [None] * self.world_size
        dist.all_gather_object(
            matrix, [int(x) for x in input_split_sizes], group=self.cpu_group
        )
        return [[int(v) for v in row] for row in matrix]  # type: ignore[union-attr]

    def all_to_all_single(
        self,
        output: torch.Tensor,
        input_: torch.Tensor,
        output_split_sizes=None,
        input_split_sizes=None,
    ) -> torch.Tensor:
        """``torch.distributed.all_to_all_single`` ueber HTCCL.

        Teilt ``input_`` entlang Achse 0 in ``world_size`` Bloecke, schickt
        Block j an Rang j und legt die empfangenen Bloecke in derselben
        Reihenfolge in ``output`` ab. ``output`` wird beschrieben und
        zurueckgegeben.

        ``*_split_sizes`` sind **Zeilenzahlen**, nicht Bytes -- wie bei
        torch. ``None`` heisst gleichverteilt. Ist nur
        ``input_split_sizes`` gegeben, werden die Empfangszahlen ueber
        :meth:`all_to_all_zaehlwerte` beschafft; ``output`` muss dann
        bereits gross genug sein.
        """
        if self.disabled:
            output.copy_(input_)
            return output
        inp = input_.contiguous()
        if inp.dim() == 0 or output.dim() == 0:
            raise ValueError("all_to_all_single braucht mindestens eine Achse")

        zeile_elems = 1
        for d in inp.shape[1:]:
            zeile_elems *= int(d)
        zeile_bytes = zeile_elems * inp.element_size()
        if zeile_bytes != _zeilen_bytes(output):
            raise ValueError(
                f"all_to_all_single: Zeilenbreite passt nicht -- Eingabe "
                f"{zeile_bytes} Byte, Ausgabe {_zeilen_bytes(output)} Byte. "
                f"Nur Achse 0 wird geteilt."
            )

        w = self.world_size
        # Die Zaehlwerte. Reihenfolge: erst besorgen, was der RUECKFALL auch
        # braucht -- sonst haengt an der Transportwahl, ob ein Kollektiv
        # laeuft, und das ist genau die Sorte Rangabhaengigkeit, die haengt.
        matrix = None
        if input_split_sizes is None:
            if inp.shape[0] % w:
                raise ValueError(
                    f"all_to_all_single ohne input_split_sizes braucht eine "
                    f"durch {w} teilbare Achse 0, hat aber {inp.shape[0]}."
                )
            ein = [inp.shape[0] // w] * w
        else:
            ein = [int(x) for x in input_split_sizes]
            if len(ein) != w or sum(ein) != inp.shape[0]:
                raise ValueError(
                    f"input_split_sizes {ein} passt nicht zu Achse 0 = "
                    f"{inp.shape[0]} bei {w} Raengen."
                )
        if output_split_sizes is None:
            if input_split_sizes is None:
                if output.shape[0] % w:
                    raise ValueError(
                        f"all_to_all_single ohne output_split_sizes braucht "
                        f"eine durch {w} teilbare Achse 0 der Ausgabe, hat "
                        f"aber {output.shape[0]}."
                    )
                aus = [output.shape[0] // w] * w
            else:
                matrix = self.all_to_all_zaehlwerte(ein)
                aus = [matrix[i][self.rank] for i in range(w)]
        else:
            aus = [int(x) for x in output_split_sizes]
            if len(aus) != w:
                raise ValueError(f"output_split_sizes hat Laenge {len(aus)}")
        if sum(aus) > output.shape[0]:
            raise ValueError(
                f"Ausgabe traegt {output.shape[0]} Zeilen, empfangen werden "
                f"aber {sum(aus)}."
            )

        sende = [n * zeile_bytes for n in ein]
        empf = [n * zeile_bytes for n in aus]
        nbytes = sum(sende)

        t = self._select("all_to_all", nbytes)
        if t is not None and not (
            hasattr(t, "htccl_all_to_all_single") and hasattr(t, "traegt_a2a")
        ):
            # handles() hat zugesagt, aber die Methoden fehlen. Das ist ein
            # Fehler IM TRANSPORT und kein Laufzeitzustand: er faellt auf
            # jedem Rang gleich aus, weil die Klasse rangeinheitlich ist.
            # Also ist der Rueckfall sicher -- und die Warnung nennt den
            # Schuldigen, statt ihn im Rueckfall verschwinden zu lassen.
            logger.warning(
                "HTCCL: Transport %s sagt handles('all_to_all') zu, hat aber "
                "keine htccl_all_to_all_single/traegt_a2a. Es laeuft die "
                "CPU-Ebene.", type(t).__name__,
            )
            t = None
        if t is not None:
            # Die genaue Schlitzpruefung braucht den groessten Block ueber
            # ALLE Paare, nicht ueber die eigene Zeile -- sonst koennte ein
            # Rang zusagen und ein anderer absagen, und daraus wird ein
            # Haenger statt eines Fehlers. Gleichverteilt ist das Maximum auf
            # jedem Rang dieselbe Zahl und es braucht kein Kollektiv; sonst
            # kommt es aus der Zaehlwertmatrix oder, wenn der Aufrufer beide
            # Listen selbst mitgebracht hat, aus einem Maximum ueber die
            # Gruppe.
            if input_split_sizes is None and output_split_sizes is None:
                groesster = max(sende + empf)
            elif matrix is not None:
                groesster = max(max(z) for z in matrix) * zeile_bytes
            else:
                groesster = _gruppen_max(
                    max(sende + empf), self.cpu_group
                )
            if t.traegt_a2a(groesster):
                return t.htccl_all_to_all_single(self, output, inp, sende, empf)

        # Rueckfall: dieselbe Zerlegung ueber die CPU-Gruppe. Gepinnt, damit
        # die beiden Kopien nicht ueber einen pageable-Zwischenpuffer laufen.
        host_in = torch.empty(inp.shape, dtype=inp.dtype, pin_memory=True)
        host_in.copy_(inp, non_blocking=False)
        host_out = torch.empty(
            (sum(aus),) + tuple(output.shape[1:]),
            dtype=output.dtype, pin_memory=True,
        )
        try:
            dist.all_to_all_single(
                host_out, host_in,
                output_split_sizes=aus, input_split_sizes=ein,
                group=self.cpu_group,
            )
        except (RuntimeError, NotImplementedError) as e:
            raise NotImplementedError(
                f"all_to_all_single: weder der BAR1-Direktpfad noch die "
                f"CPU-Ebene der Gruppe koennen diesen Aufruf fahren ({e}). "
                f"Kein stiller Rueckfall auf NCCL: auf einer Gruppe ueber "
                f"zwei Hersteller ist das kein langsamerer Weg, sondern ein "
                f"Haenger."
            ) from e
        output[: sum(aus)].copy_(host_out, non_blocking=False)
        return output

    # ------------------------------------------------------------------
    # out-parameter forms
    #
    # sglang calls these directly (they are NOT reachable from the dim-based
    # variants above), so leaving them out would silently route part of the
    # traffic back to NCCL -- which on a mixed-vendor group is not a slow
    # path but a hang. Both are pure compositions of the collectives above:
    # they introduce NO new collective, which keeps the rank-uniformity
    # argument unchanged.
    # ------------------------------------------------------------------

    def all_gather_into_tensor(
        self, output: torch.Tensor, input_: torch.Tensor
    ) -> None:
        """`output[i*n:(i+1)*n] = input_` of rank i, matching
        torch.distributed.all_gather_into_tensor."""
        if self.disabled:
            output.copy_(input_)
            return
        # dim=0 concatenation IS the [world, n]-flattened layout this API
        # specifies, so no extra transposition is needed.
        gathered = self.all_gather(input_, dim=0)
        output.copy_(gathered.reshape(output.shape))

    def reduce_scatter_tensor(
        self, output: torch.Tensor, input_: torch.Tensor
    ) -> None:
        """Sum-reduce `input_` and scatter along dim 0 into `output`,
        matching torch.distributed.reduce_scatter_tensor."""
        if self.disabled:
            output.copy_(input_)
            return
        shard = self.reduce_scatter(input_, dim=0)
        output.copy_(shard.reshape(output.shape))

    def broadcast(self, tensor: torch.Tensor, src: int = 0) -> torch.Tensor:
        if self.disabled:
            return tensor
        # A transport that can broadcast on-device does so: the host-staged
        # path below synchronizes with the host and is therefore ILLEGAL
        # inside a CUDA-graph capture, which the speculative draft-pick sync
        # performs. See HTCCLDeviceTransport.htccl_broadcast.
        t = self._select("broadcast", tensor.numel() * tensor.element_size())
        if t is not None:
            return t.htccl_broadcast(self, tensor, src)
        host = torch.empty(tensor.shape, dtype=tensor.dtype, pin_memory=True)
        if self.rank == src:
            host.copy_(tensor, non_blocking=False)
        dist.broadcast(host, src=dist.get_global_rank(self.cpu_group, src),
                       group=self.cpu_group)
        if self.rank != src:
            tensor.copy_(host, non_blocking=False)
        return tensor

    # ------------------------------------------------------------------
    # teardown
    # ------------------------------------------------------------------

    def close(self) -> None:
        """Release the POSIX shm segment backing the shm/device transports.

        Without this the segment survives the process and leaks a /dev/shm
        entry per run (rank 0 owns the unlink). Called from
        GroupCoordinator.destroy().
        """
        if self.transport is None:
            return
        try:
            self.transport.close()
        except Exception as e:  # teardown must never mask the real error
            logger.warning("HTCCL: transport close failed (%s).", e)
