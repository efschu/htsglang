# SPDX-License-Identifier: Apache-2.0
"""MoE-Token-Dispatcher ueber den BAR1-Direktpfad.

WARUM DIESE DATEI UEBERHAUPT EXISTIERT
--------------------------------------
``all_to_all`` in die barlink-Naht zu bauen war richtig und **wirkungslos**:
die MoE-Dispatcher rufen dort nicht an. Nachgesehen, nicht angenommen --
``deepep.py:578`` ``buffer.dispatch(...)``, ``flashinfer.py:259``
``moe_a2a.dispatch(...)``, ``mooncake.py:236``, ``nixl.py:293``,
``moriep.py:724`` gehen alle an ``torch.distributed`` vorbei in ihre eigene
Bibliothek. Wer an die MoE-Last will, muss **einen Dispatcher schreiben**,
nicht ein Kollektiv.

DER VERTRAG, DEN DIESE KLASSE ERFUELLT -- mit Datei und Zeile belegt
--------------------------------------------------------------------
``token_dispatcher/base.py``:

* ``:279`` ``dispatch(hidden_states, topk_output) -> DispatchOutput``
  (abstrakt), ``:304`` ``combine(combine_input) -> torch.Tensor``
  (abstrakt). Mehr verlangt ``BaseDispatcher`` nicht.
* ``:187`` ``DispatchOutput`` ist ein ``Protocol`` mit dem Feld
  ``hidden_states`` und der Eigenschaft ``format``; ``:242``
  ``CombineInput`` nur mit ``format``.
* ``:161`` ``DispatchOutputFormat`` und ``:235`` ``CombineInputFormat``
  sind **geschlossene** Aufzaehlungen. Ein neuer Wert waere ein Format,
  das kein Runner kennt -- der Dispatcher liefe, und niemand koennte sein
  Ergebnis rechnen.
* ``:361`` ``set_quant_config(dict)``, ``:364`` ``set_overlap_args``,
  ``:370`` ``clear_overlap_args`` -- der Rahmen ruft sie.
* ``:285``/``:308`` haengen Haken vor und hinter beide Richtungen; sie
  arbeiten auf ``self.dispatch``/``self.combine`` und brauchen von einer
  Unterklasse nichts weiter.

Deshalb liefert diese Klasse **kein eigenes Format**, sondern
``DEEPEP_NORMAL``: dieselben ``NamedTuple``-Klassen aus ``deepep.py:95``
(``DeepEPNormalDispatchOutput``) und ``deepep.py:128``
(``DeepEPNormalCombineInput``). Wer sie auspackt, ist nachgesehen:

* ``moe_runner/deep_gemm.py:779`` ``pre_permute_deepep_normal_to_deep_gemm``
  entpackt das 5-Tupel ``(hidden_states, hidden_states_scale, topk_ids,
  topk_weights, num_recv_tokens_per_expert)``, bildet
  ``all_tokens = sum(num_recv_tokens_per_expert)`` -- also eine
  **Python-Liste auf der CPU** -- und ruft ``ep_scatter``.
* ``ep_moe/kernels.py:1108`` ``ep_scatter`` liest ``recv_topk`` als
  **lokale** Expertennummer in ``[0, num_local_experts)`` und ``-1`` fuer
  jeden Platz, der nicht hierher gehoert (``_fwd_kernel_ep_scatter_2``:
  ``if expert_id >= 0``), und indiziert ``expert_start_loc`` damit.
  ``m_indices.shape[0] % 128 == 0`` wird geprueft -- daher die Ausrichtung
  der Zaehlwerte auf 128, genau wie ``deepep.py:589``
  ``expert_alignment=128 if ENABLE_JIT_DEEPGEMM``.
* ``ep_moe/kernels.py:1234`` ``ep_gather`` gewichtet nur dort, wo
  ``expert_id >= 0`` -- die Gewichte reisen also **unmaskiert**, wie bei
  DeepEP.
* ``moe_runner/deep_gemm.py:867`` ``post_permute_deep_gemm_to_deepep_normal``
  baut daraus ``DeepEPNormalCombineInput(hidden_states, topk_ids,
  topk_weights)`` mit ``hidden_states`` in **bf16** und in der Zeilenzahl
  der empfangenen Token.
* ``ep_moe/layer.py:207`` und ``quantization/unquant.py:837`` unterscheiden
  die Faelle ueber ``DispatchOutputChecker.format_is_deepep_normal`` --
  also ueber genau dieses Format.

WAS DEEPEP VOR DEN DATEN AUSTAUSCHT, UND IN WELCHER REIHENFOLGE
---------------------------------------------------------------
``deepep.py:559`` ``buffer.get_dispatch_layout(topk_ids, num_experts)``
rechnet **rein lokal** aus ``topk_ids``:

1. ``num_tokens_per_rank`` -- wieviele Token gehen an jeden Rang,
2. ``num_tokens_per_rdma_rank`` -- dasselbe je RDMA-Knoten (hier
   gegenstandslos, es gibt einen Knoten),
3. ``num_tokens_per_expert`` -- wieviele Token je **globalem** Experten,
4. ``is_token_in_rank`` -- die Bitmaske [T, R].

Erst danach (``deepep.py:578``) laeuft ``buffer.dispatch(...)``, und **darin**
steckt das eigentliche Kollektiv der Zaehlwerte (DeepEPs ``notify_dispatch``),
denn der Empfaenger kann seine Puffergroesse nicht kennen, bevor der Sender
gezaehlt hat. Genau diese Reihenfolge steht hier: lokale Zerlegung, dann
**ein** ``all_gather`` der Zaehlwerte ueber die CPU-Gruppe, dann die Daten.

Der Zaehlwerte-Abgleich ist ein Host-Kollektiv und braucht die Zahlen auf der
CPU. Das ist der Grund, warum dieser Pfad **nicht CUDA-Graph-faehig** ist --
derselbe Grund, aus dem ``barlink.py:525`` den ungleich geteilten
``all_to_all_single`` dort ausnimmt. ``server_args`` schaltet deshalb fuer
``--moe-a2a-backend bar1ep`` die Graphen ab, wie es das fuer
``deepep_mode=normal`` schon tut.

WAS AUF DER LEITUNG LIEGT
-------------------------
Zwei Aufrufe je Richtung, nicht vier:

* **Nutzlast** -- ``[Token, hidden_size]``, als ``uint8`` angefasst. Sie
  bleibt unangetastet und landet zeilenweise genau dort, wo der Runner sie
  erwartet.
* **Metadaten** -- ``[Token, topk*8 + topk*4 + Skalenzeile]``: lokale
  Expertennummern (int64), Gewichte (float32) und, wenn fp8 laeuft, die
  Skalenzeile. Drei kleine Felder in einem Block.

Vier Aufrufe waeren vier Sperren; einer waere ein Auspacken der ganzen
Nutzlast. Zwei ist die Mitte, und die kleine Umkopie trifft nur die
Metadaten.

Ist ein Block groesser als ein a2a-Schlitz, laeuft er ueber mehrere Runden.
Die Rundenzahl folgt aus dem **gruppenweiten** Maximum ueber alle R*R
Bloecke; jeder Rang zaehlt damit gleich viele Runden. Zaehlten zwei Raenge
verschieden, waere das ein Haenger und kein Fehler.

FP8
---
Der Kern bewegt **Bytes** (``barlink_bar1_ext.py:1136``, "Kein Datentyp, keine
Reduktion"). ``torch.float8_e4m3fn`` braucht deshalb keinen Sonderweg -- die
Nutzlast wird ohnehin als ``uint8`` angefasst, damit auch kein
``index_select`` auf einem fp8-Tensor noetig ist. Was **nicht** von selbst
mitreist, sind die Skalierungsfaktoren: ``deepep.py:512`` quantisiert mit
``sglang_per_token_group_quant_fp8(hidden_states, 128, ...)`` und laesst
DeepEP das Paar ``(x_q, x_s)`` tragen; hier reist ``x_s`` je Token im
Metadatenblock mit, Zeile fuer Zeile neben ``topk_ids`` und ``topk_weights``.
Die Schalterstellung ist von ``deepep.py:512`` uebernommen, nicht
nachempfunden.

WAS DIESER DISPATCHER NICHT KANN
--------------------------------
* **Low-Latency-Form.** ``DEEPEP_LL`` hat ein anderes Layout (feste
  Bucketgroesse je Experte, ``masked_m``, ``expected_m``) und einen anderen
  Runner-Pfad. Hier ist nur die Normalform gebaut. ``--deepep-mode auto``
  oder ``low_latency`` wird deshalb **abgelehnt**, nicht stillschweigend
  umgebogen: sonst rechnete ``DeepEPMoE`` mit LL-Annahmen und bekaeme
  Normalform-Tensoren.
* **NVFP4.** Die Skalen liegen dort verschraenkt und je Token nicht
  zusammenhaengend. Nicht gebaut, also nicht angeboten.
* **Mehr als ein Knoten.** Der Direktpfad ist BAR1 zu BAR1 ueber PCIe.
"""

from __future__ import annotations

import logging
import os
from typing import TYPE_CHECKING, List, Optional, Tuple

import torch
import torch.distributed as dist

from sglang.srt.eplb.expert_distribution import get_global_expert_distribution_recorder
from sglang.srt.layers import deep_gemm_wrapper
from sglang.srt.layers.moe.token_dispatcher.base import (
    BaseDispatcher,
    CombineInput,
    DispatchOutput,
)
from sglang.srt.layers.moe.token_dispatcher.deepep import (
    DeepEPNormalCombineInput,
    DeepEPNormalDispatchOutput,
    DeepEPPDispatchHooks,
)
from sglang.srt.layers.moe.utils import (
    DeepEPMode,
    DeepEPOutputDtype,
    get_deepep_output_dtype,
)

if TYPE_CHECKING:
    from sglang.srt.batch_overlap.single_batch_overlap import CombineOverlapArgs
    from sglang.srt.layers.moe.topk import TopKOutput

logger = logging.getLogger(__name__)


class Bar1EPUnverfuegbar(RuntimeError):
    """Der Direktpfad traegt diesen Dispatcher hier nicht.

    Ausdruecklich **kein** stiller Rueckfall: wer ``bar1ep`` gewaehlt hat,
    bekommt entweder BAR1 oder eine Fehlermeldung mit Grund. Ein Rueckfall,
    der etwas anderes tut und wie BAR1 aussieht, waere die schlechteste aller
    Antworten -- die Messung sagte dann etwas ueber einen Weg, den niemand
    gewaehlt hat.
    """


def _umgebungs_flagge(name: str, vorgabe: str = "1") -> bool:
    return os.environ.get(name, vorgabe) not in ("0", "nein", "aus", "false")


# ---------------------------------------------------------------------------
# Verfuegbarkeit
# ---------------------------------------------------------------------------

#: The transport methods this dispatcher calls, probed by name before the
#: first call. The names are the ones ``BarlinkBar1Transport`` publishes
#: (``barlink_bar1.py``) -- and they are ONLY here as strings, which is what
#: makes them dangerous: a ``hasattr`` probe survives every rename tool and
#: every import check, so a stale spelling here closes the gate silently and
#: forever. Task #295 renamed ``traegt_a2a``/``a2a_schlitz_bytes`` to
#: ``supports_a2a``/``a2a_slot_bytes``, this probe was not renamed with them,
#: and the BAR1 EP dispatch was unreachable until task #361.
#: ``test_bar1ep_transport_gate.py`` pins every name against the real class.
TRANSPORT_A2A_ATTRS = ("barlink_all_to_all_single", "supports_a2a", "a2a_slot_bytes")

#: Decline reasons already announced in this process. The gate is asked once
#: per MoE layer and once per dispatcher, so an unconditional log line would
#: repeat dozens of times per boot; a set keeps it loud without being noise.
_DECLINE_ANNOUNCED: set = set()


def _declined(reason: str):
    """Announce a closed gate exactly once, then return ``(None, reason)``.

    A gate that declines without a word is how this path died: the condition
    was false on every rank, and nothing in the log said so. The caller that
    asked for ``bar1ep`` explicitly still gets an exception carrying the same
    text (``create_moe_dispatcher``); this line is for the boot where the
    reason would otherwise only exist as a return value nobody prints.
    """
    if reason not in _DECLINE_ANNOUNCED:
        _DECLINE_ANNOUNCED.add(reason)
        logger.warning("bar1ep: BAR1 dispatch path not available -- %s", reason)
    return None, reason


def bar1ep_transport(gruppe_koordinator=None):
    """Der BAR1-Transport dieser Gruppe, oder ``(None, Grund)``.

    Jede Bedingung ist rangeinheitlich: sie haengt an gruppenweit
    abgeglichenem Zustand (Umgebungsvariablen, ``_a2a_proof`` aus einem
    ``all_gather_object`` in ``barlink_bar1.byte_proof_a2a``, Geometrie aus
    rangeinheitlichen Groessen). Zwei Raenge duerfen hier nie verschieden
    antworten -- der eine liefe ins Kollektiv, der andere nicht, und daraus
    wuerde ein Haenger statt eines Fehlers.

    Jede Ablehnung geht durch ``_declined`` und steht damit im Protokoll.
    """
    if gruppe_koordinator is None:
        from sglang.srt.distributed.parallel_state import get_tp_group

        gruppe_koordinator = get_tp_group()

    comm = getattr(gruppe_koordinator, "barlink_comm", None)
    if comm is None:
        return _declined(
            "barlink ist nicht aktiv (SGLANG_BARLINK=0 oder world_size==1). Der "
            "BAR1-Direktpfad haengt am BarlinkCommunicator; ohne ihn gibt es "
            "weder Peer-Zeiger-Tabelle noch Schlitze."
        )
    if getattr(comm, "disabled", False):
        return _declined(
            "BarlinkCommunicator ist abgeschaltet (world_size == 1)."
        )
    t = getattr(comm, "transport", None)
    if t is None:
        return _declined(
            "barlink laeuft auf der gloo-Ebene -- kein Transport. "
            "SGLANG_BARLINK_TRANSPORT=bar1 oder =matrix waehlt den Direktpfad."
        )
    fehlend = [n for n in TRANSPORT_A2A_ATTRS if not hasattr(t, n)]
    if fehlend:
        return _declined(
            f"Transport {type(t).__name__} hat kein all_to_all "
            f"({', '.join(fehlend)} fehlt). Das ist kein BAR1-Transport."
        )
    schlitz = int(t.a2a_slot_bytes())
    if schlitz <= 0:
        return _declined(
            "Der BAR1-Transport steht, aber sein a2a-Byte-Beleg ist nicht "
            "bestanden (oder SGLANG_BARLINK_BAR1_A2A=0). Ohne bestandenen Beleg "
            "meldet sich all_to_all ab -- siehe barlink_bar1.byte_proof_a2a."
        )
    return t, ""


def bar1ep_verfuegbar(gruppe_koordinator=None) -> Tuple[bool, str]:
    """``(True, "")`` genau dann, wenn die Auswahl ``bar1ep`` anbieten darf.

    Prueft, was **ohne** die Modellgeometrie pruefbar ist. Die Fragen, die
    erst mit ``hidden_size``/``topk`` beantwortbar sind (passt eine Zeile in
    einen Schlitz?) und der Byte-Beleg stehen im Konstruktor des
    Dispatchers, weil sie die Zahlen brauchen.
    """
    t, grund = bar1ep_transport(gruppe_koordinator)
    return (t is not None), grund


#: Ein bestandener Selbsttest je (CPU-Gruppe, Geometrie). Der Test ist die
#: Voraussetzung dafuer, dass sich der Dispatcher anbietet; er kostet aber
#: Startzeit, und bei aktivem TBO werden zwei Dispatcher derselben Geometrie
#: gebaut, bei jeder MoE-Schicht noch einmal. Der Schluessel enthaelt alles,
#: was der Test wirklich prueft.
_SELBSTTEST_STAND: dict = {}


def _schneide(quelle: torch.Tensor, off: int, breite: int,
              dtype: torch.dtype, spalten: int) -> torch.Tensor:
    """Ein Spaltenstueck aus einem ``uint8``-Block, umgedeutet.

    Bewusst ueber einen **frischen** Puffer und ``copy_`` statt ueber
    ``.contiguous().view(dtype)``: bei einer Zeile (oder null Zeilen) haelt
    PyTorch den Spaltenschnitt schon fuer zusammenhaengend, ``contiguous()``
    gibt dann die Sicht mit ihrem Speicherversatz zurueck, und ``view(dtype)``
    haengt damit an einer Ausrichtungsbedingung, die von ``topk`` abhaengt.
    Ein frischer Puffer beginnt bei Versatz 0 -- die Bedingung entfaellt,
    statt fast immer zu gelten.
    """
    n = quelle.shape[0]
    ziel = torch.empty((n, breite), dtype=torch.uint8, device=quelle.device)
    ziel.copy_(quelle[:, off : off + breite])
    return ziel.view(dtype).reshape(n, spalten)


# ---------------------------------------------------------------------------
# Der Dispatcher
# ---------------------------------------------------------------------------


class Bar1EPDispatcher(BaseDispatcher):
    """Dispatch/Combine ueber ``bar1_all_to_all``, Normalform.

    Der Zustand zwischen ``dispatch`` und ``combine`` (Sortierindex,
    Zaehlwerte, Rundenzahl) haengt an der Instanz, nicht am
    ``DispatchOutput`` -- dieselbe Loesung wie ``deepep.py:566`` ("`handle`
    should be transmitted with tokens ... keeping `handle` as a member
    variable works"), und aus demselben Grund: das Tupelformat ist
    geschlossen, ein sechstes Feld waere ein neues Format.
    """

    def __init__(
        self,
        group: Optional[torch.distributed.ProcessGroup] = None,
        router_topk: int = None,
        permute_fusion: bool = False,
        num_experts: int = None,
        num_local_experts: int = None,
        hidden_size: int = None,
        params_dtype: torch.dtype = None,
        deepep_mode: DeepEPMode = DeepEPMode.NORMAL,
        async_finish: bool = False,
        return_recv_hook: bool = False,
        **_ungenutzt,
    ):
        super().__init__()

        from sglang.srt.distributed.parallel_state import get_tp_group

        self.gruppe = get_tp_group()
        self.comm = getattr(self.gruppe, "barlink_comm", None)
        self.transport, grund = bar1ep_transport(self.gruppe)
        if self.transport is None:
            raise Bar1EPUnverfuegbar(
                f"--moe-a2a-backend bar1ep gewaehlt, aber: {grund}"
            )

        if group is not None and group is not getattr(
            self.gruppe, "device_group", None
        ):
            # create_moe_dispatcher reicht get_tp_group().device_group herein
            # (fused_moe_triton/layer.py:96/125). Kaeme hier eine andere Gruppe
            # an, liefe der Zaehlwerteabgleich ueber eine andere Menge von
            # Raengen als die Peer-Zeiger-Tabelle -- ein Haenger, kein Fehler.
            raise Bar1EPUnverfuegbar(
                "bar1ep laeuft nur auf der TP-Gruppe: der BAR1-Transport "
                "haengt an get_tp_group().barlink_comm, und eine zweite Gruppe "
                "haette weder Peer-Zeiger noch Schlitze."
            )

        self.cpu_group = self.comm.cpu_group
        self.welt = int(self.comm.world_size)
        self.rank = int(self.comm.rank)

        self.router_topk = int(router_topk)
        self.num_experts = int(num_experts)
        self.num_local_experts = int(num_local_experts)
        self.hidden_size = int(hidden_size)
        self.params_dtype = params_dtype or torch.bfloat16
        self.device = torch.device("cuda", torch.cuda.current_device())

        if self.num_experts != self.num_local_experts * self.welt:
            raise Bar1EPUnverfuegbar(
                f"num_experts {self.num_experts} ist nicht "
                f"{self.num_local_experts} * {self.welt}. bar1ep bildet den "
                f"Experten e auf Rang e // num_local_experts ab -- dieselbe "
                f"Abbildung wie DeepEP; ohne gleiche Teilung gibt es sie nicht."
            )
        if self.hidden_size % 128 != 0 and self._skalen_moeglich():
            raise Bar1EPUnverfuegbar(
                f"hidden_size {self.hidden_size} ist kein Vielfaches von 128; "
                f"die 128er-Blockquantisierung des fp8-Wegs (deepep.py:512) "
                f"gibt es dafuer nicht."
            )

        self.deepep_mode = deepep_mode
        if deepep_mode is not None and not deepep_mode.is_normal():
            raise Bar1EPUnverfuegbar(
                f"bar1ep baut nur die Normalform (DEEPEP_NORMAL), deepep_mode "
                f"ist aber {deepep_mode}. Die Low-Latency-Form hat ein anderes "
                f"Ausgabeformat (masked_m/expected_m) und einen anderen "
                f"Runner-Pfad; sie hier stillschweigend durch die Normalform "
                f"zu ersetzen hiesse, DeepEPMoE mit LL-Annahmen "
                f"Normalform-Tensoren zu geben. Bitte --deepep-mode normal."
            )

        # DeepEP/Mooncake/Nixl markieren ungueltige topk-Plaetze mit -1; der
        # AITER-pre_permute leitet sie auf einen Senkenplatz um. Ohne AITER
        # gibt es hier nichts zu maskieren -- aber MaybeTboDeepEPDispatcher
        # liest das Feld unbedingt (two_batch_overlap.py:1097).
        self.expert_mask_gpu = None

        self.quant_config: Optional[dict] = None
        self.use_fp8 = False
        self._schlitz = int(self.transport.a2a_slot_bytes())
        self._setze_ausgabetyp()

        # Zustand zwischen dispatch und combine.
        self._sende_zeilen: List[int] = []
        self._empf_zeilen: List[int] = []
        self._sende_index: Optional[torch.Tensor] = None
        self._token_zahl = 0
        self._max_zeilen = 0
        self._ausgabe_dtype = self.params_dtype
        self._dispatch_zwischenstand = None
        self._combine_zwischenstand = None

        self._bar1_dispatch_hooks = DeepEPPDispatchHooks()

        self._pruefe_fenster()
        self._selbsttest_wenn_noetig()

    # -- Faehigkeit --------------------------------------------------------

    def _skalen_moeglich(self) -> bool:
        return deep_gemm_wrapper.ENABLE_JIT_DEEPGEMM

    def _setze_ausgabetyp(self) -> None:
        typ = get_deepep_output_dtype(self)
        if typ == DeepEPOutputDtype.NVFP4:
            raise Bar1EPUnverfuegbar(
                "bar1ep traegt nvfp4 nicht: dessen Skalen liegen verschraenkt "
                "und je Token nicht zusammenhaengend. Nicht gebaut heisst "
                "nicht angeboten."
            )
        if typ == DeepEPOutputDtype.INT8:
            raise Bar1EPUnverfuegbar("bar1ep traegt int8-Dispatch nicht (NPU-Weg).")
        # Wie deepep.py:510: quantisiert wird nur, wenn DeepGEMM den fp8-Weg
        # ueberhaupt rechnet. Ohne ihn faehrt auch DeepEP bf16.
        self.use_fp8 = (
            typ == DeepEPOutputDtype.FP8 and deep_gemm_wrapper.ENABLE_JIT_DEEPGEMM
        )

    def _skalen_dtype(self) -> torch.dtype:
        return torch.int32 if deep_gemm_wrapper.DEEPGEMM_SCALE_UE8M0 else torch.float32

    def _skalen_spalten(self) -> int:
        """Spalten des Skalentensors je Token -- 0, wenn ohne fp8.

        Die Zahlen sind aus ``fp8_kernel.py:488-511`` uebernommen: mit
        ue8m0 packt der Quantisierer je vier Skalen in ein ``int32`` und
        richtet auf vier aus, sonst ist es ein ``float32`` je 128er-Block.
        ``ep_scatter`` prueft genau diese Spaltenzahl
        (``kernels.py:1104``).
        """
        if not self.use_fp8:
            return 0
        s = self.hidden_size // 128
        if deep_gemm_wrapper.DEEPGEMM_SCALE_UE8M0:
            return -(-s // 4)
        return s

    def _nutz_zeilenbytes(self) -> int:
        e = 1 if self.use_fp8 else (torch.finfo(self.params_dtype).bits // 8)
        return self.hidden_size * e

    def _meta_zeilenbytes(self) -> int:
        # topk_ids int64, topk_weights float32, dazu die Skalenzeile (int32
        # oder float32 -- beide vier Byte).
        return self.router_topk * 8 + self.router_topk * 4 + self._skalen_spalten() * 4

    def _pruefe_fenster(self) -> None:
        """Passt ueberhaupt EINE Zeile in einen Schlitz?

        Die Frage ist nicht akademisch: der Schlitz ist ``chunk_max`` aus
        ``barlink_bar1.geometrie`` und faellt mit der Fenstergroesse. Eine
        Zeile, die nicht hineinpasst, laesst sich auch nicht in Runden
        zerlegen -- die Zerlegung teilt Zeilen, nicht Zeileninhalte. Dann
        meldet sich der Dispatcher ab, statt spaeter im heissen Pfad zu
        scheitern.
        """
        self._schlitz = int(self.transport.a2a_slot_bytes())
        for name, zb in (
            ("Nutzlast", self._nutz_zeilenbytes()),
            ("Metadaten", self._meta_zeilenbytes()),
        ):
            if zb > self._schlitz:
                raise Bar1EPUnverfuegbar(
                    f"Eine {name}-Zeile ist {zb} Byte gross und passt nicht in "
                    f"den a2a-Schlitz von {self._schlitz} Byte. Groesseres "
                    f"Fenster (SGLANG_BARLINK_BAR1_WINDOW_MIB) -- ein Rueckfall "
                    f"waere hier keine Loesung, sondern eine andere Messung "
                    f"unter demselben Namen."
                )

    def _zeilen_pro_runde(self, zeilenbytes: int) -> int:
        return max(1, int(self._schlitz // max(1, zeilenbytes)))

    # -- Der Datenweg ------------------------------------------------------

    def _a2a_zeilen(
        self,
        aus: torch.Tensor,
        ein: torch.Tensor,
        sende_zeilen: List[int],
        empf_zeilen: List[int],
        zeilenbytes: int,
        max_zeilen: int,
        zeilen_pro_runde: Optional[int] = None,
    ) -> torch.Tensor:
        """``all_to_all`` mit ungleichen Bloecken, notfalls ueber mehrere Runden.

        ``ein``/``aus`` sind zusammenhaengende ``uint8``-Tensoren der Form
        ``[Zeilen, zeilenbytes]``. ``max_zeilen`` ist das **gruppenweite**
        Maximum ueber alle R*R Bloecke; daraus folgt die Rundenzahl, und weil
        die Zahl gruppenweit dieselbe ist, zaehlt jeder Rang gleich viele
        Runden.

        Die Versaetze werden ausgerechnet und **uebergeben**, statt die Naht
        sie als Praefixsumme raten zu lassen: eine Runde bewegt aus jedem
        Block nur ein Stueck, und die Bloecke bleiben dabei stehen, wo sie
        sind. Genau dafuer nimmt ``barlink_bar1.barlink_all_to_all_single`` seit
        dieser Aenderung ``send_offsets``/``recv_offsets`` entgegen; der
        Kernel hat Versaetze und Laengen ohnehin immer getrennt bekommen.
        """
        R = self.welt
        zpr = zeilen_pro_runde or self._zeilen_pro_runde(zeilenbytes)
        runden = max(1, -(-int(max_zeilen) // zpr))

        s_basis, acc = [], 0
        for n in sende_zeilen:
            s_basis.append(acc)
            acc += int(n)
        e_basis, acc = [], 0
        for n in empf_zeilen:
            e_basis.append(acc)
            acc += int(n)

        ein_flach = ein.reshape(-1)
        aus_flach = aus.reshape(-1)
        # Ein leerer Tensor hat data_ptr() == 0. Sind BEIDE Seiten leer,
        # zeigen sie auf dieselbe Adresse, und die Erweiterung lehnt das ab
        # ("in und out duerfen nicht dasselbe sein", barlink_bar1_ext.py:1209) --
        # zu Recht, denn sie kann nicht wissen, dass hier nichts zu bewegen
        # ist. Aussteigen darf man trotzdem nicht: die anderen Raenge warten
        # in derselben Sperre auf meine Flagge, auch wenn ich null Byte
        # schicke. Also zwei Platzhalter; alle Laengen sind 0, es wird nichts
        # aus ihnen gelesen und nichts in sie geschrieben.
        if ein_flach.numel() == 0:
            ein_flach = torch.zeros(16, dtype=torch.uint8, device=ein.device)
        if aus_flach.numel() == 0:
            aus_flach = torch.zeros(16, dtype=torch.uint8, device=aus.device)

        for k in range(runden):
            s_off, s_len, e_off, e_len = [], [], [], []
            for j in range(R):
                a = min(k * zpr, int(sende_zeilen[j]))
                b = min((k + 1) * zpr, int(sende_zeilen[j]))
                s_off.append((s_basis[j] + a) * zeilenbytes)
                s_len.append((b - a) * zeilenbytes)
                a = min(k * zpr, int(empf_zeilen[j]))
                b = min((k + 1) * zpr, int(empf_zeilen[j]))
                e_off.append((e_basis[j] + a) * zeilenbytes)
                e_len.append((b - a) * zeilenbytes)
            groesster = max(s_len + e_len)
            if not self.transport.supports_a2a(groesster):
                raise Bar1EPUnverfuegbar(
                    f"Runde {k}: groesster Block {groesster} Byte passt nicht "
                    f"in den Schlitz von {self._schlitz} Byte. Die "
                    f"Rundenzerlegung haette das verhindern muessen -- diese "
                    f"Zeile ist der Beweis, dass sie es nicht hat."
                )
            self.transport.barlink_all_to_all_single(
                self.comm, aus_flach, ein_flach, s_len, e_len, s_off, e_off,
            )
        return aus

    def _zerlegung(self, topk_ids: torch.Tensor):
        """Die lokale Zerlegung -- das Gegenstueck zu ``get_dispatch_layout``.

        Liefert ``(is_token_in_rank, num_tokens_per_rank,
        num_tokens_per_expert)``. ``topk_ids`` darf ``-1`` enthalten
        (ungueltige Plaetze, so markiert sie die DeepEP-Familie); die zaehlen
        nirgends mit.
        """
        T = topk_ids.shape[0]
        R, nle = self.welt, self.num_local_experts
        gueltig = topk_ids >= 0
        ziel = torch.where(
            gueltig,
            torch.div(topk_ids, nle, rounding_mode="floor"),
            torch.zeros_like(topk_ids),
        )
        # scatter_add_ statt scatter_: liegen mehrere topk-Plaetze auf
        # demselben Rang, ueberschriebe scatter_ in unbestimmter Reihenfolge,
        # und ein ungueltiger Platz (Ziel 0) koennte einen echten Treffer
        # loeschen.
        zahl = torch.zeros((T, R), dtype=torch.int32, device=topk_ids.device)
        zahl.scatter_add_(1, ziel, gueltig.to(torch.int32))
        in_rang = zahl > 0
        ntpr = in_rang.sum(dim=0).to(torch.int64)
        ntpe = torch.bincount(
            topk_ids[gueltig].reshape(-1), minlength=self.num_experts
        ).to(torch.int64)[: self.num_experts]
        return in_rang, ntpr, ntpe

    def _zaehlwerte_tauschen(self, ntpr: torch.Tensor, ntpe: torch.Tensor):
        """Ein ``all_gather`` ueber die CPU-Gruppe. Der einzige Host-Sync.

        Genau der Schritt, den DeepEP in ``notify_dispatch`` faehrt, aus
        demselben Grund: der Empfaenger kann seine Puffergroesse nicht kennen,
        bevor der Sender gezaehlt hat. Er steht **vor** dem Datenpfad, nicht
        darin. Zeile ``i`` ist ``[num_tokens_per_rank (R),
        num_tokens_per_expert (num_experts)]`` des Rangs ``i``.
        """
        flach = torch.cat([ntpr, ntpe]).to("cpu")
        eingang = [torch.empty_like(flach) for _ in range(self.welt)]
        dist.all_gather(eingang, flach, group=self.cpu_group)
        return [t.tolist() for t in eingang]

    # -- Hinweg ------------------------------------------------------------

    def dispatch(
        self, hidden_states: torch.Tensor, topk_output: "TopKOutput"
    ) -> DispatchOutput:
        self.dispatch_a(hidden_states, topk_output)
        if self._bar1_dispatch_hooks is not None:
            self._bar1_dispatch_hooks(self)
        return self.dispatch_b()

    def dispatch_a(self, hidden_states: torch.Tensor, topk_output: "TopKOutput"):
        self._dispatch_zwischenstand = self._dispatch_vorbereiten(
            hidden_states, topk_output
        )

    def dispatch_b(self, *zustand) -> DispatchOutput:
        if not zustand:
            zustand = self._dispatch_zwischenstand
            self._dispatch_zwischenstand = None
        return self._dispatch_kern(*zustand)

    def _dispatch_vorbereiten(
        self, hidden_states: torch.Tensor, topk_output: "TopKOutput"
    ):
        """Alles bis zum Datenpfad: quantisieren, zerlegen, Zaehlwerte tauschen."""
        topk_weights = topk_output.topk_weights.to(torch.float32).contiguous()
        topk_ids = topk_output.topk_ids.to(torch.int64).contiguous()
        self._ausgabe_dtype = hidden_states.dtype
        hidden_states = hidden_states.contiguous()

        skala = None
        if self.use_fp8:
            from sglang.srt.layers.quantization.fp8_kernel import (
                sglang_per_token_group_quant_fp8,
            )

            # Dieselben Schalter wie deepep.py:512 -- nicht aehnliche.
            hidden_states, skala = sglang_per_token_group_quant_fp8(
                hidden_states,
                128,
                column_major_scales=deep_gemm_wrapper.DEEPGEMM_SCALE_UE8M0,
                scale_tma_aligned=deep_gemm_wrapper.DEEPGEMM_SCALE_UE8M0,
                scale_ue8m0=deep_gemm_wrapper.DEEPGEMM_SCALE_UE8M0,
            )
            # Bei ue8m0 legt der Quantisierer den Skalentensor spaltenweise an
            # (fp8_kernel.py:499 `.transpose(-1,-2)`). Fuer den Transport
            # braucht es die Zeile zusammenhaengend; ep_scatter liest ohnehin
            # ueber Schrittweiten und nimmt beide Formen.
            skala = skala.contiguous()

        in_rang, ntpr, ntpe = self._zerlegung(topk_ids)
        matrix = self._zaehlwerte_tauschen(ntpr, ntpe)
        return (hidden_states, skala, topk_ids, topk_weights, in_rang, matrix)

    def _dispatch_kern(
        self, hidden_states, skala, topk_ids, topk_weights, in_rang, matrix
    ) -> DispatchOutput:
        R, nle, K = self.welt, self.num_local_experts, self.router_topk
        T = hidden_states.shape[0]
        geraet = hidden_states.device

        sende_zeilen = [int(matrix[self.rank][j]) for j in range(R)]
        empf_zeilen = [int(matrix[i][self.rank]) for i in range(R)]
        max_zeilen = max(int(matrix[i][j]) for i in range(R) for j in range(R))
        S, N = sum(sende_zeilen), sum(empf_zeilen)

        # Sendereihenfolge: je Zielrang die eigenen Tokennummern aufsteigend,
        # die Zielraenge aufsteigend hintereinander. Der Empfaenger kennt sie
        # damit ohne ein einziges uebertragenes Indexbyte -- und der Rueckweg
        # findet dieselbe Ordnung vor.
        if T:
            lauf = torch.nonzero(
                in_rang.t().reshape(-1), as_tuple=False
            ).reshape(-1)
            sende_index = torch.remainder(lauf, T)
        else:
            sende_index = torch.zeros(0, dtype=torch.int64, device=geraet)

        # -- Nutzlast: als Bytes angefasst, damit fp8 keinen Sonderweg und
        #    kein index_select auf einem fp8-Tensor braucht.
        nzb = self._nutz_zeilenbytes()
        x_bytes = hidden_states.view(torch.uint8).reshape(T, nzb)
        sende_x = x_bytes[sende_index]
        empf_x = torch.empty((N, nzb), dtype=torch.uint8, device=geraet)

        # -- Metadaten: lokale Expertennummern, Gewichte, Skalenzeile.
        eigner = torch.repeat_interleave(
            torch.arange(R, device=geraet, dtype=torch.int64),
            torch.tensor(sende_zeilen, device=geraet, dtype=torch.int64),
        )
        ids_roh = topk_ids[sende_index]
        passt = (
            torch.div(ids_roh, nle, rounding_mode="floor") == eigner.unsqueeze(1)
        )
        ids_lokal = torch.where(
            (ids_roh >= 0) & passt,
            ids_roh - eigner.unsqueeze(1) * nle,
            torch.full_like(ids_roh, -1),
        ).contiguous()
        gew = topk_weights[sende_index].contiguous()

        teile = [
            ids_lokal.view(torch.uint8).reshape(S, K * 8),
            gew.view(torch.uint8).reshape(S, K * 4),
        ]
        sb = self._skalen_spalten() * 4
        if sb:
            teile.append(
                skala[sende_index].contiguous().view(torch.uint8).reshape(S, sb)
            )
        mzb = self._meta_zeilenbytes()
        sende_meta = torch.cat(teile, dim=1)
        empf_meta = torch.empty((N, mzb), dtype=torch.uint8, device=geraet)

        self._a2a_zeilen(empf_x, sende_x, sende_zeilen, empf_zeilen, nzb, max_zeilen)
        self._a2a_zeilen(
            empf_meta, sende_meta, sende_zeilen, empf_zeilen, mzb, max_zeilen
        )

        # -- Auspacken.
        recv_ids = _schneide(empf_meta, 0, K * 8, torch.int64, K)
        recv_gew = _schneide(empf_meta, K * 8, K * 4, torch.float32, K)
        recv_skala = (
            _schneide(
                empf_meta, K * 12, sb, self._skalen_dtype(), self._skalen_spalten()
            )
            if sb
            else None
        )
        recv_x = empf_x.view(
            torch.float8_e4m3fn if self.use_fp8 else self._ausgabe_dtype
        ).reshape(N, self.hidden_size)

        # -- Zaehlwerte je lokalem Experten. Eine CPU-Liste, so wie der Runner
        #    sie braucht (moe_runner/deep_gemm.py:797 `sum(...)`).
        roh = [0] * nle
        for i in range(R):
            zeile = matrix[i]
            for e in range(nle):
                roh[e] += int(zeile[R + self.rank * nle + e])
        if deep_gemm_wrapper.ENABLE_JIT_DEEPGEMM:
            # ep_scatter prueft m_indices.shape[0] % 128 == 0 -- dieselbe
            # Ausrichtung, die deepep.py:589 als expert_alignment uebergibt.
            num_recv_tokens_per_expert = [(-(-c // 128)) * 128 for c in roh]
        else:
            num_recv_tokens_per_expert = roh

        get_global_expert_distribution_recorder().on_deepep_dispatch_normal(
            num_recv_tokens_per_expert,
            num_tokens_per_rank=torch.tensor(sende_zeilen, dtype=torch.int64),
            num_tokens_per_rdma_rank=None,
            num_tokens_per_expert=torch.tensor(
                [int(x) for x in matrix[self.rank][R:]], dtype=torch.int64
            ),
        )

        self._sende_zeilen = sende_zeilen
        self._empf_zeilen = empf_zeilen
        self._sende_index = sende_index
        self._token_zahl = T
        self._max_zeilen = max_zeilen

        return DeepEPNormalDispatchOutput(
            recv_x, recv_skala, recv_ids, recv_gew, num_recv_tokens_per_expert
        )

    # -- Rueckweg ----------------------------------------------------------

    def combine(self, combine_input: CombineInput) -> torch.Tensor:
        self.combine_a(combine_input)
        return self.combine_b()

    def combine_a(self, combine_input: CombineInput):
        hidden_states, topk_ids, topk_weights = combine_input
        self._combine_zwischenstand = (hidden_states,)

    def combine_b(self, *zustand) -> torch.Tensor:
        if not zustand:
            zustand = self._combine_zwischenstand
            self._combine_zwischenstand = None
        return self._combine_kern(*zustand)

    def _combine_kern(self, hidden_states: torch.Tensor) -> torch.Tensor:
        if self._sende_index is None:
            raise RuntimeError(
                "bar1ep: combine ohne vorangegangenes dispatch. Der "
                "Sortierindex des Rueckwegs entsteht im Hinweg."
            )
        T = self._token_zahl
        H = hidden_states.shape[1]
        geraet, dtype = hidden_states.device, hidden_states.dtype
        zb = H * (torch.finfo(dtype).bits // 8)

        # Rueckweg: dieselbe Maschinerie in Gegenrichtung. Was im Hinweg
        # empfangen wurde, wird jetzt gesendet -- Block fuer Block, in
        # derselben Ordnung, also ohne ein einziges Indexbyte auf der Leitung.
        sende_zeilen = list(self._empf_zeilen)
        empf_zeilen = list(self._sende_zeilen)
        S = sum(empf_zeilen)

        ein = hidden_states.contiguous().view(torch.uint8).reshape(-1, zb)
        aus = torch.empty((S, zb), dtype=torch.uint8, device=geraet)
        self._a2a_zeilen(aus, ein, sende_zeilen, empf_zeilen, zb, self._max_zeilen)
        zurueck = aus.view(dtype).reshape(S, H)

        # Die Reduktion: ein Token, das auf mehreren Raengen Experten hatte,
        # bekommt je Rang einen Beitrag. index_add_ ueber den Sortierindex des
        # Hinwegs summiert sie. In float32, weil DeepEPs Combine-Kern es auch
        # tut -- in bf16 zu summieren waere billiger und eine andere Zahl.
        if _umgebungs_flagge("SGLANG_BAR1EP_COMBINE_FP32", "1"):
            acc = torch.zeros((T, H), dtype=torch.float32, device=geraet)
            acc.index_add_(0, self._sende_index, zurueck.to(torch.float32))
            ergebnis = acc.to(dtype)
        else:
            ergebnis = torch.zeros((T, H), dtype=dtype, device=geraet)
            ergebnis.index_add_(0, self._sende_index, zurueck)

        self._sende_index = None
        return ergebnis

    # -- Rahmenwerk --------------------------------------------------------

    def set_quant_config(self, quant_config: dict) -> None:
        super().set_quant_config(quant_config)
        self.quant_config = quant_config
        self._setze_ausgabetyp()
        self._pruefe_fenster()

    def set_overlap_args(
        self, combine_overlap_args: "CombineOverlapArgs", meta_overlap_args: dict
    ) -> None:
        # Der Direktpfad hat keine zweite Warteschlange und keinen
        # Empfangshaken: dispatch/combine sind je ein Kernelstart mit einer
        # Sperre. Die Ueberlappungsargumente werden angenommen (der Rahmen
        # setzt sie unbedingt) und ausdruecklich nicht benutzt -- sie zu
        # nehmen und zu ignorieren ist ehrlicher als eine Attrappe, die
        # aussieht wie Ueberlappung.
        super().set_overlap_args(combine_overlap_args, meta_overlap_args)

    def register_deepep_dispatch_hook(self, hook):
        return self._bar1_dispatch_hooks.register_hook(hook)

    # -- Byte-Beleg --------------------------------------------------------

    def _selbsttest_wenn_noetig(self) -> None:
        if not _umgebungs_flagge("SGLANG_BAR1EP_SELFTEST", "1"):
            logger.warning(
                "bar1ep: Byte-Beleg per SGLANG_BAR1EP_SELFTEST=0 "
                "uebersprungen. Damit steht hinter jeder Zahl dieses Laufs "
                "keine Aussage darueber, ob die Bytes ankommen."
            )
            return
        schluessel = (
            id(self.cpu_group),
            self.hidden_size,
            self.router_topk,
            self.num_experts,
            bool(self.use_fp8),
            str(self.params_dtype),
        )
        stand = _SELBSTTEST_STAND.get(schluessel)
        if stand is True:
            return
        if stand is False:
            raise Bar1EPUnverfuegbar(
                "bar1ep: der Byte-Beleg ist in diesem Prozess schon gefallen."
            )
        ok, grund = self.byte_beleg()
        _SELBSTTEST_STAND[schluessel] = ok
        if not ok:
            raise Bar1EPUnverfuegbar(f"bar1ep: Byte-Beleg gefallen -- {grund}")

    def byte_beleg(self) -> Tuple[bool, str]:
        """Der Beleg, ohne den sich der Dispatcher nicht anbietet.

        Drei Durchgaenge, alle ueber den ECHTEN Weg:

        1. **Rohe Bytes, ungleich und unausgerichtet, ueber mehrere Runden.**
           Blocklaengen ``97*(1+((q+z)%3)) + ((q*5+z*3)%7)`` -- der Faktor
           macht die Bloecke ungleich (der MoE-Normalfall), der Summand macht
           sie zu Nicht-Vielfachen von 16 und schiebt damit jeden folgenden
           Versatz aus der Ausrichtung. Die Rundenzahl wird auf mindestens
           drei gedrueckt, damit die Doppelpufferung (``runde & 1``) und die
           Versatzrechnung wirklich laufen und nicht nur existieren. Geprueft
           wird je Sender einzeln, Byte fuer Byte, auf der EMPFANGENDEN Karte
           -- auch der eigene Block, der gar nicht ueber die Apertur geht.
        2. **Dispatch, strukturiert, je gerichtetem Paar.** Die Zuordnung
           Token -> Experten folgt einer Regel, die jeder Rang fuer jeden
           anderen nachrechnen kann. Damit weiss jeder Empfaenger vorher,
           welche Zeile mit welchem Inhalt von wem kommen muss -- die Pruefung
           haengt also nicht an derselben Buchhaltung, die sie pruefen soll.
           Der Inhalt haengt an (Quellrang, Tokennummer, Spalte): eine
           vertauschte Zeile und eine verschobene Spalte fallen beide auf.
           Laeuft in der **konfigurierten** Form; mit fp8 wird gegen die
           lokal quantisierte Sollform Byte fuer Byte verglichen, Skalen
           eingeschlossen.
        3. **Combine, strukturiert.** Der Rueckweg bekommt genau das zurueck,
           was der Hinweg gebracht hat. Dann muss ``combine(dispatch(x))``
           gleich ``x * (Zahl der Raenge, die fuer dieses Token zustaendig
           sind)`` sein -- eine geschlossene Formel, kein zweiter Nachbau
           derselben Buchhaltung.

        Faellt irgendetwas davon, meldet sich **bar1ep** ab. ``all_reduce``
        und der a2a der barlink-Naht bleiben unberuehrt: sie haben ihre eigenen
        Belege, und aus einem gefallenen Dispatcher folgt nichts ueber sie.
        """
        ok, grund = True, ""
        try:
            ok, grund = self._beleg_rohe_bytes()
            if ok:
                ok, grund = self._beleg_struktur()
        except Exception as ex:  # noqa: BLE001 -- Grund ins Protokoll
            ok, grund = False, repr(ex)
            logger.warning("bar1ep: Byte-Beleg abgebrochen: %r", ex)

        # Ab hier gruppenweit, IN JEDEM FALL. Ein Rang, der vor dem
        # all_gather_object aussteigt, laesst die anderen darin stehen -- aus
        # einem gefallenen Beleg wuerde ein Haenger, und ein Haenger sagt
        # nicht, was kaputt ist.
        traeger: list = [None] * self.welt
        dist.all_gather_object(traeger, (bool(ok), str(grund)), group=self.cpu_group)
        schlecht = [i for i, (o, _) in enumerate(traeger) if not o]
        if schlecht:
            gruende = "; ".join(f"Rang {i}: {traeger[i][1]}" for i in schlecht)
            logger.warning("bar1ep: Byte-Beleg gruppenweit gefallen -- %s", gruende)
            return False, gruende
        logger.info(
            "bar1ep: Byte-Beleg bestanden (rohe Bytes ungleich/unausgerichtet "
            "ueber mehrere Runden; Dispatch je gerichtetem Paar gegen die "
            "Routenregel; Combine gegen die geschlossene Formel) -- %d Raenge, "
            "hidden=%d, topk=%d, fp8=%s.",
            self.welt, self.hidden_size, self.router_topk, self.use_fp8,
        )
        return True, ""

    @staticmethod
    def _marke(quelle: int, ziel: int) -> int:
        """Ein je gerichtetem Paar verschiedenes Byte, nie 0x00 und nie 0xFF.

        0xFF ist die Vorbelegung des Ausgabepuffers, 0x00 die des
        Empfangsschlitzes; beide sind damit vom Muster unterscheidbar, und ein
        NICHT geschriebener Block faellt als solcher auf, statt zufaellig wie
        ein Treffer auszusehen.
        """
        return 0x40 | ((quelle * 8 + ziel) & 0x3F)

    def _beleg_rohe_bytes(self) -> Tuple[bool, str]:
        R, r = self.welt, self.rank

        def laenge(q: int, z: int) -> int:
            return 97 * (1 + ((q + z) % 3)) + ((q * 5 + z * 3) % 7)

        sende = [laenge(r, z) for z in range(R)]
        empf = [laenge(q, r) for q in range(R)]
        max_zeilen = max(laenge(q, z) for q in range(R) for z in range(R))
        # Zeilenbreite 1 Byte: dann sind Blocklaengen gleich Zeilenzahlen und
        # die Versaetze durchweg unausgerichtet. Die Rundenzahl wird erzwungen,
        # statt sich aus einem Schlitz zu ergeben, der auf diesem Rig fuer
        # alles auf einmal reichen wuerde.
        zpr = max(1, max_zeilen // 3)

        ein = torch.empty((sum(sende), 1), dtype=torch.uint8, device=self.device)
        o = 0
        for z in range(R):
            ein[o : o + sende[z]] = self._marke(r, z)
            o += sende[z]
        aus = torch.full((sum(empf), 1), 0xFF, dtype=torch.uint8, device=self.device)

        dist.barrier(group=self.cpu_group)
        self._a2a_zeilen(aus, ein, sende, empf, 1, max_zeilen, zeilen_pro_runde=zpr)
        torch.cuda.synchronize(self.device)

        rueck = aus.reshape(-1).cpu()
        o = 0
        for q in range(R):
            soll = self._marke(q, r)
            stueck = rueck[o : o + empf[q]]
            schlecht = int((stueck != soll).sum().item())
            if schlecht:
                return False, (
                    f"rohe Bytes {q}->{r}: {schlecht} von {empf[q]} Byte falsch"
                )
            o += empf[q]
        return True, ""

    def _probe_routen(self, q: int) -> List[List[int]]:
        """Die topk-Zeilen des Rangs ``q`` in der Probe. Rein rechnerisch.

        Zwei Bloecke:

        * Block 1 macht die **Paare** ungleich und keines leer: fuer jedes
          Ziel ``z`` genau ``1 + ((q*3+z*5) % 5)`` Token, die NUR dorthin
          gehen. Damit traegt jedes der R*R gerichteten Paare Bytes -- ein
          Beleg ueber ein leeres Paar waere keiner.
        * Block 2 macht die **Mehrfachzuordnung**: ``R`` Token, von denen
          jedes ``min(topk, R)`` verschiedene Raenge trifft. Ohne diesen Block
          bekaeme jedes Token genau einen Beitrag, und der Combine-Test waere
          blind fuer die Summe.
        """
        R, nle, K = self.welt, self.num_local_experts, self.router_topk
        zeilen: List[List[int]] = []
        for z in range(R):
            for _ in range(1 + ((q * 3 + z * 5) % 5)):
                zeile = [-1] * K
                for k in range(min(K, nle)):
                    zeile[k] = z * nle + k
                zeilen.append(zeile)
        for j in range(R):
            zeile = [-1] * K
            for k in range(min(K, R)):
                z = (j + k) % R
                zeile[k] = z * nle + ((j + k) % nle)
            zeilen.append(zeile)
        return zeilen

    @staticmethod
    def _probe_wert(q: int, t: int, spalten: int, geraet) -> torch.Tensor:
        """Der Inhalt einer Probezeile -- verschieden je (Rang, Token, Spalte).

        Ganze Zahlen unter 128: in bf16 exakt, in float32 exakt, und nach der
        128er-Blockquantisierung auf jedem Rang dieselbe Bitfolge, weil die
        Quantisierung je Zeile und je Block unabhaengig ist.
        """
        s = torch.arange(spalten, device=geraet, dtype=torch.int32)
        return (q * 131 + t * 17 + s) % 113

    def _beleg_struktur(self) -> Tuple[bool, str]:
        R, r, nle, K = self.welt, self.rank, self.num_local_experts, self.router_topk
        H, geraet = self.hidden_size, self.device

        routen = {q: self._probe_routen(q) for q in range(R)}
        meine = routen[r]
        T = len(meine)

        topk_ids = torch.tensor(meine, dtype=torch.int64, device=geraet)
        topk_gew = (
            torch.arange(T * K, device=geraet, dtype=torch.float32).reshape(T, K) % 7.0
        ) + 1.0
        muster = torch.stack(
            [self._probe_wert(r, t, H, geraet) for t in range(T)], dim=0
        )
        x = muster.to(torch.bfloat16 if self.use_fp8 else self.params_dtype)

        class _ProbeTopK:
            pass

        tk = _ProbeTopK()
        tk.topk_ids = topk_ids
        tk.topk_weights = topk_gew

        dist.barrier(group=self.cpu_group)
        zustand = self._dispatch_vorbereiten(x, tk)
        aus = self._dispatch_kern(*zustand)
        torch.cuda.synchronize(self.device)

        # Was muss angekommen sein? Jeder Rang rechnet die Sendeordnung JEDES
        # Rangs aus der Regel nach -- unabhaengig von der Buchhaltung, die
        # gerade geprueft wird.
        erwartet: List[Tuple[int, int]] = []  # (Quellrang, Tokennummer)
        for q in range(R):
            for t, zeile in enumerate(routen[q]):
                if any(e >= 0 and e // nle == r for e in zeile):
                    erwartet.append((q, t))
        N = aus.hidden_states.shape[0]
        if N != len(erwartet):
            return False, f"Dispatch: {N} Zeilen empfangen, erwartet {len(erwartet)}"

        recv_ids = aus.topk_ids.cpu()
        for p, (q, t) in enumerate(erwartet):
            soll = [
                (e - r * nle) if (e >= 0 and e // nle == r) else -1
                for e in routen[q][t]
            ]
            if recv_ids[p].tolist() != soll:
                return False, (
                    f"Dispatch: Zeile {p} (von Rang {q}, Token {t}) traegt "
                    f"topk_ids {recv_ids[p].tolist()}, erwartet {soll}"
                )

        # Nutzlast und Skalen byteweise gegen die lokal gebaute Sollform. Das
        # ist kein zweiter Nachbau der Buchhaltung: gebaut wird nur die
        # ERWARTETE Eingabe der fremden Raenge (aus der Regel), quantisiert
        # mit demselben Kern, und verglichen werden die Bytes.
        soll_x, soll_sk = self._probe_soll(erwartet)
        ist_x = aus.hidden_states.view(torch.uint8).reshape(N, -1)
        schlecht = int((ist_x != soll_x).sum().item())
        if schlecht:
            return False, (
                f"Dispatch: {schlecht} von {ist_x.numel()} Nutzbyte falsch"
            )
        if (soll_sk is None) != (aus.hidden_states_scale is None):
            return False, (
                f"Dispatch: Skalen erwartet={soll_sk is not None}, "
                f"angekommen={aus.hidden_states_scale is not None}"
            )
        if soll_sk is not None:
            ist_sk = aus.hidden_states_scale.contiguous().view(torch.uint8)
            schlecht = int((ist_sk != soll_sk.view(torch.uint8)).sum().item())
            if schlecht:
                return False, (
                    f"Dispatch: {schlecht} von {ist_sk.numel()} Skalenbyte falsch"
                )

        # -- Combine: zurueck genau das, was gekommen ist.
        rueck = (
            torch.stack(
                [
                    self._probe_wert(q, t, H, geraet).to(torch.bfloat16)
                    for (q, t) in erwartet
                ],
                dim=0,
            )
            if N
            else torch.zeros((0, H), dtype=torch.bfloat16, device=geraet)
        )
        ergebnis = self.combine(
            DeepEPNormalCombineInput(rueck, aus.topk_ids, aus.topk_weights)
        )
        torch.cuda.synchronize(self.device)

        raenge_je_token = torch.tensor(
            [len({e // nle for e in zeile if e >= 0}) for zeile in meine],
            dtype=torch.float32,
            device=geraet,
        ).unsqueeze(1)
        soll = (muster.to(torch.float32) * raenge_je_token).to(ergebnis.dtype)
        schlecht = int((ergebnis != soll).sum().item())
        if schlecht:
            return False, (
                f"Combine: {schlecht} von {soll.numel()} Werten falsch "
                f"(erwartet Eingabe * Zahl der zustaendigen Raenge)"
            )
        return True, ""

    def _probe_soll(self, erwartet):
        """Die Sollform der empfangenen Nutzlast, lokal gebaut.

        Ohne fp8 ist das die Regel selbst. Mit fp8 wird die Regel durch
        DENSELBEN Quantisierer geschickt, mit denselben Schaltern -- er
        arbeitet je Zeile und je 128er-Block unabhaengig, also ist das Ergebnis
        auf jedem Rang dieselbe Bitfolge wie beim Sender.
        """
        H, geraet = self.hidden_size, self.device
        if not erwartet:
            leer_x = torch.empty(
                (0, self._nutz_zeilenbytes()), dtype=torch.uint8, device=geraet
            )
            if not self.use_fp8:
                return leer_x, None
            return leer_x, torch.empty(
                (0, self._skalen_spalten()), dtype=self._skalen_dtype(), device=geraet
            )
        roh = torch.stack(
            [self._probe_wert(q, t, H, geraet) for (q, t) in erwartet], dim=0
        )
        if not self.use_fp8:
            w = roh.to(self.params_dtype).contiguous()
            return w.view(torch.uint8).reshape(len(erwartet), -1), None

        from sglang.srt.layers.quantization.fp8_kernel import (
            sglang_per_token_group_quant_fp8,
        )

        q8, s8 = sglang_per_token_group_quant_fp8(
            roh.to(torch.bfloat16).contiguous(),
            128,
            column_major_scales=deep_gemm_wrapper.DEEPGEMM_SCALE_UE8M0,
            scale_tma_aligned=deep_gemm_wrapper.DEEPGEMM_SCALE_UE8M0,
            scale_ue8m0=deep_gemm_wrapper.DEEPGEMM_SCALE_UE8M0,
        )
        return (
            q8.contiguous().view(torch.uint8).reshape(len(erwartet), -1),
            s8.contiguous(),
        )
