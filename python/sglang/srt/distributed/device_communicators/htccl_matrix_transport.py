# SPDX-License-Identifier: Apache-2.0
"""HTCCL-Transport ``matrix``: Planer plus BAR1-Direktpfad.

Die Naht, die dieses Modul schliesst
------------------------------------
``htccl_matrix.py`` plant, ``htccl_bar1.py`` transportiert. Zwischen beiden
fehlte bisher das Stueck, das

1. den Direktpfad aufbaut (``baue_bar1``),
2. dessen **tatsaechliche** Faehigkeit als ``fenster_bytes`` in den Planer
   reicht -- das Minimum ueber alle Ziele und alle Raenge, nicht die
   Bruttogroesse aus sysfs und nicht die angeforderte Groesse,
3. ``plan()`` ruft und ``plan.erklaerung()`` auf Rang 0 protokolliert,
4. den Plan an den Direktpfad zurueckgibt, damit ``handles`` und die
   Kernwahl je Groesse aus derselben Quelle kommen.

Die Reihenfolge ist nicht beliebig. Der Planer schliesst Algorithmen aus,
deren Fensterbedarf die Abbildung sprengt (``plane(..., fenster_bytes=)``);
diese Zahl kennt erst der aufgebaute Transport. Umgekehrt braucht der
Transport den Plan, um je Groesse Netz oder Ring zu waehlen. Also: erst
bauen, dann planen, dann den Plan hereinreichen.

Warum der Direktpfad zugleich der Messfuehler ist
------------------------------------------------
``HTCCLBar1Transport`` erfuellt das ``Fuehler``-Protokoll
(``name``/``eigenlast``/``eigenlast_duplex``/``paar``/``paar_empfang``).
Steht er, misst der Planer **echte gerichtete Kanten** statt der
Eigenlast-Schaetzung. Steht er nicht, faellt der Planer auf die Eigenlast
zurueck und schreibt das in die Erklaerung -- er tut nicht so, als haette
er die Kante gemessen.

Was passiert, wenn der Direktpfad fehlt
---------------------------------------
Dann meldet dieser Transport ``handles(...) == False`` fuer alles. Der
Planer laeuft trotzdem und protokolliert seine Erklaerung -- die Wahl von
Rollen und Algorithmus ist auch fuer die anderen Transporte eine
Information. Aber es wird nichts ueber einen Pfad geschickt, den es nicht
gibt.

Ein- und ausschalten
--------------------
Nichts davon passiert ohne ausdrueckliche Wahl:

* ``SGLANG_HTCCL_TRANSPORT=matrix`` waehlt diesen Transport,
  ``SGLANG_HTCCL_TRANSPORT=bar1`` den nackten Direktpfad ohne Planer.
  Vorgabe bleibt ``device``; ohne Umschalten aendert sich nichts.
* ``SGLANG_HTCCL_MATRIX_DIRECT=0`` schaltet den Direktpfad ab, der Planer
  laeuft weiter.
* ``SGLANG_HTCCL_BAR1_FENSTER_MIB`` (Vorgabe 96) ist die **angeforderte**
  Regionsgroesse je Rang. 96 MiB, weil die vermessene Region 90,69 MiB
  gross ist und in das 256-MiB-BAR1 einer RTX 3080 passt.
"""

from __future__ import annotations

import logging
import os
from typing import Optional

logger = logging.getLogger(__name__)

#: Angeforderte Empfangsregion je Rang, in MiB.
FENSTER_MIB_VORGABE = 96

#: Was in BAR1 unangetastet bleibt. RM belegt selbst einen Teil der
#: Apertur, und die Zahl aus sysfs ist BRUTTO. 32 MiB ist eine Vorgabe und
#: keine Messaussage -- wer NVML hat, braucht sie gar nicht, weil dann die
#: echte freie Groesse bekannt ist.
RESERVE_MIB_VORGABE = 32

#: Fensterkasse: je Geraeteordinal die Liste ``(gruppe, bytes)``, die dieser
#: PROZESS bereits in BAR1 festgenagelt hat.
#:
#: Es gibt sie, weil die erste Gruppe bisher nahm, was sie wollte. Mit
#: ``SGLANG_UNEVEN_DCP=1`` gibt es zwei Kommunikatorgruppen; ``tp`` griff
#: sich 96 MiB, und ``dcp`` bekam vom Halter ein nacktes ``[Errno 12]``.
#: Ein ENOMEM aus einem Ioctl sagt nicht, WER den Platz hat. Diese Tabelle
#: sagt es.
_KASSE: dict[int, list[tuple[str, int]]] = {}


def _ordinal(device) -> int:
    idx = getattr(device, "index", None)
    if idx is not None:
        return int(idx)
    import torch

    return int(torch.cuda.current_device())


def _gruppen_schluessel(gruppe: str) -> str:
    """``dcp`` -> ``DCP``, ``tp:0`` -> ``TP_0``. Fuer den Variablennamen."""
    return "".join(c if c.isalnum() else "_" for c in gruppe).upper()


def _angefordert(gruppe: str) -> tuple[int, str]:
    """Die angeforderte Regionsgroesse dieser Gruppe, und woher sie kommt.

    **Je Gruppe getrennt einstellbar, und das ist der Kern der Sache.** Die
    Gruppen tragen verschiedene Nachrichten: die tp-Gruppe im Prefill 20 MiB
    (chunked_prefill_size 2048 x hidden 5120 x 2 B), die dcp-Gruppe etwas
    ganz anderes. Ihnen dieselbe Region zu geben heisst, entweder der einen
    zu wenig oder der anderen zu viel zu geben -- und BAR1 ist auf einer
    3080 mit 256 MiB brutto zu knapp fuer "zu viel".

    ``SGLANG_HTCCL_BAR1_FENSTER_MIB_DCP=16`` setzt die dcp-Gruppe herunter,
    ohne die tp-Gruppe anzufassen.
    """
    eigen = f"SGLANG_HTCCL_BAR1_FENSTER_MIB_{_gruppen_schluessel(gruppe)}"
    if gruppe and eigen in os.environ:
        return int(os.environ[eigen]) * 1024 * 1024, eigen
    return (
        int(os.environ.get("SGLANG_HTCCL_BAR1_FENSTER_MIB",
                           str(FENSTER_MIB_VORGABE))) * 1024 * 1024,
        "SGLANG_HTCCL_BAR1_FENSTER_MIB",
    )


def bar1_frei(device) -> tuple[Optional[int], int, str]:
    """``(frei, brutto, quelle)`` der BAR1-Apertur dieser Karte.

    ``frei`` ist ``None``, wenn es sich nicht ermitteln liess -- dann muss
    der Aufrufer mit ``brutto`` minus Reserve rechnen und das auch sagen.
    Geraten wird hier nichts.

    NVML (``nvmlDeviceGetBAR1MemoryInfo``) ist die einzige Quelle, die
    ``used``/``free`` wirklich kennt; sysfs kennt nur die Bruttogroesse der
    Apertur, und wieviel davon RM selbst belegt, steht dort nirgends. Das
    ist genau die Luecke, in die der ENOMEM des Halters faellt.
    """
    brutto = 0
    try:
        from sglang.srt.distributed.device_communicators.htccl_bar1 import (
            bar1_fenster,
        )
        from sglang.srt.distributed.device_communicators.htccl_matrix import (
            bdf_der_karte,
        )

        brutto = bar1_fenster(bdf_der_karte(device)).groesse
    except Exception as e:
        logger.debug("HTCCL-BAR1: BAR1-Bruttogroesse nicht aus sysfs (%r)", e)
    try:
        import pynvml

        pynvml.nvmlInit()
        try:
            h = pynvml.nvmlDeviceGetHandleByIndex(_ordinal(device))
            info = pynvml.nvmlDeviceGetBAR1MemoryInfo(h)
            return int(info.bar1Free), (brutto or int(info.bar1Total)), "nvml"
        finally:
            pynvml.nvmlShutdown()
    except Exception as e:
        logger.debug("HTCCL-BAR1: NVML liefert keine BAR1-Belegung (%r)", e)
    return None, brutto, "sysfs-brutto"


def kasse_eintragen(device, gruppe: str, bytes_: int) -> None:
    _KASSE.setdefault(_ordinal(device), []).append((gruppe, int(bytes_)))


def kasse_austragen(device, gruppe: str) -> None:
    posten = _KASSE.get(_ordinal(device))
    if not posten:
        return
    for i, (g, _) in enumerate(posten):
        if g == gruppe:
            posten.pop(i)
            return


def kasse_stand(device) -> list[tuple[str, int]]:
    return list(_KASSE.get(_ordinal(device), []))


def fenster_fuer(gruppe: str, device) -> int:
    """Die Regionsgroesse, die diese Gruppe auf DIESEM Rang anfordern darf.

    Der lokale VORSCHLAG, nicht die Entscheidung: die Karten der Gruppe
    haben verschieden grosse Aperturen (3080 mit 256 MiB, 5090 mit deutlich
    mehr), und eine je Rang verschiedene Region waere eine je Rang
    verschiedene Schlitzordnung -- also falsche Adressen statt eines
    Fehlers. Das Minimum ueber die Gruppe zieht ``_baue_auf``.

    Gerechnet wird gegen das, was WIRKLICH frei ist, abzueglich dessen, was
    dieser Prozess in anderen Gruppen bereits festgenagelt hat. Die
    Verkleinerung wird protokolliert, mit der Rechnung -- eine stille
    Verkleinerung waere schlimmer als der ENOMEM: sie senkt still die
    groesste tragbare Nutzlast, und Nachrichten darueber fallen ohne einen
    einzigen Hinweis auf die gloo-Ebene zurueck.
    """
    angefordert, quelle = _angefordert(gruppe)
    frei, brutto, woher = bar1_frei(device)
    reserve = int(os.environ.get("SGLANG_HTCCL_BAR1_RESERVE_MIB",
                                 str(RESERVE_MIB_VORGABE))) * 1024 * 1024
    schon = sum(b for _, b in kasse_stand(device))

    if frei is None:
        if brutto <= 0:
            logger.info(
                "HTCCL-BAR1: BAR1-Groesse dieser Karte unbekannt (weder NVML "
                "noch sysfs). Es wird angefordert, was %s sagt (%d MiB); "
                "reicht die Apertur nicht, meldet sich der Halter mit ENOMEM.",
                quelle, angefordert // 2**20,
            )
            return angefordert
        # sysfs kennt nur BRUTTO. Was RM selbst belegt, steht dort nicht --
        # deshalb muss hier die eigene Buchfuehrung abgezogen werden, und
        # deshalb ist diese Schaetzung optimistisch. Genau daran ist die
        # zweite Gruppe mit ENOMEM gescheitert.
        deckel = brutto - reserve - schon
        rechnung = (
            f"BAR1 brutto laut sysfs {brutto // 2**20} MiB - Reserve "
            f"{reserve // 2**20} MiB - in diesem Prozess bereits "
            f"festgenagelt {schon // 2**20} MiB = {max(deckel, 0) // 2**20} "
            f"MiB. ACHTUNG: sysfs kennt nur die Bruttoapertur; was RM selbst "
            f"belegt, ist darin NICHT abgezogen. Ohne NVML ist diese Zahl "
            f"eine Obergrenze, keine Zusage."
        )
    else:
        # NVML kennt `used` -- darin steckt bereits, was dieser Prozess in
        # anderen Gruppen festgenagelt hat. `schon` deshalb NICHT noch
        # einmal abziehen; es steht unten nur zur Zuordnung dabei.
        deckel = frei - reserve
        rechnung = (
            f"frei laut NVML {frei // 2**20} MiB - Reserve "
            f"{reserve // 2**20} MiB = {max(deckel, 0) // 2**20} MiB "
            f"(darin bereits enthalten, was dieser Prozess haelt: "
            f"{', '.join(f'{g}: {b // 2**20} MiB' for g, b in kasse_stand(device)) or 'nichts'})"
        )

    if deckel >= angefordert:
        return angefordert

    logger.warning(
        "HTCCL-BAR1: Gruppe %r fordert %d MiB (%s), nutzbar sind aber nur "
        "%d MiB. %s Die Anforderung wird auf %d MiB gekuerzt, und das SENKT "
        "die groesste tragbare Nutzlast dieser Gruppe -- Nachrichten "
        "darueber fallen ohne weiteren Hinweis auf die gloo-Ebene zurueck. "
        "Wer das nicht will, setzt SGLANG_HTCCL_BAR1_FENSTER_MIB_%s "
        "ausdruecklich, gibt der anderen Gruppe weniger, oder laesst diese "
        "Gruppe bewusst ueber NCCL fahren.",
        gruppe or "<ohne Namen>", angefordert // 2**20, quelle,
        max(deckel, 0) // 2**20, rechnung, max(deckel, 0) // 2**20,
        _gruppen_schluessel(gruppe),
    )
    return max(deckel, 0)


def _fenster_bytes() -> int:
    """Die alte, gruppenlose Form. Nur noch fuer Aufrufer ohne Gruppennamen
    (``benchmark/bar1_diag.py``, ``benchmark/bar1_graph_check.py``): dort
    gibt es genau eine Gruppe, also auch nichts aufzuteilen."""
    return int(os.environ.get("SGLANG_HTCCL_BAR1_FENSTER_MIB",
                              str(FENSTER_MIB_VORGABE))) * 1024 * 1024


class HTCCLMatrixTransport:
    """Zusammengesetzter Transport: Plan + Unterpfad je Operation und Groesse.

    Heute gibt es genau **einen** Unterpfad (BAR1) und zwei Operationen
    (``all_reduce``, ``all_to_all_single``). Das ist die ehrliche Fassung:
    NIC- und System-RAM-Kanten je gerichteter Kante zu mischen ist entworfen
    (``ENTWURF_PFADMATRIX.md``), aber nicht gemessen, und ein
    Auswahlgeruest fuer Pfade, die es nicht gibt, waere eine Attrappe.

    Die Auswahl, die es wirklich gibt, ist die zwischen den **Kernen** von
    ``all_reduce``: ``netz`` oder ``ring`` je Groesse, aus dem Plan statt
    aus einer eingebauten Zahl. ``all_to_all`` hat keine solche Wahl -- es
    gibt genau einen Weg -- und wird deshalb ungeplant durchgereicht.
    """

    HTCCL_OPS: frozenset = frozenset(
        {"all_reduce", "all_to_all", "all_to_all_single"}
    )

    def __init__(self, cpu_group, device, gruppe: str = ""):
        import torch.distributed as dist

        from sglang.srt.distributed.device_communicators.htccl_bar1 import (
            baue_bar1,
        )
        from sglang.srt.distributed.device_communicators.htccl_matrix import (
            HTCCLMatrixPlaner,
            lade_konfig,
        )

        self.cpu_group = cpu_group
        self.device = device
        self.gruppe = gruppe
        self.rank = dist.get_rank(cpu_group)
        self.welt = dist.get_world_size(cpu_group)

        # 1. Direktpfad. `None` heisst: diese Maschine kann ihn nicht, mit
        #    protokolliertem Grund. Kein Werfen -- der Planer ist auch ohne
        #    ihn nuetzlich. Der GRUND wird aber festgehalten: ein Planer ohne
        #    Direktpfad sagt zu allem `handles -> False`, und dann laeuft
        #    jedes Kollektiv ueber die gloo-Ebene, waehrend das Protokoll
        #    "transport=matrix" meldet. Genau diese Verwechslung hat eine
        #    Messung entwertet.
        bericht: dict = {}
        self.bar1 = baue_bar1(
            cpu_group, device, fenster_fuer(gruppe, device), bericht,
            gruppe=gruppe,
        )
        if bericht.get("haelt_belegt") and self.bar1 is not None:
            # Er steht, traegt aber nichts (Byte-Beleg gefallen). Abraeumen
            # statt liegenlassen: er haelt sonst die BAR1-Seiten fest, die
            # die naechste Gruppe braucht.
            self.bar1.close()
            self.bar1 = None
        self.bar1_grund = bericht.get("grund", "")
        self.bar1_stufe = bericht.get("stufe", "")

        # 2. Faehigkeit an den Planer. Minimum ueber ALLE Ziele und alle
        #    Raenge; `None` heisst "unbekannt" und schliesst nichts aus --
        #    ausdruecklich nicht "unbegrenzt".
        fenster = self.bar1.fenster_minimum() if self.bar1 is not None else None

        # 3. Planen. Der Direktpfad ist zugleich der Paar-Fuehler, wenn er
        #    steht: dann misst der Planer echte gerichtete Kanten.
        planer = HTCCLMatrixPlaner(
            cpu_group, device, konfig=lade_konfig(),
            fuehler=self.bar1, fenster_bytes=fenster,
        )
        self.plan = planer.plan()

        # 4. Erklaerung. Pflichtausgabe und nicht an ein Debug-Flag
        #    gebunden -- ohne sie debuggt niemand auf fremder Hardware.
        #    Nur auf Rang 0, weil der Plan gruppenweit identisch ist (die
        #    Pruefsumme ist beim Planen abgeglichen worden) und R Kopien
        #    desselben Blocks das Protokoll unlesbar machen.
        if self.rank == 0:
            logger.info("%s", self.plan.erklaerung())

        # 5. Plan zurueck in den Direktpfad: ab hier waehlt er Netz oder
        #    Ring aus derselben Quelle, aus der der Planer sie begruendet.
        if self.bar1 is not None:
            self.bar1.setze_plan(self.plan)
            logger.info(
                "HTCCL-Matrix: Direktpfad steht. Abgebildetes Fenster "
                "gruppenweit %d KiB, groesste Nutzlast %d KiB, Leiter: %s.",
                (fenster or 0) // 1024, self.bar1.max_bytes // 1024,
                ", ".join(
                    (f"bis {s.max_bytes // 1024} KiB" if s.max_bytes > 0
                     else "darueber") + f": {s.algorithmus}"
                    for s in self.plan.leiter
                ),
            )
        else:
            logger.info(
                "HTCCL-Matrix: kein Direktpfad -- der Plan ist protokolliert, "
                "aber handles() gibt fuer alles False. Es wird nichts ueber "
                "einen Pfad geschickt, den es nicht gibt."
            )

    # -- Transport-Naht ----------------------------------------------------

    def handles(self, op: str, nbytes: int) -> bool:
        """Genau dann, wenn ein Unterpfad die Operation wirklich fahren kann.

        Die Entscheidung wird an den Unterpfad weitergereicht statt hier
        nachgebaut: eine zweite Fassung derselben Bedingung waere die
        Stelle, an der die beiden auseinanderlaufen -- und ein Transport,
        der zusagt und dann scheitert, ist schlechter als einer, der sich
        abmeldet.
        """
        if op not in self.HTCCL_OPS or self.bar1 is None:
            return False
        return self.bar1.handles(op, nbytes)

    def htccl_all_reduce(self, comm, inp):
        self._muss_stehen()
        return self.bar1.htccl_all_reduce(comm, inp)

    # -- all_to_all --------------------------------------------------------
    #
    # Der Planer hat dazu NICHTS zu sagen: er waehlt zwischen den
    # all_reduce-Zerlegungen (netz/ring/stern/hierarchisch), und all_to_all
    # hat keine Zerlegung -- es gibt genau einen Weg, jeder schreibt jedem
    # seinen Block. Deshalb wird hier nur durchgereicht, ohne dass der Plan
    # gefragt wuerde. Eine Planzeile fuer eine Wahl, die es nicht gibt, waere
    # eine Attrappe.

    def traegt_a2a(self, groesster_block: int) -> bool:
        if self.bar1 is None:
            return False
        return self.bar1.traegt_a2a(groesster_block)

    def a2a_schlitz_bytes(self) -> int:
        return 0 if self.bar1 is None else self.bar1.a2a_schlitz_bytes()

    def htccl_all_to_all_single(self, comm, output, inp, sende_bytes,
                                empfangs_bytes, sende_versatz=None,
                                empfangs_versatz=None):
        self._muss_stehen()
        return self.bar1.htccl_all_to_all_single(
            comm, output, inp, sende_bytes, empfangs_bytes,
            sende_versatz, empfangs_versatz,
        )

    def _muss_stehen(self) -> None:
        if self.bar1 is None:
            raise NotImplementedError(
                "Der Matrix-Transport hat heute genau einen Unterpfad (BAR1), "
                "und der steht nicht. Erreichbar ist diese Zeile nur, wenn "
                "jemand handles() umgangen hat."
            )

    def close(self) -> None:
        if self.bar1 is not None:
            self.bar1.close()
            self.bar1 = None
