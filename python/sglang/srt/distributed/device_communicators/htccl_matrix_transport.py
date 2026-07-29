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


def _fenster_bytes() -> int:
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

    def __init__(self, cpu_group, device):
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
        self.rank = dist.get_rank(cpu_group)
        self.welt = dist.get_world_size(cpu_group)

        # 1. Direktpfad. `None` heisst: diese Maschine kann ihn nicht, mit
        #    protokolliertem Grund. Kein Werfen -- der Planer ist auch ohne
        #    ihn nuetzlich.
        self.bar1 = baue_bar1(cpu_group, device, _fenster_bytes())

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
                                empfangs_bytes):
        self._muss_stehen()
        return self.bar1.htccl_all_to_all_single(
            comm, output, inp, sende_bytes, empfangs_bytes
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
