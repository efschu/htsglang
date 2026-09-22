"""#108 ERSTBOOT-ADOPTION: D nimmt P's Bytes, statt sie neu von Platte zu holen.

Nutzer-Order 22.09. 07:25Z, verbatim:

    "ich versteh auch nicht warum beim erstmaligen laden von D er nochmal
     alles von der platte zieht. das ist doch eigentlich schon der erste
     zeitpunkt wo man die layerbytes aus P fuer D wiederverwenden kann.
     dann laedt alles viel schneller und etwaige probleme sieht man auch
     direkt"

ZWEI GEWINNE, und der zweite ist der groessere. Gemessen fnFL2w51: P
laedt 28,24 s, D danach nochmal 71,69 s DENSELBEN Checkpoint, obwohl P
die Bytes bereits auf den Karten und im Store hat. Wichtiger aber: wer
beim ERSTEN Laden den Flip-Pfad benutzt, sieht dessen Fehler in der
ersten Minute statt nach der fuenften. Die Nacht vom 22.09. ist der
Beleg -- acht Boots (w42-w49) a ~5 min Ladezeit, und der Defekt zeigte
sich jedes Mal erst am Ende.

DER WEG: D laedt mit ``--load-format dummy``. ``DummyModelLoader`` ruft
``initialize_dummy_weights`` und danach ``process_weights_after_loading``
je Modul -- die Marlin-Repacks, der Presplit und die Pool-Geometrie
entstehen also IDENTISCH zum Plattenweg, nur ohne Bytes von Platte. In
diese fertige Struktur injiziert der Erstflip P's echte Bytes.

ZWEI RIEGEL, OHNE DIE DAS GEFAEHRLICH WAERE:

(1) D DARF UNTER ADOPTION NICHT IN DEN GETEILTEN STORE SCHREIBEN.
    Der Host-Store ist EINE Datei je Layer/Attribut fuer beide Gruppen
    (#107). Wuerde D seinen dummy-Presplit hineinspillen, ueberschriebe
    es P's echte Experten-Bytes mit Zufallszahlen -- Datenverlust, und
    zwar stiller: die Dateigroesse bliebe korrekt. :func:`store_writes_denied`
    ist dieser Riegel.

(2) D DARF NICHT ANTWORTEN, SOLANGE DIE GEWICHTE ZUFALL SIND.
    Zwischen dem dummy-Laden und dem verifizierten Erstflip stehen in
    D's Tensoren Zufallswerte. Ein Request in diesem Fenster bekaeme
    syntaktisch gueltigen Unsinn -- der schlimmste Fehlerfall, weil er
    wie ein Ergebnis aussieht. :func:`weights_are_placeholder` haelt den
    Zustand, und er wird NUR durch :func:`mark_adopted` geloescht, den
    der Erstflip nach erfolgreichem Inject ruft.
"""

from __future__ import annotations

import os
from typing import Optional

#: Vom Launcher an die D-Gruppe publiziert (wie alle weg2-Entscheidungen
#: ueber `build_env`, nie ueber einen zweiten Kanal).
ADOPT_ENV = "SGLANG_WEG2_D_ADOPT"

ADOPT_ON = "on"
ADOPT_OFF = "off"

#: Prozess-lokal: steht D noch auf den dummy-Gewichten?
_PLACEHOLDER = {"pending": False, "reason": ""}

REFUSAL_MARKER = "W115 Weg2AdoptWeightsArePlaceholder"


class Weg2AdoptWeightsArePlaceholder(RuntimeError):
    """W115 -- eine Anfrage traf D, bevor der Erstflip die Bytes brachte."""


def adopt_armed(explicit: Optional[str] = None) -> bool:
    """Faehrt dieser Rang die Erstboot-Adoption?

    Wie ``inject_mode``/``weights_cpu_backup``: der Launcher-Prozess hat
    das Flag auf seinem argv (``explicit``), ein Rang hat es nicht und
    liest die Env. Zwei Leser, eine Antwort -- die Trennung existiert,
    weil ein Launcher-Aufrufer ohne ``explicit`` sonst still seine eigene
    (ungesetzte) Umgebung liest; S6 Fix E hat genau das gemessen.
    """
    raw = explicit if explicit is not None else os.environ.get(ADOPT_ENV, "")
    return str(raw).strip().lower() == ADOPT_ON


def arm_placeholder(reason: str = "dummy-load") -> None:
    """D hat mit Platzhaltern geladen -- ab jetzt keine Generierung."""
    _PLACEHOLDER["pending"] = True
    _PLACEHOLDER["reason"] = str(reason)


def mark_adopted(filled: int, expected: int) -> None:
    """Der Erstflip hat die Bytes gebracht. NUR mit vollstaendiger Deckung.

    ``filled``/``expected`` kommen aus der Halter-Karte: sie sagt, welche
    Tensoren dieser Rang halten muss. Deckt der Inject sie nicht alle,
    bleibt der Riegel stehen -- ein halb gefuelltes Modell ist gefaehrlicher
    als ein leeres, weil es rechnet.
    """
    if int(filled) < int(expected) or int(expected) <= 0:
        _PLACEHOLDER["reason"] = (
            f"Erstflip deckte {filled} von {expected} erwarteten Tensoren")
        return
    _PLACEHOLDER["pending"] = False
    _PLACEHOLDER["reason"] = ""


def weights_are_placeholder() -> bool:
    """Stehen in diesem Rang noch dummy-Gewichte?"""
    return bool(_PLACEHOLDER["pending"])


def placeholder_reason() -> str:
    return str(_PLACEHOLDER["reason"])


def refuse_if_placeholder() -> None:
    """Der Riegel am Generierungspfad. Wirft, statt Unsinn zu rechnen."""
    if weights_are_placeholder():
        raise Weg2AdoptWeightsArePlaceholder(
            f"{REFUSAL_MARKER}: dieser Rang haelt PLATZHALTER-Gewichte "
            f"({placeholder_reason() or 'dummy-load'}). Der Erstflip hat sie "
            f"nicht (vollstaendig) ersetzt. Eine Antwort waere syntaktisch "
            f"gueltiger Unsinn -- der Fehlerfall, der wie ein Ergebnis "
            f"aussieht.")


def store_writes_denied() -> bool:
    """Darf dieser Rang in den GETEILTEN Host-Store schreiben?

    Unter Adoption NEIN, solange die Gewichte Platzhalter sind: der Store
    ist EINE Datei fuer beide Gruppen (#107), und ein dummy-Presplit
    wuerde P's echte Experten-Bytes mit Zufall ueberschreiben, ohne dass
    sich Groesse oder Struktur aendern. Lesen bleibt erlaubt -- genau
    davon lebt die Adoption.
    """
    return adopt_armed() and weights_are_placeholder()
