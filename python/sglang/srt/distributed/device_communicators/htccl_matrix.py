# SPDX-License-Identifier: Apache-2.0
"""HTCCL-Pfadmatrix: der Planer.

Der Planer misst beim Start die **gerichteten** Kanten des Verbundes und
leitet daraus Rollen, Domaenen, Algorithmus, Chunkgroesse, Aufteilung und
einen Fan-in-Staffelplan ab.

Vier Grundsaetze, die den ganzen Aufbau tragen:

1. **Gemessen, nicht abgeleitet.** Es wird keine ``lspci``-Breite als
   Wahrheit genommen. ``lspci`` sagt x8; ob die Kante 12,7 GB/s traegt,
   entscheidet die Messung. Die Linkbreite darf hoechstens als
   Startschaetzung dienen und wird deshalb nur fuer *Unentschieden*
   (Ringreihenfolge) und fuer die Plausibilitaetsmeldung gelesen, nie als
   Entscheidungsgrundlage.
2. **Faehigkeit und Politik getrennt.** Was die Hardware kann, wird
   gemessen (``Messung``); was benutzt wird, wird konfiguriert
   (``HtcclKonfig``). Beides beruehrt sich erst in ``plane()``.
3. **Erklaerbar.** ``Plan.erklaerung()`` gibt die gemessenen Kapazitaeten,
   die daraus folgenden Rollen und den gewaehlten Algorithmus samt
   Vorhersagezeiten aus. Ohne das debuggt niemand auf fremder Hardware.
4. **Auf jeder Ebene ueberstimmbar.** Planer aus, einzelne Rolle
   festnageln, Domaenen von Hand, Algorithmus erzwingen, Chunk erzwingen,
   Aufteilung erzwingen -- und das ist zugleich der einzige Weg, den Planer
   gegen sich selbst zu messen.

**Der Eintrag gehoert an die gerichtete Kante (Quelle->Ziel), nicht an die
Verbindung.** PCIe ist vollduplex, Aus- und Eingang sind getrennte
Budgets; die Zuordnung darf asymmetrisch sein.

Entscheidungsregeln und ihre Belege
-----------------------------------
Alle vier stammen aus Messungen dieses Rigs; die Belege stehen in
``/spinning/nvidia-smallbar-p2p/MESSUNG_NEBENLAEUFIGKEIT.md``. Sie sind
hier als *Modell* codiert, dessen Parameter aus der Startmessung kommen --
nicht als eingebaute Zahlen.

R1  **Netz gegen Ring.** Unterhalb der Saettigung ist die Gleichzeitigkeit
    des Netzes gratis (gemessener Quotient 0,99 bei 20 KiB), und das Netz
    braucht nur 2 statt 2(R-1) Schritte -> Netz. An der Saettigung bringt
    die Gleichzeitigkeit nichts (1,03x bei 1 MiB) **und** die Aufteilung
    ist gleichmaessig statt proportional, verschenkt also die schnellen
    Kanten -> Ring. Codiert in ``_zeit_netz`` / ``_zeit_ring``: der
    Netz-Term teilt den gemessenen Fan-in-Deckel *gleichmaessig* je Quelle
    auf, der Ring-Term nicht.

R2  **Blaetter.** Wessen gemessene Kapazitaet deutlich unter dem Median
    liegt, traegt keinen Transitverkehr. Liegen alle gleichauf, gibt es
    keine Blaetter und es bleibt beim flachen Verfahren. Schwelle
    ``blatt_schwelle`` (Vorgabe 0,6 x Median).

R3  **Fan-in-Deckel.** In keine Karte gehen mehr als der gemessene Deckel
    hinein, unabhaengig von der Zahl der Quellen (dieses Rig: ~13 GB/s).
    Der Deckel wird **gleichmaessig je Quelle** aufgeteilt, nicht
    proportional -- die x8-Quelle fiel von 12,81 auf 6,75 GB/s, die
    x4-Quelle behielt ihre 6,46. Folge: eine schnelle und eine langsame
    Karte nie gleichzeitig ins selbe Ziel schreiben lassen, sondern
    staffeln (``Plan.staffeln``).

R4  **Vollduplex verdoppelt nicht.** Bei Saettigung faellt jede Richtung
    auf ~65 %, Summe 1,32x. Unterhalb der Saettigung ist es fast gratis
    (0,99 bei 20 KiB). Der Faktor wird gemessen (``Messung.duplex``) und
    geht als Ueberlappungsgutschrift in die Pipelining-Rechnung ein --
    nirgends steht eine 2 im Code.

Was der Planer NICHT tut
------------------------
Er waehlt keinen Pfad je Kante (BAR1 / NIC / System-RAM). Das ist Aufgabe
des zusammengesetzten Transports (``htccl_bar1.py`` liefert einen der
Unterpfade). Der Planer beantwortet die davon getrennte Frage: welche
*Rolle* hat jeder Rang und mit welchem *Algorithmus* wird zerlegt. Die
beiden Ebenen treffen sich ueber ``Fuehler``: wer einen echten
Punkt-zu-Punkt-Pfad hat, reicht ihn als Paar-Fuehler herein und bekommt
echte Kantenmessungen statt der Eigenlast-Schaetzung.

Rangeinheitlichkeit
-------------------
Der Plan MUSS auf jedem Rang identisch sein -- die SPMD-Annahme der
Kollektive haengt daran. Deshalb: messen, Rohwerte quantisiert per
``all_gather_object`` teilen, jeder Rang rechnet aus **denselben** Daten
mit demselben Code, danach Abgleich der Planpruefsumme ueber die Gruppe.
Weicht ein Rang ab, ist das ein benannter Startfehler, keine Warnung.
"""

from __future__ import annotations

import hashlib
import json
import logging
import math
import os
import pathlib
import statistics
import time
from dataclasses import asdict, dataclass, field, replace
from typing import Any, Iterable, Mapping, Optional, Protocol, Sequence

logger = logging.getLogger(__name__)

# Fingerabdruck-Bestandteil: aendert sich die Planlogik, ist ein alter
# Zwischenspeicher ungueltig, auch wenn die Hardware dieselbe ist.
PLANER_VERSION = 1

# Vorgabe-Groessenleiter der Messung, in KiB. Die drei Punkte sind die
# Betriebspunkte, um die es geht: Decode (20), Uebergang (80), Prefill (1024).
VORGABE_GROESSEN_KIB = (20, 80, 1024)

ALGORITHMEN = ("netz", "ring", "stern", "hierarchisch")
PLANER_MODI = ("auto", "fest", "aus")
NIC_MODI = ("nie", "bei_bedarf", "immer")
ROLLEN = ("blatt", "domaene", "nabe")

# ---------------------------------------------------------------------------
# Referenzwerte dieses Rigs -- NUR zur Plausibilitaetsmeldung.
#
# Sie gehen in KEINE Entscheidung ein. Weicht eine Startmessung stark ab,
# wird das protokolliert, damit ein kaputter Messaufbau auffaellt, bevor er
# als Rig-Eigenheit durchgeht. Beleg: MESSUNG_NEBENLAEUFIGKEIT.md.
# ---------------------------------------------------------------------------
_BELEG_FANIN_GBPS = 13.16          # Deckel bei 1 MiB, zwei Quellen
_BELEG_DUPLEX_SUMME_1MIB = 1.32    # Summe beider Richtungen / eine Richtung
_BELEG_DUPLEX_SUMME_20KIB = 1.47


# ===========================================================================
# Konfiguration
# ===========================================================================


class KonfigFehler(ValueError):
    """Benannter Fehler in der Konfiguration -- nie stillschweigend geheilt."""


@dataclass(frozen=True)
class MessKonfig:
    """Was der Planer beim Start misst und wie lange er dafuer darf."""

    groessen_kib: tuple[int, ...] = VORGABE_GROESSEN_KIB
    wiederholungen: int = 32
    vorlauf: int = 8
    # Obergrenze fuer die gesamte Startmessung. Wird sie ueberschritten,
    # duennt der Planer die Groessenleiter und dann die Wiederholungen aus
    # und protokolliert das -- er bricht NICHT ab und rechnet auch nicht
    # heimlich mit weniger Belegen weiter, ohne es zu sagen.
    budget_ms: float = 2000.0
    # Fan-in (R3) und Duplex (R4) kosten eigene Runden. Abschaltbar, weil
    # sie nur mit einem Paar-Fuehler echte Werte liefern.
    fanin: bool = True
    duplex: bool = True
    # Zwischenspeicher der Matrix samt Fingerabdruck. None = Vorgabepfad.
    cache: Optional[str] = None
    cache_aus: bool = False


@dataclass(frozen=True)
class KollektivKonfig:
    planer: str = "auto"                 # auto | fest | aus
    algorithmus: str = "auto"            # auto | netz | ring | stern | hierarchisch
    chunk_kib: Optional[int] = None      # None == "auto"
    blatt_schwelle: float = 0.6          # Kapazitaet < 0,6 x Median -> Blatt
    aufteilung: Any = "auto"             # auto | gleich | proportional | {bdf: [...]}
    rollen: Mapping[str, str] = field(default_factory=dict)      # bdf -> Rolle
    domaenen: tuple[tuple[str, ...], ...] = ()                   # Listen von BDFs
    # R3: innerhalb einer Fan-in-Welle duerfen sich die Quellkapazitaeten
    # hoechstens um diesen Faktor unterscheiden. Darueber wird gestaffelt.
    staffel_verhaeltnis: float = 1.5
    # Anteil der gemessenen Kapazitaet, ab dem eine Kante als gesaettigt
    # gilt. Nur fuer die Erklaerung/Meldung -- die Wahl selbst faellt ueber
    # den Kostenvergleich, nicht ueber diese Schwelle.
    saettigung_anteil: float = 0.75
    mess: MessKonfig = field(default_factory=MessKonfig)


@dataclass(frozen=True)
class HtcclKonfig:
    kollektiv: KollektivKonfig = field(default_factory=KollektivKonfig)
    nic: str = "nie"                     # nie | bei_bedarf | immer


# -- Datei -----------------------------------------------------------------

_MESS_SCHLUESSEL = {
    "groessen_kib", "wiederholungen", "vorlauf", "budget_ms",
    "fanin", "duplex", "cache", "cache_aus",
}
_KOLLEKTIV_SCHLUESSEL = {
    "planer", "algorithmus", "chunk_kib", "blatt_schwelle", "aufteilung",
    "rollen", "domaenen", "staffel_verhaeltnis", "saettigung_anteil", "mess",
}
_WURZEL_SCHLUESSEL = {"kollektiv", "nic"}


def _bool(wert: Any, wo: str) -> bool:
    if isinstance(wert, bool):
        return wert
    if isinstance(wert, str):
        s = wert.strip().lower()
        if s in ("1", "ja", "true", "an", "yes"):
            return True
        if s in ("0", "nein", "false", "aus", "no"):
            return False
    raise KonfigFehler(f"{wo}: {wert!r} ist kein Wahrheitswert")


def _pruefe_schluessel(gegeben: Iterable[str], erlaubt: set[str], wo: str) -> None:
    unbekannt = sorted(set(gegeben) - erlaubt)
    if unbekannt:
        raise KonfigFehler(
            f"{wo}: unbekannte Schluessel {unbekannt}; erlaubt sind "
            f"{sorted(erlaubt)}. (Ein stillschweigend ignorierter Tippfehler "
            f"in der Konfiguration ist genau die Sorte Fehler, nach der man "
            f"spaeter Leistung sucht, die per Konfiguration abgeschaltet war.)"
        )


def _norm_bdf(s: str) -> str:
    """``05:00.0`` und ``0000:05:00.0`` sind derselbe Schluessel."""
    s = str(s).strip().lower()
    if s.count(":") == 1:
        s = "0000:" + s
    return s


def _lies_datei(pfad: str) -> dict:
    p = pathlib.Path(pfad).expanduser()
    if not p.is_file():
        raise KonfigFehler(f"Konfigurationsdatei {pfad!r} existiert nicht")
    text = p.read_text()
    daten: Any
    if p.suffix.lower() in (".yaml", ".yml"):
        try:
            import yaml
        except ImportError as e:  # pragma: no cover - PyYAML ist Projektabhaengigkeit
            raise KonfigFehler(
                f"{pfad}: YAML angefordert, aber PyYAML fehlt ({e}). "
                f"JSON (.json) geht ohne Zusatzpaket."
            ) from e
        daten = yaml.safe_load(text)
    else:
        daten = json.loads(text)
    if daten is None:
        return {}
    if not isinstance(daten, dict):
        raise KonfigFehler(f"{pfad}: erwartet wird eine Abbildung, nicht {type(daten)}")
    # Sowohl `htccl: {...}` als auch der blanke Teilbaum sind zulaessig.
    if set(daten) == {"htccl"}:
        daten = daten["htccl"] or {}
    elif "htccl" in daten:
        daten = daten["htccl"] or {}
    if not isinstance(daten, dict):
        raise KonfigFehler(f"{pfad}: `htccl` muss eine Abbildung sein")
    return daten


def _aus_abbildung(basis: HtcclKonfig, d: Mapping[str, Any], wo: str) -> HtcclKonfig:
    _pruefe_schluessel(d, _WURZEL_SCHLUESSEL, wo)
    koll = basis.kollektiv
    nic = basis.nic
    if "nic" in d:
        nic = str(d["nic"]).strip().lower().replace("-", "_")
    kd = d.get("kollektiv") or {}
    if not isinstance(kd, Mapping):
        raise KonfigFehler(f"{wo}.kollektiv muss eine Abbildung sein")
    _pruefe_schluessel(kd, _KOLLEKTIV_SCHLUESSEL, f"{wo}.kollektiv")
    aend: dict[str, Any] = {}
    if "planer" in kd:
        aend["planer"] = str(kd["planer"]).strip().lower()
    if "algorithmus" in kd:
        aend["algorithmus"] = str(kd["algorithmus"]).strip().lower()
    if "chunk_kib" in kd:
        v = kd["chunk_kib"]
        aend["chunk_kib"] = None if str(v).strip().lower() == "auto" else int(v)
    if "blatt_schwelle" in kd:
        aend["blatt_schwelle"] = float(kd["blatt_schwelle"])
    if "staffel_verhaeltnis" in kd:
        aend["staffel_verhaeltnis"] = float(kd["staffel_verhaeltnis"])
    if "saettigung_anteil" in kd:
        aend["saettigung_anteil"] = float(kd["saettigung_anteil"])
    if "aufteilung" in kd:
        aend["aufteilung"] = _lies_aufteilung(kd["aufteilung"], f"{wo}.kollektiv")
    if "rollen" in kd:
        r = kd["rollen"] or {}
        if not isinstance(r, Mapping):
            raise KonfigFehler(f"{wo}.kollektiv.rollen muss eine Abbildung sein")
        aend["rollen"] = {
            _norm_bdf(k): str(v).strip().lower() for k, v in r.items()
        }
    if "domaenen" in kd:
        dom = kd["domaenen"] or []
        if not isinstance(dom, Sequence) or isinstance(dom, (str, bytes)):
            raise KonfigFehler(f"{wo}.kollektiv.domaenen muss eine Liste von Listen sein")
        aend["domaenen"] = tuple(
            tuple(_norm_bdf(x) for x in gruppe) for gruppe in dom
        )
    if "mess" in kd:
        md = kd["mess"] or {}
        if not isinstance(md, Mapping):
            raise KonfigFehler(f"{wo}.kollektiv.mess muss eine Abbildung sein")
        _pruefe_schluessel(md, _MESS_SCHLUESSEL, f"{wo}.kollektiv.mess")
        maend: dict[str, Any] = {}
        if "groessen_kib" in md:
            maend["groessen_kib"] = tuple(int(x) for x in md["groessen_kib"])
        for schl in ("wiederholungen", "vorlauf"):
            if schl in md:
                maend[schl] = int(md[schl])
        if "budget_ms" in md:
            maend["budget_ms"] = float(md["budget_ms"])
        for schl in ("fanin", "duplex", "cache_aus"):
            if schl in md:
                maend[schl] = _bool(md[schl], f"{wo}.kollektiv.mess.{schl}")
        if "cache" in md:
            maend["cache"] = None if md["cache"] is None else str(md["cache"])
        aend["mess"] = replace(koll.mess, **maend)
    return HtcclKonfig(kollektiv=replace(koll, **aend), nic=nic)


def _lies_aufteilung(v: Any, wo: str) -> Any:
    if isinstance(v, str):
        s = v.strip().lower()
        if s in ("auto", "gleich", "proportional"):
            return s
        raise KonfigFehler(
            f"{wo}.aufteilung: {v!r} unbekannt "
            f"(auto | gleich | proportional | Abbildung BDF -> Gewichte)"
        )
    if isinstance(v, Mapping):
        aus: dict[str, tuple[float, ...]] = {}
        for k, gew in v.items():
            if not isinstance(gew, Sequence) or isinstance(gew, (str, bytes)):
                raise KonfigFehler(f"{wo}.aufteilung[{k!r}]: Liste von Gewichten erwartet")
            g = tuple(float(x) for x in gew)
            if not g or any(x < 0 for x in g) or sum(g) <= 0:
                raise KonfigFehler(f"{wo}.aufteilung[{k!r}]: Gewichte muessen >= 0 sein, Summe > 0")
            aus[_norm_bdf(k)] = g
        return aus
    raise KonfigFehler(f"{wo}.aufteilung: {v!r} nicht lesbar")


# -- Umgebung --------------------------------------------------------------
#
# Reihenfolge: Vorgabe < Datei < Umgebungsvariable. Die Umgebung gewinnt
# immer, weil sie der Weg ist, einen Lauf ohne Dateiaenderung gegen sich
# selbst zu messen.

_ENV_PRAEFIX = "SGLANG_HTCCL_"


def lade_konfig(env: Optional[Mapping[str, str]] = None) -> HtcclKonfig:
    """Vorgabe < Datei (``SGLANG_HTCCL_KONFIG``) < Umgebung.

    RANGEINHEITLICHKEIT: liest ausschliesslich prozessglobalen Zustand, der
    auf allen Raengen gleich ist (Umgebung, Datei). Nichts hierin darf je
    vom Rang abhaengen -- sonst planen zwei Raenge verschieden und die
    SPMD-Annahme der Kollektive faellt.
    """
    env = os.environ if env is None else env
    k = HtcclKonfig()
    pfad = env.get(_ENV_PRAEFIX + "KONFIG")
    if pfad:
        k = _aus_abbildung(k, _lies_datei(pfad), wo=pfad)

    ueber: dict[str, Any] = {}
    kueber: dict[str, Any] = {}
    mueber: dict[str, Any] = {}

    def _e(name: str) -> Optional[str]:
        return env.get(_ENV_PRAEFIX + name)

    if (v := _e("NIC")) is not None or (v := _e("MATRIX_NIC")) is not None:
        # `MATRIX_NIC` ist der in ENTWURF_HTCCL_TRANSPORT.md genannte Name;
        # `bei-bedarf` dort, `bei_bedarf` hier -- beide werden akzeptiert.
        ueber["nic"] = v.strip().lower().replace("-", "_")
    if (v := _e("PLANER")) is not None:
        kueber["planer"] = v.strip().lower()
    if (v := _e("ALGORITHMUS")) is not None:
        kueber["algorithmus"] = v.strip().lower()
    if (v := _e("CHUNK_KIB")) is not None:
        kueber["chunk_kib"] = None if v.strip().lower() == "auto" else int(v)
    if (v := _e("BLATT_SCHWELLE")) is not None:
        kueber["blatt_schwelle"] = float(v)
    if (v := _e("STAFFEL_VERHAELTNIS")) is not None:
        kueber["staffel_verhaeltnis"] = float(v)
    if (v := _e("SAETTIGUNG_ANTEIL")) is not None:
        kueber["saettigung_anteil"] = float(v)
    if (v := _e("AUFTEILUNG")) is not None:
        s = v.strip()
        kueber["aufteilung"] = _lies_aufteilung(
            json.loads(s) if s.startswith("{") else s, "SGLANG_HTCCL_AUFTEILUNG"
        )
    if (v := _e("ROLLEN")) is not None:
        # "0000:05:00.0=blatt,0000:0a:00.0=domaene" oder JSON
        s = v.strip()
        if s.startswith("{"):
            roh = json.loads(s)
        else:
            roh = {}
            for teil in s.split(","):
                if not teil.strip():
                    continue
                if "=" not in teil:
                    raise KonfigFehler(
                        f"SGLANG_HTCCL_ROLLEN: {teil!r} hat kein '='; erwartet "
                        f"'<bdf>=<rolle>[,<bdf>=<rolle>]' oder JSON"
                    )
                bdf, rolle = teil.split("=", 1)
                roh[bdf] = rolle
        kueber["rollen"] = {_norm_bdf(a): str(b).strip().lower() for a, b in roh.items()}
    if (v := _e("DOMAENEN")) is not None:
        # "05:00.0+0b:00.0;0a:00.0" oder JSON-Liste von Listen
        s = v.strip()
        if s.startswith("["):
            roh = json.loads(s)
        else:
            roh = [g.split("+") for g in s.split(";") if g.strip()]
        kueber["domaenen"] = tuple(tuple(_norm_bdf(x) for x in g) for g in roh)
    if (v := _e("MESS_GROESSEN_KIB")) is not None:
        mueber["groessen_kib"] = tuple(int(x) for x in v.replace(";", ",").split(","))
    if (v := _e("MESS_WIEDERHOLUNGEN")) is not None:
        mueber["wiederholungen"] = int(v)
    if (v := _e("MESS_VORLAUF")) is not None:
        mueber["vorlauf"] = int(v)
    if (v := _e("MESS_BUDGET_MS")) is not None:
        mueber["budget_ms"] = float(v)
    if (v := _e("MESS_FANIN")) is not None:
        mueber["fanin"] = _bool(v, "SGLANG_HTCCL_MESS_FANIN")
    if (v := _e("MESS_DUPLEX")) is not None:
        mueber["duplex"] = _bool(v, "SGLANG_HTCCL_MESS_DUPLEX")
    if (v := _e("MATRIX_CACHE")) is not None:
        mueber["cache"] = v
    if (v := _e("MATRIX_CACHE_AUS")) is not None:
        mueber["cache_aus"] = _bool(v, "SGLANG_HTCCL_MATRIX_CACHE_AUS")

    koll = k.kollektiv
    if mueber:
        kueber["mess"] = replace(koll.mess, **mueber)
    if kueber:
        koll = replace(koll, **kueber)
    k = replace(k, kollektiv=koll, **ueber)
    _validiere(k)
    return k


def _validiere(k: HtcclKonfig) -> None:
    if k.nic not in NIC_MODI:
        raise KonfigFehler(f"nic={k.nic!r} unbekannt; erlaubt {list(NIC_MODI)}")
    c = k.kollektiv
    if c.planer not in PLANER_MODI:
        raise KonfigFehler(f"kollektiv.planer={c.planer!r}; erlaubt {list(PLANER_MODI)}")
    if c.algorithmus != "auto" and c.algorithmus not in ALGORITHMEN:
        raise KonfigFehler(
            f"kollektiv.algorithmus={c.algorithmus!r}; erlaubt "
            f"{['auto', *ALGORITHMEN]}"
        )
    if not 0.0 < c.blatt_schwelle <= 1.0:
        raise KonfigFehler(
            f"kollektiv.blatt_schwelle={c.blatt_schwelle} muss in (0, 1] liegen "
            f"(Anteil am Median; 1.0 heisst 'alles unter dem Median ist Blatt')"
        )
    if c.staffel_verhaeltnis < 1.0:
        raise KonfigFehler("kollektiv.staffel_verhaeltnis muss >= 1 sein")
    if not 0.0 < c.saettigung_anteil <= 1.0:
        raise KonfigFehler("kollektiv.saettigung_anteil muss in (0, 1] liegen")
    if c.chunk_kib is not None and c.chunk_kib <= 0:
        raise KonfigFehler("kollektiv.chunk_kib muss > 0 sein oder 'auto'")
    for bdf, rolle in c.rollen.items():
        if rolle not in ROLLEN:
            raise KonfigFehler(
                f"kollektiv.rollen[{bdf!r}]={rolle!r}; erlaubt {list(ROLLEN)}"
            )
    m = c.mess
    if not m.groessen_kib or any(g <= 0 for g in m.groessen_kib):
        raise KonfigFehler("kollektiv.mess.groessen_kib: positive Werte erwartet")
    if m.wiederholungen <= 0 or m.vorlauf < 0:
        raise KonfigFehler("kollektiv.mess.wiederholungen > 0, vorlauf >= 0")
    if m.budget_ms <= 0:
        raise KonfigFehler("kollektiv.mess.budget_ms muss > 0 sein")
    if c.planer == "aus" and c.algorithmus == "auto":
        raise KonfigFehler(
            "kollektiv.planer=aus verlangt einen festen kollektiv.algorithmus "
            "-- 'aus' + 'auto' heisst 'nicht messen und trotzdem waehlen', und "
            "das gibt es nicht. Kein stiller Rueckfall."
        )
    if c.planer == "fest" and c.mess.cache_aus:
        raise KonfigFehler(
            "kollektiv.planer=fest braucht den Zwischenspeicher; "
            "mess.cache_aus=1 nimmt ihm die einzige Quelle. Gemeint ist "
            "vermutlich planer=auto."
        )


# ===========================================================================
# Fuehler: was gemessen wird
# ===========================================================================


class Fuehler(Protocol):
    """Messfuehler des Planers.

    Zwei Auspraegungen, absichtlich getrennt:

    ``eigenlast``   misst die eigene PCIe-Anbindung (GPU <-> gepinnter
                    Host-Speicher) und braucht keinen Peer-Pfad. Immer
                    verfuegbar, liefert **Knotenkapazitaeten** je Richtung.
    ``paar``        misst die gerichtete Kante Rang->Rang ueber den echten
                    Punkt-zu-Punkt-Pfad. Nur verfuegbar, wenn ein Transport
                    ihn liefert (``htccl_bar1``). Liefert **Kantenkapazitaeten**.

    Ein Fuehler, der ``paar`` nicht kann, gibt ``None`` zurueck; der Planer
    faellt dann auf die Eigenlast-Schaetzung zurueck und schreibt das in die
    Erklaerung. Er tut NICHT so, als haette er die Kante gemessen.
    """

    def name(self) -> str: ...

    def eigenlast(self, nbytes: int, richtung: str) -> float:
        """GB/s. ``richtung`` ist ``"aus"`` (D2H) oder ``"ein"`` (H2D)."""
        ...

    def eigenlast_duplex(self, nbytes: int) -> Optional[float]:
        """Summe beider Richtungen gleichzeitig, GB/s. ``None`` wenn nicht messbar."""
        ...

    def paar(self, ziel: int, nbytes: int) -> Optional[float]:
        """GB/s von *diesem* Rang zum Rang ``ziel``. ``None`` = kein Paarpfad."""
        ...

    def paar_empfang(self, quelle: int, nbytes: int) -> None:
        """Gegenstueck zu ``paar`` auf der Zielseite (nur Synchronisation)."""
        ...


class EigenlastFuehler:
    """Vorgabe-Fuehler: misst die eigene Anbindung, ohne jeden Peer-Pfad.

    Das ist eine echte Messung dieser Karte an diesem Steckplatz -- nicht
    die ``lspci``-Breite. Ihre Grenze ist eine andere: sie misst GPU <->
    Host, nicht GPU <-> GPU. Eine Kante ueber einen Switch-Uplink oder
    ueber zwei Root-Complexe kann deutlich langsamer sein als das Minimum
    der beiden Knotenraten. Der Planer markiert daraus abgeleitete Kanten
    deshalb als ``quelle="eigenlast"``, und die Erklaerung sagt es.
    """

    def __init__(self, device, max_bytes: int = 4 << 20,
                 wiederholungen: Optional[int] = None):
        import torch

        self.device = device
        self._torch = torch
        self._max_bytes = max_bytes
        # None = die groessenabhaengige Vorgabe (kleine Nachrichten brauchen
        # mehr Runden, sonst misst man die Uhr statt die Leitung).
        self._reps = wiederholungen
        self._dev = torch.empty(max_bytes, dtype=torch.uint8, device=device)
        self._host = torch.empty(max_bytes, dtype=torch.uint8, pin_memory=True)
        self._host2 = torch.empty(max_bytes, dtype=torch.uint8, pin_memory=True)
        self._s1 = torch.cuda.Stream(device=device)
        self._s2 = torch.cuda.Stream(device=device)

    def name(self) -> str:
        return "eigenlast"

    def _lauf(self, nbytes: int, richtung: str, n: int) -> None:
        d = self._dev[:nbytes]
        h = self._host[:nbytes]
        for _ in range(n):
            if richtung == "aus":
                h.copy_(d, non_blocking=True)   # D2H == Ausgang der Karte
            else:
                d.copy_(h, non_blocking=True)   # H2D == Eingang der Karte

    def eigenlast(self, nbytes: int, richtung: str) -> float:
        torch = self._torch
        nbytes = min(nbytes, self._max_bytes)
        self._lauf(nbytes, richtung, 8)
        torch.cuda.synchronize(self.device)
        n = self._reps or _wiederholungen_fuer(nbytes)
        t0 = time.perf_counter()
        self._lauf(nbytes, richtung, n)
        torch.cuda.synchronize(self.device)
        dt = time.perf_counter() - t0
        return (n * nbytes) / dt / 1e9 if dt > 0 else 0.0

    def eigenlast_duplex(self, nbytes: int) -> Optional[float]:
        """R4 gemessen, nicht angenommen: beide Richtungen gleichzeitig.

        Zwei Stroeme auf zwei Streams, damit die Kopiermaschinen wirklich
        parallel laufen. Rueckgabe ist die *Summe* beider Richtungen; der
        Planer bildet daraus den Duplexfaktor gegen die Summe der
        Einzelraten. Auf diesem Rig war die Summe bei 1 MiB 1,32x einer
        Richtung -- also gerade nicht 2x.
        """
        torch = self._torch
        nbytes = min(nbytes, self._max_bytes)
        d = self._dev[:nbytes]
        h1 = self._host[:nbytes]
        h2 = self._host2[:nbytes]
        n = self._reps or _wiederholungen_fuer(nbytes)

        def runde(k: int) -> None:
            for _ in range(k):
                with torch.cuda.stream(self._s1):
                    h1.copy_(d, non_blocking=True)
                with torch.cuda.stream(self._s2):
                    d.copy_(h2, non_blocking=True)

        runde(8)
        torch.cuda.synchronize(self.device)
        t0 = time.perf_counter()
        runde(n)
        torch.cuda.synchronize(self.device)
        dt = time.perf_counter() - t0
        return (2 * n * nbytes) / dt / 1e9 if dt > 0 else None

    def paar(self, ziel: int, nbytes: int) -> Optional[float]:
        return None   # kein Punkt-zu-Punkt-Pfad -- ehrlich statt geschaetzt

    def paar_empfang(self, quelle: int, nbytes: int) -> None:
        return None


def _wiederholungen_fuer(nbytes: int) -> int:
    """Mehr Wiederholungen bei kleinen Nachrichten.

    Die 20-KiB-Zeilen der Referenzmessung liegen trotz 64 Wiederholungen
    nahe an der Aufloesung der Uhr; unter ~64 Runden sind die Prozentzahlen
    dort Rauschen. Grosse Nachrichten brauchen sie nicht und wuerden das
    Zeitbudget auffressen.
    """
    if nbytes <= 64 * 1024:
        return 64
    if nbytes <= 512 * 1024:
        return 32
    return 16


# ===========================================================================
# Messung: das Ergebnis der Startvermessung
# ===========================================================================


@dataclass
class Messung:
    """Rohbefund. Enthaelt nur Gemessenes, keine Politik.

    Alle Raten in GB/s (10^9 Byte/s, wie im uebrigen Repo).
    """

    welt: int
    groessen: tuple[int, ...]                      # Bytes
    bdfs: tuple[str, ...]                          # je Rang
    namen: tuple[str, ...]                         # je Rang (Kartenname)
    fuehler: str                                   # welcher Fuehler mass
    # Knotenraten je Rang und Groesse.
    aus: dict[int, list[float]] = field(default_factory=dict)     # rang -> je Groesse
    ein: dict[int, list[float]] = field(default_factory=dict)
    duplex_summe: dict[int, list[float]] = field(default_factory=dict)
    # Kantenraten, wenn ein Paar-Fuehler da war: (von, nach) -> je Groesse
    kante: dict[tuple[int, int], list[float]] = field(default_factory=dict)
    # R3: Fan-in-Deckel je Ziel und Groesse, plus Anteile je Quelle.
    fanin_deckel: dict[int, list[float]] = field(default_factory=dict)
    fanin_anteile: dict[tuple[int, int], list[float]] = field(default_factory=dict)
    # Aus der Groessenleiter angepasste Gerade t(N) = latenz + N/rate,
    # je Rang: Startkosten eines Schrittes in Sekunden.
    latenz_s: dict[int, float] = field(default_factory=dict)
    # Geteilte Engpaesse: um wieviel eine Kante langsamer wird, wenn ALLE
    # Paare gleichzeitig reden (Netz) statt nur eines (Ring). 1,0 heisst
    # "keine Behinderung" UND ist zugleich der Wert fuer "nicht gemessen"
    # -- welches von beiden gilt, sagt `netz_faktor_gemessen`.
    #
    # Das ist der Term, an dem die Ring-gewinnt-Hypothese haengt: ein Netz
    # zwingt JEDES Paar zu reden, also auch ueber Switch-Uplinks und
    # NUMA-Spruenge, waehrend ein nach Topologie geordneter Ring geteilte
    # Engpaesse nur einmal je Runde quert. Per-Kanten-Kapazitaeten allein
    # koennen das nicht abbilden -- sie sind je Kante einzeln gemessen.
    netz_faktor: list[float] = field(default_factory=list)
    netz_faktor_gemessen: bool = False
    dauer_s: float = 0.0
    hinweise: list[str] = field(default_factory=list)

    # -- abgeleitete Sichten (kein Zustand) ---------------------------------

    def kap(self, von: int, nach: int, gi: int) -> float:
        """Gerichtete Kapazitaet Rang->Rang bei Groesse ``groessen[gi]``.

        Gemessene Kante gewinnt. Sonst die Eigenlast-Schaetzung
        ``min(Ausgang der Quelle, Eingang des Ziels)`` -- ausdruecklich eine
        **Obergrenze**, weil sie geteilte Engpaesse (Switch-Uplink, zweiter
        Root-Complex) nicht sehen kann.
        """
        e = self.kante.get((von, nach))
        if e is not None:
            return e[gi]
        return min(self.aus[von][gi], self.ein[nach][gi])

    def kante_gemessen(self, von: int, nach: int) -> bool:
        return (von, nach) in self.kante

    def deckel(self, nach: int, gi: int) -> float:
        d = self.fanin_deckel.get(nach)
        if d is not None:
            return d[gi]
        return self.ein[nach][gi]

    def deckel_gemessen(self, nach: int) -> bool:
        return nach in self.fanin_deckel

    def netz_strafe(self, gi: int) -> float:
        """Faktor auf den Netz-Uebertragungsterm. >= 1."""
        if not self.netz_faktor or gi >= len(self.netz_faktor):
            return 1.0
        return max(1.0, self.netz_faktor[gi])

    def duplex_faktor(self, rang: int, gi: int) -> float:
        """Summe beider Richtungen / groessere Einzelrichtung. 1,0 = kein Gewinn."""
        d = self.duplex_summe.get(rang)
        if d is None:
            return 1.0
        einzeln = max(self.aus[rang][gi], self.ein[rang][gi])
        return d[gi] / einzeln if einzeln > 0 else 1.0

    def schritt_s(self) -> float:
        """Startkosten eines Kollektivschrittes, Sekunden (Median ueber Raenge)."""
        if not self.latenz_s:
            return 0.0
        return statistics.median(self.latenz_s.values())

    def als_dict(self) -> dict:
        d = asdict(self)
        # Tupel-Schluessel sind nicht JSON-faehig.
        d["kante"] = {f"{a}->{b}": v for (a, b), v in self.kante.items()}
        d["fanin_anteile"] = {f"{a}->{b}": v for (a, b), v in self.fanin_anteile.items()}
        d["aus"] = {str(k): v for k, v in self.aus.items()}
        d["ein"] = {str(k): v for k, v in self.ein.items()}
        d["duplex_summe"] = {str(k): v for k, v in self.duplex_summe.items()}
        d["fanin_deckel"] = {str(k): v for k, v in self.fanin_deckel.items()}
        d["latenz_s"] = {str(k): v for k, v in self.latenz_s.items()}
        return d

    @staticmethod
    def aus_dict(d: Mapping[str, Any]) -> "Messung":
        def paar(s: str) -> tuple[int, int]:
            a, b = s.split("->")
            return int(a), int(b)

        m = Messung(
            welt=int(d["welt"]),
            groessen=tuple(int(x) for x in d["groessen"]),
            bdfs=tuple(d["bdfs"]),
            namen=tuple(d["namen"]),
            fuehler=str(d["fuehler"]),
            dauer_s=float(d.get("dauer_s", 0.0)),
            hinweise=list(d.get("hinweise", [])),
        )
        m.aus = {int(k): list(v) for k, v in d["aus"].items()}
        m.ein = {int(k): list(v) for k, v in d["ein"].items()}
        m.duplex_summe = {int(k): list(v) for k, v in d.get("duplex_summe", {}).items()}
        m.kante = {paar(k): list(v) for k, v in d.get("kante", {}).items()}
        m.fanin_deckel = {int(k): list(v) for k, v in d.get("fanin_deckel", {}).items()}
        m.fanin_anteile = {
            paar(k): list(v) for k, v in d.get("fanin_anteile", {}).items()
        }
        m.latenz_s = {int(k): float(v) for k, v in d.get("latenz_s", {}).items()}
        m.netz_faktor = [float(x) for x in d.get("netz_faktor", [])]
        m.netz_faktor_gemessen = bool(d.get("netz_faktor_gemessen", False))
        return m


def _quant(x: float) -> float:
    """Quantisieren, bevor entschieden wird.

    Der Plan muss auf allen Raengen bitgleich herauskommen. Die Rohwerte
    kommen zwar als identische ``float`` ueber ``all_gather_object`` an,
    aber eine Entscheidungsschwelle, die auf der 12. Nachkommastelle
    kippt, ist trotzdem ein Zufallsgenerator. Drei Nachkommastellen in
    GB/s sind rund 1 MB/s -- weit unter jeder Messstreuung.
    """
    if not math.isfinite(x):
        return 0.0
    return round(float(x), 3)


# ===========================================================================
# Der Plan
# ===========================================================================


@dataclass(frozen=True)
class Stufe:
    """Eine Stufe der Groessenleiter: bis ``max_bytes`` gilt ``algorithmus``."""

    von_bytes: int                       # kleinste Groesse, fuer die gemessen wurde
    max_bytes: int                       # inklusive; -1 == "und alles darueber"
    algorithmus: str
    vorhersage_s: Mapping[str, float]    # algorithmus -> vorhergesagte Zeit
    grund: str


@dataclass(frozen=True)
class Plan:
    welt: int
    bdfs: tuple[str, ...]
    rollen: tuple[str, ...]                     # je Rang
    domaene: tuple[int, ...]                    # Raenge der Reduktionsdomaene
    blaetter: tuple[int, ...]
    eltern: Mapping[int, tuple[int, ...]]       # Blatt -> Domaenenknoten
    aufteilung: Mapping[int, tuple[int, ...]]   # Blatt -> Promille je Elternteil
    ringfolge: tuple[int, ...]
    leiter: tuple[Stufe, ...]
    chunk_bytes: int
    staffeln: Mapping[int, tuple[tuple[int, ...], ...]]   # Ziel -> Wellen von Quellen
    konfig_zusammenfassung: Mapping[str, Any]
    messung: Optional[Messung] = None
    quelle: str = "gemessen"                    # gemessen | zwischenspeicher | fest

    # -- Abfragen, die ein Transport braucht --------------------------------

    def algorithmus_fuer(self, nbytes: int) -> str:
        for stufe in self.leiter:
            if stufe.max_bytes < 0 or nbytes <= stufe.max_bytes:
                return stufe.algorithmus
        return self.leiter[-1].algorithmus

    def ist_blatt(self, rang: int) -> bool:
        return self.rollen[rang] == "blatt"

    def pruefsumme(self) -> str:
        """Fingerabdruck der *Entscheidungen*, nicht der Rohmessung.

        Bewusst ohne ``messung``: die Rohwerte weichen zwischen Raengen um
        Messrauschen ab, sobald jeder Rang eigene Zahlen beisteuert. Was
        uebereinstimmen MUSS, ist der Plan.
        """
        kern = {
            "welt": self.welt,
            "rollen": list(self.rollen),
            "domaene": list(self.domaene),
            "blaetter": list(self.blaetter),
            "eltern": {str(k): list(v) for k, v in sorted(self.eltern.items())},
            "aufteilung": {str(k): list(v) for k, v in sorted(self.aufteilung.items())},
            "ringfolge": list(self.ringfolge),
            "leiter": [(s.von_bytes, s.max_bytes, s.algorithmus) for s in self.leiter],
            "chunk_bytes": self.chunk_bytes,
            "staffeln": {
                str(k): [list(w) for w in v] for k, v in sorted(self.staffeln.items())
            },
        }
        roh = json.dumps(kern, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(roh.encode()).hexdigest()[:16]

    # -- Erklaerung ---------------------------------------------------------

    def erklaerung(self) -> str:
        """Menschenlesbar: was gemessen, was gefolgert, was gewaehlt wurde.

        Ohne diesen Block debuggt niemand auf fremder Hardware -- deshalb
        ist er Pflichtausgabe und nicht an ein Debug-Flag gebunden.
        """
        z: list[str] = []
        a = z.append
        a("=" * 78)
        a(f"HTCCL-Pfadmatrix: Plan {self.pruefsumme()} ({self.quelle}), "
          f"{self.welt} Raenge")
        a("=" * 78)
        m = self.messung
        if m is not None:
            a(f"Fuehler: {m.fuehler}   Messdauer: {m.dauer_s * 1000:.0f} ms")
            a("")
            a("-- gemessene Knotenraten (GB/s, Ausgang/Eingang je Groesse) --")
            a("   Die sysfs-Spalte steht NUR zum Vergleich da und geht in "
              "keine Entscheidung ein:")
            a("   sagt sie x8 und die Messung 6 GB/s, ist entweder die Karte "
              "heruntergetaktet")
            a("   oder der Messaufbau kaputt -- lspci ist Startschaetzung, "
              "nicht Wahrheit.")
            kopf = "  Rang  Karte                  sysfs   " + "".join(
                f"{_kib(g):>18}" for g in m.groessen
            )
            a(kopf)
            for r in range(self.welt):
                lb = linkbreite(self.bdfs[r])
                lbs = f"x{lb[0]}" if lb else "?"
                zeile = (f"  {r:>4}  {self.bdfs[r]:<14}{m.namen[r][:8]:<8}"
                         f"{lbs:<8}")
                for gi in range(len(m.groessen)):
                    zeile += f"{m.aus[r][gi]:>8.2f}/{m.ein[r][gi]:<9.2f}"
                a(zeile)
            if m.duplex_summe:
                a("")
                a("-- Vollduplex (Summe beider Richtungen / staerkere Einzelrichtung) --")
                a("   R4: gemessen, nicht angenommen. 1,00 = Gegenrichtung ist gratis "
                  "nicht vorhanden;")
                a("   2,00 waere die naive Erwartung. Referenz dieses Rigs: "
                  f"{_BELEG_DUPLEX_SUMME_20KIB:.2f}x bei 20 KiB, "
                  f"{_BELEG_DUPLEX_SUMME_1MIB:.2f}x bei 1 MiB.")
                for r in range(self.welt):
                    if r not in m.duplex_summe:
                        continue
                    werte = "  ".join(
                        f"{_kib(g)}: {m.duplex_faktor(r, gi):.2f}x"
                        for gi, g in enumerate(m.groessen)
                    )
                    a(f"  Rang {r}: {werte}")
            a("")
            a("-- gerichtete Kantenkapazitaeten (GB/s) --")
            a("   'M' = am Paar gemessen, 'S' = Schaetzung min(Ausgang, Eingang) "
              "aus der Eigenlast")
            for gi, g in enumerate(m.groessen):
                a(f"  bei {_kib(g)}:")
                a("        " + "".join(f"{'->' + str(j):>12}" for j in range(self.welt)))
                for i in range(self.welt):
                    zeile = f"    {i:>2}  "
                    for j in range(self.welt):
                        if i == j:
                            zeile += f"{'-':>12}"
                        else:
                            mark = "M" if m.kante_gemessen(i, j) else "S"
                            zeile += f"{m.kap(i, j, gi):>10.2f}{mark}"
                    a(zeile)
            a("")
            a("-- Fan-in-Deckel je Ziel (GB/s) --")
            a("   R3: in keine Karte geht mehr hinein als der Deckel, egal von "
              "wie vielen Quellen.")
            a(f"   Referenz dieses Rigs: {_BELEG_FANIN_GBPS:.2f} GB/s bei 1 MiB "
              "mit zwei Quellen.")
            for j in range(self.welt):
                mark = "gemessen" if m.deckel_gemessen(j) else "aus Eingangsrate geschaetzt"
                werte = "  ".join(
                    f"{_kib(g)}: {m.deckel(j, gi):.2f}" for gi, g in enumerate(m.groessen)
                )
                a(f"  Ziel {j} ({self.bdfs[j]}): {werte}   [{mark}]")
            a(f"  Schrittkosten (Startkosten je Kollektivschritt): "
              f"{m.schritt_s() * 1e6:.2f} us")
            a("")
            a("-- geteilte Engpaesse: Strafe des Netzes gegen den Ring --")
            if m.netz_faktor_gemessen:
                a("   Gemessen: alle Paare gleichzeitig gegen ein Paar allein. "
                  ">1 heisst, das Netz")
                a("   verliert an Uplinks/NUMA-Spruengen, die eine "
                  "Kantenkapazitaet nicht sieht.")
            else:
                a("   NICHT GEMESSEN -- auf 1,0 gesetzt. Bei 1,0 bewegen Netz "
                  "und Ring dieselben")
                a("   Bytes durch denselben Deckel, und das Netz gewinnt "
                  "dann immer ueber die")
                a("   Schrittzahl. Der Ring kann hier nur ueber die "
                  "Fenstergrenze gewaehlt werden.")
            werte = "  ".join(
                f"{_kib(g)}: {m.netz_strafe(gi):.2f}x"
                for gi, g in enumerate(m.groessen)
            )
            a(f"  {werte}")
            for h in m.hinweise:
                a(f"  ! {h}")
        else:
            a("Keine Messung (Planer aus oder fest konfiguriert).")

        a("")
        a("-- Rollen (R2: Kapazitaet < blatt_schwelle x Median -> Blatt) --")
        for r in range(self.welt):
            zusatz = ""
            if r in self.eltern:
                el = ", ".join(str(x) for x in self.eltern[r])
                anteile = "/".join(f"{p / 10:.0f}%" for p in self.aufteilung.get(r, ()))
                zusatz = f"  -> Eltern [{el}]  Aufteilung {anteile}"
            a(f"  Rang {r} ({self.bdfs[r]}): {self.rollen[r]}{zusatz}")
        a(f"  Reduktionsdomaene: {list(self.domaene)}   Blaetter: {list(self.blaetter)}")
        a(f"  Ringreihenfolge (nach gemessener Kapazitaet geordnet): "
          f"{list(self.ringfolge)}")

        a("")
        a("-- Algorithmus je Groessenklasse --")
        for stufe in self.leiter:
            grenze = (f"ab {_kib(stufe.von_bytes)}" if stufe.max_bytes < 0
                      else f"{_kib(stufe.von_bytes)}..{_kib(stufe.max_bytes)}")
            vor = "  ".join(
                f"{k}={v * 1e6:.1f}us" for k, v in sorted(stufe.vorhersage_s.items())
            )
            a(f"  {grenze:>16}: {stufe.algorithmus:<14} [{vor}]")
            a(f"                    {stufe.grund}")
        a(f"  Chunk: {self.chunk_bytes // 1024} KiB")

        if self.staffeln:
            a("")
            a("-- Fan-in-Staffelplan (R3) --")
            a("   Eine schnelle und eine langsame Quelle duerfen nicht "
              "gleichzeitig ins selbe Ziel")
            a("   schreiben: der Deckel wird gleichmaessig je Quelle "
              "aufgeteilt, die schnelle wird")
            a("   auf den Anteil der langsamen heruntergezogen. Mehr als eine "
              "Welle heisst gestaffelt.")
            for ziel, wellen in sorted(self.staffeln.items()):
                w = " | ".join("+".join(str(q) for q in welle) for welle in wellen)
                a(f"  Ziel {ziel} ({self.bdfs[ziel]}): {w}")

        a("")
        a("-- wirksame Konfiguration --")
        for k, v in sorted(self.konfig_zusammenfassung.items()):
            a(f"  {k}: {v}")
        a("=" * 78)
        return "\n".join(z)


def _kib(n: int) -> str:
    if n >= 1024 * 1024:
        return f"{n // (1024 * 1024)} MiB"
    return f"{n // 1024} KiB"


# ===========================================================================
# Das Kostenmodell -- hier stecken R1, R3 und R4
# ===========================================================================


def _zeit_netz(m: Messung, gi: int, n_bytes: int, raenge: Sequence[int],
               wellen: Mapping[int, tuple[tuple[int, ...], ...]],
               schritt_s: float) -> float:
    """Direktes Netz: Reduce-Scatter + Allgather, je genau ein Schritt.

    ``raenge`` sind ECHTE Raenge, keine Indizes -- die Funktion wird auch
    fuer eine Teilmenge (die Reduktionsdomaene) aufgerufen, und ein
    Index-in-die-Teilmenge waere dort ein anderer Rang.

    Byte-Last je Rang wie beim Ring: 2N(R-1)/R. Schritte aber nur **2**
    statt 2(R-1) -- das ist der ganze Grund, warum das Netz bei kleinen
    Nachrichten gewinnt.

    R3 steckt in der effektiven Rate: ``R-1`` Quellen beschreiben dasselbe
    Ziel, und der Deckel wird **gleichmaessig je Quelle** aufgeteilt. Die
    effektive Rate einer Quelle ist also
    ``min(Kantenkapazitaet, Deckel / Zahl der gleichzeitigen Quellen)``.
    Genau dieser Term bestraft das Netz bei Saettigung: die schnelle Kante
    wird auf den Deckelanteil zusammengedrueckt und verschenkt.

    Der Staffelplan multipliziert die Schritte: wer in zwei Wellen
    schreiben muss, zahlt die Schritte zweimal.
    """
    r = len(raenge)
    if r < 2:
        return 0.0
    menge = set(raenge)
    anteil = n_bytes / r             # jeder schickt N/R an jeden Nachbarn
    schlimmste = 0.0
    schritte = 2
    for ziel in raenge:
        w = tuple(
            tuple(q for q in welle if q in menge)
            for welle in wellen.get(ziel, (tuple(x for x in raenge if x != ziel),))
        )
        w = tuple(welle for welle in w if welle)
        deckel = m.deckel(ziel, gi)
        # Wellen laufen NACHEINANDER -- das ist ihr ganzer Zweck. Also
        # summieren, nicht das Maximum nehmen: gestaffelter Fan-in kostet
        # Zeit, und genau dieser Preis muss im Vergleich gegen den Ring
        # sichtbar sein.
        summe = 0.0
        for welle in w:
            gleichzeitig = len(welle)
            pro_quelle = deckel / gleichzeitig if gleichzeitig else deckel
            dauer = 0.0
            for quelle in welle:
                rate = min(m.kap(quelle, ziel, gi), pro_quelle)
                dauer = max(dauer, anteil / (rate * 1e9) if rate > 0 else float("inf"))
            summe += dauer
        schlimmste = max(schlimmste, summe)
        schritte = max(schritte, 2 * len(w))
    return schritte * schritt_s + 2 * schlimmste * m.netz_strafe(gi)


def _zeit_ring(m: Messung, gi: int, n_bytes: int, folge: Sequence[int],
               schritt_s: float) -> float:
    """Ring: 2(R-1) Schritte, aber jeder Rang empfaengt von genau EINEM Nachbarn.

    Kein Fan-in, keine unfaire Aufteilung -- der Grund, warum der Ring an
    der Saettigung gewinnt. Bezahlt wird das mit der Schrittzahl, die bei
    R=8 vierzehn statt zwei betraegt.
    """
    welt = len(folge)
    if welt < 2:
        return 0.0
    anteil = n_bytes / welt
    langsamste = 0.0
    for k in range(welt):
        von, nach = folge[k], folge[(k + 1) % welt]
        rate = m.kap(von, nach, gi)
        langsamste = max(langsamste, anteil / (rate * 1e9) if rate > 0 else float("inf"))
    return 2 * (welt - 1) * (schritt_s + langsamste)


def _zeit_stern(m: Messung, gi: int, n_bytes: int, nabe: int, welt: int,
                schritt_s: float) -> float:
    """Stern: alles auf die Nabe, dort reduzieren, zurueckverteilen.

    Zwei streng getrennte Phasen mit (R-1)N je Richtung auf der Nabe. Steht
    im Modell drin, weil er der heutige Zustand ist und weil
    ``algorithmus: stern`` erzwingbar bleiben muss -- nicht weil er je
    gewinnt.
    """
    if welt < 2:
        return 0.0
    ein = m.deckel(nabe, gi)
    aus_raten = [m.kap(nabe, j, gi) for j in range(welt) if j != nabe]
    aus = min(aus_raten) if aus_raten else 0.0
    if ein <= 0 or aus <= 0:
        return float("inf")
    hin = (welt - 1) * n_bytes / (ein * 1e9)
    rueck = (welt - 1) * n_bytes / (aus * 1e9)
    return 2 * schritt_s + hin + rueck


def _zeit_hierarchisch(m: Messung, gi: int, n_bytes: int,
                       domaene: Sequence[int], blaetter: Sequence[int],
                       eltern: Mapping[int, tuple[int, ...]],
                       aufteilung: Mapping[int, tuple[int, ...]],
                       innen_s: float, schritt_s: float, welt: int) -> float:
    """Blatt + Reduktionsdomaene, gechunkt und ueberlappt.

    Das Blatt schickt N hinaus und holt N herein -- algorithmusunabhaengig
    der Boden. Mit Chunking ueberlappen die beiden Richtungen, aber
    **nicht gratis**: die Gutschrift ist der gemessene Duplexfaktor (R4),
    nicht 2.
    """
    if not blaetter:
        return innen_s
    schlimmstes_blatt = 0.0
    for b in blaetter:
        el = eltern.get(b, tuple(domaene))
        gew = aufteilung.get(b, tuple(1000 // max(len(el), 1) for _ in el))
        gesamt = sum(gew) or 1
        hoch = 0.0
        runter = 0.0
        for e, g in zip(el, gew):
            teil = n_bytes * (g / gesamt)
            r_hoch = m.kap(b, e, gi)
            r_runter = m.kap(e, b, gi)
            hoch = max(hoch, teil / (r_hoch * 1e9) if r_hoch > 0 else float("inf"))
            runter = max(runter, teil / (r_runter * 1e9) if r_runter > 0 else float("inf"))
        # Ueberlappung: der Duplexfaktor sagt, wieviel der Gegenrichtung
        # tatsaechlich gratis ist. f=1 -> keine Ueberlappung (hoch+runter),
        # f=2 -> volle Ueberlappung (max). Gemessen liegt er dazwischen.
        f = max(1.0, min(2.0, m.duplex_faktor(b, gi)))
        seriell = hoch + runter
        parallel = max(hoch, runter)
        schlimmstes_blatt = max(schlimmstes_blatt, seriell - (f - 1.0) * (seriell - parallel))
    return 2 * schritt_s + schlimmstes_blatt + innen_s


# ===========================================================================
# Ableitungen: Rollen, Domaenen, Ring, Staffeln, Chunk
# ===========================================================================


def _knotenkapazitaet(m: Messung, r: int) -> float:
    """Eine Zahl je Rang fuer den Rollenvergleich (R2).

    Bewusst das **Minimum** von Aus- und Eingang ueber alle Groessen
    gemittelt: eine Karte, die schnell sendet, aber langsam empfaengt, ist
    fuer Transitverkehr genauso ungeeignet wie umgekehrt -- Transit heisst
    beides. Fuer die Kantenwahl selbst bleiben die Richtungen getrennt;
    zusammengefasst wird nur hier, wo eine Rangfolge gebraucht wird.
    """
    gi = len(m.groessen) - 1        # groesste gemessene Groesse: dort trennt es sich
    return min(m.aus[r][gi], m.ein[r][gi])


def _rollen(m: Messung, k: KollektivKonfig) -> tuple[list[str], list[int], list[int]]:
    welt = m.welt
    kap = [_knotenkapazitaet(m, r) for r in range(welt)]
    med = statistics.median(kap) if kap else 0.0
    schwelle = k.blatt_schwelle * med
    rollen = ["domaene"] * welt
    for r in range(welt):
        if welt > 2 and kap[r] < schwelle:
            rollen[r] = "blatt"
    # Von Hand gesetzte Rollen ueberstimmen die Messung -- ausdruecklich.
    for r in range(welt):
        fest = k.rollen.get(m.bdfs[r])
        if fest is not None:
            rollen[r] = "domaene" if fest == "nabe" else fest
    # Es muss mindestens zwei Domaenenknoten geben, sonst gibt es nichts,
    # worin reduziert wird. Notfalls die staerksten zuruecknehmen.
    dom = [r for r in range(welt) if rollen[r] != "blatt"]
    if len(dom) < min(2, welt):
        stark = sorted(range(welt), key=lambda r: (-kap[r], r))[: min(2, welt)]
        for r in stark:
            rollen[r] = "domaene"
    dom = [r for r in range(welt) if rollen[r] != "blatt"]
    blaetter = [r for r in range(welt) if rollen[r] == "blatt"]
    return rollen, dom, blaetter


def _domaenen_aus_konfig(m: Messung, k: KollektivKonfig) -> Optional[list[list[int]]]:
    if not k.domaenen:
        return None
    index = {bdf: r for r, bdf in enumerate(m.bdfs)}
    aus: list[list[int]] = []
    for gruppe in k.domaenen:
        raenge = []
        for bdf in gruppe:
            if bdf not in index:
                raise KonfigFehler(
                    f"kollektiv.domaenen nennt {bdf!r}, aber dieser Verbund hat "
                    f"nur {sorted(index)}. Kein stiller Rueckfall -- entweder "
                    f"die Adresse ist falsch oder die Karte fehlt."
                )
            raenge.append(index[bdf])
        aus.append(sorted(raenge))
    gesehen = [r for g in aus for r in g]
    if len(gesehen) != len(set(gesehen)):
        raise KonfigFehler("kollektiv.domaenen: ein Rang steht in zwei Domaenen")
    return aus


def _eltern_und_aufteilung(
    m: Messung, domaene: Sequence[int], blaetter: Sequence[int], k: KollektivKonfig
) -> tuple[dict[int, tuple[int, ...]], dict[int, tuple[int, ...]]]:
    """Wer haengt woran, und mit welchem Verhaeltnis.

    Das Aufteilungsverhaeltnis ist ein eigener Stellhebel: konzentriert das
    Blatt seinen Verkehr auf einen Elternteil, bleibt den schnellen Karten
    untereinander weniger uebrig (in Lane-Einheiten gerechnet: gleichmaessig
    2+2 laesst B<->C sechs Lanes, konzentriert 4+0 nur vier). Fuer die
    Gesamtdauer auf einem Drei-Karten-Rig aendert es nichts, bei mehr
    Raengen sehr wohl -- deshalb steuerbar.

    ``auto`` = proportional zur gemessenen Kante, ABER durch R3 gedeckelt:
    wo die Kapazitaeten der Eltern zu weit auseinanderliegen, bringt
    Proportionalitaet nichts, weil der Fan-in-Deckel ohnehin gleichmaessig
    teilt -- dort wird gleichmaessig aufgeteilt.
    """
    gi = len(m.groessen) - 1
    eltern: dict[int, tuple[int, ...]] = {}
    aufteilung: dict[int, tuple[int, ...]] = {}
    modus = k.aufteilung
    for b in blaetter:
        kand = sorted(domaene, key=lambda d: (-m.kap(b, d, gi), d))
        if not kand:
            continue
        eltern[b] = tuple(kand)
        if isinstance(modus, Mapping):
            gew = modus.get(m.bdfs[b])
            if gew is not None:
                if len(gew) != len(kand):
                    raise KonfigFehler(
                        f"kollektiv.aufteilung[{m.bdfs[b]!r}] nennt {len(gew)} "
                        f"Gewichte, das Blatt hat aber {len(kand)} Eltern "
                        f"{[m.bdfs[d] for d in kand]}"
                    )
                aufteilung[b] = _promille(gew)
                continue
            modus_b = "auto"
        else:
            modus_b = modus
        raten = [m.kap(b, d, gi) for d in kand]
        if modus_b == "gleich" or not any(raten):
            aufteilung[b] = _promille([1.0] * len(kand))
        elif modus_b == "proportional":
            aufteilung[b] = _promille(raten)
        else:  # auto
            spanne = (max(raten) / min(raten)) if min(raten) > 0 else float("inf")
            aufteilung[b] = _promille(
                [1.0] * len(kand) if spanne > k.staffel_verhaeltnis else raten
            )
    return eltern, aufteilung


def _promille(gew: Sequence[float]) -> tuple[int, ...]:
    s = sum(gew)
    if s <= 0:
        n = len(gew)
        return tuple([1000 // n] * (n - 1) + [1000 - (1000 // n) * (n - 1)])
    roh = [g / s * 1000.0 for g in gew]
    ganz = [int(x) for x in roh]
    rest = 1000 - sum(ganz)
    # Groesste Nachkommareste bekommen den Rest -- deterministisch, damit
    # jeder Rang dasselbe Ergebnis bekommt.
    ordn = sorted(range(len(roh)), key=lambda i: (-(roh[i] - ganz[i]), i))
    for i in ordn[:rest]:
        ganz[i] += 1
    return tuple(ganz)


def _ringfolge(m: Messung, welt: int) -> tuple[int, ...]:
    """Nach gemessener Kapazitaet geordneter Ring.

    Der Ring gewinnt unter anderem, weil er die Nachbarschaften so legen
    kann, dass der Verkehr lokal bleibt und geteilte Engpaesse nur einmal
    je Runde gequert werden. Die Ordnung entsteht hier **aus den
    Messwerten** (gierige Rundreise ueber die staerksten Kanten); die
    PCI-Naehe geht nur als Unentschieden-Brecher ein und nie als
    Entscheidung -- ``lspci`` ist Startschaetzung, nicht Wahrheit.
    """
    if welt < 2:
        return tuple(range(welt))
    gi = len(m.groessen) - 1
    start = min(range(welt), key=lambda r: (-_knotenkapazitaet(m, r), r))
    folge = [start]
    offen = set(range(welt)) - {start}
    while offen:
        letzter = folge[-1]
        naechster = min(
            offen,
            key=lambda j: (
                -min(m.kap(letzter, j, gi), m.kap(j, letzter, gi)),
                -_pci_naehe(m.bdfs[letzter], m.bdfs[j]),
                j,
            ),
        )
        folge.append(naechster)
        offen.discard(naechster)
    return tuple(folge)


def _pci_naehe(a: str, b: str) -> int:
    """Nur Unentschieden-Brecher: Laenge des gemeinsamen sysfs-Pfades.

    Zwei Karten unter demselben Switch teilen einen laengeren Pfad als
    zwei an verschiedenen Root-Ports. Das ist eine Topologieaussage, keine
    Kapazitaetsaussage -- deshalb steht sie hier und nirgends sonst.
    """
    try:
        pa = os.path.realpath(f"/sys/bus/pci/devices/{a}")
        pb = os.path.realpath(f"/sys/bus/pci/devices/{b}")
    except OSError:
        return 0
    n = 0
    for x, y in zip(pa.split("/"), pb.split("/")):
        if x != y:
            break
        n += 1
    return n


def _staffelplan(
    m: Messung, welt: int, verhaeltnis: float
) -> dict[int, tuple[tuple[int, ...], ...]]:
    """R3 als Fahrplan: wer darf gleichzeitig in dasselbe Ziel schreiben.

    Regel aus der Fan-in-Messung: der Deckel wird gleichmaessig je Quelle
    aufgeteilt, nicht nach Leistungsfaehigkeit. Eine x8-Quelle fiel neben
    einer x4-Quelle von 12,81 auf 6,75 GB/s, waehrend die x4-Quelle ihre
    6,46 behielt. Folge: Quellen, deren Kapazitaeten sich um mehr als
    ``verhaeltnis`` unterscheiden, gehoeren in verschiedene Wellen.
    """
    gi = len(m.groessen) - 1
    plan: dict[int, tuple[tuple[int, ...], ...]] = {}
    for ziel in range(welt):
        quellen = [q for q in range(welt) if q != ziel]
        if len(quellen) < 2:
            plan[ziel] = (tuple(quellen),)
            continue
        quellen.sort(key=lambda q: (-m.kap(q, ziel, gi), q))
        wellen: list[list[int]] = []
        for q in quellen:
            rate = m.kap(q, ziel, gi)
            gelegt = False
            for welle in wellen:
                raten = [m.kap(x, ziel, gi) for x in welle] + [rate]
                lo, hi = min(raten), max(raten)
                if lo > 0 and hi / lo <= verhaeltnis:
                    welle.append(q)
                    gelegt = True
                    break
            if not gelegt:
                wellen.append([q])
        plan[ziel] = tuple(tuple(w) for w in wellen)
    return plan


def _chunk_bytes(m: Messung, k: KollektivKonfig, welt: int) -> int:
    """Chunkgroesse aus gemessenen Schrittkosten und gemessener Rate.

    Pipelining ist nach dem Kollektiv-Entwurf wahrscheinlich mehr wert als
    die Topologiewahl (ohne Chunking laufen Hin- und Rueckweg des Blatts
    nacheinander: 266 statt 133 us). Zu kleine Chunks ersaeufen im
    Schrittaufwand, zu grosse ueberlappen nicht mehr.

    Regel: ein Chunk soll rund viermal so lange uebertragen, wie ein
    Schritt kostet. Beide Zahlen sind gemessen.
    """
    if k.chunk_kib is not None:
        return k.chunk_kib * 1024
    gi = len(m.groessen) - 1
    raten = [m.kap(i, j, gi) for i in range(welt) for j in range(welt) if i != j]
    rate = statistics.median(raten) if raten else 1.0
    schritt = m.schritt_s()
    if schritt <= 0 or rate <= 0:
        return 256 * 1024
    ziel = 4.0 * schritt * rate * 1e9
    kib = max(16, min(4096, 2 ** int(round(math.log2(max(ziel, 1.0) / 1024)))))
    return int(kib) * 1024


# ===========================================================================
# plane(): Messung + Konfiguration -> Plan
# ===========================================================================


def _fensterbedarf(algorithmus: str, nbytes: int, welt: int) -> int:
    """Gleiche Rechnung wie ``htccl_bar1.fensterbedarf`` -- hier lokal, damit
    der Planer ohne den BAR1-Transport benutzbar (und pruefbar) bleibt.

    **An den portierten Kernen nachgezaehlt**, nicht geschaetzt: Netz und
    Ring brauchen BEIDE ``2(R-1)`` Schlitze zu ``ceil(N/R)``.

    * Netz: ``R-1`` fuer den Reduce-Scatter und noch einmal ``R-1`` fuer den
      Allgather. Getrennt, weil zwischen "ich lese meine RS-Schlitze" und
      "der andere schreibt seinen AG-Chunk" keine Ordnung steht.
    * Ring: einer je Schritt, und es gibt ``2(R-1)`` Schritte. Zwei Schlitze
      abwechselnd zu benutzen ginge nur, wenn der Sender den Fortschritt
      seines NACHFOLGERS beobachtete -- er beobachtet aber nur seinen
      Vorgaenger.
    * Stern: ``R-1`` volle Puffer auf der Nabe.

    KORREKTUR: hier stand fuer den Ring ``2*2*anteil``, also vier Schlitze
    unabhaengig von R. Bei ``R=3`` ist das derselbe Wert (``2(R-1) = 4``)
    und deshalb nie aufgefallen; ab ``R=4`` war er zu klein. Ein zu kleiner
    Bedarf laesst einen Algorithmus zur Wahl zu, den die Abbildung nicht
    traegt -- und der Fehler faellt dann erst im Transport auf.
    """
    if welt < 2:
        return 0
    anteil = -(-nbytes // welt)
    if algorithmus in ("netz", "ring", "hierarchisch"):
        return 2 * (welt - 1) * anteil
    if algorithmus == "stern":
        return 2 * (welt - 1) * nbytes
    return 0


def plane(m: Messung, k: HtcclKonfig, quelle: str = "gemessen",
          fenster_bytes: Optional[int] = None) -> Plan:
    """Rein rechnend, ohne jede Ein-/Ausgabe -- damit pruefbar ohne Hardware.

    Ausschliesslich Funktion von ``(m, k, fenster_bytes)``. Zwei Raenge mit
    denselben Eingaben bekommen zwingend denselben Plan; genau das prueft
    ``HTCCLMatrixPlaner`` danach ueber die Pruefsumme.

    ``fenster_bytes`` ist, wieviel BAR1 je Ziel gleichzeitig abgebildet
    werden kann -- also eine **Faehigkeit**, keine Politik. Ist sie bekannt,
    faellt jeder Algorithmus heraus, dessen Fensterbedarf sie ueberschreitet.
    Das ist der zweite Grund, aus dem der Ring gewinnen kann: er braucht
    zwei Schlitze zu ``N/R``, das Netz ``R-1``. ``None`` heisst "unbekannt"
    und schliesst nichts aus -- nicht "unbegrenzt".
    """
    c = k.kollektiv
    welt = m.welt
    rollen, dom, blaetter = _rollen(m, c)

    fest_dom = _domaenen_aus_konfig(m, c)
    if fest_dom is not None:
        # Von Hand gesetzte Domaenen: der erste Eintrag ist die
        # Reduktionsdomaene, alles nicht Genannte wird Blatt.
        genannt = {r for g in fest_dom for r in g}
        dom = sorted(fest_dom[0])
        blaetter = [r for r in range(welt) if r not in dom]
        rollen = ["domaene" if r in dom else "blatt" for r in range(welt)]
        for r in range(welt):
            if r not in genannt and c.rollen.get(m.bdfs[r]) is None:
                rollen[r] = "blatt"

    eltern, aufteilung = _eltern_und_aufteilung(m, dom, blaetter, c)
    folge = _ringfolge(m, welt)
    staffeln = _staffelplan(m, welt, c.staffel_verhaeltnis)
    chunk = _chunk_bytes(m, c, welt)
    schritt = m.schritt_s()
    nabe = folge[0] if folge else 0

    stufen: list[Stufe] = []
    for gi, g in enumerate(m.groessen):
        vorhersage: dict[str, float] = {}
        vorhersage["netz"] = _zeit_netz(m, gi, g, range(welt), staffeln, schritt)
        vorhersage["ring"] = _zeit_ring(m, gi, g, folge, schritt)
        vorhersage["stern"] = _zeit_stern(m, gi, g, nabe, welt, schritt)
        if blaetter and len(dom) >= 2:
            # Innerhalb der Domaene gewinnt, was dort gewinnt.
            innen_netz = _zeit_netz(m, gi, g, dom, staffeln, schritt)
            innen_ring = _zeit_ring(
                m, gi, g, [d for d in folge if d in dom] or dom, schritt
            )
            vorhersage["hierarchisch"] = _zeit_hierarchisch(
                m, gi, g, dom, blaetter, eltern, aufteilung,
                min(innen_netz, innen_ring), schritt, welt,
            )

        # Faehigkeitsgrenze: was nicht ins Fenster passt, steht nicht zur
        # Wahl. Ausdruecklich VOR der Politik -- eine Konfiguration darf
        # keinen Algorithmus erzwingen, den die Hardware nicht abbilden kann.
        zu_gross = {
            a: _fensterbedarf(a, g, welt) for a in list(vorhersage)
            if fenster_bytes is not None
            and _fensterbedarf(a, g, welt) > fenster_bytes
        }
        moeglich = {a: v for a, v in vorhersage.items() if a not in zu_gross}

        if c.algorithmus != "auto":
            gewaehlt = c.algorithmus
            if gewaehlt in zu_gross:
                raise KonfigFehler(
                    f"kollektiv.algorithmus={gewaehlt} erzwungen, aber bei "
                    f"{_kib(g)} und {welt} Raengen braucht er "
                    f"{zu_gross[gewaehlt] // 1024} KiB Fenster; abbildbar sind "
                    f"{fenster_bytes // 1024} KiB. Kein stiller Rueckfall -- "
                    f"entweder kleiner chunken oder einen anderen Algorithmus "
                    f"waehlen."
                )
            grund = f"erzwungen ueber kollektiv.algorithmus={c.algorithmus}"
        else:
            if not moeglich:
                raise KonfigFehler(
                    f"Bei {_kib(g)} und {welt} Raengen passt KEIN Algorithmus "
                    f"in das abbildbare Fenster von {fenster_bytes // 1024} "
                    f"KiB (Bedarf: "
                    f"{ {a: b // 1024 for a, b in zu_gross.items()} } KiB). "
                    f"Das ist ein Startfehler und keine leise Umleitung: "
                    f"kleiner chunken oder den Direktpfad fuer diese Groesse "
                    f"ausschliessen."
                )
            gewaehlt = min(moeglich, key=lambda a: (moeglich[a], a))
            grund = _grund(m, gi, g, welt, gewaehlt, vorhersage, staffeln,
                           c.saettigung_anteil, blaetter)
            if zu_gross:
                grund += (
                    "  Ausgeschlossen, weil zu gross fuers Fenster: "
                    + ", ".join(
                        f"{a} ({b // 1024} KiB > {fenster_bytes // 1024} KiB)"
                        for a, b in sorted(zu_gross.items())
                    ) + "."
                )
        stufen.append(
            Stufe(
                von_bytes=g,
                max_bytes=g if gi + 1 < len(m.groessen) else -1,
                algorithmus=gewaehlt,
                vorhersage_s={a: round(v, 9) for a, v in vorhersage.items()},
                grund=grund,
            )
        )

    stufen = _leiter_glaetten(stufen)

    zus = {
        "planer": c.planer,
        "algorithmus": c.algorithmus,
        "chunk_kib": "auto" if c.chunk_kib is None else c.chunk_kib,
        "blatt_schwelle": c.blatt_schwelle,
        "aufteilung": c.aufteilung if isinstance(c.aufteilung, str) else "von Hand",
        "staffel_verhaeltnis": c.staffel_verhaeltnis,
        "rollen (von Hand)": dict(c.rollen) or "-",
        "domaenen (von Hand)": [list(g) for g in c.domaenen] or "-",
        "nic": k.nic,
        "mess.groessen_kib": list(c.mess.groessen_kib),
        "mess.budget_ms": c.mess.budget_ms,
    }
    return Plan(
        welt=welt,
        bdfs=m.bdfs,
        rollen=tuple(rollen),
        domaene=tuple(dom),
        blaetter=tuple(blaetter),
        eltern={k2: v for k2, v in sorted(eltern.items())},
        aufteilung={k2: v for k2, v in sorted(aufteilung.items())},
        ringfolge=folge,
        leiter=tuple(stufen),
        chunk_bytes=chunk,
        staffeln=staffeln,
        konfig_zusammenfassung=zus,
        messung=m,
        quelle=quelle,
    )


def _grund(m: Messung, gi: int, g: int, welt: int, gewaehlt: str,
           vorhersage: Mapping[str, float],
           staffeln: Mapping[int, tuple[tuple[int, ...], ...]],
           saettigung_anteil: float, blaetter: Sequence[int]) -> str:
    """Ein Satz, warum. Ohne den ist die Wahl nicht nachvollziehbar."""
    netz, ring = vorhersage.get("netz", 0.0), vorhersage.get("ring", 0.0)
    anteil = g / welt
    # Auslastung, die das Netz auf dem am staerksten beschriebenen Ziel
    # erzeugen wuerde: was die R-1 Quellen zusammen anliefern wollen,
    # gegen den gemessenen Deckel dieses Ziels.
    last = 0.0
    for ziel in range(welt):
        deckel = m.deckel(ziel, gi)
        quellen = [m.kap(q, ziel, gi) for q in range(welt) if q != ziel]
        if deckel > 0 and quellen:
            last = max(last, sum(quellen) / deckel)
    gesaettigt = last > 1.0 / max(saettigung_anteil, 1e-9)
    gestaffelt = sum(1 for w in staffeln.values() if len(w) > 1)
    t = []
    if gewaehlt == "netz":
        if gesaettigt:
            t.append(
                f"Netz trotz Saettigung (Nachfrage/Deckel {last:.2f}): die "
                f"Schrittzahl ueberwiegt. 2 Schritte statt {2 * (welt - 1)}, "
                f"und ein Schritt kostet hier {m.schritt_s() * 1e6:.1f} us."
            )
        else:
            t.append(
                f"Netz: 2 Schritte statt {2 * (welt - 1)}; Nachfrage/Deckel "
                f"{last:.2f} liegt unter der Saettigung -- die Gleichzeitigkeit "
                f"ist dort nahezu gratis (gemessen Quotient 0,99 bei 20 KiB)."
            )
        if gestaffelt:
            t.append(f"{gestaffelt} Ziel(e) muessen dabei gestaffelt werden.")
    elif gewaehlt == "ring":
        t.append(
            f"Ring: Nachfrage/Deckel {last:.2f} (Schwelle "
            f"{1.0 / saettigung_anteil:.2f}) -- an der Saettigung bringt die "
            f"Gleichzeitigkeit des Netzes nichts (gemessen 1,03x bei 1 MiB) "
            f"und der Deckel wird gleichmaessig statt proportional geteilt, "
            f"verschenkt also die schnellen Kanten. Im Ring empfaengt jeder "
            f"Rang von genau einem Nachbarn."
        )
        if gestaffelt:
            t.append(f"{gestaffelt} Ziel(e) muessten im Netz gestaffelt werden.")
    elif gewaehlt == "hierarchisch":
        t.append(
            f"Hierarchisch: {len(blaetter)} Blatt/Blaetter tragen keinen "
            f"Transit, sondern senden nur den eigenen Beitrag und empfangen "
            f"nur das Ergebnis; die Domaene reduziert untereinander."
        )
    elif gewaehlt == "stern":
        t.append("Stern: nur gewaehlt, weil alles andere schlechter vorhersagt "
                 "oder weil er erzwungen wurde.")
    t.append(f"(netz {netz * 1e6:.1f} us / ring {ring * 1e6:.1f} us, "
             f"Anteil je Kante {anteil / 1024:.0f} KiB)")
    return " ".join(t)


def _leiter_glaetten(stufen: list[Stufe]) -> list[Stufe]:
    """Aufeinanderfolgende Stufen mit gleichem Algorithmus zusammenfassen.

    Eine Leiter mit drei Eintraegen desselben Algorithmus ist keine Leiter,
    sondern eine Zeile -- und sie liest sich in der Erklaerung auch so.
    """
    aus: list[Stufe] = []
    for s in stufen:
        if aus and aus[-1].algorithmus == s.algorithmus:
            # von_bytes bleibt stehen: die Stufe reicht dann von der
            # kleinsten bis zur groessten Groesse, fuer die sie gilt.
            aus[-1] = replace(aus[-1], max_bytes=s.max_bytes,
                              vorhersage_s=s.vorhersage_s, grund=s.grund)
        else:
            aus.append(s)
    if aus:
        aus[-1] = replace(aus[-1], max_bytes=-1)
    return aus


# ===========================================================================
# Zwischenspeicher
# ===========================================================================


def _vorgabe_cache() -> str:
    basis = os.environ.get("XDG_CACHE_HOME") or os.path.expanduser("~/.cache")
    return os.path.join(basis, "sglang", "htccl_matrix.json")


def fingerabdruck(bdfs: Sequence[str], namen: Sequence[str],
                  k: HtcclKonfig) -> str:
    """Kartenliste, PCI-Adressen, Treiberversion, Patch-Stand, Messparameter.

    Aendert sich davon etwas, wird neu gemessen. Der Patch-Stand gehoert
    ausdruecklich dazu: ohne den Regkey traegt der Direktpfad nicht, und
    eine mit ihm gemessene Matrix ist danach falsch.
    """
    m = k.kollektiv.mess
    teile = {
        "version": PLANER_VERSION,
        "bdfs": list(bdfs),
        "namen": list(namen),
        "treiber": _treiberversion(),
        "patch": _patchstand(),
        "groessen_kib": list(m.groessen_kib),
        "wiederholungen": m.wiederholungen,
        "fanin": m.fanin,
        "duplex": m.duplex,
    }
    roh = json.dumps(teile, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(roh.encode()).hexdigest()[:24]


def _treiberversion() -> str:
    try:
        with open("/proc/driver/nvidia/version") as f:
            return f.read().strip().split("\n")[0]
    except OSError:
        return "unbekannt"


def _patchstand() -> str:
    """Regkeys des Treibers, soweit sichtbar.

    ``RegistryDwords`` ist die Stelle, an der ``RMSmallBarP2PPeerBar1`` und
    ``RMPcieP2PType`` gesetzt werden. Steht dort nichts, ist der Direktpfad
    nicht freigeschaltet -- eine damit gemessene Matrix darf nicht fuer
    einen Lauf mit Patch wiederverwendet werden und umgekehrt.
    """
    try:
        with open("/proc/driver/nvidia/params") as f:
            zeilen = [
                z.strip() for z in f
                if z.startswith(("RegistryDwords", "EnableResizableBar"))
            ]
        return "|".join(zeilen)
    except OSError:
        return "unbekannt"


def lies_cache(pfad: str, fa: str) -> Optional[Messung]:
    try:
        with open(pfad) as f:
            d = json.load(f)
    except (OSError, json.JSONDecodeError):
        return None
    if d.get("fingerabdruck") != fa:
        return None
    try:
        return Messung.aus_dict(d["messung"])
    except (KeyError, TypeError, ValueError) as e:
        logger.warning("HTCCL-Matrix: Zwischenspeicher %s unlesbar (%s); neu messen.",
                       pfad, e)
        return None


def schreibe_cache(pfad: str, fa: str, m: Messung) -> None:
    try:
        p = pathlib.Path(pfad)
        p.parent.mkdir(parents=True, exist_ok=True)
        tmp = p.with_suffix(p.suffix + f".{os.getpid()}.tmp")
        tmp.write_text(json.dumps(
            {"fingerabdruck": fa, "messung": m.als_dict()},
            sort_keys=True, indent=1,
        ))
        tmp.replace(p)
    except OSError as e:
        logger.warning("HTCCL-Matrix: Zwischenspeicher %s nicht schreibbar (%s).",
                       pfad, e)


# ===========================================================================
# Der Planer: Ablauf beim Start
# ===========================================================================


class HTCCLMatrixPlaner:
    """Misst, plant, prueft die Rangeinheitlichkeit, erklaert.

    Aufruf::

        planer = HTCCLMatrixPlaner(cpu_group, device, konfig=lade_konfig())
        plan = planer.plan()            # misst beim ersten Aufruf
        logger.info("%s", plan.erklaerung())

    Der Plan ist danach **eingefroren**. Das ist keine Bequemlichkeit,
    sondern Auflage: Decode laeuft in aufgezeichneten CUDA-Graphen, und die
    Wahl muss zum Aufzeichnungszeitpunkt feststehen. Eine Umschaltung je
    Nachricht erzwaenge ein Re-Capture. Dynamik nach Last ist nur ausserhalb
    aufgezeichneter Bereiche zulaessig und dann ueber mehrere
    aufgezeichnete Varianten, nicht ueber eine Aenderung an diesem Plan.
    """

    def __init__(self, cpu_group, device, konfig: Optional[HtcclKonfig] = None,
                 fuehler: Optional[Fuehler] = None,
                 fenster_bytes: Optional[int] = None):
        import torch.distributed as dist

        self.cpu_group = cpu_group
        self.device = device
        self.konfig = konfig if konfig is not None else lade_konfig()
        self.rank = dist.get_rank(cpu_group)
        self.welt = dist.get_world_size(cpu_group)
        self._fuehler = fuehler
        # Faehigkeit, nicht Politik: wieviel BAR1 je Ziel gleichzeitig
        # abbildbar ist. None = unbekannt (schliesst nichts aus).
        self.fenster_bytes = fenster_bytes
        self._plan: Optional[Plan] = None

    # -- oeffentlich --------------------------------------------------------

    def plan(self) -> Plan:
        if self._plan is None:
            self._plan = self._baue()
        return self._plan

    # -- innen --------------------------------------------------------------

    def _baue(self) -> Plan:
        import torch.distributed as dist

        c = self.konfig.kollektiv
        bdfs, namen = self._karten()
        fa = fingerabdruck(bdfs, namen, self.konfig)

        if c.planer == "aus":
            m = self._synthetische_messung(bdfs, namen)
            p = plane(m, self.konfig, quelle="fest",
                      fenster_bytes=self.fenster_bytes)
            self._pruefe_einheitlich(p)
            return p

        pfad = c.mess.cache or _vorgabe_cache()
        m = None
        if not c.mess.cache_aus:
            # Nur Rang 0 liest und verteilt -- sonst koennten zwei Raenge
            # verschieden alte Dateien sehen und verschieden planen.
            traeger: list[Any] = [None]
            if self.rank == 0:
                gefunden = lies_cache(pfad, fa)
                traeger = [gefunden.als_dict() if gefunden is not None else None]
            dist.broadcast_object_list(
                traeger, src=dist.get_global_rank(self.cpu_group, 0),
                group=self.cpu_group,
            )
            if traeger[0] is not None:
                m = Messung.aus_dict(traeger[0])
                logger.info(
                    "HTCCL-Matrix: Startmessung uebersprungen, "
                    "Zwischenspeicher %s passt (Fingerabdruck %s). "
                    "Neu messen mit SGLANG_HTCCL_MATRIX_CACHE_AUS=1.", pfad, fa,
                )

        quelle = "zwischenspeicher"
        if m is None:
            if c.planer == "fest":
                # 'fest' heisst ausdruecklich: nicht messen, den abgelegten
                # Befund verwenden. Fehlt er, ist das ein benannter Fehler --
                # heimlich doch zu messen waere genau der stille Rueckfall,
                # den dieser Entwurf verbietet.
                raise KonfigFehler(
                    f"kollektiv.planer=fest, aber unter {pfad!r} liegt kein "
                    f"Befund mit dem Fingerabdruck {fa} (Kartenliste, "
                    f"PCI-Adressen, Treiberversion, Patch-Stand, "
                    f"Messparameter). Entweder einmal mit planer=auto messen "
                    f"lassen oder den Pfad ueber SGLANG_HTCCL_MATRIX_CACHE "
                    f"auf einen gueltigen Befund zeigen."
                )
            quelle = "gemessen"
            m = self._messe(bdfs, namen)
            if not c.mess.cache_aus and self.rank == 0:
                schreibe_cache(pfad, fa, m)

        p = plane(m, self.konfig, quelle=quelle,
                  fenster_bytes=self.fenster_bytes)
        self._pruefe_einheitlich(p)
        return p

    # -- Karten -------------------------------------------------------------

    def _karten(self) -> tuple[tuple[str, ...], tuple[str, ...]]:
        import torch
        import torch.distributed as dist

        bdf = bdf_der_karte(self.device)
        try:
            name = torch.cuda.get_device_name(self.device)
        except Exception:                       # pragma: no cover
            name = "unbekannt"
        gesammelt: list[Any] = [None] * self.welt
        dist.all_gather_object(gesammelt, (bdf, name), group=self.cpu_group)
        bdfs = tuple(str(x[0]) for x in gesammelt)   # type: ignore[index]
        namen = tuple(str(x[1]) for x in gesammelt)  # type: ignore[index]
        if len(set(bdfs)) != len(bdfs):
            logger.warning(
                "HTCCL-Matrix: doppelte PCI-Adressen %s. Rollen und Domaenen "
                "werden ueber die Adresse angesprochen; mit Doppelungen "
                "greift eine Konfiguration nicht eindeutig.", bdfs,
            )
        return bdfs, namen

    # -- Messung ------------------------------------------------------------

    def _messe(self, bdfs, namen) -> Messung:
        import torch.distributed as dist

        c = self.konfig.kollektiv
        mk = self._budget_anpassen(c.mess)
        groessen = tuple(g * 1024 for g in mk.groessen_kib)
        f = self._fuehler if self._fuehler is not None else EigenlastFuehler(
            self.device, max_bytes=max(groessen),
            wiederholungen=mk.wiederholungen,
        )
        t_start = time.perf_counter()
        m = Messung(
            welt=self.welt, groessen=groessen, bdfs=bdfs, namen=namen,
            fuehler=f.name(),
        )

        # -- Phase 1: Eigenlast, gestaffelt. -------------------------------
        # Nacheinander, nicht gleichzeitig: sonst konkurrieren R Karten um
        # denselben Host-Speicher und die Zahlen messen die Gegenseite.
        eigen_aus: list[float] = []
        eigen_ein: list[float] = []
        eigen_dup: list[float] = []
        for besitzer in range(self.welt):
            dist.barrier(group=self.cpu_group)
            if besitzer == self.rank:
                for g in groessen:
                    eigen_aus.append(_quant(f.eigenlast(g, "aus")))
                    eigen_ein.append(_quant(f.eigenlast(g, "ein")))
                if mk.duplex:
                    for g in groessen:
                        d = f.eigenlast_duplex(g)
                        eigen_dup.append(_quant(d) if d is not None else 0.0)
            dist.barrier(group=self.cpu_group)

        gesammelt: list[Any] = [None] * self.welt
        dist.all_gather_object(
            gesammelt, (eigen_aus, eigen_ein, eigen_dup), group=self.cpu_group
        )
        for r, (a, e, d) in enumerate(gesammelt):    # type: ignore[misc]
            m.aus[r] = list(a)
            m.ein[r] = list(e)
            if d and any(x > 0 for x in d):
                m.duplex_summe[r] = list(d)
            m.latenz_s[r] = _latenz_fit(groessen, list(a))

        # -- Phase 2: echte Kantenmessung, wenn ein Paar-Fuehler da ist. ----
        hat_paar = self._hat_paarfuehler(f, groessen)
        if hat_paar:
            for von in range(self.welt):
                for nach in range(self.welt):
                    if von == nach:
                        continue
                    dist.barrier(group=self.cpu_group)
                    werte: list[float] = []
                    if self.rank == von:
                        for g in groessen:
                            r = f.paar(nach, g)
                            werte.append(_quant(r) if r is not None else 0.0)
                    elif self.rank == nach:
                        for g in groessen:
                            f.paar_empfang(von, g)
                    dist.barrier(group=self.cpu_group)
                    traeger: list[Any] = [werte if self.rank == von else None]
                    dist.broadcast_object_list(
                        traeger, src=dist.get_global_rank(self.cpu_group, von),
                        group=self.cpu_group,
                    )
                    if traeger[0]:
                        m.kante[(von, nach)] = list(traeger[0])

            # -- Phase 3: Fan-in (R3). Alle Quellen gleichzeitig ins Ziel. --
            if mk.fanin:
                for ziel in range(self.welt):
                    dist.barrier(group=self.cpu_group)
                    werte = []
                    if self.rank != ziel:
                        for g in groessen:
                            r = f.paar(ziel, g)
                            werte.append(_quant(r) if r is not None else 0.0)
                    else:
                        for g in groessen:
                            f.paar_empfang(-1, g)
                    dist.barrier(group=self.cpu_group)
                    alle: list[Any] = [None] * self.welt
                    dist.all_gather_object(alle, werte, group=self.cpu_group)
                    deckel = []
                    for gi in range(len(groessen)):
                        s = sum(v[gi] for r, v in enumerate(alle) if r != ziel and v)
                        deckel.append(_quant(s))
                    m.fanin_deckel[ziel] = deckel
                    for r, v in enumerate(alle):
                        if r != ziel and v:
                            m.fanin_anteile[(r, ziel)] = list(v)
                m.hinweise.append(
                    "Fan-in: die Quellen starten gemeinsam an einer Barriere, "
                    "laufen die Groessenleiter danach aber unabhaengig ab. Bei "
                    "stark ungleichen Karten ueberlappen die Groessen nicht "
                    "vollstaendig; der gemessene Deckel ist damit eher eine "
                    "UNTERgrenze der Gleichzeitigkeit. Eine verschraenkte "
                    "Schleife hinter einem gemeinsamen Startgatter (wie in "
                    "sonden/nebenlauf_probe.cu) waere genauer und braeuchte "
                    "einen Fuehler, der beide Straenge selbst verschraenkt."
                )
            # -- Phase 4: geteilte Engpaesse. ALLE Paare gleichzeitig. ------
            # Der einzige Weg, den Unterschied zwischen "Netz" und "Ring" zu
            # messen statt zu behaupten: im Netz reden alle Paare
            # gleichzeitig, im Ring nur die Nachbarn. Was dabei ueber
            # Switch-Uplinks und NUMA-Spruenge verlorengeht, steht in keiner
            # Kantenkapazitaet -- die wurde je Kante EINZELN gemessen.
            #
            # Methodik wie in sonden/nebenlauf_probe.cu: derselbe Strang
            # zweimal, allein und gemeinsam, und verglichen werden RATEN,
            # nicht Wanduhrzeiten. Ein Vergleich von Wanduhr gegen
            # Einzeltransfer waere um die Wiederholungszahl daneben.
            #
            # Je Runde schreibt Rang i an Rang (i+versatz) mod R. Das ist
            # eine perfekte Paarung: jeder sendet genau einmal und empfaengt
            # genau einmal, also keine Fan-in-Konkurrenz -- gemessen wird
            # hier ausschliesslich der geteilte Engpass, nicht der Deckel.
            gemeinsam: dict[tuple[int, int], list[float]] = {}
            for versatz in range(1, self.welt):
                nach = (self.rank + versatz) % self.welt
                dist.barrier(group=self.cpu_group)
                meine: list[float] = []
                for g in groessen:
                    r = f.paar(nach, g)
                    meine.append(_quant(r) if r is not None else 0.0)
                gesammelt2: list[Any] = [None] * self.welt
                dist.all_gather_object(gesammelt2, meine, group=self.cpu_group)
                for von, werte2 in enumerate(gesammelt2):
                    if werte2:
                        gemeinsam[(von, (von + versatz) % self.welt)] = list(werte2)

            faktoren = []
            for gi in range(len(groessen)):
                # Schlimmste Verschlechterung ueber alle Kanten: der Netz-
                # Schritt ist so schnell wie seine langsamste Kante.
                schlimmster = 1.0
                for (von, nach), raten in gemeinsam.items():
                    allein = m.kap(von, nach, gi)
                    zus = raten[gi]
                    if allein > 0 and zus > 0:
                        schlimmster = max(schlimmster, allein / zus)
                faktoren.append(_quant(schlimmster))
            m.netz_faktor = [max(1.0, x) for x in faktoren]
            m.netz_faktor_gemessen = True
        else:
            m.hinweise.append(
                "Kein Paar-Fuehler: die Kantenkapazitaeten sind aus der "
                "Eigenlast geschaetzt (min(Ausgang, Eingang)) und damit eine "
                "OBERGRENZE -- geteilte Engpaesse wie ein Switch-Uplink oder "
                "ein zweiter Root-Complex sind darin nicht sichtbar. Der "
                "Fan-in-Deckel ist die Eingangsrate, nicht der gemessene "
                "BAR1-Deckel."
            )
            m.netz_faktor = [1.0] * len(groessen)
            m.netz_faktor_gemessen = False
            m.hinweise.append(
                "GETEILTE ENGPAESSE UNGEMESSEN (netz_faktor = 1,0 gesetzt). "
                "Ohne Paar-Fuehler laesst sich nicht messen, wieviel eine "
                "Kante verliert, wenn ALLE Paare gleichzeitig reden statt nur "
                "eines. Genau daran haengt die Erwartung, dass der Ring an "
                "der Saettigung gewinnt: bei reinen Kantenkapazitaeten und "
                "Fan-in-Deckeln bewegen Netz und Ring dieselben Bytes durch "
                "denselben Deckel, und das Netz gewinnt dann IMMER ueber die "
                "Schrittzahl. Solange dieser Faktor 1,0 ist, waehlt der "
                "Planer folgerichtig nie den Ring -- das ist ein fehlender "
                "Messwert, kein Ergebnis. Ueberstimmbar mit "
                "SGLANG_HTCCL_NETZ_FAKTOR."
            )

        ueber = os.environ.get(_ENV_PRAEFIX + "NETZ_FAKTOR")
        if ueber:
            werte = [float(x) for x in ueber.replace(";", ",").split(",")]
            if len(werte) == 1:
                werte = werte * len(groessen)
            if len(werte) != len(groessen):
                raise KonfigFehler(
                    f"SGLANG_HTCCL_NETZ_FAKTOR={ueber!r}: erwartet ein Wert "
                    f"oder {len(groessen)} Werte (je Groesse "
                    f"{[g // 1024 for g in groessen]} KiB)."
                )
            m.netz_faktor = [max(1.0, x) for x in werte]
            m.netz_faktor_gemessen = False
            m.hinweise.append(
                f"netz_faktor von Hand auf {m.netz_faktor} gesetzt "
                f"(SGLANG_HTCCL_NETZ_FAKTOR) -- nicht gemessen."
            )

        m.dauer_s = time.perf_counter() - t_start
        self._plausibel(m)
        return m

    def _hat_paarfuehler(self, f: Fuehler, groessen) -> bool:
        """Gruppenweit entscheiden, nicht je Rang.

        Ein Rang, der glaubt zu messen, waehrend die anderen an der Barriere
        stehen, verklemmt den Start. Deshalb: alle fragen, und nur wenn
        ALLE koennen, wird gemessen.
        """
        import torch.distributed as dist

        try:
            kann = f.paar(-1, groessen[0]) is not None
        except NotImplementedError:
            kann = False
        except Exception as e:
            logger.warning("HTCCL-Matrix: Paar-Fuehler meldet Fehler (%s); "
                           "Eigenlast-Schaetzung.", e)
            kann = False
        alle: list[Any] = [None] * self.welt
        dist.all_gather_object(alle, bool(kann), group=self.cpu_group)
        return all(bool(x) for x in alle)

    def _budget_anpassen(self, mk: MessKonfig) -> MessKonfig:
        """Grobe Vorabschaetzung gegen ``budget_ms``.

        Kosten wachsen mit R^2, sobald ein Paar-Fuehler da ist. Statt die
        Startzeit explodieren zu lassen, wird zuerst die Groessenleiter
        gekuerzt (die mittlere Groesse faellt zuerst -- sie trennt am
        wenigsten) und dann die Wiederholungszahl. Was gekuerzt wurde, wird
        protokolliert; heimlich duennere Belege gibt es nicht.
        """
        paare = self.welt * (self.welt - 1)
        # Rundenzahl je Phase:
        #   1 Eigenlast, gestaffelt          -> R
        #   2 Kanten einzeln (Paar-Fuehler)  -> R(R-1)
        #   3 Fan-in, ein Ziel je Runde      -> R
        #   4 alle Paare gleichzeitig        -> R-1
        # Ohne Paar-Fuehler entfallen 2 bis 4.
        mit_paar = self._fuehler is not None
        runden = self.welt + (paare + self.welt + self.welt - 1 if mit_paar else 0)

        def schaetz(g_kib: Sequence[int], reps: int) -> float:
            # ~6 GB/s als grobe Annahme fuer die Vorabschaetzung; sie
            # entscheidet nur, wieviel gemessen wird, nie was herauskommt.
            uebertragung = sum((g * 1024) * reps / 6e9 * 1000 for g in g_kib)
            # 2 Barrieren je Runde, gloo grob 0,3 ms.
            return runden * (uebertragung + 0.6 * len(g_kib))

        g = list(mk.groessen_kib)
        reps = mk.wiederholungen
        gekuerzt = []
        while schaetz(g, reps) > mk.budget_ms and len(g) > 2:
            weg = g.pop(len(g) // 2)
            gekuerzt.append(f"{weg} KiB")
        while schaetz(g, reps) > mk.budget_ms and reps > 4:
            reps //= 2
        if gekuerzt or reps != mk.wiederholungen:
            logger.info(
                "HTCCL-Matrix: Messbudget %.0f ms -- Groessen %s entfernt, "
                "Wiederholungen %d -> %d. Voll messen mit "
                "SGLANG_HTCCL_MESS_BUDGET_MS=<mehr>.",
                mk.budget_ms, gekuerzt or "keine", mk.wiederholungen, reps,
            )
        return replace(mk, groessen_kib=tuple(g), wiederholungen=reps)

    def _plausibel(self, m: Messung) -> None:
        """Meldet Abweichungen von den Referenzwerten dieses Rigs.

        Kein Abbruch und keine Entscheidung -- nur ein Hinweis, damit ein
        kaputter Messaufbau auffaellt, bevor er als Rig-Eigenheit
        durchgeht. Vierzehn plausible Annahmen sind in diesem Vorhaben
        bereits an der Hardware gescheitert; eine stille Messung waere die
        fuenfzehnte.
        """
        gi = len(m.groessen) - 1
        for r in range(m.welt):
            if m.aus[r][gi] <= 0.05 or m.ein[r][gi] <= 0.05:
                m.hinweise.append(
                    f"Rang {r} ({m.bdfs[r]}): gemessene Rate nahe null "
                    f"(aus={m.aus[r][gi]:.2f}, ein={m.ein[r][gi]:.2f}) -- das "
                    f"ist ein Messfehler, keine Karteneigenschaft."
                )
            if m.duplex_summe:
                f = m.duplex_faktor(r, gi)
                if f > 1.9:
                    m.hinweise.append(
                        f"Rang {r}: Duplexfaktor {f:.2f} bei der groessten "
                        f"Groesse. Gemessen wurde auf diesem Rig "
                        f"{_BELEG_DUPLEX_SUMME_1MIB:.2f}; ein Wert nahe 2 "
                        f"deutet auf nicht wirklich gleichzeitige Stroeme hin."
                    )
        if m.fanin_deckel:
            for j, d in m.fanin_deckel.items():
                if d[gi] > 2.0 * m.ein[j][gi] and m.ein[j][gi] > 0:
                    m.hinweise.append(
                        f"Ziel {j}: gemessener Fan-in-Deckel {d[gi]:.2f} GB/s "
                        f"liegt weit ueber der Eingangsrate {m.ein[j][gi]:.2f} "
                        f"GB/s -- pruefen, ob die Quellen wirklich gleichzeitig "
                        f"gelaufen sind."
                    )

    def _synthetische_messung(self, bdfs, namen) -> Messung:
        """``planer: aus`` -- nichts messen, alles aus der Konfiguration.

        Gleichverteilte Platzhalterraten, damit die Rollen ausschliesslich
        aus ``rollen``/``domaenen`` kommen. Der Plan meldet ``quelle=fest``,
        damit in der Erklaerung nicht der Eindruck entsteht, hier sei etwas
        gemessen worden.
        """
        groessen = tuple(g * 1024 for g in self.konfig.kollektiv.mess.groessen_kib)
        m = Messung(welt=self.welt, groessen=groessen, bdfs=bdfs, namen=namen,
                    fuehler="keiner (planer=aus)")
        for r in range(self.welt):
            m.aus[r] = [1.0] * len(groessen)
            m.ein[r] = [1.0] * len(groessen)
            m.latenz_s[r] = 0.0
        m.hinweise.append(
            "planer=aus: nichts gemessen. Rollen, Domaenen und Algorithmus "
            "stammen ausschliesslich aus der Konfiguration."
        )
        return m

    def _pruefe_einheitlich(self, p: Plan) -> None:
        """Der Plan MUSS auf allen Raengen gleich sein."""
        import torch.distributed as dist

        summen: list[Any] = [None] * self.welt
        dist.all_gather_object(summen, p.pruefsumme(), group=self.cpu_group)
        if len(set(summen)) != 1:
            abweichler = {r: s for r, s in enumerate(summen) if s != summen[0]}
            raise RuntimeError(
                f"HTCCL-Matrix: die Raenge haben VERSCHIEDENE Plaene "
                f"({summen}). Abweichler: {abweichler}. Das ist ein "
                f"Startfehler und keine Warnung -- die Kollektive setzen "
                f"voraus, dass alle Raenge dieselbe Zerlegung fahren. "
                f"Haeufigste Ursachen: eine SGLANG_HTCCL_*-Variable oder eine "
                f"Konfigurationsdatei ist nicht auf allen Raengen gleich, "
                f"oder fenster_bytes wurde je Rang verschieden hereingereicht "
                f"(es muss das Minimum ueber alle Ziele sein -- massgeblich "
                f"ist, was sich UEBERALL abbilden laesst)."
            )


def _latenz_fit(groessen: Sequence[int], raten: Sequence[float]) -> float:
    """t(N) = latenz + N/rate, angepasst ueber die Groessenleiter.

    Aus den gemessenen Raten je Groesse ergeben sich Zeiten; die Gerade
    darueber trennt Startkosten von Durchsatz. Die Startkosten sind das,
    was ein Kollektivschritt mindestens kostet -- und bei 20 KiB war die
    Quittungsschleife mit ~3 us von 13,47 us der groesste Einzelposten des
    Kollektivs, also genau die Groesse, um die es geht.

    VORBEHALT: gemessen wird hier die Startkosten der Kopiermaschine, nicht
    die der Quittungsschleife des Kollektivs. Das ist eine UNTERGRENZE.
    Wer es besser weiss, setzt ``SGLANG_HTCCL_SCHRITT_US``.
    """
    ueber = os.environ.get(_ENV_PRAEFIX + "SCHRITT_US")
    if ueber:
        return float(ueber) / 1e6
    punkte = [
        (float(n), n / (r * 1e9)) for n, r in zip(groessen, raten) if r > 0
    ]
    if len(punkte) < 2:
        return 0.0
    n_mittel = sum(x for x, _ in punkte) / len(punkte)
    t_mittel = sum(y for _, y in punkte) / len(punkte)
    zaehler = sum((x - n_mittel) * (y - t_mittel) for x, y in punkte)
    nenner = sum((x - n_mittel) ** 2 for x, _ in punkte)
    if nenner <= 0:
        return 0.0
    steigung = zaehler / nenner
    achsenabschnitt = t_mittel - steigung * n_mittel
    return max(0.0, achsenabschnitt)


# ===========================================================================
# Hilfsmittel
# ===========================================================================


def bdf_der_karte(device) -> str:
    """PCI-Adresse ``0000:05:00.0`` der Karte hinter ``device``.

    Ueber ``cudaDeviceGetPCIBusId``, weil das die einzige Zuordnung ist,
    die auch bei gesetztem ``CUDA_VISIBLE_DEVICES`` stimmt -- das Ordinal
    allein ist ueber Prozessgrenzen bedeutungslos, und die Konfiguration
    spricht Karten ueber ihre Adresse an.
    """
    import torch

    ordinal = device.index if hasattr(device, "index") else int(device)
    if ordinal is None:
        ordinal = torch.cuda.current_device()
    # Bevorzugt der Weg ohne ctypes, wenn torch ihn anbietet.
    try:
        props = torch.cuda.get_device_properties(ordinal)
        bus_id = getattr(props, "pci_bus_id", None)
        if bus_id:
            return _norm_bdf(str(bus_id))
    except Exception:
        pass
    try:
        import ctypes

        lib = ctypes.CDLL(
            "libamdhip64.so" if torch.version.hip is not None else "libcudart.so"
        )
        fn = (lib.hipDeviceGetPCIBusId if torch.version.hip is not None
              else lib.cudaDeviceGetPCIBusId)
        puffer = ctypes.create_string_buffer(32)
        fn.restype = ctypes.c_int
        if fn(puffer, ctypes.c_int(32), ctypes.c_int(int(ordinal))) == 0:
            return _norm_bdf(puffer.value.decode())
    except Exception as e:                      # pragma: no cover
        logger.warning("HTCCL-Matrix: PCI-Adresse nicht ermittelbar (%s).", e)
    return f"unbekannt-{ordinal}"


def linkbreite(bdf: str) -> Optional[tuple[int, str]]:
    """``(Breite, Geschwindigkeit)`` aus sysfs -- NUR als Startschaetzung.

    Der Wert geht in KEINE Entscheidung ein. Er steht in der Erklaerung
    neben der Messung, damit auffaellt, wenn beide auseinanderlaufen: sagt
    sysfs x8 und die Messung 6 GB/s, ist entweder die Karte
    heruntergetaktet oder der Messaufbau kaputt -- und genau diese Frage
    hat in diesem Vorhaben mehrfach den Ausschlag gegeben.
    """
    try:
        basis = f"/sys/bus/pci/devices/{bdf}"
        with open(f"{basis}/current_link_width") as f:
            breite = int(f.read().strip())
        with open(f"{basis}/current_link_speed") as f:
            tempo = f.read().strip()
        return breite, tempo
    except (OSError, ValueError):
        return None
