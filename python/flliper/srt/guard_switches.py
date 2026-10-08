"""WAECHTER-SCHALTER 06.10. (27B-Sitz, Nutzerauftrag 06.10. ~10:20Z): Lesehilfen fuer die Schwellen und Modus-Woerter
der Waechter im Servercode.

Regeln, in einem Ort statt in jedem Waechter nachgebaut:

* Eine Schwelle ist eine Zahl; ``<= 0`` (auch das Ungesetzt-Sentinel -1 der EnvFloat-Felder), ein nicht lesbarer Wert
  oder ein nicht endlicher Wert = die BISHERIGE Konstante (der Default des Aufrufers). Ein Waechter, der bei
  Schwelle 0 bei JEDEM Poll anschlaegt, ist keine "Konfiguration", sondern ein Tippfehler (dieselbe Lesart wie
  ``invariant_checker._admission_wedge_recovery_threshold``).
* Ein Modus ist ein Wort aus einer festen Menge; ein unbekanntes Wort ergibt den Default (mit EINER Warnung je
  Feld und Wort) -- ein Tippfehler im Modus darf weder einen Waechter lautlos abschalten noch einen harten Stopp
  lautlos einschalten.
* Nichts hier raised. Ein Waechter, dessen Konfiguration beim Lesen wirft, waere schlimmer als einer ohne Schalter.

Reine Funktionen ueber ``flliper.srt.environ.envs``; der Default-Pfad (alles ungesetzt) liefert exakt die alten
Konstanten.
"""

from __future__ import annotations

import logging
import math
from typing import Iterable

logger = logging.getLogger(__name__)

MODE_OFF = "off"
MODE_LOG = "log"
MODE_ACT = "act"

_warned: set = set()


def positive_float(field, default: float) -> float:
    """``field.get()`` as a float > 0, else ``default``. Never raises."""
    try:
        v = float(field.get())
    except Exception:  # noqa: BLE001 - a bad value reads as unset
        return float(default)
    if not math.isfinite(v) or v <= 0.0:
        return float(default)
    return v


def positive_int(field, default: int, minimum: int = 1) -> int:
    """``field.get()`` as an int >= ``minimum``, else ``default``. Never raises."""
    try:
        v = int(field.get())
    except Exception:  # noqa: BLE001
        return int(default)
    return v if v >= int(minimum) else int(default)


def mode_of(field, allowed: Iterable[str], default: str) -> str:
    """The mode word of ``field`` (stripped, lower-cased), else ``default``.

    A word outside ``allowed`` is logged ONCE (per field and word) and reads as ``default``.
    """
    allowed = tuple(allowed)
    try:
        v = field.get()
        raw = str(v if v is not None else "").strip().lower()
    except Exception:  # noqa: BLE001
        return default
    if not raw:
        return default
    if raw in allowed:
        return raw
    key = (getattr(field, "name", "?"), raw)
    if key not in _warned:
        _warned.add(key)
        logger.warning(
            "%s=%r is not one of %s -- using the default %r", key[0], raw, "/".join(allowed), default
        )
    return default


def flag_on(field, default: bool = True) -> bool:
    """An AN/AUS switch: ``on/1/true/yes`` = True, ``off/0/false/no`` = False (case-insensitive). Empty = ``default``.
    An unknown word reads as ``default`` with ONE warning -- for the host guards (default AN) a typo therefore never
    switches a guard off."""
    word = mode_of(field, ("on", "1", "true", "yes", "off", "0", "false", "no"), "on" if default else "off")
    return word in ("on", "1", "true", "yes")
