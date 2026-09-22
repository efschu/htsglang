"""W100: ``--extra-*`` darf ein Kartenbudget senken, nie heben.

DER ANLASS (fnFL2w51, 22.09.): im argv von D stand ``--rank-gpu-memory-mib``
ZWEIMAL -- 28240,16672,16672 vom Launcher (Korridor) und 29900,18500,18500
aus ``--extra-d``. argparse nimmt den letzten, also gewann die Handzahl, und
D-TP0 hatte nach dem Zielmodell 1,30 GB frei. Der SOLO-Draft, der danach
geladen wird, starb an ``cu_mem_create CUresult 2``.

Die Tests binden die AUFRUF-KANTE (argv_d/argv_p), nicht nur den Helfer:
ein Test, der nur die Funktion ruft, beweist nie, dass die argv-Bauer sie
rufen (Indikator-Gesetz (b)).
"""
import inspect
import re

import pytest

from sglang.srt.weg2 import launcher as lx


def test_heben_wird_benannt_verweigert():
    with pytest.raises(RuntimeError) as ei:
        lx._refuse_if_extra_raises_budget(
            [28240, 16672, 16672],
            ["--rank-gpu-memory-mib", "29900,18500,18500"],
            "D",
        )
    t = str(ei.value)
    assert "W100" in t
    assert "28240" in t and "29900" in t and "+1660" in t


def test_senken_ist_erlaubt():
    lx._refuse_if_extra_raises_budget(
        [28240, 16672, 16672], ["--rank-gpu-memory-mib", "27000,16000,16000"], "D"
    )


def test_gleich_ist_erlaubt():
    lx._refuse_if_extra_raises_budget(
        [28240, 16672], ["--rank-gpu-memory-mib", "28240,16672"], "D"
    )


def test_gleichheitszeichen_form_wird_auch_gesehen():
    with pytest.raises(RuntimeError, match="W100"):
        lx._refuse_if_extra_raises_budget(
            [100, 100], ["--rank-gpu-memory-mib=200,100"], "P"
        )


def test_ohne_die_flag_passiert_nichts():
    lx._refuse_if_extra_raises_budget([1, 2], ["--irgendwas", "5"], "D")
    lx._refuse_if_extra_raises_budget([1, 2], [], "D")


def test_unlesbarer_wert_ist_kein_freibrief():
    # Er ueberschreibt trotzdem -- Schweigen waere hier das Gefaehrlichste.
    with pytest.raises(RuntimeError, match="W100"):
        lx._refuse_if_extra_raises_budget([1, 2], ["--rank-gpu-memory-mib", "auto"], "D")


def test_andere_laenge_wird_verweigert():
    with pytest.raises(RuntimeError, match="W100"):
        lx._refuse_if_extra_raises_budget(
            [1, 2, 3], ["--rank-gpu-memory-mib", "1,2"], "D"
        )


@pytest.mark.parametrize("fn", ["argv_d", "argv_p"])
def test_die_argv_bauer_rufen_den_riegel(fn):
    src = inspect.getsource(getattr(lx, fn))
    assert "_refuse_if_extra_raises_budget(budgets" in src, (
        f"{fn} baut argv, ohne den Riegel zu rufen -- dann gewinnt die "
        f"extra-Flag wieder still."
    )
    # Und zwar VOR dem return, sonst ist er unerreichbar.
    i_riegel = src.index("_refuse_if_extra_raises_budget")
    i_return = src.index("return ", src.index("List[str]:"))
    assert i_riegel < i_return
