"""#131: die EINE PP-Fraktion ist das layer-gewichtete Mittel.

`resident_unsharded` modelliert die PP-Gruppe als EINEN Rang. Die eine
Zahl, die sie bekommt, muss darum die Residenz der GRUPPE sein -- nicht
die der ersten Stufe. fnFL2w62: FR_P 0.367,0.75,0.95 auf Stufen 29,11,8
ergab mit `_fr_p[0]` resident 188 / slots 324 / Store 37 GB, waehrend die
Karten 71/83/78 % trugen und der Host bei 2 GB freiem RAM stand.
"""
import inspect

from sglang.srt.weg2 import launcher as L


def _eff(w, f):
    return sum(a * b for a, b in zip(w, f)) / float(sum(w))


def test_gewichtetes_mittel_statt_erster_stufe():
    w, f = [29, 11, 8], [0.367, 0.75, 0.95]
    eff = _eff(w, f)
    assert abs(eff - 0.5519) < 1e-3, eff
    assert round(512 * eff) == 283          # resident
    assert 512 - round(512 * eff) == 229    # slots, heute 324
    assert eff > f[0], "die erste Stufe allein unterschaetzt die Gruppe"


def test_gleiche_fractions_aendern_nichts():
    """Sind alle Stufen gleich, ist das Mittel genau dieser Wert --
    der Fix darf den alten Fall nicht verschieben."""
    assert abs(_eff([29, 11, 8], [0.367] * 3) - 0.367) < 1e-9


def test_der_launcher_gewichtet_mit_den_pp_stufen():
    src = inspect.getsource(L)
    i = src.index("_pp_frac = ")
    code = "\n".join(z for z in src[i - 600:i + 400].split("\n")
                     if not z.lstrip().startswith("#"))
    assert 'pp_stage_ratio' in code, "gewichtet nicht mit den PP-Stufen"
    assert "_geom_ratios" not in code.split("_pp_frac")[0][-300:], (
        "gewichtet mit D's MoE-Ratios -- falsche Groesse"
    )


def test_fallback_bleibt_die_erste_stufe():
    """Ohne Stufen-Vektor oder bei Laengen-Mismatch faellt es auf das
    alte Verhalten zurueck -- nie auf None, nie auf einen Zufallswert."""
    src = inspect.getsource(L)
    i = src.index("_pp_frac = ")
    assert "float(_fr_p[0]) if _fr_p else None" in src[i - 200:i + 600]
