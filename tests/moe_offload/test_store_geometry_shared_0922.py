"""#106: die gemeinsame Store-Geometrie hatte einen LESER OHNE SCHREIBER.

`expert_store.shared_geometry()` existiert seit #91, und sein Docstring
sagt woertlich "Gesetzt vom Launcher ueber ``STORE_GEOMETRY_ENV``". Der
Launcher setzte sie nie -- im ganzen Baum stand kein einziger Schreiber
(`grep -r SGLANG_MOE_EXPERT_STORE_GEOMETRY` fand nur expert_store.py).

Also rechnete jede Ranggruppe ihre eigene Abbildung globale Id -> Slot auf
DERSELBEN Datei. Genau das, was #91s eigener Kommentar als Gefahr
beschreibt ("zwei Zuordnungen auf DERSELBEN Datei, und die zweite schreibt
in die Zeilen der ersten"), war der Normalfall.

Am Metall, fnFL2w42 UND w43, beide beim Laden von D:

    ValueError: shared store /mnt/nf-experts/fnFL2/L0-w13_weight_scale.bin
                has 23091200 bytes, this layout wants 26214400

23091200 / 51200 = 451 Plaetze (P), 26214400 / 51200 = 512 (D). w41 lief
davor nur, weil FR_P zufaellig auf dieselbe Zahl fuehrte -- ein Zufall,
kein Beweis, und er verdeckte den fehlenden Schreiber drei Boots lang.
"""

import pytest

from sglang.srt.weg2 import launcher as L
from sglang.srt.layers.moe import expert_store as es


EXTRA_D = ('--rank-role host,worker,worker --rank-moe-ratio 183,137,168 '
           '--rank-moe-resident-fraction 0.12,0.319,0.284 '
           '--rank-gpu-memory-mib 29900,18500,18500 --page-size 64')


def test_die_vektoren_kommen_aus_extra_d():
    assert L._argv_vector(EXTRA_D, "--rank-moe-ratio") == ["183", "137", "168"]
    assert L._argv_vector(EXTRA_D, "--rank-moe-resident-fraction") == [
        "0.12", "0.319", "0.284"]


def test_gleichheitsform_wird_auch_gelesen():
    assert L._argv_vector("--rank-moe-ratio=1,2,3", "--rank-moe-ratio") == [
        "1", "2", "3"]


@pytest.mark.parametrize("extra", ["", "--page-size 64", "--rank-moe-ratio"])
def test_fehlende_flagge_gibt_none_nicht_leere_liste(extra):
    """Eine leere Liste waere vom Aufrufer nicht von 'fehlt' zu trennen."""
    assert L._argv_vector(extra, "--rank-moe-ratio") is None


def test_der_leser_versteht_was_der_schreiber_schreibt():
    """DIE NAHT, um die es geht: Schreiber und Leser an EINEM Strang.

    Der Test prueft nicht zwei Formate gegeneinander, sondern faehrt das
    Ergebnis des Schreibers durch den echten Leser -- genau die Kopplung,
    die drei Boots lang niemand geprueft hat, weil es den Schreiber nicht
    gab.
    """
    ratios = L._argv_vector(EXTRA_D, "--rank-moe-ratio")
    fracs = L._argv_vector(EXTRA_D, "--rank-moe-resident-fraction")
    published = f"{','.join(ratios)}|{','.join(fracs)}"

    import os
    old = os.environ.get(es.STORE_GEOMETRY_ENV)
    os.environ[es.STORE_GEOMETRY_ENV] = published
    try:
        got = es.shared_geometry()
    finally:
        if old is None:
            os.environ.pop(es.STORE_GEOMETRY_ENV, None)
        else:
            os.environ[es.STORE_GEOMETRY_ENV] = old

    assert got is not None, (
        f"der Leser verwirft, was der Schreiber publiziert: {published!r}")
    got_ratios, got_fracs = got
    assert got_ratios == [183, 137, 168]
    assert got_fracs == [0.12, 0.319, 0.284]


def test_ohne_geometrie_bleibt_der_leser_bei_none():
    """Der konservative Fall (#91): lieber keine als eine halbe Geometrie."""
    import os
    old = os.environ.get(es.STORE_GEOMETRY_ENV)
    os.environ[es.STORE_GEOMETRY_ENV] = ""
    try:
        assert es.shared_geometry() is None
    finally:
        if old is None:
            os.environ.pop(es.STORE_GEOMETRY_ENV, None)
        else:
            os.environ[es.STORE_GEOMETRY_ENV] = old
