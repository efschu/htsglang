"""#95: die Slot-Zahl folgt der GLOBALEN Residenz, nicht der des eigenen Rangs.

fnFL2w24, gemessen: P (tp1) schrieb 324 Slots, D (tp3) wollte 512 --
und hermetisch nachgestellt rechnet jeder D-Rang eine EIGENE Zahl:

    TP0 hot=91  -> 421 Slots
    TP1 hot=45  -> 467
    TP2 hot=49  -> 463

`#91/3` nahm `_hot_ids` als die ganze residente Menge. Unter tp1 stimmt
das (ein Rang haelt alles), unter tp3 kennt jeder Rang nur seinen
Bereich. Die kalte Menge des Stores ist aber global: 512 minus ALLE 188.

Der Launcher ist die einzige Stelle, die beide Gruppen kennt -- er gibt
die globale Menge vor, genau wie bei der Store-Geometrie (#91/1).
"""

import json
import os

import pytest

from sglang.srt.layers.moe import expert_store as es


@pytest.fixture(autouse=True)
def _saubere_env(tmp_path):
    alt = os.environ.pop(es.RESIDENT_IDS_ENV, None)
    yield tmp_path
    if alt is not None:
        os.environ[es.RESIDENT_IDS_ENV] = alt
    else:
        os.environ.pop(es.RESIDENT_IDS_ENV, None)


def test_ohne_env_bleibt_alles_wie_bisher():
    assert es.shared_resident_ids() is None


def test_komma_liste_wird_gelesen():
    os.environ[es.RESIDENT_IDS_ENV] = "3,1,2,1"
    assert es.shared_resident_ids() == frozenset({1, 2, 3})


def test_datei_wird_gelesen(_saubere_env):
    p = _saubere_env / "res.json"
    p.write_text(json.dumps(sorted({0, 5, 9})))
    os.environ[es.RESIDENT_IDS_ENV] = str(p)
    assert es.shared_resident_ids() == frozenset({0, 5, 9})


def test_beide_gruppen_kommen_auf_DIESELBE_slotzahl(_saubere_env):
    """Der Kern: 188 global resident -> 324 Plaetze, egal wer fragt."""
    p = _saubere_env / "res.json"
    global_res = sorted(set(range(0, 92)) | set(range(183, 229)) | set(range(320, 370)))
    assert len(global_res) == 188
    p.write_text(json.dumps(global_res))
    os.environ[es.RESIDENT_IDS_ENV] = str(p)
    ids = es.shared_resident_ids()
    assert len(ids) == 188
    assert 512 - len(ids) == 324, "das ist die Zahl, die P am Metall schrieb"


@pytest.mark.parametrize("roh", ["", "abc", "1,,2", "-4", "[1,2", "1.5"])
def test_krumme_angaben_geben_none_statt_halber_menge(roh, _saubere_env):
    """Eine falsche globale Menge waere eine falsche Slot-Zuordnung -- und die
    ist Datenverlust, nicht Speicherverlust."""
    os.environ[es.RESIDENT_IDS_ENV] = roh
    assert es.shared_resident_ids() is None


def test_leere_menge_gilt_nicht_als_angabe(_saubere_env):
    p = _saubere_env / "leer.json"
    p.write_text("[]")
    os.environ[es.RESIDENT_IDS_ENV] = str(p)
    assert es.shared_resident_ids() is None
