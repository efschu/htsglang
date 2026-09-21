"""#91/3: das Hotset ist die Autoritaet ueber die kalte Menge.

fnFL2w19 starb daran, dass die RATIO-RECHNUNG die Slot-Zuordnung machte,
waehrend das Hotset andere Experten resident setzte:
``IndexError: index 451 is out of bounds for dimension 0 with size 451``
(expert_store.py:310). Steht ein Hotset, folgt die Zuordnung aus ihm.
"""

import json
import os

import pytest

from sglang.srt.layers.moe import expert_offload as eo


class Layer:
    def __init__(self, layer_id=0, rank=0):
        self.layer_id = layer_id
        self.moe_tp_rank = rank
        self._sglang_prefix = "model.layers.0.mlp"


@pytest.fixture
def hotset(tmp_path, monkeypatch):
    def schreibe(ids_je_rang, layers=(0,)):
        muster = str(tmp_path / "hs_tp{rank}.json")
        for r, ids in enumerate(ids_je_rang):
            with open(muster.replace("{rank}", str(r)), "w") as f:
                json.dump({str(l): list(ids) for l in layers}, f)
        monkeypatch.setenv("SGLANG_MOE_HOTSET_FILE", muster)
        eo._HOTSET_FILE_CACHE.clear()
        return muster
    return schreibe


def test_ohne_hotset_bleibt_alles_wie_bisher(monkeypatch):
    monkeypatch.delenv("SGLANG_MOE_HOTSET_FILE", raising=False)
    assert eo._hotset_global_ids(Layer(), 512, 0, False) is None


def test_lokale_ids_werden_global(hotset):
    """Rang 1 haelt lokal 0..2; mit lo=183 sind das global 183..185."""
    hotset([[], [0, 1, 2], []])
    ids = eo._hotset_global_ids(Layer(rank=1), 512, 183, False)
    assert ids == {183, 184, 185}


def test_pad_wird_beruecksichtigt(hotset):
    """Beim generischen Shard ist lokal 0 der Null-Pad-Experte (keine Zeile)."""
    hotset([[0, 1, 2]])
    assert eo._hotset_global_ids(Layer(), 512, 10, True) == {10, 11}


def test_ids_ausserhalb_werden_verworfen(hotset):
    hotset([[0, 1, 9999]])
    assert eo._hotset_global_ids(Layer(), 512, 0, False) == {0, 1}


def test_layer_ohne_eintrag_gibt_none(hotset):
    hotset([[1, 2]], layers=(7,))          # nur Layer 7 belegt
    assert eo._hotset_global_ids(Layer(layer_id=0), 512, 0, False) is None


def test_unlesbare_datei_gibt_none_statt_zu_raten(tmp_path, monkeypatch):
    p = tmp_path / "kaputt_tp0.json"
    p.write_text("{kein json")
    monkeypatch.setenv("SGLANG_MOE_HOTSET_FILE", str(tmp_path / "kaputt_tp{rank}.json"))
    eo._HOTSET_FILE_CACHE.clear()
    assert eo._hotset_global_ids(Layer(), 512, 0, False) is None


def test_die_kalte_menge_ist_der_rest(hotset):
    """Der Zweck: 156 resident -> 356 Store-Plaetze statt 512."""
    hotset([list(range(156))])
    ids = eo._hotset_global_ids(Layer(), 512, 0, False)
    assert len(ids) == 156
    assert len([e for e in range(512) if e not in ids]) == 356
