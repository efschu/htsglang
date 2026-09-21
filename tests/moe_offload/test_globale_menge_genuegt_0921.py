"""#96: steht die globale Residenzmenge, braucht die Slot-Zahl KEINE Vektoren.

fnFL2w26 starb wie w24 -- D wollte 512 Slots, obwohl
SGLANG_MOE_EXPERT_STORE_RESIDENT_IDS gesetzt war und im --env-d ankam.
Die Ursache steht eine Zeile ueber dem #95-Fix:

    if _ratios and _fracs:          # <- der GANZE Slot-Block haengt daran
        ...
        _global_res = _es.shared_resident_ids()

Fehlt einer der beiden Vektoren, ist auch die globale Menge unerreichbar
-- der richtige Wert hinter einer Bedingung, die er selbst ueberfluessig
macht. Zehnte Instanz der Klasse an einem Tag, und diesmal habe ich den
Riegel mit #95 SELBST davorgesetzt.

Die Plaetzezahl folgt aus der globalen Menge allein:
    slots = num_global - |global_res|
"""

import json
import os

import pytest

from sglang.srt.layers.moe import expert_offload as eo
from sglang.srt.layers.moe import expert_store as es


class _Layer:
    num_experts = 512
    layer_id = 0
    _sglang_prefix = "model.layers.0.mlp"

    def __init__(self, rank, lo, n, moe_ratio=None):
        self.moe_tp_rank = rank
        self._expert_shard_generic = True
        self._gguf_expert_range = (lo, lo + n)
        self.num_local_experts = n + 1
        if moe_ratio is not None:
            self.moe_ratio = moe_ratio


class _Plan:
    def __init__(self, spill):
        self.spill_ids = list(spill)


@pytest.fixture
def _welt(tmp_path, monkeypatch):
    res = sorted(set(range(0, 92)) | set(range(183, 229)) | set(range(320, 370)))
    p = tmp_path / "global.json"
    p.write_text(json.dumps(res))
    monkeypatch.setenv(es.RESIDENT_IDS_ENV, str(p))
    monkeypatch.setenv("SGLANG_MOE_EXPERT_STORE_DIR", str(tmp_path))
    monkeypatch.setenv("SGLANG_MOE_EXPERT_STORE_SLOT_FRACTION", "0.82")
    monkeypatch.delenv("SGLANG_MOE_HOTSET_FILE", raising=False)
    return res


def _slots(layer, res):
    spill = [e for e in range(1, layer.num_local_experts)
             if (layer._gguf_expert_range[0] + e - 1) not in set(res)]
    r = eo._expert_store_rows_for(layer, _Plan(spill))
    return None if r is None else r[3]


def test_ohne_moe_ratio_trotzdem_324(_welt, monkeypatch):
    """DER FALL, DEN w27 GEMESSEN HAT -- die #96-Zeile woertlich:

        ratios=None fracs=None global_res=188 | SLOTS=None

    Mein erster Test hier war gruen OHNE Fix, weil `_rank_moe_ratio_vector`
    ohne aktives TP in den tp1-Fallback `[num_experts]` laeuft und damit
    `_ratios` doch setzt. Am Metall ist `_TP.world_size == 3`, der Fallback
    greift nicht, `moe_ratio` fehlt auf dem Layer -> None.
    Der Fallback wird hier ABGESCHALTET, damit der Test den Boot abbildet.
    """
    import sglang.srt.layers.moe.expert_offload as _eo
    monkeypatch.setattr(_eo, "_rank_moe_ratio_vector", lambda layer: None)
    lay = _Layer(1, 192, 137)          # lo=192, wie w27 es gedruckt hat
    assert _slots(lay, _welt) == 324


def test_mit_vektor_dieselbe_zahl(_welt):
    lay = _Layer(1, 183, 137, moe_ratio=[183, 137, 168])
    assert _slots(lay, _welt) == 324


def test_alle_raenge_einig(_welt):
    z = {_slots(_Layer(r, lo, n), _welt)
         for r, (lo, n) in enumerate([(0, 183), (183, 137), (320, 168)])}
    assert z == {324}, f"die Raenge sind uneins: {z}"


def test_ohne_env_bleibt_es_beim_alten_weg(_welt, monkeypatch):
    monkeypatch.delenv(es.RESIDENT_IDS_ENV)
    lay = _Layer(1, 183, 137)
    assert _slots(lay, _welt) != 324, "ohne globale Menge darf nichts geraten werden"
