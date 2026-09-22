"""#134 zweite Haelfte: die Experten-BYTES unter ihren Band-Tags.

Der Befund, der diese Datei erzwungen hat, steht im Metall-Log fnFL2w62:

    WEG2-XCHG-COVER rank=0 tag=weights_1 planned_mib=226.4 ... (DREI Layer)

Die Experten EINES Layers sind ein Vielfaches davon. Der Austausch baut sein
Inventar aus `named_parameters` / `named_buffers` / `vars(module)`; der
Presplit ersetzt den Experten-Parameter durch einen 0-Zeilen-Platzhalter und
legt die Bytes in ein DICT, das `walk_live_tensors` laut eigenem Docstring
nicht findet. Der Flip transportierte Dense und Attention -- und NULL
Experten-Bytes.
"""
import torch

import pytest

from sglang.srt.layers.moe.expert_offload import (
    expert_band_slot_ranges,
    publish_expert_bands,
)
from sglang.srt.managers import weg2_memory_saver as m
from sglang.srt.weg2.weight_exchange import tag_of_parameter_name, walk_live_tensors


@pytest.fixture
def band_env(monkeypatch):
    monkeypatch.setenv(m.WEIGHT_CHUNK_ENV_LAYERS, "3")
    monkeypatch.setenv(m.WEIGHT_CHUNK_ENV_COUNT, "16")
    monkeypatch.setenv(m.EXPERT_BAND_ENV_SIZE, "16")
    monkeypatch.setenv(m.EXPERT_BAND_ENV_COUNT, "32")


def test_ohne_env_wird_nichts_veroeffentlicht(monkeypatch):
    monkeypatch.delenv(m.EXPERT_BAND_ENV_SIZE, raising=False)
    monkeypatch.delenv(m.EXPERT_BAND_ENV_COUNT, raising=False)
    assert expert_band_slot_ranges(list(range(64))) == []
    layer = torch.nn.Module()
    assert publish_expert_bands(layer, "w13_weight_packed", torch.zeros(64, 4), range(64)) == 0
    assert not [k for k in vars(layer) if "eband" in k]


def test_bereiche_sind_konsekutiv_und_vollstaendig(band_env):
    # 512 Experten, jeder vierte resident -> 128 Slots, 32 Baender a 16 Ids
    ids = list(range(0, 512, 4))
    bereiche = expert_band_slot_ranges(ids)
    assert len(bereiche) == 32, "jedes Band haelt hier 4 der 16 Ids"
    for band, s0, s1 in bereiche:
        assert s1 > s0
        for slot in range(s0, s1):
            assert band * 16 <= ids[slot] < (band + 1) * 16, (band, slot, ids[slot])
    # Jeder Slot gehoert GENAU einem Band -- keine Luecke, keine Ueberlappung.
    belegt = [s for _, s0, s1 in bereiche for s in range(s0, s1)]
    assert belegt == list(range(len(ids)))


def test_leere_baender_fallen_raus(band_env):
    # Nur die ersten 20 Ids resident: Baender 2..31 halten nichts.
    bereiche = expert_band_slot_ranges(list(range(20)))
    assert [b for b, _, _ in bereiche] == [0, 1]
    # Ein leeres Band waere ein 0-Byte-Tensor, den build_plan als Parameter
    # ohne Deskriptor sieht -> W74 Weg2XchgSourceMissing.
    assert all(s1 > s0 for _, s0, s1 in bereiche)


def test_unsortiert_wird_verweigert_nicht_geraten(band_env):
    with pytest.raises(ValueError, match="nicht sortiert"):
        expert_band_slot_ranges([5, 3, 9])


def test_baender_sind_VIEWS_auf_denselben_speicher(band_env):
    buf = torch.arange(128 * 4, dtype=torch.float32).reshape(128, 4)
    layer = torch.nn.Module()
    n = publish_expert_bands(layer, "w13_weight_packed", buf, list(range(0, 512, 4)))
    assert n == 32
    t0 = getattr(layer, m.expert_band_attr_name(0, "w13_weight_packed"))
    assert t0.data_ptr() == buf.data_ptr(), "ein Band ist ein VIEW, keine Kopie"
    assert t0.shape == (4, 4)
    buf[0, 0] = -1.0
    assert t0[0, 0] == -1.0
    # und die Summe der Baender ist genau die Residenz, kein Byte doppelt
    gesamt = sum(
        getattr(layer, m.expert_band_attr_name(b, "w13_weight_packed")).shape[0]
        for b in range(32)
    )
    assert gesamt == 128


def test_der_walk_sieht_die_baender_mit_ihrem_tag(band_env):
    """Die ganze Naht in einem Zug: Publikation -> named_modules -> Tag."""
    experts = torch.nn.Module()
    mlp = torch.nn.Module(); mlp.experts = experts
    layer7 = torch.nn.Module(); layer7.mlp = mlp
    layers = torch.nn.ModuleList([torch.nn.Module() for _ in range(7)] + [layer7])
    model = torch.nn.Module(); model.layers = layers
    root = torch.nn.Module(); root.model = model

    buf = torch.zeros(128, 4)
    publish_expert_bands(experts, "w13_weight_packed", buf, list(range(0, 512, 4)))

    gefunden = {
        t.name: t.tag for t in walk_live_tensors(root) if "eband" in t.name
    }
    assert len(gefunden) == 32, f"der Walk fand {len(gefunden)} Baender"
    name = "model.layers.7.mlp.experts.weg2_eband3_w13_weight_packed"
    assert name in gefunden, sorted(gefunden)[:3]
    # Layer 7 -> Chunk 7//3 = 2; Band 3
    assert gefunden[name] == "weights_2_e3"
    # und JEDER gefundene Tag ist in der Familie -- sonst W74 im Plan
    familie = set(m.weights_family_tags())
    for nm, tag in gefunden.items():
        assert tag in familie, (nm, tag)


def test_ohne_band_env_traegt_derselbe_name_den_chunk_tag(monkeypatch):
    monkeypatch.setenv(m.WEIGHT_CHUNK_ENV_LAYERS, "3")
    monkeypatch.setenv(m.WEIGHT_CHUNK_ENV_COUNT, "16")
    monkeypatch.delenv(m.EXPERT_BAND_ENV_SIZE, raising=False)
    monkeypatch.delenv(m.EXPERT_BAND_ENV_COUNT, raising=False)
    n = "model.layers.7.mlp.experts.weg2_eband3_w13_weight_packed"
    assert tag_of_parameter_name(n) == "weights_2", (
        "ohne Bandteilung muss ein Name mit Marker den Chunk-Tag liefern -- "
        "sonst traegt er einen Tag, den die Familie nicht aufzaehlt"
    )


def test_der_PLAN_liefert_bereiche_die_haelt(band_env):
    """Ueber den echten Plan, nicht ueber eine handgemachte Liste.

    Mit einem gepinnten Experten war `resident_ids` frueher `pinned + rest`,
    also NICHT sortiert -- und ein Experten-Band waere dann kein konsekutiver
    Slot-Bereich gewesen. Der Test faehrt den Weg, den der Presplit faehrt.
    """
    from sglang.srt.layers.moe.expert_offload import plan_load_time_staging

    E = 512
    plan = plan_load_time_staging(E, fraction=0.25, pinned_experts=(E - 1, 300))
    assert plan is not None
    # Die Refusal wuerde hier feuern, waere der Plan unsortiert.
    bereiche = expert_band_slot_ranges(plan.resident_ids)
    ids = list(plan.resident_ids)
    assert bereiche, "kein einziges Band -- die Residenz ist leer?"
    for band, s0, s1 in bereiche:
        for slot in range(s0, s1):
            assert band * 16 <= ids[slot] < (band + 1) * 16, (band, ids[slot])
    belegt = [s for _, s0, s1 in bereiche for s in range(s0, s1)]
    assert belegt == list(range(len(ids))), "jeder residente Slot gehoert genau einem Band"
    assert E - 1 in ids and 300 in ids, "die gepinnten bleiben resident"
