"""#134 zweite Haelfte: die Experten-BYTES unter ihren Band-Tags.

Der Befund, der diese Datei erzwungen hat, steht im Metall-Log fnFL2w62:

    PDFLIP-XCHG-COVER rank=0 tag=weights_1 planned_mib=226.4 ... (DREI Layer)

Die Experten EINES Layers sind ein Vielfaches davon. Der Austausch baut sein
Inventar aus `named_parameters` / `named_buffers` / `vars(module)`; der
Presplit ersetzt den Experten-Parameter durch einen 0-Zeilen-Platzhalter und
legt die Bytes in ein DICT, das `walk_live_tensors` laut eigenem Docstring
nicht findet. Der Flip transportierte Dense und Attention -- und NULL
Experten-Bytes.
"""
import torch

import pytest

from flliper.srt.layers.moe.expert_offload import (
    expert_band_slot_ranges,
    publish_expert_bands,
)
from flliper.srt.managers import pdflip_memory_saver as m
from flliper.srt.pdflip.weight_exchange import tag_of_parameter_name, walk_live_tensors


@pytest.fixture
def band_env(monkeypatch):
    monkeypatch.setenv(m.WEIGHT_CHUNK_ENV_LAYERS, "3")
    monkeypatch.setenv(m.WEIGHT_CHUNK_ENV_COUNT, "16")
    monkeypatch.setenv(m.EXPERT_BAND_ENV_SIZE, "16")
    monkeypatch.setenv(m.EXPERT_BAND_ENV_COUNT, "32")


def test_without_env_nothing_is_published(monkeypatch):
    monkeypatch.delenv(m.EXPERT_BAND_ENV_SIZE, raising=False)
    monkeypatch.delenv(m.EXPERT_BAND_ENV_COUNT, raising=False)
    assert expert_band_slot_ranges(list(range(64))) == []
    layer = torch.nn.Module()
    assert publish_expert_bands(layer, "w13_weight_packed", torch.zeros(64, 4), range(64)) == 0
    assert not [k for k in vars(layer) if "eband" in k]


def test_ranges_are_consecutive_and_complete(band_env):
    # 512 Experten, jeder vierte resident -> 128 Slots, 32 Baender a 16 Ids
    ids = list(range(0, 512, 4))
    ranges = expert_band_slot_ranges(ids)
    assert len(ranges) == 32, "each band holds 4 of the 16 Ids here"
    for band, s0, s1 in ranges:
        assert s1 > s0
        for slot in range(s0, s1):
            assert band * 16 <= ids[slot] < (band + 1) * 16, (band, slot, ids[slot])
    # Jeder Slot gehoert GENAU einem Band -- keine Luecke, keine Ueberlappung.
    occupied_rows = [s for _, s0, s1 in ranges for s in range(s0, s1)]
    assert occupied_rows == list(range(len(ids)))


def test_empty_bands_are_dropped(band_env):
    # Nur die ersten 20 Ids resident: Baender 2..31 halten nichts.
    ranges = expert_band_slot_ranges(list(range(20)))
    assert [b for b, _, _ in ranges] == [0, 1]
    # Ein leeres Band waere ein 0-Byte-Tensor, den build_plan als Parameter
    # ohne Deskriptor sieht -> W74 PdFlipXchgSourceMissing.
    assert all(s1 > s0 for _, s0, s1 in ranges)


def test_unsorted_is_refused_not_guessed(band_env):
    with pytest.raises(ValueError, match="nicht sortiert"):
        expert_band_slot_ranges([5, 3, 9])


def test_bands_are_views_on_same_memory(band_env):
    buf = torch.arange(128 * 4, dtype=torch.float32).reshape(128, 4)
    layer = torch.nn.Module()
    n = publish_expert_bands(layer, "w13_weight_packed", buf, list(range(0, 512, 4)))
    assert n == 32
    t0 = getattr(layer, m.expert_band_attr_name(0, "w13_weight_packed"))
    assert t0.data_ptr() == buf.data_ptr(), "a band is a VIEW, no copy"
    assert t0.shape == (4, 4)
    buf[0, 0] = -1.0
    assert t0[0, 0] == -1.0
    # und die Summe der Baender ist genau die Residenz, kein Byte doppelt
    total = sum(
        getattr(layer, m.expert_band_attr_name(b, "w13_weight_packed")).shape[0]
        for b in range(32)
    )
    assert total == 128


def test_walk_sees_bands_with_their_tag(band_env):
    """Die ganze Naht in einem Zug: Publikation -> named_modules -> Tag."""
    experts = torch.nn.Module()
    mlp = torch.nn.Module(); mlp.experts = experts
    layer7 = torch.nn.Module(); layer7.mlp = mlp
    layers = torch.nn.ModuleList([torch.nn.Module() for _ in range(7)] + [layer7])
    model = torch.nn.Module(); model.layers = layers
    root = torch.nn.Module(); root.model = model

    buf = torch.zeros(128, 4)
    publish_expert_bands(experts, "w13_weight_packed", buf, list(range(0, 512, 4)))

    hit = {
        t.name: t.tag for t in walk_live_tensors(root) if "eband" in t.name
    }
    assert len(hit) == 32, f"the walk found {len(hit)} bands"
    name = "model.layers.7.mlp.experts.pdflip_eband3_w13_weight_packed"
    assert name in hit, sorted(hit)[:3]
    # Layer 7 -> Chunk 7//3 = 2; Band 3
    assert hit[name] == "weights_2_e3"
    # und JEDER gefundene Tag ist in der Familie -- sonst W74 im Plan
    family = set(m.weights_family_tags())
    for nm, tag in hit.items():
        assert tag in family, (nm, tag)


def test_without_band_env_same_name_has_chunk_tag(monkeypatch):
    monkeypatch.setenv(m.WEIGHT_CHUNK_ENV_LAYERS, "3")
    monkeypatch.setenv(m.WEIGHT_CHUNK_ENV_COUNT, "16")
    monkeypatch.delenv(m.EXPERT_BAND_ENV_SIZE, raising=False)
    monkeypatch.delenv(m.EXPERT_BAND_ENV_COUNT, raising=False)
    n = "model.layers.7.mlp.experts.pdflip_eband3_w13_weight_packed"
    assert tag_of_parameter_name(n) == "weights_2", (
        "without band splitting a name with marker must deliver the chunk tag -- "
        "sonst traegt er einen Tag, den die Familie nicht aufzaehlt"
    )


def test_plan_yields_ranges_that_hold(band_env):
    """Ueber den echten Plan, nicht ueber eine handgemachte Liste.

    Mit einem gepinnten Experten war `resident_ids` frueher `pinned + rest`,
    also NICHT sortiert -- und ein Experten-Band waere dann kein konsekutiver
    Slot-Bereich gewesen. Der Test faehrt den Weg, den der Presplit faehrt.
    """
    from flliper.srt.layers.moe.expert_offload import plan_load_time_staging

    E = 512
    plan = plan_load_time_staging(E, fraction=0.25, pinned_experts=(E - 1, 300))
    assert plan is not None
    # Die Refusal wuerde hier feuern, waere der Plan unsortiert.
    ranges = expert_band_slot_ranges(plan.resident_ids)
    ids = list(plan.resident_ids)
    assert ranges, "not a single band -- is the residency empty?"
    for band, s0, s1 in ranges:
        for slot in range(s0, s1):
            assert band * 16 <= ids[slot] < (band + 1) * 16, (band, ids[slot])
    occupied_rows = [s for _, s0, s1 in ranges for s in range(s0, s1)]
    assert occupied_rows == list(range(len(ids))), "each resident slot belongs to exactly one band"
    assert E - 1 in ids and 300 in ids, "the pinned ones stay resident"
