"""#109: der geteilte Store ist QUELLE, nicht nur Ziel.

Der Anlass ist ein Nutzer-Befund am Disk-I/O (22.09.): "der disk I/O sagt
mir auch eher, dass auch die ganzen MOE experten die bereits im systemram
liegen TROTZDEM nochmal von der platte geladen werden". Der Codegraph gab
ihm recht -- `write_rows` hatte genau einen Aufrufer und kein Gegenstueck.
"""
import inspect

import pytest
import torch

from sglang.srt.layers.moe import expert_offload as eo
from sglang.srt.layers.moe import expert_store as es


def test_fill_rows_holt_die_zeilen():
    store = torch.arange(12, dtype=torch.float32).reshape(4, 3)
    dst = torch.zeros(2, 3)
    g = es.fill_rows(store, dst, {0: 3, 1: 1})
    assert g == {0: 3, 1: 1}
    assert torch.equal(dst[0], store[3])
    assert torch.equal(dst[1], store[1])


def test_eine_zeile_ohne_beleg_wird_NICHT_gelesen():
    # Eine frische Store-Datei ist genullt, und genullte Gewichte sehen aus
    # wie Gewichte. Nur der Sentinel unterscheidet sie.
    store = torch.ones(4, 3)
    dst = torch.zeros(2, 3)
    g = es.fill_rows(store, dst, {0: 3, 1: 1}, valid=[1])
    assert g == {1: 1}
    assert torch.equal(dst[0], torch.zeros(3)), "Zeile 3 hatte keinen Beleg"
    assert torch.equal(dst[1], torch.ones(3))


def test_zeile_ausserhalb_der_datei_wird_benannt():
    with pytest.raises(RuntimeError, match="#109"):
        es.fill_rows(torch.zeros(2, 3), torch.zeros(1, 3), {0: 7})


def test_fill_rows_ist_das_gegenstueck_zu_write_rows():
    # Schreiben und Lesen muessen dieselbe Abbildung meinen, sonst ist der
    # Store nach einem Boot-Paar verschoben (#94 auf der Leseseite).
    store = torch.zeros(5, 3)
    src = torch.arange(6, dtype=torch.float32).reshape(2, 3)
    rows = {0: 4, 1: 2}
    es.write_rows(store, src, [0, 1], 0, True, rows=rows)
    zurueck = torch.zeros(2, 3)
    es.fill_rows(store, zurueck, rows)
    assert torch.equal(zurueck, src)


def test_der_presplit_ruft_den_adoptionszweig():
    src = inspect.getsource(eo.presplit_expert_offload_after_repack)
    assert "_fill_experts_from_store" in src, (
        "der Presplit holt nichts aus dem Store -- dann liest D die Shards "
        "weiter von Platte, wie in w52 gemessen"
    )
    # UND VOR write_rows / vor dem Kopieren in den GPU-Puffer: `t` muss
    # gefuellt sein, BEVOR die residenten Zeilen daraus auf die Karte
    # gehen, sonst traegt der Puffer Zufall.
    i_fill = src.index("_fill_experts_from_store")
    i_write = src.index("_es.write_rows(")
    i_buf = src.rindex("buf[:R].copy_")
    assert i_fill < i_write, "gefuellt wird erst nach dem Schreiben"
    assert i_fill < i_buf, "gefuellt wird erst nach dem Kopieren auf die Karte"


def test_der_adoptionszweig_laeuft_nur_unter_platzhaltern():
    src = inspect.getsource(eo.presplit_expert_offload_after_repack)
    i = src.index("_fill_experts_from_store")
    davor = src[max(0, i - 400):i]
    assert "weights_are_placeholder" in davor, (
        "ohne diesen Riegel wuerde auch ein normal geladener Rang seine "
        "echten Bytes mit Store-Zeilen ueberschreiben"
    )


def test_ohne_karte_meldet_er_alles_als_fehlend(monkeypatch):
    monkeypatch.setattr(es, "expert_map", lambda: None)
    g, f = eo._fill_experts_from_store(
        torch.zeros(2, 3), torch.zeros(4, 3), "/nx", "L0", "w", 0, 4, True
    )
    assert (g, f) == (0, 4), "ohne Karte darf nichts als gefuellt gelten"


# --- DIE REIHENFOLGE, die #108 falsch hatte ------------------------------


def test_der_platzhalter_riegel_steht_VOR_dem_loader():
    """`process_weights_after_loading` laeuft INNERHALB von
    `loader.load_model`. Steht `arm_placeholder` erst bei "Load weight end",
    ist der Zustand waehrend des Presplits False -- dann greift weder der
    Store-Schreibriegel (#108) noch der Store-Lesepfad (#109).
    """
    import inspect

    from sglang.srt.model_executor import model_runner as mr

    src = inspect.getsource(mr.ModelRunner.load_model)
    i_arm = src.index("arm_placeholder")
    i_load = src.index("self.loader.load_model(")
    assert i_arm < i_load, (
        "arm_placeholder steht hinter dem Loader -- der Presplit sieht den "
        "Platzhalter-Zustand dann nicht"
    )
    # und die Bedingung muss das load_format pruefen, nicht nur das Flag
    davor = src[max(0, i_arm - 400):i_arm]
    assert "dummy" in davor and "load_format" in davor
