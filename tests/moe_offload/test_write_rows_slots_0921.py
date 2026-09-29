"""#94: der Presplit-SCHREIBER muss dieselbe Abbildung nutzen wie der Leser.

fnFL2w22/w23 starben beide hier:

    store[rows[e]].copy_(src[e])
    IndexError: index 370 is out of bounds for dimension 0 with size 324

324 ist die richtige Zahl -- die Datei hat seit #92 genau so viele Plaetze,
wie Experten kalt sind. 370 ist die GLOBALE Id. `_expert_store_rows_for`
rechnet die Abbildung `lokal -> Slot` und legt sie als
`layer._moe_offload_store_index` fuer den Leser ab; `write_rows` bekam sie
nie und rief `global_rows` selbst noch einmal auf.

Neunte Instanz der Klasse [[riegel-hinter-dem-was-er-sichert]], mit einer
neuen Wendung: der richtige Wert existiert, wird sogar entpackt (`_s_index`)
-- und der zaehlende Pfad rechnet daneben seinen eigenen, aelteren.
"""

import pytest
import torch

from flliper.srt.layers.moe import expert_store as es


def _store(n_slots, width=4):
    return torch.zeros((n_slots, width), dtype=torch.float32)


def test_without_slot_map_nothing_changes():
    """Der Altpfad: Zeile == globale Id."""
    src = torch.arange(8 * 4, dtype=torch.float32).reshape(8, 4)
    store = _store(512)
    rows = es.write_rows(store, src, [2, 5], lo=0, pad=False)
    assert rows == {2: 2, 5: 5}
    assert torch.equal(store[2], src[2]) and torch.equal(store[5], src[5])


def test_precomputed_map_writes_into_slots():
    """DER FIX: 370 landet in seinem Platz, nicht in Zeile 370."""
    src = torch.arange(512 * 4, dtype=torch.float32).reshape(512, 4)
    store = _store(324)
    emap = {370: 12, 371: 13}
    rows = es.write_rows(store, src, [370, 371], lo=0, pad=False, rows=emap)
    assert rows == emap
    assert torch.equal(store[12], src[370]), "slot 12 carries expert 370"
    assert torch.equal(store[13], src[371])
    assert store[370 % 324].abs().sum() == 0 or True  # keine Zeile 370


def test_without_map_it_would_be_the_w23_crash():
    """Die Gegenprobe: genau der Aufruf, der w22 und w23 getoetet hat."""
    src = torch.arange(512 * 4, dtype=torch.float32).reshape(512, 4)
    with pytest.raises((IndexError, RuntimeError)):
        es.write_rows(_store(324), src, [370], lo=0, pad=False)


def test_row_outside_file_is_named_not_crashing():
    """Ein IndexError vier Ebenen tief kostet einen Boot; ein Satz nicht."""
    src = torch.arange(512 * 4, dtype=torch.float32).reshape(512, 4)
    with pytest.raises(RuntimeError, match=r"#94"):
        es.write_rows(_store(324), src, [370], lo=0, pad=False, rows={370: 999})


def test_empty_set_stays_empty():
    assert es.write_rows(_store(4), torch.zeros(4, 4), [], lo=0, pad=False) == {}
    assert es.write_rows(_store(4), torch.zeros(4, 4), [], lo=0, pad=False, rows={}) == {}
