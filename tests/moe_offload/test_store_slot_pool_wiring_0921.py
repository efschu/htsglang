"""#72, die Verdrahtung: der Store haelt Plaetze fuer die KALTEN, nicht fuer
alle Experten -- Nutzer-Order 21.09.: "auf den karten liegt IMMER ein teil des
modells... waehrend decode oder prefill muss niemals alles im systemram
liegen".

`test_store_slot_pool_0921.py` nagelt die Rechnung fest (slots_for,
slot_base_for_rank, slot_rows). Dieser Test nagelt die STELLE fest, an der sie
wirkt: `_expert_store_rows_for`. Drei Eigenschaften, jede aus einem Fehler
geboren, den die erste Fassung der Verdrahtung wirklich hatte:

1. DIE DATEIGROESSE IST AUF JEDEM RANG DIESELBE. Der Store ist EINE Datei je
   Layer/Attribut; wer sie zuerst oeffnet, legt sie an. `_base + meine` haette
   Rang 0 eine Datei mit 25 Plaetzen anlegen lassen, in die Rang 2 auf Platz
   118 schreibt -- ein Schreiber hinter dem Dateiende.
2. DIE BEREICHE ZWEIER RAENGE UEBERLAPPEN NICHT. Kollision im geteilten Store
   ist Datenverlust, nicht Speicherverlust.
3. MEHR KALTE ALS RESERVIERT -> DER ALTE, VOLLE STORE. Der sichere Ausgang.
"""

import pytest

torch = pytest.importorskip("torch")

from sglang.srt.layers.moe import expert_offload as eo
from sglang.srt.layers.moe import expert_store as es


class _Layer:
    """Das Wenige, das `_expert_store_rows_for` wirklich liest."""

    def __init__(self, rank, lo, span, num_experts=512, ratios=(60, 226, 226)):
        self.layer_id = 0
        self.num_experts = num_experts
        self.num_local_experts = span
        self._expert_shard_generic = True
        self._gguf_expert_range = (lo, lo + span)
        self.moe_tp_rank = rank
        self.moe_ratio = list(ratios)


class _Plan:
    def __init__(self, spill_ids):
        self.spill_ids = list(spill_ids)


RATIOS = (60, 226, 226)
LO = {0: 0, 1: 60, 2: 286}


def _cold_plan(rank, n_cold):
    """`n_cold` kalte Experten dieses Rangs, als LOKALE Ids mit Pad-Offset.

    `pad=True`: lokal 0 ist der Null-Pad-Experte, lokal i>=1 ist global
    ``lo + i - 1``. Die Residenten sind hier die ERSTEN des Bereichs -- welche
    genau, ist fuer die Slot-Rechnung egal, die Ordnung macht sie eindeutig.
    """
    span = RATIOS[rank]
    first_cold = span - n_cold
    return _Plan([i + 1 for i in range(first_cold, span)])


def _rows_for(monkeypatch, tmpdir, rank, n_cold, fraction="0.41",
              resident_vec="0.59,0.59,0.59"):
    monkeypatch.setenv(es.STORE_DIR_ENV, str(tmpdir))
    monkeypatch.setenv(es.SLOT_FRACTION_ENV, fraction)
    monkeypatch.setenv("SGLANG_MOE_RESIDENT_EXPERT_FRACTION", resident_vec)
    layer = _Layer(rank, LO[rank], RATIOS[rank])
    return eo._expert_store_rows_for(layer, _cold_plan(rank, n_cold))


def test_every_rank_sizes_the_file_the_same(monkeypatch, tmp_path):
    """Sonst legt der erste Oeffner sie zu klein an."""
    sizes = {r: _rows_for(monkeypatch, tmp_path, r, 10)[3] for r in (0, 1, 2)}
    assert len(set(sizes.values())) == 1, sizes
    # 60-35 + 226-133 + 226-133 = 25 + 93 + 93 = 211 (gemessen am Rig-Vektor)
    assert sizes[0] == 211


def test_rank_slot_ranges_do_not_overlap(monkeypatch, tmp_path):
    used = {}
    for rank in (0, 1, 2):
        _, _, _, n_slots, index, _ = _rows_for(monkeypatch, tmp_path, rank, 20)
        for slot in index.values():
            assert 0 <= slot < n_slots, (rank, slot, n_slots)
            assert slot not in used, f"rank {rank} collides with rank {used[slot]} on slot {slot}"
            used[slot] = rank
    assert len(used) == 60


def test_slots_are_packed_from_the_rank_base(monkeypatch, tmp_path):
    """Lueckenlos ab der eigenen Basis -- das ist die ganze Ersparnis."""
    _, _, _, _, index, _ = _rows_for(monkeypatch, tmp_path, 2, 5)
    # Basis Rang 2 = 25 + 93 = 118
    assert sorted(index.values()) == [118, 119, 120, 121, 122]


def test_more_cold_than_reserved_falls_back_to_the_full_store(monkeypatch, tmp_path):
    """Der sichere Ausgang: lieber Host-RAM verlieren als Zeilen."""
    # Rang 0 reserviert 25 Plaetze; hier sind 40 Experten kalt.
    _, _, _, n_slots, index, _ = _rows_for(monkeypatch, tmp_path, 0, 40)
    assert n_slots == 512, "die Datei muss wieder einen Platz je Experte haben"
    # und der Index ist wieder die globale Id
    assert sorted(index.values()) == list(range(20, 60))


def test_without_the_env_nothing_changes(monkeypatch, tmp_path):
    """Byte-identisch zu vor #72 -- die Voraussetzung dafuer, dass ein Boot
    ohne die Variable derselbe Boot ist."""
    _, _, _, n_slots, index, _ = _rows_for(monkeypatch, tmp_path, 2, 5,
                                           fraction="")
    assert n_slots == 512
    assert sorted(index.values()) == [507, 508, 509, 510, 511]


def test_a_scalar_resident_fraction_broadcasts(monkeypatch, tmp_path):
    """Ein Skalar gilt fuer JEDEN Rang -- und genau das braucht die Basis.

    Der Test hiess erst "ist nicht genug" und war damit falsch: ein Skalar
    sagt sehr wohl etwas ueber die anderen Raenge, naemlich dieselbe Zahl.
    Entscheidend ist nur, dass die LAENGE zu den Ratios passt -- sonst faellt
    ein Rang auf fraction 0.0 zurueck (sein ganzer Bereich gilt als kalt) und
    die Summe waere 477 statt 211. Das war real: ohne `tp_size` gab
    `resident_fraction_vector` einen Einer-Vektor zurueck, weil die
    TP-Groesse ausserhalb des initialisierten Prozesses unbekannt ist.
    """
    _, _, _, n_slots, index, _ = _rows_for(monkeypatch, tmp_path, 1, 10,
                                           resident_vec="0.59")
    assert n_slots == 211
    # Basis Rang 1 = 25
    assert sorted(index.values()) == list(range(25, 35))


def test_a_short_fraction_vector_does_not_silently_shift_a_rank(monkeypatch, tmp_path):
    """Zwei Eintraege fuer drei Raenge: der dritte haette fraction 0.0, sein
    ganzer Bereich gaelte als kalt, und die Datei waere groesser statt
    kleiner. Der vorgesehene Leser verweigert das (uneindeutige Laenge),
    also bleibt es beim vollen Store -- der sichere Ausgang."""
    _, _, _, n_slots, _, _ = _rows_for(monkeypatch, tmp_path, 1, 10,
                                       resident_vec="0.59,0.59")
    assert n_slots == 512
