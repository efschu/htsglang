"""#90: Gruppe P (tp1/pp3) bekommt KEIN ``--rank-moe-ratio`` -- der Server
verweigert die Flag ohne ``--rank-tp-ratio``. Ohne Vektor fiel der
Slot-Pool (#72) bei genau der Gruppe durch, die den Store FUELLT.

Gemessen fnFL2w16: die Store-Datei blieb bei 512 Slots (800,0 MiB) fuer
JEDE Residenz-Fraction (0.30, 0.55, 0.90), weil `_rank_moe_ratio_vector`
None lieferte und der Block in `_expert_store_rows_for` uebersprungen wurde.
"""

from sglang.srt.layers.moe import expert_offload as eo


def test_p_ohne_ratio_bekommt_trivialen_vektor():
    """Ein Rang haelt alle Experten -- das ist keine Annahme, sondern die
    Definition von tp_size==1. VOR dem Fix war das None."""

    class LayerP:
        moe_ratio = None
        num_experts = 512

    assert eo._rank_moe_ratio_vector(LayerP()) == [512]


def test_gesetzter_vektor_bleibt_unveraendert():
    """Gruppe D faehrt --rank-moe-ratio 183,137,168 (fnFA22-Bestform)."""

    class LayerD:
        moe_ratio = [183, 137, 168]
        num_experts = 183

    assert eo._rank_moe_ratio_vector(LayerD()) == [183, 137, 168]


def test_ohne_expertenzahl_wird_nicht_geraten():
    """Kein Vektor UND keine Expertenzahl -> None. Ein Default auf einem
    Rechenpfad waere eine Zahl, die nichts bedeutet und trotzdem zaehlt."""

    class LayerLeer:
        moe_ratio = None

    assert eo._rank_moe_ratio_vector(LayerLeer()) is None
