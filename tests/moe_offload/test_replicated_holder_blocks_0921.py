"""#83: REPLIZIERT heisst nicht "jeder Rang haelt ihn".

DREI BOOTS an derselben Stelle: fnFL2w3 (STALL 129,2 s), w7 (125,2 s), w8
(W29 am Wake). P sammelte auf 9 Lanes, D bediente 6; die drei ueberzaehligen
waren c1, c2 und p3.

Die Wurzel steht in `_blocks_of` (weight_exchange.py): der REPLICATED-Zweig
stand VOR dem `dst_widths`-Zweig und gab JEDEM Rang einen vollen Block --
also fuer jeden Rang ein `d_rank`, und daraus (weight_exchange.py:1953
`for d_rank, d_blocks in enumerate(dst_blocks)`) je eine Diagonal-Lane
c{d_rank}. Unter Form A haelt den Tensor nur der Attention-Host; die
Worker-Karten haben auf ihrer Diagonalen nichts abzulegen.

DESHALB BLIEB #80 (418e685ce4) WIRKUNGSLOS: der Fix dort setzt die
Null-Breiten korrekt ins Manifest -- aber diese Funktion kam bei REPLICATED
nie bis zu der Zeile, die sie liest. Ein richtiger Wert, den niemand liest,
sieht im Test aus wie ein Fix.
"""

import pytest

from sglang.srt.weg2 import weight_exchange as wx


class _Layout:
    def __init__(self, n_ranks=3):
        self.n_ranks = n_ranks
        self.tp_size = n_ranks

    def ratios_for(self, family):
        return None


class _Geom:
    def __init__(self, dst_widths=None, units=384):
        self.shard_axis = wx.REPLICATED
        self.content_units = units
        self.dst_widths = dst_widths
        self.name = "model.layers.29.attn_hyper_connection.block_inject_weight.weight"
        self.family = None
        self.blocks = None
        self.shard_total = units
        self.groups = 1
        self.units = units


def _blocks(geom, is_dst=True, n_ranks=3):
    return wx._blocks_of(geom, _Layout(n_ranks), is_dst=is_dst)


def test_a_non_holder_gets_no_block():
    """Der Kern: Breite 0 = kein Block = kein d_rank = keine Lane."""
    out = _blocks(_Geom(dst_widths=(384, 0, 0)))
    assert len(out) == 3
    assert len(out[0]) == 1, "der Attention-Host haelt ihn"
    assert out[1] == [] and out[2] == [], \
        "die Worker-Karten halten ihn NICHT -- sonst entstehen c1 und c2"


def test_a_holder_carries_the_FULL_form_not_its_width():
    """Bei REPLICATED traegt jeder Halter die ganze Form. Die Breite sagt
    nur, OB er sie traegt -- sie als Praefixsumme zu lesen (wie im
    gesharderten Zweig) wuerde den Tensor zerschneiden."""
    out = _blocks(_Geom(dst_widths=(384, 0, 0), units=384))
    assert out[0][0].size == 384
    assert out[0][0].global_start == 0


def test_two_holders_both_carry_everything():
    out = _blocks(_Geom(dst_widths=(384, 0, 384)))
    assert [len(b) for b in out] == [1, 0, 1]
    assert out[0][0].size == out[2][0].size == 384
    assert out[0][0].global_start == out[2][0].global_start == 0


def test_without_seeded_widths_nothing_changes():
    """Byte-identisch zu vor #83, damit ein Lauf ohne Halterschaftswissen
    derselbe Lauf bleibt."""
    out = _blocks(_Geom(dst_widths=None))
    assert [len(b) for b in out] == [1, 1, 1]
    assert all(b[0].size == 384 for b in out)


def test_the_source_side_is_untouched():
    """`is_dst=False` liest keine Zielbreiten -- die Quelle haelt, was sie
    haelt, unabhaengig davon, wer sie empfaengt."""
    out = _blocks(_Geom(dst_widths=(384, 0, 0)), is_dst=False)
    assert [len(b) for b in out] == [1, 1, 1]


def test_a_width_vector_of_the_wrong_length_is_refused():
    """Drei Raenge, zwei Breiten: welcher Rang haelt den dritten Eintrag?
    Raten waere wieder eine Lane, die niemand bedient."""
    with pytest.raises(wx.Weg2XchgPlanDisagree) as ei:
        _blocks(_Geom(dst_widths=(384, 0)))
    assert "W68" in str(ei.value)
