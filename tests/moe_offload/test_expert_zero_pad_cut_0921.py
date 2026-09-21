"""#74 (21.09.): der Zero-Pad-Experte ist ein Schnitt, kein Zwist.

fnFL2w1 erreichte P READY 159,0 s, D READY 237,0 s, Front 9,1 s und den
ERSTEN FLIP -- und starb am Wake:

    W68 Weg2XchgPlanDisagree: model.layers.0.mlp.experts.w13_weight_shape:
    the PP side holds (512, 2) and the TP rows hold
    [(61, 2), (227, 2), (227, 2)]      61+227+227 = 515 gegen 512

Beide Seiten sind RICHTIG und beschreiben dasselbe verschieden. Die Regel
steht im Baum selbst (`expert_store.global_rows`):

    ``pad=True`` is the generic expert-dim shard: local 0 is the zero pad
    expert (no row), local i >= 1 is global ``lo + i - 1``. ``pad=False`` is
    an unsharded layer (a PP stage holding every expert).

Gruppe P faehrt den Layer ungeshartet (512, kein Pad), Gruppe D shardet die
Experten-Dimension (60/226/226 aus `--rank-moe-ratio`) und traegt je Rang die
Nullzeile an local 0 -- 515 = 512 + 3 Raenge.
"""

import pytest

from sglang.srt.weg2 import xchg_manifest as xm
from sglang.srt.weg2 import weight_exchange as wx


def _piece(name, rows, cols=2, itemsize=8):
    return xm.ManifestPiece(
        param_name=name, tensor_class="CompressedTensorsWNA16MoE",
        rows_full=int(rows), cols_full=int(cols), itemsize=int(itemsize),
        tag="weights_0", nbytes=int(rows) * int(cols) * int(itemsize),
    )


EXPERT = "model.layers.0.mlp.experts.w13_weight_shape"


def test_the_measured_w1_shapes_classify_as_a_row_cut():
    """Die Zahlen des Boots, die den Flip gefaellt haben."""
    whole = _piece(EXPERT, 512)
    cut = [_piece(EXPERT, 61), _piece(EXPERT, 227), _piece(EXPERT, 227)]
    axis, rows_full, cols_full, widths, pad = xm._axis_of(EXPERT, whole, cut)
    assert axis == wx.ROWS
    assert widths == (61, 227, 227)
    assert pad == 3          # genau eine Nullzeile je Rang
    assert rows_full == 515  # die geshartete Seite, Pad eingerechnet


def test_a_genuine_skew_still_refuses():
    """EIN Rang mit einer Zeile zuviel ist kein Pad, sondern Zwist -- und
    genau die Gefahrenrichtung, vor der die Meldung selbst warnt ('a
    tolerance band here would read a genuine disagreement as padding')."""
    whole = _piece(EXPERT, 512)
    cut = [_piece(EXPERT, 62), _piece(EXPERT, 227), _piece(EXPERT, 227)]  # 516-3=513
    with pytest.raises(wx.Weg2XchgPlanDisagree):
        xm._axis_of(EXPERT, whole, cut)


def test_only_expert_tensors_may_carry_the_pad():
    """Die Konvention gehoert dem Experten-Shard. Ein qkv_proj mit demselben
    Zahlenbild bleibt ein Zwist -- sonst verschluckt der Fix genau den
    Drei-Zeilen-Skew, den der `_padded`-Kommentar als Beinahe-Unfall
    festhaelt."""
    nm = "model.layers.0.self_attn.qkv_proj.weight"
    whole = _piece(nm, 512)
    cut = [_piece(nm, 61), _piece(nm, 227), _piece(nm, 227)]
    with pytest.raises(wx.Weg2XchgPlanDisagree):
        xm._axis_of(nm, whole, cut)


def test_a_rank_of_width_one_is_all_pad_and_refuses():
    """Breite 1 hiesse: nur die Nullzeile, kein echter Experte. Dafuer kann
    dieser Zweig nicht buergen."""
    whole = _piece(EXPERT, 2)
    cut = [_piece(EXPERT, 1), _piece(EXPERT, 2), _piece(EXPERT, 2)]  # 5-3=2
    with pytest.raises(wx.Weg2XchgPlanDisagree):
        xm._axis_of(EXPERT, whole, cut)


def test_the_exact_row_cut_is_untouched():
    """Ein Schnitt OHNE Pad (Summe trifft genau) muss weiter vor der neuen
    Klausel greifen -- sie darf nichts umdeuten, was vorher schon passte."""
    whole = _piece(EXPERT, 512)
    cut = [_piece(EXPERT, 60), _piece(EXPERT, 226), _piece(EXPERT, 226)]
    axis, rows_full, _cols, widths, pad = xm._axis_of(EXPERT, whole, cut)
    assert axis == wx.ROWS and pad == 0 and rows_full == 512
    assert widths == (60, 226, 226)


def test_replica_still_wins_before_everything():
    """Identische Form auf allen Raengen bleibt ein Replikat, kein Pad-Cut."""
    whole = _piece(EXPERT, 512)
    cut = [_piece(EXPERT, 512), _piece(EXPERT, 512), _piece(EXPERT, 512)]
    axis, _r, _c, _w, pad = xm._axis_of(EXPERT, whole, cut)
    assert axis == wx.REPLICATED and pad == 0


def test_the_convention_this_leans_on_still_exists():
    """Der Fix ist nur zulaessig, solange `global_rows` die Nullzeile fuehrt.
    Verschwindet sie, faellt dieser Test -- und die Klausel oben mit ihr."""
    import inspect

    from sglang.srt.layers.moe import expert_store as es

    src = inspect.getsource(es.global_rows)
    assert "local 0 is the zero pad expert" in src
    assert "lo + i - 1" in src or "int(lo) + e - 1" in src
    # und die Semantik selbst, nicht nur ihr Kommentar
    assert es.global_rows([0, 1, 2], lo=10, pad=True) == {1: 10, 2: 11}
    assert es.global_rows([0, 1, 2], lo=10, pad=False) == {0: 10, 1: 11, 2: 12}
