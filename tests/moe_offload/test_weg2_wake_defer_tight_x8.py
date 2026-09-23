"""fnFL2x8 (23.09.): der D->P-Wake gab der LETZTEN P-Stufe in Runde 1 einen
Tag; ihr Chunk braucht ~2,7 GB auf einer 3080 mit 1383 MiB frei, die TP-Quelle
gibt dort ~0,6 GB je Tag frei -> Kredit-Warten gegen eine ungeleerte Lane ->
W108 auf PP2, 90 s 'budget expired' auf PP0. Knappe Zielkarten warten hinter
den geraeumigen (SGLANG_WEG2_WAKE_DEFER_BELOW_MIB), knappste zuletzt."""

import pytest

from sglang.srt.weg2 import front as F

TAGS = [f"weights_{i}" for i in range(16)] + ["weights_draft", "weights"]


def _dst_x8():
    dst = {}
    for i in range(16):
        cards = []
        for layer in (3 * i, 3 * i + 1, 3 * i + 2):
            c = 1 if layer < 29 else (0 if layer < 40 else 2)
            if c not in cards:
                cards.append(c)
        dst[f"weights_{i}"] = tuple(cards)
    return dst


FREE_X8 = {0: 1745, 1: 6048, 2: 1383}


def test_ohne_schalter_bleibt_der_round_robin(monkeypatch):
    monkeypatch.delenv("SGLANG_WEG2_WAKE_DEFER_BELOW_MIB", raising=False)
    order, why = F.interleave_pause_order(TAGS, {}, FREE_X8, dst_cards=_dst_x8())
    assert order[:3] == ["weights_0", "weights_10", "weights_14"]
    assert "TIGHT" not in why


def test_knappe_karten_kommen_hinter_die_geraeumigen(monkeypatch):
    monkeypatch.setenv("SGLANG_WEG2_WAKE_DEFER_BELOW_MIB", "3072")
    order, why = F.interleave_pause_order(TAGS, {}, FREE_X8, dst_cards=_dst_x8())
    assert sorted(order) == sorted(TAGS)                    # Permutation
    assert order[-2:] == ["weights_draft", "weights"]       # Basis schliesst
    pos = {t: i for i, t in enumerate(order)}
    # PP2 (Karte 2, knappste) ganz hinten, PP1 (Karte 0) dazwischen
    assert pos["weights_14"] > pos["weights_12"] > pos["weights_9"]
    assert "TIGHT destination cards [0, 2]" in why


def test_mindestens_eine_karte_bleibt_geraeumig(monkeypatch):
    monkeypatch.setenv("SGLANG_WEG2_WAKE_DEFER_BELOW_MIB", "999999")
    roomy, tight = F._split_tight_destination_cards([0, 1, 2], FREE_X8)
    assert roomy == [1] and tight == [0, 2]


@pytest.mark.parametrize("val", ["", "abc", "-5"])
def test_kaputter_schalter_ist_aus(monkeypatch, val):
    monkeypatch.setenv("SGLANG_WEG2_WAKE_DEFER_BELOW_MIB", val)
    roomy, tight = F._split_tight_destination_cards([0, 1, 2], FREE_X8)
    assert tight == [] and roomy == [0, 1, 2]


def test_karte_ohne_messung_wird_nicht_zurueckgestellt(monkeypatch):
    monkeypatch.setenv("SGLANG_WEG2_WAKE_DEFER_BELOW_MIB", "3072")
    roomy, tight = F._split_tight_destination_cards([-1, 2], {2: 100})
    assert roomy == [-1] and tight == [2]
