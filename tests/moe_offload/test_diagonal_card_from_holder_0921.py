"""#82 (fnFL2w8, 21.09.): die Diagonal-Lane kommt aus der HALTERSCHAFT, nicht
aus mir selbst.

Drei Boots sind an derselben Stelle gestorben (w3 STALL 129,2 s, w7 125,2 s,
w8 W29 am Wake), und die Wurzel waren zwei Defaults, die dieselbe Luecke
verdecken:

    group_descs_by_pair:2064   pair_of(getattr(d,"src_rank",-1),
                                       getattr(d,"dst_rank",-1))
    weight_updater.py:5712     card=int(getattr(group[0],"dst_rank", device))

Fehlt ``dst_rank``, faellt der Deskriptor erst in die Diagonal-Gruppe und
bekommt dann MEINE Karte. GEMESSEN an fnFL2w8: P sammelte auf 9 Lanes, D
bediente 6; die drei ueberzaehligen (c1, c2, p3) waren genau die Diagonalen
der Karten, auf denen die sammelnden PP-Stufen selbst sitzen.

Unter symmetrischen Layouts faellt das nie auf -- da ist die eigene Karte die
Zielkarte. Unter Form A haelt der Attention-Host alles, und die Worker haben
auf ihrer Diagonalen nichts abzulegen.
"""

import pytest

from sglang.srt.managers.scheduler_components.weight_updater import (
    _diagonal_card_of,
)


class _Desc:
    def __init__(self, dst_rank=None, name="model.layers.29.x.weight"):
        if dst_rank is not None:
            self.dst_rank = dst_rank
        self.name = name


def test_the_holder_decides_not_me():
    """Der Tensor liegt auf Karte 0, ich sitze auf Karte 1 -- c0, nicht c1."""
    assert _diagonal_card_of([_Desc(dst_rank=0)], device=1, phase="collect") == 0


def test_a_descriptor_without_a_holder_is_refused(capsys):
    """Der Kern von #82: lieber laut verweigern als still meine Karte nehmen.

    Ein Flip auf der falschen Lane stirbt ohnehin -- nur 120 s spaeter und
    ohne zu sagen, woran.
    """
    with pytest.raises(RuntimeError) as ei:
        _diagonal_card_of([_Desc()], device=1, phase="collect")
    msg = str(ei.value)
    assert "W82" in msg
    assert "dst_rank" in msg
    assert "model.layers.29.x.weight" in msg, "die Verweigerung nennt den Tensor"
    assert "c1/c2/p3" in msg, "und den gemessenen Fall, der sie erzwungen hat"


def test_the_refusal_names_the_card_it_would_have_taken():
    with pytest.raises(RuntimeError) as ei:
        _diagonal_card_of([_Desc()], device=2, phase="deposit")
    assert "(2)" in str(ei.value)
    assert "phase=deposit" in str(ei.value)


def test_an_empty_group_is_refused_too():
    """Kein Deskriptor, keine Halterschaft -- und erst recht kein Grund,
    meine eigene Karte zu raten."""
    with pytest.raises(RuntimeError):
        _diagonal_card_of([], device=0, phase="collect")


def test_card_zero_is_a_card_not_a_falsy_value():
    """Karte 0 ist auf diesem Rig die 5090 und traegt unter Form A ALLES --
    ein `or`-Default haette sie in die eigene Karte umgebogen."""
    assert _diagonal_card_of([_Desc(dst_rank=0)], device=2, phase="c") == 0
