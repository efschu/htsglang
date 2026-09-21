"""#82 (fnFL2w8, 21.09.): die Diagonal-Lane kommt aus der HALTERSCHAFT, nicht
aus mir selbst.

Drei Boots sind an dieser Zeile gestorben (w3, w7, w8):

    card=int(getattr(group[0], "dst_rank", device))

Der Default war nicht der Ausnahmefall, sondern der stille Normalfall. Eine
Stufe vorher liest `group_descs_by_pair` denselben Namen mit demselben
Default (`pair_of(-1, -1)`, weight_exchange_bounce.py:2064) -- ein Deskriptor
ohne `dst_rank` landet also ZUERST in der Diagonal-Gruppe und bekommt DANN
meine eigene Karte. Zwei Defaults, eine Luecke, und zusammen ergeben sie eine
Lane, auf der niemand deponiert.

GEMESSEN an fnFL2w8:
    D deponierte auf: c0(25) p0(12) p1(8) p2(8) p4(8) p5(4)  = 6 Lanes
    P sammelte auf:   c0 c1 c2 p0 p1 p2 p3 p4 p5             = 9 Lanes
    P scheiterte an:  c1/weights_9, c2/weights_14, p3/weights_14
Der Tensor `layers.29.attn_hyper_connection.block_inject_weight.weight` liegt
auf BEIDEN Seiten auf rank 0 (der 5090); P wartete trotzdem auf c1 -- der
Diagonalen der Karte, auf der die sammelnde PP-Stufe SELBST sitzt.

Unter symmetrischen Layouts faellt das nie auf: da IST die eigene Karte die
Zielkarte. Unter Form A haelt der Attention-Host alles, und die Worker-Karten
haben auf ihrer Diagonalen nichts abzulegen.
"""

import pytest

from sglang.srt.managers.scheduler_components.weight_updater import (
    _diagonal_card_of,
)


class _Desc:
    def __init__(self, **kw):
        for k, v in kw.items():
            setattr(self, k, v)


def test_the_holder_decides_not_me():
    """Der Tensor liegt auf Karte 0, ich sitze auf Karte 1 -> c0."""
    group = [_Desc(dst_rank=0, name="layers.29.attn_hyper_connection")]
    assert _diagonal_card_of(group, device=1, phase="collect") == 0


def test_a_missing_dst_rank_refuses_instead_of_taking_mine():
    """Der ganze Punkt von #82: lieber eine benannte Verweigerung als eine
    Lane, auf der 120 s lang niemand deponiert."""
    group = [_Desc(name="layers.42.mlp.experts.w13_weight_shape")]
    with pytest.raises(RuntimeError) as exc:
        _diagonal_card_of(group, device=2, phase="collect")
    msg = str(exc.value)
    assert "W82" in msg
    assert "w13_weight_shape" in msg, "die Verweigerung muss den Tensor nennen"
    assert "2" in msg, "und die Karte, die sie genommen haette"


def test_the_refusal_says_the_phase():
    """Deposit und Collect scheitern an verschiedenen Enden derselben Lane --
    ohne die Phase weiss ein Leser nicht, welche Seite ihn ruft."""
    with pytest.raises(RuntimeError) as exc:
        _diagonal_card_of([_Desc(name="x")], device=0, phase="deposit")
    assert "phase=deposit" in str(exc.value)


def test_an_empty_group_refuses_too():
    """Eine Lane ohne Deskriptoren hat keine Halterschaft -- und `group[0]`
    waere ein IndexError statt einer Aussage."""
    with pytest.raises(RuntimeError):
        _diagonal_card_of([], device=0, phase="collect")


def test_dst_rank_zero_is_a_card_not_a_falsy_value():
    """Karte 0 ist auf diesem Rig die 5090 und traegt unter Form A ALLES --
    ein `or`-Default haette sie in die eigene Karte verwandelt."""
    assert _diagonal_card_of([_Desc(dst_rank=0)], device=2, phase="c") == 0
