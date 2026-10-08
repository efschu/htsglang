"""#91 Baustein 2: der Store-Platz wird VERGEBEN, nicht gerechnet.

Nutzer 21.09. 21:02Z: "die hot experts auf den karten wechseln ja, also
muss das ein tausch von slot auf karte zu slot in vram sein".

Die Eigenschaft, auf die es ankommt: die Zahl der Plaetze bleibt die Zahl
der gleichzeitig Kalten -- egal wie oft getauscht wird.
"""

import pytest

from flliper.srt.layers.moe.slot_ledger import (
    SlotExhausted,
    SlotLedger,
    ledger_for_cold_set,
)


def test_slots_only_for_cold_ones():
    """Die Nutzer-Order: Plaetze fuer das, was NICHT auf einer Karte liegt."""
    board = ledger_for_cold_set(range(154, 512))  # P bei fraction 0.30
    assert board.capacity == 358
    assert board.free_count == 0
    assert board.slot_of(154) == 0 and board.slot_of(511) == 357


def test_swap_keeps_slot_count_constant():
    """DAS ist der Kern: 1000 Wechsel, und kein Platz kommt dazu."""
    board = ledger_for_cold_set(range(100, 200))  # 100 Kalte
    hot, cold_ids = 100, 50
    for i in range(1000):
        slot = board.swap(hot, cold_ids)
        assert board.slot_of(cold_ids) == slot
        assert board.slot_of(hot) is None, "der Heisse liegt jetzt auf der Karte"
        hot, cold_ids = cold_ids, hot
    assert board.capacity == 100
    assert len(board.as_map()) == 100


def test_swap_gives_exactly_the_hot_ones_slot():
    board = ledger_for_cold_set([7, 8, 9])
    alt = board.slot_of(8)
    assert board.swap(8, 42) == alt
    assert board.slot_of(42) == alt
    assert board.expert_at(alt) == 42
    assert board.slot_of(8) is None


def test_swap_when_hot_one_had_no_slot():
    """Lag er schon auf der Karte, ist es kein Tausch, sondern eine Vergabe."""
    board = SlotLedger(4)
    slot = board.swap(99, 5)  # 99 war nie im Store
    assert board.slot_of(5) == slot
    assert board.free_count == 3


def test_cold_with_own_slot_returns_it():
    board = ledger_for_cold_set([1, 2, 3])
    assert board.free_count == 0
    board.swap(1, 2)  # 2 hatte schon einen -- der wird frei
    assert board.free_count == 1
    assert len(board.as_map()) == 2


def test_full_table_refuses_instead_of_losing_silently():
    """Ein Experte ohne Platz waere der stille Verlust seiner Zeile."""
    board = ledger_for_cold_set([1, 2])
    with pytest.raises(SlotExhausted):
        board.assign(3)


def test_assign_ist_idempotent():
    board = SlotLedger(8)
    assert board.assign(5) == board.assign(5)
    assert board.free_count == 7


def test_release_really_frees_the_slot():
    board = ledger_for_cold_set([4, 5])
    slot = board.release(4)
    assert slot is not None and board.free_count == 1
    assert board.assign(6) == slot, "der freie Platz wird wiederverwendet"


def test_release_of_unknown_is_not_an_error():
    assert SlotLedger(2).release(77) is None


def test_two_processes_compute_same_occupancy():
    """Ohne Absprache: gleiche Menge, gleiche Tafelgroesse -> gleiche Plaetze.
    Genau das braucht der geteilte Store (#91 Baustein 1)."""
    cold_ids = [9, 3, 7, 1]
    a = SlotLedger(4).assign_many(cold_ids)
    b = SlotLedger(4).assign_many(reversed(cold_ids))
    assert a == b


def test_handover_across_the_flip():
    """Die Belegung muss den Wechsel der Gruppe ueberleben (Baustein 3)."""
    p = ledger_for_cold_set([10, 20, 30])
    d = SlotLedger(p.capacity, occupied=p.as_map())
    assert d.as_map() == p.as_map()
    assert d.free_count == 0
    d.swap(20, 40)
    assert d.slot_of(40) == p.slot_of(20)


def test_capacity_must_be_positive():
    with pytest.raises(ValueError):
        SlotLedger(0)


def test_double_assignment_is_refused():
    with pytest.raises(ValueError):
        SlotLedger(4, occupied={1: 0, 2: 0})


def test_slot_outside_the_table_is_refused():
    with pytest.raises(ValueError):
        SlotLedger(4, occupied={1: 9})


def test_swap_with_itself_is_none():
    with pytest.raises(ValueError):
        ledger_for_cold_set([1, 2]).swap(1, 1)


def test_swap_does_not_use_the_free_list():
    """Der Platz wandert DIREKT weiter. Ginge er ueber die Freiliste, bekaeme
    der Kalte den kleinsten freien statt genau des Platzes des Heissen -- und
    ein dritter Aufrufer koennte ihn dazwischen wegnehmen."""
    board = SlotLedger(6, occupied={7: 3, 8: 4, 9: 5})
    assert board.free_count == 3          # 0, 1, 2 sind frei und KLEINER
    alt = board.slot_of(8)
    assert board.swap(8, 42) == alt == 4
    assert board.slot_of(42) == 4, "ueber die Freiliste waere es Platz 0 geworden"


def test_assign_takes_smallest_free_slot():
    """Eine frisch angelegte Datei wird von vorne gefuellt; die hinteren
    Seiten bleiben unberuehrt und kosten auf tmpfs nichts."""
    board = SlotLedger(4)
    board.assign(100)
    board.assign(200)
    board.release(100)
    assert board.assign(300) == 0, "Platz 0 wurde frei und ist der kleinste"
