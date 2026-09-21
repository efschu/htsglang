"""#91 Baustein 2: der Store-Platz wird VERGEBEN, nicht gerechnet.

Nutzer 21.09. 21:02Z: "die hot experts auf den karten wechseln ja, also
muss das ein tausch von slot auf karte zu slot in vram sein".

Die Eigenschaft, auf die es ankommt: die Zahl der Plaetze bleibt die Zahl
der gleichzeitig Kalten -- egal wie oft getauscht wird.
"""

import pytest

from sglang.srt.layers.moe.slot_ledger import (
    SlotExhausted,
    SlotLedger,
    ledger_for_cold_set,
)


def test_plaetze_nur_fuer_die_kalten():
    """Die Nutzer-Order: Plaetze fuer das, was NICHT auf einer Karte liegt."""
    tafel = ledger_for_cold_set(range(154, 512))  # P bei fraction 0.30
    assert tafel.capacity == 358
    assert tafel.free_count == 0
    assert tafel.slot_of(154) == 0 and tafel.slot_of(511) == 357


def test_tausch_haelt_die_platzzahl_konstant():
    """DAS ist der Kern: 1000 Wechsel, und kein Platz kommt dazu."""
    tafel = ledger_for_cold_set(range(100, 200))  # 100 Kalte
    heiss, kalt = 100, 50
    for i in range(1000):
        slot = tafel.swap(heiss, kalt)
        assert tafel.slot_of(kalt) == slot
        assert tafel.slot_of(heiss) is None, "der Heisse liegt jetzt auf der Karte"
        heiss, kalt = kalt, heiss
    assert tafel.capacity == 100
    assert len(tafel.as_map()) == 100


def test_tausch_gibt_genau_den_platz_des_heissen():
    tafel = ledger_for_cold_set([7, 8, 9])
    alt = tafel.slot_of(8)
    assert tafel.swap(8, 42) == alt
    assert tafel.slot_of(42) == alt
    assert tafel.expert_at(alt) == 42
    assert tafel.slot_of(8) is None


def test_tausch_wenn_der_heisse_keinen_platz_hatte():
    """Lag er schon auf der Karte, ist es kein Tausch, sondern eine Vergabe."""
    tafel = SlotLedger(4)
    slot = tafel.swap(99, 5)  # 99 war nie im Store
    assert tafel.slot_of(5) == slot
    assert tafel.free_count == 3


def test_kalter_mit_eigenem_platz_gibt_ihn_zurueck():
    tafel = ledger_for_cold_set([1, 2, 3])
    assert tafel.free_count == 0
    tafel.swap(1, 2)  # 2 hatte schon einen -- der wird frei
    assert tafel.free_count == 1
    assert len(tafel.as_map()) == 2


def test_volle_tafel_verweigert_statt_still_zu_verlieren():
    """Ein Experte ohne Platz waere der stille Verlust seiner Zeile."""
    tafel = ledger_for_cold_set([1, 2])
    with pytest.raises(SlotExhausted):
        tafel.assign(3)


def test_assign_ist_idempotent():
    tafel = SlotLedger(8)
    assert tafel.assign(5) == tafel.assign(5)
    assert tafel.free_count == 7


def test_release_gibt_den_platz_wirklich_frei():
    tafel = ledger_for_cold_set([4, 5])
    slot = tafel.release(4)
    assert slot is not None and tafel.free_count == 1
    assert tafel.assign(6) == slot, "der freie Platz wird wiederverwendet"


def test_release_eines_unbekannten_ist_kein_fehler():
    assert SlotLedger(2).release(77) is None


def test_zwei_prozesse_rechnen_dieselbe_belegung():
    """Ohne Absprache: gleiche Menge, gleiche Tafelgroesse -> gleiche Plaetze.
    Genau das braucht der geteilte Store (#91 Baustein 1)."""
    kalt = [9, 3, 7, 1]
    a = SlotLedger(4).assign_many(kalt)
    b = SlotLedger(4).assign_many(reversed(kalt))
    assert a == b


def test_uebergabe_ueber_den_flip():
    """Die Belegung muss den Wechsel der Gruppe ueberleben (Baustein 3)."""
    p = ledger_for_cold_set([10, 20, 30])
    d = SlotLedger(p.capacity, occupied=p.as_map())
    assert d.as_map() == p.as_map()
    assert d.free_count == 0
    d.swap(20, 40)
    assert d.slot_of(40) == p.slot_of(20)


def test_kapazitaet_muss_positiv_sein():
    with pytest.raises(ValueError):
        SlotLedger(0)


def test_doppelbelegung_wird_verweigert():
    with pytest.raises(ValueError):
        SlotLedger(4, occupied={1: 0, 2: 0})


def test_platz_ausserhalb_der_tafel_wird_verweigert():
    with pytest.raises(ValueError):
        SlotLedger(4, occupied={1: 9})


def test_tausch_mit_sich_selbst_ist_keiner():
    with pytest.raises(ValueError):
        ledger_for_cold_set([1, 2]).swap(1, 1)


def test_tausch_geht_nicht_ueber_die_freiliste():
    """Der Platz wandert DIREKT weiter. Ginge er ueber die Freiliste, bekaeme
    der Kalte den kleinsten freien statt genau des Platzes des Heissen -- und
    ein dritter Aufrufer koennte ihn dazwischen wegnehmen."""
    tafel = SlotLedger(6, occupied={7: 3, 8: 4, 9: 5})
    assert tafel.free_count == 3          # 0, 1, 2 sind frei und KLEINER
    alt = tafel.slot_of(8)
    assert tafel.swap(8, 42) == alt == 4
    assert tafel.slot_of(42) == 4, "ueber die Freiliste waere es Platz 0 geworden"


def test_vergabe_nimmt_den_kleinsten_freien_platz():
    """Eine frisch angelegte Datei wird von vorne gefuellt; die hinteren
    Seiten bleiben unberuehrt und kosten auf tmpfs nichts."""
    tafel = SlotLedger(4)
    tafel.assign(100)
    tafel.assign(200)
    tafel.release(100)
    assert tafel.assign(300) == 0, "Platz 0 wurde frei und ist der kleinste"
