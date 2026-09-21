"""#72 (Nutzer-Order 21.09.): der Host-Store ist ein SLOT-POOL, keine
Zweitkopie des Modells.

    "auf den karten liegt IMMER ein teil des modells, der rest liegt im
     systemram. aber waehrend decode oder prefill muss niemals alles im
     systemram liegen."

Es kommt auf die ANZAHL an, nicht auf die Identitaet: liegen zu jedem
Zeitpunkt R von N Experten auf den Karten, braucht der Store nie mehr als
N-R Plaetze -- auch wenn dauernd andere Experten darin stehen.

GEMESSEN heute (fnFL2w5, fnFL2w7): der Store haelt alle 512 Experten, 59 GiB
auf Platte / 61,10 GiB shmem. shmem ist nicht reclaimable und der groesste
Posten gegen die ~93-GiB-Decke (anon 23,40, Datei-Cache ~4,4). Residenz laut
DESIGN_FORM_A_0920.md: 301/512, erreichbar 357/512.

DIESER TEST NAGELT DIE GROESSENRECHNUNG FEST. Die Indirektion global_id ->
slot ist der zweite Teil und noch NICHT gebaut; ohne sie aendert eine
kleinere Datei nichts, weil `write_rows` die globale Id als Zeilenindex
nimmt. Der Test sagt das ausdruecklich, damit die halbe Strecke nicht fuer
die ganze gehalten wird.
"""

import os

from sglang.srt.layers.moe import expert_store as es


def test_default_is_byte_identical_to_before(monkeypatch):
    """Ohne Env bleibt alles wie bisher: ein Platz je Experte."""
    monkeypatch.delenv(es.SLOT_FRACTION_ENV, raising=False)
    assert es.slot_fraction() == 1.0
    assert es.slots_for(512) == 512
    assert es.slots_for(1) == 1


def test_the_measured_residency_gives_the_measured_saving(monkeypatch):
    """301 von 512 resident -> 41 % der Plaetze reichen."""
    monkeypatch.setenv(es.SLOT_FRACTION_ENV, "0.41")
    assert es.slots_for(512) == 210
    # 210/512 von 59 GiB sind rund 24 GiB -- der Posten, an dem w4 starb.
    assert round(210 / 512 * 59) == 24


def test_nonsense_falls_back_to_all_slots(monkeypatch):
    """KONSERVATIV: jeder unlesbare Wert ergibt 1.0. Ein Irrtum in diese
    Richtung kostet Host-RAM; der andere liesse einen Experten ohne Platz."""
    for bad in ("", "quatsch", "0", "-0.5", "1.5", "nan"):
        monkeypatch.setenv(es.SLOT_FRACTION_ENV, bad)
        assert es.slots_for(512) == 512, bad


def test_rounding_never_loses_a_slot(monkeypatch):
    """Aufgerundet: lieber ein Platz zuviel als ein Experte ohne Platz."""
    monkeypatch.setenv(es.SLOT_FRACTION_ENV, "0.5")
    assert es.slots_for(7) == 4        # 3.5 -> 4
    monkeypatch.setenv(es.SLOT_FRACTION_ENV, "0.01")
    assert es.slots_for(3) == 1        # nie 0


def test_open_store_takes_the_slot_count():
    """Der Parameter ist da und wird durchgereicht -- ohne ihn waere die
    Groessenrechnung ein Werkzeug ohne Verbraucher."""
    import inspect

    sig = inspect.signature(es.open_store)
    assert "num_slots" in sig.parameters
    assert sig.parameters["num_slots"].default is None
    src = inspect.getsource(es.open_store)
    assert "slots_for(int(num_experts))" in src


def test_the_missing_half_is_named():
    """OFFEN und hier festgehalten: die Indirektion global_id -> slot. Ohne
    sie nimmt `write_rows` die globale Id als Zeilenindex, und eine kleinere
    Datei wuerde daneben schreiben."""
    import inspect

    src = inspect.getsource(es.write_rows)
    assert "global_rows" in src, (
        "write_rows indiziert noch ueber global_rows -- der Slot-Pool braucht "
        "hier die Indirektion, sonst ist die kleinere Datei ein Fehler")
