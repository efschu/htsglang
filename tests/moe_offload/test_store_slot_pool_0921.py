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


def test_only_cold_rows_reach_the_host():
    """#72-Klaerung (21.09.): an der Schreibstelle standen ZWEI Regeln
    uebereinander -- "writes every row it loaded, residents included" und
    "Only the COLD rows go to the host". `build_plan` entscheidet es:

        spill_ids = [e for e in range(E) if e not in resident_set]

    Nur die Nicht-Residenten. Der veraltete Satz ist entfernt; dieser Test
    haelt die Entscheidung fest, damit sie nicht zurueckdriftet."""
    import inspect

    from sglang.srt.layers.moe import expert_offload as eo

    src = inspect.getsource(eo)
    assert "spill_ids = [e for e in range(E) if e not in resident_set]" in src
    assert "residents\n            # included" not in src, (
        "der veraltete Satz ist wieder da")


def test_the_store_is_big_because_of_reserved_slots_not_content():
    """Und die Folgerung, die daraus faellt: der Store ist nicht gross, weil
    Residente hineingeschrieben wuerden -- sondern weil `open_store` die
    Datei ueber ALLE globalen Ids anlegt. Genau da setzt der Slot-Pool an."""
    import inspect

    from sglang.srt.layers.moe import expert_store as es

    src = inspect.getsource(es.open_store)
    assert "slots_for(int(num_experts))" in src


def test_slot_rows_packs_the_cold_experts_without_gaps():
    """Die Indirektion: 8 Experten, 3 resident -> 5 Plaetze, und die kalten
    Ids liegen lueckenlos auf 0..4."""
    res, N = {1, 4, 6}, 8
    got = es.slot_rows([0, 2, 3, 5, 7], lo=0, pad=False,
                       resident_ids=res, num_experts=N)
    assert got == {0: 0, 2: 1, 3: 2, 5: 3, 7: 4}
    assert es.slots_for(N, len(got) / N) == 5


def test_every_rank_computes_the_same_slot_without_talking():
    """Der Kern: der Store ist GETEILT. Zwei Raenge mit disjunkten lokalen
    Ids muessen fuer dieselbe globale Id denselben Platz errechnen -- sonst
    braeuchte es einen Konsens ueber Prozessgrenzen."""
    res, N = {1, 4, 6}, 8
    a = es.slot_rows([0, 2], lo=0, pad=False, resident_ids=res, num_experts=N)
    b = es.slot_rows([2, 3], lo=0, pad=False, resident_ids=res, num_experts=N)
    assert a[2] == b[2] == 1


def test_a_resident_gets_no_slot():
    """Residente liegen auf der Karte und brauchen keinen Host-Platz. Kaeme
    einer doch, ist Weglassen der sichere Ausgang -- eine fehlende Zeile
    holt der Leser von der Karte, eine ueberschriebene ist Datenverlust."""
    assert es.slot_rows([1], lo=0, pad=False,
                        resident_ids={1, 4, 6}, num_experts=8) == {}


def test_without_residency_knowledge_nothing_changes():
    """Byte-identisch zu heute, wenn der Aufrufer keine Residenz kennt."""
    plain = es.global_rows([1, 2, 3], lo=10, pad=True)
    assert es.slot_rows([1, 2, 3], lo=10, pad=True) == plain


def test_the_pad_convention_survives():
    """Die Nullzeile des Experten-Shards (#74) bleibt ausgenommen, auch im
    Slot-Pool -- sie hat in keiner der beiden Welten eine Zeile."""
    got = es.slot_rows([0, 1, 2], lo=0, pad=True,
                       resident_ids=set(), num_experts=4)
    assert 0 not in got and got == {1: 0, 2: 1}
