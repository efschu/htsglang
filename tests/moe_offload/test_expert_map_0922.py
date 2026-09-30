"""Die Experten-Karte (Nutzer-Gesetz: "alles was geshardet wird braucht ne karte").

Die Tests halten die zwei Fallen fest, die am 22.09. Boots gekostet haben,
und die eine Zahl, die am Metall gemessen ist (fnFL2w24/w30: 324 Plaetze).
"""

import pytest

from flliper.srt.layers.moe import expert_map as em


RATIOS = [183, 137, 168]      # summiert 488, NICHT 512
FR_PP = 0.367
FR_TP = [0.479, 0.319, 0.284]
TOTAL = 512


def test_ratios_are_scaled_not_taken_raw():
    """fnFL2w49 starb an genau 5 Ids (183..187).

    Ich hatte die rohen Ratios als Bereichsgrenzen genommen (0..182,
    183..319, 320..487); der Server skaliert auf 512. Die Karte muss die
    Grenzen des SERVERS fuehren, sonst beschreibt sie eine Aufteilung, die
    es nicht gibt.
    """
    assert sum(RATIOS) == 488, "otherwise this test tests nothing"
    assert em.scaled_spans(RATIOS, TOTAL) == [192, 144, 176]
    assert em.bounds(em.scaled_spans(RATIOS, TOTAL)) == [0, 192, 336]
    assert sum(em.scaled_spans(RATIOS, TOTAL)) == TOTAL, (
        "no Id may disappear between two bands")


def test_slot_count_is_the_one_measured_on_metal():
    """fnFL2w24 und w30: 324 Plaetze, 506,25 MiB je Tensor, 36,71 GiB."""
    k = em.build(TOTAL, RATIOS, FR_PP, FR_TP)
    assert k["slots"] == 324


def test_store_holds_the_swap_not_the_union():
    """Nutzer 22.09.: "kein zusaetzlicher systemram dafuer notwendig".

    Die Vereinigung aller je kalten Ids waere 420 (512 minus die 92, die
    beide Phasen teilen) -- 96 Plaetze mehr, also ~11 GiB Host-RAM, die
    der Tausch nicht braucht. Beide Phasen halten 188 resident, also
    haelt der Store zu JEDEM Zeitpunkt 324.
    """
    k = em.build(TOTAL, RATIOS, FR_PP, FR_TP)
    res_p = {g for ids in k["phases"]["P"]["resident"] for g in ids}
    res_d = {g for ids in k["phases"]["D"]["resident"] for g in ids}
    assert len(res_p) == 188 and len(res_d) == 188
    union = TOTAL - len(res_p & res_d)
    assert union == 420, "the number that w49 has occupied"
    assert k["slots"] == 324 < union
    assert k["shared_resident"] == 92, "what stays on the cards"
    assert k["moves"] == 192, "96 out + 96 in, slot against slot"


@pytest.mark.parametrize("phase", ["P", "D"])
def test_each_id_is_resident_or_in_store(phase):
    """Die Bedingung, an der #97 VIERMAL gescheitert ist -- jetzt eine
    Eigenschaft der Karte statt einer Uebereinkunft zweier Rechnungen."""
    k = em.build(TOTAL, RATIOS, FR_PP, FR_TP)
    res = {g for ids in k["phases"][phase]["resident"] for g in ids}
    cold_ids = {int(x) for x in k["phases"][phase]["slot_of"]}
    assert not (res & cold_ids), "no Id may be both"
    assert len(res | cold_ids) == TOTAL, "and none may be missing"


def test_slots_are_gap_and_collision_free():
    k = em.build(TOTAL, RATIOS, FR_PP, FR_TP)
    for phase in ("P", "D"):
        slots = sorted(int(v) for v in k["phases"][phase]["slot_of"].values())
        assert slots == list(range(len(slots))), (
            f"{phase}: Slots must be 0..n-1, gap- and duplicate-free")
        assert len(slots) <= k["slots"]


def test_map_checks_itself():
    k = em.build(TOTAL, RATIOS, FR_PP, FR_TP)
    assert em.refuse_if_inconsistent(k) is None


def test_broken_map_is_named_not_swallowed():
    """Nach dem Umbau kann #97 nur noch aus einer falschen KARTE kommen --
    dann muss sie es sagen, mit Zahl."""
    k = em.build(TOTAL, RATIOS, FR_PP, FR_TP)
    k["phases"]["P"]["slot_of"]["0"] = 0        # Id 0 ist resident UND im Store
    refuse_reason = em.refuse_if_inconsistent(k)
    assert refuse_reason and "resident UND im Store" in refuse_reason


def test_slot_of_gives_none_for_residents():
    k = em.build(TOTAL, RATIOS, FR_PP, FR_TP)
    assert em.slot_of(k, "P", 0) is None, "Id 0 holds P on the card"
    assert em.slot_of(k, "P", 511) is not None, "Id 511 lies in the store"


def test_phase_of_separates_the_groups():
    assert em.phase_of("P") == "P" and em.phase_of("PP") == "P"
    assert em.phase_of("D") == "D" and em.phase_of("TP") == "D"


def test_fraction_zero_holds_nothing_one_holds_all():
    # #160: TOTAL-1, nicht TOTAL. `resident_slot_count` hat ein
    # `max(1, ...)`, und das ist keine Kosmetik -- `plan_load_time_staging`
    # rechnet bei fraction 0.0 wirklich R=1 und legt EINEN Experten auf die
    # Karte (nur `R >= E` schaltet den Offload ganz ab). Die alte
    # Kartenformel schrieb 0 auf und liess die Store-Datei einen Platz zu
    # gross werden. Die Karte schreibt auf, was passiert.
    k0 = em.build(TOTAL, RATIOS, 0.0, [0.0, 0.0, 0.0])
    assert k0["slots"] == TOTAL - 1, "an expert always stays on the card"
    k1 = em.build(TOTAL, RATIOS, 1.0, [1.0, 1.0, 1.0])
    assert k1["slots"] == 0, "fully resident means: no store"


# --------------------------------------------------------------------------
# #107: DIE NAHT. Schreiber -> Datei -> ECHTER Leser.
# Der Grund, warum dieser Test existiert: #106 war ein Leser OHNE Schreiber,
# und das fiel erst nach drei toten Boots auf. Ein Test, der nur die
# Datenstruktur prueft, haette das nie gezeigt.
# --------------------------------------------------------------------------

def test_reader_reads_what_writer_writes(tmp_path):
    import json, os
    from flliper.srt.layers.moe import expert_store as es

    k = em.build(TOTAL, RATIOS, FR_PP, FR_TP)
    p = tmp_path / "expert_map.json"
    p.write_text(json.dumps(k))

    alt = os.environ.get(es.EXPERT_MAP_ENV)
    es._EXPERT_MAP_CACHE.clear()
    os.environ[es.EXPERT_MAP_ENV] = str(p)
    try:
        read_back = es.expert_map()
    finally:
        es._EXPERT_MAP_CACHE.clear()
        if alt is None:
            os.environ.pop(es.EXPERT_MAP_ENV, None)
        else:
            os.environ[es.EXPERT_MAP_ENV] = alt

    assert read_back is not None, "the reader discards what the writer writes"
    assert read_back["slots"] == 324
    assert em.slot_of(read_back, "P", 511) == em.slot_of(k, "P", 511)


def test_contradictory_map_is_discarded_not_used(tmp_path):
    """#91s Regel, hier fuer die Karte: lieber keine als eine falsche --
    eine falsche Slot-Zuordnung ist Datenverlust, kein Speicherverlust."""
    import json, os
    from flliper.srt.layers.moe import expert_store as es

    k = em.build(TOTAL, RATIOS, FR_PP, FR_TP)
    k["phases"]["D"]["slot_of"]["0"] = 0     # Id 0 ist bei D resident
    p = tmp_path / "kaputt.json"
    p.write_text(json.dumps(k))

    alt = os.environ.get(es.EXPERT_MAP_ENV)
    es._EXPERT_MAP_CACHE.clear()
    os.environ[es.EXPERT_MAP_ENV] = str(p)
    try:
        assert es.expert_map() is None
    finally:
        es._EXPERT_MAP_CACHE.clear()
        if alt is None:
            os.environ.pop(es.EXPERT_MAP_ENV, None)
        else:
            os.environ[es.EXPERT_MAP_ENV] = alt


def test_without_env_no_map():
    import os
    from flliper.srt.layers.moe import expert_store as es

    alt = os.environ.pop(es.EXPERT_MAP_ENV, None)
    es._EXPERT_MAP_CACHE.clear()
    try:
        assert es.expert_map() is None, "without Env everything stays as before #107"
    finally:
        if alt is not None:
            os.environ[es.EXPERT_MAP_ENV] = alt


def test_writer_takes_vectors_from_extra_d():
    """Die Karte entsteht aus denselben Flaggen, die #106 publiziert --
    nicht aus einer zweiten Quelle, die abweichen koennte."""
    from flliper.srt.pdflip.launcher import _argv_vector

    extra_d = ('--rank-moe-ratio 183,137,168 '
               '--rank-moe-resident-fraction 0.479,0.319,0.284')
    assert _argv_vector(extra_d, "--rank-moe-ratio") == ["183", "137", "168"]
    k = em.build(TOTAL,
                 [int(x) for x in _argv_vector(extra_d, "--rank-moe-ratio")],
                 FR_PP,
                 [float(x) for x in
                  _argv_vector(extra_d, "--rank-moe-resident-fraction")])
    assert k["slots"] == 324 and k["bounds"] == [0, 192, 336]


def test_map_disables_other_paths_not_just_precedes():
    """fnFL2w50 (07:15Z): meine erste Fassung war ein VORSPANN, kein Zweig.

    Sie setzte `_index`/`_slots` aus der Karte und liess den Rest der
    Funktion weiterlaufen -- `_global_only` war None, also fiel es in
    `elif _ratios and _fracs`, rechnete alles neu und starb an
    `#91: 92 kalte Experten stehen im Hotset`. Der Test liest die
    Verzweigung selbst, weil genau sie der Defekt war.
    """
    import inspect

    from flliper.srt.layers.moe import expert_offload as eo

    src = inspect.getsource(eo._expert_store_rows_for)
    i_card = src.index("if _karte is not None:")
    i_global = src.index("elif _global_only is not None:")
    i_ratios = src.index("elif _ratios and _fracs:")
    assert i_card < i_global < i_ratios, (
        "the card must be the FIRST branch, and the two old paths "
        "muessen `elif` sein -- sonst rechnen sie hinter ihr weiter")

    between = src[i_card:i_global]
    assert "_global_only = None if _karte is not None" in between, (
        "without this line the old path reads shared_resident_ids() again")
