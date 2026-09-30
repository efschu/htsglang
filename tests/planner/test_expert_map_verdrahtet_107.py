"""#107: die Karte zaehlt ALLE Karten -- und wird wirklich publiziert.

Nutzer 22.09.: "warum 1 id und die anderen karten halten doch auch noch
was???" und "ja dann wird dort vergessen ALLE karten zu zaehlen -- fixen".
"""
import json
import pytest
from flliper.srt.layers.moe import expert_map as em

W128 = dict(total=512, ratios=[183, 137, 168],
            fr_pp=[0.377, 0.700, 0.442], fr_tp=[0.006, 0.545, 0.449])


def test_checker_accepts_the_measured_form():
    """VORHER ROT: 'Phase P: 165 Ids sind resident UND im Store'.

    `build` bildet kalt_p als "was MINDESTENS EINE Stufe nicht haelt"
    (#132), der Pruefer las die VEREINIGUNG -- zwei Seiten einer Naht, die
    verschieden fragen. Die Karte wurde verworfen und der Lauf fiel auf die
    globale Menge zurueck.
    """
    assert em.refuse_if_inconsistent(em.build(**W128)) is None


def test_other_ranks_count_too():
    k = em.build(**W128)
    res_d = k["phases"]["D"]["resident"]
    # #160: die Karte zaehlt jetzt wie der Rang (ceil statt round). Rang 0
    # haelt bei FR_D 0.006 ueber 192 Experten ZWEI, nicht einen -- die alte
    # Kartenformel schrieb 1 auf und erklaerte damit eine residente Id fuer
    # kalt. Das ist die Zahl aus Aufgabe #79, hier an der Quelle.
    assert [len(x) for x in res_d] == [2, 79, 80], "the 3080s hold 79+80"
    # Ohne Karte bleibt im Schnitt aller Stufen/Raenge genau EINE Id -> 511
    # Store-Slots. Mit Karte sind es weniger, weil 79+80 mitzaehlen.
    assert k["slots"] < 511, f"slots={k['slots']} -- the card saves nothing"
    # 3 Plaetze weniger als vor #160: mehr resident heisst weniger kalt.
    assert k["slots"] == 351


def test_mutant_contradictory_map_still_discarded():
    """Der Pruefer darf nicht einfach alles durchwinken."""
    k = em.build(**W128)
    k["phases"]["D"]["slot_of"][str(k["phases"]["D"]["resident"][1][0])] = 0
    assert em.refuse_if_inconsistent(k) is not None


def test_mutant_slot_past_end_of_file():
    k = em.build(**W128)
    k["slots"] = 3
    assert "hinter dem Ende" in (em.refuse_if_inconsistent(k) or "")


def test_build_env_publishes_map_to_both_groups():
    from flliper.srt.pdflip.launcher import build_env
    kw = dict(tree="/t", venv="/v", cvd="0,1,2", store_dir="/s",
              debug_hold=False, tag="t1")
    for grp in ("P", "D"):
        env = build_env(group=grp, expert_map_path="/ev/expert_map_t1.json", **kw)
        assert env["FLLIPER_MOE_EXPERT_MAP"] == "/ev/expert_map_t1.json", grp
    # Ohne Pfad KEIN Schluessel -- ein leerer Wert waere ein Leser ohne
    # Inhalt, und `expert_map()` wuerde ihn als "nicht gesetzt" lesen.
    assert "FLLIPER_MOE_EXPERT_MAP" not in build_env(group="P", **kw)


def test_publish_writes_file_and_reader_finds_it(tmp_path, monkeypatch):
    """Die ganze Naht: Launcher schreibt -> expert_store liest."""
    from flliper.srt.pdflip import launcher as L
    from flliper.srt.layers.moe import expert_store as es

    class NS:
        extra_d = ('--rank-moe-ratio 183,137,168 '
                   '--rank-moe-resident-fraction 0.006,0.545,0.449')
        extra_p = ""
        pp_cut_expert_device_fraction = "0.377,0.700,0.442"
        tag = "t1"
    row_list = []
    from flliper.srt.planner import pp_cut as _pc
    monkeypatch.setattr(_pc, "checkpoint_weight_terms",
                        lambda m: type("T", (), {"num_experts": 512})())
    file_path = L.publish_expert_map(NS(), "/modell", str(tmp_path), row_list.append)
    assert file_path and json.load(open(file_path))["slots"] == 351
    assert any("#107 EXPERTEN-KARTE" in z and "351 Store-Plaetze" in z
               for z in row_list), row_list

    es._EXPERT_MAP_CACHE.clear()
    monkeypatch.setenv("FLLIPER_MOE_EXPERT_MAP", file_path)
    emap = es.expert_map()
    assert emap is not None, "the reader does not find the file"
    assert em.slot_of(emap, "D", 200) is None, "Id 200 is resident on D-rank 1"
    assert em.slot_of(emap, "D", 500) is not None, "no one holds Id 500"


def test_publish_writes_nothing_without_vectors(tmp_path):
    from flliper.srt.pdflip import launcher as L

    class NS:
        extra_d = ""
        extra_p = ""
        pp_cut_expert_device_fraction = ""
        tag = "t1"
    row_list = []
    assert L.publish_expert_map(NS(), "/m", str(tmp_path), row_list.append) == ""
    assert any("ENTFAELLT" in z for z in row_list)


def test_join_verdict_catches_w132_form():
    """#159: der Grund, aus dem w130/w131/w132 beim Wake starben."""
    k = em.build(**W128)
    row_list = em.join_verdict(k)
    assert len(row_list) == 3, "all three PP stages do not fit D"
    assert "P haelt 194 Experten, D haelt 161, gemeinsam 4" in row_list[0]


def test_equal_count_is_not_enough():
    """Weg A (uniforme FR_P) loest es NICHT -- die Ids bleiben verschieden."""
    # #160 verschiebt die D-Vereinigung auf 161, also muss FR_P mitwandern:
    # 0.314 ist das uniforme f mit resident_slot_count(512, f) == 161. Ohne
    # das prueft der Test nicht mehr "gleiche ANZAHL, andere Ids".
    k = em.build(total=512, ratios=[183, 137, 168],
                 fr_pp=[0.314] * 3, fr_tp=[0.006, 0.545, 0.449])
    row_list = em.join_verdict(k)
    assert len(row_list) == 3
    assert "P haelt 161 Experten, D haelt 161, gemeinsam 2" in row_list[0]


def test_identical_sets_are_joinable():
    """Die einzige Form, die der Austausch akzeptiert."""
    k = em.build(total=512, ratios=[183, 137, 168],
                 fr_pp=[1.0] * 3, fr_tp=[1.0, 1.0, 1.0])
    assert em.join_verdict(k) == [], em.join_verdict(k)


def test_mirror_makes_form_joinable():
    """#159: P spiegelt Ds Auswahl -- die einzige Form, die der Tausch kann."""
    k = em.build(**W128, mirror=True)
    assert em.join_verdict(k) == []
    assert em.refuse_if_inconsistent(k) is None
    assert k["moves"] == 0, "identical sets -> the Flip moves NICHTS"


def test_mirror_caps_on_too_small_fr_p():
    """FR_P wird zur OBERGRENZE: reicht sie nicht, bricht der Join sichtbar."""
    k = em.build(total=512, ratios=[183, 137, 168],
                 fr_pp=[0.377, 0.700, 0.442],
                 fr_tp=[0.688, 0.545, 0.449], mirror=True)
    row_list = em.join_verdict(k)
    assert row_list, "FR_P 0.377 -> 194 < D set 292, that must stand out"
    assert "P haelt 194" in row_list[0] and "D haelt 292" in row_list[0]


def test_mirror_with_matching_upper_bound():
    # 0.570 statt 0.564: nach #160 ist die D-Vereinigung 292, und der
    # Deckel muss sie ERREICHEN, sonst kuerzt er P und der Join meldet es
    # (genau das tut der Test darueber). 0.570 ist das kleinste uniforme f
    # mit resident_slot_count(512, f) == 292.
    k = em.build(total=512, ratios=[183, 137, 168], fr_pp=[0.570] * 3,
                 fr_tp=[0.688, 0.545, 0.449], mirror=True)
    assert em.join_verdict(k) == []
    assert [len(x) for x in k["phases"]["P"]["resident"]] == [292, 292, 292]
    assert k["moves"] == 0


def test_without_mirror_all_byte_identical():
    """Der alte Weg darf sich nicht still aendern."""
    a = em.build(**W128)
    b = em.build(**W128, mirror=False)
    assert a == b
    # +1 je Stufe gegen vor #160 -- das IST die Korrektur, nicht ein Bruch:
    # ceil(512*0.377)=194 ist, was der Rang haelt.
    assert [len(x) for x in a["phases"]["P"]["resident"]] == [194, 359, 227]
