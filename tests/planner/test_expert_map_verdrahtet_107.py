"""#107: die Karte zaehlt ALLE Karten -- und wird wirklich publiziert.

Nutzer 22.09.: "warum 1 id und die anderen karten halten doch auch noch
was???" und "ja dann wird dort vergessen ALLE karten zu zaehlen -- fixen".
"""
import json
import pytest
from sglang.srt.layers.moe import expert_map as em

W128 = dict(total=512, ratios=[183, 137, 168],
            fr_pp=[0.377, 0.700, 0.442], fr_tp=[0.006, 0.545, 0.449])


def test_pruefer_akzeptiert_die_gemessene_form():
    """VORHER ROT: 'Phase P: 165 Ids sind resident UND im Store'.

    `build` bildet kalt_p als "was MINDESTENS EINE Stufe nicht haelt"
    (#132), der Pruefer las die VEREINIGUNG -- zwei Seiten einer Naht, die
    verschieden fragen. Die Karte wurde verworfen und der Lauf fiel auf die
    globale Menge zurueck.
    """
    assert em.refuse_if_inconsistent(em.build(**W128)) is None


def test_die_anderen_raenge_zaehlen_mit():
    k = em.build(**W128)
    res_d = k["phases"]["D"]["resident"]
    # #160: die Karte zaehlt jetzt wie der Rang (ceil statt round). Rang 0
    # haelt bei FR_D 0.006 ueber 192 Experten ZWEI, nicht einen -- die alte
    # Kartenformel schrieb 1 auf und erklaerte damit eine residente Id fuer
    # kalt. Das ist die Zahl aus Aufgabe #79, hier an der Quelle.
    assert [len(x) for x in res_d] == [2, 79, 80], "die 3080er halten 79+80"
    # Ohne Karte bleibt im Schnitt aller Stufen/Raenge genau EINE Id -> 511
    # Store-Slots. Mit Karte sind es weniger, weil 79+80 mitzaehlen.
    assert k["slots"] < 511, f"slots={k['slots']} -- die Karte spart nichts"
    # 3 Plaetze weniger als vor #160: mehr resident heisst weniger kalt.
    assert k["slots"] == 351


def test_mutant_widerspruechliche_karte_wird_weiter_verworfen():
    """Der Pruefer darf nicht einfach alles durchwinken."""
    k = em.build(**W128)
    k["phases"]["D"]["slot_of"][str(k["phases"]["D"]["resident"][1][0])] = 0
    assert em.refuse_if_inconsistent(k) is not None


def test_mutant_platz_hinter_dem_dateiende():
    k = em.build(**W128)
    k["slots"] = 3
    assert "hinter dem Ende" in (em.refuse_if_inconsistent(k) or "")


def test_build_env_publiziert_die_karte_in_beide_gruppen():
    from sglang.srt.weg2.launcher import build_env
    kw = dict(tree="/t", venv="/v", cvd="0,1,2", store_dir="/s",
              debug_hold=False, tag="t1")
    for grp in ("P", "D"):
        env = build_env(group=grp, expert_map_path="/ev/expert_map_t1.json", **kw)
        assert env["SGLANG_MOE_EXPERT_MAP"] == "/ev/expert_map_t1.json", grp
    # Ohne Pfad KEIN Schluessel -- ein leerer Wert waere ein Leser ohne
    # Inhalt, und `expert_map()` wuerde ihn als "nicht gesetzt" lesen.
    assert "SGLANG_MOE_EXPERT_MAP" not in build_env(group="P", **kw)


def test_publish_schreibt_die_datei_und_der_leser_findet_sie(tmp_path, monkeypatch):
    """Die ganze Naht: Launcher schreibt -> expert_store liest."""
    from sglang.srt.weg2 import launcher as L
    from sglang.srt.layers.moe import expert_store as es

    class NS:
        extra_d = ('--rank-moe-ratio 183,137,168 '
                   '--rank-moe-resident-fraction 0.006,0.545,0.449')
        extra_p = ""
        pp_cut_expert_device_fraction = "0.377,0.700,0.442"
        tag = "t1"
    zeilen = []
    from sglang.srt.planner import pp_cut as _pc
    monkeypatch.setattr(_pc, "checkpoint_weight_terms",
                        lambda m: type("T", (), {"num_experts": 512})())
    pfad = L.publish_expert_map(NS(), "/modell", str(tmp_path), zeilen.append)
    assert pfad and json.load(open(pfad))["slots"] == 351
    assert any("#107 EXPERTEN-KARTE" in z and "351 Store-Plaetze" in z
               for z in zeilen), zeilen

    es._EXPERT_MAP_CACHE.clear()
    monkeypatch.setenv("SGLANG_MOE_EXPERT_MAP", pfad)
    karte = es.expert_map()
    assert karte is not None, "der Leser findet die Datei nicht"
    assert em.slot_of(karte, "D", 200) is None, "Id 200 ist auf D-Rang 1 resident"
    assert em.slot_of(karte, "D", 500) is not None, "Id 500 haelt niemand"


def test_publish_schreibt_nichts_ohne_vektoren(tmp_path):
    from sglang.srt.weg2 import launcher as L

    class NS:
        extra_d = ""
        extra_p = ""
        pp_cut_expert_device_fraction = ""
        tag = "t1"
    zeilen = []
    assert L.publish_expert_map(NS(), "/m", str(tmp_path), zeilen.append) == ""
    assert any("ENTFAELLT" in z for z in zeilen)


def test_join_verdikt_faengt_die_w132_form():
    """#159: der Grund, aus dem w130/w131/w132 beim Wake starben."""
    k = em.build(**W128)
    zeilen = em.join_verdict(k)
    assert len(zeilen) == 3, "alle drei PP-Stufen passen nicht zu D"
    assert "P haelt 194 Experten, D haelt 161, gemeinsam 4" in zeilen[0]


def test_gleiche_anzahl_genuegt_nicht():
    """Weg A (uniforme FR_P) loest es NICHT -- die Ids bleiben verschieden."""
    # #160 verschiebt die D-Vereinigung auf 161, also muss FR_P mitwandern:
    # 0.314 ist das uniforme f mit resident_slot_count(512, f) == 161. Ohne
    # das prueft der Test nicht mehr "gleiche ANZAHL, andere Ids".
    k = em.build(total=512, ratios=[183, 137, 168],
                 fr_pp=[0.314] * 3, fr_tp=[0.006, 0.545, 0.449])
    zeilen = em.join_verdict(k)
    assert len(zeilen) == 3
    assert "P haelt 161 Experten, D haelt 161, gemeinsam 2" in zeilen[0]


def test_identische_mengen_sind_joinbar():
    """Die einzige Form, die der Austausch akzeptiert."""
    k = em.build(total=512, ratios=[183, 137, 168],
                 fr_pp=[1.0] * 3, fr_tp=[1.0, 1.0, 1.0])
    assert em.join_verdict(k) == [], em.join_verdict(k)


def test_mirror_macht_die_form_joinbar():
    """#159: P spiegelt Ds Auswahl -- die einzige Form, die der Tausch kann."""
    k = em.build(**W128, mirror=True)
    assert em.join_verdict(k) == []
    assert em.refuse_if_inconsistent(k) is None
    assert k["moves"] == 0, "identische Mengen -> der Flip bewegt NICHTS"


def test_mirror_deckelt_bei_zu_kleinem_fr_p():
    """FR_P wird zur OBERGRENZE: reicht sie nicht, bricht der Join sichtbar."""
    k = em.build(total=512, ratios=[183, 137, 168],
                 fr_pp=[0.377, 0.700, 0.442],
                 fr_tp=[0.688, 0.545, 0.449], mirror=True)
    zeilen = em.join_verdict(k)
    assert zeilen, "FR_P 0.377 -> 194 < D-Menge 292, das muss auffallen"
    assert "P haelt 194" in zeilen[0] and "D haelt 292" in zeilen[0]


def test_mirror_mit_passender_obergrenze():
    # 0.570 statt 0.564: nach #160 ist die D-Vereinigung 292, und der
    # Deckel muss sie ERREICHEN, sonst kuerzt er P und der Join meldet es
    # (genau das tut der Test darueber). 0.570 ist das kleinste uniforme f
    # mit resident_slot_count(512, f) == 292.
    k = em.build(total=512, ratios=[183, 137, 168], fr_pp=[0.570] * 3,
                 fr_tp=[0.688, 0.545, 0.449], mirror=True)
    assert em.join_verdict(k) == []
    assert [len(x) for x in k["phases"]["P"]["resident"]] == [292, 292, 292]
    assert k["moves"] == 0


def test_ohne_mirror_bleibt_alles_byte_identisch():
    """Der alte Weg darf sich nicht still aendern."""
    a = em.build(**W128)
    b = em.build(**W128, mirror=False)
    assert a == b
    # +1 je Stufe gegen vor #160 -- das IST die Korrektur, nicht ein Bruch:
    # ceil(512*0.377)=194 ist, was der Rang haelt.
    assert [len(x) for x in a["phases"]["P"]["resident"]] == [194, 359, 227]
