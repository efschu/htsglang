"""Die Experten-Karte (Nutzer-Gesetz: "alles was geshardet wird braucht ne karte").

Die Tests halten die zwei Fallen fest, die am 22.09. Boots gekostet haben,
und die eine Zahl, die am Metall gemessen ist (fnFL2w24/w30: 324 Plaetze).
"""

import pytest

from sglang.srt.layers.moe import expert_map as em


RATIOS = [183, 137, 168]      # summiert 488, NICHT 512
FR_PP = 0.367
FR_TP = [0.479, 0.319, 0.284]
TOTAL = 512


def test_ratios_werden_skaliert_nicht_roh_genommen():
    """fnFL2w49 starb an genau 5 Ids (183..187).

    Ich hatte die rohen Ratios als Bereichsgrenzen genommen (0..182,
    183..319, 320..487); der Server skaliert auf 512. Die Karte muss die
    Grenzen des SERVERS fuehren, sonst beschreibt sie eine Aufteilung, die
    es nicht gibt.
    """
    assert sum(RATIOS) == 488, "sonst testet dieser Test nichts"
    assert em.scaled_spans(RATIOS, TOTAL) == [192, 144, 176]
    assert em.bounds(em.scaled_spans(RATIOS, TOTAL)) == [0, 192, 336]
    assert sum(em.scaled_spans(RATIOS, TOTAL)) == TOTAL, (
        "keine Id darf zwischen zwei Baendern verschwinden")


def test_die_plaetzezahl_ist_die_am_metall_gemessene():
    """fnFL2w24 und w30: 324 Plaetze, 506,25 MiB je Tensor, 36,71 GiB."""
    k = em.build(TOTAL, RATIOS, FR_PP, FR_TP)
    assert k["slots"] == 324


def test_der_store_haelt_den_TAUSCH_nicht_die_vereinigung():
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
    vereinigung = TOTAL - len(res_p & res_d)
    assert vereinigung == 420, "die Zahl, die w49 belegt hat"
    assert k["slots"] == 324 < vereinigung
    assert k["shared_resident"] == 92, "was auf den Karten liegen bleibt"
    assert k["moves"] == 192, "96 raus + 96 rein, Platz gegen Platz"


@pytest.mark.parametrize("phase", ["P", "D"])
def test_jede_id_ist_entweder_resident_oder_im_store(phase):
    """Die Bedingung, an der #97 VIERMAL gescheitert ist -- jetzt eine
    Eigenschaft der Karte statt einer Uebereinkunft zweier Rechnungen."""
    k = em.build(TOTAL, RATIOS, FR_PP, FR_TP)
    res = {g for ids in k["phases"][phase]["resident"] for g in ids}
    kalt = {int(x) for x in k["phases"][phase]["slot_of"]}
    assert not (res & kalt), "keine Id darf beides sein"
    assert len(res | kalt) == TOTAL, "und keine darf fehlen"


def test_plaetze_sind_luecken_und_kollisionsfrei():
    k = em.build(TOTAL, RATIOS, FR_PP, FR_TP)
    for phase in ("P", "D"):
        plaetze = sorted(int(v) for v in k["phases"][phase]["slot_of"].values())
        assert plaetze == list(range(len(plaetze))), (
            f"{phase}: Plaetze muessen 0..n-1 sein, luecken- und doppelfrei")
        assert len(plaetze) <= k["slots"]


def test_die_karte_prueft_sich_selbst():
    k = em.build(TOTAL, RATIOS, FR_PP, FR_TP)
    assert em.refuse_if_inconsistent(k) is None


def test_eine_kaputte_karte_wird_benannt_nicht_verschluckt():
    """Nach dem Umbau kann #97 nur noch aus einer falschen KARTE kommen --
    dann muss sie es sagen, mit Zahl."""
    k = em.build(TOTAL, RATIOS, FR_PP, FR_TP)
    k["phases"]["P"]["slot_of"]["0"] = 0        # Id 0 ist resident UND im Store
    grund = em.refuse_if_inconsistent(k)
    assert grund and "resident UND im Store" in grund


def test_slot_of_gibt_None_fuer_residente():
    k = em.build(TOTAL, RATIOS, FR_PP, FR_TP)
    assert em.slot_of(k, "P", 0) is None, "Id 0 haelt P auf der Karte"
    assert em.slot_of(k, "P", 511) is not None, "Id 511 liegt im Store"


def test_phase_of_trennt_die_gruppen():
    assert em.phase_of("P") == "P" and em.phase_of("PP") == "P"
    assert em.phase_of("D") == "D" and em.phase_of("TP") == "D"


def test_fraction_null_haelt_nichts_und_eins_haelt_alles():
    # #160: TOTAL-1, nicht TOTAL. `resident_slot_count` hat ein
    # `max(1, ...)`, und das ist keine Kosmetik -- `plan_load_time_staging`
    # rechnet bei fraction 0.0 wirklich R=1 und legt EINEN Experten auf die
    # Karte (nur `R >= E` schaltet den Offload ganz ab). Die alte
    # Kartenformel schrieb 0 auf und liess die Store-Datei einen Platz zu
    # gross werden. Die Karte schreibt auf, was passiert.
    k0 = em.build(TOTAL, RATIOS, 0.0, [0.0, 0.0, 0.0])
    assert k0["slots"] == TOTAL - 1, "ein Experte bleibt immer auf der Karte"
    k1 = em.build(TOTAL, RATIOS, 1.0, [1.0, 1.0, 1.0])
    assert k1["slots"] == 0, "voll resident heisst: kein Store"


# --------------------------------------------------------------------------
# #107: DIE NAHT. Schreiber -> Datei -> ECHTER Leser.
# Der Grund, warum dieser Test existiert: #106 war ein Leser OHNE Schreiber,
# und das fiel erst nach drei toten Boots auf. Ein Test, der nur die
# Datenstruktur prueft, haette das nie gezeigt.
# --------------------------------------------------------------------------

def test_der_leser_liest_was_der_schreiber_schreibt(tmp_path):
    import json, os
    from sglang.srt.layers.moe import expert_store as es

    k = em.build(TOTAL, RATIOS, FR_PP, FR_TP)
    p = tmp_path / "expert_map.json"
    p.write_text(json.dumps(k))

    alt = os.environ.get(es.EXPERT_MAP_ENV)
    es._EXPERT_MAP_CACHE.clear()
    os.environ[es.EXPERT_MAP_ENV] = str(p)
    try:
        gelesen = es.expert_map()
    finally:
        es._EXPERT_MAP_CACHE.clear()
        if alt is None:
            os.environ.pop(es.EXPERT_MAP_ENV, None)
        else:
            os.environ[es.EXPERT_MAP_ENV] = alt

    assert gelesen is not None, "der Leser verwirft, was der Schreiber schreibt"
    assert gelesen["slots"] == 324
    assert em.slot_of(gelesen, "P", 511) == em.slot_of(k, "P", 511)


def test_eine_widerspruechliche_karte_wird_verworfen_nicht_benutzt(tmp_path):
    """#91s Regel, hier fuer die Karte: lieber keine als eine falsche --
    eine falsche Slot-Zuordnung ist Datenverlust, kein Speicherverlust."""
    import json, os
    from sglang.srt.layers.moe import expert_store as es

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


def test_ohne_env_keine_karte():
    import os
    from sglang.srt.layers.moe import expert_store as es

    alt = os.environ.pop(es.EXPERT_MAP_ENV, None)
    es._EXPERT_MAP_CACHE.clear()
    try:
        assert es.expert_map() is None, "ohne Env bleibt alles wie vor #107"
    finally:
        if alt is not None:
            os.environ[es.EXPERT_MAP_ENV] = alt


def test_der_schreiber_zieht_die_vektoren_aus_extra_d():
    """Die Karte entsteht aus denselben Flaggen, die #106 publiziert --
    nicht aus einer zweiten Quelle, die abweichen koennte."""
    from sglang.srt.weg2.launcher import _argv_vector

    extra_d = ('--rank-moe-ratio 183,137,168 '
               '--rank-moe-resident-fraction 0.479,0.319,0.284')
    assert _argv_vector(extra_d, "--rank-moe-ratio") == ["183", "137", "168"]
    k = em.build(TOTAL,
                 [int(x) for x in _argv_vector(extra_d, "--rank-moe-ratio")],
                 FR_PP,
                 [float(x) for x in
                  _argv_vector(extra_d, "--rank-moe-resident-fraction")])
    assert k["slots"] == 324 and k["bounds"] == [0, 192, 336]


def test_die_karte_schaltet_die_anderen_wege_AB_nicht_nur_vor():
    """fnFL2w50 (07:15Z): meine erste Fassung war ein VORSPANN, kein Zweig.

    Sie setzte `_index`/`_slots` aus der Karte und liess den Rest der
    Funktion weiterlaufen -- `_global_only` war None, also fiel es in
    `elif _ratios and _fracs`, rechnete alles neu und starb an
    `#91: 92 kalte Experten stehen im Hotset`. Der Test liest die
    Verzweigung selbst, weil genau sie der Defekt war.
    """
    import inspect

    from sglang.srt.layers.moe import expert_offload as eo

    src = inspect.getsource(eo._expert_store_rows_for)
    i_karte = src.index("if _karte is not None:")
    i_global = src.index("elif _global_only is not None:")
    i_ratios = src.index("elif _ratios and _fracs:")
    assert i_karte < i_global < i_ratios, (
        "die Karte muss der ERSTE Zweig sein, und die beiden alten Wege "
        "muessen `elif` sein -- sonst rechnen sie hinter ihr weiter")

    zwischen = src[i_karte:i_global]
    assert "_global_only = None if _karte is not None" in zwischen, (
        "ohne diese Zeile liest der alte Weg wieder shared_resident_ids()")
