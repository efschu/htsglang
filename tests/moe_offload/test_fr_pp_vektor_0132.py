"""#132: die PP-Fractions gehen als VEKTOR in die Karte.

Zwei Layouts existieren, damit jede Phase ihr Optimum bekommt -- eine
Zahl fuer alle drei Stufen wirft das weg. #131 hatte das layer-gewichtete
Mittel gebildet; fnFL2w64 starb daran mit "95 eigene kalte Experten haben
in der KARTE keinen Platz", weil die Karte eine Residenz annahm, die
KEINE Stufe hat.
"""
import inspect

from sglang.srt.layers.moe import expert_map as em


def test_vektor_wird_je_stufe_gerechnet():
    res = em.resident_unsharded([0.367, 0.75, 0.95], 512)
    # #160: 487 statt 486 -- die Karte zaehlt jetzt wie der Rang.
    # ceil(512*0.95) = 487, round(512*0.95) = 486. Die 188 und 384 sind
    # unberuehrt, weil 0.367 und 0.75 auf beiden Wegen dasselbe geben.
    assert [len(x) for x in res] == [188, 384, 487]


def test_skalar_bleibt_die_alte_form():
    assert [len(x) for x in em.resident_unsharded(0.367, 512)] == [188]


def test_kalt_ist_was_die_schlechteste_stufe_nicht_haelt():
    """Nicht die Vereinigung (die saehe 26) -- der Store muss Stufe 0
    bedienen koennen, die nur 188 von 512 haelt."""
    k = em.build(512, [183, 137, 168], [0.367, 0.75, 0.95], [0.70, 0.60, 0.55])
    assert k["slots"] == 512 - 188, k["slots"]


def test_die_karte_sieht_die_gute_versorgung_der_3080er():
    """shared_resident muss steigen, wenn die hinteren Stufen mehr halten
    -- sonst plant der Flip Bewegungen, die unnoetig sind."""
    hoch = em.build(512, [183, 137, 168], [0.367, 0.75, 0.95], [0.70, 0.60, 0.55])
    flach = em.build(512, [183, 137, 168], 0.367, [0.70, 0.60, 0.55])
    assert hoch["shared_resident"] > flach["shared_resident"]


def test_der_launcher_reicht_den_vektor_durch():
    from sglang.srt.weg2 import launcher as L

    src = inspect.getsource(L)
    i = src.index("_pp_frac = ")
    code = "\n".join(z for z in src[i:i + 200].split("\n")
                     if not z.lstrip().startswith("#"))
    assert "list(_fr_p)" in code, "der Launcher mittelt wieder"
