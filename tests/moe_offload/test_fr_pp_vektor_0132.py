"""#132: die PP-Fractions gehen als VEKTOR in die Karte.

Zwei Layouts existieren, damit jede Phase ihr Optimum bekommt -- eine
Zahl fuer alle drei Stufen wirft das weg. #131 hatte das layer-gewichtete
Mittel gebildet; fnFL2w64 starb daran mit "95 eigene kalte Experten haben
in der KARTE keinen Platz", weil die Karte eine Residenz annahm, die
KEINE Stufe hat.
"""
import inspect

from flliper.srt.layers.moe import expert_map as em


def test_vector_is_computed_per_stage():
    res = em.resident_unsharded([0.367, 0.75, 0.95], 512)
    # #160: 487 statt 486 -- die Karte zaehlt jetzt wie der Rang.
    # ceil(512*0.95) = 487, round(512*0.95) = 486. Die 188 und 384 sind
    # unberuehrt, weil 0.367 und 0.75 auf beiden Wegen dasselbe geben.
    assert [len(x) for x in res] == [188, 384, 487]


def test_scalar_keeps_the_old_form():
    assert [len(x) for x in em.resident_unsharded(0.367, 512)] == [188]


def test_cold_is_what_worst_stage_does_not_hold():
    """Nicht die Vereinigung (die saehe 26) -- der Store muss Stufe 0
    bedienen koennen, die nur 188 von 512 haelt."""
    k = em.build(512, [183, 137, 168], [0.367, 0.75, 0.95], [0.70, 0.60, 0.55])
    assert k["slots"] == 512 - 188, k["slots"]


def test_map_sees_good_supply_of_3080s():
    """shared_resident muss steigen, wenn die hinteren Stufen mehr halten
    -- sonst plant der Flip Bewegungen, die unnoetig sind."""
    hoch = em.build(512, [183, 137, 168], [0.367, 0.75, 0.95], [0.70, 0.60, 0.55])
    flat = em.build(512, [183, 137, 168], 0.367, [0.70, 0.60, 0.55])
    assert hoch["shared_resident"] > flat["shared_resident"]


def test_launcher_passes_vector_through():
    from flliper.srt.pdflip import launcher as L

    src = inspect.getsource(L)
    i = src.index("_pp_frac = ")
    code = "\n".join(z for z in src[i:i + 200].split("\n")
                     if not z.lstrip().startswith("#"))
    assert "list(_fr_p)" in code, "the launcher averages again"
