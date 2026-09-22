"""#158: der Verdikt bewertet den Korridor -- getrennt von der Physik.

fnFL2w130 und w131 starben beide daran, dass der Verdikt gruen sagte und die
Karte dann auf 124 MiB frei lief. Die Bytes PASSTEN (fits=True); die
Betriebsregel war es, die riss.
"""
from sglang.srt.planner.pp_cut import d_rank_budget_verdict as verdikt

# Die gemessene Form von fnFL2w131, Reihenfolge TP0,TP1,TP2 = nvml1,nvml0,nvml2
W131 = dict(budgets_mib=[28240, 16672, 16672],
            card_total_mib=[32607, 20480, 20480],
            foreign_context_mib=[1446, 896, 894],
            nontorch_mib=[1981, 528, 524])
FLOOR = 1055.0


def test_ohne_floor_wird_NICHT_STILL_GRUEN_gesagt():
    """Ein fehlender Term heisst 'nicht gebucht', nicht 'gibt es nicht'."""
    for v in verdikt(**W131):
        assert v.corridor_ok is True
        assert "NICHT GEPRUEFT" in v.corridor_note


def test_die_5090_reisst_den_korridor_obwohl_die_bytes_passen():
    v0 = verdikt(**W131, corridor_floor_mib=FLOOR)[0]
    assert v0.fits is True, "die Physik passt -- das war nie das Problem"
    assert v0.corridor_ok is False
    assert round(v0.rest_mib) == 940
    assert "115 MiB unter dem Floor" in v0.corridor_note


def test_die_3080er_sind_in_ordnung():
    for v in verdikt(**W131, corridor_floor_mib=FLOOR)[1:]:
        assert v.corridor_ok is True and v.fits is True


def test_mutant_ohne_die_145_terme_sieht_alles_gruen_aus():
    """Der Zustand VOR dem heutigen Fix: fremd=0, nichttorch=0.

    Genau so lief jeder Boot bis 22.09. 20:2xZ -- und genau deshalb fiel
    nicht auf, dass die 5090 an der Grenze laeuft."""
    blind = dict(W131, foreign_context_mib=[0, 0, 0], nontorch_mib=[0, 0, 0])
    v0 = verdikt(**blind, corridor_floor_mib=FLOOR)[0]
    assert v0.corridor_ok is True, "ohne die Terme faellt nichts auf"
    assert round(v0.rest_mib) == 4367
    # Mit den Termen sind es 940 -- 3427 MiB Unterschied, die der Planer
    # als frei verbucht hat.
    assert round(verdikt(**W131, corridor_floor_mib=FLOOR)[0].rest_mib) == 940


def test_mutant_knapp_drueber_und_knapp_drunter():
    v = verdikt(**W131, corridor_floor_mib=940.0)[0]
    assert v.corridor_ok is True, "genau auf dem Floor ist ok"
    v = verdikt(**W131, corridor_floor_mib=941.0)[0]
    assert v.corridor_ok is False, "ein MiB darunter ist es nicht"
