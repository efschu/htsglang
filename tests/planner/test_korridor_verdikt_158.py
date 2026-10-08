"""#158: der Verdikt bewertet den Korridor -- getrennt von der Physik.

fnFL2w130 und w131 starben beide daran, dass der Verdikt gruen sagte und die
Karte dann auf 124 MiB frei lief. Die Bytes PASSTEN (fits=True); die
Betriebsregel war es, die riss.
"""
from flliper.srt.planner.pp_cut import d_rank_budget_verdict as budget_verdict

# Die gemessene Form von fnFL2w131, Reihenfolge TP0,TP1,TP2 = nvml1,nvml0,nvml2
W131 = dict(budgets_mib=[28240, 16672, 16672],
            card_total_mib=[32607, 20480, 20480],
            foreign_context_mib=[1446, 896, 894],
            nontorch_mib=[1981, 528, 524])
FLOOR = 1055.0


def test_without_floor_not_silently_green():
    """Ein fehlender Term heisst 'nicht gebucht', nicht 'gibt es nicht'."""
    for v in budget_verdict(**W131):
        assert v.corridor_ok is True
        assert "NICHT GEPRUEFT" in v.corridor_note


def test_5090_breaks_corridor_although_bytes_fit():
    v0 = budget_verdict(**W131, corridor_floor_mib=FLOOR)[0]
    assert v0.fits is True, "the physics fits -- that was never the problem"
    assert v0.corridor_ok is False
    assert round(v0.rest_mib) == 940
    assert "115 MiB unter dem Floor" in v0.corridor_note


def test_the_3080s_are_fine():
    for v in budget_verdict(**W131, corridor_floor_mib=FLOOR)[1:]:
        assert v.corridor_ok is True and v.fits is True


def test_mutant_without_145_terms_looks_green():
    """Der Zustand VOR dem heutigen Fix: fremd=0, nichttorch=0.

    Genau so lief jeder Boot bis 22.09. 20:2xZ -- und genau deshalb fiel
    nicht auf, dass die 5090 an der Grenze laeuft."""
    blind = dict(W131, foreign_context_mib=[0, 0, 0], nontorch_mib=[0, 0, 0])
    v0 = budget_verdict(**blind, corridor_floor_mib=FLOOR)[0]
    assert v0.corridor_ok is True, "without the terms nothing stands out"
    assert round(v0.rest_mib) == 4367
    # Mit den Termen sind es 940 -- 3427 MiB Unterschied, die der Planer
    # als frei verbucht hat.
    assert round(budget_verdict(**W131, corridor_floor_mib=FLOOR)[0].rest_mib) == 940


def test_mutant_just_above_and_just_below():
    v = budget_verdict(**W131, corridor_floor_mib=940.0)[0]
    assert v.corridor_ok is True, "exactly on the floor is ok"
    v = budget_verdict(**W131, corridor_floor_mib=941.0)[0]
    assert v.corridor_ok is False, "one MiB below it is not"
