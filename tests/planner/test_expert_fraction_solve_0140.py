"""#140: der Planner spuckt die Experten-Fraction aus, statt sie zu nehmen.

Nutzer-Order 22.09., woertlich: *"diese falschen sizing zahlen tauchen jetzt
immer und immer wieder auf, werden dann endlos analysiert und verworfen und
halluziniert und neu berechnet. der planner muss sie ausspucken"*.

Der Anlass mit Zahl, fnFL2w73: PP2 haelt bei FR_P 0.95 487 von 512 Experten,
danach bleiben 4,93 GB und der Draft-KV-Produzent stirbt still im C++
(`cu_mem_create: out of memory`, kein Python-Traceback).
"""
import pytest

from sglang.srt.planner.pp_cut import solve_expert_fraction_per_stage


def test_die_loesung_passt_genau_ins_budget():
    # Eine Stufe, 10 Layer, 100 MiB dense + 1000 MiB Experten je Layer,
    # Budget 6000 MiB -> (6000 - 1000) / 10000 = 0.5
    f = solve_expert_fraction_per_stage(
        budgets_mib=[6000], stage_layers=[10], mean_layer_mib=100.0,
        expert_layer_mib=1000.0, num_experts=512, lru_rows=[0],
    )
    assert f == [0.5]
    # und die Rueckrechnung trifft das Budget
    assert 10 * (100.0 + 1000.0 * f[0]) == 6000.0


def test_die_reserve_geht_ab_und_das_ist_der_ganze_punkt():
    """Ohne Reserve ist das Ergebnis die DECKE, nicht die Empfehlung -- der
    Draft, das KV und die Aktivierungen stehen noch aus. w73 starb genau in
    dieser Luecke."""
    ohne = solve_expert_fraction_per_stage(
        budgets_mib=[6000], stage_layers=[10], mean_layer_mib=100.0,
        expert_layer_mib=1000.0, num_experts=512, lru_rows=[0],
    )[0]
    mit = solve_expert_fraction_per_stage(
        budgets_mib=[6000], stage_layers=[10], mean_layer_mib=100.0,
        expert_layer_mib=1000.0, num_experts=512, lru_rows=[0],
        reserve_mib_by_stage=[2000],
    )[0]
    assert mit < ohne and mit == pytest.approx(0.3)


def test_lru_zeilen_kosten_mit():
    f = solve_expert_fraction_per_stage(
        budgets_mib=[6000], stage_layers=[10], mean_layer_mib=100.0,
        expert_layer_mib=1024.0, num_experts=512, lru_rows=[32],
    )[0]
    # row_mib = 1024/512 = 2; 32 Zeilen = 64 MiB je Layer
    assert f == pytest.approx((6000 - 10 * (100 + 64)) / (10 * 1024))


def test_eine_stufe_die_nicht_mal_dense_traegt_gibt_null():
    """Geklemmt, aber der Aufrufer soll es benennen: 0.0 heisst hier NICHT
    'passt knapp', sondern 'diese Stufe traegt ihre Dense-Gewichte nicht'."""
    f = solve_expert_fraction_per_stage(
        budgets_mib=[500], stage_layers=[10], mean_layer_mib=100.0,
        expert_layer_mib=1000.0, num_experts=512, lru_rows=[0],
    )
    assert f == [0.0]


def test_ueber_eins_wird_geklemmt():
    f = solve_expert_fraction_per_stage(
        budgets_mib=[999999], stage_layers=[1], mean_layer_mib=1.0,
        expert_layer_mib=10.0, num_experts=512, lru_rows=[0],
    )
    assert f == [1.0]


def test_halbe_geometrie_wird_verweigert():
    with pytest.raises(ValueError, match="halbe Geometrie"):
        solve_expert_fraction_per_stage(
            budgets_mib=[1000, 1000], stage_layers=[10, 10, 10],
            mean_layer_mib=1.0, expert_layer_mib=1.0, num_experts=1,
            lru_rows=[0, 0, 0],
        )


def test_die_w73_form_je_stufe():
    """Die echte Form: PP-Cut 29/11/8, Budgets aus dem Arm (BUD_P
    25900/16000/15600 MiB), 512 Experten. Kein Urteil ueber die Zahlen --
    der Test haelt fest, dass je Stufe EINE eigene Decke herauskommt und
    dass die engste Stufe die kleinste hat."""
    f = solve_expert_fraction_per_stage(
        budgets_mib=[25900, 16000, 15600], stage_layers=[29, 11, 8],
        mean_layer_mib=60.0, expert_layer_mib=800.0, num_experts=512,
        lru_rows=[32, 32, 32],
    )
    assert len(f) == 3 and all(0.0 <= x <= 1.0 for x in f)
    # GEMESSEN, nicht erwartet: [0.979, 1.0, 1.0]. Meine erste Fassung
    # verlangte f[0] < f[1] < f[2] und war falsch -- Stufe 1 und 2 sind
    # GEKLEMMT, ihre rechnerische Decke liegt ueber 1.0. Genau das ist der
    # Befund: OHNE Reserve sagt die Decke "0.95 passt", und w73 ist
    # trotzdem gestorben, weil der Draft danach kommt. Die Decke allein
    # entscheidet nichts; die RESERVE ist der tragende Eingang.
    assert f[1] == 1.0 and f[2] == 1.0, f
    assert f[0] < 1.0, f          # 29 Layer auf 25,9 GB sind die enge Stufe

    # WIE die Reserve wirkt, als Tabelle statt als behaupteter Schwellwert.
    # Meine zweite Annahme ("5 GB Draft druecken Stufe 2 unter 0.95") war
    # ebenfalls falsch -- gemessen:
    #     Reserve     0 MiB -> 0.979 / 1.000 / 1.000
    #     Reserve  5000 MiB -> 0.763 / 1.000 / 1.000
    #     Reserve 11000 MiB -> 0.505 / 0.431 / 0.581
    # Stufe 2 (8 Layer auf 15,6 GB) haelt ihre volle Residenz bis ~11 GB
    # Reserve. Der Test bindet die MONOTONIE und die Richtung, nicht eine
    # Zahl, die ich nicht gemessen habe.
    vorher = f
    for r in (2000, 5000, 8000, 11000):
        jetzt = solve_expert_fraction_per_stage(
            budgets_mib=[25900, 16000, 15600], stage_layers=[29, 11, 8],
            mean_layer_mib=60.0, expert_layer_mib=800.0, num_experts=512,
            lru_rows=[32, 32, 32], reserve_mib_by_stage=[r, r, r],
        )
        assert all(a <= b for a, b in zip(jetzt, vorher)), (r, jetzt, vorher)
        vorher = jetzt
    assert vorher[2] < 0.95, (
        f"bei 11 GB Reserve muss auch die kleinste Stufe unter die "
        f"gefahrenen 0.95 fallen: {vorher}"
    )
