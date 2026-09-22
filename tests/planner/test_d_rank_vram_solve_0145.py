"""#145: die D-Seite des Planners spuckt ihre VRAM-Zahlen aus.

Nutzer-Order 22.09., woertlich: *"diese falschen sizing zahlen tauchen jetzt
immer und immer wieder auf, werden dann endlos analysiert und verworfen und
halluziniert und neu berechnet. der planner muss sie ausspucken"*. #140/#141
haben das fuer die P-Stufen getan; hier ist das Gegenstueck fuer die D-Raenge.

DIE TABELLEN IN DIESER DATEI SIND GEMESSEN, NICHT ERWARTET. Quellen je Zahl
stehen neben der Zahl; alles stammt aus /spinning/evidence-665-f1:

  * ``vramwatch_fnFL2w80..w89.csv``  (ts,idx,name,used_mib,total_mib,pct)
  * ``boot_weg2_fnFL2w8*.D.log``     ([ct-stream-presplit] layer N ... |
    torch allocated X GiB reserved Y GiB, und [vram-census] ... after load)
  * ``fnFL2w83_374996e86478_0922_133145.log`` (die budget-D-Zeilen des Arms)

Rang -> NVML ist NICHT geraten: die D-Log-Zeile ``rank->card vector`` und die
``budget D ... ordinal=i nvml_idx=j``-Zeilen sagen uebereinstimmend
rang0 -> nvml1 (5090), rang1 -> nvml0 (3080), rang2 -> nvml2 (3080).
"""
import pytest

from sglang.srt.planner.pp_cut import (
    d_rank_available_mib,
    d_rank_budget_verdict,
    solve_expert_fraction_per_d_rank,
)

# --- DIE GEMESSENEN TABELLEN, in Rang-/Ordinal-Reihenfolge (5090 zuerst) ----

#: NVML-Gesamtgroesse. vramwatch-Spalte ``total_mib``, alle sieben Boots.
KARTE_MIB = [32607.0, 20480.0, 20480.0]

#: TERM (b) -- der Fremd-Kontext der SCHLAFENDEN Phase je Karte. Plateau
#: zwischen P's Einschlafen und D's Anstieg, IDENTISCH in sieben Boots
#: (w80/w81/w82/w83/w86/w87/w89; je 6-10 aufeinanderfolgende Proben):
#:     nvml0 = 852, nvml1 = 1342, nvml2 = 844
FREMD_MIB = [1342.0, 852.0, 844.0]

#: TERM (c) -- was der Rang AUSSERHALB des Torch-Allokators haelt, gemessen
#: als ``nvml_used - fremd - torch_reserved`` am Ende des Ladens:
#:     rang1/nvml0: 18498 - 852 - 17091 =  555   (w86, w89)
#:     rang2/nvml2: 20034 - 844 - 18668 =  522   (w83, w86, w89)
#:     rang0/nvml1: 31860 - 1342 - 26890 = 3628  (w83) -- UNTERGRENZE!
#: Rang 0 hat das Laden nie beendet (cu_mem_create OOM), w86 las an derselben
#: Stelle 32050 -> 3818. Die 3628 sind also ein unteres Ende, kein Endstand.
NICHTTORCH_MIB = [3628.0, 555.0, 522.0]

#: Was der Launcher den Raengen in w83 gegeben hat -- die drei
#: ``WEG2-LAUNCH budget D group=D ordinal=i``-Zeilen des Arm-Logs, identisch
#: mit ``rank_gpu_memory_mib=[29680, 18560, 18552]`` in D's server_args.
BUD_D_W83 = [29680.0, 18560.0, 18552.0]

#: Was die Raenge am Ende des Ladens WIRKLICH belegten, ohne den Fremd-Kontext
#: (vramwatch letzter Stand vor dem Tod minus FREMD_MIB):
#:     rang0/nvml1: >= 32050 - 1342 = 30708 (w86; noch steigend, dann OOM)
#:     rang1/nvml0:    18498 -  852 = 17646
#:     rang2/nvml2:    20034 -  844 = 19190
BELEGT_W83 = [30708.0, 17646.0, 19190.0]


# ---------------------------------------------------------------------------
# 1. Die Arithmetik, von Hand nachgerechnet
# ---------------------------------------------------------------------------

def test_verfuegbar_ist_karte_minus_die_drei_terme():
    a = d_rank_available_mib(
        card_total_mib=[10000.0], foreign_context_mib=[1000.0],
        nontorch_mib=[500.0], reserve_mib_by_rank=[2000.0],
    )
    assert a == [6500.0]


def test_ohne_reserve_ist_es_die_decke_nicht_die_empfehlung():
    ohne = d_rank_available_mib(
        card_total_mib=[10000.0], foreign_context_mib=[1000.0],
        nontorch_mib=[500.0],
    )[0]
    mit = d_rank_available_mib(
        card_total_mib=[10000.0], foreign_context_mib=[1000.0],
        nontorch_mib=[500.0], reserve_mib_by_rank=[2000.0],
    )[0]
    assert ohne == 8500.0 and mit == 6500.0 and mit < ohne


def test_halbe_bilanz_wird_verweigert():
    with pytest.raises(ValueError, match="halbe Bilanz"):
        d_rank_available_mib(
            card_total_mib=[1.0, 2.0, 3.0], foreign_context_mib=[1.0],
            nontorch_mib=[1.0, 1.0, 1.0],
        )


# ---------------------------------------------------------------------------
# 2. DER W83-NACHSPIELTEST -- der eigentliche Punkt dieser Datei
# ---------------------------------------------------------------------------

def test_w83_das_verdikt_nennt_rang_null_und_nur_rang_null():
    """Mit den gemessenen Termen faellt das Urteil, das w83 gefehlt hat.

    Rang 0 fragt 29680 MiB gegen 32607 - 1342 - 3628 = 27637 verfuegbar.
    Rang 1 und Rang 2 passen. Der Solver muss GENAU Rang 0 nennen -- ein
    Verdikt, das alle drei verdaechtigt, ist so wertlos wie keins.
    """
    v = d_rank_budget_verdict(
        budgets_mib=BUD_D_W83, card_total_mib=KARTE_MIB,
        foreign_context_mib=FREMD_MIB, nontorch_mib=NICHTTORCH_MIB,
    )
    assert [x.rank for x in v if not x.fits] == [0]
    assert v[0].available_mib == pytest.approx(27637.0)
    assert v[0].over_mib == pytest.approx(2043.0)
    # und die beiden 3080er passen, mit nennbarem Rest
    assert v[1].available_mib == pytest.approx(20480 - 852 - 555)   # 19073
    assert v[2].available_mib == pytest.approx(20480 - 844 - 522)   # 19114
    assert v[1].over_mib == pytest.approx(-513.0)
    assert v[2].over_mib == pytest.approx(-562.0)


def test_w83_ohne_den_fremd_kontext_bleibt_rang_null_unentdeckt():
    """MUTANT (b): Term (b) entfernt -> das Verdikt kippt.

    Ohne den Fremd-Kontext meldet Rang 0 nur noch 701 MiB Ueberschreitung
    statt 2043, und -- der eigentliche Schaden -- der zweite Term allein
    kann die Karte nicht mehr richtig bilanzieren: die verfuegbare Zahl liegt
    um exakt den Fremd-Kontext zu hoch, auf JEDEM Rang.
    """
    ohne_b = d_rank_budget_verdict(
        budgets_mib=BUD_D_W83, card_total_mib=KARTE_MIB,
        foreign_context_mib=[0.0, 0.0, 0.0], nontorch_mib=NICHTTORCH_MIB,
    )
    mit = d_rank_budget_verdict(
        budgets_mib=BUD_D_W83, card_total_mib=KARTE_MIB,
        foreign_context_mib=FREMD_MIB, nontorch_mib=NICHTTORCH_MIB,
    )
    for i, (a, b) in enumerate(zip(ohne_b, mit)):
        assert a.available_mib - b.available_mib == pytest.approx(FREMD_MIB[i]), (
            f"Rang {i}: der fehlende Term --d-foreign-context-mib verschiebt "
            f"die Bilanz um genau {FREMD_MIB[i]} MiB"
        )
    assert ohne_b[0].over_mib == pytest.approx(701.0)


def test_w83_ohne_den_nichttorch_term_faellt_das_verdikt_falsch_aus():
    """MUTANT (c): Term (c) entfernt -> Rang 0 sieht aus, als passte er.

    Das ist der Mutant, der zaehlt: OHNE --d-nontorch-mib lautet das Urteil
    'alle drei passen', und genau dieses Urteil hat w83 in den Tod geschickt.
    Der Test nennt den fehlenden Term ausdruecklich.
    """
    ohne_c = d_rank_budget_verdict(
        budgets_mib=BUD_D_W83, card_total_mib=KARTE_MIB,
        foreign_context_mib=FREMD_MIB, nontorch_mib=[0.0, 0.0, 0.0],
    )
    assert [x.rank for x in ohne_c if not x.fits] == [], (
        "ohne --d-nontorch-mib sieht w83 wie ein passender Boot aus -- "
        "genau das war der Defekt"
    )
    assert ohne_c[0].fits and ohne_c[0].over_mib == pytest.approx(-1585.0)
    # mit dem Term kippt es
    mit_c = d_rank_budget_verdict(
        budgets_mib=BUD_D_W83, card_total_mib=KARTE_MIB,
        foreign_context_mib=FREMD_MIB, nontorch_mib=NICHTTORCH_MIB,
    )
    assert not mit_c[0].fits


def test_die_gemessene_belegung_deckt_sich_mit_der_bilanz():
    """Gegenprobe am Metall, nicht am Modell.

    Was die Raenge am Ende WIRKLICH belegten (BELEGT_W83, aus vramwatch minus
    Fremd-Kontext) muss zu 'Karte minus Fremd' passen: die beiden 3080er
    darunter, die 5090 an der Decke. Auf der 5090 bleiben von 31265
    verfuegbaren MiB nur 557 uebrig -- und der Rang war noch am Wachsen.
    """
    frei = [k - f for k, f in zip(KARTE_MIB, FREMD_MIB)]
    assert frei == [31265.0, 19628.0, 19636.0]
    rest = [a - b for a, b in zip(frei, BELEGT_W83)]
    assert rest[0] == pytest.approx(557.0)      # 5090: an der physischen Kante
    assert rest[1] == pytest.approx(1982.0)
    assert rest[2] == pytest.approx(446.0)
    # Rang 1 blieb unter seinem Budget, Rang 2 hat seins UEBERSCHRITTEN --
    # gemessen, nicht erwartet: 19190 belegt gegen 18552 gegeben.
    assert BELEGT_W83[1] < BUD_D_W83[1]
    assert BELEGT_W83[2] > BUD_D_W83[2]
    assert BELEGT_W83[2] - BUD_D_W83[2] == pytest.approx(638.0)


def test_die_presplit_transiente_ist_gemessen_null():
    """WIDERLEGUNG, mit Gegenbeleg, statt einer uebernommenen Vermutung.

    Die Arbeitshypothese war: 'die Spitze, die der Experten-Presplit ueber den
    Endstand hinaus braucht' sei der fehlende Term. Sie ist es nicht. Ueber
    sechs Boots (w80/w81/w82/w83/w86/w89) ist das MAXIMUM von
    ``torch ... reserved`` ueber alle 48 ``[ct-stream-presplit] layer N``-
    Zeilen eines Rangs IDENTISCH mit dem ``[vram-census] ... after load``-Wert
    desselben Rangs. Der Presplit ueberschiesst nicht -- er waechst linear.

    Der Test haelt die gemessenen Paare fest, damit die Hypothese nicht
    wiederkommt.
    """
    # (rang, max reserved waehrend Presplit GiB, reserved after load GiB)
    gemessen = [(1, 16.69, 16.69), (2, 18.23, 18.23)]
    for rang, spitze, endstand in gemessen:
        assert spitze - endstand == pytest.approx(0.0), rang
    # Rang 0 hat den Presplit VOLLSTAENDIG durchlaufen (layer 47, reserved
    # 26.26 GiB = 26890 MiB) und ist erst DANACH gestorben, hinter
    # '#68 WORKER-TRANSPOSE'. 26890 + 555 (eigener Kontext, wie Rang 1) +
    # 1342 (fremd) = 28787 MiB haetten in 32607 gepasst -- der Tod kam aus
    # dem, was NACH dem Presplit noch ausserhalb von torch dazukam.
    assert 26890.0 + 555.0 + 1342.0 < KARTE_MIB[0]


# ---------------------------------------------------------------------------
# 3. Die Fraction-Decke je D-Rang (das Gegenstueck zu #140)
# ---------------------------------------------------------------------------

#: Die Zeilengroesse, zweimal unabhaengig gemessen und gleich:
#:   Launcher-Zeile PP-CUT FRACTION-SOLVE: 1238 MiB/Layer auf 512 Zeilen
#:   D-Log w83, TP0 layer 47: 'resident buffers 0.42 GiB' auf 180 Zeilen
EXPERT_LAYER_MIB = 1238.0
NUM_EXPERTS = 512
N_LAYERS = 48


def test_fraction_decke_rechnet_die_bedingung_rueckwaerts():
    f = solve_expert_fraction_per_d_rank(
        card_total_mib=[10000.0], foreign_context_mib=[1000.0],
        nontorch_mib=[500.0], n_layers=10,
        dense_layer_mib_by_rank=[100.0], expert_layer_mib=1000.0,
        num_experts=500, expert_span_by_rank=[250],
        scratch_rows_by_rank=[0],
    )[0]
    # verfuegbar 8500; dense 10*100 = 1000; Zeile = 1000/500 = 2 MiB
    # -> 7500 / (10 * 2 * 250) = 1.5 -> geklemmt auf 1.0
    assert f == 1.0
    f2 = solve_expert_fraction_per_d_rank(
        card_total_mib=[10000.0], foreign_context_mib=[1000.0],
        nontorch_mib=[500.0], n_layers=10,
        dense_layer_mib_by_rank=[100.0], expert_layer_mib=1000.0,
        num_experts=500, expert_span_by_rank=[500],
        scratch_rows_by_rank=[0], reserve_mib_by_rank=[3000.0],
    )[0]
    # verfuegbar 5500; -1000 dense = 4500 / (10*2*500) = 0.45
    assert f2 == pytest.approx(0.45)


def test_scratch_zeilen_kosten_mit():
    ohne = solve_expert_fraction_per_d_rank(
        card_total_mib=[10000.0], foreign_context_mib=[0.0],
        nontorch_mib=[0.0], n_layers=10, dense_layer_mib_by_rank=[0.0],
        expert_layer_mib=1000.0, num_experts=500, expert_span_by_rank=[500],
        scratch_rows_by_rank=[0],
    )[0]
    mit = solve_expert_fraction_per_d_rank(
        card_total_mib=[10000.0], foreign_context_mib=[0.0],
        nontorch_mib=[0.0], n_layers=10, dense_layer_mib_by_rank=[0.0],
        expert_layer_mib=1000.0, num_experts=500, expert_span_by_rank=[500],
        scratch_rows_by_rank=[50],
    )[0]
    # 50 Zeilen x 2 MiB x 10 Layer = 1000 MiB weniger fuer die Residenz
    assert mit < ohne
    assert (ohne - mit) * (10 * 2 * 500) == pytest.approx(1000.0)


def test_halbe_geometrie_wird_verweigert():
    with pytest.raises(ValueError, match="halbe Geometrie"):
        solve_expert_fraction_per_d_rank(
            card_total_mib=[1.0, 2.0, 3.0], foreign_context_mib=[0.0] * 3,
            nontorch_mib=[0.0] * 3, n_layers=1,
            dense_layer_mib_by_rank=[1.0], expert_layer_mib=1.0,
            num_experts=1, expert_span_by_rank=[1, 1, 1],
        )


def test_die_w83_form_je_rang():
    """Die echte Form: 48 Layer, 512 Experten zu 1238 MiB/Layer, Spannen aus
    --rank-moe-ratio 183,137,168, Scratch 44,48,48, mit den gemessenen
    Terms (b) und (c). Kein Urteil ueber die Zahlen -- festgehalten wird,
    dass je Rang EINE eigene Decke herauskommt und dass die 5090 mit dem
    gemessenen Nicht-Torch-Term nicht mehr die gefahrenen 0.70 traegt.
    """
    span = [192, 144, 176]  # 183/137/168 auf 512 skaliert, Rest auf den letzten
    f = solve_expert_fraction_per_d_rank(
        card_total_mib=KARTE_MIB, foreign_context_mib=FREMD_MIB,
        nontorch_mib=NICHTTORCH_MIB, n_layers=N_LAYERS,
        dense_layer_mib_by_rank=[62.0, 0.0, 0.0],   # --rank-tp-ratio 1,0,0
        expert_layer_mib=EXPERT_LAYER_MIB, num_experts=NUM_EXPERTS,
        expert_span_by_rank=span, scratch_rows_by_rank=[44, 48, 48],
    )
    assert len(f) == 3 and all(0.0 <= x <= 1.0 for x in f)
    gefahren = [0.70, 0.60, 0.55]

    # EIGENE ANNAHME, VOM TEST WIDERLEGT (dieselbe Klasse wie in #140, wo
    # 'f[0] < f[1] < f[2]' fiel): ich hatte erwartet, die Decke nenne Rang 0.
    # GEMESSEN ohne Reserve: [0.878, 0.808, 0.663] -- ALLE DREI liegen ueber
    # den gefahrenen Fractions. Die Decke allein entscheidet auch auf der
    # D-Seite nichts; die RESERVE (KV-Pool, Draft, Aktivierungen) ist der
    # tragende Eingang, und ohne sie sagt die Decke 'passt' ueber einen Boot,
    # der gestorben ist. Genau dafuer gibt es --d-reserve-mib.
    assert all(m > g for g, m in zip(gefahren, f)), f
    assert f == pytest.approx([0.8775, 0.8079, 0.6630], abs=1e-3), f

    # WIE die Reserve wirkt, als GEMESSENE Tabelle statt als behaupteter
    # Schwellwert (alle drei Raenge dieselbe Reserve, MiB):
    #        0 -> 0.878 / 0.808 / 0.663
    #     2000 -> 0.788 / 0.688 / 0.565
    #     3400 -> 0.725 / 0.604 / 0.497
    #     4000 -> 0.698 / 0.569 / 0.467
    #     6000 -> 0.608 / 0.449 / 0.369
    # Und die REIHENFOLGE, in der die Raenge unter ihre gefahrene Fraction
    # fallen (Bisektion auf derselben Geometrie):
    #     Rang 2 bei 2308 MiB Reserve -> Rang 1 bei 3474 -> Rang 0 bei 3955
    # Auch das widerspricht meiner Erwartung "die 5090 zuerst" -- und deckt
    # sich mit dem Metall, wo genau Rang 2 sein Budget ueberschritten hat.
    # Der Test bindet die MONOTONIE und diese REIHENFOLGE, nicht eine Zahl,
    # die kein Boot gemessen hat.
    vorher = f
    for r in (1000.0, 2000.0, 3000.0, 4000.0, 6000.0):
        jetzt = solve_expert_fraction_per_d_rank(
            card_total_mib=KARTE_MIB, foreign_context_mib=FREMD_MIB,
            nontorch_mib=NICHTTORCH_MIB, n_layers=N_LAYERS,
            dense_layer_mib_by_rank=[62.0, 0.0, 0.0],
            expert_layer_mib=EXPERT_LAYER_MIB, num_experts=NUM_EXPERTS,
            expert_span_by_rank=span, scratch_rows_by_rank=[44, 48, 48],
            reserve_mib_by_rank=[r, r, r],
        )
        assert all(a <= b for a, b in zip(jetzt, vorher)), (r, jetzt, vorher)
        vorher = jetzt
    # Rang 2 ist der engste relativ zu seiner gefahrenen Fraction -- und das
    # deckt sich mit dem Metall: Rang 2 hat sein Budget als einziger
    # ueberschritten (19190 belegt gegen 18552 gegeben, Test darueber).
    def _drueber(r):
        jetzt = solve_expert_fraction_per_d_rank(
            card_total_mib=KARTE_MIB, foreign_context_mib=FREMD_MIB,
            nontorch_mib=NICHTTORCH_MIB, n_layers=N_LAYERS,
            dense_layer_mib_by_rank=[62.0, 0.0, 0.0],
            expert_layer_mib=EXPERT_LAYER_MIB, num_experts=NUM_EXPERTS,
            expert_span_by_rank=span, scratch_rows_by_rank=[44, 48, 48],
            reserve_mib_by_rank=[r, r, r],
        )
        return [i for i, (g, m) in enumerate(zip(gefahren, jetzt)) if g > m]

    assert _drueber(2000.0) == []            # unter 2308 ist noch keiner drueber
    assert _drueber(2400.0) == [2]           # Rang 2 zuerst
    assert _drueber(3500.0) == [1, 2]        # dann Rang 1
    assert _drueber(4000.0) == [0, 1, 2]     # zuletzt die 5090

    # MUTANT (c), eine Ebene tiefer als im Verdikt: ohne --d-nontorch-mib
    # steigt JEDE Decke, und die der 5090 am staerksten -- der Rang, der den
    # Boot verloren hat, ist genau der, dem der fehlende Term am meisten
    # Luft vorgaukelt.
    ohne_c = solve_expert_fraction_per_d_rank(
        card_total_mib=KARTE_MIB, foreign_context_mib=FREMD_MIB,
        nontorch_mib=[0.0, 0.0, 0.0], n_layers=N_LAYERS,
        dense_layer_mib_by_rank=[62.0, 0.0, 0.0],
        expert_layer_mib=EXPERT_LAYER_MIB, num_experts=NUM_EXPERTS,
        expert_span_by_rank=span, scratch_rows_by_rank=[44, 48, 48],
    )
    assert all(a > b for a, b in zip(ohne_c, f)), (ohne_c, f)
    zuwachs = [a - b for a, b in zip(ohne_c, f)]
    assert zuwachs[0] == max(zuwachs), (
        "ohne --d-nontorch-mib gewinnt Rang 0 (5090) die groesste Schein-Luft "
        f"({zuwachs[0]:.3f} gegen {zuwachs[1]:.3f} / {zuwachs[2]:.3f}) -- das "
        "ist die Falschaussage, an der w83 gestorben ist"
    )
