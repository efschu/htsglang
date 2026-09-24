# SPDX-License-Identifier: Apache-2.0
"""fnFL2 H34 -- der Kredit des Wakes P->D, in Millisekunden gerechnet.

DER BEFUND, aus den Logzeilen: im P->D-Wake von fnFL2x141 (Draft auf P) wartet
D TP2 (3080, nvml2) 73/83/94 ms an ``weights_4`` und 472/427/455 ms an
``weights_6`` auf VRAM-Kredit (die drei 12k/8k-Flips); ``weights_6`` bekommt
seinen Kredit erst, als PP2 ``weights_draft`` (2770 MiB, 470 ms Deposit ueber
eine Lane zur 5090) pausiert. PP0 hat ``weights_6`` da schon begonnen, seine
Lane p1 zu TP2 wartet 161 ms, PP0s Kette -- der kritische Pfad -- kommt ~90 ms
spaeter an. In der H25-Form (fnFL2x144, kein Draft auf P) wartet TP2 hoechstens
12 ms. H14 rechnet nur D->P; niemand rechnete diesen Weg.

Alle Zahlen sind Logzeilen: ``fixtures/wake_credit_pd_h34/`` sind woertliche,
hinter dem letzten gelesenen Feld abgeschnittene Auszuege aus
/spinning/evidence-665-f1/boot_weg2_fnFL2x{141,144,145}_*.{P,D,front}.log (die
Zeilen, die ``pd_reference_from_logs`` liest, P->D-Flips 0 und 1). x145 (H30
Stufe K in der H25-Form, 24.09. 11:13Z) lief NACH dem Bau des Planers und ist
hier nur Pruefstein: der Plan aus x144 + Pufferregel sagt seine Waende voraus.
"""

import json
import os
import types

import pytest

from sglang.srt.planner import expert_residency as er
from sglang.srt.weg2 import wake_credit as wc
from sglang.srt.weg2 import wake_credit_pd as pd
from sglang.srt.weg2 import wake_credit_pd_refs as refs
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=40, suite="base-a-test-cpu")

FIX = os.path.join(os.path.dirname(__file__), "fixtures", "wake_credit_pd_h34")
MODEL = "Qwen3.8-Flash-Next-INT4-Mixed-AutoRound-Minachist"
P_CARD = (1, 0, 2)
SLOT_MIB = 1297637376 / 512 / (1 << 20)
SPLIT, CHUNK, N_LAYERS = (29, 11, 8), 3, 48
BASE = (0.06, 0.44, 0.365)
EDGE_K = (0.06, 0.51, 0.48)   # H30 Stufe K, Boot x145
P_ROWS = {"x141": (166, 263, 232), "x144": (166, 263, 408)}
FR_P = {"x141": (0.26, 0.45, 0.39), "x144": (0.26, 0.45, 0.733887)}  # x144: H25 draft post
E_D, S_D = (193, 145, 177), (82, 48, 48)


def _ref(boot, flip):
    def rd(kind):
        with open(os.path.join(FIX, "fnFL2%s.%s.lines" % (boot, kind))) as fh:
            return fh.read()

    return pd.pd_reference_from_logs(rd("P"), rd("D"), rd("front"),
                                     source="fnFL2%s/%d" % (boot, flip), p_card=P_CARD, flip=flip)


def _d_rows(fr):
    return [er.buffer_rows(local_experts=e, fraction=f, scratch_rows=s)
            for e, f, s in zip(E_D, fr, S_D)]


def _d_res(fr):
    return [er.resident_rows(e, f) for e, f in zip(E_D, fr)]


def _p_res(boot):
    return [er.resident_rows(512, f) for f in FR_P[boot]]


def _plan(boot, fr_d, **kw):
    return pd.plan_wake_credit_pd(
        model=MODEL, p_split=SPLIT, chunk_layers=CHUNK, n_layers=N_LAYERS, p_card=P_CARD,
        d_ratio="183,137,168", draft_on_p=(boot == "x141"), p_rows=P_ROWS[boot],
        d_rows=_d_rows(fr_d), slot_mib=SLOT_MIB, label="D(dry)", p_resident=_p_res(boot),
        d_resident=_d_res(fr_d), **kw)


def _planned(boot, flip, fr_d):
    return pd.planned_reference_pd(
        pd.reference_from_dict(refs.REFERENCES["fnFL2%s/%d" % (boot, flip)]), p_rows=P_ROWS[boot],
        d_rows=_d_rows(fr_d), slot_mib=SLOT_MIB, p_split=SPLIT, chunk_layers=CHUNK,
        n_layers=N_LAYERS, p_resident=_p_res(boot), d_resident=_d_res(fr_d))


TAIL = ("weights_4", "weights_5", "weights_6", "weights_7", "weights_8", "weights")


def _reproduces(ref, run):
    """Der Schwanz auf TP2 wie gemessen: Kredit je Tag +-25 ms, Warten an
    weights_6 +-30 ms, an weights_4 hoechstens 150 ms, Leg-Ende +-30 ms gegen
    PP0s letzte Pause (die Kette, die den Flip traegt)."""
    for t in TAIL:
        m, g = ref.metal[(2, t)], run.tag(2, t)
        if g.grant_ms is None or abs(g.grant_ms - m.grant_ms) > 25:
            return False
    if abs(run.tag(2, "weights_6").waited_ms - ref.metal[(2, "weights_6")].waited_ms) > 30:
        return False
    if run.tag(2, "weights_4").waited_ms > 150:
        return False
    return abs(run.leg_ms - ref.metal_publish[(0, "weights")]) <= 30


# --------------------------------------------------------------------------- Messung


@pytest.mark.parametrize("key", sorted(refs.REFERENCES))
def test_the_shipped_references_are_the_logs_own_measurement(key):
    boot, flip = key[len("fnFL2"):].split("/")
    got = pd.reference_to_dict(_ref(boot, int(flip)))
    assert got == refs.REFERENCES[key]
    assert got["p_resident"] == [er.resident_rows(512, f) for f in FR_P[boot]]
    assert got["d_resident"] == _d_res(BASE)


def test_the_x141_tail_is_what_the_logs_say():
    ref = _ref("x141", 1)
    # the finding, as the D log wrote it (WEG2-VRAM-CREDIT ... waited=)
    assert ref.metal[(2, "weights_4")].waited_ms == 73
    assert ref.metal[(2, "weights_6")].waited_ms == 472
    assert ref.metal[(2, "weights_6")].free_mib == 2231
    assert ref.d_floor[2] == 701 and ref.free[2] == 7825
    # PP2's releases on nvml2 as its tms census says, draft the biggest
    assert dict(ref.p_release[2]) == {"weights": 1866, "weights_13": 1462, "weights_14": 2202,
                                      "weights_15": 2138, "weights_draft": 2770}
    # the draft lane: PP2 -> TP0 (p4), 2754 MiB in 461 ms
    lane = [ln for ln in ref.lanes if ln.stage == 2 and ln.tag == "weights_draft"]
    assert [(ln.rank, round(ln.mib), round(ln.ms)) for ln in lane] == [(0, 2754, 461)]


# --------------------------------------------------------------------------- Modell = Metall


@pytest.mark.parametrize("flip", [1])
def test_the_simulation_reproduces_the_x141_tail(flip):
    ref = _ref("x141", flip)
    run = pd.simulate_pd(ref)
    assert run.complete
    assert _reproduces(ref, run), [(t, run.tag(2, t).grant_ms, ref.metal[(2, t)].grant_ms)
                                   for t in TAIL]
    # PP2's releases on nvml2 land when the metal published them
    for t in ("weights_13", "weights_14", "weights_15", "weights_draft"):
        assert abs(run.publish_ms[(2, t)] - ref.metal_publish[(2, t)]) <= 25, t
    # the waits of the finding: 73 -> model 50..150, 472 -> model within 30
    assert 50 <= run.tag(2, "weights_4").waited_ms <= 150
    assert abs(run.tag(2, "weights_6").waited_ms - 472) <= 30


def test_the_h25_form_has_no_tail():
    ref = _ref("x144", 1)
    run = pd.simulate_pd(ref)
    assert run.complete
    assert max(m.waited_ms for (r, _t), m in ref.metal.items() if r == 2) <= 12
    assert run.wait_sum(2) <= 30
    assert pd.credit_cost_ms(ref, 2) == 0


def test_the_tail_costs_the_leg_what_pp0_paid():
    """Metall: PP0 weights_6 deposit 203 ms gegen 111-123 ms seiner ungestoerten
    Nachbarn -> ~85 ms auf der Kette, die den Flip traegt."""
    ref = _ref("x141", 1)
    cost = pd.credit_cost_ms(ref, 2)
    assert 60 <= cost <= 140
    assert pd.credit_cost_ms(ref, 0) == 0 and pd.credit_cost_ms(ref, 1) == 0


@pytest.mark.parametrize("mutant", [
    dict(depth=0),                  # staging ring not held on the card
    dict(runahead=99),              # no collect run-ahead bound
    dict(unbounded_cards=(2,)),     # no corridor floor / no credit on nvml2
    dict(alone_fraction=1.0),       # a late collector alone costs the whole lane again
    dict(alone_fraction=0.0),       # a late collector costs nothing
    dict(workers=1),                # one collect thread per D rank
])
def test_a_mutant_model_does_not_reproduce_the_metal(mutant):
    ref = _ref("x141", 1)
    assert _reproduces(ref, pd.simulate_pd(ref))
    assert not _reproduces(ref, pd.simulate_pd(ref, **mutant))


# --------------------------------------------------------------------------- Ordnung


def test_the_timed_order_takes_the_tail_off_the_x141_leg():
    main, first = _planned("x141", 1, BASE), _planned("x141", 0, BASE)
    chosen, run, base, why = pd.best_order_pd(main, also=[first])
    assert chosen != list(main.order) and sorted(chosen) == sorted(main.order)
    assert chosen[-1] == "weights"
    # only PP2's OWN bands moved (weights_13 is shared with PP1 and stays)
    moved = {t for a, t in zip(main.order, chosen) if a != t}
    assert moved <= {"weights_14", "weights_15", "weights_draft"} | {
        t for t in main.order if float(main.p_release[2].get(t, 0)) == 0}
    assert chosen.index("weights_13") == list(main.order).index("weights_13")
    # the leg: the whole credit cost comes off (model), TP2 gets weights_6 before PP0 needs it
    assert base.leg_ms - run.leg_ms >= 60, why
    assert run.tag(2, "weights_6").grant_ms < base.tag(2, "weights_6").grant_ms
    # the 97k flip of the same form is not made longer
    f0 = pd.simulate_pd(first, chosen)
    assert f0.leg_ms <= pd.simulate_pd(first).leg_ms + pd.MIN_GAIN_MS / 4
    # and the H14 fixpoint finds no credit cycle in it
    table = pd.untimed_table_pd(main)
    new, note = wc.front_order(chosen, table, free_mib=dict(main.free),
                               floor_mib={1: 767.0, 0: 700.0, 2: 701.0}, double_staging=False)
    assert new == chosen and "unchanged" in note


def test_a_mutant_order_search_does_not_shorten_the_leg():
    """Der Mutant, der nur die Plaetze der geteilten Baender tauscht (die Suche
    ohne die eigenen Baender der knappen Stufe), findet keinen Gewinn."""
    main = _planned("x141", 1, BASE)
    stage = 2
    orig = pd._stage_bands
    try:
        pd._stage_bands = lambda ref, order, s: []  # noqa: E731 -- the mutant
        chosen, run, base, why = pd.best_order_pd(main)
    finally:
        pd._stage_bands = orig
    assert chosen == list(main.order) and "kept" in why
    assert pd.best_order_pd(main)[0] != list(main.order)
    assert stage == 2


def test_the_proven_h25_form_gets_no_recommendation():
    plan = _plan("x144", BASE, apply=True)
    assert plan.refusal is None
    assert "P->D" in plan.front_plan and "P->D-order" not in plan.front_plan
    assert any("costs the given order's leg 0 ms" in ln for ln in plan.lines)


def test_the_plan_from_x144_predicts_x145():
    """x145 (K in der H25-Form) lief nach dem Bau: der Plan aus x144 +
    Pufferregel trifft seine Lane-Bytes auf 1 %, seine drei Waende auf TP2
    (weights_11, weights_3, weights_7) am richtigen Tag, und die empfohlene
    Ordnung kuerzt auch das an x145 selbst geeichte Modell."""
    plan1 = _planned("x144", 1, EDGE_K)
    for flip in (0, 1):
        metal = _ref("x145", flip)
        planned = _planned("x144", flip, EDGE_K)
        got = {(ln.stage, ln.rank, ln.tag): ln.mib for ln in planned.lanes}
        for ln in metal.lanes:
            assert abs(got[(ln.stage, ln.rank, ln.tag)] - ln.mib) <= 0.01 * ln.mib + 1, ln
    metal = _ref("x145", 1)
    walls = {t for (r, t), m in metal.metal.items() if r == 2 and m.waited_ms > 50}
    assert walls == {"weights_11", "weights_3", "weights_7"}
    run = pd.simulate_pd(plan1)
    model_walls = {t.tag for t in run.rank_tags(2) if t.need_mib > 0 and t.waited_ms > 50}
    assert model_walls == walls
    assert abs(run.tag(2, "weights_7").waited_ms - metal.metal[(2, "weights_7")].waited_ms) <= 40
    # the recommendation made from x144 shortens the x145-calibrated leg, never the 97k flip
    chosen, _run, _base, why = pd.best_order_pd(plan1, also=[_planned("x144", 0, EDGE_K)])
    assert chosen != list(plan1.order), why
    own = pd.simulate_pd(metal)
    assert pd.simulate_pd(metal, chosen).leg_ms <= own.leg_ms - 30
    first = _ref("x145", 0)
    assert pd.simulate_pd(first, chosen).leg_ms <= pd.simulate_pd(first).leg_ms + 5


# --------------------------------------------------------------------------- Dry-Run-Zeilen


def _card2(plan):
    return [ln for ln in plan.lines if " P->D D(dry) card2 D TP2/P PP2: " in ln]


def test_the_dry_run_line_for_the_proven_form_matches_x141():
    plan = _plan("x141", BASE, apply=False)
    (line,) = _card2(plan)
    assert line.startswith(wc.MARKER + " P->D ")
    assert "free 7825 - floor 701 + freigegeben 10438 - verbraucht 13990 -> FERTIG" in line
    assert "groesstes 466 ms bei weights_6" in line
    assert "weights_draft 2770@1243" in line
    cost = float(line.split("kostet das Leg ")[1].split(" ms")[0])
    assert 60 <= cost <= 140
    assert "P->D-order" in plan.front_plan
    assert any("Front faehrt die gegebene" in ln for ln in plan.lines)


def test_the_edge_k_is_judged_per_form():
    """K (FR_D 0.06,0.51,0.48): +20 Zeilen TP2 = +145 MiB je Tag (gemessen x145:
    +132), der On-card-Ring auf nvml2 +290 MiB. Mit Draft auf P kostet der
    nvml2-Kredit die kurzen Flips ~0,5 s Leg und der 97k-Vergleichsflip steht;
    in der H25-Form ~70 ms bei ~15 MiB engster Luft (x145: +76..+112 ms auf PP0s
    Kette gegen x144)."""
    k141, k144 = _plan("x141", EDGE_K, apply=False), _plan("x144", EDGE_K, apply=False)
    assert k141.refusal is None and k144.refusal is None
    (l141,) = _card2(k141)
    (l144,) = _card2(k144)
    assert "verbraucht 16310 -> FERTIG" in l141 and "verbraucht 16310 -> FERTIG" in l144
    cost141 = float(l141.split("kostet das Leg ")[1].split(" ms")[0])
    cost144 = float(l144.split("kostet das Leg ")[1].split(" ms")[0])
    assert cost141 >= 300 and 30 <= cost144 <= 150
    luft144 = float(l144.split("engste Luft ")[1].split(" MiB")[0])
    assert 0 <= luft144 < 50
    assert any("Vergleichsflip fnFL2x141/0+plan, Ordnung gegeben -- Leg STEHT" in ln
               for ln in k141.lines)
    assert "P->D-order" in k144.front_plan


def test_a_card_that_cannot_hold_the_d_tags_is_refused():
    plan = _plan("x144", (0.06, 0.44, 0.90), apply=False)
    assert plan.refusal is not None
    assert plan.refusal.startswith(wc.REFUSAL_CODE + ": P->D")
    assert "D TP2 steht bei" in plan.refusal and "FEHLT" in plan.refusal
    assert "D TP0 steht" not in plan.refusal  # only a rank at its credit gate is short
    assert any("STEHT bei" in ln for ln in _card2(plan))


def test_a_foreign_geometry_is_named_not_graded():
    plan = pd.plan_wake_credit_pd(
        model=MODEL, p_split=(24, 12, 12), chunk_layers=CHUNK, n_layers=N_LAYERS, p_card=P_CARD,
        d_ratio="183,137,168", draft_on_p=True, p_rows=P_ROWS["x141"], d_rows=_d_rows(BASE),
        slot_mib=SLOT_MIB, label="D", apply=False)
    assert plan.refusal is None and plan.front_plan is None
    assert "ENTFAELLT" in plan.lines[0] and "p_split" in plan.lines[0]


# --------------------------------------------------------------------------- Front


def _card_free(nvml, uuid, free):
    from sglang.srt.weg2.front import CardFree

    return CardFree(nvml_index=nvml, uuid=uuid, free_mib=free, reserved_mib=0)


def test_the_front_applies_the_recommendation_only_when_asked_and_exact():
    from sglang.srt.environ import envs
    from sglang.srt.weg2 import front

    plan = _plan("x141", BASE, apply=True).front_plan
    rec = plan["P->D-order"]
    given, timed = rec["given"], rec["timed"]
    cards = [_card_free(c, "u%d" % c, f) for c, f in {0: 9333, 1: 11300, 2: 7825}.items()]
    floors = {"u0": 700, "u1": 767, "u2": 701}.get
    assert envs.SGLANG_WEG2_ENABLE_PD_TIMED_ORDER.get() is False
    new, why = front.timed_pause_order(given, "rr", plan, "P", "D", cards, floor_of=floors)
    assert new == given and "OFF" in why
    with envs.SGLANG_WEG2_ENABLE_PD_TIMED_ORDER.override(True):
        new, why = front.timed_pause_order(given, "rr", plan, "P", "D", cards, floor_of=floors)
        assert new == timed and "applied" in why
        # a live order the planner did not time is left alone
        other = list(given)
        other[1], other[2] = other[2], other[1]
        new, why = front.timed_pause_order(other, "rr", plan, "P", "D", cards, floor_of=floors)
        assert new == other and "SKIPPED" in why
        # the other direction is never touched
        assert front.timed_pause_order(given, "rr", plan, "D", "P", cards,
                                       floor_of=floors) == (given, "rr")
        # the H14 fixpoint still keeps the given P->D order unchanged
        new, why = front.credit_pause_order(given, "rr", plan, "P", "D", cards, floor_of=floors)
        assert new == given and "unchanged" in why


# --------------------------------------------------------------------------- Launcher


def _launcher_ns(tmp_path, fr_d):
    cfg = {"text_config": {"num_hidden_layers": 48, "vocab_size": 248320,
                           "hidden_size": 2560}}
    (tmp_path / MODEL).mkdir(parents=True)
    (tmp_path / MODEL / "config.json").write_text(json.dumps(cfg))
    return types.SimpleNamespace(
        model=str(tmp_path / MODEL),
        extra_d=("--rank-tp-ratio 1,0,0 --rank-moe-ratio 183,137,168 "
                 "--rank-moe-resident-fraction " + fr_d),
        env_d=("SGLANG_MOE_POOL_STAGING=8;SGLANG_MOE_SCRATCH_SLOTS=82,48,48;"
               "SGLANG_UNEVEN_MOE_EXPERT_SHARD=1"),
        extra_p="--rank-moe-resident-fraction 0.26,0.45,0.39",
        env_p="SGLANG_MOE_SCRATCH_SLOTS=32",
        pp_cut_expert_device_fraction="",
        d_foreign_context_mib="", d_nontorch_mib="", d_reserve_mib="",
        d_residency_reference_logs="", wake_credit_reference_logs="",
    )


def test_the_launcher_prints_both_directions_and_hands_the_front_its_table(tmp_path, monkeypatch):
    from sglang.srt.environ import envs
    from sglang.srt.planner import pp_cut
    from sglang.srt.weg2 import launcher

    monkeypatch.setattr(
        pp_cut, "checkpoint_weight_terms",
        lambda _p: types.SimpleNamespace(expert_layer_weight_bytes=1297637376.0,
                                         num_experts=512, n_layers=48))
    cards = [launcher.Card(1, "u1", "RTX 5090", 32607),
             launcher.Card(0, "u0", "RTX 3080", 20480),
             launcher.Card(2, "u2", "RTX 3080", 20480)]
    budgets = (29368, 18664, 18656)  # x141 live "budget D group=D"
    lines = []
    with envs.SGLANG_WEG2_DRAFT_ON_P.override(True):
        launcher.log_d_rank_vram_solve(
            _launcher_ns(tmp_path / "a", "0.06,0.44,0.365"), cards, budgets, lines.append,
            "D(dry, expectation)", p_split=[29, 11, 8], chunk_layers=3)
    pdl = [ln for ln in lines if ln.startswith(wc.MARKER + " P->D ")]
    assert any("card2 D TP2/P PP2" in ln and "FERTIG" in ln for ln in pdl), pdl
    assert any("fnFL2x141/1" in ln for ln in pdl)
    plan = launcher._WAKE_CREDIT_FRONT_PLAN
    assert plan and "D->P" in plan and [p["card"] for p in plan["P->D"]] == [1, 0, 2]
    assert "P->D-order" in plan


def test_a_short_p_to_d_wake_refuses_the_boot_unless_switched_off(tmp_path, monkeypatch):
    from sglang.srt.environ import envs
    from sglang.srt.planner import pp_cut
    from sglang.srt.weg2 import launcher

    monkeypatch.setattr(
        pp_cut, "checkpoint_weight_terms",
        lambda _p: types.SimpleNamespace(expert_layer_weight_bytes=1297637376.0,
                                         num_experts=512, n_layers=48))
    monkeypatch.setattr(
        pd, "plan_wake_credit_pd",
        lambda **kw: pd.WakeCreditPlanPD(lines=("x",), refusal="W126 Weg2WakeCreditCycle: P->D t",
                                         front_plan={"P->D": []}))
    cards = [launcher.Card(1, "u1", "RTX 5090", 32607),
             launcher.Card(0, "u0", "RTX 3080", 20480),
             launcher.Card(2, "u2", "RTX 3080", 20480)]
    budgets = (29368, 18664, 18656)
    with pytest.raises(launcher.Weg2LaunchRefused, match=r"^W126 Weg2WakeCreditCycle: P->D"):
        launcher.log_d_rank_vram_solve(
            _launcher_ns(tmp_path / "a", "0.06,0.44,0.365"), cards, budgets, [].append,
            "D", p_split=[29, 11, 8], chunk_layers=3)
    lines = []
    with envs.SGLANG_WEG2_ENABLE_PD_CREDIT_REFUSAL.override(False):
        launcher.log_d_rank_vram_solve(
            _launcher_ns(tmp_path / "b", "0.06,0.44,0.365"), cards, budgets, lines.append,
            "D", p_split=[29, 11, 8], chunk_layers=3)
    assert any("SGLANG_WEG2_ENABLE_PD_CREDIT_REFUSAL=0" in ln for ln in lines)


def test_the_levers_are_registered():
    from sglang.srt.environ import envs

    assert envs.SGLANG_WEG2_ENABLE_PD_TIMED_ORDER.get() is False
    assert envs.SGLANG_WEG2_ENABLE_PD_CREDIT_REFUSAL.get() is True
