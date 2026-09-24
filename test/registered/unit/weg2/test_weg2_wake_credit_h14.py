# SPDX-License-Identifier: Apache-2.0
"""fnFL2 H14 -- der Kreditbedarf des ersten Wakes (D schlaeft, P wacht).

DER BUG, schwarz-weiss: die bewiesene Form (FR_P 0.26/0.45/0.39, FR_D
0.06/0.44/0.365) flippt seit x104. Mit hoeherer Residenz starb der ERSTE Wake
nach 4 s an W109 (x114c: "sleeper1 blocked depositing weights_14 to waker2 ->
sleeper2 blocked depositing weights_10 to waker1"; x114d: dasselbe mit der
5090). Der Planer preiste Residenz (H8) und P-Puffer (H5), aber keinen Weg:
der Schlaefer auf Karte A pausiert den Tag seines Wakers erst, nachdem er einen
Tag an einen Waker auf Karte B deponiert hat -- und der wartet auf den
Schlaefer auf Karte B.

Alle Zahlen hier sind Logzeilen: ``fixtures/wake_credit_h14/`` sind woertliche
Auszuege aus /spinning/evidence-665-f1/boot_weg2_fnFL2x{104,113,114c,114d,
115,116}_*.{P,D,front}.log (die Zeilen, die ``reference_from_logs`` liest).
x104/x113/x114c/x114d liefen VOR H11 (Staging doppelt abgezogen), x115/x116
auf 2f4dcf6893 (H11).
"""

import json
import os
import types

import pytest

from sglang.srt.planner import expert_residency as er
from sglang.srt.weg2 import wake_credit as wc
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=20, suite="base-a-test-cpu")

FIX = os.path.join(os.path.dirname(__file__), "fixtures", "wake_credit_h14")
MODEL = "Qwen3.8-Flash-Next-INT4-Mixed-AutoRound-Minachist"
P_CARD = (1, 0, 2)  # WEG2-FLIP-ORDER MAP: nvml [1, 0, 2] in stage order
SLOT_MIB = 1297637376 / 512 / (1 << 20)  # checkpoint: experts / 512 / MiB
SPLIT, CHUNK, N_LAYERS = (29, 11, 8), 3, 48
#: Metall: kam der erste Wake durch (WEG2-FLIP-CHUNK epoch=0 im Front-Log)?
METAL = {"x104": True, "x113": True, "x114c": False, "x114d": False,
         "x115": True, "x116": True}
H11 = {"x115", "x116"}


def _ref(tag):
    def rd(kind):
        with open(os.path.join(FIX, "fnFL2%s.%s.lines" % (tag, kind))) as fh:
            return fh.read()

    return wc.reference_from_logs(rd("P"), rd("D"), rd("front"), source="fnFL2" + tag,
                                  p_card=P_CARD)


def _p_rows(fr, scratch=32):
    return [er.buffer_rows(local_experts=512, fraction=f, scratch_rows=scratch) for f in fr]


def _d_rows(fr):
    return [er.buffer_rows(local_experts=e, fraction=f, scratch_rows=s)
            for e, f, s in zip((193, 145, 177), fr, (44, 48, 48))]


def _planned(fp, fd, ref=None):
    return wc.planned_cards(ref or wc.REFERENCE_FNFL2X114D, p_rows=_p_rows(fp),
                            d_rows=_d_rows(fd), slot_mib=SLOT_MIB, p_split=SPLIT,
                            chunk_layers=CHUNK, n_layers=N_LAYERS)


def test_the_shipped_reference_is_the_logs_own_measurement():
    got = _ref("x114d")
    ref = wc.REFERENCE_FNFL2X114D
    assert dict(got.free) == dict(ref.free)
    assert got.order == ref.order
    assert got.p_rows == ref.p_rows == (263, 391, 391)
    assert got.d_rows == ref.d_rows == (84, 128, 128)
    assert got.p_floor == ref.p_floor
    assert [dict(t) for t in got.p_tags] == [dict(t) for t in ref.p_tags]
    assert [dict(t) for t in got.d_tags] == [dict(t) for t in ref.d_tags]
    for a, b in zip(got.d_oncard, ref.d_oncard):
        assert a.keys() == b.keys()
        assert all(abs(a[k] - b[k]) < 0.01 for k in a)


@pytest.mark.parametrize("tag", sorted(METAL))
def test_the_simulation_says_what_the_metal_did(tag):
    """Jeder Boot unter SEINER Kredit-Regel: vor H11 zog ``wait_for`` das
    Staging doppelt ab, ab H11 nicht. Durchgelaufen/Zyklus == Metall."""
    ref = _ref(tag)
    run = wc.simulate(ref.order, wc.wake_cards(ref), double_staging=tag not in H11)
    assert run.complete is METAL[tag], [wc.card_line(c, s) for c, s in
                                       zip(wc.wake_cards(ref), run.cards)]
    if not run.complete:
        assert run.chain, "a stuck first wake that is no cycle"


def test_x114c_is_the_exact_w109_chain():
    """x114c P.log: 'W109 ... card=GPU-5c648f96 (nvml0) tag=weights_10: sleeper1
    blocked depositing weights_14 to waker2 -> sleeper2 blocked depositing
    weights_10 to waker1' -- Rang 1 = Karte 0, Rang 2 = Karte 2."""
    ref = _ref("x114c")
    run = wc.simulate(ref.order, wc.wake_cards(ref), double_staging=True)
    assert set(run.chain) == {(0, 2, "weights_14"), (2, 0, "weights_10")}
    by = {c.card: c for c in run.cards}
    assert by[0].waker_tag == "weights_10" and by[0].need_mib == 4210
    assert by[2].waker_tag == "weights_14" and by[2].need_mib == 4210


@pytest.mark.parametrize("tag", ["x104", "x113", "x115", "x116"])
def test_the_credit_order_keeps_every_proven_order(tag):
    """Die Kredit-Ordnung ist fuer die bewiesene Form die Identitaet -- der
    Default darf an sein."""
    ref = _ref(tag)
    order, run, why = wc.credit_order(list(ref.order), wc.wake_cards(ref),
                                      double_staging=tag not in H11)
    assert order == list(ref.order), why
    assert run.complete and "unchanged" in why


def test_the_credit_order_funds_a_form_the_round_robin_cycles():
    """P 0.45/0.80/0.80 bei D 0.207/0.55/0.45: die Round-Robin-Ordnung endet
    im Zyklus, eine Umordnung (jede Karte bekommt ihre grossen Tags erst, wenn
    ihr Schlaefer genug davor freigegeben hat) laeuft durch."""
    cards = _planned((0.45, 0.80, 0.80), (0.207, 0.55, 0.45))
    order = list(wc.REFERENCE_FNFL2X114D.order)
    assert not wc.simulate(order, cards).complete
    new, run, why = wc.credit_order(order, cards)
    assert run.complete, why
    assert sorted(new) == sorted(order) and new != order
    assert new[-1] == "weights"  # der Basis-Tag schliesst den Schlaf
    assert wc.simulate(new, cards).complete


def test_the_delta_on_the_reference_predicts_x114c():
    """Referenz x114d + Pufferregel -> x114c: Tags je Stufe/Rang auf wenige MiB,
    free beim Flip-Start auf < 60 MiB, und derselbe Verdikt."""
    got = {c.card: c for c in _planned((0.45, 0.95, 0.996), (0.207, 0.60, 0.48))}
    meas = {c.card: c for c in wc.wake_cards(_ref("x114c"))}
    for card, tag in ((0, "weights_10"), (2, "weights_14"), (1, "weights_1")):
        assert abs(got[card].demand_mib[tag] - meas[card].demand_mib[tag]) < 10
        assert abs(got[card].release_mib[tag] - meas[card].release_mib[tag]) < 10
    for card in (0, 1, 2):
        assert abs(got[card].free_mib - meas[card].free_mib) < 60


def test_the_riegel_refuses_x114c_with_the_arithmetic_per_card():
    plan = wc.plan_wake_credit(
        model="/m/" + MODEL, p_split=SPLIT, chunk_layers=CHUNK, n_layers=N_LAYERS,
        p_card=P_CARD, d_ratio="183,137,168", p_rows=_p_rows((0.45, 0.95, 0.996)),
        d_rows=_d_rows((0.207, 0.60, 0.48)), slot_mib=SLOT_MIB, label="D(dry)",
        reorder=True, double_staging=False)
    assert plan.refusal is not None and plan.refusal.startswith("W126 Weg2WakeCreditCycle")
    assert "im Kreditzyklus" in plan.refusal
    assert "card0" in plan.refusal and "card2" in plan.refusal and "FEHLT" in plan.refusal


def test_the_riegel_passes_the_proven_form_and_form_a():
    """Die bewiesene Form (x115, H11) und Form A (FR_P 0.26/0.45/0.39 mit FR_D
    0.207/0.55/0.45) laufen durch, beide mit Luft auf jeder Karte."""
    for fp, fd in (((0.26, 0.45, 0.39), (0.06, 0.44, 0.365)),
                   ((0.26, 0.45, 0.39), (0.207, 0.55, 0.45))):
        plan = wc.plan_wake_credit(
            model=MODEL, p_split=SPLIT, chunk_layers=CHUNK, n_layers=N_LAYERS,
            p_card=P_CARD, d_ratio="183,137,168", p_rows=_p_rows(fp), d_rows=_d_rows(fd),
            slot_mib=SLOT_MIB, label="D", reorder=True, double_staging=False)
        assert plan.refusal is None, plan.lines
        run = wc.simulate(wc.REFERENCE_FNFL2X114D.order, _planned(fp, fd))
        assert all(c.min_headroom_mib > 100 for c in run.cards), plan.lines
        assert "unchanged" in plan.lines[1]


def test_a_foreign_geometry_is_named_not_graded():
    plan = wc.plan_wake_credit(
        model=MODEL, p_split=(24, 12, 12), chunk_layers=CHUNK, n_layers=N_LAYERS,
        p_card=P_CARD, d_ratio="183,137,168", p_rows=[263] * 3, d_rows=[84, 128, 128],
        slot_mib=SLOT_MIB, label="D", reorder=True, double_staging=False)
    assert plan.refusal is None and plan.front_plan is None
    assert "ENTFAELLT" in plan.lines[0] and "p_split" in plan.lines[0]


def _launcher_ns(tmp_path, fr_p, fr_d):
    cfg = {"text_config": {"num_hidden_layers": 48, "vocab_size": 248320,
                           "hidden_size": 2560}}
    (tmp_path / MODEL).mkdir(parents=True)
    (tmp_path / MODEL / "config.json").write_text(json.dumps(cfg))
    return types.SimpleNamespace(
        model=str(tmp_path / MODEL),
        extra_d=("--rank-tp-ratio 1,0,0 --rank-moe-ratio 183,137,168 "
                 "--rank-moe-resident-fraction " + fr_d),
        # x114c/d liefen vor H39 (H50: der Zustand waehlt die Referenzen)
        env_d=("SGLANG_MOE_POOL_STAGING=8;SGLANG_MOE_SCRATCH_SLOTS=44,48,48;"
               "SGLANG_UNEVEN_MOE_EXPERT_SHARD=1;SGLANG_WEG2_DENSE_REPACK_OUTSIDE_POOL=0"),
        extra_p="--rank-moe-resident-fraction " + fr_p,
        env_p="SGLANG_MOE_SCRATCH_SLOTS=32",
        pp_cut_expert_device_fraction="",
        d_foreign_context_mib="", d_nontorch_mib="", d_reserve_mib="",
        d_residency_reference_logs="", wake_credit_reference_logs="",
    )


def test_the_launcher_refuses_x114c_before_a_rank_loads(tmp_path, monkeypatch):
    """Die Naht: ``log_d_rank_vram_solve`` (beide budgets_d-Stellen, auch der
    Dry-Run) laesst x114c's argv durch H8 (D passt ins Budget) und verweigert
    es dann mit W126; Form A geht durch und die Front bekommt ihre Tabelle."""
    from sglang.srt.planner import pp_cut
    from sglang.srt.weg2 import launcher

    monkeypatch.setattr(
        pp_cut, "checkpoint_weight_terms",
        lambda _p: types.SimpleNamespace(expert_layer_weight_bytes=1297637376.0,
                                         num_experts=512, n_layers=48))
    cards = [launcher.Card(1, "u1", "RTX 5090", 32607),
             launcher.Card(0, "u0", "RTX 3080", 20480),
             launcher.Card(2, "u2", "RTX 3080", 20480)]
    budgets = (29312, 18512, 18504)  # x114c "budget D group=D"
    lines = []
    with pytest.raises(launcher.Weg2LaunchRefused, match=r"^W126 Weg2WakeCreditCycle"):
        launcher.log_d_rank_vram_solve(
            _launcher_ns(tmp_path / "a", "0.45,0.95,0.996", "0.207,0.60,0.48"), cards,
            budgets, lines.append, "D(dry, expectation)", p_split=[29, 11, 8],
            chunk_layers=3)
    assert any("FRACTION-SOLVE D(dry, expectation) rang1" in ln and "PASST" in ln
               for ln in lines)
    assert any(ln.startswith(wc.MARKER) for ln in lines)
    lines.clear()
    launcher.log_d_rank_vram_solve(
        _launcher_ns(tmp_path / "b", "0.26,0.45,0.39", "0.207,0.55,0.45"), cards,
        budgets, lines.append, "D", p_split=[29, 11, 8], chunk_layers=3)
    plan = launcher._WAKE_CREDIT_FRONT_PLAN
    assert plan and [p["card"] for p in plan["D->P"]] == [1, 0, 2]
    ns = types.SimpleNamespace(tag="t", fairness_w_s=45.0, drain_deadline_s=90.0,
                               min_dwell_ms=None, d_admit_max_tokens=None,
                               weg2_weight_source="exchange")
    base = launcher.front_argv_for("py", "/s", 1, 2, {}, [], ns, 0, 0, 8, 8, 1, 1, "D")
    argv = launcher.front_argv_for("py", "/s", 1, 2, {}, [], ns, 0, 0, 8, 8, 1, 1, "D",
                                   wake_credit_plan=plan)
    assert "--wake-credit-plan" not in base
    i = argv.index("--wake-credit-plan")
    assert json.loads(argv[i + 1]) == json.loads(json.dumps(plan))
    assert [a for a in argv if a not in (argv[i], argv[i + 1])] == base


def _card_free(nvml, uuid, free):
    from sglang.srt.weg2.front import CardFree

    return CardFree(nvml_index=nvml, uuid=uuid, free_mib=free, reserved_mib=0)


def test_the_front_reorders_only_a_cycling_order():
    from sglang.srt.environ import envs
    from sglang.srt.weg2 import front

    ref = wc.REFERENCE_FNFL2X114D
    floors = {"u%d" % c: f for c, f in zip(ref.p_card, ref.p_floor)}

    def table(fp, fd):
        return {"D->P": [{"card": c.card, "release": dict(c.release_mib),
                          "demand": dict(c.demand_mib), "oncard": dict(c.oncard_mib)}
                         for c in _planned(fp, fd)]}

    def free_now(fp, fd):
        return [_card_free(c.card, "u%d" % c.card, c.free_mib) for c in _planned(fp, fd)]

    order = list(ref.order)
    cyc = ((0.45, 0.80, 0.80), (0.207, 0.55, 0.45))
    new, why = front.credit_pause_order(order, "rr", table(*cyc), "D", "P", free_now(*cyc),
                                        floor_of=floors.get)
    assert new != order and sorted(new) == sorted(order) and new[-1] == "weights", why
    ok = ((0.26, 0.45, 0.39), (0.207, 0.55, 0.45))
    new, why = front.credit_pause_order(order, "rr", table(*ok), "D", "P", free_now(*ok),
                                        floor_of=floors.get)
    assert new == order and "unchanged" in why
    # the other direction has no table: untouched, no note
    assert front.credit_pause_order(order, "rr", table(*cyc), "P", "D", free_now(*cyc),
                                    floor_of=floors.get) == (order, "rr")
    with envs.SGLANG_WEG2_FLIP_ORDER_CREDIT.override(False):
        new, why = front.credit_pause_order(order, "rr", table(*cyc), "D", "P",
                                            free_now(*cyc), floor_of=floors.get)
    assert new == order and "OFF" in why


def test_the_lever_is_registered_and_on_by_default():
    from sglang.srt.environ import envs

    assert envs.SGLANG_WEG2_FLIP_ORDER_CREDIT.get() is True
