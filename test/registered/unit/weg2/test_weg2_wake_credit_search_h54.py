# SPDX-License-Identifier: Apache-2.0
"""fnFL2 H54 -- die Kredit-Ordnung sucht ueber VOLLE Simulationen.

DER BEFUND: der Dry-Run fnFL2x162 bei FR_P 0.410/0.712 (FR_D 0.06/0.51/0.48,
Scratch D 118/48/48) verweigerte mit W126 "NO order funds the wake (stuck
after 3 of 18 tags)" -- ``fixtures/wake_credit_h14/dry_fnFL2x162_refused_0410
.lines`` sind die woertlichen Zeilen aus
/spinning/evidence-665-f1/dry_fnFL2x162_refused_0410.log. Der gierige Bau in
``credit_order`` prueft PRAEFIXE, und ein Praefix kennt keinen Vorlauf der
Schlaefer: schon ``[weights_0]`` allein traegt Karte 1 nicht (D TP0 gibt 1426
MiB frei, staged 715, P PP0 braucht 2246), obwohl D TP0 im Leg sofort
``weights_1`` weiter freigibt. Die volle Simulation derselben Karten findet
eine tragende Ordnung. Mit SGLANG_WEG2_ENABLE_FLIP_ORDER_CREDIT_SEARCH sucht
``credit_order`` so weiter -- im Riegel UND in der Front (dieselbe Funktion,
die Tabelle traegt die Marke ``D->P-search``).

DIE WURZEL DAHINTER (fnFL2x162 selbst, ``fixtures/wake_credit_h14/fnFL2x162
.*.lines``, woertlich aus boot_weg2_fnFL2x162_7517ff598d_0924_171421.*.log):
die eingebaute Referenz fnFL2x114d sagt fuer Karte 1 beim Flip-Start free 1971
voraus, x162 mass bei denselben D-Zeilen 130/122/133 free 6764. Gegen die
gemessene Referenz laeuft 0.410/0.712 in der GEGEBENEN Ordnung durch.
"""

import os
import types

import pytest

from sglang.srt.environ import envs
from sglang.srt.planner import expert_residency as er
from sglang.srt.weg2 import wake_credit as wc
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=30, suite="base-a-test-cpu")

FIX = os.path.join(os.path.dirname(__file__), "fixtures", "wake_credit_h14")
MODEL = "Qwen3.8-Flash-Next-INT4-Mixed-AutoRound-Minachist"
P_CARD = (1, 0, 2)
SLOT_MIB = 1297637376 / 512 / (1 << 20)
SPLIT, CHUNK, N_LAYERS = (29, 11, 8), 3, 48
#: x162 refused: FR_P nach dem H25-Draft-Posten (PP-CUT draft post: 0.39 -> 0.733887)
FR_P = (0.41, 0.712, 0.733887)
FR_D = (0.06, 0.51, 0.48)
D_SPAN, D_SCRATCH = (192, 144, 176), (118, 48, 48)
LABEL = "D(dry, expectation)"


def _p_rows(fr=FR_P):
    return [er.buffer_rows(local_experts=512, fraction=f, scratch_rows=32) for f in fr]


def _d_rows(fr=FR_D):
    # FRACTION-SOLVE D: E = Spanne + 1 Pad-Zeile
    return [er.buffer_rows(local_experts=s + 1, fraction=f, scratch_rows=sc)
            for s, f, sc in zip(D_SPAN, fr, D_SCRATCH)]


def _plan(search, reference=None, reference_key=None):
    return wc.plan_wake_credit(
        model="/m/" + MODEL, p_split=SPLIT, chunk_layers=CHUNK, n_layers=N_LAYERS,
        p_card=P_CARD, d_ratio="183,137,168", p_rows=_p_rows(), d_rows=_d_rows(),
        slot_mib=SLOT_MIB, label=LABEL, reorder=True, double_staging=False,
        search=search, reference=reference, reference_key=reference_key)


def _dry_lines():
    """Die D->P-Zeilen des Dry-Runs ohne Zeitstempel/Launcher-Praefix."""
    out = []
    with open(os.path.join(FIX, "dry_fnFL2x162_refused_0410.lines")) as fh:
        for ln in fh:
            body = ln.rstrip("\n").split("] WEG2-LAUNCH ", 1)[1]
            out.append(body)
    return out


def _planned_cards():
    return wc.planned_cards(wc.REFERENCE_FNFL2X114D, p_rows=_p_rows(), d_rows=_d_rows(),
                            slot_mib=SLOT_MIB, p_split=SPLIT, chunk_layers=CHUNK,
                            n_layers=N_LAYERS)


def test_the_rows_are_the_dry_runs():
    assert _p_rows() == [242, 397, 408]
    assert _d_rows() == [130, 122, 133]


def test_switch_off_reproduces_the_dry_run_refusal_byte_for_byte():
    """Ohne Schalter: dieselben Zeilen, dieselbe Verweigerung wie x162_refused."""
    dry = _dry_lines()
    plan = _plan(search=False)
    assert list(plan.lines) == dry[:5]
    assert "%s %s" % (wc.MARKER, plan.refusal) == dry[5]
    assert dry[6] == "REFUSED: " + plan.refusal
    assert wc.SEARCH_KEY not in plan.front_plan


def test_switch_on_the_same_form_passes_with_a_funded_order():
    plan = _plan(search=True)
    assert plan.refusal is None, plan.lines
    assert "full-simulation search (H54)" in plan.lines[1]
    assert "Ordnungssuche ueber volle Simulationen AN (H54)" in plan.lines[0]
    assert plan.front_plan[wc.SEARCH_KEY] is True
    chosen = eval(plan.lines[1].split(": order ", 1)[1].split(" -- ", 1)[0])  # noqa: S307
    given = list(wc.REFERENCE_FNFL2X114D.order)
    assert sorted(chosen) == sorted(given) and chosen != given
    assert chosen[-1] == "weights"
    run = wc.simulate(chosen, _planned_cards())
    assert run.complete
    # ohne Reserve: jede Karte traegt jeden Grant (engste Luft >= 0), nichts zurueckgelegt
    assert all(c.min_headroom_mib is not None and c.min_headroom_mib >= 0 for c in run.cards)
    for ln in plan.lines[2:5]:
        assert "FERTIG" in ln


def test_the_greedy_prefix_build_is_what_fails():
    """Der Defekt ist der Praefix-Bau: jede einzelne Anfangsposition ausser
    weights_9 ist als Praefix 'nicht tragend', obwohl die volle Ordnung traegt."""
    cards = _planned_cards()
    order = list(wc.REFERENCE_FNFL2X114D.order)
    assert not wc.simulate(order, cards).complete
    assert not wc.simulate(["weights_0"], cards).complete
    _o, _run, why = wc.credit_order(order, cards)
    assert "NO order funds the wake (stuck after 3 of 18 tags)" in why
    new, run, why = wc.credit_order(order, cards, search=True)
    assert run.complete and "full-simulation search (H54)" in why
    assert not all(wc.simulate(new[:k], cards).complete for k in range(1, len(new)))


def test_mutant_prefix_gated_search_keeps_the_refusal(monkeypatch):
    """Mutant: eine Suche, die nur Zuege nimmt, deren Praefix allein traegt
    (die Praefix-Logik des gierigen Baus) -- der Riegel bleibt bei W126. Der
    Test oben faengt diesen Mutanten."""
    real = wc.search_order

    def prefix_gated(order, cards, **kw):
        new, run, moved = real(order, cards, **kw)
        if not all(wc.simulate(new[:k], cards).complete for k in range(1, len(new))):
            return list(order), wc.simulate(list(order), cards), 0
        return new, run, moved

    monkeypatch.setattr(wc, "search_order", prefix_gated)
    plan = _plan(search=True)
    assert plan.refusal is not None and plan.refusal.startswith(wc.REFUSAL_CODE)


@pytest.mark.parametrize("tag", ["x104", "x113", "x115", "x116"])
def test_the_search_leaves_every_proven_order_unchanged(tag):
    def rd(kind):
        with open(os.path.join(FIX, "fnFL2%s.%s.lines" % (tag, kind))) as fh:
            return fh.read()

    ref = wc.reference_from_logs(rd("P"), rd("D"), rd("front"), source=tag, p_card=P_CARD)
    order, run, why = wc.credit_order(list(ref.order), wc.wake_cards(ref), search=True,
                                      double_staging=tag not in ("x115", "x116"))
    assert order == list(ref.order) and "unchanged" in why


def test_the_greedy_order_wins_where_it_exists():
    """H14's Beispiel (P 0.45/0.80/0.80): der gierige Bau findet eine Ordnung --
    die Suche aendert daran nichts."""
    cards = wc.planned_cards(
        wc.REFERENCE_FNFL2X114D,
        p_rows=[er.buffer_rows(local_experts=512, fraction=f, scratch_rows=32)
                for f in (0.45, 0.80, 0.80)],
        d_rows=[er.buffer_rows(local_experts=e, fraction=f, scratch_rows=s)
                for e, f, s in zip((193, 145, 177), (0.207, 0.55, 0.45), (44, 48, 48))],
        slot_mib=SLOT_MIB, p_split=SPLIT, chunk_layers=CHUNK, n_layers=N_LAYERS)
    order = list(wc.REFERENCE_FNFL2X114D.order)
    assert wc.credit_order(order, cards) == wc.credit_order(order, cards, search=True)


def _card_free(nvml, uuid, free):
    from sglang.srt.weg2.front import CardFree

    return CardFree(nvml_index=nvml, uuid=uuid, free_mib=free, reserved_mib=0)


def test_the_front_runs_the_search_the_riegel_promised():
    """Die Naht: der Riegel laesst die Form nur mit der Suche durch, also muss
    die Front sie fahren -- auch wenn IHRE Env den Schalter nicht traegt. Die
    Marke in der Tabelle reicht; ohne Marke und ohne Env bleibt die Ordnung
    mit W126-Notiz stehen (der Mutant 'Front ignoriert die Marke')."""
    from sglang.srt.weg2 import front

    ref = wc.REFERENCE_FNFL2X114D
    floors = {"u%d" % c: f for c, f in zip(ref.p_card, ref.p_floor)}
    free_now = [_card_free(c.card, "u%d" % c.card, c.free_mib) for c in _planned_cards()]
    order = [t for t in ref.order]
    plan = _plan(search=True).front_plan
    with envs.SGLANG_WEG2_ENABLE_FLIP_ORDER_CREDIT_SEARCH.override(False):
        new, why = front.credit_pause_order(order, "rr", plan, "D", "P", free_now,
                                            floor_of=floors.get)
        assert new != order and sorted(new) == sorted(order) and new[-1] == "weights", why
        assert "full-simulation search (H54)" in why
        unmarked = {k: v for k, v in plan.items() if k != wc.SEARCH_KEY}
        kept, why = front.credit_pause_order(order, "rr", unmarked, "D", "P", free_now,
                                             floor_of=floors.get)
        assert kept == order and wc.REFUSAL_CODE in why
    with envs.SGLANG_WEG2_ENABLE_FLIP_ORDER_CREDIT_SEARCH.override(True):
        new2, _why = front.credit_pause_order(order, "rr", unmarked, "D", "P", free_now,
                                              floor_of=floors.get)
    assert new2 == new


def test_the_switch_is_registered_and_off_by_default():
    assert envs.SGLANG_WEG2_ENABLE_FLIP_ORDER_CREDIT_SEARCH.get() is False
    with envs.SGLANG_WEG2_ENABLE_FLIP_ORDER_CREDIT_SEARCH.override(True):
        assert envs.SGLANG_WEG2_ENABLE_FLIP_ORDER_CREDIT_SEARCH.get() is True


def test_the_launcher_reads_the_switch(monkeypatch):
    """Die Launcher-Naht (echter Lauf und Dry-Run): ``log_wake_credit_solve``
    mit x162's argv -- ohne Schalter W126 mit den Zeilen des Dry-Runs, mit
    Schalter Durchlass und die Front-Tabelle traegt die Marke."""
    from sglang.srt.weg2 import launcher

    ns = types.SimpleNamespace(
        model="/m/" + MODEL,
        extra_d=("--rank-tp-ratio 1,0,0 --rank-moe-ratio 183,137,168 "
                 "--rank-moe-resident-fraction " + ",".join("%g" % f for f in FR_D)),
        env_d="SGLANG_MOE_SCRATCH_SLOTS=118,48,48;SGLANG_WEG2_DENSE_REPACK_OUTSIDE_POOL=1",
        extra_p="--rank-moe-resident-fraction " + ",".join("%g" % f for f in FR_P),
        env_p="SGLANG_MOE_SCRATCH_SLOTS=32",
        pp_cut_expert_device_fraction=",".join("%g" % f for f in FR_P),
        wake_credit_reference_logs="",
    )
    fits = [types.SimpleNamespace(span=s, n_layers=N_LAYERS, slot_mib=SLOT_MIB, buffer_rows=b,
                                  resident_rows=b - sc)
            for s, b, sc in zip(D_SPAN, _d_rows(), D_SCRATCH)]
    cards = [launcher.Card(1, "u1", "RTX 5090", 32607), launcher.Card(0, "u0", "RTX 3080", 20480),
             launcher.Card(2, "u2", "RTX 3080", 20480)]
    # nur die D->P-Seite ist hier gefragt; P->D (H34) hat ihre eigenen Tests
    monkeypatch.setattr(launcher, "log_wake_credit_solve_pd", lambda *a, **k: None)
    dry = _dry_lines()
    lines = []
    with envs.SGLANG_WEG2_ENABLE_FLIP_ORDER_CREDIT_SEARCH.override(False):
        with pytest.raises(launcher.Weg2LaunchRefused, match=r"^W126 Weg2WakeCreditCycle"):
            launcher.log_wake_credit_solve(ns, cards, fits, lines.append, LABEL,
                                           p_split=list(SPLIT), chunk_layers=CHUNK)
    assert lines == dry[:6]
    lines.clear()
    with envs.SGLANG_WEG2_ENABLE_FLIP_ORDER_CREDIT_SEARCH.override(True):
        launcher.log_wake_credit_solve(ns, cards, fits, lines.append, LABEL,
                                       p_split=list(SPLIT), chunk_layers=CHUNK)
    assert not any("W126" in ln for ln in lines)
    assert launcher._WAKE_CREDIT_FRONT_PLAN[wc.SEARCH_KEY] is True


def _ref162():
    def rd(kind):
        with open(os.path.join(FIX, "fnFL2x162.%s.lines" % kind)) as fh:
            return fh.read()

    return wc.reference_from_logs(rd("P"), rd("D"), rd("front"), source="fnFL2x162",
                                  p_card=P_CARD)


def test_the_measured_x162_reference_passes_0410_in_the_given_order():
    """Die Wurzel: gegen die GEMESSENE Referenz der aktuellen Form (x162, dieselben
    D-Zeilen) laeuft 0.410/0.712 ohne Umordnung und ohne Suche durch; free auf
    Karte 1 beim Flip-Start ist 6764 (die x114d-Referenz sagt 1971)."""
    ref = _ref162()
    assert dict(ref.free) == {0: 3463.0, 1: 6764.0, 2: 2331.0}
    assert ref.p_rows == (217, 360, 408) and ref.d_rows == (130, 122, 133)
    assert "weights_draft" not in ref.order
    stale = {c.card: c.free_mib for c in _planned_cards()}
    assert round(stale[1]) == 1971
    plan = _plan(search=False, reference=ref, reference_key={
        "p_card": P_CARD, "p_split": SPLIT, "chunk_layers": CHUNK})
    assert plan.refusal is None, plan.lines
    assert "unchanged" in plan.lines[1]
    run = wc.simulate(list(ref.order), wc.planned_cards(
        ref, p_rows=_p_rows(), d_rows=_d_rows(), slot_mib=SLOT_MIB, p_split=SPLIT,
        chunk_layers=CHUNK, n_layers=N_LAYERS))
    head = {c.card: round(c.min_headroom_mib) for c in run.cards}
    assert head == {1: 264, 0: 51, 2: 876}
