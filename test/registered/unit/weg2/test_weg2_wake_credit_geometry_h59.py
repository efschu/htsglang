# SPDX-License-Identifier: Apache-2.0
"""fnFL2 H59 -- der erste Wake D->P gegen eine GEMESSENE Referenz kennt deren
P-Schnitt.

DER BEFUND (Dry-Run 26,10,12 attn 6,3,3 gegen ``--wake-credit-reference-logs``
der x162-Logs, Schnitt 29,11,8): der Launcher setzte in den Referenz-Schluessel
den ``p_split`` des GEPLANTEN Boots. Die Pruefung "fremde Geometrie ist kein
Delta der Pufferregel" verglich damit den Plan mit sich selbst und war immer
erfuellt; ``planned_cards`` rechnete die gemessenen P-Tags der ALTEN Karten
weiter (weights_12 mit 2866 MiB auf PP1, obwohl es bei 26,10,12 auf PP2 liegt)
und druckte FERTIG. Die Gegenrichtung P->D (H34, eingebaute Referenz x158)
entfiel richtig mit Namen. Jetzt liest ``reference_from_logs`` die ersten
Layer je PP-Stufe (``MoE expert-offload active on layer N``) und der Schluessel
traegt den Schnitt der REFERENZ.
"""

import os
import types

import pytest

from sglang.srt.environ import envs
from sglang.srt.planner import expert_residency as er
from sglang.srt.weg2 import wake_credit as wc
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=10, suite="base-a-test-cpu")

FIX = os.path.join(os.path.dirname(__file__), "fixtures", "wake_credit_h14")
MODEL = "Qwen3.8-Flash-Next-INT4-Mixed-AutoRound-Minachist"
P_CARD = (1, 0, 2)
SLOT_MIB = 1297637376 / 512 / (1 << 20)
CHUNK, N_LAYERS = 3, 48
FR_D = (0.06, 0.51, 0.48)
D_SPAN, D_SCRATCH = (192, 144, 176), (118, 48, 48)
LABEL = "D(dry, expectation)"


def _texts(tag):
    out = []
    for kind in ("P", "D", "front"):
        with open(os.path.join(FIX, "fnFL2%s.%s.lines" % (tag, kind))) as fh:
            out.append(fh.read())
    return out


def _ref(tag):
    return wc.reference_from_logs(*_texts(tag), source="fnFL2" + tag, p_card=P_CARD)


def test_the_reference_knows_its_own_cut():
    for tag in ("x162", "x114d"):
        ref = _ref(tag)
        assert ref.p_first_layers == (0, 29, 40)
        assert wc.reference_p_split(ref, N_LAYERS) == (29, 11, 8)
    assert wc.REFERENCE_FNFL2X114D.p_first_layers == _ref("x114d").p_first_layers


def test_without_layer_lines_the_cut_is_unknown_not_guessed():
    p, d, f = _texts("x162")
    p = "\n".join(ln.replace("MoE expert-offload active on layer", "MoE expert-offload on")
                  if "PP" in ln and "expert-offload active" in ln else ln
                  for ln in p.splitlines())
    # die Zeilen fehlen dann auch fuer die Pufferzeilen -> die Referenz ist unvollstaendig
    with pytest.raises(ValueError, match="Referenzzeilen fehlen"):
        wc.reference_from_logs(p, d, f, source="x", p_card=P_CARD)
    ref = _ref("x162")
    broken = wc.WakeReference(**{**ref.__dict__, "p_first_layers": (0, 40, 29)})
    assert wc.reference_p_split(broken, N_LAYERS) is None
    assert wc.reference_p_split(wc.WakeReference(**{**ref.__dict__, "p_first_layers": ()}),
                                N_LAYERS) is None


def _run_launcher(split, fr_p, monkeypatch, tmp_path, model=MODEL):
    from sglang.srt.weg2 import launcher

    paths = []
    for kind, text in zip(("P", "D", "front"), _texts("x162")):
        if kind == "front":
            # H87: the reference key now carries the MODEL the logs name; the
            # .lines excerpts dropped the argv, the real x162 front log has it.
            text = ("group P argv: python -m sglang.launch_server --model-path /m/%s\n"
                    % MODEL) + text
        p = tmp_path / ("boot_weg2_fnFL2x162_x.%s.log" % kind)
        p.write_text(text)
        paths.append(str(p))
    ns = types.SimpleNamespace(
        model="/m/" + model,
        extra_d=("--rank-tp-ratio 1,0,0 --rank-moe-ratio 183,137,168 "
                 "--rank-moe-resident-fraction " + ",".join("%g" % f for f in FR_D)),
        env_d="SGLANG_MOE_SCRATCH_SLOTS=118,48,48;SGLANG_WEG2_DENSE_REPACK_OUTSIDE_POOL=1",
        extra_p="--rank-moe-resident-fraction " + ",".join("%g" % f for f in fr_p),
        env_p="SGLANG_MOE_SCRATCH_SLOTS=32",
        pp_cut_expert_device_fraction=",".join("%g" % f for f in fr_p),
        wake_credit_reference_logs=",".join(paths),
    )
    d_rows = [er.buffer_rows(local_experts=s + 1, fraction=f, scratch_rows=sc)
              for s, f, sc in zip(D_SPAN, FR_D, D_SCRATCH)]
    fits = [types.SimpleNamespace(span=s, n_layers=N_LAYERS, slot_mib=SLOT_MIB, buffer_rows=b,
                                  resident_rows=b - sc)
            for s, b, sc in zip(D_SPAN, d_rows, D_SCRATCH)]
    cards = [launcher.Card(1, "u1", "RTX 5090", 32607), launcher.Card(0, "u0", "RTX 3080", 20480),
             launcher.Card(2, "u2", "RTX 3080", 20480)]
    monkeypatch.setattr(launcher, "log_wake_credit_solve_pd", lambda *a, **k: None)
    lines = []
    with envs.SGLANG_WEG2_ENABLE_FLIP_ORDER_CREDIT_SEARCH.override(False):
        launcher.log_wake_credit_solve(ns, cards, fits, lines.append, LABEL,
                                       p_split=list(split), chunk_layers=CHUNK)
    return lines


def test_another_cut_against_the_x162_logs_is_named_not_priced(monkeypatch, tmp_path):
    lines = _run_launcher((26, 10, 12), (0.472, 0.796, 0.468262), monkeypatch, tmp_path)
    assert len(lines) == 1, lines
    assert "ENTFAELLT" in lines[0]
    assert "{'p_split': (29, 11, 8)}" in lines[0] and "{'p_split': (26, 10, 12)}" in lines[0]
    assert not any("FERTIG" in ln for ln in lines)


def test_another_model_against_the_x162_logs_is_named_not_priced(monkeypatch, tmp_path):
    """H87: the x162 logs are INT4 (they name their --model-path); an NVFP4 boot
    was priced against them without any model comparison. Now: ENTFAELLT, named."""
    lines = _run_launcher((29, 11, 8), (0.41, 0.712, 0.733887), monkeypatch, tmp_path,
                          model="Qwen3.8-Flash-Next-NVFP4-nvidia")
    assert len(lines) == 1, lines
    assert "ENTFAELLT" in lines[0] and "'model'" in lines[0] and "H87" in lines[0], lines[0]


def test_the_measured_cut_still_prices(monkeypatch, tmp_path):
    # x162's own cut and rows: the given order funds every step (H54's root)
    lines = _run_launcher((29, 11, 8), (0.41, 0.712, 0.733887), monkeypatch, tmp_path)
    assert not any("ENTFAELLT" in ln for ln in lines)
    assert any("the given order funds every wake step (unchanged)" in ln for ln in lines)
    luft = [ln for ln in lines if "engste Luft" in ln]
    assert len(luft) == 3 and all("FERTIG" in ln for ln in luft)


def test_mutant_planned_split_in_the_key_prices_the_foreign_geometry():
    """Der alte Schluessel (p_split des Plans) laesst 26,10,12 durchrechnen --
    mit weights_12 auf der PP1-Karte, wo es bei 26,10,12 nicht mehr liegt."""
    ref = _ref("x162")
    p_rows = [er.buffer_rows(local_experts=512, fraction=f, scratch_rows=32)
              for f in (0.472, 0.796, 0.468262)]
    d_rows = [er.buffer_rows(local_experts=s + 1, fraction=f, scratch_rows=sc)
              for s, f, sc in zip(D_SPAN, FR_D, D_SCRATCH)]
    kw = dict(model="/m/" + MODEL, p_split=(26, 10, 12), chunk_layers=CHUNK,
              n_layers=N_LAYERS, p_card=P_CARD, d_ratio="183,137,168", p_rows=p_rows,
              d_rows=d_rows, slot_mib=SLOT_MIB, label=LABEL, reorder=True,
              double_staging=False, reference=ref)
    old = wc.plan_wake_credit(reference_key={"p_card": P_CARD, "p_split": (26, 10, 12),
                                             "chunk_layers": CHUNK}, **kw)
    assert not any("ENTFAELLT" in ln for ln in old.lines)
    # the foreign geometry: card0 (PP1) still carries weights_12 of the old cut
    demand0 = old.front_plan["D->P"][1]["demand"]
    assert "weights_12" in demand0 and demand0["weights_12"] > 2800
    new = wc.plan_wake_credit(reference_key={"p_card": P_CARD,
                                             "p_split": wc.reference_p_split(ref, N_LAYERS),
                                             "chunk_layers": CHUNK}, **kw)
    assert new.refusal is None and new.front_plan is None
    assert "ENTFAELLT" in new.lines[0]
