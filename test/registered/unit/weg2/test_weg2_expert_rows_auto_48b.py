# SPDX-License-Identifier: Apache-2.0
"""#48b -- die Experten-Fractions vom Planer statt aus dem Profil (Register-Audit #48/#81).

DER BEFUND: ``nf.env`` pinnt ``NF_FR_P=0.332,0.64,0.39`` und
``NF_FR_D=0.06,0.51,0.48`` an drei Stellen. Die Decken, die der Planer heute
rechnet, liegen weit darueber (fnFL2x177 Launcher-Log Z.64-66/195/200):
P-KARTE 238/395/576 Zeilen gegen 202/360/408, D-KARTE 136/140/141 gegen
130/122/133. Die Pins sind die RETRY-Kante (x167: oberhalb 0.332 wiederholt der
Caching-Allokator, H59b), die Karten des Planers die OOM-Kante.

DER TERM: ``expert_rows_auto`` liest die Retry-Kante je Rang aus ``[vram-peak]``
(WEG2-VRAM-PEAK card_free + alloc_retries) der Referenz-Logs. Die Fixtures sind
die woertlichen Zeilen aus /spinning/evidence-665-f1/boot_weg2_fnFL2x177_
09729d97c4_0925_044625.{P,D}.log (nur ``MoE expert-offload active``,
``max_running_requests`` und ``WEG2-VRAM-PEAK``).

ABNAHME (#48 B): auto liegt auf PP0/PP1 und allen D-Raengen hoechstens eine
Zeile vom Pin. PP2 liegt 104 Zeilen DARUEBER -- die Karte hat dort am Metall
2682 MiB frei und kein VRAM-Term bindet; das ist benannt, nicht gepinnt.
"""

import os
import shlex
import types

import pytest

from sglang.srt.planner import expert_residency as er
from sglang.srt.planner import expert_rows_auto as ra
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=10, suite="base-a-test-cpu")

FIX = os.path.join(os.path.dirname(__file__), "fixtures", "expert_rows_auto_48b")
SLOT_MIB = 1297637376 / 512 / (1 << 20)
PIN_P = (0.332, 0.64, 0.733887)          # nach dem H25-Posten (Launcher-Log Z.57)
PIN_D = (0.06, 0.51, 0.48)
E_D, S_D = (193, 145, 177), (118, 48, 48)
LAYERS_P = (29, 11, 8)


def _text(kind):
    with open(os.path.join(FIX, "fnFL2x177.%s.lines" % kind)) as fh:
        return fh.read()


def _p(**kw):
    args = dict(group="P", reference_texts=[("fnFL2x177", _text("P"))], tag="PP",
                phases=ra.P_PHASES, slot_mib=SLOT_MIB, local_experts=[512] * 3,
                layers=list(LAYERS_P), scratch=[32] * 3, seats=4, chunk=16384)
    args.update(kw)
    return ra.solve_group(**args)


def _d(**kw):
    args = dict(group="D", reference_texts=[("fnFL2x177", _text("D"))], tag="TP",
                phases=ra.D_PHASES, slot_mib=SLOT_MIB, local_experts=list(E_D),
                layers=[48] * 3, scratch=list(S_D), seats=1)
    args.update(kw)
    return ra.solve_group(**args)


def _rows(E, f, S):
    return er.buffer_rows(local_experts=E, fraction=f, scratch_rows=S)


def test_the_logs_describe_the_pinned_form():
    """Die Referenz ist selbstbeschreibend -- keine Zahl kommt aus dem Profil."""
    p = ra.rank_forms(_text("P"), "PP")
    d = ra.rank_forms(_text("D"), "TP")
    assert [p[r].buffer_rows for r in range(3)] == [202, 360, 408]
    assert [p[r].layers for r in range(3)] == list(LAYERS_P)
    assert [d[r].buffer_rows for r in range(3)] == [130, 122, 133]
    assert [d[r].local_experts for r in range(3)] == list(E_D)
    assert ra.seats_of(_text("P"), "PP") == 4 and ra.seats_of(_text("D"), "TP") == 1
    assert ra.chunk_rows_of(_text("P"), "PP") == 16384


def test_auto_p_is_within_one_row_of_the_pin_on_pp0_and_pp1():
    g = _p()
    assert g.refusal is None, g.lines
    pin = [_rows(512, f, 32) for f in PIN_P]
    assert pin == [202, 360, 408]
    assert abs(g.rows[0] - pin[0]) <= 1 and abs(g.rows[1] - pin[1]) <= 1
    assert list(g.rows) == [203, 359, 512]
    # PP1 retried once at the pin (chunk 4) -> the pin lies ABOVE the retry edge
    assert g.edges[1][0].retries == 1 and g.edges[1][0].free_min_mib == 84.0
    # PP2: 2682 MiB measured free, no retry -> the card carries every expert row
    assert g.edges[2][0].free_min_mib == 2682.0 and g.edges[2][0].retries == 0
    assert g.rows[2] == 512 and g.fractions[2] == 0.996


def test_auto_d_is_within_one_row_of_the_pin_on_every_rank():
    g = _d()
    assert g.refusal is None, g.lines
    pin = [_rows(E, f, S) for E, f, S in zip(E_D, PIN_D, S_D)]
    assert pin == [130, 122, 133]
    assert all(abs(a - p) <= 1 for a, p in zip(g.rows, pin)), (g.rows, pin)
    # every D rank retried at the pin (x177: 22 / 3 / 6 in chunk+round)
    assert [g.edges[r][0].retries for r in range(3)] == [22, 3, 6]
    assert list(g.rows) == [129, 121, 132]
    assert list(g.fractions) == [0.056, 0.503, 0.474]


def test_mutant_without_the_retry_count_is_the_oom_edge_again():
    """Die Retry-Zahl traegt die Abnahme: wer nur card_free liest (die OOM-Sicht),
    landet ueber dem Pin -- dieselbe Fehlerform wie die Planer-Karten heute."""
    text = _text("D").replace("alloc_retries=", "alloc_retries=0 was=")
    g = _d(reference_texts=[("mutant", text)])
    assert g.refusal is None
    assert list(g.rows) == [132, 122, 134]           # 130+2, 122+0 (28 MiB), 133+1
    assert any(abs(a - p) > 1 for a, p in zip(g.rows, (130, 122, 133)))


def test_fraction_follows_rows_by_ceil_not_int():
    """rows -> f ueber die Pufferregel: ceil(f x E) = R, nie int (rank-ratios)."""
    g = _p()
    E, S = 512, 32
    for rows, f in zip(g.rows, g.fractions):
        R = er.resident_rows(E, f)
        assert min(R + S, E) == rows
    assert g.fractions[1] == 0.638 and er.resident_rows(512, 0.638) == 327
    assert int(0.638 * 512) == 326          # int waere eine Zeile daneben


def test_ratio_is_a_ratio():
    """183,137,168 summiert 488, nicht 512 -- Largest-Remainder, Summe geprueft."""
    assert sum((183, 137, 168)) == 488
    assert ra.ratio_spans(512, [183, 137, 168], pad=1) == [193, 145, 177]


@pytest.mark.parametrize("kw, why", [
    (dict(seats=6), "Sitze 1, der Boot 6"),
    (dict(local_experts=[190, 145, 180]), "fremde Geometrie"),
    (dict(reference_texts=[]), "keine Referenz-Logs"),
])
def test_foreign_form_refuses_by_name(kw, why):
    g = _d(**kw)
    assert g.refusal is not None and ra.REFUSAL_CODE in g.refusal
    assert why in g.refusal
    assert g.rows == () and g.fractions == ()


def test_foreign_chunk_refuses_by_name():
    g = _p(chunk=8192)
    assert g.refusal is not None and "Chunk 16384, der Boot 8192" in g.refusal


# ---------------------------------------------------------------------------
# Launcher: auto an ALLEN Stellen, Override bleibt, Widerspruch verweigert
# ---------------------------------------------------------------------------


def _ns(tmp_path, *, fr_p=None, fr_d=None, env_fr_p=None, env_fr_d=None, refs=True):
    for kind in ("P", "D"):
        (tmp_path / ("boot_weg2_fnFL2x177_x.%s.log" % kind)).write_text(_text(kind))
    extra_p = "--max-running-requests 4 --page-size 64"
    if fr_p:
        extra_p += " --rank-moe-resident-fraction " + fr_p
    extra_d = ("--max-running-requests 1 --rank-tp-ratio 1,0,0 --rank-moe-ratio 183,137,168")
    if fr_d:
        extra_d += " --rank-moe-resident-fraction " + fr_d
    env_p = "SGLANG_MOE_SCRATCH_SLOTS=32;SGLANG_UNEVEN_MOE_EXPERT_SHARD=1"
    if env_fr_p:
        env_p += ";SGLANG_MOE_RESIDENT_EXPERT_FRACTION=" + env_fr_p
    env_d = "SGLANG_MOE_SCRATCH_SLOTS=118,48,48;SGLANG_UNEVEN_MOE_EXPERT_SHARD=1"
    if env_fr_d:
        env_d += ";SGLANG_MOE_RESIDENT_EXPERT_FRACTION=" + env_fr_d
    return types.SimpleNamespace(
        model="/m/Qwen3.8-Flash-Next-INT4-Mixed-AutoRound-Minachist",
        expert_rows_reference_logs=(str(tmp_path / "boot_weg2_fnFL2x177_x.P.log") if refs else ""),
        pp_cut_expert_device_fraction=fr_p or "", pp_cut_expert_lru_rows="32,32,32",
        pp_stage_ratio="29,11,8", p_bs=4, d_bs=1, draft_kv_on_p="off",
        extra_p=extra_p, extra_d=extra_d, env_p=env_p, env_d=env_d)


@pytest.fixture
def launcher(monkeypatch):
    from sglang.srt.planner import pp_cut
    from sglang.srt.weg2 import launcher as L

    monkeypatch.setattr(pp_cut, "checkpoint_weight_terms", lambda _m: types.SimpleNamespace(
        expert_layer_weight_bytes=1297637376.0, num_experts=512, n_layers=48))
    monkeypatch.setattr(L, "P_CHUNKED_PREFILL_TOKENS", 16384)
    return L


def test_launcher_publishes_auto_to_every_place_of_both_groups(tmp_path, launcher):
    ns = _ns(tmp_path)
    lines = []
    launcher.apply_expert_rows_auto(ns, lines.append)
    assert ns.pp_cut_expert_device_fraction == "0.333,0.638,0.996"
    assert ra.flag_value(ns.extra_p, "--rank-moe-resident-fraction",
                         shlex_split=shlex.split) == "0.333,0.638,0.996"
    assert ra.env_value(ns.env_p, "SGLANG_MOE_RESIDENT_EXPERT_FRACTION") == "0.333,0.638,0.996"
    assert ra.flag_value(ns.extra_d, "--rank-moe-resident-fraction",
                         shlex_split=shlex.split) == "0.056,0.503,0.474"
    assert ra.env_value(ns.env_d, "SGLANG_MOE_RESIDENT_EXPERT_FRACTION") == "0.056,0.503,0.474"
    assert ns._expert_rows_auto_p is True
    assert any("EXPERT-ROWS-AUTO P FR 0.333,0.638,0.996 (published to" in ln for ln in lines)
    # the other flags of the extra strings survive untouched
    assert "--rank-moe-ratio 183,137,168" in ns.extra_d and "--page-size 64" in ns.extra_p


def test_literal_auto_is_replaced_everywhere(tmp_path, launcher):
    ns = _ns(tmp_path, fr_p="auto", fr_d="auto", env_fr_p="auto", env_fr_d="auto")
    launcher.apply_expert_rows_auto(ns, lambda s: None)
    assert "auto" not in ns.extra_p and "auto" not in ns.env_p
    assert "auto" not in ns.extra_d and "auto" not in ns.env_d
    assert ns.pp_cut_expert_device_fraction == "0.333,0.638,0.996"


def test_a_given_vector_is_an_override_and_named_as_pin(tmp_path, launcher):
    ns = _ns(tmp_path, fr_p="0.332,0.64,0.39", env_fr_p="0.332,0.64,0.39",
             fr_d="0.06,0.51,0.48", env_fr_d="0.06,0.51,0.48")
    before = (ns.pp_cut_expert_device_fraction, ns.extra_p, ns.env_p, ns.extra_d, ns.env_d)
    lines = []
    launcher.apply_expert_rows_auto(ns, lines.append)
    assert (ns.pp_cut_expert_device_fraction, ns.extra_p, ns.env_p, ns.extra_d, ns.env_d) == before
    ov = [ln for ln in lines if "OVERRIDE" in ln]
    assert len(ov) == 2 and all("HAND-PIN (Planer-Schuld #48)" in ln for ln in ov)
    d = [ln for ln in ov if " D OVERRIDE" in ln][0]
    assert "rang0 130 gegen auto 129 (+1 Zeilen, +116 MiB)" in d
    assert "rang1 122 gegen auto 121 (+1 Zeilen, +116 MiB)" in d
    p = [ln for ln in ov if " P OVERRIDE" in ln][0]
    assert "rang0 202 gegen auto 203 (-1 Zeilen, -70 MiB)" in p
    assert not getattr(ns, "_expert_rows_auto_p", False)


def test_auto_without_reference_refuses(tmp_path, launcher):
    ns = _ns(tmp_path, fr_d="auto", refs=False)
    with pytest.raises(launcher.Weg2LaunchRefused, match="W160.*expert-rows-reference-logs"):
        launcher.apply_expert_rows_auto(ns, lambda s: None)


def test_no_reference_and_no_auto_is_the_old_behaviour(tmp_path, launcher):
    ns = _ns(tmp_path, fr_p="0.332,0.64,0.39", env_fr_p="0.332,0.64,0.39", refs=False)
    before = dict(vars(ns))
    launcher.apply_expert_rows_auto(ns, lambda s: None)
    assert dict(vars(ns)) == before


def test_auto_next_to_a_pin_in_the_same_group_refuses(tmp_path, launcher):
    ns = _ns(tmp_path, fr_d="auto", env_fr_d="0.06,0.51,0.48")
    with pytest.raises(launcher.Weg2LaunchRefused, match="W160 .*D: die Stellen der Gruppe"):
        launcher.apply_expert_rows_auto(ns, lambda s: None)


def test_auto_p_skips_the_h25_post_it_already_contains(tmp_path, launcher):
    """Die Referenz-Karte PP2 traegt den H25-Posten schon (buffer=408 = 0.39 + 176)."""
    assert ra.rank_forms(_text("P"), "PP")[2].buffer_rows == _rows(512, 0.39, 32) + 176
