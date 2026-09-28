"""#239 rc12z29c: the Platztausch map is built from the D form D runs.

rc12z29b (f833fcbb2d, profile -st-cut) died at D's load, 28.09. 19:25:40, TP1:
  RuntimeError: Platztausch-Karte: Layer 0 Phase D nennt Ids ausserhalb dieses
  Rangs (lokal [-33, -32, -31, -30], lo=226, pad=True, E=119) -- Karte und
  Expertenfenster beschreiben nicht denselben Rang
The launcher built the map (publish_expert_map, before P starts) from the STATED
--rank-moe-ratio 183,137,168 (TP1 window [192, 336)); the owned solve after P's
sleep published 215,113,160 into --extra-d (TP1 window [226, 345)), and D's
expert window followed the published vector. The map cannot follow afterwards --
P loaded its common prefix and store slots from it -- so the form is solved
BEFORE the map, the map reads the published vectors, and the later D solve
checks exactly that form instead of solving it again.
"""
from __future__ import annotations

import inspect
import json
import os
from types import SimpleNamespace

import pytest

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

from sglang.srt.layers.moe import expert_map as em  # noqa: E402
from sglang.test.ci.ci_register import register_cpu_ci  # noqa: E402

register_cpu_ci(est_time=6, suite="stage-a-weg2-unit")

STATED = "183,137,168"
SOLVED = (215, 113, 160)
FR_P = "0.26,0.45,0.39"
FR_D_STATED = "0.06,0.44,0.365"
FR_D_SOLVED = (0.096, 0.504, 0.431)
CUT = (0.0, 48.0, 16.0)
P_STAGE_LAYERS = [29, 11, 8]


def _ns():
    return SimpleNamespace(
        extra_d=f"--rank-moe-ratio {STATED} --rank-moe-resident-fraction {FR_D_STATED} "
                f"--rank-tp-ratio 1,0,0",
        extra_p=f"--rank-moe-resident-fraction {FR_P}",
        env_d="SGLANG_MOE_SCRATCH_SLOTS=91,48,48;SGLANG_MOE_RESIDENT_EXPERT_FRACTION="
              + FR_D_STATED,
        pp_cut_expert_device_fraction=FR_P, tag="t", model="m",
        d_kv_token_cut="owned")


def _solve_before_the_map(ns, lines):
    """What the D solve before the map publishes (log_d_rank_vram_solve):
    the ownership, FR_D and the cut of the owned solve."""
    from sglang.srt.weg2 import draft_post as dp
    from sglang.srt.weg2 import launcher as L

    plan = SimpleNamespace(solved_owner_ratio=SOLVED, owner_record={})
    L.publish_d_owner_ratio(ns, plan, "D(Karte, Erwartung)", lines.append)
    ns.extra_d = dp.replace_vector_flag(ns.extra_d, "--rank-moe-resident-fraction",
                                        FR_D_SOLVED)
    ns._d_solved_cut = CUT


def _publish(monkeypatch, tmp_path, ns):
    from sglang.srt.planner import pp_cut
    from sglang.srt.weg2 import launcher as L

    monkeypatch.setattr(pp_cut, "checkpoint_weight_terms",
                        lambda model: SimpleNamespace(num_experts=512))
    lines = []
    path = L.publish_expert_map(ns, "m", str(tmp_path), lines.append,
                                p_stage_layers=P_STAGE_LAYERS, chunk_layers=3)
    with open(path) as fh:
        return json.load(fh)


def _local_ids(karte, rank, window_lo, pad=True):
    """``_karten_layout_lokal``'s view: the map's D ids of this rank, local to
    the window D's argv gives the rank."""
    praefix, extra, _ = em.rank_layout(karte, em.PHASE_TP, 0, rank)
    return [int(g) - int(window_lo) + (1 if pad else 0) for g in list(praefix) + list(extra)]


def test_the_stated_map_puts_tp1_outside_the_published_window(monkeypatch, tmp_path):
    """The rc12z29b death, reproduced: map from 183,137,168, window from 215,113,160."""
    karte = _publish(monkeypatch, tmp_path, _ns())
    lo_published = em.bounds(em.scaled_spans(list(SOLVED), 512))[1]
    assert lo_published == 226
    local = _local_ids(karte, 1, lo_published)
    assert min(local) < 0 and local[:4] == [-33, -32, -31, -30]


def test_the_map_is_built_from_the_pinned_published_form(monkeypatch, tmp_path):
    """RED before: no pin; the map read the stated vector."""
    from sglang.srt.weg2 import launcher as L

    ns = _ns()
    lines = []
    _solve_before_the_map(ns, lines)
    form = L.pin_d_form_for_map(ns, lines.append)
    assert form["ratios"] == [str(x) for x in SOLVED]
    assert form["cut"] == CUT and ns._d_map_form is form
    assert any("KARTE-FORM (#239 rc12z29c) GEPINNT" in ln for ln in lines)
    karte = _publish(monkeypatch, tmp_path, ns)
    # the rank's own cut (partition_units), not a second rounding (Blocker 2)
    assert karte["spans"] == [226, 118, 168]
    for rank, (lo, span) in enumerate(zip(karte["bounds"], karte["spans"])):
        local = _local_ids(karte, rank, lo)
        assert all(0 <= e < span + 1 for e in local), (rank, local[:4])
    # P's phase stays whole-stage (PP, tp=1): unequal P/D vectors are the
    # normal case, the common prefix is a subset of D's residency per span
    assert em.refuse_if_inconsistent(karte) in (None, "")
    d_all = {g for ids in karte["phases"][em.PHASE_TP]["resident"] for g in ids}
    for common in karte["phases"][em.PHASE_PP]["common"]:
        assert set(common) <= d_all


def test_main_solves_the_form_before_the_map_is_built():
    """RED before: the map was published first, the owned solve ran after P's sleep."""
    from sglang.srt.weg2 import launcher as L

    src = inspect.getsource(L.main)
    pre = src.index('"D(Karte, Erwartung)"')
    pin = src.index("pin_d_form_for_map(ns, log)")
    emap = src.index("_emap = publish_expert_map(")
    assert pre < pin < emap


def test_which_forms_the_planner_solves():
    from sglang.srt.weg2 import launcher as L

    for mode in ("owned", "joint", "maxmin"):
        assert L.d_form_solved_by_planner(SimpleNamespace(d_kv_token_cut=mode))
    for stated in ("0,48,16", "off"):
        assert not L.d_form_solved_by_planner(SimpleNamespace(d_kv_token_cut=stated))


class _Stop(Exception):
    pass


def test_a_pinned_form_is_checked_not_solved_again(monkeypatch):
    """RED before: the post-P solve restarted from the stated vector and the
    'owned' request -- a second, possibly different ownership under a map
    that describes the first."""
    from sglang.srt.planner import expert_residency as er
    from sglang.srt.weg2 import launcher as L

    ns = _ns()
    lines = []
    _solve_before_the_map(ns, lines)
    L.pin_d_form_for_map(ns, lines.append)
    for attr in ("d_foreign_context_mib", "d_nontorch_mib", "d_reserve_mib",
                 "d_residency_reference_logs", "d_card_reference_logs", "profile"):
        setattr(ns, attr, "")
    ns.weg2_boot_form = None
    seen = {}

    def _plan(**kw):
        seen.update(kw)
        raise _Stop()

    monkeypatch.setattr(er, "plan_d_residency", _plan)
    cards = [SimpleNamespace(total_mib=t, nvml_index=i, name="c%d" % i, uuid="u%d" % i)
             for i, t in enumerate((32607, 20480, 20480))]
    with pytest.raises(_Stop):
        L.log_d_rank_vram_solve(ns, cards, [29000, 18000, 18000], lines.append, "D")
    assert tuple(seen["kv_token_shares"]) == CUT
    assert [int(x) for x in seen["ratios"]] == list(SOLVED)
    assert [round(x, 3) for x in seen["fractions"]] == list(FR_D_SOLVED)
    assert any("geprueft, nicht neu" in ln for ln in lines)
