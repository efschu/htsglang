"""R2 (28.09.): the P1c FR-D ceiling is adopted BEFORE the Platztausch map.

NF stayed on the profile's FR_D 0.060/0.510/0.480 although FRACTION-SOLVE D
computed the DECKE 0.062/0.586/0.491 (~1.3 GiB idle on nv0): the map reads FR_D
from --extra-d before P starts, and after it P1c is refused (a514587c52, W170).
Now an armed P1c (P0 in --env-d) solves and pins the D form from the expectation
budgets like an owned/joint/maxmin form (#239 rc12z29c): the map, STORE-GEOMETRY
and D's start all carry the ceiling, and the solve after P only checks it.
"""
from __future__ import annotations

import inspect
import json
import os
from types import SimpleNamespace

import pytest

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

from sglang.srt.weg2 import draft_post as dp  # noqa: E402
from sglang.srt.weg2 import launcher as L  # noqa: E402

TOTAL = 512
RATIOS = (183, 137, 168)                 # NF's stated ownership (no token cut)
PROFILE_FR = (0.06, 0.51, 0.48)          # the profile's FR_D
DECKE = (0.062, 0.586, 0.491)            # FRACTION-SOLVE D's edge, every NF boot
FR_P = (0.26, 0.45, 0.39)
P_STAGE_LAYERS = [29, 11, 8]


def _ns(env_extra=""):
    fr = ",".join(map(str, PROFILE_FR))
    return SimpleNamespace(
        extra_d=f"--rank-moe-ratio {','.join(map(str, RATIOS))} "
                f"--rank-moe-resident-fraction {fr} --rank-tp-ratio 1,0,0",
        extra_p=f"--rank-moe-resident-fraction {','.join(map(str, FR_P))}",
        env_d=";".join(x for x in (f"SGLANG_MOE_RESIDENT_EXPERT_FRACTION={fr}", env_extra) if x),
        pp_cut_expert_device_fraction=",".join(map(str, FR_P)), tag="t", model="m")


def _plan(ceil):
    return SimpleNamespace(fits=[SimpleNamespace(ceiling_fraction=x) for x in ceil],
                           solved_fractions=(), solved_owner_ratio=())


def test_an_armed_p1c_is_solved_before_the_map():
    armed = _ns("SGLANG_WEG2_TORCH_CACHE_CAP=1")
    assert L.d_fr_ceiling_before_map(armed)
    assert not L.d_fr_ceiling_before_map(_ns())                       # P0 off
    assert not L.d_fr_ceiling_before_map(
        _ns("SGLANG_WEG2_TORCH_CACHE_CAP=1;SGLANG_WEG2_FR_D_CEILING=0"))
    owned = _ns("SGLANG_WEG2_TORCH_CACHE_CAP=1")
    owned.d_kv_token_cut = "owned"
    assert not L.d_fr_ceiling_before_map(owned)                       # pinned by its own solve
    dense = SimpleNamespace(extra_d="", env_d="SGLANG_WEG2_TORCH_CACHE_CAP=1")
    assert not L.d_fr_ceiling_before_map(dense)                       # no MoE D form
    src = inspect.getsource(L.main)
    cond = src.index("d_form_solved_by_planner(ns) or d_fr_ceiling_before_map(ns)")
    assert cond < src.index("_pin = pin_d_form_for_map(ns, log)") < src.index(
        "_emap = publish_expert_map(")


def test_p1c_before_the_map_and_the_map_is_the_d_form(monkeypatch, tmp_path):
    """The whole chain with NF's numbers: the pre-map pass adopts the DECKE into
    --extra-d, the form is pinned without a token cut, the map, STORE-GEOMETRY
    and D's start carry 0.062/0.586/0.491, and the solve after P keeps it."""
    from sglang.srt.planner import pp_cut

    monkeypatch.setattr(pp_cut, "checkpoint_weight_terms",
                        lambda model: SimpleNamespace(num_experts=TOTAL))
    ns = _ns("SGLANG_WEG2_TORCH_CACHE_CAP=1")
    env_d = L.parse_group_env(ns.env_d)
    # the pre-map pass (label "D(Karte, Erwartung)"): no map yet -> adopted
    new, lines = L.d_fr_ceiling_adopt(_plan(DECKE), PROFILE_FR, env_d, owned=False,
                                      map_built=False)
    assert new == list(DECKE), lines
    ns.extra_d = dp.replace_vector_flag(ns.extra_d, "--rank-moe-resident-fraction", new)
    ns.env_d = dp.replace_env_vector(ns.env_d, "SGLANG_MOE_RESIDENT_EXPERT_FRACTION", new)
    ns._d_solved_cut = None                                           # no token cut asked
    log = []
    form = L.pin_d_form_for_map(ns, log.append)
    assert form is not None and form["cut"] is None, log
    assert [float(x) for x in form["fractions"]] == list(DECKE)
    assert [int(float(x)) for x in form["ratios"]] == list(RATIOS)
    # STORE-GEOMETRY follows the pinned form, not the profile vector
    xchg = {"SGLANG_MOE_EXPERT_STORE_GEOMETRY": "183,137,168|0.06,0.51,0.48"}
    L.repoint_store_geometry_at_pinned_form(xchg, form, TOTAL, log.append)
    assert xchg["SGLANG_MOE_EXPERT_STORE_GEOMETRY"] == "183,137,168|0.062,0.586,0.491"
    # the map is built from exactly this form
    path = L.publish_expert_map(ns, "m", str(tmp_path), log.append,
                                p_stage_layers=P_STAGE_LAYERS, chunk_layers=3)
    with open(path) as fh:
        assert json.load(fh)["spans"] == [192, 144, 176]
    assert [float(x) for x in ns._expert_map_d_form["fractions"]] == list(DECKE)
    # D's start carries it (W170 passes), in both carriers
    L.refuse_d_form_off_the_map(ns, log.append)
    # the solve after P checks the pinned form: nothing moves again
    again, again_lines = L.d_fr_ceiling_adopt(_plan(DECKE), DECKE, env_d, owned=False,
                                              map_built=True)
    assert again is None, again_lines
    # and a profile FR sneaking back is refused by name
    back = SimpleNamespace(**vars(ns))
    back.extra_d = dp.replace_vector_flag(ns.extra_d, "--rank-moe-resident-fraction",
                                          PROFILE_FR)
    with pytest.raises(L.Weg2DFormOffTheMap, match="W170"):
        L.refuse_d_form_off_the_map(back, log.append)


def test_a_pinned_form_without_a_cut_solves_without_token_shares():
    src = inspect.getsource(L.log_d_rank_vram_solve)
    assert '_kv_cut = tuple(_pinned["cut"]) if _pinned.get("cut") else None' in src
    # a cut that was ASKED for and not published still leaves the form unpinned
    ns = _ns("SGLANG_WEG2_TORCH_CACHE_CAP=1")
    ns.d_kv_token_cut = "0.5,0.25,0.25"
    ns._d_solved_cut = None
    log = []
    assert L.pin_d_form_for_map(ns, log.append) is None
    assert any("NICHT GEPINNT" in ln for ln in log)
