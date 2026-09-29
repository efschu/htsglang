"""#239 rc12z29c-Blocker 2: map and rank count the same D experts.

rc12z29c (83a1a92eb7, -st-cut) died at D's load, TP1:
  RuntimeError: Platztausch-Karte nennt 61 residente Zeilen fuer Layer 0 Rang 1
  (Praefix 60, Pad+Extra 1), der Rang rechnet bei Fraction 0.504 ueber 119
  Experten 60
and TP2 logged '#96 STORE-SLOTS rank=2 lo=344 pad=True ... ratios=[183, 137, 168]
fracs=[0.06, 0.51, 0.48]'. Two roots:

1. The map cut the D windows with its own rounding (``round`` per rank, the rest
   on the last): 215,113,160 over 512 -> 226,119,167. The rank cuts with
   ``distributed.utils.partition_units`` (largest remainder): 226,118,168 (TP2
   lo=344, TP1 118 experts + pad = 119). For 183,137,168 both happen to give
   192,144,176, which is why no earlier boot saw it.
2. The store geometry (and the band/V1-map geometry) was published from --extra-d
   BEFORE the pre-map solve -- the stated vectors. With the map it is only the
   STORE-SLOTS diagnostic; without one it would be the slot authority.

The D load path is walked here on all three ranks for the pinned form
215,113,160 / FR_D 0.101,0.504,0.431 / pad: the runtime window, the map's layout
(_karten_layout_lokal), the Platztausch count at the repack door
(presplit_expert_offload_after_repack vs resident_slot_count) and STORE-SLOTS
(_expert_store_rows_for). Plus the launcher side: a D start whose form moved
after the map (e.g. the P1c FR-D ceiling) is refused by name (W170).
"""
from __future__ import annotations

import inspect
import json
import logging
import os
from types import SimpleNamespace

import pytest

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

from sglang.srt.layers.moe import expert_map as em  # noqa: E402
from sglang.test.ci.ci_register import register_cpu_ci  # noqa: E402

register_cpu_ci(est_time=8, suite="stage-a-weg2-unit")

TOTAL = 512
SOLVED = (215, 113, 160)
STATED = (183, 137, 168)
FR_D = (0.101, 0.504, 0.431)
FR_P = (0.26, 0.45, 0.39)
P_STAGE_LAYERS = [29, 11, 8]
LAYERS = (0, 30, 45)  # one layer per P stage (the D layout is kept per stage)


def _runtime_windows(ratios):
    """The window the rank's FusedMoE layer takes (fused_moe_triton/layer.py:
    tp_partition_offset/size over num_experts units, family "moe"), under the
    Form A plan the D group installs (dense 1,0,0 with empty ranks)."""
    from sglang.srt.distributed import utils as du

    out = []
    with du.scoped_tp_partition_ratios([1, 0, 0], {"moe": list(ratios)}, allow_zero=True):
        for r in range(len(ratios)):
            lo = du.tp_partition_offset(TOTAL, len(ratios), r, TOTAL, "moe")
            n = du.tp_partition_size(TOTAL, len(ratios), r, TOTAL, "moe")
            out.append((lo, n))
    return out


def _map(ratios=SOLVED, fr_d=FR_D):
    stages = [s for s, n in enumerate(P_STAGE_LAYERS) for _ in range(n)]
    karte = em.build_nested(total=TOTAL, ratios=list(ratios), fr_pp=list(FR_P),
                            fr_tp=list(fr_d), p_layer_stage=stages, pad_tp=1)
    return json.loads(json.dumps(karte))  # what the ranks read: the JSON file


def _layer(rank, layer_id, ratios=SOLVED, fr_d=FR_D):
    lo, n = _runtime_windows(ratios)[rank]
    return SimpleNamespace(num_experts=TOTAL, num_local_experts=n + 1, layer_id=layer_id,
                           moe_tp_rank=rank, _expert_shard_generic=True,
                           _gguf_expert_range=(lo, lo + n),
                           _expert_offload_fraction=float(fr_d[rank]),
                           _moe_offload_excluded=False)


class _PastTheDoor(Exception):
    """plan_load_time_staging reached: every Platztausch check before it held."""


@pytest.fixture
def d_rank(monkeypatch, tmp_path):
    from sglang.srt.layers.moe import expert_offload, expert_store

    def _arm(karte):
        monkeypatch.setattr(expert_store, "expert_map", lambda: karte)
        monkeypatch.setenv("SGLANG_WEG2_GROUP", "D")
        monkeypatch.setenv(expert_store.STORE_DIR_ENV, str(tmp_path))
        monkeypatch.setenv(expert_store.SLOT_FRACTION_ENV, "0.5")
        monkeypatch.setenv(expert_store.STORE_GEOMETRY_ENV,
                           ",".join(map(str, SOLVED)) + "|" + ",".join(map(str, FR_D)))

        def _stop(*a, **k):
            raise _PastTheDoor()

        monkeypatch.setattr(expert_offload, "plan_load_time_staging", _stop)
        return expert_offload

    return _arm


def test_the_map_cuts_the_d_windows_like_the_rank():
    """RED before: 226,119,167 against the rank's 226,118,168."""
    for ratios in (SOLVED, STATED, (3991, 1000, 1000), (60, 226, 226), (1, 1, 1)):
        spans = em.scaled_spans(list(ratios), TOTAL)
        runtime = _runtime_windows(ratios)
        assert spans == [n for _, n in runtime], ratios
        assert em.bounds(spans) == [lo for lo, _ in runtime], ratios
    assert em.scaled_spans(list(SOLVED), TOTAL) == [226, 118, 168]
    assert em.bounds([226, 118, 168])[2] == 344  # TP2 lo=344 in the rc12z29c D log


@pytest.mark.parametrize("rank", [0, 1, 2])
def test_the_full_d_load_path_holds_on_every_rank(rank, d_rank, caplog):
    """RED before: TP1 'Platztausch-Karte nennt 61 residente Zeilen ... 60'
    (the rc12z29c death), TP2's cold id 344 without a store slot."""
    eo = d_rank(_map())
    for layer_id in LAYERS:
        layer = _layer(rank, layer_id)
        E = int(layer.num_local_experts)
        lo, _ = layer._gguf_expert_range
        # _karten_layout_lokal: every id the map names lies in the rank's window
        order, n_praefix, refill = eo._karten_layout_lokal(layer, E)
        assert all(0 <= e < E for e in order)
        assert order.count(0) == 1  # the pad row, exactly once
        # the count the repack door compares, and the door itself
        assert len(order) == eo.resident_slot_count(E, FR_D[rank]), (layer_id, len(order))
        with pytest.raises(_PastTheDoor):
            eo.presplit_expert_offload_after_repack(layer)
        assert layer._moe_offload_exchange_rows == n_praefix
        # STORE-SLOTS: every cold expert of this rank has a slot in the map
        spill = [e for e in range(E) if e not in set(order)]
        with caplog.at_level(logging.INFO, logger=eo.__name__):
            rows = eo._expert_store_rows_for(layer, SimpleNamespace(spill_ids=spill))
        _dir, _key, lo_store, slots, index, pad = rows
        assert lo_store == lo and pad is True
        assert sorted(index) == sorted(spill)
        assert len(set(index.values())) == len(spill)
        assert all(0 <= v < slots for v in index.values())
    line = next(r.getMessage() for r in caplog.records if "#96 STORE-SLOTS" in r.getMessage())
    assert f"ratios={list(SOLVED)}" in line and "karte=D/" in line


def test_the_stated_map_under_the_solved_window_is_what_killed_rc12z29c(d_rank):
    """The death reproduced through the same door: the map counts TP1 over 119
    ids at 0.504 (60 + pad), the rank over 118 + pad."""
    eo = d_rank(_map())
    from sglang.srt.layers.moe import expert_map as _em

    old_spans = [226, 119, 167]  # the old scaled_spans of 215,113,160
    assert sum(old_spans) == TOTAL
    assert _em.resident_count_like_the_rank(old_spans[1] + 1, FR_D[1]) == 61
    assert eo.resident_slot_count(118 + 1, FR_D[1]) == 60


def test_a_d_form_moved_after_the_map_is_refused_by_name(monkeypatch, tmp_path):
    """W170: the map describes one D form; the D start carries it or refuses.
    The P1c FR-D ceiling (rc12z30b) raises FR_D after P's sleep -- exactly
    this move, caught before D loads."""
    from sglang.srt.planner import pp_cut
    from sglang.srt.weg2 import draft_post as dp
    from sglang.srt.weg2 import launcher as L

    monkeypatch.setattr(pp_cut, "checkpoint_weight_terms",
                        lambda model: SimpleNamespace(num_experts=TOTAL))
    fr_d = ",".join(map(str, FR_D))
    ns = SimpleNamespace(
        extra_d=f"--rank-moe-ratio {','.join(map(str, SOLVED))} "
                f"--rank-moe-resident-fraction {fr_d} --rank-tp-ratio 1,0,0",
        extra_p=f"--rank-moe-resident-fraction {','.join(map(str, FR_P))}",
        env_d=f"SGLANG_MOE_RESIDENT_EXPERT_FRACTION={fr_d}",
        pp_cut_expert_device_fraction=",".join(map(str, FR_P)), tag="t", model="m")
    lines = []
    path = L.publish_expert_map(ns, "m", str(tmp_path), lines.append,
                                p_stage_layers=P_STAGE_LAYERS, chunk_layers=3)
    with open(path) as fh:
        assert json.load(fh)["spans"] == [226, 118, 168]
    assert ns._expert_map_d_form == {"ratios": [str(x) for x in SOLVED],
                                     "fractions": [str(x) for x in FR_D]}
    L.refuse_d_form_off_the_map(ns, lines.append)
    assert any("D-START geprueft" in ln for ln in lines)
    # the equal form written in another spelling is the same form
    ns.env_d = "SGLANG_MOE_RESIDENT_EXPERT_FRACTION=0.1010,0.5040,0.4310"
    L.refuse_d_form_off_the_map(ns, lines.append)

    raised = (0.101, 0.586, 0.491)  # FRACTION-SOLVE D's DECKE (P1c, rc12z30b)
    ceiling = getattr(L, "d_fr_ceiling_adopt", None)
    if ceiling is not None and "map_built" in inspect.signature(ceiling).parameters:
        plan = SimpleNamespace(fits=[SimpleNamespace(ceiling_fraction=x) for x in raised],
                               solved_fractions=(), solved_owner_ratio=())
        env = {"SGLANG_WEG2_TORCH_CACHE_CAP": "1"}
        assert ceiling(plan, FR_D, env, False, map_built=False)[0] is not None
        assert ceiling(plan, FR_D, env, False, map_built=True)[0] is None
    # what P1c would write if it ran after the map: refused, by name, per carrier
    moved_argv = SimpleNamespace(**vars(ns))
    moved_argv.extra_d = dp.replace_vector_flag(ns.extra_d, "--rank-moe-resident-fraction", raised)
    with pytest.raises(L.Weg2DFormOffTheMap, match="W170 .*--rank-moe-resident-fraction"):
        L.refuse_d_form_off_the_map(moved_argv, lines.append)
    moved_env = SimpleNamespace(**vars(ns))
    moved_env.env_d = dp.replace_env_vector(ns.env_d, "SGLANG_MOE_RESIDENT_EXPERT_FRACTION", raised)
    with pytest.raises(L.Weg2DFormOffTheMap, match="W170 .*--env-d"):
        L.refuse_d_form_off_the_map(moved_env, lines.append)
    moved_owner = SimpleNamespace(**vars(ns))
    moved_owner.extra_d = dp.replace_vector_flag(ns.extra_d, "--rank-moe-ratio", STATED)
    with pytest.raises(L.Weg2DFormOffTheMap, match="W170 .*--rank-moe-ratio"):
        L.refuse_d_form_off_the_map(moved_owner, lines.append)
    # no map published: nothing to hold against
    L.refuse_d_form_off_the_map(SimpleNamespace(extra_d="", env_d=""), lines.append)


def test_every_d_start_is_checked_against_the_map():
    """RED before: no check; d-only, dry run and the real start build D's env."""
    from sglang.srt.weg2 import launcher as L

    src = inspect.getsource(L.main)
    starts = [i for i in range(len(src)) if src.startswith("env_d = build_env(", i)]
    assert len(starts) == 3
    for i in starts:
        before = src[:i].rsplit("\n", 3)
        assert any("refuse_d_form_off_the_map(ns, log)" in ln for ln in before), before


def test_the_store_geometry_follows_the_pinned_form():
    """RED before: STORE-GEOMETRY, bands and the V1 map kept the stated vectors."""
    from sglang.srt.weg2 import launcher as L

    stated = "183,137,168|0.06,0.51,0.48"
    xchg = {"SGLANG_MOE_EXPERT_STORE_GEOMETRY": stated, em.MAP_ENV: "/x/v1.json",
            "SGLANG_WEG2_EXPERT_BAND_SIZE": "16", "SGLANG_WEG2_EXPERT_BANDS": "32"}
    form = {"ratios": [str(x) for x in SOLVED], "fractions": [str(x) for x in FR_D],
            "cut": (0.0, 48.0, 16.0)}
    lines = []
    L.repoint_store_geometry_at_pinned_form(xchg, form, TOTAL, lines.append)
    assert xchg["SGLANG_MOE_EXPERT_STORE_GEOMETRY"] == "215,113,160|0.101,0.504,0.431"
    assert em.MAP_ENV not in xchg
    bs, bc = em.band_geometry(list(SOLVED), TOTAL)
    assert xchg.get("SGLANG_WEG2_EXPERT_BAND_SIZE") == (str(bs) if bs > 0 else None)
    src = inspect.getsource(L.main)
    pin = src.index("_pin = pin_d_form_for_map(ns, log)")
    repoint = src.index("repoint_store_geometry_at_pinned_form(")
    emap_at = src.index("_emap = publish_expert_map(")
    env_p = src.index('group="P", xchg_env=xchg_env')
    assert pin < repoint < emap_at < env_p
