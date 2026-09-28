"""#239 S3h2: the P->D wake credit under the KV token cut.

``plan_wake_credit_pd`` times the P->D wake against a measured Form A
reference, where the KV pool sat on TP0. Under the cut the KV bytes sit on
the KV-holding 3080 workers, and whether a rank maps its KV pool BEFORE the
weight legs (``weight_updater._weg2_wake_kv_first_ok``: EARLY / LATE) moves
those bytes off the card's free while the legs still need it. The forecast
prints the verdict per rank (first wake: LATE, no leg record; later wakes: the
runtime rule), the later wake's legs with the EARLY pools mapped, and a record
that ``kv_first_check`` holds against the metal's ``WEG2-WAKE-KV-FIRST``
lines. Without the cut the plan is byte-identical.
"""

from __future__ import annotations

import inspect

from sglang.srt.weg2 import launcher as L
from sglang.srt.weg2 import wake_credit_pd as W
from sglang.srt.weg2 import wake_credit_pd_refs as R

#: fnFL2x144/1 (H25 form) in its measured geometry
KEY = dict(R.FORM_KEYS["fnFL2x144"])
MAIN = R.REFERENCES["fnFL2x144/1"]


def _plan(**kw):
    return W.plan_wake_credit_pd(
        model=KEY["model"], p_split=KEY["p_split"], chunk_layers=KEY["chunk_layers"],
        n_layers=48, p_card=KEY["p_card"], d_ratio=KEY["d_ratio"],
        draft_on_p=KEY["draft_on_p"], p_rows=MAIN["p_rows"], d_rows=MAIN["d_rows"],
        slot_mib=2.73, label="D(test)", apply=False,
        dense_repack=KEY.get("dense_repack", False), **kw)


def test_without_the_cut_the_plan_is_byte_identical():
    a, b = _plan(), _plan(kv_first_mib=None)
    assert a.lines == b.lines and a.refusal == b.refusal
    assert not any(W.KV_FIRST_MARKER in ln for ln in a.lines)
    assert not any(k.endswith("-kv-first") for k in (a.front_plan or {}))


def test_the_cut_prints_one_verdict_per_rank_and_the_record():
    kv = [120.0, 3264.0, 1088.0]
    p = _plan(kv_first_mib=kv, kv_cut_shares=[0.0, 0.75, 0.25])
    rows = [ln for ln in p.lines if W.KV_FIRST_MARKER in ln and " TP" in ln and "| {" in ln]
    assert len(rows) == 3
    fc = p.front_plan["P->D-kv-first"]
    assert [f["rank"] for f in fc] == [0, 1, 2]
    assert all(f["first"] == "LATE" for f in fc)  # no leg record on the first wake
    assert fc[1]["kv_mib"] == 3264.0 and fc[1]["share"] == 0.75
    assert "48/64" in rows[1]


def test_a_rank_whose_kv_does_not_fit_before_the_legs_stays_late():
    ref = W.reference_from_dict(MAIN)
    big = [1e6, 1e6, 1e6]
    assert all(f["later"] == "LATE" for f in W.kv_first_forecast(ref, big))
    small = [1.0, 1.0, 1.0]
    fc = W.kv_first_forecast(ref, small, reserve_mib=0.0)
    assert any(f["later"] == "EARLY" for f in fc)


def test_early_pools_come_off_the_card_before_the_later_legs():
    ref = W.reference_from_dict(MAIN)
    fc = W.kv_first_forecast(ref, [1.0, 500.0, 1.0], reserve_mib=0.0)
    adj = W.kv_first_adjusted_reference(ref, fc)
    for f in fc:
        c = int(f["card"])
        want = float(ref.free[c]) - (float(f["kv_mib"]) if f["later"] == "EARLY" else 0.0)
        assert abs(float(adj.free[c]) - want) < 1e-6
    none_early = W.kv_first_forecast(ref, [1e6, 1e6, 1e6])
    assert W.kv_first_adjusted_reference(ref, none_early) is ref


def test_the_check_holds_the_forecast_against_the_metal_lines():
    p = _plan(kv_first_mib=[120.0, 3264.0, 1088.0], kv_cut_shares=[0.0, 0.75, 0.25])
    launch = "\n".join(p.lines)
    d_log = "\n".join([
        "[2026-09-28 22:05:01 TP1] WEG2-WAKE-KV-FIRST LATE free=9000 MiB floor=700 MiB "
        "need=3300 MiB (kv_cache resumed after the weight legs; no leg record yet)",
        "[2026-09-28 22:05:01 TP0] WEG2-WAKE-KV-FIRST EARLY free=9000 MiB floor=767 MiB "
        "need=130 MiB (kv_cache resumed before the weight legs; x)",
    ])
    out = W.kv_first_check(launch, d_log)
    assert out[0].startswith("TP1 Wake 1: Vorhersage LATE kv 3264 MiB, Metall LATE need 3300")
    assert "MATCH" in out[0] and "need +36 MiB" in out[0]
    assert out[1].startswith("TP0 Wake 1: Vorhersage LATE") and "MISS" in out[1]


def test_the_launcher_hands_the_cut_kv_only_under_the_cut():
    src = inspect.getsource(L.log_wake_credit_solve)
    assert "kv_first_mib=" in src and "kv_token_share" in src
    assert "kv_first_mib=kv_first_mib" in inspect.getsource(L.log_wake_credit_solve_pd)
