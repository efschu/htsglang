"""H91 Teil B (Nutzer-Design 25.09., Stufe 1): D decodes with two seats; the
youngest request PARKS (span retained, HiCache write/read path) instead of
being discarded, and parked requests resume before newer ones.

Pure policy tests of weg2/d_seats.py -- every verdict is a function of
replicated request state, so the desk tests are hermetic."""
from __future__ import annotations

import os
import types

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

from sglang.srt.weg2 import d_seats as ds  # noqa: E402

D = {"SGLANG_WEG2_GROUP": "D"}


def _req(rid, seq, *, site=None, n_in=100, n_out=10, spill_class=None):
    r = types.SimpleNamespace(
        rid=rid, kv_arrival_seq=seq, origin_input_ids=[0] * n_in, output_ids=[0] * n_out,
        is_fast_lane=False, spill_class=spill_class,
    )
    if site is not None:
        ds.mark_parked(r, site, epoch=7, now=100.0)
    return r


def test_the_park_is_standard_on_group_d_only():
    assert ds.d_park_active(D)
    assert ds.d_park_active({"SGLANG_WEG2_GROUP": "d"})
    assert not ds.d_park_active({"SGLANG_WEG2_GROUP": "P"})
    assert not ds.d_park_active({})
    assert not ds.d_park_active({**D, ds.PARK_ENV: "0"})


def test_group_env_mirrors_corridor_guard():
    from sglang.srt.managers.corridor_guard import GROUP_ENV

    assert ds.GROUP_ENV == GROUP_ENV


def test_the_youngest_parks_by_the_kvso_protection_key():
    old, young = _req("old", 1), _req("young", 2)
    # batch order young-first: the retraction still pops the youngest
    order = ds.retraction_order([young, old], spec_active=False)
    assert [young, old][order[-1]] is young
    assert [young, old][order[0]] is old
    # a 'preferred' spill class parks first even when older (kvso key)
    pref = _req("pref", 0, spill_class="preferred")
    order = ds.retraction_order([pref, young], spec_active=False)
    assert [pref, young][order[-1]] is pref


def test_under_spec_only_the_back_leaves_and_a_wrong_back_is_named_not_reordered():
    old, young = _req("old", 1), _req("young", 2)
    assert ds.retraction_order([old, young], spec_active=True) == [0, 1]
    assert ds.retraction_order([young, old], spec_active=True) is None


def test_parked_first_oldest_first_the_rest_keeps_the_group_order():
    n1, n2 = _req("n1", 9), _req("n2", 5)  # group order n1 before n2 is KEPT
    p_young, p_old = _req("py", 4, site=ds.SITE_PRESSURE), _req("po", 3, site=ds.SITE_FLIP)
    got = ds.order_waiting([n1, p_young, n2, p_old])
    assert [r.rid for r in got] == ["po", "py", "n1", "n2"]
    assert [r.rid for r in ds.park_running_order([_req("b", 2), _req("a", 1)])] == ["a", "b"]


def test_a_pressure_park_resumes_when_the_older_request_is_done():
    old = _req("old", 1)
    young = _req("young", 2, site=ds.SITE_PRESSURE)
    new = _req("new", 3)
    gate = ds.admission_gate([young, new], running=[old])
    assert gate.barrier
    assert gate.skip(young) == "weg2_d_park_older_live"
    assert gate.skip(new) == "weg2_d_park_first"  # the newcomer never takes its seat
    # the older one finished: the young one comes back, the newcomer still waits
    gate = ds.admission_gate([young, new], running=[])
    assert gate.skip(young) is None
    assert gate.skip(new) == "weg2_d_park_first"


def test_a_flip_park_resumes_at_once_and_holds_newcomers_back():
    a = _req("a", 1, site=ds.SITE_FLIP)
    b = _req("b", 2, site=ds.SITE_FLIP)
    new = _req("new", 3)
    gate = ds.admission_gate([a, b, new], running=[])
    assert gate.skip(a) is None and gate.skip(b) is None
    assert gate.skip(new) == "weg2_d_park_first"


def test_a_parked_request_still_settling_outside_the_queue_holds_the_barrier():
    settling = _req("s", 1, site=ds.SITE_FLIP)
    new = _req("new", 2)
    gate = ds.admission_gate([new], running=[], pending_outside=[settling])
    assert gate.barrier and gate.skip(new) == "weg2_d_park_first"
    assert ds.admission_gate([new], running=[]).skip(new) is None  # nothing parked: stock


def test_stage2_early_resume_is_off_by_default_and_hysteretic_when_armed():
    old = _req("old", 1)
    young = _req("young", 2, site=ds.SITE_PRESSURE, n_in=1000, n_out=0)
    off = ds.ResumeBook.from_env({})
    assert off.margin_tokens == -1
    assert ds.admission_gate([young], running=[old], avail_tokens=10**9, resume_book=off).skip(young)
    book = ds.ResumeBook.from_env({ds.RESUME_MARGIN_ENV: "100", ds.RESUME_STEPS_ENV: "2"})
    g1 = ds.admission_gate([young], running=[old], avail_tokens=1100, resume_book=book)
    assert g1.skip(young) == "weg2_d_park_older_live"  # held once, not yet twice
    g2 = ds.admission_gate([young], running=[old], avail_tokens=1100, resume_book=book)
    assert g2.skip(young) is None
    g3 = ds.admission_gate([young], running=[old], avail_tokens=1099, resume_book=book)
    assert g3.skip(young) == "weg2_d_park_older_live"  # the streak broke


def test_awake_requeue_waits_for_the_bound_and_can_be_disabled():
    p = _req("p", 1, site=ds.SITE_FLIP)  # parked at t=100
    assert not ds.awake_requeue_due([p], now=129.0, bound_s=30.0)
    assert ds.awake_requeue_due([p], now=130.0, bound_s=30.0)
    assert not ds.awake_requeue_due([p], now=10**6, bound_s=0.0)
    assert ds.awake_requeue_s({}) == ds.AWAKE_REQUEUE_DEFAULT_S


def test_mamba_slot_formula_is_the_runtimes():
    from sglang.srt.model_executor import model_runner_kv_cache_mixin as m

    assert ds.MAMBA_RATIO_BASE == m.MAMBA_CACHE_SIZE_MAX_RUNNING_REQUESTS_RATIO
    assert ds.MAMBA_RATIO_OVERLAP_EXTRA == m.MAMBA_CACHE_V2_ADDITIONAL_RATIO_OVERLAP
    assert ds.MAMBA_SAFETY == m.MAMBA_AUTO_SAFETY_MARGIN
    # the NF references' 393.2 MiB are 7 slots = one seat; two seats are 13
    assert ds.mamba_slots_for_seats(1) == 7
    assert ds.mamba_slots_for_seats(2) == 13


def test_graph_list_coverage():
    assert ds.graph_bs_covers([1, 2], 2)
    assert not ds.graph_bs_covers([1], 2)
    assert ds.graph_bs_covers(None, 6)
