"""weg2xsn261 (17.09.2026): the first flip of a boot paid ~10 s before its
second tag -- cudaHostRegister of the lane buffers at first use (P lanes
5.3-6.4 s each) and the co-located sleeper's pause() blocked behind it on the
shared card (D-TP0 pause_ms=6174, TP1/TP2 20 ms). The buffers persist for the
boot anyway; `_weg2_prewarm_lanes` registers them on a boot thread, sized
from the same lane derivation the legs use, both roles, both slots.
"""

from __future__ import annotations

import os

import pytest

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

from sglang.srt.managers.scheduler_components import weight_updater as wu  # noqa: E402
from sglang.srt.weg2 import weight_exchange_bounce as bx  # noqa: E402
from sglang.srt.weg2 import weight_exchange_region as xr  # noqa: E402

Manager = wu.SchedulerWeightUpdaterManager


class _FakeRunner:
    model = None
    model_config = None


class _FakeWorker:
    def __init__(self):
        self.model_runner = _FakeRunner()


def _manager(monkeypatch, *, group="P", rank=1):
    monkeypatch.setattr(Manager, "_weg2_group_name", lambda self: group, raising=True)
    monkeypatch.setattr(Manager, "_weg2_rank", lambda self: rank, raising=True)
    monkeypatch.setenv(xr.ENV_REGION_BOOT, "prewarm-boot")
    return Manager(tp_worker=_FakeWorker(), draft_worker=None, tp_cpu_group=None,
                   memory_saver_adapter=None, flush_cache=lambda *a, **k: True,
                   is_fully_idle=lambda *a, **k: True)


def test_every_lane_of_this_rank_is_registered_at_its_max_over_roles_tags_and_both_slots(monkeypatch):
    m = _manager(monkeypatch, group="P", rank=1)
    # bytes per (hook, lane, tag): the collector role sees big p-lanes, the
    # depositor role a big diagonal -- the max over both must win per lane
    table = {
        ("authoritative", "p0", "weights_4"): 233_000_000,
        ("authoritative", "p0", "weights_5"): 1_987_482_680,
        ("source", "p0", "weights_1"): 400_000_000,
        ("authoritative", "c1", "weights_4"): 72_571_968,
        ("source", "c1", "weights_5"): 1_754_669_048,
        ("source", "p3", "weights_6"): 553_646_404,
    }
    calls = []

    def lane_bytes_of(hook, lane_key, tag):
        calls.append((hook, lane_key, tag))
        return table.get((hook, lane_key, tag), 0)

    registered = []

    def persist(path, nbytes, lk):
        registered.append((os.path.basename(path), int(nbytes), lk))

    monkeypatch.setattr(bx, "seq_buffer_depth", lambda: 2)
    out = m._weg2_prewarm_lanes(
        manifests_ready=lambda: True, lane_bytes_of=lane_bytes_of, persist=persist,
        family=["weights_1", "weights_4", "weights_5", "weights_6"])
    assert out["p0"] == 1_987_482_680
    assert out["c1"] == 1_754_669_048
    assert out["p3"] == 553_646_404
    # only this rank's lanes were sized: its diagonal and the cross pairs it touches
    lanes_asked = {lk for _h, lk, _t in calls}
    expected = {"c1"} | {f"p{i}" for i, (s, d) in enumerate(xr.CROSS_PAIRS) if 1 in (s, d)}
    assert lanes_asked == expected
    # both slots per lane, at the lane's max; lanes that carry nothing are skipped
    by_lane = {}
    for name, nbytes, lk in registered:
        by_lane.setdefault(lk, []).append((name, nbytes))
    assert sorted(by_lane) == ["c1", "p0", "p3"]
    for lk, rows in by_lane.items():
        assert sorted(n for n, _b in rows) == sorted([f"{lk}_unit_buffer.bin", f"{lk}_s1_unit_buffer.bin"])
        assert {b for _n, b in rows} == {out[lk]}


def test_no_manifests_within_the_budget_means_no_registration_and_no_raise(monkeypatch):
    m = _manager(monkeypatch)
    registered = []
    out = m._weg2_prewarm_lanes(manifests_ready=lambda: False,
                                lane_bytes_of=lambda *a: 1, persist=lambda *a: registered.append(a),
                                family=["weights_0"], poll_s=0.01, budget_s=0.05)
    assert out == {}
    assert registered == []


def test_a_lane_that_cannot_be_sized_does_not_stop_the_others(monkeypatch):
    m = _manager(monkeypatch, rank=0)

    def lane_bytes_of(hook, lk, tag):
        if lk == "c0":
            raise RuntimeError("no address")
        return 10 if lk == "p0" else 0

    registered = []
    monkeypatch.setattr(bx, "seq_buffer_depth", lambda: 1)
    out = m._weg2_prewarm_lanes(manifests_ready=lambda: True, lane_bytes_of=lane_bytes_of,
                                persist=lambda p, b, lk: registered.append((lk, b)),
                                family=["weights_0"])
    assert out.get("p0") == 10 and out.get("c0", 0) == 0
    assert registered == [("p0", 10)]


def test_the_scheduler_starts_the_prewarm_after_the_manager_is_built():
    from sglang.srt.managers import scheduler as sched
    src = open(sched.__file__).read()
    i = src.index("self.weight_updater = SchedulerWeightUpdaterManager(")
    j = src.index("_weg2_prewarm_lanes_start()", i)
    assert j - i < 900, "the prewarm start is not right after the manager's construction"


def test_start_is_a_no_op_when_the_exchange_is_not_armed(monkeypatch):
    m = _manager(monkeypatch)
    from sglang.srt.weg2 import weight_exchange as wx
    monkeypatch.setattr(wx, "exchange_armed", lambda: False)
    m._weg2_prewarm_lanes_start()
    assert m._weg2_prewarm_thread is None


def test_the_boot_time_prewarm_is_off_by_default_after_xsn263(monkeypatch):
    """xsn263: pinning every lane at its max for both slots on six ranks at
    once (10-12 GiB per rank) latched the host ledger's W98 during the
    launch. Default off; the tmpfs populate before cudaHostRegister keeps
    the lazy first-use registration cheap."""
    m = _manager(monkeypatch)
    from sglang.srt.weg2 import weight_exchange as wx
    monkeypatch.setattr(wx, "exchange_armed", lambda: True)
    monkeypatch.delenv("SGLANG_WEG2_LANE_PREWARM", raising=False)
    m._weg2_prewarm_lanes_start()
    assert m._weg2_prewarm_thread is None


# -- KR 26.09.: form "c" (Befund 2: register_ms=21636 on P's first flip) ------


def test_form_c_registers_only_the_diagonal_lane_slot_0_within_the_priced_bytes(monkeypatch):
    m = _manager(monkeypatch, group="P", rank=0)
    table = {("source", "c0", "weights_1"): 1_845_575_008,
             ("authoritative", "c0", "weights_2"): 1_100_317_440,
             ("source", "p0", "weights_1"): 900_000_000}
    asked = []

    def lane_bytes_of(hook, lk, tag):
        asked.append(lk)
        return table.get((hook, lk, tag), 0)

    registered = []
    monkeypatch.setattr(bx, "seq_buffer_depth", lambda: 2)
    out = m._weg2_prewarm_lanes(
        manifests_ready=lambda: True, lane_bytes_of=lane_bytes_of,
        persist=lambda p, b, lk: registered.append((os.path.basename(p), int(b), lk)),
        family=["weights_1", "weights_2"], diagonal_only=True, max_slots=1,
        priced_lane_bytes=2_111_575_056)
    assert set(asked) == {"c0"}                 # the cross lanes are not even sized
    assert out == {"c0": 1_845_575_008}         # the max over roles and tags
    assert registered == [("c0_unit_buffer.bin", 1_845_575_008, "c0")]  # slot 0 only


def test_form_c_never_pins_more_than_the_ledger_priced(monkeypatch):
    m = _manager(monkeypatch, group="D", rank=2)
    registered = []
    monkeypatch.setattr(bx, "seq_buffer_depth", lambda: 1)
    out = m._weg2_prewarm_lanes(
        manifests_ready=lambda: True, lane_bytes_of=lambda h, lk, t: 3_000_000_000,
        persist=lambda p, b, lk: registered.append(lk), family=["weights_0"],
        diagonal_only=True, max_slots=1, priced_lane_bytes=2_111_575_056)
    assert out == {"c2": 3_000_000_000}
    assert registered == []                      # NOT-BOOKED: the flip registers lazily


def _start_with(monkeypatch, mode, terms_env):
    m = _manager(monkeypatch, group="P", rank=0)
    from sglang.srt.weg2 import weight_exchange as wx
    from sglang.srt.weg2 import xchg_bounce as xb
    monkeypatch.setattr(wx, "exchange_armed", lambda: True)
    monkeypatch.setattr(Manager, "_weg2_bar1_start", lambda self: None, raising=True)
    started = []
    monkeypatch.setattr(Manager, "_weg2_prewarm_lanes",
                        lambda self, **kw: started.append(kw) or {}, raising=True)
    monkeypatch.setenv("SGLANG_WEG2_LANE_PREWARM", mode)
    if terms_env is None:
        monkeypatch.delenv(xb.ENV_BOUNCE_TERMS, raising=False)
    else:
        monkeypatch.setenv(xb.ENV_BOUNCE_TERMS, terms_env)
    m._weg2_prewarm_lanes_start()
    t = m._weg2_prewarm_thread
    if t is not None:
        t.join(5)
    return t, started


_TERMS = ("bytes_per_direction=17637110912,n_layers=64,widest_layer_bytes=1011344880,"
          "pairs=3,depth=1,slot_bytes=134217728,n_lanes=9,max_tag_bytes=2111575056,"
          "lanes_concurrent=0,band_credit=0,n_cross_lanes=0,price_lane_cap=0")


def test_form_c_starts_with_the_published_per_lane_price(monkeypatch):
    t, started = _start_with(monkeypatch, "c", _TERMS)
    assert t is not None
    assert started == [{"diagonal_only": True, "max_slots": 1,
                        "priced_lane_bytes": 2_111_575_056}]


def test_form_c_without_a_published_price_does_not_start(monkeypatch):
    t, started = _start_with(monkeypatch, "c", None)
    assert t is None and started == []


def test_form_1_is_unchanged_every_lane_every_slot(monkeypatch):
    t, started = _start_with(monkeypatch, "1", _TERMS)
    assert t is not None and started == [{}]
