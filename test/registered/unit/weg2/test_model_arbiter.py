"""Dual-model flip: the model arbiter's switch policy (pure).

Concept DUAL-MODEL-FLIP-KONZEPT-1001.md section 4 / 8.5: idle switch, T_max
anti-starvation, T_min dwell, optional time slice. The policy is a pure
function of one snapshot; the side-car that polls the fronts and calls
/weg2/model_sleep|model_wake is wiring around it.
"""
import pytest

from sglang.srt.weg2.model_arbiter import (
    ArbiterConfig,
    ArbiterConfigRefused,
    ModelLoad,
    Snapshot,
    decide,
)


def snap(awake="27b", now=1000.0, last=0.0, switching=False, a=(0, 0.0), b=(0, 0.0), switch_s=6.0):
    return Snapshot(
        now=now, awake=awake, last_switch_at=last, switching=switching,
        switch_s=switch_s,
        models={"27b": ModelLoad(outstanding=a[0], oldest_wait_s=a[1]),
                "nf": ModelLoad(outstanding=b[0], oldest_wait_s=b[1])})


CFG = ArbiterConfig(t_max_s=90.0, t_min_factor=5.0, t_min_floor_s=10.0)


def test_no_demand_on_the_sleeping_model_stays():
    d = decide(snap(a=(3, 0.0), b=(0, 0.0)), CFG)
    assert (d.action, d.reason) == ("stay", "no-demand")


def test_idle_switch_when_awake_has_nothing_and_other_waits():
    d = decide(snap(a=(0, 0.0), b=(1, 2.0)), CFG)
    assert (d.action, d.target, d.reason, d.park) == ("switch", "nf", "idle", False)


def test_dwell_blocks_even_the_idle_switch():
    # switch_s 6 -> T_min = max(10, 5*6) = 30 s; only 20 s since the last switch
    d = decide(snap(now=1020.0, last=1000.0, a=(0, 0.0), b=(1, 5.0)), CFG)
    assert (d.action, d.reason) == ("stay", "dwell")


def test_dwell_uses_the_floor_when_the_switch_is_fast():
    # switch_s 1 -> 5 s < floor 10 s; 12 s dwell passes
    d = decide(snap(now=1012.0, last=1000.0, a=(0, 0.0), b=(1, 1.0), switch_s=1.0), CFG)
    assert d.action == "switch"


def test_busy_awake_model_keeps_running_below_t_max():
    d = decide(snap(a=(2, 0.0), b=(1, 89.0)), CFG)
    assert (d.action, d.reason) == ("stay", "busy")


def test_t_max_forces_a_switch_and_parks_the_awake_decodes():
    d = decide(snap(a=(2, 0.0), b=(1, 90.0)), CFG)
    assert (d.action, d.target, d.reason, d.park) == ("switch", "nf", "t_max", True)


def test_switch_in_progress_holds():
    d = decide(snap(switching=True, a=(0, 0.0), b=(5, 200.0)), CFG)
    assert (d.action, d.reason) == ("hold", "switching")


def test_time_slice_switches_two_busy_models():
    cfg = ArbiterConfig(t_max_s=600.0, t_min_factor=5.0, t_min_floor_s=10.0, slice_s=300.0)
    assert decide(snap(now=1299.0, last=1000.0, a=(2, 0.0), b=(1, 50.0)), cfg).reason == "busy"
    d = decide(snap(now=1300.0, last=1000.0, a=(2, 0.0), b=(1, 50.0)), cfg)
    assert (d.action, d.reason, d.park) == ("switch", "slice", True)


def test_no_slice_without_a_waiter():
    cfg = ArbiterConfig(t_max_s=600.0, slice_s=300.0)
    assert decide(snap(now=5000.0, last=0.0, a=(2, 0.0), b=(0, 0.0)), cfg).reason == "no-demand"


def test_cold_start_wakes_the_model_with_the_oldest_waiter():
    d = decide(snap(awake=None, a=(1, 3.0), b=(2, 7.0)), CFG)
    assert (d.action, d.target, d.reason) == ("switch", "nf", "cold-start")
    assert decide(snap(awake=None), CFG).action == "stay"


def test_bound_t_max_plus_one_switch_holds_for_t_min_below_t_max():
    # the concept's guarantee: no request waits longer than T_max + one switch.
    # Worst case: a request arrives right after a switch away from its model.
    cfg = ArbiterConfig(t_max_s=90.0, t_min_factor=5.0, t_min_floor_s=10.0)
    for arrival_after_switch in (0.0, 5.0, 29.0):
        waited = 0.0
        while True:
            now = 1000.0 + arrival_after_switch + waited
            d = decide(snap(now=now, last=1000.0, a=(1, 0.0), b=(1, waited)), cfg)
            if d.action == "switch":
                break
            waited += 1.0
        assert waited <= cfg.t_max_s


@pytest.mark.parametrize("kw", [
    dict(t_max_s=0.0),
    dict(t_max_s=60.0, t_min_floor_s=61.0),
    dict(t_max_s=90.0, slice_s=5.0, t_min_floor_s=10.0),
    dict(t_max_s=90.0, t_min_factor=-1.0),
])
def test_config_refuses_settings_that_break_the_bound(kw):
    with pytest.raises(ArbiterConfigRefused):
        ArbiterConfig(**kw)


def test_t_min_above_t_max_from_a_slow_switch_is_clamped_and_named():
    # a measured switch of 30 s with factor 5 would be T_min 150 s > T_max 90 s:
    # the dwell is clamped to T_max so the bound survives, and the decision says so
    d = decide(snap(now=1095.0, last=1000.0, a=(2, 0.0), b=(1, 90.0), switch_s=30.0), CFG)
    assert (d.action, d.reason) == ("switch", "t_max")
    assert "clamped" in d.note
