"""Dual-model M0 probe: the pure summary over recorded NVML steps.

The probe itself needs three GPUs and runs in a metal window; what is pinned
here is the arithmetic that turns its recorded steps into the numbers the
design addendum (DUAL-MODEL-FLIP-KONZEPT-1001.md section 8) is decided on:
per-slot context deltas, BAR1 per built group, the deep-sleep floor per
process and the build/close wall-time range of the model-flip cycles.
"""
import importlib.util
import pathlib

_PATH = pathlib.Path(__file__).resolve().parents[4] / "benchmark" / "dual_m0_probe.py"
_spec = importlib.util.spec_from_file_location("dual_m0_probe", _PATH)
probe = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(probe)


def _card(used, bar1, procs=0):
    return {"name": "x", "used_mib": used, "total_mib": 20480, "bar1_used_mib": bar1,
            "bar1_total_mib": 256, "procs": procs, "proc_used_mib": []}


def _step(name, used, bar1):
    return {"step": name, "nvml": {"0": _card(used, bar1)}}


def test_context_deltas_are_per_slot_not_cumulative():
    rec = {"steps": [
        _step("baseline", 300, 2),
        _step("ctx+27B-P", 560, 2),
        _step("ctx+27B-D", 820, 2),
        _step("ctx+NF-P", 1080, 2),
        _step("ctx+NF-D", 1340, 2),
        _step("ctx+gloo(all 4 slots)", 1400, 2),
    ]}
    s = probe.summarize(rec)
    assert s["ctx_delta_mib_per_slot"] == {"0": [260, 260, 260, 260]}
    assert s["bar1_baseline_mib"] == {"0": 2}


def test_bar1_per_group_and_deep_sleep_floor():
    rec = {"steps": [
        _step("baseline", 300, 2),
        _step("build 27B-P", 1700, 122),
        _step("build 27B-D", 1800, 210),
        _step("build NF-P", 1800, 210),  # refused: nothing added
        _step("deep-sleep floor (12 procs, no windows, cache emptied)", 2300, 2),
    ]}
    s = probe.summarize(rec)
    assert s["bar1_after_build_mib"]["27B-P"] == {"0": 122}
    assert s["bar1_after_build_mib"]["NF-P"] == {"0": 210}
    assert s["deep_sleep_floor_mib_per_proc"] == {"0": 500}


def test_flip_cycle_ranges_count_only_successful_builds():
    rec = {"steps": [_step("baseline", 0, 0)], "flips": [
        {"build": {"27B-P": {"ok": True, "wall_ms": 120}, "27B-D": {"ok": True, "wall_ms": 90}},
         "close": {"27B-P": {"wall_ms": 8}, "27B-D": {"wall_ms": 6}}},
        {"build": {"NF-P": {"ok": False, "wall_ms": 60000}, "NF-D": {"ok": True, "wall_ms": 95}},
         "close": {"NF-P": {"wall_ms": 1}, "NF-D": {"wall_ms": 7}}},
    ]}
    s = probe.summarize(rec)
    assert s["group_build_wall_ms"] == {"n": 3, "min": 90, "max": 120}
    assert s["group_close_wall_ms"] == {"n": 4, "min": 1, "max": 8}
    assert s["flip_builds_failed"] == 1


def test_no_baseline_no_summary():
    assert probe.summarize({"steps": []}) == {}
