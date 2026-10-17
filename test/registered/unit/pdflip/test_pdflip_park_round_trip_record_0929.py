"""PARK ROUND-TRIP RECORD (29.09.; split out of the PARK-COLLECT-WINDOW tests on 17.10.).

SOLL v2 (R7) replaced the Ski-Rental collect window by the set's T_halb timer, so the window's own tests are gone with
it. What stays, because the machinery stays (``phase_policy.park_round_trip_record`` / ``read_park_round_trip`` /
``warm_resume_ms``, ``Front._note_park_round_trip``): the measured round trip per checkpoint x form (never shared between
the models), the warm-resume median, and the sidecar write of the boot's first warm round trips."""
from __future__ import annotations

import json
import os
import types

import pytest

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

from flliper.srt.environ import envs  # noqa: E402
from flliper.srt.pdflip import phase_policy as pp  # noqa: E402


def test_round_trip_record_is_per_checkpoint_and_form(tmp_path):
    nf = "Qwen3.8-Flash-Next|arch=moe,experts=offload,draft=mtp,kv=full,flip=on"
    q27 = "Qwen3.8-27B|arch=dense,experts=none,draft=dflash,kv=full,flip=on"
    rec = lambda key, rt_ms, at: pp.park_round_trip_record(  # noqa: E731
        form_key=key, dp_ms=rt_ms, pd_ms=2000.0, resume_ms=1000.0, boot_tag="b", commit=None, at=at)
    path = tmp_path / "weg2_measured_record.json"
    path.write_text(json.dumps({"samples": [
        {"group": "P", "rss_shmem_gib": 1.0, "at": "2026-09-29 09:00:00,000"},
        rec(nf, 2525.0, "2026-09-29 08:40:00,000"), rec(nf, 2600.0, "2026-09-29 08:50:00,000"),
        rec(q27, 2167.0, "2026-09-29 09:10:00,000")]}))
    got = pp.read_park_round_trip(str(path), nf)
    assert got["round_trip_s"] == pytest.approx(5.6) and got["form_key"] == nf   # newest of NF only
    assert pp.read_park_round_trip(str(path), q27)["round_trip_s"] == pytest.approx(5.167)
    assert pp.read_park_round_trip(str(path), "other|x") is None
    assert pp.read_park_round_trip(str(tmp_path / "missing.json"), nf) is None
    assert pp.read_park_round_trip(str(path), "") is None                        # no form, no seed
    assert pp.park_round_trip_s(2525.0, 0.0, 1000.0) is None                     # K7 before its flip


def _flip_log(dp_ms=2525, pd_ms=2218, warm_pairs=3, cold_ms=24600):
    """The boot's cold first flip (H34b: lane registration) and warm pairs."""
    log = [{"sleep": "D", "wake": "P", "flip_ms": cold_ms}, {"sleep": "P", "wake": "D", "flip_ms": pd_ms}]
    for _ in range(warm_pairs):
        log += [{"sleep": "D", "wake": "P", "flip_ms": dp_ms}, {"sleep": "P", "wake": "D", "flip_ms": pd_ms}]
    return log


def test_warm_resume_median_ignores_the_cold_first_and_one_outlier():
    # the first resume (first park, JIT, pinning) is never a sample
    assert pp.warm_resume_ms([11800.0]) is None
    assert pp.warm_resume_ms([11800.0, 3100.0]) == 3100.0
    # one 10 s outlier among warm resumes does not move the median
    assert pp.warm_resume_ms([11800.0, 3100.0, 3123.0, 10000.0, 3150.0]) == pytest.approx(3136.5)
    # only the last `window` warm samples count
    assert pp.warm_resume_ms([9e3, 1e4, 1e4, 1e4, 3000.0, 3000.0, 3000.0], window=3) == 3000.0


def test_front_writes_its_first_round_trips_per_form(tmp_path):
    from flliper.srt.pdflip import front as F

    path = tmp_path / "weg2_measured_record.json"
    submitted = []
    ns = types.SimpleNamespace(
        _park_form_key="Qwen3.8-27B|arch=dense", measured_record=str(path), _park_rt_written=0,
        _resume_ms_log=[9800.0], tag="dkr27b", commit="bb82fbcb68", epoch=4,
        flip_log=_flip_log(dp_ms=2167, pd_ms=2421, warm_pairs=0)[:1],
        _sidecar_submit=lambda fn, *a: submitted.append(fn(*a)))
    ns._park_warm_legs_ms = lambda: F.Front._park_warm_legs_ms(ns)
    with envs.FLLIPER_PDFLIP_ENABLE_PARK_COLLECT_WINDOW.override(True):
        # the cold first park (first flip 24.6 s, first resume 9.8 s) is no record
        F.Front._note_park_round_trip(ns)
        assert submitted == [] and not path.exists()
        # warm flips and resumes: the warm medians are written, the cold ones never
        ns.flip_log = _flip_log(dp_ms=2167, pd_ms=2421, warm_pairs=2)
        ns._resume_ms_log = [9800.0, 2351.0, 2360.0, 2340.0]
        for _ in range(pp.PARK_ROUND_TRIP_RECORDS + 2):
            F.Front._note_park_round_trip(ns)
        assert len(submitted) == pp.PARK_ROUND_TRIP_RECORDS            # the first ones only
        got = pp.read_park_round_trip(str(path), "Qwen3.8-27B|arch=dense")
        assert got["round_trip_s"] == pytest.approx(6.939) and got["boot_tag"] == "dkr27b"
        assert got["resume_ms"] == 2351.0 and got["dp_ms"] == 2167.0          # no cold sample in it
        ns._park_form_key, ns._park_rt_written = "", 0                   # no form: nothing written
        F.Front._note_park_round_trip(ns)
        assert len(submitted) == pp.PARK_ROUND_TRIP_RECORDS
