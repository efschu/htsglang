# SPDX-License-Identifier: Apache-2.0
"""#584 second half -- the pass that turns #363 traces into a canon record.

The pass measures nothing new: it reads the controller's own verdict trace
(which already carries the group's ms/round split per boundary) and the
actuator's own DONE line (which already times the flip), and projects them
onto the three numbers ``StageTable`` refuses a candidate for lacking.

So the suite's duty is the projection and, above all, the REFUSALS. Each one
below corresponds to a way a previous measurement window produced a number
that did not mean what it said:

* no A-vs-A floor           -> a delta with a sign and no scale;
* drift read as noise       -> #459, a monotone 13.0 % reported as a floor;
* an arm with no summary     -> "zero so far" read as "zero";
* a boot with no stage clock -> ms_decision is None on every record (#363
  window F4), which must refuse by name and not as an empty series;
* two ranks that disagree    -> two boots averaged into one that never ran;
* no instrumented flip       -> an unpriced term reading as a free one.
"""

from __future__ import annotations

import json

import pytest

from sglang.srt.planner.stage_measure_pass import (
    MIN_BOUNDARIES,
    ArmSeries,
    StageMeasurePassError,
    avs_a_floor_pct,
    band_from_floor_and_drift,
    build_measurement,
    drift_pct,
    flip_seconds_from_log,
    gain_pct,
    main,
    merge_rank_arms,
    read_arm,
)
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=8, suite="base-a-test-cpu")

RIG = "3:GPU-aaa,GPU-bbb,GPU-ccc"
MODEL = "/models/Qwen3.6-27B-FP8"


def write_trace(
    path,
    round_ms,
    *,
    regime="prefill_heavy",
    interval=10,
    rank=0,
    summary=True,
    with_clock=True,
):
    """A verdict trace in the exact shape ``regime_runtime._write_trace`` writes."""
    lines = [json.dumps({"kind": "header", "mode": "act", "rank": rank, "interval": interval})]
    for i, ms in enumerate(round_ms):
        decision = (
            {
                "target": None,
                "reason": "held",
                "signal_pct": 0.0,
                "band_pct": 1.0,
                "mean_total_ms": ms,
                "mean_wait_share": 0.4,
            }
            if with_clock
            else None
        )
        lines.append(
            json.dumps(
                {
                    "kind": "verdict",
                    "rank": rank,
                    "round": (i + 1) * interval,
                    "regime": regime,
                    "ms_decision": decision,
                }
            )
        )
    if summary:
        lines.append(json.dumps({"kind": "summary", "rank": rank, "rounds": 100}))
    path.write_text("\n".join(lines) + "\n")
    return str(path)


def series(mean, n=30):
    return [mean] * n


# -- reading an arm ----------------------------------------------------------


def test_a_trace_without_a_summary_line_is_refused(tmp_path):
    p = write_trace(tmp_path / "t.jsonl", series(100.0), summary=False)
    with pytest.raises(StageMeasurePassError) as exc:
        read_arm(p)
    assert "killed" in str(exc.value)


def test_a_boot_without_the_stage_clock_refuses_by_name(tmp_path):
    p = write_trace(tmp_path / "t.jsonl", series(100.0), with_clock=False)
    with pytest.raises(StageMeasurePassError) as exc:
        read_arm(p)
    assert "--regime-stage-clock" in str(exc.value)


def test_the_regime_filter_selects_the_labelled_boundaries(tmp_path):
    p = write_trace(tmp_path / "t.jsonl", series(100.0), regime="decode_heavy")
    with pytest.raises(StageMeasurePassError):
        read_arm(p, regime="prefill_heavy")
    assert len(read_arm(p, regime="decode_heavy")) == 30


def test_warmup_drops_the_leading_boundaries(tmp_path):
    p = write_trace(tmp_path / "t.jsonl", list(range(1, 31)))
    assert len(read_arm(p, warmup=5)) == 25


def test_a_round_window_selects_one_arm_out_of_one_boots_trace(tmp_path):
    """A stage arm is often a SEGMENT: the same boot after a /kv_reshard."""
    p = write_trace(tmp_path / "t.jsonl", [100.0] * 15 + [90.0] * 15, interval=10)
    # Rounds are 10, 20, ... 300; the reshard landed at round 150.
    before = read_arm(p, to_round=150)
    after = read_arm(p, from_round=160)
    assert len(before) == 15 and len(after) == 15
    assert before.mean_ms == pytest.approx(100.0)
    assert after.mean_ms == pytest.approx(90.0)


def test_an_empty_round_window_refuses_naming_the_bounds(tmp_path):
    p = write_trace(tmp_path / "t.jsonl", series(100.0), interval=10)
    with pytest.raises(StageMeasurePassError) as exc:
        read_arm(p, from_round=100_000)
    assert "in rounds [100000, None]" in str(exc.value)


def test_covered_seconds_are_device_time_because_the_trace_has_no_clock(tmp_path):
    # 30 boundaries x 10 rounds x 100 ms = 30 s of device time.
    p = write_trace(tmp_path / "t.jsonl", series(100.0), interval=10)
    assert read_arm(p).covered_s == pytest.approx(30.0)


# -- merging ranks -----------------------------------------------------------


def test_agreeing_ranks_merge(tmp_path):
    a = read_arm(write_trace(tmp_path / "a.jsonl", series(100.0), rank=0))
    b = read_arm(write_trace(tmp_path / "b.jsonl", series(100.2), rank=1))
    assert merge_rank_arms([a, b]).mean_ms == pytest.approx(100.0, rel=1e-3)


def test_ranks_that_disagree_are_refused_not_averaged(tmp_path):
    a = read_arm(write_trace(tmp_path / "a.jsonl", series(100.0), rank=0))
    b = read_arm(write_trace(tmp_path / "b.jsonl", series(140.0), rank=1))
    with pytest.raises(StageMeasurePassError) as exc:
        merge_rank_arms([a, b])
    assert "never happened" in str(exc.value)


# -- the arithmetic ----------------------------------------------------------


def arm(values, interval=10):
    return ArmSeries("<mem>", values, interval=interval, rank=0, has_summary=True)


def test_gain_is_percent_of_the_reference_round():
    assert gain_pct(arm(series(100.0)), arm(series(90.0))) == pytest.approx(10.0)


def test_the_floor_is_the_pairs_own_difference():
    assert avs_a_floor_pct(arm(series(100.0)), arm(series(102.0))) == pytest.approx(
        100.0 * 2.0 / 101.0
    )


def test_drift_compares_the_two_halves_of_one_run():
    walking = arm([100.0] * 10 + [110.0] * 10)
    assert drift_pct(walking) == pytest.approx(100.0 * 10.0 / 105.0)
    assert drift_pct(arm(series(100.0))) == pytest.approx(0.0)


def test_the_band_takes_the_larger_of_floor_and_drift_not_their_sum():
    """Two estimates of ONE quantity combine as the larger, not in quadrature."""
    assert band_from_floor_and_drift(2.0, 9.0, 1.0) == 9.0
    assert band_from_floor_and_drift(4.0, 1.0) == 4.0


def test_a_drifting_run_widens_the_band_459():
    quiet = arm(series(100.0))
    walking = arm([100.0] * 15 + [113.0] * 15)
    record = build_measurement(
        stage="s",
        regime="prefill_heavy",
        reference_arm=quiet,
        stage_arm=walking,
        floor_a=quiet,
        floor_b=arm(series(100.5)),
        flip_samples_s=[0.5],
        reference_name="booted",
        model_key=MODEL,
        rig=RIG,
        source="unit",
    )
    assert record.band_pct > record.avs_a_floor_pct
    assert record.drift_pct == pytest.approx(record.band_pct)


# -- flip cost ---------------------------------------------------------------


def test_the_flip_cost_comes_from_the_actuators_own_line():
    log = (
        "KV-RESHARD ARMED 2,11,10\n"
        "KV-RESHARD DONE 2,11,10 -> 3,10,10 (epoch 4) in 812.4 ms: 12 live slots\n"
        "[#631 seam-census] tp->pp rank 0 elapsed_ms=1500.0 trough=...\n"
    )
    assert flip_seconds_from_log(log) == pytest.approx([0.8124, 1.5])


def test_a_log_with_no_flip_yields_nothing_and_the_record_is_refused():
    assert flip_seconds_from_log("nothing happened here") == []
    with pytest.raises(StageMeasurePassError) as exc:
        build_measurement(
            stage="s",
            regime="prefill_heavy",
            reference_arm=arm(series(100.0)),
            stage_arm=arm(series(90.0)),
            floor_a=arm(series(100.0)),
            floor_b=arm(series(100.5)),
            flip_samples_s=[],
            reference_name="booted",
            model_key=MODEL,
            rig=RIG,
            source="unit",
        )
    assert "not defaulted to zero" in str(exc.value)


def test_the_flip_cost_is_the_maximum_not_the_mean():
    record = build_measurement(
        stage="s",
        regime="prefill_heavy",
        reference_arm=arm(series(100.0)),
        stage_arm=arm(series(90.0)),
        floor_a=arm(series(100.0)),
        floor_b=arm(series(100.5)),
        flip_samples_s=[0.2, 0.4, 1.8],
        reference_name="booted",
        model_key=MODEL,
        rig=RIG,
        source="unit",
    )
    assert record.flip_cost_s == pytest.approx(1.8)
    assert record.flip_cost_mean_s == pytest.approx(0.8)


def test_a_thin_arm_is_refused_before_anything_is_computed():
    with pytest.raises(StageMeasurePassError) as exc:
        build_measurement(
            stage="s",
            regime="prefill_heavy",
            reference_arm=arm(series(100.0, n=MIN_BOUNDARIES - 1)),
            stage_arm=arm(series(90.0)),
            floor_a=arm(series(100.0)),
            floor_b=arm(series(100.5)),
            flip_samples_s=[0.5],
            reference_name="booted",
            model_key=MODEL,
            rig=RIG,
            source="unit",
        )
    assert "boundaries" in str(exc.value)


# -- end to end --------------------------------------------------------------


def test_the_cli_writes_a_usable_record(tmp_path, monkeypatch):
    store = tmp_path / "store.json"
    monkeypatch.setenv("SGLANG_STAGE_MEASUREMENTS", str(store))
    ref = write_trace(tmp_path / "ref.jsonl", series(100.0))
    fast = write_trace(tmp_path / "fast.jsonl", series(90.0))
    fa = write_trace(tmp_path / "fa.jsonl", series(100.0))
    fb = write_trace(tmp_path / "fb.jsonl", series(100.4))
    log = tmp_path / "boot.log"
    log.write_text("KV-RESHARD DONE a -> b (epoch 1) in 500.0 ms: ...\n")
    rc = main(
        [
            "--stage", "solved-enc",
            "--regime", "prefill_heavy",
            "--reference", "booted",
            "--model-key", MODEL,
            "--rig-uuid", "GPU-aaa",
            "--rig-uuid", "GPU-bbb",
            "--reference-trace", ref,
            "--stage-trace", fast,
            "--floor-a", fa,
            "--floor-b", fb,
            "--flip-log", str(log),
            "--source", "unit test",
            "--write",
        ]
    )
    assert rc == 0
    from sglang.srt.planner.stage_measure_store import StageMeasurementLibrary

    lib = StageMeasurementLibrary.load(str(store))
    rec, why = lib.lookup("solved-enc", rig="2:GPU-aaa,GPU-bbb", model=MODEL)
    assert why == "measured", why
    assert rec.gain_pct == pytest.approx(10.0)
    assert rec.flip_cost_s == pytest.approx(0.5)
    assert rec.covered_s_stage >= 10.0


def test_the_cli_refuses_a_run_with_no_floor(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("SGLANG_STAGE_MEASUREMENTS", str(tmp_path / "store.json"))
    ref = write_trace(tmp_path / "ref.jsonl", series(100.0))
    rc = main(
        [
            "--stage", "s",
            "--regime", "prefill_heavy",
            "--model-key", MODEL,
            "--reference-trace", ref,
            "--stage-trace", ref,
        ]
    )
    assert rc == 2
    assert "--floor-a" in capsys.readouterr().err
