# SPDX-License-Identifier: Apache-2.0
"""#584 second half -- the per-stage measurement canon.

WHAT THIS SUITE IS FOR. The #363 window failed with "0 flip targets" and the
code named the owner: a planner-solved stage carries no ``measured_gain_pct``,
``measured_band_pct`` or ``flip_cost_s``, and the solver cannot predict any of
the three. This module is where the three live once measured. The suite's
duties are symmetric, in the same shape as the ms-clock suite's:

* IT MUST ADMIT. A complete, attributed, long-enough record measured on THIS
  card set for THIS checkpoint is usable and is returned by ``lookup``.
* IT MUST REFUSE, BY NAME. Every one of the four ways a lookup misses --
  never measured, other card set, other checkpoint, self-refused -- returns a
  DIFFERENT reason, because the four have different remedies and a consumer
  that receives a bare ``None`` writes "not measured" over all of them.

The load-bearing tests are ``test_a_record_from_another_rig_is_refused_by_name``
and ``test_an_unpriced_flip_is_refused_rather_than_read_as_free``. The first
is #584's own borrowed-rates lesson (rates from an earlier shift were 12-22 %
wrong after a power-target change on the same cards); the second is the
transient census's (an unpriced term must not read as a free one).
"""

from __future__ import annotations

import json
import os

import pytest

from sglang.srt.planner.stage_measure_store import (
    MIN_MEASURE_SECONDS,
    StageMeasurement,
    StageMeasurementError,
    StageMeasurementLibrary,
    rig_key_from_uuids,
    stage_measure_path,
)
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

RIG = rig_key_from_uuids(["GPU-aaa", "GPU-bbb", "GPU-ccc"])
OTHER_RIG = rig_key_from_uuids(["GPU-zzz"])
MODEL = "/models/Qwen3.6-27B-FP8"


def make_record(**over) -> StageMeasurement:
    kwargs = dict(
        stage="solved-enc",
        regime="prefill_heavy",
        reference="booted",
        rig_key=RIG,
        model_key=MODEL,
        gain_pct=8.0,
        band_pct=1.5,
        flip_cost_s=0.9,
        avs_a_floor_pct=1.2,
        drift_pct=1.5,
        covered_s_reference=42.0,
        covered_s_stage=44.0,
        boundaries_reference=30,
        boundaries_stage=31,
        flip_samples=3,
        flip_cost_mean_s=0.7,
        source="363-act window boots B1/B4, 2026-08-14",
    )
    kwargs.update(over)
    return StageMeasurement(**kwargs)


def test_the_rig_key_is_sorted_so_enumeration_order_cannot_change_it():
    assert rig_key_from_uuids(["GPU-b", "GPU-a"]) == rig_key_from_uuids(
        ["GPU-a", "GPU-b"]
    )
    assert rig_key_from_uuids(["GPU-a", "GPU-b"]).startswith("2:")


def test_an_empty_rig_key_is_refused_because_it_would_match_every_rig():
    with pytest.raises(StageMeasurementError) as exc:
        rig_key_from_uuids([])
    assert "match every rig" in str(exc.value)


def test_a_duplicated_uuid_is_refused_as_a_per_rank_list():
    with pytest.raises(StageMeasurementError):
        rig_key_from_uuids(["GPU-a", "GPU-a"])


def test_the_path_prefers_an_explicit_argument_then_the_env(monkeypatch, tmp_path):
    explicit = str(tmp_path / "explicit.json")
    assert stage_measure_path(explicit) == explicit
    monkeypatch.setenv("SGLANG_STAGE_MEASUREMENTS", str(tmp_path / "env.json"))
    assert stage_measure_path() == str(tmp_path / "env.json")


def test_the_default_path_sits_beside_the_card_library(monkeypatch, tmp_path):
    monkeypatch.delenv("SGLANG_STAGE_MEASUREMENTS", raising=False)
    monkeypatch.setenv("SGLANG_CARD_LIBRARY", str(tmp_path / "cards" / "lib.json"))
    got = stage_measure_path()
    assert os.path.dirname(got) == str(tmp_path / "cards")
    assert got.endswith("stage_measurements.json")


# -- the refusal surface -----------------------------------------------------


def test_a_complete_record_is_usable():
    assert make_record().usable
    assert make_record().refusals == []


def test_an_unattributed_record_is_refused():
    refusals = make_record(source="  ").refusals
    assert any("unattributed" in r for r in refusals)


def test_an_unpriced_flip_is_refused_rather_than_read_as_free():
    """The transient census's lesson, applied to the flip cost."""
    refusals = make_record(flip_samples=0, flip_cost_s=0.0).refusals
    assert any("not instrumented" in r for r in refusals)


def test_an_arm_below_the_ten_second_floor_is_refused_naming_the_arm():
    refusals = make_record(covered_s_stage=4.0).refusals
    assert any("the stage arm covers 4.0 s" in r for r in refusals)
    assert MIN_MEASURE_SECONDS == 10.0


def test_a_gain_inside_its_own_band_is_refused_360():
    refusals = make_record(gain_pct=1.0, band_pct=1.5).refusals
    assert any("does not clear its own band" in r for r in refusals)


def test_a_gain_exactly_on_the_band_does_not_clear_it():
    assert not make_record(gain_pct=1.5, band_pct=1.5).usable


def test_a_negative_band_is_refused_at_construction():
    with pytest.raises(StageMeasurementError):
        make_record(band_pct=-0.1)


def test_a_record_without_a_reference_is_refused():
    with pytest.raises(StageMeasurementError) as exc:
        make_record(reference="")
    assert "gain OVER something" in str(exc.value)


# -- lookup: four misses, four reasons ---------------------------------------


def test_a_measured_stage_is_returned():
    lib = StageMeasurementLibrary([make_record()])
    rec, why = lib.lookup("solved-enc", rig=RIG, model=MODEL)
    assert rec is not None and why == "measured"


def test_a_stage_never_measured_names_the_pass_that_measures_it():
    lib = StageMeasurementLibrary([make_record()])
    rec, why = lib.lookup("solved-dec", rig=RIG, model=MODEL)
    assert rec is None
    assert "never been measured" in why and "stage_measure_pass" in why


def test_a_record_from_another_rig_is_refused_by_name():
    lib = StageMeasurementLibrary([make_record(rig_key=OTHER_RIG)])
    rec, why = lib.lookup("solved-enc", rig=RIG, model=MODEL)
    assert rec is None
    assert "DIFFERENT card set" in why and OTHER_RIG in why


def test_a_record_for_another_checkpoint_is_refused_by_name():
    lib = StageMeasurementLibrary([make_record(model_key="/models/other")])
    rec, why = lib.lookup("solved-enc", rig=RIG, model=MODEL)
    assert rec is None
    assert "checkpoint" in why


def test_a_self_refused_record_is_reported_as_refused_not_as_absent():
    lib = StageMeasurementLibrary([make_record(flip_samples=0, flip_cost_s=0.0)])
    rec, why = lib.lookup("solved-enc", rig=RIG, model=MODEL)
    assert rec is None
    assert "HAS a measurement and it is refused" in why


# -- persistence -------------------------------------------------------------


def test_save_then_load_round_trips(tmp_path):
    path = str(tmp_path / "store.json")
    StageMeasurementLibrary([make_record()]).save(path)
    back = StageMeasurementLibrary.load(path)
    assert len(back) == 1
    rec, _ = back.lookup("solved-enc", rig=RIG, model=MODEL)
    assert rec is not None and rec.gain_pct == 8.0


def test_the_staging_file_carries_the_pid_so_two_writers_cannot_share_it(tmp_path):
    """The 2026-08-14 census corruption, pinned in the other direction."""
    path = str(tmp_path / "store.json")
    lib = StageMeasurementLibrary([make_record()])
    lib.save(path)
    leftovers = [p for p in os.listdir(tmp_path) if p.endswith(".tmp")]
    assert leftovers == []
    # The name is derived from the pid, so two processes cannot collide.
    assert f"{path}.{os.getpid()}.tmp" != f"{path}.{os.getpid() + 1}.tmp"


def test_a_missing_store_is_an_empty_library_not_an_error(tmp_path):
    assert len(StageMeasurementLibrary.load(str(tmp_path / "absent.json"))) == 0


def test_a_corrupt_store_raises_because_it_would_read_as_never_measured(tmp_path):
    path = tmp_path / "store.json"
    path.write_text("{not json")
    with pytest.raises(StageMeasurementError) as exc:
        StageMeasurementLibrary.load(str(path))
    assert "would read as 'never measured'" in str(exc.value)


def test_an_unknown_field_is_refused_rather_than_ignored(tmp_path):
    path = tmp_path / "store.json"
    payload = {"version": 1, "measurements": [make_record().to_json()]}
    payload["measurements"][0]["confidence_pct"] = 99
    path.write_text(json.dumps(payload))
    with pytest.raises(StageMeasurementError) as exc:
        StageMeasurementLibrary.load(str(path))
    assert "unknown field" in str(exc.value)
