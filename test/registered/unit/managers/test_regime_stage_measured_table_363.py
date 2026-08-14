# SPDX-License-Identifier: Apache-2.0
"""#363 -- a SOLVED candidate becomes a MEASURED flip target, or is refused.

THE STATE THIS SUITE CHANGES. On the merged line the #363 window measured, on
metal, ``1 stage(s), 1 reachable at runtime, 0 flip target(s)``; #584's card
half then produced a solved candidate and the table refused the whole thing:

    RegimeError: stage table refused (#578): the planner solved 1 stage(s) ...
    but they carry no measurement.

Both refusals are correct. What was missing is the third state -- a candidate
that HAS a measurement -- and the fourth -- a candidate that does not, named
individually instead of taking the table down with it.

FOUR PROPERTIES, and the first is the one that keeps the change honest:

1. WITHOUT a library the behaviour is the old one, statement for statement.
   ``measurements=None`` is the default and the #578 refusal still fires.
2. WITH a library, a priced candidate is promoted and becomes a flip target.
3. WITH a library, an unpriced candidate is DROPPED and NAMED. It is never
   admitted with the solver's placeholder zeros -- which would read as a
   measured gain of zero and let the clock price the flip as free.
4. The refusal reason travels: it is on the plan, in ``summary()``, and it
   distinguishes the four ways a lookup can miss.
"""

from __future__ import annotations

import pytest

from sglang.srt.managers.regime_classifier import (
    REGIME_DECODE_HEAVY,
    REGIME_PREFILL_HEAVY,
    RegimeError,
    Stage,
)
from sglang.srt.managers.regime_stages import apply_measurements, build_stage_table
from sglang.srt.planner.stage_measure_store import (
    StageMeasurement,
    StageMeasurementLibrary,
)
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

RIG = "3:GPU-aaa,GPU-bbb,GPU-ccc"
MODEL = "/models/Qwen3.6-27B-FP8"

BOOTED = Stage(
    name="booted",
    regime=REGIME_DECODE_HEAVY,
    weight_vector=None,
    kv_token_vector=(2, 11, 10),
    vram_budget_mib=(31400, 19300, 19300),
    max_total_num_tokens=280_000,
    measured_gain_pct=0.0,
    measured_band_pct=0.0,
    flip_cost_s=0.0,
)

SOLVED = Stage(
    name="solved-enc",
    regime=REGIME_PREFILL_HEAVY,
    weight_vector=None,
    kv_token_vector=(3, 10, 10),
    vram_budget_mib=(31400, 19300, 19300),
    max_total_num_tokens=260_000,
    measured_gain_pct=0.0,
    measured_band_pct=0.0,
    flip_cost_s=0.0,
    unmeasured=True,
)

DECLARED = [(2, 11, 10), (3, 10, 10)]


def record(**over) -> StageMeasurement:
    kwargs = dict(
        stage="solved-enc",
        regime=REGIME_PREFILL_HEAVY,
        reference="booted",
        rig_key=RIG,
        model_key=MODEL,
        gain_pct=9.0,
        band_pct=1.4,
        flip_cost_s=0.8,
        covered_s_reference=30.0,
        covered_s_stage=31.0,
        boundaries_reference=25,
        boundaries_stage=25,
        flip_samples=2,
        flip_cost_mean_s=0.7,
        source="unit",
    )
    kwargs.update(over)
    return StageMeasurement(**kwargs)


def build(library=None, candidates=(SOLVED,)):
    return build_stage_table(
        booted=BOOTED,
        candidates=list(candidates),
        declared_vectors=DECLARED,
        measurements=library,
        rig_key=RIG,
        model_key=MODEL,
    )


# 1 -- the default path is untouched -----------------------------------------


def test_without_a_library_the_578_refusal_still_fires():
    with pytest.raises(RegimeError) as exc:
        build_stage_table(
            booted=BOOTED, candidates=[SOLVED], declared_vectors=DECLARED
        )
    assert "#578" in str(exc.value)
    assert "measurement pass" in str(exc.value)


# 2 -- a measured candidate becomes a flip target ----------------------------


def test_a_measured_candidate_becomes_a_flip_target():
    plan = build(StageMeasurementLibrary([record()]))
    assert len(plan) == 2
    assert [p.stage.name for p in plan.flip_targets] == ["solved-enc"]
    target = plan.plan_for("solved-enc").stage
    assert target.unmeasured is False
    assert (target.measured_gain_pct, target.measured_band_pct, target.flip_cost_s) == (
        9.0,
        1.4,
        0.8,
    )
    assert plan.summary()["flip_targets"] == 1


def test_the_promoted_stage_keeps_its_solved_layout():
    plan = build(StageMeasurementLibrary([record()]))
    target = plan.plan_for("solved-enc").stage
    assert target.kv_token_vector == (3, 10, 10)
    assert target.max_total_num_tokens == 260_000
    assert plan.plan_for("solved-enc").reach == "reshard"


# 3 -- an unpriced candidate is dropped and named ----------------------------


def test_an_unmeasured_candidate_is_dropped_and_named_not_admitted_with_zeros():
    plan = build(StageMeasurementLibrary([]))
    assert len(plan) == 1
    assert plan.flip_targets == ()
    assert len(plan.measurement_refusals) == 1
    why = plan.measurement_refusals[0]
    assert why.startswith("solved-enc: NOT SELECTABLE")
    assert "never been measured" in why
    assert "stage_measure_pass" in why


def test_the_refusal_is_visible_in_the_summary():
    plan = build(StageMeasurementLibrary([]))
    assert plan.summary()["measurement_refusals"] == list(plan.measurement_refusals)


@pytest.mark.parametrize(
    "over, marker",
    [
        ({"rig_key": "1:GPU-zzz"}, "DIFFERENT card set"),
        ({"model_key": "/models/other"}, "checkpoint"),
        ({"flip_samples": 0, "flip_cost_s": 0.0}, "HAS a measurement and it is refused"),
        ({"gain_pct": 1.0, "band_pct": 1.4}, "does not clear its own band"),
    ],
)
def test_every_way_a_lookup_misses_names_itself(over, marker):
    plan = build(StageMeasurementLibrary([record(**over)]))
    assert plan.flip_targets == ()
    assert marker in plan.measurement_refusals[0]


# 4 -- the seam itself --------------------------------------------------------


def test_apply_measurements_passes_an_already_measured_stage_through_untouched():
    measured = Stage(
        name="measured-dec",
        regime=REGIME_PREFILL_HEAVY,
        weight_vector=None,
        kv_token_vector=(3, 10, 10),
        vram_budget_mib=(31400, 19300, 19300),
        max_total_num_tokens=260_000,
        measured_gain_pct=4.0,
        measured_band_pct=1.0,
        flip_cost_s=0.3,
    )
    kept, refusals = apply_measurements(
        [measured], measurements=StageMeasurementLibrary([]), rig_key=RIG, model_key=MODEL
    )
    assert kept == [measured] and refusals == []


def test_the_production_wiring_reads_the_canon_and_gates_on_the_file(
    tmp_path, monkeypatch
):
    """The #584 T4 lesson: a suite can pass while the wiring is absent.

    Every other test here constructs its own library, so none of them would
    notice if `build_regime_stage_table` stopped passing one. This one goes
    through the real `_stage_measurements` with nothing but a file on disk.
    """
    import types

    from sglang.srt.managers import regime_runtime

    store = tmp_path / "stage_measurements.json"
    monkeypatch.setenv("SGLANG_STAGE_MEASUREMENTS", str(store))
    server_args = types.SimpleNamespace(model_path=MODEL)

    # No file -> the pre-#584 path, and nothing was read.
    assert regime_runtime._stage_measurements(server_args) == (None, "", "")

    # A file -> the canon binds, with the rig resolved through NVML's own
    # IdentityMap (patched here; the point is that nothing else is consulted).
    StageMeasurementLibrary([record()]).save(str(store))
    fake = types.SimpleNamespace(
        cards=[types.SimpleNamespace(uuid=u) for u in ("GPU-ccc", "GPU-aaa", "GPU-bbb")]
    )
    monkeypatch.setattr(
        "sglang.srt.registry.nvml.identity_map", lambda *a, **k: fake, raising=False
    )
    library, rig, model = regime_runtime._stage_measurements(server_args)
    assert library is not None and len(library) == 1
    assert rig == RIG  # sorted, so enumeration order cannot change it
    assert model == MODEL
    rec, why = library.lookup("solved-enc", rig=rig, model=model)
    assert why == "measured" and rec is not None


def test_the_boot_table_builder_actually_hands_the_canon_over():
    """The other half of the wiring, checked in the syntax tree.

    `_stage_measurements` can be perfect and unused. The production caller has
    to PASS what it loaded, and a stub scheduler rich enough to drive
    `build_regime_stage_table` end to end would be a fixture testing itself --
    so this walks the function's AST instead, the way the phase-2 import-graph
    test pins that observe cannot reach an actuator.
    """
    import ast
    import inspect

    from sglang.srt.managers import regime_runtime

    tree = ast.parse(inspect.getsource(regime_runtime.build_regime_stage_table))
    calls = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and getattr(node.func, "id", "") == "build_stage_table"
    ]
    assert calls, "build_regime_stage_table no longer calls build_stage_table"
    passed = {kw.arg for call in calls for kw in call.keywords}
    assert {"measurements", "rig_key", "model_key"} <= passed, (
        f"build_stage_table is called without the measurement canon "
        f"({sorted(passed)}); the boot would silently take the pre-#584 path "
        f"even on a rig that HAS measurements on disk."
    )


def test_a_canon_that_cannot_be_read_is_skipped_not_raised(tmp_path, monkeypatch):
    """An unreadable canon must not take the boot down, and must not silently
    admit anything either: it lands on the same refusal a missing file does."""
    import types

    from sglang.srt.managers import regime_runtime

    store = tmp_path / "stage_measurements.json"
    store.write_text("{not json")
    monkeypatch.setenv("SGLANG_STAGE_MEASUREMENTS", str(store))
    assert regime_runtime._stage_measurements(
        types.SimpleNamespace(model_path=MODEL)
    ) == (None, "", "")


def test_a_mixed_table_keeps_the_priced_one_and_refuses_the_other():
    other = Stage(
        name="solved-maxkv",
        regime=REGIME_DECODE_HEAVY,
        weight_vector=None,
        kv_token_vector=(4, 9, 10),
        vram_budget_mib=(31400, 19300, 19300),
        max_total_num_tokens=300_000,
        measured_gain_pct=0.0,
        measured_band_pct=0.0,
        flip_cost_s=0.0,
        unmeasured=True,
    )
    kept, refusals = apply_measurements(
        [SOLVED, other],
        measurements=StageMeasurementLibrary([record()]),
        rig_key=RIG,
        model_key=MODEL,
    )
    assert [s.name for s in kept] == ["solved-enc"]
    assert len(refusals) == 1 and "solved-maxkv" in refusals[0]
