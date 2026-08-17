"""#700: the ReplaySSM byte-identity gate.

The enable decision needs BOTH halves. The wiring contradiction is now resolved
in-code (the kernel header's "NOT yet wired" was stale in all four places it
named; corrected in fused_recurrent_linear_replayssm.py). This is the other
half: the measurement that decides whether the feature is byte-identical.

The flag claims only "numerically correct". The buffered reconstruction sums
rank-1 updates in a different order than the sequential path, so byte-identity
must be assumed ABSENT until measured -- and on a lossless queue that decides
everything.

Four gating rules, each tied to a way this measurement could lie:

* **A-vs-A first.** Until the baseline reproduces itself bit-for-bit, an
  A-vs-B difference measures the harness, not the feature.
* **Probe length capped.** GDN prefill is non-reproducible above ~109 tokens
  (upstream, pre-existing). A longer probe measures that, not ReplaySSM.
* **CPU-sampled inputs.** ``torch.randn`` on device is not arch-identical, so a
  GPU-sampled input cannot support an identity claim.
* **GDN scalar-gate only.** The flag's own text says KDA decode is SLOWER than
  the packed baseline, so a KDA arm cannot inform the enable decision.

Hermetic: pure decision logic, no CUDA.
"""

import pytest

from sglang.srt.planner.replayssm_identity import (
    ProbePlan,
    classify_identity,
    gate_verdict,
    validate_probe_plan,
)


def _plan(**kw):
    d = dict(max_tokens=64, gate="gdn", sample_device="cpu", arms=("off", "on"))
    d.update(kw)
    return ProbePlan(**d)


def test_a_valid_plan_passes():
    validate_probe_plan(_plan())


def test_a_probe_longer_than_the_nondeterminism_ceiling_is_refused():
    with pytest.raises(ValueError, match="109"):
        validate_probe_plan(_plan(max_tokens=256))


def test_gpu_sampled_inputs_are_refused():
    with pytest.raises(ValueError, match="CPU"):
        validate_probe_plan(_plan(sample_device="cuda"))


def test_a_kda_arm_is_refused():
    with pytest.raises(ValueError, match="KDA"):
        validate_probe_plan(_plan(gate="kda"))


def test_identical_outputs_classify_as_byte_identical():
    r = classify_identity(
        baseline_tokens=[1, 2, 3], treatment_tokens=[1, 2, 3], max_abs_logit_delta=0.0
    )
    assert r.byte_identical
    assert not r.changes_emitted_tokens


def test_same_tokens_but_nonzero_logit_delta_is_not_byte_identical():
    """The likely outcome, and it must not be rounded up to 'identical'."""
    r = classify_identity(
        baseline_tokens=[1, 2, 3], treatment_tokens=[1, 2, 3], max_abs_logit_delta=1e-6
    )
    assert not r.byte_identical
    assert not r.changes_emitted_tokens


def test_changed_tokens_are_flagged_as_lossy():
    r = classify_identity(
        baseline_tokens=[1, 2, 3], treatment_tokens=[1, 2, 4], max_abs_logit_delta=0.2
    )
    assert not r.byte_identical
    assert r.changes_emitted_tokens


def test_the_gate_refuses_when_the_a_vs_a_floor_did_not_hold():
    """A harness that cannot reproduce itself cannot measure anything."""
    noisy_floor = classify_identity([1, 2, 3], [1, 2, 9], 0.5)
    real = classify_identity([1, 2, 3], [1, 2, 3], 0.0)
    v = gate_verdict(a_vs_a=noisy_floor, a_vs_b=real)
    assert not v.enable
    assert "a-vs-a" in v.reason.lower()


def test_the_gate_enables_only_on_a_clean_floor_and_identical_arms():
    clean = classify_identity([1, 2, 3], [1, 2, 3], 0.0)
    v = gate_verdict(a_vs_a=clean, a_vs_b=clean)
    assert v.enable
    assert not v.lossy


def test_a_divergent_arm_on_a_clean_floor_is_refused_as_lossy():
    clean = classify_identity([1, 2, 3], [1, 2, 3], 0.0)
    diverged = classify_identity([1, 2, 3], [1, 2, 3], 1e-6)
    v = gate_verdict(a_vs_a=clean, a_vs_b=diverged)
    assert not v.enable
    assert v.lossy
    assert "byte-identical" in v.reason.lower()


def test_an_unrun_measurement_is_refused_not_assumed():
    v = gate_verdict(a_vs_a=None, a_vs_b=None)
    assert not v.enable
    assert "not been run" in v.reason.lower()
