# SPDX-License-Identifier: Apache-2.0
"""CPU tests for the class-level assertion bundles (check_class dispatch).

The headline property (the reason #124 exists): a single fabricated
real-corruption flip -- large-delta argmax flip at token 1 against a
confident reference -- FAILS ALL THREE classes, while each legitimate
synthetic pattern passes exactly its own class. This is the logic that
would have flagged the FP8 "byte-identical" overclaim (the near-tie-fork
pair passes SELF_DET_NEAR_TIE but fails MACHINE_ZERO) and that catches
silent regressions (the corruption pair fails everything).
"""

import pytest
import torch

from determinism_harness import (
    CLASS_SPECS,
    ByteIdentityClass,
    FlipKind,
    check_class,
)
from test_primitives import (
    BAND,
    NEAR_TIE,
    confident_logits,
    make_corruption_pair,
    make_near_tie_pair,
    traj,
    with_noise,
)

ALL_CLASSES = list(ByteIdentityClass)


def _check(cls, ref, test, rerun=None):
    band = None if cls is ByteIdentityClass.MACHINE_ZERO else BAND
    ntm = None if cls is ByteIdentityClass.MACHINE_ZERO else NEAR_TIE
    return check_class(cls, ref=ref, test=test, rerun=rerun, band=band, near_tie_margin=ntm)


def test_identical_pair_passes_machine_zero_and_everything_else():
    ref = traj(confident_logits(), "ref")
    same = traj(ref.logits.clone(), "same")
    for cls in ALL_CLASSES:
        v = _check(cls, ref, same, rerun=traj(same.logits.clone()))
        assert v.ok, f"{cls}: {v.summary()}"


def test_tiny_delta_pair_is_decode_class_not_machine_zero():
    ref = traj(confident_logits(), "ref")
    test = traj(with_noise(ref.logits, 1e-4), "test")
    assert not _check(ByteIdentityClass.MACHINE_ZERO, ref, test).ok
    v = _check(ByteIdentityClass.DECODE_CLASS, ref, test)
    assert v.ok, v.summary()
    # and with a bit-identical rerun it also satisfies the offload class
    v = _check(ByteIdentityClass.SELF_DET_NEAR_TIE, ref, test, rerun=traj(test.logits.clone()))
    assert v.ok, v.summary()


def test_near_tie_fork_passes_self_det_near_tie_only():
    """The FP8-rerun pattern: self-deterministic, forks at a genuine
    near-tie, cascades afterwards. Correct class: SELF_DET_NEAR_TIE.
    Wrong claims (MACHINE_ZERO -- the retracted overclaim -- and
    DECODE_CLASS) must fail on it."""
    ref, test, fork = make_near_tie_pair()
    rerun = traj(test.logits.clone(), "rerun")
    v = _check(ByteIdentityClass.SELF_DET_NEAR_TIE, ref, test, rerun=rerun)
    assert v.ok, v.summary()
    assert not _check(ByteIdentityClass.MACHINE_ZERO, ref, test).ok
    v_dec = _check(ByteIdentityClass.DECODE_CLASS, ref, test)
    assert not v_dec.ok
    assert v_dec.flip_kind is FlipKind.NEAR_TIE  # diagnosable as benign-kind flip


def test_real_corruption_fails_every_class():
    """THE key regression property: token-1 large-delta argmax flip against
    a confident reference is caught by all three classes -- even with a
    perfectly self-deterministic rerun."""
    ref, test = make_corruption_pair()
    rerun = traj(test.logits.clone(), "rerun")
    for cls in ALL_CLASSES:
        v = _check(cls, ref, test, rerun=rerun)
        assert not v.ok, f"{cls} failed to catch corruption: {v.summary()}"
    v = _check(ByteIdentityClass.SELF_DET_NEAR_TIE, ref, test, rerun=rerun)
    assert v.flip_kind is FlipKind.CORRUPTION


def test_self_det_violation_fails_offload_class_even_when_fork_is_legit():
    ref, test, _ = make_near_tie_pair()
    drifted = test.logits.clone()
    drifted[3, 4] = torch.nextafter(drifted[3, 4], torch.tensor(torch.inf))
    v = _check(ByteIdentityClass.SELF_DET_NEAR_TIE, ref, test, rerun=traj(drifted))
    assert not v.ok
    assert any(s.check == "self_determinism" and not s.ok for s in v.sub_verdicts)


def test_missing_rerun_is_a_failing_verdict_not_a_pass():
    ref = traj(confident_logits())
    test = traj(with_noise(ref.logits, 1e-4))
    v = _check(ByteIdentityClass.SELF_DET_NEAR_TIE, ref, test, rerun=None)
    assert not v.ok and "missing rerun" in v.summary()


def test_missing_band_is_a_configuration_error():
    ref = traj(confident_logits())
    with pytest.raises(ValueError):
        check_class(ByteIdentityClass.DECODE_CLASS, ref=ref, test=ref, band=None, near_tie_margin=1e-2)
    with pytest.raises(ValueError):
        check_class(
            ByteIdentityClass.SELF_DET_NEAR_TIE,
            ref=ref, test=ref, rerun=ref, band=1e-2, near_tie_margin=None,
        )


def test_under_band_drift_fails_decode_class():
    ref = traj(confident_logits())
    test_l = ref.logits.clone()
    for t in range(test_l.shape[0]):
        test_l[t] += min(1e-6 * (1.7 ** t), 0.8 * BAND)
    v = _check(ByteIdentityClass.DECODE_CLASS, ref, traj(test_l))
    assert not v.ok
    assert any(s.check == "non_compounding" and not s.ok for s in v.sub_verdicts)


def test_class_specs_registry_is_complete_and_self_describing():
    assert set(CLASS_SPECS) == set(ByteIdentityClass)
    for cls, spec in CLASS_SPECS.items():
        assert spec.cls is cls
        assert spec.summary and spec.assertions and spec.provenance
    # the offload class must carry the retraction provenance, permanently
    sd = CLASS_SPECS[ByteIdentityClass.SELF_DET_NEAR_TIE]
    assert "0fb3d8007" in sd.provenance
    assert "rerun" in sd.required_inputs
