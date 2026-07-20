# SPDX-License-Identifier: Apache-2.0
"""CPU unit tests for the #124 comparison primitives on SYNTHETIC trajectories.

No GPU, no server, no model: fabricated logits with controlled margins and
deltas prove that each primitive passes its legitimate pattern and -- the
key property -- that a real-corruption argmax flip is caught by every class.

Run:  python -m pytest tests/determinism/ -q
"""

import pytest
import torch

from determinism_harness import (
    FlipKind,
    PINNED_SEED,
    Trajectory,
    check_argmax_clean_trajectory,
    check_delta_band,
    check_machine_zero,
    check_near_tie_only_divergence,
    check_non_compounding,
    check_self_determinism,
    classify_flip,
    per_step_max_abs_delta,
)

T, V = 24, 32
BAND = 1e-2
NEAR_TIE = 1e-2


def confident_logits(T=T, V=V, margin=4.0, seed=0):
    """[T, V] float32 logits with a confident winner per step (top-2 margin
    >= ~margin) on a small deterministic noise floor."""
    g = torch.Generator().manual_seed(seed)
    x = torch.rand((T, V), generator=g) * 0.5
    winners = [(7 * t + 3) % V for t in range(T)]
    for t, w in enumerate(winners):
        x[t, w] += margin
    return x


def traj(logits, label=""):
    return Trajectory.from_greedy(logits, seed=PINNED_SEED, label=label)


def with_noise(logits, amp, seed=1):
    g = torch.Generator().manual_seed(seed)
    return logits + (torch.rand(logits.shape, generator=g) * 2 - 1) * amp


# ---------------------------------------------------------------- fixtures


@pytest.fixture
def ref():
    return traj(confident_logits(), "ref")


@pytest.fixture
def identical(ref):
    return traj(ref.logits.clone(), "identical")


@pytest.fixture
def tiny_delta(ref):
    """Argmax-clean, sub-band noise: the legit DECODE_CLASS pattern."""
    t = traj(with_noise(ref.logits, 1e-4), "tiny_delta")
    assert t.token_ids == ref.token_ids  # construction sanity
    return t


def make_near_tie_pair():
    """Reference with a genuine near-tie at step FORK; test flips there and
    legitimately forks afterwards (the FP8-rerun pattern)."""
    fork = 10
    ref_l = confident_logits()
    runner_up = (int(ref_l[fork].argmax()) + 1) % V
    ref_l[fork, runner_up] = ref_l[fork].max() - 2e-4  # margin 2e-4 << NEAR_TIE
    test_l = with_noise(ref_l, 1e-4)
    test_l[fork, runner_up] = test_l[fork].max() + 2e-4  # coin lands the other way
    # post-fork: different token history -> arbitrarily different logits
    test_l[fork + 1 :] = confident_logits(T - fork - 1, V, seed=99)
    a, b = traj(ref_l, "ref_neartie"), traj(test_l, "test_neartie")
    assert a.token_ids[:fork] == b.token_ids[:fork]
    assert a.token_ids[fork] != b.token_ids[fork]
    return a, b, fork


def make_corruption_pair():
    """Large-delta argmax flip at step 1 against a CONFIDENT reference --
    real corruption that every class must catch."""
    ref_l = confident_logits()
    test_l = with_noise(ref_l, 1e-4)
    bad = (int(ref_l[1].argmax()) + 5) % V
    test_l[1, bad] = ref_l[1].max() + 2.0  # decisively wins with a huge delta
    return traj(ref_l, "ref_corrupt"), traj(test_l, "test_corrupt")


# ------------------------------------------------------------ machine zero


def test_machine_zero_pass_on_identical(ref, identical):
    v = check_machine_zero(ref, identical)
    assert v.ok and v.max_abs_delta == 0.0


def test_machine_zero_fails_on_tiny_delta(ref, tiny_delta):
    v = check_machine_zero(ref, tiny_delta)
    assert not v.ok
    assert v.first_divergence_step is not None
    assert v.max_abs_delta > 0


def test_machine_zero_is_dtype_strict(ref):
    other = Trajectory(
        token_ids=list(ref.token_ids),
        logits=ref.logits.to(torch.float64),
        seed=PINNED_SEED,
    )
    assert not check_machine_zero(ref, other).ok


def test_machine_zero_fails_on_length_mismatch(ref):
    short = traj(ref.logits[:-1].clone(), "short")
    v = check_machine_zero(ref, short)
    assert not v.ok and "length mismatch" in v.message


# ------------------------------------------------------------- argmax clean


def test_argmax_clean_passes_tiny_delta(ref, tiny_delta):
    v = check_argmax_clean_trajectory(ref, tiny_delta, NEAR_TIE)
    assert v.ok


def test_argmax_clean_fails_near_tie_flip_labelled_near_tie():
    a, b, fork = make_near_tie_pair()
    v = check_argmax_clean_trajectory(a, b, NEAR_TIE)
    assert not v.ok
    assert v.first_divergence_step == fork
    assert v.flip_kind is FlipKind.NEAR_TIE
    assert v.margin_at_divergence <= NEAR_TIE


def test_argmax_clean_fails_corruption_labelled_corruption():
    a, b = make_corruption_pair()
    v = check_argmax_clean_trajectory(a, b, NEAR_TIE)
    assert not v.ok
    assert v.first_divergence_step == 1
    assert v.flip_kind is FlipKind.CORRUPTION
    assert v.margin_at_divergence > NEAR_TIE


def test_classify_flip_margin_both_sides_of_boundary(ref):
    step = 0
    row = ref.logits[step]
    tok = (int(row.argmax()) + 1) % V
    # comfortably inside the near-tie margin -> NEAR_TIE
    row[tok] = row.max() - 0.5 * NEAR_TIE
    kind, margin = classify_flip(ref, step, tok, NEAR_TIE)
    assert kind is FlipKind.NEAR_TIE and margin <= NEAR_TIE
    # comfortably outside -> CORRUPTION
    row[tok] = row.max() - 2.0 * NEAR_TIE
    kind, margin = classify_flip(ref, step, tok, NEAR_TIE)
    assert kind is FlipKind.CORRUPTION and margin > NEAR_TIE


# ---------------------------------------------------------------- delta band


def test_delta_band_pass_and_fail(ref):
    inside = traj(with_noise(ref.logits, 1e-4), "inside")
    outside = traj(with_noise(ref.logits, 5 * BAND, seed=7), "outside")
    assert check_delta_band(ref, inside, BAND).ok
    v = check_delta_band(ref, outside, BAND)
    assert not v.ok and v.max_abs_delta > BAND and v.first_divergence_step is not None


def test_delta_band_upto_ignores_post_fork(ref):
    test_l = ref.logits.clone()
    test_l[20:] += 10.0  # wild post-fork region
    v = check_delta_band(ref, traj(test_l), BAND, upto=20)
    assert v.ok


def test_per_step_delta_shape(ref, tiny_delta):
    d = per_step_max_abs_delta(ref, tiny_delta)
    assert d.shape == (T,) and d.dtype == torch.float64


# ------------------------------------------------------------ non-compounding


def test_non_compounding_passes_stationary_noise(ref, tiny_delta):
    assert check_non_compounding(ref, tiny_delta, BAND).ok


def test_non_compounding_catches_under_band_drift(ref):
    """Geometric drift that never leaves the band: the plain band check
    passes, only the drift detector sees the corruption signature."""
    test_l = ref.logits.clone()
    for t in range(T):
        amp = min(1e-6 * (1.7 ** t), 0.8 * BAND)
        test_l[t] += amp  # uniform shift: argmax unchanged, delta = amp
    drifty = traj(test_l, "drifty")
    assert check_delta_band(ref, drifty, BAND).ok  # invisible to the band
    v = check_non_compounding(ref, drifty, BAND)
    assert not v.ok and "compounding" in v.message


def test_non_compounding_trivial_on_short_trajectories(ref):
    short_ref = traj(ref.logits[:4].clone())
    short_test = traj(with_noise(ref.logits[:4], 1e-3))
    assert check_non_compounding(short_ref, short_test, BAND).ok


# ---------------------------------------------------------- self-determinism


def test_self_determinism_pass(ref, identical):
    assert check_self_determinism(ref, identical).ok


def test_self_determinism_fails_on_one_ulp(ref):
    other_l = ref.logits.clone()
    other_l[5, 0] = torch.nextafter(other_l[5, 0], torch.tensor(torch.inf))
    v = check_self_determinism(ref, traj(other_l))
    assert not v.ok and v.check == "self_determinism"


# ------------------------------------------------- near-tie-only divergence


def test_near_tie_only_passes_no_divergence(ref, tiny_delta):
    v = check_near_tie_only_divergence(ref, tiny_delta, BAND, NEAR_TIE)
    assert v.ok and v.first_divergence_step is None


def test_near_tie_only_passes_legit_fork():
    a, b, fork = make_near_tie_pair()
    v = check_near_tie_only_divergence(a, b, BAND, NEAR_TIE)
    assert v.ok
    assert v.first_divergence_step == fork
    assert v.flip_kind is FlipKind.NEAR_TIE
    # post-fork wildly-different logits did NOT fail the check (cascade legit)


def test_near_tie_only_fails_corruption():
    a, b = make_corruption_pair()
    v = check_near_tie_only_divergence(a, b, BAND, NEAR_TIE)
    assert not v.ok
    assert v.flip_kind is FlipKind.CORRUPTION
    assert v.first_divergence_step == 1


def test_near_tie_only_fails_over_band_before_fork():
    """Even with a near-tie flip, over-band deltas BEFORE the fork are
    corruption, not fp reassociation."""
    a, b, fork = make_near_tie_pair()
    bad_l = b.logits.clone()
    bad_l[2] += 50 * BAND  # uniform shift keeps argmax, blows the band pre-fork
    v = check_near_tie_only_divergence(a, traj(bad_l), BAND, NEAR_TIE)
    assert not v.ok and v.flip_kind is FlipKind.CORRUPTION
