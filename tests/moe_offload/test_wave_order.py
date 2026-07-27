# SPDX-License-Identifier: Apache-2.0
"""CPU tests for the expert-major prefill wave order (#254).

Gates:
  * PLANNING -- ``plan_expert_waves`` partitions exactly the routed SPILL
    experts, into groups of at most ``scratch``, deterministically, under both
    static and hot residency, with ``-1`` padding ignored.
  * ORDER INDEPENDENCE -- the fixed k-slot buffer makes the per-token summation
    order independent of the wave split: filling [T, top_k] contributions in
    token-major order, in expert-major order, or in a random order and reducing
    over the k axis at the end gives BIT-identical results to the unsplit sum.
    The naive alternative (accumulating per-wave partial sums) is shown to
    differ, i.e. the test would fail on the implementation this replaces.
  * DEFAULT -- the env flag defaults to "token" and only accepts the two
    documented values.

Run:  python -m pytest tests/moe_offload/test_wave_order.py -q
"""

import os
import random
import sys

import pytest
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "python"))

from sglang.srt.environ import envs  # noqa: E402
from sglang.srt.layers.moe.expert_offload import (  # noqa: E402
    plan_expert_waves,
    plan_token_waves,
    resolve_wave_order,
)


def _routing(T, K, E, seed, pad=0.0):
    rng = random.Random(seed)
    rows = []
    for _ in range(T):
        row = rng.sample(range(E), K)
        if pad and rng.random() < pad:
            row[-1] = -1
        rows.append(row)
    return rows


# --------------------------------------------------------------------------
# PLANNING
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "T,K,E,R,C", [(64, 8, 64, 16, 8), (7, 4, 32, 4, 3), (1, 2, 8, 8, 2)]
)
def test_expert_waves_partition_the_spill_set(T, K, E, R, C):
    ids = _routing(T, K, E, seed=T + K + E, pad=0.2)
    resident_used, spill_waves = plan_expert_waves(ids, R, C)

    routed = {e for row in ids for e in row if e >= 0}
    assert set(resident_used) == {e for e in routed if e < R}
    flat = [e for wave in spill_waves for e in wave]
    assert flat == sorted(e for e in routed if e >= R)  # each exactly once, sorted
    assert len(set(flat)) == len(flat)
    assert all(len(w) <= C for w in spill_waves)
    assert all(len(w) > 0 for w in spill_waves)


def test_expert_waves_hot_residency():
    E, R, C = 32, 8, 4
    hot = frozenset({3, 9, 11, 17, 20, 25, 28, 31})
    ids = _routing(48, 6, E, seed=5)
    resident_used, spill_waves = plan_expert_waves(ids, R, C, resident_ids=hot)
    routed = {e for row in ids for e in row if e >= 0}
    assert set(resident_used) == routed & hot
    assert {e for w in spill_waves for e in w} == routed - hot


def test_expert_waves_are_deterministic_and_never_overflow():
    """Unlike the token-major split, expert-major cannot fail: a token's top-k
    may straddle waves, so scratch=1 is still a valid configuration."""
    ids = _routing(32, 8, 40, seed=99)
    a = plan_expert_waves(ids, 4, 1)
    b = plan_expert_waves(ids, 4, 1)
    assert a == b
    assert all(len(w) == 1 for w in a[1])
    with pytest.raises(ValueError):
        plan_token_waves(ids, 4, 1)  # the token-major split does fail here


def test_expert_waves_ignore_padding_only_rows():
    assert plan_expert_waves([[-1, -1], [-1, -1]], 4, 2) == ([], [])


# --------------------------------------------------------------------------
# ORDER INDEPENDENCE  (the load-bearing claim)
# --------------------------------------------------------------------------


def _contributions(ids, K, dtype=torch.bfloat16, seed=0):
    """Per-(token, k-slot) contribution V[t,k] = f(token, expert). Deliberately
    ill-conditioned (mixed magnitudes) so any re-association shows up."""
    g = torch.Generator().manual_seed(seed)
    T = len(ids)
    H = 32
    base = torch.randn(T, K, H, generator=g)
    scale = torch.tensor(
        [[10.0 ** ((e % 7) - 3) if e >= 0 else 0.0 for e in row] for row in ids]
    )
    return (base * scale.unsqueeze(-1)).to(dtype)


def _reduce(partials):
    """The reduction the fused kernel applies over the k axis."""
    return torch.sum(partials, dim=1)


def _fill_token_major(V, ids, R, C):
    """Fill the k-slot buffer wave by wave over TOKEN subsets (today's split)."""
    out = torch.zeros_like(V)
    for rows in plan_token_waves(ids, R, C):
        for t in rows:
            out[t] = V[t]
    return out


def _fill_expert_major(V, ids, R, C, resident_ids=None):
    """Fill the same buffer wave by wave over SPILL-EXPERT groups."""
    out = torch.zeros_like(V)
    resident_used, spill_waves = plan_expert_waves(ids, R, C, resident_ids)
    for wave in [resident_used] + spill_waves:
        members = set(wave)
        for t, row in enumerate(ids):
            for k, e in enumerate(row):
                if e in members:
                    out[t, k] = V[t, k]
    return out


@pytest.mark.parametrize(
    "T,K,E,R,C,pad",
    [
        # C >= K, so the token-major split is legal and can be compared too.
        (64, 8, 64, 16, 8, 0.0),
        (64, 8, 64, 16, 8, 0.25),  # -1 padded slots
        (33, 6, 48, 12, 16, 0.0),
        (128, 4, 96, 8, 4, 0.1),
        (16, 2, 24, 6, 2, 0.0),
    ],
)
def test_fixed_kslot_buffer_is_split_invariant(T, K, E, R, C, pad):
    ids = _routing(T, K, E, seed=T * 31 + K, pad=pad)
    V = _contributions(ids, K, seed=T)
    ref = _reduce(V)

    tok = _reduce(_fill_token_major(V, ids, R, C))
    exp = _reduce(_fill_expert_major(V, ids, R, C))

    assert torch.equal(ref, tok)
    assert torch.equal(ref, exp)
    assert torch.equal(tok, exp)


@pytest.mark.parametrize(
    "T,K,E,R,C", [(64, 8, 64, 16, 4), (128, 4, 96, 8, 2), (16, 2, 24, 6, 1)]
)
def test_fixed_kslot_buffer_invariant_where_token_major_cannot_run(T, K, E, R, C):
    """C < top_k: the token-major split refuses (a token's spilled top-k does
    not fit the scratch), expert-major still reproduces the unsplit sum."""
    ids = _routing(T, K, E, seed=T * 7 + C)
    V = _contributions(ids, K, seed=T + C)
    with pytest.raises(ValueError):
        plan_token_waves(ids, R, C)
    assert torch.equal(_reduce(V), _reduce(_fill_expert_major(V, ids, R, C)))


def test_fixed_kslot_buffer_is_split_invariant_under_hot_residency():
    E, R, C = 64, 16, 4
    hot = frozenset(random.Random(3).sample(range(E), R))
    ids = _routing(96, 8, E, seed=11)
    V = _contributions(ids, 8, seed=11)
    ref = _reduce(V)
    exp = _reduce(_fill_expert_major(V, ids, R, C, resident_ids=hot))
    assert torch.equal(ref, exp)


def test_fixed_kslot_buffer_survives_duplicate_slots():
    """Two k-slots of one token holding the SAME expert id: they are distinct
    slots in the buffer, so the split still reproduces the reference exactly."""
    ids = [[5, 5, 9, 40], [7, 7, 7, 7], [12, 40, 12, 3]]
    V = _contributions(ids, 4, seed=2)
    ref = _reduce(V)
    assert torch.equal(ref, _reduce(_fill_expert_major(V, ids, 8, 2)))
    assert torch.equal(ref, _reduce(_fill_token_major(V, ids, 8, 2)))


def test_wave_order_is_irrelevant_to_the_buffer():
    """Shuffling the order in which waves are executed cannot change the sum."""
    ids = _routing(64, 8, 64, seed=17)
    V = _contributions(ids, 8, seed=17)
    resident_used, spill_waves = plan_expert_waves(ids, 16, 4)
    waves = [resident_used] + spill_waves
    random.Random(0).shuffle(waves)
    out = torch.zeros_like(V)
    for wave in waves:
        members = set(wave)
        for t, row in enumerate(ids):
            for k, e in enumerate(row):
                if e in members:
                    out[t, k] = V[t, k]
    assert torch.equal(_reduce(V), _reduce(out))


def test_per_wave_partial_sums_would_reassociate():
    """Falsifier for the naive expert-major implementation: accumulating each
    wave's already-reduced partial sum re-associates the top-k reduction, so it
    is NOT bit-identical. This is exactly what the k-slot buffer avoids."""
    ids = _routing(64, 8, 64, seed=23)
    V = _contributions(ids, 8, seed=23)
    ref = _reduce(V)
    resident_used, spill_waves = plan_expert_waves(ids, 16, 4)
    acc = torch.zeros_like(ref)
    for wave in [resident_used] + spill_waves:
        members = set(wave)
        masked = torch.zeros_like(V)
        for t, row in enumerate(ids):
            for k, e in enumerate(row):
                if e in members:
                    masked[t, k] = V[t, k]
        acc = acc + _reduce(masked)
    assert not torch.equal(ref, acc)


# --------------------------------------------------------------------------
# DEFAULT
# --------------------------------------------------------------------------


def test_wave_order_default_is_token():
    assert envs.SGLANG_MOE_OFFLOAD_WAVE_ORDER.get() == "token"


@pytest.mark.parametrize(
    "raw,expected",
    [(None, "token"), ("", "token"), (" Expert ", "expert"), ("TOKEN", "token")],
)
def test_wave_order_is_normalized(raw, expected):
    assert resolve_wave_order(raw) == expected


def test_unknown_wave_order_is_rejected():
    with pytest.raises(RuntimeError, match="wave_order|WAVE_ORDER"):
        resolve_wave_order("sideways")
