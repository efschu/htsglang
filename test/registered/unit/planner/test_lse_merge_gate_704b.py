"""#704b: the byte-identity gate for the decoupled attention path.

DESIGN_704 sec 4.4 originally demanded byte-identity "decoupled vs coupled".
That is A-vs-B, not A-vs-A: an LSE merge sums partial results in a different
floating-point order than monolithic attention, so bit-exact agreement is not a
property correct code has. As written the gate would fail forever on a correct
implementation, and the predictable outcome is that someone waives it -- which
is worse than having no gate.

Respecified as two gates, and this file builds the harness for both:

  GATE 1 (never waivable): DETERMINISM, byte-identical A-vs-A. The same inputs
  merged twice -- across runs, across boots -- must give bit-identical output.
  This is what catches a genuine non-determinism such as an arrival-ordered
  merge, which is the most likely silent defect in a distributed merge.

  GATE 2: AGREEMENT with the coupled reference within a tolerance fixed BEFORE
  the run, plus a greedy-decode token-sequence match.

Inputs are sampled on CPU and moved to device: `torch.randn` on-GPU is not
architecture-identical across the 3080s and the 5090, and this rig has been
bitten by that before. A harness that seeds on-GPU would make gate 1 fail for a
reason that has nothing to do with the merge.

The reference merge here is the CONTRACT the GPU path must satisfy. It is
deliberately written to match the semantics of
`layers/dcp/comm.py:228-262 cp_lse_ag_out_ar_mha_uneven`, which all-gathers
every rank's LSE and reduces with a single logsumexp over the stacked axis --
so merge order is RANK order, fixed by the collective, never arrival order.

Hermetic: CPU tensors only, no CUDA, no server.
"""

import pytest

torch = pytest.importorskip("torch")

from sglang.srt.planner.lse_merge_gate import (
    LseMergeGateError,
    agreement_report,
    assert_deterministic,
    cpu_sampled_inputs,
    merge_partials,
)

HEADS, HEAD_DIM, TOKENS = 4, 16, 32


def _partials(n_ranks, seed=0):
    return cpu_sampled_inputs(
        n_ranks=n_ranks, tokens=TOKENS, heads=HEADS, head_dim=HEAD_DIM, seed=seed
    )


def test_the_merge_is_bit_identical_across_repeated_runs():
    """GATE 1, the one that is never waived."""
    parts = _partials(3)
    a = merge_partials(parts)
    b = merge_partials(parts)
    assert torch.equal(a, b)


def test_inputs_are_sampled_on_cpu_and_reproducible_from_a_seed():
    """The cuda-randn cross-arch rule, enforced by the harness itself."""
    x = _partials(3, seed=7)
    y = _partials(3, seed=7)
    for (oa, la), (ob, lb) in zip(x, y):
        assert oa.device.type == "cpu" and la.device.type == "cpu"
        assert torch.equal(oa, ob) and torch.equal(la, lb)
    z = _partials(3, seed=8)
    assert not torch.equal(x[0][0], z[0][0])


def test_merge_order_is_rank_order_not_arrival_order():
    """THE silent defect this gate exists to catch.

    A merge that folded partials in arrival order would give a different
    floating-point result per run. Feeding the same partials in a shuffled
    order must NOT change the answer, because the contract fixes the order by
    rank -- the all_gather places rank i at stack index i.
    """
    parts = _partials(3)
    straight = merge_partials(parts)
    shuffled = merge_partials(list(reversed(parts)), rank_order=[2, 1, 0])
    assert torch.equal(straight, shuffled)


def test_a_shuffled_merge_without_the_rank_map_is_caught_not_tolerated():
    """If the caller reorders partials and does NOT say so, that is the bug.

    The harness must not silently produce a plausible answer for it -- the
    whole hazard is that the wrong answer looks fine.
    """
    parts = _partials(3)
    straight = merge_partials(parts)
    wrong = merge_partials(list(reversed(parts)))
    assert not torch.equal(straight, wrong)


def test_assert_deterministic_can_fail():
    """CAN-FAIL PROOF: a deliberately non-deterministic merge must be caught."""
    parts = _partials(3)
    assert_deterministic(lambda: merge_partials(parts), runs=4)

    calls = {"n": 0}

    def _flaky():
        calls["n"] += 1
        out = merge_partials(parts)
        if calls["n"] == 3:
            out = out.clone()
            out[0, 0, 0] += 1e-6  # a single perturbed element
        return out

    with pytest.raises(LseMergeGateError, match="not deterministic"):
        assert_deterministic(_flaky, runs=4)


def test_merging_one_partial_returns_it_unchanged():
    """The degenerate all-local case must be exactly a no-op."""
    parts = _partials(1)
    out = merge_partials(parts)
    assert torch.equal(out, parts[0][0])


def test_merge_matches_a_single_monolithic_softmax_within_tolerance():
    """GATE 2 in miniature, on data where the true answer is computable.

    Splitting one attention problem into per-rank partials and merging must
    reproduce the monolithic result -- to floating-point tolerance, NOT bit
    exactly, which is precisely why gate 2 is a tolerance gate.
    """
    torch.manual_seed(3)
    q = torch.randn(TOKENS, HEADS, HEAD_DIM, dtype=torch.float64)
    k = torch.randn(3, TOKENS, HEADS, HEAD_DIM, dtype=torch.float64)
    v = torch.randn(3, TOKENS, HEADS, HEAD_DIM, dtype=torch.float64)

    # Monolithic: attend over all 3*TOKENS keys at once.
    k_all = k.reshape(-1, HEADS, HEAD_DIM)
    v_all = v.reshape(-1, HEADS, HEAD_DIM)
    scores = torch.einsum("thd,shd->hts", q, k_all)
    ref = torch.einsum("hts,shd->thd", torch.softmax(scores, dim=-1), v_all)

    # Sharded: one partial per rank, then LSE merge.
    parts = []
    for r in range(3):
        s = torch.einsum("thd,shd->hts", q, k[r])
        lse = torch.logsumexp(s, dim=-1)  # (heads, tokens)
        out = torch.einsum("hts,shd->thd", torch.softmax(s, dim=-1), v[r])
        parts.append((out, lse.transpose(0, 1).contiguous()))
    got = merge_partials(parts)
    assert torch.allclose(got, ref, atol=1e-10)


def test_agreement_report_gates_on_a_tolerance_fixed_in_advance():
    parts = _partials(3)
    a = merge_partials(parts)
    b = a.clone()
    b[0, 0, 0] += 1e-3
    rep = agreement_report(a, b, max_abs=1e-2, max_rel=1e-2)
    assert rep.passed
    tight = agreement_report(a, b, max_abs=1e-9, max_rel=1e-9)
    assert not tight.passed
    assert tight.max_abs_seen > 0


def test_shape_mismatches_are_refused_rather_than_broadcast():
    parts = _partials(3)
    bad = list(parts)
    bad[1] = (bad[1][0][:, :, :-1], bad[1][1])
    with pytest.raises(LseMergeGateError, match="shape"):
        merge_partials(bad)
