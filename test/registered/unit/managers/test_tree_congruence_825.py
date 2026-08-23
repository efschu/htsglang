"""#825 -- the prefix trees must be provably congruent at the cutover.

WHY THIS EXISTS (measured, not hypothesised). Under the phase flip the
uniformity floors are scoped to ``tp_cpu_group``
(``scheduler.py:4684 _update_uniform_pool_budget``), and in the PP-prefill
phase that group has world 1 on every rank, so ``scheduler.py:4770`` switches
all three floors OFF and logs it. Three PP ranks then evict their radix trees
independently -- ``_publish_uniform_evict_floor``'s own docstring says the
consequence out loud: "so the radix trees stop being replicas". The cutover
does not reconcile them (``phase_flip_runtime.py`` ``_cutover`` never touches
``tree_cache``) while ``build_flip_live_slots_fn``'s docstring assumes "the
tree and the batch state are rank-replicated between rounds". The TP-decode
phase then REQUIRES that identity.

THE SPECIMEN these numbers come from: boot 0516, 2026-08-23, recorded in
COORD-strand16f-801-build.md B.9/B.10 -- two ranks entered the same iteration
with ``#new-seq 1 vs 3`` and ``#cached-token 0 vs 16384``. The 16384 is
literally ``len(req.prefix_indices)`` on one rank against 0 on another
(``schedule_policy.py:1653`` -> ``:1141`` -> ``metrics_reporter.py:117/947``).

These tests are written against that measured pair BEFORE the module exists.
"""

import pytest

from sglang.srt.managers import tree_congruence as tc


# --------------------------------------------------------------------------
# The digest is over KEYS, never over values.
# --------------------------------------------------------------------------


def test_digest_ignores_rank_local_kv_indices():
    """THE POINT OF THE WHOLE MODULE.

    A node's ``value`` is a tensor of KV slot indices allocated by THIS rank's
    allocator. Two ranks holding identical cached text hold it at different
    physical slots by construction. A digest that mixed those in would report
    disagreement on every healthy boot and would make the reconcile fire
    always -- the failure mode that looks like a working feature.
    """
    a = tc.fold_digest([tc.node_fingerprint((1, 2, 3)), tc.node_fingerprint((4, 5))])
    b = tc.fold_digest([tc.node_fingerprint((1, 2, 3)), tc.node_fingerprint((4, 5))])
    assert a == b
    # node_fingerprint takes token ids only -- it has no parameter for values.
    with pytest.raises(TypeError):
        tc.node_fingerprint((1, 2, 3), [900, 901, 902])


def test_digest_is_order_independent():
    """Ranks reach identical content by DIFFERENT insertion orders.

    Python dict iteration is insertion-ordered, so a traversal-order-dependent
    fold would disagree on trees that are in fact identical. The fold must be
    commutative.
    """
    fps = [tc.node_fingerprint(k) for k in [(1, 2), (3, 4, 5), (6,), (7, 8)]]
    forward = tc.fold_digest(fps)
    backward = tc.fold_digest(list(reversed(fps)))
    shuffled = tc.fold_digest([fps[2], fps[0], fps[3], fps[1]])
    assert forward == backward == shuffled


def test_distinct_content_gives_distinct_digest():
    assert tc.node_fingerprint((1, 2, 3)) != tc.node_fingerprint((1, 2, 4))
    assert tc.node_fingerprint((1, 2, 3)) != tc.node_fingerprint((1, 2, 3, 0))
    # A prefix must not fold to the same value as its extension.
    assert tc.fold_digest([tc.node_fingerprint((1, 2))]) != tc.fold_digest(
        [tc.node_fingerprint((1, 2, 3))]
    )


def test_empty_tree_has_a_defined_digest_and_all_empty_agrees():
    """All ranks empty is the COMMON case right after a reset. It must not
    read as disagreement, or the reconcile would fire forever."""
    e = tc.fold_digest([])
    assert e == tc.EMPTY_TREE_DIGEST
    assert tc.agreement(*tc.reduce_pair_result([tc.digest_pair(e), tc.digest_pair(e)]))


# --------------------------------------------------------------------------
# The (x, -x) MIN pair -- the codebase's own idiom.
# --------------------------------------------------------------------------


def test_digest_is_bounded_so_the_int64_pair_cannot_overflow():
    """The pair rides ONE int64 all_reduce. If a digest could reach 2**63 the
    negation would wrap and MIN would silently return a value that decodes as
    agreement. Bound it and pin the bound."""
    for k in [(), (0,), (2**31 - 1,), tuple(range(500)), (999999, 1, 7)]:
        d = tc.fold_digest([tc.node_fingerprint(k)])
        assert 0 <= d < tc.DIGEST_MODULUS
        assert tc.DIGEST_MODULUS <= 2**62
        lo, hi = tc.digest_pair(d)
        assert -(2**63) < lo < 2**63
        assert -(2**63) < hi < 2**63


def test_agreement_is_exactly_min_equals_negated_max():
    assert tc.agreement(7, -7) is True
    assert tc.agreement(7, -9) is False  # min 7, max 9
    assert tc.agreement(0, 0) is True


def test_the_0516_specimen_pair_is_reported_as_divergence():
    """The measured pair: one rank matched a 16384-token prefix, its peer
    matched nothing. Encoded as the trees that produce it -- a rank holding a
    16384-token cached prefix vs a rank holding an empty tree."""
    cached = tc.fold_digest([tc.node_fingerprint(tuple(range(16384)))])
    empty = tc.fold_digest([])
    assert cached != empty
    mn, mneg = tc.reduce_pair_result([tc.digest_pair(cached), tc.digest_pair(empty)])
    assert tc.agreement(mn, mneg) is False


def test_three_ranks_two_agreeing_one_not_is_still_divergence():
    """#new-seq 1 vs 3 came from THREE ranks, not two. A majority must not
    read as agreement -- the TP phase needs all three identical."""
    same = tc.fold_digest([tc.node_fingerprint((1, 2, 3))])
    other = tc.fold_digest([tc.node_fingerprint((1, 2, 4))])
    mn, mneg = tc.reduce_pair_result(
        [tc.digest_pair(same), tc.digest_pair(same), tc.digest_pair(other)]
    )
    assert tc.agreement(mn, mneg) is False


def test_all_three_ranks_identical_agrees():
    d = tc.fold_digest([tc.node_fingerprint((1, 2, 3)), tc.node_fingerprint((9,))])
    mn, mneg = tc.reduce_pair_result([tc.digest_pair(d)] * 3)
    assert tc.agreement(mn, mneg) is True


# --------------------------------------------------------------------------
# The verdict, and the reason it carries.
# --------------------------------------------------------------------------


def test_verdict_agreement_does_not_reconcile():
    d = tc.fold_digest([tc.node_fingerprint((1, 2, 3))])
    v = tc.congruence_verdict(local_digest=d, group_min=d, group_neg_min=-d)
    assert v.congruent is True
    assert v.must_reconcile is False
    assert v.reason == ""


def test_verdict_divergence_reconciles_and_names_the_numbers():
    a = tc.fold_digest([tc.node_fingerprint((1, 2, 3))])
    b = tc.fold_digest([tc.node_fingerprint((1, 2, 4))])
    mn, mneg = tc.reduce_pair_result([tc.digest_pair(a), tc.digest_pair(b)])
    v = tc.congruence_verdict(local_digest=a, group_min=mn, group_neg_min=mneg)
    assert v.congruent is False
    assert v.must_reconcile is True
    # The reason must carry the actual numbers, so the log line is evidence
    # rather than an assertion that something happened.
    assert str(min(a, b)) in v.reason and str(max(a, b)) in v.reason


def test_the_verdict_is_identical_on_every_rank():
    """DECIDED FROM THE COLLECTIVE, NOT FROM THE LOCAL VALUE.

    Every rank must take the SAME branch, or the reconcile itself splits the
    group -- which is the class of defect this whole module exists to close.
    A rank whose own digest happens to equal the group min must still
    reconcile if the group disagreed.
    """
    a = tc.fold_digest([tc.node_fingerprint((1, 2, 3))])
    b = tc.fold_digest([tc.node_fingerprint((1, 2, 4))])
    mn, mneg = tc.reduce_pair_result([tc.digest_pair(a), tc.digest_pair(b)])
    va = tc.congruence_verdict(local_digest=a, group_min=mn, group_neg_min=mneg)
    vb = tc.congruence_verdict(local_digest=b, group_min=mn, group_neg_min=mneg)
    assert va.must_reconcile == vb.must_reconcile is True
    assert va.congruent == vb.congruent is False


def test_absent_tree_contributes_uniform_width_and_reads_as_divergence():
    """A rank with no tree cache must still contribute to the reduce, or the
    payload width becomes a per-rank capability -- the exact hazard
    ``_update_uniform_pool_budget`` documents for its host and mamba pairs
    ("contributing it unconditionally is what makes that a property of the
    code rather than of the flagset")."""
    assert tc.ABSENT_TREE_DIGEST != tc.EMPTY_TREE_DIGEST
    pair = tc.digest_pair(tc.ABSENT_TREE_DIGEST)
    assert len(pair) == 2
    present = tc.fold_digest([tc.node_fingerprint((1, 2))])
    mn, mneg = tc.reduce_pair_result([tc.digest_pair(present), pair])
    assert tc.agreement(mn, mneg) is False


def test_pair_width_is_two_and_fixed():
    """The reduce payload width must not depend on anything per-rank."""
    for d in [0, 1, tc.EMPTY_TREE_DIGEST, tc.ABSENT_TREE_DIGEST, tc.DIGEST_MODULUS - 1]:
        assert len(tc.digest_pair(d)) == 2


# --------------------------------------------------------------------------
# The two arms that exist because a mutant survived without them.
# --------------------------------------------------------------------------


def test_fold_preserves_multiplicity():
    """KILLS THE XOR MUTANT.

    Every other property in this file (commutativity, distinctness,
    boundedness) holds for XOR as well as for modular addition, so the first
    version of this suite could not tell them apart. XOR cancels duplicates:
    a tree holding the same key twice would fold IDENTICALLY to one holding
    it zero times, and two ranks differing by exactly a duplicated node would
    read as congruent. Addition is what makes multiplicity visible.
    """
    fp = tc.node_fingerprint((1, 2, 3))
    once = tc.fold_digest([fp])
    twice = tc.fold_digest([fp, fp])
    thrice = tc.fold_digest([fp, fp, fp])
    assert once != twice != thrice
    assert once != thrice
    # And the cancelling case stated directly: two copies must not fold to
    # the empty tree.
    assert twice != tc.EMPTY_TREE_DIGEST


def test_digest_is_stable_across_processes_with_different_hash_seeds():
    """KILLS THE ``hash()`` MUTANT -- the one no in-process test can catch.

    The ranks are separate OS processes. Python salts the hash of str/bytes
    (and therefore of tuples containing them) per process via PYTHONHASHSEED,
    so a digest built on ``hash()`` is stable within one process and differs
    between processes. Every test above would pass; the feature would report
    divergence on every healthy multi-rank boot and reconcile forever.

    This arm therefore has to leave the process. Two children with different
    explicit seeds must produce the same digest.
    """
    import os
    import subprocess
    import sys

    prog = (
        "from sglang.srt.managers import tree_congruence as tc;"
        "print(tc.fold_digest([tc.node_fingerprint((1,2,3)),"
        "tc.node_fingerprint((4,5,6))]))"
    )
    outs = []
    for seed in ("0", "1", "12345"):
        env = dict(os.environ, PYTHONHASHSEED=seed, CUDA_VISIBLE_DEVICES="")
        r = subprocess.run(
            [sys.executable, "-c", prog], env=env, capture_output=True, text=True
        )
        assert r.returncode == 0, r.stderr[-2000:]
        outs.append(r.stdout.strip())
    assert len(set(outs)) == 1, f"digest is hash-seed dependent: {outs}"


def test_node_fingerprint_is_itself_bounded_not_only_the_fold():
    """KILLS THE 'unbounded fingerprint' MUTANT.

    ``fold_digest`` re-applies the modulus, so a ``node_fingerprint`` that
    returned an unbounded value was invisible to every boundedness assertion
    that went through the fold -- the mutant survived the first run of this
    suite. ``node_fingerprint`` is public and callable on its own, so its
    range is part of its contract and gets pinned here directly.
    """
    for k in [(), (0,), (2**31 - 1,), tuple(range(500)), (999999, 1, 7), (-1,)]:
        fp = tc.node_fingerprint(k)
        assert 0 <= fp < tc.DIGEST_MODULUS, (k, fp)


# --------------------------------------------------------------------------
# The adapter over a live tree.
# --------------------------------------------------------------------------


class _Key:
    def __init__(self, token_ids, extra_key=None):
        self.token_ids = token_ids
        self.extra_key = extra_key


class _Node:
    def __init__(self, token_ids=None, extra_key=None, children=None, value=None):
        self.key = _Key(token_ids, extra_key) if token_ids is not None else None
        self.children = children or {}
        self.value = value


def _tree(root):
    class T:
        root_node = root

    return T()


def test_adapter_folds_a_live_tree_and_ignores_values():
    """Two ranks with identical KEYS but different rank-local KV VALUES must
    read as congruent. This is constraint 1 exercised through the adapter, not
    just through the pure fold."""
    a = _tree(
        _Node(
            (),
            children={
                1: _Node((1, 2), value=[10, 11]),
                2: _Node((3,), value=[12]),
            },
        )
    )
    b = _tree(
        _Node(
            (),
            children={
                1: _Node((1, 2), value=[999, 998]),
                2: _Node((3,), value=[777]),
            },
        )
    )
    assert tc.tree_digest_of(a) == tc.tree_digest_of(b)


def test_adapter_detects_a_missing_node():
    """The 0516 shape through the adapter: one rank evicted a node its peer
    kept, so one holds a cached prefix the other does not."""
    kept = _tree(_Node((), children={1: _Node((1, 2)), 2: _Node((3,))}))
    evicted = _tree(_Node((), children={1: _Node((1, 2))}))
    assert tc.tree_digest_of(kept) != tc.tree_digest_of(evicted)


def test_adapter_distinguishes_extra_key():
    """Identical token ids under different cache salts are DIFFERENT entries."""
    x = _tree(_Node((), children={1: _Node((1, 2), extra_key="lora-a")}))
    y = _tree(_Node((), children={1: _Node((1, 2), extra_key="lora-b")}))
    assert tc.tree_digest_of(x) != tc.tree_digest_of(y)


def test_adapter_is_traversal_order_independent():
    """Same content, children inserted in the opposite order."""
    fwd = _tree(_Node((), children={1: _Node((1, 2)), 2: _Node((3,))}))
    rev = _tree(_Node((), children={2: _Node((3,)), 1: _Node((1, 2))}))
    assert tc.tree_digest_of(fwd) == tc.tree_digest_of(rev)


def test_adapter_never_raises_on_a_stub_without_a_tree():
    """This runs on the flip's consensus path. A raise there takes the
    instance down, so a missing tree must degrade to a sentinel exactly the
    way the census handle degrades to a no-op."""

    class NoTree:
        pass

    assert tc.tree_digest_of(NoTree()) == tc.ABSENT_TREE_DIGEST
    assert tc.tree_digest_of(None) == tc.ABSENT_TREE_DIGEST


def test_adapter_is_iterative_and_survives_a_deep_tree():
    """A recursive walk would RecursionError on a deep radix tree, on the
    seam, where a raise is fatal."""
    node = _Node((0,))
    for i in range(1, 5000):
        node = _Node((i,), children={0: node})
    d = tc.tree_digest_of(_tree(node))
    assert 0 <= d < tc.DIGEST_MODULUS
