"""L15-12c-E1: aggregate the per-rank sample-check votes of the wake into
ONE group decision (plan L15-12-PART3-PLAN section 4).

Pure functions on the vote list (no process group, no I/O), mirroring
l15_fp_reduce's group-uniformity: every rank passes the same gathered list
and computes the identical GroupCheck and refusal message.
"""

import pickle

from sglang.srt.weg2 import l15_restore


def _vote(fp=7, ok=10, bad=0, missing=0, drops=()):
    return l15_restore.check_vote(fp, ok, bad, missing, drops)


def test_check_vote_is_plain_picklable_and_sorted():
    v = _vote(7, 10, 0, 2, ["b", "a", "a"])
    assert v == (7, 10, 0, 2, ("a", "b"))
    assert pickle.loads(pickle.dumps(v)) == v
    assert all(isinstance(x, int) for x in v[:4])


def test_bad_on_one_rank_refuses_every_rank_identically():
    votes = [_vote(7, 5, 0, 1), _vote(7, 4, 1, 0), None]
    gc = l15_restore.group_check(votes)
    assert gc.refuse is True
    assert gc.verdict == "refuse"
    msg_a = l15_restore.refusal_message(gc, 3)
    msg_b = l15_restore.refusal_message(l15_restore.group_check(list(votes)), 3)
    assert msg_a == msg_b
    assert msg_a == "L15-CHECK REFUSED epoch=3 bad_ranks=1"


def test_drop_union_over_three_ranks_order_independent():
    votes = [
        _vote(7, 3, 0, 0, ["z", "m"]),
        None,
        _vote(7, 3, 0, 0, ["m", "a"]),
        _vote(7, 3, 0, 0, []),
    ]
    gc = l15_restore.group_check(votes)
    assert gc.drop_rids == ("a", "m", "z")
    for perm in (list(reversed(votes)), votes[1:] + votes[:1]):
        assert l15_restore.group_check(perm) == gc


def test_mixed_fingerprints_fallback():
    gc = l15_restore.group_check([_vote(7), _vote(8), _vote(8)])
    assert gc.fp_mixed is True
    assert gc.refuse is False
    assert gc.verdict == "fallback"


def test_hold_on_some_ranks_none_on_others_fallback():
    gc = l15_restore.group_check([_vote(7, 5, 0, 0), None])
    assert gc.fp_mixed is True
    assert gc.verdict == "fallback"


def test_all_none_is_none():
    gc = l15_restore.group_check([None, None])
    assert gc.verdict == "none"
    assert gc.refuse is False
    assert gc.fp_mixed is False
    assert gc.drop_rids == ()


def test_all_equal_no_bad_is_hold():
    gc = l15_restore.group_check([_vote(7, 5, 0, 1), _vote(7, 4, 0, 0)])
    assert gc.verdict == "hold"
    assert gc.refuse is False
    assert gc.fp_mixed is False


def test_bad_beats_fp_mixed_in_verdict():
    gc = l15_restore.group_check([_vote(7, 1, 2, 0), _vote(8, 1, 0, 0)])
    assert gc.refuse is True
    assert gc.fp_mixed is True  # reported, but refuse dominates the verdict
    assert gc.verdict == "refuse"
    assert l15_restore.refusal_message(gc, 9) == (
        "L15-CHECK REFUSED epoch=9 bad_ranks=0"
    )


def test_shuffled_vote_order_gives_identical_group_check():
    votes = [
        _vote(7, 2, 0, 1, ["r2"]),
        _vote(7, 3, 0, 0, ["r1"]),
        _vote(7, 1, 0, 2, ["r0"]),
    ]
    base = l15_restore.group_check(votes)
    assert l15_restore.group_check(list(reversed(votes))) == base
    assert base.verdict == "hold"
    assert base.drop_rids == ("r0", "r1", "r2")
