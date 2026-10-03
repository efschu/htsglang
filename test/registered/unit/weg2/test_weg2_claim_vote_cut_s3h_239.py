"""#239 S3h: the packed prefetch claim vote under the token cut.

The claim reduce of the prefetch thread is ONE MIN all_reduce over the packed
``[claim, -claim]`` (fnFL2x22 / #1233 L8): min and max of the group, and a
split is the named WEG2 DRAFT-DISAGREE STOP. Under Form A the expert workers
held no bytes and abstained. Under the token cut a KV-holding worker gets the
real file backend with its KV window (S4b part 2) and would VOTE: its claim
counts the KV pages it can read, the host's is capped at its deepest mamba
anchor -- the fnFL2x22 shape (4053 vs 4052) the other way round, a STOP on
every prefetch whose end anchor is not the last page. The worker therefore
votes in the MIN arm only, and a worker BELOW the host settles as the group
MIN (named), never as a disagreement.
"""

from __future__ import annotations

import logging

import pytest

from sglang.srt import rank_role
from sglang.srt.managers import cache_controller as cc


def _reduce_min(*packed):
    return [min(p[i] for p in packed) for i in range(2)]


def test_a_min_only_vote_cannot_set_the_max():
    assert cc.encode_claim_vote(4053, False, min_only=True) == [4053, cc.CLAIM_VOTE_ABSTAIN]
    # the default form is unchanged
    assert cc.encode_claim_vote(4052, False) == [4052, -4052]
    assert cc.encode_claim_vote(9, True, min_only=True) == [cc.CLAIM_VOTE_ABSTAIN] * 2


def test_worker_above_the_anchor_capped_host_is_no_split():
    host = cc.encode_claim_vote(4052, False)
    w1 = cc.encode_claim_vote(4053, False, min_only=True)
    w2 = cc.encode_claim_vote(4053, False, min_only=True)
    mn, mx = cc.decode_claim_vote(_reduce_min(host, w1, w2))
    assert (mn, mx) == (4052, 4052)
    # the same three votes WITHOUT the min-only arm: the STOP the fix removes
    mn0, mx0 = cc.decode_claim_vote(_reduce_min(host, [4053, -4053], [4053, -4053]))
    with pytest.raises(cc.Weg2DraftDisagree):
        cc.assert_draft_claims_agree(mn0, mx0, "weg2-1-1")


def test_worker_below_the_host_settles_as_the_group_min(caplog):
    host = cc.encode_claim_vote(4052, False)
    w1 = cc.encode_claim_vote(3968, False, min_only=True)
    w2 = cc.encode_claim_vote(4053, False, min_only=True)
    mn, mx = cc.decode_claim_vote(_reduce_min(host, w1, w2))
    assert (mn, mx) == (3968, 4052)
    with caplog.at_level(logging.WARNING):
        got = cc.settle_claim_split(mn, mx, "weg2-2-2", cut_active=True)
    assert got == 3968
    assert cc.CLAIM_CUT_MIN_ADOPT_MARKER in caplog.text


def test_without_the_cut_a_split_stays_the_stop():
    with pytest.raises(cc.Weg2DraftDisagree):
        cc.settle_claim_split(3968, 4052, "weg2-3-3", cut_active=False)
    assert cc.settle_claim_split(4052, 4052, "weg2-3-3", cut_active=False) == 4052


def test_only_min_voters_decode_to_their_min():
    assert cc.decode_claim_vote(_reduce_min([7, cc.CLAIM_VOTE_ABSTAIN],
                                            [5, cc.CLAIM_VOTE_ABSTAIN])) == (5, 5)


class _Ctl:
    def __init__(self, abstain=False):
        self.storage_backend = type("B", (), {"abstains_from_claim_vote": abstain})()


def test_min_only_is_the_kv_holding_worker(monkeypatch):
    monkeypatch.setattr(rank_role, "form_a_worker_holds_kv", lambda: True)
    assert cc.claim_vote_min_only(_Ctl()) is True
    # a byteless tier abstains in both arms, it is not a min voter
    assert cc.claim_vote_min_only(_Ctl(abstain=True)) is False
    monkeypatch.setattr(rank_role, "form_a_worker_holds_kv", lambda: False)
    assert cc.claim_vote_min_only(_Ctl()) is False


def test_the_prefetch_loop_encodes_with_the_min_only_arm():
    import inspect

    src = inspect.getsource(cc.HiCacheController.prefetch_thread_func)
    assert "min_only=claim_vote_min_only(self)" in src
    assert "settle_claim_split(" in src
