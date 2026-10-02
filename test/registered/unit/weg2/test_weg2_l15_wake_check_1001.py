# SPDX-License-Identifier: Apache-2.0
"""L15-12c-E1W: the E1 gather -- pure half (l15_wake_check).

check_vote/group_check/refusal_message already exist in l15_restore (E1,
pure). This module adds the COLLECTIVE half, testable without a process
group: gather_votes (one all_gather_object over the passed group), decide
(gather + group_check + the named F11 refusal), and sample_rows_equal
(the byte-comparison primitive of the sample check).

Hermetic: torch.distributed.all_gather_object is monkeypatched with a fake
that fills the gathered list from a fixed per-rank vote table -- the same
table every "rank" sees, exactly like a real gather.
"""

from __future__ import annotations

import torch

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5)

from sglang.srt.weg2 import l15_restore  # noqa: E402
from sglang.srt.weg2 import l15_wake_check  # noqa: E402


def _vote(fp, ok=1, bad=0, missing=0, drop_rids=()):
    return l15_restore.check_vote(fp, ok, bad, missing, drop_rids)


def _fake_gather(monkeypatch, votes_per_rank, calls):
    """Fills the gathered list from the fixed table (in place, as dist does)."""
    def _fake(gathered, obj, group=None, **kw):
        calls.append((group, kw))
        gathered[:] = list(votes_per_rank)
    monkeypatch.setattr(
        torch.distributed, "all_gather_object", _fake
    )


class _FakeGroup:
    """Stand-in for the tp cpu process group (identity matters only)."""

    name = "tp_cpu_D"


def test_gather_votes_world_one_no_collective(monkeypatch):
    calls = []
    _fake_gather(monkeypatch, [99], calls)  # 99: must NOT be used
    assert l15_wake_check.gather_votes(7, None, 1) == [7]
    assert l15_wake_check.gather_votes(7, _FakeGroup(), 1) == [7]
    assert calls == [], "world <= 1 must not call the collective"


def test_gather_votes_world_three_uses_group(monkeypatch):
    calls = []
    votes = [_vote(7), None, _vote(7, ok=2)]
    _fake_gather(monkeypatch, votes, calls)
    grp = _FakeGroup()
    got = l15_wake_check.gather_votes(votes[0], grp, 3)
    assert got == votes
    assert len(calls) == 1
    assert calls[0][0] is grp, "the group must be passed through"


def test_decide_bad_on_one_rank_refuses_with_identical_text(monkeypatch):
    calls = []
    votes = [_vote(7), _vote(7, bad=1), _vote(7, ok=2, drop_rids=("r3",))]
    _fake_gather(monkeypatch, votes, calls)
    msgs = set()
    for own in votes:
        try:
            l15_wake_check.decide(own, _FakeGroup(), 3, epoch=77)
            raise AssertionError("refusal not raised")
        except l15_wake_check.L15CheckRefused as exc:
            msgs.add(str(exc))
    # The text is a pure function of the gathered list: every rank equal.
    assert len(msgs) == 1
    gc = l15_restore.group_check(votes)
    assert gc.refuse and gc.verdict == "refuse"
    assert msgs == {l15_restore.refusal_message(gc, 77)}
    assert "L15-CHECK REFUSED epoch=77" in msgs.pop()


def test_decide_hold_and_fallback_no_raise(monkeypatch):
    calls = []
    votes = [_vote(7), _vote(7, ok=2), _vote(7)]
    _fake_gather(monkeypatch, votes, calls)
    gc = l15_wake_check.decide(votes[0], _FakeGroup(), 3, epoch=77)
    assert gc.verdict == "hold" and not gc.refuse
    votes = [_vote(7), None, _vote(7)]
    _fake_gather(monkeypatch, votes, calls)
    gc = l15_wake_check.decide(None, _FakeGroup(), 3, epoch=77)
    assert gc.verdict == "fallback" and not gc.refuse


def test_sample_rows_equal_counts_ok_bad():
    a, b = torch.tensor([1, 2, 3]), torch.tensor([1, 2, 3])
    c = torch.tensor([1, 2, 4])
    # a == b; pairs: (a,b) equal, (c,b) unequal, (a,c) unequal -> 1 ok, 2 bad.
    ok, bad = l15_wake_check.sample_rows_equal([a, c, a], [b, b, c])
    assert (ok, bad) == (1, 2)
    # Unequal lengths: the unpaired rows count as bad, never as ok.
    ok, bad = l15_wake_check.sample_rows_equal([a], [b, b])
    assert (ok, bad) == (1, 1)
    ok, bad = l15_wake_check.sample_rows_equal([], [])
    assert (ok, bad) == (0, 0)
