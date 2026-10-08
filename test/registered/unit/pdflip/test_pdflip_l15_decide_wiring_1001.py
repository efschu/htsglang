# SPDX-License-Identifier: Apache-2.0
"""L15-12c-E1X: wire l15_wake_check.decide() into the wake (plan sec 9).

The E1 gather becomes the ONE decision source for the L1.5 keep-or-
fallback-drop: behind the master switch every rank builds its check_vote
(the fingerprint tuple, or None for a no-hold rank -- xsn410: no rank may
skip the collective) and runs ONE all_gather_object + group_check over the
tp cpu group the fence already uses (scheduler.world_group.cpu_group); the
group-uniform GroupCheck.verdict replaces today's per-rank fingerprint
verdict and _l15_wake_act consumes it unchanged. Master off -> no
collective at all (byte-identical wake).

No sample check yet (that is the next AP): bad is always 0, so an
L15CheckRefused cannot fire here. A FAILING collective, though, must
propagate -- it is never swallowed per rank.

Hermetic: torch.distributed.all_gather_object / get_world_size are
monkeypatched; the unbound production method runs on a FakeSelf with a
fake scheduler (world_group.cpu_group), the style of
test_pdflip_l15_fallback_1001.py.
"""

import pathlib
import sys
from types import SimpleNamespace

import torch

REPO = pathlib.Path(__file__).resolve().parents[4]
sys.path.insert(0, str(REPO / "python"))

from flliper.srt.managers.scheduler_components import weight_updater as wu  # noqa: E402
from flliper.srt.pdflip import l15_restore  # noqa: E402

WU = wu.SchedulerWeightUpdaterManager


class _FakeGroup:
    """Identity stand-in for the tp cpu process group."""


class FakeSelf:
    """The unbound-method fake-self style: the production method runs
    verbatim against the instance attributes."""

    _l15_decide_wake_verdict = WU._l15_decide_wake_verdict

    def __init__(self, group):
        self.scheduler = SimpleNamespace(
            world_group=SimpleNamespace(cpu_group=group)
        )


def _fake_gather(monkeypatch, votes_per_rank, calls):
    """Fills the gathered list from the fixed per-rank table (in place, as
    dist does) and pins the group's world size to the table length."""

    def _fake(gathered, obj, group=None, **kw):
        calls.append(group)
        gathered[:] = list(votes_per_rank)

    monkeypatch.setattr(torch.distributed, "all_gather_object", _fake)
    monkeypatch.setattr(
        torch.distributed, "get_world_size",
        lambda group=None: len(votes_per_rank),
    )


def _cv(fp):
    return l15_restore.check_vote(fp, 0, 0, 0, ())


def test_mixed_votes_every_rank_fallback(monkeypatch):
    # rank 0 held nothing, ranks 1-2 hold the same fingerprint -> fp_mixed.
    votes = [None, _cv(7), _cv(7)]
    calls = []
    _fake_gather(monkeypatch, votes, calls)
    grp = _FakeGroup()
    for own in (None, 7, 7):
        got = FakeSelf(grp)._l15_decide_wake_verdict(True, own, epoch=9)
        assert got == "fallback", f"rank with own fp={own} got {got!r}"
    # every rank called, each on the SAME group the fence uses.
    assert calls == [grp, grp, grp]


def test_all_holders_hold(monkeypatch):
    votes = [_cv(7), _cv(7), _cv(7)]
    calls = []
    _fake_gather(monkeypatch, votes, calls)
    grp = _FakeGroup()
    for _ in range(3):
        assert FakeSelf(grp)._l15_decide_wake_verdict(True, 7, epoch=9) == "hold"
    assert len(calls) == 3


def test_all_none_none(monkeypatch):
    votes = [None, None, None]
    calls = []
    _fake_gather(monkeypatch, votes, calls)
    grp = _FakeGroup()
    for _ in range(3):
        assert FakeSelf(grp)._l15_decide_wake_verdict(True, None, epoch=9) == "none"
    # the no-hold ranks still ran the collective (xsn410: none may skip).
    assert len(calls) == 3


def test_master_off_no_collective(monkeypatch):
    calls = []
    _fake_gather(monkeypatch, [None], calls)
    got = FakeSelf(_FakeGroup())._l15_decide_wake_verdict(False, 7, epoch=9)
    assert got is None
    assert calls == [], "master off must not touch the collective"


def test_raising_collective_propagates(monkeypatch):
    def _boom(gathered, obj, group=None, **kw):
        raise RuntimeError("nccl down")

    monkeypatch.setattr(torch.distributed, "all_gather_object", _boom)
    monkeypatch.setattr(
        torch.distributed, "get_world_size", lambda group=None: 3
    )
    try:
        FakeSelf(_FakeGroup())._l15_decide_wake_verdict(True, 7, epoch=9)
        raise AssertionError("collective failure was swallowed")
    except RuntimeError as exc:
        assert "nccl down" in str(exc)
