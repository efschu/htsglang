# SPDX-License-Identifier: Apache-2.0
"""L15-W2C: wire the wake SAMPLE CHECK between the optimistic refill and
the decide() collective (design L15-WIRE2-NOTES.md sec 1, 2 and 4).

Pinned contract:
  * _l15_wake_check_and_decide(wake_on, fp, *, epoch) is the ONE fence-
    tail entry: master off -> None with NO sample call and NO collective
    (byte-identical wake); otherwise the sample check runs FIRST, then
    decide() -- the group's single gather stays the only collective.
  * _l15_wake_sample_check() returns (ok, bad, missing) from
    l15_check.sample_check over the stashed manifest's rid-tagged plan
    rows (cap>0: held rows; cap-0: the refilled rows -- the same owned-
    with-L2-source set), frees the scratch pool on every path, and logs
    l15_restore.check_line.  Any raise that is NOT the sampled mismatch
    itself folds into an all-bad vote (0, k_or_len, missing) -- the
    gather is never skipped, the collective is never caught.
  * bad > 0 on any rank -> decide() raises L15CheckRefused on EVERY
    rank with the identical refusal_message text.

Hermetic: torch.distributed monkeypatched as in
test_weg2_l15_decide_wiring_1001.py; the unbound production methods run
against a FakeSelf as in test_weg2_l15_opt_refill_1001.py.
"""

import pathlib
import sys
from types import SimpleNamespace

import pytest
import torch

REPO = pathlib.Path(__file__).resolve().parents[4]
sys.path.insert(0, str(REPO / "python"))

from sglang.srt.managers.scheduler_components import weight_updater as wu  # noqa: E402
from sglang.srt.weg2 import (  # noqa: E402
    l15_bind,
    l15_check,
    l15_restore,
    l15_scratch,
    l15_wake_check,
)

WU = wu.SchedulerWeightUpdaterManager


class _FakeGroup:
    """Identity stand-in for the tp cpu process group."""


class _Scratch:
    """Scratch stand-in that records every free()."""

    def __init__(self):
        self.freed = 0

    def free(self):
        self.freed += 1


def _mk_manifest():
    # rank 0 (tp=2, prefix [0,1,2], S=2) owns slots 2 and 6 (slot % 2 == 0);
    # both carry an L2 source, so the sample plan is non-empty.
    return SimpleNamespace(
        epoch=9,
        rows_by_rank=(2, 2),
        spans=[SimpleNamespace(rid="r1", slots=[2, 6],
                               l2_slots=[10, 11], l2_gens=[1, 1])],
    )


def _mk_sched(group):
    return SimpleNamespace(
        world_group=SimpleNamespace(cpu_group=group),
        tp_size=2,
        server_args=SimpleNamespace(tp_size=2),
        tp_worker=SimpleNamespace(
            model_runner=SimpleNamespace(token_to_kv_pool=object())),
        tree_cache=object(),
    )


class FakeSelf:
    """Unbound-method style: the production methods run verbatim."""

    _l15_wake_sample_check = WU._l15_wake_sample_check
    _l15_decide_wake_verdict = WU._l15_decide_wake_verdict
    _l15_wake_check_and_decide = WU._l15_wake_check_and_decide

    def __init__(self, group, manifest=None):
        self.scheduler = _mk_sched(group)
        self._l15_wake_manifest = manifest
        self._rank = 0

    def _weg2_rank(self):
        return self._rank


def _fake_gather(monkeypatch, votes_per_rank, calls, objs):
    """dist.all_gather_object stand-in: records the group and the vote
    obj, fills the gathered list from the table (in place, as dist does)
    and pins the world size to the table length."""

    def _fake(gathered, obj, group=None, **kw):
        calls.append(group)
        objs.append(obj)
        gathered[:] = list(votes_per_rank) if votes_per_rank else [obj]

    monkeypatch.setattr(torch.distributed, "all_gather_object", _fake)
    monkeypatch.setattr(
        torch.distributed, "get_world_size",
        lambda group=None: len(votes_per_rank) if votes_per_rank else 2,
    )


def _patch_pools(monkeypatch, scratch):
    """Minimal live_host_pools / make_scratch_pool / sample_check seams."""
    host = SimpleNamespace(_arena_page_tokens=1)
    monkeypatch.setattr(l15_bind, "live_host_pools", lambda tc: (host, None))
    monkeypatch.setattr(l15_scratch, "make_scratch_pool",
                        lambda pool, rows: scratch)


def test_vote_carries_the_sample_check_counts(monkeypatch):
    # clean rows: the check's (ok, bad, missing) lands in the gathered
    # vote, verdict unchanged ("hold").
    calls, objs = [], []
    votes = [l15_restore.check_vote(7, 5, 0, 0, ()),
             l15_restore.check_vote(7, 5, 0, 0, ())]
    _fake_gather(monkeypatch, votes, calls, objs)
    fs = FakeSelf(_FakeGroup(), manifest=_mk_manifest())
    got = fs._l15_decide_wake_verdict(True, 7, (5, 0, 0), epoch=9)
    assert got == "hold"
    assert objs == [(7, 5, 0, 0, ())]
    assert calls == [fs.scheduler.world_group.cpu_group]


def test_corrupted_row_refuses_every_rank_with_the_same_text(monkeypatch):
    # one corrupted held row on rank 1 (bad=1) -> L15CheckRefused on all
    # three ranks, identical refusal text.
    calls, objs = [], []
    votes = [l15_restore.check_vote(7, 3, 0, 0, ()),
             l15_restore.check_vote(7, 0, 1, 0, ()),
             l15_restore.check_vote(7, 3, 0, 0, ())]
    _fake_gather(monkeypatch, votes, calls, objs)
    msgs = []
    for rank in range(3):
        fs = FakeSelf(_FakeGroup(), manifest=_mk_manifest())
        fs._rank = rank
        with pytest.raises(l15_wake_check.L15CheckRefused) as ei:
            fs._l15_decide_wake_verdict(
                True, 7, (0, 1, 0) if rank == 1 else (3, 0, 0), epoch=9)
        msgs.append(str(ei.value))
    assert len(set(msgs)) == 1, msgs
    assert "epoch=9" in msgs[0]
    assert len(calls) == 3, "every rank entered the collective"


def test_sample_check_raise_votes_all_bad_gather_still_runs(monkeypatch):
    # a sample_check raising for a reason OTHER than the mismatch itself
    # must not skip the gather: all-bad vote, then the collective runs and
    # the group refusal comes out of decide() like any other.
    calls, objs = [], []
    _fake_gather(monkeypatch, None, calls, objs)
    scratch = _Scratch()
    _patch_pools(monkeypatch, scratch)

    def _boom(*a, **kw):
        raise RuntimeError("device compare exploded")

    monkeypatch.setattr(l15_check, "sample_check", _boom)
    fs = FakeSelf(_FakeGroup(), manifest=_mk_manifest())
    ok, bad, missing = fs._l15_wake_sample_check()
    assert ok == 0 and bad > 0
    assert scratch.freed == 1, "scratch freed even on the failing path"
    # L15-FIX-CHECK-FALLBACK (N3r): the refusal no longer escapes -- every
    # rank gets the same "fallback" verdict instead of an exception.
    assert fs._l15_wake_check_and_decide(True, 7, epoch=9) == "fallback"
    assert len(calls) == 1, "one collective per rank, even after the raise"


def test_scratch_freed_on_every_path(monkeypatch):
    # the clean path and the failing path both free the scratch pool.
    calls, objs = [], []
    _fake_gather(monkeypatch, None, calls, objs)
    scratch = _Scratch()
    _patch_pools(monkeypatch, scratch)
    monkeypatch.setattr(l15_check, "sample_check",
                        lambda plan, host, live, scr, pt, k=64: (k, 0, 0))
    fs = FakeSelf(_FakeGroup(), manifest=_mk_manifest())
    assert fs._l15_wake_sample_check() == (64, 0, 0)
    assert scratch.freed == 1

    def _boom(*a, **kw):
        raise RuntimeError("compare blew up after partial load")

    monkeypatch.setattr(l15_check, "sample_check", _boom)
    fs._l15_wake_sample_check()
    assert scratch.freed == 2, "freed again after the raising path"


def test_master_off_no_sample_call_no_collective(monkeypatch):
    calls, objs = [], []
    _fake_gather(monkeypatch, None, calls, objs)
    scratch = _Scratch()
    _patch_pools(monkeypatch, scratch)

    def _never(*a, **kw):
        raise AssertionError("master off must not sample-check")

    monkeypatch.setattr(l15_check, "sample_check", _never)
    fs = FakeSelf(_FakeGroup(), manifest=_mk_manifest())
    assert fs._l15_wake_check_and_decide(False, 7, epoch=9) is None
    assert calls == []
    assert scratch.freed == 0


class _Tree:
    def __init__(self):
        self.nodes = {}
        self.resets = 0

    def reset(self):
        self.nodes = {}
        self.resets += 1


class _Alloc:
    TOTAL = set(range(16))

    def __init__(self):
        self.free = set(self.TOTAL)


class _ActSelf:
    """Deferred-path probe: a successful optimistic refill followed by
    the W114 "deferred" verdict (kv refusal -> group_ok False)."""

    _l15_wake_act = WU._l15_wake_act

    def __init__(self, sched):
        self.scheduler = sched
        self._l15_wake_manifest = _mk_manifest()
        self._l15_wake_refill = False


def test_deferred_keeps_refilled_rows_resident():
    # OPT open point: optimistic refill SUCCEEDED, verdict "deferred" ->
    # the act must neither drop nor reset: refilled rows STAY resident
    # (tree + allocator as the refill left them), drop count 0.
    sched = SimpleNamespace(
        tree_cache=_Tree(), token_to_kv_pool_allocator=_Alloc())
    sched.tree_cache.nodes["r1"] = [2, 6]
    sched.token_to_kv_pool_allocator.free -= {2, 6}
    before = (dict(sched.tree_cache.nodes),
              set(sched.token_to_kv_pool_allocator.free))
    a = _ActSelf(sched)
    dropped = WU._l15_wake_act(a, sched, "deferred",
                               group_ok=False, master_on=True)
    assert dropped == 0
    assert (dict(sched.tree_cache.nodes),
            set(sched.token_to_kv_pool_allocator.free)) == before
    assert sched.tree_cache.resets == 0, "deferred must not flush"


def test_no_manifest_still_votes_none(monkeypatch):
    # a rank without a manifest has nothing to sample; it still enters
    # the collective (xsn410) and votes None.
    calls, objs = [], []
    votes = [None, None]
    _fake_gather(monkeypatch, votes, calls, objs)
    fs = FakeSelf(_FakeGroup(), manifest=None)
    check = fs._l15_wake_sample_check()
    assert check is None
    got = fs._l15_decide_wake_verdict(True, None, check, epoch=9)
    assert got == "none"
    assert len(calls) == 1, "the manifest-less rank entered the collective"
