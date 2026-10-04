# SPDX-License-Identifier: Apache-2.0
"""L15-SLEEP-DECIDE-FIRST (INT8 boot 4cf740ad50, 1004).

Three D->P sleeps paid a retain of 0.38-0.54 s (plus arm, snap, post gather)
and were then rolled back by the POST vote: "cap-0 rank: NN owned held
token(s) without an L2 source". The refusal is a pure function of the
manifest, which is fully known after retain's planning steps -- so the group
votes BEFORE the moves.

Pinned here:
* the pre-retain vote equals the POST vote on the manifest the retain
  publishes (no hold is lost that POST would have kept, none is kept that it
  would have refused), for fully backed / mixed / anchor-less chains and for
  capped and cap-0 ranks;
* planning is pure: no buffer, tree or allocator is touched before the vote;
* all ranks reach ONE gather whatever their local state and get the SAME
  verdict (threaded 3-rank simulation; also: a rank whose bind raised votes
  no instead of skipping the collective);
* a refused round runs retain on no rank; an agreed round hands the plan to
  retain, whose result is identical to the unplanned retain;
* switch off (=0) -> the scheduler keeps the order of 4cf740ad50.
"""

from __future__ import annotations

import inspect
import threading
from types import SimpleNamespace

import pytest
import torch

from sglang.srt.weg2 import l15_retain
from sglang.srt.weg2 import l15_sleep_agree as A
from sglang.srt.weg2.l15_manifest import fingerprint
from sglang.srt.weg2.l15_policy import Candidate

PREFIX = [0, 1, 2, 3]  # three ranks, ratio 1:1:1: rank r owns slot % 3 == r
SLOTS = tuple(range(1, 13))
CAND = (Candidate(rid="weg2-8-8", kind="served", last_active=1.0,
                  rows_by_rank=(4, 4, 4), anchor_depth=12, kv_depth=12),)


def _kwargs(l2, anchor_l2=(7, 1), caps=(0, 64, 64), rank=0, events=None):
    events = events if events is not None else []
    buf = torch.arange(40, dtype=torch.float32).reshape(20, 2)
    mbuf = torch.arange(16, dtype=torch.float32).reshape(8, 2)
    ref = {"buf": buf.clone(), "mbuf": mbuf.clone()}

    class _Alloc:
        def clear(self):
            events.append("clear")
            self.free_pages = torch.arange(1, 33, dtype=torch.int64)
            self.release_pages = torch.empty(0, dtype=torch.int64)

        free_pages = torch.empty(0, dtype=torch.int64)
        release_pages = torch.empty(0, dtype=torch.int64)
        size = 32

    kw = dict(
        candidates=list(CAND),
        node_of=lambda rid: SimpleNamespace(),
        slots_of=lambda rid: SLOTS,
        anchor_slot_of=lambda rid: 5,
        l2_of=lambda rid: (tuple(l2), tuple(1 if x >= 0 else -1 for x in l2)),
        anchor_l2_of=lambda rid: tuple(anchor_l2),
        l2_lanes_of=lambda rid: (),
        rewrite_tree=lambda node, kvm, am, vis: events.append("rewrite"),
        caps_rows_by_rank=caps,
        cap_anchor_slots=4,
        prefix=PREFIX,
        rank=rank,
        epoch=8,
        pid=1,
        kv_buffers=[buf],
        mamba_buffers=[mbuf],
        allocator=_Alloc(),
        reset_keep=lambda nodes: events.append("reset_keep"),
        set_keep=lambda b, r: events.append("set_keep"),
        manifest_path=None,
        log=lambda s: events.append("log:" + s),
    )
    return kw, ref, events


SCENARIOS = {
    "backed": ([100 + i for i in range(12)], (7, 1)),
    "mixed": ([-1] * 9 + [100, 101, 102], (7, 1)),
    "no-anchor": ([100 + i for i in range(12)], (-1, -1)),
}


@pytest.mark.parametrize("name", sorted(SCENARIOS))
@pytest.mark.parametrize("rank,caps", [(0, (0, 64, 64)), (1, (0, 64, 64)), (0, (64, 64, 64))])
def test_pre_vote_equals_post_vote(tmp_path, name, rank, caps):
    l2, anchor = SCENARIOS[name]
    kw, ref, ev = _kwargs(l2, anchor, caps=caps, rank=rank)
    kw["manifest_path"] = str(tmp_path / "m.json")
    cap = caps[rank]
    # PRE: planning only
    rp = l15_retain.plan_round(
        candidates=kw["candidates"], slots_of=kw["slots_of"],
        anchor_slot_of=kw["anchor_slot_of"], caps_rows_by_rank=caps,
        cap_anchor_slots=4, prefix=PREFIX, epoch=8, log=lambda s: None)
    assert rp is not None
    assert not [e for e in ev if e in ("rewrite", "reset_keep", "clear", "set_keep")]
    assert torch.equal(kw["kv_buffers"][0], ref["buf"]), "planning moved a buffer"
    man = None
    if cap <= 0:
        man = l15_retain.manifest_of_plan(
            rp, candidates=kw["candidates"], l2_of=kw["l2_of"],
            anchor_l2_of=kw["anchor_l2_of"], l2_lanes_of=kw["l2_lanes_of"],
            epoch=8, pid=1)
    pre = A.pre_retain_vote(rp, man, cap, prefix=PREFIX, rank=rank)
    # POST: the real retain, then the unchanged post vote
    res = l15_retain.retain_at_sleep(rewrite_tree=kw.pop("rewrite_tree"), **kw)
    assert res is not None
    post = A.post_vote(res, cap, prefix=PREFIX, rank=rank)
    assert pre == post, (name, rank, caps, pre, post)
    if name == "backed" or cap > 0:
        assert pre is None            # no hold lost that POST would keep
    else:
        assert pre is not None        # ...and none kept that POST refuses
    if man is not None:
        assert fingerprint(man) == fingerprint(res.manifest)


def test_planned_retain_is_the_unplanned_retain(tmp_path):
    kw1, _, _ = _kwargs(SCENARIOS["backed"][0], caps=(64, 64, 64), rank=1)
    kw2, _, _ = _kwargs(SCENARIOS["backed"][0], caps=(64, 64, 64), rank=1)
    kw1["manifest_path"] = str(tmp_path / "a.json")
    kw2["manifest_path"] = str(tmp_path / "b.json")
    rt1, rt2 = kw1.pop("rewrite_tree"), kw2.pop("rewrite_tree")
    a = l15_retain.retain_at_sleep(rewrite_tree=rt1, **kw1)
    rp = l15_retain.plan_round(
        candidates=kw2["candidates"], slots_of=kw2["slots_of"],
        anchor_slot_of=kw2["anchor_slot_of"], caps_rows_by_rank=(64, 64, 64),
        cap_anchor_slots=4, prefix=PREFIX, epoch=8, log=lambda s: None)
    b = l15_retain.retain_at_sleep(rewrite_tree=rt2, planned=rp, **kw2)
    assert fingerprint(a.manifest) == fingerprint(b.manifest)
    assert a.a_h == b.a_h and a.hold.rids == b.hold.rids
    assert torch.equal(kw1["kv_buffers"][0], kw2["kv_buffers"][0])


class _Group:
    """3 ranks as threads; gather = barrier + shared slots. Counts gathers."""

    def __init__(self, n):
        self.n = n
        self.bar = threading.Barrier(n, timeout=20)
        self.slots = [None] * n
        self.calls = [0] * n

    def gather_for(self, r):
        def g(v):
            self.calls[r] += 1
            self.slots[r] = v
            self.bar.wait()
            out = list(self.slots)
            self.bar.wait()
            return out
        return g


def _run_group(per_rank):
    """per_rank: list of (kwargs_or_None, cap). Returns [(planned, dec)] and the group."""
    grp = _Group(len(per_rank))
    out = [None] * len(per_rank)

    def work(r):
        kw, cap = per_rank[r]
        out[r] = A.decide_first(
            kw, False, lambda: cap, lambda: (r, PREFIX), grp.gather_for(r),
            lambda s: None, lambda s: None)

    ts = [threading.Thread(target=work, args=(r,)) for r in range(len(per_rank))]
    [t.start() for t in ts]
    [t.join(30) for t in ts]
    assert all(o is not None for o in out), "a rank never returned (hang)"
    return out, grp


def test_group_verdict_is_uniform_when_only_the_cap0_rank_refuses():
    mixed = SCENARIOS["mixed"][0]
    ranks = []
    for r, cap in enumerate((0, 64, 64)):
        kw, _, _ = _kwargs(mixed, caps=(0, 64, 64), rank=r)
        ranks.append((kw, cap))
    out, grp = _run_group(ranks)
    decs = [d for _p, d in out]
    assert decs[0] is not None and "without an L2 source" in decs[0]
    assert len(set(decs)) == 1, "ranks disagree on the verdict"
    assert all(p is None for p, _d in out), "a rank would still retain"
    assert grp.calls == [1, 1, 1]


def test_group_agrees_to_hold_when_every_rank_can():
    ranks = []
    for r, cap in enumerate((0, 64, 64)):
        kw, _, _ = _kwargs(SCENARIOS["backed"][0], caps=(0, 64, 64), rank=r)
        ranks.append((kw, cap))
    out, grp = _run_group(ranks)
    assert [d for _p, d in out] == [None, None, None]
    assert all(p is not None and p.hs.rids for p, _d in out)
    assert grp.calls == [1, 1, 1]


def test_a_rank_whose_bind_raised_votes_no_and_still_gathers_once():
    ranks = []
    for r, cap in enumerate((0, 64, 64)):
        kw, _, _ = _kwargs(SCENARIOS["backed"][0], caps=(0, 64, 64), rank=r)
        ranks.append((None if r == 2 else kw, cap))
    out, grp = _run_group(ranks)
    decs = [d for _p, d in out]
    assert decs == ["round did not arm"] * 3
    assert all(p is None for p, _d in out)
    assert grp.calls == [1, 1, 1]


def test_a_planning_exception_is_a_no_vote_not_a_skipped_gather():
    ranks = []
    for r, cap in enumerate((0, 64, 64)):
        kw, _, _ = _kwargs(SCENARIOS["backed"][0], caps=(0, 64, 64), rank=r)
        if r == 1:
            kw["slots_of"] = lambda rid: (_ for _ in ()).throw(RuntimeError("boom"))
        ranks.append((kw, cap))
    out, grp = _run_group(ranks)
    assert len({d for _p, d in out}) == 1 and out[0][1] is not None
    assert grp.calls == [1, 1, 1]


def test_reuse_round_votes_ok_and_gathers():
    grp = _Group(1)
    p, d = A.decide_first(None, True, lambda: 0, lambda: (0, PREFIX),
                          grp.gather_for(0), lambda s: None, lambda s: None)
    assert (p, d) == (None, None) and grp.calls == [1]


def test_switch_default_on_and_off(monkeypatch):
    assert A.decide_first_on({}) is True
    assert A.decide_first_on({"SGLANG_WEG2_L15_SLEEP_DECIDE_FIRST": "0"}) is False
    assert A.decide_first_on({"SGLANG_WEG2_L15_SLEEP_DECIDE_FIRST": "1"}) is True


def test_scheduler_wiring_one_gather_outside_the_pre_move_try():
    from sglang.srt.managers import scheduler

    src = inspect.getsource(scheduler)
    pre_try_end = src.index("L15-RETAIN failed before the move (flushing as today)")
    dec = src.index("_l15_sa2.decide_first(")
    retain = src.index("_l15_res = l15_retain.retain_at_sleep(")
    post = src.index("_l15_sa2.post_vote(_l15_res, l15_shadow.own_cap_rows(")
    assert pre_try_end < dec < retain < post
    guard = src[dec - 600:dec]
    assert "_l15_dfirst" in guard
    # the switch guards the whole decision; off = no planned kwarg effect
    assert "decide_first_on(os.environ)" in src[pre_try_end:dec]
    assert "planned=_l15_planned," in src[retain:retain + 900]
    # a refusal rides the PRE guard: the POST gather is then skipped uniformly
    tail = src[dec:retain]
    assert "_l15_pre_why = _l15_dec" in tail and "_l15_kwargs = None" in tail
    assert "if _l15_agree_on and _l15_pre_why is None:" in src[retain:post + 400]
    assert "decide_ms=%.0f" in src


def test_switch_off_scheduler_path_is_the_old_order():
    """=0: _l15_dfirst False -> no decide_first, no planned plan: retain plans
    itself exactly as before (planned=None)."""
    from sglang.srt.managers import scheduler

    src = inspect.getsource(scheduler)
    dec = src.index("if _l15_dfirst:")
    nxt = src.index("if _l15_reuse is not None:\n                _l15_res = _l15_reuse", dec)
    block = src[dec:nxt]
    assert block.count("decide_first(") == 1
    # planned defaults to None before the guard
    assert "_l15_planned = None\n            _l15_dfirst = False" in src
