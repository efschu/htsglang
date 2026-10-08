# SPDX-License-Identifier: Apache-2.0
"""L15-SLEEP-AGREE (N4f: ranks diverged across the sleep -> W50; bind and
retain paid for discarded holds) + L15-FLIPCOST-4 (bind owner counts) +
L15-FIX-REFILL-POOL (refill wrote to the hybrid wrapper -> bad=64)."""

from __future__ import annotations

import inspect
from types import SimpleNamespace

import pytest

from flliper.srt.pdflip import l15_bind, l15_keep_split, l15_sleep_agree as A


def test_can_hold_rules():
    env_on = {"FLLIPER_PDFLIP_L15_REFILL": "1"}
    assert A.can_hold(0, {}, lambda: True) is not None          # cap 0, no refill
    assert A.can_hold(0, env_on, lambda: False) is None          # cap 0 + refill
    assert "not split" in A.can_hold(100, {}, lambda: False)     # first sleep
    assert A.can_hold(100, {}, lambda: True) is None


def test_agree_needs_every_rank():
    assert A.agree(None, lambda v: [None, None, v]) is None
    assert A.agree(None, lambda v: [None, "cap-0 rank without refill", v]) \
        == "cap-0 rank without refill"


def test_split_ready_follows_the_saver(monkeypatch):
    from flliper.srt.pdflip import l15_hold_share

    l15_keep_split.forget_all()
    assert A.split_ready_native() is False                       # nothing split
    l15_keep_split._HOLD[0x10] = ((0, 4),)
    monkeypatch.setattr(l15_hold_share, "list_extents", lambda p: [(0, 4), (4, 4)])
    assert A.split_ready_native() is True
    monkeypatch.setattr(l15_hold_share, "list_extents", lambda p: [])   # stock
    assert A.split_ready_native() is False
    l15_keep_split.forget_all()


def test_undo_gives_everything_back(tmp_path):
    m = tmp_path / "m.json"
    m.write_text("{}")
    calls = []
    wu = SimpleNamespace(_l15_clear_tms_keep_spans=lambda s: calls.append("keep") or 3,
                         _l15_release_host_hold_refs=lambda s: calls.append("refs") or 1)
    pub = SimpleNamespace(close=lambda: calls.append("pub"))
    sched = SimpleNamespace(weight_updater=wu, _l15_share_pub=pub)
    A.undo_armed(sched, str(m), lambda s: None)
    assert not m.exists() and calls == ["keep", "refs", "pub"]
    assert sched._l15_share_pub is None


def test_flush_wiring_pre_before_bind_post_after_arm():
    from flliper.srt.managers import scheduler

    src = inspect.getsource(scheduler)
    pre = src.index("_l15_pre_why = _l15_sa2.agree(_l15_mine, _l15_gather)")
    bind = src.index("_l15_kwargs = l15_bind.build_retain_kwargs(")
    arm = src.index("if not l15_keep_arm.arm_keep_spans(")
    post = src.index('_l15_sa2.post_vote(_l15_res, l15_shadow.own_cap_rows(')
    assert pre < bind < arm < post
    assert ") and _l15_pre_why is None:" in src[pre:bind]
    assert "_l15_sa2.undo_armed(self" in src[post:post + 800]


def test_owner_counts_match_the_scalar_rule():
    from flliper.srt.pdflip.l15_compact import owner_of

    prefix = [0, 7, 11, 16]
    slots = list(range(1, 4000, 3))
    want = [0, 0, 0]
    for s_ in slots:
        want[owner_of(s_, prefix)] += 1
    assert l15_bind._owner_counts(slots, prefix) == tuple(want)
    with pytest.raises(ValueError):
        l15_bind._owner_counts([1], [2, 3, 6])      # residue below prefix[0]


def test_refill_and_sample_use_the_full_attention_pool():
    from flliper.srt.managers.scheduler_components import weight_updater as wu

    src = inspect.getsource(wu.SchedulerWeightUpdaterManager._l15_do_refill)
    assert 'device_pool = _l15_kvp(getattr(mr, "token_to_kv_pool", None))' in src
    s2 = inspect.getsource(wu.SchedulerWeightUpdaterManager._l15_wake_sample_check)
    assert "device_pool = _l15_kvp(" in s2
    s3 = inspect.getsource(wu.SchedulerWeightUpdaterManager._l15_wake_check_and_decide)
    assert "self._l15_wake_sample_check() if fp is not None else None" in s3


def test_post_vote_cap0_needs_every_anchor_identity():
    sp_ok = SimpleNamespace(rid="a", anchor_l2_slot=5)
    sp_no = SimpleNamespace(rid="b", anchor_l2_slot=-1)
    res = lambda *sps: SimpleNamespace(manifest=SimpleNamespace(spans=sps))
    assert A.post_vote(None, 10) == "round did not arm"
    assert A.post_vote(res(sp_ok, sp_no), 10) is None          # capped: no refill
    assert A.post_vote(res(sp_ok), 0) is None
    assert "['b']" in A.post_vote(res(sp_ok, sp_no), 0)
