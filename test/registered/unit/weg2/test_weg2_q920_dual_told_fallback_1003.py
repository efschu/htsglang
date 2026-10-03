# SPDX-License-Identifier: Apache-2.0
"""Auftrag 920 (analysis 860): the dual told fallback re-prefills big prompts whose prefix is there.

Boots fs10031727 bc2bd121c0 / fs10031814 36b5a4d3e9 / fs10031909 ceff4aae7b (27B NVFP4 dual):
40 'big prompt, prefix in the store, P recomputes all (pc=0)' cases, 0.7-0.9 per minute,
0.5 M new tokens per 13 minutes. Four fixes, all behind the dual P gate (dual_p_kv_stage.armed):

  1. Q-920 OBSERVABILITY  -- ROOM-NOHOLD why=... on every refusal of the Q-693 hold, END on 'stable'
                             without a hold, the pool terms in the NO-ROOM line.
  2. Q-920 B ADOPT        -- UNRESUMABLE: every follower acks the SAME a > 0 below told and PP0's own
                             tree resumes at a -> told=a on every rank instead of told=0.
  3. Q-920 C FIDELITY     -- the PF TOLD-FALLBACK line names the pp0_admissible value (measurement).
  4. Q-920 A1 JUST-FINISHED -- the NO-ROOM ack is held when the predecessor finished a moment ago
                             and its rows are still pinned (publish pins).

Each test is RED on 5342040a72 and green with the fix; every fix has a gate-off twin (flip unchanged).
"""
from __future__ import annotations

import logging
import os
import sys
import time
import types
import unittest.mock as mock

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

import pytest  # noqa: E402

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "managers"))

import _told_ring_pf as R  # noqa: E402

from sglang.srt.managers import weg2_store_told as ST  # noqa: E402
from sglang.srt.managers import weg2_told_fallback as FB  # noqa: E402
from sglang.srt.managers import weg2_told_fidelity as TF  # noqa: E402
from sglang.srt.weg2 import dual_p_kv_stage as DPK  # noqa: E402
from sglang.test.ci.ci_register import register_cpu_ci  # noqa: E402

register_cpu_ci(est_time=4, suite="base-a-test-cpu")

RID = "weg2-0-201"
TOLD = 20224
_KV = {"SGLANG_WEG2_DUAL_P_KV_MAX_TOKENS": "196608"}
DUAL_P_ENV = dict(_KV, SGLANG_WEG2_DUAL_LAYOUT="1", SGLANG_WEG2_GROUP="P")
OFF_ENVS = [dict(_KV, SGLANG_WEG2_DUAL_LAYOUT="", SGLANG_WEG2_GROUP="P"),
            dict(_KV, SGLANG_WEG2_DUAL_LAYOUT="1", SGLANG_WEG2_GROUP="D"),
            dict(_KV, SGLANG_WEG2_DUAL_LAYOUT="1", SGLANG_WEG2_GROUP=""),
            dict(_KV, SGLANG_WEG2_DUAL_LAYOUT="0", SGLANG_WEG2_GROUP="P")]


@pytest.fixture
def dual_p():
    with mock.patch.dict(os.environ, DUAL_P_ENV):
        yield


class _Alloc:
    def __init__(self, avail):
        self.avail = avail
        self.size = 100000

    def available_size(self):
        return self.avail


class _Tree:
    def __init__(self, avail, evictable=0, protected=0, publish=0):
        self.token_to_kv_pool_allocator = _Alloc(avail)
        self.evictable = evictable
        self.protected = protected
        self.ongoing_write_through = {i: None for i in range(publish)}
        self.ongoing_backup = {}

    def evictable_size(self):
        return self.evictable

    def protected_size(self):
        return self.protected


def _pred(rid="weg2-0-200", fill=21766, origin=21766):
    return types.SimpleNamespace(rid=rid, fill_ids=list(range(fill)), origin_input_ids=list(range(origin)))


def _sched(pred=None, *, avail=2697, publish=0, protected=0):
    req = types.SimpleNamespace(rid=RID, _weg2_early_told=None)
    s = types.SimpleNamespace(
        tree_cache=_Tree(avail, 0, protected, publish), ps=types.SimpleNamespace(pp_rank=1, pp_size=3),
        waiting_queue=[req], max_total_num_tokens=100000,
        mbs=[types.SimpleNamespace(reqs=[pred]) if pred is not None else None, None, None],
        running_mbs=[], running_batch=types.SimpleNamespace(reqs=[]), chunked_req=None,
        max_running_requests=1)
    return s, req


@pytest.fixture
def rows(monkeypatch):
    monkeypatch.setattr(FB, "_loadback_rows", lambda s, r, t: int(t))
    monkeypatch.setattr(TF, "pp0_admissible", lambda s, r, own: None)


def _msgs(caplog):
    return [r.getMessage() for r in caplog.records]


# --- 1. observability -----------------------------------------------------------------------


def test_nohold_names_why_with_the_full_rid_when_no_predecessor_is_in_flight(dual_p, rows, caplog):
    """y8z 19:22:56 PP1 weg2-0-201 room=2697: the check ran, refused, and left no trace."""
    s, req = _sched(None, publish=2, protected=9000)
    with caplog.at_level(logging.WARNING, logger=FB.logger.name):
        assert FB._room_own(s, req, RID, TOLD) == 0
    line = next(m for m in _msgs(caplog) if FB.NOHOLD_MARK in m and "why=no_predecessor" in m)
    for term in ("rid=%s " % RID, "room=2697", "rows=%d" % TOLD, "pending=0", "usage=", "protected=9000",
                 "pending_publish=2"):
        assert term in line, (term, line)


def test_nohold_predecessor_too_small_and_left_queue(dual_p, rows, caplog):
    s, req = _sched(_pred(fill=500, origin=500))
    with caplog.at_level(logging.WARNING, logger=FB.logger.name):
        assert FB._room_own(s, req, RID, TOLD) == 0
        s.waiting_queue = []
        assert FB._room_own(s, req, RID, TOLD) == 0
    ms = _msgs(caplog)
    assert any("why=predecessor_too_small" in m and "pending=500" in m for m in ms)
    assert any("why=left_queue" in m and "rid=%s " % RID in m for m in ms)


def test_stable_without_a_hold_logs_its_end(dual_p, rows, caplog):
    s, req = _sched(None)
    with caplog.at_level(logging.WARNING, logger=FB.logger.name):
        assert FB._room_own(s, req, RID, TOLD) == 0
    assert any(FB.ROOM_HOLD_MARK in m and "END" in m and "how=stable" in m and RID in m for m in _msgs(caplog))


def test_nohold_lines_are_capped(dual_p, rows, caplog):
    s, req = _sched(None)
    with caplog.at_level(logging.WARNING, logger=FB.logger.name):
        for _ in range(100):
            FB._room_own(s, req, RID, TOLD)
    assert sum(FB.NOHOLD_MARK in m for m in _msgs(caplog)) == 32


def test_noroom_line_names_the_pool_terms_and_the_grant(dual_p, rows, caplog):
    s, req = _sched(None, publish=3, protected=7777)
    s.tree_cache.evictable = 5
    actor = types.SimpleNamespace(last_grant=(time.monotonic() - 1.0, 36864))
    s.tp_worker = types.SimpleNamespace(model_runner=types.SimpleNamespace(**{DPK.ACTOR_ATTR: actor}))
    with caplog.at_level(logging.WARNING, logger=FB.logger.name):
        FB._room_own(s, req, RID, TOLD)
    line = next(m for m in _msgs(caplog) if "PF TOLD-ACK NO-ROOM" in m)
    for term in ("full_rid=%s" % RID, "available=2697", "evictable=5", "protected=7777",
                 "pending_publish=3", "grant_new=36864", "grant_age_s="):
        assert term in line, (term, line)


def test_map_granted_records_the_grant_size_and_time():
    actor = DPK.PKvStage.__new__(DPK.PKvStage)
    actor.top, actor.step, actor.mapped_tokens, actor.last_grant = 196608, 4096, 4096, None
    actor.bytes_for = lambda t: int(t)
    actor._move = lambda t: None
    actor._committed = 0
    with mock.patch.object(DPK, "check_cover", lambda *a, **k: None):
        actor.ledger = types.SimpleNamespace(release=lambda n: None)
        actor.map_granted(36864, charged=0)
    assert actor.last_grant[1] == 36864 - 4096 and time.monotonic() - actor.last_grant[0] < 5


class TestQ920ObservabilityFlipUnchanged:
    """Every wrong gate: no NOHOLD line, no END line, the NO-ROOM line is the base text."""

    def test_flip_forms_log_exactly_the_base_lines(self, rows, caplog):
        for env in OFF_ENVS:
            caplog.clear()
            with mock.patch.dict(os.environ, env), caplog.at_level(logging.WARNING, logger=FB.logger.name):
                s, req = _sched(_pred())
                assert FB._room_own(s, req, RID, TOLD) == 0, env
            ms = _msgs(caplog)
            assert len(ms) == 1 and ms[0].startswith("PF TOLD-ACK NO-ROOM"), (env, ms)
            assert "Q-920" not in ms[0] and "NOHOLD" not in ms[0], env
            assert not hasattr(s, "_q920_nohold_n") and not hasattr(s, "_q920_stable_nohold_n"), env


# --- 2. B UNRESUMABLE: told=a instead of told=0 ------------------------------------------------

BIG = "big-prompt-1"
BTOLD = 40960      # y8x weg2-0-45: told 40960, both followers acked 34588
BRES = 34588


def _resumable_at(depth):
    """pp0_admissible stand-in: the tree resumes up to ``depth`` keys, never beyond."""
    return lambda s, r, told: min(int(told), int(depth))


def _b_ring(monkeypatch, *, res_follower=BRES, res_pp0=None, adopt_env=None, acks=None):
    for k in list(os.environ):
        if k.startswith("SGLANG_WEG2_TOLD") or k == "SGLANG_WEG2_DUAL_TOLD_ADOPT":
            monkeypatch.delenv(k, raising=False)
    monkeypatch.setenv(FB.ENV_FALLBACK, "1")
    monkeypatch.setenv("SGLANG_WEG2_TOLD_ABSOLUTE", "1")
    if adopt_env is not None:
        monkeypatch.setenv(FB.ENV_ADOPT, adopt_env)
    depth = {1: res_follower, 2: res_follower}
    if acks:
        depth.update(acks)
    pp0_depth = BTOLD if res_pp0 is None else res_pp0
    calls = {"rank": None}

    def fake(sched, req, told):
        rank = int(sched.ps.pp_rank)
        return min(int(told), int(pp0_depth if rank == 0 else depth[rank]))

    monkeypatch.setattr(TF, "pp0_admissible", fake)
    read = {r: {BIG: 0.1} for r in range(3)}
    return R.Ring(ST, monkeypatch, {BIG: BTOLD}, read)


def _drive(ring):
    ring.arrive(BIG)
    ring.run(80)
    return ring.plans(BIG), ring.wire_objs()


def test_b_all_followers_ack_the_same_resumable_depth_pp0_admits_at_it(dual_p, monkeypatch, caplog):
    """RED on 5342040a72: PF TOLD-FALLBACK told=40960 -> 0 reason=mismatch (y8x 292 k, y8z 224 k tokens)."""
    ring = _b_ring(monkeypatch)
    with caplog.at_level(logging.WARNING):
        plans, wire = _drive(ring)
    assert all(len(p) == 1 for p in plans) and plans[0] == plans[1] == plans[2], plans
    assert plans[0][0][2] == BRES                     # the prefix cap: a on EVERY rank, PP0 included
    admit = [o for o in wire if type(o).__name__ == "Weg2StoreAdmit"]
    assert len(admit) == 1 and admit[0].told == BRES
    assert getattr(admit[0], FB.WIRE_ADOPT, None) == 1 and getattr(admit[0], FB.WIRE_FALLBACK, None) is None
    assert all(not s.tree_cache.released for s in ring.stages)       # no read was cut
    assert all(s.tree_cache.op_refs == 0 for s in ring.stages)
    assert [a[3] for s in ring.stages for a in s.admitted] == [0, 0, 0]  # credit popped on every rank
    assert not getattr(ring.stages[0], "_pf_fallback_n", 0)
    ms = [r.getMessage() for r in caplog.records]
    assert any("PF TOLD-ADOPT rid=%s told=%d -> %d" % (BIG, BTOLD, BRES) in m for m in ms)
    assert sum("PF TOLD-ADOPT ABSORBED" in m for m in ms) == 2


def test_b_followers_that_disagree_still_get_told_zero(dual_p, monkeypatch):
    ring = _b_ring(monkeypatch, acks={2: 30000})
    plans, wire = _drive(ring)
    assert plans[0] == plans[1] == plans[2] and plans[0][0][2] == 0
    assert getattr(wire[-1], FB.WIRE_FALLBACK, None) == 1 and getattr(wire[-1], FB.WIRE_ADOPT, None) is None


def test_b_a_follower_with_no_room_acks_zero_and_the_group_gets_told_zero(dual_p, monkeypatch):
    ring = _b_ring(monkeypatch, acks={2: 0})
    plans, wire = _drive(ring)
    assert plans[0] == plans[1] == plans[2] and plans[0][0][2] == 0


def test_b_pp0_that_cannot_resume_at_a_falls_back_to_zero(dual_p, monkeypatch):
    ring = _b_ring(monkeypatch, res_pp0=BRES - 4096)
    plans, wire = _drive(ring)
    assert plans[0] == plans[1] == plans[2] and plans[0][0][2] == 0
    assert getattr(wire[-1], FB.WIRE_ADOPT, None) is None


def test_b_switch_off_keeps_told_zero(dual_p, monkeypatch):
    ring = _b_ring(monkeypatch, adopt_env="0")
    plans, wire = _drive(ring)
    assert plans[0] == plans[1] == plans[2] and plans[0][0][2] == 0


def test_b_waits_for_the_second_ack_inside_the_frist_only(dual_p, monkeypatch):
    class Sch:
        ps = types.SimpleNamespace(pp_size=3)

    sch = Sch()
    monkeypatch.setattr(TF, "pp0_admissible", _resumable_at(BRES))
    o = FB._Open(told=BTOLD, deadline=10.0, acks={1: BRES})
    FB._pp0_open_map(sch)[BIG] = o
    req = types.SimpleNamespace(rid=BIG)
    monkeypatch.setenv("SGLANG_WEG2_TOLD_ABSOLUTE", "1")
    assert FB.pp0_decide(sch, BIG, 5.0, req, True) is None
    o.acks[2] = BRES
    assert FB.pp0_decide(sch, BIG, 5.0, req, True) == (BRES, FB.REASON_ADOPT)
    o.acks.pop(2)
    assert FB.pp0_decide(sch, BIG, 10.5, req, True) == (0, FB.REASON_MISMATCH)    # the Frist decides: today's 0


def test_b_span_relative_told_is_never_adopted(dual_p, monkeypatch):
    class Sch:
        ps = types.SimpleNamespace(pp_size=3)

    sch = Sch()
    monkeypatch.setattr(TF, "pp0_admissible", _resumable_at(BRES))
    FB._pp0_open_map(sch)[BIG] = FB._Open(told=BTOLD, deadline=10.0, acks={1: BRES, 2: BRES})
    assert FB.pp0_decide(sch, BIG, 1.0, types.SimpleNamespace(rid=BIG), False) == (0, FB.REASON_MISMATCH)


class TestQ920AdoptFlipUnchanged:
    """Every wrong gate: equal follower acks below told still mean told=0 (reason=mismatch), no
    marker on the wire, nothing satisfied."""

    def test_pp0_decide_answers_zero_on_every_wrong_gate(self, monkeypatch):
        monkeypatch.setattr(TF, "pp0_admissible", _resumable_at(BRES))
        monkeypatch.setenv("SGLANG_WEG2_TOLD_ABSOLUTE", "1")
        for env in OFF_ENVS:
            with mock.patch.dict(os.environ, env):
                sch = types.SimpleNamespace(ps=types.SimpleNamespace(pp_size=3))
                FB._pp0_open_map(sch)[BIG] = FB._Open(told=BTOLD, deadline=10.0, acks={1: BRES, 2: BRES})
                assert FB.pp0_decide(sch, BIG, 1.0, types.SimpleNamespace(rid=BIG), True) == \
                    (0, FB.REASON_MISMATCH), env

    def test_the_ring_answers_told_zero_with_the_fallback_marker(self, monkeypatch):
        for env in OFF_ENVS:
            with mock.patch.dict(os.environ, env):
                ring = _b_ring(monkeypatch)
                plans, wire = _drive(ring)
                assert plans[0] == plans[1] == plans[2] and plans[0][0][2] == 0, env
                assert getattr(wire[-1], FB.WIRE_FALLBACK, None) == 1, env
                assert getattr(wire[-1], FB.WIRE_ADOPT, None) is None, env
                assert not any(getattr(s, "_weg2_store_told_satisfied", None) for s in ring.stages), env
