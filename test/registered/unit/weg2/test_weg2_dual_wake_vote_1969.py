# SPDX-License-Identifier: Apache-2.0
"""#1969 WAKE-VOTE: a /resume_memory_occupation that finds P NOT asleep is a no-op on every rank,
decided before any rank-local effect (wake_reclaim, group fence). Default OFF.

Metal b9p (Image b9p, Kopf 6b9d995bdd, 05.10. 06:28-06:30Z, deskq/done/1966): the front issued the sleep
(p_state=sleeping at once), the sleep leg stood in its quiesce (/flush_cache, 1548 ms, answered 400, P not
idle), the next tick saw `sleeping` + seat done + room_ok and woke P (front 19442-44); P had never slept.
Stage-1 loans stood: PP0 1520435200 B (reclaim refused, free 405471232), PP1 1182793728, PP2 1147142144
(reclaim ok). Only PP0's wake_reclaim raised Weg2DualPWakeShort; the raise on the FIRST PP rank sent the
leg-failed group fence into a send PP1 never took (#973 RingCommitTimeout, W17).

DANGER DIRECTIONS: (a) the rule fires on a P that IS asleep (a real wake dropped), (b) it fires on a
rank-local reading (loan, clock) so ranks disagree, (c) it fires outside the dual P gate (flip/NF/D).
"""
from __future__ import annotations

import ast
import asyncio
import collections
import inspect
import os
import tempfile
import types

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

import pytest

from sglang.srt.managers import io_struct
from sglang.srt.managers.scheduler_components import weight_updater as WU
from sglang.srt.weg2 import card_kv_ledger as K
from sglang.srt.weg2 import dual_d_priority as DP
from sglang.srt.weg2 import dual_p_kv_stage as PK
from sglang.srt.weg2 import front as FR
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=3, suite="base-a-test-cpu")

# b9p F.P:197414-16 (stage-1 lend), 197549 (PP0 reclaim refused, free=405471232), 197569/197605 (PP1/PP2 ok)
LENT = (1520435200, 1182793728, 1147142144)
PP0_FREE = 405471232
WAKE_TAGS = ["weights_0", "weights_1", "kv_cache"]
ENV_ON = {"SGLANG_WEG2_DUAL_LAYOUT": "1", "SGLANG_WEG2_GROUP": "P", PK.WAKE_VOTE_ENV: "1"}


@pytest.fixture(autouse=True)
def _env(monkeypatch):
    for k in ("SGLANG_WEG2_DUAL_LAYOUT", "SGLANG_WEG2_GROUP", PK.WAKE_VOTE_ENV):
        monkeypatch.delenv(k, raising=False)
    monkeypatch.setattr(PK, "_republish_stage", lambda s, a: None)


class _Rank:
    """One P rank at the b9p stage: awake, stage-1 loan standing (PP1/PP2 already reclaimed theirs)."""

    def __init__(self, r, offload=()):
        path = os.path.join(tempfile.mkdtemp(prefix="wv1969"), "card")
        d = K.CardKvLedger(path, "D")
        d.contribute(2 << 30, committed=0)
        p = K.CardKvLedger(path, "P")
        p.contribute(0)
        self.r = r
        self.ledger = p
        self.actor = types.SimpleNamespace(ledger=p, mapped_tokens=0, _sleep_lent=0,
                                           _awake_lent=LENT[r] if r == 0 else 0)
        sched = types.SimpleNamespace(tp_worker=types.SimpleNamespace(
            model_runner=types.SimpleNamespace(pp_rank=r, **{PK.ACTOR_ATTR: self.actor})))
        self.calls = []
        outer = self
        # the Manager's own methods are replaced by recorders: any call past the no-op decision shows up
        class Fake:
            scheduler = sched
            offload_tags = set(offload)

            def _weg2_raise_pending_seam_refusal(self, *a, **k):
                outer.calls.append("seam")

            def _weg2_leg_replay(self, op, req):
                return None

            def _weg2_leg_failed(self, what, exc):
                outer.calls.append("leg_failed:%s" % type(exc).__name__)   # = the group fence with ok=False

            def _weg2_group_fence(self, *a, **k):
                outer.calls.append("fence")
                return {}

            def _weg2_leg_commit(self, *a, **k):
                outer.calls.append("commit")

        self.fake = Fake()
        if r == 0:   # PP0: P lent 1520435200 B, D grew into it: the pool has only 405471232 B free
            p.lend(LENT[0])
            d.request(p.state().budget - PP0_FREE)

    def wake(self, monkeypatch, env):
        for k, v in env.items():
            monkeypatch.setenv(k, v)
        req = io_struct.ResumeMemoryOccupationReqInput(tags=list(WAKE_TAGS), epoch="dpw1")
        return WU.SchedulerWeightUpdaterManager.resume_memory_occupation(self.fake, req)


def _ranks(offload=()):
    return [_Rank(r, offload) for r in range(3)]


def test_default_off_b9p_pp0_throws_into_the_fence(monkeypatch):
    """OFF = old path: PP0 (loan standing, pool short) raises WAKE-SHORT in the wake and enters the
    leg-failed group fence alone; PP1/PP2 have nothing to take back."""
    rk = _ranks()
    # state first: PP0's pool really is short of its loan
    assert rk[0].ledger.state().free < LENT[0]
    with pytest.raises(PK.Weg2DualPWakeShort, match="1520435200 B"):
        rk[0].wake(monkeypatch, {"SGLANG_WEG2_DUAL_LAYOUT": "1", "SGLANG_WEG2_GROUP": "P"})
    assert rk[0].calls == ["seam", "leg_failed:Weg2DualPWakeShort"]
    monkeypatch.setenv("SGLANG_WEG2_DUAL_LAYOUT", "1")
    monkeypatch.setenv("SGLANG_WEG2_GROUP", "P")
    assert PK.wake_reclaim(rk[1].fake.scheduler) == 0 and PK.wake_reclaim(rk[2].fake.scheduler) == 0


def test_on_not_asleep_wake_is_a_noop_on_every_rank(monkeypatch):
    """The b9p case: quiesce 400, P never slept, the wake comes anyway. ON: all three ranks answer the
    no-op, none touches the loan, none enters a fence, none records the leg."""
    rk = _ranks()
    before = [(x.actor._awake_lent, x.actor._sleep_lent, x.ledger.state().free) for x in rk]
    for x in rk:
        out = x.wake(monkeypatch, ENV_ON)
        assert isinstance(out, io_struct.ResumeMemoryOccupationReqOutput) and out.per_tag is None
        assert x.calls == ["seam"], x.calls                     # no leg_failed, no fence, no commit
    assert before == [(x.actor._awake_lent, x.actor._sleep_lent, x.ledger.state().free) for x in rk]
    assert rk[0].actor._awake_lent == LENT[0]                   # the stage-1 loan stays for awake_reclaim


def test_on_after_quiesce_400_then_stopped_the_late_wake_is_a_noop(monkeypatch):
    """The whole b9p order: tick issues sleep (p_state=sleeping), next tick wakes (old fallback, 1956 off),
    then the quiesce answers 400 and _dual_p_sleep writes `stopped`; the wake RPC reaches P regardless."""
    st = DP.PressureStages(sleep_capable=True, sleep_after=1)
    st.p_state = "lent"
    assert st.tick(pressure=1, p_committed=0, free_min=0, p_grant_bytes=0, d_air_bytes=0,
                   seats_done=0, host_ok=True)[0] == "sleep"
    action, _ = st.tick(pressure=0, p_committed=0, free_min=PP0_FREE, p_grant_bytes=0, d_air_bytes=0, seats_done=1)
    assert action == "wake"                                     # front 19442-44: wake before the sleep leg moved

    async def quiesce(group):
        return False, "HTTP 400 Cache not flushed ... not-idle because: last_batch, pp_microbatches"
    f = types.SimpleNamespace(_dual_p_sleep_n=0, groups={"P": object()}, quiesce=quiesce,
                              counters=collections.Counter(), _dual_stages_obj=st)
    asyncio.run(types.MethodType(FR.Front._dual_p_sleep, f)())
    assert st.p_state == "stopped" and f.counters["dual_kv_pressure_sleep_not_idle"] == 1
    # ... and the wake the front issued 1.5 s earlier lands on P, which never slept
    rk = _ranks()
    for x in rk:
        x.wake(monkeypatch, ENV_ON)
        assert x.calls == ["seam"], x.calls
    with pytest.raises(PK.Weg2DualPWakeShort):                  # OFF: still the b9p death on PP0
        _ranks()[0].wake(monkeypatch, {PK.WAKE_VOTE_ENV: "0"})


def test_on_asleep_wake_is_untouched(monkeypatch):
    """Danger (a): a P that IS asleep (tags paused) gets the old wake -- PP0 still raises its honest
    WAKE-SHORT when D really sits in the loan; no real wake is swallowed."""
    rk = _ranks(offload=WAKE_TAGS)
    with pytest.raises(PK.Weg2DualPWakeShort):
        rk[0].wake(monkeypatch, ENV_ON)
    assert rk[0].calls == ["seam", "leg_failed:Weg2DualPWakeShort"]
    for x in rk:   # decision itself, per rank: partly paused counts as asleep too
        monkeypatch.setenv("SGLANG_WEG2_DUAL_LAYOUT", "1")
        monkeypatch.setenv("SGLANG_WEG2_GROUP", "P")
        monkeypatch.setenv(PK.WAKE_VOTE_ENV, "1")
        assert PK.wake_without_sleep(x.fake.scheduler, {"kv_cache"}, WAKE_TAGS) is False
        assert PK.wake_without_sleep(x.fake.scheduler, set(), WAKE_TAGS) is True


def test_decision_is_rank_uniform_and_reads_no_rank_local_state(monkeypatch):
    """Danger (b): the answer depends on the request + offload_tags only. Same inputs -> same answer for
    every loan/pool state (the b9p split), and the function's code touches no ledger, clock or loan."""
    monkeypatch.setenv("SGLANG_WEG2_DUAL_LAYOUT", "1")
    monkeypatch.setenv("SGLANG_WEG2_GROUP", "P")
    monkeypatch.setenv(PK.WAKE_VOTE_ENV, "1")
    for offload, want in ((set(), True), ({"kv_cache"}, False)):
        answers = {PK.wake_without_sleep(x.fake.scheduler, offload, WAKE_TAGS) for x in _ranks()}
        assert answers == {want}
    no_actor = types.SimpleNamespace()                           # a rank with NO stage actor decides the same
    assert PK.wake_without_sleep(no_actor, set(), WAKE_TAGS) is True
    fn = ast.parse(inspect.getsource(PK.wake_without_sleep))
    body = fn.body[0].body[1:]                                   # drop the docstring
    names = {n.id for s in body for n in ast.walk(s) if isinstance(n, ast.Name)}
    attrs = {n.attr for s in body for n in ast.walk(s) if isinstance(n, ast.Attribute)}
    assert not ({"time", "monotonic", "ledger", "lent_bytes", "phys_free_bytes"} & (names | attrs))
    assert "state" not in attrs and "reclaim" not in attrs


@pytest.mark.parametrize("env", [
    {}, {PK.WAKE_VOTE_ENV: "1"},                                                          # flip form / env alone
    {PK.WAKE_VOTE_ENV: "1", "SGLANG_WEG2_GROUP": "P"},                                    # no dual layout
    {PK.WAKE_VOTE_ENV: "1", "SGLANG_WEG2_DUAL_LAYOUT": "1"},                              # no group
    {PK.WAKE_VOTE_ENV: "1", "SGLANG_WEG2_DUAL_LAYOUT": "1", "SGLANG_WEG2_GROUP": "D"},    # D never
    {PK.WAKE_VOTE_ENV: "1", "SGLANG_WEG2_DUAL_LAYOUT": "0", "SGLANG_WEG2_GROUP": "P"},
    {"SGLANG_WEG2_DUAL_LAYOUT": "1", "SGLANG_WEG2_GROUP": "P"},                           # default OFF in dual P
    {"SGLANG_WEG2_DUAL_LAYOUT": "1", "SGLANG_WEG2_GROUP": "P", PK.WAKE_VOTE_ENV: "0"},
])
def test_flip_and_off_unchanged(monkeypatch, env):
    """Danger (c) + 'Flip unveraendert': outside dual P with the env on, and in dual P with it off, the rule
    never fires (the wake handler runs its old body), reads no scheduler attribute, writes nothing."""
    for k, v in env.items():
        monkeypatch.setenv(k, v)
    assert PK.wake_vote_armed() is False
    bare = types.SimpleNamespace()                               # any attribute read would raise
    assert PK.wake_without_sleep(bare, set(), WAKE_TAGS) is False
    assert vars(bare) == {}
    from sglang.srt.environ import envs
    assert envs.SGLANG_WEG2_DUAL_WAKE_VOTE.get() is (env.get(PK.WAKE_VOTE_ENV) == "1")
    if env.get(PK.WAKE_VOTE_ENV) != "1" or "SGLANG_WEG2_GROUP" not in env or env["SGLANG_WEG2_GROUP"] != "P":
        # the handler hands over to the old body: wake_reclaim is reached (a flip P rank has no stage actor -> 0)
        rk = _Rank(1)
        rk.fake.scheduler = bare
        try:
            rk.wake(monkeypatch, {})
        except Exception:    # noqa: BLE001 -- the old body runs on past the fake; only the hand-over matters
            pass
        assert rk.calls[:1] == ["seam"] and "fence" not in rk.calls[:1]


def test_source_order_decision_precedes_wake_reclaim():
    """The no-op decision sits after the replay check and BEFORE wake_reclaim and the fence."""
    src = inspect.getsource(WU.SchedulerWeightUpdaterManager.resume_memory_occupation)
    i_replay = src.index("replay = self._weg2_leg_replay(\"resume\"")
    i_vote = src.index("wake_without_sleep(")
    i_reclaim = src.index("_dpk_wake.wake_reclaim(")
    i_fence = src.index("_weg2_group_fence(")
    assert i_replay < i_vote < i_reclaim < i_fence
