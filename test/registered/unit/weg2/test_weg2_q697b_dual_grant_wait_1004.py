# SPDX-License-Identifier: Apache-2.0
"""Q-697b DUAL GRANT-WAIT (27B NVFP4 dual, desk 1010 section 2.4: weg2-0-44 waited 181-184 s in an
abort / resend loop).

The loop: a leg whose card grant is short is HELD in P's waiting queue (``_dual_kv_wait``); P releases its
mapping only at FULL idle (``Scheduler.on_idle`` -> ``is_fully_idle`` wants an empty queue); under D
pressure the front therefore pauses (aborts) EVERY leg in flight, grant-less ones included, and the
aborted leg waits for RESUME (all zeros), is resent, granted, paused again.

Three parts, and the first draft that had only the front part DEADLOCKED (the waiter keeps P from full
idle, P never releases, D waits for P's answer). Every test below is named for what it pins:

  1. RELEASE    P rank, queue = only grant-waiting legs, nothing else live -> the normal idle release
  2. FRONT      a leg named in PP0's marker is not paused (not while P's stage is 'sleeping')
  3. GRANT HOLD PP0 takes no grant while a card shows D's demand / pressure on P (bounded)

RED on c2f6e819cb: the module is missing and, behaviourally, ``Scheduler.on_idle`` releases nothing with a
waiter queued, ``_dual_pause_inflight`` pauses the waiter, ``pp0_grant`` grants into D's demand.
GREEN with the fix. FLIP UNCHANGED: no scheduler attribute, file or ledger is read off the dual P layout.
"""
from __future__ import annotations

import asyncio
import json
import os
import tempfile
import time
import types
import uuid

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

import pytest  # noqa: E402

from sglang.srt.managers import scheduler as SC  # noqa: E402
from sglang.srt.managers import weg2_store_told as ST  # noqa: E402
from sglang.srt.weg2 import card_kv_ledger as K  # noqa: E402
from sglang.srt.weg2 import dual_p_kv_stage as S  # noqa: E402
from sglang.srt.weg2 import front as F  # noqa: E402
from sglang.test.ci.ci_register import register_cpu_ci  # noqa: E402

register_cpu_ci(est_time=8, suite="base-a-test-cpu")

MIB = 1 << 20
PORT = 31337
STATE_DIR_ENV = "SGLANG_WEG2_P_READ_STATE_DIR"
DUAL = {"SGLANG_WEG2_DUAL_LAYOUT": "1", "SGLANG_WEG2_GROUP": "P"}


def _flip_envs():
    return [{}, {"SGLANG_WEG2_GROUP": "P"}, {"SGLANG_WEG2_DUAL_LAYOUT": "1"},
            {"SGLANG_WEG2_DUAL_LAYOUT": "1", "SGLANG_WEG2_GROUP": "D"},
            {"SGLANG_WEG2_DUAL_LAYOUT": "0", "SGLANG_WEG2_GROUP": "P"}]


class Boom:
    """Any attribute read raises: proves a gate returned before touching the object."""

    def __getattr__(self, name):
        raise AssertionError("read %s off the dual P layout" % name)


@pytest.fixture(autouse=True)
def rig(monkeypatch, tmp_path):
    """Dual P env, a private state dir (the marker), a private idle-stamp tag."""
    monkeypatch.setenv(STATE_DIR_ENV, str(tmp_path))
    for k, v in DUAL.items():
        monkeypatch.setenv(k, v)
    monkeypatch.delenv("SGLANG_WEG2_DUAL_GRANT_WAIT", raising=False)
    monkeypatch.delenv("SGLANG_WEG2_DUAL_GRANT_HOLD_S", raising=False)
    tag = "q697b-%s" % uuid.uuid4().hex[:8]
    monkeypatch.setenv("SGLANG_WEG2_DUAL_KV_TAG", tag)
    S._reset_wait_log()
    try:
        from sglang.srt.weg2 import dual_grant_wait as G

        G._reset_for_tests()
    except ImportError:  # RED on the base: the module does not exist
        pass
    yield tmp_path
    try:
        os.unlink(S._idle_marker(tag))
    except OSError:
        pass


def _marker_path(tmp_path):
    return os.path.join(str(tmp_path), "weg2_p_grantwait_%d.json" % PORT)


def _write_marker(tmp_path, rids, t=None):
    """PP0's marker, written by hand (the format is the contract; no import needed)."""
    with open(_marker_path(tmp_path), "w") as f:
        json.dump({"rids": sorted(rids), "pid": os.getpid(), "t": time.time() if t is None else t}, f)


# ------------------------------------------------------------------------------------------ the rank

class _Tree:
    def __init__(self):
        self.ongoing_load_back = {}
        self.ongoing_write_through = {}
        self.ongoing_prefetch = {}
        self.ongoing_backup = {}
        self.enable_storage = False

    def flush_write_through_acks(self):
        pass

    def evictable_size(self):
        return 0

    def protected_size(self):
        return 0

    def evict(self, params):
        pass


class _Actor:
    """P's KV actor as far as the idle release reads it, on a real card ledger."""

    def __init__(self, ledger, nbytes):
        self.ledger = ledger
        self.mapped_tokens = 8192
        self._committed = nbytes
        self._phys_next = float("inf")           # no NVML read in a unit test
        self.released = []

    def release_all(self):
        n = self._committed
        self.released.append(n)
        self.ledger.release(n)
        self.mapped_tokens = 0
        self._committed = 0
        return n


def _card(root, name, p_committed, d_short):
    """One card ledger: P holds ``p_committed`` bytes of leftover mapping, D is ``d_short`` bytes short
    (that is the pressure on P, and D's unmet demand)."""
    path = os.path.join(root, name)
    d, p = K.CardKvLedger(path, "D"), K.CardKvLedger(path, "P")
    d.contribute(1000 * MIB, committed=0)
    p.contribute(0)
    p.request(p_committed)
    d.request(1000 * MIB - p_committed + d_short)       # D takes the rest and asks for more: pressure on P
    return path, d, p


def _req(rid, waits=False):
    return types.SimpleNamespace(rid=rid, _dual_kv_wait=waits, origin_input_ids=list(range(100)))


def make_rank(pp_rank, queue, held_waiters=(), ledger=None, p_bytes=0, running=False, chunked=None,
              port=PORT):
    """A P rank the real ``Scheduler.on_idle`` / ``is_fully_idle`` run on."""
    from sglang.srt.disaggregation.utils import DisaggregationMode

    ns = types.SimpleNamespace(
        ps=types.SimpleNamespace(pp_rank=pp_rank, pp_size=3), server_args=types.SimpleNamespace(port=port),
        waiting_queue=list(queue), _weg2_store_held={r.rid: r for r in queue if r.rid in set(held_waiters)},
        running_batch=types.SimpleNamespace(is_empty=lambda: not running), chunked_req=chunked,
        anchor_tails=None, dllm_manager=types.SimpleNamespace(any_staging_reqs=lambda: False),
        last_batch=None, enable_overlap=False, result_queue=[], running_mbs=[], mbs=[],
        kv_session_offload=None, grammar_manager=types.SimpleNamespace(grammar_queue=[]),
        disaggregation_mode=DisaggregationMode.NULL, enable_hisparse=False, enable_hierarchical_cache=False,
        tree_cache=_Tree(), idle_sleeper=None, idle_census=None, forward_ct=1,
        _pending_chunked_abort_req=None)
    ns.is_fully_idle = types.MethodType(SC.Scheduler.is_fully_idle, ns)
    ns._pp_microbatches_drained = types.MethodType(SC.Scheduler._pp_microbatches_drained, ns)
    ns.actor = _Actor(ledger, p_bytes) if ledger is not None else None
    return ns


@pytest.fixture(autouse=True)
def actor_of(monkeypatch):
    monkeypatch.setattr(S, "_actor", lambda sched: getattr(sched, "actor", None))
    monkeypatch.setattr(SC.Scheduler, "_idle_census_control_hold", lambda self: False, raising=False)


def on_idle(rank):
    SC.Scheduler.on_idle(rank)


def committed_p(path):
    return K.peek(path).committed["P"]


# ------------------------------------------------------------------------------------------ 1. RELEASE

def test_the_deadlock_case_a_queue_of_only_grant_waiters_still_releases_p(rig):
    """THE NAMED CASE (the discarded first draft): the front no longer pauses the grant-less leg, P's queue
    is not empty, D waits for P's answer. Without the release P keeps the whole mapping on every card for
    ever. PP0 knows its waiter; PP1/PP2 learn it from PP0's marker."""
    W = "weg2-0-44"
    cards = [_card(str(rig), "card%d" % i, 300 * MIB, 150 * MIB) for i in range(3)]
    for path, _d, _p in cards:
        assert K.peek(path).pressure["P"] > 0, "precondition: D presses P on every card"
    _write_marker(rig, [W])
    ranks = [make_rank(i, [_req(W, True)], held_waiters=[W], ledger=cards[i][2], p_bytes=300 * MIB)
             for i in range(3)]
    for r in ranks:
        on_idle(r)
    for i, (path, _d, _p) in enumerate(cards):
        assert committed_p(path) == 0, "PP%d kept %d B for a leg that holds nothing -- D waits for ever" % (
            i, committed_p(path))
        assert K.peek(path).pressure["P"] == 0, "D's pressure outlived P's mapping"
        assert ranks[i].actor.mapped_tokens == 0
        assert [r.rid for r in ranks[i].waiting_queue] == [W], "the waiter itself is never dropped"


def test_pp0_stamps_idle_for_the_followers_held_aborts(rig):
    """The followers apply a held abort once PP0 has been idle since (dual20 stamp). A PP0 whose queue is
    only waiters never reached the stamp; the release path writes it (else the followers' chunked_req of
    the paused leg would block THEIR idle for ever)."""
    W = "weg2-0-44"
    path, _d, p = _card(str(rig), "card0", 100 * MIB, 50 * MIB)
    pp0 = make_rank(0, [_req(W, True)], held_waiters=[W], ledger=p, p_bytes=100 * MIB)
    assert not os.path.exists(S._idle_marker(os.environ["SGLANG_WEG2_DUAL_KV_TAG"]))
    on_idle(pp0)
    parts = open(S._idle_marker(os.environ["SGLANG_WEG2_DUAL_KV_TAG"])).read().split()
    assert len(parts) == 2 and int(parts[1]) == 1


def test_a_live_leg_in_the_queue_keeps_the_mapping(rig):
    """DANGER DIRECTION: a leg PP0 has granted (not a waiter) is queued next to the waiter -- it needs the
    mapping, nothing is released."""
    W, LIVE = "weg2-0-44", "weg2-0-45"
    path, _d, p = _card(str(rig), "card0", 100 * MIB, 50 * MIB)
    _write_marker(rig, [W])
    for pp in (0, 1):
        r = make_rank(pp, [_req(W, True), _req(LIVE, False)], held_waiters=[W], ledger=p, p_bytes=100 * MIB)
        on_idle(r)
        assert committed_p(path) == 100 * MIB and r.actor.mapped_tokens == 8192


@pytest.mark.parametrize("busy", ["running", "chunked"])
def test_anything_else_live_keeps_the_mapping(rig, busy):
    W = "weg2-0-44"
    path, _d, p = _card(str(rig), "card0", 100 * MIB, 50 * MIB)
    kw = {"running": True} if busy == "running" else {"chunked": object()}
    r = make_rank(0, [_req(W, True)], held_waiters=[W], ledger=p, p_bytes=100 * MIB, **kw)
    on_idle(r)
    assert committed_p(path) == 100 * MIB, "released under a running / chunked request"


def test_a_follower_without_a_fresh_marker_behaves_as_before(rig):
    """No marker, an old marker, another rid: the follower names no waiter and keeps its mapping."""
    W = "weg2-0-44"
    path, _d, p = _card(str(rig), "card0", 100 * MIB, 50 * MIB)
    for how in ("missing", "stale", "other_rid"):
        if how == "stale":
            _write_marker(rig, [W], t=time.time() - 60.0)
        elif how == "other_rid":
            _write_marker(rig, ["weg2-0-99"])
        r = make_rank(1, [_req(W, True)], ledger=p, p_bytes=100 * MIB)
        on_idle(r)
        assert committed_p(path) == 100 * MIB, how


def test_the_masked_idle_check_always_puts_the_queue_back(rig):
    from sglang.srt.weg2 import dual_grant_wait as G

    W = "weg2-0-44"
    r = make_rank(0, [_req(W, True)], held_waiters=[W])
    q = r.waiting_queue

    def boom(*a, **k):
        raise RuntimeError("is_fully_idle raised")

    r.is_fully_idle = boom
    with pytest.raises(RuntimeError):
        G._idle_but_queue(r)
    assert r.waiting_queue is q and len(q) == 1


def test_the_release_is_wired_into_on_idle_after_the_idle_gate():
    import inspect

    src = inspect.getsource(SC.Scheduler.on_idle)
    i = src.index("if not self.is_fully_idle():")
    j = src.index("release_for_grant_waiters(self)")
    k = src.index("self.idle_sleeper.reset()")
    assert i < j < k


def test_the_switch_off_is_the_pre_q697b_behaviour(rig, monkeypatch):
    monkeypatch.setenv("SGLANG_WEG2_DUAL_GRANT_WAIT", "0")
    W = "weg2-0-44"
    path, _d, p = _card(str(rig), "card0", 100 * MIB, 50 * MIB)
    r = make_rank(0, [_req(W, True)], held_waiters=[W], ledger=p, p_bytes=100 * MIB)
    on_idle(r)
    assert committed_p(path) == 100 * MIB


# ------------------------------------------------------------------------------------------ the marker

def _pp0(rids_held):
    held = {r: _req(r, True) for r in rids_held}
    return types.SimpleNamespace(ps=types.SimpleNamespace(pp_rank=0, pp_size=3),
                                 server_args=types.SimpleNamespace(port=PORT), _weg2_store_held=held,
                                 waiting_queue=list(held.values()))


def test_marker_roundtrip_heartbeat_and_clear(rig):
    from sglang.srt.weg2 import dual_grant_wait as G

    pp0 = _pp0(["weg2-0-44"])
    t0 = 1000.0
    assert G.publish(pp0, ["weg2-0-44"], now=t0) is True
    rd = G.GrantWaitState(PORT)
    assert rd.rids(now=t0 + 0.1) == frozenset({"weg2-0-44"})
    assert G.publish(pp0, ["weg2-0-44"], now=t0 + 0.2) is False, "unchanged within the heartbeat: no write"
    assert G.publish(pp0, ["weg2-0-44"], now=t0 + 1.5) is True, "heartbeat while waiters stand"
    assert rd.rids(now=t0 + 1.6) == frozenset({"weg2-0-44"})
    assert rd.rids(now=t0 + 1.5 + G.FRESH_S + 1) == frozenset(), "a PP0 that stopped writing names no waiter"
    assert G.publish(pp0, ["weg2-0-44", "weg2-0-50"], now=t0 + 10) is True
    assert G.publish(pp0, (), now=t0 + 11) is True, "the set became empty: cleared once"
    assert rd.rids(now=t0 + 11.1) == frozenset()
    assert G.publish(pp0, (), now=t0 + 12) is False and G.publish(pp0, (), now=t0 + 20) is False


def test_marker_is_written_by_pp0_only(rig):
    from sglang.srt.weg2 import dual_grant_wait as G

    f = _pp0(["weg2-0-44"])
    f.ps.pp_rank = 1
    assert G.publish(f, ["weg2-0-44"]) is False
    assert not os.path.exists(_marker_path(rig))


def test_dual_kv_retry_publishes_what_is_still_waiting_after_its_retries(rig, monkeypatch):
    from sglang.srt.weg2 import dual_grant_wait as G

    a, b = _req("weg2-0-44", True), _req("weg2-0-45", True)
    sched = types.SimpleNamespace(ps=types.SimpleNamespace(pp_rank=0, pp_size=3),
                                  server_args=types.SimpleNamespace(port=PORT),
                                  _weg2_store_held={a.rid: a, b.rid: b},
                                  _prefetch_kvcache=lambda req: "issued")
    monkeypatch.setattr(S, "pp0_grant", lambda s, req: 1 if req.rid == "weg2-0-45" else 0)
    ST._dual_kv_retry(sched)
    assert G.GrantWaitState(PORT).rids() == frozenset({"weg2-0-44"}), "45 was granted: not a waiter any more"
    assert a._dual_kv_wait is True and b._dual_kv_wait is False
    monkeypatch.setattr(S, "pp0_grant", lambda s, req: 1)
    ST._dual_kv_retry(sched)
    ST._dual_kv_retry(sched)
    assert G.GrantWaitState(PORT).rids() == frozenset(), "all granted: the marker is cleared"


# ------------------------------------------------------------------------------------------ 2. FRONT

def _front(dual=True):
    f = F.Front(prefill="http://127.0.0.1:%d" % PORT, decode="http://127.0.0.1:%d" % (PORT + 1), awake="D",
                tag="q697b", store_dir="/tmp", prefill_sid=0, decode_sid=0, dc_reserve={}, w_s=45.0,
                weight_chunks=2, flip_min_work_tokens=1, dual_layout=dual)
    f.aborted = []

    async def rpc(g, path, body, timeout):
        f.aborted.append(body["rid"])
        return 200, b""

    f.rpc = rpc
    return f


def _inflight(f, rids):
    loop = asyncio.get_event_loop()
    for rid in rids:
        f._dual_inflight[rid] = F.Pending(rid, "/generate", {}, "x", time.time(), loop.create_future(),
                                          est_prompt=100, est_uncached=100)


def _front_with(rids):
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    f = _front()
    _inflight(f, rids)
    return f, loop


def test_the_front_does_not_pause_a_leg_pp0_holds_for_its_grant(rig):
    """RED on the base: both legs are paused (aborted on P) -- including weg2-0-44, which held nothing since
    'PP0 WAIT ... nothing held' (22:04:43 -> P-PAUSE 22:05:38)."""
    f, loop = _front_with(["weg2-0-43", "weg2-0-44"])
    _write_marker(rig, ["weg2-0-44"])
    try:
        async def run():
            f._dual_pause_inflight(360710144)
            await asyncio.sleep(0.05)

        loop.run_until_complete(run())
    finally:
        loop.close()
        asyncio.set_event_loop(asyncio.new_event_loop())
    paused = sorted(r for r, p in f._dual_inflight.items() if p.dual_pause)
    assert paused == ["weg2-0-43"], "the grant-less leg was paused again: %s" % paused
    assert f.aborted == ["weg2-0-43"], "only the leg holding KV is aborted on P: %s" % f.aborted


@pytest.mark.parametrize("how", ["no_marker", "stale", "sleeping"])
def test_the_front_pauses_everything_when_it_must(rig, how):
    """No fresh marker -> as before. P's stage 'sleeping' -> as before (stage 2 needs P quiescent)."""
    f, loop = _front_with(["weg2-0-43", "weg2-0-44"])
    if how == "stale":
        _write_marker(rig, ["weg2-0-44"], t=time.time() - 60.0)
    else:
        _write_marker(rig, ["weg2-0-44"])
    if how == "no_marker":
        os.unlink(_marker_path(rig))
    if how == "sleeping":
        f._dual_stages().p_state = "sleeping"
    try:
        async def run():
            f._dual_pause_inflight(360710144)
            await asyncio.sleep(0.05)

        loop.run_until_complete(run())
    finally:
        loop.close()
        asyncio.set_event_loop(asyncio.new_event_loop())
    assert sorted(f.aborted) == ["weg2-0-43", "weg2-0-44"]


def test_the_front_hook_is_wired_into_the_pause():
    import inspect

    src = inspect.getsource(F.Front._dual_pause_inflight)
    assert "front_skip_set(self)" in src and src.index("front_skip_set") < src.index("p.dual_pause = True")


# ------------------------------------------------------------------------------------------ 3. GRANT HOLD

class _Pp0:
    """The q670 fixture: three temp cards, stage files, D holds all but 96 MiB and asked for more."""

    def __init__(self, monkeypatch, root, p_holds=True):
        self.p_holds = p_holds
        self.paths = []
        for r, per in enumerate((2048, 4096, 6144)):
            pth = os.path.join(root, "card%d" % r)
            d, p = K.CardKvLedger(pth, "D"), K.CardKvLedger(pth, "P")
            d.contribute(4096 * MIB, committed=0)
            p.contribute(0)
            self.paths.append(pth)
            with open(os.path.join(root, "stage%d" % r), "w") as f:
                json.dump({"ledger": pth, "step": 4096, "top": 196608,
                           "bytes": [k * 4096 * per for k in range(196608 // 4096 + 1)]}, f)
        monkeypatch.setattr(S, "stage_file", lambda tag, r, root_=root: os.path.join(root_, "stage%d" % r))
        monkeypatch.setattr(S, "_actor", lambda sched: types.SimpleNamespace(
            page=64, _committed=0, map_granted=lambda lvl, charged=None: None))

    def d_short_with_room(self):
        """D took the whole card, asked for 500 MiB more (demand stands), then gave 2000 MiB back: the
        card HAS room for a small P grant while D's demand still stands."""
        for pth in self.paths:
            d = K.CardKvLedger(pth, "D")
            if self.p_holds:
                K.CardKvLedger(pth, "P").request(50 * MIB)       # P's leftover mapping (the pressure episode)
            d.request(4096 * MIB)
            d.request(500 * MIB)
            d.release(2000 * MIB)
            assert K.peek(pth).demand["D"] > 0 and K.peek(pth).free >= 2000 * MIB
            assert (K.peek(pth).committed["P"] > 0) == self.p_holds

    def sched(self):
        return types.SimpleNamespace(ps=types.SimpleNamespace(pp_rank=0, pp_size=3), waiting_queue=[],
                                     _weg2_store_held={})

    @staticmethod
    def req(rid, tokens=1000):
        return types.SimpleNamespace(rid=rid, origin_input_ids=list(range(tokens)), _dual_grant_untold=None)


def test_pp0_gives_no_grant_while_d_demand_stands_and_grants_once_it_is_gone(rig, monkeypatch):
    """RED on the base: the waiter takes the bytes P just released before D's tick took them."""
    fx = _Pp0(monkeypatch, str(rig))
    fx.d_short_with_room()
    sched, req = fx.sched(), fx.req("weg2-0-44")
    assert S.pp0_grant(sched, req) == 0, "granted into D's unmet demand"
    for pth in fx.paths:
        assert K.peek(pth).committed["P"] == 50 * MIB, "a refused grant left bytes taken"
    for pth in fx.paths:                       # D fits: its request is granted in full, demand 0
        K.CardKvLedger(pth, "D").request(10 * MIB)
        assert K.peek(pth).demand["D"] == 0
    assert S.pp0_grant(sched, req) > 0


def test_the_grant_hold_is_bounded(rig, monkeypatch):
    """A stale ledger never wedges P: past SGLANG_WEG2_DUAL_GRANT_HOLD_S the grant is tried."""
    monkeypatch.setenv("SGLANG_WEG2_DUAL_GRANT_HOLD_S", "0.2")
    fx = _Pp0(monkeypatch, str(rig))
    fx.d_short_with_room()
    sched, req = fx.sched(), fx.req("weg2-0-44")
    assert S.pp0_grant(sched, req) == 0
    time.sleep(0.3)
    assert S.pp0_grant(sched, req) > 0


def test_the_hold_is_named_once_per_stretch(rig, monkeypatch, caplog):
    import logging

    fx = _Pp0(monkeypatch, str(rig))
    fx.d_short_with_room()
    with caplog.at_level(logging.INFO):
        S.pp0_grant(fx.sched(), fx.req("weg2-0-44"))
    assert any("Q-697b DUAL GRANT-HOLD rid=weg2-0-44" in m for m in caplog.messages), caplog.messages


# ------------------------------------------------------------------------------------------ follower guard

def test_a_held_abort_whose_rid_pp0_holds_again_is_not_applied(rig):
    """The hold's verdict reads the rid, not the object: applied now it would take the NEW instance (the front
    sent the rid again and PP0 holds it for its grant) with it -- the Q-697 zombie class."""
    f = types.SimpleNamespace(ps=types.SimpleNamespace(pp_rank=1), server_args=types.SimpleNamespace(port=PORT),
                              _pending_chunked_abort_req=None, _weg2_pending_waiting_aborts={"weg2-0-44": [None, 0]},
                              forward_ct=5, _pp_microbatches_drained=lambda: True,
                              process_pending_chunked_abort=lambda: setattr(f, "applied", True))
    pp0 = types.SimpleNamespace(ps=types.SimpleNamespace(pp_rank=0))
    _write_marker(rig, ["weg2-0-44"])
    assert S.follower_release_aborted_chunk(f, now=100.0) is False      # sees the abort
    S.mark_pp0_idle(pp0, now=102.0)                                     # PP0 idle after it
    assert S.follower_release_aborted_chunk(f, now=103.0) is False
    assert not getattr(f, "applied", False), "the hold was applied against a fresh waiting instance"
    os.unlink(_marker_path(rig))                                  # the waiter is granted: no longer named
    assert S.follower_release_aborted_chunk(f, now=104.0) is True and f.applied


def test_r5_the_guard_holds_only_the_conflicting_rids_back(rig):
    """A chunked abort of ANOTHER rid and a hold whose rid is not a waiter proceed; only the conflicting
    hold stays held (and is still there afterwards)."""
    seen = {}
    f = types.SimpleNamespace(ps=types.SimpleNamespace(pp_rank=1), server_args=types.SimpleNamespace(port=PORT),
                              _pending_chunked_abort_req=types.SimpleNamespace(rid="weg2-0-30"),
                              _weg2_pending_waiting_aborts={"weg2-0-44": [None, 0], "weg2-0-50": [None, 0]},
                              forward_ct=5, _pp_microbatches_drained=lambda: True)

    def process():
        seen["pend"] = sorted(f._weg2_pending_waiting_aborts)
        f._pending_chunked_abort_req = None

    f.process_pending_chunked_abort = process
    pp0 = types.SimpleNamespace(ps=types.SimpleNamespace(pp_rank=0))
    _write_marker(rig, ["weg2-0-44"])
    S.follower_release_aborted_chunk(f, now=100.0)
    S.mark_pp0_idle(pp0, now=102.0)
    assert S.follower_release_aborted_chunk(f, now=103.0) is True, "the chunked abort of another rid was held back"
    assert seen["pend"] == ["weg2-0-50"], "the conflicting hold was handed to the verdict: %s" % seen
    assert sorted(f._weg2_pending_waiting_aborts) == ["weg2-0-44", "weg2-0-50"], "the held abort was lost"


# ------------------------------------------------------------------------------------------ FLIP UNCHANGED

@pytest.mark.parametrize("env", _flip_envs())
def test_flip_unchanged_the_release_reads_nothing(env, rig, monkeypatch):
    from sglang.srt.weg2 import dual_grant_wait as G

    for k in DUAL:
        monkeypatch.delenv(k, raising=False)
    assert G.release_for_grant_waiters(Boom(), env) == 0


@pytest.mark.parametrize("env", _flip_envs())
def test_flip_unchanged_publish_writes_nothing_and_a_never_publishing_process_returns_first(env, rig):
    from sglang.srt.weg2 import dual_grant_wait as G

    G._reset_for_tests()
    assert G.publish(Boom(), (), env=env) is False                  # empty + never published: not even the gate
    assert G.publish(Boom(), ["weg2-0-44"], env=env) is False        # a waiter set off the gate: still nothing read
    assert not os.path.exists(_marker_path(rig)) and os.listdir(str(rig)) == []


@pytest.mark.parametrize("env", _flip_envs())
def test_flip_unchanged_grant_hold_reads_no_ledger(env):
    from sglang.srt.weg2 import dual_grant_wait as G

    assert G.grant_held_by_d(Boom(), [Boom()], env=env) is None


def test_flip_unchanged_front_and_follower_read_no_marker(rig, monkeypatch):
    from sglang.srt.weg2 import dual_grant_wait as G

    _write_marker(rig, ["weg2-0-44"])
    f = _front(dual=False)
    assert G.front_skip_set(f) == frozenset()
    assert not hasattr(f, "_dual_gw_state"), "the flip front opened the marker"
    for k in DUAL:
        monkeypatch.delenv(k, raising=False)
    assert G.same_rid_waits(Boom(), ["weg2-0-44"]) is False


@pytest.mark.parametrize("env", _flip_envs()[:4])
def test_flip_unchanged_on_idle_releases_nothing_with_a_waiter_queued(env, rig, monkeypatch):
    """Behavioural: the real on_idle with a marker naming the queued leg -- off the dual P layout the
    mapping stays exactly as on c2f6e819cb."""
    for k in DUAL:
        monkeypatch.delenv(k, raising=False)
    for k, v in env.items():
        monkeypatch.setenv(k, v)
    W = "weg2-0-44"
    path, _d, p = _card(str(rig), "card0", 100 * MIB, 50 * MIB)
    _write_marker(rig, [W])
    for pp in (0, 1):
        r = make_rank(pp, [_req(W, True)], held_waiters=[W], ledger=p, p_bytes=100 * MIB)
        on_idle(r)
        assert committed_p(path) == 100 * MIB and r.actor.mapped_tokens == 8192
    assert not os.path.exists(S._idle_marker(os.environ["SGLANG_WEG2_DUAL_KV_TAG"])), "a flip PP0 stamped idle"


def test_flip_unchanged_pp0_grant_ignores_d_demand_off_the_dual_layout(rig, monkeypatch):
    for k in DUAL:
        monkeypatch.delenv(k, raising=False)
    fx = _Pp0(monkeypatch, str(rig))
    fx.d_short_with_room()
    assert S.pp0_grant(fx.sched(), fx.req("weg2-0-44")) > 0, "the flip-form grant waited for D's demand"


# ------------------------------------------------------------------------------------------ review 1250

def _follower_with_held_abort(rid):
    f = types.SimpleNamespace(ps=types.SimpleNamespace(pp_rank=1), server_args=types.SimpleNamespace(port=PORT),
                              _pending_chunked_abort_req=None, _weg2_pending_waiting_aborts={rid: [None, 0]},
                              forward_ct=5, _pp_microbatches_drained=lambda: True,
                              process_pending_chunked_abort=lambda: setattr(f, "applied", True))
    return f


def test_r2_pp0_names_its_waiters_before_it_stamps_idle(rig):
    """The review probe (/tmp/rev697b/test_rev_same_rid_window.py), made a test. PP0 took the resent rid R in this
    loop iteration (intake -> grant waiter); the marker would name it only at the NEXT pass's _dual_kv_retry.
    The follower holds the abort of the OLD instance of R and reads PP0's stamp: it must not apply it."""
    R = "weg2-0-44"
    path, _d, p = _card(str(rig), "card0", 100 * MIB, 50 * MIB)
    f = _follower_with_held_abort(R)
    S.follower_release_aborted_chunk(f, now=100.0)                  # the follower records the held abort
    assert not os.path.exists(_marker_path(rig)), "precondition: the marker does not name R yet"
    pp0 = make_rank(0, [_req(R, True)], held_waiters=[R], ledger=p, p_bytes=100 * MIB)
    on_idle(pp0)                                                    # real on_idle: the release path stamps idle
    assert os.path.exists(_marker_path(rig)), "PP0 stamped idle without naming its waiter"
    assert S.follower_release_aborted_chunk(f, now=time.time() + 1) is False, \
        "applied the held abort while PP0 holds the same rid as a waiter"
    assert not getattr(f, "applied", False)


def test_r2_the_guard_is_repeated_right_before_the_apply(rig):
    """The marker may appear between the follower's first sight and its apply (the stamp is read, THEN the marker
    changes): the guard sits after the stamp read, not before it."""
    R = "weg2-0-44"
    f = _follower_with_held_abort(R)
    pp0 = types.SimpleNamespace(ps=types.SimpleNamespace(pp_rank=0))
    S.follower_release_aborted_chunk(f, now=100.0)
    S.mark_pp0_idle(pp0, now=102.0)
    from sglang.srt.weg2 import dual_grant_wait as G

    real = G.conflicting_rids

    def late(sched, held):
        _write_marker(rig, [R])                                     # PP0 names R between the stamp read and the apply
        return real(sched, held)

    G.conflicting_rids = late
    try:
        assert S.follower_release_aborted_chunk(f, now=103.0) is False
    finally:
        G.conflicting_rids = real
    assert not getattr(f, "applied", False)


def test_r4_marker_write_failures_are_logged_once(rig, monkeypatch, caplog):
    import logging

    from sglang.srt.weg2 import dual_grant_wait as G

    pp0 = _pp0(["weg2-0-44"])
    monkeypatch.setenv(STATE_DIR_ENV, os.path.join(str(rig), "no", "such", "dir"))
    with caplog.at_level(logging.WARNING):
        for i in range(50):
            G.publish(pp0, ["weg2-0-4%d" % (i % 3)], now=1000.0 + i)
    assert sum("marker write failed" in m for m in caplog.messages) == 1, caplog.messages


def test_r4_the_marker_survives_a_long_pass():
    from sglang.srt.weg2 import dual_grant_wait as G

    assert G.FRESH_S >= 30.0


def test_r1_no_hold_when_p_holds_nothing_and_never_did(rig, monkeypatch):
    """A fresh request under a STALE demand with P idle and empty pays nothing (no 30 s TTFT)."""
    fx = _Pp0(monkeypatch, str(rig), p_holds=False)
    fx.d_short_with_room()
    assert S.pp0_grant(fx.sched(), fx.req("weg2-0-60")) > 0


def test_r1_grace_after_p_released_then_none(rig):
    from sglang.srt.weg2 import dual_grant_wait as G

    fx_root = str(rig)
    paths = []
    for r in range(2):
        pth = os.path.join(fx_root, "g%d" % r)
        K.CardKvLedger(pth, "D").contribute(1000 * MIB, committed=0)
        K.CardKvLedger(pth, "P").contribute(0)
        K.CardKvLedger(pth, "P").request(50 * MIB)
        K.CardKvLedger(pth, "D").request(950 * MIB)
        K.CardKvLedger(pth, "D").request(100 * MIB)               # demand stands
        paths.append(pth)
    stages = [{"ledger": p} for p in paths]
    req = types.SimpleNamespace(rid="weg2-0-61")
    assert G.grant_held_by_d(req, stages, now=100.0) is not None            # P holds bytes
    for pth in paths:
        K.CardKvLedger(pth, "P").release(50 * MIB)                          # P released
    assert G.grant_held_by_d(req, stages, now=101.0) is not None            # within the grace: D's first
    assert G.grant_held_by_d(req, stages, now=100.0 + G.GRACE_S + 1) is None  # stale demand: no wait


def test_r1_the_clock_belongs_to_the_rid_not_the_object(rig, monkeypatch):
    """A resent instance of the same rid does not start a new 30 s; a new episode (no demand) does."""
    from sglang.srt.weg2 import dual_grant_wait as G

    fx = _Pp0(monkeypatch, str(rig))
    fx.d_short_with_room()
    stages = [{"ledger": p} for p in fx.paths]
    a, b = types.SimpleNamespace(rid="weg2-0-62"), types.SimpleNamespace(rid="weg2-0-62")   # two objects, one rid
    assert G.grant_held_by_d(a, stages, now=100.0) is not None
    assert G.grant_held_by_d(b, stages, now=100.0 + G.hold_s() + 1) is None, "the resent instance was held again"
    for pth in fx.paths:                                                    # D fits: the episode is over
        K.CardKvLedger(pth, "D").request(10 * MIB)
    assert G.grant_held_by_d(b, stages, now=200.0) is None
    for pth in fx.paths:                                                    # a new episode
        K.CardKvLedger(pth, "D").request(10000 * MIB)
    assert G.grant_held_by_d(b, stages, now=300.0) is not None


# ------------------------------------------------------------------------------------------ T1 interleavings

def test_t1_release_axis_pp0_and_follower_decide_unequally_and_converge(rig):
    """PP0 and a follower over time: a granted leg LIVE sits next to the waiter W (nobody releases); the front's
    abort of LIVE reaches PP0 first (PP0 releases, the follower still holds LIVE: the safe direction); then the
    follower's queue drains too and it releases."""
    W, LIVE = "weg2-0-44", "weg2-0-45"
    cards = [_card(str(rig), "card%d" % i, 200 * MIB, 100 * MIB) for i in range(2)]
    pp0 = make_rank(0, [_req(W, True), _req(LIVE, False)], held_waiters=[W], ledger=cards[0][2], p_bytes=200 * MIB)
    pp1 = make_rank(1, [_req(W, True), _req(LIVE, False)], ledger=cards[1][2], p_bytes=200 * MIB)
    _write_marker(rig, [W])
    on_idle(pp0), on_idle(pp1)
    assert committed_p(cards[0][0]) == committed_p(cards[1][0]) == 200 * MIB, "released with a live leg queued"
    pp0.waiting_queue = [r for r in pp0.waiting_queue if r.rid == W]          # LIVE aborted at PP0
    on_idle(pp0), on_idle(pp1)
    assert committed_p(cards[0][0]) == 0, "PP0 kept its mapping for a waiter-only queue"
    assert committed_p(cards[1][0]) == 200 * MIB, "the follower released while LIVE was still queued there"
    pp1.waiting_queue = [r for r in pp1.waiting_queue if r.rid == W]          # the abort reaches the follower
    on_idle(pp1)
    assert committed_p(cards[1][0]) == 0
    assert K.peek(cards[0][0]).pressure["P"] == K.peek(cards[1][0]).pressure["P"] == 0


def test_t1_release_axis_the_marker_lags_behind_pp0(rig):
    """The follower's queue holds a rid the marker does not name yet (PP0 took it a moment ago): no release; once PP0
    publishes it, the follower releases."""
    W, NEW = "weg2-0-44", "weg2-0-46"
    path, _d, p = _card(str(rig), "card0", 100 * MIB, 50 * MIB)
    _write_marker(rig, [W])
    pp1 = make_rank(1, [_req(W, True), _req(NEW, True)], ledger=p, p_bytes=100 * MIB)
    on_idle(pp1)
    assert committed_p(path) == 100 * MIB
    _write_marker(rig, [W, NEW])
    on_idle(pp1)
    assert committed_p(path) == 0


def test_t1_front_axis_a_waiter_that_gets_its_grant_is_paused_at_the_next_tick(rig):
    """Front + marker over time: W waits (named): not paused; PP0 grants it and the marker clears: the next pressure
    tick pauses it like any leg holding KV."""
    from sglang.srt.weg2 import dual_grant_wait as G

    f, loop = _front_with(["weg2-0-44"])
    pp0 = _pp0(["weg2-0-44"])
    G.publish(pp0, ["weg2-0-44"])
    try:
        async def tick():
            f._dual_pause_inflight(360710144)
            await asyncio.sleep(0.05)

        loop.run_until_complete(tick())
        assert f.aborted == [] and not any(p.dual_pause for p in f._dual_inflight.values())
        G.publish(pp0, ())                                           # granted: no longer a waiter
        loop.run_until_complete(tick())
    finally:
        loop.close()
        asyncio.set_event_loop(asyncio.new_event_loop())
    assert f.aborted == ["weg2-0-44"]


def test_t1_grant_axis_release_then_d_claims_then_the_waiter_is_granted(rig, monkeypatch):
    """P + D over time on real ledgers: P holds a leftover mapping, D is short; the waiter-only queue releases it
    (P committed 0); the waiter's grant is held for D (grace) while the freed bytes are free; D claims them (demand
    gone); only then the waiter is granted."""
    fx = _Pp0(monkeypatch, str(rig))
    for pth in fx.paths:
        d, p = K.CardKvLedger(pth, "D"), K.CardKvLedger(pth, "P")
        p.request(100 * MIB)
        d.request(4096 * MIB)
        d.request(100 * MIB)                                         # D's demand: exactly what P holds
    W = "weg2-0-44"
    _write_marker(rig, [W])
    sched, req = fx.sched(), fx.req(W)
    monkeypatch.setattr(S, "_actor", lambda s: types.SimpleNamespace(page=64, _committed=0,
                                                                     map_granted=lambda lvl, charged=None: None))
    assert S.pp0_grant(sched, req) == 0, "precondition: the waiter is held while P holds bytes (PP0 retries it every pass)"
    ranks = [make_rank(i, [_req(W, True)], held_waiters=[W], ledger=K.CardKvLedger(fx.paths[i], "P"),
                       p_bytes=100 * MIB) for i in range(3)]
    monkeypatch.setattr(S, "_actor", lambda s: getattr(s, "actor", None))
    for r in ranks:
        on_idle(r)
    for pth in fx.paths:
        assert K.peek(pth).committed["P"] == 0, "P kept its mapping"
        assert K.peek(pth).free >= 100 * MIB
    monkeypatch.setattr(S, "_actor", lambda s: types.SimpleNamespace(page=64, _committed=0,
                                                                     map_granted=lambda lvl, charged=None: None))
    assert S.pp0_grant(sched, req) == 0, "the waiter took the bytes P just released before D did"
    for pth in fx.paths:
        K.CardKvLedger(pth, "D").request(100 * MIB)                  # D's tick claims the freed bytes: demand gone
        assert K.peek(pth).demand["D"] == 0
        K.CardKvLedger(pth, "D").release(2000 * MIB)                 # a D seat ends: room for P
    assert S.pp0_grant(sched, req) > 0


@pytest.mark.parametrize("state,paused", [("serving", False), ("stopped", False), ("lent", False),
                                          ("reclaiming", False), ("sleeping", True)])
def test_t1_every_p_stage_but_sleeping_leaves_the_waiter_unpaused(rig, state, paused):
    f, loop = _front_with(["weg2-0-43", "weg2-0-44"])
    _write_marker(rig, ["weg2-0-44"])
    f._dual_stages().p_state = state
    try:
        async def run():
            f._dual_pause_inflight(360710144)
            await asyncio.sleep(0.05)

        loop.run_until_complete(run())
    finally:
        loop.close()
        asyncio.set_event_loop(asyncio.new_event_loop())
    assert ("weg2-0-44" in f.aborted) is paused, (state, f.aborted)
    assert "weg2-0-43" in f.aborted


# ------------------------------------------------------------------------------------------ R3 (documented)

def test_r3_a_follower_releases_once_its_early_read_terminated(rig):
    """Review R3: with HiCache on, a follower's early read (ongoing_prefetch) keeps the rank not idle; PP0 has no
    read for a waiter and releases at once. The follower follows with delay (the read), not never."""
    W = "weg2-0-44"
    path, _d, p = _card(str(rig), "card0", 100 * MIB, 50 * MIB)
    _write_marker(rig, [W])
    pp1 = make_rank(1, [_req(W, True)], ledger=p, p_bytes=100 * MIB)
    pp1.enable_hierarchical_cache = True
    pp1.tree_cache.enable_storage = True
    pp1.tree_cache.ongoing_prefetch = {W: object()}
    on_idle(pp1)
    assert committed_p(path) == 100 * MIB
    pp1.tree_cache.ongoing_prefetch = {}
    on_idle(pp1)
    assert committed_p(path) == 0
