"""PRIORITY LANES DEFAULT ON (int24f5, user word 13.10.2026: lanes "muss im boot defaultmaessig an sein").

What changed: ``SGLANG_WEG2_LANES`` is ``EnvBool(True)`` in the NF tree's ``environ.py`` (it was ``EnvBool(False)``); ``=0`` is the old path.
The acceptance line of the order: with NO lane request (every request in lane 0: no ``priority``, or 0) the behaviour equals the
int24f4 Dauerstand -- no LANE-* action, no RPC to P or D, no park, routing as with the switch off.

What is pinned here (hermetic: no GPU, loopback aiohttp fakes, every async body under ``asyncio.run``):

1. the default itself: unset -> on, ``=0`` -> off, ``=1`` -> on; the two companion values keep their defaults;
2. lane 0 only, front end to end (the real controller, the real lane loop, fake groups): a D phase with running lane-0 requests, and a
   P phase with a flip -- once with the switch at its DEFAULT, once with ``=0``.  Both: no floor RPC, no lane park, no LANE-* log line, no
   lane counter, ``lane_floor`` 0 / ``lane_epoch`` 0, the same ordered RPC sequence to P and D, the same generate order, every answer 200;
3. the fields: an absent ``priority`` stays absent on the forwarded payload (the groups see the request exactly as without lanes), an explicit
   0 stays 0;
4. the P side told ledger: ``forget_aborted_told`` (fix 5) touches nothing on a rank no lane floor ever reached -- an abort in the Spur-0-only
   run leaves the ledger exactly as the Dauerstand does -- and works as before on a rank that has a lane state;
5. the D side: the lane-aware displacement candidate equals the lane-less one when every request is lane 0.
"""
from __future__ import annotations

import asyncio
import contextlib
import functools
import logging
import os
import random
import re
import sys
from types import SimpleNamespace

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import pytest  # noqa: E402

from sglang.srt.environ import envs  # noqa: E402
from sglang.srt.weg2 import d_park_runtime as dpr  # noqa: E402
from sglang.srt.weg2 import lanes as LN  # noqa: E402
from sglang.srt.weg2 import lanes_p  # noqa: E402
from sglang.srt.weg2 import phase_policy as PP  # noqa: E402
from sglang.srt.weg2 import seat_age as SA  # noqa: E402
from sglang.srt.weg2.front import Front  # noqa: E402
from test_weg2_lanes_l4_1008 import LaneHarness, _front  # noqa: E402
from test_weg2_phase_policy_h91c import _until  # noqa: E402


# ------------------------------------------------------------------ helpers
@contextlib.contextmanager
def _switch(value):
    """``value`` None = the switch is NOT in the environment (the code default decides); else the env value.  Restored on every exit."""
    had = LN.ENV_LANES in os.environ
    old = os.environ.get(LN.ENV_LANES)
    if value is None:
        os.environ.pop(LN.ENV_LANES, None)
    else:
        os.environ[LN.ENV_LANES] = str(value)
    try:
        yield
    finally:
        if had:
            os.environ[LN.ENV_LANES] = old
        else:
            os.environ.pop(LN.ENV_LANES, None)


def _served_front_wrap(fn):
    """The served front wraps the generate handler in ``ipc_out_wrap`` (main()): its finally takes the rid out of the lane register --
    the harness registers the bare handler, so the same wrap is put on the class (as the L4 end-to-end tests do)."""
    @functools.wraps(fn)
    def wrapper(*a, **k):
        orig = Front.handle_generate

        async def wrapped(self, request):
            return await self.ipc_out_wrap(functools.partial(orig, self))(request)

        Front.handle_generate = wrapped
        try:
            return fn(*a, **k)
        finally:
            Front.handle_generate = orig
    return wrapper


START_LINE = "WEG2 LANES on: lane loop started"


def _lane_lines(caplog):
    """Every lane log line EXCEPT the one start line of the lane loop (it names the switch, it is no lane action; with the switch at
    ``=0`` the loop does not start and the line does not exist)."""
    return [r.getMessage() for r in caplog.records if "LANE" in r.getMessage() and not r.getMessage().startswith(START_LINE)]


def _start_lines(caplog):
    return [r.getMessage() for r in caplog.records if r.getMessage().startswith(START_LINE)]


def _lane_counters(front):
    return {k: v for k, v in front.counters.items() if k.startswith("lane") and v}


def _lane_parks(group):
    """The park RPCs a LANE sends: they name rids and/or hold="lane" (the flip's own park body carries neither)."""
    return [b for b in group.park_bodies if b.get("rids") or b.get("hold")]


def _rpc_sequence(group):
    """The ordered RPC paths one fake group received (generate calls are compared separately)."""
    return [e for e in group.timeline if e.startswith("rpc:")]


def _rpc_sequence_without_flip_park(group):
    """As _rpc_sequence, without the flip's own park (``weg2/park_running`` with neither rids nor hold).

    13.10.: whether a D>P flip still finds running decodes on the fake D (and so sends the flip park) is timing of
    the fake group, in both switch settings -- 1 of 3 runs differed in exactly this one entry. A lane park always names
    rids and/or hold="lane" and is asserted empty on its own (_lane_parks), so dropping only the bare flip park here
    keeps the lane-0 comparison exact."""
    seq = _rpc_sequence(group)
    if not any(b.get("rids") or b.get("hold") for b in group.park_bodies):
        seq = [e for e in seq if e != "rpc:weg2/park_running"]
    return seq


# ------------------------------------------------------------------ 1. the default
def test_the_code_default_is_on_and_the_env_decides_when_it_is_set():
    with _switch(None):
        assert envs.SGLANG_WEG2_LANES.get() is True and LN.enabled() is True
    with _switch("0"):
        assert LN.enabled() is False
    with _switch("1"):
        assert LN.enabled() is True


def test_the_companion_values_keep_their_defaults():
    with _switch(None):
        assert envs.SGLANG_WEG2_LANE_KEEPALIVE_S.get() == 15
        assert envs.SGLANG_WEG2_LANE_PREEMPT_CHUNK_TOKENS.get() == 0


def test_the_default_is_declared_on_in_environ_py_and_in_nothing_else():
    """The flip is the one line in ``environ.py`` -- the launcher, the profile registry and the scheduler declare no second default."""
    import inspect

    from sglang.srt import environ as environ_mod

    src = inspect.getsource(environ_mod)
    assert len(re.findall(r"^\s*SGLANG_WEG2_LANES = EnvBool\(True\)\s*$", src, re.M)) == 1
    assert not re.findall(r"SGLANG_WEG2_LANES = EnvBool\(False\)", src)


# ------------------------------------------------------------------ 3. the fields
def test_an_absent_priority_stays_absent_on_the_forwarded_payload_and_an_explicit_zero_stays_zero():
    with _switch(None):
        f = _front("D")
        absent, zero = {"prompt": "x"}, {"prompt": "x", "priority": 0}
        f._lane_note("weg2-0-1", absent)
        f._lane_note("weg2-0-2", zero)
        assert "priority" not in absent and zero["priority"] == 0
        assert f._lane_for("weg2-0-1") == 0 and f._lane_for("weg2-0-2") == 0
        ls = f._lane_state()
        assert (ls.lane_floor, ls.lane_epoch) == (0, 0)


def test_a_front_with_lane_0_only_prints_the_old_flip_begin_tail_and_holds_nothing():
    from sglang.srt.weg2 import lane_ctl as LC

    with _switch(None):
        f = _front("D")
        f.groups["D"].outstanding["weg2-0-1"] = 1.0
        assert LC.flip_begin_note(f, f.groups["D"]) == ""      # "lane_held=K ledger=L" only with a booked lane hold
        assert f._lane_parked_set() == set() and f._lane_held_pendings() == []
        assert f._lane_wake_fields() is not None


# ------------------------------------------------------------------ 2. lane 0 only, end to end
async def _d_phase_scenario(h):
    """Three lane-0 requests decode on D (one with an explicit priority 0), a fourth arrives while they run, then all finish."""
    h.d.hold = {}
    lo = [h.post_lane("lo0"), h.post_lane("lo1", priority=0), h.post_lane("lo2")]
    assert await _until(lambda: len(h.d.running) == 3, 10)
    assert await _until(lambda: len(h.front.groups["D"].outstanding) == 3, 10)
    fourth = h.post_lane("lo3", priority=0)
    assert await _until(lambda: "gen:lo3" in h.d.timeline or len(h.d.running) == 4, 10)
    await asyncio.sleep(0.5)   # the lane loop ticks several times (TICK_S 0.1): nothing may move
    h.d.release_all()
    res = await asyncio.wait_for(asyncio.gather(*lo, fourth), 20)
    return [s for s, _ in res]


def _run_d_phase(switch, caplog):
    out = {}

    async def body():
        async with LaneHarness(awake="D", d_bs=4, tp_prefill_max_tokens=100_000) as h:
            out["status"] = await _d_phase_scenario(h)
            out["d_gen"] = sorted(h.d.gen_marks)
            out["p_gen"] = sorted(h.p.gen_marks)
            out["d_rpc"], out["p_rpc"] = _rpc_sequence(h.d), _rpc_sequence(h.p)
            out["d_park"], out["d_floor"], out["p_floor"] = _lane_parks(h.d), list(h.d.floor_bodies), list(h.p.floor_bodies)
            out["counters"] = _lane_counters(h.front)
            out["floor"] = (h.front._lane_state().lane_floor, h.front._lane_state().lane_epoch) if switch != "0" else None
            out["parked_book"] = dict(h.front._d_parked)
            out["state_keys"] = sorted(k for k in h.front.state_dict() if k.startswith("lane"))
    with _switch(switch):
        caplog.clear()
        caplog.set_level(logging.INFO, logger="weg2.front")
        asyncio.run(body())
    out["lane_lines"] = _lane_lines(caplog)
    out["start_lines"] = _start_lines(caplog)
    return out


@_served_front_wrap
def test_lane_0_only_while_d_decodes_nothing_is_parked_floored_or_logged_and_it_equals_the_switch_off_run(caplog):
    on = _run_d_phase(None, caplog)          # the code default
    off = _run_d_phase("0", caplog)
    for r in (on, off):
        assert r["status"] == [200, 200, 200, 200]
        assert r["d_park"] == [] and r["d_floor"] == [] and r["p_floor"] == []     # no park RPC, no floor RPC
        assert r["counters"] == {} and r["parked_book"] == {}
        assert r["lane_lines"] == [], r["lane_lines"]                               # no LANE-* marker at all
    assert on["floor"] == (0, 0)
    # (the harness starts the lane loop in BOTH runs -- the served front starts it only with the switch on, pinned below)
    # the same work went to the same group, in the same RPC order
    assert (on["d_gen"], on["p_gen"], on["d_rpc"], on["p_rpc"]) == (off["d_gen"], off["p_gen"], off["d_rpc"], off["p_rpc"])
    # the only state difference is the lane block the default now writes (zeros)
    assert off["state_keys"] == [] and set(on["state_keys"]) <= {"lane_epoch", "lane_floor", "lanes"}


async def _p_phase_scenario(h):
    """A P phase: nine lane-0 requests (a mix of absent / explicit 0), the cap of six, then the flip to D and the second phase."""
    h.p.delay = 0.2
    posts = [h.post_lane(f"c{i}", priority=(0 if i % 2 else None), chars=3000) for i in range(9)]
    res = await asyncio.wait_for(asyncio.gather(*posts), 40)
    return [s for s, _ in res]


def _run_p_phase(switch, caplog):
    out = {}

    async def body():
        async with LaneHarness(awake="P", p_concurrency=8, d_bs=8, p_phase_max_requests=6) as h:
            out["status"] = await _p_phase_scenario(h)
            out["p_gen"], out["d_gen"] = sorted(h.p.gen_marks), sorted(h.d.gen_marks)
            first = h.p.timeline.index("rpc:release_memory_occupation") if "rpc:release_memory_occupation" in h.p.timeline else -1
            out["p_before_first_flip"] = len([e for e in h.p.timeline[:first] if e.startswith("gen:")])
            out["p_rpc"], out["d_rpc"] = _rpc_sequence(h.p), _rpc_sequence_without_flip_park(h.d)
            out["floor_rpc"] = (list(h.p.floor_bodies), list(h.d.floor_bodies), _lane_parks(h.d))
            out["counters"] = _lane_counters(h.front)
            out["epoch"] = h.front.epoch
            out["resume_handoff"] = [b.get("handoff_n") for b in h.d.resume_bodies if "kv_cache" in (b.get("tags") or [])]
            out["floor"] = (h.front._lane_state().lane_floor, h.front._lane_state().lane_epoch) if switch != "0" else None
    with _switch(switch):
        caplog.clear()
        caplog.set_level(logging.INFO, logger="weg2.front")
        asyncio.run(body())
    out["lane_lines"] = _lane_lines(caplog)
    out["start_lines"] = _start_lines(caplog)
    out["flip_begin_tails"] = [re.sub(r"epoch=\d+", "epoch=N", re.sub(r"queue=\d+", "queue=Q", re.sub(r"outstanding=\d+", "outstanding=O", r.getMessage())))
                               for r in caplog.records if r.getMessage().startswith("WEG2-FLIP begin")]
    return out


@_served_front_wrap
def test_lane_0_only_across_a_p_phase_and_its_flips_routing_and_log_equal_the_switch_off_run(caplog):
    on = _run_p_phase(None, caplog)
    off = _run_p_phase("0", caplog)
    for r in (on, off):
        assert r["status"] == [200] * 9
        assert r["floor_rpc"] == ([], [], []) and r["counters"] == {}      # (flip parks carry no rids / hold: see _lane_parks)
        assert r["lane_lines"] == [], r["lane_lines"]
        assert r["p_before_first_flip"] == 6          # rule 1 of the phase policy (cap 6) is untouched
    assert on["floor"] == (0, 0)
    assert (on["p_gen"], on["d_gen"], on["p_rpc"], on["d_rpc"], on["epoch"], on["resume_handoff"]) == \
           (off["p_gen"], off["d_gen"], off["p_rpc"], off["d_rpc"], off["epoch"], off["resume_handoff"])
    # the WEG2-FLIP begin lines carry no lane tail (lane_held=/ledger= only with a booked lane hold)
    assert on["flip_begin_tails"] == off["flip_begin_tails"] and on["flip_begin_tails"]
    assert not any("lane_held=" in t for t in on["flip_begin_tails"])


# ------------------------------------------------------------------ 4. the P side told ledger
class _Abort:
    rid = "weg2-0-5"
    abort_all = False


def test_the_told_forget_touches_nothing_on_a_rank_no_lane_floor_ever_reached(monkeypatch):
    from sglang.srt.managers import weg2_store_told as told

    monkeypatch.setenv("SGLANG_WEG2_GROUP", "P")
    monkeypatch.delenv("SGLANG_WEG2_PARK", raising=False)
    monkeypatch.delenv("SGLANG_WEG2_LANES", raising=False)          # the code default: on
    calls = []
    monkeypatch.setattr(told, "forget_aborted", lambda *a, **k: calls.append((a, k)) or {"told": 1})
    sched = SimpleNamespace(ps=SimpleNamespace(pp_rank=1, pp_size=3, tp_size=1))
    assert lanes_p.active() is True                                   # switch on (default), group P, park on
    assert lanes_p.state_of(sched) is None
    assert lanes_p.forget_aborted_told(sched, _Abort()) is None
    assert calls == []                                                # the ledger was not looked at: the Dauerstand's abort


def test_the_told_forget_runs_on_a_rank_that_has_a_lane_state(monkeypatch):
    from sglang.srt.managers import weg2_store_told as told

    monkeypatch.setenv("SGLANG_WEG2_GROUP", "P")
    monkeypatch.delenv("SGLANG_WEG2_PARK", raising=False)
    monkeypatch.delenv("SGLANG_WEG2_LANES", raising=False)
    calls = []
    monkeypatch.setattr(told, "forget_aborted", lambda *a, **k: calls.append((a, k)) or {"told": 7})
    sched = SimpleNamespace(ps=SimpleNamespace(pp_rank=1, pp_size=3, tp_size=1))
    lanes_p.state_of(sched, create=True)                              # what a floor RPC (PP0) / a stamp (PP1, PP2) does
    assert lanes_p.forget_aborted_told(sched, _Abort()) == {"told": 7}
    assert len(calls) == 1


def test_the_told_forget_is_off_with_the_switch_at_zero_even_with_a_lane_state(monkeypatch):
    from sglang.srt.managers import weg2_store_told as told

    monkeypatch.setenv("SGLANG_WEG2_GROUP", "P")
    monkeypatch.setenv("SGLANG_WEG2_LANES", "0")
    monkeypatch.setattr(told, "forget_aborted", lambda *a, **k: pytest.fail("the ledger was touched with the switch off"))
    sched = SimpleNamespace(ps=SimpleNamespace(pp_rank=1, pp_size=3, tp_size=1))
    lanes_p.state_of(sched, create=True)
    assert lanes_p.forget_aborted_told(sched, _Abort()) is None


# ------------------------------------------------------------------ 5. the D side displacement
def test_the_lane_aware_displacement_candidate_equals_the_laneless_one_when_every_request_is_lane_0():
    rng = random.Random(1013)
    for _ in range(300):
        n_wait, n_run = rng.randint(0, 5), rng.randint(0, 6)
        ages = rng.sample(range(1, 60), n_wait + n_run)
        waiting = [SimpleNamespace(rid="weg2-0-%d" % a, priority=rng.choice([None, 0])) for a in ages[:n_wait]]
        running = [SimpleNamespace(rid="weg2-0-%d" % a, priority=rng.choice([None, 0])) for a in ages[n_wait:]]
        off = dpr._displace_candidate(None, SA, waiting, running, False)
        on = dpr._displace_candidate(None, SA, waiting, running, True)
        assert on == off, (waiting, running, on, off)


def test_the_served_front_starts_the_lane_loop_only_with_the_switch_on():
    """The loop (and with it its one start line) belongs to the switch: ``=0`` starts no task, as in the Dauerstand."""
    import inspect

    from sglang.srt.weg2 import front as fr

    src = inspect.getsource(fr)
    i = src.index('app["lane_ctl"] = asyncio.create_task(_lctl.loop(self))')
    assert "if _lanes.enabled():" in src[i - 200:i]


# ------------------------------------------------------------------ 6. earlycap x lanes (the merge's one shared surface in the P-phase begin)
def test_a_lane_held_early_leg_frees_its_place_under_the_earlycap_seat_cap():
    """int24f earlycap (``_leg1_early_limit``: the seat gate's cap minus the early legs in flight IN THE QUEUE) and the lanes' DEFER/hold path
    (``lane_ctl._cancel_early``: the held Pending leaves ``queue`` and its early leg is cancelled by ``_leg1_early_cancel``) meet at
    ``Pending._leg1_early``.  Both intents: the cap counts the early legs of the queue, and a lane-held one no longer counts."""
    from sglang.srt.weg2 import lane_ctl as LC
    from test_weg2_lanes_l4_1008 import _pend

    async def body():
        with _switch(None):
            f = _front("D")
            f.p_concurrency = 4
            f._dc_p_cap = 2                                   # the seat gate's cap for this D>P flip
            a, b = _pend(f, "weg2-0-1", 0, 100.0), _pend(f, "weg2-0-2", 0, 101.0)
            for p in (a, b):
                p._leg1_early = asyncio.ensure_future(asyncio.sleep(30))
                f.queue.append(p)
            assert Front._leg1_early_limit(f) == 0            # cap 2, two early legs in flight: no third
            f.groups["P"].outstanding[a.rid] = 100.0
            lc = LC.ctl(f)
            assert LC._cancel_early(f, lc, a, 102.0, when="hold", target=1) is True      # the lane takes the early leg back
            f.queue.remove(a)                                 # held out of the queue (what lane_ctl does with a held Pending)
            assert a._leg1_early is None
            assert Front._leg1_early_limit(f) == 1            # one early leg left in flight: the cap has room for one more
            for p in (a, b):
                t = p._leg1_early
                if t is not None:
                    t.cancel()
            await asyncio.sleep(0)
    asyncio.run(body())
