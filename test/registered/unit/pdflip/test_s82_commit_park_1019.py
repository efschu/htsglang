"""S82 (19.10.2026): PP0 parks for its next set at the pass top, not blind in the output receive.

Metal (rc18 7737b78948, P log ..._1019_200330, PP0):
  20:42:22  '#969N ADMIT slot=0 ... extend=16117 rids=[pdflip-28-233]'   (forward ~3.1 s, host-blocking)
  20:42:26  '#TW TWIN-REGISTER rid=pdflip-28-232 twin=False verdict=issued head=17984'  (after the forward)
  20:42:30  '#1466 PASS-STALL pp_rank=0 slot=0 pass_ms=4317 ... commit_ms=4153'
  20:42:30  'PDFLIP-LOAD-DEVICE req=pdflip-28-232 ... read_ms=102 harvest_ms=4206'
and the paced/PF told: the followers ack at once ('#1400 FOLLOWER-EARLY-SETTLE ... at=ack'), PP0 decides
only at its next pass top ('PF TOLD-ACKED ... after=2.28s').

Pinned (hermetic, CPU; RED on rc18 7737b78948 -- no park, no twin-at-plan):
  1. the pure verdict: park only when the receive will block and a waiter can resolve without PP0;
  2. the wait ends at the FIRST of acks / read / output, never past the receive's own budget;
  3. the output probe reads the same counters as the receive's readiness gate;
  4. METAL: acks that land while PP0 would sit in the receive admit the paced rid IN THIS pass;
  5. output first = the pre-S82 pass (no Admit), control/no-blocking-commit/switch-off never park;
  6. TWIN-AT-PLAN: the twin's read is registered right after the source's plan (before the forward);
  7. the loop wiring (slot context, after-plan hook, park before the paced branch).
"""

import inspect
import os
import sys
import time as _time
from types import SimpleNamespace

import pytest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from flliper.srt.managers import pdflip_store_told as told  # noqa: E402
from flliper.srt.managers import pdflip_told_fallback as fb  # noqa: E402
from flliper.srt.managers import pdflip_told_fidelity as tf  # noqa: E402
from flliper.srt.managers import pdflip_told_raise as traise  # noqa: E402
from flliper.srt.pdflip import p_twin_defer as tw  # noqa: E402
from flliper.srt.pdflip import pp0_commit_park as cp  # noqa: E402
from flliper.srt.pdflip import prefill_clock as pfc  # noqa: E402
from flliper.test.ci.ci_register import register_cpu_ci  # noqa: E402

register_cpu_ci(est_time=10, suite="stage-a-test-cpu")

RID = "pdflip-28-232"
TOLD = 103872           # metal: '#1416 STORE-TOLD ANCHOR-CLAMP rid=pdflip-28-232 ... anchored=103872'


# ---------------------------------------------------------------------------
# 1. the pure verdict

def _d(**kw):
    base = dict(output_pending=True, control=False, carried=False, admissible_any=False,
                decidable_any=False, open_reads=(), ack_waits=(RID,))
    base.update(kw)
    return cp.decide(**base)


def test_verdict_parks_only_for_a_blocking_receive_with_a_waiter():
    assert _d().park is True and _d().acks == (RID,)
    assert _d(ack_waits=(), open_reads=("r1",)).park is True
    assert _d(output_pending=False).reason == "no-blocking-commit"
    assert _d(output_pending=None).reason == "commit-unknown"
    assert _d(control=True).reason == "control"
    assert _d(carried=True).reason == "carried-chunk"
    assert _d(admissible_any=True).reason == "joins-anyway"
    assert _d(decidable_any=True).reason == "joins-anyway"
    assert _d(ack_waits=(), open_reads=()).reason == "no-waiter"


# ---------------------------------------------------------------------------
# 2. the wait

class _Clock:
    def __init__(self):
        self.t = 100.0

    def __call__(self):
        return self.t

    def sleep(self, s):
        self.t += s


def _wait(verdict, *, acks_at=None, read_at=None, output_at=None, bound=30.0, raise_at=None):
    clk = _Clock()
    n = {"i": 0}

    def harvest():
        n["i"] += 1
        if raise_at is not None and n["i"] >= raise_at:
            raise RuntimeError("probe")

    out = cp.wait(
        verdict,
        output_pending=lambda: not (output_at is not None and n["i"] >= output_at),
        read_done=lambda r: read_at is not None and n["i"] >= read_at,
        acks_done=lambda r: acks_at is not None and n["i"] >= acks_at,
        harvest=harvest, bound_s=bound, clock=clk, sleep=clk.sleep,
    )
    return out, n["i"]


def test_wait_ends_at_the_first_event():
    v = cp.Verdict(True, "park", ("r1",), (RID,))
    (o, rid, _w), polls = _wait(v, acks_at=3, output_at=5)
    assert (o, rid, polls) == ("acks", RID, 3)
    (o, rid, _w), polls = _wait(v, read_at=2, acks_at=4)
    assert (o, rid, polls) == ("read", "r1", 2)
    (o, rid, _w), polls = _wait(v, acks_at=9, output_at=4)
    assert (o, rid, polls) == ("output", None, 4)


def test_wait_never_outlives_the_receive_budget_and_a_broken_probe_ends_it():
    v = cp.Verdict(True, "park", (), (RID,))
    (o, _rid, waited), _ = _wait(v, bound=0.05)
    assert o == "bound" and 0.05 <= waited < 0.053
    (o, _rid, _w), polls = _wait(v, raise_at=2)
    assert (o, polls) == ("error", 2)


# ---------------------------------------------------------------------------
# 3./4./5. the adapter on a PP0 stand-in (paced + PF told, the NF form)

class _Ctr:
    """pp_flip_counters with the per-kind axis: one output owed by PP2."""

    def __init__(self):
        self.sent_n, self.att_n, self.cons_n = 4, 4, 4   # caught up: the owed output is not posted yet

    def sent(self, chan, rank):
        return self.sent_n

    def kind_sent_total(self, chan, rank):
        return self.sent_n

    def sent_of_kind(self, chan, kind, rank):
        return self.sent_n if kind == "output" else 0

    def attempted_of_kind(self, chan, kind, rank):
        return self.att_n if kind == "output" else 0

    def local_consumed_of_kind(self, chan, kind):
        return self.cons_n if kind == "output" else 0


def _batch():
    return SimpleNamespace(reqs=[SimpleNamespace(rid="pdflip-28-233")],
                           forward_mode=SimpleNamespace(is_prebuilt=lambda: False),
                           contains_last_prefill_chunk=True, return_logprob=False)


def _sched():
    req = SimpleNamespace(rid=RID, origin_input_ids=list(range(TOLD + 30)), extra_key=None)
    now = told._clock()
    s = SimpleNamespace(
        ps=SimpleNamespace(pp_rank=0, pp_size=3, tp_size=1),
        waiting_queue=[req], chunked_req=None, pdflip_dormant=False,
        _pdflip_store_held={}, _pdflip_store_told={},
        _pdflip_told_paced_on=True, _pdflip_told_fallback_on=True, _pdflip_told_pass_n=5,
        _pdflip_told_pacing={RID: told._Pace(req=req, told=TOLD, published_at=now - 0.05, published_pass=4,
                                              window_s=5.0, absolute=True)},
        _pdflip_fb_open={RID: fb._Open(told=TOLD, deadline=now + 30.0, published_pass=4, published_at=now - 0.05)},
        tree_cache=SimpleNamespace(ongoing_prefetch={}),
        mbs=[_batch(), None, None], _s82_slot=(2, 0),          # slot 2 plans, slot 0's output is owed
        server_args=SimpleNamespace(pp_async_batch_depth=0),
        pp_flip_counters=_Ctr(), pp_group=SimpleNamespace(rank_in_group=0, world_size=3),
        _pp_flip_upstream=lambda: 2,
        # the older set in slot 0: pdflip-28-233 (16117 new on 17984 cached), launched just before the park
        _s82_launch_rec={0: (_time.monotonic() - 0.1, 16117, 17984)},
    )
    return s


# PP0's measured rings, numbers from the rc18/181004 P logs (Prefill rank batch PP0, PDFLIP-START-LOADING PP0,
# PP0 commit of passes whose output was not posted at entry = launch end -> output)
FWD = [(16117, 17984, 3134.7), (30, 103872, 691.8), (2431, 409472, 2300.1), (2243, 132096, 2355.1),
       (13025, 24576, 2877.9), (9620, 25024, 2685.4), (538, 82432, 1617.2), (56, 83072, 1662.5), (186, 0, 1293.5),
       (16384, 17920, 3126.0), (9013, 28672, 3246.0), (12083, 65536, 3615.0), (10159, 112384, 2752.0),
       (7949, 26304, 2558.0), (4900, 85760, 2652.0), (11595, 17984, 2801.0)]
LOAD = [(337024, 807.0), (132096, 165.0), (85888, 109.0), (17984, 217.0)]
DOWN = [(16117, 17984, 4153.0), (2431, 409472, 3064.0), (9013, 28672, 2667.0), (12083, 65536, 3446.0),
        (10159, 112384, 3402.0), (7949, 26304, 2411.0), (16334, 17984, 4171.0), (4900, 85760, 2124.0),
        (11595, 17984, 3222.0), (2915, 318272, 2017.0), (2711, 0, 2052.0), (3163, 0, 2256.0), (315, 96768, 2521.0)]


def _seed_rings():
    pfc._reset_for_tests()
    cp.reset_rings()
    for n, c, ms in FWD:
        pfc.note_batch_cost(n, c, ms, 1)
    for n, ms in LOAD:
        cp.note_start_loading(n, ms)
    for n, c, ms in DOWN:
        cp.note_downstream(n, c, ms)


@pytest.fixture
def pp0(monkeypatch):
    """The PF/paced pieces that are not under test answer as on the metal."""
    cp.reset()
    monkeypatch.delenv(cp.ENV, raising=False)
    monkeypatch.setattr(fb, "pp0_has_stream", lambda s: True)
    monkeypatch.setattr(fb, "_frist_verdict", lambda *a, **k: None)          # inside the Frist: keep waiting
    monkeypatch.setattr(traise, "pp0_raise_step", lambda *a, **k: None)
    monkeypatch.setattr(tf, "pp0_verdict", lambda s, req, t, absolute: (t, t))
    _seed_rings()
    yield _sched()
    pfc._reset_for_tests()
    cp.reset_rings()


def _acks_after(monkeypatch, sched, k, ctr_post_at=None):
    """The PF ack stream: both followers' acks land with the k-th harvest (they early-settled at once);
    optionally the owed output is posted with harvest ``ctr_post_at``."""
    n = {"i": 0}

    def harvest(s, now=None):
        n["i"] += 1
        if n["i"] >= k:
            fb._pp0_open_map(s)[RID].acks.update({1: TOLD, 2: TOLD})
        if ctr_post_at is not None and n["i"] >= ctr_post_at:
            s.pp_flip_counters.sent_n = 5
        return 0

    monkeypatch.setattr(fb, "pp0_harvest", harvest)
    return n


def _admits(out):
    return [o for o in out if isinstance(o, told.PdFlipStoreAdmit)]


def test_output_probe_reads_the_receive_gate(pp0):
    assert told._s82_output_pending(pp0) is True
    pp0.pp_flip_counters.sent_n = 5                           # PP2 posted it
    assert told._s82_output_pending(pp0) is False
    pp0.pp_flip_counters.sent_n, pp0.pp_flip_counters.att_n = 4, 5   # PP2 is inside the send
    assert told._s82_output_pending(pp0) is False
    pp0.pp_flip_counters.att_n = 4
    from flliper.srt.distributed.pp_typed_channel import typed_inbox

    typed_inbox(pp0.pp_group)[(2, "output")].append({"stashed": 1})
    assert told._s82_output_pending(pp0) is False             # already off the wire
    typed_inbox(pp0.pp_group)[(2, "output")].clear()
    pp0._s82_slot = (0, 1)
    assert told._s82_output_pending(pp0) is False             # next slot empty: nothing owed
    pp0._s82_slot = (2, 0)
    pp0._pp_gapped_wire = True
    assert told._s82_output_pending(pp0) is None
    pp0._pp_gapped_wire = False
    pp0.pp_flip_counters = None
    assert told._s82_output_pending(pp0) is None


def test_metal_2042_acks_landing_in_the_receive_admit_in_this_pass(pp0, monkeypatch):
    """rc18 20:42:26-30: the acks are in within milliseconds, the output of the 16117-token chunk
    takes ~4 s. RED on rc18: one harvest at the top, no acks, no Admit -- the rid waits a lap."""
    n = _acks_after(monkeypatch, pp0, k=3)
    out = told.pp0_publish(pp0, [])
    assert [a.rid for a in _admits(out)] == [RID] and _admits(out)[0].told == TOLD
    assert pp0._pdflip_store_told[RID] == TOLD                # PP0's own plan admits it in this pass
    assert n["i"] >= 3
    st = cp.stats()
    assert st["n"] == 1 and st["acks"] == 1 and pp0._s82_park_ms >= 0.0
    assert st.get("park_admit") == 1 and pp0._s82_park_admit is not None   # S103: PARK-ADMIT, A3 as instrument


def test_output_first_is_the_pre_s82_pass(pp0, monkeypatch):
    _acks_after(monkeypatch, pp0, k=10 ** 9, ctr_post_at=2)
    out = told.pp0_publish(pp0, [])
    assert _admits(out) == [] and RID in pp0._pdflip_told_pacing
    assert cp.stats()["output"] == 1


@pytest.mark.parametrize("case", ["no_blocking_receive", "control", "switch_off", "carried"])
def test_never_parks_where_the_receive_would_not_block_or_the_pass_must_run(pp0, monkeypatch, case):
    n = _acks_after(monkeypatch, pp0, k=2)
    recv = []
    if case == "no_blocking_receive":
        pp0._s82_slot = (0, 1)
    elif case == "control":
        recv = [type("FlushCacheReqInput", (), {})()]
    elif case == "switch_off":
        monkeypatch.setenv(cp.ENV, "0")
    else:
        pp0.chunked_req = SimpleNamespace(rid="pdflip-28-233")
    out = told.pp0_publish(pp0, recv)
    assert _admits(out) == [] and n["i"] == 1                  # the one harvest of the paced top, no park
    assert cp.stats()["n"] == 0


def test_a_broken_probe_ends_the_park_never_the_pass(pp0, monkeypatch):
    def harvest(s, now=None):
        if getattr(s, "_hv", 0):
            raise RuntimeError("ack stream")
        s._hv = 1
        return 0

    monkeypatch.setattr(fb, "pp0_harvest", harvest)
    monkeypatch.setattr(fb, "pp0_watching", lambda s: False)
    # the top's own harvest succeeds, the park's raises: the park ends 'error', the pass returns
    told._commit_park(pp0, [])
    assert cp.stats()["error"] == 1


# ---------------------------------------------------------------------------
# 6. TWIN-AT-PLAN (metal numbers: shared 17984, chunk 16384, page 64, source start 17984)

SHARED, PAGE, CHUNK = 17984, 64, 16384


def _twin_sched(monkeypatch):
    monkeypatch.setenv(tw.ENV, "1")
    monkeypatch.setenv(tw.ENV_MIN_TOKENS, "8192")
    monkeypatch.setattr(tw, "_twin_anchor_armed", lambda: False, raising=False)   # rule 4 (gain)
    src_ids = list(range(SHARED)) + [10 ** 6 + i for i in range(16117)]
    twin_ids = list(range(SHARED)) + [2 * 10 ** 6 + i for i in range(30)]

    class _R(SimpleNamespace):
        def finished(self):
            return False

    src = _R(rid="pdflip-28-233", origin_input_ids=src_ids, extra_key=None, prefix_indices=None,
             fill_ids=None, _prefetch_registered_prefix_len=0)
    twin = _R(rid=RID, origin_input_ids=twin_ids, extra_key=None, prefix_indices=None, fill_ids=None,
              _prefetch_registered_prefix_len=0)
    calls = []
    s = SimpleNamespace(ps=SimpleNamespace(pp_rank=0, pp_size=3, tp_size=1), waiting_queue=[src], chunked_req=None,
                        running_batch=None, page_size=PAGE, chunked_prefill_size=CHUNK, mbs=[None, None, None],
                        _prefetch_kvcache=lambda r, **k: calls.append(r.rid) or "issued")
    tw.state(s)
    s.waiting_queue = [src, twin]
    assert tw.intake_defer(s, twin) is True          # the source's start is unknown at the twin's intake
    # PP0 plans slot 0: the source leaves the queue with its matched prefix (cached 17984)
    s.waiting_queue = [twin]
    src.prefix_indices = [0] * SHARED
    s.mbs[0] = SimpleNamespace(reqs=[src])
    return s, calls


def test_rc18_base_holds_the_twin_through_the_forward(monkeypatch):
    """What rc18 does: nothing between the plan and the next top asks TW -- release_due without the
    plan's s0 keeps the twin held, so its read starts only after PP0's forward."""
    s, calls = _twin_sched(monkeypatch)
    assert tw.release_due(s, {RID}) == [] and calls == []
    assert tw.is_deferred(s, RID)


def test_twin_at_plan_registers_the_read_before_the_forward(monkeypatch):
    cp.reset()
    s, calls = _twin_sched(monkeypatch)
    told.pp0_after_plan(s)
    assert calls == [RID] and not tw.is_deferred(s, RID)     # registered as an ordinary request (twin=False)
    assert cp.stats()["twin_at_plan"] == 1


def test_twin_at_plan_switch_off_keeps_the_top(monkeypatch):
    s, calls = _twin_sched(monkeypatch)
    monkeypatch.setenv(cp.ENV_TWIN_AT_PLAN, "0")
    told.pp0_after_plan(s)
    assert calls == [] and tw.is_deferred(s, RID)


# ---------------------------------------------------------------------------
# 7. wiring

def test_loop_wiring():
    from flliper.srt.managers import scheduler_pp_mixin as m

    body = inspect.getsource(m.SchedulerPPMixin._event_loop_pp_body)
    assert "self._s82_slot = (mb_id, next_mb_id)" in body
    a = body.index("_pp_void_pass.after_plan(self, mb_id)")
    b = body.index("pdflip_store_told.pp0_after_plan(self)")
    c = body.index("self._pp_launch_batch(")
    assert a < b < c                                   # after the final plan, before the forward
    pub = inspect.getsource(told.pp0_publish)
    assert pub.index("_commit_park(scheduler, recv_reqs)") < pub.index("_pp0_publish_paced(scheduler, recv_reqs)")


# ---------------------------------------------------------------------------
# review nf-review-s82-1019: A1 one deadline, A2 a probe that turns unknown, A3 Prio = Alter, H2

def test_a2_a_probe_that_turns_unknown_mid_park_ends_it():
    """M3 (``is False`` instead of ``is not True``) would park on until the budget."""
    v = cp.Verdict(True, "park", (), (RID,))
    clk = _Clock()
    n = {"i": 0}

    def harvest():
        n["i"] += 1

    o, rid, waited = cp.wait(v, output_pending=lambda: None if n["i"] >= 3 else True,
                             read_done=lambda r: False, acks_done=lambda r: False, harvest=harvest,
                             bound_s=30.0, clock=clk, sleep=clk.sleep)
    assert (o, n["i"]) == ("output", 3) and waited < 0.01


def test_a2_wire_turning_gapped_mid_park_ends_it(pp0, monkeypatch):
    monkeypatch.setenv("FLLIPER_PP_PROXY_READINESS_BUDGET_S", "5")

    def harvest(s, now=None):
        s._hv = getattr(s, "_hv", 0) + 1
        if s._hv >= 3:
            s._pp_gapped_wire = True        # the probe can no longer tell
        return 0

    monkeypatch.setattr(fb, "pp0_harvest", harvest)
    t0 = _time.monotonic()
    told._commit_park(pp0, [])
    assert cp.stats()["output"] == 1 and _time.monotonic() - t0 < 1.0


def test_a1_park_and_receive_share_one_deadline(pp0, monkeypatch):
    """A park ending on 'bound' leaves the receive no fresh budget: #789 is named after ONE budget."""
    from flliper.srt.managers.scheduler_pp_mixin import SchedulerPPMixin

    monkeypatch.setenv("FLLIPER_PP_PROXY_READINESS_BUDGET_S", "0.3")
    _acks_after(monkeypatch, pp0, k=10 ** 9)                 # acks never, output never: a wedge
    t0 = _time.monotonic()
    told._commit_park(pp0, [])
    assert cp.stats()["bound"] == 1 and 0.3 <= _time.monotonic() - t0 < 0.45
    t1 = _time.monotonic()
    with pytest.raises(RuntimeError, match="READINESS TIMEOUT"):
        SchedulerPPMixin._pp_wait_for_dict_readiness(pp0, 0, kind="output")
    assert _time.monotonic() - t1 < 0.12                     # the rest of the shared budget, not 0.3 s more
    assert getattr(pp0, "_s82_park_admit", None) is None     # 'bound': nothing became admissible, nothing admitted


def test_a1_receive_without_park_keeps_its_own_budget(pp0, monkeypatch):
    from flliper.srt.managers.scheduler_pp_mixin import SchedulerPPMixin

    monkeypatch.setenv("FLLIPER_PP_PROXY_READINESS_BUDGET_S", "0.2")
    pp0._s82_receive_deadline = None
    t1 = _time.monotonic()
    with pytest.raises(RuntimeError, match="READINESS TIMEOUT"):
        SchedulerPPMixin._pp_wait_for_dict_readiness(pp0, 0, kind="output")
    assert _time.monotonic() - t1 >= 0.2


def test_a3_measured_prices_of_the_two_metal_cases():
    """20:42: the 30-token twin ends long before 28-233's output; 20:16: the four younger (2431 new on
    409472 cached) cost more than the time left until 8-74's output -- the older first."""
    _seed_rings()
    try:
        twin = cp.price_ms(30, TOLD)
        old42 = cp.down_ms(16117, 17984)
        v42 = cp.age_rule(now=100.0, price=twin, launch_old=100.0 - 0.05, down=old42)
        assert v42.admit and v42.reason == "before-due"
        four = cp.price_ms(2431, 409472)
        old16 = cp.down_ms(2243, 132096)
        # 8-74 launched (forward end) 20:16:38.36, the reads landed ~38.8: 0.45 s into its downstream
        v16 = cp.age_rule(now=100.0, price=four, launch_old=100.0 - 0.45, down=old16)
        assert not v16.admit and v16.reason == "older-first" and four > v16.due_in_ms
    finally:
        pfc._reset_for_tests()
        cp.reset_rings()


def test_s103_metal_2016_the_younger_set_runs_in_the_park_pass(pp0, monkeypatch):
    """S103 replaces A3's hold: the four younger (2431 on 409472) are admitted while 8-74 is downstream --
    A3 would have held them (price 3338 > 1754 left), the verdict rides along as an instrument only."""
    req = pp0.waiting_queue[0]
    req.origin_input_ids = list(range(409472 + 2431))
    pp0._pdflip_told_pacing[RID].told = 409472
    fb._pp0_open_map(pp0)[RID].told = 409472
    pp0._s82_launch_rec = {0: (_time.monotonic() - 0.45, 2243, 132096)}
    n = {"i": 0}

    def harvest(s, now=None):
        n["i"] += 1
        if n["i"] >= 3:
            fb._pp0_open_map(s)[RID].acks.update({1: 409472, 2: 409472})
        return 0

    monkeypatch.setattr(fb, "pp0_harvest", harvest)
    out = told.pp0_publish(pp0, [])
    assert [a.rid for a in _admits(out)] == [RID] and _admits(out)[0].told == 409472
    st = cp.stats()
    assert st.get("park_admit") == 1 and st.get("a3_would_hold") == 1 and pp0._s82_park_admit is not None


def test_s103_unmeasured_price_admits(pp0, monkeypatch):
    """rc24 shape: no measured price yet (A3 'price-unmeasured', 9 of rc24's 14 verdicts) -> admitted all the same."""
    cp.reset_rings()                                          # no START-LOADING / downstream samples yet
    _acks_after(monkeypatch, pp0, k=3)
    out = told.pp0_publish(pp0, [])
    assert [a.rid for a in _admits(out)] == [RID]
    assert cp.stats().get("a3_would_hold") == 1


def test_s103_output_first_is_the_pre_s82_pass(pp0, monkeypatch):
    """The owed output is posted before the acks: the park ends on 'output', nothing was admitted BY the park
    and nothing is held either (A3's hold set is gone) -- the publish decides as before S82."""
    _acks_after(monkeypatch, pp0, k=4, ctr_post_at=2)         # output posted before the acks
    told.pp0_publish(pp0, [])
    assert cp.stats()["output"] == 1 and getattr(pp0, "_s82_park_admit", None) is None
    assert not hasattr(pp0, "_s82_hold")


def test_a3_after_commit_samples_and_output_late(pp0):
    cp.reset_rings()
    pp0._s82_launch_rec = {0: (_time.monotonic() - 2.0, 16117, 17984)}
    told.s82_after_commit(pp0, 0, (_time.monotonic(), True))   # not posted at entry: exact sample
    assert len(cp._DOWN) == 1 and 1900.0 <= cp._DOWN[0][2] <= 2500.0
    pp0._s82_launch_rec = {0: (_time.monotonic() - 2.0, 9000, 1000)}
    told.s82_after_commit(pp0, 0, (_time.monotonic(), False))  # already posted at entry: no sample
    assert len(cp._DOWN) == 1
    cp.reset()
    pp0._s82_park_admit = (_time.monotonic() - 1.0, 500.0)
    told.s82_after_commit(pp0, 0, (_time.monotonic(), False))
    st = cp.stats()
    assert st["late_n"] == 1 and 900.0 <= st["late_max_ms_sum"] <= 1200.0 and pp0._s82_park_admit is None
    pp0._s82_park_admit = (_time.monotonic() - 1.0, 500.0)
    told.s82_after_commit(pp0, 0, (_time.monotonic(), True))   # not yet posted at entry: no lateness
    assert cp.stats()["late_n"] == 2 and cp.stats()["late_max_ms_sum"] <= 1200.0


def test_h2_no_park_while_p_sleeps(pp0, monkeypatch):
    n = _acks_after(monkeypatch, pp0, k=2)
    pp0.pdflip_dormant = True
    told._commit_park(pp0, [])
    assert cp.stats()["n"] == 0 and n["i"] == 0


def test_h2_paced_read_does_not_end_a_park_with_an_ack_waiter(pp0, monkeypatch):
    """M11: a landed read only publishes a read-ahead (its Admit is a pass later); the waiting Admit ends it."""
    other = SimpleNamespace(rid="pdflip-28-240", origin_input_ids=list(range(50)), extra_key=None,
                            prefetch_deferred=None)
    pp0.waiting_queue.append(other)
    pp0._pdflip_store_held = {other.rid: other}
    op = SimpleNamespace(host_indices=[1], hash_value=[1])
    pp0.tree_cache = SimpleNamespace(ongoing_prefetch={other.rid: SimpleNamespace(operation=op)},
                                     can_terminate_prefetch=lambda op, tail_hold=0: True)
    _acks_after(monkeypatch, pp0, k=4)
    told._commit_park(pp0, [])
    assert cp.stats()["acks"] == 1 and cp.stats()["read"] == 0


# ---------------------------------------------------------------------------
# Nach-Review 20.10.: B1 (a new request is never read behind the park), B2 (D2, D5, D8, D9)

def _arrival(rid="pdflip-28-240"):
    return type("TokenizedGenerateReqInput", (), {"rid": rid})()


def test_b1_a_request_arriving_at_the_parking_top_is_not_read_behind_the_park(pp0, monkeypatch):
    """rc18 20:42 shape: a request reaches PP0 in the pass whose top would park ~4 s. Its intake + store read
    run in process_input_requests AFTER pp0_publish -- so that pass must not park (before S82 the read ran
    while PP0 sat in the receive). RED on c36320ceac: it parks and the newcomer's read waits."""
    n = _acks_after(monkeypatch, pp0, k=3)
    out = told.pp0_publish(pp0, [_arrival()])
    assert cp.stats()["n"] == 0 and n["i"] == 1                # the paced top's one harvest, no park
    assert _admits(out) == []                                  # the acked rid follows at a later top
    v = cp.decide(output_pending=True, control=False, carried=False, arrivals=True, admissible_any=False,
                  decidable_any=False, open_reads=(), ack_waits=(RID,))
    assert (v.park, v.reason) == (False, "arrivals")


@pytest.mark.parametrize("name,arrival", [("TokenizedGenerateReqInput", True), ("BatchTokenizedGenerateReqInput", True),
                                          ("TokenizedEmbeddingReqInput", True), ("PdFlipStoreTold", False),
                                          ("FlushCacheReqInput", False), ("PdFlipBurstClock", False)])
def test_b1_arrival_kinds(name, arrival):
    assert cp.is_arrival(type(name, (), {})()) is arrival


def test_d2_the_parks_deadline_never_outlives_its_pass(pp0):
    """D2: a deadline left from an earlier pass would fail a later receive at once (#789, group death). The
    slot top resets it before the park can run, and the pass's receive drops it after use."""
    from flliper.srt.managers import scheduler_pp_mixin as m

    body = inspect.getsource(m.SchedulerPPMixin._event_loop_pp_body)
    slot = body.index("self._s82_slot = (mb_id, next_mb_id)")
    reset = body.index("self._s82_receive_deadline = None")
    intake = body.index("self._pp_forward_and_process_input_requests(")   # pp0_publish -> the park runs in here
    assert slot < reset < intake
    pp0._s82_receive_deadline = _time.monotonic() - 100.0       # a stale deadline
    told.s82_after_commit(pp0, 0, (_time.monotonic(), True))
    assert pp0._s82_receive_deadline is None


def test_d5_the_price_carries_the_load(monkeypatch):
    """D5: without START-LOADING the younger set looks cheap enough and delays the older output."""
    _seed_rings()
    try:
        line = cp._fit(cp.fwd_rows())
        fwd = cp._line_ms(line, 2431, 409472)
        price = cp.price_ms(2431, 409472)
        rates = sorted(v / n for n, v in LOAD)
        assert abs(price - fwd - rates[len(rates) // 2] * 409472) < 1e-6 and price - fwd > 500.0
        due_in = (fwd + price) / 2.0                               # forward alone would fit, with the load not
        v = cp.age_rule(now=100.0, price=price, launch_old=100.0, down=due_in)
        assert not v.admit and v.reason == "older-first"
    finally:
        pfc._reset_for_tests()
        cp.reset_rings()


def _single_phase(pp0, monkeypatch, k_read=3):
    """The non-paced (#1400 single-phase) form: the park waits on PP0's own open read of a held rid."""
    req = pp0.waiting_queue[0]
    req.prefetch_deferred = None
    pp0._pdflip_told_paced_on = False
    pp0._pdflip_told_fallback_on = False
    pp0._pdflip_told_pacing = {}
    pp0._pdflip_store_held = {RID: req}
    n = {"i": 0}

    def can_terminate(op, tail_hold=0):
        n["i"] += 1
        return n["i"] >= k_read

    op = SimpleNamespace(host_indices=[1], hash_value=[1])
    pp0.tree_cache = SimpleNamespace(ongoing_prefetch={RID: SimpleNamespace(operation=op)},
                                     can_terminate_prefetch=can_terminate,
                                     check_prefetch_progress=lambda rid: True)
    monkeypatch.setattr(told, "_p_held_short_reread", lambda *a, **k: False)
    monkeypatch.setattr(told, "_pp0_told_any", lambda s, tree, req, rid, twin: (TOLD, True))
    monkeypatch.setattr(told, "_ple_admit_at_told", lambda *a, **k: None)
    monkeypatch.setattr(told, "_pp0_fork", lambda *a, **k: 0)
    return req


def _tolds(out):
    return [o for o in out if isinstance(o, told.PdFlipStoreTold)]


def test_d9_a_single_phase_read_landing_in_the_park_is_admitted_in_this_pass(pp0, monkeypatch):
    """D9 (S103): the non-paced told IS the admission -- a read that landed in the park is admitted like an ack."""
    _single_phase(pp0, monkeypatch)
    cp.reset_rings()                                             # unmeasured: A3 would have held it
    told._commit_park(pp0, [])
    assert cp.stats()["read"] == 1 and cp.stats()["park_admit"] == 1 and cp.stats()["a3_would_hold"] == 1


def test_d8_the_single_phase_publish_puts_the_parked_read_on_the_wire_this_pass(pp0, monkeypatch):
    """D8 (S103): the read that landed in the park is told in THIS pass (A3 held it until the next one)."""
    _single_phase(pp0, monkeypatch)
    cp.reset_rings()
    out = told.pp0_publish(pp0, [])
    assert [t.rid for t in _tolds(out)] == [RID] and pp0._pdflip_store_told[RID] == TOLD
    assert RID not in pp0._pdflip_store_held


# ---------------------------------------------------------------------------
# S103 (20.10.2026): the park's ready set runs in THIS pass; every park and every non-park with a waiter is printed.
#
# Metal rc24 781238542c, P log ..._1020_113244, PP0:
#   11:47:50  '#969N ADMIT slot=0 fwd_ct=7 bs=1 extend=6672 rids=[pdflip-2-141]'   (104192 cached)
#   11:47:52  'S82 COMMIT-PARK n=5 outcome=acks rid=pdflip-2-138 waited_ms=6.8'
#   11:47:52  'S82 PARK-AGE HOLD rids=['pdflip-2-138', 'pdflip-2-139'] reason=price-unmeasured'
#   11:47:55  '#1463 PASS-TAIL pp_rank=0 bs=1 finished=1 recv_ms=0 commit_ms=2608 process_ms=185'
#   11:47:59  '#PGAP pp_rank=0 fwd=9 tokens=84 gpu_gap_ms=3004.9 ... output_commit=2608'   (the 5090 idle 3.0 s)
# front: 'LEG2-FIRST-CONTENT rid=pdflip-2-138 epoch=4 via=after_p', 'P-DRAIN epoch=3 ... queue_at_exit=0',
# 'kind=phase_end action=FLIP_D' -- nobody saw pdflip-2-141's P output before the P phase drained.

RC24_FWD = [(6038, 83328, 2603.9), (18, 140608, 931.1), (769, 249664, 1825.8), (53, 71104, 1531.2),
            (173, 63552, 1192.1), (180, 132736, 1196.0), (215, 71104, 1370.8)]
RC24_LOAD = [(83328, 190.0), (103872, 126.0), (159168, 186.0), (52864, 64.0), (104192, 207.0)]
RC24_TOLD = 103232          # '#1416 STORE-TOLD ANCHOR-CLAMP rid=pdflip-2-138 completed=103360 anchored=103232'


class _Log:
    def __init__(self):
        self.lines = []

    def info(self, fmt, *args):
        self.lines.append(fmt % args)

    warning = info


def test_s103_metal_rc24_1147_the_acked_pair_runs_while_the_6672_set_is_downstream(pp0, monkeypatch):
    """RED on rc25 e1dea5463a: PARK-AGE HOLD price-unmeasured, no Admit, PP0 idles 2608 ms in the receive."""
    pfc._reset_for_tests()
    cp.reset_rings()
    for n_, c_, ms in RC24_FWD:
        pfc.note_batch_cost(n_, c_, ms, 1)
    for n_, ms in RC24_LOAD:
        cp.note_start_loading(n_, ms)
    req = pp0.waiting_queue[0]
    req.origin_input_ids = list(range(RC24_TOLD + 40))
    pp0._pdflip_told_pacing[RID].told = RC24_TOLD
    fb._pp0_open_map(pp0)[RID].told = RC24_TOLD
    pp0._s82_launch_rec = {0: (_time.monotonic() - 0.007, 6672, 104192)}
    n = {"i": 0}

    def harvest(s, now=None):
        n["i"] += 1
        if n["i"] >= 2:
            fb._pp0_open_map(s)[RID].acks.update({1: RC24_TOLD, 2: RC24_TOLD})
        return 0

    log = _Log()
    monkeypatch.setattr(fb, "pp0_harvest", harvest)
    monkeypatch.setattr(cp, "logger", log)
    out = told.pp0_publish(pp0, [])
    assert [a.rid for a in _admits(out)] == [RID] and _admits(out)[0].told == RC24_TOLD
    assert pp0._pdflip_store_told[RID] == RC24_TOLD          # PP0's own plan runs it in this pass
    admit = [ln for ln in log.lines if ln.startswith("S82 PARK-ADMIT")]
    assert len(admit) == 1 and "a3=hold:price-unmeasured" in admit[0]
    assert any(ln.startswith("S82 COMMIT-PARK n=1 outcome=acks") for ln in log.lines)


def test_s103_a_blocking_receive_without_park_names_its_reason(pp0, monkeypatch):
    """rc24 11:40:00 shape (B1): a newcomer arrives in the pass, the acked rid waits, the receive blocks 3380 ms --
    the pass does not park and now says why."""
    log = _Log()
    monkeypatch.setattr(cp, "logger", log)
    _acks_after(monkeypatch, pp0, k=3)
    told.pp0_publish(pp0, [_arrival()])
    np_ = [ln for ln in log.lines if ln.startswith("S82 NO-PARK")]
    assert len(np_) == 1 and "reason=arrivals" in np_[0] and RID in np_[0]
    assert cp.stats()["no_park_arrivals"] == 1 and cp.stats()["n"] == 0


def test_s103_no_trace_without_a_waiter_or_without_a_blocking_receive(pp0, monkeypatch):
    log = _Log()
    monkeypatch.setattr(cp, "logger", log)
    pp0.mbs[0] = None                                          # the receive will not block: no verdict at all
    told._commit_park(pp0, [_arrival()])
    pp0.mbs[0] = _batch()
    pp0.waiting_queue = []                                     # nobody waiting: no trace
    told._commit_park(pp0, [_arrival()])
    assert not [ln for ln in log.lines if ln.startswith("S82 NO-PARK")]


def test_s103_every_park_is_printed():
    """rc24 printed parks n=1..16 only; its parks after 12:09:18 were invisible."""
    cp.reset()
    log = _Log()
    orig = cp.logger
    cp.logger = log
    try:
        v = cp.Verdict(True, "park", (), (RID,))
        for _ in range(40):
            cp.log_park(v, "acks", RID, 0.01, (2, 0))
    finally:
        cp.logger = orig
    assert len([ln for ln in log.lines if ln.startswith("S82 COMMIT-PARK")]) == 40


def _follower(rank):
    return SimpleNamespace(ps=SimpleNamespace(pp_rank=rank, pp_size=3), _pdflip_store_told={}, _pdflip_store_held={},
                           _pdflip_fb_follower=None, _pdflip_told_early={RID: RC24_TOLD})


def test_s103_three_ranks_absorb_the_park_admit_identically(pp0, monkeypatch):
    """Rank uniformity (R11, RAENGE-NIE-UNEINS): the Admit PP0 decides in the park is the ordinary wire object --
    PP1 and PP2 apply exactly PP0's told, one PP0 pass earlier than under A3, nothing decided off PP0."""
    cp.reset_rings()                                             # A3 would hold (price-unmeasured)
    req = pp0.waiting_queue[0]
    req.origin_input_ids = list(range(RC24_TOLD + 40))
    pp0._pdflip_told_pacing[RID].told = RC24_TOLD
    fb._pp0_open_map(pp0)[RID].told = RC24_TOLD
    _acks = {"i": 0}

    def harvest(s, now=None):
        _acks["i"] += 1
        if _acks["i"] >= 2:
            fb._pp0_open_map(s)[RID].acks.update({1: RC24_TOLD, 2: RC24_TOLD})
        return 0

    monkeypatch.setattr(fb, "pp0_harvest", harvest)
    out = told.pp0_publish(pp0, [])
    admits = _admits(out)
    assert [a.rid for a in admits] == [RID]
    ranks = {0: pp0._pdflip_store_told[RID]}
    for rank in (1, 2):
        f = _follower(rank)
        for a in admits:
            told._follower_admit(f, a)
        ranks[rank] = f._pdflip_store_told[RID]
    assert set(ranks.values()) == {RC24_TOLD}, ranks
