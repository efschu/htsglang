"""LBV-ADMIT (NF h16h 853534a9ff, P, 14.10. 19:33:48Z, deadman 19:36;
P.log ``boot_weg2_..._853534a9ff_1014_180031.P.log`` 293940-294223).

The burst of four load-back requests (weg2-70-656..659) out of one collect window. PP0 decided verdicts for 659,
658 and 657 (``LB-VERDICT VERDICT`` 293693/293714/293904) -- each one a load-back on PP0 too. -656 is -657's twin
(PP0 ``#TW TWIN-RELEASE ... shared=98144``, its prefetch started late: ``PLE-PREFETCH ... waited_ms``): -657's
load-back left the shared prefix DEVICE-resident on PP0 (``MATCH-CENSUS-DEEP rid=weg2-70-656 reached=98112``) but
host-only on PP1/PP2, who had evicted for it (``reached=94528 prefix=98112``, ``MAMBA-HOST-RESUME ... this match
triggers load_back``). PP0 admitted -656 from its device (``#969N ADMIT slot=1 fwd_ct=113 rids=[weg2-70-656]``) with
no hold and no verdict; PP1/PP2 held it for a verdict that never came (``#queue-req: 1``). The slot-1 batch existed
on PP0 only -> PP0 in ``recv_object[src=2]`` for its output, PP1/PP2 in their chain receive -> ring wait, silent
until the deadman.

These tests run the shipped module (``weg2/pp_loadback_verdict.py``) on three stage doubles, with the scheduler's
admission gate and the ``load_back`` seam reduced to their decisions (pinned against the sources below).
RED without the fix (``SGLANG_WEG2_LBV_ADMIT=0`` = the load-back-only trigger of 853534a9ff), GREEN with it. One
mutant per new decision.
"""

import inspect
import logging
import os
import types

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

import pytest

from sglang.srt.weg2 import pp_loadback_verdict as LBV
from sglang.srt.weg2.pp_slot_fidelity import FLOOR_LOCAL_PP_ATTR

# the h16h burst, queue order as on metal (PP0 verdicts 659, 658, 657 in that order)
R659, R658, R657, R656 = "weg2-70-659", "weg2-70-658", "weg2-70-657", "weg2-70-656"
BURST = [R659, R658, R657, R656]
# own host hit per stage (P.log: LB-VERDICT kv_tokens; -656: PP0 reached=98112 = its prefix, PP1/PP2 reached=94528)
HOST_PP0 = {R659: 121600, R658: 90816, R657: 90240, R656: 0}
HOST_FOLLOWER = {R659: 121600, R658: 90816, R657: 90240, R656: 98112 - 94528}


@pytest.fixture(autouse=True)
def _env(monkeypatch):
    for k in (LBV.ENV, LBV.ADMIT_ENV, LBV.ROW_YIELD_ENV, "SGLANG_PP_ROW_AUTHORITY"):
        monkeypatch.delenv(k, raising=False)


class Stage:
    """One P stage: its queue, its own host hits, PP0-only deferrals (the twin wait), and the two seams."""

    def __init__(self, rank, host, defer=None, room=lambda kv, rid: True, hold_breaks=False):
        self.rank = rank
        self.tree = types.SimpleNamespace()
        setattr(self.tree, FLOOR_LOCAL_PP_ATTR, True)
        # carrierless P: no #631 row (no ROW_ONLY_ATTR, no pp_flip_counters) -- the h16h form
        self.sched = types.SimpleNamespace(ps=types.SimpleNamespace(pp_size=3, pp_rank=rank), tree_cache=self.tree)
        assert LBV.arm(self.sched)
        self.host = dict(host)
        self.defer = defer or (lambda stage, rid: False)
        self.room = room
        self.hold_breaks = hold_breaks  # mutant switch for the gate's loop semantics
        self.queue = []

    def plan(self):
        """``_get_new_batch_prefill_raw`` reduced to its decisions: deferrals (continue), the LBV-ADMIT gate
        (scheduler.py, before add_one_req; a deferral), add_one_req's load-back seam (URC.load_back: hold ->
        WEG2-LOADBACK-WAIT NO_TOKEN -> break)."""
        admitted = []
        for rid in list(self.queue):
            if self.defer(self, rid):
                continue
            kv = int(self.host.get(rid, 0))
            req = types.SimpleNamespace(rid=rid, pp_load_back_extent=kv, host_hit_length=kv)
            if LBV.admit_holds(self.sched, req, pp0_room=self.room):
                if self.hold_breaks:
                    break
                continue
            try:
                if kv:
                    # URC.load_back: `if _lbv_gated and not _lbv.live(self, req): hold ... return False`
                    if LBV.gated(self.tree, req) and not LBV.live(self.tree, req):
                        LBV.hold(self.tree, rid, kv, lambda: self.room(kv, rid))
                        break  # applied=0 under an adopted anchor -> NO_TOKEN -> the loop breaks
                    LBV.consume(self.tree, rid)
                admitted.append(rid)
            finally:
                LBV.admit_done(req)
        for rid in admitted:
            self.queue.remove(rid)
        return admitted


def _twin_wait(stage, rid):
    """PP0's ``p_twin_defer``: -656 waits until its twin -657 has run (TWIN-RELEASE reason=published)."""
    return stage.rank == 0 and rid == R656 and R657 in stage.queue


def _h16h_stages(**kw):
    stages = [Stage(0, HOST_PP0, defer=_twin_wait, **kw), Stage(1, HOST_FOLLOWER, **kw), Stage(2, HOST_FOLLOWER, **kw)]
    for s in stages:
        s.queue = list(BURST)
    return stages


def _run(stages, passes=12):
    """Pass n: every stage absorbs the verdicts PP0 put on the wire at this pass's intake, then plans. A pass
    whose batches differ is the h16h split: PP0's slot holds a batch PP2 never ran, so PP0 waits in
    ``recv_object[src=2]`` for an output that is never sent while PP1/PP2 wait for PP0's next chain send."""
    intake, history = [], []
    for n in range(passes):
        for s in stages:
            for v in intake:
                LBV.absorb(s.tree, v)
        batches = [s.plan() for s in stages]
        history.append(batches)
        if any(b != batches[0] for b in batches[1:]):
            return "RING-WAIT", n, history
        intake = LBV.take(stages[0].tree)
    return "OK", None, history


# ---------------------------------------------------------------------------
# the repro
# ---------------------------------------------------------------------------


def test_h16h_red_without_the_fix(monkeypatch):
    """853534a9ff's trigger (own host hit): PP0 admits -656 from its device, PP1/PP2 hold it -> ring wait."""
    monkeypatch.setenv(LBV.ADMIT_ENV, "0")
    verdict, n, hist = _run(_h16h_stages())
    assert verdict == "RING-WAIT", hist
    pp0, pp1, pp2 = hist[n]
    assert pp0 == [R656] and pp1 == [] and pp2 == [], hist[n]
    # the three earlier members went through their verdicts on every stage alike
    assert [h[0] for h in hist[:n] if h[0]] == [[R659], [R658], [R657]]


def test_h16h_green_with_the_fix():
    stages = _h16h_stages()
    verdict, n, hist = _run(stages)
    assert verdict == "OK", hist
    for k in range(3):
        assert sorted(r for h in hist for r in h[k]) == sorted(BURST)
        assert stages[k].queue == []
    # -656 is admitted in the same pass on every stage, after its twin
    i656 = next(i for i, h in enumerate(hist) if R656 in h[0])
    i657 = next(i for i, h in enumerate(hist) if R657 in h[0])
    assert i656 > i657 and hist[i656] == [[R656]] * 3
    assert not any(LBV.released(s.tree, r) for s in stages for r in BURST)


def test_a_follower_without_a_host_hit_does_not_run_ahead_of_pp0(monkeypatch):
    """The mirror image of h16h, the same rank-local trigger: PP0 loads back (holds a pass), a follower holds the
    prefix on its device. Without the fix the follower admits a pass early (#1004); with it all wait for PP0."""
    def stages():
        out = [Stage(0, {"r": 50000}), Stage(1, {"r": 0}), Stage(2, {"r": 50000})]
        for s in out:
            s.queue = ["r"]
        return out

    assert _run(stages())[0] == "OK"
    monkeypatch.setenv(LBV.ADMIT_ENV, "0")
    verdict, n, hist = _run(stages())
    assert verdict == "RING-WAIT" and hist[n] == [[], ["r"], []]


# ---------------------------------------------------------------------------
# one mutant per new decision
# ---------------------------------------------------------------------------


def _pp0_room_decides(monkeypatch=None):
    """D2: PP0 releases its OWN load-back only when its room holds it."""
    s = Stage(0, {"r": 90240}, room=lambda kv, rid: False)
    s.queue = ["r"]
    s.plan()
    return LBV.take(s.tree) == []


def test_d2_pp0_room_decides_its_own_load_back(monkeypatch):
    assert _pp0_room_decides()
    s = Stage(0, {"r": 90240}, room=lambda kv, rid: True)
    s.queue = ["r"]
    s.plan()
    assert [v.rid for v in LBV.take(s.tree)] == ["r"]


def test_d2_mutant_pp0_ignores_its_own_load_back(monkeypatch):
    monkeypatch.setattr(LBV, "_pp0_load_back_tokens", lambda req: 0)
    assert not _pp0_room_decides(), "the mutant (release without PP0's room) must be caught"


def _second_load_back_waits():
    """D3: a second load-back verdict of the same pass is judged on top of the first one's rows."""
    seen = []

    def room(kv, rid):
        seen.append(kv)
        return kv <= 100000

    s = Stage(0, {R658: 90816, R657: 90240}, room=room)
    s.queue = [R658, R657]
    s.plan()
    first = [v.rid for v in LBV.take(s.tree)]
    s.plan()
    second = [v.rid for v in LBV.take(s.tree)]
    return first == [R658] and second == [R657] and seen[1] == 90816 + 90240


def test_d3_cumulative_room_per_pass():
    assert _second_load_back_waits()


def test_d3_mutant_room_judged_per_request(monkeypatch):
    class _NoAccum(LBV._State):
        __slots__ = ()

        def __setattr__(self, name, value):
            if name == "pass_lb":
                value = 0
            super().__setattr__(name, value)

    monkeypatch.setattr(LBV, "_State", _NoAccum)
    assert not _second_load_back_waits(), "the mutant (each load-back judged alone) must be caught"


def _skip_not_stop(hold_breaks):
    """D4: the gate is a deferral (continue), never a stop (break). PP0 twin-defers A (PP0-only deferral), the
    followers hold A for want of a verdict; B behind it is released and must run on every stage alike."""
    def defer(stage, rid):
        return stage.rank == 0 and rid == "A" and stage.twin_pending

    stages = [Stage(k, {}, defer=defer, hold_breaks=hold_breaks) for k in range(3)]
    for s in stages:
        s.queue = ["A", "B"]
        s.twin_pending = True
    return _run(stages, passes=3)[0]


def test_d4_the_gate_skips():
    assert _skip_not_stop(hold_breaks=False) == "OK"


def test_d4_mutant_the_gate_breaks_the_loop():
    assert _skip_not_stop(hold_breaks=True) == "RING-WAIT", "the mutant (hold = break) must be caught"


def _row_form_untouched():
    """D5: the row form is untouched -- the row is the verdict, no stage holds an admission."""
    from sglang.srt.weg2 import p_row_authority as prow

    s = Stage(1, {"r": 3584})
    setattr(s.sched, prow.ROW_ONLY_ATTR, True)
    req = types.SimpleNamespace(rid="r", pp_load_back_extent=3584, host_hit_length=3584)
    return not LBV.admit_armed(s.sched) and not LBV.admit_holds(s.sched, req)


def test_d5_row_form_untouched():
    assert _row_form_untouched()


def test_d5_mutant_gate_on_the_row_form(monkeypatch):
    monkeypatch.setattr(LBV, "_carrierless", lambda sched, st: True)
    assert not _row_form_untouched(), "the mutant (admission gate on the row form) must be caught"


def _spent_on_every_stage():
    """D6: the verdict is spent AT the gate on every stage alike -- a request that comes back (void, retract) needs
    PP0's next verdict on every stage, never a leftover release on the stages that had no host hit."""
    stages = [Stage(0, {"r": 0}), Stage(1, {"r": 3584}), Stage(2, {"r": 0})]
    for s in stages:
        s.queue = ["r"]
    _run(stages, passes=2)
    return all(s.queue == [] for s in stages) and not any(LBV.released(s.tree, "r") for s in stages)


def test_d6_spent_at_the_gate():
    assert _spent_on_every_stage()


def test_d6_mutant_spent_only_by_a_load_back(monkeypatch):
    orig = LBV.admit_holds

    def keep_release(sched, req, pp0_room=None):
        st = LBV.state(sched.tree_cache)
        had = str(req.rid) in st.released
        out = orig(sched, req, pp0_room)
        if had:
            st.released.add(str(req.rid))  # the mutant: only the load-back seam's consume spends it
        return out

    monkeypatch.setattr(LBV, "admit_holds", keep_release)
    assert not _spent_on_every_stage(), "the mutant (release left on the stages without a load-back) must be caught"


# ---------------------------------------------------------------------------
# instrument, PP0's room helper, and the production seams
# ---------------------------------------------------------------------------


def test_hold_is_said_once_per_rid_without_a_shared_cap(caplog):
    s = Stage(2, {})
    caplog.set_level(logging.INFO, logger=LBV.__name__)
    for i in range(20):
        req = types.SimpleNamespace(rid=f"r{i}")
        assert LBV.admit_holds(s.sched, req)
        assert LBV.admit_holds(s.sched, req)
    holds = [r for r in caplog.records if "LB-VERDICT HOLD" in r.getMessage()]
    assert len(holds) == 20  # the h16h log fell silent after 16 lines shared by HOLD and RELEASE


def test_weg2_lbv_pp0_room_uses_the_seam_decision():
    """URC.weg2_lbv_pp0_room on the 657 numbers of the h16h log (SF LOADBACK-ROOM SAME-PASS kv_tokens=90240
    floor=45248 avail=45248 evictable=197568): the local-PP eviction makes the room."""
    from sglang.srt.mem_cache.unified_radix_cache import UnifiedRadixCache

    class Alloc:
        def __init__(self):
            self.avail = 45248

        def available_size(self):
            return self.avail

    alloc = Alloc()

    def evict(params):
        alloc.avail += int(params.num_tokens)
        return types.SimpleNamespace(num_tokens_evicted=int(params.num_tokens))

    tree = types.SimpleNamespace(token_to_kv_pool_allocator=alloc, uniform_avail_floor=45248,
                                 evictable_size=lambda: 197568, evict=evict)
    setattr(tree, FLOOR_LOCAL_PP_ATTR, True)
    assert UnifiedRadixCache.weg2_lbv_pp0_room(tree, 90240, R657) is True
    assert alloc.avail >= 90240
    tree.evictable_size = lambda: 0
    alloc.avail = 45248
    assert UnifiedRadixCache.weg2_lbv_pp0_room(tree, 90240, R657) is False
    tree.uniform_avail_floor = None
    with pytest.raises(RuntimeError, match="#1045 UNIFORM FLOOR ABSENT AT LB-VERDICT ADMIT"):
        UnifiedRadixCache.weg2_lbv_pp0_room(tree, 90240, R657)


def test_pp0_without_a_room_helper_stops_by_name():
    s = Stage(0, {"r": 1000})
    req = types.SimpleNamespace(rid="r", pp_load_back_extent=1000, host_hit_length=1000)
    with pytest.raises(RuntimeError, match="LB-VERDICT ADMIT rid=r"):
        LBV.admit_holds(s.sched, req)


def test_scheduler_gate_sits_before_add_one_req_and_skips():
    from sglang.srt.managers import scheduler as S

    src = inspect.getsource(S.Scheduler._get_new_batch_prefill_raw)
    i_gate = src.index("if _weg2_lbv.admit_holds(self, req):")
    i_add = src.index("res = adder.add_one_req(")
    i_done = src.index("_weg2_lbv.admit_done(req)")
    assert i_gate < i_add < i_done
    # after every deferral of the loop (the last one before the #791 block is the W102 vision skip)
    assert src.index('_note_skip("weg2_vision_skip_refuted"') < i_gate
    tail = src[src.index("if _lbv_held:"):]
    assert tail.index('_note_skip("weg2_lbv_unreleased", req.rid)') < tail.index("continue") < tail.index("break")


def test_load_back_seam_reads_live_and_retreat_reads_live():
    from sglang.srt.mem_cache.unified_radix_cache import UnifiedRadixCache
    from sglang.srt.weg2 import prefill_oom_retreat as P

    src = inspect.getsource(UnifiedRadixCache.load_back)
    assert "if _lbv_gated and not _lbv.live(self, req):" in src
    assert "_lbv.gated(tree, req) and _lbv.live(tree, req)" in inspect.getsource(P._lb_verdict_released)
    room = inspect.getsource(UnifiedRadixCache.weg2_lbv_pp0_room)
    for term in ("_form_a_load_back_floor(self, floor, kv_tokens, rid)", "_sf.agreed_floor(self, floor)",
                 "_sf.local_pp_room(self, kv_tokens, floor, rid)"):
        assert term in room


def test_default_on():
    assert LBV.admit_enabled()
