"""S79 (19.10.2026): co-released fork twins wait behind the OLDER sibling that
computes their deeper shared prefix (p_twin_defer rule 6, TWIN CHAIN).

Metal (NF rc17, P log ..._1019_171636, PP0):
  17:20:23 '#TW TWIN-DEFER rid=pdflip-0-1 len=81725 shared=10747 sources=[pdflip-0-2]'
           (0-4 len=81607 and 0-5 len=53709 the same, shared=10747 with 0-2)
  17:20:42 '#TW TWIN-RELEASE rid=pdflip-0-1 / 0-4 / 0-5 reason=published' -- ONE pass
  17:20:42 'PDFLIP-CLAIM-SHORT rid=pdflip-0-1/0-4/0-5 asked=... kv=111 claimed=0 lost=111 by=mamba'
  17:21:03 '#969 EXTENT ... (pdflip-0-1, 69120, 81725) , (pdflip-0-4, 3584, 7360)'
  front:   'SESSION-PREFIX rid=pdflip-0-4 prev_rid=pdflip-0-1 common=81565'
0-4 shared 81565 leading ids with 0-1 but, released in the same pass, registered its
read at told 3584 and prefilled [3584, 81607) -- 78023 tokens 0-1 published at the same
time; 0-5 (17943 shared with 0-1) prefilled [3584, 10688) again.

Pinned (hermetic, CPU; RED on rc17 7d93470b5e):
  1. the metal shape: of three twins released by one source, the oldest registers, the
     two that share a deeper boundary with it wait behind it ('#TW TWIN-CHAIN');
  2. they are released as TWINS once that sibling finished and settled -- together,
     since their boundary with each other is not deeper than with it;
  3. identical prompts (equal boundary) chain nothing (TW-WAKE y5a stays); twins that
     are identical to EACH OTHER but share less with the source do chain (one computes);
  4. an OLDER sibling in flight (not held, registered in an earlier pass) counts as
     well; a YOUNGER one is never waited for (Prio = Alter);
  5. a candidate without a promise for the band chains nothing;
  6. the Frist bounds the chain: a Frist release never chains;
  7. FLLIPER_PDFLIP_P_TWIN_CHAIN=0 is rc17 byte for byte (all three released).
Both the promise form (rule 5, metal: twin anchor armed on P) and rule 4 run.
"""

from types import SimpleNamespace

import pytest

from flliper.srt.pdflip import p_twin_defer as tw
from flliper.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="stage-a-test-cpu")

MIN = 32          # FLLIPER_PDFLIP_P_TWIN_MIN_TOKENS
CHUNK = 16        # chunked_prefill_size: a whole chunk fits (s0, B] of the source
S0 = 8            # the source's start depth
N_SRC = 120       # source (0-2) prompt
SH_SRC = 40       # every twin shares ids[:40] with the source  (10747 on the metal)
N_A = 160         # 0-1
SH_AB = 150       # 0-4 shares ids[:150] with 0-1               (81565)
SH_AC = 70        # 0-5 shares ids[:70] with 0-1                (17943)


def _ids(prefix, n, salt):
    return list(prefix) + [salt * 100_000 + i for i in range(n - len(prefix))]


BASE = list(range(1000, 1000 + SH_SRC))                 # the shared head of all
SRC = _ids(BASE, N_SRC, 1)
A = _ids(BASE, N_A, 2)
B = _ids(A[:SH_AB], 155, 3)
C = _ids(A[:SH_AC], 140, 4)


class _Sched:
    def __init__(self):
        self.ps = SimpleNamespace(pp_rank=0, pp_size=3, tp_size=1)
        self.waiting_queue = []
        self.chunked_req = None
        self.running_batch = None
        self.page_size = 1
        self.chunked_prefill_size = CHUNK


class _R:
    def __init__(self, rid, ids, head=0):
        self.rid = rid
        self.origin_input_ids = list(ids)
        self.extra_key = None
        self.done = False
        self._prefetch_registered_prefix_len = head
        self.prefix_indices = None
        self.fill_ids = None
        self.output_ids = []

    def finished(self):
        return self.done


class _Clock:
    def __init__(self):
        self.t = 1000.0

    def __call__(self):
        return self.t


def _reset():
    from flliper.srt.pdflip import twin_anchor as ta2

    ta2._reset_for_test()


@pytest.fixture(params=["promise", "gain"])
def pp0(request, monkeypatch):
    monkeypatch.setenv(tw.ENV, "1")
    monkeypatch.setenv(tw.ENV_MIN_TOKENS, str(MIN))
    monkeypatch.setenv("FLLIPER_PDFLIP_GROUP", "P")
    monkeypatch.delenv("FLLIPER_PDFLIP_P_TWIN_CHAIN", raising=False)
    armed = request.param == "promise"
    monkeypatch.setattr(tw, "_twin_anchor_armed", lambda: armed, raising=False)
    clock = _Clock()
    monkeypatch.setattr(tw, "_now", clock)
    _reset()
    s = _Sched()
    s.clock = clock
    yield s
    _reset()


def _admit(sched, req, s0):
    """PP0 sees ``req`` admitted at ``s0`` (in flight, out of the queue)."""
    req.prefix_indices = [0] * s0
    req.fill_ids = list(req.origin_input_ids)
    sched.waiting_queue = [q for q in sched.waiting_queue if q is not req]
    sched.chunked_req = req
    tw.tick(sched)


def _finish_and_settle(sched, req, queued):
    req.done = True
    if sched.chunked_req is req:
        sched.chunked_req = None
    tw.tick(sched)
    assert tw.release_due(sched, set(queued)) == []        # the settle starts here
    for _ in range(3):
        tw.tick(sched)
    sched.clock.t += 1.0


def _metal_intake(sched):
    src = _R("pdflip-0-2", SRC, head=S0)
    a, b, c = _R("pdflip-0-1", A), _R("pdflip-0-4", B), _R("pdflip-0-5", C)
    _admit(sched, src, S0)
    sched.waiting_queue = []
    for t in (a, b, c):
        # the scheduler queues a request after its intake; held twins are no
        # sources of each other: all three wait for 0-2 only (the metal lines)
        assert tw.intake_defer(sched, t) is True
        sched.waiting_queue.append(t)
        assert [x.rid for x in tw.state(sched).waits[t.rid].sources] == ["pdflip-0-2"]
    return src, a, b, c


def test_metal_shape_the_deeper_twins_wait_behind_the_oldest(pp0):
    src, a, b, c = _metal_intake(pp0)
    _finish_and_settle(pp0, src, {a.rid, b.rid, c.rid})
    # rc17: [(a, True), (b, True), (c, True)] -- all three told at the shallow head
    assert tw.release_due(pp0, {a.rid, b.rid, c.rid}) == [(a, True)]
    st = tw.state(pp0)
    assert tw.is_deferred(pp0, b.rid) and tw.is_deferred(pp0, c.rid)
    assert [x.rid for x in st.waits[b.rid].sources] == [a.rid]
    assert st.waits[b.rid].shared == SH_AB
    assert [x.rid for x in st.waits[c.rid].sources] == [a.rid]
    assert st.waits[c.rid].shared == SH_AC
    assert st.n_chain == 2
    assert tw.take_pp0_twin(pp0, a.rid) is True
    assert tw.take_pp0_twin(pp0, b.rid) is False          # no told goes out for the chained


def test_chained_twins_are_released_as_twins_after_the_older_finished(pp0):
    src, a, b, c = _metal_intake(pp0)
    _finish_and_settle(pp0, src, {a.rid, b.rid, c.rid})
    assert tw.release_due(pp0, {a.rid, b.rid, c.rid}) == [(a, True)]
    assert tw.take_pp0_twin(pp0, a.rid) is True
    pp0.waiting_queue = [b, c]
    _admit(pp0, a, SH_SRC - 1)                             # 0-1 reads to B(0-2) and computes
    assert tw.release_due(pp0, {b.rid, c.rid}) == []        # 0-1 still prefilling
    _finish_and_settle(pp0, a, {b.rid, c.rid})
    # b and c share only 70 with each other (= their boundary with 0-1 for c): no chain
    assert tw.release_due(pp0, {b.rid, c.rid}) == [(b, True), (c, True)]
    assert tw.take_pp0_twin(pp0, b.rid) is True
    assert pp0._pdflip_twin_state.twin_pp0 == {c.rid: SH_AC}


def test_identical_prompts_chain_nothing(pp0):
    """TW-WAKE y5a (six identical prompts): equal boundaries -- the twins still
    register together behind the source."""
    src = _R("pdflip-52-75", SRC, head=S0)
    twins = [_R("pdflip-52-%d" % i, SRC) for i in (76, 77, 78)]
    _admit(pp0, src, S0)
    pp0.waiting_queue = []
    for t in twins:
        assert tw.intake_defer(pp0, t) is True
        pp0.waiting_queue.append(t)
    _finish_and_settle(pp0, src, {t.rid for t in twins})
    assert tw.release_due(pp0, {t.rid for t in twins}) == [(t, True) for t in twins]
    assert tw.state(pp0).n_chain == 0


def test_twins_identical_to_each_other_but_not_to_the_source_chain(pp0):
    """They share 100 with the source but all 130 with each other: the oldest
    computes [99, 130) once, the others read it (rc17: three times)."""
    src = _R("pdflip-9-1", SRC, head=S0)
    same = _ids(SRC[:100], 130, 7)
    twins = [_R("pdflip-9-%d" % i, same) for i in (2, 3, 4)]
    _admit(pp0, src, S0)
    pp0.waiting_queue = []
    for t in twins:
        assert tw.intake_defer(pp0, t) is True
        pp0.waiting_queue.append(t)
    _finish_and_settle(pp0, src, {t.rid for t in twins})
    assert tw.release_due(pp0, {t.rid for t in twins}) == [(twins[0], True)]
    st = tw.state(pp0)
    assert [x.rid for x in st.waits["pdflip-9-3"].sources] == ["pdflip-9-2"]
    assert [x.rid for x in st.waits["pdflip-9-4"].sources] == ["pdflip-9-2"]


def test_an_older_sibling_registered_in_an_earlier_pass_counts(pp0):
    """Not co-released: 0-1 (older) left its hold by the Frist while 0-2 still ran,
    so it registered and is in flight, not held, when 0-4 becomes due."""
    src = _R("pdflip-0-2", SRC, head=S0)
    a, b = _R("pdflip-0-1", A), _R("pdflip-0-4", B)
    _admit(pp0, src, S0)
    assert tw.intake_defer(pp0, a) is True
    pp0.waiting_queue = [a]
    pp0.clock.t += 60.0
    assert tw.intake_defer(pp0, b) is True              # a is held: b waits for 0-2 only
    pp0.waiting_queue = [a, b]
    assert [x.rid for x in tw.state(pp0).waits[b.rid].sources] == [src.rid]
    pp0.clock.t += tw.DEFAULT_WAIT_S - 60.0             # a's Frist (from its intake) is up
    assert tw.release_due(pp0, {a.rid, b.rid}) == [(a, False)]
    _finish_and_settle(pp0, src, {a.rid, b.rid})
    assert tw.release_due(pp0, {a.rid, b.rid}) == []     # b chained behind the older a
    assert [x.rid for x in tw.state(pp0).waits[b.rid].sources] == [a.rid]


def test_a_younger_sibling_is_never_waited_for(pp0):
    """Prio = Alter: 0-4 is older than the plain request 0-9 that shares more."""
    src = _R("pdflip-0-2", SRC, head=S0)
    b = _R("pdflip-0-4", B)
    _admit(pp0, src, S0)
    assert tw.intake_defer(pp0, b) is True
    pp0.waiting_queue = [b]
    src.done = True
    pp0.chunked_req = None
    tw.tick(pp0)
    assert tw.release_due(pp0, {b.rid}) == []
    for _ in range(3):
        tw.tick(pp0)
    pp0.clock.t += 1.0
    tw.tick(pp0)                                         # 0-2's settle window is over
    y = _R("pdflip-0-9", B[:152] + [77] * 10)            # younger, shares 152 with 0-4
    assert tw.intake_defer(pp0, y) is False              # no source left: registers plainly
    pp0.waiting_queue = [b, y]
    assert tw.release_due(pp0, {b.rid, y.rid}) == [(b, True)]
    assert tw.state(pp0).n_chain == 0


def test_a_candidate_without_a_promise_chains_nothing(pp0):
    """0-1's registered head already lies at/after B(0-4): it brings 0-4 nothing new."""
    src, a, b, c = _metal_intake(pp0)
    a._prefetch_registered_prefix_len = 10_000
    a.origin_input_ids = A[:SH_AB] + [5]        # its end is the boundary itself ...
    b.origin_input_ids = A[:SH_AB] + [6] * 5    # ... and it already holds past it
    a.prefix_indices = [0] * (SH_AB + 1)
    _finish_and_settle(pp0, src, {a.rid, b.rid, c.rid})
    _admit(pp0, a, SH_AB + 1)                   # s0 >= B: neither its end nor a chunk lands
    pp0.waiting_queue = [b, c]
    out = tw.release_due(pp0, {b.rid, c.rid})
    assert (b, True) in out


def test_the_frist_bounds_the_chain(pp0):
    src, a, b, c = _metal_intake(pp0)
    _finish_and_settle(pp0, src, {a.rid, b.rid, c.rid})
    pp0.clock.t += tw.DEFAULT_WAIT_S                       # the Frist from the intake is up
    out = tw.release_due(pp0, {a.rid, b.rid, c.rid})
    assert [r.rid for r, _t in out] == [a.rid, b.rid, c.rid]
    assert tw.state(pp0).n_chain == 0


def test_switch_off_is_rc17(pp0, monkeypatch):
    monkeypatch.setenv("FLLIPER_PDFLIP_P_TWIN_CHAIN", "0")
    src, a, b, c = _metal_intake(pp0)
    _finish_and_settle(pp0, src, {a.rid, b.rid, c.rid})
    assert tw.release_due(pp0, {a.rid, b.rid, c.rid}) == [(a, True), (b, True), (c, True)]


def test_default_is_on():
    from flliper.srt.environ import envs

    assert envs.FLLIPER_PDFLIP_P_TWIN_CHAIN.get() is True


# ---- S79r (review nf-review-s79-1019): A1, A2a, A2b, H1 ------------------------------


def _chain_metal(sched):
    """The metal shape up to the chain: 0-1 released, 0-4 and 0-5 chained behind it."""
    src, a, b, c = _metal_intake(sched)
    _finish_and_settle(sched, src, {a.rid, b.rid, c.rid})
    assert tw.release_due(sched, {a.rid, b.rid, c.rid}) == [(a, True)]
    tw.take_pp0_twin(sched, a.rid)
    return src, a, b, c


def test_a1_a_chained_sibling_aborted_in_the_queue_releases_its_twins(pp0):
    """'Abort method 1' pops 0-1 from the queue without a finished_reason: finished()
    stays False. rc17 form of rules 1-5 and S79 before the review: 0-4/0-5 held to the
    Frist (119 s). Now: released in the next pass, as ordinary requests."""
    src, a, b, c = _chain_metal(pp0)
    pp0.waiting_queue = [b, c]                     # 0-1 aborted in the queue, never finished
    tw.tick(pp0)
    out = tw.release_due(pp0, {b.rid, c.rid})
    assert [r.rid for r, _t in out] == [b.rid, c.rid]
    assert all(t is False for _r, t in out)
    assert tw.state(pp0).n_gone == 2
    assert not tw.is_deferred(pp0, b.rid) and not tw.is_deferred(pp0, c.rid)


def test_a1_a_gone_source_of_rules_1_to_5_releases_too(pp0):
    """The same class without a chain: the source 0-2 aborted in the queue before it ran."""
    src = _R("pdflip-0-2", SRC, head=S0)
    pp0.waiting_queue = [src]
    b = _R("pdflip-0-4", B)
    assert tw.intake_defer(pp0, b) is True           # provisional behind the queued source
    pp0.waiting_queue = [b]                          # 0-2 aborted
    tw.tick(pp0)
    assert tw.release_due(pp0, {b.rid}) == [(b, False)]


def test_a1_a_live_source_is_not_gone(pp0):
    """A source still in flight (chunked) or just finished (settle) stays a source."""
    src, a, b, c = _chain_metal(pp0)
    pp0.waiting_queue = [b, c]
    _admit(pp0, a, SH_SRC - 1)                       # in flight: chunked_req
    assert tw.release_due(pp0, {b.rid, c.rid}) == []
    assert tw.state(pp0).n_gone == 0


def test_a2a_the_frist_from_the_intake_bounds_two_chain_links(pp0):
    """0-4 chains behind 0-1, then behind 0-3 (older than 0-4, chained behind 0-1 at the
    first link, shares 154 with 0-4); 0-3 never finishes. 0-4 leaves its hold at intake +
    Frist -- not at the second link + Frist (mutant M5)."""
    src = _R("pdflip-0-2", SRC, head=S0)
    a = _R("pdflip-0-1", A)
    a3 = _R("pdflip-0-3", B[:154] + [55] * 6)
    b = _R("pdflip-0-4", B)
    _admit(pp0, src, S0)
    pp0.waiting_queue = []
    for t in (a, a3, b):
        assert tw.intake_defer(pp0, t) is True
        pp0.waiting_queue.append(t)
    t_intake = pp0.clock.t
    _finish_and_settle(pp0, src, {a.rid, a3.rid, b.rid})
    assert tw.release_due(pp0, {a.rid, a3.rid, b.rid}) == [(a, True)]          # link 1
    st = tw.state(pp0)
    assert [x.rid for x in st.waits[a3.rid].sources] == [a.rid]
    assert [x.rid for x in st.waits[b.rid].sources] == [a.rid]
    pp0.waiting_queue = [a3, b]
    _admit(pp0, a, SH_SRC - 1)
    _finish_and_settle(pp0, a, {a3.rid, b.rid})
    assert tw.release_due(pp0, {a3.rid, b.rid}) == [(a3, True)]                # link 2 for b
    assert [x.rid for x in st.waits[b.rid].sources] == [a3.rid]
    _admit(pp0, a3, SH_SRC - 1)                      # 0-3 runs (its promise holds) and never finishes
    pp0.clock.t = t_intake + tw.DEFAULT_WAIT_S - 0.5
    assert tw.release_due(pp0, {b.rid}) == []
    pp0.clock.t = t_intake + tw.DEFAULT_WAIT_S + 0.5
    assert tw.release_due(pp0, {b.rid}) == [(b, False)]                         # the Frist


def test_a2b_a_candidate_without_an_intake_age_never_chains(pp0):
    """0-9 never passed PP0's intake (no age): it is no chain source (mutant M2)."""
    src = _R("pdflip-0-2", SRC, head=S0)
    b = _R("pdflip-0-4", B)
    _admit(pp0, src, S0)
    assert tw.intake_defer(pp0, b) is True
    y = _R("pdflip-0-9", B[:152] + [66] * 8)          # deeper, in the queue, no intake age
    pp0.waiting_queue = [y, b]
    _finish_and_settle(pp0, src, {b.rid, y.rid})
    assert tw.release_due(pp0, {b.rid, y.rid}) == [(b, True)]
    assert tw.state(pp0).n_chain == 0


def test_h1_a_held_twin_is_no_chain_source(pp0):
    """0-5 shares 152 with the held (chained) 0-4 but 150 with the released 0-1: it waits
    behind 0-1, never behind a twin that computes nothing yet (mutant M7)."""
    src = _R("pdflip-0-2", SRC, head=S0)
    a, b = _R("pdflip-0-1", A), _R("pdflip-0-4", B)
    c = _R("pdflip-0-5", B[:152] + [44] * 8)
    _admit(pp0, src, S0)
    pp0.waiting_queue = []
    for t in (a, b, c):
        assert tw.intake_defer(pp0, t) is True
        pp0.waiting_queue.append(t)
    _finish_and_settle(pp0, src, {a.rid, b.rid, c.rid})
    assert tw.release_due(pp0, {a.rid, b.rid, c.rid}) == [(a, True)]
    st = tw.state(pp0)
    assert [x.rid for x in st.waits[c.rid].sources] == [a.rid]


# ---- S79r B1 (Nach-Review): the post-wake settle (#1471) is in flight ----------------
# Reviewer test /tmp/rv79c-tests/test_rv79c_post_wake_settle_1019.py, expectation turned:
# a LIVE source parked in pdflip_post_wake_settle (released from the dormant hold at the
# wake with a short read, parked until its re-read lands) is no GONE source.


def _held_twin_b1(sched):
    src = _R("pdflip-0-2", SRC, head=S0)
    sched.waiting_queue = [src]
    b = _R("pdflip-0-4", B)
    assert tw.intake_defer(sched, b) is True
    return src, b


def test_b1_dormant_hold_source_is_not_gone(pp0):
    src, b = _held_twin_b1(pp0)
    pp0.waiting_queue = [b]
    pp0.pdflip_dormant_hold = [src]
    tw.tick(pp0)
    assert tw.release_due(pp0, {b.rid}) == [] and tw.state(pp0).n_gone == 0


def test_b1_post_wake_settle_source_is_not_gone(pp0):
    src, b = _held_twin_b1(pp0)
    pp0.waiting_queue = [b]
    pp0.pdflip_post_wake_settle = [src]       # live: back to the queue once its read lands
    tw.tick(pp0)
    assert tw.release_due(pp0, {b.rid}) == []
    assert tw.state(pp0).n_gone == 0
    assert tw.is_deferred(pp0, b.rid)
