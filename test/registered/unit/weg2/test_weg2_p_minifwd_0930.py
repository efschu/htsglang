"""P-MINIFWD (30.09.): no lone mini forwards on P for the two classes that do
not need extra VRAM.

Metal (PP0, y3r ...dauer09292330 / y3t ...0930_000239): P forwards below 1000
tokens at bs 1 cost 1.0-1.6 s each on PP0 (the expert socket of ~4200 spilled
experts alone ~0.8 s); y3r 46 of them, 15-18 % of PP0 prefill time.

* (c) PAGE-END FOLD -- N % 64 == 0 (weg2-30-38/39, weg2-74-*: N = 32832): the
  END-ANCHOR split held the last 4 tokens back (a forward of 1.37-1.57 s) and
  truncated the body, so no queued request joined it and the 60/124-token
  body ran alone too. The CLAIM ANCHOR track puts the recurrent anchor on the
  reader's claim N - 64 inside the last chunk; the fold applies there.
* (a) TOLD WAIT -- weg2-62-92: its 66-token rest ran alone (1144.7 gpu-ms)
  because the told of the queued weg2-62-93 went on the wire one pass later;
  PP0 now waits for that read, bounded by the measured price of a lone rest.

Hermetic, CPU. Red on 5bedac26f1 (no page-end fold, no hold), green with it.
"""

import os
import sys
from types import SimpleNamespace

import pytest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from sglang.srt.environ import envs  # noqa: E402
from sglang.srt.managers import weg2_store_told as told  # noqa: E402
from sglang.srt.weg2 import fork_anchor as fa  # noqa: E402
from sglang.srt.weg2 import tail_handoff as th  # noqa: E402
from test_weg2_tail_fold_h63 import (  # noqa: E402,F401  (fixtures by name)
    PAGE,
    RATIO,
    RID,
    _e2,
    _p_pools,
    _p_req,
    _qsa_allocator,
    p_arena,
)

N_METAL = 32832  # y3r weg2-30-38 / weg2-74-*: 2 x 16384 + 64
CLAIM_METAL = N_METAL - PAGE  # the bigram reader's claim floor_page(N-2)


def _claim_tree(monkeypatch):
    monkeypatch.setenv("SGLANG_WEG2_GROUP", "P")
    return SimpleNamespace(page_size=PAGE, bigram_anchor_exact=True)


def _split(monkeypatch, start, length, total, tree=None):
    import sglang.srt.managers.schedule_policy as sp

    monkeypatch.setattr(sp, "_WEG2_END_ANCHOR", True)  # group P
    adder = SimpleNamespace(rem_chunk_tokens=16384, page_size=PAGE, token_to_kv_pool_allocator=_qsa_allocator(),
                            tree_cache=tree)
    ids = list(range(total))
    req = SimpleNamespace(full_untruncated_fill_ids=ids, origin_input_ids=ids, rid="rid")
    return sp.PrefillAdder._weg2_end_anchor_split(adder, req, start, length)


# ------------------------------------------------------------------ (c) the split
def test_page_end_fold_takes_the_tail_into_the_last_chunk(monkeypatch):
    """The metal chunk [16384, 32832) and a whole-fit [0, 32832): no split,
    not truncated -- base: (32828 - start, True)."""
    tree = _claim_tree(monkeypatch)
    with _e2(fold=True):
        assert _split(monkeypatch, 16384, N_METAL - 16384, N_METAL, tree) == (N_METAL - 16384, False)
        assert _split(monkeypatch, 32768 - 16384 * 2, N_METAL, N_METAL, tree) == (N_METAL, False)
        # the H63 test's page-multiple sizes, now with the claim tree
        for n in (256, 97792):
            start = n - 64 * 3
            assert _split(monkeypatch, start, n - start, n, tree) == (n - start, False)


def test_page_end_keeps_the_split_where_the_track_cannot_reach_the_claim(monkeypatch):
    tree = _claim_tree(monkeypatch)
    with _e2(fold=True):
        # the last chunk starts AT the claim: no grid point inside, anchor would sit at N
        assert _split(monkeypatch, CLAIM_METAL, PAGE, N_METAL, tree) == (PAGE - 4, True)
        # no claim (not group P / no bigram tree): the split, as before
        assert _split(monkeypatch, 16384, N_METAL - 16384, N_METAL, None) == (N_METAL - 4 - 16384, True)
        monkeypatch.setenv("SGLANG_WEG2_GROUP", "D")
        assert _split(monkeypatch, 16384, N_METAL - 16384, N_METAL, tree) == (N_METAL - 4 - 16384, True)
        monkeypatch.setenv("SGLANG_WEG2_GROUP", "P")
        with envs.SGLANG_WEG2_TAIL_FOLD_PAGE_END.override(False):
            assert _split(monkeypatch, 16384, N_METAL - 16384, N_METAL, tree) == (N_METAL - 4 - 16384, True)
    with _e2(fold=False):  # no H63 fold at all: the split of every N
        assert _split(monkeypatch, 16384, N_METAL - 16384, N_METAL, tree) == (N_METAL - 4 - 16384, True)


def test_page_end_fold_is_exactly_where_the_claim_track_lands_on_the_claim():
    """One predicate with ``schedule_batch._weg2_claim_track``: the fold holds
    iff ``fork_anchor.track_target`` moves the step's default anchor N onto
    the claim N - page (grid 64, every page-aligned and unaligned start)."""
    with _e2(fold=True):
        for n in range(128, 1600, 64):
            claim = th.reader_claim_end(n, PAGE, True)
            assert claim == n - PAGE
            for start in list(range(0, n, 16)) + [n - 1]:
                expect = fa.track_target(start, n, claim, 64, n) == claim
                assert th.page_end_fold_applies(n, PAGE, start, claim, grid=64) == expect, (n, start)
                assert th.fold_applies(n, PAGE, start=start, claim=claim, grid=64) == expect, (n, start)


def test_non_page_multiples_are_unchanged():
    with _e2(fold=True):
        for n in (241, 2113, 16449, 16450, 65602):
            assert th.fold_applies(n, PAGE) is True
            assert th.fold_applies(n, PAGE, start=0, claim=None) is True
    with _e2(fold=False):
        assert th.fold_applies(16450, PAGE) is False


def test_arm_fold_registers_the_page_end_part(p_arena, monkeypatch):
    """END-only capture at N % 64 == 0: page_prefix = the claim N - 64, cut =
    N - 4 (the same geometry spec_for gave the split), no stash at c."""
    _kv, _rp, alloc = _p_pools()
    n = 256
    req = _p_req(list(range(n)), 64, n)
    with _e2(fold=True):
        assert th.arm_fold([req], alloc, PAGE, None) == 0  # no claim: the split's form, nothing armed
        assert th.arm_fold([req], alloc, PAGE, None, tree_cache=_claim_tree(monkeypatch)) == 1
    cap = th._CAPTURES[RID]
    assert cap.e1 is False
    assert (cap.spec.page_prefix, cap.spec.cut, cap.spec.n_tokens) == (n - PAGE, n - 4, n)
    assert cap.spec == th.spec_for(RID, list(range(n)), None, PAGE, RATIO)


# ------------------------------------------------------------------ (a) the verdict
try:  # the module the fix adds; on the base every (a) case fails on it by name
    from sglang.srt.weg2 import p_minifwd_hold as mh  # noqa: E402
except ImportError:  # pragma: no cover -- base tree
    mh = None


@pytest.fixture(autouse=True)
def _fresh():
    if mh is not None:
        mh.reset()
    yield
    if mh is not None:
        mh.reset()


def _decide(**kw):
    base = dict(rest=66, queued=["b"], admissible=lambda r: False, open_read=lambda r: True,
                control=False, bound_s=1.1)
    base.update(kw)
    return mh.decide(**base)


def test_verdict_table():
    assert _decide().wait and _decide().rids == ("b",)
    assert _decide(rest=None).reason == "no-final-rest"
    assert _decide(rest=1000).reason == "rest-not-mini"
    assert _decide(control=True).reason == "control"
    assert _decide(admissible=lambda r: True).reason == "joins-anyway"
    assert _decide(open_read=lambda r: False).reason == "no-waiter"
    assert _decide(queued=[]).reason == "no-waiter"
    assert _decide(bound_s=None).reason == "unmeasured"


def test_final_rest():
    assert mh.final_rest(16450, 16384, 16384, split=False) == 66
    assert mh.final_rest(16450, 16384, 16384, split=True) is None  # a split piece admits nobody
    assert mh.final_rest(40000, 16384, 16384, split=False) is None  # not the last piece
    assert mh.final_rest(16384, 16384, 16384, split=False) is None


def test_bound_is_the_measured_lone_rest_price():
    assert mh.wait_bound_s() is None
    mh.note_forward(16384, 1, 2611.8)  # a full chunk: no sample
    mh.note_forward(66, 2, 900.0)  # two forwards folded into one line: no sample
    assert mh.wait_bound_s() is None
    mh.note_forward(66, 1, 1144.7)
    assert mh.wait_bound_s() == pytest.approx(1.1447)
    mh.note_forward(124, 1, 1106.3)
    assert mh.wait_bound_s() == pytest.approx((1144.7 + 1106.3) / 2 / 1000.0)


def test_wait_ends_at_the_told_or_at_the_bound():
    t = [0.0]
    clock = lambda: t[0]  # noqa: E731

    def sleep(s):
        t[0] += s

    calls = {"b": 0}

    def terminable(rid):
        calls[rid] += 1
        return calls[rid] >= 4

    outcome, rid, waited = mh.wait(["b"], terminable, 1.0, clock=clock, sleep=sleep)
    assert (outcome, rid) == ("told", "b") and waited == pytest.approx(3 * mh.POLL_S)
    outcome, rid, waited = mh.wait(["b"], lambda r: False, 0.05, clock=clock, sleep=sleep)
    assert (outcome, rid) == ("timeout", None) and waited == pytest.approx(0.05)


# ------------------------------------------------------------------ (a) the pass
class _Op:
    host_indices = object()


class _Tree:
    """A store read that can terminate after ``polls`` looks."""

    page_size = PAGE

    def __init__(self, polls):
        self.polls = polls
        self.ongoing_prefetch = {}
        self.loaded = {}

    def register(self, rid):
        self.ongoing_prefetch[rid] = SimpleNamespace(operation=_Op())

    def can_terminate_prefetch(self, op, tail_hold=0):
        self.polls -= 1
        return self.polls <= 0

    def check_prefetch_progress(self, rid):
        if rid not in self.ongoing_prefetch:
            return True
        if not self.can_terminate_prefetch(self.ongoing_prefetch[rid].operation):
            return False
        del self.ongoing_prefetch[rid]
        self.loaded[rid] = 0
        return True

    def completed_prefetch_tokens(self, rid):
        return 0

    def pop_prefetch_loaded_tokens(self, rid):
        return self.loaded.pop(rid, 0)


def _pass_scheduler(polls, rest_done=16384, fill=16450):
    s = SimpleNamespace(
        ps=SimpleNamespace(pp_rank=0, pp_size=3, tp_size=1), enable_hicache_storage=True,
        tree_cache=_Tree(polls), waiting_queue=[], pp_flip_counters=None, page_size=PAGE,
        chunked_prefill_size=16384,
        _prefetch_kvcache=lambda req, rematch=True, limit_tokens=None: "issued",
    )
    ids = list(range(fill))
    s.chunked_req = SimpleNamespace(rid="weg2-62-92", full_untruncated_fill_ids=ids, origin_input_ids=ids,
                                    extend_range=SimpleNamespace(start=0, end=rest_done), prefix_indices=[])
    assert told.armed(s)
    s._weg2_told_paced_on = False
    return s


def _queue(s, rid="weg2-62-93"):
    r = SimpleNamespace(rid=rid, prefetch_deferred=None, origin_input_ids=list(range(16450)))
    s.waiting_queue.append(r)
    told.intake(s, r, lambda *a: None)
    s.tree_cache.register(rid)
    return r


@pytest.fixture
def _p_form(monkeypatch):
    import sglang.srt.managers.schedule_policy as sp

    monkeypatch.delenv(told.ENV_ARMED, raising=False)
    monkeypatch.setattr(sp, "_WEG2_END_ANCHOR", False)
    monkeypatch.setattr(mh, "POLL_S", 0.0)
    yield


def test_the_waiters_told_rides_the_tail_pass(_p_form):
    """y3r weg2-62-92: the rest's pass carries 93's told (base: the next pass)."""
    s = _pass_scheduler(polls=3)
    _queue(s)
    mh.note_forward(66, 1, 1144.7)
    wire = told.pp0_publish(s, ["x"])
    assert [type(w).__name__ for w in wire] == ["str", "Weg2StoreTold"]
    assert wire[1].rid == "weg2-62-93"
    assert mh._STATE["told"] == 1 and mh._STATE["timeout"] == 0


def test_no_wait_without_a_measurement_or_a_final_rest(_p_form):
    s = _pass_scheduler(polls=3)
    _queue(s)
    assert told.pp0_publish(s, ["x"]) == ["x"]  # unmeasured: compute at once
    mh.note_forward(66, 1, 1144.7)
    s2 = _pass_scheduler(polls=3, rest_done=0, fill=40000)
    _queue(s2)
    assert told.pp0_publish(s2, ["x"]) == ["x"]  # a middle chunk: nobody could join it
    assert mh._STATE["n"] == 0


def test_no_wait_before_a_flip(_p_form):
    class ReleaseMemoryOccupationReqInput:  # noqa: N801 -- the control kind by name
        pass

    s = _pass_scheduler(polls=3)
    _queue(s)
    mh.note_forward(66, 1, 1144.7)
    wire = told.pp0_publish(s, [ReleaseMemoryOccupationReqInput()])
    assert len(wire) == 1 and mh._STATE["n"] == 0


def test_switch_off_is_the_base(_p_form):
    s = _pass_scheduler(polls=3)
    _queue(s)
    mh.note_forward(66, 1, 1144.7)
    with envs.SGLANG_WEG2_P_MINIFWD_TOLD_WAIT.override(False):
        assert told.pp0_publish(s, ["x"]) == ["x"]


def test_a_read_that_never_ends_costs_at_most_the_bound(_p_form, monkeypatch):
    s = _pass_scheduler(polls=10 ** 9)
    _queue(s)
    mh.note_forward(66, 1, 5.0)  # a 5 ms price: the wait may not exceed it
    wire = told.pp0_publish(s, ["x"])
    assert wire == ["x"]
    assert mh._STATE["timeout"] == 1 and mh._STATE["waited_ms"] <= 5.0 + 5.0
