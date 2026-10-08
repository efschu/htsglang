# SPDX-License-Identifier: Apache-2.0
"""D-HANDBACK-DEFER (dual D): a hand-back whose tail P has not made readable yet is
DEFERRED, not refused -- no second P prefill of the tail, no 9-40 s frozen stream.

Metal gmps13 (boot ...dual1mpsleepbar1fs10020145_4a632a3a94), pdflip-0-19: P's leg 1
N=19869, 17425 cached, tail 2443; D's first presence probe with P's hand-off keys
answered covered=2443 pages=0 present=False -> X-GATE W31 -> W50 RESUME-VIA-P (P
prefilled the 2443-token tail again) -> D resumed 29.8 s later, when the probe
answered present. The cached negative verdict made the later attempts refuse
without asking.

DANGER DIRECTIONS, one test + one mutant each (asserted in-suite):
* the first W31 of a dual-D request defers (mark set), a second W31 refuses (spent);
  off the dual D the W31 refuses as before;
* the mark is a VOTE into the existing X-completion arm (MIN-reduced) inside the
  length-priced bound -- the group defers while any rank waits, and prices once the
  bound is past;
* the deferred hand-back re-issues its read on a pass-counted back-off and drops
  the cached negative presence verdict before each re-issue -- so the tail that
  lands is seen (pdflip-0-19: present=False on the first pass, True later);
* the scheduler wiring: the defer sits between the W31 verdict and the refusal,
  the retry beside _retry_deferred_prefetches, the vote in _pdflip_store_read_is_pending.
"""
from __future__ import annotations

import inspect
import os
import textwrap
import types

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

import pytest

from flliper.srt.managers import scheduler as SC
from flliper.srt.managers import tp_head_congruence as THC
from flliper.srt.pdflip import dual_handback_defer as HB
from flliper.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=3, suite="base-a-test-cpu")

DUAL_D = {"FLLIPER_PDFLIP_DUAL_LAYOUT": "1", "FLLIPER_PDFLIP_GROUP": "D"}
N, CACHED, TAIL = 19869, 17425, 2443          # pdflip-0-19


@pytest.fixture()
def dual_d(monkeypatch):
    for k, v in DUAL_D.items():
        monkeypatch.setenv(k, v)


class _Clock:
    def __init__(self, t=1000.0):
        self.t = float(t)

    def __call__(self):
        return self.t


def _req(rid="pdflip-0-19", seq=19):
    return types.SimpleNamespace(rid=rid, kv_arrival_seq=seq, full_untruncated_fill_ids=list(range(N)),
                                 origin_input_ids=list(range(N)), _pp_store_presence_cache=None,
                                 _pdflip_store_match_cache=None)


class _Store:
    """The arena as D's probe sees it: P's tail pages land at some point."""

    def __init__(self):
        self.tail_landed = False
        self.asked = 0

    def probe(self, tokens, last_hash, prefix_keys, page_keys=None):
        self.asked += 1
        return len(page_keys or ()) if self.tail_landed else 0


def _sched(store, waiting, monkeypatch):
    """The real H108 presence verdict (cached per request) behind a stand-in
    _prefetch_kvcache: present -> the read registers ('issued'), absent -> declined."""
    monkeypatch.setattr(SC, "_pdflip_presence_keys", lambda sched, req, m, n: (["k%d" % i for i in range(TAIL)],
                                                                            "handoff"))
    s = types.SimpleNamespace(waiting_queue=list(waiting), page_size=1, tree_cache=None, verdicts=[])

    def _prefetch_kvcache(req):
        present = SC._pdflip_store_presence(s, req, store.probe, list(range(TAIL + 1)), None, None, CACHED, 0)
        v = "issued" if present else "declined:store_absent"
        s.verdicts.append(v)
        return v

    s._prefetch_kvcache = _prefetch_kvcache
    return s


# -- the first W31 defers, the second refuses -------------------------------------

def test_first_w31_of_a_dual_d_handback_defers_second_refuses(dual_d, caplog):
    caplog.set_level("WARNING")
    clock = _Clock()
    req = _req()
    req._pp_store_presence_cache = ((0, CACHED, TAIL + 1, TAIL), False)     # the cached negative of pass 0
    assert HB.begin(req, TAIL + 1, now=clock) is True, "deferred, not refused"
    assert req._pp_store_presence_cache is None, "the negative presence verdict is dropped"
    clock.t += 21.5
    assert HB.begin(req, TAIL + 1, now=clock) is False, "spent: the refusal follows"
    msgs = [r.getMessage() for r in caplog.records if HB.LINE in r.getMessage()]
    assert "D-HANDBACK-DEFER n=" in msgs[0] and "passes=0 ms=0 tail=2444" in msgs[0] and "state=begin" in msgs[0]
    assert "ms=21500" in msgs[1] and "state=refused" in msgs[1]


@pytest.mark.parametrize("env", [{}, {"FLLIPER_PDFLIP_GROUP": "D"},
                                 {"FLLIPER_PDFLIP_DUAL_LAYOUT": "1", "FLLIPER_PDFLIP_GROUP": "P"}],
                         ids=["no-pdflip", "flip-D", "dual-P"])
def test_off_the_dual_d_a_w31_refuses_as_before(env):
    assert HB.begin(_req(), TAIL + 1, env=env) is False and getattr(_req(), HB.MARK_ATTR, None) is None


# -- the vote: MIN-reduced, inside the length-priced bound -------------------------

def _pending_host(req):
    tc = types.SimpleNamespace(prefetch_timeout_base=2.0, prefetch_timeout_per_page=1.0 / 1024, page_size=1,
                               ongoing_prefetch={})
    h = types.SimpleNamespace(tree_cache=tc, waiting_queue=[req])
    for name in ("_pdflip_store_read_is_pending", "_deferred_prefetch_bound_s",
                 "_pdflip_local_store_read_pending_ages"):
        setattr(h, name, types.MethodType(getattr(SC.Scheduler, name), h))
    return h


def test_the_marked_handback_votes_pending_until_its_read_is_issued(dual_d):
    req = _req()
    h = _pending_host(req)
    assert not h._pdflip_store_read_is_pending(req), "unmarked: nothing pending (old behaviour)"
    HB.begin(req, TAIL + 1)
    assert h._pdflip_store_read_is_pending(req)
    ages = h._pdflip_local_store_read_pending_ages([req.rid])
    assert req.rid in ages, "this rank's vote enters the packed MIN reduce"
    bound = h._deferred_prefetch_bound_s(N)
    assert 21.0 < bound < 22.0, "2 s + 19869 pages x 1/1024 s: the existing length-priced bound"
    assert THC.x_completion_verdict(0, bound) == THC.X_DEFER
    assert THC.x_completion_verdict(int(bound * 1000) + 1, bound) == THC.X_BOUND_EXPIRED
    getattr(req, HB.MARK_ATTR)["issued"] = True
    assert not h._pdflip_store_read_is_pending(req), "the read took over (ongoing_prefetch is the pending term)"


def test_the_vote_stops_on_its_own_past_the_bound(dual_d):
    clock = _Clock()
    req = _req()
    HB.begin(req, TAIL + 1, now=clock)
    assert HB.pending(req, 21.4, now=clock)
    clock.t += 21.5
    assert not HB.pending(req, 21.4, now=clock), "a windowed gate (unbounded) cannot wedge on it"
    assert not HB.pending(req, 0.0, now=clock), "an unpriceable bound is not a wait"


def test_one_rank_still_waiting_keeps_the_group_deferring():
    # MIN over the ranks' votes: a rank whose read is in hand votes neutral
    payload_a = THC.build_x_pending_payload(["pdflip-0-19"], {"pdflip-0-19": 1200}, slots=4)
    payload_b = THC.build_x_pending_payload(["pdflip-0-19"], {}, slots=4)
    assert min(payload_a[0], payload_b[0]) == 1200


# -- the pdflip-0-19 sequence: present=False first, present=True later ---------------

def test_pdflip_0_19_absent_first_pass_then_admitted_without_a_second_p_prefill(dual_d, monkeypatch, caplog):
    caplog.set_level("WARNING")
    store = _Store()
    req = _req()
    s = _sched(store, [req], monkeypatch)
    clock = _Clock()
    # pass 0: the X gate's W31 on the first probe's present=False
    assert s._prefetch_kvcache(req) == "declined:store_absent"
    assert HB.begin(req, TAIL + 1, now=clock) is True
    # passes 1..3: still absent, deferred, the read is re-asked on the back-off passes 1 and 2
    for _ in range(3):
        clock.t += 0.05
        assert HB.retry(s, now=clock) == 0
        assert HB.pending(req, 21.4, now=clock)
    asked_absent = store.asked
    assert asked_absent >= 3, "re-asked, not answered from the cached negative"
    # P's tail lands; the next due pass (4th) sees it and the read registers
    store.tail_landed = True
    clock.t += 0.05
    assert HB.retry(s, now=clock) == 1 and s.verdicts[-1] == "issued"
    assert not HB.pending(req, 21.4, now=clock)
    HB.note_admit(req, now=clock)
    msgs = [r.getMessage() for r in caplog.records if HB.LINE in r.getMessage()]
    assert any("state=read" in m and "passes=4" in m for m in msgs), msgs
    assert any("state=admitted" in m and "tail=2444" in m for m in msgs), msgs
    assert not any("state=refused" in m for m in msgs)


def test_the_back_off_is_pass_counted():
    st = {"retry_seen": 0}
    due = [HB._due(st) for _ in range(40)]
    assert [i + 1 for i, d in enumerate(due) if d][:7] == [1, 2, 4, 8, 16, 24, 32]


# -- mutants ----------------------------------------------------------------------

def test_the_cached_negative_kept_mutant_never_sees_the_tail(dual_d, monkeypatch, caplog):
    monkeypatch.setattr(HB, "forget_presence", lambda req: None)
    with pytest.raises(AssertionError):
        test_pdflip_0_19_absent_first_pass_then_admitted_without_a_second_p_prefill(dual_d, monkeypatch, caplog)


def test_the_refuse_at_once_mutant_is_the_gmps13_refusal(dual_d, monkeypatch, caplog):
    monkeypatch.setattr(HB, "begin", lambda req, tail, now=None, env=None: False)
    with pytest.raises(AssertionError):
        test_pdflip_0_19_absent_first_pass_then_admitted_without_a_second_p_prefill(dual_d, monkeypatch, caplog)


def test_the_no_vote_mutant_lets_the_gate_price_at_once(dual_d, monkeypatch):
    src = textwrap.dedent(inspect.getsource(SC.Scheduler._pdflip_store_read_is_pending))
    fixed = "if _pdflip_hbd.pending(req, _hb_bound):"
    assert src.count(fixed) == 1, "the vote moved -- re-aim the mutant"
    ns = dict(vars(SC))
    exec(compile(src.replace(fixed, "if False:"), SC.__file__, "exec"), ns)
    monkeypatch.setattr(SC.Scheduler, "_pdflip_store_read_is_pending", ns["_pdflip_store_read_is_pending"])
    with pytest.raises(AssertionError):
        test_the_marked_handback_votes_pending_until_its_read_is_issued(dual_d)


# -- wiring -----------------------------------------------------------------------

def test_the_scheduler_wiring():
    src = inspect.getsource(SC.Scheduler)
    i_ref = src.index("if self._pdflip_x_refuses(req, _head_inputs):")
    i_beg = src.index("_pdflip_hbd.begin(", i_ref)
    i_app = src.index("_x_refused.append(req)", i_ref)
    assert i_ref < i_beg < i_app, "the defer sits between the W31 verdict and the refusal"
    i_retry = src.index("self._retry_deferred_prefetches()")
    assert src.index("_pdflip_hbd.retry(self)", i_retry) - i_retry < 400
    assert "_pdflip_hbd.note_admit(req)" in src
