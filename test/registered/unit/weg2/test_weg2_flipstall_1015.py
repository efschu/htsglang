"""FLIPSTALL-1015: a slow D park must not undo itself, and a clean park walks no L3.

Metal (NF h16n 17c2563647, boot ...042125, D log, epoch 90):

* 06:56:07.914 the front sends ``/weg2/park_running`` to D (PARK-IMMEDIATE over-x).
* D's park took 31.3 s on the scheduler thread under the 06:55Z host swap
  storm: ``WEG2-FLIPCYCLE stage=park epoch=90 n=0 ms=31311 sub=...,
  yield:21732,l3mark:9334`` -- it retracted NOTHING (n=0); 21.7 s of it was the
  HY settle's L3 stat walk over the held spans, gathered before a vote that
  then found nothing refused.
* ``park_running`` stamped the awake re-queue clock right after the
  retraction, ahead of those 31 s. 06:57:02 the answer left and in the same
  second the next pass re-queued the whole park (``WEG2-D-PARK requeue
  (awake): 2 parked request(s) at the queue head ['weg2-88-1054',
  'weg2-89-1056'], 1 back to the #1471 settle``).
* D admitted weg2-88-1054 (205760 tokens) and its load-back held the scheduler
  thread 72 s (``WEG2-POST-WAKE-PASS schedule_ms=72024 init_new_ms=66794``);
  the front had begun the D->P flip and its drain witness got no
  /get_server_info answer: ``WEG2 STOP W3 Weg2DrainWitnessUnreachable``
  06:57:16. The front had lapsed the park too (its stamp was taken before the
  54 s RPC: ``PARK-PARTIAL still_running=[...]``).

The defect tests are red on 17c2563647 and green with the fix.
"""
from __future__ import annotations

import os
import sys
import types

import pytest

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

from sglang.srt.weg2 import d_park_runtime as rt  # noqa: E402
from sglang.srt.weg2 import d_seats as ds  # noqa: E402
from sglang.srt.weg2 import park_hold_yield as hy  # noqa: E402
from sglang.srt.weg2 import phase_policy as pp  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from test_weg2_park_wait_h91c2 import FakeClock, _req, _Sched  # noqa: E402

PARK_S = 31.3  # the metal park: yield 21.7 s + l3mark 9.3 s + depth 0.2 s


@pytest.fixture(autouse=True)
def _group_d(monkeypatch):
    monkeypatch.setenv("SGLANG_WEG2_GROUP", "D")
    monkeypatch.delenv(ds.PARK_ENV, raising=False)
    monkeypatch.delenv(ds.RESUME_MARGIN_ENV, raising=False)
    monkeypatch.setenv(ds.AWAKE_REQUEUE_ENV, "30")


@pytest.fixture
def clock(monkeypatch):
    c = FakeClock()
    monkeypatch.setattr(rt, "time", c)
    return c


def _slow_park(monkeypatch, clock, seconds=PARK_S):
    """The park's own phases after the retraction take ``seconds`` (the HY
    settle stands for all of them: yield + l3mark on the metal)."""
    real = hy.settle

    def slow(sched, *, retracted, parked):
        clock.t += seconds
        return real(sched, retracted=retracted, parked=parked)

    monkeypatch.setattr(hy, "settle", slow)


def _park(s, epoch=90):
    from sglang.srt.managers.io_struct import Weg2ParkRunningReqInput

    return rt.park_running(s, Weg2ParkRunningReqInput(epoch=epoch, reason="immediate-over-x"),
                           late_hold_armed=True)


def _metal_sched():
    """epoch 90's park: nothing running, one settle request (weg2-88-1053, its
    read still short), two queued behind (weg2-88-1054, weg2-89-1056)."""
    s = _Sched(running=[], waiting=[_req("weg2-88-1054", 2), _req("weg2-89-1056", 3)])
    s.weg2_post_wake_settle = [_req("weg2-88-1053", 1)]
    return s


# ============================================================ D: the park clock
def test_epoch90_a_31s_park_is_not_requeued_by_the_next_pass(monkeypatch, clock):
    """RED on 17c2563647: the stamp sat before the 31 s, the next pass found the
    park 31 s old and re-queued it (D ran weg2-88-1054's load-back, W3)."""
    _slow_park(monkeypatch, clock)
    s = _metal_sched()
    out = _park(s)
    assert out.success
    parked_before = [str(r.rid) for r in s.weg2_d_parked]
    assert sorted(parked_before) == ["weg2-88-1053", "weg2-88-1054", "weg2-89-1056"]
    # the very next pass, same second as the answer (06:57:02)
    assert rt.park_tick(s) == 0
    assert [str(r.rid) for r in s.weg2_d_parked] == parked_before
    assert s.waiting_queue == []
    # the net still works: 30 s after the ANSWER without a sleep, the park re-queues
    clock.t += 29.0
    assert rt.park_tick(s) == 0
    clock.t += 1.0
    assert rt.park_tick(s) == 3
    assert [str(r.rid) for r in s.waiting_queue] == ["weg2-88-1054", "weg2-89-1056"]


def test_the_late_hold_clock_starts_at_the_answer_too(monkeypatch, clock):
    """A hand-off reaching D after the park takes the late hold's stamp as its
    own re-queue clock (hold_late_arrival): it must be the answer's, or a late
    arrival would lapse the whole park at once."""
    _slow_park(monkeypatch, clock)
    s = _metal_sched()
    out = _park(s)
    t_answer = clock.t
    assert out.late_hold
    assert getattr(s, rt.LATE_HOLD_ATTR) == t_answer
    late = _req("weg2-89-1057", 4)
    assert rt.hold_late_arrival(s, late)
    assert getattr(late, ds.SINCE_ATTR) == t_answer
    assert rt.park_tick(s) == 0


def test_the_answer_carries_its_wall_stamp(monkeypatch, clock):
    _slow_park(monkeypatch, clock)
    s = _metal_sched()
    t0 = clock.t
    out = _park(s)
    assert out.answered_at == pytest.approx(t0 + PARK_S)


def test_a_fast_park_behaves_as_before(clock):
    """A park of no measurable duration: stamp == retraction moment, the
    awake re-queue after 30 s exactly as H91c2 pinned it."""
    s = _Sched(running=[_req("a", 1), _req("b", 2)])
    out = _park(s)
    assert sorted(out.parked) == ["a", "b"]
    clock.t += 29.9
    assert rt.park_tick(s) == 0
    clock.t += 0.2
    assert rt.park_tick(s) == 2


# ============================================================ front: the lapse clock
def test_front_lapse_starts_at_the_answer_not_before_the_rpc():
    """RED on 17c2563647 (the helper did not exist; the front stamped t_park):
    t_park 06:56:07.914, answer 06:57:02.1 (54.3 s RPC) -- with the pre-RPC
    stamp the front lapsed at once (PARK-PARTIAL still_running=3)."""
    t_park, t_answer, t_recv = 1000.0, 1054.2, 1054.3
    body = '{"success": true, "parked": ["weg2-88-1053"], "held": ["weg2-88-1054"], ' \
           '"late_hold": true, "epoch": 90, "answered_at": %r}' % t_answer
    got = pp.park_answered_at(200, body)
    assert got == pytest.approx(t_answer)
    stamp = pp.park_lapse_stamp(t_park, got, t_recv)
    assert stamp == pytest.approx(t_answer)
    assert not pp.parked_lapsed(stamp, t_recv)
    # front lapse no later than D's re-queue (answer + 30 s on every D rank)
    assert pp.parked_lapsed(stamp, t_answer + pp.PARK_REQUEUE_S)


def test_front_without_the_field_keeps_the_h91c3_stamp():
    assert pp.park_answered_at(200, '{"parked": []}') is None
    assert pp.park_answered_at(200, '{"answered_at": 0}') is None
    assert pp.park_answered_at(200, '{"answered_at": true}') is None
    assert pp.park_answered_at(200, '{"answered_at": "x"}') is None
    assert pp.park_answered_at(409, '{"answered_at": 5.0}') is None
    assert pp.park_answered_at(200, "not json") is None
    assert pp.park_lapse_stamp(1000.0, None, 1054.3) == 1000.0


def test_front_clamps_a_skewed_stamp_into_the_rpc_window():
    assert pp.park_lapse_stamp(1000.0, 990.0, 1054.3) == 1000.0     # before the RPC began
    assert pp.park_lapse_stamp(1000.0, 1100.0, 1054.3) == 1054.3    # after the answer arrived


def test_the_http_body_carries_answered_at():
    """The wiring: http_server's /weg2/park_running copies the field."""
    # read as text: importing http_server pulls optional audio deps this desk lacks
    path = os.path.join(os.path.dirname(os.path.dirname(rt.__file__)), "entrypoints", "http_server.py")
    src = open(path).read()
    i = src.index("async def weg2_park_running(")
    seg = src[i:src.index("\n@app.", i)]
    assert 'body["answered_at"] = answered_at' in seg
    assert 'getattr(ret, "answered_at", 0.0)' in seg


def test_the_front_stamps_parked_rids_with_the_lapse_stamp():
    """The wiring: front's park handler writes park_lapse_stamp(...) into
    _d_parked, not the pre-RPC t_park."""
    import inspect

    from sglang.srt.weg2 import front

    src = inspect.getsource(front)
    i = src.index("phase_policy.park_late_hold(code, text):\n            late =")
    seg = src[i:i + 600]
    assert "park_lapse_stamp(t_park, phase_policy.park_answered_at(code, text)" in seg
    assert "self._d_parked[r] = t_lapse" in seg


# ============================================================ HY: no walk in a clean park
class _CountingTree:
    """The UnifiedRadixCache surface settle touches; every L3 stat is counted."""

    def __init__(self, refused=None, peer=None):
        self._weg2_park_track = None
        self.refused = refused or {}
        self.votes = []
        self.stats = 0
        self.peer = peer

    def _weg2_arena_pools(self):
        from sglang.srt.mem_cache.unified_cache_components.tree_component import ComponentType

        return {ComponentType.FULL: types.SimpleNamespace(arena=None)}

    def _all_reduce_attn_groups(self, t, op, label=""):
        self.votes.append((label, t.numel()))
        if self.peer is not None:
            import torch

            t.copy_(torch.minimum(t, torch.tensor(self.peer(label, t.tolist()), dtype=t.dtype)))


def test_a_clean_park_walks_no_l3_and_votes_two_integers(monkeypatch):
    """RED on 17c2563647: held_facts (the L3 stat walk) ran for every held
    candidate before the vote -- 21.7 s of epoch 90's 31.3 s park."""
    tree = _CountingTree()
    walked = []
    monkeypatch.setattr(hy, "held_facts", lambda t, rid: walked.append(rid) or (3215, 3215))
    hy.begin(tree)
    tree._weg2_park_track = {}
    parked = [types.SimpleNamespace(rid=r) for r in ("weg2-88-1053", "weg2-88-1054", "weg2-89-1056")]
    verdict = hy.settle(types.SimpleNamespace(tree_cache=tree), retracted=[], parked=parked)
    assert verdict == "clean"
    assert walked == []
    assert tree.votes == [("park_hold_yield", 2)]


def test_a_peer_refusal_makes_every_rank_walk_and_vote_again(monkeypatch):
    """Rank-uniform: this rank refused nothing, a peer did -- the first MIN
    says refused on every rank, so this rank walks and enters the second vote
    too (the peers stand in it; skipping it would hang them)."""
    tree = _CountingTree(peer=lambda label, vec: ([-1, -5] if label == "park_hold_yield" else vec))
    walked = []
    monkeypatch.setattr(hy, "held_facts", lambda t, rid: walked.append(rid) or (4, 4))
    hy.begin(tree)
    tree._weg2_park_track = {}
    parked = [types.SimpleNamespace(rid="h")]
    # need=5 > on-disk 4: nothing is given back, but both votes were taken
    verdict = hy.settle(types.SimpleNamespace(tree_cache=tree), retracted=[], parked=parked)
    assert verdict == "stands"
    assert walked == ["h"]
    assert tree.votes == [("park_hold_yield", 2), ("park_hold_yield_facts", 4)]


# ============================================================ PLE warm: the touch releases the GIL
def test_the_ple_page_touch_releases_the_gil():
    """RED on 17c2563647 (``ctypes.string_at``: a PYFUNCTYPE -- the GIL stays
    held while the kernel serves the fault; D TP0 06:56:07-06:57:02Z: 6049
    faults in 55270 ms with the scheduler thread of the same process locked
    out). ``ctypes.memmove`` is a CFUNCTYPE: ctypes drops the GIL around it."""
    import ctypes

    from sglang.srt.models import qwen4_exp_ple_table as pt

    touch = getattr(pt, "_PAGE_TOUCH", None)
    assert isinstance(touch, ctypes._CFuncPtr), touch
    assert not (touch._flags_ & ctypes._FUNCFLAG_PYTHONAPI)
    # the hazard it replaces, pinned so a later "simplification" sees it
    assert ctypes._string_at._flags_ & ctypes._FUNCFLAG_PYTHONAPI


def test_the_warm_faults_every_cold_page_through_the_gil_free_touch():
    import tempfile
    from unittest import mock

    from sglang.srt.models import qwen4_exp_ple_table as pt
    from test_weg2_ple_decode_warm_h29a import _mapped_table

    with tempfile.TemporaryDirectory() as d:
        table, mm = _mapped_table(d)
        seen = []
        real = pt._touch_page

        def spy(addr, scratch):
            real(addr, scratch)
            seen.append((addr, int(scratch[0])))

        with mock.patch.object(pt, "_mincore_page", return_value=0), \
                mock.patch.object(pt, "_touch_page", side_effect=spy):
            c = pt.warm_ple_rows(table, list(range(0, 128, 5)), max_pages=0)
        assert c["cold"] == c["pages"] == c["touched"] == len(seen) > 0
        import ctypes

        for addr, byte in seen:  # the byte really came from the mapping
            assert byte == ctypes.c_ubyte.from_address(addr).value
        del table
        mm.close()
