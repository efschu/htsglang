"""W-SEAT: the COUNT vote reads the pass's running batch, not the stale one.

Hermetic (no CUDA). Metal (z30x2 424346f693 base, D log TP0 11:47:54 ->
11:48:10, cycle bench bs6 x 2k follow-up turn): the wake set a D phase of
n=2 seats (weg2-30-87/88). Both finished at 11:48:04, the running batch was
EMPTY, weg2-32-93 was admitted alone and prefilled. The next pass merged it
(``#1031 MERGE-PATH n=73 running_bs_after=1``) and admitted weg2-32-89 AND
weg2-32-90 beside it (``#969 EXTENT n=74`` two requests) -- three running in
an n=2 phase, ``Weg2DSeatOverrun: a forward batch of 3 requests in a D phase
of n=2 seats``, all three D ranks dead.

Mechanism: ``get_next_batch_to_run`` merges into an EMPTY running batch by
REBINDING its local name (``running_batch = last_batch``); the event loop
stores it into ``self.running_batch`` only after the plan returns. The #823
COUNT vote (``_local_admit_limit``) was taken in between from
``self.running_batch`` -- 0 running, vote n-0 = 2 -- and
``admit_limit_decision`` takes the group vote as THE limit, so the count arm
let two candidates through where the local count had one seat left.
"""

import ast
import inspect
import textwrap
from types import SimpleNamespace

import pytest

from sglang.srt.managers import scheduler as sched_mod
from sglang.srt.managers import tp_head_congruence
from sglang.srt.managers.scheduler import Scheduler
from sglang.srt.weg2 import d_seat_vram as dsv

CAP = 6


class _Pool:
    def available_size(self):
        return 30


def _stub_d_phase_n2():
    """D after the wake of epoch ...31: n=2 of cap 6, the running batch EMPTY
    (87/88 finished) -- what ``self.running_batch`` still names mid-pass."""
    st = SimpleNamespace(
        admission_limiter=SimpleNamespace(current=CAP),
        _parked_carrier_discount=lambda running_bs: 0,
        req_to_token_pool=_Pool(),
        parked_decode_set=SimpleNamespace(admission_headroom=lambda running_bs, res: res),
        running_batch=SimpleNamespace(reqs=[]),
        chunked_req=None,
        ps=SimpleNamespace(pp_size=1),
    )
    setattr(st, dsv.PHASE_ATTR, dsv.PhaseState(epoch="1790681695.31", n=2, cap=CAP, slot_limit=13))
    st._weg2_d_seat_cap = lambda: Scheduler._weg2_d_seat_cap(st)
    st.get_num_allocatable_reqs = lambda bs, slot_held=0: Scheduler.get_num_allocatable_reqs(st, bs, slot_held)
    st._chunk_rest_slot_held = lambda running: Scheduler._chunk_rest_slot_held(st, running)
    return st


@pytest.fixture(autouse=True)
def _armed_d(monkeypatch):
    monkeypatch.setattr(sched_mod, "get_server_args", lambda: SimpleNamespace(pp_max_micro_batch_size=CAP))
    monkeypatch.setattr(dsv, "armed", lambda env=None: True)


def _admitted(st, vote, running_bs, queue):
    """The count arm (scheduler.py '#823 W9 COUNT arm'): candidates join
    ``can_run_list`` until its length reaches the group-decided limit."""
    local = st.get_num_allocatable_reqs(running_bs)
    limit, _source = tp_head_congruence.admit_limit_decision(local, vote, True)
    can_run = []
    for rid in queue:
        if len(can_run) >= limit:
            break
        can_run.append(rid)
    return can_run


def test_metal_bs6_follow_up_keeps_the_n2_phase_at_two():
    st = _stub_d_phase_n2()
    merged = SimpleNamespace(reqs=[SimpleNamespace(rid="weg2-32-93", req_pool_idx=1)])
    vote = Scheduler._local_admit_limit(st, merged)
    assert vote == 1  # one seat left beside weg2-32-93
    admitted = _admitted(st, vote, len(merged.reqs), ["weg2-32-89", "weg2-32-90"])
    assert admitted == ["weg2-32-89"]
    dsv.guard(st, SimpleNamespace(reqs=merged.reqs + admitted))  # 2 of n=2: no overrun


def test_the_stale_read_is_what_overran():
    """The vote read off ``self.running_batch`` (empty) is n=2 and admits
    both -- a batch of 3, the metal death. Pins that the pass's batch, not
    ``self``'s, is what the vote must count."""
    st = _stub_d_phase_n2()
    merged = SimpleNamespace(reqs=[SimpleNamespace(rid="weg2-32-93", req_pool_idx=1)])
    stale_vote = Scheduler._local_admit_limit(st)
    admitted = _admitted(st, stale_vote, len(merged.reqs), ["weg2-32-89", "weg2-32-90"])
    with pytest.raises(dsv.Weg2DSeatOverrun):
        dsv.guard(st, SimpleNamespace(reqs=merged.reqs + admitted))


def test_the_merge_site_hands_its_running_batch_to_the_vote():
    """The one call site: ``get_next_batch_to_run`` must pass its LOCAL
    ``running_batch`` (rebound by the merge) into the reduce that takes the
    vote. A bare call reads ``self.running_batch`` and reopens the overrun."""
    src = textwrap.dedent(inspect.getsource(Scheduler.get_next_batch_to_run))
    calls = [
        n for n in ast.walk(ast.parse(src))
        if isinstance(n, ast.Call)
        and isinstance(n.func, ast.Attribute)
        and n.func.attr == "_update_uniform_pool_budget"
    ]
    assert len(calls) == 1
    args = calls[0].args
    assert len(args) == 1 and isinstance(args[0], ast.Name) and args[0].id == "running_batch"
