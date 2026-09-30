"""CHUNK-REST: the chunked continuation counts once against the seats.

Hermetic (no CUDA). Metal (bridge f833fcbb2d, D log TP0 19:51:39,
'PDFLIP-POST-WAKE-PASS n=3 ... admit_partial=admitted=1 left=1
batch_full_break=1(first=pdflip-16-54)'): 6 seats (max_running_requests=6 =
the request-slot pool), 4 running (14-39, 16-40, 16-44, 16-52), pdflip-16-53's
last chunk (#794 cut its extend) carried in can_run_list. The count arm
compared ``len(can_run_list)=1 >= min(6-4, available_size()=1)`` and closed
the batch: the continuation was counted twice, once as a list member and
once as a used request slot. pdflip-16-54 waited the pass and then every decode
round behind the sticky ``batch_is_full``.

What these cases pin:

* 4 running + the chunked rest + 1 waiting -> the batch is NOT full; after
  the waiting one it is (six seats, six requests, never seven);
* the vote (``_local_admit_limit``) gives the held slot back the same way,
  so the group MIN does not re-close the batch; not under PP (seam refusal);
* without a held slot / with anchor tails armed nothing changes.
"""

from types import SimpleNamespace

import pytest

from flliper.srt.managers import scheduler as sched_mod
from flliper.srt.managers.scheduler import Scheduler

SEATS = 6


class _Pool:
    def __init__(self, used):
        self.used = used

    def available_size(self):
        return SEATS - self.used


def _stub(running=4, chunked_holds=True, pp_size=1):
    chunked = SimpleNamespace(rid="pdflip-16-53", req_pool_idx=5 if chunked_holds else None)
    running_reqs = [SimpleNamespace(rid=f"r{i}", req_pool_idx=i + 1) for i in range(running)]
    used = running + (1 if chunked_holds else 0)
    st = SimpleNamespace(
        admission_limiter=SimpleNamespace(current=SEATS),
        _pdflip_d_seat_cap=lambda: SEATS,
        _parked_carrier_discount=lambda running_bs: 0,
        req_to_token_pool=_Pool(used),
        parked_decode_set=SimpleNamespace(admission_headroom=lambda running_bs, res: res),
        running_batch=SimpleNamespace(reqs=running_reqs),
        chunked_req=chunked,
        ps=SimpleNamespace(pp_size=pp_size),
    )
    st.get_num_allocatable_reqs = lambda bs, slot_held=0: Scheduler.get_num_allocatable_reqs(st, bs, slot_held)
    st._chunk_rest_slot_held = lambda running: Scheduler._chunk_rest_slot_held(st, running)
    return st, chunked


@pytest.fixture(autouse=True)
def _server_args(monkeypatch):
    monkeypatch.setattr(sched_mod, "get_server_args", lambda: SimpleNamespace(pp_max_micro_batch_size=SEATS))


def _full(st, can_run_list, anchor_tails=False):
    """The count arm's comparison (scheduler.py, '#823 W9 COUNT arm')."""
    held = Scheduler._carried_slot_held(can_run_list, anchor_tails)
    carried = len(can_run_list) if anchor_tails else 0
    return len(can_run_list) >= st.get_num_allocatable_reqs(len(st.running_batch.reqs), held) + carried


def test_metal_form_six_seats_carry_six_requests():
    st, chunked = _stub()
    waiting = SimpleNamespace(rid="pdflip-16-54", req_pool_idx=None)
    assert not _full(st, [chunked])  # 4 running + the rest: pdflip-16-54 gets its seat
    assert _full(st, [chunked, waiting])  # six in flight: the seventh waits


def test_the_vote_gives_the_held_slot_back():
    """The group MIN reads every rank's vote; a vote still at 1 would close
    the batch again through ``admit_limit_decision``."""
    st, _chunked = _stub()
    assert Scheduler._local_admit_limit(st) == 2


def test_under_pp_the_vote_keeps_the_lower_number():
    st, _chunked = _stub(pp_size=3)
    assert Scheduler._local_admit_limit(st) == 1


def test_no_held_slot_changes_nothing():
    st, chunked = _stub(chunked_holds=False)
    assert Scheduler._carried_slot_held([chunked], False) == 0
    assert st.get_num_allocatable_reqs(4) == 2
    assert Scheduler._local_admit_limit(st) == 2


def test_anchor_tails_count_the_carried_members_already():
    st, chunked = _stub()
    assert Scheduler._carried_slot_held([chunked], True) == 0


def test_default_call_is_the_old_expression():
    st, _chunked = _stub()
    assert st.get_num_allocatable_reqs(4) == 1  # every other call site: unchanged
