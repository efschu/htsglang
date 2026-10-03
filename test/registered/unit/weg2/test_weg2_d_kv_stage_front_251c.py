"""#251c front half: the P->D wake carries the phase's KV demand.

``phase_kv_tokens`` rides the kv_cache resume next to ``handoff_n``/``parked_n``
(H91 part C). It is the front's OWN count -- the seat charge its D gate already
uses (``d_seat_need``) -- over the phase's first n seats in D's order: the
parked ones D resumes first, the held ones, the hand-offs in flight, then the
waiting ones. Without the field (an older front, nothing priced) D takes S0,
exactly as before the stages existed.
"""
from __future__ import annotations

import os
import pickle
import types

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

from sglang.srt.managers.io_struct import ResumeMemoryOccupationReqInput  # noqa: E402
from sglang.srt.weg2 import d_seat_vram as dsv  # noqa: E402
from sglang.srt.weg2.front import Front  # noqa: E402


def _seat(rid, tokens):
    return types.SimpleNamespace(rid=rid, tokens=tokens)


def _ready(est, realised=0):
    return types.SimpleNamespace(fut=None, est_prompt=est, leg1_prompt_tokens=realised)


def _front(*, parked=(), outstanding=(), seats=(), ready=(), d_bs=6):
    live = [_seat(r, t) for r, t in seats]
    return types.SimpleNamespace(
        _d_parked={r: 0.0 for r in parked},
        groups={"D": types.SimpleNamespace(outstanding={r: 1.0 for r in outstanding})},
        _ready_for_d=list(ready), _d_seats_live=live, d_bs=d_bs,
        _handoff_in_flight=lambda: max(0, len(live) - len(outstanding)))


def test_the_wake_carries_the_phase_kv_tokens_of_its_seats():
    # D holds a (40k) and b (30k), one hand-off c (20k) is in flight, p1 (50k)
    # is parked, one request waits with a realised 60k prompt (arrival 90k)
    front = _front(parked=["p1"], outstanding=["a", "b", "p1"],
                   seats=[("a", 40000), ("b", 30000), ("c", 20000), ("p1", 50000)],
                   ready=[_ready(90000, realised=60000)])
    extra = Front._wake_handoff_fields(front, "D")
    assert (extra["handoff_n"], extra["parked_n"]) == (4, 1)
    assert extra["phase_kv_tokens"] == 50000 + 40000 + 30000 + 20000 + 60000
    req = ResumeMemoryOccupationReqInput(tags=["kv_cache"], epoch="e3", **extra)
    ranks = [pickle.loads(pickle.dumps(req)) for _ in range(3)]
    assert {r.phase_kv_tokens for r in ranks} == {200000}


def test_only_the_phase_seats_count_the_rest_waits_for_a_seat():
    # d_bs 2: the parked one and the first held one are the phase; the second
    # held one and the waiting one are not this phase's KV
    front = _front(parked=["p1"], outstanding=["a", "b", "p1"],
                   seats=[("a", 40000), ("b", 30000), ("p1", 50000)],
                   ready=[_ready(70000)], d_bs=2)
    extra = Front._wake_handoff_fields(front, "D")
    assert extra["phase_kv_tokens"] == 50000 + 40000
    # a waiting request never seated is priced like its seat would be: the
    # arrival estimate before leg 1 answered
    assert Front._wake_handoff_fields(_front(ready=[_ready(70000)]), "D")["phase_kv_tokens"] == 70000


def test_without_the_field_d_takes_s0_exactly_as_before():
    # a partial front (no seats, no sizes) prices nothing: the field is left
    # off and the wake is byte-identical to H91c's
    old = types.SimpleNamespace(
        _d_parked={"p1": 0.0}, groups={"D": types.SimpleNamespace(outstanding={"a": 1, "b": 1})},
        _ready_for_d=[types.SimpleNamespace(fut=None)], _handoff_in_flight=lambda: 1)
    extra = Front._wake_handoff_fields(old, "D")
    assert extra == {"handoff_n": 4, "parked_n": 1}
    req = ResumeMemoryOccupationReqInput(tags=["kv_cache"], **extra)
    assert req.phase_kv_tokens is None
    form = dsv.StageForm(tokens=(262144, 393216, 524288), rows_on=32, max_by_seats=())
    ch = dsv.choose_form_stage(form, 5, req.phase_kv_tokens)
    assert (ch.stage, ch.tokens, ch.over) == (0, 262144, False)
    # and a P wake carries nothing at all
    assert Front._wake_handoff_fields(_front(seats=[("a", 1)], outstanding=["a"]), "P") == {}


def test_the_demand_picks_the_stage_d_maps():
    form = dsv.StageForm(tokens=(262144, 393216, 524288), rows_on=32, max_by_seats=())
    front = _front(outstanding=["a", "b", "c", "d", "e"],
                   seats=[(r, 60000) for r in "abcde"])
    kv = Front._wake_handoff_fields(front, "D")["phase_kv_tokens"]
    assert kv == 300000
    ch = dsv.choose_form_stage(form, 5, kv)
    assert (ch.stage, ch.tokens) == (1, 393216)
