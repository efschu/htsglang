"""NF-TWIN-ROW 1021 (rc27 893d09fa3f, P log ..._1021_020616): the twin-anchor bounds are
PP0's decision; a PP follower adopts them from the pass's proxy frame and never derives
them from its own waiting queue.

THE ARRIVAL ORDER OF THE CASE (P log):
  11124 02:09:27 PP0 P-FORK-CUT SKIP rid=pdflip-0-1 site=add_one_req prefix=18432 extend=16384
  11171 02:09:28 PP0 FORK-ANCHOR TRACK rid=pdflip-0-1 step=[18432,34816) at=18688
  11317 02:09:31 PP0 REQ-TRACE r9 ... rids=['pdflip-0-4', ...]        <- AFTER PP0's plan
  11345 02:09:31 PP1 REQ-TRACE r10 ... rids=['pdflip-0-4', ...]       <- BEFORE PP1's plan
  11477 02:09:31 PP1 TWIN-ANCHOR TRACK rid=pdflip-0-1 ... at=18880
  11647 02:09:34 PP2 TWIN-ANCHOR TRACK rid=pdflip-0-1 ... at=18880
  14793 02:10:07 PP0 MATCH-CENSUS-DEEP rid=pdflip-0-4 reached=18880 accepted=18688
  14926 02:10:07 PP1 MATCH-CENSUS-DEEP rid=pdflip-0-4 reached=18880 accepted=18880

pdflip-0-1 (139176 ids) and pdflip-0-4 (176268) share 18903 leading ids: B =
floor_page(18902) = 18880. pdflip-0-3 (52736) shares 18747: B = 18688.
"""
from __future__ import annotations

from types import SimpleNamespace

import pytest

from flliper.srt.pdflip import twin_anchor as ta

PAGE = 64


def _req(rid, n, shared_with=None, shared=0, salt=0):
    if shared_with is None:
        ids = [1_000_000 * (salt + 1) + i for i in range(n)]
    else:
        base = shared_with.origin_input_ids
        ids = list(base[:shared]) + [7_000_000 + 100_000 * (salt + 1) + i for i in range(n - shared)]
    return SimpleNamespace(rid=rid, origin_input_ids=ids, extra_key=None, output_ids=[])


@pytest.fixture(autouse=True)
def _armed(monkeypatch):
    ta._reset_for_test()
    monkeypatch.setattr(ta, "_ARMED", True)
    yield
    ta._reset_for_test()


def _sched(rank, size=3):
    return SimpleNamespace(ps=SimpleNamespace(pp_rank=rank, pp_size=size))


def _case():
    src = _req("pdflip-0-1", 139176, salt=1)
    t3 = _req("pdflip-0-3", 52736, shared_with=src, shared=18747, salt=3)
    t4 = _req("pdflip-0-4", 176268, shared_with=src, shared=18903, salt=4)
    return src, t3, t4


def _send_and_receive(pp0, batch0, follower):
    """PP0's frame extras for its batch -> the follower's receive + pre-plan plant."""
    frame = {"hidden_states": "tensor"}
    extras = ta.frame_extra(pp0, batch0, {"__admission_decision__": ("row",)})
    frame.update(extras or {})
    ta.note_frame(follower, frame)
    assert ta.ROW_KEY not in frame, "the row must be popped before anything maps over the frame"
    ta.plant_for_plan(follower)
    return frame


def test_the_metal_arrival_order_plans_the_same_tracks_on_every_stage():
    """PP0 plans pass 1 with nothing twin-shaped queued; the followers plan it with
    pdflip-0-3 and pdflip-0-4 already absorbed. Every stage must plan PP0's tracks:
    none at 18880 anywhere (was: PP1/PP2 at 18688 and 18880, PP0 at neither)."""
    src, t3, t4 = _case()
    pp0, pp1, pp2 = _sched(0), _sched(1), _sched(2)
    b0 = ta.plan_bounds(pp0, [src], [], page=PAGE)
    assert b0 is None
    batch0 = SimpleNamespace(pdflip_twin_bounds=b0)
    _send_and_receive(pp0, batch0, pp1)
    b1 = ta.plan_bounds(pp1, [src], [t3, t4], page=PAGE)
    ta.clear_after_plan(pp1)
    batch1 = SimpleNamespace(pdflip_twin_bounds=b1)
    _send_and_receive(pp1, batch1, pp2)
    b2 = ta.plan_bounds(pp2, [src], [t3, t4], page=PAGE)
    assert b1 == b0 and b2 == b0, (b0, b1, b2)
    assert ta.row_stats()["differs"] == 2  # both followers' own queues would have drawn 18688/18880


def test_pp0_bounds_ride_the_frame_and_the_middle_rank_relays_them():
    src, t3, t4 = _case()
    pp0, pp1, pp2 = _sched(0), _sched(1), _sched(2)
    b0 = ta.plan_bounds(pp0, [src], [t3, t4], page=PAGE)
    assert b0 == {"pdflip-0-1": [18688, 18880]}
    _send_and_receive(pp0, SimpleNamespace(pdflip_twin_bounds=b0), pp1)
    b1 = ta.plan_bounds(pp1, [src], [], page=PAGE)  # PP1's own queue is EMPTY: still PP0's bounds
    ta.clear_after_plan(pp1)
    _send_and_receive(pp1, SimpleNamespace(pdflip_twin_bounds=b1), pp2)
    b2 = ta.plan_bounds(pp2, [src], [t4], page=PAGE)
    assert b1 == b0 == b2


def test_the_last_stage_sends_no_row_and_a_rowless_frame_means_none():
    src, t3, t4 = _case()
    pp2 = _sched(2)
    assert ta.frame_extra(pp2, SimpleNamespace(pdflip_twin_bounds={"x": [64]}), None) is None
    pp0 = _sched(0)
    assert ta.frame_extra(pp0, SimpleNamespace(pdflip_twin_bounds=None), None) is None
    f = _sched(1)
    ta.note_frame(f, {"hidden_states": "t"})
    ta.plant_for_plan(f)
    assert ta.plan_bounds(f, [src], [t3, t4], page=PAGE) is None


def test_danger_follower_without_a_planted_row_derives_nothing(caplog):
    """DANGER DIRECTION (Raenge uneins): a follower plan with no row of PP0's must not
    fall back to its own queue."""
    src, t3, t4 = _case()
    f = _sched(1)
    with caplog.at_level("WARNING", logger=ta.logger.name):
        assert ta.plan_bounds(f, [src], [t3, t4], page=PAGE) is None
    assert "TWIN-ROW ABSENT" in caplog.text


def test_a_row_feeds_only_the_plan_of_its_frame():
    """Planted for ONE plan: after the clear, a later plan without a new frame has no row."""
    src, t3, t4 = _case()
    pp0, pp1 = _sched(0), _sched(1)
    b0 = ta.plan_bounds(pp0, [src], [t3, t4], page=PAGE)
    _send_and_receive(pp0, SimpleNamespace(pdflip_twin_bounds=b0), pp1)
    assert ta.plan_bounds(pp1, [src], [], page=PAGE) == b0
    ta.clear_after_plan(pp1)
    assert ta.plan_bounds(pp1, [src], [t3, t4], page=PAGE) is None
    # a receive on another path (no pre-plan plant) does not reach a plan
    ta.note_frame(pp1, {ta.ROW_KEY: ta.encode(b0)})
    assert ta.plan_bounds(pp1, [src], [], page=PAGE) is None


def test_the_row_is_restricted_to_the_followers_batch():
    src, t3, t4 = _case()
    other = _req("pdflip-0-9", 20000, salt=9)
    f = _sched(1)
    ta.note_frame(f, {ta.ROW_KEY: ta.encode({"pdflip-0-1": [18688, 18880], "ghost": [64]})})
    ta.plant_for_plan(f)
    assert ta.plan_bounds(f, [src, other], [], page=PAGE) == {"pdflip-0-1": [18688, 18880]}


def test_non_pp_and_pp0_plan_exactly_as_before():
    src, t3, t4 = _case()
    for s in (_sched(0, 1), _sched(0, 3), SimpleNamespace()):
        assert ta.plan_bounds(s, [src], [t3, t4], page=PAGE) == ta.batch_bounds([src], [t3, t4], page=PAGE)


def test_wire_codec_round_trip():
    b = {"b": [128, 64, 64], "a": [192]}
    assert ta.decode(ta.encode(b)) == {"a": [192], "b": [64, 128]}
    assert ta.decode(None) == {} and ta.decode((("x", ("bad",)),)) == {}
