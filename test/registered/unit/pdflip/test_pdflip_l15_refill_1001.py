"""L15-12c-D: TP0 (cap 0) refills its held rows from L2 at the wake.

Pure-module tests for flliper.srt.pdflip.l15_refill: gen_check (one
slot_gens call, whole-rid drops) and refill (page-grouped single
_load_pages_all_layers call, all-or-nothing). Fake host pool records the
load arguments; no CUDA, no arena, no scheduler.
"""

import torch

from flliper.srt.pdflip import l15_refill


class FakeHostPool:
    """Records slot_gens and _load_pages_all_layers calls."""

    def __init__(self, gens, fail=None):
        self._gens = dict(gens)
        self._fail = fail
        self.gens_calls = []
        self.load_calls = []

    def slot_gens(self, slots):
        self.gens_calls.append([int(s) for s in slots])
        return [self._gens.get(int(s), -1) for s in slots]

    def _load_pages_all_layers(self, device_pool, slots, device_indices,
                               lanes=None, mode=None):
        self.load_calls.append(
            (slots, device_indices, lanes, mode)
        )
        if self._fail is not None:
            raise self._fail


def _row(rid, compact_row, l2_slot, l2_gen):
    return (rid, compact_row, l2_slot, l2_gen)


# C2 semantics: l2_slot is the arena PAGE slot (one per token at P == 1).
# rid A owns page slots 0-3, rid B page slots 4-7.
_PLAN_A_OK = [
    _row("A", 10, 0, 7), _row("A", 11, 1, 7),
    _row("A", 12, 2, 7), _row("A", 13, 3, 7),
]
_PLAN_B = [
    _row("B", 20, 4, 3), _row("B", 21, 5, 3),
    _row("B", 22, 6, 3), _row("B", 23, 7, 3),
]


def test_gen_check_all_ok_single_slot_gens_call():
    pool = FakeHostPool({0: 7, 1: 7, 2: 7, 3: 7})
    ok, dropped = l15_refill.gen_check(_PLAN_A_OK, pool)
    assert dropped == []
    assert ok == _PLAN_A_OK
    # ONE call, over the unique slots of the plan
    assert len(pool.gens_calls) == 1
    assert sorted(pool.gens_calls[0]) == [0, 1, 2, 3]


def test_gen_check_gen_mismatch_drops_whole_rid():
    # slot 5 was re-claimed: its generation moved 3 -> 9
    pool = FakeHostPool({0: 7, 1: 7, 2: 7, 3: 7, 4: 3, 5: 9, 6: 3, 7: 3})
    ok, dropped = l15_refill.gen_check(_PLAN_A_OK + _PLAN_B, pool)
    # B absent ENTIRELY (never partial), A untouched
    assert all(r[0] == "A" for r in ok)
    assert ok == _PLAN_A_OK
    assert list(dropped) == ["B"]


def test_gen_check_missing_l2_slot_drops_rid():
    # l2_slot -1 = no L2 copy: its whole rid goes
    plan = _PLAN_A_OK + [_row("C", 30, -1, 0)]
    pool = FakeHostPool({0: 7, 1: 7, 2: 7, 3: 7})
    ok, dropped = l15_refill.gen_check(plan, pool)
    assert ok == _PLAN_A_OK
    assert list(dropped) == ["C"]
    # -1 is not looked up: the census call names only real slots
    assert 0 in pool.gens_calls[0]
    assert -1 not in pool.gens_calls[0]


def test_gen_check_absent_slot_gen_minus_one_drops_rid():
    # slot_gens answers -1 for non-COMPLETE slots -> F1 drop
    pool = FakeHostPool({0: 7, 1: 7, 2: 7})  # slot 3 absent -> -1
    ok, dropped = l15_refill.gen_check(_PLAN_A_OK, pool)
    assert ok == []
    assert list(dropped) == ["A"]


def test_refill_p1_loads_each_page_slot_once():
    # C2 semantics: l2_slot IS the arena page slot; with page_tokens == 1
    # one slot carries one token, lane is always 0. Interleaved plan order
    # must still yield ascending unique slots and their rows in slot order.
    plan = [
        _row("B", 21, 5, 3), _row("A", 11, 1, 7),
        _row("B", 20, 4, 3), _row("A", 10, 0, 7),
        _row("A", 13, 3, 7), _row("B", 23, 7, 3),
        _row("A", 12, 2, 7), _row("B", 22, 6, 3),
    ]
    pool = FakeHostPool({})
    n = l15_refill.refill(plan, pool, device_pool=None, page_tokens=1)
    assert n == 8
    assert len(pool.load_calls) == 1
    slots, didx, lanes, mode = pool.load_calls[0]
    assert slots.dtype == torch.int64
    assert [int(s) for s in slots] == [0, 1, 2, 3, 4, 5, 6, 7]
    assert [int(i) for i in didx] == [10, 11, 12, 13, 20, 21, 22, 23]
    assert lanes is None
    assert mode is None


def test_refill_duplicate_slot_raises():
    # Two rows on one page slot under P == 1 is a genuine duplicate target.
    plan = [_row("A", 10, 3, 7), _row("A", 11, 3, 7)]
    pool = FakeHostPool({})
    try:
        l15_refill.refill(plan, pool, device_pool=None, page_tokens=1)
    except l15_refill.L15RefillError:
        pass
    else:
        raise AssertionError("duplicate slot must raise L15RefillError")
    assert pool.load_calls == []


def test_refill_page_tokens_gt_one_raises_before_any_load():
    # C2 records the PAGE slot only (l15_bind.py: (row - S) // P); the lane
    # per token is not recorded, so P > 1 cannot be mapped - silently the
    # worst way: slots 0-3 are FOUR page slots, but a divmod reading sees
    # four lanes of page 0 and copies every one of the four pages into the
    # row of the first. Must raise before touching the pool.
    plan = [
        _row("A", 10, 0, 7), _row("A", 11, 1, 7),
        _row("A", 12, 2, 7), _row("A", 13, 3, 7),
    ]
    pool = FakeHostPool({})
    try:
        l15_refill.refill(plan, pool, device_pool=None, page_tokens=4)
    except l15_refill.L15RefillError:
        pass
    else:
        raise AssertionError("page_tokens > 1 must raise L15RefillError")
    assert pool.load_calls == []
    assert pool.gens_calls == []


def test_refill_load_failure_raises_named_error_no_partial_count():
    pool = FakeHostPool({}, fail=RuntimeError("boom"))
    try:
        l15_refill.refill(_PLAN_A_OK, pool, device_pool=None, page_tokens=4)
    except l15_refill.L15RefillError:
        pass
    else:
        raise AssertionError("load failure must raise L15RefillError")
