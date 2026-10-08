# SPDX-License-Identifier: Apache-2.0
"""L15-DEDUPE: the wake refill accepts identical L2 rows shared by spans.

Finding L15-LCHOST-NOTES Q4 (defect 1, HIGH): chain_host_rows walks every
held chain independently, so two spans that share a prefix (one common
system prompt -- the normal agent-load shape) carry IDENTICAL
l2_slots/l2_gens for the shared tokens. The plan builders iterated
spans x tokens with no cross-span dedupe, and l15_refill.refill refused
the second occurrence with "duplicate page slot" even when the row is the
same row -- so every multi-rid hold with a shared prefix voted None and
fell back. Device-side the same case was fixed long ago (compact_plan F7,
test_pdflip_l15_shared_prefix_1001): a slot shared by several rids is
planned ONCE. This is the L2-side twin.

Semantics the fix must hold (brief L15-DEDUPE):
  * identical (compact_row, l2_slot, l2_gen) entries collapse to ONE plan
    entry; the row is loaded once;
  * the entry keeps the rid set: a generation mismatch still drops EVERY
    rid that references the row (whole-request drop, never partial);
  * a row claimed by two DIFFERENT l2 sources is a real conflict and
    stays a hard refusal;
  * l15_refill.refill / _refill_pgt keep their duplicate refusals for
    genuinely conflicting rows.

Hermetic: manifest from l15_manifest.HoldSpan, FakeHostPool records
slot_gens/_load_pages_all_layers (test_pdflip_l15_refill_1001 shape); no
CUDA, no arena, no scheduler boot. Geometry: prefix (0, 8) -> rank 0 owns
every slot and compact_row == slot.
"""

from types import SimpleNamespace

import pytest

from flliper.srt.managers.scheduler_components import weight_updater as wu
from flliper.srt.pdflip import l15_refill, l15_restore
from flliper.srt.pdflip.l15_manifest import HoldSpan, Manifest

PREFIX = (0, 8)
CAPS = (0,)


class FakeHostPool:
    """Records slot_gens and _load_pages_all_layers calls (refill_1001
    shape): gens maps slot -> current generation, absent -> -1."""

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
        self.load_calls.append((slots, device_indices, lanes, mode))
        if self._fail is not None:
            raise self._fail


def _manifest():
    # A and B share device rows 0,1,2 (the shared system prompt) with the
    # IDENTICAL L2 identity (slots 10,11,12 / gens 7); each keeps its own
    # tail row (A: 3 <- slot 13, B: 4 <- slot 14). C is disjoint.
    span_a = HoldSpan(
        rid="A", depth=4, slots=(0, 1, 2, 3), anchor_slot=3,
        l2_slots=(10, 11, 12, 13), l2_gens=(7, 7, 7, 7),
    )
    span_b = HoldSpan(
        rid="B", depth=4, slots=(0, 1, 2, 4), anchor_slot=4,
        l2_slots=(10, 11, 12, 14), l2_gens=(7, 7, 7, 7),
    )
    span_c = HoldSpan(
        rid="C", depth=2, slots=(5, 6), anchor_slot=6,
        l2_slots=(15, 16), l2_gens=(4, 4),
    )
    return Manifest(
        epoch=1, pid=1, spans=(span_a, span_b, span_c),
        rows_by_rank=(8,), anchor_slots=1,
    )


def _gens_all_current():
    return FakeHostPool({10: 7, 11: 7, 12: 7, 13: 7, 14: 7, 15: 4, 16: 4})


def test_shared_prefix_plan_carries_each_row_once():
    # The three shared prefix rows appear ONCE, first-appearance order,
    # then the two tails and C's rows: the plan is the dedupe's contract.
    plan = l15_restore.refill_plan(_manifest(), 0, PREFIX, CAPS)
    assert plan == [
        (0, 10, 7), (1, 11, 7), (2, 12, 7),
        (3, 13, 7), (4, 14, 7), (5, 15, 4), (6, 16, 4),
    ]
    # The rid-tagged ACT plan (weight_updater._l15_do_refill's shape)
    # names EVERY sharing rid on the shared rows, first rid first.
    act = l15_restore.rid_tagged_plan(_manifest(), 0, PREFIX)
    assert [tuple(e[1:]) for e in act] == plan
    assert [tuple(e[0]) for e in act] == [
        ("A", "B"), ("A", "B"), ("A", "B"), ("A",), ("B",), ("C",), ("C",),
    ]


def test_shared_prefix_refill_loads_each_page_once():
    # End to end over the shipped plan builders (the mutant seam: drop
    # the dedupe and the second occurrence of the shared slot raises
    # "refill: duplicate page slot" right here).
    act = l15_restore.rid_tagged_plan(_manifest(), 0, PREFIX)
    pool = _gens_all_current()
    ok, dropped = l15_refill.gen_check(act, pool)
    assert dropped == []
    assert ok == act
    n = l15_refill.refill(ok, pool, device_pool=None, page_tokens=1)
    assert n == 7
    assert len(pool.load_calls) == 1
    slots, didx, lanes, mode = pool.load_calls[0]
    assert [int(s) for s in slots] == [10, 11, 12, 13, 14, 15, 16]
    assert [int(d) for d in didx] == [0, 1, 2, 3, 4, 5, 6]
    assert lanes is None


def test_shared_row_gen_bump_drops_every_sharing_rid():
    # Slot 11 (shared by A and B) was re-claimed: gen 7 -> 9. The deduped
    # entry carries BOTH rids, so BOTH whole rids drop -- never just the
    # first tag -- and the disjoint rid C survives.
    act = l15_restore.rid_tagged_plan(_manifest(), 0, PREFIX)
    pool = _gens_all_current()
    pool._gens[11] = 9
    ok, dropped = l15_refill.gen_check(act, pool)
    assert list(dropped) == ["A", "B"]
    assert [e[1] for e in ok] == [5, 6]


def test_row_from_two_l2_sources_still_refused():
    # Same device row claimed from two DIFFERENT l2 slots is a real
    # conflict (the copy could only pick one source): hard refusal, no
    # silent first-wins.
    span_a = HoldSpan(
        rid="A", depth=4, slots=(0, 1, 2, 3), anchor_slot=3,
        l2_slots=(10, 11, 12, 13), l2_gens=(7, 7, 7, 7),
    )
    span_b = HoldSpan(
        rid="B", depth=4, slots=(0, 1, 2, 4), anchor_slot=4,
        l2_slots=(10, 99, 12, 14), l2_gens=(7, 7, 7, 7),
    )
    m = Manifest(epoch=1, pid=1, spans=(span_a, span_b),
                 rows_by_rank=(8,), anchor_slots=1)
    with pytest.raises(l15_refill.L15RefillError):
        l15_restore.owned_l2_rows(m, 0, PREFIX)
    with pytest.raises(l15_refill.L15RefillError):
        l15_restore.refill_plan(m, 0, PREFIX, CAPS)


def _laned_manifest():
    # P>1 form: A and B share the rows of page 20 (lanes 0 and 1) with
    # identical identity; B adds page 21's rows 2,3 (same lane set).
    span_a = HoldSpan(
        rid="A", depth=2, slots=(0, 1), anchor_slot=1,
        l2_slots=(20, 20), l2_gens=(7, 7), l2_lanes=(0, 1),
    )
    span_b = HoldSpan(
        rid="B", depth=4, slots=(0, 1, 2, 3), anchor_slot=3,
        l2_slots=(20, 20, 21, 21), l2_gens=(7, 7, 7, 7),
        l2_lanes=(0, 1, 0, 1),
    )
    return Manifest(epoch=1, pid=1, spans=(span_a, span_b),
                    rows_by_rank=(8,), anchor_slots=1)


def test_laned_plan_dedupes_and_pgt_loads_each_page_lane_once():
    # The lane-tagged builder shares the SAME dedupe: each (row, page,
    # lane) once; refill(P>1) then loads one call per page with the
    # common lane set and never fires "duplicate (page, lane)".
    laned = l15_refill.refill_plan_laned(_laned_manifest(), 0, PREFIX, CAPS)
    assert laned == [(0, 20, 7, 0), (1, 20, 7, 1), (2, 21, 7, 0),
                     (3, 21, 7, 1)]
    act = [(e[4], e[0], e[1], e[2], e[3])
           for e in l15_restore.owned_l2_rows(_laned_manifest(), 0, PREFIX)]
    pool = FakeHostPool({20: 7, 21: 7})
    ok, dropped = l15_refill.gen_check(act, pool)
    assert dropped == []
    n = l15_refill.refill(ok, pool, device_pool=None, page_tokens=2)
    assert n == 4
    assert len(pool.load_calls) == 1
    slots, didx, lanes, mode = pool.load_calls[0]
    assert [int(s) for s in slots] == [20, 21]
    assert [int(l) for l in lanes] == [0, 1]
    assert [int(d) for d in didx] == [0, 1, 2, 3]


def test_refill_still_refuses_genuine_duplicates_p1_and_pgt():
    # The refusal l15_refill keeps: NOT identical rows. P==1: two device
    # rows on one page slot. P>1: the same (page, lane) claimed twice --
    # the pre-dedupe shape of a conflict, refused before any copy.
    pool = FakeHostPool({})
    with pytest.raises(l15_refill.L15RefillError):
        l15_refill.refill([(("A",), 10, 3, 7), (("A",), 11, 3, 7)],
                          pool, device_pool=None, page_tokens=1)
    with pytest.raises(l15_refill.L15RefillError) as exc:
        l15_refill.refill([(("A",), 0, 20, 7, 0), (("B",), 0, 20, 7, 0)],
                          pool, device_pool=None, page_tokens=2)
    assert "duplicate (page 20, lane 0)" in str(exc.value)
    assert pool.load_calls == []


class _SampleFake:
    """Runs the unbound _l15_wake_sample_check on a manifest; every
    outside object is faked, sample_check captures the plan."""

    _l15_wake_sample_check = wu.SchedulerWeightUpdaterManager \
        ._l15_wake_sample_check

    def __init__(self, manifest):
        self._l15_wake_manifest = manifest
        self.scheduler = SimpleNamespace(
            tp_size=1,
            tree_cache=object(),
            tp_worker=SimpleNamespace(model_runner=SimpleNamespace(
                token_to_kv_pool=object())),
        )

    def _pdflip_rank(self):
        return 0


def test_wake_sample_check_plans_shared_rows_once(monkeypatch):
    # The wake sample check walks the SAME owned rows: shared prefix rows
    # must be sampled once, tagged with the first rid as a plain string
    # (l15_sample.sample_plan str()-es the tag; rid is display there).
    captured = []

    def fake_sample_check(plan, host_pool, live_pool, scratch, page_tokens,
                          k=64):
        captured.append(list(plan))
        return (0, 0, 0)

    pool = _gens_all_current()
    monkeypatch.setattr("flliper.srt.pdflip.l15_check.sample_check",
                        fake_sample_check)
    monkeypatch.setattr("flliper.srt.pdflip.l15_bind.live_host_pools",
                        lambda tree: (pool, None))
    monkeypatch.setattr("flliper.srt.pdflip.l15_scratch.make_scratch_pool",
                        lambda pool, k: SimpleNamespace(free=lambda: None))
    monkeypatch.setattr("flliper.srt.distributed.utils.get_cp_token_ratios",
                        lambda: None)
    fake = _SampleFake(_manifest())
    assert fake._l15_wake_sample_check() == (0, 0, 0)
    assert captured and [t[1] for t in captured[0]] == [0, 1, 2, 3, 4, 5, 6]
    assert all(isinstance(t[0], str) for t in captured[0])
    assert captured[0][0][0] == "A"  # first rid tags the shared row
