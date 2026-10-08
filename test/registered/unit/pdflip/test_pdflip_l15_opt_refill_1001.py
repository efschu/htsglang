# SPDX-License-Identifier: Apache-2.0
"""L15-OPT: the optimistic cap-0 refill BEFORE decide() (operator order
01.10. ~20:45Z, design L15-WIRE2-NOTES.md sec 1).

With FLLIPER_PDFLIP_L15_REFILL=1 the cap-0 rank refills its held rows from
L2 BEFORE the group decides, so the sample check (next AP) can verify
live rows; the act after decide() only KEEPS (hold) or DROPS (fallback)
and must never refill a second time.  The safety property the operator
named: after an optimistic refill followed by a group "fallback" drop the
pool census must be IDENTICAL to the same wake without the refill -- the
refill may not leave a trace the drop does not erase.  REFILL=0 (boot 1)
must never call the refill at all (byte-identical).

An optimistic refill FAILURE must not drop per rank (xsn409: the drop is
a group act); it only flips this rank's vote to None -- the helper
returns True and the pools stay untouched.

Hermetic: the unbound production methods run on a FakeSelf whose pools
record a census (allocator free set, req_to_token rows, mamba occupancy,
tree_cache resets); the refill is stubbed at the seam _l15_do_refill --
the real copy is AP E2's own tested territory.
"""

import pathlib
import sys
from types import SimpleNamespace

REPO = pathlib.Path(__file__).resolve().parents[4]
sys.path.insert(0, str(REPO / "python"))

from flliper.srt.managers.scheduler_components import weight_updater as wu  # noqa: E402

WU = wu.SchedulerWeightUpdaterManager

# The hold this rank owns: KV device rows 3 and 7, mamba anchor row 5.
HELD_ROWS = (3, 7)
HELD_ANCHOR = 5


class _TreeCache:
    def __init__(self):
        self.nodes = {}      # rid -> device rows the tree references
        self.resets = 0

    def reset(self):
        self.nodes = {}
        self.resets += 1


class _ReqPool:
    def __init__(self):
        self.rows = {}       # rid -> device row
        self.mamba_pool = SimpleNamespace(occupied=set())

    def clear(self):
        # the flush shape: the mamba rows go with the req clear
        self.rows = {}
        self.mamba_pool.occupied = set()


class _Allocator:
    TOTAL = set(range(16))

    def __init__(self):
        self.free = set(self.TOTAL)

    def alloc(self, rows):
        self.free -= set(rows)

    def clear(self):
        self.free = set(self.TOTAL)


def _mk_sched():
    return SimpleNamespace(
        tree_cache=_TreeCache(),
        req_to_token_pool=_ReqPool(),
        token_to_kv_pool_allocator=_Allocator(),
        memory_saver_adapter=SimpleNamespace(
            alloc_info_ok=lambda base: False,
            set_keep_byte_spans=lambda base, spans: None,
        ),
        _kv_pools_for_flush=lambda: [],
    )


def _mk_manifest():
    return SimpleNamespace(spans=[SimpleNamespace(rid="r1",
                                                 slots=[0] + list(HELD_ROWS))])


class FakeSelf:
    """Unbound-method style; _l15_do_refill is the seam stub that mutates
    the pools exactly the way a successful copy would (rows become visible
    in tree/req, allocator slots leave the free set, the anchor is
    occupied) and counts the calls."""

    _l15_optimistic_refill = WU._l15_optimistic_refill
    _l15_wake_act = WU._l15_wake_act
    _l15_fallback_drop = WU._l15_fallback_drop
    _l15_clear_tms_keep_spans = WU._l15_clear_tms_keep_spans

    def __init__(self, sched, refill_mark, fail=False):
        self.scheduler = sched
        self._l15_wake_refill = refill_mark
        self._l15_wake_manifest = _mk_manifest()
        self.refills = 0
        self._fail = fail

    def _l15_do_refill(self, sched, optimistic=False):
        assert optimistic, "the optimistic seam must pass optimistic=True"
        self.refills += 1
        if self._fail:
            raise RuntimeError("simulated L2 read failure")
        sched.token_to_kv_pool_allocator.alloc(HELD_ROWS)
        sched.req_to_token_pool.rows["r1"] = HELD_ROWS[0]
        sched.req_to_token_pool.mamba_pool.occupied.add(HELD_ANCHOR)
        sched.tree_cache.nodes["r1"] = list(HELD_ROWS)
        return len(HELD_ROWS)


def _census(sched):
    """The operator's yardstick: allocator free set, req_to_token rows,
    mamba occupancy, tree state -- observable pool end state, not calls."""
    return (
        sched.token_to_kv_pool_allocator.free,
        dict(sched.req_to_token_pool.rows),
        set(sched.req_to_token_pool.mamba_pool.occupied),
        dict(sched.tree_cache.nodes),
    )


def test_opt_refill_then_fallback_census_equals_no_refill():
    # REFILL=1: optimistic refill, then the group decides "fallback".
    sched1 = _mk_sched()
    s1 = FakeSelf(sched1, refill_mark=True)
    assert s1._l15_optimistic_refill() is False
    assert s1.refills == 1
    assert s1._l15_wake_refill is False, "the mark must not survive to the act"
    d1 = WU._l15_wake_act(s1, sched1, "fallback", group_ok=True, master_on=True)
    # REFILL=0 (boot 1): the mark is never set, nothing is refilled.
    sched0 = _mk_sched()
    s0 = FakeSelf(sched0, refill_mark=False)
    assert s0._l15_optimistic_refill() is False
    assert s0.refills == 0
    d0 = WU._l15_wake_act(s0, sched0, "fallback", group_ok=True, master_on=True)
    # identical census, identical drop accounting, nothing left over:
    assert _census(sched1) == _census(sched0) == (
        _Allocator.TOTAL, {}, set(), {})
    assert d1 == d0
    # the mutant (drop skips the refilled rows) lands here:
    assert sched1.req_to_token_pool.mamba_pool.occupied == set()


def test_opt_refill_hold_runs_exactly_once():
    sched = _mk_sched()
    s = FakeSelf(sched, refill_mark=True)
    assert s._l15_optimistic_refill() is False
    assert s.refills == 1
    # the act on "hold" must NOT refill a second time:
    WU._l15_wake_act(s, sched, "hold", group_ok=True, master_on=True)
    assert s.refills == 1


def test_refill_off_never_calls_refill():
    sched = _mk_sched()
    s = FakeSelf(sched, refill_mark=False)
    assert s._l15_optimistic_refill() is False
    WU._l15_wake_act(s, sched, "hold", group_ok=True, master_on=True)
    assert s.refills == 0
    assert _census(sched) == (_Allocator.TOTAL, {}, set(), {})


def test_failed_opt_refill_votes_none_without_dropping():
    sched = _mk_sched()
    s = FakeSelf(sched, refill_mark=True, fail=True)
    assert s._l15_optimistic_refill() is True, "failure flips the vote to None"
    assert s.refills == 1
    # NO per-rank fallback drop (xsn409: the drop is the group's act):
    assert sched.tree_cache.resets == 0
    assert _census(sched) == (_Allocator.TOTAL, {}, set(), {})
    assert s._l15_wake_manifest is not None, "the record survives for the group"
