# SPDX-License-Identifier: Apache-2.0
"""L15-HOSTLOCK: the held chains' L2 arena refs survive the P phase.

LCHOST defect 2 on fakes: retain_at_sleep step (5) reset_keep releases every
arena reference the kept chains hold, so the recorded L2 slots must already
carry a sleep-hold reference: exactly ONE per distinct held L2 slot (KV +
anchor, -1 = not-in-L2 skipped), taken BEFORE reset_keep runs, and released
exactly once when the wake verdict is acted -- hold: after the refill copied,
fallback: inside _l15_fallback_drop. A refused/deferred wake (group not ok)
keeps the hold armed; master off takes and releases nothing.
"""

from __future__ import annotations

import pathlib
import sys
import types

import torch

REPO = pathlib.Path(__file__).resolve().parents[4]
sys.path.insert(0, str(REPO / "python"))

from flliper.srt.managers.scheduler_components.weight_updater import (
    SchedulerWeightUpdaterManager as WU,
)
from flliper.srt.pdflip import l15_hostlock, l15_retain
from flliper.srt.pdflip.l15_policy import Candidate
from flliper.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=10)

PREFIX = (0, 1, 2, 3)
SLOTS_OF = {"sa": (10, 11, 12, 13), "sb": (10, 11, 12, 14)}
ANCHOR_SLOT_OF = {"sa": 5, "sb": 5}
CANDIDATES = [
    Candidate(rid="sa", kind="served", last_active=9.0,
              rows_by_rank=(2, 2, 2), anchor_depth=3, kv_depth=3),
    Candidate(rid="sb", kind="served", last_active=8.0,
              rows_by_rank=(2, 2, 2), anchor_depth=3, kv_depth=3),
]
# L2 (arena) rows of the held chains: shared prefix 201/202 (dedupe), a row
# not in L2 (-1, must be skipped), unique tail 203; anchor 301 / not-in-L2.
L2_OF = {"sa": ((201, 202, -1), (5, 6, -1)),
         "sb": ((201, 202, 203), (5, 6, 7))}
ANCHOR_L2_OF = {"sa": (301, 9), "sb": (-1, -1)}
HELD_KV = (201, 202, 203)
HELD_ANCHOR = (301,)


class FakeAllocator:
    def __init__(self, size=16):
        self.size = size
        self.free_pages = torch.empty(0, dtype=torch.int64)
        self.release_pages = torch.empty(0, dtype=torch.int64)

    def clear(self):
        self.free_pages = torch.arange(1, self.size + 1, dtype=torch.int64)
        self.release_pages = torch.empty(0, dtype=torch.int64)


class FakeMambaAllocator:
    def __init__(self, size=8):
        self.size = size
        self.free_slots = torch.empty(0, dtype=torch.int64)

    def clear(self):
        self.free_slots = torch.arange(1, self.size + 1, dtype=torch.int64)


class FakeNode:
    pass


class FakeArena:
    """Records every ref delta on the shared events log, tracks counts."""

    def __init__(self, events, name):
        self.events = events
        self.name = name
        self.refs = {}

    def ref_slots(self, slots, delta):
        slots = tuple(int(s) for s in slots)
        self.events.append(("ref", self.name, int(delta), slots))
        for s in slots:
            self.refs[s] = self.refs.get(s, 0) + int(delta)
        return len(slots)


class FakePool:
    def __init__(self, events, name):
        self.arena = FakeArena(events, name)


def _fake_tree(kv_pool, mamba_pool):
    return types.SimpleNamespace(
        cache_controller=types.SimpleNamespace(
            mem_pool_host=types.SimpleNamespace(entry_map={
                "kv": types.SimpleNamespace(host_pool=kv_pool),
                "mamba": types.SimpleNamespace(host_pool=mamba_pool),
            }),
            mamba_pool_host=None,
        ),
        mamba_pool_host=None,
    )


def _recorder(node, kv_map, anchor_map, visited):
    pass


def _retain(tmp_path, events, hold=None, rank=1, caps=(4, 4, 4)):
    """One rank-1 sleep against fresh fakes; hold is the hold_l2_refs kwarg
    (None = master off / hook not wired)."""
    alloc = FakeAllocator()
    alloc.clear()
    nodes = {"sa": FakeNode(), "sb": FakeNode()}
    res = l15_retain.retain_at_sleep(
        candidates=list(CANDIDATES),
        node_of=lambda rid: nodes[rid],
        slots_of=lambda rid: SLOTS_OF[rid],
        anchor_slot_of=lambda rid: ANCHOR_SLOT_OF[rid],
        l2_of=lambda rid: L2_OF[rid],
        anchor_l2_of=lambda rid: ANCHOR_L2_OF[rid],
        rewrite_tree=_recorder,
        caps_rows_by_rank=caps,
        cap_anchor_slots=4,
        prefix=PREFIX,
        rank=rank,
        epoch=777,
        pid=4242,
        kv_buffers=[torch.zeros(8, 3), torch.zeros(8, 3)],
        mamba_buffers=[torch.zeros(8, 3), torch.zeros(8, 3)],
        allocator=alloc,
        mamba_allocator=FakeMambaAllocator(),
        reset_keep=lambda ns: events.append("reset_keep"),
        set_keep=lambda buf, spans: None,
        manifest_path=str(tmp_path / "l15_manifest.json"),
        log=lambda line: None,
        **({"hold_l2_refs": hold} if hold is not None else {}),
    )
    return res


def _sched(events):
    kv, mamba = FakePool(events, "kv"), FakePool(events, "mamba")
    sched = types.SimpleNamespace(
        _l15_host_hold=(HELD_KV, HELD_ANCHOR),
        tree_cache=_fake_tree(kv, mamba),
        req_to_token_pool=types.SimpleNamespace(
            clear=lambda: events.append("req_clear")),
        token_to_kv_pool_allocator=types.SimpleNamespace(
            clear=lambda: events.append("alloc_clear")),
    )
    return sched, kv, mamba


def _hold_first(sched, kv, mamba):
    """Simulate the sleep having pinned exactly the recorded slots."""
    sched._l15_host_hold = l15_hostlock.hold_sleep_refs(
        kv, mamba, HELD_KV, HELD_ANCHOR, lambda m: None)


def _wu_stub(**over):
    base = dict(_l15_wake_manifest=None, _l15_wake_refill=False,
                _l15_clear_tms_keep_spans=lambda s: None)
    base.update(over)
    ns = types.SimpleNamespace(**base)
    # the REAL wake-side release helper, bound to the stub like on the class
    ns._l15_release_host_hold_refs = lambda sc, _ns=ns: (
        WU._l15_release_host_hold_refs(_ns, sc))
    return ns


def _refs(events, name, delta):
    return [e[3] for e in events
            if e[0] == "ref" and e[1] == name and e[2] == delta]


def test_sleep_hold_one_ref_per_distinct_slot_before_reset(tmp_path):
    events: list = []
    kv, mamba = FakePool(events, "kv"), FakePool(events, "mamba")
    taken = {}

    def hold(kv_slots, anchor_slots):
        taken["rec"] = l15_hostlock.hold_sleep_refs(
            kv, mamba, kv_slots, anchor_slots, lambda m: None)

    assert _retain(tmp_path, events, hold=hold) is not None
    # exactly one +1 batch per pool, distinct slots, -1 skipped, dups merged
    assert _refs(events, "kv", +1) == [HELD_KV]
    assert _refs(events, "mamba", +1) == [HELD_ANCHOR]
    assert taken["rec"] == (HELD_KV, HELD_ANCHOR)
    # the refs landed BEFORE reset_keep's release
    assert min(i for i, e in enumerate(events) if e[0] == "ref") < \
        events.index("reset_keep")
    # every held slot pinned exactly once
    assert all(v == 1 for v in kv.arena.refs.values())
    assert all(v == 1 for v in mamba.arena.refs.values())


def test_sleep_cap0_rank_takes_own_refs(tmp_path):
    # A rank's l2_slots are its OWN shard's arena slots; TP0 (cap 0) is the
    # rank that refills from L2, so it pins its own slots like cap>0 ranks.
    events: list = []
    kv, mamba = FakePool(events, "kv"), FakePool(events, "mamba")

    def hold(kv_slots, anchor_slots):
        l15_hostlock.hold_sleep_refs(kv, mamba, kv_slots, anchor_slots,
                                     lambda m: None)

    assert _retain(tmp_path, events, hold=hold, rank=0,
                   caps=(0, 4, 4)) is not None
    assert _refs(events, "kv", +1) == [HELD_KV]
    assert _refs(events, "mamba", +1) == [HELD_ANCHOR]


def test_sleep_master_off_takes_no_ref(tmp_path):
    events: list = []
    assert _retain(tmp_path, events) is not None
    assert not [e for e in events if e[0] == "ref"]
    assert "reset_keep" in events


def test_fallback_release_exactly_those_once():
    events: list = []
    sched, kv, mamba = _sched(events)
    _hold_first(sched, kv, mamba)
    WU._l15_fallback_drop(_wu_stub(), sched)
    assert _refs(events, "kv", -1) == [HELD_KV]
    assert _refs(events, "mamba", -1) == [HELD_ANCHOR]
    assert sched._l15_host_hold is None
    assert all(v == 0 for v in kv.arena.refs.values())
    assert all(v == 0 for v in mamba.arena.refs.values())
    n = len(events)
    WU._l15_fallback_drop(_wu_stub(), sched)  # idempotent
    assert len(events) == n


def test_wake_hold_releases_after_refill():
    events: list = []
    sched, kv, mamba = _sched(events)
    _hold_first(sched, kv, mamba)

    def refill(s, optimistic=False):
        events.append("refill")
        return 3

    stub = _wu_stub(_l15_wake_refill=True, _l15_do_refill=refill)
    assert WU._l15_wake_act(stub, sched, "hold",
                            group_ok=True, master_on=True) == 3
    assert events.index("refill") < min(
        i for i, e in enumerate(events) if e[0] == "ref" and e[2] == -1)
    assert all(v == 0 for v in kv.arena.refs.values())
    assert all(v == 0 for v in mamba.arena.refs.values())


def test_deferred_or_master_off_wake_releases_nothing():
    events: list = []
    sched, kv, mamba = _sched(events)
    assert WU._l15_wake_act(_wu_stub(), sched, "hold",
                            group_ok=False, master_on=True) == 0
    assert WU._l15_wake_act(_wu_stub(), sched, "hold",
                            group_ok=True, master_on=False) == 0
    assert not [e for e in events if e[0] == "ref"]
    assert sched._l15_host_hold == (HELD_KV, HELD_ANCHOR)
