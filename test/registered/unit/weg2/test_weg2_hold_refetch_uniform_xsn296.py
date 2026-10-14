"""weg2xsn296 (Task #9, the xsn287 class): while group D slept, TP0 re-issued
a held request's store read from _weg2_hold_refetch (a TP collective inside
prefetch_from_storage) on its own rank-local 2 s timer while TP1/TP2 sat in
the next pass's request broadcast -- D stood for the rest of the boot
(stall_weg2xsn296_ep7_pid1041230..32.pyspy). The verdict "due" is now taken
without side effects, MIN-reduced over the group, and only the agreed set is
re-issued, on every rank in the same pass."""
from __future__ import annotations

import os
import time
from types import SimpleNamespace

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

from sglang.srt.managers import scheduler as sch  # noqa: E402


def _double(hold, gmin):
    obj = sch.Scheduler.__new__(sch.Scheduler)
    obj.weg2_dormant_hold = hold
    obj.weg2_dormant = True
    obj._weg2_drain_prefetch_revokes = lambda: None
    obj._weg2_group_min_flags = gmin
    return obj


def test_hold_refetch_reissues_only_the_group_agreed_set():
    r0, r1 = SimpleNamespace(rid="a"), SimpleNamespace(rid="b")
    calls = []

    def _refetch(req, now, allow_reissue=True):
        calls.append((req.rid, allow_reissue))
        return "reissued" if allow_reissue else "due"

    seen = {}

    def _gmin(flags):
        seen["flags"] = list(flags)
        return [1, 0]                       # a peer says b is not due yet

    obj = _double([r0, r1], _gmin)
    obj._weg2_refetch_one = _refetch
    n = sch.Scheduler._weg2_hold_refetch(obj)
    assert seen["flags"] == [1, 1]
    assert n == 1
    assert calls == [("a", False), ("b", False), ("a", True)]   # b never re-issued alone


def test_refetch_one_answers_due_without_side_effects_when_reissue_is_not_allowed():
    obj = sch.Scheduler.__new__(sch.Scheduler)
    issued = []
    obj.tree_cache = SimpleNamespace(check_prefetch_progress=lambda rid: True,
                                     prefetch_loaded_tokens_by_reqid={}, ongoing_prefetch={})
    obj._weg2_note_store_shortfall = lambda req: "record-short"
    obj._prefetch_kvcache = lambda req: issued.append(req.rid) or "issued"
    obj._clear_prefetch_deferral_fields = lambda req: None
    req = SimpleNamespace(rid="x", _1456_last=0.0)
    now = time.monotonic()
    assert sch.Scheduler._weg2_refetch_one(obj, req, now, allow_reissue=False) == "due"
    assert issued == [] and getattr(req, "_1456_n", 0) == 0     # no side effect
    assert sch.Scheduler._weg2_refetch_one(obj, req, now, allow_reissue=True) == "reissued"
    assert issued == ["x"]


def test_settle_and_wake_paths_take_the_same_form():
    src = open(sch.__file__).read()
    i = src.index("def _weg2_post_wake_settle_tick")
    blk = src[i:i + 6000]
    assert "allow_reissue=False" in blk and "_agreed_due" in blk
    j = src.index("def _weg2_release_dormant_hold")
    assert "allow_reissue=False" in src[j:j + 4000]
