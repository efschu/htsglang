"""#1233: the per-pass prefetch drain collects records whose request is no
longer queued (admitted without waiting), so a slept group's flush witness
cannot be blocked by an orphan forever.

SPECIMEN (boot weg2ls3b1, 2026-09-07 09:27:26Z): PP0 '#973 PP0 PREFETCH
WAIT DISARMED (n=2 pending=2) ... local_prefetch_done=False' admitted rid
14d4e2c4; the request answered at 09:27:29; every HICACHE-ROUND line after
that printed ongoing_prefetch=1 on all three ranks; at 09:28:58 flush_cache
logged 'not-idle because: unknown' and the front STOPPED with W3.

Predicate-level (speed mode, one targeted check with a stated reason): the
drain is a pure function of (waiting_queue, ongoing_prefetch) and the tree
cache's check_prefetch_progress; the wedge is only where it surfaces 90 s
later. RED on 9ec68d0cc9 (the drain visits the waiting queue only), GREEN
after the fix.
"""

import types


class _Tree:
    def __init__(self, ongoing):
        self.ongoing_prefetch = dict.fromkeys(ongoing, object())
        self.checked = []
        self.retired = 0

    def drain_retired_prefetch(self):
        self.retired += 1

    def check_prefetch_progress(self, rid):
        self.checked.append(rid)
        self.ongoing_prefetch.pop(rid, None)
        return True


def _drain():
    from sglang.srt.managers.scheduler import Scheduler

    return Scheduler._drain_prefetch_progress


def test_orphan_records_are_collected():
    tree = _Tree(["queued-1", "orphan-a", "orphan-b"])
    sched = types.SimpleNamespace(
        enable_hicache_storage=True,
        tree_cache=tree,
        waiting_queue=[types.SimpleNamespace(rid="queued-1")],
    )
    verdicts = _drain()(sched, )
    assert verdicts == {"queued-1": True}
    assert tree.checked == ["queued-1", "orphan-a", "orphan-b"]
    assert tree.ongoing_prefetch == {}
    assert sched._1233_prefetch_orphans_collected == 2


def test_no_orphans_is_byte_identical():
    tree = _Tree(["queued-1"])
    sched = types.SimpleNamespace(
        enable_hicache_storage=True,
        tree_cache=tree,
        waiting_queue=[types.SimpleNamespace(rid="queued-1")],
    )
    assert _drain()(sched) == {"queued-1": True}
    assert tree.checked == ["queued-1"]
    assert not hasattr(sched, "_1233_prefetch_orphans_collected")


def test_storage_off_touches_nothing():
    tree = _Tree(["orphan-a"])
    sched = types.SimpleNamespace(enable_hicache_storage=False, tree_cache=tree, waiting_queue=[])
    assert _drain()(sched) == {}
    assert tree.checked == [] and tree.retired == 0
