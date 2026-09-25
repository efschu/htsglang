"""H91a (25.09.2026), 27B port of the NF fix bf6c97171b, part 1 only.

NF cu130 acceptance dkrnfbar1agent09252021: rid 53572be2 sat 61 s in P's
queue with nothing running and was answered 503 WEG2-INTAKE-STALL by the
admission-wedge path. Root: #1400's told verdict was consumed on the first
admission visit; a visit the adder could not seat lost it, and every later
visit skipped the rid as ``weg2_store_told_pending`` for ever. The 27B tree
(desk/27b-rc9-0925 @ d7f588e017) carries the identical gate and call site.

Red on d7f588e017: the told verdict survives an admission visit that did not
seat the request. Part 2 of the NF commit (intake_verdict: no 503 for a
request that fits free + evictable) is deliberately NOT ported here.
"""

from __future__ import annotations

import os
import re
from types import SimpleNamespace

import pytest

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

from sglang.srt.managers import weg2_store_told as told_mod  # noqa: E402

# The scheduler module is NOT imported: without a GPU its import dies in the
# Triton device probe (torchao -> torch.utils._triton, IndexError). The shipped
# source is read from disk instead, and the one method under test is compiled
# out of it -- the base and the fix are measured through the same door.
_SCHED_PY = os.path.join(os.path.dirname(told_mod.__file__), "scheduler.py")


def _scheduler_src() -> str:
    with open(_SCHED_PY) as f:
        return f.read()


# --------------------------------------------------------------------------
# 1. the told verdict is kept until the request leaves the queue
# --------------------------------------------------------------------------


class _Tree:
    """A PP0 tree double whose store read terminated with ``completed``."""

    def __init__(self):
        self.prefetch_loaded_tokens_by_reqid = {}
        self._completed = {}

    def check_prefetch_progress(self, rid):
        return True

    def completed_prefetch_tokens(self, rid):
        return self._completed.get(rid)

    def pop_prefetch_loaded_tokens(self, rid):
        return self.prefetch_loaded_tokens_by_reqid.pop(rid, 0)


class _Sched:
    def __init__(self, pp_rank=0):
        self.ps = SimpleNamespace(pp_rank=pp_rank, pp_size=3, tp_size=1)
        self.enable_hicache_storage = True
        self.tree_cache = _Tree()
        self.waiting_queue = []
        self.pp_flip_counters = None

    def _prefetch_kvcache(self, req, rematch=True, limit_tokens=None):
        return "issued"


def _scheduler_told_gate():
    """The callable the scheduler's admission loop applies at the #1400 told
    site, resolved from the shipped source (so the base and the fix are
    measured through the same door)."""
    src = _scheduler_src()
    i = src.index("if self.enable_hicache_storage and _pp_group and _told_armed:")
    blk = src[i:i + 900]
    m = re.search(r"_told_loaded = ([\w\.]+)\(", blk)
    assert m, blk
    name = m.group(1)
    if name == "weg2_store_told.admission":
        return told_mod.admission
    assert name == "_p_intake.told_admission", name
    from sglang.srt.weg2 import p_intake

    return lambda s, r, note: p_intake.told_admission(s, r, note, told_mod.admission)


def _published(sched, rid, completed):
    req = SimpleNamespace(rid=rid, prefetch_deferred=None, origin_input_ids=None)
    sched.waiting_queue.append(req)
    assert told_mod.armed(sched)
    told_mod.intake(sched, req, lambda k: None)
    sched.tree_cache._completed[rid] = completed
    wire = told_mod.pp0_publish(sched, [])
    assert [(w.rid, w.told) for w in wire] == [(rid, completed)]
    return req


@pytest.fixture(autouse=True)
def _env(monkeypatch):
    monkeypatch.delenv(told_mod.ENV_ARMED, raising=False)
    yield


def test_the_told_verdict_survives_a_visit_the_adder_could_not_seat():
    """53572be2: told published at 20:40:58, the visit in the same pass ran
    beside d762c570's chunk and was not seated; with the verdict consumed
    there the rid skipped as weg2_store_told_pending until the 503."""
    gate = _scheduler_told_gate()
    s = _Sched(0)
    req = _published(s, "53572be2", completed=0)  # #1416 anchor clamp -> 0
    skips = []
    note = lambda kind, rid: skips.append(kind)  # noqa: E731
    assert gate(s, req, note) == 0            # visit 1: verdict taken ...
    assert req._weg2_prefix_cap == 0
    # ... the adder did not seat it (chunk budget spent): it stays queued
    for _ in range(3):                         # every later visit admits
        assert gate(s, req, note) == 0
    assert told_mod.SKIP_TOLD_PENDING not in skips
    assert req._weg2_prefix_cap == 0


def test_a_kept_verdict_goes_with_the_request_and_a_fresh_told_replaces_it():
    from sglang.srt.weg2 import p_intake

    gate = lambda s, r, note: p_intake.told_admission(s, r, note, told_mod.admission)  # noqa: E731
    s = _Sched(0)
    req = _published(s, "aaaa0001", completed=4096)
    s.tree_cache.prefetch_loaded_tokens_by_reqid["aaaa0001"] = 4096
    assert gate(s, req, lambda k, r: None) == 4096
    assert gate(s, req, lambda k, r: None) == 4096      # kept, credit kept
    # the pass seats it: the queue loses it, settle drops the verdict
    s.waiting_queue.remove(req)
    assert p_intake.settle_told(s, s.waiting_queue) == 1
    assert getattr(s, p_intake.KEPT_ATTR) == {}
    # a NEW request object under the rid (re-intake) never inherits it
    skips = []
    other = SimpleNamespace(rid="aaaa0001", prefetch_deferred=None)
    assert gate(s, other, lambda k, r: skips.append(k)) is None
    assert skips == [told_mod.SKIP_TOLD_PENDING]
    # a fresh told for a still-queued request replaces the kept one
    s2 = _Sched(0)
    r2 = _published(s2, "bbbb0001", completed=0)
    assert gate(s2, r2, lambda k, r: None) == 0
    s2.tree_cache._completed["bbbb0001"] = 8192
    s2._weg2_store_told["bbbb0001"] = 8192               # re-published
    assert gate(s2, r2, lambda k, r: None) == 0          # credit popped fresh
    assert r2._weg2_prefix_cap == 8192
    assert getattr(s2, p_intake.KEPT_ATTR)["bbbb0001"].told == 8192


def test_the_scheduler_settles_kept_verdicts_where_the_queue_is_committed():
    src = _scheduler_src()
    i = src.index("self.waiting_queue = [x for x in self.waiting_queue if x not in can_run_set]")
    assert "_p_intake.settle_told(self, self.waiting_queue)" in src[i:i + 300]
