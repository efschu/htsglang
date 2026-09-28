"""P4b-fix (28.09., review of 9d1ca0fff0): the settle hold's writer view is RANK-LOCAL.

The hand-off record of a P-prefilled rid is removed at the wake on every D rank
(scheduler._weg2_release_dormant_hold). A rank that did not read it before (the xsn331
race: leg 2 lands on D the moment P finishes) sees "no writer" while its peers see P's
hand-off. In 9d1ca0fff0 that rank put 0 into the group's DUE vote ("decide" = no
re-read) and so vetoed every re-read of the group: the request sat the whole 20 s
bound out although P's pages could be re-read. Before P4b every rank re-read every 2 s.

Fix: (i) "decide" is neutral in the due vote and votes separately; the group skips
the re-read only when every rank decides; (ii) a rank that once read the record keeps
the hand-off in a sticky attribute, so the wake's removal does not turn it into "none".
"""

import os
import threading
import time
import types

import pytest

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

from sglang.srt.managers import scheduler as sched_mod
from sglang.srt.weg2 import handoff as ho
from sglang.srt.weg2 import handoff_keys as hk
from sglang.srt.weg2 import settle_writer as sw
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="stage-a-weg2-unit")

import sys  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import test_weg2_settle_writer_p4b_0928 as base  # noqa: E402  (the #1471 harness of the P4b tests)

S = sched_mod.Scheduler


class _Group:
    """An element-wise MIN over two ranks in lockstep (the TP cpu group's all_reduce)."""

    def __init__(self, n=2):
        self.barrier = threading.Barrier(n)
        self.slots = {}
        self.calls = []

    def bind(self, rank):
        def gmin(flags):
            vals = [1 if f else 0 for f in flags]
            self.slots[rank] = vals
            self.barrier.wait(5)
            out = [min(v) for v in zip(*[self.slots[r] for r in sorted(self.slots)])]
            self.barrier.wait(5)
            if rank == 0:
                self.calls.append(list(out))
            return out
        return gmin


def _run_both(holders):
    out, err = {}, []

    def one(rank, h):
        try:
            out[rank] = h._weg2_post_wake_settle_tick()
        except Exception as e:  # noqa: BLE001
            err.append(e)

    ts = [threading.Thread(target=one, args=(r, h)) for r, h in enumerate(holders)]
    for t in ts:
        t.start()
    for t in ts:
        t.join(10)
    assert not err, err
    return out


@pytest.fixture
def arena(tmp_path, monkeypatch):
    monkeypatch.setenv("SGLANG_HICACHE_ARENA_DIR", str(tmp_path))
    monkeypatch.setenv("SGLANG_WEG2_STORE_SHORT_TAIL", "0")
    (tmp_path / "handoff").mkdir()
    return tmp_path


def test_a_rank_without_the_hand_off_does_not_veto_the_groups_re_read(arena):
    """Rank 0 never read the record (writer none -> decide), rank 1 holds P's chain
    (p-handoff -> poll) and its 2 s timer is due: the group re-reads on both ranks and
    releases nothing (the read is in flight)."""
    g = _Group()
    old = time.monotonic() - 3.0
    r0 = base._req("weg2-1-1", 41021, 40512, last=old)
    r1 = base._req("weg2-1-1", 41021, 40512, last=old)
    setattr(r1, hk.CHAIN_ATTR, ["k0", "k1"])
    hs = []
    for rank, r in enumerate((r0, r1)):
        h = base._holder()
        h.ps = types.SimpleNamespace(tp_size=2)
        h._weg2_group_min_flags = g.bind(rank)
        h.weg2_post_wake_settle = [r]
        hs.append(h)
    assert sw.observe(r0) == sw.NONE and sw.observe(r1) == sw.P_HANDOFF
    out = _run_both(hs)
    assert g.calls[0][:1] == [1], "the group's due vote must be 1 (a no-writer rank is neutral)"
    assert hs[0]._weg2_refetch_one.reissued == ["weg2-1-1"] == hs[1]._weg2_refetch_one.reissued
    assert out == {0: 0, 1: 0} and hs[0].weg2_post_wake_settle == [r0]


def test_every_rank_without_a_writer_still_skips_the_re_read(arena):
    g = _Group()
    old = time.monotonic() - 3.0
    rs = [base._req("weg2-2-2", 41021, 40512, last=old) for _ in range(2)]
    hs = []
    for rank, r in enumerate(rs):
        h = base._holder()
        h.ps = types.SimpleNamespace(tp_size=2)
        h._weg2_group_min_flags = g.bind(rank)
        h.weg2_post_wake_settle = [r]
        hs.append(h)
    out = _run_both(hs)
    assert out == {0: 1, 1: 1}, "all ranks decide: released at once"
    assert hs[0]._weg2_refetch_one.reissued == [] == hs[1]._weg2_refetch_one.reissued


def test_a_hand_off_once_read_survives_the_wakes_removal(arena):
    """The record is read at registration (resolve_chain), then the wake removes it and
    the chain attribute is cleared: the rank still knows P handed this rid off."""
    ho.write("weg2-3-3", [1, 2, 3], ["k0", "k1"])
    r = types.SimpleNamespace(rid="weg2-3-3")
    assert hk.resolve_chain(r, ho.read) == ["k0", "k1"]
    os.remove(ho.path("weg2-3-3"))
    setattr(r, hk.CHAIN_ATTR, None)
    assert sw.observe(r) == sw.P_HANDOFF
