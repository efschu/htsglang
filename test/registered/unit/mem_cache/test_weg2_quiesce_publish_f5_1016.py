# SPDX-License-Identifier: Apache-2.0
"""F5 (NF-PGAP 1016): the P->D quiesce of 11:29:08 -- 24 /flush_cache polls, 14278 ms, refused by the group's own
publish in flight (boot ..._1f4b3e73d4_1016_103428).

MEASURED (P.log 11:29:08-11:29:22): 200 ``WEG2 PUBLISH-SWEEP`` lines against 40 ``#1470 FLUSH-PUBLISH`` on the three
ranks: between the polls the BUBBLE PUBLISHER swept every 0.2 s of P's idle loop and issued write-throughs whose acks
and ``#106S SIDECAR-WRITE-ISSUED`` store writes held the #1268 vote (``not-idle because: hicache_backup(1)`` /
``hicache_write_through``); 31 ``#1427r CLAIM-RELEASE site=claim_refused fresh=16`` and 70 ``#1421 BACKUP-REFUSED
why=arena_claim``: each sweep's first claim on the full arena was a real one, partially taken and given back.

(a) While a quiesce is open (first poll until the reset or a launched batch) the bubble publisher issues no sweep --
the flush's #1470 loop publishes alone; no node is lost. (b) MEMO-LITE: a node whose last claim was refused for
room is asked ``claim_would_refuse`` before any claim, also as a sweep's first claim. (c) ``#1470 FLUSH-UNBACKED``:
what goes into the reset un-backed is named. The NF-W3-DRAIN invariant (an un-acked store backup blocks the vote,
test_weg2_backup_ack_drain_on_vote_1014.py) is untouched.
"""
from __future__ import annotations

import inspect
import logging
import os
import types

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

import pytest

from sglang.srt.managers import weg2_bubble_publish as BP
from sglang.srt.managers import weg2_flush_nonblock as NB
from test_sweep_full_arena_skip_1007 import _DirectPool, _sweep, _tree


class _SweepTree:
    def __init__(self):
        self.sweeps = 0
        self.cache_controller = types.SimpleNamespace()

    def publish_unbacked_sweep(self, max_issue=64, **kw):
        self.sweeps += 1
        return {"issued": 1, "unbacked": 30}


def _sched():
    return types.SimpleNamespace(enable_hicache_storage=True, tree_cache=_SweepTree())


def _idle_loop(monkeypatch, sched, seconds, step=0.05):
    """P's idle loop in the quiesce: ``bubble_begin`` on every pass (``MIN_INTERVAL_S`` paces the sweeps)."""
    clock = {"t": 1000.0}
    monkeypatch.setattr(BP.time, "monotonic", lambda: clock["t"])
    for _ in range(int(seconds / step)):
        BP.bubble_begin(sched)
        clock["t"] += step


def test_11_29_08_the_bubble_publisher_issues_no_sweep_while_the_quiesce_is_open(monkeypatch):
    monkeypatch.setenv(BP.ENV, "1")
    sched = _sched()
    BP.quiesce_begin(sched)                       # the first /flush_cache poll of the quiesce
    _idle_loop(monkeypatch, sched, 14.0)          # 11:29:08 -> 11:29:22
    assert sched.tree_cache.sweeps == 0, sched.tree_cache.sweeps
    assert sched._weg2_quiesce_paused > 0
    BP.bubble_end(sched)                          # a batch is launched: the quiesce is over, bubbles publish again
    assert not sched._weg2_quiesce_open
    _idle_loop(monkeypatch, sched, 1.0)
    assert sched.tree_cache.sweeps > 0


def test_without_an_open_quiesce_the_bubble_publisher_sweeps_as_before(monkeypatch):
    """Control (the 1f4b3e73d4 behaviour in the same window): ~70 opportunistic sweeps in 14 s."""
    monkeypatch.setenv(BP.ENV, "1")
    sched = _sched()
    _idle_loop(monkeypatch, sched, 14.0)
    assert sched.tree_cache.sweeps >= 50   # MIN_INTERVAL_S 0.2 s on a 0.05 s pass


def test_the_flush_opens_the_quiesce_names_the_unbacked_and_closes_it_at_the_reset():
    from sglang.srt.managers.scheduler import Scheduler

    src = inspect.getsource(Scheduler.flush_cache)
    assert src.index("_f5_bubble.quiesce_begin(self)") < src.index('_sweep = getattr(self.tree_cache, "publish_unbacked_sweep"')
    i_note = src.index("note_unbacked_at_reset(self.tree_cache")
    assert i_note < src.index("self.tree_cache.reset()"), "named BEFORE the reset drops it"
    assert src.index('_f5_bubble.quiesce_end(self, "reset")') < src.index("self.tree_cache.reset()")


def test_memo_lite_no_futile_first_claim_per_sweep_while_the_room_is_not_there(monkeypatch):
    """Full arena: the first sweep makes ONE real claim (SWEEP-FULL-ARENA asks the rest by room). RED before: every
    later sweep makes that first futile claim again (31 x CLAIM-RELEASE in the window). GREEN: none, until the room
    for the node's whole need is there -- then it is claimed and issued as before."""
    cache = _tree()
    pool = _DirectPool(room=())
    s1 = _sweep(cache, pool)
    assert len(pool.claims) == 1 and s1["issued"] == 0
    for _ in range(5):
        s = _sweep(cache, pool)
        assert s["issued"] == 0 and s.get("memo_skipped", 0) >= 1, s
    assert len(pool.claims) == 1, f"futile claims repeated: {pool.claims}"
    first = pool.claims[0]
    pool.room.add(first)                          # the room for that node's whole need appears
    s = _sweep(cache, pool)
    assert s["issued"] >= 1 and first in pool.claims[1:], pool.claims


def test_unbacked_census_names_device_kv_that_is_neither_on_the_host_nor_in_the_store(caplog):
    cache = _tree()
    nodes = [n for n in cache._collect_all_nodes() if n is not cache.root_node]
    c = NB.unbacked_census(cache)
    assert c["nodes"] == len(nodes) and c["tokens"] == sum(len(n.key) for n in nodes)
    import torch
    from sglang.srt.mem_cache.unified_cache_components.tree_component import ComponentType

    nodes[0].component_data[ComponentType.FULL].host_value = torch.arange(len(nodes[0].key))   # backed on the host
    nodes[1].l3_present = True                    # its KV is in the store
    c2 = NB.unbacked_census(cache)
    assert c2["nodes"] == len(nodes) - 2
    with caplog.at_level(logging.INFO):
        NB.note_unbacked_at_reset(cache, site="flush")
    line = [r.getMessage() for r in caplog.records if "FLUSH-UNBACKED" in r.getMessage()]
    assert line and "nodes=%d" % c2["nodes"] in line[-1] and "tokens=%d" % c2["tokens"] in line[-1], line


def test_every_rank_opens_the_quiesce_where_it_receives_the_poll_and_the_sleep_leg_ends_it():
    """A follower only forwards /flush_cache (CTRL-RECV -> CTRL-FWD, P.log 11:29:08 PP1) and never runs flush_cache
    per poll -- in the window PP1 was the blocking rank: the quiesce must open at the intake on every rank."""
    from sglang.srt.managers.scheduler_components import request_receiver as rr

    src = inspect.getsource(rr)
    i = src.index("#1458 CTRL-RECV kind=%s")
    j = src.index("_f5_bubble.quiesce_begin(self)")
    k = src.index('_f5_bubble.quiesce_end(self, "sleep")')
    assert i < j and i < k
