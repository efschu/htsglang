# SPDX-License-Identifier: Apache-2.0
"""B1 (30.09., NF y4i): the HiCache publish leaves the D->P flip's quiesce.

DER BEFUND (y4i 09301011, 12 D->P-Flips, Front-Uhr ms-genau): die erste
Quiesce-/flush_cache wurde in 11/12 Flips abgewiesen ("hicache_backup(n)"),
#1470 FLUSH-PUBLISH wartete davor 32-166 ms, FLIP begin -> Quiesce fertig
Median 98 ms (53-410); D war vor jedem Flip 137-333 ms leer.

(1) D-IDLE-PUBLISH: der Sweep aus Scheduler.on_idle -- nur bei leerem
    Lauf/Warteschlange, nicht schlafend, ohne Uhr, mit Knotendeckel.
(2) FLUSH-QUIESCE-NONBLOCK: die Quiesce antwortet "quiesced", wenn die
    einzigen Blocker aller Raenge die eigene Sicherung sind; Baum, Pools und
    laufende Schreibvorgaenge bleiben -- der Schlaf-Leg drained, publiziert,
    joint und setzt zurueck, bevor kv_cache pausiert.
"""

from __future__ import annotations

import inspect
import logging
import os
import types

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

import pytest

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=20, suite="stage-a-test-cpu")


def _nb():
    from sglang.srt.managers import weg2_flush_nonblock as nb

    return nb


@pytest.fixture
def armed_d(monkeypatch):
    monkeypatch.setenv("SGLANG_WEG2_GROUP", "D")
    monkeypatch.setenv("SGLANG_WEG2_ENABLE_D_IDLE_PUBLISH", "1")
    monkeypatch.setenv("SGLANG_WEG2_ENABLE_FLUSH_QUIESCE_NONBLOCK", "1")


class _Batch:
    def __init__(self, n=0):
        self.reqs = [object()] * n

    def is_empty(self):
        return not self.reqs


class _Tree:
    def __init__(self, sweeps=(), group_max=0):
        self.sweeps = list(sweeps)
        self.sweep_calls = []
        self.drains = 0
        self.resets = 0
        self.group_max_calls = []
        self._gm = group_max

    def publish_unbacked_sweep(self, max_issue=64, **_):
        self.sweep_calls.append(max_issue)
        return dict(self.sweeps.pop(0)) if self.sweeps else {"unbacked": 0, "issued": 0, "pending": 0}

    def writing_check(self, write_back=False):
        if write_back:
            self.drains += 1

    def hicache_group_max(self, vec, label=""):
        self.group_max_calls.append((list(vec), label))
        return [max(int(vec[0]), int(self._gm))]

    def reset(self):
        self.resets += 1


def _sched(tree, *, blockers=("hicache_backup(1)",), group_idle=False, running=0, waiting=0,
           dormant=False, fct=7):
    s = types.SimpleNamespace(
        enable_hierarchical_cache=True, tree_cache=tree, running_batch=_Batch(running),
        waiting_queue=[object()] * waiting, chunked_req=None, anchor_tails=None,
        weg2_dormant=dormant, _engine_paused=False, forward_ct=fct,
        ps=types.SimpleNamespace(pp_size=1, pp_rank=0))
    s.idle_blockers = lambda exempt_prefetch=(): list(blockers)
    s.group_idle_verdict = lambda tp_group_verdict=False: (group_idle, "stub verdict")
    return s


# ---- the switches ------------------------------------------------------------------

def test_switches_default_on_after_metal():
    # y4l metal (rc12z30y4l, -clk-po-pm-b1): on by default; "off" via env
    from sglang.srt.environ import envs

    assert envs.SGLANG_WEG2_ENABLE_D_IDLE_PUBLISH.get() is True
    assert envs.SGLANG_WEG2_ENABLE_FLUSH_QUIESCE_NONBLOCK.get() is True
    assert envs.SGLANG_WEG2_FLUSH_NONBLOCK_GROUPS.get() == "D"
    assert envs.SGLANG_WEG2_D_IDLE_PUBLISH_MAX_ISSUE.get() == 1


def test_groups_gate(monkeypatch, armed_d):
    nb = _nb()
    assert nb.idle_publish_on() and nb.quiesce_nonblock_on()
    monkeypatch.setenv("SGLANG_WEG2_GROUP", "P")
    assert not nb.idle_publish_on() and not nb.quiesce_nonblock_on()


def test_hard_blockers_keep_every_other_clause():
    nb = _nb()
    own = ["hicache_write_through(2)", "hicache_backup(1)", "hicache_prefetch(1: ab)",
           "waiting_queue", "hicache_load_back(1)"]
    assert nb.hard_blockers(own) == ["hicache_prefetch(1: ab)", "waiting_queue", "hicache_load_back(1)"]


# ---- part 2: the quiesce verdict ---------------------------------------------------

def test_quiesce_verdict_soft_only_answers_quiesced(armed_d):
    nb = _nb()
    tree = _Tree()
    s = _sched(tree, blockers=["hicache_write_through(1)", "hicache_backup(1)"])
    detail = nb.quiesce_verdict(s, group_idle=False, tp_group_verdict=True)
    assert detail and "quiesced" in detail and "BEFORE the kv_cache pause" in detail
    assert tree.group_max_calls == [([0], "flush_cache/nonblock_hard")]


def test_quiesce_verdict_refuses_on_any_hard_blocker_in_the_group(armed_d):
    nb = _nb()
    # this rank soft only, another rank hard (the reduced MAX says 1)
    tree = _Tree(group_max=1)
    s = _sched(tree, blockers=["hicache_backup(1)"])
    assert nb.quiesce_verdict(s, group_idle=False, tp_group_verdict=True) is None
    # this rank hard
    tree = _Tree()
    s = _sched(tree, blockers=["hicache_backup(1)", "hicache_prefetch(1: ab)"])
    assert nb.quiesce_verdict(s, group_idle=False, tp_group_verdict=True) is None
    assert tree.group_max_calls == [([1], "flush_cache/nonblock_hard")]


def test_quiesce_verdict_inert_outside_its_case(armed_d, monkeypatch):
    nb = _nb()
    tree = _Tree()
    s = _sched(tree)
    assert nb.quiesce_verdict(s, group_idle=True, tp_group_verdict=True) is None     # idle: the reset
    assert nb.quiesce_verdict(s, group_idle=False, tp_group_verdict=False) is None   # sleep leg's own flush
    s.ps.pp_size = 3
    assert nb.quiesce_verdict(s, group_idle=False, tp_group_verdict=True) is None    # PP: the vote lap
    s.ps.pp_size = 1
    monkeypatch.setenv("SGLANG_WEG2_ENABLE_FLUSH_QUIESCE_NONBLOCK", "0")
    assert nb.quiesce_verdict(s, group_idle=False, tp_group_verdict=True) is None
    assert tree.group_max_calls == []    # no collective was posted in any of these


def test_quiesce_sweep_blocking(armed_d, monkeypatch):
    nb = _nb()
    s = _sched(_Tree())
    assert nb.quiesce_sweep_blocking(s, {}, True) is False
    assert nb.quiesce_sweep_blocking(s, {"unbacked": 3, "issued": 3}, True) is False
    assert nb.quiesce_sweep_blocking(s, {"unbacked": 5, "issued": 3}, True) is True  # a node left
    assert nb.quiesce_sweep_blocking(s, {}, False) is True                           # not the quiesce
    monkeypatch.setenv("SGLANG_WEG2_ENABLE_FLUSH_QUIESCE_NONBLOCK", "0")
    assert nb.quiesce_sweep_blocking(s, {}, True) is True


# ---- part 2 through Scheduler.flush_cache ------------------------------------------

def _flush(s, tp=True):
    from sglang.srt.managers.scheduler import Scheduler

    return Scheduler.flush_cache(s, tp_group_verdict=tp)


def test_flush_base_drains_and_refuses(monkeypatch):
    monkeypatch.setenv("SGLANG_WEG2_GROUP", "D")
    monkeypatch.setenv("SGLANG_WEG2_ENABLE_FLUSH_QUIESCE_NONBLOCK", "0")   # the off form
    tree = _Tree(sweeps=[{"unbacked": 1, "issued": 1, "pending": 1}, {"unbacked": 0}])
    s = _sched(tree)
    assert _flush(s) is False                 # the y4i 400: hicache_backup(1)
    assert tree.drains >= 1 and tree.resets == 0


def test_flush_nonblock_issues_does_not_join_and_answers_quiesced(armed_d, caplog):
    tree = _Tree(sweeps=[{"unbacked": 1, "issued": 1, "pending": 1}])
    s = _sched(tree, blockers=["hicache_write_through(1)"])
    with caplog.at_level(logging.INFO):
        assert _flush(s) is True
    assert tree.sweep_calls == [256]          # one issuing pass
    assert tree.drains == 0                   # no write-back join in the quiesce
    assert tree.resets == 0                   # the tree is kept for the sleep leg
    assert "B1 NONBLOCK: issued, not joined here" in caplog.text
    assert "WEG2-FLUSH-NONBLOCK quiesced" in caplog.text


def test_flush_nonblock_keeps_the_1470_loop_when_a_node_is_left(armed_d):
    # pin budget: 3 un-backed, 1 issued -> the blocking #1470 loop, then quiesced
    tree = _Tree(sweeps=[{"unbacked": 3, "issued": 1, "pending": 1},
                         {"unbacked": 2, "issued": 2, "pending": 2},
                         {"unbacked": 0}])
    s = _sched(tree, blockers=["hicache_backup(3)"])
    assert _flush(s) is True
    assert len(tree.sweep_calls) == 3 and tree.drains >= 2 and tree.resets == 0


def test_flush_nonblock_still_refuses_a_hard_blocker(armed_d):
    tree = _Tree(sweeps=[{"unbacked": 0}], group_max=1)
    s = _sched(tree, blockers=["hicache_backup(1)"])
    assert _flush(s) is False and tree.resets == 0


# ---- part 1: the idle publisher ----------------------------------------------------

def test_idle_publish_gate_and_cap(armed_d, monkeypatch):
    nb = _nb()
    monkeypatch.setenv("SGLANG_WEG2_D_IDLE_PUBLISH_MAX_ISSUE", "2")
    for kw in ({"dormant": True}, {"running": 1}, {"waiting": 1}):
        tree = _Tree()
        assert nb.idle_publish(_sched(tree, **kw)) is None and tree.sweep_calls == []
    tree = _Tree(sweeps=[{"unbacked": 3, "issued": 2, "pending": 0}])
    s = _sched(tree)
    st = nb.idle_publish(s)
    assert st["issued"] == 2 and tree.sweep_calls == [2]
    monkeypatch.setenv("SGLANG_WEG2_ENABLE_D_IDLE_PUBLISH", "0")
    assert nb.idle_publish(_sched(_Tree())) is None


def test_idle_publish_walks_once_per_forward(armed_d):
    nb = _nb()
    tree = _Tree(sweeps=[{"unbacked": 1, "issued": 1}, {"unbacked": 0, "issued": 0}])
    s = _sched(tree, fct=10)
    nb.idle_publish(s)                  # issued all it saw -> clean at forward 10
    nb.idle_publish(s)
    nb.idle_publish(s)
    assert tree.sweep_calls == [1]      # no repeat walk while nothing ran
    s.forward_ct = 11                   # a forward ran: walk again
    nb.idle_publish(s)
    assert tree.sweep_calls == [1, 1]
    # a pass left nodes (cap) -> the next idle pass continues
    tree2 = _Tree(sweeps=[{"unbacked": 3, "issued": 1}, {"unbacked": 2, "issued": 1}])
    s2 = _sched(tree2, fct=3)
    nb.idle_publish(s2)
    nb.idle_publish(s2)
    assert tree2.sweep_calls == [1, 1]
    # refused (nothing issuable): no spin
    tree3 = _Tree(sweeps=[{"unbacked": 2, "issued": 0, "refused": 2}])
    s3 = _sched(tree3, fct=5)
    nb.idle_publish(s3)
    nb.idle_publish(s3)
    assert tree3.sweep_calls == [1]


def test_idle_publish_never_raises(armed_d, caplog):
    nb = _nb()

    class _Bad(_Tree):
        def publish_unbacked_sweep(self, **_):
            raise RuntimeError("arena")

    with caplog.at_level(logging.WARNING):
        assert nb.idle_publish(_sched(_Bad())) is None
    assert "WEG2-D-IDLE-PUBLISH raised RuntimeError" in caplog.text


# ---- the wiring and the sleep leg's order (the data-safety premise) ----------------

def test_on_idle_publishes_before_the_idle_test():
    from sglang.srt.managers.scheduler import Scheduler

    src = inspect.getsource(Scheduler.on_idle)
    assert src.index("_weg2_flush_nonblock.idle_publish(self)") < src.index("if not self.is_fully_idle():")


def test_sleep_leg_drains_asserts_flushes_before_the_kv_pause():
    """Part 2 hands the drain, the #1470 publish and the reset to the sleep
    leg; this pins the order that makes it lossless."""
    from sglang.srt.managers.scheduler_components import weight_updater as wu

    src = open(wu.__file__).read()
    i = src.index("    def _weg2_release_memory_occupation_body") if \
        "    def _weg2_release_memory_occupation_body" in src else \
        src.index("self._weg2_drain_hicache_before_sleep()\n        _weg2_ph(\"drain_hicache\")")
    body = src[i:]
    i_drain = body.index("self._weg2_drain_hicache_before_sleep()")
    i_assert = body.index("self._weg2_sleep_idle()")
    # Q-570: the flush runs through the guard that leaves no device value in any rank's tree
    i_flush = body.index("self._weg2_sleep_flush()")
    i_pause = body.index("self.memory_saver_adapter.pause(GPU_MEMORY_TYPE_KV_CACHE)")
    assert i_drain < i_assert < i_flush < i_pause


def test_the_1470_join_and_reset_order_is_unchanged():
    from sglang.srt.managers.scheduler import Scheduler

    src = inspect.getsource(Scheduler.flush_cache)
    i_q = src.index("_weg2_flush_nonblock.quiesce_verdict(")
    i_join = src.index("self._weg2_join_store_writes_before_reset()")
    i_reset = src.index("self.tree_cache.reset()")
    # the quiesced answer returns before any reset; the idle path is #1470b as before
    assert i_q < i_join < i_reset
