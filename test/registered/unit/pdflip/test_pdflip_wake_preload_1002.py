"""DP-NACHLAUF 02.10.: WAKE-PRELOAD -- the held requests' prefix H2D starts in
the kv resume RPC after the re-zero, group-voted, and the first batch waits on
its producer.

N5q (9126170083) P>D: D's Nachlauf = schedule 232 ms (START-LOADING) + first
extend 332 ms waiting for the KV copy that began only after the reply, the
hold release and the admission. The 18.09. preload ran on a per-rank verdict
and died in PrefixLensRankDivergence (xsn377); here the extents are voted.

Pinned (red before): eligibility (switch, PP/told form, L15 hold-aware, no
hicache); three ranks with the same hold and tree all issue the same loads and
START them; one rank with a different extent -> NO rank issues; an issued
count that differs after equal extents stops by name; the first batch takes
the preload producer only when it loads nothing itself, once; the vote is a
real element-wise MIN/MAX on a 3-process gloo group; wiring: the RPC calls it
after the W26 re-zero check, the scheduler hands the producer over.
"""
from __future__ import annotations

import inspect
import multiprocessing as mp
import os
import socket
from types import SimpleNamespace

import pytest

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

from flliper.srt.pdflip import wake_preload as wp  # noqa: E402


class _Tree:
    def __init__(self):
        self.started = 0
        self.next_producer = 4

    def ready_to_load_host_cache(self):
        self.started += 1
        return self.next_producer


def _rank(exts, pp=1, told=False, hicache=True):
    tree = _Tree()
    sched = SimpleNamespace(
        ps=SimpleNamespace(pp_size=pp, tp_size=3), enable_hierarchical_cache=hicache,
        tree_cache=tree, pdflip_dormant_hold=[SimpleNamespace(rid=f"r{i}") for i in range(len(exts))],
        _told=told)
    upd = SimpleNamespace(scheduler=sched, preloads=0)

    def _pre():
        upd.preloads += 1
        return sum(1 for e in exts if e)

    upd._pdflip_preload_hold = _pre
    sched._exts = list(exts)
    return upd


def _group(monkeypatch, ranks):
    """The vote as the TP group answers it: element-wise min/max over ranks."""
    monkeypatch.delenv(wp.ENV, raising=False)
    monkeypatch.setattr(wp, "_probe_extents", lambda sched, hold: list(sched._exts))
    from flliper.srt.managers import pdflip_store_told as st
    monkeypatch.setattr(st, "armed", lambda s: bool(getattr(s, "_told", False)))
    calls = {"n": 0}

    def gmm(sched, vals):
        # the k-th vote of every rank pairs with the k-th vote of the others
        k = sched.__dict__.setdefault("_vote_k", 0)
        sched._vote_k = k + 1
        cols = []
        for u in ranks:
            s = u.scheduler
            if k == 0:
                cols.append(list(s._exts))
            else:
                cols.append([sum(1 for e in s._exts if e)] if not hasattr(s, "_n_override") else [s._n_override])
        lo = [min(c[i] for c in cols) for i in range(len(vals))]
        hi = [max(c[i] for c in cols) for i in range(len(vals))]
        calls["n"] += 1
        return lo, hi

    monkeypatch.setattr(wp, "group_min_max", gmm)
    return calls


def test_eligibility(monkeypatch):
    from flliper.srt.managers import pdflip_store_told as st
    monkeypatch.setattr(st, "armed", lambda s: bool(getattr(s, "_told", False)))
    monkeypatch.delenv(wp.ENV, raising=False)
    assert wp.ineligible(_rank([5]).scheduler) == ""
    assert wp.ineligible(_rank([5], pp=3).scheduler) == "pp_form"
    assert wp.ineligible(_rank([5], told=True).scheduler) == "told_form"
    assert wp.ineligible(_rank([5], hicache=False).scheduler) == "no_hicache"
    assert wp.ineligible(_rank([5]).scheduler, l15_hold_aware=True) == "l15_hold_aware"
    monkeypatch.setenv(wp.ENV, "0")
    assert wp.ineligible(_rank([5]).scheduler) == "switch_off"


def test_three_equal_ranks_issue_and_start_the_same(monkeypatch):
    ranks = [_rank([98756]) for _ in range(3)]
    _group(monkeypatch, ranks)
    out = [wp.run(u) for u in ranks]
    assert out == [1, 1, 1]
    for u in ranks:
        assert u.preloads == 1 and u.scheduler.tree_cache.started == 1
        assert getattr(u.scheduler.tree_cache, wp.ATTR) == 4


def test_one_rank_with_another_extent_means_no_rank_issues(monkeypatch):
    ranks = [_rank([98756]), _rank([98756]), _rank([40000])]
    _group(monkeypatch, ranks)
    assert [wp.run(u) for u in ranks] == [0, 0, 0]
    assert all(u.preloads == 0 and u.scheduler.tree_cache.started == 0 for u in ranks)


def test_issued_count_mismatch_stops_by_name(monkeypatch):
    ranks = [_rank([98756]) for _ in range(3)]
    ranks[2].scheduler._n_override = 0
    _group(monkeypatch, ranks)
    with pytest.raises(RuntimeError, match="RANKS DISAGREE"):
        wp.run(ranks[0])


def test_empty_hold_and_no_extent_issue_nothing(monkeypatch):
    ranks = [_rank([0]) for _ in range(3)]
    _group(monkeypatch, ranks)
    assert [wp.run(u) for u in ranks] == [0, 0, 0]
    u = _rank([])
    assert wp.run(u) == 0


def test_consumer_index_hands_over_once():
    t = SimpleNamespace()
    setattr(t, wp.ATTR, 4)
    assert wp.consumer_index(t, 7) == 7             # own load wins (newer, same stream)
    assert getattr(t, wp.ATTR) is None
    setattr(t, wp.ATTR, 4)
    assert wp.consumer_index(t, -1) == 4            # nothing of its own: the preload
    assert wp.consumer_index(t, -1) == -1           # consumed once
    assert wp.consumer_index(SimpleNamespace(), -1) == -1


def _gloo_worker(rank, port, vals, out):
    import torch.distributed as dist

    dist.init_process_group("gloo", init_method=f"tcp://127.0.0.1:{port}", rank=rank, world_size=3)
    sched = SimpleNamespace(ps=SimpleNamespace(tp_size=3), tp_cpu_group=dist.group.WORLD)
    out.put((rank, wp.group_min_max(sched, vals[rank])))
    dist.destroy_process_group()


def test_vote_is_a_real_group_min_max_on_gloo():
    s = socket.socket(); s.bind(("127.0.0.1", 0)); port = s.getsockname()[1]; s.close()
    ctx = mp.get_context("spawn")
    out = ctx.Queue()
    vals = {0: [98756, 5], 1: [98756, 7], 2: [40000, 5]}
    ps = [ctx.Process(target=_gloo_worker, args=(r, port, vals, out)) for r in range(3)]
    for p in ps:
        p.start()
    res = dict(out.get(timeout=60) for _ in range(3))
    for p in ps:
        p.join(timeout=30)
    for r in range(3):
        assert res[r] == ([40000, 5], [98756, 7])     # every rank sees the same verdict


def test_wiring():
    from flliper.srt.managers.scheduler import Scheduler
    from flliper.srt.managers.scheduler_components import weight_updater as wu

    src = inspect.getsource(wu)
    i_w26 = src.index("W26 PdFlipWakeInvariantRefused")
    i_pre = src.index("_wpl.run(self, l15_hold_aware=")
    assert i_w26 < i_pre
    assert "_pdflip_wake_preload.consumer_index(" in inspect.getsource(Scheduler)
