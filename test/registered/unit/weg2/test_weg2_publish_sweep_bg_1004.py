# SPDX-License-Identifier: Apache-2.0
"""PUBLISH-SWEEP-BG (04.10., INT8 boot 4cf740ad50).

After a long D wake phase 110-121 finished-request nodes sat un-backed
(D-IDLE-PUBLISH runs only when D is idle, one node per pass) and the D->P
flip's flush paid 1.9-2.5 s of write_backup issue for them (ep15 flush 3.6 s,
ep35 4.26 s). The same sweep now also runs between D decode rounds.

Pinned:
* the tick is a no-op when the switch is off / not group D / D idle (the idle
  publisher's case) / waiting / dormant / paused / chunked;
* cadence = forward_ct % EVERY only (replicated, no clock), once per forward;
* the background sweep never issues a node a running request references
  (lock_ref > 0) nor one whose parent is not backed yet, never anchor-only
  backups; ``background=False`` (the flush) is the old walk, unchanged:
  the same tree publishes parent-first and in full;
* nothing is lost: what the background pass left un-backed is exactly what the
  flush sweep still publishes;
* the scheduler hook sits at the group-uniform point after the evictor.
"""

from __future__ import annotations

import inspect
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
    monkeypatch.delenv("SGLANG_WEG2_PUBLISH_SWEEP_BG", raising=False)


class _Batch:
    def __init__(self, n=0):
        self.reqs = [object()] * n

    def is_empty(self):
        return not self.reqs


class _Tree:
    def __init__(self):
        self.calls = []

    def publish_unbacked_sweep(self, max_issue=64, **kw):
        self.calls.append((max_issue, kw))
        return {"unbacked": 3, "issued": 1, "pending": 1, "skipped_bg": 2, "refused": 0, "issue_ms": 1.0}


def _sched(tree=None, *, running=1, waiting=0, dormant=False, paused=False, fct=64, chunked=None, rank=0):
    return types.SimpleNamespace(
        tp_rank=rank, ps=types.SimpleNamespace(tp_rank=rank),
        enable_hierarchical_cache=True, tree_cache=tree if tree is not None else _Tree(),
        running_batch=_Batch(running), waiting_queue=[object()] * waiting, chunked_req=chunked,
        anchor_tails=None, weg2_dormant=dormant, _engine_paused=paused, forward_ct=fct)


def test_switch_default_on_and_knobs():
    from sglang.srt.environ import envs

    assert envs.SGLANG_WEG2_PUBLISH_SWEEP_BG.get() is True
    assert envs.SGLANG_WEG2_PUBLISH_SWEEP_BG_EVERY.get() == 64
    assert envs.SGLANG_WEG2_PUBLISH_SWEEP_BG_MAX_TOKENS.get() == 8192
    assert envs.SGLANG_WEG2_PUBLISH_SWEEP_BG_MAX_ISSUE.get() == 1


def test_tick_runs_one_bounded_background_pass(armed_d):
    nb = _nb()
    s = _sched()
    st = nb.bg_publish_tick(s)
    assert st is not None
    assert s.tree_cache.calls == [(1, {"background": True, "bg_max_tokens": 8192})]


def test_tick_inert_outside_its_case(armed_d, monkeypatch):
    nb = _nb()
    for kw in (dict(running=0), dict(waiting=1), dict(dormant=True), dict(paused=True),
               dict(chunked=object()), dict(fct=63), dict(fct=16), dict(fct=0)):
        s = _sched(**kw)
        assert nb.bg_publish_tick(s) is None, kw
        assert s.tree_cache.calls == [], kw
    monkeypatch.setenv("SGLANG_WEG2_GROUP", "P")
    s = _sched()
    assert nb.bg_publish_tick(s) is None and s.tree_cache.calls == []
    monkeypatch.setenv("SGLANG_WEG2_GROUP", "D")
    monkeypatch.setenv("SGLANG_WEG2_PUBLISH_SWEEP_BG", "0")
    s = _sched()
    assert nb.bg_publish_tick(s) is None and s.tree_cache.calls == []


def test_cadence_is_forward_ct_only_and_once_per_forward(armed_d, monkeypatch):
    nb = _nb()
    s = _sched(fct=64)
    assert nb.bg_publish_tick(s) is not None
    assert nb.bg_publish_tick(s) is None          # same forward count: no repeat
    s.forward_ct = 128
    assert nb.bg_publish_tick(s) is not None
    monkeypatch.setenv("SGLANG_WEG2_PUBLISH_SWEEP_BG_EVERY", "4")
    s.forward_ct = 132
    assert nb.bg_publish_tick(s) is not None and len(s.tree_cache.calls) == 3
    monkeypatch.setenv("SGLANG_WEG2_PUBLISH_SWEEP_BG_MAX_ISSUE", "3")
    s.forward_ct = 136
    nb.bg_publish_tick(s)
    assert s.tree_cache.calls[-1][0] == 3


def test_tick_never_raises(armed_d):
    nb = _nb()

    class _Boom(_Tree):
        def publish_unbacked_sweep(self, **kw):
            raise RuntimeError("x")

    assert nb.bg_publish_tick(_sched(_Boom())) is None


def test_scheduler_hook_is_at_the_uniform_point_after_the_evictor():
    from sglang.srt.managers import scheduler

    src = inspect.getsource(scheduler)
    ev = src.index("_d_kv_evict_step(self)")
    bg = src.index("_weg2_flush_nonblock.bg_publish_tick(self)")
    pre = src.index("prefetch_verdicts = self.__dict__.pop(\"_pass_prefetch_verdicts\", None)")
    assert ev < bg < pre
    assert "self.enable_hierarchical_cache or self.server_args.enable_flexkv" in src[ev - 3500:ev]


# ---- the sweep itself ---------------------------------------------------------------

class _CD:
    def __init__(self, value=(1,), lock_ref=0):
        self.value = value
        self.lock_ref = lock_ref


class _Node:
    def __init__(self, parent, nid, lock_ref=0, backuped=False, pending=None):
        from sglang.srt.mem_cache.unified_radix_cache import BASE_COMPONENT_TYPE

        self.parent = parent
        self.id = nid
        self.children = {}
        self.evicted = False
        self.backuped = backuped
        self.l3_present = False
        self.write_through_pending_id = pending
        self.key = [nid] * 4
        self.component_data = {BASE_COMPONENT_TYPE: _CD(lock_ref=lock_ref)}
        if parent is not None:
            parent.children[nid] = self


class _Fake:
    """Just enough of UnifiedRadixCache for publish_unbacked_sweep."""

    def __init__(self, root):
        self.cache_controller = types.SimpleNamespace(_draft_l3_write_issued=0, _draft_l3_write_refused=0)
        self.disable = False
        self.root_node = root
        self.ongoing_write_through = {}
        self.ongoing_backup = {}
        self.issued = []
        self.anchor_only = []
        from sglang.srt.mem_cache.unified_radix_cache import BASE_COMPONENT_TYPE

        self.component_protected_size_ = {BASE_COMPONENT_TYPE: 0}
        self._mamba_pin_budget = 8

    def _weg2_publish_window(self):
        return 1 << 30

    def _weg2_anchor_only_candidate(self, node):
        return getattr(node, "anchor_only", False)

    def write_backup_anchor_only(self, node):
        self.anchor_only.append(node.id)
        return 1

    def _mamba_pins_held(self):
        return 0

    def write_backup(self, node):
        # the real one recurses into an un-backed parent first
        if node.parent is not self.root_node and not node.parent.backuped:
            if self.write_backup(node.parent) <= 0:
                return 0
        node.backuped = True
        self.issued.append(node.id)
        return 1


def _tree():
    from sglang.srt.mem_cache.unified_radix_cache import UnifiedRadixCache  # noqa: F401

    root = _Node(None, 0)
    a = _Node(root, 1)            # chain a -> b -> c, all un-backed, unlocked
    b = _Node(a, 2)
    c = _Node(b, 3)
    d = _Node(root, 4, lock_ref=1)      # in use by a running request
    e = _Node(d, 5)                     # un-backed child of the locked, un-backed d
    f = _Node(root, 6, backuped=True)
    g = _Node(f, 7)                     # un-backed child of a backed parent
    h = _Node(root, 8)
    h.anchor_only = True                # KV backed elsewhere, only the anchor open
    h.backuped = True
    return root, dict(a=a, b=b, c=c, d=d, e=e, f=f, g=g, h=h)


def _sweep(fake, **kw):
    from sglang.srt.mem_cache.unified_radix_cache import UnifiedRadixCache

    return UnifiedRadixCache.publish_unbacked_sweep(fake, **kw)


def test_background_pass_skips_locked_and_unbacked_parent_and_anchor_only():
    root, n = _tree()
    fk = _Fake(root)
    st = _sweep(fk, max_issue=64, background=True)
    # `backuped` is the host_value, set at ISSUE: a chain goes parent-first in
    # one pass (a, b, c; g under the backed f). Not touched: d (a running
    # request references it), e (its parent d is un-backed), h (anchor-only:
    # the flush's job).
    assert sorted(fk.issued) == [1, 2, 3, 7]
    assert fk.anchor_only == []
    assert fk.issued.index(1) < fk.issued.index(2) < fk.issued.index(3)
    assert st["skipped_bg"] == 2 and st["issued"] == 4


def test_background_pass_max_issue_bounds_the_work():
    root, n = _tree()
    fk = _Fake(root)
    _sweep(fk, max_issue=1, background=True)
    assert len(fk.issued) == 1


def test_flush_sweep_after_background_publishes_the_rest_nothing_lost():
    root, n = _tree()
    bgf = _Fake(root)
    for _ in range(4):                      # background passes until it can do no more
        _sweep(bgf, max_issue=1, background=True)
    left_after_bg = sorted(x.id for x in n.values()
                           if not x.backuped or getattr(x, "anchor_only", False))
    # the flush (background=False): drive it on a fresh copy without bg
    root2, n2 = _tree()
    plain = _Fake(root2)
    _sweep(plain, max_issue=256)
    # unlock d as at the flip (no request runs)
    from sglang.srt.mem_cache.unified_radix_cache import BASE_COMPONENT_TYPE

    for x in n.values():
        x.component_data[BASE_COMPONENT_TYPE].lock_ref = 0
    after = _Fake(root)
    after.issued = []
    _sweep(after, max_issue=256)
    # every node ends backed in both orders: bg-then-flush == flush alone
    assert all(x.backuped for x in n.values()) and all(x.backuped for x in n2.values())
    # the flush after the bg pass issued strictly the not-yet-backed nodes
    assert set(bgf.issued).isdisjoint(after.issued)
    assert set(bgf.issued) | set(after.issued) >= {1, 2, 3, 4, 5, 7}
    assert left_after_bg  # the bg pass really did leave work for the flush


def test_flush_sweep_default_is_the_old_walk_parent_first():
    root, n = _tree()
    fk = _Fake(root)
    st = _sweep(fk, max_issue=256)          # background defaults to False
    assert "skipped_bg" not in st
    # locked d and its child e ARE published by the flush (nothing runs there),
    # a chain parent-first, anchor-only candidate handled
    order = fk.issued
    assert order.index(1) < order.index(2) < order.index(3)
    assert order.index(4) < order.index(5)
    assert fk.anchor_only == [8]
    assert set(order) == {1, 2, 3, 4, 5, 7}


def test_sweep_signature_background_defaults_false():
    from sglang.srt.mem_cache.unified_radix_cache import UnifiedRadixCache

    p = inspect.signature(UnifiedRadixCache.publish_unbacked_sweep).parameters
    assert p["background"].default is False


# ---- review 1270 additions ------------------------------------------------------------

def test_size_threshold_leaves_long_nodes_to_the_flush():
    root, n = _tree()
    n["a"].key = list(range(9000))          # a P hand-over sized node
    fk = _Fake(root)
    st = _sweep(fk, max_issue=64, background=True, bg_max_tokens=8192)
    assert 1 not in fk.issued and 2 not in fk.issued and 3 not in fk.issued   # a too long, b/c wait for it
    assert fk.issued == [7]
    assert st["skipped_bg"] >= 3
    # 0 = no limit
    root, n = _tree()
    n["a"].key = list(range(9000))
    fk = _Fake(root)
    _sweep(fk, max_issue=64, background=True, bg_max_tokens=0)
    assert 1 in fk.issued
    # the flush ignores the threshold (background False)
    root, n = _tree()
    n["a"].key = list(range(9000))
    fk = _Fake(root)
    _sweep(fk, max_issue=256, bg_max_tokens=8192)
    assert 1 in fk.issued


def test_background_claim_never_spills_the_arena():
    """The bg flag is up exactly while write_backup runs under a BG sweep (the direct claim reads it
    before _w3_arena_spill); the flush never sets it."""
    from sglang.srt.mem_cache import unified_radix_cache as urc

    seen = []

    class _Rec(_Fake):
        def write_backup(self, node):
            seen.append(bool(getattr(self, "_weg2_bg_publish", False)))
            return super().write_backup(node)

    root, _n = _tree()
    fk = _Rec(root)
    _sweep(fk, max_issue=64, background=True)
    assert seen and all(seen) and fk._weg2_bg_publish is False
    seen.clear()
    root, _n = _tree()
    fk = _Rec(root)
    _sweep(fk, max_issue=64)
    assert seen and not any(seen)
    src = inspect.getsource(urc.UnifiedRadixCache._weg2_direct_claim)
    i = src.index("_w3_arena_spill(pool, len(hashes)")
    assert 'getattr(self, "_weg2_bg_publish", False)' in src[i - 300:i]


def test_marker_line_at_most_once_per_30s(armed_d, caplog):
    import logging

    nb = _nb()
    s = _sched(fct=64)
    with caplog.at_level(logging.INFO):
        for k in range(1, 6):
            s.forward_ct = 64 * k
            nb.bg_publish_tick(s)
    assert len(s.tree_cache.calls) == 5
    assert caplog.text.count("WEG2-PUBLISH-SWEEP-BG pass=") == 1


def test_two_ranks_take_the_same_decision_at_every_tick(armed_d):
    """TP lockstep: both ranks see the same replicated gate inputs; the tick must decide alike at every
    forward count and every state (a rank-local gate would split the group's publish)."""
    nb = _nb()
    a, b = _sched(rank=0), _sched(rank=1)
    states = [dict(), dict(running=0), dict(waiting=2), dict(dormant=True), dict(paused=True),
              dict(chunked=object())]
    for fct in range(0, 400):
        for st in states:
            for sc in (a, b):
                sc.forward_ct = fct
                sc.running_batch = _Batch(st.get("running", 1))
                sc.waiting_queue = [object()] * st.get("waiting", 0)
                sc.weg2_dormant = st.get("dormant", False)
                sc._engine_paused = st.get("paused", False)
                sc.chunked_req = st.get("chunked")
                sc._weg2_bg_publish_last_fct = None
            ra, rb = nb.bg_publish_tick(a), nb.bg_publish_tick(b)
            assert (ra is None) == (rb is None), (fct, st)
    assert a.tree_cache.calls == b.tree_cache.calls and a.tree_cache.calls


def test_dual_layout_keeps_its_default_no_background_sweep(armed_d, monkeypatch):
    """INT8+L15 integration (04.10.): the y9e port is flip/INT8 only. The dual layout
    (SGLANG_WEG2_DUAL_LAYOUT=1) never runs the default-ON background sweep, so the dual
    default of b9j-b9l stays unchanged; the flip/INT8 form (no dual env) still does."""
    nb = _nb()
    monkeypatch.setenv("SGLANG_WEG2_DUAL_LAYOUT", "1")
    assert nb.bg_publish_on() is False
    s = _sched()
    assert nb.bg_publish_tick(s) is None and s.tree_cache.calls == []
    monkeypatch.delenv("SGLANG_WEG2_DUAL_LAYOUT", raising=False)
    assert nb.bg_publish_on() is True
    s = _sched()
    assert nb.bg_publish_tick(s) is not None and len(s.tree_cache.calls) == 1
