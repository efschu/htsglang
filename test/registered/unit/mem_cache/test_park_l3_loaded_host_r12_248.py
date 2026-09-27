"""#248 + #249: after a load-back TP0 gives its arena KV host rows back, and the
Form A workers' byteless mirror rows go with them (R12 STATE verdict).

THE METAL.
* rc12s 17:33:41 (#248, tmp/r989/befund_248_park_l2_pinnt.md): a woken D kept
  ``tree=5213`` arena references until the next reset -- the spans it had
  loaded to the device held their host rows, and P's claims found no slot.
* rc12t (#249, tmp/r989/befund_w88_host_pool.md): the Form A workers keep a
  byteless mirror row of every TP0 host node in a FIXED pool (353573 tokens)
  and free one only on TP0's STATE verdict -- which came only when TP0 itself
  was short. The pool ratcheted until the reset (``R12 SHADOW-OWN-EVICT-
  REFUSED n=1->592``), the prefetch vote took the workers' shortfall, W88.

THE FIX (``weg2.park_l3.release_loaded_host``, called from ``loading_check``):
the span is on the device -> TP0 frees the KV host rows of the loaded path
(the arena page stays COMPLETE), records a STATE event per node, and the next
request broadcast takes the workers' mirror rows with it. The mamba anchor
stays (it may live on the host alone).

Hermetic (no CUDA): the R12 harness (real UnifiedRadixCache host-life methods,
real role plans, pickle broadcast) from test_form_a_host_shadow_r12.py.
"""

from __future__ import annotations

import importlib.util
import os

import pytest

_HERE = os.path.dirname(os.path.abspath(__file__))
_spec = importlib.util.spec_from_file_location(
    "_r12_harness_248", os.path.join(_HERE, "test_form_a_host_shadow_r12.py"))
h = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(h)

try:  # absent on the base -- that absence is part of the red
    from sglang.srt.weg2 import park_l3
except ImportError:  # pragma: no cover - base
    park_l3 = None

pytestmark = pytest.mark.skipif(h.m is None, reason="R12 module absent")

NODE, FULL = h.NODE, h.FULL


class _Ev:
    def query(self):
        return True

    def synchronize(self):
        pass


class _ArenaRank(h._Rank):
    """TP0 owns an arena KV host pool; the workers do not (Form A)."""

    def _weg2_arena_pools(self):
        return {FULL: object()} if self.is_host else {}


def _ranks(n_nodes=3):
    ranks = [_ArenaRank(i) for i in range(3)]
    for r in ranks:
        r.add_chain(n_nodes)
    return ranks


@pytest.fixture(autouse=True)
def _group_d(monkeypatch):
    monkeypatch.setenv("SGLANG_WEG2_GROUP", "D")


def _load_back(r, depth=None):
    """The wake read landed and the load-back finished on every rank."""
    node = r.node_at(depth) if depth else r.nodes()[-1]
    r._ids += 1
    r.ongoing_load_back[r._ids] = (node, None, None)
    r.cache_controller.ack_load_queue.append((_Ev(), _Ev(), [r._ids]))
    r.loading_check()


def _worker_kv_rows(r):
    return sum(1 for n in r.nodes() if n.component_data[FULL].host_value is not None)


def test_metal_shape_a_load_back_gives_tp0s_arena_rows_and_the_workers_mirror_back():
    """The span on the device: TP0's KV host rows go (anchors stay), the STATE
    verdict takes the workers' byteless mirror rows at the next broadcast.
    RED on the base: every rank keeps (1, 1) until the reset."""
    ranks = _ranks()
    with h._switch(True):
        h._each(ranks, lambda r: r.ack_store_writes())
        h._broadcast(ranks)
        h._each(ranks, lambda r: _load_back(r, 3 * NODE))
        h._broadcast(ranks)
    assert h._same(ranks) == {NODE: (0, 1), 2 * NODE: (0, 1), 3 * NODE: (0, 1)}
    assert all(r.node_at(3 * NODE).component_data[FULL].value is not None for r in ranks)


def test_the_worker_pool_does_not_ratchet_over_park_and_resume_cycles():
    """#249: five park/resume cycles (the wake read re-registers the span on
    every rank, the load-back follows). The workers' mirror rows come back to 0
    after every cycle instead of piling up until the reset."""
    ranks = _ranks()
    seen = []
    with h._switch(True):
        for cycle in range(5):
            def reread(r, c=cycle):
                for n in r.nodes():  # the wake read: host rows (arena on TP0, byteless on a worker)
                    n.component_data[FULL].host_value = n.component_data[FULL].value.clone()
                r._leaf_sets()
            h._each(ranks, reread)
            h._each(ranks, lambda r: _load_back(r, 3 * NODE))
            h._broadcast(ranks)
            seen.append([_worker_kv_rows(r) for r in ranks])
    assert seen == [[0, 0, 0]] * 5, seen


def test_a_worker_never_releases_on_its_own():
    """The worker's own load-back ack changes nothing -- only TP0's verdict."""
    ranks = _ranks()
    with h._switch(True):
        h._each(ranks, lambda r: r.ack_store_writes())
        h._broadcast(ranks)
        with h._as_rank(ranks[1]):
            _load_back(ranks[1], 3 * NODE)
    assert _worker_kv_rows(ranks[1]) == 3


def test_a_node_under_a_host_lock_or_without_its_device_copy_is_kept():
    """Another load of the same span in flight (host lock), or a node whose
    device copy is gone (a host leaf): its rows stay."""
    ranks = _ranks()
    tp0 = ranks[0]
    with h._switch(True):
        with h._as_rank(tp0):
            tp0.node_at(NODE).component_data[FULL].host_lock_ref = 1
            tp0.node_at(2 * NODE).component_data[FULL].value = None
            n = park_l3.release_loaded_host(tp0, tp0.node_at(3 * NODE))
    assert n == 1
    assert tp0.node_at(NODE).component_data[FULL].host_value is not None
    assert tp0.node_at(2 * NODE).component_data[FULL].host_value is not None
    assert tp0.node_at(3 * NODE).component_data[FULL].host_value is None


def test_switch_off_and_group_p_are_the_old_path(monkeypatch):
    from sglang.srt.environ import envs

    ranks = _ranks()
    with h._switch(True), envs.SGLANG_WEG2_ENABLE_PARK_L3.override(False):
        h._each(ranks, lambda r: r.ack_store_writes())
        h._each(ranks, lambda r: _load_back(r, 3 * NODE))
        h._broadcast(ranks)
    assert h._same(ranks) == {NODE: (1, 1), 2 * NODE: (1, 1), 3 * NODE: (1, 1)}
    monkeypatch.setenv("SGLANG_WEG2_GROUP", "P")
    with h._as_rank(ranks[0]):
        assert park_l3.release_loaded_host(ranks[0], ranks[0].node_at(3 * NODE)) == 0
