"""H107b (28.09.): H107 armed only when ``planner.resident_ids is None``. The
NF D pool runs on the store (Task #47), whose static layout spells the
identity ``[0, R)`` at ``slot == id`` out as maps
(``MoEExpertOffloadCache.__init__``: ``resident_ids = frozenset(range(R))``,
``resident_slot = {e: e}``) -- so H107 never armed on the metal: the bridge
rc12z29b (f833fcbb2d) ran 14 min under agent load 28.09. 19:38-19:58 with
0x 'H107 EAGER-LRU', and every post-wake extend still moved its spill experts
in scratch waves (1.3-4.3 s per pass).

What must hold:

* the store's identity maps arm H107 exactly like "no map": a warm LRU
  expert is read from its row, not fetched; the output is bit-identical;
* a layout that moves a resident off its own slot (hot set, load-time
  layout) keeps the plain plan -- ``_run_eager_lru`` computes wave 0 at
  ``slot == id``.
"""

import os
import sys

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from types import SimpleNamespace

from sglang.srt.layers.moe import expert_offload as eo
from sglang.test.ci.ci_register import register_cpu_ci

from test_pool_eager_lru_hits_h107 import R, ROUTES, SPILL, _extend, _pool_cache

register_cpu_ci(__file__)


def _store_identity(cache):
    """What __init__ does when the layer carries a store index (Task #47)."""
    cache.planner.resident_ids = frozenset(range(cache.resident_count))
    cache.planner.resident_slot = {e: e for e in range(cache.resident_count)}
    return cache


def test_store_identity_maps_arm_h107_warm_experts_are_not_fetched(monkeypatch):
    cache, fetched = _pool_cache(monkeypatch, warm=[3, 7])
    _store_identity(cache)
    got = _extend(cache)
    assert not {3, 7} & set(fetched), f"store layout: warm experts fetched again: {fetched}"
    assert set(fetched) == set(SPILL) - {3, 7}
    assert got[:, 0].tolist() == [float(sum(row)) for row in ROUTES]


def test_store_identity_output_is_bit_identical_to_the_plain_plan(monkeypatch):
    with_hits, _ = _pool_cache(monkeypatch, warm=[3, 7])
    _store_identity(with_hits)
    a = _extend(with_hits)
    plain, _ = _pool_cache(monkeypatch, warm=[3, 7])
    _store_identity(plain)
    plain._pool_eager_lru_hits = False
    b = _extend(plain)
    assert a.tolist() == b.tolist()


def test_store_identity_logs_the_metal_marker(monkeypatch, caplog):
    import logging

    monkeypatch.setattr(eo, "_H107_LOG_LAYER", None)
    cache, _ = _pool_cache(monkeypatch, warm=[3, 7])
    _store_identity(cache)
    with caplog.at_level(logging.INFO, logger=eo.__name__):
        _extend(cache)
    assert any("H107 EAGER-LRU" in r.getMessage() for r in caplog.records)


def test_a_moved_resident_keeps_the_plain_plan(monkeypatch):
    cache, fetched = _pool_cache(monkeypatch, warm=[3, 7])
    cache.planner.resident_ids = frozenset(range(R))
    cache.planner.resident_slot = {0: 1, 1: 0}  # a load-time layout that swapped slots
    assert not eo.eager_lru_static_residency(cache.planner, R)
    _extend(cache)
    assert {3, 7} <= set(fetched), "a non-identity layout must not run the H107 plan"


def test_eager_lru_static_residency():
    P = lambda ids, slot=None: SimpleNamespace(resident_ids=ids, resident_slot=slot)  # noqa: E731
    assert eo.eager_lru_static_residency(P(None), 4)
    assert eo.eager_lru_static_residency(P(frozenset(range(4)), {e: e for e in range(4)}), 4)
    assert eo.eager_lru_static_residency(P(frozenset(range(4))), 4)
    assert not eo.eager_lru_static_residency(P(frozenset({0, 1, 2, 7})), 4)  # hot set
    assert not eo.eager_lru_static_residency(P(frozenset(range(4)), {0: 0, 1: 2, 2: 1, 3: 3}), 4)
    assert not eo.eager_lru_static_residency(P(frozenset(range(3))), 4)
