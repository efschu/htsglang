"""S44 (18.10.): D's extends kept streaming every spill expert because H107 never
armed on the NF D pool, and when it does arm a big extend must not fall into
dozens of tiny waves.

Metal (rc12 boot 1018_153151, D TP0): 0x 'H107 EAGER-LRU' against 1587x
'MoE offload layer 0: expert-major prefill, 3 waves, 0.33-0.44 GiB H2D'; a
618-token mixed extend held the decode rows for ~2.2 s and its sync said
'eager pass wrote 101 rows' -- all scratch rows, the warm decode LRU wiped.
Root: the Platztausch Karte orders D's bank ``[prefix | pad | extra]``
(``_local_map_layout``), pad 0 sits behind the prefix, so ``resident_slot``
is not the identity and ``eager_lru_static_residency`` (H107b) said no.

What must hold:

* H107c: a MAPPED frozen layout (residents one-to-one on ``[0, R)``, slot !=
  id) arms H107 -- a warm LRU expert is read from its row, not fetched -- and
  wave 0 reads every resident at ITS slot (output bit-identical to the plain
  plan on the same layout); FLLIPER_OPT_MOE_POOL_EAGER_LRU_MAPPED=0 keeps the
  plain plan;
* H107d: when the misses over the rows holding no hit would need more waves
  than the plain plan + 1, the coldest hit rows are reclaimed: waves <= plain
  + 1, every miss fetched once, no warm hit fetched, output bit-identical, the
  hottest hit keeps its row, the reclaimed expert loses its row, the pool keeps
  one owner per expert; a reclaimed row is never overwritten before wave 0
  read it; FLLIPER_OPT_MOE_POOL_EAGER_LRU_RECLAIM=0 restores H107's plan.
"""

import os
import sys

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from collections import Counter

import torch

from flliper.srt.environ import envs
from flliper.srt.layers.moe import expert_offload as eo
from flliper.srt.layers.moe import expert_pool_device as ep
from flliper.test.ci.ci_register import register_cpu_ci

from test_pool_eager_lru_hits_h107 import C, E, R, ROUTES, S, SPILL, W, _extend, _pool_cache

register_cpu_ci(__file__)

# the Karte shape for R=2: prefix expert 1 in slot 0, the pad (expert 0) behind it
KARTE_SLOT = {1: 0, 0: 1}


def _mapped_cache(monkeypatch, warm=()):
    """``_pool_cache`` rebuilt in the Karte order: the bank's resident rows hold
    the residents at ``resident_slot`` and the pool tables are laid out from the
    same map (``_pool_layout``), exactly what a D rank installs."""
    cache, fetched = _pool_cache(monkeypatch)
    cache.planner.resident_ids = frozenset(KARTE_SLOT)
    cache.planner.resident_slot = dict(KARTE_SLOT)
    bank = cache._resident["w13"]
    spill = cache._pinned["w13"]
    bank.fill_(-1.0)
    for e, slot in KARTE_SLOT.items():
        bank[slot].fill_(e)
    hot_slot_of, host_row = cache._pool_layout()
    cache._pool_tables = ep.allocate_pool_tables("cpu", E, R + C, R, S, hot_slot_of, host_row)
    for host, dst in ep.seed_lru_rows(cache._pool_tables, list(warm)):
        bank[dst].copy_(spill[host])
    return cache, fetched


def _want(routes=ROUTES):
    return [float(sum(row)) for row in routes]


# --- H107c: the mapped layout --------------------------------------------------


def test_the_karte_layout_arms_h107_and_reads_residents_at_their_slots(monkeypatch):
    cache, fetched = _mapped_cache(monkeypatch, warm=[3, 7])
    assert not eo.eager_lru_static_residency(cache.planner, R)  # H107b said no
    got = _extend(cache)
    assert not {3, 7} & set(fetched), f"mapped layout: warm experts fetched again: {fetched}"
    assert set(fetched) == set(SPILL) - {3, 7}
    assert max(fetched.values()) == 1
    # a resident read at slot == id would compute expert 1 for 0 and vice versa
    assert got[:, 0].tolist() == _want()


def test_the_karte_layout_is_bit_identical_to_the_plain_plan(monkeypatch):
    on_cache, _ = _mapped_cache(monkeypatch, warm=[3, 7])
    on = _extend(on_cache)
    with envs.FLLIPER_OPT_MOE_POOL_EAGER_LRU_HITS.override(False):
        off_cache, _ = _mapped_cache(monkeypatch, warm=[3, 7])
    off = _extend(off_cache)
    assert torch.equal(on, off)


def test_the_mapped_switch_off_keeps_the_plain_plan(monkeypatch):
    cache, fetched = _mapped_cache(monkeypatch, warm=[3, 7])
    with envs.FLLIPER_OPT_MOE_POOL_EAGER_LRU_MAPPED.override(False):
        got = _extend(cache)
    assert {3, 7} <= set(fetched)
    assert got[:, 0].tolist() == _want()


def test_eager_lru_resident_slots():
    from types import SimpleNamespace

    P = lambda ids, slot=None: SimpleNamespace(resident_ids=ids, resident_slot=slot)  # noqa: E731
    assert eo.eager_lru_resident_slots(P(None), 4) == {}
    assert eo.eager_lru_resident_slots(P(frozenset(range(4)), {e: e for e in range(4)}), 4) == {}
    assert eo.eager_lru_resident_slots(P(frozenset(range(4))), 4) == {}
    # Karte: prefix 1..3 in slots 0..2, pad 0 behind it
    karte = {1: 0, 2: 1, 3: 2, 0: 3}
    assert eo.eager_lru_resident_slots(P(frozenset(karte), karte), 4) == karte
    assert eo.eager_lru_resident_slots(P(frozenset(karte), karte), 4, mapped=False) is None
    # residents outside [0, R) as ids (D TP1/TP2 extras) are fine as long as the slots are
    ext = {5: 0, 9: 1, 0: 2, 11: 3}
    assert eo.eager_lru_resident_slots(P(frozenset(ext), ext), 4) == ext
    # not one-to-one onto [0, R): no H107
    assert eo.eager_lru_resident_slots(P(frozenset(range(4)), {0: 0, 1: 0, 2: 1, 3: 2}), 4) is None
    assert eo.eager_lru_resident_slots(P(frozenset(range(4)), {0: 0, 1: 1, 2: 2, 3: 7}), 4) is None
    assert eo.eager_lru_resident_slots(P(frozenset(range(4)), {0: 0, 1: 1}), 4) is None
    assert eo.eager_lru_resident_slots(P(frozenset({0, 1, 2, 7})), 4) is None  # hot set, no map
    assert eo.eager_lru_resident_slots(P(frozenset(range(3))), 4) is None


# --- H107d: reclaim ------------------------------------------------------------

# warm 3, 7, 9 fill all three LRU rows; only the staging row is writable
WARM = [3, 7, 9]


def _reclaim_cache(monkeypatch, mapped=False):
    build = _mapped_cache if mapped else _pool_cache
    cache, fetched = build(monkeypatch, warm=WARM)
    t = cache._pool_tables
    hot = t.hot_phys.tolist()
    # recency: 7 hottest, then 3, 9 coldest
    for e, use in ((7, 900), (3, 500), (9, 100)):
        t.row_use[hot[e]] = use
    calls = []
    counting = cache._fetch

    def _waves(plan, join=True):
        if plan:
            calls.append(list(plan))
        return counting(plan, join)

    cache._fetch = _waves
    return cache, fetched, calls


def test_reclaim_bounds_the_waves_and_fetches_only_misses(monkeypatch):
    cache, fetched, calls = _reclaim_cache(monkeypatch)
    t = cache._pool_tables
    row_of = {e: int(t.hot_phys[e]) for e in WARM}
    got = _extend(cache)
    misses = set(SPILL) - set(WARM)
    assert set(fetched) == misses and max(fetched.values()) == 1, fetched
    # plain plan: 8 spill experts / 4 scratch rows = 2 waves; bound = 2 + 1
    assert len(calls) <= 3, f"{len(calls)} miss waves: {calls}"
    assert got[:, 0].tolist() == _want()
    # the coldest hit row (9's) was reclaimed, the hotter ones were not written
    written = {row for plan in calls for _e, row in plan}
    assert row_of[9] in written
    assert row_of[7] not in written and row_of[3] not in written
    # after the sync: one owner per expert, 7 and 3 keep their rows, 9 lost its
    assert ep.bijection_breaks(t) == 0
    assert int(t.hot_phys[7]) == row_of[7] and int(t.hot_phys[3]) == row_of[3]
    assert int(t.hot_phys[9]) != row_of[9]
    lo, hi = t.lru_start, t.pool_rows
    for e, r in enumerate(t.hot_phys.tolist()):
        if lo <= r < hi:
            assert float(cache._resident["w13"][r, 0]) == float(e), (e, r)


def test_reclaim_is_bit_identical_to_the_plain_plan(monkeypatch):
    cache, _, _ = _reclaim_cache(monkeypatch)
    on = _extend(cache)
    with envs.FLLIPER_OPT_MOE_POOL_EAGER_LRU_HITS.override(False):
        plain, _, _ = _reclaim_cache(monkeypatch)
    off = _extend(plain)
    assert torch.equal(on, off)


def test_reclaim_on_the_karte_layout(monkeypatch):
    cache, fetched, calls = _reclaim_cache(monkeypatch, mapped=True)
    got = _extend(cache)
    assert not set(WARM) & set(fetched)
    assert len(calls) <= 3
    assert got[:, 0].tolist() == _want()


def test_the_reclaim_switch_off_keeps_h107s_plan(monkeypatch):
    cache, fetched, calls = _reclaim_cache(monkeypatch)
    with envs.FLLIPER_OPT_MOE_POOL_EAGER_LRU_RECLAIM.override(False):
        got = _extend(cache)
    # one writable row (staging) for 5 misses: H107's plan runs 5 waves
    assert len(calls) == 5, calls
    assert not set(WARM) & set(fetched)
    assert got[:, 0].tolist() == _want()


def test_a_reclaimed_row_is_read_before_it_is_overwritten(monkeypatch):
    """Danger direction: a plan with ONE miss wave that writes a reclaimed hit
    row must still run wave 0 on its own -- fused into one apply the hit's
    bytes would be gone before the apply read them."""
    cache, fetched, _ = _reclaim_cache(monkeypatch)
    t = cache._pool_tables
    row9 = int(t.hot_phys[9])
    real = eo.plan_eager_lru_waves_reclaim

    def _one_wave(spill, lru, rows, **kw):
        hits, waves, _rec = real(spill, lru, rows, **kw)
        misses = [e for w in waves for e, _r in w]
        assert misses == [2] and hits.get(9) == row9
        return hits, [[(2, row9)]], [row9]  # the miss lands in 9's reclaimed row

    routes = [[2, 3, 0], [9, 7, 1], [3, 7, 9]]  # misses {2}; 9 is a hit in a reclaimed row
    monkeypatch.setattr(eo, "plan_eager_lru_waves_reclaim", _one_wave)
    got = _extend(cache, routes)
    assert got[:, 0].tolist() == _want(routes)


def test_plan_eager_lru_waves_reclaim():
    plan = eo.plan_eager_lru_waves_reclaim
    # no bound, no use clocks: H107's plan exactly
    hits, waves, rec = plan([2, 3, 4, 5, 6], {3: 4, 9: 5}, range(2, 6))
    assert (hits, waves, rec) == ({3: 4}, [[(2, 2), (4, 3), (5, 5)], [(6, 2)]], [])
    assert plan([2, 3], {3: 2}, [2]) is None
    assert plan([3], {3: 2}, [2]) == ({3: 2}, [], [])
    # use clocks: free rows first, then owned rows coldest first
    use = [0, 0, 50, 70, 10, 0]
    hits, waves, rec = plan([2, 4], {7: 2, 8: 3, 9: 4}, range(2, 6), row_use=use)
    assert hits == {} and rec == []
    assert waves == [[(2, 5), (4, 4)]]  # row 5 owns nothing, row 4 is the coldest owner
    # bound: 5 misses over 1 writable row, max 3 waves -> reclaim 1 hit row (coldest)
    use = [0, 0, 500, 900, 100, 0]
    hits, waves, rec = plan(
        [2, 3, 4, 5, 6, 7, 8, 9], {3: 2, 7: 3, 9: 4}, range(2, 6), row_use=use, max_waves=3
    )
    assert hits == {3: 2, 7: 3, 9: 4}
    assert rec == [4]
    assert len(waves) == 3 and waves[0] == [(2, 5), (4, 4)]
    assert sorted(e for w in waves for e, _r in w) == [2, 4, 5, 6, 8]
    # every scratch row holds a hit: the bound reclaims instead of giving up
    hits, waves, rec = plan([2, 3], {3: 2}, [2], row_use=[0, 0, 5], max_waves=2)
    assert rec == [2] and waves == [[(2, 2)]]
    # a hit row outside the scratch rows is never reclaimed
    assert plan([2, 3], {3: 9}, [], row_use=[0] * 10, max_waves=2) is None


def test_reclaim_on_the_vector_planned_route(monkeypatch):
    """The numpy route (FLLIPER_MOE_OFFLOAD_PLAN_VECTOR, >= PLAN_VECTOR_MIN_PAIRS)
    carries the use clocks in its one D2H as well."""
    cache, fetched, calls = _reclaim_cache(monkeypatch, mapped=True)
    cache._plan_vector = True
    monkeypatch.setattr(eo, "PLAN_VECTOR_MIN_PAIRS", 1)
    seen = []
    real = cache._run_eager_lru
    real_vec = cache._run_waves_vector
    vec = []

    def _spy(*a, **kw):
        seen.append(kw.get("pool_use") is not None)
        return real(*a, **kw)

    def _spy_vec(*a, **kw):
        vec.append(1)
        return real_vec(*a, **kw)

    cache._run_eager_lru = _spy
    cache._run_waves_vector = _spy_vec
    got = _extend(cache)
    assert vec == [1], "the forward did not take the vector-planned route"
    assert seen == [True]
    assert not set(WARM) & set(fetched)
    assert len(calls) <= 3
    assert got[:, 0].tolist() == _want()
