"""L15 wake cost (N6k 3e0b2cd3a3: every held P>D was slower than unheld --
L15-WAKE-TIMING refill_ms 246-763 against a 64-66 ms refill DMA, proc_input
929-1703 ms vs 330-405 unheld; the peers' check_decide waits for TP0).

- L15-GENS-VEC: ArenaHostPool.slot_gens answers from one numpy scatter/gather
  over the census, not a per-call Python dict of every COMPLETE slot.
- L15-PLAN-CACHE: owned_l2_rows is cached per (manifest fingerprint, rank,
  prefix) and warmed at the sleep, so the wake's refill + sample reuse it.
"""
import inspect
import time
from types import SimpleNamespace

import numpy as np

from flliper.srt.mem_cache.pool_host import arena_pool
from flliper.srt.pdflip import l15_restore
from flliper.srt.pdflip.l15_manifest import HoldSpan, Manifest


def test_slot_gens_vectorised_same_answer():
    cs = np.array([0, 3, 7, 9], dtype=np.int64)
    cg = np.array([5, 6, 7, 8], dtype=np.int64)
    arena = SimpleNamespace(slots=12, complete_census=lambda: (cs, cg, None, None))
    pool = SimpleNamespace(arena=arena)
    fn = arena_pool.ArenaMHAHostPool.slot_gens if hasattr(arena_pool, "ArenaMHAHostPool") else None
    if fn is None:
        fn = next(v.slot_gens for v in vars(arena_pool).values()
                  if isinstance(v, type) and "slot_gens" in vars(v))
    assert fn(pool, [3, 1, 9, 50, -2, 0]) == [6, -1, 8, -1, -1, 5]
    assert fn(pool, []) == []
    assert "L15-GENS-VEC" in inspect.getsource(fn)


def _m(n):
    slots = tuple(range(1, n + 1))
    sp = HoldSpan(rid="r", depth=n, slots=slots, anchor_slot=5,
                  l2_slots=tuple(range(1000, 1000 + n)), l2_gens=tuple([1] * n))
    return Manifest(epoch=1, pid=1, spans=(sp,), rows_by_rank=(n, n, n), anchor_slots=2)


def test_plan_cache_returns_the_same_rows_and_is_warmable():
    prefix = [0, 62, 83, 108]
    m = _m(20000)
    l15_restore._PLAN_CACHE.clear()
    ref = l15_restore.owned_l2_rows_uncached(m, 0, prefix)
    l15_restore.warm_plan_async(m, 0, prefix)
    for _ in range(200):
        if l15_restore._PLAN_CACHE:
            break
        time.sleep(0.01)
    t = time.perf_counter()
    got = l15_restore.owned_l2_rows(m, 0, prefix)
    assert got == ref
    assert (time.perf_counter() - t) < 0.05
    # another rank / manifest misses and replaces the one entry
    assert l15_restore.owned_l2_rows(m, 1, prefix) == l15_restore.owned_l2_rows_uncached(m, 1, prefix)


def test_scheduler_warms_the_plan_at_the_sleep():
    from flliper.srt.managers import scheduler

    assert "_l15_rs4.warm_plan_async(" in inspect.getsource(scheduler)
