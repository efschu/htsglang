"""KV-STAGE warm refill (01.10.): a seat-row shrink remembers the experts it
sent to the store, the grow that follows refills them at once.

y6h (boot 10011531, D.log D-MEM-SCHED lines): 179 stage-downs, 178 of them on
an END event, 109 down->up pairs, 72 of them within 5 s (median 2.0 s) -- the
agent's next turn. Every grow then left its freed rows empty and each departed
expert came back through one cold miss on the decode's critical path."""

import types

import pytest
import torch

from flliper.srt.layers.moe import expert_pool_device as epd


def _tables(keys, uses, *, lru_start, seat_base, seat_rows, seat_on, clock, E=16):
    hot = torch.full((E,), -1, dtype=torch.int32)
    for r, e in enumerate(keys):
        if 0 <= e < E:
            hot[e] = r
    return types.SimpleNamespace(
        num_experts=E, lru_start=lru_start, seat_base=seat_base, seat_rows=seat_rows,
        seat_on=seat_on, row_key=torch.tensor(keys, dtype=torch.int32),
        row_use=torch.tensor(uses, dtype=torch.int64), hot_phys=hot,
        pf_row=torch.full((len(keys),), -1, dtype=torch.int64),
        clock=torch.tensor([clock], dtype=torch.int64))


def _cache(monkeypatch, *, warm_switch):
    from flliper.srt.environ import envs
    from flliper.srt.layers.moe.expert_offload import MoEExpertOffloadCache

    monkeypatch.setattr(envs.FLLIPER_PDFLIP_D_SEAT_WARM_REFILL, "get", lambda: warm_switch)
    monkeypatch.setattr(envs.FLLIPER_PDFLIP_DISABLE_D_ELASTIC_ROWS, "get", lambda: True)
    cache = object.__new__(MoEExpertOffloadCache)
    cache._pool_tables = _tables([0, 1, 2, 3, 4, 5, 6, 7], [0, 0, 5, 6, 7, 8, 40, 50],
                                 lru_start=2, seat_base=6, seat_rows=2, seat_on=2, clock=60)
    cache._pool_ready, cache.seat_rows = True, 2
    cache._pdflip_seat_recall, cache._pdflip_seat_warmed = [], 0
    return cache


def test_a_shrink_names_every_expert_it_sent_to_the_store_hottest_first():
    """The coldest-first move keeps the hot seat-row experts and displaces cold
    kept-row ones instead; BOTH kinds leave the card. The list must hold exactly
    the experts that left (none that stayed), hottest first -- a grow refills
    from its head and a limit cuts its tail."""
    keys = [0, 1, 2, -1, 4, 5, 6, 7, 8]
    uses = [0, 0, 5, 0, 1, 9, 2, 50, 30]
    t = _tables(keys, uses, lru_start=2, seat_base=6, seat_rows=3, seat_on=3, clock=60)
    gone = []
    epd.set_seat_rows_on(t, 0, device_write=True, move_rows=lambda m: None, departed_out=gone)
    left = {e for e in (2, 4, 5, 6, 7, 8) if int(t.hot_phys[e]) < 0}
    assert set(gone) == left and len(gone) == len(left)
    assert int(t.hot_phys[7]) >= 0 and int(t.hot_phys[8]) >= 0   # the hot ones stayed
    use = dict(zip(keys, uses))
    assert [use[e] for e in gone] == sorted((use[e] for e in gone), reverse=True)


def test_the_grow_after_a_shrink_refills_the_departed_experts_at_once(monkeypatch):
    cache = _cache(monkeypatch, warm_switch=True)
    warmed = []
    cache.warm_lru_local = lambda ids, limit=0: warmed.append((list(ids), limit)) or min(
        len(ids), limit)
    cache.set_seat_rows_on(0, device_write=True)          # shrink: 7 and 6 leave
    assert not warmed
    cache.set_seat_rows_on(2, device_write=True)          # grow: back at once, hottest first
    assert warmed == [([7, 6], 2)]
    assert cache._pdflip_seat_warmed == 2 and cache._pdflip_seat_recall == []


def test_without_the_switch_the_grow_stays_the_lazy_fill(monkeypatch):
    cache = _cache(monkeypatch, warm_switch=False)
    cache.warm_lru_local = lambda ids, limit=0: pytest.fail("no warm refill without the switch")
    cache.set_seat_rows_on(0, device_write=True)
    cache.set_seat_rows_on(2, device_write=True)
    t = cache._pool_tables
    assert t.row_key[6:8].tolist() == [-1, -1]            # free rows, filled by the next miss
