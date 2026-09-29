"""H107 (D extend, 28.09.): the eager expert-major forward under the device-
planned pool reads a routed spill expert from the LRU row that already OWNS it
instead of fetching it again, and fetches only the misses into scratch rows
that hold no hit.

rc12z26 D TP0: 12 residents + 98 LRU rows = 110 of 193 experts on the card,
yet every real extend moved 0.20 GiB per layer in 3 waves (~1.6 s gpu-ms) --
the eager plan (``begin_eager_pool`` clears the holds, ``resolve`` places every
spill expert into ``[R, R + scratch)``) never looked at the pool's LRU, and it
overwrote the decode working set on the way.

What must hold, black-box through the FusedMoE pool branch
(``run_eager_pool``) on a CPU cache:

* an expert the pool's LRU already owns is NOT fetched; every miss once;
* the output is bit-identical with and without the switch;
* after the sync the pool keeps one owner per expert (#104: no twin rows),
  the hit rows keep their expert and are stamped as used, and the next
  decode step routes every lane to a row holding that expert's bytes;
* the pool map crosses in the SAME D2H as the routing (no extra sync);
* FLLIPER_OPT_MOE_POOL_EAGER_LRU_HITS=0 restores the plain plan.
"""

import os

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

from collections import Counter, namedtuple
from types import SimpleNamespace

import torch

from flliper.srt.environ import envs
from flliper.srt.layers.moe import expert_offload as eo
from flliper.srt.layers.moe import expert_pool_device as ep
from flliper.srt.layers.moe.topk import StandardTopKOutput
from flliper.test.ci.ci_register import register_cpu_ci

register_cpu_ci(__file__)

# residents 0,1 in rows 0,1; scratch C=4 -> rows 2..5 (3 LRU + 1 staging)
E, R, C, S, W = 12, 2, 4, 1, 3

DispatchOutput = namedtuple("DispatchOutput", "hidden_states hidden_states_scale topk_output")
CombineOutput = namedtuple("CombineOutput", "hidden_states")

# 8 distinct spill experts {2..9}: two scratch waves of 4 in the plain plan
ROUTES = [
    [2, 3, 0], [4, 5, 1],
    [2, 6, 0], [3, 7, 1],
    [8, 9, 2],
]
SPILL = sorted({e for row in ROUTES for e in row if e >= R})


def _pool_cache(monkeypatch, warm=()):
    """A CPU pool cache whose bank rows carry their expert id as bytes; the
    experts in ``warm`` are seeded into LRU rows as a decode round would
    have left them."""
    monkeypatch.setenv("FLLIPER_MOE_SCRATCH_SLOTS", str(C))
    monkeypatch.setitem(eo._PARTIALS_MODE, "mode", "stream")  # no combine kernel on CPU
    layer = SimpleNamespace(
        num_local_experts=E, layer_id=23,
        moe_runner_config=SimpleNamespace(routed_scaling_factor=1.0),
    )
    cache = eo.MoEExpertOffloadCache(layer, R / E)
    assert (cache.resident_count, cache.scratch) == (R, C)
    spill = torch.zeros((E - R, W), dtype=torch.float32)
    for row in range(E - R):
        spill[row].fill_(R + row)  # a row's bytes ARE its expert id
    bank = torch.full((R + C, W), -1.0, dtype=torch.float32)
    for e in range(R):
        bank[e].fill_(e)
    cache._pinned = {"w13": spill}
    cache._resident = {"w13": bank}
    cache._installed = True
    hot_slot_of, host_row = cache._pool_layout()
    cache._pool_tables = ep.allocate_pool_tables("cpu", E, R + C, R, S, hot_slot_of, host_row)
    cache._pool_buffers = ep.allocate_step_buffers("cpu", E, 16)
    cache._pool_srcs = [spill]
    cache._pool_dsts = [bank]
    cache._pool_ready = True
    for host, dst in ep.seed_lru_rows(cache._pool_tables, list(warm)):
        bank[dst].copy_(spill[host])
    fetched = Counter()
    real_fetch = cache._fetch

    def _counting_fetch(plan, join=True):
        fetched.update(e for e, _slot in plan)
        return real_fetch(plan, join)

    cache._fetch = _counting_fetch
    return cache, fetched


def _apply_over(bank):
    def _apply(sub):
        rows = sub.topk_output.topk_ids.long()
        per_pair = bank[rows.clamp(min=0)][..., 0] * sub.topk_output.topk_weights
        out = per_pair.sum(dim=-1, keepdim=True).expand(-1, W).contiguous()
        return CombineOutput(hidden_states=out)

    return _apply


def _extend(cache, routes=ROUTES):
    """The FusedMoE pool branch's eager forward. A pair routed to a row that
    does not hold its expert computes the wrong expert id."""
    ids = torch.tensor(routes, dtype=torch.int32)
    topk = StandardTopKOutput(
        topk_weights=torch.ones(ids.shape), topk_ids=ids, router_logits=None
    )
    out = cache.run_eager_pool(
        DispatchOutput(torch.zeros(ids.shape[0], W), None, topk),
        _apply_over(cache._resident["w13"]),
    )
    return out.hidden_states


def _owned(tables):
    """expert -> LRU row, and the rows naming an expert, after the sync."""
    hot = tables.hot_phys.tolist()
    key = tables.row_key.tolist()
    lo, hi = tables.lru_start, tables.pool_rows
    rows_naming = Counter(k for k in key[lo:hi] if k >= 0)
    return {e: r for e, r in enumerate(hot) if lo <= r < hi}, rows_naming


def test_a_warm_lru_expert_is_read_from_its_row_not_fetched(monkeypatch):
    cache, fetched = _pool_cache(monkeypatch, warm=[3, 7])
    got = _extend(cache)
    assert not {3, 7} & set(fetched), f"the warm experts were fetched again: {fetched}"
    assert set(fetched) == set(SPILL) - {3, 7}
    assert max(fetched.values()) == 1, f"re-fetched in a later wave: {fetched}"
    assert got[:, 0].tolist() == [float(sum(row)) for row in ROUTES]


def test_output_is_bit_identical_with_and_without_the_switch(monkeypatch):
    routes = [[2, 3, 0], [4, 5, 1], [2, 6, 0], [3, 7, 1], [8, 9, 2], [7, 3, 9]]
    with_switch, _ = _pool_cache(monkeypatch, warm=[3, 7, 9])
    on = _extend(with_switch, routes)
    with envs.FLLIPER_OPT_MOE_POOL_EAGER_LRU_HITS.override(False):
        without, _ = _pool_cache(monkeypatch, warm=[3, 7, 9])
    off = _extend(without, routes)
    assert torch.equal(on, off)


def test_the_sync_keeps_one_owner_per_expert_and_stamps_the_hit_rows(monkeypatch):
    cache, _ = _pool_cache(monkeypatch, warm=[3, 7])
    t = cache._pool_tables
    row_of_3 = int(t.hot_phys[3])
    _extend(cache)
    owned, rows_naming = _owned(t)
    assert ep.bijection_breaks(t) == 0
    assert all(n == 1 for n in rows_naming.values()), f"twin rows: {rows_naming}"
    assert owned.get(3) == row_of_3, "a hit row must keep its expert"
    assert int(t.row_use[row_of_3]) == int(t.clock[0]), "a hit row is stamped as used"
    for e, r in owned.items():  # every owned row holds its expert's bytes
        assert float(cache._resident["w13"][r, 0]) == float(e)


def test_the_next_decode_step_routes_every_lane_to_its_bytes(monkeypatch):
    cache, _ = _pool_cache(monkeypatch, warm=[3, 7])
    _extend(cache)
    # a verify step within the pool's bound (<= LRU + staging distinct ids):
    # 3 and 7 are the extend's hits, 9 was fetched by it, 0 is resident
    ids = torch.tensor([[3, 9], [7, 0]], dtype=torch.int32)
    routes = cache.prepare_pool(ids)
    bank = cache._resident["w13"]
    for (e_row, r_row) in zip(ids.tolist(), routes.tolist()):
        for e, r in zip(e_row, r_row):
            assert float(bank[r, 0]) == float(e), f"expert {e} routed to row {r}"
    assert ep.bijection_breaks(cache._pool_tables) == 0


def test_the_pool_map_crosses_with_the_routing_in_one_host_read(monkeypatch):
    cache, _ = _pool_cache(monkeypatch, warm=[3, 7])
    reads = Counter()
    real_tolist = torch.Tensor.tolist

    def _counting_tolist(self):
        reads["tolist"] += 1
        return real_tolist(self)

    monkeypatch.setattr(torch.Tensor, "tolist", _counting_tolist)
    ids = torch.tensor(ROUTES, dtype=torch.int32)
    topk = StandardTopKOutput(topk_weights=torch.ones(ids.shape), topk_ids=ids, router_logits=None)
    cache.begin_eager_pool()
    cache._eager_lru_armed = True
    cache.run_waves(
        DispatchOutput(torch.zeros(ids.shape[0], W), None, topk),
        _apply_over(cache._resident["w13"]),
        lookahead=None,
        order="expert",
    )
    # one rendezvous: the ids and the pool map in ONE tolist, none of its own
    assert reads["tolist"] == 1, reads


def test_the_switch_off_fetches_the_warm_experts_again(monkeypatch):
    with envs.FLLIPER_OPT_MOE_POOL_EAGER_LRU_HITS.override(False):
        cache, fetched = _pool_cache(monkeypatch, warm=[3, 7])
    got = _extend(cache)
    assert {3, 7} <= set(fetched)
    assert got[:, 0].tolist() == [float(sum(row)) for row in ROUTES]


def test_plan_eager_lru_waves():
    # hits read from their rows, misses chunked over the rows holding no hit
    hits, waves = eo.plan_eager_lru_waves([2, 3, 4, 5, 6], {3: 4, 9: 5}, range(2, 6))
    assert hits == {3: 4}
    assert waves == [[(2, 2), (4, 3), (5, 5)], [(6, 2)]]
    # every scratch row holds a hit and misses remain: the plain plan runs
    assert eo.plan_eager_lru_waves([2, 3], {3: 2}, [2]) is None
    # all hits: no fetch at all
    assert eo.plan_eager_lru_waves([3], {3: 2}, [2]) == ({3: 2}, [])


def test_pool_lru_rows_lists_only_owned_lru_rows():
    # residents (< lru_start), unowned (-1) and rows past the pool are skipped
    assert eo.pool_lru_rows([0, 1, -1, 4, 9, 5], lru_start=2, row_limit=6) == {3: 4, 5: 5}
