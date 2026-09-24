"""H12 (fnFL2x104, 90k needle): D's extend after the flip is an eager forward
under the device-planned expert pool, and it ran TOKEN-major -- the arm sets
SGLANG_MOE_OFFLOAD_WAVE_ORDER=expert only for P. On D TP0 (12 residents, 181
spill rows, scratch 44) the 49-token extend took 3 waves per layer and every
wave re-fetched the hot spill experts its tokens shared with the previous
wave: 0.27 GiB H2D per layer, gpu-ms 1069 for 64 tokens -- the stream WAS the
extend. (The same split left twins in the pool rows, Blocker #104.)

What must hold, black-box through the FusedMoE pool branch
(``run_eager_pool``): an eager forward that overflows the scratch region
fetches every spill expert EXACTLY ONCE and computes every (token, k) pair;
SGLANG_OPT_MOE_POOL_EAGER_EXPERT_MAJOR=0 gives the old token-major split back.
Hermetic: a CPU cache, no CUDA.
"""

import os

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

from collections import Counter, namedtuple
from types import SimpleNamespace

import torch

from sglang.srt.environ import envs
from sglang.srt.layers.moe import expert_offload as eo
from sglang.srt.layers.moe import expert_pool_device as ep
from sglang.srt.layers.moe.topk import StandardTopKOutput
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(__file__)

# residents 0,1 in rows 0,1; scratch C=4 -> rows 2..5 (3 LRU + 1 staging)
E, R, C, S, W = 12, 2, 4, 1, 3

DispatchOutput = namedtuple("DispatchOutput", "hidden_states hidden_states_scale topk_output")
CombineOutput = namedtuple("CombineOutput", "hidden_states")

# Two tokens per token-major wave: tokens 0/1 need spill {2,3,4,5}, tokens 2/3
# need {2,3,6,7} -- 2 and 3 are the hot experts both waves share, token 4 needs
# {8,9} and 2. 8 distinct spill experts over 3 token-major waves.
ROUTES = [
    [2, 3, 0], [4, 5, 1],
    [2, 6, 0], [3, 7, 1],
    [8, 9, 2],
]


def _pool_cache(monkeypatch):
    monkeypatch.setenv("SGLANG_MOE_SCRATCH_SLOTS", str(C))
    monkeypatch.setenv("SGLANG_MOE_OFFLOAD_WAVE_ORDER", "token")  # D's launcher env
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
    cache._pool_buffers = ep.allocate_step_buffers("cpu", E, 8)
    cache._pool_ready = True
    fetched = Counter()
    real_fetch = cache._fetch

    def _counting_fetch(plan, join=True):
        fetched.update(e for e, _slot in plan)
        return real_fetch(plan, join)

    cache._fetch = _counting_fetch
    return cache, fetched


def _extend(cache):
    """The FusedMoE pool branch's eager forward. The apply reads, per routed
    pair, the bank row the remapped id points at -- so a pair routed to a row
    that does not hold its expert computes the wrong expert id."""
    ids = torch.tensor(ROUTES, dtype=torch.int32)
    bank = cache._resident["w13"]
    topk = StandardTopKOutput(
        topk_weights=torch.ones(ids.shape), topk_ids=ids, router_logits=None
    )

    def _apply(sub):
        rows = sub.topk_output.topk_ids.long()
        per_pair = bank[rows.clamp(min=0)][..., 0] * sub.topk_output.topk_weights
        out = per_pair.sum(dim=-1, keepdim=True).expand(-1, W).contiguous()
        return CombineOutput(hidden_states=out)

    out = cache.run_eager_pool(DispatchOutput(torch.zeros(ids.shape[0], W), None, topk), _apply)
    return out.hidden_states[:, 0].tolist()


def test_the_pool_extend_fetches_each_spill_expert_once_and_computes_every_pair(monkeypatch):
    cache, fetched = _pool_cache(monkeypatch)
    got = _extend(cache)
    spill_routed = {e for row in ROUTES for e in row if e >= R}
    assert set(fetched) == spill_routed
    assert max(fetched.values()) == 1, f"re-fetched in a later wave: {fetched}"
    # every token's output is the sum of its routed expert ids
    assert got == [float(sum(row)) for row in ROUTES]


def test_the_switch_off_restores_the_token_major_refetch(monkeypatch):
    with envs.SGLANG_OPT_MOE_POOL_EAGER_EXPERT_MAJOR.override(False):
        cache, fetched = _pool_cache(monkeypatch)
    got = _extend(cache)
    assert fetched[2] >= 2, f"token-major should re-fetch the shared hot expert: {fetched}"
    assert got == [float(sum(row)) for row in ROUTES]
