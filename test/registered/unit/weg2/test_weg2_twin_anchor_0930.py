"""TWIN ANCHOR (NF y4a, 30.09.; P log ...0930_031042).

Metal: two ``#TW TWIN-NO-GAIN``.
* weg2-16-28 (62856) shared 62439 with weg2-16-27 (62779, start 59008). 16-27's
  one step [59008,62779) tracked the turn anchor 62592 and the end anchor
  62720, both past 62439: '#TW TWIN-DEFER' held 16-28 3 s, then NO-GAIN, and it
  re-prefilled 3847 tokens from 59008.
* weg2-12-19 (87844) shared 87678 with weg2-12-21, which started at 87680 --
  past every boundary; 12-19 waited 3 s for an anchor no source could write.

Pinned (hermetic, CPU; RED on 1eac8b3461):
  1. a source whose step holds the twin boundary ``B = floor_page(shared-1)``
     of a QUEUED twin plans one more extend track there, and on the REAL
     UnifiedRadixCache the twin then resumes AT ``B`` with that state;
  2. no promise, no wait: a twin whose source cannot write an anchor <= shared
     registers at once; a provisional promise (queued source) is dropped at
     the source's first plan that passes ``B`` without the track;
  3. several twins of one source: one track per boundary, each twin resumes
     at its own.
"""

from array import array
from types import SimpleNamespace

import pytest
import torch

from sglang.srt.configs.mamba_utils import Mamba2CacheParams, Mamba2StateShape
from sglang.srt.environ import envs
from sglang.srt.layers.attention.fla.chunk_delta_h import CHUNK_SIZE as FLA_CHUNK_SIZE
from sglang.srt.managers.schedule_batch import Req
from sglang.srt.mem_cache.allocator import TokenToKVPoolAllocator
from sglang.srt.mem_cache.base_prefix_cache import MatchPrefixParams
from sglang.srt.mem_cache.cache_init_params import CacheInitParams
from sglang.srt.mem_cache.memory_pool import HybridLinearKVPool, HybridReqToTokenPool
from sglang.srt.mem_cache.radix_cache import RadixKey
from sglang.srt.mem_cache.unified_cache_components.tree_component import ComponentType
from sglang.srt.mem_cache.unified_radix_cache import UnifiedRadixCache
from sglang.srt.sampling.sampling_params import SamplingParams
from sglang.srt.server_args import ServerArgs, set_global_server_args_for_scheduler
from sglang.srt.weg2 import p_twin_defer as tw
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=20, suite="stage-a-test-cpu")

TWIN_PENDING = "_weg2_twin_pending"   # turn_anchor.TWIN_PENDING_ATTR, by name
MIN = 32                              # SGLANG_WEG2_P_TWIN_MIN_TOKENS in these tests
N = 120                               # the source prompt
SHARED_A, SHARED_B = 70, 95           # two twins' shared leading ids
S_END, S_A, S_B = 7.5, 2.25, 4.75


def _ids(shared, total, salt):
    return list(range(1000, 1000 + shared)) + [
        900_000 + salt * 10_000 + i for i in range(total - shared)]


SRC = list(range(1000, 1000 + N))
TWIN_A = _ids(SHARED_A, 110, 1)
TWIN_B = _ids(SHARED_B, 130, 2)


@pytest.fixture(autouse=True)
def armed(monkeypatch):
    from sglang.srt.weg2 import twin_anchor as ta2

    monkeypatch.setenv(tw.ENV, "1")
    monkeypatch.setenv(tw.ENV_MIN_TOKENS, str(MIN))
    monkeypatch.setenv("SGLANG_WEG2_GROUP", "P")
    ta2._reset_for_test()
    yield
    ta2._reset_for_test()


# -- 1a/3a. the source plans a track per queued twin's boundary ------------------------
class _Alloc:
    def __init__(self, free):
        self.free_ids = list(free)
        self.freed = []

    def alloc(self, n):
        if len(self.free_ids) < n:
            return None
        out, self.free_ids = self.free_ids[:n], self.free_ids[n:]
        return torch.tensor(out, dtype=torch.int64)

    def free(self, t):
        self.freed.extend(int(x) for x in t.view(-1).tolist())


def _q(rid, ids):
    return SimpleNamespace(rid=rid, origin_input_ids=list(ids), extra_key=None,
                           output_ids=[], cache_protected_len=0)


@pytest.mark.parametrize("queued,want", [
    ([("weg2-16-28", TWIN_A)], [SHARED_A - 1]),
    ([("weg2-16-28", TWIN_A), ("weg2-16-29", TWIN_B)], [SHARED_A - 1, SHARED_B - 1]),
])
def test_the_source_step_tracks_every_queued_twins_boundary(queued, want):
    from sglang.srt.weg2 import turn_anchor as ta
    from sglang.srt.weg2 import twin_anchor as ta2

    src = _q("weg2-16-27", SRC)
    q = [_q(r, ids) for r, ids in queued] + [_q("short", list(range(10)))]
    bounds = ta2.batch_bounds([src], q, page=1)
    assert bounds == {"weg2-16-27": want}
    alloc = _Alloc([11, 12, 13])
    batch = SimpleNamespace(reqs=[src], req_to_token_pool=SimpleNamespace(mamba_allocator=alloc))
    desc = ta.note_step(batch=batch, desc=None, req=src, row=0, prefix=0, end=N,
                        track_mask=True, main_track=N - 1, chunk=1, page=1, tok=248045,
                        twin_bounds=bounds["weg2-16-27"])
    plans = getattr(src, TWIN_PENDING)
    assert [p[0] for p in plans] == want and sorted(desc.targets) == want
    assert all(ta2.status("weg2-16-27", b) == "planned" for b in want)
    # a step that starts past the boundary (12-19's source: start 87680 > 87616)
    late = _q("weg2-12-21", SRC)
    batch2 = SimpleNamespace(reqs=[late], req_to_token_pool=SimpleNamespace(mamba_allocator=_Alloc([21])))
    assert ta.note_step(batch=batch2, desc=None, req=late, row=0, prefix=want[-1] + 1, end=N,
                        track_mask=True, main_track=N - 1, chunk=1, page=1, tok=248045,
                        twin_bounds=want) is None


# -- 1b/3b. the real tree: each twin resumes at its boundary ------------------------
MAMBA_SLOTS = 20
KV_SIZE = 512
NUM_LAYERS = 8
FULL_LAYER_IDS = (3, 7)
MAMBA_LAYER_IDS = [i for i in range(NUM_LAYERS) if i not in FULL_LAYER_IDS]


@pytest.fixture
def group_p(monkeypatch):
    from sglang.srt.mem_cache import unified_radix_cache as urc

    monkeypatch.setenv("SGLANG_WEG2_BIGRAM_ANCHOR_EXACT", "1")
    monkeypatch.setattr(urc, "_WEG2_END_ANCHOR", True)
    return monkeypatch


def _fixture():
    server_args = ServerArgs(model_path="dummy", page_size=1)
    server_args._mamba_cache_chunk_size = FLA_CHUNK_SIZE
    server_args.chunked_prefill_size = 16
    set_global_server_args_for_scheduler(server_args)
    with envs.SGLANG_MAMBA_SSM_DTYPE.override("bfloat16"):
        shape = Mamba2StateShape.create(
            tp_world_size=1, intermediate_size=256, n_groups=1, num_heads=2,
            head_dim=16, state_size=16, conv_kernel=4,
        )
        cache_params = Mamba2CacheParams(shape=shape, layers=MAMBA_LAYER_IDS)
    pool = HybridReqToTokenPool(
        size=10, mamba_size=MAMBA_SLOTS, mamba_spec_state_size=10, max_context_len=256,
        device="cpu", enable_memory_saver=False, cache_params=cache_params,
        mamba_layer_ids=MAMBA_LAYER_IDS, enable_mamba_extra_buffer=False,
        speculative_num_draft_tokens=3,
    )
    kv_pool = HybridLinearKVPool(
        size=KV_SIZE, dtype=torch.bfloat16, page_size=1, head_num=2, head_dim=64,
        full_attention_layer_ids=list(FULL_LAYER_IDS), device="cpu",
        enable_memory_saver=False, mamba_pool=pool.mamba_pool,
    )
    allocator = TokenToKVPoolAllocator(
        size=KV_SIZE, dtype=torch.bfloat16, device="cpu", kvcache=kv_pool, need_sort=False,
    )
    params = CacheInitParams(
        req_to_token_pool=pool, token_to_kv_pool_allocator=allocator, page_size=1,
        disable=False, sliding_window_size=None,
        tree_components=(ComponentType.FULL, ComponentType.MAMBA),
        enable_mamba_extra_buffer=False, enable_kv_cache_events=False,
        eviction_policy="lru", is_eagle=False,
    )
    cache = UnifiedRadixCache(params=params)
    cache.cache_init_params = params
    return SimpleNamespace(cache=cache, allocator=allocator, pool=pool)


def _key(ids):
    return RadixKey(array("q", ids), is_bigram=False)


def _state(fx, slot, value):
    fx.pool.mamba_pool.mamba_cache.temporal[:, slot] = value


def _source_prefill(fx, bounds_states):
    """16-27's shape: the whole prompt in ONE step; the forward left each
    twin-boundary track's state in its slot (the plan's marks set)."""
    from sglang.srt.weg2 import turn_anchor as ta

    req = Req(rid="weg2-16-27", origin_input_text="", origin_input_ids=array("q", SRC),
              sampling_params=SamplingParams(temperature=0, max_new_tokens=1))
    fx.pool.alloc([req])
    req.output_ids = array("q")
    req.full_untruncated_fill_ids = array("q", SRC)
    req.swa_uuid_for_lock = None
    req.extra_key = None
    mr = fx.cache.match_prefix(MatchPrefixParams(key=_key(SRC)))
    req.prefix_indices = mr.device_indices.to(torch.int64)
    req.cache_protected_len = len(req.prefix_indices)
    req.last_node = mr.last_device_node
    fx.cache.inc_lock_ref(req.last_node)
    fx.pool.write((req.req_pool_idx, slice(0, N)), fx.allocator.alloc(N))
    req.set_extend_range(0, N)
    desc = ta.TurnTracks(1)
    plans = []
    for b, value in bounds_states:
        slot = fx.pool.mamba_allocator.alloc(1)
        _state(fx, slot, value)
        desc.add(0, slot, b, 0, N)
        plans.append((b, slot, desc, 0, N))
    desc.done.add("gdn")
    setattr(req, TWIN_PENDING, plans)
    _state(fx, req.mamba_pool_idx, S_END)
    req.kv_committed_len = N
    req.kv_allocated_len = N
    fx.cache.cache_finished_req(req, is_insert=True)


def _claim(fx, ids):
    claim = Req._compute_max_prefix_len(
        SimpleNamespace(return_logprob=False, logprob_start_len=0), len(ids))
    mr = fx.cache.match_prefix(MatchPrefixParams(key=_key(ids[:claim])))
    slot = mr.last_device_node.component_data[ComponentType.MAMBA].value
    state = None if slot is None else float(
        fx.pool.mamba_pool.mamba_cache.temporal[:, slot].float().mean())
    return len(mr.device_indices), state


def test_the_twin_resumes_at_the_boundary_the_source_tracked(group_p):
    """16-28: the source's own anchors lie past shared; with the boundary track
    the twin resumes at floor_page(shared - 1) with that state (red on the
    base: the plan is never inserted, the twin resumes from nothing)."""
    fx = _fixture()
    _source_prefill(fx, [(SHARED_A - 1, S_A)])
    assert _claim(fx, TWIN_A) == (SHARED_A - 1, pytest.approx(S_A))
    assert _claim(fx, SRC + [5, 6])[0] >= N - 1, "the source's own end anchor is unchanged"


def test_several_twins_each_resume_at_their_own_boundary(group_p):
    fx = _fixture()
    _source_prefill(fx, [(SHARED_B - 1, S_B), (SHARED_A - 1, S_A)])  # any order in
    assert _claim(fx, TWIN_A) == (SHARED_A - 1, pytest.approx(S_A))
    assert _claim(fx, TWIN_B) == (SHARED_B - 1, pytest.approx(S_B))


# -- 2. no promise, no wait -----------------------------------------------------------
class _Sched:
    def __init__(self):
        self.ps = SimpleNamespace(pp_rank=0, pp_size=3, tp_size=1)
        self.waiting_queue = []
        self.chunked_req = None
        self.running_batch = None
        self.page_size = 64
        self.chunked_prefill_size = 16384


class _R:
    def __init__(self, rid, ids, head=0):
        self.rid = rid
        self.origin_input_ids = list(ids)
        self.extra_key = None
        self.done = False
        self._prefetch_registered_prefix_len = head
        self.prefix_indices = None
        self.fill_ids = None

    def finished(self):
        return self.done


BIG_SHARED = 87678                       # 12-19 / 12-21
BIG_B = (BIG_SHARED - 1) // 64 * 64      # 87616


def _big(shared, total, salt):
    return list(range(shared)) + [50_000_000 + salt * 1_000_000 + i for i in range(total - shared)]


@pytest.fixture
def pp0(monkeypatch):
    monkeypatch.setenv(tw.ENV_MIN_TOKENS, "8192")
    monkeypatch.setattr(tw, "_twin_anchor_armed", lambda: True)
    return _Sched()


def test_a_source_that_starts_past_the_boundary_holds_nobody(pp0):
    """12-21 is still queued, but its registered head (87680) already lies past
    12-19's boundary 87616: no source anchor can serve the twin -- it registers
    at once (base: rule 4 could not decide an unadmitted source and held it)."""
    src = _R("weg2-12-21", _big(90418, 90418, 1), head=87680)
    pp0.waiting_queue = [src]
    twin = _R("weg2-12-19", _big(BIG_SHARED, 87844, 2))
    assert tw.intake_defer(pp0, twin) is False
    assert not tw.is_deferred(pp0, "weg2-12-19")


def test_a_provisional_promise_is_dropped_when_the_source_plans_past_the_boundary(pp0):
    """16-28's shape with the source still queued at intake (registered head 0
    < boundary): provisional hold; once the source is admitted and its step
    past the boundary was planned WITHOUT the track, the twin is released the
    same pass as an ordinary request, not after the source's finish."""
    from sglang.srt.weg2 import twin_anchor as ta2

    shared = 62439
    src = _R("weg2-16-27", _big(62779, 62779, 3), head=0)
    pp0.waiting_queue = [src]
    twin = _R("weg2-16-28", _big(shared, 62856, 4))
    assert tw.intake_defer(pp0, twin) is True
    pp0.waiting_queue = [twin]
    # admitted at 59008, one step [59008, 62779) planned without the track
    src.prefix_indices = [0] * 59008
    src.fill_ids = list(range(62779))
    pp0.chunked_req = src
    tw.tick(pp0)
    assert tw.release_due(pp0, {"weg2-16-28"}) == [(twin, False)]
    # and with the track planned (the twin boundary 62400) it keeps holding
    ta2._reset_for_test()
    pp0.waiting_queue = [src]
    src2 = _R("weg2-16-37", _big(62779, 62779, 3), head=0)
    pp0.waiting_queue = [src2]
    twin2 = _R("weg2-16-38", _big(shared, 62856, 4))
    assert tw.intake_defer(pp0, twin2) is True
    pp0.waiting_queue = [twin2]
    src2.prefix_indices = [0] * 59008
    src2.fill_ids = list(range(62779))
    pp0.chunked_req = src2
    ta2.note_planned("weg2-16-37", ta2.boundary(shared, 64))
    tw.tick(pp0)
    assert tw.release_due(pp0, {"weg2-16-38"}) == []
    assert tw.is_deferred(pp0, "weg2-16-38")
