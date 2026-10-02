"""FORK TRACK (NF y5k, 30.09.; P PP0 22:12:15, weg2-0-4).

Metal: a reviewer agent's follow-up turn leaves its stored copy late.
  'WEG2 P-FORK-CUT TOLD rid=weg2-0-4 fork=12544 src=store'
  '#1472 READ-TRACE asked=220 readable=196 ... why=no-file'
  '#1028B FETCH CAP kv=196 claimed=40 lost=156 caps={QSA:196, MAMBA:40}
   anchors_in_range MAMBA=(1, 39)'
The store held 196 KV pages of the prefix; the only recurrent anchor below the
fork was page 39 (2560). The step ran [2560, 14096) in ONE forward, so the
P-FORK-CUT at 12544 was 'paid' (a second forward) and not taken -- and no
anchor was ever written at the fork. Every later request with that fork caps
at 2560 again: 9984 readable KV tokens recomputed each time.

Hermetic, CPU. Pinned (RED on e107314a32):
  A. the geometry: the cut is 'paid' there, and the fork-track position of the
     step is floor_page(fork) on the step's FLA grid, strictly inside the step
     and below its own track;
  B. the scheduler side plans one more track at the told fork in the SAME
     forward (one slot, no eviction), with or without the turn anchor, never
     twice at a position the turn track already writes, frees a stale plan;
  C. on the REAL UnifiedRadixCache (unigram and exact bigram) the next request
     with that fork resumes AT the fork with the fork state (red without the
     insert: from nothing), the END anchor is unchanged, an inner chunk too;
  D. wiring: prepare_for_extend hands the request's told fork to note_step.
"""

import ast
import inspect
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
from sglang.srt.weg2 import p_fork_cut as pfc
from sglang.srt.weg2 import turn_anchor as ta
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=15, suite="stage-a-test-cpu")

FORK_PENDING = "_weg2_fork_pending"     # turn_anchor.FORK_PENDING_ATTR, by name
IM = 248045

# y5k weg2-0-4, in tokens (page 64, FLA chunk 64, chunk 16384)
Y_PREFIX, Y_END, Y_FORK, Y_PAGE = 2560, 14096, 12544, 64

# the tree tests: a small prompt, page 1
N = 120
F = 80                                  # the told fork
SRC = list(range(1000, 1000 + N))
READER = SRC[: F + 1] + [900_000 + i for i in range(30)]   # the next request, same fork
S_EARLY, S_FORK, S_END = 0.5, 3.25, 7.5


# -- A. geometry --------------------------------------------------------------------
def test_the_y5k_cut_is_paid_so_no_chunk_ends_at_the_fork():
    """The premise: one forward [2560, 14096); a cut at 12544 costs a second."""
    assert pfc.fork_cut(Y_PREFIX, Y_END - Y_PREFIX, Y_END, Y_FORK, 16384, Y_PAGE) == (None, "paid")


def test_the_fork_track_sits_at_the_fork_inside_the_step():
    main = (Y_END // Y_PAGE) * Y_PAGE
    assert ta.fork_target(Y_PREFIX, Y_END, Y_FORK, Y_END, 64, Y_PAGE, main) == Y_FORK
    # the fork past the step's own track, at/below the step start, no track
    assert ta.fork_target(Y_PREFIX, Y_END, main + 10, Y_END + 50, 64, Y_PAGE, main) is None
    assert ta.fork_target(Y_PREFIX, Y_END, Y_PREFIX, Y_END, 64, Y_PAGE, main) is None
    assert ta.fork_target(Y_PREFIX, Y_END, Y_FORK, Y_END, 64, Y_PAGE, None) is None
    # an unaligned fork floors to the page; a step start off the page grid floors
    # further to the FLA grid (the kernel keeps states at prefix + k * chunk only)
    assert ta.fork_target(Y_PREFIX, Y_END, Y_FORK + 17, Y_END, 64, Y_PAGE, main) == Y_FORK
    assert ta.fork_target(Y_PREFIX + 32, Y_END, Y_FORK, Y_END, 64, 32, main) == Y_FORK - 32


# -- B. the scheduler side ------------------------------------------------------------
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


def _sreq(ids, rid="weg2-0-4", cpl=0):
    return SimpleNamespace(rid=rid, origin_input_ids=array("q", ids), output_ids=[],
                           cache_protected_len=cpl)


def _plan(req, alloc, prefix, end, main, fork, tok=None, chunk=1, page=1, desc=None):
    batch = SimpleNamespace(reqs=[req], req_to_token_pool=SimpleNamespace(mamba_allocator=alloc))
    return ta.note_step(batch=batch, desc=desc, req=req, row=0, prefix=prefix, end=end,
                        track_mask=True, main_track=main, chunk=chunk, page=page, tok=tok,
                        fork_told=fork)


@pytest.mark.parametrize("tok", [None, IM], ids=["turn-off", "turn-on"])
def test_the_step_plans_a_track_at_the_told_fork(tok):
    """weg2-0-4's shape: one step through the fork; one more track at it."""
    alloc = _Alloc([11, 12, 13])
    req = _sreq(SRC)
    desc = _plan(req, alloc, 0, N, N - 1, F, tok=tok)
    assert desc is not None and F in desc.targets
    t, slot, d, s0, s1 = getattr(req, FORK_PENDING)
    assert (t, int(slot), d, s0, s1) == (F, 11, desc, 0, N)
    assert ta.has_pending(req)
    assert [k for k, _ in ta.pop_all_pending(req)] == ["fork"]
    assert getattr(req, FORK_PENDING, None) is None


def test_the_y5k_step_plans_on_the_real_grid():
    alloc = _Alloc([7])
    req = _sreq(list(range(Y_END)))
    main = (Y_END // Y_PAGE) * Y_PAGE
    desc = _plan(req, alloc, Y_PREFIX, Y_END, main, Y_FORK, chunk=64, page=Y_PAGE)
    assert desc.targets == [Y_FORK] and desc.prefixes == [Y_PREFIX]
    assert desc.ext_lens == [Y_END - Y_PREFIX]


@pytest.mark.parametrize("why,kw", [
    ("no told fork", dict(alloc=[5], prefix=0, end=N, main=N - 1, fork=0)),
    ("no slot free", dict(alloc=[], prefix=0, end=N, main=N - 1, fork=F)),
    ("step after the fork", dict(alloc=[5], prefix=F, end=N, main=N - 1, fork=F)),
    ("the step's own track at the fork (a free cut)", dict(alloc=[5], prefix=0, end=F + 1, main=F, fork=F)),
    ("step before the fork", dict(alloc=[5], prefix=0, end=F - 10, main=F - 11, fork=F)),
])
def test_the_step_plans_no_fork_track(why, kw):
    alloc = _Alloc(kw.pop("alloc"))
    req = _sreq(SRC)
    desc = _plan(req, alloc, kw["prefix"], kw["end"], kw["main"], kw["fork"])
    assert desc is None and getattr(req, FORK_PENDING, None) is None, why
    assert alloc.freed == []


def test_a_protected_prefix_at_the_fork_plans_nothing():
    req = _sreq(SRC, cpl=F)
    assert _plan(req, _Alloc([5]), 0, N, N - 1, F) is None


def test_a_fork_at_the_turn_position_is_one_track():
    """The turn track already writes that position: no second slot."""
    ids = [IM, 1, 2] + list(range(3000, 3000 + 40)) + [IM, 5, 6] + list(range(4000, 4010)) + [IM, 7, 8]
    req = _sreq(ids)
    t_turn = ta.req_anchor_pos(req, IM, 1)
    alloc = _Alloc([11, 12])
    desc = _plan(req, alloc, 0, len(ids), len(ids) - 1, t_turn, tok=IM)
    assert desc.targets == [t_turn] and getattr(req, FORK_PENDING, None) is None
    assert alloc.free_ids == [12]


def test_a_stale_fork_plan_is_freed_before_a_new_one():
    alloc = _Alloc([11, 12])
    req = _sreq(SRC)
    _plan(req, alloc, 0, N, N - 1, F)
    _plan(req, alloc, 0, N, N - 1, F)
    assert alloc.freed == [11] and int(getattr(req, FORK_PENDING)[1]) == 12


# -- C. the real tree ---------------------------------------------------------------
MAMBA_SLOTS = 20
KV_SIZE = 512
NUM_LAYERS = 8
FULL_LAYER_IDS = (3, 7)
MAMBA_LAYER_IDS = [i for i in range(NUM_LAYERS) if i not in FULL_LAYER_IDS]


@pytest.fixture
def group_p(monkeypatch):
    from sglang.srt.mem_cache import unified_radix_cache as urc

    monkeypatch.setenv("SGLANG_WEG2_GROUP", "P")
    monkeypatch.setenv("SGLANG_WEG2_BIGRAM_ANCHOR_EXACT", "1")
    monkeypatch.setattr(urc, "_WEG2_END_ANCHOR", True)
    return monkeypatch


def _fixture(bigram):
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
        eviction_policy="lru", is_eagle=bigram,
    )
    cache = UnifiedRadixCache(params=params)
    cache.cache_init_params = params
    return SimpleNamespace(cache=cache, allocator=allocator, pool=pool)


def _key(fx, ids):
    return RadixKey(array("q", ids), is_bigram=fx.cache.is_eagle)


def _req(fx, ids):
    req = Req(rid="weg2-0-4", origin_input_text="", origin_input_ids=array("q", ids),
              sampling_params=SamplingParams(temperature=0, max_new_tokens=1))
    fx.pool.alloc([req])
    req.output_ids = array("q")
    req.full_untruncated_fill_ids = array("q", ids)
    req.swa_uuid_for_lock = None
    req.extra_key = None
    mr = fx.cache.match_prefix(MatchPrefixParams(key=_key(fx, ids)))
    req.prefix_indices = mr.device_indices.to(torch.int64)
    req.cache_protected_len = len(req.prefix_indices)
    req.last_node = mr.last_device_node
    fx.cache.inc_lock_ref(req.last_node)
    return req


def _state(fx, slot, value):
    fx.pool.mamba_pool.mamba_cache.temporal[:, slot] = value


def _write_kv(fx, req, end):
    start = len(req.prefix_indices)
    if end > start:
        fx.pool.write((req.req_pool_idx, slice(start, end)), fx.allocator.alloc(end - start))
    req.set_extend_range(start, end)


def _planned_step(fx, req, prefix, end, main):
    """The scheduler's own plan against the real pool, then what the forward
    leaves: the fork slot holds the state after F tokens, the GDN mark set."""
    batch = SimpleNamespace(reqs=[req], req_to_token_pool=fx.pool)
    desc = ta.note_step(batch=batch, desc=None, req=req, row=0, prefix=prefix, end=end,
                        track_mask=True, main_track=main, chunk=1, page=1, tok=None,
                        fork_told=F)
    assert desc is not None, "the step runs through the told fork: a plan is drawn"
    _state(fx, getattr(req, FORK_PENDING)[1], S_FORK)
    desc.done.add("gdn")


def _claim(fx, ids):
    claim = Req._compute_max_prefix_len(
        SimpleNamespace(return_logprob=False, logprob_start_len=0), len(ids))
    mr = fx.cache.match_prefix(MatchPrefixParams(key=_key(fx, ids[:claim])))
    slot = mr.last_device_node.component_data[ComponentType.MAMBA].value
    state = None if slot is None else float(
        fx.pool.mamba_pool.mamba_cache.temporal[:, slot].float().mean())
    return len(mr.device_indices), state


def _one_forward(fx, plan=True):
    req = _req(fx, SRC)
    _write_kv(fx, req, N)
    if plan:
        _planned_step(fx, req, 0, N, N - 1)
    _state(fx, req.mamba_pool_idx, S_END)
    req.kv_committed_len = N
    req.kv_allocated_len = N
    fx.cache.cache_finished_req(req, is_insert=True)
    return req


@pytest.mark.parametrize("bigram", [False, True], ids=["unigram", "bigram"])
def test_the_next_request_with_that_fork_resumes_at_the_fork(group_p, bigram):
    """Red on the base: the one-step prompt holds only its END anchor, the
    reader resumes from nothing (on the boot: the old anchor 2560)."""
    fx = _fixture(bigram)
    _one_forward(fx)
    assert _claim(fx, READER) == (F, pytest.approx(S_FORK))


@pytest.mark.parametrize("bigram", [False, True], ids=["unigram", "bigram"])
def test_the_end_anchor_is_unchanged(group_p, bigram):
    later = SRC + [5, 6, 7]
    fx = _fixture(bigram)
    _one_forward(fx)
    fx0 = _fixture(bigram)
    _one_forward(fx0, plan=False)
    assert _claim(fx, later) == _claim(fx0, later)
    assert _claim(fx, later) == (N - (1 if bigram else 0), pytest.approx(S_END))
    assert _claim(fx0, READER)[1] != pytest.approx(S_FORK), "the gap itself on the base"


def test_the_tree_consumes_a_fork_plan(group_p):
    """What prepare_for_extend + the forward leave (plan set by hand, no
    planner call): the finish insert puts the fork node in. On the base the
    tree ignores the plan -- the reader resumes from nothing, the slot leaks."""
    fx = _fixture(False)
    free_before = fx.pool.mamba_allocator.available_size()
    req = _req(fx, SRC)
    _write_kv(fx, req, N)
    slot = fx.pool.mamba_allocator.alloc(1)
    _state(fx, slot, S_FORK)
    desc = ta.TurnTracks(1)
    desc.add(0, slot, F, 0, N)
    desc.done.add("gdn")
    setattr(req, FORK_PENDING, (F, slot, desc, 0, N))
    _state(fx, req.mamba_pool_idx, S_END)
    req.kv_committed_len = N
    req.kv_allocated_len = N
    fx.cache.cache_finished_req(req, is_insert=True)
    assert _claim(fx, READER) == (F, pytest.approx(S_FORK))
    assert getattr(req, FORK_PENDING, None) is None
    # both slots (END + fork) now belong to the tree
    assert fx.pool.mamba_allocator.available_size() == free_before - 2


def test_an_inner_chunk_through_the_fork_also_inserts_it(group_p):
    fx = _fixture(False)
    req = _req(fx, SRC)
    end = F + 16
    _write_kv(fx, req, end)
    _planned_step(fx, req, 0, end, end)
    _state(fx, req.mamba_pool_idx, S_EARLY)
    fx.cache.cache_unfinished_req(req, chunked=True)
    assert req.cache_protected_len == end, "the chunk's own insert still lands at its end"
    assert _claim(fx, READER) == (F, pytest.approx(S_FORK))
    assert _claim(fx, SRC[:end] + [1, 2]) == (end, pytest.approx(S_EARLY))


def test_an_unmarked_fork_plan_inserts_nothing_and_frees_its_slot(group_p):
    fx = _fixture(False)
    free_before = fx.pool.mamba_allocator.available_size()
    req = _req(fx, SRC)
    _write_kv(fx, req, N)
    batch = SimpleNamespace(reqs=[req], req_to_token_pool=fx.pool)
    ta.note_step(batch=batch, desc=None, req=req, row=0, prefix=0, end=N, track_mask=True,
                 main_track=N - 1, chunk=1, page=1, tok=None, fork_told=F)
    _state(fx, req.mamba_pool_idx, S_END)
    req.kv_committed_len = N
    req.kv_allocated_len = N
    fx.cache.cache_finished_req(req, is_insert=True)   # no GDN mark: a graph replay
    assert _claim(fx, READER)[1] != pytest.approx(S_FORK)
    assert fx.pool.mamba_allocator.available_size() == free_before - 1


# -- D. wiring -----------------------------------------------------------------------
def test_prepare_for_extend_hands_the_told_fork_to_the_planner():
    """The told fork rides on the request (weg2_store_told.admission sets
    ``req._weg2_fork_told`` on every rank); the planner call must read it."""
    from sglang.srt.managers import schedule_batch as sb

    tree = ast.parse(inspect.getsource(sb))
    calls = [n for n in ast.walk(tree) if isinstance(n, ast.Call)
             and isinstance(n.func, ast.Attribute) and n.func.attr == "note_step"
             and isinstance(n.func.value, ast.Name) and n.func.value.id == "_weg2_turn"]
    assert len(calls) == 1
    kw = {k.arg: k.value for k in calls[0].keywords}
    assert "fork_told" in kw
    assert "_weg2_fork_told" in ast.unparse(kw["fork_told"])
