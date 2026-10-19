"""S57 (19.10.2026): GEN-FORK TRACK -- a sibling that forks at the generation
prompt finds an anchor at floor_page(g), not only the step's read start.

Metal (NF rc15, boot ..._1019_071429, image 489909d138):
  P  07:57:35 '#969 EXTENT ... reqs=[('pdflip-40-260', 39424, 49667, 39424, 10243)]'
              (ONE step [39424, 49667), no TURN-ANCHOR TRACK: the last message started
              in the step's first page, turn position 39424 = the step start)
  P  07:57:40 'PDFLIP END-ANCHOR ... rid=pdflip-40-260 tokens=49667 anchor=49664 target=49666'
  fr 07:57:49 'PREFIX-DIVERGE rid=pdflip-42-262 verdict=store_short prompt=49826 credit=39424
              ... best_prev=pdflip-40-260 prev_prompt=49667 ... common=49663'
  P  07:57:54 'PDFLIP-CLAIM-SHORT rid=pdflip-42-262 ... asked=778 kv=775 claimed=616 lost=159 by=mamba'
  P  07:57:56 '#969 EXTENT ... reqs=[('pdflip-42-262', 39424, 49826, 39424, 10402)]'
  P  07:57:56 'FORK-ANCHOR SHARED rid=pdflip-42-262 at=49600 ...' (42-262's own last message
              starts at 40-260's generation-prompt <|im_start|>: index 49662 -> page 49600)
The sibling shares ids[:49663] = through the generation prompt's <|im_start|> (g = N-5 =
49662) and claims floor_page(g) = 49600. The END anchor floor_page(N-1) = 49664 lies one
token past it; KV was in the store to 49600, the recurrent anchor only at 39424:
'RECOMPUTE-ECHT-P rid=pdflip-42-262: 10239 Token neu'.

Pinned (hermetic, CPU; RED on rc15 489909d138):
  A. geometry: gen_fork_pos is floor_page(g); with NF's 5-token generation prompt the
     step plans the track exactly for N % 64 in {1, 2, 3, 4} (the END anchor misses), never
     otherwise; 40-260's turn position is the step start (no turn track);
  B. the planner: 40-260's step on the real grid plans one more track at 49600; switch off
     = nothing (rc15 byte for byte); no slot = no plan, no promise; shared with the turn/
     fork track = one slot; a stale gen plan is freed and its promise declined;
  C. the promise (S52 ledger): a twin of 40-260 sharing 49663 is held for the gen track
     (p_twin_defer rule 5, promise at 49600) instead of TWIN-NO-COMMIT;
  D. the REAL UnifiedRadixCache (unigram and exact bigram): the sibling resumes AT g with
     the state the gen track wrote; the END anchor and the turn anchor are unchanged;
  E. wiring: the radix cache consumes a gen plan (finish and chunk insert), the switch is
     registered default ON.
"""

import ast
import inspect
from array import array
from types import SimpleNamespace

import pytest
import torch

from flliper.srt.configs.mamba_utils import Mamba2CacheParams, Mamba2StateShape
from flliper.srt.environ import envs
from flliper.srt.layers.attention.fla.chunk_delta_h import CHUNK_SIZE as FLA_CHUNK_SIZE
from flliper.srt.managers.schedule_batch import Req
from flliper.srt.mem_cache.allocator import TokenToKVPoolAllocator
from flliper.srt.mem_cache.base_prefix_cache import MatchPrefixParams
from flliper.srt.mem_cache.cache_init_params import CacheInitParams
from flliper.srt.mem_cache.memory_pool import HybridLinearKVPool, HybridReqToTokenPool
from flliper.srt.mem_cache.radix_cache import RadixKey
from flliper.srt.mem_cache.unified_cache_components.tree_component import ComponentType
from flliper.srt.mem_cache.unified_radix_cache import UnifiedRadixCache
from flliper.srt.sampling.sampling_params import SamplingParams
from flliper.srt.server_args import ServerArgs, set_global_server_args_for_scheduler
from flliper.srt.pdflip import turn_anchor as ta
from flliper.srt.pdflip import twin_anchor as twa
from flliper.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=15, suite="stage-a-test-cpu")

GEN_PENDING = "_pdflip_gen_pending"      # turn_anchor.GEN_PENDING_ATTR, by name
FORK_PENDING = "_pdflip_fork_pending"
TURN_PENDING = "_pdflip_turn_pending"
IM = 248045                               # <|im_start|> on NF
GEN = 5                                   # '<|im_start|>assistant\n<think>\n' on NF

# rc15 pdflip-40-260 / 42-262 in tokens (page 64, FLA chunk 64)
R_PREFIX, R_N, R_PAGE = 39424, 49667, 64
R_LAST_MSG = 39430                        # inside the step's first page (turn pos = step start)
R_G = R_N - GEN                           # 49662
R_COMMON = 49663                          # front PREFIX-DIVERGE common
R_GEN_POS = 49600


def _ids(n, last_msg, gen=GEN, base=1000):
    ids = [base + (i % 50_000) for i in range(n)]
    ids[0] = IM
    ids[last_msg] = IM
    ids[n - gen] = IM
    return ids


R_SRC = _ids(R_N, R_LAST_MSG)


@pytest.fixture(autouse=True)
def _gen_on(monkeypatch):
    monkeypatch.setenv("FLLIPER_PDFLIP_GEN_FORK_ANCHOR", "1")
    twa._reset_for_test()
    yield
    twa._reset_for_test()


# -- A. geometry ---------------------------------------------------------------------
def test_the_gen_fork_position_is_the_page_under_the_generation_prompt():
    assert ta.gen_fork_pos(R_SRC, IM, R_PAGE) == R_GEN_POS
    # the sibling shares ids[:common] = ids[:g + 1]: its claim is floor_page(g)
    assert R_COMMON == R_G + 1 and (R_G // R_PAGE) * R_PAGE == R_GEN_POS
    assert ta.gen_fork_pos(R_SRC, None, R_PAGE) is None
    assert ta.gen_fork_pos(R_SRC[:-GEN] + [7] * GEN, IM, R_PAGE) is None   # no generation prompt


def test_the_rc15_step_holds_no_turn_track_but_the_end_anchor_misses_the_sibling():
    """The premise on the metal: turn position = the step start, END past g."""
    main = ((R_N - 1) // R_PAGE) * R_PAGE
    assert main == 49664 and main > R_G
    t_turn = ta.anchor_pos(R_SRC, IM, R_PAGE)
    assert t_turn == R_PREFIX
    assert ta.step_target(R_PREFIX, R_N, t_turn, 64, main) is None
    assert ta.step_target(R_PREFIX, R_N, R_GEN_POS, 64, main) == R_GEN_POS


@pytest.mark.parametrize("n", list(range(49664, 49664 + 2 * R_PAGE + 1)))
def test_the_track_is_planned_exactly_where_the_end_anchor_misses(n):
    """END anchor floor_page(n-1) serves a sibling iff it is <= g = n-5; the
    gen track is planned in the complement only (4 of 64 lengths on NF)."""
    ids = _ids(n, R_LAST_MSG)
    main = ((n - 1) // R_PAGE) * R_PAGE
    end_serves = main <= n - GEN
    alloc = _Alloc([3, 4])
    req = _sreq(ids)
    _plan(req, alloc, R_PREFIX, n, main, tok=IM, chunk=64, page=R_PAGE)
    planned = getattr(req, GEN_PENDING, None) is not None
    assert planned == (not end_serves)
    assert planned == (n % R_PAGE in (1, 2, 3, 4))
    if planned:
        assert getattr(req, GEN_PENDING)[0] == ((n - GEN) // R_PAGE) * R_PAGE


# -- B. the planner ------------------------------------------------------------------
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


def _sreq(ids, rid="pdflip-40-260", cpl=0):
    return SimpleNamespace(rid=rid, origin_input_ids=array("q", ids), output_ids=[],
                           cache_protected_len=cpl)


def _plan(req, alloc, prefix, end, main, tok=IM, chunk=1, page=1, fork=0, desc=None,
          twin_bounds=()):
    batch = SimpleNamespace(reqs=[req], req_to_token_pool=SimpleNamespace(mamba_allocator=alloc))
    return ta.note_step(batch=batch, desc=desc, req=req, row=0, prefix=prefix, end=end,
                        track_mask=True, main_track=main, chunk=chunk, page=page, tok=tok,
                        twin_bounds=twin_bounds, fork_told=fork)


def test_the_rc15_step_plans_a_gen_track_at_49600():
    alloc = _Alloc([7, 8])
    req = _sreq(R_SRC)
    desc = _plan(req, alloc, R_PREFIX, R_N, 49664, chunk=64, page=R_PAGE)
    assert desc is not None and desc.targets == [R_GEN_POS]
    t, slot, d, s0, s1 = getattr(req, GEN_PENDING)
    assert (t, int(slot), d, s0, s1) == (R_GEN_POS, 7, desc, R_PREFIX, R_N)
    # the forward gathers the state at the very position the tree inserts (no off-by-page)
    assert desc.prefixes == [R_PREFIX] and desc.ext_lens == [R_N - R_PREFIX]
    assert getattr(req, TURN_PENDING, None) is None
    assert ta.has_pending(req)
    assert [k for k, _ in ta.pop_all_pending(req)] == ["gen"]
    assert getattr(req, GEN_PENDING, None) is None


def test_switch_off_plans_nothing(monkeypatch):
    monkeypatch.setenv("FLLIPER_PDFLIP_GEN_FORK_ANCHOR", "0")
    alloc = _Alloc([7])
    req = _sreq(R_SRC)
    assert _plan(req, alloc, R_PREFIX, R_N, 49664, chunk=64, page=R_PAGE) is None
    assert getattr(req, GEN_PENDING, None) is None and alloc.free_ids == [7]


def test_no_slot_no_plan_no_promise(monkeypatch):
    monkeypatch.setattr(twa, "_ARMED", True)
    req = _sreq(R_SRC)
    assert _plan(req, _Alloc([]), R_PREFIX, R_N, 49664, chunk=64, page=R_PAGE) is None
    assert getattr(req, GEN_PENDING, None) is None
    assert twa.status(req.rid, R_GEN_POS) is None, "a promise without a slot holds a twin for nothing"


@pytest.mark.parametrize("why,kw", [
    ("decode step (output exists)", dict(out=[1])),
    ("protected prefix at the position", dict(cpl=R_GEN_POS)),
    ("step starts past the position", dict(prefix=R_GEN_POS)),
    ("own track at/below the position", dict(main=R_GEN_POS)),
    ("track mask off", dict(mask=False)),
])
def test_no_gen_track(why, kw):
    alloc = _Alloc([7, 8])
    req = _sreq(R_SRC, cpl=kw.get("cpl", 0))
    req.output_ids = kw.get("out", [])
    batch = SimpleNamespace(reqs=[req], req_to_token_pool=SimpleNamespace(mamba_allocator=alloc))
    ta.note_step(batch=batch, desc=None, req=req, row=0, prefix=kw.get("prefix", R_PREFIX),
                 end=R_N, track_mask=kw.get("mask", True), main_track=kw.get("main", 49664),
                 chunk=64, page=R_PAGE, tok=IM)
    assert getattr(req, GEN_PENDING, None) is None, why


def test_a_gen_position_the_turn_track_writes_is_one_slot():
    """A one-line last message: turn floor_page(i) == floor_page(g)."""
    n = 49667
    ids = _ids(n, n - GEN - 20)
    alloc = _Alloc([11, 12])
    req = _sreq(ids)
    desc = _plan(req, alloc, R_PREFIX, n, 49664, chunk=64, page=R_PAGE)
    assert desc.targets == [R_GEN_POS]
    assert getattr(req, TURN_PENDING)[0] == R_GEN_POS and getattr(req, GEN_PENDING, None) is None
    assert alloc.free_ids == [12]


def test_a_gen_position_the_fork_track_writes_is_one_slot():
    alloc = _Alloc([11, 12])
    req = _sreq(R_SRC)
    desc = _plan(req, alloc, R_PREFIX, R_N, 49664, chunk=64, page=R_PAGE, fork=R_GEN_POS + 10)
    assert desc.targets == [R_GEN_POS] and getattr(req, FORK_PENDING)[0] == R_GEN_POS
    assert getattr(req, GEN_PENDING, None) is None and alloc.free_ids == [12]


def test_a_stale_gen_plan_is_freed_and_its_promise_declined(monkeypatch):
    monkeypatch.setattr(twa, "_ARMED", True)
    alloc = _Alloc([11, 12])
    req = _sreq(R_SRC)
    _plan(req, alloc, R_PREFIX, R_N, 49664, chunk=64, page=R_PAGE)
    assert twa.status(req.rid, R_GEN_POS) == "planned"
    # a later step (here: the same one re-planned) frees the unconsumed plan first
    _plan(req, alloc, R_PREFIX, R_N, 49664, chunk=64, page=R_PAGE)
    assert alloc.freed == [11] and int(getattr(req, GEN_PENDING)[1]) == 12
    # the declined-then-replanned promise stands again (planned), never left dangling
    assert twa.status(req.rid, R_GEN_POS) == "planned"
    req.output_ids = [1]
    _plan(req, alloc, R_N, R_N + 1, R_N, chunk=64, page=R_PAGE)   # decode: stale freed
    assert alloc.freed == [11, 12] and twa.status(req.rid, R_GEN_POS) == "declined"


def test_a_skipped_insert_declines_the_promise(monkeypatch):
    """Haenger direction: an unmarked plan (graph replay) writes nothing -- the
    ledger must say declined, or a twin waits on it until its Frist."""
    monkeypatch.setattr(twa, "_ARMED", True)
    req = _sreq(R_SRC)
    _plan(req, _Alloc([5]), R_PREFIX, R_N, 49664, chunk=64, page=R_PAGE)
    (kind, pend), = ta.pop_all_pending(req)
    ok, why = ta.insert_verdict(pend)
    assert (kind, ok) == ("gen", False) and why.startswith("unmarked")
    ta.note_skip(req, why, pend[0], kind=kind)
    assert twa.status(req.rid, R_GEN_POS) == "declined"


# -- C. the promise ------------------------------------------------------------------
def test_a_twin_of_40_260_is_held_for_the_gen_track(monkeypatch):
    """42-262 had arrived while 40-260 still ran: S52's ledger sees the gen track
    as the source's own track in (s0, B] -- PROMISE_TRACK at 49600, not NO-COMMIT."""
    from flliper.srt.pdflip import p_twin_defer as tw

    monkeypatch.setattr(twa, "_ARMED", True)
    src = _sreq(R_SRC)
    src.finished = lambda: False
    src.fill_ids = list(range(R_N))     # its one step [39424, 49667) was planned (metal)
    st = SimpleNamespace(page=R_PAGE, chunk=16384, s0={src.rid: R_PREFIX})
    assert tw._promise(st, src, R_COMMON) is None, "rc15: nothing to wait for"
    _plan(src, _Alloc([5]), R_PREFIX, R_N, 49664, chunk=64, page=R_PAGE)
    # the gen track sits exactly on the twin boundary B: booked in the S52 ledger, the
    # boundary rule finds it first (any promise at 49600 holds the twin)
    assert tw._promise(st, src, R_COMMON) in ((tw.PROMISE_BOUNDARY, R_GEN_POS),
                                             (tw.PROMISE_TRACK, R_GEN_POS))
    assert twa.boundary(R_COMMON, R_PAGE) == R_GEN_POS


# -- D. the real tree ----------------------------------------------------------------
MAMBA_SLOTS = 20
KV_SIZE = 512
NUM_LAYERS = 8
FULL_LAYER_IDS = (3, 7)
MAMBA_LAYER_IDS = [i for i in range(NUM_LAYERS) if i not in FULL_LAYER_IDS]

N = 120
TURN = 40
G = N - GEN                                 # 115
SRC = [1000 + i for i in range(N)]
SRC[0] = IM
SRC[TURN] = IM
SRC[G] = IM
SIBLING = SRC[: G + 1] + [900_000 + i for i in range(30)]   # a different message after <|im_start|>
NEXT_TURN = SRC[: TURN + 1] + [800_000 + i for i in range(30)]
S_TURN, S_GEN, S_END = 2.25, 4.5, 7.5


@pytest.fixture
def group_p(monkeypatch):
    from flliper.srt.mem_cache import unified_radix_cache as urc

    monkeypatch.setenv("FLLIPER_PDFLIP_GROUP", "P")
    monkeypatch.setenv("FLLIPER_PDFLIP_BIGRAM_ANCHOR_EXACT", "1")
    monkeypatch.setattr(urc, "_PDFLIP_END_ANCHOR", True)
    return monkeypatch


def _fixture(bigram):
    server_args = ServerArgs(model_path="dummy", page_size=1)
    server_args._mamba_cache_chunk_size = FLA_CHUNK_SIZE
    server_args.chunked_prefill_size = 16
    set_global_server_args_for_scheduler(server_args)
    with envs.FLLIPER_MAMBA_SSM_DTYPE.override("bfloat16"):
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
    req = Req(rid="pdflip-40-260", origin_input_text="", origin_input_ids=array("q", ids),
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


def _claim(fx, ids):
    claim = Req._compute_max_prefix_len(
        SimpleNamespace(return_logprob=False, logprob_start_len=0), len(ids))
    mr = fx.cache.match_prefix(MatchPrefixParams(key=_key(fx, ids[:claim])))
    slot = mr.last_device_node.component_data[ComponentType.MAMBA].value
    state = None if slot is None else float(
        fx.pool.mamba_pool.mamba_cache.temporal[:, slot].float().mean())
    return len(mr.device_indices), state


def _one_forward(fx, plan=True):
    """40-260's shape at page 1: ONE step [0, N), turn track at TURN, END at N-1;
    the forward leaves each planned slot holding the state after its position."""
    req = _req(fx, SRC)
    _write_kv(fx, req, N)
    if plan:
        batch = SimpleNamespace(reqs=[req], req_to_token_pool=fx.pool)
        desc = ta.note_step(batch=batch, desc=None, req=req, row=0, prefix=0, end=N,
                            track_mask=True, main_track=N - 1, chunk=1, page=1, tok=IM)
        assert desc is not None
        if getattr(req, TURN_PENDING, None) is not None:
            _state(fx, getattr(req, TURN_PENDING)[1], S_TURN)
        if getattr(req, GEN_PENDING, None) is not None:
            _state(fx, getattr(req, GEN_PENDING)[1], S_GEN)
        desc.done.add("gdn")
    _state(fx, req.mamba_pool_idx, S_END)
    req.kv_committed_len = N
    req.kv_allocated_len = N
    fx.cache.cache_finished_req(req, is_insert=True)
    return req


@pytest.mark.parametrize("bigram", [False, True], ids=["unigram", "bigram"])
def test_the_sibling_resumes_at_the_generation_prompt(group_p, bigram):
    """RED on rc15: the sibling falls back to the turn anchor (on the metal: 39424)."""
    fx = _fixture(bigram)
    _one_forward(fx)
    assert _claim(fx, SIBLING) == (G, pytest.approx(S_GEN))


@pytest.mark.parametrize("bigram", [False, True], ids=["unigram", "bigram"])
def test_end_and_turn_anchor_are_unchanged(group_p, bigram, monkeypatch):
    later = SRC + [5, 6, 7]
    fx = _fixture(bigram)
    _one_forward(fx)
    monkeypatch.setenv("FLLIPER_PDFLIP_GEN_FORK_ANCHOR", "0")
    fx0 = _fixture(bigram)
    _one_forward(fx0)
    assert _claim(fx, later) == _claim(fx0, later)
    assert _claim(fx, later) == (N - (1 if bigram else 0), pytest.approx(S_END))
    assert _claim(fx, NEXT_TURN) == _claim(fx0, NEXT_TURN) == (TURN, pytest.approx(S_TURN))
    assert _claim(fx0, SIBLING) == (TURN, pytest.approx(S_TURN)), "the gap itself, switch off"


def test_the_tree_consumes_a_gen_plan(group_p):
    """A plan set by hand (no planner): the finish insert puts the node in, both
    slots (END + gen) belong to the tree, the attribute is cleared."""
    fx = _fixture(False)
    free_before = fx.pool.mamba_allocator.available_size()
    req = _req(fx, SRC)
    _write_kv(fx, req, N)
    slot = fx.pool.mamba_allocator.alloc(1)
    _state(fx, slot, S_GEN)
    desc = ta.TurnTracks(1)
    desc.add(0, slot, G, 0, N)
    desc.done.add("gdn")
    setattr(req, GEN_PENDING, (G, slot, desc, 0, N))
    _state(fx, req.mamba_pool_idx, S_END)
    req.kv_committed_len = N
    req.kv_allocated_len = N
    fx.cache.cache_finished_req(req, is_insert=True)
    assert _claim(fx, SIBLING) == (G, pytest.approx(S_GEN))
    assert getattr(req, GEN_PENDING, None) is None
    assert fx.pool.mamba_allocator.available_size() == free_before - 2


def test_an_inner_chunk_consumes_a_gen_plan(group_p):
    fx = _fixture(False)
    req = _req(fx, SRC)
    end = G + 3
    _write_kv(fx, req, end)
    slot = fx.pool.mamba_allocator.alloc(1)
    _state(fx, slot, S_GEN)
    desc = ta.TurnTracks(1)
    desc.add(0, slot, G, 0, end)
    desc.done.add("gdn")
    setattr(req, GEN_PENDING, (G, slot, desc, 0, end))
    _state(fx, req.mamba_pool_idx, S_END)
    fx.cache.cache_unfinished_req(req, chunked=True)
    assert req.cache_protected_len == end
    assert _claim(fx, SIBLING) == (G, pytest.approx(S_GEN))


def test_an_unmarked_gen_plan_inserts_nothing_and_frees_its_slot(group_p):
    fx = _fixture(False)
    free_before = fx.pool.mamba_allocator.available_size()
    req = _req(fx, SRC)
    _write_kv(fx, req, N)
    slot = fx.pool.mamba_allocator.alloc(1)
    desc = ta.TurnTracks(1)
    desc.add(0, slot, G, 0, N)          # no GDN mark: a graph replay
    setattr(req, GEN_PENDING, (G, slot, desc, 0, N))
    _state(fx, req.mamba_pool_idx, S_END)
    req.kv_committed_len = N
    req.kv_allocated_len = N
    fx.cache.cache_finished_req(req, is_insert=True)
    assert _claim(fx, SIBLING)[1] != pytest.approx(S_GEN)
    assert fx.pool.mamba_allocator.available_size() == free_before - 1


# -- E. wiring -----------------------------------------------------------------------
def test_the_switch_is_registered_default_on(monkeypatch):
    monkeypatch.delenv("FLLIPER_PDFLIP_GEN_FORK_ANCHOR", raising=False)
    assert envs.FLLIPER_PDFLIP_GEN_FORK_ANCHOR.get() is True
    assert ta.gen_fork_armed() is True


def test_the_gen_position_reads_no_rank_local_state():
    """R11 / #1004: the position is a pure function of the ids, the env token and
    the page -- no pp rank, no tree, no store, no clock in its derivation."""
    src = inspect.getsource(ta.gen_fork_pos) + inspect.getsource(ta.req_gen_pos)
    tree = ast.parse(src)
    names = {n.id for n in ast.walk(tree) if isinstance(n, ast.Name)}
    attrs = {n.attr for n in ast.walk(tree) if isinstance(n, ast.Attribute)}
    # getattr(x, "pp_rank") hides the attribute in a string constant
    attrs |= {n.value for n in ast.walk(tree) if isinstance(n, ast.Constant) and isinstance(n.value, str)}
    for bad in ("pp_rank", "tree_cache", "cache_controller", "time", "random", "torch",
                "cache_protected_len", "prefix_indices"):
        assert bad not in names and bad not in attrs, bad
