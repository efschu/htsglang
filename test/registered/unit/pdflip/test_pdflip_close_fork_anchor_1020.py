"""CLOSE-FORK TRACK (20.10.2026): a side request that continues the message BEFORE the
last one finds an anchor at floor_page(e - 1), not only the step's read start.

Metal (NF rc27, boot ..._1020_145331, image 893d09fa3f):
  fr 17:02:12 'PDFLIP-ROUTE rid=pdflip-50-825 LONG -> P leg 1 (uncached=17824 ...)' (prompt 36064)
  P  17:02:20 '#969 EXTENT ... reqs=[('pdflip-50-825', 18240, 34624, 18240, 16384)]'
  P  17:02:20 'TURN-ANCHOR TRACK ... rid=pdflip-50-825 step=[18240,34624) turn=29056
              boundary=29057 prompt=36064 main=34624'
  P  17:02:27 '#59 FINISH-TRACK rid=pdflip-50-825 cache_len=36032' / 'END-ANCHOR ... anchor=36032'
  fr 17:02:24 'PROMPT-DIVERGE rid=pdflip-51-826 unit=token at=29055 ...
              prev=...</system-reminder><|im_end|>\\n<|im_start|>user ... new=...</system-reminder>\\nDescribe'
  fr 17:02:24 'SESSION-PREFIX rid=pdflip-51-826 ... prev_rid=pdflip-50-825 common=29055'
  P  17:02:27 'MATCH-CENSUS-DEEP rid=pdflip-51-826 reached=28992 accepted=18240 branching=28992'
              (after 50-825's FINISH-TRACK: P's OWN device tree held the KV to 28992)
  P  17:02:27 '#969 EXTENT ... ('pdflip-51-826', 18240, 28992, 18240, 10752)'
  mon        'STORE-MISS-P rid=pdflip-51-826 Start-Cache 18240 < belegte Tiefe 28992 -> 10752'
The side request shares ids[:29055] -- through the content of the message before the last
one, up to its <|im_end|> at e = i - 2 -- and claims floor_page(e - 1) = 28992. The turn
anchor floor_page(i) = 29056 lies ONE token past the fork (i % 64 = 1), the recurrent anchor
at or below 28992 was the step start 18240.

Pinned (hermetic, CPU; RED on rc28 297198745d):
  A. geometry: close_fork_pos is floor_page(i - 3); planned exactly for i % 64 in {0, 1, 2}
     (where the turn anchor misses), never otherwise; the separator is verified;
  B. the planner: 50-825's step plans the turn track AND one more at 28992; no slot = no
     plan, no promise; shared with another track = one slot; stale plan freed + declined;
  C. the promise (S52 ledger): a twin sharing 29055 is held for the close track;
  D. PP3 rank agreement: three stages plan the same positions from the same ids;
  E. the REAL UnifiedRadixCache (unigram and exact bigram): the side request resumes at
     e - 1 with the state the close track wrote; END / turn / gen anchors unchanged;
  F. wiring: the radix cache consumes a close plan (finish and chunk insert); the position
     reads no rank-local state; no own switch (armed with the turn anchor).
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

CLOSE_PENDING = "_pdflip_close_pending"   # turn_anchor.CLOSE_PENDING_ATTR, by name
GEN_PENDING = "_pdflip_gen_pending"
FORK_PENDING = "_pdflip_fork_pending"
TURN_PENDING = "_pdflip_turn_pending"
IM = 248045                               # <|im_start|> on NF
END = 248046                              # <|im_end|>
NL = 198                                  # '\n'
GEN = 5                                   # '<|im_start|>assistant\n<think>\n' on NF

# rc27 pdflip-50-825 / 51-826 in tokens (page 64, FLA chunk 64)
R_PREFIX, R_STEP_END, R_N, R_PAGE = 18240, 34624, 36064, 64
R_LAST_MSG = 29057                        # boundary=29057 (P TURN-ANCHOR TRACK)
R_TURN_POS = 29056                        # turn=29056
R_COMMON = 29055                          # SESSION-PREFIX / PROMPT-DIVERGE at=29055
R_CLOSE_POS = 28992                       # MATCH-CENSUS-DEEP reached=28992


def _ids(n, last_msg, gen=GEN, base=1000, sep=True):
    ids = [base + (i % 50_000) for i in range(n)]
    ids[0] = IM
    ids[last_msg] = IM
    ids[n - gen] = IM
    if sep:
        for at in (last_msg, n - gen):
            ids[at - 2], ids[at - 1] = END, NL
    return ids


R_SRC = _ids(R_N, R_LAST_MSG)


@pytest.fixture(autouse=True)
def _ledger(monkeypatch):
    # S62 SYS-GRID tracks of the first message would join desc.targets; pinned elsewhere
    monkeypatch.setenv("FLLIPER_PDFLIP_SYS_GRID_ANCHOR", "0")
    twa._reset_for_test()
    yield
    twa._reset_for_test()


# -- A. geometry ---------------------------------------------------------------------
def test_the_close_position_is_the_measured_fork_page():
    assert ta.close_fork_pos(R_SRC, IM, R_PAGE) == R_CLOSE_POS
    assert ta.anchor_pos(R_SRC, IM, R_PAGE) == R_TURN_POS        # the metal's turn track
    # the side request shares ids[:e], e = i - 2 (its <|im_end|>): claim floor_page(e - 1)
    assert R_COMMON == R_LAST_MSG - 2
    assert twa.boundary(R_COMMON, R_PAGE) == R_CLOSE_POS
    assert R_TURN_POS > R_COMMON - 1, "the premise: the turn anchor lies past the fork"
    assert ta.close_fork_pos(R_SRC, None, R_PAGE) is None


def test_the_separator_is_verified_not_assumed():
    assert ta.close_fork_pos(_ids(R_N, R_LAST_MSG, sep=False), IM, R_PAGE) is None
    bad = list(R_SRC)
    bad[R_LAST_MSG - 2] = 7                # the last message is not opened behind <|im_end|>\n
    assert ta.close_fork_pos(bad, IM, R_PAGE) is None
    bad = list(R_SRC)
    bad[R_N - GEN - 1] = 9                 # the generation prompt is opened differently
    assert ta.close_fork_pos(bad, IM, R_PAGE) is None
    # a last message without content (g - 2 <= i) carries no close fork
    tiny = _ids(40, 33)
    assert ta.close_fork_pos(tiny, IM, 1) is None


@pytest.mark.parametrize("i", list(range(29056 - 3, 29056 + 2 * R_PAGE + 1)))
def test_the_track_is_planned_exactly_where_the_turn_anchor_misses(i):
    """The turn anchor floor_page(i) serves a close-fork sibling iff it is <= e - 1 =
    i - 3; the close track is planned in the complement only (3 of 64 on page 64)."""
    ids = _ids(R_N, i)
    alloc = _Alloc([3, 4, 5, 6])
    req = _sreq(ids)
    _plan(req, alloc, R_PREFIX, R_STEP_END, R_STEP_END, chunk=64, page=R_PAGE)
    planned = getattr(req, CLOSE_PENDING, None) is not None
    turn_serves = (i // R_PAGE) * R_PAGE <= i - 3
    assert planned == (not turn_serves)
    assert planned == (i % R_PAGE in (0, 1, 2))
    if planned:
        assert getattr(req, CLOSE_PENDING)[0] == ((i - 3) // R_PAGE) * R_PAGE


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


def _sreq(ids, rid="pdflip-50-825", cpl=0):
    return SimpleNamespace(rid=rid, origin_input_ids=array("q", ids), output_ids=[],
                           cache_protected_len=cpl)


def _plan(req, alloc, prefix, end, main, tok=IM, chunk=1, page=1, fork=0, desc=None,
          twin_bounds=()):
    batch = SimpleNamespace(reqs=[req], req_to_token_pool=SimpleNamespace(mamba_allocator=alloc))
    return ta.note_step(batch=batch, desc=desc, req=req, row=0, prefix=prefix, end=end,
                        track_mask=True, main_track=main, chunk=chunk, page=page, tok=tok,
                        twin_bounds=twin_bounds, fork_told=fork)


def test_the_rc27_step_plans_the_turn_and_the_close_track():
    alloc = _Alloc([7, 8, 9])
    req = _sreq(R_SRC)
    desc = _plan(req, alloc, R_PREFIX, R_STEP_END, R_STEP_END, chunk=64, page=R_PAGE)
    assert desc is not None and sorted(desc.targets) == [R_CLOSE_POS, R_TURN_POS]
    assert getattr(req, TURN_PENDING)[0] == R_TURN_POS
    t, slot, d, s0, s1 = getattr(req, CLOSE_PENDING)
    assert (t, d, s0, s1) == (R_CLOSE_POS, desc, R_PREFIX, R_STEP_END)
    assert int(slot) != int(getattr(req, TURN_PENDING)[1])
    assert desc.prefixes == [R_PREFIX, R_PREFIX]
    assert ta.has_pending(req)
    kinds = ta.pop_all_pending(req)
    # ascending by position: the close node (28992) goes in below the turn node (29056)
    assert [(k, int(p[0])) for k, p in kinds] == [("close", R_CLOSE_POS), ("turn", R_TURN_POS)]
    assert getattr(req, CLOSE_PENDING, None) is None and not ta.has_pending(req)


def test_turn_anchor_unarmed_plans_nothing():
    alloc = _Alloc([7])
    req = _sreq(R_SRC)
    assert _plan(req, alloc, R_PREFIX, R_STEP_END, R_STEP_END, tok=None, chunk=64,
                 page=R_PAGE) is None
    assert getattr(req, CLOSE_PENDING, None) is None and alloc.free_ids == [7]


def test_no_slot_no_plan_no_promise(monkeypatch):
    monkeypatch.setattr(twa, "_ARMED", True)
    req = _sreq(R_SRC)
    _plan(req, _Alloc([4]), R_PREFIX, R_STEP_END, R_STEP_END, chunk=64, page=R_PAGE)
    # the turn track took the one slot; the close track had none
    assert getattr(req, TURN_PENDING) is not None
    assert getattr(req, CLOSE_PENDING, None) is None
    assert twa.status(req.rid, R_CLOSE_POS) is None, "a promise without a slot holds a twin for nothing"


@pytest.mark.parametrize("why,kw", [
    ("decode step (output exists)", dict(out=[1])),
    ("protected prefix at the position", dict(cpl=R_CLOSE_POS)),
    ("step starts at the position", dict(prefix=R_CLOSE_POS)),
    ("own track at the position", dict(main=R_CLOSE_POS)),
    ("step ends at the position", dict(end=R_CLOSE_POS)),
    ("track mask off", dict(mask=False)),
])
def test_no_close_track(why, kw):
    alloc = _Alloc([7, 8, 9])
    req = _sreq(R_SRC, cpl=kw.get("cpl", 0))
    req.output_ids = kw.get("out", [])
    batch = SimpleNamespace(reqs=[req], req_to_token_pool=SimpleNamespace(mamba_allocator=alloc))
    ta.note_step(batch=batch, desc=None, req=req, row=0, prefix=kw.get("prefix", R_PREFIX),
                 end=kw.get("end", R_STEP_END), track_mask=kw.get("mask", True),
                 main_track=kw.get("main", R_STEP_END), chunk=64, page=R_PAGE, tok=IM)
    assert getattr(req, CLOSE_PENDING, None) is None, why


def test_the_close_track_runs_where_the_turn_position_is_in_a_later_step():
    """A chunk ending between the close fork and the turn position: the close track
    goes into THIS step (main track past it), the turn track into the next one."""
    alloc = _Alloc([7, 8])
    req = _sreq(R_SRC)
    desc = _plan(req, alloc, R_PREFIX, 29024, 29024, chunk=64, page=R_PAGE)
    assert desc.targets == [R_CLOSE_POS] and getattr(req, TURN_PENDING, None) is None


def test_a_close_position_the_fork_track_writes_is_one_slot():
    alloc = _Alloc([11, 12, 13])
    req = _sreq(R_SRC)
    desc = _plan(req, alloc, R_PREFIX, R_STEP_END, R_STEP_END, chunk=64, page=R_PAGE,
                 fork=R_CLOSE_POS + 10)
    assert sorted(desc.targets) == [R_CLOSE_POS, R_TURN_POS]
    assert getattr(req, FORK_PENDING)[0] == R_CLOSE_POS
    assert getattr(req, CLOSE_PENDING, None) is None and alloc.free_ids == [13]


def test_a_stale_close_plan_is_freed_and_its_promise_declined(monkeypatch):
    monkeypatch.setattr(twa, "_ARMED", True)
    alloc = _Alloc([11, 12, 13, 14])
    req = _sreq(R_SRC)
    _plan(req, alloc, R_PREFIX, R_STEP_END, R_STEP_END, chunk=64, page=R_PAGE)
    assert twa.status(req.rid, R_CLOSE_POS) == "planned"
    close_slot = int(getattr(req, CLOSE_PENDING)[1])
    req.output_ids = [1]
    _plan(req, alloc, R_N, R_N + 1, R_N, chunk=64, page=R_PAGE)   # decode: stale freed
    assert close_slot in alloc.freed and getattr(req, CLOSE_PENDING, None) is None
    assert twa.status(req.rid, R_CLOSE_POS) == "declined"


def test_a_skipped_insert_declines_the_promise(monkeypatch):
    """Haenger direction: an unmarked plan (graph replay) writes nothing -- the ledger
    must say declined, or a twin waits on it until its Frist."""
    monkeypatch.setattr(twa, "_ARMED", True)
    req = _sreq(R_SRC)
    _plan(req, _Alloc([5, 6]), R_PREFIX, R_STEP_END, R_STEP_END, chunk=64, page=R_PAGE)
    kind, pend = ta.pop_all_pending(req)[0]
    ok, why = ta.insert_verdict(pend)
    assert (kind, ok) == ("close", False) and why.startswith("unmarked")
    ta.note_skip(req, why, pend[0], kind=kind)
    assert twa.status(req.rid, R_CLOSE_POS) == "declined"


# -- C. the promise ------------------------------------------------------------------
def test_a_twin_of_50_825_is_held_for_the_close_track(monkeypatch):
    """Had 51-826 reached P while 50-825's step was still planned-not-past: S52's ledger
    sees the close track as the source's own track at B -- not TWIN-NO-COMMIT."""
    from flliper.srt.pdflip import p_twin_defer as tw

    monkeypatch.setattr(twa, "_ARMED", True)
    src = _sreq(R_SRC)
    src.finished = lambda: False
    src.fill_ids = list(range(R_STEP_END))
    st = SimpleNamespace(page=R_PAGE, chunk=16384, s0={src.rid: R_PREFIX})
    _plan(src, _Alloc([5, 6]), R_PREFIX, R_STEP_END, R_STEP_END, chunk=64, page=R_PAGE)
    assert tw._promise(st, src, R_COMMON) in ((tw.PROMISE_BOUNDARY, R_CLOSE_POS),
                                             (tw.PROMISE_TRACK, R_CLOSE_POS))


# -- D. PP3 rank agreement -----------------------------------------------------------
def test_three_pp_stages_plan_the_same_positions():
    """R11 / #1400: no rank-local verdict -- each PP stage plans its own copy of the
    request from the same ids and the same forwarded step and reaches the same
    positions; only the slot ids are stage-local."""
    plans = []
    for stage, free in enumerate(([3, 4], [17, 18], [40, 41])):
        req = _sreq(R_SRC)
        desc = _plan(req, _Alloc(free), R_PREFIX, R_STEP_END, R_STEP_END, chunk=64, page=R_PAGE)
        plans.append((sorted(desc.targets), getattr(req, CLOSE_PENDING)[0],
                      getattr(req, CLOSE_PENDING)[3:]))
    assert plans[0] == plans[1] == plans[2] == (
        [R_CLOSE_POS, R_TURN_POS], R_CLOSE_POS, (R_PREFIX, R_STEP_END))


# -- E. the real tree ----------------------------------------------------------------
MAMBA_SLOTS = 20
KV_SIZE = 512
NUM_LAYERS = 8
FULL_LAYER_IDS = (3, 7)
MAMBA_LAYER_IDS = [i for i in range(NUM_LAYERS) if i not in FULL_LAYER_IDS]

N = 120
TURN = 40
G = N - GEN                                 # 115
E = TURN - 2                                # the <|im_end|> of the message before the last
SRC = _ids(N, TURN)
SIDE = SRC[:E] + [NL] + [900_000 + i for i in range(30)]    # continues that message
NEXT_TURN = SRC[: TURN + 1] + [800_000 + i for i in range(30)]
GEN_SIB = SRC[: G + 1] + [700_000 + i for i in range(30)]
S_CLOSE, S_TURN, S_GEN, S_END = 1.25, 2.25, 4.5, 7.5


@pytest.fixture
def group_p(monkeypatch):
    from flliper.srt.mem_cache import unified_radix_cache as urc

    monkeypatch.setenv("FLLIPER_PDFLIP_GROUP", "P")
    monkeypatch.setenv("FLLIPER_PDFLIP_BIGRAM_ANCHOR_EXACT", "1")
    monkeypatch.setenv("FLLIPER_PDFLIP_GEN_FORK_ANCHOR", "1")
    monkeypatch.setenv("FLLIPER_PDFLIP_SYS_GRID_ANCHOR", "0")
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
    req = Req(rid="pdflip-50-825", origin_input_text="", origin_input_ids=array("q", ids),
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


def _one_forward(fx, close=True):
    """50-825's shape at page 1: ONE step [0, N) with turn, gen and close tracks, END
    at N-1; the forward leaves each planned slot holding the state after its position.
    ``close=False`` emulates rc28 (the planner without the close track)."""
    req = _req(fx, SRC)
    _write_kv(fx, req, N)
    batch = SimpleNamespace(reqs=[req], req_to_token_pool=fx.pool)
    orig = ta._note_close_step
    if not close:
        ta._note_close_step = lambda batch, desc, *a, **k: desc
    try:
        desc = ta.note_step(batch=batch, desc=None, req=req, row=0, prefix=0, end=N,
                            track_mask=True, main_track=N - 1, chunk=1, page=1, tok=IM)
    finally:
        ta._note_close_step = orig
    assert desc is not None
    for attr, val in ((TURN_PENDING, S_TURN), (GEN_PENDING, S_GEN), (CLOSE_PENDING, S_CLOSE)):
        if getattr(req, attr, None) is not None:
            _state(fx, getattr(req, attr)[1], val)
    assert (getattr(req, CLOSE_PENDING, None) is not None) == close
    desc.done.add("gdn")
    _state(fx, req.mamba_pool_idx, S_END)
    req.kv_committed_len = N
    req.kv_allocated_len = N
    fx.cache.cache_finished_req(req, is_insert=True)
    return req


@pytest.mark.parametrize("bigram", [False, True], ids=["unigram", "bigram"])
def test_the_side_request_resumes_at_the_close_fork(group_p, bigram):
    """RED on rc28: the side request finds no anchor at or below e - 1."""
    fx = _fixture(bigram)
    _one_forward(fx)
    assert _claim(fx, SIDE) == (E - 1, pytest.approx(S_CLOSE))
    fx0 = _fixture(bigram)
    _one_forward(fx0, close=False)
    assert _claim(fx0, SIDE)[0] < E - 1, "the gap itself, rc28"


@pytest.mark.parametrize("bigram", [False, True], ids=["unigram", "bigram"])
def test_end_turn_and_gen_anchors_are_unchanged(group_p, bigram):
    later = SRC + [5, 6, 7]
    fx = _fixture(bigram)
    _one_forward(fx)
    fx0 = _fixture(bigram)
    _one_forward(fx0, close=False)
    assert _claim(fx, later) == _claim(fx0, later) == (N - (1 if bigram else 0),
                                                       pytest.approx(S_END))
    assert _claim(fx, NEXT_TURN) == _claim(fx0, NEXT_TURN) == (TURN, pytest.approx(S_TURN))
    assert _claim(fx, GEN_SIB) == _claim(fx0, GEN_SIB) == (G, pytest.approx(S_GEN))


def _hand_plan(fx, end):
    req = _req(fx, SRC)
    _write_kv(fx, req, end)
    slot = fx.pool.mamba_allocator.alloc(1)
    _state(fx, slot, S_CLOSE)
    desc = ta.TurnTracks(1)
    desc.add(0, slot, E - 1, 0, end)
    return req, slot, desc


def test_the_tree_consumes_a_close_plan(group_p):
    fx = _fixture(False)
    free_before = fx.pool.mamba_allocator.available_size()
    req, slot, desc = _hand_plan(fx, N)
    desc.done.add("gdn")
    setattr(req, CLOSE_PENDING, (E - 1, slot, desc, 0, N))
    _state(fx, req.mamba_pool_idx, S_END)
    req.kv_committed_len = N
    req.kv_allocated_len = N
    fx.cache.cache_finished_req(req, is_insert=True)
    assert _claim(fx, SIDE) == (E - 1, pytest.approx(S_CLOSE))
    assert getattr(req, CLOSE_PENDING, None) is None
    assert fx.pool.mamba_allocator.available_size() == free_before - 2


def test_an_inner_chunk_consumes_a_close_plan(group_p):
    fx = _fixture(False)
    end = TURN + 3
    req, slot, desc = _hand_plan(fx, end)
    desc.done.add("gdn")
    setattr(req, CLOSE_PENDING, (E - 1, slot, desc, 0, end))
    _state(fx, req.mamba_pool_idx, S_END)
    fx.cache.cache_unfinished_req(req, chunked=True)
    assert req.cache_protected_len == end
    assert getattr(req, CLOSE_PENDING, None) is None
    assert _claim(fx, SIDE) == (E - 1, pytest.approx(S_CLOSE))


def test_an_unmarked_close_plan_inserts_nothing_and_frees_its_slot(group_p):
    fx = _fixture(False)
    free_before = fx.pool.mamba_allocator.available_size()
    req, slot, desc = _hand_plan(fx, N)          # no GDN mark: a graph replay
    setattr(req, CLOSE_PENDING, (E - 1, slot, desc, 0, N))
    _state(fx, req.mamba_pool_idx, S_END)
    req.kv_committed_len = N
    req.kv_allocated_len = N
    fx.cache.cache_finished_req(req, is_insert=True)
    assert _claim(fx, SIDE)[1] != pytest.approx(S_CLOSE)
    assert fx.pool.mamba_allocator.available_size() == free_before - 1


# -- F. wiring -----------------------------------------------------------------------
def test_the_radix_cache_asks_for_a_close_plan_at_both_inserts():
    from flliper.srt.mem_cache import unified_radix_cache as urc

    src = inspect.getsource(urc.UnifiedRadixCache)
    assert src.count("_CLOSE_PENDING_ATTR") == 2
    assert urc._CLOSE_PENDING_ATTR == ta.CLOSE_PENDING_ATTR == CLOSE_PENDING


def test_no_own_switch_armed_with_the_turn_anchor():
    """User order 20.10.: kein Schalter, Default AN -- the close track runs wherever the
    turn token is set; no FLLIPER_* switch reads gate it."""
    src = inspect.getsource(ta._note_close_step) + inspect.getsource(ta.close_fork_pos)
    assert "envs." not in src and "environ" not in src
    body = inspect.getsource(ta.note_step)
    assert "_note_close_step" in body


def test_the_close_position_reads_no_rank_local_state():
    """R11 / #1004: the position is a pure function of the ids, the env token and the
    page -- no pp rank, no tree, no store, no clock in its derivation."""
    src = inspect.getsource(ta.close_fork_pos) + inspect.getsource(ta.req_close_pos)
    tree = ast.parse(src)
    names = {n.id for n in ast.walk(tree) if isinstance(n, ast.Name)}
    attrs = {n.attr for n in ast.walk(tree) if isinstance(n, ast.Attribute)}
    attrs |= {n.value for n in ast.walk(tree) if isinstance(n, ast.Constant) and isinstance(n.value, str)}
    for bad in ("pp_rank", "tree_cache", "cache_controller", "time", "random", "torch",
                "cache_protected_len", "prefix_indices"):
        assert bad not in names and bad not in attrs, bad


@pytest.mark.parametrize("env", ["FLLIPER_PDFLIP_GEN_FORK_ANCHOR", "FLLIPER_PDFLIP_TWIN_ANCHOR"])
def test_no_other_switch_gates_the_close_track(monkeypatch, env):
    """Mutant M14: the close track must not hide behind another track's switch."""
    monkeypatch.setenv(env, "0")
    monkeypatch.setattr(twa, "_ARMED", None)
    req = _sreq(R_SRC)
    desc = _plan(req, _Alloc([7, 8]), R_PREFIX, R_STEP_END, R_STEP_END, chunk=64, page=R_PAGE)
    assert getattr(req, CLOSE_PENDING)[0] == R_CLOSE_POS and R_CLOSE_POS in desc.targets
