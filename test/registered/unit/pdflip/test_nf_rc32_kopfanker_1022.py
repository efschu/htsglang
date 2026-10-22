"""NF rc32 KOPF-ANKER (22.10.2026): group P tracks the recurrent state at every '</system-reminder' close of the
prompt's FIRST user message (the Claude-Code head), so a later prompt that branches there -- a compaction, a sibling
agent with the same head -- resumes where the store's KV ends instead of at the previous chunk anchor.

Metal (NF rc32 ded00bd5a7, boot ..._1022_030836):
  P  04:45:17 PP0 'PDFLIP-CLAIM-SHORT rid=pdflip-14-469 pp=0 asked=476 kv=281 claimed=172 lost=109 ... by=mamba
               tokens=11008'
  P  04:45:18 PP0 'S80 STORE-HOLE rid=pdflip-14-469 have=11008 kv_end=17984 ... class=anchor'
  fr 04:45:16 'PDFLIP PREFIX-DIVERGE rid=pdflip-14-469 verdict=store_short prompt=30480 credit=11008 ...
               best_prev=pdflip-14-383 ... common=18040 common_page=17984 lost=6976'
  fr 04:45:16 'PREFIX-DIVERGE-TEXT ... seg=1 seg_role=user ... before="...reported back.\\n</system-reminder"
               prev_after="><system-reminder>\\nAttribution ..." cur_after=">\\nThis session is being continued ..."'
  P  03:46:33 PP0 '#969 EXTENT n=9 fwd=1 reqs=[('pdflip-8-255', 10816, 19792, 10816, 8976)]' -- the step that wrote
               the head of this prompt family had tracks at 11008 (fork), 19456 (turn), 19776 (END) and none between.
The head closes of pdflip-8-255's prompt (rebuilt from its client transcript with the NF tokenizer, aligned on the
front's 18040 and checked against the front's 11045 and P's 'boundary=19471'): b = 17503, 18040, 18175, 19468.

Front PROMPT-DIVERGE 'moved' (30 of 472 prompts, held_back=0): 9 branch exactly at a head close (17936 x8,
18040 x1), 13 inside the text right after one, on the close's page (17977 x2, 18098 x9, 18102 x2); family A's
closes 17399 / 17936 / 18071 (family B = A + 104 tokens behind the memory edit at 11045).

Pinned (hermetic, CPU):
  A. geometry: the close ids (NF and 27B tokenizer), positions = floor_page(b - 1) of every close inside the first
     user message only, every measured head branch finds its anchor on its claim page, page-aligned and
     page-straddling closes, cap;
  B. the planner on pdflip-8-255's real step: three head tracks, 14-469 then resumes at 17984 (lost 0, was 6976);
     slots, stale plans, protected prefix, decode, group D plans none;
  C. the REAL UnifiedRadixCache (unigram and exact bigram): a variant branching at a close resumes there with the
     head track's state; END and turn anchor unchanged; finish and chunk insert consume the plans; unmarked frees;
  D. R11: the positions are a pure function of the ids (every P stage plans the same ones), a stage without slots
     plans a subset, never another position; on group D (rank-local main track at this base) no head plan at all;
     no switch.
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

HEAD_PENDING = "_pdflip_head_pending"     # turn_anchor.HEAD_PENDING_ATTR, by name
TURN_PENDING = "_pdflip_turn_pending"
IM = 248045                               # <|im_start|> on NF and 27B
GEN = 5
CLOSE = (510, 8678, 12, 75240)            # '</' 'system' '-' 'reminder' (NF + Qwen3.8-27B tokenizer, 22.10.)
OPEN = (27, 8678, 12, 75240)              # '<' 'system' '-' 'reminder'
PAGE = 64


def _fill(n, base=100000):
    return [base + (i % 40000) for i in range(n)]


def _put_close(ids, b):
    """A '</system-reminder' close whose '>' token sits at index ``b`` (the four ids end at b - 1)."""
    ids[b - len(CLOSE):b] = list(CLOSE)
    ids[b] = 29                           # '>' (merged '><' 1721 when the next block opens -- any id)


# ------------------------------------------------- pdflip-8-255 (family B) in tokens, page 64, FLA 64
N_8_255 = 19792
U_B = 3626                                # the first user message's <|im_start|>
CLOSES_B = (17503, 18040, 18175, 19468)   # instructions / session context / attribution / SubagentHandback
LAST_MSG = 19471                          # P 'TURN-ANCHOR TRACK ... turn=19456 boundary=19471'
STEP = (10816, 19792, 19776)              # '#969 EXTENT ... (pdflip-8-255, 10816, 19792 ...)', main = END 19776
B_14_469 = 17984                          # store KV end of pdflip-14-469 (S80 kv_end), front common_page
FORK_8_255 = 11008                        # 'FORK-ANCHOR TRACK ... at=11008'


def _src_b(closes=CLOSES_B, n=N_8_255, u=U_B, last=LAST_MSG):
    ids = _fill(n)
    ids[0] = IM
    ids[u] = IM
    for b in closes:
        _put_close(ids, b)
    ids[last] = IM
    ids[n - GEN] = IM
    return ids


# family A (before the memory edit at 11045: B - 104 behind it) -- three closes before the task text
CLOSES_A = (17399, 17936, 18071)
# every front PROMPT-DIVERGE 'moved' at the head (rid, at): 9 exactly at a close, 13 on a close's page
HEAD_BRANCHES_A = [("pdflip-0-5", 17936), ("pdflip-0-21", 17936), ("pdflip-2-81", 17936), ("pdflip-4-131", 17936),
                   ("pdflip-4-140", 17936), ("pdflip-4-149", 17936), ("pdflip-4-157", 17936), ("pdflip-6-163", 17936),
                   ("pdflip-4-159", 17977), ("pdflip-6-173", 17977),
                   ("pdflip-0-6", 18098), ("pdflip-0-9", 18098), ("pdflip-0-28", 18098), ("pdflip-2-38", 18098),
                   ("pdflip-2-68", 18098), ("pdflip-2-77", 18098), ("pdflip-4-87", 18098), ("pdflip-4-90", 18098),
                   ("pdflip-4-109", 18098), ("pdflip-0-2", 18102), ("pdflip-2-31", 18102)]


@pytest.fixture(autouse=True)
def _clean(monkeypatch):
    monkeypatch.delenv("FLLIPER_PDFLIP_GROUP", raising=False)
    twa._reset_for_test()
    yield
    twa._reset_for_test()


# -- A. geometry ----------------------------------------------------------------------------------------
def test_the_close_ids_are_pinned_without_a_switch():
    assert tuple(ta.HEAD_CLOSE_IDS) == CLOSE
    assert ta.HEAD_MAX >= 4
    names = [n for n in dir(envs) if "HEAD_ANCHOR" in n or "KOPF" in n]
    assert names == [], "no switch: the head track is armed wherever the turn anchor is (order 22.10.)"


def test_8_255_head_positions_are_the_closes_floor_page():
    pts = ta.head_positions(_src_b(), IM, PAGE)
    assert pts == [17472, 17984, 18112, 19456]
    assert all(p == ((b - 1) // PAGE) * PAGE for p, b in zip(pts, CLOSES_B))


def test_14_469_claim_page_holds_a_head_anchor():
    """front 'common=18040 common_page=17984': the branch's claim ends exactly on the context close's anchor."""
    assert B_14_469 in ta.head_positions(_src_b(), IM, PAGE)


@pytest.mark.parametrize("rid,at", HEAD_BRANCHES_A, ids=[r for r, _ in HEAD_BRANCHES_A])
def test_every_measured_head_branch_finds_its_anchor_on_its_claim_page(rid, at):
    n = 33140
    ids = _src_b(closes=CLOSES_A, n=n, last=30000)
    pts = ta.head_positions(ids, IM, PAGE)
    claim = ((at - 1) // PAGE) * PAGE         # a reader sharing ids[:at] claims floor_page(at - 1)
    assert claim in pts, (rid, at, claim, pts)


def test_a_page_aligned_close_anchors_one_page_below():
    """b % 64 == 0: a reader sharing ids[:b] claims floor_page(b - 1) = b - 64; an anchor AT b is past its claim."""
    ids = _src_b(closes=(12800, 17503), n=19000, last=18000)
    assert ta.head_positions(ids, IM, PAGE)[0] == 12800 - PAGE


def test_a_close_straddling_a_page_anchors_the_page_of_its_last_id():
    """The four ids start in page k-1, the '>' sits in page k: the shared prefix reaches into page k."""
    b = 12802
    ids = _src_b(closes=(b,), n=19000, last=18000)
    assert ta.head_positions(ids, IM, PAGE) == [12800]


def test_only_the_first_user_message_counts():
    """A close in a later message (the CLOSE-FORK shape '</system-reminder><|im_end|>' of a side request) is the
    turn/close tracks' business, not the head's; an opening tag is no close."""
    ids = _src_b(closes=CLOSES_B[:2])                   # two closes in the head, below the cap
    _put_close(ids, LAST_MSG + 200)                     # inside the last message
    ids[15000:15004] = list(OPEN)                       # an opening tag in the head
    assert ta.head_positions(ids, IM, PAGE) == [17472, 17984]


def test_the_count_is_capped_lowest_first():
    closes = tuple(5000 + 300 * k for k in range(ta.HEAD_MAX + 5))
    ids = _src_b(closes=closes, n=19792)
    pts = ta.head_positions(ids, IM, PAGE)
    assert len(pts) == ta.HEAD_MAX
    assert pts == sorted({((b - 1) // PAGE) * PAGE for b in closes})[: ta.HEAD_MAX]


@pytest.mark.parametrize("ids", [
    [5, 6, 7, IM, 8],                   # no turn-start at 0
    [IM] + [5] * 900,                   # one message only
    None,
])
def test_no_first_user_message_no_positions(ids):
    assert ta.head_positions(ids, IM, PAGE) == []


# -- B. the planner -------------------------------------------------------------------------------------
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


def _sreq(ids, rid="pdflip-8-255", cpl=STEP[0]):
    return SimpleNamespace(rid=rid, origin_input_ids=array("q", ids), output_ids=[], cache_protected_len=cpl)


def _plan(req, alloc, prefix=STEP[0], end=STEP[1], main=STEP[2], tok=IM, chunk=64, page=PAGE):
    batch = SimpleNamespace(reqs=[req], req_to_token_pool=SimpleNamespace(mamba_allocator=alloc))
    return ta.note_step(batch=batch, desc=None, req=req, row=0, prefix=prefix, end=end, track_mask=True,
                        main_track=main, chunk=chunk, page=page, tok=tok)


def test_8_255_step_plans_three_head_tracks():
    alloc = _Alloc(range(100, 110))
    req = _sreq(_src_b())
    desc = _plan(req, alloc)
    heads = getattr(req, HEAD_PENDING)
    assert [p[0] for p in heads] == [17472, 17984, 18112]          # 19456 is the turn track's position
    assert getattr(req, TURN_PENDING)[0] == 19456
    assert set(desc.targets) >= {17472, 17984, 18112, 19456} and set(desc.prefixes) == {STEP[0]}
    assert len(desc.targets) == len(set(desc.targets)), "one slot per position"
    kinds = [k for k, _ in ta.pop_all_pending(req)]
    assert kinds.count("head") == 3 and not ta.has_pending(req)


def test_14_469_resumes_at_its_store_kv_end():
    """RED on the base: 8-255's anchors under 14-469's claim were 10816 (prefix) and 11008 (fork) -> 6976 lost."""
    req = _sreq(_src_b())
    _plan(req, _Alloc(range(100, 110)))
    anchors = {STEP[0], FORK_8_255, STEP[2]} | {p[0] for _, p in ta.pop_all_pending(req)}
    deepest = max(a for a in anchors if a <= B_14_469)
    assert deepest == B_14_469 and B_14_469 - deepest == 0           # was 17984 - 11008 = 6976


def test_no_slot_no_plan_and_never_another_position(monkeypatch):
    monkeypatch.setattr(twa, "_ARMED", True)
    req = _sreq(_src_b())
    _plan(req, _Alloc([7, 8]))               # turn takes one, the lowest head close the other
    heads = [p[0] for p in getattr(req, HEAD_PENDING, None) or ()]
    assert set(heads) <= {17472, 17984, 18112} and len(heads) == 1
    for t in {17472, 17984, 18112} - set(heads):
        assert twa.status(req.rid, t) is None, "a promise without a slot"


def test_a_stale_head_plan_is_freed_and_declined(monkeypatch):
    monkeypatch.setattr(twa, "_ARMED", True)
    alloc = _Alloc(range(11, 20))
    req = _sreq(_src_b())
    _plan(req, alloc)
    planned = [int(p[1]) for p in getattr(req, HEAD_PENDING)]
    req.output_ids = [1]
    _plan(req, alloc, prefix=N_8_255, end=N_8_255 + 1, main=N_8_255)   # a decode step: stale plans freed
    assert set(planned) <= set(alloc.freed)
    assert getattr(req, HEAD_PENDING, None) is None
    assert all(twa.status(req.rid, t) == "declined" for t in (17472, 17984, 18112))


@pytest.mark.parametrize("why,kw", [
    ("decode step", dict(out=[1])),
    ("protected prefix past the head", dict(cpl=18112, prefix=18112)),
    ("step past the head", dict(prefix=19456, end=N_8_255, main=STEP[2], cpl=19456)),
    ("track mask off", dict(mask=False)),
    ("off the FLA grid", dict(prefix=10817, cpl=10817)),
])
def test_no_head_track(why, kw):
    alloc = _Alloc(range(7, 17))
    req = _sreq(_src_b(), cpl=kw.get("cpl", STEP[0]))
    req.output_ids = kw.get("out", [])
    batch = SimpleNamespace(reqs=[req], req_to_token_pool=SimpleNamespace(mamba_allocator=alloc))
    ta.note_step(batch=batch, desc=None, req=req, row=0, prefix=kw.get("prefix", STEP[0]),
                 end=kw.get("end", STEP[1]), track_mask=kw.get("mask", True), main_track=kw.get("main", STEP[2]),
                 chunk=64, page=PAGE, tok=IM)
    assert not getattr(req, HEAD_PENDING, None), why


def test_14_469_own_steps_plan_the_close_below_its_fork_cut():
    """14-469 on the base ran [11008, 17984) (P-FORK-CUT at the told fork 17984, its chunk-end anchor) and
    [17984, 30480): the head track takes the instructions close 17472 below the cut, the cut itself is the step's
    own track (no second slot), the summary carries no close."""
    ids = _src_b(closes=CLOSES_B[:2], n=30480, last=26197)
    req = _sreq(ids, rid="pdflip-14-469", cpl=11008)
    _plan(req, _Alloc(range(30, 40)), prefix=11008, end=17984, main=17984)
    assert [p[0] for p in getattr(req, HEAD_PENDING)] == [17472]


# -- C. the real tree -----------------------------------------------------------------------------------
MAMBA_SLOTS = 24
KV_SIZE = 1024
NUM_LAYERS = 8
FULL_LAYER_IDS = (3, 7)
MAMBA_LAYER_IDS = [i for i in range(NUM_LAYERS) if i not in FULL_LAYER_IDS]


def _small_src():
    """system (12) | user: reminder, reminder, task text | assistant | user (last) | generation prompt."""
    ids = [IM] + [700000 + i for i in range(11)]
    u = len(ids)
    ids += [IM, 701, 702] + [710000 + i for i in range(14)] + list(CLOSE) + [1721]
    c1 = len(ids) - 1
    ids += [720000 + i for i in range(13)] + list(CLOSE) + [29]
    c2 = len(ids) - 1
    ids += [730000 + i for i in range(12)] + [248046, 198]
    ids += [IM] + [740000 + i for i in range(10)] + [248046, 198]
    turn = len(ids)
    ids += [IM] + [750000 + i for i in range(9)]
    ids += [IM, 1, 2, 3, 4]
    return ids, u, c1, c2, turn


SMALL, U, C1, C2, TURN = _small_src()
N = len(SMALL)
COMPACT = SMALL[:C2] + [990000 + i for i in range(30)]   # leaves at the '>' token: '>\nThis session is ...'


@pytest.fixture
def group_p(monkeypatch):
    from flliper.srt.mem_cache import unified_radix_cache as urc

    monkeypatch.setenv("FLLIPER_PDFLIP_GROUP", "P")
    monkeypatch.setenv("FLLIPER_PDFLIP_BIGRAM_ANCHOR_EXACT", "1")
    monkeypatch.setenv("FLLIPER_PDFLIP_SYS_GRID_ANCHOR", "0")
    monkeypatch.setattr(urc, "_PDFLIP_END_ANCHOR", True)
    return monkeypatch


def _fixture(bigram):
    server_args = ServerArgs(model_path="dummy", page_size=1)
    server_args._mamba_cache_chunk_size = FLA_CHUNK_SIZE
    server_args.chunked_prefill_size = 16
    set_global_server_args_for_scheduler(server_args)
    with envs.FLLIPER_MAMBA_SSM_DTYPE.override("bfloat16"):
        shape = Mamba2StateShape.create(tp_world_size=1, intermediate_size=256, n_groups=1, num_heads=2,
                                        head_dim=16, state_size=16, conv_kernel=4)
        cache_params = Mamba2CacheParams(shape=shape, layers=MAMBA_LAYER_IDS)
    pool = HybridReqToTokenPool(
        size=10, mamba_size=MAMBA_SLOTS, mamba_spec_state_size=10, max_context_len=512, device="cpu",
        enable_memory_saver=False, cache_params=cache_params, mamba_layer_ids=MAMBA_LAYER_IDS,
        enable_mamba_extra_buffer=False, speculative_num_draft_tokens=3,
    )
    kv_pool = HybridLinearKVPool(
        size=KV_SIZE, dtype=torch.bfloat16, page_size=1, head_num=2, head_dim=64,
        full_attention_layer_ids=list(FULL_LAYER_IDS), device="cpu", enable_memory_saver=False,
        mamba_pool=pool.mamba_pool,
    )
    allocator = TokenToKVPoolAllocator(size=KV_SIZE, dtype=torch.bfloat16, device="cpu", kvcache=kv_pool,
                                       need_sort=False)
    params = CacheInitParams(
        req_to_token_pool=pool, token_to_kv_pool_allocator=allocator, page_size=1, disable=False,
        sliding_window_size=None, tree_components=(ComponentType.FULL, ComponentType.MAMBA),
        enable_mamba_extra_buffer=False, enable_kv_cache_events=False, eviction_policy="lru", is_eagle=bigram,
    )
    cache = UnifiedRadixCache(params=params)
    cache.cache_init_params = params
    return SimpleNamespace(cache=cache, allocator=allocator, pool=pool)


def _key(fx, ids):
    return RadixKey(array("q", ids), is_bigram=fx.cache.is_eagle)


def _req(fx, ids):
    req = Req(rid="pdflip-8-255", origin_input_text="", origin_input_ids=array("q", ids),
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
    claim = Req._compute_max_prefix_len(SimpleNamespace(return_logprob=False, logprob_start_len=0), len(ids))
    mr = fx.cache.match_prefix(MatchPrefixParams(key=_key(fx, ids[:claim])))
    slot = mr.last_device_node.component_data[ComponentType.MAMBA].value
    state = None if slot is None else float(fx.pool.mamba_pool.mamba_cache.temporal[:, slot].float().mean())
    return len(mr.device_indices), state


def _sv(t):
    return 1.0 + t / 64.0


def _one_forward(fx, chunked=False, mark=True):
    req = _req(fx, SMALL)
    end = TURN - 2 if chunked else N
    _write_kv(fx, req, end)
    batch = SimpleNamespace(reqs=[req], req_to_token_pool=fx.pool)
    desc = ta.note_step(batch=batch, desc=None, req=req, row=0, prefix=0, end=end, track_mask=True,
                        main_track=end - 1, chunk=1, page=1, tok=IM)
    assert desc is not None
    for pend in getattr(req, HEAD_PENDING, None) or ():
        _state(fx, pend[1], _sv(pend[0]))
    if getattr(req, TURN_PENDING, None) is not None:
        _state(fx, getattr(req, TURN_PENDING)[1], 0.5)
    if mark:
        desc.done.add("gdn")
    _state(fx, req.mamba_pool_idx, 9.0)
    if chunked:
        fx.cache.cache_unfinished_req(req, chunked=True)
        return req
    req.kv_committed_len = N
    req.kv_allocated_len = N
    fx.cache.cache_finished_req(req, is_insert=True)
    return req


@pytest.mark.parametrize("bigram", [False, True], ids=["unigram", "bigram"])
def test_the_compaction_resumes_at_the_context_close(group_p, bigram):
    """RED on the base: the compaction falls back to the root (no state inside the head)."""
    fx = _fixture(bigram)
    _one_forward(fx)
    assert _claim(fx, COMPACT) == (C2 - 1, pytest.approx(_sv(C2 - 1)))


@pytest.mark.parametrize("bigram", [False, True], ids=["unigram", "bigram"])
def test_end_and_turn_anchor_are_unchanged(group_p, bigram):
    later = SMALL + [5, 6, 7]
    nxt = SMALL[: TURN + 1] + [880000 + i for i in range(20)]
    fx = _fixture(bigram)
    _one_forward(fx)
    assert _claim(fx, later) == (N - (1 if bigram else 0), pytest.approx(9.0))
    assert _claim(fx, nxt) == (TURN, pytest.approx(0.5))


def test_an_inner_chunk_consumes_the_head_plans(group_p):
    fx = _fixture(False)
    req = _one_forward(fx, chunked=True)
    assert req.cache_protected_len == TURN - 2
    assert getattr(req, HEAD_PENDING, None) is None
    assert _claim(fx, COMPACT) == (C2 - 1, pytest.approx(_sv(C2 - 1)))


def test_a_finish_step_with_head_plans_only_inserts_them(group_p, monkeypatch):
    """The finish insert must ask for a pending head plan by itself: a last step whose turn/close/gen tracks plan
    nothing (their position not inside the step, or shared) still puts the head nodes in."""
    for name in ("_note_turn_twin_steps", "_note_close_step", "_note_gen_step"):
        monkeypatch.setattr(ta, name, lambda batch, desc, *a, **k: desc)
    fx = _fixture(False)
    req = _one_forward(fx)
    assert getattr(req, HEAD_PENDING, None) is None
    assert _claim(fx, COMPACT) == (C2 - 1, pytest.approx(_sv(C2 - 1)))


def test_an_unmarked_head_plan_inserts_nothing_and_frees_its_slots(group_p):
    fx = _fixture(False)
    free_before = fx.pool.mamba_allocator.available_size()
    _one_forward(fx, mark=False)
    assert _claim(fx, COMPACT)[0] < C1 - 1
    assert fx.pool.mamba_allocator.available_size() == free_before - 1   # the END slot only


# -- D. R11 / wiring ------------------------------------------------------------------------------------
def test_every_p_stage_plans_the_same_head_positions(monkeypatch):
    """Three PP stages: own allocator, own free list, own request object -- the same ids, the same positions."""
    got = []
    for stage, free in ((0, range(10, 30)), (1, range(200, 205)), (2, range(300, 340))):
        monkeypatch.setenv("FLLIPER_PDFLIP_PP_RANK", str(stage))
        req = _sreq(_src_b(), rid="pdflip-8-255")
        req.prefix_indices = list(range(stage))         # rank-local noise the positions must not read
        _plan(req, _Alloc(free))
        got.append([p[0] for p in getattr(req, HEAD_PENDING)])
    assert got[0] == got[1] == got[2] == [17472, 17984, 18112]


def test_a_stage_short_of_slots_plans_a_subset_never_another_position():
    full = _sreq(_src_b())
    _plan(full, _Alloc(range(10, 30)))
    short = _sreq(_src_b())
    _plan(short, _Alloc([1, 2, 3]))
    a = [p[0] for p in getattr(full, HEAD_PENDING)]
    b = [p[0] for p in getattr(short, HEAD_PENDING, None) or ()]
    # lowest first on every stage: a short stage's anchors are the full stage's lowest ones, so the anchors it has
    # are complete across the stages (a free-list-dependent choice would leave every anchor partial somewhere)
    assert b and b == a[: len(b)]


def test_group_d_plans_no_head_track_on_any_rank(monkeypatch):
    """At this base a D step's own track is rank-local (TP0's branching point replaces it, review A2 of
    nf-rc31-forkanchor): a head plan bounded by it would split the group. TP0 (main = branching 17920) and a worker
    (main = step end 19776) must agree -- here: neither plans one."""
    monkeypatch.setenv("FLLIPER_PDFLIP_GROUP", "D")
    monkeypatch.setenv("FLLIPER_PDFLIP_TURN_ANCHOR_TOKEN", str(IM))
    got = []
    for split_only, main in ((False, 17920), (True, STEP[2])):
        monkeypatch.setattr(ta, "d_split_only_rank", lambda v=split_only: v)
        req = _sreq(_src_b())
        _plan(req, _Alloc(range(50, 70)), main=main)
        got.append([p[0] for p in getattr(req, HEAD_PENDING, None) or ()])
        for _k, pend in ta.pop_all_pending(req):
            pass
    assert got[0] == got[1] == []


def test_the_positions_read_no_rank_local_state():
    """R11 / #1004: a pure function of the ids, the env token and the page."""
    src = "".join(inspect.getsource(f) for f in (ta.head_positions, ta.req_head_pos, ta.segment_end, ta._find))
    tree = ast.parse(src)
    names = {n.id for n in ast.walk(tree) if isinstance(n, ast.Name)}
    attrs = {n.attr for n in ast.walk(tree) if isinstance(n, ast.Attribute)}
    attrs |= {n.value for n in ast.walk(tree) if isinstance(n, ast.Constant) and isinstance(n.value, str)}
    for bad in ("pp_rank", "tree_cache", "cache_controller", "time", "random", "torch", "environ",
                "cache_protected_len", "prefix_indices", "output_ids", "fill_ids"):
        assert bad not in names and bad not in attrs, bad


def test_the_tree_insert_hooks_see_the_head_plan():
    from flliper.srt.mem_cache import unified_radix_cache as urc

    src = inspect.getsource(urc.UnifiedRadixCache.cache_finished_req) + inspect.getsource(
        urc.UnifiedRadixCache.cache_unfinished_req)
    assert src.count("_HEAD_PENDING_ATTR") == 2
    assert ta._label("head") == "HEAD-ANCHOR" and ta._kind_prefix("head") == "head_"
