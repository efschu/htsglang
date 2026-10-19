"""S62 (19.10.2026) 4b: SYS-GRID TRACK -- a tool-list variant finds an anchor at most one grid window
below its fork, and at a tool-GROUP boundary exactly.

Metal (NF rc15, boot ..._1019_071429, image 489909d138):
  fr 07:18:48 'PREFIX-DIVERGE rid=pdflip-2-8 verdict=store_short prompt=66944 credit=32768 ...
              best_prev=pdflip-0-2 prev_prompt=61345 ... common=37859 common_page=37824 lost=5056'
  fr 07:18:48 'PREFIX-DIVERGE-TEXT ... seg_role=system ... prev={...tools:list86} cur={...tools:list99}
              before=' ..."required": ["file_path", "content"] ...}\\n{"type": "function", "function":
              {"description": "' prev_after='Summarize a torch.cuda.memory._dump_snapshot() ...'
              cur_after='Create a doc, or apply several operations ...", "name": "mcp__claude_ai_Claude_Docs__batch"'
  P  0-2 anchors 64 / 16384 / 32768 / 49152 / turn 55872 / END 61312; 2-8 read 32768 -> 37824 recomputed.
The fork is the built-in -> MCP line of the <tools> block ('Write' ends, then a different first MCP server):
13 tokens after the line head ('{"type": "function", "function": {"description": "'), head = 37846.

Pinned (hermetic, CPU):
  A. geometry: the Qwen3.x tool-line ids (verified against the NF tokenizer.json 19.10.), the group bounds,
     the grid positions; with the rc15 0-2 steps the deepest anchor under 2-8's claim is 37824 (lost 0 instead
     of 5056); the plain grid (no tool ids) gives 36864 (lost 960, the review's number); bounded point count;
  B. the planner on 0-2's real steps: slots, no slot = no plan and no promise, stale plans freed + declined,
     one slot where another track writes the position, nothing outside the first message / switch off;
  C. the REAL UnifiedRadixCache (unigram and exact bigram): a variant forking at a group boundary resumes there
     with the grid track's state; END and turn anchor unchanged; finish and chunk insert consume the plans;
     an unmarked plan frees its slots;
  D. wiring: switches registered default ON, positions read no rank-local state.
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

GRID_PENDING = "_pdflip_sysgrid_pending"   # turn_anchor.SYSGRID_PENDING_ATTR, by name
TURN_PENDING = "_pdflip_turn_pending"
IM = 248045                                # <|im_start|> on NF
GEN = 5
HEAD = (4754, 1267, 763, 328, 1628, 487, 328, 1628, 763)   # '{"type": "function", "function":'
DESC = (5046, 4532, 763, 328)                               # ' {"description": "'
NAME = (591, 763, 328)                                      # 'name": "'
MCP = (76, 4520, 548)                                       # 'mcp__'
SEP = 548
WRITE = (7734,)                                             # 'Write'


def _line(name, body, base):
    """One <tools> line: head, description (``body`` filler ids), name, parameters."""
    return list(HEAD) + list(DESC) + [base + i for i in range(body)] + [487, 328] + list(NAME) + list(name) + \
        [487, 328, 13390, 763, 5046]


def _mcp(server, tool):
    return list(MCP) + list(server) + [SEP] + list(tool)


# ---------------------------------------------------------------- rc15 0-2 / 2-8 in tokens (page 64, FLA 64)
R_PAGE = 64
R_N = 61345                       # 0-2 prompt
R_COMMON = 37859                  # front PREFIX-DIVERGE common (2-8 vs 0-2)
R_COMMON_PAGE = 37824
R_MCP_HEAD = R_COMMON - 13        # 37846: the built-in -> MCP line head
R_STORE = 32768                   # 0-2's deepest anchor under the fork (2-8 read start)
R_TURN = 55872
R_END = 61312
R_STEPS = [(0, 16384, 16384), (16384, 32768, 32768), (32768, 49152, 49152), (49152, R_N, R_END)]


def _rc15_src(u, mcp_servers=((3464, 1209), (8090, 15449), (51860, 768))):
    """0-2's shape: system text, built-in tool lines, the first MCP line head at 37846, more MCP servers,
    the second <|im_start|> at ``u`` (segment end), the turn message at 55872+, the generation prompt."""
    ids = [100000 + (i % 40000) for i in range(R_N)]
    ids[0] = IM
    pos = 2000
    k = 0
    while pos + 400 < R_MCP_HEAD:      # built-in tool lines of ~400-1200 ids
        body = min(1200, R_MCP_HEAD - pos - 40) - 40
        ln = _line(WRITE if k % 2 else (9000 + k,), max(10, body), 200000 + 7 * k)
        ids[pos:pos + len(ln)] = ln
        pos += len(ln)
        k += 1
    pos = R_MCP_HEAD
    for j, srv in enumerate(mcp_servers):
        for tool in range(2):
            ln = _line(_mcp(srv, (60000 + tool,)), 300, 300000 + 1000 * j + 10 * tool)
            ids[pos:pos + len(ln)] = ln
            pos += len(ln)
    assert pos < u, (pos, u)
    ids[u] = IM
    ids[R_TURN + 10] = IM
    ids[R_N - GEN] = IM
    return ids


@pytest.fixture(autouse=True)
def _grid_on(monkeypatch):
    for k in ("FLLIPER_PDFLIP_SYS_GRID_ANCHOR", "FLLIPER_PDFLIP_SYS_GRID_TOKENS", "FLLIPER_PDFLIP_SYS_GRID_MAX",
              "FLLIPER_PDFLIP_SYS_GRID_TOOL_HEAD", "FLLIPER_PDFLIP_SYS_GRID_TOOL_NAME",
              "FLLIPER_PDFLIP_SYS_GRID_MCP_PREFIX"):
        monkeypatch.delenv(k, raising=False)
    twa._reset_for_test()
    yield
    twa._reset_for_test()


# -- A. geometry ----------------------------------------------------------------------------------------
def test_tool_group_bounds_builtin_then_one_per_mcp_server():
    ids = [IM, 5, 6] + _line(WRITE, 5, 1000) + _line((9001,), 5, 1100) + _line(_mcp((3464, 1209), (1,)), 5, 1200) \
        + _line(_mcp((3464, 1209), (2,)), 5, 1300) + _line(_mcp((8090, 15449), (1,)), 5, 1400) + [IM, 7]
    heads = [i for i in range(len(ids)) if tuple(ids[i:i + len(HEAD)]) == HEAD]
    assert len(heads) == 5
    got = ta.tool_group_bounds(ids, 0, len(ids) - 2, HEAD, NAME, MCP)
    assert got == [heads[0], heads[2], heads[4]]       # built-in group, server 1, server 2
    assert ta.tool_group_bounds(ids, 0, len(ids), (), NAME, MCP) == []


@pytest.mark.parametrize("u", [40990, 45000, 52000])
def test_rc15_positions_anchor_the_mcp_line_exactly(u):
    ids = _rc15_src(u)
    pts = ta.sys_grid_positions(ids, IM, R_PAGE, 4096, 16, HEAD, NAME, MCP)
    assert all(p % R_PAGE == 0 and 0 < p <= (u // R_PAGE) * R_PAGE for p in pts)
    assert R_COMMON_PAGE in pts                         # floor_page(37846) == 2-8's claim
    assert (u // R_PAGE) * R_PAGE in pts                # segment end
    assert len(pts) <= -(-u // 4096) + 2


def _anchors_of_0_2(ids, monkeypatch, tool=True):
    """Run the planner over 0-2's four steps; the anchors 0-2 leaves: chunk ends, END, turn, grid."""
    if not tool:
        monkeypatch.setenv("FLLIPER_PDFLIP_SYS_GRID_TOOL_HEAD", "")
    req = _sreq(ids, rid="pdflip-0-2")
    alloc = _Alloc(range(100, 200))
    grid = []
    for prefix, end, main in R_STEPS:
        req.cache_protected_len = prefix
        _plan(req, alloc, prefix, end, main, chunk=64, page=R_PAGE)
        grid += [p[0] for p in (getattr(req, GRID_PENDING, None) or ())]
        setattr(req, GRID_PENDING, None)
        for a in (TURN_PENDING, "_pdflip_gen_pending", "_pdflip_fork_pending"):
            setattr(req, a, None)
    return sorted(set(grid)), {64, 16384, 32768, 49152, R_TURN, R_END}


@pytest.mark.parametrize("u", [40990, 45000, 52000])
def test_rc15_2_8_loses_nothing_below_the_page(u, monkeypatch):
    """RED on the base: 2-8's deepest anchor under its claim is 32768 (5056 lost)."""
    grid, other = _anchors_of_0_2(_rc15_src(u), monkeypatch)
    deepest = max(a for a in set(grid) | other if a <= R_COMMON_PAGE)
    assert deepest == R_COMMON_PAGE, (grid, deepest)
    assert R_COMMON_PAGE - deepest == 0                  # was 5056


def test_rc15_2_8_plain_grid_is_the_reviews_960(monkeypatch):
    grid, other = _anchors_of_0_2(_rc15_src(40990), monkeypatch, tool=False)
    deepest = max(a for a in set(grid) | other if a <= R_COMMON_PAGE)
    assert deepest == 36864 and R_COMMON_PAGE - deepest == 960
    # chunk ends need no grid track (the chunk anchor sits there): 7 grid tracks + the segment end
    assert grid == [4096, 8192, 12288, 20480, 24576, 28672, 36864, 40960]


def test_rc15_cost_is_one_track_per_window(monkeypatch):
    grid, _ = _anchors_of_0_2(_rc15_src(40990), monkeypatch)
    # one track per window below the segment end (the chunk ends need none), the first window snapped to the
    # first tool line (1984), the window holding the MCP line snapped to it (37824), plus the segment end:
    # 9 slots on 0-2's three system steps, 9 x 56.11 MiB through the arena/L3 once per tool-list variant
    assert grid == [1984, 8192, 12288, 20480, 24576, 28672, 36864, 37824, 40960]


def test_a_long_first_message_widens_the_grid():
    ids = [IM] + [5] * 200_000 + [IM, 1, 2, 3, IM, 4, 5]
    pts = ta.sys_grid_positions(ids, IM, 64, 4096, 16, HEAD, NAME, MCP)
    assert len(pts) <= 17 and pts[-1] == (200_001 // 64) * 64
    assert pts[0] == 4096 * 4                            # 48 windows > 16: grid x 3 -> x4 (power of two, review A4)


@pytest.mark.parametrize("L", [40_000, 64_000, 100_000, 131_000, 150_000, 175_000, 192_000, 250_000])
def test_grid_tracks_plus_chunk_anchors_stay_inside_the_arena_share(L):
    """Review S62 (A4), cost bound: the grid tracks (points off the 16384 chunk ends, which need no track)
    plus the chunk anchors, the 64 anchor, turn and END stay within the per-request arena share of 21
    (``MAMBA-ARENA ... cap=21 slots=84``). Grid x3 (12288) gave 22-27 at 150k-192k: SHARE displaced the
    request's own anchors."""
    ids = [IM] + [5] * (L - 2) + [IM] + [7] * 30
    pts = ta.sys_grid_positions(ids, IM, 64, 4096, 16, HEAD, NAME, MCP)
    tracks = [p for p in pts if p % 16384]
    chunks = (len(ids) - 1) // 16384
    assert 1 + chunks + len(tracks) + 2 <= 21, (L, len(tracks), chunks)


@pytest.mark.parametrize("ids", [
    [5, 6, 7, IM, 8],              # no turn-start at 0: no first message
    [IM] + [5] * 9000,             # one message only
    None,
])
def test_no_segment_no_positions(ids):
    assert ta.sys_grid_positions(ids, IM, 64, 4096, 16, HEAD, NAME, MCP) == []


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


def _sreq(ids, rid="pdflip-0-2", cpl=0):
    return SimpleNamespace(rid=rid, origin_input_ids=array("q", ids), output_ids=[], cache_protected_len=cpl)


def _plan(req, alloc, prefix, end, main, tok=IM, chunk=1, page=1, desc=None):
    batch = SimpleNamespace(reqs=[req], req_to_token_pool=SimpleNamespace(mamba_allocator=alloc))
    return ta.note_step(batch=batch, desc=desc, req=req, row=0, prefix=prefix, end=end, track_mask=True,
                        main_track=main, chunk=chunk, page=page, tok=tok)


R_SRC = _rc15_src(40990)


def test_the_rc15_third_step_plans_three_grid_tracks():
    alloc = _Alloc([7, 8, 9, 10])
    req = _sreq(R_SRC, cpl=32768)
    desc = _plan(req, alloc, 32768, 49152, 49152, chunk=64, page=R_PAGE)
    plans = getattr(req, GRID_PENDING)
    assert [p[0] for p in plans] == [36864, 37824, 40960]
    assert [int(p[1]) for p in plans] == [7, 8, 9] and alloc.free_ids == [10]
    # gather position == insert position, the step's geometry
    assert desc.targets == [36864, 37824, 40960] and set(desc.prefixes) == {32768}
    assert set(desc.ext_lens) == {16384}
    assert [k for k, _ in ta.pop_all_pending(req)] == ["grid", "grid", "grid"]
    assert not ta.has_pending(req)


def test_switch_off_plans_nothing(monkeypatch):
    monkeypatch.setenv("FLLIPER_PDFLIP_SYS_GRID_ANCHOR", "0")
    alloc = _Alloc([7, 8, 9])
    req = _sreq(R_SRC, cpl=32768)
    assert _plan(req, alloc, 32768, 49152, 49152, chunk=64, page=R_PAGE) is None
    assert getattr(req, GRID_PENDING, None) is None and alloc.free_ids == [7, 8, 9]


def test_no_slot_no_plan_no_promise(monkeypatch):
    monkeypatch.setattr(twa, "_ARMED", True)
    req = _sreq(R_SRC, cpl=32768)
    alloc = _Alloc([7])
    _plan(req, alloc, 32768, 49152, 49152, chunk=64, page=R_PAGE)
    assert [p[0] for p in getattr(req, GRID_PENDING)] == [36864]
    assert twa.status(req.rid, 36864) == "planned"
    for t in (37824, 40960):
        assert twa.status(req.rid, t) is None, "a promise without a slot holds a twin for nothing"


def test_a_stale_grid_plan_is_freed_and_declined(monkeypatch):
    monkeypatch.setattr(twa, "_ARMED", True)
    alloc = _Alloc([11, 12, 13, 14, 15, 16])
    req = _sreq(R_SRC, cpl=32768)
    _plan(req, alloc, 32768, 49152, 49152, chunk=64, page=R_PAGE)
    req.output_ids = [1]
    _plan(req, alloc, R_N, R_N + 1, R_N, chunk=64, page=R_PAGE)           # decode: stale freed
    assert sorted(alloc.freed) == [11, 12, 13]
    assert getattr(req, GRID_PENDING, None) is None
    assert all(twa.status(req.rid, t) == "declined" for t in (36864, 37824, 40960))


def test_a_skipped_insert_declines_the_promise(monkeypatch):
    monkeypatch.setattr(twa, "_ARMED", True)
    req = _sreq(R_SRC, cpl=32768)
    _plan(req, _Alloc([5, 6, 7]), 32768, 49152, 49152, chunk=64, page=R_PAGE)
    for kind, pend in ta.pop_all_pending(req):
        ok, why = ta.insert_verdict(pend)
        assert (kind, ok) == ("grid", False) and why.startswith("unmarked")
        ta.note_skip(req, why, pend[0], kind=kind)
        assert twa.status(req.rid, pend[0]) == "declined"


@pytest.mark.parametrize("why,kw", [
    ("decode step", dict(out=[1])),
    ("protected prefix past the segment", dict(cpl=40960, prefix=40960)),
    ("step past the segment", dict(prefix=49152, end=R_N, main=R_END)),
    ("track mask off", dict(mask=False)),
])
def test_no_grid_track(why, kw):
    alloc = _Alloc([7, 8, 9])
    req = _sreq(R_SRC, cpl=kw.get("cpl", 32768))
    req.output_ids = kw.get("out", [])
    batch = SimpleNamespace(reqs=[req], req_to_token_pool=SimpleNamespace(mamba_allocator=alloc))
    ta.note_step(batch=batch, desc=None, req=req, row=0, prefix=kw.get("prefix", 32768), end=kw.get("end", 49152),
                 track_mask=kw.get("mask", True), main_track=kw.get("main", 49152), chunk=64, page=R_PAGE, tok=IM)
    assert not getattr(req, GRID_PENDING, None), why


def test_a_position_the_turn_track_writes_is_one_slot():
    """A two-message prompt: the turn anchor floor_page(u) IS the segment end."""
    ids = [IM] + [100000 + i for i in range(9000)] + [IM] + [5] * 300 + [IM, 1, 2, 3, 4]
    req = _sreq(ids)
    alloc = _Alloc([11, 12, 13, 14])
    _plan(req, alloc, 0, len(ids), ((len(ids) - 1) // 64) * 64, chunk=64, page=64)
    turn = getattr(req, TURN_PENDING)[0]
    assert turn == (9001 // 64) * 64
    assert [p[0] for p in getattr(req, GRID_PENDING)] == [4096, 8192]   # 8960 (= turn) not twice
    assert alloc.free_ids == [14], "a second slot for the turn position"


# -- C. the real tree -----------------------------------------------------------------------------------
MAMBA_SLOTS = 24
KV_SIZE = 1024
NUM_LAYERS = 8
FULL_LAYER_IDS = (3, 7)
MAMBA_LAYER_IDS = [i for i in range(NUM_LAYERS) if i not in FULL_LAYER_IDS]
T_GRID = 32                       # FLLIPER_PDFLIP_SYS_GRID_TOKENS for the page-1 tree


def _small_src():
    ids = [IM] + [700000 + i for i in range(10)]
    ids += _line(WRITE, 12, 710000) + _line((9001,), 12, 720000)
    mcp_at = len(ids)
    ids += _line(_mcp((3464, 1209), (1,)), 8, 730000) + _line(_mcp((8090, 15449), (2,)), 8, 740000)
    u = len(ids)
    ids += [IM] + [750000 + i for i in range(20)]
    turn = len(ids)
    ids += [IM] + [760000 + i for i in range(15)]
    ids += [IM, 1, 2, 3, 4]
    return ids, mcp_at, u, turn


SMALL, MCP_AT, U, TURN = _small_src()
N = len(SMALL)
VARIANT = SMALL[: MCP_AT + 13] + [990000 + i for i in range(40)]   # another first MCP server's description


@pytest.fixture
def group_p(monkeypatch):
    from flliper.srt.mem_cache import unified_radix_cache as urc

    monkeypatch.setenv("FLLIPER_PDFLIP_GROUP", "P")
    monkeypatch.setenv("FLLIPER_PDFLIP_BIGRAM_ANCHOR_EXACT", "1")
    monkeypatch.setenv("FLLIPER_PDFLIP_SYS_GRID_TOKENS", str(T_GRID))
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
    req = Req(rid="pdflip-0-2", origin_input_text="", origin_input_ids=array("q", ids),
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
    return 1.0 + t / 64.0     # the state "after t tokens" the forward leaves in a grid slot


def _one_forward(fx, chunked=False):
    """0-2's shape at page 1: ONE step [0, N) (or [0, U+3) as an inner chunk), turn track, END at N-1;
    the forward leaves each grid slot holding the state after its position."""
    req = _req(fx, SMALL)
    end = U + 3 if chunked else N
    _write_kv(fx, req, end)
    batch = SimpleNamespace(reqs=[req], req_to_token_pool=fx.pool)
    desc = ta.note_step(batch=batch, desc=None, req=req, row=0, prefix=0, end=end, track_mask=True,
                        main_track=end - 1, chunk=1, page=1, tok=IM)
    assert desc is not None
    for pend in getattr(req, GRID_PENDING, None) or ():
        _state(fx, pend[1], _sv(pend[0]))
    if getattr(req, TURN_PENDING, None) is not None:
        _state(fx, getattr(req, TURN_PENDING)[1], 0.5)
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
def test_the_variant_resumes_at_the_mcp_line(group_p, bigram):
    """RED on the base: the variant falls back to the root (no anchor inside the first message)."""
    fx = _fixture(bigram)
    _one_forward(fx)
    assert _claim(fx, VARIANT) == (MCP_AT, pytest.approx(_sv(MCP_AT)))


@pytest.mark.parametrize("bigram", [False, True], ids=["unigram", "bigram"])
def test_end_and_turn_anchor_are_unchanged(group_p, bigram, monkeypatch):
    later = SMALL + [5, 6, 7]
    nxt = SMALL[: TURN + 1] + [880000 + i for i in range(20)]
    fx = _fixture(bigram)
    _one_forward(fx)
    monkeypatch.setenv("FLLIPER_PDFLIP_SYS_GRID_ANCHOR", "0")
    fx0 = _fixture(bigram)
    _one_forward(fx0)
    assert _claim(fx, later) == _claim(fx0, later) == (N - (1 if bigram else 0), pytest.approx(9.0))
    assert _claim(fx, nxt) == _claim(fx0, nxt) == (TURN, pytest.approx(0.5))
    assert _claim(fx0, VARIANT)[0] < MCP_AT, "the gap itself, switch off"


def test_an_inner_chunk_consumes_the_grid_plans(group_p):
    fx = _fixture(False)
    req = _one_forward(fx, chunked=True)
    assert req.cache_protected_len == U + 3
    assert getattr(req, GRID_PENDING, None) is None
    assert _claim(fx, VARIANT) == (MCP_AT, pytest.approx(_sv(MCP_AT)))


def test_an_unmarked_grid_plan_inserts_nothing_and_frees_its_slots(group_p):
    fx = _fixture(False)
    free_before = fx.pool.mamba_allocator.available_size()
    req = _req(fx, SMALL)
    _write_kv(fx, req, N)
    batch = SimpleNamespace(reqs=[req], req_to_token_pool=fx.pool)
    monkey_turn = ta.note_step(batch=batch, desc=None, req=req, row=0, prefix=0, end=N, track_mask=True,
                               main_track=N - 1, chunk=1, page=1, tok=IM)
    assert monkey_turn is not None and getattr(req, GRID_PENDING)
    # no GDN mark: a graph replay -- nothing inserted, every slot back
    _state(fx, req.mamba_pool_idx, 9.0)
    req.kv_committed_len = N
    req.kv_allocated_len = N
    fx.cache.cache_finished_req(req, is_insert=True)
    assert _claim(fx, VARIANT)[0] < MCP_AT
    assert fx.pool.mamba_allocator.available_size() == free_before - 1   # the END slot only


# -- D. wiring ------------------------------------------------------------------------------------------
def test_the_switches_are_registered_default_on(monkeypatch):
    assert envs.FLLIPER_PDFLIP_SYS_GRID_ANCHOR.get() is True
    assert envs.FLLIPER_PDFLIP_SYS_GRID_TOKENS.get() == 4096
    assert envs.FLLIPER_PDFLIP_SYS_GRID_MAX.get() == 16
    assert ta.sys_grid_armed() is True
    assert ta.sys_grid_config()[2:] == (HEAD, NAME, MCP)


def test_the_positions_read_no_rank_local_state():
    """R11 / #1004: a pure function of the ids, the env token, the page and the env config."""
    src = "".join(inspect.getsource(f) for f in (
        ta.sys_grid_positions, ta.req_sysgrid_pos, ta.tool_group_bounds, ta.segment_end, ta._find))
    tree = ast.parse(src)
    names = {n.id for n in ast.walk(tree) if isinstance(n, ast.Name)}
    attrs = {n.attr for n in ast.walk(tree) if isinstance(n, ast.Attribute)}
    attrs |= {n.value for n in ast.walk(tree) if isinstance(n, ast.Constant) and isinstance(n.value, str)}
    for bad in ("pp_rank", "tree_cache", "cache_controller", "time", "random", "torch",
                "cache_protected_len", "prefix_indices"):
        assert bad not in names and bad not in attrs, bad
