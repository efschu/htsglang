"""FORK ANCHOR (27B line, 26.09., boot dkr27brc10bar1agent09261821).

9 of 78 D-direct requests prefilled 1.2k-4.7k tokens: each was a SIBLING of a
request just served -- the same chat up to the predecessor's last message, then
another message instead of the generation prompt ``<|im_start|>assistant\\n
<think>\\n``. D's device match reached exactly N-5 of the predecessor
(``#904 ... reached=31441`` against weg2-10-14's 31446), where no recurrent
anchor existed: P anchored at N-1 (P-TRIM-END-ANCHOR), 4 tokens past the fork,
so the read fell back to the previous anchor (``#1028B FETCH CAP kv=31441
claimed=27384``). On a D-direct predecessor, D's prefill track landed at N-1
when ``extend % 64 == 1`` (26-66 -> 26-67).

Hermetic, CPU. Pinned, with ``SGLANG_WEG2_FORK_ANCHOR_TOKEN`` set:
  * P's intake cuts a leg-1 prompt at the last fork token of its tail, and
    keeps today's N-1 cut when the switch is off or no fork token is there;
  * on the REAL UnifiedRadixCache (unigram and bigram keys): a sibling that
    forks after that token resumes AT the fork with P's state there -- today
    it falls back to the previous chunk anchor -- and D's leg 2 of the same
    prompt resumes at the fork too (the tail is its own extend);
  * group D's store read ends at the same cut (so a fork-cut leg-2 read lands
    complete), group P's span and every switch-off span are unchanged;
  * group D's extend track keeps the anchor at or below the fork (up to 63
    tokens earlier); group P's track never moves;
  * the launcher switch reaches both groups, needs --p-trim-end-anchor, and is
    default off.
"""

import logging
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
from sglang.srt.weg2 import p_trim_end_anchor as pt

TOKEN_ENV = "SGLANG_WEG2_FORK_ANCHOR_TOKEN"   # the switch, by name (red on a tree without it)
IM = 248045                                   # <|im_start|> on Qwen3.8-27B
GEN = [IM, 74455, 198, 248068, 198]           # "<|im_start|>assistant\n<think>\n"
BODY = list(range(2000, 2018))                # the chat up to its last message
PROMPT = BODY + GEN                           # N = 23
N = len(PROMPT)
F = len(BODY)                                 # the fork cut: 18 = N-5
SIBLING = BODY + [IM, 872, 198, 3000, 3001, 3002, 3003] + GEN   # another message, then GEN
CHUNK = 4
RID = "weg2-10-14"
S_EARLY, S_FORK, S_END = 0.5, 1.25, 7.5


@pytest.fixture
def fork_on(monkeypatch):
    monkeypatch.setenv(TOKEN_ENV, str(IM))
    return monkeypatch


def _recv(ids, **kw):
    base = dict(rid=RID, input_ids=array("q", ids), input_embeds=None, return_logprob=False,
                sampling_params=SamplingParams(max_new_tokens=1), session_params=None,
                session_id=None, mm_inputs=None)
    base.update(kw)
    return SimpleNamespace(**base)


# -- A. the pure rule ----------------------------------------------------------


def test_the_cut_is_the_last_fork_token_of_the_tail():
    from sglang.srt.weg2 import fork_anchor as fa

    assert fa.fork_cut(PROMPT, IM) == F
    assert fa.fork_cut(PROMPT, None) is None, "switch off"
    assert fa.fork_cut(BODY, IM) is None, "no fork token -> today's form"
    assert fa.fork_cut(PROMPT, IM, tail=4) is None, "outside the window -> today's form"
    assert fa.fork_cut(PROMPT[:-4], IM) is None, "a fork token as the LAST id is no cut below N-1"
    assert fa.fork_cut([IM, 5, 6], IM) is None, "never an empty prefix"
    two = BODY + [IM, 1, 2] + GEN
    assert fa.fork_cut(two, IM) == len(two) - 5, "the LAST one"


def test_the_switch_reads_one_positive_id():
    from sglang.srt.weg2 import fork_anchor as fa

    assert fa.fork_token({}) is None
    assert fa.fork_token({TOKEN_ENV: ""}) is None
    assert fa.fork_token({TOKEN_ENV: "0"}) is None
    assert fa.fork_token({TOKEN_ENV: "x"}) is None
    assert fa.fork_token({TOKEN_ENV: str(IM)}) == IM
    assert fa.max_tail({}) == 16


# -- B. P's intake ---------------------------------------------------------------


def test_p_cuts_a_leg1_prompt_before_its_generation_prompt(fork_on):
    recv = _recv(PROMPT)
    ids, tail = pt.split_ids(recv)
    assert list(ids) == BODY and list(tail) == GEN
    assert list(recv.input_ids) == PROMPT, "a PP relay must hand every rank the same ids"


def test_p_keeps_todays_cut_without_the_switch_or_a_fork_token(monkeypatch):
    monkeypatch.delenv(TOKEN_ENV, raising=False)
    ids, tail = pt.split_ids(_recv(PROMPT))
    assert list(ids) == PROMPT[:-1] and list(tail) == PROMPT[-1:]
    monkeypatch.setenv(TOKEN_ENV, str(IM))
    ids, tail = pt.split_ids(_recv(BODY + [5, 6, 7]))
    assert list(tail) == [7], "no fork token in the tail: N-1 as today"
    ids, tail = pt.split_ids(_recv(PROMPT, return_logprob=True))
    assert tail is None, "what P keeps stays kept"


def test_the_fork_census_line_formats(fork_on, caplog):
    with caplog.at_level(logging.INFO):
        pt.split_ids(_recv(PROMPT))
    assert not [r for r in caplog.records if r.levelno >= logging.ERROR]
    assert any("fork" in r.getMessage() and f"tokens={N}->{F}" in r.getMessage()
               for r in caplog.records)


# -- C. the real tree: the sibling and D's leg 2 resume at the fork ------------

MAMBA_SLOTS = 20
KV_SIZE = 512
NUM_LAYERS = 8
FULL_LAYER_IDS = (3, 7)
MAMBA_LAYER_IDS = [i for i in range(NUM_LAYERS) if i not in FULL_LAYER_IDS]


@pytest.fixture
def group_p(monkeypatch):
    from sglang.srt.mem_cache import unified_radix_cache as urc
    from sglang.srt.weg2 import form as _F

    monkeypatch.setenv("SGLANG_WEG2_GROUP", "P")
    monkeypatch.setenv(_F.FORM_ENV, _F.Weg2Form(
        arch="dense", experts="none", draft="dflash", p_draft="none", kv="paged_dcp",
        flip="family", vision="off", profile="qwen27b", model="m").env_value())
    monkeypatch.setattr(urc, "_WEG2_END_ANCHOR", True)
    return monkeypatch


def _fixture(bigram):
    server_args = ServerArgs(model_path="dummy", page_size=1)
    server_args._mamba_cache_chunk_size = FLA_CHUNK_SIZE
    server_args.chunked_prefill_size = CHUNK
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


def _req(fx, ids, tail):
    req = Req(rid=RID, origin_input_text="", origin_input_ids=array("q", ids),
              sampling_params=SamplingParams(temperature=0, max_new_tokens=1))
    setattr(req, pt.TRIM_ATTR, array("q", tail))
    fx.pool.alloc([req])
    req.output_ids = array("q")
    req.full_untruncated_fill_ids = array("q", ids)
    req.swa_uuid_for_lock = None
    req.extra_key = None
    mr = fx.cache.match_prefix(MatchPrefixParams(key=_key(fx, ids)))
    req.prefix_indices = mr.device_indices.to(torch.int64)
    req.cache_protected_len = len(req.prefix_indices)
    req.key_match_depth = mr.key_match_depth
    req.last_node = mr.last_device_node
    fx.cache.inc_lock_ref(req.last_node)
    return req


def _set_state(fx, req, value):
    fx.pool.mamba_pool.mamba_cache.temporal[:, req.mamba_pool_idx] = value


def _chunk(fx, req, end):
    start = len(req.prefix_indices)
    fx.pool.write((req.req_pool_idx, slice(start, end)), fx.allocator.alloc(end - start))
    req.set_extend_range(start, end)
    fx.cache.cache_unfinished_req(req, chunked=True)


def _finish(fx, req):
    n, start = len(req.origin_input_ids), len(req.prefix_indices)
    if start < n:
        fx.pool.write((req.req_pool_idx, slice(start, n)), fx.allocator.alloc(n - start))
    req.set_extend_range(start, n)
    req.kv_committed_len = n
    req.kv_allocated_len = n
    fx.cache.cache_finished_req(req, is_insert=True)


def _p_leg1(fx):
    """P's intake decision (the real split_ids), then the prefill of what it
    admitted: chunk anchors every CHUNK tokens, the state after token k is
    S_EARLY up to the last chunk end, S_FORK after BODY, S_END after N-1."""
    ids, tail = pt.split_ids(_recv(PROMPT))
    ids, tail = list(ids), list(tail)
    req = _req(fx, ids, tail)
    _set_state(fx, req, S_EARLY)
    for end in range(CHUNK, len(ids), CHUNK):
        _chunk(fx, req, end)
    _set_state(fx, req, S_FORK if len(ids) == F else S_END)
    _finish(fx, req)
    return req


def _claim(fx, ids):
    """What a reader of ``ids`` resumes from (max prefix, mamba validator)."""
    claim = Req._compute_max_prefix_len(
        SimpleNamespace(return_logprob=False, logprob_start_len=0), len(ids))
    mr = fx.cache.match_prefix(MatchPrefixParams(key=_key(fx, ids[:claim])))
    node = mr.last_device_node
    slot = node.component_data[ComponentType.MAMBA].value
    state = None if slot is None else float(
        fx.pool.mamba_pool.mamba_cache.temporal[:, slot].float().mean())
    return len(mr.device_indices), state


@pytest.mark.parametrize("bigram", [False, True], ids=["unigram", "bigram"])
def test_the_sibling_resumes_at_the_fork_not_at_the_previous_anchor(group_p, bigram):
    group_p.setenv(TOKEN_ENV, str(IM))
    fx = _fixture(bigram)
    _p_leg1(fx)
    depth, state = _claim(fx, SIBLING)
    fork_units = F - (1 if bigram else 0)
    # today: P anchors at N-1 (past the fork) and the sibling falls back below
    # the fork -- the 27384-of-31441 shape of weg2-12-17.
    assert depth == fork_units, (depth, fork_units)
    assert state == pytest.approx(S_FORK)


@pytest.mark.parametrize("bigram", [False, True], ids=["unigram", "bigram"])
def test_todays_cut_leaves_the_sibling_on_the_previous_anchor(group_p, bigram):
    group_p.delenv(TOKEN_ENV, raising=False)
    fx = _fixture(bigram)
    _p_leg1(fx)
    depth, state = _claim(fx, SIBLING)
    # P's only anchor on the tail is the END anchor at N-1, 4 tokens past the
    # fork (its inner chunk anchors are released once the end anchor stands),
    # so the sibling resumes from nothing on this path -- on the boot, from
    # whatever older anchor the store still held (weg2-12-17: 27384 of 31441).
    assert depth < F - (1 if bigram else 0) and state != pytest.approx(S_FORK), (depth, state)
    d_depth, d_state = _claim(fx, PROMPT)
    assert d_depth == (N - 1) - (1 if bigram else 0) and d_state == pytest.approx(S_END)


@pytest.mark.parametrize("bigram", [False, True], ids=["unigram", "bigram"])
def test_d_leg2_resumes_at_the_fork_and_extends_the_tail(group_p, bigram):
    group_p.setenv(TOKEN_ENV, str(IM))
    fx = _fixture(bigram)
    _p_leg1(fx)
    depth, state = _claim(fx, PROMPT)
    assert depth == F - (1 if bigram else 0) and state == pytest.approx(S_FORK)


def test_the_end_anchor_line_names_the_fork_target(group_p, caplog):
    group_p.setenv(TOKEN_ENV, str(IM))
    fx = _fixture(True)
    with caplog.at_level(logging.WARNING):
        _p_leg1(fx)
    lines = [r.getMessage() for r in caplog.records if "WEG2 END-ANCHOR n=" in r.getMessage()]
    assert lines and f"tokens={N} anchor={F} target={F}" in lines[-1] and "ok=True" in lines[-1]
    assert lines[-1].endswith(f"trim={N - F}")


# -- D. group D's store read ends at the fork --------------------------------


def _dreq(ids=PROMPT, **kw):
    base = dict(rid=RID, origin_input_ids=array("q", ids), output_ids=[], return_logprob=False,
                input_embeds=None, session_id=None, multimodal_inputs=None)
    base.update(kw)
    return SimpleNamespace(**base)


def test_group_d_store_read_ends_at_the_fork(monkeypatch):
    from sglang.srt.managers import scheduler as S

    monkeypatch.setenv("SGLANG_WEG2_GROUP", "D")
    monkeypatch.delenv(TOKEN_ENV, raising=False)
    assert S._weg2_fork_match_end(_dreq(), N - 1) == N - 1, "switch off: today's span"
    monkeypatch.setenv(TOKEN_ENV, str(IM))
    assert S._weg2_fork_match_end(_dreq(), N - 1) == F
    assert S._weg2_fork_match_end(_dreq(), F - 3) == F - 3, "never widened"
    for kw in (dict(rid="abc"), dict(return_logprob=True), dict(multimodal_inputs=object()),
               dict(session_id="s"), dict(output_ids=[1])):
        assert S._weg2_fork_match_end(_dreq(**kw), N - 1) == N - 1, kw
    monkeypatch.setenv("SGLANG_WEG2_GROUP", "P")
    assert S._weg2_fork_match_end(_dreq(), N - 1) == N - 1, "group P keeps its span"


def test_the_span_cap_sits_where_the_span_is_cut():
    import inspect

    from sglang.srt.managers import scheduler as S

    src = inspect.getsource(S.Scheduler._prefetch_kvcache)
    a = src.index("_match_end = req._compute_max_prefix_len(")
    b = src.index("_match_end = _weg2_fork_match_end(req, _match_end)")
    c = src.index("_new_input_tokens = req.full_untruncated_fill_ids[_matched_len:_match_end]")
    assert a < b < c


# -- E. group D's extend track stays at or below the fork ----------------------


@pytest.fixture
def d_track(monkeypatch):
    import sglang.srt.managers.schedule_batch as sb

    monkeypatch.setenv("SGLANG_WEG2_GROUP", "D")
    monkeypatch.setattr(sb, "get_server_args", lambda: SimpleNamespace(
        mamba_cache_chunk_size=64, mamba_checkpoint_interval=None,
        enable_mamba_extra_buffer_lazy=lambda: False))
    return monkeypatch, sb


def _track(sb, ids, prefix, rid=RID):
    batch = SimpleNamespace(req_to_token_pool=SimpleNamespace(get_mamba_ping_pong_other_idx=lambda i: 1 - i))
    req = SimpleNamespace(rid=rid, origin_input_ids=array("q", ids), output_ids=[],
                          return_logprob=False, input_embeds=None, session_id=None,
                          multimodal_inputs=None, prefix_indices=list(range(prefix)),
                          mamba_ping_pong_track_buffer=torch.tensor([0, 1]), mamba_next_track_idx=0,
                          mamba_branching_seqlen=None, mamba_last_track_seqlen=None)
    ext = len(ids) - prefix
    req.extend_range = SimpleNamespace(start=prefix, end=len(ids), length=ext)
    entry = sb.ScheduleBatch._mamba_radix_cache_v2_req_prepare_for_extend(batch, req)
    return req.mamba_last_track_seqlen, entry.track_seqlen


def test_d_track_moves_below_the_fork_when_the_grid_point_is_inside_it(d_track):
    mp, sb = d_track
    body = list(range(5000, 5000 + 1025 - 5))        # 26-66: extend 1025 = 16*64 + 1
    ids = body + GEN
    mp.delenv(TOKEN_ENV, raising=False)
    assert _track(sb, ids, 0) == (1024, 1025), "today: the anchor at N-1"
    mp.setenv(TOKEN_ENV, str(IM))
    aligned, seqlen = _track(sb, ids, 0)
    assert aligned == 960 and aligned <= len(body), aligned
    assert seqlen == 961, "a mid-step grid point reads h (+1), like the branching track"


def test_d_track_works_from_an_unaligned_resume_prefix(d_track):
    """26-66 resumed at 78886 (not a multiple of 64): the grid is the step's."""
    mp, sb = d_track
    prefix = 37
    ids = list(range(5000, 5000 + prefix + 1025 - 5)) + GEN
    mp.delenv(TOKEN_ENV, raising=False)
    assert _track(sb, ids, prefix) == (prefix + 1024, prefix + 1025)
    mp.setenv(TOKEN_ENV, str(IM))
    aligned, seqlen = _track(sb, ids, prefix)
    assert aligned == prefix + 960 and seqlen == prefix + 961
    assert (seqlen - prefix) % 64 == 1, "the kernel reads h[(seqlen - prefix) // 64] = h[15]"


def test_the_track_never_moves_on_group_p(d_track):
    """Review RV: on P the ids are already cut at the fork; a fork token in a
    short LAST message (< ~14 tokens) would otherwise pull P's inner track
    back one chunk. The track rule is group D's alone."""
    mp, sb = d_track
    mp.setenv(TOKEN_ENV, str(IM))
    # a P-cut prompt: ... <|im_start|>user\n + 9-token last message, fork-cut
    last = [IM, 872, 198] + list(range(7000, 7009))
    ids = list(range(5000, 5000 + 1025 - len(last))) + last
    mp.setenv("SGLANG_WEG2_GROUP", "D")
    assert _track(sb, ids, 0)[0] == 960, "the rule itself fires on these ids"
    mp.setenv("SGLANG_WEG2_GROUP", "P")
    assert _track(sb, ids, 0) == (1024, 1025), "group P: today's track"
    mp.delenv("SGLANG_WEG2_GROUP", raising=False)
    assert _track(sb, ids, 0) == (1024, 1025), "no group: today's track"


def test_d_track_is_unchanged_where_the_grid_point_is_below_the_fork(d_track):
    mp, sb = d_track
    mp.setenv(TOKEN_ENV, str(IM))
    body = list(range(5000, 5000 + 1000 - 5))         # extend 1000: grid point 960 < fork 995
    assert _track(sb, body + GEN, 0) == (960, 1000)
    assert _track(sb, list(range(5000, 6025)), 0) == (1024, 1025), "no fork token: today"
    over = list(range(5000, 5000 + 1025 - 5)) + GEN
    assert _track(sb, over, 0, rid="x") == (1024, 1025), "not a front rid: today"
    short = list(range(5000, 5030)) + GEN             # shorter than one chunk: no track, as today
    assert _track(sb, short, 0) == (None, -1)
    near = list(range(5000, 5060)) + GEN              # no grid point of the step below the fork: today
    assert _track(sb, near, 0) == (64, 65)


# -- F. the launcher switch --------------------------------------------------------


def test_the_launcher_switch_is_default_off_and_needs_the_trim():
    from sglang.srt.weg2 import launcher as L

    base = ["--tree", "/t", "--tag", "t"]
    ns = L.build_parser().parse_args(base)
    assert ns.fork_anchor_token is None
    assert L.fork_anchor_env(None, False) == {} and L.fork_anchor_env(None, True) == {}
    assert L.fork_anchor_env(IM, True) == {TOKEN_ENV: str(IM)}
    with pytest.raises(SystemExit):
        L.fork_anchor_env(IM, False)
    with pytest.raises(SystemExit):
        L.fork_anchor_env(0, True)
    ns_on = L.build_parser().parse_args(base + ["--fork-anchor-token", str(IM), "--p-trim-end-anchor"])
    assert ns_on.fork_anchor_token == IM
