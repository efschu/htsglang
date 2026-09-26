"""P-TRIM-END-ANCHOR (27B line, 24.09.): group P takes a leg-1 prompt of N
tokens as N-1 and never runs the 1-token END-ANCHOR forward; D claims N-1 and
computes the last token itself, as it always did.

Hermetic, CPU. The radix/store halves run on a REAL UnifiedRadixCache with the
REAL MambaComponent, allocator and pools (Agent G's #924/#773 harness shape),
driven through the shipped cache_unfinished_req / cache_finished_req, in the
unigram AND the bigram key scheme (group P runs SGLANG_HICACHE_BIGRAM_KEYS=1).
Pinned:
  * the intake trims a front leg-1 prompt on group P only, never mutates the
    relayed request, and keeps N=1 (and every read output, sessions, embeds,
    multimodal) on today's path by name;
  * with the trim there is no tokens=1 extend on P, and the chunk ends up to
    N-1 are today's;
  * D's resume point after the trimmed prefill equals today's: the same
    claimed depth, the recurrent state written at N-1, the #1481 end-anchor
    mark on that node -- and the trimmed tree holds nothing deeper;
  * P still reports prompt_tokens = N, and the #1442 hand-off still hands D
    all N ids (its tokenizer takes them AS the prompt);
  * the launcher switch is P-only and default off.
"""

import ast
import inspect
import json
import logging
import os
import textwrap
from array import array
from types import SimpleNamespace

import pytest
import torch

from sglang.srt.configs.mamba_utils import Mamba2CacheParams, Mamba2StateShape
from sglang.srt.environ import envs
from sglang.srt.layers.attention.fla.chunk_delta_h import CHUNK_SIZE as FLA_CHUNK_SIZE
from sglang.srt.managers.schedule_batch import FINISH_LENGTH, Req
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

N = 23                                  # prompt length
CHUNK = 4                               # chunked_prefill_size of the test ("512" scaled down)
PROMPT = list(range(2000, 2000 + N))
RID = "weg2-3-7"
S1, S2 = 1.25, 7.5                      # the state after N-1 tokens / after N tokens


# -- A. the intake -----------------------------------------------------------


def _recv(ids, **kw):
    base = dict(rid=RID, input_ids=array("q", ids), input_embeds=None, return_logprob=False,
                sampling_params=SamplingParams(max_new_tokens=1), session_params=None,
                session_id=None, mm_inputs=None)
    base.update(kw)
    return SimpleNamespace(**base)


def test_the_switch_is_group_p_only_and_default_off():
    assert not pt.trim_armed({})
    assert not pt.trim_armed({"SGLANG_WEG2_GROUP": "P"})
    assert not pt.trim_armed({"SGLANG_WEG2_GROUP": "D", pt.TRIM_ENV: "1"})
    assert pt.trim_armed({"SGLANG_WEG2_GROUP": "P", pt.TRIM_ENV: "1"})


def test_a_leg1_prompt_is_cut_at_n_minus_1_and_the_relayed_request_is_untouched():
    recv = _recv(PROMPT)
    ids, tail = pt.split_ids(recv)
    assert list(ids) == PROMPT[:-1] and list(tail) == PROMPT[-1:]
    assert list(recv.input_ids) == PROMPT, "a PP relay must hand every rank the same ids"
    two = _recv(PROMPT[:2])
    ids2, tail2 = pt.split_ids(two)
    assert list(ids2) == PROMPT[:1] and list(tail2) == PROMPT[1:2]


@pytest.mark.parametrize("recv, why", [
    (_recv(PROMPT[:1]), "n<2"),                                   # the N=1 special case
    (_recv(PROMPT, rid="HEALTH_CHECK_1"), "not_leg1"),
    (_recv(PROMPT, sampling_params=SamplingParams(max_new_tokens=2)), "output_read"),
    (_recv(PROMPT, return_logprob=True), "logprob"),
    (_recv(PROMPT, input_embeds=[[0.0]] * N), "input_embeds"),
    (_recv(PROMPT, session_id="s"), "session"),
    (_recv(PROMPT, mm_inputs=object()), "mm"),
])
def test_what_stays_on_todays_path_is_named(recv, why):
    assert pt.keep_reason(recv) == why
    ids, tail = pt.split_ids(recv)
    assert tail is None and ids is recv.input_ids


def test_both_census_lines_format_without_a_logging_error(caplog, monkeypatch):
    """RC2 metal (weg2rc2 P log 23:27:58, the image request weg2-50-74): the kept()
    line raised in logging -- "TypeError: %d format: a real number is required, not
    str" -- because _note() puts n FIRST and the kept format put the reason first.
    Every kept() line was lost as '--- Logging error ---' plus a traceback."""
    monkeypatch.setattr(pt, "_counts", {})
    with caplog.at_level(logging.INFO, logger=pt.logger.name):
        pt.split_ids(_recv(PROMPT, mm_inputs=object()))
        pt.split_ids(_recv(PROMPT))
    msgs = [r.getMessage() for r in caplog.records if "P-TRIM-END-ANCHOR" in str(r.msg)]
    assert any("kept(mm)" in m and "n=1 " in m and "tokens=%d" % N in m for m in msgs), msgs
    assert any("tokens=%d->%d" % (N, N - 1) in m and "n=1 " in m for m in msgs), msgs


def test_the_scheduler_trims_only_when_armed_and_marks_the_req():
    """The normal-path Req is built from the trimmed ids and carries the tail;
    unarmed, the Req gets recv_req.input_ids as before."""
    from sglang.srt.managers.scheduler import Scheduler

    src = textwrap.dedent(inspect.getsource(Scheduler._handle_generate_request_impl))
    tree = ast.parse(src)
    armed = [n for n in ast.walk(tree) if isinstance(n, ast.Call)
             and getattr(n.func, "attr", None) == "trim_armed"]
    assert armed, "the trim is gated on trim_armed()"
    reqs = [n for n in ast.walk(tree) if isinstance(n, ast.Call)
            and getattr(n.func, "id", None) == "Req" and len(n.args) >= 3
            and isinstance(n.args[2], ast.Name) and n.args[2].id == "_weg2_ids"]
    assert len(reqs) == 1, "exactly the normal path builds its Req from the trim decision"
    assert "setattr(req, _weg2_trim.TRIM_ATTR, _weg2_tail)" in src


# -- B./C. the split and the extend plan --------------------------------------


def _schedule_policy(armed=True):
    """A private copy of schedule_policy with SGLANG_WEG2_END_ANCHOR read at
    import (the module constant the split reads), as the #1233 tests do."""
    import importlib.util

    import sglang.srt.managers.schedule_policy as sp

    old = os.environ.get("SGLANG_WEG2_END_ANCHOR")
    os.environ["SGLANG_WEG2_END_ANCHOR"] = "1" if armed else "0"
    try:
        spec = importlib.util.spec_from_file_location(
            "sglang.srt.managers._schedule_policy_probe_trim", sp.__file__)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        return mod
    finally:
        if old is None:
            os.environ.pop("SGLANG_WEG2_END_ANCHOR", None)
        else:
            os.environ["SGLANG_WEG2_END_ANCHOR"] = old


def _plan(sp, ids, tail):
    """The extends P schedules for one prompt, chunk by chunk, through the
    real split."""
    adder = SimpleNamespace(rem_chunk_tokens=CHUNK)
    req = SimpleNamespace(full_untruncated_fill_ids=list(ids), rid=RID)
    if tail is not None:
        setattr(req, pt.TRIM_ATTR, tail)
    start, extends = 0, []
    while start < len(ids):
        length = min(CHUNK, len(ids) - start)
        length, _ = sp.PrefillAdder._weg2_end_anchor_split(adder, req, start, length)
        extends.append(length)
        start += length
    return extends


def test_with_the_trim_p_runs_no_one_token_extend_and_the_same_chunk_ends():
    sp = _schedule_policy(armed=True)
    today = _plan(sp, PROMPT, None)
    ids, tail = pt.split_ids(_recv(PROMPT))
    trimmed = _plan(sp, ids, tail)
    assert today[-1] == 1, today                     # the END-ANCHOR forward
    assert 1 not in trimmed, trimmed
    ends = lambda xs: [sum(xs[: i + 1]) for i in range(len(xs))]
    assert ends(trimmed) == ends(today)[:-1] and ends(trimmed)[-1] == N - 1


def test_the_split_is_identity_for_a_trimmed_request():
    sp = _schedule_policy(armed=True)
    adder = SimpleNamespace(rem_chunk_tokens=CHUNK)
    ids, tail = pt.split_ids(_recv(PROMPT))
    req = SimpleNamespace(full_untruncated_fill_ids=list(ids), rid=RID)
    setattr(req, pt.TRIM_ATTR, tail)
    assert sp.PrefillAdder._weg2_end_anchor_split(adder, req, 20, 2) == (2, False)
    plain = SimpleNamespace(full_untruncated_fill_ids=list(PROMPT), rid=RID)
    assert sp.PrefillAdder._weg2_end_anchor_split(adder, plain, 20, 3) == (2, True)


# -- D. the real tree: D's resume point ----------------------------------------

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
    # UNIFY S7/S8: the 27B form -- its profile keeps the upstream bigram anchor
    # keying (bigram_anchor_exact off) and its store-short tail on.
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


def _req(fx, ids, tail=None):
    req = Req(rid=RID, origin_input_text="", origin_input_ids=array("q", ids),
              sampling_params=SamplingParams(temperature=0, max_new_tokens=1))
    if tail is not None:
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


def _prefill_today(fx):
    """#1233 END-ANCHOR split: the last regular chunk ends at N-1 (the chunk
    publish writes the anchor), the held token runs as a 1-token chunk."""
    req = _req(fx, PROMPT)
    for end in range(CHUNK, N - 1, CHUNK):
        _chunk(fx, req, end)
    _set_state(fx, req, S1)
    _chunk(fx, req, N - 1)
    _set_state(fx, req, S2)                 # the held-back token's update
    _finish(fx, req)                        # the 1-token END-ANCHOR forward
    return req


def _prefill_trimmed(fx):
    """P's intake decision (the real split_ids on a leg-1 request), then the
    prefill of what it admitted: the last chunk is the finish."""
    ids, tail = pt.split_ids(_recv(PROMPT))
    ids, tail = list(ids), list(tail)
    req = _req(fx, ids, tail)
    for end in range(CHUNK, len(ids), CHUNK):
        _chunk(fx, req, end)
    _set_state(fx, req, S1)
    _finish(fx, req)                        # the finish insert IS the N-1 anchor
    return req


def _d_resume(fx):
    """What D's admission claims: the first N-1 tokens (max prefix), the
    mamba validator included -- the probe the #1481 instrument runs."""
    claim = Req._compute_max_prefix_len(SimpleNamespace(return_logprob=False, logprob_start_len=0), N)
    mr = fx.cache.match_prefix(MatchPrefixParams(key=_key(fx, PROMPT[:claim])))
    node = mr.last_device_node
    slot = node.component_data[ComponentType.MAMBA].value
    state = None if slot is None else float(
        fx.pool.mamba_pool.mamba_cache.temporal[:, slot].float().mean())
    return len(mr.device_indices), node, state


def _deepest(fx):
    stack, best = [(c, len(c.key)) for c in fx.cache.root_node.children.values()], 0
    while stack:
        node, depth = stack.pop()
        best = max(best, depth)
        stack.extend((c, depth + len(c.key)) for c in node.children.values())
    return best


@pytest.mark.parametrize("bigram", [False, True], ids=["unigram", "bigram"])
def test_d_resumes_from_the_same_point_with_the_n_minus_1_state_and_mark(group_p, bigram):
    today = _fixture(bigram)
    _prefill_today(today)
    t_depth, t_node, t_state = _d_resume(today)

    trim = _fixture(bigram)
    _prefill_trimmed(trim)
    r_depth, r_node, r_state = _d_resume(trim)

    claim_units = (N - 1) - (1 if bigram else 0)
    assert t_depth == r_depth == claim_units, (t_depth, r_depth)
    assert t_state == pytest.approx(S1) and r_state == pytest.approx(S1)
    assert getattr(t_node, "_weg2_end_anchor", False), "today's #1481 mark"
    assert getattr(r_node, "_weg2_end_anchor", False), "the trimmed request's final node is the end anchor"
    assert _deepest(trim) == claim_units, "nothing deeper than N-1 on P: no 1-token insert"
    assert _deepest(today) == claim_units + 1


def test_the_end_anchor_line_reports_n_and_the_trim(group_p, caplog):
    fx = _fixture(True)
    with caplog.at_level(logging.WARNING):
        _prefill_trimmed(fx)
    lines = [r.getMessage() for r in caplog.records if "WEG2 END-ANCHOR n=" in r.getMessage()]
    assert lines, "the #1233 instrument still speaks once per finished request"
    assert f"tokens={N} anchor={N - 1} target={N - 1}" in lines[-1] and "ok=True" in lines[-1]
    assert lines[-1].endswith("trim=1")


# -- E./F. what P tells the outside: prompt_tokens and the #1442 hand-off -----


def test_p_still_reports_prompt_tokens_n():
    from sglang.srt.managers.scheduler_components.output_streamer import (
        _GenerationStreamAccumulator,
    )
    from sglang.srt.speculative.spec_info import SpeculativeAlgorithm
    from sglang.srt.disaggregation.utils import DisaggregationMode

    req = Req(rid=RID, origin_input_text="", origin_input_ids=array("q", PROMPT[:-1]),
              sampling_params=SamplingParams(temperature=0, max_new_tokens=1))
    setattr(req, pt.TRIM_ATTR, array("q", PROMPT[-1:]))
    req.output_ids = array("q", [7])
    req.finished_reason = FINISH_LENGTH(length=1)
    acc = _GenerationStreamAccumulator(
        return_logprob=False, return_hidden_states=False, return_routed_experts=False,
        return_indexer_topk=False, spec_algorithm=SpeculativeAlgorithm.NONE,
        disaggregation_mode=DisaggregationMode.NULL, default_stream_interval=1,
        default_force_stream_interval=1, get_cached_tokens_details=lambda r: None,
    )
    acc.accept(req=req)
    assert acc.prompt_tokens == [N]
    assert pt.full_prompt_len(req) == N and pt.full_prompt_ids(req) == PROMPT


def test_the_handoff_gives_d_all_n_ids_and_the_trimmed_path_keys(group_p, tmp_path):
    group_p.setenv("SGLANG_HICACHE_ARENA_DIR", str(tmp_path))
    from sglang.srt.weg2 import handoff as ho

    fx = _fixture(True)
    req = _prefill_trimmed(fx)
    mr = fx.cache.match_prefix(MatchPrefixParams(key=_key(fx, list(req.origin_input_ids))))
    node, k = mr.last_device_node, 0
    while node is not None and node is not fx.cache.root_node:   # keys the store would carry
        node.hash_value = [f"h{k + i}" for i in range(len(node.key))]
        k += len(node.key)
        node = node.parent
    fx.cache.enable_storage = True
    fx.cache._weg2_handoff_write(req, _key(fx, list(req.origin_input_ids)))
    rec = json.load(open(ho.path(RID)))
    assert rec["input_ids"] == PROMPT, "D's tokenizer takes these as its prompt"
    assert len(rec["page_keys"]) == N - 2        # bigram units of [0, N-1): D's whole claim


# -- G. the launcher switch -----------------------------------------------------


def test_the_launcher_switch_is_p_only_and_default_off():
    from sglang.srt.weg2 import launcher as L

    base = ["--tree", "/t", "--tag", "t"]
    ns = L.build_parser().parse_args(base)
    assert ns.p_trim_end_anchor is False
    assert L.p_trim_end_anchor_env(False) == {}
    assert L.p_trim_end_anchor_env(True) == {pt.TRIM_ENV: "1"}
    ns_on = L.build_parser().parse_args(base + ["--p-trim-end-anchor"])
    assert ns_on.p_trim_end_anchor is True
