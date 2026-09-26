"""xsn437 (weg2xsn437, --p-trim-end-anchor on): two leg-2 requests answered 503.

  (a) rid weg2-6-2, N=4316: D delivered=4095 deliverable=4314 shortfall=219,
      the re-read refused (#915 vote_negative need=219 threshold=256), W88.
  (b) rid weg2-20-37, N=2050: delivered=2047 deliverable=2048 shortfall=1,
      HOLD-REFETCH declined:too_short, W88 after 64 passes, 503.

THE PUBLISH WAS WHOLE. P's own lines say so (END-ANCHOR units=4314/4314 and
2048/2048 ok=True trim=1, #1442 HANDOFF page_keys=4314 = D's covered=4314):
the trimmed request's chain covers exactly D's deliverable. Its LAST node is
published at the retain (RETAIN-PUBLISH issued=1 pending=1), i.e. at the
moment P answers leg 1 -- without the trim the N-1 node went out through the
chunk publish one forward earlier -- so D's first read landed before it:
4095 = the chunk-published prefix up to the 4096 interval anchor, 2047 = the
chunk-published prefix up to the boundary 2048. That is a race the
#1324 deferral exists for. What failed is D: the owed re-read of a tail below
the prefetch threshold is refused, so the chain can only stand still, and the
standstill answered 503 for remainders of 221 and 3 tokens.

Hermetic, CPU. Pinned, each on the metal numbers:
  * P (real tree, 512 chunks, anchor interval 4096, bigram): the trimmed
    chain's units equal D's deliverable; before the final node the published,
    resumable prefix is 4095 (a, the interval anchor) and 2047 (b, the
    boundary 2048 is the trimmed request's own N'-1 'end' step) -- the metal
    reads, short of the span by 219 and 1;
  * D: a read that completes a short store read is issued below the
    threshold (219 and 1 tokens); a fresh read keeps the threshold;
  * D: a store-short read that stands still with a remainder within X is
    admitted (D prefills it) instead of W88; over X (the sn6s form) it stays
    the named W88; SGLANG_WEG2_STORE_SHORT_TAIL=0 restores the old forms.
"""

import ast
import inspect
import logging
import textwrap
import types
from array import array
from types import SimpleNamespace

import pytest
import torch

from sglang.srt.configs.mamba_utils import Mamba2CacheParams, Mamba2StateShape
from sglang.srt.environ import envs
from sglang.srt.layers.attention.fla.chunk_delta_h import CHUNK_SIZE as FLA_CHUNK_SIZE
from sglang.srt.managers import scheduler as sched_mod
from sglang.srt.managers.schedule_batch import Req
from sglang.srt.mem_cache.allocator import TokenToKVPoolAllocator
from sglang.srt.mem_cache.base_prefix_cache import MatchPrefixParams
from sglang.srt.mem_cache.cache_init_params import CacheInitParams
from sglang.srt.mem_cache.hicache_storage import PrefetchOutcome
from sglang.srt.mem_cache.mamba_ckpt_utils import ANCHOR_INTERVAL_ENV
from sglang.srt.mem_cache.memory_pool import HybridLinearKVPool, HybridReqToTokenPool
from sglang.srt.mem_cache.radix_cache import RadixKey
from sglang.srt.mem_cache.unified_cache_components.tree_component import ComponentType
from sglang.srt.mem_cache.unified_radix_cache import UnifiedRadixCache
from sglang.srt.sampling.sampling_params import SamplingParams
from sglang.srt.server_args import ServerArgs, set_global_server_args_for_scheduler
from sglang.srt.weg2 import p_trim_end_anchor as pt

CHUNK = 512
INTERVAL = 4096
X = 4096
THRESHOLD = 256
#: (N, delivered on the first read, D's deliverable) as measured on weg2xsn437
METAL_A = (4316, 4095, 4314)     # rid weg2-6-2, the 4096 interval anchor
METAL_B = (2050, 2047, 2048)     # rid weg2-20-37, bigram, last chunk of 1 token
SN6S = (109132, 53247, 109131)   # the #1324 form: remainder 55,885 over X=11,101
SN6S_X = 11101


# -- P: the trimmed chain against D's deliverable (real tree) ------------------

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
    monkeypatch.setenv(ANCHOR_INTERVAL_ENV, str(INTERVAL))
    monkeypatch.setattr(urc, "_WEG2_END_ANCHOR", True)
    return monkeypatch


def _fixture(kv_size=8192):
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
        size=4, mamba_size=24, mamba_spec_state_size=4, max_context_len=kv_size,
        device="cpu", enable_memory_saver=False, cache_params=cache_params,
        mamba_layer_ids=MAMBA_LAYER_IDS, enable_mamba_extra_buffer=False,
        speculative_num_draft_tokens=3,
    )
    kv_pool = HybridLinearKVPool(
        size=kv_size, dtype=torch.bfloat16, page_size=1, head_num=1, head_dim=8,
        full_attention_layer_ids=list(FULL_LAYER_IDS), device="cpu",
        enable_memory_saver=False, mamba_pool=pool.mamba_pool,
    )
    allocator = TokenToKVPoolAllocator(
        size=kv_size, dtype=torch.bfloat16, device="cpu", kvcache=kv_pool, need_sort=False,
    )
    params = CacheInitParams(
        req_to_token_pool=pool, token_to_kv_pool_allocator=allocator, page_size=1,
        disable=False, sliding_window_size=None,
        tree_components=(ComponentType.FULL, ComponentType.MAMBA),
        enable_mamba_extra_buffer=False, enable_kv_cache_events=False,
        eviction_policy="lru", is_eagle=True,        # group P: SGLANG_HICACHE_BIGRAM_KEYS=1
    )
    cache = UnifiedRadixCache(params=params)
    cache.cache_init_params = params
    return SimpleNamespace(cache=cache, allocator=allocator, pool=pool)


def _key(fx, ids):
    return RadixKey(array("q", ids), is_bigram=True)


def _p_leg1(fx, n):
    """P's intake (the real trim) and the prefill of what it admitted, 512
    chunks; returns (req, resumable units before the finish, KV units
    before the finish, resumable units after it)."""
    prompt = list(range(10_000, 10_000 + n))
    recv = SimpleNamespace(rid="weg2-7-1", input_ids=array("q", prompt), input_embeds=None,
                           return_logprob=False, sampling_params=SamplingParams(max_new_tokens=1),
                           session_params=None, session_id=None, mm_inputs=None)
    ids, tail = pt.split_ids(recv)
    ids = list(ids)
    req = Req(rid="weg2-7-1", origin_input_text="", origin_input_ids=array("q", ids),
              sampling_params=SamplingParams(temperature=0, max_new_tokens=1))
    setattr(req, pt.TRIM_ATTR, array("q", list(tail)))
    fx.pool.alloc([req])
    req.output_ids = array("q")
    req.full_untruncated_fill_ids = array("q", ids)
    req.swa_uuid_for_lock = None
    req.extra_key = None
    req.prefix_indices = torch.empty(0, dtype=torch.int64)
    req.cache_protected_len = 0
    req.last_node = fx.cache.root_node
    for end in range(CHUNK, len(ids), CHUNK):
        start = len(req.prefix_indices)
        fx.pool.write((req.req_pool_idx, slice(start, end)), fx.allocator.alloc(end - start))
        req.set_extend_range(start, end)
        fx.cache.cache_unfinished_req(req, chunked=True)
    d_claim = prompt[: n - 1]                                   # D's max prefix, N-1 raw
    before = fx.cache.match_prefix(MatchPrefixParams(key=_key(fx, d_claim)))
    kv_units_before = _path_units(fx, d_claim)
    start = len(req.prefix_indices)
    fx.pool.write((req.req_pool_idx, slice(start, len(ids))), fx.allocator.alloc(len(ids) - start))
    req.set_extend_range(start, len(ids))
    req.kv_committed_len = len(ids)
    req.kv_allocated_len = len(ids)
    fx.cache.cache_finished_req(req, is_insert=True)
    after = fx.cache.match_prefix(MatchPrefixParams(key=_key(fx, d_claim)))
    return req, len(before.device_indices), kv_units_before, len(after.device_indices)


def _path_units(fx, ids):
    """KV units on the tree along ``ids`` (mamba anchors or not)."""
    node, units, key = fx.cache.root_node, 0, _key(fx, ids)
    while True:
        child = None
        for c in node.children.values():
            ck = list(c.key)
            if list(key[units: units + len(ck)]) == ck:
                child = c
                break
        if child is None:
            return units
        units += len(child.key)
        node = child


@pytest.mark.parametrize("metal", [METAL_A, METAL_B], ids=["weg2-6-2_N4316", "weg2-20-37_N2050"])
def test_the_trimmed_chain_covers_d_s_deliverable_and_the_first_read_is_the_chunk_part(group_p, metal):
    n, first_read, deliverable = metal
    fx = _fixture()
    _req, resumable_before, kv_before, resumable_after = _p_leg1(fx, n)
    assert resumable_after == deliverable, "the publish is WHOLE: [0, N-1) in D's units"
    assert kv_before == first_read, "what the chunk publish put out before the finish = D's read"
    # (a) the 4096 interval anchor; (b) the chunk boundary 2048 is the trimmed
    # request's own N'-1 (weg2_anchor_step 'end': pos >= prompt_len - 1), so
    # the 2047 D read were resumable -- the read was short of its span all the same
    assert resumable_before == first_read
    assert deliverable - first_read == {METAL_A: 219, METAL_B: 1}[metal], "the metal shortfalls"


def test_n_2049_the_operator_s_form_also_covers_the_deliverable(group_p):
    fx = _fixture()
    _req, _rb, _kb, resumable_after = _p_leg1(fx, 2049)
    assert resumable_after == 2049 - 2           # bigram units of N-1 raw tokens


# -- D: the read that completes a short store read ------------------------------


class _Issued(Exception):
    """Raised where the read passes the gate and starts allocating."""


def _gate_cache():
    cache = object.__new__(UnifiedRadixCache)
    cache.enable_storage = True
    cache.page_size = 1
    cache.is_eagle = True
    cache.prefetch_threshold = THRESHOLD
    cache.cache_controller = SimpleNamespace(prefetch_rate_limited=lambda: False)
    cache._hicache_prefetch_symmetric = lambda: False
    cache.refused = []
    cache._log_prefetch_refused = lambda reason, rid, n: cache.refused.append((reason, n))

    def _stop(node):
        raise _Issued()

    cache.inc_host_lock_ref = _stop
    return cache


@pytest.mark.parametrize("missing_units", [219, 1], ids=["weg2-6-2_tail219", "weg2-20-37_tail1"])
def test_a_completion_read_below_the_threshold_is_issued(missing_units):
    cache = _gate_cache()
    tail = list(range(missing_units + 1))           # bigram: units + 1 raw tokens
    node = SimpleNamespace(key=None)
    cache.prefetch_from_storage("weg2-6-2", node, tail)
    assert cache.refused == [("too_short", missing_units)], "a fresh read keeps the threshold"
    with pytest.raises(_Issued):
        cache.prefetch_from_storage("weg2-6-2", node, tail, min_tokens=1)


def test_the_completion_minimum_follows_the_short_read_stamp(monkeypatch):
    req = SimpleNamespace()
    assert sched_mod._weg2_store_tail_min_tokens(req) is None
    req._weg2_store_delivered = 4095
    assert sched_mod._weg2_store_tail_min_tokens(req) == 1
    monkeypatch.setenv("SGLANG_WEG2_STORE_SHORT_TAIL", "0")
    assert sched_mod._weg2_store_tail_min_tokens(req) is None


def test_both_prefetch_calls_carry_the_completion_minimum():
    src = textwrap.dedent(inspect.getsource(sched_mod.Scheduler._prefetch_kvcache))
    calls = [n for n in ast.walk(ast.parse(src)) if isinstance(n, ast.Call)
             and getattr(n.func, "attr", None) == "prefetch_from_storage"]
    assert len(calls) == 2
    for c in calls:
        assert any(k.arg is None and getattr(k.value, "id", None) == "_tail_kw" for k in c.keywords)


# -- D: the standstill of a store-short read --------------------------------------


def _sched(outcome, n_tokens, x):
    s = types.SimpleNamespace()
    s.tree_cache = types.SimpleNamespace(
        prefetch_loaded_tokens_by_reqid={"r1": outcome},
        ongoing_prefetch={},
        prefetch_timeout_base=1.0,
        prefetch_timeout_per_page=0.01,
        page_size=1,
        cache_controller=types.SimpleNamespace(host_role="staging"),
        release_aborted_request=lambda rid: None,
    )
    s.server_args = types.SimpleNamespace(tp_prefill_max_tokens=x)
    s.enable_hicache_storage = True
    s.enable_hierarchical_cache = False
    s.waiting_queue = []
    s.ipc_channels = types.SimpleNamespace(
        send_to_tokenizer=types.SimpleNamespace(send_output=lambda *a, **k: None))
    for name in (
        "_weg2_note_store_shortfall",
        "_apply_prefetch_deferral",
        "_apply_group_shortfall_deferral",
        "_weg2_store_read_is_pending",
        "_weg2_note_prefetch_progress",
        "_weg2_prefetch_progress_terms",
        "_weg2_prefetch_stall_passes",
        "_weg2_windowed_store_read_active",
        "_weg2_store_load_terminal",
        "_prefetch_deferral_refusal_reason",
        "_prefetch_capacity_limit_or_none",
        "_clear_prefetch_deferral_fields",
    ):
        setattr(s, name, getattr(sched_mod.Scheduler, name).__get__(s))
    r = types.SimpleNamespace(
        rid="r1", prefetch_deferred=None, _prefetch_span_tokens=n_tokens - 1,
        prefix_indices=None, host_hit_length=0,
        full_untruncated_fill_ids=list(range(n_tokens)),
        time_stats=types.SimpleNamespace(trace_ctx=types.SimpleNamespace(abort=lambda **k: None)),
    )
    s.waiting_queue = [r]
    return s, r


def _stand_still(s, r):
    outcomes = []
    for _ in range(s._weg2_prefetch_stall_passes() + 3):
        outcomes.append(s._weg2_note_store_shortfall(r))
        if outcomes[-1] in ("failed", "expired"):
            break
    return outcomes[-1]


@pytest.mark.parametrize("metal", [METAL_A, METAL_B], ids=["weg2-6-2_rem221", "weg2-20-37_rem3"])
def test_a_store_short_standstill_within_x_is_recomputed_not_503(metal, caplog):
    n, delivered, deliverable = metal
    s, r = _sched(PrefetchOutcome(delivered, matched=0, deliverable=deliverable, synced=delivered), n, X)
    with caplog.at_level(logging.WARNING, logger=sched_mod.logger.name):
        verdict = _stand_still(s, r)
    assert verdict == "expired", f"a remainder of {n - delivered} within X={X} must be recomputed"
    assert r in s.waiting_queue and r.prefetch_deferred is None
    assert "STORE-SHORT TAIL RECOMPUTE" in caplog.text and f"remainder={n - delivered}" in caplog.text
    assert "W88 Weg2StoreLoadNotProgressing" not in caplog.text, "genuine marker (#995: the recompute line names W88 in prose)"


def test_over_x_the_standstill_stays_the_named_w88(caplog):
    n, delivered, deliverable = SN6S
    s, r = _sched(PrefetchOutcome(delivered, matched=0, deliverable=deliverable, synced=delivered),
                  n, SN6S_X)
    with caplog.at_level(logging.WARNING, logger=sched_mod.logger.name):
        verdict = _stand_still(s, r)
    assert verdict == "failed" and "W88 Weg2StoreLoadNotProgressing" in caplog.text
    assert r not in s.waiting_queue, "the user's veto: never a prefill over X"


def test_the_switch_off_restores_the_w88(monkeypatch, caplog):
    monkeypatch.setenv("SGLANG_WEG2_STORE_SHORT_TAIL", "0")
    n, delivered, deliverable = METAL_A
    s, r = _sched(PrefetchOutcome(delivered, matched=0, deliverable=deliverable, synced=delivered), n, X)
    with caplog.at_level(logging.WARNING, logger=sched_mod.logger.name):
        verdict = _stand_still(s, r)
    assert verdict == "failed" and "W88" in caplog.text
