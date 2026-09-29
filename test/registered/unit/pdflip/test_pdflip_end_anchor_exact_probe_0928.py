"""END-ANCHOR (W123 / #241): the #1233 probe asks with the insert's key form.

MEASURED. rc12z10 P (boot ...09280831, Z. 19761-19776, rid pdflip-2-12, N=9537,
one extend chunk): ``#1469 RETAIN ... token_ids_len=9537 cache_len=9536``, then
``PDFLIP END-ANCHOR ... tokens=9537 anchor=0 target=9536 units=0/9472 ok=False``
and ``MAMBA-ARENA ... deepest=9536 end_anchor=none``; D found no N-1 anchor and
died ``W123 PdFlipVisionImageNotInPrefixOnD covered=0``. #241 (rc12u, pdflip-6-18,
N=23361, two chunks): ``anchor=16384 target=23360 units=16384/23296 ok=False``.
Over every NF P log: N % 64 == 1 -> 105 ok=False / 3 ok=True, every other N
7764 ok=True / 0 ok=False.

ROOT. ``cache_finished_req`` files the N-1 node with the EXACT bigram key
(``bigram_anchor_key``, N-1 units); the probe built the upstream slice
``token_ids[:-1]`` (N-2 units). Page alignment of N-2 units drops a whole page
exactly when N-1 is a page multiple (9535 -> 9472), so the probe never reached
the N-1 node and the mamba validator fell back to the anchor below it.

A. The REAL ``UnifiedRadixCache`` (real MambaComponent, pools, allocator; the
   P-TRIM harness shape) with the NF keying, through the shipped
   ``cache_unfinished_req`` / ``cache_finished_req``. The harness runs at
   page_size 1 (upstream MambaComponent refuses page > 1 without the extra
   buffer); at page 1 every N is "1 mod page", and the base reproduces both
   metal signatures: one chunk -> anchor=0, two chunks -> the first chunk.
B. The REAL ``_pdflip_note_end_anchor`` at page_size 64 on a tree whose anchors
   sit at the node ends the real ``bigram_anchor_key`` files (chunk ends and
   the N-1 retain), for N in {65, 1025, 9537, 23361} and the two metal rids.

RED on 82fa502795 (A: all; B: every N = 1 mod 64), GREEN with the fix.
Controls (N not 1 mod 64, upstream keying, the P-TRIM form) are identical on
both.
"""

import logging
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
from flliper.srt.mem_cache import unified_radix_cache as urc
from flliper.srt.mem_cache.unified_radix_cache import UnifiedRadixCache
from flliper.srt.sampling.sampling_params import SamplingParams
from flliper.srt.server_args import ServerArgs, set_global_server_args_for_scheduler
from flliper.srt.pdflip import p_trim_end_anchor as pt

PAGE = 1
KV_SIZE = 512
NUM_LAYERS = 4
FULL_LAYER_IDS = (1, 3)
MAMBA_LAYER_IDS = [i for i in range(NUM_LAYERS) if i not in FULL_LAYER_IDS]
S_END = 1.25


@pytest.fixture
def armed(monkeypatch):
    monkeypatch.setattr(urc, "_PDFLIP_END_ANCHOR", True)
    return monkeypatch


def _fixture(exact=True, bigram=True):
    server_args = ServerArgs(model_path="dummy", page_size=PAGE)
    server_args._mamba_cache_chunk_size = FLA_CHUNK_SIZE
    set_global_server_args_for_scheduler(server_args)
    with envs.FLLIPER_MAMBA_SSM_DTYPE.override("bfloat16"):
        shape = Mamba2StateShape.create(
            tp_world_size=1, intermediate_size=64, n_groups=1, num_heads=1,
            head_dim=16, state_size=8, conv_kernel=4,
        )
        cache_params = Mamba2CacheParams(shape=shape, layers=MAMBA_LAYER_IDS)
    pool = HybridReqToTokenPool(
        size=4, mamba_size=8, mamba_spec_state_size=4, max_context_len=KV_SIZE,
        device="cpu", enable_memory_saver=False, cache_params=cache_params,
        mamba_layer_ids=MAMBA_LAYER_IDS, enable_mamba_extra_buffer=False,
        speculative_num_draft_tokens=3,
    )
    kv_pool = HybridLinearKVPool(
        size=KV_SIZE, dtype=torch.bfloat16, page_size=1, head_num=1, head_dim=8,
        full_attention_layer_ids=list(FULL_LAYER_IDS), device="cpu",
        enable_memory_saver=False, mamba_pool=pool.mamba_pool,
    )
    allocator = TokenToKVPoolAllocator(
        size=KV_SIZE, dtype=torch.bfloat16, device="cpu", kvcache=kv_pool, need_sort=False,
    )
    params = CacheInitParams(
        req_to_token_pool=pool, token_to_kv_pool_allocator=allocator, page_size=PAGE,
        disable=False, sliding_window_size=None,
        tree_components=(ComponentType.FULL, ComponentType.MAMBA),
        enable_mamba_extra_buffer=False, enable_kv_cache_events=False,
        eviction_policy="lru", is_eagle=bigram,
    )
    cache = UnifiedRadixCache(params=params)
    cache.cache_init_params = params
    # The keying is the profile's (nextflash on, qwen27b off); pinned per tree.
    cache.__dict__["_pdflip_bigram_anchor_exact"] = bool(exact and bigram)
    return SimpleNamespace(cache=cache, allocator=allocator, pool=pool)


def _req(fx, ids, rid, tail=None):
    req = Req(rid=rid, origin_input_text="", origin_input_ids=array("q", ids),
              sampling_params=SamplingParams(temperature=0, max_new_tokens=1))
    if tail is not None:
        setattr(req, pt.TRIM_ATTR, array("q", tail))
    fx.pool.alloc([req])
    req.output_ids = array("q")
    req.full_untruncated_fill_ids = array("q", ids)
    req.swa_uuid_for_lock = None
    req.extra_key = None
    req.prefix_indices = torch.empty(0, dtype=torch.int64)
    req.cache_protected_len = 0
    req.last_node = fx.cache.root_node
    fx.cache.inc_lock_ref(req.last_node)
    return req


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
    fx.pool.mamba_pool.mamba_cache.temporal[:, req.mamba_pool_idx] = S_END
    fx.cache.cache_finished_req(req, is_insert=True)


def _prefill(fx, n, rid, chunks=(), trim=False):
    """One finished prompt of ``n`` tokens, chunk ends ``chunks`` before it."""
    prompt = list(range(1000, 1000 + n))
    ids, tail = (prompt[:-1], prompt[-1:]) if trim else (prompt, None)
    req = _req(fx, ids, rid, tail)
    for end in chunks:
        _chunk(fx, req, end)
    _finish(fx, req)
    return prompt


def _line(caplog, rid):
    lines = [r.getMessage() for r in caplog.records
             if "PDFLIP END-ANCHOR n=" in r.getMessage() and f"rid={rid[:12]} " in r.getMessage()]
    assert len(lines) == 1, lines
    return lines[0]


def _n1_node(fx, prompt):
    """The node the N-1 insert filed (exact key: N-1 units)."""
    mr = fx.cache.match_prefix(MatchPrefixParams(
        key=RadixKey(array("q", prompt), is_bigram=fx.cache.is_eagle).page_aligned(PAGE)))
    return mr.last_device_node, len(mr.device_indices)


def _assert_exact_anchor(fx, caplog, prompt, rid):
    n = len(prompt)
    line = _line(caplog, rid)
    assert f"tokens={n} anchor={n - 1} target={n - 1} units={n - 1}/{n - 1} ok=True" in line, line
    node, depth = _n1_node(fx, prompt)
    assert depth == n - 1
    assert getattr(node, "_pdflip_end_anchor", False), "the #1481 mark on the N-1 node"


# -- A. the real tree (page 1, NF keying) ------------------------------------


@pytest.mark.parametrize("n, chunks", [(23, ()), (40, (16,)), (65, (16, 32, 48))],
                         ids=["one-chunk-rc12z10-form", "two-chunks-241-form", "four-chunks"])
def test_real_tree_probe_reaches_the_n_minus_1_node(armed, caplog, n, chunks):
    fx = _fixture()
    rid = f"pdflip-a-{n}"
    with caplog.at_level(logging.WARNING):
        prompt = _prefill(fx, n, rid, chunks=chunks)
    _assert_exact_anchor(fx, caplog, prompt, rid)


def test_real_tree_upstream_keying_is_unchanged(armed, caplog):
    """27B profile (exact off): the probe key is the old slice and the answer
    stays N-2 units (the P-TRIM pin's claim)."""
    fx = _fixture(exact=False)
    with caplog.at_level(logging.WARNING):
        _prefill(fx, 23, "pdflip-a-27b", chunks=(8, 16, 22))  # the #1233 split
    line = _line(caplog, "pdflip-a-27b")
    assert "tokens=23 anchor=22 target=22 units=21/21 ok=True" in line, line


# -- B. the real probe at page 64 ----------------------------------------------

PAGE64 = 64


def _page64(n, chunks=(), exact=True, trim=False):
    """The real probe on a page-64 tree whose recurrent anchors sit at the node
    ends the real inserts file: every chunk end and the finished request's
    retain (``#1469 RETAIN ... cache_len``), each keyed by ``bigram_anchor_key``
    exactly as ``cache_unfinished_req`` / ``cache_finished_req`` key them.
    The match returns the deepest anchor at or below the probe (the mamba
    validator's answer)."""
    prompt = array("q", range(1000, 1000 + n))
    ids = prompt[:-1] if trim else prompt

    def key_units(cache_len):
        return len(urc.bigram_anchor_key(ids, cache_len, None, is_bigram=True,
                                         exact=exact, page_size=PAGE64))

    retain = len(ids) if trim else n - 1
    anchors = sorted({key_units(c) for c in chunks} | {key_units(retain)})
    root = SimpleNamespace(name="root")
    nodes = {a: SimpleNamespace(name=f"node@{a}") for a in anchors}
    probes = []

    def match_prefix(params):
        units = len(params.key)
        probes.append(units)
        depth = max([a for a in anchors if a <= units], default=0)
        return SimpleNamespace(device_indices=torch.empty(depth),
                               last_device_node=nodes.get(depth, root))

    tree = SimpleNamespace(is_eagle=True, page_size=PAGE64, bigram_anchor_exact=exact,
                           match_prefix=match_prefix, root_node=root)
    req = SimpleNamespace(rid=f"pdflip-b-{n}", extra_key=None)
    if trim:
        setattr(req, urc._P_TRIM_ATTR, prompt[-1:])
    UnifiedRadixCache._pdflip_note_end_anchor(tree, req, ids)
    return nodes, anchors, probes


def _b_line(caplog, n):
    return _line(caplog, f"pdflip-b-{n}")


@pytest.mark.parametrize("n", [65, 1025, 9537, 23361])
def test_page64_n_equal_1_mod_page_marks_the_n_minus_1_node(caplog, n):
    with caplog.at_level(logging.WARNING):
        nodes, anchors, probes = _page64(n)
    assert anchors[-1] == n - 1
    assert probes == [n - 1], "the probe asks with the insert's N-1 units"
    line = _b_line(caplog, n)
    assert f"tokens={n} anchor={n - 1} target={n - 1} units={n - 1}/{n - 1} ok=True" in line, line
    assert getattr(nodes[n - 1], "_pdflip_end_anchor", False), "the #1481 mark"


def test_page64_rc12z10_pdflip_2_12_one_chunk(caplog):
    with caplog.at_level(logging.WARNING):
        nodes, _a, _p = _page64(9537)
    line = _b_line(caplog, 9537)
    assert "anchor=9536 target=9536 units=9536/9536 ok=True" in line, line
    assert getattr(nodes[9536], "_pdflip_end_anchor", False)


def test_page64_rc12u_pdflip_6_18_two_chunks(caplog):
    with caplog.at_level(logging.WARNING):
        nodes, anchors, _p = _page64(23361, chunks=(16384,))
    assert anchors == [16384, 23360]
    line = _b_line(caplog, 23361)
    assert "anchor=23360 target=23360 units=23360/23360 ok=True" in line, line
    assert getattr(nodes[23360], "_pdflip_end_anchor", False)
    assert not getattr(nodes[16384], "_pdflip_end_anchor", False)


# -- controls: identical before and after the fix ------------------------------


def test_page64_control_n_not_1_mod_page(caplog):
    """pdflip-2-9 (rc12z10 Z. 19736): N=1055 -> units=1024/1024 ok=True."""
    with caplog.at_level(logging.WARNING):
        _nodes, _a, probes = _page64(1055)
    assert probes == [1024]
    assert "units=1024/1024 ok=True" in _b_line(caplog, 1055)


def test_page64_control_upstream_keying_is_unchanged(caplog):
    """27B profile (exact off): probe = the old slice (N-1 tokens, N-2 units)."""
    with caplog.at_level(logging.WARNING):
        _nodes, _a, probes = _page64(1025, exact=False)
    assert probes == [960]
    assert "tokens=1025 anchor=961 target=1024 units=960/960 ok=True" in _b_line(caplog, 1025)


def test_page64_control_p_trim_form_is_unchanged(caplog):
    with caplog.at_level(logging.WARNING):
        _nodes, _a, probes = _page64(1025, trim=True)
    assert probes == [960]
    line = _b_line(caplog, 1025)
    assert "tokens=1025 anchor=960 target=1024 units=960/960 ok=True" in line, line
    assert line.endswith("trim=1")
