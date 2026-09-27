"""PARK-RETAIN READ (27.09.): a flip-parked D request reads back exactly what its
park retained (weg2/d_park_read.py), and the controller honours a read's own
floor (managers/weg2_min_hit.py).

Metal: 27B dkr27bparkdraftbar1w209270645 weg2-2-5 (``#1469 RETAIN
token_ids_len=417 cache_len=256``, 255 pages stored = one under the 256
threshold, four zero answers, 20.1 s settle, 418 tokens prefilled from 0); NF
dkrnfh91bar1dauer09270859 weg2-1-13 (47262 -> 47104 retained, the read asked
47232, 128 short forever, 20.0 s settle).

The real tree (UnifiedRadixCache + MambaComponent, the #924/#773 harness shape)
proves the stamp is what the insert kept; the park runs against the h91b
stand-in scheduler; ``scheduler.py`` / the controller are READ by the wiring
ratchets (their import pulls the GPU stack)."""
from __future__ import annotations

import os
import types
from array import array
from collections import deque

import pytest

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

from sglang.srt.managers import weg2_min_hit as mh  # noqa: E402
from sglang.srt.mem_cache.radix_cache import RadixKey  # noqa: E402
from sglang.srt.weg2 import d_park_read as pr  # noqa: E402
from sglang.srt.weg2 import d_park_runtime as rt  # noqa: E402
from sglang.srt.weg2 import d_seats as ds  # noqa: E402

_SRT = os.path.dirname(os.path.dirname(ds.__file__))


def _read(*parts):
    with open(os.path.join(_SRT, *parts), encoding="utf-8") as f:
        return f.read()


@pytest.fixture(autouse=True)
def _env(monkeypatch):
    monkeypatch.setenv("SGLANG_WEG2_GROUP", "D")
    monkeypatch.delenv(ds.PARK_ENV, raising=False)
    monkeypatch.delenv(pr.ENV, raising=False)


# -- the retained span ---------------------------------------------------------


def test_retained_raw_tokens_counts_the_raw_span_of_the_inserted_key():
    ids = array("q", range(417))
    # bigram (27B DFlash / NF MTP): 255 units span 256 raw tokens
    assert pr.retained_raw_tokens(RadixKey(ids[:256], is_bigram=True)) == 256
    assert len(RadixKey(ids[:256], is_bigram=True)) == 255
    # plain keys: units == raw tokens
    assert pr.retained_raw_tokens(RadixKey(ids[:256])) == 256
    assert pr.retained_raw_tokens(RadixKey(array("q"), is_bigram=True)) == 0
    assert pr.retained_raw_tokens(RadixKey(ids[:1], is_bigram=True)) == 0


def test_the_read_span_of_the_cap_is_exactly_the_stored_units():
    """The store read of [0, cap) raw tokens asks exactly the units the park
    inserted -- bigram and plain alike."""
    ids = array("q", range(417))
    for bigram in (True, False):
        kept = RadixKey(ids[:256], is_bigram=bigram)
        cap = pr.retained_raw_tokens(kept)
        asked = RadixKey(ids[0:cap], is_bigram=bigram)
        assert len(asked) == len(kept) and list(asked) == list(kept)


def _req(ntok=417, prompt=115):
    return types.SimpleNamespace(rid="weg2-2-5", origin_input_ids=[1] * prompt,
                                 output_ids=[2] * (ntok - prompt))


def test_the_cap_applies_while_the_request_has_not_grown():
    r = _req()
    setattr(r, pr.RETAINED_ATTR, 256)
    assert pr.stamp_parked(r) == 256
    assert pr.read_cap(r) == 256
    assert pr.park_match_end(r, 416) == 256  # compute_max_prefix_len = 417-1
    assert pr.park_match_end(r, 200) == 200  # never widens
    assert pr.park_read_min_tokens(r) == 1
    r.output_ids.append(3)  # resumed and decoded: past the park
    assert pr.read_cap(r) is None and pr.park_match_end(r, 417) == 417
    assert pr.park_read_min_tokens(r) is None


def test_no_stamp_or_switch_off_is_the_old_read(monkeypatch):
    r = _req()
    assert pr.stamp_parked(r) is None  # the insert stamped nothing
    assert pr.park_match_end(r, 416) == 416 and pr.park_read_min_tokens(r) is None
    setattr(r, pr.RETAINED_ATTR, 256)
    monkeypatch.setenv(pr.ENV, "0")
    assert pr.stamp_parked(r) is None and pr.park_match_end(r, 416) == 416
    monkeypatch.delenv(pr.ENV)
    assert pr.stamp_parked(r) == 256
    monkeypatch.setenv(pr.ENV, "off")  # switched off after the park: inert too
    assert pr.park_match_end(r, 416) == 416 and pr.park_read_min_tokens(r) is None


# -- the controller floor ------------------------------------------------------


def test_the_floor_lowers_the_revoke_threshold_once_and_never_raises_it():
    ctrl = types.SimpleNamespace(prefetch_threshold=256)
    op = types.SimpleNamespace(request_id="weg2-2-5")
    assert mh.revoke_threshold(ctrl, op) == 256  # nothing registered
    mh.note_min_hit_tokens(ctrl, "weg2-2-5", 1)
    # the metal numbers: 255 pages stored, one under 256
    assert not (255 < mh.revoke_threshold(ctrl, op)), "the park's 255 pages must land"
    assert mh.revoke_threshold(ctrl, op) == 256, "consumed by the one operation"
    mh.note_min_hit_tokens(ctrl, "weg2-2-5", 4096)
    assert mh.revoke_threshold(ctrl, op) == 256, "a floor never raises the threshold"
    mh.note_min_hit_tokens(ctrl, "weg2-2-5", 1)
    mh.note_min_hit_tokens(ctrl, "weg2-2-5", None)  # a later read without a floor
    assert mh.revoke_threshold(ctrl, op) == 256


def test_the_floor_table_is_bounded_and_skips_a_controller_without_one():
    ctrl = types.SimpleNamespace(prefetch_threshold=256)
    for i in range(mh._CAP + 50):
        mh.note_min_hit_tokens(ctrl, f"r{i}", 1)
    assert len(getattr(ctrl, mh.ATTR)) == mh._CAP
    mh.note_min_hit_tokens(None, "x", 1)  # no controller: no-op
    bare = types.SimpleNamespace(prefetch_threshold=256)
    mh.note_min_hit_tokens(bare, "x", None)
    assert not hasattr(bare, mh.ATTR), "clearing creates no table"


# -- the park stamps what the retraction's insert retained ---------------------


class _Batch:
    def __init__(self, reqs, retained):
        self.reqs = list(reqs)
        self.batch_is_full = True
        self.retained = retained

    def is_empty(self):
        return not self.reqs

    def filter_batch(self, **_kw):
        pass

    def retract_all(self, server_args, offload_kv=True, retain=False):
        # what cache_finished_req does on the retaining insert
        for r in self.reqs:
            if r.rid in self.retained:
                setattr(r, pr.RETAINED_ATTR, self.retained[r.rid])
        out, self.reqs = self.reqs, []
        return out


class _Sched:
    def __init__(self, running, retained):
        self.running_batch = _Batch(running, retained)
        self.waiting_queue = []
        self.last_batch = None
        self.enable_overlap = False
        self.result_queue = deque()
        self.chunked_req = None
        self.anchor_tails = []
        self.server_args = types.SimpleNamespace()
        self.weg2_dormant = False
        self.noted = []

    def _969ad_note_retract(self, req, site):
        self.noted.append((req.rid, site))


def _park(s):
    from sglang.srt.managers.io_struct import Weg2ParkRunningReqInput

    return rt.park_running(s, Weg2ParkRunningReqInput(epoch=2, reason="immediate-over-x"))


def test_park_running_stamps_the_retained_span_and_says_so(caplog):
    a = _req()
    a.kv_arrival_seq, a.is_fast_lane, a.spill_class = 1, False, None
    setattr(a, pr.RETAINED_ATTR, 64)  # a stale stamp from an earlier insert
    s = _Sched([a], {"weg2-2-5": 256})
    with caplog.at_level("INFO", logger=rt.logger.name):
        out = _park(s)
    assert out.success and out.parked == ["weg2-2-5"]
    assert pr.read_cap(a) == 256 and pr.park_match_end(a, 416) == 256
    assert any("WEG2-D-PARK RETAINED rid=weg2-2-5 retained=256 of 417 (tail 161" in m
               for m in caplog.messages)


def test_a_park_whose_insert_stamped_nothing_keeps_the_old_read():
    """The stale stamp of an earlier insert is cleared before the retraction:
    an insert that did not stamp (a session path) leaves NO cap, not an old one."""
    a = _req()
    a.kv_arrival_seq, a.is_fast_lane, a.spill_class = 1, False, None
    setattr(a, pr.RETAINED_ATTR, 64)
    s = _Sched([a], {})
    assert _park(s).success
    assert pr.read_cap(a) is None and pr.park_match_end(a, 416) == 416


# -- the real tree: the stamp is what the insert kept --------------------------


def _mamba_fixture():
    import torch

    from sglang.srt.configs.mamba_utils import Mamba2CacheParams, Mamba2StateShape
    from sglang.srt.environ import envs
    from sglang.srt.layers.attention.fla.chunk_delta_h import CHUNK_SIZE as FLA_CHUNK_SIZE
    from sglang.srt.mem_cache.allocator import TokenToKVPoolAllocator
    from sglang.srt.mem_cache.cache_init_params import CacheInitParams
    from sglang.srt.mem_cache.memory_pool import HybridLinearKVPool, HybridReqToTokenPool
    from sglang.srt.mem_cache.unified_cache_components.tree_component import ComponentType
    from sglang.srt.mem_cache.unified_radix_cache import UnifiedRadixCache
    from sglang.srt.server_args import ServerArgs, set_global_server_args_for_scheduler

    layers, full = 8, (3, 7)
    mamba_layers = [i for i in range(layers) if i not in full]
    server_args = ServerArgs(model_path="dummy", page_size=1)
    server_args._mamba_cache_chunk_size = FLA_CHUNK_SIZE
    set_global_server_args_for_scheduler(server_args)
    with envs.SGLANG_MAMBA_SSM_DTYPE.override("bfloat16"):
        shape = Mamba2StateShape.create(
            tp_world_size=1, intermediate_size=256, n_groups=1, num_heads=2,
            head_dim=16, state_size=16, conv_kernel=4,
        )
        cache_params = Mamba2CacheParams(shape=shape, layers=mamba_layers)
    pool = HybridReqToTokenPool(
        size=10, mamba_size=20, mamba_spec_state_size=10, max_context_len=256,
        device="cpu", enable_memory_saver=False, cache_params=cache_params,
        mamba_layer_ids=mamba_layers, enable_mamba_extra_buffer=False,
        speculative_num_draft_tokens=3,
    )
    kv_pool = HybridLinearKVPool(
        size=512, dtype=torch.bfloat16, page_size=1, head_num=2, head_dim=64,
        full_attention_layer_ids=list(full), device="cpu",
        enable_memory_saver=False, mamba_pool=pool.mamba_pool,
    )
    allocator = TokenToKVPoolAllocator(
        size=512, dtype=torch.bfloat16, device="cpu", kvcache=kv_pool, need_sort=False,
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
    return types.SimpleNamespace(cache=cache, allocator=allocator, pool=pool,
                                 kv0=allocator.available_size())


@pytest.mark.parametrize("anchor", [None, 16])
def test_cache_finished_req_stamps_the_span_it_retained(monkeypatch, anchor):
    """``anchor=16``: the mamba component answers its track point (the
    extra_buffer ``mamba_last_track_seqlen`` on the rig) below the 21 tokens;
    the tree keeps 16, frees the rest, and the stamp says 16 -- the read of the
    park asks for exactly the 16 the store can hold."""
    import torch

    from sglang.srt.managers.schedule_batch import Req
    from sglang.srt.mem_cache.base_prefix_cache import MatchPrefixParams
    from sglang.srt.mem_cache.unified_cache_components import mamba_component as mc
    from sglang.srt.sampling.sampling_params import SamplingParams

    fx = _mamba_fixture()
    if anchor is not None:
        orig = mc.MambaComponent._prepare_for_caching_req_impl

        def track_point(self, req, insert_params, token_ids_len, is_finished):
            cl = orig(self, req, insert_params, token_ids_len, is_finished)
            return min(cl, anchor) if cl is not None else cl

        monkeypatch.setattr(mc.MambaComponent, "_prepare_for_caching_req_impl", track_point)
    prompt = list(range(1000, 1021))
    req = Req(rid="weg2-2-5", origin_input_text="", origin_input_ids=array("q", prompt),
              sampling_params=SamplingParams(temperature=0, max_new_tokens=1))
    fx.pool.alloc([req])
    req.output_ids = array("q")
    req.full_untruncated_fill_ids = array("q", prompt)
    req.swa_uuid_for_lock = None
    req.extra_key = None
    req.prefix_indices = torch.empty(0, dtype=torch.int64)
    req.cache_protected_len = 0
    req.last_node = fx.cache.root_node
    fx.cache.inc_lock_ref(req.last_node)
    fx.pool.write((req.req_pool_idx, slice(0, 21)), fx.allocator.alloc(21))
    req.set_extend_range(0, 21)
    req.kv_committed_len = 21
    req.kv_allocated_len = 21
    fx.cache.cache_finished_req(req, is_insert=True)

    kept = 21 if anchor is None else anchor
    assert getattr(req, pr.RETAINED_ATTR) == kept
    mr = fx.cache.match_prefix(MatchPrefixParams(key=RadixKey(array("q", prompt))))
    assert len(mr.device_indices) == kept, "the stamp is what the tree holds"
    assert fx.allocator.available_size() == fx.kv0 - kept, "the KV above it was freed"


# -- wiring (read, not imported) -----------------------------------------------


def test_wiring_scheduler_caps_the_span_after_the_fork_cap_and_lowers_the_floor():
    sch = _read("managers", "scheduler.py")
    fork = sch.index("_match_end = _weg2_fork_match_end(req, _match_end)")
    park = sch.index("_match_end = _weg2_park_read.park_match_end(req, _match_end)")
    span = sch.index("_new_input_tokens = req.full_untruncated_fill_ids[_matched_len:_match_end]")
    assert fork < park < span
    tail = sch.index("_tail_min = _weg2_store_tail_min_tokens(req)")
    pmin = sch.index("_park_min = _weg2_park_read.park_read_min_tokens(req)")
    kw = sch.index('_tail_kw = {"min_tokens": _tail_min} if _tail_min is not None else {}')
    assert tail < pmin < kw


def test_wiring_the_tree_registers_the_floor_before_queueing_and_the_controller_reads_it():
    tree = _read("mem_cache", "unified_radix_cache.py")
    note = tree.index("note_min_hit_tokens(self.cache_controller, req_id, None if min_tokens is None else _min_len)")
    queue = tree.index("operation = self.cache_controller.prefetch(", note)
    assert note < queue
    assert "setattr(req, _weg2_park_read.RETAINED_ATTR, _weg2_park_read.retained_raw_tokens(radix_key))" in tree
    ctrl = _read("managers", "cache_controller.py")
    assert "if storage_hit_count < revoke_threshold(self, operation):" in ctrl
    assert "if storage_hit_count < self.prefetch_threshold:" not in ctrl


def test_wiring_park_running_clears_then_stamps():
    src = _read("weg2", "d_park_runtime.py")
    clear = src.index("setattr(req, d_park_read.RETAINED_ATTR, None)")
    retract = src.index("sched.running_batch.retract_all(sched.server_args, offload_kv=False, retain=True)")
    stamp = src.index("d_park_read.stamp_parked(req)")
    assert clear < retract < stamp
