# SPDX-License-Identifier: Apache-2.0
"""Two NF vision/front fixes of 02.10. (y7t, fc03626e0e).

(b) MM-PERSIST-1002 -- the first image after a restart. y7t front log
``..._1002_150037.front.log`` 15:05:18: ``X-EXACT-FALLBACK rid=weg2-10-10
reason=multimodal-unseen-image est_uncached=438`` -> LONG -> D->P flip ->
``WEG2-SERVED group=P leg=1 prompt_tokens=727 cached_tokens=704`` (P computed
23; the image KV was in the persistent L3 store from y7o). The front learned
the image's token count in RAM only (``X-EXACT-MM-LEARN image=9cbec2e1b5d3
tokens=64``) and lost it -- and its store anchor -- with the restart.

(a) VISION-GC-SKIP-1002 / VISION-LOAD-WARM-1002 -- the image D>P nachlauf.
y7t P log 15:05:22: ``W102 Weg2VisionStage run=1 ... legs_ms=(build 47, load
493, encode 1317, attach 0, teardown 398) ... teardown_split_ms=(strip 1, gc
389, sync 0, tail 0, empty_cache 7)``; y7o: teardown 459 (gc 448) and 433
(gc 422). The full ``gc.collect`` of PP0's heap is the teardown; the load
re-parsed the 47-shard index (94-126 ms at the desk) and read 856 MiB O_DIRECT
(270 ms) on every stage.

Hermetic, CPU.
"""
from __future__ import annotations

import asyncio
import collections
import json
import os
import time
import types

import numpy as np
import pytest
import torch

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

from sglang.srt.weg2 import front as F  # noqa: E402
from sglang.srt.weg2 import vision_rank_runner as vrr  # noqa: E402
from sglang.srt.weg2 import vision_rank_stage as vrs  # noqa: E402
from sglang.srt.weg2.front_tokens import Count, TokenSpans, mm_image_key  # noqa: E402

# ======================================================================= (b) ==

IMG = 248056
K = 64                 # weg2-10-10's image (X-EXACT-MM-LEARN tokens=64)
URL = "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAA-weg2-10-10"


def _text(n, seed):
    rng = np.random.default_rng(seed)
    out = rng.integers(1, 240000, size=n, dtype=np.int64).astype(np.int32)
    out[0] = 248045
    return out


# weg2-10-10: compact 664 (one placeholder), realised 727; the image after 200
# text tokens, so P's END-ANCHOR 704 covers it (image_end 264)
HEAD = _text(200, 11)
TAIL = _text(727 - 200 - K, 12)
COMPACT = np.concatenate([HEAD, np.array([IMG], np.int32), TAIL])
PAGE = 64


def _key():
    return mm_image_key({"url": URL, "detail": "auto", "max_dynamic_patch": None})


class _Tok:
    state = "ready"
    why = ""
    image_token_id = IMG

    def __init__(self):
        self.m = {}
        self.executor = None

    def ids_for(self, text):
        return self.m.get(text)

    def remember(self, text, ids):
        self.m[text] = ids

    def count_mm(self, path, payload):
        return (Count(n=int(COMPACT.size), ids=COMPACT, ms=1.0, reused=0,
                      encoded=int(COMPACT.size)), [_key()])


class _Probe:
    """The L3-INDEX probe: asked with the REAL ids before the first image."""

    def __init__(self, depth):
        self.d = depth
        self.asked = []

    def depth(self, ids):
        from sglang.srt.weg2.front_store import Depth

        self.asked.append(int(ids.size))
        t = min(int(self.d), int(ids.size)) // PAGE * PAGE
        return Depth(tokens=t, kv_pages=t // PAGE, pages=t // PAGE, ms=0.1, tier="l3_index",
                     l3_pages=t // PAGE)


def _front(store_dir, probe_depth=None):
    f = object.__new__(F.Front)
    f.counters = collections.Counter()
    f.x_exact = True
    f.epoch = 0
    f.awake = "D"
    f.state = "serving"
    f.tp_prefill_max_tokens = 5016
    f.queue = collections.deque()
    f.tspans = TokenSpans(agent_span=True, anchor_page=PAGE)
    f.ftok = _Tok()
    f._x_exact_rid = collections.OrderedDict()
    f.store_dir = str(store_dir)
    f._store_probe_info = {"page_size": PAGE}
    f.store_probe = _Probe(probe_depth) if probe_depth is not None else None
    f._store_probe_t = time.monotonic()
    return f


def _price(f, rid):
    return asyncio.run(f._x_exact_price(rid, "/v1/chat/completions", {"t": "t"}, "t", 438, 440,
                                        multimodal=True))


def _boot_y7o(store_dir):
    """The previous boot: weg2-10-10 fresh -> fallback, P serves 727, P's flush."""
    f = _front(store_dir, probe_depth=0)
    assert _price(f, "weg2-10-10") is None
    f._p_leg1_store_note("weg2-10-10", "t", 727)
    assert f._p_flush_store_presence() == 1
    return f


def test_y7t_weg2_10_10_the_restarted_front_prices_the_cached_image_on_d(tmp_path, caplog):
    import logging

    caplog.set_level(logging.INFO)
    _boot_y7o(tmp_path)
    doc = json.loads((tmp_path / "WEG2_FRONT_MM.json").read_text())
    rec = doc["images"][_key()]
    assert rec["k"] == K and [a[0] for a in rec["anchors"]] == [704]
    # the restart: a NEW front on the same store, the store still holds the head
    f = _front(tmp_path, probe_depth=10 ** 6)
    xx = _price(f, "weg2-10-10")
    assert xx is not None, "no multimodal-unseen-image fallback after the restart"
    assert (xx.n, xx.credit, xx.pending) == (727, 704, 23)
    assert xx.mm and xx.mm_cached and xx.image_end == 264, "D serves it: no D->P flip"
    assert xx.src == "mm_persist"
    assert f.store_probe.asked == [200], "the probe is asked only before the image"
    assert any(m.startswith("WEG2 MM-PERSIST-CREDIT rid=weg2-10-10 depth=704 image_end=264 "
                            "first_image=200 store_preimage=192/192") for m in caplog.messages)


def test_an_evicted_head_refutes_the_restored_anchor_and_p_stages_it(tmp_path):
    _boot_y7o(tmp_path)
    f = _front(tmp_path, probe_depth=64)  # 64 of the 192 full pages before the image
    xx = _price(f, "weg2-10-10")
    assert xx is not None and xx.n == 727, "K is still restored: priced exactly"
    assert not xx.mm_cached and xx.credit < xx.image_end
    assert f.counters["mm_persist_refuted"] == 1
    doc = json.loads((tmp_path / "WEG2_FRONT_MM.json").read_text())
    assert doc["images"][_key()]["anchors"] == [], "refuted anchors are dropped"


def test_no_store_probe_no_restored_credit(tmp_path):
    _boot_y7o(tmp_path)
    f = _front(tmp_path, probe_depth=None)
    xx = _price(f, "weg2-10-10")
    assert xx is not None and not xx.mm_cached


def test_the_switch_off_is_the_ram_only_table(tmp_path):
    from sglang.srt.environ import envs

    with envs.SGLANG_WEG2_ENABLE_FRONT_MM_PERSIST.override(False):
        _boot_y7o(tmp_path)
        assert not (tmp_path / "WEG2_FRONT_MM.json").exists()
        f = _front(tmp_path, probe_depth=10 ** 6)
        assert _price(f, "weg2-10-10") is None


def test_a_restored_count_the_group_contradicts_is_learned_again(tmp_path, caplog):
    import logging

    caplog.set_level(logging.INFO)
    _boot_y7o(tmp_path)
    f = _front(tmp_path, probe_depth=0)
    assert f._mm_ktok()[_key()] == K
    _price(f, "weg2-10-11")
    f._p_leg1_store_note("weg2-10-11", "t", 727 + 16)  # the group expanded it to 80
    assert f._mm_ktok()[_key()] == 80
    assert any("relearned_from=64" in m for m in caplog.messages)
    doc = json.loads((tmp_path / "WEG2_FRONT_MM.json").read_text())
    assert doc["images"][_key()] == {**doc["images"][_key()], "k": 80, "anchors": []}


def test_the_table_round_trips_and_is_bounded(tmp_path):
    from sglang.srt.weg2 import front_mm_persist as P

    t = P.MMPersist.open(str(tmp_path))
    assert t.why == "no file yet"
    ids = np.arange(1000, dtype=np.int32)
    assert t.note_k("a", 64) and not t.note_k("a", 64)
    for d in range(64, 64 * 12, 64):
        t.note_anchor(["a", "unknown"], ids, d)
    P.write_snapshot(*t.snapshot())
    assert t.snapshot() is None, "unchanged: nothing to write"
    u = P.MMPersist.open(str(tmp_path))
    assert u.ktok() == {"a": 64} and len(u.images["a"]["anchors"]) == P.MAX_ANCHORS
    assert u.credit(["a"], ids) == 64 * 11, "the deepest prefix anchor"
    assert u.credit(["a"], ids[:500]) == 448
    other = ids.copy()
    other[10] = 7
    assert u.credit(["a"], other) == 0, "a different prefix is no anchor"
    assert P.MMPersist.open("").path == "" and P.MMPersist.open(str(tmp_path / "nope")).why.startswith("no store")
    (tmp_path / P.FILE_NAME).write_text("{broken")
    assert P.MMPersist.open(str(tmp_path)).why.startswith("unreadable")
    assert P.preimage_verified(192, 200, 64) == (True, 192)
    assert P.preimage_verified(191, 200, 64) == (False, 192)
    assert P.preimage_verified(0, 30, 64) == (True, 0)


def test_the_store_walk_never_takes_the_table_for_a_page():
    from sglang.srt.weg2 import front_mm_persist as P

    assert not P.FILE_NAME.endswith(".bin") and ".tmp." not in P.FILE_NAME


# ======================================================================= (a) ==

CKPT = "model-00001-of-00001.safetensors"
NUM_PAGES = 64


class _Alloc:
    def __init__(self, kv, num_pages=NUM_PAGES, page_size=1):
        self.num_pages = num_pages
        self.page_size = page_size
        self.size = num_pages * page_size
        self.need_sort = True
        self.free_pages = torch.arange(1, num_pages + 1, dtype=torch.int64)
        self.release_pages = torch.empty((0,), dtype=torch.int64)
        self._kv = kv

    def get_kvcache(self):
        return self._kv


def _kv(num_pages=NUM_PAGES, layers=2, heads=2, dim=64):
    shape = (num_pages + 1, heads, dim)
    return types.SimpleNamespace(
        k_buffer=[torch.zeros(shape, dtype=torch.uint8) for _ in range(layers)],
        v_buffer=[torch.zeros(shape, dtype=torch.uint8) for _ in range(layers)])


class _Tower(torch.nn.Module):
    out_hidden_size = 6

    def __init__(self):
        super().__init__()
        self.blocks = torch.nn.ModuleList([torch.nn.Module()])
        self.blocks[0].attn = torch.nn.Module()
        self.blocks[0].attn.qkv_proj = torch.nn.Linear(8, 24, dtype=torch.bfloat16)
        self.merger = torch.nn.Module()
        self.merger.linear_fc1 = torch.nn.Linear(24, 6, dtype=torch.bfloat16)
        self.register_buffer("scale", torch.ones(1, dtype=torch.bfloat16), persistent=False)
        self.cache = torch.zeros(4)  # a plain tensor attribute (a rope-style cache)

    @property
    def dtype(self):
        return self.merger.linear_fc1.weight.dtype

    def forward(self, x, grid_thw):
        return self.merger.linear_fc1(self.blocks[0].attn.qkv_proj(x)) * self.scale


def _write_model(tmp_path):
    from safetensors.torch import save_file

    torch.manual_seed(1)
    t = {
        "model.language_model.embed_tokens.weight": torch.randn(10, 4),
        "model.visual.blocks.0.attn.qkv.weight": torch.randn(24, 8).to(torch.bfloat16),
        "model.visual.blocks.0.attn.qkv.bias": torch.randn(24).to(torch.bfloat16),
        "model.visual.merger.linear_fc1.weight": torch.randn(6, 24).to(torch.bfloat16),
        "model.visual.merger.linear_fc1.bias": torch.randn(6).to(torch.bfloat16),
    }
    save_file(t, str(tmp_path / CKPT))
    (tmp_path / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": {k: CKPT for k in t}}))
    return t


def _build(hf_config, device):
    with vrs.params_on_meta():
        return _Tower(), None


class _Item:
    def __init__(self, n=4):
        self.feature = torch.randn(n, 8)
        self.precomputed_embeddings = None
        self.image_grid_thw = torch.tensor([[1, 2, 2]])
        self.modality = "image"

    def is_image(self):
        return True


def _run(tmp_path, monkeypatch, encode=None):
    calls = []
    monkeypatch.setattr(vrr.gc, "collect", lambda *a: calls.append(a) or 0)
    s = types.SimpleNamespace(token_to_kv_pool_allocator=_Alloc(_kv()))
    req = types.SimpleNamespace(rid="weg2-10-10",
                                multimodal_inputs=types.SimpleNamespace(mm_items=[_Item()]))
    kw = {"encode": encode} if encode is not None else {}
    out = vrr.run_rank_stage(s, [req], model_dir=str(tmp_path), hf_config=None,
                             device=torch.device("cpu"), build=_build, **kw)
    return out, calls, s


def test_teardown_skips_the_full_gc_when_no_tower_tensor_survives(tmp_path, monkeypatch, caplog):
    import logging

    _write_model(tmp_path)
    out, calls, s = _run(tmp_path, monkeypatch)
    assert out.ok, out.detail
    assert calls == [], "y7t: gc 389 of the 398 ms teardown -- not needed when nothing survives"
    assert out.gc_mode == "skipped" and out.gc_alive == 0
    assert torch.equal(s.token_to_kv_pool_allocator.free_pages,
                       torch.arange(1, NUM_PAGES + 1, dtype=torch.int64)), "the tail came back"
    caplog.set_level(logging.INFO, logger=vrr.logger.name)
    vrr.log_outcome(out, ["weg2-10-10"], 1)
    line = next(m for m in caplog.messages if "teardown_split_ms=" in m)
    assert " gc=skipped gc_alive=0 read_cached_mib=" in line and " index=" in line


def test_a_surviving_tower_tensor_still_gets_the_full_gc(tmp_path, monkeypatch):
    _write_model(tmp_path)
    kept = []

    def encode(module, items):
        kept.append(module.blocks[0].attn.qkv_proj.weight)  # a reference cycle would do this
        return vrr.encode_items(module, items)

    out, calls, _ = _run(tmp_path, monkeypatch, encode=encode)
    assert out.ok and out.gc_mode == "full" and out.gc_alive == 1 and len(calls) == 1


def test_a_failed_stage_always_collects(tmp_path, monkeypatch):
    _write_model(tmp_path)

    def encode(module, items):
        raise RuntimeError("boom")

    out, calls, _ = _run(tmp_path, monkeypatch, encode=encode)
    assert not out.ok and out.gc_mode == "full" and len(calls) == 1


def test_gc_skip_switch_off_collects_as_before(tmp_path, monkeypatch):
    from sglang.srt.environ import envs

    _write_model(tmp_path)
    with envs.SGLANG_WEG2_ENABLE_VISION_GC_SKIP.override(False):
        out, calls, _ = _run(tmp_path, monkeypatch)
    assert out.ok and out.gc_mode == "full" and len(calls) == 1


def test_the_strip_drops_plain_tensor_attributes_too():
    with vrs.params_on_meta():
        m = _Tower()
    refs = vrr._strip_module(m)
    assert m.cache is None and list(m.parameters()) == [] and list(m.buffers()) == []
    assert len(refs) == 6  # 4 parameters, 1 buffer, 1 plain tensor


def test_the_tower_index_is_parsed_once_per_rank(tmp_path):
    _write_model(tmp_path)
    calls = []

    def find(d):
        calls.append(d)
        return str(tmp_path / CKPT)

    from sglang.srt.planner.vision_stage_load import is_vision_weight

    a = vrs.tower_index(str(tmp_path), find, is_vision_weight)
    b = vrs.tower_index(str(tmp_path), find, is_vision_weight)
    assert calls == [str(tmp_path)] and (a[2], b[2]) == (False, True) and a[1] == b[1]
    idx = tmp_path / "model.safetensors.index.json"
    os.utime(idx, ns=(time.time_ns(), time.time_ns() + 10 ** 9))
    c = vrs.tower_index(str(tmp_path), find, is_vision_weight)
    assert len(calls) == 2 and c[2] is False, "a changed index is parsed again"


def _plan(tmp_path, n=3 * vrs.MIB + 123):
    data = np.random.default_rng(3).integers(0, 255, size=n + 4096, dtype=np.uint8)
    p = tmp_path / "shard.bin"
    p.write_bytes(data.tobytes())
    cks = [vrs.CkptTensor("a", torch.uint8, (n // 2,), 4096, n // 2),
           vrs.CkptTensor("b", torch.uint8, (n - n // 2,), 4096 + n // 2, n - n // 2)]
    plan = [(ck, torch.empty(ck.nbytes, dtype=torch.uint8)) for ck in cks]
    return str(p), data, plan


def _buffered_as_direct(monkeypatch):
    # tmpfs refuses O_DIRECT: stand in a buffered fd so the cached-first path runs
    monkeypatch.setattr(vrs, "_open_direct", lambda path: (os.open(path, os.O_RDONLY), True))


def test_cached_first_takes_every_chunk_the_cache_holds(tmp_path, monkeypatch):
    path, data, plan = _plan(tmp_path)
    _buffered_as_direct(monkeypatch)
    lo, hi = vrs.tensors_extent([ck for ck, _ in plan])
    assert vrs.warm_host_cache(path, lo, hi, threads=3, chunk=vrs.MIB) == hi - lo
    direct = []

    def cached(fd, view, pos, need):  # the cache holds the whole extent
        return os.preadv(fd, [view], pos)

    real_preadv = os.preadv
    monkeypatch.setattr(vrs, "_read_cached", cached)
    monkeypatch.setattr(vrs.os, "preadv", lambda fd, bufs, pos, *f: direct.append(pos) or real_preadv(fd, bufs, pos, *f))
    rep = vrs.read_into(path, plan, cached_first=True, bounce_bytes=vrs.MIB)
    assert rep.cached_bytes >= hi - lo and rep.cached_chunks == len(direct), "no O_DIRECT read"
    for ck, dst in plan:
        assert np.array_equal(dst.numpy(), data[ck.file_offset:ck.file_offset + ck.nbytes])


def test_the_nowait_read_answers_or_refuses_never_blocks(tmp_path):
    """RWF_NOWAIT on this rig's ZFS: a cached chunk is read, an uncached one is
    EAGAIN, a file the dataset will not serve that way EOPNOTSUPP -- the last
    two are 0 (read O_DIRECT)."""
    import mmap

    path, data, _plan_ = _plan(tmp_path)
    fd = os.open(path, os.O_RDONLY)
    buf = mmap.mmap(-1, vrs.MIB)
    try:
        got = vrs._read_cached(fd, memoryview(buf), 0, vrs.MIB)
        assert got in (0, vrs.MIB)
        if got:
            assert bytes(buf[:64]) == data[:64].tobytes()
    finally:
        os.close(fd)
        buf.close()


def test_a_cold_chunk_falls_back_to_the_direct_read(tmp_path, monkeypatch):
    path, data, plan = _plan(tmp_path)
    _buffered_as_direct(monkeypatch)
    monkeypatch.setattr(vrs, "_read_cached", lambda fd, view, pos, need: 0)  # EAGAIN everywhere
    rep = vrs.read_into(path, plan, cached_first=True, bounce_bytes=vrs.MIB)
    assert rep.cached_bytes == 0 and rep.bytes_read > 0
    for ck, dst in plan:
        assert np.array_equal(dst.numpy(), data[ck.file_offset:ck.file_offset + ck.nbytes])


def test_the_stage_asks_the_cache_first_only_when_switched_on(tmp_path, monkeypatch):
    from sglang.srt.environ import envs

    _write_model(tmp_path)
    seen = []
    real = vrs.read_into

    def spy(*a, **kw):
        seen.append(kw.get("cached_first"))
        return real(*a, **kw)

    monkeypatch.setattr(vrs, "read_into", spy)
    _run(tmp_path, monkeypatch)
    with envs.SGLANG_WEG2_ENABLE_VISION_LOAD_WARM.override(False):
        _run(tmp_path, monkeypatch)
    assert seen == [True, False]


def test_the_front_warms_the_extent_once_at_an_image_arrival(tmp_path, monkeypatch):
    _write_model(tmp_path)
    f = object.__new__(F.Front)
    f.counters = collections.Counter()
    f._store_probe_info = {"model_path": str(tmp_path)}
    got = []
    monkeypatch.setattr(vrs, "warm_host_cache", lambda p, lo, hi, **kw: got.append((p, lo, hi)) or hi - lo)
    assert f._vision_warm("weg2-10-10") == "started"
    for _ in range(200):
        if f.counters["vision_warm"]:
            break
        time.sleep(0.01)
    assert got and got[0][0].endswith(CKPT) and f.counters["vision_warm"] == 1
    assert f._vision_warm("weg2-10-11") == "fresh", "a burst of image arrivals warms once"
    g = object.__new__(F.Front)
    g._store_probe_info = {}
    assert g._vision_warm("weg2-0-1") == "no_model"


def test_wiring_the_warm_starts_at_the_w102_routing_line():
    import inspect

    src = inspect.getsource(F)
    i = src.index('"W102 Weg2VisionStage rid=%s image_parts=%d -- routing to P; "')
    j = src.index('self._vision_warm(f"weg2-{self.epoch}-{self._rid + 1}")')
    assert 0 < j - i < 600
