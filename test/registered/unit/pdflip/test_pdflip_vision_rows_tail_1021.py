# SPDX-License-Identifier: Apache-2.0
"""VISION-ROWS (NF rc30 visiontail, 21.10.): an image P encoded once costs no second flip.

METAL (boot ..._1d822bd150_1021_063252, session a4c8b097ac):
  front 07:23:52  PDFLIP D-PROBE rid=pdflip-8-511 n=127831 device=127232 store=127232 covered=127232 uncached_d=599
                  image_in_tail=1 -> W102 route short -> long (P), SOLL R9 (the image is NEW: correct)
  front 07:24:22  PDFLIP SESSION-PREFIX rid=pdflip-10-516 common=127827 prompt=127987 prev_prompt=127831
                  PDFLIP D-PROBE rid=pdflip-10-516 n=127987 device=127808 host=0 store=127808 covered=127808
                  uncached_d=179 uncached_p=179 image_in_tail=1 class_u=179
                  SET-VERDICT ... X=6827 halfX=3413 classes=p_only ... verdict=COLLECT why=R4 big member waits 4.5 s
  front 07:24:27  SET-VERDICT ... waited_s=4.7 verdict=FLIP_P why=R4 T_halb over, big member alone (S=179)
  P     07:24:29  W102 PdFlipVisionStage PROBE-WAIT rid=pdflip-10-516 promise=127808 image_end=127821
                  W102 PdFlipVisionStage EMB-CACHE HIT rid=pdflip-10-516 items=1 rows=506 bytes=2590720
The image [127315, 127821) was computed on P one turn earlier (P's in-process EMB-CACHE holds its 506 rows), but it
ends inside the partial page past the last full one (127808): the probe counted it as P-only and the 179-token turn
flipped a second time. The fix publishes P's rows to a host store D reads (``vision_row_store``): the front counts
an image as P-only only when D cannot get its rows, D attaches them at its admission (rank 0 reads, one broadcast)
and computes the tail itself.

Pinned here: the repro (red before the fix: no ``mm_rows``, no store), SOLL R9 for an image without rows and R2
for the 179-token turn with rows, the store's identity/atomicity/LRU, the D verdict on 1 and on 3 ranks (rank 0
reads, the others take the broadcast -- R11 / Rang-Einigkeit), mm_utils taking the attached rows before the tower,
and P publishing from its attach.
"""
from __future__ import annotations

import asyncio
import json
import os
import types

import numpy as np
import pytest
import torch

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

from flliper.srt.pdflip import d_match_probe as dmp  # noqa: E402
from flliper.srt.pdflip import phase_policy as pp  # noqa: E402
from flliper.srt.pdflip import vision_d_guard as g  # noqa: E402
from flliper.srt.pdflip import vision_row_store as vrs  # noqa: E402
from flliper.test.ci.ci_register import register_cpu_ci  # noqa: E402

register_cpu_ci(est_time=10, suite="stage-a-pdflip-unit")

# pdflip-10-516, the front's and P's own numbers
N = 127987
COVERED = 127808          # 1997 full 64-token pages (device = store)
IMAGE = (127315, 127821)  # [start, end): 506 rows, image_end=127821 (P's PROBE-WAIT line)
X, HALF_X = 6827, 3413
WIDTH = 2560              # 2590720 bytes / 506 rows / 2 (bf16)
HASH = 0x9F1E3A5B7C2D4E60
GRID = (1, 22, 92)        # 22 * 92 / 4 = 506 rows (merge 2x2)


def _ans(n, device, spans, rows=(), host=0):
    return dmp.ProbeAnswer(ids=np.arange(n, dtype=np.int32), device_tokens=device, host_tokens=host,
                           key_match_depth=device + host, mm_spans=tuple(spans), mm_rows=tuple(rows))


def _rows(n=506, w=WIDTH, seed=0):
    gen = torch.Generator().manual_seed(seed)
    return torch.randn((n, w), generator=gen).to(torch.bfloat16)


def _item(span=IMAGE, h=HASH, grid=GRID, rows=None):
    """A scheduler-side item: inclusive offsets, content hash, grid in model_specific_data (MultimodalDataItem)."""
    from flliper.srt.managers.schedule_batch import Modality, MultimodalDataItem

    it = MultimodalDataItem(modality=Modality.IMAGE, hash=h, pad_value=h % (1 << 30),
                            offsets=[(span[0], span[1] - 1)])
    it.model_specific_data["image_grid_thw"] = torch.tensor([list(grid)])
    if rows is not None:
        it.precomputed_embeddings = rows
    return it


@pytest.fixture
def store(tmp_path, monkeypatch):
    d = str(tmp_path / "vrows")
    monkeypatch.setenv(vrs.ENV_DIR, d)
    monkeypatch.delenv(vrs.ENV_MB, raising=False)
    return d


# ------------------------------------------------------------------------------------------ the repro, front side
def test_repro_pdflip_10_516_without_rows_stays_p_only_r9():
    """No rows published (another boot, an evicted entry): the image in the partial page is P-only, as before."""
    p = dmp.exact_price(_ans(N, COVERED, [IMAGE]), COVERED)
    assert (p.covered, p.uncached, p.image_in_tail, p.tail_rows) == (COVERED, 179, True, 0)
    assert pp.classify_arrival(dmp.class_uncached(p), X, 0.5, True) == pp.CLASS_P_ONLY


def test_repro_pdflip_10_516_with_published_rows_is_kurz_on_d_r2():
    """RED before the fix (ProbeAnswer had no ``mm_rows``): the image P encoded one turn earlier is computed by D
    from its rows -- no P-only, the turn is 179 < halfX=3413 tokens: kurz, D at once (R2), no second flip."""
    p = dmp.exact_price(_ans(N, COVERED, [IMAGE], rows=[IMAGE]), COVERED)
    assert (p.covered, p.uncached, p.image_in_tail, p.tail_rows) == (COVERED, 179, False, 1)
    u = dmp.class_uncached(p)
    assert u == 179 and u < HALF_X
    assert pp.classify_arrival(u, X, 0.5, bool(p.image_in_tail)) == pp.CLASS_KURZ


def test_pdflip_8_511_a_new_image_is_still_p_only_r9():
    """The first turn: the image of 506 tokens is NEW (nothing published) -> P-only at P's number (R9 unchanged)."""
    span = (127232 + 93, 127232 + 93 + 506)
    p = dmp.exact_price(_ans(127831, 127232, [span]), 127232)
    assert p.image_in_tail and p.uncached_p == 599 and dmp.class_uncached(p) == 599


def test_rows_of_one_image_do_not_cover_another_in_the_tail():
    other = (127830, 127900)
    p = dmp.exact_price(_ans(N, COVERED, [IMAGE, other], rows=[IMAGE]), COVERED)
    assert p.image_in_tail and p.tail_rows == 1


def test_answer_carries_mm_rows_and_a_reprobe_keeps_the_arrivals():
    body = {"success": True, "ids": dmp.ids_to_b64(np.arange(5)), "device_tokens": 2, "host_tokens": 0,
            "key_match_depth": 2, "mm_spans": [[1, 3]], "mm_rows": [[1, 3]]}
    a = dmp.parse_answer(json.loads(json.dumps(body)))
    assert a.mm_rows == ((1, 3),)
    old = dict(body)
    old.pop("mm_rows")
    assert dmp.parse_answer(old).mm_rows == ()          # an older D build: no rows, P-only as before
    bare = dmp.parse_answer(dict(body, mm_spans=[], mm_rows=[]))
    re = dmp.with_spans(bare, [(1, 3)], [(1, 3)])
    assert re.mm_spans == ((1, 3),) and re.mm_rows == ((1, 3),)
    e = dmp.ReprobeEntry(ids=np.arange(5), spans=((1, 3),), extra_key=None, extra_key_known=True, path="/x",
                         payload=None, ns_known=True, ns=None, t=0.0, rows=((1, 3),))
    assert e.rows == ((1, 3),)


def test_front_prices_10_516_from_a_probe_with_rows(monkeypatch):
    """End to end through ``_x_exact_price``: D's probe names the image's rows -> not image_in_tail, the image is
    cached for the W102 terms (route stands), class number 179."""
    from test_pdflip_front_x_exact import _FakeTokens, _front, _payload

    f = _front(True)
    f.ftok = _FakeTokens(127482)
    f.session = object()
    body = {"success": True, "ids": dmp.ids_to_b64(np.arange(N) + 10), "device_tokens": COVERED, "host_tokens": 0,
            "key_match_depth": COVERED, "mm_spans": [list(IMAGE)], "mm_rows": [list(IMAGE)],
            "tokenize_ms": 641.3, "handle_ms": 23.4, "total_ms": 734.4}

    async def post(gr, b, timeout):
        return 200, json.dumps(body)

    async def store_depth(rid, ids, timeout_s, extra_key=None):
        return COVERED, "l3_index"

    f._d_probe_post = post
    f._store_probe_depth = store_depth
    payload = _payload(100)
    payload["messages"].append({"role": "user", "content": [{"type": "tool_result", "tool_use_id": "t", "content": [
        {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": "AAAA"}}]}]})
    xx = asyncio.run(f._x_exact_price("pdflip-10-516", "/v1/messages", payload, "t", 179, 127482))
    assert xx.pending == 179 and xx.image_in_tail is False and xx.mm_cached is True
    assert f._reprobe_book["pdflip-10-516"].rows == (IMAGE,)   # the set verdict's re-probe keeps the rows
    body["mm_rows"] = []
    xx = asyncio.run(f._x_exact_price("pdflip-10-516b", "/v1/messages", payload, "t2", 179, 127482))
    assert xx.image_in_tail is True and xx.mm_cached is False      # no rows: R9 as before


# --------------------------------------------------------------------------------------------------- the store
def test_store_roundtrip_identity_and_width(store):
    r = _rows()
    assert vrs.put(store, HASH, GRID, r)
    got = vrs.get(store, HASH, GRID, WIDTH)
    assert got is not None and got.dtype == torch.bfloat16 and torch.equal(got, r)
    assert vrs.get(store, HASH, (1, 92, 22), WIDTH) is None      # S66c-r B1: same bytes, other grid = other rows
    assert vrs.get(store, HASH ^ 1, GRID, WIDTH) is None
    assert vrs.get(store, HASH, GRID, WIDTH * 2) is None          # another tower width: never served
    assert [f for f in os.listdir(store) if ".tmp." in f] == []   # atomic: no half file left


def test_store_refuses_a_corrupt_or_foreign_file(store):
    from safetensors.torch import save

    vrs.put(store, HASH, GRID, _rows())
    p = os.path.join(store, vrs.key_name(HASH, GRID))
    data = open(p, "rb").read()
    open(p, "wb").write(data[:-10])                               # truncated payload
    assert vrs.get(store, HASH, GRID, WIDTH) is None
    # a valid safetensors file under this name whose metadata names another image: never served
    open(p, "wb").write(save({"rows": _rows()}, metadata={"hash": "%016x" % (HASH ^ 1), "grid": "1x22x92"}))
    assert vrs.get(store, HASH, GRID, WIDTH) is None


def test_the_file_is_safetensors_with_identity_metadata(store):
    """A6: the checkpoint format (Apache-2.0), tensor ``rows`` + metadata {hash, grid} -- no own format."""
    from safetensors import safe_open

    r = _rows()
    vrs.put(store, HASH, GRID, r)
    with safe_open(os.path.join(store, vrs.key_name(HASH, GRID)), framework="pt") as f:
        assert f.metadata() == {"hash": "%016x" % HASH, "grid": "1x22x92"}
        assert torch.equal(f.get_tensor("rows"), r)


def test_put_replaces_a_short_existing_file_and_keeps_a_whole_one(store):
    """A5: an existing file is trusted only at its exact size."""
    vrs.put(store, HASH, GRID, _rows())
    p = os.path.join(store, vrs.key_name(HASH, GRID))
    good = open(p, "rb").read()
    open(p, "wb").write(good[:100])
    assert vrs.get(store, HASH, GRID, WIDTH) is None
    assert vrs.put(store, HASH, GRID, _rows())                     # republished
    assert os.path.getsize(p) == len(good) and torch.equal(vrs.get(store, HASH, GRID, WIDTH), _rows())
    ino = os.stat(p).st_ino
    assert vrs.put(store, HASH, GRID, _rows()) and os.stat(p).st_ino == ino   # whole: not rewritten


def test_a1_put_never_takes_the_tmpfs_floor(store, monkeypatch):
    """A1: the container /dev/shm (16 GiB, 3.6 GiB free live) also carries the arena, the seq lanes and the tail
    hand-off parts -- a write leaves max(1 GiB, TAIL_KEEP_MIB) free: own LRU entries go first, then a named skip."""
    one = len(vrs._blob(HASH, GRID, _rows()))
    vrs.put(store, HASH + 1, GRID, _rows(seed=1))
    vrs.put(store, HASH + 2, GRID, _rows(seed=2))
    os.utime(os.path.join(store, vrs.key_name(HASH + 1, GRID)), (1000.0, 1000.0))
    floor = vrs.free_floor()
    assert floor >= 1 << 30
    def free(d):  # two own files leave 10 bytes over the floor; each file gone frees one more
        return floor + 10 + (2 - sum(1 for f in os.listdir(d) if f.endswith(vrs.SUFFIX))) * one

    monkeypatch.setattr(vrs, "_free_bytes", free)
    # room after dropping the OLDEST own entry exactly
    assert vrs.put(store, HASH, GRID, _rows())
    names = os.listdir(store)
    assert vrs.key_name(HASH + 1, GRID) not in names and vrs.key_name(HASH, GRID) in names
    # no room even with every own entry gone: a named skip, False, nothing written
    monkeypatch.setattr(vrs, "_free_bytes", lambda d: floor - 1)
    assert vrs.put(store, HASH + 9, GRID, _rows()) is False
    assert not vrs.has(store, HASH + 9, GRID)


def test_a1_the_floor_follows_tail_keep_mib(monkeypatch):
    monkeypatch.setenv("FLLIPER_PDFLIP_TAIL_KEEP_MIB", "512")
    assert vrs.free_floor() == 1 << 30
    monkeypatch.setenv("FLLIPER_PDFLIP_TAIL_KEEP_MIB", "3072")
    assert vrs.free_floor() == 3072 << 20


def test_store_is_bounded_lru(store):
    one = len(vrs._blob(HASH, GRID, _rows()))
    cap = int(one * 2.5)
    for i in range(3):
        assert vrs.put(store, HASH + i, GRID, _rows(seed=i), cap=cap)
        t = 1000.0 + i
        os.utime(os.path.join(store, vrs.key_name(HASH + i, GRID)), (t, t))
    names = sorted(os.listdir(store))
    assert len(names) == 2 and vrs.key_name(HASH, GRID) not in names  # the oldest went


def _ckpt(root, name, vision=None, visual_shard_bytes=b"V"):
    d = root / name
    d.mkdir()
    (d / "config.json").write_text(json.dumps({"vision_config": vision or {"out_hidden_size": WIDTH, "depth": 27}}))
    (d / "model.safetensors.index.json").write_text(json.dumps({"weight_map": {
        "model.visual.blocks.0.attn.qkv.weight": "model-00001.safetensors",
        "model.language_model.layers.0.mlp.weight": "model-00002.safetensors"}}))
    (d / "model-00001.safetensors").write_bytes(visual_shard_bytes)
    (d / "model-00002.safetensors").write_bytes(b"L")
    return str(d)


def test_store_namespace_is_the_tower_identity_not_the_served_name(tmp_path, monkeypatch):
    """A2: rc30 boots P and D from the SAME checkpoint (...INT4-Mixed-AutoRound-Minachist-abl-wxp; the MTP-g32 path
    is D's draft) -> one directory. A served-name alias of another checkpoint must never share it."""
    monkeypatch.delenv(vrs.ENV_DIR, raising=False)
    monkeypatch.delenv(vrs.ENV_MB, raising=False)
    vrs.tower_id.cache_clear()
    a = _ckpt(tmp_path, "Qwen3.8-Flash-Next-INT4-Mixed-AutoRound-Minachist-abl-wxp")
    b = _ckpt(tmp_path, "other-checkpoint", visual_shard_bytes=b"VV")
    link = tmp_path / "alias"
    link.symlink_to(a)
    p = types.SimpleNamespace(keep_mm_feature_on_device=False, served_model_name="NF", model_path=a)
    d = types.SimpleNamespace(keep_mm_feature_on_device=False, served_model_name="NF", model_path=str(link))
    o = types.SimpleNamespace(keep_mm_feature_on_device=False, served_model_name="NF", model_path=b)
    assert vrs.store_dir_for(p) == vrs.store_dir_for(d)                    # same checkpoint, resolved path
    assert vrs.store_dir_for(p).startswith("/dev/shm/pdflip-vrows-")
    assert vrs.store_dir_for(o) != vrs.store_dir_for(p)                    # same served name, another tower
    c = _ckpt(tmp_path, "same-shards-other-vision", vision={"out_hidden_size": WIDTH, "depth": 32})
    assert vrs.store_dir_for(types.SimpleNamespace(keep_mm_feature_on_device=False, model_path=c)) not in (
        vrs.store_dir_for(p), None)
    assert vrs.store_dir_for(types.SimpleNamespace(keep_mm_feature_on_device=False, model_path="hf/remote-id")) is None
    monkeypatch.setenv(vrs.ENV_MB, "0")
    assert vrs.store_dir_for(p) is None
    monkeypatch.setenv(vrs.ENV_MB, "1024")
    assert vrs.store_dir_for(types.SimpleNamespace(keep_mm_feature_on_device=True, model_path=a)) is None
    vrs.tower_id.cache_clear()


def test_publish_and_available_spans(store):
    it = _item(rows=_rows())
    no_grid = _item(span=(10, 20), h=7, rows=_rows(10))
    no_grid.model_specific_data.pop("image_grid_thw")
    assert vrs.publish([it, no_grid], store) == 1                  # B1: no grid, no identity -- never published
    d_item = _item()                                              # D's feature-less copy: hash + grid only
    assert vrs.available_spans([d_item], store) == [list(IMAGE)]
    assert vrs.available_spans([_item(h=HASH + 5)], store) == []


# ------------------------------------------------------------------------------------------- the D verdict
def _sched(tp=1, rank=0, store_dir=None, exchange=None):
    from flliper.srt.managers.scheduler import Scheduler

    h = types.SimpleNamespace(
        ps=types.SimpleNamespace(tp_size=tp),
        tp_group=types.SimpleNamespace(rank=rank, ranks=[0, 1, 2]),
        server_args=types.SimpleNamespace(keep_mm_feature_on_device=False, served_model_name="NF"),
        model_config=types.SimpleNamespace(hf_config=types.SimpleNamespace(
            vision_config=types.SimpleNamespace(out_hidden_size=WIDTH, deepstack_visual_indexes=[]))),
    )
    h._pdflip_vision_d_covered = lambda r, hi: COVERED
    h._pdflip_vision_d_rows = lambda r, b, c: Scheduler._pdflip_vision_d_rows(h, r, b, c)
    if exchange is not None:
        h._form_a_tp_exchange = exchange
    return h


def _req(rid="pdflip-10-516"):
    return types.SimpleNamespace(rid=rid, full_untruncated_fill_ids=list(range(N)),
                                 multimodal_inputs=types.SimpleNamespace(mm_items=[_item()]))


def test_d_verdict_refuses_without_rows_and_admits_with_published_rows(store):
    from flliper.srt.managers.scheduler import Scheduler

    req = _req()
    assert Scheduler._pdflip_vision_d_verdict(_sched(), req, None) == g.REFUSE   # nothing published: W123 path
    vrs.put(store, HASH, GRID, _rows())
    req = _req()
    assert Scheduler._pdflip_vision_d_verdict(_sched(), req, None) == g.ADMIT
    it = req.multimodal_inputs.mm_items[0]
    assert vrs.has_rows(it) and int(vrs.rows_of(it).shape[0]) == 506
    assert req._pdflip_vision_need == 0                    # the admission belt has no image left to refuse
    assert g.image_spans(req) == []


def test_d_verdict_on_three_ranks_rank0_reads_one_broadcast(store, monkeypatch):
    """R11 / Rang-Einigkeit: only TP rank 0 reads the directory; every rank attaches what rank 0 broadcast."""
    from flliper.srt.managers.scheduler import Scheduler

    vrs.put(store, HASH, GRID, _rows())
    sent = []

    def ex_src(site, payload):
        sent.append((site, payload))
        return payload

    req0 = _req()
    assert Scheduler._pdflip_vision_d_verdict(_sched(3, 0, exchange=ex_src), req0, None) == g.ADMIT
    assert sent and sent[0][0] == "pdflip-vision-rows/tp<-rows" and sent[0][1]["rows"] is not None
    real_load = vrs.load

    def no_read(*a, **k):
        raise AssertionError("a worker rank must never read the directory itself")

    monkeypatch.setattr(vrs, "load", no_read)
    got = []

    def ex_worker(site, payload):
        got.append(payload)
        return sent[0][1]

    req1 = _req()
    assert Scheduler._pdflip_vision_d_verdict(_sched(3, 1, exchange=ex_worker), req1, None) == g.ADMIT
    assert got == [None]
    assert torch.equal(vrs.rows_of(req1.multimodal_inputs.mm_items[0]),
                       vrs.rows_of(req0.multimodal_inputs.mm_items[0]))
    # the source's miss is the workers' miss: REFUSE on every rank
    monkeypatch.setattr(vrs, "load", real_load)
    req2 = _req("other")
    req2.multimodal_inputs.mm_items[0].hash = HASH + 99
    miss = []
    assert Scheduler._pdflip_vision_d_verdict(
        _sched(3, 0, exchange=lambda s, p: miss.append(p) or p), req2, None) == g.REFUSE
    assert miss[0]["rows"] is None


def test_d_verdict_all_or_nothing_and_per_image_only(store):
    from flliper.srt.managers.scheduler import Scheduler

    vrs.put(store, HASH, GRID, _rows())
    req = _req()
    req.multimodal_inputs.mm_items.append(_item(span=(127830, 127900), h=HASH + 1))   # unpublished second image
    assert Scheduler._pdflip_vision_d_verdict(_sched(), req, None) == g.REFUSE
    assert not any(vrs.has_rows(it) for it in req.multimodal_inputs.mm_items)
    req = _req()
    req.multimodal_inputs.mm_items[0].offsets = [(127315, 127500), (127501, 127820)]   # not per-image
    assert Scheduler._pdflip_vision_d_verdict(_sched(), req, None) == g.REFUSE


def test_the_rows_branch_sits_in_the_d_verdict_behind_a_replicated_refuse():
    import inspect

    from flliper.srt.managers.scheduler import Scheduler

    src = inspect.getsource(Scheduler._pdflip_vision_d_verdict)
    assert "out == _vdg.REFUSE and covered is not None" in src
    rows = inspect.getsource(Scheduler._pdflip_vision_d_rows)
    assert '_form_a_tp_exchange("pdflip-vision-rows/tp<-rows"' in rows
    assert rows.index("if src:") < rows.index("_vrs.load(items, d, w)")


# ------------------------------------------------------------------------------------ the forward, mm_utils
def test_mm_utils_takes_the_attached_rows_before_the_tower():
    from flliper.srt.managers import mm_utils

    mm_utils.init_mm_embedding_cache(100 << 20)
    it = _item()
    r = _rows()
    vrs.attach([it], [r])

    def tower(items):
        raise AssertionError("the tower must not be asked for an image whose rows are attached")

    out = mm_utils._get_chunked_embedding_by_item(
        tower, [it], [it.offsets[0]], extend_prefix_len=COVERED, extend_seq_len=N - COVERED,
        device=torch.device("cpu"))
    # the extend [127808, 127987) holds the image's last 13 positions 127808..127820
    assert out.shape == (IMAGE[1] - COVERED, WIDTH)
    assert torch.equal(out, r[COVERED - IMAGE[0]:])


# ------------------------------------------------------------------------------------------------ P publishes
def test_p_publishes_after_every_attach(store, monkeypatch):
    from flliper.srt import runtime_context
    from flliper.srt.pdflip import vision_rank_runner as vrr

    monkeypatch.setattr(runtime_context, "get_server_args",
                        lambda: types.SimpleNamespace(keep_mm_feature_on_device=False, served_model_name="NF"))
    it = _item(rows=_rows())
    vrr.emb_cache_store([it])
    assert vrs.has(store, HASH, GRID)
    assert torch.equal(vrs.get(store, HASH, GRID, WIDTH), it.precomputed_embeddings)


def test_a_crashed_writes_temporary_file_is_cleared(store):
    os.makedirs(store, exist_ok=True)
    stale = os.path.join(store, vrs.key_name(HASH, GRID) + ".tmp.1.2")
    fresh = os.path.join(store, vrs.key_name(HASH + 1, GRID) + ".tmp.1.3")
    for p in (stale, fresh):
        open(p, "wb").write(b"x")
    old = 1000.0
    os.utime(stale, (old, old))
    assert vrs.put(store, HASH + 2, GRID, _rows())
    assert not os.path.exists(stale) and os.path.exists(fresh)    # a write in flight is never touched


def test_a7_rows_also_after_a_bounded_tail_wait(store, monkeypatch):
    """A7: a tail wait (DEFER) that ``bounded`` turns into REFUSE tries the rows too, from the covered prefix."""
    from flliper.srt.managers.scheduler import Scheduler
    from flliper.srt.pdflip import tail_adopt

    vrs.put(store, HASH, GRID, _rows())
    monkeypatch.setattr(tail_adopt, "peek_target_start", lambda req, c, batch_empty=True: (COVERED + 5, True))
    req = _req()
    h = _sched()
    h._pdflip_vision_d_defers = {req.rid: g.DEFER_BOUND_PASSES}
    assert Scheduler._pdflip_vision_d_verdict(h, req, None) == g.ADMIT
    assert vrs.has_rows(req.multimodal_inputs.mm_items[0])


def test_a3_a_pp_stage_never_reads_on_its_own(store, monkeypatch):
    from flliper.srt.managers.scheduler import Scheduler

    vrs.put(store, HASH, GRID, _rows())
    monkeypatch.setattr(vrs, "load", lambda *a, **k: (_ for _ in ()).throw(AssertionError("read per stage")))
    h = _sched()
    h.ps.pp_size = 2
    assert Scheduler._pdflip_vision_d_verdict(h, _req(), None) == g.REFUSE


def test_a4_the_launcher_sweeps_the_family():
    from flliper.srt.pdflip import launcher

    assert any(vrs.PREFIX.startswith(p) for p in launcher.SHM_OWN_PREFIXES)


def test_a5_p_republishes_on_an_emb_cache_hit(store, monkeypatch):
    from flliper.srt import runtime_context
    from flliper.srt.pdflip import vision_rank_runner as vrr

    monkeypatch.setattr(runtime_context, "get_server_args",
                        lambda: types.SimpleNamespace(keep_mm_feature_on_device=False, model_path="/x"))
    calls = []
    monkeypatch.setattr(vrr, "_publish_rows", lambda items: calls.append(len(items)) or 1)
    import inspect

    src = inspect.getsource(vrr.emb_cache_attach)
    assert src.rindex("_publish_rows(items)") > src.index("attach_precomputed_embeddings(items, rows")
