# SPDX-License-Identifier: Apache-2.0
"""Review tests for VISION-ROWS (NF rc30 visiontail, fix 47e0703e37), written by the reviewer, not the author.

What the author's suite (test_pdflip_vision_rows_tail_1021.py) does not pin and these do:

  1. OUTPUT EQUALITY D vs P at the partial page (pdflip-10-516 geometry, image [127315, 127821), covered 127808):
     * the extend's image rows on D (attached rows) are bit-equal to what P's per-image path scatters for the
       same positions (tower returning the same rows) -- the offset into the image is handled by the slice;
     * the whole ``embed_mm_inputs`` on a TOWER-LESS model (get_image_feature raises) scatters exactly the 13
       attached rows at the 13 image positions of the extend and text embeddings elsewhere;
     * the 3D mrope positions D's known-image path builds (D-MM-ITEM-CACHE hit, as on 10-516) are the very
       positions P's processor path builds, and equal ``get_rope_index`` for qwen4_exp; the extend slice at the
       page boundary is a genuine 3D slice (t/h/w rows differ), so D must take the 3D path (it does: mrope=True);
  2. RANK UNIFORMITY (R11): equal verdicts AND an equal number of collectives on every rank in every path (HIT,
     MISS, rank-0 read exception, eviction between probe and admission, not-per-image, nothing past the bound, a
     second pass after a miss); the real ``broadcast_pyobj`` over a 3-process gloo group carries the bf16 rows
     (frombuffer storage) bit-exactly;
  3. STORE: a concurrent writer never shows a reader a partial file; ENOSPC is a False, never an exception;
  4. SOLL R9: a new image (no rows, or rows under another grid) stays p_only; an image whose rows exist is priced
     with its exact u INCLUDING its image tokens (user 19.10. 12:48:53Z: "die ganz normale regel").

Two Auflagen are pinned as strict xfails (they turn red when fixed, so the marker must then be removed):
  A3 a D with pp_size > 1 reads the store on every stage -> stages can disagree;
  A4 the launcher's /dev/shm residue sweep does not know the ``pdflip-vrows-`` family.
"""
from __future__ import annotations

import errno
import multiprocessing as mp
import os
import socket
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

register_cpu_ci(est_time=30, suite="stage-a-pdflip-unit")

N = 127987
COVERED = 127808
IMAGE = (127315, 127821)          # [start, end)
X, HALF_X = 6827, 3413
WIDTH = 2560
HASH = 0x9F1E3A5B7C2D4E60
GRID = (1, 22, 92)                # merged 11 x 46 = 506
MERGE = 2


def _rows(n=506, w=WIDTH, seed=0):
    gen = torch.Generator().manual_seed(seed)
    return torch.randn((n, w), generator=gen).to(torch.bfloat16)


def _item(span=IMAGE, h=HASH, grid=GRID, rows=None):
    from flliper.srt.managers.schedule_batch import Modality, MultimodalDataItem

    it = MultimodalDataItem(modality=Modality.IMAGE, hash=h, pad_value=h % (1 << 30),
                            offsets=[(span[0], span[1] - 1)])
    it.model_specific_data["image_grid_thw"] = torch.tensor([list(grid)])
    if rows is not None:
        it.precomputed_embeddings = rows
    return it


def _ans(n, device, spans, rows=()):
    return dmp.ProbeAnswer(ids=np.arange(n, dtype=np.int32), device_tokens=device, host_tokens=0,
                           key_match_depth=device, mm_spans=tuple(spans), mm_rows=tuple(rows))


@pytest.fixture
def store(tmp_path, monkeypatch):
    d = str(tmp_path / "vrows")
    monkeypatch.setenv(vrs.ENV_DIR, d)
    monkeypatch.delenv(vrs.ENV_MB, raising=False)
    return d


# =============================================================================== 1. output equality D vs P
def test_d_extend_rows_equal_p_per_image_scatter_at_the_page_boundary():
    """P (tower) and D (attached rows) hand the SAME 13 rows to the extend [127808, 127987)."""
    from flliper.srt.managers import mm_utils

    mm_utils.init_mm_embedding_cache(0)                      # nothing cached: P really asks its tower
    r = _rows()
    p_item = _item()
    calls = []

    def p_tower(items):
        calls.append(len(items))
        return r.clone()

    p_out = mm_utils._get_chunked_embedding_by_item(p_tower, [p_item], [p_item.offsets[0]], COVERED, N - COVERED,
                                                    torch.device("cpu"))
    mm_utils.init_mm_embedding_cache(0)
    d_item = _item()
    vrs.attach([d_item], [r.clone()])

    def no_tower(items):
        raise AssertionError("D has no tower")

    d_out = mm_utils._get_chunked_embedding_by_item(no_tower, [d_item], [d_item.offsets[0]], COVERED, N - COVERED,
                                                    torch.device("cpu"))
    assert calls == [1]
    assert d_out.shape == (IMAGE[1] - COVERED, WIDTH) and torch.equal(d_out, p_out)
    assert torch.equal(d_out, r[COVERED - IMAGE[0]:])


class _TowerlessVL(torch.nn.Module):
    """A D-shaped multimodal model: the embed routine finds get_image_feature, which must never run."""

    def get_image_feature(self, items):
        raise AssertionError("_require_visual: group D has no tower")


def test_embed_mm_inputs_on_a_towerless_model_scatters_exactly_the_attached_rows():
    """The whole D embed path (mask, length check, masked_scatter) on a mini sequence with the 10-516 shape: an
    image whose last 13 positions fall into the extend after a page boundary, plus an older image in the prefix
    (no rows, no feature -- skipped by overlap)."""
    from flliper.srt.managers import mm_utils
    from flliper.srt.managers.schedule_batch import MultimodalInputs

    mm_utils.init_mm_embedding_cache(0)
    w, vocab = 16, 50
    img_old, img = (4, 4 + 12), (40, 40 + 30)             # [start, end)
    prefix, n = 64, 80                                    # img ends at 70: 6 image positions in the extend
    r = _rows(30, w, seed=3)
    it_old = _item(span=img_old, h=11, grid=(1, 6, 8))     # 3 x 4 = 12 merged
    it = _item(span=img, h=HASH, grid=(1, 10, 12))         # 5 x 6 = 30 merged
    vrs.attach([it], [r])
    ids_full = torch.arange(n) % (vocab - 1)
    for a, b, iv in ((img_old[0], img_old[1], it_old), (img[0], img[1], it)):
        ids_full[a:b] = iv.pad_value
    mm = MultimodalInputs(mm_items=[it_old, it])
    emb = torch.nn.Embedding(vocab, w).to(torch.bfloat16)
    input_ids = ids_full[prefix:].clone()
    out, _info = mm_utils.embed_mm_inputs(
        mm_inputs_list=[mm], extend_prefix_lens=[prefix], extend_seq_lens=[n - prefix], input_ids=input_ids,
        input_embedding=emb, multimodal_model=_TowerlessVL(), data_embedding_func_mapping=None, use_deepstack={})
    k = img[1] - prefix                                    # image positions inside the extend
    assert torch.equal(out[:k], r[prefix - img[0]:])
    text_ids = ids_full[img[1]:].clamp(max=vocab - 1)
    assert torch.equal(out[k:], emb(text_ids))


def _processor_stub():
    from flliper.srt.multimodal.processors.qwen_vl import QwenVLImageProcessor

    return types.SimpleNamespace(model_type="qwen4_exp", _spatial_merge_size=MERGE,
                                 _as_grid_batch=QwenVLImageProcessor._as_grid_batch)


def test_mrope_d_known_image_path_equals_p_and_get_rope_index_and_is_3d_at_the_boundary():
    """D's known-image build (D-MM-ITEM-CACHE hit -- 10-516's case) and P's processor path both call
    ``_compute_image_only_mrope_positions_from_offsets``; it equals ``get_rope_index`` for qwen4_exp on the 10-516
    geometry, and the extend slice D computes from 127808 holds 13 genuinely 3D positions."""
    from flliper.srt.layers.rotary_embedding.mrope_rope_index import get_rope_index
    from flliper.srt.multimodal.processors.qwen_vl import QwenVLImageProcessor

    it = _item()
    pos, delta = QwenVLImageProcessor._compute_image_only_mrope_positions_from_offsets(
        _processor_stub(), input_len=N, mm_items=[it], dtype=torch.int64, device=torch.device("cpu"))
    pos = pos.squeeze(1)
    img_tok, vstart, vid_tok = 900, 899, 901
    ids = torch.full((N,), 7, dtype=torch.int64)
    ids[IMAGE[0] - 1] = vstart
    ids[IMAGE[0]:IMAGE[1]] = img_tok
    ref, ref_delta = get_rope_index(MERGE, img_tok, vid_tok, vstart, "qwen4_exp", None, ids.unsqueeze(0),
                                    torch.tensor([list(GRID)]), None, None)
    assert torch.equal(pos, ref.squeeze(1)) and int(delta) == int(ref_delta)
    # the extend D computes: forward_batch_info._compute_mrope_positions slices [prefix, prefix + extend)
    ext = pos[:, COVERED:N]
    img_part = ext[:, : IMAGE[1] - COVERED]
    assert img_part.shape == (3, 13)
    assert not torch.equal(img_part[0], img_part[1]) or not torch.equal(img_part[1], img_part[2])  # 3D
    # last bottom-row patches: h index constant (last row), w index running to the right edge
    h_rows, w_cols = GRID[1] // MERGE, GRID[2] // MERGE
    assert int(img_part[1, -1] - img_part[0, -1]) == h_rows - 1
    assert int(img_part[2, -1] - img_part[0, -1]) == w_cols - 1
    # text after the image is 1D-equal in all three rows (what D already served before the fix)
    txt = ext[:, IMAGE[1] - COVERED:]
    assert torch.equal(txt[0], txt[1]) and torch.equal(txt[1], txt[2])


# =============================================================================== 2. rank uniformity
COV = COVERED


def _sched(tp, rank, exchange, pp_size=1):
    from flliper.srt.managers.scheduler import Scheduler

    h = types.SimpleNamespace(
        ps=types.SimpleNamespace(tp_size=tp, pp_size=pp_size),
        tp_group=types.SimpleNamespace(rank=rank, ranks=[0, 1, 2]),
        server_args=types.SimpleNamespace(keep_mm_feature_on_device=False, served_model_name="NF"),
        model_config=types.SimpleNamespace(hf_config=types.SimpleNamespace(
            vision_config=types.SimpleNamespace(out_hidden_size=WIDTH, deepstack_visual_indexes=[]))),
    )
    h._pdflip_vision_d_covered = lambda r, hi: COV
    h._pdflip_vision_d_rows = lambda r, b, c: Scheduler._pdflip_vision_d_rows(h, r, b, c)
    h._form_a_tp_exchange = exchange
    return h


def _req(rid="pdflip-10-516", items=None):
    return types.SimpleNamespace(rid=rid, full_untruncated_fill_ids=list(range(N)), output_ids=[],
                                 multimodal_inputs=types.SimpleNamespace(mm_items=items or [_item()]))


def _lockstep(make_req, before_src=None, between=None, passes=1):
    """Run the D verdict on TP rank 0 and then on the two workers, each with its OWN request object (as on three
    processes), the workers receiving exactly what rank 0 broadcast. Returns (verdicts, exchange counts)."""
    from flliper.srt.managers.scheduler import Scheduler

    reqs = [make_req() for _ in range(3)]
    sent = []
    counts = [0, 0, 0]
    verdicts = [[], [], []]
    for p in range(passes):
        def ex_src(site, payload):
            counts[0] += 1
            assert site == "pdflip-vision-rows/tp<-rows" and payload is not None
            sent.append(payload)
            return payload

        if before_src is not None and p == 0:
            before_src()
        n_before = len(sent)
        verdicts[0].append(Scheduler._pdflip_vision_d_verdict(_sched(3, 0, ex_src), reqs[0], None))
        if between is not None and p == 0:
            between()                                      # e.g. an eviction right after rank 0's read
        for rk in (1, 2):
            it = iter(sent[n_before:])

            def ex_w(site, payload, rk=rk, it=it):
                counts[rk] += 1
                assert payload is None
                return next(it)

            verdicts[rk].append(Scheduler._pdflip_vision_d_verdict(_sched(3, rk, ex_w), reqs[rk], None))
    return verdicts, counts, reqs


def test_rank_uniform_hit_one_collective_each(store):
    vrs.put(store, HASH, GRID, _rows())
    v, c, reqs = _lockstep(_req, between=lambda: [os.unlink(os.path.join(store, f)) for f in os.listdir(store)])
    # the eviction right after rank 0's read cannot split the group: the workers never read
    assert v == [[g.ADMIT]] * 3 and c == [1, 1, 1]
    r0 = vrs.rows_of(reqs[0].multimodal_inputs.mm_items[0])
    for r in reqs[1:]:
        assert torch.equal(vrs.rows_of(r.multimodal_inputs.mm_items[0]), r0)


def test_rank_uniform_miss_and_no_second_collective_on_the_next_pass(store):
    v, c, _ = _lockstep(_req, passes=2)
    assert v == [[g.REFUSE, g.REFUSE]] * 3 and c == [1, 1, 1]   # the tried flag: one collective per request


def test_rank_uniform_when_rank0_read_raises(store, monkeypatch):
    def boom(*a, **k):
        raise OSError(errno.EIO, "tmpfs read error")

    monkeypatch.setattr(vrs, "get", boom)
    v, c, _ = _lockstep(_req)
    assert v == [[g.REFUSE]] * 3 and c == [1, 1, 1]


def test_rank_uniform_eviction_between_probe_and_admission_is_a_uniform_reroute(store):
    """The front saw the rows (probe), they are evicted before the admission: every rank refuses, and the request
    (nothing streamed) is eligible for the W50 reroute through P -- a flip, never a split group."""
    vrs.put(store, HASH, GRID, _rows())
    assert vrs.available_spans([_item()], store) == [list(IMAGE)]
    p = dmp.exact_price(_ans(N, COVERED, [IMAGE], rows=[IMAGE]), COVERED)
    assert pp.classify_arrival(dmp.class_uncached(p), X, 0.5, p.image_in_tail) == pp.CLASS_KURZ
    for f in os.listdir(store):
        os.unlink(os.path.join(store, f))
    v, c, reqs = _lockstep(_req)
    assert v == [[g.REFUSE]] * 3 and c == [1, 1, 1]
    assert all(g.reroute_eligible(r, env={}) for r in reqs)


def test_rank_uniform_no_collective_when_not_per_image_or_nothing_past_the_bound(store):
    vrs.put(store, HASH, GRID, _rows())

    def multi_offset():
        it = _item()
        it.offsets = [(127315, 127500), (127501, 127820)]
        return _req(items=[it])

    v, c, _ = _lockstep(multi_offset)
    assert v == [[g.REFUSE]] * 3 and c == [0, 0, 0]

    def in_prefix():
        return _req(items=[_item(span=(1000, 1506))])

    v, c, _ = _lockstep(in_prefix)
    assert v == [[g.ADMIT]] * 3 and c == [0, 0, 0]


def test_partial_hit_is_all_or_nothing_on_every_rank(store):
    vrs.put(store, HASH, GRID, _rows())

    def two():
        return _req(items=[_item(), _item(span=(127830, 127900), h=HASH + 1, grid=(1, 10, 28))])

    v, c, reqs = _lockstep(two)
    assert v == [[g.REFUSE]] * 3 and c == [1, 1, 1]
    assert not any(vrs.has_rows(it) for r in reqs for it in r.multimodal_inputs.mm_items)


def _free_port():
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    return port


def _gloo_rank(rank, port, path, q, item):
    import torch.distributed as dist

    from flliper.srt.utils import broadcast_pyobj

    dist.init_process_group("gloo", init_method=f"tcp://127.0.0.1:{port}", rank=rank, world_size=3)
    try:
        payload = None
        if rank == 0:
            payload = {"rows": vrs.load([item], path, WIDTH), "why": ""}
        out = broadcast_pyobj([payload] if payload is not None else [], rank, None, src=0)
        rows = out[0]["rows"]
        q.put((rank, rows[0].dtype == torch.bfloat16, rows[0].view(torch.int16).numpy().tobytes()))
    finally:
        dist.destroy_process_group()


def test_real_broadcast_pyobj_carries_the_frombuffer_bf16_rows_bit_exactly(store):
    r = _rows()
    vrs.put(store, HASH, GRID, r)
    ctx = mp.get_context("fork")
    q = ctx.Queue()
    port = _free_port()
    item = _item()
    procs = [ctx.Process(target=_gloo_rank, args=(k, port, store, q, item)) for k in range(3)]
    for p in procs:
        p.start()
    try:
        got = [q.get(timeout=120) for _ in range(3)]
        for p in procs:
            p.join(60)
            assert p.exitcode == 0
    finally:
        for p in procs:
            if p.is_alive():
                p.kill()
    want = r.view(torch.int16).numpy().tobytes()
    assert sorted(k for k, _d, _b in got) == [0, 1, 2]
    assert all(d and b == want for _k, d, b in got)


@pytest.mark.xfail(strict=True, reason="AUFLAGE A3: a D with pp_size > 1 reads the store on every PP stage itself "
                                       "(tp == 1 -> src=True): an eviction between two stages' reads splits them")
def test_auflage_a3_pp_stages_never_read_independently(store):
    from flliper.srt.managers.scheduler import Scheduler

    vrs.put(store, HASH, GRID, _rows())
    r0, r1 = _req(), _req()
    v0 = Scheduler._pdflip_vision_d_verdict(_sched(1, 0, None, pp_size=2), r0, None)
    for f in os.listdir(store):
        os.unlink(os.path.join(store, f))
    v1 = Scheduler._pdflip_vision_d_verdict(_sched(1, 0, None, pp_size=2), r1, None)
    assert v0 == v1


# =============================================================================== 3. store
def _writer(path, n, q, r):
    # r comes from the parent: no torch compute in a forked child (OpenMP after fork can deadlock)
    for i in range(n):
        vrs.put(path, HASH, GRID, r)
        try:
            os.unlink(os.path.join(path, vrs.key_name(HASH, GRID)))
        except OSError:
            pass
    q.put("done")


def test_concurrent_writer_never_shows_a_partial_file(store):
    os.makedirs(store, exist_ok=True)
    want = _rows()
    ctx = mp.get_context("fork")
    q = ctx.Queue()
    want_b = want.view(torch.int16).numpy().tobytes()
    w = ctx.Process(target=_writer, args=(store, 300, q, want))
    w.start()
    import time as _time

    seen = bad = 0
    deadline = _time.monotonic() + 120
    try:
        while w.is_alive() and _time.monotonic() < deadline:
            got = vrs.get(store, HASH, GRID, WIDTH)
            if got is not None:
                seen += 1
                bad += 0 if got.view(torch.int16).numpy().tobytes() == want_b else 1
        w.join(30)
    finally:
        if w.is_alive():
            w.kill()
    assert w.exitcode == 0 and bad == 0 and seen > 0, (seen, bad)


def test_enospc_is_false_not_an_exception(store, monkeypatch):
    import builtins

    real_open = builtins.open

    def full(path, mode="r", *a, **k):
        if ".tmp." in str(path) and "w" in mode:
            raise OSError(errno.ENOSPC, "No space left on device")
        return real_open(path, mode, *a, **k)

    monkeypatch.setattr(builtins, "open", full)
    assert vrs.put(store, HASH, GRID, _rows()) is False
    assert not vrs.has(store, HASH, GRID)


@pytest.mark.xfail(strict=True, reason="AUFLAGE A4: pdflip-vrows- is not one of the launcher's SHM_OWN_PREFIXES -- "
                                       "residue is never swept and the host census books it as other_tmpfs")
def test_auflage_a4_launcher_sweep_knows_the_vrows_family():
    from flliper.srt.pdflip import launcher

    assert any(vrs.PREFIX.startswith(p) for p in launcher.SHM_OWN_PREFIXES)


# =============================================================================== 4. SOLL R9
def test_r9_new_image_and_other_grid_stay_p_only(store):
    vrs.put(store, HASH, (1, 92, 22), _rows())             # same pixels, another grid published
    assert vrs.available_spans([_item()], store) == []     # D's tokenizer reports no rows for THIS grid
    p = dmp.exact_price(_ans(N, COVERED, [IMAGE]), COVERED)
    assert pp.classify_arrival(dmp.class_uncached(p), X, 0.5, p.image_in_tail) == pp.CLASS_P_ONLY


def test_r9_rows_for_an_image_never_clear_a_different_span():
    other = (127900, 127950)
    p = dmp.exact_price(_ans(N, COVERED, [other], rows=[IMAGE]), COVERED)    # rows name a span not in the tail
    assert p.image_in_tail and p.tail_rows == 0


def test_r9_known_image_fully_in_the_tail_is_priced_with_its_image_tokens():
    """User 19.10. 12:48:53Z: an image D can compute is classified by its exact u (new text AND image tokens)."""
    n, cov = 10000, 2048
    span = (3000, 3000 + 2024)                             # an image of 2024 tokens, wholly uncached, rows known
    p = dmp.exact_price(_ans(n, cov, [span], rows=[span]), cov)
    u = dmp.class_uncached(p)
    assert not p.image_in_tail and u == n - cov            # the image tokens are inside u
    assert pp.classify_arrival(u, X, 0.5, False) == pp.CLASS_GROSS   # 7952 >= X: big, not short
