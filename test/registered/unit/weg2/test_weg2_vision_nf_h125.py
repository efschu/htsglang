"""H125 (NF): the transient vision stage on Qwen3.8-Flash-Next.

User order 2026-09-24 10:15Z: the tower exists on NF only transiently, its
source is selectable (RAM or disk), text-only is the default, and an image
request loads the tower before the P start (free VRAM or a small temporary
offload), runs it and takes it down again. The 27B stage (V1-V3b, xsn410,
xsn438) is carried over unchanged; this file pins what H125 adds for NF:

  * SOURCE: ``disk`` (default, the 27B O_DIRECT reader) or ``ram`` (a tmpfs
    image of the tower extent, staged once, reused by key, read buffered) --
    both give the checkpoint's numbers; a vanished image is staged again; a
    refused staging falls back to disk BY NAME;
  * PLACE: ``auto`` (default) = the KV tail when wholly free, else the card's
    free VRAM, else a named refusal (W105) with both reasons; ``free`` never
    touches the allocator; ``kvtail`` is the 27B stage (its own tests);
  * arming: an unknown source/place is an arming refusal (images aborted by
    name), never a silent default;
  * model: images_tokenized() is False without image tokenization (the
    pre-H125 NF model, mrope off under language_model_only); PLE sees
    image_token_id at image pad positions, in the forward AND in the prefetch
    host mirror; text ids are untouched;
  * launcher: source/place are P-only env, published only when not the
    default (text-only and default-transient env unchanged), refused by name
    outside ``transient``; the NF arm's EXTRA override (language_model_only)
    passes the W111 riegel on P and D.

Hermetic, CPU, no server.
"""

import json
import logging
import os
import shlex
import types

import pytest
import torch
import torch.nn.functional as F

from sglang.srt.weg2 import vision_rank_runner as vrr
from sglang.srt.weg2 import vision_rank_stage as vrs

NUM_PAGES = 64
CKPT = "model-00001-of-00001.safetensors"


# ----------------------------------------------------------------- fakes --
# (the shapes of test_weg2_vision_rank_runner, kept local on purpose)


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


def _kv(num_pages=NUM_PAGES, page_size=1, layers=2, heads=2, dim=64):
    shape = ((num_pages + 1) * page_size, heads, dim)
    return types.SimpleNamespace(
        k_buffer=[torch.zeros(shape, dtype=torch.uint8) for _ in range(layers)],
        v_buffer=[torch.zeros(shape, dtype=torch.uint8) for _ in range(layers)],
    )


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

    @property
    def dtype(self):
        return self.merger.linear_fc1.weight.dtype

    def forward(self, x, grid_thw):
        return self.merger.linear_fc1(self.blocks[0].attn.qkv_proj(x)) * self.scale


def _write_model(tmp_path):
    from safetensors.torch import save_file

    torch.manual_seed(7)
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


def _build(built=None):
    def build(hf_config, device):
        with vrs.params_on_meta():
            m = _Tower()
        if built is not None:
            built.append(m)
        return m, None

    return build


class _Item:
    def __init__(self, n=4):
        self.feature = torch.randn(n, 8)
        self.precomputed_embeddings = None
        self.image_grid_thw = torch.tensor([[1, 2, 2]])
        self.modality = "image"

    def is_image(self):
        return True


def _req(rid, items=()):
    return types.SimpleNamespace(
        rid=rid, multimodal_inputs=types.SimpleNamespace(mm_items=list(items)))


def _sched():
    return types.SimpleNamespace(token_to_kv_pool_allocator=_Alloc(_kv()))


def _run(s, reqs, tmp_path, **kw):
    kw.setdefault("build", _build())
    return vrr.run_rank_stage(s, reqs, model_dir=str(tmp_path), hf_config=None,
                              device=torch.device("cpu"), **kw)


def _ref(ck, px):
    w1, b1 = ck["model.visual.blocks.0.attn.qkv.weight"], ck["model.visual.blocks.0.attn.qkv.bias"]
    w2, b2 = ck["model.visual.merger.linear_fc1.weight"], ck["model.visual.merger.linear_fc1.bias"]
    return F.linear(F.linear(px.to(torch.bfloat16), w1, b1), w2, b2)


def _tower_tensors(tmp_path):
    from sglang.srt.planner.vision_stage_load import is_vision_weight

    shard = str(tmp_path / CKPT)
    return shard, vrs.checkpoint_tensors(shard, is_vision_weight)


BIG_AIR = lambda dev: (1 << 40, 0)  # noqa: E731
NO_AIR = lambda dev: (1 << 20, 0)  # noqa: E731


# ---------------------------------------------------------------- source --


@pytest.mark.parametrize("raw,want", [(None, "disk"), ("", "disk"), ("disk", "disk"),
                                      ("ram", "ram"), (" RAM ", "ram")])
def test_source_resolution(raw, want):
    env = {} if raw is None else {vrs.SOURCE_ENV: raw}
    assert vrs.vision_source(env) == want


def test_an_unknown_source_is_refused_not_defaulted():
    with pytest.raises(vrs.VisionRankStageRefused, match="tmpfs"):
        vrs.vision_source({vrs.SOURCE_ENV: "tmpfs"})


def test_the_ram_image_is_the_exact_extent_and_is_reused_by_key(tmp_path):
    _write_model(tmp_path)
    shard, tensors = _tower_tensors(tmp_path)
    rd = str(tmp_path / "shm")
    src = vrs.stage_tower_to_ram(shard, tensors, rd)
    lo = min(t.file_offset for t in tensors)
    hi = max(t.file_offset + t.nbytes for t in tensors)
    assert (src.kind, src.shift, src.direct, src.host_bytes, src.reused) == ("ram", lo, False, hi - lo, False)
    with open(shard, "rb") as fh:
        fh.seek(lo)
        assert open(src.path, "rb").read() == fh.read(hi - lo)
    mtime = os.stat(src.path).st_mtime_ns
    again = vrs.stage_tower_to_ram(shard, tensors, rd)
    assert again.reused and again.path == src.path and os.stat(src.path).st_mtime_ns == mtime
    # a changed shard (new mtime) is never served from a stale image
    os.utime(shard, ns=(mtime + 10**9, mtime + 10**9))
    third = vrs.stage_tower_to_ram(shard, tensors, rd)
    assert not third.reused
    assert not [p for p in os.listdir(rd) if ".tmp." in p], "no half image left behind"
    vrs.remove_ram_image(third)
    assert os.listdir(rd) == []


def test_ram_and_disk_give_the_checkpoints_numbers(tmp_path):
    ck = _write_model(tmp_path)
    shard, tensors = _tower_tensors(tmp_path)
    src = vrs.stage_tower_to_ram(shard, tensors, str(tmp_path / "shm"))
    for source, kind, direct in ((None, "disk", None), (src, "ram", False)):
        it = _Item()
        px = it.feature.clone()
        s = _sched()
        full = s.token_to_kv_pool_allocator.free_pages.clone()
        out = _run(s, [_req("r", [it])], tmp_path, source=source)
        assert out.ok, out.detail
        assert out.source == kind and out.place == "kvtail"
        if direct is not None:
            assert out.direct is direct
        assert out.read_bytes >= sum(t.nbytes for t in tensors)
        assert torch.equal(it.precomputed_embeddings, _ref(ck, px))
        assert torch.equal(s.token_to_kv_pool_allocator.free_pages, full)


def test_a_vanished_ram_image_is_staged_again_by_name(tmp_path, caplog):
    ck = _write_model(tmp_path)
    shard, tensors = _tower_tensors(tmp_path)
    src = vrs.stage_tower_to_ram(shard, tensors, str(tmp_path / "shm"))
    vrs.remove_ram_image(src)
    it = _Item()
    px = it.feature.clone()
    s = _sched()
    s._weg2_vision_source = src
    out = _run(s, [_req("r", [it])], tmp_path, source=src)
    assert out.ok and out.source == "ram"
    assert os.path.exists(src.path)
    assert torch.equal(it.precomputed_embeddings, _ref(ck, px))
    assert any("is gone" in r.getMessage() for r in caplog.records)


def test_a_refused_ram_staging_falls_back_to_disk_by_name(tmp_path, caplog):
    assert vrr.arm_ram_source(str(tmp_path / "no-model-here"), env={}) is None
    assert any("RAM source" in r.getMessage() and "DISK" in r.getMessage()
               for r in caplog.records)


def test_ram_staging_registers_its_own_removal(tmp_path, monkeypatch):
    _write_model(tmp_path)
    seen = []
    monkeypatch.setattr(vrr.atexit, "register", lambda fn, *a: seen.append((fn, a)))
    env = {vrs.RAM_DIR_ENV: str(tmp_path / "shm")}
    src = vrr.arm_ram_source(str(tmp_path), env=env)
    assert src is not None and src.kind == "ram" and os.path.exists(src.path)
    assert seen and seen[0][0] is vrs.remove_ram_image and seen[0][1][0] == src
    seen.clear()
    env["SGLANG_WEG2_VISION_RAM_KEEP"] = "1"
    assert vrr.arm_ram_source(str(tmp_path), env=env).reused and seen == []


# ----------------------------------------------------------------- place --


@pytest.mark.parametrize("raw,want", [(None, "auto"), ("", "auto"), ("kvtail", "kvtail"),
                                      ("FREE", "free")])
def test_place_resolution(raw, want):
    env = {} if raw is None else {vrs.PLACE_ENV: raw}
    assert vrs.vision_place(env) == want


def test_an_unknown_place_is_refused():
    with pytest.raises(vrs.VisionRankStageRefused):
        vrs.vision_place({vrs.PLACE_ENV: "host"})


def test_slab_and_air_arithmetic():
    assert vrs.slab_bytes([1, 1, 300]) == 256 + 256 + 300
    ok, why = vrs.free_vram_verdict(100 * vrs.MIB, 700 * vrs.MIB, 0)
    assert not ok and "headroom 512 MiB" in why
    ok, _ = vrs.free_vram_verdict(100 * vrs.MIB, 500 * vrs.MIB, 200 * vrs.MIB)
    assert ok  # the idle allocator cache counts as air
    assert vrs.free_headroom(4096 * vrs.MIB) == 1024 * vrs.MIB


def _tail_in_use(s):
    a = s.token_to_kv_pool_allocator
    a.free_pages = a.free_pages[a.free_pages != NUM_PAGES]  # a prefill holds the last page
    return a.free_pages.clone()


def test_auto_takes_free_vram_when_the_tail_is_busy(tmp_path):
    ck = _write_model(tmp_path)
    s = _sched()
    before = _tail_in_use(s)
    it = _Item()
    px = it.feature.clone()
    built = []
    out = _run(s, [_req("r", [it])], tmp_path, place=vrs.PLACE_AUTO, air=BIG_AIR,
               build=_build(built))
    assert out.ok, out.detail
    assert out.place == "free" and out.tail_pages == 0
    assert torch.equal(s.token_to_kv_pool_allocator.free_pages, before)  # the pool never moved
    assert torch.equal(it.precomputed_embeddings, _ref(ck, px))
    assert list(built[0].parameters()) == []


def test_auto_refuses_by_name_when_neither_place_fits(tmp_path):
    _write_model(tmp_path)
    s = _sched()
    before = _tail_in_use(s)
    it = _Item()
    out = _run(s, [_req("r", [it])], tmp_path, place=vrs.PLACE_AUTO, air=NO_AIR)
    assert not out.ok and out.code == vrr.W_NO_ROOM
    assert "not wholly free" in out.detail and "free VRAM too small" in out.detail
    assert torch.equal(s.token_to_kv_pool_allocator.free_pages, before)
    assert it.precomputed_embeddings is None and it.feature is not None


def test_free_never_touches_the_allocator(tmp_path):
    _write_model(tmp_path)
    s = _sched()
    full = s.token_to_kv_pool_allocator.free_pages.clone()

    def _no_kv():  # pragma: no cover - must not be asked
        raise AssertionError("place=free asked the KV pool")

    s.token_to_kv_pool_allocator.get_kvcache = _no_kv
    out = _run(s, [_req("r", [_Item()])], tmp_path, place=vrs.PLACE_FREE, air=BIG_AIR)
    assert out.ok and out.place == "free"
    assert torch.equal(s.token_to_kv_pool_allocator.free_pages, full)


def test_kvtail_keeps_the_27b_refusal(tmp_path):
    _write_model(tmp_path)
    s = _sched()
    _tail_in_use(s)
    out = _run(s, [_req("r", [_Item()])], tmp_path, place=vrs.PLACE_KVTAIL, air=BIG_AIR)
    assert not out.ok and out.code == vrr.W_NO_ROOM and "free VRAM" not in out.detail


def test_the_log_line_names_place_source_and_card(caplog):
    out = vrr.StageOutcome(place="free", source="ram", card=1, items=1)
    with caplog.at_level(logging.INFO):
        vrr.log_outcome(out, ["r"], 1)
    msg = [r.getMessage() for r in caplog.records][-1]
    assert "place=free source=ram card=nvml1" in msg


# ---------------------------------------------------------------- arming --


P_TRANSIENT = {"SGLANG_WEG2_VISION": "transient", "SGLANG_WEG2_GROUP": "P"}


def _arm_sched(model_path="/m"):
    return types.SimpleNamespace(
        ps=types.SimpleNamespace(pp_rank=0, tp_size=1),
        model_config=types.SimpleNamespace(hf_config=types.SimpleNamespace(vision_config=object())),
        server_args=types.SimpleNamespace(model_path=model_path),
    )


def test_default_arming_is_disk_and_auto():
    s = _arm_sched()
    assert vrr.arm_rank_stage(s, env=dict(P_TRANSIENT)) is True
    assert s._weg2_vision_arm_refusal == ""
    assert s._weg2_vision_place == "auto" and s._weg2_vision_source is None


@pytest.mark.parametrize("key,val", [(vrs.SOURCE_ENV, "nvme"), (vrs.PLACE_ENV, "host")])
def test_an_unknown_knob_is_an_arming_refusal(key, val):
    s = _arm_sched()
    assert vrr.arm_rank_stage(s, env={**P_TRANSIENT, key: val}) is True
    assert val in s._weg2_vision_arm_refusal


def test_ram_arming_stages_the_image(tmp_path, monkeypatch):
    _write_model(tmp_path)
    monkeypatch.setattr(vrr.atexit, "register", lambda *a: None)
    s = _arm_sched(str(tmp_path))
    env = {**P_TRANSIENT, vrs.SOURCE_ENV: "ram", vrs.RAM_DIR_ENV: str(tmp_path / "shm")}
    assert vrr.arm_rank_stage(s, env=env) is True
    assert s._weg2_vision_source is not None and s._weg2_vision_source.kind == "ram"
    assert os.path.exists(s._weg2_vision_source.path)


# ----------------------------------------------------------------- model --


def test_ple_ids_map_only_image_pads():
    from sglang.srt.managers.schedule_batch import MM_PAD_SHIFT_VALUE
    from sglang.srt.models.qwen4_exp import ple_ids_for_images

    ids = torch.tensor([5, 248056, MM_PAD_SHIFT_VALUE, MM_PAD_SHIFT_VALUE + 123456, 7],
                       dtype=torch.int64)
    out = ple_ids_for_images(ids, 248056)
    assert out.tolist() == [5, 248056, 248056, 248056, 7]
    assert out.dtype == ids.dtype and ids[2] == MM_PAD_SHIFT_VALUE  # input untouched


@pytest.mark.parametrize("flag,want", [(False, False), (None, True), (True, True)])
def test_images_tokenized_follows_the_multimodal_tristate(monkeypatch, flag, want):
    import sglang.srt.runtime_context as rc
    from sglang.srt.models import qwen4_exp

    monkeypatch.setattr(rc, "get_server_args", lambda: types.SimpleNamespace(enable_multimodal=flag))
    assert qwen4_exp.images_tokenized() is want


def test_images_tokenized_without_server_args_is_the_old_model(monkeypatch):
    import sglang.srt.runtime_context as rc
    from sglang.srt.models import qwen4_exp

    def boom():
        raise RuntimeError("no server args")

    monkeypatch.setattr(rc, "get_server_args", boom)
    assert qwen4_exp.images_tokenized() is False


def test_the_mrope_gate_keeps_the_upstream_form_without_image_tokenization():
    import inspect

    from sglang.srt.models.qwen4_exp import Qwen4ExpForConditionalGeneration as M

    src = inspect.getsource(M.__init__)
    assert '"mrope_section" in rope_config and (' in src
    assert "not self.language_model_only or self.images_tokenized" in src


def test_the_prefetch_mirror_maps_pads_only_when_armed():
    from sglang.srt.managers.schedule_batch import MM_PAD_SHIFT_VALUE
    from sglang.srt.models import qwen4_exp_ple_prefetch as pf

    seq = [1, MM_PAD_SHIFT_VALUE + 5, 2]
    try:
        pf.set_mm_pad_token(None)
        assert pf._as_int64(seq).tolist() == seq  # every boot without images
        pf.set_mm_pad_token(248056)
        assert pf._as_int64(seq).tolist() == [1, 248056, 2]
    finally:
        pf.set_mm_pad_token(None)


# -------------------------------------------------------------- launcher --


def _env(group, vision, **kw):
    from sglang.srt.weg2 import launcher as lz

    return lz.build_env("/t", "/v", "0,1,2", "/s", False, "tag", group=group, vision=vision, **kw)


def test_default_knobs_publish_nothing(monkeypatch):
    from sglang.srt.weg2 import launcher as lz

    monkeypatch.setenv(lz.VISION_SOURCE_ENV, "ram")  # an operator's shell value
    monkeypatch.setenv(lz.VISION_PLACE_ENV, "free")
    for group in ("P", "D"):
        for vision in (lz.VISION_OFF, lz.VISION_TRANSIENT):
            env = _env(group, vision)
            assert lz.VISION_SOURCE_ENV not in env and lz.VISION_PLACE_ENV not in env


def test_non_default_knobs_reach_p_only():
    from sglang.srt.weg2 import launcher as lz

    p = _env("P", lz.VISION_TRANSIENT, vision_source="ram", vision_place="free")
    assert p[lz.VISION_SOURCE_ENV] == "ram" and p[lz.VISION_PLACE_ENV] == "free"
    d = _env("D", lz.VISION_TRANSIENT, vision_source="ram", vision_place="free")
    assert lz.VISION_SOURCE_ENV not in d and lz.VISION_PLACE_ENV not in d


def test_knobs_outside_transient_are_refused_by_name():
    from sglang.srt.weg2 import launcher as lz

    ns = types.SimpleNamespace(weg2_vision="off", weg2_vision_source="ram", weg2_vision_place="auto")
    with pytest.raises(lz.Weg2LaunchRefused, match="W111"):
        lz._vision_knob(ns, "weg2_vision_source", lz.VISION_SOURCE_DISK)
    ns.weg2_vision = "transient"
    assert lz._vision_knob(ns, "weg2_vision_source", lz.VISION_SOURCE_DISK) == "ram"
    assert lz.vision_stage_knobs_refusal("off", "disk", "auto") == ""


def test_the_parser_knows_both_knobs():
    from sglang.srt.weg2 import launcher as lz
    import inspect

    src = inspect.getsource(lz)
    assert '"--weg2-vision-source", choices=list(VISION_SOURCES), default=VISION_SOURCE_DISK' in src
    assert '"--weg2-vision-place", choices=list(VISION_PLACES), default=VISION_PLACE_AUTO' in src
    assert "_vision_knob(ns, \"weg2_vision_source\"" in src


def test_the_host_price_line(tmp_path):
    from sglang.srt.weg2 import launcher as lz

    _write_model(tmp_path)
    assert lz.vision_source_host_line(str(tmp_path), "disk") == ""
    line = lz.vision_source_host_line(str(tmp_path), "ram")
    assert line.startswith("W102 Weg2VisionStage SOURCE=ram") and "MiB" in line


#: the NF arm's EXTRA_P / EXTRA_D override, as arm_fnFL2_h91.sh spells it
NF_EXTRA = shlex.split('--max-total-tokens 262144 --json-model-override-args '
                       '"{\\"language_model_only\\":true}" --page-size 64')


def test_the_nf_arm_extra_passes_the_riegel_on_both_groups():
    from sglang.srt.weg2 import launcher as lz

    lz._refuse_if_extra_drops_transient_override(lz.VISION_TRANSIENT, NF_EXTRA, "P")
    lz._refuse_if_extra_drops_transient_override(lz.VISION_TRANSIENT, NF_EXTRA, "D")
    bad = ["--json-model-override-args", '{"rope_theta": 1}']
    with pytest.raises(lz.Weg2LaunchRefused, match="--extra-p"):
        lz._refuse_if_extra_drops_transient_override(lz.VISION_TRANSIENT, bad, "P")
    # the LAST occurrence is the one argparse keeps
    lz._refuse_if_extra_drops_transient_override(lz.VISION_TRANSIENT, bad + NF_EXTRA, "P")
    with pytest.raises(lz.Weg2LaunchRefused):
        lz._refuse_if_extra_drops_transient_override(lz.VISION_TRANSIENT, NF_EXTRA + bad, "P")
    lz._refuse_if_extra_drops_transient_override(lz.VISION_OFF, bad, "P")  # off: operator's business
