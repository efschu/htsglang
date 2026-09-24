"""The in-rank vision stage runner (user design 2026-09-24) -- slice V2a.

Hermetic, CPU. A tiny tower stands in for the Qwen3-VL one, with the real
checkpoint names; the KV pool and allocator are the shapes the core reads.

Pinned:
  * one stage encodes from the CHECKPOINT bytes, with the tower living on the
    KV tail, and gives everything back: pages (tail last), rope entries, the
    module's parameters; the embeddings are plain host tensors;
  * every failure is a named verdict (W105 tail in use, W106 load, W107
    encode) with the rig intact -- pages back, nothing attached;
  * the pass stages only on an idle PP0 and holds EVERY waiting request while
    it drains, holds refused rids until their abort lands and never restages
    them, and does nothing while dormant;
  * refusals become AbortReqs for the origin, the W-code as finish reason
    (the scheduler/receiver side is pinned in test_weg2_vision_rank_wiring);
  * arming: transient P PP0 only; TP>1, no vision_config, ViT graphs are
    named refusals that still arm the pass (so images are aborted by name).
"""

import json
import types
from http import HTTPStatus

import pytest
import torch
import torch.nn.functional as F

from sglang.srt.weg2 import vision_rank_runner as vrr
from sglang.srt.weg2 import vision_rank_stage as vrs

NUM_PAGES = 64
CKPT = "model-00001-of-00001.safetensors"


# ----------------------------------------------------------------- fakes --


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
    """Checkpoint-named like the real tower: blocks.0.attn.qkv(_proj), merger."""

    out_hidden_size = 6

    def __init__(self, extra=False):
        super().__init__()
        self.blocks = torch.nn.ModuleList([torch.nn.Module()])
        self.blocks[0].attn = torch.nn.Module()
        self.blocks[0].attn.qkv_proj = torch.nn.Linear(8, 24, dtype=torch.bfloat16)
        self.merger = torch.nn.Module()
        self.merger.linear_fc1 = torch.nn.Linear(24, 6, dtype=torch.bfloat16)
        if extra:
            self.extra = torch.nn.Parameter(torch.zeros(3, dtype=torch.bfloat16))
        self.register_buffer("scale", torch.ones(1, dtype=torch.bfloat16), persistent=False)

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


def _build(built=None, rope_key=None, extra=False):
    def build(hf_config, device):
        from sglang.srt.layers.rotary_embedding import factory

        if rope_key is not None:
            factory._ROPE_DICT[rope_key] = object()  # what get_rope does
        with vrs.params_on_meta():
            m = _Tower(extra=extra)
        if built is not None:
            built.append(m)
        return m, None

    return build


class _Item:
    def __init__(self, n=4, modality="image"):
        self.feature = torch.randn(n, 8)
        self.precomputed_embeddings = None
        self.image_grid_thw = torch.tensor([[1, 2, 2]])
        self.modality = modality

    def is_image(self):
        return self.modality == "image"


def _req(rid, items=()):
    return types.SimpleNamespace(
        rid=rid, multimodal_inputs=types.SimpleNamespace(mm_items=list(items)))


def _stage_sched():
    return types.SimpleNamespace(token_to_kv_pool_allocator=_Alloc(_kv()))


def _run(s, reqs, tmp_path, **kw):
    kw.setdefault("build", _build())
    return vrr.run_rank_stage(s, reqs, model_dir=str(tmp_path), hf_config=None,
                              device=torch.device("cpu"), **kw)


# ------------------------------------------------------------- the stage --


def test_one_stage_encodes_from_the_checkpoint_on_the_kv_tail_and_gives_all_back(tmp_path):
    from sglang.srt.layers.rotary_embedding import factory

    ck = _write_model(tmp_path)
    s = _stage_sched()
    alloc = s.token_to_kv_pool_allocator
    full = alloc.free_pages.clone()
    factory._ROPE_DICT[("kept-by-the-model",)] = "kept"
    built = []
    items = [_Item(4), _Item(2)]
    pixels = [it.feature.clone() for it in items]
    try:
        out = _run(s, [_req("r1", items[:1]), _req("r2", items[1:])], tmp_path,
                   build=_build(built, rope_key=("tower-rope",)))
        assert ("tower-rope",) not in factory._ROPE_DICT
        assert factory._ROPE_DICT[("kept-by-the-model",)] == "kept"
    finally:
        factory._ROPE_DICT.pop(("kept-by-the-model",), None)
        factory._ROPE_DICT.pop(("tower-rope",), None)
    assert out.ok, out.detail
    assert out.items == 2 and out.tail_pages > 0
    w1, b1 = ck["model.visual.blocks.0.attn.qkv.weight"], ck["model.visual.blocks.0.attn.qkv.bias"]
    w2, b2 = ck["model.visual.merger.linear_fc1.weight"], ck["model.visual.merger.linear_fc1.bias"]
    for it, px in zip(items, pixels):
        ref = F.linear(F.linear(px.to(torch.bfloat16), w1, b1), w2, b2)
        assert torch.equal(it.precomputed_embeddings, ref)  # the checkpoint's numbers
        assert it.feature is None
        assert not it.precomputed_embeddings.is_inference()
    # the tower LIVED on the KV tail: the first parameter's bytes are its first bytes
    lo = NUM_PAGES - out.tail_pages + 1
    seg0 = alloc.get_kvcache().k_buffer[0][lo:].reshape(-1)
    assert torch.equal(seg0[: w1.numel() * 2], w1.view(torch.uint8).reshape(-1))
    # everything back: pages (tail last), parameters dropped
    assert torch.equal(alloc.free_pages, full)
    assert list(built[0].parameters()) == []
    assert out.tower_bytes == sum(v.numel() * 2 for k, v in ck.items() if "visual" in k)
    assert {"build", "load", "encode", "attach", "teardown"} <= set(out.legs_ms)


def test_a_tail_in_use_is_W105_and_nothing_moves(tmp_path):
    _write_model(tmp_path)
    s = _stage_sched()
    alloc = s.token_to_kv_pool_allocator
    alloc.free_pages = alloc.free_pages[alloc.free_pages != NUM_PAGES]  # a prefill holds it
    before = alloc.free_pages.clone()
    it = _Item()
    built = []
    out = _run(s, [_req("r", [it])], tmp_path, build=_build(built))
    assert not out.ok and out.code == vrr.W_NO_ROOM and "not wholly free" in out.detail
    assert torch.equal(alloc.free_pages, before)
    assert it.precomputed_embeddings is None and it.feature is not None
    assert list(built[0].parameters()) == []


def test_an_encode_failure_is_W107_and_the_pages_come_back(tmp_path):
    _write_model(tmp_path)
    s = _stage_sched()
    full = s.token_to_kv_pool_allocator.free_pages.clone()

    def boom(module, items):
        raise RuntimeError("kernel launch failed")

    it = _Item()
    out = _run(s, [_req("r", [it])], tmp_path, encode=boom)
    assert not out.ok and out.code == vrr.W_ENCODE and "kernel launch failed" in out.detail
    assert torch.equal(s.token_to_kv_pool_allocator.free_pages, full)
    assert it.precomputed_embeddings is None


def test_an_unfilled_parameter_is_W106_and_the_pages_come_back(tmp_path):
    _write_model(tmp_path)
    s = _stage_sched()
    full = s.token_to_kv_pool_allocator.free_pages.clone()
    out = _run(s, [_req("r", [_Item()])], tmp_path, build=_build(extra=True))
    assert not out.ok and out.code == vrr.W_LOAD and "no checkpoint tensor" in out.detail
    assert torch.equal(s.token_to_kv_pool_allocator.free_pages, full)


def test_a_non_image_item_is_refused_before_anything_is_built(tmp_path):
    _write_model(tmp_path)
    built = []
    out = _run(_stage_sched(), [_req("r", [_Item(modality="video")])], tmp_path,
               build=_build(built))
    assert not out.ok and out.code == vrr.W_ENCODE and "images only" in out.detail
    assert built == []


# -------------------------------------------------------------- the pass --


def _pass_sched(queue, *, idle=True, refusal="", dormant=False):
    return types.SimpleNamespace(
        waiting_queue=list(queue),
        weg2_dormant=dormant,
        _weg2_vision_refused=set(),
        _weg2_vision_arm_refusal=refusal,
        _weg2_vision_origin_aborts=[],
        _weg2_vision_runs=0,
        running_batch=types.SimpleNamespace(is_empty=lambda: idle),
        chunked_req=None,
        _pp_microbatches_drained=lambda: True,
        server_args=types.SimpleNamespace(model_path="/m"),
        model_config=types.SimpleNamespace(hf_config=None),
    )


@pytest.fixture
def stage_calls(monkeypatch):
    calls = []
    verdict = {"ok": True}

    def fake(scheduler, reqs, **kw):
        calls.append([r.rid for r in reqs])
        if verdict["ok"]:
            for r in reqs:
                for it in vrr.unstaged_items(r):
                    it.precomputed_embeddings, it.feature = torch.zeros(1, 6), None
            return vrr.StageOutcome()
        return vrr.StageOutcome(ok=False, code=vrr.W_NO_ROOM, detail="reserve: tail busy")

    monkeypatch.setattr(vrr, "run_rank_stage", fake)
    monkeypatch.setattr(vrr, "_rank_device", lambda: torch.device("cpu"))
    return calls, verdict


def test_text_only_passes_untouched(stage_calls):
    calls, _ = stage_calls
    s = _pass_sched([_req("t1"), _req("t2")])
    assert vrr.vision_rank_pass(s) == []
    assert calls == [] and [r.rid for r in s.waiting_queue] == ["t1", "t2"]


def test_an_idle_pp0_stages_every_pending_image_with_one_load(stage_calls):
    calls, _ = stage_calls
    s = _pass_sched([_req("t1"), _req("i1", [_Item()]), _req("i2", [_Item(), _Item()])])
    assert vrr.vision_rank_pass(s) == []
    assert calls == [["i1", "i2"]] and s._weg2_vision_runs == 1
    assert vrr.vision_rank_pass(s) == [] and len(calls) == 1  # staged: nothing pending


def test_a_busy_pp0_holds_EVERY_waiting_request_until_it_drained(stage_calls):
    calls, _ = stage_calls
    queue = [_req("t1"), _req("i1", [_Item()]), _req("t2")]
    s = _pass_sched(queue, idle=False)
    parked = vrr.vision_rank_pass(s)
    assert calls == [] and s.waiting_queue == []
    assert [r.rid for _, r in parked] == ["t1", "i1", "t2"]
    s.waiting_queue.append(_req("new"))  # arrived during the admission
    vrr.vision_unpark(s, parked)
    assert [r.rid for r in s.waiting_queue] == ["t1", "i1", "t2", "new"]


def test_a_refused_stage_aborts_by_name_and_is_never_restaged(stage_calls):
    calls, verdict = stage_calls
    verdict["ok"] = False
    s = _pass_sched([_req("t1"), _req("i1", [_Item()])])
    parked = vrr.vision_rank_pass(s)
    assert [r.rid for _, r in parked] == ["i1"]      # text goes on
    assert "i1" in s._weg2_vision_refused
    vrr.vision_unpark(s, parked)
    again = vrr.vision_rank_pass(s)                   # the abort has not landed yet
    assert [r.rid for _, r in again] == ["i1"] and len(calls) == 1
    vrr.vision_unpark(s, again)
    aborts = vrr.take_origin_aborts(s)
    assert [a.rid for a in aborts] == ["i1"]
    assert aborts[0].finished_reason["message"].startswith(vrr.W_NO_ROOM)
    assert aborts[0].finished_reason["status_code"] == HTTPStatus.SERVICE_UNAVAILABLE
    assert vrr.take_origin_aborts(s) == []
    s.waiting_queue = [r for r in s.waiting_queue if r.rid != "i1"]  # the abort landed
    vrr.vision_rank_pass(s)
    assert s._weg2_vision_refused == set()


def test_a_stage_that_raises_is_a_named_abort_not_a_dead_group(monkeypatch):
    def broken(scheduler, reqs, **kw):
        raise AttributeError("allocator without free_pages")

    monkeypatch.setattr(vrr, "run_rank_stage", broken)
    monkeypatch.setattr(vrr, "_rank_device", lambda: torch.device("cpu"))
    s = _pass_sched([_req("i1", [_Item()])])
    parked = vrr.vision_rank_pass(s)
    assert [r.rid for _, r in parked] == ["i1"]
    msg = vrr.take_origin_aborts(s)[0].finished_reason["message"]
    assert msg.startswith(vrr.W_LOAD) and "allocator without free_pages" in msg


def test_an_arming_refusal_aborts_images_by_name_without_a_stage(stage_calls):
    calls, _ = stage_calls
    s = _pass_sched([_req("i1", [_Item()]), _req("t1")], refusal="PP0 runs tensor parallel 2")
    parked = vrr.vision_rank_pass(s)
    assert calls == [] and [r.rid for _, r in parked] == ["i1"]
    msg = vrr.take_origin_aborts(s)[0].finished_reason["message"]
    assert msg.startswith(vrr.W_NOT_ARMED) and "tensor parallel 2" in msg


def test_a_dormant_group_does_nothing(stage_calls):
    calls, _ = stage_calls
    s = _pass_sched([_req("i1", [_Item()])], dormant=True)
    assert vrr.vision_rank_pass(s) == [] and calls == []


# ---------------------------------------------------------------- arming --


def _arm_sched(pp_rank=0, tp=1, vision=True):
    hf = types.SimpleNamespace(vision_config=object()) if vision else types.SimpleNamespace()
    return types.SimpleNamespace(ps=types.SimpleNamespace(pp_rank=pp_rank, tp_size=tp),
                                 model_config=types.SimpleNamespace(hf_config=hf))


P_TRANSIENT = {"SGLANG_WEG2_VISION": "transient", "SGLANG_WEG2_GROUP": "P"}


@pytest.mark.parametrize("env,pp_rank", [
    ({}, 0),
    ({"SGLANG_WEG2_VISION": "transient", "SGLANG_WEG2_GROUP": "D"}, 0),
    (P_TRANSIENT, 1),
    (P_TRANSIENT, 2),
])
def test_only_pp0_of_a_transient_p_group_arms(env, pp_rank):
    s = _arm_sched(pp_rank=pp_rank)
    assert vrr.arm_rank_stage(s, env=env) is False
    assert not hasattr(s, "_weg2_vision_refused")


def test_pp0_arms_clean():
    s = _arm_sched()
    assert vrr.arm_rank_stage(s, env=P_TRANSIENT) is True
    assert s._weg2_vision_arm_refusal == "" and s._weg2_vision_origin_aborts == []


@pytest.mark.parametrize("kw,needle", [
    ({"tp": 2}, "tensor parallel 2"),
    ({"vision": False}, "vision_config"),
])
def test_an_arming_refusal_still_arms_the_pass_so_images_are_aborted(kw, needle):
    s = _arm_sched(**kw)
    assert vrr.arm_rank_stage(s, env=P_TRANSIENT) is True
    assert needle in s._weg2_vision_arm_refusal


def test_captured_vit_graphs_are_refused(monkeypatch):
    from sglang.srt.environ import envs

    monkeypatch.setattr(envs.SGLANG_VIT_ENABLE_CUDA_GRAPH, "get", lambda: True)
    s = _arm_sched()
    assert vrr.arm_rank_stage(s, env=P_TRANSIENT) is True
    assert "SGLANG_VIT_ENABLE_CUDA_GRAPH" in s._weg2_vision_arm_refusal
