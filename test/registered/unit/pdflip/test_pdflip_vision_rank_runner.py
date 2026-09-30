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
    (the scheduler/receiver side is pinned in test_pdflip_vision_rank_wiring);
  * arming: transient P PP0 only; TP>1, no vision_config, ViT graphs are
    named refusals that still arm the pass (so images are aborted by name).
"""

import json
import types
from http import HTTPStatus

import pytest
import torch
import torch.nn.functional as F

from flliper.srt.pdflip import vision_rank_runner as vrr
from flliper.srt.pdflip import vision_rank_stage as vrs

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
        from flliper.srt.layers.rotary_embedding import factory

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
    from flliper.srt.layers.rotary_embedding import factory

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


def test_a_stage_while_a_batch_is_in_flight_never_touches_its_pages_H125e(tmp_path):
    """H125e: the stage now runs while microbatches are in flight. Their KV
    pages are out of the free list, and the tail is reserved from the free
    list only, so the tower's pages are disjoint from every in-flight batch's
    pages by construction; the in-flight rows keep their bytes."""
    _write_model(tmp_path)
    s = _stage_sched()
    alloc = s.token_to_kv_pool_allocator
    kv = alloc.get_kvcache()
    in_flight = alloc.free_pages[:3].clone()          # an admitted batch's pages (front)
    alloc.free_pages = alloc.free_pages[3:]
    for layer in kv.k_buffer:
        layer[in_flight] = 7                           # its KV bytes
    free_before = alloc.free_pages.clone()
    out = _run(s, [_req("r", [_Item()])], tmp_path)
    assert out.ok, out.detail
    lo = NUM_PAGES - out.tail_pages + 1
    tail = set(range(lo, NUM_PAGES + 1))
    assert tail.isdisjoint(set(in_flight.tolist()))
    for layer in kv.k_buffer:
        assert bool((layer[in_flight] == 7).all())     # untouched
    assert torch.equal(torch.sort(alloc.free_pages)[0], torch.sort(free_before)[0])


def test_a_batch_in_flight_on_the_tail_is_W105_not_a_shared_page_H125e(tmp_path):
    _write_model(tmp_path)
    s = _stage_sched()
    alloc = s.token_to_kv_pool_allocator
    kv = alloc.get_kvcache()
    held = torch.tensor([NUM_PAGES], dtype=alloc.free_pages.dtype)  # in flight, in the tail
    alloc.free_pages = alloc.free_pages[alloc.free_pages != NUM_PAGES]
    for layer in kv.k_buffer:
        layer[held] = 9
    out = _run(s, [_req("r", [_Item()])], tmp_path, place=vrs.PLACE_KVTAIL)
    assert not out.ok and out.code == vrr.W_NO_ROOM
    for layer in kv.k_buffer:
        assert bool((layer[held] == 9).all())


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
        pdflip_dormant=dormant,
        _pdflip_vision_refused=set(),
        _pdflip_vision_arm_refusal=refusal,
        _pdflip_vision_origin_aborts=[],
        _pdflip_vision_runs=0,
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
    assert calls == [["i1", "i2"]] and s._pdflip_vision_runs == 1
    assert vrr.vision_rank_pass(s) == [] and len(calls) == 1  # staged: nothing pending


def test_a_busy_pp0_stages_in_this_pass_and_holds_nothing_H125e(stage_calls):
    """H125e (V1 dkrnfh91visbar1dauer09270822, #1004 SLOT DISAGREEMENT):
    with microbatches in flight the old pass held EVERY waiting request until
    PP0 drained. PP0 then cycled its in-flight slots alone while PP1, drained
    earlier, parked on the next slot, and PP0 admitted the held work two slots
    later than the follower that took its row. Admissible work is admitted in
    the pass that finds it: the stage runs now, nothing is parked."""
    calls, _ = stage_calls
    queue = [_req("t1"), _req("i1", [_Item()]), _req("t2")]
    s = _pass_sched(queue, idle=False)
    s._pp_microbatches_drained = lambda: False  # fwd in flight, as on the metal
    parked = vrr.vision_rank_pass(s)
    assert parked == []
    assert calls == [["i1"]] and s._pdflip_vision_runs == 1
    assert [r.rid for r in s.waiting_queue] == ["t1", "i1", "t2"]
    assert vrr.unstaged_items(s.waiting_queue[1]) == []  # staged: admissible now


def test_a_chunked_request_in_flight_does_not_defer_the_stage_H125e(stage_calls):
    calls, _ = stage_calls
    s = _pass_sched([_req("i1", [_Item()])])
    s.chunked_req = object()
    assert vrr.vision_rank_pass(s) == [] and calls == [["i1"]]


def test_only_a_refused_image_is_held_and_text_beside_it_is_not_H125e(stage_calls):
    calls, verdict = stage_calls
    verdict["ok"] = False
    s = _pass_sched([_req("t1"), _req("i1", [_Item()]), _req("t2")], idle=False)
    parked = vrr.vision_rank_pass(s)
    assert [r.rid for _, r in parked] == ["i1"]
    assert [r.rid for r in s.waiting_queue] == ["t1", "t2"]


def test_a_refused_stage_aborts_by_name_and_is_never_restaged(stage_calls):
    calls, verdict = stage_calls
    verdict["ok"] = False
    s = _pass_sched([_req("t1"), _req("i1", [_Item()])])
    parked = vrr.vision_rank_pass(s)
    assert [r.rid for _, r in parked] == ["i1"]      # text goes on
    assert "i1" in s._pdflip_vision_refused
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
    assert s._pdflip_vision_refused == set()


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


P_TRANSIENT = {"FLLIPER_PDFLIP_VISION": "transient", "FLLIPER_PDFLIP_GROUP": "P"}


@pytest.mark.parametrize("env,pp_rank", [
    ({}, 0),
    ({"FLLIPER_PDFLIP_VISION": "transient", "FLLIPER_PDFLIP_GROUP": "D"}, 0),
    (P_TRANSIENT, 1),
    (P_TRANSIENT, 2),
])
def test_only_pp0_of_a_transient_p_group_arms(env, pp_rank):
    s = _arm_sched(pp_rank=pp_rank)
    assert vrr.arm_rank_stage(s, env=env) is False
    assert not hasattr(s, "_pdflip_vision_refused")


def test_pp0_arms_clean():
    s = _arm_sched()
    assert vrr.arm_rank_stage(s, env=P_TRANSIENT) is True
    assert s._pdflip_vision_arm_refusal == "" and s._pdflip_vision_origin_aborts == []


@pytest.mark.parametrize("kw,needle", [
    ({"tp": 2}, "tensor parallel 2"),
    ({"vision": False}, "vision_config"),
])
def test_an_arming_refusal_still_arms_the_pass_so_images_are_aborted(kw, needle):
    s = _arm_sched(**kw)
    assert vrr.arm_rank_stage(s, env=P_TRANSIENT) is True
    assert needle in s._pdflip_vision_arm_refusal


def test_captured_vit_graphs_are_refused(monkeypatch):
    from flliper.srt.environ import envs

    monkeypatch.setattr(envs.FLLIPER_VIT_ENABLE_CUDA_GRAPH, "get", lambda: True)
    s = _arm_sched()
    assert vrr.arm_rank_stage(s, env=P_TRANSIENT) is True
    assert "FLLIPER_VIT_ENABLE_CUDA_GRAPH" in s._pdflip_vision_arm_refusal


def test_the_two_long_legs_are_split_in_the_outcome_and_the_line(tmp_path, caplog):
    """(d) Befund 28.09.: 27B W102 run=1 legs encode 1773 / teardown 434 ms --
    the line now says where: encode first vs rest, teardown strip / gc / sync
    / tail / empty_cache. Instrument only: the stage's result is unchanged."""
    import logging

    _write_model(tmp_path)
    s = _stage_sched()
    out = _run(s, [_req("r1", [_Item(4), _Item(2)])], tmp_path)
    assert out.ok, out.detail
    assert set(out.encode_ms) == {"first", "rest"}
    assert set(out.teardown_ms) == {"strip", "gc", "sync", "tail", "empty_cache"}
    assert abs(sum(out.teardown_ms.values()) - out.legs_ms["teardown"]) < 5.0
    with caplog.at_level(logging.INFO, logger=vrr.logger.name):
        vrr.log_outcome(out, ["r1"], 1)
    line = next(r.getMessage() for r in caplog.records if "encode_split_ms=" in r.getMessage())
    assert "teardown_split_ms=(" in line and "gc " in line and "first " in line


# --------------------------------------------------- (d) the async stage --


class _ManualPool:
    """The worker thread under the test's hand: submit() records the job,
    finish() runs it and completes the future."""

    def __init__(self):
        from concurrent.futures import Future

        self._Future = Future
        self.jobs = []

    def submit(self, fn):
        f = self._Future()
        self.jobs.append((fn, f))
        return f

    def finish(self):
        for fn, f in self.jobs:
            f.set_result(fn())
        self.jobs = []


def _async_sched(tmp_path, queue):
    s = _pass_sched(queue)
    alloc = _Alloc(_kv())
    s.token_to_kv_pool_allocator = alloc
    s.server_args = types.SimpleNamespace(model_path=str(tmp_path))
    s._pdflip_vision_source = None
    return s, alloc


def test_async_holds_the_lease_over_two_passes_and_gives_it_back_after_attach(tmp_path, monkeypatch):
    """(d) user decision 28.09.: the KV-tail LEASE is held while the encode
    runs; the admission sees those pages as used (they are out of the free
    list), the image request is held out, other work is admitted in the same
    pass; after the attach the lease goes back and the request is admitted."""
    monkeypatch.delenv(vrr.VISION_ASYNC_ENV, raising=False)
    _write_model(tmp_path)
    pool = _ManualPool()
    monkeypatch.setattr(vrr, "_async_pool", lambda: pool)
    monkeypatch.setattr(vrr, "build_tower_meta", _build())
    monkeypatch.setattr(vrr, "_rank_device", lambda: torch.device("cpu"))
    img, txt = _req("img", [_Item(4)]), _req("txt")
    s, alloc = _async_sched(tmp_path, [img, txt])
    full = alloc.free_pages.clone()
    for _pass in range(2):
        parked = vrr.vision_rank_pass(s)
        assert [r.rid for _, r in parked] == ["img"]          # held, the text is not
        assert vrr.vision_async_inflight(s)
        lease = s._pdflip_vision_inflight.res
        assert lease.pages > 0
        assert int((alloc.free_pages >= lease.lo_page).sum()) == 0   # the admission sees them used
        vrr.vision_unpark(s, parked)
    assert s._pdflip_vision_inflight.passes == 1   # held one pass beyond the start
    pool.finish()
    parked = vrr.vision_rank_pass(s)
    assert parked == []                                        # attached: admitted now
    assert not vrr.vision_async_inflight(s)
    assert img.multimodal_inputs.mm_items[0].precomputed_embeddings is not None
    assert torch.equal(alloc.free_pages, full)                 # the lease is back
    assert s._pdflip_vision_runs == 1


def test_async_off_is_the_synchronous_stage(tmp_path, monkeypatch, stage_calls):
    calls, _ = stage_calls
    monkeypatch.setenv(vrr.VISION_ASYNC_ENV, "0")
    s = _pass_sched([_req("img", [_Item(4)])])
    assert vrr.vision_rank_pass(s) == []
    assert calls == [["img"]] and not vrr.vision_async_inflight(s)


def test_a_group_with_a_lease_is_never_idle():
    import inspect
    from flliper.srt.managers import scheduler as sm

    src = inspect.getsource(sm.Scheduler.is_fully_idle)
    assert '_pdflip_vision_inflight' in src
    assert '"vision_async"' in inspect.getsource(sm.Scheduler.idle_blockers)


def _pp3_async_sched(tmp_path, monkeypatch, *, row_only=False):
    """A PP0 of a 3-stage group with everything the async start needs."""
    _write_model(tmp_path)
    pool = _ManualPool()
    monkeypatch.setattr(vrr, "_async_pool", lambda: pool)
    monkeypatch.setattr(vrr, "build_tower_meta", _build())
    s, _ = _async_sched(tmp_path, [_req("img", [_Item(4)]), _req("txt")])
    s.ps = types.SimpleNamespace(pp_size=3, pp_rank=0)
    s.pp_flip_counters = None
    if row_only:
        from flliper.srt.pdflip import p_row_authority as prow

        setattr(s, prow.ROW_ONLY_ATTR, True)
    return s, pool


def test_async_is_not_taken_where_the_followers_plan_for_themselves(tmp_path, monkeypatch,
                                                                     stage_calls, caplog):
    """NF rc12z30c -st, 28.09. 20:48:53Z: '#631 ROW AUTHORITY DISABLED' on the
    followers, PP0 held pdflip-6-33 for the async stage, PP1/PP2 admitted it
    ('#969 EXTENT n=9 fwd=1') and waited for a frame PP0 never owed -> '#973
    RING COMMIT TIMEOUT' 120 s later. Without a row carrier PP0 must not
    withhold: the synchronous stage stages and admits in the same pass."""
    import logging

    calls, _ = stage_calls
    monkeypatch.delenv(vrr.VISION_ASYNC_ENV, raising=False)
    monkeypatch.delenv("FLLIPER_PDFLIP_P_ROW_VISION_ASYNC", raising=False)
    s, pool = _pp3_async_sched(tmp_path, monkeypatch)
    with caplog.at_level(logging.WARNING, logger=vrr.logger.name):
        parked = vrr.vision_rank_pass(s)
        vrr.vision_rank_pass(types.SimpleNamespace(**{**vars(s), "waiting_queue": []}))
    assert parked == []                              # nothing held out: every rank admits it
    assert calls == [["img"]] and not vrr.vision_async_inflight(s) and pool.jobs == []
    said = [r.getMessage() for r in caplog.records if vrr.W_ASYNC_NOT_ADMISSIBLE in r.getMessage()]
    assert len(said) == 1 and "pp_size=3" in said[0]


def test_async_on_the_row_form_needs_its_own_term(tmp_path, monkeypatch, stage_calls):
    calls, _ = stage_calls
    monkeypatch.delenv(vrr.VISION_ASYNC_ENV, raising=False)
    monkeypatch.delenv("FLLIPER_PDFLIP_P_ROW_VISION_ASYNC", raising=False)
    s, pool = _pp3_async_sched(tmp_path, monkeypatch, row_only=True)
    assert vrr.vision_rank_pass(s) == [] and calls == [["img"]]     # term off: synchronous
    monkeypatch.setenv("FLLIPER_PDFLIP_P_ROW_VISION_ASYNC", "1")
    s2, pool2 = _pp3_async_sched(tmp_path, monkeypatch, row_only=True)
    parked = vrr.vision_rank_pass(s2)
    assert [r.rid for _, r in parked] == ["img"] and vrr.vision_async_inflight(s2)
    assert len(pool2.jobs) == 1 and calls == [["img"]]              # no second sync stage
