"""VISION-WEIGHTS AP1 (plan PLAN-VISION-GEWICHTE-VERDRAENGEN-1009): the tower
borrows victim WEIGHT memory instead of KV pages -- the model-neutral core.

Hermetic, CPU (``run_rank_stage`` treats a non-cuda device as the card). The
victims are plain uint8 tensors behind a fake source; the tower and the
checkpoint are the runner tests' fakes.

Pinned:
  * T1: the dual-P shape (KV tail not in the free list, no air) is W105 under
    place=auto and runs under place=weights;
  * T3/T4: the plan's first-fit-decreasing placement is what the stage does,
    the tower views carry the checkpoint bytes, and the victims are bitwise
    back afterwards (host image released);
  * T5: every way out of the stage returns the victims (encode error, load
    error, deadline, client gone mid-encode); a failed or skipped return is
    W110c and the pass raises (crash-stop); a failed host image is W105b
    before any byte moved;
  * the encoder's work memory is computed from the real image and refused by
    name when the card's air cannot hold it -- nothing moved, nothing cut;
  * the default places are untouched by the new code (no victim fields, no
    source resolved, the async form unchanged), and place=weights never
    takes the async form.
"""

import types

import pytest
import torch

from sglang.srt.weg2 import vision_rank_runner as vrr
from sglang.srt.weg2 import vision_rank_stage as vrs
from sglang.srt.weg2 import vision_victim as vv
from test_weg2_vision_rank_runner import (  # noqa: E402 -- the runner tests' fakes
    NUM_PAGES,
    _Alloc,
    _build,
    _Item,
    _kv,
    _req,
    _write_model,
)

import torch.nn.functional as F

NO_AIR = lambda device: (0, 0)  # noqa: E731


class _Victims(vv.HostImageVictims):
    """Victim storages = the given CPU tensors. ``mode``: restore normally,
    ``skip`` (the mutant 'restore skipped') or ``raise``."""

    kind = "fake"

    def __init__(self, sizes=(4096, 2048, 1024), mode="ok", alloc=None, seed=7):
        super().__init__(vv.HostImage(alloc) if alloc is not None else None)
        g = torch.Generator().manual_seed(seed)
        self.tensors = {f"v{i}": torch.randint(0, 256, (n,), generator=g, dtype=torch.uint8)
                        for i, n in enumerate(sizes)}
        self.before = {k: t.clone() for k, t in self.tensors.items()}
        self.mode = mode
        self.stashed = 0

    def _named_storages(self):
        return [(n, vv.storage_view(t)) for n, t in self.tensors.items()]

    def stash(self, views):
        self.stashed += 1
        super().stash(views)

    def restore(self, views):
        if self.mode == "skip":
            return
        if self.mode == "raise":
            raise RuntimeError("H2D failed")
        super().restore(views)

    def intact(self):
        return all(torch.equal(self.tensors[k], self.before[k]) for k in self.tensors)


def _sched(alloc=None):
    return types.SimpleNamespace(token_to_kv_pool_allocator=alloc or _Alloc(_kv()))


def _run(s, reqs, tmp_path, **kw):
    kw.setdefault("build", _build())
    kw.setdefault("place", vrs.PLACE_WEIGHTS)
    return vrr.run_rank_stage(s, reqs, model_dir=str(tmp_path), hf_config=kw.pop("hf_config", None),
                              device=torch.device("cpu"), **kw)


def _ref_rows(ck, px):
    w1, b1 = ck["model.visual.blocks.0.attn.qkv.weight"], ck["model.visual.blocks.0.attn.qkv.bias"]
    w2, b2 = ck["model.visual.merger.linear_fc1.weight"], ck["model.visual.merger.linear_fc1.bias"]
    return F.linear(F.linear(px.to(torch.bfloat16), w1, b1), w2, b2)


# ------------------------------------------------------------------- plan --


def _cand(name, nbytes, key):
    return vv.VictimCandidate(name=name, key=key, offset=0, nbytes=nbytes, storage_nbytes=nbytes)


def test_plan_is_ffd_trimmed_deterministic_and_refuses_with_numbers():
    """R5: moved bytes = what the tower occupies (each victim trimmed to its
    used prefix, 256 B grain), largest victims first, the same plan for the
    same input; a tensor larger than every run and a short inventory are
    W105b with both numbers -- never a fallback."""
    cands = [_cand("small", 3000, 0x10000), _cand("big", 10000, 0x20000), _cand("mid", 5000, 0x40000)]
    tower = [("a", 600), ("b", 4000), ("c", 300), ("d", 4100)]
    plan = vv.plan_victims(cands, tower)
    assert plan == vv.plan_victims(list(reversed(cands)), tower)       # deterministic
    assert [s.name for s in plan.segments] == ["big"]                   # fewest storages
    # descending placement: d(4100) @0, b(4000) @4352, a(600) @8448, c(300) @9216
    assert {sl.name: sl.offset for sl in plan.slots} == {"d": 0, "b": 4352, "a": 8448, "c": 9216}
    assert plan.segments[0].nbytes == 9728                              # trimmed: 9516 -> 256 grain
    with pytest.raises(vv.VisionVictimShort, match=r"needs 0\.0 MiB in one piece.*largest victim run"):
        vv.plan_victims(cands, [("huge", 10001)])
    with pytest.raises(vv.VisionVictimShort, match="the victims hold"):
        vv.plan_victims(cands, [("x", 9000), ("y", 9000)])


def test_split_checkpoint_cuts_rows_into_contiguous_file_ranges():
    ck = vrs.CkptTensor("model.visual.merger.linear_fc2.weight", torch.bfloat16, (10, 4), 1000, 80)
    other = vrs.CkptTensor("model.visual.x.bias", torch.bfloat16, (4,), 2000, 8)
    split = {"merger.linear_fc2.weight": (("merger.linear_fc2.weight_parts.0", 6),
                                         ("merger.linear_fc2.weight_parts.1", 4))}
    name = lambda n: n[len("model.visual."):]  # noqa: E731
    out, mapped = vv.split_checkpoint([ck, other], split, name)
    assert [(t.file_offset, t.nbytes, t.shape) for t in out[:2]] == [(1000, 48, (6, 4)), (1048, 32, (4, 4))]
    assert [mapped(t.name) for t in out] == ["merger.linear_fc2.weight_parts.0",
                                             "merger.linear_fc2.weight_parts.1", "x.bias"]
    with pytest.raises(vrs.VisionRankStageRefused, match="cover 9 of 10 rows"):
        vv.split_checkpoint([ck], {"merger.linear_fc2.weight": (("p", 9),)}, name)


def test_checksum_sees_one_changed_word_and_the_tail():
    t = torch.randint(0, 256, (vv.CHECKSUM_CHUNK + 7,), dtype=torch.uint8)
    a = vv.segment_checksum(t)
    t[vv.CHECKSUM_CHUNK // 2] ^= 1
    assert vv.segment_checksum(t) != a
    t[vv.CHECKSUM_CHUNK // 2] ^= 1
    t[-1] ^= 1                                                          # a tail byte
    assert vv.segment_checksum(t) != a


def test_checksum_sees_two_rows_swapped_inside_one_chunk():
    """Review S3: a plain word sum per chunk is blind to rows exchanged inside
    the chunk (same words, other places) -- a store row loaded into the wrong
    slot would pass. The position-weighted sum sees it, also in the tail."""
    row = 128                                                           # 512 B rows
    g = torch.Generator().manual_seed(11)
    t = torch.randint(0, 256, (8 * row * 4 + 3,), generator=g, dtype=torch.uint8)
    a = vv.segment_checksum(t)
    r = t[:8 * row * 4].view(8, row * 4)
    r[[1, 5]] = r[[5, 1]].clone()                                       # rows 1 and 5 exchanged
    assert vv.segment_checksum(t) != a
    assert vv.segment_checksum(t)[0] == a[0]                            # the plain word sum is blind to it
    t2 = torch.cat([t[:-3], torch.tensor([1, 2, 3], dtype=torch.uint8)])
    b = vv.segment_checksum(t2)
    t2[-3:] = torch.tensor([3, 2, 1], dtype=torch.uint8)                # tail bytes exchanged
    assert vv.segment_checksum(t2) != b


# ------------------------------------------------------------------ stage --


def test_T1_dual_p_shape_is_W105_on_auto_and_runs_on_weights(tmp_path):
    """Dual: P's KV pool is trimmed, the tail is not in the free list and the
    card has no air for the free form -- place=auto refuses (W105); the
    weights place encodes from the checkpoint on victim memory."""
    ck = _write_model(tmp_path)
    alloc = _Alloc(_kv())
    alloc.free_pages = torch.arange(1, NUM_PAGES // 2 + 1, dtype=torch.int64)  # the cap
    out = _run(_sched(alloc), [_req("r", [_Item()])], tmp_path, place=vrs.PLACE_AUTO, air=NO_AIR)
    assert not out.ok and out.code == vrr.W_NO_ROOM
    it = _Item()
    px = it.feature.clone()
    victims = _Victims()
    out = _run(_sched(alloc), [_req("r", [it])], tmp_path, victims=victims, air=NO_AIR)
    assert out.ok, out.detail
    assert out.place == vv.PLACE_WEIGHTS and out.tail_pages == 0 and out.checksum == "ok"
    assert torch.equal(it.precomputed_embeddings, _ref_rows(ck, px))


def test_T3_T4_the_views_carry_the_checkpoint_and_the_victims_come_back_bitwise(tmp_path):
    ck = _write_model(tmp_path)
    victims = _Victims()
    seen = {}

    def encode(module, items):
        # mid-stage: the planned victim bytes ARE the tower (first-fit-decreasing)
        w = module.blocks[0].attn.qkv_proj.weight
        seen["ptr"] = w.data_ptr()
        seen["victim_ptrs"] = {t.data_ptr() for t in victims.tensors.values()}
        seen["host"] = victims.host_bytes
        return vrr.encode_items(module, items)

    it = _Item()
    px = it.feature.clone()
    out = _run(_sched(), [_req("r", [it])], tmp_path, victims=victims, encode=encode)
    assert out.ok, out.detail
    assert torch.equal(it.precomputed_embeddings, _ref_rows(ck, px))
    big = victims.tensors["v0"]
    assert big.data_ptr() <= seen["ptr"] < big.data_ptr() + big.numel()  # largest victim first
    assert seen["host"] > 0 and victims.host_bytes == 0                   # image held, then freed
    assert victims.intact()
    assert "victim=fake" in out.victim_fields and "segments=1" in out.victim_fields
    assert {"stash", "restore", "verify"} <= set(out.legs_ms)
    assert out.victim_host_line.startswith(vv.W_VICTIM_HOST)


def _client_gone(module, items, req):
    req.to_abort = True  # the client disconnects while the tower encodes
    return vrr.encode_items(module, items)


@pytest.mark.parametrize("case", ["encode_error", "load_error", "deadline", "client_gone"])
def test_T5_every_way_out_returns_the_victims(tmp_path, case):
    _write_model(tmp_path)
    victims = _Victims()
    it = _Item()
    req = _req("r", [it])
    kw = {}
    if case == "encode_error":
        kw["encode"] = lambda m, i: (_ for _ in ()).throw(RuntimeError("CUDA out of memory"))
    elif case == "load_error":
        kw["build"] = _build(extra=True)          # an unfilled parameter: W106 in the load leg
    elif case == "deadline":
        ticks = iter(range(0, 1000, 10))
        kw["clock"] = lambda: float(next(ticks))  # 10 s per reading
        kw["deadline_s"] = 15.0
    else:
        kw["encode"] = lambda m, i: _client_gone(m, i, req)
    out = _run(_sched(), [req], tmp_path, victims=victims, **kw)
    assert victims.intact() and victims.host_bytes == 0
    assert out.checksum == "ok" and not out.fatal
    want = {"encode_error": vrr.W_ENCODE, "load_error": vrr.W_LOAD, "deadline": vrr.W_TIMEOUT,
            "client_gone": vrr.W_STAGE_OK}[case]
    assert out.code == want, out.detail
    if case == "client_gone":  # the stage runs to its end; the abort lands next pass
        assert out.ok and it.precomputed_embeddings is not None


@pytest.mark.parametrize("mode,needle", [("skip", "checksum MISMATCH"), ("raise", "restore raised")])
def test_T5_a_failed_return_is_W110c_and_the_pass_stops_the_group(tmp_path, mode, needle, monkeypatch):
    """Mutant 'restore skipped' and an H2D failure: the stage outcome is
    W110c (fatal), and the scheduler pass RAISES -- the group never admits on
    foreign bytes in its weights."""
    _write_model(tmp_path)
    victims = _Victims(mode=mode)
    out = _run(_sched(), [_req("r", [_Item()])], tmp_path, victims=victims)
    assert not out.ok and out.code == vv.W_VICTIM_NOT_RESTORED and needle in out.fatal
    assert victims.host_bytes == 0                       # the host image is freed anyway

    monkeypatch.setattr(vrr, "run_rank_stage", lambda s, reqs, **kw: out)
    monkeypatch.setattr(vrr, "_rank_device", lambda: torch.device("cpu"))
    s = types.SimpleNamespace(
        waiting_queue=[_req("r", [_Item()])], weg2_dormant=False, _weg2_vision_refused=set(),
        _weg2_vision_arm_refusal="", _weg2_vision_origin_aborts=[], _weg2_vision_runs=0,
        _weg2_vision_place=vrs.PLACE_WEIGHTS, _weg2_vision_victims=victims,
        server_args=types.SimpleNamespace(model_path=str(tmp_path)),
        model_config=types.SimpleNamespace(hf_config=None))
    with pytest.raises(vv.VisionVictimNotRestored, match="W110c"):
        vrr.vision_rank_pass(s)


def test_the_W110c_verdict_survives_a_raise_in_the_teardown(tmp_path, monkeypatch):
    """Review S1 (F0-H round 3, 27B line, where place=weights is the release
    standard): the return failed (W110c, written first), then the teardown
    raises. The pass must still stop the group with the FIRST verdict -- not
    rebuild a W_LOAD outcome without ``fatal`` and keep serving with foreign
    bytes in the PP0 MLP storages."""
    _write_model(tmp_path)
    victims = _Victims(mode="skip")                      # the return is skipped: checksum MISMATCH
    monkeypatch.setattr(vrr, "_strip_module", lambda module: (_ for _ in ()).throw(RuntimeError("teardown boom")))
    real = vrr.run_rank_stage
    monkeypatch.setattr(vrr, "run_rank_stage", lambda s, reqs, **kw: real(s, reqs, build=_build(), **kw))
    monkeypatch.setattr(vrr, "_rank_device", lambda: torch.device("cpu"))
    s = types.SimpleNamespace(
        waiting_queue=[_req("r", [_Item()])], weg2_dormant=False, _weg2_vision_refused=set(),
        _weg2_vision_arm_refusal="", _weg2_vision_origin_aborts=[], _weg2_vision_runs=0,
        _weg2_vision_place=vrs.PLACE_WEIGHTS, _weg2_vision_victims=victims,
        token_to_kv_pool_allocator=_Alloc(_kv()),
        server_args=types.SimpleNamespace(model_path=str(tmp_path)),
        model_config=types.SimpleNamespace(hf_config=None))
    with pytest.raises(vv.VisionVictimNotRestored, match="W110c.*checksum MISMATCH"):
        vrr.vision_rank_pass(s)


def test_a_host_image_that_cannot_be_written_is_W105b_before_any_byte_moved(tmp_path):
    _write_model(tmp_path)

    def no_ram(n):
        raise MemoryError("malloc failed")

    victims = _Victims(alloc=no_ram)
    it = _Item()
    out = _run(_sched(), [_req("r", [it])], tmp_path, victims=victims)
    assert not out.ok and out.code == vv.W_VICTIM_SHORT and "host image" in out.detail
    assert victims.intact() and not out.fatal and it.precomputed_embeddings is None


def test_the_encode_work_memory_comes_from_the_real_image_and_is_refused_by_name(tmp_path):
    """No own area cap (user 09.10.): the work memory is computed from the
    item's patches -- quadratic under sdpa for an item of more than one
    segment (test_weg2_vision_encoder_nomask_1009) -- and an encode the card's air
    cannot hold is W105b with the numbers, before any victim byte moved."""
    kw = dict(hidden=1152, intermediate=4304, heads=16, out_hidden=5120, merge=2, deepstack=3,
              in_dim=1536, quadratic=True)
    small, big = vv.encode_work_bytes(4096, **kw), vv.encode_work_bytes(16384, **kw)
    assert big - 4 * small == 3 * 16384 * 16384 - 4 * 3 * 4096 * 4096  # only the quad term is not linear
    assert vv.encode_work_bytes(16384, **{**kw, "quadratic": False}) * 4 < big * 4
    _write_model(tmp_path)
    victims = _Victims()
    vc = types.SimpleNamespace(hidden_size=1152, intermediate_size=4304, num_heads=16, out_hidden_size=5120,
                               spatial_merge_size=2, deepstack_visual_indexes=[8, 16, 24], patch_size=16,
                               in_channels=3, temporal_patch_size=2)
    it = _Item()
    it.image_grid_thw = torch.tensor([[1, 128, 128]])   # 2048x2048 at patch 16
    out = _run(_sched(), [_req("r", [it])], tmp_path, victims=victims,
               hf_config=types.SimpleNamespace(vision_config=vc), air=lambda d: (512 << 20, 0))
    assert not out.ok and out.code == vv.W_VICTIM_SHORT
    assert "16384 patches" in out.detail and "air is 512 MiB" in out.detail
    assert victims.stashed == 0 and victims.intact()


# ------------------------------------------------------- default unchanged --


def test_default_places_are_untouched_and_weights_is_never_async(tmp_path, monkeypatch, caplog):
    """The default (auto) and kvtail stages carry no victim fields and arm no
    source; place=weights never starts the async form, and an explicit
    SGLANG_WEG2_VISION_ASYNC=1 with it is a named arming refusal."""
    _write_model(tmp_path)
    out = _run(_sched(), [_req("r", [_Item()])], tmp_path, place=vrs.PLACE_KVTAIL)
    assert out.ok and not out.victim_fields and out.checksum == "" and out.place == vrs.PLACE_KVTAIL
    with caplog.at_level("INFO"):
        vrr.log_outcome(out, ["r"], 1)
    assert "victim=" not in caplog.text and vv.W_VICTIM_HOST not in caplog.text

    resolved = []
    monkeypatch.setattr(vv, "resolve_source", lambda s: resolved.append(s))
    s = types.SimpleNamespace(ps=types.SimpleNamespace(pp_rank=0, tp_size=1),
                              model_config=types.SimpleNamespace(hf_config=types.SimpleNamespace(vision_config=1)),
                              server_args=types.SimpleNamespace(model_path=str(tmp_path)))
    assert vrr.arm_rank_stage(s, {"SGLANG_WEG2_VISION": "transient"})
    assert s._weg2_vision_victims is None and resolved == [] and not s._weg2_vision_arm_refusal
    vrr.arm_rank_stage(s, {"SGLANG_WEG2_VISION": "transient", vrs.PLACE_ENV: "weights",
                           vrr.VISION_ASYNC_ENV: "1"})
    assert "VISION-SYNC LAW" in s._weg2_vision_arm_refusal and resolved == []

    started = []
    monkeypatch.setattr(vrr, "start_async_stage", lambda *a, **k: started.append(1))
    monkeypatch.setattr(vrr, "run_rank_stage", lambda s, reqs, **kw: vrr.StageOutcome())
    monkeypatch.setattr(vrr, "_rank_device", lambda: torch.device("cpu"))
    p = types.SimpleNamespace(
        waiting_queue=[_req("r", [_Item()])], weg2_dormant=False, _weg2_vision_refused=set(),
        _weg2_vision_arm_refusal="", _weg2_vision_origin_aborts=[], _weg2_vision_runs=0,
        _weg2_vision_place=vrs.PLACE_WEIGHTS, _weg2_vision_victims=None,
        server_args=types.SimpleNamespace(model_path=str(tmp_path)),
        model_config=types.SimpleNamespace(hf_config=None))
    vrr.vision_rank_pass(p)
    assert started == [] and p._weg2_vision_runs == 1
