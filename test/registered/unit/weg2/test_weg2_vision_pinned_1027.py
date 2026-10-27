"""VISION pinned standing copies (user order 27.10.: "pinned Dauerkopie im Host RAM").

Two copies stay in pinned host RAM for the life of PP0, so an image no longer
reads the tower from disk (load 356 ms) nor writes the victims to a pageable
buffer (stash 346 ms, metal 10.10., 1024^2): the TOWER (``SOURCE_PINNED``, one
H2D per image) and the VICTIM IMAGE (``HostImage(pinned=True)``, the D2H only
when the victims' device checksums changed).

Hermetic, CPU: the pinned buffers are plain CPU tensors (``alloc=``), the stage
is the runner tests' tiny tower.

Pinned (each case a distinct silent-failure path):
  * the victim image is rewritten whenever the victims' bytes or the plan
    changed -- a stale standing image restored after a weight exchange would
    put the OLD layout's weights back and corrupt the model silently;
  * a stage whose victims are unchanged copies nothing out, and still restores
    bitwise;
  * the per-stage pageable image (rollback switch / allocation failure) frees
    itself as before;
  * the tower loaded from the pinned copy is bitwise the one read from disk,
    and the stage does NOT touch the disk when the copy stands;
  * the default of the source follows the place and the rollback switch.
"""

import pytest
import torch

from sglang.srt.weg2 import vision_rank_runner as vrr
from sglang.srt.weg2 import vision_rank_stage as vrs
from sglang.srt.weg2 import vision_victim as vv
from test_weg2_vision_rank_runner import _build, _Item, _req, _write_model
from test_weg2_vision_victim_core import NO_AIR, _run, _sched, _Victims


def _plain(n):
    return torch.empty(n, dtype=torch.uint8)


def _views(*sizes, seed=3):
    g = torch.Generator().manual_seed(seed)
    return [torch.randint(0, 256, (n,), generator=g, dtype=torch.uint8) for n in sizes]


# ------------------------------------------------------- the victim image --


def test_standing_image_skips_the_d2h_when_unchanged_and_is_rewritten_when_not():
    img = vv.HostImage(_plain, pinned=True)
    views = _views(64, 32)
    sums = [vv.segment_checksum(v) for v in views]
    img.stash(views, sums=sums)
    assert not img.reused
    original = [v.clone() for v in views]

    for v in views:                      # the tower occupies the victims
        v.zero_()
    img.restore(views)
    img.release()                        # standing: the buffer stays
    assert img.host is not None and all(torch.equal(a, b) for a, b in zip(views, original))

    img.stash(views, sums=sums)          # same bytes, same plan: nothing copied out
    assert img.reused

    views[0][0] ^= 0x55                  # a weight exchange changed the victims
    img.stash(views, sums=[vv.segment_checksum(v) for v in views])
    assert not img.reused                # different checksums: rewritten
    assert img.host[0].item() == views[0][0].item()
    changed = [v.clone() for v in views]
    for v in views:
        v.zero_()                        # the tower occupies them
    img.restore(views)
    assert all(torch.equal(a, b) for a, b in zip(views, changed))   # the NEW layout comes back


def test_standing_image_is_rewritten_when_the_plan_changes_with_equal_sums():
    """Same victim bytes under another span layout (the plan moved a segment):
    the host bytes are laid out per span, so reusing them would restore each
    view from the wrong offset."""
    img = vv.HostImage(_plain, pinned=True)
    a, b = _views(48, 16)
    img.stash([a, b], sums=[vv.segment_checksum(a), vv.segment_checksum(b)])
    img.release()
    img.stash([b, a], sums=[vv.segment_checksum(a), vv.segment_checksum(b)])
    assert not img.reused
    b2, a2 = torch.zeros_like(b), torch.zeros_like(a)
    img.restore([b2, a2])
    assert torch.equal(b2, b) and torch.equal(a2, a)


def test_pageable_image_frees_itself_and_never_reuses():
    img = vv.HostImage(_plain)
    views = _views(40)
    sums = [vv.segment_checksum(views[0])]
    img.stash(views, sums=sums)
    img.release()
    assert img.host is None
    img.stash(views, sums=sums)
    assert not img.reused


# ------------------------------------------------------------ the stage ----


def test_two_stages_on_a_standing_image_restore_bitwise_and_the_second_copies_nothing(tmp_path):
    ck = _write_model(tmp_path)
    victims = _Victims()
    victims.keep_pinned(alloc=_plain)
    states = []
    for _ in range(2):
        it = _Item()
        out = _run(_sched(), [_req("r", [it])], tmp_path, victims=victims, air=NO_AIR)
        assert out.ok, out.detail
        assert victims.intact() and out.checksum == "ok"
        states.append(out.victim_fields.rsplit("host_image=", 1)[1])
    assert states == ["pinned-written", "pinned-reused"]
    assert victims.host_bytes > 0          # standing: still held after the release
    assert ck                               # the checkpoint was real


# ------------------------------------------------------- the tower source ---


def test_pinned_tower_is_bitwise_the_disk_tower_and_the_stage_does_not_read_the_disk(tmp_path, monkeypatch):
    _write_model(tmp_path)
    shard = vrr.find_tower_shard(str(tmp_path))
    src = vrs.stage_tower_to_pinned(shard, vrs.checkpoint_tensors(shard, vrr.is_vision_weight))
    assert src.kind == vrs.SOURCE_PINNED and src.host_bytes > 0 and src.shift % vrs.DIRECT_ALIGN == 0

    it_disk = _Item()
    px = it_disk.feature.clone()
    out = _run(_sched(), [_req("r", [it_disk])], tmp_path, victims=_Victims(), air=NO_AIR)
    assert out.ok and out.source == vrs.SOURCE_DISK

    def no_disk(*a, **k):
        raise AssertionError("the pinned source read the disk")

    monkeypatch.setattr(vrs, "read_into", no_disk)
    it_pin = _Item()
    it_pin.feature = px.clone()
    out = _run(_sched(), [_req("r", [it_pin])], tmp_path, victims=_Victims(), air=NO_AIR, source=src)
    assert out.ok, out.detail
    assert out.source == vrs.SOURCE_PINNED and out.direct is False
    assert torch.equal(it_pin.precomputed_embeddings, it_disk.precomputed_embeddings)


def test_a_tower_that_cannot_be_staged_pinned_falls_back_to_disk_by_name(tmp_path, caplog):
    """No tower in the checkpoint: the arming helper answers None and says so
    -- an image request is never refused for its source."""
    (tmp_path / "model.safetensors.index.json").write_text('{"weight_map": {}}')
    with caplog.at_level("ERROR"):
        assert vrr.arm_pinned_source(str(tmp_path)) is None
    assert vrr.W_ARM_REFUSED in caplog.text and "DISK" in caplog.text


# ------------------------------------------------------ the default source --


@pytest.mark.parametrize("env,place,want", [
    ({}, vrs.PLACE_WEIGHTS, vrs.SOURCE_PINNED),                                   # the new default
    ({vrs.PINNED_ENV: "0"}, vrs.PLACE_WEIGHTS, vrs.SOURCE_DISK),                  # rollback switch
    ({}, vrs.PLACE_AUTO, vrs.SOURCE_DISK),                                        # other places untouched
    ({}, None, vrs.SOURCE_DISK),
    ({vrs.SOURCE_ENV: "disk"}, vrs.PLACE_WEIGHTS, vrs.SOURCE_DISK),               # explicit wins
    ({vrs.SOURCE_ENV: "ram"}, vrs.PLACE_WEIGHTS, vrs.SOURCE_RAM),
])
def test_source_default_follows_place_and_switch(env, place, want):
    assert vrs.vision_source(env, place=place) == want


# ----------------------------------------------------------------- arming --

P_TRANSIENT = {"SGLANG_WEG2_VISION": "transient", "SGLANG_WEG2_GROUP": "P"}


def _arm_sched(model_path):
    import types

    return types.SimpleNamespace(
        ps=types.SimpleNamespace(pp_rank=0, tp_size=1),
        model_config=types.SimpleNamespace(hf_config=types.SimpleNamespace(vision_config=object())),
        server_args=types.SimpleNamespace(model_path=model_path, tokenizer_path=""),
    )


@pytest.mark.parametrize("extra,want", [
    ({}, vrs.SOURCE_PINNED),                      # place=weights: the standing copy is the default
    ({vrs.PINNED_ENV: "0"}, None),                # rollback switch: the per-stage disk read
])
def test_arming_under_place_weights_stages_the_pinned_tower_by_default(tmp_path, monkeypatch, extra, want):
    _write_model(tmp_path)
    monkeypatch.setattr(vrr, "arm_victims", lambda scheduler, env=None: (object(), ""))
    s = _arm_sched(str(tmp_path))
    env = {**P_TRANSIENT, vrs.PLACE_ENV: "weights", **extra}
    assert vrr.arm_rank_stage(s, env=env) is True and s._weg2_vision_arm_refusal == ""
    got = s._weg2_vision_source
    assert (got.kind if got is not None else None) == want


def test_arming_sizes_the_standing_victim_image_from_the_plan_and_survives_a_failed_pin(
        tmp_path, monkeypatch, caplog):
    """The image is allocated at arming for exactly the plan's victim bytes
    (so no image pays the cudaHostAlloc); when the pinned allocation fails the
    per-stage pageable image stays and the error is named -- host RAM is a
    warning, never a refusal."""
    _write_model(tmp_path)
    victims = _Victims()
    monkeypatch.setattr(vv, "resolve_source", lambda scheduler: victims)
    asked = []
    real = victims.keep_pinned

    def spy(nbytes, **kw):
        asked.append(nbytes)
        real(nbytes, alloc=_plain)

    monkeypatch.setattr(victims, "keep_pinned", spy)
    src, why = vrr.arm_victims(_arm_sched(str(tmp_path)), env={})
    assert why == "" and src is victims and victims.image_pinned
    assert len(asked) == 1 and asked[0] > 0

    victims2 = _Victims()
    monkeypatch.setattr(vv, "resolve_source", lambda scheduler: victims2)

    def no_pin(nbytes, **kw):
        raise RuntimeError("cudaHostAlloc failed")

    monkeypatch.setattr(victims2, "keep_pinned", no_pin)
    with caplog.at_level("ERROR"):
        src, why = vrr.arm_victims(_arm_sched(str(tmp_path)), env={})
    assert why == "" and src is victims2 and not victims2.image_pinned
    assert "NOT allocated" in caplog.text
