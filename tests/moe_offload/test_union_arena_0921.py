"""The cross-process UNION weights arena, slice 1 (2026-09-21): plan, manifest,
peer verification.

fnFL2 v28 died of a CUDA OOM on the 5090 with both groups' weights resident at
once (16.08 + 14.99 of 31.34 GiB). The union arena holds the byte-identical
half ONCE. These tests pin the two properties that make that safe: one offset
per name across both phases, and a content check that refuses two tensors that
merely LOOK alike (the expert-index-sharding trap).

Hermetic: CPU tensors only, no CUDA.
"""

import pytest
import torch

from sglang.srt.weg2 import union_arena as ua


def _phase_sets():
    torch.manual_seed(11)
    shared_attn = {
        "layers.0.attn.qkv": torch.randn(8, 16),
        "layers.0.attn.o": torch.randn(16, 8),
        "layers.1.attn.qkv": torch.randn(8, 16),
    }
    p_named = dict(shared_attn)
    p_named["layers.0.mlp.up"] = torch.randn(32, 8)  # only P holds this
    d_named = dict(shared_attn)
    d_named["draft.mtp.head"] = torch.randn(4, 16)  # only D holds this
    return p_named, d_named


def test_shared_names_get_one_offset_and_the_saving_is_reported():
    p_named, d_named = _phase_sets()
    plan = ua.plan_card_union({ua.PHASE_P: p_named, ua.PHASE_D: d_named})
    # every name appears exactly once in the union layout
    names = [s.name for s in plan.layout.slots]
    assert len(names) == len(set(names)) == 5
    saving = ua.union_saving(plan, {ua.PHASE_P: p_named, ua.PHASE_D: d_named})
    shared_bytes = sum(
        t.numel() * t.element_size()
        for n, t in p_named.items()
        if n in d_named
    )
    # the union is the two own layouts minus the overlap counted twice
    assert saving["separate"] == saving["own_P"] + saving["own_D"]
    assert saving["saved"] > 0
    # the saving is the shared bytes (alignment makes it >=, never less)
    assert saving["saved"] >= shared_bytes - 4 * 256
    assert saving["union"] == plan.total_bytes


def test_a_name_that_means_two_different_tensors_is_refused():
    p_named, d_named = _phase_sets()
    d_named["layers.0.attn.qkv"] = torch.randn(9, 16)  # different shape
    with pytest.raises(ua.UnionArenaError, match="disagree about tensor"):
        ua.plan_card_union({ua.PHASE_P: p_named, ua.PHASE_D: d_named})


def test_the_phase_pair_and_aliases_are_validated():
    p_named, d_named = _phase_sets()
    with pytest.raises(ua.UnionShareError, match="exactly the phases"):
        ua.plan_card_union({"rung0": p_named, "rung1": d_named})
    with pytest.raises(ua.UnionShareError, match="out of scope for V1"):
        ua.plan_card_union(
            {ua.PHASE_P: p_named, ua.PHASE_D: d_named},
            alias_of_by_phase={ua.PHASE_P: {"a": "b"}},
        )


def _manifest(p_named, d_named, owner=ua.PHASE_P):
    plan = ua.plan_card_union({ua.PHASE_P: p_named, ua.PHASE_D: d_named})
    merged = dict(d_named)
    merged.update(p_named)
    return plan, ua.build_manifest(
        plan,
        tag="fnFL2",
        card="GPU-31d7ef41",
        owner_phase=owner,
        checksums=ua.checksums_for(merged),
    )


def test_manifest_round_trips_through_json():
    p_named, d_named = _phase_sets()
    _, manifest = _manifest(p_named, d_named)
    back = ua.UnionManifest.from_json(manifest.to_json())
    assert back == manifest
    assert back.slot_of("layers.0.attn.o").dtype is torch.float32
    assert set(back.active_for(ua.PHASE_D)) == set(d_named)
    assert back.checksum_of("draft.mtp.head") == manifest.checksum_of(
        "draft.mtp.head"
    )


def test_a_manifest_slot_without_a_checksum_is_refused():
    p_named, d_named = _phase_sets()
    plan = ua.plan_card_union({ua.PHASE_P: p_named, ua.PHASE_D: d_named})
    sums = ua.checksums_for(p_named)  # D's own tensor has none
    with pytest.raises(ua.UnionShareError, match="carry no checksum"):
        ua.build_manifest(
            plan, tag="t", card="c", owner_phase=ua.PHASE_P, checksums=sums
        )


def test_the_peer_binds_the_shared_half_and_keeps_its_own():
    p_named, d_named = _phase_sets()
    _, manifest = _manifest(p_named, d_named)
    binding = ua.verify_peer(manifest, ua.PHASE_D, d_named)
    assert sorted(s.name for s in binding.shared) == [
        "draft.mtp.head",
        "layers.0.attn.o",
        "layers.0.attn.qkv",
        "layers.1.attn.qkv",
    ]
    # D's own tensor IS in the union (the owner planned over both phases), so
    # nothing of D's is private here; P's mlp.up is simply not D's to bind.
    assert binding.private == ()
    assert binding.shared_bytes == sum(
        t.numel() * t.element_size() for t in d_named.values()
    )


def test_same_shape_different_bytes_is_refused_by_the_checksum():
    """The expert-index-sharding trap: rank r holds experts [r*k, (r+1)*k)."""
    p_named, d_named = _phase_sets()
    _, manifest = _manifest(p_named, d_named)
    impostor = dict(d_named)
    impostor["layers.0.attn.qkv"] = torch.randn(8, 16)  # same shape, other bytes
    with pytest.raises(ua.UnionShareError, match="different BYTES"):
        ua.verify_peer(manifest, ua.PHASE_D, impostor)


def test_a_shape_mismatch_against_the_manifest_is_refused():
    p_named, d_named = _phase_sets()
    _, manifest = _manifest(p_named, d_named)
    bad = dict(d_named)
    bad["layers.0.attn.o"] = torch.randn(16, 9)
    with pytest.raises(ua.UnionShareError, match="do not hold the same tensor"):
        ua.verify_peer(manifest, ua.PHASE_D, bad, require_checksums=False)


def test_an_unknown_phase_is_refused():
    p_named, d_named = _phase_sets()
    _, manifest = _manifest(p_named, d_named)
    with pytest.raises(ua.UnionShareError, match="manifest covers"):
        ua.verify_peer(manifest, "X", d_named)


def test_tensors_outside_the_manifest_stay_private():
    p_named, d_named = _phase_sets()
    _, manifest = _manifest(p_named, d_named)
    extra = dict(d_named)
    extra["experts.shard.3"] = torch.randn(64, 8)  # never planned
    binding = ua.verify_peer(manifest, ua.PHASE_D, extra)
    assert binding.private == ("experts.shard.3",)
    assert binding.private_bytes == 64 * 8 * 4


def test_byte_view_refuses_a_partial_view():
    base = torch.randn(64)
    with pytest.raises(ua.UnionShareError):
        ua.byte_view(base[8:16])


# ---------------------------------------------------------------- the census


def _side(tmp_path, phase, tensors, sums=None, card="GPU-aaaabbbbcccc"):
    import torch as _t

    return ua.publish_side(
        str(tmp_path),
        tag="fnFL2",
        card=card,
        phase=phase,
        rank=0,
        named=tensors,
        with_checksums=sums is None,
    )


def test_the_census_separates_shareable_from_colliding(tmp_path):
    torch.manual_seed(5)
    same = torch.randn(64, 8)
    p_named = {"attn.qkv": same, "attn.o": torch.randn(8, 8), "p.only": torch.randn(4)}
    d_named = {
        "attn.qkv": same.clone(),  # SAME bytes -> shareable
        "attn.o": torch.randn(8, 8),  # same shape, other bytes -> collide
        "d.only": torch.randn(9),
    }
    _side(tmp_path, ua.PHASE_P, p_named)
    _side(tmp_path, ua.PHASE_D, d_named)
    census = ua.join_sides(str(tmp_path), "GPU-aaaabbbbcccc")
    assert census.shareable == ("attn.qkv",)
    assert census.shareable_bytes == 64 * 8 * 4
    assert census.colliding == ("attn.o",)
    assert census.colliding_bytes == 8 * 8 * 4
    assert census.incompatible == () and census.incompatible_bytes == 0
    assert census.saved_bytes == census.shareable_bytes
    assert dict(census.private_bytes)["P"] == 4 * 4  # p.only
    own = dict(census.own_bytes)
    assert census.union_bytes == own["P"] + own["D"] - census.shareable_bytes
    line = census.line()
    assert "SHAREABLE=" in line and "collide=1" in line


def test_a_shape_difference_is_incompatible_not_colliding(tmp_path):
    p_named = {"w": torch.randn(4, 4)}
    d_named = {"w": torch.randn(4, 5)}
    _side(tmp_path, ua.PHASE_P, p_named, card="GPU-shape")
    _side(tmp_path, ua.PHASE_D, d_named, card="GPU-shape")
    census = ua.join_sides(str(tmp_path), "GPU-shape")
    assert census.incompatible == ("w",) and census.shareable == ()
    assert census.incompatible_bytes == 4 * 5 * 4  # the larger of the two
    name, nbytes, p_shape, d_shape = census.top_incompatible[0]
    assert name == "w" and p_shape == "float32(4, 4)" and d_shape == "float32(4, 5)"


def test_a_missing_side_is_a_named_refusal(tmp_path):
    _side(tmp_path, ua.PHASE_P, {"w": torch.randn(2)}, card="GPU-lonely")
    with pytest.raises(ua.UnionShareError, match="has not published"):
        ua.join_sides(str(tmp_path), "GPU-lonely")


def test_a_side_without_checksums_is_never_read_as_identical(tmp_path):
    """No proof is not evidence of sameness."""
    same = torch.randn(8, 8)
    for phase in (ua.PHASE_P, ua.PHASE_D):
        ua.publish_side(
            str(tmp_path),
            tag="t",
            card="GPU-nosum",
            phase=phase,
            rank=0,
            named={"w": same},
            with_checksums=False,
        )
    census = ua.join_sides(str(tmp_path), "GPU-nosum")
    assert census.shareable == () and census.incompatible == ("w",)


def test_a_draft_worker_does_not_overwrite_the_main_model_side(tmp_path):
    """fnFL2 v29: the draft worker is a second rank of the same phase on the
    same card; keyed without the role it overwrote the main model's side and
    the census compared P's model against D's DRAFT."""
    main = {"w": torch.randn(4, 4)}
    draft = {"d": torch.randn(2, 2)}
    ua.publish_side(str(tmp_path), tag="t", card="GPU-role", phase=ua.PHASE_D,
                    rank=0, named=main, role="main")
    ua.publish_side(str(tmp_path), tag="t", card="GPU-role", phase=ua.PHASE_D,
                    rank=0, named=draft, role="draft")
    ua.publish_side(str(tmp_path), tag="t", card="GPU-role", phase=ua.PHASE_P,
                    rank=0, named=main, role="main")
    census = ua.join_sides(str(tmp_path), "GPU-role", role="main")
    assert census.shareable == ("w",)


def test_meta_tensors_are_skipped_not_checksummed(tmp_path):
    """A Form A worker holds the draft model as meta; uint8_checksum raises
    on meta, and the census must survive that rather than skip the boot."""
    named = {"real": torch.randn(4), "ghost": torch.empty(8, device="meta")}
    path = ua.publish_side(str(tmp_path), tag="t", card="GPU-meta",
                           phase=ua.PHASE_P, rank=0, named=named)
    import json as _json

    side = _json.load(open(path))
    assert side["skipped_meta"] == 1
    assert [t["name"] for t in side["tensors"]] == ["real"]


def test_the_census_hook_is_off_without_its_env(monkeypatch):
    monkeypatch.delenv(ua.UNION_DIR_ENV, raising=False)
    assert ua.maybe_union_census(object(), rank=0, device=0) is None
