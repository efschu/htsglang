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
