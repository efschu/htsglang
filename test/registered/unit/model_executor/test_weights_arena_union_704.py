"""#704 slice 1a: the union arena, and the zero-copy rung change.

The ladder needs to move a PP layer boundary at runtime. There is no cross-rank
weight mover (`regime_stages.py:100`, REACH_NO_WEIGHT_MOVER), so the design
assumed an H2D `arena_refill` per rung change -- one contiguous copy of the
target layout's host image, 451-901 MiB per step.

That is more work than the primitives require. `plan_arena_layout` fixes slot
offsets by sorted tensor name, and `bind_arena_views` rebinds parameters to
arena views **without copying any bytes**. So if the arena is planned over the
UNION of every rung's tensors rather than per rung, then:

  * every tensor shared by two rungs sits at the SAME offset in both, and
  * a rung change is a rebind plus a PP-boundary change -- zero weight bytes.

The price is residency, not bandwidth: a layer that changes hands must be
resident on BOTH ranks, so the world holds `64 + (boundary layers that move)`
layer-images instead of 64. For the slice-1a pair that is one extra layer.

This is the same trade the arena model already prices (DESIGN_704 §3.7): pay in
resident VRAM, not in per-change link time. It is strictly better than the
refill design on this rig, where a rank-to-rank transfer would stage through
host anyway.

WHAT THIS FILE DOES NOT CLAIM. It proves the WEIGHT side is copy-free. GDN
(linear) layers also carry per-sequence recurrent state -- temporal_state 19.5
MiB/layer + conv_state 0.762 -- which lives with the layer and must either move
or be absent. Slice 1a therefore flips only at quiescence, where there is no
live state to preserve. Live-state transfer is slice 1b's problem and is NOT
addressed here.

Hermetic: CPU tensors only, no CUDA, no server.
"""

import pytest

torch = pytest.importorskip("torch")

from sglang.srt.model_executor.weights_arena import plan_arena_layout
from sglang.srt.model_executor.weights_arena_union import (
    UnionArenaError,
    flip_delta,
    plan_union_arena,
)

HIDDEN = 8


def _layer_tensors(idx: int) -> dict:
    """One layer's tensors, deterministic content keyed by layer index."""
    g = torch.Generator().manual_seed(1000 + idx)
    return {
        f"model.layers.{idx}.attn.w": torch.randn(
            HIDDEN, HIDDEN, generator=g, dtype=torch.float32
        ),
        f"model.layers.{idx}.mlp.w": torch.randn(
            HIDDEN, 2 * HIDDEN, generator=g, dtype=torch.float32
        ),
    }


def _named_for(layer_ids) -> dict:
    out = {}
    for i in layer_ids:
        out.update(_layer_tensors(i))
    return out


# Slice 1a pair on rank0: [28,20,16] -> [29,19,16]. Rank0 goes 0..27 -> 0..28.
# Both cuts have attention profile (7,5,4) and layer 28 is a LINEAR layer under
# full_attention_interval 4, so no attention layer and no KV row moves.
RUNG_A_RANK0 = tuple(range(28))
RUNG_B_RANK0 = tuple(range(29))


def _rank0_rungs() -> dict:
    return {
        "[28,20,16]": _named_for(RUNG_A_RANK0),
        "[29,19,16]": _named_for(RUNG_B_RANK0),
    }


def test_shared_tensors_keep_identical_offsets_across_rungs():
    """THE zero-copy invariant. If this fails the whole slice is unfounded.

    A tensor present in both rungs must occupy the same arena bytes in both,
    so switching rungs cannot require moving it.
    """
    plan = plan_union_arena(_rank0_rungs())
    a = plan.active_slots("[28,20,16]")
    b = plan.active_slots("[29,19,16]")
    shared = set(a) & set(b)
    assert shared, "the two rungs share no tensors at all; fixture is wrong"
    for name in shared:
        assert a[name].offset == b[name].offset
        assert a[name].nbytes == b[name].nbytes


def test_a_rung_change_moves_zero_weight_bytes():
    """The claim, stated as bytes: the delta is entirely 'newly active'."""
    plan = plan_union_arena(_rank0_rungs())
    delta = flip_delta(plan, "[28,20,16]", "[29,19,16]")
    assert delta.bytes_to_copy == 0
    # Layer 28 becomes active on this rank; nothing else changes.
    assert all(".28." in n for n in delta.activated)
    assert delta.deactivated == ()


def test_the_price_is_residency_and_it_is_quantified():
    """The union costs exactly the layers that are not in every rung."""
    rungs = _rank0_rungs()
    plan = plan_union_arena(rungs)
    per_rung = {
        name: plan_arena_layout(named).total_bytes for name, named in rungs.items()
    }
    # The arena must hold the union, which is the larger rung here.
    assert plan.total_bytes >= max(per_rung.values())
    # And the overhead over the SMALLER rung is exactly layer 28's tensors.
    layer28 = sum(t.untyped_storage().nbytes() for t in _layer_tensors(28).values())
    assert plan.total_bytes - per_rung["[28,20,16]"] == pytest.approx(layer28, rel=0.01)


def test_binding_a_rung_copies_nothing_and_preserves_content():
    """Rebind is not a copy: the arena's bytes must be untouched by binding."""
    plan = plan_union_arena(_rank0_rungs())
    arena = torch.zeros(plan.total_bytes, dtype=torch.uint8)
    # Write a recognisable pattern, then bind each rung and confirm no change.
    arena.random_(0, 256, generator=torch.Generator().manual_seed(7))
    before = arena.clone()
    for rung in ("[28,20,16]", "[29,19,16]"):
        views = plan.bind(rung, arena)
        assert views
        assert torch.equal(arena, before), f"binding {rung} mutated the arena"


def test_views_alias_the_arena_rather_than_copying_it():
    """A view that does not alias would silently make the flip a no-op."""
    plan = plan_union_arena(_rank0_rungs())
    arena = torch.zeros(plan.total_bytes, dtype=torch.uint8)
    views = plan.bind("[29,19,16]", arena)
    name = "model.layers.28.attn.w"
    views[name].fill_(1.0)
    assert arena.sum().item() > 0, "the view is a copy, not an alias"


def test_conflicting_tensor_definitions_across_rungs_are_refused():
    """The same name must mean the same bytes in every rung, or the union is
    not a union and an 'unchanged' tensor could silently change shape."""
    rungs = _rank0_rungs()
    bad = dict(rungs["[29,19,16]"])
    bad["model.layers.5.attn.w"] = torch.randn(HIDDEN + 1, HIDDEN)
    with pytest.raises(UnionArenaError, match="disagree"):
        plan_union_arena({"a": rungs["[28,20,16]"], "b": bad})


def test_dtype_conflicts_are_refused_too():
    rungs = _rank0_rungs()
    bad = dict(rungs["[29,19,16]"])
    bad["model.layers.5.attn.w"] = rungs["[29,19,16]"]["model.layers.5.attn.w"].to(
        torch.float16
    )
    with pytest.raises(UnionArenaError, match="disagree"):
        plan_union_arena({"a": rungs["[28,20,16]"], "b": bad})


def test_an_unknown_rung_is_refused_not_silently_empty():
    plan = plan_union_arena(_rank0_rungs())
    with pytest.raises(UnionArenaError, match="not a planned rung"):
        plan.active_slots("[35,14,15]")
    with pytest.raises(UnionArenaError, match="not a planned rung"):
        flip_delta(plan, "[28,20,16]", "[35,14,15]")


def test_a_single_rung_plan_is_refused():
    """A union arena with one rung cannot flip and is a configuration error."""
    with pytest.raises(UnionArenaError, match="at least two"):
        plan_union_arena({"only": _named_for(range(4))})


def test_an_arena_too_small_for_the_union_is_refused_before_binding():
    plan = plan_union_arena(_rank0_rungs())
    small = torch.zeros(plan.total_bytes - 1, dtype=torch.uint8)
    with pytest.raises(UnionArenaError, match="holds"):
        plan.bind("[29,19,16]", small)


def test_flip_is_symmetric_in_bytes_and_antisymmetric_in_sets():
    """Flipping back must cost the same (zero) and invert the delta."""
    plan = plan_union_arena(_rank0_rungs())
    fwd = flip_delta(plan, "[28,20,16]", "[29,19,16]")
    back = flip_delta(plan, "[29,19,16]", "[28,20,16]")
    assert fwd.bytes_to_copy == back.bytes_to_copy == 0
    assert set(fwd.activated) == set(back.deactivated)
    assert set(fwd.deactivated) == set(back.activated)


def test_gdn_state_is_declared_out_of_scope_not_silently_ignored():
    """Slice 1a proves the WEIGHT side only.

    The plan must say, in its own data, that a moved layer's per-sequence
    state is not covered -- so nobody reads 'zero bytes' as 'nothing to do'.
    """
    plan = plan_union_arena(_rank0_rungs())
    delta = flip_delta(plan, "[28,20,16]", "[29,19,16]")
    assert delta.requires_quiescence is True
    assert "state" in delta.out_of_scope.lower()


def test_per_rung_layouts_really_would_move_resident_weights():
    """CAN-FAIL PROOF that the union is doing work rather than restating a
    coincidence.

    If per-rung layouts happened to place shared tensors at the same offsets,
    the union would be pointless and every test above would pass vacuously.
    They do not: `plan_arena_layout` orders slots by SORTED NAME, and layer
    names sort lexically, so inserting `model.layers.28.*` lands between
    `...27.*` and `...3.*` and shifts every slot after it.

    The consequence is the concrete motivation for this module: with per-rung
    layouts, a rung change would have to relocate weights that are already
    resident and unchanged, purely because a name sorted differently.
    """
    rungs = _rank0_rungs()
    a_named, b_named = rungs["[28,20,16]"], rungs["[29,19,16]"]
    per_a = plan_arena_layout(a_named)
    per_b = plan_arena_layout(b_named)
    shared = set(a_named) & set(b_named)

    displaced = [
        n for n in shared if per_a.slot_of(n).offset != per_b.slot_of(n).offset
    ]
    assert displaced, (
        "per-rung layouts placed every shared tensor identically, so the union "
        "buys nothing and the zero-copy tests above prove nothing"
    )

    # Under the union, none of them move.
    plan = plan_union_arena(rungs)
    ua = plan.active_slots("[28,20,16]")
    ub = plan.active_slots("[29,19,16]")
    assert [n for n in shared if ua[n].offset != ub[n].offset] == []
