# SPDX-License-Identifier: Apache-2.0
"""#785: the arena layout must be answerable WITHOUT allocating the weights.

WHY. The flip seam's arena tail is ``max(0, layout_pp.total_bytes -
layout_tp.total_bytes)`` (``phase_flip_boot.refill_high_water_bytes``), and it
is the term that binds the KV pool on this rig: 2215 MiB on rank 2 at cut
(31,16,17) / vector (32,16,16), which raises that rank's arming floor from
1523 to 3226 MiB and holds the pool at 161378 tokens.

Today it can only be MEASURED, because ``layout_tp`` is built in
``phase_flip_boot`` from a real TP-shaped worker -- long after the KV pool has
been sized. That ordering, not the physics, is what forces the two-boot
protocol and the cold/warm cliff: a first boot prices the tail at 0 and sizes
4.4x too large, and every later boot inherits a number from its predecessor.

The layout itself needs no storage. ``shape``, ``stride``, ``dtype``,
``storage_offset`` and the storage SIZE are all correct on a meta tensor, so a
meta-device parameter set answers ``total_bytes`` exactly -- same alignment,
same ordering, same code path.

THE ONE EXCEPTION IS THE WHOLE HAZARD. ``data_ptr()`` is 0 for every meta
tensor, and that is what ``plan_arena_layout`` uses to detect aliasing. Fed
meta tensors, it folds the entire model into a single slot and returns a total
that is wrong by orders of magnitude and structurally perfect. A number like
that goes straight into the pool solve.

So: aliasing comes from the caller, and the inference path refuses meta input.
"""

import pytest
import torch

from sglang.srt.model_executor.weights_arena import (
    WeightsArenaError,
    plan_arena_layout,
)


def _real_set():
    """A parameter set with a genuine alias (tied embedding / lm_head)."""
    embed = torch.empty(64, 32, dtype=torch.float16)
    named = {
        "lm_head.weight": embed,  # tied: SAME storage as the embedding
        "model.embed_tokens.weight": embed,
        "model.layers.0.mlp.weight": torch.empty(128, 32, dtype=torch.float16),
        "model.layers.1.mlp.weight": torch.empty(128, 32, dtype=torch.float16),
    }
    return named


def _meta_like(named):
    """The same set, described on the meta device -- no storage anywhere."""
    return {
        name: torch.empty(tuple(t.shape), dtype=t.dtype, device="meta")
        for name, t in named.items()
    }


# ---------------------------------------------------------------------------
# The hazard, demonstrated rather than asserted.
# ---------------------------------------------------------------------------


def test_meta_input_is_refused_by_the_inference_path():
    """It must REFUSE, not return a plausible wrong number."""
    with pytest.raises(WeightsArenaError, match="meta device"):
        plan_arena_layout(_meta_like(_real_set()))


def test_the_refusal_is_not_pedantry_the_silent_answer_was_badly_wrong():
    """CAN-FAIL PROOF for the guard: price the bug it prevents.

    Reproduces the pre-#785 arithmetic -- storage identity on meta tensors --
    and shows what it would have charged into the pool solve.
    """
    named = _real_set()
    honest = plan_arena_layout(named).total_bytes

    # What data_ptr()-based aliasing yields when every pointer is 0: one slot.
    meta = _meta_like(named)
    first = sorted(meta)[0]
    collapsed = meta[first].numel() * meta[first].element_size()

    assert collapsed < honest, "expected the collapse to UNDERSTATE the arena"
    assert honest / collapsed > 3, (
        "the silent answer was not marginally wrong -- on a real checkpoint "
        "this ratio is the whole model over one tensor"
    )


# ---------------------------------------------------------------------------
# The fix: same total, no allocation.
# ---------------------------------------------------------------------------


def test_meta_layout_equals_the_real_layout_byte_for_byte():
    """THE ACCEPTANCE PROPERTY for #785's sizing input.

    Equality is exact, not approximate: this number is subtracted from a
    memory budget, so a rounding difference is a real difference.
    """
    named = _real_set()
    real = plan_arena_layout(named)

    meta = _meta_like(named)
    # The tie is a property of the CHECKPOINT, known without loading it.
    derived = plan_arena_layout(
        meta, alias_of={"model.embed_tokens.weight": "lm_head.weight"}
    )

    assert derived.total_bytes == real.total_bytes
    assert [s.name for s in derived.slots] == [s.name for s in real.slots]
    assert [s.offset for s in derived.slots] == [s.offset for s in real.slots]
    assert [s.nbytes for s in derived.slots] == [s.nbytes for s in real.slots]
    assert sorted(derived.aliases) == sorted(real.aliases)


def test_alias_of_also_works_on_real_tensors_and_agrees_with_inference():
    """The explicit relation is not a meta-only code path.

    If it disagreed with storage inference on real tensors, the derived number
    would not be the one the runtime later measures.
    """
    named = _real_set()
    inferred = plan_arena_layout(named)
    declared = plan_arena_layout(
        named, alias_of={"model.embed_tokens.weight": "lm_head.weight"}
    )
    assert declared.total_bytes == inferred.total_bytes
    assert sorted(declared.aliases) == sorted(inferred.aliases)


def test_untied_weights_are_two_slots_not_one():
    """The alias must be declared to exist -- absence means two slots.

    Guards the direction that would UNDER-size the arena: silently treating an
    untied lm_head as tied.
    """
    named = _real_set()
    named["lm_head.weight"] = torch.empty(64, 32, dtype=torch.float16)  # untied
    untied = plan_arena_layout(named)
    tied = plan_arena_layout(_real_set())
    assert untied.total_bytes > tied.total_bytes
    assert untied.aliases == ()


# ---------------------------------------------------------------------------
# The declared relation has to be checkable, or it is just another guess.
# ---------------------------------------------------------------------------


def test_a_mismatched_alias_is_refused():
    meta = _meta_like(_real_set())
    meta["model.layers.0.mlp.weight"] = torch.empty(
        8, 8, dtype=torch.float16, device="meta"
    )
    with pytest.raises(WeightsArenaError, match="different view"):
        plan_arena_layout(
            meta, alias_of={"model.layers.0.mlp.weight": "lm_head.weight"}
        )


def test_a_chained_alias_is_refused():
    """A chain leaves the owning tensor unslotted, i.e. undersizes the arena."""
    meta = _meta_like(_real_set())
    with pytest.raises(WeightsArenaError, match="chained"):
        plan_arena_layout(
            meta,
            alias_of={
                "model.embed_tokens.weight": "lm_head.weight",
                "lm_head.weight": "model.layers.0.mlp.weight",
            },
        )


def test_an_alias_to_an_absent_tensor_is_refused():
    meta = _meta_like(_real_set())
    with pytest.raises(WeightsArenaError, match="absent canonical"):
        plan_arena_layout(meta, alias_of={"lm_head.weight": "nope.weight"})


def test_alias_of_naming_an_unknown_tensor_is_refused():
    meta = _meta_like(_real_set())
    with pytest.raises(WeightsArenaError, match="not in the set"):
        plan_arena_layout(meta, alias_of={"ghost.weight": "lm_head.weight"})
