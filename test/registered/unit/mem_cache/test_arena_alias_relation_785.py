# SPDX-License-Identifier: Apache-2.0
"""#785: the alias relation a meta layout needs, and the proof it can fail.

``plan_arena_layout`` detects aliasing by storage identity and refuses meta
input, because ``data_ptr()`` is 0 for every meta tensor and would fold a whole
model into one slot. ``storage_alias_relation`` supplies the relation instead.

The hazard specific to THIS helper is that its healthy answer on the model that
motivated it is the EMPTY relation -- that architecture has no two distinct
parameter objects sharing a storage. An empty dict is also exactly what a
detector that does not work returns, so "it returned {}" proves nothing on its
own. These tests separate the two.
"""

import torch

from sglang.srt.managers.arena_tail_probe import (
    arena_tail_bytes,
    plan_meta_layout,
    storage_alias_relation,
)
from sglang.srt.model_executor.weights_arena import plan_arena_layout


def _independent(device):
    return {
        "b.weight": torch.empty(64, 32, dtype=torch.float16, device=device),
        "a.weight": torch.empty(128, 32, dtype=torch.float16, device=device),
    }


def _with_alias(device):
    shared = torch.empty(64, 32, dtype=torch.float16, device=device)
    return {
        "b.weight": shared,
        # a DIFFERENT tensor object over the SAME storage
        "a.weight": shared.view(64, 32),
        "c.weight": torch.empty(128, 32, dtype=torch.float16, device=device),
    }


# ---------------------------------------------------------------------------
# The detector has to FIND an alias, on meta, or its empty answer means nothing.
# ---------------------------------------------------------------------------


def test_it_finds_an_alias_on_the_meta_device():
    """CAN-FAIL PROOF. data_ptr() is 0 for all three of these tensors."""
    named = _with_alias("meta")
    assert all(t.data_ptr() == 0 for t in named.values())
    assert named["a.weight"] is not named["b.weight"]

    relation = storage_alias_relation(named)
    # canonical is the first name in sorted order, i.e. "a.weight"
    assert relation == {"b.weight": "a.weight"}


def test_it_reports_no_alias_when_there_is_none():
    """The other direction: independent tensors must not be merged.

    Without this, a detector that called everything an alias would also 'find'
    the alias above, and would undersize every arena it is asked about.
    """
    assert storage_alias_relation(_independent("meta")) == {}


def test_the_meta_relation_is_the_one_storage_inference_finds_on_real_tensors():
    """The derived number has to be the one the runtime later measures.

    The runtime infers aliasing from real storages. If this helper disagreed
    with that inference, the boot would derive one total and measure another.
    """
    real = _with_alias("cpu")
    inferred = plan_arena_layout(real)
    assert inferred.aliases, "expected storage inference to find the alias"

    declared = storage_alias_relation(real)
    assert declared == {a: c for a, c in inferred.aliases}


def test_a_meta_layout_equals_the_real_layout_it_stands_in_for():
    """THE ACCEPTANCE PROPERTY. Equality is exact: this is a budget input."""
    real = _with_alias("cpu")
    meta = {
        name: torch.empty(tuple(t.shape), dtype=t.dtype, device="meta")
        for name, t in real.items()
    }
    # rebuild the sharing on the meta side the way the module graph would
    meta["b.weight"] = meta["a.weight"].view(64, 32)

    assert plan_meta_layout(meta).total_bytes == plan_arena_layout(real).total_bytes


def test_the_relation_is_stable_and_not_an_artifact_of_repeated_calls():
    """``untyped_storage()`` returns a fresh wrapper per call.

    If the handle behind it were not stable, the relation would depend on how
    many times each tensor happened to be inspected.
    """
    named = _with_alias("meta")
    assert storage_alias_relation(named) == storage_alias_relation(named)


# ---------------------------------------------------------------------------
# The subtraction itself.
# ---------------------------------------------------------------------------


def test_the_tail_is_the_difference_of_the_two_totals():
    assert arena_tail_bytes(10789, 8573) == 2216


def test_a_smaller_pp_layout_has_no_tail_rather_than_a_negative_one():
    """The leg that killed all three ranks on 2026-08-11, as a clamp.

    A negative tail would be SUBTRACTED from the arming floor and would hand
    the pool memory the arena still owns.
    """
    assert arena_tail_bytes(8008, 8573) == 0
