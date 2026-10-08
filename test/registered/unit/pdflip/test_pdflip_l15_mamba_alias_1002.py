# SPDX-License-Identifier: Apache-2.0
"""L15-FIX-MAMBA-ALIAS (N3l 02.10. 02:29:57Z): after a held retain round the
radix tree must not name a mamba slot the allocator holds as free.

N3l: the first armed hold (L15-RETAIN epoch=6 n=1, KEEP-ALIGN ok) was
followed on all three D ranks by the idle leak check
"[mamba] total=38 available=37 evictable=6 ... free_and_cached=5, #924 MAMBA
SLOT ALIASING: mamba_num_used=-5" -> ValueError -> boot dead. reset_keep
keeps the whole chain of a held request; the mamba allocator is re-armed
with ONLY the held anchors' slots (retain step 6); rewrite_tree_chain only
REMAPPED the chain's mamba values, so every intermediate checkpoint on the
chain kept naming a slot that went back to the free list.

The check used here is the REAL one: InvariantChecker._mamba_double_claimed
(the free_and_cached term of the idle leak check), on a REAL
UnifiedRadixCache built with the fixture of test_pdflip_l15_tree_rewrite_1001.
"""

from __future__ import annotations

import importlib.util
import os

import torch

from flliper.srt.managers.scheduler_components.invariant_checker import (
    SchedulerInvariantChecker as InvariantChecker,
)
from flliper.srt.mem_cache.unified_cache_components.tree_component import (
    ComponentType,
)
from flliper.srt.pdflip.l15_bind import rewrite_tree_chain
from flliper.srt.pdflip.l15_retain import reserve_mamba_slots

_HERE = os.path.dirname(__file__)


def _rewrite_fixture_module():
    spec = importlib.util.spec_from_file_location(
        "test_pdflip_l15_tree_rewrite_1001",
        os.path.join(_HERE, "test_pdflip_l15_tree_rewrite_1001.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _mamba_values_on_chain(node, root):
    out = []
    cur = node
    while cur is not None and cur is not root:
        mv = cur.component_data[ComponentType.MAMBA].value
        if mv is not None:
            out.extend(int(x) for x in mv.flatten().tolist())
        cur = cur.parent
    return out


def _held_round(fx, last):
    """Retain steps 4-6 for one held request whose anchor stays put."""
    anchor = last.component_data[ComponentType.MAMBA].value
    held = {int(x): int(x) for x in anchor.flatten().tolist()}
    rewrite_tree_chain(last, {}, held, set())
    fx.cache.reset_keep([last])
    mba = fx.pool.mamba_allocator
    mba.clear()
    reserve_mamba_slots(mba, sorted(held.values()))
    return mba, held


def test_held_round_leaves_no_free_and_cached_mamba_slot():
    m = _rewrite_fixture_module()
    fx = m._fixture()
    ids = list(range(300, 340))
    # a chunk boundary leaves an intermediate mamba checkpoint on the chain
    last = m._insert(fx, "R1", ids, chunk_end=16)
    before = _mamba_values_on_chain(last, fx.cache.root_node)
    assert len(before) >= 2, "the fixture must put >= 2 mamba values on the chain"
    mba, held = _held_round(fx, last)
    dup, _ids, shared = InvariantChecker._mamba_double_claimed(mba, fx.cache)
    assert dup == 0
    assert shared == 0, "tree names a free mamba slot (#924 aliasing)"
    # the held anchor itself survives and the prefix still matches
    after = _mamba_values_on_chain(last, fx.cache.root_node)
    assert set(after) == set(held.values())
    assert len(m._match(fx, ids).device_indices) > 0


def test_a_real_alias_is_still_detected():
    # the checker is not blinded: put a held anchor back on the free list by
    # hand -> free_and_cached == 1
    m = _rewrite_fixture_module()
    fx = m._fixture()
    last = m._insert(fx, "R2", list(range(400, 440)), chunk_end=16)
    mba, held = _held_round(fx, last)
    slot = next(iter(held.values()))
    mba.free_slots = torch.cat([mba.free_slots,
                                torch.tensor([slot], dtype=mba.free_slots.dtype,
                                             device=mba.free_slots.device)])
    _dup, _ids, shared = InvariantChecker._mamba_double_claimed(mba, fx.cache)
    assert shared == 1
