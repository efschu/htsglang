"""#239 S4b (F14) part 6: #1424g on the worker trees -- a PIN, no fix.

Under the token cut a Form A worker that owns token rows carries the arena
(S4b part 3: ``get_mha_host_pool_cls`` gives it ``ArenaMHAHostPool`` with owner
rows; its anchor is that pool). The reset's orphan pass is gated on
``mem_pool_host.arena_read`` (``UnifiedRadixCache._reset_full``) and armed by
``init_hicache`` on every rank; the reference ledger is per process and per
arena file. Nothing in the pass reads the rank role or the owner rows -- so the
worker's tree gives back what no holder names exactly like TP0's does. Pinned
here so a later role gate cannot switch it off for the worker unnoticed:

* the group's ``arena_read`` is the ANCHOR pool's (the worker's arena pool);
* an owner-row pool's references (slot per page, whatever token rows the rank
  owns in it) are named by the tree and given back by the reset like any other.

Hermetic: the real C arena and ledger, the real ``_reset_full`` on the #1424g
tree shell. GREEN on ebeccfd190 (a pin, not a red->green)."""
from __future__ import annotations

import importlib.util
import os
import types

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

import pytest  # noqa: E402
import torch  # noqa: E402

from sglang.srt.mem_cache.memory_pool_host import HostPoolGroup  # noqa: E402

_G = os.path.join(os.path.dirname(os.path.abspath(__file__)), "test_arena_reset_orphans_1424g.py")
_spec = importlib.util.spec_from_file_location("_g1424_harness", _G)
g = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(g)

arena = g.arena  # the fixture: a real ShmArena with the per-process ledger


def _owner_pool(a):
    """A KV worker's arena pool under the cut: whole-page slots, this rank's
    token rows of each page (S4b part 3 ``bind(owner_rows=...)``)."""
    pool = g._pool(a)
    pool._owner_tok = torch.tensor([2, 3], dtype=torch.int64)
    return pool


def test_group_arena_read_is_the_anchor_pools(arena):
    group = object.__new__(HostPoolGroup)
    group.anchor_entry = types.SimpleNamespace(host_pool=_owner_pool(arena))
    assert group.arena_read is True, "the worker's reset must see its arena (anchor pool)"
    group.anchor_entry = types.SimpleNamespace(host_pool=types.SimpleNamespace())
    assert group.arena_read is False, "a byteless anchor keeps the pass off"


def test_kv_worker_reset_gives_back_what_no_holder_names(arena):
    slot_of = g._publish(arena, g.PREFIX)
    pool = _owner_pool(arena)
    t, FULL = g._tree(arena, pool)
    t.root_node = types.SimpleNamespace(
        children={1: g._node(FULL, 1, g._resolve(arena, slot_of, g.PREFIX))})
    assert arena.ref_slots([slot_of["a2"]], +1) == 1     # a reference no class names
    assert "gap=1" in t.weg2_arena_holder_census(arena)
    t._reset_full()
    assert g._refs(arena) == [0] * g.SLOTS
    assert g._own(arena) == 0
    assert arena._ledger.refused == 0


if __name__ == "__main__":
    pytest.main([__file__, "-q"])
