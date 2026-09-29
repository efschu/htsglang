"""rc12r D-TP0 17:17:01/04 (gap=-2251): the ARENA-REF-HOLDERS census thread
walked the tree it found BEFORE a reset (tree=5013, the old tree's count) and
read the ledger AFTER the reset and the park refetch (own_held=2762). The
next census (17:18:04) was exact: pinned=refs=own_held=tree_in_use=2541,
gap=0. A snapshot a reset moved under is now named (``snapshot=torn``, gap
``-``) instead of being printed as a leak; a ledger that only moved during
the walk is printed as ``own_drift``.

Hermetic: the #1424g tree shell (real ``pdflip_arena_holder_census``, real
``_reset_full``) on a real C arena."""
from __future__ import annotations

import importlib.util
import os
import shutil

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

import pytest  # noqa: E402

_spec = importlib.util.spec_from_file_location(
    "_t1424g", os.path.join(os.path.dirname(__file__), "test_arena_reset_orphans_1424g.py"))
g = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(g)

arena = g.arena
pytestmark = pytest.mark.skipif(shutil.which("gcc") is None, reason="needs gcc (arena.c)")


def test_a_reset_during_the_walk_is_named_torn_not_a_gap(arena, monkeypatch):
    """RED on e39b37d011: the walk counts the old tree, the ledger is read
    after the reset -- the shipped line says gap=-5 (a leak that is not)."""
    t, pool, _ = g._parked_two_rids_same_prefix(arena)
    real = g.ap.arena_ref_pages
    fired = []

    def _pages_then_reset(p, hv):
        n = real(p, hv)
        if not fired:                       # the scheduler resets mid-walk
            fired.append(1)
            t._reset_full()
        return n

    from flliper.srt.mem_cache.pool_host import arena_pool as ap_mod
    monkeypatch.setattr(ap_mod, "arena_ref_pages", _pages_then_reset)
    line = t.pdflip_arena_holder_census(arena)
    assert "snapshot=torn" in line and "gap=-" in line and "gap=-5" not in line, line
    monkeypatch.setattr(ap_mod, "arena_ref_pages", real)
    assert "own_held=0 gap=0" in t.pdflip_arena_holder_census(arena)


def test_a_node_mid_write_through_names_no_page_it_holds_no_reference_for(arena):
    """RED on e39b37d011. rc12r P: FULL gap -50/-57/-64 and MAMBA -1 after
    resets, never confirmed by the next census. A node mid write-through
    (tree_in_use) names the slots of its pending claim -- no reader reference
    exists on a claimed slot until complete_write -- and the census counted
    them: gap = -(the publish). The release rule skips pending slots; the
    census now does too."""
    t, pool, slot_of = g._parked_two_rids_same_prefix(arena)
    (s, st, gen), = arena.claim_slots(["w0"], [g.SLOT_BYTES])
    assert st == 0
    pool._pending = {s: (gen, True)}                  # the claim is pending, no ref
    node = g._node(g._tree(arena, pool)[1], 9, g._rows([s]))
    node.write_through_pending_id = 7
    t.root_node.children[9] = node
    line = t.pdflip_arena_holder_census(arena)
    assert "tree_in_use=0 " in line and "gap=0" in line, line


def test_a_quiet_census_is_unchanged(arena):
    t, _, _ = g._parked_two_rids_same_prefix(arena)
    line = t.pdflip_arena_holder_census(arena)
    assert "gap=0" in line and "snapshot=torn" not in line and "own_drift" not in line, line
