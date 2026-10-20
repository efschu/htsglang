# SPDX-License-Identifier: Apache-2.0
"""MAMBA-LAST-RESORT (NF int22, D TP0 boot 1008_171755 17:51:57): the D mamba
anchor arena (32 slots, 6 staging) held only END and deepest anchors, so no
spill found a victim.

DER BEFUND (D-Log boot_weg2_dkrnfint4h6ablxcw3spbar1dauer10081717_6b3bd1a6df_1008_171755):
* ``#1427 ARENA-DROP ... freed=0 stages=i:0,ii:0,iii:0 slot_bytes=58834944``
  x59 (all TP0, the mamba arena; the 786432-byte KV arena of D never ran full),
  e.g. D.log:125727..125742 at 17:51:57 right after ``F4 PARK-END
  rid=pdflip-24-128``.
* ``#1421 BACKUP-REFUSED why=mamba_claim node=408`` then every child
  ``parent_unbacked`` (D.log:125728..125744).
* ``PDFLIP MAMBA-ARENA FLUSH-SPILL node=416 ... victim=none`` and
  ``PDFLIP-ANCHOR-LOST at=flush n=15 depths=[107776 .. 112896 .. 198336]``: the
  END anchor 112896 of the parked pdflip-24-128 and the chain of another
  request were lost; D recomputes what the L3 could have served.
* The old rules never offer an END anchor (``_pdflip_anchor_of`` slots=None) and
  keep every request's deepest one (``pick_foreign_victim``), so an arena of
  END anchors has no victim.

THE FIX: with ``FLLIPER_PDFLIP_MAMBA_SPILL_LAST_RESORT`` on (the default since the
user decision of 09.10.), a claim the old rules refuse spills the shallowest
settled anchor (END too) off the claimer's park chain -- secured to L3 first
(``arena_secure_to_disk``), released only when that worked. ``0`` = the old
answer, byte for byte.

Hermetic: the real shared arena (arena.c, gcc), three rank pools, real tree
nodes, the real ``UnifiedRadixCache._pdflip_mamba_claim`` (the harness of the
H19 / PARK-END-ANCHOR-FIRST tests).
"""
from __future__ import annotations

import importlib.util
import os
import shutil
import types

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

import pytest  # noqa: E402

from flliper.srt.mem_cache.unified_cache_components.tree_component import ComponentType  # noqa: E402
from flliper.srt.pdflip import mamba_arena_displace as mad  # noqa: E402

M = ComponentType.MAMBA
needs_gcc = pytest.mark.skipif(shutil.which("gcc") is None, reason="needs gcc")
SWITCH = "FLLIPER_PDFLIP_MAMBA_SPILL_LAST_RESORT"


def _h19():
    spec = importlib.util.spec_from_file_location(
        "_h19_1008", os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                  "test_pdflip_mamba_arena_displace_h19.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


class _Layout:
    """A D tree of the log: ``n_end`` requests, each with ONE anchor that is
    its END anchor (untagged, settled), filling the arena; then the claimer
    ``NEW`` (the next request's first anchor) that finds no slot."""

    def __init__(self, tmp_path, slots, n_end, secured=True):
        self.H = _h19()
        H = self.H
        self.arena = H._arena(tmp_path, slots)
        self.ranks = [H._Rank(self.arena, r, cfg=0) for r in range(H.RANKS)]
        self.spilled = []
        spilled = self.spilled

        class _Sec(H._Backend):
            def arena_secure_to_disk(self, arena, cands, writer="claim_room"):
                spilled.append((writer, [c[0] for c in cands]))
                return {"on_disk": len(cands) if secured else 0, "written": 0,
                        "lost": 0 if secured else len(cands)}

        for rk in self.ranks:
            rk.mp._backend = _Sec()
        self.ends = [f"end{i}" for i in range(n_end)]
        for rk in self.ranks:
            for i, h in enumerate(self.ends):          # root siblings, shallow -> deep
                rk.add(None, h, tokens=512 * (i + 1), end_anchor=True, parent=rk.cache.root_node)
            assert rk.sweep()
        self.new = []
        for rk in self.ranks:
            self.new.append(rk.add(None, "NEW", tokens=512, parent=rk.cache.root_node))

    def run(self, rounds=4, flush=True):
        for rk in self.ranks:
            rk.cache._pdflip_flush_spill = flush
        for _ in range(rounds):
            if all([rk.sweep() for rk in self.ranks]):
                break

    def state(self, h):
        return self.H._state(self.arena, h)


@needs_gcc
def test_an_arena_of_end_anchors_spills_the_shallowest_one_secured_first(tmp_path, monkeypatch):
    """4 slots, all held by END anchors (the D tree of the log); the next
    request's anchor needs one. Before: FLUSH-SPILL victim=none, NEW stays
    un-backed, ``PDFLIP-ANCHOR-LOST``. Now: the shallowest END anchor is secured
    to L3 (every rank) and its slot goes to NEW; the deeper ones stay."""
    monkeypatch.setenv(SWITCH, "1")
    L = _Layout(tmp_path, 4, n_end=4)
    assert [L.state(h) for h in L.ends] == [2] * 4
    L.run()
    assert L.state("NEW") == 2, "the claimer landed"
    assert not any(rk.unbacked for rk in L.ranks)
    assert L.state(L.ends[0]) != 2, "the shallowest END anchor gave its slot"
    assert [L.state(h) for h in L.ends[1:]] == [2] * 3
    assert sum(1 for w, _ in L.spilled if w == "flush_spill") == L.H.RANKS, "secured to L3 first, on every rank"


@needs_gcc
def test_switch_off_is_the_old_answer_byte_for_byte(tmp_path, monkeypatch):
    """The log's failure, unchanged with the switch at 0: nothing is spilled,
    the claimer stays un-backed, every END anchor stays."""
    monkeypatch.setenv(SWITCH, "0")
    L = _Layout(tmp_path, 4, n_end=4)
    L.run()
    assert all(rk.unbacked for rk in L.ranks)
    assert [L.state(h) for h in L.ends] == [2] * 4 and L.spilled == []


@needs_gcc
def test_default_unset_is_on_the_shallowest_end_anchor_gives_its_slot(tmp_path, monkeypatch):
    """DEFAULT ON (user decision 09.10.): with the variable unset the claim
    behaves as with ``=1`` -- the shallowest END anchor is secured to L3 on
    every rank and its slot goes to the claimer (red on the base dacc89e543,
    whose default was off)."""
    monkeypatch.delenv(SWITCH, raising=False)
    L = _Layout(tmp_path, 4, n_end=4)
    L.run()
    assert L.state("NEW") == 2, "the claimer landed without the variable"
    assert not any(rk.unbacked for rk in L.ranks)
    assert L.state(L.ends[0]) != 2
    assert [L.state(h) for h in L.ends[1:]] == [2] * 3
    assert sum(1 for w, _ in L.spilled if w == "flush_spill") == L.H.RANKS


def test_env_default_is_on_and_0_is_off(monkeypatch):
    from flliper.srt.environ import envs

    monkeypatch.delenv(SWITCH, raising=False)
    assert envs.FLLIPER_PDFLIP_MAMBA_SPILL_LAST_RESORT.get() is True, "default on"
    monkeypatch.setenv(SWITCH, "0")
    assert envs.FLLIPER_PDFLIP_MAMBA_SPILL_LAST_RESORT.get() is False, "0 = off"
    monkeypatch.setenv(SWITCH, "1")
    assert envs.FLLIPER_PDFLIP_MAMBA_SPILL_LAST_RESORT.get() is True


@needs_gcc
def test_an_unsecured_end_anchor_is_never_released(tmp_path, monkeypatch):
    """No anchor is traded for another: its L3 copy could not be secured."""
    monkeypatch.setenv(SWITCH, "1")
    L = _Layout(tmp_path, 3, n_end=3, secured=False)
    L.run()
    assert all(rk.unbacked for rk in L.ranks)
    assert [L.state(h) for h in L.ends] == [2] * 3


@needs_gcc
def test_a_pinned_end_anchor_is_not_a_victim(tmp_path, monkeypatch):
    """A host-locked anchor (a load reads it) is skipped: the next shallowest
    gives its slot; with none left the claimer stays un-backed."""
    monkeypatch.setenv(SWITCH, "1")
    L = _Layout(tmp_path, 2, n_end=2)
    for rk in L.ranks:
        rk.cache.root_node.children[L.ends[0]].component_data[M].host_lock_ref = 1
    L.run()
    assert L.state("NEW") == 2
    assert L.state(L.ends[0]) == 2, "the locked shallowest one stays"
    assert L.state(L.ends[1]) != 2


# ---- the pure pick ---------------------------------------------------------------------

def _anchor(node_id, depth, slots=(1,)):
    return mad.OwnedAnchor(node=types.SimpleNamespace(id=node_id), depth=depth, rid="", slots=slots)


def test_pick_is_shallowest_then_node_id_never_pinned_never_on_the_chain():
    a, b, c = _anchor(5, 300), _anchor(6, 100), _anchor(2, 100)
    pinned = _anchor(1, 10, slots=None)
    on_chain = _anchor(9, 5)
    got = mad.pick_last_resort_victim([a, b, c, pinned, on_chain], chain_ids={id(on_chain.node)})
    assert got is c, "shallowest (100), the smaller node id first; the pinned and the chain node are out"
    assert mad.pick_last_resort_victim([pinned, on_chain], chain_ids={id(on_chain.node)}) is None
