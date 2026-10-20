# SPDX-License-Identifier: Apache-2.0
"""PARK-END-ANCHOR-FIRST (auftrag 1303, NF y9nf4 e23f7dff30 boot 1004_031945,
D TP0 03:33:18): the sleep found the Mamba anchor arena full and lost the park
END anchors of the parked requests.

DER BEFUND (D-Log, Zeilen = grep -n):
* ``#1427 ARENA-CLAIM REFUSED statuses=[4]`` x8 (D:71244..71290), ``#1421
  BACKUP-REFUSED why=mamba_claim`` on node=192 depth=47296 and node=195
  depth=72704 (rid=None: a D tree's nodes carry no writer), every child
  ``parent_unbacked`` up to depth 49408 (node 219) / 73728 (node 209).
* ``PDFLIP MAMBA-ARENA FLUSH-SPILL ... victim=none`` (D:71392): the H19 flush
  spill offers TAGGED intermediate anchors only (``tree_anchors`` skips
  ``pdflip_anchor_rid is None``); a D tree's anchors are all untagged.
* ``PDFLIP-ANCHOR-LOST at=flush n=20 depths=[...49408...76032...]`` (D:71461):
  the park anchors of pdflip-8-79 / pdflip-8-82 (``#59b PARK-RESUMABLE``, D:71352).

THE FIX: ``park_running`` marks the request it retracts; the finish insert
marks the node it ended at (``pdflip_park_end``). A claim of a node on that
chain (ancestor-or-self) that the H19 rules refuse may spill a releasable
anchor OFF the chain -- tagged or not, secured to L3 first, shallowest first;
the END node, with none off the chain, a chain intermediate. A chain
intermediate that still finds no slot goes down KV-only and is not retried.
Switch ``FLLIPER_PDFLIP_PARK_END_ANCHOR_FIRST`` (default on). No mark = no
change: the H19 answer stands byte for byte.

MAMBA-LAST-RESORT (``FLLIPER_PDFLIP_MAMBA_SPILL_LAST_RESORT``, default on since
the user decision of 09.10.) is the step AFTER these rules: the tests that pin
the answer of the H19 / PARK-END-ANCHOR-FIRST rules alone set it to 0
(``_no_last_resort``); its own proof is test_pdflip_mamba_last_resort_1008.py.

Hermetic: the real shared arena (arena.c, gcc), three rank pools, real tree
nodes, the real ``UnifiedRadixCache._pdflip_mamba_claim`` (the harness of the
H19 / D-NORECOMPUTE tests).
"""
from __future__ import annotations

import importlib.util
import inspect
import logging
import os
import shutil
import types

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

import pytest  # noqa: E402

from flliper.srt.mem_cache.unified_cache_components.tree_component import ComponentType  # noqa: E402
from flliper.srt.mem_cache.unified_radix_cache import UnifiedRadixCache, UnifiedTreeNode  # noqa: E402
from flliper.srt.pdflip import mamba_arena_displace as mad  # noqa: E402

M = ComponentType.MAMBA
needs_gcc = pytest.mark.skipif(shutil.which("gcc") is None, reason="needs gcc")


def _no_last_resort(monkeypatch):
    """The rules of this file alone: MAMBA-LAST-RESORT (default on) off."""
    monkeypatch.setenv("FLLIPER_PDFLIP_MAMBA_SPILL_LAST_RESORT", "0")


def _h19():
    spec = importlib.util.spec_from_file_location(
        "_h19_1303", os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                  "test_pdflip_mamba_arena_displace_h19.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


class _Layout:
    """A D tree of the log: older anchors off the park (untagged, settled, on
    L3 already) and the parked request's chain A -> B -> END (all unbacked)."""

    def __init__(self, tmp_path, slots, off_chain, chain=("A", "B", "END"), secured=True,
                 filler_first=True):
        self.H = _h19()
        H = self.H
        self.arena = H._arena(tmp_path, slots)
        self.ranks = [H._Rank(self.arena, r, cfg=0) for r in range(H.RANKS)]
        self.secured = secured
        self.spilled = []
        spilled = self.spilled

        class _Sec(H._Backend):
            def arena_secure_to_disk(self, arena, cands, writer="claim_room"):
                spilled.append((writer, [c[0] for c in cands]))
                return {"on_disk": len(cands) if secured else 0, "written": 0,
                        "lost": 0 if secured else len(cands)}

        for rk in self.ranks:
            rk.mp._backend = _Sec()
        self.off = [f"old{i}" for i in range(off_chain)]
        for rk in self.ranks:
            for i, h in enumerate(self.off):          # siblings of the root, shallow -> deep
                rk.add(None, h, tokens=512 * (i + 1), parent=rk.cache.root_node)
            assert rk.sweep()
        self.chain = list(chain)
        self.nodes = {}
        for rk in self.ranks:
            parent = rk.cache.root_node
            for h in self.chain:
                n = rk.add(None, h, tokens=512, parent=parent)
                parent = n
                self.nodes.setdefault(h, []).append(n)

    def mark(self, rank_marks=True):
        for rk, end in zip(self.ranks, self.nodes[self.chain[-1]]):
            end.pdflip_park_end = True
            rk.cache._pdflip_park_end_nodes = [end]

    def run(self, rounds=8, flush=True):
        for rk in self.ranks:
            rk.cache._pdflip_flush_spill = flush
        for _ in range(rounds):
            if all([rk.sweep() for rk in self.ranks]):
                break

    def state(self, h):
        return self.H._state(self.arena, h)


# ---- the model of the log: arena full of anchors without park reference ----------------

@needs_gcc
def test_park_chain_takes_slots_from_untagged_anchors_off_the_chain(tmp_path, monkeypatch):
    """4 slots, all held by older untagged anchors (the D tree of the log); the
    parked chain A -> B -> END needs 3. Before: FLUSH-SPILL victim=none, the
    sweep stuck on A, END un-backed. Now: the three shallowest off-chain
    anchors are spilled (secured first), the chain lands, the deepest older
    anchor stays."""
    _no_last_resort(monkeypatch)   # every spill is park_first's, none the flush's last resort
    L = _Layout(tmp_path, 4, off_chain=4)
    assert [L.state(h) for h in L.off] == [2] * 4
    L.mark()
    L.run()
    assert [L.state(h) for h in L.chain] == [2, 2, 2], "the END anchor and its chain landed"
    assert not any(rk.unbacked for rk in L.ranks)
    assert all(L.state(h) != 2 for h in L.off[:3]), "the three shallowest older anchors were spilled"
    assert L.state(L.off[3]) == 2, "the deepest older anchor was not needed"
    assert sum(1 for w, _ in L.spilled if w == "park_first_spill") == 3 * L.H.RANKS


@needs_gcc
def test_without_the_mark_the_h19_answer_stands(tmp_path, monkeypatch):
    """Regression proof / byte-identity: the same layout, no park mark. The
    flush spill finds no TAGGED victim (a D tree has none), the chain stays
    un-backed, nothing is spilled -- exactly the log's failure (MAMBA-LAST-RESORT
    at 0; with it on, the flush's last resort takes over)."""
    _no_last_resort(monkeypatch)
    L = _Layout(tmp_path, 4, off_chain=4)
    L.run()
    assert all(L.state(h) != 2 for h in L.chain)
    assert all(rk.unbacked for rk in L.ranks)
    assert [L.state(h) for h in L.off] == [2] * 4
    assert L.spilled == []


@needs_gcc
def test_switch_off_is_the_old_behaviour(tmp_path, monkeypatch):
    monkeypatch.setenv("FLLIPER_PDFLIP_PARK_END_ANCHOR_FIRST", "0")
    _no_last_resort(monkeypatch)
    L = _Layout(tmp_path, 4, off_chain=4)
    L.mark()
    L.run()
    assert all(rk.unbacked for rk in L.ranks)
    assert [L.state(h) for h in L.off] == [2] * 4 and L.spilled == []


@needs_gcc
def test_end_node_takes_a_chain_intermediate_when_nothing_is_off_the_chain(tmp_path):
    """3 slots, the chain A -> B -> C -> END alone: A, B, C take the three
    slots (parents first), the END finds none and nothing is off the chain ->
    the END takes the shallowest chain INTERMEDIATE (A), secured to L3 first."""
    L = _Layout(tmp_path, 3, off_chain=0, chain=("A", "B", "C", "END"))
    L.mark()
    L.run()
    assert L.state("END") == 2
    assert L.state("A") != 2, "the shallowest chain intermediate gave its slot"
    assert [L.state(h) for h in ("B", "C")] == [2, 2]
    assert sum(1 for w, _ in L.spilled if w == "park_first_spill") == L.H.RANKS


@needs_gcc
def test_an_unsecured_victim_is_never_released(tmp_path):
    L = _Layout(tmp_path, 2, off_chain=2, chain=("END",), secured=False)
    L.mark()
    L.run()
    assert all(rk.unbacked for rk in L.ranks)
    assert [L.state(h) for h in L.off] == [2, 2], "no anchor is traded for another"


@needs_gcc
def test_a_non_end_chain_node_never_takes_a_chain_anchor(tmp_path):
    """The intermediate A has no off-chain victim and may not cannibalise its
    own chain: it stays un-backed (the KV-only fallback in the direct claim
    takes it from there)."""
    L = _Layout(tmp_path, 1, off_chain=0, chain=("X", "A", "END"))
    L.mark()
    # X (a chain node) holds the single slot, A is un-backed
    for rk in L.ranks:
        rk.unbacked = [(L.nodes["X"][L.ranks.index(rk)], "X")]
    L.run()
    assert L.state("X") == 2
    for rk in L.ranks:
        rk.unbacked = [(L.nodes["A"][L.ranks.index(rk)], "A")]
    L.run()
    assert L.state("A") != 2 and L.state("X") == 2
    assert all(rk.unbacked for rk in L.ranks)


@needs_gcc
def test_spill_tries_are_bounded(tmp_path):
    """The single slot is held by an anchor nobody may release (host lock): the
    END's claim says victim=none -- and asks at most PDFLIP_PARK_SPILL_TRIES times."""
    L = _Layout(tmp_path, 1, off_chain=1, chain=("END",))
    L.mark()
    rk = L.ranks[0]
    held = rk.cache.root_node.children["old0"]
    held.component_data[M].host_lock_ref = 1
    end = L.nodes["END"][0]
    for _ in range(10):
        assert rk.cache._pdflip_mamba_claim(end, rk.mp, "END") is None
    assert end.__dict__["_pdflip_park_spill_tries"] == UnifiedRadixCache.PDFLIP_PARK_SPILL_TRIES
    assert L.spilled == []


# ---- the pure pick ---------------------------------------------------------------------

def _anchor(node_id, depth, slots=(1,), end=False, rid=""):
    n = types.SimpleNamespace(id=node_id, pdflip_park_end=end)
    return mad.OwnedAnchor(node=n, depth=depth, rid=rid, slots=slots)


def test_pick_prefers_off_chain_shallowest_then_node_id():
    a, b, c = _anchor(5, 300), _anchor(6, 100), _anchor(2, 100)
    chain = {id(a.node)}
    got = mad.pick_park_victim([a, b, c], chain_ids=chain, claimer_is_end=False)
    assert got is c, "off the chain (b, c): shallowest, then the smaller node id"


def test_pick_never_offers_a_pinned_anchor_or_a_chain_node_to_a_non_end_claimer():
    pinned = _anchor(1, 10, slots=None)
    on_chain = _anchor(2, 20)
    assert mad.pick_park_victim([pinned, on_chain], chain_ids={id(on_chain.node)}, claimer_is_end=False) is None


def test_pick_end_claimer_falls_back_to_chain_intermediates_never_another_end():
    mid, end2 = _anchor(2, 20), _anchor(3, 5, end=True)
    chain = {id(mid.node), id(end2.node)}
    assert mad.pick_park_victim([mid, end2], chain_ids=chain, claimer_is_end=True) is mid
    assert mad.pick_park_victim([end2], chain_ids=chain, claimer_is_end=True) is None


def test_chain_ids_are_the_root_paths_of_the_marked_ends():
    root = types.SimpleNamespace(parent=None)
    a = types.SimpleNamespace(parent=root)
    b = types.SimpleNamespace(parent=a)
    other = types.SimpleNamespace(parent=root)
    gone = types.SimpleNamespace(parent=None)
    assert mad.park_chain_ids([b, gone]) == {id(a), id(b)}
    assert id(other) not in mad.park_chain_ids([b])


# ---- the wiring ------------------------------------------------------------------------

def test_park_running_marks_every_retracted_request():
    from flliper.srt.pdflip import d_park_runtime as DPR

    src = inspect.getsource(DPR.park_running)
    i = src.index("setattr(req, FORCE_HOST_WRITE_THROUGH_ATTR, True)")
    assert "setattr(req, _mad_park.PARK_REQ_ATTR, True)" in src[i:i + 900]
    assert src.index("setattr(req, _mad_park.PARK_REQ_ATTR, True)") < src.index("retract_all")


def test_finish_insert_marks_before_the_retain_publish():
    src = inspect.getsource(UnifiedRadixCache.cache_finished_req)
    assert src.index("self._pdflip_mark_park_end(req, radix_key)") < src.index(
        "self._pdflip_publish_at_retain(req, radix_key)")


def _bare_tree():
    t = object.__new__(UnifiedRadixCache)
    t.root_node = UnifiedTreeNode((ComponentType.FULL, M))
    return t


def test_mark_takes_the_last_chain_node_once_and_consumes_the_request_mark(caplog):
    t = _bare_tree()
    a = UnifiedTreeNode((ComponentType.FULL, M))
    a.key, a.parent = [0] * 512, t.root_node
    b = UnifiedTreeNode((ComponentType.FULL, M))
    b.key, b.parent = [0] * 256, a
    t._pdflip_chain_nodes_strict = lambda key: [a, b]
    req = types.SimpleNamespace(rid="pdflip-8-79")
    setattr(req, mad.PARK_REQ_ATTR, True)
    with caplog.at_level(logging.INFO):
        t._pdflip_mark_park_end(req, object())
    assert b.pdflip_park_end and not a.pdflip_park_end
    assert t._pdflip_park_end_nodes == [b]
    assert getattr(req, mad.PARK_REQ_ATTR) is False, "one use: a resumed finish is no park"
    assert "PARK-END-ANCHOR-FIRST MARK rid=pdflip-8-79 node=" in caplog.text and "depth=768" in caplog.text
    t._pdflip_mark_park_end(req, object())          # the request mark is gone: nothing more happens
    assert t._pdflip_park_end_nodes == [b]
    # a request without the mark (a normal finish) marks nothing
    other = types.SimpleNamespace(rid="x")
    t._pdflip_mark_park_end(other, object())
    assert t._pdflip_park_end_nodes == [b]


def test_no_mark_means_no_park_first_anywhere():
    t = _bare_tree()
    n = UnifiedTreeNode((ComponentType.FULL, M))
    n.parent = t.root_node
    assert t._pdflip_park_first_node(n) is False
    assert t._pdflip_park_chain() == set()


def test_a_node_that_left_the_tree_drops_its_mark():
    t = _bare_tree()
    n = UnifiedTreeNode((ComponentType.FULL, M))
    n.parent = None
    t._pdflip_park_end_nodes = [n]
    assert t._pdflip_park_chain() == set()
    assert t._pdflip_park_end_nodes == []


def test_the_anchor_it_gave_up_is_not_retried_by_the_anchor_only_sweep():
    from flliper.srt.environ import envs  # noqa: F401

    t = _bare_tree()
    t.tree_components = {M: object()}
    n = UnifiedTreeNode((ComponentType.FULL, M))
    n.parent = t.root_node
    n.hash_value = ["h"]
    n.component_data[ComponentType.FULL].value = object()
    n.component_data[M].value = object()
    n.component_data[M].host_value = None
    n.l3_present = True
    assert t._pdflip_anchor_only_candidate(n) is True
    n.pdflip_park_anchor_dropped = True
    assert t._pdflip_anchor_only_candidate(n) is False


def test_the_direct_claim_drops_a_refused_chain_intermediate_kv_only():
    src = inspect.getsource(UnifiedRadixCache._pdflip_direct_claim)
    i = src.index("PARK-END-ANCHOR-FIRST: an INTERMEDIATE")
    seg = src[i - 300:i + 700]
    assert "not kv_only_if_mamba_refused" in seg and "_pdflip_park_first_node(node)" in seg
    assert "comp_xfers.pop(mct, None)" in seg and "node.pdflip_park_anchor_dropped = True" in seg


def test_reset_drops_the_marks():
    assert "self._pdflip_park_end_nodes = []" in inspect.getsource(UnifiedRadixCache._reset_full)
