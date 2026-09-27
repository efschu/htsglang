"""#1424c (rc12n D-TP0 12:57:43): the chain proof refused a park loadback
by name -- "the chain's rows are not whole distinct pages although every
node's pages match their keys (5 node(s))". Two parked rids of ONE
conversation (weg2-1-18, weg2-1-20), each read in two pieces (second piece
768 / 512 tokens), shared prefix to the Mamba anchor at 24320, H98 depth
scissors (extent TP0 24832, TP1/TP2 21824).

Why a page twice in one chain is never legal: a page key is content-chained
over the whole prefix, so one prefix never holds one key at two depths, and
the arena index maps a key to one slot (``find_slot`` checks the 128-bit
key). Rows twice in one chain therefore mean one stored key twice -- a node
carries the key of ANOTHER depth, and its rows (resolved by that key) are
that depth's KV. Loading them "because every node matches its keys" puts
page j-1's KV under page j's tokens. The proof now runs against the TOKENS:
P's hand-off key for the depth (the request's chain, from token 0) or D's own
key of the page chained from the proven page before it; a page carrying
neither is re-keyed to the admissible key whose slot is COMPLETE and loads
that slot's bytes.

Hermetic: the real ``verify_load_chain`` and ``ArenaMHAHostPool._load_arena``
on CPU tensors, the real ``UnifiedRadixCache._1424_verify_load_chain`` over a
parented tree."""
from __future__ import annotations

import os
import types

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

import pytest  # noqa: E402
import torch  # noqa: E402

from sglang.srt.mem_cache.pool_host import arena_pool as ap  # noqa: E402

P = 4
SLOTS = 8
CELL = 2
LAYERS = 2
S = 5  # staging rows before the arena ids

# P's hand-off chain of the conversation, indexed from token 0 (page j -> key)
CHAIN = ["a0", "a1", "a2", "a3", "a4", "a5"]
# the arena: every P page is COMPLETE in its own slot
SLOT_OF = {"a0": 3, "a1": 0, "a2": 6, "a3": 1, "a4": 5, "a5": 2}


class _Arena:
    def __init__(self, key_slot):
        self.key_slot = dict(key_slot)
        self.finds = 0
        self.refs = []

    def find_slots_np(self, stems):
        import numpy as np
        self.finds += 1
        slots = np.array([self.key_slot.get(s, -1) for s in stems], dtype=np.int64)
        st = np.array([2 if s in self.key_slot else 0 for s in stems], dtype=np.int8)
        return slots, st

    def ref_slots(self, slots, delta):
        self.refs.append((list(slots), delta))
        return len(slots)


def _pool(arena):
    pool = object.__new__(ap.ArenaMHAHostPool)
    pool._arena_page_tokens = P
    pool._page_view = None               # per-layer path: the loaded bytes are checked
    pool._page_key_hint = None
    pool._page_loaded_key = None
    pool.row_slot = None
    base = torch.arange(SLOTS * P * CELL, dtype=torch.float32).view(SLOTS, P, CELL)
    pool.arena_k_refs = [base + 1000 * layer for layer in range(LAYERS)]
    pool.arena_v_refs = [-(base + 1000 * layer) for layer in range(LAYERS)]
    pool.pin_slots = lambda slots: 0
    pool._arena_load_guard = lambda *a, **k: None
    pool.staging_rows = S
    pool.arena = arena
    return pool


def _rows(slots):
    return torch.cat([torch.arange(s * P, s * P + P) for s in slots]) + S


def _node(nid, keys, parent=None, tokens=None):
    n = types.SimpleNamespace(id=nid, hash_value=list(keys), parent=parent,
                              key=list(tokens if tokens is not None else range(len(keys) * P)))
    n.host_value = _rows([SLOT_OF.get(k, 7) for k in keys])
    return n


def _metal_chain():
    """Shared prefix a0..a2 (node 11 on the device, node 12 host), then the
    parked rid's own pages in two pieces: piece 1 = a3, piece 2 = a4,a5 --
    but piece 2 was inserted with the keys of ONE PAGE EARLIER (a3,a4), so its
    rows are a3's and a4's slots: the chain holds slot 1 twice."""
    n12 = _node(12, ["a1", "a2"])
    n13 = _node(13, ["a3"])
    n14 = _node(14, ["a3", "a4"])          # shifted: should be a4, a5
    return [n12, n13, n14]


def _load(pool, host):
    rows, order = (host - S).sort()
    dev = torch.arange(int(host.numel()))[order]
    dp = types.SimpleNamespace(k_buffer=[torch.zeros(int(host.numel()), CELL) for _ in range(LAYERS)],
                               v_buffer=[torch.zeros(int(host.numel()), CELL) for _ in range(LAYERS)])
    for layer in range(LAYERS):
        pool._load_arena(dp, rows, dev, layer)
    return dp


def _verify(pool, nodes, **kw):
    host = torch.cat([n.host_value for n in nodes])
    return ap.verify_load_chain(pool, nodes, host, rows_of=lambda n: n.host_value,
                                stems_of=lambda h: list(h), **kw)


def test_the_duplicate_is_a_shifted_key_and_loading_it_would_put_the_wrong_kv_there():
    """The premise, checked: the refused chain's twin page is a3's slot at
    depths 3 and 4 -- the gather would load a3's KV under page 4's tokens."""
    nodes = _metal_chain()
    host = torch.cat([n.host_value for n in nodes]) - S
    first = host.view(-1, P)[:, 0] // P
    assert first.tolist() == [0, 6, 1, 1, 5], "slot 1 (a3) at depth 3 AND 4"
    with pytest.raises(ap.ArenaChainMismatch):
        _verify(_pool(_Arena(SLOT_OF)), nodes)          # the stored keys alone prove nothing


def test_metal_shape_the_shifted_piece_is_rekeyed_to_its_depth_and_loads_its_own_bytes():
    """RED on 7cac9f5372 (the chain dies by name, as at 12:57:43). With P's
    chain for the depths the second piece is re-keyed a3,a4 -> a4,a5,
    re-pointed to their slots, the node's keys corrected, and the load returns
    each page's own slot."""
    arena = _Arena(SLOT_OF)
    pool = _pool(arena)
    nodes = _metal_chain()
    host = _verify(pool, nodes, p_chain=CHAIN, page0=1, prior0="a0")
    want = [SLOT_OF[k] for k in CHAIN[1:6]]
    assert torch.equal(host - S, _rows(want) - S)
    assert nodes[2].hash_value == ["a4", "a5"], "the node carries its depth's keys now"
    assert torch.equal(nodes[2].host_value - S, _rows([SLOT_OF["a4"], SLOT_OF["a5"]]) - S)
    assert sorted(s for sl, d in arena.refs for s in sl) == sorted([SLOT_OF["a4"], SLOT_OF["a5"]])
    dp = _load(pool, host)
    for layer in range(LAYERS):
        src = pool.arena_k_refs[layer]
        for page, slot in enumerate(want):
            assert torch.equal(dp.k_buffer[layer][page * P:(page + 1) * P], src[slot]), (layer, page)


def test_a_shifted_hand_off_key_without_a_twin_is_caught_too():
    """No duplicate -- only the last page of a piece carries the key of
    another depth. The chain is whole distinct pages and the old check let it
    through silently; P's chain names the depth and the page is re-keyed."""
    arena = _Arena(SLOT_OF)
    pool = _pool(arena)
    nodes = [_node(12, ["a1", "a2"]), _node(13, ["a3", "a5"])]    # page 4 carries a5
    host = _verify(pool, nodes, p_chain=CHAIN, page0=1, prior0="a0")
    assert nodes[1].hash_value == ["a3", "a4"]
    assert torch.equal(host - S, _rows([SLOT_OF[k] for k in ("a1", "a2", "a3", "a4")]) - S)


def test_pages_beyond_ps_chain_are_proven_by_ds_own_key_chained_from_the_proven_page():
    """D's decoded pages (the park's retained span) are keyed by D's own
    hash, chained from the page before. A stored key that is neither is
    re-keyed to the own key whose slot is COMPLETE."""
    own = {("a3", 0): "d4", ("d4", 1): "d5"}
    arena = _Arena({**SLOT_OF, "d4": 4, "d5": 7})
    pool = _pool(arena)
    n13 = _node(13, ["a3"])
    n14 = types.SimpleNamespace(id=14, hash_value=["a3", "d4"], parent=None, key=list(range(2 * P)),
                                host_value=_rows([1, 4]))          # shifted onto a3
    host = _verify(pool, [n13, n14], p_chain=CHAIN[:4], page0=3, prior0="a2",
                   own_hash=lambda node, i, prior: own.get((prior, i), f"own({prior},{i})"))
    assert n14.hash_value == ["d4", "d5"]
    assert torch.equal(host - S, _rows([1, 4, 7]) - S)


def test_an_admissible_stored_key_keeps_its_slot_and_a_clean_chain_costs_no_lookup():
    arena = _Arena(SLOT_OF)
    pool = _pool(arena)
    nodes = [_node(12, ["a1", "a2"]), _node(13, ["a3"]), _node(14, ["a4", "a5"])]
    host = torch.cat([n.host_value for n in nodes])
    out = ap.verify_load_chain(pool, nodes, host, rows_of=lambda n: n.host_value,
                               stems_of=lambda h: list(h), p_chain=CHAIN, page0=1, prior0="a0")
    assert out is host and arena.finds == 0


def test_a_page_no_admissible_key_can_prove_is_a_named_stop_with_the_depth():
    slot_of = {k: s for k, s in SLOT_OF.items() if k != "a5"}      # a5 not complete
    pool = _pool(_Arena(slot_of))
    with pytest.raises(ap.ArenaChainMismatch, match=r"node=14 page=1 depth_page=5"):
        _verify(pool, _metal_chain(), p_chain=CHAIN, page0=1, prior0="a0")


def test_the_real_wrapper_reads_the_depth_and_ps_chain_from_the_tree_and_the_request():
    """``UnifiedRadixCache._1424_verify_load_chain`` on a parented tree: the
    depth of the first loaded page comes from the ancestors, P's chain from
    the request (``handoff_keys.CHAIN_ATTR``), own keys from get_hash_str."""
    from sglang.srt.mem_cache import unified_radix_cache as urc
    from sglang.srt.weg2.handoff_keys import CHAIN_ATTR

    arena = _Arena({k + "#": s for k, s in SLOT_OF.items()})
    pool = _pool(arena)
    backend = types.SimpleNamespace(_suffix_for_key=lambda k: ("#", False))
    group = types.SimpleNamespace(arena_read=True, anchor_entry=types.SimpleNamespace(host_pool=pool))
    tree = object.__new__(urc.UnifiedRadixCache)
    tree.cache_controller = types.SimpleNamespace(mem_pool_host=group, storage_backend=backend)
    tree.page_size = P
    root = types.SimpleNamespace(key=[], hash_value=[], parent=None)
    tree.root_node = root
    n11 = types.SimpleNamespace(key=list(range(P)), hash_value=["a0"], parent=root,
                                get_last_hash_value=lambda: "a0")
    nodes = _metal_chain()
    nodes[0].parent, nodes[1].parent, nodes[2].parent = n11, nodes[0], nodes[1]
    for n in nodes:
        n.component_data = {urc.BASE_COMPONENT_TYPE: types.SimpleNamespace(host_value=n.host_value)}
    req = types.SimpleNamespace(rid="weg2-1-18")
    setattr(req, CHAIN_ATTR, list(CHAIN))
    xfer = types.SimpleNamespace(nodes_to_load=nodes,
                                 host_indices=torch.cat([n.host_value for n in nodes]))
    tree._1424_verify_load_chain(xfer, req=req)
    assert torch.equal(xfer.host_indices - S, _rows([SLOT_OF[k] for k in CHAIN[1:6]]) - S)
    assert nodes[2].hash_value == ["a4", "a5"]
