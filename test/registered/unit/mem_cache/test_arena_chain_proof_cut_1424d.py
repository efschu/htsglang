"""#1424d (rc12n2 D-TP0 13:36:10): "#1424 CHAIN MISMATCH node=234 page=0
depth_page=484: the stored key 37d19e9e2e7a409b is not this page's key (P's -,
own 4d1f1b4bf648aed6) and no admissible key has a COMPLETE slot" -- D dead in
load_back, TP1/TP2 (Form A workers, 0-byte KV) had already taken #988 LOADBACK
33024. pdflip-2-15: first piece to 30976 (INCOMPLETE, shortfall 2048), second
piece 2048 tokens (success), P's hand-off chain ends at page 483.

ROOT (the proof's own blind spot): the tree keys an EAGLE (bigram) RadixKey,
the store READS -- and the prefetch-inserted node stores -- the page keys of
``_storage_hit_query``: ``get_hash_str`` over the PLAIN token ids chained from
the node's last hash. The #1424c proof computed D's own key only in the
tree's bigram convention, so a second-piece page beyond P's chain could never
be proven and D died on a page it could have loaded.

Two fixes, both tested here on the real code:
1. D's own key is admissible in BOTH conventions (read + tree).
2. A page nothing proves never kills D: the Form A host's admission vote is
   cut at the last proven page (group-uniform through the usable-match MIN),
   and the X gate prices the group's floor -- rest <= X: D re-prefills it;
   rest > X: refused by name (W50 -> via P). The load's stop stays the last
   latch."""
from __future__ import annotations

import os
import types
from array import array

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

import pytest  # noqa: E402
import torch  # noqa: E402

from flliper.srt.mem_cache.pool_host import arena_pool as ap  # noqa: E402
from flliper.srt.mem_cache.radix_cache import RadixKey  # noqa: E402
from flliper.srt.mem_cache.utils import get_hash_str  # noqa: E402

P = 4
S = 5
SLOTS = 16
CELL = 2
LAYERS = 2
RAW = list(range(1000, 1000 + 6 * P + 1))  # 6 pages + the bigram boundary token


def _plain(page, prior):
    return get_hash_str(RAW[page * P:(page + 1) * P], prior, page_size=P)[0]


def _bigram(page, prior):
    k = RadixKey(array("q", RAW[page * P:(page + 1) * P + 1]), None, is_bigram=True)
    return get_hash_str(k, prior, page_size=P)[0]


# P's hand-off chain covers pages 0..2 (read convention, from token 0)
PK = []
for _j in range(3):
    PK.append(_plain(_j, PK[-1] if PK else None))
# the second piece (pages 3, 4) was READ with D's own plain keys chained from P's page 2
K3 = _plain(3, PK[2])
K4 = _plain(4, K3)
SLOT_OF = {PK[0]: 9, PK[1]: 1, PK[2]: 6, K3: 3, K4: 12}


class _Arena:
    def __init__(self, key_slot):
        self.key_slot = dict(key_slot)
        self.refs = []

    def find_slots_np(self, stems):
        import numpy as np
        slots = np.array([self.key_slot.get(s, -1) for s in stems], dtype=np.int64)
        st = np.array([2 if s in self.key_slot else 0 for s in stems], dtype=np.int8)
        return slots, st

    def ref_slots(self, slots, delta):
        self.refs.append((list(slots), delta))
        return len(slots)


def _pool(arena):
    pool = object.__new__(ap.ArenaMHAHostPool)
    pool._arena_page_tokens = P
    pool._page_view = None
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


def _tree(arena, *, own_keys):
    """A parented tree: root -> n11 (page 0, on device) -> n12 (pages 1-2,
    host; page 1's rows point at page 2's slot -- a twin, so the chain is
    proven page by page) -> n13 (pages 3-4, the second piece, host, read-convention keys)."""
    from flliper.srt.mem_cache import unified_radix_cache as urc

    pool = _pool(arena)
    backend = types.SimpleNamespace(_suffix_for_key=lambda k: ("", False))
    group = types.SimpleNamespace(arena_read=True, anchor_entry=types.SimpleNamespace(host_pool=pool))
    tree = object.__new__(urc.UnifiedRadixCache)
    tree.cache_controller = types.SimpleNamespace(mem_pool_host=group, storage_backend=backend)
    tree.page_size = P
    root = types.SimpleNamespace(key=RadixKey(array("q"), None, is_bigram=True), hash_value=[],
                                 parent=None, evicted=False)
    tree.root_node = root

    def node(nid, p0, p1, keys, rows, parent, evicted):
        n = types.SimpleNamespace(
            id=nid, parent=parent, evicted=evicted, hash_value=list(keys),
            key=RadixKey(array("q", RAW[p0 * P:p1 * P + 1]), None, is_bigram=True))
        n.get_last_hash_value = (lambda n=n: n.hash_value[-1] if n.hash_value else None)
        n.component_data = {urc.BASE_COMPONENT_TYPE: types.SimpleNamespace(host_value=rows.clone())}
        return n

    n11 = node(11, 0, 1, [PK[0]], _rows([9]), root, False)
    n12 = node(12, 1, 3, [PK[1], PK[2]], _rows([6, 6]), n11, True)      # page 1 on page 2's slot: a twin
    n13 = node(13, 3, 5, own_keys, _rows([SLOT_OF.get(k, 13) for k in own_keys]), n12, True)
    req = types.SimpleNamespace(rid="pdflip-2-15")
    from flliper.srt.pdflip.handoff_keys import CHAIN_ATTR
    setattr(req, CHAIN_ATTR, list(PK))
    return tree, pool, (n12, n13), req


def _xfer(nodes):
    from flliper.srt.mem_cache import unified_radix_cache as urc
    return types.SimpleNamespace(
        nodes_to_load=list(nodes),
        host_indices=torch.cat([n.component_data[urc.BASE_COMPONENT_TYPE].host_value for n in nodes]))


def test_the_two_hash_conventions_differ_on_an_eagle_tree():
    """The premise: the read convention (plain ids) and the tree's bigram
    RadixKey convention give different keys for the same page."""
    assert _plain(3, PK[2]) != _bigram(3, PK[2])


def test_metal_shape_a_read_convention_second_piece_is_proven_and_loads_its_own_bytes():
    """RED on de6c6350f4: the second piece's page 3 carries the key the store
    read it with (plain ids chained from P's page 2); the #1424c proof knew
    only the tree's bigram key -> 'CHAIN MISMATCH ... cannot be proven', the
    13:36:10 death. With the read convention admissible, the chain is proven,
    page 1 re-pointed, and the load returns each page's own slot."""
    from flliper.srt.mem_cache import unified_radix_cache as urc

    arena = _Arena(SLOT_OF)
    tree, pool, nodes, req = _tree(arena, own_keys=[K3, K4])
    xfer = _xfer(nodes)
    tree._1424_verify_load_chain(xfer, req=req)
    want = [SLOT_OF[k] for k in (PK[1], PK[2], K3, K4)]
    assert torch.equal(xfer.host_indices - S, _rows(want) - S)
    assert nodes[1].hash_value == [K3, K4], "an admissible key is kept, never re-keyed"
    rows, order = (xfer.host_indices - S).sort()
    dev = torch.arange(len(want) * P)[order]
    dp = types.SimpleNamespace(k_buffer=[torch.zeros(len(want) * P, CELL) for _ in range(LAYERS)],
                               v_buffer=[torch.zeros(len(want) * P, CELL) for _ in range(LAYERS)])
    for layer in range(LAYERS):
        pool._load_arena(dp, rows, dev, layer)
    for layer in range(LAYERS):
        for page, slot in enumerate(want):
            assert torch.equal(dp.k_buffer[layer][page * P:(page + 1) * P], pool.arena_k_refs[layer][slot])
    assert urc.BASE_COMPONENT_TYPE in nodes[0].component_data


def test_an_unprovable_page_cuts_the_proof_depth_instead_of_the_rank():
    """A page nothing proves (its stored key is neither convention's, and its
    tokens' key has no COMPLETE slot): the side-effect-free proof names the
    depth of the last proven page -- page 3 starts at token 12, so the cut is
    12 -- and the load still refuses by name (the last latch)."""
    arena = _Arena({k: v for k, v in SLOT_OF.items() if k != K3})
    tree, _pool_, nodes, req = _tree(arena, own_keys=["ab" * 32, K4])
    assert tree.pdflip_chain_proof_depth(nodes[1], req) == 3 * P
    with pytest.raises(ap.ArenaChainMismatch, match="depth_page=3"):
        tree._1424_verify_load_chain(_xfer(nodes), req=req)
    # a fully proven chain has nothing to cut
    tree2, _p2, nodes2, req2 = _tree(_Arena(SLOT_OF), own_keys=[K3, K4])
    assert tree2.pdflip_chain_proof_depth(nodes2[1], req2) is None


class _ProbeTree:
    """match_prefix answers the host's admission: the match reaches the key's
    limit, snapped down to a recurrent anchor every 2 pages (the anchor rule),
    and the chain proof cuts at ``cut``."""

    def __init__(self, total, cut):
        self.total, self.cut = total, cut
        self.limits = []

    def match_prefix(self, params):
        lim = params.key.limit
        self.limits.append(lim)
        n = min(self.total, lim if lim is not None else self.total)
        n -= n % (2 * P)
        return types.SimpleNamespace(device_indices=torch.empty(0), host_hit_length=n,
                                     best_match_node=object())

    def pdflip_chain_proof_depth(self, node, req):
        return self.cut


def _req(total):
    return types.SimpleNamespace(rid="pdflip-2-15", origin_input_ids=list(range(total)), output_ids=[],
                                 _compute_max_prefix_len=lambda n: n - 1)


def test_the_form_a_host_votes_the_last_proven_page_group_uniform():
    """RED on de6c6350f4 (no cut: the host votes 40 and the whole group loads
    to 40, TP0 then dies on page 3's region). The host's vote is its own
    admission on the key cut at the proven depth (anchor rule included): 13
    proven tokens -> the anchor at 8. The group MIN then takes every rank
    there; the workers adopt it."""
    from flliper.srt.managers import tp_match_floor as tmf

    t = _ProbeTree(total=40 + P, cut=13)
    assert tmf.admission_probe(t, _req(45), follow=False) == 8
    assert t.limits[-1] == 13
    # nothing to cut -> the vote is the admission match, unchanged
    t2 = _ProbeTree(total=40 + P, cut=None)
    assert tmf.admission_probe(t2, _req(45), follow=False) == 40
    # a worker's follow vote never runs the proof (it holds no KV bytes)
    t3 = _ProbeTree(total=40 + P, cut=0)
    assert tmf.admission_probe(t3, _req(45), follow=True) == 40


def _x_gate(monkeypatch, *, total, head, floor, x):
    from flliper.srt.managers import scheduler as sched
    from flliper.srt.managers import tp_head_congruence as thc
    from flliper.srt.managers import tp_match_floor as tmf

    monkeypatch.setattr(thc, "group_match_for", lambda inputs, rid: head)
    monkeypatch.setattr(thc, "group_store_match_for", lambda inputs, rid: None)
    s = object.__new__(sched.Scheduler)
    s.server_args = types.SimpleNamespace(tp_prefill_max_tokens=x)
    s.ps = types.SimpleNamespace(tp_size=3)
    s.tree_cache = types.SimpleNamespace()
    tmf.plant(s.tree_cache, {"pdflip-2-15": floor} if floor is not None else None)
    req = types.SimpleNamespace(rid="pdflip-2-15", full_untruncated_fill_ids=list(range(total)),
                                prefix_indices=[], host_hit_length=head)
    return s, req


def test_rest_within_x_is_prefilled_on_d(monkeypatch):
    """Cut at 30976 of 33175 (rc12n2 shape): rest 2199 <= X=12288 -> the X
    gate admits, D re-prefills the rest from the last proven page."""
    s, req = _x_gate(monkeypatch, total=33175, head=33024, floor=30976, x=12288)
    assert s.pdflip_uncached_extent(req, head_inputs=object()) == 33175 - 30976
    assert s._pdflip_x_refuses(req, head_inputs=object()) is False


def test_rest_beyond_x_goes_via_p(monkeypatch):
    """RED on de6c6350f4 (the gate priced the head term 33024 -> 151 uncached
    -> admit, and D would prefill 20k+ itself): a cut at 12288 leaves 20887 >
    X -> W50 by name, the request goes via P (X-REQUEUE / RESUME-VIA-P)."""
    s, req = _x_gate(monkeypatch, total=33175, head=33024, floor=12288, x=12288)
    assert s.pdflip_uncached_extent(req, head_inputs=object()) == 33175 - 12288
    assert s._pdflip_x_refuses(req, head_inputs=object()) is True


def test_no_floor_prices_exactly_as_before(monkeypatch):
    s, req = _x_gate(monkeypatch, total=33175, head=33024, floor=None, x=12288)
    assert s.pdflip_uncached_extent(req, head_inputs=object()) == 151
    s, req = _x_gate(monkeypatch, total=33175, head=33024, floor=33024, x=12288)
    assert s.pdflip_uncached_extent(req, head_inputs=object()) == 151
