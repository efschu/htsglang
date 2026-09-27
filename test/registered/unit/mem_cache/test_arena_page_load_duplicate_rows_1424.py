"""#1424 (rc12m-dpr D-TP0 11:49:05 and 12:19:06): park loadbacks over a
prefix that arrived in two prefetch pieces died in the paged arena load.

11:49: weg2-17-71 and weg2-11-56, each re-read its last 384 tokens
(START-LOADING nodes=3 tokens=37376). 12:19: weg2-18-63 alone (15168 +
1984, #988 17152) -- one request, so not two copies of a shared page; the
chain check below proves (and re-points) such a chain. ``move_indices``
(io_backend direct, layer_first) sorts the merged host indices, so each
duplicated row sits next to its twin -- r0,r0,r1,r1,... -- and
``_page_slots`` raised 'a page's token rows are not consecutive from its
first id'. Loading one row into two device rows is legal; only the
whole-page fast path cannot express it. The control batch at 11:45:38
(weg2-0-3 + weg2-5-28, same prefix 34304) survived because the second
request found the shared nodes already on the device and loaded only its
own 3904 tokens -- no duplicate rows. 27B's arena is unpaged (P == 1).

Hermetic: the real ``ArenaMHAHostPool._load_arena`` and ``_transfer_paged``
on a bare instance with CPU tensors."""
from __future__ import annotations

import os
import types

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

import torch  # noqa: E402

from sglang.srt.mem_cache.pool_host import arena_pool as ap  # noqa: E402

P = 4
SLOTS = 3
CELL = 2
LAYERS = 2


def _pool():
    pool = object.__new__(ap.ArenaMHAHostPool)
    pool._arena_page_tokens = P
    pool._page_view = object()
    pool._page_key_hint = None
    pool._page_loaded_key = None
    pool.row_slot = None
    # arena[layer][slot, token, cell] = distinct values per row
    base = torch.arange(SLOTS * P * CELL, dtype=torch.float32).view(SLOTS, P, CELL)
    pool.arena_k_refs = [base + 1000 * layer for layer in range(LAYERS)]
    pool.arena_v_refs = [-(base + 1000 * layer) for layer in range(LAYERS)]
    pool.page_loads = []
    pool.pin_slots = lambda slots: 0
    pool._arena_load_guard = lambda *a, **k: None
    pool._load_pages_all_layers = lambda dp, slots, di: pool.page_loads.append(slots.tolist())
    return pool


def _device(n):
    return types.SimpleNamespace(
        k_buffer=[torch.zeros(n, CELL) for _ in range(LAYERS)],
        v_buffer=[torch.zeros(n, CELL) for _ in range(LAYERS)],
    )


def test_metal_shape_duplicated_rows_after_the_sort_load_row_by_row():
    """RED on b6e2235d8a: RuntimeError '#1424 paged arena load ...'."""
    pool = _pool()
    page = torch.arange(P, P + P)                      # slot 1, rows 4..7
    rows, _ = torch.cat([page, page]).sort()           # 4,4,5,5,6,6,7,7
    dev_idx = torch.arange(2 * P)
    dp = _device(2 * P)
    for layer in range(LAYERS):
        pool._load_arena(dp, rows, dev_idx, layer)
    for layer in range(LAYERS):
        want = pool.arena_k_refs[layer].view(-1, CELL)[rows]
        assert torch.equal(dp.k_buffer[layer], want)
        assert torch.equal(dp.v_buffer[layer], pool.arena_v_refs[layer].view(-1, CELL)[rows])
    assert pool.page_loads == [], "the whole-page path cannot express duplicated rows"


def test_whole_consecutive_pages_keep_the_page_path():
    pool = _pool()
    rows = torch.arange(0, 2 * P)                      # slots 0 and 1, whole pages
    pool._load_arena(_device(2 * P), rows, torch.arange(2 * P), 0)
    assert pool.page_loads == [[0, 1]]


def test_the_fallback_is_named():
    pool = _pool()
    rows, _ = torch.cat([torch.arange(P), torch.arange(P)]).sort()
    before = ap._PAGE_FALLBACK_N[0]
    pool._load_arena(_device(2 * P), rows, torch.arange(2 * P), 0)
    assert ap._PAGE_FALLBACK_N[0] == before + 1


# -- the park loadback over a prefix that arrived in TWO prefetch pieces --------
#
# 12:19:06 (weg2-18-63, ONE request in the batch): prefetch 1 INCOMPLETE
# (matched 3840 + loaded 11328 = 15168, shortfall 1984), prefetch 2 loaded
# the rest (matched 0, loaded 1984), then the #988 park loadback of 17152
# died in the paged load. One request, one chain -- so the rows the page
# path refused were not two requests' copies of one page. A correct chain is
# whole consecutive pages by construction; the fallback gather alone would
# load whatever a wrong row points at. ``verify_load_chain`` proves such a
# chain against the page keys and re-points a wrong page to its key's slot.

S = 5  # staging rows before the arena ids


class _Arena:
    def __init__(self, key_slot, state=2):
        self.key_slot = dict(key_slot)
        self.state = state
        self.finds = 0
        self.refs = []

    def find_slots_np(self, stems):
        import numpy as np
        self.finds += 1
        slots = np.array([self.key_slot.get(s, -1) for s in stems], dtype=np.int64)
        st = np.array([self.state if s in self.key_slot else 0 for s in stems], dtype=np.int8)
        return slots, st

    def ref_slots(self, slots, delta):
        self.refs.append((list(slots), delta))
        return len(slots)


def _node(nid, keys, slots):
    rows = torch.cat([torch.arange(s * P, s * P + P) for s in slots]) + S
    return types.SimpleNamespace(id=nid, hash_value=list(keys), host_value=rows.clone())


def _chain_pool(arena):
    pool = _pool()
    pool.staging_rows = S
    pool.arena = arena
    return pool


def _verify(pool, nodes):
    host = torch.cat([n.host_value for n in nodes])
    return ap.verify_load_chain(pool, nodes, host, rows_of=lambda n: n.host_value, stems_of=lambda h: h)


def test_two_piece_chain_that_is_whole_pages_passes_on_shape_alone():
    arena = _Arena({"a": 0, "b": 2, "c": 1})
    pool = _chain_pool(arena)
    a, b, c = _node(1, ["a"], [0]), _node(2, ["b"], [2]), _node(3, ["c"], [1])  # piece 1 = a+b, piece 2 = c
    host = torch.cat([a.host_value, b.host_value, c.host_value])
    out = ap.verify_load_chain(pool, [a, b, c], host, rows_of=lambda n: n.host_value, stems_of=lambda h: h)
    assert out is host and arena.finds == 0, "a clean chain costs no arena lookup"


def test_metal_shape_second_piece_on_a_foreign_slot_is_repointed_and_loads_its_own_bytes():
    """The one-request death: the second piece's page addresses the slot of a
    page of the first piece. The page path refused (RED on 2f800cac1a: there
    is no chain check); the per-row gather alone would load slot 2's KV under
    the tokens of key 'c'. The chain check re-points the page to the slot its
    key names -- and the load returns slot 1's bytes there."""
    arena = _Arena({"a": 0, "b": 2, "c": 1})
    pool = _chain_pool(arena)
    a, b = _node(1, ["a"], [0]), _node(2, ["b"], [2])
    c = _node(3, ["c"], [2])                               # wrong: slot 2 is b's page
    host = _verify(pool, [a, b, c])
    want_slots = [0, 2, 1]
    assert torch.equal(host - S, torch.cat([torch.arange(s * P, s * P + P) for s in want_slots]))
    assert torch.equal(c.host_value - S, torch.arange(1 * P, 1 * P + P)), "the node itself is corrected"
    assert arena.refs == [([1], 1)], "the key's slot is referenced"
    # the load as start_loading runs it: host rows sorted with their device rows
    rows, order = (host - S).sort()
    dev = torch.arange(3 * P)[order]
    pool._page_view = None                                 # per-layer path: the bytes are checked
    dp = _device(3 * P)
    for layer in range(LAYERS):
        pool._load_arena(dp, rows, dev, layer)
    for layer in range(LAYERS):
        arena_rows = pool.arena_k_refs[layer].view(-1, CELL)
        for block, slot in enumerate(want_slots):
            got = dp.k_buffer[layer][block * P:(block + 1) * P]
            assert torch.equal(got, arena_rows[slot * P:(slot + 1) * P]), (layer, block, slot)


def test_a_page_its_key_cannot_prove_is_a_named_stop():
    arena = _Arena({"a": 0, "b": 2})                       # key 'c' is not complete in the arena
    pool = _chain_pool(arena)
    a, b, c = _node(1, ["a"], [0]), _node(2, ["b"], [2]), _node(3, ["c"], [2])
    try:
        _verify(pool, [a, b, c])
    except ap.ArenaChainMismatch as exc:
        assert "#1424 CHAIN MISMATCH" in str(exc) and "node=3" in str(exc)
    else:
        raise AssertionError("an unprovable page must not be loaded")


def test_rows_that_are_not_whole_pages_of_their_keys_are_a_named_stop():
    arena = _Arena({"a": 0, "b": 2})
    pool = _chain_pool(arena)
    a = _node(1, ["a"], [0])
    b = types.SimpleNamespace(id=2, hash_value=["b"], host_value=torch.arange(2 * P, 2 * P + P - 1) + S)
    try:
        _verify(pool, [a, b])
    except ap.ArenaChainMismatch as exc:
        assert "node=2" in str(exc)
    else:
        raise AssertionError("rows that are not whole pages of their keys must not be loaded")


def test_load_back_proves_the_chain_before_it_is_queued():
    """The real ``UnifiedRadixCache._1424_verify_load_chain`` on the tree's
    own objects: the kv transfer's host indices and the node's host value
    come back re-pointed, keys run through the store's stem suffix."""
    from sglang.srt.mem_cache import unified_radix_cache as urc

    arena = _Arena({"a#": 0, "b#": 2, "c#": 1})
    pool = _chain_pool(arena)
    backend = types.SimpleNamespace(_suffix_for_key=lambda k: ("#", False))
    group = types.SimpleNamespace(arena_read=True, anchor_entry=types.SimpleNamespace(host_pool=pool))
    tree = object.__new__(urc.UnifiedRadixCache)
    tree.cache_controller = types.SimpleNamespace(mem_pool_host=group, storage_backend=backend)

    def node(nid, key, slot):
        n = _node(nid, [key], [slot])
        n.component_data = {urc.BASE_COMPONENT_TYPE: types.SimpleNamespace(host_value=n.host_value)}
        return n

    nodes = [node(1, "a", 0), node(2, "b", 2), node(3, "c", 2)]   # c on b's slot
    xfer = types.SimpleNamespace(nodes_to_load=nodes,
                                 host_indices=torch.cat([n.host_value for n in nodes]))
    tree._1424_verify_load_chain(xfer)
    want = torch.cat([torch.arange(s * P, s * P + P) for s in (0, 2, 1)]) + S
    assert torch.equal(xfer.host_indices, want)
    assert torch.equal(nodes[2].component_data[urc.BASE_COMPONENT_TYPE].host_value - S, torch.arange(P, 2 * P))


def test_two_requests_sharing_the_second_piece_merge_into_duplicates_the_gather_loads():
    """11:49:05 shape: two parked requests, each chain proven clean, share a
    second-piece page; the MERGED load holds its rows twice. The gather loads
    that page's bytes into both requests' device rows."""
    arena = _Arena({"a": 0, "b": 2, "c": 1})
    pool = _chain_pool(arena)
    chain1 = [_node(1, ["a"], [0]), _node(3, ["c"], [1])]
    chain2 = [_node(2, ["b"], [2]), _node(3, ["c"], [1])]
    h1, h2 = _verify(pool, chain1), _verify(pool, chain2)
    assert arena.finds == 0
    rows, order = (torch.cat([h1, h2]) - S).sort()
    dev = torch.arange(4 * P)[order]
    dp = _device(4 * P)
    for layer in range(LAYERS):
        pool._load_arena(dp, rows, dev, layer)
    merged = torch.cat([h1, h2]) - S
    for layer in range(LAYERS):
        assert torch.equal(dp.k_buffer[layer], pool.arena_k_refs[layer].view(-1, CELL)[merged])
