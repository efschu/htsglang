"""AP L15-12c-C2: l2_of maps each held token's L2 host row to
(arena page slot, generation); the host rows are SNAPSHOTTED at bind time
(reset_keep nulls host_value later), staging rows have no L2 copy (-1), a
chain longer than the KV span is truncated.

Hermetic: CPU tensors only, SimpleNamespace fakes for the radix chain and
the arena host pool; no scheduler import, no GPU.
"""

from types import SimpleNamespace

import torch

from sglang.srt.mem_cache.unified_cache_components.tree_component import (
    ComponentType,
)
from sglang.srt.weg2.l15_bind import build_retain_kwargs, chain_host_rows


def _req(rid, pool_idx, in_len, out_len, node):
    return SimpleNamespace(
        rid=rid,
        req_pool_idx=pool_idx,
        origin_input_ids=list(range(in_len)),
        output_ids=list(range(out_len)),
        mamba_pool_idx=1,
        last_node=node,
        l15_kind="served",
        l15_last_active=0.0,
    )


def _req_to_token(n_slots):
    rtt = torch.zeros(4, 16, dtype=torch.int64)
    rtt[0, :n_slots] = torch.arange(1, n_slots + 1, dtype=torch.int64)
    return rtt


def _node(parent, host_rows):
    """Fake chain node: one FULL component whose host_value carries the
    node's host rows (the real shape is a list indexed by ComponentType)."""
    full = SimpleNamespace(value=None, host_value=host_rows)
    return SimpleNamespace(parent=parent, component_data=[full])


class _FakePool:
    """ArenaMHAHostPool duck: staging range, tokens-per-slot, optional
    draft row_slot, and the slot_gens accessor (generation per slot)."""

    def __init__(self, staging_rows, pages=1, row_slot=None, gens=None):
        self.staging_rows = staging_rows
        self._arena_page_tokens = pages
        self.row_slot = row_slot
        self._gens = gens or {}
        self.calls = []

    def slot_gens(self, slots):
        self.calls.append(tuple(int(s) for s in slots))
        return [int(self._gens.get(int(s), -1)) for s in slots]


def _kwargs(host_pool=None):
    kw = dict(
        caps_rows_by_rank=(1000,),
        cap_anchor_slots=10,
        prefix=[0, 1],
        rank=0,
        epoch=1,
        pid=1,
        kv_buffers=[],
        mamba_buffers=[],
        allocator=None,
        reset_keep=lambda _ns: None,
        set_keep=lambda _b, _s: None,
        manifest_path="/tmp/weg2_l15_l2_of_test.json",
        log=lambda _msg: None,
    )
    if host_pool is not None:
        kw["host_pool"] = host_pool
    return kw


def _bound(rid, in_len, out_len, node, pool, span_rows):
    req = _req(rid, 0, in_len, out_len, node)
    kw = build_retain_kwargs([req], _req_to_token(span_rows), **_kwargs(pool))
    return kw["l2_of"]


def test_chain_host_rows_walks_last_node_to_root_in_token_order():
    root = _node(None, torch.tensor([2, 3], dtype=torch.int64))
    mid = _node(root, torch.tensor([4], dtype=torch.int64))
    leaf = _node(mid, torch.tensor([5, 6], dtype=torch.int64))
    # root first: the walk runs last_node -> root and is reversed.
    assert chain_host_rows(leaf) == (2, 3, 4, 5, 6)
    # a node without a host_value contributes no rows (write-pending node).
    gap = _node(root, None)
    assert chain_host_rows(gap) == (2, 3)


def test_l2_of_slot_gen_aligned_with_slots_p1_27b_form():
    # span 5 (seqlen 6 - 1); host rows 2..6 on staging_rows=2, P=1:
    # row -> slot (row - 2) // 1; gens come from slot_gens.
    pool = _FakePool(2, pages=1, gens={0: 7, 1: 8, 2: 9, 3: 10, 4: 11})
    root = _node(None, torch.tensor([2, 3, 4], dtype=torch.int64))
    leaf = _node(root, torch.tensor([5, 6], dtype=torch.int64))
    l2_of = _bound("r1", 4, 2, leaf, pool, 5)
    slots, gens = l2_of("r1")
    assert slots == (0, 1, 2, 3, 4)
    assert gens == (7, 8, 9, 10, 11)


def test_l2_of_pages_per_slot_divides_the_arena_span():
    # P=2: host ids are TOKENS, slot = (row - S) // P; rows 4..7 -> slots 1,1,2,2.
    pool = _FakePool(2, pages=2, gens={1: 5, 2: 6})
    root = _node(None, torch.tensor([4, 5, 6, 7], dtype=torch.int64))
    leaf = _node(root, None)
    l2_of = _bound("r2", 4, 1, leaf, pool, 4)
    slots, gens = l2_of("r2")
    assert slots == (1, 1, 2, 2)
    assert gens == (5, 5, 6, 6)


def test_l2_of_staging_rows_are_minus_one():
    # rows 0,1 are staging (< S): no L2 copy -> (-1, -1), no slot_gens entry.
    pool = _FakePool(2, pages=1, gens={0: 3})
    root = _node(None, torch.tensor([0, 1, 2], dtype=torch.int64))
    l2_of = _bound("r3", 2, 2, root, pool, 3)
    slots, gens = l2_of("r3")
    assert slots == (-1, -1, 0)
    assert gens == (-1, -1, 3)


def test_l2_of_draft_role_uses_row_slot_and_misses_stay_minus_one():
    # row_slot maps rows directly to draft slots; an unmapped row is a miss.
    pool = _FakePool(2, pages=1, row_slot={3: 9}, gens={9: 4})
    root = _node(None, torch.tensor([3, 4], dtype=torch.int64))
    l2_of = _bound("r4", 1, 2, root, pool, 3)
    slots, gens = l2_of("r4")
    assert slots == (9, -1)
    assert gens == (4, -1)


def test_l2_of_truncates_a_chain_longer_than_the_kv_span():
    # chain carries 7 host rows, the KV span is 5: the snapshot is cut to
    # the span (the seqlen-1 rule, schedule_batch.py:2821).
    pool = _FakePool(2, pages=1, gens={i: i + 1 for i in range(7)})
    root = _node(None, torch.tensor([2, 3, 4, 5, 6, 7, 8], dtype=torch.int64))
    l2_of = _bound("r5", 5, 1, root, pool, 5)
    slots, gens = l2_of("r5")
    assert slots == (0, 1, 2, 3, 4)
    assert gens == (1, 2, 3, 4, 5)


def test_l2_of_without_pool_stays_empty():
    # No host_pool kwarg and a plain lambda reset_keep: nothing to derive,
    # the pre-L15-12c-C2 behaviour (empty columns) is kept.
    root = _node(None, torch.tensor([2], dtype=torch.int64))
    l2_of = _bound("r6", 1, 1, root, None, 2)
    assert l2_of("r6") == ((), ())
    assert l2_of("no-such-rid") == ((), ())


def test_l2_of_pool_derived_from_reset_keep_self():
    # Production wiring: the scheduler passes reset_keep=self.tree_cache.reset_keep;
    # the pool is reached via __self__.cache_controller.mem_pool_host.
    pool = _FakePool(2, pages=1, gens={0: 2})
    tree_cache = SimpleNamespace(
        cache_controller=SimpleNamespace(mem_pool_host=pool),
        reset_keep=lambda _ns: None,
    )

    def reset_keep(_ns):  # bound-method stand-in with __self__
        return tree_cache.reset_keep(_ns)

    reset_keep.__self__ = tree_cache
    root = _node(None, torch.tensor([2], dtype=torch.int64))
    req = _req("r7", 0, 1, 1, root)
    kw = build_retain_kwargs(
        [req], _req_to_token(2), **{**_kwargs(), "reset_keep": reset_keep}
    )
    assert kw["l2_of"]("r7") == ((0,), (2,))
