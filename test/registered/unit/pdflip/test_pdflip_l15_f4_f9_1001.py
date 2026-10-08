"""AP L15-12c-F4/F9: two findings of the L15 retain E2E review.

F4: chain_host_rows must keep token POSITIONS when a chain node has no
host_value -- the node contributes len(its tokens) placeholders -1, so a
hole mid-chain does not shift the later nodes' host rows onto the wrong
token positions (a shifted row would map, via l2_of, to a foreign
(slot, gen)). -1 placeholders map to (slot, gen) = (-1, -1), which the
wake's gen check drops.

F9: a radix tree carrying a THIRD radix component (e.g. SWA) beyond
FULL+MAMBA must hard-gate (L15UnsupportedTreeError), not silently keep
un-remapped, un-kept state on the kept nodes.

Hermetic: CPU tensors only, SimpleNamespace fakes for the radix chain and
the arena host pool; no scheduler import, no GPU.
"""

from types import SimpleNamespace

import pytest
import torch

from flliper.srt.mem_cache.unified_cache_components.tree_component import (
    ComponentType,
)
from flliper.srt.pdflip.l15_bind import build_retain_kwargs, chain_host_rows


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


def _full(value=None, host_value=None):
    return SimpleNamespace(value=value, host_value=host_value)


def _node(parent, cd_full, key=None):
    """Fake chain node: one FULL component slot; host_value carries the
    node's host rows, value carries its own device slot ids (token count)."""
    node = SimpleNamespace(parent=parent, component_data=[cd_full])
    if key is not None:
        node.key = key
    return node


class _FakePool:
    """ArenaMHAHostPool duck: staging range, tokens-per-slot and the
    slot_gens accessor (generation per arena page slot)."""

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
        manifest_path="/tmp/pdflip_l15_f4_f9_1001.json",
        log=lambda _msg: None,
    )
    if host_pool is not None:
        kw["host_pool"] = host_pool
    return kw


# ---------------------------------------------------------------- F4


def _hole_chain():
    """root -> first(host) -> mid(NO host_value, 4 tokens) -> last(host)."""
    first = _node(None, _full(value=[0, 1, 2], host_value=[10, 11, 12]))
    mid = _node(first, _full(value=[3, 4, 5, 6], host_value=None))
    last = _node(mid, _full(value=[7, 8], host_value=[30, 31]))
    return first, mid, last


def test_chain_hole_keeps_positions_with_placeholders():
    # F4 core: the mid node's 4 tokens become 4 x -1 placeholders, so the
    # last node's rows sit at THEIR token positions, not shifted left.
    _, _, last = _hole_chain()
    assert chain_host_rows(last) == (10, 11, 12, -1, -1, -1, -1, 30, 31)


def test_chain_hole_token_count_falls_back_to_node_key():
    # No component value -> token count comes from the node's radix key.
    first = _node(None, _full(host_value=[10, 11, 12]))
    mid = _node(first, _full(host_value=None), key=[3, 4, 5])
    last = _node(mid, _full(host_value=[30, 31]))
    assert chain_host_rows(last) == (10, 11, 12, -1, -1, -1, 30, 31)


def test_chain_hole_without_any_token_count_skips():
    # Documented fallback: neither component value nor key -> today's
    # positional skip (nothing sensible to count).
    first = _node(None, _full(host_value=[10, 11]))
    mid = _node(first, _full(host_value=None))
    last = _node(mid, _full(host_value=[30, 31]))
    assert chain_host_rows(last) == (10, 11, 30, 31)


def test_l2_of_hole_maps_placeholders_to_minus_one_in_place():
    # End to end through build_retain_kwargs: staging_rows=5, page=1, so
    # host rows 10,11,12 -> arena slots 5,6,7 and rows 30,31 -> 25,26;
    # the -1 placeholders come out as (slot, gen) = (-1, -1) EXACTLY at
    # the middle positions, not shifting the last node's columns.
    _, _, last = _hole_chain()
    pool = _FakePool(5, pages=1, gens={5: 7, 6: 7, 7: 7, 25: 3, 26: 3})
    req = _req("rh", 0, 9, 1, last)
    kw = build_retain_kwargs([req], _req_to_token(9), **_kwargs(pool))
    slots, gens = kw["l2_of"]("rh")
    assert slots == (5, 6, 7, -1, -1, -1, -1, 25, 26)
    assert gens == (7, 7, 7, -1, -1, -1, -1, 3, 3)


# ---------------------------------------------------------------- F9


def test_third_component_raises_hard_gate():
    # F9: a tree registering a THIRD radix component (SWA) beyond
    # FULL+MAMBA must abort the retain round hard, naming the component
    # (the scheduler hook catches the pre-move error and flushes).
    from flliper.srt.pdflip.l15_bind import L15UnsupportedTreeError

    full = _full(value=[0, 1], host_value=None)
    swa = SimpleNamespace(value=[5, 6], host_value=None)
    mamba = SimpleNamespace(value=[1], host_value=None)
    node = SimpleNamespace(
        parent=None,
        key=[0, 1],
        component_data=[full, swa, mamba],
        tree_components=(
            ComponentType.FULL,
            ComponentType.SWA,
            ComponentType.MAMBA,
        ),
    )
    req = _req("rs", 0, 2, 1, node)
    with pytest.raises(L15UnsupportedTreeError) as ei:
        build_retain_kwargs([req], _req_to_token(2), **_kwargs())
    msg = str(ei.value)
    assert "swa" in msg.lower()
    assert "FULL+MAMBA" in msg


def test_full_mamba_tree_passes_the_gate():
    # The 27B/NF shape: component_data is a fixed-length list, so the SWA
    # data slot EXISTS but carries nothing and is not registered -> no
    # raise, the retain kwargs are assembled as before.
    full = _full(value=[0, 1], host_value=None)
    swa = SimpleNamespace(value=None, host_value=None)
    mamba = SimpleNamespace(value=[1], host_value=None)
    node = SimpleNamespace(
        parent=None,
        key=[0, 1],
        component_data=[full, swa, mamba],
        tree_components=(ComponentType.FULL, ComponentType.MAMBA),
    )
    req = _req("rp", 0, 2, 1, node)
    kw = build_retain_kwargs([req], _req_to_token(2), **_kwargs())
    assert kw["l2_of"]("rp") == ((), ())  # ran through, gate not hit
