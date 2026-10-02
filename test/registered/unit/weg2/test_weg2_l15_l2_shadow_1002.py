"""L15-L2-SHADOW (N5n 14:44:07, dac8b62b8c): the #248 release dropped the KV
host rows of a freshly loaded LONG (P's prefill pages, read from the shared
arena) right after the load -- the pages stayed COMPLETE in L2, the node
forgot where. At the L1.5 sleep the chain was "unbacked" (L15-HOSTLOCK
slots=2304 of 133k tokens). The release now records (rows, arena gen) per
released node; the bind adopts a shadow row only while its slot still carries
that generation (a re-claim bumps it -> -1, never a foreign source).

Hermetic: CPU tensors, SimpleNamespace fakes; no scheduler, no GPU.
"""
import os
from types import SimpleNamespace

import torch

from sglang.srt.mem_cache.unified_cache_components.tree_component import ComponentType
from sglang.srt.weg2 import park_l3
from sglang.srt.weg2.l15_bind import build_retain_kwargs, chain_host_rows, chain_host_rows_ex


class _Pool:
    def __init__(self, staging_rows=10, gens=None):
        self.staging_rows = staging_rows
        self._arena_page_tokens = 1
        self.row_slot = None
        self._gens = dict(gens or {})

    def slot_gens(self, slots):
        return [int(self._gens.get(int(s), -1)) for s in slots]


def _node(parent, n_tok, host_rows=None):
    full = SimpleNamespace(value=list(range(n_tok)), host_value=host_rows)
    nd = SimpleNamespace(parent=parent, component_data=[full], key=list(range(n_tok)))
    return nd


def _bind(node, pool, n_slots, logs):
    rtt = torch.zeros(2, 16, dtype=torch.int64)
    # the span excludes the unwritten last position: one input id more than held slots
    rtt[0, :n_slots + 1] = torch.arange(1, n_slots + 2, dtype=torch.int64)
    req = SimpleNamespace(rid="weg2-8-8", req_pool_idx=0, origin_input_ids=list(range(n_slots + 1)),
                          output_ids=[], mamba_pool_idx=1, last_node=node,
                          l15_kind="served", l15_last_active=0.0)
    kw = build_retain_kwargs(
        [req], rtt, caps_rows_by_rank=(1000,), cap_anchor_slots=10, prefix=[0, 1], rank=0,
        epoch=1, pid=1, kv_buffers=[], mamba_buffers=[], allocator=None,
        reset_keep=lambda _ns: None, set_keep=lambda _b, _s: None,
        manifest_path="/tmp/weg2_l15_l2_shadow_test.json", log=logs.append, host_pool=pool)
    return kw["l2_of"]("weg2-8-8")


def test_record_shadow_takes_the_gen_at_release():
    pool = _Pool(staging_rows=10, gens={0: 7, 1: 7, 2: 9})
    nd = _node(None, 3)
    n = park_l3.record_l2_shadow(pool, [(nd, [10, 11, 12])])
    assert n == 3
    assert getattr(nd, park_l3.L2_SHADOW_ATTR) == ((10, 11, 12), (7, 7, 9))


def test_shadow_rows_are_adopted_where_the_gen_still_matches():
    # prefix node (P's 3 pages, released -> shadow), tail node (D's decode, live host_value)
    pool = _Pool(staging_rows=10, gens={0: 7, 1: 7, 2: 9, 5: 4})
    root = _node(None, 3)
    setattr(root, park_l3.L2_SHADOW_ATTR, ((10, 11, 12), (7, 7, 9)))
    tail = _node(root, 1, host_rows=torch.tensor([15], dtype=torch.int64))
    rows, rec = chain_host_rows_ex(tail)
    assert rows == (10, 11, 12, 15) and rec == (7, 7, 9, None)
    logs = []
    slots, gens = _bind(tail, pool, 4, logs)
    assert slots == (0, 1, 2, 5) and gens == (7, 7, 9, 4)
    assert any("L15-L2-SHADOW-ADOPT rid=weg2-8-8 tokens=4 live=1 shadow=3 valid=3 stale=0" in m
               for m in logs)


def test_a_reclaimed_slot_is_unbacked_never_foreign():
    pool = _Pool(staging_rows=10, gens={0: 7, 1: 8, 2: 9})   # slot 1 re-claimed: gen 7 -> 8
    root = _node(None, 3)
    setattr(root, park_l3.L2_SHADOW_ATTR, ((10, 11, 12), (7, 7, 9)))
    logs = []
    slots, gens = _bind(root, pool, 3, logs)
    assert slots == (0, -1, 2) and gens == (7, -1, 9)
    assert any("valid=2 stale=1" in m for m in logs)


def test_without_shadow_the_old_placeholders_stay():
    root = _node(None, 3)
    assert chain_host_rows(root) == (-1, -1, -1)
    # a shadow whose length no longer matches the node (radix split) is ignored
    setattr(root, park_l3.L2_SHADOW_ATTR, ((10, 11), (7, 7)))
    assert chain_host_rows(root) == (-1, -1, -1)


def test_shadow_only_under_the_l15_master(monkeypatch):
    monkeypatch.delenv("SGLANG_WEG2_L15", raising=False)
    assert park_l3._l15_shadow_on() is False
    monkeypatch.setenv("SGLANG_WEG2_L15", "1")
    assert park_l3._l15_shadow_on() is True
    monkeypatch.setenv("SGLANG_WEG2_L15_L2_SHADOW", "0")
    assert park_l3._l15_shadow_on() is False


def test_a_slot_not_complete_at_the_bind_is_no_l2_source(monkeypatch):
    """L15-L2-REQUIRE-COMPLETE: a live tail row whose page the census does not
    see COMPLETE (gen -1, write-through in flight) is unbacked, never sampled
    or refilled. =0 restores the old pairing (slot kept with gen -1)."""
    monkeypatch.delenv("SGLANG_WEG2_L15_L2_REQUIRE_COMPLETE", raising=False)
    pool = _Pool(staging_rows=10, gens={0: 7, 1: 7})          # slot 2 not COMPLETE
    root = _node(None, 3, host_rows=torch.tensor([10, 11, 12], dtype=torch.int64))
    logs = []
    slots, gens = _bind(root, pool, 3, logs)
    assert slots == (0, 1, -1) and gens == (7, 7, -1)
    assert any("L15-L2-INCOMPLETE rid=weg2-8-8 rows=1 of 3" in m for m in logs)
    monkeypatch.setenv("SGLANG_WEG2_L15_L2_REQUIRE_COMPLETE", "0")
    slots, gens = _bind(root, pool, 3, [])
    assert slots == (0, 1, 2) and gens == (7, 7, -1)
