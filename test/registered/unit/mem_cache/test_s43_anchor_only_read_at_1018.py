"""S43: e1 ANCHOR-ONLY READ at a GIVEN depth (``weg2_anchor_only_read_at``).

PP0's TOLD-FIDELITY floor search (weg2_told_fallback.pp0_floor_below) asks
for the anchor blob of a depth every follower proved; the read is e1's
(node at exactly that depth, no Mamba state, last-page hash -> batch_get_v2
-> host Mamba value), guarded the same way (switch, GDN rank, pool), one read
per depth per request.
"""

import os
import types

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

import torch

from sglang.srt.environ import envs
from sglang.srt.managers import cache_controller as ccmod
from sglang.srt.mem_cache import unified_radix_cache as urc
from sglang.srt.mem_cache.hicache_storage import PoolName
from sglang.srt.mem_cache.radix_cache import RadixKey
from sglang.srt.mem_cache.unified_cache_components.tree_component import ComponentType

PAGE = 64
S, A = 4, 32
TOKENS = list(range(4 * PAGE))
D1, D2 = PAGE, 3 * PAGE  # two node ends on the path


class _Lru:
    def __init__(self):
        self.nodes = []

    def in_list(self, n):
        return n in self.nodes

    def insert_mru(self, n):
        self.nodes.append(n)


class _Pool:
    def __init__(self):
        self.freed = []

    def alloc_read(self, n):
        return torch.tensor([S + A + 7], dtype=torch.int64)

    def is_placeholder(self, i):
        return int(i) >= S + A

    def free(self, idx):
        self.freed.append(int(idx[0]))


class _Backend:
    def __init__(self, ok=True):
        self.ok, self.calls = ok, []

    def batch_get_v2(self, transfers):
        t = transfers[0]
        self.calls.append((t.name, list(t.keys)))
        if self.ok:
            t.host_indices[0] = 3
        return {PoolName.MAMBA: [self.ok]}


def _node(key, parent, tag, full_host=True):
    n = types.SimpleNamespace(
        key=key, parent=parent, children={},
        component_data={ComponentType.MAMBA: types.SimpleNamespace(value=None, host_value=None),
                        ComponentType.FULL: types.SimpleNamespace(
                            value=None, host_value=torch.tensor([9]) if full_host else None)},
        hash_value=[f"{tag}-last"],
    )
    n.get_last_hash_value = lambda: n.hash_value[-1]
    return n


def _tree(monkeypatch, backend, gdn=True):
    t = object.__new__(urc.UnifiedRadixCache)
    t.page_size = PAGE
    t.is_eagle = False
    root = types.SimpleNamespace(children={}, parent=None)
    t.root_node = root
    k1 = RadixKey(token_ids=TOKENS[:D1])
    n1 = _node(k1, root, "n1")
    root.children[k1.child_key(PAGE)] = n1
    k2 = RadixKey(token_ids=TOKENS[D1:D2])
    n2 = _node(k2, n1, "n2")
    n1.children[k2.child_key(PAGE)] = n2
    t.components = {ComponentType.MAMBA: types.SimpleNamespace(_mamba_pool_host=_Pool())}
    t.cache_controller = types.SimpleNamespace(storage_backend=backend)
    t.host_lru_lists = {ComponentType.MAMBA: _Lru()}
    monkeypatch.setattr(ccmod, "rank_holds_gdn_state", lambda cc: gdn)
    return t, n1, n2


def _req():
    return types.SimpleNamespace(rid="pdflip-3-186", key_match_depth=4 * PAGE, state_anchor_depth=0,
                                 full_untruncated_fill_ids=TOKENS, extra_key=None)


def test_reads_the_blob_of_the_node_at_the_given_depth(monkeypatch):
    be = _Backend()
    t, n1, n2 = _tree(monkeypatch, be)
    assert t.weg2_anchor_only_read_at(_req(), D2) == D2
    assert be.calls == [(PoolName.MAMBA, ["n2-last"])]
    assert n2.component_data[ComponentType.MAMBA].host_value is not None
    assert n1.component_data[ComponentType.MAMBA].host_value is None


def test_one_read_per_depth_and_a_miss_releases(monkeypatch):
    be = _Backend(ok=False)
    t, n1, _ = _tree(monkeypatch, be)
    req = _req()
    assert t.weg2_anchor_only_read_at(req, D1) == 0
    assert t.weg2_anchor_only_read_at(req, D1) == 0
    assert len(be.calls) == 1
    assert t.components[ComponentType.MAMBA]._mamba_pool_host.freed
    assert n1.component_data[ComponentType.MAMBA].host_value is None


def test_guards_switches_gdn_and_alignment(monkeypatch):
    be = _Backend()
    t, _, _ = _tree(monkeypatch, be, gdn=False)
    assert t.weg2_anchor_only_read_at(_req(), D2) == 0
    t, _, _ = _tree(monkeypatch, be)
    assert t.weg2_anchor_only_read_at(_req(), D2 + 1) == 0  # not a page boundary
    assert t.weg2_anchor_only_read_at(_req(), 2 * PAGE) == 0  # no node ends there
    with envs.SGLANG_WEG2_TF_ANCHOR_READ.override(False):
        assert t.weg2_anchor_only_read_at(_req(), D2) == 0
    with envs.SGLANG_WEG2_ANCHOR_ONLY_READ.override(False):
        assert t.weg2_anchor_only_read_at(_req(), D2) == 0
    assert be.calls == []


def test_a_node_with_a_state_is_not_read(monkeypatch):
    be = _Backend()
    t, _, n2 = _tree(monkeypatch, be)
    n2.component_data[ComponentType.MAMBA].host_value = torch.tensor([1])
    assert t.weg2_anchor_only_read_at(_req(), D2) == 0
    assert be.calls == []


def test_aux_host_law_no_anchor_on_a_node_without_a_full_host_copy(monkeypatch):
    """h17 15:51:02 shape (sanity_check 'mamba host present but
    Full.host_value=None'): the KV of the node is device-only -> no read."""
    be = _Backend()
    t, _, n2 = _tree(monkeypatch, be)
    n2.component_data[ComponentType.FULL].host_value = None
    assert t.weg2_anchor_only_read_at(_req(), D2) == 0
    assert be.calls == []
    assert n2.component_data[ComponentType.MAMBA].host_value is None
