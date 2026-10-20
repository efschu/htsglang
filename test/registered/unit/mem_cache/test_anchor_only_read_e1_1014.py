"""e1 ANCHOR-ONLY READ (NF h13 84e787ec96, D 14:35:36, pdflip-24-181).

The match walked the KV to key_match_depth but no node carried a Mamba state
(#904 MambaComponent:absent): a lower lane's park had spilled the anchor to
L3. The GDN-holding rank now reads ONLY that blob (keyed by the node's last
page hash, through the mamba-pool transfer and ``batch_get_v2``) and attaches
it as the node's host Mamba value. A rank without GDN state never reads --
the group's agreement stays the head vote's MIN (the mutant below reads on
such a rank and is caught)."""

import os
import types

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

import pytest
import torch

from flliper.srt.environ import envs
from flliper.srt.managers import cache_controller as ccmod
from flliper.srt.mem_cache import unified_radix_cache as urc
from flliper.srt.mem_cache.hicache_storage import PoolName
from flliper.srt.mem_cache.radix_cache import RadixKey
from flliper.srt.mem_cache.unified_cache_components.tree_component import ComponentType

PAGE = 64
S, A = 4, 32
TOKENS = list(range(3 * PAGE))
DEPTH = 2 * PAGE  # the KV chain end the walk reached


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
    def __init__(self, ok=True, row=2):
        self.ok, self.row, self.calls = ok, row, []

    def batch_get_v2(self, transfers):
        t = transfers[0]
        self.calls.append((t.name, list(t.keys)))
        if self.ok:
            t.host_indices[0] = self.row  # resolved: a staging row (ANCHOR-BYPASS) or a slot
        return {PoolName.MAMBA: [self.ok]}


def _node(key, parent, mamba=None, hashes=("h0", "h1")):
    n = types.SimpleNamespace(
        key=key, parent=parent, children={},
        component_data={ComponentType.MAMBA: types.SimpleNamespace(value=None, host_value=mamba)},
        hash_value=list(hashes),
    )
    n.get_last_hash_value = lambda: n.hash_value[-1] if n.hash_value else None
    return n


def _tree(*, backend, gdn=True, monkeypatch, mamba=None):
    t = object.__new__(urc.UnifiedRadixCache)
    t.page_size = PAGE
    t.is_eagle = False
    root = types.SimpleNamespace(children={}, parent=None)
    t.root_node = root
    k = RadixKey(token_ids=TOKENS[:DEPTH])
    child = _node(k, root, mamba=mamba)
    root.children[k.child_key(PAGE)] = child
    pool = _Pool()
    t.components = {ComponentType.MAMBA: types.SimpleNamespace(_mamba_pool_host=pool)}
    t.cache_controller = types.SimpleNamespace(storage_backend=backend)
    t.host_lru_lists = {ComponentType.MAMBA: _Lru()}
    monkeypatch.setattr(ccmod, "rank_holds_gdn_state", lambda cc: gdn)
    return t, child, pool


def _req(**kw):
    base = dict(rid="pdflip-24-181", key_match_depth=DEPTH, state_anchor_depth=0,
                full_untruncated_fill_ids=TOKENS, extra_key=None)
    base.update(kw)
    return types.SimpleNamespace(**base)


def test_the_anchor_blob_is_read_and_attached(monkeypatch):
    be = _Backend(ok=True, row=2)
    t, node, pool = _tree(backend=be, monkeypatch=monkeypatch)
    with envs.FLLIPER_PDFLIP_ANCHOR_ONLY_READ.override(True):
        assert t.pdflip_anchor_only_read(_req()) == DEPTH
    assert be.calls == [(PoolName.MAMBA, ["h1"])], "only the node's last-page anchor"
    assert int(node.component_data[ComponentType.MAMBA].host_value[0]) == 2
    assert t.host_lru_lists[ComponentType.MAMBA].nodes == [node]
    assert pool.freed == []


def test_a_rank_without_gdn_state_never_reads(monkeypatch):
    """Rank agreement: only the GDN-holding rank reads. MUTANT (guard removed:
    every rank reads and attaches) fails here."""
    be = _Backend(ok=True)
    t, node, _ = _tree(backend=be, gdn=False, monkeypatch=monkeypatch)
    with envs.FLLIPER_PDFLIP_ANCHOR_ONLY_READ.override(True):
        assert t.pdflip_anchor_only_read(_req()) == 0
    assert be.calls == []
    assert node.component_data[ComponentType.MAMBA].host_value is None


def test_a_failed_read_attaches_nothing_and_releases(monkeypatch):
    be = _Backend(ok=False)
    t, node, pool = _tree(backend=be, monkeypatch=monkeypatch)
    with envs.FLLIPER_PDFLIP_ANCHOR_ONLY_READ.override(True):
        req = _req()
        assert t.pdflip_anchor_only_read(req) == 0
        assert t.pdflip_anchor_only_read(req) == 0
    assert node.component_data[ComponentType.MAMBA].host_value is None
    assert pool.freed == [S + A + 7]
    assert len(be.calls) == 1, "one read per depth, not one per re-issue"


def test_nothing_to_do_when_the_anchor_is_there_or_the_switch_is_off(monkeypatch):
    be = _Backend(ok=True)
    t, node, _ = _tree(backend=be, monkeypatch=monkeypatch,
                       mamba=torch.tensor([5], dtype=torch.int64))
    with envs.FLLIPER_PDFLIP_ANCHOR_ONLY_READ.override(True):
        assert t.pdflip_anchor_only_read(_req()) == 0
        assert t.pdflip_anchor_only_read(_req(state_anchor_depth=DEPTH)) == 0
    t2, _, _ = _tree(backend=be, monkeypatch=monkeypatch)
    with envs.FLLIPER_PDFLIP_ANCHOR_ONLY_READ.override(False):
        assert t2.pdflip_anchor_only_read(_req()) == 0
    assert be.calls == []


def _mutant_without_gdn_guard():
    """The rank-agreement MUTANT: ``pdflip_anchor_only_read`` with its
    ``rank_holds_gdn_state`` guard deleted (the method's own source, the two
    guard lines dropped), compiled into the module's namespace."""
    import inspect
    import textwrap

    src = textwrap.dedent(inspect.getsource(urc.UnifiedRadixCache.pdflip_anchor_only_read))
    body = src.replace("        if not rank_holds_gdn_state(cc):\n            return 0\n", "")
    assert body != src, "the guard text moved -- update the mutant"
    ns = dict(vars(urc))
    exec(compile(body, "<mutant>", "exec"), ns)
    return ns["pdflip_anchor_only_read"]


def test_the_rank_agreement_test_kills_the_guardless_mutant(monkeypatch):
    """The never-reads invariant above, run against the mutant: it must FAIL
    there (the mutant reads and attaches on a rank without GDN state)."""
    be = _Backend(ok=True)
    t, node, _ = _tree(backend=be, gdn=False, monkeypatch=monkeypatch)
    mutant = _mutant_without_gdn_guard()
    with envs.FLLIPER_PDFLIP_ANCHOR_ONLY_READ.override(True):
        got = mutant(t, _req())
    assert got == DEPTH and be.calls, "the mutant reads on a non-GDN rank"
    assert node.component_data[ComponentType.MAMBA].host_value is not None
