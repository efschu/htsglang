"""Q-697c DUAL HOST-ONLY SPILL (27B NVFP4 dual C, boot ...10032243 @54a30199b1, P PP0
22:51:19Z): the KV L2 arena of group P was full (720896 slots, every one referenced
by P's own host tree) and every backup was refused ('#1421 BACKUP-REFUSED
why=arena_claim'); the W3-ARENA spill wrote not one line.

Two causes, both in ``UnifiedRadixCache._w3_arena_spill``:

1. the 27B's host pool is the hybrid HostPoolGroup, whose ``__getattr__`` forwards only
   the claim calls -- ``hasattr(pool, "secure_rows_to_l3")`` was False and the spill
   returned 0 before its first log line;
2. the spill took only nodes that still carry a device value, and P's dual tree is
   host-only after every idle release.

The fix (dual P layout only): the spill resolves the group's anchor pool and, when the
device-resident round is not enough, spills host-only H-LEAVES (L3 copy first, then the
leaf goes). Hermetic: the real C arena on a temp file, the real ArenaMHAHostPool, the
real ``_weg2_direct_claim`` / ``_w3_arena_spill``; pages of 64 B, page_size 1.

This file is deliberately written without importing the new module at the top: on the
base the tests fail by BEHAVIOUR (the claim is refused), not by an import error."""

import logging
import os
import shutil
import sys
import types

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

import pytest  # noqa: E402
import torch  # noqa: E402

sys.path.insert(0, os.path.dirname(__file__))
import test_w3_arena_spill_0929 as W  # noqa: E402

from sglang.srt.mem_cache.unified_radix_cache import UnifiedTreeNode  # noqa: E402
from sglang.srt.weg2 import dual_p_kv_stage as PK  # noqa: E402

pytestmark = pytest.mark.skipif(shutil.which("gcc") is None, reason="needs gcc")

FULL = W.FULL
SLOTS = 8
DUAL_P = {"SGLANG_WEG2_DUAL_LAYOUT": "1", "SGLANG_WEG2_GROUP": "P", PK.MAX_TOKENS_ENV: "131072"}
NOT_DUAL_P = [
    {},
    {"SGLANG_WEG2_GROUP": "P"},
    {"SGLANG_WEG2_DUAL_LAYOUT": "1"},
    {"SGLANG_WEG2_DUAL_LAYOUT": "1", "SGLANG_WEG2_GROUP": "D", PK.MAX_TOKENS_ENV: "131072"},
    {"SGLANG_WEG2_DUAL_LAYOUT": "1", "SGLANG_WEG2_GROUP": "P"},   # no P KV cap: not armed
]
GATE_KEYS = ("SGLANG_WEG2_DUAL_LAYOUT", "SGLANG_WEG2_GROUP", PK.MAX_TOKENS_ENV)


def _env(monkeypatch, env):
    for k in GATE_KEYS:
        monkeypatch.delenv(k, raising=False)
    for k, v in env.items():
        monkeypatch.setenv(k, v)


class _Group:
    """The hybrid HostPoolGroup as the 27B's P has it: the claim calls are forwarded
    to the anchor (arena) pool, everything else -- secure_rows_to_l3 included -- is
    NOT (``HostPoolGroup.__getattr__`` raises AttributeError for it)."""

    FORWARDED = ("alloc_read", "alloc_write", "complete_write", "abort_write", "pin_slots")

    def __init__(self, pool):
        self.anchor_entry = types.SimpleNamespace(host_pool=pool)
        self.arena_read = True

    @property
    def arena(self):
        return self.anchor_entry.host_pool.arena

    @property
    def staging_rows(self):
        return self.anchor_entry.host_pool.staging_rows

    def ensure_bound(self, storage_backend, role="kv"):
        return True

    def __getattr__(self, name):
        if name in self.FORWARDED:
            return getattr(self.anchor_entry.host_pool, name)
        raise AttributeError(name)


def _host_only_tree(tmp_path, slots=SLOTS, refuse=(), group=True, chain=True):
    """The dual P shape: every slot COMPLETE and referenced by a HOST-ONLY node (the
    idle release took the device KV away). One chain by default (each node the child
    of the last), else siblings."""
    p, arena, root = W._pool(tmp_path, slots, refuse)
    t = W._tree(p)
    t._components_tuple = (t.components[FULL],)
    t.tree_components = (FULL,)
    t._update_evictable_leaf_sets = lambda node: None
    t._record_remove_event = lambda node, medium=None: None
    t._remove_leaf_from_parent = lambda node: node.parent.children.pop(
        next(k for k, c in node.parent.children.items() if c is node))
    t._iteratively_delete_tombstone_leaf = lambda node, tracker: None
    if group:
        g = _Group(p)
        t.cache_controller.mem_pool_host = g
        t._weg2_direct_pool = lambda: g
    nodes, parent = [], t.root_node
    for i in range(slots):
        rows = W._finished(p, arena, [f"a{i}"], 0x40 + i)
        n = W._node(t, parent if chain else t.root_node, [f"a{i}"], rows)
        n.component_data[FULL].value = None              # host-only: the device KV is gone
        n.l3_present = False
        nodes.append(n)
        if chain:
            parent = n
    for n in nodes:
        n.component_data[FULL].host_lock_ref = 0
    return p, arena, root, t, nodes


def _spilled(nodes):
    return [n for n in nodes if n.parent is None or n.component_data[FULL].host_value is None]


def _in_tree(t, n):
    return n.parent is not None and any(c is n for c in n.parent.children.values())


def test_group_pool_without_the_gate_stops_as_before_flip_unchanged(tmp_path, monkeypatch):
    """Gate off (flip / INT8 / NF / dual D / half-set gates): the claim is refused,
    nothing reaches L3, no node moves -- the old behaviour byte for byte."""
    for env in NOT_DUAL_P:
        _env(monkeypatch, env)
        sub = tmp_path / ("e%d" % len(os.listdir(tmp_path)))
        sub.mkdir()
        p, arena, root, t, nodes = _host_only_tree(sub)
        m = W._claimer(t)
        assert t._weg2_direct_claim(m) is False, env
        assert t.refused == ["arena_claim"], env
        assert not list(root.iterdir()), env
        assert all(_in_tree(t, n) and n.component_data[FULL].host_value is not None for n in nodes), env


def test_plain_pool_host_only_nodes_stay_without_the_gate(tmp_path, monkeypatch):
    """A pool that carries secure_rows_to_l3 itself (the non-hybrid forms): host-only
    nodes stay -- their host copy is their only copy (the W3 rule, unchanged)."""
    _env(monkeypatch, {})
    p, arena, root, t, nodes = _host_only_tree(tmp_path, group=False)
    assert t._weg2_direct_claim(W._claimer(t)) is False
    assert all(_in_tree(t, n) for n in nodes)
    assert not list(root.iterdir())


def test_c_wedge_a_full_arena_of_host_only_nodes_spills_to_l3_and_the_claim_gets_room(
        tmp_path, monkeypatch, caplog):
    """RED on ad50a50bfb: refused (#1421 arena_claim), nothing in L3 -- the dual C wedge.
    GREEN: the host-only leaves get their L3 copy, go, and the claim finds its room."""
    _env(monkeypatch, DUAL_P)
    p, arena, root, t, nodes = _host_only_tree(tmp_path)
    caplog.set_level(logging.INFO)
    m = W._claimer(t)
    pre = t._weg2_direct_claim(m)
    assert pre is not False, f"claim refused: {t.refused}"
    assert pre is not None and int(pre.numel()) == 2
    gone = _spilled(nodes)
    assert gone, "no host-only node spilled"
    for n in gone:
        st = n.hash_value[0] + "_sfx"
        assert (root / (st + ".bin")).read_bytes() == bytes([0x40 + int(n.hash_value[0][1:])]) * W.PAGE
        assert not _in_tree(t, n)
    assert any("Q-697c DUAL HOST-ONLY SPILL" in r.getMessage() for r in caplog.records)


def test_a_chain_is_spilled_leaf_first_and_the_parent_follows(tmp_path, monkeypatch):
    """Only an H-leaf may go; once it did, its parent is one and goes next."""
    _env(monkeypatch, DUAL_P)
    from sglang.srt.weg2 import dual_arena_spill as D

    p, arena, root, t, nodes = _host_only_tree(tmp_path)
    got = D.spill_host_only(t, t.cache_controller.mem_pool_host.anchor_entry.host_pool, 3, 1, set())
    assert got["released"] == 3 and got["leaves"] == 3
    assert [_in_tree(t, n) for n in nodes] == [True] * 5 + [False] * 3     # the tail of the chain
    assert all((root / (n.hash_value[0] + "_sfx.bin")).exists() for n in nodes[5:])
    assert not any((root / (n.hash_value[0] + "_sfx.bin")).exists() for n in nodes[:5])


def test_siblings_go_in_node_id_order(tmp_path, monkeypatch):
    """The PP ranks' trees are replicas: every rank releases the same nodes."""
    _env(monkeypatch, DUAL_P)
    from sglang.srt.weg2 import dual_arena_spill as D

    p, arena, root, t, nodes = _host_only_tree(tmp_path, chain=False)
    ids = sorted(n.id for n in nodes)
    got = D.spill_host_only(t, p, 3, 1, set())
    assert got["released"] == 3
    left = sorted(n.id for n in nodes if _in_tree(t, n))
    assert left == [i for i in ids][3:]


def test_a_page_without_an_l3_copy_is_never_released(tmp_path, monkeypatch):
    _env(monkeypatch, DUAL_P)
    p, arena, root, t, nodes = _host_only_tree(tmp_path, refuse=("a0_sfx",), chain=False)
    t._weg2_direct_claim(W._claimer(t))
    assert _in_tree(t, nodes[0]) and nodes[0].component_data[FULL].host_value is not None
    assert not (root / "a0_sfx.bin").exists()
    assert any(not _in_tree(t, n) for n in nodes[1:])


def test_pending_write_host_lock_aux_state_and_children_keep_their_node(tmp_path, monkeypatch):
    _env(monkeypatch, DUAL_P)
    from sglang.srt.weg2 import dual_arena_spill as D

    p, arena, root, t, nodes = _host_only_tree(tmp_path, chain=False)
    t.ongoing_write_through[nodes[1].id] = object()          # slots not COMPLETE yet
    nodes[2].component_data[FULL].host_lock_ref = 1          # a load-back reads these rows
    child = UnifiedTreeNode((FULL,))                         # nodes[3] has a child: not an H-leaf
    child.parent = nodes[3]
    nodes[3].children[("c",)] = child
    nodes[4].hash_value = None                               # page count cannot be checked
    got = D.spill_host_only(t, p, 100, 1, set())
    for k in (1, 2, 3, 4):
        assert _in_tree(t, nodes[k]) and nodes[k].component_data[FULL].host_value is not None, k
    assert got["leaves"] == 4
    assert [_in_tree(t, nodes[k]) for k in (0, 5, 6, 7)] == [False] * 4


def test_device_resident_nodes_are_spilled_first_and_the_claimers_chain_stays(tmp_path, monkeypatch):
    """The W3 round of device-resident nodes is unchanged and runs first: when it
    frees enough, no host-only leaf is touched."""
    _env(monkeypatch, DUAL_P)
    p, arena, root, t, nodes = _host_only_tree(tmp_path, chain=False)
    for n in nodes[:4]:                                      # four nodes still on the device
        n.component_data[FULL].value = torch.arange(1, dtype=torch.int64)
    m = W._claimer(t)
    pre = t._weg2_direct_claim(m)
    assert pre is not False, t.refused
    # the host-only ones: not needed for two slots once W3 gave the device ones back
    assert all(_in_tree(t, n) for n in nodes[4:])
    assert any(n.component_data[FULL].host_value is None for n in nodes[:4])
