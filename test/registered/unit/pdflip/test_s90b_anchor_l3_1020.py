"""S90b (NF rc16 bf9ca46f45, boot ..._1020_090554, P): interne Mamba-Anker wurden unter KV-Arena-Druck
tombstoned statt nach L3 geschrieben, und ein Follower ohne Zustand bei told ackte die flache Tiefe.

Live (P.log):
  09:11:44 PP1 ``#1421 BACKUP-REFUSED why=arena_claim node=58 tokens=4096 depth=35584 ... backuped=False``
           ``PDFLIP-ANCHOR-LOST at=evict n=1 node=58 depth=35584 backuped=False l3_present=False why=arena_claim``
           17 such lines on PP0/PP1/PP2 (depths 35584 51968 68352 71872 70720 87104), ``#1427 ARENA-DROP need=62
           freed=0`` (KV arena full), MAMBA-ARENA displaced=0 refused=0 (mamba arena NOT full).
  09:12:59 PP0 ``PDFLIP-CLAIM-SHORT rid=pdflip-1-23 asked=1079 kv=1075 claimed=252 ... by=mamba`` -- the store
           held the KV to 71872 but no anchor past 19200 (front ``L3-INDEX-PRESENCE depth=19200 kv_pages=1123``).
  09:13:10 PP2 ``#1400 OVER-UNRESUMABLE rid=pdflip-1-23 told=19200 own=47808 resumable_at_told=3072``
  09:13:16 PP0 ``PF TOLD-FLOOR rid=pdflip-1-23 told=19200 -> 3072 acks={1: 19200, 2: 3072}``
  -> pdflip-1-23/-25/-28 recomputed 16128/46336/88000 tokens (front PDFLIP-SERVED cached_tokens=3072).

Wurzel 1: ``pdflip_save_anchor_before_tombstone`` (K15) took the anchor-only path only for a KV-backed node
(``backuped`` / ``l3_present``, S25) and asked a device-KV node for a FULL backup; the KV arena's refusal
(``arena_claim``) ended the save -- the spill retry only follows a MAMBA refusal (K15 A1) -- and the anchor was
tombstoned although its own pool had room. Wurzel 2: ``_over_resumable_at_told`` only probed; a follower holding
the KV past told never read the anchor of told (COMPLETE in its arena slot: ``S25 ANCHOR-SECURED-HELD node=54
depth=19200``) and acked 3072.
"""
from __future__ import annotations

import logging
import os
import sys
import types

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

import pytest  # noqa: E402
import torch  # noqa: E402

from flliper.srt.environ import envs  # noqa: E402
from flliper.srt.managers import cache_controller as ccmod  # noqa: E402
from flliper.srt.mem_cache import unified_radix_cache as urc  # noqa: E402
from flliper.srt.mem_cache.hicache_storage import PoolName  # noqa: E402
from flliper.srt.mem_cache.radix_cache import RadixKey  # noqa: E402
from flliper.srt.mem_cache.unified_cache_components.mamba_component import MambaComponent  # noqa: E402
from flliper.srt.mem_cache.unified_cache_components.tree_component import ComponentType  # noqa: E402

UnifiedRadixCache = urc.UnifiedRadixCache

#: the 17 live tombstones (stage, node, depth) of boot _090554, P.log 09:11:44-09:13:03
LIVE_LOST = [
    ("PP1", 58, 35584), ("PP1", 62, 51968), ("PP1", 63, 68352), ("PP0", 49, 35584), ("PP0", 50, 51968),
    ("PP0", 63, 71872), ("PP0", 57, 68352), ("PP1", 70, 71872), ("PP2", 58, 35584), ("PP2", 62, 51968),
    ("PP2", 66, 68352), ("PP1", 137, 70720), ("PP1", 141, 87104), ("PP0", 128, 70720), ("PP0", 129, 87104),
    ("PP2", 137, 70720), ("PP2", 141, 87104),
]


@pytest.fixture(autouse=True)
def _env(monkeypatch):
    for k in ("FLLIPER_PDFLIP_EVICT_ANCHOR_SAVE", "FLLIPER_PDFLIP_ANCHOR_ONLY_BACKUP",
              "FLLIPER_PDFLIP_EVICT_ANCHOR_DEVICE_KV", "FLLIPER_PDFLIP_FOLLOWER_STATE_AT_TOLD"):
        monkeypatch.delenv(k, raising=False)
    # S90b review A1: the class-level log-rate counters advance per save; restored after every test so this file does
    # not push a later file's K15 / S90b log assertions past their n <= 32 limit (k15 test_save_backs_up_and_joins)
    for attr in ("_k15_saved_n", "_s90b_saved_n", "_s90b_follower_n"):
        monkeypatch.setattr(urc.UnifiedRadixCache, attr, 0, raising=False)


# ---------------------------------------------------------------------------
# Teil 1: the eviction save of a device-KV node (pdflip_save_anchor_before_tombstone)

def _node(nid=58, depth=35584, backuped=False, l3_present=False):
    cd = types.SimpleNamespace(value=object(), host_value=None)
    full = types.SimpleNamespace(value=torch.arange(4096), host_value=None)
    return types.SimpleNamespace(id=nid, depth=depth, backuped=backuped, l3_present=l3_present,
                                 pdflip_anchor_secured=False, _pdflip_end_anchor=False,
                                 write_through_pending_id=None, hash_value=[f"h{depth}"],
                                 component_data={ComponentType.MAMBA: cd, ComponentType.FULL: full})


class _SaveTree(types.SimpleNamespace):
    """The K15 save's collaborators; the KV arena refuses every FULL backup (live: ARENA-DROP freed=0)."""

    def __init__(self, full_refusal="arena_claim", mamba_room=True, held=None, pp_size=3):
        super().__init__()
        self.cache_controller = types.SimpleNamespace(write_policy="write_back")
        self.calls = []
        self.full_refusal, self.mamba_room, self.held = full_refusal, mamba_room, held
        self.pp_size = pp_size
        self._pdflip_sweep_last_refusal = None
        self._k15_last_refusal = None
        self.slots = {}

    def write_backup(self, node, write_back=False):
        self.calls.append("full_kv_backup")
        self._k15_last_refusal = self.full_refusal
        return 0

    def write_backup_anchor_only(self, node, kv_on_device=False):
        self.calls.append("anchor_only_dev" if kv_on_device else "anchor_only")
        if not self.mamba_room:
            self._k15_last_refusal = "anchor_only_claim"
            return 0
        self.slots[node.hash_value[-1]] = "claimed"
        self._pending = node
        return 1

    def writing_check(self, write_back=False):
        self.calls.append("join")
        node = getattr(self, "_pending", None)
        if node is not None:      # the joined ack: _pdflip_direct_complete (detached) -> secured
            self.slots[node.hash_value[-1]] = "complete"
            node.pdflip_anchor_secured = True
            self._pending = None

    def _pdflip_mamba_pool(self):
        held = self.held
        return types.SimpleNamespace(anchor_held=lambda h: held)

    def _pdflip_evict_spill_room(self, node, mp):
        return UnifiedRadixCache._pdflip_evict_spill_room(self, node, mp)

    def _s90b_anchor_held(self, node):
        return UnifiedRadixCache._s90b_anchor_held(self, node)

    def pdflip_node_depth(self, n):
        return n.depth


def _save(t, n):
    return UnifiedRadixCache.pdflip_save_anchor_before_tombstone(t, n)


def test_live_node_58_device_kv_anchor_goes_to_its_slot_instead_of_the_tombstone(caplog):
    """RED on rc16/rc21: full backup refused (arena_claim) -> PDFLIP-ANCHOR-LOST at=evict node=58 depth=35584.
    GREEN: the anchor alone into its mamba arena slot (detached), joined, COMPLETE -> saved, nothing lost."""
    t, n = _SaveTree(), _node()
    with caplog.at_level(logging.INFO):
        assert _save(t, n) is True
    assert t.calls == ["full_kv_backup", "anchor_only_dev", "join"]
    assert t.slots == {"h35584": "complete"}
    assert n.pdflip_anchor_secured is True
    assert n.component_data[ComponentType.MAMBA].host_value is None      # detached: no aux host on a device-KV node
    assert any("S90b EVICT-ANCHOR-DEVICE-KV" in m and "node=58" in m and "depth=35584" in m
               and "kv_refusal=arena_claim" in m for m in caplog.messages)
    assert not any("PDFLIP-ANCHOR-LOST" in m for m in caplog.messages)


def test_all_17_live_tombstones_are_saved(caplog):
    with caplog.at_level(logging.INFO):
        for _stage, nid, depth in LIVE_LOST:
            t, n = _SaveTree(), _node(nid, depth)
            assert _save(t, n) is True, (nid, depth)
            assert t.slots == {f"h{depth}": "complete"}
    assert not any("PDFLIP-ANCHOR-LOST" in m for m in caplog.messages)


@pytest.mark.parametrize("why", ["arena_claim", "host_floor", "host_evict_short:10+0<4096", "staging_ring"])
def test_every_kv_side_refusal_takes_the_anchor_alone(why):
    t, n = _SaveTree(full_refusal=why), _node()
    assert _save(t, n) is True and t.calls[:2] == ["full_kv_backup", "anchor_only_dev"]


@pytest.mark.parametrize("why", ["mamba_pin", "mamba_pool_unbound", "direct_no_hashes", "arena_unbound", None])
def test_a_refusal_that_is_not_the_kv_side_keeps_the_named_loss(why, caplog):
    """The pin budget / an unbound pool / missing hashes refuse the anchor-only write too: no second write."""
    t, n = _SaveTree(full_refusal=why), _node()
    with caplog.at_level(logging.WARNING):
        assert _save(t, n) is False
    assert t.calls == ["full_kv_backup"]
    assert any("PDFLIP-ANCHOR-LOST at=evict" in m for m in caplog.messages)


def test_an_ack_that_lost_its_slot_is_a_named_loss(caplog):
    """Issued is not secured: an A2 late ack (the slot taken over, ``keep < mrows``) leaves the mark False --
    the save reports the loss instead of trusting the issue."""
    t, n = _SaveTree(), _node()

    def _lost_ack(write_back=False):
        t.calls.append("join")
        t._pending = None                      # completed, but no reference kept -> not secured

    t.writing_check = _lost_ack
    with caplog.at_level(logging.WARNING):
        assert _save(t, n) is False
    assert t.calls == ["full_kv_backup", "anchor_only_dev", "join"]
    assert any("PDFLIP-ANCHOR-LOST at=evict" in m for m in caplog.messages)


def test_full_mamba_arena_on_p_is_still_a_named_loss(caplog):
    """On P (pp>1) the K15 spill is skipped (shared slots); a full mamba arena stays a NAMED loss."""
    t, n = _SaveTree(mamba_room=False), _node()
    with caplog.at_level(logging.INFO):
        assert _save(t, n) is False
    assert t.calls == ["full_kv_backup", "anchor_only_dev"]
    assert any("PDFLIP-ANCHOR-LOST at=evict" in m and "why=anchor_only_claim" in m for m in caplog.messages)


def test_an_anchor_demonstrably_held_is_not_written_again():
    t, n = _SaveTree(held=True), _node()
    n.pdflip_anchor_secured = True
    assert _save(t, n) is True
    assert t.calls == ["full_kv_backup"]            # no second slot claim, no join


@pytest.mark.parametrize("held", [False, None])
def test_the_mark_alone_is_no_proof_for_a_device_kv_node(held):
    t, n = _SaveTree(held=held), _node()
    n.pdflip_anchor_secured = True
    assert _save(t, n) is True
    assert t.calls == ["full_kv_backup", "anchor_only_dev", "join"]


def test_switch_off_is_the_old_tombstone(monkeypatch, caplog):
    monkeypatch.setenv("FLLIPER_PDFLIP_EVICT_ANCHOR_DEVICE_KV", "0")
    t, n = _SaveTree(), _node()
    with caplog.at_level(logging.WARNING):
        assert _save(t, n) is False
    assert t.calls == ["full_kv_backup"]
    assert any("PDFLIP-ANCHOR-LOST at=evict" in m and "why=arena_claim" in m for m in caplog.messages)


def test_kv_backed_nodes_unchanged():
    for kw in ({"backuped": True}, {"l3_present": True}):
        t, n = _SaveTree(), _node(**kw)
        _save(t, n)
        assert t.calls[0] == "anchor_only", kw


# ---------------------------------------------------------------------------
# Teil 1b: the anchor-only write itself (real candidate / write_backup_anchor_only)

class _Ctl:
    write_policy = "write_back"

    def __init__(self):
        self.writes = []

    def write(self, dev, node_id=None, extra_pools=None, host_indices=None):
        self.writes.append((int(dev.numel()), int(host_indices.numel()), [x.name for x in extra_pools]))
        return host_indices


def _aotree(monkeypatch):
    t = object.__new__(UnifiedRadixCache)
    t.root_node = types.SimpleNamespace(id=0)
    t.tree_components = (ComponentType.FULL, ComponentType.MAMBA)
    t.cache_controller = _Ctl()
    t._pdflip_direct_mamba_rows = {}
    t._pdflip_anchor_only_ids = set()
    t._pdflip_anchor_only_detached = set()
    t.ongoing_write_through = {}
    mp = types.SimpleNamespace(aborted=[], abort_write=lambda r: mp.aborted.append(r))
    t._pdflip_mamba_pool = lambda: mp
    t._pdflip_mamba_claim = lambda node, mp_, h: torch.tensor([4 + 7])
    t._mamba_write_through_pin_admissible = lambda node, write_back=False: True
    committed = []

    class _Comp:
        def build_hicache_transfers(self, node, phase):
            return [types.SimpleNamespace(name=PoolName.MAMBA, host_indices=None, device_indices=torch.tensor([3]))]

        def commit_hicache_transfer(self, node, phase, transfers=()):
            committed.append(node.id)

    t.components = {ComponentType.MAMBA: _Comp()}
    t.inc_lock_ref = lambda node: types.SimpleNamespace(to_dec_params=lambda: None)
    t.pdflip_node_depth = lambda n: n.depth
    monkeypatch.setattr(UnifiedRadixCache, "_pdflip_anchor_secured_holds", lambda self, node: False)
    return t, committed


def _real_node():
    n = _node()
    n.evicted = False
    n.component_data = [types.SimpleNamespace(value=torch.arange(4096), host_value=None),
                        types.SimpleNamespace(value=None, host_value=None),
                        types.SimpleNamespace(value=torch.tensor([3]), host_value=None)]
    assert int(ComponentType.FULL) == 0 and int(ComponentType.MAMBA) == 2
    return n


def test_candidate_admits_a_device_kv_node_only_for_the_eviction_save(monkeypatch):
    t, _ = _aotree(monkeypatch)
    n = _real_node()
    assert t._pdflip_anchor_only_candidate(n) is False               # the flush sweep: unchanged
    assert t._pdflip_anchor_only_candidate(n, kv_on_device=True) is True
    n.component_data[0].value = None                                    # no device KV -> nothing to file it under
    assert t._pdflip_anchor_only_candidate(n, kv_on_device=True) is False


def test_device_kv_anchor_only_write_carries_no_kv_and_stays_detached(monkeypatch):
    t, committed = _aotree(monkeypatch)
    n = _real_node()
    assert t.write_backup_anchor_only(n, kv_on_device=True) == 1
    assert t.cache_controller.writes == [(0, 0, [PoolName.MAMBA])]   # no KV row claimed or copied
    assert committed == []                                             # not in the tree: Full has no host copy
    assert n.id in t._pdflip_anchor_only_detached and n.id in t._pdflip_anchor_only_ids
    assert n.write_through_pending_id == n.id                          # tracked + joined like every write-through


# ---------------------------------------------------------------------------
# Teil 2: the follower reads the state AT told (pdflip_follower_state_at_told)

PAGE = 64
TOKENS = list(range(6 * PAGE))
D_TOLD, D_DEV = 3 * PAGE, 5 * PAGE      # scaled: told 19200, device KV 47808 (live PP2)


class _Lru:
    def __init__(self):
        self.nodes = []

    def in_list(self, n):
        return n in self.nodes

    def insert_mru(self, n):
        self.nodes.append(n)

    def remove_node(self, n):
        self.nodes.remove(n)


class _HostPool:
    """The mamba host pool: the read row resolves to the anchor's arena slot; per-layer copies to the device."""

    def __init__(self, state):
        self.state, self.freed, self.loads = state, [], []

    def alloc_read(self, n):
        return torch.tensor([10_000], dtype=torch.int64)      # a placeholder until the read resolves it

    def is_placeholder(self, i):
        return int(i) >= 10_000

    def free(self, idx):
        self.freed.append(int(idx[0]))

    def load_to_device_per_layer(self, device_pool, host_indices, device_indices, layer_id, io_backend="direct"):
        self.loads.append(layer_id)
        if layer_id == 0:   # the arena pool's all-layers state load remembers its key (arena_mamba_pool)
            key = (id(host_indices), id(device_indices), int(host_indices.numel()), int(device_indices.numel()))
            self._state_loaded_key = key
            self._state_loaded_keys = getattr(self, "_state_loaded_keys", []) + [key]
        slot = int(host_indices[0])
        device_pool.temporal[layer_id][int(device_indices[0])] = self.state[slot][layer_id]


class _Backend:
    def __init__(self, ok=True, slot=5):
        self.ok, self.slot, self.calls = ok, slot, []

    def batch_get_v2(self, transfers):
        t = transfers[0]
        self.calls.append((t.name, list(t.keys)))
        if self.ok:
            t.host_indices[0] = self.slot
        return {PoolName.MAMBA: [self.ok]}


def _fnode(key, parent, tag, dev_kv=True, full_host=False):
    n = types.SimpleNamespace(
        id=tag, key=key, parent=parent, children={},
        component_data={ComponentType.MAMBA: types.SimpleNamespace(value=None, host_value=None),
                        ComponentType.FULL: types.SimpleNamespace(
                            value=torch.arange(len(key)) if dev_kv else None,
                            host_value=torch.tensor([9]) if full_host else None)},
        hash_value=[f"{tag}-last"])
    n.get_last_hash_value = lambda: n.hash_value[-1]
    return n


def _ftree(monkeypatch, backend, full_host=False, slots_free=(7,), layers=2):
    t = object.__new__(UnifiedRadixCache)
    t.page_size = PAGE
    t.is_eagle = False
    root = types.SimpleNamespace(children={}, parent=None)
    t.root_node = root
    k1 = RadixKey(token_ids=TOKENS[:D_TOLD])
    n1 = _fnode(k1, root, "n19200", full_host=full_host)
    root.children[k1.child_key(PAGE)] = n1
    k2 = RadixKey(token_ids=TOKENS[D_TOLD:D_DEV])
    n2 = _fnode(k2, n1, "n47808")
    n1.children[k2.child_key(PAGE)] = n2
    # the store's anchor of 19200 (COMPLETE arena slot 5), one temporal row per local layer
    state = {5: [torch.full((4,), 19.2 + l) for l in range(layers)]}
    hp = _HostPool(state)
    dev = types.SimpleNamespace(temporal=[torch.zeros(16, 4) for _ in range(layers)])
    free = list(slots_free)
    alloc = types.SimpleNamespace(alloc=lambda k: (torch.tensor([free.pop(0)]) if free else None),
                                  freed=[], free=lambda s: alloc.freed.append(int(s[0])))
    t.req_to_token_pool = types.SimpleNamespace(mamba_allocator=alloc)
    t.cache = t
    comp = object.__new__(MambaComponent)
    comp.cache = t
    comp.component_type = ComponentType.MAMBA
    comp._mamba_pool_host = hp
    comp._alloc_mamba_slot = lambda: t.req_to_token_pool.mamba_allocator.alloc(1)
    t.components = {ComponentType.MAMBA: comp}
    entry = types.SimpleNamespace(host_pool=hp, device_pool=dev,
                                  layer_mapping={40 + l: l for l in range(layers)})
    t.cache_controller = types.SimpleNamespace(storage_backend=backend, io_backend="direct",
                                               mem_pool_host=types.SimpleNamespace(
                                                   entry_map={PoolName.MAMBA: entry}))
    t.host_lru_lists = {ComponentType.MAMBA: _Lru()}
    t.lru_lists = {ComponentType.MAMBA: _Lru()}
    t.component_evictable_size_ = {ComponentType.MAMBA: 0}
    t._update_evictable_leaf_sets = lambda node: None
    monkeypatch.setattr(ccmod, "rank_holds_gdn_state", lambda cc: True)
    return t, n1, n2, hp, dev


def _freq():
    return types.SimpleNamespace(rid="pdflip-1-23", full_untruncated_fill_ids=TOKENS, extra_key=None)


def test_device_kv_follower_attaches_the_told_anchor_into_a_device_slot(monkeypatch, caplog):
    """RED on rc16/rc21 (no such read): PP2 acked 3072. GREEN: the anchor of 19200 is read from its arena slot and
    copied into a device mamba slot of the node at told -- every local layer, the read row released."""
    be = _Backend()
    t, n1, n2, hp, dev = _ftree(monkeypatch, be)
    with caplog.at_level(logging.INFO):
        assert t.pdflip_follower_state_at_told(_freq(), D_TOLD) == D_TOLD
    assert be.calls == [(PoolName.MAMBA, ["n19200-last"])]
    cd = n1.component_data[ComponentType.MAMBA]
    assert cd.value is not None and int(cd.value[0]) == 7
    assert cd.host_value is None                                   # AUX-HOST-LAW: no host state on device-only KV
    assert hp.loads == [0, 1]
    for l in range(2):
        assert torch.equal(dev.temporal[l][7], torch.full((4,), 19.2 + l))
    assert hp.freed == [5]                                         # the slot reference goes after the copy
    assert t.lru_lists[ComponentType.MAMBA].nodes == [n1]
    assert t.component_evictable_size_[ComponentType.MAMBA] == 1
    assert n2.component_data[ComponentType.MAMBA].value is None
    assert any("S90b FOLLOWER-STATE-AT-TOLD" in m and "how=device_attach" in m for m in caplog.messages)


def test_the_attach_leaves_no_state_key_a_reused_tensor_id_could_match(monkeypatch):
    be = _Backend()
    t, n1, _, hp, _ = _ftree(monkeypatch, be)
    hp._state_loaded_keys = [("other", "load", 1, 1)]
    assert t.pdflip_follower_state_at_told(_freq(), D_TOLD) == D_TOLD
    assert hp._state_loaded_keys == [("other", "load", 1, 1)] and hp._state_loaded_key is None


def test_a_slot_eviction_that_took_the_node_kv_attaches_nothing(monkeypatch):
    be = _Backend()
    t, n1, _, hp, _ = _ftree(monkeypatch, be)
    comp = t.components[ComponentType.MAMBA]
    real = comp._alloc_mamba_slot

    def _alloc_evicting():
        n1.component_data[ComponentType.FULL].value = None      # the slot's eviction walk took the device KV
        return real()

    comp._alloc_mamba_slot = _alloc_evicting
    assert t.pdflip_follower_state_at_told(_freq(), D_TOLD) == 0
    assert n1.component_data[ComponentType.MAMBA].value is None
    assert t.req_to_token_pool.mamba_allocator.freed == [7] and hp.freed == [5] and hp.loads == []


def test_host_kv_follower_uses_the_e1_host_attach(monkeypatch):
    be = _Backend()
    t, n1, _, hp, _ = _ftree(monkeypatch, be, full_host=True)
    seen = []
    monkeypatch.setattr(UnifiedRadixCache, "pdflip_anchor_only_read_at",
                        lambda self, req, depth, site="tf_floor": seen.append((depth, site)) or depth)
    assert t.pdflip_follower_state_at_told(_freq(), D_TOLD) == D_TOLD
    assert seen == [(D_TOLD, "follower_told")] and hp.loads == []


def test_store_miss_attaches_nothing_and_releases_the_row(monkeypatch):
    be = _Backend(ok=False)
    t, n1, _, hp, _ = _ftree(monkeypatch, be)
    req = _freq()
    assert t.pdflip_follower_state_at_told(req, D_TOLD) == 0
    assert t.pdflip_follower_state_at_told(req, D_TOLD) == 0       # one read per depth
    assert len(be.calls) == 1 and hp.freed == [10_000]
    assert n1.component_data[ComponentType.MAMBA].value is None
    assert t.req_to_token_pool.mamba_allocator.freed == []


def test_no_device_slot_attaches_nothing(monkeypatch):
    be = _Backend()
    t, n1, _, hp, _ = _ftree(monkeypatch, be, slots_free=())
    assert t.pdflip_follower_state_at_told(_freq(), D_TOLD) == 0
    assert hp.freed == [5] and hp.loads == []
    assert n1.component_data[ComponentType.MAMBA].value is None


class _CommitRaises(Exception):
    pass


def _raising_commit(monkeypatch, t, set_value_first):
    comp = t.components[ComponentType.MAMBA]
    real = comp.commit_hicache_transfer

    def _commit(node, phase, transfers=None):
        if set_value_first:
            real(node, phase, transfers=transfers)     # the tree took the slot, then the bookkeeping raised
        raise _CommitRaises("commit failed")

    monkeypatch.setattr(comp, "commit_hicache_transfer", _commit)


def test_a2_a_commit_that_raises_before_the_tree_took_the_slot_frees_slot_and_row_once(monkeypatch):
    """S90b review A2: the LOAD_BACK commit runs inside the guard. It raised before setting the node's value: the
    device slot and the read row are released exactly once, nothing is attached."""
    be = _Backend()
    t, n1, _, hp, _ = _ftree(monkeypatch, be)
    _raising_commit(monkeypatch, t, set_value_first=False)
    assert t.pdflip_follower_state_at_told(_freq(), D_TOLD) == 0
    assert t.req_to_token_pool.mamba_allocator.freed == [7]
    assert hp.freed == [5]
    assert n1.component_data[ComponentType.MAMBA].value is None


def test_a2_a_commit_that_raises_after_the_tree_took_the_slot_does_not_free_it(monkeypatch):
    """S90b review A2: the commit set the node's value (the tree holds the slot) and then raised -- freeing the slot
    here would leave the tree with a freed slot (a later double free by its eviction)."""
    be = _Backend()
    t, n1, _, hp, _ = _ftree(monkeypatch, be)
    _raising_commit(monkeypatch, t, set_value_first=True)
    assert t.pdflip_follower_state_at_told(_freq(), D_TOLD) == 0
    assert t.req_to_token_pool.mamba_allocator.freed == []
    assert int(n1.component_data[ComponentType.MAMBA].value[0]) == 7
    assert hp.freed == [5]


def test_a2_a_raising_free_on_node_changed_is_not_followed_by_a_second_free(monkeypatch):
    """S90b review A2: on the node-changed exit ``free(dst)`` raised -- the handler must not free the slot again;
    the read row is still released once."""
    be = _Backend()
    t, n1, _, hp, _ = _ftree(monkeypatch, be)
    comp = t.components[ComponentType.MAMBA]
    real = comp._alloc_mamba_slot
    alloc = t.req_to_token_pool.mamba_allocator
    calls = []

    def _alloc_evicting():
        n1.component_data[ComponentType.FULL].value = None
        return real()

    def _free_raises(s):
        calls.append(int(s[0]))
        raise RuntimeError("free failed")

    comp._alloc_mamba_slot = _alloc_evicting
    alloc.free = _free_raises
    assert t.pdflip_follower_state_at_told(_freq(), D_TOLD) == 0
    assert calls == [7]
    assert hp.freed == [5]
    assert n1.component_data[ComponentType.MAMBA].value is None


def test_guards(monkeypatch):
    be = _Backend()
    t, n1, _, _, _ = _ftree(monkeypatch, be)
    assert t.pdflip_follower_state_at_told(_freq(), D_TOLD + 1) == 0        # not a page boundary
    assert t.pdflip_follower_state_at_told(_freq(), 2 * PAGE) == 0          # no node ends there
    n1.component_data[ComponentType.MAMBA].value = torch.tensor([1])
    assert t.pdflip_follower_state_at_told(_freq(), D_TOLD) == 0            # the node has a state
    n1.component_data[ComponentType.MAMBA].value = None
    with envs.FLLIPER_PDFLIP_FOLLOWER_STATE_AT_TOLD.override(False):
        assert t.pdflip_follower_state_at_told(_freq(), D_TOLD) == 0
    assert be.calls == []


# ---------------------------------------------------------------------------
# Teil 2b: the settle -- PP2 is satisfied at told instead of acking 3072 (live numbers)

from flliper.srt.managers import pdflip_store_told as st  # noqa: E402
from flliper.srt.managers import pdflip_told_fallback as fb  # noqa: E402

TOLD, OWN, RES = 19200, 47808, 3072


class _Tree:
    def __init__(self, attaches):
        self.attaches, self.asked = attaches, []

    def pdflip_follower_state_at_told(self, req, told):
        self.asked.append(int(told))
        if self.attaches:
            self.state_at = int(told)
            return int(told)
        return 0


def _sched(monkeypatch, attaches):
    tree = _Tree(attaches)
    s = types.SimpleNamespace(tree_cache=tree, ps=types.SimpleNamespace(pp_rank=2))
    monkeypatch.setattr(st._tf, "enabled", lambda: True)
    monkeypatch.setattr(st._fb, "env_on", lambda: True)
    monkeypatch.delenv("FLLIPER_PDFLIP_DUAL_LAYOUT", raising=False)

    def probe(scheduler, req, depth):
        # PP2's tree: KV 0..47808, recurrent state at 3072 -- and at 19200 once the told anchor is attached
        if getattr(scheduler.tree_cache, "state_at", None) == int(depth):
            return int(depth)
        return min(int(depth), RES)

    monkeypatch.setattr(st._tf, "rank_resumable", probe)
    return s, tree


def test_live_pp2_over_read_is_satisfied_at_told_after_reading_the_anchor(monkeypatch):
    """RED on rc16/rc21: OVER-UNRESUMABLE told=19200 own=47808 resumable_at_told=3072 -> PF floor 3072."""
    s, tree = _sched(monkeypatch, attaches=True)
    req = types.SimpleNamespace(rid="pdflip-1-23")
    assert st._over_resumable_at_told(s, req, TOLD, OWN) is True
    assert tree.asked == [TOLD]
    assert getattr(s, "_pdflip_over_unresumable_n", 0) == 0


def test_without_an_anchor_in_the_store_the_old_unresumable_ack_stands(monkeypatch, caplog):
    s, tree = _sched(monkeypatch, attaches=False)
    req = types.SimpleNamespace(rid="pdflip-1-23")
    with caplog.at_level(logging.WARNING):
        assert st._over_resumable_at_told(s, req, TOLD, OWN) is False
    assert any("OVER-UNRESUMABLE" in m and "resumable_at_told=3072" in m for m in caplog.messages)


def test_satisfied_ack_reads_the_anchor_before_acking_less(monkeypatch):
    """The other way into ``satisfied`` (local prefix covering told): the ack is told, not 3072."""
    s, tree = _sched(monkeypatch, attaches=True)
    monkeypatch.setattr(fb, "env_on", lambda: True)
    monkeypatch.setattr(fb, "_room_own", lambda scheduler, req, rid, own: own)
    import flliper.srt.managers.pdflip_told_fidelity as tfmod
    monkeypatch.setattr(tfmod, "enabled", lambda: True)
    monkeypatch.setattr(tfmod, "rank_resumable", st._tf.rank_resumable)
    req = types.SimpleNamespace(rid="pdflip-1-23")
    assert fb._satisfied_resumable(s, req, "pdflip-1-23", TOLD) == TOLD
    s2, _ = _sched(monkeypatch, attaches=False)
    monkeypatch.setattr(tfmod, "rank_resumable", st._tf.rank_resumable)
    assert fb._satisfied_resumable(s2, req, "pdflip-1-23", TOLD) == RES


# ---------------------------------------------------------------------------
# Teil 2c: the PF ring (three stages, PP0 decides) -- one uniform plan at told, no floor

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "managers"))


def test_pf_ring_one_plan_at_told_on_every_rank(monkeypatch):
    """The ring of test_pdflip_over_unresumable_pf_968_1003 with PP2's tree able to read the told anchor: every
    stage gets the SAME plan at told (PP0's one verdict, SOLL R11) -- no PF fallback, no OVER-UNRESUMABLE."""
    import _told_ring_pf as R

    m = st
    RID, RTOLD, ROVER = "aaaa-told", 100_000, 120_000
    monkeypatch.delenv("FLLIPER_PDFLIP_TOLD_FIDELITY", raising=False)
    for k in list(os.environ):
        if k.startswith("FLLIPER_PDFLIP_TOLD") or k in ("FLLIPER_PDFLIP_P_TWIN_DEFER", m.ENV_FOLLOWER_EARLY_READ):
            monkeypatch.delenv(k, raising=False)
    monkeypatch.setenv(fb.ENV_FALLBACK, "1")
    monkeypatch.delenv("FLLIPER_PDFLIP_DUAL_SHARE", raising=False)
    monkeypatch.setenv(m.ENV_FOLLOWER_EARLY_READ, "1")
    monkeypatch.setattr(m, "_absolute_armed", lambda: True)
    ring = R.Ring(m, monkeypatch, {RID: RTOLD}, {r: {RID: (0.8, 0.6, 0.6)[r]} for r in range(3)})
    over = ring.stages[2]
    over.prompts = {RID: ROVER}
    attached = {}
    over.tree_cache.pdflip_follower_state_at_told = lambda req, told: attached.setdefault(int(told), int(told))

    def probe(scheduler, req, depth):
        if scheduler is not over:
            return None
        if int(depth) in attached or int(depth) >= ROVER:
            return min(int(depth), ROVER)
        return 0

    monkeypatch.setattr(m._tf, "rank_resumable", probe)
    ring.arrive(RID)
    ring.run(100)
    plans = ring.plans(RID)
    assert plans[0] == plans[1] == plans[2] and len(plans[0]) == 1, plans
    assert plans[0][0][2] == RTOLD, plans
    assert getattr(ring.stages[0], "_pf_fallback_n", 0) == 0
    assert getattr(over, "_pdflip_over_unresumable_n", 0) == 0
    assert attached == {RTOLD: RTOLD}
