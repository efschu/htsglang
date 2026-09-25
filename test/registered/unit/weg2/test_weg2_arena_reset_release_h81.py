"""H81 (fnNV4f2, NVFP4 f2 on d93a17316b, 25.09. 05:08-05:09Z): the flip's tree
reset gives the arena references back, a claim makes room without disk I/O,
and group P holds the phase's END anchors across D's phase.

THE METAL (burst 8 x 4.2k, 16-slot mamba arena, P = PP3): every anchor P and D
ever wrote kept its reader references -- the flush before every sleep dropped
the tree and called `mem_pool_host.clear()`, which resets the staging
bookkeeping only (`UnifiedRadixCache._reset_full`). At the burst the arena was
pinned: ARENA-EVICT found nothing, "#1427 ARENA-CLAIM REFUSED statuses=[4]",
the END anchors of weg2-8-12/-14/-15/-16/-17 were refused ("MAMBA-ARENA ...
end_anchor=refused", RETAIN-PUBLISH stopped=mamba_full), D found the KV pages
but no mamba anchor ("#1028B FETCH CAP kv=64 claimed=0 ... mamba: (0, -1)",
"#1035c CAPPED by=mamba"), W50 -> requeue -> P prefilled into the same pinned
arena -> W50 -> W53 -> 413. INT4 x176/x177 ran the same leak with 32 slots,
which its probe set (18 P anchors) never filled. The 27B line fixed the leak
on 24.09. (479f6eccb0 + c255e10ddb); it never reached the NF line.

Hermetic, CPU: ONE real C arena (arena.c, gcc) on a temp file, three P rank
pools and three D rank pools (real ArenaMambaPoolHost, each rank its own
extent of the blob), real UnifiedTreeNode chains, the real H19 claim path
(`_weg2_mamba_claim`), the real `_reset_full` and the real wake hook. The
claim's disk round is a stub that frees unreferenced slots WITHOUT disk I/O
and counts its calls (the shipped tree calls it; the fixed one never does).
"""
from __future__ import annotations

import inspect
import os
import shutil
import threading
import types

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

import pytest  # noqa: E402
import torch  # noqa: E402

from sglang.srt.environ import envs  # noqa: E402
from sglang.srt.mem_cache import unified_radix_cache as u  # noqa: E402
from sglang.srt.mem_cache.pool_host.arena_mamba_pool import ArenaMambaPoolHost  # noqa: E402
from sglang.srt.mem_cache.pool_host.arena_pool import ArenaMHAHostPool  # noqa: E402
from sglang.srt.mem_cache.storage.file.hicache_arena import ShmArena  # noqa: E402
from sglang.srt.mem_cache.unified_cache_components.tree_component import ComponentType  # noqa: E402
from sglang.srt.mem_cache.unified_radix_cache import UnifiedRadixCache, UnifiedTreeNode  # noqa: E402
from sglang.srt.weg2 import mamba_arena_displace as mad  # noqa: E402

pytestmark = pytest.mark.skipif(shutil.which("gcc") is None, reason="needs gcc (arena.c)")

RANKS = 3
EXT = 32
SB = RANKS * EXT            # one mamba blob = three rank extents
STAGING = 2
TC = (ComponentType.FULL, ComponentType.MAMBA)
M = ComponentType.MAMBA
HOLD_ENV = "SGLANG_WEG2_ENABLE_MAMBA_CARRIER_HOLD"


class _Backend:
    """Stem naming, and the shipped claim's evict round WITHOUT the disk:
    only unreferenced complete slots leave (a leaked reference pins its
    slot). `disk_rounds` counts the calls -- the fixed claim never makes one."""

    def __init__(self):
        self.disk_rounds = 0

    def _get_suffixed_key(self, k):
        return f"{k}.sfx"

    def _log_key(self, pool, k):
        return f"{k}.mamba"

    def _arena_evict_to_disk(self, arena, want):
        self.disk_rounds += 1
        cands = [c[0] for c in arena.evict_candidates(int(want))]
        if cands:
            arena.free_slots(cands)
        return len(cands)


def _stem(h):
    return f"{h}.mamba.sfx"   # the pool's stem: backend._get_suffixed_key(backend._log_key(MAMBA, h))


def _pool(arena, rank):
    p = object.__new__(ArenaMambaPoolHost)
    p.size = STAGING
    p._arena_init_fields()
    p.device = "cpu"
    p.lock = threading.RLock()
    p.page_size = 1
    p.mem_state = torch.zeros((STAGING,), dtype=torch.uint8)
    p.free_slots = torch.arange(STAGING, dtype=torch.int64)
    p.arena = arena
    p.arena_slots = arena.slots
    p.id_space = STAGING + arena.slots
    p._backend = _Backend()
    p._page_bytes = SB
    p._own_extents = [(rank * EXT, EXT)]
    return p


class _HostGroup:
    """The controller's host pool group at the flush: `clear()` every pool --
    MambaPoolHost.clear resets the STAGING bookkeeping only (the shipped
    reset's whole give-back)."""

    arena_read = True

    def __init__(self, pools):
        self.pools = pools

    def clear(self):
        for p in self.pools:
            p.clear()


class _Rank:
    """One rank of group P or D: its pool on the shared arena and its tree,
    with every attribute the real `_reset_full` and `_weg2_mamba_claim` use."""

    def __init__(self, arena, rank, group):
        self.group = group
        self.mp = _pool(arena, rank)
        comp = types.SimpleNamespace(component_type=M, _mamba_pool_host=self.mp)
        c = object.__new__(UnifiedRadixCache)
        c.tree_components = TC
        c.root_node = UnifiedTreeNode(TC)
        c.components = {M: comp}
        c._components_tuple = (comp,)
        c._weg2_rid_anchor_cfg = -1                # auto: max(2, slots // 4), 16 -> 4 (the arm)
        c._weg2_anchor_ledger = mad.RidAnchorLedger()
        c._weg2_direct_mamba_rows = {}
        c._evict_component_and_detach_lru = self._evict_host
        c._update_evictable_leaf_sets = lambda n: None
        c.session = types.SimpleNamespace(slots={})
        c.device = torch.device("cpu")
        c.enable_kv_cache_events = False
        c.cache_controller = types.SimpleNamespace(
            reset=lambda: None, mem_pool_host=_HostGroup([self.mp]), enable_storage=True)
        self.cache = c
        self.tips = {}

    def _evict_host(self, node, comp, target=None, tracker=None):
        """The mamba component's HOST eviction: free the rank's host value."""
        cd = node.component_data[M]
        self.mp.free(cd.host_value)
        cd.host_value = None
        return 0, 1

    def node(self, rid, end_anchor=False, tokens=64):
        parent = self.tips.get(rid, self.cache.root_node)
        n = UnifiedTreeNode(TC)
        n.key = [0] * tokens
        n.parent = parent
        parent.children[len(parent.children)] = n
        n._weg2_end_anchor = end_anchor
        if rid is not None:
            self.cache._weg2_tag_anchor(n, rid)
            self.tips[rid] = n
        return n

    def reset(self, monkeypatch):
        monkeypatch.setenv("SGLANG_WEG2_GROUP", self.group)
        self.cache.reset()               # the REAL _reset_full
        self.tips = {}


class _Rig:
    """Group P (PP3) and group D (TP3) on one shared mamba arena."""

    def __init__(self, tmp_path, slots, monkeypatch):
        self.arena = ShmArena(str(tmp_path / f"arena-mamba-{slots}.bin"), SB, slots)
        self.P = [_Rank(self.arena, r, "P") for r in range(RANKS)]
        self.D = [_Rank(self.arena, r, "D") for r in range(RANKS)]
        self.mp = monkeypatch
        # the #1481 mark is armed on P (launcher: SGLANG_WEG2_END_ANCHOR=1)
        monkeypatch.setattr(u, "_WEG2_END_ANCHOR", True)

    # -- group P ---------------------------------------------------------------
    def p_prefill(self, rid, hashes, end=True):
        """Every rank publishes each anchor of the chain in turn (a chunk's
        node, its three stages); the last one is the request's END anchor
        (#1481 mark, set at the finish). Returns the hashes refused on ANY
        rank (a blob a rank could not write is never COMPLETE)."""
        refused = []
        for i, h in enumerate(hashes):
            is_end = end and i == len(hashes) - 1
            ok = True
            for r in self.P:
                n = r.node(rid, end_anchor=is_end)
                rows = r.cache._weg2_mamba_claim(n, r.mp, h)
                if rows is None:
                    ok = False
                    continue
                n.component_data[M].host_value = rows
                r.mp.complete_write(rows)
            if not ok:
                refused.append(h)
        return refused

    def p_sleep(self):
        """The flush of the sleeping group: quiesce + release_memory_occupation
        (the second reset sees an empty tree)."""
        for r in self.P:
            r.reset(self.mp)
            r.reset(self.mp)

    def p_wake(self):
        """The resume handler after the pools were restored (weight_updater's
        wake invariant block): the carrier hold goes back -- the hook is
        H81's, the shipped tree has none."""
        from sglang.srt.managers.scheduler_components.weight_updater import (
            SchedulerWeightUpdaterManager as W,
        )

        hook = getattr(W, "_weg2_release_carrier_hold_at_wake", None)
        if hook is None:
            return
        for r in self.P:
            hook(types.SimpleNamespace(scheduler=types.SimpleNamespace(tree_cache=r.cache)))

    # -- group D ---------------------------------------------------------------
    def d_read(self, h):
        """D's prefetch of a hand-over anchor: COMPLETE -> a reader reference
        per D rank and a node of D's tree holding it (arena_resolve_reads)."""
        slot, state = self.arena.find_slots([_stem(h)])[0]
        if slot < 0 or state != 2:
            return False
        for r in self.D:
            assert self.arena.ref_slots([slot], +1) == 1
            n = r.node(None)
            n.component_data[M].host_value = torch.tensor([STAGING + slot], dtype=torch.int64)
        return True

    def d_publish(self, h):
        """D's own anchor (its flush publish of a decode extension): an
        untagged node, the plain claim."""
        ok = True
        for r in self.D:
            n = r.node(None)
            rows = r.cache._weg2_mamba_claim(n, r.mp, h)
            if rows is None:
                ok = False
                continue
            n.component_data[M].host_value = rows
            r.mp.complete_write(rows)
        return ok

    def d_sleep(self):
        for r in self.D:
            r.reset(self.mp)
            r.reset(self.mp)

    def flip_through_d(self, handed, d_writes=()):
        """P sleeps, D reads the hand-over anchors and writes its own, D
        sleeps, P wakes. Returns the hand-overs D did NOT find."""
        self.p_sleep()
        missing = [h for h in handed if not self.d_read(h)]
        for h in d_writes:
            self.d_publish(h)
        self.d_sleep()
        self.p_wake()
        return missing

    def complete(self, h):
        slot, state = self.arena.find_slots([_stem(h)])[0]
        return slot >= 0 and state == 2

    def refs(self, h):
        slot, _ = self.arena.find_slots([_stem(h)])[0]
        return None if slot < 0 else int(_hdr_refs(self.arena)[slot])


def _hdr_refs(arena):
    import numpy as np

    out = (np.ctypeslib.ctypes.c_int64 * 6)()
    arena._lib.arena_layout(arena.slots, arena.slot_bytes, out)
    hb, hoff = int(out[0]), int(out[3])
    u32 = np.frombuffer(arena._mm, dtype=np.uint32, count=arena.slots * hb // 4, offset=hoff)
    return u32.reshape(arena.slots, hb // 4)[:, 1].copy()


def _chain(rid, n):
    return [f"{rid}/c{i}" for i in range(n)]


# ----------------------------------------------------------------------------------------
def test_fnNV4f2_burst_every_end_anchor_reaches_the_arena(tmp_path, monkeypatch):
    """The metal shape, scaled to its own numbers: 16 slots, cap 4, the probe
    set before the burst (97k needle with six 16k chunk anchors, three single
    anchor prompts, D's flush anchors), then 8 x (inner + end anchor).

    RED on d93a17316b: every reference of the earlier phases leaked at the
    resets, the burst's end anchors are refused (weg2-8-12.. on metal) and D
    cannot find them -- the W50 of the boot. GREEN: all eight hand over."""
    rig = _Rig(tmp_path, 16, monkeypatch)
    assert rig.p_prefill("weg2-0-4", _chain("weg2-0-4", 6)) == []
    assert rig.flip_through_d(["weg2-0-4/c5"], d_writes=["d/weg2-0-4"]) == []
    for rid in ("weg2-2-6", "weg2-4-7", "weg2-6-8"):
        assert rig.p_prefill(rid, [f"{rid}/c0"]) == []
        assert rig.flip_through_d([f"{rid}/c0"], d_writes=[f"d/{rid}"]) == []

    burst = [f"weg2-8-{i}" for i in range(10, 18)]
    refused = []
    for rid in burst:
        refused += rig.p_prefill(rid, [f"{rid}/inner", f"{rid}/end"])
    ends = [f"{rid}/end" for rid in burst]
    missing = rig.flip_through_d(ends)
    assert [h for h in refused if h.endswith("/end")] == [], f"end anchors refused: {refused}"
    assert missing == [], f"D found no anchor for {missing} (#1035c CAPPED by=mamba -> W50)"


def test_the_shipped_reset_pins_the_arena_the_releasing_reset_frees_it(tmp_path, monkeypatch):
    """One phase of four hand-overs fills a 4-slot arena; after the flip the
    next prompt's anchor must find room. RED on the shipped reset (every
    reference stays at 3, nothing is a candidate, statuses=[4])."""
    rig = _Rig(tmp_path, 4, monkeypatch)
    for i in range(4):
        assert rig.p_prefill(f"r{i}", [f"r{i}/end"]) == []
    assert rig.flip_through_d([f"r{i}/end" for i in range(4)]) == []
    assert rig.p_prefill("next", ["next/end"]) == []
    assert rig.complete("next/end")


def test_a_claim_makes_room_without_disk_io(tmp_path, monkeypatch):
    """User rule 24.09.: no copy in the compute path. A claim that finds the
    arena full frees what it needs in C -- the shipped claim ran the disk
    round (up to 256 x 56 MiB on NVFP4) in the scheduler thread."""
    rig = _Rig(tmp_path, 2, monkeypatch)
    assert rig.p_prefill("a", ["a/end"]) == [] and rig.p_prefill("b", ["b/end"]) == []
    for r in rig.P:  # the anchors are released (a finished phase): unreferenced, COMPLETE
        for p in (r.cache.root_node,):
            stack = list(p.children.values())
            while stack:
                n = stack.pop()
                stack.extend(n.children.values())
                if n.component_data[M].host_value is not None:
                    r.mp.free(n.component_data[M].host_value)
                    n.component_data[M].host_value = None
    assert rig.refs("a/end") == 0 and rig.refs("b/end") == 0
    assert rig.p_prefill("c", ["c/end"]) == []
    assert all(r.mp._backend.disk_rounds == 0 for r in rig.P), "a claim wrote to disk in the compute path"


def test_the_hand_over_survives_d_claims_only_with_the_carrier_hold(tmp_path, monkeypatch):
    """Why P holds its END anchors across D's phase: D admits a burst one seat
    at a time, and any claim D makes meanwhile (write-back, flush publish, L3
    fill) takes the oldest unreferenced slot. Held, P's hand-overs are never
    candidates; D's own write waits for room instead."""
    rig = _Rig(tmp_path, 4, monkeypatch)
    for i in range(4):
        assert rig.p_prefill(f"r{i}", [f"r{i}/end"]) == []
    rig.p_sleep()
    assert all(rig.refs(f"r{i}/end") == RANKS for i in range(4)), "the end anchors stay referenced"
    assert not rig.d_publish("d/own"), "D's own write finds no room: every slot is a held hand-over"
    assert all(rig.d_read(f"r{i}/end") for i in range(4)), "every hand-over is still there for D"


def test_without_the_hold_a_d_claim_can_take_an_unread_hand_over(tmp_path, monkeypatch):
    """The same flip with SGLANG_WEG2_ENABLE_MAMBA_CARRIER_HOLD=0 (the 27B A
    form): the reset releases the end anchors too, D's claim drops one of
    them before D read it -- the loss the hold exists for."""
    rig = _Rig(tmp_path, 4, monkeypatch)
    for i in range(4):
        assert rig.p_prefill(f"r{i}", [f"r{i}/end"]) == []
    with envs.SGLANG_WEG2_ENABLE_MAMBA_CARRIER_HOLD.override(False):
        rig.p_sleep()
    assert rig.d_publish("d/own")
    assert sum(not rig.complete(f"r{i}/end") for i in range(4)) == 1


def test_inner_anchors_go_back_at_the_reset_end_anchors_at_the_wake(tmp_path, monkeypatch):
    rig = _Rig(tmp_path, 8, monkeypatch)
    assert rig.p_prefill("a", ["a/inner", "a/end"]) == []
    assert rig.p_prefill("b", ["b/inner", "b/end"]) == []
    assert [rig.refs(h) for h in ("a/inner", "a/end", "b/inner", "b/end")] == [RANKS] * 4
    rig.p_sleep()
    assert [rig.refs(h) for h in ("a/inner", "b/inner")] == [0, 0], "inner anchors: prefix cache now"
    assert [rig.refs(h) for h in ("a/end", "b/end")] == [RANKS, RANKS], "end anchors: held for D"
    assert rig.d_read("a/end") and rig.refs("a/end") == 2 * RANKS
    rig.d_sleep()
    assert rig.refs("a/end") == RANKS, "D's reset gives D's references back"
    rig.p_wake()
    assert [rig.refs(h) for h in ("a/end", "b/end")] == [0, 0]
    assert all(rig.complete(h) for h in ("a/inner", "a/end", "b/inner", "b/end")), "released, not dropped"


def test_an_empty_reset_keeps_the_hold_the_next_nonempty_reset_rotates_it(tmp_path, monkeypatch):
    """Fallback without a wake: the hold of one phase is given back at the next
    NON-EMPTY reset (the second flush of a sleep sees an empty tree)."""
    rig = _Rig(tmp_path, 8, monkeypatch)
    assert rig.p_prefill("a", ["a/end"]) == []
    rig.p_sleep()                            # two resets: the empty one keeps the hold
    assert rig.refs("a/end") == RANKS
    assert rig.p_prefill("b", ["b/end"]) == []
    for r in rig.P:
        r.reset(monkeypatch)                 # the next phase's flush, no wake in between
    assert rig.refs("a/end") == 0 and rig.refs("b/end") == RANKS


def test_group_d_and_the_switch_hold_nothing(tmp_path, monkeypatch):
    rig = _Rig(tmp_path, 8, monkeypatch)
    for r in rig.D:                           # a D node marked like an end anchor (never on metal)
        n = r.node("x", end_anchor=True)
        rows = r.cache._weg2_mamba_claim(n, r.mp, "x/end")
        n.component_data[M].host_value = rows
        r.mp.complete_write(rows)
    rig.d_sleep()
    assert rig.refs("x/end") == 0
    assert rig.p_prefill("y", ["y/end"]) == []
    with envs.SGLANG_WEG2_ENABLE_MAMBA_CARRIER_HOLD.override(False):
        rig.p_sleep()
    assert rig.refs("y/end") == 0
    monkeypatch.setattr(u, "_WEG2_END_ANCHOR", False)   # the #1481 mark unarmed: nothing to hold
    assert rig.p_prefill("z", ["z/end"]) == []
    rig.p_sleep()
    assert rig.refs("z/end") == 0


def test_int4_form_single_prompts_are_unchanged(tmp_path, monkeypatch):
    """INT4 arm (32 slots, cap 8): a 97k needle (6 chunk anchors) and single
    prompts hand over exactly as before, no displacement, no refusal."""
    rig = _Rig(tmp_path, 32, monkeypatch)
    assert rig.p_prefill("weg2-0-4", _chain("weg2-0-4", 6)) == []
    assert rig.flip_through_d(["weg2-0-4/c5"]) == []
    for rid in ("weg2-2-6", "weg2-4-7"):
        assert rig.p_prefill(rid, [f"{rid}/c0"]) == []
        assert rig.flip_through_d([f"{rid}/c0"]) == []
    st = rig.P[0].cache._weg2_anchor_ledger.peek("weg2-0-4")
    assert (st.written, st.displaced_share, st.displaced_full, st.refused) == (6, 0, 0, 0)


def test_rows_in_use_and_pending_rows_are_left_alone(tmp_path, monkeypatch):
    rig = _Rig(tmp_path, 16, monkeypatch)    # cap 4: three anchors of one chain, no displacement
    monkeypatch.setattr(u, "_WEG2_END_ANCHOR", False)
    assert rig.p_prefill("a", ["a/0", "a/1", "a/2"], end=False) == []
    r = rig.P[0]
    nodes = []
    stack = list(r.cache.root_node.children.values())
    while stack:
        n = stack.pop()
        stack.extend(n.children.values())
        nodes.append(n)
    nodes.sort(key=lambda n: int(n.component_data[M].host_value[0]))
    nodes[0].write_through_pending_id = 17          # a write-through in flight
    nodes[1].component_data[M].host_lock_ref = 1    # a host operation holds the rows
    monkeypatch.setenv("SGLANG_WEG2_GROUP", "P")
    assert UnifiedRadixCache._release_host_values_before_reset(r.cache) == 1
    slots = [int(n.component_data[M].host_value[0]) - STAGING for n in nodes]
    refs = _hdr_refs(rig.arena)
    assert [int(refs[s]) for s in slots] == [RANKS, RANKS, RANKS - 1]
    pend = r.mp._claim(["claimed-not-acked.sfx"])       # claimed, copy not acked
    assert r.mp.release_tree_rows(torch.tensor([STAGING + pend[0]])) == 0
    assert pend[0] in r.mp._pending


def test_one_failing_pool_never_breaks_the_reset(tmp_path, monkeypatch):
    class _Bad:
        def release_tree_rows(self, rows):
            raise AssertionError("inconsistent pool")

    comp = types.SimpleNamespace(component_type=M, _mamba_pool_host=_Bad())
    root = UnifiedTreeNode(TC)
    n = UnifiedTreeNode(TC)
    n.parent = root
    root.children[0] = n
    n.component_data[M].host_value = torch.tensor([5])
    cache = types.SimpleNamespace(
        cache_controller=types.SimpleNamespace(mem_pool_host=types.SimpleNamespace(arena_read=True)),
        root_node=root, _components_tuple=(comp,))
    monkeypatch.setenv("SGLANG_WEG2_GROUP", "D")
    assert UnifiedRadixCache._release_host_values_before_reset(cache) == 0
    fresh = types.SimpleNamespace(_components_tuple=())  # __init__'s first reset: no controller yet
    assert UnifiedRadixCache._release_host_values_before_reset(fresh) == 0


def test_the_wake_hook_releases_and_never_fails_a_wake():
    from sglang.srt.managers.scheduler_components import weight_updater as wu

    W = wu.SchedulerWeightUpdaterManager
    calls = []

    class _Tree:
        def weg2_release_carrier_hold(self, reason):
            calls.append(reason)
            return 7

    class _Raising:
        def weg2_release_carrier_hold(self, reason):
            raise RuntimeError("boom")

    ns = types.SimpleNamespace
    assert W._weg2_release_carrier_hold_at_wake(ns(scheduler=ns(tree_cache=_Tree()))) == 7
    assert calls == ["wake"]
    assert W._weg2_release_carrier_hold_at_wake(ns(scheduler=ns(tree_cache=_Raising()))) == 0
    assert W._weg2_release_carrier_hold_at_wake(ns(scheduler=None)) == 0
    assert W._weg2_release_carrier_hold_at_wake(ns(scheduler=ns(tree_cache=object()))) == 0
    # the resume handler releases AFTER the pools were restored, in both wake forms
    src = inspect.getsource(wu)
    restore = src.index("flushed = self._weg2_wake_restore_pools()")
    hook = src.index("self._weg2_release_carrier_hold_at_wake()", restore)
    assert hook < src.index('_weg2_ph("flush")', restore)


def test_the_reset_releases_before_it_replaces_the_tree_and_claims_do_no_disk_io():
    src = inspect.getsource(UnifiedRadixCache._reset_full)
    assert src.index("self._release_host_values_before_reset()") < src.index(
        "self.root_node = UnifiedTreeNode(")
    for fn in (ArenaMHAHostPool._claim, ArenaMHAHostPool._claim_np):
        body = inspect.getsource(fn)
        assert "_arena_evict_to_disk" not in body and "_evict_for_claim" in body
    assert ArenaMambaPoolHost._evict_for_claim is ArenaMHAHostPool._evict_for_claim
    assert ArenaMambaPoolHost.release_tree_rows is ArenaMHAHostPool.release_tree_rows
    assert envs.SGLANG_WEG2_ENABLE_MAMBA_CARRIER_HOLD.get() is True  # default ON
