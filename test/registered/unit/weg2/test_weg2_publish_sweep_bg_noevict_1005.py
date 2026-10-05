# SPDX-License-Identifier: Apache-2.0
"""PUBLISH-SWEEP-BG option A (desk report 1830 §5, y9e boot 4f96e9b89b): a background claim never makes room.

y9e: ``issue_ms`` of a BG pass is the wall time of one ``write_backup`` in the scheduler thread, and it
jumps from 1-16 ms to 19-109 ms exactly when the KV arena is full and the claim has to make room
(``_evict_for_claim``: 12 of 12 passes with an ARENA-DROP vs 0 of 22 without; need=2769 slots -> 108.7 ms).
The ``_weg2_bg_publish`` comment says "BG: no spill", but that only skips ``_w3_arena_spill``; the
room-making INSIDE ``alloc_write`` still ran. Option A: with ``SGLANG_WEG2_PUBLISH_SWEEP_BG_NO_EVICT=1`` a BG
claim on a full arena is the named refusal ``arena_claim`` (the node stays un-backed for the flush, which
keeps its spill rights and its own room-making).

Pinned here:
* the switch exists and defaults OFF; it only ever arms inside a running BG sweep (``bg_publish_on()``),
  which the dual layout never runs (the dual gate);
* OFF (the default) and in the flush (``_weg2_bg_publish`` False): ``alloc_write`` is called exactly as before
  (no new keyword on the call), the arena makes room as before;
* ON inside BG: a claim that needs room is refused, nothing is evicted, no reference leaks, the arena is
  unchanged; a claim that fits (free slot / found COMPLETE) is unaffected; the flush then gets the room;
* both claim paths (numpy mask path and the dict path) behave alike;
* the decision reads only replicated state (env + the BG flag set by the replicated tick): no clock,
  no rank-local value.
"""

from __future__ import annotations

import inspect
import os
import shutil
import types

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

import pytest  # noqa: E402
import torch  # noqa: E402

from sglang.srt.mem_cache.hicache_storage import HiCacheFile  # noqa: E402
from sglang.srt.mem_cache.pool_host.arena_pool import ArenaMHAHostPool  # noqa: E402
from sglang.srt.mem_cache.storage.file.hicache_arena import ShmArena  # noqa: E402
from sglang.srt.mem_cache.unified_cache_components.tree_component import ComponentType  # noqa: E402
from sglang.srt.mem_cache.unified_radix_cache import UnifiedRadixCache, UnifiedTreeNode  # noqa: E402
from sglang.test.ci.ci_register import register_cpu_ci  # noqa: E402

register_cpu_ci(est_time=20, suite="stage-a-test-cpu")

NOEVICT = "SGLANG_WEG2_PUBLISH_SWEEP_BG_NO_EVICT"
needs_gcc = pytest.mark.skipif(shutil.which("gcc") is None, reason="needs gcc")

L, H, D = 2, 2, 4
CELL = H * D
PAGE = 64
S = 5
FULL = ComponentType.FULL


def _nb():
    from sglang.srt.managers import weg2_flush_nonblock as nb

    return nb


@pytest.fixture(autouse=True)
def _env(monkeypatch):
    monkeypatch.delenv("SGLANG_WEG2_DUAL_LAYOUT", raising=False)
    monkeypatch.delenv(NOEVICT, raising=False)
    monkeypatch.delenv("SGLANG_WEG2_PUBLISH_SWEEP_BG", raising=False)
    monkeypatch.setenv("SGLANG_WEG2_GROUP", "D")


# ---- the switch and its gate ---------------------------------------------------------

def test_switch_exists_and_defaults_off():
    from sglang.srt.environ import envs

    assert envs.SGLANG_WEG2_PUBLISH_SWEEP_BG_NO_EVICT.get() is False
    assert _nb().bg_no_evict_on() is False


def test_switch_arms_only_inside_the_flip_line_bg_sweep(monkeypatch):
    nb = _nb()
    monkeypatch.setenv(NOEVICT, "1")
    assert nb.bg_no_evict_on() is True
    # the dual gate: the dual layout never runs the BG sweep, so the lever never arms there
    monkeypatch.setenv("SGLANG_WEG2_DUAL_LAYOUT", "1")
    assert nb.bg_publish_on() is False and nb.bg_no_evict_on() is False
    monkeypatch.delenv("SGLANG_WEG2_DUAL_LAYOUT")
    # BG off = no BG claim exists = nothing to arm
    monkeypatch.setenv("SGLANG_WEG2_PUBLISH_SWEEP_BG", "0")
    assert nb.bg_no_evict_on() is False
    monkeypatch.delenv("SGLANG_WEG2_PUBLISH_SWEEP_BG")
    # not group D (e.g. P)
    monkeypatch.setenv("SGLANG_WEG2_GROUP", "P")
    assert nb.bg_no_evict_on() is False


def test_decision_reads_only_env_and_the_bg_flag_no_clock_no_rank_local():
    import ast
    import textwrap

    fn = ast.parse(textwrap.dedent(inspect.getsource(_nb().bg_no_evict_on))).body[0]
    fn.body = fn.body[1:]                                   # the docstring is prose, not code
    src = ast.unparse(fn)
    for bad in ("time.", "monotonic", "perf_counter", "tp_rank", "rank", "random", "arena"):
        assert bad not in src, bad
    c = inspect.getsource(UnifiedRadixCache._weg2_direct_claim)
    i = c.index("alloc_write(hashes, allow_evict=False)")
    seg = c[i - 700:i + 100]
    assert "_weg2_bg_publish" in seg and "_weg2_bg_no_evict()" in seg
    assert "bg_no_evict_on()" in inspect.getsource(UnifiedRadixCache._weg2_bg_no_evict)
    for bad in ("time.", "monotonic", "perf_counter", "tp_rank"):
        assert bad not in seg, bad


# ---- the real arena ------------------------------------------------------------------

class _Win:
    total_bytes = PAGE
    extents = ((1 * CELL, L * CELL), (PAGE // 2 + 1 * CELL, L * CELL))


class _Evictor:
    def reserve(self, stem, size, key=None, owner_writes_whole_file=False):
        return True

    def commit(self, stem):
        pass

    def abort(self, stem):
        pass


def _file_backend(root):
    be = object.__new__(HiCacheFile)

    def _path(stem):
        return os.path.join(root, stem + ".bin")

    be._existing_path = _path
    be._sharded_path = _path
    be._ensure_shard_dir = lambda path: None
    be._stat_stems = lambda stems: {s: os.path.getsize(_path(s)) for s in stems if os.path.exists(_path(s))}
    be._evictor = _Evictor()
    be._key_geom = {"is_mla_model": False}
    be._arena_evict_to_disk = lambda arena, want: 0
    be._get_suffixed_key = lambda key: key + "_sfx"
    be._suffix_for_key = lambda key: ("_sfx",)
    return be


def _pool(tmp_path, slots, np_path=True):
    p = object.__new__(ArenaMHAHostPool)
    p.layout = "layer_first"; p.page_size = 1; p.layer_num = L; p.head_num = H; p.head_dim = D
    p.dtype = torch.uint8; p.device = "cpu"; p.pin_memory = False; p.size = S
    p.element_dim = H * D; p.can_use_jit = True
    p.free_slots = torch.arange(S, dtype=torch.int64); p.slot_used = torch.zeros(S, dtype=torch.bool)
    p.kv_buffer = torch.zeros(2, L, S, H, D, dtype=torch.uint8)
    p._arena_init_fields()
    arena = ShmArena(str(tmp_path / "kv.bin"), PAGE, slots)
    p.bind(arena, _Win(), role="kv", pin=False)
    root = tmp_path / "store"
    root.mkdir(exist_ok=True)
    p._backend = _file_backend(str(root))
    if not np_path:
        p._pending_mask = None      # the dict claim path
    return p, arena


def _fill_unreferenced(p, arena, n):
    """n COMPLETE pages nobody references: exactly what a claim's room-making may take."""
    stems = p._stems([f"a{i}" for i in range(n)])
    pay = torch.full((PAGE,), 0x41, dtype=torch.uint8)
    for st in stems:
        assert arena.write([st], [PAGE], [((0, PAGE),)], [pay.data_ptr()]) == [1]
    return stems


def _present(arena, stems):
    return sum(1 for slot, _st in arena.find_slots(stems) if slot >= 0)


class _Spy:
    """Counts the room-making rounds of a pool."""

    def __init__(self, pool):
        self.n = 0
        self._orig = pool._evict_for_claim

        def spy(arena, need, claim_stem=None):
            self.n += 1
            return self._orig(arena, need, claim_stem=claim_stem)

        pool._evict_for_claim = spy


@needs_gcc
@pytest.mark.parametrize("np_path", [True, False], ids=["np-claim", "dict-claim"])
class TestPoolClaim:
    def test_default_claim_still_makes_room(self, tmp_path, np_path):
        p, arena = _pool(tmp_path, 4, np_path)
        old = _fill_unreferenced(p, arena, 4)
        spy = _Spy(p)
        ids = p.alloc_write(["b0", "b1"])
        assert ids is not None and int(ids.numel()) == 2 and spy.n == 1
        assert _present(arena, old) == 2

    def test_allow_evict_false_refuses_a_full_arena_and_changes_nothing(self, tmp_path, np_path):
        p, arena = _pool(tmp_path, 4, np_path)
        old = _fill_unreferenced(p, arena, 4)
        spy = _Spy(p)
        assert p.alloc_write(["b0", "b1"], allow_evict=False) is None
        assert spy.n == 0
        assert _present(arena, old) == 4                       # nothing evicted
        assert not getattr(p, "_pending", None) or not p._pending
        # the refusal left no trace: the same claim with room-making allowed now succeeds in full
        ids = p.alloc_write(["b0", "b1"])
        assert ids is not None and int(ids.numel()) == 2 and spy.n == 1

    def test_allow_evict_false_does_not_touch_a_claim_that_fits(self, tmp_path, np_path):
        p, arena = _pool(tmp_path, 4, np_path)
        _fill_unreferenced(p, arena, 2)                         # 2 free slots left
        spy = _Spy(p)
        ids = p.alloc_write(["b0", "b1"], allow_evict=False)
        assert ids is not None and int(ids.numel()) == 2 and spy.n == 0

    def test_allow_evict_false_found_complete_pages_keep_no_stray_reference(self, tmp_path, np_path):
        """a claim that finds one page COMPLETE (a join of a finished page) and needs room for the other:
        refused, and the early reader reference taken on the found page is given back (#1424f)."""
        p, arena = _pool(tmp_path, 4, np_path)
        old = _fill_unreferenced(p, arena, 4)
        h_found = "a0"
        refs_before = arena.ref_counts() if hasattr(arena, "ref_counts") else None
        assert p.alloc_write([h_found, "b1"], allow_evict=False) is None
        if refs_before is not None:
            assert arena.ref_counts() == refs_before
        # nothing referenced -> the whole arena is still evictable: a default claim of 4 pages succeeds
        ids = p.alloc_write(["c0", "c1", "c2", "c3"])
        assert ids is not None and int(ids.numel()) == 4


# ---- the tree's direct claim ---------------------------------------------------------

class _NoLRU:
    def in_list(self, node):
        return False


class _CallRec:
    """Wraps a real pool; records the exact alloc_write call shape."""

    def __init__(self, pool):
        self._p = pool
        self.calls = []

    def __getattr__(self, k):
        return getattr(self._p, k)

    def alloc_write(self, hashes, *a, **kw):
        self.calls.append((a, dict(kw)))
        return self._p.alloc_write(hashes, *a, **kw)


def _tree(pool):
    t = object.__new__(UnifiedRadixCache)
    t.root_node = UnifiedTreeNode((FULL,))
    t.page_size = 1
    t.ongoing_write_through = {}
    t.evictable_host_leaves = set()
    t._r12_rec = None
    t.lru_lists = {FULL: _NoLRU()}
    t.host_lru_lists = {FULL: _NoLRU()}
    t.components = {}
    t.cache_controller = types.SimpleNamespace(mem_pool_host=pool, mem_pool_host_draft=None)
    t.refused = []
    t._1421_refused = lambda why, node: t.refused.append(why)
    t._weg2_direct_pool = lambda: pool
    t.spilled = []
    t._w3_arena_spill = lambda pool_, need, claimer=None: t.spilled.append(need) or 0
    return t


def _claimer(t, n=2):
    m = UnifiedTreeNode((FULL,))
    m.parent = t.root_node
    t.root_node.children[("b",)] = m
    m.hash_value = [f"b{i}" for i in range(n)]
    m.component_data[FULL].value = torch.arange(n, dtype=torch.int64)
    return m


@needs_gcc
class TestDirectClaim:
    def _setup(self, tmp_path):
        p, arena = _pool(tmp_path, 4)
        old = _fill_unreferenced(p, arena, 4)
        rec = _CallRec(p)
        return p, arena, old, rec, _tree(rec)

    def test_off_the_default_call_shape_is_unchanged_in_bg_and_flush(self, tmp_path):
        """the flip form: OFF -> alloc_write(hashes) with no new keyword, BG or not, and the arena makes room."""
        for bg in (False, True):
            (tmp_path / f"bg{int(bg)}").mkdir()
            p, arena, old, rec, t = self._setup(tmp_path / f"bg{int(bg)}")
            t._weg2_bg_publish = bg
            pre = t._weg2_direct_claim(_claimer(t))
            assert pre is not False and pre is not None, t.refused
            assert rec.calls == [((), {})], rec.calls
            assert _present(arena, old) == 2 and t.refused == []

    def test_on_in_bg_a_claim_that_needs_room_is_the_named_refusal(self, tmp_path, monkeypatch):
        monkeypatch.setenv(NOEVICT, "1")
        p, arena, old, rec, t = self._setup(tmp_path)
        t._weg2_bg_publish = True
        assert t._weg2_direct_claim(_claimer(t)) is False
        assert t.refused == ["arena_claim"]
        assert rec.calls == [((), {"allow_evict": False})]
        assert _present(arena, old) == 4                       # nothing evicted: no ARENA-DROP, no slot freed
        assert t.spilled == []                                  # BG never spills (unchanged)

    def test_on_the_flush_afterwards_gets_its_room_with_its_spill_rights(self, tmp_path, monkeypatch):
        monkeypatch.setenv(NOEVICT, "1")
        p, arena, old, rec, t = self._setup(tmp_path)
        t._weg2_bg_publish = True
        m = _claimer(t)
        assert t._weg2_direct_claim(m) is False
        t._weg2_bg_publish = False                              # the flush
        rec.calls.clear()
        pre = t._weg2_direct_claim(m)
        assert pre is not False and pre is not None and int(pre.numel()) == 2
        assert rec.calls == [((), {})]                          # the flush call is the old call
        assert _present(arena, old) == 2

    def test_on_a_bg_claim_that_fits_is_unaffected(self, tmp_path, monkeypatch):
        monkeypatch.setenv(NOEVICT, "1")
        p, arena = _pool(tmp_path, 4)
        _fill_unreferenced(p, arena, 2)
        rec = _CallRec(p)
        t = _tree(rec)
        t._weg2_bg_publish = True
        pre = t._weg2_direct_claim(_claimer(t))
        assert pre is not False and pre is not None and int(pre.numel()) == 2

    def test_on_but_the_flush_is_never_restricted(self, tmp_path, monkeypatch):
        monkeypatch.setenv(NOEVICT, "1")
        p, arena, old, rec, t = self._setup(tmp_path)
        t._weg2_bg_publish = False
        got = t._weg2_direct_claim(_claimer(t))
        assert got is not False and got is not None
        assert rec.calls == [((), {})]

    def test_dual_gate_the_lever_is_inert_under_the_dual_layout(self, tmp_path, monkeypatch):
        """the dual layout never runs a BG sweep, but even a stray flag with the switch on must not restrict
        a claim there."""
        monkeypatch.setenv(NOEVICT, "1")
        monkeypatch.setenv("SGLANG_WEG2_DUAL_LAYOUT", "1")
        p, arena, old, rec, t = self._setup(tmp_path)
        t._weg2_bg_publish = True
        got = t._weg2_direct_claim(_claimer(t))
        assert got is not False and got is not None
        assert rec.calls == [((), {})]


def test_two_ranks_decide_alike_from_the_same_replicated_inputs(monkeypatch):
    """TP lockstep: same env + same BG flag -> same call shape on every rank; no input differs per rank."""
    nb = _nb()
    for env in ({}, {NOEVICT: "1"}, {NOEVICT: "1", "SGLANG_WEG2_DUAL_LAYOUT": "1"},
                {NOEVICT: "1", "SGLANG_WEG2_PUBLISH_SWEEP_BG": "0"}):
        for k in (NOEVICT, "SGLANG_WEG2_DUAL_LAYOUT", "SGLANG_WEG2_PUBLISH_SWEEP_BG"):
            monkeypatch.delenv(k, raising=False)
        for k, v in env.items():
            monkeypatch.setenv(k, v)
        assert nb.bg_no_evict_on() == nb.bg_no_evict_on()


# ---- flip unchanged (pattern: test_dual_fixes_flip_unchanged_1003) -------------------

class TestFlipUnchanged:
    """The flip form (NF / 27B flip, switch not set): the lever is inert in every part it touched."""

    def test_signatures_default_to_the_old_behaviour(self):
        for fn in (ArenaMHAHostPool.alloc_write, ArenaMHAHostPool._claim, ArenaMHAHostPool._claim_np):
            assert inspect.signature(fn).parameters["allow_evict"].default is True, fn
        from sglang.srt.mem_cache.pool_host import arena_mamba_pool as amp

        # the mamba anchor arena is deliberately untouched (its drops are need=1 and cheap, desk 1830 §5 A)
        assert "allow_evict" not in inspect.signature(amp.ArenaMambaPoolHost.alloc_write).parameters

    def test_the_bg_claim_call_is_the_old_call_when_the_switch_is_off(self):
        """a pool whose alloc_write takes ONLY hashes (the pre-option-A shape) still works in BG and flush."""
        seen = []

        class _OldPool:
            def alloc_write(self, hashes):
                seen.append(tuple(hashes))
                return None

        t = object.__new__(UnifiedRadixCache)
        t.page_size = 1
        t.root_node = UnifiedTreeNode((FULL,))
        t.cache_controller = types.SimpleNamespace(mem_pool_host=None, mem_pool_host_draft=None)
        t.refused = []
        t._1421_refused = lambda why, node: t.refused.append(why)
        t._weg2_direct_pool = lambda: _OldPool()
        t._w3_arena_spill = lambda *a, **k: 0
        for bg in (False, True):
            t._weg2_bg_publish = bg
            assert t._weg2_direct_claim(_claimer(t)) is False
        assert t.refused == ["arena_claim", "arena_claim"] and len(seen) == 2

    def test_environ_default_is_off_in_the_flip_env(self):
        from sglang.srt.environ import envs

        assert envs.SGLANG_WEG2_PUBLISH_SWEEP_BG_NO_EVICT.get() is False
        assert envs.SGLANG_WEG2_PUBLISH_SWEEP_BG.get() is True      # the y9e BG default itself is not touched


# ---- the BG pass ends at the first no-room refusal ------------------------------------

def _sweep_tree(refuse_bg_with):
    from sglang.srt.mem_cache.unified_radix_cache import BASE_COMPONENT_TYPE

    class _CDn:
        value = (1,)
        lock_ref = 0

    class _N:
        def __init__(self, parent, nid):
            self.parent = parent; self.id = nid; self.children = {}; self.evicted = False
            self.backuped = False; self.l3_present = False; self.write_through_pending_id = None
            self.key = [nid] * 4
            self.component_data = {BASE_COMPONENT_TYPE: _CDn()}
            if parent is not None:
                parent.children[nid] = self

    class _Fk:
        def __init__(self, root):
            self.cache_controller = types.SimpleNamespace(_draft_l3_write_issued=0, _draft_l3_write_refused=0)
            self.disable = False; self.root_node = root
            self.ongoing_write_through = {}; self.ongoing_backup = {}
            self.component_protected_size_ = {BASE_COMPONENT_TYPE: 0}
            self._mamba_pin_budget = 8
            self.tried = []

        def _weg2_publish_window(self):
            return 1 << 30

        def _weg2_anchor_only_candidate(self, node):
            return False

        def _mamba_pins_held(self):
            return 0

        def write_backup(self, node):
            self.tried.append(node.id)
            if getattr(self, "_weg2_bg_publish", False) and refuse_bg_with:
                self._weg2_sweep_last_refusal = refuse_bg_with
                return 0
            node.backuped = True
            return 1

    root = _N(None, 0)
    for i in range(1, 6):
        _N(root, i)
    return _Fk(root)


def test_bg_pass_stops_at_the_first_no_room_refusal_and_the_flush_does_not():
    fk = _sweep_tree("arena_full_bg")
    st = UnifiedRadixCache.publish_unbacked_sweep(fk, max_issue=1, background=True)
    assert fk.tried == [1] and st["stopped"] == "arena_full" and st["refused"] == 1
    # the flush sweep (background=False) is the old walk: the marker never stops it, it tries on
    fk = _sweep_tree("arena_full_bg")
    UnifiedRadixCache.publish_unbacked_sweep(fk, max_issue=64)
    assert fk.tried == [1, 2, 3, 4, 5]
    # switch off: no marker is ever set, a refused BG node does not end the pass (as before)
    fk = _sweep_tree(None)
    UnifiedRadixCache.publish_unbacked_sweep(fk, max_issue=1, background=True)
    assert fk.tried == [1]   # issued 1 -> the rest is counted only
    fk = _sweep_tree("some_other_refusal")
    UnifiedRadixCache.publish_unbacked_sweep(fk, max_issue=1, background=True)
    assert fk.tried == [1, 2, 3, 4, 5]


def test_direct_claim_sets_the_marker_only_for_the_no_room_refusal(tmp_path, monkeypatch):
    if shutil.which("gcc") is None:
        pytest.skip("needs gcc")
    (tmp_path / "off").mkdir()
    (tmp_path / "on").mkdir()
    p, arena = _pool(tmp_path / "off", 4)
    _fill_unreferenced(p, arena, 4)
    t = _tree(_CallRec(p))
    t._weg2_bg_publish = True
    t._weg2_direct_claim(_claimer(t))                      # switch OFF: makes room, no marker
    assert getattr(t, "_weg2_sweep_last_refusal", None) is None
    monkeypatch.setenv(NOEVICT, "1")
    p, arena = _pool(tmp_path / "on", 4)
    _fill_unreferenced(p, arena, 4)
    t = _tree(_CallRec(p))
    t._weg2_bg_publish = True
    assert t._weg2_direct_claim(_claimer(t)) is False
    assert t._weg2_sweep_last_refusal == "arena_full_bg"
