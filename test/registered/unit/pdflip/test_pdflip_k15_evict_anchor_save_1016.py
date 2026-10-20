"""K15 (NF h16q2 1cfbf32584, boot _215812, D TP0): pdflip-56-530 priced 1339, computed 9787.

Metal: pdflip-56-528 resumed at 73984 (``MATCH-CENSUS-DEEP reached=73984
accepted=73984`` 00:21:41) and extended to its end anchor 74752 (``#59
RESUMABLE depth=74752 last_track=74752``) -- 73984 became an INTERNAL anchor.
The device mamba eviction tombstones internal anchors (``drive_eviction``
internal branch, ``target=DEVICE``) and the host mamba arena was full of D's own
tree anchors (``ARENA-REF-CENSUS arena-58834944 complete=32 pinned=32
own_held=31``), so 73984 had no host copy and was gone. The follow-up 530
(fork 74560, ``#1040 key_match_depth=74560 anchor_depth=65536``) resumed at
65536: 75323 - 65536 = 9787 computed instead of 75323 - 73984 = 1339.

Pinned:
  1. the internal branch backs the anchor up before the tombstone (the real
     ``MambaComponent.drive_eviction`` over a stub LRU);
  2. ``UnifiedRadixCache.pdflip_save_anchor_before_tombstone`` (real method):
     own backup -> joined write -> host copy; a refused backup takes the spill
     room once and retries; no room at all -> ``PDFLIP-ANCHOR-LOST at=evict``;
  3. the leaf eviction's P-FUND KV-only drop tries the spill room first;
  4. ``FLLIPER_PDFLIP_EVICT_ANCHOR_SAVE=0`` = the old tombstone.
RED on 1cfbf32584 (no save, no method)."""
from __future__ import annotations

import inspect
import logging
import os
import types

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

import pytest  # noqa: E402

from flliper.srt.mem_cache.unified_cache_components.mamba_component import MambaComponent  # noqa: E402
from flliper.srt.mem_cache.unified_radix_cache import UnifiedRadixCache  # noqa: E402
from flliper.srt.mem_cache.unified_cache_components.tree_component import ComponentType  # noqa: E402

PROMPT, ANCHOR_TOMBSTONED, ANCHOR_LEFT = 75323, 73984, 65536


@pytest.fixture(autouse=True)
def _env(monkeypatch):
    monkeypatch.delenv("FLLIPER_PDFLIP_EVICT_ANCHOR_SAVE", raising=False)


def _node(nid, depth, backuped=False):
    cd = types.SimpleNamespace(value=object(), host_value=None)
    return types.SimpleNamespace(id=nid, depth=depth, backuped=backuped, _pdflip_end_anchor=False,
                                 write_through_pending_id=None,
                                 component_data={"mamba": cd, ComponentType.MAMBA: cd})


def _drive(nodes, save):
    evicted, order = [], list(nodes)
    lru = types.SimpleNamespace(
        get_lru_no_lock=lambda: order[0] if order else None,
        get_prev_no_lock=lambda x: order[order.index(x) + 1] if order.index(x) + 1 < len(order) else None,
        in_list=lambda x: x is not None and x in order,
    )

    def _evict(x, comp, target=None, tracker=None):
        evicted.append(x.id)
        x.component_data["mamba"].value = None      # the device state goes
        order.remove(x)
        tracker["mamba"] += 1

    cache = types.SimpleNamespace(lru_lists={"mamba": lru}, evictable_device_leaves=[],
                                  _evict_component_and_detach_lru=_evict,
                                  _cascade_evict=lambda x, comp, tracker: None)
    if save is not None:
        cache.pdflip_save_anchor_before_tombstone = save
    comp = types.SimpleNamespace(cache=cache, component_type="mamba")
    tracker = {"mamba": 0}
    MambaComponent.drive_eviction(comp, types.SimpleNamespace(mamba_num=1), tracker)
    return evicted


def test_pdflip_56_530_internal_anchor_is_saved_before_the_tombstone():
    n73984 = _node(1, ANCHOR_TOMBSTONED)
    calls = []

    def save(x):
        calls.append(x.id)
        x.component_data["mamba"].host_value = object()   # the joined backup landed
        return True

    assert _drive([n73984, _node(2, 80000)], save) == [1]
    assert calls == [1]
    assert n73984.component_data["mamba"].value is None             # device freed (admission served)
    assert n73984.component_data["mamba"].host_value is not None     # the anchor survives on the host
    # what the follow-up resumes from: the deepest anchor <= its fork 74560
    resume = ANCHOR_TOMBSTONED if n73984.component_data["mamba"].host_value is not None else ANCHOR_LEFT
    assert PROMPT - resume == 1339


def test_a_hosted_anchor_is_not_backed_up_again():
    n = _node(1, ANCHOR_TOMBSTONED)
    n.component_data["mamba"].host_value = object()
    calls = []
    assert _drive([n], lambda x: calls.append(x.id)) == [1]
    assert calls == []


# --- the real save method on a stub tree -------------------------------------

def _tree(written_seq, spill_ok=True):
    t = types.SimpleNamespace()
    t.cache_controller = types.SimpleNamespace(write_policy="write_back")
    t.calls = []
    seq = list(written_seq)

    def _wb(node, write_back=False):
        t.calls.append("backup")
        w = seq.pop(0) if seq else 0
        if w:
            node._landed = True
        else:
            t._k15_last_refusal = t.refusal     # what #1421 names (default: the mamba arena)
        return w

    def _wc(write_back=False):
        t.calls.append("join")
        for n in t.nodes:
            if getattr(n, "_landed", False):
                n.component_data[ComponentType.MAMBA].host_value = object()

    t.write_backup = _wb
    t.write_backup_anchor_only = lambda node, kv_on_device=False: _wb(node)
    t.writing_check = _wc
    t._pdflip_mamba_pool = lambda: object()
    t._pdflip_flush_spill_room = lambda node, mp: (t.calls.append("spill"), spill_ok)[1]
    t._pdflip_evict_spill_room = types.MethodType(UnifiedRadixCache._pdflip_evict_spill_room, t)
    t.pdflip_node_depth = lambda node: node.depth
    t._pdflip_sweep_last_refusal = None
    t.refusal = "mamba_claim"
    t.nodes = []
    return t


def _save(t, node):
    t.nodes.append(node)
    return UnifiedRadixCache.pdflip_save_anchor_before_tombstone(t, node)


def test_save_backs_up_and_joins(caplog):
    t, n = _tree([1]), _node(7, ANCHOR_TOMBSTONED)
    with caplog.at_level(logging.INFO):
        assert _save(t, n) is True
    assert t.calls == ["backup", "join"]
    assert any("K15 ANCHOR-SAVED" in m and "depth=73984" in m for m in caplog.messages)


def test_full_arena_takes_the_spill_room_once_then_saves():
    """The metal state: arena 32/32 pinned -> the first backup is refused."""
    t, n = _tree([0, 1]), _node(7, ANCHOR_TOMBSTONED)
    assert _save(t, n) is True
    assert t.calls == ["backup", "spill", "backup", "join"]


def test_no_room_anywhere_is_named_and_tombstoned(caplog):
    t, n = _tree([0, 0], spill_ok=False), _node(7, ANCHOR_TOMBSTONED)
    with caplog.at_level(logging.WARNING):
        assert _save(t, n) is False
    assert t.calls == ["backup", "spill"]
    assert any("PDFLIP-ANCHOR-LOST at=evict" in m and "depth=73984" in m for m in caplog.messages)


def test_spill_is_asked_once_per_node():
    t, n = _tree([0, 0, 0, 0]), _node(7, ANCHOR_TOMBSTONED)
    _save(t, n)
    _save(t, n)
    assert t.calls.count("spill") == 1


def test_switch_off_is_the_old_tombstone(monkeypatch):
    monkeypatch.setenv("FLLIPER_PDFLIP_EVICT_ANCHOR_SAVE", "0")
    t, n = _tree([1]), _node(7, ANCHOR_TOMBSTONED)
    assert _save(t, n) is False and t.calls == []


def test_capped_or_in_flight_anchor_is_left_alone():
    t, n = _tree([1]), _node(7, ANCHOR_TOMBSTONED)
    n._pdflip_capped = True
    assert _save(t, n) is False and t.calls == []
    n2 = _node(8, ANCHOR_TOMBSTONED)
    n2.write_through_pending_id = 8
    assert _save(t, n2) is False and t.calls == []


def test_pfund_kv_only_tries_the_spill_room_first():
    """The leaf eviction's KV-only drop (``P-FUND EVICT KV-ONLY``, 6x in this
    boot) asks the spill room before it gives the anchor up."""
    src = inspect.getsource(UnifiedRadixCache._pdflip_direct_claim)
    a = src.index("self._pdflip_evict_spill_room(node, mp)")
    b = src.index("# P-FUND: the eviction needs this node's KV rows")
    assert a < b



# --- review K15 (wf_37a4dbeb-6e1) Auflagen A1-A5: the reviewer's probes R1-R5, taken over ---

from flliper.srt.mem_cache import form_a_host_shadow as _r12  # noqa: E402


@pytest.mark.parametrize("reason", ["arena_claim", "mamba_pin", "anchor_only_pin", "write_none"])
def test_a1_only_a_mamba_arena_refusal_spills(reason):
    """R1: boot _215812 D TP0 had 38 non-mamba refusals (arena_claim 21, anchor_only_pin 17), P 51
    arena_claim -- a spill there cannot serve the retry and costs the victim its host anchor."""
    t, n = _tree([0, 0]), _node(7, ANCHOR_TOMBSTONED)
    t.refusal = reason
    assert _save(t, n) is False
    # S90b: a KV-arena refusal (arena_claim) of a device-KV node asks the anchor alone once (the 2nd "backup"
    # here, refused again by the stub) -- still no spill
    assert "spill" not in t.calls and t.calls == (["backup", "backup"] if reason == "arena_claim" else ["backup"])


@pytest.mark.parametrize("reason", ["mamba_claim", "anchor_only_claim"])
def test_a1_mamba_arena_refusals_do_spill(reason):
    t, n = _tree([0, 1]), _node(7, ANCHOR_TOMBSTONED)
    t.refusal = reason
    assert _save(t, n) is True and t.calls == ["backup", "spill", "backup", "join"]


def test_a2_p_with_pp_stages_saves_without_spill(caplog):
    """B2: on P (PP3) the mamba slot is shared by the stages; a stage-local spill frees nothing."""
    t, n = _tree([0, 0]), _node(7, ANCHOR_TOMBSTONED)
    t.pp_size, t.pp_rank = 3, 1
    with caplog.at_level(logging.INFO):
        assert _save(t, n) is False
    assert "spill" not in t.calls and t.calls == ["backup"]
    assert any("K15 EVICT-SPILL SKIPPED pp_size=3 pp_rank=1" in m for m in caplog.messages)


def test_a3_form_a_worker_names_no_loss(monkeypatch, caplog):
    """R2: a Form A worker has no mamba arena (anchor_only_no_arena, 43x per worker in the boot)."""
    monkeypatch.setattr(_r12, "role", lambda: "worker")
    t, n = _tree([0, 0]), _node(7, ANCHOR_TOMBSTONED, backuped=True)
    t._pdflip_mamba_pool = lambda: None
    lost0 = UnifiedRadixCache._k15_lost_n
    with caplog.at_level(logging.WARNING):
        assert _save(t, n) is False
    assert not any("PDFLIP-ANCHOR-LOST at=evict" in m for m in caplog.messages)
    assert UnifiedRadixCache._k15_lost_n == lost0 and t.calls == []


def _drive_with(nodes, save, request=None):
    evicted, order = [], list(nodes)
    lru = types.SimpleNamespace(
        get_lru_no_lock=lambda: order[0] if order else None,
        get_prev_no_lock=lambda x: (order[order.index(x) + 1]
                                    if x in order and order.index(x) + 1 < len(order) else None),
        in_list=lambda x: x is not None and x in order,
    )

    def _evict(x, comp, target=None, tracker=None):
        if getattr(x, "_pinned", False):
            raise ValueError("#904: refusing to free MAMBA device rows of node %s while lock_ref=1" % x.id)
        evicted.append(x.id)
        x.component_data["mamba"].value = None
        order.remove(x)
        tracker["mamba"] += 1

    cache = types.SimpleNamespace(lru_lists={"mamba": lru}, evictable_device_leaves=[],
                                  _evict_component_and_detach_lru=_evict,
                                  _cascade_evict=lambda x, comp, tracker: None,
                                  pdflip_save_anchor_before_tombstone=lambda x: save(x, order))
    comp = types.SimpleNamespace(cache=cache, component_type="mamba")
    tracker = {"mamba": 0}
    MambaComponent.drive_eviction(comp, types.SimpleNamespace(mamba_num=request or len(nodes)), tracker)
    return evicted


def test_a5_r3_pinned_after_save_is_not_tombstoned():
    """R3 (#904 guard): the save's anchor-only backup pins `a` (inc_lock_ref) -- no tombstone."""
    a, b = _node(1, 1000), _node(2, 2000)

    def save(x, order):
        if x is a:
            x._pinned = True
            order.remove(x)
        return False

    assert _drive_with([a, b], save) == [2]


def test_r4_walk_terminates_when_every_save_takes_its_node_off():
    nodes = [_node(i, 1000 * i) for i in range(1, 9)]

    def save(x, order):
        x._pinned = True
        order.remove(x)
        return False

    assert _drive_with(nodes, save) == []


def test_a4_r5_successor_taken_off_by_the_save_does_not_end_the_walk():
    """R5: request 2; the save of `a` pins `b` -- the walk goes on at the LRU end AFTER a's tombstone."""
    a, b, c = _node(1, 1000), _node(2, 2000), _node(3, 3000)

    def save(x, order):
        if x is a and b in order:
            b._pinned = True
            order.remove(b)
        return False

    assert _drive_with([a, b, c], save, request=2) == [1, 3]
