# SPDX-License-Identifier: Apache-2.0
"""END-ANCHOR-GUARD-1024 (auftrag #231) -- pdflip/end_anchor_guard.py; SOLL R7.

THE METAL (27B nvfp4duo 93eae72472, boot ..._1024_012625.D.log, D TP0; #224 MUSS-2, #228b). pdflip-22-176 ended its
turn (``#59 FINISH-TRACK cache_len=124928``); the S108 turn-end copy found the mamba arena full and sent the END node
down KV-only (``anchor_deferred`` 436 -> 437). The node was ``backuped`` (KV), its mamba state on the device only.
Within a second the D-leaf demotion (``_evict_device_leaf`` -> ``_evict_to_host``) freed that state -- no K15 save, no
line -- and the follow-up pdflip-22-194 (883 new tokens, < halbX) read ``anchor_was=119552``: u = 124928 + 883 -
119552 = 6259 -> ``RESUME-VIA-P``. R7: D could not take a request under X -> a defect.

The tree below is that shape: the turn-end node X (5376 tokens over a backed parent at 119552 whose anchor has a host
copy) plus a backed leaf Y with a host-copied anchor elsewhere (the evictable rest). The LRU order reaches X first.

WHAT MUST HOLD.
(1) Repro (red before the fix): the peel holds X and pays from Y; X keeps its device state -> the follow-up is short.
(2) R7: the short follow-up of a KV-only turn end stays under halbX (resumes at the turn end, on D).
(3) Danger (a), hang: when nothing else can pay, pass 2 evicts the held anchor (named DEMOTE-AFTER), only as many as
    needed; MAX_HELD bounds the hold per pass (named CAP).
(4) Danger (b), R11: on a pure TP group sharing the mamba slot the hold is decided without any rank-local claim; the
    copy is made by the PARK-REARM pass's agreed victim (3 ranks, one shared arena, the real anchor-only write).
(5) Danger (c), host memory: at most one K15 claim per node and pass; a held node claims nothing.
(6) Gap 2: an unprotected backed leaf gets the K15 save before its demotion (outcome does not change the demotion).
(7) The interior branch (#1481 gap): a BACKED END anchor without a copy is held in pass 1, tombstoned in pass 2.
(8) Switch off / dual layout / Form A worker: the old path.

REVIEW 24.10. (review-endanchor-guard-1024.md, JA MIT AUFLAGEN) -- red on 2fb3bbd61b, green after:
MUSS-1 (SOLL R11) on a pure TP group sharing the mamba slot the hold reads the turn-end REGISTRY (filed at the turn-end
       events on every rank, never from a claim result), the AGREED mark (reduced values of the PARK-REARM pass) and
       the device residency -- never ``_s108_anchor_deferred`` (each rank's own free-claim on the shared arena; metal:
       18 of 307 turn ends rank-uneven) nor the rank-local copy state. Two ranks uneven at the deferral decide the same
       hold; in-test code mutants ("read the rank-local deferral again", "read the rank-local copy state") part them.
       The re-arm candidates come from the registry only, as their own class AFTER the park END anchors (SOLL-3).
MUSS-2 the guard's K15 saves have their own timed, counted line (``END-ANCHOR-GUARD SAVE``) apart from #1465.
SOLL-1 unprotected leaves keep the old demotion (no save); SOLL-2 a node is held once per pass; SOLL-4 the real hybrid
       tree pays the whole evictable request with held anchors (KV peel and mamba slots); SOLL-5 age bound
       (PROTECT_NEWEST = REARM_SCAN); SOLL-6 the pass sets keep the nodes, an id() given back never matches.
"""

from __future__ import annotations

import importlib.util
import logging
import os
import shutil
import types

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

import pytest  # noqa: E402

from flliper.srt.mem_cache.unified_cache_components.tree_component import ComponentType  # noqa: E402

F = ComponentType.FULL
M = ComponentType.MAMBA
LOGGER = "flliper.srt.pdflip.end_anchor_guard"
GUARD_ENV = "FLLIPER_PDFLIP_END_ANCHOR_GUARD"
HALF_X = 2048          # halbX of the 27B D curve at this depth (SOLL: 0,5 * X; X = 4096 here)
TURN_END = 124928      # #59 FINISH-TRACK cache_len of pdflip-22-176
ANCHOR_WAS = 119552    # the deepest anchor with a copy (ANCHOR-ONLY-READ anchor_was)
NEW_TOKENS = 883       # pdflip-22-194's new tokens


@pytest.fixture(autouse=True)
def _env(monkeypatch):
    for k in (GUARD_ENV, "FLLIPER_PDFLIP_END_ANCHOR_GUARD_MAX_HELD", "FLLIPER_PDFLIP_DUAL_LAYOUT",
              "FLLIPER_PDFLIP_UD_LAST"):
        monkeypatch.delenv(k, raising=False)
    try:  # (before the fix the module does not exist: the behaviour tests still run and show the loss)
        from flliper.srt.pdflip import end_anchor_guard as EAG
    except ImportError:
        EAG = None
    if EAG is not None:
        monkeypatch.setattr(EAG, "_N", {k: 0 for k in EAG._N})   # fresh line samplers per test
        if hasattr(EAG, "_SAVE"):
            monkeypatch.setattr(EAG, "_SAVE", {k: [0, 0, 0.0, 0.0] for k in EAG._SAVE})
    yield


# ---------------------------------------------------------------------------------------------------- fake tree
class _CD:
    def __init__(self, value=None, host_value=None):
        self.value = value
        self.host_value = host_value
        self.lock_ref = 0
        self.host_lock_ref = 0


class _Node:
    def __init__(self, nid, parent, tokens, backuped=True, mamba=True, mamba_host=False, prio=0.0):
        self.id = nid
        self.parent = parent
        self.children = {}
        self.key = range(tokens)
        self.backuped = backuped
        self.evicted = False
        self.l3_present = False
        self.write_through_pending_id = None
        self.prio = prio
        self.hash_value = [f"h{nid}"]
        self.pdflip_anchor_rid = None
        self.pdflip_anchor_secured = False
        self._pdflip_end_anchor = False
        self.component_data = [
            _CD(value=range(tokens), host_value=(range(tokens) if backuped else None)),
            _CD(),
            _CD(value=(object() if mamba else None), host_value=(object() if mamba_host else None)),
        ]
        if parent is not None:
            parent.children[nid] = self

    def __lt__(self, other):
        return str(self.id) < str(other.id)


def _fake_tree(tp=1, save_ok=False):
    from flliper.srt.mem_cache import unified_radix_cache as U

    class T:
        _evict_device_leaf = U.UnifiedRadixCache._evict_device_leaf
        pdflip_node_depth = U.UnifiedRadixCache.pdflip_node_depth

    t = T()
    root = _Node("root", None, 0)
    t.root_node = root
    t.tree_components = (F, M)
    t.tp_world_size = tp
    t.cache_controller = types.SimpleNamespace(write_policy="write_back")
    t.ongoing_write_through = {}
    t.demoted, t.lost, t.saves, t.backups = [], [], [], []
    t.save_ok = save_ok
    t._s108_anchor_deferred = {}

    def is_leaf(n):
        return (n is not root and not n.evicted and not any(cd.lock_ref for cd in n.component_data)
                and not any(c.component_data[F].value is not None for c in n.children.values()))

    t._is_device_leaf = is_leaf

    def upd(n):
        if is_leaf(n):
            t.evictable_device_leaves.add(n)
        else:
            t.evictable_device_leaves.discard(n)

    t._update_evictable_leaf_sets = upd

    def evict_to_host(n, tracker):
        assert not n.evicted and n.backuped
        cd = n.component_data[M]
        if cd.value is not None and cd.host_value is None and not n.pdflip_anchor_secured:
            t.lost.append(n.id)          # the device state leaves without a copy
        cd.value = None
        tracker[F] += len(n.key)
        n.component_data[F].value = None
        n.evicted = True
        t.demoted.append(n.id)
        t.evictable_device_leaves.discard(n)
        upd(n.parent)

    t._evict_to_host = evict_to_host

    def save(n):                          # K15: pdflip_save_anchor_before_tombstone
        t.saves.append(n.id)
        if t.save_ok:
            n.component_data[M].host_value = object()
            return True
        return False

    t.pdflip_save_anchor_before_tombstone = save

    def write_backup(n, write_back=False, kv_only_if_mamba_refused=False):
        t.backups.append(n.id)
        n.backuped = True                 # P-FUND: KV backed, the anchor had no arena slot (KV-only)
        n.component_data[F].host_value = range(len(n.key))
        return len(n.key)

    t.write_backup = write_backup
    t.writing_check = lambda write_back=False: None
    t.eviction_strategy = types.SimpleNamespace(get_priority=lambda n: n.prio)
    t.evictable_device_leaves = set()
    return t, root


def _metal(tp=1, save_ok=False, rest=40000, deferred=True, end_mark=False):
    """root <- P(119552, backed, anchor host copy) <- X(5376: the turn end 124928, KV-only, anchor device only);
    root <- Y(rest, backed, anchor host copy) -- more recently used than X."""
    t, root = _fake_tree(tp, save_ok)
    p = _Node("P", root, ANCHOR_WAS, mamba_host=True, prio=0.5)
    x = _Node("X", p, TURN_END - ANCHOR_WAS, prio=1.0)
    x.pdflip_anchor_rid = "pdflip-22-176"
    if deferred:
        t._s108_anchor_deferred[x.id] = x
    x._pdflip_end_anchor = end_mark
    nodes = {"P": p, "X": x}
    t.evictable_device_leaves = {x}
    if rest:
        y = _Node("Y", root, rest, mamba_host=True, prio=2.0)
        nodes["Y"] = y
        t.evictable_device_leaves.add(y)
    return t, nodes


def _peel(t, request):
    from flliper.srt.mem_cache.unified_cache_components.full_component import FullComponent

    comp = types.SimpleNamespace(cache=t, component_type=F)
    tracker = {F: 0, M: 0}
    FullComponent._peel(comp, request, tracker)
    return tracker[F]


def _resume_depth(nodes):
    """The deepest anchor of the chain that survives (device state or a copy): where the follow-up resumes on D."""
    x, p = nodes["X"], nodes["P"]
    xm = x.component_data[M]
    if xm.value is not None or xm.host_value is not None or x.pdflip_anchor_secured:
        return TURN_END
    return ANCHOR_WAS if (p.component_data[M].host_value is not None) else 0


def _msgs(caplog, tag):
    return [r.getMessage() for r in caplog.records if r.name == LOGGER and tag in r.getMessage()]


# ---------------------------------------------------------------------------------------------------- (1) + (2)
def test_metal_22_194_turn_end_anchor_survives_the_peel(caplog):
    """Red before the fix: the peel demoted X first (LRU) and freed its device state without a copy."""
    t, nodes = _metal()
    with caplog.at_level(logging.INFO, logger=LOGGER):
        got = _peel(t, 5000)
    assert got >= 5000
    assert t.lost == [], "the turn-end anchor of pdflip-22-176 left the device without a copy"
    assert t.demoted == ["Y"], "the backed leaf with a host-copied anchor pays the eviction"
    assert nodes["X"].component_data[M].value is not None
    assert nodes["X"] in t.evictable_device_leaves, "held = still evictable for a later pass"
    assert any("HELD" in m and "kinds=s108:1" in m for m in _msgs(caplog, "END-ANCHOR-GUARD")), caplog.messages


def test_r7_short_followup_of_a_kv_only_turn_end_stays_on_d():
    """SOLL R7 (+R2): the follow-up's u is measured against the anchor D still holds. With the guard u = 883 < halbX
    (a short request, D prefills it); the metal's u = 6259 came from the lost turn-end anchor."""
    t, nodes = _metal()
    _peel(t, 5000)
    u = TURN_END + NEW_TOKENS - _resume_depth(nodes)
    assert u == NEW_TOKENS and u < HALF_X, u


def test_switch_off_reproduces_the_metal_loss(monkeypatch):
    """The old peel, byte for byte: X demoted without a copy, u = 6259 (the metal's RESUME-VIA-P u)."""
    monkeypatch.setenv(GUARD_ENV, "0")
    t, nodes = _metal()
    _peel(t, 5000)
    assert t.lost == ["X"] and t.saves == []
    assert TURN_END + NEW_TOKENS - _resume_depth(nodes) == 6259


def test_end_mark_alone_is_protected_too():
    """#1481 END anchor, KV-only backed (no S108 entry): the same hold."""
    t, nodes = _metal(deferred=False, end_mark=True)
    _peel(t, 5000)
    assert t.lost == [] and t.demoted == ["Y"]


def test_kv_only_write_back_branch_is_held_without_a_second_claim():
    """Gap 2, site 2 (``_evict_device_leaf`` write_back branch): the leaf's own backup went KV-only a moment ago
    (P-FUND) -- held, and no K15 claim right after the refused one (``just_tried``)."""
    t, nodes = _metal()
    x = nodes["X"]
    x.backuped = False
    x.component_data[F].host_value = None
    _peel(t, 5000)
    assert t.backups == ["X"] and t.saves == []
    assert t.lost == [] and t.demoted == ["Y"]


# ---------------------------------------------------------------------------------------------------- (3) danger (a)
def test_pass2_evicts_the_held_anchor_when_nothing_else_pays(caplog):
    """Liveness: no other evictable leaf -- the eviction is paid from the held anchor (old order), named."""
    t, nodes = _metal(rest=0)
    with caplog.at_level(logging.INFO, logger=LOGGER):
        got = _peel(t, 5000)
    assert got >= 5000, "the guard turned a payable eviction into an unpaid one (admission would hang)"
    assert t.demoted == ["X"] and t.lost == ["X"]
    assert _msgs(caplog, "DEMOTE-AFTER"), caplog.messages


def test_pass2_takes_only_as_many_held_anchors_as_needed():
    t, root = _fake_tree()
    a = _Node("A", root, 3000, prio=1.0)
    b = _Node("B", root, 3000, prio=1.5)
    for n in (a, b):
        t._s108_anchor_deferred[n.id] = n
    t.evictable_device_leaves = {a, b}
    got = _peel(t, 2000)
    assert got >= 2000
    assert t.demoted == ["A"], "pass 2 stops at the request"
    assert b.component_data[M].value is not None


def test_max_held_bounds_the_hold_per_pass(monkeypatch, caplog):
    monkeypatch.setenv("FLLIPER_PDFLIP_END_ANCHOR_GUARD_MAX_HELD", "1")
    t, root = _fake_tree()
    a = _Node("A", root, 3000, prio=1.0)
    b = _Node("B", root, 6000, prio=1.5)
    y = _Node("Y", root, 40000, mamba_host=True, prio=2.0)
    for n in (a, b):
        t._s108_anchor_deferred[n.id] = n
    t.evictable_device_leaves = {a, b, y}
    with caplog.at_level(logging.INFO, logger=LOGGER):
        _peel(t, 5000)
    assert t.demoted == ["B"], "the second protected node goes the old way once the cap is reached"
    assert a.component_data[M].value is not None
    assert _msgs(caplog, "CAP"), caplog.messages


def test_mamba_slot_eviction_is_paid_from_a_held_leaf_when_it_must():
    """#1470b (alloc_req_slots: mamba_available=0): the mamba drive's pass 2 frees a held turn-end leaf when no other
    mamba state can pay -- the admission is never starved by the guard."""
    t, nodes = _metal(rest=0)
    order = [nodes["X"]]
    lru = types.SimpleNamespace(get_lru_no_lock=lambda: order[0] if order else None,
                                get_prev_no_lock=lambda x: None,
                                in_list=lambda x: x is not None and x in order and x.component_data[M].value is not None)
    t.lru_lists = {M: lru}
    from flliper.srt.mem_cache.unified_cache_components.mamba_component import MambaComponent

    def evict_to_host(n, tracker, _orig=t._evict_to_host):
        _orig(n, tracker)
        tracker[M] += 1

    t._evict_to_host = evict_to_host
    comp = types.SimpleNamespace(cache=t, component_type=M)
    tracker = {F: 0, M: 0}
    MambaComponent.drive_eviction(comp, types.SimpleNamespace(mamba_num=1), tracker)
    assert tracker[M] == 1 and t.demoted == ["X"]


# ---------------------------------------------------------------------------------------------------- (4) danger (b)
def test_shared_slot_tp_group_holds_without_a_rank_local_claim(monkeypatch):
    """R11: on a pure TP group sharing the mamba blob slot a claim landing on some ranks only would decide the hold per
    rank -- no K15 claim in pass 1; the hold reads the structural turn-end registry (MUSS-1)."""
    from flliper.srt.pdflip import end_anchor_guard as EAG

    monkeypatch.setattr(EAG, "_role", lambda: None)
    t, nodes = _metal(tp=3, save_ok=True)
    EAG.register_turn_end(t, [nodes["P"], nodes["X"]])
    _peel(t, 5000)
    assert "X" not in t.saves, "a rank-local claim decided the hold on a shared-slot TP group"
    assert t.lost == [] and t.demoted == ["Y"]
    assert nodes["X"].component_data[M].host_value is None


def test_single_rank_saves_first_and_demotes_with_the_copy():
    """A rank whose claim is the group's claim: K15 first; saved -> the demotion goes ahead WITH a copy, no hold."""
    t, nodes = _metal(save_ok=True)
    _peel(t, 5000)
    assert t.saves[0] == "X" and t.demoted[0] == "X" and t.lost == []


def test_form_a_worker_never_holds(monkeypatch):
    from flliper.srt.pdflip import end_anchor_guard as EAG

    monkeypatch.setattr(EAG, "_role", lambda: "worker")
    t, _ = _metal()
    _peel(t, 5000)
    assert t.demoted[0] == "X" and t.saves == []


def _rearm_world():
    spec = importlib.util.spec_from_file_location(
        "_t_rearm_eag1024", os.path.join(os.path.dirname(os.path.abspath(__file__)), "test_park_anchor_rearm_1024.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.mark.skipif(shutil.which("gcc") is None, reason="needs gcc")
def test_rearm_pass_secures_a_turn_end_anchor_with_one_agreed_victim(tmp_path, monkeypatch, caplog):
    """The copy the guard waits for, made rank-agreed: 3 ranks over ONE shared arena (full), the turn-end END node
    (KV backed, anchor on the device only) is a class-0 target; every rank releases the SAME L3-secured victim and
    writes the anchor (the real anchor-only write); after the acks the slot is COMPLETE on every rank."""
    R = _rearm_world()
    from flliper.srt.mem_cache import form_a_host_shadow as _r12
    from flliper.srt.mem_cache import l3_secured as _l3s
    from flliper.srt.mem_cache.unified_radix_cache import UnifiedRadixCache
    from flliper.srt.pdflip import end_anchor_guard as EAG

    monkeypatch.setattr(_r12, "role", lambda: None)
    monkeypatch.setenv("FLLIPER_PDFLIP_MAMBA_SPILL_LAST_RESORT", "0")
    monkeypatch.delenv("FLLIPER_PDFLIP_PARK_ANCHOR_REARM", raising=False)
    monkeypatch.delenv("FLLIPER_PDFLIP_ANCHOR_ONLY_BACKUP", raising=False)
    monkeypatch.setattr(UnifiedRadixCache, "_spill_tp_skip_n", 0, raising=False)
    monkeypatch.setattr(R._Ops, "raise_issue_left", {}, raising=False)
    monkeypatch.setattr(_l3s, "secured", lambda arena, slots, backend=None, evicting_ok=False:
                        __import__("numpy").ones(len(list(slots)), dtype=bool))
    W = R._World(tmp_path, 2, off=2)
    for rk, end in zip(W.ranks, W.nodes["END"]):
        end._pdflip_end_anchor = True                 # the finish insert's #1481 mark (no park)
        EAG.register_end(rk.cache, end)
    with caplog.at_level(logging.INFO):
        res, err = W.run_pass()
    R._ok(res, err, issued=1, spilled=1, split=0, healed=0)
    assert W.anchor_on("END") == [True, True, True]
    W.ack("END")
    assert W.state("END") == 2
    assert W.anchor_on("old0") == [False, False, False], "the agreed victim left every rank alike"
    assert W.anchor_on("old1") == [True, True, True]


def test_rearm_candidates_carry_turn_end_anchors_only_with_the_switch(monkeypatch):
    from flliper.srt.pdflip import end_anchor_guard as EAG

    t, nodes = _metal(deferred=False)
    nodes["X"]._pdflip_end_anchor = True
    assert EAG.rearm_candidates(t) == []
    EAG.register_end(t, nodes["X"])
    assert nodes["X"] in EAG.rearm_candidates(t)
    nodes["P"].children.pop("X")                          # cut out of the tree: no candidate
    assert nodes["X"] not in EAG.rearm_candidates(t)
    nodes["P"].children["X"] = nodes["X"]
    monkeypatch.setenv(GUARD_ENV, "0")
    assert EAG.rearm_candidates(t) == []


def test_end_registry_is_bounded(monkeypatch):
    from flliper.srt.pdflip import end_anchor_guard as EAG

    t, root = _fake_tree()
    for i in range(EAG.REG_KEEP + 10):
        EAG.register_end(t, types.SimpleNamespace(id=i))
    reg = getattr(t, EAG.REG_ATTR)
    assert len(reg) == EAG.REG_KEEP and 0 not in reg and EAG.REG_KEEP + 9 in reg


# ---------------------------------------------------------------------------------------------------- (5) danger (c)
def test_at_most_one_k15_claim_per_node_and_pass():
    """Single rank, the save fails: pass 1 tries once and holds, pass 2 (nothing else pays) demotes WITHOUT a second
    claim -- nothing that frees an arena slot happened in between."""
    t, nodes = _metal(rest=0)
    _peel(t, 5000)
    assert t.saves.count("X") == 1, t.saves
    assert t.demoted == ["X"]


def test_a_held_node_claims_nothing_on_the_shared_slot_group(monkeypatch):
    from flliper.srt.pdflip import end_anchor_guard as EAG

    monkeypatch.setattr(EAG, "_role", lambda: None)
    t, nodes = _metal(tp=3)
    EAG.register_turn_end(t, [nodes["P"], nodes["X"]])
    _peel(t, 5000)
    assert "X" not in t.saves and t.backups == []
    assert t.demoted == ["Y"], "held through the registry"


# ---------------------------------------------------------------------------------------------------- (6) gap 2
def test_unprotected_leaf_keeps_the_old_demotion_without_a_save():
    """SOLL-1 (Ruecknahme): a backed leaf whose anchor has no copy and that is no turn-end anchor is demoted as before
    #231 -- no synchronous K15 save in the scheduler thread (the 27 metal cases were all turn-end anchors)."""
    t, root = _fake_tree()
    z = _Node("Z", root, 3000, prio=1.0)
    t.evictable_device_leaves = {z}
    _peel(t, 2000)
    assert t.saves == [] and t.demoted == ["Z"]


def test_unprotected_leaf_is_not_saved_even_when_a_save_would_succeed():
    t, root = _fake_tree(save_ok=True)
    z = _Node("Z", root, 3000, prio=1.0)
    t.evictable_device_leaves = {z}
    _peel(t, 2000)
    assert t.saves == [] and t.demoted == ["Z"] and t.lost == ["Z"], "the pre-#231 demotion, unchanged"


def test_secured_anchor_is_neither_saved_nor_held():
    t, nodes = _metal()
    nodes["X"].pdflip_anchor_secured = True
    _peel(t, 5000)
    assert t.saves == [] and t.demoted[0] == "X" and t.lost == []


# ---------------------------------------------------------------------------------------------------- (7) interior
def _interior_run(nodes, request=1, deferred=None):
    from flliper.srt.mem_cache.unified_cache_components.mamba_component import MambaComponent

    evicted = []
    order = list(nodes)
    lru = types.SimpleNamespace(
        get_lru_no_lock=lambda: order[0] if order else None,
        get_prev_no_lock=lambda x: order[order.index(x) + 1] if order.index(x) + 1 < len(order) else None,
        in_list=lambda x: x is not None and x in order,
    )

    def tomb(x, comp, target=None, tracker=None):
        evicted.append(x.id)
        order.remove(x)
        tracker[M] += 1

    cache = types.SimpleNamespace(
        lru_lists={M: lru}, evictable_device_leaves=[], tree_components=(F, M),
        _evict_component_and_detach_lru=tomb, _cascade_evict=lambda x, comp, tracker: None,
        _s108_anchor_deferred=dict(deferred or {}))
    comp = types.SimpleNamespace(cache=cache, component_type=M)
    tracker = {M: 0}
    MambaComponent.drive_eviction(comp, types.SimpleNamespace(mamba_num=request), tracker)
    return evicted


def _inode(nid, end=False, backuped=True, host=False):
    n = types.SimpleNamespace(id=nid, backuped=backuped, _pdflip_end_anchor=end, pdflip_anchor_secured=False,
                              key=range(64), parent=None,
                              component_data=[_CD(value=range(64)), _CD(), _CD(value=object(),
                                                                              host_value=(object() if host else None))])
    return n


def test_interior_backed_end_anchor_without_copy_is_held_in_pass1():
    """#1481's ``not backuped`` gap: a KV-only END anchor is held while another state can pay."""
    assert _interior_run([_inode(1, end=True), _inode(2)]) == [2]


def test_interior_held_anchor_is_tombstoned_in_pass2_when_it_must():
    assert _interior_run([_inode(1, end=True), _inode(2)], request=2) == [2, 1]


def test_interior_s108_deferred_anchor_is_held():
    n1 = _inode(1)
    assert _interior_run([n1, _inode(2)], deferred={1: n1}) == [2]


def test_interior_end_anchor_with_host_copy_is_evictable():
    assert _interior_run([_inode(1, end=True, host=True), _inode(2)]) == [1]


# ---------------------------------------------------------------------------------------------------- (8) switches
def test_dual_layout_keeps_the_old_peel(monkeypatch):
    monkeypatch.setenv("FLLIPER_PDFLIP_DUAL_LAYOUT", "1")
    t, _ = _metal()
    _peel(t, 5000)
    assert t.lost == ["X"] and t.saves == []


def test_default_is_on():
    from flliper.srt.pdflip import end_anchor_guard as EAG

    assert EAG.on({}) is True
    assert EAG.on({GUARD_ENV: "0"}) is False
    assert EAG.max_held({}) == EAG.DEFAULT_MAX_HELD


def test_probe_names_the_reason_through_the_two_pass_frames():
    """#228b's site rule skips the drives' own frames: the 22-194 chain through the full peel and through the mamba
    drive's pass still names the caller (``alloc_token_slots`` / ``_alloc_mamba_slot``), not the new frames."""
    from flliper.srt.mem_cache import mamba_evict_probe as MEP

    assert MEP.classify(["_cascade_evict", "_evict_to_host", "_evict_device_leaf", "_peel_heap", "_peel",
                         "drive_eviction", "evict", "alloc_token_slots"]) == ("alloc", "alloc_token_slots")
    assert MEP.classify(["_cascade_evict", "_tombstone_interior", "_drive_eviction_pass", "drive_eviction", "evict",
                         "_alloc_mamba_slot"]) == ("alloc", "_alloc_mamba_slot")


# ==================================================================================================== review 24.10.
# ---------------------------------------------------------------------------------------------------- MUSS-1 (R11)
def _eag():
    from flliper.srt.pdflip import end_anchor_guard as EAG

    return EAG


def _eag_mutant(monkeypatch, old, new):
    """The guard module with ``old`` replaced by ``new`` in its SOURCE, executed as a separate module and wired in where
    the callers look it up (a code mutant)."""
    import sys

    import flliper.srt.pdflip as pkg
    from flliper.srt.mem_cache import unified_radix_cache as U

    EAG = _eag()
    src = open(EAG.__file__, encoding="utf-8").read()
    assert src.count(old) == 1, old
    mod = types.ModuleType(EAG.__name__)
    mod.__file__ = EAG.__file__
    exec(compile(src.replace(old, new), EAG.__file__, "exec"), mod.__dict__)
    monkeypatch.setattr(pkg, "end_anchor_guard", mod)
    monkeypatch.setitem(sys.modules, EAG.__name__, mod)
    monkeypatch.setattr(U, "_end_anchor_guard", mod)
    monkeypatch.setattr(mod, "_role", lambda: None)
    return mod


def _local_copy(t, x, how):
    """Rank B's own free-claim on the shared arena succeeded (the W65 asymmetry): its anchor copy is in flight /
    acked / marked secured -- rank-local state the other rank does not have."""
    if how == "in_flight":
        t._pdflip_direct_mamba_rows = {x.id: object()}
    elif how == "acked":
        x.component_data[M].host_value = object()
    else:
        x.pdflip_anchor_secured = True


def _two_ranks(how, register=True, end_mark=True):
    """Two replicas of the 22-194 tree on a 2-rank shared-slot D group. Rank A: the turn end's free-claim was refused
    (anchor deferred BY NAME, no copy). Rank B: its claim got the slot (``how``). Both filed the turn end in the
    structural registry when ``register``."""
    EAG = _eag()
    out = []
    for r in (0, 1):
        t, nodes = _metal(tp=2, deferred=(r == 0), end_mark=end_mark)
        if register:
            EAG.register_end(t, nodes["X"])
        if r == 1:
            _local_copy(t, nodes["X"], how)
        out.append((t, nodes))
    return out


def _residency(t, nodes):
    return {k: n.component_data[M].value is not None for k, n in sorted(nodes.items())}


@pytest.mark.parametrize("how", ["in_flight", "acked", "secured_mark"])
def test_r11_two_ranks_uneven_at_the_deferral_decide_the_same_hold(monkeypatch, how):
    """MUSS-1, SOLL R11 (metal: boot ..._93eae72472_1024_012625, anchor_deferred TP0=605 TP1=607 from n=290 on). Red on
    2fb3bbd61b: rank A (deferred, no copy) held X, rank B (its claim landed) demoted it -- the trees' device residency
    parted (K21: "a rank-local skip on TP0 alone parted the trees"). Green: both hold, both pay from Y."""
    EAG = _eag()
    monkeypatch.setattr(EAG, "_role", lambda: None)
    (ta, na), (tb, nb) = _two_ranks(how)
    _peel(ta, 5000)
    _peel(tb, 5000)
    assert ta.demoted == tb.demoted, f"ranks parted: A demoted {ta.demoted}, B demoted {tb.demoted}"
    assert _residency(ta, na) == _residency(tb, nb)
    assert ta.demoted == ["Y"] and ta.saves == [] and tb.saves == []


def test_r11_a_rank_local_deferral_alone_never_holds_on_the_shared_slot(monkeypatch):
    """No turn-end registry entry, only rank A's deferral by name: red on 2fb3bbd61b (A held X from its own
    ``_s108_anchor_deferred``, B demoted it); green: neither holds -- the same old demotion on both."""
    EAG = _eag()
    monkeypatch.setattr(EAG, "_role", lambda: None)
    (ta, na), (tb, nb) = _two_ranks("in_flight", register=False, end_mark=False)
    _peel(ta, 5000)
    _peel(tb, 5000)
    assert ta.demoted == tb.demoted == ["X"], (ta.demoted, tb.demoted)


def _interior_ranks(how):
    """The interior branch on two shared-slot ranks: END anchor X (registered on both), another state Z pays."""
    from flliper.srt.mem_cache.unified_cache_components.mamba_component import MambaComponent

    EAG = _eag()
    res = []
    for r in (0, 1):
        x, z = _inode("X", end=True), _inode("Z")
        evicted, order = [], [x, z]
        lru = types.SimpleNamespace(
            get_lru_no_lock=lambda o=order: o[0] if o else None,
            get_prev_no_lock=lambda n, o=order: o[o.index(n) + 1] if o.index(n) + 1 < len(o) else None,
            in_list=lambda n, o=order: n is not None and n in o)

        def tomb(n, comp, target=None, tracker=None, ev=evicted, o=order):
            ev.append(n.id)
            o.remove(n)
            tracker[M] += 1

        cache = types.SimpleNamespace(
            lru_lists={M: lru}, evictable_device_leaves=[], tree_components=(F, M), tp_world_size=2,
            root_node=None, _evict_component_and_detach_lru=tomb, _cascade_evict=lambda n, comp, tracker: None,
            _s108_anchor_deferred=({"X": x} if r == 0 else {}))
        EAG.register_end(cache, x)
        if r == 1:
            if how == "acked":
                x.component_data[M].host_value = object()
            else:
                cache._pdflip_direct_mamba_rows = {"X": object()}
        comp = types.SimpleNamespace(cache=cache, component_type=M)
        MambaComponent.drive_eviction(comp, types.SimpleNamespace(mamba_num=1), {M: 0})
        res.append(evicted)
    return res


@pytest.mark.parametrize("how", ["in_flight", "acked"])
def test_r11_interior_branch_holds_alike_on_both_ranks(monkeypatch, how):
    """Red on 2fb3bbd61b: rank B (local copy) tombstoned the registered END anchor, rank A held it."""
    monkeypatch.setattr(_eag(), "_role", lambda: None)
    a, b = _interior_ranks(how)
    assert a == b == ["Z"], (a, b)


def test_r11_mutant_reading_the_rank_local_deferral_parts_the_ranks(monkeypatch):
    """MUTANT (MUSS-1): the shared-slot branch of ``protected`` reads ``_s108_anchor_deferred`` again -- the same two
    ranks part (A holds X, B demotes it). Proves the R11 test above load-bearing."""
    _eag_mutant(monkeypatch, "        return None   # R11 shared slot", "        pass   # R11 shared slot")
    (ta, _), (tb, _) = _two_ranks("in_flight", register=False, end_mark=False)
    _peel(ta, 5000)
    _peel(tb, 5000)
    assert ta.demoted != tb.demoted, "the mutant did not part the ranks -- the R11 test is not load-bearing"


def test_r11_mutant_reading_the_rank_local_copy_state_parts_the_ranks(monkeypatch):
    """MUTANT (MUSS-1): the shared-slot hold also asks the rank's own copy state (``unsecured``) -- B demotes X."""
    _eag_mutant(monkeypatch,
                "        if _hold(tree, node, kind):\n            return True\n    if kind is not None and unsecured",
                "        if unsecured(tree, node, cd) and _hold(tree, node, kind):\n            return True\n"
                "    if kind is not None and unsecured")
    (ta, _), (tb, _) = _two_ranks("in_flight")
    _peel(ta, 5000)
    _peel(tb, 5000)
    assert ta.demoted != tb.demoted, "the mutant did not part the ranks -- the R11 test is not load-bearing"


def test_r11_agreed_mark_lifts_the_hold_on_every_rank_alike(monkeypatch):
    """'Secured' only as the PARK-REARM pass's reduced result: the mark on both ranks -> both demote X (the copy
    exists on every rank); before the mark both hold, whatever their local copy state."""
    EAG = _eag()
    monkeypatch.setattr(EAG, "_role", lambda: None)
    (ta, na), (tb, nb) = _two_ranks("acked")
    for t, nodes in ((ta, na), (tb, nb)):
        EAG.mark_agreed(t, nodes["X"])
    _peel(ta, 5000)
    _peel(tb, 5000)
    assert ta.demoted == tb.demoted and ta.demoted[0] == "X"


def test_turn_end_registry_files_the_deepest_device_anchor_of_the_chain():
    EAG = _eag()
    t, nodes = _metal(deferred=False)
    nodes["X"].component_data[M].value = None          # the END node without a device state: the parent is filed
    assert EAG.register_turn_end(t, [nodes["P"], nodes["X"]]) is nodes["P"]
    nodes["X"].component_data[M].value = object()
    assert EAG.register_turn_end(t, [nodes["P"], nodes["X"]]) is nodes["X"]
    assert EAG.protected(t, nodes["X"]) == "turn_end"
    assert nodes["X"] in EAG.rearm_candidates(t) and nodes["P"] in EAG.rearm_candidates(t)


def test_turn_end_registration_is_wired_before_the_claims():
    """The S108 turn end files the chain's END node BEFORE ``_s108_turn_end_active`` (before any claim answers on this
    rank); the deferral by name only stamps its generation."""
    import inspect

    from flliper.srt.mem_cache.unified_radix_cache import UnifiedRadixCache

    src = inspect.getsource(UnifiedRadixCache._s108_turn_end_l2)
    i = src.index("_end_anchor_guard.register_turn_end(self, chain)")
    assert i < src.index("self._s108_turn_end_active = True")
    assert "_end_anchor_guard.note_s108_deferred(self, node)" in inspect.getsource(UnifiedRadixCache)


# ---------------------------------------------------------------------------------------------------- re-arm (R11 + SOLL-3)
class _MarkOps:
    """Fake re-arm adapter with the guard's two mark hooks (records what it was told)."""

    def __init__(self, needs, boom=False):
        self.n = needs
        self.boom = boom
        self.secured, self.unneeded = [], []

    def busy(self):
        return False

    def needing(self):
        if self.boom:
            raise RuntimeError("rank-local defect")
        return {k: k for k in self.n}

    def locate(self, k):
        return k

    def state(self, node):
        return (1 if node in self.n else 0, 0, 0, 0)

    def victims(self, node=None):
        return {}

    def release(self, v):
        return 1

    def issue(self, node):
        return True

    def agreed_secured(self, node):
        self.secured.append(node)

    def agreed_unneeded(self, union):
        self.unneeded.append(sorted(union))
        return 0


def test_rearm_marks_secured_only_on_reduced_outcomes():
    """ISSUED / done marks the target on EVERY rank (reduced values); the complete gather tells every rank which
    registry keys no rank needs."""
    R = _rearm_world()
    from flliper.srt.pdflip import park_anchor_rearm as par

    ops = {0: _MarkOps([5]), 1: _MarkOps([]), 2: _MarkOps([5, 7])}
    res, g = R._run_ranks(3, lambda r, gm: par.run_pass(ops[r], gm, r, 3))
    assert all(o.secured == [5, 7] for o in ops.values()), {r: o.secured for r, o in ops.items()}
    assert all(o.unneeded == [[5, 7]] for o in ops.values())


@pytest.mark.parametrize("why", ["more_than_k", "raised"])
def test_rearm_incomplete_gather_marks_nothing_as_unneeded(why):
    """A rank whose needing list does not fit its gather segment (or raised) makes the union incomplete -- no rank may
    read "not in the union" as "nobody needs it" (fail closed, the same on every rank)."""
    R = _rearm_world()
    from flliper.srt.pdflip import park_anchor_rearm as par

    big = list(range(100, 100 + par.K_PER_RANK + 1))
    ops = {0: _MarkOps([5]), 1: _MarkOps(big if why == "more_than_k" else [], boom=(why == "raised")),
           2: _MarkOps([])}
    res, g = R._run_ranks(3, lambda r, gm: par.run_pass(ops[r], gm, r, 3))
    assert all(o.unneeded == [] for o in ops.values())


@pytest.mark.skipif(shutil.which("gcc") is None, reason="needs gcc")
def test_rearm_issued_turn_end_is_agreed_secured_on_every_rank(tmp_path, monkeypatch):
    """The real pass over ONE shared arena: rank 0 already holds END's copy (its own claim landed), ranks 1/2 do not.
    The pass issues (rank 0 the holder) and marks END secured on EVERY rank -> the guard lifts the hold everywhere.
    A registered node whose copy every rank holds (A) is marked off the complete gather."""
    R = _rearm_world()
    from flliper.srt.mem_cache import form_a_host_shadow as _r12
    from flliper.srt.mem_cache import l3_secured as _l3s
    from flliper.srt.mem_cache.unified_radix_cache import UnifiedRadixCache
    import torch

    EAG = _eag()
    monkeypatch.setattr(_r12, "role", lambda: None)
    monkeypatch.setenv("FLLIPER_PDFLIP_MAMBA_SPILL_LAST_RESORT", "0")
    monkeypatch.delenv("FLLIPER_PDFLIP_PARK_ANCHOR_REARM", raising=False)
    monkeypatch.delenv("FLLIPER_PDFLIP_ANCHOR_ONLY_BACKUP", raising=False)
    monkeypatch.setattr(UnifiedRadixCache, "_spill_tp_skip_n", 0, raising=False)
    monkeypatch.setattr(R._Ops, "raise_issue_left", {}, raising=False)
    monkeypatch.setattr(_l3s, "secured", lambda arena, slots, backend=None, evicting_ok=False:
                        __import__("numpy").ones(len(list(slots)), dtype=bool))
    W = R._World(tmp_path, 6, off=2)
    for r, rk in enumerate(W.ranks):
        a, end = W.nodes["A"][r], W.nodes["END"][r]
        a.component_data[M].host_value = torch.zeros(1, dtype=torch.int64)   # A: every rank holds the copy
        EAG.register_end(rk.cache, a)
        EAG.register_end(rk.cache, end)
        assert EAG.protected(rk.cache, end) == "turn_end"
    W.node(0, "END").component_data[M].host_value = torch.zeros(1, dtype=torch.int64)   # rank 0's own claim landed
    res, err = W.run_pass()
    R._ok(res, err, issued=1, split=0)
    assert [EAG.protected(rk.cache, W.node(r, "END")) for r, rk in enumerate(W.ranks)] == [None, None, None]
    assert [EAG.protected(rk.cache, W.node(r, "A")) for r, rk in enumerate(W.ranks)] == [None, None, None]


@pytest.mark.skipif(shutil.which("gcc") is None, reason="needs gcc")
def test_rearm_park_end_goes_before_turn_end_candidates(tmp_path, monkeypatch):
    """SOLL-3: nine turn-end anchors DEEPER than the park END fill the 8-key window under "class 0, deeper first";
    red on 2fb3bbd61b (the park END never entered the window, not issued). Green: the park END comes first."""
    R = _rearm_world()
    from flliper.srt.mem_cache import form_a_host_shadow as _r12
    from flliper.srt.mem_cache import l3_secured as _l3s
    from flliper.srt.mem_cache.unified_radix_cache import UnifiedRadixCache

    EAG = _eag()
    monkeypatch.setattr(_r12, "role", lambda: None)
    monkeypatch.setenv("FLLIPER_PDFLIP_MAMBA_SPILL_LAST_RESORT", "0")
    monkeypatch.delenv("FLLIPER_PDFLIP_PARK_ANCHOR_REARM", raising=False)
    monkeypatch.delenv("FLLIPER_PDFLIP_ANCHOR_ONLY_BACKUP", raising=False)
    monkeypatch.setattr(UnifiedRadixCache, "_spill_tp_skip_n", 0, raising=False)
    monkeypatch.setattr(R._Ops, "raise_issue_left", {}, raising=False)
    monkeypatch.setattr(_l3s, "secured", lambda arena, slots, backend=None, evicting_ok=False:
                        __import__("numpy").ones(len(list(slots)), dtype=bool))
    tes = [f"T{i}" for i in range(9)]
    W = R._World(tmp_path, 2 + 12, off=2, chain=("END",) + tuple(tes))
    for r, rk in enumerate(W.ranks):
        end = W.nodes["END"][r]
        end.pdflip_park_end = True
        rk.cache._pdflip_park_end_nodes = [end]
        for h in tes:
            EAG.register_end(rk.cache, W.nodes[h][r])
    res, err = W.run_pass()
    assert err == {}, err
    assert W.anchor_on("END") == [True, True, True], "the park END lost the window to turn-end candidates"


# ---------------------------------------------------------------------------------------------------- MUSS-2
def test_save_instrument_times_and_counts_each_occasion(caplog):
    """MUSS-2: the guard's K15 save prints its own line (ms, occasion, site) and counts per occasion -- apart from
    #1465 WRITE-BACK DRAIN; the pass's save sum rides the HELD line."""
    EAG = _eag()
    t, nodes = _metal()
    with caplog.at_level(logging.INFO, logger=LOGGER):
        _peel(t, 5000)
    saves = _msgs(caplog, "END-ANCHOR-GUARD SAVE ")
    assert len(saves) == 1 and "why=pass1" in saves[0] and "site=leaf" in saves[0] and "ok=0" in saves[0], saves
    assert " ms=" in saves[0] and "ms_sum=" in saves[0] and "ms_max=" in saves[0]
    c = EAG.counters()
    assert c["save_pass1_n"] == 1 and c["save_pass1_ok"] == 0 and c["save_pass2_n"] == 0
    assert c["save_pass1_ms_sum"] >= 0.0
    held = _msgs(caplog, "HELD")
    assert held and "save_n=1" in held[0] and "save_ms_sum=" in held[0], held


def test_save_instrument_names_the_no_pass_and_pass2_occasions():
    """Outside a guarded pass the save counts as ``no_pass``; a pass-1 hold right after the write-back branch's own
    refused backup (``just_tried``: no save) gets its save in pass 2 (``pass2``); one claim per node and pass."""
    EAG = _eag()
    t, nodes = _metal()
    assert EAG.before_demote(t, nodes["X"]) is False
    c = EAG.counters()
    assert c["save_no_pass_n"] == 1 and c["save_pass1_n"] == 0, c
    t, nodes = _metal(rest=0)
    x = nodes["X"]
    x.backuped = False
    x.component_data[F].host_value = None
    _peel(t, 5000)
    c = EAG.counters()
    assert c["save_pass2_n"] == 1 and t.demoted == ["X"], (c, t.demoted)


# ---------------------------------------------------------------------------------------------------- SOLL-2
def test_a_node_is_held_once_per_pass_across_the_walks_restart_round(monkeypatch):
    """SOLL-2: the mamba walk restarts at the LRU end after progress and meets the held leaf again -- red on 2fb3bbd61b
    (held twice: it counted twice against MAX_HELD and inflated the HELD line)."""
    from flliper.srt.mem_cache.unified_cache_components.mamba_component import MambaComponent

    EAG = _eag()
    t, root = _fake_tree()
    x = _Node("X", root, 3000, prio=1.0)
    y1 = _Node("Y1", root, 1000, mamba_host=True, prio=2.0)
    y2 = _Node("Y2", root, 1000, mamba_host=True, prio=3.0)
    t._s108_anchor_deferred[x.id] = x
    t.evictable_device_leaves = {x, y1, y2}
    order = [x, y1, y2]

    def in_list(n):
        return n is not None and n in order and n.component_data[M].value is not None

    t.lru_lists = {M: types.SimpleNamespace(
        get_lru_no_lock=lambda: next((n for n in order if in_list(n)), None),
        get_prev_no_lock=lambda n: next((m for m in order[order.index(n) + 1:] if in_list(m)), None),
        in_list=in_list)}

    def evict_to_host(n, tracker, _orig=t._evict_to_host):
        _orig(n, tracker)
        tracker[M] += 1

    t._evict_to_host = evict_to_host
    seen = []
    monkeypatch.setattr(EAG, "note_pass", lambda tree, held, *a, **k: seen.append(list(held)))
    comp = types.SimpleNamespace(cache=t, component_type=M)
    MambaComponent.drive_eviction(comp, types.SimpleNamespace(mamba_num=3), {F: 0, M: 0})
    assert seen and [n.id for n in seen[0]] == ["X"], [[n.id for n in s] for s in seen]
    assert t.demoted == ["Y1", "Y2", "X"]


# ---------------------------------------------------------------------------------------------------- SOLL-5
def test_age_bound_protects_only_the_newest_registry_generations():
    """SOLL-5: a turn-end anchor older than PROTECT_NEWEST registry generations (the window the re-arm pass can still
    secure) no longer protects -- red on 2fb3bbd61b (the registry protected nothing, the deferral set forever)."""
    EAG = _eag()
    t, nodes = _metal(deferred=False)
    EAG.register_turn_end(t, [nodes["X"]])
    _peel(t, 5000)
    assert t.demoted == ["Y"], "fresh: held"
    t2, n2 = _metal(deferred=False)
    EAG.register_turn_end(t2, [n2["X"]])
    for i in range(EAG.PROTECT_NEWEST):
        EAG.register_end(t2, types.SimpleNamespace(id=f"later{i}"))
    assert EAG.registry_age(t2, n2["X"]) == EAG.PROTECT_NEWEST
    assert EAG.protected(t2, n2["X"]) is None and n2["X"] not in EAG.rearm_candidates(t2)
    _peel(t2, 5000)
    assert t2.demoted[0] == "X", "stale: the old demotion"


def test_age_bound_covers_the_s108_deferral_off_the_shared_slot():
    EAG = _eag()
    t, nodes = _metal()                                   # deferred by name, one rank
    EAG.note_s108_deferred(t, nodes["X"])
    assert EAG.protected(t, nodes["X"]) == "s108"
    for i in range(EAG.PROTECT_NEWEST):
        EAG.register_end(t, types.SimpleNamespace(id=f"later{i}"))
    assert EAG.protected(t, nodes["X"]) is None


# ---------------------------------------------------------------------------------------------------- SOLL-6
def test_pass_sets_keep_the_node_an_id_given_back_never_matches():
    """SOLL-6: a node deleted mid-pass gives its id() back; a new node with that id is NOT 'already tried' / 'held'.
    Simulated: the pass set maps the new node's id to another (dead) node -- red on 2fb3bbd61b (id sets)."""
    EAG = _eag()
    t, nodes = _metal()
    old, new = nodes["Y"], nodes["X"]
    setattr(t, EAG.TRIED_ATTR, {id(new): old})
    setattr(t, EAG.PASS2_ATTR, {id(new): old})
    assert not EAG.already_tried(t, new) and not EAG.in_pass2(t, new)
    setattr(t, EAG.TRIED_ATTR, {id(new): new})
    assert EAG.already_tried(t, new)


# ---------------------------------------------------------------------------------------------------- SOLL-4 real tree
def _real_tree():
    from array import array

    import torch

    from flliper.srt.mem_cache.base_prefix_cache import InsertParams
    from flliper.srt.mem_cache.radix_cache import RadixKey

    here = os.path.dirname(os.path.abspath(__file__))
    b = os.path.join(here, "..", "mem_cache", "_hybrid_tree_builder.py")
    if not os.path.exists(b):
        b = os.path.join(here, "test_pdflip_d_seat_compact_0930.py")
    spec = importlib.util.spec_from_file_location("_t_build_eag1024", b)
    T = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(T)
    cache, pool, kalloc = T._build()
    leaves = []
    for start in (1000, 3000, 5000):
        before = {id(n) for n in cache._collect_all_nodes()}
        slot = pool.mamba_allocator.alloc(1)
        toks = list(range(start, start + 64))
        cache.insert(InsertParams(key=RadixKey(array("q", toks)), value=kalloc.alloc(64),
                                  mamba_value=slot.reshape(1)))
        node = next(n for n in cache._collect_all_nodes() if id(n) not in before and n is not cache.root_node)
        node.component_data[F].host_value = torch.arange(64, dtype=torch.int64)   # KV backed (KV-only: no mamba copy)
        cache._update_evictable_leaf_sets(node)
        leaves.append(node)
    return cache, pool, kalloc, leaves


@pytest.mark.parametrize("tp", [1, 2])
def test_real_tree_pays_the_whole_evictable_request_with_held_anchors(monkeypatch, caplog, tp):
    """SOLL-4 (danger d): the real hybrid UnifiedRadixCache, two of three backed leaves are registered turn-end anchors
    without a mamba copy; the request is the whole evictable KV -- evict() delivers all of it (no extend OOM), the
    held anchors go in pass 2, named."""
    from flliper.srt.mem_cache import form_a_host_shadow as _r12
    from flliper.srt.mem_cache.base_prefix_cache import EvictParams

    EAG = _eag()
    monkeypatch.setattr(_r12, "role", lambda: None)
    monkeypatch.setattr(EAG, "_role", lambda: None)
    cache, pool, kalloc, leaves = _real_tree()
    cache.tp_world_size = tp
    for n in leaves[:2]:
        EAG.register_turn_end(cache, [n])
    free0 = kalloc.available_size()
    total = sum(len(n.key) for n in leaves)
    with caplog.at_level(logging.INFO, logger=LOGGER):
        res = cache.evict(EvictParams(num_tokens=total, mamba_num=0))
    assert res.num_tokens_evicted >= total, res
    assert kalloc.available_size() == free0 + total
    assert all(n.component_data[F].value is None for n in leaves)
    assert len(_msgs(caplog, "DEMOTE-AFTER")) == 2, caplog.messages


@pytest.mark.parametrize("tp", [1, 2])
def test_real_tree_serves_the_mamba_slots_with_held_anchors(monkeypatch, tp):
    """SOLL-4: ``alloc_req_slots`` with mamba_available short -> evict(mamba_num=all states): every slot comes back and
    the allocation is served, the held turn-end leaves included (pass 2)."""
    from flliper.srt.mem_cache import form_a_host_shadow as _r12
    from flliper.srt.mem_cache.base_prefix_cache import EvictParams

    EAG = _eag()
    monkeypatch.setattr(_r12, "role", lambda: None)
    monkeypatch.setattr(EAG, "_role", lambda: None)
    cache, pool, kalloc, leaves = _real_tree()
    cache.tp_world_size = tp
    for n in leaves[:2]:
        EAG.register_turn_end(cache, [n])
    m0 = pool.mamba_allocator.available_size()
    res = cache.evict(EvictParams(num_tokens=0, mamba_num=3))
    assert res.mamba_num_evicted == 3, res
    assert pool.mamba_allocator.available_size() == m0 + 3
    assert pool.mamba_allocator.alloc(3) is not None


# ---------------------------------------------------------------------------------------------------- re-arm mutants
def _rearm_mutant(old, new):
    from flliper.srt.pdflip import park_anchor_rearm as par

    src = open(par.__file__, encoding="utf-8").read()
    assert src.count(old) == 1, old
    mod = types.ModuleType("park_anchor_rearm_eag_mutant")
    mod.__file__ = par.__file__
    exec(compile(src.replace(old, new), par.__file__, "exec"), mod.__dict__)
    return mod


def test_rearm_mutant_ignoring_the_incomplete_flag_marks_a_needed_anchor_secured():
    """MUTANT (danger: false 'secured' lifts the hold while a rank still lacks the copy): the incomplete flag ignored
    -- the ranks read "not in the union" as "nobody needs it" although rank 1's list did not fit its segment."""
    R = _rearm_world()
    from flliper.srt.pdflip import park_anchor_rearm as par

    mut = _rearm_mutant("    if not any_incomplete:\n", "    if True:\n")
    big = list(range(100, 100 + par.K_PER_RANK + 1))
    ops = {0: _MarkOps([5]), 1: _MarkOps(big), 2: _MarkOps([])}
    R._run_ranks(3, lambda r, gm: mut.run_pass(ops[r], gm, r, 3))
    assert all(o.unneeded for o in ops.values()), "the mutant changed nothing -- the incomplete test is not load-bearing"


@pytest.mark.skipif(shutil.which("gcc") is None, reason="needs gcc")
def test_rearm_mutant_without_the_turn_end_class_starves_the_park_end(tmp_path, monkeypatch):
    """MUTANT (SOLL-3): turn-end keys without their own class bit -- the deeper turn-end anchors fill the 8-key window
    and the park END is not issued (the green test above is load-bearing)."""
    R = _rearm_world()
    from flliper.srt.mem_cache import form_a_host_shadow as _r12
    from flliper.srt.mem_cache import l3_secured as _l3s
    from flliper.srt.mem_cache.unified_radix_cache import UnifiedRadixCache

    EAG = _eag()
    mut = _rearm_mutant("    return target_key(0, depth, last_hash) | TURN_END_BIT\n",
                        "    return target_key(0, depth, last_hash)\n")
    monkeypatch.setattr(_r12, "role", lambda: None)
    monkeypatch.setenv("FLLIPER_PDFLIP_MAMBA_SPILL_LAST_RESORT", "0")
    monkeypatch.delenv("FLLIPER_PDFLIP_PARK_ANCHOR_REARM", raising=False)
    monkeypatch.delenv("FLLIPER_PDFLIP_ANCHOR_ONLY_BACKUP", raising=False)
    monkeypatch.setattr(UnifiedRadixCache, "_spill_tp_skip_n", 0, raising=False)
    monkeypatch.setattr(_l3s, "secured", lambda arena, slots, backend=None, evicting_ok=False:
                        __import__("numpy").ones(len(list(slots)), dtype=bool))
    tes = [f"T{i}" for i in range(9)]
    W = R._World(tmp_path, 2 + 12, off=2, chain=("END",) + tuple(tes))
    for r, rk in enumerate(W.ranks):
        end = W.nodes["END"][r]
        end.pdflip_park_end = True
        rk.cache._pdflip_park_end_nodes = [end]
        for h in tes:
            EAG.register_end(rk.cache, W.nodes[h][r])
    res, err = W.run_pass(ops_cls=mut.TreeOps, mod=mut)
    assert err == {}, err
    assert W.anchor_on("END") == [False, False, False], "the mutant still issued the park END"
