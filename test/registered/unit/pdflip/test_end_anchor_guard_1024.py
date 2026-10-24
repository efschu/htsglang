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
    rank -- no K15 claim in pass 1; the hold reads replicas only (the mark, the copy state)."""
    from flliper.srt.pdflip import end_anchor_guard as EAG

    monkeypatch.setattr(EAG, "_role", lambda: None)
    t, nodes = _metal(tp=3, save_ok=True)
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
    _peel(t, 5000)
    assert "X" not in t.saves and t.backups == []


# ---------------------------------------------------------------------------------------------------- (6) gap 2
def test_unprotected_leaf_gets_the_k15_save_before_its_demotion():
    """A backed leaf whose anchor has no copy and that is no turn-end anchor: the K15 save runs before the demotion
    (it was never asked on this branch); a refused save does not stop the demotion."""
    t, root = _fake_tree()
    z = _Node("Z", root, 3000, prio=1.0)
    t.evictable_device_leaves = {z}
    _peel(t, 2000)
    assert t.saves == ["Z"] and t.demoted == ["Z"]


def test_unprotected_leaf_saved_keeps_its_copy():
    t, root = _fake_tree(save_ok=True)
    z = _Node("Z", root, 3000, prio=1.0)
    t.evictable_device_leaves = {z}
    _peel(t, 2000)
    assert t.saves == ["Z"] and t.demoted == ["Z"] and t.lost == []


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
