# SPDX-License-Identifier: Apache-2.0
"""END-ANCHOR-GUARD-1024 (#231) round 3 -- the Nachpruefung's two probes as ACCEPTANCE tests (SOLL R7, R11).

The probes (desk/27b-endanchor-guard-recheck-1024 @ 640e473490, remote_results/640e473490_20261024T165337Z, 2/2
green = both splits existed on b636bdf7c3) are turned around: each test is GREEN when the named split is GONE.

P1 / MUSS-4 Form A (NF D: TP0 = host, TP1/TP2 = workers): the host held a protected, unsecured leaf after its K15
   save failed, a worker never holds and demoted it -> the trees' device residency parted (K21: "ALL ranks keep the
   node on the device (a rank-local skip on TP0 alone parted the trees)"). Now: Form A holds nothing -- TP0 saves
   first, then every rank demotes alike; a failed save is named (``FORM-A-DEMOTE``).
P2 / MUSS-3 shared-slot TP (27B D): ``TreeOps.agreed_unneeded`` marked a registry node secured on the rank whose
   ``_has_anchor`` saw a rank-local mamba host copy (device state tombstoned on every rank); the other rank had no
   candidate for it. Now: only device-resident registry nodes are re-arm candidates (a replica) -> the same marks,
   ``protected`` and ``interior_hold`` on every rank.
SOLL-A the #1481 hold no longer runs ahead of the guard on a shared slot (the HOLD lists, and so pass 2, are equal).
SOLL-B a mark that fails after the gather is W203; a failed candidate listing flags the gather incomplete; a failed
       turn-end chain lookup is named (REGISTER-FAILED). SOLL-C ``incomplete`` in the pass stats / PASS line.
SOLL-D class 1 (chain intermediates) keeps a quota in the gather segment and the pass window.
Mutants (in-test code mutants, green = the mutant breaks the invariant): "rank-local candidate set" (the device
filter removed) parts the marks; "Form A holds again" parts host and worker.
"""

from __future__ import annotations

import importlib.util
import inspect
import logging
import os
import types

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

import pytest  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
LOGGER = "flliper.srt.pdflip.end_anchor_guard"


def _load(name, fname):
    spec = importlib.util.spec_from_file_location(name, os.path.join(HERE, fname))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture()
def G(monkeypatch):
    for k in ("FLLIPER_PDFLIP_END_ANCHOR_GUARD", "FLLIPER_PDFLIP_END_ANCHOR_GUARD_MAX_HELD",
              "FLLIPER_PDFLIP_DUAL_LAYOUT", "FLLIPER_PDFLIP_UD_LAST"):
        monkeypatch.delenv(k, raising=False)
    from flliper.srt.pdflip import end_anchor_guard as EAG

    monkeypatch.setattr(EAG, "_N", {k: 0 for k in EAG._N})
    monkeypatch.setattr(EAG, "_SAVE", {k: [0, 0, 0.0, 0.0] for k in EAG._SAVE})
    return _load("_recheck_guard_helpers", "test_end_anchor_guard_1024.py")


@pytest.fixture()
def R():
    return _load("_recheck_rearm_world", "test_park_anchor_rearm_1024.py")


# ---------------------------------------------------------------------------------------------------- P1 / MUSS-4
def _form_a_pair(G, EAG, monkeypatch, save_ok, mod=None):
    mod = mod or EAG
    out = {}
    for role in ("host", "worker"):
        monkeypatch.setattr(mod, "_role", lambda r=role: r)
        t, nodes = G._metal(tp=3, save_ok=save_ok)        # TP0's own S108 deferral (Form A: the host's claim is the group's)
        assert not mod.tp_slot_shared(t)
        G._peel(t, 5000)
        out[role] = (list(t.demoted), {k: n.component_data[G.M].value is not None for k, n in sorted(nodes.items())},
                     list(t.saves))
    return out


@pytest.mark.parametrize("save_ok", [False, True])
def test_p1_form_a_host_and_worker_demote_alike(G, monkeypatch, caplog, save_ok):
    """Probe P1 turned around (MUSS-4, SOLL R11). Red on b636bdf7c3 (save_ok=False): host demoted ['Y'], worker ['X'].
    Green: no hold on Form A -- TP0 runs the K15 save first, then both demote X; a failed save is named."""
    from flliper.srt.pdflip import end_anchor_guard as EAG

    caplog.set_level(logging.INFO, logger=LOGGER)
    out = _form_a_pair(G, EAG, monkeypatch, save_ok)
    assert out["host"][0] == out["worker"][0] == ["X"], out
    assert out["host"][1] == out["worker"][1], "device residency parted between host and worker"
    assert out["host"][2] == ["X"] and out["worker"][2] == [], "TP0 saves first, the worker never claims"
    lines = [r.getMessage() for r in caplog.records if r.name == LOGGER and "FORM-A-DEMOTE" in r.getMessage()]
    if save_ok:
        assert lines == [] and EAG._N["form_a_demote"] == 0
    else:
        assert len(lines) == 1 and "node=X" in lines[0] and EAG._N["form_a_demote"] == 1, lines


def test_p1_form_a_write_back_branch_holds_nothing(G, monkeypatch):
    """The ``just_tried`` site (the leaf's own backup went KV-only a moment ago): on Form A no hold either -- host and
    worker demote X alike, no second claim."""
    from flliper.srt.pdflip import end_anchor_guard as EAG

    out = {}
    for role in ("host", "worker"):
        monkeypatch.setattr(EAG, "_role", lambda r=role: r)
        t, nodes = G._metal(tp=3)
        x = nodes["X"]
        x.backuped = False
        x.component_data[G.F].host_value = None
        G._peel(t, 5000)
        out[role] = (list(t.demoted), list(t.saves))
    assert out["host"] == out["worker"] == (["X"], []), out


def test_p1_form_a_interior_holds_nothing(G, monkeypatch):
    """The interior branch on Form A: the KV-backed END anchor is tombstoned (existing K15 + tombstone) on host and
    worker alike -- no hold on TP0 alone."""
    from flliper.srt.pdflip import end_anchor_guard as EAG

    res = {}
    for role in ("host", "worker"):
        monkeypatch.setattr(EAG, "_role", lambda r=role: r)
        res[role] = G._interior_run([G._inode(1, end=True), G._inode(2)])
    assert res["host"] == res["worker"] == [1], res


def test_p1_form_a_save_that_took_the_leaf_off_the_device_demotes_nothing_twice(G, monkeypatch):
    """After TP0's save the leaf is already off the device: TP0 frees nothing again (no double demotion) -- the same
    residency the worker's demotion gives."""
    from flliper.srt.pdflip import end_anchor_guard as EAG

    monkeypatch.setattr(EAG, "_role", lambda: "host")
    t, nodes = G._metal(tp=3)
    x = nodes["X"]

    def save(n):
        t.saves.append(n.id)
        n.component_data[G.M].value = None
        n.component_data[G.F].value = None
        n.evicted = True
        t.evictable_device_leaves.discard(n)
        return False

    t.pdflip_save_anchor_before_tombstone = save
    assert EAG.before_demote(t, x) is True
    assert t.demoted == [] and x.component_data[G.M].value is None and EAG._N["form_a_split"] == 0


def test_p1_mutant_form_a_holding_again_parts_host_and_worker(G, monkeypatch):
    """MUTANT (MUSS-4): the Form A host takes the one-rank branch again (save, then HOLD) -- host and worker part.
    Proves the P1 acceptance test load-bearing."""
    mod = G._eag_mutant(monkeypatch, '    if role == "host":\n        return _decide_form_a(',
                        '    if False:\n        return _decide_form_a(')
    out = _form_a_pair(G, mod, monkeypatch, save_ok=False, mod=mod)
    assert out["host"][0] != out["worker"][0], f"the mutant did not part host and worker: {out}"


# ---------------------------------------------------------------------------------------------------- P2 / MUSS-3
def _p2(G, EAG, par, R, monkeypatch):
    monkeypatch.setattr(EAG, "_role", lambda: None)
    M = G.M
    trees, xs = [], []
    for r in (0, 1):
        t, root = G._fake_tree(tp=2)
        x = G._Node("X", root, 4096)                    # interior-shaped registry node, KV on the device
        EAG.register_end(t, x)
        x._pdflip_end_anchor = True
        x.component_data[M].value = None                # pass 2 tombstoned its mamba state on EVERY rank
        if r == 0:
            x.component_data[M].host_value = object()   # rank 0 alone kept a host copy (its own claim)
        t._pdflip_anchor_secured_holds = lambda n: False
        trees.append(t)
        xs.append(x)
    assert all(EAG.tp_slot_shared(t) for t in trees)
    ops = {r: par.TreeOps(trees[r], None, busy=False) for r in (0, 1)}
    res, g = R._run_ranks(2, lambda r, gm: par.run_pass(ops[r], gm, r, 2))
    marks = [bool(getattr(x, EAG.AGREED_ATTR, False)) for x in xs]
    # the device state comes back on both ranks (anchor read / re-insert), no new turn end at X
    for x in xs:
        x.component_data[M].value = object()
    prot = [EAG.protected(t, x) for t, x in zip(trees, xs)]
    holds = []
    for t, x in zip(trees, xs):
        prev = EAG.begin(t)
        try:
            holds.append(EAG.interior_hold(t, x, x.component_data[M]))
        finally:
            EAG.end(t, prev)
    return marks, {r: res[r].get("agreed") for r in res}, prot, holds


def test_p2_agreed_marks_stay_rank_equal_with_a_rank_local_host_copy(G, R, monkeypatch):
    """Probe P2 turned around (MUSS-3, SOLL R11). Red on b636bdf7c3: marks [True, False], protected [None, 'end'],
    interior_hold [False, True]. Green: the node without device state is no candidate on either rank -- no mark,
    both protect and hold it alike once the device state is back."""
    from flliper.srt.pdflip import end_anchor_guard as EAG
    from flliper.srt.pdflip import park_anchor_rearm as par

    marks, agreed, prot, holds = _p2(G, EAG, par, R, monkeypatch)
    assert marks[0] == marks[1], f"marks parted: {marks} agreed={agreed}"
    assert agreed[0] == agreed[1], agreed
    assert prot[0] == prot[1] == "end", prot
    assert holds[0] == holds[1] is True, holds


def test_rearm_candidates_are_device_resident_registry_nodes_only(G, monkeypatch):
    """MUSS-3: a registry node with a host copy but no device mamba state is no candidate; with the device state
    back it is (on every rank alike, whatever its host copy)."""
    from flliper.srt.pdflip import end_anchor_guard as EAG

    t, nodes = G._metal(deferred=False)
    x = nodes["X"]
    EAG.register_end(t, x)
    x.component_data[G.M].value = None
    x.component_data[G.M].host_value = object()
    assert x not in EAG.rearm_candidates(t)
    x.component_data[G.M].value = object()
    assert x in EAG.rearm_candidates(t)


def test_p2_mutant_rank_local_candidate_set_parts_the_marks(G, R, monkeypatch):
    """MUTANT (MUSS-3): ``rearm_candidates`` lists the node again although only a rank-local host copy keeps it an
    anchor -- the marks part (probe P2). Proves the P2 acceptance test load-bearing."""
    from flliper.srt.pdflip import park_anchor_rearm as par

    mod = G._eag_mutant(monkeypatch, '        if getattr(_mamba_cd(tree, n), "value", None) is None:\n            continue',
                        '        pass')
    marks, agreed, prot, holds = _p2(G, mod, par, R, monkeypatch)
    assert marks[0] != marks[1], f"the mutant did not part the marks: {marks}"


# ---------------------------------------------------------------------------------------------------- SOLL-A
def _interior_shared(G, monkeypatch, request):
    """Two shared-slot ranks, interior END anchor X registered on both, NOT KV-backed; rank 0 has no host copy (the
    #1481 hold fires there), rank 1 kept one (#1481 does not fire). Another state Z pays."""
    from flliper.srt.mem_cache.unified_cache_components.mamba_component import MambaComponent
    from flliper.srt.pdflip import end_anchor_guard as EAG

    monkeypatch.setattr(EAG, "_role", lambda: None)
    out = []
    for r in (0, 1):
        x, z = G._inode("X", end=True, backuped=False, host=(r == 1)), G._inode("Z")
        evicted, order = [], [x, z]
        lru = types.SimpleNamespace(
            get_lru_no_lock=lambda o=order: o[0] if o else None,
            get_prev_no_lock=lambda n, o=order: o[o.index(n) + 1] if o.index(n) + 1 < len(o) else None,
            in_list=lambda n, o=order: n is not None and n in o)

        def tomb(n, comp, target=None, tracker=None, ev=evicted, o=order):
            ev.append(n.id)
            o.remove(n)
            tracker[G.M] += 1

        cache = types.SimpleNamespace(
            lru_lists={G.M: lru}, evictable_device_leaves=[], tree_components=(G.F, G.M), tp_world_size=2,
            root_node=None, _evict_component_and_detach_lru=tomb, _cascade_evict=lambda n, comp, tracker: None,
            _s108_anchor_deferred={})
        EAG.register_end(cache, x)
        comp = types.SimpleNamespace(cache=cache, component_type=G.M)
        MambaComponent.drive_eviction(comp, types.SimpleNamespace(mamba_num=request), {G.M: 0})
        out.append(evicted)
    return out


def test_solla_1481_hold_does_not_part_the_hold_lists_on_the_shared_slot(G, monkeypatch):
    """SOLL-A. Red on b636bdf7c3: rank 0 held X through #1481 (outside the guard's list), rank 1 through the guard --
    pass 2 (request 2) tombstoned X on rank 1 only: (['Z'], ['Z', 'X']). Green: the guard decides first on the shared
    slot, X is in the HOLD list on both ranks, pass 2 pays it alike."""
    a, b = _interior_shared(G, monkeypatch, request=2)
    assert a == b == ["Z", "X"], (a, b)


def test_solla_pass1_still_holds_on_both_ranks(G, monkeypatch):
    a, b = _interior_shared(G, monkeypatch, request=1)
    assert a == b == ["Z"], (a, b)


# ---------------------------------------------------------------------------------------------------- SOLL-B / C
class _Ops:
    def __init__(self, needs, boom_unneeded=False, cand_failed=False):
        self.n = needs
        self.boom_unneeded = boom_unneeded
        self.candidates_incomplete = cand_failed
        self.unneeded, self.issued = [], []

    def busy(self):
        return False

    def needing(self):
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
        self.issued.append(node)
        return True

    def agreed_secured(self, node):
        pass

    def agreed_unneeded(self, union):
        if self.boom_unneeded:
            raise RuntimeError("rank-local defect")
        self.unneeded.append(sorted(union))
        return 0


def test_sollb_a_failed_mark_after_the_gather_is_w203(R):
    """SOLL-B: a mark that raises on one rank after the gather is ParkRearmDivergence (W203), never swallowed."""
    from flliper.srt.pdflip import park_anchor_rearm as par

    ops = {0: _Ops([]), 1: _Ops([], boom_unneeded=True)}
    with pytest.raises(par.ParkRearmDivergence, match="W203"):
        R._run_ranks(2, lambda r, gm: par.run_pass(ops[r], gm, r, 2))


def test_sollb_a_failed_candidate_listing_flags_the_gather_incomplete_on_every_rank(R):
    """MUSS-3/SOLL-B: a rank whose turn-end candidate listing failed has a short set -- the gather is incomplete on
    every rank, no rank marks off it."""
    from flliper.srt.pdflip import park_anchor_rearm as par

    ops = {0: _Ops([]), 1: _Ops([], cand_failed=True), 2: _Ops([])}
    res, g = R._run_ranks(3, lambda r, gm: par.run_pass(ops[r], gm, r, 3))
    assert all(o.unneeded == [] for o in ops.values())
    assert all(res[r]["incomplete"] == 1 for r in res), res


def test_sollb_tree_ops_flags_a_raising_turn_end_listing(G, monkeypatch):
    from flliper.srt.pdflip import end_anchor_guard as EAG
    from flliper.srt.pdflip import park_anchor_rearm as par

    t, nodes = G._metal(deferred=False)
    t._pdflip_anchor_secured_holds = lambda n: False

    def boom(tree):
        raise RuntimeError("rank-local defect")

    monkeypatch.setattr(EAG, "rearm_candidates", boom)
    ops = par.TreeOps(t, None, busy=False)
    ops.needing()
    assert ops.candidates_incomplete is True


def test_sollb_a_failed_chain_lookup_is_named_and_counted(G, caplog):
    """SOLL-B: ``_pdflip_chain_nodes`` swallows a raising lookup (as before) but leaves the exception for the S108
    turn end, which names it (REGISTER-FAILED) before the registration."""
    from flliper.srt.mem_cache.unified_radix_cache import UnifiedRadixCache
    from flliper.srt.pdflip import end_anchor_guard as EAG

    caplog.set_level(logging.WARNING, logger=LOGGER)

    def boom(params):
        raise RuntimeError("lookup defect")

    t = types.SimpleNamespace(match_prefix=boom, root_node=None)
    assert UnifiedRadixCache._pdflip_chain_nodes(t, object()) == []
    assert isinstance(t._pdflip_chain_lookup_exc, RuntimeError)
    EAG.note_register_failed(t, "chain_lookup", t._pdflip_chain_lookup_exc)
    assert EAG._N["register_failed"] == 1
    assert any("REGISTER-FAILED" in r.getMessage() and "chain_lookup" in r.getMessage() for r in caplog.records)
    src = inspect.getsource(UnifiedRadixCache._s108_turn_end_l2)
    i = src.index('_end_anchor_guard.note_register_failed(self, "chain_lookup", _cexc)')
    assert src.index("self._pdflip_chain_lookup_exc = None") < i < src.index(
        "_end_anchor_guard.register_turn_end(self, chain)")


def test_sollc_incomplete_is_in_the_pass_stats_and_the_pass_line(R):
    from flliper.srt.mem_cache.unified_radix_cache import UnifiedRadixCache
    from flliper.srt.pdflip import park_anchor_rearm as par

    ops = {0: _Ops([5]), 1: _Ops(list(range(100, 100 + par.K_PER_RANK + 1)))}
    res, g = R._run_ranks(2, lambda r, gm: par.run_pass(ops[r], gm, r, 2))
    assert res[0]["incomplete"] == res[1]["incomplete"] == 1
    ops = {0: _Ops([5]), 1: _Ops([])}
    res, g = R._run_ranks(2, lambda r, gm: par.run_pass(ops[r], gm, r, 2))
    assert res[0]["incomplete"] == res[1]["incomplete"] == 0
    src = inspect.getsource(UnifiedRadixCache.pdflip_park_rearm_group)
    assert 'stats.get("incomplete")' in src and 'stats.get("agreed")' in src


# ---------------------------------------------------------------------------------------------------- SOLL-D
def _keys(par):
    end = par.target_key(0, 100000, "end")
    tes = [par.turn_end_key(200000 + i, f"t{i}") for i in range(10)]
    chain = [par.target_key(1, 50000 + i, f"c{i}") for i in range(3)]
    return end, tes, chain


def test_solld_window_keeps_a_chain_quota_taken_from_the_turn_end_class(R):
    from flliper.srt.pdflip import park_anchor_rearm as par

    end, tes, chain = _keys(par)
    w = par.select_window([end] + tes + chain)
    assert len(w) == par.K_PER_RANK and end in w
    assert sum(1 for k in w if par.class_of_target_key(k) == 1) == par.CHAIN_QUOTA
    assert sum(1 for k in w if par.is_turn_end_key(k)) == par.K_PER_RANK - 1 - par.CHAIN_QUOTA
    # park END anchors are never displaced (SOLL-3); without turn-end keys: exactly the k smallest (pre-#231)
    ends = [par.target_key(0, 300000 + i, f"e{i}") for i in range(par.K_PER_RANK + 2)]
    assert par.select_window(ends + chain) == sorted(ends + chain)[:par.K_PER_RANK]
    few = [end] + chain + [par.target_key(1, 60000 + i, f"d{i}") for i in range(8)]
    assert par.select_window(few) == sorted(few)[:par.K_PER_RANK]
    assert par.select_window(tes[:3] + chain) == sorted(tes[:3] + chain)


def test_solld_chain_target_is_issued_although_many_turn_end_targets_wait(R):
    """SOLL-D. Red on b636bdf7c3: ten turn-end needs fill the rank's segment and the window -- the chain target never
    reaches the pass. Green: it is in the window and gets an issue of the pass budget."""
    from flliper.srt.pdflip import park_anchor_rearm as par

    end, tes, chain = _keys(par)
    ops = {0: _Ops(tes + chain[:1]), 1: _Ops([])}
    res, g = R._run_ranks(2, lambda r, gm: par.run_pass(ops[r], gm, r, 2))
    assert chain[0] in ops[0].issued, ops[0].issued
    assert sum(1 for k in ops[0].issued if par.is_turn_end_key(k)) == par.MAX_ISSUES_PER_PASS - 1
    assert res[0]["issued"] == res[1]["issued"] == par.MAX_ISSUES_PER_PASS
