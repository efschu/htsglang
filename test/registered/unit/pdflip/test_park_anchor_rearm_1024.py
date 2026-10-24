# SPDX-License-Identifier: Apache-2.0
"""PARK-REARM-1024 (auftrag #203): the park-end anchors a full host mamba arena refused are secured again, group-agreed.

THE METAL (27B INT8 rc39a, boot ..._b9cddabda1_1024_080355, D TP=3 off Form A, 08:17:01): mamba arena 112/112
(``arena_pinned=112/112`` in 330 of 334 mamba ARENA-DROP samples on TP0), ``#1421 BACKUP-REFUSED why=mamba_claim
node=250/366/368``, ``SPILL SKIPPED site=park_first`` / ``flush_spill`` (W65: a rank-local spill of the shared slot frees
it only at the last releaser), the intermediates "given up on purpose, not retried", ``PDFLIP-ANCHOR-LOST at=flush
n=27`` with node 366's depth 78336 among them.

RED on the rc41 head 133069b099: ``flliper.srt.pdflip.park_anchor_rearm`` does not exist (ImportError), the rank-local
claim of the END stays refused on every rank (the W65 skip), a dropped chain intermediate is never an anchor-only
candidate again. GREEN: one group pass backs the END on EVERY rank of the 3-rank shared arena (every rank releases the
SAME agreed victim, the barrier, every rank claims -> the slot COMPLETE: all three extents in), a slot given back
re-arms without a spill, a split heals by the late rank's join, and the protocol posts the same collectives on every
rank whatever its local state (threads + a barrier all-reduce: a rank that skipped one would break the barrier).

Mutants (both danger directions, in-test): a rank-local victim choice (rank disunity) and a victim without its L3 copy
(loss) are each caught by the invariants of the green tests.

Hermetic: the real shared arena (arena.c, gcc) with three rank extents and the real UnifiedRadixCache claim /
release of the H19 harness (test_pdflip_mamba_arena_displace_h19.py); no torch.distributed, no GPU.
"""
from __future__ import annotations

import importlib.util
import logging
import os
import shutil
import threading
import types

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

import pytest  # noqa: E402
import torch  # noqa: E402

from flliper.srt.mem_cache import form_a_host_shadow as _r12  # noqa: E402
from flliper.srt.mem_cache import l3_secured as _l3s  # noqa: E402
from flliper.srt.mem_cache.unified_cache_components.tree_component import ComponentType  # noqa: E402
from flliper.srt.mem_cache.unified_radix_cache import UnifiedRadixCache, UnifiedTreeNode  # noqa: E402
from flliper.srt.pdflip import park_anchor_rearm as par  # noqa: E402

M = ComponentType.MAMBA
F = ComponentType.FULL
needs_gcc = pytest.mark.skipif(shutil.which("gcc") is None, reason="needs gcc")


# ---------------------------------------------------------------------------------------------------------------------
# a 3-rank collective: threads + a barrier all-reduce (MAX). A rank that skips or adds a collective breaks the barrier.
class _Group:
    def __init__(self, n):
        self.n = n
        self.b = threading.Barrier(n, timeout=20)
        self.vals = [None] * n
        self.calls = [0] * n
        self.shapes = []

    def max_for(self, r):
        def f(values):
            self.vals[r] = [int(v) for v in values]
            self.calls[r] += 1
            self.b.wait()
            lens = {len(v) for v in self.vals}
            if len(lens) != 1:
                self.b.abort()
                raise AssertionError(f"collective shape mismatch {lens}")
            out = [max(col) for col in zip(*self.vals)]
            self.b.wait()
            return out
        return f


def _run_ranks(n, fn):
    """fn(rank, group_max) on n threads; returns {rank: result}; re-raises the first error."""
    g = _Group(n)
    res, err = {}, []

    def _w(r):
        try:
            res[r] = fn(r, g.max_for(r))
        except BaseException as exc:  # noqa: BLE001
            err.append(exc)
            g.b.abort()

    th = [threading.Thread(target=_w, args=(r,)) for r in range(n)]
    for t in th:
        t.start()
    for t in th:
        t.join(30)
    if err:
        raise err[0]
    assert len(set(g.calls)) == 1, f"ranks posted different collective counts: {g.calls}"
    return res, g


# ---- the protocol ---------------------------------------------------------------------------------------------------

def test_agree_min_common_finds_the_smallest_key_every_rank_holds():
    sets = {0: [5, 7, 9], 1: [7, 9], 2: [3, 7, 11]}
    res, g = _run_ranks(3, lambda r, gm: par.agree_min_common(sets[r], gm))
    assert {k for k, _ in res.values()} == {7}
    assert len({rounds for _, rounds in res.values()}) == 1


def test_agree_min_common_none_when_nothing_is_common():
    sets = {0: [1], 1: [2], 2: []}
    res, g = _run_ranks(3, lambda r, gm: par.agree_min_common(sets[r], gm))
    assert {k for k, _ in res.values()} == {None}


def test_gather_union_is_the_same_on_every_rank():
    sets = {0: [10, 4], 1: [4], 2: [99]}
    res, g = _run_ranks(3, lambda r, gm: par.gather_union(sets[r], r, 3, gm, veto=(r == 1)))
    assert all(v == ([4, 10, 99], True) for v in res.values())


def test_keys_order_end_first_deeper_first():
    a = par.target_key(0, 78336, "h366")
    b = par.target_key(0, 173312, "h361")
    c = par.target_key(1, 200000, "hX")
    assert b < a < c
    assert par.depth_of_target_key(a) == 78336 and par.class_of_target_key(c) == 1
    assert par.victim_key(512, "x") < par.victim_key(1024, "y")


class _FakeOps:
    """Divergent local states; ``boom`` raises inside every rank-local step."""

    def __init__(self, needs, boom=False):
        self.n = needs
        self.boom = boom

    def _chk(self):
        if self.boom:
            raise RuntimeError("rank-local defect")

    def busy(self):
        return False

    def needing(self):
        self._chk()
        return {k: k for k in self.n}

    def locate(self, k):
        self._chk()
        return k

    def state(self, node):
        self._chk()
        return (1 if node in self.n else 0, 0, 0, 1)

    def victims(self):
        self._chk()
        return {7: "v7"}

    def release(self, v):
        self._chk()
        return 1

    def issue(self, node):
        self._chk()
        return True


def test_collectives_are_uniform_under_divergent_states_and_errors():
    ops = {0: _FakeOps([1, 2]), 1: _FakeOps([2]), 2: _FakeOps([1, 2], boom=True)}
    res, g = _run_ranks(3, lambda r, gm: par.run_pass(ops[r], gm, r, 3))
    # rank 2 failed closed (needs nothing, locates nothing) -> no target is located on every rank
    assert all(st["targets"] == 0 for st in res.values())
    ops = {0: _FakeOps([1, 2]), 1: _FakeOps([2]), 2: _FakeOps([1])}
    res, g = _run_ranks(3, lambda r, gm: par.run_pass(ops[r], gm, r, 3))
    assert len({st["collectives"] for st in res.values()}) == 1
    assert all(st["issued"] == 2 and st["spilled"] == 2 for st in res.values())


# ---- the real shared arena (H19 harness) ------------------------------------------------------------------------------

def _h19():
    spec = importlib.util.spec_from_file_location(
        "_h19_1024", os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                  "test_pdflip_mamba_arena_displace_h19.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


class _Ops(par.TreeOps):
    """The real adapter; only the WRITE is the harness's (the real claim + this rank's extent completed: the ack)."""

    fail_issue = False

    def issue(self, n):
        if self.fail_issue:
            return False
        t = self.tree
        t._pdflip_park_rearm_active = True
        try:
            n.pdflip_park_anchor_dropped = False
            rows = t._pdflip_mamba_claim(n, self.mp, n.hash_value[-1])
        finally:
            t._pdflip_park_rearm_active = False
        if rows is None:
            return False
        n.component_data[M].host_value = rows
        self.mp.complete_write(rows)
        return True


class _World:
    """A D tree on 3 ranks over ONE shared arena: off-chain anchors (untagged, settled, on L3) fill the arena; the
    parked chain A -> END (KV backed, anchor on the device only)."""

    def __init__(self, tmp_path, slots, off, chain=("A", "END")):
        H = self.H = _h19()
        self.arena = H._arena(tmp_path, slots)
        self.ranks = [H._Rank(self.arena, r, cfg=0) for r in range(H.RANKS)]

        class _Sec(H._Backend):
            def arena_secure_to_disk(self, arena, cands, writer="claim_room"):
                return {"on_disk": len(cands), "written": 0, "lost": 0}

        for rk in self.ranks:
            rk.mp._backend = _Sec()
            c = rk.cache
            c.tp_world_size, c.pp_size = 3, 1
            c.tree_components = (F, M)
            c._mamba_write_through_pin_admissible = lambda n, write_back=False: True
        self.off = [f"old{i}" for i in range(off)]
        for rk in self.ranks:
            for i, h in enumerate(self.off):
                n = rk.add(None, h, tokens=512 * (i + 1), parent=rk.cache.root_node)
                self._dev(n, h)
            assert rk.sweep()
        self.chain = list(chain)
        self.nodes = {h: [] for h in self.chain}
        for rk in self.ranks:
            parent = rk.cache.root_node
            for h in self.chain:
                n = rk.add(None, h, tokens=4096, parent=parent)
                rk.unbacked.remove((n, h))
                self._dev(n, h)
                self.nodes[h].append(n)
                parent = n

    @staticmethod
    def _dev(n, h):
        n.hash_value = [h]
        n.component_data[F].value = torch.zeros(1, dtype=torch.int64)       # KV on the device
        n.component_data[F].host_value = torch.zeros(1, dtype=torch.int64)  # ... and backed (anchor-only path)
        n.component_data[M].value = torch.zeros(1, dtype=torch.int64)       # the device anchor

    def mark(self):
        for rk, end in zip(self.ranks, self.nodes[self.chain[-1]]):
            end.pdflip_park_end = True
            rk.cache._pdflip_park_end_nodes = [end]

    def state(self, h):
        return self.H._state(self.arena, h)

    def anchor_on(self, h):
        """per rank: does the rank's node ``h`` hold its anchor's host copy?"""
        out = []
        for rk in self.ranks:
            n = next(x for x in rk.cache.root_node.children.values() if x.hash_value == [h]) if h in self.off else \
                self.nodes[h][self.ranks.index(rk)]
            out.append(n.component_data[M].host_value is not None)
        return out

    def run_pass(self, ops_cls=_Ops, fails=()):
        def _one(r, gm):
            rk = self.ranks[r]
            ops = ops_cls(rk.cache, rk.mp, busy=False)
            ops.fail_issue = r in fails
            return par.run_pass(ops, gm, r, 3)
        return _run_ranks(3, _one)


@pytest.fixture
def w65(monkeypatch):
    monkeypatch.setattr(_r12, "role", lambda: None)          # pure TP, off Form A: the slot is shared (W65 class)
    monkeypatch.setenv("FLLIPER_PDFLIP_MAMBA_SPILL_LAST_RESORT", "0")
    monkeypatch.delenv("FLLIPER_PDFLIP_PARK_ANCHOR_REARM", raising=False)
    monkeypatch.setattr(UnifiedRadixCache, "_spill_tp_skip_n", 0, raising=False)
    monkeypatch.setattr(_l3s, "secured", lambda arena, slots, backend=None, evicting_ok=False:
                        __import__("numpy").ones(len(list(slots)), dtype=bool))   # the write-behind has run


@needs_gcc
def test_w65_full_arena_rank_local_claim_is_refused_on_every_rank(tmp_path, w65, caplog):
    """The metal's shape before the pass: the END's own claim finds no slot, the spill is skipped on every rank."""
    W = _World(tmp_path, 2, off=2)
    W.mark()
    with caplog.at_level(logging.INFO):
        for rk, end in zip(W.ranks, W.nodes["END"]):
            assert rk.cache._pdflip_mamba_claim(end, rk.mp, "END") is None
    assert W.state("END") != 2
    assert any("SPILL SKIPPED site=park_first tp_size=3" in m and "DEFERRED" in m for m in caplog.messages)


@needs_gcc
def test_group_spill_backs_the_end_on_every_rank(tmp_path, w65, caplog):
    """Part 2: every rank releases the SAME agreed victim (the shallowest off-chain anchor), then every rank claims:
    the END slot is COMPLETE (all three extents written), the victim gone on every rank, the deeper one kept."""
    W = _World(tmp_path, 2, off=2)
    W.mark()
    with caplog.at_level(logging.INFO):
        res, g = W.run_pass()
    assert all(st["issued"] == 1 and st["spilled"] == 1 and st["split"] == 0 for st in res.values()), res
    assert W.state("END") == 2, "the END anchor's slot is COMPLETE: every rank wrote its extent"
    assert W.anchor_on("END") == [True, True, True]
    assert W.anchor_on("old0") == [False, False, False], "the agreed victim left every rank alike"
    assert W.anchor_on("old1") == [True, True, True]
    assert any("PDFLIP PARK-REARM ISSUED depth=8192 class=end spill_victim_depth=512" in m for m in caplog.messages)


@needs_gcc
def test_rearm_after_a_slot_came_back_needs_no_spill(tmp_path, w65):
    """Part 1: the arena gave a slot back (every rank released old1 -- ARENA-DROP freed>0); the next pass re-arms the
    refused END without spilling anything."""
    W = _World(tmp_path, 2, off=2)
    W.mark()
    for rk in W.ranks:
        n = next(x for x in rk.cache.root_node.children.values() if x.hash_value == ["old1"])
        hv = n.component_data[M].host_value
        slots = rk.mp.settled_anchor_slots(hv)
        n.component_data[M].host_value = None
        rk.mp.free(hv)
        rk.mp.drop_unreferenced(slots)
    res, g = W.run_pass()
    assert all(st["issued"] == 1 and st["spilled"] == 0 for st in res.values()), res
    assert W.state("END") == 2 and W.anchor_on("old0") == [True, True, True]


@needs_gcc
def test_dropped_chain_intermediate_is_rearmed_after_the_end(tmp_path, w65):
    """Part 1, class 1: the chain intermediate A was given up KV-only; with room for both, the END goes first, then A
    -- and A's drop flag is cleared on every rank."""
    W = _World(tmp_path, 4, off=2)
    W.mark()
    for a in W.nodes["A"]:
        a.pdflip_park_anchor_dropped = True
    res, g = W.run_pass()
    assert all(st["issued"] == 2 and st["spilled"] == 0 for st in res.values()), res
    assert W.state("END") == 2 and W.state("A") == 2
    assert all(not a.pdflip_park_anchor_dropped for a in W.nodes["A"])


@needs_gcc
def test_split_heals_by_the_late_ranks_join(tmp_path, w65, caplog):
    """Rank 2 cannot write in the first pass -> SPLIT, named; the next pass asks rank 2 alone, its claim JOINS the stem
    ranks 0/1 hold (no room needed, no second spill) and the slot completes."""
    W = _World(tmp_path, 2, off=2)
    W.mark()
    with caplog.at_level(logging.INFO):
        res, g = W.run_pass(fails=(2,))
    assert all(st["split"] == 1 for st in res.values()), res
    assert any("PDFLIP PARK-REARM SPLIT" in m for m in caplog.messages)
    assert W.state("END") != 2
    res, g = W.run_pass()
    assert all(st["issued"] == 1 and st["spilled"] == 0 for st in res.values()), res
    assert W.state("END") == 2 and W.anchor_on("END") == [True, True, True]


@needs_gcc
def test_no_secured_victim_releases_nothing(tmp_path, w65, monkeypatch, caplog):
    """#257: an anchor without its L3 copy is never the victim -- victim=none, nothing released, named."""
    monkeypatch.setattr(_l3s, "secured", lambda arena, slots, backend=None, evicting_ok=False:
                        __import__("numpy").zeros(len(list(slots)), dtype=bool))
    monkeypatch.setattr(_l3s, "request", lambda arena, slots: len(slots))
    W = _World(tmp_path, 2, off=2)
    W.mark()
    with caplog.at_level(logging.INFO):
        res, g = W.run_pass()
    assert all(st["issued"] == 0 and st["spilled"] == 0 for st in res.values())
    assert W.anchor_on("old0") == [True, True, True] and W.anchor_on("old1") == [True, True, True]
    assert any("victim=none" in m for m in caplog.messages)


@needs_gcc
def test_switch_off_is_the_old_skip(tmp_path, w65, monkeypatch):
    monkeypatch.setenv("FLLIPER_PDFLIP_PARK_ANCHOR_REARM", "0")
    W = _World(tmp_path, 2, off=2)
    W.mark()
    for rk in W.ranks:
        assert UnifiedRadixCache.pdflip_park_rearm_group(rk.cache) == {}
    assert W.state("END") != 2


# ---- mutants (both danger directions) --------------------------------------------------------------------------------

@needs_gcc
def test_mutant_rank_local_victim_choice_is_caught(tmp_path, w65, monkeypatch):
    """RANK DISUNITY: the victim chosen rank-locally (the same collectives, the local minimum instead of the agreed
    one). Rank 1's shallowest anchor is pinned (a write in flight), so the ranks release DIFFERENT victims: no slot
    frees and the ranks' anchors part -- the invariant of the green test fails."""
    orig = par.agree_min_common

    def _local(keys, gm, rounds=par.AGREE_ROUNDS):
        _k, r = orig(keys, gm, rounds)
        ks = sorted(keys)
        return (ks[0] if ks else None), r

    W = _World(tmp_path, 2, off=2)
    W.mark()
    n1 = next(x for x in W.ranks[1].cache.root_node.children.values() if x.hash_value == ["old0"])
    n1.write_through_pending_id = 99         # rank 1 cannot offer old0 this pass
    W.run_pass()                            # the agreed form: old1 is common -> released on every rank
    assert W.anchor_on("old1") == [False, False, False] and W.anchor_on("old0") == [True, True, True]
    W2 = _World(tmp_path / "m", 2, off=2)
    W2.mark()
    n1 = next(x for x in W2.ranks[1].cache.root_node.children.values() if x.hash_value == ["old0"])
    n1.write_through_pending_id = 99
    monkeypatch.setattr(par, "agree_min_common", _local)
    W2.run_pass()
    with pytest.raises(AssertionError):     # the green invariant: every anchor rank-equal
        for h in W2.off:
            assert W2.anchor_on(h) in ([True] * 3, [False] * 3), (h, W2.anchor_on(h))


@needs_gcc
def test_mutant_unsecured_victim_is_caught(tmp_path, w65, monkeypatch):
    """LOSS: the victim filter without the L3 proof -- an anchor with no other copy is released. The invariant of
    test_no_secured_victim_releases_nothing (nothing released) fails."""
    monkeypatch.setattr(_l3s, "secured", lambda arena, slots, backend=None, evicting_ok=False:
                        __import__("numpy").zeros(len(list(slots)), dtype=bool))
    monkeypatch.setattr(_l3s, "request", lambda arena, slots: len(slots))
    orig_v = par.TreeOps.victims

    def _no_proof(self):
        saved = _l3s.secured
        _l3s.secured = lambda arena, slots, backend=None, evicting_ok=False: \
            __import__("numpy").ones(len(list(slots)), dtype=bool)
        try:
            return orig_v(self)
        finally:
            _l3s.secured = saved

    monkeypatch.setattr(par.TreeOps, "victims", _no_proof)
    W = _World(tmp_path, 2, off=2)
    W.mark()
    W.run_pass()
    with pytest.raises(AssertionError):
        assert W.anchor_on("old0") == [True, True, True] and W.anchor_on("old1") == [True, True, True]


# ---- the rank-local re-arm, the MARK contract, the W65 message ------------------------------------------------------

def _stub_tree(tp=1):
    t = types.SimpleNamespace(tp_world_size=tp, root_node=None)
    t.pdflip_node_depth = lambda n: 4096
    t._s104_byteless_anchor_rank = lambda: False
    return t


def test_local_rearm_on_a_single_rank_once_room_is_free(monkeypatch, caplog):
    monkeypatch.setattr(_r12, "role", lambda: None)
    t = _stub_tree(tp=1)
    t._pdflip_mamba_pool = lambda: types.SimpleNamespace(claim_needs_room=lambda h: False)
    node = types.SimpleNamespace(pdflip_park_anchor_dropped=True, hash_value=["h"], id=7)
    with caplog.at_level(logging.INFO):
        assert UnifiedRadixCache._pdflip_park_rearm_local(t, node) is True
    assert node.pdflip_park_anchor_dropped is False
    assert any("PDFLIP PARK-REARM LOCAL" in m for m in caplog.messages)
    node.pdflip_park_anchor_dropped = True
    t._pdflip_mamba_pool = lambda: types.SimpleNamespace(claim_needs_room=lambda h: True)
    assert UnifiedRadixCache._pdflip_park_rearm_local(t, node) is False and node.pdflip_park_anchor_dropped


def test_local_rearm_never_on_a_shared_tp_group(monkeypatch):
    monkeypatch.setattr(_r12, "role", lambda: None)
    t = _stub_tree(tp=3)
    t._pdflip_mamba_pool = lambda: types.SimpleNamespace(claim_needs_room=lambda h: False)
    node = types.SimpleNamespace(pdflip_park_anchor_dropped=True, hash_value=["h"], id=7)
    assert UnifiedRadixCache._pdflip_park_rearm_local(t, node) is False
    monkeypatch.setattr(_r12, "role", lambda: "host")        # Form A host: its claim is the group's
    assert UnifiedRadixCache._pdflip_park_rearm_local(t, node) is True


def test_local_rearm_switch_off(monkeypatch):
    monkeypatch.setenv("FLLIPER_PDFLIP_PARK_ANCHOR_REARM", "0")
    t = _stub_tree(tp=1)
    t._pdflip_mamba_pool = lambda: types.SimpleNamespace(claim_needs_room=lambda h: False)
    node = types.SimpleNamespace(pdflip_park_anchor_dropped=True, hash_value=["h"], id=7)
    assert UnifiedRadixCache._pdflip_park_rearm_local(t, node) is False


def test_backed_line_names_the_park_end_once(caplog):
    t = _stub_tree()
    end = UnifiedTreeNode((F, M))
    end.pdflip_park_end = True
    end.pdflip_park_rid = "pdflip-4-66"
    other = UnifiedTreeNode((F, M))
    with caplog.at_level(logging.INFO):
        UnifiedRadixCache._pdflip_park_note_backed(t, end, "full")
        UnifiedRadixCache._pdflip_park_note_backed(t, end, "full")
        UnifiedRadixCache._pdflip_park_note_backed(t, other, "full")
    lines = [m for m in caplog.messages if "PDFLIP PARK-END-ANCHOR BACKED" in m]
    assert len(lines) == 1 and "rid=pdflip-4-66" in lines[0] and "depth=4096 via=full" in lines[0]


def test_mark_names_a_priority_and_a_copy_already_there(monkeypatch, caplog):
    from flliper.srt.pdflip import mamba_arena_displace as mad

    monkeypatch.delenv("FLLIPER_PDFLIP_PARK_END_ANCHOR_FIRST", raising=False)
    end = UnifiedTreeNode((F, M))
    end.component_data[M].host_value = torch.zeros(1)
    t = _stub_tree()
    t.__dict__["tree_components"] = (F, M)
    t.pdflip_park_first_on = lambda: True
    t.pdflip_lane_rank_on = lambda: False
    t._pdflip_chain_nodes_strict = lambda key: [end]
    req = types.SimpleNamespace(rid="pdflip-4-66")
    setattr(req, mad.PARK_REQ_ATTR, True)
    with caplog.at_level(logging.INFO):
        UnifiedRadixCache._pdflip_mark_park_end(t, req, None)
    mark = [m for m in caplog.messages if "PARK-END-ANCHOR-FIRST MARK" in m]
    assert mark and "rid=pdflip-4-66 node=" in mark[0] and "prio=claim" in mark[0] and "not a backup" in mark[0]
    assert any("PARK-END-ANCHOR BACKED" in m and "via=before_mark" in m for m in caplog.messages)
