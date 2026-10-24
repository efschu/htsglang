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

REVIEW 24.10. (review-mamba-arena-rearm-1024.md, JA MIT AUFLAGEN) -- red on 9df38c024d, green after:
A1 every pass collective carries a [TAG,-TAG] pair (a rank in another collective = HiCacheCollectiveDesyncError on every
   rank), anything raised after the first collective is re-raised, the scheduler hook re-raises HiCacheCollectiveError;
A2 "issued" = THIS rank holds mamba rows for the target (a KV-only write is REFUSED), the drop flag is cleared at the
   ACK only, class 1 is located by the tree's structure -- with the REAL TreeOps.issue (anchor-only write path);
A3 a split is joined in the SAME pass, else W203 ParkRearmDivergence on every rank; the release has its own vote, a
   release on part of the group is W203 on every rank. Code mutants (the module source mutated and re-executed) prove
   each guard load-bearing.

Hermetic: the real shared arena (arena.c, gcc) with three rank extents and the real UnifiedRadixCache claim /
release of the H19 harness (test_pdflip_mamba_arena_displace_h19.py); no torch.distributed, no GPU.
"""
from __future__ import annotations

import importlib.util
import sys
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
    def __init__(self, n, timeout=20):
        self.n = n
        self.b = threading.Barrier(n, timeout=timeout)
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




def _run_all(n, fn):
    """fn(rank, group_max) on n threads; returns ({rank: result}, {rank: exception}) -- every rank's outcome. A rank
    that leaves early is NOT announced to the others (no barrier abort -- an abort would also break peers that are
    just waking from a completed collective): they meet the collective's bound (3 s, the analogue of bounded_wait)."""
    g = _Group(n, timeout=3)
    res, err = {}, {}

    def _w(r):
        try:
            res[r] = fn(r, g.max_for(r))
        except BaseException as exc:  # noqa: BLE001
            err[r] = exc

    th = [threading.Thread(target=_w, args=(r,)) for r in range(n)]
    for t in th:
        t.start()
    for t in th:
        t.join(30)
    return res, err


def _mutant(old, new):
    """The pass module with ``old`` replaced by ``new`` in its SOURCE, executed as a separate module (a code mutant)."""
    src = open(par.__file__, encoding="utf-8").read()
    assert src.count(old) == 1, old
    mod = types.ModuleType("park_anchor_rearm_mutant")
    mod.__file__ = par.__file__
    exec(compile(src.replace(old, new), par.__file__, "exec"), mod.__dict__)
    return mod


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

    def victims(self, node=None):
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


# ---- A1: self-identifying collectives, errors re-raised ----------------------------------------------------------------

def test_a1_a_rank_in_another_collective_is_a_named_desync_on_every_rank():
    """Equal-sized foreign payloads (gloo pairs them silently): rank 0 is in the STATE vote, ranks 1/2 in the ISSUE
    vote with the same length -- every rank raises HiCacheCollectiveDesyncError, none continues."""
    def _one(r, gm):
        f = par.tagged(gm, par.PH_STATE if r == 0 else par.PH_ISSUE)
        return f([1, 0, 0])
    res, err = _run_all(3, _one)
    assert res == {} and set(err) == {0, 1, 2}
    assert all(isinstance(e, par.HiCacheCollectiveDesyncError) for e in err.values()), err


def test_a1_a_rank_whose_collective_raises_stops_every_rank():
    """One rank's collective raises (a bounded-wait timeout): no rank runs on past it."""
    from flliper.srt.mem_cache.hicache_collective import HiCacheCollectiveTimeoutError

    ops = {r: _FakeOps([1]) for r in range(3)}

    def _one(r, gm):
        def g(v):
            if r == 2:
                raise HiCacheCollectiveTimeoutError("rank 2 wedged")
            return gm(v)
        return par.run_pass(ops[r], g, r, 3)
    res, err = _run_all(3, _one)
    assert res == {} and set(err) == {0, 1, 2}


def test_a1_tree_reraises_after_the_first_collective(monkeypatch):
    """The tree method: a collective error re-raised as itself, any other error after the first collective as W203;
    before the first collective a rank-local error is logged and no pass runs."""
    t = types.SimpleNamespace(tp_world_size=3, tree_components=(F, M), cache_controller=object())
    monkeypatch.setattr(_r12, "role", lambda: None)
    t._pdflip_mamba_pool = lambda: None
    t.hicache_group_max = lambda v, label: list(v)
    monkeypatch.setattr(UnifiedRadixCache, "_park_rearm_group_rank", lambda self: (0, 3))
    calls = {"n": 0}

    def _bug(ops, gm, rank, world, log=None, allow_class1=True):
        gm([par.TAG_BASE + 1, -(par.TAG_BASE + 1), 0])
        raise KeyError("a bug after the gather")
    monkeypatch.setattr(par, "run_pass", _bug)
    with pytest.raises(par.ParkRearmDivergence):
        UnifiedRadixCache.pdflip_park_rearm_group(t)

    def _pre(ops, gm, rank, world, log=None, allow_class1=True):
        raise KeyError("before any collective")
    monkeypatch.setattr(par, "run_pass", _pre)
    assert UnifiedRadixCache.pdflip_park_rearm_group(t) == {}

    def _coll(ops, gm, rank, world, log=None, allow_class1=True):
        raise par.HiCacheCollectiveDesyncError("foreign tag")
    monkeypatch.setattr(par, "run_pass", _coll)
    with pytest.raises(par.HiCacheCollectiveDesyncError):
        UnifiedRadixCache.pdflip_park_rearm_group(t)


def test_a1_the_scheduler_hook_reraises_collective_errors():
    import inspect

    from flliper.srt.managers.scheduler import Scheduler

    src = inspect.getsource(Scheduler.flush_cache)
    i = src.index("PARK-REARM-1024")
    seg = src[i:i + 2600]
    assert "except _HCE:\n                    raise" in seg
    assert seg.index("except _HCE:") < seg.index("except Exception:")


def test_a1_mutant_without_the_tag_check_lets_the_ranks_run_on():
    """MUTANT (A1): the tag check removed -- the same misaligned collectives return values and every rank continues
    with a reduce of foreign payloads. The green test above raises; here nothing does: the check is load-bearing."""
    mut = _mutant("        if hi != tag or lo != tag:\n", "        if False:\n")

    def _one(r, gm):
        f = mut.tagged(gm, mut.PH_STATE if r == 0 else mut.PH_ISSUE)
        return f([1, 0, 0])
    res, err = _run_all(3, _one)
    assert err == {} and set(res) == {0, 1, 2}


# ---- the real shared arena (H19 harness) and the REAL TreeOps.issue ----------------------------------------------------

def _h19():
    spec = importlib.util.spec_from_file_location(
        "_h19_1024", os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                  "test_pdflip_mamba_arena_displace_h19.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


class _Ctl:
    """The cache controller's write: issued (1) unless this rank is told to refuse it (None, as a full host ring)."""

    def __init__(self):
        self.refuse = 0          # writes left to refuse
        self._pdflip_last_write_refusal = "test"

    def write(self, *a, **k):
        if self.refuse > 0:
            self.refuse -= 1
            return None
        return 1


class _Comp:
    """The mamba component's backup transfer (one anchor row) and its commit into the tree's host value."""

    def build_hicache_transfers(self, node, phase):
        return [types.SimpleNamespace(host_indices=None, name="mamba")]

    def commit_hicache_transfer(self, node, phase, transfers=None):
        node.component_data[M].host_value = transfers[0].host_indices


class _Ops(par.TreeOps):
    """The REAL adapter. Fault injection only: ``raise_issue`` (raise in issue n times), ``raise_release``."""

    raise_issue = 0
    raise_release = False

    def issue(self, n):
        if self.raise_issue > 0:
            type(self).raise_issue_left[self.tree] = type(self).raise_issue_left.get(self.tree, self.raise_issue) - 1
            if type(self).raise_issue_left[self.tree] >= 0:
                raise RuntimeError("issue defect on one rank")
        return super().issue(n)

    def release(self, victim):
        if self.raise_release:
            raise RuntimeError("release defect on one rank")
        return super().release(victim)


class _World:
    """A D tree on 3 ranks over ONE shared arena: off-chain anchors (untagged, settled, on L3) fill the arena; the
    parked chain A -> END (KV backed, anchor on the device only). The anchor-only write of the real cache runs."""

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
            c.cache_controller = _Ctl()
            c.components = {M: _Comp()}
            c._pdflip_mamba_pool = (lambda mp: (lambda: mp))(rk.mp)
            c._pdflip_anchor_only_ids = set()
            c.ongoing_write_through = {}
            c.inc_lock_ref = lambda n: types.SimpleNamespace(to_dec_params=lambda: None)
            c._1421_refused = lambda why, n: None
            c._note_mamba_pin_skipped = lambda: None
            c._s104_byteless_anchor_rank = lambda: False
            c._pdflip_direct_pool = lambda: None
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

    def node(self, r, h):
        if h in self.off:
            return next(x for x in self.ranks[r].cache.root_node.children.values() if x.hash_value == [h])
        return self.nodes[h][r]

    def anchor_on(self, h):
        """per rank: does the rank's node ``h`` hold its anchor's host copy?"""
        return [self.node(r, h).component_data[M].host_value is not None for r in range(3)]

    def ack(self, h):
        """the write-through acks of node ``h`` on every rank that has one in flight (the real ack helper)"""
        for r, rk in enumerate(self.ranks):
            n, c = self.node(r, h), rk.cache
            rows = c._pdflip_direct_mamba_rows.pop(n.id, None)
            if rows is None:
                continue
            rk.mp.complete_write(rows)
            n.write_through_pending_id = None
            c.ongoing_write_through.pop(n.id, None)
            c._pdflip_anchor_only_ids.discard(n.id)
            UnifiedRadixCache._pdflip_park_anchor_acked(c, n, "anchor_only")

    def refuse_claim(self, r, times=1):
        """rank ``r``'s next ``times`` anchor claims are refused BEFORE any slot is touched (the arena's no-room answer
        on that rank alone -- deterministic, no claim to abort)."""
        c = self.ranks[r].cache
        orig = c._pdflip_mamba_claim
        left = [times]

        def _claim(node, mp, last_hash):
            if left[0] > 0:
                left[0] -= 1
                return None
            return orig(node, mp, last_hash)
        c._pdflip_mamba_claim = _claim

    def run_pass(self, ops_cls=_Ops, mod=None, inject=None):
        mod = mod or par
        inject = inject or {}

        def _one(r, gm):
            rk = self.ranks[r]
            ops = ops_cls(rk.cache, rk.mp, busy=False)
            for k, v in inject.get(r, {}).items():
                setattr(ops, k, v)
            return mod.run_pass(ops, gm, r, 3)
        return _run_all(3, _one)


@pytest.fixture
def w65(monkeypatch):
    monkeypatch.setattr(_r12, "role", lambda: None)          # pure TP, off Form A: the slot is shared (W65 class)
    monkeypatch.setenv("FLLIPER_PDFLIP_MAMBA_SPILL_LAST_RESORT", "0")
    monkeypatch.delenv("FLLIPER_PDFLIP_PARK_ANCHOR_REARM", raising=False)
    monkeypatch.delenv("FLLIPER_PDFLIP_ANCHOR_ONLY_BACKUP", raising=False)
    monkeypatch.setattr(UnifiedRadixCache, "_spill_tp_skip_n", 0, raising=False)
    monkeypatch.setattr(_Ops, "raise_issue_left", {}, raising=False)
    monkeypatch.setattr(_l3s, "secured", lambda arena, slots, backend=None, evicting_ok=False:
                        __import__("numpy").ones(len(list(slots)), dtype=bool))   # the write-behind has run


def _ok(res, err, **want):
    assert err == {}, err
    for st in res.values():
        for k, v in want.items():
            assert st.get(k, 0) == v, (k, res)


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
    """Part 2: every rank releases the SAME agreed victim (the shallowest off-chain anchor), then every rank claims and
    writes (the real anchor-only write); after the acks the END slot is COMPLETE, the victim gone on every rank, the
    deeper one kept."""
    W = _World(tmp_path, 2, off=2)
    W.mark()
    with caplog.at_level(logging.INFO):
        res, err = W.run_pass()
    _ok(res, err, issued=1, spilled=1, split=0, healed=0)
    assert W.anchor_on("END") == [True, True, True]
    W.ack("END")
    assert W.state("END") == 2, "the END anchor's slot is COMPLETE: every rank wrote its extent"
    assert W.anchor_on("old0") == [False, False, False], "the agreed victim left every rank alike"
    assert W.anchor_on("old1") == [True, True, True]
    assert any("PDFLIP PARK-REARM ISSUED depth=8192 class=end spill_victim_depth=512" in m for m in caplog.messages)
    assert sum("PARK-END-ANCHOR BACKED" in m and "slot=COMPLETE" in m for m in caplog.messages) >= 1


@needs_gcc
def test_rearm_after_a_slot_came_back_needs_no_spill(tmp_path, w65):
    """Part 1: the arena gave a slot back (every rank released old1 -- ARENA-DROP freed>0); the next pass re-arms the
    refused END without spilling anything."""
    W = _World(tmp_path, 2, off=2)
    W.mark()
    for r, rk in enumerate(W.ranks):
        n = W.node(r, "old1")
        hv = n.component_data[M].host_value
        slots = rk.mp.settled_anchor_slots(hv)
        n.component_data[M].host_value = None
        rk.mp.free(hv)
        rk.mp.drop_unreferenced(slots)
    res, err = W.run_pass()
    _ok(res, err, issued=1, spilled=0)
    W.ack("END")
    assert W.state("END") == 2 and W.anchor_on("old0") == [True, True, True]


@needs_gcc
def test_dropped_chain_intermediate_is_rearmed_and_its_flag_cleared_at_the_ack(tmp_path, w65):
    """Part 1, class 1, the REAL issue: A was given up KV-only; with room for both, the END goes first, then A. The drop
    flag survives the write (review A2) and is cleared at the ACK on every rank."""
    W = _World(tmp_path, 4, off=2)
    W.mark()
    for a in W.nodes["A"]:
        a.pdflip_park_anchor_dropped = True
    res, err = W.run_pass()
    _ok(res, err, issued=2, spilled=0)
    assert all(a.pdflip_park_anchor_dropped for a in W.nodes["A"]), "cleared only at the ack"
    W.ack("END")
    W.ack("A")
    assert W.state("END") == 2 and W.state("A") == 2
    assert all(not a.pdflip_park_anchor_dropped for a in W.nodes["A"])


@needs_gcc
def test_a2_class1_refused_write_is_retried_next_pass(tmp_path, w65, caplog):
    """A2: every rank's anchor claim of A is refused (no room on any rank at the claim) -> REFUSED, the drop flag stays,
    the next pass finds A again (structure-located) and backs it. Red before: the flag was cleared before the write,
    the target never came back."""
    W = _World(tmp_path, 4, off=2)
    W.mark()
    for rk, end in zip(W.ranks, W.nodes["END"]):
        end.component_data[M].host_value = torch.zeros(1, dtype=torch.int64)   # the END stands: A alone is asked
    for a in W.nodes["A"]:
        a.pdflip_park_anchor_dropped = True
    for r in range(3):
        W.refuse_claim(r)
    with caplog.at_level(logging.INFO):
        res, err = W.run_pass()
    _ok(res, err, issued=0, refused=1)
    assert all(a.pdflip_park_anchor_dropped for a in W.nodes["A"])
    assert W.anchor_on("A") == [False, False, False]
    res, err = W.run_pass()
    _ok(res, err, issued=1)
    W.ack("A")
    assert W.state("A") == 2 and all(not a.pdflip_park_anchor_dropped for a in W.nodes["A"])


@needs_gcc
def test_a2_kv_only_fallback_is_refused_not_issued(tmp_path, w65, caplog):
    """A2: A's KV is not backed; its full backup goes KV-only (the PARK-END-ANCHOR-FIRST fallback: KV written, anchor
    dropped). That is no anchor copy -> REFUSED on every rank, never ISSUED."""
    W = _World(tmp_path, 4, off=2)
    W.mark()
    for r, rk in enumerate(W.ranks):
        W.nodes["END"][r].component_data[M].host_value = torch.zeros(1, dtype=torch.int64)
        a = W.nodes["A"][r]
        a.pdflip_park_anchor_dropped = True
        a.component_data[F].host_value = None              # KV on the device only

        def _wb(n, write_back=False, kv_only_if_mamba_refused=False, _a=a):
            n.component_data[F].host_value = torch.zeros(1, dtype=torch.int64)   # the KV went out ...
            n.pdflip_park_anchor_dropped = True                                  # ... the anchor did not
            return 1
        rk.cache.write_backup = _wb
    with caplog.at_level(logging.INFO):
        res, err = W.run_pass()
    _ok(res, err, issued=0, refused=1)
    assert not any("PDFLIP PARK-REARM ISSUED" in m for m in caplog.messages)


@needs_gcc
def test_a2_mutant_issue_counting_any_write_is_caught(tmp_path, w65):
    """MUTANT (A2): issue answers "something was written" (the old ``got > 0``) -- the KV-only fallback counts as
    ISSUED again; the invariant of the green test (REFUSED, no ISSUED) fails."""
    class _Old(_Ops):
        def issue(self, n):
            t = self.tree
            t._pdflip_park_rearm_active = n
            try:
                got = t.write_backup_anchor_only(n) if self._kv_backed(n) else t.write_backup(n)
            finally:
                t._pdflip_park_rearm_active = None
            return int(got or 0) > 0

    W = _World(tmp_path, 4, off=2)
    W.mark()
    for r, rk in enumerate(W.ranks):
        W.nodes["END"][r].component_data[M].host_value = torch.zeros(1, dtype=torch.int64)
        a = W.nodes["A"][r]
        a.pdflip_park_anchor_dropped = True
        a.component_data[F].host_value = None

        def _wb(n, write_back=False, kv_only_if_mamba_refused=False):
            n.component_data[F].host_value = torch.zeros(1, dtype=torch.int64)
            return 1
        rk.cache.write_backup = _wb
    res, err = W.run_pass(ops_cls=_Old)
    with pytest.raises(AssertionError):
        _ok(res, err, issued=0, refused=1)


@needs_gcc
def test_a3_split_is_joined_in_the_same_pass(tmp_path, w65, caplog):
    """A3a: rank 2's first claim of A is refused while ranks 0/1 claim the stem -- the same pass asks rank 2 again, its
    claim JOINS the stem (no room needed), SPLIT HEALED; after the acks the slot is COMPLETE. Red before: SPLIT named
    and left to the next pass (and for class 1 never healed)."""
    W = _World(tmp_path, 4, off=2)
    W.mark()
    for r in range(3):
        W.nodes["END"][r].component_data[M].host_value = torch.zeros(1, dtype=torch.int64)
    for a in W.nodes["A"]:
        a.pdflip_park_anchor_dropped = True
    W.refuse_claim(2)
    with caplog.at_level(logging.INFO):
        res, err = W.run_pass()
    _ok(res, err, issued=1, healed=1, split=0)
    assert W.anchor_on("A") == [True, True, True]
    W.ack("A")
    assert W.state("A") == 2
    assert any("SPLIT HEALED" in m for m in caplog.messages)


@needs_gcc
def test_a3_issue_raising_once_on_one_rank_after_the_spill_heals(tmp_path, w65):
    W = _World(tmp_path, 2, off=2)
    W.mark()
    res, err = W.run_pass(inject={1: {"raise_issue": 1}})
    _ok(res, err, issued=1, spilled=1, healed=1)
    W.ack("END")
    assert W.state("END") == 2 and W.anchor_on("old0") == [False, False, False]


@needs_gcc
def test_a3_issue_raising_always_on_one_rank_is_w203_on_every_rank(tmp_path, w65):
    W = _World(tmp_path, 2, off=2)
    W.mark()
    res, err = W.run_pass(inject={1: {"raise_issue": 99}})
    assert res == {} and set(err) == {0, 1, 2}
    assert all(isinstance(e, (par.ParkRearmDivergence, threading.BrokenBarrierError)) for e in err.values()), err
    assert sum(isinstance(e, par.ParkRearmDivergence) for e in err.values()) == 3, err


@needs_gcc
def test_a3_release_raising_on_one_rank_is_w203_on_every_rank(tmp_path, w65):
    """A3b: the agreed victim is released on ranks 0/2, rank 1's release raises -- the release vote is split: W203 on
    every rank (not "freed=0, another process still references it")."""
    W = _World(tmp_path, 2, off=2)
    W.mark()
    res, err = W.run_pass(inject={1: {"raise_release": True}})
    assert res == {} and set(err) == {0, 1, 2}
    assert all(isinstance(e, par.ParkRearmDivergence) for e in err.values()), err


@needs_gcc
def test_a3_mutant_without_the_release_vote_check_parts_the_ranks(tmp_path, w65):
    """MUTANT (A3b): the split check of the release vote removed -- the pass ends quietly and the victim's anchor is
    gone on two ranks of three: the rank-equality invariant fails."""
    mut = _mutant("            if any_bad and any_ok:\n", "            if False:\n")
    W = _World(tmp_path, 2, off=2)
    W.mark()
    res, err = W.run_pass(mod=mut, inject={1: {"raise_release": True}})
    assert err == {}
    with pytest.raises(AssertionError):
        assert W.anchor_on("old0") in ([True] * 3, [False] * 3), W.anchor_on("old0")


@needs_gcc
def test_a3_mutant_without_the_same_pass_join_is_caught(tmp_path, w65):
    """MUTANT (A3a): the same-pass join removed -- the one-shot refusal on rank 2 is no longer healed; the green
    invariant (HEALED, anchor on every rank) fails (the mutant stops the group instead)."""
    mut = _mutant("            if needs and not issued:\n                issued", "            if False:\n                issued")
    W = _World(tmp_path, 4, off=2)
    W.mark()
    for r in range(3):
        W.nodes["END"][r].component_data[M].host_value = torch.zeros(1, dtype=torch.int64)
    for a in W.nodes["A"]:
        a.pdflip_park_anchor_dropped = True
    W.refuse_claim(2)
    res, err = W.run_pass(mod=mut)
    with pytest.raises(AssertionError):
        _ok(res, err, issued=1, healed=1)
        assert W.anchor_on("A") == [True, True, True]


@needs_gcc
def test_no_secured_victim_releases_nothing(tmp_path, w65, monkeypatch, caplog):
    """#257: an anchor without its L3 copy is never the victim -- victim=none, nothing released, named."""
    monkeypatch.setattr(_l3s, "secured", lambda arena, slots, backend=None, evicting_ok=False:
                        __import__("numpy").zeros(len(list(slots)), dtype=bool))
    monkeypatch.setattr(_l3s, "request", lambda arena, slots: len(slots))
    W = _World(tmp_path, 2, off=2)
    W.mark()
    with caplog.at_level(logging.INFO):
        res, err = W.run_pass()
    _ok(res, err, issued=0, spilled=0)
    assert W.anchor_on("old0") == [True, True, True] and W.anchor_on("old1") == [True, True, True]
    assert any("victim=none" in m for m in caplog.messages)


@needs_gcc
def test_victim_keys_shared_by_two_anchors_are_no_victims(tmp_path, w65, monkeypatch):
    """S6: two anchors with one key (a crc collision at one depth) -- the key is dropped on every rank, no rank-dependent
    tie decides the victim."""
    monkeypatch.setattr(par, "victim_key", lambda depth, h: 7)
    W = _World(tmp_path, 2, off=2)
    W.mark()
    ops = _Ops(W.ranks[0].cache, W.ranks[0].mp, busy=False)
    assert ops.victims(W.nodes["END"][0]) == {}


@needs_gcc
def test_switch_off_is_the_old_skip(tmp_path, w65, monkeypatch):
    monkeypatch.setenv("FLLIPER_PDFLIP_PARK_ANCHOR_REARM", "0")
    W = _World(tmp_path, 2, off=2)
    W.mark()
    for rk in W.ranks:
        assert UnifiedRadixCache.pdflip_park_rearm_group(rk.cache) == {}
    assert W.state("END") != 2


# ---- mutants of the first round (both danger directions) -----------------------------------------------------------

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
    W.node(1, "old0").write_through_pending_id = 99   # rank 1 cannot offer old0 this pass
    W.run_pass()                                       # the agreed form: old1 is common -> released on every rank
    assert W.anchor_on("old1") == [False, False, False] and W.anchor_on("old0") == [True, True, True]
    (tmp_path / "m").mkdir()
    W2 = _World(tmp_path / "m", 2, off=2)
    W2.mark()
    W2.node(1, "old0").write_through_pending_id = 99
    monkeypatch.setattr(par, "agree_min_common", _local)
    W2.run_pass()
    with pytest.raises(AssertionError):     # the green invariant: every anchor rank-equal
        for h in W2.off:
            assert W2.anchor_on(h) in ([True] * 3, [False] * 3), (h, W2.anchor_on(h))


@needs_gcc
def test_mutant_unsecured_victim_is_caught(tmp_path, w65, monkeypatch):
    """LOSS: the victim filter without the L3 proof -- an anchor with no other copy is released. The invariant of
    test_no_secured_victim_releases_nothing (nothing released) fails. (The proof answers "secured" to the victim
    filter only -- thread-safe, by the caller's frame -- and "not secured" everywhere else.)"""
    import numpy as np

    def _proof(arena, slots, backend=None, evicting_ok=False):
        n = len(list(slots))
        return np.ones(n, dtype=bool) if sys._getframe(1).f_code.co_name == "victims" else np.zeros(n, dtype=bool)

    monkeypatch.setattr(_l3s, "secured", _proof)
    monkeypatch.setattr(_l3s, "request", lambda arena, slots: len(slots))
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
    """S7: the predicate admits the candidate while room is free and does NOT clear the flag (the ack does)."""
    monkeypatch.setattr(_r12, "role", lambda: None)
    t = _stub_tree(tp=1)
    t._pdflip_mamba_pool = lambda: types.SimpleNamespace(claim_needs_room=lambda h: False)
    node = types.SimpleNamespace(pdflip_park_anchor_dropped=True, hash_value=["h"], id=7)
    with caplog.at_level(logging.INFO):
        assert UnifiedRadixCache._pdflip_park_rearm_local(t, node) is True
    assert node.pdflip_park_anchor_dropped is True, "cleared at the ack, never in the predicate"
    assert any("PDFLIP PARK-REARM LOCAL" in m for m in caplog.messages)
    t._pdflip_mamba_pool = lambda: types.SimpleNamespace(claim_needs_room=lambda h: True)
    assert UnifiedRadixCache._pdflip_park_rearm_local(t, node) is False


def test_the_ack_clears_the_drop_flag():
    t = _stub_tree()
    node = UnifiedTreeNode((F, M))
    node.pdflip_park_anchor_dropped = True
    UnifiedRadixCache._pdflip_park_anchor_acked(t, node, "anchor_only")
    assert node.pdflip_park_anchor_dropped is False


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


def test_backed_line_names_the_park_end_once_and_its_slot_state(caplog):
    t = _stub_tree()
    t._pdflip_mamba_pool = lambda: types.SimpleNamespace(anchor_held=lambda h: False)
    end = UnifiedTreeNode((F, M))
    end.pdflip_park_end = True
    end.pdflip_park_rid = "pdflip-4-66"
    end.hash_value = ["h"]
    other = UnifiedTreeNode((F, M))
    with caplog.at_level(logging.INFO):
        UnifiedRadixCache._pdflip_park_note_backed(t, end, "full")
        UnifiedRadixCache._pdflip_park_note_backed(t, end, "full")
        UnifiedRadixCache._pdflip_park_note_backed(t, other, "full")
    lines = [m for m in caplog.messages if "PDFLIP PARK-END-ANCHOR BACKED" in m]
    assert len(lines) == 1 and "rid=pdflip-4-66" in lines[0] and "depth=4096 via=full slot=partial" in lines[0]


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


def test_direct_claim_during_a_rearm_gives_the_slot_to_the_target():
    """During the issue of a re-arm target, every OTHER un-backed park chain intermediate of the write (write_backup's
    parent recursion) goes KV-only without asking the arena -- the one slot the pass made is the target's; the target
    itself (an END, or a class-1 intermediate) claims normally. Placed before the arena claim."""
    import inspect

    src = inspect.getsource(UnifiedRadixCache._pdflip_direct_claim)
    i = src.index("PARK-REARM-1024: the group-agreed re-arm")
    seg = src[i - 500:i + 700]
    assert "node is not _prr_target" in seg and 'not getattr(node, "pdflip_park_end", False)' in seg
    assert "comp_xfers.pop(mct, None)" in seg and "node.pdflip_park_anchor_dropped = True" in seg
    assert i < src.index("mrows = self._pdflip_mamba_claim(node, mp, hashes[-1])")
