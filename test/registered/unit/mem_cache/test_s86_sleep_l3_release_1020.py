"""S86 (rc19 boot ..._10e9b2b492_1019_230333, D group TP=3 Form A): a park-end chain is never dropped un-backed by
the sleep because the trees' own references fill the KV arena.

THE METAL (D log, sleep 05:37:22-23): ``ARENA-REF-HOLDERS ... tree=6409 queue=0`` (TP1; TP0 tree=6126 queue=0),
``complete=6467 pinned=6409`` -- no queue reference (S83's class is 0 here), the trees alone. ``#1427 ARENA-DROP
need=47 freed=0 stages=i:0,ii:0,iii:0`` (dozens per rank), ``#1421 BACKUP-REFUSED why=arena_claim`` then
``parent_unbacked`` down the chains, ``W3-ARENA FORMA-REQUEST need=53 released_pages=0`` (the spill's SPILL events
need TP0's next broadcast; the flush has none and the reset clears the ledger), ``#1470 FLUSH-PUBLISH
ARENA-FULL-STOP round=1 unbacked=19 arena_refused=19``, then ``#1470 FLUSH-UNBACKED site=flush nodes=17
tokens=13632`` with the park-end nodes 6498 (pdflip-112-2708, depth 40960), 6495 (-2717, 93952), 6497 (-2712,
56576). The pinned pages had their L3 copies: the boot freed arena slots with ``l3=on_disk`` and wrote 164 pages
in total (``dropped_without_l3=0``); the reset gave the references back moments later (``pinned=531``).

THE FIX (``pdflip/sleep_l3_release.py``, called by ``WeightUpdater._pdflip_sleep_flush`` on group D before the
flush): every rank vetoes the slots it must keep (no proven L3 copy, write in flight, host lock, host-only, ...),
the group MAX of the vetoes decides, every rank lets go of the same slots, a digest closes it -- the #1470 claim
then finds unreferenced slots and the chain is written.

Hermetic: the real C arena on a temp file, one real ``ArenaMHAHostPool`` per rank on the SAME file, the real
``secure_rows_to_l3`` / ``arena_secure_to_disk``, ``_pdflip_direct_claim`` (``alloc_write``, ``_evict_for_claim``,
the W3 spill and the R12 worker request), ``form_a_host_shadow`` roles and the S85 census
``note_unbacked_at_reset``. The two ranks run the step in two threads; the group collective is an element-wise MAX
over both behind a barrier (what ``hicache_group_max`` posts)."""

import collections
import logging
import os
import shutil
import sys
import threading
import types
import unittest.mock as mock

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

import pytest  # noqa: E402
import torch  # noqa: E402

sys.path.insert(0, os.path.dirname(__file__))
import test_w3_arena_spill_0929 as W  # noqa: E402
import test_w3_arena_spill_forma_dual15_0930 as D2  # noqa: E402

from flliper.srt.managers import pdflip_flush_nonblock as F  # noqa: E402
from flliper.srt.mem_cache import form_a_host_shadow as R  # noqa: E402
from flliper.srt.mem_cache.unified_radix_cache import UnifiedTreeNode  # noqa: E402

pytestmark = pytest.mark.skipif(shutil.which("gcc") is None, reason="needs gcc")

FULL = W.FULL
SLOTS = 8

try:  # absent on the S85 base: the sleep leg then has no release step (the red baseline)
    from flliper.srt.pdflip import sleep_l3_release as S86  # noqa: E402
except ImportError:  # pragma: no cover - the base tree
    S86 = None

_ROLE = threading.local()


def _role():
    return getattr(_ROLE, "r", None)


@pytest.fixture(autouse=True)
def _thread_roles():
    with mock.patch.object(R, "role", _role):
        yield


class _Group:
    """``hicache_group_max`` over ``n`` ranks in threads: element-wise MAX behind a barrier; records each rank's
    collective labels (the count and order must be the same on every rank)."""

    def __init__(self, n):
        self.n = n
        self.bar = threading.Barrier(n, timeout=20)
        self.vals = {}
        self.calls = collections.defaultdict(list)

    def member(self, rank):
        def gm(values, label=""):
            k = len(self.calls[rank])
            self.calls[rank].append(label)
            self.vals[(k, rank)] = list(values)
            self.bar.wait()
            vs = [self.vals[(k, r)] for r in range(self.n)]
            assert len({len(v) for v in vs}) == 1, "ranks posted vectors of different length"
            out = [max(col) for col in zip(*vs)]
            self.bar.wait()
            return out
        return gm


def _on(rank_role, fn, *a, **kw):
    _ROLE.r = rank_role
    try:
        return fn(*a, **kw)
    finally:
        _ROLE.r = None


def _parallel(jobs):
    """jobs = [(role, fn, args)]; runs them in threads, re-raises the first error, returns the results."""
    res, err = [None] * len(jobs), []

    def _run(i, role, fn, args):
        try:
            res[i] = _on(role, fn, *args)
        except BaseException as e:  # noqa: BLE001 - the peer then breaks at the barrier's timeout
            err.append(e)

    ths = [threading.Thread(target=_run, args=(i, r, f, a)) for i, (r, f, a) in enumerate(jobs)]
    for t in ths:
        t.start()
    for t in ths:
        t.join(30)
    if err:
        raise err[0]
    return res


def _chain(t, parent, names, rid):
    """The park-end chain of a parked request: device KV, no host copy (refused), the last node the park end."""
    nodes = []
    for nm in names:
        n = UnifiedTreeNode((FULL,))
        n.parent = parent
        parent.children[(nm,)] = n
        n.hash_value = [nm]
        n.key = [nm]
        n.component_data[FULL].value = torch.arange(1, dtype=torch.int64)
        nodes.append(n)
        parent = n
    end = nodes[-1]
    end.pdflip_park_end = True
    end.pdflip_park_rid = rid
    t._pdflip_park_end_nodes = [end]
    return nodes


@pytest.fixture
def rc19(tmp_path):
    """TP0 and a worker on ONE arena: every slot COMPLETE, held by BOTH trees, every page already in L3 (the store
    writer kept L2 and L3 in step: rc19 dropped only ``on_disk`` slots); under the last tree node the park-end chain
    of pdflip-112-2708, two nodes with no copy anywhere."""
    R._reset_state_for_tests()
    g = _group(tmp_path)
    for n in g.n0:                       # L3 copies of every pinned page (on disk, as on the metal)
        sec = g.p0.secure_rows_to_l3(n.component_data[FULL].host_value)
        assert sec["lost"] == 0 and sec["pages"] == 1
    g.c0 = _chain(g.t0, g.n0[-1], ["c0", "c1"], "pdflip-112-2708")
    g.c1 = _chain(g.t1, g.n1[-1], ["c0", "c1"], "pdflip-112-2708")
    return g


def _group(tmp_path):
    store = tmp_path / "store"
    store.mkdir()
    ap = str(tmp_path / "kv.bin")
    p0, a0 = D2._rank_pool(tmp_path, ap, str(store))
    p1, a1 = D2._rank_pool(tmp_path, ap, str(store))
    hashes = [[f"a{i}"] for i in range(SLOTS)]
    pay = torch.zeros((W.PAGE,), dtype=torch.uint8)
    for i, hs in enumerate(hashes):
        pay.fill_(0x40 + i)
        st = p0._stems(hs)
        assert a0.write(st, [W.PAGE], [((0, W.PAGE),)], [pay.data_ptr()]) == [1]
    t0, t1 = D2._tree(p0), D2._tree(p1)
    n0 = D2._mirror(t0, p0, a0, hashes)
    n1 = D2._mirror(t1, p1, a1, hashes)
    return types.SimpleNamespace(p0=p0, p1=p1, a0=a0, a1=a1, t0=t0, t1=t1, n0=n0, n1=n1, store=store)


def _refs(g):
    return [int(r) for r in g.a0.slot_refs(list(range(SLOTS)))]


def _sleep_release(g, group=None):
    """The sleep leg's S86 step on both ranks at once (absent on the base: nothing happens)."""
    if S86 is None:
        return None
    grp = group or _Group(2)
    return _parallel([("host", S86.run, (g.t0, grp.member(0), g.p0)),
                      ("worker", S86.run, (g.t1, grp.member(1), g.p1))]), grp


def _flush_round(g):
    """The #1470 rounds' claims of the chain, head first, on every rank (the worker first, as on the metal: TP1
    was refused first); the write_backup's host rows on success."""
    out = []
    for i in range(len(g.c0)):
        for role, t, chain in (("worker", g.t1, g.c1), ("host", g.t0, g.c0)):
            pre = _on(role, t._pdflip_direct_claim, chain[i])
            if pre is not False and pre is not None:
                chain[i].component_data[FULL].host_value = pre
            out.append(pre)
    return out


def _lost(t):
    return sum(int(e["lost_tokens"]) for e in F.note_unbacked_at_reset(t, site="flush").get("park_end") or [])


# ---- (1) the metal, composed ------------------------------------------------------------------------------------

def test_rc19_b_tree_pinned_arena_the_sleep_release_lets_the_park_end_chain_in(rc19, caplog):
    """RED on the S85 base (no release step): both claims refused, the reset loses the chain's tokens
    (S85 PARK-END-UNBACKED lost_tokens=2 here, 13632 tokens / 3 chains on the metal). GREEN: lost_tokens 0."""
    g = rc19
    caplog.set_level(logging.INFO)
    assert _refs(g) == [2] * SLOTS                                        # complete=pinned=all, both trees
    assert g.a0.evict_candidates(SLOTS) == []                            # ARENA-DROP freed=0
    _sleep_release(g)
    _flush_round(g)
    lost = _lost(g.t0)
    assert lost == 0, ("S85 PARK-END-UNBACKED lost_tokens=%d: the trees' references of pages L3 holds kept the "
                       "chain out of the arena" % lost)
    assert not any("S85 PARK-END-UNBACKED" in m for m in caplog.messages)


def test_rc19_b_without_the_step_the_chain_is_lost_by_the_reset(rc19):
    """The baseline, pinned: the same tree with the switch off loses the chain -- TP0's own W3 spill lets go of
    TP0's references, the worker's stay (its request waits for a broadcast the flush never sends)."""
    g = rc19
    with mock.patch.dict(os.environ, {"FLLIPER_PDFLIP_SLEEP_L3_RELEASE": "0"}):
        _sleep_release(g)
        _flush_round(g)
    assert _lost(g.t0) == 2
    assert all(n.component_data[FULL].host_value is not None for n in g.n1), "a worker let go without the step"


def test_the_release_frees_every_l3_page_on_both_ranks_and_keeps_the_nodes_on_the_device(rc19):
    g = rc19
    (r0, r1), grp = _sleep_release(g)
    assert r0["agreed"] and r1["agreed"] and r0["pages"] == r1["pages"] == SLOTS
    assert r0["group_released"] == SLOTS
    assert _refs(g) == [0] * SLOTS                                        # every rank let go: the clock may take them
    for a, b in zip(g.n0, g.n1):
        for n in (a, b):
            assert n.component_data[FULL].host_value is None and n.l3_present
            assert n.component_data[FULL].value is not None              # still on the device, still in the tree
    for n in g.n0:                                                       # the L3 copy is there (never released without)
        assert (g.store / (n.hash_value[0] + "_sfx.bin")).exists()
    assert grp.calls[0] == grp.calls[1] and len(grp.calls[0]) == 3


# ---- (2) never a page without its L3 copy -----------------------------------------------------------------------

def test_a_page_without_an_l3_copy_is_never_released(rc19):
    """Mutant guard (an un-secured slot released = data loss): a page whose L3 write the store refuses keeps its
    node's host copy on EVERY rank -- the slot stays referenced, nothing names it on disk."""
    g = rc19
    victim = g.n0[3]
    st = victim.hash_value[0] + "_sfx"
    os.remove(g.store / (st + ".bin"))
    g.p0._backend._evictor.refuse.add(st)
    g.p1._backend._evictor.refuse.add(st)
    _sleep_release(g)
    assert g.n0[3].component_data[FULL].host_value is not None
    assert g.n1[3].component_data[FULL].host_value is not None
    assert _refs(g)[3] == 2
    assert not (g.store / (st + ".bin")).exists()
    assert [r for i, r in enumerate(_refs(g)) if i != 3] == [0] * (SLOTS - 1)


def test_an_incomplete_page_is_never_released(rc19):
    """A node one of whose pages no owner finished (not COMPLETE in the shared header) keeps its host copy."""
    g = rc19
    g.n0[5].hash_value = ["a5", "zz"]
    g.n1[5].hash_value = ["a5", "zz"]
    _sleep_release(g)
    assert g.n0[5].component_data[FULL].host_value is not None
    assert g.n1[5].component_data[FULL].host_value is not None
    assert _refs(g)[5] == 2


# ---- (3) rank unity ---------------------------------------------------------------------------------------------

def test_a_slot_one_rank_must_keep_is_kept_by_every_rank(rc19):
    """Mutant guard (rank disagreement): the worker's node 2 is host-locked (a load reads it); TP0 has no reason
    of its own to keep it -- the group veto keeps it on TP0 too, the trees stay alike, the digest agrees."""
    g = rc19
    g.n1[2].component_data[FULL].host_lock_ref = 1
    (r0, r1), _grp = _sleep_release(g)
    assert g.n0[2].component_data[FULL].host_value is not None, "TP0 let go of a slot the worker keeps"
    for a, b in zip(g.n0, g.n1):
        assert (a.component_data[FULL].host_value is None) == (b.component_data[FULL].host_value is None)
    assert r0["agreed"] and r1["agreed"] and r0["pages"] == r1["pages"] == SLOTS - 1
    assert _refs(g)[2] == 2


def test_every_rank_posts_the_same_collectives_whatever_it_holds(rc19):
    """Three collectives on every rank, same labels, same order -- also when one rank's veto raises (it then keeps
    everything) and when one rank has nothing to release."""
    g = rc19
    grp = _Group(2)
    for n in g.n1:
        n.component_data[FULL].host_lock_ref = 1                        # the worker keeps everything
    (r0, r1), grp = _sleep_release(g, grp)
    assert grp.calls[0] == grp.calls[1] == [
        "s86_sleep_l3_release/slots", "s86_sleep_l3_release/veto", "s86_sleep_l3_release/digest"]
    assert r0["pages"] == r1["pages"] == 0 and r0["agreed"]
    assert _refs(g) == [2] * SLOTS


def test_a_raising_veto_keeps_everything_and_still_posts_every_collective(rc19):
    g = rc19
    grp = _Group(2)
    real = S86._reason

    def boom(tree, node, pool):
        if tree is g.t1:
            raise RuntimeError("veto read failed")
        return real(tree, node, pool)

    with mock.patch.object(S86, "_reason", boom):
        (r0, r1), grp = _sleep_release(g, grp)
    assert grp.calls[0] == grp.calls[1] and len(grp.calls[0]) == 3
    assert r0["pages"] == r1["pages"] == 0
    assert all(n.component_data[FULL].host_value is not None for n in g.n0 + g.n1)


def test_switch_off_posts_no_collective(rc19):
    g = rc19
    grp = _Group(2)
    with mock.patch.dict(os.environ, {"FLLIPER_PDFLIP_SLEEP_L3_RELEASE": "0"}):
        _sleep_release(g, grp)
    assert grp.calls[0] == grp.calls[1] == []


# ---- (4) the wiring ---------------------------------------------------------------------------------------------

def test_the_sleep_flush_runs_the_release_on_group_d_before_the_flush():
    import pathlib

    here = pathlib.Path(__file__).resolve()
    src = next(p for p in here.parents if (p / "python" / "flliper").is_dir())
    text = (src / "python/flliper/srt/managers/scheduler_components/weight_updater.py").read_text()
    i = text.index("    def _pdflip_sleep_flush(self) -> int:")
    body = text[i:text.index("\n    def ", i + 10)]
    assert "self._pdflip_sleep_l3_release(sch, tc)" in body
    assert body.index("self._pdflip_sleep_l3_release(sch, tc)") < body.index("return sleep_flush_until_reset(")
    gate = body[:body.index("self._pdflip_sleep_l3_release(sch, tc)")]
    assert 'self._pdflip_group_name() == "D"' in gate


def test_a_group_release_voids_the_flush_memo():
    """The quiesce's no-progress memo (arena full, issued 0) must not skip the sleep flush's #1470 round after the
    group released: the slots it found pinned are free to take now."""
    from flliper.srt.managers.scheduler_components.weight_updater import SchedulerWeightUpdaterManager as WU

    sch = types.SimpleNamespace()
    F.sweep_memo_set(sch, types.SimpleNamespace())
    wu = object.__new__(WU)
    with mock.patch.object(S86, "run", return_value={"group_released": 3}), \
            mock.patch.object(F, "sweep_memo_clear", wraps=F.sweep_memo_clear) as clr:
        wu._pdflip_sleep_l3_release(sch, object())
    clr.assert_called_once_with(sch)
    with mock.patch.object(S86, "run", return_value={"group_released": 0}), \
            mock.patch.object(F, "sweep_memo_clear") as clr2:
        wu._pdflip_sleep_l3_release(sch, object())
    clr2.assert_not_called()
