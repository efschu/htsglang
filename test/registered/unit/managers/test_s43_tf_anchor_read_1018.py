"""S43 TF-ANCHOR-READ (NF rc12 1018, P.log 16:39:16, pdflip-3-186).

PP0: ``PF TOLD-FLOOR rid=pdflip-3-186 told=110336 -> 10688 acks={1: 100224, 2: 100224}``
after ``#TF TOLD-FIDELITY told=100224 pp0_admissible=96128``; both followers sent
``reads=[('pdflip-3-186', 100224, (83840, 67456, 51072, 34688, 18304, 10688))]``.
PP0's tree held the KV (host_hit 99648) but its Mamba arena had dropped those
anchors to L3 (13 mamba ARENA-DROPs l3=on_disk on PP0 vs 8 on PP1, 1 on PP2), so
none of the common depths was resumable on PP0 and 103977 tokens were
recomputed (front PDFLIP-USAGE-DETAILS p_computed=103977). The fix: PP0 reads
the anchor blob of a depth every follower proved (e1 at that depth) and asks
its tree again -- still PP0's ONE verdict for every rank (SOLL R11).
"""

import os
import types

import pytest

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

from sglang.srt.environ import envs  # noqa: E402
from sglang.srt.managers import weg2_told_fallback as fb  # noqa: E402
from sglang.srt.managers import weg2_told_fidelity as tf  # noqa: E402

RID = "pdflip-3-186"
TOLD = 110336
ACK = 100224
ANCHORS = (83840, 67456, 51072, 34688, 18304, 10688)
PP0_KV_END = 96128  # PP0's tree reaches the KV this deep (pp0_admissible at 100224)


@pytest.fixture(autouse=True)
def _env(monkeypatch):
    for k in (fb.ENV_ANCHOR_ACKS, fb.ENV_FLOOR, fb.ENV_MIN_PASSES,
              "SGLANG_WEG2_TF_ANCHOR_READ", "SGLANG_WEG2_TF_ANCHOR_READ_MAX"):
        monkeypatch.delenv(k, raising=False)


class _Tree:
    """PP0's tree in the 16:39:16 state: Mamba states only at 10688 (device)
    and 96128; the store holds the anchors of the follower grid."""

    def __init__(self, store=ANCHORS):
        self.states = {10688, PP0_KV_END}
        self.store = set(store)
        self.reads = []

    def weg2_anchor_only_read_at(self, req, depth, site="tf_floor"):
        self.reads.append(int(depth))
        if depth <= PP0_KV_END and depth in self.store:
            self.states.add(int(depth))
            return int(depth)
        return 0

    def admissible(self, depth):
        kv = min(int(depth), PP0_KV_END)
        ok = [s for s in self.states if s <= kv]
        return max(ok) if ok else 0


def _sched(tree):
    return types.SimpleNamespace(ps=types.SimpleNamespace(pp_size=3, pp_rank=0), tree_cache=tree)


def _open(s):
    fb.pp0_open(s, RID, TOLD, published_at=0.0, own_read_s=0.0, window_s=1.0, published_pass=0)
    o = fb._pp0_open_map(s)[RID]
    o.acks.update({1: ACK, 2: ACK})
    o.anchors.update({1: ANCHORS, 2: ANCHORS})
    return o


@pytest.fixture
def tree(monkeypatch):
    t = _Tree()
    monkeypatch.setattr(tf, "pp0_admissible", lambda sched, req, depth: sched.tree_cache.admissible(depth))
    monkeypatch.setattr(tf, "absolute_depth", lambda req, d, absolute: int(d))
    return t


def test_red_without_the_read_the_floor_falls_to_10688(monkeypatch, tree):
    """The rc12 log, reproduced: with the read off PP0 can only take 10688."""
    monkeypatch.setenv("SGLANG_WEG2_TF_ANCHOR_READ", "0")
    s = _sched(tree)
    _open(s)
    assert fb.pp0_decide(s, RID, now=100.0, pass_n=10) == (ACK, fb.REASON_FLOOR)
    assert fb.pp0_floor_below(s, RID, object(), ACK, True) == 10688
    assert tree.reads == []


def test_green_pp0_reads_the_anchor_and_takes_83840(tree):
    s = _sched(tree)
    _open(s)
    assert fb.pp0_floor_below(s, RID, object(), ACK, True) == 83840
    assert tree.reads == [83840]  # one blob, the deepest common depth below the refused floor


def test_a_store_miss_tries_the_next_common_depth_within_the_budget(tree):
    tree.store = {51072}
    s = _sched(tree)
    _open(s)
    assert fb.pp0_floor_below(s, RID, object(), ACK, True) == 51072
    assert tree.reads == [83840, 67456, 51072]


def test_the_read_budget_bounds_the_search(monkeypatch, tree):
    monkeypatch.setenv("SGLANG_WEG2_TF_ANCHOR_READ_MAX", "2")
    tree.store = {51072}
    s = _sched(tree)
    _open(s)
    assert fb.pp0_floor_below(s, RID, object(), ACK, True) == 10688  # 2 misses, then PP0's own state
    assert tree.reads == [83840, 67456]


def test_no_read_where_pp0_already_resumes():
    t = _Tree()
    t.states |= set(ANCHORS)
    s = _sched(t)
    _open(s)
    import pytest as _p
    mp = _p.MonkeyPatch()
    try:
        mp.setattr(tf, "pp0_admissible", lambda sched, req, depth: sched.tree_cache.admissible(depth))
        mp.setattr(tf, "absolute_depth", lambda req, d, absolute: int(d))
        assert fb.pp0_floor_below(s, RID, object(), ACK, True) == 83840
    finally:
        mp.undo()
    assert t.reads == []


def test_a_tree_without_the_reader_keeps_the_old_answer(tree):
    s = types.SimpleNamespace(ps=types.SimpleNamespace(pp_size=3, pp_rank=0),
                              tree_cache=types.SimpleNamespace(admissible=tree.admissible))
    _open(s)
    assert fb.pp0_floor_below(s, RID, object(), ACK, True) == 10688


def test_a_raising_reader_never_breaks_the_floor(tree):
    def boom(*a, **k):
        raise RuntimeError("store down")

    tree.weg2_anchor_only_read_at = boom
    s = _sched(tree)
    _open(s)
    assert fb.pp0_floor_below(s, RID, object(), ACK, True) == 10688


def test_switch_default_on():
    assert envs.SGLANG_WEG2_TF_ANCHOR_READ.get() is True
    assert envs.SGLANG_WEG2_TF_ANCHOR_READ_MAX.get() == 3


def test_an_attach_pp0s_tree_still_refuses_is_not_admitted(tree):
    """Danger direction (mutant: admit on the READ's word, no re-probe): the
    read reports an attach at 83840 but PP0's tree still cannot resume there
    (e.g. the re-match rejects the node) -- PP0's tree decides, the search
    goes on to the next depth it can really resume from."""
    real = tree.weg2_anchor_only_read_at

    def liar(req, depth, site="tf_floor"):
        if depth == 83840:
            tree.reads.append(depth)
            return depth  # claims an attach, adds no state
        return real(req, depth, site)

    tree.weg2_anchor_only_read_at = liar
    s = _sched(tree)
    _open(s)
    got = fb.pp0_floor_below(s, RID, object(), ACK, True)
    assert got == 67456
    assert tree.admissible(got) >= got
