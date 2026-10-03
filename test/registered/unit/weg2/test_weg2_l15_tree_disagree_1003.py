# SPDX-License-Identifier: Apache-2.0
"""L15-TREE-DISAGREE (desk item 210): can the D ranks disagree on the tree tips?

The qwen review of the tree candidates (040/160) named two cases. Both are
traced here as walk (rank-local decision) -> collective -> bind.

(a) ASYMMETRIC EVICTION. One rank loses a tip's device rows between the walk
    and the bind.
      walk -> agree gather: the rank is blocked in the collective (same
        scheduler thread), nothing evicts there.
      eviction BEFORE the probe: the probe gather (``agree_holdable``) takes
        that rank's "no_device_span" vote and every rank drops the tip
        (green on base: test ``eviction_before_the_probe_is_dropped_everywhere``).
      eviction AFTER the probe gather, before the bind: the only seam left, and
        the bind SKIPPED the tip rank-locally ("unholdable at=bind") while its
        peers bound it -> the ranks armed DIFFERENT hold sets. RED on base;
        the fix: an agreed tip failing at the bind raises ``L15TreeDisagree``
        (this rank does not arm, the POST vote refuses the hold on all ranks).
(b) CAP TRUNCATION. ``local_candidates(limit = N + n_live)`` cut every rank's
    list by its OWN recency clock BEFORE the gather. The agreed list is still
    one answer on every rank (a pure function of the votes), but two windows
    cut from different orders (or from the capped rank's list vs the cap-0
    rank's L2-filtered list) can be disjoint: agreed=0 although every tip is
    common. RED on base; the fix gathers every tip's digest and cuts AFTER
    the agreement.
"""

from __future__ import annotations

import inspect
import pathlib
import sys
import threading
from types import SimpleNamespace

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

import torch  # noqa: E402

from sglang.srt.mem_cache.unified_cache_components.tree_component import (  # noqa: E402
    ComponentType,
)
from sglang.srt.weg2 import l15_bind, l15_sleep_agree, l15_tree_cand  # noqa: E402
from test_weg2_l15_bind_1001 import _load_retain_test_module, _req_to_token  # noqa: E402

FULL, MAMBA = int(ComponentType.FULL), int(ComponentType.MAMBA)


# ------------------------------------------------------------------ fixtures


class _Node:
    def __init__(self, parent, toks, clock, anchor=None, host=False):
        self.parent = parent
        self.children = {}
        self.key = SimpleNamespace(token_ids=list(toks), extra_key=None)
        self.last_access_time = clock
        cds = [SimpleNamespace(value=None, host_value=None) for _ in range(3)]
        if toks:
            cds[FULL].value = torch.arange(len(toks), dtype=torch.int64) + 1
            if host:
                cds[FULL].host_value = torch.arange(len(toks), dtype=torch.int64) + 100
        if anchor is not None:
            cds[MAMBA].value = torch.tensor([anchor], dtype=torch.int64)
            if host:
                cds[MAMBA].host_value = torch.tensor([7], dtype=torch.int64)
        self.component_data = cds
        if parent is not None:
            parent.children[tuple(toks[:1])] = self


class RankTree:
    """root -> one tip per entry of ``tips`` = {i: clock}; tip i = tokens
    [i, i, i] with anchor i + 1. ``match_prefix`` serves them like the real
    tree (device_indices + last_device_node); ``evict`` drops a tip's rows."""

    def __init__(self, tips, host=()):
        self.root_node = _Node(None, [], 0)
        self.by_tokens = {}
        for i, clock in tips.items():
            n = _Node(self.root_node, [i, i, i], clock, anchor=i + 1,
                      host=i in host)
            self.by_tokens[(i, i, i)] = n

    def match_prefix(self, params):
        tok = tuple(params.key.token_ids)
        n = self.by_tokens.get(tok)
        if n is None or n.component_data[FULL].value is None:
            return SimpleNamespace(device_indices=torch.zeros(0, dtype=torch.int64),
                                   last_device_node=None)
        return SimpleNamespace(device_indices=n.component_data[FULL].value,
                               last_device_node=n)

    def evict(self, i):
        n = self.by_tokens[(i, i, i)]
        n.component_data[FULL].value = None
        n.component_data[MAMBA].value = None
        self.root_node.children.pop((i,), None)


class Group:
    """An all_gather_object over N threads (lockstep, like the D group)."""

    def __init__(self, n):
        self.n = n
        self.bar = threading.Barrier(n)
        self.slots = [None] * n
        self.calls = [0] * n

    def gather_for(self, rank, after=None):
        def gather(v):
            self.slots[rank] = v
            self.bar.wait()
            out = list(self.slots)
            self.bar.wait()
            k = self.calls[rank]
            self.calls[rank] += 1
            if after is not None:
                after(rank, k)
            return out
        return gather


def run_ranks(trees, env, n_live=0, require_l2=(), after=None):
    """build() on every rank in lockstep, with the real tree probe."""
    g = Group(len(trees))
    res = [None] * len(trees)
    errs = []

    def one(rk):
        try:
            res[rk] = l15_tree_cand.build(
                trees[rk], g.gather_for(rk, after), n_live, env, None,
                require_l2=rk in require_l2,
                probe=lambda r: l15_bind.tree_probe(r, trees[rk]))
        except BaseException as exc:  # noqa: BLE001
            errs.append(exc)

    ts = [threading.Thread(target=one, args=(i,)) for i in range(len(trees))]
    [t.start() for t in ts]
    [t.join(20) for t in ts]
    assert not errs, errs
    return res


def rids(reqs):
    return [r.rid for r in reqs]


# ----------------------------------------------------------- (b) truncation


def test_cap_truncation_does_not_leave_disjoint_windows_when_recency_orders_differ():
    # 10 tips common to the 3 ranks; each rank's recency clock orders them
    # differently (per-process clocks). N=2: the old per-rank cut kept
    # {0,1} / {9,8} / {4,5} -> nothing common -> agreed=0.
    ids = list(range(10))
    trees = [
        RankTree({i: 100 + i for i in ids}),
        RankTree({i: 100 + (9 - i) for i in ids}),
        RankTree({i: 100 + (i * 3) % 10 for i in ids}),
    ]
    out = run_ranks(trees, {"SGLANG_WEG2_L15_TREE_CAND_N": "2"})
    assert rids(out[0]) == rids(out[1]) == rids(out[2]), "one answer on every rank"
    assert len(out[0]) == 2, "every tip is common: the cap cuts AFTER the agreement"


def test_cap0_rank_l2_filter_does_not_empty_the_agreement():
    # the capped ranks' freshest tips (7,8,9) are not L2-backed yet on the
    # cap-0 rank, which offers only the older backed ones: with a per-rank cut
    # (N=2) the windows {9,8} and {5,4} never meet.
    ids = list(range(10))
    capped = {i: 100 + i for i in ids}
    trees = [RankTree(capped), RankTree(capped),
             RankTree(capped, host=set(range(0, 7)))]
    out = run_ranks(trees, {"SGLANG_WEG2_L15_TREE_CAND_N": "2"}, require_l2=(2,))
    assert rids(out[0]) == rids(out[1]) == rids(out[2])
    assert len(out[0]) == 2, "tips 0..6 are held by all three ranks"


def test_n_live_widens_the_cap_after_the_agreement_not_before():
    ids = list(range(10))
    trees = [RankTree({i: 100 + i for i in ids}),
             RankTree({i: 100 + (9 - i) for i in ids})]
    out = run_ranks(trees, {"SGLANG_WEG2_L15_TREE_CAND_N": "2"}, n_live=3)
    assert rids(out[0]) == rids(out[1])
    assert len(out[0]) == 5  # N + n_live, cut once, on the agreed list


def test_streaming_digest_is_byte_identical_to_the_chain_digest():
    plain = RankTree({i: 10 + i for i in range(4)})
    lazy = l15_tree_cand.local_candidates(plain, None, lazy_tokens=True)
    eager = l15_tree_cand.local_candidates(plain, 99)
    assert [(c.digest, c.n_tokens) for c in lazy] == [
        (c.digest, c.n_tokens) for c in eager]
    assert all(c.tokens == () for c in lazy)
    assert [l15_tree_cand.with_tokens(c, plain).tokens for c in lazy] == [
        c.tokens for c in eager]


def _bigram_tree(extra_key=None, bad_boundary=False):
    """root -> A(raw 1..4) -> B(raw 4..7) and A -> C(raw 4, 9, 9); bigram keys
    carry units + 1 raw tokens, a child's first raw token = its parent's last."""
    root = _Node(None, [], 0)

    def node(parent, raw, clock, anchor=None):
        n = _Node(parent, raw, clock, anchor=anchor)
        n.key = SimpleNamespace(token_ids=list(raw), is_bigram=True,
                                extra_key=extra_key)
        return n

    a = node(root, [1, 2, 3, 4], 1)
    b = node(a, [4 if not bad_boundary else 5, 5, 6, 7], 2, anchor=11)
    c = node(a, [4, 9, 9], 3, anchor=12)
    # bigram children are keyed by their first bigram (4,5) / (4,9)
    a.children = {(4, 5): b, (4, 9): c}
    return SimpleNamespace(root_node=root)


def test_streaming_digest_matches_for_bigram_keys_and_extra_key():
    for ek in (None, "lora-7"):
        t = _bigram_tree(extra_key=ek)
        lazy = l15_tree_cand.local_candidates(t, None, lazy_tokens=True)
        eager = l15_tree_cand.local_candidates(t, 99)
        assert len(lazy) == len(eager) == 2
        assert [(c.digest, c.n_tokens) for c in lazy] == [
            (c.digest, c.n_tokens) for c in eager]
        assert {c.n_tokens for c in lazy} == {7, 6}
    # a boundary token that does not compose: that tip is skipped in both
    t = _bigram_tree(bad_boundary=True)
    lazy = l15_tree_cand.local_candidates(t, None, lazy_tokens=True)
    eager = l15_tree_cand.local_candidates(t, 99)
    assert [c.digest for c in lazy] == [c.digest for c in eager]
    assert len(lazy) == 1


def test_a_duplicate_digest_on_one_rank_is_one_vote():
    c = l15_tree_cand.TreeCand("aa", 3, 1.0)
    d = l15_tree_cand.TreeCand("bb", 3, 1.0)
    # rank 0 reports "aa" twice, rank 1 never: not "present on every rank"
    votes = [[("aa", 3), ("aa", 3), ("bb", 3)], [("bb", 3)]]
    got = l15_tree_cand.agree([c, d], lambda _v: votes, 8)
    assert [x.digest for x in got] == ["bb"]


# --------------------------------------------------------- (a) eviction


def _holds_after_bind(trees, agreed, tmp_path):
    """Each rank binds its agreed tips as the hook does; the armed rid set per
    rank (None = this rank's round does not arm)."""
    rt = _load_retain_test_module()
    out = []
    sig = inspect.signature(l15_bind.build_retain_kwargs).parameters
    for rk, tree in enumerate(trees):
        sc = rt.make_scenario(tmp_path / ("r%d" % rk), [])
        kwargs = dict(sc["kwargs"])
        for key in ("candidates", "node_of", "slots_of", "anchor_slot_of",
                    "l2_of", "rewrite_tree"):
            kwargs.pop(key)
        kwargs["log"] = lambda _m: None
        kwargs["caps_rows_by_rank"] = (100, 100)
        if "tree_agreed" in sig:
            kwargs["tree_agreed"] = True  # what the hook passes
        try:
            bound = l15_bind.build_retain_kwargs(
                agreed[rk], _req_to_token(), tree_cache=tree, **kwargs)
            out.append(frozenset(c.rid for c in bound["candidates"]))
        except Exception:  # noqa: BLE001 -- the hook catches: this rank does not arm
            out.append(None)
    return out


def test_eviction_before_the_probe_is_dropped_everywhere():
    trees = [RankTree({i: 10 + i for i in range(4)}) for _ in range(3)]

    def after(rank, k):
        if rank == 1 and k == 0:  # between the agree gather and the probe
            trees[1].evict(2)

    out = run_ranks(trees, {}, after=after)
    assert rids(out[0]) == rids(out[1]) == rids(out[2])
    assert len(out[0]) == 3
    assert "tree:" + l15_tree_cand.digest_of([2, 2, 2]) not in rids(out[0])


def test_eviction_after_the_probe_never_arms_different_hold_sets(tmp_path):
    trees = [RankTree({i: 10 + i for i in range(4)}) for _ in range(3)]
    agreed = run_ranks(trees, {})
    assert rids(agreed[0]) == rids(agreed[1]) == rids(agreed[2])
    assert len(agreed[0]) == 4
    # rank 1 loses a tip's device rows after the probe gather, before the bind
    trees[1].evict(2)
    armed = _holds_after_bind(trees, agreed, tmp_path)
    sets = {s for s in armed if s is not None}
    assert len(sets) <= 1, "ranks armed different hold sets: %r" % (armed,)
    # a rank that does not arm vetoes the hold on EVERY rank (POST vote)
    for s in armed:
        if s is None:
            assert l15_sleep_agree.post_vote(None, 100) is not None


def test_bind_without_agreement_still_skips_quietly(tmp_path):
    # the opt-out (tree_agreed False, unit callers without a probe) keeps the
    # old per-rid skip: the candidate is dropped, the round goes on
    trees = [RankTree({i: 10 + i for i in range(2)})]
    agreed = run_ranks(trees, {})
    trees[0].evict(1)
    rt = _load_retain_test_module()
    sc = rt.make_scenario(tmp_path, [])
    kwargs = dict(sc["kwargs"])
    for key in ("candidates", "node_of", "slots_of", "anchor_slot_of",
                "l2_of", "rewrite_tree"):
        kwargs.pop(key)
    kwargs["log"] = lambda _m: None
    kwargs["caps_rows_by_rank"] = (100, 100)
    bound = l15_bind.build_retain_kwargs(agreed[0], _req_to_token(),
                                         tree_cache=trees[0], **kwargs)
    assert len(bound["candidates"]) == 1


def test_hook_tells_the_bind_that_the_tips_were_agreed():
    import ast

    src = (pathlib.Path(__file__).resolve().parents[4] / "python" / "sglang"
           / "srt" / "managers" / "scheduler.py").read_text()
    calls = [c for c in ast.walk(ast.parse(src)) if isinstance(c, ast.Call)
             and getattr(c.func, "attr", None) == "build_retain_kwargs"]
    kws = {k.arg: ast.unparse(k.value) for k in calls[0].keywords}
    assert kws.get("tree_agreed") == "True"
    # and the walk that feeds the collective is not cut by a rank-local cap
    tc = (pathlib.Path(l15_tree_cand.__file__)).read_text()
    assert "local_candidates(tree_cache, None" in tc
