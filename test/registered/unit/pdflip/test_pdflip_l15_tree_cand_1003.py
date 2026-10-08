# SPDX-License-Identifier: Apache-2.0
"""L15-TREE-CAND: finished requests stay L1.5 hold candidates.

27B boot ..._1002_195124: the 15-s DECODE-COLLECT window lets every decode
finish before the sleep (FLIP begin outstanding=0), so running_batch.reqs and
pdflip_d_parked are both empty: held=0, SHADOW n=0 at every sleep. The finished
requests' spans are in the radix tree with their END anchor; they are the hold
candidates. The choice is ONE collective (never a per-rank term before it).
"""

from __future__ import annotations

import pathlib
import sys
from types import SimpleNamespace

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

import pytest  # noqa: E402
import torch  # noqa: E402

from flliper.srt.mem_cache.unified_cache_components.tree_component import (  # noqa: E402
    ComponentType,
)
from flliper.srt.pdflip import l15_bind, l15_retain, l15_tree_cand  # noqa: E402
from test_pdflip_l15_bind_1001 import (  # noqa: E402
    _load_retain_test_module,
    _req,
    _req_to_token,
    _rewrite_recorder,
)

FULL, MAMBA = int(ComponentType.FULL), int(ComponentType.MAMBA)


class N:
    """A radix-tree node: parent/children/key/component_data/clock."""

    def __init__(self, parent, toks, clock, *, anchor=None, kv=True,
                 host_kv=False, host_mamba=False, extra_key=None):
        self.parent = parent
        self.children = {}
        self.key = SimpleNamespace(token_ids=list(toks), extra_key=extra_key)
        self.last_access_time = clock
        cds = [SimpleNamespace(value=None, host_value=None) for _ in range(3)]
        if kv:
            cds[FULL].value = torch.arange(len(toks), dtype=torch.int64) + 1
            if host_kv:
                cds[FULL].host_value = torch.arange(len(toks), dtype=torch.int64) + 100
        if anchor is not None:
            cds[MAMBA].value = torch.tensor([anchor], dtype=torch.int64)
            if host_mamba:
                cds[MAMBA].host_value = torch.tensor([7], dtype=torch.int64)
        self.component_data = cds
        if parent is not None:
            parent.children[tuple(toks[:1])] = self


def _tree(clocks=(10, 20, 30), **kw):
    """root -> A(0,1) -> {B(2,3)[anchor], C(4,5)[anchor]} and root -> D(9,9)[anchor].
    B, C, D are the three finished requests (tips); A is shared prefix."""
    root = N(None, [], 0, kv=False)
    a = N(root, [0, 1], 1, anchor=None, **kw)
    b = N(a, [2, 3], clocks[0], anchor=11, **kw)
    c = N(a, [4, 5], clocks[1], anchor=12, **kw)
    d = N(root, [9, 9], clocks[2], anchor=13, **kw)
    return SimpleNamespace(root_node=root), (a, b, c, d)


# ---------------------------------------------------------------- selection


def test_tips_are_device_nodes_with_an_end_anchor_and_none_below():
    tree, (a, b, c, d) = _tree()
    # an intermediate checkpoint above a deeper tip is not a tip
    a.component_data[MAMBA].value = torch.tensor([5], dtype=torch.int64)
    tips = l15_tree_cand.tips_of(tree)
    assert {id(t) for t in tips} == {id(b), id(c), id(d)}
    # an evicted tip (no device KV) is no candidate
    d.component_data[FULL].value = None
    assert {id(t) for t in l15_tree_cand.tips_of(tree)} == {id(b), id(c)}


def test_local_candidates_most_recently_used_first_with_full_chain():
    tree, _ = _tree(clocks=(10, 20, 30))
    got = l15_tree_cand.local_candidates(tree, 8)
    assert [c.tokens for c in got] == [(9, 9), (0, 1, 4, 5), (0, 1, 2, 3)]
    assert [c.n_tokens for c in got] == [2, 4, 4]
    assert len({c.digest for c in got}) == 3
    assert [c.tokens for c in l15_tree_cand.local_candidates(tree, 2)] == [
        (9, 9), (0, 1, 4, 5)]


def test_digest_is_token_identity_not_a_per_process_clock():
    t1, _ = _tree(clocks=(10, 20, 30))
    t2, _ = _tree(clocks=(7000, 9, 123))  # another rank's counter values
    d1 = {c.tokens: c.digest for c in l15_tree_cand.local_candidates(t1, 8)}
    d2 = {c.tokens: c.digest for c in l15_tree_cand.local_candidates(t2, 8)}
    assert d1 == d2


def test_cap0_rank_offers_only_l2_backed_tips():
    tree, (a, b, c, d) = _tree(host_kv=True, host_mamba=True)
    assert len(l15_tree_cand.local_candidates(tree, 8, require_l2=True)) == 3
    c.component_data[MAMBA].host_value = None          # anchor without L2 identity
    d.component_data[FULL].host_value = None           # chain hole
    got = l15_tree_cand.local_candidates(tree, 8, require_l2=True)
    assert [x.tokens for x in got] == [(0, 1, 2, 3)]
    # a capped rank (no L2 need) still offers all three
    assert len(l15_tree_cand.local_candidates(tree, 8, require_l2=False)) == 3


# ------------------------------------------------------------ rank agreement


def _gather_of(per_rank_locals):
    votes = [[(c.digest, c.n_tokens) for c in loc] for loc in per_rank_locals]
    return lambda _mine: votes


def test_agreement_is_one_answer_on_every_rank_despite_rank_local_clocks():
    # three ranks, different per-process clocks -> different recency ORDER
    trees = [_tree(clocks=(10, 20, 30))[0], _tree(clocks=(30, 20, 10))[0],
             _tree(clocks=(20, 10, 30))[0]]
    locs = [l15_tree_cand.local_candidates(t, 8) for t in trees]
    gather = _gather_of(locs)
    answers = [[c.digest for c in l15_tree_cand.agree(loc, gather, 8)] for loc in locs]
    assert answers[0] == answers[1] == answers[2]
    assert len(answers[0]) == 3


def test_agreement_drops_a_tip_some_rank_does_not_have():
    trees = [_tree()[0], _tree()[0]]
    # rank 1 lost tip D (evicted there)
    _, (_, _, _, d1) = _tree()
    trees[1].root_node.children.pop((9,))
    locs = [l15_tree_cand.local_candidates(t, 8) for t in trees]
    gather = _gather_of(locs)
    a0 = l15_tree_cand.agree(locs[0], gather, 8)
    a1 = l15_tree_cand.agree(locs[1], gather, 8)
    assert [c.digest for c in a0] == [c.digest for c in a1]
    assert (9, 9) not in {c.tokens for c in a0}
    assert len(a0) == 2


def test_agreement_respects_the_limit_and_an_empty_vote_stops_everyone():
    tree, _ = _tree()
    loc = l15_tree_cand.local_candidates(tree, 8)
    assert len(l15_tree_cand.agree(loc, _gather_of([loc, loc]), 2)) == 2
    # a rank that votes the empty list (walk failed / filtered) -> no candidate anywhere
    assert l15_tree_cand.agree(loc, _gather_of([loc, []]), 8) == []


def test_build_enters_the_collective_even_when_the_local_walk_fails():
    calls = []

    def gather(v):
        calls.append(v)
        return [v]

    class Boom:
        @property
        def root_node(self):
            raise RuntimeError("tree walk blew up")

    logged = []
    assert l15_tree_cand.build(Boom(), gather, 0, {}, logged.append) == []
    assert calls == [[]], "a failed walk must still vote (empty) in the gather"
    assert any("walk_failed=RuntimeError" in m for m in logged)


def test_build_returns_pseudo_reqs_marked_served_with_rank_uniform_recency():
    tree, _ = _tree()
    out = l15_tree_cand.build(tree, lambda v: [v], 0, {}, None)
    assert len(out) == 3
    assert all(r.rid.startswith("tree:") and r.l15_kind == "served" for r in out)
    assert all(r.req_pool_idx is None for r in out)
    assert [r.l15_last_active for r in out] == [3.0, 2.0, 1.0]


def test_switch_defaults_on_and_zero_turns_it_off():
    assert l15_tree_cand.env_on({}) is True
    assert l15_tree_cand.env_on({"FLLIPER_PDFLIP_L15_TREE_CAND": "1"}) is True
    assert l15_tree_cand.env_on({"FLLIPER_PDFLIP_L15_TREE_CAND": "0"}) is False
    assert l15_tree_cand.max_n({}) == l15_tree_cand.DEFAULT_MAX_N
    assert l15_tree_cand.max_n({"FLLIPER_PDFLIP_L15_TREE_CAND_N": "2"}) == 2


# ----------------------------------------------------------- bind + retain


class _MatchTree:
    """match_prefix by exact token chain -> (device slots, node with anchor)."""

    def __init__(self, by_tokens):
        self.by_tokens = by_tokens
        self.calls = []

    def match_prefix(self, params):
        from array import array

        tid = params.key.token_ids
        assert isinstance(tid, array) and tid.typecode == "q"
        self.calls.append(tuple(tid))
        slots, node = self.by_tokens[tuple(tid)]
        return SimpleNamespace(
            device_indices=torch.tensor(slots, dtype=torch.int64),
            last_device_node=node)


def _anchor_node(rt, anchor):
    node = rt.FakeNode([])
    cds = [SimpleNamespace(value=None, host_value=None) for _ in range(3)]
    cds[MAMBA] = SimpleNamespace(
        value=torch.tensor([anchor], dtype=torch.int64), host_value=None)
    node.component_data = cds
    return node


def _kwargs(rt, tmp_path, logged):
    sc = rt.make_scenario(tmp_path, [])
    kwargs = dict(sc["kwargs"])
    for key in ("candidates", "node_of", "slots_of", "anchor_slot_of",
                "l2_of", "rewrite_tree"):
        kwargs.pop(key)
    kwargs["log"] = logged.append
    kwargs["caps_rows_by_rank"] = (10, 10)
    return kwargs


def _pseudo(tokens, active=1.0):
    c = l15_tree_cand.TreeCand(
        digest=l15_tree_cand.digest_of(tokens), n_tokens=len(tokens),
        last_access=1.0, tokens=tuple(tokens))
    return l15_tree_cand.pseudo_req(c, 0, 1)


def test_nothing_running_nothing_parked_the_finished_tips_are_held(tmp_path):
    """The 195124 situation: outstanding=0 -> live list empty. Before the fix
    the candidate list stays empty (held=0); with the tree tips it is not."""
    rt = _load_retain_test_module()
    logged = []
    kwargs = _kwargs(rt, tmp_path, logged)
    tree = _MatchTree({
        (0, 1, 2): ([5, 7, 8], _anchor_node(rt, 3)),
        (4, 4, 4): ([9, 10, 11], _anchor_node(rt, 4)),
    })
    # base behaviour: no live reqs -> nothing to hold
    empty = l15_bind.build_retain_kwargs([], _req_to_token(), tree_cache=tree, **kwargs)
    assert empty["candidates"] == []
    reqs = [_pseudo([0, 1, 2]), _pseudo([4, 4, 4])]
    bound = l15_bind.build_retain_kwargs(reqs, _req_to_token(), tree_cache=tree,
                                         **kwargs)
    rids = {c.rid for c in bound["candidates"]}
    assert rids == {r.rid for r in reqs}, (rids, logged)
    # the WHOLE chain is matched (a tip has KV for every token), not len - 1
    assert (0, 1, 2) in tree.calls and (4, 4, 4) in tree.calls
    assert bound["slots_of"](reqs[0].rid) == (5, 7, 8)
    assert bound["anchor_slot_of"](reqs[0].rid) == 3
    assert any("L15-TREE-CAND bind used=2 covered=0 unholdable=0" in m for m in logged)
    res = l15_retain.retain_at_sleep(rewrite_tree=_rewrite_recorder, **bound)
    assert isinstance(res, l15_retain.RetainResult)
    assert len(res.hold.rids) >= 1, "held must be > 0 for finished requests"
    assert all(r.startswith("tree:") for r in res.hold.rids)


def test_a_tip_already_covered_by_a_live_req_is_not_held_twice(tmp_path):
    rt = _load_retain_test_module()
    logged = []
    kwargs = _kwargs(rt, tmp_path, logged)
    node_p = _anchor_node(rt, 3)
    other = _anchor_node(rt, 4)
    # the parked req (tokens 0,1,2,0,1 -> span 4) resolves to the SAME node
    # the tip (0,1,2) is: the live req already covers it
    tree = _MatchTree({
        (0, 1, 2, 0): ([5, 7, 8], node_p),
        (0, 1, 2): ([5, 7, 8], node_p),
        (4, 4, 4): ([9, 10, 11], other),
    })
    parked = _req("r_park", None, 3, 2, None, None, "parked", 3.0)
    reqs = [parked, _pseudo([0, 1, 2]), _pseudo([4, 4, 4])]
    bound = l15_bind.build_retain_kwargs(reqs, _req_to_token(), tree_cache=tree,
                                         **kwargs)
    rids = {c.rid for c in bound["candidates"]}
    assert "r_park" in rids
    assert l15_tree_cand.RID_PREFIX + l15_tree_cand.digest_of([0, 1, 2]) not in rids
    assert l15_tree_cand.RID_PREFIX + l15_tree_cand.digest_of([4, 4, 4]) in rids
    assert any("covered=1" in m for m in logged)


def test_tree_cand_max_caps_the_number_of_tips(tmp_path):
    rt = _load_retain_test_module()
    logged = []
    kwargs = _kwargs(rt, tmp_path, logged)
    tree = _MatchTree({
        (0, 1, 2): ([5, 7, 8], _anchor_node(rt, 3)),
        (4, 4, 4): ([9, 10, 11], _anchor_node(rt, 4)),
    })
    reqs = [_pseudo([0, 1, 2]), _pseudo([4, 4, 4])]
    bound = l15_bind.build_retain_kwargs(reqs, _req_to_token(), tree_cache=tree,
                                         tree_cand_max=1, **kwargs)
    assert len(bound["candidates"]) == 1


def test_an_unmatchable_tip_is_skipped_quietly(tmp_path):
    rt = _load_retain_test_module()
    logged = []
    kwargs = _kwargs(rt, tmp_path, logged)
    no_anchor = rt.FakeNode([])
    no_anchor.component_data = [SimpleNamespace(value=None, host_value=None)] * 3
    tree = _MatchTree({(0, 1, 2): ([5, 7, 8], no_anchor)})
    bound = l15_bind.build_retain_kwargs([_pseudo([0, 1, 2])], _req_to_token(),
                                         tree_cache=tree, **kwargs)
    assert bound["candidates"] == []
    assert any("unholdable=1" in m for m in logged)
    assert not any("L15-RETAIN skipped" in m for m in logged)


# ------------------------------------------------------------- hook wiring


def _scheduler_src():
    return (pathlib.Path(__file__).resolve().parents[4] / "python" / "flliper"
            / "srt" / "managers" / "scheduler.py").read_text()


def test_hook_agrees_the_tips_before_binding_and_hands_them_to_the_bind():
    import ast

    src = _scheduler_src()
    assert "l15_tree_cand.build(" in src
    # the single collective sits BEFORE the first rank-local step of the branch
    i_build = src.index("l15_tree_cand.build(")
    i_tp = src.index("_tp = int(", i_build)
    i_base = src.index("_base_ok = all(", i_build)
    assert i_build < i_tp < i_base
    # it is guarded by the SLEEP-AGREE gather (no collective without it)
    guard = src[src.rindex("_l15_tree_reqs = []", 0, i_build):i_build]
    assert "_l15_agree_on" in guard
    calls = [c for c in ast.walk(ast.parse(src)) if isinstance(c, ast.Call)
             and getattr(c.func, "attr", None) == "build_retain_kwargs"]
    kws = {k.arg: ast.unparse(k.value) for k in calls[0].keywords}
    assert "tree_cand_max" in kws
    assert "_l15_tree_reqs" in src[src.index("_reqs = list("):src.index("_rgid = getattr(", src.index("_reqs = list("))]
