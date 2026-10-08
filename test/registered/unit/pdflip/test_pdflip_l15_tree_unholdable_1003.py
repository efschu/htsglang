# SPDX-License-Identifier: Apache-2.0
"""L15-UNHOLDABLE: agreed tree tips must actually be held.

27B boot y8j (image rc12z30y8j-27b-nf, f77461846b), D log 06:31:38:
``L15-TREE-CAND local=4 agreed=4 rids=tree:c2610ea159871156(24651),...`` on
TP0/1/2, then ``L15-TREE-CAND bind used=0 covered=0 unholdable=4`` on every
rank, ``#904 match-census ... refusers=MambaComponent:39
why=MambaComponent:absent=39`` between them, and ``L15-RETAIN n=1`` (only the
live rid). 06:36:10: agreed=5, unholdable=5, ``n=0 nothing_to_hold``.

ROOT. The boot runs bigram radix keys (DFLASH + FLLIPER_HICACHE_BIGRAM_KEYS=1,
``bigram=True`` in the log). A bigram node key carries ``units + 1`` raw
tokens and a child's first raw token IS its parent's last one.
``l15_tree_cand.chain_tokens`` concatenated the keys as-is, so the candidate's
chain repeated the boundary token at every node edge; ``match_parked`` then
matched the first node only (the next child key ``(t, t)`` does not exist),
landed on a node without a mamba value and raised -> "unholdable", silently.

A. The REAL UnifiedRadixCache with bigram keys (the end-anchor harness): two
   finished requests sharing a prefix -> two tips below a split node. RED on
   base (chain doubled, bind unholdable=2), GREEN with the fix (both held).
B. The probe gather: every rank tests every agreed tip, ONE more gather keeps
   the tips every rank can hold (rank-uniform), the dropped ones are named per
   candidate (``L15-TREE-CAND unholdable at=probe ... rids=tree:..(why@rN)``).
C. The bind names the reason per candidate (``at=bind``).
"""

from __future__ import annotations

import pathlib
import sys
from array import array
from types import SimpleNamespace

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

import pytest  # noqa: E402
import torch  # noqa: E402

from flliper.srt.mem_cache.radix_cache import RadixKey  # noqa: E402
from flliper.srt.mem_cache.unified_cache_components.tree_component import (  # noqa: E402
    ComponentType,
)
from flliper.srt.pdflip import l15_bind, l15_tree_cand  # noqa: E402
from test_pdflip_l15_bind_1001 import (  # noqa: E402
    _load_retain_test_module,
    _req_to_token,
)

FULL, MAMBA = int(ComponentType.FULL), int(ComponentType.MAMBA)


# ------------------------------------------------- A. real bigram tree


def _real_tree_two_tips():
    import test_pdflip_end_anchor_exact_probe_0928 as ea

    fx = ea._fixture(exact=True, bigram=True)
    shared = list(range(1000, 1020))
    a = shared + list(range(2000, 2012))
    b = shared + list(range(3000, 3009))
    for rid, ids in (("a", a), ("b", b)):
        req = ea._req(fx, ids, rid)
        ea._finish(fx, req)
    return fx, a, b


def test_real_bigram_tree_chain_is_the_inserted_token_sequence():
    fx, a, b = _real_tree_two_tips()
    tips = l15_tree_cand.tips_of(fx.cache)
    assert len(tips) == 2
    root = fx.cache.root_node
    chains = sorted(l15_tree_cand.chain_tokens(t, root) for t in tips)
    # each tip's chain is a prefix of exactly one inserted prompt and runs
    # through the split node: no doubled boundary token
    for ch in chains:
        full = a if ch[20] == 2000 else b
        assert ch == full[: len(ch)], (ch, full)
        assert len(ch) >= 21
    # the real match resolves the WHOLE chain to the tip
    for t in tips:
        ch = l15_tree_cand.chain_tokens(t, root)
        from flliper.srt.mem_cache.base_prefix_cache import MatchPrefixParams

        res = fx.cache.match_prefix(MatchPrefixParams(
            key=RadixKey(array("q", ch)), cow_mamba=False))
        assert res.last_device_node is t
        assert len(res.device_indices) == len(ch) - 1


def test_real_bigram_tree_agreed_tips_are_holdable():
    fx, _, _ = _real_tree_two_tips()
    logged = []
    reqs = l15_tree_cand.build(fx.cache, lambda v: [v, v, v], 0, {}, logged.append,
                               probe=lambda r: l15_bind.tree_probe(r, fx.cache))
    assert len(reqs) == 2, logged
    assert any("L15-TREE-CAND probe agreed=2 holdable=2 unholdable=0" in m
               for m in logged), logged
    for r in reqs:
        slots, node, anchor = l15_bind.match_parked(r, fx.cache)
        assert len(slots) == len(r.l15_tree_tokens) - 1
        assert node.component_data[ComponentType.MAMBA].value is not None
        assert anchor > 0


def test_real_bigram_tree_bind_holds_the_tips(tmp_path):
    fx, _, _ = _real_tree_two_tips()
    rt = _load_retain_test_module()
    logged = []
    sc = rt.make_scenario(tmp_path, [])
    kwargs = dict(sc["kwargs"])
    for key in ("candidates", "node_of", "slots_of", "anchor_slot_of",
                "l2_of", "rewrite_tree"):
        kwargs.pop(key)
    kwargs["log"] = logged.append
    kwargs["caps_rows_by_rank"] = (10_000, 10_000)
    reqs = l15_tree_cand.build(fx.cache, lambda v: [v], 0, {}, None)
    bound = l15_bind.build_retain_kwargs(reqs, _req_to_token(), tree_cache=fx.cache,
                                         **kwargs)
    assert any("L15-TREE-CAND bind used=2 covered=0 unholdable=0" in m
               for m in logged), logged
    assert {c.rid for c in bound["candidates"]} == {r.rid for r in reqs}


# --------------------------------------------- chain under bigram keys


class _BN:
    def __init__(self, parent, raw, *, anchor=None):
        self.parent = parent
        self.children = {}
        self.key = RadixKey(array("q", raw), None, is_bigram=True)
        self.last_access_time = 1.0
        cds = [SimpleNamespace(value=None, host_value=None) for _ in range(3)]
        cds[FULL].value = torch.arange(max(len(raw) - 1, 0), dtype=torch.int64) + 1
        if anchor is not None:
            cds[MAMBA].value = torch.tensor([anchor], dtype=torch.int64)
        self.component_data = cds
        if parent is not None:
            parent.children[len(parent.children)] = self


def test_bigram_chain_keeps_the_shared_boundary_token_once():
    root = _BN(None, [])
    a = _BN(root, [10, 11, 12, 13])          # 3 bigrams
    b = _BN(a, [13, 14, 15], anchor=5)       # starts at a's last raw token
    c = _BN(b, [15, 16], anchor=6)
    assert l15_tree_cand.chain_tokens(c, root) == [10, 11, 12, 13, 14, 15, 16]
    assert l15_tree_cand.chain_tokens(b, root) == [10, 11, 12, 13, 14, 15]
    # a key not composing (boundary differs) is no candidate, not a crash
    bad = _BN(a, [99, 98], anchor=7)
    with pytest.raises(l15_tree_cand.ChainError):
        l15_tree_cand.chain_tokens(bad, root)
    tree = SimpleNamespace(root_node=root)
    got = l15_tree_cand.local_candidates(tree, 8)
    assert [x.tokens for x in got] == [(10, 11, 12, 13, 14, 15, 16)]


# ------------------------------------------------ B. the probe gather


def _pseudo(tokens):
    c = l15_tree_cand.TreeCand(
        digest=l15_tree_cand.digest_of(tokens), n_tokens=len(tokens),
        last_access=1.0, tokens=tuple(tokens))
    return l15_tree_cand.pseudo_req(c, 0, 1)


def test_probe_gather_keeps_only_what_every_rank_can_hold():
    r1, r2, r3 = _pseudo([1, 2]), _pseudo([3, 4]), _pseudo([5, 6])
    reqs = [r1, r2, r3]
    # rank 0 can hold all, rank 1 not r2, rank 2 not r3
    votes = [
        [(r1.rid, None), (r2.rid, None), (r3.rid, None)],
        [(r1.rid, None), (r2.rid, "no_mamba"), (r3.rid, None)],
        [(r1.rid, None), (r2.rid, None), (r3.rid, "partial_match")],
    ]
    kept, dropped = l15_tree_cand.agree_holdable(reqs, lambda r: None,
                                                 lambda _mine: votes)
    assert [r.rid for r in kept] == [r1.rid]
    assert dropped == {r2.rid: "no_mamba@r1", r3.rid: "partial_match@r2"}


def test_probe_gather_is_entered_with_an_empty_vote_when_nothing_agreed():
    calls = []

    def gather(v):
        calls.append(v)
        return [v]

    class Boom:
        @property
        def root_node(self):
            raise RuntimeError("tree walk blew up")

    assert l15_tree_cand.build(Boom(), gather, 0, {}, None,
                               probe=lambda r: None) == []
    assert calls == [[], []], "both gathers entered, both with empty votes"


def test_probe_error_is_a_named_refusal_not_an_escape():
    r = _pseudo([1, 2])

    def boom(_r):
        raise KeyError("x")

    kept, dropped = l15_tree_cand.agree_holdable([r], boom, lambda m: [m])
    assert kept == [] and dropped == {r.rid: "probe_error:KeyError@r0"}


def test_build_names_the_unholdable_reason_per_candidate():
    tree = SimpleNamespace(root_node=_BN(None, []))
    a = _BN(tree.root_node, [1, 2, 3], anchor=4)
    _ = a
    logged = []
    out = l15_tree_cand.build(tree, lambda v: [v], 0, {}, logged.append,
                              probe=lambda r: "no_mamba")
    assert out == []
    line = [m for m in logged if "L15-TREE-CAND unholdable at=probe" in m]
    assert line and "n=1 why=no_mamba@r0:1" in line[0], logged
    assert "(no_mamba@r0)" in line[0]


# ---------------------------------------------- C. reason codes at the bind


class _PartialTree:
    """match_prefix returns a SHORTER span than the tip's chain (the y8j
    shape: the doubled boundary token stops the walk)."""

    is_eagle = True
    page_size = 1

    def __init__(self, rt, node_anchor):
        self.node = rt.FakeNode([])
        cds = [SimpleNamespace(value=None, host_value=None) for _ in range(3)]
        cds[MAMBA] = SimpleNamespace(value=node_anchor, host_value=None)
        self.node.component_data = cds

    def match_prefix(self, params):
        return SimpleNamespace(device_indices=torch.tensor([5, 6], dtype=torch.int64),
                               last_device_node=self.node)


def test_match_parked_reasons_are_named():
    rt = _load_retain_test_module()
    tip = _pseudo([1, 2, 3, 4, 5])  # bigram: 4 units expected
    t = _PartialTree(rt, torch.tensor([3], dtype=torch.int64))
    assert l15_bind.tree_probe(tip, t) == "partial_match"
    t2 = _PartialTree(rt, None)
    tip3 = _pseudo([1, 2, 3])        # 2 units expected, 2 matched -> anchor check
    assert l15_bind.tree_probe(tip3, t2) == "no_mamba"
    t2.node.component_data[MAMBA].value = torch.tensor([3], dtype=torch.int64)
    assert l15_bind.tree_probe(tip3, t2) is None


def test_bind_logs_the_reason_per_unholdable_candidate(tmp_path):
    rt = _load_retain_test_module()
    logged = []
    sc = rt.make_scenario(tmp_path, [])
    kwargs = dict(sc["kwargs"])
    for key in ("candidates", "node_of", "slots_of", "anchor_slot_of",
                "l2_of", "rewrite_tree"):
        kwargs.pop(key)
    kwargs["log"] = logged.append
    kwargs["caps_rows_by_rank"] = (10, 10)
    tip = _pseudo([1, 2, 3, 4, 5])
    bound = l15_bind.build_retain_kwargs(
        [tip], _req_to_token(), tree_cache=_PartialTree(rt, torch.tensor([3])),
        **kwargs)
    assert bound["candidates"] == []
    assert any("unholdable=1" in m for m in logged)
    line = [m for m in logged if "L15-TREE-CAND unholdable at=bind" in m]
    assert line and "why=partial_match:1" in line[0], logged
    assert "%s(partial_match)" % tip.rid in line[0]


# -------------------------------------------------------- hook wiring


def test_hook_passes_the_probe_to_the_tree_candidate_build():
    src = (pathlib.Path(__file__).resolve().parents[4] / "python" / "flliper"
           / "srt" / "managers" / "scheduler.py").read_text()
    i = src.index("l15_tree_cand.build(")
    call = src[i:src.index("\n                                )", i)]
    assert "probe=" in call and "l15_bind.tree_probe(" in call
