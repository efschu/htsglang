"""Review NF rc32 KOPF-ANKER (d69a44348b) merged onto rc33 (fbb680d6d3, KV-Leiter + forkanchor2), 22.10.2026.

Pins the danger directions the author's file does not reach on the merged tree:

  R1  head (group P) and branch (group D) never plan in the same process: a P step with a branching point handed over
      plans the head tracks and no branch track (and consumes the hand-over); a D step plans the branch track and no
      head track. Mutant: branch on P (a rank-local branching point on a PP stage) -> red.
  R2  the head anchors stay OUT of the per-request arena share (H19 SHARE, cap 21): their nodes carry no
      ``pdflip_anchor_rid`` after the retain / chunk tagging, so ``owned_anchors`` never counts them and SHARE / the
      FULL foreign pick never take them or take a turn-end / branch anchor because of them. Mutant: tag the whole
      retain chain -> red.
  R3  the stale-plan paths of both kinds free their slots in the merged ``note_step``; ``pop_all_pending`` hands both
      kinds out in position order with their own labels.
  R4  the anchor sits where a BIGRAM reader that shares the head through the close can still match: the node key of
      ``t = floor_page(b - 1)`` ends at token ``t`` <= ``b - 1`` (the last shared id), ``floor_page(b)`` on a
      page-aligned close would need ``ids[b]`` -- the very id where the branch differs.
"""

from types import SimpleNamespace

import pytest

from flliper.srt.pdflip import mamba_arena_displace as mad
from flliper.srt.pdflip import turn_anchor as ta
from flliper.srt.pdflip import twin_anchor as twa
from flliper.test.ci.ci_register import register_cpu_ci

import test_nf_rc32_kopfanker_1022 as K   # the author's harness (PYTEST_PYTHONPATH carries the pdflip test dir)

register_cpu_ci(est_time=10, suite="stage-a-test-cpu")

HEADS_8_255 = [17472, 17984, 18112]


@pytest.fixture(autouse=True)
def _clean(monkeypatch):
    monkeypatch.delenv("FLLIPER_PDFLIP_GROUP", raising=False)
    monkeypatch.setattr(ta, "_INFLIGHT", [])
    twa._reset_for_test()
    yield
    twa._reset_for_test()


def _d(monkeypatch, worker=False):
    monkeypatch.setenv("FLLIPER_PDFLIP_GROUP", "D")
    monkeypatch.setenv("FLLIPER_PDFLIP_TURN_ANCHOR_TOKEN", str(K.IM))
    monkeypatch.setenv("FLLIPER_PDFLIP_GEN_FORK_ANCHOR", "0")
    monkeypatch.setattr(ta, "d_split_only_rank", lambda: worker)


# -- R1: head on P, branch on D, never both -------------------------------------------------------------------------
def test_p_step_with_a_branch_handover_plans_heads_and_no_branch():
    req = K._sreq(K._src_b())
    setattr(req, ta.BRANCH_POS_ATTR, 17920)            # a stray hand-over on a P stage
    K._plan(req, K._Alloc(range(100, 120)))
    assert [p[0] for p in getattr(req, ta.HEAD_PENDING_ATTR)] == HEADS_8_255
    assert getattr(req, ta.BRANCH_PENDING_ATTR, None) is None
    assert getattr(req, ta.BRANCH_POS_ATTR, None) is None, "the hand-over is consumed on every group"


@pytest.mark.parametrize("worker", [False, True], ids=["tp0", "worker"])
def test_d_step_plans_the_branch_and_no_head(monkeypatch, worker):
    _d(monkeypatch, worker)
    req = K._sreq(K._src_b())
    setattr(req, ta.BRANCH_POS_ATTR, 17920)
    K._plan(req, K._Alloc(range(100, 120)))
    assert not getattr(req, ta.HEAD_PENDING_ATTR, None)
    br = getattr(req, ta.BRANCH_PENDING_ATTR, None)
    assert br is not None and br[0] == 17920
    assert (br[1] is None) == worker                     # split-only on a worker, a slot on TP0


def test_both_kinds_are_known_to_the_pending_helpers():
    req = SimpleNamespace()
    setattr(req, ta.HEAD_PENDING_ATTR, [(17472, None, None, 0, 1), (18112, None, None, 0, 1)])
    setattr(req, ta.BRANCH_PENDING_ATTR, (17920, None, None, 0, 1))
    assert ta.has_pending(req)
    got = [(k, p[0]) for k, p in ta.pop_all_pending(req)]
    assert got == [("head", 17472), ("branch", 17920), ("head", 18112)]
    assert not ta.has_pending(req)
    assert ta._label("head") == "HEAD-ANCHOR" and ta._label("branch") == "BRANCH-ANCHOR"
    assert ta._kind_prefix("head") == "head_" and ta._kind_prefix("branch") == "branch_"
    only_head = SimpleNamespace()
    setattr(only_head, ta.HEAD_PENDING_ATTR, [(17472, None, None, 0, 1)])
    assert ta.has_pending(only_head), "a head plan alone is a pending plan"


# -- R3: stale plans of both kinds ----------------------------------------------------------------------------------
def test_stale_head_and_branch_plans_free_their_slots():
    alloc = K._Alloc(range(1, 40))
    req = K._sreq(K._src_b())
    setattr(req, ta.HEAD_PENDING_ATTR, [(17472, K.torch.tensor([31]), None, 0, 1),
                                        (17984, K.torch.tensor([32]), None, 0, 1)])
    setattr(req, ta.BRANCH_PENDING_ATTR, (17920, K.torch.tensor([33]), None, 0, 1))
    req.output_ids = [1]
    K._plan(req, alloc, prefix=K.N_8_255, end=K.N_8_255 + 1, main=K.N_8_255)
    assert {31, 32, 33} <= set(alloc.freed)
    assert getattr(req, ta.HEAD_PENDING_ATTR, None) is None and getattr(req, ta.BRANCH_PENDING_ATTR, None) is None


# -- R2: the head anchors are outside the per-request arena share -----------------------------------------------------
def _nodes_by_depth(cache):
    out, queue = {}, [(c, len(c.key)) for c in cache.root_node.children.values()]
    while queue:
        n, d = queue.pop()
        out[d] = n
        queue.extend((c, d + len(c.key)) for c in n.children.values())
    return out


def test_head_nodes_stay_untagged_and_out_of_the_share(monkeypatch):
    from flliper.srt.mem_cache import unified_radix_cache as urc

    monkeypatch.setenv("FLLIPER_PDFLIP_GROUP", "P")
    monkeypatch.setenv("FLLIPER_PDFLIP_BIGRAM_ANCHOR_EXACT", "1")
    monkeypatch.setenv("FLLIPER_PDFLIP_SYS_GRID_ANCHOR", "0")
    monkeypatch.setattr(urc, "_PDFLIP_END_ANCHOR", True)
    fx = K._fixture(False)
    req = K._one_forward(fx)
    rid = str(req.rid)
    nodes = _nodes_by_depth(fx.cache)
    heads = [t for t in (K.C1 - 1, K.C2 - 1) if t in nodes]
    assert heads == [K.C1 - 1, K.C2 - 1], sorted(nodes)
    # what the retain (``_pdflip_tag_retain_chain``) and the chunk publish (``_pdflip_publish_at_chunk``) tag
    chain = fx.cache._pdflip_chain_from(req.last_node)
    fx.cache._pdflip_tag_retain_chain(chain, rid)
    fx.cache._pdflip_tag_anchor(chain[-1], rid)
    for t in heads:
        assert nodes[t].pdflip_anchor_rid is None, t
    path, _depth = mad.ancestor_path(target=chain[-1], root=fx.cache.root_node)
    owned = mad.owned_anchors(path=path + [(chain[-1], _depth)], rid=rid, anchor_of=lambda n: (True, (1,)))
    assert all(a.depth not in heads for a in owned), [a.depth for a in owned]
    # the FULL foreign pick reads tagged anchors only: a head node is never another request's victim either
    tree = mad.tree_anchors(root=fx.cache.root_node, anchor_of=lambda n: (True, (1,)))
    assert all(a.depth not in heads for a in tree)


# -- R4: bigram key bound at a page-aligned close ---------------------------------------------------------------------
def test_page_aligned_close_anchor_keeps_the_bigram_key_inside_the_shared_ids():
    b = 18048                                             # '>' token on a page start
    ids = K._src_b(closes=(b,))
    t = ta.head_positions(ids, K.IM, K.PAGE)[0]
    assert t == b - K.PAGE
    # a bigram node key of length t pairs ids[i] with ids[i + 1] for i < t: its last pair reads ids[t] -- shared
    # with the reader (ids[:b]) only while t <= b - 1
    assert t <= b - 1
    assert (b // K.PAGE) * K.PAGE == b, "floor_page(b) would read ids[b], where the branch differs"


# -- R5: the system part's closes never eat the cap ------------------------------------------------------------------
def test_a_close_in_the_system_part_is_not_a_head_close():
    """A system prompt that quotes '</system-reminder>' (SYS-GRID's domain, below u) must not take head slots: with
    four such quotes the lowest-first cap would otherwise leave the real head closes without a track."""
    ids = K._src_b()
    for b in (1000, 1500, 2000, 2500):
        K._put_close(ids, b)
    assert ta.head_positions(ids, K.IM, K.PAGE) == HEADS_8_255 + [19456]
