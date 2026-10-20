"""KNOTENTEILUNG (NF rc28, 20.10.): every rank of the D group holds the same radix nodes.

Boot ``boot_weg2_..._297198745d_1020_174205`` D log, Form A token cut (TP0 host, TP1/TP2
workers): ``PDFLIP PUBLISH-SPLIT`` TP0 7x, TP1/TP2 8x. At 18:10:02 TP0 held 446/447
[83072, 87168) + 434 [87168, 87296) / 437 [87168, 87616), the workers 454 [83072, 86976) +
452 [86976, 87296) and 456 [83072, 85248) + 457 [85248, 87616). Two rank-local inputs to a
tree edit (pdflip/rank_uniform_split.py):

1. pdflip-4-201's chunk inserts. TP0's admission walk found the branching point 83072
   (``MATCH-CENSUS-DEEP reached=83072 accepted=82880 branching=83072``), the workers'
   follow re-match (key cut at the group depth 82880) found none (``#59 FINISH-TRACK
   branching=83072`` on TP0, ``branching=-`` on TP1/TP2). The extend tracked the mamba
   state at 83072 on TP0 and at the chunk end 85248 on the workers, and the chunk insert
   stops at the track: TP0 [.., 83072) then one leaf [83072, 87616), the workers
   [83072, 85248) + [85248, 87616).
2. The publish split. 18:09:39 TP0's sweep stopped after ``BACKUP-REFUSED
   why=mamba_claim`` (``PUBLISH-SWEEP unbacked=1 refused=1``), the workers' went on and
   split pdflip-4-185's 4416-token node from its start (+4096 -> 86976). 18:10:02 TP0
   split its nodes from THEIR starts (83072 + 4096 -> 87168).

Driven through the REAL ``MambaComponent.finalize_match_result`` (+ the real follow
re-match), the REAL ``ScheduleBatch._mamba_radix_cache_v2_req_prepare_for_extend`` (the
track = the insert length) and a REAL CPU ``UnifiedRadixCache`` (insert, split, publish
split). RED on rc29 15943627cd, GREEN with the fix; switch 0 = the rc28 form.
"""

from __future__ import annotations

import contextlib
import types
import unittest
from array import array

import torch

from flliper.srt import rank_role
from flliper.srt.configs.mamba_utils import Mamba2CacheParams, Mamba2StateShape
from flliper.srt.environ import envs
from flliper.srt.managers import tp_match_floor as m
from flliper.srt.managers.schedule_batch import ScheduleBatch
from flliper.srt.mem_cache.allocator import TokenToKVPoolAllocator
from flliper.srt.mem_cache.base_prefix_cache import InsertParams, MatchPrefixParams, MatchResult
from flliper.srt.mem_cache.cache_init_params import CacheInitParams
from flliper.srt.mem_cache.memory_pool import HybridLinearKVPool, HybridReqToTokenPool
from flliper.srt.mem_cache.radix_cache import RadixKey
from flliper.srt.mem_cache.unified_cache_components.mamba_component import MambaComponent
from flliper.srt.mem_cache.unified_cache_components.tree_component import ComponentType
from flliper.srt.mem_cache.unified_radix_cache import UnifiedRadixCache
try:  # absent on rc29 15943627cd (the red run)
    from flliper.srt.pdflip import rank_uniform_split as rus
except ImportError:  # pragma: no cover
    rus = None
from flliper.srt.server_args import ServerArgs, set_global_server_args_for_scheduler

PAGE = 64
CHUNK = 64  # FLA chunk = mamba_cache_chunk_size on the NF D group
ROLES = ("host", "worker", "worker")
FLOOR_ATTR = "_tp_match_floor_group"
FOLLOW_ATTR = "_tp_match_floor_follow_walk"

# rc28 depths (D log 116585 ff., 118399 ff.)
OLD_ANCHOR = 82880  # pdflip-4-185's anchor = the group depth for pdflip-4-201
BRANCH = 83072  # where pdflip-4-201 leaves pdflip-4-185 (MATCH-CENSUS-DEEP)
CHUNK1_END = 85248  # 82880 + 2368 (#969 EXTENT n=211)
CHUNK2_END = 87616  # 85248 + 2368 (#969 EXTENT n=212)
OLD_END = 87296  # pdflip-4-185's finish (FINISH-TRACK cache_len=87296)


def _server_args():
    sa = ServerArgs(model_path="dummy", page_size=PAGE)
    sa._mamba_cache_chunk_size = CHUNK
    set_global_server_args_for_scheduler(sa)
    return sa


@contextlib.contextmanager
def _as_rank(rank, roles=ROLES):
    prev = (rank_role._INSTALLED_PLAN, rank_role._INSTALLED_RANK)
    plan = None if roles is None else rank_role.RankRolePlan(tuple(roles))
    rank_role.set_form_a_role_plan(plan, rank)
    try:
        yield
    finally:
        rank_role._INSTALLED_PLAN, rank_role._INSTALLED_RANK = prev


@contextlib.contextmanager
def _switches(follow=True, branching=True, grid=True):
    """The switches by NAME: on rc29 15943627cd the two new ones do not exist
    (= the rc28 form), so this file runs there too -- that is the red."""
    with contextlib.ExitStack() as st:
        for name, val in (
            ("FLLIPER_PDFLIP_ENABLE_FORM_A_TP0_FOLLOW", follow),
            ("FLLIPER_PDFLIP_FORM_A_UNIFORM_BRANCHING", branching),
            ("FLLIPER_PDFLIP_PUBLISH_SPLIT_GRID", grid),
        ):
            field = getattr(envs, name, None)
            if field is not None:
                st.enter_context(field.override(val))
        yield


# ---------------------------------------------------------------------------
# 1. the branching point at admission (real finalize_match_result)


class _WalkTree:
    """One rank's tree as the match walk sees it: node ends on the request's key path
    (``bounds``) and the depths at which this rank's validator accepts an anchor. The
    walk's value chunks are the node spans up to the reach (a key ending inside a node
    ends the last chunk there, as the real walk's split does); best = the deepest
    accepted anchor."""

    def __init__(self, bounds, anchors, reach):
        self.bounds = sorted(bounds)
        self.anchors = sorted(anchors)
        self.reach = reach
        self.root_node = types.SimpleNamespace(
            name="root", component_data=[None, None, types.SimpleNamespace(value=None, host_value=None)]
        )
        self.cache_controller = object()
        self.is_eagle = False
        self.is_chunk_cache = lambda: False
        self.supports_mamba = lambda: True
        self.swa_reprefill_tail_tokens = lambda: 0

    def _node(self, n):
        if n <= 0:
            return self.root_node
        host = torch.tensor([3]) if n in self.anchors else None
        return types.SimpleNamespace(
            name=f"n{n}", component_data=[None, None, types.SimpleNamespace(value=None, host_value=host)]
        )

    def match_prefix(self, params):
        reach = min(len(params.key) // PAGE * PAGE, self.reach)
        ends = [b for b in self.bounds if b < reach] + [reach]
        chunks, prev = [], 0
        for e in ends:
            chunks.append(torch.zeros(e - prev, dtype=torch.int64))
            prev = e
        following = getattr(self, FOLLOW_ATTR, False)
        if following:
            best_depth = reach
        else:
            best_depth = max([a for a in self.anchors if a <= reach], default=0)
        best_len = sum(1 for e in ends if e <= best_depth)
        res = MatchResult(
            device_indices=torch.empty(0, dtype=torch.int64),
            last_device_node=self.root_node,
            last_host_node=self._node(best_depth),
            best_match_node=self._node(best_depth),
            host_hit_length=best_depth,
        )
        if not params.cow_mamba:
            return res
        comp = types.SimpleNamespace(
            component_type=ComponentType.MAMBA,
            mamba_checkpoint_interval=None,
            mamba_ckpt_strict_resume=False,
            cache=self,
            _stateless_resume_refusals=0,
            _foreign_pool_resume_refusals=0,
        )
        comp._raw_token_pos = lambda depth: depth
        comp.create_match_validator = types.MethodType(MambaComponent.create_match_validator, comp)
        self._components_tuple = (comp,)
        out = MambaComponent.finalize_match_result(
            comp, result=res, params=params, value_chunks=chunks, best_value_len=best_len
        )
        # the last line of the REAL UnifiedRadixCache._match_post_processor (its wiring is
        # pinned by TestPostProcessorWiring below); on rc29 the hook does not exist
        hook = getattr(UnifiedRadixCache, "_knoten_uniform_branching", None)
        return out if hook is None else hook(self, out, chunks)


def _req(rid, n):
    return types.SimpleNamespace(
        rid=rid,
        origin_input_ids=array("q", range(n)),
        output_ids=array("q"),
        extra_key=None,
        mamba_pool_idx=0,
        positional_embed_overrides=None,
        _compute_max_prefix_len=lambda k: max(k - 1, 0),
    )


def _admit(tree, rank, planted, rid, n, roles=ROLES):
    """The admission match on this rank (cow_mamba, the group depth planted)."""
    req = _req(rid, n)
    toks = array("q", list(req.origin_input_ids))
    params = MatchPrefixParams(
        key=RadixKey(toks, None, limit=req._compute_max_prefix_len(len(toks))), cow_mamba=True, req=req
    )
    setattr(tree, FLOOR_ATTR, planted)
    try:
        with _as_rank(rank, roles):
            out = tree.match_prefix(params)
    finally:
        setattr(tree, FLOOR_ATTR, None)
    return out


def _rc28_trees():
    # the same KV path on every rank (pdflip-4-185 to 87296, pdflip-4-201 shares it to
    # 83072); TP0's anchor at 82880 (host, accepted), the workers' byteless anchors at
    # 82880 and 79936 (MAMBA-HOST-RESUME on TP1/TP2 at 79936, 18:09:38)
    bounds = [79936, OLD_ANCHOR]
    return {
        0: _WalkTree(bounds, [OLD_ANCHOR], BRANCH),
        1: _WalkTree(bounds, [79936, OLD_ANCHOR], BRANCH),
        2: _WalkTree(bounds, [79936, OLD_ANCHOR], BRANCH),
    }


def _track(branching, prefix, extend):
    """The REAL extend track (= the chunk insert's length, ``effective_cache_len``)."""
    sa = _server_args()
    self = types.SimpleNamespace(
        req_to_token_pool=types.SimpleNamespace(get_mamba_ping_pong_other_idx=lambda i: 1 - i),
        tree_cache=None,
    )
    req = types.SimpleNamespace(
        rid="pdflip-4-201",
        prefix_indices=torch.zeros(prefix, dtype=torch.int64),
        extend_range=types.SimpleNamespace(length=extend),
        mamba_ping_pong_track_buffer=torch.tensor([5, 6]),
        mamba_next_track_idx=0,
        mamba_branching_seqlen=branching,
        mamba_last_track_seqlen=None,
    )
    assert sa.mamba_checkpoint_interval is None
    ScheduleBatch._mamba_radix_cache_v2_req_prepare_for_extend(self, req)
    return req.mamba_last_track_seqlen


class TestBranchingIsTheGroups(unittest.TestCase):
    def setUp(self):
        _server_args()

    def _branchings(self, **sw):
        trees = _rc28_trees()
        planted = {"pdflip-4-201": OLD_ANCHOR}
        with _switches(**sw):
            return {r: _admit(t, r, planted, "pdflip-4-201", 87987).mamba_branching_seqlen for r, t in trees.items()}

    def test_rc28_every_rank_carries_the_branching_point(self):
        got = self._branchings()
        self.assertEqual(got, {0: BRANCH, 1: BRANCH, 2: BRANCH})

    def test_rc28_chunk_inserts_stop_at_the_same_depth(self):
        got = self._branchings()
        tracks = {r: _track(b, OLD_ANCHOR, CHUNK1_END - OLD_ANCHOR) for r, b in got.items()}
        self.assertEqual(tracks, {0: BRANCH, 1: BRANCH, 2: BRANCH})
        # the second chunk: the branching point lies below its prefix -> the chunk end
        tracks2 = {r: _track(b, CHUNK1_END, CHUNK2_END - CHUNK1_END) for r, b in got.items()}
        self.assertEqual(tracks2, {0: CHUNK2_END, 1: CHUNK2_END, 2: CHUNK2_END})

    def test_no_branching_when_the_walk_ends_at_the_admitted_depth(self):
        """pdflip-4-185's own resume (the key ends on the anchor): nothing beyond the
        admitted depth, no branching point on any rank."""
        trees = {r: _WalkTree([79936, OLD_ANCHOR], [79936, OLD_ANCHOR], OLD_ANCHOR) for r in range(3)}
        planted = {"pdflip-4-185": OLD_ANCHOR}
        with _switches():
            got = {r: _admit(t, r, planted, "pdflip-4-185", 87987).mamba_branching_seqlen for r, t in trees.items()}
        self.assertEqual(got, {0: None, 1: None, 2: None})

    def test_switch_off_is_the_rc28_split(self):
        got = self._branchings(branching=False)
        self.assertEqual(got[0], BRANCH)
        self.assertIsNone(got[1])
        tracks = {r: _track(b, OLD_ANCHOR, CHUNK1_END - OLD_ANCHOR) for r, b in got.items()}
        self.assertEqual(tracks, {0: BRANCH, 1: CHUNK1_END, 2: CHUNK1_END})

    def test_rule_has_no_anchor_verdict(self):
        """A worker whose byteless anchor sits at the reach (its own best = reach): the
        old condition gave it no branching point; the group's rule does."""
        trees = _rc28_trees()
        trees[1] = _WalkTree([79936, OLD_ANCHOR, BRANCH], [OLD_ANCHOR, BRANCH], BRANCH)
        planted = {"pdflip-4-201": OLD_ANCHOR}
        with _switches():
            got = {r: _admit(t, r, planted, "pdflip-4-201", 87987).mamba_branching_seqlen for r, t in trees.items()}
        self.assertEqual(set(got.values()), {BRANCH})

    def test_host_h96_cap_rematch_carries_it(self):
        """TP0 capped below its own anchor by the group (H96 re-match on a cut key)."""
        tree = _WalkTree([79936, OLD_ANCHOR], [79936, OLD_ANCHOR], BRANCH)
        orig = m.group_floor_cap
        m_cap = lambda cache, req, result: (  # noqa: E731
            79936 if len(result.device_indices) + int(result.host_hit_length) > 79936 else None
        )
        import flliper.srt.mem_cache.unified_cache_components.mamba_component as mc

        mc.group_floor_cap, prev = m_cap, mc.group_floor_cap
        try:
            with _switches():
                out = _admit(tree, 0, None, "pdflip-4-201", 87987)
        finally:
            mc.group_floor_cap = prev
        self.assertIs(orig, m.group_floor_cap)
        self.assertEqual(out.mamba_branching_seqlen, BRANCH)

    def test_classic_boot_zero_path_untouched(self):
        """No Form A plan, RU zero (group_floor_zeroes): the pre-fix rule stays (the walk
        ended on its best anchor -> no branching point), the group rule would set one."""
        import flliper.srt.mem_cache.unified_cache_components.mamba_component as mc

        tree = _WalkTree([79936, OLD_ANCHOR, BRANCH], [OLD_ANCHOR, BRANCH], BRANCH)
        mc.group_floor_zeroes, prev = (lambda cache, req, result: True), mc.group_floor_zeroes
        try:
            with _switches():
                out = _admit(tree, 0, None, "pdflip-4-201", 87987, roles=None)
        finally:
            mc.group_floor_zeroes = prev
        self.assertEqual(len(out.device_indices) + int(out.host_hit_length or 0), 0)
        self.assertIsNone(out.mamba_branching_seqlen)

    def test_classic_boot_untouched(self):
        """No Form A plan: the pre-fix rule (anchor verdict in the condition)."""
        tree = _WalkTree([79936, OLD_ANCHOR, BRANCH], [OLD_ANCHOR, BRANCH], BRANCH)
        with _switches():
            out = _admit(tree, 0, None, "pdflip-4-201", 87987, roles=None)
        self.assertIsNone(out.mamba_branching_seqlen)


class TestPostProcessorWiring(unittest.TestCase):
    """The REAL ``_match_post_processor`` sets the group's branching point after the
    components (here a component that hands back a result without one, as the worker's
    follow re-match does)."""

    def test_post_processor_sets_it_after_the_components(self):
        _server_args()
        root = types.SimpleNamespace(parent=None, last_access_time=0.0)
        node = types.SimpleNamespace(parent=root, last_access_time=0.0)
        followed = MatchResult(
            device_indices=torch.empty(0, dtype=torch.int64),
            last_device_node=root,
            last_host_node=node,
            best_match_node=node,
            host_hit_length=OLD_ANCHOR,
        )
        comp = types.SimpleNamespace(
            component_type=ComponentType.MAMBA,
            mamba_checkpoint_interval=None,
            refresh_lru=lambda *a, **k: None,
            finalize_match_result=lambda result, params, value_chunks, best_value_len: followed,
        )
        tree = types.SimpleNamespace(
            _components_tuple=(comp,),
            cache_controller=object(),
            _empty_match_result=followed,
            _state_anchor_depth=lambda n: None,
            root_node=root,
        )
        value = [torch.zeros(79936, dtype=torch.int64), torch.zeros(OLD_ANCHOR - 79936, dtype=torch.int64),
                 torch.zeros(BRANCH - OLD_ANCHOR, dtype=torch.int64)]
        params = MatchPrefixParams(key=RadixKey(array("q", range(8)), None), cow_mamba=True, req=None)
        with _as_rank(1), _switches():
            out = UnifiedRadixCache._match_post_processor(tree, params, value, node, root, 0, 0)
        self.assertEqual(out.mamba_branching_seqlen, BRANCH)
        with _as_rank(1), _switches(branching=False):
            out = UnifiedRadixCache._match_post_processor(tree, params, value, node, root, 0, 0)
        self.assertIsNone(out.mamba_branching_seqlen)


# ---------------------------------------------------------------------------
# 2. the node cuts (real CPU UnifiedRadixCache)

W = 16  # the publish window, scaled (page_size 1 in the CPU fixture)


def _cpu_tree(world=3):
    sa = ServerArgs(model_path="dummy", page_size=1)
    sa._mamba_cache_chunk_size = 1
    set_global_server_args_for_scheduler(sa)
    with envs.FLLIPER_MAMBA_SSM_DTYPE.override("bfloat16"):
        shape = Mamba2StateShape.create(
            tp_world_size=1, intermediate_size=256, n_groups=1, num_heads=2, head_dim=16,
            state_size=16, conv_kernel=4,
        )
        cache_params = Mamba2CacheParams(shape=shape, layers=[0, 1, 2])
    pool = HybridReqToTokenPool(
        size=4, mamba_size=8, mamba_spec_state_size=4, max_context_len=512, device="cpu",
        enable_memory_saver=False, cache_params=cache_params, mamba_layer_ids=[0, 1, 2],
        enable_mamba_extra_buffer=False, speculative_num_draft_tokens=3,
    )
    kv_pool = HybridLinearKVPool(
        size=1024, dtype=torch.bfloat16, page_size=1, head_num=2, head_dim=8,
        full_attention_layer_ids=[3], device="cpu", enable_memory_saver=False, mamba_pool=pool.mamba_pool,
    )
    alloc = TokenToKVPoolAllocator(size=1024, dtype=torch.bfloat16, device="cpu", kvcache=kv_pool, need_sort=False)
    params = CacheInitParams(
        req_to_token_pool=pool, token_to_kv_pool_allocator=alloc, page_size=1, disable=False,
        sliding_window_size=None, tree_components=(ComponentType.FULL, ComponentType.MAMBA),
        enable_mamba_extra_buffer=False, enable_kv_cache_events=False, eviction_policy="lru", is_eagle=False,
    )
    cache = UnifiedRadixCache(params=params)
    cache.cache_init_params = params
    cache._attn_reduce_world = lambda: world
    cache._pdflip_publish_window = lambda: W
    cache._alloc = alloc
    return cache


def _insert(cache, toks):
    value = cache._alloc.alloc(len(toks))
    cache.insert(InsertParams(key=RadixKey(array("q", toks)), value=value))


def _spans(cache):
    out, stack = [], [(cache.root_node, 0)]
    while stack:
        node, depth = stack.pop()
        for child in node.children.values():
            end = depth + len(child.key)
            out.append((depth, end))
            stack.append((child, end))
    return sorted(out)


def _sweep_split(cache):
    """What a rank's publish sweep does to every node it reaches (all of them here):
    split a node longer than the window."""
    for node in [n for n in _nodes(cache) if len(n.key) > W]:
        cache._pdflip_split_for_publish(node, W)


def _nodes(cache):
    out, stack = [], [cache.root_node]
    while stack:
        node = stack.pop()
        for c in node.children.values():
            out.append(c)
            stack.append(c)
    return out


# scaled rc28: a common prefix node, pdflip-4-185 continues it to OLD (finish insert),
# pdflip-4-201 leaves it at BR and prefills two chunks (C1, C2)
PRE, OLD, BR, C1, C2 = 5, 50, 24, 32, 40
A = list(range(1, OLD + 1))
B = A[:BR] + list(range(1000, 1000 + (C2 - BR)))


def _rc28_history(cache, sweep, track1):
    _insert(cache, A[:PRE])
    _insert(cache, A)  # pdflip-4-185's finish: one leaf [PRE, OLD)
    if sweep:
        _sweep_split(cache)  # only where this rank's sweep reached
    _insert(cache, B[:track1])  # pdflip-4-201 chunk 1, up to the track
    _insert(cache, B[:C2])  # chunk 2


class TestNodeCutsAreTheGroups(unittest.TestCase):
    def test_rc28_form_parts_the_trees(self):
        """RED shape, switches 0: TP0 (sweep stopped, track at the branch) vs worker
        (sweep split from the node start, track at the chunk end)."""
        with _switches(branching=False, grid=False):
            tp0, wk = _cpu_tree(), _cpu_tree()
            _rc28_history(tp0, sweep=False, track1=BR)
            _rc28_history(wk, sweep=True, track1=C1)
        self.assertNotEqual(_spans(tp0), _spans(wk))

    def test_fixed_every_rank_same_nodes(self):
        with _switches():
            tp0, wk = _cpu_tree(), _cpu_tree()
            _rc28_history(tp0, sweep=False, track1=BR)
            _rc28_history(wk, sweep=True, track1=BR)
        self.assertEqual(_spans(tp0), _spans(wk))
        self.assertTrue(all(e - s <= W for s, e in _spans(tp0)))
        # a leaf not longer than the window stays whole (chunk 2: [24, 40), 16 = W)
        self.assertIn((BR, C2), _spans(tp0))
        self.assertEqual(
            _spans(tp0),
            [(0, PRE), (PRE, 16), (16, BR), (BR, 32), (BR, C2), (32, 48), (48, OLD)],
        )

    def test_insert_cuts_long_leaf_at_absolute_multiples(self):
        with _switches():
            t = _cpu_tree()
            _insert(t, A[:PRE])
            _insert(t, A)
        self.assertEqual(_spans(t), [(0, PRE), (PRE, 16), (16, 32), (32, 48), (48, OLD)])

    def test_sweep_split_is_absolute_and_confluent(self):
        """Fallback: a long node the insert did not cut (grid off at its insert) is cut at
        the same absolute depths whenever a rank's sweep reaches it."""
        with _switches(grid=False):
            a, b = _cpu_tree(), _cpu_tree()
            for t in (a, b):
                _insert(t, A[:PRE])
                _insert(t, A)
        with _switches():
            _sweep_split(a)
            _sweep_split(b)
            _sweep_split(b)  # again: nothing new
        self.assertEqual(_spans(a), _spans(b))
        self.assertEqual(_spans(a), [(0, PRE), (PRE, 16), (16, 32), (32, 48), (48, OLD)])

    def test_grid_switch_off_is_the_rc28_split(self):
        """Switch 0 on a TP group: no insert cut, the publish split from the node start."""
        with _switches(grid=False):
            t = _cpu_tree()
            _insert(t, A[:PRE])
            _insert(t, A)
            self.assertEqual(_spans(t), [(0, PRE), (PRE, OLD)])
            _sweep_split(t)
        self.assertEqual(_spans(t), [(0, PRE), (PRE, 21), (21, 37), (37, OLD)])

    def test_single_rank_group_untouched(self):
        """P (PP stages, world 1): no insert cut, the rc28 relative publish split."""
        with _switches():
            t = _cpu_tree(world=1)
            _insert(t, A[:PRE])
            _insert(t, A)
            self.assertEqual(_spans(t), [(0, PRE), (PRE, OLD)])
            _sweep_split(t)
        self.assertEqual(_spans(t), [(0, PRE), (PRE, 21), (21, 37), (37, OLD)])

    def test_tail_keeps_the_anchor_commit(self):
        """The insert attaches its mamba value to the TAIL of a cut leaf."""
        with _switches():
            t = _cpu_tree()
            _insert(t, A[:PRE])
            slot = t.req_to_token_pool.mamba_allocator.alloc(1)
            value = t._alloc.alloc(OLD)
            t.insert(InsertParams(key=RadixKey(array("q", A)), value=value, mamba_value=slot))
        by_span = {}
        for n in _nodes(t):
            s, p = 0, n.parent
            while p is not None:
                s += len(p.key or ())
                p = p.parent
            by_span[(s, s + len(n.key))] = n
        tail = by_span[(48, OLD)]
        self.assertIsNotNone(tail.component_data[ComponentType.MAMBA].value)
        for span in ((PRE, 16), (16, 32), (32, 48)):
            self.assertIsNone(by_span[span].component_data[ComponentType.MAMBA].value)


@unittest.skipIf(rus is None, "rank_uniform_split absent (rc29 15943627cd)")
class TestGridCuts(unittest.TestCase):
    def test_pure_and_confluent(self):
        self.assertEqual(rus.grid_cuts(83072, 4544, 4096), [86016 - 83072])
        self.assertEqual(rus.grid_cuts(82880, 4416, 4096), [86016 - 82880])
        self.assertEqual(rus.grid_cuts(0, 4096, 4096), [])
        self.assertEqual(rus.grid_cuts(5, 45, 16), [11, 27, 43])
        for s, n in ((5, 45), (83072, 9000), (4096, 8193)):
            pts = {s + c for c in rus.grid_cuts(s, n, 16)}
            # cutting each piece again yields nothing new
            edges = sorted({s, s + n} | pts)
            for lo, hi in zip(edges, edges[1:]):
                self.assertEqual(rus.grid_cuts(lo, hi - lo, 16), [])

    def test_uniform_branching(self):
        self.assertEqual(rus.uniform_branching([82880, 192], 64), BRANCH)
        self.assertEqual(rus.uniform_branching([82880, 200], 64), BRANCH)
        self.assertIsNone(rus.uniform_branching([32], 64))


if __name__ == "__main__":
    unittest.main()
