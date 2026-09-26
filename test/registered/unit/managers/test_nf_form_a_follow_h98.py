"""H98: on a Form A D group the attention host (TP0) is the ONLY authority for
the prefix match; the expert workers (TP1/TP2) follow it.

THE BUG (rc9i/rc9k/rc9l/rc9m are one bug). The NF D group runs Form A:
TP0 holds every attention/GDN head, the KV, the mamba state and the draft
(``q heads split [24, 0, 0]``); TP1/TP2 log ``Form A: rank N is an EXPERT
WORKER (routed experts + router only; no dense weights, no KV, no draft)``
and allocate ``KV Cache ... K size: 0.00 GB``, ``Mamba Cache ... ssm_state
size: 0.00GB``, ``0.00 GB pinned host memory for MHATokenToKVPoolHost``,
``0.00 GB host memory for hierarchical Mamba cache`` and a null storage tier
(rc9m D log lines 3720-3739, 4623-4626, 4943-4950). Their radix trees are
bookkeeping without bytes -- yet their mamba "anchors" voted in the RU/H97
usable-match MIN. Their host rows are released as transit, their anchors
die on eviction while TP0's survive in the arena, so the anchors sit at
different depths:

* rc9l (H96): TP0 19712, workers 16384 -> group 16384.
* rc9m (dkrnfbar1rc9m09260642, D log 53969-54036): TP0 18112 (host anchor
  there), workers 15552 -> group 15552; TP0 has no anchor at 15552 -> H96
  CAP-MISS, D dead. H97 (138d9df01c) turns that into ``RU FLOOR SKEW-ZERO``:
  every rank RE-PREFILLS 18112 tokens it could have resumed.

A worker's forward reads only the batch ROW COUNT (model_runner
``_forward_form_a_worker``: ``num_tokens = input_ids.shape[0]`` for the
MoE-input carrier), i.e. seq_len - prefix_len per request -- never a KV index
or a recurrent state of its own. So the worker must admit TP0's depth, and
its anchors must not vote.

Driven through the REAL pure helpers every rank runs in
``Scheduler._update_uniform_pool_budget`` (usable MIN arm, MAX arm, skew, the
realizability round) and the REAL ``MambaComponent.finalize_match_result`` /
``create_match_validator`` at admission, each rank with its REAL installed
Form A role plan. RED on 138d9df01c: rc9m/rc9i/rc9l re-prefill (or resume
shallower) instead of resuming at TP0's depth. GREEN with H98.
"""

from __future__ import annotations

import contextlib
import types
import unittest
from array import array

import torch

from sglang.srt.mem_cache.base_prefix_cache import MatchPrefixParams, MatchResult
from sglang.srt.mem_cache.radix_cache import RadixKey
from sglang.srt.mem_cache.unified_cache_components.mamba_component import (
    MambaComponent,
)
from sglang.srt.mem_cache.unified_cache_components.tree_component import (
    ComponentType,
)
from sglang.srt.managers import tp_match_floor as m
from sglang.srt import rank_role

#: Written by NAME so this file collects and runs on 138d9df01c too, where
#: nothing sets or reads them (that is the red).
FLOOR_ATTR = "_tp_match_floor_group"
FOLLOW_ATTR = "_tp_match_floor_follow_walk"
SWITCH = "SGLANG_WEG2_ENABLE_FORM_A_TP0_FOLLOW"

PAGE = 64
PROMPT = 20000
SLOTS = 8
ROLES = ("host", "worker", "worker")


def _floor(n):
    return n // PAGE * PAGE


class _Tree:
    """One rank's radix tree, reduced to what decides a match.

    ``kv``: how far the KV path (device + host-backed nodes) reaches.
    ``anchors``: the depths at which this rank's tree holds a mamba anchor
    (host-backed, usable). An ordinary walk ends on the deepest anchor within
    reach (the mamba validator); a walk under the follow attribute ends on the
    KV reach itself -- on a tombstone if no anchor sits there.
    """

    def __init__(self, kv, anchors):
        self.kv = kv
        self.anchors = sorted(anchors)
        self.root_node = types.SimpleNamespace(
            name="root",
            component_data=[None, None, types.SimpleNamespace(value=None, host_value=None)],
        )
        self.cache_controller = object()
        self.is_eagle = False
        self.is_chunk_cache = lambda: False
        self.supports_mamba = lambda: True
        self.swa_reprefill_tail_tokens = lambda: 0
        self.walks = 0

    def _node(self, n):
        if n <= 0:
            return self.root_node
        host = torch.tensor([3]) if n in self.anchors else None
        data = [None, None, types.SimpleNamespace(value=None, host_value=host)]
        return types.SimpleNamespace(name=f"n{n}", component_data=data)

    def result(self, n):
        node = self._node(n)
        return MatchResult(
            device_indices=torch.empty(0, dtype=torch.int64),
            last_device_node=self.root_node,
            last_host_node=node,
            best_match_node=node,
            host_hit_length=n,
        )

    def match_prefix(self, params):
        self.walks += 1
        reach = min(_floor(len(params.key)), self.kv)
        if getattr(self, FOLLOW_ATTR, False):
            n = reach
        else:
            n = max([a for a in self.anchors if a <= reach], default=0)
        res = self.result(n)
        if params.cow_mamba:
            return _component(self).finalize_match_result(
                result=res, params=params, value_chunks=[torch.zeros(1)], best_value_len=1
            )
        return res


def _component(tree):
    comp = types.SimpleNamespace(
        component_type=ComponentType.MAMBA,
        mamba_checkpoint_interval=None,
        mamba_ckpt_strict_resume=False,
        cache=tree,
        _stateless_resume_refusals=0,
        _foreign_pool_resume_refusals=0,
    )
    comp.finalize_match_result = types.MethodType(MambaComponent.finalize_match_result, comp)
    comp._raw_token_pos = lambda depth: depth
    comp.create_match_validator = types.MethodType(MambaComponent.create_match_validator, comp)
    return comp


def _req(rid, n=PROMPT):
    return types.SimpleNamespace(
        rid=rid,
        origin_input_ids=array("q", range(n)),
        output_ids=array("q"),
        extra_key=None,
        mamba_pool_idx=0,
        positional_embed_overrides=None,
        _compute_max_prefix_len=lambda k: max(k - 1, 0),
    )


@contextlib.contextmanager
def _as_rank(rank, roles=ROLES):
    """Install this rank's Form A role plan (None = classic boot)."""
    prev = (rank_role._INSTALLED_PLAN, rank_role._INSTALLED_RANK)
    plan = None if roles is None else rank_role.RankRolePlan(tuple(roles))
    rank_role.set_form_a_role_plan(plan, rank)
    try:
        yield
    finally:
        rank_role._INSTALLED_PLAN, rank_role._INSTALLED_RANK = prev


@contextlib.contextmanager
def _switch(value):
    from sglang.srt.environ import envs

    field = getattr(envs, SWITCH, None)
    if field is None:  # 138d9df01c: the switch does not exist (= off)
        yield
        return
    with field.override(value):
        yield


def _head_walk(tree, req):
    """What `_local_head_prefix_matches` measures: an ordinary cow=False walk
    on the full fill (no limit), capped at max_prefix_len."""
    toks = list(req.origin_input_ids) + list(req.output_ids)
    res = tree.match_prefix(
        MatchPrefixParams(key=RadixKey(array("q", toks), None), cow_mamba=False, req=None)
    )
    req.best_match_node = res.best_match_node
    n = len(res.device_indices) + int(res.host_hit_length)
    return min(n, req._compute_max_prefix_len(len(toks)))


def _plant(trees, rid, roles=ROLES):
    """The scheduler's reduce, rank by rank with each rank's own plan: usable
    MIN arm, MAX arm, and -- on skew -- H97's realizability MIN."""
    canonical = [rid]
    local = {}
    for r, t in trees.items():
        with _as_rank(r, roles):
            req = _req(rid)
            local[r] = m.local_usable_matches(t, {rid: req}, {rid: _head_walk(t, req)})
    red = lambda rows: [min(col) for col in zip(*rows)]  # noqa: E731
    rows_min, rows_max = [], []
    for r in trees:
        with _as_rank(r, roles):
            rows_min.append(m.build_usable_match_payload(canonical, local[r], SLOTS))
            rows_max.append(m.build_usable_max_payload(canonical, local[r], SLOTS))
    g = m.decode_group_usable(canonical, red(rows_min))
    skew = m.skewed_rids(g, m.decode_group_max(canonical, red(rows_max)))
    if skew:
        flags = []
        for r, t in trees.items():
            with _as_rank(r, roles):
                flags.append(
                    m.build_realize_payload(canonical, skew, local[r], t, {rid: _req(rid)}, SLOTS)
                )
        g = m.apply_realize_verdict(g, canonical, skew, red(flags))
    return g, local


def _admit(tree, rank, planted, rid, roles=ROLES):
    """`Req.init_next_round_input` on this rank: the admission key (limit =
    max_prefix_len), cow_mamba=True, the group value planted for the call."""
    req = _req(rid)
    toks = array("q", list(req.origin_input_ids))
    params = MatchPrefixParams(
        key=RadixKey(toks, None, limit=req._compute_max_prefix_len(len(toks))),
        cow_mamba=True,
        req=req,
    )
    setattr(tree, FLOOR_ATTR, planted)
    try:
        with _as_rank(rank, roles):
            out = tree.match_prefix(params)
    finally:
        setattr(tree, FLOOR_ATTR, None)
    return len(out.device_indices) + int(out.host_hit_length)


def _run(trees, rid, roles=ROLES):
    planted, _ = _plant(trees, rid, roles)
    return planted, {r: _admit(t, r, planted, rid, roles) for r, t in trees.items()}


def _rc9m():
    # TP0: host anchors at 2560 and 18112, none at 15552. TP1/TP2: the same
    # 18112-token KV path (prefetch completed_synced=18112 on all three, D log
    # 53963-53966), their byteless anchors at 2560 and 15552.
    return {
        0: _Tree(18112, [2560, 18112]),
        1: _Tree(18112, [2560, 15552]),
        2: _Tree(18112, [2560, 15552]),
    }


class TestRc9mWorkersFollowTp0(unittest.TestCase):
    def test_rc9m_resumes_at_tp0_depth_on_every_rank(self):
        with _switch(True):
            planted, geometry = _run(_rc9m(), "weg2-18-15")
        self.assertEqual(
            geometry,
            {0: 18112, 1: 18112, 2: 18112},
            f"rc9m: planted={planted} geometry={geometry} -- on 138d9df01c the "
            "workers' byteless anchors at 15552 drag the group to a depth TP0 "
            "cannot realize, H97 plants 0 and every rank RE-PREFILLS 18112 "
            "tokens instead of resuming (the H96 death before that)",
        )
        self.assertEqual(planted, {"weg2-18-15": 18112})

    def test_rc9l_resumes_at_tp0_depth(self):
        # rc9l (H96): TP0 host anchor 19712, workers 16384; KV path 19712.
        trees = {0: _Tree(19712, [16384, 19712]), 1: _Tree(19712, [16384]), 2: _Tree(19712, [16384])}
        with _switch(True):
            planted, geometry = _run(trees, "weg2-21-21")
        self.assertEqual(set(geometry.values()), {19712}, f"rc9l: {planted} {geometry}")

    def test_rc9i_tombstone_on_workers_follows(self):
        # rc9i: TP0 host anchor at 18112, TP1/TP2 a tombstone there (no anchor
        # on the path at all); KV reached 18112 on every rank.
        trees = {0: _Tree(18112, [18112]), 1: _Tree(18112, []), 2: _Tree(18112, [])}
        with _switch(True):
            planted, geometry = _run(trees, "weg2-32-26")
        self.assertEqual(set(geometry.values()), {18112}, f"rc9i: {planted} {geometry}")

    def test_follow_line_names_both_depths(self):
        with _switch(True), self.assertLogs(m.logger, level="WARNING") as cm:
            _run(_rc9m(), "weg2-18-15")
        lines = [l for l in cm.output if "RU FORM-A FOLLOW rid=" in l]
        self.assertEqual(len(lines), 2, cm.output)
        self.assertIn("tp0_depth=18112 worker_local=15552", lines[0])


class TestRiegelStays(unittest.TestCase):
    def test_worker_kv_short_falls_back_to_h97(self):
        # A worker whose KV path is really shorter (dead node): its reach is
        # the honest constraint -> skew -> H97: TP0 has no anchor at 15552 ->
        # the group re-prefills (uniform), never a split.
        trees = {0: _Tree(18112, [18112]), 1: _Tree(15552, [15552]), 2: _Tree(18112, [15552])}
        with _switch(True):
            planted, geometry = _run(trees, "r-short")
        self.assertEqual(set(geometry.values()), {0}, f"{planted} {geometry}")

    def test_worker_kv_short_tp0_realizes_it(self):
        trees = {0: _Tree(18112, [15552, 18112]), 1: _Tree(15552, []), 2: _Tree(18112, [])}
        with _switch(True):
            planted, geometry = _run(trees, "r-short2")
        self.assertEqual(set(geometry.values()), {15552}, f"{planted} {geometry}")

    def test_host_below_group_is_a_named_stop(self):
        tree = _Tree(18112, [2560])
        with _switch(True), self.assertRaises(m.FormAHostBelowGroup):
            _admit(tree, 0, {"r": 18112}, "r")

    def test_follow_miss_is_a_named_stop(self):
        tree = _Tree(15552, [2560])
        with _switch(True), self.assertRaises(m.FormAFollowMiss):
            _admit(tree, 1, {"r": 18112}, "r")

    def test_group_zero_zeroes_the_worker(self):
        with _switch(True):
            self.assertEqual(_admit(_Tree(18112, [15552]), 1, {"r": 0}, "r"), 0)


class TestSwitchOffIsH97(unittest.TestCase):
    """Off (or a classic boot without a role plan) = 138d9df01c's behaviour."""

    def test_switch_off(self):
        with _switch(False):
            planted, geometry = _run(_rc9m(), "weg2-18-15")
        self.assertEqual(planted, {"weg2-18-15": 0})
        self.assertEqual(set(geometry.values()), {0})

    def test_classic_boot(self):
        with _switch(True):
            planted, geometry = _run(_rc9m(), "weg2-18-15", roles=None)
        self.assertEqual(planted, {"weg2-18-15": 0})
        self.assertEqual(set(geometry.values()), {0})


class TestPieces(unittest.TestCase):
    def test_worker_abstains_in_max_arm(self):
        with _switch(True), _as_rank(1):
            p = m.build_usable_max_payload(["a"], {"a": 18112}, 4)
        self.assertTrue(all(v == m.MAX_ARM_NEUTRAL for v in p), p)
        with _switch(True), _as_rank(0):
            self.assertEqual(m.build_usable_max_payload(["a"], {"a": 18112}, 4)[0], -18112)

    def test_worker_votes_kv_reach(self):
        tree = _Tree(18112, [15552])
        req = _req("a")
        with _switch(True), _as_rank(1):
            self.assertEqual(m.local_usable_matches(tree, {"a": req}, {"a": 15552}), {"a": 18112})
        self.assertFalse(getattr(tree, FOLLOW_ATTR, False), "the follow walk must not leak")

    def test_host_votes_its_admission_not_the_head_walk(self):
        # A page-multiple prompt fully cached: the head walk (no limit) sees
        # the anchor at 18112 and votes min(18112, max_prefix_len)=18111; the
        # admission key (limit 18111) is cut to 18048, where the only anchor
        # within reach is 16384. The host must vote what it will ADMIT.
        tree = _Tree(18112, [16384, 18112])
        req = _req("a", n=18112)
        self.assertEqual(_head_walk(tree, req), 18111)
        with _switch(True), _as_rank(0):
            self.assertEqual(m.admission_probe(tree, req, follow=False), 16384)

    def test_real_validator_suspended_only_inside_follow_walk(self):
        tree = _Tree(0, [])
        comp = _component(tree)
        tomb = types.SimpleNamespace(
            component_data=[None, None, types.SimpleNamespace(value=None, host_value=None)]
        )
        self.assertFalse(comp.create_match_validator()(tomb, 64))
        with m.follow_walk(tree):
            self.assertTrue(comp.create_match_validator()(tomb, 64))
            self.assertTrue(comp.create_match_validator(match_device_only=True)(tomb, 64))
        self.assertFalse(comp.create_match_validator()(tomb, 64))

    def test_realize_round_on_worker_is_reach(self):
        with _switch(True), _as_rank(2):
            p = m.build_realize_payload(["a", "b"], {"a": 15552, "b": 16384}, {"a": 18112, "b": 15552}, None, {}, 4)
        self.assertEqual(p[:2], [1, 0])


if __name__ == "__main__":
    unittest.main()
