"""#59b (D side): the park names the depth each parked request resumes from.

#59 stamps ``weg2_resumable_depth`` on a leg 2's FINISHING output. A flip park
(``POST /weg2/park_running``, also the front's WAIT-BOUND park) retracts the
running requests with their spans retained -- their streams stay open and
finish only after the next wake. The front prices a parked request's presence
before that, so the park answer carries the same group-uniform depth per
parked rid: ``{"weg2_resumable_depth": {rid: depth}}``; absent when there is
nothing to name (not group D, no parked request, DP attention without Form A),
and the front then keeps its old price.

RED on the base: no ``park_depths``, no output field, no wiring.
"""

from __future__ import annotations

import contextlib
import inspect
import types
import unittest
from array import array

import torch

from sglang.srt import rank_role
from sglang.srt.mem_cache.base_prefix_cache import MatchResult

ROLES = ("host", "worker", "worker")
TRACK = 47104  # NF weg2-1-13: the park retained 47104 of 47262 (track grid)


def _mod():
    from sglang.srt.managers import weg2_resumable_depth as m

    return m


@contextlib.contextmanager
def _as_rank(rank, roles=ROLES):
    prev = (rank_role._INSTALLED_PLAN, rank_role._INSTALLED_RANK)
    plan = None if roles is None else rank_role.RankRolePlan(tuple(roles))
    rank_role.set_form_a_role_plan(plan, rank)
    try:
        yield
    finally:
        rank_role._INSTALLED_PLAN, rank_role._INSTALLED_RANK = prev


def _node(name):
    return types.SimpleNamespace(
        name=name,
        component_data=[None, None, types.SimpleNamespace(value=None, host_value=torch.tensor([3]))],
    )


class _ParkedTree:
    """After the park's retraction: the key and the anchor at the track point."""

    def __init__(self, depth=TRACK):
        self.depth = depth
        self.root_node = _node("root")
        self.root_node.component_data[2].host_value = None
        self.cache_controller = None
        self.is_eagle = False
        self.supports_mamba = lambda: True
        self.swa_reprefill_tail_tokens = lambda: 0

    def match_prefix(self, params):
        d = min(self.depth, len(params.key))
        node = _node(f"n{d}")
        return MatchResult(
            device_indices=torch.arange(d, dtype=torch.int64),
            last_device_node=node,
            last_host_node=node,
            best_match_node=node,
        )


def _req(rid, n=47000, out=262):
    return types.SimpleNamespace(
        rid=rid,
        origin_input_ids=array("q", range(n)),
        output_ids=list(range(n, n + out)),
        extra_key=None,
        positional_embed_overrides=None,
    )


def _ps(tp=3, dp=3):
    return types.SimpleNamespace(tp_size=tp, attn_dp_size=dp, pp_size=1, attn_tp_rank=0)


class TestParkDepths(unittest.TestCase):
    def test_form_a_host_names_each_parked_rid(self):
        m = _mod()
        with _as_rank(0):
            got = m.park_depths(_ParkedTree(), [_req("weg2-1-13"), _req("weg2-6-29")], _ps())
        self.assertEqual(got, {"weg2-1-13": TRACK, "weg2-6-29": TRACK})

    def test_classic_group_takes_the_min(self):
        m = _mod()
        with _as_rank(0, roles=None):
            got = m.park_depths(
                _ParkedTree(), [_req("a")], _ps(tp=2, dp=1),
                reduce_min=lambda v: [min(x, 45568) for x in v],
            )
        self.assertEqual(got, {"a": 45568})

    def test_nothing_to_name_is_empty(self):
        m = _mod()
        with _as_rank(1):  # a Form A worker
            self.assertEqual(m.park_depths(_ParkedTree(), [_req("a")], _ps()), {})
        with _as_rank(0, roles=None):
            self.assertEqual(m.park_depths(_ParkedTree(), [_req("a")], _ps(tp=2, dp=2)), {})
            self.assertEqual(m.park_depths(_ParkedTree(), [], _ps(tp=2, dp=1),
                                           reduce_min=lambda v: 1 / 0), {})


class TestParkCarrier(unittest.TestCase):
    def test_output_field_defaults_empty(self):
        from sglang.srt.managers.io_struct import Weg2ParkRunningReqOutput

        out = Weg2ParkRunningReqOutput(success=True)
        self.assertEqual(out.weg2_resumable_depth, {})
        out = Weg2ParkRunningReqOutput(success=True, weg2_resumable_depth={"a": 0})
        self.assertEqual(out.weg2_resumable_depth, {"a": 0})

    def test_park_running_fills_it(self):
        from sglang.srt.weg2 import d_park_runtime

        src = inspect.getsource(d_park_runtime.park_running)
        self.assertIn("weg2_resumable_depth.park_depths", src)
        self.assertIn("weg2_resumable_depth=", src)

    def test_http_answer_names_it_only_when_set(self):
        from sglang.srt.entrypoints import http_server

        src = inspect.getsource(http_server.weg2_park_running)
        self.assertIn('"weg2_resumable_depth"', src)


if __name__ == "__main__":
    unittest.main()
