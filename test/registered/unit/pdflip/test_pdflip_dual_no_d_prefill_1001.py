# SPDX-License-Identifier: Apache-2.0
"""DUAL-TP3PP3: in the dual layout EVERY prefill runs on P (user decision 01.10.).

Metal dual1m (...10011343): the front routed SHORT (uncached <= X=12288) straight to D --
39 "SHORT -> D", D admitted 2..3957 uncached tokens, a 2185-token extend stalled the decode
~2.5 s. The launcher kept the flip-design X (floored at --chunked-prefill-size) in the dual
layout.

DANGER DIRECTIONS guarded here:
* dual layout: D's X = 1 + --dual-d-prefill-tokens (default 0) -- the 1 is the N-1 anchor
  token; it is NOT floored to 4096 like the flip X;
* every uncached > X routes to P ("long"), including small remainders (2, 663, 3957);
* outside the dual layout nothing changes (None -> the flip X path);
* the launcher hands that X to D (its W31 riegel) AND the front, turns the front's live-X
  ceiling off and the flip-design short drain to 0;
* a negative allowance is refused.
"""
from __future__ import annotations

import inspect
import os
import types

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

from flliper.srt.pdflip import front as F
from flliper.srt.pdflip import launcher as L
from flliper.test.ci.ci_register import register_cpu_ci
from flliper.test.test_utils import CustomTestCase

register_cpu_ci(est_time=2, suite="base-a-test-cpu")


class DualNoDPrefill(CustomTestCase):
    def test_x_outside_dual_unchanged(self):
        self.assertIsNone(L.dual_x_tokens(types.SimpleNamespace(dual_layout=False, dual_d_prefill_tokens=0)))

    def test_x_in_dual(self):
        self.assertEqual(L.dual_x_tokens(types.SimpleNamespace(dual_layout=True, dual_d_prefill_tokens=0)), 1)
        self.assertEqual(L.dual_x_tokens(types.SimpleNamespace(dual_layout=True, dual_d_prefill_tokens=7)), 8)
        with self.assertRaises(SystemExit):
            L.dual_x_tokens(types.SimpleNamespace(dual_layout=True, dual_d_prefill_tokens=-1))

    def test_every_remainder_above_the_anchor_token_routes_to_p(self):
        x = L.dual_x_tokens(types.SimpleNamespace(dual_layout=True, dual_d_prefill_tokens=0))
        for uncached in (2, 25, 663, 2185, 3957, 12288, 90000):
            self.assertEqual(F.serviceable_route(uncached, uncached + 100, x, 648806), "long", uncached)
        self.assertEqual(F.serviceable_route(1, 40000, x, 648806), "short")
        self.assertEqual(F.serviceable_route(0, 40000, x, 648806), "short")

    def test_flag_default_zero(self):
        ns = L.build_parser().parse_args(["--tree", "/x", "--tag", "t"])
        self.assertEqual(ns.dual_d_prefill_tokens, 0)

    def test_launcher_applies_it_to_d_and_front(self):
        src = inspect.getsource(L)
        i = src.index("_dual_x = dual_x_tokens(ns)")
        block = src[i:i + 1500]
        self.assertIn("x_tokens = d_x_tokens = _dual_x", block)
        self.assertIn("front_x_ceiling = 0", block)
        self.assertIn("ns.d_short_drain_tokens = 0", block)
        # applied after the flip-X resolution, before the short-drain refusal reads it
        self.assertLess(i, src.index("refuse_short_drain_above_x(int(ns.d_short_drain_tokens or 0)"))
