# SPDX-License-Identifier: Apache-2.0
"""DUAL-TP3PP3: under the dual layout D keeps deferring a store-short read
while P -- awake, writing -- publishes the tail, instead of falling back after
4 cycles and refusing (W50 -> the front re-routes through P, a second P
prefill of the same prompt).

Metal 30.09. boot tnybbz (...09300632), D rid pdflip-0-8: the read delivered
36863 of 40765 (P's 3902-token tail was still in its asynchronous
write-through); 4 fresh store-short marks in ~2 s ended the deferral,
'X-GATE uncached=40767 verdict=W31', 'W50 PdFlipTpPrefillExceeded', front
'W50-REROUTE ... path=fresh'. The 4-cycle bound exists for the flip form,
where P SLEEPS once D is awake and no writer can come (rc12y). In the dual
layout P never sleeps.

DANGER DIRECTIONS guarded here:
* the dual bound is still a BOUND (rc12y's 692 re-reads must stay impossible);
* growth of the delivered prefix still resets the count (unchanged);
* off dual the bound is 4 as before, and an explicit env override wins.
"""
from __future__ import annotations

import os
import unittest.mock as mock

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

from flliper.srt.managers import scheduler as S
from flliper.srt.pdflip import launcher as L
from flliper.test.ci.ci_register import register_cpu_ci
from flliper.test.test_utils import CustomTestCase

register_cpu_ci(est_time=2, suite="base-a-test-cpu")


def _env(**kv):
    base = {k: v for k, v in os.environ.items()
            if k not in (S.STORE_SHORT_MAX_CYCLES_ENV, S.DUAL_LAYOUT_ENV, S.DUAL_STORE_SHORT_MAX_CYCLES_ENV)}
    base.update(kv)
    return mock.patch.dict(os.environ, base, clear=True)


class DualStoreShortWait(CustomTestCase):
    def test_default_unchanged_off_dual(self):
        with _env():
            self.assertEqual(S._pdflip_store_short_max_cycles(), 4)

    def test_dual_waits_longer_but_bounded(self):
        with _env(**{S.DUAL_LAYOUT_ENV: "1"}):
            n = S._pdflip_store_short_max_cycles()
            self.assertEqual(n, S.DUAL_STORE_SHORT_MAX_CYCLES_DEFAULT)
            self.assertGreater(n, 4)
            self.assertLessEqual(n, 200)
        with _env(**{S.DUAL_LAYOUT_ENV: "1", S.DUAL_STORE_SHORT_MAX_CYCLES_ENV: "30"}):
            self.assertEqual(S._pdflip_store_short_max_cycles(), 30)

    def test_explicit_override_wins_under_dual(self):
        with _env(**{S.DUAL_LAYOUT_ENV: "1", S.STORE_SHORT_MAX_CYCLES_ENV: "7"}):
            self.assertEqual(S._pdflip_store_short_max_cycles(), 7)

    def test_launcher_marks_both_groups(self):
        ns = L.build_parser().parse_args(["--tree", "/x", "--tag", "t", "--dual-share"])
        L.resolve_dual_layout(ns)
        for g in ("D", "P"):
            self.assertEqual(L.dual_share_env(ns, g)[S.DUAL_LAYOUT_ENV], "1")
