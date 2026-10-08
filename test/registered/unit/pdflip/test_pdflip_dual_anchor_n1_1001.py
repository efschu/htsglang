# SPDX-License-Identifier: Apache-2.0
"""DUAL ANCHOR N-1: after a P hand-back D holds N-1 tokens and computes exactly one.

Metal dual1m (...10011503, ef9b3a899c): N=41188, P trimmed to 41187 but published
41186 page keys (END-ANCHOR units=41186/41186, #1442 page_keys=41186), D read 41186
(EXTENT anchor_depth=41186) -> uncached=2 -> X=1 refused every hand-back (W50/W35/W53).
Root: bigram keys with the upstream keying -- P keyed its N-1 tokens with N-2 units
and D's N-1 raw claim is N-2 units.

DANGER DIRECTIONS guarded here (the real key arithmetic of the hand-back path):
* P's trimmed finish key (exact keying + the held-back prompt token as next token)
  has N-1 units = the N-1 KV rows P computed;
* D's claim on a dual D rank is N raw tokens = N-1 units, and the bigram key of
  P's node and D's claim are EQUAL -> D matches N-1, uncached = 1;
* the old arithmetic (upstream keying, N-1 raw claim) gives uncached = 2 (the metal
  bug), and outside the dual layout / on group P the claim stays N-1 raw;
* the id+tail helper keeps the container type (array('q') needs its typecode);
* wiring: the finish insert uses the trim tail, the dual env sets the exact keying.
"""
from __future__ import annotations

import inspect
import os
import types
from array import array

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

from flliper.srt.managers.schedule_batch import Req
from flliper.srt.mem_cache import unified_radix_cache as U
from flliper.srt.mem_cache.radix_cache import RadixKey
from flliper.srt.pdflip import dual_anchor_claim as C
from flliper.srt.pdflip import launcher as L
from flliper.test.ci.ci_register import register_cpu_ci
from flliper.test.test_utils import CustomTestCase

register_cpu_ci(est_time=2, suite="base-a-test-cpu")

N = 41188
IDS = array("q", range(1000, 1000 + N))
D_ENV = {"FLLIPER_PDFLIP_DUAL_LAYOUT": "1", "FLLIPER_PDFLIP_GROUP": "D"}


def _claim(env):
    fake = types.SimpleNamespace(return_logprob=False, logprob_start_len=-1)
    old = dict(os.environ)
    try:
        os.environ.update(env)
        for k in ("FLLIPER_PDFLIP_DUAL_LAYOUT", "FLLIPER_PDFLIP_GROUP"):
            if k not in env:
                os.environ.pop(k, None)
        return Req._compute_max_prefix_len(fake, N)
    finally:
        os.environ.clear()
        os.environ.update(old)


def _p_units(exact):
    origin, tail = IDS[: N - 1], [IDS[N - 1]]
    full = U._ids_with_tail(origin, tail)                      # the patched finish insert
    return len(U.bigram_anchor_key(full, N - 1, None, is_bigram=True, exact=exact, page_size=1))


def _d_units(raw_claim):
    return len(RadixKey(token_ids=IDS, extra_key=None, is_bigram=True, limit=raw_claim))


class DualAnchorN1(CustomTestCase):
    def setUp(self):
        self._flag = C.BIGRAM_EXACT_TREE[0]

    def tearDown(self):
        C.BIGRAM_EXACT_TREE[0] = self._flag

    def test_the_real_handback_arithmetic_leaves_one_token(self):
        C.BIGRAM_EXACT_TREE[0] = True
        p = _p_units(exact=True)
        claim = _claim(D_ENV)
        d = _d_units(claim)
        self.assertEqual(p, N - 1)
        self.assertEqual(claim, N)
        self.assertEqual(d, N - 1)
        # the same key: D's claim matches P's node unit for unit
        pk = U.bigram_anchor_key(U._ids_with_tail(IDS[: N - 1], [IDS[N - 1]]), N - 1, None,
                                 is_bigram=True, exact=True, page_size=1)
        dk = RadixKey(token_ids=IDS, extra_key=None, is_bigram=True, limit=claim)
        self.assertEqual(list(pk), list(dk))
        self.assertEqual(N - min(p, d), 1)                      # d_uncached

    def test_the_metal_bug_arithmetic(self):
        C.BIGRAM_EXACT_TREE[0] = False
        p = _p_units(exact=False)
        d = _d_units(_claim(D_ENV))
        self.assertEqual((p, d), (N - 2, N - 2))
        self.assertEqual(N - min(p, d), 2)                      # what X=1 refused

    def test_claim_unchanged_outside_dual_d(self):
        C.BIGRAM_EXACT_TREE[0] = True
        self.assertEqual(_claim({}), N - 1)
        self.assertEqual(_claim({"FLLIPER_PDFLIP_DUAL_LAYOUT": "1", "FLLIPER_PDFLIP_GROUP": "P"}), N - 1)
        C.BIGRAM_EXACT_TREE[0] = False
        self.assertEqual(_claim(D_ENV), N - 1)

    def test_logprob_cap_still_applies(self):
        C.BIGRAM_EXACT_TREE[0] = True
        fake = types.SimpleNamespace(return_logprob=True, logprob_start_len=5)
        old = dict(os.environ)
        try:
            os.environ.update(D_ENV)
            self.assertEqual(Req._compute_max_prefix_len(fake, N), 5)
        finally:
            os.environ.clear()
            os.environ.update(old)

    def test_ids_with_tail_keeps_type(self):
        a = U._ids_with_tail(array("q", [1, 2]), [3])
        self.assertIsInstance(a, array)
        self.assertEqual((a.typecode, list(a)), ("q", [1, 2, 3]))
        self.assertEqual(U._ids_with_tail([1, 2], (3,)), [1, 2, 3])

    def test_wiring(self):
        src = inspect.getsource(U.UnifiedRadixCache)
        self.assertIn("token_ids_full = _ids_with_tail(req.origin_input_ids[:kv_committed_len], _tail)", src)
        self.assertIn("_dac_note(bool(v))", src)
        self.assertIn('"FLLIPER_PDFLIP_BIGRAM_ANCHOR_EXACT": "1"', inspect.getsource(L.dual_share_env))
        # RELEASE-HEAD 1002: the claim goes through handback_claim (flip + dual)
        self.assertIn("if handback_bigram_claim():", inspect.getsource(Req._compute_max_prefix_len))
