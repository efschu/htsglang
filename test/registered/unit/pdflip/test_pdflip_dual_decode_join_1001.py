# SPDX-License-Identifier: Apache-2.0
"""DUAL-TP3PP3 stage 2 step 1: the decode-join riegel and bookkeeping (pdflip/dual_decode_join.py).

DANGER DIRECTIONS guarded here:
* switch off (default) or not the dual layout -> 'extend' (the old path, unchanged);
* only a fresh P hand-off with EXACTLY the N-1 anchor token outstanding joins; any
  other remainder at D is REFUSED by name (never a silent D prefill);
* DFlash only; multimodal and input-logprob requests are refused (a decode round cannot
  produce them); output-token logprobs are fine;
* the pending token is the prompt's last token and is NOT an output: no output token is
  appended, the invariant committed KV == seqlen - 1 holds with an empty output;
* cached_tokens grows by exactly the prefix D did not compute (no double count).
"""
from __future__ import annotations

import types

from flliper.srt.pdflip import dual_decode_join as J
from flliper.test.ci.ci_register import register_cpu_ci
from flliper.test.test_utils import CustomTestCase

register_cpu_ci(est_time=1, suite="base-a-test-cpu")


def _req(n=10, out=0, **kw):
    r = types.SimpleNamespace(origin_input_ids=list(range(100, 100 + n)), output_ids=[7] * out,
                              multimodal_inputs=None, return_logprob=False, logprob_start_len=-1,
                              cached_tokens=0, cached_tokens_device=0, already_computed=0, rid="r")
    for k, v in kw.items():
        setattr(r, k, v)
    return r


def _v(req, prefix, **kw):
    kw.setdefault("dual_layout", True)
    kw.setdefault("spec_is_dflash", True)
    kw.setdefault("enabled", True)
    return J.join_verdict(req, prefix_len=prefix, **kw)


class DualDecodeJoin(CustomTestCase):
    def test_switch_default_off(self):
        self.assertFalse(J.join_enabled({}))
        self.assertTrue(J.join_enabled({J.JOIN_ENV: "1"}))
        self.assertEqual(_v(_req(), 9, enabled=False).verdict, J.EXTEND)

    def test_not_dual_is_the_old_path(self):
        self.assertEqual(_v(_req(), 9, dual_layout=False).verdict, J.EXTEND)

    def test_exactly_the_anchor_token_joins(self):
        v = _v(_req(10), 9)
        self.assertTrue(v)
        self.assertEqual(v.verdict, J.JOIN)

    def test_any_other_remainder_is_refused_by_name(self):
        for prefix in (0, 5, 8, 10):
            v = _v(_req(10), prefix)
            self.assertEqual(v.verdict, J.REFUSE, prefix)
            self.assertIn("uncached=", v.reason)
        self.assertFalse(_v(_req(10), 8))

    def test_refusals(self):
        self.assertEqual(_v(_req(), 9, spec_is_dflash=False).verdict, J.REFUSE)
        self.assertEqual(_v(_req(multimodal_inputs=object()), 9).verdict, J.REFUSE)
        self.assertEqual(_v(_req(return_logprob=True, logprob_start_len=0), 9).verdict, J.REFUSE)
        self.assertEqual(_v(_req(out=1), 9).verdict, J.REFUSE)
        # output-token logprobs only: fine
        self.assertEqual(_v(_req(return_logprob=True, logprob_start_len=-1), 9).verdict, J.JOIN)
        self.assertEqual(_v(_req(return_logprob=True, logprob_start_len=10), 9).verdict, J.JOIN)

    def test_tiny_prompt_takes_the_old_path(self):
        self.assertEqual(_v(_req(1), 0).verdict, J.EXTEND)

    def test_state_and_accounting(self):
        r = _req(10)
        s = J.join_state(r, 9)
        self.assertEqual((s.committed_kv, s.pending_token, s.seqlen), (9, 109, 10))
        self.assertEqual(s.committed_kv, s.seqlen - 1)   # the decode invariant, empty output
        J.apply_join_accounting(r, s)
        self.assertEqual(r.output_ids, [])                # the prompt token is not an output
        self.assertEqual((r.cached_tokens, r.cached_tokens_device, r.already_computed), (9, 9, 9))
        self.assertTrue(r._pdflip_decode_joined)

    def test_no_double_count_of_cached(self):
        r = _req(10, already_computed=4, cached_tokens=4, cached_tokens_device=4)
        J.apply_join_accounting(r, J.join_state(r, 9))
        self.assertEqual(r.cached_tokens, 9)

    def test_state_refuses_a_non_joinable_request(self):
        with self.assertRaises(ValueError):
            J.join_state(_req(10), 8)
        with self.assertRaises(ValueError):
            J.join_state(_req(10, out=1), 9)
