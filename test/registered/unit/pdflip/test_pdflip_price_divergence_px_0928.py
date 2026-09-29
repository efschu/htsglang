# SPDX-License-Identifier: Apache-2.0
"""PX (28.09.): the front never credits past the divergence of a text from the
entry that witnesses it -- a hybrid D resumes only at a Mamba anchor.

NF rc12z30e (ca2a9706ec), front log ...09282117_ca2a9706ec_0928_211748:
  pdflip-14-37 (16869 tokens): 'X-EXACT-PRICE ... pending=779 tokens=16869
  credit=16090 src=d_leg2_cached' -> SHORT; D: '[#928 anchor] REFUSING resume
  ... match_tokens=3840 best_value_len=0', 'X-GATE uncached=16869' -> W50
  x_refusal_midstream, PARK-IMMEDIATE of 4 running decodes (rpc 4.42 s), flip,
  P prefill of all 16869. The credit 16090 is not page-aligned: it is the LCP
  with the witness entry, i.e. the divergence point, where D holds no state.
  Same shape: 10-32 27250, 16-40 29285, 16-41 17017, 44-79 33484.
"""

from __future__ import annotations

import os

import numpy as np

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

from flliper.srt.pdflip.front_tokens import TokenSpans  # noqa: E402

X = 4096


def _ids(n: int, common: int = None, salt: int = 7) -> np.ndarray:
    a = np.arange(n, dtype=np.int32)
    if common is not None and common < n:
        a[common:] = 10 ** 6 + salt * 10 ** 5 + np.arange(n - common, dtype=np.int32)
    return a


def test_pdflip_14_37_credit_is_not_the_divergence_point():
    ts = TokenSpans(agent_span=True)
    # the witness: a text D served, measured cached 20160 (an anchor), of
    # which 14-37 shares only the first 16090 tokens
    ts.record_presence(_ids(20171, salt=1), 20160, prompt_tokens=20171,
                       held_epoch=9)
    cur = _ids(16869, common=16090, salt=2)
    pending, credit, known, _src = ts.pending(cur, epoch=13)
    assert known
    assert credit == 0, "D holds no anchor at or below the divergence 16090"
    assert pending == 16869 > X, "LONG via P, not SHORT into a W50 reroute"


def test_resume_point_below_the_divergence_still_credits():
    ts = TokenSpans(agent_span=True)
    # held in this epoch: D computed 20171 itself, it resumed at 16384 (ct)
    ts.record_presence(_ids(20171, salt=1), 16384, prompt_tokens=20171,
                       held_epoch=13)
    cur = _ids(20200, common=20170, salt=2)  # leaves the entry 1 token early
    pending, credit, _k, _s = ts.pending(cur, epoch=13)
    assert credit == 16384, "the resume point is on the shared path"
    assert pending == 20200 - 16384


def test_anchor_within_the_shared_prefix_is_unchanged():
    ts = TokenSpans(agent_span=True)
    ts.record_presence(_ids(57396), 55424, prompt_tokens=57396, held_epoch=8,
                       resumable_depth=37952)
    pending, credit, _k, _s = ts.pending(_ids(57792, common=57395), epoch=12)
    assert (credit, pending) == (37952, 57792 - 37952)  # #59 metal, unchanged


def test_follow_up_that_contains_the_entry_is_unchanged():
    ts = TokenSpans(agent_span=True)
    ts.record_presence(_ids(30000), 29952, prompt_tokens=30000, held_epoch=4)
    pending, credit, _k, _s = ts.pending(_ids(31000), epoch=4)
    assert credit == 30000 and pending == 1000  # #49 held credit, lcp = 30000


def test_resume_point_past_d_depth_is_no_credit():
    ts = TokenSpans(agent_span=True)
    ts.record_presence(_ids(20000, salt=1), 18000, prompt_tokens=20000,
                       held_epoch=5, resumable_depth=17000)
    # lcp 19000 < raw 17000? no: raw = min(max(18000, 20000), 17000) = 17000
    pending, credit, _k, _s = ts.pending(_ids(19500, common=19000, salt=2), epoch=5)
    assert credit == 17000  # D's deepest anchor lies on the shared path
    # leaves before D's anchor: ct 18000 is past D's depth, nothing is left
    pending, credit, _k, _s = ts.pending(_ids(16900, common=16800, salt=3), epoch=5)
    assert credit == 0 and pending == 16900


def test_known_prefix_depth_past_the_divergence_is_zero():
    ts = TokenSpans(agent_span=True)
    ts.record_presence(_ids(20171, salt=1), 20160, prompt_tokens=20171,
                       held_epoch=9, resumable_depth=20160)
    assert ts.known_prefix_depth(_ids(16869, common=16090, salt=2)) == 0
    assert ts.known_prefix_depth(_ids(22000, common=20165, salt=2)) == 20160
