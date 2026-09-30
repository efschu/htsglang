"""#49 L2 (30.09.): a text D served becomes a STORE presence after D's sleep leg published it -- to the
depth D witnessed (#59 weg2_resumable_depth), never the prompt. Switch SGLANG_WEG2_ENABLE_D_EPOCH_PUBLISH_PRESENCE
(front only, default off). Measured need: 7 of the 62 avoidable P routes of the real agent trace w109290020
were 'a D-direct serve teaches no span beyond its epoch' (tools/front_route_gap DDIRECT_GAP_served).

DANGER DIRECTION = OVER-CREDIT -> the turn routes SHORT, D's X gate finds less, W31/W50 reroute (parks
D's decodes, P prefills anyway). Pinned:
  * the credit is the #59 depth, never the prompt, never past the text; no #59 depth -> no credit;
  * only texts of an ENDED epoch (the held credit of the current epoch is unchanged);
  * a published depth past a new text's divergence credits nothing there (the shared path decides),
    and the entry's own lower measured anchor still credits it;
  * D contradicts it (a W31 refusal's small cached_tokens for that text) -> the credit is gone, the next
    pricing is LONG again: the over-credit is REFUSED, not repeated;
  * the front promotes only at a D->P flip done and only with the switch on.
"""

import inspect
import os
from unittest import mock

import numpy as np

from sglang.srt.weg2 import front as F
from sglang.srt.weg2.front_tokens import TokenSpans

X = 4096


def ids(n, base=0):
    return np.arange(base, base + n, dtype=np.int64)


def spans_with_d_serve(prompt=22653, arrival_ct=21265, depth=22592, epoch=14):
    ts = TokenSpans(agent_span=True)
    ts.record_presence(ids(prompt), arrival_ct, prompt_tokens=prompt, held_epoch=epoch, resumable_depth=depth)
    return ts


def test_credit_is_the_witnessed_depth_never_the_prompt():
    ts = spans_with_d_serve()
    nxt = np.concatenate([ids(22653), ids(3535, 10**7)])            # the next turn: +3535 tokens
    pend0, credit0, _k, _s = ts.pending(nxt, epoch=None)             # after the epoch: arrival reading only
    assert credit0 == 21265 and pend0 > X                            # today: LONG (weg2-14-17)
    n, gained = ts.promote_published(15)                             # D slept at epoch 15
    assert (n, gained) == (1, 22592 - 21265)
    pend1, credit1, _k, _s = ts.pending(nxt, epoch=None)
    assert credit1 == 22592 and pend1 == 26188 - 22592 <= X          # the #59 depth, not 22653
    # promoting again changes nothing
    assert ts.promote_published(16) == (0, 0)


def test_no_witnessed_depth_no_credit_and_current_epoch_untouched():
    ts = TokenSpans(agent_span=True)
    ts.record_presence(ids(22653), 21265, prompt_tokens=22653, held_epoch=14, resumable_depth=None)
    assert ts.promote_published(15) == (0, 0)                        # no #59 field: nothing witnessed
    ts2 = spans_with_d_serve(epoch=15)
    assert ts2.promote_published(15) == (0, 0)                       # the epoch D still serves in
    # the depth never passes the prompt / the text
    ts3 = spans_with_d_serve(prompt=1000, arrival_ct=500, depth=5000)
    ts3.promote_published(15)
    _p, credit, _k, _s = ts3.pending(np.concatenate([ids(1000), ids(10, 10**7)]))
    assert credit <= 1000


def test_published_depth_past_the_divergence_keeps_the_lower_anchor():
    ts = spans_with_d_serve(prompt=62114, arrival_ct=60482, depth=62080)
    ts.promote_published(140)
    diverges = np.concatenate([ids(61000), ids(2000, 10**7)])        # leaves the text at 61000
    _p, credit, _k, _s = ts.pending(diverges)
    assert credit == 60482                                           # the measured anchor on the path
    follows = np.concatenate([ids(62114), ids(500, 10**7)])
    _p, credit, _k, _s = ts.pending(follows)
    assert credit == 62080


def test_danger_direction_d_refusal_retracts_the_over_credit():
    """An over-credit (the store did not hold the published depth): the turn routed SHORT, D's X gate
    refused it (W31) and answered its small reading -> the entry is replaced, the next turn prices LONG."""
    ts = spans_with_d_serve()
    ts.promote_published(15)
    nxt = np.concatenate([ids(22653), ids(3535, 10**7)])
    assert ts.pending(nxt)[0] <= X                                   # credited -> SHORT
    # D's leg 2 for `nxt` refused: its reading is what D really found (e.g. only 16384 of it)
    ts.record_presence(nxt, 16384, prompt_tokens=26188, held_epoch=None, resumable_depth=16384)
    # the source text's own published depth is superseded only by a reading OF THAT TEXT; D's refusal
    # of the follow-up records the follow-up's small reading, and a THIRD turn on the same prefix
    third = np.concatenate([ids(22653), ids(3535, 10**7), ids(100, 2 * 10**7)])
    # still credited by the source's published depth unless D contradicts the SOURCE text
    ts.record_presence(ids(22653), 16384, prompt_tokens=22653, held_epoch=None, resumable_depth=16384)
    pend, credit, _k, _s = ts.pending(third)
    assert credit == 16384 and pend > X                              # refused: back to LONG


def test_front_promotes_only_at_d_to_p_and_only_switched_on():
    src = inspect.getsource(F.Front)
    i = src.index('self._dp_report(t_flip0, _dp_drain_end)')
    assert 'self._d_epoch_publish_presence(int(rec["epoch"]))' in src[i:i + 200]
    assert 'if src == "D" and dst == "P":' in src[i - 200:i]
    f = F.Front.__new__(F.Front)
    f.x_exact = True
    f.tspans = spans_with_d_serve()
    import collections

    f.counters = collections.Counter()
    f._x_exact_reprice_queue = lambda why: 0
    with mock.patch.dict(os.environ, {"SGLANG_WEG2_ENABLE_D_EPOCH_PUBLISH_PRESENCE": "0"}):
        assert f._d_epoch_publish_presence(15) == 0
    with mock.patch.dict(os.environ, {"SGLANG_WEG2_ENABLE_D_EPOCH_PUBLISH_PRESENCE": "1"}):
        assert f._d_epoch_publish_presence(15) == 1
    assert f.counters["d_epoch_publish_presence"] == 1
    assert f.counters["d_epoch_publish_presence_tokens"] == 22592 - 21265
