# SPDX-License-Identifier: Apache-2.0
"""K15-FRONT (16.10.): the front credits only what the store or D proves.

(a) h111 weg2-114-865 (boot 1016_051456): 07:24:17 ``L3-INDEX-PRESENCE depth=88064`` -> LONG (7649
over X); 07:24:26 ``STORE-PRESENCE src=p_flush anchors=['weg2-114-862:95552']`` (P's END-ANCHOR,
unprobed) -> ``X-EXACT-REPRICE why=store_presence 7649 -> 161`` -> D-SHORT-DRAIN; D's store read ended
at 88192 (``deliverable=88192``) and D computed 7521. Now the P-flush presence is capped at the
front's store-probe depth.
(b) boot 1016_103428 ``W50-REROUTE x_refusal weg2-80-497 front_price=14 d_extent=220878``: D resumes
at ~0 while the front credits the whole prompt; the credit stays and routes the next turn SHORT
again. Now the entries above D's measured resume depth are dropped at the refusal.
RED on 9095d19351."""
from __future__ import annotations

import collections
import importlib.util
import os
import types

import numpy as np

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

from sglang.srt.weg2 import front as F  # noqa: E402
from sglang.srt.weg2.front_tokens import TokenSpans  # noqa: E402

_spec = importlib.util.spec_from_file_location(
    "_xcs_1002", os.path.join(os.path.dirname(__file__), "test_weg2_x_credit_store_1002.py"))
X = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(X)


class _Probe:
    def __init__(self, tokens):
        self.tokens = tokens
        self.asked = 0

    def depth(self, ids, **kw):
        self.asked += 1
        return types.SimpleNamespace(tokens=min(self.tokens, int(ids.size)))


def _ids(n, tail=0):
    a = np.arange(n, dtype=np.int32)
    if tail:
        a[-tail:] += 1_000_000            # the follow-up diverges after the shared prefix
    return a


def test_h111_p_flush_presence_is_capped_at_the_store_proof(monkeypatch):
    monkeypatch.delenv("SGLANG_WEG2_STORE_PRESENCE_PROBED", raising=False)
    f = X._front()
    f.store_probe = _Probe(88064)                     # what the store holds (07:24:17 presence)
    f.ftok.remember("t862", _ids(95556))
    f._p_phase_served = collections.OrderedDict({"weg2-114-862": ("t862", 95556)})
    assert f._p_flush_store_presence([]) == 1
    ids865 = np.concatenate([_ids(95552), np.arange(161, dtype=np.int32) + 2_000_000])
    pending, credit, _k, _src = f.tspans.pending(ids865)
    assert credit == 88064 and pending == 95713 - 88064, (credit, pending)
    assert f.counters["store_presence_capped"] == 1


def test_no_probe_no_credit(monkeypatch):
    monkeypatch.delenv("SGLANG_WEG2_STORE_PRESENCE_PROBED", raising=False)
    f = X._front()
    f.store_probe = None
    f.ftok.remember("t", _ids(95556))
    f._p_phase_served = collections.OrderedDict({"r": ("t", 95556)})
    assert f._p_flush_store_presence([]) == 0
    assert f.counters["store_presence_unprobed"] == 1


def test_switch_off_is_the_unprobed_credit(monkeypatch):
    monkeypatch.setenv("SGLANG_WEG2_STORE_PRESENCE_PROBED", "0")
    f = X._front()
    f.store_probe = _Probe(88064)
    f.ftok.remember("t862", _ids(95556))
    f._p_phase_served = collections.OrderedDict({"weg2-114-862": ("t862", 95556)})
    f._p_flush_store_presence([])
    _p, credit, _k, _s = f.tspans.pending(_ids(95556))
    assert credit == 95552


def test_x_refusal_retracts_the_credit_above_d_depth():
    """weg2-80-497: front price 14 of 220892 (credit 220878), D resumes at 14 (d_extent 220878)."""
    ts = TokenSpans(agent_span=True)
    ids = _ids(220892)
    ts.record_store_anchor(ids, 220892, source="store_anchor")
    _p, credit, _k, _s = ts.pending(ids)
    assert credit >= 220800
    gone = F.retract_refused_credit(ts, ids, d_depth=220892 - 220878)
    assert gone, "the refused credit stays"
    _p, credit2, _k, _s = ts.pending(ids)
    assert credit2 <= 220892 - 220878 + 64


def test_a_credit_d_can_deliver_is_kept():
    ts = TokenSpans(agent_span=True)
    ids = _ids(80000)
    ts.record_store_anchor(ids, 80000, source="store_anchor")
    assert F.retract_refused_credit(ts, ids, d_depth=79936) == []


def test_wiring_the_x_refusal_reroute_retracts_before_it_logs():
    import inspect

    src = inspect.getsource(F)
    i = src.index("X-REFUSAL-CREDIT-RETRACT rid=")
    j = src.index("reason=x_refusal path=fresh")
    k = src.rindex("retract_refused_credit(_ts", 0, i)
    assert k < i < j


# --- every D loss path: the eviction-time anchor loss (K15 WEG2-ANCHOR-LOST at=evict) reaches the front ---

def _k15():
    spec = importlib.util.spec_from_file_location(
        "_k15_1016", os.path.join(os.path.dirname(__file__), "test_weg2_k15_evict_anchor_save_1016.py"))
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def test_an_anchor_lost_at_eviction_is_kept_for_the_sleep_answer():
    from sglang.srt.mem_cache.unified_radix_cache import UnifiedRadixCache as U

    K = _k15()
    t, n = K._tree([0, 0], spill_ok=False), K._node(7, K.ANCHOR_TOMBSTONED)
    assert K._save(t, n) is False
    assert U.weg2_take_evict_lost(t) == [73984]
    assert U.weg2_take_evict_lost(t) == []


def test_the_flush_ledger_carries_the_eviction_losses(monkeypatch):
    from sglang.srt.managers import scheduler as S

    monkeypatch.setattr(S, "this_rank_is_form_a_worker", lambda: False)
    sched = types.SimpleNamespace(tree_cache=types.SimpleNamespace(
        weg2_unbacked_anchors=lambda: [], weg2_take_evict_lost=lambda: [73984, 73984]))
    S.Scheduler._weg2_note_lost_anchors(sched)
    assert S.Scheduler.weg2_take_anchors_lost(sched) == [73984]
    # and the front drops the presence credit standing on that anchor
    ts = TokenSpans(agent_span=True)
    ids = _ids(80000)
    ts.record_store_anchor(ids, 73984, source="store_anchor")
    _p, credit, _k, _s = ts.pending(ids)
    assert credit >= 73984
    F.retract_lost_anchors(ts, [73984])
    _p, credit2, _k, _s = ts.pending(ids)
    assert credit2 < 73984
