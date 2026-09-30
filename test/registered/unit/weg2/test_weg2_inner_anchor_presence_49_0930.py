"""#49 L3 (30.09.): credit for P's INNER mamba anchors on a later text's shared path (dual16 finding P2).

Measured need (dual 0930 12:11, tools/front_route_gap): 11 of 18 LONG routes were repeated 40k/62k documents
whose token LCP ends BEFORE the earlier text's end anchor; P hit 36863 / 61439 / 94207 (its inner anchors),
the front credited 0 (presence_src=none): the PX anchor rule knows only the end anchor.

The chain: P (group P, SGLANG_WEG2_MAMBA_ANCHOR_INTERVAL) records the interval anchors a request donates
(mamba_component) -> stamped on its finishing output (req_time_stats.weg2_anchor_depths) -> meta_info /
sglext (OpenAI + Anthropic) -> the front reads them from P's leg 1 (anchor_depths_of) -> with the P-anchor
witness (SGLANG_WEG2_ENABLE_P_ANCHOR_PRESENCE, D resumed P's end anchor at its first content) and
SGLANG_WEG2_ENABLE_INNER_ANCHOR_PRESENCE the deepest (MAX_STATES_PER_PATH - 1) of them are kept beside the
entry; pending() credits the deepest one <= the token LCP.

DANGER DIRECTION = over-credit (W31/W50 reroute). Pinned:
  * only under --dual-layout: the flip form releases inner anchors at P's reset (INNER_ANCHOR_RELEASE);
  * only anchors below the end anchor, only the ones that survive P's per-path cap;
  * an anchor past the shared path credits nothing; a later D reading of the text retracts them.
"""

import collections
import os
import types
from unittest import mock

import numpy as np

from sglang.srt.weg2 import front as F
from sglang.srt.weg2.front_tokens import TokenSpans


def ids(n, base=0):
    return np.arange(base, base + n, dtype=np.int64)


# ---------------- P side: the anchors a request donates ----------------

def test_p_records_its_interval_anchors():
    from sglang.srt.mem_cache.unified_cache_components.mamba_component import MambaComponent

    class _C:
        enable_mamba_extra_buffer = False

        def _raw_token_pos(self, k):
            return k

    fake = _C()
    req = types.SimpleNamespace(rid="r", origin_input_ids=list(range(40767)), cache_protected_len=0)
    with mock.patch.dict(os.environ, {"SGLANG_WEG2_GROUP": "P", "SGLANG_WEG2_MAMBA_ANCHOR_INTERVAL": "4096"}):
        kept = []
        for pos in range(1024, 40767, 1024):                 # dynamic chunks of 1024
            declined = MambaComponent._weg2_anchor_step_declines(fake, req, pos)
            if not declined:
                kept.append(pos)
                req.cache_protected_len = pos                 # the tree-owned prefix moves to the anchor
    assert req._weg2_anchor_depths == [p for p in kept if p < 40766]
    assert req._weg2_anchor_depths[:3] == [4096, 8192, 12288]


def test_time_stats_carry_the_depths():
    from sglang.srt.observability.req_time_stats import SchedulerReqTimeStats as TS

    ts = TS.__new__(TS)
    ts.enable_metrics = False
    ts.weg2_prefill_s, ts.weg2_resumable_depth, ts.weg2_anchor_depths = 0.0, -1, (4096, 8192)
    assert ts.__getstate__()["weg2_anchor_depths"] == (4096, 8192)
    ts.weg2_anchor_depths = ()
    assert "weg2_anchor_depths" not in ts.__getstate__()


def test_anthropic_sglext_carries_them():
    from sglang.srt.entrypoints.anthropic import serving as S

    obj = types.SimpleNamespace(sglext=types.SimpleNamespace(weg2_resumable_depth=None,
                                                             cached_tokens_details=None,
                                                             weg2_anchor_depths=[4096, 8192]))
    assert S._sglext_of(obj).weg2_anchor_depths == [4096, 8192]


def test_front_reads_them_from_the_leg1_answer():
    assert F.anchor_depths_of({"sglext": {"weg2_anchor_depths": [8192, 4096, 4096]}}) == (4096, 8192)
    assert F.anchor_depths_of({"meta_info": {"weg2_anchor_depths": [36863]}}) == (36863,)
    assert F.anchor_depths_of({"sglext": {"weg2_resumable_depth": 5}}) == ()
    assert F.anchor_depths_of("x") == ()


# ---------------- front: the credit ----------------

def test_deepest_inner_anchor_on_the_shared_path():
    ts = TokenSpans(agent_span=True)
    src = ids(40767)
    ts.record_store_anchor(src, 40767, inner=[4096 * k for k in range(1, 10)], inner_keep=3)
    assert ts.inner[ts._key(src)] == (28672, 32768, 36864)            # the cap keeps the deepest 3
    repeat = np.concatenate([ids(37500), ids(3267, 10**7)])            # diverges at 37500 (needle moved)
    pend, credit, _k, _s = ts.pending(repeat)
    assert credit == 36864 and pend == 40767 - 36864                    # was 0 (the PX rule)
    early = np.concatenate([ids(20000), ids(20767, 10**7)])            # diverges before every kept anchor
    assert ts.pending(early)[1] == 0


def test_a_d_reading_retracts_the_inner_anchors():
    ts = TokenSpans(agent_span=True)
    src = ids(40767)
    ts.record_store_anchor(src, 40767, inner=[36864], inner_keep=3)
    ts.record_presence(src, 4096, prompt_tokens=40767, held_epoch=None, resumable_depth=4096)
    repeat = np.concatenate([ids(37500), ids(3267, 10**7)])
    assert ts.pending(repeat)[1] <= 4096


def _front(dual):
    f = F.Front.__new__(F.Front)
    f.x_exact, f.dual_layout, f.epoch = True, dual, 3
    f.tspans = TokenSpans(agent_span=True)
    f.counters = collections.Counter()
    f.ftok = types.SimpleNamespace(ids_for=lambda text: ids(40767))
    f._x_exact_reprice_queue = lambda why: 0
    return f


def test_front_credits_inner_anchors_only_under_the_dual_layout_and_switched_on():
    pending = types.SimpleNamespace(leg1_prompt_tokens=40767, leg1_anchor_depths=(4096, 32768, 36864))
    for dual, on, want in ((True, "1", True), (False, "1", False), (True, "0", False)):
        f = _front(dual)
        with mock.patch.dict(os.environ, {"SGLANG_WEG2_ENABLE_INNER_ANCHOR_PRESENCE": on}):
            assert F.Front._p_anchor_presence(f, "r", "t", pending) > 0
        assert bool(f.tspans.inner.get(f.tspans._key(ids(40767)))) is want, (dual, on)
        assert (f.counters["inner_anchor_presence"] == 1) is want
