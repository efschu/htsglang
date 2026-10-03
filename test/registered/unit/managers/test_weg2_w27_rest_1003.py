"""W27-UNIFORM rest (item 140, the open points of item 025, fdb4a69573).

(a) H91 STORE-TOLD KEPT: a verdict kept across visits is admitted at a later
    visit without asking the follower's live tree again; the tree can move in
    between (the sibling's read released, eviction) and only the TOLD-PIN held.
    The later admission runs ``follower_reach_told`` as well.
(b) The PF-ack path (off on NF today) acked through ``_resumable_own`` ->
    ``pp0_admissible``, which does not know the #988 load-back's state-aligned
    extent: it acked told where this rank's admission stops at the anchor.
    It asks ``rank_resumable`` now.
(c) The re-read wait is bounded: ``WAIT_CAP_S`` is the whole stall budget of a
    rid across its re-reads (a kept verdict asks again every pass), and the
    wait is on the RE-READ log line.

Helpers are the stage doubles of test_weg2_pp_width_uniform_1003.
"""
from __future__ import annotations

import os
import time

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

import pytest  # noqa: E402

from sglang.srt.managers import weg2_store_told as m  # noqa: E402
from sglang.srt.managers import weg2_told_fallback as fb  # noqa: E402
from sglang.srt.weg2 import p_intake  # noqa: E402
from sglang.srt.weg2 import p_twin_defer as twin  # noqa: E402

from test_weg2_pp_width_uniform_1003 import (  # noqa: E402
    HEAD, RID, TOLD, _StageTree, _req, _stage,
)


@pytest.fixture(autouse=True)
def _env(monkeypatch):
    monkeypatch.delenv("SGLANG_WEG2_FOLLOWER_REACH_TOLD", raising=False)
    monkeypatch.setenv(m.ENV_FOLLOWER_EARLY_READ, "1")
    monkeypatch.setattr(m, "_absolute_armed", lambda: True)
    monkeypatch.delenv("SGLANG_WEG2_DUAL_SHARE", raising=False)


def _gate(stage, req):
    return p_intake.told_admission(stage, req, lambda *a: None, m.admission)


# -- (a) ----------------------------------------------------------------------

def _kept_follower(rank=1, twin_told=True):
    """Visit 1 admits at told (the tree reaches it) and the adder does not seat
    the request: the verdict is kept."""
    tree = _StageTree(dev=HEAD, host=TOLD - HEAD, anchor=TOLD, record=TOLD)
    stage = _stage(rank, tree)
    req = _req(head=0)
    stage._weg2_store_told[RID] = TOLD
    if twin_told:
        twin.note_follower_twin(stage, RID)
    assert _gate(stage, req) == TOLD
    assert stage.calls == []
    assert RID in getattr(stage, p_intake.KEPT_ATTR)
    return stage, req, tree


def _evict_host_span(tree):
    # between the visits: the span this rank held beyond the device head is gone
    tree.host, tree.anchor, tree.record = 0, None, 0


def test_kept_verdict_is_admitted_only_when_the_live_tree_still_reaches_told():
    stage, req, tree = _kept_follower()
    _evict_host_span(tree)
    prefix_before = tree.admitted_prefix(m.prefix_cap_tokens(tree, TOLD))
    assert prefix_before == HEAD
    _gate(stage, req)                       # visit 2: H91 STORE-TOLD KEPT
    assert stage.calls == [(RID, TOLD)], "the told-limited re-read of [13184, 16640)"
    assert tree.admitted_prefix(m.prefix_cap_tokens(tree, TOLD)) == TOLD


def test_kept_verdict_whose_tree_is_intact_reads_nothing(caplog):
    stage, req, tree = _kept_follower()
    assert _gate(stage, req) == TOLD
    assert stage.calls == []


def test_kept_verdict_of_a_non_twin_follower_is_not_probed():
    stage, req, tree = _kept_follower(twin_told=False)
    _evict_host_span(tree)
    _gate(stage, req)
    assert stage.calls == [], "only the absolute twin told is compared with a live tree"


def test_kept_verdict_on_pp0_is_not_probed():
    stage, req, tree = _kept_follower(rank=0)
    _evict_host_span(tree)
    _gate(stage, req)
    assert stage.calls == []


def test_kept_reread_credit_replaces_the_stale_one():
    stage, req, tree = _kept_follower()
    _evict_host_span(tree)
    assert _gate(stage, req) == TOLD - HEAD, "the re-read's own loaded count"


# -- (b) ----------------------------------------------------------------------

def test_pf_ack_names_the_state_aligned_extent():
    """Host KV to told whose recurrent state sits at the twin anchor: the
    admission loads back to HEAD only -- the ack says HEAD, PP0 answers told=0
    for every rank (no START-SPLIT)."""
    tree = _StageTree(dev=HEAD, host=TOLD - HEAD, anchor=HEAD)
    s = _stage(1, tree)
    assert fb._resumable_own(s, _req(), RID, TOLD) == HEAD


def test_pf_ack_of_a_tree_that_reaches_told_is_unchanged():
    tree = _StageTree(dev=HEAD, host=TOLD - HEAD, anchor=TOLD)
    s = _stage(1, tree)
    assert fb._resumable_own(s, _req(), RID, TOLD) == TOLD


# -- (c) ----------------------------------------------------------------------

class _NeverReadyTree(_StageTree):
    def check_prefetch_progress(self, rid):
        return False


def test_rereads_wait_is_one_budget_per_rid_not_one_cap_per_visit(monkeypatch, caplog):
    monkeypatch.setattr(m, "WAIT_CAP_S", 0.2)
    stage = _stage(1, _NeverReadyTree(dev=HEAD, host=0, record=TOLD), store_has=HEAD + 64)
    req = _req(head=0)
    t0 = time.monotonic()
    with caplog.at_level("WARNING", logger=m.logger.name):
        for _ in range(5):          # five passes of a kept verdict
            assert m.follower_reach_told(stage, req, TOLD, TOLD) == TOLD
    took = time.monotonic() - t0
    assert took < 0.2 + 0.15, f"five visits stalled the scheduler {took:.2f}s, cap is 0.2s in total"
    assert len(stage.calls) == 1, "no second read once the budget is spent"
    assert any("STILL SHORT" in x and "wait=" in x and "CAPPED" in x for x in caplog.messages)
    assert any("RE-READ BUDGET SPENT" in x for x in caplog.messages)


def test_reread_log_line_carries_the_wait(caplog):
    stage = _stage(1, _StageTree(dev=HEAD, host=0, record=TOLD))
    with caplog.at_level("WARNING", logger=m.logger.name):
        m.follower_reach_told(stage, _req(head=0), TOLD, TOLD)
    line = [x for x in caplog.messages if "FOLLOWER RE-READ" in x]
    assert len(line) == 1 and " wait=" in line[0] and " ms " in line[0], line


def test_a_finished_reread_gives_the_budget_back(monkeypatch):
    stage = _stage(1, _StageTree(dev=HEAD, host=0, record=TOLD))
    m.follower_reach_told(stage, _req(head=0), TOLD, TOLD)
    assert RID not in stage._w27u_wait_spent_s
