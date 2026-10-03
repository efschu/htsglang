"""W27-UNIFORM kept-verdict budget and short-tree divergence (item 180, the
findings of the y8k load review on item 140, 4daab4fcbe).

(1) LEAK. ``told_admission``'s replace path (a new request object for a kept
    rid) popped ``kept[rid]`` but not the rid's re-read wait budget
    ``_w27u_wait_spent_s``. When no fresh told follows, ``settle_told`` never
    sees the rid again (the kept entry is gone): the spent entry is orphaned,
    and a reused rid inherits a shrunk budget.
(2) DIVERGENCE. ``follower_reach_told`` returned its record unchanged when the
    live tree stayed short (budget spent / STILL SHORT) and the callers
    (kept visit, admission, satisfied) went on: the follower seated at told
    over a shorter tree while PP0 seated at told -- the PP0-vs-follower width
    split W27 exists to prevent. Now the rank does not seat on its own tree:
    it holds the request (skipped like a told that has not arrived), re-asks
    PP0 through the PF ack stream when the boot has one (PP0 alone answers
    told=0 for every rank), and admits at told when its tree reaches it.

Helpers are the stage doubles of test_weg2_pp_width_uniform_1003.
"""
from __future__ import annotations

import os

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


def _gate(stage, req, skips=None):
    note = (lambda *a: skips.append(a)) if skips is not None else (lambda *a: None)
    return p_intake.told_admission(stage, req, note, m.admission)


def _kept_follower(rank=1):
    tree = _StageTree(dev=HEAD, host=TOLD - HEAD, anchor=TOLD, record=TOLD)
    stage = _stage(rank, tree, store_has=HEAD)     # a re-read finds nothing
    req = _req(head=0)
    stage._weg2_store_told[RID] = TOLD
    twin.note_follower_twin(stage, RID)
    assert _gate(stage, req) == TOLD
    assert RID in getattr(stage, p_intake.KEPT_ATTR)
    return stage, req, tree


def _evict_host_span(tree):
    tree.host, tree.anchor, tree.record = 0, None, 0


def _restore_host_span(tree):
    tree.host, tree.anchor, tree.record = TOLD - HEAD, TOLD, TOLD - HEAD


# -- (1) leak -----------------------------------------------------------------

def test_replaced_kept_verdict_drops_the_rids_wait_budget():
    stage, req, tree = _kept_follower()
    stage._w27u_wait_spent_s = {RID: 12.0}
    new_req = _req(head=0)                  # re-intake: another object, no fresh told
    skips = []
    assert _gate(stage, new_req, skips) is None
    assert skips and skips[0][0] == m.SKIP_TOLD_PENDING
    assert RID not in stage._w27u_wait_spent_s, "orphaned: settle_told never sees the rid again"
    assert RID not in getattr(stage, p_intake.KEPT_ATTR)


def test_a_reused_rid_starts_with_the_full_budget():
    stage, req, tree = _kept_follower()
    stage._w27u_wait_spent_s = {RID: m.WAIT_CAP_S}      # spent to the cap
    new_req = _req(head=0)
    _gate(stage, new_req)                                # replace, no fresh told
    _evict_host_span(tree)
    stage._weg2_store_told[RID] = TOLD                   # the rid comes back
    twin.note_follower_twin(stage, RID)
    stage.calls.clear()
    m.follower_reach_told(stage, new_req, TOLD, TOLD)
    assert stage.calls, "the reused rid inherited the spent budget: no re-read"


def test_settle_still_drops_budget_and_held_mark_of_a_departed_rid():
    stage, req, tree = _kept_follower()
    stage._w27u_wait_spent_s = {RID: 3.0}
    stage._w27u_unreached = {RID: (TOLD, HEAD)}
    assert p_intake.settle_told(stage, []) == 1
    assert RID not in stage._w27u_wait_spent_s and RID not in stage._w27u_unreached


# -- (2) divergence: kept site ----------------------------------------------------

def test_kept_visit_with_a_short_tree_and_spent_budget_is_held_not_seated(caplog):
    stage, req, tree = _kept_follower()
    _evict_host_span(tree)
    stage._w27u_wait_spent_s = {RID: m.WAIT_CAP_S}
    skips = []
    with caplog.at_level("WARNING", logger=m.logger.name):
        assert _gate(stage, req, skips) is None, "seated at told over a tree short of it"
    assert skips and skips[0][0] == m.SKIP_TOLD_PENDING
    assert stage.calls == [], "no read once the budget is spent"
    held = [x for x in caplog.messages if "FOLLOWER HELD" in x]
    assert len(held) == 1 and "route=" in held[0] and "live=%d" % HEAD in held[0]
    assert RID in getattr(stage, p_intake.KEPT_ATTR), "the verdict stands, the request is queued"


def test_kept_visit_whose_reread_does_not_bring_the_tree_is_held():
    stage, req, tree = _kept_follower()
    _evict_host_span(tree)                  # store_has=HEAD: the re-read finds nothing
    assert _gate(stage, req) is None
    assert stage.calls == [(RID, TOLD)], "one told-limited re-read first"


def test_held_kept_request_is_seated_when_its_tree_reaches_told_again():
    stage, req, tree = _kept_follower()
    _evict_host_span(tree)
    stage._w27u_wait_spent_s = {RID: m.WAIT_CAP_S}
    assert _gate(stage, req) is None
    _restore_host_span(tree)                # the span is back (read finished, sibling kept it)
    assert _gate(stage, req) == TOLD
    assert RID not in (getattr(stage, "_w27u_unreached", None) or {})


def test_hold_is_named_once_not_every_pass(caplog):
    stage, req, tree = _kept_follower()
    _evict_host_span(tree)
    stage._w27u_wait_spent_s = {RID: m.WAIT_CAP_S}
    with caplog.at_level("DEBUG", logger=m.logger.name):
        for _ in range(6):
            assert _gate(stage, req) is None
    assert len([x for x in caplog.messages if "FOLLOWER HELD" in x]) == 1
    errs = [r for r in caplog.records if r.levelname == "ERROR" and "BUDGET SPENT" in r.getMessage()]
    assert len(errs) <= 1


def test_kept_visit_with_an_intact_tree_is_not_held():
    stage, req, tree = _kept_follower()
    assert _gate(stage, req) == TOLD
    assert stage.calls == []


def test_pp0_is_never_held():
    tree = _StageTree(dev=HEAD, host=0, anchor=None, record=0)
    stage = _stage(0, tree, store_has=HEAD)
    req = _req(head=0)
    stage._w27u_wait_spent_s = {RID: m.WAIT_CAP_S}
    assert m.follower_hold_unreached(stage, req, "kept") is False


# -- the re-ask of PP0 ---------------------------------------------------------------

def test_hold_re_asks_pp0_on_the_pf_ack_stream_when_the_boot_has_one(caplog):
    stage, req, tree = _kept_follower()
    stage._weg2_fb_follower = fb._FState()
    _evict_host_span(tree)
    stage._w27u_wait_spent_s = {RID: m.WAIT_CAP_S}
    with caplog.at_level("WARNING", logger=m.logger.name):
        assert _gate(stage, req) is None
        assert _gate(stage, req) is None
    st = stage._weg2_fb_follower
    assert st.outbox == [(RID, HEAD)], "the live reach goes to PP0 once, PP0 decides"
    held = [x for x in caplog.messages if "FOLLOWER HELD" in x]
    assert len(held) == 1 and "route=pf-ack" in held[0]


def test_hold_without_an_ack_stream_names_it(caplog):
    stage, req, tree = _kept_follower()
    _evict_host_span(tree)
    stage._w27u_wait_spent_s = {RID: m.WAIT_CAP_S}
    with caplog.at_level("WARNING", logger=m.logger.name):
        assert _gate(stage, req) is None
    held = [x for x in caplog.messages if "FOLLOWER HELD" in x]
    assert len(held) == 1 and "route=none" in held[0]


def test_pp0_answer_told_zero_ends_the_hold():
    stage, req, tree = _kept_follower()
    stage._weg2_fb_follower = fb._FState()
    _evict_host_span(tree)
    stage._w27u_wait_spent_s = {RID: m.WAIT_CAP_S}
    assert _gate(stage, req) is None
    fb.follower_release(stage, RID)         # Admit(0, fallback) absorbed
    assert RID not in (stage._w27u_unreached or {})
    assert RID not in getattr(stage, p_intake.KEPT_ATTR)
    assert RID not in stage._w27u_wait_spent_s


# -- (2) divergence: admission and satisfied sites ----------------------------------------

def _plain_follower(host, record, store_has=HEAD):
    tree = _StageTree(dev=HEAD, host=host, anchor=TOLD if host else None, record=record)
    stage = _stage(1, tree, store_has=store_has)
    stage._weg2_store_told[RID] = TOLD
    twin.note_follower_twin(stage, RID)
    req = _req(head=0)
    req._weg2_early_told = TOLD
    return stage, req, tree


def test_admission_at_told_over_a_short_tree_holds_instead_of_seating():
    stage, req, tree = _plain_follower(host=0, record=TOLD)   # the record says told, the tree not
    skips = []
    got = m.admission(stage, req, lambda *a: skips.append(a))
    assert got is None, "seated at told (credit) over a tree short of it"
    assert skips and skips[0][0] == m.SKIP_TOLD_PENDING
    assert stage._weg2_store_told.get(RID) == TOLD, "the verdict is still to be admitted"
    assert twin.take_follower_twin(stage, RID), "the absolute-told mark survives the hold"


def test_held_admission_seats_when_the_tree_reaches_told():
    stage, req, tree = _plain_follower(host=0, record=TOLD)
    assert m.admission(stage, req, lambda *a: None) is None
    _restore_host_span(tree)
    got = m.admission(stage, req, lambda *a: None)
    assert got is not None
    assert RID not in stage._weg2_store_told
    assert tree.admitted_prefix(m.prefix_cap_tokens(tree, TOLD)) == TOLD


def test_satisfied_follower_over_a_short_tree_holds():
    stage, req, tree = _plain_follower(host=0, record=0)
    stage._weg2_store_told_satisfied = {RID: TOLD}
    req._weg2_early_told = None
    skips = []
    assert m.admission(stage, req, lambda *a: skips.append(a)) is None
    assert skips and skips[0][0] == m.SKIP_TOLD_PENDING
    assert stage._weg2_store_told_satisfied.get(RID) == TOLD
    assert stage._weg2_store_told.get(RID) == TOLD


def test_admission_with_a_tree_that_reaches_told_is_unchanged():
    stage, req, tree = _plain_follower(host=TOLD - HEAD, record=TOLD - HEAD, store_has=TOLD)
    req._prefetch_registered_prefix_len = HEAD     # the registered head + the span = told
    assert m.admission(stage, req, lambda *a: None) is not None
    assert stage.calls == []
