"""S35 (18.10.): rc9 death 09:53:19Z DEADMAN_PROGRESS_STALL -- the D ranks disagreed in admission.

TP0 (Form A attention host) read NO_TOKEN from budget_state() for pdflip-29-190 right after admitting it (its own,
binding pool), TP1/TP2 read CONTINUE. The loop took the NO_TOKEN as the D-HOL head on TP0 only, and after the loop
the HOL head enters PW-R's group MIN (d_wall_head.pick_unservable_head): one rank in an all_reduce, the group hung.

These tests drive the REAL HolPass, the REAL pick_unservable_head and the REAL post-admit rule with a recording
stand-in for the group MIN, once per rank, and compare the collective entries of host and worker (the RAENGE-NIE-UNEINS
property). The control without the rule reproduces the split (host 1 entry, worker 0) -- the death's shape."""
from __future__ import annotations

import ast
import os
import pathlib
import types

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

import pytest  # noqa: E402

from sglang.srt.managers.schedule_policy import AddReqResult  # noqa: E402
from sglang.srt.weg2 import d_wall_head, hol_overtake  # noqa: E402
from sglang.srt.weg2 import form_a_post_admit as fpa  # noqa: E402

SRT = pathlib.Path(__import__("sglang.srt", fromlist=["x"]).__path__[0])
FOLLOW = object()  # the adder's H105 follow callable on a Form A group (identity only)


class _Req:
    def __init__(self, rid):
        self.rid = rid
        self.full_untruncated_fill_ids = list(range(88145))
        self.prefix_indices = list(range(88128))


class _Tree:
    """No backup wall: wall_unservable() says False -- the MIN is still ENTERED (that is the hang)."""

    def evictable_size(self):
        return 0


def _rank_pass(raw_results, follow, use_rule=True):
    """One rank's admission pass over (req, raw add_one_req result, added) triples, as the scheduler loop does it:
    the post-admit rule, the D-HOL head on NO_TOKEN, then PW-R after the loop. Returns (hol head, MIN entries)."""
    sched = types.SimpleNamespace(ps=types.SimpleNamespace(pp_size=1))
    hol = hol_overtake.HolPass(sched)
    can_run, waiting = [], []
    for req, res, added in raw_results:
        waiting.append(req)
        if added:
            can_run.append(req)
        if use_rule:
            res = fpa.uniform_result(res, bool(can_run) and req is can_run[-1], follow)
        if res != AddReqResult.CONTINUE:
            if res == AddReqResult.NO_TOKEN:
                if hol.may_overtake(req):
                    continue
            break
    calls = []

    def group_min(vals):
        calls.append(list(vals))
        return [0 for _ in vals]

    d_wall_head.pick_unservable_head(head_rid=hol.head, waiting=waiting, tree=_Tree(), available=0, group_min=group_min)
    return hol.head, len(calls)


@pytest.fixture(autouse=True)
def _group_d(monkeypatch):
    monkeypatch.setenv("SGLANG_WEG2_GROUP", "D")
    monkeypatch.delenv("SGLANG_WEG2_D_HOL_OVERTAKE", raising=False)


def _rc9(host_res):
    """The rc9 pass: pdflip-29-190 admitted on every rank (host's budget_state = host_res), pdflip-30-192 refused
    behind the skip (OTHER, uniform)."""
    a, b = _Req("pdflip-29-190"), _Req("pdflip-30-192")
    host = [(a, host_res, True), (b, AddReqResult.OTHER, False)]
    worker = [(a, AddReqResult.CONTINUE, True), (b, AddReqResult.OTHER, False)]
    return host, worker


def test_control_without_the_rule_reproduces_the_rc9_split():
    host, worker = _rc9(AddReqResult.NO_TOKEN)
    h = _rank_pass(host, FOLLOW, use_rule=False)
    w = _rank_pass(worker, FOLLOW, use_rule=False)
    assert h == ("pdflip-29-190", 1) and w == (None, 0), (h, w)   # TP0 alone in PW-R's all_reduce


def test_with_the_rule_host_and_worker_take_the_same_collectives():
    host, worker = _rc9(AddReqResult.NO_TOKEN)
    assert _rank_pass(host, FOLLOW) == _rank_pass(worker, FOLLOW) == (None, 0)


def test_a_refused_request_stays_the_hol_head_on_every_rank():
    """The group's NO_TOKEN (the host's H105 verdict, identical everywhere) is untouched: the head and the MIN stay."""
    c = _Req("pdflip-30-201")
    for raw in ([(c, AddReqResult.NO_TOKEN, False)],):
        assert _rank_pass(raw, FOLLOW) == ("pdflip-30-201", 1)


def test_rule_is_narrow():
    N, C, O = AddReqResult.NO_TOKEN, AddReqResult.CONTINUE, AddReqResult.OTHER
    assert fpa.uniform_result(N, True, FOLLOW) is C
    assert fpa.uniform_result(N, False, FOLLOW) is N          # not added: a refusal
    assert fpa.uniform_result(N, True, None) is N             # off a Form A group: budgets are the group's already
    assert fpa.uniform_result(O, True, FOLLOW) is O
    assert fpa.uniform_result(C, True, FOLLOW) is C


def test_call_edge_the_rule_runs_right_after_add_one_req_before_any_result_branch():
    t = ast.parse((SRT / "managers/scheduler.py").read_text())
    fn = next(n for n in ast.walk(t) if isinstance(n, ast.FunctionDef) and n.name == "_get_new_batch_prefill_raw")
    src = ast.unparse(fn)
    i_add = src.index("res = adder.add_one_req(")
    i_rule = src.index("res = _weg2_fa_post_admit.uniform_result(res, bool(adder.can_run_list) and req is adder.can_run_list[-1], getattr(adder, 'form_a_admission_follow', None))")
    i_branch = src.index("if res != AddReqResult.CONTINUE:", i_add)
    assert i_add < i_rule < i_branch
    # the rule is the last statement before the result branch: nothing reads the raw result in between
    assert src[i_rule:i_branch].count("\n") == 1, src[i_rule:i_branch]
