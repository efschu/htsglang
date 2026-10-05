# SPDX-License-Identifier: Apache-2.0
"""L15-NOPARK-INIT (desk 2018): an idle D enters the L15 retain block.

Cause (deskq/done/2017): the D-sleep L15 block (reuse / SLEEP-AGREE / TREE-CAND / retain /
KEEP-CLEAR) is gated on ``weg2_d_parked is not None``, an attribute only
``d_park_runtime.parked_list`` creates. An idle D (outstanding=0 at 42 of 42 flips) never
parks, so the block never ran. Fix: ``l15_plan.noparkinit_apply`` sets ``[]`` (the state
``hold_parked`` leaves after a sleep) when L15 master + SGLANG_WEG2_L15_NOPARK_INIT + group D
and not dual-layout. Everything else stays byte for byte.

Hermetic, no CUDA. Mutants (gate removed one by one, default ON) are exec'd from the mutated
module source and must fail the matrix.

Run (own worktree, capped):
  cd /spinning/wt-27b-l15-consume-1005 && /spinning/gpu-arb/pytest_gedeckelt.sh \
    test/registered/unit/weg2/test_l15_noparkinit_1005.py
"""
from __future__ import annotations

import inspect
import os
import pathlib
import re
import types
from types import SimpleNamespace

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

import pytest  # noqa: E402

from sglang.srt.weg2 import d_park_runtime as DPR  # noqa: E402
from sglang.srt.weg2 import l15_plan  # noqa: E402

ROOT = pathlib.Path(l15_plan.__file__).resolve().parents[1]  # python/sglang/srt
SCHED = (ROOT / "managers" / "scheduler.py").read_text()
ENVIRON = (ROOT / "environ.py").read_text()

ON = {"SGLANG_WEG2_L15": "1", "SGLANG_WEG2_L15_NOPARK_INIT": "1", "SGLANG_WEG2_GROUP": "D"}


def _s(**kw):
    return SimpleNamespace(**kw)


def matrix(fn):
    """Behavioural matrix of one implementation; returns the list of violated rows."""
    bad = []

    def run(env, **attrs):
        s = _s(**attrs)
        r = fn(s, env)
        return r, getattr(s, "weg2_d_parked", "ABSENT"), s

    r, v, _ = run(ON, weg2_d_parked=None)
    if not (r is True and v == []):
        bad.append("on: None -> []")
    r, v, _ = run(ON)  # attribute absent entirely (never created)
    if not (r is True and v == []):
        bad.append("on: absent -> []")
    # switch off (absent / 0 / junk): untouched
    for sw in (None, "0", "", "no"):
        env = dict(ON)
        if sw is None:
            env.pop("SGLANG_WEG2_L15_NOPARK_INIT")
        else:
            env["SGLANG_WEG2_L15_NOPARK_INIT"] = sw
        r, v, _ = run(env, weg2_d_parked=None)
        if not (r is False and v is None):
            bad.append("switch off %r stays None" % (sw,))
    # no master
    for m in (None, "0"):
        env = dict(ON)
        if m is None:
            env.pop("SGLANG_WEG2_L15")
        else:
            env["SGLANG_WEG2_L15"] = m
        r, v, _ = run(env, weg2_d_parked=None)
        if not (r is False and v is None):
            bad.append("no L15 master %r stays None" % (m,))
    # dual layout
    r, v, _ = run(dict(ON, SGLANG_WEG2_DUAL_LAYOUT="1"), weg2_d_parked=None)
    if not (r is False and v is None):
        bad.append("dual layout stays None")
    # other groups / no group (P shares the flush code and must stay out of the block)
    for g in ("P", "", None):
        env = dict(ON)
        if g is None:
            env.pop("SGLANG_WEG2_GROUP")
        else:
            env["SGLANG_WEG2_GROUP"] = g
        r, v, _ = run(env, weg2_d_parked=None)
        if not (r is False and v is None):
            bad.append("group %r stays None" % (g,))
    # an existing list (a real park) is never replaced
    lst = [object()]
    r, v, _ = run(ON, weg2_d_parked=lst)
    if not (r is False and v is lst):
        bad.append("parked list kept")
    empty = []
    r, v, _ = run(ON, weg2_d_parked=empty)
    if not (r is False and v is empty):
        bad.append("existing [] kept (same object)")
    return bad


def _mutant(old: str, new: str):
    src = inspect.getsource(l15_plan)
    assert src.count(old) == 1, old
    mod = types.ModuleType("l15_plan_mutant")
    mod.__file__ = l15_plan.__file__
    import sys

    sys.modules[mod.__name__] = mod  # @dataclass resolves its module through sys.modules
    try:
        exec(compile(src.replace(old, new), "l15_plan_mutant.py", "exec"), mod.__dict__)
    finally:
        sys.modules.pop(mod.__name__, None)
    return mod.noparkinit_apply


# ---------------------------------------------------------------- the real function

def test_matrix_real_function_clean():
    assert matrix(l15_plan.noparkinit_apply) == []


def test_switch_off_is_byte_equal_to_no_call():
    """Switch off: no attribute is created, nothing is written, the return is False."""
    for env in ({}, {"SGLANG_WEG2_L15": "1", "SGLANG_WEG2_GROUP": "D"}):
        s = _s()
        assert l15_plan.noparkinit_apply(s, env) is False
        assert vars(s) == {}


def test_dual_group_d_never_touched_even_with_both_switches():
    """The dual layout runs a group D too; it must stay None there."""
    s = _s(weg2_d_parked=None)
    env = dict(ON, SGLANG_WEG2_DUAL_LAYOUT="1")
    assert l15_plan.noparkinit_apply(s, env) is False
    assert s.weg2_d_parked is None
    # and the launcher refusal for L15 on --dual-layout is still in place
    assert l15_plan.refuse_dual(["--dual-layout"], {"SGLANG_WEG2_L15": "1"}) is not None


def test_env_default_off_and_declared():
    from sglang.srt.environ import envs

    assert "SGLANG_WEG2_L15_NOPARK_INIT = EnvBool(False)" in ENVIRON
    assert l15_plan.NOPARK_INIT_ENV == "SGLANG_WEG2_L15_NOPARK_INIT"
    old = os.environ.pop("SGLANG_WEG2_L15_NOPARK_INIT", None)
    try:
        assert envs.SGLANG_WEG2_L15_NOPARK_INIT.get() is False
    finally:
        if old is not None:
            os.environ["SGLANG_WEG2_L15_NOPARK_INIT"] = old


# ---------------------------------------------------------------- [] carries "parked, nothing in it"

def test_gate_becomes_true_and_empty_list_is_inert_for_the_readers():
    s = _s(weg2_d_parked=None)
    assert l15_plan.noparkinit_apply(s, ON) is True
    # the four L15 gates of the sleep flush: `is not None`
    assert getattr(s, "weg2_d_parked", None) is not None
    # every other reader is truthiness / `or []` based: [] == nothing parked
    assert not getattr(s, "weg2_d_parked", None)
    assert DPR.parked_list(s) is s.weg2_d_parked          # no re-creation, same list
    assert DPR.hold_parked(s, hold_armed=True) == 0       # dormant point: nothing to hold
    assert DPR.hold_parked(s, hold_armed=False) == 0
    assert s.weg2_d_parked == []                          # unchanged by the sleep leg
    assert DPR.park_tick(s) == 0                          # awake tick returns early
    assert s.weg2_d_parked == []


def test_scheduler_readers_of_weg2_d_parked_are_none_gates_only_in_the_l15_block():
    """Every `weg2_d_parked ... is not None` in scheduler.py is one of the four L15 gates (reuse,
    agree-on, retain branch, KEEP-CLEAR); all other uses are truthiness / `or []` and so treat []
    as 'nothing parked'. A new `is not None` reader would need a look."""
    hits = [m.start() for m in re.finditer(r'"weg2_d_parked", None\)\s*is not None', SCHED)]
    assert len(hits) == 4, len(hits)
    assert len(re.findall(r'"weg2_d_parked", None\)\s*is None', SCHED)) == 0
    # no == None / != None / identity compare elsewhere in the weg2 package
    for p in (ROOT / "weg2").glob("*.py"):
        t = p.read_text()
        assert not re.search(r"weg2_d_parked[^\n]{0,40}(==|!=)\s*None", t), p.name


# ---------------------------------------------------------------- call site in the scheduler

def test_scheduler_call_site_precedes_the_reuse_gate_inside_the_try():
    call = "l15_plan.noparkinit_apply(self, os.environ)"
    assert SCHED.count(call) == 1
    i = SCHED.index(call)
    j = SCHED.index("_l15_reuse = (\n", i)
    assert 0 < j - i < 400, "call must sit directly before `_l15_reuse = (`"
    # after the imports of the retain block (l15_plan is bound there), and the four gates remain
    k = SCHED.rindex("l15_sleep_once,\n                )", 0, i)
    assert i - k < 1500
    assert SCHED.count('getattr(self, "weg2_d_parked", None) is not None') >= 3


# ---------------------------------------------------------------- mutants

MUTANTS = {
    "gate group D removed": (
        '        and str(env.get("SGLANG_WEG2_GROUP", "") or "").strip().upper() == "D"\n', ""),
    "gate dual removed": (
        '        and str(env.get("SGLANG_WEG2_DUAL_LAYOUT", "") or "").strip() != "1"\n', ""),
    "gate master removed": ("        master_on(env)\n        and _switch(env, NOPARK_INIT_ENV)",
                            "        _switch(env, NOPARK_INIT_ENV)"),
    "default ON": ("_switch(env, NOPARK_INIT_ENV)",
                   "(str(env.get(NOPARK_INIT_ENV, '1')).strip().lower() in _ON_VALUES)"),
    "switch ignored": ("        and _switch(env, NOPARK_INIT_ENV)\n", ""),
    "overwrites a real park": (
        '    if getattr(sched, "weg2_d_parked", None) is not None:\n        return False\n', ""),
}


@pytest.mark.parametrize("name", sorted(MUTANTS))
def test_mutant_goes_red(name):
    old, new = MUTANTS[name]
    bad = matrix(_mutant(old, new))
    assert bad, "mutant %r survived the matrix" % name
