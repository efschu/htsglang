"""UNIFY S5: the H91a / #1400 intake exists exactly once on the unified tree.

Gabel G5. Both lines built "keep #1400's told verdict until the request leaves
P's queue" (27B 6c30f7a650 = port of NF bf6c97171b, part 1; NF a3b9479f29). The
NF form is chosen: it carries the 27B part 1 byte for byte (``told_admission``,
``settle_told``; the 27B's three tests are textually the first three of
test_weg2_p_intake_h91a.py) and adds part 2 -- ``intake_phase_verdict``: the
intake stall is only a request this phase cannot fund at all (``need`` > pool,
or scarcity with nothing in flight); a request that fits free + evictable rows
stays queued, no 503, no extra flip. The verdict therefore lives until
ADMISSION, and the stall only for real impossibility.

A textual merge of the 27B tip would put a second ``p_intake`` import (and its
hooks) into scheduler.py -- the probe merge's DUP-HOOK. These pins keep the
tree at one import, one settle point, one stall gate.

Hermetic, CPU, AST only.
"""
from __future__ import annotations

import ast
import importlib.util
from pathlib import Path

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=2, suite="stage-a-weg2-unit")


def _src(mod: str) -> str:
    return Path(importlib.util.find_spec(mod).origin).read_text()


SCHED = _src("sglang.srt.managers.scheduler")
TREE = ast.parse(SCHED)


def _calls(attr: str) -> list:
    return [n for n in ast.walk(TREE) if isinstance(n, ast.Call)
            and isinstance(n.func, ast.Attribute) and n.func.attr == attr
            and isinstance(n.func.value, ast.Name) and n.func.value.id == "_p_intake"]


def test_scheduler_imports_the_intake_module_once():
    imports = [n for n in ast.walk(TREE) if isinstance(n, ast.ImportFrom)
               and n.module == "sglang.srt.weg2"
               and any(a.name == "p_intake" for a in n.names)]
    assert len(imports) == 1
    assert [a.asname for a in imports[0].names if a.name == "p_intake"] == ["_p_intake"]


def test_one_settle_point_one_told_reader_one_stall_gate():
    assert len(_calls("settle_told")) == 1
    assert len(_calls("told_admission")) == 1
    assert len(_calls("intake_phase_verdict")) == 1
    assert len(_calls("forget")) == 1


def test_the_stall_gate_sits_in_the_one_stall_observer():
    fn = next(n for n in ast.walk(TREE) if isinstance(n, ast.FunctionDef)
              and n.name == "_weg2_intake_stall_observe")
    inside = {id(n) for n in ast.walk(fn)}
    (gate,) = _calls("intake_phase_verdict")
    assert id(gate) in inside
    assert sum(1 for n in ast.walk(TREE) if isinstance(n, ast.FunctionDef)
               and n.name == "_weg2_intake_stall_observe") == 1


def test_the_intake_module_defines_each_piece_once():
    tree = ast.parse(_src("sglang.srt.weg2.p_intake"))
    names = [n.name for n in tree.body if isinstance(n, (ast.FunctionDef, ast.ClassDef))]
    for name in ("told_admission", "settle_told", "intake_phase_verdict", "forget", "pool_terms"):
        assert names.count(name) == 1, name
