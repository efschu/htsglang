#!/usr/bin/env python3
"""Class C3 ratchet, stage C: a latch that GATES WORK and cannot clear itself.

WHY THIS EXISTS ON TOP OF latch_clear_reachability.py
-----------------------------------------------------
The stage-A/B rule (/spinning/gpu-arb/devtools/latch_clear_reachability.py,
2026-09-04) is sound but wide: on this tree it reports 307 candidates over
python/sglang/srt.  A 307-line list is a triage list, not a finding list, and
SWEEP_UNREACHABLE_CLEAR_0828 had to hand-judge its way down from 107 raw sites
to 22 verdicts for exactly this reason.

The discriminator it hand-applied is mechanisable.  A latch is only a DEFECT
when something reads it to decide whether work happens.  ``_announced``,
``_slope_logged``, ``_flight_serving_warned`` are permanently set by
construction and that is their purpose -- they suppress a repeated log line and
nothing else.  ``batch_is_full`` (#888b) and ``_escalated`` (#977) are the same
syntactic shape with a behavioural consumer, and that is the whole difference.

So stage C classifies every READ of the attribute:

    GATES-LOG    every statement the read guards is a logger/print/warn call,
                 or an assignment to another log-dedup latch.  No behaviour.
    GATES-WORK   the read guards anything else: a return, a call, an
                 assignment, a raise, a continue/break.
    RETURNED     the read is the value of a return statement -- the guard is
                 the CALLER's, so it is behavioural by construction.

A latch whose reads are all GATES-LOG is dropped.  A latch with at least one
GATES-WORK or RETURNED read and no unconditional runtime clear is the class.

WHAT THIS STILL CANNOT DECIDE (read this before acting on a count)
-----------------------------------------------------------------
Whether the guard on a conditional clear-site is NEGATED BY the state the latch
describes is interprocedural dominator analysis and is NOT decided here.
ALL-CLEARS-CONDITIONAL means "the question is live", never "the answer is yes".
The second half of that judgement is call_path/by hand, and the
``reach_evidence`` column of the sweep report carries it.

Reads are matched by attribute NAME across the whole scanned subtree, not by
resolved receiver type -- two unrelated classes with a same-named attribute are
merged.  That inflates read counts (safe direction: it can only move a latch
from LOG-ONLY into BEHAVIOURAL, i.e. towards being looked at) and it means a
BEHAVIOURAL verdict names a question, not a proven consumer.  The per-read
file:line list is printed so that inflation is visible rather than hidden.

USAGE
    classC_latch_rule.py <path> [--json] [--attr NAME] [--include-log-only]
                                [--fail-on BEHAVIOURAL]

    # the finding list this sweep was built from
    classC_latch_rule.py python/sglang/srt/managers

Exit 0 unless --fail-on is given; then exit 1 if any candidate reaches that
class.  That is the ratchet form: a new work-gating latch with no unconditional
clear cannot land silently.
"""

from __future__ import annotations

import argparse
import ast
import json
import os
import sys
from collections import defaultdict

sys.path.insert(0, "/spinning/gpu-arb/devtools")
from latch_clear_reachability import collect, rule_c3  # noqa: E402

_LOG_CALLS = {
    "debug",
    "info",
    "warning",
    "warn",
    "error",
    "exception",
    "critical",
    "log",
    "print",
    "write",
    "flush",
}


def _is_log_stmt(stmt: ast.AST, log_latches: set[str]) -> bool:
    """True if this statement does nothing but emit a diagnostic."""
    if isinstance(stmt, ast.Expr) and isinstance(stmt.value, ast.Call):
        fn = stmt.value.func
        if isinstance(fn, ast.Attribute) and fn.attr in _LOG_CALLS:
            return True
        if isinstance(fn, ast.Name) and fn.id in _LOG_CALLS:
            return True
        return False
    if isinstance(stmt, ast.Pass):
        return True
    # `self._announced = True` next to the log line is part of the dedup, not work.
    if isinstance(stmt, ast.Assign):
        for tgt in stmt.targets:
            if isinstance(tgt, ast.Attribute) and tgt.attr in log_latches:
                return True
    if isinstance(stmt, (ast.If, ast.Try, ast.With)):
        bodies = [stmt.body]
        bodies.append(getattr(stmt, "orelse", []) or [])
        bodies.append(getattr(stmt, "finalbody", []) or [])
        for h in getattr(stmt, "handlers", []) or []:
            bodies.append(h.body)
        return all(_is_log_stmt(s, log_latches) for b in bodies for s in b)
    return False


class _ReadVisitor(ast.NodeVisitor):
    """Find every ``<recv>.<attr>`` LOAD and classify what it decides."""

    def __init__(self, path: str, names: set[str], log_latches: set[str]):
        self.path = path
        self.names = names
        self.log_latches = log_latches
        self.func = "<module>"
        self.reads: dict[str, list[dict]] = defaultdict(list)

    def _names_in(self, node) -> set[str]:
        out = set()
        for n in ast.walk(node):
            if isinstance(n, ast.Attribute) and n.attr in self.names:
                if isinstance(n.ctx, ast.Load):
                    out.add(n.attr)
        return out

    def _emit(self, attr, line, kind, snippet):
        self.reads[attr].append(
            {
                "file": self.path,
                "line": line,
                "func": self.func,
                "kind": kind,
                "src": snippet[:120],
            }
        )

    def visit_FunctionDef(self, node):  # noqa: N802
        prev, self.func = self.func, node.name
        self.generic_visit(node)
        self.func = prev

    visit_AsyncFunctionDef = visit_FunctionDef

    def visit_If(self, node):  # noqa: N802
        for attr in self._names_in(node.test):
            body = list(node.body) + list(node.orelse)
            only_log = bool(body) and all(
                _is_log_stmt(s, self.log_latches) for s in body
            )
            self._emit(
                attr,
                node.test.lineno,
                "GATES-LOG" if only_log else "GATES-WORK",
                ast.unparse(node.test),
            )
        self.generic_visit(node)

    def visit_While(self, node):  # noqa: N802
        for attr in self._names_in(node.test):
            self._emit(attr, node.test.lineno, "GATES-WORK", ast.unparse(node.test))
        self.generic_visit(node)

    def visit_Return(self, node):  # noqa: N802
        if node.value is not None:
            for attr in self._names_in(node.value):
                self._emit(attr, node.lineno, "RETURNED", ast.unparse(node.value))
        self.generic_visit(node)

    def visit_IfExp(self, node):  # noqa: N802
        for attr in self._names_in(node.test):
            self._emit(attr, node.lineno, "GATES-WORK", ast.unparse(node.test))
        self.generic_visit(node)

    def visit_Assert(self, node):  # noqa: N802
        for attr in self._names_in(node.test):
            self._emit(attr, node.lineno, "GATES-WORK", ast.unparse(node.test))
        self.generic_visit(node)


def _log_latch_names(c3) -> set[str]:
    """Attribute names whose own name declares them diagnostics."""
    out = set()
    for f in c3:
        a = f["attr"]
        if (
            a.endswith(("_logged", "_announced", "_warned", "_reported", "_emitted"))
            or "_log_" in a
        ):
            out.add(a)
    return out


def scan_reads(root: str, names: set[str], log_latches: set[str]):
    reads = defaultdict(list)
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [
            d
            for d in dirnames
            if d
            not in {"__pycache__", ".git", "3rdparty", "build", ".venv", "node_modules"}
        ]
        for fn in filenames:
            if not fn.endswith(".py"):
                continue
            path = os.path.join(dirpath, fn)
            try:
                src = open(path, encoding="utf-8", errors="replace").read()
                tree = ast.parse(src, filename=path)
            except (SyntaxError, OSError):
                continue
            v = _ReadVisitor(path, names, log_latches)
            v.visit(tree)
            for attr, rs in v.reads.items():
                reads[attr].extend(rs)
    return reads


def rank(root: str, only_attr: str | None, include_log_only: bool):
    per_true, per_false, docs, comments, files = collect(root)
    c3 = rule_c3(per_true, per_false, want_all=False, only_attr=only_attr)
    names = {f["attr"] for f in c3}
    log_latches = _log_latch_names(c3)
    reads = scan_reads(root, names, log_latches)

    out = []
    for f in c3:
        rs = reads.get(f["attr"], [])
        behavioural = [r for r in rs if r["kind"] in ("GATES-WORK", "RETURNED")]
        if not rs:
            consumer = "NO-READ"
        elif behavioural:
            consumer = "BEHAVIOURAL"
        else:
            consumer = "LOG-ONLY"
        if consumer != "BEHAVIOURAL" and not include_log_only:
            continue
        out.append(
            {
                **f,
                "consumer": consumer,
                "behavioural_reads": [
                    f"{r['file']}:{r['line']} in {r['func']} [{r['kind']}] {r['src']}"
                    for r in behavioural
                ],
                "n_reads": len(rs),
            }
        )
    order = {"UNCLEARED": 0, "INIT-ONLY-CLEAR": 1, "ALL-CLEARS-CONDITIONAL": 2}
    out.sort(key=lambda f: (order.get(f["verdict"], 9), -len(f["behavioural_reads"])))
    return out, files, len(c3)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("path", nargs="?", default="python/sglang/srt/managers")
    ap.add_argument("--json", action="store_true")
    ap.add_argument("--attr", default=None)
    ap.add_argument("--include-log-only", action="store_true")
    ap.add_argument("--fail-on", default=None, choices=["BEHAVIOURAL"])
    args = ap.parse_args(argv)

    findings, files, n_c3 = rank(args.path, args.attr, args.include_log_only)
    if args.json:
        print(json.dumps({"files": files, "stage_b": n_c3, "findings": findings}, indent=2))
    else:
        print(f"# scanned {files} files under {args.path}")
        print(f"# stage-B candidates: {n_c3}   stage-C findings: {len(findings)}")
        for f in findings:
            print(f"\n[{f['verdict']} / {f['consumer']}] {f['attr']}"
                  f"  ({f['n_reads']} reads)")
            for s in f["set_sites"][:5]:
                print(f"    SET   {s}")
            for c in f["clear_sites"][:6]:
                print(f"    CLEAR {c}")
            if not f["clear_sites"]:
                print("    CLEAR <none in tree>")
            for r in f["behavioural_reads"][:6]:
                print(f"    READ  {r}")
    if args.fail_on and findings:
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
