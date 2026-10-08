# SPDX-License-Identifier: Apache-2.0
"""L15-FIX-DUPKW: every l15_* CALL SEAM in the scheduler and the weight
updater must match the callee signature (generic version of the
hook call-shape pin -- the boot killer was a seam nobody checked).

For each `l15_<mod>.<attr>(...)` call in scheduler.py and
scheduler_components/weight_updater.py whose module is imported there,
resolve the callee via importlib and inspect.signature().bind() with
placeholders. Calls with *args/**kwargs are skipped (the one ** call,
the retain hook, is pinned by test_pdflip_l15_hook_callshape_1001).
"""

from __future__ import annotations

import ast
import importlib
import inspect
import pathlib
import sys

REPO = pathlib.Path(__file__).resolve().parents[4]
sys.path.insert(0, str(REPO / "python"))

L15_MODULES = {
    "l15_bind", "l15_retain", "l15_restore", "l15_refill", "l15_manifest",
    "l15_plan", "l15_shadow", "l15_keep_arm", "l15_compact",
    "l15_wake_check", "l15_sample", "l15_check",
}

FILES = [
    REPO / "python/flliper/srt/managers/scheduler.py",
    REPO / "python/flliper/srt/managers/scheduler_components/weight_updater.py",
]


def _imported_l15(tree):
    names = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and (node.module or "").startswith(
            "flliper.srt.pdflip"
        ):
            names |= {a.asname or a.name for a in node.names}
        elif isinstance(node, ast.Import):
            for a in node.names:
                if a.name.startswith("flliper.srt.pdflip."):
                    names.add((a.asname or a.name).split(".")[-1])
    return names & L15_MODULES


def test_l15_call_seams_bind():
    checked, failures = [], []
    for path in FILES:
        tree = ast.parse(path.read_text())
        mods = _imported_l15(tree)
        rel = str(path.relative_to(REPO))
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            fn = node.func
            if not (
                isinstance(fn, ast.Attribute)
                and isinstance(fn.value, ast.Name)
                and fn.value.id in mods
            ):
                continue
            mod_name, attr = fn.value.id, fn.attr
            if any(isinstance(a, ast.Starred) for a in node.args) or any(
                kw.arg is None for kw in node.keywords
            ):
                continue  # ** call: pinned by the hook call-shape test
            where = "%s:%d %s.%s" % (rel, node.lineno, mod_name, attr)
            try:
                callee = getattr(
                    importlib.import_module("flliper.srt.pdflip." + mod_name), attr
                )
                if isinstance(callee, type) and issubclass(callee, BaseException):
                    # an exception raise: constructor takes *args by design
                    checked.append(where + " (exception *args)")
                    continue
                inspect.signature(callee).bind(
                    *("p",) * len(node.args),
                    **{kw.arg: "p" for kw in node.keywords},
                )
                checked.append(where)
            except (AttributeError, TypeError, ValueError) as exc:
                failures.append("%s: %s" % (where, exc))
    print("checked %d l15 call seams:\n%s" % (len(checked), "\n".join(checked)))
    assert not failures, "call seams that do not bind:\n" + "\n".join(failures)
