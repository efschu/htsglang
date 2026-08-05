# SPDX-License-Identifier: Apache-2.0
"""Renders BAR1 log-marker sample lines DIRECTLY from the emitter source.

Task #315: scripts/gpu_battery/s11_bar1_e2e.py's regexes (and the BAR1 test
fixtures next to them) had drifted onto dead German strings that the #295
translation moved barlink_bar1.py / barlink.py / benchmark/bar1_graph_check.py
away from. Nothing matched on a real run any more; nothing noticed, because
the fixtures were themselves still German and matched each other.

A test fixture that is a Python string literal typed next to the emitter it
claims to represent is exactly that trap, waiting to happen again on the
NEXT rename. This module removes the hand-typing step: every sample line is
built by

  1. parsing the ACTUAL emitter source file with ``ast``,
  2. finding the format-string argument of the exact logger/print/
     RuntimeError call at a known line -- asserting that line still holds
     the kind of node expected, a loud ``AssertionError`` instead of a
     silent wrong match the moment the source moves,
  3. applying it to sample values: ``%`` for the logger calls' positional
     style, ``eval()`` of the unparsed f-string for the f-string ones.

A sample line built this way can only be textually wrong if the source
itself is wrong -- retyping the English by hand is no longer a step that can
introduce the mismatch. test_bar1_marker_coupling.py uses these renderers to
check the gpu_battery consumer regexes against them directly.
"""

from __future__ import annotations

import ast
import os

REPO_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", "..")
)
BARLINK_BAR1_PY = os.path.join(
    REPO_ROOT, "python", "sglang", "srt", "distributed", "device_communicators",
    "barlink_bar1.py",
)
BARLINK_PY = os.path.join(
    REPO_ROOT, "python", "sglang", "srt", "distributed", "device_communicators",
    "barlink.py",
)
PARALLEL_STATE_PY = os.path.join(
    REPO_ROOT, "python", "sglang", "srt", "distributed", "parallel_state.py"
)
GRAPH_CHECK_PY = os.path.join(REPO_ROOT, "benchmark", "bar1_graph_check.py")

#: The exact source lines the renderers below are pinned to. Kept in one
#: place so a source move shows up as one changed number, not a hunt through
#: every render function.
LINE_BAR1_SETUP = 2236
LINE_BAR1_LEDGER = 2250
LINE_BAR1_PIPE_POOL_EXHAUSTED = 3207
LINE_BARLINK_CAPTURE_BOLT = 834
LINE_PARALLEL_STATE_GROUP_OK = 787
LINE_PARALLEL_STATE_GROUP_FALLBACK = 797
LINE_GRAPH_CHECK_HEADER = 619
LINE_GRAPH_CHECK_SUMMARY_HEADING = 663
LINE_GRAPH_CHECK_CASE_LINE = 667
LINE_GRAPH_CHECK_FAILED_GATES = 677
LINE_GRAPH_CHECK_ALL_PASSED = 680
LINE_BAR1_UNAVAILABLE_CLASS = 209


def _tree(path: str) -> ast.Module:
    with open(path, encoding="utf-8") as f:
        return ast.parse(f.read(), filename=path)


def _call_at(tree: ast.Module, lineno: int, path: str) -> ast.Call:
    """The OUTERMOST Call node that STARTS at exactly `lineno`.

    Exact match, not a containing range: a range risks picking up an outer
    call that merely wraps the one meant, and would silently extract the
    wrong string the day the source grows a wrapper around this call.

    "Outermost" because a call whose arguments themselves call something on
    the same line (``print(f"... {len(devs)} ...")``) starts two Call nodes
    at that line -- the inner one is never what an anchor line means here,
    so the one with the larger span wins rather than raising ambiguous.
    """
    hits = [
        node for node in ast.walk(tree)
        if isinstance(node, ast.Call) and node.lineno == lineno
    ]
    assert hits, (
        f"{path}:{lineno} no longer starts a call -- the emitter moved, "
        "point LINE_* in _bar1_marker_source.py at its new line"
    )
    return max(hits, key=lambda n: (n.end_lineno, n.end_col_offset))


def _percent_template(tree: ast.Module, lineno: int, path: str) -> str:
    call = _call_at(tree, lineno, path)
    arg = call.args[0]
    assert isinstance(arg, ast.Constant) and isinstance(arg.value, str), (
        f"{path}:{lineno} first argument is no longer a plain string "
        "literal -- the %-style extraction no longer applies here"
    )
    return arg.value


def _render_fstring(tree: ast.Module, lineno: int, path: str, **names) -> str:
    call = _call_at(tree, lineno, path)
    arg = call.args[0]
    src = ast.unparse(arg)
    return eval(src, {"__builtins__": {"len": len}}, dict(names))  # noqa: S307


def _class_name_at(tree: ast.Module, lineno: int, path: str) -> str:
    hits = [
        node for node in ast.walk(tree)
        if isinstance(node, ast.ClassDef) and node.lineno == lineno
    ]
    assert hits, f"{path}:{lineno} no longer starts a class definition"
    return hits[0].name


# ---------------------------------------------------------------------------
# barlink_bar1.py
# ---------------------------------------------------------------------------


def bar1_unavailable_class_name() -> str:
    return _class_name_at(_tree(BARLINK_BAR1_PY), LINE_BAR1_UNAVAILABLE_CLASS, BARLINK_BAR1_PY)


def render_setup_line(
    dauer_ms: float = 46,
    peer_targets: int = 2,
    region_mib: float = 96.0,
    slots_desc: str = "12 slots (of which 2(R-1) for all_to_all)",
    slot_kib: int = 8188,
    payload_kib: int = 24564,
    flags_bytes: int = 5376,
    export: str = "NV_ESC_EXPORT_TO_DMABUF_FD",
) -> str:
    """``barlink-BAR1: setup in ...`` -- one BAR1 region actually built."""
    tmpl = _percent_template(_tree(BARLINK_BAR1_PY), LINE_BAR1_SETUP, BARLINK_BAR1_PY)
    return tmpl % (
        dauer_ms, peer_targets, region_mib, slots_desc, slot_kib, payload_kib,
        flags_bytes, export,
    )


def render_ledger_line(group: str = "tp:0", balance: str = "tp:0: 24.0 MiB") -> str:
    """``barlink-BAR1: BAR1 ledger of this card after group ...``."""
    tmpl = _percent_template(_tree(BARLINK_BAR1_PY), LINE_BAR1_LEDGER, BARLINK_BAR1_PY)
    return tmpl % (group, balance)


def render_pipe_pool_exhausted_line(
    assigned: int = 6, total: int = 6, ring_l: int = 4, stride_bytes: int = 2048,
) -> str:
    """``barlink-BAR1-PIPE: the graph pool of the result ring is exhausted ...``."""
    tmpl = _percent_template(
        _tree(BARLINK_BAR1_PY), LINE_BAR1_PIPE_POOL_EXHAUSTED, BARLINK_BAR1_PY
    )
    return tmpl % (assigned, total, ring_l, stride_bytes)


# ---------------------------------------------------------------------------
# barlink.py
# ---------------------------------------------------------------------------


def render_capture_bolt_message(
    op: str = "all_gather",
    nbytes: int = 10600448,
    reason: str = "bar1 reports handles('all_gather', 10600448) -> False",
) -> str:
    """The RuntimeError barlink._select raises during an ungated CUDA graph
    capture -- the coverage-gap "bolt" s11_bar1_e2e.py's RE_CAPTURE_BOLT extracts.
    """
    return _render_fstring(
        _tree(BARLINK_PY), LINE_BARLINK_CAPTURE_BOLT, BARLINK_PY,
        op=op, nbytes=nbytes, reason=reason,
    )


# ---------------------------------------------------------------------------
# parallel_state.py
# ---------------------------------------------------------------------------


def render_group_ok_line(
    group: str = "tp:0", requested: str = "bar1", achieved: str = "bar1",
) -> str:
    tmpl = _percent_template(
        _tree(PARALLEL_STATE_PY), LINE_PARALLEL_STATE_GROUP_OK, PARALLEL_STATE_PY
    )
    return tmpl % (group, requested, achieved)


def render_group_fallback_line(
    group: str = "dcp:0",
    requested: str = "bar1",
    achieved: str = "gloo",
    stage: str = "setup",
    reason: str = "Bar1Unavailable: the holder reports ENOMEM",
) -> str:
    tmpl = _percent_template(
        _tree(PARALLEL_STATE_PY), LINE_PARALLEL_STATE_GROUP_FALLBACK,
        PARALLEL_STATE_PY,
    )
    return tmpl % (group, requested, achieved, stage, reason, requested, requested)


# ---------------------------------------------------------------------------
# benchmark/bar1_graph_check.py
# ---------------------------------------------------------------------------


def render_graph_check_header(devs: list = None, replays: int = 5) -> str:
    devs = [0, 1, 2] if devs is None else devs
    tree = _tree(GRAPH_CHECK_PY)
    # WIEDERGABEN is a module-level constant; read it the same way the
    # header f-string does rather than hardcoding it a second time.
    wiedergaben = None
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.Assign)
            and len(node.targets) == 1
            and isinstance(node.targets[0], ast.Name)
            and node.targets[0].id == "WIEDERGABEN"
            and isinstance(node.value, ast.Constant)
        ):
            wiedergaben = node.value.value
    assert wiedergaben is not None, (
        f"{GRAPH_CHECK_PY}: WIEDERGABEN constant not found"
    )
    return _render_fstring(
        tree, LINE_GRAPH_CHECK_HEADER, GRAPH_CHECK_PY,
        devs=devs, WIEDERGABEN=wiedergaben,
    )


def render_graph_check_summary_heading() -> str:
    return _percent_template(
        _tree(GRAPH_CHECK_PY), LINE_GRAPH_CHECK_SUMMARY_HEADING, GRAPH_CHECK_PY
    )


def render_graph_check_case_line(
    marke: str, gate: bool, name: str, grund: str = ""
) -> str:
    return _render_fstring(
        _tree(GRAPH_CHECK_PY), LINE_GRAPH_CHECK_CASE_LINE, GRAPH_CHECK_PY,
        marke=marke, gate=gate, name=name, grund=grund,
    )


def render_graph_check_failed_gates(fehlend: list) -> str:
    return _render_fstring(
        _tree(GRAPH_CHECK_PY), LINE_GRAPH_CHECK_FAILED_GATES, GRAPH_CHECK_PY,
        fehlend=fehlend,
    )


def render_graph_check_all_passed() -> str:
    return _percent_template(
        _tree(GRAPH_CHECK_PY), LINE_GRAPH_CHECK_ALL_PASSED, GRAPH_CHECK_PY
    )


def render_graph_check_transcript(
    devs: list = None,
    gate_cases: tuple = (
        ("einfach", True, True),
        ("two-graphs", True, True),
        ("wechselnde-form", True, True),
        ("reservation", True, True),
        ("gitter", False, True),
    ),
) -> str:
    """A synthetic but source-derived stand-in for one full graph_check.txt.

    Only the lines s11_bar1_e2e.py's RE_GATE_CASE and the "Summary" substring
    check actually parse are load-bearing; the surrounding shape mirrors
    bar1_graph_check.py's real layout closely enough to read as a real
    transcript without claiming to be a byte-exact capture of one.
    """
    devs = [0, 1, 2] if devs is None else devs
    lines = [render_graph_check_header(devs)]
    for name, gate, ok in gate_cases:
        lines.append(f"--- case {name!r} " + "-" * 40)
        lines.append(f"    => {'PASSED' if ok else 'FAILED'}")
    lines.append("=" * 62)
    lines.append(render_graph_check_summary_heading())
    lines.append("=" * 62)
    for name, gate, ok in gate_cases:
        marke = "PASSED" if ok else "FAILED"
        lines.append(render_graph_check_case_line(marke, gate, name))
    lines.append("")
    failed = [name for name, gate, ok in gate_cases if gate and not ok]
    if failed:
        lines.append(render_graph_check_failed_gates(failed))
    else:
        lines.append(render_graph_check_all_passed())
    return "\n".join(lines) + "\n"
