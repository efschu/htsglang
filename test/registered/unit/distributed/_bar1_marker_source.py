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

#: STRUCTURAL anchors: a distinctive fragment of the emitted string itself.
#:
#: These used to be absolute line numbers (``MARK_BAR1_SETUP = 2236``). That
#: coupled this module to the LAYOUT of files it does not own: an unrelated
#: insertion anywhere above the pinned line silently re-aimed the AST lookup,
#: and the whole test module failed at COLLECTION. It happened on 2026-08-05
#: -- adding a counter to ``parallel_state.py`` shifted line 787 and took the
#: entire BAR1 marker suite down with it.
#:
#: A line number is the same fragility class as a hand-typed fixture string
#: (#380): it is a second, unmaintained copy of a fact the source already
#: states. Anchoring on the MARKER TEXT instead couples this module to the
#: one thing it actually cares about -- so it survives every edit that does
#: not touch the marker, and still fails loudly on the edit that does, which
#: is precisely the drift this module exists to catch.
#:
#: Each anchor must match exactly one call in its file; ``_call_with_marker``
#: raises naming all candidates if that stops being true.
MARK_BAR1_SETUP = "barlink-BAR1: setup in "
MARK_BAR1_LEDGER = "barlink-BAR1: BAR1 ledger of this card after group "
MARK_BAR1_PIPE_POOL_EXHAUSTED = (
    "barlink-BAR1-PIPE: the graph pool of the result ring is exhausted"
)
MARK_BARLINK_CAPTURE_BOLT = "bytes during a CUDA graph capture"
MARK_PARALLEL_STATE_GROUP_OK = "barlink enabled for group "
MARK_PARALLEL_STATE_GROUP_FALLBACK = "barlink group '%s': requested="
MARK_GRAPH_CHECK_HEADER = "BAR1 graph proof: devices "
MARK_GRAPH_CHECK_SUMMARY_HEADING = "Summary"
MARK_GRAPH_CHECK_CASE_LINE = "[Gate]"
MARK_GRAPH_CHECK_FAILED_GATES = "Failed gate cases:"
MARK_GRAPH_CHECK_ALL_PASSED = "All gate cases passed."

#: The unavailable-class anchor is its DOCSTRING, not its name: the whole
#: point of `bar1_unavailable_class_name` is to report a RENAME to the
#: consumer regexes, so anchoring on the name would make it a tautology.
MARK_BAR1_UNAVAILABLE_CLASS = "The BAR1 path is not available on this machine."


def _tree(path: str) -> ast.Module:
    with open(path, encoding="utf-8") as f:
        return ast.parse(f.read(), filename=path)


def _first_arg_text(node: ast.Call) -> str | None:
    """Source text of a call's first argument, or None if it has none.

    A plain string literal yields its VALUE; anything else (f-string,
    concatenation) yields its unparsed source, so a marker can be found in
    either style without the caller knowing which one the emitter uses.
    """
    if not node.args:
        return None
    arg = node.args[0]
    if isinstance(arg, ast.Constant) and isinstance(arg.value, str):
        return arg.value
    return ast.unparse(arg)


def _call_with_marker(tree: ast.Module, marker: str, path: str) -> ast.Call:
    """The unique Call whose first argument contains `marker`.

    Structural, not positional: unaffected by edits elsewhere in the file.

    "Outermost" for calls that start at the same place (``print(f"...
    {len(devs)} ...")`` nests a Call inside the argument): the one with the
    larger span wins, since an inner helper call is never what a marker
    means here.

    Ambiguity is an ERROR, not a silent first-match: two calls carrying the
    same marker means the anchor no longer identifies one emitter, and
    guessing would reintroduce exactly the silent-wrong-match failure this
    module was written to remove.
    """
    hits = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and marker in (_first_arg_text(node) or "")
    ]
    assert hits, (
        f"{path}: no call's first argument contains {marker!r} any more -- "
        "the emitter's text changed (or the call moved to another file). "
        "Update the MARK_* anchor in _bar1_marker_source.py AND the "
        "gpu_battery consumer regexes that read this marker."
    )
    # Collapse nested candidates that share a start position.
    by_start: dict = {}
    for node in hits:
        key = (node.lineno, node.col_offset)
        best = by_start.get(key)
        if best is None or (node.end_lineno, node.end_col_offset) > (
            best.end_lineno,
            best.end_col_offset,
        ):
            by_start[key] = node
    if len(by_start) > 1:
        where = ", ".join(f"line {ln}" for ln, _ in sorted(by_start))
        raise AssertionError(
            f"{path}: marker {marker!r} matches {len(by_start)} distinct "
            f"calls ({where}) -- it no longer identifies one emitter. "
            "Make the MARK_* anchor more specific."
        )
    return next(iter(by_start.values()))


def _percent_template(tree: ast.Module, marker: str, path: str) -> str:
    call = _call_with_marker(tree, marker, path)
    arg = call.args[0]
    assert isinstance(arg, ast.Constant) and isinstance(arg.value, str), (
        f"{path}: the call carrying {marker!r} no longer takes a plain "
        "string literal first argument -- the %-style extraction no longer "
        "applies here"
    )
    return arg.value


def _render_fstring(tree: ast.Module, marker: str, path: str, **names) -> str:
    call = _call_with_marker(tree, marker, path)
    arg = call.args[0]
    src = ast.unparse(arg)
    return eval(src, {"__builtins__": {"len": len}}, dict(names))  # noqa: S307


def _class_name_by_doc(tree: ast.Module, marker: str, path: str) -> str:
    """Name of the unique class whose DOCSTRING contains `marker`."""
    hits = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.ClassDef)
        and marker in (ast.get_docstring(node) or "")
    ]
    assert hits, (
        f"{path}: no class docstring contains {marker!r} any more -- update "
        "the MARK_* anchor in _bar1_marker_source.py"
    )
    assert len(hits) == 1, (
        f"{path}: marker {marker!r} matches {len(hits)} classes "
        f"({', '.join(n.name for n in hits)}) -- make the anchor specific"
    )
    return hits[0].name


# ---------------------------------------------------------------------------
# barlink_bar1.py
# ---------------------------------------------------------------------------


def bar1_unavailable_class_name() -> str:
    return _class_name_by_doc(
        _tree(BARLINK_BAR1_PY), MARK_BAR1_UNAVAILABLE_CLASS, BARLINK_BAR1_PY
    )


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
    tmpl = _percent_template(_tree(BARLINK_BAR1_PY), MARK_BAR1_SETUP, BARLINK_BAR1_PY)
    return tmpl % (
        dauer_ms, peer_targets, region_mib, slots_desc, slot_kib, payload_kib,
        flags_bytes, export,
    )


def render_ledger_line(group: str = "tp:0", balance: str = "tp:0: 24.0 MiB") -> str:
    """``barlink-BAR1: BAR1 ledger of this card after group ...``."""
    tmpl = _percent_template(_tree(BARLINK_BAR1_PY), MARK_BAR1_LEDGER, BARLINK_BAR1_PY)
    return tmpl % (group, balance)


def render_pipe_pool_exhausted_line(
    assigned: int = 6, total: int = 6, ring_l: int = 4, stride_bytes: int = 2048,
) -> str:
    """``barlink-BAR1-PIPE: the graph pool of the result ring is exhausted ...``."""
    tmpl = _percent_template(
        _tree(BARLINK_BAR1_PY), MARK_BAR1_PIPE_POOL_EXHAUSTED, BARLINK_BAR1_PY
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
        _tree(BARLINK_PY), MARK_BARLINK_CAPTURE_BOLT, BARLINK_PY,
        op=op, nbytes=nbytes, reason=reason,
    )


# ---------------------------------------------------------------------------
# parallel_state.py
# ---------------------------------------------------------------------------


def render_group_ok_line(
    group: str = "tp:0", requested: str = "bar1", achieved: str = "bar1",
) -> str:
    tmpl = _percent_template(
        _tree(PARALLEL_STATE_PY), MARK_PARALLEL_STATE_GROUP_OK, PARALLEL_STATE_PY
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
        _tree(PARALLEL_STATE_PY), MARK_PARALLEL_STATE_GROUP_FALLBACK,
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
        tree, MARK_GRAPH_CHECK_HEADER, GRAPH_CHECK_PY,
        devs=devs, WIEDERGABEN=wiedergaben,
    )


def render_graph_check_summary_heading() -> str:
    return _percent_template(
        _tree(GRAPH_CHECK_PY), MARK_GRAPH_CHECK_SUMMARY_HEADING, GRAPH_CHECK_PY
    )


def render_graph_check_case_line(
    marke: str, gate: bool, name: str, grund: str = ""
) -> str:
    return _render_fstring(
        _tree(GRAPH_CHECK_PY), MARK_GRAPH_CHECK_CASE_LINE, GRAPH_CHECK_PY,
        marke=marke, gate=gate, name=name, grund=grund,
    )


def render_graph_check_failed_gates(fehlend: list) -> str:
    return _render_fstring(
        _tree(GRAPH_CHECK_PY), MARK_GRAPH_CHECK_FAILED_GATES, GRAPH_CHECK_PY,
        fehlend=fehlend,
    )


def render_graph_check_all_passed() -> str:
    return _percent_template(
        _tree(GRAPH_CHECK_PY), MARK_GRAPH_CHECK_ALL_PASSED, GRAPH_CHECK_PY
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
