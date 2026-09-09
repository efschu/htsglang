"""The two shipped scheduling batch sizes, and the pin that keeps them single.

USER ORDER 2026-09-09, verbatim: "nimm jetzt vorerst bs4 fuer decode und bs2
fuer prefill. wenn alles fertig ist kann man das immernoch nachmessen wo da das
optimum fuer meinen anwendungsfall liegt."

Two things are asserted here, and the second is the one that will still be
earning its keep in a month:

1.  THE VALUES. ``--p-bs`` parses to 2 and ``--d-bs`` to 4, and the argv the
    launcher builds for each group carries those as ``--max-running-requests``.
    A default that parses correctly but does not reach the group's argv is the
    failure this catches.

2.  THE SINGLENESS. Every default site for these two knobs -- argparse
    defaults, function-signature defaults, ``getattr`` fallbacks, and the
    ``_flag`` fallback that reads a bs back off an argv -- must name
    ``DEFAULT_P_BS`` / ``DEFAULT_D_BS`` rather than repeat a number. This is an
    AST pin over the whole ``weg2`` package, not a grep: it reads the actual
    default EXPRESSION of each parameter and each ``add_argument`` call, so a
    numeric literal cannot hide behind formatting, a line break or a comment.
    The order calls the pair provisional and explicitly re-measurable, so the
    cost of the eventual re-measurement is exactly the number of places that
    carry the value -- the pin is what keeps that number at one apiece.

Hermetic: no GPU, no model, no launcher side effects. The AST half reads
source text and imports nothing at all.
"""

import ast
import pathlib

import pytest

from sglang.srt.weg2 import DEFAULT_D_BS, DEFAULT_P_BS
from sglang.test.ci.ci_register import register_cpu_ci

WEG2_DIR = pathlib.Path(__file__).resolve().parents[4] / "python" / "sglang" / "srt" / "weg2"

#: No disk is touched by the argv builders, exactly as the sibling #1236 file
#: relies on -- a nonexistent path proves that as it goes.
MODEL = "/nonexistent/model"

#: The knob names, in every spelling they wear across the two modules. The
#: launcher's ``_max_running_requests`` takes the bs as a bare ``bs``, and the
#: front calls P's knob ``p_concurrency`` -- both are the same two numbers and
#: both are pinned.
P_PARAMS = {"p_bs", "p_concurrency"}
D_PARAMS = {"d_bs"}
#: ``_max_running_requests(model, group, bs)`` -- group-generic name, D-shaped
#: default (its own ``group`` default is "D").
GENERIC_PARAMS = {"bs"}

P_FLAGS = {"--p-bs", "--p-concurrency"}
D_FLAGS = {"--d-bs"}

ALLOWED_NAMES = {"DEFAULT_P_BS", "DEFAULT_D_BS"}


def _weg2_sources():
    for path in sorted(WEG2_DIR.glob("*.py")):
        yield path, ast.parse(path.read_text(), filename=str(path))


# --------------------------------------------------------------- the values


def test_the_parsed_defaults_are_the_ordered_pair():
    """--p-bs 2 and --d-bs 4, off a parse with neither flag given."""
    from sglang.srt.weg2.launcher import build_parser

    ns = build_parser().parse_args(["--tree", "/t", "--tag", "x"])
    assert ns.p_bs == 2, "user order 2026-09-09: bs2 fuer prefill"
    assert ns.d_bs == 4, "user order 2026-09-09: bs4 fuer decode"
    # ...and the constants are those same two numbers, so no site can be
    # "correct" against a constant that has itself drifted from the order.
    assert (DEFAULT_P_BS, DEFAULT_D_BS) == (2, 4)


def test_the_defaults_are_independent_knobs_not_one_number():
    """Law 2 / C1-R-12. Their being UNEQUAL is the cheapest proof that the two
    are not secretly one value read twice."""
    assert DEFAULT_P_BS != DEFAULT_D_BS


def test_each_group_argv_carries_its_own_default_as_max_running_requests():
    """The default has to reach the argv, not merely parse."""
    from sglang.srt.weg2.launcher import (RING_FORM_SENTINEL_STORE_CFG, argv_d,
                                          argv_p)

    p = argv_p("py", MODEL, [1, 1, 1], 1, 1, RING_FORM_SENTINEL_STORE_CFG, [])
    d = argv_d("py", MODEL, [1, 1, 1], 1, 1, RING_FORM_SENTINEL_STORE_CFG, [])
    assert p[p.index("--max-running-requests") + 1] == str(DEFAULT_P_BS)
    assert d[d.index("--max-running-requests") + 1] == str(DEFAULT_D_BS)
    # and they differ ON THE ARGV, which is where a shared default would show.
    assert (p[p.index("--max-running-requests") + 1]
            != d[d.index("--max-running-requests") + 1])


def test_the_pricing_helpers_default_to_the_same_pair():
    """``_max_running_requests`` and ``d_mamba_ping_pong_cost`` price the
    device residency off a bs. Called without one they must charge the SHIPPED
    bs, or every unpriced call site prices a boot nobody runs."""
    from sglang.srt.weg2.launcher import (_max_running_requests,
                                          d_mamba_ping_pong_cost)

    assert _max_running_requests(MODEL, "P", DEFAULT_P_BS) == DEFAULT_P_BS
    assert _max_running_requests(MODEL) == DEFAULT_D_BS
    _strategy, per_req, extra, mrr = d_mamba_ping_pong_cost(MODEL, False)
    assert mrr == DEFAULT_D_BS
    # the product is DERIVED from the resolved bs, never a frozen literal
    assert extra == per_req * DEFAULT_D_BS


def test_the_front_defaults_agree_with_the_launcher_it_is_written_by():
    """The launcher always writes both flags (R-6/R-12), so these bind only for
    a hand-run front -- which is exactly when a divergent second copy would go
    unnoticed.

    The front builds its parser inside ``main()`` and exposes no builder, so the
    ARGPARSE half of this is covered by the AST pin below (a default that IS the
    shared constant cannot carry a different value); what is checked here is the
    object the front actually runs with."""
    from sglang.srt.weg2.front import Front

    f = Front("http://p", "http://d", "D", "t", "", 0, 0, {}, 45.0)
    assert f.p_concurrency == DEFAULT_P_BS
    assert f.d_bs == DEFAULT_D_BS
    # the seat semaphore is sized from d_bs, so this is the number of D seats
    assert f.seats_free() == DEFAULT_D_BS


# ----------------------------------------------------------- the provenance


@pytest.mark.parametrize(
    "argv, flag, expected",
    [
        ([], "--p-bs", "default"),
        ([], "--d-bs", "default"),
        (["--p-bs", "8"], "--p-bs", "flag"),
        (["--d-bs=8"], "--d-bs", "flag"),
        # an explicit value EQUAL to the default is still a choice, and a
        # value-comparison would have called this boot a default boot
        (["--d-bs", str(DEFAULT_D_BS)], "--d-bs", "flag"),
        # the other knob's presence must not colour this one
        (["--p-bs", "8"], "--d-bs", "default"),
        # a substring must not match: --p-bs-something is not --p-bs
        (["--p-bs-nonesuch", "8"], "--p-bs", "default"),
    ],
)
def test_bs_source_reads_the_argv_not_the_value(argv, flag, expected):
    from sglang.srt.weg2.launcher import bs_source

    assert bs_source(flag, argv) == expected


def _rendered_fstrings(tree):
    """Every f-string in the tree, reassembled as `text{expr}text`.

    Read through the AST rather than off the source lines: the provenance line
    is one implicitly-concatenated f-string spread over four physical lines, so
    a line-based scrape sees only its first quarter and would pass or fail on
    where the author happened to wrap. (It did: the first cut of this test
    asserted `source=` against a single line and failed on a launcher that
    prints it correctly.) The AST joins the parts the way Python does.
    """
    out = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.JoinedStr):
            continue
        parts = []
        for v in node.values:
            if isinstance(v, ast.Constant):
                parts.append(str(v.value))
            elif isinstance(v, ast.FormattedValue):
                parts.append("{" + ast.unparse(v.value) + "}")
        out.append("".join(parts))
    return out


def test_the_boot_record_prints_the_pair_with_its_source():
    """Every boot record must carry `p_bs=.. d_bs=.. source=..`; a record that
    shows only the value cannot be sorted into told/inherited later, which is
    what the order's 'nachmessen' will need."""
    tree = ast.parse((WEG2_DIR / "launcher.py").read_text())
    line = next((s for s in _rendered_fstrings(tree)
                 if "SCHEDULING BS PROVENANCE:" in s), None)
    assert line is not None, "no SCHEDULING BS PROVENANCE line in the launcher"
    assert "p_bs={p_bs}" in line, line
    assert "d_bs={d_bs}" in line, line
    # the source of EACH knob, read off the argv by the helper -- not a
    # comparison against the default, which would misfile an explicit
    # `--d-bs 4` as inherited
    assert "source={bs_source('--p-bs', argv)}|{bs_source('--d-bs', argv)}" in line, line
    # and the shipped pair travels with it, so a record is readable without
    # knowing which commit it came from
    assert "{DEFAULT_P_BS}" in line and "{DEFAULT_D_BS}" in line, line


# ------------------------------------------------------- the singleness pin


def test_no_signature_default_for_these_knobs_is_a_bare_literal():
    """AST pin: every function parameter named for one of these knobs defaults
    to the shared constant, never to a number of its own."""
    offenders = []
    for path, tree in _weg2_sources():
        for node in ast.walk(tree):
            if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            a = node.args
            params = list(a.posonlyargs) + list(a.args)
            pairs = list(zip(params[len(params) - len(a.defaults):], a.defaults))
            pairs += [(k, d) for k, d in zip(a.kwonlyargs, a.kw_defaults) if d is not None]
            for arg, default in pairs:
                if arg.arg not in (P_PARAMS | D_PARAMS | GENERIC_PARAMS):
                    continue
                if isinstance(default, ast.Name) and default.id in ALLOWED_NAMES:
                    continue
                offenders.append(
                    f"{path.name}:{default.lineno} {node.name}({arg.arg}=...) "
                    f"-> {ast.dump(default)}"
                )
    assert not offenders, (
        "a bs default that is not DEFAULT_P_BS/DEFAULT_D_BS (user order "
        "2026-09-09 makes the pair re-measurable; a second copy drifts):\n  "
        + "\n  ".join(offenders)
    )


def test_no_argparse_default_for_these_flags_is_a_bare_literal():
    """AST pin over every ``add_argument`` in the package."""
    offenders = []
    for path, tree in _weg2_sources():
        for node in ast.walk(tree):
            if not (isinstance(node, ast.Call)
                    and isinstance(node.func, ast.Attribute)
                    and node.func.attr == "add_argument"):
                continue
            flags = {a.value for a in node.args
                     if isinstance(a, ast.Constant) and isinstance(a.value, str)}
            if not flags & (P_FLAGS | D_FLAGS):
                continue
            default = next((kw.value for kw in node.keywords if kw.arg == "default"), None)
            if isinstance(default, ast.Name) and default.id in ALLOWED_NAMES:
                continue
            offenders.append(
                f"{path.name}:{node.lineno} {sorted(flags)} -> "
                + ("<no default>" if default is None else ast.dump(default))
            )
    assert not offenders, (
        "an argparse default for a bs flag that is not the shared "
        "constant:\n  " + "\n  ".join(offenders)
    )


def test_no_flag_readback_fallback_reinvents_a_bs():
    """The two fallback shapes that are neither a signature nor an
    ``add_argument`` and would therefore slip both pins above:
    ``_flag("--max-running-requests", <n>)`` and
    ``getattr(ns, "p_bs", <n>)``."""
    offenders = []
    for path, tree in _weg2_sources():
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            name = getattr(node.func, "id", None) or getattr(node.func, "attr", None)
            if name not in ("_flag", "getattr"):
                continue
            keys = [a.value for a in node.args
                    if isinstance(a, ast.Constant) and isinstance(a.value, str)]
            if not any(k in ("--max-running-requests", "p_bs", "d_bs", "p_concurrency")
                       for k in keys):
                continue
            for arg in node.args:
                if isinstance(arg, ast.Constant) and isinstance(arg.value, int):
                    offenders.append(
                        f"{path.name}:{node.lineno} {name}({keys}) fallback={arg.value!r}"
                    )
    assert not offenders, (
        "a bs fallback typed as a number instead of the shared "
        "constant:\n  " + "\n  ".join(offenders)
    )


def test_each_number_is_written_exactly_once_in_the_package():
    """The whole point of the pair: re-measuring the operating point is an edit
    to two lines. Counted as ASSIGNMENTS of the constants, so the many honest
    uses of 2 and 4 elsewhere (tp-size, ratios, measured records) are not
    swept in."""
    assignments = {"DEFAULT_P_BS": [], "DEFAULT_D_BS": []}
    for path, tree in _weg2_sources():
        for node in ast.walk(tree):
            if not isinstance(node, (ast.Assign, ast.AnnAssign)):
                continue
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            for t in targets:
                if isinstance(t, ast.Name) and t.id in assignments:
                    assignments[t.id].append(f"{path.name}:{node.lineno}")
    for name, sites in assignments.items():
        assert len(sites) == 1, f"{name} is assigned in {len(sites)} places: {sites}"
        assert sites[0].startswith("__init__.py:"), (
            f"{name} must live in the package root, not in {sites[0]}"
        )


def test_the_order_is_recorded_where_the_numbers_live():
    """A provisional number without its order is an unexplained magic constant
    the next reader will 'clean up'."""
    src = (WEG2_DIR / "__init__.py").read_text()
    assert "2026-09-09" in src
    assert "bs4" in src and "bs2" in src
    assert "nachmessen" in src, "the re-measurability clause must survive"


register_cpu_ci(__file__)
