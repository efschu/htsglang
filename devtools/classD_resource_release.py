#!/usr/bin/env python3
"""CLASS D rule -- a resource GRANTED on one path with no RELEASE reachable
from a layout that can hold it.

    #1189 / #888b / #902 / #715 / #681 / #852 / #814 / #844 / #919 / #684 / #1050

PRIOR ART, and why this file exists anyway.
`/spinning/gpu-arb/devtools/release_reachability.py` (2026-09-04 09:00) built
the call-graph half of this rule and its own falsification section then named
three defects in it. This file is that tool with those defects closed, plus
the half it never had, and it lives IN THE TREE next to `classB_wait_rules.py`
and `classC_latch_rule.py` -- the sweep's own rule table records the D rules
as "ausserhalb des Baums", which is a rule nobody runs.

WHAT WAS BROKEN IN THE PRIOR TOOL (all three verified by reading it):

  1. `_pattern_matches` ended
         return "?" not in pattern and not head.endswith("?") or True
     `X or True` is unconditionally true, so the opaque-hop guard was dead
     code: `a[i].free(x)` matched the pattern `pool.free`. That OVER-COUNTS
     RELEASE SITES, which pushes verdicts toward "ok" -- the wrong direction
     for a tool whose job is to find instances.
  2. Only `Scope.calls` was iterated. The form that gives the class its name --
     a plain counter, `self._inflight += 1` -- is an `ast.AugAssign` and was
     structurally invisible: `--acquire _inflight` returned "NO ACQUIRE SITE
     MATCHED", exit 2, i.e. a query error dressed as a clean run.
  3. Nothing swept. One resource pair per invocation means the enumeration
     lives in whoever remembers to type the pairs.

THE RULE, stated so it can be run instead of remembered:

    For a resource named by (grant pattern, release pattern), and for each
    LAYOUT (pp phase, tp phase, cutover, retract, abort, finish, ...):

        layout can reach a GRANT site
        and layout cannot reach any RELEASE site
        -> the layout can hold the resource and never give it back.

A GRANT or RELEASE site is any of FOUR syntactic forms, not just a call --
this is the half the prior tool lacked, and #1189 lives in form (c):

    (a) CALL        pool.free(req)              dotted-suffix pattern
    (b) COUNTER     self._inflight += 1 / -= 1  ast.AugAssign
    (c) MEMBERSHIP  self.reqs.extend(other)     append/extend/add/insert/update
                    self.reqs = [r for r ...]   and the removing twins:
                                                remove/discard/pop/clear/del
    (d) FLAG        req.pin_held = True/False   ast.Assign of a bool

WHY A NAME-BASED, DELIBERATELY OVER-APPROXIMATING CALL GRAPH.
Every edge is built by NAME: a call to `foo(...)` in scope S creates an edge
S -> every function named `foo`. That invents edges. It is the correct bias:

  * a FALSE edge can only make a release look REACHABLE, never unreachable;
  * so "no path from this layout to any release" SURVIVES the
    over-approximation and is a STRONG claim;
  * "reachable" is WEAK -- possibly a phantom name collision. The report says
    so on its own output line, not in a docstring nobody reads.

LAYOUT CUTS ARE THE POINT, NOT AN OPTION. #888b's D1 is not "no release
exists" -- `release_req` exists and is proven. It is "the only path to it is
the DECODE path, and strict phase purity forbids decode inside PP"
(`managers/phase_purity.py`: decode_forbidden_in_pp). A tool without cuts
answers "can the process ever call the release", reads CLEAN on #888b, and is
worthless for this class. `--cut` removes scopes from the graph before the
walk; the preset `pp-strict` carries the tree's own prohibition.

WHAT THIS CANNOT SEE -- read before quoting a zero:
  * dispatch that is not a syntactic call of the named attribute: a bound
    method in a dict/list, `getattr(obj, name)()`, thread/queue `target=`,
    a C-extension callback. A release reached only that way reads ABSENT.
  * anything outside the scanned root. No C++, no CUDA.
  * REACHABILITY IS NOT EXECUTION: no path conditions, no ordering. A release
    behind `if False:` counts as reachable.
  * layouts and cuts are what you name. A layout you forgot is a hole.
  * form (c) sees the MUTATION, not its cardinality. `extend` granting N
    memberships against a release that frees one OBJECT is a cardinality
    defect (#1189's actual root, `schedule_batch.py` merge_batch/filter_batch)
    and is reported by `--cardinality`, a separate, weaker check.

USAGE
    classD_resource_release.py --root python/sglang/srt --sweep
    classD_resource_release.py --root python/sglang/srt \
        --grant alloc_req_slots --release req_to_token_pool.free \
        --layouts-preset flip --cuts-preset pp-strict
    classD_resource_release.py --selftest      # can-fire proof, both ways

Exit: 0 clean, 1 at least one instance, 2 usage/query error.
"""

from __future__ import annotations

import argparse
import ast
import os
import sys
from collections import defaultdict, deque
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Set, Tuple

# --------------------------------------------------------------------------
# layouts and cuts
# --------------------------------------------------------------------------

LAYOUT_PRESETS: Dict[str, List[str]] = {
    "flip": [
        "event_loop_pp",                    # PP phase steady state
        "event_loop_normal",                # TP phase steady state
        "event_loop_overlap",               # TP phase, overlap scheduler
        "_cutover",                         # the flip seam itself
        "retract_decode",                   # pressure retraction
        "abort_request",                    # client abort
        "process_batch_result",             # normal finish
    ],
    "phases": ["event_loop_pp", "event_loop_normal"],
}

#: The tree's own prohibition, as a graph cut. `phase_purity.PhasePurity.
#: decode_forbidden_in_pp` is True on every mode except "off"; the PP layout
#: may therefore not enter the decode path, and every release site that lives
#: only there is unreachable IN THAT LAYOUT. Named scopes, so a reader can
#: check each one against the tree.
CUT_PRESETS: Dict[str, List[str]] = {
    "pp-strict": [
        "update_running_batch",     # the decode-batch maintenance pass
        "run_batch",                # ScheduleBatch execution
        "process_batch_result_decode",
        "prepare_for_decode",
    ],
    "none": [],
}


# --------------------------------------------------------------------------
# graph construction
# --------------------------------------------------------------------------


@dataclass
class Site:
    """One grant/release-shaped statement inside a scope."""

    scope: str
    lineno: int
    form: str        # call | counter | membership | flag
    spelling: str    # what the source says, dotted where possible
    direction: str   # grant | release | ambiguous


@dataclass
class Scope:
    key: str
    qualname: str
    path: str
    lineno: int
    calls: List[Tuple[str, str, int]] = field(default_factory=list)
    sites: List[Site] = field(default_factory=list)


#: form (c): container mutations that ADD a membership, and the ones that
#: remove it. `pop` is in both -- it removes from a container and is also the
#: only spelling many free-lists use to TAKE, so it is ambiguous by
#: construction and reported as such rather than guessed.
_ADD_METHODS = {"append", "extend", "add", "insert", "update", "put", "push"}
_DEL_METHODS = {"remove", "discard", "clear", "popleft", "pop", "difference_update"}
_AMBIGUOUS_METHODS = {"pop", "popleft"}


def _callee_name(func: ast.AST) -> Optional[str]:
    if isinstance(func, ast.Name):
        return func.id
    if isinstance(func, ast.Attribute):
        return func.attr
    return None


def _dotted(node: ast.AST) -> str:
    """Dotted spelling of an attribute chain.

    An opaque hop (subscript, call, comprehension) becomes a literal "?"
    segment so a pattern can never match THROUGH it. The prior tool had this
    guard and then disabled it with `or True`; here the guard is live and
    `_pattern_matches` has a unit assertion below proving it.
    """
    parts: List[str] = []
    cur = node
    while True:
        if isinstance(cur, ast.Attribute):
            parts.append(cur.attr)
            cur = cur.value
        elif isinstance(cur, ast.Name):
            parts.append(cur.id)
            break
        else:
            parts.append("?")
            break
    return ".".join(reversed(parts))


class _Collector(ast.NodeVisitor):
    def __init__(self, relpath: str) -> None:
        self.relpath = relpath
        self.scopes: List[Scope] = []
        self._stack: List[str] = []
        self._scope_stack: List[Scope] = []

    def _enter(self, node) -> None:
        self._stack.append(node.name)
        sc = Scope(
            key=f"{self.relpath}::{'.'.join(self._stack)}",
            qualname=".".join(self._stack),
            path=self.relpath,
            lineno=node.lineno,
        )
        self.scopes.append(sc)
        self._scope_stack.append(sc)

    def _leave(self) -> None:
        self._stack.pop()
        self._scope_stack.pop()

    def visit_ClassDef(self, node):  # noqa: N802
        self._stack.append(node.name)
        for child in node.body:
            self.visit(child)
        self._stack.pop()

    def visit_FunctionDef(self, node):  # noqa: N802
        self._enter(node)
        for child in node.body:
            self.visit(child)
        self._leave()

    visit_AsyncFunctionDef = visit_FunctionDef  # noqa: N815

    # ---- form (a) call, and form (c) membership (a call, different meaning)

    def visit_Call(self, node):  # noqa: N802
        if self._scope_stack:
            sc = self._scope_stack[-1]
            name = _callee_name(node.func)
            spell = _dotted(node.func)
            if name:
                sc.calls.append((name, spell, node.lineno))
                sc.sites.append(
                    Site(sc.key, node.lineno, "call", spell, "ambiguous")
                )
                if name in _ADD_METHODS or name in _DEL_METHODS:
                    if name in _AMBIGUOUS_METHODS:
                        d = "ambiguous"
                    elif name in _ADD_METHODS:
                        d = "grant"
                    else:
                        d = "release"
                    # the CONTAINER is what holds the membership, so the
                    # site's spelling is the dotted callee MINUS the method:
                    # `self.reqs.extend(...)` is a grant on `self.reqs`.
                    container = spell.rsplit(".", 1)[0] if "." in spell else spell
                    sc.sites.append(
                        Site(
                            sc.key,
                            node.lineno,
                            "membership",
                            container,
                            d,
                        )
                    )
        self.generic_visit(node)

    # ---- form (b) counter -------------------------------------------------

    def visit_AugAssign(self, node):  # noqa: N802
        if self._scope_stack and isinstance(node.op, (ast.Add, ast.Sub)):
            sc = self._scope_stack[-1]
            sc.sites.append(
                Site(
                    sc.key,
                    node.lineno,
                    "counter",
                    _dotted(node.target),
                    "grant" if isinstance(node.op, ast.Add) else "release",
                )
            )
        self.generic_visit(node)

    # ---- form (d) flag ----------------------------------------------------

    def visit_Assign(self, node):  # noqa: N802
        # ---- form (e) REBUILD: `self.reqs = [self.reqs[i] for i in keep]`
        # A container released by WHOLESALE REBUILD is released per OBJECT
        # (an element appears once in the rebuilt list however many entries it
        # held), while `.extend`/`.append` grants per ENTRY. That granularity
        # mismatch is #1189's root and #902's `release_granularity`; it is a
        # release the reachability half must see, and a mismatch the report
        # must name -- not one to silently count as "released".
        if self._scope_stack and isinstance(
            node.value, (ast.ListComp, ast.SetComp, ast.DictComp, ast.BinOp, ast.Subscript)
        ):
            sc = self._scope_stack[-1]
            for tgt in node.targets:
                if isinstance(tgt, (ast.Attribute, ast.Name)):
                    sc.sites.append(
                        Site(sc.key, node.lineno, "rebuild", _dotted(tgt), "release")
                    )
        # an empty-literal reset is also a rebuild-shaped release
        if self._scope_stack and isinstance(node.value, (ast.List, ast.Set, ast.Dict)):
            empty = not getattr(node.value, "elts", None) and not getattr(
                node.value, "keys", None
            )
            if empty:
                sc = self._scope_stack[-1]
                for tgt in node.targets:
                    if isinstance(tgt, (ast.Attribute, ast.Name)):
                        sc.sites.append(
                            Site(sc.key, node.lineno, "rebuild", _dotted(tgt), "release")
                        )
        if self._scope_stack and isinstance(node.value, ast.Constant):
            if isinstance(node.value.value, bool):
                sc = self._scope_stack[-1]
                for tgt in node.targets:
                    if isinstance(tgt, (ast.Attribute, ast.Name)):
                        sc.sites.append(
                            Site(
                                sc.key,
                                node.lineno,
                                "flag",
                                _dotted(tgt),
                                "grant" if node.value.value else "release",
                            )
                        )
        self.generic_visit(node)


@dataclass
class Graph:
    scopes: Dict[str, Scope]
    by_name: Dict[str, List[str]]
    edges: Dict[str, Set[str]]
    files_scanned: int
    parse_errors: List[Tuple[str, str]]


def build_graph(root: str, exclude: Sequence[str] = ()) -> Graph:
    scopes: Dict[str, Scope] = {}
    by_name: Dict[str, List[str]] = defaultdict(list)
    parse_errors: List[Tuple[str, str]] = []
    files = 0
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [
            d for d in dirnames if d != "__pycache__" and not d.startswith(".")
        ]
        for fn in sorted(filenames):
            if not fn.endswith(".py"):
                continue
            full = os.path.join(dirpath, fn)
            rel = os.path.relpath(full, root)
            if any(x in rel for x in exclude):
                continue
            try:
                with open(full, "r", encoding="utf-8") as fh:
                    tree = ast.parse(fh.read(), filename=full)
            except (SyntaxError, UnicodeDecodeError) as exc:
                parse_errors.append((rel, str(exc)))
                continue
            files += 1
            col = _Collector(rel)
            col.visit(tree)
            for sc in col.scopes:
                scopes[sc.key] = sc
                by_name[sc.qualname.rsplit(".", 1)[-1]].append(sc.key)

    edges: Dict[str, Set[str]] = defaultdict(set)
    for key, sc in scopes.items():
        for name, _spell, _line in sc.calls:
            for tgt in by_name.get(name, ()):
                if tgt != key:
                    edges[key].add(tgt)
    return Graph(scopes, dict(by_name), dict(edges), files, parse_errors)


# --------------------------------------------------------------------------
# site matching
# --------------------------------------------------------------------------


def _pattern_matches(spelling: str, pattern: str) -> bool:
    """Does this site's spelling match the pattern?

    Bare pattern (no dot): match the LAST segment.
    Dotted pattern: match as a dot-boundary suffix, and NEVER through an
    opaque "?" hop -- the guard the prior tool had and disabled.
    """
    if not spelling:
        return False
    if "?" in pattern:
        return False
    if "." not in pattern:
        return spelling.rsplit(".", 1)[-1] == pattern
    if spelling == pattern:
        return True
    if spelling.endswith("." + pattern):
        head = spelling[: -(len(pattern) + 1)]
        return not head.endswith("?")
    return False


# proof the guard is live, asserted at import so it cannot rot back into
# `or True` unnoticed (the exact regression this file exists to close).
assert _pattern_matches("self.req_to_token_pool.free", "req_to_token_pool.free")
assert not _pattern_matches("?.free", "req_to_token_pool.free")
assert not _pattern_matches("batches.?.pool.free", "?.pool.free")
assert _pattern_matches("self.pool.free", "free")


def find_sites(
    graph: Graph,
    pattern: str,
    direction: Optional[str] = None,
    forms: Sequence[str] = ("call", "counter", "membership", "flag", "rebuild"),
) -> List[Site]:
    out: List[Site] = []
    seen: Set[Tuple[str, int, str]] = set()
    for sc in graph.scopes.values():
        for s in sc.sites:
            if s.form not in forms:
                continue
            if direction and s.direction not in (direction, "ambiguous"):
                continue
            if _pattern_matches(s.spelling, pattern):
                k = (s.scope, s.lineno, s.form)
                if k not in seen:
                    seen.add(k)
                    out.append(s)
    return out


def resolve_scopes(graph: Graph, spec: str) -> List[str]:
    """`file:<frag>` cuts a whole subsystem; otherwise a qualname suffix."""
    if spec.startswith("file:"):
        frag = spec[5:]
        return [k for k, sc in graph.scopes.items() if frag in sc.path]
    return [
        k
        for k, sc in graph.scopes.items()
        if sc.qualname == spec or sc.qualname.endswith("." + spec)
    ]


def reachable_from(
    graph: Graph, starts: Sequence[str], cut: Set[str] = frozenset()
) -> Set[str]:
    seen = {s for s in starts if s not in cut}
    dq = deque(seen)
    while dq:
        cur = dq.popleft()
        for nxt in graph.edges.get(cur, ()):
            if nxt not in seen and nxt not in cut:
                seen.add(nxt)
                dq.append(nxt)
    return seen


def shortest_path(
    graph: Graph,
    starts: Sequence[str],
    targets: Set[str],
    cut: Set[str] = frozenset(),
) -> List[str]:
    prev: Dict[str, Optional[str]] = {s: None for s in starts if s not in cut}
    dq = deque(prev)
    while dq:
        cur = dq.popleft()
        if cur in targets:
            path = [cur]
            while prev[path[-1]] is not None:
                path.append(prev[path[-1]])  # type: ignore[arg-type]
            return list(reversed(path))
        for nxt in graph.edges.get(cur, ()):
            if nxt not in prev and nxt not in cut:
                prev[nxt] = cur
                dq.append(nxt)
    return []


# --------------------------------------------------------------------------
# the check
# --------------------------------------------------------------------------


@dataclass
class LayoutVerdict:
    layout: str
    entries: List[str]
    holds: bool
    releases: bool
    grant_witness: List[str]
    release_witness: List[str]


def analyse(
    graph: Graph,
    grant: str,
    release: str,
    layouts: Sequence[str],
    cuts: Sequence[str] = (),
    forms: Sequence[str] = ("call", "counter", "membership", "flag", "rebuild"),
) -> Tuple[List[LayoutVerdict], List[Site], List[Site], Set[str]]:
    g_sites = find_sites(graph, grant, "grant", forms)
    r_sites = find_sites(graph, release, "release", forms)
    g_scopes = {s.scope for s in g_sites}
    r_scopes = {s.scope for s in r_sites}
    cut_scopes: Set[str] = set()
    for spec in cuts:
        cut_scopes |= set(resolve_scopes(graph, spec))

    verdicts: List[LayoutVerdict] = []
    for spec in layouts:
        entries = resolve_scopes(graph, spec)
        if not entries:
            verdicts.append(LayoutVerdict(spec, [], False, False, [], []))
            continue
        reach = reachable_from(graph, entries, cut_scopes)
        hit_g = sorted(reach & g_scopes)
        hit_r = sorted(reach & r_scopes)
        verdicts.append(
            LayoutVerdict(spec, entries, bool(hit_g), bool(hit_r), hit_g[:3], hit_r[:3])
        )
    return verdicts, g_sites, r_sites, cut_scopes


def _fmt(graph: Graph, key: str, line: Optional[int] = None) -> str:
    sc = graph.scopes[key]
    return f"{sc.path}:{line or sc.lineno} {sc.qualname}"


def report(
    graph: Graph,
    grant: str,
    release: str,
    verdicts: Sequence[LayoutVerdict],
    g_sites: Sequence[Site],
    r_sites: Sequence[Site],
    cut_scopes: Set[str],
    root: str,
    cuts: Sequence[str],
    quiet: bool = False,
) -> int:
    if not quiet:
        print(f"root          : {root}")
        print(
            f"files scanned : {graph.files_scanned} "
            f"({len(graph.scopes)} scopes, {sum(len(e) for e in graph.edges.values())} edges)"
        )
        if graph.parse_errors:
            print(f"parse errors  : {len(graph.parse_errors)} (files NOT in the graph)")
            for rel, msg in graph.parse_errors[:5]:
                print(f"                {rel}: {msg[:70]}")
        if cuts:
            print(
                f"layout cuts   : {', '.join(cuts)} -> {len(cut_scopes)} scope(s) removed"
            )
        print(f"grant   {grant!r} -> {len(g_sites)} site(s)")
        for s in sorted(g_sites, key=lambda x: (x.scope, x.lineno))[:6]:
            print(f"                {_fmt(graph, s.scope, s.lineno)}  [{s.form}: {s.spelling}]")
        if len(g_sites) > 6:
            print(f"                ... {len(g_sites) - 6} more")
        print(f"release {release!r} -> {len(r_sites)} site(s)")
        for s in sorted(r_sites, key=lambda x: (x.scope, x.lineno))[:6]:
            print(f"                {_fmt(graph, s.scope, s.lineno)}  [{s.form}: {s.spelling}]")
        if len(r_sites) > 6:
            print(f"                ... {len(r_sites) - 6} more")
        print()

    if not g_sites:
        print("NO GRANT SITE MATCHED -- a query result, not a finding.")
        return 2

    bad = 0
    for v in verdicts:
        if not v.entries:
            print(f"  {v.layout:34s} ENTRY NOT FOUND (spec unmatched -- a hole, not a pass)")
            continue
        if v.holds and not v.releases:
            bad += 1
            print(f"  {v.layout:34s} *** INSTANCE: holds, NO reachable release")
        elif v.holds:
            print(f"  {v.layout:34s} ok (release reachable -- WEAK: edges over-approximate)")
        else:
            print(f"  {v.layout:34s} n/a (layout cannot reach a grant site)")
    if not quiet:
        print()
        for v in verdicts:
            if v.holds and not v.releases:
                print(f"INSTANCE {v.layout}")
                for e in v.entries[:2]:
                    print(f"  entry    {_fmt(graph, e)}")
                for a in v.grant_witness:
                    path = shortest_path(graph, v.entries, {a}, cut_scopes)
                    print(f"  grants   {_fmt(graph, a)}")
                    for hop in path[:6]:
                        print(f"             via {_fmt(graph, hop)}")
                print("  releases NONE reachable over the over-approximated graph")
        print()
        print("HONEST BOUND: 'no reachable release' is STRONG (a false edge could")
        print("only have created a path). 'ok' is WEAK (may be a phantom name")
        print("collision). Container/getattr/thread dispatch is invisible here.")
    return 1 if bad else 0


# --------------------------------------------------------------------------
# the sweep: the resource table, so the class is enumerated by one command
# --------------------------------------------------------------------------

#: (label, grant pattern, release pattern, cuts preset)
#: Every row is a resource #902 names, anchored at a real site in this tree.
SWEEP: List[Tuple[str, str, str, str]] = [
    ("kv rows (#822, covered)",        "token_to_kv_pool_allocator.alloc",
                                       "token_to_kv_pool_allocator.free",  "none"),
    ("request seat (#888b)",           "alloc_req_slots",
                                       "req_to_token_pool.free",           "none"),
    ("request seat under PP purity",   "alloc_req_slots",
                                       "req_to_token_pool.free",           "pp-strict"),
    ("mamba slot",                     "alloc_req_slots",
                                       "free_mamba_cache",                 "none"),
    ("mamba slot under PP purity",     "alloc_req_slots",
                                       "free_mamba_cache",                 "pp-strict"),
    ("mamba anchor pin (#811)",        "note_anchor_pin",
                                       "release_acked_anchor_pin",         "none"),
    ("mamba anchor pin under purity",  "note_anchor_pin",
                                       "release_acked_anchor_pin",         "pp-strict"),
    ("kvso host region (#902)",        "offload_session",
                                       "release_finished_spilled_req",     "none"),
    ("parked-carrier seat (#888b)",    "park",
                                       "readmit",                          "none"),
    ("device lock ref",                "inc_lock_ref",
                                       "dec_lock_ref",                     "none"),
    ("host lock ref",                  "inc_host_lock_ref",
                                       "dec_host_lock_ref",                "none"),
    ("host lock ref under purity",     "inc_host_lock_ref",
                                       "dec_host_lock_ref",                "pp-strict"),
    ("pinned host post",               "register_pinned_post",
                                       "unregister_pinned_post",           "none"),
    ("hicache checkpoint pin",         "pin_checkpoint",
                                       "unpin_checkpoint",                 "none"),
    ("short-term offload family",      "register_family",
                                       "unregister_family",                "none"),
    ("cuda graph buffer slot",         "register_slot",
                                       "free_slot",                        "none"),
    ("barlink abort window",           "register_abort_window",
                                       "unregister_abort_window",          "none"),
]


def run_sweep(graph: Graph, root: str, layouts: Sequence[str]) -> int:
    worst = 0
    print(f"CLASS D SWEEP  root={root}  layouts={','.join(layouts)}")
    print(f"graph: {graph.files_scanned} files, {len(graph.scopes)} scopes")
    print()
    for label, grant, release, cutname in SWEEP:
        cuts = CUT_PRESETS[cutname]
        v, g, r, cs = analyse(graph, grant, release, layouts, cuts)
        tag = f"  [cut={cutname}]" if cutname != "none" else ""
        print(f"--- {label}{tag}")
        print(f"    grant={grant!r} ({len(g)} sites)  release={release!r} ({len(r)} sites)")
        if not g:
            print("    NO GRANT SITE MATCHED -- query result, not a finding")
            worst = max(worst, 2)
            print()
            continue
        rc = report(graph, grant, release, v, g, r, cs, root, cuts, quiet=True)
        worst = max(worst, rc)
        print()
    return worst


# --------------------------------------------------------------------------
# DISCOVERY: derive the resource pairs FROM THE TREE, not from a typed table
# --------------------------------------------------------------------------
#
# The sweep table above is curated, and a curated table enumerates the class
# as it is VOCABULARISED, not as it is physically defined -- the exact
# limitation the 2026-09-04 sweep named against its own Class-A rule
# ("Vokabular statt Physik"). Discovery closes that: for every state name in
# the tree it collects the GRANT-shaped and RELEASE-shaped sites and reports
# the names that have grants and NO release at all (the strongest form: a
# counter with no decrement anywhere, a flag never cleared, a container never
# removed from), plus the names whose release exists but is unreachable from
# a layout that can hold it.
#
# Honest bound of discovery itself: it keys on the SPELLING of the state
# (`self._inflight`), so the same slot reached through a different receiver
# spelling (`other._inflight`, `batch.inflight`) is a different name here and
# a release written only that way reads as absent. Names are therefore
# reported with their spellings so the reader can collapse aliases by hand.

#: names too generic to be a resource: loop counters, indices, accumulators.
_DISCOVERY_NOISE = {
    "i", "j", "k", "n", "idx", "index", "count", "total", "num", "cnt",
    "offset", "pos", "start", "end", "step", "size", "length", "len",
    "acc", "sum", "s", "x", "y", "t", "out", "result", "ret", "res",
}


def discover(
    graph: Graph,
    min_grants: int = 1,
    forms: Sequence[str] = ("counter", "flag", "membership", "rebuild"),
) -> Dict[str, Dict[str, List[Site]]]:
    """-> {spelling: {"grant": [...], "release": [...]}} over the whole tree."""
    by_name: Dict[str, Dict[str, List[Site]]] = defaultdict(
        lambda: {"grant": [], "release": [], "ambiguous": []}
    )
    for sc in graph.scopes.values():
        for s in sc.sites:
            if s.form not in forms:
                continue
            tail = s.spelling.rsplit(".", 1)[-1]
            if not tail or tail in _DISCOVERY_NOISE or "?" in s.spelling:
                continue
            # A RESOURCE IS STATE THAT OUTLIVES THE CALL. A bare local name
            # (`reasons.append(...)`) is an accumulator inside one frame and
            # dies with it -- it cannot be held across a layout, so it is not
            # of this class. Only dotted spellings (`self.X`, `req.X`) qualify.
            if "." not in s.spelling:
                continue
            # A write in the CONSTRUCTOR is construction, not release: it runs
            # once, before anything can hold the resource. Counting it as a
            # release is how a never-cleared latch reads clean.
            if s.direction == "release" and graph.scopes[s.scope].qualname.endswith(
                ("__init__", "__new__")
            ):
                continue
            by_name[s.spelling][s.direction].append(s)
    return {
        k: v for k, v in by_name.items() if len(v["grant"]) >= min_grants
    }


def run_discovery(
    graph: Graph,
    root: str,
    layouts: Sequence[str],
    path_filter: str = "",
    top: int = 60,
) -> int:
    found = discover(graph)
    # (1) grants with NO release site anywhere -- the strongest form.
    orphans = []
    for name, d in found.items():
        if d["release"] or d["ambiguous"]:
            continue
        sites = [s for s in d["grant"] if path_filter in graph.scopes[s.scope].path]
        if sites:
            orphans.append((name, sites))
    orphans.sort(key=lambda kv: -len(kv[1]))

    print(f"DISCOVERY  root={root}  filter={path_filter or '(none)'}")
    print(f"state names with grant sites : {len(found)}")
    print(f"  of them with NO release site anywhere (in filter): {len(orphans)}")
    print()
    print("=== A. GRANTED, NEVER RELEASED (no release spelling exists at all) ===")
    for name, sites in orphans[:top]:
        s0 = sites[0]
        print(
            f"  {name:44s} {len(sites):3d} grant(s)  "
            f"{graph.scopes[s0.scope].path}:{s0.lineno} [{s0.form}]"
        )
    if len(orphans) > top:
        print(f"  ... {len(orphans) - top} more")
    print()

    # (1b) GRANULARITY MISMATCH: granted per ENTRY, released by REBUILD.
    # `#902 release_granularity`. A rebuild keyed on an object property
    # (`[r for r in self.reqs if not r.finished()]`) cannot undo a DUPLICATE
    # entry, so N grants of the same object survive one release. This is
    # #1189's root shape and it is invisible to plain reachability: the
    # release IS reachable, it is simply the wrong cardinality.
    mism = []
    for name, d in found.items():
        gr = [s for s in d["grant"] if s.form == "membership"]
        if not gr:
            continue
        rel = d["release"] + d["ambiguous"]
        if not rel:
            continue
        if all(s.form == "rebuild" for s in rel):
            if path_filter and not any(
                path_filter in graph.scopes[s.scope].path for s in gr
            ):
                continue
            mism.append((name, gr, rel))
    mism.sort(key=lambda kv: -len(kv[1]))
    print("=== A2. GRANTED PER ENTRY, RELEASED ONLY BY REBUILD (granularity) ===")
    for name, gr, rel in mism[:top]:
        print(f"  {name:40s} {len(gr)} per-entry grant(s) vs {len(rel)} rebuild release(s)")
        for s in gr[:4]:
            print(
                f"      grant   {graph.scopes[s.scope].path}:{s.lineno} "
                f"{graph.scopes[s.scope].qualname}"
            )
        for s in rel[:2]:
            print(
                f"      rebuild {graph.scopes[s.scope].path}:{s.lineno} "
                f"{graph.scopes[s.scope].qualname}"
            )
    if len(mism) > top:
        print(f"  ... {len(mism) - top} more")
    print()

    # (2) release exists, but not reachable from a layout that can hold it.
    print("=== B. RELEASE EXISTS BUT UNREACHABLE FROM A LAYOUT THAT HOLDS ===")
    bad = 0
    for name, d in sorted(found.items()):
        if not d["release"]:
            continue
        if path_filter and not any(
            path_filter in graph.scopes[s.scope].path for s in d["grant"]
        ):
            continue
        v, g, r, cs = analyse(graph, name, name, layouts, ())
        holes = [x.layout for x in v if x.holds and not x.releases]
        if holes:
            bad += 1
            s0 = d["grant"][0]
            print(
                f"  {name:40s} grants={len(d['grant']):3d} releases={len(d['release']):3d} "
                f"holes={','.join(holes)}"
            )
            print(
                f"      first grant  {graph.scopes[s0.scope].path}:{s0.lineno} "
                f"{graph.scopes[s0.scope].qualname}"
            )
            s1 = d["release"][0]
            print(
                f"      release site {graph.scopes[s1.scope].path}:{s1.lineno} "
                f"{graph.scopes[s1.scope].qualname}"
            )
    if not bad:
        print("  (none)")
    print()
    print("HONEST BOUND: discovery keys on the SPELLING of the state. A release")
    print("written through a different receiver spelling reads as absent here;")
    print("collapse aliases by reading the sites it prints.")
    return 1 if (orphans or mism or bad) else 0


def run_granularity(
    graph: Graph, root: str, grant_scope: str, path_filter: str = "", top: int = 60
) -> int:
    """GRANULARITY mode -- the #1189 rule, runnable as written.

        classD_resource_release.py --root python/sglang/srt \
            --granularity --grant-scope merge_batch

    Reports every container mutated PER ENTRY inside a scope whose qualname
    contains `--grant-scope`, whose only releases are wholesale REBUILDS. A
    rebuild keyed on an object property cannot undo a duplicate entry, so N
    grants of the same object survive one release. That is #1189.
    """
    found = discover(graph)
    rows = []
    for name, d in found.items():
        gr = [
            s
            for s in d["grant"]
            if s.form == "membership" and grant_scope in graph.scopes[s.scope].qualname
        ]
        if not gr:
            continue
        if path_filter and not any(
            path_filter in graph.scopes[s.scope].path for s in gr
        ):
            continue
        rel = d["release"] + d["ambiguous"]
        reb = [s for s in rel if s.form == "rebuild"]
        per_entry = [s for s in rel if s.form in ("membership", "counter")]
        if per_entry:
            continue  # a per-entry release exists: cardinality can be undone
        rows.append((name, gr, reb))
    rows.sort()
    print(f"GRANULARITY  root={root}  grant-scope={grant_scope!r}")
    print(f"containers granted per ENTRY with no per-entry release: {len(rows)}")
    print()
    for name, gr, reb in rows[:top]:
        tag = "MISMATCH (rebuild-only release)" if reb else "NO RELEASE AT ALL"
        print(f"  {name:34s} {tag}")
        for s in gr:
            print(
                f"     grant   {graph.scopes[s.scope].path}:{s.lineno} "
                f"{graph.scopes[s.scope].qualname}"
            )
        for s in reb[:3]:
            print(
                f"     rebuild {graph.scopes[s.scope].path}:{s.lineno} "
                f"{graph.scopes[s.scope].qualname}"
            )
        print()
    print("PRECISION NOTE: this is a FINDING list only when --grant-scope names a")
    print("mutation point whose release is known to be keyed on object identity")
    print("(merge_batch/filter_batch). Called with a broad scope it is a TRIAGE")
    print("list: 94 containers tree-wide, most of them build-once config lists.")
    return 1 if rows else 0


# --------------------------------------------------------------------------
# self-test: proof the rule CAN fire, both directions, on all four forms
# --------------------------------------------------------------------------

_VIOLATION = '''
class Pool:
    def take(self, n): ...
    def give(self, n): ...

class Runtime:
    def decode_only_helper(self):
        self.pool.give(1)
        self.inflight -= 1
        self.members.remove(1)

    def phase_pp(self):
        self.pool.take(1)
        self.inflight += 1
        self.members.append(1)

    def phase_tp(self):
        self.pool.take(1)
        self.inflight += 1
        self.members.append(1)
        self.decode_only_helper()
'''

_CLEAN = '''
class Pool:
    def take(self, n): ...
    def give(self, n): ...

class Runtime:
    def decode_only_helper(self):
        self.pool.give(1)
        self.inflight -= 1
        self.members.remove(1)

    def phase_pp(self):
        self.pool.take(1)
        self.inflight += 1
        self.members.append(1)
        self.decode_only_helper()

    def phase_tp(self):
        self.pool.take(1)
        self.inflight += 1
        self.members.append(1)
        self.decode_only_helper()
'''


def _selftest() -> int:
    import tempfile

    forms = [
        ("call",       "pool.take",   "pool.give"),
        ("counter",    "self.inflight", "self.inflight"),
        ("membership", "self.members", "self.members"),
    ]
    ok = True
    for label, src in (("VIOLATION", _VIOLATION), ("CLEAN", _CLEAN)):
        with tempfile.TemporaryDirectory() as td:
            with open(os.path.join(td, "m.py"), "w") as fh:
                fh.write(src)
            g = build_graph(td)
            for form, gp, rp in forms:
                v, gs, rs, cs = analyse(
                    g, gp, rp, ["phase_pp", "phase_tp"], ()
                )
                inst = [x.layout for x in v if x.holds and not x.releases]
                expect = ["phase_pp"] if label == "VIOLATION" else []
                mark = "OK " if inst == expect else "FAIL"
                if inst != expect:
                    ok = False
                print(
                    f"{mark} {label:9s} form={form:10s} "
                    f"grants={len(gs)} releases={len(rs)} instances={inst} "
                    f"(expected {expect})"
                )
    # the opaque-hop guard, the prior tool's dead `or True`
    print()
    print("opaque-hop guard (the prior tool's dead `or True`):")
    for spell, pat, want in (
        ("self.req_to_token_pool.free", "req_to_token_pool.free", True),
        ("?.free", "req_to_token_pool.free", False),
        ("batches.?.pool.free", "pool.free", False),
    ):
        got = _pattern_matches(spell, pat)
        mark = "OK " if got == want else "FAIL"
        if got != want:
            ok = False
        print(f"{mark} {spell!r:34s} vs {pat!r:26s} -> {got} (want {want})")
    print()
    print("PASS" if ok else "FAIL")
    return 0 if ok else 1


# --------------------------------------------------------------------------


def main(argv: Sequence[str]) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--root", default="python/sglang/srt")
    ap.add_argument("--grant")
    ap.add_argument("--release")
    ap.add_argument("--layout", action="append", default=[])
    ap.add_argument("--layouts-preset", default="flip", choices=sorted(LAYOUT_PRESETS))
    ap.add_argument("--cut", action="append", default=[])
    ap.add_argument("--cuts-preset", default="none", choices=sorted(CUT_PRESETS))
    ap.add_argument("--exclude", action="append", default=[])
    ap.add_argument("--sweep", action="store_true")
    ap.add_argument("--discover", action="store_true")
    ap.add_argument("--granularity", action="store_true")
    ap.add_argument("--grant-scope", default="merge_batch")
    ap.add_argument("--filter", default="", help="path substring for --discover")
    ap.add_argument("--top", type=int, default=60)
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args(argv)

    if args.selftest:
        return _selftest()

    if not os.path.isdir(args.root):
        print(f"root not a directory: {args.root}", file=sys.stderr)
        return 2
    graph = build_graph(args.root, args.exclude)
    layouts = args.layout or LAYOUT_PRESETS[args.layouts_preset]

    if args.sweep:
        return run_sweep(graph, args.root, layouts)

    if args.granularity:
        return run_granularity(graph, args.root, args.grant_scope, args.filter, args.top)

    if args.discover:
        return run_discovery(graph, args.root, layouts, args.filter, args.top)

    if not args.grant or not args.release:
        print("need --grant and --release (or --sweep / --selftest)", file=sys.stderr)
        return 2
    cuts = list(args.cut) + CUT_PRESETS[args.cuts_preset]
    v, g, r, cs = analyse(graph, args.grant, args.release, layouts, cuts)
    return report(graph, args.grant, args.release, v, g, r, cs, args.root, cuts)


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
