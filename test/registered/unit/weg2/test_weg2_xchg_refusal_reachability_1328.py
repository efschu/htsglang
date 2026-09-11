# SPDX-License-Identifier: Apache-2.0
"""#1328 (B4l, spec AMENDMENT 7 point 3): the REACHABILITY RATCHET for the
exchange lane's refusals -- a W-code that no production path can reach is a
guard that computes, records and refuses NOTHING, and it is green in every
suite ever written for it.

THE DEFECT CLASS, measured, not argued.  Boot weg2xsn14 printed
``uncovered=12`` on 39 of 40 ``WEG2-XCHG-COVER`` lines while
``W84 Weg2XchgCoverageRefused`` read genuine 0
(``BOOT_weg2xsn14_0911.md`` WALL 1).  Nothing was mis-predicated: the
verdict was computed by ``weight_exchange.arm_coverage``, recorded through
``_record_boot_vote``, and then read by nobody -- ``boot_vote()`` had ZERO
production callers at ``692b1e8698`` (only its own ``def``, its ``__all__``
entry, and a COMMENT in ``model_executor/model_runner.py:2561`` that names
it, which is the #1001 mention-vs-use trap in its purest form), and the
only raiser of the class, ``refuse_if_not_ok``, had none either.  B4g
(``a5eb06cdaa``) closed it by adding ``coverage_leg_decision()`` -- which
DOES read the vote -- and wiring it at the wake fence
(``weight_updater.py:1270``).  The suite was green before and after; only
the metal knew.

THE TWO RATCHETS, and why there are two rather than one.

* :class:`TestNoOrphanVerdict` is the W84 form EXACTLY, with an EMPTY
  allowlist and nothing to keep current: for every module-level verdict
  slot in the lane that a production-reachable function WRITES, some
  production-reachable function must READ it.  This is the assertion that
  is RED at ``692b1e8698`` and GREEN at ``dfceb7004e`` -- see RED-FIRST
  below.  It generalises beyond W84 because it never mentions W84: any
  future "compute it, record it, act on it nowhere" lands on it.

* :class:`TestEveryWCodeRaiserHasAProductionCaller` asks the plain
  reachability question at EVERY ``raise Weg2*`` site in the lane (165 of
  them on this tip), with no list of good names -- and carries the sites
  that DO NOT pass as a FROZEN, NAMED DEBT, closed in both directions
  (a new unwired raiser FAILS; a debt entry that became wired FAILS as
  stale; the debt's length is pinned so it cannot grow by accident).

WHY THE DEBT EXISTS -- a DEVIATION from AMENDMENT 7's "allowlist empty",
reported to the operator and ACCEPTED with conditions (2026-09-11: each
entry carries file:line, one line of reason, AND ITS CLASS -- "a debt whose
entries are classified is a work list; an unclassified one is an exemption
with extra steps").  An empty allowlist on THIS assertion cannot be green
on the shipping line, and that is a measurement, not a preference: a ratchet
that is red on arrival enforces nothing.  The intent it must serve is "no
W-code raise without a proven production path", and both directions of a
pinned, classified debt serve it better than an allowlist that hides by
omission.

THE DEBT'S FOURTH ENTRY IS ALREADY PAID, and the way it left is the
evidence that the GONE direction has teeth on a real change rather than a
synthetic one: ``launcher.xchg_form_dormant_reserve`` had no caller
anywhere in production on ``dfceb7004e`` -- which is statically why boot
weg2xsn15 emitted ``WEG2-XCHG-RESERVE`` ZERO times (operator handover
section 12, "Launcher-Instrument-Defekt") -- and B4i (``2c9592fd19``) wired
it into ``main()`` before the dry-return.  This file was rebased onto that
tip, ``test_the_debt_is_not_stale`` fired on the entry, and the entry was
DELETED.  Debt 4 -> 3.

The three that remain, each with its class (the operator's ruling of
2026-09-11 after verifying all three at ``2c9592fd19``):

1. ``weight_exchange.refuse_if_not_ok`` (W84,
   ``weight_exchange.py:2513``) -- class UNWIRED-BY-DESIGN-AND-PINNED.  An
   EXISTING test asserts it stays that way
   (``test_weg2_coverage_verdict_1273.py::test_refuse_if_not_ok_stays_unwired``);
   its docstring gives the reason (no group fence at the end of weight
   loading, so a rank raising there dies while the other five walk into a
   collective with five members).  Requiring a caller would demand the
   opposite of a decision already taken and tested.  The VERDICT it hands
   over is what must be reachable, and that is
   :class:`TestNoOrphanVerdict`'s question.
2. ``weight_exchange_transport.arm_oncard_lane``
   (``Weg2XchgOnCardUnavailable``, ``weight_exchange_transport.py:2325``,
   raise at ``:2377``) -- class DECLARED-TODO(S6), NOT a silent defect, and
   this is the operator's CORRECTION of this ratchet's first reading.  The
   on-card MODE IS decided and wired: the launcher publishes
   ``env[ENV_ONCARD_MODE]`` from argv at ``launcher.py:3512``.  What has no
   producer yet is the S6 PROBE-AND-DEGRADE arm (the W72 two-arm shape),
   and that is DECLARED: the docstring carries ``TODO(S6)``, an existing
   test asserts the marker is present
   (``test_weg2_xchg_transport_1273.py:1995``), and
   ``weight_exchange_shadow.py:32`` says in prose that this arm has no
   producer on the boot path.  Owed S6 work with a name -- filing it as a
   fresh defect would commission a second implementation of a function that
   is waiting for its spec.
3. ``weight_exchange_transport.diagonal_carrier_bytes``
   (``Weg2XchgOncardSlotRefused``, ``:2410``, raise at ``:2429``, #1334) --
   class SECOND-AUTHORITY-CANDIDATE, and the operator's correction here
   matters: **the W82 slot-size guard is LIVE**.  ``Weg2XchgOncardSlotRefused``
   has REACHABLE raisers at ``weight_exchange_transport.py:198`` and
   ``:204`` inside ``validate_oncard_slot_mib``, which the launcher calls at
   ``:3518``; only the raise inside THIS unused helper is unreachable, so
   "the class is unreachable" was wrong as written.  What the finding
   actually uncovers is better: ``:2453``'s docstring names
   ``diagonal_carrier_bytes`` as the sizer "and the ledger charges the
   group-wide sum", while #1334 sized the deposit inline as 2x the
   published 128-MiB slot -- two authorities for one quantity, one of them
   documented and dead.  That is the second-bookkeeping class under the
   upstream-minimal law, where the default is DELETE, not wire.  Routed to
   seat 6 to pick ONE authority and correct whichever text lies.

Entries 2 and 3 LEAVE the debt the moment seat 6 rules on them, and the
pinned length is what makes that visible.  Neither file is edited here
(both are another seat's) -- the ownership law's answer to a wall in a
foreign file is file:line plus a report.

The two bounce legs
(``weight_exchange_bounce.run_bounce_leg`` / ``run_agreed_leg``) and
``refuse_if_plan_exceeds_slot`` are NOT in the debt: they HAVE production
call sites (``weight_updater.py:2370`` / ``:2394``) whose own caller chain
is cut at ``_weg2_xchg_inject_from_peer``, which refuses with W4 before
delegating and says so in its docstring ("NOT YET REACHABLE ON THE METAL,
and it says so rather than pretending").  A chain cut by a NAMED REFUSAL
is a different state from no chain at all, and this file distinguishes the
two structurally -- see :func:`_wired_owners`.

RED-FIRST, both directions, with the SHAs:

* ``TestNoOrphanVerdict.test_every_written_verdict_has_a_reachable_reader``
  is RED at ``692b1e8698`` (a5eb06cdaa's parent): ``_BOOT_VOTE`` is written
  by a reachable ``_record_boot_vote`` and read only by an unreachable
  ``boot_vote``.  GREEN at ``dfceb7004e``.
* ``TestEveryWCodeRaiserHasAProductionCaller`` is RED against a PLANTED
  raise site with no caller and GREEN again once it is removed
  (:class:`TestTheRatchetCanFail`, which runs the same analyser over a
  synthetic three-module tree in a temp dir -- so the ratchet's ability to
  fail is proven inside the suite rather than asserted in a docstring).

RESOLUTION, and its honest bound.  Edges are NAME-BASED over every
non-test module under ``python/sglang/srt`` (a reference to a name defined
in the scope is an edge), and a REFERENCE counts, not only a ``Call``:
this codebase wires its RPC handlers by handing bound methods to a
dispatch table (``scheduler.py:2759`` registers
``self.weight_updater.resume_memory_occupation`` as a VALUE), so a
call-only graph reports every RPC handler as dead.  Name-based resolution
OVER-approximates reachability, which makes this ratchet CONSERVATIVE: it
can call an unwired function wired, never the reverse.  That is the right
direction for a ratchet whose failures must be real, and it is why the
four debt entries are meaningful -- they survive the over-approximation.
"""

import ast
import os
import tempfile
import textwrap
import unittest
from collections import defaultdict

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

from sglang.test.test_utils import CustomTestCase

#: ``python/sglang/srt`` of the tree this test file belongs to, found from
#: this file's own path so a worktree/checkout copy analyses ITSELF and never
#: an installed sglang somewhere else on the box.
SRT = os.path.normpath(
    os.path.join(os.path.dirname(os.path.abspath(__file__)),
                 "..", "..", "..", "..", "python", "sglang", "srt")
)

#: THE EXCHANGE LANE (the files the S6 campaign owns), relative to ``SRT``.
#: Ownership boundaries are named in OPERATOR_HANDOVER_S6_0911.md section 2;
#: this tuple is the union of both desk seats' file lists plus the two
#: launcher/ledger modules the arm layer's refusals live in.
LANE_FILES = (
    "weg2/weight_exchange.py",
    "weg2/weight_exchange_bounce.py",
    "weg2/weight_exchange_region.py",
    "weg2/weight_exchange_shadow.py",
    "weg2/weight_exchange_transport.py",
    "weg2/xchg_bounce.py",
    "weg2/xchg_census.py",
    "weg2/xchg_residency.py",
    "weg2/host_ledger.py",
    "weg2/ring_table.py",
    "weg2/launcher.py",
    "managers/scheduler_components/weight_updater.py",
)

#: The production ENTRIES reachability is measured from.  ``cli`` is the
#: launcher's real process entry and ``main`` is what it calls (the XSN13
#: traceback reads ``launcher.py:10283 <module> -> :10249 cli -> :9263
#: main``); ``main`` is listed beside it because it is the one every
#: launcher test and every record names, and a chain that starts there must
#: not depend on ``cli`` still existing.  ``run_scheduler_process`` is every
#: rank's entry.  ``cli`` is not redundant: it owns the post-spawn refusal
#: funnel and is the ONLY reader of ``_ACTIVE_BOOT_STATE``, so without it
#: that verdict slot reads as an orphan while the boot depends on it.
PRODUCTION_ENTRIES = (
    ("weg2/launcher.py", "cli"),
    ("weg2/launcher.py", "main"),
    ("managers/scheduler.py", "run_scheduler_process"),
)

#: The four debt CLASSES the operator's condition of 2026-09-11 requires.
#: A class is not a label for its own sake: it says WHO owes WHAT next, and
#: ``UNKNOWN`` is the one that must never sit here quietly -- an unclassified
#: entry is an exemption with extra steps.
DEBT_CLASSES = (
    "UNWIRED-BY-DESIGN-AND-PINNED",   # a test asserts it stays unwired
    "DECLARED-TODO",                  # owed work with a name and a pin
    "SECOND-AUTHORITY-CANDIDATE",     # upstream-minimal law: delete, not wire
    "UNKNOWN",                        # unclassified: triage, never ship here
)

#: THE FROZEN DEBT: every ``raise Weg2*`` site in the lane whose owner has NO
#: production reference at all on the shipping tip, with file:line (as of
#: ``2c9592fd19``; informational, the tests use the CURRENT line), one line
#: of reason, and its CLASS.  Argued in full in this module's docstring.
#: Closed in BOTH directions by the tests below and its length is pinned, so
#: it can only ever SHRINK -- it already has: B4i wired
#: ``xchg_form_dormant_reserve`` and that entry was deleted here.
UNWIRED_DEBT = (
    {
        "file": "weg2/weight_exchange.py",
        "fn": "refuse_if_not_ok",
        "code": "Weg2XchgCoverageRefused",
        "line": 2513,
        "class": "UNWIRED-BY-DESIGN-AND-PINNED",
        "why": "no group fence at the end of weight loading; "
               "test_weg2_coverage_verdict_1273.py::test_refuse_if_not_ok_stays_unwired "
               "asserts it stays unwired, and TestNoOrphanVerdict covers its verdict",
    },
    {
        "file": "weg2/weight_exchange_transport.py",
        "fn": "arm_oncard_lane",
        "code": "Weg2XchgOnCardUnavailable",
        "line": 2325,
        "class": "DECLARED-TODO",
        "why": "the MODE is decided and wired (launcher.py:3512 publishes "
               "env[ENV_ONCARD_MODE] from argv); the missing half is S6's "
               "probe-and-degrade arm, declared by TODO(S6) here, pinned by "
               "test_weg2_xchg_transport_1273.py:1995 and stated in prose at "
               "weight_exchange_shadow.py:32 -- owed S6 work, not a hidden defect",
    },
    {
        "file": "weg2/weight_exchange_transport.py",
        "fn": "diagonal_carrier_bytes",
        "code": "Weg2XchgOncardSlotRefused",
        "line": 2410,
        "class": "SECOND-AUTHORITY-CANDIDATE",
        "why": "#1334. The W82 guard itself is LIVE (reachable raisers at :198 "
               "and :204 in validate_oncard_slot_mib, called from launcher.py:3518); "
               "this helper is a SECOND authority for the carrier size -- :2453's "
               "docstring names it as the sizer while #1334 sized the deposit "
               "inline as 2x the published slot. Upstream-minimal law: seat 6 "
               "picks one authority, default DELETE",
    },
)

#: Pinned so the debt cannot grow by accident.  4 on dfceb7004e, 3 on
#: 2c9592fd19 after B4i wired the reserve instrument.
DEBT_SIZE_ON_THE_LINE = 3


# --------------------------------------------------------------------------
# The analyser.  One AST pass per file; every Name/Attribute LOAD and every
# ``raise`` is attributed to the innermost enclosing function by the visitor's
# own stack, so no line-containment search is needed.
# --------------------------------------------------------------------------

class _Index:
    """Name-resolved reference graph over one source tree."""

    def __init__(self) -> None:
        self.defs: dict[str, tuple[str, str, int]] = {}   # qual -> (rel, fn, lineno)
        self.by_name: dict[str, set[str]] = defaultdict(set)
        self.classes: dict[str, set[str]] = defaultdict(set)
        self.refs: dict[str, set[str]] = defaultdict(set)  # qual -> referenced names
        #: ``(rel, lineno, exc_name, owner_qual)`` for every lane raise site.
        self.raises: list[tuple[str, int, str, str | None]] = []
        #: module-level names written under ``global`` -> {writer qual}
        self.global_writers: dict[tuple[str, str], set[str]] = defaultdict(set)
        #: module-level names read -> {reader qual}
        self.global_readers: dict[tuple[str, str], set[str]] = defaultdict(set)
        #: every module-level assignment target, per module
        self.module_globals: dict[str, set[str]] = defaultdict(set)


def _module_qual(rel: str, scope: list[str], name: str) -> str:
    return "::".join([rel, *scope, name]) if scope else f"{rel}::{name}"


def _py_files(root: str) -> list[tuple[str, str]]:
    out = []
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = sorted(d for d in dirnames if d != "__pycache__")
        for fn in sorted(filenames):
            if not fn.endswith(".py"):
                continue
            path = os.path.join(dirpath, fn)
            rel = os.path.relpath(path, root).replace(os.sep, "/")
            # A test module is not production wiring (#1001: a name that only
            # a test calls is not reachable from a boot).
            if rel.startswith("test") or "/test" in rel:
                continue
            out.append((path, rel))
    return out


def build_index(root: str, lane_files: tuple[str, ...]) -> _Index:
    """Index every non-test module under ``root``; collect raise sites for
    the members of ``lane_files`` only."""
    idx = _Index()
    lane = set(lane_files)
    for path, rel in _py_files(root):
        try:
            # utf-8-SIG, not utf-8: one upstream module in this tree carries a
            # BOM (``models/qwen2_classification.py``, U+FEFF at line 1) and
            # plain utf-8 makes ``ast.parse`` reject it as an invalid
            # non-printable character -- which would have entered the scope as
            # an unparsed module, i.e. a hole in the reachability graph that
            # reads exactly like "nothing there".
            with open(path, encoding="utf-8-sig") as fh:
                tree = ast.parse(fh.read())
        except (SyntaxError, UnicodeDecodeError, OSError):
            # A module this analyser cannot parse is REPORTED as a gap by
            # `test_the_scope_parsed`, never silently dropped.
            idx.refs[f"{rel}::<unparsed>"].add("")
            continue
        _IndexVisitor(idx, rel, rel in lane).visit(tree)
    return idx


class _IndexVisitor(ast.NodeVisitor):

    def __init__(self, idx: _Index, rel: str, is_lane: bool) -> None:
        self.idx = idx
        self.rel = rel
        self.is_lane = is_lane
        self.scope: list[str] = []
        self.fn_stack: list[str] = []
        self.global_here: list[set[str]] = []

    # -- scopes ------------------------------------------------------------
    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        self.idx.classes[node.name].add(_module_qual(self.rel, self.scope, node.name))
        self.scope.append(node.name)
        self.generic_visit(node)
        self.scope.pop()

    def _function(self, node) -> None:
        qual = _module_qual(self.rel, self.scope, node.name)
        self.idx.defs[qual] = (self.rel, node.name, node.lineno)
        self.idx.by_name[node.name].add(qual)
        self.scope.append(node.name)
        self.fn_stack.append(qual)
        self.global_here.append(set())
        self.generic_visit(node)
        self.global_here.pop()
        self.fn_stack.pop()
        self.scope.pop()

    visit_FunctionDef = _function
    visit_AsyncFunctionDef = _function

    def _owner(self) -> str | None:
        return self.fn_stack[-1] if self.fn_stack else None

    # -- module-level state ------------------------------------------------
    def visit_Global(self, node: ast.Global) -> None:
        if self.global_here:
            self.global_here[-1].update(node.names)
        self.generic_visit(node)

    def visit_Assign(self, node: ast.Assign) -> None:
        names = [t.id for t in node.targets if isinstance(t, ast.Name)]
        if not self.fn_stack:
            self.idx.module_globals[self.rel].update(names)
        else:
            declared = self.global_here[-1] if self.global_here else set()
            owner = self._owner()
            for nm in names:
                if nm in declared and owner is not None:
                    self.idx.global_writers[(self.rel, nm)].add(owner)
        self.generic_visit(node)

    # -- edges -------------------------------------------------------------
    def visit_Name(self, node: ast.Name) -> None:
        if isinstance(node.ctx, ast.Load):
            owner = self._owner()
            if owner is not None:
                self.idx.refs[owner].add(node.id)
                self.idx.global_readers[(self.rel, node.id)].add(owner)
        self.generic_visit(node)

    def visit_Attribute(self, node: ast.Attribute) -> None:
        if isinstance(node.ctx, ast.Load):
            owner = self._owner()
            if owner is not None:
                self.idx.refs[owner].add(node.attr)
        self.generic_visit(node)

    # -- raises ------------------------------------------------------------
    def visit_Raise(self, node: ast.Raise) -> None:
        if self.is_lane and node.exc is not None:
            exc = node.exc
            func = exc.func if isinstance(exc, ast.Call) else exc
            name = func.id if isinstance(func, ast.Name) else getattr(func, "attr", None)
            if name and name.startswith("Weg2"):
                self.idx.raises.append((self.rel, node.lineno, name, self._owner()))
        self.generic_visit(node)


def reachable(idx: _Index, entries: tuple[tuple[str, str], ...]) -> set[str]:
    """Transitive closure of reference edges from ``entries``.

    A class name resolves to that class's ``__init__`` as well, so
    ``Scheduler(...)`` reaches ``Scheduler.__init__`` and everything it
    wires (which is where the RPC dispatch table is built).
    """
    seeds = [q for rel, fn in entries
             for q in idx.by_name.get(fn, ()) if idx.defs.get(q, ("",))[0] == rel]
    seen: set[str] = set()
    stack = list(seeds)
    while stack:
        qual = stack.pop()
        if qual in seen:
            continue
        seen.add(qual)
        for name in idx.refs.get(qual, ()):
            for target in idx.by_name.get(name, ()):
                if target not in seen:
                    stack.append(target)
            for cls in idx.classes.get(name, ()):
                init = f"{cls}::__init__"
                if init in idx.defs and init not in seen:
                    stack.append(init)
    return seen


def _referenced_anywhere(idx: _Index, qual: str) -> bool:
    """Is this function NAMED by any production function other than itself?

    The #859 delivery criterion: code presence at file:line PLUS a referencer
    outside its own definition.  ``False`` is the W84 state -- nothing in
    production so much as spells the name.
    """
    _rel, fn, _lineno = idx.defs[qual]
    for owner, names in idx.refs.items():
        if owner != qual and fn in names:
            return True
    return False


def _wired_owners(idx: _Index, reach: set[str]) -> set[str]:
    """Owners whose raise sites count as WIRED.

    Two ways, and the second is why this is a function and not ``in reach``:

    * the owner is reachable from a production entry, or
    * the owner is NAMED by some production function -- its chain exists but
      is cut higher up (``_weg2_xchg_inject_from_peer`` refuses with W4
      before delegating to the two bounce legs).  A chain cut by a named
      refusal is a boot that stops with a reason; no chain at all is the
      W84 state.
    """
    wired = set(reach)
    for qual in idx.defs:
        if qual not in wired and _referenced_anywhere(idx, qual):
            wired.add(qual)
    return wired


def unwired_raise_sites(idx: _Index, entries: tuple[tuple[str, str], ...]):
    """``[(rel, owner_fn, exc_name, lineno), ...]``, sorted, for every lane
    raise site whose owner is neither reachable nor named in production."""
    reach = reachable(idx, entries)
    wired = _wired_owners(idx, reach)
    out = []
    for rel, lineno, exc, owner in idx.raises:
        if owner is None or owner in wired:
            continue
        out.append((rel, idx.defs[owner][1], exc, lineno))
    return sorted(set(out))


def orphan_verdicts(idx: _Index, entries: tuple[tuple[str, str], ...],
                    lane_files: tuple[str, ...]):
    """``[(rel, global_name, writers, readers), ...]`` for every lane module
    global that a REACHABLE function writes and NO reachable function reads.

    This is the W84 shape with no W-code in it: a verdict recorded by a live
    path whose readers are all dead is a value computed, stored and acted on
    by nobody -- the class this fork keeps a semgrep rule for.
    """
    reach = reachable(idx, entries)
    out = []
    for (rel, name), writers in sorted(idx.global_writers.items()):
        if rel not in set(lane_files):
            continue
        live_writers = sorted(w for w in writers if w in reach)
        if not live_writers:
            continue          # nothing records it on a live path -> no debt
        readers = {r for r in idx.global_readers.get((rel, name), ()) if r not in writers}
        live_readers = sorted(r for r in readers if r in reach)
        if not live_readers:
            out.append((rel, name, live_writers, sorted(readers)))
    return out


# --------------------------------------------------------------------------
# One index for the whole module (the walk costs ~10 s; both ratchets and
# the scope check share it).
# --------------------------------------------------------------------------

_INDEX: _Index | None = None


def _index() -> _Index:
    global _INDEX
    if _INDEX is None:
        _INDEX = build_index(SRT, LANE_FILES)
    return _INDEX


class TestTheScopeIsWhatItClaims(CustomTestCase):
    """A ratchet over an empty or half-parsed scope passes vacuously."""

    def test_every_lane_file_exists_and_was_indexed(self):
        idx = _index()
        for rel in LANE_FILES:
            self.assertTrue(os.path.isfile(os.path.join(SRT, rel)),
                            f"lane file {rel} is gone -- update LANE_FILES")
            self.assertTrue(
                any(q.startswith(rel + "::") for q in idx.defs),
                f"lane file {rel} contributed no definitions: not indexed",
            )

    def test_the_scope_parsed(self):
        idx = _index()
        unparsed = sorted(q for q in idx.refs if q.endswith("::<unparsed>"))
        self.assertEqual(unparsed, [], f"modules the analyser could not parse: {unparsed}")

    def test_the_entries_resolve(self):
        idx = _index()
        for rel, fn in PRODUCTION_ENTRIES:
            self.assertTrue(
                any(idx.defs[q][0] == rel for q in idx.by_name.get(fn, ())),
                f"production entry {rel}::{fn} no longer exists",
            )

    def test_reachability_is_not_vacuous(self):
        """INDIKATOR-GESETZ: an instrument is a finding only once it is shown
        to measure what it claims.  With an empty or broken entry set every
        ratchet here goes VACUOUSLY green (no reachable writer -> no orphan
        obligation), so the closure's size and four LANDMARKS a boot
        provably runs are pinned.  ``coverage_leg_decision`` is the landmark
        B4g added and is the whole point of :class:`TestNoOrphanVerdict`."""
        idx = _index()
        reach = reachable(idx, PRODUCTION_ENTRIES)
        self.assertGreater(len(reach), 10000,
                           f"reference closure collapsed to {len(reach)} -- "
                           "the entry set or the edge builder is broken, and "
                           "every ratchet below would pass vacuously")
        for rel, fn in (
            ("managers/scheduler_components/weight_updater.py", "resume_memory_occupation"),
            ("managers/scheduler_components/weight_updater.py", "_weg2_wake_reload_weights"),
            ("weg2/weight_exchange.py", "arm_coverage_at_load"),
            ("weg2/weight_exchange.py", "coverage_leg_decision"),
        ):
            quals = [q for q in idx.by_name.get(fn, ()) if idx.defs[q][0] == rel]
            self.assertTrue(quals, f"landmark {rel}::{fn} no longer exists")
            self.assertTrue(
                any(q in reach for q in quals),
                f"landmark {rel}::{fn} is not reachable from the production "
                "entries -- the graph lost the RPC/dispatch wiring",
            )

    def test_the_lane_actually_raises_w_codes(self):
        idx = _index()
        self.assertGreater(len(idx.raises), 100,
                           "the lane's Weg2* raise population collapsed -- "
                           "either the lane moved or the raise scan broke")


class TestEveryWCodeRaiserHasAProductionCaller(CustomTestCase):
    """RATCHET 1: the plain reachability question at every raise site, with
    the failures carried as a frozen, named, shrinking debt."""

    @staticmethod
    def _debt_keys():
        return {(e["file"], e["fn"], e["code"]) for e in UNWIRED_DEBT}

    def test_no_new_unwired_raiser(self):
        """THE RATCHET.  A raise site whose owner nothing in production even
        names is the W84 state; a NEW one fails here by name."""
        found = unwired_raise_sites(_index(), PRODUCTION_ENTRIES)
        known = self._debt_keys()
        new = sorted(site for site in found if (site[0], site[1], site[2]) not in known)
        self.assertEqual(
            new, [],
            "W-code raise site(s) with NO production caller and not in "
            f"UNWIRED_DEBT: {new} -- either wire the raiser (preferred: this "
            "is the W84 class, a guard that cannot fire) or add it to the "
            "debt WITH its class and one line of reason (see DEBT_CLASSES "
            "and this module's docstring)",
        )

    def test_the_debt_is_not_stale(self):
        """The GONE direction.  A debt entry that became wired must be
        DELETED, so the debt can only shrink; leaving it would let a second
        regression hide behind a paid-off entry.

        This has already fired for real, which is the proof it has teeth:
        ``launcher.xchg_form_dormant_reserve`` was entry 2 on
        ``dfceb7004e`` and B4i (``2c9592fd19``) wired it, so rebasing this
        file onto that tip turned this test RED and the entry was deleted.
        """
        found = {(rel, fn, exc) for rel, fn, exc, _ln in
                 unwired_raise_sites(_index(), PRODUCTION_ENTRIES)}
        stale = sorted(site for site in self._debt_keys() if site not in found)
        self.assertEqual(
            stale, [],
            f"UNWIRED_DEBT entries that are now wired (or gone): {stale} -- "
            "delete them from the tuple; the debt only shrinks",
        )

    def test_the_debt_cannot_grow_silently(self):
        self.assertEqual(len(UNWIRED_DEBT), DEBT_SIZE_ON_THE_LINE)

    def test_every_debt_entry_carries_its_class_and_reason(self):
        """The operator's condition of 2026-09-11: "a debt whose entries are
        classified is a work list; an unclassified one is an exemption with
        extra steps."  So the shape is enforced, not trusted -- and
        ``UNKNOWN`` is rejected outright, because an entry nobody has
        classified is triage that has not happened yet."""
        for entry in UNWIRED_DEBT:
            self.assertEqual(sorted(entry), ["class", "code", "file", "fn", "line", "why"],
                             f"debt entry has the wrong fields: {entry}")
            self.assertIn(entry["class"], DEBT_CLASSES, f"unknown class: {entry}")
            self.assertNotEqual(
                entry["class"], "UNKNOWN",
                f"{entry['file']}::{entry['fn']} is unclassified -- classify it "
                "(who owes what next) or wire the raiser",
            )
            self.assertGreater(len(entry["why"]), 40,
                               f"one line of REASON, please: {entry}")
            self.assertIsInstance(entry["line"], int)

    def test_every_debt_entry_still_names_a_real_function(self):
        """A debt entry pointing at a function that no longer exists is not a
        debt, it is a stale comment -- and it would mask a NEW unwired raiser
        of the same name somewhere else."""
        idx = _index()
        for entry in UNWIRED_DEBT:
            quals = [q for q in idx.by_name.get(entry["fn"], ())
                     if idx.defs[q][0] == entry["file"]]
            self.assertEqual(len(quals), 1,
                             f"{entry['file']}::{entry['fn']} is gone or duplicated")

    def test_the_two_bounce_legs_are_wired_not_debt(self):
        """Named on purpose: ``run_bounce_leg`` / ``run_agreed_leg`` are the
        S6 product's two executors and they are NOT in the debt, because
        ``weight_updater.py`` calls them -- their chain is cut at
        ``_weg2_xchg_inject_from_peer``'s W4 refusal, which is a boot that
        stops with a reason.  If that call site ever disappears they become
        debt, and this test says so before the ratchet's totals do."""
        idx = _index()
        for fn in ("run_bounce_leg", "run_agreed_leg"):
            quals = [q for q in idx.by_name.get(fn, ())
                     if idx.defs[q][0] == "weg2/weight_exchange_bounce.py"]
            self.assertEqual(len(quals), 1, f"{fn} moved or vanished")
            self.assertTrue(
                _referenced_anywhere(idx, quals[0]),
                f"{fn} lost its production call site -- it is now the W84 state",
            )


class TestNoOrphanVerdict(CustomTestCase):
    """RATCHET 2: the W84 form itself, EMPTY allowlist, nothing to keep
    current.

    RED at ``692b1e8698``: ``_BOOT_VOTE`` (``weight_exchange.py``) is
    written by ``_record_boot_vote`` -- reachable, the arming call runs at
    the end of every rank's weight load -- and read only by ``boot_vote``,
    which had zero production callers there.  GREEN at ``dfceb7004e``,
    where ``coverage_leg_decision`` reads it and the wake fence calls that
    (``weight_updater.py:1270``).
    """

    def test_every_written_verdict_has_a_reachable_reader(self):
        orphans = orphan_verdicts(_index(), PRODUCTION_ENTRIES, LANE_FILES)
        pretty = [f"{rel}::{name} written by {w} read by {r or 'NOBODY'}"
                  for rel, name, w, r in orphans]
        self.assertEqual(
            pretty, [],
            "lane verdict slot(s) written on a live path and read by no "
            f"reachable function: {pretty} -- this is the W84 class "
            "(computed, recorded, acted on by nobody)",
        )

    def test_the_w84_vote_slot_is_the_one_this_ratchet_was_built_from(self):
        """Names the historical instance directly, so a refactor that keeps
        the orphan count at zero by DELETING the vote (rather than by
        keeping its reader) is visible."""
        idx = _index()
        writers = idx.global_writers.get(("weg2/weight_exchange.py", "_BOOT_VOTE"), set())
        self.assertTrue(writers, "_BOOT_VOTE is no longer written: the coverage "
                                 "vote was removed, not wired -- re-derive B4g")
        readers = idx.global_readers.get(("weg2/weight_exchange.py", "_BOOT_VOTE"), set())
        reach = reachable(idx, PRODUCTION_ENTRIES)
        live = sorted(r for r in readers if r not in writers and r in reach)
        self.assertTrue(live, "_BOOT_VOTE has no reachable reader (the exact "
                              "692b1e8698 state B4g closed)")


# --------------------------------------------------------------------------
# CAN-FAIL: the analyser proves its own failure modes on a synthetic tree.
# --------------------------------------------------------------------------

_LANE_SRC_WIRED = '''
class Weg2Refused(RuntimeError):
    pass


def leaf(x):
    if x:
        raise Weg2Refused("wall")
    return x
'''

_LANE_SRC_PLANTED = _LANE_SRC_WIRED + '''

def planted(x):
    """Nothing names this function anywhere."""
    raise Weg2Refused("planted wall with no caller")
'''

_ENTRY_SRC = '''
from lane import leaf


def main():
    return leaf(1)
'''

_VOTE_SRC_ORPHAN = '''
_VOTE = None


def record(v):
    global _VOTE
    _VOTE = v


def read_vote():
    return _VOTE
'''

_VOTE_SRC_WIRED = _VOTE_SRC_ORPHAN + '''

def decide():
    return read_vote()
'''

_ENTRY_SRC_WITH_VOTE = '''
from lane import leaf
from vote import record


def main():
    record(leaf(1))
    return 1
'''

_ENTRY_SRC_WITH_DECIDE = '''
from lane import leaf
from vote import record, decide


def main():
    record(leaf(1))
    return decide()
'''


class TestTheRatchetCanFail(CustomTestCase):
    """Desk-written-never-executed law: a gate with no proof that it CAN
    fail is not a gate.  Both ratchets are run over a three-module
    synthetic tree in a temp dir, once in the passing and once in the
    failing shape."""

    ENTRIES = (("entry.py", "main"),)
    LANE = ("lane.py", "vote.py")

    def _tree(self, lane_src: str, entry_src: str, vote_src: str) -> str:
        root = tempfile.mkdtemp(prefix="b4l-plant-")
        for name, src in (("lane.py", lane_src), ("entry.py", entry_src),
                          ("vote.py", vote_src)):
            with open(os.path.join(root, name), "w", encoding="utf-8") as fh:
                fh.write(textwrap.dedent(src))
        return root

    def test_a_planted_raise_with_no_caller_is_RED_and_its_removal_GREEN(self):
        green = self._tree(_LANE_SRC_WIRED, _ENTRY_SRC, _VOTE_SRC_WIRED)
        idx = build_index(green, self.LANE)
        self.assertEqual(unwired_raise_sites(idx, self.ENTRIES), [],
                         "the baseline synthetic tree must be GREEN")

        red = self._tree(_LANE_SRC_PLANTED, _ENTRY_SRC, _VOTE_SRC_WIRED)
        idx = build_index(red, self.LANE)
        found = unwired_raise_sites(idx, self.ENTRIES)
        self.assertEqual([(r, f, e) for r, f, e, _ln in found],
                         [("lane.py", "planted", "Weg2Refused")],
                         "the planted caller-less raise must be the one finding")

    def test_an_orphan_verdict_is_RED_and_its_reader_GREEN(self):
        red = self._tree(_LANE_SRC_WIRED, _ENTRY_SRC_WITH_VOTE, _VOTE_SRC_ORPHAN)
        idx = build_index(red, self.LANE)
        orphans = orphan_verdicts(idx, self.ENTRIES, self.LANE)
        self.assertEqual([(rel, name) for rel, name, _w, _r in orphans],
                         [("vote.py", "_VOTE")],
                         "a recorded verdict with no reachable reader must be RED")

        green = self._tree(_LANE_SRC_WIRED, _ENTRY_SRC_WITH_DECIDE, _VOTE_SRC_WIRED)
        idx = build_index(green, self.LANE)
        self.assertEqual(orphan_verdicts(idx, self.ENTRIES, self.LANE), [],
                         "once a reachable function reads the verdict it is GREEN")

    def test_a_reference_without_a_call_counts_as_wiring(self):
        """The dispatch-table fact (``scheduler.py:2759`` hands
        ``resume_memory_occupation`` to a registry as a VALUE).  A
        call-only graph reports every RPC handler as dead, so this is
        pinned rather than left to the reader."""
        root = self._tree(_LANE_SRC_WIRED,
                          "from lane import leaf\n\n"
                          "def main():\n"
                          "    table = {'x': leaf}\n"
                          "    return table\n",
                          _VOTE_SRC_WIRED)
        idx = build_index(root, self.LANE)
        self.assertEqual(unwired_raise_sites(idx, self.ENTRIES), [],
                         "a bare reference (no call) must count as an edge")


if __name__ == "__main__":
    unittest.main()
