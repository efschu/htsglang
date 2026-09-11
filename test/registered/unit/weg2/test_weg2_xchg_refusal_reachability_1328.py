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
reported rather than papered over.  An empty allowlist on THIS assertion
cannot be green on ``dfceb7004e``, and that is a measurement, not a
preference.  Four raise sites on the tip have no production caller at all:

1. ``weight_exchange.refuse_if_not_ok`` (W84) -- unwired BY DESIGN, and an
   EXISTING test pins it that way
   (``test_weg2_coverage_verdict_1273.py::test_refuse_if_not_ok_stays_unwired``).
   Its own docstring states the reason: at the end of weight loading there
   is no group fence, and a rank that raised there would die while the
   other five walked into a collective with five members.  Requiring a
   caller here would demand the opposite of a decision already taken and
   tested.  The VERDICT it hands over is what must be reachable, and that
   is :class:`TestNoOrphanVerdict`'s question.
2. ``launcher.xchg_form_dormant_reserve`` (W71) -- has no caller ANYWHERE
   in production (``grep -rn`` over ``python/sglang/srt``: one hit, its own
   ``def``).  Its docstring calls itself "the pricing and REPORTING
   instrument that makes the refusal auditable"; an instrument nobody
   calls reports nothing, which is why boot weg2xsn15 emitted
   ``WEG2-XCHG-RESERVE`` ZERO times (operator handover section 12,
   "Launcher-Instrument-Defekt").  Owned by desk seat 6 as B4i part 2;
   named here so the fix is visible to this ratchet the moment it lands.
3. ``weight_exchange_transport.arm_oncard_lane`` (W72/W? ``Weg2XchgOnCardUnavailable``)
   -- docstring: "Decide the on-card lane's mode ONCE, at the launcher".
   The launcher does not call it (zero references outside its own ``def``).
4. ``weight_exchange_transport.diagonal_carrier_bytes``
   (``Weg2XchgOncardSlotRefused``, #1334) -- same: zero references.

3 and 4 are NEW FINDINGS of this ratchet, desk-proven and reported to the
operator with these file:lines; they are NOT fixed here (both files are
another seat's).  The two bounce legs
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

#: THE FROZEN DEBT: ``(relpath, owner function, W-code class)`` for every
#: ``raise Weg2*`` site in the lane whose owner has NO production reference
#: at all on ``dfceb7004e``.  Each entry is argued in this module's
#: docstring.  Closed in BOTH directions by the tests below, and its length
#: is pinned, so it can only ever SHRINK.
UNWIRED_DEBT = (
    ("weg2/weight_exchange.py", "refuse_if_not_ok", "Weg2XchgCoverageRefused"),
    ("weg2/launcher.py", "xchg_form_dormant_reserve", "Weg2XchgResidencyUnarmable"),
    ("weg2/weight_exchange_transport.py", "arm_oncard_lane", "Weg2XchgOnCardUnavailable"),
    ("weg2/weight_exchange_transport.py", "diagonal_carrier_bytes", "Weg2XchgOncardSlotRefused"),
)

DEBT_SIZE_ON_DFCEB7004E = 4


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

    def test_no_new_unwired_raiser(self):
        """THE RATCHET.  A raise site whose owner nothing in production even
        names is the W84 state; a NEW one fails here by name."""
        found = unwired_raise_sites(_index(), PRODUCTION_ENTRIES)
        known = {(rel, fn, exc) for rel, fn, exc in UNWIRED_DEBT}
        new = sorted(site for site in found if (site[0], site[1], site[2]) not in known)
        self.assertEqual(
            new, [],
            "W-code raise site(s) with NO production caller and not in "
            f"UNWIRED_DEBT: {new} -- either wire the raiser (preferred: this "
            "is the W84 class, a guard that cannot fire) or add it to the "
            "debt WITH its argument in this module's docstring",
        )

    def test_the_debt_is_not_stale(self):
        """The GONE direction.  A debt entry that became wired must be
        DELETED, so the debt can only shrink; leaving it would let a second
        regression hide behind a paid-off entry."""
        found = {(rel, fn, exc) for rel, fn, exc, _ln in
                 unwired_raise_sites(_index(), PRODUCTION_ENTRIES)}
        stale = sorted(site for site in
                       {(rel, fn, exc) for rel, fn, exc in UNWIRED_DEBT}
                       if site not in found)
        self.assertEqual(
            stale, [],
            f"UNWIRED_DEBT entries that are now wired (or gone): {stale} -- "
            "delete them from the tuple; the debt only shrinks",
        )

    def test_the_debt_cannot_grow_silently(self):
        self.assertEqual(len(UNWIRED_DEBT), DEBT_SIZE_ON_DFCEB7004E)

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
