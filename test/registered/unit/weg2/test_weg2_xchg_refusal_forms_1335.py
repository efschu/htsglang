# SPDX-License-Identifier: Apache-2.0
"""#1335 (B4r) -- a refusal does not have to be a ``raise`` to be a refusal.

WHY THIS FILE EXISTS, measured on boots XSN16/17/18/19.  ``slack_mib`` reads
NEGATIVE on ``weights_draft`` -- -107.7 / -115.4 / -115.2 / -110.9 -- with
``uncovered=0 short=0 missing=0`` and ``tms_answered=yes``, and **nothing gates
it**.  The root is NOT arithmetic: the refusal that would catch it ALREADY
EXISTS and is unreachable.

THE MEASURED ROOT, so this file's subject is not a guess:

* ``slack_bytes`` (``weight_exchange.py``) is
  ``tms_bytes - planned_bytes - buffers_bytes``, and it mixes TWO AUTHORITIES
  for one fact -- "which tag do these bytes belong to".  ``planned``/``buffers``
  come from ``walk_live_tensors``' region+name attribution; ``tms_bytes`` is the
  SAVER's own per-tag ledger (``model_runner._weg2_xchg_tag_bytes`` ->
  ``memory_saver_adapter.tag_bytes``).
* GLOBALLY the two do NOT disagree: summed over every weights-family tag the
  saver is LARGER on every rank (+72.9 MiB on P rank 0, +188.9 / +169.0 / +184.5
  on D ranks 0/1/2 of weg2xsn19), which is the expected allocator-overhang
  direction.  No bytes are missing -- the negative slack is a MISATTRIBUTION
  BETWEEN TAGS.  Cross-check that fixes the mechanism as attribution and not as
  measurement: ``weights_0`` carries one extra ~64 MiB buffer (21 buffers /
  64.2 MiB against 20 / 0.2 MiB on every other chunk tag) and THERE the saver
  agrees (``tms_mib`` 614.0 against 550.0).
* The instrument that DOES run, ``plan_param_lines``, cannot see it: its
  ``verdict`` is ``here``/``elsewhere``/``absent``, i.e. the WALK's tag against
  the PLAN's tag -- both sides the same authority.  Hence 5739 of 5739 lines
  read ``verdict=here`` next to a -115.4 MiB deficit, and the docstring's named
  hypothesis (a tied embedding counted into the draft plan while living in the
  target's tag) is REFUTED by that census plus ``missing=0``.
* ``weight_exchange_region.gate0_check`` carries exactly the missing
  comparison, per tag: ``census < planned`` -> ``[tag=... planned=... >
  tms_tag_bytes=... delta=...]`` -> ``write_matrix_verdict(row, False)``, i.e. a
  GROUP-WIDE refusal, with ``tms_tag_bytes=ABSENT`` handled as its own case
  rather than as a zero.  **It has zero production references.**

AND THAT IS THE RATCHET'S SUBJECT, because the ratchet built to make this class
impossible cannot see this instance.  B4l's
``test_weg2_xchg_refusal_reachability_1328.py`` matches a raise whose raised
expression's callee NAME starts with ``Weg2``.  ``gate0_check`` raises
``region._refuse(Weg2XchgPlanDisagree, ...)`` -- the callee is ``_refuse`` --
and its tag-mismatch path does not raise AT ALL, it VOTES.  Measured here, and
the correlation is one-to-one rather than argued: the zero-reference
refusal-bearing owners of this lane are exactly THREE, the two B4l's frozen debt
names are exactly the two whose only form is a bare ``raise Weg2*``, and the one
it misses is exactly the one whose forms are the two its predicate cannot match.
Eight refusal sites of this lane sit outside that predicate (7 via the helper,
1 as a vote); seven have wired owners.

WHAT THIS FILE DOES NOT DO, stated so it is not mistaken for done: it does NOT
wire ``gate0_check``.  Wiring Gate 0 means publishing the 6x6 matrix row and
waiting on it, i.e. ADDING COLLECTIVES TO THE WAKE SEAM, whose danger direction
is already named in the plan of record (the ``:1367-1375`` re-entry HANG).  That
is a design call for the operator, and the plan's LAWS put it before the build,
not after.  Until it is taken, this ratchet is what keeps the gap from being
lost again -- and it is deliberately NOT a second copy of B4l's reachability
engine: that one asks transitive reachability from entry points, this one asks
the narrower and cheaper question "does anything in the lane reference this
owner at all", whose zero is the strictly stronger statement.
"""

from __future__ import annotations

import ast
import collections
import os
import pathlib

import pytest

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

from sglang.srt.weg2 import weight_exchange_region as xr  # noqa: E402

ROOT = pathlib.Path(xr.__file__).resolve().parents[2]  # .../srt/..  -> sglang

#: The exchange lane, by path prefix.  Same lane as B4l's ratchet so the two
#: censuses are comparable; a wider one would mix in refusals nobody owns here.
LANE = (
    "srt/weg2/",
    "srt/managers/scheduler_components/weight_updater.py",
    "srt/model_executor/model_runner.py",
)

#: The three FORMS a refusal takes in this lane.  The first is B4l's; the other
#: two are the ones it cannot see, and they are why this file exists.
FORM_RAISE = "raise_direct"          # raise Weg2X(...)
FORM_HELPER = "raise_via_refuse"     # raise <recv>._refuse(Weg2X, ...)
FORM_VOTE = "vote"                   # write_matrix_verdict(row, False)

#: MEASURED FLOORS at 04cd920add, so a narrowed predicate fails instead of
#: quietly finding less.  A scan that sees zero of a form is blind to it, and a
#: blind scan passes by finding nothing (#1300's shape one tool over).
FORM_FLOOR = {FORM_RAISE: 174, FORM_HELPER: 7, FORM_VOTE: 1}

#: THE FROZEN SET: every refusal-bearing owner in the lane that NOTHING in the
#: lane references.  It SHRINKS on its own and GROWS only at a merge -- a new
#: entry means an unreachable refusal was added (and on a train it says WHICH
#: slice added it), a stale entry means one was wired and the debt is paid.
ZERO_REFERENCE_REFUSERS = (
    {
        "fn": "arm_oncard_lane",
        "file": "srt/weg2/weight_exchange_transport.py",
        "forms": (FORM_RAISE,),
        "class": "DECLARED-TODO",
        "why": "S6's probe-and-degrade arm, declared by TODO(S6); already B4l's "
               "frozen-debt entry 2, and listed here so the two censuses agree",
    },
    {
        "fn": "refuse_if_not_ok",
        "file": "srt/weg2/weight_exchange.py",
        "forms": (FORM_RAISE,),
        "class": "UNWIRED-BY-DESIGN-AND-PINNED",
        "why": "no group fence at the end of weight loading (B4i); already "
               "B4l's frozen-debt entry 1",
    },
    {
        "fn": "gate0_check",
        "file": "srt/weg2/weight_exchange_region.py",
        "forms": (FORM_HELPER, FORM_VOTE),
        "class": "REFUSAL-EXISTS-BUT-UNREACHABLE",
        "why": "#1335/B4r. It carries the ONLY per-tag comparison of the plan's "
               "claim against the SAVER's tag ledger, which is the one thing "
               "that would gate the negative slack_mib measured on four boots. "
               "NOT in B4l's debt because its refusal travels by the _refuse "
               "helper and by a vote, neither of which a `raise Weg2*` scan "
               "matches. Wiring it adds collectives to the wake seam -> "
               "operator design call, named in the module docstring",
    },
    {
        "fn": "waves_for_plan",
        "file": "srt/weg2/weight_exchange.py",
        "forms": (FORM_RAISE,),
        "class": "DECLARED-TODO",
        "why": "arrived with #1273 B4d on the merge train; the producer half is "
               "wired and the reader's ONE call-site line "
               "(weight_exchange_shadow.py:3094, `or wx.waves_for_plan`) is "
               "seat 3's, named in B4d's own commit message and not on the "
               "train. Also B4l's frozen-debt entry 3, so the two censuses agree",
    },
    # ---------------------------------------------------------------- #1352
    # THE REMAP SLICE'S FIVE ZERO-REFERENCE ENTRY POINTS.  (``plan_leg`` is NOT
    # here: it is referenced by ``minimum_fund`` inside the module, so the
    # census sees it -- which is the reference-not-call rule earning itself.)  They are PRESENT-BUT-UNWIRED
    # -- the middle of the three delivery states, and the expensive one to
    # mis-read in either direction (#859).  The page plan is built, tested at
    # the measured geometries and executed by its own smoke; what does NOT
    # exist yet is the ONE manifest field it consumes (the piece's byte offset
    # inside its tag's page sequence, AMENDMENT 8 section 8.9), so wiring the
    # production caller before that field lands would be planning on MODELLED
    # placement -- which the module itself refuses by name.
    #
    # They are listed INDIVIDUALLY rather than as one entry because each is a
    # separate caller the wiring seat has to add, and a single collective line
    # would let four of them be forgotten while the fifth was wired.
    {
        "fn": "plan_pages",
        "file": "srt/weg2/xchg_pageplan.py",
        "forms": (FORM_RAISE,),
        "class": "UNWIRED-BY-DESIGN-AND-PINNED",
        "why": "#1352. plan_leg's single-destination case, kept public because "
               "the on-card lane is a legitimate one-destination plan; same "
               "wiring site and same precondition as plan_leg",
    },
    {
        "fn": "minimum_fund",
        "file": "srt/weg2/xchg_pageplan.py",
        "forms": (FORM_RAISE,),
        "class": "UNWIRED-BY-DESIGN-AND-PINNED",
        "why": "#1352. Sizes the BOOT-time page fund against the schedule "
               "rather than the net image difference. Its caller is the "
               "launcher's arm arithmetic, which the wiring seat adds with the "
               "seam-fund reservation",
    },
    {
        "fn": "verify_leg",
        "file": "srt/weg2/xchg_pageplan.py",
        "forms": (FORM_RAISE,),
        "class": "UNWIRED-BY-DESIGN-AND-PINNED",
        "why": "#1352. The INDEPENDENT replay of a plan's own steps -- it "
               "already earned itself twice at the desk by rejecting two "
               "schedules this seat's scheduler had just produced. Belongs "
               "beside the plan's arm-time print, not in the flip's hot path",
    },
    {
        "fn": "extents_from_manifest_order",
        "file": "srt/weg2/xchg_pageplan.py",
        "forms": (FORM_RAISE,),
        "class": "REFUSAL-EXISTS-BUT-UNREACHABLE",
        "why": "#1352. The DESK-ONLY prefix-sum model of arena placement. It "
               "is deliberately unreachable from production: a plan built on "
               "it is refused unless the caller says it is desk arithmetic, so "
               "wiring it would be the defect, not the fix. It disappears when "
               "the manifest carries measured offsets",
    },
    {
        "fn": "cut1_cost_ms",
        "file": "srt/weg2/xchg_pageplan.py",
        "forms": (FORM_RAISE,),
        "class": "REFUSAL-EXISTS-BUT-UNREACHABLE",
        "why": "#1352. Prices Cut 1 (on-card bytes crossing PCIe twice) so "
               "Cut 2 is a decision with a number. Its refusal guards a zero "
               "transfer rate; the number is read by the spec and the smoke, "
               "and a production caller only appears with the Cut-2 decision",
    },
)

#: 4 until #1352 added the remap slice's five zero-reference entry points, each of them
#: PRESENT-BUT-UNWIRED behind ONE missing manifest field (AMENDMENT 8 section
#: 8.9) rather than behind a decision -- so each carries its own wiring site and
#: its own reason above, and the list shrinks entry by entry as the wiring seat
#: lands them.
#: 3 until the merge train landed #1273 B4d, which ADDED a zero-reference
#: raiser.  The "can only SHRINK" rule above holds for a tree that gains no new
#: refusal-bearing owner; a merge train is the event that can, and a growth is
#: admissible only WITH a class and a reason -- enforced by the shape tests,
#: not trusted.
SIZE_ON_THE_LINE = 9

#: B4l's ratchet, by file name, because the two lists must be read TOGETHER.
#: It lives on ``weg2/xchg-b4l-b4m-0911`` (@ ``fec81234e9``, based on RE-STAMP 5
#: ``ec753f00d9``) and is NOT in this tree yet, so the check below is DORMANT
#: rather than passing -- see its skip reason.
B4L_RATCHET = "test_weg2_xchg_refusal_reachability_1328.py"

#: Classes a frozen entry may carry.  ``UNKNOWN`` is refused outright: an entry
#: nobody has classified is a triage note, not a debt anyone owns.
DEBT_CLASSES = (
    "DECLARED-TODO",
    "UNWIRED-BY-DESIGN-AND-PINNED",
    "REFUSAL-EXISTS-BUT-UNREACHABLE",
)


class _Scan(ast.NodeVisitor):
    """Refusal sites by form, plus a REFERENCE census (never a call census).

    THE REFERENCE HALF IS LOAD-BEARING and was found by a false positive of my
    own: counting CALLS put ``diagonal`` (``weight_exchange_transport.py``) in
    the zero set, because it is never called by name -- it is handed to
    ``threading.Thread(target=guarded(diagonal))``.  A thunk passed by reference
    is wired, and a census that cannot see that reports a live path as dead.
    Docstrings and ``__all__`` entries are string constants and therefore
    invisible here by construction, which is the other half of why this is an
    AST question and not a grep.
    """

    def __init__(self, rel: str, sites, refs, defs) -> None:
        self.rel, self.stack = rel, []
        self.sites, self.refs, self.defs = sites, refs, defs

    def visit_FunctionDef(self, node):  # noqa: N802
        self.defs.setdefault(node.name, (self.rel, node.lineno))
        self.stack.append(node.name)
        self.generic_visit(node)
        self.stack.pop()

    visit_AsyncFunctionDef = visit_FunctionDef

    def _owner(self) -> str:
        return self.stack[-1] if self.stack else "<module>"

    def visit_Name(self, node):  # noqa: N802
        if isinstance(node.ctx, ast.Load):
            self.refs[node.id] += 1
        self.generic_visit(node)

    def visit_Attribute(self, node):  # noqa: N802
        if isinstance(node.ctx, ast.Load):
            self.refs[node.attr] += 1
        self.generic_visit(node)

    def visit_Call(self, node):  # noqa: N802
        name = getattr(node.func, "attr", getattr(node.func, "id", None))
        # A VOTE is a refusal: `write_matrix_verdict(row, False)` makes all six
        # ranks refuse on the shared row without this rank raising anything.
        if (name == "write_matrix_verdict" and len(node.args) >= 2
                and isinstance(node.args[1], ast.Constant)
                and node.args[1].value is False):
            self.sites[FORM_VOTE].append((self.rel, node.lineno, self._owner()))
        self.generic_visit(node)

    def visit_Raise(self, node):  # noqa: N802
        if node.exc is None:
            return self.generic_visit(node)
        exc = node.exc
        func = exc.func if isinstance(exc, ast.Call) else exc
        name = func.id if isinstance(func, ast.Name) else getattr(func, "attr", None)
        if name and name.startswith("Weg2"):
            self.sites[FORM_RAISE].append((self.rel, node.lineno, self._owner()))
        elif name == "_refuse" and isinstance(exc, ast.Call) and exc.args:
            first = exc.args[0]
            cls = getattr(first, "id", getattr(first, "attr", None))
            if cls and cls.startswith("Weg2"):
                self.sites[FORM_HELPER].append(
                    (self.rel, node.lineno, self._owner()))
        self.generic_visit(node)


def _census():
    sites = collections.defaultdict(list)
    refs: collections.Counter = collections.Counter()
    defs: dict = {}
    for path in sorted(ROOT.rglob("*.py")):
        rel = str(path.relative_to(ROOT))
        if not any(rel.startswith(p) or rel == p for p in LANE):
            continue
        try:
            tree = ast.parse(path.read_text(), filename=str(path))
        except SyntaxError:  # pragma: no cover -- a broken file is not our news
            continue
        _Scan(rel, sites, refs, defs).visit(tree)
    owners = {o for f in sites for (_, _, o) in sites[f]} - {"<module>"}
    zero = {o for o in owners if refs.get(o, 0) == 0}
    forms_of = {
        o: tuple(sorted(f for f in sites if any(x[2] == o for x in sites[f])))
        for o in owners
    }
    return sites, refs, defs, owners, zero, forms_of


def test_the_zero_reference_refusers_are_exactly_the_frozen_three():
    """Closed in BOTH directions, because each direction is a different order.

    A NEW name means an unreachable refusal was added: wire it, or add it here
    with its class and one line of reason.  A STALE name means one was wired and
    the entry must go -- that is the debt being PAID, and it must not pass
    silently either, or the list rots into a permanent excuse.
    """
    _, _, defs, _, zero, _ = _census()
    frozen = {e["fn"] for e in ZERO_REFERENCE_REFUSERS}
    new = sorted(zero - frozen)
    stale = sorted(frozen - zero)
    assert not new, (
        f"unreachable refusals not in the frozen list: "
        f"{[(n, defs.get(n)) for n in new]} -- wire the owner, or add it with a "
        f"class from {DEBT_CLASSES} and one line of reason"
    )
    assert not stale, (
        f"frozen entries that are now referenced (or gone): {stale} -- the debt "
        f"is paid, so delete the entry and lower SIZE_ON_THE_LINE"
    )


def test_the_frozen_list_cannot_grow():
    assert len(ZERO_REFERENCE_REFUSERS) == SIZE_ON_THE_LINE
    assert len({e["fn"] for e in ZERO_REFERENCE_REFUSERS}) == SIZE_ON_THE_LINE


def test_every_frozen_entry_carries_a_class_and_a_reason():
    for entry in ZERO_REFERENCE_REFUSERS:
        assert entry["class"] in DEBT_CLASSES, entry
        assert len(entry["why"].split()) >= 6, entry
        assert entry["forms"], entry


def test_the_scan_SEES_all_three_refusal_forms():
    """THE CAN-FAIL PROOF, and the reason the floors are numbers.

    A predicate narrowed back to B4l's (``raise Weg2*`` only) finds zero of the
    other two forms, and a scan that finds zero of a form passes by finding
    nothing.  The floors are the measured counts at 04cd920add.
    """
    sites, _, _, _, _, _ = _census()
    for form, floor in FORM_FLOOR.items():
        assert len(sites[form]) >= floor, (
            f"{form}: {len(sites[form])} sites, measured floor {floor} -- the "
            f"predicate got narrower, so this census is blind to that form"
        )


def test_a_thunk_handed_to_a_thread_counts_as_WIRED():
    """The false positive that the reference census exists to close.

    ``diagonal`` is refusal-bearing and is never CALLED by name; it is handed to
    ``threading.Thread(target=guarded(diagonal))``.  A call census reports it
    dead, and the on-card lane demonstrably ran on XSN17/18/19.
    """
    _, refs, _, owners, zero, _ = _census()
    assert "diagonal" in owners, "the on-card thunk stopped carrying a refusal"
    assert refs["diagonal"] >= 1
    assert "diagonal" not in zero


def test_gate0_checks_refusal_travels_BY_HELPER_AND_BY_VOTE_never_by_a_bare_raise():
    """The blind spot, as a test rather than as a sentence.

    This is the one-to-one correlation: the two entries B4l's frozen debt names
    are exactly those whose only form is a bare ``raise Weg2*``; the entry it
    misses is exactly the one that has NO such site.
    """
    _, _, _, _, _, forms_of = _census()
    assert forms_of["gate0_check"] == (FORM_HELPER, FORM_VOTE), forms_of["gate0_check"]
    assert FORM_RAISE not in forms_of["gate0_check"]
    for fn in ("arm_oncard_lane", "refuse_if_not_ok"):
        assert forms_of[fn] == (FORM_RAISE,), (fn, forms_of[fn])


def test_the_comparison_gate0_check_carries_is_the_one_the_slack_needs():
    """Not a scan: the SUBSTANCE, so a refactor cannot hollow the entry out.

    ``planned`` against the saver's ``tms_tag_bytes`` PER TAG, with ABSENT as its
    own case and the delta printed -- that is what the negative ``slack_mib``
    needs and what no running instrument does.
    """
    import inspect
    src = inspect.getsource(xr.gate0_check)
    assert "tms_tag_bytes" in src and "tag_totals" in src
    assert "ABSENT" in src, "an absent census must not become a quiet zero"
    assert "delta=" in src, "a refusal without the delta is not actionable"
    assert "write_matrix_verdict" in src, "the verdict must reach the other five"


def test_the_vote_predicate_REJECTS_an_ok_verdict():
    """MUTANT R4's lesson: a FLOOR catches a narrowed predicate, never a widened
    one.  Dropping the ``is False`` literal check made the census match every
    ``write_matrix_verdict(...)`` call and the floor still passed -- a refusal
    census that counts an OK verdict as a refusal.

    The tree has no counter-example to pin it against: there is exactly ONE
    production call site (``weight_exchange_region.py:1550``) and it passes
    ``False``.  So the predicate is pinned against a SYNTHETIC pair instead,
    which tests the predicate rather than the tree -- the only way to bound a
    classifier whose negative case does not exist upstream.
    """
    src = (
        "def refuses(region, row):\n"
        "    region.write_matrix_verdict(row, False)\n"
        "def approves(region, row):\n"
        "    region.write_matrix_verdict(row, True)\n"
        "def unknown(region, row, ok):\n"
        "    region.write_matrix_verdict(row, ok)\n"
    )
    sites: collections.defaultdict = collections.defaultdict(list)
    refs: collections.Counter = collections.Counter()
    _Scan("<synthetic>", sites, refs, {}).visit(ast.parse(src))
    owners = [o for (_, _, o) in sites[FORM_VOTE]]
    assert owners == ["refuses"], (
        f"the vote predicate classified {owners} -- only an explicit False "
        f"literal is a refusal; True is an approval and a variable is neither"
    )

# ===========================================================================
# (5) THE TWO FROZEN DEBT LISTS, AND WHY THEIR SIZES DIFFER BY CONSTRUCTION.
# ===========================================================================
#
# After the merge train this tree carries TWO frozen debt lists over the same
# lane, of DIFFERENT SIZES, and that is CORRECT.  Written down here because the
# obvious "fix" is the defect: pulling one list to match the other loses
# `gate0_check`, i.e. exactly the refusal the operator decided on 2026-09-11 to
# leave UNWIRED with a reason.  Losing it would turn a recorded decision back
# into an invisible gap.
#
#   B4l  (`test_weg2_xchg_refusal_reachability_1328.py`): predicate = a raise
#        whose raised expression's callee NAME starts with `Weg2`, i.e.
#        `raise Weg2X(...)`.  Its debt therefore holds exactly those
#        zero-reference owners whose ONLY refusal form is that bare raise.
#        After #1333 paid its `diagonal_carrier_bytes` entry: TWO.
#
#   THIS FILE: predicate = the same bare raise PLUS
#        `raise <recv>._refuse(Weg2X, ...)` (the helper form) PLUS
#        `write_matrix_verdict(row, False)` (the VOTE form -- a group-wide
#        refusal that never raises locally).  THREE.
#
# The difference is one name, `gate0_check`, and it is not an oversight in
# either file: its refusal travels ONLY by the helper and by the vote, so B4l's
# predicate cannot match it.  MEASURED in the lane, and the correlation is
# one-to-one rather than argued: 174 `raise_direct` sites, 7 `raise_via_refuse`,
# 1 `vote`; the zero-reference owners are exactly THREE; the two B4l names are
# exactly those whose only form is `raise_direct`.
#
# So the invariant that ties the two lists is NOT "same size" but:
#
#     B4l's set == { entries of THIS list whose forms include FORM_RAISE }
#
# and the test below ENFORCES that instead of asserting it in prose -- because a
# prose pointer on one side only is the half nobody reads.  The mirror pointer
# for B4l's own file travels as a named merge-train item
# (`/spinning/gpu-arb/weg2/MERGE_TRAIN_WEG2_0911.md`), since that file is not in
# this tree and a divergent copy of it would itself be the defect.


def _b4l_frozen_debt():
    """``(set of fn names, pinned size)`` from B4l's ratchet, or ``None``.

    Read out of its AST, not imported: importing a sibling test module to
    inspect its constants couples two suites at collection time, and a literal
    is the one thing an AST can read with no side effect at all.
    """
    path = pathlib.Path(__file__).with_name(B4L_RATCHET)
    if not path.is_file():
        return None
    tree = ast.parse(path.read_text(), filename=str(path))
    names, size = None, None
    for node in ast.walk(tree):
        if not isinstance(node, ast.Assign):
            continue
        for target in node.targets:
            if getattr(target, "id", None) == "UNWIRED_DEBT":
                entries = ast.literal_eval(node.value)
                names = {e["fn"] for e in entries}
            elif getattr(target, "id", None) == "DEBT_SIZE_ON_THE_LINE":
                size = ast.literal_eval(node.value)
    return names, size


def test_the_two_frozen_debt_lists_agree_BY_CONSTRUCTION():
    """The enforced half of the cross-reference.

    DORMANT until B4l lands, and dormant LOUDLY: a skip that reads as a pass is
    the #1300 shape, so the reason names the branch, the SHA and the base.
    """
    debt = _b4l_frozen_debt()
    if debt is None:
        pytest.skip(
            f"{B4L_RATCHET} is not in this tree -- it lives on "
            f"weg2/xchg-b4l-b4m-0911 @ fec81234e9, based on RE-STAMP 5 "
            f"ec753f00d9. This check is DORMANT, not passing, and arms itself "
            f"the moment that file lands (merge-train Kante F)."
        )
    theirs, their_size = debt
    expected = {e["fn"] for e in ZERO_REFERENCE_REFUSERS
                if FORM_RAISE in e["forms"]}
    assert theirs == expected, (
        f"the two frozen debt lists have drifted. B4l's predicate is the bare "
        f"`raise Weg2*`, so its set must be exactly this file's entries whose "
        f"forms include {FORM_RAISE}: expected {sorted(expected)}, found "
        f"{sorted(theirs)}. DO NOT make the two lists equal -- the difference "
        f"is the point."
    )
    assert their_size == len(expected), (
        f"B4l pins DEBT_SIZE_ON_THE_LINE={their_size} but the derived set has "
        f"{len(expected)}; #1333 paid its `diagonal_carrier_bytes` entry, so "
        f"that entry goes and the size drops with it"
    )
    # The two names that must NEVER appear in B4l's list, each for its own
    # reason -- stated so a future reader does not "restore" either.
    assert "gate0_check" not in theirs, (
        "gate0_check belongs to THIS file only: its refusal travels by the "
        "_refuse helper and by write_matrix_verdict(row, False), neither of "
        "which B4l's `raise Weg2*` predicate can match. Adding it there would "
        "claim a detection B4l's scan does not perform."
    )
    assert "diagonal_carrier_bytes" not in theirs, (
        "#1333 deleted it -- that entry's own reason said "
        "'seat 6 picks one authority, default DELETE', and the debt was paid"
    )
