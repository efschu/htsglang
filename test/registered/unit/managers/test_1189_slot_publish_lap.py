"""#1189 -- every lap of a PP slot must PUBLISH what that slot ran, "nothing"
included. Today three publish sites are nested inside a truthy guard with no
`else`, so a slot that runs nothing KEEPS ITS PREVIOUS ENTRY.

THE ROOT, read at this tree (`/spinning/wt-weg1`, `feat/weg1-store-read`,
HEAD `6ef2c7f313`), never from the index -- the codegraph pin stands on
`/spinning/htsglang @ merge/train-0901` and prints a STALE-PIN warning:

    scheduler_pp_mixin.py:5046  if self.mbs[next_mb_id] is not None:      <- no else
    scheduler_pp_mixin.py:5077      if next_batch_result is None:          <- the #1009 lap
    scheduler_pp_mixin.py:5091      else:
    scheduler_pp_mixin.py:5134          self._pp_record_slot_last_batch(next_mb_id)
    scheduler_pp_mixin.py:5135  if not self.pp_group.is_last_rank:

THOSE LINE NUMBERS ARE THE ROOT AS FIRST READ, AT HEAD `6ef2c7f313`, AND THEY
HAVE MOVED. Measured in the working tree 2026-09-04 by the same AST locators
this file uses, RE-MEASURED 2026-09-04 after the review-round-2 edits (which
moved everything below :5091 again): the three publishing guards are at :5046,
:5512 and :5717, the #1009 arm at :5077, the lap-(B) `last_mbs[next_mb_id] =
None` at :5174, and the six helper calls at :5218, :5244, :5520, :5529, :5726,
:5733. Every locator below is STRUCTURAL and re-derives them at run time -- the
numbers in prose are the drifting half and are dated wherever they appear.

The publish is nested TWO levels deep, not one, and that is a correction to
every earlier framing of this defect. There are therefore TWO laps that leave
`last_mbs[slot]` holding a stale EXTEND batch:

  (A) the slot holds NO batch          -- outer `if` False, no `else`;
  (B) the slot holds a batch that has  -- the #1009 arm at :5077, which the
      NOT RUN                             tree itself documents as a real
                                          state on this box.

WHY THE STALE ENTRY IS FATAL, in the tree's own words
(`_pp_record_slot_last_batch.__doc__`, `scheduler_pp_mixin.py:7274-7367`, working tree 2026-09-04):
`get_next_batch_to_run` reads `last_batch`, sees `forward_mode.is_extend()`,
and reaches `running_batch.merge_batch(last_batch)` -- which extends `reqs`
IN PLACE. The same requests are appended once per visit, forever. Boot 8
(`/spinning/evidence-665-f1/boot_855_weg1b8_e9d1a719ac_0904_064622.log`):
`running=7768` stood against `max_running_requests=8` (log `:458441`, a `#788
PP-ADMISSION verdict=DECLINE` line, and the max of that series). NAMED BY
INSTRUMENT: `running_bs=7768` occurs ZERO times in that log -- its `running_bs`
series tops out at 7771. The admission split was 183 `PP-ADMISSION
verdict=ADMIT` against 27127 `PP-ADMISSION verdict=DECLINE` -- that instrument
is named because `#969N ADMIT` counts 150 lines for the same word. And
`Scheduler.on_idle`'s whole ownership/leak checker family stayed unreachable
behind `running_batch.is_empty()`.

STRICT PHASE PURITY IS WHAT CREATED THE STATE. Upstream has no lap where a
slot holds requests and runs nothing, so upstream's placement of the publish
inside the guard was never wrong FOR UPSTREAM. That is the L6 reconciliation:
this is coverage for a state the fork's own purity feature invented, not a
break with upstream. The helper `_pp_record_slot_last_batch` was already
extracted to be unconditional and its docstring already argues the case --
only its single call site was never lifted out of the guard.

WHAT THIS FILE ASSERTS. Status column re-measured 2026-09-04, after the fix
landed in the working tree -- items 1-4 were RED when this file was written
and are GREEN now, which is the only honest way to carry a red-first spec
forward.

  1. `_event_loop_pp_body`          -- lap (A) publishes.        GREEN.
  2. `_event_loop_pp_body`          -- lap (B) publishes None.   GREEN.
  3. `event_loop_pp_disagg_prefill` -- same shape.               GREEN.
  4. `event_loop_pp_disagg_decode`  -- same shape.               GREEN.
  5. the helper itself is unconditional AND publishes `mbs[<its own slot>]`
     -- GREEN, and must stay green: it localises the defect at the CALL SITE
     and refuses the tempting "fix" of touching the helper.
  6. a CENSUS of every `self.mbs[...] is not None` guard in the file, so a
     fourth publishing site cannot appear silently. GREEN.
  7. THE OUTCOME, EXECUTED, LAP (A) (`LapAOutcomeIsAClearedSlotRecord`): the
     real `ast.If` guard is lifted out of the shipping source, its true arm
     replaced by `pass`, compiled, and RUN against a stand-in scheduler
     carrying the real `_pp_record_slot_last_batch`. The assertion is on the
     STATE AFTERWARDS -- `last_mbs[slot] is None`, the record CLEARED -- and
     then on what the shipping `ScheduleBatch.merge_batch` consumer does
     with it, counted in DISTINCT rids, never in list length.

  8. THE OUTCOME, EXECUTED, LAP (B) (`LapBOutcomeIsAClearedSlotRecord`):
     the same treatment for the #1009 arm, whose `ast.If` is lifted out with
     only its `else` replaced by `pass`. ADDED 2026-09-04 after a measured
     hole: lap (B) had NO executed coverage (`_compiled_lap` replaces the
     arm it lives in with `pass`) and its three structural pins are all
     conditionality-blind, so a never-true `if` around
     `self.last_mbs[next_mb_id] = None` -- #1189 lap B fully open -- survived
     the whole suite at 14 passed / 0 failed. The same blindness on lap (A)
     is now also pinned structurally by the `unconditional` term of
     `_lap_a_publish_effect` and, for lap (B), by the direct-child assertion
     in `test_lap_b_publishes_nothing_not_the_unrun_batch`.

WHY ITEM 7 EXISTS, measured rather than feared: items 1-4 rest on
`_publishes()`, which answers True for ANY call of that name and ANY write to
`last_mbs`, so it cannot tell this fix from a publish that leaves #1189 fully
open. Two such mutants survived the whole suite at 8 passed / 0 failed on
2026-09-04 (R1/R2, both reproduced on disk, verified by unified diff and
md5-restored afterwards; they are spelled out on `_lap_a_publish_effect`).
Item 7 and the slot/value assertions added to items 1-4 are what kill them.

This file is therefore NO LONGER pure-`ast`/no-torch: item 7 imports the
shipping `ScheduleBatch` and `SchedulerPPMixin` and drives real code on CPU.
Items 1-6 remain structural and run without either.

WHY NOT A DRIVEN LOOP TEST: `_event_loop_pp_body` is 990 lines and blocks on
the PP request chain; the existing driven suite for this loop
(`test_pp_flip_slot_hold_631.py`) needs ~200 lines of ring stubs and still
does not reach the publish. The invariant here is a CONTROL-FLOW property --
"no lap leaves without publishing" -- and control flow is exactly what an AST
can decide without a fake ring that could itself encode the defect (#630).
The BEHAVIOURAL half of #1189 -- what the stale entry does when it is merged
-- is driven against the shipping `ScheduleBatch.merge_batch` in
`test_1189_merge_cardinality_1189.py`.

Line numbers in the messages are informational and re-derived at run time;
every locator is structural, so the file may drift without this suite lying.

BASE CLASS: plain ``unittest.TestCase``, not ``CustomTestCase``. Every
assertion here is a deterministic read of a file on disk, so
``CustomTestCase``'s retry wrapper can only run each failure three times and
then replace the reason with "retry() exceed maximum number of retries." in
the summary line -- measured on the first run of this suite. A red-first
suite whose red does not name its reason is the thing this campaign keeps
paying for. 658 of the registered manager tests already use plain
``unittest.TestCase``.
"""

import ast
import copy
import importlib.util
import pathlib
import types
import unittest

from sglang.srt.managers.schedule_batch import ScheduleBatch
from sglang.srt.managers.scheduler_pp_mixin import SchedulerPPMixin
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=25, suite="base-a-test-cpu")

MIXIN = (
    pathlib.Path(__file__).resolve().parents[4]
    / "python"
    / "sglang"
    / "srt"
    / "managers"
    / "scheduler_pp_mixin.py"
)

# The two forms the publish takes in this file. Both write `last_mbs[slot]`;
# one goes through the extracted helper, one is still the raw subscript.
PUBLISH_CALL = "_pp_record_slot_last_batch"
PUBLISH_ATTR = "last_mbs"


def _tree():
    return ast.parse(MIXIN.read_text())


def _funcs(tree):
    out = {}
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            out.setdefault(node.name, []).append(node)
    return out


def _publishes(node) -> bool:
    """Does this subtree write ``self.last_mbs[...]``, by either form?

    DELIBERATELY COARSE, AND ITS BLINDNESS IS MEASURED. It answers True for
    ANY call named ``_pp_record_slot_last_batch`` with ANY argument, and for
    ANY Subscript write to ``last_mbs`` with ANY value. That is right for
    CLASSIFYING a guard ("does this branch publish at all") and wrong as a
    correctness assertion: two mutants that leave #1189 fully open survived
    the whole suite at 8 passed / 0 failed. Use ``_lap_a_publish_effect``
    for the outcome, never this.
    """
    for n in ast.walk(node):
        if isinstance(n, ast.Call):
            f = n.func
            if isinstance(f, ast.Attribute) and f.attr == PUBLISH_CALL:
                return True
        if isinstance(n, (ast.Assign, ast.AugAssign, ast.AnnAssign)):
            targets = n.targets if isinstance(n, ast.Assign) else [n.target]
            for t in targets:
                if (
                    isinstance(t, ast.Subscript)
                    and isinstance(t.value, ast.Attribute)
                    and t.value.attr == PUBLISH_ATTR
                ):
                    return True
    return False


def _slot_guards(fn):
    """Every ``if self.mbs[<x>] is not None:`` inside ``fn``.

    Located by SHAPE, not by line number: a `Compare` of `IsNot` against
    `None` whose left side subscripts `self.mbs`.
    """
    found = []
    for n in ast.walk(fn):
        if not isinstance(n, ast.If):
            continue
        t = n.test
        if not isinstance(t, ast.Compare) or len(t.ops) != 1:
            continue
        if not isinstance(t.ops[0], ast.IsNot):
            continue
        rhs = t.comparators[0]
        if not (isinstance(rhs, ast.Constant) and rhs.value is None):
            continue
        lhs = t.left
        if (
            isinstance(lhs, ast.Subscript)
            and isinstance(lhs.value, ast.Attribute)
            and lhs.value.attr == "mbs"
            and isinstance(lhs.value.value, ast.Name)
            and lhs.value.value.id == "self"
        ):
            found.append(n)
    return found


def _1009_arms(guard):
    """Every ``if next_batch_result is None:`` inside ``guard``'s TRUE arm.

    That arm is lap (B): the slot HOLDS a batch whose pass has not delivered.
    Located by SHAPE like every other locator in this file, so it may move
    without this suite lying. Returned as a list because each caller states
    its own count expectation in its own failure message.
    """
    return [
        n
        for n in ast.walk(ast.Module(body=guard.body, type_ignores=[]))
        if isinstance(n, ast.If)
        and isinstance(n.test, ast.Compare)
        and len(n.test.ops) == 1
        and isinstance(n.test.ops[0], ast.Is)
        and isinstance(n.test.left, ast.Name)
        and n.test.left.id == "next_batch_result"
    ]


def _enclosing_body(fn, node):
    """The statement list that directly holds ``node``."""
    for n in ast.walk(fn):
        for field in ("body", "orelse", "finalbody"):
            body = getattr(n, field, None)
            if isinstance(body, list) and node in body:
                return body
    return None


def _lap_publishes_when_guard_is_false(fn, guard) -> bool:
    """Is a publish reachable on the lap where ``guard`` is False?

    Two accepted shapes, and only two, because only these two make "nothing"
    a real answer for this slot on this lap:

      * an ``else:`` on the guard that publishes, or
      * an unguarded publish beside the guard in the same statement list.

    A publish in a *different* branch of an unrelated `if` does not count --
    that is how a partial fix would sneak past.
    """
    if guard.orelse and _publishes(ast.Module(body=guard.orelse, type_ignores=[])):
        return True
    body = _enclosing_body(fn, guard)
    if body is None:
        return False
    for stmt in body:
        if stmt is guard:
            continue
        if isinstance(stmt, (ast.If, ast.For, ast.While, ast.Try, ast.With)):
            continue  # guarded: not reachable on every lap
        if _publishes(stmt):
            return True
    return False


def _helper_value_slot(funcs):
    """The helper's own slot parameter, IF its body is ``last_mbs[p] = mbs[p]``.

    The whole lap-(A) outcome argument hangs on this one link and nothing
    else: the else arm calls the helper with the SAME slot the guard tested,
    and the helper writes ``last_mbs[slot] = mbs[slot]``. The guard being
    False IS ``self.mbs[slot] is None``, so the value that lands in the
    record is None -- the record is CLEARED, not merely "written". Returns
    None if that link is broken, and every caller then fails loudly rather
    than deriving an outcome from a helper that no longer has that shape.
    """
    fns = funcs.get(PUBLISH_CALL) or []
    if len(fns) != 1:
        return None
    fn = fns[0]
    args = fn.args.args
    if len(args) != 2 or args[0].arg != "self":
        return None
    param = args[1].arg
    body = [st for st in fn.body if not isinstance(st, ast.Expr)]
    if len(body) != 1 or not isinstance(body[0], ast.Assign):
        return None
    assign = body[0]
    if len(assign.targets) != 1:
        return None
    tgt, val = assign.targets[0], assign.value
    target_ok = (
        isinstance(tgt, ast.Subscript)
        and isinstance(tgt.value, ast.Attribute)
        and tgt.value.attr == PUBLISH_ATTR
        and ast.unparse(tgt.slice) == param
    )
    value_ok = (
        isinstance(val, ast.Subscript)
        and isinstance(val.value, ast.Attribute)
        and val.value.attr == "mbs"
        and ast.unparse(val.slice) == param
    )
    return param if (target_ok and value_ok) else None


def _lap_a_publish_effect(fn, guard, funcs):
    """WHICH SLOT the guard-False lap writes, and WHAT VALUE -- not merely THAT.

    Returns ``(slot_expr, value_expr, lineno, unconditional)`` with the value
    resolved THROUGH the helper when the publish is a call, or ``None`` when
    no publish is reachable on that lap or more than one is.

    ``unconditional`` IS THE FOURTH TERM AND IT WAS ADDED AFTER A SURVIVING
    MUTANT OF ITS OWN. It says the publish is a DIRECT statement of the lap,
    not merely somewhere inside it. Every locator in this file finds the
    publish with ``ast.walk``, which walks THROUGH an ``if``, so a never-true
    wrapper around the publish leaves the whole structural layer green --
    measured on lap (A) as mutant M-e (the ``else``'s publish wrapped in a
    false ``if``: all six structural tests stayed GREEN, caught only by the
    executed outcome class) and on lap (B) as the ``_lapb_enabled`` mutant
    (whole suite 14 passed / 0 failed). A gate added around a fix is the
    realistic future mutation on this codebase, so it is pinned here rather
    than left to the executed classes alone.

    THIS IS THE ASSERTION ``_publishes()`` CANNOT MAKE, and it exists
    because two mutants that leave #1189 FULLY OPEN survived the whole suite
    at 8 passed / 0 failed. Both were applied on disk against the golden
    tree, verified by unified diff, and the golden md5-restored afterwards
    (2026-09-04):

      R1  ``self._pp_record_slot_last_batch(mb_id)`` in the lap-(A) else of
          ``_event_loop_pp_body``. WRONG SLOT: ``last_mbs[next_mb_id]`` keeps
          its stale EXTEND entry -- the original defect, untouched -- while
          ``last_mbs[mb_id]`` is additionally corrupted with a batch already
          consumed at the top of the same iteration.
      R2  ``self.last_mbs[next_mb_id] = self.mbs[mb_id]`` in the lap-(A) else
          of ``event_loop_pp_disagg_decode``. Right slot, WRONG SOURCE: the
          record becomes a foreign slot's batch instead of "nothing", so the
          consumer merges a batch this slot never ran.

    Neither is hypothetical -- R1 is the exact shape a copy-paste of the
    neighbouring ``mb_id``-indexed statements produces, and R2 is the shape
    the two disagg loops carried before the conversion.
    """
    if guard.orelse:
        stmts = list(guard.orelse)
    else:
        body = _enclosing_body(fn, guard)
        if body is None:
            return None
        stmts = [
            st
            for st in body
            if st is not guard
            and not isinstance(st, (ast.If, ast.For, ast.While, ast.Try, ast.With))
        ]
    calls, raws = [], []
    for st in stmts:
        # ``st`` is a DIRECT statement of the lap; ``n`` may sit anywhere
        # below it. Carrying both is what makes the fourth return term
        # possible: the publish is still FOUND when it is wrapped, and then
        # reported as conditional instead of vanishing into "no publish".
        for n in ast.walk(st):
            if (
                isinstance(n, ast.Call)
                and isinstance(n.func, ast.Attribute)
                and n.func.attr == PUBLISH_CALL
            ):
                calls.append((n, isinstance(st, ast.Expr) and st.value is n))
            if isinstance(n, ast.Assign):
                for t in n.targets:
                    if (
                        isinstance(t, ast.Subscript)
                        and isinstance(t.value, ast.Attribute)
                        and t.value.attr == PUBLISH_ATTR
                    ):
                        raws.append((t, n, st is n))
    if len(calls) + len(raws) != 1:
        return None
    if calls:
        call, unconditional = calls[0]
        if len(call.args) != 1 or call.keywords:
            return None
        slot = ast.unparse(call.args[0])
        if _helper_value_slot(funcs) is None:
            return None
        return (slot, f"self.mbs[{slot}]", call.lineno, unconditional)
    tgt, assign, unconditional = raws[0]
    return (
        ast.unparse(tgt.slice),
        ast.unparse(assign.value),
        assign.lineno,
        unconditional,
    )


class SlotPublishLapInvariant(unittest.TestCase):
    """Every lap through a slot must say what that slot ran."""

    def setUp(self):
        self.tree = _tree()
        self.funcs = _funcs(self.tree)

    # ---- lap (A): the slot holds no batch --------------------------------

    def _assert_lap_a(self, fname):
        fns = self.funcs.get(fname) or []
        self.assertEqual(
            len(fns), 1, f"{fname}: expected exactly one definition in the file"
        )
        fn = fns[0]
        guards = [g for g in _slot_guards(fn) if _publishes(g)]
        self.assertEqual(
            len(guards),
            1,
            f"{fname}: expected exactly one slot guard that carries the "
            f"publish; found {[g.lineno for g in guards]}",
        )
        guard = guards[0]
        self.assertTrue(
            _lap_publishes_when_guard_is_false(fn, guard),
            f"#1189 lap (A) at {MIXIN.name}:{guard.lineno} in {fname}: "
            f"`if self.mbs[...] is not None:` has has_else="
            f"{bool(guard.orelse)} and no unguarded publish beside it, so a "
            f"slot that holds NO batch keeps its PREVIOUS last_mbs entry. "
            f"That entry is an EXTEND batch; scheduler.py's "
            f"get_next_batch_to_run then reaches "
            f"running_batch.merge_batch(last_batch) on every later visit and "
            f"re-appends the same requests in place (boot 8: `running=7768` "
            f"against max_running_requests=8, "
            f"`boot_855_weg1b8_e9d1a719ac_0904_064622.log:458441`, a `#788 "
            f"PP-ADMISSION verdict=DECLINE` line; NAMED BY INSTRUMENT because "
            f"`running_bs=7768` occurs ZERO times in that log). The publish "
            f"helper is already "
            f"unconditional (see test_publish_helper_is_unconditional); only "
            f"this CALL SITE is nested.",
        )

        # ---- THE OUTCOME, not the call. Everything above this line is
        # satisfied by a publish that leaves #1189 fully open (mutants R1 and
        # R2, both measured surviving at 8/8 on 2026-09-04).
        guard_slot = ast.unparse(guard.test.left.slice)
        effect = _lap_a_publish_effect(fn, guard, self.funcs)
        self.assertIsNotNone(
            effect,
            f"{fname}: the guard-False lap at {MIXIN.name}:{guard.lineno} must "
            f"carry EXACTLY ONE publish whose slot and value can be resolved. "
            f"Either none is reachable, or several are, or "
            f"{PUBLISH_CALL} no longer has the shape "
            f"`last_mbs[<slot>] = mbs[<same slot>]` that makes the outcome "
            f"derivable (see _helper_value_slot).",
        )
        slot_expr, value_expr, ln, unconditional = effect
        self.assertTrue(
            unconditional,
            f"#1189 lap (A) at {MIXIN.name}:{ln} in {fname}: the publish is "
            f"present on the guard-False lap but NOT unconditional -- it sits "
            f"under a further branch instead of being a direct statement of "
            f"the `else`. A publish that can be skipped leaves #1189 open on "
            f"exactly the laps this fix was added to close, while every "
            f"`ast.walk`-based assertion in this file stays green (measured: "
            f"mutant M-e).",
        )
        self.assertEqual(
            slot_expr,
            guard_slot,
            f"#1189 lap (A) at {MIXIN.name}:{ln} in {fname}: the publish names "
            f"slot `{slot_expr}` while the guard tested `self.mbs[{guard_slot}]`. "
            f"A publish to a DIFFERENT slot leaves `last_mbs[{guard_slot}]` "
            f"holding its stale EXTEND entry -- #1189 untouched -- and "
            f"corrupts `last_mbs[{slot_expr}]` on top. This is mutant R1, "
            f"which the call-only assertions above cannot see.",
        )
        self.assertEqual(
            value_expr,
            f"self.mbs[{guard_slot}]",
            f"#1189 lap (A) at {MIXIN.name}:{ln} in {fname}: the publish lands "
            f"`{value_expr}` in `last_mbs[{guard_slot}]`. The only value that "
            f"CLEARS the record is `self.mbs[{guard_slot}]`, which this arm "
            f"proves is None -- the guard it is the else of is literally "
            f"`self.mbs[{guard_slot}] is not None`. Any other source (mutant "
            f"R2: `self.mbs[mb_id]`) publishes a batch this slot never ran.",
        )

    def test_lap_a_event_loop_pp_body(self):
        self._assert_lap_a("_event_loop_pp_body")

    def test_lap_a_event_loop_pp_disagg_prefill(self):
        self._assert_lap_a("event_loop_pp_disagg_prefill")

    def test_lap_a_event_loop_pp_disagg_decode(self):
        self._assert_lap_a("event_loop_pp_disagg_decode")

    # ---- lap (B): the slot holds a batch that has not run ----------------

    def test_lap_b_the_1009_arm_publishes(self):
        """The #1009 lap is the SECOND way to leave without publishing.

        `_event_loop_pp_body` guards the publish a second time on
        `next_batch_result is None` -- the lap the tree itself documents at
        :5050-5076 as a real state on a PP=3 box ("the slot is filled by this
        round's admission and the result-processing step for the same slot
        runs before any forward has happened"). On that lap the publish is
        skipped exactly as on lap (A), and the stale entry is preserved with
        the same consequence. A fix that adds only the outer `else` leaves
        this one open.
        """
        fn = self.funcs["_event_loop_pp_body"][0]
        guard = [g for g in _slot_guards(fn) if _publishes(g)][0]
        inner = _1009_arms(guard)
        self.assertEqual(
            len(inner),
            1,
            "expected exactly one `if next_batch_result is None:` arm inside "
            "the slot guard (the #1009 lap)",
        )
        none_arm = inner[0]
        self.assertTrue(
            _publishes(ast.Module(body=none_arm.body, type_ignores=[])),
            f"#1189 lap (B) at {MIXIN.name}:{none_arm.lineno}: the #1009 arm "
            f"(`next_batch_result is None`, the slot holds a batch whose pass "
            f"has NOT run) skips the publish, so last_mbs[slot] keeps the "
            f"PREVIOUS lap's EXTEND batch -- the same stale entry lap (A) "
            f"produces, by a second route. The tree's own comment at "
            f"{MIXIN.name}:5050-5076 (working tree, 2026-09-04) documents "
            f"this lap as measured on metal "
            f"(boot 68, 07:50:17). A fix that only adds the outer `else` "
            f"leaves this lap defective.",
        )

    def test_lap_b_publishes_nothing_not_the_unrun_batch(self):
        """The #1009 lap must publish None -- publishing the HELD batch is #969.

        ADDED AFTER A SURVIVING MUTANT, and that is the whole reason it
        exists. `_publishes()` above accepts ANY write to `last_mbs[...]`,
        so it cannot tell `= None` from `= self.mbs[next_mb_id]`. Mutant M6
        (`self.last_mbs[next_mb_id] = self.mbs[next_mb_id]` in the #1009 arm)
        was verified applied on disk and the suite stayed 7/7 GREEN -- a
        measured coverage hole, not a suspicion.

        The mutant is not hypothetical: publishing the held batch on the lap
        where its result has NOT arrived is exactly the #969 CUT L defect the
        comment at :5091-5173 (working tree 2026-09-04) records, where the un-run prefill batch was
        merged into `running_batch` and its requests decoded before they were
        ever prefilled (`causal_conv1d_update: conv_state_indices has 1
        entr(ies) for a batch of 22`; `#1007 DECODE GRAPH REFUSED:
        input_ids=22 but batch_size=1`, boot_969cut_ba2efeb6a7_0829_132233).
        So the two ways to get lap (B) wrong -- not publishing, and
        publishing the wrong value -- are BOTH boot killers, and until now
        only one of them was pinned.
        """
        fn = self.funcs["_event_loop_pp_body"][0]
        guard = [g for g in _slot_guards(fn) if _publishes(g)][0]
        none_arm = _1009_arms(guard)[0]
        writes = [
            n
            for n in ast.walk(ast.Module(body=none_arm.body, type_ignores=[]))
            if isinstance(n, ast.Assign)
            and any(
                isinstance(t, ast.Subscript)
                and isinstance(t.value, ast.Attribute)
                and t.value.attr == PUBLISH_ATTR
                for t in n.targets
            )
        ]
        calls = [
            n
            for n in ast.walk(ast.Module(body=none_arm.body, type_ignores=[]))
            if isinstance(n, ast.Call)
            and isinstance(n.func, ast.Attribute)
            and n.func.attr == PUBLISH_CALL
        ]
        self.assertEqual(
            len(calls),
            0,
            f"#1189/#969 at {MIXIN.name}:{none_arm.lineno}: the #1009 arm "
            f"must NOT call {PUBLISH_CALL}, which publishes "
            f"self.mbs[slot] -- on this lap that batch has NOT RUN, and "
            f"merging it is the #969 CUT L defect.",
        )
        self.assertEqual(
            len(writes),
            1,
            f"expected exactly one last_mbs write in the #1009 arm; "
            f"found {len(writes)}",
        )
        self.assertIn(
            writes[0],
            none_arm.body,
            f"#1189 lap (B) at {MIXIN.name}:{writes[0].lineno}: the "
            f"`last_mbs[...] = None` write is inside the #1009 arm but is NOT "
            f"a direct statement of it -- it sits under a further branch, so "
            f"the arm can be taken with the record left holding the previous "
            f"lap's EXTEND batch and #1189 lap (B) stays open. MEASURED, not "
            f"feared: wrapping this exact line in "
            f"`if getattr(self, '_lapb_enabled', False):` -- behaviourally a "
            f"deletion, which gives 3 failures -- left the whole suite at "
            f"14 passed / 0 failed on 2026-09-04, because `ast.walk` walks "
            f"through the wrapper, the VALUE is still `None`, and the source "
            f"line stays character-identical for the string pin in "
            f"test_every_publish_goes_through_the_one_helper. The executed "
            f"twin of this assertion is LapBOutcomeIsAClearedSlotRecord.",
        )
        value = writes[0].value
        self.assertTrue(
            isinstance(value, ast.Constant) and value.value is None,
            f"#1189/#969 at {MIXIN.name}:{writes[0].lineno}: the #1009 arm "
            f"publishes {ast.unparse(value)!r}, not None. The slot holds a "
            f"batch whose pass has not delivered; the only honest value for "
            f"'what did this slot run' is nothing. Publishing the held batch "
            f"re-creates #969 CUT L -- an un-run prefill batch merged into "
            f"running_batch and decoded before it was prefilled.",
        )

    # ---- controls that are GREEN today and must stay green ---------------

    def test_publish_helper_is_unconditional(self):
        """The helper is NOT the defect. Do not 'fix' it.

        `_pp_record_slot_last_batch` already writes `last_mbs[slot] =
        mbs[slot]` unconditionally, and its docstring already argues why
        ("both loop families now answer 'what did the previous iteration
        run' the same way, and 'nothing' is a real answer rather than a hole
        that preserves the previous answer"). If this test ever goes red, a
        fix moved the guard INTO the helper -- which relocates #1189 rather
        than closing it.
        """
        fn = self.funcs[PUBLISH_CALL][0]
        stmts = [s for s in fn.body if not isinstance(s, ast.Expr)]
        self.assertEqual(
            len(stmts),
            1,
            f"{PUBLISH_CALL} should hold exactly one statement besides its "
            f"docstring; found {len(stmts)}",
        )
        self.assertIsInstance(
            stmts[0],
            ast.Assign,
            f"{PUBLISH_CALL} must publish unconditionally, not under a guard",
        )
        self.assertTrue(_publishes(stmts[0]))
        # AND ITS VALUE, because the lap-(A) outcome is derived from it: the
        # else arm publishes `mbs[slot]` for the slot the guard just proved
        # None. Pinning only "one Assign" leaves that derivation resting on a
        # docstring. Reported precision, since an earlier report quoted this
        # AST census as `['Assign']`: the body is `['Expr', 'Assign']` --
        # docstring plus assignment -- and it is the Assign that is pinned.
        self.assertEqual(
            [type(s).__name__ for s in fn.body],
            ["Expr", "Assign"],
            f"{PUBLISH_CALL} must be a docstring plus exactly one assignment",
        )
        self.assertIsNotNone(
            _helper_value_slot(self.funcs),
            f"{PUBLISH_CALL} must read `self.mbs[<its own slot parameter>]` "
            f"and write `self.last_mbs[<the same parameter>]`; it now reads "
            f"`{ast.unparse(stmts[0])}`. Every lap-(A) outcome assertion in "
            f"this file derives 'the record is CLEARED' from that exact "
            f"shape, so a change here voids them silently.",
        )

    def test_every_publish_goes_through_the_one_helper(self):
        """Both arms of all three guards publish, and only via the helper.

        RENAMED AND REPAIRED 2026-09-04. This test was called
        `test_only_one_call_site_exists_for_the_helper`, its docstring said
        it "pins the count", and its assertion was
        `assertGreaterEqual(len(calls), 1)` -- true for any number at all,
        including the one it claimed to forbid. A control whose name and
        docstring claim more than its assertion checks is the same
        reporting-honesty class as the mutants above, so it is now an exact
        pin with the census printed in its failure message.

        Measured 2026-09-04, after the review-round-2 edits: six call sites
        (:5218, :5244, :5520, :5529, :5726, :5733) -- one in each arm of each
        of the three publishing guards -- plus exactly one remaining raw
        subscript write outside the helper, the lap-(B)
        `self.last_mbs[next_mb_id] = None` at :5174, whose value is pinned
        separately by `test_lap_b_publishes_nothing_not_the_unrun_batch` and
        whose UNCONDITIONALITY is pinned there too (the source string alone
        cannot see a gate around it).
        """
        calls = sorted(
            n.lineno
            for n in ast.walk(self.tree)
            if isinstance(n, ast.Call)
            and isinstance(n.func, ast.Attribute)
            and n.func.attr == PUBLISH_CALL
        )
        self.assertEqual(
            len(calls),
            6,
            f"expected 6 calls to {PUBLISH_CALL}, one per arm of the three "
            f"publishing guards; found {calls}. A fourth loop family, or a "
            f"publish that stopped going through the one writer, both land "
            f"here -- re-derive the census before changing the number.",
        )
        helper = self.funcs[PUBLISH_CALL][0]
        raw = [
            (n.lineno, ast.unparse(n))
            for n in ast.walk(self.tree)
            if isinstance(n, ast.Assign)
            and not (helper.lineno <= n.lineno <= helper.end_lineno)
            and any(
                isinstance(t, ast.Subscript)
                and isinstance(t.value, ast.Attribute)
                and t.value.attr == PUBLISH_ATTR
                for t in n.targets
            )
        ]
        self.assertEqual(
            [src for _, src in raw],
            ["self.last_mbs[next_mb_id] = None"],
            f"outside {PUBLISH_CALL} there must be exactly ONE raw "
            f"`last_mbs[...]` write left -- the lap-(B) `= None`, which "
            f"cannot go through the helper because the helper would publish "
            f"the HELD batch (#969 CUT L). Found {raw}.",
        )
        for guard in [g for g in self._all_slot_guards() if _publishes(g)]:
            self.assertTrue(
                _publishes(ast.Module(body=guard.body, type_ignores=[])),
                f"guard at {MIXIN.name}:{guard.lineno}: the TRUE arm stopped "
                f"publishing",
            )
            self.assertTrue(
                guard.orelse
                and _publishes(ast.Module(body=guard.orelse, type_ignores=[])),
                f"guard at {MIXIN.name}:{guard.lineno}: the FALSE arm does not "
                f"publish -- this is #1189 itself",
            )

    def _all_slot_guards(self):
        out = []
        for fns in self.funcs.values():
            for fn in fns:
                out.extend(_slot_guards(fn))
        return sorted(out, key=lambda g: g.lineno)

    def test_slot_guard_census_no_seventh_site(self):
        """Which `self.mbs[...] is not None` guards carry a publish.

        RE-MEASURED IN THE WORKING TREE 2026-09-04, and the earlier count in
        this docstring was wrong about its own locator. It said "six such
        guards exist, at :4714, :4896, :5046, :5402, :5598, :8806". Running
        `_slot_guards` over the whole file returns FOUR, because two of those
        six do not have the shape this locator matches at all:

          :4714 is not a bare `Compare` -- it is a FIVE-term `if (_row_auth
                and self.ps.pp_size > 1 and self.ps.pp_rank != 0 and
                _pre_proxy is None and self.mbs[mb_id] is not None)`, the #631
                row-authority VOID branch, so the `mbs` read at :4719 is one
                term of a `BoolOp` rather than the whole test. (An earlier
                revision of this docstring said "four-term" and located an
                `IfExp` in a log argument at :4718; both are withdrawn --
                :4718 is `_pre_proxy is None`, and there is no `self.mbs[...]`
                `IfExp` in the `logger.info` at :4726-4732.)
          :4896 is `if _pre_proxy is not None and self.mbs[mb_id] is None:` --
                a `BoolOp` again, and its `mbs` compare is an `Is`, not an
                `IsNot`; the mirror test of a #631 ROW-DELIVER void trace. It
                fails the locator on either count.

        The four the locator does find, by line in the working tree
        (re-measured 2026-09-04 after the review-round-2 edits):

          :5046 `_event_loop_pp_body`            -- publishes. #1189 family.
          :5512 `event_loop_pp_disagg_prefill`   -- publishes. #1189 family.
          :5717 `event_loop_pp_disagg_decode`    -- publishes. #1189 family.
          :8957 `_pp_occupant_horizon_message`   -- indexed by `_s`, a message
                 formatter appending to a local `occupied` list. No publish.

        The count is pinned so a new nested publish cannot be added without
        this suite noticing.
        """
        all_guards = []
        for fns in self.funcs.values():
            for fn in fns:
                for g in _slot_guards(fn):
                    all_guards.append((g.lineno, _publishes(g)))
        publishing = sorted(ln for ln, p in all_guards if p)
        self.assertEqual(
            len(publishing),
            3,
            f"expected exactly 3 publishing slot guards (the #1189 family); "
            f"found {publishing} out of all guards "
            f"{sorted(ln for ln, _ in all_guards)}. A new one means a fourth "
            f"site to convert in the same pass.",
        )


# ---------------------------------------------------------------------------
# THE OUTCOME, EXECUTED. Everything above is a structural argument; this runs
# the real guard and asserts on the STATE it leaves behind.
# ---------------------------------------------------------------------------

_MERGE_FIXTURE = pathlib.Path(__file__).with_name("test_1189_merge_cardinality.py")

# The members ``ScheduleBatch.merge_batch`` grows by exactly one entry per
# merged request, index-parallel to ``reqs``. A dedup on ``reqs`` alone leaves
# these LONGER than ``reqs`` and mis-indexed, so every per-request lookup after
# the merge reads another request's row -- worse than the defect it replaces.
PARALLEL_MEMBERS = (
    ("top_logprobs_nums", "schedule_batch.py:4747/:4750/:4753"),
    ("token_ids_logprobs", "schedule_batch.py:4748/:4751/:4754"),
    ("multimodal_inputs", "schedule_batch.py:4757"),
)


def _load_merge_fixture():
    """The faithful ``merge_batch`` stand-in, borrowed rather than re-typed.

    ``test_1189_merge_cardinality.py`` already builds a ``self``/``other``
    stand-in and AST-checks it against every ``self.<attr>``/``other.<attr>``
    the shipping method touches. A second stub here would be a second
    bookkeeping of the same fixture and could drift into certifying its own
    assumptions (#630), so this loads that one by path.
    """
    spec = importlib.util.spec_from_file_location(
        "_weg1_1189_merge_fixture", _MERGE_FIXTURE
    )
    if spec is None or spec.loader is None:  # pragma: no cover - path guard
        raise unittest.SkipTest(f"merge fixture not found at {_MERGE_FIXTURE}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _compiled_lap(fn, guard):
    """Compile the REAL guard statement into a runnable ``_visit``.

    NOT a re-implementation of the loop. The ``ast.If`` node is taken verbatim
    out of the shipping source, only its TRUE arm replaced by ``pass``, and
    wrapped in ``def _visit(self, mb_id, next_mb_id)``. So the guard test, the
    presence or absence of the ``else``, the slot the else names and the value
    it publishes are all production code, executed.

    BOUND, stated so it is not over-read: this decides the guard-FALSE lap
    ONLY. The true arm needs torch, ``torch.profiler``, a D2H event and the
    whole PP request chain, and is covered structurally by
    ``test_lap_b_publishes_nothing_not_the_unrun_batch``. A revert of the fix
    leaves the ``If`` with no ``orelse``, so ``_visit`` becomes a no-op -- the
    defect itself, running.
    """
    node = copy.deepcopy(guard)
    node.body = [ast.Pass()]
    func = ast.FunctionDef(
        name="_visit",
        args=ast.arguments(
            posonlyargs=[],
            args=[ast.arg(arg="self"), ast.arg(arg="mb_id"), ast.arg(arg="next_mb_id")],
            vararg=None,
            kwonlyargs=[],
            kw_defaults=[],
            kwarg=None,
            defaults=[],
        ),
        body=[node],
        decorator_list=[],
        returns=None,
        type_params=[],
    )
    module = ast.Module(body=[func], type_ignores=[])
    ast.fix_missing_locations(module)
    ns = {}
    exec(  # noqa: S102 - the source is this repository's own shipping file
        compile(module, f"<lap-A slice of {MIXIN.name}:{guard.lineno}>", "exec"), ns
    )
    return ns["_visit"]


class _Sched:
    """The two rows the lap touches, plus the REAL one writer."""

    _pp_record_slot_last_batch = SchedulerPPMixin.__dict__[PUBLISH_CALL]

    def __init__(self, mbs, last_mbs):
        self.mbs = list(mbs)
        self.last_mbs = list(last_mbs)


class _OutcomeFixture(unittest.TestCase):
    """Shared ground for both executed outcome classes -- ONE bookkeeping.

    Lap (A) and lap (B) assert the same state property on two different laps
    and hand the result to the same shipping consumer. A second copy of this
    fixture would be a second bookkeeping of it and could drift into
    certifying its own assumptions (#630), which is the same reason
    ``_load_merge_fixture`` borrows the merge stand-in rather than re-typing
    it. Carries no tests of its own.

    Everything here is counted in DISTINCT rids and in per-rid multiplicity
    -- never in ``len(reqs)`` alone, because the #1189 growth is duplicate
    rids inside ONE batch, so the distinct set stays flat while the list runs
    away, and a bad dedup can hold the list flat while losing a request.
    """

    SLOT = 1  # the slot under visit
    OTHER = 0  # a neighbouring slot, so a write to the wrong slot shows

    @classmethod
    def setUpClass(cls):
        cls.fx = _load_merge_fixture()
        cls.tree = _tree()
        cls.funcs = _funcs(cls.tree)

    def _batch(self, tag, n, *, extend=False):
        reqs = [self.fx._Req(f"{tag}{i}") for i in range(n)]
        b = self.fx._batch(reqs, tag=tag)
        if extend:
            b.forward_mode = types.SimpleNamespace(is_extend=lambda: True)
        return b

    def _publishing_guard(self, fname):
        fn = self.funcs[fname][0]
        guards = [g for g in _slot_guards(fn) if _publishes(g)]
        self.assertEqual(
            len(guards),
            1,
            f"{fname}: expected exactly one publishing slot guard; found "
            f"{[g.lineno for g in guards]}",
        )
        return fn, guards[0]

    @staticmethod
    def _consume(running, last):
        """``scheduler.py:8334-8338`` and ``:8396``, the reachable part.

        The gate is `not self.enable_hisparse and last_batch and
        last_batch.forward_mode.is_extend()`; the merge is
        `running_batch.merge_batch(last_batch)`, with a self-merge refusal in
        between at :8362. Nothing else on that path is reachable from a lap
        that published for one slot.
        """
        if last is not None and last.forward_mode.is_extend() and last is not running:
            ScheduleBatch.merge_batch(running, last)
        return running

    @staticmethod
    def _rid_counts(batch):
        counts = {}
        for req in batch.reqs:
            counts[req.rid] = counts.get(req.rid, 0) + 1
        return counts


class LapAOutcomeIsAClearedSlotRecord(_OutcomeFixture):
    """After a visit to a slot that ran nothing, the record must be CLEARED.

    The structural tests above say a publish happens and name its slot and
    value. This one executes the guard and reads the state afterwards, then
    hands that state to the shipping consumer.
    """

    def _visit_empty_slot(self, fname):
        """Run the real guard-False lap of ``fname`` and return the scheduler."""
        fn, guard = self._publishing_guard(fname)
        held = self._batch("held", 2)
        stale = self._batch("stale", 3, extend=True)
        sched = _Sched(
            mbs=[held, None],  # SLOT holds nothing; OTHER holds a batch
            last_mbs=[None, stale],  # SLOT carries the previous lap's EXTEND
        )
        _compiled_lap(fn, guard)(sched, self.OTHER, self.SLOT)
        return sched, held, stale

    def _assert_cleared(self, fname):
        sched, held, stale = self._visit_empty_slot(fname)
        self.assertIsNone(
            sched.last_mbs[self.SLOT],
            f"#1189 lap (A) OUTCOME, {fname}: after a visit to slot "
            f"{self.SLOT} which holds NO batch, last_mbs[{self.SLOT}] is "
            f"{sched.last_mbs[self.SLOT]!r}, not None. The record still names "
            f"a batch this slot did not run, so the next visit merges it "
            f"again. Three ways to land here, all measured: the else arm is "
            f"missing (the pre-fix tree); it publishes another slot (mutant "
            f"R1); or it publishes another slot's batch (mutant R2).",
        )
        self.assertIsNone(
            sched.last_mbs[self.OTHER],
            f"#1189 lap (A) OUTCOME, {fname}: the lap wrote "
            f"last_mbs[{self.OTHER}] = {sched.last_mbs[self.OTHER]!r} while "
            f"visiting slot {self.SLOT}. A lap may only speak for the slot it "
            f"visited; slot {self.OTHER} still holds a batch and its record "
            f"now names one that was consumed earlier in this same iteration "
            f"(mutant R1).",
        )

    def test_outcome_event_loop_pp_body(self):
        self._assert_cleared("_event_loop_pp_body")

    def test_outcome_event_loop_pp_disagg_prefill(self):
        self._assert_cleared("event_loop_pp_disagg_prefill")

    def test_outcome_event_loop_pp_disagg_decode(self):
        self._assert_cleared("event_loop_pp_disagg_decode")

    # ---- what the consumer then does with that record --------------------

    def test_a_later_visit_does_not_grow_the_merged_request_set(self):
        """The cleared record makes the next visit a no-op for the consumer.

        This is the acceptance shape of #1189 stated as an outcome: the
        request set the scheduler is running must be the SAME set, by rid and
        by multiplicity, after a lap that ran nothing.
        """
        sched, _, stale = self._visit_empty_slot("_event_loop_pp_body")
        running = self._batch("run", 3)
        # The defect shape: a DISTINCT batch object holding the SAME Req
        # objects, which is what defeats both existing defences (the
        # `last_batch is running_batch` identity guard and harvest's dedup by
        # id(batch)) at once.
        stale.reqs = list(running.reqs)
        before = self._rid_counts(running)
        self._consume(running, sched.last_mbs[self.SLOT])
        after = self._rid_counts(running)
        self.assertEqual(
            after,
            before,
            f"#1189 OUTCOME: a visit to an empty slot grew the running "
            f"request set from {before} to {after}. Counted in DISTINCT rids "
            f"AND their multiplicity, because the growth is the SAME rid "
            f"appended again inside ONE batch (`merge_batch` extends `reqs` "
            f"IN PLACE), so the distinct SET alone cannot see it -- boot 8 "
            f"reached `running=7768` against max_running_requests=8 "
            f"(`boot_855_weg1b8_e9d1a719ac_0904_064622.log:458441`, a `#788 "
            f"PP-ADMISSION verdict=DECLINE` line, and the max of that "
            f"instrument's series). NAMED BY INSTRUMENT: `running_bs=7768` "
            f"occurs ZERO times in that log -- its `running_bs` series tops "
            f"out at 7771.",
        )
        self.assertEqual(
            max(after.values()),
            1,
            f"no rid may appear twice in one batch; got {after}",
        )

    def test_a_retained_record_is_what_this_probe_catches(self):
        """CAN-FAIL PROOF. Feed the probe the pre-fix outcome; it must catch it.

        Without this, the test above is satisfied by a probe that cannot
        detect anything at all (desk-written-never-executed). The retained
        record is constructed directly here rather than by mutating the tree,
        so the proof travels with the file.

        It also shows why the assertion is on rids: `len(reqs)` grows from 3
        to 6 here, but the DISTINCT rid set is 3 before and 3 after. A probe
        reading list length alone reports growth; a probe reading the distinct
        set alone reports none. Only the multiplicity says what happened.
        """
        running = self._batch("run", 3)
        stale = self._batch("stale", 3, extend=True)
        stale.reqs = list(running.reqs)
        before = self._rid_counts(running)
        before_len = len(running.reqs)
        self._consume(running, stale)  # the record was NOT cleared
        after = self._rid_counts(running)
        self.assertNotEqual(
            after,
            before,
            "the probe cannot see a retained record: merging a distinct batch "
            "holding the same Reqs left the rid multiset unchanged. Either "
            "merge_batch grew a dedup (then #1189's behavioural half moved) "
            "or this fixture stopped reaching it.",
        )
        self.assertEqual(sorted(after), sorted(before), "distinct rid SET moved")
        self.assertEqual(max(after.values()), 2, f"expected each rid twice: {after}")
        self.assertEqual(len(running.reqs), before_len * 2)

    def test_index_parallel_siblings_stay_parallel_across_the_merge(self):
        """A dedup on ``reqs`` alone is WORSE than the defect it replaces.

        ``merge_batch`` maintains lists index-parallel to ``reqs``. If a
        "fix" for #1189 filters ``reqs`` without filtering these, they stay
        LONGER than ``reqs`` and every per-request lookup after the merge
        reads another request's row -- a silent wrong answer instead of a
        loud ramp. Asserted as exact CONCATENATION, not just equal widths, so
        a filter that keeps the widths equal by dropping the wrong entries
        also lands here.
        """
        running = self._batch("run", 3)
        stale = self._batch("stale", 3, extend=True)
        stale.reqs = list(running.reqs)
        expect = {
            name: list(getattr(running, name)) + list(getattr(stale, name))
            for name, _ in PARALLEL_MEMBERS
        }
        expect_reqs = [r.rid for r in running.reqs] + [r.rid for r in stale.reqs]
        self._consume(running, stale)
        for name, anchor in PARALLEL_MEMBERS:
            got = getattr(running, name)
            self.assertEqual(
                len(got),
                len(running.reqs),
                f"{name} ({anchor}) is {len(got)} entries against "
                f"{len(running.reqs)} reqs after the merge. Index-parallel "
                f"state that outgrows `reqs` mis-indexes every later "
                f"per-request lookup -- the failure mode a reqs-only dedup "
                f"introduces while appearing to fix #1189.",
            )
            self.assertEqual(
                got,
                expect[name],
                f"{name} ({anchor}) is no longer the concatenation of the two "
                f"inputs in order; entries were dropped or reordered while "
                f"`reqs` was not, so the two are mis-paired.",
            )
        self.assertEqual([r.rid for r in running.reqs], expect_reqs)


class _NullLogger:
    """The #1009 arm emits a rate-limited warning; the assertion is on STATE.

    Substituted for the module's ``logger`` in the compiled slice below, so
    the arm's own logging cannot decide whether the test passes.
    """

    def warning(self, *args, **kwargs):
        pass


def _compiled_lap_b(none_arm):
    """Compile the REAL #1009 arm into a runnable ``_visit_lap_b``.

    The exact mirror of ``_compiled_lap``: the ``ast.If`` node is taken
    verbatim out of the shipping source and only the arm NOT under test is
    replaced by ``pass`` -- here the ``else``, which needs torch,
    ``torch.profiler``, a D2H event and the whole PP request chain. The test,
    the arm's statements, the slot it names and the value it publishes are
    all production code, executed.

    BOUND, stated so it is not over-read: this decides the lap where
    ``next_batch_result is None``, and nothing else. ``logger`` is the one
    module global the arm reaches and it is stubbed. A revert of the fix
    leaves the arm without the write, and a GATE around the write leaves it
    without an effective one -- either way ``last_mbs[slot]`` keeps the stale
    entry: the defect itself, running.
    """
    node = copy.deepcopy(none_arm)
    node.orelse = [ast.Pass()]
    func = ast.FunctionDef(
        name="_visit_lap_b",
        args=ast.arguments(
            posonlyargs=[],
            args=[
                ast.arg(arg="self"),
                ast.arg(arg="next_mb_id"),
                ast.arg(arg="next_batch_result"),
            ],
            vararg=None,
            kwonlyargs=[],
            kw_defaults=[],
            kwarg=None,
            defaults=[],
        ),
        body=[node],
        decorator_list=[],
        returns=None,
        type_params=[],
    )
    module = ast.Module(body=[func], type_ignores=[])
    ast.fix_missing_locations(module)
    ns = {"logger": _NullLogger()}
    exec(  # noqa: S102 - the source is this repository's own shipping file
        compile(module, f"<lap-B slice of {MIXIN.name}:{none_arm.lineno}>", "exec"), ns
    )
    return ns["_visit_lap_b"]


class LapBOutcomeIsAClearedSlotRecord(_OutcomeFixture):
    """After a lap whose pass has NOT delivered, the record must be CLEARED.

    THE LAP: the slot HOLDS a batch and ``next_batch_result is None``, so its
    pass has not delivered. Two ways to get it wrong, both measured boot
    killers: not clearing the record (#1189 -- the previous lap's EXTEND entry
    survives and ``get_next_batch_to_run`` re-merges it on every later visit)
    and publishing the HELD batch (#969 CUT L -- an un-run prefill batch
    merged into ``running_batch`` and decoded before it was ever prefilled).

    WHY THIS CLASS EXISTS, MEASURED RATHER THAN FEARED. Until it was written,
    lap (B) had NO executed coverage at all: ``_compiled_lap`` replaces the
    guard's TRUE arm -- where this lap lives -- with ``pass``. Its three
    structural pins are all conditionality-blind: ``_publishes()`` accepts any
    write, ``test_lap_b_publishes_nothing_not_the_unrun_batch`` reads the
    Assign's VALUE, and ``test_every_publish_goes_through_the_one_helper``
    pins the SOURCE STRING ``self.last_mbs[next_mb_id] = None``, which a
    never-true wrapper leaves character-identical. Measured 2026-09-04 on a
    full scratch copy of this tree (``/tmp/weg1_mut``; golden
    ``scheduler_pp_mixin.py`` md5 ``33901a5d10fc251d0f3708716eb6fb24``, mutant
    verified applied by unified diff before the run, ``sglang.__file__``
    checked to resolve inside the scratch tree): wrapping that write in ``if
    getattr(self, '_lapb_enabled', False):`` -- behaviourally identical to
    DELETING it, which gives 3 failures -- left the whole suite at
    14 passed / 0 failed. Lap (A) had the same hole; it is closed by
    ``LapAOutcomeIsAClearedSlotRecord`` and by the ``unconditional`` term of
    ``_lap_a_publish_effect``. This class plus the ``assertIn(writes[0],
    none_arm.body)`` pin above is the lap-(B) equivalent.

    The consumer half is shared with lap (A), including its can-fail proof
    (``test_a_retained_record_is_what_this_probe_catches``): the probe is
    shown to catch a RETAINED record before it is used to certify a cleared
    one.
    """

    def _visit_unrun_slot(self):
        """Run the real #1009 arm and return the state it left behind.

        Slot layout: ``SLOT`` holds a batch whose pass has not run (that is
        what makes this lap (B) rather than lap (A)) and carries the previous
        lap's EXTEND entry in ``last_mbs``; ``OTHER`` carries an entry of its
        own, so a write to the wrong slot shows.
        """
        _fn, guard = self._publishing_guard("_event_loop_pp_body")
        arms = _1009_arms(guard)
        self.assertEqual(
            len(arms),
            1,
            "expected exactly one `if next_batch_result is None:` arm inside "
            "the slot guard (the #1009 lap)",
        )
        held = self._batch("held", 2, extend=True)
        stale = self._batch("stale", 3, extend=True)
        other_stale = self._batch("othstale", 2, extend=True)
        sched = _Sched(
            mbs=[self._batch("other", 2), held],
            last_mbs=[other_stale, stale],
        )
        _compiled_lap_b(arms[0])(sched, self.SLOT, None)
        return sched, held, stale, other_stale

    def test_outcome_the_1009_lap_clears_the_record(self):
        sched, _held, _stale, _other = self._visit_unrun_slot()
        self.assertIsNone(
            sched.last_mbs[self.SLOT],
            f"#1189 lap (B) OUTCOME: after a visit to slot {self.SLOT}, whose "
            f"held batch's pass has NOT delivered, last_mbs[{self.SLOT}] is "
            f"{sched.last_mbs[self.SLOT]!r}, not None. Two ways to land here, "
            f"both boot killers: the write is missing OR GATED, so the record "
            f"keeps the previous lap's EXTEND batch and the consumer re-merges "
            f"it on every later visit (#1189); or the write publishes the HELD "
            f"batch, which is then merged into running_batch and decoded "
            f"before it was ever prefilled (#969 CUT L). The structural pins "
            f"above cannot separate either from the fix.",
        )

    def test_outcome_the_lap_speaks_only_for_its_own_slot(self):
        sched, _held, _stale, other_stale = self._visit_unrun_slot()
        self.assertIs(
            sched.last_mbs[self.OTHER],
            other_stale,
            f"#1189 lap (B) OUTCOME: visiting slot {self.SLOT} rewrote "
            f"last_mbs[{self.OTHER}]. A lap may only speak for the slot it "
            f"visited; the neighbouring slot's record is not this lap's to "
            f"answer for.",
        )

    def test_outcome_the_unrun_batch_stays_in_its_slot(self):
        """"Nothing is dropped" -- the arm's own claim, pinned.

        The #1009 arm skips result processing and clears the RECORD; it must
        not also empty the SLOT, or the held batch would never be processed
        when its result lands -- the failure the arm exists to avoid.
        """
        sched, held, _stale, _other = self._visit_unrun_slot()
        self.assertIs(
            sched.mbs[self.SLOT],
            held,
            f"#1189/#1009 lap (B) OUTCOME: the lap emptied mbs[{self.SLOT}]. "
            f"Clearing the RECORD is the fix; clearing the SLOT drops a batch "
            f"whose pass is still in flight.",
        )

    def test_a_later_visit_does_not_grow_the_merged_request_set(self):
        """The cleared record makes the next visit a no-op for the consumer.

        The lap-(B) twin of the lap-(A) test of the same name, and the same
        acceptance shape: the request set the scheduler is running must be the
        SAME set, by rid AND by multiplicity, after a lap that ran nothing.
        """
        sched, _held, stale, _other = self._visit_unrun_slot()
        running = self._batch("run", 3)
        # The defect shape: a DISTINCT batch object holding the SAME Req
        # objects, which defeats the `last_batch is running_batch` identity
        # guard and harvest's dedup by id(batch) at once.
        stale.reqs = list(running.reqs)
        before = self._rid_counts(running)
        self._consume(running, sched.last_mbs[self.SLOT])
        after = self._rid_counts(running)
        self.assertEqual(
            after,
            before,
            f"#1189 lap (B) OUTCOME: a visit to a slot whose pass had not "
            f"delivered grew the running request set from {before} to "
            f"{after}. Counted in DISTINCT rids and their multiplicity, "
            f"because the growth is the SAME rid appended again inside ONE "
            f"batch -- the distinct SET alone would have looked flat "
            f"throughout.",
        )
        self.assertEqual(
            max(after.values()),
            1,
            f"no rid may appear twice in one batch; got {after}",
        )


if __name__ == "__main__":
    unittest.main()
