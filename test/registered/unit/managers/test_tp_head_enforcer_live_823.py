# Copyright 2023-2024 SGLang Team
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""#823 W9b: the enforcer was WIRED and INERT on metal. Close both roots.

WHY THIS FILE EXISTS AND `test_tp_head_congruence_823.py` DID NOT CATCH IT.
That suite drives the DECISION -- pure functions, real gloo reduces, fifteen
green cases -- and every one of them constructs its inputs directly. None of
them puts a Scheduler into the state the production pass is actually in when
the decision is consumed. So the decision was proven correct and the
consumption was never exercised, and the boot said so 105 times:

    #823 head-congruence: could not apply the group head order
    ('Scheduler' object has no attribute '_uniform_prefetch_ballot');
    this pass forms rank-locally

(boot_window2_0823_1554.log, all three ranks, first admission pass 15:58:43
to last 15:59:58; specimen
/spinning/evidence-665-f1/SPECIMEN_window2_w9_inert_and_631_death_0823.txt)

ROOT 1 -- A LIFECYCLE, NOT A MISSING DEFAULT.
`_get_new_batch_prefill_raw` POPS `_uniform_prefetch_ballot` off `__dict__`
(scheduler.py:7429, #791b's deliberate consume-once discipline) and then,
55 lines further down the SAME function, calls `_apply_uniform_head_order`,
which read that same attribute directly (scheduler.py:5314). Pop, then read:
the read raises on every pass, the handler catches it, and the fallback is
the rank-local formation this ticket states it had eliminated.

THE OBVIOUS FIX IS THE WRONG ONE, and it is banned here explicitly. Making
:5314 match its three `getattr(..., None)` neighbours (:5286, :5304, :5305)
stops the exception and produces the SECOND silent inertness -- which is
exactly what the COUNT arm already had: `_uniform_allocatable_reqs` reads
its memo through `getattr(..., None)` and falls back to the local number
with NO LOG LINE AT ALL. The order arm at least announced its own inertness
105 times; the count arm's was invisible, so a future boot showing zero #823
lines still says nothing about it. That is the #606 getattr-default family:
a default that turns a structural defect into a plausible number.

So the shape of the fix is fixed by the defect:
  * the group decision must EXIST at the consumption point by construction,
    not by a default -- it is taken ONCE per pass, at one point, and handed
    to the consumers as a VALUE (a parameter cannot be missing);
  * and every degradation must be LOUD. Under
    `SGLANG_TP_HEAD_CONGRUENCE=1` a rank-local fallback is a DEFECT, so it
    carries a named counter and a rate-limited log on both arms, plus the
    recovery edge -- the half a latch can never report.

ROOT 2 -- THE GATE. FALSIFIED AS A DIVERGENCE, KEPT AS A GATE QUESTION.
The window recorded "45 of 51 prefill batches ran in the PP phase, where the
enforcer is gated off by `ps.tp_size > 1`, and 7 of 12 comparable passes
diverged there". The first half is true. The second half is a MEASUREMENT
ARTEFACT and this file records the falsification, because a fix built on it
would have been a fix for nothing:

    The `Prefill batch phase=pp` log line carries no mb_id, and under PP the
    three stages sit at different microbatch offsets BY DESIGN (#737,
    scheduler.py:7402-7419). The divergence table paired those lines by
    WALL-CLOCK SECOND -- and in both seconds it names, two ranks emitted TWO
    prefill batches. 15:59:18 has PP0 at log lines 1962 AND 1973; the table
    took PP1's first line and PP0's second. Line 1962 (PP0, 1/0/2) and line
    1971 (PP1, 1/0/2) agree exactly.

    Compared properly -- the ORDERED sequence of
    (#new-seq, #new-token, #cached-token) over each rank's own 15 PP-phase
    prefill batches -- the three ranks are BYTE-IDENTICAL:
        1:122:0 1:13:0 2:2492:0 2:4096:0 1:4096:0 1:4096:0 1:4096:0
        2:4096:0 2:4096:0 1:4096:0 1:4096:0 1:4096:0 2:3333:0 2:2492:0
        1:1246:0
    on PP0, PP1 and PP2 alike, with zero `#791 FORWARDED SCHEDULE
    UNEXECUTABLE` and zero `pass_voided` in the whole boot.

That is the INDIKATOR-GESETZ in one specimen: the indicator was never tested
against a known state, and it did not measure what it claimed.

AND THE CODE SAYS THE GATE IS RIGHT. Three independent reasons, each
anchored, for NOT widening `_tp_head_enforcer_enabled` to the PP phase:

  1. Under `--tp-size 1 --pp-size 3` the enforcer's own reduce group,
     `tp_cpu_group`, has EXACTLY ONE MEMBER per rank
     (parallel_state.py:3166-3188 chunks the world into `world_size //
     tp_size` groups of size 1). A MIN-reduce over a singleton returns this
     rank's own numbers. Widening the gate without a new collective would
     make each rank enforce its OWN verdict while calling it the group's --
     strictly worse than off, because it would also be silent.
  2. Supplying that missing collective at the consumption point is
     forbidden, with a measured casualty. #737 (scheduler.py:7402-7419):
     "A collective placed here therefore requires an alignment this position
     cannot supply. The HiCache ack-count reduction leaned on this comment
     and deadlocked on 2026-08-17 (PP0/PP1 in the drain, PP2 in the pipeline
     recv)."
  3. The PP phase ALREADY HAS this actuator, and a stronger one. #791
     forwards PP0's committed decision around the ring and downstream ranks
     EXECUTE it: membership divergence raises `PPScheduleRefused` in both
     directions (scheduler.py:7903-7941, "EVERY NAMED REQUEST, OR NONE OF
     THEM"), and order is re-imposed as a total permutation by
     `order_batch_by_schedule` (scheduler.py:7946-7949). `tp_head_congruence`
     itself says where its rule came from: "THE RULE, transplanted from
     #791" (tp_head_congruence.py:48-62).

So the gate stays, and what changes is that it stops being SILENT: the
enforcer now reports WHY it is off, once per transition, naming #791 as the
mechanism that covers the PP phase. "Inert" and "correctly gated off" read
identically in a log that says neither, and that ambiguity cost this ticket
a whole GPU window.

CPU-only. No CUDA, no collectives, no server.
"""

import ast
import inspect
import textwrap
import unittest
from types import SimpleNamespace
from typing import Optional

from sglang.srt.managers import tp_head_congruence as thc
from sglang.srt.managers.scheduler import Scheduler

# ---------------------------------------------------------------------------
# Part 1 -- ROOT 1 as an executable invariant. THIS IS THE RED-FIRST CASE.
# ---------------------------------------------------------------------------
#
# Source-level on purpose, and for the reason the sibling guard in
# test_collective_family_siblings_610.py gives for its own extractor: the
# drift being caught is "this pass destroys an attribute and something
# downstream still reads it", and that is a property of the METHOD'S TEXT.
# Executing the methods to find out would need the very Scheduler whose
# construction is impossible at the desk.


def _parse(func) -> ast.AST:
    return ast.parse(textwrap.dedent(inspect.getsource(func)))


# AST AND NOT A REGEX, and the difference is not fastidiousness -- the regex
# version of this guard, written first, went red on the FIXED tree. The
# repaired `_apply_uniform_head_order` carries a comment naming the defect it
# replaced ("`digest_agreed=self._uniform_prefetch_ballot is not None` did
# this on every one of the 105 passes"), and a text-level extractor cannot
# tell that sentence from a live read. A guard that punishes a comment for
# quoting the bug it fixed teaches people to delete the explanation, which is
# the opposite of what this file is for. `ast` sees loads and calls, and
# prose is invisible to it.


def _self_attributes_read_by(func) -> set:
    """Names this function LOADS off ``self`` (stores excluded)."""
    return {
        node.attr
        for node in ast.walk(_parse(func))
        if isinstance(node, ast.Attribute)
        and isinstance(node.value, ast.Name)
        and node.value.id == "self"
        and isinstance(node.ctx, ast.Load)
    }


def _self_attributes_consumed_by(func) -> set:
    """Names this function REMOVES from ``self.__dict__`` for the pass.

    The consume-once discipline #791b introduced, made machine-readable.
    Anything in here is guaranteed absent for the remainder of the pass, so
    anything that reads it afterwards reads a hole.
    """
    consumed = set()
    for node in ast.walk(_parse(func)):
        if not isinstance(node, ast.Call):
            continue
        fn = node.func
        if not (isinstance(fn, ast.Attribute) and fn.attr == "pop"):
            continue
        owner = fn.value
        if not (
            isinstance(owner, ast.Attribute)
            and owner.attr == "__dict__"
            and isinstance(owner.value, ast.Name)
            and owner.value.id == "self"
        ):
            continue
        if node.args and isinstance(node.args[0], ast.Constant):
            consumed.add(str(node.args[0].value))
    return consumed


def _methods_called_on_self(func) -> set:
    return {
        node.func.attr
        for node in ast.walk(_parse(func))
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and isinstance(node.func.value, ast.Name)
        and node.func.value.id == "self"
    }


def _pass_consumed_attributes() -> set:
    """Everything the prefill pass consumes, one level down.

    One level down through Scheduler methods the pass calls, which is the
    same scope rule the #610 harness guard uses -- and it is the level the
    fix moves the pops to, so a guard that only looked at the top frame
    would go green on the refactor without checking anything.
    """
    func = Scheduler._get_new_batch_prefill_raw
    consumed = _self_attributes_consumed_by(func)
    for name in sorted(_methods_called_on_self(func)):
        member = getattr(Scheduler, name, None)
        if callable(member):
            consumed |= _self_attributes_consumed_by(member)
    return consumed


#: The two methods that turn the group's decision into this rank's batch.
#: Named explicitly rather than discovered, so adding a third consumer is a
#: deliberate act that has to be written down here.
_DECISION_CONSUMERS = (
    Scheduler._apply_uniform_head_order,
    Scheduler._uniform_allocatable_reqs,
)


class TheGroupDecisionSurvivesTheConsumeOncePass(unittest.TestCase):
    """ROOT 1, exactly as the boot log stated it.

    RED ON THE UNFIXED TREE, naming `_uniform_prefetch_ballot` -- which is
    the attribute the 105 log lines named. This is the window's finding as a
    predicate rather than as a paragraph.
    """

    def test_no_consumer_reads_an_attribute_the_pass_already_consumed(self):
        consumed = _pass_consumed_attributes()
        read = set()
        for consumer in _DECISION_CONSUMERS:
            read |= _self_attributes_read_by(consumer)
        collisions = sorted(consumed & read)
        self.assertEqual(
            collisions,
            [],
            "#823 W9b: the prefill pass consumes "
            f"{sorted(consumed)} off self.__dict__, and the batch-formation "
            f"consumers still read {collisions} afterwards. Every such read "
            "raises AttributeError on every pass and lands in the rank-local "
            "fallback -- the 105 '#823 head-congruence: could not apply the "
            "group head order' lines of boot_window2_0823_1554. Hand the "
            "decision to the consumers as a VALUE taken once per pass; do "
            "NOT paper the read over with getattr(..., None), which is the "
            "#606 family and is how the COUNT arm went silently inert.",
        )

    def test_the_guard_can_fail(self):
        """A drift detector that cannot report a drift is decoration.

        Stands in for the next incident: a memo the pass consumes and a
        consumer that still reads it. Built from synthetic sets rather than
        by adding a name to the production ones, so this arm reports the
        planted collision ALONE -- on the unfixed tree the production sets
        already collide, and an arm that mixed the two would report the real
        defect here as well and hide the fact that the detector works.
        """
        consumed = {"_a_memo_popped_next_quarter", "_a_memo_nobody_reads"}
        read = {"_a_memo_popped_next_quarter", "_an_attribute_never_popped"}
        self.assertEqual(sorted(consumed & read), ["_a_memo_popped_next_quarter"])

    def test_the_extractor_really_sees_the_production_pop(self):
        """Pin the extractor against a known state, both directions.

        A guard whose extractor silently matched nothing would pass this
        file forever. The prefill pass provably pops at least the prefetch
        verdicts memo (#791b, scheduler.py:7426), so an empty result here
        means the regex stopped matching, not that the tree got clean.
        """
        consumed = _pass_consumed_attributes()
        self.assertIn("_pass_prefetch_verdicts", consumed)
        self.assertNotIn(
            "_pass_prefetch_verdicts",
            _self_attributes_read_by(Scheduler._apply_uniform_head_order),
        )

    def test_the_guard_follows_the_pop_one_level_down(self):
        """The scope rule, pinned -- a guard that only read the top frame
        would go green on this ticket's own refactor without checking it.

        W9b moved the group verdict's pop OUT of `_get_new_batch_prefill_raw`
        and into `_take_uniform_head_inputs`, which that function calls. If
        the extractor stopped at the top frame it would no longer see
        `_uniform_head_inputs` being consumed at all, and the invariant would
        hold vacuously for exactly the attribute this ticket is about.
        """
        self.assertIn("_uniform_head_inputs", _pass_consumed_attributes())
        self.assertNotIn(
            "_uniform_head_inputs",
            _self_attributes_consumed_by(Scheduler._get_new_batch_prefill_raw),
            "if this ever becomes a top-frame pop again, the one-level-down "
            "descent stops being the thing under test here",
        )


if __name__ == "__main__":
    unittest.main()


# ---------------------------------------------------------------------------
# Part 2 -- the same defect, driven through the real consumers.
# ---------------------------------------------------------------------------
#
# Part 1 is the red-first case and it is a source-level invariant. This part
# is the behavioural companion: it puts a Scheduler surface into the state
# the production pass is REALLY in at the moment the decision is consumed --
# every consume-once memo of the pass already popped -- and drives the two
# real methods. It is stated plainly that this part could not run on the
# unfixed tree, because the consumers took no decision parameter there and
# there was nothing to hand them; Part 1 is what carries the red.


class _PassHarness:
    """A Scheduler surface with the four members the two arms actually touch.

    Bound off ``Scheduler`` rather than reimplemented, so the arms under test
    are the production ones. Kept to exactly what they read -- a harness that
    binds more than the methods reach is a harness that fails on state the
    test legitimately does not model (the scope rule
    ``test_collective_family_siblings_610.py`` states for its own stub).
    """

    _apply_uniform_head_order = Scheduler._apply_uniform_head_order
    _uniform_allocatable_reqs = Scheduler._uniform_allocatable_reqs
    _take_uniform_head_inputs = Scheduler._take_uniform_head_inputs
    _tp_head_enforcer_gate = Scheduler._tp_head_enforcer_gate
    _tp_head_enforcer_enabled = Scheduler._tp_head_enforcer_enabled
    _note_tp_head_degradation = Scheduler._note_tp_head_degradation

    def __init__(self, rids, tp_size=3, pp_size=1, allocatable=99):
        self.ps = SimpleNamespace(tp_size=tp_size, pp_size=pp_size)
        self.waiting_queue = [SimpleNamespace(rid=r) for r in rids]
        self._allocatable = allocatable
        self._tp_head_count_degraded_this_pass = False

    def get_num_allocatable_reqs(self, running_bs):
        return self._allocatable

    def publish(self, canonical, group_lens, admit_limit, digest_agreed=True):
        """What ``_update_uniform_pool_budget`` leaves behind for the pass."""
        self._uniform_prefetch_ballot = {} if digest_agreed else None
        self._uniform_head_inputs = thc.build_uniform_head_inputs(
            canonical, group_lens, admit_limit, digest_agreed
        )

    def run_pass_prologue(self) -> Optional[thc.UniformHeadInputs]:
        """The prefill pass's consume-once block, in production order.

        THIS IS THE STATE THE 105 LOG LINES CAME FROM. The ballot is popped
        here, exactly as ``_get_new_batch_prefill_raw`` pops it, BEFORE the
        decision is taken -- so if the decision ever goes back to reading it,
        this harness is in the state that breaks.
        """
        self.__dict__.pop("_pass_prefetch_verdicts", None)
        self.__dict__.pop("_uniform_prefetch_ballot", None)
        return self._take_uniform_head_inputs()

    def rids(self):
        return [req.rid for req in self.waiting_queue]


#: Four requests; the group's MIN-reduced match lengths put them in a
#: specific order that is NOT the queue order they arrive in, so "the order
#: changed" cannot be satisfied by doing nothing.
_CANON = ["r-alpha", "r-bravo", "r-charlie", "r-delta"]
_GROUP_LENS = [16, 4, 4096, 0]
_EXPECTED_GROUP_ORDER = ["r-charlie", "r-alpha", "r-bravo", "r-delta"]


class TheOrderArmActsOnAPassThatAlreadyPoppedTheBallot(unittest.TestCase):
    def test_the_group_order_is_applied(self):
        h = _PassHarness(["r-delta", "r-bravo", "r-alpha", "r-charlie"])
        h.publish(_CANON, _GROUP_LENS, admit_limit=None)
        head_inputs = h.run_pass_prologue()
        h._apply_uniform_head_order(head_inputs)
        self.assertEqual(h.rids(), _EXPECTED_GROUP_ORDER)

    def test_the_ballot_is_gone_and_the_decision_is_not(self):
        """The precise shape of the boot's defect, pinned.

        The pass really did destroy the ballot; what must not follow is that
        the group order goes with it.
        """
        h = _PassHarness(["r-delta", "r-bravo", "r-alpha", "r-charlie"])
        h.publish(_CANON, _GROUP_LENS, admit_limit=None)
        head_inputs = h.run_pass_prologue()
        self.assertNotIn("_uniform_prefetch_ballot", h.__dict__)
        self.assertIsNotNone(head_inputs)
        self.assertTrue(head_inputs.digest_agreed)

    def test_a_void_digest_still_yields_the_group_order(self):
        """Change 2 of this ticket, re-pinned at the consumption point.

        A digest mismatch is what the wedge case looks like, so it is the
        one case the enforcer must NOT hand back to the rank-local rule.
        """
        h = _PassHarness(["r-delta", "r-bravo", "r-alpha", "r-charlie"])
        h.publish(_CANON, _GROUP_LENS, admit_limit=None, digest_agreed=False)
        head_inputs = h.run_pass_prologue()
        h._apply_uniform_head_order(head_inputs)
        self.assertEqual(h.rids(), _EXPECTED_GROUP_ORDER)

    def test_reorders_and_never_drops(self):
        """A request the group did not name waits; it does not vanish."""
        h = _PassHarness(["r-delta", "r-echo", "r-alpha", "r-charlie"])
        h.publish(_CANON, _GROUP_LENS, admit_limit=None)
        h._apply_uniform_head_order(h.run_pass_prologue())
        self.assertIn("r-echo", h.rids())
        self.assertEqual(len(h.rids()), 4)


class TheCountArmStopsAtTheGroupNumber(unittest.TestCase):
    def test_the_group_limit_replaces_the_local_one(self):
        h = _PassHarness(_CANON, allocatable=9)
        h.publish(_CANON, _GROUP_LENS, admit_limit=2)
        self.assertEqual(h._uniform_allocatable_reqs(0, h.run_pass_prologue()), 2)

    def test_an_unpriced_group_leaves_the_local_number_alone(self):
        h = _PassHarness(_CANON, allocatable=9)
        h.publish(_CANON, _GROUP_LENS, admit_limit=None)
        self.assertEqual(h._uniform_allocatable_reqs(0, h.run_pass_prologue()), 9)


# ---------------------------------------------------------------------------
# Part 3 -- LOUDNESS. The count arm's inertness was invisible; it is the
# reason a boot with zero #823 lines proved nothing.
# ---------------------------------------------------------------------------


class ADegradationUnderAnArmedEnforcerIsLoud(unittest.TestCase):
    """The window's criterion needs a POSITIVE signal, not an absence.

    boot_window2_0823_1554 logged the ORDER arm's failure 105 times and the
    COUNT arm's failure zero times -- not because the count arm worked, but
    because `getattr(..., None)` gave it a plausible number to return. So
    "no #823 lines in the log" was compatible with total inertness. Both
    arms now count, and a counter is what a boot-log grep can assert on.
    """

    def test_the_order_arm_says_so_when_the_decision_is_missing(self):
        h = _PassHarness(_CANON)
        with self.assertLogs("sglang.srt.managers.scheduler", level="WARNING") as cm:
            h._apply_uniform_head_order(h.run_pass_prologue())
        self.assertTrue(any("DEGRADED" in line and "order" in line for line in cm.output))
        self.assertEqual(h._tp_head_degrade_total_order, 1)

    def test_the_count_arm_says_so_too_and_this_is_the_new_instrument(self):
        h = _PassHarness(_CANON, allocatable=7)
        with self.assertLogs("sglang.srt.managers.scheduler", level="WARNING") as cm:
            self.assertEqual(h._uniform_allocatable_reqs(0, h.run_pass_prologue()), 7)
        self.assertTrue(any("DEGRADED" in line and "count" in line for line in cm.output))
        self.assertEqual(h._tp_head_degrade_total_count, 1)

    def test_the_count_arm_reports_once_per_pass_not_once_per_candidate(self):
        """The admission loop calls it per request; the cadence is per pass.

        Unbounded per-candidate logging is how the same boot produced 7710
        void lines in seven seconds.
        """
        h = _PassHarness(_CANON, allocatable=7)
        head_inputs = h.run_pass_prologue()
        for _ in range(20):
            h._uniform_allocatable_reqs(0, head_inputs)
        self.assertEqual(h._tp_head_degrade_total_count, 1)

    def test_can_fail_with_the_gate_off_a_rank_local_pass_is_not_a_defect(self):
        """The can-fail arm, and it pins the OTHER error.

        With the enforcer off, rank-local IS the contract. Counting it would
        make every single-rank boot report a defect on every pass, and an
        instrument that fires on the healthy state is not an instrument.
        """
        h = _PassHarness(_CANON, tp_size=1, pp_size=3, allocatable=7)
        h._apply_uniform_head_order(h.run_pass_prologue())
        h._uniform_allocatable_reqs(0, None)
        self.assertEqual(getattr(h, "_tp_head_degrade_total_order", 0), 0)
        self.assertEqual(getattr(h, "_tp_head_degrade_total_count", 0), 0)

    def test_the_recovery_edge_is_reported(self):
        """"It healed after N passes" and "it never healed" are the same
        silence to a latched logger. The window criterion needs the edge."""
        h = _PassHarness(_CANON)
        h._apply_uniform_head_order(h.run_pass_prologue())
        self.assertEqual(h._tp_head_degrade_streak_order, 1)
        h.publish(_CANON, _GROUP_LENS, admit_limit=None)
        with self.assertLogs("sglang.srt.managers.scheduler", level="INFO") as cm:
            h._apply_uniform_head_order(h.run_pass_prologue())
        self.assertTrue(any("RESTORED" in line for line in cm.output))
        self.assertEqual(h._tp_head_degrade_streak_order, 0)


# ---------------------------------------------------------------------------
# Part 4 -- ROOT 2. The gate, and the PP phase it is accused of missing.
# ---------------------------------------------------------------------------


class TheGateNamesItsReason(unittest.TestCase):
    """"Inert" and "correctly gated off" must stop reading identically.

    The window could not tell them apart and had to spend its whole budget
    finding out. A reason string is the entire remedy.
    """

    def test_a_real_tp_group_arms_the_enforcer(self):
        self.assertTrue(thc.enforcer_gate(True, 3, 1).enabled)
        self.assertEqual(thc.enforcer_gate(True, 3, 1).reason, thc.GATE_ON)

    def test_the_kill_switch_is_distinguishable_from_the_topology(self):
        self.assertEqual(
            thc.enforcer_gate(False, 3, 1).reason, thc.GATE_OFF_KILL_SWITCH
        )
        self.assertEqual(
            thc.enforcer_gate(True, 1, 3).reason, thc.GATE_OFF_TP_WORLD_OF_ONE
        )
        self.assertEqual(
            thc.enforcer_gate(True, None, None).reason,
            thc.GATE_OFF_NO_PARALLEL_STATE,
        )

    def test_the_pp_phase_off_reason_names_the_mechanism_that_covers_it(self):
        """The line the boot needed and did not have."""
        detail = thc.enforcer_gate(True, 1, 3).detail
        self.assertIn("#791", detail)
        self.assertIn("one member", detail)

    def test_the_report_is_a_transition_not_a_latch(self):
        """Under --enable-phase-flip this gate switches at every cutover."""
        reason, first = thc.advance_gate_report(None, thc.enforcer_gate(True, 1, 3))
        self.assertIsNotNone(first)
        reason, again = thc.advance_gate_report(reason, thc.enforcer_gate(True, 1, 3))
        self.assertIsNone(again, "a repeated state must not re-log")
        reason, back = thc.advance_gate_report(reason, thc.enforcer_gate(True, 3, 1))
        self.assertIsNotNone(back, "coverage COMING BACK must be reported too")


class WideningTheGateToTheTpWorldOfOneWouldNotProduceCongruence(unittest.TestCase):
    """THE DANGEROUS DIRECTION, and the one this ticket was pushed toward.

    The window's reading was "45 of 51 prefill batches ran where the enforcer
    is gated off, so widen the gate". This case shows what that would buy: on
    `--tp-size 1 --pp-size 3` the reduce group has ONE member per rank
    (parallel_state.py:3166-3188), so the "group" match lengths each rank
    reads back are its OWN. Enforcing them produces three different orders
    with every rank logging `source=group`.

    That is worse than the status quo, not better: today the ranks form
    rank-locally and say so; with the gate widened they would form
    rank-locally and claim agreement.
    """

    def test_the_gate_refuses_the_world_of_one(self):
        self.assertFalse(thc.enforcer_gate(True, 1, 3).enabled)

    def test_and_here_is_why_a_singleton_reduce_still_diverges(self):
        # A MIN-reduce over one member returns that member's own vote, so
        # each rank's "group" payload is its own local match lengths.
        local = {
            0: {"r-alpha": 16, "r-bravo": 4, "r-charlie": 4096},
            1: {"r-alpha": 8192, "r-bravo": 4, "r-charlie": 16},
            2: {"r-alpha": 16, "r-bravo": 4096, "r-charlie": 16},
        }
        canonical = thc.canonical_head_rids(["r-alpha", "r-bravo", "r-charlie"])
        orders = []
        for rank in (0, 1, 2):
            singleton_reduce = thc.build_head_order_payload(canonical, local[rank])
            order, source = thc.head_decision(
                canonical,
                singleton_reduce,
                list(local[rank]),
                local[rank],
                digest_agreed=True,
                enforcer_enabled=True,
            )
            self.assertEqual(source, thc.SOURCE_GROUP, "and it would say 'group'")
            orders.append(order)
        self.assertFalse(
            thc.head_order_is_uniform(orders),
            "if a singleton reduce produced congruence, widening the gate "
            "would be harmless and this whole argument would be wrong",
        )


class ThePpPhaseFormsCongruentlyThroughItsOwnActuator(unittest.TestCase):
    """Three PP schedulers, one formation. The gate's third reason, executed.

    This is the claim that makes leaving the gate alone defensible, so it is
    driven rather than asserted: PP0 commits a decision, PP1 and PP2 reconcile
    it against their OWN diverged prefix caches, and all three end with the
    same membership in the same order.

    The can-fail arm is the pre-#791 rule on the same inputs -- each rank
    ordering by its own match lengths -- which diverges. Without it this case
    would pass on inputs that never disagreed in the first place.
    """

    #: Three independently evolved radix caches over the same three requests
    #: (#616B family). Rank 1 has a hot prefix for bravo, rank 2 for charlie.
    LOCAL = {
        0: {"r-alpha": 128, "r-bravo": 64, "r-charlie": 32},
        1: {"r-alpha": 128, "r-bravo": 2048, "r-charlie": 32},
        2: {"r-alpha": 128, "r-bravo": 64, "r-charlie": 4096},
    }
    TOTAL = {"r-alpha": 512, "r-bravo": 512, "r-charlie": 512}

    def _reqs(self, rank, order):
        return [
            SimpleNamespace(
                rid=rid,
                prefix_indices=[0] * self.LOCAL[rank][rid],
                extend_input_len=self.TOTAL[rid] - self.LOCAL[rank][rid],
            )
            for rid in order
        ]

    def test_all_three_stages_form_the_same_batch(self):
        from sglang.srt.managers import pp_admission_congruence as ppc

        # PP0 owns admission truth and commits its own batch order.
        pp0_order = ["r-charlie", "r-alpha", "r-bravo"]
        decision = ppc.build_pp_admission_decision(
            0, self._reqs(0, pp0_order), pp_size=3
        )

        formed = {0: [req.rid for req in self._reqs(0, pp0_order)]}
        for rank in (1, 2):
            effective, decision = ppc.reconcile_pp_admission_decision(
                decision, self.LOCAL[rank], rank=rank, pp_size=3
            )
            self.assertEqual(
                sorted(effective), sorted(pp0_order), "membership must survive"
            )
            schedule = ppc.forwarded_schedule(decision)
            # This rank's own queue arrived in a DIFFERENT order -- that is
            # the whole point; #791 re-imposes the forwarded one.
            local_order = sorted(pp0_order, key=lambda r: -self.LOCAL[rank][r])
            reordered = ppc.order_batch_by_schedule(
                self._reqs(rank, local_order), schedule
            )
            formed[rank] = [req.rid for req in reordered]

        self.assertEqual(formed[0], formed[1])
        self.assertEqual(formed[0], formed[2])

    def test_can_fail_the_pre_791_rank_local_rule_diverges_on_these_inputs(self):
        orders = [
            thc.local_head_order(list(self.LOCAL[rank]), self.LOCAL[rank])
            for rank in (0, 1, 2)
        ]
        self.assertFalse(
            thc.head_order_is_uniform(orders),
            "the premise is asserted, not assumed: if these three caches "
            "produced the same order anyway, the congruence case above "
            "would prove nothing",
        )
