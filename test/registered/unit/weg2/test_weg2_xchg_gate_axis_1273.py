# SPDX-License-Identifier: Apache-2.0
"""#1273 B4q: THE GATE AXIS -- two leg gates read one arm while the order arms
the other, so the whole bounce lane was dead by construction on the arm the
S6I order actually launches.

WHAT BOOT weg2xsn16 MEASURED (record BOOT_weg2xsn16_0911.md, order DONE FAIL).
ARGV ``--weg2-weight-source exchange`` with ``SGLANG_WEG2_XCHG_INJECT=shadow``
(read live from ``/proc/<pid>/environ`` on both leaders and all five workers).
B4i's coverage half and B4k both CONFIRMED on metal -- ``uncovered=0`` on 44/44
``WEG2-XCHG-COVER`` lines, ``exempt=0`` on all 44, W84 bare 0 / genuine 0 on
both groups, ``WEG2-DC group=D measured=1646/1310/1310`` against a predicted
1644/1308/1308, ``RESIDENT tag=weights_draft ... in_family=yes`` on all three D
ranks. And ZERO ``WEG2-XCHG-INJECT`` / ``-SHADOW`` / ``-PLAN`` / ``-AGREED`` /
``-BOUNCE-LEG`` lines on either group, while ``WEG2-GROUP-FENCE`` ran 48x on D.
The flip path executed. Only the leg hook never fired.

THE ROOT, and it is one predicate read in two places:

* ``weight_updater.py`` ``_weg2_shadow_hook`` -- the ONLY production caller of
  ``run_leg_hook`` -- opened with ``if not sh.shadow_armed(): return``.
* ``weight_exchange_shadow.run_leg_hook`` itself -- ``if armed is None: armed =
  shadow_armed()`` then ``if not armed: return None``.

``shadow_armed()`` is ``weight_source() == "shadow"`` and ``exchange_armed()``
is ``weight_source() == "exchange"``; the flag is THREE-VALUED
(``{ring|exchange|shadow}``), so the two are MUTUALLY EXCLUSIVE. On the
``exchange`` arm both gates return before touching anything. ``inject_mode()``
carries the S6 answer and was never consulted on the hook path: the fourth
instance in this lane of "published, read, never acted upon".

WHY THE DESK SUITE WAS GREEN THROUGH IT, and why this file is shaped the way it
is: every existing test drives ``run_leg_hook`` with ``armed=True`` and walks
straight past the gate. So the red-first test here goes through the PRODUCTION
path -- the real code object of ``_weg2_shadow_hook`` -- and passes no ``armed``
at all.

THE FIX (B4q, operator ruling 2): ONE new predicate meaning "the bounce lane is
armed at all", ``weight_exchange.bounce_lane_armed()`` = ``weight_source() !=
WEIGHT_SOURCE_RING``, owned beside the other two and delegated by
``weight_exchange_shadow`` in exactly the delegation shape ``shadow_armed``
already uses. BOTH gate sites read it. The candidate
``exchange_armed() and inject_mode() in (shadow, authoritative)`` was REJECTED
before it was built: its second conjunct is a TAUTOLOGY (``inject_mode()``
always returns one of those two, default ``shadow``), so it reduces to
``exchange_armed()`` and would disarm the ``shadow`` arm XSN12 proved runs --
trading one silent no-op for another.

WHAT DOES NOT MOVE, each with its own test below, because a widening is exactly
where semantics leak:

* ``exchange_armed()`` keeps deciding the weights-region TAG and the draft
  tag's family membership -- i.e. whether the shadow still HAS a ground truth.
* ``inject_authoritative()`` keeps deciding AUTHORITY, read at
  ``weight_updater.py:852`` as ``exchange_armed() and inject_authoritative()``.
  ``host weights`` 42.96 -> 0.00 GiB is B7's acceptance, not this slice's.
* ``ring`` stays byte for byte today's boot.

THE SPEC HALF, closed by measurement and then pinned here rather than asserted
(see :class:`TestTheCompareGroundExistsOnThisArm`): ``exchange_armed()``'s
docstring claimed it was "the one predicate that decides
``enable_cpu_backup``", which would have meant that arming the exchange DELETES
the refilled copy the shadow compares against -- the reason to fear this
widening. It is STALE. ``enable_cpu_backup`` is decided at
``model_runner.py:2440`` from ``server_args.enable_weights_cpu_backup`` (or the
draft-worker variant), and the weg2 launcher passes
``--enable-weights-cpu-backup`` UNCONDITIONALLY in ``common_flags``
(``launcher.py:2605``) -- both groups, every arm. So the ground truth exists on
this arm, no arm predicate gates it, and XSN16's measured ``host_weights=42.96
GiB`` is the corroborating reading (``host_ledger.py:86``: everything with
``enable_cpu_backup`` is in that image). The docstring is corrected in the same
commit.

RED-FIRST: ``TestTheProductionHookReachesTheLaneOnTheExchangeArm`` is RED at
``ec753f00d9`` (the production gate returns before ``_weg2_group_name`` is ever
called) and GREEN after this slice. Hermetic: ``CUDA_VISIBLE_DEVICES=""``, no
CUDA, no device, no server -- the hook's identity accessors are stubbed on the
instance, which is what lets the REAL method body run on a box with no GPU.
"""

import ast
import inspect
import os
import unittest
from typing import ClassVar

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

from sglang.srt.managers.scheduler_components import weight_updater as wu
from sglang.srt.weg2 import weight_exchange as wx
from sglang.srt.weg2 import weight_exchange_shadow as sh
from sglang.test.test_utils import CustomTestCase

WU_PATH = os.path.abspath(inspect.getsourcefile(wu))


def _wu_ast(name: str) -> ast.FunctionDef:
    with open(WU_PATH, encoding="utf-8") as fh:
        tree = ast.parse(fh.read())
    return next(n for n in ast.walk(tree)
                if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))
                and n.name == name)


class _Req:
    """The minimum of a flip request: the front publishes the epoch and the
    hook reads it off the request (never a rank-local counter)."""

    def __init__(self, epoch="1789200000.0") -> None:
        self.epoch = epoch


class _HookProbe:
    """A stand-in the REAL ``_weg2_shadow_hook`` body can run on.

    Not a mock of the method -- the method under test is the production
    function object, bound to this object. Only the identity accessors it
    calls are provided, which is what makes the body runnable with no CUDA
    device and no scheduler.

    ``_weg2_group_name`` is the PROBE: it is the first thing the body does
    AFTER the arm gate, so "was it called" is exactly "did the arm gate let
    execution through". It records and then returns a group the identity
    sweep will reject, so the body takes its own named W79 exit instead of
    walking into the region machinery -- the gate is what this file grades,
    and nothing past it is touched.
    """

    def __init__(self, group: str = "ZZ") -> None:
        self.calls: list = []
        self._group = group

    def _weg2_group_name(self) -> str:
        self.calls.append("group")
        return self._group

    def _weg2_rank(self) -> int:
        self.calls.append("rank")
        return 0

    def _weg2_device_index(self) -> int:
        self.calls.append("device")
        return 0

    def run(self, hook: str = "destination") -> None:
        wu.SchedulerWeightUpdaterManager._weg2_shadow_hook(
            self, hook, recv_req=_Req())


class _Arm:
    """``--weg2-weight-source`` for the duration of a block, restored after.

    Uses the module's own ``weight_source_for_test`` contract by hand rather
    than importing a pytest fixture: this file is a unittest one (the gate
    decides with ``os.environ``, so the env is the only thing to set).
    """

    def __init__(self, value) -> None:
        self.value = value

    def __enter__(self):
        self.previous = os.environ.get(wx.WEIGHT_SOURCE_ENV)
        if self.value is None:
            os.environ.pop(wx.WEIGHT_SOURCE_ENV, None)
        else:
            os.environ[wx.WEIGHT_SOURCE_ENV] = self.value
        return self

    def __exit__(self, *exc):
        if self.previous is None:
            os.environ.pop(wx.WEIGHT_SOURCE_ENV, None)
        else:
            os.environ[wx.WEIGHT_SOURCE_ENV] = self.previous
        return False


# --------------------------------------------------------------------------
# The predicate itself.
# --------------------------------------------------------------------------

class TestTheAxisPredicate(CustomTestCase):

    def test_it_answers_the_axis_question_on_all_three_arms(self):
        for arm, expected in ((None, False),
                              (wx.WEIGHT_SOURCE_RING, False),
                              (wx.WEIGHT_SOURCE_EXCHANGE, True),
                              (wx.WEIGHT_SOURCE_SHADOW, True)):
            with _Arm(arm):
                self.assertIs(wx.bounce_lane_armed(), expected,
                              f"arm {arm!r} answered the wrong axis")

    def test_an_unrecognised_arm_reads_as_ring_and_therefore_disarmed(self):
        """``weight_source()`` maps anything unknown to ``ring``, and the axis
        must inherit that: a typo in an env var may not arm a lane."""
        with _Arm("EXHCANGE"):
            self.assertEqual(wx.weight_source(), wx.WEIGHT_SOURCE_RING)
            self.assertIs(wx.bounce_lane_armed(), False)

    def test_the_two_arms_it_covers_are_mutually_exclusive(self):
        """The premise of the whole defect, asserted so it cannot be forgotten:
        no arm makes both old predicates true, which is why gating on one of
        them is gating on half the axis."""
        for arm in (wx.WEIGHT_SOURCE_EXCHANGE, wx.WEIGHT_SOURCE_SHADOW):
            with _Arm(arm):
                self.assertFalse(wx.exchange_armed() and wx.shadow_armed())
                self.assertTrue(wx.exchange_armed() or wx.shadow_armed())
                self.assertIs(wx.bounce_lane_armed(), True)

    def test_it_is_spelled_as_not_ring_not_as_a_disjunction(self):
        """Operator ruling 2, and it is a FUTURE-PROOFING choice with a reason:
        ``exchange_armed() or shadow_armed()`` is equivalent only while the
        flag has exactly three values, and would silently stop covering a
        fourth arm the day one is added. The negation keeps meaning "not the
        untouched default"."""
        src = inspect.getsource(wx.bounce_lane_armed)
        body = src.split('"""')[-1]
        self.assertIn("WEIGHT_SOURCE_RING", body)
        self.assertIn("!=", body)
        self.assertNotIn("shadow_armed()", body)
        self.assertNotIn("exchange_armed()", body)

    def test_the_shadow_module_delegates_and_does_not_respell_it(self):
        """The Zweitbuchhaltung rule this module already paid for once: the
        delegation forwards to the owner and never re-spells the comparison
        (``weight_exchange_shadow.py`` says so about ``shadow_armed`` in its
        own comment)."""
        src = inspect.getsource(sh.bounce_lane_armed)
        body = src.split('"""')[-1]
        self.assertIn("wxm.bounce_lane_armed()", body)
        self.assertNotIn("weight_source()", body)
        self.assertNotIn("WEIGHT_SOURCE_RING", body)
        for arm, expected in ((wx.WEIGHT_SOURCE_RING, False),
                              (wx.WEIGHT_SOURCE_EXCHANGE, True),
                              (wx.WEIGHT_SOURCE_SHADOW, True)):
            with _Arm(arm):
                self.assertIs(sh.bounce_lane_armed(), expected)


# --------------------------------------------------------------------------
# RED-FIRST: the production path, on the arm the order actually launches.
# --------------------------------------------------------------------------

class TestTheProductionHookReachesTheLaneOnTheExchangeArm(CustomTestCase):
    """RED at ``ec753f00d9``, GREEN after B4q.

    Drives the REAL ``_weg2_shadow_hook`` code object. No ``armed=`` override
    anywhere -- that override is precisely why the suite was green while the
    metal emitted nothing.
    """

    def test_the_exchange_arm_gets_past_the_arm_gate(self):
        probe = _HookProbe()
        with _Arm(wx.WEIGHT_SOURCE_EXCHANGE):
            self.assertIs(wx.shadow_armed(), False,
                          "premise: the OLD gate is False on this arm")
            probe.run()
        self.assertIn(
            "group", probe.calls,
            "the production hook returned at the arm gate on the `exchange` "
            "arm -- this is boot weg2xsn16's defect: 48 group fences, 0 leg "
            "lines",
        )

    def test_the_shadow_arm_still_gets_past_it(self):
        """XSN12's form stays armed (18 stripes, 18 MATCH, oncard-*.bin
        3 x 33,554,432 B). The rejected candidate A would have broken exactly
        this."""
        probe = _HookProbe()
        with _Arm(wx.WEIGHT_SOURCE_SHADOW):
            probe.run()
        self.assertIn("group", probe.calls)

    def test_the_ring_arm_still_returns_at_the_gate(self):
        """The default boot, byte for byte. The gate must fire BEFORE any
        identity accessor is touched -- not merely produce no output."""
        for arm in (None, wx.WEIGHT_SOURCE_RING):
            probe = _HookProbe()
            with _Arm(arm):
                probe.run()
            self.assertEqual(
                probe.calls, [],
                f"the ring arm ({arm!r}) walked past the arm gate",
            )

    def test_the_ring_arm_is_mined_at_the_PRODUCTION_level_too(self):
        """Danger direction (i), and the NEW door is what it mines.

        ``test_the_hooks_are_never_reached_on_the_ring_arm``
        (``test_weg2_xchg_shadow_1273.py``) mines every door out of the arm
        check -- but it enters at ``run_leg_hook``, so the door B4q moved is
        one level ABOVE its entry point. This is the same tripwire idea
        applied at the production caller: on ``ring``, ``_weg2_shadow_hook``
        must not reach the shadow module's machinery at all, and every piece
        of it raises if touched.

        "It returned None" would pass for a hook that did all its work and
        threw it away, which is why the doors are mined rather than the
        result inspected.
        """
        mined = []

        def tripwire(*_a, **_k):
            mined.append("reached")
            raise AssertionError("the ring arm reached the shadow machinery")

        saved = {name: getattr(sh, name) for name in
                 ("plan_for_leg", "run_leg_hook", "rank_local_skip_message")}
        try:
            for name in saved:
                setattr(sh, name, tripwire)
            for arm in (None, wx.WEIGHT_SOURCE_RING):
                probe = _HookProbe()
                with _Arm(arm):
                    probe.run()
                self.assertEqual(probe.calls, [], f"arm {arm!r} walked past the gate")
            self.assertEqual(mined, [], "a mined door was reached on the ring arm")
        finally:
            for name, value in saved.items():
                setattr(sh, name, value)

    def test_both_hooks_go_through_the_same_gate(self):
        """``source`` and ``destination`` are one gate, not two -- the
        delegation calls itself THE ONE GATE ON BOTH HOOKS."""
        for hook in ("source", "destination"):
            probe = _HookProbe()
            with _Arm(wx.WEIGHT_SOURCE_EXCHANGE):
                probe.run(hook)
            self.assertIn("group", probe.calls, f"hook={hook} did not pass")


class TestTheGateSitesReadTheAxisAndNothingElse(CustomTestCase):
    """Structural pins on BOTH sites, so a future edit cannot quietly put one
    of them back on a single arm -- which is the defect, and it would be
    invisible in a suite that drives the other site directly."""

    def test_the_updater_gate_reads_the_axis(self):
        src = inspect.getsource(
            wu.SchedulerWeightUpdaterManager._weg2_shadow_hook)
        self.assertIn("sh.bounce_lane_armed()", src)
        self.assertNotIn("sh.shadow_armed()", src)
        self.assertNotIn("sh.exchange_armed()", src)

    def test_the_run_leg_hook_gate_reads_the_axis(self):
        src = inspect.getsource(sh.run_leg_hook)
        self.assertIn("armed = bounce_lane_armed()", src)
        self.assertNotIn("armed = shadow_armed()", src)

    def test_the_updater_hook_is_still_the_only_production_caller(self):
        """If a second caller appears it needs the same gate, and this is what
        says so before the next boot does."""
        import subprocess
        # THE REPO ROOT FROM THIS FILE, not from the module under test: this
        # file sits at test/registered/unit/weg2, so four levels up is the
        # root -- the same walk the sibling weg2 tests use. Deriving it from
        # weight_updater.py instead needs FIVE levels
        # (python/sglang/srt/managers/scheduler_components), and the off-by-one
        # made `git grep` search python/ for a python/-prefixed pathspec and
        # match nothing, which this assertion reported as "the caller set
        # changed" -- a search that finds nothing is not a finding about the
        # tree, and the message has to be able to tell those apart.
        root = os.path.abspath(
            os.path.join(os.path.dirname(os.path.abspath(__file__)),
                         "..", "..", "..", ".."))
        out = subprocess.run(
            ["git", "-C", root, "grep", "-n", "run_leg_hook(", "--",
             "python/sglang/srt"],
            capture_output=True, text=True, check=False)
        self.assertEqual(out.returncode, 0,
                         f"git grep failed in {root!r}: {out.stderr[:200]}")
        sites = [ln for ln in out.stdout.splitlines()
                 if "def run_leg_hook" not in ln]
        self.assertTrue(sites, "git grep found run_leg_hook NOWHERE -- the "
                               "search is broken, not the tree")
        callers = [ln for ln in sites if "weight_updater.py" in ln]
        self.assertEqual(
            len(callers), 1,
            f"run_leg_hook's production caller set changed: {callers} "
            f"(all sites: {sites})",
        )


class TestAuthorityDoesNotMoveOntoTheAxis(CustomTestCase):
    """Danger direction (ii): the widening must not make ``authoritative``
    reachable. ``host weights`` 42.96 -> 0.00 GiB is B7's acceptance."""

    def test_the_authority_site_still_reads_inject_authoritative(self):
        src = inspect.getsource(wu.SchedulerWeightUpdaterManager)
        self.assertIn("wx.exchange_armed() and wx.inject_authoritative()", src)

    def test_the_axis_predicate_is_not_read_on_the_authority_path(self):
        """The axis says the legs RUN; it must never be the thing that says the
        injection OWNS the bytes."""
        src = inspect.getsource(wu.SchedulerWeightUpdaterManager)
        for line in src.splitlines():
            if "bounce_lane_armed" in line:
                self.assertNotIn("inject_authoritative", line)
                self.assertNotIn("_weg2_wake_weight_carrier", line)

    def test_the_axis_is_true_under_shadow_inject_where_authority_is_false(self):
        """The two questions answered oppositely at the same time, which is
        exactly the S6I shadow boot: the legs run AND the refill stays the
        authority."""
        previous = os.environ.get(wx.INJECT_ENV)
        os.environ[wx.INJECT_ENV] = wx.INJECT_SHADOW
        try:
            with _Arm(wx.WEIGHT_SOURCE_EXCHANGE):
                self.assertIs(wx.bounce_lane_armed(), True)
                self.assertIs(wx.inject_authoritative(), False)
                self.assertIs(wx.exchange_armed(), True)
        finally:
            if previous is None:
                os.environ.pop(wx.INJECT_ENV, None)
            else:
                os.environ[wx.INJECT_ENV] = previous

    def test_the_rejected_candidate_A_is_a_tautology(self):
        """Operator ruling 1, kept as a test so nobody re-derives it: the
        second conjunct of ``exchange_armed() and inject_mode() in (shadow,
        authoritative)`` is always true, so the expression reduces to
        ``exchange_armed()`` and would disarm the shadow arm."""
        for value in ("", "shadow", "authoritative", "nonsense", "SHADOW"):
            previous = os.environ.get(wx.INJECT_ENV)
            os.environ[wx.INJECT_ENV] = value
            try:
                self.assertIn(wx.inject_mode(), wx.INJECT_CHOICES,
                              f"inject_mode({value!r}) left the choice set")
            finally:
                if previous is None:
                    os.environ.pop(wx.INJECT_ENV, None)
                else:
                    os.environ[wx.INJECT_ENV] = previous
        with _Arm(wx.WEIGHT_SOURCE_SHADOW):
            candidate_a = wx.exchange_armed() and wx.inject_mode() in wx.INJECT_CHOICES
            self.assertFalse(candidate_a,
                             "candidate A disarms the shadow arm, as ruled")
            self.assertIs(wx.bounce_lane_armed(), True,
                          "B4q's predicate keeps it armed, which is the point")


class TestTheCompareGroundExistsOnThisArm(CustomTestCase):
    """THE SPEC HALF, pinned rather than asserted.

    The fear this widening has to answer: if arming ``exchange`` removed the
    refilled host copy, the shadow legs would compare against nothing, and a
    silent compare with no ground is worse than a refusal. The answer is that
    no arm predicate gates that copy at all.
    """

    def test_enable_cpu_backup_is_decided_by_server_args_not_by_an_arm(self):
        import sglang.srt.model_executor.model_runner as mr
        src = inspect.getsource(mr.ModelRunner)
        self.assertIn("enable_cpu_backup = self.server_args.enable_weights_cpu_backup", src)
        decider = next(ln for ln in src.splitlines()
                       if "enable_cpu_backup = self.server_args" in ln)
        for predicate in ("exchange_armed", "shadow_armed", "bounce_lane_armed",
                          "inject_mode", "weight_source"):
            self.assertNotIn(predicate, decider)

    def test_the_launcher_passes_the_flag_unconditionally(self):
        """``common_flags`` -- both groups, every arm. If this ever becomes
        conditional on the weight source, THIS test is what says the shadow's
        ground truth just became arm-dependent."""
        from sglang.srt.weg2 import launcher as L
        src = inspect.getsource(L.common_flags)
        self.assertIn('"--enable-weights-cpu-backup"', src)
        tree = ast.parse(inspect.cleandoc(src) if src.startswith("def") else src)
        fn = next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef))
        guarded = [
            n for n in ast.walk(fn)
            if isinstance(n, (ast.If, ast.IfExp))
            and "--enable-weights-cpu-backup" in ast.unparse(n)
        ]
        self.assertEqual(
            guarded, [],
            "--enable-weights-cpu-backup is now inside a conditional: the "
            "shadow's compare ground may have become arm-dependent, which is "
            "the one thing the B4q widening assumed it was not",
        )

    def test_exchange_armed_no_longer_claims_to_decide_it(self):
        """The stale claim that made this look dangerous. Corrected in the same
        commit as the widening, and pinned here so it cannot rot back."""
        doc = inspect.getdoc(wx.exchange_armed) or ""
        self.assertIn("DOES NOT DECIDE ``enable_cpu_backup``", doc)
        self.assertIn("model_runner.py:2440", doc)
        self.assertIn("common_flags", doc)

    def test_what_exchange_armed_does_decide_is_still_true(self):
        """The correction must not overshoot: the predicate still owns the tag
        and the draft family, and those two consumers are what keep the ground
        truth honest under ``shadow``."""
        from sglang.srt.managers import weg2_memory_saver as ms
        src = inspect.getsource(ms.draft_tag_in_family)
        self.assertIn("exchange_armed", src)
        with _Arm(wx.WEIGHT_SOURCE_SHADOW):
            self.assertIs(wx.exchange_armed(), False)
            self.assertIs(ms.draft_tag_in_family(), False)
        with _Arm(wx.WEIGHT_SOURCE_EXCHANGE):
            self.assertIs(ms.draft_tag_in_family(), True)


class TestTheInjectInstrumentNeverLiesTowardsAuthority(CustomTestCase):
    """#1336 -- danger direction (ii) in the INSTRUMENT rather than in the code
    path, and it rides in B4q because it is the same direction: the
    ``WEG2-XCHG-BOUNCE-LEG`` line is what RE-STAMP 6's grading plan reads as
    the evidence that ``authoritative`` did not become reachable.

    WHAT IT USED TO SAY:
    ``mode={self.inject.mode if self.inject else wx.INJECT_AUTHORITATIVE}``.

    MY READING DIFFERS FROM THE ONE I WAS HANDED, and the difference changed
    the fix, so it is written here. The report was that the line "may mean
    'no inject object' rather than 'ran authoritative'". At the ONE
    construction site that is not so: ``run_bounce_leg`` validates ``mode``
    (an unknown one RAISES -- "a default here would decide whether this leg
    owns 27 GiB of weights"), sets ``comparing = mode == INJECT_SHADOW``, and
    passes ``inject=verdict if comparing else None`` -- so there
    ``inject is None`` really did mean authoritative, and :936's comment says
    exactly that.

    THE HOLE IS THE DATACLASS DEFAULT, which is worse because it is silent:
    ``inject: Optional[InjectVerdict] = None`` is a FIELD DEFAULT, so any
    other construction -- an error path, a future early return, a test
    double -- printed ``mode=authoritative`` for a leg that never ran
    authoritative. The invariant was enforced at one call site, not by the
    type.

    THE FIX therefore removes the inference instead of flipping its default:
    the validated ``mode`` TRAVELS on the result (``BounceResult.mode``), so
    the line and the ``comparing`` decision read ONE source, and an absent
    mode prints as the named state ``unset`` -- which is not a member of
    ``INJECT_CHOICES`` and so can never be mistaken for a mode a leg ran.
    Flipping the default to ``shadow`` would have been a second decision with
    a safer direction; carrying the value is no decision at all.
    """

    #: Every required field of ``BounceResult``, at values that say "nothing
    #: moved": the class is a measurement record, so its required fields have
    #: no defaults and a test that wants to grade ONE token still has to state
    #: all of them. Kept in one place so the six tests below differ only in
    #: the token under test.
    EMPTY: ClassVar[dict] = {
        "units": 0, "bands": 0, "deposited_bytes": 0, "collected_bytes": 0,
        "planned_bytes": 0, "host_bytes_peak": 0, "slot_bytes": 0, "depth": 0,
        "widest_unit_key": ("", ""), "widest_unit_bytes": 0,
        "widest_run_bytes": 0, "overlap": "n/a",
    }

    def _line(self, **kw) -> str:
        from sglang.srt.weg2 import weight_exchange_bounce as bx

        return bx.BounceResult(**{**self.EMPTY, **kw}).line()

    def test_a_result_with_no_mode_prints_the_named_state(self):
        line = self._line()
        self.assertIn(f"mode={wx.INJECT_MODE_UNSET}", line)
        self.assertNotIn(f"mode={wx.INJECT_AUTHORITATIVE}", line)
        self.assertNotIn(f"mode={wx.INJECT_SHADOW}", line)

    def test_the_named_state_is_not_a_mode_a_leg_can_run(self):
        self.assertNotIn(wx.INJECT_MODE_UNSET, wx.INJECT_CHOICES)
        self.assertEqual(wx.inject_mode.__module__, wx.__name__)
        for value in ("", wx.INJECT_MODE_UNSET, "nonsense"):
            previous = os.environ.get(wx.INJECT_ENV)
            os.environ[wx.INJECT_ENV] = value
            try:
                self.assertNotEqual(wx.inject_mode(), wx.INJECT_MODE_UNSET)
                self.assertIn(wx.inject_mode(), wx.INJECT_CHOICES)
            finally:
                if previous is None:
                    os.environ.pop(wx.INJECT_ENV, None)
                else:
                    os.environ[wx.INJECT_ENV] = previous

    def test_each_real_mode_prints_itself(self):
        for mode in wx.INJECT_CHOICES:
            self.assertIn(f"mode={mode}", self._line(mode=mode))

    def test_the_emitter_does_not_infer_the_mode_from_the_verdict(self):
        """THE MUTANT'S TARGET, pinned structurally as well as behaviourally:
        the line may not read ``self.inject`` to decide the MODE. A behavioural
        test alone would pass for an emitter that inferred correctly by
        accident at the one site that sets both."""
        from sglang.srt.weg2 import weight_exchange_bounce as bx

        src = inspect.getsource(bx.BounceResult.line)
        mode_lines = [ln for ln in src.splitlines() if "mode=" in ln]
        self.assertTrue(mode_lines, "the mode token left the line")
        for line in mode_lines:
            self.assertNotIn("INJECT_AUTHORITATIVE", line)
            self.assertNotIn("if self.inject else", line)
        self.assertIn("self.mode", " ".join(mode_lines))

    def test_the_mode_travels_from_the_one_validated_source(self):
        """``run_bounce_leg`` validates and refuses; the RESULT carries THAT
        value. One authority, which is what makes the instrument readable as
        evidence at all.

        PINNED BY AST, and the first cut of this test is why. It read
        ``assertIn("mode=mode,", src)`` -- and ``run_bounce_leg`` ALSO
        contains ``InjectVerdict(mode=mode, ...)``, so deleting the
        ``BounceResult`` keyword left the substring in place and the mutant
        that removes it SURVIVED (harness M10, measured). That is seat 5's M5
        precedent exactly: a text-scan pin on wiring breaks on the second
        occurrence. So the assertion now walks to the ``BounceResult(...)``
        call itself and checks ITS keywords.
        """
        from sglang.srt.weg2 import weight_exchange_bounce as bx

        src = inspect.getsource(bx.run_bounce_leg)
        self.assertIn("comparing = mode == wx.INJECT_SHADOW", src)
        self.assertIn("Refused rather than defaulted", src)

        tree = ast.parse(inspect.cleandoc(src))
        fn = next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef))
        ctors = [n for n in ast.walk(fn) if isinstance(n, ast.Call)
                 and getattr(n.func, "id", "") == "BounceResult"]
        self.assertEqual(len(ctors), 1,
                         "expected exactly one BounceResult construction")
        kwargs = {k.arg: ast.unparse(k.value) for k in ctors[0].keywords}
        self.assertIn("mode", kwargs,
                      "BounceResult is built without a mode: the line would "
                      "fall back to the named unset state for a leg that DID "
                      "run one of the two real modes")
        self.assertEqual(kwargs["mode"], "mode",
                         "the mode must be the validated parameter itself, "
                         f"not {kwargs['mode']!r}")

    def test_an_unset_mode_cannot_be_read_as_authority_by_the_grader(self):
        """The grading plan's own question: does this line prove authority did
        not run?  With ``unset`` distinct from both real modes, a grader can
        answer it from the line alone -- which is the whole point of the
        ticket."""
        for kw, expect_authority in (({}, False),
                                     ({"mode": wx.INJECT_SHADOW}, False),
                                     ({"mode": wx.INJECT_AUTHORITATIVE}, True)):
            line = self._line(**kw)
            claims = f"mode={wx.INJECT_AUTHORITATIVE}" in line
            self.assertIs(claims, expect_authority, line)


class TestThePlanParamInstrument(CustomTestCase):
    """#1273 B4r: the emitter that makes weg2xsn16's NEGATIVE slack answerable
    from the next boot's log, with no second campaign.

    weg2xsn16 printed ``slack_mib`` -107.7 / -115.4 / -115.2 / -110.9 on
    ``weights_draft`` while ``uncovered=0 short=0 missing=0`` and
    ``tms_answered=yes``. Both terms measure their own population correctly
    and ``slack_bytes`` is "PRINTED, NEVER COMPARED FOR EQUALITY" by design,
    so nothing gates it -- but a NEGATIVE slack is the one direction that
    cannot be allocator overhang: the plan claims more bytes than the tag
    holds. The named hypothesis (not diagnosed): a tied/shared embedding
    counted into the draft plan while living under the target's ``weights``
    tag, suggested by ~66 MiB/parameter on ``params=20`` against 515 MiB on
    ``params=114``.

    This class drives the REAL emitter over a torch module built to have that
    exact shape, so the instrument is proven to name the case before a boot
    is spent on it -- the desk-written-never-executed law applied to an
    instrument rather than to a fix.
    """

    @staticmethod
    def _model():
        import torch

        class Inner(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.w = torch.nn.Parameter(torch.zeros(4, 4))

        class M(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.here = Inner()
                self.tied = torch.nn.Parameter(torch.zeros(8, 8))

        return M()

    def _lines(self, planned, tag="weights_draft"):
        from sglang.srt.weg2 import weight_exchange as w

        model = self._model()
        return w.plan_param_lines(
            model, rank=2, tag=tag,
            planned_bytes_by_tag={tag: planned},
        )

    def test_a_parameter_the_plan_claims_but_another_tag_holds_reads_elsewhere(self):
        """THE HYPOTHESIS' OWN SHAPE. The live tensors of this model carry the
        BASE weights tag, so a draft plan claiming one of them must read
        ``verdict=elsewhere`` with the holding tag named -- which is exactly
        the reading weg2xsn16's log could not produce."""
        lines = self._lines({"tied": 256})
        self.assertEqual(len(lines), 1, lines)
        self.assertIn("verdict=elsewhere", lines[0])
        self.assertIn("name=tied", lines[0])
        self.assertIn("planned_mib=0.000", lines[0])
        self.assertIn("live_mib=0.000", lines[0])
        self.assertRegex(lines[0], r"live_tag=\S+")
        self.assertNotIn("live_tag=weights_draft", lines[0])

    def test_a_parameter_the_plan_claims_and_nothing_holds_reads_absent(self):
        lines = self._lines({"not.a.real.param": 1024})
        self.assertEqual(len(lines), 1, lines)
        self.assertIn("verdict=absent", lines[0])
        self.assertIn("live_tag=-", lines[0])

    def test_a_parameter_under_the_asked_tag_reads_here(self):
        """The control: without it the emitter could read ``elsewhere`` for
        everything and still pass the test above."""
        from sglang.srt.weg2 import weight_exchange as w

        model = self._model()
        live = [t for t in w.walk_live_tensors(model) if t.kind == w.PARAMETER]
        self.assertTrue(live, "the fixture model exposes no parameters")
        tag = live[0].tag
        lines = w.plan_param_lines(
            model, rank=2, tag=tag,
            planned_bytes_by_tag={tag: {live[0].name: live[0].nbytes}})
        self.assertEqual(len(lines), 1, lines)
        self.assertIn("verdict=here", lines[0])
        self.assertIn(f"live_tag={tag}", lines[0])

    def test_it_dumps_one_line_per_planned_parameter_sorted(self):
        """A per-parameter dump, not an aggregate: the population on the tag
        in question is 20, and choosing an aggregate before the finding is
        understood is how this campaign has lost boots."""
        lines = self._lines({"b": 2, "a": 1, "c": 3})
        self.assertEqual(len(lines), 3)
        names = [ln.split("name=")[1].split()[0] for ln in lines]
        self.assertEqual(names, ["a", "b", "c"])

    def test_every_line_carries_the_prefix_and_the_rank(self):
        from sglang.srt.weg2 import weight_exchange as w

        for line in self._lines({"tied": 1}):
            self.assertTrue(line.startswith(w.PLAN_PARAM_LINE_PREFIX), line)
            self.assertIn("rank=2", line)

    def test_an_empty_plan_emits_nothing_rather_than_a_zero_row(self):
        """An absence prints as no lines; a tag with no planned parameters is
        not a tag with one zero-byte parameter."""
        self.assertEqual(self._lines({}), [])
        self.assertEqual(self._lines(None if False else {}, tag="weights_0"), [])

    def test_it_is_wired_into_the_arming_path(self):
        """PRESENT-AND-WIRED, not present-but-unwired (#859's middle state,
        and the class B4l's ratchet exists for): the emitter must be called
        from the function that emits the cover lines, for EVERY tag."""
        from sglang.srt.weg2 import weight_exchange as w

        src = inspect.getsource(w.arm_coverage)
        self.assertIn("plan_param_lines(", src)
        tree = ast.parse(inspect.cleandoc(src))
        fn = next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef))
        calls = [n for n in ast.walk(fn) if isinstance(n, ast.Call)
                 and getattr(n.func, "id", "") == "plan_param_lines"]
        self.assertEqual(len(calls), 1, "expected exactly one call site")
        self.assertIn("tag=tag", ast.unparse(calls[0]),
                      "the call must run per tag, not for one hardcoded tag")


if __name__ == "__main__":
    unittest.main()
