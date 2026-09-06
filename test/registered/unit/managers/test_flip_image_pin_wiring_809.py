"""#809 G1-C3: the arm site and the abandon site, wired to the read-ahead.

THE GAP THIS CLOSES, measured rather than argued. The refill half of #809 is
pinned by ``unit/model_executor/test_flip_image_pin_809.py``, which drives
``FlipImagePin`` and ``_refill_from_pin_and_file`` DIRECTLY. Nothing reached
the three sites that make the feature happen in a boot, and four independent
mutations of them each left that suite fully green:

* ``PhaseFlipStacks.start_incoming_image_prefetch`` returning ``False``
  unconditionally -- the reader never arms and every leg reads the file
  exactly as it did before #809;
* ``PhaseFlipStacks.incoming_image``'s direction-to-layout mapping SWAPPED --
  the pin is filled with the OUTGOING layout, so every leg logs
  ``#809 FLIP IMAGE PIN STALE`` while the boot log still shows the pin armed
  and its post registered;
* ``PhaseFlipRuntime._enter_armed_state``'s call to the starter removed -- the
  read-ahead never begins, and the drain it exists to overlap stays idle;
* ``PhaseFlipRuntime._abandon_parked_flip``'s ``stop_prefetch`` removed -- an
  abandoned arm leaves a reader competing with serving for the pool.

That is the #742 silently-inert-flag class, which this slice's own code cites
three times as the reason for other decisions (``weights_arena.py:1980-1983``
verbatim: *"A pin the operator armed and that silently did not happen is the
#742 class, and it would be read off the acceptance as 'the prefetch did not
help' rather than as 'the prefetch was never there'."*). A feature whose
acceptance is a TIMING band cannot detect its own absence from the log -- an
inert pin and a pin that did not help print the same slow boot -- so the
wiring has to be pinned here instead.

THE MAPPING IS THE LOAD-BEARING ONE. ``incoming_image`` decides WHICH of the
two layout images is read ahead, and an inversion is the difference between a
pin that serves every leg and a pin that serves none. It is checked against
``refill``'s own direction handling, which is the consumer of the same fact.
"""

import ast
import inspect
import textwrap
import unittest

import torch

from sglang.srt.model_executor import weights_arena
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class _StubPin:
    """A pin recording what it was armed with. No buffer, no reader.

    ``install_flip_image_pin`` publishes whatever it is given, and the two
    sites under test call exactly ``start_prefetch`` and ``stop_prefetch``,
    so a real ``FlipImagePin`` would only add a thread to the assertions.
    """

    post_name = None

    def __init__(self, started=True):
        self.started = []
        self.stopped = 0
        self._answer = started

    def start_prefetch(self, image):
        self.started.append(image)
        return self._answer

    def stop_prefetch(self):
        self.stopped += 1
        return True


class _RaisingPin(_StubPin):
    def start_prefetch(self, image):
        self.started.append(image)
        raise RuntimeError("the pool went away")


def _stacks(image_pp=None, image_tp=None):
    """A ``PhaseFlipStacks`` carrying only the two images these methods read.

    The stub idiom this file's neighbours use (``test_flip_arm_snapshot_746``'s
    ``_runtime``): the dataclass's remaining fields are a tp_worker, an arena
    and two layouts, none of which the image accessors touch.
    """
    from sglang.srt.managers import phase_flip_boot as boot

    stacks = boot.PhaseFlipStacks.__new__(boot.PhaseFlipStacks)
    stacks.image_pp = image_pp
    stacks.image_tp = image_tp
    return stacks


def _image(nbytes=8):
    return torch.zeros(nbytes, dtype=torch.uint8)


class TestTheIncomingLayoutIsTheOneReadAhead(CustomTestCase):
    """G1-C3: which image the arm hands the pin, per direction."""

    def setUp(self):
        super().setUp()
        weights_arena.install_flip_image_pin(None)
        self.addCleanup(weights_arena.install_flip_image_pin, None)

    def test_each_direction_reads_ahead_the_layout_its_leg_will_stream(self):
        """PP_TO_TP fills the pin with the TP image, TP_TO_PP with the PP one.

        The direction of the answer is the whole content of the test: a
        swapped mapping fills the pin with the layout the arena is about to
        LEAVE, every leg refuses it as stale, and the boot log still shows an
        armed pin and a registered post.
        """
        from sglang.srt.layers.dcp.phase_flip_plan import PP_TO_TP, TP_TO_PP

        img_pp, img_tp = _image(), _image()
        stacks = _stacks(image_pp=img_pp, image_tp=img_tp)
        for direction, want, name in (
            (PP_TO_TP, img_tp, "tp"),
            (TP_TO_PP, img_pp, "pp"),
        ):
            with self.subTest(direction=direction):
                pin = _StubPin()
                weights_arena.install_flip_image_pin(pin)
                self.assertTrue(stacks.start_incoming_image_prefetch(direction))
                self.assertEqual(len(pin.started), 1, pin.started)
                self.assertIs(
                    pin.started[0],
                    want,
                    f"{direction} must read ahead the {name} image -- the "
                    f"layout its leg streams INTO the arena",
                )

    def test_the_accessor_and_the_leg_agree_on_the_mapping(self):
        """``incoming_image`` names the same layout ``refill`` streams.

        Two readers of one fact, so the pin cannot be filled from the image
        the leg does not use. ``refill``'s own source is quoted here rather
        than restated: it is the consumer, and it is what a boot obeys.
        """
        import inspect

        from sglang.srt.layers.dcp.phase_flip_plan import PP_TO_TP, TP_TO_PP
        from sglang.srt.managers import phase_flip_boot as boot

        src = inspect.getsource(boot.PhaseFlipStacks.refill)
        img_pp, img_tp = _image(), _image()
        stacks = _stacks(image_pp=img_pp, image_tp=img_tp)
        self.assertIs(stacks.incoming_image(PP_TO_TP), img_tp)
        self.assertIs(stacks.incoming_image(TP_TO_PP), img_pp)
        # `refill` names its incoming layout as the last argument of each
        # `_timed_arena_refill` call; both spellings must be present, or the
        # assertions above are being checked against a leg that moved.
        self.assertIn('"pp_to_tp"', src)
        self.assertIn('"tp_to_pp"', src)

    def test_an_unknown_direction_reads_nothing_ahead(self):
        pin = _StubPin()
        weights_arena.install_flip_image_pin(pin)
        stacks = _stacks(image_pp=_image(), image_tp=_image())
        self.assertFalse(stacks.start_incoming_image_prefetch("sideways"))
        self.assertEqual(pin.started, [])

    def test_without_the_two_file_arm_nothing_is_read_ahead(self):
        """The pin reads the incoming layout's OWN file, which one-file has not.

        The dangerous half is the direction whose image happens to EXIST:
        under a one-file stack ``image_tp`` may be set while ``image_pp`` is
        not, so a missing ``two_file_arm`` guard would arm the reader on the
        single rotating buffer's image -- the one every leg rewrites.
        """
        from sglang.srt.layers.dcp.phase_flip_plan import PP_TO_TP

        pin = _StubPin()
        weights_arena.install_flip_image_pin(pin)
        stacks = _stacks(image_pp=None, image_tp=_image())
        self.assertFalse(stacks.two_file_arm())
        self.assertFalse(stacks.start_incoming_image_prefetch(PP_TO_TP))
        self.assertEqual(pin.started, [])

    def test_with_no_pin_installed_the_arm_says_so(self):
        from sglang.srt.layers.dcp.phase_flip_plan import PP_TO_TP

        self.assertIsNone(weights_arena.flip_image_pin())
        stacks = _stacks(image_pp=_image(), image_tp=_image())
        self.assertFalse(stacks.start_incoming_image_prefetch(PP_TO_TP))

    def test_a_reader_that_cannot_start_does_not_break_the_arm(self):
        """A read-ahead never refuses a flip: the file is still the carrier."""
        from sglang.srt.layers.dcp.phase_flip_plan import PP_TO_TP

        pin = _RaisingPin()
        weights_arena.install_flip_image_pin(pin)
        stacks = _stacks(image_pp=_image(), image_tp=_image())
        self.assertFalse(stacks.start_incoming_image_prefetch(PP_TO_TP))
        self.assertEqual(len(pin.started), 1)


def _armed_runtime(stacks):
    """A ``PhaseFlipRuntime`` carrying only what ``_enter_armed_state`` reads."""
    from sglang.srt.managers import phase_flip_runtime as m

    class _Scheduler:
        pass

    scheduler = _Scheduler()
    scheduler.phase_flip_stacks = stacks

    rt = m.PhaseFlipRuntime.__new__(m.PhaseFlipRuntime)
    rt._pending = None
    rt._clock = lambda: 0.0
    rt._snapshot_parked_extent = lambda: None
    rt._at_arm_census_due = lambda: False
    rt._pool_census = lambda label, direction: None
    rt._enforce_exposure_at_seam = lambda label: None
    rt._census_scheduler = scheduler
    return rt


class TestTheArmStartsTheReadAhead(CustomTestCase):
    """G1-C3: the arm site is where the drain begins, so it is where the
    read starts. Without this the whole slice is inert in a boot."""

    def test_entering_the_armed_state_starts_the_read_ahead(self):
        from sglang.srt.layers.dcp.phase_flip_plan import PP_TO_TP

        seen = []
        stacks = _stacks(image_pp=_image(), image_tp=_image())
        stacks.start_incoming_image_prefetch = seen.append
        rt = _armed_runtime(stacks)
        rt._enter_armed_state(PP_TO_TP)
        self.assertEqual(
            seen,
            [PP_TO_TP],
            "the arm did not start the read-ahead, so the drain it exists to "
            "overlap is spent with the storage idle and the whole leg is read "
            "at the cutover as it was before #809",
        )
        self.assertEqual(rt._pending, PP_TO_TP)

    def test_a_scheduler_without_stacks_still_arms(self):
        """The read-ahead is not a precondition of arming, on any rank.

        A rank whose stacks are absent must arm exactly as its peers do --
        the alternative is one rank refusing an arm the group already took,
        which is the divergence this site's own comment forbids.
        """
        from sglang.srt.layers.dcp.phase_flip_plan import PP_TO_TP

        rt = _armed_runtime(None)
        rt._enter_armed_state(PP_TO_TP)
        self.assertEqual(rt._pending, PP_TO_TP)


class TestTheAbandonStopsTheReadAhead(CustomTestCase):
    """T-G1-6's second half, at the site that performs it."""

    def setUp(self):
        super().setUp()
        import logging

        from sglang.srt.managers import phase_flip_runtime as m

        # The abandon logs at ERROR by design. Silencing it is local to this
        # class, but the logger is process-global, so the level is restored
        # (the reason `test_flip_arm_snapshot_746` gives for the same dance).
        flip_logger = logging.getLogger(m.__name__)
        previous = flip_logger.level
        self.addCleanup(flip_logger.setLevel, previous)
        flip_logger.setLevel(logging.CRITICAL)
        weights_arena.install_flip_image_pin(None)
        self.addCleanup(weights_arena.install_flip_image_pin, None)

    def _parked(self):
        from sglang.srt.managers import phase_flip_runtime as m

        rt = m.PhaseFlipRuntime.__new__(m.PhaseFlipRuntime)
        rt._pending = m.PP_TO_TP
        rt._parked_extent = (1, 2)
        rt._armed_residents = {}
        rt._armed_at = 0.0
        rt._last_hold_reason = None
        rt._clock = lambda: 1.0
        rt.park_deadline_aborts = 0
        rt._phase = m.PHASE_PP
        rt._park_deadline_s = 30.0
        return rt

    def test_an_abandoned_arm_stops_the_reader_exactly_once(self):
        pin = _StubPin()
        weights_arena.install_flip_image_pin(pin)
        rt = self._parked()
        rt._abandon_parked_flip(0)
        self.assertEqual(
            pin.stopped,
            1,
            "the abandoned arm left the reader running, so it goes on "
            "pulling the image file while serving resumes on the same pool",
        )
        self.assertIsNone(rt._pending)

    def test_an_abandon_without_a_pin_is_still_an_abandon(self):
        self.assertIsNone(weights_arena.flip_image_pin())
        rt = self._parked()
        rt._abandon_parked_flip(0)
        self.assertIsNone(rt._pending)
        self.assertIsNone(rt._parked_extent)


def _boot_stack_tree():
    """The AST of ``build_phase_flip_tp_stack``, dedented so it parses alone."""
    from sglang.srt.managers import phase_flip_boot

    body = textwrap.dedent(inspect.getsource(phase_flip_boot.build_phase_flip_tp_stack))
    return ast.parse(body).body[0]


def _calls_named(tree, name):
    return [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == name
    ]


class TestTheBootCreatesThePin(CustomTestCase):
    """The BOOT half of the wiring, which the arm-site tests do not reach.

    THE SAME GAP, ONE STEP EARLIER. The class docstring above records four
    mutations of the ARM sites that left the slice suite green; three more at
    the CREATION site do the same, and one of them makes the whole slice inert
    in a boot without changing a single string the log prints:

    * ``if flip_image_pin_enabled():`` short-circuited (``if False and ...``)
      -- no pin is ever created, ``flip_image_pin()`` is ``None`` on every
      leg, and every refill falls through to ``source=file`` with no #809
      line at all;
    * ``max(...)`` over the two layout images turned into ``min(...)`` -- the
      buffer is sized to the SMALLER image, so exactly one of the two flip
      directions silently never fits (``weights_arena.py``'s
      ``if int(meta.nbytes) > self.nbytes:``) while the boot still logs the
      post as registered;
    * ``require_pin_preconditions()`` dropped -- the FILE_BACKED/TWO_FILE
      refusal never runs at boot, so an armed pin without its carrier is
      discovered at the first flip instead of at the boot.

    WHY THIS IS STRUCTURAL AND NOT A STRING MATCH. A substring pin would
    survive the first mutation, because ``if False and
    flip_image_pin_enabled():`` still contains every name the pin looks for.
    The guard is therefore read as a TREE: the call must be the whole
    condition, not one term of a wider one.
    """

    def test_the_pin_is_created_under_its_own_flag_and_nothing_wider(self):
        tree = _boot_stack_tree()
        guards = [
            node
            for node in ast.walk(tree)
            if isinstance(node, ast.If)
            and _calls_named(node.test, "flip_image_pin_enabled")
        ]
        self.assertEqual(
            len(guards),
            1,
            "#809's creation site is not exactly one guarded block",
        )
        guard = guards[0]
        self.assertIsInstance(
            guard.test,
            ast.Call,
            "the guard is not the flag call ITSELF but a wider expression, so "
            "a term beside it can make the whole slice inert while every name "
            "the log and a grep look for is still present",
        )
        self.assertEqual(ast.unparse(guard.test), "flip_image_pin_enabled()")
        self.assertEqual(
            guard.orelse,
            [],
            "the creation site grew an else the slice has no shape for",
        )

    def test_the_buffer_is_sized_to_the_larger_of_the_two_layout_images(self):
        """ONE buffer serves BOTH directions, so the size is a MAX.

        A MIN fits one direction and silently refuses the other: the leg for
        the larger layout finds ``meta.nbytes > self.nbytes`` and reads the
        file, while the boot log still shows the post registered and the
        acceptance reads it as "the prefetch did not help".
        """
        tree = _boot_stack_tree()
        made = _calls_named(tree, "create_flip_image_pin")
        self.assertEqual(len(made), 1, "the pin is created somewhere else as well")
        self.assertEqual(len(made[0].args), 1)
        self.assertEqual(
            ast.unparse(made[0].args[0]),
            "max(int(image_pp.numel()), int(image_tp.numel()))",
            "the pin is not sized to the larger of the two layout images",
        )

    def test_the_creation_sits_inside_the_flag_and_inside_the_two_file_arm(self):
        """The call is REACHED only under the flag, and only with two files.

        Both containments are load-bearing: outside the flag the default tree
        stops being byte-identical, and outside the two-file arm there is no
        per-layout image file for the read-ahead to read.
        """
        tree = _boot_stack_tree()
        guards = [
            node
            for node in ast.walk(tree)
            if isinstance(node, ast.If)
            and _calls_named(node.test, "flip_image_pin_enabled")
        ]
        self.assertEqual(len(guards), 1)
        self.assertEqual(
            len(_calls_named(guards[0], "create_flip_image_pin")),
            1,
            "the creation escaped its own flag",
        )
        two_file = [
            node
            for node in ast.walk(tree)
            if isinstance(node, ast.If)
            and isinstance(node.test, ast.Name)
            and node.test.id == "two_file"
        ]
        self.assertTrue(
            any(_calls_named(node, "create_flip_image_pin") for node in two_file),
            "the pin is created outside the two-file arm, which is the only "
            "arm that has a per-layout image file to read ahead from",
        )

    def test_the_boot_asks_the_precondition_beside_the_two_file_one(self):
        """#1078's refusal and #809's run at the same instant, unguarded.

        Dropped, the FILE_BACKED/TWO_FILE refusal never runs at boot and a
        misconfigured pin is discovered at the first flip instead -- inside
        the no-return window rather than in a boot message.
        """
        tree = _boot_stack_tree()
        asked = _calls_named(tree, "require_pin_preconditions")
        self.assertEqual(
            len(asked), 1, "the #809 precondition is not asked at boot exactly once"
        )
        top = [
            node.value.func.id
            for node in tree.body
            if isinstance(node, ast.Expr)
            and isinstance(node.value, ast.Call)
            and isinstance(node.value.func, ast.Name)
        ]
        self.assertIn(
            "require_pin_preconditions",
            top,
            "the precondition moved under a branch, so a configuration that "
            "does not take that branch is never refused",
        )
        self.assertIn("require_two_file_preconditions", top)
        self.assertLess(
            top.index("require_two_file_preconditions"),
            top.index("require_pin_preconditions"),
            "#1078's carrier is refused before #809's read-ahead of it",
        )


if __name__ == "__main__":
    unittest.main()
