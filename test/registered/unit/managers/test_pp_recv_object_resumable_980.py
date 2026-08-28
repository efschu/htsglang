"""#980: the bounded ``recv_object`` must be RESUMABLE, not merely bounded.

WHAT THIS FILE IS FOR, and why its arms are shaped as mutants rather than as
a before/after. #980 shipped 1202 lines across seven files with no tests, and
its central module (`sglang/srt/distributed/pp_object_recv.py`) is NEW. A
"red-first against the previous commit" for new code is the weak proof I have
already reported twice in this window: every arm dies on `ImportError` /
`AttributeError`, which is equally red against a module whose functions are
all `pass`. So the honest substitute is used here instead -- each invariant
is paired with a MUTANT that breaks exactly that invariant and nothing else,
and the arm is required to go red for it. That is the same can-fail
discipline the surrounding suites use, applied where a predecessor commit
cannot supply the red.

THE ONE INVARIANT EVERYTHING ELSE SERVES. `recv_object` is a two-step
protocol -- an `irecv` of a size, then an `irecv` of exactly that many bytes.
The module's own docstring states the law: "Once the size header has been
received, the payload is already on the wire and MUST be taken off it; a
receiver that gives up mid-frame and later re-posts reads a payload AS a
size, and every later message on that stream is garbage."

So a bound that ABANDONS is worse than no bound at all: the unbounded wait
merely hangs one rank, while an abandoning bound silently corrupts every
subsequent message on that stream. The arms below therefore do not check
"does it time out" -- that part is easy and uninteresting. They check that
the `irecv` calls are NOT repeated: one size post and one payload post per
object, no matter how many steps expire and no matter how many times the
caller is handed a stall and comes back.

HERMETIC. No gloo, no ranks, no CUDA: `dist.irecv` and `ParkedWait` are
stubbed inside the module's own namespace, so the state machine under test is
the shipped one and only the transport is stood in.
"""

import inspect
import pickle
import types
import unittest

import torch

from sglang.srt.distributed import pp_object_recv as m
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=10, suite="base-a-test-cpu")

PAYLOAD = {"boot": 7, "shape": "recv_object wedge"}


class _FakeWork:
    """A `Work` handle. Deliberately has NO `wait` -- see the #829 arm."""


class _FakeParked:
    """Stands in for `ParkedWait`, with a scripted sequence of join results.

    `join(budget)` pops the next scripted result: True = the step completed,
    False = the budget expired with the receive STILL POSTED, an exception
    instance = the transport raised inside the parked wait.
    """

    instances = []

    def __init__(self, work, site):
        self.work = work
        self.site = site
        self.joins = 0
        self.script = []
        _FakeParked.instances.append(self)

    def join(self, budget):
        self.joins += 1
        if not self.script:
            return True
        step = self.script.pop(0)
        if isinstance(step, BaseException):
            raise step
        return step


class _Recorder:
    """Records every `irecv` the frame posts -- the thing that must not repeat."""

    def __init__(self, payload=PAYLOAD):
        self.posts = []
        self._blob = None
        self.payload = payload

    def irecv(self, tensor, src=None, group=None, tag=None):
        self.posts.append({"tensor": tensor, "src": src, "tag": tag})
        if tensor.dtype == torch.long:
            # The size header. Fill it as the peer would.
            self._blob = pickle.dumps(self.payload)
            tensor[0] = len(self._blob)
        else:
            # The payload buffer.
            assert self._blob is not None, "payload posted before the size"
            tensor[:] = torch.frombuffer(bytearray(self._blob), dtype=torch.uint8)
        return _FakeWork()

    @property
    def size_posts(self):
        return [p for p in self.posts if p["tensor"].dtype == torch.long]

    @property
    def payload_posts(self):
        return [p for p in self.posts if p["tensor"].dtype == torch.uint8]


class _Harness(unittest.TestCase):
    def setUp(self):
        _FakeParked.instances = []
        self.rec = _Recorder()
        self._saved_dist = m.dist
        self._saved_parked = m.ParkedWait
        m.dist = types.SimpleNamespace(irecv=self.rec.irecv)
        m.ParkedWait = _FakeParked
        for k in list(m.RECV_OBJECT_STATS):
            m.RECV_OBJECT_STATS[k] = 0
        self.addCleanup(self._restore)

    def _restore(self):
        m.dist = self._saved_dist
        m.ParkedWait = self._saved_parked

    def _frame(self):
        return m.ObjectRecvFrame(
            group=object(),
            src_global=1,
            tag=7,
            site="test/recv_object",
            rank_desc="PP1",
        )

    def _script(self, *results):
        """Apply a join script to every ParkedWait the frame creates."""
        pending = list(results)

        original_init = _FakeParked.__init__

        def _init(inner, work, site):
            original_init(inner, work, site)
            inner.script = pending.pop(0) if pending else []

        _FakeParked.__init__ = _init
        self.addCleanup(lambda: setattr(_FakeParked, "__init__", original_init))


class TheFrameIsResumableAcrossExpiredSteps(_Harness):
    """A step may expire; the receive it belongs to may not be restarted."""

    def test_an_expired_size_step_does_not_re_post_the_irecv(self):
        """THE LAW: one size post per object, however many steps expire."""
        self._script([False, False, True], [True])
        frame = self._frame()

        obj = frame.receive(step_budget_s=0.01, abort_after_s=0.0)

        self.assertEqual(obj, PAYLOAD)
        self.assertEqual(
            len(self.rec.size_posts),
            1,
            "the size irecv was posted more than once: two expired steps "
            "restarted the receive instead of resuming it. A re-post reads "
            "the PAYLOAD as a size and every later message on this stream "
            "is garbage -- the corruption the module exists to prevent",
        )
        self.assertEqual(len(self.rec.payload_posts), 1)
        self.assertEqual(frame.state, "idle", "a taken frame returns to idle")

    def test_the_payload_step_resumes_the_same_way(self):
        self._script([True], [False, False, True])
        frame = self._frame()

        self.assertEqual(frame.receive(0.01, 0.0), PAYLOAD)
        self.assertEqual(len(self.rec.size_posts), 1)
        self.assertEqual(
            len(self.rec.payload_posts),
            1,
            "the payload irecv was re-posted after an expired step -- the "
            "bytes are already on the wire and must be taken off once",
        )

    def test_expiries_are_counted_and_reset_by_take(self):
        self._script([False, True], [True])
        frame = self._frame()
        frame.receive(0.01, 0.0)
        self.assertEqual(
            frame.expiries, 0, "take() resets the frame for the next object"
        )
        self.assertGreaterEqual(m.RECV_OBJECT_STATS["step_expired"], 1)
        self.assertEqual(m.RECV_OBJECT_STATS["completed"], 1)


class AStallIsRaisedWithoutLosingTheFrame(_Harness):
    """The abort must hand the caller an error AND keep the receive alive."""

    def test_the_stall_raises_and_the_same_receive_then_completes(self):
        """The docstring's promise, tested: 'The frame is RESUMABLE'.

        This is the arm that separates #980 from a plain timeout. Raising is
        the easy half; the module claims the caller may come back and
        continue the SAME receive. If the raise abandoned the frame, this
        second call would post a second size irecv and misframe the stream.
        """
        self._script([False, False, False, True], [True])
        frame = self._frame()

        with self.assertRaises(m.ObjectRecvStalled) as caught:
            # A negative-ish abort window: the first expired step is already
            # past it, so the stall is raised on the first expiry.
            frame.receive(step_budget_s=0.01, abort_after_s=1e-9)

        message = str(caught.exception)
        self.assertIn("#980 RECV-OBJECT STALL", message)
        self.assertIn("RESUMABLE", message)
        self.assertIn(m.ENV_ABORT_AFTER, message)

        self.assertNotEqual(
            frame.state,
            "idle",
            "the raise reset the frame to idle -- the posted receive is now "
            "orphaned and the next call will re-post over a live wire",
        )
        posts_at_raise = len(self.rec.posts)

        # The caller comes back, as the message tells it to.
        self.assertEqual(frame.receive(step_budget_s=0.01, abort_after_s=0.0), PAYLOAD)

        self.assertEqual(
            len(self.rec.size_posts),
            1,
            "the resumed receive posted a SECOND size irecv -- the frame was "
            "not resumed, it was restarted",
        )
        self.assertGreaterEqual(len(self.rec.posts), posts_at_raise)
        self.assertEqual(m.RECV_OBJECT_STATS["aborted"], 1)

    def test_a_transport_error_is_not_dressed_up_as_a_stall(self):
        """#734: a DEAD peer must stay distinguishable from a SLOW one.

        Converting this into `ObjectRecvStalled` would tell the caller to
        retry a stream whose peer is gone.
        """
        boom = RuntimeError("Connection closed by peer")
        self._script([boom], [True])
        frame = self._frame()

        with self.assertRaises(RuntimeError) as caught:
            frame.receive(0.01, 10.0)

        self.assertNotIsInstance(
            caught.exception,
            m.ObjectRecvStalled,
            "a transport failure was reported as a #980 stall, which tells "
            "the caller the frame is resumable when the peer is gone",
        )
        self.assertIn("Connection closed by peer", str(caught.exception))

    def test_abort_disabled_waits_for_ever_exactly_as_before_980(self):
        """`abort_after_s <= 0` restores the pre-#980 unbounded wait.

        The default path of every boot that does not opt in must not acquire
        a new way to fail -- it acquires only the log line.
        """
        self._script([False] * 25 + [True], [True])
        frame = self._frame()

        self.assertEqual(frame.receive(step_budget_s=0.01, abort_after_s=0.0), PAYLOAD)
        self.assertEqual(len(self.rec.size_posts), 1)
        self.assertEqual(m.RECV_OBJECT_STATS["aborted"], 0)
        self.assertGreaterEqual(m.RECV_OBJECT_STATS["step_expired"], 25)


class TheseArmsCanFail(_Harness):
    """Mutants, because a new module has no predecessor to be red against.

    Each one breaks exactly one invariant and must drive its own arm red. An
    invariant whose violation cannot be constructed is not being measured.
    """

    def test_a_restarting_frame_is_caught_by_the_re_post_assertion(self):
        """The mutant that matters: abandon and re-arm on an expired step."""
        self._script([False, True], [True])
        frame = self._frame()

        original_arm = m.ObjectRecvFrame._arm
        original_advance = m.ObjectRecvFrame.advance

        def _restarting_advance(inner, step_budget_s):
            # The defect: treat an expired step as "start over" rather than
            # "continue" -- exactly what a naive timeout+retry would do.
            done = original_advance(inner, step_budget_s)
            if not done:
                inner._state = "idle"
            return done

        m.ObjectRecvFrame.advance = _restarting_advance
        self.addCleanup(
            lambda: setattr(m.ObjectRecvFrame, "advance", original_advance)
        )
        self.addCleanup(lambda: setattr(m.ObjectRecvFrame, "_arm", original_arm))

        frame.receive(step_budget_s=0.01, abort_after_s=0.0)

        self.assertGreater(
            len(self.rec.size_posts),
            1,
            "the mutant did NOT re-post the size irecv, so the assertion in "
            "`test_an_expired_size_step_does_not_re_post_the_irecv` is not "
            "actually watching for a restart and proves nothing",
        )

    def test_a_stall_that_resets_the_frame_is_caught(self):
        """The mutant for the resumability-after-raise arm."""
        self._script([False, True], [True])
        frame = self._frame()

        with self.assertRaises(m.ObjectRecvStalled):
            frame.receive(step_budget_s=0.01, abort_after_s=1e-9)

        # The mutant: the caller's error handler "cleans up" the frame, which
        # is the abandonment the module forbids.
        frame._state = "idle"
        frame.receive(step_budget_s=0.01, abort_after_s=0.0)

        self.assertGreater(
            len(self.rec.size_posts),
            1,
            "resetting the frame after a stall did NOT cause a second size "
            "post, so the resumability assertion is not measuring resumption",
        )


class TheBoundNeverClosesTheGlooPair(unittest.TestCase):
    """#829, asserted against the source because it is a PROHIBITION.

    Measured and pinned in `ParkedWait`'s own docstring: `Work.wait(timeout=)`
    does fire on time and CLOSES THE GLOO PAIR while doing it -- the waiter
    then gets "Application timeout caused pair closure" from every later call
    and the PEER gets "Connection closed by peer" from its next send. One
    expired wait takes the whole group down, on both sides. That is why this
    module bounds a JOIN and never a wait, and it is a property no runtime
    arm can observe: the wrong call would simply work in a hermetic test and
    take down a real boot.
    """

    def test_the_module_never_calls_wait_with_a_timeout(self):
        """Parsed, not grepped -- the first version of this arm was a FALSE
        POSITIVE and is worth recording rather than quietly rewriting.

        It asserted `"wait(timeout" not in source`, and went red on the
        module's own docstring, which explains at length why it does NOT do
        that: "``Work.wait(timeout=...)`` does fire on time -- and CLOSES THE
        GLOO PAIR while doing it." The prohibition was being checked against
        prose, so the better a module documented the hazard, the more surely
        the arm failed. Same class as the reds-for-the-wrong-reason found
        elsewhere in this window, this time in my own test.

        The AST walk asks the real question: is there a CALL to `.wait(...)`
        anywhere in the module that passes a `timeout` keyword.
        """
        import ast

        tree = ast.parse(inspect.getsource(m))
        offenders = [
            node.lineno
            for node in ast.walk(tree)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "wait"
            and any(kw.arg == "timeout" for kw in node.keywords)
        ]
        self.assertEqual(
            offenders,
            [],
            "#829: a timed `Work.wait` closes the gloo pair and kills the "
            f"group on both sides. Offending call(s) at line(s) {offenders}. "
            "The bound belongs on the ParkedWait join",
        )

    def test_that_arm_would_notice_a_real_timed_wait(self):
        """Can-fail for the AST arm, since the grep version could not fail
        honestly. A module that DOES call `work.wait(timeout=...)` must be
        detected -- otherwise the prohibition above is decoration."""
        import ast

        mutant = ast.parse("def f(work):\n    return work.wait(timeout=5.0)\n")
        found = [
            node.lineno
            for node in ast.walk(mutant)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "wait"
            and any(kw.arg == "timeout" for kw in node.keywords)
        ]
        self.assertEqual(found, [2], "the detector cannot see the thing it forbids")

    def test_the_frame_owns_one_parked_wait_per_posted_receive(self):
        """'ONE FRAME PER STREAM, and the frame owns both the Work and its
        ParkedWait' -- two parked waits on one posted receive is the
        misframing this class exists to avoid."""
        src = inspect.getsource(m.ObjectRecvFrame)
        self.assertEqual(
            src.count("ParkedWait("),
            2,
            "exactly two ParkedWait constructions belong in the frame -- one "
            "for the size step and one for the payload step",
        )


if __name__ == "__main__":
    unittest.main()
