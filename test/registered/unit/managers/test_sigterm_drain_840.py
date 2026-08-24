"""#840: SIGTERM must actually end the instance, and it did not.

THE SPECIMEN. ``/spinning/evidence-665-f1/window4A_teardown_hang_2114/``,
2026-08-23 21:15, a PP=3 boot with ``--scheduler-distributed-teardown`` armed.
SIGTERM reached the parent at 21:15:21 and the cards were still held when the
dump was taken 17 s later; they only came free ~3 s after an explicit TERM to
each rank PID. The four drain lines in that log are the whole story:

    21:15:21  Remaining number of requests 4   [fa8d, ed7c, fcc9, fab8]
    21:15:26  Remaining number of requests 3   [fa8d,       fcc9, fab8]
    21:15:31  Remaining number of requests 4   [fa8d,       fcc9, fab8, 873e]
    21:15:36  Remaining number of requests 3   [fa8d,       fcc9, fab8]

``873e6e9d2d324804a657a74abb328fe1`` appears for the first time at 21:15:31 --
a request ADMITTED ten seconds after the server was told to shut down. The
count does not converge because nothing stops it from being refilled.

THE OMISSION, in two halves, and neither half is sufficient alone:

* ``gracefully_exit`` gated exactly one thing in the whole tree -- ``/health``
  (``http_server.py`` and ``grpc_bridge.py``). The request entrypoints never
  consulted it, so every route kept admitting into ``rid_to_state`` for the
  entire drain.
* the drain loop in ``sigterm_watchdog`` had NO deadline. Its only exits were
  "zero requests left", a failed health check, and an env var. So a drain that
  is being refilled is a drain that never ends -- and ``ShutdownReq`` is
  dispatched to the schedulers only AFTER that loop breaks. The schedulers
  therefore never leave their event loop, the ``finally`` that runs
  ``release_distributed`` never runs, and the GPUs stay held.

The barlink frames in the specimen's py-spy dumps (``_wait_ctl_event`` ->
``check_aborted``) are NOT the cause and no test here pins them: that wait is
already bounded (#818), and all three ranks kept emitting round-cadence
PHASE-POLICY lines at 21:15:24 and 21:15:30 -- a rank parked in a collective
cannot do that. The host thread simply spends most of a healthy collective
inside that poll's 0.5 ms sleep, which is where a sampler finds it.

WHAT IS PINNED HERE, and how each one can fail:

* the refill gate REFUSES a new request once the drain has started -- the fix;
* it does NOT fire while the server is serving normally. This is the dangerous
  direction: a gate that trips early refuses live traffic, which is worse than
  the hang it prevents. Pinned explicitly, in both directions;
* the drain TERMINATES even when requests keep arriving, and reaches
  ``ShutdownReq`` + the kill path;
* a HEALTHY drain is not cut short: one that empties before the deadline exits
  by the clean path and abandons nobody;
* the budget-0 mode restores the pre-#840 unbounded loop verbatim, so the
  change can be bisected against.

Hermetic: no GPU, no sockets, no subprocesses. Every collaborator is a stub and
the two functions under test are invoked UNBOUND against it, which is this
suite's documented pattern for methods whose real ``self`` needs a live server.
"""

import asyncio
import types
import unittest

from sglang.srt.managers import shutdown_gate
from sglang.srt.managers.shutdown_gate import ServerShuttingDown
from sglang.srt.managers.tokenizer_manager import TokenizerManager
from sglang.test.test_utils import CustomTestCase


class _Obj:
    """The shape ``_init_req_state`` reads off a request object."""

    def __init__(self, rid="rid-0"):
        self.rid = rid
        self.is_single = True
        self.received_time = 0.0
        self.external_trace_header = None
        self.bootstrap_room = None


class _Args:
    def __init__(self, drain_timeout_s=None):
        self.shutdown_drain_timeout_s = drain_timeout_s


class _Manager:
    """Stands in for TokenizerManager across both functions under test."""

    def __init__(self, *, gracefully_exit=False, rids=(), drain_timeout_s=None):
        self.gracefully_exit = gracefully_exit
        self.enable_trace = False
        self.disaggregation_mode = None
        self.rid_to_state = {rid: object() for rid in rids}
        self.server_args = _Args(drain_timeout_s)
        self.server_status = None
        self._subprocess_watchdog = None
        self.dispatched = []
        self.forced = 0
        self.dumped = 0

    # -- collaborators the drain calls -------------------------------------
    def _dispatch_to_scheduler(self, req):
        self.dispatched.append(req)

    def force_exit_handler(self):
        self.forced += 1

    def dump_requests_before_crash(self):
        self.dumped += 1


class _ServerStatus:
    """`UnHealthy` must compare unequal to the stub's `server_status=None`."""

    UnHealthy = "unhealthy"


class ShutdownGateTest(CustomTestCase):
    """The refill gate: a request that arrives during the drain is refused."""

    def test_refuses_new_request_once_the_drain_has_started(self):
        mgr = _Manager(gracefully_exit=True)
        with self.assertRaises(ServerShuttingDown):
            TokenizerManager._init_req_state(mgr, _Obj())
        self.assertEqual(
            mgr.rid_to_state,
            {},
            "the refused request must not be recorded: an entry here is a "
            "request the drain would then wait for, which is the #840 refill "
            "in a different disguise.",
        )

    def test_does_not_fire_while_serving(self):
        # THE DANGEROUS DIRECTION. A gate that trips on a healthy server
        # refuses live traffic; that is a worse outcome than the hang.
        mgr = _Manager(gracefully_exit=False)
        TokenizerManager._init_req_state(mgr, _Obj("rid-live"))
        self.assertIn("rid-live", mgr.rid_to_state)

    def test_missing_flag_admits_rather_than_raising(self):
        # The gate reads the flag with a defaulting ``getattr``, because the
        # two failure directions are not symmetric: on a request hot path,
        # degrading to "admit" restores the pre-#840 behaviour, while an
        # AttributeError would turn every request on a HEALTHY server into a
        # 500. Pinned so the default cannot be tightened without a decision.
        mgr = _Manager()
        del mgr.gracefully_exit
        TokenizerManager._init_req_state(mgr, _Obj("rid-no-flag"))
        self.assertIn("rid-no-flag", mgr.rid_to_state)

    def test_the_flag_the_gate_reads_is_really_set_by_init(self):
        # THE ANTI-INERTNESS PIN. The defaulting ``getattr`` above is what
        # makes a rename survivable -- and it is also what would make a rename
        # SILENT, leaving the gate permanently inert while every test that
        # drives it through a stub still passes. So the name is pinned against
        # the real constructor's source, not against a stub.
        import inspect

        # Against ``init_running_status``, which is where the real constructor
        # establishes it -- NOT against ``__init__``, which does not touch it.
        # Asserted by name so that moving the assignment to another method is
        # itself a failure worth reading, rather than a silently inert gate.
        src = inspect.getsource(TokenizerManager.init_running_status)
        self.assertIn(
            "self.gracefully_exit = False",
            src,
            "TokenizerManager.init_running_status no longer sets "
            "`gracefully_exit`. The #840 refill gate reads that exact name "
            "with a defaulting getattr, so a rename does not raise -- it makes "
            "the gate inert and the SIGTERM drain unable to converge again. "
            "Rename the read in `_init_req_state` in the same change.",
        )

    def test_refusal_is_a_value_error(self):
        # Every request entrypoint in this tree already has an
        # `except ValueError` arm. Inheriting from it is what makes the
        # refusal reach the client as a response on routes that have never
        # heard of this type -- including ones added after this lands.
        self.assertTrue(issubclass(ServerShuttingDown, ValueError))


class DrainTerminationTest(CustomTestCase):
    """The drain loop: bounded, and bounded only where it should be."""

    def setUp(self):
        super().setUp()
        self._poll = shutdown_gate.DRAIN_POLL_INTERVAL_S
        # Drive many drain ticks without sleeping minutes of wall clock.
        shutdown_gate.DRAIN_POLL_INTERVAL_S = 0.01

    def tearDown(self):
        shutdown_gate.DRAIN_POLL_INTERVAL_S = self._poll
        super().tearDown()

    def _run_drain(self, mgr, *, patches, wait_s=5.0):
        """Run ``sigterm_watchdog`` unbound, with the exit path neutralised."""
        import sglang.srt.managers.tokenizer_manager as tm

        saved = {name: getattr(tm, name) for name in patches}
        for name, value in patches.items():
            setattr(tm, name, value)
        try:
            return asyncio.run(
                asyncio.wait_for(TokenizerManager.sigterm_watchdog(mgr), wait_s)
            )
        finally:
            for name, value in saved.items():
                setattr(tm, name, value)

    def _exit_patches(self, killed):
        return {
            "kill_process_tree": lambda *a, **k: killed.append(a),
            "collect_scheduler_processes": lambda *a, **k: [],
            "ServerStatus": _ServerStatus,
        }

    def test_drain_ends_even_while_requests_keep_arriving(self):
        # THE FIX. The specimen's refill, reproduced: the count never reaches
        # zero because something keeps putting entries back. Before #840 this
        # loop had no deadline and the coroutine below never returned.
        mgr = _Manager(gracefully_exit=True, rids=("a", "b"), drain_timeout_s=0.2)
        killed = []
        with self.assertRaises(SystemExit):
            self._run_drain(mgr, patches=self._exit_patches(killed))
        self.assertEqual(
            len(mgr.dispatched),
            1,
            "the drain must reach ShutdownReq. Until it does, the schedulers "
            "never leave their event loop and the cards stay held.",
        )
        self.assertTrue(killed, "the kill path must run after the dispatch")

    def test_healthy_drain_is_not_cut_short(self):
        # THE OTHER DANGEROUS DIRECTION. A drain that is converging must exit
        # by the clean path, not by the deadline -- otherwise the deadline is
        # what ends normal shutdowns and in-flight requests are lost for no
        # reason. A generous budget with an emptying queue must never expire.
        mgr = _Manager(gracefully_exit=True, rids=("a",), drain_timeout_s=30.0)

        original = TokenizerManager._init_req_state  # keep the import honest
        self.assertTrue(callable(original))

        # One tick later the request finishes, exactly as a real drain does.
        async def _finish_soon():
            await asyncio.sleep(0.02)
            mgr.rid_to_state.clear()

        killed = []
        import sglang.srt.managers.tokenizer_manager as tm

        patches = self._exit_patches(killed)
        saved = {name: getattr(tm, name) for name in patches}
        for name, value in patches.items():
            setattr(tm, name, value)

        async def _both():
            task = asyncio.ensure_future(_finish_soon())
            try:
                await asyncio.wait_for(TokenizerManager.sigterm_watchdog(mgr), 5.0)
            finally:
                task.cancel()

        try:
            with self.assertRaises(SystemExit):
                asyncio.run(_both())
        finally:
            for name, value in saved.items():
                setattr(tm, name, value)

        self.assertEqual(len(mgr.dispatched), 1)
        self.assertEqual(
            mgr.forced,
            0,
            "a converging drain must not take the force-exit path",
        )

    def test_budget_zero_restores_the_unbounded_loop(self):
        # The bisecting aid, and the proof that the deadline is what ends the
        # first test rather than some other new exit: with the budget off, the
        # same refilling drain must still hang.
        mgr = _Manager(gracefully_exit=True, rids=("a",), drain_timeout_s=0.0)
        killed = []
        with self.assertRaises(asyncio.TimeoutError):
            self._run_drain(mgr, patches=self._exit_patches(killed), wait_s=0.5)
        self.assertEqual(mgr.dispatched, [])


class DrainBudgetTest(CustomTestCase):
    """The budget helper's own arithmetic."""

    def test_default_applies_when_unset(self):
        self.assertEqual(
            shutdown_gate.drain_timeout_s(_Args(None)),
            shutdown_gate.DEFAULT_DRAIN_TIMEOUT_S,
        )

    def test_default_applies_when_the_attribute_is_absent(self):
        self.assertEqual(
            shutdown_gate.drain_timeout_s(types.SimpleNamespace()),
            shutdown_gate.DEFAULT_DRAIN_TIMEOUT_S,
        )

    def test_explicit_zero_is_kept_as_unbounded(self):
        # `0` must survive as 0 and not be swallowed by a falsy-default, or the
        # documented bisecting mode silently becomes the bounded one.
        self.assertEqual(shutdown_gate.drain_timeout_s(_Args(0)), 0.0)

    def test_negative_is_treated_as_unbounded(self):
        self.assertEqual(shutdown_gate.drain_timeout_s(_Args(-1)), 0.0)

    def test_explicit_value_is_honoured(self):
        self.assertEqual(shutdown_gate.drain_timeout_s(_Args(7.5)), 7.5)


if __name__ == "__main__":
    unittest.main()
