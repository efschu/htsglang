"""#718: device-tier HiCache is bound to the pool it was BUILT with.

REACHABILITY FIRST, because it decides what this file is for.

``HiCacheController`` exists only when a hierarchical cache is built. Today's
serving flagset (``evidence-665-f1/argv_*.txt``) carries ``--enable-phase-flip``
and its family and NO hicache flags at all, so no controller is constructed and
none of this is reachable on the live line. The hazard is LATENT, and it stops
being latent on exactly the boot #703/#706 are aiming at -- flip plus
hierarchical cache plus the file backend -- because nothing refuses that
combination any more (the #630 blocker was removed from both the boot-time and
the runtime guard lists on purpose, so a prefix cache could ride the flip).

So this suite pins three things:

1. THE PIN: with the flip inactive -- every non-flipping deployment, and the PP
   phase of a flipping one -- the guard is invisible. That is what makes the
   default path byte-identical.
2. THE HAZARD IS REAL WHEN THE COMBINATION IS BOOTED: while the flip routes to
   its TP stack, both device transfers are refused instead of running against
   the wrong pool.
3. THE SHAPE, stated as a test rather than a comment: the refusal returns None
   from ``write``/``load`` BEFORE touching the pools, so a controller whose
   pools would explode if touched still returns cleanly. That is the executable
   form of "it does not copy against the bound pool".

Why the guard has no flag: ``phase_flip_tp_routing_active`` is False whenever
the flip's secondary groups were never built, so a deployment without
``--enable-phase-flip`` cannot enter the guarded state at all.
"""

import unittest

import torch

from sglang.srt.managers.cache_controller import HiCacheController
from sglang.srt.mem_cache import hicache_phase_guard as guard
from sglang.test.test_utils import CustomTestCase


class _Exploding:
    """A pool that fails loudly if the guard lets anything through."""

    def alloc(self, n):  # pragma: no cover - reaching this IS the failure
        raise AssertionError(
            "device-tier I/O ran while the flip was routing to the TP stack: "
            "the copy would have used the pool this controller was built with, "
            "which is not the pool the model is using"
        )


class _Stub:
    """The parts of the controller the guarded prologue can see."""

    # #923 added a second prologue question -- "is the row this copy would read
    # a row of this rank's device pool?" -- ahead of the pool touch. It is the
    # controller's own method, so the stub answers it the controller's way
    # rather than stubbing the answer: with no readable device buffer below,
    # ``_kv_device_row_capacity`` returns None and the check fails OPEN, which
    # is the same "absence is not a mismatch" contract the #760 seam guard
    # carries. The exploding pool is still what this suite is proving.
    _refuse_unaddressable_kv_rows = HiCacheController._refuse_unaddressable_kv_rows
    _kv_device_row_capacity = HiCacheController._kv_device_row_capacity

    def __init__(self):
        self.mem_pool_host = _Exploding()
        self.mem_pool_device_allocator = _Exploding()
        self.mem_pool_device = object()
        self.write_queue = []
        self.load_queue = []


class TestPhaseGuard(CustomTestCase):
    def setUp(self):
        guard.reset_warnings()
        self._routing = False
        # Stand in for parallel_state's module flag without building groups:
        # the guard reads it through one function, which is the seam a test
        # can honestly take over.
        self._orig = guard.flip_routing_active
        guard.flip_routing_active = lambda: self._routing
        self.addCleanup(setattr, guard, "flip_routing_active", self._orig)
        self.addCleanup(guard.reset_warnings)

    # -- 1. the pin: invisible unless the flip is routing --------------------

    def test_disarmed_is_false_without_the_flip(self):
        self._routing = False
        self.assertFalse(guard.device_tier_disarmed("write"))
        self.assertFalse(guard.device_tier_disarmed("load"))

    def test_write_runs_normally_when_the_flip_is_not_routing(self):
        """Byte-identical default: the guard returns, the real prologue runs,
        and the exploding pool proves the call reached it."""
        self._routing = False
        stub = _Stub()
        with self.assertRaises(AssertionError):
            HiCacheController.write(stub, torch.zeros(2, dtype=torch.int64))

    def test_load_runs_normally_when_the_flip_is_not_routing(self):
        self._routing = False
        stub = _Stub()
        with self.assertRaises(AssertionError):
            HiCacheController.load(stub, torch.zeros(2, dtype=torch.int64))

    # -- 2 + 3. the cut, and its shape --------------------------------------

    def test_write_is_refused_while_the_flip_routes_to_tp(self):
        self._routing = True
        stub = _Stub()
        self.assertIsNone(
            HiCacheController.write(stub, torch.zeros(2, dtype=torch.int64))
        )
        self.assertEqual(stub.write_queue, [])

    def test_load_is_refused_while_the_flip_routes_to_tp(self):
        self._routing = True
        stub = _Stub()
        self.assertIsNone(
            HiCacheController.load(stub, torch.zeros(2, dtype=torch.int64))
        )
        self.assertEqual(stub.load_queue, [])

    def test_the_refusal_happens_before_any_pool_is_touched(self):
        """The whole point: not 'the copy is wrong and we discard it', but
        'the copy never runs'. The pools would raise if reached."""
        self._routing = True
        stub = _Stub()
        for call in (HiCacheController.write, HiCacheController.load):
            self.assertIsNone(call(stub, torch.zeros(1, dtype=torch.int64)))

    # -- the log ------------------------------------------------------------

    def test_each_direction_warns_once(self):
        """The condition lasts a whole phase; warning per operation would bury
        the log at decode rates."""
        self._routing = True
        with self.assertLogs(guard.logger, level="WARNING") as first:
            guard.device_tier_disarmed("write")
        self.assertEqual(len(first.output), 1)
        self.assertIn("DISARMED", first.output[0])
        # The second call must be SILENT, and that has to be asserted rather
        # than implied: a "warn once" that warns every time still passes any
        # test which only counts the first call.
        with self.assertNoLogs(guard.logger, level="WARNING"):
            guard.device_tier_disarmed("write")
        with self.assertLogs(guard.logger, level="WARNING") as second:
            guard.device_tier_disarmed("load")
        self.assertEqual(len(second.output), 1)
        with self.assertNoLogs(guard.logger, level="WARNING"):
            guard.device_tier_disarmed("load")

    def test_the_two_directions_name_their_own_hazard(self):
        """A generic 'disarmed' line would leave the reader to guess which of
        the two failure shapes was avoided."""
        self._routing = True
        with self.assertLogs(guard.logger, level="WARNING") as w:
            guard.device_tier_disarmed("write")
        self.assertIn("content-addressed key", w.output[0])
        guard.reset_warnings()
        with self.assertLogs(guard.logger, level="WARNING") as r:
            guard.device_tier_disarmed("load")
        self.assertIn("rows the model does not read", r.output[0])

    def test_a_deployment_without_flip_groups_is_not_disarming(self):
        """The real predicate, unpatched: with no flip groups built there is no
        second pool, so nothing is guarded."""
        guard.flip_routing_active = self._orig
        self.assertFalse(guard.device_tier_disarmed("write"))

    def test_an_unreadable_phase_module_fails_OPEN(self):
        """Fail OPEN, not closed, and exercised rather than asserted about: the
        phase module is replaced by one that cannot supply the predicate, which
        is what an import failure looks like from in here. Disarming every
        deployment on an import error would be a far bigger outage than the
        hazard this guards."""
        import sys

        guard.flip_routing_active = self._orig
        name = "sglang.srt.distributed.parallel_state"
        real = sys.modules.get(name)

        class _Broken:
            def __getattr__(self, item):
                raise ImportError(f"no {item} here")

        sys.modules[name] = _Broken()
        try:
            self.assertFalse(guard.flip_routing_active())
            self.assertFalse(guard.device_tier_disarmed("write"))
        finally:
            if real is not None:
                sys.modules[name] = real
            else:  # pragma: no cover
                sys.modules.pop(name, None)


class TestReachability(CustomTestCase):
    """The premise of the guard, pinned so it cannot rot silently."""

    def test_hierarchical_cache_is_off_by_default(self):
        """No hierarchical cache, no controller, no device-tier I/O. This is
        why the hazard is LATENT on the live flagset rather than live: the
        serving argv carries --enable-phase-flip and no hicache flags."""
        from sglang.srt.server_args import ServerArgs

        self.assertFalse(ServerArgs(model_path="dummy").enable_hierarchical_cache)

    def test_the_dangerous_combination_is_accepted_which_is_why_this_guard_exists(
        self,
    ):
        """Nothing refuses flip + hierarchical cache any more -- the #630
        blocker was deliberately removed from both the boot-time and the
        runtime guard lists so a prefix cache could ride the flip. That
        decision is what makes the binding hazard reachable, and this test is
        the tripwire: if a refusal is ever added back, the guard's premise
        changed and this test says so."""
        from sglang.srt.server_args import ServerArgs

        args = ServerArgs(
            model_path="dummy",
            enable_phase_flip=True,
            phase_flip_tp_vector="30,17,17",
            pp_size=3,
            tp_size=1,
            page_size=1,
            enable_hierarchical_cache=True,
        )
        args._handle_phase_flip()  # no raise
        self.assertTrue(args.enable_phase_flip)
        self.assertTrue(args.enable_hierarchical_cache)


if __name__ == "__main__":
    unittest.main()
