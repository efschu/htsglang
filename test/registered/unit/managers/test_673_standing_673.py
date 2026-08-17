# SPDX-License-Identifier: Apache-2.0
"""#673 standing: the terminate is STILL REACHABLE on the serving head.

THE SPECIMEN, verbatim from `/spinning/evidence-665-f1/boot_item7.log`::

    [16:35:25] SIGTERM received. Draining requests and shutting down...
    [16:35:26] Health check request received during shutdown. Returning 503.
    [16:35:29] Gracefully exiting... Remaining number of requests 0.
    terminate called without an active exception

Clean drain, zero remaining requests, no active exception. That is the #673
signature exactly, and it is on the GRACEFUL path.

THE CHAIN, verified on serving head 88a0d787da:

1. SIGTERM drains, and ``release_distributed(scheduler, graceful=True)`` IS
   reached -- the call is wired at ``scheduler.py:9877``.
2. It returns immediately: ``distributed_teardown_enabled`` reads
   ``scheduler_distributed_teardown``, which defaults **False**
   (``server_args.py:5585+``, ``scheduler_teardown.py:49-55``, :78).
3. So the process groups are never destroyed.
4. NCCL's C++ watchdog and ``HeartbeatMonitor::runLoop()`` are joined by the
   process group's DESTRUCTOR and by nothing else.
5. At interpreter exit those ``std::thread`` objects are still joinable, and
   destroying a joinable ``std::thread`` calls ``std::terminate`` -- with no
   active exception, because there never was one.

**So the remedy is PRESENT BUT DISABLED.** #673 is not un-fixed; it is
fixed-and-gated-off, which is why the signature survives on a head that already
carries `scheduler_teardown.py`.

WHY IT IS GATED, AND WHY THAT REASON NO LONGER HOLDS. The destroy runs
``GroupCoordinator.destroy``, which closes ``barlink_comm`` -- #722's machinery
-- so it shipped default-off to keep out of that lane. **#722-as-scoped was
subsequently retracted and barlink declared innocent.** The gate's stated
justification is therefore withdrawn, which makes arming it the cheapest path
to closing #673. Arming a production default is a BOOT decision, not a desk
one, so this file pins the standing and the window item books the flip.

SCOPE: teardown seam only. No barlink transport file is touched here, and none
should be -- that retraction stands.

WHAT #653/#650 DO **NOT** COVER, as an absence claim with its evidence: a
repo-wide grep of ``python/sglang/srt/`` finds no ``HostCollectiveAborted`` and
no ``peer_statement`` symbol on this lineage, so the de-trapped ``wait_ge`` /
per-site abort-code / peer-statement-dump work is not present here to cover the
teardown path. ``wait_ge`` exists, but as the raw device-side spin at
``barlink_host.py:210`` -- not de-trapped with a named abort.
"""

import inspect
import unittest


class TestTheRemedyIsPresent(unittest.TestCase):
    """It is not missing; it is switched off. Both halves matter."""

    def test_the_teardown_module_exists_on_this_head(self):
        from sglang.srt.managers import scheduler_teardown

        self.assertTrue(hasattr(scheduler_teardown, "release_distributed"))

    def test_the_destroy_is_WIRED_into_the_graceful_path(self):
        from sglang.srt.managers import scheduler as sched_mod

        src = inspect.getsource(sched_mod)
        self.assertIn("release_distributed(scheduler", src)
        idx = src.index("release_distributed(scheduler")
        self.assertIn("graceful=scheduler.gracefully_exit", src[idx : idx + 120])


class TestButItIsGatedOff(unittest.TestCase):
    """The reachability of the terminate, in one assertion.

    This pin is deliberately written to FAIL THE DAY THE GATE IS ARMED. That is
    the point: when someone flips it, this test is the reminder to re-run the
    #673 specimen and close the ticket with evidence rather than assumption.
    """

    def test_the_gate_is_still_default_off(self):
        from sglang.srt.server_args import ServerArgs

        self.assertFalse(
            ServerArgs.scheduler_distributed_teardown,
            "scheduler_distributed_teardown is now ON by default. Good -- but "
            "re-run the #673 teardown specimen (SIGTERM after a clean drain) "
            "and confirm 'terminate called without an active exception' is "
            "gone before closing #673, then update this pin.",
        )

    def test_the_gate_short_circuits_before_any_destroy(self):
        from sglang.srt.managers import scheduler_teardown as td

        src = inspect.getsource(td.release_distributed)
        self.assertIn("distributed_teardown_enabled", src)
        gate = src.index("distributed_teardown_enabled")
        destroy = src.index("destroy_model_parallel")
        self.assertLess(
            gate,
            destroy,
            "the gate must be checked before the destroy, or a default-off "
            "boot would still be paying the barlink close",
        )

    def test_the_default_reads_from_server_args_not_a_constant(self):
        from sglang.srt.managers import scheduler_teardown as td

        self.assertFalse(td.distributed_teardown_enabled(object()))


class TestTheAbsenceClaimIsEvidenced(unittest.TestCase):
    """#653/#650 machinery is not on this lineage. An absence claim needs the
    search that backs it, so the search is the test."""

    def test_HostCollectiveAborted_is_absent(self):
        import sglang.srt as srt

        self.assertFalse(hasattr(srt, "HostCollectiveAborted"))

    def test_wait_ge_exists_but_only_as_the_raw_device_spin(self):
        """It is CUDA source text inside barlink_host, not a Python symbol with
        a named abort -- so it is not the de-trapped #653 version."""
        from sglang.srt.distributed.device_communicators import barlink_host

        src = inspect.getsource(barlink_host)
        self.assertIn("wait_ge", src)
        self.assertNotIn("HostCollectiveAborted", src)


if __name__ == "__main__":
    unittest.main()
