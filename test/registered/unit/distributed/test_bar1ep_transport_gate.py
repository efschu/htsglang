"""bar1ep's availability gate must name attributes the transport really has.

WHY THIS FILE EXISTS
--------------------
``token_dispatcher/bar1ep.py`` asks a transport whether it can carry
``all_to_all`` by probing attribute names with ``hasattr``. Task #295 renamed
the transport's German method names to English (``traegt_a2a`` ->
``supports_a2a``, ``a2a_schlitz_bytes`` -> ``a2a_slot_bytes``) and task #358
renamed the class (``BarlinkBar1Transport``). The probe kept the old spellings,
so the gate answered "this is not a BAR1 transport" for the one class that is
one -- and the BAR1 EP dispatch was unreachable, on every rank, without a
single log line saying so.

A ``hasattr`` probe is invisible to every rename tool and to every import
check: the name only exists as a string. That is the whole hazard, so the
tests below pin the string against the real class rather than against a copy
of the string, and they pin that a decline is announced instead of returned in
silence.

CPU only: nothing here builds a transport, a communicator, or touches a device.
"""

import logging
import unittest

from sglang.srt.distributed.device_communicators.barlink_bar1 import (
    BarlinkBar1Transport,
)
from sglang.srt.distributed.device_communicators.barlink_env_guard import (
    RETIRED_ENV_VARS,
    RetiredEnvVarError,
    check_retired_env_vars,
)
from sglang.srt.layers.moe.token_dispatcher import bar1ep as bar1ep_mod
from sglang.srt.layers.moe.token_dispatcher.bar1ep import (
    TRANSPORT_A2A_ATTRS,
    bar1ep_transport,
    bar1ep_verfuegbar,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=2, suite="base-a-test-cpu")

SLOT = 4 * 1024 * 1024


class _RealNamedTransport:
    """A stand-in carrying exactly the BAR1 transport's public a2a surface.

    Deliberately not a mock with auto-created attributes: ``hasattr`` is true
    for everything on those, which is precisely the failure this file guards
    against.
    """

    def __init__(self, slot_bytes=SLOT):
        self._slot = int(slot_bytes)

    def a2a_slot_bytes(self) -> int:
        return self._slot

    def supports_a2a(self, largest_block: int) -> bool:
        return 0 <= largest_block <= self._slot

    def barlink_all_to_all_single(self, comm, output, inp, send_bytes,
                                  recv_bytes, send_offsets=None,
                                  recv_offsets=None, kernel_bytes=None,
                                  rounds=None):
        raise AssertionError("not reached in a gate test")


class _NotATransport:
    """Something plugged into the seam that cannot carry all_to_all."""


class _Comm:
    def __init__(self, transport, disabled=False):
        self.transport = transport
        self.disabled = disabled
        self.cpu_group = None
        self.world_size = 2
        self.rank = 0


class _Group:
    def __init__(self, comm):
        self.barlink_comm = comm


class TestGateNamesMatchTheRealTransport(CustomTestCase):
    """The falsifier: every probed name must exist on the shipped class."""

    def test_every_probed_attribute_exists_on_BarlinkBar1Transport(self):
        missing = [
            n for n in TRANSPORT_A2A_ATTRS if not hasattr(BarlinkBar1Transport, n)
        ]
        self.assertEqual(
            missing,
            [],
            f"bar1ep probes for {missing}, which {BarlinkBar1Transport.__name__} "
            f"does not have. A hasattr probe against a stale name makes the "
            f"gate answer False forever.",
        )

    def test_probe_covers_the_three_methods_the_dispatcher_calls(self):
        """Pinned as a set, so a name can be renamed but not quietly dropped."""
        self.assertEqual(
            set(TRANSPORT_A2A_ATTRS),
            {"barlink_all_to_all_single", "supports_a2a", "a2a_slot_bytes"},
        )

    def test_the_old_german_spellings_are_gone_from_the_probe(self):
        for stale in ("traegt_a2a", "a2a_schlitz_bytes"):
            self.assertNotIn(stale, TRANSPORT_A2A_ATTRS)


class TestGateAcceptsARealBar1Transport(CustomTestCase):
    """The behaviour the stale names broke: a real transport is accepted."""

    def test_transport_with_real_names_is_accepted(self):
        t = _RealNamedTransport()
        got, reason = bar1ep_transport(_Group(_Comm(t)))
        self.assertIs(got, t, f"gate declined a real BAR1 transport: {reason}")
        self.assertEqual(reason, "")

    def test_verfuegbar_says_yes_for_a_real_transport(self):
        ok, reason = bar1ep_verfuegbar(_Group(_Comm(_RealNamedTransport())))
        self.assertTrue(ok, reason)

    def test_a_zero_slot_transport_is_declined(self):
        """A transport whose byte proof did not pass carries nothing."""
        got, reason = bar1ep_transport(_Group(_Comm(_RealNamedTransport(0))))
        self.assertIsNone(got)
        self.assertIn("a2a", reason)


class TestDeclineIsAnnounced(CustomTestCase):
    """A closed gate must say which condition closed it."""

    def _decline(self, group):
        with self.assertLogs(bar1ep_mod.logger, level=logging.INFO) as caught:
            got, reason = bar1ep_transport(group)
        self.assertIsNone(got)
        return reason, "\n".join(caught.output)

    def test_missing_attributes_are_logged_by_name(self):
        reason, logged = self._decline(_Group(_Comm(_NotATransport())))
        self.assertIn("supports_a2a", reason)
        self.assertIn("supports_a2a", logged)
        self.assertIn("_NotATransport", logged)

    def test_absent_communicator_is_logged(self):
        _, logged = self._decline(_Group(None))
        self.assertIn("SGLANG_BARLINK", logged)

    def test_absent_transport_is_logged(self):
        _, logged = self._decline(_Group(_Comm(None)))
        self.assertIn("SGLANG_BARLINK_TRANSPORT", logged)

    def test_disabled_communicator_is_logged(self):
        _, logged = self._decline(_Group(_Comm(_RealNamedTransport(), disabled=True)))
        self.assertIn("world_size", logged)

    def test_the_same_reason_is_announced_once_per_process(self):
        """Loud, not chatty: the gate is asked once per MoE layer."""
        bar1ep_mod._DECLINE_ANNOUNCED.clear()
        group = _Group(_Comm(_NotATransport()))
        with self.assertLogs(bar1ep_mod.logger, level=logging.INFO) as caught:
            for _ in range(5):
                bar1ep_transport(group)
        self.assertEqual(
            len(caught.output), 1, f"expected one line, got:\n{caught.output}"
        )

    def test_an_accepted_transport_logs_no_decline(self):
        bar1ep_mod._DECLINE_ANNOUNCED.clear()
        logger = bar1ep_mod.logger
        records = []
        handler = logging.Handler()
        handler.emit = records.append
        logger.addHandler(handler)
        try:
            bar1ep_transport(_Group(_Comm(_RealNamedTransport())))
        finally:
            logger.removeHandler(handler)
        self.assertEqual([r.getMessage() for r in records], [])


class TestTheSelftestSwitchWasRenamedLoudly(CustomTestCase):
    """``SGLANG_BAR1EP_SELBSTTEST`` -> ``SGLANG_BAR1EP_SELFTEST``.

    The switch turns off bar1ep's byte proof. Ignoring a stale spelling would
    silently turn the proof back ON for someone who meant to skip it, or --
    worse, once the polarity of such a switch ever changes -- off for someone
    who did not. It never carried the ``SGLANG_HTCCL`` prefix, so the guard's
    prefix scan alone did not see it.
    """

    def test_the_old_spelling_is_retired_to_the_new_one(self):
        self.assertEqual(
            RETIRED_ENV_VARS.get("SGLANG_BAR1EP_SELBSTTEST"),
            "SGLANG_BAR1EP_SELFTEST",
        )

    def test_a_stale_launch_script_fails_at_startup(self):
        with self.assertRaises(RetiredEnvVarError) as caught:
            check_retired_env_vars({"SGLANG_BAR1EP_SELBSTTEST": "0"})
        message = str(caught.exception)
        self.assertIn("SGLANG_BAR1EP_SELBSTTEST", message)
        self.assertIn("SGLANG_BAR1EP_SELFTEST", message)

    def test_the_current_spelling_passes(self):
        check_retired_env_vars({"SGLANG_BAR1EP_SELFTEST": "0"})

    def test_the_source_reads_only_the_current_spelling(self):
        import inspect

        source = inspect.getsource(bar1ep_mod)
        self.assertIn("SGLANG_BAR1EP_SELFTEST", source)
        self.assertNotIn("SGLANG_BAR1EP_SELBSTTEST", source)

    def test_no_live_variable_is_listed_as_retired(self):
        """The exact-match branch turns every table key into a hard failure.

        A current name that slipped into the table as a KEY would therefore
        kill every boot that sets it -- cheap to pin, expensive to discover.
        """
        live = sorted(k for k in RETIRED_ENV_VARS if k in set(RETIRED_ENV_VARS.values()))
        self.assertEqual(live, [])


if __name__ == "__main__":
    unittest.main()
