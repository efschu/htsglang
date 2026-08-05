"""#603 (A): a BAR1 window is never silently made smaller than asked for.

Two invariants, and they are deliberately different for the two ways a size
can be asked for:

* An EXPLICIT window (either env var actually present) that does not fit
  refuses at group build. A number somebody typed is an instruction, and
  quietly serving a smaller one rewrites the input to every later size
  decision of that group.
* The DEFAULT window may be reduced -- that is an adaptation, not an
  override -- but the reduction is RECORDED and shows up in the state
  summary. Before this, a reduced group and a fully served group printed
  the identical "ACHIEVED=bar1" line.

The third invariant is about the WORDING, and it is pinned because the old
text was wrong in a way that misdirected an incident: it claimed reduced
groups "fall back to the gloo layer without further notice". Both
all_reduce and all_to_all decompose oversized payloads into rounds, so a
reduction costs launches, not coverage -- and the eventual fallback past
the round caps is warned about (outside a capture) or raised (inside one).
"""

import os
import unittest
from unittest import mock

from sglang.srt.distributed.device_communicators import (
    barlink_matrix_transport as mt,
)

MiB = 1024 * 1024


class WindowClipTest(unittest.TestCase):
    def setUp(self):
        mt.reset_clips_for_test()
        self.addCleanup(mt.reset_clips_for_test)

    def _with_free(self, free_mib):
        """bar1_free stubbed: (free, gross, source)."""
        return mock.patch.object(
            mt, "bar1_free",
            return_value=(free_mib * MiB, 256 * MiB, "nvml"),
        )

    def _no_ledger(self):
        return mock.patch.object(mt, "ledger_balance", return_value=[])

    # -- the default may be reduced, but never invisibly -------------------

    def test_default_request_is_reduced_and_recorded(self):
        env = {k: v for k, v in os.environ.items()
               if not k.startswith("SGLANG_BARLINK_BAR1_WINDOW_MIB")
               and k != "SGLANG_BARLINK_BAR1_RESERVE_MIB"}
        with mock.patch.dict(os.environ, env, clear=True), \
                self._with_free(56), self._no_ledger():
            got = mt.window_for("dcp:0", device=None)
        # 56 free - 32 reserve = 24 MiB, exactly the production case.
        self.assertEqual(got, 24 * MiB)
        clips = mt.window_clips()
        self.assertIn("dcp:0", clips)
        self.assertEqual(clips["dcp:0"]["granted_bytes"], 24 * MiB)
        self.assertEqual(clips["dcp:0"]["requested_bytes"],
                         mt.WINDOW_MIB_DEFAULT * MiB)

    def test_window_that_fits_is_not_recorded_as_reduced(self):
        """The can-fail proof's counterpart: the recorder must stay silent
        when nothing was reduced, otherwise the test above would pass on a
        table that simply records everything."""
        env = {k: v for k, v in os.environ.items()
               if not k.startswith("SGLANG_BARLINK_BAR1_WINDOW_MIB")
               and k != "SGLANG_BARLINK_BAR1_RESERVE_MIB"}
        with mock.patch.dict(os.environ, env, clear=True), \
                self._with_free(400), self._no_ledger():
            got = mt.window_for("tp:0", device=None)
        self.assertEqual(got, mt.WINDOW_MIB_DEFAULT * MiB)
        self.assertEqual(mt.window_clips(), {})

    # -- an explicit request is an instruction -----------------------------

    def test_explicit_per_group_window_refuses_instead_of_shrinking(self):
        with mock.patch.dict(
            os.environ,
            {"SGLANG_BARLINK_BAR1_WINDOW_MIB_DCP_0": "96"},
        ), self._with_free(56), self._no_ledger():
            with self.assertRaises(mt.Bar1WindowRefused) as caught:
                mt.window_for("dcp:0", device=None)
        message = str(caught.exception)
        # The refusal has to carry the arithmetic, or it cannot be acted on.
        self.assertIn("SGLANG_BARLINK_BAR1_WINDOW_MIB_DCP_0", message)
        self.assertIn("96 MiB", message)
        self.assertIn("24 MiB", message)
        # And nothing is recorded as "reduced": it did not come up at all.
        self.assertEqual(mt.window_clips(), {})

    def test_explicit_global_window_also_refuses(self):
        with mock.patch.dict(
            os.environ, {"SGLANG_BARLINK_BAR1_WINDOW_MIB": "96"},
        ), self._with_free(56), self._no_ledger():
            with self.assertRaises(mt.Bar1WindowRefused):
                mt.window_for("dcp:0", device=None)

    def test_explicit_window_that_fits_is_served_unchanged(self):
        """The gate must be able to NOT fire -- otherwise the two refusal
        tests above would pass against a function that always raises."""
        with mock.patch.dict(
            os.environ, {"SGLANG_BARLINK_BAR1_WINDOW_MIB_TP_0": "64"},
        ), self._with_free(400), self._no_ledger():
            got = mt.window_for("tp:0", device=None)
        self.assertEqual(got, 64 * MiB)

    # -- the reduction reaches the place people read -----------------------

    def test_state_summary_names_the_reduced_window(self):
        from sglang.srt.distributed.device_communicators import barlink

        barlink._STATE.clear()
        self.addCleanup(barlink._STATE.clear)
        barlink.report_state("dcp:0", "bar1", "bar1")
        # Without a recorded reduction the summary must not mention one.
        self.assertNotIn("REDUCED", barlink.state_summary())
        mt.record_clip("dcp:0", 96 * MiB, 24 * MiB,
                       "SGLANG_BARLINK_BAR1_WINDOW_MIB", "arithmetic here")
        summary = barlink.state_summary()
        self.assertIn("REDUCED", summary)
        self.assertIn("24 MiB granted of 96 MiB requested", summary)

    # -- the refusal must not be swallowed ---------------------------------

    def test_the_refusal_reaches_the_caller_on_the_direct_paths(self):
        """A refusal that ``_build_transport`` caught and turned into a gloo
        fallback would be the very defect this change removes -- louder in
        the source, identical in behaviour.

        ``bar1`` and ``matrix`` are in the no-fallback set, so the factory is
        invoked outside the try/except and the exception propagates. Pinned
        because the two live far apart: the raise is in
        barlink_matrix_transport, the swallow would be in barlink.
        """
        from sglang.srt.distributed.device_communicators import barlink

        self.assertTrue(barlink._no_fallback("bar1"))
        self.assertTrue(barlink._no_fallback("matrix"))

        boom = mt.Bar1WindowRefused("window does not fit")
        with mock.patch.dict(
            barlink.TRANSPORT_REGISTRY,
            {"bar1": mock.Mock(side_effect=boom)},
        ):
            with self.assertRaises(mt.Bar1WindowRefused):
                barlink._build_transport(
                    "bar1", cpu_group=None, device=None, disabled=False,
                    group="tp:0",
                )

    # -- the wording that misdirected the incident -------------------------

    def test_reduction_warning_does_not_claim_a_silent_gloo_fallback(self):
        env = {k: v for k, v in os.environ.items()
               if not k.startswith("SGLANG_BARLINK_BAR1_WINDOW_MIB")
               and k != "SGLANG_BARLINK_BAR1_RESERVE_MIB"}
        with mock.patch.dict(os.environ, env, clear=True), \
                self._with_free(56), self._no_ledger(), \
                self.assertLogs(mt.logger, level="WARNING") as logs:
            mt.window_for("dcp:0", device=None)
        text = "\n".join(logs.output)
        self.assertNotIn("without further notice", text)
        self.assertIn("rounds", text)


if __name__ == "__main__":
    unittest.main()
