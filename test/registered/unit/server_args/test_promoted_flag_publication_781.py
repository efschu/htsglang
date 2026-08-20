"""#781: promoted flags publish themselves; an unset flag touches nothing.

The boot env used to be assembled by concatenating a captured shell
environment, a heredoc and EXTRA_ENV, and a key written twice resolved silently
as "last one wins". Now the flag is the single source of truth and the process
publishes it from its own argv, so the unit environment can go identity-only and
the boot env gate can refuse any SGLANG_* it finds there.

What has to hold:
  1. a set flag reaches the environment the consumers read,
  2. an UNSET flag leaves the environment completely alone -- this is what keeps
     every existing deployment byte-identical,
  3. booleans publish as "1"/"0" and False actually publishes "0" rather than
     being skipped as falsy (the bug that would silently re-enable a check the
     operator turned off),
  4. the TP-imbalance polarity is POSITIVE, because its retired predecessor
     SGL_DISABLE_TP_MEMORY_INBALANCE_CHECK was inverted and a careless mapping
     flips a correctness check off,
  5. the BAR1 window flag expands to the per-group suffixed keys the consumer
     actually builds.

Hermetic: pure attribute-to-environ plumbing, no model, device or server.
"""

import os
import unittest
from unittest import mock

from sglang.srt.server_args import ServerArgs


def _publish(**kw):
    """Run only the publication step against a bare ServerArgs instance."""
    sa = ServerArgs.__new__(ServerArgs)
    for field in (
        "barlink", "barlink_transport", "barlink_bar1_window_mib",
        "barlink_bar1_cap_cycles", "seam_entry_margin_mib",
        "seam_entry_delay_budget_s", "flip_seam_chunk_mib",
        "collective_census_interval", "mamba_slot_reorder", "uneven_dcp",
        "uneven_dcp_weighted", "kv_backing_relief",
        "phase_flip_image_file_backed", "corridor_rebalance",
        "enable_tp_memory_imbalance_check", "enable_health_endpoint_generation",
    ):
        setattr(sa, field, None)
    for k, v in kw.items():
        setattr(sa, k, v)
    sa._publish_promoted_781_flags()
    return sa


class TestPromotedFlagPublication781(unittest.TestCase):
    def test_set_flags_reach_the_environment(self):
        cases = [
            (dict(seam_entry_margin_mib=512), "SGLANG_SEAM_ENTRY_MARGIN_MIB", "512"),
            (dict(seam_entry_delay_budget_s=2), "SGLANG_SEAM_ENTRY_DELAY_BUDGET", "2"),
            (dict(flip_seam_chunk_mib=8), "SGLANG_FLIP_SEAM_CHUNK_MIB", "8"),
            (dict(collective_census_interval=50),
             "SGLANG_COLLECTIVE_CENSUS_INTERVAL", "50"),
            (dict(barlink_transport="bar1"), "SGLANG_BARLINK_TRANSPORT", "bar1"),
            (dict(barlink_bar1_cap_cycles=300000000000),
             "SGLANG_BARLINK_BAR1_CAP_CYCLES", "300000000000"),
            (dict(barlink=True), "SGLANG_BARLINK", "1"),
            (dict(mamba_slot_reorder=True), "SGLANG_MAMBA_SLOT_REORDER", "1"),
            (dict(uneven_dcp=True), "SGLANG_UNEVEN_DCP", "1"),
            (dict(uneven_dcp_weighted=True), "SGLANG_UNEVEN_DCP_WEIGHTED", "1"),
            (dict(kv_backing_relief=True), "SGLANG_KV_BACKING_RELIEF", "1"),
            (dict(phase_flip_image_file_backed=True),
             "SGLANG_PHASE_FLIP_IMAGE_FILE_BACKED", "1"),
        ]
        for kw, env, expected in cases:
            with self.subTest(env=env):
                with mock.patch.dict(os.environ, {}, clear=False):
                    os.environ.pop(env, None)
                    _publish(**kw)
                    self.assertEqual(os.environ.get(env), expected)

    def test_unset_flag_leaves_environment_untouched(self):
        """The compatibility guarantee: not specifying a flag changes nothing."""
        watched = [
            "SGLANG_BARLINK", "SGLANG_BARLINK_TRANSPORT",
            "SGLANG_SEAM_ENTRY_MARGIN_MIB", "SGLANG_FLIP_SEAM_CHUNK_MIB",
            "SGLANG_UNEVEN_DCP", "SGLANG_KV_BACKING_RELIEF",
            "SGLANG_COLLECTIVE_CENSUS_INTERVAL",
        ]
        with mock.patch.dict(os.environ, {}, clear=False):
            for k in watched:
                os.environ.pop(k, None)
            os.environ["SGLANG_BARLINK"] = "preexisting"
            _publish()  # every flag None
            self.assertEqual(
                os.environ.get("SGLANG_BARLINK"), "preexisting",
                "an unset flag overwrote an existing env value",
            )
            for k in watched[1:]:
                self.assertIsNone(
                    os.environ.get(k), f"unset flag invented {k}"
                )

    def test_false_publishes_zero_not_nothing(self):
        """A flag must be able to turn something OFF.

        `if self.x:` instead of `if self.x is not None:` would skip False and
        leave the feature on -- the operator asks for off and silently gets on.
        """
        for field, env in [
            ("enable_tp_memory_imbalance_check",
             "SGLANG_ENABLE_TP_MEMORY_INBALANCE_CHECK"),
            ("enable_health_endpoint_generation",
             "SGLANG_ENABLE_HEALTH_ENDPOINT_GENERATION"),
            ("corridor_rebalance", "SGLANG_CORRIDOR_REBALANCE"),
            ("kv_backing_relief", "SGLANG_KV_BACKING_RELIEF"),
            ("barlink", "SGLANG_BARLINK"),
        ]:
            with self.subTest(field=field):
                with mock.patch.dict(os.environ, {}, clear=False):
                    os.environ.pop(env, None)
                    _publish(**{field: False})
                    self.assertEqual(
                        os.environ.get(env), "0",
                        f"{field}=False did not publish '0'",
                    )

    def test_tp_imbalance_polarity_is_positive(self):
        """Its retired predecessor was inverted; getting this backwards
        silently disables a correctness check."""
        with mock.patch.dict(os.environ, {}, clear=False):
            _publish(enable_tp_memory_imbalance_check=True)
            self.assertEqual(
                os.environ["SGLANG_ENABLE_TP_MEMORY_INBALANCE_CHECK"], "1"
            )
            _publish(enable_tp_memory_imbalance_check=False)
            self.assertEqual(
                os.environ["SGLANG_ENABLE_TP_MEMORY_INBALANCE_CHECK"], "0"
            )

    def test_bar1_window_expands_to_per_group_keys(self):
        with mock.patch.dict(os.environ, {}, clear=False):
            for k in list(os.environ):
                if k.startswith("SGLANG_BARLINK_BAR1_WINDOW_MIB"):
                    os.environ.pop(k)
            _publish(
                barlink_bar1_window_mib="24,PP_0=96,FLIP_TP_0=48,FLIP_DCP_0=32"
            )
            self.assertEqual(os.environ["SGLANG_BARLINK_BAR1_WINDOW_MIB"], "24")
            self.assertEqual(
                os.environ["SGLANG_BARLINK_BAR1_WINDOW_MIB_PP_0"], "96"
            )
            self.assertEqual(
                os.environ["SGLANG_BARLINK_BAR1_WINDOW_MIB_FLIP_TP_0"], "48"
            )
            self.assertEqual(
                os.environ["SGLANG_BARLINK_BAR1_WINDOW_MIB_FLIP_DCP_0"], "32"
            )

    def test_bar1_window_bare_scalar(self):
        with mock.patch.dict(os.environ, {}, clear=False):
            os.environ.pop("SGLANG_BARLINK_BAR1_WINDOW_MIB", None)
            _publish(barlink_bar1_window_mib="64")
            self.assertEqual(os.environ["SGLANG_BARLINK_BAR1_WINDOW_MIB"], "64")


if __name__ == "__main__":
    unittest.main()
