"""report_effective reads RESOLVED config from the log, not the flags (#349).

The #340 lesson as a test: the dcp-engaged fact must come from the scheduler's
own token-sizing line, so a log that had the env set but never installed a plan
reads as NOT engaged. Every parser here runs on synthetic log text; no server.
"""

import unittest

from sglang.srt.boot_matrix.effective import (
    READY_MARKER,
    first_refusal,
    report_effective,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=2, suite="base-a-test-cpu")


def _args_dump(**over):
    """A synthetic ServerArgs dump line with the fields report_effective reads.
    Overrides let a test flip exactly one."""
    fields = {
        "tp_size": "3",
        "dcp_size": "3",
        "speculative_algorithm": "'EAGLE'",
        "speculative_eagle_topk": "1",
        "speculative_cross_algorithm": "False",
        "enable_kv_session_offload": "False",
        "dual_group_lane": "False",
        "draft_kv_layout": "'replicated'",
        "rank_tp_ratio": "[29607, 17780, 17780]",
    }
    fields.update(over)
    body = ", ".join(f"{k}={v}" for k, v in fields.items())
    return f"[2026-08-01 00:00:00] server_args=ServerArgs({body})"


GOOD_BOOT = "\n".join(
    [
        _args_dump(),
        "[2026-08-01 00:00:01 TP0] Uneven-DCP token sizing: rank 0 local "
        "capacity 320340 tokens / ratio 30 = unit 10678 (vector [30, 17, 17]).",
        "[2026-08-01 00:00:02 TP0] sglang is using nccl==2.28.9",
        "[2026-08-01 00:00:05 TP0] Capture draft decode CUDA graph begin.",
        # A real spec boot prints BOTH roles; the target line is the one
        # `graphs` is resolved from (#349).
        "[2026-08-01 00:00:06 TP0] Capture target verify CUDA graph begin.",
        f"[2026-08-01 00:00:09] {READY_MARKER}",
    ]
)


class TestReportEffective(CustomTestCase):
    def test_a_clean_boot_resolves_every_axis(self):
        eff = report_effective(GOOD_BOOT)
        self.assertTrue(eff.ready)
        self.assertEqual(eff.tp_size, 3)
        self.assertEqual(eff.dcp_size, 3)
        self.assertTrue(eff.dcp_engaged)
        self.assertEqual(eff.spec_algorithm, "EAGLE")
        self.assertEqual(eff.eagle_topk, 1)
        self.assertFalse(eff.cross_algorithm)
        self.assertEqual(eff.draft_kv_layout, "replicated")
        self.assertTrue(eff.graphs)
        self.assertEqual(eff.barlink, "nccl")

    def test_dcp_engaged_is_read_from_the_scheduler_not_the_env(self):
        """THE #340 falsifier: dcp_size>1 in the dump but NO token-sizing line
        means the weighted path did not engage -- report False, not True, so an
        arm that thought it was token-sharding and silently was not goes red."""
        log = "\n".join(
            [
                _args_dump(dcp_size="2"),
                "[2026-08-01 00:00:02 TP0] sglang is using nccl==2.28.9",
                f"[2026-08-01 00:00:09] {READY_MARKER}",
            ]
        )
        eff = report_effective(log)
        self.assertEqual(eff.dcp_size, 2)
        self.assertFalse(
            eff.dcp_engaged,
            "no scheduler token-sizing line: the path did not engage, "
            "whatever the environment said",
        )

    def test_the_auto_set_line_also_counts_as_engaged(self):
        log = "\n".join(
            [
                _args_dump(),
                "[2026-08-01 00:00:01 TP0] Uneven DCP: auto-set dcp_size to 3.",
                f"[2026-08-01 00:00:09] {READY_MARKER}",
            ]
        )
        self.assertTrue(report_effective(log).dcp_engaged)

    def test_resolved_alias_is_read_not_the_flag(self):
        """The dump already carries NEXTN rewritten to EAGLE; reading it gives
        the resolved value for free -- the whole reason we do not parse flags."""
        self.assertEqual(report_effective(GOOD_BOOT).spec_algorithm, "EAGLE")

    def test_spec_none_is_none_not_the_string(self):
        eff = report_effective(_args_dump(speculative_algorithm="None"))
        self.assertIsNone(eff.spec_algorithm)

    def test_cross_algorithm_and_offload_flags(self):
        eff = report_effective(
            _args_dump(
                speculative_cross_algorithm="True", enable_kv_session_offload="True"
            )
        )
        self.assertTrue(eff.cross_algorithm)
        self.assertTrue(eff.offload)

    # The three lines below are copied VERBATIM from a real boot log
    # (gpu-battery-results/2026-08-01_349_boot_matrix/E_barlink/server.log).
    # The previous fixture invented "barlink transport=device selected on TP0",
    # which no build has ever printed -- so the reader passed its test and
    # still failed arm E on a real run. A parser test written against a made-up
    # line tests the fixture, not the parser.
    _REAL_ACHIEVED = (
        "barlink enabled for group 'world:0': requested=device, ACHIEVED=device\n"
        "barlink enabled for group 'tp:0': requested=device, ACHIEVED=device\n"
        "barlink enabled for group 'dcp:0': requested=device, ACHIEVED=device\n"
    )
    #: Also real, and the reason the old regex answered "up": both of these
    #: appear in the same log, so "the word after transport" was ambiguous
    #: before it was wrong.
    _REAL_NOISE = "barlink device transport up\nbarlink shm transport up\n"

    def test_barlink_transport_read_from_the_achieved_lines(self):
        log = _args_dump() + "\n" + self._REAL_NOISE + self._REAL_ACHIEVED
        self.assertEqual(report_effective(log).barlink, "device")

    def test_the_state_word_up_is_not_mistaken_for_a_transport(self):
        """The exact sweep-1 failure: declared 'device', resolved 'up'."""
        log = _args_dump() + "\n" + self._REAL_NOISE + self._REAL_ACHIEVED
        self.assertNotEqual(report_effective(log).barlink, "up")

    def test_a_mixed_run_is_reported_as_mixed_not_flattened(self):
        """One group falling back must not be hidden behind the majority."""
        log = _args_dump() + (
            "\nbarlink enabled for group 'tp:0': requested=bar1, ACHIEVED=bar1\n"
            "barlink group 'dcp:0': requested=bar1, ACHIEVED=gloo (holder missing)\n"
        )
        got = report_effective(log).barlink
        self.assertTrue(got.startswith("mixed:"), got)
        self.assertIn("bar1", got)
        self.assertIn("gloo", got)

    def test_bar1_achieved_reads_as_bar1(self):
        """Real K_bar1_graphs shape: all three groups on the direct path."""
        log = _args_dump() + (
            "\nbarlink enabled for group 'world:0': requested=bar1, ACHIEVED=bar1\n"
            "barlink enabled for group 'tp:0': requested=bar1, ACHIEVED=bar1\n"
            "barlink enabled for group 'dcp:0': requested=bar1, ACHIEVED=bar1\n"
        )
        self.assertEqual(report_effective(log).barlink, "bar1")

    def test_eager_boot_reports_graphs_false(self):
        log = _args_dump() + "\n[..] Disabled cuda graph\n" + READY_MARKER
        self.assertFalse(report_effective(log).graphs)

    def test_not_ready_when_marker_absent(self):
        self.assertFalse(report_effective(_args_dump()).ready)

    def test_a_truncated_log_still_parses_what_it_has(self):
        """A hung boot's log is truncated; the parser must not throw and must
        report what the dump line said."""
        eff = report_effective(_args_dump(dcp_size="3"))
        self.assertEqual(eff.dcp_size, 3)
        self.assertFalse(eff.ready)


class TestFirstRefusal(CustomTestCase):
    REFUSAL = (
        "[2026-08-01 00:00:01] ValueError: --draft-kv-layout dcp is not usable "
        "yet: the draft-EXTEND forward has no uneven-DCP metadata split."
    )

    def test_finds_the_line_with_all_markers(self):
        got = first_refusal(self.REFUSAL, ["--draft-kv-layout dcp", "draft-EXTEND"])
        self.assertIsNotNone(got)
        self.assertIn("draft-EXTEND", got)

    def test_none_when_a_marker_is_missing(self):
        self.assertIsNone(
            first_refusal(self.REFUSAL, ["--draft-kv-layout dcp", "not-in-here"])
        )


if __name__ == "__main__":
    unittest.main()
