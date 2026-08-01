"""check_arm turns collected artifacts into one verdict (#349).

Hermetic and file-only: every case writes a synthetic arm.json / server.log /
probes.json into a tempdir and asserts the verdict. No server, no card. The
coherence path uses the real #274 grader against real alphabet/squares text, so
the whole chain -- log -> effective -> coherence -> verdict -- is proven end to
end on a card-less host.
"""

import json
import os
import tempfile
import unittest

from sglang.srt.boot_matrix.arms import Arm, arm_by_name
from sglang.srt.boot_matrix.check import FAIL, PASS, STOP, check_arm
from sglang.srt.boot_matrix.effective import READY_MARKER
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=3, suite="base-a-test-cpu")


def _args_dump(**over):
    fields = {
        "tp_size": "3",
        "dcp_size": "3",
        "speculative_algorithm": "'EAGLE'",
        "speculative_eagle_topk": "1",
        "speculative_cross_algorithm": "False",
        "enable_kv_session_offload": "False",
        "draft_kv_layout": "'replicated'",
    }
    fields.update(over)
    body = ", ".join(f"{k}={v}" for k, v in fields.items())
    return f"[2026-08-01 00:00:00] server_args=ServerArgs({body})"


_ENGAGED = (
    "[2026-08-01 00:00:01 TP0] Uneven-DCP token sizing: rank 0 (vector [30, 17, 17])."
)
_GRAPHS = "[2026-08-01 00:00:05 TP0] Capture draft decode CUDA graph begin."


def _boot_log(**over):
    return "\n".join(
        [_args_dump(**over), _ENGAGED, _GRAPHS, f"[..] {READY_MARKER}"]
    )


def _write(d, *, arm, boot_status, log_text, probes=None):
    with open(os.path.join(d, "arm.json"), "w") as f:
        json.dump({"name": arm.name, "kind": arm.kind, "boot_status": boot_status}, f)
    with open(os.path.join(d, "server.log"), "w") as f:
        f.write(log_text)
    if probes is not None:
        with open(os.path.join(d, "probes.json"), "w") as f:
            json.dump(probes, f)


# A boot arm declaring the base config, byte+graded coherence.
BOOT_ARM = Arm(
    name="t_boot",
    axis="test",
    catches="test",
    expect={"tp_size": 3, "dcp_size": 3, "dcp_engaged": True, "spec_algorithm": "EAGLE"},
    coherence="byte+graded",
)
GOOD_PROBES = [
    {"name": "byte_count", "tier": "byte", "text": "4\n5\n6\n", "ref_text": "4\n5\n6"},
    {"name": "alphabet", "tier": "graded", "text": "w\nx\ny\nz", "min_score": 4},
    {"name": "squares", "tier": "graded", "text": "12 144\n13 169", "min_score": 2},
]


class TestBootArm(CustomTestCase):
    def test_ready_matching_coherent_is_pass(self):
        with tempfile.TemporaryDirectory() as d:
            _write(d, arm=BOOT_ARM, boot_status="ready", log_text=_boot_log(),
                   probes=GOOD_PROBES)
            v = check_arm(BOOT_ARM, d)
            self.assertEqual(v.status, PASS, v.reason)

    def test_declared_config_mismatch_is_fail(self):
        """THE #340 catch: the arm declared dcp_engaged=True but the log shows
        the path never installed a plan (no token-sizing line)."""
        arm = Arm(name="t", axis="a", catches="c",
                  expect={"dcp_engaged": True}, coherence="none")
        with tempfile.TemporaryDirectory() as d:
            silent = "\n".join(
                [_args_dump(dcp_size="2"), f"[..] {READY_MARKER}"]  # no _ENGAGED line
            )
            _write(d, arm=arm, boot_status="ready", log_text=silent)
            v = check_arm(arm, d)
            self.assertEqual(v.status, FAIL)
            self.assertIn("dcp_engaged", v.reason)

    def test_hang_with_no_fatal_is_fail(self):
        """The #132 x weightless class: never reached ready, no fatal marker,
        timed out -- a silent hang is a FAIL, not a STOP."""
        arm = Arm(name="t", axis="a", catches="c", expect={"tp_size": 3},
                  coherence="none")
        with tempfile.TemporaryDirectory() as d:
            _write(d, arm=arm, boot_status="timeout",
                   log_text=_args_dump())  # no ready, no fatal
            v = check_arm(arm, d)
            self.assertEqual(v.status, FAIL)
            self.assertIn("hang", v.reason)

    def test_crash_before_ready_is_fail(self):
        arm = Arm(name="t", axis="a", catches="c", expect={"tp_size": 3},
                  coherence="none")
        with tempfile.TemporaryDirectory() as d:
            log = _args_dump() + "\nTraceback (most recent call last):\nRuntimeError\n"
            _write(d, arm=arm, boot_status="crashed", log_text=log)
            v = check_arm(arm, d)
            self.assertEqual(v.status, FAIL)
            self.assertIn("died", v.reason)

    def test_fatal_after_ready_is_fail(self):
        arm = Arm(name="t", axis="a", catches="c", expect={"tp_size": 3},
                  coherence="none")
        with tempfile.TemporaryDirectory() as d:
            log = _boot_log() + "\n[..] NCCL error: unhandled system error\n"
            _write(d, arm=arm, boot_status="ready", log_text=log)
            v = check_arm(arm, d)
            self.assertEqual(v.status, FAIL)

    def test_unconfirmed_field_is_stop(self):
        """Declared a field the log does not carry: could-not-confirm is STOP,
        not a wrong-config FAIL."""
        arm = Arm(name="t", axis="a", catches="c",
                  expect={"barlink": "device"}, coherence="none")
        with tempfile.TemporaryDirectory() as d:
            _write(d, arm=arm, boot_status="ready", log_text=_boot_log())
            v = check_arm(arm, d)
            self.assertEqual(v.status, STOP)
            self.assertIn("barlink", v.reason)

    def test_coherence_below_floor_is_fail(self):
        with tempfile.TemporaryDirectory() as d:
            bad = [{"name": "alphabet", "tier": "graded", "text": "w\nx", "min_score": 4}]
            _write(d, arm=BOOT_ARM, boot_status="ready", log_text=_boot_log(),
                   probes=bad)
            v = check_arm(BOOT_ARM, d)
            self.assertEqual(v.status, FAIL)
            self.assertIn("coherence", v.reason)

    def test_missing_probes_is_stop(self):
        with tempfile.TemporaryDirectory() as d:
            _write(d, arm=BOOT_ARM, boot_status="ready", log_text=_boot_log())
            v = check_arm(BOOT_ARM, d)
            self.assertEqual(v.status, STOP)


class TestRejectArm(CustomTestCase):
    def test_clean_refusal_is_pass(self):
        arm = arm_by_name("reject_dcp_draftextend")
        with tempfile.TemporaryDirectory() as d:
            log = (
                _args_dump(draft_kv_layout="'dcp'")
                + "\nValueError: --draft-kv-layout dcp is not usable yet: the "
                "draft-EXTEND forward has no uneven-DCP metadata split.\n"
            )
            _write(d, arm=arm, boot_status="refused", log_text=log)
            v = check_arm(arm, d)
            self.assertEqual(v.status, PASS, v.reason)
            self.assertIsNotNone(v.refusal)

    def test_a_config_that_must_refuse_but_booted_is_fail(self):
        arm = arm_by_name("reject_dcp_topk")
        with tempfile.TemporaryDirectory() as d:
            _write(d, arm=arm, boot_status="ready", log_text=_boot_log())
            v = check_arm(arm, d)
            self.assertEqual(v.status, FAIL)
            self.assertIn("booted", v.reason)

    def test_refusal_with_the_wrong_error_is_fail(self):
        arm = arm_by_name("reject_dcp_topk")
        with tempfile.TemporaryDirectory() as d:
            log = _args_dump() + "\nTraceback (most recent call last):\nKeyError\n"
            _write(d, arm=arm, boot_status="crashed", log_text=log)
            v = check_arm(arm, d)
            self.assertEqual(v.status, FAIL)
            self.assertIn("unexpected error", v.reason)

    def test_nothing_learned_is_stop(self):
        arm = arm_by_name("reject_dcp_topk")
        with tempfile.TemporaryDirectory() as d:
            _write(d, arm=arm, boot_status="timeout", log_text=_args_dump())
            v = check_arm(arm, d)
            self.assertEqual(v.status, STOP)


class TestArtifactContract(CustomTestCase):
    def test_missing_arm_json_is_stop(self):
        with tempfile.TemporaryDirectory() as d:
            with open(os.path.join(d, "server.log"), "w") as f:
                f.write(_boot_log())
            v = check_arm(BOOT_ARM, d)
            self.assertEqual(v.status, STOP)

    def test_missing_log_is_stop(self):
        with tempfile.TemporaryDirectory() as d:
            with open(os.path.join(d, "arm.json"), "w") as f:
                json.dump({"name": BOOT_ARM.name, "kind": "boot",
                           "boot_status": "ready"}, f)
            v = check_arm(BOOT_ARM, d)
            self.assertEqual(v.status, STOP)

    def test_verdict_renders_one_block(self):
        with tempfile.TemporaryDirectory() as d:
            _write(d, arm=BOOT_ARM, boot_status="ready", log_text=_boot_log(),
                   probes=GOOD_PROBES)
            v = check_arm(BOOT_ARM, d)
            self.assertIn("BOOTMATRIX-PASS", v.render())


if __name__ == "__main__":
    unittest.main()
