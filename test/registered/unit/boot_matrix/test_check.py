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
from sglang.srt.boot_matrix.check import FAIL, PASS, STOP, _scan_fatals, check_arm
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
#: H_ps2_prefill_spill is graded_only (no-spec runs produce long output), so
#: its artifact carries no byte probe.
GRADED_PROBES = [
    {"name": "alphabet", "tier": "graded", "text": "w\nx\ny\nz", "min_score": 4},
    {"name": "squares", "tier": "graded", "text": "12 144\n13 169", "min_score": 2},
]

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
        arm = arm_by_name("reject_dcp_offload")
        with tempfile.TemporaryDirectory() as d:
            log = (
                _args_dump(draft_kv_layout="'dcp'")
                + "\nValueError: --draft-kv-layout dcp is not supported together "
                "with --enable-kv-session-offload: the spilled session's draft "
                "rows have no owner under the token-sharded rule.\n"
            )
            _write(d, arm=arm, boot_status="refused", log_text=log)
            v = check_arm(arm, d)
            self.assertEqual(v.status, PASS, v.reason)
            self.assertIsNotNone(v.refusal)

    def test_a_refusal_from_an_UNRELATED_guard_is_a_fail(self):
        """The false-pass falsifier. This is sweep 1's headline reject bug.

        ``reject_dcp_offload`` died on the KVSO_ALLOW_SPEC bring-up gate --
        which never mentions --draft-kv-layout and never tested the crossing --
        and reported PASS, because the old matcher asked whether each marker
        appeared ANYWHERE in the log and then returned any line carrying ANY
        one of them. The message below is the real one, verbatim.
        """
        arm = arm_by_name("reject_dcp_offload")
        with tempfile.TemporaryDirectory() as d:
            log = (
                _args_dump(draft_kv_layout="'dcp'")
                + "\nValueError: --enable-kv-session-offload does not yet support "
                "speculative decoding (--speculative-algorithm=NEXTN). Set "
                "KVSO_ALLOW_SPEC=1 to opt into the spill+MTP bring-up path.\n"
            )
            _write(d, arm=arm, boot_status="refused", log_text=log)
            v = check_arm(arm, d)
            self.assertEqual(v.status, FAIL, v.reason)

    def test_markers_split_across_two_unrelated_messages_is_a_fail(self):
        """Both markers present in the LOG, neither guard the arm's own."""
        arm = arm_by_name("reject_dcp_crossalgo")
        with tempfile.TemporaryDirectory() as d:
            log = (
                _args_dump(draft_kv_layout="'dcp'")
                + "\nValueError: --speculative-cross-algorithm: --speculative-"
                "cross-algorithm-force must be 'nextn', 'dflash', ...; got None.\n"
                "\n[2026-08-01 00:00:00] note: --draft-kv-layout dcp requested\n"
            )
            _write(d, arm=arm, boot_status="refused", log_text=log)
            v = check_arm(arm, d)
            self.assertEqual(v.status, FAIL, v.reason)

    def test_a_wrapped_refusal_message_still_matches(self):
        """A guard whose sentence wraps must not be split into two blocks."""
        arm = arm_by_name("reject_dcp_multilayer")
        with tempfile.TemporaryDirectory() as d:
            log = (
                _args_dump(draft_kv_layout="'dcp'")
                + "\nValueError: --draft-kv-layout dcp is not supported with\n"
                "    --enable-multi-layer-eagle: multi-layer EAGLE holds one\n"
                "    draft model runner per chain position.\n"
            )
            _write(d, arm=arm, boot_status="refused", log_text=log)
            v = check_arm(arm, d)
            self.assertEqual(v.status, PASS, v.reason)

    def test_an_ignored_import_traceback_is_not_a_fatal(self):
        """Verbatim from K_bar1_graphs/server.log, which served correctly.

        The image ships torchcodec without a matching ffmpeg, so every
        containerised boot prints this block -- sweep 1 scored a healthy
        bar1-under-graphs arm FAIL on it, and bar1 arms can ONLY run in that
        image.
        """
        arm = arm_by_name("A_default")
        with tempfile.TemporaryDirectory() as d:
            log = (
                _args_dump()
                + "\n[2026-08-01 07:23:08] Ignore import error when loading "
                "sglang.srt.multimodal.processors.mimo_audio: Could not load "
                "libtorchcodec. Likely causes:\n"
                "        The following exceptions were raised as we tried to "
                "load libtorchcodec:\n"
                "[start of libtorchcodec loading traceback]\n"
                "FFmpeg version 8:\n"
                "Traceback (most recent call last):\n"
                '  File "/usr/local/lib/python3.12/dist-packages/torch/_ops.py", '
                "line 1503, in load_library\n"
                "    ctypes.CDLL(path)\n"
                "OSError: libavutil.so.60: cannot open shared object file\n"
                "[2026-08-01 07:23:10] Load weight begin.\n"
                + READY_MARKER
                + "\n"
            )
            _write(d, arm=arm, boot_status="ready", log_text=log)
            v = check_arm(arm, d)
            # Asserted against the fatal detector specifically, not against the
            # whole verdict: this synthetic log does not carry A_default's
            # geometry lines, so the effective-config comparison has its own
            # (correct) opinion. What must not happen is the ignored block
            # being reported as a fatal.
            self.assertNotIn("fatal", v.reason, v.reason)
            self.assertIsNone(_scan_fatals(log))

    def test_a_real_fatal_after_ready_is_still_a_fatal(self):
        """The ignore-frame must not become a blanket amnesty."""
        arm = arm_by_name("A_default")
        with tempfile.TemporaryDirectory() as d:
            log = (
                _args_dump()
                + "\n[2026-08-01 07:23:08] Ignore import error when loading "
                "sglang.srt.multimodal.processors.mimo_audio: nope\n"
                "Traceback (most recent call last):\n"
                "[2026-08-01 07:23:10] Load weight begin.\n"
                + READY_MARKER
                + "\n[2026-08-01 07:30:00] NCCL error: unhandled system error\n"
            )
            _write(d, arm=arm, boot_status="ready", log_text=log)
            v = check_arm(arm, d)
            self.assertEqual(v.status, FAIL, v.reason)

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


class TestAbsenceIsAnAssertion(CustomTestCase):
    """A declared None means "this axis must be ABSENT", not "unknown".

    Sweep 2 STOPped H_ps2_prefill_spill on exactly this. The arm turns
    speculation off, so the server legitimately prints no speculative line;
    the arm declared spec_algorithm=None and eagle_topk=None; and the check
    reported "could not confirm spec_algorithm, eagle_topk". Absence WAS the
    confirmation. Half of every crossing is a feature switched off, so this
    shape recurs.
    """

    def _log_without_spec(self):
        """What the server really prints with speculation off: the fields are
        in the dump with the value None."""
        return _boot_log(
            enable_kv_session_offload="True",
            speculative_algorithm="None",
            speculative_eagle_topk="None",
        )

    def _log_with_spec(self):
        return _boot_log(enable_kv_session_offload="True")

    def test_a_log_without_the_spec_lines_confirms_an_absence_expect(self):
        arm = arm_by_name("H_ps2_prefill_spill")
        with tempfile.TemporaryDirectory() as d:
            _write(d, arm=arm, boot_status="ready",
                   log_text=self._log_without_spec(), probes=GRADED_PROBES)
            v = check_arm(arm, d)
            self.assertEqual(v.status, PASS, v.reason)

    def test_a_log_that_OMITS_the_fields_entirely_also_confirms(self):
        """The other shape of absence: no such key in the dump at all."""
        arm = arm_by_name("H_ps2_prefill_spill")
        body = (
            "[2026-08-01 00:00:00] server_args=ServerArgs(tp_size=3, dcp_size=3, "
            "enable_kv_session_offload=True, draft_kv_layout='replicated')"
        )
        log = "\n".join([body, _ENGAGED, _GRAPHS, f"[..] {READY_MARKER}"])
        with tempfile.TemporaryDirectory() as d:
            _write(d, arm=arm, boot_status="ready", log_text=log,
                   probes=GRADED_PROBES)
            v = check_arm(arm, d)
            self.assertEqual(v.status, PASS, v.reason)

    def test_a_log_WITH_the_spec_lines_fails_an_absence_expect(self):
        """The falsifier: spec present where the arm declared it off is a real
        disagreement, and must be FAIL rather than a quiet pass."""
        arm = arm_by_name("H_ps2_prefill_spill")
        with tempfile.TemporaryDirectory() as d:
            _write(d, arm=arm, boot_status="ready",
                   log_text=self._log_with_spec(), probes=GRADED_PROBES)
            v = check_arm(arm, d)
            self.assertEqual(v.status, FAIL, v.reason)
            self.assertIn("declared absent", v.reason)

    def test_a_field_declared_with_a_VALUE_can_still_be_unconfirmed(self):
        """The absence rule must not swallow the genuine can't-tell case."""
        arm = arm_by_name("A_default")
        with tempfile.TemporaryDirectory() as d:
            # No tp_size line at all: A_default declares tp_size=3, a VALUE, so
            # its absence is "could not confirm" and stays a STOP.
            _write(d, arm=arm, boot_status="ready",
                   log_text="[..] nothing useful\n" + READY_MARKER + "\n")
            v = check_arm(arm, d)
            self.assertEqual(v.status, STOP, v.reason)
            self.assertIn("could not confirm", v.reason)
