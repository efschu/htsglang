"""#614 (a)+(b): the spill arms' load precondition and the pinned control leg.

Hermetic and file-only, like the rest of the boot-matrix checks: every case
writes a synthetic arm.json / server.log / probes.json into a tempdir and
asserts the verdict. No server, no card, no GPU.

THE DEFECT THESE PIN. A boot arm passed when it booted, resolved as declared
and stayed coherent. For an offload arm that is not enough: a load that does
not press the KV pool spills zero times, and such a run satisfies all three
conditions while proving nothing about the spill path. The #550 window met the
same shape from the other side (``docs/dev/FEATURE_CATALOG.md:1793-1800``) --
"one of the two WITH runs spilled and the other did not, so the arm measured
two different regimes and called the difference noise", and "The control never
spilled, so SPILL COST and HICACHE CONTENTION are confounded and neither is
isolated".

CAN-FAIL PROOFS (each is a single-line revert of the shipped change):
  * delete the ``if not _load_precondition_held(...)`` block before the final
    PASS in ``check.py._check_boot`` -> ``test_spilling_absent_is_void`` and
    ``test_void_replaces_pass_only`` report PASS and go red.
  * delete the ``forbidden`` block in ``check.py._check_boot`` ->
    ``test_control_that_spilled_is_fail`` goes red.
  * drop ``require_any_markers`` from arm ``O_hicache_contention`` ->
    ``test_every_spill_subject_arm_declares_its_trigger`` goes red.
  * drop ``forbid_markers`` from ``P_hicache_nospill_control`` ->
    ``test_control_arm_pins_spill_off`` goes red.
All four were run in this state and observed red before the fix was restored.
"""

import json
import os
import tempfile
import unittest

from sglang.srt.boot_matrix.arms import (
    ARMS,
    SPILL_MARKER_DECODE,
    SPILL_MARKER_PREFILL,
    Arm,
    arm_by_name,
)
from sglang.srt.boot_matrix.check import (
    FAIL,
    PASS,
    STOP,
    VOID,
    Verdict,
    check_arm,
    check_pairing,
)
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
        "enable_kv_session_offload": "True",
        "draft_kv_layout": "'replicated'",
    }
    fields.update(over)
    body = ", ".join(f"{k}={v}" for k, v in fields.items())
    return f"[2026-08-01 00:00:00] server_args=ServerArgs({body})"


_ENGAGED = (
    "[2026-08-01 00:00:01 TP0] Uneven-DCP token sizing: rank 0 (vector [30, 17, 17])."
)
# Both roles, as a real spec boot prints them. `graphs` is resolved from the
# TARGET line only (#349).
_GRAPHS = (
    "[2026-08-01 00:00:05 TP0] Capture draft decode CUDA graph begin.\n"
    "[2026-08-01 00:00:06 TP0] Capture target verify CUDA graph begin."
)
#: A real spill line, copied from the format string at
#: ``managers/kv_session_offload.py:3865`` with the placeholders filled in.
_SPILL_LINE = (
    "[2026-08-01 00:01:00 TP0] kv-session-offload SPILL(partial): rid=4076a152 "
    "arrival_seq=5 L=1557 boundary=1026 host_tail=531"
)
_PREFILL_SPILL_LINE = (
    "[2026-08-01 00:01:00 TP0] kv-session-offload PREFILL-SPILL (PS2, "
    "born-spilled deep): rid=bf447dd1 L=9000 boundary=0 host_tail=9000"
)

GOOD_PROBES = [
    {"name": "byte_count", "tier": "byte", "text": "4\n5\n6\n", "ref_text": "4\n5\n6"},
    {"name": "alphabet", "tier": "graded", "text": "w\nx\ny\nz", "min_score": 4},
    {"name": "squares", "tier": "graded", "text": "12 144\n13 169", "min_score": 2},
]
#: A byte probe whose text does NOT match its reference: the coherence gate
#: goes red on this, which is how the "VOID never masks a FAIL" case is built.
RED_PROBES = [
    {"name": "byte_count", "tier": "byte", "text": "4\n5\nZ\n", "ref_text": "4\n5\n6"},
    {"name": "alphabet", "tier": "graded", "text": "w\nx\ny\nz", "min_score": 4},
    {"name": "squares", "tier": "graded", "text": "12 144\n13 169", "min_score": 2},
]


def _boot_log(*extra, **over):
    return "\n".join(
        [_args_dump(**over), _ENGAGED, _GRAPHS, *extra, f"[..] {READY_MARKER}"]
    )


def _write(d, *, arm, boot_status, log_text, probes=None):
    with open(os.path.join(d, "arm.json"), "w") as f:
        json.dump({"name": arm.name, "kind": arm.kind, "boot_status": boot_status}, f)
    with open(os.path.join(d, "server.log"), "w") as f:
        f.write(log_text)
    if probes is not None:
        with open(os.path.join(d, "probes.json"), "w") as f:
            json.dump(probes, f)


_EXPECT = {
    "tp_size": 3,
    "dcp_size": 3,
    "dcp_engaged": True,
    "spec_algorithm": "EAGLE",
    "offload": True,
}

SPILL_ARM = Arm(
    name="t_spill",
    axis="test",
    catches="test",
    expect=dict(_EXPECT),
    coherence="byte+graded",
    require_any_markers=(SPILL_MARKER_DECODE,),
)
#: Same arm with no coherence tier, so the "coherence == none" early PASS is
#: covered too -- that branch has its own return and would otherwise keep
#: handing out green null soaks.
SPILL_ARM_NO_COH = Arm(
    name="t_spill_nocoh",
    axis="test",
    catches="test",
    expect=dict(_EXPECT),
    coherence="none",
    require_any_markers=(SPILL_MARKER_DECODE,),
)
EITHER_ARM = Arm(
    name="t_either",
    axis="test",
    catches="test",
    expect=dict(_EXPECT),
    coherence="byte+graded",
    require_any_markers=(SPILL_MARKER_PREFILL, SPILL_MARKER_DECODE),
)
CONTROL_ARM = Arm(
    name="t_control",
    axis="test",
    catches="test",
    expect=dict(_EXPECT),
    coherence="byte+graded",
    forbid_markers=(SPILL_MARKER_DECODE, SPILL_MARKER_PREFILL),
)


class TestLoadPrecondition(CustomTestCase):
    """(a) an arm whose trigger never fired is VOID, not PASS."""

    def _verdict(self, arm, *extra, probes=GOOD_PROBES):
        with tempfile.TemporaryDirectory() as d:
            _write(
                d,
                arm=arm,
                boot_status="ready",
                log_text=_boot_log(*extra),
                probes=probes,
            )
            return check_arm(arm, d)

    def test_spilling_present_is_pass(self):
        """The control of the control: with the trigger in the log, green."""
        v = self._verdict(SPILL_ARM, _SPILL_LINE)
        self.assertEqual(v.status, PASS, v.render())

    def test_spilling_absent_is_void(self):
        """The null soak. Everything else about this run is perfect."""
        v = self._verdict(SPILL_ARM)
        self.assertEqual(v.status, VOID, v.render())
        self.assertEqual(v.label, "VOID")
        # The reason has to tell the operator what to change; "VOID" alone
        # sends them back into the source to find out what a marker is.
        self.assertIn("load precondition never held", v.reason)
        self.assertIn("--max-total-tokens", v.reason)

    def test_void_is_not_stop(self):
        """STOP means the harness failed; VOID means the load did. A VOID run
        has complete artifacts, so conflating them would tell the operator to
        re-run the identical command."""
        v = self._verdict(SPILL_ARM)
        self.assertNotEqual(v.status, STOP)
        self.assertNotEqual(v.status, FAIL)

    def test_coherence_none_arm_is_also_voidable(self):
        v = self._verdict(SPILL_ARM_NO_COH, probes=None)
        self.assertEqual(v.status, VOID, v.render())

    def test_coherence_none_arm_passes_with_trigger(self):
        v = self._verdict(SPILL_ARM_NO_COH, _SPILL_LINE, probes=None)
        self.assertEqual(v.status, PASS, v.render())

    def test_void_replaces_pass_only_never_a_fail(self):
        """A red coherence gate stays red even with no spill in the log.

        This is the ordering that keeps VOID honest: if the precondition were
        checked FIRST, a genuinely divergent arm would be excused as "the load
        was too light" and the defect would leave the report.
        """
        v = self._verdict(SPILL_ARM, probes=RED_PROBES)
        self.assertEqual(v.status, FAIL, v.render())

    def test_either_mechanism_satisfies_the_precondition(self):
        for line in (_SPILL_LINE, _PREFILL_SPILL_LINE):
            with self.subTest(line=line.split("kv-session-offload ")[1][:20]):
                v = self._verdict(EITHER_ARM, line)
                self.assertEqual(v.status, PASS, v.render())

    def test_arm_without_a_declared_trigger_is_unaffected(self):
        """Most arms prove a boot property. Demanding a runtime marker from
        them would invent a gate, so an empty tuple must stay trivially true."""
        plain = Arm(
            name="t_plain",
            axis="test",
            catches="test",
            expect=dict(_EXPECT),
            coherence="byte+graded",
        )
        with tempfile.TemporaryDirectory() as d:
            _write(
                d,
                arm=plain,
                boot_status="ready",
                log_text=_boot_log(),
                probes=GOOD_PROBES,
            )
            self.assertEqual(check_arm(plain, d).status, PASS)


class TestControlPin(CustomTestCase):
    """(b) a control leg's spill-off pin is asserted, not assumed."""

    def _verdict(self, arm, *extra):
        with tempfile.TemporaryDirectory() as d:
            _write(
                d,
                arm=arm,
                boot_status="ready",
                log_text=_boot_log(*extra),
                probes=GOOD_PROBES,
            )
            return check_arm(arm, d)

    def test_clean_control_is_pass(self):
        self.assertEqual(self._verdict(CONTROL_ARM).status, PASS)

    def test_control_that_spilled_is_fail(self):
        v = self._verdict(CONTROL_ARM, _SPILL_LINE)
        self.assertEqual(v.status, FAIL, v.render())
        self.assertIn("declared absent", v.reason)
        self.assertIn("cannot serve as a control", v.reason)

    def test_control_that_born_spilled_is_also_fail(self):
        """Both mechanisms are forbidden. A control contaminated by PS2 is
        just as contaminated as one contaminated by a decode spill."""
        v = self._verdict(CONTROL_ARM, _PREFILL_SPILL_LINE)
        self.assertEqual(v.status, FAIL, v.render())


class TestPairing(CustomTestCase):
    """(b) a delta arm has no result without its baseline."""

    def _pass(self, name="t_treatment"):
        return Verdict(PASS, name, "ok")

    def test_pass_control_leaves_treatment_alone(self):
        v = check_pairing(self._pass(), self._pass("t_control"))
        self.assertEqual(v.status, PASS)

    def test_missing_control_voids_the_treatment(self):
        v = check_pairing(self._pass(), None)
        self.assertEqual(v.status, VOID, v.render())
        self.assertIn("no baseline", v.reason)

    def test_contaminated_control_voids_the_treatment(self):
        control = Verdict(FAIL, "t_control", "the pin did not hold")
        v = check_pairing(self._pass(), control)
        self.assertEqual(v.status, VOID, v.render())
        self.assertIn("t_control", v.reason)

    def test_pairing_never_rescues_a_red_treatment(self):
        red = Verdict(FAIL, "t_treatment", "coherence gate red")
        self.assertEqual(check_pairing(red, self._pass("t_control")).status, FAIL)
        self.assertEqual(check_pairing(red, None).status, FAIL)


class TestPairingIsWired(CustomTestCase):
    """A mechanism nobody calls is not a mechanism (AUDIT_421 class).

    ``check_pairing`` is folded over the sweep's verdict list in a SECOND pass,
    because the control leg can run after the treatment in arm order.
    """

    def test_sweep_folds_declared_pairings(self):
        from sglang.srt.boot_matrix.sweep import _apply_pairings

        treatment = Verdict(PASS, "O_hicache_contention", "ok")
        control = Verdict(FAIL, "P_hicache_nospill_control", "the pin did not hold")
        out = {v.arm: v for v in _apply_pairings([treatment, control])}
        self.assertEqual(out["O_hicache_contention"].status, VOID)
        self.assertEqual(out["P_hicache_nospill_control"].status, FAIL)

    def test_sweep_voids_a_treatment_run_without_its_control(self):
        """`--only O_hicache_contention` must not be able to print a
        contention number."""
        from sglang.srt.boot_matrix.sweep import _apply_pairings

        out = _apply_pairings([Verdict(PASS, "O_hicache_contention", "ok")])
        self.assertEqual(out[0].status, VOID, out[0].render())

    def test_the_sweep_main_loop_actually_calls_the_fold(self):
        """AUDIT_421 class: a fold that exists and is never invoked leaves the
        matrix exactly as confounded as before, with a green function to point
        at."""
        import inspect

        from sglang.srt.boot_matrix import sweep

        self.assertIn("_apply_pairings(verdicts)", inspect.getsource(sweep._main))

    def test_sweep_leaves_unpaired_arms_alone(self):
        from sglang.srt.boot_matrix.sweep import _apply_pairings

        out = _apply_pairings([Verdict(PASS, "A_default", "ok")])
        self.assertEqual(out[0].status, PASS)

    def test_sweep_tolerates_a_non_arm_verdict(self):
        """The A-vs-A band baseline is appended to the verdict list but is not
        a matrix arm; the fold must not raise on it."""
        from sglang.srt.boot_matrix.sweep import _apply_pairings

        out = _apply_pairings([Verdict(PASS, "not_an_arm", "ok")])
        self.assertEqual(out[0].status, PASS)


class TestMatrixDeclarations(CustomTestCase):
    """The declarations in the shipped matrix, not just the machinery."""

    #: Every arm whose SUBJECT is a spill -- read off each arm's own
    #: ``catches`` line, not guessed from its flags.
    SPILL_SUBJECT_ARMS = (
        "B_offload",
        "D_offload_x_crossalgo",
        "G_all_axes",
        "H_ps2_prefill_spill",
        "I_dflash_shards",
        "J_waveback_ps2",
        "N_resume_under_spec",
        "O_hicache_contention",
    )

    def test_every_spill_subject_arm_declares_its_trigger(self):
        for name in self.SPILL_SUBJECT_ARMS:
            with self.subTest(arm=name):
                arm = arm_by_name(name)
                self.assertTrue(
                    arm.require_any_markers,
                    f"{name} can boot green with zero spills",
                )

    def test_prefill_arms_accept_either_mechanism(self):
        """PS2 arms can be exercised by a born-spilled prefill; requiring the
        decode line alone would VOID a run that did exercise the subject."""
        for name in ("H_ps2_prefill_spill", "J_waveback_ps2"):
            with self.subTest(arm=name):
                self.assertIn(
                    SPILL_MARKER_PREFILL, arm_by_name(name).require_any_markers
                )

    def test_control_arm_pins_spill_off(self):
        control = arm_by_name("P_hicache_nospill_control")
        self.assertEqual(
            control.forbid_markers, (SPILL_MARKER_DECODE, SPILL_MARKER_PREFILL)
        )
        # The pin itself: #236's total-volume regler declines every real spill
        # at admission. Without the flag the "control" is only uncontaminated
        # by luck of the load, which is the #550 confound verbatim.
        self.assertIn("--kv-session-offload-budget-total-tokens", control.flags)
        idx = control.flags.index("--kv-session-offload-budget-total-tokens")
        self.assertEqual(control.flags[idx + 1], "1")

    def test_the_pin_flag_is_a_real_server_flag(self):
        """A misspelled flag is refused by argparse, but a flag that exists and
        means something ELSE is not -- and a pin that does not pin is worse
        than no pin, because forbid_markers would then be the only thing
        standing between a contaminated control and a published number. Pinned
        against the dataclass field the server actually reads."""
        import dataclasses

        from sglang.srt.server_args import ServerArgs

        names = {f.name for f in dataclasses.fields(ServerArgs)}
        self.assertIn("kv_session_offload_budget_total_tokens", names)
        # And its default is OFF, so the flag is a deliberate arm-level pin
        # rather than something every recipe already carries.
        default = next(
            f
            for f in dataclasses.fields(ServerArgs)
            if f.name == "kv_session_offload_budget_total_tokens"
        ).default
        self.assertEqual(default, 0)

    def test_control_keeps_the_treatments_memory_posture(self):
        """The pin must not be "drop kvso": the pinned host pool is what
        HiCache contends with, so removing it changes the thing under study."""
        control = arm_by_name("P_hicache_nospill_control")
        treatment = arm_by_name("O_hicache_contention")
        for flag in (
            "--enable-kv-session-offload",
            "--enable-hierarchical-cache",
            "--hicache-mem-layout",
        ):
            with self.subTest(flag=flag):
                self.assertIn(flag, control.flags)
                self.assertIn(flag, treatment.flags)

    def test_contention_arm_names_its_control_in_data(self):
        """Machine-readable, not prose. Arm O carried its pairing only in
        ``capture_note``, so nothing could check the control had been run."""
        self.assertEqual(
            arm_by_name("O_hicache_contention").control_arm,
            "P_hicache_nospill_control",
        )

    def test_declared_controls_exist(self):
        names = {a.name for a in ARMS}
        for arm in ARMS:
            if arm.control_arm is not None:
                with self.subTest(arm=arm.name):
                    self.assertIn(arm.control_arm, names)

    def test_reject_arms_carry_no_load_declarations(self):
        with self.assertRaises(ValueError):
            Arm(
                name="t_bad",
                axis="test",
                catches="test",
                kind="reject",
                reject_markers=("x",),
                require_any_markers=("y",),
            )

    def test_a_marker_cannot_be_required_and_forbidden(self):
        with self.assertRaises(ValueError):
            Arm(
                name="t_bad2",
                axis="test",
                catches="test",
                require_any_markers=(SPILL_MARKER_DECODE,),
                forbid_markers=(SPILL_MARKER_DECODE,),
            )


class TestMarkersMatchTheSource(CustomTestCase):
    """A marker that no longer matches the emitter is an always-VOID arm.

    Pinned against the format strings themselves rather than against a copy,
    so renaming the log line is a test failure here and not a silent matrix
    that voids every offload arm forever.
    """

    def test_markers_are_prefixes_of_the_real_format_strings(self):
        import inspect

        from sglang.srt.managers import kv_session_offload

        src = inspect.getsource(kv_session_offload)
        # The format strings are wrapped across source lines, so compare
        # against the source with its string-concatenation whitespace removed.
        flat = src.replace('"\n            "', "").replace('"\n                "', "")
        for marker in (SPILL_MARKER_DECODE, SPILL_MARKER_PREFILL):
            with self.subTest(marker=marker):
                # %s / %d stand where the check sees a value; compare the
                # literal head up to the first placeholder.
                head = marker.split("rid=")[0]
                self.assertIn(head, flat)


if __name__ == "__main__":
    unittest.main()
