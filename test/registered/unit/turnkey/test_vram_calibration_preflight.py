"""Preflight names an uncalibrated rig instead of letting it fall back quietly.

The VRAM ledger is the sizing authority and is on by default. When a term
cannot be priced it does the safe thing -- refuses to guess, says which term in
the log, and lets the inherited heuristic size the boot. That keeps a fresh rig
bootable, which is right, and it is also easy to never notice: the rig then
runs on the ``512 + tokens*1.5`` catch-all the ledger exists to replace.

So the fact gets stated where an operator reads before a boot. Opt-in, because
whether an unpriced rig may boot is the config's call and not this check's:
``preflight.require_vram_calibration`` defaults False so a fresh machine is
never bricked by a default it did not choose.

The probe is injected, per this module's rule that every failure mode must be
reachable without hardware -- and here the interesting state, "never probed",
is the one every new rig starts in.
"""

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

import dataclasses
import types
import unittest

from sglang.srt.turnkey.config import PreflightSpec
from sglang.srt.turnkey.preflight import (
    CalibrationObs,
    check_vram_calibration,
    default_probes,
)
from sglang.test.test_utils import CustomTestCase


def _cfg(require: bool):
    """Only ``cfg.preflight`` is read by this check, so only that is supplied.

    A full StackConfig would need repo/venv/log_dir/cards/serving, none of
    which this check looks at -- fabricating them would make the test look
    like it covers more than it does.
    """
    return types.SimpleNamespace(
        preflight=PreflightSpec(require_vram_calibration=require)
    )


def _probes(*, cached: bool, fingerprint: str = "abc123"):
    return dataclasses.replace(
        default_probes(),
        vram_calibration=lambda: CalibrationObs(fingerprint=fingerprint, cached=cached),
    )


class TestTheCheckIsOptIn(CustomTestCase):
    def test_it_is_silent_when_not_required(self):
        self.assertIsNone(check_vram_calibration(_cfg(False), _probes(cached=False)))

    def test_the_config_default_is_permissive(self):
        self.assertFalse(PreflightSpec().require_vram_calibration)


class TestTheCheckWhenRequired(CustomTestCase):
    def test_a_calibrated_rig_passes(self):
        self.assertIsNone(check_vram_calibration(_cfg(True), _probes(cached=True)))

    def test_an_uncalibrated_rig_is_refused_by_name(self):
        r = check_vram_calibration(_cfg(True), _probes(cached=False))
        self.assertIsNotNone(r)
        self.assertEqual(r.name, "vram_calibration_missing")

    def test_the_refusal_names_the_fingerprint_it_looked_for(self):
        """Without the key, 'no calibration' is unactionable: the operator
        cannot tell a missing probe from a fingerprint that moved."""
        r = check_vram_calibration(_cfg(True), _probes(cached=False))
        self.assertIn("abc123", r.observed)

    def test_an_unresolved_fingerprint_still_reads(self):
        r = check_vram_calibration(_cfg(True), _probes(cached=False, fingerprint=""))
        self.assertIn("<unresolved>", r.observed)

    def test_the_remedy_is_the_command_that_fixes_it(self):
        r = check_vram_calibration(_cfg(True), _probes(cached=False))
        self.assertIn("sglang.srt.mem_ledger.probe", r.remedy)

    def test_the_remedy_also_names_the_way_out(self):
        """An operator who cannot probe right now needs the escape hatch in
        the same sentence, or they will find a worse one."""
        r = check_vram_calibration(_cfg(True), _probes(cached=False))
        self.assertIn("require_vram_calibration", r.remedy)


class TestItChecksAndNeverRepairs(CustomTestCase):
    """This module's stated rule. Running the probe here would both break it
    and touch a card before the boot is allowed to."""

    def test_the_real_probe_only_loads_never_measures(self):
        import inspect

        from sglang.srt.turnkey import preflight

        src = inspect.getsource(preflight._real_vram_calibration)
        self.assertIn("load_calibration", src)
        self.assertNotIn("measure_calibration", src)


if __name__ == "__main__":
    unittest.main()
