# Copyright 2026 SGLang Team
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# ==============================================================================
"""The captured ship environment is the source of truth, and it is checkable.

WHY THIS FILE EXISTS, measured 2026-08-12: the documented restore path booted
an instance that came up, answered ``/model_info`` with 200 and never answered
``/generate`` -- the #622 wedge signature. The cause was not a bug in any of
the code the boot runs. It was that ``scripts/route_a_631_prod_boot.sh`` kept
its OWN hand-maintained copy of the ship environment and had drifted from the
capture in seven keys at once, including
``SGLANG_UNEVEN_TOKEN_VECTOR=28,26,20`` where the ship process carried
``14,10,8``.

Patching those keys is not a fix; the defect is the private copy. So the
falsifiers here are about the MECHANISM:

* round-trip -- a capture rendered to shell and sourced back reproduces the
  capture exactly, including values that contain spaces, quotes, ``$`` and
  JSON. Anything less and a boot script has a reason to keep its own copy.
* ``--check`` fires, and NAMES the key, for a changed value, a missing key and
  an extra key. A gate that cannot fail is not a gate.
* ``--check`` passes when only the sanctioned per-boot keys differ, because a
  gate that cries wolf on ``SGLANG_BOOT_COMMIT`` gets bypassed within a week.
* each of the seven REAL divergences of 2026-08-12 is caught, driven from the
  capture that is actually shipped.

Hermetic: no GPU, no server, no systemd. The only subprocess is ``bash``,
because "sourceable shell" is a claim about bash and asserting it in Python
would test the assertion.
"""

import importlib.util
import json
import os
import pathlib
import subprocess
import sys
import tempfile
import unittest

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=10, suite="base-a-test-cpu")

_ROOT = pathlib.Path(__file__).resolve().parents[4]
_TOOL = _ROOT / "scripts" / "turnkey_539_export_env.py"
#: The capture the boot path replays, versioned in the repo so the boot does
#: not depend on an evidence directory an agent may clean up.
_CAPTURE = _ROOT / "deploy" / "turnkey" / "ship_env.capture"
#: The immutable evidence the repo copy was taken from. Read-only, and absent
#: on a machine that is not this rig -- the drift test skips there.
_EVIDENCE = pathlib.Path(
    "/spinning/evidence-631/val-r4/ship_env_captured.txt")


def _load_tool():
    spec = importlib.util.spec_from_file_location(
        "turnkey_539_export_env", str(_TOOL))
    mod = importlib.util.module_from_spec(spec)
    # Registered before exec: the module defines a dataclass, and dataclasses
    # resolve their own module out of sys.modules while the class body runs.
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


EX = _load_tool()


def _run(*args, env=None):
    """Run the tool as a subprocess; return (rc, stdout, stderr)."""
    p = subprocess.run([sys.executable, str(_TOOL), *args],
                       capture_output=True, text=True, env=env)
    return p.returncode, p.stdout, p.stderr


def _write(tmp, name, text):
    path = os.path.join(tmp, name)
    with open(path, "w") as fh:
        fh.write(text)
    return path


#: A capture exercising every quoting hazard the ship capture can carry: a
#: value with spaces, one with double quotes and JSON, one with a literal
#: ``$``, one that is empty, and one whose VALUE contains ``=``.
HAZARD_CAPTURE = "\n".join([
    "# a comment line, and the next line is blank",
    "",
    "SGLANG_PLAIN=1",
    "SGLANG_WITH_SPACES=14, 10, 8 and a trailing word",
    'SGLANG_JSON={"preserve_thinking": true}',
    "SGLANG_DOLLAR=$HOME and `date` and \\backslash",
    "SGLANG_EMPTY=",
    "SGLANG_HAS_EQUALS=a=b=c",
    "SGLANG_SINGLE_QUOTE=it's here",
    "LD_LIBRARY_PATH=:/usr/local/cuda-12.2/lib64",
    "PYTHONPATH=/spinning/wt-merge-r4/python",
    "SGLANG_BOOT_COMMIT=75c86bf255",
]) + "\n"


class TestCaptureParsing(CustomTestCase):
    def test_blank_and_comment_lines_are_skipped(self):
        got = EX.parse_capture("# c\n\nA=1\n")
        self.assertEqual(got, {"A": "1"})

    def test_value_may_contain_equals_signs(self):
        self.assertEqual(EX.parse_capture("A=a=b=c\n")["A"], "a=b=c")

    def test_empty_value_is_a_value_not_an_absence(self):
        got = EX.parse_capture("A=\n")
        self.assertIn("A", got)
        self.assertEqual(got["A"], "")

    def test_a_line_without_an_equals_sign_is_refused_by_line_number(self):
        with self.assertRaises(EX.CaptureError) as cm:
            EX.parse_capture("A=1\nJUST_A_WORD\n", source="cap.txt")
        self.assertIn("cap.txt:2", str(cm.exception))

    def test_a_duplicate_key_is_refused_rather_than_silently_last_wins(self):
        with self.assertRaises(EX.CaptureError) as cm:
            EX.parse_capture("A=1\nA=2\n", source="cap.txt")
        self.assertIn("A", str(cm.exception))
        self.assertIn("cap.txt:2", str(cm.exception))


class TestRoundTrip(CustomTestCase):
    """capture -> export -> source -> environment == capture."""

    def _round_trip(self, capture_text, extra_args=()):
        with tempfile.TemporaryDirectory() as tmp:
            cap = _write(tmp, "cap.txt", capture_text)
            rc, out, err = _run(cap, *extra_args)
            self.assertEqual(rc, 0, f"tool failed: {err}")
            sh = _write(tmp, "exports.sh", out)
            dump = os.path.join(tmp, "env.json")
            # env -i: nothing survives from this process, so what the sourced
            # file sets is the whole answer.
            script = (
                f'set -eu; . "{sh}"; '
                f'"{sys.executable}" -c '
                f"'import json,os,sys; "
                f'json.dump(dict(os.environ), open(sys.argv[1], "w"))\' '
                f'"{dump}"')
            p = subprocess.run(["env", "-i", "/bin/bash", "-c", script],
                               capture_output=True, text=True)
            self.assertEqual(p.returncode, 0, p.stderr)
            with open(dump) as fh:
                return json.load(fh)

    def test_every_hazard_value_survives_the_round_trip(self):
        got = self._round_trip(HAZARD_CAPTURE)
        want = EX.parse_capture(HAZARD_CAPTURE)
        for k, v in want.items():
            if k in EX.PER_BOOT_KEYS:
                continue
            self.assertIn(k, got, f"{k} was not exported")
            self.assertEqual(got[k], v, f"{k} did not survive")

    def test_a_value_with_spaces_survives_as_one_value(self):
        got = self._round_trip(HAZARD_CAPTURE)
        self.assertEqual(got["SGLANG_WITH_SPACES"],
                         "14, 10, 8 and a trailing word")

    def test_json_and_quotes_survive_unmangled(self):
        got = self._round_trip(HAZARD_CAPTURE)
        self.assertEqual(got["SGLANG_JSON"], '{"preserve_thinking": true}')
        self.assertEqual(got["SGLANG_SINGLE_QUOTE"], "it's here")

    def test_a_dollar_sign_is_not_expanded_by_the_shell(self):
        """The latent bug in val-r4/restore_ship.sh: ``export "$line"`` works
        only because no captured value happens to contain a space or a ``$``.
        Quoted output must not leave the shell a chance to interpret."""
        got = self._round_trip(HAZARD_CAPTURE)
        self.assertEqual(got["SGLANG_DOLLAR"],
                         "$HOME and `date` and \\backslash")

    def test_the_per_boot_keys_are_not_emitted(self):
        got = self._round_trip(HAZARD_CAPTURE)
        # SGLANG_BOOT_COMMIT and PYTHONPATH are per-boot identities; emitting
        # the capture's dead values is how a boot claims to be a commit it is
        # not.
        self.assertNotIn("SGLANG_BOOT_COMMIT", got)
        self.assertNotIn("PYTHONPATH", got)

    def test_governed_only_drops_keys_the_stack_does_not_own(self):
        with tempfile.TemporaryDirectory() as tmp:
            cap = _write(tmp, "cap.txt", "SGLANG_A=1\nHOME=/root\nTERM=linux\n")
            rc, out, _ = _run(cap, "--governed-only")
            self.assertEqual(rc, 0)
            self.assertIn("SGLANG_A", out)
            self.assertNotIn("HOME", out)
            self.assertNotIn("TERM", out)


def _env_from(capture_text, **changes):
    """A live environment built from a capture, with named changes applied.
    A value of None removes the key."""
    env = dict(EX.parse_capture(capture_text))
    for k, v in changes.items():
        if v is None:
            env.pop(k, None)
        else:
            env[k] = v
    return env


class TestCheckGate(CustomTestCase):
    def _check(self, capture_text, env, allow=()):
        with tempfile.TemporaryDirectory() as tmp:
            cap = _write(tmp, "cap.txt", capture_text)
            args = [cap, "--check"]
            for a in allow:
                args += ["--allow", a]
            # PATH is kept so the tool can start at all; it is not governed.
            full = {"PATH": os.environ.get("PATH", "")}
            full.update(env)
            return _run(*args, env=full)

    def test_a_changed_value_fails_and_names_the_key(self):
        """The real one: the ship carried 14,10,8 and the boot script set
        28,26,20."""
        rc, out, err = self._check(
            HAZARD_CAPTURE,
            _env_from(HAZARD_CAPTURE, SGLANG_PLAIN="99"))
        self.assertNotEqual(rc, 0)
        text = out + err
        self.assertIn("SGLANG_PLAIN", text)
        self.assertIn("CHANGED", text)
        self.assertIn("99", text)
        self.assertIn("1", text)

    def test_a_missing_key_fails_and_names_the_key(self):
        rc, out, err = self._check(
            HAZARD_CAPTURE,
            _env_from(HAZARD_CAPTURE, SGLANG_PLAIN=None))
        self.assertNotEqual(rc, 0)
        text = out + err
        self.assertIn("SGLANG_PLAIN", text)
        self.assertIn("MISSING", text)

    def test_an_extra_key_fails_and_names_the_key(self):
        rc, out, err = self._check(
            HAZARD_CAPTURE,
            _env_from(HAZARD_CAPTURE,
                      PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"))
        self.assertNotEqual(rc, 0)
        text = out + err
        self.assertIn("PYTORCH_CUDA_ALLOC_CONF", text)
        self.assertIn("EXTRA", text)

    def test_an_unchanged_environment_passes(self):
        rc, out, err = self._check(HAZARD_CAPTURE, _env_from(HAZARD_CAPTURE))
        self.assertEqual(rc, 0, out + err)

    def test_only_the_sanctioned_per_boot_keys_differing_passes(self):
        rc, out, err = self._check(
            HAZARD_CAPTURE,
            _env_from(HAZARD_CAPTURE,
                      PYTHONPATH="/spinning/wt-631-routea/python",
                      SGLANG_BOOT_COMMIT="deadbeef01",
                      SGLANG_PHASE_FLIP_INSTANCE="1786600000-4242",
                      CUDA_VISIBLE_DEVICES="GPU-aaaa,GPU-bbbb"))
        self.assertEqual(rc, 0, out + err)

    def test_an_ungoverned_key_is_not_policed(self):
        """The stack owns SGLANG_*/HTSGLANG_*/PYTORCH_* plus three path keys.
        Policing HOME or TERM would make the gate unusable from a shell."""
        rc, out, err = self._check(
            HAZARD_CAPTURE, _env_from(HAZARD_CAPTURE, TERM="xterm"))
        self.assertEqual(rc, 0, out + err)

    def test_allow_is_per_key_and_does_not_widen_to_other_keys(self):
        rc, out, err = self._check(
            HAZARD_CAPTURE,
            _env_from(HAZARD_CAPTURE, SGLANG_PLAIN="99", SGLANG_EMPTY="x"),
            allow=["SGLANG_PLAIN"])
        self.assertNotEqual(rc, 0, "one --allow must not sanction the rest")
        text = out + err
        self.assertIn("SGLANG_EMPTY", text)

    def test_an_allowed_key_is_reported_loudly_rather_than_hidden(self):
        rc, out, err = self._check(
            HAZARD_CAPTURE,
            _env_from(HAZARD_CAPTURE, SGLANG_PLAIN="99"),
            allow=["SGLANG_PLAIN"])
        self.assertEqual(rc, 0, out + err)
        text = out + err
        self.assertIn("SGLANG_PLAIN", text)
        self.assertIn("OVERRIDE", text)

    def test_check_reads_an_env_file_when_asked(self):
        with tempfile.TemporaryDirectory() as tmp:
            cap = _write(tmp, "cap.txt", "SGLANG_A=1\n")
            envf = _write(tmp, "env.txt", "SGLANG_A=2\n")
            rc, out, err = _run(cap, "--check", "--env-file", envf)
            self.assertNotEqual(rc, 0)
            self.assertIn("SGLANG_A", out + err)


class TestTheRealDrift(CustomTestCase):
    """The seven divergences measured on 2026-08-12, driven from the capture
    that ships in the repo."""

    #: prod_boot.sh's hand-maintained env as it stood at 0d3d973aed, expressed
    #: as its delta from the capture. Every entry is a measured fact.
    DRIFT = {
        "SGLANG_UNEVEN_TOKEN_VECTOR": "28,26,20",       # capture: 14,10,8
        "SGLANG_CORRIDOR_FLOOR_MIB": None,              # capture: 1536
        "SGLANG_CORRIDOR_REBALANCE": None,              # capture: 0
        "SGLANG_KV_BACKING_RELIEF": None,               # capture: 1
        "SGLANG_SEAM_ENTRY_DELAY_BUDGET": None,         # capture: 2
        "SGLANG_SEAM_ENTRY_MARGIN_MIB": None,           # capture: 512
        "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",  # not captured
    }

    def setUp(self):
        self.assertTrue(_CAPTURE.exists(),
                        f"the shipped capture is missing: {_CAPTURE}")
        self.capture = EX.load_capture(str(_CAPTURE))

    def test_the_capture_carries_the_ship_token_vector(self):
        self.assertEqual(self.capture["SGLANG_UNEVEN_TOKEN_VECTOR"], "14,10,8")

    def test_every_one_of_the_seven_is_caught_and_named(self):
        env = dict(self.capture)
        for k, v in self.DRIFT.items():
            if v is None:
                env.pop(k, None)
            else:
                env[k] = v
        divs = EX.diff_env(self.capture, env)
        named = {d.key for d in divs}
        for k in self.DRIFT:
            self.assertIn(k, named, f"{k} slipped through the gate")
        kinds = {d.key: d.kind for d in divs}
        self.assertEqual(kinds["SGLANG_UNEVEN_TOKEN_VECTOR"], "CHANGED")
        self.assertEqual(kinds["SGLANG_CORRIDOR_FLOOR_MIB"], "MISSING")
        self.assertEqual(kinds["PYTORCH_CUDA_ALLOC_CONF"], "EXTRA")

    def test_each_divergence_is_caught_on_its_own(self):
        """Jointly is not the same as individually: a gate that only fires on
        the whole set would pass the next single-key drift."""
        for k, v in self.DRIFT.items():
            env = dict(self.capture)
            if v is None:
                env.pop(k, None)
            else:
                env[k] = v
            divs = EX.diff_env(self.capture, env)
            self.assertEqual([d.key for d in divs], [k],
                             f"{k} alone was not caught cleanly")

    def test_the_shipped_capture_matches_the_evidence_it_came_from(self):
        if not _EVIDENCE.exists():
            self.skipTest(f"evidence not on this machine: {_EVIDENCE}")
        want = EX.load_capture(str(_EVIDENCE))
        self.assertEqual(self.capture, want,
                         "the repo copy of the ship capture has drifted from "
                         "the evidence file it was taken from")


class TestOneDefinitionOfTheSanctionedKeys(CustomTestCase):
    def test_the_parity_proof_reads_the_same_list(self):
        """Two lists of per-boot keys is the same defect one directory over."""
        spec = importlib.util.spec_from_file_location(
            "turnkey_539_parity_proof",
            str(_ROOT / "scripts" / "turnkey_539_parity_proof.py"))
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        self.assertIs(mod.EXPECTED_DIVERGENCE, EX.PER_BOOT_KEYS)

    def test_every_sanctioned_key_carries_a_stated_reason(self):
        for k, why in EX.PER_BOOT_KEYS.items():
            self.assertTrue(why.strip(), f"{k} is sanctioned without a reason")


if __name__ == "__main__":
    unittest.main()
