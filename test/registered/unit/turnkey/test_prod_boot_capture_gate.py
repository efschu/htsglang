# Copyright 2026 SGLang Team
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# ==============================================================================
"""route_a_631_prod_boot.sh must be structurally unable to boot a drifted env.

The defect class, measured 2026-08-12: the boot script kept its own
hand-maintained copy of the ship environment and drifted from the capture in
seven keys, one of which (``SGLANG_UNEVEN_TOKEN_VECTOR``) put the KV token
split at odds with the layout. The instance came up, answered ``/model_info``
and never answered ``/generate``.

Patching seven keys would leave the class intact, so these tests are about the
class:

* the script must not RE-DECLARE any key the capture owns -- a private copy is
  the defect, and a grep is the only way to keep it from growing back;
* the script must run the ``--check`` gate immediately before exec and REFUSE
  on divergence, proven by a run in which the gate actually fires;
* the gate must not be bypassable in bulk -- an override is per key, named,
  and printed;
* and the assembled environment must actually carry the ship values, proven by
  running the script in ``DRY_RUN`` and reading what it built.

``DRY_RUN=1`` assembles env and argv, runs the gate, prints both and exits
without launching. Nothing here starts a server, touches a card or writes to
the production log: PATH holds stub ``nvidia-smi``/``git``, and every path the
script writes is redirected into a temp dir.
"""

import os
import pathlib
import re
import stat
import subprocess
import tempfile
import unittest

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=10, suite="base-a-test-cpu")

_ROOT = pathlib.Path(__file__).resolve().parents[4]
_BOOT = _ROOT / "scripts" / "route_a_631_prod_boot.sh"
_CAPTURE = _ROOT / "deploy" / "turnkey" / "ship_env.capture"

_UUID_BIG = "GPU-31d7ef41-f574-4d0e-21ad-e773fd938f6d"
_UUID_A = "GPU-5c648f96-be1d-42d5-0221-34d11ab137f7"
_UUID_B = "GPU-62dbbae1-e859-9ccc-f9c2-d9f2443a84f4"

_NVIDIA_SMI_STUB = f"""#!/bin/sh
cat <<'EOF'
0, NVIDIA GeForce RTX 5090, {_UUID_BIG}, 32607 MiB
1, NVIDIA GeForce RTX 3080, {_UUID_A}, 20480 MiB
2, NVIDIA GeForce RTX 3080, {_UUID_B}, 20480 MiB
EOF
"""

_GIT_STUB = """#!/bin/sh
case "$*" in
  *rev-parse*) echo 75c86bf255 ;;
  *branch*)    echo ops-test ;;
  *status*)    : ;;
  *)           : ;;
esac
"""


def _stub_dir(tmp):
    d = os.path.join(tmp, "bin")
    os.makedirs(d, exist_ok=True)
    for name, body in (("nvidia-smi", _NVIDIA_SMI_STUB), ("git", _GIT_STUB)):
        p = os.path.join(d, name)
        with open(p, "w") as fh:
            fh.write(body)
        os.chmod(p, os.stat(p).st_mode | stat.S_IEXEC | stat.S_IXGRP
                 | stat.S_IXOTH)
    return d


def _dry_run(extra_env=None, tmp=None, args=()):
    """Run the boot script in DRY_RUN and return (rc, stdout, stderr)."""
    own = tempfile.TemporaryDirectory() if tmp is None else None
    tmp = tmp or own.name
    try:
        stubs = _stub_dir(tmp)
        env = {
            "PATH": stubs + ":" + os.environ.get("PATH", ""),
            "HOME": tmp,
            "DRY_RUN": "1",
            "WT": str(_ROOT),
            "LOGDIR": os.path.join(tmp, "logs"),
            "SERVING_LOG": os.path.join(tmp, "logs", "boot.log"),
        }
        env.update(extra_env or {})
        p = subprocess.run(["/bin/bash", str(_BOOT), *args], env=env,
                           capture_output=True, text=True, timeout=120)
        return p.returncode, p.stdout, p.stderr
    finally:
        if own is not None:
            own.cleanup()


def _parse_dump(stdout):
    """Read the DRY_RUN dump: an ENV block then an ARGV block."""
    env, argv, where = {}, [], None
    for line in stdout.splitlines():
        if line.strip() == "=== DRY RUN ENV ===":
            where = "env"
            continue
        if line.strip() == "=== DRY RUN ARGV ===":
            where = "argv"
            continue
        if line.strip() == "=== DRY RUN: nothing was launched ===":
            where = None
            continue
        if where == "env" and "=" in line:
            k, v = line.split("=", 1)
            env[k] = v
        elif where == "argv" and line != "":
            argv.append(line)
    return env, argv


def _capture_pairs():
    pairs = {}
    for line in _CAPTURE.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        k, v = line.split("=", 1)
        pairs[k] = v
    return pairs


class TestNoPrivateCopyOfTheEnvironment(CustomTestCase):
    """The class falsifier: a key the capture owns may not be re-declared."""

    #: Set once here rather than re-derived: keys whose value is per boot and
    #: therefore cannot come from a capture.
    PER_BOOT = {"PYTHONPATH", "CUDA_VISIBLE_DEVICES",
                "SGLANG_PHASE_FLIP_INSTANCE", "SGLANG_BOOT_COMMIT"}

    def test_the_script_declares_no_key_the_capture_owns(self):
        owned = set(_capture_pairs()) - self.PER_BOOT
        text = _BOOT.read_text()
        offenders = []
        for n, line in enumerate(text.splitlines(), 1):
            s = line.strip()
            if s.startswith("#"):
                continue
            m = re.match(r"export\s+([A-Za-z_][A-Za-z0-9_]*)=", s)
            if m and m.group(1) in owned:
                offenders.append(f"{_BOOT.name}:{n}: {s[:70]}")
        self.assertFalse(
            offenders,
            "these keys come from the ship capture and must not be "
            "re-declared by the boot script (that private copy IS the "
            "2026-08-12 defect):\n  " + "\n  ".join(offenders))

    def test_the_script_sources_the_capture(self):
        text = _BOOT.read_text()
        self.assertIn("turnkey_539_export_env.py", text)
        self.assertIn("ship_env.capture", text)

    def test_the_gate_runs_before_the_launch_and_can_refuse(self):
        text = _BOOT.read_text()
        gate = text.index("--check")
        launch = text.index("setsid")
        self.assertLess(gate, launch,
                        "the --check gate must run BEFORE exec, not after")

    def test_there_is_no_blanket_bypass(self):
        text = _BOOT.read_text()
        for bad in ("--allow-all", "ALLOW_ALL", "SKIP_ENV_CHECK",
                    "NO_ENV_CHECK", "--no-check"):
            self.assertNotIn(bad, text,
                             f"{bad} would re-open the whole defect class")
        # A gate whose failure is swallowed is not a gate.
        self.assertNotRegex(text, r"--check[^\n]*\|\|\s*true")


class TestAssembledEnvironment(CustomTestCase):
    def setUp(self):
        rc, out, err = _dry_run()
        self.assertEqual(rc, 0, f"dry run failed:\n{out}\n{err}")
        self.env, self.argv = _parse_dump(out)
        self.out, self.err = out, err

    def test_the_ship_token_vector_is_what_the_boot_carries(self):
        """The single value that differed on the wedged boot."""
        self.assertEqual(self.env.get("SGLANG_UNEVEN_TOKEN_VECTOR"), "14,10,8")

    def test_the_five_keys_the_script_used_to_drop_are_present(self):
        want = {
            "SGLANG_CORRIDOR_FLOOR_MIB": "1536",
            "SGLANG_CORRIDOR_REBALANCE": "0",
            "SGLANG_KV_BACKING_RELIEF": "1",
            "SGLANG_SEAM_ENTRY_DELAY_BUDGET": "2",
            "SGLANG_SEAM_ENTRY_MARGIN_MIB": "512",
        }
        for k, v in want.items():
            self.assertEqual(self.env.get(k), v, f"{k} missing or wrong")

    def test_every_captured_key_is_reproduced_or_declared(self):
        cap = _capture_pairs()
        for k, v in cap.items():
            if k in TestNoPrivateCopyOfTheEnvironment.PER_BOOT:
                continue
            got = self.env.get(k)
            if got == v:
                continue
            # A divergence is allowed only if the script announced it.
            self.assertIn(f"OVERRIDE {k}", self.out + self.err,
                          f"{k} diverges ({got!r} vs {v!r}) without a "
                          f"declared override")

    def test_the_declared_overrides_are_printed_loudly_with_a_reason(self):
        text = self.out + self.err
        for k in ("PYTORCH_CUDA_ALLOC_CONF", "LD_LIBRARY_PATH"):
            self.assertIn(f"OVERRIDE {k}", text,
                          f"{k} genuinely diverges from the capture and must "
                          f"be announced, not applied silently")
        self.assertIn("reason", text.lower())

    def test_the_argv_still_launches_the_server(self):
        self.assertIn("sglang.launch_server", self.argv)
        self.assertIn("--enable-phase-flip", self.argv)
        self.assertIn("--port", self.argv)


class TestTheGateActuallyFires(CustomTestCase):
    def test_a_stray_governed_key_in_the_operators_shell_refuses_the_boot(self):
        """The failure mode the gate exists for: an SGLANG_* variable left
        over in the shell silently changing what boots."""
        rc, out, err = _dry_run({"SGLANG_A_STRAY_KNOB": "1"})
        self.assertNotEqual(rc, 0, "an unsanctioned key must refuse the boot")
        text = out + err
        self.assertIn("SGLANG_A_STRAY_KNOB", text)
        self.assertIn("REFUSE", text.upper())

    def test_a_stray_key_is_refused_before_anything_is_launched(self):
        rc, out, err = _dry_run({"SGLANG_A_STRAY_KNOB": "1"})
        self.assertNotEqual(rc, 0)
        self.assertNotIn("=== DRY RUN ARGV ===", out)

    def test_a_named_operator_tunable_is_accepted_and_announced(self):
        rc, out, err = _dry_run({"SGLANG_UNEVEN_TOKEN_VECTOR": "7,39,18"})
        self.assertEqual(rc, 0, f"{out}\n{err}")
        env, _ = _parse_dump(out)
        self.assertEqual(env.get("SGLANG_UNEVEN_TOKEN_VECTOR"), "7,39,18")
        self.assertIn("OVERRIDE SGLANG_UNEVEN_TOKEN_VECTOR", out + err)

    def test_a_declared_addition_not_in_the_capture_is_accepted(self):
        """PHASE_POLICY_FLIP_TOKENS sets a key the capture does not carry.
        Declared, so the gate lets it through and says so."""
        rc, out, err = _dry_run({"POLICY": "auto",
                                 "PHASE_POLICY_FLIP_TOKENS": "9000"})
        self.assertEqual(rc, 0, f"{out}\n{err}")
        env, _ = _parse_dump(out)
        self.assertEqual(env.get("SGLANG_PHASE_POLICY_FLIP_TOKENS"), "9000")
        self.assertIn("OVERRIDE SGLANG_PHASE_POLICY_FLIP_TOKENS", out + err)

    def test_a_missing_capture_refuses_rather_than_booting_bare(self):
        rc, out, err = _dry_run(
            {"SHIP_ENV_CAPTURE": "/nonexistent/ship_env.capture"})
        self.assertNotEqual(rc, 0)
        self.assertIn("REFUSE", (out + err).upper())


class TestTheTunableArmsStillWork(CustomTestCase):
    """Rewiring the environment must not have moved the argv. Each arm below
    was compared token for token against the pre-rewire script; these keep
    the observable half of that comparison after the old script is gone."""

    def _argv(self, extra_env):
        rc, out, err = _dry_run(extra_env)
        self.assertEqual(rc, 0, f"{out}\n{err}")
        return _parse_dump(out)[1]

    def test_default_arm_speculates_and_leaves_hicache_off(self):
        argv = self._argv({})
        self.assertIn("--speculative-algorithm", argv)
        self.assertNotIn("--enable-hierarchical-cache", argv)
        self.assertNotIn("--kv-pressure-ladder", argv)

    def test_spec_off_drops_the_draft_flags_only(self):
        argv = self._argv({"SPEC": "off"})
        self.assertNotIn("--speculative-algorithm", argv)
        self.assertIn("--enable-phase-flip", argv)

    def test_hicache_on_adds_the_hierarchical_flags(self):
        argv = self._argv({"HICACHE": "1"})
        self.assertIn("--enable-hierarchical-cache", argv)
        self.assertIn("--hicache-ratio", argv)

    def test_kv_ladder_is_opt_in_and_carries_its_vector(self):
        argv = self._argv({"KV_LADDER": "8000,4000,4000"})
        self.assertIn("--kv-pressure-ladder", argv)
        self.assertEqual(argv[argv.index("--kv-pressure-ladder") + 1],
                         "8000,4000,4000")

    def test_trailing_arguments_are_passed_through_intact(self):
        """Including a value with a space, which is the one that would break
        if the argv array had been assembled by string concatenation."""
        rc, out, err = _dry_run({}, args=["--extra", "a b"])
        self.assertEqual(rc, 0, f"{out}\n{err}")
        argv = _parse_dump(out)[1]
        self.assertEqual(argv[-2:], ["--extra", "a b"])

    def test_barlink_off_declares_both_keys_and_still_passes_the_gate(self):
        """BARLINK=0 diverges from the capture in two keys. It must remain
        reachable for an A/B, and it must say so rather than slip past."""
        rc, out, err = _dry_run({"BARLINK": "0"})
        self.assertEqual(rc, 0, f"{out}\n{err}")
        env, _ = _parse_dump(out)
        self.assertEqual(env.get("SGLANG_BARLINK"), "0")
        self.assertNotIn("SGLANG_BARLINK_TRANSPORT", env)
        text = out + err
        self.assertIn("OVERRIDE SGLANG_BARLINK", text)
        self.assertIn("OVERRIDE SGLANG_BARLINK_TRANSPORT", text)

    def test_rank_mib_and_context_still_reach_the_argv(self):
        argv = self._argv({"RANK_MIB": "1,2,3", "CTX": "4096"})
        self.assertEqual(argv[argv.index("--rank-gpu-memory-mib") + 1],
                         "1,2,3")
        self.assertEqual(argv[argv.index("--context-length") + 1], "4096")


if __name__ == "__main__":
    unittest.main()
