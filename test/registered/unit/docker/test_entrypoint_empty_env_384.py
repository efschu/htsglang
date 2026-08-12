"""#384/#416: an empty flag variable must REMOVE the flag, not restore a default.

THE FAILURE THIS PINS, as it actually happened. `docker/htsglang-entrypoint.sh`
once declared `: "${RANK_GPU_ID:=0,1,2}"`. Bash's `:=` treats a set-but-empty
variable as unset, so clearing RANK_GPU_ID in an env file did not disable the
fork's rank mapping -- it re-applied the reference rig's profile. #416 booted an
image for stock even TP=2 and got `--rank-gpu-id 0,1,2`, which died on a
ValueError naming GPU 2 with two devices visible. The script's own header
meanwhile promised "Empty ENV => the flag is omitted entirely".

Commit 25d3a5ded2 emptied those four defaults. This file is the pin that keeps
them empty, because the defect is invisible on inspection: `:=` and `:-` differ
by one character, and a non-empty default reads like a helpful convenience
right up until someone clears the variable to turn a feature OFF.

HOW IT IS OBSERVED. The entrypoint ends in `exec "${args[@]}"`, so the
assembled argument list is only visible to whatever it execs. The test plants a
stub `python3` on PATH that prints its argv and exits, which is the same
technique test_prod_boot_capture_gate.py uses for the prod boot script. No
container, no GPU, no server -- the subject is bash argument assembly, and
asserting it in Python rather than by running bash would be testing the
assertion instead of the script.
"""

import os
import pathlib
import re
import subprocess
import tempfile
import unittest

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

_ROOT = pathlib.Path(__file__).resolve().parents[4]
_ENTRY = _ROOT / "docker" / "htsglang-entrypoint.sh"

#: The four the ticket names, with the non-empty default each ONCE had and the
#: flag it wrongly produced. Written out literally rather than derived from the
#: script, so that a re-introduced default cannot also update the expectation.
_ONCE_DEFAULTED = {
    "RANK_GPU_ID": "--rank-gpu-id",
    "SPECULATIVE_ALGORITHM": "--speculative-algorithm",
    "CHAT_TEMPLATE": "--chat-template",
    "ENABLE_HIERARCHICAL_CACHE": "--enable-hierarchical-cache",
}

#: The scalars that legitimately keep a non-empty default: the server needs a
#: value for each, so "" falling back is correct. Listed so the count is
#: pinned -- a NEW name appearing here is what the test is really watching for.
_ALLOWED_NONEMPTY_DEFAULTS = {
    "HICACHE_STORAGE_DIR",
    "MODE",
    "PLANNER_HOST",
    "PLANNER_PORT",
    "TP_SIZE",
    "HOST",
    "PORT",
}


def _run(env_over):
    """Run the entrypoint with a stub python3; return the argv it would exec."""
    with tempfile.TemporaryDirectory() as tmp:
        stub = pathlib.Path(tmp) / "python3"
        stub.write_text('#!/bin/bash\nprintf "ARGV:%s\\n" "$*"\nexit 0\n')
        stub.chmod(0o755)
        env = {
            "PATH": f"{tmp}:{os.environ.get('PATH', '')}",
            "HOME": tmp,
            "MODEL_PATH": "/models/fake",
        }
        env.update(env_over)
        p = subprocess.run(
            ["/bin/bash", str(_ENTRY)],
            env=env,
            capture_output=True,
            text=True,
            timeout=60,
        )
        assert p.returncode == 0, p.stdout + p.stderr
        for line in p.stdout.splitlines():
            if line.startswith("ARGV:"):
                return line[len("ARGV:") :].split()
        raise AssertionError(f"stub never ran:\n{p.stdout}\n{p.stderr}")


class TestEntrypointEmptyEnv(CustomTestCase):
    def test_unset_variables_produce_a_stock_launch(self):
        """The baseline the other assertions are read against."""
        argv = _run({})
        self.assertEqual(argv[:3], ["-m", "sglang.launch_server", "--model-path"])
        for flag in _ONCE_DEFAULTED.values():
            self.assertNotIn(flag, argv)

    def test_set_but_empty_removes_the_flag(self):
        """THE regression. Each of the four, cleared, must vanish.

        Note these are set-but-EMPTY, not unset: that is the distinction `:=`
        gets wrong, and an unset-only test would pass against the broken
        script.
        """
        argv = _run({k: "" for k in _ONCE_DEFAULTED})
        for var, flag in _ONCE_DEFAULTED.items():
            self.assertNotIn(flag, argv, f"{var}='' still produced {flag}")
        self.assertNotIn("0,1,2", " ".join(argv))

    def test_a_real_value_still_reaches_the_flag(self):
        """The other half: the seam is not dead, it is just off by default."""
        argv = _run({"RANK_GPU_ID": "0,1", "CHAT_TEMPLATE": "/tpl.jinja"})
        self.assertIn("--rank-gpu-id", argv)
        self.assertEqual(argv[argv.index("--rank-gpu-id") + 1], "0,1")
        self.assertIn("--chat-template", argv)

    def test_no_flag_variable_has_a_non_empty_default(self):
        """The structural pin, so the next flag added cannot repeat #416.

        Any `: "${VAR:=something}"` is either one of the seven scalars that
        legitimately carry a default, or a new instance of the defect.
        """
        found = dict(
            re.findall(
                r'^\s*:\s*"\$\{([A-Z0-9_]+):=([^}]+)\}"', _ENTRY.read_text(), re.M
            )
        )
        unexpected = set(found) - _ALLOWED_NONEMPTY_DEFAULTS
        self.assertEqual(
            unexpected,
            set(),
            f"new non-empty default(s) {unexpected}: a flag variable must use "
            f': "${{VAR:=}}" so that clearing it removes the flag (#416). If '
            f"this is genuinely a scalar the server always needs, add it to "
            f"_ALLOWED_NONEMPTY_DEFAULTS and say why in the entrypoint header.",
        )
        for var in _ONCE_DEFAULTED:
            self.assertNotIn(
                var, found, f"{var} regained a non-empty default; see #416"
            )


if __name__ == "__main__":
    unittest.main()
