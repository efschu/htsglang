"""Unit test for the CUDA MPS control-daemon liveness probe (#82 co-location).

The co-location preflight used to warn based on os.path.exists() of the MPS
pipe DIRECTORY (/tmp/nvidia-mps). That directory lingers after a crashed or
killed daemon and can be a custom path for a healthy one, so it is not a
liveness signal. `_mps_control_daemon_responsive` instead probes
nvidia-cuda-mps-control directly. These tests patch shutil.which /
subprocess.run so they run on CPU with no GPU, no MPS, no real daemon.
"""

import subprocess
import unittest
from unittest import mock

from sglang.srt.entrypoints.engine import _mps_control_daemon_responsive
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=2, suite="base-a-test-cpu")

_MOD = "sglang.srt.entrypoints.engine"


def _completed(returncode=0, stdout="100.0\n"):
    return subprocess.CompletedProcess(
        args=["nvidia-cuda-mps-control"], returncode=returncode, stdout=stdout, stderr=""
    )


class TestMPSControlProbe(CustomTestCase):
    def test_responsive_daemon(self):
        with mock.patch("shutil.which", return_value="/usr/bin/nvidia-cuda-mps-control"), \
             mock.patch("subprocess.run", return_value=_completed(0, "100.0\n")):
            self.assertTrue(_mps_control_daemon_responsive())

    def test_binary_missing(self):
        # No nvidia-cuda-mps-control on PATH -> not responsive (never runs it).
        with mock.patch("shutil.which", return_value=None) as which, \
             mock.patch("subprocess.run") as run:
            self.assertFalse(_mps_control_daemon_responsive())
            run.assert_not_called()

    def test_daemon_down_nonzero_exit(self):
        # "Cannot find MPS control daemon process" exits non-zero.
        with mock.patch("shutil.which", return_value="/usr/bin/nvidia-cuda-mps-control"), \
             mock.patch("subprocess.run", return_value=_completed(1, "")):
            self.assertFalse(_mps_control_daemon_responsive())

    def test_empty_stdout_is_not_responsive(self):
        # Clean exit but no reply text -> treat as not responsive.
        with mock.patch("shutil.which", return_value="/usr/bin/nvidia-cuda-mps-control"), \
             mock.patch("subprocess.run", return_value=_completed(0, "  \n")):
            self.assertFalse(_mps_control_daemon_responsive())

    def test_timeout_is_not_responsive(self):
        with mock.patch("shutil.which", return_value="/usr/bin/nvidia-cuda-mps-control"), \
             mock.patch("subprocess.run",
                        side_effect=subprocess.TimeoutExpired(cmd="x", timeout=5)):
            self.assertFalse(_mps_control_daemon_responsive())

    def test_oserror_is_not_responsive(self):
        with mock.patch("shutil.which", return_value="/usr/bin/nvidia-cuda-mps-control"), \
             mock.patch("subprocess.run", side_effect=OSError("boom")):
            self.assertFalse(_mps_control_daemon_responsive())


if __name__ == "__main__":
    unittest.main()
