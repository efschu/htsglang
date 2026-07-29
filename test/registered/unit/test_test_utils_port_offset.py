"""Importing sglang.test.test_utils must survive any CUDA_VISIBLE_DEVICES.

The test port carries an offset derived from the first visible device, so
two concurrent runs on different cards do not grab the same port. It was
computed as::

    int(os.environ.get("CUDA_VISIBLE_DEVICES", "0")[0])

at MODULE level, which turned three legal values of the variable into an
import-time exception:

* ``""`` -- the usual way to run a suite with no cards, e.g. while another
  run holds them -- raised ``IndexError: string index out of range``;
* the UUID form (``GPU-1a2b...``) would have raised ``ValueError``;
* and reading only the first character read ``"10"`` as ordinal 1.

Because ``test_utils`` is imported by essentially every test in the tree,
the first case meant that **no** test was runnable without cards, including
the ones that never touch a GPU. That is the hole this test closes.

Hermetic: only the helper is exercised, no card and no server.
"""

import importlib
import os
import subprocess
import sys
import unittest

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import (
    _MAX_DEVICE_ORDINAL_OFFSET,
    CustomTestCase,
    _first_visible_device_ordinal,
)

register_cpu_ci(est_time=15, suite="base-a-test-cpu")


class TestPortOffsetHelper(CustomTestCase):
    def _with_cvd(self, value):
        """Evaluate the helper with CUDA_VISIBLE_DEVICES set to ``value``."""
        old = os.environ.get("CUDA_VISIBLE_DEVICES")
        if value is None:
            os.environ.pop("CUDA_VISIBLE_DEVICES", None)
        else:
            os.environ["CUDA_VISIBLE_DEVICES"] = value
        try:
            return _first_visible_device_ordinal()
        finally:
            if old is None:
                os.environ.pop("CUDA_VISIBLE_DEVICES", None)
            else:
                os.environ["CUDA_VISIBLE_DEVICES"] = old

    def test_empty_string_means_no_offset(self):
        """The case that made the whole tree un-runnable without cards."""
        self.assertEqual(self._with_cvd(""), 0)

    def test_unset_means_no_offset(self):
        self.assertEqual(self._with_cvd(None), 0)

    def test_uuid_form_means_no_offset(self):
        self.assertEqual(self._with_cvd("GPU-1a2b3c4d-5e6f"), 0)
        self.assertEqual(self._with_cvd("MIG-GPU-1a2b3c4d"), 0)

    def test_plain_ordinals(self):
        self.assertEqual(self._with_cvd("0"), 0)
        self.assertEqual(self._with_cvd("2"), 2)
        self.assertEqual(self._with_cvd("1,2,3"), 1)
        self.assertEqual(self._with_cvd(" 3 , 4 "), 3)

    def test_large_ordinal_is_clamped_into_the_port_range(self):
        """A placeholder ordinal must not compute a port above 65535."""
        for value in ("99", "1000"):
            got = self._with_cvd(value)
            self.assertEqual(got, _MAX_DEVICE_ORDINAL_OFFSET)
            self.assertLess(20000 + got * 1000, 65536)
            self.assertLess(10000 + got * 2000, 65536)

    def test_module_imports_with_empty_cuda_visible_devices(self):
        """The real failure mode: import, in a fresh interpreter, no cards.

        Done in a subprocess because the constant is computed once at import
        and this process has already imported the module.
        """
        env = dict(os.environ)
        env["CUDA_VISIBLE_DEVICES"] = ""
        proc = subprocess.run(
            [
                sys.executable,
                "-c",
                "import sglang.test.test_utils as t; "
                "print(t.DEFAULT_PORT_FOR_SRT_TEST_RUNNER)",
            ],
            capture_output=True,
            text=True,
            env=env,
            timeout=300,
        )
        self.assertEqual(
            proc.returncode,
            0,
            msg=(
                "importing sglang.test.test_utils with CUDA_VISIBLE_DEVICES='' "
                f"failed:\n{proc.stderr[-3000:]}"
            ),
        )
        port = int(proc.stdout.strip().splitlines()[-1])
        self.assertTrue(1024 < port < 65536, msg=f"port out of range: {port}")

    def test_reimport_is_consistent(self):
        module = importlib.import_module("sglang.test.test_utils")
        self.assertTrue(1024 < module.DEFAULT_PORT_FOR_SRT_TEST_RUNNER < 65536)


if __name__ == "__main__":
    unittest.main()
