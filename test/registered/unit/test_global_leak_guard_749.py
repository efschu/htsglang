"""#749: the process-global leak guard must FIRE, and must name the leaker.

The guard lives in ``test/conftest.py`` (``pytest_runtest_teardown``). It
exists because a leak of this class is invisible where it happens and loud
somewhere else entirely: three ``unittest.mock.patch`` context managers entered
from inside worker THREADS raced on mock's restore stack, one restore was lost,
and ``sglang.srt.distributed.utils.uneven_dcp_active`` stayed ``lambda: True``
for the rest of the process -- producing ~50 failures across six error
signatures in a different test directory, intermittently.

A guard for that must be proved to fail. These tests run pytest in a
SUBPROCESS against a generated leaking test, because a guard that fires in this
process would fail the very test asserting it fired.
"""

import subprocess
import sys
import tempfile
import textwrap
import unittest
from pathlib import Path

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=3, suite="base-a-test-cpu")

_REPO = Path(__file__).resolve().parents[3]
_CONFTEST = _REPO / "test" / "conftest.py"

_LEAKING_TEST = """
import sglang.srt.distributed.utils as u

def test_a_leaks_a_global():
    u.uneven_dcp_active = lambda *a: True     # never restored

def test_b_is_innocent():
    assert True
"""

_MONKEYPATCH_TEST = """
import sglang.srt.distributed.utils as u

def test_a_uses_monkeypatch(monkeypatch):
    # Restored by a FIXTURE FINALIZER, not synchronously inside the test.
    monkeypatch.setattr(u, "uneven_dcp_active", lambda *a: True, raising=False)
    assert u.uneven_dcp_active() is True

def test_b_is_innocent():
    assert True
"""

_CLEAN_TEST = """
import unittest.mock
import sglang.srt.distributed.utils as u

def test_a_patches_and_restores():
    with unittest.mock.patch.object(u, "uneven_dcp_active", lambda *a: True):
        assert u.uneven_dcp_active() is True

def test_b_is_innocent():
    assert True
"""


def _run(body: str):
    """Run one generated test file under the real conftest, in a subprocess.

    Returns ``(exitcode, output)``.
    """
    with tempfile.TemporaryDirectory() as d:
        d = Path(d)
        (d / "conftest.py").write_text(_CONFTEST.read_text())
        (d / "test_generated_749.py").write_text(textwrap.dedent(body))
        proc = subprocess.run(
            [
                sys.executable,
                "-m",
                "pytest",
                str(d),
                "-q",
                "--tb=line",
                "-p",
                "no:randomly",
            ],
            capture_output=True,
            text=True,
            cwd=str(_REPO),
            env={
                "PATH": "/usr/bin:/bin",
                "PYTHONPATH": str(_REPO / "python"),
                "CUDA_VISIBLE_DEVICES": "99",
                "HOME": "/root",
            },
            timeout=900,
        )
        return proc.returncode, proc.stdout + proc.stderr


class TestTheGuardFires(CustomTestCase):
    def test_an_unrestored_global_fails_the_run(self):
        code, out = _run(_LEAKING_TEST)
        self.assertNotEqual(code, 0, "a leaked global must fail the session")
        self.assertIn("#749 PROCESS-GLOBAL LEAK GUARD", out)

    def test_the_report_names_the_leaker_the_attribute_and_the_cause(self):
        _, out = _run(_LEAKING_TEST)
        self.assertIn("test_a_leaks_a_global", out)
        self.assertIn("uneven_dcp_active", out)
        self.assertIn("THREAD", out.upper())

    def test_the_leak_does_NOT_cascade(self):
        """REPAIR, not just report -- the property that ends the pathology.

        One unrestored patch used to become ~50 downstream failures in another
        directory. The guard restores the baseline after recording, so the
        tests that follow a leaker still pass and the report keeps pointing at
        the cause instead of being buried by its consequences.

        This also pins that the recording does NOT raise inside
        ``pytest_runtest_teardown``: raising there aborts the teardown chain
        and pytest errors the next item with "previous item was not torn down
        properly" -- one leak, two failures, which is the cascade again in
        miniature. An earlier version of this guard did exactly that and this
        test is what found it.
        """
        _, out = _run(_LEAKING_TEST)
        self.assertIn("2 passed", out)
        self.assertNotIn("not torn down properly", out)

    def test_a_MONKEYPATCHED_global_does_NOT_fire(self):
        """CAN-FAIL GUARD for the hook ORDER, and it caught a real defect.

        pytest runs fixture finalizers from its own ``pytest_runtest_teardown``.
        A guard hook registered without ``trylast`` runs BEFORE them, sees
        monkeypatch's value still in place, and accuses an honest test. That is
        exactly what happened: the first version of this guard reported
        ``test_barlink_port.py`` as a leaker, and it was innocent.

        The other clean-case test cannot catch this, because it restores
        synchronously inside the test body via a ``with`` block -- it is
        already clean by the time ANY teardown hook runs. Only a
        fixture-restored patch distinguishes the two orderings.
        """
        code, out = _run(_MONKEYPATCH_TEST)
        self.assertEqual(code, 0, "monkeypatch is restored by pytest; not a leak")
        self.assertNotIn("#749 PROCESS-GLOBAL LEAK GUARD", out)

    def test_a_correctly_scoped_patch_does_NOT_fire(self):
        """CAN-FAIL GUARD for the guard: it must not accuse honest tests.

        A guard that fired on every mock.patch would be useless noise, and the
        suite is full of correctly-scoped patches.
        """
        code, out = _run(_CLEAN_TEST)
        self.assertEqual(code, 0)
        self.assertNotIn("#749 PROCESS-GLOBAL LEAK GUARD", out)
        self.assertIn("2 passed", out)


if __name__ == "__main__":
    unittest.main()
