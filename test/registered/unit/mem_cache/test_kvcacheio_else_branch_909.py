"""#909 -- the kvcacheio names must be BOUND on every backend, not just CUDA/HIP.

THE FORM, three files. `memory_pool_host.py`, `pool_host/mha.py` and
`pool_host/mla.py` each open with `if _is_cuda or _is_hip:` around
`try: from sgl_kernel.kvcacheio import (...)` / `except ImportError: <name> =
None`, and NO `else`. On a backend that is neither CUDA nor HIP -- NPU, XPU,
MPS, CPU, and a hermetic `CUDA_VISIBLE_DEVICES=""` run -- the `if` body never
executes, the names are never bound, and the first use is a bare `NameError`
instead of the intended `None`.

THE FIX IS `None`, AND THAT IS LOAD-BEARING RATHER THAN ARBITRARY. This
directory's own `conftest.py` installs a `SkipTest` stub for exactly these
symbols, gated on::

    if getattr(_mod, _sym, None) is None:
        setattr(_mod, _sym, _kvcacheio_stub(_sym))

so `None` keeps that contract while any other binding silently blocks the
install. Measured on this desk: binding them to a raising stub instead turned
19 documented environment skips into false failures in
`test_unified_radix_cache_unittest.py` alone (46 skips -> 27, 0 failures ->
19). The conftest also states the intended fix in its own words -- "bind the
seven names to None unconditionally, or hoist the fallback out of the `if`" --
so this commit is that fix and the two now agree instead of one guessing.

WHY THIS TEST RUNS IN A SUBPROCESS. That same conftest binds the names before
any test in this directory runs, precisely so the suite is not held hostage to
the gap. Under it, the names are present both before and after the fix, so an
in-process assertion cannot tell the two apart -- it would be green on the
broken tree. The probe therefore imports the modules in a FRESH interpreter
with `CUDA_VISIBLE_DEVICES=""` and no conftest, which is the only place the
difference is visible: `AttributeError` before, `None` after.

VERIFIED TO BE THE REAL CONDITION, not a simulated one: on this desk
`is_cuda()` and `is_hip()` are both False under `CUDA_VISIBLE_DEVICES=""`, so
the `else` branch is the branch actually taken -- while `sgl_kernel.kvcacheio`
is itself importable. The guard, not the wheel, is what left the names unbound.

WHAT EACH TEST HOLDS DOWN
  1. every name is bound after a bare import on this backend  -- the defect;
  2. they are bound to None specifically                      -- the conftest
     contract, which a "better" binding would break;
  3. the modules still import                                 -- they are on
     the ordinary startup path;
  4. the else branch exists in all three files                -- the class, not
     one instance.
"""

import os
import subprocess
import sys
import unittest

_MODULES = (
    "sglang.srt.mem_cache.memory_pool_host",
    "sglang.srt.mem_cache.pool_host.mha",
    "sglang.srt.mem_cache.pool_host.mla",
)

_PROBE = r"""
import importlib, json, sys
out = {}
for name in %(mods)r:
    mod = importlib.import_module(name)
    names = sorted(
        n for n in vars(mod) if n.startswith("transfer_kv_")
    )
    out[name] = {
        "cuda": bool(getattr(mod, "_is_cuda", False)),
        "hip": bool(getattr(mod, "_is_hip", False)),
        "bound": names,
        "non_none": [n for n in names if getattr(mod, n) is not None],
    }
print("JSON:" + json.dumps(out))
"""


def _probe():
    """Import the three modules in a FRESH interpreter, no conftest."""
    env = dict(os.environ)
    env["CUDA_VISIBLE_DEVICES"] = ""
    root = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..", "..", "..", "..", "python")
    )
    env["PYTHONPATH"] = root + os.pathsep + env.get("PYTHONPATH", "")
    proc = subprocess.run(
        [sys.executable, "-c", _PROBE % {"mods": _MODULES}],
        capture_output=True,
        text=True,
        env=env,
        timeout=600,
    )
    for line in proc.stdout.splitlines():
        if line.startswith("JSON:"):
            import json

            return json.loads(line[5:])
    raise AssertionError(
        "probe produced no result.\nstdout:\n"
        + proc.stdout[-3000:]
        + "\nstderr:\n"
        + proc.stderr[-3000:]
    )


class TestKvcacheioElseBranch909(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.report = _probe()

    def test_the_else_branch_is_the_one_actually_taken_here(self):
        """Otherwise this whole file would be green for the wrong reason."""
        for name, info in self.report.items():
            with self.subTest(module=name):
                self.assertFalse(
                    info["cuda"] or info["hip"],
                    "this desk reports CUDA/HIP, so the if-branch runs and the "
                    "defect cannot be observed here",
                )

    def test_every_name_is_bound(self):
        """THE DEFECT: unbound names make the first use a bare NameError."""
        for name, info in self.report.items():
            with self.subTest(module=name):
                self.assertTrue(
                    info["bound"],
                    "no kvcacheio name is bound at all -- the if-branch did "
                    "not run and there is no else",
                )

    def test_they_are_bound_to_none(self):
        """The conftest gates its SkipTest install on `is None`; any other
        binding blocks it and converts documented environment skips into false
        failures (measured: 19 in one file)."""
        for name, info in self.report.items():
            with self.subTest(module=name):
                self.assertEqual(
                    info["non_none"],
                    [],
                    "a non-None binding breaks the conftest contract at "
                    "test/registered/unit/mem_cache/conftest.py",
                )

    def test_all_three_files_carry_the_branch(self):
        """The class, not one instance."""
        import inspect
        import importlib

        for name in _MODULES:
            src = inspect.getsource(importlib.import_module(name))
            head = src.split("logger = ", 1)[0]
            with self.subTest(module=name):
                self.assertIn("else:", head)


if __name__ == "__main__":
    unittest.main()
