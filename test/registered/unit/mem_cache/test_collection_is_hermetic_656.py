"""#656: importing this directory's test modules must not need an optional backend.

WHAT THIS EXISTS FOR
--------------------
``test_hicache_nixl_storage.py`` imported ``HiCacheNixl`` at module scope
without a guard. NIXL is an optional storage backend and is not installed on
this rig, so the import raised at COLLECTION time -- and pytest treats a
collection error as an interrupt for the whole run, not as one red module.
The measured effect: ``pytest test/registered/unit/mem_cache/`` reported
``2331 tests collected, 1 error`` and then ``Interrupted``, i.e. every test in
the directory was skipped by a failure in a module none of them import.

That is why no canonical arm covers this directory, and therefore why
``test_gdn_resident_cap_floor_656.py`` -- the only gate on an
instance-killing fix -- was red in a place nothing looks (MERGE-R9 12.7).

THE RULE THIS PINS
------------------
A test module may REQUIRE an optional dependency; it may not require it to be
IMPORTABLE in order to be COLLECTED. The dependency belongs behind a guard
that turns "absent" into a skip at run time, where it costs one module, and
never into a raise at import time, where it costs the directory.

Kept cheap on purpose: importing the modules is what collection does anyway,
so this adds no new machinery and cannot drift from the thing it protects.
"""

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=20, suite="base-a-test-cpu")

import importlib
import os
import pathlib
import unittest

_HERE = pathlib.Path(__file__).resolve().parent

#: Modules whose import is known to need something this check cannot supply
#: (a GPU, a running server). Empty by design -- an entry here is a hole in
#: the guarantee and must name the reason, not merely the module.
_EXEMPT: dict = {}


def _module_names():
    for path in sorted(_HERE.glob("test_*.py")):
        if path.name == os.path.basename(__file__):
            continue
        yield path.stem


class TheDirectoryImportsWithoutOptionalBackendsTest(unittest.TestCase):
    def test_every_test_module_here_imports(self):
        """The whole directory, not a named list.

        A named list would have passed on the day the defect landed, because
        the module that broke collection was not on anybody's list.
        """
        failures = []
        for name in _module_names():
            if name in _EXEMPT:
                continue
            try:
                importlib.import_module(name)
            except Exception as e:  # noqa: BLE001 -- the point is to report it
                failures.append(f"{name}: {type(e).__name__}: {e}")
        self.assertEqual(
            [],
            failures,
            "these modules raise at IMPORT time, which interrupts collection "
            "for the whole directory rather than failing one module:\n  "
            + "\n  ".join(failures),
        )

    def test_the_nixl_storage_module_imports_without_nixl(self):
        """The specific regression, named, so a re-break says what broke.

        Asserted unconditionally: with NIXL present the import succeeds for
        the ordinary reason, and with NIXL absent it must succeed because the
        guard caught it. Both are the same assertion, which is what makes the
        pin hold on a rig that later installs the backend.
        """
        mod = importlib.import_module("test_hicache_nixl_storage")
        self.assertTrue(hasattr(mod, "TestNixlFileLayout"))

    def test_the_layout_tests_do_not_need_the_backend_at_all(self):
        """``TestNixlFileLayout`` was hidden by the import, not gated by it.

        It exercises ``nixl_routing``, which is pure key arithmetic and
        imports on any host. Collection-blocking hid a suite that had no
        dependency on the thing that was missing -- the clearest statement of
        why the import guard belongs at the import and not at the class.
        """
        mod = importlib.import_module("test_hicache_nixl_storage")
        cls = mod.TestNixlFileLayout
        names = [n for n in dir(cls) if n.startswith("test_")]
        self.assertTrue(names, "TestNixlFileLayout carries no tests")


if __name__ == "__main__":
    unittest.main()
