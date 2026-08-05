#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""pytest fixtures for the #337 / #591 guard tests.

WHY THIS FILE EXISTS
====================
``test_guards.py`` was written as a standalone script: ``main()`` loads the
harness module once and passes it to each check explicitly. Run that way it is
14 for 14.

But the file is named ``test_*.py`` and its functions are named ``test_*``, so
pytest collects it, sees the ``mb`` and ``tmp_plan`` parameters, and treats them
as fixture requests. With no fixtures defined that is ten collection errors --
one per parameterised test -- and the suite never runs. The file list in the
commit was complete; what was missing was the pytest entry point for a file
whose name promises one.

Two ways in, same checks:

    python3 scripts/trt_337/test_guards.py      # standalone, no pytest needed
    pytest scripts/trt_337/test_guards.py       # via these fixtures

The standalone path stays because these tests must be runnable on a bare desk
machine mid-window without pytest installed, and because it prints the can-fail
twins as first-class results rather than hiding them behind ``pytest.raises``.
"""

from __future__ import annotations

import importlib.util
import os
import sys

import pytest

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)


@pytest.fixture(scope="session")
def mb():
    """The harness module under test, loaded by path.

    Session-scoped: loading it executes the module, and there is no reason to
    pay that per test. Identical to what ``test_guards.main()`` does, so the two
    entry points exercise the same object.
    """
    spec = importlib.util.spec_from_file_location(
        "mb337", os.path.join(_HERE, "microbench_trt.py")
    )
    mod = importlib.util.module_from_spec(spec)
    sys.modules["mb337"] = mod
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture
def tmp_plan(tmp_path):
    """A stub plan file carrying the mock magic header."""
    p = tmp_path / "conformance.plan"
    p.write_bytes(b"MOCK337\x00{}")
    return str(p)
