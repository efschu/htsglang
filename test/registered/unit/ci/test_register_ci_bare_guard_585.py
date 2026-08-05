"""Test that register_*_ci markers reject bare-decorator misuse.

Guard against the pattern:
    @register_cpu_ci
    class MyTests: ...

which passes the class as est_time, the function returns None,
and the class is silently replaced by None -- pytest collects zero tests.
"""

import inspect
import re

import pytest

from sglang.test.ci import ci_register


class DummyClass:
    """A simple class to simulate passing a class as est_time."""
    pass


def dummy_function():
    """A simple function to simulate passing a callable as est_time."""
    pass


# ---------------------------------------------------------------------------
# Collect all register_*_ci functions dynamically via introspection
# ---------------------------------------------------------------------------
def _get_all_register_cis():
    """Return every module-level register_*_ci function from ci_register."""
    results = []
    seen_names = set()
    for name, obj in inspect.getmembers(ci_register):
        if name.startswith("register_") and name.endswith("_ci") and callable(obj):
            if name not in seen_names:
                seen_names.add(name)
                results.append((name, obj))
    return results


ALL_REGISTERS = _get_all_register_cis()

# There are 7 unique register functions (register_musa_ci is defined twice
# in the source but only one binding survives at runtime).
EXPECTED_UNIQUE_NAMES = {
    "register_cpu_ci",
    "register_cuda_ci",
    "register_amd_ci",
    "register_musa_ci",
    "register_npu_ci",
    "register_xpu_ci",
    "register_mlx_ci",
}


# ---------------------------------------------------------------------------
# (a) Bare-decorator simulation: passing a class raises TypeError
# ---------------------------------------------------------------------------
_class_params = [
    pytest.param(n, f, id=f"class_bare_{n}") for n, f in ALL_REGISTERS
]


@pytest.mark.parametrize("name, func", _class_params)
def test_bare_decorator_class_raises(name, func):
    """Passing a class as est_time must raise TypeError naming the function."""
    err_pattern = re.compile(
        rf"register_\w+_ci was applied as a bare decorator.*pytest collection"
    )
    with pytest.raises(TypeError) as exc:
        func(DummyClass)
    msg = str(exc.value)
    assert err_pattern.search(msg), (
        f"Error message did not match expected pattern: {msg!r}"
    )
    # The error must name the specific function
    assert name in msg, f"Error message should name {name}, got: {msg!r}"


# ---------------------------------------------------------------------------
# (b) Bare-decorator simulation: passing a function raises TypeError
# ---------------------------------------------------------------------------
_func_params = [
    pytest.param(n, f, id=f"func_bare_{n}") for n, f in ALL_REGISTERS
]


@pytest.mark.parametrize("name, func", _func_params)
def test_bare_decorator_function_raises(name, func):
    """Passing a function as est_time must raise TypeError naming the function."""
    err_pattern = re.compile(
        rf"register_\w+_ci was applied as a bare decorator.*pytest collection"
    )
    with pytest.raises(TypeError) as exc:
        func(dummy_function)
    msg = str(exc.value)
    assert err_pattern.search(msg), (
        f"Error message did not match expected pattern: {msg!r}"
    )
    assert name in msg, f"Error message should name {name}, got: {msg!r}"


# ---------------------------------------------------------------------------
# (c) Valid call-style: normal usage returns None without raising
# ---------------------------------------------------------------------------
_call_params = [
    pytest.param(n, f, id=f"call_ok_{n}") for n, f in ALL_REGISTERS
]


@pytest.mark.parametrize("name, func", _call_params)
def test_valid_call_returns_none(name, func):
    """Call-style invocation must return None and not raise."""
    result = func(est_time=1.0, suite="test-suite")
    assert result is None, (
        f"{name}() should return None for valid call-style usage"
    )


# ---------------------------------------------------------------------------
# (d) Introspection sweep: all register_*_ci have the guard
# ---------------------------------------------------------------------------
def test_all_registers_are_introspectable():
    """Verify we found the expected unique register functions."""
    names = {n for n, _ in ALL_REGISTERS}
    assert names == EXPECTED_UNIQUE_NAMES, (
        f"Unexpected register functions found. "
        f"Expected: {EXPECTED_UNIQUE_NAMES}, Got: {names}"
    )


def test_guard_source_inspection():
    """Every register_*_ci function must contain the guard pattern in its source."""
    for name, func in ALL_REGISTERS:
        source = inspect.getsource(func)
        assert "isinstance(est_time, type)" in source, (
            f"{name} is missing the isinstance(est_time, type) guard"
        )
        assert "callable(est_time)" in source, (
            f"{name} is missing the callable(est_time) guard"
        )
        assert "TypeError" in source, (
            f"{name} is missing the TypeError raise"
        )


if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([__file__, "-v"]))
