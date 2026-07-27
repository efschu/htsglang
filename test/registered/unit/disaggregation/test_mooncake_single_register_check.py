"""CPU unit tests: single-region Mooncake registration must fail loudly.

``MooncakeTransferEngine.register`` wraps ``engine.register_memory`` and used
to swallow the status entirely: the return value was dropped and failure was
logged at DEBUG level only. Its callers -- the encode-disaggregation paths
that register embedding buffers and pools for RDMA -- could not observe a
failed memory-region registration, so the failure surfaced later as a
transfer error or as a payload read from an unregistered region, far from
the cause. The batch variants (``batch_register`` / ``batch_deregister``)
already return the status; these tests pin the same contract onto the
single-region calls and onto the raising helper the encode paths now use.
"""

import re
import unittest
from pathlib import Path

from sglang.srt.distributed.device_communicators.mooncake_transfer_engine import (
    MooncakeTransferEngine,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=4, suite="base-a-test-cpu")


class FakeBinding:
    """Stands in for the mooncake TransferEngine binding."""

    def __init__(self, register_result=0, deregister_result=0, raise_on=None):
        self.register_result = register_result
        self.deregister_result = deregister_result
        self.raise_on = raise_on or set()
        self.register_calls = []
        self.deregister_calls = []

    def register_memory(self, ptr, length):
        if "register" in self.raise_on:
            raise OSError("binding failure")
        self.register_calls.append((ptr, length))
        return self.register_result

    def unregister_memory(self, ptr):
        if "deregister" in self.raise_on:
            raise OSError("binding failure")
        self.deregister_calls.append(ptr)
        return self.deregister_result


def _make_engine(binding):
    engine = MooncakeTransferEngine.__new__(MooncakeTransferEngine)
    engine.engine = binding
    return engine


class TestSingleRegisterStatus(CustomTestCase):
    def test_register_returns_engine_status(self):
        for status in (0, -1, -7):
            with self.subTest(status=status):
                engine = _make_engine(FakeBinding(register_result=status))
                self.assertEqual(engine.register(0x1000, 4096), status)

    def test_register_returns_failure_when_binding_raises(self):
        engine = _make_engine(FakeBinding(raise_on={"register"}))
        self.assertNotEqual(engine.register(0x1000, 4096), 0)

    def test_deregister_returns_engine_status(self):
        for status in (0, -2):
            with self.subTest(status=status):
                engine = _make_engine(FakeBinding(deregister_result=status))
                self.assertEqual(engine.deregister(0x1000), status)


class TestRegisterChecked(CustomTestCase):
    def test_success_is_silent_and_registers(self):
        binding = FakeBinding(register_result=0)
        engine = _make_engine(binding)

        engine.register_checked(0x2000, 8192, "test buffer")

        self.assertEqual(binding.register_calls, [(0x2000, 8192)])

    def test_failure_raises_naming_region_and_status(self):
        engine = _make_engine(FakeBinding(register_result=-5))

        with self.assertRaises(RuntimeError) as ctx:
            engine.register_checked(0x2000, 8192, "test buffer")

        msg = str(ctx.exception)
        # The message must name what failed, where, and how.
        self.assertIn("test buffer", msg)
        self.assertIn(hex(0x2000), msg)
        self.assertIn("8192", msg)
        self.assertIn("-5", msg)

    def test_binding_exception_raises(self):
        engine = _make_engine(FakeBinding(raise_on={"register"}))

        with self.assertRaises(RuntimeError):
            engine.register_checked(0x2000, 8192, "test buffer")


class TestEncodePathsUseCheckedRegistration(CustomTestCase):
    """Source pin: the encode paths must not call bare ``register`` again.

    ``encode_server`` / ``encode_receiver`` import heavyweight serving
    dependencies, so the pin scans their source instead of importing them.
    A bare ``.register(`` on a transfer engine is exactly the call whose
    dropped status this change removes; ``register_checked`` raises at the
    point of failure. Deregistration stays unchecked by design: it sits on
    release/teardown paths, where a failure leaks a registration but cannot
    corrupt data, and raising would mask the original cause.
    """

    BARE_REGISTER = re.compile(r"engine\.register\(")

    def _source(self, module_name):
        import importlib.util

        spec = importlib.util.find_spec(
            f"sglang.srt.disaggregation.{module_name}"
        )
        return Path(spec.origin).read_text()

    def test_encode_server_has_no_bare_register(self):
        src = self._source("encode_server")
        self.assertIsNone(self.BARE_REGISTER.search(src))
        self.assertEqual(src.count("register_checked("), 2)

    def test_encode_receiver_has_no_bare_register(self):
        src = self._source("encode_receiver")
        self.assertIsNone(self.BARE_REGISTER.search(src))
        self.assertEqual(src.count("register_checked("), 3)


if __name__ == "__main__":
    unittest.main()
