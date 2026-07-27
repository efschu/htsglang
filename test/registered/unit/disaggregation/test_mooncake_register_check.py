"""CPU unit tests: Mooncake memory registration must fail loudly.

``MooncakeTransferEngine.batch_register`` returns an int status (0 = success,
non-zero = failure) and only logs at DEBUG level on failure. A caller that
discards that status turns a failed RDMA memory-region registration into a
silent condition that only surfaces much later, as a transfer error or as
corrupted data.
"""

import unittest
from types import SimpleNamespace
from unittest.mock import patch

from sglang.srt.disaggregation.mooncake.conn import MooncakeKVManager
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class FakeEngine:
    """Stands in for MooncakeTransferEngine; records calls, returns a status."""

    def __init__(self, register_result=0, deregister_result=0):
        self.register_result = register_result
        self.deregister_result = deregister_result
        self.register_calls = []
        self.deregister_calls = []

    def batch_register(self, ptrs, lengths):
        self.register_calls.append((list(ptrs), list(lengths)))
        return self.register_result

    def batch_deregister(self, ptrs):
        self.deregister_calls.append(list(ptrs))
        return self.deregister_result


def _make_kv_args(with_state=True):
    return SimpleNamespace(
        kv_data_ptrs=[0x1000, 0x2000],
        kv_data_lens=[4096, 8192],
        aux_data_ptrs=[0x3000],
        aux_data_lens=[256],
        state_data_ptrs=[[0x4000]] if with_state else [],
        state_data_lens=[[512]] if with_state else [],
        gpu_id=0,
    )


def _make_manager(engine, kv_args=None):
    mgr = MooncakeKVManager.__new__(MooncakeKVManager)
    mgr.engine = engine
    mgr.kv_args = kv_args if kv_args is not None else _make_kv_args()
    mgr._staging_ctx = SimpleNamespace(buffers=None, allocator=None)
    mgr.server_args = SimpleNamespace(chunked_prefill_size=8192)
    return mgr


class TestMooncakeRegisterBufferCheck(CustomTestCase):
    def test_register_buffer_success_registers_all_regions(self):
        engine = FakeEngine(register_result=0)
        mgr = _make_manager(engine)

        mgr.register_buffer_to_engine()

        # kv buffers, aux buffers, one state component
        self.assertEqual(len(engine.register_calls), 3)

    def test_register_buffer_raises_on_kv_failure(self):
        engine = FakeEngine(register_result=-1)
        mgr = _make_manager(engine)

        with self.assertRaises(RuntimeError) as ctx:
            mgr.register_buffer_to_engine()

        msg = str(ctx.exception)
        self.assertIn("-1", msg)
        # The message must name what failed, not just that something failed.
        self.assertIn("KV", msg)

    def test_register_buffer_raises_on_aux_failure(self):
        class AuxFailEngine(FakeEngine):
            def batch_register(self, ptrs, lengths):
                self.register_calls.append((list(ptrs), list(lengths)))
                # first call (kv) succeeds, second (aux) fails
                return 0 if len(self.register_calls) == 1 else -7

        engine = AuxFailEngine()
        mgr = _make_manager(engine)

        with self.assertRaises(RuntimeError) as ctx:
            mgr.register_buffer_to_engine()

        msg = str(ctx.exception)
        self.assertIn("-7", msg)
        self.assertIn("aux", msg.lower())

    def test_register_buffer_raises_on_state_failure(self):
        class StateFailEngine(FakeEngine):
            def batch_register(self, ptrs, lengths):
                self.register_calls.append((list(ptrs), list(lengths)))
                return 0 if len(self.register_calls) < 3 else -3

        engine = StateFailEngine()
        mgr = _make_manager(engine)

        with self.assertRaises(RuntimeError) as ctx:
            mgr.register_buffer_to_engine()

        self.assertIn("-3", str(ctx.exception))
        self.assertIn("state", str(ctx.exception).lower())


class TestMooncakeStagingRegisterCheck(CustomTestCase):
    """The staging register_fn handed to staging_handler must raise on failure.

    Buffer allocation itself needs a GPU, so the allocation helpers are
    replaced by stubs that capture the register callback and invoke it. The
    callback -- the part that drops the status code -- is the real one.
    """

    def _capture_register_fn(self, mgr, init_name, method):
        captured = {}

        def fake_init(register_fn, kv_args, *args, **kwargs):
            captured["fn"] = register_fn
            register_fn(0xDEADB000, 64 * 1024 * 1024)
            return [] if init_name == "init_staging_buffers" else object()

        with patch(
            f"sglang.srt.disaggregation.common.staging_handler.{init_name}",
            fake_init,
        ):
            method()
        return captured

    def test_staging_buffer_registration_failure_raises(self):
        engine = FakeEngine(register_result=-5)
        mgr = _make_manager(engine)

        with self.assertRaises(RuntimeError) as ctx:
            self._capture_register_fn(
                mgr,
                "init_staging_buffers",
                lambda: mgr._init_staging_buffers(2),
            )

        msg = str(ctx.exception)
        self.assertIn("-5", msg)
        self.assertIn("staging", msg.lower())
        # ptr and size identify which region failed
        self.assertIn(hex(0xDEADB000), msg)
        self.assertIn(str(64 * 1024 * 1024), msg)

    def test_staging_buffer_registration_success_passes(self):
        engine = FakeEngine(register_result=0)
        mgr = _make_manager(engine)

        self._capture_register_fn(
            mgr,
            "init_staging_buffers",
            lambda: mgr._init_staging_buffers(2),
        )

        self.assertEqual(engine.register_calls, [([0xDEADB000], [64 * 1024 * 1024])])

    def test_staging_allocator_registration_failure_raises(self):
        engine = FakeEngine(register_result=-9)
        mgr = _make_manager(engine)

        with self.assertRaises(RuntimeError) as ctx:
            self._capture_register_fn(
                mgr,
                "init_staging_allocator",
                mgr._init_staging_allocator,
            )

        self.assertIn("-9", str(ctx.exception))
        self.assertIn("staging", str(ctx.exception).lower())

    def test_staging_allocator_registration_success_passes(self):
        engine = FakeEngine(register_result=0)
        mgr = _make_manager(engine)

        self._capture_register_fn(
            mgr,
            "init_staging_allocator",
            mgr._init_staging_allocator,
        )

        self.assertEqual(engine.register_calls, [([0xDEADB000], [64 * 1024 * 1024])])


class TestStagingHandlerRegisterContract(CustomTestCase):
    """staging_handler must not accept a failure status from register_fn.

    Transport backends differ: NIXL raises from its callback, Mooncake returns
    a status code. The shared helper enforces the contract so a future backend
    that returns a code cannot regress into a silent failure.
    """

    def test_check_register_result_accepts_none_and_zero(self):
        from sglang.srt.disaggregation.common.staging_handler import (
            _check_register_result,
        )

        _check_register_result(None, 0x1000, 4096, "test buffer")
        _check_register_result(0, 0x1000, 4096, "test buffer")

    def test_check_register_result_rejects_nonzero(self):
        from sglang.srt.disaggregation.common.staging_handler import (
            _check_register_result,
        )

        with self.assertRaises(RuntimeError) as ctx:
            _check_register_result(-1, 0x1000, 4096, "test buffer")

        msg = str(ctx.exception)
        self.assertIn("-1", msg)
        self.assertIn("test buffer", msg)
        self.assertIn(hex(0x1000), msg)
        self.assertIn("4096", msg)

    def test_check_register_result_rejects_false(self):
        from sglang.srt.disaggregation.common.staging_handler import (
            _check_register_result,
        )

        with self.assertRaises(RuntimeError):
            _check_register_result(False, 0x1000, 4096, "test buffer")


if __name__ == "__main__":
    unittest.main()
