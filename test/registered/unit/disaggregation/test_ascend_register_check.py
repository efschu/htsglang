"""CPU unit tests: Ascend memory registration must fail loudly.

``AscendKVManager`` subclasses ``MooncakeKVManager`` and ``AscendTransferEngine``
subclasses ``MooncakeTransferEngine``, but both override the registration path
with a variant that drops the status: the engine does not return it, and the
manager does not check it. Same failure family as the Mooncake path.
"""

import unittest
from types import SimpleNamespace

from sglang.srt.disaggregation.ascend.conn import AscendKVManager
from sglang.srt.disaggregation.ascend.transfer_engine import AscendTransferEngine
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class FakeNpuEngine:
    """Stands in for the memfabric_hybrid TransferEngine binding."""

    def __init__(self, result=0, raises=False):
        self.result = result
        self.raises = raises
        self.calls = []

    def batch_register_memory(self, ptrs, lengths):
        self.calls.append((list(ptrs), list(lengths)))
        if self.raises:
            raise RuntimeError("npu registration blew up")
        return self.result


class FakeAscendEngine:
    """Stands in for AscendTransferEngine at the manager level."""

    def __init__(self, result=0):
        self.result = result
        self.calls = []

    def batch_register(self, ptrs, lengths):
        self.calls.append((list(ptrs), list(lengths)))
        return self.result


def _make_kv_args():
    return SimpleNamespace(
        kv_data_ptrs=[0x1000, 0x2000],
        kv_data_lens=[4096, 8192],
        aux_data_ptrs=[0x3000],
        aux_data_lens=[256],
        state_data_ptrs=[[0x4000]],
        state_data_lens=[[512]],
        gpu_id=0,
    )


def _make_manager(engine):
    mgr = AscendKVManager.__new__(AscendKVManager)
    mgr.engine = engine
    mgr.kv_args = _make_kv_args()
    return mgr


class TestAscendTransferEngineBatchRegister(CustomTestCase):
    def test_batch_register_returns_status_on_success(self):
        engine = AscendTransferEngine.__new__(AscendTransferEngine)
        engine.engine = FakeNpuEngine(result=0)

        self.assertEqual(engine.batch_register([0x1000], [4096]), 0)

    def test_batch_register_returns_status_on_failure(self):
        engine = AscendTransferEngine.__new__(AscendTransferEngine)
        engine.engine = FakeNpuEngine(result=-2)

        self.assertEqual(engine.batch_register([0x1000], [4096]), -2)

    def test_batch_register_returns_failure_status_when_binding_raises(self):
        engine = AscendTransferEngine.__new__(AscendTransferEngine)
        engine.engine = FakeNpuEngine(raises=True)

        self.assertNotEqual(engine.batch_register([0x1000], [4096]), 0)


class TestAscendRegisterBufferCheck(CustomTestCase):
    def test_register_buffer_success_registers_all_regions(self):
        engine = FakeAscendEngine(result=0)
        mgr = _make_manager(engine)

        mgr.register_buffer_to_engine()

        self.assertEqual(len(engine.calls), 3)

    def test_register_buffer_raises_on_kv_failure(self):
        engine = FakeAscendEngine(result=-1)
        mgr = _make_manager(engine)

        with self.assertRaises(RuntimeError) as ctx:
            mgr.register_buffer_to_engine()

        msg = str(ctx.exception)
        self.assertIn("-1", msg)
        self.assertIn("Ascend", msg)
        self.assertIn("KV", msg)

    def test_register_buffer_raises_on_state_failure(self):
        class StateFailEngine(FakeAscendEngine):
            def batch_register(self, ptrs, lengths):
                self.calls.append((list(ptrs), list(lengths)))
                return 0 if len(self.calls) < 3 else -4

        engine = StateFailEngine()
        mgr = _make_manager(engine)

        with self.assertRaises(RuntimeError) as ctx:
            mgr.register_buffer_to_engine()

        self.assertIn("-4", str(ctx.exception))
        self.assertIn("state", str(ctx.exception).lower())


if __name__ == "__main__":
    unittest.main()
