"""CPU unit tests for EIC storage status handling.

``eic_storage`` imports the ``eic`` pybind module at import time, so a stub is
installed in ``sys.modules`` before the import. The stub mirrors only what the
functions under test touch: a status enum, a string vector and an exist/set
option.
"""

import sys
import types
import unittest
from types import SimpleNamespace

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class _StatusCode:
    SUCCESS = 0
    PARTIAL_FAILED = 1
    FAILED = 2


def _install_eic_stub():
    if "eic" in sys.modules and getattr(sys.modules["eic"], "_sglang_stub", False):
        return sys.modules["eic"]
    stub = types.ModuleType("eic")
    stub._sglang_stub = True
    stub.StatusCode = _StatusCode
    stub.StringVector = list

    class _Opt:
        def __init__(self):
            self.ns = None
            self.ttl_second = None

    stub.ExistOption = _Opt
    stub.SetOption = _Opt
    stub.DelOption = _Opt
    stub.GetOption = _Opt

    class _IOBuffers:
        def __init__(self):
            self.items = []

        def append(self, *args):
            self.items.append(args)

        def __len__(self):
            return len(self.items)

    stub.IOBuffers = _IOBuffers
    stub.MemoryInfo = _Opt
    stub.MemoryType = SimpleNamespace(MEMORY_CUDA=0)
    sys.modules["eic"] = stub
    return stub


_install_eic_stub()

from sglang.srt.mem_cache.storage.eic.eic_storage import EICStorage  # noqa: E402


class FakeConnection:
    """Records mexist/mset calls and replays canned outcomes."""

    def __init__(self, exist_outcomes=None, mset_result=None):
        self.exist_outcomes = list(exist_outcomes or [])
        self.mset_result = mset_result
        self.mexist_calls = []
        self.mset_calls = []

    def mexist(self, keys_vec, option):
        self.mexist_calls.append(list(keys_vec))
        return self.exist_outcomes.pop(0)

    def mset(self, keys_vec, vals_vec, option):
        self.mset_calls.append(list(keys_vec))
        return self.mset_result


class FakeTensor:
    """Minimal stand-in for a KV value tensor."""

    def __init__(self, ptr=0x1000):
        self._ptr = ptr

    def data_ptr(self):
        return self._ptr

    def element_size(self):
        return 2

    def numel(self):
        return 128


def _outcome(codes):
    return SimpleNamespace(status_codes=list(codes))


def _make_storage(connection):
    st = EICStorage.__new__(EICStorage)
    st.connection = connection
    st.eic_namespace = "ns"
    st.enable_kv_set_direct = False
    st.kv_cache_write_mem_pool = None
    st._get_eic_key = lambda keys: list(keys)
    return st


class TestEICBatchExistsMaskAlignment(CustomTestCase):
    """A failed mexist batch must not shift the existence mask.

    _batch_exists_impl returns one bool per key; batch_exists counts the
    contiguous True prefix and hands that to the controller as the number of
    KV pages that can be loaded from L3. A mask longer than the key list
    misaligns every entry after the failing batch, so a page can be counted as
    present when it is not.
    """

    def test_failed_batch_yields_one_entry_per_key(self):
        conn = FakeConnection(
            exist_outcomes=[(_StatusCode.FAILED, _outcome([_StatusCode.SUCCESS] * 3))]
        )
        st = _make_storage(conn)

        mask = st._batch_exists_impl(["k0", "k1", "k2"])

        self.assertEqual(len(mask), 3)
        self.assertEqual(mask, [False, False, False])

    def test_successful_batch_maps_codes_one_to_one(self):
        conn = FakeConnection(
            exist_outcomes=[
                (
                    _StatusCode.SUCCESS,
                    _outcome(
                        [_StatusCode.SUCCESS, _StatusCode.FAILED, _StatusCode.SUCCESS]
                    ),
                )
            ]
        )
        st = _make_storage(conn)

        mask = st._batch_exists_impl(["k0", "k1", "k2"])

        self.assertEqual(mask, [True, False, True])

    def test_empty_keys_returns_empty_list_not_int(self):
        st = _make_storage(FakeConnection())

        result = st._batch_exists_impl([])

        self.assertEqual(result, [])
        self.assertIsInstance(result, list)


class TestEICGenericBatchSetStatus(CustomTestCase):
    """A failed mset must be reported as failure, not as all-success."""

    def test_failed_mset_reports_all_keys_failed(self):
        conn = FakeConnection(
            mset_result=(_StatusCode.FAILED, _outcome([_StatusCode.SUCCESS] * 2))
        )
        st = _make_storage(conn)

        result = st.generic_batch_set(
            ["k0", "k1"], [FakeTensor(0x1000), FakeTensor(0x2000)]
        )

        self.assertEqual(result, [False, False])

    def test_partial_failure_beyond_first_key_reports_failure(self):
        conn = FakeConnection(
            mset_result=(
                _StatusCode.SUCCESS,
                _outcome([_StatusCode.SUCCESS, _StatusCode.FAILED]),
            )
        )
        st = _make_storage(conn)

        result = st.generic_batch_set(
            ["k0", "k1"], [FakeTensor(0x1000), FakeTensor(0x2000)]
        )

        self.assertEqual(result, [False, False])

    def test_all_success_reports_success(self):
        conn = FakeConnection(
            mset_result=(_StatusCode.SUCCESS, _outcome([_StatusCode.SUCCESS] * 2))
        )
        st = _make_storage(conn)

        result = st.generic_batch_set(
            ["k0", "k1"], [FakeTensor(0x1000), FakeTensor(0x2000)]
        )

        self.assertEqual(result, [True, True])

    def test_empty_keys_returns_empty_list_not_bool(self):
        st = _make_storage(FakeConnection())

        result = st.generic_batch_set([], [])

        self.assertEqual(result, [])
        self.assertIsInstance(result, list)


if __name__ == "__main__":
    unittest.main()
