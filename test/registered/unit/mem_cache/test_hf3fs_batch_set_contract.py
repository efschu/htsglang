"""CPU unit test: HiCacheHF3FS._batch_set must honour its List[bool] contract.

On the MLA skip-backup path (only one rank writes the KV cache) ``_batch_set``
returned a scalar ``True``. ``batch_set_v1`` passes that through unchanged and
``_page_set_zero_copy`` in the cache controller does ``all(...)`` on it, which
raises ``TypeError: 'bool' object is not iterable``.
"""

import unittest

from sglang.srt.mem_cache.storage.hf3fs.storage_hf3fs import HiCacheHF3FS
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


def _make_store(skip_backup: bool):
    store = HiCacheHF3FS.__new__(HiCacheHF3FS)
    store.skip_backup = skip_backup
    store.rank = 1
    return store


class TestHF3FSBatchSetContract(CustomTestCase):
    def test_skip_backup_returns_one_bool_per_key(self):
        store = _make_store(skip_backup=True)

        result = store._batch_set(["k0", "k1", "k2"])

        self.assertIsInstance(result, list)
        self.assertEqual(result, [True, True, True])

    def test_skip_backup_result_is_consumable_by_all(self):
        # The exact expression used by
        # cache_controller._page_set_zero_copy.
        store = _make_store(skip_backup=True)

        self.assertTrue(all(store._batch_set(["k0", "k1"])))

    def test_skip_backup_empty_keys_returns_empty_list(self):
        store = _make_store(skip_backup=True)

        result = store._batch_set([])

        self.assertIsInstance(result, list)
        self.assertEqual(result, [])


if __name__ == "__main__":
    unittest.main()
