"""kv_displace_would_fit counts tensor fields with len(), never `x or ()`.

Metal z30y6 16:47:04 (27B dual24, all three D ranks, W17): prefix_indices is a torch
tensor on D, and `len(getattr(older, "prefix_indices", None) or ())` raised
"Boolean value of Tensor with no values is ambiguous". The same line sat in the NF
seat code (d_park_runtime); NF never reached the path (DISPLACE-VERDICT 0 in y4x).
"""

import types
import unittest

import torch

from sglang.srt.weg2 import d_park_runtime as DP


class KvDisplaceTensorFields(unittest.TestCase):
    def _sched(self, older):
        return types.SimpleNamespace(
            waiting_queue=[older],
            token_to_kv_pool_allocator=types.SimpleNamespace(available_size=lambda: 400),
            tree_cache=types.SimpleNamespace(evictable_size=lambda: 0),
        )

    def test_tensor_fields_are_counted_not_truth_tested(self):
        older = types.SimpleNamespace(rid="weg2-0-6", origin_input_ids=list(range(1000)), output_ids=[],
                                      prefix_indices=torch.empty(0, dtype=torch.int64))
        young = types.SimpleNamespace(rid="weg2-0-10", origin_input_ids=torch.zeros(700, dtype=torch.int64),
                                      output_ids=[], prefix_indices=torch.arange(3))
        sched = self._sched(older)
        self.assertTrue(DP.kv_displace_would_fit(sched, "weg2-0-6", [young]))
        self.assertTrue(DP.kv_displace_would_fit(sched, "weg2-0-6", [young], seat=True))
        older.prefix_indices = torch.arange(900)  # non-empty tensor: 100 left, fits the free 400
        self.assertFalse(DP.kv_displace_would_fit(sched, "weg2-0-6", [young]))

    def test_req_kv_tokens_counts_tensors(self):
        r = types.SimpleNamespace(origin_input_ids=torch.zeros(5, dtype=torch.int64), output_ids=torch.empty(0))
        self.assertEqual(DP._req_kv_tokens(r), 5)


if __name__ == "__main__":
    unittest.main()
