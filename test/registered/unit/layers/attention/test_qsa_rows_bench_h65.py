"""fnFL2 H65: weg2/tools/qsa_rows_bench.py, the QSA attention micro-bench for
a GPU window. CPU only: it refuses before any CUDA call without a running
gpuq booking that holds the card, and the top-k rows it builds have the shape
the kernels get (valid first, causal, 2051 wide = 512 groups x 4 + 3 tail).
"""

import unittest
from unittest import mock

import torch

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=2, suite="base-a-test-cpu")


class BenchToolTest(unittest.TestCase):
    """weg2/tools/qsa_rows_bench.py: the window bench refuses before any
    CUDA call without a running booking, and its top-k rows have the shape
    the kernel gets (valid first, causal, 2051 wide)."""

    def test_refuses_without_a_running_booking(self):
        from sglang.srt.weg2.tools import qsa_rows_bench as qb

        with self.assertRaisesRegex(SystemExit, "--booking"):
            qb.main(["--card", "0"])

        class _Resp:
            def __init__(self, body):
                self.body = body

            def __enter__(self):
                return self

            def __exit__(self, *a):
                return False

            def read(self):
                return self.body.encode()

        with mock.patch.object(qb.urllib.request, "urlopen", lambda *a, **k: _Resp('{"state": "pending", "cards": [0]}')):
            with self.assertRaisesRegex(SystemExit, "not running"):
                qb.main(["--card", "0", "--booking", "abc"])
        with mock.patch.object(qb.urllib.request, "urlopen", lambda *a, **k: _Resp('{"state": "running", "cards": [1]}')):
            with self.assertRaisesRegex(SystemExit, "not 0"):
                qb.main(["--card", "0", "--booking", "abc"])

    def test_selection_is_what_the_kernel_gets(self):
        from sglang.srt.weg2.tools import qsa_rows_bench as qb

        for prefix, pattern in ((0, "shared"), (32768, "shared"), (32768, "random"), (32768, "same")):
            gen = torch.Generator().manual_seed(7)
            pos = prefix + torch.arange(640)
            sel = qb._selection(torch, pos, pattern, gen, torch.device("cpu")).long()
            self.assertEqual(tuple(sel.shape), (640, qb.TOPK_COLS))
            valid = sel >= 0
            # valid first, then -1 padding
            self.assertTrue(bool((valid[:, 1:] <= valid[:, :-1]).all()))
            # causal: nothing beyond the query's own position
            self.assertTrue(bool((torch.where(valid, sel, 0) <= pos.unsqueeze(1)).all()))
            counts = valid.sum(1)
            if prefix:
                self.assertTrue(bool((counts >= 2048).all()))
            else:  # prefix-free: row r sees min(r + 1, 2048..2051) tokens
                self.assertEqual(int(counts[0]), 1)
                self.assertEqual(int(counts[99]), 100)


if __name__ == "__main__":
    unittest.main()
