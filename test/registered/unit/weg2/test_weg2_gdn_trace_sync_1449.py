# SPDX-License-Identifier: Apache-2.0
"""#1449: the #631b GDN-EXTEND trace no longer forces a device sync on every
linear layer.  py-spy on PP0 (boot weg2xsn206, 60 s at 50 Hz): 58 % of the
samples sat in `_q[-1].item()` -- one sync per GDN extend, per layer, per
chunk -- which stops the host from launching the next layer until the GPU
has drained.  The value is read only for the calls the trace prints (n<=40).
Hermetic: source pin."""
import inspect
import os
import unittest

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=3, suite="base-a-test-cpu")


class TraceDoesNotSync(CustomTestCase):
    def test_item_is_gated_by_the_trace_count(self):
        from sglang.srt.layers.attention.linear.kernels import gdn_triton
        src = inspect.getsource(gdn_triton.TritonGDNKernel.extend)
        self.assertIn("_total = (int(_q[-1].item()) if (_n <= 40 and", src)
        self.assertNotIn('_total = int(_q[-1].item()) if _q is not None and _q.numel() else -1', src)
        # the gate is evaluated BEFORE any .item(): _n is assigned first
        self.assertLess(src.index('_n = getattr(TritonGDNKernel, "_631b_n", 0) + 1'), src.index(".item()"))


if __name__ == "__main__":
    unittest.main()
