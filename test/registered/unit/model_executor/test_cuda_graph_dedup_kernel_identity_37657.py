"""CUDA-graph dedup signatures carry the kernel's function identity (#37657).

Ported from upstream sglang #37657 ("[Bugfix] Key CUDA graph dedup signatures
on kernel function identity"). ``kernel_node_payload`` keyed a kernel node on
its NAME plus launch geometry. Two graphs whose kernels share a name and a
geometry but are different functions (a template/JIT variant, the same Triton
kernel compiled for another constexpr set or another card's cubin) hashed to
the same signature, so ``DedupedCudaGraphRegistry`` could route one graph's
replay through the other's executable -- a wrong kernel, silently.

Upstream had no unit test (one-line fix). This one drives the production
``kernel_node_payload`` with a fake driver that returns the same name and
geometry for two nodes and different ``kern``/``func`` handles: before the
fix the payloads are equal (red), after it they differ (green); a node pair
that IS the same function still collapses, so dedup keeps working.

Scope note for the fork: ``SGLANG_ENABLE_CUDA_GRAPH_DEDUP`` defaults to False
and no weg2 launcher sets it, so the 27B boot does not take this path today.
"""

import unittest
from types import SimpleNamespace
from unittest.mock import patch

from sglang.srt.model_executor.runner_backend import cuda_graph_dedup_mixin as mod
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=2, suite="base-a-test-cpu")


class _FakeDriver:
    """Just the driver surface ``kernel_node_payload`` touches."""

    class CUkernelNodeAttrID:  # no attributes -> kernel_attrs() == ()
        pass

    def __init__(self, nodes):
        self._nodes = nodes

    def cuGraphKernelNodeGetParams(self, node):
        return (0, self._nodes[node])

    def cuKernelGetName(self, handle):
        return (0, b"_fused_qk_rmsnorm_rope_gate_kernel")

    def cuFuncGetName(self, handle):
        return (0, b"_fused_qk_rmsnorm_rope_gate_kernel")

    def cuGraphKernelNodeGetAttribute(self, node, attr):
        return (1, None)


def _params(kern, func):
    return SimpleNamespace(
        kern=kern,
        func=func,
        gridDimX=5,
        gridDimY=7,
        gridDimZ=1,
        blockDimX=128,
        blockDimY=1,
        blockDimZ=1,
        sharedMemBytes=0,
    )


class TestDedupKernelIdentity(CustomTestCase):
    def payloads(self, nodes):
        fake = _FakeDriver(nodes)
        with patch.object(mod, "cuda_drv", fake), patch.object(
            mod, "checkCudaErrors", lambda result: result[1]
        ):
            return {node: mod.kernel_node_payload(node) for node in nodes}

    def test_same_name_and_geometry_different_function_do_not_collide(self):
        got = self.payloads({"a": _params(0x1000, 0x2000), "b": _params(0x1000, 0x3000)})
        self.assertEqual(got["a"][0], got["b"][0])  # same kernel name
        self.assertNotEqual(got["a"], got["b"])

    def test_different_library_kernel_handle_does_not_collide(self):
        got = self.payloads({"a": _params(0x1000, 0), "b": _params(0x1100, 0)})
        self.assertNotEqual(got["a"], got["b"])

    def test_same_function_still_dedups(self):
        got = self.payloads({"a": _params(0x1000, 0x2000), "b": _params(0x1000, 0x2000)})
        self.assertEqual(got["a"], got["b"])


if __name__ == "__main__":
    unittest.main()
