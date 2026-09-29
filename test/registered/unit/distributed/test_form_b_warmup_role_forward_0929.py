# SPDX-License-Identifier: Apache-2.0
"""The warmup / autotune dummy forward takes the rank's ROLE route (nvfp4lane 09291648).

METAL (a3_formb, a74de9c84a): "FlashInfer warmup autotune: another rank of this TP group tunes; this rank ...
runs the same dummy forward UNTUNED" -> base_runner._dummy_run -> mr.model.forward on KV-only rank 2 ->
qwen3_5 layer 0 input_layernorm -> gemma_rmsnorm -> "ValueError: Expected a cuda device, but got: meta".
A weightless KV worker holds a META model; its served forward is _forward_weightless_worker (_forward_raw).
"""
import types
import unittest
from unittest import mock

try:
    from sglang.test.ci.ci_register import register_cpu_ci
except ImportError:  # pragma: no cover

    def register_cpu_ci(*args, **kwargs):
        return None


from sglang.srt.layers.dcp import collective_guard as cg
from sglang.srt.model_executor.runner import base_runner as br
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=2, suite="base-a-test-cpu")


def _meta_forward(*a, **k):
    raise ValueError("Expected a cuda device, but got: meta")


def _runner(*, worker=False, head=False, form_a=False):
    calls = []
    mr = types.SimpleNamespace(
        is_weightless_worker=worker, is_weightless_head=head, is_form_a_worker=form_a,
        model=types.SimpleNamespace(forward=_meta_forward if (worker or form_a)
                                    else (lambda *a, **k: calls.append("model") or "logits")),
        _forward_weightless_worker=lambda fb: calls.append("worker") or "worker-out",
        _forward_form_a_worker=lambda fb: calls.append("form_a") or "form-a-out",
    )
    return mr, calls


class WarmupRoleForwardTest(CustomTestCase):
    def setUp(self):
        self.resets = []
        p = mock.patch.object(cg, "reset_forward_guard", lambda: self.resets.append(1))
        p.start()
        self.addCleanup(p.stop)

    def test_kv_only_rank_takes_the_worker_route(self):
        mr, calls = _runner(worker=True)
        self.assertEqual(br.role_dummy_forward(mr, None, types.SimpleNamespace(positions=None), {}), "worker-out")
        self.assertEqual(calls, ["worker"])
        self.assertEqual(len(self.resets), 1)  # the same guard step reset _forward_raw does

    def test_head_runs_model_forward_with_the_reset(self):
        mr, calls = _runner(head=True)
        br.role_dummy_forward(mr, None, types.SimpleNamespace(positions=None), {})
        self.assertEqual(calls, ["model"])
        self.assertEqual(len(self.resets), 1)

    def test_form_a_worker_takes_its_route(self):
        mr, calls = _runner(form_a=True)
        br.role_dummy_forward(mr, None, types.SimpleNamespace(positions=None), {})
        self.assertEqual(calls, ["form_a"])

    def test_plain_rank_is_byte_identical(self):
        mr, calls = _runner()
        out = br.role_dummy_forward(mr, "ids", types.SimpleNamespace(positions="pos"), {"k": 1})
        self.assertEqual((out, calls, self.resets), ("logits", ["model"], []))

    def test_dummy_run_uses_the_role_entry(self):
        """_dummy_run's run_once reaches the forward only through role_dummy_forward."""
        import inspect

        src = inspect.getsource(br.BaseRunner._dummy_run) if hasattr(br, "BaseRunner") else inspect.getsource(br)
        self.assertIn("role_dummy_forward(", src)
        self.assertNotIn("mr.model.forward(", src)


if __name__ == "__main__":
    unittest.main()
