"""Per-request flashinfer float-workspace zeroing (#50 root fix).

GPU round-11 bisection proof: zeroing exactly the wrappers'
_float_workspace_buffer after each finished request flattens the
request-ordinal output sequence at the natural run-1 value (int workspace /
kv_lens wipes do not). Mechanism: the fa2 split-KV kernels read workspace
regions the current forward did not write; on a fresh boot those read as
first-touch zeros (the contract the kernels were effectively validated
against), afterwards as the previous request's partials.

These tests pin the fix's semantics on CPU:
1. A miniature split-KV merge simulation shows the failure mode (request B's
   result depends on request A's residue) and that restoring the boot
   contract via zero_flashinfer_workspaces() makes request B bitwise equal
   to a fresh-boot request B.
2. Registry bookkeeping: registration is weak (no lifetime extension), the
   zero call covers every registered workspace and reports the count.
"""

import unittest

import torch

from sglang.srt.layers.attention import flashinfer_backend as fib
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=6, suite="base-a-test-cpu")


def _sim_forward(workspace: torch.Tensor, kv: torch.Tensor) -> torch.Tensor:
    """Miniature stand-in for a split-KV attention forward.

    Writes partials only for the CURRENT kv chunks (len(kv) slots), but the
    'merge' reads a fixed window of the persistent workspace — the exact
    read-beyond-write shape of the fa2 kernels. Stale slots contribute with
    a tiny weight, modeling the last-bit logit perturbation that flips
    greedy near-ties.
    """
    n = kv.shape[0]
    ws = workspace.view(torch.float32)
    ws[:n] = kv  # partials the current forward actually writes
    read_window = ws[: ws.shape[0]]  # merge reads the full window
    return kv.sum() + 1e-7 * read_window[n:].sum()


class TestWorkspaceResidueMechanism(CustomTestCase):
    def _make_ws(self):
        ws = torch.zeros(64 * 4, dtype=torch.uint8)  # boot: first-touch zeros
        fib.register_flashinfer_workspace_buffer(ws)
        return ws

    def test_residue_changes_next_request_without_fix(self):
        ws = self._make_ws()
        req_a = torch.full((48,), 3.0)
        req_b = torch.full((8,), 5.0)
        # Fresh-boot reference for request B.
        fresh_ref = _sim_forward(torch.zeros_like(ws), req_b)
        # Request A leaves residue; request B without the fix differs.
        _sim_forward(ws, req_a)
        dirty = _sim_forward(ws, req_b)
        self.assertFalse(
            torch.equal(dirty, fresh_ref),
            "simulation must exhibit the read-beyond-write coupling "
            "(otherwise this falsifier is dead)",
        )

    def test_zeroing_restores_boot_contract(self):
        ws = self._make_ws()
        req_a = torch.full((48,), 3.0)
        req_b = torch.full((8,), 5.0)
        fresh_ref = _sim_forward(torch.zeros_like(ws), req_b)
        _sim_forward(ws, req_a)  # request A finishes...
        fib.zero_flashinfer_workspaces()  # ...fix runs at request end
        fixed = _sim_forward(ws, req_b)
        self.assertTrue(torch.equal(fixed, fresh_ref))

    def test_repeated_requests_are_ordinal_independent_with_fix(self):
        ws = self._make_ws()
        req = torch.full((16,), 2.5)
        outs = []
        for _ in range(4):
            outs.append(_sim_forward(ws, req).clone())
            fib.zero_flashinfer_workspaces()
        for o in outs[1:]:
            self.assertTrue(torch.equal(o, outs[0]))


class TestWorkspaceRegistry(CustomTestCase):
    def test_zero_covers_all_registered_and_counts(self):
        a = torch.ones(16, dtype=torch.uint8)
        b = torch.ones(32, dtype=torch.uint8)
        fib.register_flashinfer_workspace_buffer(a)
        fib.register_flashinfer_workspace_buffer(b)
        n = fib.zero_flashinfer_workspaces()
        self.assertGreaterEqual(n, 2)  # other tests may have registered more
        self.assertTrue(torch.all(a == 0))
        self.assertTrue(torch.all(b == 0))

    def test_registration_is_weak(self):
        import gc

        before = len(fib._WORKSPACE_BUFFERS)
        t = torch.ones(8, dtype=torch.uint8)
        fib.register_flashinfer_workspace_buffer(t)
        self.assertEqual(len(fib._WORKSPACE_BUFFERS), before + 1)
        del t
        gc.collect()
        self.assertEqual(len(fib._WORKSPACE_BUFFERS), before)


if __name__ == "__main__":
    unittest.main()
