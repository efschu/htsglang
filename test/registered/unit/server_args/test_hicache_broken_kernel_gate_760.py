"""#760: the hicache layout gate that keeps host write-back off a broken kernel.

WHAT IS UNDER TEST. ``page_first_direct`` sends the host write-back through
``transfer_kv_all_layer_direct_lf_pf -> cudaMemcpyBatchAsync``, which took three
SIGSEGVs on this build on 2026-08-18 with the transfer shapes proven matched.
``ServerArgs._gate_broken_host_transfer_kernels`` substitutes a layout that
converts nothing.

THE ORDERING IS THE POINT, not a detail. ``_resolve_layout_io_compatibility``
rewrites ``page_first`` + ``direct`` straight back into ``page_first_direct``,
and ``_resolve_storage_layout_compatibility`` rewrites ``layer_first`` back into
``page_first_direct`` for mooncake. A fallback that ran before either rule would
be silently undone and the boot would take the crashing route believing it had
avoided it. So the tests below assert on the layout AFTER the whole
``_handle_hicache`` pipeline, never on the gate in isolation.

Both directions of every gate, plus the default: with hicache off, nothing here
runs and no argument changes meaning.

    python -m pytest test/registered/unit/server_args/test_hicache_broken_kernel_gate_760.py -v
"""

import unittest
from unittest import mock

from sglang.srt.server_args import ServerArgs
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


def _resolved(cuda=True, **kwargs):
    """Run the full hicache normalization pipeline, not just the gate.

    ``model_path='dummy'`` short-circuits ``__post_init__`` (the repo-wide
    convention for argument tests), so ``_handle_hicache`` is invoked
    explicitly. ``is_cuda`` is patched because the gate is CUDA-scoped and CI
    hosts are not: the evidence is from CUDA kernels, so refusing on ROCm/NPU
    would gate a path this evidence says nothing about.
    """
    base = dict(enable_hierarchical_cache=True, hicache_storage_backend="file")
    base.update(kwargs)
    args = ServerArgs(model_path="dummy", **base)
    with mock.patch("sglang.srt.server_args.is_cuda", return_value=cuda):
        args._handle_hicache()
    return args


class TestHicacheBrokenKernelGate(CustomTestCase):
    def test_page_first_direct_falls_back_to_layer_first(self):
        args = _resolved(hicache_mem_layout="page_first_direct")
        self.assertEqual(args.hicache_mem_layout, "layer_first")
        # The io backend is deliberately NOT changed: layer_first is valid under
        # both backends, and MambaPoolHost accepts only 'direct'.
        self.assertEqual(args.hicache_io_backend, "direct")

    def test_page_first_plus_direct_is_gated_after_being_rewritten(self):
        """The trap that nearly wasted an isolation boot.

        ``page_first`` + ``direct`` never stays page_first: step 1 rewrites it
        to ``page_first_direct``. Asking for it must therefore end at the
        fallback, not at page_first.
        """
        args = _resolved(hicache_mem_layout="page_first", hicache_io_backend="direct")
        self.assertEqual(args.hicache_mem_layout, "layer_first")

    def test_page_first_direct_plus_kernel_is_gated_after_io_rewrite(self):
        """Step 1 turns this into page_first_direct + direct; the gate still fires."""
        args = _resolved(
            hicache_mem_layout="page_first_direct", hicache_io_backend="kernel"
        )
        self.assertEqual(args.hicache_mem_layout, "layer_first")
        self.assertEqual(args.hicache_io_backend, "direct")

    def test_override_keeps_the_crashing_route(self):
        args = _resolved(
            hicache_mem_layout="page_first_direct",
            hicache_allow_page_first_direct=True,
        )
        self.assertEqual(args.hicache_mem_layout, "page_first_direct")

    def test_not_cuda_is_left_alone(self):
        args = _resolved(cuda=False, hicache_mem_layout="page_first_direct")
        self.assertEqual(args.hicache_mem_layout, "page_first_direct")

    def test_mooncake_gets_a_layout_it_will_not_rewrite_back(self):
        """layer_first is unavailable here -- step 2 would undo it."""
        args = _resolved(
            hicache_mem_layout="page_first_direct",
            hicache_storage_backend="mooncake",
        )
        self.assertEqual(args.hicache_mem_layout, "page_first")
        self.assertEqual(args.hicache_io_backend, "kernel")

    def test_untouched_layouts_stay_untouched(self):
        for layout in ("layer_first", "page_head"):
            with self.subTest(layout=layout):
                args = _resolved(hicache_mem_layout=layout)
                self.assertEqual(args.hicache_mem_layout, layout)

    def test_hicache_off_changes_nothing(self):
        """The default path must not move. ``_handle_hicache`` returns early."""
        args = ServerArgs(
            model_path="dummy",
            enable_hierarchical_cache=False,
            hicache_mem_layout="page_first_direct",
            hicache_io_backend="direct",
        )
        with mock.patch("sglang.srt.server_args.is_cuda", return_value=True):
            args._handle_hicache()
        self.assertEqual(args.hicache_mem_layout, "page_first_direct")

    def test_gate_runs_last_in_the_pipeline(self):
        """Pin the ORDER, not just the outcome.

        A future edit that moves the gate ahead of the storage rule would still
        pass every outcome test above for the file backend, and would silently
        break mooncake. Assert the call sequence itself.
        """
        args = ServerArgs(model_path="dummy", enable_hierarchical_cache=True)
        calls = []
        for name in (
            "_resolve_layout_io_compatibility",
            "_resolve_storage_layout_compatibility",
            "_gate_broken_host_transfer_kernels",
        ):
            setattr(
                args, name, (lambda n=name: (lambda: calls.append(n)))()
            )
        args._handle_hicache()
        self.assertEqual(
            calls,
            [
                "_resolve_layout_io_compatibility",
                "_resolve_storage_layout_compatibility",
                "_gate_broken_host_transfer_kernels",
            ],
        )


class TestMambaPoolHostAcceptsLayerFirst(CustomTestCase):
    """The gate is useless if the mamba host pool refuses its fallback.

    ``MambaPoolHost`` used to accept only ``page_first_direct`` -- the one route
    that is broken -- so gating it in ServerArgs without this relaxation would
    have made a hybrid-mamba model unable to have a host tier at all.
    """

    def _layout_check(self, layout):
        """Drive only the layout guard, with no torch pools involved."""
        from sglang.srt.mem_cache.memory_pool_host import MambaPoolHost

        return MambaPoolHost.__init__(
            mock.Mock(), mock.Mock(), 2.0, 1, layout=layout
        )

    def test_page_first_is_still_refused(self):
        from sglang.srt.mem_cache.memory_pool_host import MambaPoolHost

        with self.assertRaises(ValueError) as ctx:
            self._layout_check("page_first")
        self.assertIn("staging", str(ctx.exception))
        self.assertIn("layer_first", str(ctx.exception))
        del MambaPoolHost

    def test_page_head_is_still_refused(self):
        with self.assertRaises(ValueError):
            self._layout_check("page_head")

    def test_layer_first_passes_the_layout_guard(self):
        """It must get PAST the guard. It then fails on the mocked device pool,
        which is a different exception and proves the guard let it through."""
        try:
            self._layout_check("layer_first")
        except ValueError as exc:
            self.assertNotIn("MambaPoolHost supports layout", str(exc))
        except Exception:
            pass


if __name__ == "__main__":
    unittest.main()
