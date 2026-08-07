# SPDX-License-Identifier: Apache-2.0
"""``HybridMambaDecodeReqToTokenPool`` must establish what it inherits (#631).

The pool subclasses ``HybridReqToTokenPool`` but deliberately does NOT call its
``__init__`` -- it takes ``DecodeReqToTokenPool``'s sizing instead and then
re-establishes the hybrid attributes by hand. That is a legitimate design and
also a standing hazard: the hand-written list is a COPY of the base's, and a
copy drifts. It did. ``self.tree_cache = None`` was missing, and the reader is
the base's own inherited allocator (``_alloc_mamba_slots_or_evict``,
``memory_pool.py:1450-1469``), which branches on ``self.tree_cache is not
None`` for every mamba slot allocation.

The cost of that shape of defect is why this test exists rather than a fix
alone: nothing failed at boot. Both PD arms loaded weights, captured graphs,
warmed sampling and reported healthy; the AttributeError arrived on the FIRST
preallocated request, i.e. at the first real handover, in whatever the first
traffic happened to be.

So the test does not pin one attribute name. It pins the RULE -- every
instance attribute the base's ``__init__`` establishes is also established by
the subclass -- which is the invariant the copy is supposed to maintain and
the only form of this test that catches the NEXT omission rather than the last
one. The construction is stubbed down to the two ``__init__`` bodies because
that is the whole surface under test; a real pool needs CUDA.
"""

import unittest
from unittest import mock

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=10, suite="base-a-test-cpu")


def _attrs_assigned_by(init_func) -> set:
    """Instance attribute names assigned in a function body, by AST."""
    import ast
    import inspect
    import textwrap

    tree = ast.parse(textwrap.dedent(inspect.getsource(init_func)))
    names = set()
    for node in ast.walk(tree):
        targets = []
        if isinstance(node, ast.Assign):
            targets = node.targets
        elif isinstance(node, ast.AnnAssign):
            targets = [node.target]
        for t in targets:
            if (
                isinstance(t, ast.Attribute)
                and isinstance(t.value, ast.Name)
                and t.value.id == "self"
            ):
                names.add(t.attr)
    return names


class HybridDecodePoolAttributeContractTest(CustomTestCase):
    def test_subclass_establishes_every_base_attribute(self):
        from sglang.srt.disaggregation.decode import HybridMambaDecodeReqToTokenPool
        from sglang.srt.mem_cache.memory_pool import HybridReqToTokenPool

        base = _attrs_assigned_by(HybridReqToTokenPool.__init__)
        sub = _attrs_assigned_by(HybridMambaDecodeReqToTokenPool.__init__)

        # Attributes the subclass legitimately leaves to DecodeReqToTokenPool
        # (it calls that base's __init__ explicitly, so those ARE established).
        from sglang.srt.disaggregation.decode import DecodeReqToTokenPool

        via_decode_base = _attrs_assigned_by(DecodeReqToTokenPool.__init__)

        missing = base - sub - via_decode_base
        self.assertEqual(
            missing,
            set(),
            "HybridMambaDecodeReqToTokenPool.__init__ does not establish "
            f"{sorted(missing)}, which HybridReqToTokenPool.__init__ does. "
            "The subclass skips that base's __init__, so every attribute it "
            "declares must be re-established here or an inherited method will "
            "read a missing attribute at runtime -- which is how tree_cache "
            "reached production and failed on the first PD handover.",
        )

    def test_tree_cache_specifically_is_present_and_none(self):
        """The instance that actually failed, pinned by name and value.

        None is the CORRECT initial value rather than a placeholder: the radix
        cache binds itself later through bind_tree_cache, and a PD decode arm
        launched without --disaggregation-decode-enable-radix-cache never binds
        one, which the allocator's None branch is written to handle.
        """
        from sglang.srt.disaggregation.decode import HybridMambaDecodeReqToTokenPool

        pool = HybridMambaDecodeReqToTokenPool.__new__(HybridMambaDecodeReqToTokenPool)
        with mock.patch.object(
            HybridMambaDecodeReqToTokenPool, "_init_mamba_pool", return_value=None
        ), mock.patch(
            "sglang.srt.disaggregation.decode.DecodeReqToTokenPool.__init__",
            return_value=None,
        ):
            HybridMambaDecodeReqToTokenPool.__init__(
                pool,
                size=4,
                max_context_len=128,
                device="cpu",
                enable_memory_saver=False,
                cache_params=mock.MagicMock(),
                mamba_layer_ids=[0],
                speculative_num_draft_tokens=0,
                enable_mamba_extra_buffer=False,
                pre_alloc_size=1,
                enable_overlap_schedule=False,
            )

        self.assertTrue(
            hasattr(pool, "tree_cache"),
            "tree_cache is read by the inherited _alloc_mamba_slots_or_evict "
            "on every mamba slot allocation",
        )
        self.assertIsNone(pool.tree_cache)


if __name__ == "__main__":
    unittest.main()
