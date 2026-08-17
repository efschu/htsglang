"""#441(a): the page-head offset path has no 8-byte alignment invariant.

WHAT THE COPY HELPER REQUIRES. ``transfer_item_warp``
(sgl-kernel/csrc/kvcacheio/transfer.cu:20-37) moves bytes with 64-bit PTX:

    asm volatile("ld.global.nc.b64 %0,[%1];" ...)
    asm volatile("st.global.cg.b64 [%0],%1;" ...)

``.b64`` loads and stores require the address to be 8-byte aligned. A
misaligned address is not a slow path, it is a fault.

WHAT IS ACTUALLY CHECKED. One guard, in the launcher
(transfer.cu:285): ``TORCH_CHECK(item_size % 8 == 0)``.

WHY THAT GUARD COVERS THE lf_pf SIBLING AND NOT lf_ph. This is the
structural difference the ticket asks for:

  * ``get_global_offset_lf_tbl`` / ``get_global_offset_pf`` -- the pair
    ``transfer_kv_all_layer_lf_pf`` uses -- build offsets ONLY as multiples of
    ``item_size_bytes`` and of ``layout_dim`` (itself ``item_size * layers``).
    Every term therefore inherits the checked ``item_size % 8 == 0``, and
    alignment holds by construction.

  * ``get_global_offset_ph`` (transfer.cu:106-119) -- the destination
    ``transfer_kv_all_layer_lf_ph`` uses -- SUBDIVIDES by ``head_num`` in three
    of its four terms:

        page_dim / head_num * head_id * page_size
        page_id % page_size * page_dim / head_num
        layer_id * item_size_bytes / head_num

    and the kernel copies ``head_size_bytes = item_size_bytes / head_num``
    (transfer.cu:145) rather than a whole item. Nothing anywhere requires
    ``item_size / head_num`` to be a multiple of 8.

So there is a family of shapes that PASSES the only guard and still produces
misaligned 64-bit accesses. That is the missing invariant, and these pins
state it as arithmetic -- no CUDA context, no launch, no GPU.

HONEST SCOPE, because it would be easy to overclaim. This proves a real latent
defect and names it. It does NOT by itself explain the reported segfault of
``test_minimax_sparse_pool_host_unit``: that test's shapes (page_size=4,
float32, head_num=1, head_dim=2) give head_size_bytes=8, which is aligned, and
this file pins that fact too so the distinction cannot be lost. Attributing
that crash needs the GPU falsifier filed alongside this analysis.
"""

import unittest

# transfer.cu:20-23 -- the copy helper's unit is uint64_t.
PTX_ALIGNMENT = 8


def ph_offset(page_id, layer_id, head_id, *, page_size, page_dim, item_size, head_num):
    """``get_global_offset_ph`` (transfer.cu:106-119), transcribed exactly.

    Integer division throughout, matching C++ semantics on the non-negative
    values the kernel passes. Kept as one expression per source line so a
    reader can diff it against the kernel.
    """
    return (
        page_id // page_size * page_size * page_dim  # page_num dimension
        + page_dim // head_num * head_id * page_size  # head_num dimension
        + page_id % page_size * page_dim // head_num  # page_size dimension
        + layer_id * item_size // head_num  # layer_num dimension
    )


def pf_offset(page_id, layer_id, *, page_dim, item_size):
    """``get_global_offset_pf``: the sibling's destination, for contrast.

    No head subdivision anywhere -- which is exactly why it is safe.
    """
    return page_id * page_dim + layer_id * item_size


class Shape:
    def __init__(self, head_num, head_dim, elem_size, layers, page_size):
        self.head_num = head_num
        self.head_dim = head_dim
        self.elem_size = elem_size
        self.layers = layers
        self.page_size = page_size
        self.item_size = head_num * head_dim * elem_size
        self.page_dim = self.item_size * layers

    @property
    def passes_launcher_guard(self):
        """transfer.cu:285, the only alignment check that exists."""
        return self.item_size % PTX_ALIGNMENT == 0

    @property
    def head_size(self):
        """transfer.cu:145. What the page-head kernel actually copies."""
        return self.item_size // self.head_num

    def misaligned_ph(self):
        bad = []
        for page_id in range(self.page_size * 2):
            for layer_id in range(self.layers):
                for head_id in range(self.head_num):
                    off = ph_offset(
                        page_id,
                        layer_id,
                        head_id,
                        page_size=self.page_size,
                        page_dim=self.page_dim,
                        item_size=self.item_size,
                        head_num=self.head_num,
                    )
                    if off % PTX_ALIGNMENT:
                        bad.append((page_id, layer_id, head_id, off))
        return bad


# item_size = 8 passes the guard; head_size = 4 does not divide evenly into the
# 8-byte PTX unit. fp16 with head_dim=2 is the smallest such case.
FAULTING = Shape(head_num=2, head_dim=2, elem_size=2, layers=3, page_size=4)
# The shapes this rig actually serves.
RIG = Shape(head_num=4, head_dim=256, elem_size=1, layers=48, page_size=16)
# The shapes the reported-crashing test uses.
MINIMAX_TEST = Shape(head_num=1, head_dim=2, elem_size=4, layers=4, page_size=4)


class TestTheMissingInvariant(unittest.TestCase):
    def test_a_shape_can_pass_the_only_guard_and_still_misalign(self):
        """THE DEFECT, in one assertion pair."""
        self.assertTrue(
            FAULTING.passes_launcher_guard,
            "item_size % 8 == 0, so the launcher admits this shape",
        )
        bad = FAULTING.misaligned_ph()
        self.assertTrue(
            bad,
            "if this is empty the invariant is not missing after all",
        )
        page_id, layer_id, head_id, off = bad[0]
        self.assertNotEqual(off % PTX_ALIGNMENT, 0)

    def test_the_head_size_is_what_the_guard_fails_to_cover(self):
        self.assertEqual(FAULTING.item_size, 8)
        self.assertEqual(FAULTING.head_size, 4)
        self.assertNotEqual(
            FAULTING.head_size % PTX_ALIGNMENT,
            0,
            "head_size_bytes is the quantity the kernel copies and the "
            "quantity nothing validates",
        )

    def test_the_invariant_that_should_exist(self):
        """State it positively: head_size % 8 == 0 separates safe from unsafe.

        If a future guard is added, this is the predicate it should use --
        and this pin says so in a form that fails if the predicate is wrong.
        """
        for shape in (FAULTING, RIG, MINIMAX_TEST):
            with self.subTest(head_num=shape.head_num, head_dim=shape.head_dim):
                safe = shape.head_size % PTX_ALIGNMENT == 0
                self.assertEqual(
                    safe,
                    not shape.misaligned_ph(),
                    "head_size alignment must predict page-head offset "
                    "alignment exactly, or the proposed guard is not the "
                    "right one",
                )


class TestTheSiblingIsSafeByConstruction(unittest.TestCase):
    """Why lf_pf does not crash where lf_ph does."""

    def test_pf_offsets_are_always_aligned_when_the_guard_passes(self):
        for shape in (FAULTING, RIG, MINIMAX_TEST):
            with self.subTest(item=shape.item_size):
                if not shape.passes_launcher_guard:
                    continue
                for page_id in range(shape.page_size * 2):
                    for layer_id in range(shape.layers):
                        off = pf_offset(
                            page_id,
                            layer_id,
                            page_dim=shape.page_dim,
                            item_size=shape.item_size,
                        )
                        self.assertEqual(
                            off % PTX_ALIGNMENT,
                            0,
                            "the guard alone is sufficient for the pf path",
                        )


class TestWhatThisDoesNotExplain(unittest.TestCase):
    """Scope discipline, pinned so it cannot quietly widen.

    The reported crash is ``test_device_to_host_kernel_page_first`` in
    test_minimax_sparse_pool_host_unit, on BOTH wheels. Its shapes are
    ALIGNED, so the defect above is not its cause. Recording that here stops
    the next reader treating this analysis as a closed attribution.
    """

    def test_the_reported_crash_shapes_are_aligned(self):
        self.assertEqual(MINIMAX_TEST.item_size, 8)
        self.assertEqual(MINIMAX_TEST.head_size, 8)
        self.assertEqual(
            MINIMAX_TEST.misaligned_ph(),
            [],
            "these shapes do not misalign, so the alignment defect does not "
            "account for that test's segfault -- the GPU falsifier does",
        )

    def test_the_rig_shapes_are_aligned_too(self):
        """No production shape on this rig is exposed to the defect."""
        self.assertEqual(RIG.head_size, 256)
        self.assertEqual(RIG.misaligned_ph(), [])


if __name__ == "__main__":
    unittest.main()
