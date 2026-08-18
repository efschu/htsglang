"""#760: refuse a mis-shaped KV transfer at the seam, not inside the kernel.

DEFENSE IN DEPTH, not the root fix (F4-r5 owns that). This converts the crash
class from context-death inside a CUDA kernel into a named Python refusal that
fails one request and leaves the scheduler alive.

WHY A SECOND GUARD AT ALL -- the callsite census that motivates it:

    file                            transfer_kv_* callsites   guarded
    hisparse_memory_pool.py                     2                0
    memory_pool_host.py                        32                2
    pool_host/mha.py   <-- LIVE                28                0
    pool_host/mla.py                           12                0
    TOTAL                                      74                2

F4-r5's binding-staleness guard sits in ``DeepSeekV4PagedHostPool``. The live
boot instantiates ``MHATokenToKVPoolHost``
(``model_runner_kv_cache_mixin.py:2766``), which is a different class in a
different file with **zero** guard callsites. That is why it emitted zero
refusals: not because shapes matched, and not because a None tag disabled it,
but because it was never on the path that crashed.

The two guards check DIFFERENT invariants and both are wanted: F4-r5's asks
"is this binding stale?", this one asks "do these two pools even have the same
shape?".

THE SPECIMEN: a PP-shaped host pool (7/5/4 layers per stage) against a
TP-shaped device pool (16 layers). The pointer vectors are then 7 long and 16
long, and the kernel indexes the shorter one by the longer one's count.

OVERHEAD, stated rather than hand-waved: O(layers) integer comparisons on
METADATA only -- vector lengths, per-layer strides, and two index-bound checks
against pool sizes. No tensor element is read and nothing is copied, so the
matched-shape fast path is byte-identical.
"""

import unittest

from sglang.srt.mem_cache.kv_transfer_guard import (
    KvTransferShapeMismatch,
    validate_kv_transfer,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=1, suite="base-a-test-cpu")

#: The live layout: 64 layers, full attention every 4th, split 7/5/4 across
#: the three PP stages; the TP layout carries all 16 on one rank.
PP_STAGE_LAYERS = 7
TP_LAYERS = 16


def _ok(**over):
    base = dict(
        src_ptr_vectors=[list(range(TP_LAYERS)), list(range(TP_LAYERS))],
        dst_ptr_vectors=[list(range(TP_LAYERS)), list(range(TP_LAYERS))],
        src_indices=[0, 1, 2],
        dst_indices=[3, 4, 5],
        src_capacity=64,
        dst_capacity=64,
        src_item_size=2048,
        dst_item_size=2048,
        where="test",
    )
    base.update(over)
    return base


class TestTheSpecimenIsRefused(CustomTestCase):
    def test_a_PP_shaped_host_against_a_TP_shaped_device_refuses(self):
        """THE SPECIMEN. 7 layers vs 16 must never reach the kernel."""
        with self.assertRaises(KvTransferShapeMismatch) as ctx:
            validate_kv_transfer(
                **_ok(
                    dst_ptr_vectors=[
                        list(range(PP_STAGE_LAYERS)),
                        list(range(PP_STAGE_LAYERS)),
                    ]
                )
            )
        msg = str(ctx.exception)
        self.assertIn("7", msg)
        self.assertIn("16", msg)
        self.assertIn("#760", msg)

    def test_the_refusal_carries_both_shapes(self):
        with self.assertRaises(KvTransferShapeMismatch) as ctx:
            validate_kv_transfer(**_ok(dst_ptr_vectors=[[0], [0]]))
        self.assertIn("src", str(ctx.exception).lower())
        self.assertIn("dst", str(ctx.exception).lower())

    def test_a_ragged_vector_pair_is_refused(self):
        """k and v must agree with each other too, not only across pools."""
        with self.assertRaises(KvTransferShapeMismatch):
            validate_kv_transfer(
                **_ok(src_ptr_vectors=[list(range(16)), list(range(15))])
            )

    def test_a_per_layer_extent_mismatch_is_refused(self):
        """Same layer COUNT, different bytes per token: silently wrong output."""
        with self.assertRaises(KvTransferShapeMismatch) as ctx:
            validate_kv_transfer(**_ok(dst_item_size=1024))
        self.assertIn("item", str(ctx.exception).lower())

    def test_out_of_bounds_indices_are_refused_on_BOTH_pools(self):
        # Both sides keep the SAME index count, so the bounds check is what
        # fires -- an earlier version varied the count too and tripped the
        # count check instead, passing for the wrong reason.
        for kw in (
            {"src_indices": [0, 99], "dst_indices": [0, 1]},
            {"src_indices": [0, 1], "dst_indices": [0, 99]},
        ):
            with self.subTest(**kw):
                with self.assertRaises(KvTransferShapeMismatch) as ctx:
                    validate_kv_transfer(**_ok(src_capacity=10, dst_capacity=10, **kw))
                self.assertIn("bounds", str(ctx.exception).lower())

    def test_a_negative_index_is_refused(self):
        with self.assertRaises(KvTransferShapeMismatch):
            validate_kv_transfer(**_ok(src_indices=[-1, 0], dst_indices=[0, 1]))

    def test_mismatched_index_counts_are_refused(self):
        """One row per index, both sides -- a length skew mispairs every row."""
        with self.assertRaises(KvTransferShapeMismatch):
            validate_kv_transfer(**_ok(src_indices=[0, 1, 2], dst_indices=[0, 1]))


class TestTheFastPathIsUntouched(CustomTestCase):
    def test_CAN_FAIL_matched_shapes_pass_through(self):
        """If this ever raises, the guard has become the defect it prevents."""
        self.assertIsNone(validate_kv_transfer(**_ok()))

    def test_an_empty_transfer_is_allowed(self):
        """Zero indices is a no-op the callers already short-circuit."""
        self.assertIsNone(validate_kv_transfer(**_ok(src_indices=[], dst_indices=[])))

    def test_absent_metadata_does_not_invent_a_refusal(self):
        """Absence is not a mismatch.

        A caller that cannot supply a capacity must not be refused on that
        ground -- the guard would then fail closed on paths it does not
        understand, which is how a defense-in-depth layer becomes an outage.
        """
        self.assertIsNone(
            validate_kv_transfer(**_ok(src_capacity=None, dst_capacity=None))
        )

    def test_the_generation_stamps_are_reported_when_present(self):
        """#719 stamps ride along when the caller has them."""
        with self.assertRaises(KvTransferShapeMismatch) as ctx:
            validate_kv_transfer(
                **_ok(
                    dst_ptr_vectors=[[0], [0]],
                    src_generation=41,
                    dst_generation=42,
                )
            )
        self.assertIn("41", str(ctx.exception))
        self.assertIn("42", str(ctx.exception))


if __name__ == "__main__":
    unittest.main()
