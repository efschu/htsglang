"""The PD transfer descriptor cannot express non-contiguous layer ownership.

`shard_start_global` already refused this for the layer-shard path. That
refusal was incomplete: the PLAIN prefill/decode path builds the same kind of
descriptor from `token_to_kv_pool.start_layer` / `.end_layer` and never went
through it. Under a layer set those two are `min(owned)` and `max(owned) + 1`,
so the descriptor is the SPAN -- and the receiving side slices
`dst_kv_ptrs[start_layer:end_layer]` against it, which silently mismatches
buffers rather than failing.

Both paths now go through one rule, so the two cannot drift apart.
"""

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

import inspect
import unittest

from sglang.srt.distributed.utils import refuse_noncontiguous_layer_descriptor
from sglang.test.test_utils import CustomTestCase

FA_STAGE = [35, 39, 43, 47, 51, 55, 59, 63]


def _slot_map(owned):
    return {layer: slot for slot, layer in enumerate(sorted(owned))}


class TestTheSharedRefusal(CustomTestCase):
    def test_contiguous_ownership_passes(self):
        self.assertIsNone(refuse_noncontiguous_layer_descriptor(None, "test"))

    def test_a_layer_set_is_refused(self):
        with self.assertRaises(NotImplementedError):
            refuse_noncontiguous_layer_descriptor(_slot_map(FA_STAGE), "test")

    def test_the_message_names_the_set_and_the_caller(self):
        with self.assertRaises(NotImplementedError) as cm:
            refuse_noncontiguous_layer_descriptor(_slot_map(FA_STAGE), "prefill")
        msg = str(cm.exception)
        self.assertIn("prefill", msg)
        self.assertIn("35", msg)
        self.assertIn("63", msg)


class TestBothPathsUseTheOneRule(CustomTestCase):
    def test_the_layer_shard_path_delegates(self):
        from sglang.srt.mem_cache import dsa_cache_layer_split

        src = inspect.getsource(dsa_cache_layer_split.shard_start_global)
        self.assertIn("refuse_noncontiguous_layer_descriptor", src)

    def test_the_layer_shard_path_still_refuses(self):
        from sglang.srt.mem_cache.dsa_cache_layer_split import shard_start_global

        self.assertEqual(shard_start_global(22, 4, None), 26)
        with self.assertRaises(NotImplementedError):
            shard_start_global(35, 4, _slot_map(FA_STAGE))

    def test_the_plain_pd_path_is_guarded(self):
        """Source pin on the METHOD that builds the descriptor, not on the
        module: pinning the module passed even with the call deleted, because
        the import line alone satisfied it. A pin must state the invariant --
        "this function calls the rule" -- not count mentions of a name."""
        from sglang.srt.disaggregation.prefill import PrefillBootstrapQueue

        src = inspect.getsource(PrefillBootstrapQueue._init_kv_manager)
        self.assertIn("refuse_noncontiguous_layer_descriptor(", src)
        self.assertIn("prefill_start_layer", src)


if __name__ == "__main__":
    unittest.main()
