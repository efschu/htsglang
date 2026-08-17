"""The flashinfer full-attention index, end to end, under a gapped FA set.

The two helpers look like one site written twice. They are not, and §8.2 of
DESIGN_pp_layer_set.md records the journey:

  hop 1  `layer.layer_id` is a GLOBAL model layer id, always.
  hop 2  depends on which pool is being indexed:
         * `_wl_full_pool` is ALWAYS `token_to_kv_pool.full_kv_pool`, the
           SUB-pool, reached through `_transfer_full_attention_id` -> a DENSE
           full-attention index.
         * `_sess_full_pool` is `getattr(pool, "full_kv_pool", pool)`, so when
           there is no wrapper it falls back to the MAIN, globally indexed
           pool and there is no first hop at all.

The sub-pool branches were already correct, for a reason worth pinning rather
than trusting: `HybridLinearKVPool` builds its sub-pool with `layer_num=` and
no `start_layer=`, so `KVCache.__init__`'s `start_layer or 0` leaves it 0 and
the subtraction subtracts ZERO. The fallback branch is a genuine global->local
translation that hid behind a `getattr`.

What is real here: `_transfer_full_attention_id`, `_wl_full_layer_idx`,
`_sess_full_layer_idx` and `KVCache.local_slot` are the SHIPPED functions,
called unbound over minimal carriers. What is not: the KV tensors, since
building a real `HybridLinearKVPool` needs a `MambaPool` and real allocations.
The two facts that construction would have established -- the dense mapping and
the absent `start_layer` -- are pinned against the source instead.
"""

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

import inspect
import unittest

from sglang.srt.mem_cache.memory_pool import HybridLinearKVPool, KVCache
from sglang.test.test_utils import CustomTestCase

#: The family plan: full attention every 4th layer, 64 layers, this stage
#: owning the second half of them.
ALL_FA = list(range(3, 64, 4))
FA_STAGE = [35, 39, 43, 47, 51, 55, 59, 63]


class _Layer:
    def __init__(self, layer_id):
        self.layer_id = layer_id


class _Pool:
    """A pool that owns a (possibly gapped) set of GLOBAL layer ids."""

    local_slot = KVCache.local_slot

    def __init__(self, start_layer=0, owned=None):
        self.start_layer = start_layer
        self._local_slot_of = (
            None
            if owned is None
            else {layer: slot for slot, layer in enumerate(sorted(owned))}
        )


class _SubPool(_Pool):
    """A sub-pool: dense own frame, no global map (see `mark_as_sub_pool`)."""

    def __init__(self):
        super().__init__(start_layer=0, owned=None)


class _HybridPool(_SubPool):
    """Stands in for HybridLinearKVPool: re-indexes before delegating."""

    _transfer_full_attention_id = HybridLinearKVPool._transfer_full_attention_id

    def __init__(self, fa_layer_ids):
        super().__init__()
        # Mirrors memory_pool.py's construction; pinned below.
        self.full_attention_layer_id_mapping = {
            id: i for i, id in enumerate(fa_layer_ids)
        }
        self.full_kv_pool = _SubPool()


class _Backend:
    _wl_full_layer_idx = None  # bound below from the real class
    _sess_full_layer_idx = None


def _make_backend():
    from sglang.srt.layers.attention.flashinfer_backend import FlashInferAttnBackend

    class _B:
        _wl_full_layer_idx = FlashInferAttnBackend._wl_full_layer_idx
        _sess_full_layer_idx = FlashInferAttnBackend._sess_full_layer_idx

    return _B()


class TestTheSourceFactsTheTestRelieson(CustomTestCase):
    """Building a real hybrid pool needs a MambaPool, so the two construction
    facts are pinned against the source directly."""

    def test_the_mapping_is_dense_over_the_fa_layer_ids(self):
        src = inspect.getsource(HybridLinearKVPool.__init__)
        self.assertIn("full_attention_layer_id_mapping = {", src)
        self.assertIn("for i, id in enumerate(full_attention_layer_ids)", src)

    def test_the_sub_pool_is_built_without_a_start_layer(self):
        """If a start_layer= is ever passed here, the 'subtracts zero'
        reasoning in DESIGN §8.2 stops holding and this must be revisited."""
        src = inspect.getsource(HybridLinearKVPool.__init__)
        self.assertIn("layer_num=self.full_layer_nums", src)
        self.assertNotIn("start_layer=start_layer", src)

    def test_the_sub_pool_is_marked(self):
        src = inspect.getsource(HybridLinearKVPool.__init__)
        self.assertIn("mark_as_sub_pool(self.full_kv_pool)", src)


class TestTheSubPoolBranchUnderAGappedSet(CustomTestCase):
    def test_the_weightless_helper_returns_the_dense_index(self):
        b = _make_backend()
        b.token_to_kv_pool = _HybridPool(FA_STAGE)
        b._wl_full_pool = b.token_to_kv_pool.full_kv_pool
        got = [b._wl_full_layer_idx(_Layer(l)) for l in FA_STAGE]
        self.assertEqual(got, list(range(8)))

    def test_it_is_not_the_global_subtraction(self):
        """Layer 63 is slot 7, not 28."""
        b = _make_backend()
        b.token_to_kv_pool = _HybridPool(FA_STAGE)
        b._wl_full_pool = b.token_to_kv_pool.full_kv_pool
        self.assertEqual(b._wl_full_layer_idx(_Layer(63)), 7)
        self.assertNotEqual(b._wl_full_layer_idx(_Layer(63)), 63 - min(FA_STAGE))

    def test_an_unowned_full_attention_layer_is_refused(self):
        """A layer this stage does not own has no dense slot: the mapping
        itself refuses, before any arithmetic."""
        b = _make_backend()
        b.token_to_kv_pool = _HybridPool(FA_STAGE)
        b._wl_full_pool = b.token_to_kv_pool.full_kv_pool
        with self.assertRaises(ValueError):
            b._wl_full_layer_idx(_Layer(7))

    def test_the_session_helper_agrees_on_the_wrapper_branch(self):
        b = _make_backend()
        b._sess_pool = _HybridPool(FA_STAGE)
        b._sess_full_pool = b._sess_pool.full_kv_pool
        got = [b._sess_full_layer_idx(_Layer(l)) for l in FA_STAGE]
        self.assertEqual(got, list(range(8)))

    def test_the_contiguous_case_is_unchanged(self):
        """All 16 FA layers on one stage: dense index == mapping position."""
        b = _make_backend()
        b.token_to_kv_pool = _HybridPool(ALL_FA)
        b._wl_full_pool = b.token_to_kv_pool.full_kv_pool
        for i, l in enumerate(ALL_FA):
            with self.subTest(layer=l):
                self.assertEqual(b._wl_full_layer_idx(_Layer(l)), i)


class TestTheFallbackBranchIsAGlobalTranslation(CustomTestCase):
    """No wrapper: `_sess_full_pool` IS the main pool, so the incoming id is
    global and the answer must be the rank within the owned set."""

    def test_it_uses_the_rank_not_the_span_subtraction(self):
        b = _make_backend()
        pool = _Pool(start_layer=min(FA_STAGE), owned=FA_STAGE)
        b._sess_pool = pool
        b._sess_full_pool = pool
        self.assertEqual(b._sess_full_layer_idx(_Layer(43)), 2)
        self.assertNotEqual(b._sess_full_layer_idx(_Layer(43)), 43 - 35)

    def test_the_whole_stage_maps_onto_dense_slots(self):
        b = _make_backend()
        pool = _Pool(start_layer=min(FA_STAGE), owned=FA_STAGE)
        b._sess_pool = pool
        b._sess_full_pool = pool
        got = [b._sess_full_layer_idx(_Layer(l)) for l in FA_STAGE]
        self.assertEqual(got, list(range(8)))

    def test_contiguous_ownership_is_byte_identical(self):
        b = _make_backend()
        pool = _Pool(start_layer=22)
        b._sess_pool = pool
        b._sess_full_pool = pool
        for l in range(22, 40):
            with self.subTest(layer=l):
                self.assertEqual(b._sess_full_layer_idx(_Layer(l)), l - 22)

    def test_an_unowned_layer_is_refused_not_answered(self):
        b = _make_backend()
        pool = _Pool(start_layer=min(FA_STAGE), owned=FA_STAGE)
        b._sess_pool = pool
        b._sess_full_pool = pool
        with self.assertRaises(KeyError):
            b._sess_full_layer_idx(_Layer(36))


class TestNoRawSubtractionSurvives(CustomTestCase):
    def test_both_helpers_route_through_the_accessor(self):
        from sglang.srt.layers.attention.flashinfer_backend import FlashInferAttnBackend

        for name in ("_wl_full_layer_idx", "_sess_full_layer_idx"):
            with self.subTest(helper=name):
                src = inspect.getsource(getattr(FlashInferAttnBackend, name))
                self.assertIn("local_slot(", src)
                self.assertNotIn('getattr(self._wl_full_pool, "start_layer", 0)', src)
                self.assertNotIn('getattr(self._sess_full_pool, "start_layer", 0)', src)


if __name__ == "__main__":
    unittest.main()
