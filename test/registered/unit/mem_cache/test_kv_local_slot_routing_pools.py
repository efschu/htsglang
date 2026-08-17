"""Local-slot routing in the pools DERIVED from `KVCache`.

`memory_pool.py` was converted first (see `test_kv_local_slot_routing.py`).
The subclasses kept their own inlined `layer_id - self.start_layer`, which is
the same silent-wrongness class: under `SGLANG_PP_LAYER_SET` the subtraction
returns a plausible index belonging to a DIFFERENT layer.

Three distinct outcomes are pinned here, because the honest answer differed
per file:

* **(a) routed** — `dsa_cache_layer_split.py`, `deepseek_v4_memory_pool.py`.
  Both chain `KVCache.__init__`, so the accessor exists, and every site indexes
  a per-layer buffer. These are local-slot semantics and now go through it.

* **(b) deliberately NOT routed** — `swa_memory_pool.py`. `SWAKVPool` pins
  `start_layer = 0` and never calls `super().__init__`, so the subtraction is
  an identity on a GLOBAL layer id and the accessor is never constructed.
  Converting it would have been the wrong conversion, so the reason is pinned
  instead: if someone later adds the `super().__init__` call, the pin fires and
  says to revisit.

* **(c) refused** — the INVERSE direction. `layer_shard_start` maps a local
  offset back to a global id by ADDING `start_layer`, and `prefill.py:170`
  turns it into `prefill_start_layer + len(kv_data_ptrs)`: a start+COUNT pair
  read as a contiguous global range. For the family plan's FA stage
  (start 35, count 8) that pair claims layers 36..42 the stage does not own and
  omits 47..63 that it does. That is a WIRE FORMAT limit, not an index
  translation, so it is not silently converted -- it is refused.
"""

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

import inspect
import io
import tokenize
import unittest

from sglang.srt.mem_cache.memory_pool import KVCache
from sglang.test.test_utils import CustomTestCase

#: The family plan's second FA stage: 8 layers, none adjacent, spanning 29.
FA_STAGE = [35, 39, 43, 47, 51, 55, 59, 63]


def _slot_map(owned):
    return {layer: slot for slot, layer in enumerate(sorted(owned))}


class _Carrier:
    """A minimal stand-in for a pool: `local_slot` reads exactly these two
    attributes, and carries the REAL method so the delegation under test is the
    shipped one rather than a copy."""

    local_slot = KVCache.local_slot

    def __init__(self, start_layer=0, owned=None):
        self.start_layer = start_layer
        self._local_slot_of = None if owned is None else _slot_map(owned)


def _code_only(module):
    """Source with comments and strings stripped.

    A routing pin must say "no site still COMPUTES this", not "nobody may
    mention it" -- otherwise the docstring explaining why the accessor exists
    would itself break the pin.
    """
    src = inspect.getsource(module)
    code = " ".join(
        tok.string
        for tok in tokenize.generate_tokens(io.StringIO(src).readline)
        if tok.type not in (tokenize.COMMENT, tokenize.STRING)
    )
    return "".join(code.split())


class TestTheAccessorReachesEveryDerivedPool(CustomTestCase):
    """The conversion is only safe if every converted class actually inherits
    the accessor AND runs the __init__ that builds its map."""

    def test_the_dsa_and_dsv4_pools_inherit_it(self):
        from sglang.srt.mem_cache.deepseek_v4_memory_pool import (
            DeepSeekV4IndexerPool,
            DeepSeekV4SingleKVPool,
            DeepSeekV4TokenToKVPool,
        )

        for cls in (
            DeepSeekV4SingleKVPool,
            DeepSeekV4IndexerPool,
            DeepSeekV4TokenToKVPool,
        ):
            with self.subTest(cls=cls.__name__):
                self.assertTrue(issubclass(cls, KVCache))
                self.assertIs(cls.local_slot, KVCache.local_slot)

    def test_every_converted_class_chains_the_base_init(self):
        """The map is built in `KVCache.__init__`. A class that skips it would
        raise AttributeError on the first routed call -- which is exactly why
        SWAKVPool is excluded below rather than converted."""
        from sglang.srt.mem_cache import deepseek_v4_memory_pool as dsv4

        for name in (
            "DeepSeekV4SingleKVPool",
            "DeepSeekV4IndexerPool",
            "DeepSeekV4TokenToKVPool",
        ):
            with self.subTest(cls=name):
                src = inspect.getsource(getattr(dsv4, name).__init__)
                self.assertIn("super().__init__", src)


class TestTheLayerSplitPoolRoutesThroughTheAccessor(CustomTestCase):
    """`_local_layer_idx` was the file's own name for the subtraction. It keeps
    the name and delegates, so the file's vocabulary survives and there is
    still exactly one rule."""

    def test_it_gives_the_rank_not_the_subtraction(self):
        from sglang.srt.mem_cache.dsa_cache_layer_split import (
            LayerSplitDSATokenToKVPool,
        )

        c = _Carrier(start_layer=3, owned=[3, 7, 11])
        idx = LayerSplitDSATokenToKVPool._local_layer_idx(c, 7)
        self.assertEqual(idx, 1)
        self.assertNotEqual(idx, 7 - 3)

    def test_it_is_still_the_subtraction_when_ownership_is_contiguous(self):
        from sglang.srt.mem_cache.dsa_cache_layer_split import (
            LayerSplitDSATokenToKVPool,
        )

        c = _Carrier(start_layer=22)
        for layer in range(22, 43):
            with self.subTest(layer=layer):
                self.assertEqual(
                    LayerSplitDSATokenToKVPool._local_layer_idx(c, layer), layer - 22
                )

    def test_the_fa_stage_maps_onto_dense_slots(self):
        from sglang.srt.mem_cache.dsa_cache_layer_split import (
            LayerSplitDSATokenToKVPool,
        )

        c = _Carrier(start_layer=min(FA_STAGE), owned=FA_STAGE)
        got = [LayerSplitDSATokenToKVPool._local_layer_idx(c, l) for l in FA_STAGE]
        self.assertEqual(got, list(range(8)))

    def test_ownership_checks_still_work_on_the_dense_index(self):
        """`_is_layer_owned` compares the local index against a CP shard RANGE.
        That composition survives only because the accessor returns a DENSE
        0..N-1 index -- pin it, since a sparse local index would break the CP
        layer-shard layered on top."""
        from sglang.srt.mem_cache.dsa_cache_layer_split import (
            LayerSplitDSATokenToKVPool,
        )

        c = _Carrier(start_layer=min(FA_STAGE), owned=FA_STAGE)
        got = sorted(
            LayerSplitDSATokenToKVPool._local_layer_idx(c, l) for l in FA_STAGE
        )
        self.assertEqual(got, list(range(len(FA_STAGE))))
        self.assertEqual(max(got), len(FA_STAGE) - 1)


class TestNoSubtractionSurvivesInTheRoutedFiles(CustomTestCase):
    """CODE ONLY -- see `_code_only`."""

    def test_layer_split_file_has_no_raw_subtraction(self):
        from sglang.srt.mem_cache import dsa_cache_layer_split

        self.assertEqual(
            _code_only(dsa_cache_layer_split).count("layer_id-self.start_layer"), 0
        )

    def test_dsv4_file_has_no_raw_subtraction(self):
        from sglang.srt.mem_cache import deepseek_v4_memory_pool

        self.assertEqual(
            _code_only(deepseek_v4_memory_pool).count("layer_id-self.start_layer"), 0
        )


class TestTheSwaPoolIsExcludedOnPurpose(CustomTestCase):
    """(b). The pin is on the REASON, not on the leftover line."""

    def test_swa_pool_pins_start_layer_to_zero(self):
        from sglang.srt.mem_cache.swa_memory_pool import SWAKVPool

        src = inspect.getsource(SWAKVPool.__init__)
        self.assertIn("self.start_layer = 0", src)

    def test_swa_pool_does_not_chain_the_base_init(self):
        """If this ever starts calling `super().__init__`, the accessor becomes
        available and the exclusion must be re-argued."""
        from sglang.srt.mem_cache.swa_memory_pool import SWAKVPool

        src = inspect.getsource(SWAKVPool.__init__)
        self.assertNotIn("super().__init__", src)


class TestTheInverseDirectionIsRefused(CustomTestCase):
    """(c). Local offset -> global id, for the PD transfer's start+count label."""

    def test_contiguous_ownership_still_gets_its_global_start(self):
        from sglang.srt.mem_cache.dsa_cache_layer_split import shard_start_global

        self.assertEqual(shard_start_global(22, 4, None), 26)

    def test_set_ownership_is_refused_not_silently_mislabelled(self):
        from sglang.srt.mem_cache.dsa_cache_layer_split import shard_start_global

        with self.assertRaises(NotImplementedError):
            shard_start_global(min(FA_STAGE), 4, _slot_map(FA_STAGE))

    def test_the_refusal_explains_the_wire_format_limit(self):
        from sglang.srt.mem_cache.dsa_cache_layer_split import shard_start_global

        with self.assertRaises(NotImplementedError) as cm:
            shard_start_global(min(FA_STAGE), 4, _slot_map(FA_STAGE))
        msg = str(cm.exception).lower()
        self.assertIn("contiguous", msg)
        self.assertIn("prefill_start_layer", msg)

    def test_what_the_silent_answer_would_have_been(self):
        """Documents the bug being prevented: start+count over the FA stage
        names 36..42 (not owned) and omits 47..63 (owned)."""
        start, count = min(FA_STAGE), len(FA_STAGE)
        claimed = list(range(start, start + count))
        self.assertEqual(claimed, [35, 36, 37, 38, 39, 40, 41, 42])
        self.assertTrue(set(claimed) - set(FA_STAGE))
        self.assertTrue(set(FA_STAGE) - set(claimed))


if __name__ == "__main__":
    unittest.main()
