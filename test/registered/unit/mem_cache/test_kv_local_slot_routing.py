"""KV buffer indexing under non-contiguous layer ownership.

Every per-layer buffer in `memory_pool.py` is indexed by a stage-LOCAL slot,
and the translation from a global layer id was written inline as
`layer_id - self.start_layer` in 64 places. That subtraction is correct only
while a stage owns a contiguous RANGE.

Under `SGLANG_PP_LAYER_SET` a stage owns a set — for the family plan, the
full-attention layers `{3, 7, 11, …}` of a 64-layer hybrid — and then layer 7's
local slot is **1** while the subtraction says **4**.

**The failure mode is why this is worth a suite of its own.** Reading slot 4 of
an 8-slot buffer does not crash and does not warn. It returns another layer's
KV, confidently, for exactly one layer. That is the silent-wrongness class, and
an off-by-N index is the hardest possible version of it to notice downstream.

The tests below therefore always use a set with a GAP BEFORE the tested layer,
so subtraction and rank-lookup give different answers and a test that passed by
accident is impossible.
"""

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

import inspect
import unittest

from sglang.srt.mem_cache.memory_pool import KVCache
from sglang.test.test_utils import CustomTestCase

#: `local_slot` reads exactly these two attributes, so a minimal carrier
#: exercises the real function without constructing an abstract pool.
local_slot = KVCache.local_slot


class _Carrier:
    def __init__(self, start_layer=0, owned=None):
        self.start_layer = start_layer
        self._local_slot_of = (
            None
            if owned is None
            else {layer: slot for slot, layer in enumerate(sorted(owned))}
        )


#: The family plan's second FA stage: 8 layers, none adjacent, spanning 29.
FA_STAGE = [35, 39, 43, 47, 51, 55, 59, 63]


class TestTheContiguousPathIsUnchanged(CustomTestCase):
    """Byte-identical degeneracy. The accessor must BE the subtraction when
    ownership is contiguous, or this change is a behaviour change in disguise."""

    def test_it_is_exactly_the_subtraction_from_zero(self):
        c = _Carrier(start_layer=0)
        for layer in range(64):
            with self.subTest(layer=layer):
                self.assertEqual(local_slot(c, layer), layer - 0)

    def test_it_is_exactly_the_subtraction_from_an_offset(self):
        c = _Carrier(start_layer=22)
        for layer in range(22, 43):
            with self.subTest(layer=layer):
                self.assertEqual(local_slot(c, layer), layer - 22)


class TestTheDivergentCase(CustomTestCase):
    """A gap BEFORE the tested layer, so the two rules disagree."""

    def test_rank_lookup_not_subtraction(self):
        c = _Carrier(start_layer=3, owned=[3, 7, 11])
        self.assertEqual(local_slot(c, 7), 1)
        self.assertNotEqual(local_slot(c, 7), 7 - 3)

    def test_the_family_plan_fa_stage_maps_to_dense_slots(self):
        """8 owned layers spanning 29 must occupy slots 0..7 — the pool has 8
        of them, and the subtraction would index up to 28."""
        c = _Carrier(start_layer=min(FA_STAGE), owned=FA_STAGE)
        self.assertEqual([local_slot(c, l) for l in FA_STAGE], list(range(8)))

    def test_the_last_owned_layer_is_the_last_slot_not_the_span(self):
        c = _Carrier(start_layer=min(FA_STAGE), owned=FA_STAGE)
        self.assertEqual(local_slot(c, 63), 7)
        self.assertEqual(63 - min(FA_STAGE), 28)  # what subtraction would say

    def test_slots_are_dense_and_unique(self):
        c = _Carrier(start_layer=min(FA_STAGE), owned=FA_STAGE)
        slots = [local_slot(c, l) for l in FA_STAGE]
        self.assertEqual(sorted(slots), list(range(len(FA_STAGE))))


class TestAnUnownedLayerIsRefused(CustomTestCase):
    """The subtraction's real defect was not that it was wrong — it was that it
    ANSWERED. A layer this stage does not own has no slot, and saying so is the
    whole improvement."""

    def test_it_raises_rather_than_returning_a_plausible_index(self):
        c = _Carrier(start_layer=3, owned=[3, 7, 11])
        with self.assertRaises(KeyError):
            local_slot(c, 5)

    def test_the_refusal_names_the_layer_and_what_is_owned(self):
        c = _Carrier(start_layer=3, owned=[3, 7, 11])
        with self.assertRaises(KeyError) as cm:
            local_slot(c, 5)
        msg = str(cm.exception)
        self.assertIn("5", msg)
        self.assertIn("[3, 7, 11]", msg)

    def test_the_contiguous_path_still_answers_freely(self):
        """The falsifier for the refusal: it must be specific to set ownership,
        or every contiguous deployment would start raising."""
        c = _Carrier(start_layer=0)
        self.assertEqual(local_slot(c, 999), 999)


class TestEverySiteWasRouted(CustomTestCase):
    """The conversion, pinned. 64 inlined subtractions became one accessor; a
    single survivor would be the one layer that is silently wrong."""

    def test_no_raw_subtraction_survives_in_the_pool(self):
        """CODE ONLY. The phrase also appears in the comment that explains why
        the accessor exists, and forbidding the EXPLANATION would punish the
        documentation for being specific. Stripping comments and strings makes
        the pin say what it means -- no site still COMPUTES the subtraction --
        and makes it stronger, since a call cannot hide in a docstring."""
        import io
        import tokenize

        from sglang.srt.mem_cache import memory_pool

        src = inspect.getsource(memory_pool)
        code = " ".join(
            tok.string
            for tok in tokenize.generate_tokens(io.StringIO(src).readline)
            if tok.type not in (tokenize.COMMENT, tokenize.STRING)
        )
        squeezed = "".join(code.split())
        # The accessor's own body is the sole legitimate computation.
        self.assertEqual(squeezed.count("layer_id-self.start_layer"), 1)

    def test_the_survivor_is_the_accessor_itself(self):
        src = inspect.getsource(KVCache.local_slot)
        self.assertIn("layer_id - self.start_layer", src)

    def test_the_accessor_is_reachable_from_the_pool_base(self):
        self.assertTrue(callable(KVCache.local_slot))


if __name__ == "__main__":
    unittest.main()
