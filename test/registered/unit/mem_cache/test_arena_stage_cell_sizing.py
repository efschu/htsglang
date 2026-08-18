"""The gapped boot's arena: a stage was sized for layers it does not own.

THE SPECIMEN. Gapped boot v6, ``boot_735_760gapped.log``, 2026-08-18 16:03:07Z,
PP=3 over one 32.6 GiB and two smaller cards, with

    SGLANG_PP_LAYER_SET=
      PP0  0-2,4-6,8-10,...,60-62        48 GDN layers, span [0, 63)
      PP1  3,7,11,15,19,23,27,31          8 full-attention layers
      PP2  35,39,43,47,51,55,59,63        8 full-attention layers

The model's 16 full-attention layers are 3, 7, ... 63, so PP0 owns NONE of
them. The boot logged, in this order:

    PP0  KV pool sizing: available_bytes=2847145984, cell_size=0
             -> max_total_num_tokens=1048576
    PP1  KV pool sizing: ... cell_size=16384 -> 845283
    PP2  KV pool sizing: ... cell_size=16384 -> 754019
    all  KV token sizing: min-reduced across ranks to 754019
    PP0  KvVmmArena ready: reserved=22.6 GiB
    PP1  KvVmmArena ready: reserved=12.0 GiB
    PP2  KvVmmArena ready: reserved=12.0 GiB
    PP0  cuMemCreate: 8388608 bytes refused by the driver ...
         torch reserved 49.70 GiB / allocated 49.66 GiB

THE VERDICT THAT WAS HANDED OVER -- "the arena is sized from the TOKEN COUNT,
NOT from cell_size" -- IS REFUTED BY ITS OWN NUMBERS, and the refutation is
what this file pins. All three stages sized from the SAME 754019-token
universe. The arena reserve is a pure function of the buffer spans:

    reserve = granularity + sum over buffers of
              (align_up(slots x row_bytes, granularity) + 32 MiB VA slack)

With 754020 slots, 1024 B rows and 2 MiB granularity that is 807,403,520 B a
buffer, so

    16 buffers ( 8 layers x k+v)  -> 12.03 GiB   == the logged 12.0 on PP1/PP2
    30 buffers (15 layers x k+v)  -> 22.56 GiB   == the logged 22.6 on PP0

PP0 was sized for FIFTEEN full-attention layers while owning zero. Fifteen,
not sixteen, is itself the signature: the interval test
``self.start_layer <= i < self.end_layer`` ran over PP0's SPAN [0, 63), which
contains 3, 7, ... 59 and excludes only 63. A token-count defect could not
produce a layer-count-shaped number.

So the token count was never the problem, and no value of the 1048576 sentinel
could have helped -- the sentinel is PP0's own local capacity, and the arena
was built after the min-reduce from the universe like everyone else.

WHY IT REACHED THE ALLOCATOR. The stage's own configurator had already
computed ``cell_size=0`` for PP0 -- it agreed PP0 carries no full-attention KV.
The pool disagreed by fifteen layers, and the pool is the one that allocates.
22.56 GiB on top of the 27.1 GiB the process already held is 49.70 GiB on a
32.6 GiB card, which is the logged number to the second decimal.

THE FIX is to resolve stage ownership as a SET, from the one derivation that
already exists (``get_pp_layer_set``), instead of from the interval -- exactly
what ``get_pp_indices``' own docstring says a consumer must not do, and what
``refuse_noncontiguous_layer_descriptor`` refuses for the transfer descriptor
in the same words ("names 29 layers of which 21 are not owned"). A stage that
owns zero full-attention layers is then a legitimate configuration: empty
list, no KV buffers, and an arena of ONE granularity page.

Hermetic: no CUDA, no process group, no model. Every number below is either
from the log above or computed by the function under test.
"""

import unittest

from sglang.srt.distributed.utils import stage_owned_layer_ids
from sglang.srt.mem_cache.kv_vmm_backing import arena_reserve_bytes
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=2, suite="base-a-test-cpu")

GIB = 1 << 30
MIB = 1 << 20

# -- the specimen, verbatim ------------------------------------------------
NUM_LAYERS = 64
FULL_ATTENTION_LAYER_IDS = list(range(3, 64, 4))  # 3, 7, ... 63 -- sixteen
PP0_OWNED = frozenset(
    i for i in range(0, 63) if i % 4 != 3
)  # 0-2,4-6,...,60-62 -- 48 GDN layers
PP1_OWNED = frozenset(range(3, 32, 4))  # 3,7,11,15,19,23,27,31
PP2_OWNED = frozenset(range(35, 64, 4))  # 35,39,...,63

# The interval each stage reports: min(owned), max(owned) + 1.
PP0_SPAN = (0, 63)
PP1_SPAN = (3, 32)
PP2_SPAN = (35, 64)

UNIVERSE_TOKENS = 754019  # min-reduced across ranks
PAGE_SIZE = 1
ROW_BYTES = 1024  # cell_size 16384 / 8 layers / 2 (k and v)
GRANULARITY = 2 * MIB  # "granularity=2048 KiB"

# The card and what the process already held when the arena was built.
PP0_CARD_GIB = 32.6
PP0_TORCH_RESERVED_AT_FAILURE_GIB = 49.70


def _arena_gib(num_full_attention_layers, tokens=UNIVERSE_TOKENS):
    """The reserve for a stage carrying ``num_full_attention_layers``."""
    slots = tokens + PAGE_SIZE
    spans = [slots * ROW_BYTES] * (2 * num_full_attention_layers)  # k and v
    return arena_reserve_bytes(spans, GRANULARITY) / GIB


class TestTheSpecimenArithmetic(CustomTestCase):
    """The logged GiB figures must be reproducible, or the diagnosis is a story."""

    def test_eight_layers_reproduces_the_pp1_pp2_reserve(self):
        self.assertAlmostEqual(_arena_gib(8), 12.03, places=2)

    def test_fifteen_layers_reproduces_the_pp0_reserve(self):
        """22.6 GiB is a LAYER-shaped number, which is how the token-count
        diagnosis is refuted: PP0 sized from the same 754019 universe as
        everyone else."""
        self.assertAlmostEqual(_arena_gib(15), 22.56, places=2)

    def test_the_token_count_was_never_the_variable(self):
        """All three stages, same universe; only the layer count differs."""
        self.assertAlmostEqual(_arena_gib(8) * 30 / 16, _arena_gib(15), places=1)

    def test_pp0s_own_sentinel_capacity_was_not_what_was_used(self):
        """If the arena had been sized from PP0's local 1048576 tokens the
        reserve would be a different number, so the sentinel was never the
        lever the six boots spent themselves on."""
        self.assertNotAlmostEqual(_arena_gib(15, tokens=1048576), 22.56, places=1)

    def test_zero_layers_is_one_granularity_page(self):
        """The floor. Not 22.6 GiB, and not a special case either -- it falls
        out of the same expression."""
        self.assertEqual(arena_reserve_bytes([], GRANULARITY), GRANULARITY)

    def test_the_granularity_must_be_positive(self):
        with self.assertRaises(ValueError):
            arena_reserve_bytes([1024], 0)


class TestTheOwnershipResolutionIsSetAware(CustomTestCase):
    """RED-FIRST: the interval is the SPAN, and PP0's span is the whole model."""

    def test_pp0_owns_no_full_attention_layers(self):
        got = stage_owned_layer_ids(
            FULL_ATTENTION_LAYER_IDS, *PP0_SPAN, owned=PP0_OWNED
        )
        self.assertEqual(got, [], f"PP0 was handed {len(got)} layers it does not own")

    def test_the_interval_would_have_claimed_fifteen(self):
        """CAN-FAIL: pin the defect itself, so a revert cannot pass quietly."""
        by_interval = [
            i for i in FULL_ATTENTION_LAYER_IDS if PP0_SPAN[0] <= i < PP0_SPAN[1]
        ]
        self.assertEqual(len(by_interval), 15)
        self.assertNotEqual(
            len(by_interval),
            len(PP0_OWNED & set(FULL_ATTENTION_LAYER_IDS)),
            "the interval and the set must disagree here, or the specimen is "
            "not reproduced",
        )

    def test_pp1_and_pp2_are_unchanged_by_the_fix(self):
        """They were already right: for them the span happens to contain only
        their own full-attention layers."""
        for owned, span in ((PP1_OWNED, PP1_SPAN), (PP2_OWNED, PP2_SPAN)):
            with self.subTest(span=span):
                by_set = stage_owned_layer_ids(
                    FULL_ATTENTION_LAYER_IDS, *span, owned=owned
                )
                by_interval = [
                    i for i in FULL_ATTENTION_LAYER_IDS if span[0] <= i < span[1]
                ]
                self.assertEqual(by_set, by_interval)
                self.assertEqual(len(by_set), 8)

    def test_a_contiguous_layout_is_byte_identical(self):
        """CAN-FAIL COUNTERWEIGHT, and the one that matters for every boot this
        rig has ever taken: with no layer set the resolution IS the interval."""
        for start, end in ((0, 22), (22, 43), (43, 64)):
            with self.subTest(stage=(start, end)):
                self.assertEqual(
                    stage_owned_layer_ids(
                        FULL_ATTENTION_LAYER_IDS, start, end, owned=None
                    ),
                    [i for i in FULL_ATTENTION_LAYER_IDS if start <= i < end],
                )

    def test_ownership_order_follows_the_layer_ids(self):
        got = stage_owned_layer_ids(
            FULL_ATTENTION_LAYER_IDS, *PP2_SPAN, owned=PP2_OWNED
        )
        self.assertEqual(got, sorted(got))
        self.assertEqual(got, [35, 39, 43, 47, 51, 55, 59, 63])


class TestTheShippedCallSiteUsesIt(CustomTestCase):
    """A helper nothing calls is the documented-but-inert class.

    Pinned by source, because standing up a ModelRunner to observe which
    expression it evaluated needs three GPUs and a model. This is the same
    shape as ``test_mamba_anchor_seams_747``'s call pins, and for the same
    reason: an import alone would satisfy a weaker check.
    """

    def test_the_hybrid_pool_resolves_ownership_set_aware(self):
        import inspect

        from sglang.srt.model_executor import model_runner_kv_cache_mixin as mixin

        src = inspect.getsource(mixin)
        marker = "full_attention_layer_ids=("
        idx = src.index(marker)
        window = src[idx : idx + 700]
        self.assertIn("stage_owned_layer_ids(", window)
        self.assertNotIn(
            "if self.start_layer <= i < self.end_layer",
            window,
            "the KV pool is resolving stage ownership from the INTERVAL again; "
            "under SGLANG_PP_LAYER_SET that is the span, not the set",
        )

    def test_the_owner_uses_the_extracted_reserve_formula(self):
        import inspect

        from sglang.srt.mem_cache.kv_vmm_backing import KvVmmBufferOwner

        src = inspect.getsource(KvVmmBufferOwner.__init__)
        self.assertIn("arena_reserve_bytes(", src)

    def test_one_derivation_of_ownership(self):
        """``memory_pool`` must not restate the lookup; two spellings of one
        rule is how the lineages drift."""
        import inspect

        from sglang.srt.mem_cache import memory_pool

        src = inspect.getsource(memory_pool._owned_layers_for_pool)
        self.assertIn("current_stage_layer_set", src)
        self.assertNotIn("get_pp_layer_set(", src)


class TestTheBootNowFits(CustomTestCase):
    """The acceptance the six windows were trying to reach."""

    def _pp0_arena_gib(self, owned):
        n = len(stage_owned_layer_ids(FULL_ATTENTION_LAYER_IDS, *PP0_SPAN, owned=owned))
        return _arena_gib(n)

    def test_the_old_resolution_does_not_fit_the_card(self):
        """The failure, reconstructed: 27.1 GiB already held plus the arena."""
        already_held = PP0_TORCH_RESERVED_AT_FAILURE_GIB - _arena_gib(15)
        self.assertGreater(already_held + self._pp0_arena_gib(owned=None), PP0_CARD_GIB)

    def test_the_set_aware_resolution_fits_the_card(self):
        already_held = PP0_TORCH_RESERVED_AT_FAILURE_GIB - _arena_gib(15)
        total = already_held + self._pp0_arena_gib(owned=PP0_OWNED)
        self.assertLess(
            total,
            PP0_CARD_GIB,
            f"PP0 still does not fit: {total:.2f} GiB on a {PP0_CARD_GIB} GiB card",
        )
        # And with room to spare, not by a hair.
        self.assertLess(total, PP0_CARD_GIB - 5.0)

    def test_the_whole_saving_is_the_arena(self):
        self.assertAlmostEqual(
            self._pp0_arena_gib(owned=None) - self._pp0_arena_gib(owned=PP0_OWNED),
            22.56,
            places=1,
        )


if __name__ == "__main__":
    unittest.main()
