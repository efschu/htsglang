"""#614 (c): the HiCache host-pool composition gate refuses BY NAME.

Hermetic: no pool, no card, no model. ``_select_strategy`` takes a kvcache
object and a component set and returns a builder, so a stand-in object and a
set of enum members are the whole fixture.

WHAT IS ACTUALLY NOT SUPPORTED, AND WHY. ``is_hybrid_ssm`` and ``is_hybrid_swa``
are derived independently (``mem_cache/kv_cache_builder.py:154-163``), so a
checkpoint that is hybrid on BOTH axes -- linear/GDN state AND sliding-window
attention -- yields ``tree_components = (FULL, SWA, MAMBA)``
(``mem_cache/registry.py:176-181``). Every built-in strategy matches on set
EQUALITY: ``_MambaStrategy`` needs exactly ``{FULL, MAMBA}``, ``_SwaStrategy``
exactly ``{FULL, SWA}``, ``_DeepSeekV4Strategy`` exactly ``{FULL, SWA}``, and
``_PlainKvStrategy`` declines ``HybridLinearKVPool`` and ``SWAKVPool``
outright. So the three-component set has no builder. That is the citation this
file pins -- ``test_gdn_plus_swa_has_no_builtin_builder`` fails the moment one
of those predicates is widened, which is exactly when the refusal must go. That is the code path proving the gap.

NOT A NEW BLOCK ON A WORKING COMBINATION. The gate already refused this set; it
refused with a bare ``AssertionError`` naming two Python identifiers and
neither the flag responsible nor the way out. These tests pin the refusal's
CLASS and its CONTENT, so a supported composition is untouched -- see
``test_supported_compositions_still_resolve``.

CAN-FAIL PROOF: revert ``_select_strategy``'s ``raise ValueError(...)`` to the
old ``raise AssertionError(...)`` -> ``test_refusal_is_a_value_error`` and
``test_refusal_names_the_flag`` go red. Observed red before restoring.
"""

import unittest

from sglang.srt.mem_cache.hybrid_cache.hybrid_pool_assembler import (
    _STRATEGIES,
    _select_strategy,
    unsupported_composition_message,
)
from sglang.srt.mem_cache.unified_cache_components import ComponentType
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=2, suite="base-a-test-cpu")

GDN_PLUS_SWA = {ComponentType.FULL, ComponentType.SWA, ComponentType.MAMBA}


class _FakeHybridLinearKVPool:
    """Stands in for the real pool. The predicates that decide this case are
    ``isinstance`` checks against pool classes AND a set equality on the
    components; the set equality alone already excludes every strategy, so a
    stand-in is enough to reach the refusal and cannot accidentally satisfy a
    builder."""

    pass


class TestUnsupportedComposition(CustomTestCase):
    def test_gdn_plus_swa_has_no_builtin_builder(self):
        """The gap itself, read out of the predicates rather than
        asserted. If this ever passes, the refusal below is wrong and must go.
        """
        kvcache = _FakeHybridLinearKVPool()
        matched = [s for s in _STRATEGIES if s.matches(kvcache, GDN_PLUS_SWA)]
        self.assertEqual(matched, [], "a builder now covers GDN+SWA")

    def test_refusal_is_a_value_error(self):
        """A user configuration is refused with ValueError. AssertionError
        reads as "this build is broken" to every reader and every handler."""
        with self.assertRaises(ValueError):
            _select_strategy(_FakeHybridLinearKVPool(), GDN_PLUS_SWA)

    def test_refusal_is_not_an_assertion_error(self):
        try:
            _select_strategy(_FakeHybridLinearKVPool(), GDN_PLUS_SWA)
        except ValueError as exc:
            self.assertNotIsInstance(exc, AssertionError)
        else:
            self.fail("no refusal raised")

    def test_refusal_names_the_flag(self):
        """A refusal that does not say which flag to remove sends the reader
        into the pool assembler to find out."""
        with self.assertRaises(ValueError) as ctx:
            _select_strategy(_FakeHybridLinearKVPool(), GDN_PLUS_SWA)
        msg = str(ctx.exception)
        self.assertIn("--enable-hierarchical-cache", msg)
        self.assertIn("Drop --enable-hierarchical-cache", msg)

    def test_refusal_names_the_composition(self):
        with self.assertRaises(ValueError) as ctx:
            _select_strategy(_FakeHybridLinearKVPool(), GDN_PLUS_SWA)
        msg = str(ctx.exception)
        self.assertIn("_FakeHybridLinearKVPool", msg)
        for name in ("FULL", "MAMBA", "SWA"):
            self.assertIn(name, msg)

    def test_gdn_plus_swa_gets_the_specific_sentence(self):
        """The one composition a model can walk into without asking for
        anything exotic is called out by name, not left to the generic line."""
        msg = unsupported_composition_message(
            "HybridLinearKVPool", ["FULL", "MAMBA", "SWA"]
        )
        self.assertIn("hybrid on BOTH axes", msg)
        self.assertIn("sliding-window", msg)
        self.assertIn("silently never be backed", msg)

    def test_generic_composition_omits_the_gdn_swa_sentence(self):
        """The specific sentence must not be pasted onto every refusal, or it
        stops carrying information."""
        msg = unsupported_composition_message("SomePool", ["FULL", "INDEXER"])
        self.assertNotIn("hybrid on BOTH axes", msg)
        self.assertIn("--enable-hierarchical-cache", msg)

    def test_refusal_names_the_extension_point(self):
        """A downstream fork CAN support this; the message says how, so the
        refusal is a statement about the built-ins and not about the world."""
        msg = unsupported_composition_message("SomePool", ["FULL", "INDEXER"])
        self.assertIn("register_stack_strategy", msg)

    def test_supported_compositions_still_resolve(self):
        """The guard must not block combinations that work. A plain (non-
        hybrid) pool with only the FULL component is the ordinary case and
        still finds its builder."""

        class _PlainPool:
            pass

        strategy = _select_strategy(_PlainPool(), {ComponentType.FULL})
        self.assertEqual(type(strategy).__name__, "_PlainKvStrategy")


if __name__ == "__main__":
    unittest.main()
