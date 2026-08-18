"""#754: SGLANG_PP_LAYER_SET is a PP-stage concept -- the flip's TP stack
must not parse it.

THE CRASH (F4-r5 specimen): the phase flip builds its second, TP-only stack
with ``pp_size=1``, and every layer-set consumer re-reads the GLOBAL env
through ``get_pp_layer_set`` -- which parses "0-2,4-6;..." against
``pp_size=1`` and raises ``PPLayerSetError`` ("3 stage(s) given but pp_size
is 1"). Consequence: every layer-set boot ran without the flip, and
therefore without PP+spec.

THE FIX SHAPE, same discipline as the #752 skip: resolution is bound to the
SCOPE. Inside the flip's TP scope (``phase_flip_tp_routing_active()``, the
routing flag the TP-stack build already sets) the layer set is INAPPLICABLE
-- the TP stack owns every layer locally -- and ``get_pp_layer_set``
declaratively answers None instead of parsing. This covers every consumer
(model_runner, memory_pool, utils/common) through the one resolution
funnel, instead of masking the env at one call site.

Pinned in both directions: inside the scope the set resolves to None (the
contiguous default); outside the scope the SAME env still parses and
applies on the PP stack -- and a genuine pp_size=1 boot with the env set
still refuses, because that is a user error, not a flip.
"""

import os
import unittest

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=2, suite="base-a-test-cpu")

SET_ENV = "SGLANG_PP_LAYER_SET"
THREE_STAGE = "0-15;16-31;32-47"


class _FlipScope:
    """Context manager driving the REAL routing flag's module global --
    the setter refuses activation without initialized flip groups, which a
    hermetic test never has."""

    def __enter__(self):
        from sglang.srt.distributed import parallel_state as ps

        self._ps = ps
        self._saved = ps._PHASE_FLIP_TP_ACTIVE
        ps._PHASE_FLIP_TP_ACTIVE = True
        return self

    def __exit__(self, *exc):
        self._ps._PHASE_FLIP_TP_ACTIVE = self._saved


class TestLayerSetIsScopeBound(CustomTestCase):
    def setUp(self):
        self._saved_env = os.environ.get(SET_ENV)
        os.environ[SET_ENV] = THREE_STAGE
        self.addCleanup(self._restore)

    def _restore(self):
        if self._saved_env is None:
            os.environ.pop(SET_ENV, None)
        else:
            os.environ[SET_ENV] = self._saved_env

    def test_the_flip_tp_scope_resolves_to_none_instead_of_raising(self):
        """RED-FIRST, the specimen: pp_size=1 inside the flip scope with the
        three-stage env set. Before #754 this raises PPLayerSetError and
        kills the flip boot."""
        from sglang.srt.distributed.utils import get_pp_layer_set

        with _FlipScope():
            self.assertIsNone(get_pp_layer_set(48, 0, 1))

    def test_outside_the_scope_the_pp_stack_still_reads_the_set(self):
        """CAN-FAIL direction: the guard must not defang the PP stack.
        Mutating the scope check into a blanket skip reds this."""
        from sglang.srt.distributed.utils import get_pp_layer_set

        owned = get_pp_layer_set(48, 1, 3)
        self.assertEqual(owned, frozenset(range(16, 32)))

    def test_a_genuine_single_stage_misconfig_still_refuses(self):
        """pp_size=1 OUTSIDE the flip scope is a user error and must keep
        refusing loudly -- the scope, not the size, is what makes the set
        inapplicable."""
        from sglang.srt.distributed.utils import PPLayerSetError, get_pp_layer_set

        with self.assertRaises(PPLayerSetError):
            get_pp_layer_set(48, 0, 1)

    def test_inside_the_scope_an_unset_env_stays_none(self):
        from sglang.srt.distributed.utils import get_pp_layer_set

        os.environ.pop(SET_ENV, None)
        with _FlipScope():
            self.assertIsNone(get_pp_layer_set(48, 0, 1))

    def test_the_skip_is_documented_at_the_resolution_site(self):
        """Same rule as #752: a silent scope branch reads as an oversight
        and gets 'fixed' back."""
        import inspect

        from sglang.srt.distributed.utils import get_pp_layer_set

        src = inspect.getsource(get_pp_layer_set)
        self.assertIn("754", src)
        self.assertIn("flip", src.lower())


if __name__ == "__main__":
    unittest.main()
