"""#773: the floor may only take a reduction the BUILT cache lineage delivers.

THE COMPOSITION DEFECT. `#755` reduces the mamba floor from `1 + P + 1 + 1`
to `1 + P + 1` -- the donated slot BECOMES the next pin, so the two terms
share one slot instead of two. That is sound, and its runtime half is
implemented, gated per node, and tested. Its config gate
(`mamba_pool_floor.mamba_slot_reorder_active`) requires
`enable_hierarchical_cache`, because only a write-through host tier can
promise the released anchor still exists.

But `registry.py` routes a hybrid-SSM model WITH hierarchical cache to
`UnifiedRadixCache` (`:107-111`), and `MambaRadixCache` -- the only class
that implements the reorder -- is reachable only at `:133`, i.e. only when
hierarchical cache is OFF. So the two halves select for opposite worlds:

    hierarchical=False -> reduction NOT taken (floor 1+P+1+1)
                          class built: MambaRadixCache   (HAS the mechanism)
    hierarchical=True  -> reduction     TAKEN (floor 1+P+1)
                          class built: UnifiedRadixCache (has NO mechanism)

Inverted in both rows. `CacheInitParams.mamba_slot_reorder` is filled on
every boot from that same predicate (`kv_cache_builder.py`) and is read only
by `MambaRadixCache`, so it is always False where it is read and always
ignored where it is True.

The direction of the error is the dangerous one. A floor that is too HIGH
wastes VRAM; a floor that is too LOW is #581 -- the boot validates a pool the
runtime then over-draws, and the shortfall surfaces as a late failure after
minutes of serving, which is exactly what the validate-early rule exists to
prevent. `mamba_pool_floor.py`'s own docstring says a term may only be
dropped when "EVERY path under this config stays within the reduced budget";
here no path under this config does, because the code that would is not
built.

These tests are pure predicate/registry-shape tests: no pools, no GPU.
"""

import os
import unittest
from unittest import mock

import sglang.srt.mem_cache.mamba_pool_floor as floor_mod

from sglang.srt.mem_cache.mamba_pool_floor import (
    mamba_hard_floor,
    mamba_reorder_lineage_supported,
    mamba_slot_reorder_active,
    mamba_slots_per_running_req,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5)

REORDER_ENV = "SGLANG_MAMBA_SLOT_REORDER"


class _Args:
    """The standing boot's shape, minus the axis under test."""

    disable_radix_cache = False
    hicache_write_policy = "write_through"
    disable_overlap_schedule = True
    mamba_radix_cache_strategy = "no_buffer"
    enable_hierarchical_cache = True
    is_hybrid_ssm = True

    def __init__(self, **kw):
        for k, v in kw.items():
            setattr(self, k, v)

    def enable_mamba_extra_buffer(self) -> bool:
        return (
            self.disable_radix_cache is False
            and self.mamba_radix_cache_strategy
            in (
                "extra_buffer",
                "extra_buffer_lazy",
            )
        )


def _lineage_supported():
    """Pretend the built lineage carries the reorder.

    Patches the module-level predicate rather than planting a hook on
    ServerArgs: an operator must never be able to assert a code capability.
    """
    return mock.patch.object(
        floor_mod, "mamba_reorder_lineage_supported", lambda _sa: True
    )


class _EnvOn:
    def __enter__(self):
        self._prev = os.environ.get(REORDER_ENV)
        os.environ[REORDER_ENV] = "1"
        return self

    def __exit__(self, *exc):
        if self._prev is None:
            os.environ.pop(REORDER_ENV, None)
        else:
            os.environ[REORDER_ENV] = self._prev


class TestTheReductionRequiresTheMechanism(CustomTestCase):
    def test_the_unified_lineage_does_not_support_the_reorder(self):
        """The lineage predicate states the fact the floor has to respect."""
        with _EnvOn():
            self.assertFalse(
                mamba_reorder_lineage_supported(_Args()),
                "hybrid SSM + hierarchical cache is built as UnifiedRadixCache, "
                "which does not implement the #755 reorder",
            )

    def test_the_standing_boot_shape_does_not_take_the_reduction(self):
        """THE REGRESSION THIS FILE EXISTS FOR.

        Every other condition of the #755 gate is satisfied on the standing
        boot -- radix on, hierarchical on, write_through, operator opted in --
        and the reduction must still be refused, because the class that would
        honour it is not the class that gets built.
        """
        with _EnvOn():
            args = _Args()
            self.assertFalse(mamba_slot_reorder_active(args))
            self.assertEqual(mamba_slots_per_running_req(args), 3)
            self.assertEqual(mamba_hard_floor(args, 8), 24)

    def test_a_lineage_that_does_support_it_still_gets_the_reduction(self):
        """The gate must not become an unconditional refusal.

        If a future lineage implements the reorder, the reduction returns with
        no further edit here -- so this test is also the can-fail proof that
        the refusal above is about the LINEAGE and not about something else in
        the config drifting.
        """
        with _EnvOn():
            args = _Args()
            with _lineage_supported():
                self.assertTrue(mamba_slot_reorder_active(args))
                self.assertEqual(mamba_slots_per_running_req(args), 2)
                self.assertEqual(mamba_hard_floor(args, 8), 16)

    def test_the_other_conditions_still_bind(self):
        """The lineage check is ADDED to the #755 gate, it does not replace it.

        Each of the original three conditions must still be able to refuse on
        its own, or this change would have quietly widened the gate while
        appearing to narrow it.
        """
        with _EnvOn():
            for label, args in (
                ("radix off", _Args(disable_radix_cache=True)),
                ("write_around", _Args(hicache_write_policy="write_around")),
                ("no hierarchical", _Args(enable_hierarchical_cache=False)),
            ):
                with self.subTest(label), _lineage_supported():
                    self.assertFalse(mamba_slot_reorder_active(args), label)

    def test_the_env_opt_in_still_binds(self):
        args = _Args()
        prev = os.environ.pop(REORDER_ENV, None)
        try:
            with _lineage_supported():
                self.assertFalse(mamba_slot_reorder_active(args))
        finally:
            if prev is not None:
                os.environ[REORDER_ENV] = prev


class TestTheDirectionOfTheError(CustomTestCase):
    """A too-low floor is #581; a too-high one only costs VRAM."""

    def test_the_refused_reduction_raises_the_floor_it_does_not_lower_it(self):
        with _EnvOn():
            args = _Args()
            refused = mamba_hard_floor(args, 8)
            with _lineage_supported():
                taken = mamba_hard_floor(args, 8)
        self.assertGreater(
            refused,
            taken,
            "refusing an undeliverable reduction must move the floor UP; if it "
            "moved down, the boot would validate a pool the runtime over-draws",
        )
        self.assertEqual(refused - taken, 8, "one slot per running request")


if __name__ == "__main__":
    unittest.main()
