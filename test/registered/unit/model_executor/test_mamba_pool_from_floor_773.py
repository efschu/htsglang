"""#773 actuator: size the mamba pool from the FLOOR, not from a hand-pin.

THE GAP THIS CLOSES. The earlier #773 commits made the floor honest and made
the write-through pin bound real, but they moved no bytes: the standing boot
still pins `--max-mamba-cache-size 24`, so a corrected floor of 16 just sits
underneath an unchanged pool. A counter without an actuator.

Removing the pin is not enough on its own, and that is the trap. With the pin
gone the sizing falls through to the `--mamba-full-memory-ratio` branch, which
takes a FRACTION of post-weights VRAM (0.9/1.9 of the rest) and produces a
pool far LARGER than 24 -- which is precisely why the hand-pin was there. So
the pin can only be dissolved by giving the fall-through somewhere better to
land.

THE DEMAND PATH ALREADY EXISTS and is already floored at `mamba_hard_floor`
(#581) and already fitted to the budget (#307). It was simply gated to
uneven-DCP boots, and uneven-DCP is not active in the PP phase, so the
standing boot could never reach it.

THE WIDENING, and why it is principled rather than convenient: the demand path
needs a concurrency TARGET, and the only question is whether that number is
real. `_auto_mamba_target_concurrency` already draws exactly this distinction
-- a USER-supplied `--max-running-requests` is a stated demand, while an
auto-defaulted one (the speculative hook resets an unset value to 48) is not,
and sizing to the latter over-provisions several GB. So the gate widens by the
same predicate the target function already trusts: if the operator stated the
concurrency, size the pool to it. A boot that never stated one keeps the
fraction path, byte-identical.

These tests drive the shipped predicates on stub runners; no GPU.
"""

import os
import unittest
from types import SimpleNamespace
from unittest import mock

from sglang.srt.mem_cache.mamba_pool_floor import mamba_hard_floor
from sglang.srt.model_executor.model_runner_kv_cache_mixin import (
    MAMBA_AUTO_SAFETY_MARGIN,
    MAMBA_FULL_MEMORY_RATIO_DEFAULT,
    ModelRunnerKVCacheMixin,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5)


def _reorder_on():
    """The standing boot sets --mamba-slot-reorder, which exports this env.

    Without it the floor is 1+P+1+1 and the numbers below describe a different
    configuration than the one being sized for.
    """
    return mock.patch.dict(os.environ, {"SGLANG_MAMBA_SLOT_REORDER": "1"}, clear=False)


def _server_args(**over):
    """The standing boot's shape: PP3, hicache write_through, mrr 8 stated."""
    base = dict(
        dcp_size=1,
        pp_size=3,
        dp_size=1,
        enable_dp_attention=False,
        max_mamba_cache_size=None,
        disable_radix_cache=False,
        mamba_full_memory_ratio=MAMBA_FULL_MEMORY_RATIO_DEFAULT,
        max_running_requests=8,
        max_running_requests_user_set=True,
        disable_overlap_schedule=True,
        enable_hierarchical_cache=True,
        hicache_write_policy="write_through",
        mamba_radix_cache_strategy="no_buffer",
    )
    base.update(over)
    sa = SimpleNamespace(**base)
    sa.enable_mamba_extra_buffer = lambda: False
    return sa


def _runner(sa):
    stub = SimpleNamespace(server_args=sa, dp_size=sa.dp_size, pp_size=sa.pp_size)
    for name in (
        "_auto_mamba_demand_active",
        "_auto_mamba_target_concurrency",
        "_auto_mamba_demand_size",
        "_mamba_demand_target_is_stated",
    ):
        setattr(
            stub,
            name,
            getattr(ModelRunnerKVCacheMixin, name).__get__(stub, SimpleNamespace),
        )
    return stub


class TestTheGateReachesTheStandingBoot(CustomTestCase):
    def test_a_stated_concurrency_activates_the_demand_path(self):
        """THE POINT: the standing boot (PP, no uneven-DCP) now reaches it."""
        self.assertTrue(_runner(_server_args())._auto_mamba_demand_active())

    def test_CAN_FAIL_an_unstated_concurrency_keeps_the_fraction_path(self):
        """An auto-defaulted concurrency is not a demand, so it must not size.

        This is the guard against the failure `_auto_mamba_target_concurrency`
        already documents: the speculative hook resets an unset
        --max-running-requests to 48, and sizing a pool to that over-provisions
        several GB and OOMs at pool init.
        """
        sa = _server_args(max_running_requests_user_set=False)
        self.assertFalse(_runner(sa)._auto_mamba_demand_active())

    def test_a_pinned_pool_size_still_wins(self):
        """An operator who names a size is obeyed, not second-guessed."""
        sa = _server_args(max_mamba_cache_size=24)
        self.assertFalse(_runner(sa)._auto_mamba_demand_active())

    def test_an_explicit_fraction_still_wins(self):
        sa = _server_args(mamba_full_memory_ratio=0.5)
        self.assertFalse(_runner(sa)._auto_mamba_demand_active())

    def test_disabled_radix_keeps_its_own_branch(self):
        sa = _server_args(disable_radix_cache=True)
        self.assertFalse(_runner(sa)._auto_mamba_demand_active())

    def test_the_uneven_dcp_route_is_untouched(self):
        """The original activation must still work on its own.

        Without a stated concurrency, an uneven-DCP boot still reaches the
        demand path exactly as before -- the widening ADDS a route, it does
        not replace one.
        """
        sa = _server_args(dcp_size=3, max_running_requests_user_set=False)
        runner = _runner(sa)
        # uneven_dcp_active also consults the env; assert the stated-target
        # disjunct is not what is carrying this case.
        self.assertFalse(runner._mamba_demand_target_is_stated())


class TestTheSizeItProduces(CustomTestCase):
    """The pool must land above the floor and below the hand-pin."""

    def _size(self, ratio, sa=None):
        sa = sa or _server_args()
        return _runner(sa)._auto_mamba_demand_size(ratio)

    def test_the_size_never_sits_below_the_hard_floor(self):
        """#581: the derived value can never under-reserve the running set."""
        sa = _server_args()
        with _reorder_on():
            for ratio in (1, 2, 3, 5):
                with self.subTest(ratio=ratio):
                    self.assertGreaterEqual(
                        self._size(ratio, sa), mamba_hard_floor(sa, 8)
                    )

    def test_the_standing_boot_lands_between_the_floor_and_the_hand_pin(self):
        """The number that replaces the pin, derived rather than chosen.

        ratio 2 (the honest per-request demand once the #755 reorder is
        ported), target 8, safety margin on top -> a pool that leaves real
        room above the floor for cache retention, and still costs less than
        the 24 that was pinned by hand.
        """
        sa = _server_args()
        with _reorder_on():
            floor = mamba_hard_floor(sa, 8)
            size = self._size(2, sa)
        self.assertEqual(floor, 16)
        self.assertEqual(size, int(8 * 2 * MAMBA_AUTO_SAFETY_MARGIN))
        self.assertGreater(size, floor, "there must be room for retention")
        self.assertLess(size, 24, "it must actually cost less than the hand-pin")

    def test_the_retention_budget_is_what_is_left_above_the_floor(self):
        from sglang.srt.mem_cache.mamba_pool_floor import mamba_retention_pin_budget

        sa = _server_args()
        with _reorder_on():
            size = self._size(2, sa)
            self.assertEqual(
                mamba_retention_pin_budget(sa, 8, size),
                size - mamba_hard_floor(sa, 8),
            )
            self.assertEqual(mamba_retention_pin_budget(sa, 8, size), 4)


if __name__ == "__main__":
    unittest.main()


class TestTheSizingRatioTracksTheFloor(CustomTestCase):
    """#773: `_calculate_mamba_ratio` was a SECOND statement of the same number.

    `mamba_pool_floor` calls itself the single source of truth for "slots per
    running request", and the sizing ratio is that number under another name.
    They agreed for every shape that existed when they were written; #755 then
    taught the floor about the lock reorder and did not teach the ratio. The
    demand path multiplies by the RATIO, so an overstatement there sizes a
    pool for demand that no longer exists -- on the standing boot, 30 slots
    where 20 is the demand, which is MORE than the hand-pin it replaces. This
    class pins the repair in both directions.
    """

    RATIO_BASE = 3

    def _ratio(self, sa):
        stub = SimpleNamespace(server_args=sa)
        return ModelRunnerKVCacheMixin._calculate_mamba_ratio(stub)

    def _sa(self, **over):
        sa = _server_args(**over)
        return sa

    def test_it_no_longer_overstates_under_the_reorder(self):
        """THE BUG: 3 charged where the running set holds 2."""
        sa = self._sa()
        with _reorder_on():
            self.assertEqual(mamba_hard_floor(sa, 8) // 8, 2)
            self.assertEqual(self._ratio(sa), 2)

    def test_CAN_FAIL_the_raw_constant_would_still_say_three(self):
        """Proof the assertion above is about the repair, not about the shape."""
        sa = self._sa()
        with _reorder_on():
            self.assertEqual(
                self.RATIO_BASE,
                3,
                "the underlying constant is unchanged; only its use is capped",
            )
            self.assertLess(self._ratio(sa), self.RATIO_BASE)

    def test_every_pre_reorder_shape_is_byte_identical(self):
        """The repair must not move a single number that already agreed.

        Includes extra_buffer_lazy, where the floor is deliberately LARGER
        than the sizing ratio (it charges the transient second ping-pong
        slot). Capping with min() must keep the ratio at 4 there rather than
        inflating it to the floor's 5.
        """
        shapes = {
            "no_buffer": dict(mamba_radix_cache_strategy="no_buffer"),
            "extra_buffer_overlap": dict(
                mamba_radix_cache_strategy="extra_buffer",
                disable_overlap_schedule=False,
            ),
            "extra_buffer_no_overlap": dict(mamba_radix_cache_strategy="extra_buffer"),
            "extra_buffer_lazy": dict(
                mamba_radix_cache_strategy="extra_buffer_lazy",
                disable_overlap_schedule=False,
            ),
            "disable_radix": dict(disable_radix_cache=True),
        }
        expected = {
            "no_buffer": 3,
            "extra_buffer_overlap": 5,
            "extra_buffer_no_overlap": 4,
            "extra_buffer_lazy": 4,
            "disable_radix": 1,
        }
        for name, over in shapes.items():
            with self.subTest(name):
                sa = self._sa(**over)
                sa.enable_mamba_extra_buffer = lambda _sa=sa: (
                    _sa.disable_radix_cache is False
                    and _sa.mamba_radix_cache_strategy
                    in ("extra_buffer", "extra_buffer_lazy")
                )
                sa.enable_mamba_extra_buffer_lazy = lambda _sa=sa: (
                    _sa.disable_radix_cache is False
                    and _sa.mamba_radix_cache_strategy == "extra_buffer_lazy"
                )
                # Reorder OFF: this is the pre-#773 world, and every number
                # here must survive untouched.
                self.assertEqual(self._ratio(sa), expected[name], name)

    def test_the_derived_pool_is_now_cheaper_than_the_hand_pin(self):
        """The whole point of the actuator, with the REAL ratio.

        Passing ratio=2 by hand proved nothing about the call site; this uses
        the ratio the code actually computes.
        """
        sa = self._sa()
        with _reorder_on():
            ratio = self._ratio(sa)
            size = _runner(sa)._auto_mamba_demand_size(ratio)
            floor = mamba_hard_floor(sa, 8)
        self.assertEqual(ratio, 2)
        self.assertEqual(size, 20)
        self.assertGreater(size, floor, "room must remain for retention")
        self.assertLess(size, 24, "the derived pool must cost LESS than the hand-pin")
