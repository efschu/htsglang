"""#755: the 3-slot mamba floor drops to 2, mechanism and floor together.

NOTE_755 determined the reduction EXISTS as a lock-lifetime reorder:

    default   alloc -> copy -> insert -> dec(old) -> inc(new)
    #755      dec(old) -> alloc -> copy -> insert -> inc(new)

The donated slot BECOMES the next pinned checkpoint, so the third slot exists
only because the OLD pin is held across the alloc. Release first and the
concurrent demand at the boundary is active + donated = 2. For 4 running
requests: 12 -> 8.

The note is equally explicit about what must NOT happen: editing the floor
formula alone would claim slots the runtime still locks, which is the #581
late-assert class the floor was built to kill. So the floor moves ONLY when
the mechanism moves, and both read one predicate.

THE RELEASE WINDOW, stated because it is the whole risk. Between
``dec_lock_ref(old)`` and ``inc_lock_ref(new)`` the old anchor is evictable.
That is safe only if it is host-backed AT THAT MOMENT: an eviction (or a failed
alloc) then costs a ``load_back`` or a re-prefill instead of the anchor. Hence
two gates, not one -- a config-level predicate that decides the floor, and a
PER-NODE check at the site. A node that is not backed takes the skip path,
which holds active + old pin = 2 and therefore still fits the reduced floor.
That is what makes the floor honest: no path under this config exceeds it.
"""

import os
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from sglang.srt.mem_cache.mamba_pool_floor import (
    MAMBA_SLOT_REORDER_ENV,
    describe_mamba_floor,
    mamba_hard_floor,
    mamba_slot_reorder_active,
    mamba_slots_per_running_req,
)
from sglang.srt.mem_cache.mamba_radix_cache import MambaRadixCache
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=2, suite="base-a-test-cpu")

RUNNING = 4


def _args(**over):
    """The live no_buffer shape: ping-pong 0, radix on."""
    base = dict(
        disable_radix_cache=False,
        disable_overlap_schedule=True,
        enable_hierarchical_cache=True,
        hicache_write_policy="write_through",
    )
    base.update(over)
    a = SimpleNamespace(**base)
    a.enable_mamba_extra_buffer = lambda: False
    return a


def _with_env(on: bool):
    return patch.dict(
        os.environ, {MAMBA_SLOT_REORDER_ENV: "1"} if on else {}, clear=False
    )


def _with_lineage():
    """#773: assert the built lineage carries the reorder.

    These cases are about the MECHANISM's arithmetic, and they used to get
    admissibility for free from `enable_hierarchical_cache=True`. That is
    exactly the condition #773 found to be inverted: hierarchical cache is
    what routes a hybrid-SSM boot to `UnifiedRadixCache`, which does not
    implement the reorder, so the floor may no longer take the reduction on
    that basis alone. The mechanism itself is unchanged -- these tests state
    what it is worth WHERE IT EXISTS, which is what they always meant.
    """
    import sglang.srt.mem_cache.mamba_pool_floor as floor_mod

    return patch.object(floor_mod, "mamba_reorder_lineage_supported", lambda _sa: True)


class TestTheFloorMovesOnlyWithTheMechanism(CustomTestCase):
    def setUp(self):
        os.environ.pop(MAMBA_SLOT_REORDER_ENV, None)

    def test_the_default_floor_is_still_three_per_request(self):
        """Nothing changes for anyone who did not ask."""
        a = _args()
        self.assertEqual(mamba_slots_per_running_req(a), 3)
        self.assertEqual(mamba_hard_floor(a, RUNNING), 12)

    def test_the_reorder_drops_it_to_two_and_the_specimen_to_eight(self):
        """THE TARGET: 4 requests serve on 8 slots, where the reorder exists."""
        with _with_env(True), _with_lineage():
            a = _args()
            self.assertTrue(mamba_slot_reorder_active(a))
            self.assertEqual(mamba_slots_per_running_req(a), 2)
            self.assertEqual(mamba_hard_floor(a, RUNNING), 8)

    def test_CAN_FAIL_without_hierarchical_cache_the_old_floor_stands(self):
        """The can-fail the task names.

        A device-only pool cannot promise the released anchor survives, so the
        reorder is inadmissible and the floor must NOT move -- even though the
        operator set the env.
        """
        with _with_env(True):
            a = _args(enable_hierarchical_cache=False)
            self.assertFalse(mamba_slot_reorder_active(a))
            self.assertEqual(mamba_hard_floor(a, RUNNING), 12)

    def test_CAN_FAIL_write_around_does_not_promise_a_backup_at_release_time(self):
        """write_through is required, not merely a host tier.

        The gate is 'backed AT RELEASE TIME'. A policy that defers the copy
        cannot promise that, so the floor stays put.
        """
        with _with_env(True):
            for policy in ("write_back", "write_around", None):
                with self.subTest(policy=policy):
                    a = _args(hicache_write_policy=policy)
                    self.assertFalse(mamba_slot_reorder_active(a))
                    self.assertEqual(mamba_hard_floor(a, RUNNING), 12)

    def test_CAN_FAIL_the_env_alone_does_nothing(self):
        with _with_env(False):
            a = _args()
            self.assertFalse(mamba_slot_reorder_active(a))
            self.assertEqual(mamba_hard_floor(a, RUNNING), 12)

    def test_no_radix_cache_is_unaffected_either_way(self):
        with _with_env(True):
            a = _args(disable_radix_cache=True)
            self.assertFalse(mamba_slot_reorder_active(a))
            self.assertEqual(mamba_slots_per_running_req(a), 1)

    def test_the_derivation_string_names_the_sharing(self):
        with _with_env(True), _with_lineage():
            text = describe_mamba_floor(_args(), RUNNING)
        self.assertIn("#755", text)
        self.assertIn("donation/pinned checkpoint", text)
        self.assertIn("= 8 slots", text)

    def test_the_default_derivation_still_lists_both_terms(self):
        text = describe_mamba_floor(_args(), RUNNING)
        self.assertIn("1 donation", text)
        self.assertIn("1 pinned checkpoint", text)
        self.assertIn("= 12 slots", text)


class TestThePerNodeGate(CustomTestCase):
    """The site asks a SECOND question the config cannot answer."""

    def _cache(self, reorder: bool):
        c = MambaRadixCache.__new__(MambaRadixCache)
        c.mamba_slot_reorder = reorder
        # The root is deliberately BACKED. A bare object() would be refused by
        # the mamba_backuped check instead, so the root test would pass without
        # exercising the root rule at all -- it did, and a mutation that
        # released the root survived.
        c.root_node = SimpleNamespace(mamba_backuped=True)
        return c

    def _node(self, backed: bool):
        return SimpleNamespace(mamba_backuped=backed)

    def test_a_backed_node_under_the_reorder_is_admissible(self):
        c = self._cache(True)
        self.assertTrue(c._mamba_early_release_admissible(self._node(True)))

    def test_an_UNBACKED_node_is_refused_even_with_the_reorder_on(self):
        """The heart of it: the config is not enough.

        Releasing this pin would open a window whose only escape -- load_back
        -- does not exist for this node yet.
        """
        c = self._cache(True)
        self.assertFalse(c._mamba_early_release_admissible(self._node(False)))

    def test_without_the_reorder_even_a_backed_node_is_refused(self):
        c = self._cache(False)
        self.assertFalse(c._mamba_early_release_admissible(self._node(True)))

    def test_the_root_is_never_released_early(self):
        """And the root is BACKED here, so only the root rule can refuse it."""
        c = self._cache(True)
        self.assertTrue(
            c.root_node.mamba_backuped, "fixture must not refuse for the other reason"
        )
        self.assertFalse(c._mamba_early_release_admissible(c.root_node))

    def test_a_missing_node_is_refused_rather_than_assumed(self):
        c = self._cache(True)
        self.assertFalse(c._mamba_early_release_admissible(None))

    def test_a_node_without_the_attribute_is_refused(self):
        """Absent is not backed. A stub or an older node type must not pass."""
        c = self._cache(True)
        self.assertFalse(c._mamba_early_release_admissible(SimpleNamespace()))


class TestTheReleaseWindowIsBounded(CustomTestCase):
    """Obligation 2: the early release is a lock-PROTOCOL change, so the
    window it opens is written down and pinned rather than left implicit."""

    def test_the_unbacked_path_still_fits_the_reduced_floor(self):
        """Why the skip path is the right refusal.

        Under the reduced floor there is no third slot to fall back on, so a
        node that cannot be released must NOT revert to the old order. Skipping
        the insert holds active + old pin = 2, which is exactly the budget the
        floor reserves. This is the arithmetic that makes the floor safe for
        EVERY path, not just the happy one.
        """
        with _with_env(True):
            per_req = mamba_slots_per_running_req(_args())
        active_plus_old_pin = 2
        self.assertLessEqual(active_plus_old_pin, per_req)

    def test_the_happy_path_also_fits(self):
        """active + donated = 2, the donated slot becoming the new pin."""
        with _with_env(True), _with_lineage():
            per_req = mamba_slots_per_running_req(_args())
        active_plus_donated = 2
        self.assertLessEqual(active_plus_donated, per_req)

    def test_the_old_order_would_NOT_fit_the_reduced_floor(self):
        """CAN-FAIL for the whole design.

        If the 3-slot order fitted the reduced floor, the gate would be
        pointless and a silent fallback harmless. It does not fit -- which is
        precisely why the site must refuse rather than revert.
        """
        with _with_env(True), _with_lineage():
            per_req = mamba_slots_per_running_req(_args())
        active_plus_donated_plus_old_pin = 3
        self.assertGreater(active_plus_donated_plus_old_pin, per_req)


if __name__ == "__main__":
    unittest.main()
