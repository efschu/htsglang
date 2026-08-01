"""Task #364 slice 3: decouple the concurrency ceiling from the capped
resident GDN pool.

Slice 1 caps the physical GDN/Mamba state pool (``max_mamba_cache_size``) to
``--gdn-resident-state-slots``. But ``_resolve_max_num_reqs`` derives the
concurrency ceiling as ``slots // mamba_ratio`` from that same, now-shrunken,
number -- so the cap silently craters ``max_running_requests``: cap=4,
ratio=5 -> ``4 // 5 = 0`` (a hard boot error), cap=32 -> 16 drops to 6. That
defeats the ladder, whose whole purpose is to admit MORE sessions than fit
resident and run the overflow with a vacated state (slice 2's executor).

The fix sizes the ceiling from the SESSION admission budget (the PRE-CAP
profiled slot count), keeps the physical pool capped, and guards against
over-admission:

* the budget never exceeds the un-capped profiled concurrency -- that was
  already profiled to fit, so restoring it cannot introduce a new OOM;
* it is only raised above the resident cap when the vacate runtime is armed
  (the same flag that caps the pool arms it);
* the KV backing stays guarded by the existing ``token_capacity // 2`` clamp.

The integration cases bind the REAL ``_resolve_max_num_reqs`` to a stand-in
whose two slot quantities are modelled correctly: ``_gdn_profiled_state_slots``
is the pre-cap count, ``max_mamba_cache_size`` is the capped one -- exactly
the state ``handle_max_mamba_cache`` leaves behind.
"""

import unittest
from types import SimpleNamespace

from sglang.srt.mem_cache.gdn_slot_ladder import session_admission_slots
from sglang.srt.model_executor.model_runner_kv_cache_mixin import (
    ModelRunnerKVCacheMixin,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=3, suite="base-a-test-cpu")


class TestSessionAdmissionSlots(CustomTestCase):
    """The pure interlock helper."""

    def test_off_the_cap_is_the_identity(self):
        self.assertEqual(
            session_admission_slots(80, None, vacate_available=False), 80
        )
        self.assertEqual(
            session_admission_slots(80, None, vacate_available=True), 80
        )

    def test_non_binding_cap_is_the_identity(self):
        # cap >= profiled: a ceiling, not a demand.
        self.assertEqual(
            session_admission_slots(80, 80, vacate_available=True), 80
        )
        self.assertEqual(
            session_admission_slots(80, 128, vacate_available=True), 80
        )

    def test_binding_cap_with_vacate_restores_the_profiled_budget(self):
        # FALSIFIER (helper level): cap=4 profiled=80 -> the concurrency
        # budget is 80, not 4. The pool is physically 4; admission is 80.
        self.assertEqual(
            session_admission_slots(80, 4, vacate_available=True), 80
        )
        self.assertEqual(
            session_admission_slots(80, 32, vacate_available=True), 80
        )

    def test_never_exceeds_the_profiled_ceiling(self):
        # The over-admission interlock: the budget is bounded ABOVE by the
        # un-capped profiled count, whatever the cap.
        for cap in (1, 4, 32, 79):
            self.assertLessEqual(
                session_admission_slots(80, cap, vacate_available=True), 80
            )

    def test_without_the_vacate_runtime_it_degrades_to_the_resident_cap(self):
        # No runtime to back the overflow -> admit only what fits resident,
        # never admit-into-OOM.
        self.assertEqual(
            session_admission_slots(80, 4, vacate_available=False), 4
        )


class TestResolveMaxNumReqsDecoupled(CustomTestCase):
    """The real ``_resolve_max_num_reqs`` under the cap."""

    @staticmethod
    def runner(*, profiled, capped, ratio, user_max_running, cap_set=True):
        r = SimpleNamespace()
        r.server_args = SimpleNamespace(
            max_running_requests=user_max_running,
            # handle_max_mamba_cache leaves the pool capped here...
            max_mamba_cache_size=capped,
            gdn_resident_state_slots=(capped if cap_set else None),
            dp_size=1,
        )
        r.dp_size = 1
        r.model_config = SimpleNamespace(context_len=8192)
        r.mambaish_config = object()
        r._calculate_mamba_ratio = lambda: ratio
        # ...and stashes the pre-cap profiled count here.
        r._gdn_profiled_state_slots = profiled
        return r

    def resolve(self, r, token_capacity=200000):
        return ModelRunnerKVCacheMixin._resolve_max_num_reqs(
            r, token_capacity=token_capacity
        )

    def test_cap_32_no_longer_drops_16_to_6(self):
        # profiled 80 (= 16 reqs x ratio 5), capped to 32 resident slots.
        r = self.runner(profiled=80, capped=32, ratio=5, user_max_running=16)
        self.assertEqual(self.resolve(r), 16)

    def test_cap_4_no_longer_yields_zero(self):
        r = self.runner(profiled=80, capped=4, ratio=5, user_max_running=16)
        self.assertEqual(self.resolve(r), 16)

    def test_the_user_ceiling_still_binds(self):
        # Decoupling raises the budget to the profiled concurrency; it does
        # not raise it ABOVE what the user asked for.
        r = self.runner(profiled=80, capped=4, ratio=5, user_max_running=8)
        self.assertEqual(self.resolve(r), 8)

    def test_kv_capacity_still_guards_against_over_admission(self):
        # token_capacity // 2 is the KV backing guard and is untouched: a
        # tiny KV pool caps concurrency below the session budget.
        r = self.runner(profiled=80, capped=4, ratio=5, user_max_running=16)
        # token_capacity // 2 = 6 binds below the 16-session budget.
        self.assertEqual(self.resolve(r, token_capacity=12), 6)

    def test_profiled_ceiling_bounds_the_budget(self):
        # profiled 40 (8 reqs), user asked 16: the un-capped pool only ever
        # backed 8, so 8 is the honest ceiling even with vacate.
        r = self.runner(profiled=40, capped=4, ratio=5, user_max_running=16)
        self.assertEqual(self.resolve(r), 8)

    def test_default_path_is_byte_identical(self):
        # Cap unset: profiled == max_mamba_cache_size, the helper is the
        # identity, and the result is the stock slots // ratio.
        r = self.runner(
            profiled=80, capped=80, ratio=5, user_max_running=32, cap_set=False
        )
        r._gdn_profiled_state_slots = None  # attribute absent off the cap
        self.assertEqual(self.resolve(r), 16)  # min(32, 80//5, 200000//2)

    def test_a_genuinely_impossible_pool_still_raises(self):
        # profiled 4, ratio 5: even un-capped the pool cannot hold one
        # request. That is a real error, not the cap's fault.
        r = self.runner(
            profiled=4, capped=4, ratio=5, user_max_running=16, cap_set=False
        )
        r._gdn_profiled_state_slots = None
        with self.assertRaises(RuntimeError):
            self.resolve(r)


if __name__ == "__main__":
    unittest.main()
