# Copyright 2026 SGLang Team
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""#888b WIRING: the scheduler half, against a stand-in.

The decision is tested next door as arithmetic. What is tested HERE is the
part that only exists because the scheduler owns the state: the latched flag
is cleared exactly when its clear sites are unreachable, the actuator is
reached exactly once per pass with its #583 precondition set, and a boot
without a phase prohibition is byte-identical.

The methods are exercised UNBOUND against a stand-in, the same idiom
``test_parked_decode_wiring_677.py`` uses and for the same reason: a real
Scheduler needs a model, a device and a process group, while the logic under
test reads a handful of plain attributes and calls one actuator.

THE ACTUATOR IS STUBBED, DELIBERATELY. ``_retract_decode_and_requeue`` is
#679's extracted retraction and is proven where it lives; what must be
proven here is that it is CALLED, with ``uniform_avail_floor`` already the
reduced value, and not called on any of the refusal paths. A stand-in that
records the call and frees a seat models exactly that contract.
"""

import unittest

from sglang.srt.managers.log_cycle_collapse import CycleCollapse
from sglang.srt.managers.parked_carrier_relief import ENV_PARKED_CARRIER_RELIEF
from sglang.srt.managers.scheduler import Scheduler
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=10)

PHASE_PP = "pp"
PHASE_TP = "tp"

#: The recipe the stall was measured on: max_running_requests=8, so the
#: request seat table has eight seats and eight residents fill it.
SEATS = 8
KV_FREE = 12876
PENDING = 33


class _Pool:
    def __init__(self, free):
        self._free = free

    def available_size(self):
        return self._free


class _Req:
    def __init__(self, rid):
        self.rid = rid


class _Batch:
    def __init__(self, n):
        self.reqs = [_Req(f"r{i}") for i in range(n)]
        self.batch_is_full = False
        self.uniform_avail_floor = None

    def batch_size(self):
        return len(self.reqs)


class _ParkedSet:
    def __init__(self, n):
        self.resident_count = n


class _StandIn:
    """Exactly what the two methods read, and nothing more."""

    def __init__(
        self,
        *,
        phase=PHASE_PP,
        blocked=True,
        residents=SEATS,
        seats_free=0,
        kv_free=KV_FREE,
        mamba_free=7,
        queue=2,
        pending=PENDING,
        chunked=None,
    ):
        self.phase_flip_active_stack = phase
        self._parked_decode_verdict = (phase if blocked else None, blocked)
        self.parked_decode_set = _ParkedSet(residents)
        self.req_to_token_pool = _Pool(seats_free)
        self.req_to_token_pool.mamba_allocator = _Pool(mamba_free)
        self.token_to_kv_pool_allocator = _Pool(kv_free)
        self.waiting_queue = list(range(queue))
        self.chunked_req = chunked
        self._pending = pending
        self.running_batch = _Batch(residents)
        self._parked_relief_collapse = CycleCollapse()
        #: recorded actuator calls
        self.retractions = []
        #: seats the stubbed retraction frees per call
        self.seats_per_retraction = 1

    # -- the scheduler surface the methods use ---------------------------
    def _decode_forbidden_this_phase(self):
        """The REAL predicate, bound to this stand-in.

        Not a stub: the two methods under test both consult it, and a stubbed
        prohibition would let a wiring test pass while the clamp that makes a
        stale verdict safe was broken.
        """
        return Scheduler._decode_forbidden_this_phase(self)

    def _admissible_prefill_tokens(self):
        return self._pending

    def uniform_min_avail(self):
        return self.token_to_kv_pool_allocator.available_size()

    def _retract_decode_and_requeue(self, batch, *, kv_full_retract_flag, reason=None):
        self.retractions.append(
            dict(
                floor=batch.uniform_avail_floor,
                kv_full=kv_full_retract_flag,
                reason=reason,
                residents=len(batch.reqs),
            )
        )
        freed = min(self.seats_per_retraction, max(0, len(batch.reqs) - 1))
        for _ in range(freed):
            batch.reqs.pop()
        self.req_to_token_pool._free += freed
        self.parked_decode_set.resident_count -= freed
        return freed * 25625


def _forbidden(s):
    return Scheduler._decode_forbidden_this_phase(s)


def _rederive(s, batch):
    return Scheduler._rederive_latched_batch_full(s, batch)


def _yield(s, batch, running_bs, allocatable):
    return Scheduler._maybe_yield_parked_carrier(s, batch, running_bs, allocatable)


class ThePhaseProhibitionIsReadFromTheRecordedVerdict(unittest.TestCase):
    """Same two clamps as #677's carrier discount, reused rather than
    re-derived: a verdict from the OTHER phase is discarded outright."""

    def test_a_blocked_pp_verdict_in_pp_forbids_decode(self):
        self.assertTrue(_forbidden(_StandIn()))

    def test_a_pp_verdict_does_not_survive_into_tp(self):
        s = _StandIn(phase=PHASE_PP)
        s.phase_flip_active_stack = PHASE_TP
        self.assertFalse(_forbidden(s))

    def test_an_unblocked_verdict_permits_decode(self):
        self.assertFalse(_forbidden(_StandIn(blocked=False)))

    def test_no_verdict_at_all_permits_decode(self):
        s = _StandIn()
        del s._parked_decode_verdict
        self.assertFalse(_forbidden(s))


class TheLatchIsRederivedOnlyWhereItsClearSitesAreUnreachable(unittest.TestCase):
    def test_a_latched_flag_in_a_forbidden_phase_is_cleared(self):
        s = _StandIn()
        b = s.running_batch
        b.batch_is_full = True
        self.assertTrue(_rederive(s, b))
        self.assertFalse(b.batch_is_full)

    def test_a_latched_flag_in_a_permitting_phase_is_left_alone(self):
        """The stock owners clear it on the decode path. Touching it here
        would be a second authority over a flag that already has one."""
        s = _StandIn(blocked=False)
        b = s.running_batch
        b.batch_is_full = True
        self.assertFalse(_rederive(s, b))
        self.assertTrue(b.batch_is_full)

    def test_an_unlatched_flag_is_not_reported_as_rederived(self):
        s = _StandIn()
        b = s.running_batch
        self.assertFalse(_rederive(s, b))
        self.assertFalse(b.batch_is_full)

    def test_the_kill_switch_restores_the_latch(self):
        import os

        old = os.environ.get(ENV_PARKED_CARRIER_RELIEF)
        os.environ[ENV_PARKED_CARRIER_RELIEF] = "0"
        try:
            s = _StandIn()
            b = s.running_batch
            b.batch_is_full = True
            self.assertFalse(_rederive(s, b))
            self.assertTrue(b.batch_is_full)
        finally:
            if old is None:
                os.environ.pop(ENV_PARKED_CARRIER_RELIEF, None)
            else:
                os.environ[ENV_PARKED_CARRIER_RELIEF] = old


class TheStallResolves(unittest.TestCase):
    """The falsifier. The W38 state, and what the pass must do with it."""

    def test_one_seat_is_freed_and_the_actuator_ran_once(self):
        s = _StandIn()
        gained = _yield(s, s.running_batch, running_bs=SEATS, allocatable=0)
        self.assertEqual(1, gained)
        self.assertEqual(1, len(s.retractions))
        self.assertEqual(1, s.req_to_token_pool.available_size())

    def test_the_583_precondition_is_set_before_the_actuator_runs(self):
        """``retract_decode``'s loop bound and last-survivor test read
        ``uniform_avail_floor``. #583 is the case where the entry decision
        was uniform and the loop bound was not."""
        s = _StandIn()
        _yield(s, s.running_batch, running_bs=SEATS, allocatable=0)
        self.assertEqual(KV_FREE, s.retractions[0]["floor"])

    def test_the_receipt_does_not_claim_the_kv_pool_was_full(self):
        """``kv_full_retract_flag=True`` prints "KV cache pool is full",
        which is the exact mis-attribution this ticket corrects: 12876
        tokens were free."""
        s = _StandIn()
        _yield(s, s.running_batch, running_bs=SEATS, allocatable=0)
        self.assertFalse(s.retractions[0]["kv_full"])
        self.assertIn("req_slot", s.retractions[0]["reason"])

    def test_exactly_one_victim_per_pass(self):
        """The uniform loop bound. A constant of one cannot diverge across
        ranks the way #583's rank-local bound did."""
        s = _StandIn()
        s.seats_per_retraction = 3
        _yield(s, s.running_batch, running_bs=SEATS, allocatable=0)
        self.assertEqual(1, len(s.retractions))


class NoYieldOnTheRefusalPaths(unittest.TestCase):
    """Each of these must leave the actuator untouched. A retraction that
    should not have happened costs a user their decode progress."""

    def _assert_no_retraction(self, s, allocatable=0):
        gained = _yield(
            s,
            s.running_batch,
            running_bs=len(s.running_batch.reqs),
            allocatable=allocatable,
        )
        self.assertEqual(0, gained)
        self.assertEqual([], s.retractions)

    def test_a_phase_that_permits_decode(self):
        self._assert_no_retraction(_StandIn(blocked=False))

    def test_an_empty_queue(self):
        self._assert_no_retraction(_StandIn(queue=0))

    def test_no_pending_prefill_tokens(self):
        self._assert_no_retraction(_StandIn(pending=0))

    def test_admission_already_possible(self):
        self._assert_no_retraction(_StandIn(seats_free=2), allocatable=2)

    def test_a_chunk_in_flight(self):
        self._assert_no_retraction(_StandIn(chunked=object()))

    def test_a_single_resident(self):
        self._assert_no_retraction(_StandIn(residents=1))

    def test_the_kill_switch(self):
        import os

        old = os.environ.get(ENV_PARKED_CARRIER_RELIEF)
        os.environ[ENV_PARKED_CARRIER_RELIEF] = "0"
        try:
            self._assert_no_retraction(_StandIn())
        finally:
            if old is None:
                os.environ.pop(ENV_PARKED_CARRIER_RELIEF, None)
            else:
                os.environ[ENV_PARKED_CARRIER_RELIEF] = old


class ANonPurityBootIsUntouched(unittest.TestCase):
    """The whole backward-compatibility argument in one class: with no
    phase prohibition recorded, neither half of this fix does anything."""

    def test_neither_the_flag_nor_the_actuator_moves(self):
        s = _StandIn(blocked=False)
        b = s.running_batch
        b.batch_is_full = True
        self.assertFalse(_rederive(s, b))
        self.assertTrue(b.batch_is_full)
        self.assertEqual(0, _yield(s, b, running_bs=SEATS, allocatable=0))
        self.assertEqual([], s.retractions)


class TheBootRefusalIsNotLiftedByThisFix(unittest.TestCase):
    """#888b does NOT earn the right to drop `validate_purity_policy_pair`.

    That guard refuses a strict-purity boot with no bounded PP residency, on
    the ground that "a PP phase that cannot admit its pending prefill ... has
    NO exit except the bounded window: the load-triggered rule needs prefill
    to drain, and prefill cannot drain".

    This fix WEAKENS that premise -- a phase that can take a request seat back
    can drain -- but it does not eliminate it. The yield still refuses on
    every rule in the sibling class above: a single resident, an unreconciled
    parked set, a binder this module does not measure. A residency that meets
    one of those is exactly the state the guard describes, and it would then
    have no exit at all.

    So the refusal STAYS, and it stays until a loaded window shows the drain
    completing on its own. This test is the pin that keeps the two decisions
    apart: a later reader who lifts the refusal because "#888b fixed that" has
    to delete this class and its reasons first.
    """

    def test_the_unbounded_strict_config_is_still_refused(self):
        from sglang.srt.managers.phase_purity import (
            PhasePurityError,
            validate_purity_policy_pair,
        )

        class _Purity:
            enforced = True
            strict = True

            def describe(self):
                return "strict"

        class _Cfg:
            pp_window_s = 0.0
            decode_stall_slo_s = 0.0
            flip_cost_s = 3.2

        with self.assertRaises(PhasePurityError):
            validate_purity_policy_pair(_Purity(), _Cfg())

    def test_a_declared_slo_is_still_the_accepted_bound(self):
        from sglang.srt.managers.phase_purity import validate_purity_policy_pair

        class _Purity:
            enforced = True
            strict = True

            def describe(self):
                return "strict"

        class _Cfg:
            pp_window_s = 0.0
            decode_stall_slo_s = 180.0
            flip_cost_s = 3.2

        validate_purity_policy_pair(_Purity(), _Cfg())


if __name__ == "__main__":
    unittest.main()
