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
"""#677 phase 1 WIRING: the two clamps that make a stale verdict safe.

The core module's arithmetic is tested next door. What is tested HERE is
the part that only exists because the scheduler evaluates the purity
verdict on some rounds and not others -- the decode branch is not reached
on a round that selects a prefill batch -- so the recorded verdict and the
recorded id set can both outlive what they describe.

Both clamps degrade to the PRE-CHANGE gate rather than to over-admission,
and that direction is the whole point: a discount that is too small costs
throughput, a discount that is too large hands out state slots the pool
does not have and is refused late inside alloc_req_slots.

The methods are exercised UNBOUND against a stand-in, deliberately. A real
Scheduler cannot be constructed without a model, a device and a process
group; the logic under test reads four plain attributes and nothing else,
so binding it to an object that has exactly those four is the honest way
to test it on CPU with no CUDA present.
"""

import unittest

from sglang.srt.managers.parked_decode_set import ParkedDecodeSet
from sglang.srt.managers.scheduler import Scheduler
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=10)

SLOT_POOL = 12
MAX_RUNNING = 4
PHASE_PP = "pp"
PHASE_TP = "tp"


class _StandIn:
    """Exactly the attributes the two methods read, and nothing more."""

    def __init__(self, enabled=True, phase=PHASE_PP):
        self.parked_decode_set = ParkedDecodeSet(
            slot_pool=SLOT_POOL, max_running=MAX_RUNNING, enabled=enabled
        )
        self.phase_flip_active_stack = phase
        self._parked_decode_verdict = (None, False)


class _Req:
    def __init__(self, rid):
        self.rid = rid


class _Batch:
    def __init__(self, *rids):
        self.reqs = [_Req(r) for r in rids]


def _note(s, batch, blocked):
    return Scheduler._note_parked_carriers(s, batch, blocked)


def _discount(s, running_bs):
    return Scheduler._parked_carrier_discount(s, running_bs)


class TheVerdictIsRecordedOncePerRound(unittest.TestCase):
    def test_a_blocked_round_parks_every_resident_carrier(self):
        s = _StandIn()
        _note(s, _Batch("a", "b", "c", "d"), blocked=True)
        self.assertEqual((PHASE_PP, True), s._parked_decode_verdict)
        self.assertEqual(4, _discount(s, running_bs=4))

    def test_an_unblocked_round_releases_them(self):
        s = _StandIn()
        _note(s, _Batch("a", "b"), blocked=True)
        _note(s, _Batch("a", "b"), blocked=False)
        self.assertEqual(0, _discount(s, running_bs=2))
        self.assertEqual([], s.parked_decode_set.ids)


class AStaleVerdictFromTheOtherPhaseIsDiscarded(unittest.TestCase):
    """PP forbids decode and TP does not. Trusting a PP verdict inside TP
    would discount carriers that are actively decoding, which is the one
    direction that over-admits."""

    def test_the_pp_verdict_does_not_survive_into_tp(self):
        s = _StandIn(phase=PHASE_PP)
        _note(s, _Batch("a", "b", "c", "d"), blocked=True)
        self.assertEqual(4, _discount(s, running_bs=4))
        s.phase_flip_active_stack = PHASE_TP  # the flip commits
        self.assertEqual(0, _discount(s, running_bs=4))

    def test_no_verdict_at_all_discounts_nothing(self):
        s = _StandIn()
        self.assertEqual(0, _discount(s, running_bs=4))


class AStaleIdSetCannotOverCredit(unittest.TestCase):
    """A carrier that finished on a round the decode branch did not reach
    is still listed. Clamping to the resident count means the worst case is
    the pre-change gate, never a discount larger than what is there."""

    def test_the_discount_never_exceeds_the_resident_count(self):
        s = _StandIn()
        _note(s, _Batch("a", "b", "c", "d"), blocked=True)
        # Three of the four completed; the branch that would notice has not
        # run, so the set still names four.
        self.assertEqual(4, len(s.parked_decode_set))
        self.assertEqual(1, _discount(s, running_bs=1))
        self.assertEqual(0, _discount(s, running_bs=0))

    def test_a_negative_resident_count_cannot_underflow(self):
        s = _StandIn()
        _note(s, _Batch("a"), blocked=True)
        self.assertEqual(0, _discount(s, running_bs=-3))


class DisarmedIsTheOldGateExactly(unittest.TestCase):
    def test_nothing_is_recorded_and_nothing_is_discounted(self):
        s = _StandIn(enabled=False)
        _note(s, _Batch("a", "b", "c", "d"), blocked=True)
        self.assertEqual((None, False), s._parked_decode_verdict)
        self.assertEqual(0, _discount(s, running_bs=4))
        self.assertEqual([], s.parked_decode_set.ids)


if __name__ == "__main__":
    unittest.main()
