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
"""#888b: the W38 stall, modelled, and the two defects it is made of.

THE SPECIMEN, taken from boot_w38rerun_0826_1304.log verbatim rather than
paraphrased, because the numbers are the argument::

    13:12:25  holding in pp: prefilling in pp (33 tok pending), running bs 8
    13:12:28  #788 PP-ADMISSION verdict=DECLINE avail=12876 evictable=0
              queue=2 running=4 chunked=0
              reason=gate=batch_full_or_empty_queue(batch_is_full=1,queue=2)
    13:13:14  arming pp_to_tp: decode stall cap: 8 req stalled 180.0s

Two things must be pinned and they are different pins:

1. THE BINDER IS THE REQUEST SEAT. 12876 KV tokens were free against 33
   pending, so any relief denominated in KV tokens frees the wrong thing and
   admits nothing. A verdict that reports a KV shortfall on this input has
   reproduced the mis-attribution the ticket was framed with.

2. THE FLAG IS NOT A LATCH IN A PHASE THAT FORBIDS DECODE. Sixty of sixty
   emitted declines inside the stall name ``batch_full_or_empty_queue`` and
   zero name ``no_allocatable_reqs``: the gate returned above the seat test
   on every pass for 156 seconds. Under strict purity every clear site of
   ``batch_is_full`` lives on the decode path, which the phase forbids.

The danger direction has its own class at the bottom: a yield destroys a
request's decode progress, so every refusal rule is pinned separately and a
phase that permits decode must never yield at all.
"""

import unittest

from sglang.srt.managers.parked_carrier_relief import (
    BINDER_KV_TOKEN,
    BINDER_MAMBA_SLOT,
    BINDER_NONE,
    BINDER_REQ_SLOT,
    ENV_PARKED_CARRIER_RELIEF,
    carrier_relief_verdict,
    hold_receipt,
    latched_flag_must_be_rederived,
    name_the_binder,
    parked_carrier_relief_enabled,
    relief_receipt,
)
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=10)


#: The W38 rerun stall, 13:11:00-13:13:14, as keyword arguments.
W38_STALL = dict(
    decode_forbidden=True,
    pending_prefill_tokens=33,
    queue_len=2,
    allocatable_reqs=0,
    resident_bs=8,
    parked_count=8,
    chunk_in_flight=False,
    req_slots_free=0,
    kv_avail_tokens=12876,
    kv_need_tokens=33,
    mamba_slots_free=7,
)


def _stall(**overrides):
    kwargs = dict(W38_STALL)
    kwargs.update(overrides)
    return carrier_relief_verdict(**kwargs)


class _Env:
    """Set/restore one environment variable around a block."""

    def __init__(self, **kw):
        self.kw = kw
        self.old = {}

    def __enter__(self):
        import os

        for k, v in self.kw.items():
            self.old[k] = os.environ.get(k)
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = str(v)
        return self

    def __exit__(self, *a):
        import os

        for k, v in self.old.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v
        return False


class TheMeasuredStall(unittest.TestCase):
    def test_the_stall_yields_one_carrier(self):
        v = _stall()
        self.assertTrue(v.yield_carrier, v.reason)

    def test_the_binder_is_the_seat_and_not_the_kv_pool(self):
        """The pin against the mis-attribution that framed the ticket.

        12876 free tokens is not a KV shortfall however high the pool's
        utilisation ratio reads.
        """
        v = _stall()
        self.assertEqual(BINDER_REQ_SLOT, v.binder)
        self.assertNotEqual(BINDER_KV_TOKEN, v.binder)

    def test_the_reason_carries_the_numbers_it_decided_from(self):
        v = _stall()
        self.assertIn("33 tok", v.reason)
        self.assertIn("12876", v.reason)


class TheBinderIsNamedFromMeasurement(unittest.TestCase):
    def test_an_empty_seat_table_is_a_seat_shortfall(self):
        self.assertEqual(
            BINDER_REQ_SLOT,
            name_the_binder(
                req_slots_free=0,
                kv_avail_tokens=12876,
                kv_need_tokens=33,
                mamba_slots_free=7,
            ),
        )

    def test_a_real_kv_shortfall_is_still_named_kv(self):
        self.assertEqual(
            BINDER_KV_TOKEN,
            name_the_binder(
                req_slots_free=3,
                kv_avail_tokens=10,
                kv_need_tokens=4096,
                mamba_slots_free=7,
            ),
        )

    def test_a_kv_need_of_zero_is_not_a_kv_shortfall(self):
        """An unmeasured ask must not be reported as a shortfall.

        The case that decides it is an OVER-COMMITTED pool: several of this
        tree's availability terms are differences and can read below zero
        (``rem_total_tokens`` subtracts reservation offsets). Without the
        ``need > 0`` guard, ``avail < need`` is then true for a caller that
        asked for nothing, and the verdict reports a KV shortfall that was
        never measured -- the same false finding, one layer down.
        """
        self.assertEqual(
            BINDER_NONE,
            name_the_binder(
                req_slots_free=3,
                kv_avail_tokens=0,
                kv_need_tokens=0,
                mamba_slots_free=7,
            ),
        )
        self.assertEqual(
            BINDER_NONE,
            name_the_binder(
                req_slots_free=3,
                kv_avail_tokens=-5,
                kv_need_tokens=0,
                mamba_slots_free=7,
            ),
        )

    def test_an_exhausted_state_pool_is_named_mamba(self):
        self.assertEqual(
            BINDER_MAMBA_SLOT,
            name_the_binder(
                req_slots_free=3,
                kv_avail_tokens=12876,
                kv_need_tokens=33,
                mamba_slots_free=0,
            ),
        )

    def test_nothing_short_is_named_nothing(self):
        self.assertEqual(
            BINDER_NONE,
            name_the_binder(
                req_slots_free=4,
                kv_avail_tokens=12876,
                kv_need_tokens=33,
                mamba_slots_free=7,
            ),
        )


class TheDangerDirection(unittest.TestCase):
    """A yield destroys a request's decode progress. Every rule here refuses
    one way of destroying it for nothing."""

    def test_a_phase_that_permits_decode_never_yields(self):
        v = _stall(decode_forbidden=False)
        self.assertFalse(v.yield_carrier)
        self.assertIn("permits decode", v.reason)

    def test_an_idle_phase_never_yields(self):
        self.assertFalse(_stall(queue_len=0).yield_carrier)
        self.assertFalse(_stall(pending_prefill_tokens=0).yield_carrier)

    def test_an_unblocked_admission_never_yields(self):
        v = _stall(allocatable_reqs=1)
        self.assertFalse(v.yield_carrier)
        self.assertIn("not blocked", v.reason)

    def test_a_chunk_in_flight_defers_to_the_679_ladder(self):
        v = _stall(chunk_in_flight=True)
        self.assertFalse(v.yield_carrier)
        self.assertIn("679", v.reason)

    def test_an_unreconciled_parked_set_never_yields(self):
        """The prohibition is active but no carrier is RECORDED as parked.
        Acting on that would pick a victim by guess."""
        v = _stall(parked_count=0)
        self.assertFalse(v.yield_carrier)
        self.assertIn("no carrier is recorded", v.reason)

    def test_the_last_resident_is_never_yielded(self):
        v = _stall(resident_bs=1, parked_count=1)
        self.assertFalse(v.yield_carrier)
        self.assertIn("keeps the last", v.reason)

    def test_blocked_with_nothing_measured_short_holds_and_says_so(self):
        v = _stall(req_slots_free=2, mamba_slots_free=5, kv_avail_tokens=12876)
        self.assertFalse(v.yield_carrier)
        self.assertEqual(BINDER_NONE, v.binder)
        self.assertIn("binder is elsewhere", v.reason)


class TheLatchedFlag(unittest.TestCase):
    def test_a_forbidden_phase_must_rederive_it(self):
        self.assertTrue(
            latched_flag_must_be_rederived(decode_forbidden=True, flag_is_latched=True)
        )

    def test_a_permitting_phase_leaves_the_latch_to_its_owners(self):
        self.assertFalse(
            latched_flag_must_be_rederived(decode_forbidden=False, flag_is_latched=True)
        )

    def test_an_unset_flag_is_never_touched(self):
        self.assertFalse(
            latched_flag_must_be_rederived(decode_forbidden=True, flag_is_latched=False)
        )


class TheKillSwitch(unittest.TestCase):
    """A kill switch, NOT an opt-in. #679's ladder was off on the boot that
    wedged, which is why it could not have helped."""

    def test_the_default_is_armed(self):
        with _Env(**{ENV_PARKED_CARRIER_RELIEF: None}):
            self.assertTrue(parked_carrier_relief_enabled())

    def test_zero_disarms(self):
        with _Env(**{ENV_PARKED_CARRIER_RELIEF: 0}):
            self.assertFalse(parked_carrier_relief_enabled())

    def test_false_disarms(self):
        with _Env(**{ENV_PARKED_CARRIER_RELIEF: "false"}):
            self.assertFalse(parked_carrier_relief_enabled())


class TheReceipts(unittest.TestCase):
    def test_the_relief_receipt_names_the_seat_delta(self):
        line = relief_receipt(
            _stall(), seats_before=0, seats_after=1, tokens_gained=25625, victims=1
        )
        self.assertIn("request seats 0 -> 1", line)
        self.assertIn("req_slot", line)

    def test_an_ordinary_hold_is_silent(self):
        self.assertIsNone(hold_receipt(_stall(decode_forbidden=False)))
        self.assertIsNone(hold_receipt(_stall(queue_len=0)))
        self.assertIsNone(hold_receipt(_stall(allocatable_reqs=3)))

    def test_a_hold_that_means_something_is_logged(self):
        self.assertIsNotNone(hold_receipt(_stall(parked_count=0)))
        self.assertIsNotNone(hold_receipt(_stall(req_slots_free=2, mamba_slots_free=5)))

    def test_a_yield_has_no_hold_receipt(self):
        self.assertIsNone(hold_receipt(_stall()))


if __name__ == "__main__":
    unittest.main()
