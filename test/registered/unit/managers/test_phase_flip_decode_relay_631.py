# SPDX-License-Identifier: Apache-2.0
"""#631: the TP->PP leg must hand the PP phase its OWN last token.

WHAT THIS PINS. The first plain-decode round after a ``tp_to_pp`` cutover
does not carry its input token on the batch: it gathers it from the
future-map relay, pool-indexed. The speculative phase it is leaving does
not write that relay (the non-overlap V2 path installs
``next_draft_input`` directly), so the row still holds whatever the
PREVIOUS PP phase left there -- stale by the whole TP phase. The carry
must re-derive it from ``req.output_ids[-1]``.

THE CAN-FAIL IS EXPLICIT. ``test_stale_relay_is_what_the_gather_would_read``
asserts the pre-state these tests are about: with the seed removed the
gather reads a foreign token, so a no-op implementation cannot pass the
suite by accident.
"""

import os
import sys
import unittest

sys.path.insert(
    0,
    os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..", "..", "..", "..", "python")
    ),
)

import torch

from sglang.srt.managers.phase_flip_resident_carry import reseed_decode_input_relay


class _Req:
    def __init__(self, rid, req_pool_idx, output_ids):
        self.rid = rid
        self.req_pool_idx = req_pool_idx
        self.output_ids = list(output_ids)


class _Batch:
    def __init__(self, reqs, input_ids=None):
        self.reqs = list(reqs)
        self.input_ids = input_ids


class _FutureMap:
    """Only the surface ``reseed_decode_input_relay`` touches."""

    def __init__(self, size, fill):
        self.output_tokens_buf = torch.full((size,), fill, dtype=torch.int64)
        self.stashed = []

    def stash(self, indices, payload):
        self.stashed.append((indices.tolist(), payload.bonus_tokens.tolist()))
        self.output_tokens_buf[indices] = payload.bonus_tokens.to(torch.int64)


class _Scheduler:
    def __init__(self, batches, future_map):
        self.running_mbs = list(batches)
        self.running_batch = None
        self.future_map = future_map


def _fixture(stale=999):
    reqs = [_Req("a" * 8, 3, [11, 16, 16]), _Req("b" * 8, 5, [11, 16, 22])]
    batch = _Batch(reqs, input_ids=torch.tensor([7, 7]))
    fm = _FutureMap(size=8, fill=stale)
    return _Scheduler([batch], fm), batch, reqs, fm


class TestDecodeInputRelaySeed(unittest.TestCase):
    def test_stale_relay_is_what_the_gather_would_read(self):
        """THE CAN-FAIL: without the seed the gather reads a foreign token."""
        _, _, reqs, fm = _fixture()
        idx = torch.tensor([r.req_pool_idx for r in reqs])
        self.assertEqual(fm.output_tokens_buf[idx].tolist(), [999, 999])
        self.assertNotEqual(
            fm.output_tokens_buf[idx].tolist(), [r.output_ids[-1] for r in reqs]
        )

    def test_seed_writes_each_request_its_own_last_token(self):
        sched, _, reqs, fm = _fixture()
        n = reseed_decode_input_relay(sched)
        self.assertEqual(n, 2)
        idx = torch.tensor([r.req_pool_idx for r in reqs])
        self.assertEqual(
            fm.output_tokens_buf[idx].tolist(), [r.output_ids[-1] for r in reqs]
        )

    def test_seed_touches_no_other_row(self):
        """Pool-indexed: rows of requests that are not resident stay put."""
        sched, _, _, fm = _fixture()
        reseed_decode_input_relay(sched)
        untouched = [i for i in range(8) if i not in (3, 5)]
        self.assertEqual(
            fm.output_tokens_buf[torch.tensor(untouched)].tolist(),
            [999] * len(untouched),
        )

    def test_leftover_speculative_input_ids_is_cleared(self):
        """A draft-token ``input_ids`` names tokens the PP phase cannot
        interpret; the gather must be the only source."""
        sched, batch, _, _ = _fixture()
        reseed_decode_input_relay(sched)
        self.assertIsNone(batch.input_ids)

    def test_running_batch_is_covered_when_not_in_the_slots(self):
        sched, _, _, fm = _fixture()
        extra = _Req("c" * 8, 6, [11, 16, 24])
        sched.running_batch = _Batch([extra])
        reseed_decode_input_relay(sched)
        self.assertEqual(int(fm.output_tokens_buf[6]), 24)

    def test_running_batch_aliasing_a_slot_is_not_seeded_twice(self):
        sched, batch, _, fm = _fixture()
        sched.running_batch = batch
        n = reseed_decode_input_relay(sched)
        self.assertEqual(n, 2)
        self.assertEqual(len(fm.stashed), 1)

    def test_no_future_map_is_a_no_op_not_a_raise(self):
        """Ranks without a relay must not take the cutover down."""
        sched, _, _, _ = _fixture()
        sched.future_map = None
        self.assertEqual(reseed_decode_input_relay(sched), 0)

    def test_request_without_output_ids_is_skipped(self):
        """A request that has produced nothing has no truth to hand over,
        and inventing one would be worse than leaving the row alone."""
        fm = _FutureMap(size=8, fill=999)
        sched = _Scheduler([_Batch([_Req("d" * 8, 2, [])])], fm)
        self.assertEqual(reseed_decode_input_relay(sched), 0)
        self.assertEqual(int(fm.output_tokens_buf[2]), 999)


if __name__ == "__main__":
    unittest.main()
