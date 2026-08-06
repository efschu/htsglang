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
"""#594: the NCCL term must be able to reach "priced".

The defect these pin is not a wrong number, it is an ABSENT SUPPLY PATH.
``DemandInputs.nccl_buffer_mib_per_gpu`` and ``nccl_signature`` shipped
declared and never assigned by any caller, so on every TP>1 boot the term was
UNBOUNDED, the contract refused, and ``--enable-vram-ledger`` could not boot at
all. A test that only checked the term's arithmetic would have passed against
that defect, so these tests exercise the round trip: measure -> dump -> ingest
-> cache -> look up -> price.
"""

import json
import os
import tempfile
import unittest

from sglang.srt.mem_ledger.nccl_probe import (
    SIGNATURE_ENV,
    ingest_dumps,
    load_nccl_buffers,
    write_nccl_dump,
)
from sglang.srt.mem_ledger.nccl_transport import CommunicatorGroup, nccl_signature

UUID_A = "GPU-aaaaaaaa-0000-0000-0000-000000000001"
UUID_B = "GPU-bbbbbbbb-0000-0000-0000-000000000002"
FP = "a191a0712717"


def _dump(directory, uuid, sig, mib, exclusive=True, fp=FP):
    return write_nccl_dump(
        card_uuid=uuid,
        hw_fingerprint=fp,
        signature=sig,
        per_group_mib={"world": mib / 2.0, "tp": mib / 2.0},
        exclusive=exclusive,
        dump_dir=directory,
    )


class TestNcclSignature(unittest.TestCase):
    def test_only_groups_that_build_nccl_move_the_signature(self):
        """A group that allocates nothing may not invalidate a measurement."""
        builds = [CommunicatorGroup("world", 3), CommunicatorGroup("tp", 3)]
        plus_inert = builds + [CommunicatorGroup("dcp", 1)]
        self.assertEqual(nccl_signature(builds), nccl_signature(plus_inert))

    def test_rank_count_moves_the_signature(self):
        """Widening TP changes what libnccl allocates, so the key must move."""
        self.assertNotEqual(
            nccl_signature([CommunicatorGroup("tp", 2)]),
            nccl_signature([CommunicatorGroup("tp", 3)]),
        )

    def test_no_communicator_is_a_named_empty_set(self):
        self.assertEqual(nccl_signature([CommunicatorGroup("tp", 1)]), "no-nccl")

    def test_signature_is_order_independent(self):
        a = [CommunicatorGroup("world", 3), CommunicatorGroup("tp", 3)]
        b = [CommunicatorGroup("tp", 3), CommunicatorGroup("world", 3)]
        self.assertEqual(nccl_signature(a), nccl_signature(b))

    def test_generator_input_is_not_consumed_before_it_is_read(self):
        """The classify+zip pass must not exhaust a one-shot iterable."""
        gen = (g for g in [CommunicatorGroup("world", 3), CommunicatorGroup("tp", 3)])
        self.assertEqual(
            nccl_signature(gen),
            nccl_signature([CommunicatorGroup("world", 3), CommunicatorGroup("tp", 3)]),
        )


class TestRoundTrip(unittest.TestCase):
    def test_dump_ingest_load(self):
        sig = nccl_signature([CommunicatorGroup("world", 2), CommunicatorGroup("tp", 2)])
        with tempfile.TemporaryDirectory() as d:
            cache = os.path.join(d, "cache")
            _dump(d, UUID_A, sig, 210.0)
            _dump(d, UUID_B, sig, 190.0)
            path, notes = ingest_dumps(d, cache_dir=cache)
            self.assertIsNotNone(path, notes)
            per = load_nccl_buffers(FP, sig, cache_dir=cache)
            self.assertIsNotNone(per)
            self.assertAlmostEqual(per[UUID_A], 210.0)
            self.assertAlmostEqual(per[UUID_B], 190.0)

    def test_a_measured_zero_is_kept_and_is_not_a_miss(self):
        """A measured 0 is a value of the priced state, not an absence."""
        sig = nccl_signature([CommunicatorGroup("tp", 2)])
        with tempfile.TemporaryDirectory() as d:
            cache = os.path.join(d, "cache")
            _dump(d, UUID_A, sig, 0.0)
            path, notes = ingest_dumps(d, cache_dir=cache)
            self.assertIsNotNone(path, notes)
            per = load_nccl_buffers(FP, sig, cache_dir=cache)
            self.assertEqual(per, {UUID_A: 0.0})

    def test_wrong_signature_does_not_load(self):
        sig = nccl_signature([CommunicatorGroup("tp", 2)])
        other = nccl_signature([CommunicatorGroup("tp", 3)])
        with tempfile.TemporaryDirectory() as d:
            cache = os.path.join(d, "cache")
            _dump(d, UUID_A, sig, 210.0)
            ingest_dumps(d, cache_dir=cache)
            self.assertIsNone(load_nccl_buffers(FP, other, cache_dir=cache))

    def test_wrong_fingerprint_does_not_load(self):
        sig = nccl_signature([CommunicatorGroup("tp", 2)])
        with tempfile.TemporaryDirectory() as d:
            cache = os.path.join(d, "cache")
            _dump(d, UUID_A, sig, 210.0)
            ingest_dumps(d, cache_dir=cache)
            self.assertIsNone(load_nccl_buffers("deadbeef", sig, cache_dir=cache))


class TestIngestRefusals(unittest.TestCase):
    def test_unsigned_dump_is_refused(self):
        """No published signature means the number is valid for nothing."""
        with tempfile.TemporaryDirectory() as d:
            cache = os.path.join(d, "cache")
            _dump(d, UUID_A, "", 210.0)
            path, notes = ingest_dumps(d, cache_dir=cache)
            self.assertIsNone(path)
            self.assertTrue(any(SIGNATURE_ENV in n for n in notes), notes)

    def test_non_exclusive_card_is_refused_not_over_charged(self):
        sig = nccl_signature([CommunicatorGroup("tp", 2)])
        with tempfile.TemporaryDirectory() as d:
            cache = os.path.join(d, "cache")
            _dump(d, UUID_A, sig, 900.0, exclusive=False)
            path, notes = ingest_dumps(d, cache_dir=cache)
            self.assertIsNone(path)
            self.assertTrue(any("did not have to itself" in n for n in notes), notes)

    def test_dumps_from_two_launches_are_refused(self):
        with tempfile.TemporaryDirectory() as d:
            cache = os.path.join(d, "cache")
            _dump(d, UUID_A, nccl_signature([CommunicatorGroup("tp", 2)]), 210.0)
            _dump(d, UUID_B, nccl_signature([CommunicatorGroup("tp", 3)]), 190.0)
            path, notes = ingest_dumps(d, cache_dir=cache)
            self.assertIsNone(path)
            self.assertTrue(any("disagree" in n for n in notes), notes)

    def test_empty_dir_reports_rather_than_crashes(self):
        with tempfile.TemporaryDirectory() as d:
            path, notes = ingest_dumps(d)
            self.assertIsNone(path)
            self.assertTrue(notes)


class TestCallSiteIsWired(unittest.TestCase):
    """The #605 lesson: a supply path present in a module nobody calls is the
    same defect as no supply path. These reach for the real call sites."""

    def test_parallel_state_brackets_the_constructor(self):
        import inspect

        from sglang.srt.distributed import parallel_state

        src = inspect.getsource(parallel_state)
        self.assertIn("measure_communicator_init", src)
        # The bracket must WRAP the constructor, not merely be imported near it.
        idx_with = src.find("with measure_communicator_init")
        self.assertGreater(idx_with, 0, "the probe is imported but never entered")
        self.assertIn("PyNcclCommunicator(", src[idx_with : idx_with + 400])

    def test_demand_inputs_assigns_both_nccl_fields(self):
        import inspect

        from sglang.srt.mem_ledger.engine import DemandInputs

        src = inspect.getsource(DemandInputs.from_server_args)
        self.assertIn("nccl_buffer_mib_per_gpu=", src)
        self.assertIn("nccl_signature=", src)

    def test_uuid_to_gpu_id_mapping_is_not_positional(self):
        """cuda:0 is not NVML 0 on this rig; the map must go through uuids."""
        import inspect

        from sglang.srt.mem_ledger.engine import DemandInputs

        src = inspect.getsource(DemandInputs.from_server_args)
        self.assertIn("card_uuid_by_gpu", src)


class TestDumpShape(unittest.TestCase):
    def test_groups_sum_into_the_card_total(self):
        with tempfile.TemporaryDirectory() as d:
            path = write_nccl_dump(
                card_uuid=UUID_A,
                hw_fingerprint=FP,
                signature="sig",
                per_group_mib={"world": 100.0, "tp": 55.5},
                dump_dir=d,
            )
            with open(path) as f:
                payload = json.load(f)
            self.assertAlmostEqual(payload["total_mib"], 155.5)

    def test_unarmed_writes_nothing(self):
        os.environ.pop("SGLANG_NCCL_BUFFER_DUMP", None)
        self.assertIsNone(
            write_nccl_dump(
                card_uuid=UUID_A,
                hw_fingerprint=FP,
                signature="sig",
                per_group_mib={"tp": 1.0},
            )
        )


if __name__ == "__main__":
    unittest.main()
