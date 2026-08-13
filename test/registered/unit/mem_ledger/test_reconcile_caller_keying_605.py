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
"""Four defects found by running reconcile on live acceptance boots (#605).

RED-FIRST. All four were reproduced against
``/spinning/evidence-631/acceptance-656/flight/`` boot 1917721-1786622304
before any of them was fixed, and the numbers in these fixtures are that
boot's.

1. THE CALLER PASSED PID-KEYED MARKS INTO A RANK-KEYED LOOKUP and got a
   SILENT empty result -- "The ledger names no card whose rank left marks" --
   which reads like a statement about the data and was a statement about the
   code.
2. MEASURED DEMAND SUBTRACTED THE MODELLED KV POOL from a measured footprint
   and went NEGATIVE (-2023 and -180 MiB on this boot).
3. FIELD-STYLE TERMS took the first matching mark, i.e. always the target
   runner, while the draft runner carries the higher level.
4. NCCL BUFFERS ARE CORRECTLY ZERO under barlink (``not_applicable: true``)
   and were reported as UNMEASURED, which is a different claim.
"""

import unittest

from sglang.srt.mem_ledger.engine import (
    TERM_HARDWARE_RESIDUAL,
    TERM_NCCL_BUFFERS,
    TERM_NVML_CARVE_OUT,
)
from sglang.srt.mem_ledger.reconcile import (
    ReconcileRefusal,
    marks_by_rank_from_pids,
    reconcile,
    reconcile_card,
)

MIB = 1 << 20

UUID_5090 = "GPU-31d7ef41-f574-4d0e-21ad-e773fd938f6d"
UUID_5C64 = "GPU-5c648f96-be1d-42d5-0221-34d11ab137f7"
UUID_62DB = "GPU-62dbbae1-e859-9ccc-f9c2-d9f2443a84f4"


def _m(phase, *, pid, uuid, rank, total, draft=None, mono=0.0, **fields):
    mark = {
        "phase": phase,
        "pid": pid,
        "rank": rank,
        "card_uuid": uuid,
        "boot_id": "1917721-1786622304",
        "monotonic": float(mono),
        "nvml_total_bytes": total * MIB,
        "allocated_bytes": 0,
        "reserved_bytes": 0,
    }
    for k, v in fields.items():
        mark[k] = v * MIB
    if draft is not None:
        mark["extra"] = {"draft_worker": draft}
    return mark


def _process(pid, uuid, rank, total, self_bc, carve, arena_bc, nt_target, nt_draft):
    """One process's marks, shaped like the live boot's."""
    return [
        _m("process_start", pid=pid, uuid=uuid, rank=0, total=total, mono=0),
        _m(
            "weights_loaded",
            pid=pid,
            uuid=uuid,
            rank=0,
            total=total,
            draft=False,
            mono=1,
            non_torch_bytes=nt_target,
            allocated_bytes=9000,
        ),
        _m(
            "weights_loaded",
            pid=pid,
            uuid=uuid,
            rank=rank,
            total=total,
            draft=True,
            mono=2,
            non_torch_bytes=nt_draft,
            allocated_bytes=12000,
        ),
        _m(
            "boot_complete",
            pid=pid,
            uuid=uuid,
            rank=0,
            total=total,
            mono=3,
            nvml_self_bytes=self_bc,
            nvml_carve_out_bytes=carve,
            kv_arena_backed_bytes=arena_bc,
        ),
    ]


#: pid -> marks, exactly what flight_recorder.read_marks returns.
LIVE_BY_PID = {
    1918126: _process(1918126, UUID_5090, 0, 32607, 27386, 518, 22164, 886, 896),
    1918127: _process(1918127, UUID_5C64, 1, 20480, 17978, 425, 14328, 480, 496),
    1918128: _process(1918128, UUID_62DB, 2, 20480, 17640, 425, 14146, 480, 496),
}


def _card(gpu_id, name, total, ranks, kv_pool, uuid=None):
    card = {
        "gpu_id": gpu_id,
        "card": f"GPU {gpu_id} ({name}, NVML total {total} MiB)",
        "total_mib": total,
        "ranks": ranks,
        "kv_pool_mib": kv_pool,
        "demand_mib": 1656,
        "unbounded": [],
        "terms": [
            {"name": TERM_HARDWARE_RESIDUAL, "mib": 664, "provenance": "calibrated"},
            {
                "name": TERM_NCCL_BUFFERS,
                "mib": 0,
                "provenance": "modeled",
                "not_applicable": True,
            },
            {"name": TERM_NVML_CARVE_OUT, "mib": 518, "provenance": "reported"},
        ],
    }
    if uuid:
        card["uuid"] = uuid
    return card


LIVE_LEDGER = {
    "boot_id": "1917721-1786622304",
    "cards": [
        _card(0, "NVIDIA GeForce RTX 5090", 32607, [0], 29927),
        _card(1, "NVIDIA GeForce RTX 3080", 20480, [1], 18245),
        _card(2, "NVIDIA GeForce RTX 3080", 20480, [2], 18245),
    ],
}


# ---------------------------------------------------------------------------
# 1. The silent skip
# ---------------------------------------------------------------------------


class TestPidKeyedMarksReachTheirCards(unittest.TestCase):
    def test_the_silent_skip_signature_is_gone(self):
        """reconcile() on pid-keyed marks used to return [] and the caller
        printed 'The ledger names no card whose rank left marks'."""
        results = reconcile(LIVE_LEDGER, LIVE_BY_PID)
        self.assertEqual(len(results), 3, "every card must find its process")

    def test_each_card_gets_the_process_that_actually_ran_on_it(self):
        results = {r.gpu_id: r for r in reconcile(LIVE_LEDGER, LIVE_BY_PID)}
        # The 5090 is the only card whose carve-out is 518 MiB.
        self.assertEqual(results[0].rank, 0)
        self.assertEqual(results[1].rank, 1)
        self.assertEqual(results[2].rank, 2)

    def test_the_mapper_recovers_rank_from_the_marks_not_from_the_pid(self):
        """The per-mark rank field is 0 on the process-level phases of EVERY
        process; only the runner-tagged marks carry the true rank."""
        mapped = marks_by_rank_from_pids(LIVE_BY_PID, LIVE_LEDGER)
        self.assertEqual(sorted(mapped), [0, 1, 2])
        self.assertEqual(mapped[1][0]["pid"], 1918127)
        self.assertEqual(mapped[2][0]["pid"], 1918128)

    def test_already_rank_keyed_marks_pass_through_unharmed(self):
        """A caller that did the re-keying itself must not be punished for it."""
        rank_keyed = {
            0: LIVE_BY_PID[1918126],
            1: LIVE_BY_PID[1918127],
            2: LIVE_BY_PID[1918128],
        }
        mapped = marks_by_rank_from_pids(rank_keyed, LIVE_LEDGER)
        self.assertEqual(sorted(mapped), [0, 1, 2])
        self.assertEqual(mapped[2][0]["pid"], 1918128)

    def test_a_card_with_no_process_REFUSES_rather_than_disappearing(self):
        partial = {1918126: LIVE_BY_PID[1918126]}
        with self.assertRaises(ReconcileRefusal) as ctx:
            marks_by_rank_from_pids(partial, LIVE_LEDGER)
        self.assertIn("GPU 1", str(ctx.exception))

    def test_two_processes_claiming_one_rank_REFUSES(self):
        clash = {
            1918127: LIVE_BY_PID[1918127],
            1918128: [
                dict(m, rank=1) if m.get("rank") == 2 else m
                for m in LIVE_BY_PID[1918128]
            ],
        }
        with self.assertRaises(ReconcileRefusal):
            marks_by_rank_from_pids(clash, LIVE_LEDGER)


# ---------------------------------------------------------------------------
# 2. Measured demand
# ---------------------------------------------------------------------------


class TestMeasuredDemandUsesTheMeasuredPool(unittest.TestCase):
    def test_demand_is_positive_on_the_live_boot(self):
        results = {r.gpu_id: r for r in reconcile(LIVE_LEDGER, LIVE_BY_PID)}
        # With the MODELLED pool these were -2023 and -180 MiB.
        self.assertEqual(results[0].measured_demand_mib, 27386 + 518 - 22164)
        self.assertEqual(results[2].measured_demand_mib, 17640 + 425 - 14146)
        for r in results.values():
            self.assertGreater(r.measured_demand_mib, 0)

    def test_the_modelled_pool_is_NEVER_the_fallback(self):
        """Falling back to the budget is what produced the negative demand;
        an unmeasurable pool must refuse instead."""
        stripped = {
            pid: [
                {k: v for k, v in m.items() if k != "kv_arena_backed_bytes"}
                for m in marks
            ]
            for pid, marks in LIVE_BY_PID.items()
        }
        results = reconcile(LIVE_LEDGER, stripped)
        for r in results:
            self.assertIsNone(r.measured_demand_mib)
            self.assertIn("refus", r.render().lower())
            # The 29927 MiB budget must not appear as if it had been used.
            self.assertNotIn("-2023", r.render())


# ---------------------------------------------------------------------------
# 3. Field-style terms and the runner partition
# ---------------------------------------------------------------------------


class TestFieldTermsArePartitionedPerRunner(unittest.TestCase):
    def test_the_draft_runners_higher_level_is_the_one_taken(self):
        results = {r.gpu_id: r for r in reconcile(LIVE_LEDGER, LIVE_BY_PID)}
        row = next(
            c for c in results[0].comparisons if c.term == TERM_HARDWARE_RESIDUAL
        )
        self.assertEqual(row.measured_mib, 896)  # draft runner
        self.assertNotEqual(row.measured_mib, 886)  # target runner, first match

    def test_the_falsifier_target_zero_draft_nonzero(self):
        """The exact shape the transient defect had: a target-runner zero
        hiding a draft-runner peak."""
        marks = _process(999, UUID_5090, 0, 32607, 27386, 518, 22164, 0, 777)
        ledger_card = _card(0, "NVIDIA GeForce RTX 5090", 32607, [0], 29927)
        result = reconcile_card(ledger_card, marks)
        row = next(c for c in result.comparisons if c.term == TERM_HARDWARE_RESIDUAL)
        self.assertEqual(row.measured_mib, 777)
        self.assertNotEqual(row.measured_mib, 0)

    def test_an_untagged_single_valued_field_is_unchanged(self):
        """boot_complete is process-level; partitioning must not move it."""
        results = {r.gpu_id: r for r in reconcile(LIVE_LEDGER, LIVE_BY_PID)}
        row = next(c for c in results[0].comparisons if c.term == TERM_NVML_CARVE_OUT)
        self.assertEqual(row.measured_mib, 518)
        self.assertEqual(row.error_mib, 0)


# ---------------------------------------------------------------------------
# 4. not_applicable is not UNMEASURED
# ---------------------------------------------------------------------------


class TestNotApplicableIsItsOwnVerdict(unittest.TestCase):
    def test_nccl_under_barlink_reads_not_applicable(self):
        results = reconcile(LIVE_LEDGER, LIVE_BY_PID)
        row = next(c for c in results[0].comparisons if c.term == TERM_NCCL_BUFFERS)
        self.assertIn("not applicable", row.note.lower())
        self.assertNotIn("UNMEASURED", row.row())

    def test_it_is_not_counted_among_the_unmeasured_terms(self):
        results = reconcile(LIVE_LEDGER, LIVE_BY_PID)
        text = results[0].render()
        unmeasured_line = [
            ln for ln in text.split("\n") if "UNMEASURED terms are not evidence" in ln
        ]
        for line in unmeasured_line:
            self.assertNotIn(TERM_NCCL_BUFFERS, line)


if __name__ == "__main__":
    unittest.main()
